"""Read-only cross-session tail continuation for first-turn private DM context."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Literal, Mapping

from ..infra.chat_scope import PRIVATE_CHAT_TYPES, ChatScope
from ..infra.host_bridge import is_delegated_child_turn, is_test_session_id
from ..infra.textutil import _owner_key
from ..storage.sqlite_facade import connect_readonly

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 24 * 60 * 60
_MAX_CANDIDATE_SESSIONS = 32
_MAX_TAIL_QUERY_ROWS = 64
_MAX_TURNS = 2
_MAX_MESSAGES = 4
_MAX_MESSAGE_CHARS = 220
_MAX_TOTAL_CHARS = 650

_PRIVATE_CHAT_TYPES = PRIVATE_CHAT_TYPES | frozenset({"c2c"})
_NON_PRIVATE_SOURCES = frozenset({
    "cron",
    "subagent",
    "cli",
    "tui",
    "desktop",
    "local",
    "tool",
    "unknown",
})
_IGNORED_DISPLAY_KINDS = frozenset({
    "hidden",
    "failed_turn",
    "heartbeat",
    "system_notice",
})
_SILENT_REPLIES = frozenset({"[SILENT]", "SILENT", "HEARTBEAT_OK", "[HEARTBEAT_OK]"})

_MEMORY_CONTEXT_RE = re.compile(r"<memory-context>[\s\S]*?</memory-context>", re.IGNORECASE)
_TRANSIENT_BLOCK_RE = re.compile(
    r"(?m)^\[(?:我眼中的世界|我内心的感受|我刚才在别处说过)\][\s\S]*$"
)
_TIME_HEADER_RE = re.compile(r"^\s*\[时间：[^\]]+\]\s*")
_TEST_PREFIX_RE = re.compile(r"^\s*T:\s*")


@dataclass(frozen=True)
class PreviousSessionQuery:
    """Typed parameters for reading the previous ended private session tail."""

    home: Path
    current_session_id: str
    current_owner: str
    read_owners: tuple[str, ...]
    scope: ChatScope
    is_maintainer: bool
    partner_name: str
    now_ts: float | None = None


@dataclass(frozen=True)
class PreviousTurnMessage:
    """One cleaned user/assistant utterance from a previous ended session."""

    role: Literal["user", "assistant"]
    text: str
    timestamp: float | None


def _has_tool_calls(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped or stripped in {"[]", "null"}:
            return False
        return True
    return bool(raw)


def _extract_plain_text(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        return ""
    stripped = raw.strip()
    if not stripped:
        return ""
    if stripped.startswith("["):
        try:
            decoded = json.loads(stripped)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, list):
            parts = [
                str(item.get("text") or "").strip()
                for item in decoded
                if isinstance(item, Mapping)
                and str(item.get("type") or "").strip() in {"text", "input_text"}
            ]
            return "\n".join(part for part in parts if part)
    return raw


def _clean_message_text(raw: Any, *, role: str) -> str:
    text = _extract_plain_text(raw)
    if not text:
        return ""
    text = _MEMORY_CONTEXT_RE.sub("", text)
    text = _TRANSIENT_BLOCK_RE.sub("", text)
    text = _TIME_HEADER_RE.sub("", text)
    if role == "user" and _TEST_PREFIX_RE.match(text):
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    if role == "assistant" and cleaned.upper() in _SILENT_REPLIES:
        return ""
    return cleaned


def _is_test_prefixed_user_message(raw: Any) -> bool:
    text = _extract_plain_text(raw)
    if not text:
        return False
    text = _TIME_HEADER_RE.sub("", text)
    return bool(_TEST_PREFIX_RE.match(text))


def _parse_origin_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_new_session_first_turn(conn: sqlite3.Connection, session_id: str) -> bool:
    """Return True iff *session_id* has no completed assistant reply and no prior history turns."""
    parent_row = conn.execute(
        """
        SELECT p.end_reason
          FROM sessions s
          LEFT JOIN sessions p ON p.id = s.parent_session_id
         WHERE s.id = ?
         LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if parent_row is not None and str(parent_row["end_reason"] or "").strip().lower() == "compression":
        return False

    rows = conn.execute(
        """
        SELECT role, content, tool_call_id, tool_calls, _compressed_summary, display_kind
          FROM messages
         WHERE session_id = ?
           AND COALESCE(active, 1) = 1
         ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()
    user_turns = 0
    for row in rows:
        if row["_compressed_summary"]:
            return False
        display_kind = str(row["display_kind"] or "").strip().lower()
        if display_kind in _IGNORED_DISPLAY_KINDS:
            continue
        role = str(row["role"] or "").strip().lower()
        if role == "assistant":
            has_text = bool(_extract_plain_text(row["content"]).strip())
            if has_text or _has_tool_calls(row["tool_calls"]):
                return False
        elif role == "user":
            user_turns += 1
            if user_turns > 1:
                return False
    return True


def _load_registered_test_session_ids(home: Path) -> frozenset[str] | None:
    """Return registered test-session IDs, or ``None`` (fail closed) on read/schema error."""
    registered: set[str] = set()
    for registry_path in (
        home / "state" / "test_isolated_sessions.json",
        home / "sanae-mem" / "test_sessions.json",
    ):
        try:
            raw_text = registry_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            if registry_path.is_symlink():
                logger.warning(
                    "sanae_mem: test session registry %s is a broken symlink; failing closed",
                    registry_path,
                )
                return None
            continue
        except (OSError, ValueError) as exc:
            logger.warning(
                "sanae_mem: failed to read test session registry %s (%s); failing closed",
                registry_path,
                type(exc).__name__,
            )
            return None

        try:
            payload = json.loads(raw_text)
        except ValueError as exc:
            logger.warning(
                "sanae_mem: failed to parse test session registry %s (%s); failing closed",
                registry_path,
                type(exc).__name__,
            )
            return None

        if not isinstance(payload, dict):
            logger.warning(
                "sanae_mem: test session registry %s top-level must be dict, got %s; failing closed",
                registry_path,
                type(payload).__name__,
            )
            return None

        if "sessions" in payload:
            sessions = payload["sessions"]
            if isinstance(sessions, list):
                for item in sessions:
                    sid = str(item or "").strip() if item is not None else ""
                    if sid:
                        registered.add(sid)
            elif isinstance(sessions, dict):
                for key in sessions:
                    sid = str(key or "").strip() if key is not None else ""
                    if sid:
                        registered.add(sid)
            else:
                logger.warning(
                    "sanae_mem: test session registry %s 'sessions' must be list or dict, got %s; failing closed",
                    registry_path,
                    type(sessions).__name__,
                )
                return None
    return frozenset(registered)


def _is_test_session_candidate(home: Path, sid: str, registered_test_ids: frozenset[str]) -> bool:
    """True when *sid* matches any local filesystem registry, test-log, or host test-isolation check."""
    if not sid or sid in registered_test_ids:
        return True
    test_log_path = home / "test-log" / f"{sid}.jsonl"
    try:
        if test_log_path.exists():
            return True
    except OSError as exc:
        logger.warning(
            "sanae_mem: failed to check test-log path %s (%s); skipping candidate fail-closed",
            test_log_path,
            type(exc).__name__,
        )
        return True
    return is_test_session_id(sid)


def _session_row_matches_owner(
    row: sqlite3.Row,
    *,
    home: Path,
    registered_test_ids: frozenset[str],
    scope: ChatScope,
    allowed_owners: frozenset[str],
) -> bool:
    sid = str(row["id"] or "").strip()
    if _is_test_session_candidate(home, sid, registered_test_ids):
        return False
    title = str(row["title"] or "").lstrip()
    if _TEST_PREFIX_RE.match(title):
        return False
    source = str(row["source"] or "").strip().lower()
    if not source or source in _NON_PRIVATE_SOURCES:
        return False
    chat_type = str(row["chat_type"] or "").strip().lower()
    if chat_type not in _PRIVATE_CHAT_TYPES:
        return False
    user_id = str(row["user_id"] or "").strip()
    chat_id = str(row["chat_id"] or "").strip()
    if not user_id or not chat_id:
        return False
    session_key = str(row["session_key"] or "").strip()
    lower_key = session_key.lower()
    if lower_key.startswith("cron:") or any(
        marker in lower_key for marker in (":group:", ":channel:", ":thread:")
    ):
        return False

    origin = _parse_origin_json(row["origin_json"])
    if origin.get("is_bot") is True:
        return False
    origin_chat_type = str(origin.get("chat_type") or "").strip().lower()
    if origin_chat_type and origin_chat_type not in _PRIVATE_CHAT_TYPES:
        return False
    origin_plat = str(origin.get("platform") or "").strip().lower()
    if origin_plat and origin_plat != source:
        return False
    row_scope_id = str(origin.get("scope_id") or "").strip()

    actor_candidates = {value for value in (scope.actor, scope.actor_alt) if value}
    if source == scope.platform and user_id in actor_candidates:
        if chat_id != scope.chat_id:
            return False
        if scope.platform == "slack" and (not scope.scope_id or row_scope_id != scope.scope_id):
            return False
        if scope.scope_id and row_scope_id and row_scope_id != scope.scope_id:
            return False
        return True

    if source == "slack":
        if not row_scope_id:
            return False
        row_owner = _owner_key("slack", user_id=user_id, scope_id=row_scope_id)
        return bool(row_owner and row_owner in allowed_owners)

    candidate_owners = {_owner_key(source, user_id=user_id)}
    if row_scope_id:
        candidate_owners.add(_owner_key(source, user_id=user_id, scope_id=row_scope_id))
    candidate_owners.discard("")
    return bool(candidate_owners & allowed_owners)


def _read_session_tail_messages(
    conn: sqlite3.Connection,
    session_id: str,
) -> tuple[PreviousTurnMessage, ...]:
    rows = conn.execute(
        """
        SELECT role, content, tool_call_id, tool_calls, _compressed_summary, display_kind, timestamp
          FROM messages
         WHERE session_id = ?
           AND COALESCE(active, 1) = 1
         ORDER BY id DESC
         LIMIT ?
        """,
        (session_id, _MAX_TAIL_QUERY_ROWS),
    ).fetchall()

    selected_rev: list[PreviousTurnMessage] = []
    pending_assistant_rev: list[PreviousTurnMessage] = []
    user_count = 0
    saw_valid_user = False
    saw_test_prefixed_user = False
    for row in rows:
        if row["tool_call_id"] or _has_tool_calls(row["tool_calls"]) or row["_compressed_summary"]:
            continue
        display_kind = str(row["display_kind"] or "").strip().lower()
        if display_kind in _IGNORED_DISPLAY_KINDS:
            continue
        role = str(row["role"] or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        if role == "user" and _is_test_prefixed_user_message(row["content"]):
            saw_test_prefixed_user = True
            # Drop any trailing assistant reply that responded to this T: probe.
            pending_assistant_rev.clear()
            continue
        cleaned = _clean_message_text(row["content"], role=role)
        if not cleaned:
            continue
        ts_raw = row["timestamp"]
        ts = float(ts_raw) if isinstance(ts_raw, (int, float)) and not isinstance(ts_raw, bool) else None
        msg = PreviousTurnMessage(
            role="user" if role == "user" else "assistant",
            text=cleaned,
            timestamp=ts,
        )
        if role == "assistant":
            if user_count >= _MAX_TURNS:
                break
            pending_assistant_rev.append(msg)
            continue
        saw_valid_user = True
        if user_count >= _MAX_TURNS:
            break
        for pending in pending_assistant_rev:
            if len(selected_rev) >= _MAX_MESSAGES:
                break
            selected_rev.append(pending)
        pending_assistant_rev.clear()
        if len(selected_rev) >= _MAX_MESSAGES:
            break
        selected_rev.append(msg)
        user_count += 1
    if saw_test_prefixed_user and not saw_valid_user:
        return ()
    for pending in pending_assistant_rev:
        if len(selected_rev) >= _MAX_MESSAGES:
            break
        selected_rev.append(pending)
    selected_rev.reverse()
    return tuple(selected_rev)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _format_tail_block(
    messages: tuple[PreviousTurnMessage, ...],
    *,
    user_label: str,
) -> str:
    if not messages:
        return ""

    def _render(items: tuple[PreviousTurnMessage, ...]) -> list[str]:
        return [
            f"{user_label if msg.role == 'user' else '我'}：{_truncate_text(msg.text, _MAX_MESSAGE_CHARS)}"
            for msg in items
        ]

    lines = _render(messages)
    body = "\n".join(lines)
    if len(body) > _MAX_TOTAL_CHARS:
        last_user_idx = max(
            (idx for idx, msg in enumerate(messages) if msg.role == "user"),
            default=0,
        )
        trimmed = messages[last_user_idx:]
        lines = _render(trimmed)
        body = "\n".join(lines)
        if len(body) > _MAX_TOTAL_CHARS:
            body = _truncate_text(body, _MAX_TOTAL_CHARS)
    return f"〔上回聊到的末尾〕\n{body}"


def _build_owner_sql_filter(
    *,
    scope: ChatScope,
    allowed_owners: frozenset[str],
) -> tuple[str, list[Any], frozenset[str]]:
    """Build parameterized SQL owner filter clauses and return unresolved hash owners."""
    clauses: list[str] = []
    params: list[Any] = []

    actors = sorted({value for value in (scope.actor, scope.actor_alt) if value})
    if scope.platform and scope.chat_id and actors:
        placeholders = ", ".join("?" for _ in actors)
        clauses.append(
            f"(LOWER(COALESCE(source, '')) = ? AND user_id IN ({placeholders}) AND chat_id = ?)"
        )
        params.extend([scope.platform, *actors, scope.chat_id])

    scope_owners: set[str] = {
        _owner_key(
            scope.platform,
            user_id=scope.actor,
            user_id_alt=scope.actor_alt,
            scope_id=scope.scope_id if scope.platform == "slack" else "",
        )
    }
    for actor in actors:
        scope_owners.add(_owner_key(scope.platform, user_id=actor))
        if scope.scope_id:
            scope_owners.add(_owner_key(scope.platform, user_id=actor, scope_id=scope.scope_id))
    scope_owners.discard("")

    extra_owners = frozenset(
        owner for owner in allowed_owners if owner and owner not in scope_owners
    )
    unresolved_hashes: set[str] = set()
    explicit_pairs: dict[str, set[str]] = {}
    for token in sorted(extra_owners):
        if ":" in token:
            parts = [part.strip() for part in token.split(":") if part.strip()]
            if len(parts) >= 2:
                plat = parts[0].lower()
                uid_candidates = {parts[1]}
                if len(parts) >= 3:
                    uid_candidates.add(parts[2])
                explicit_pairs.setdefault(plat, set()).update(uid_candidates)
                continue
        unresolved_hashes.add(token)

    for plat, uids in sorted(explicit_pairs.items()):
        sorted_uids = sorted(uids)
        placeholders = ", ".join("?" for _ in sorted_uids)
        clauses.append(f"(LOWER(COALESCE(source, '')) = ? AND user_id IN ({placeholders}))")
        params.extend([plat, *sorted_uids])

    if unresolved_hashes:
        clauses.append("_sanae_matches_extra_owner(source, user_id, origin_json) = 1")

    return " OR ".join(clauses), params, frozenset(unresolved_hashes)


def _register_extra_owner_sql_fn(
    conn: sqlite3.Connection,
    extra_hash_owners: frozenset[str],
) -> None:
    """Register a deterministic read-only SQLite UDF for hashed cross-platform read_owners."""
    if not extra_hash_owners:
        return

    def _matches_extra_owner(raw_source: Any, raw_user_id: Any, raw_origin_json: Any) -> int:
        source = str(raw_source or "").strip().lower()
        user_id = str(raw_user_id or "").strip()
        if not source or not user_id or source in _NON_PRIVATE_SOURCES:
            return 0
        origin = _parse_origin_json(raw_origin_json)
        row_scope_id = str(origin.get("scope_id") or "").strip()
        if source == "slack":
            if not row_scope_id:
                return 0
            row_owner = _owner_key("slack", user_id=user_id, scope_id=row_scope_id)
            return 1 if row_owner and row_owner in extra_hash_owners else 0
        candidates = {_owner_key(source, user_id=user_id)}
        if row_scope_id:
            candidates.add(_owner_key(source, user_id=user_id, scope_id=row_scope_id))
        candidates.discard("")
        return 1 if (candidates & extra_hash_owners) else 0

    conn.create_function("_sanae_matches_extra_owner", 3, _matches_extra_owner, deterministic=True)


def _register_registered_test_session_sql_fn(
    conn: sqlite3.Connection,
    registered_test_ids: frozenset[str],
) -> None:
    """Register a deterministic read-only SQLite UDF to exclude registered test session IDs."""
    if not registered_test_ids:
        return

    def _is_registered_test_session(raw_id: Any) -> int:
        sid = str(raw_id or "").strip()
        return 1 if (not sid or sid in registered_test_ids) else 0

    conn.create_function(
        "_sanae_is_registered_test_session",
        1,
        _is_registered_test_session,
        deterministic=True,
    )


def read_previous_session_tail(query: PreviousSessionQuery) -> str:
    """Read the last 1-2 valid turns of the same owner's previous ended private DM within 24h."""
    if is_delegated_child_turn():
        return ""
    scope = query.scope
    if scope.visibility != "private" or not scope.platform or not scope.chat_id or not (scope.actor or scope.actor_alt):
        return ""
    if not query.current_session_id or not query.current_owner:
        return ""

    home = Path(query.home).expanduser()
    registered_test_ids = _load_registered_test_session_ids(home)
    if registered_test_ids is None:
        return ""

    state_db = home / "state.db"
    if not state_db.is_file():
        logger.debug("sanae_mem: state.db unavailable for previous session lookup (%s)", state_db)
        return ""

    now = float(query.now_ts) if query.now_ts is not None else time.time()
    cutoff = now - _WINDOW_SECONDS
    allowed_owners = frozenset(k for k in (query.current_owner, *query.read_owners) if k)
    owner_where_sql, owner_params, extra_hash_owners = _build_owner_sql_filter(
        scope=scope,
        allowed_owners=allowed_owners,
    )
    if not owner_where_sql:
        return ""

    try:
        with closing(connect_readonly(state_db)) as conn:
            if not _is_new_session_first_turn(conn, query.current_session_id):
                return ""
            _register_extra_owner_sql_fn(conn, extra_hash_owners)
            _register_registered_test_session_sql_fn(conn, registered_test_ids)
            registered_test_clause = (
                "AND _sanae_is_registered_test_session(id) = 0"
                if registered_test_ids
                else ""
            )
            candidates = conn.execute(
                f"""
                SELECT id, source, user_id, chat_id, chat_type, session_key, origin_json,
                       title, started_at, ended_at, end_reason, last_activity_at
                  FROM sessions
                 WHERE id != ?
                   AND id NOT LIKE 'T:%'
                   AND id NOT LIKE 'test\\_%' ESCAPE '\\'
                   AND id NOT LIKE '%:test:%'
                   AND LTRIM(COALESCE(title, '')) NOT LIKE 'T:%'
                   AND COALESCE(user_id, '') != ''
                   AND COALESCE(chat_id, '') != ''
                   AND ended_at IS NOT NULL
                   AND COALESCE(hidden, 0) = 0
                   AND COALESCE(archived, 0) = 0
                   AND COALESCE(end_reason, '') != 'compression'
                   AND LOWER(COALESCE(source, '')) NOT IN (
                       'cron', 'subagent', 'cli', 'tui', 'desktop', 'local', 'tool', 'unknown'
                   )
                   AND LOWER(COALESCE(chat_type, '')) IN ('dm', 'private', 'direct', 'c2c')
                   AND LOWER(COALESCE(session_key, '')) NOT LIKE 'cron:%'
                   AND LOWER(COALESCE(session_key, '')) NOT LIKE '%:group:%'
                   AND LOWER(COALESCE(session_key, '')) NOT LIKE '%:channel:%'
                   AND LOWER(COALESCE(session_key, '')) NOT LIKE '%:thread:%'
                   AND COALESCE(NULLIF(last_activity_at, 0), ended_at) >= ?
                   AND ({owner_where_sql})
                   {registered_test_clause}
                 ORDER BY COALESCE(ended_at, 0) DESC, COALESCE(last_activity_at, 0) DESC, started_at DESC
                 LIMIT ?
                """,
                (query.current_session_id, cutoff, *owner_params, _MAX_CANDIDATE_SESSIONS),
            ).fetchall()

            for row in candidates:
                if not _session_row_matches_owner(
                    row,
                    home=home,
                    registered_test_ids=registered_test_ids,
                    scope=scope,
                    allowed_owners=allowed_owners,
                ):
                    continue
                tail_messages = _read_session_tail_messages(conn, str(row["id"]))
                if not tail_messages:
                    continue
                user_label = query.partner_name if (query.is_maintainer and query.partner_name) else "对方"
                return _format_tail_block(tail_messages, user_label=user_label)
    except (sqlite3.Error, OSError) as exc:
        logger.warning(
            "sanae_mem: previous session tail read failed (%s)",
            type(exc).__name__,
        )
        return ""
    return ""
