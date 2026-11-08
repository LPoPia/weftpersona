from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

try:
    from .source_context import in_public_interaction
    from .textutil import _owner_key
except ImportError:
    from source_context import in_public_interaction
    from textutil import _owner_key


logger = logging.getLogger(__name__)
HEADER = "[我刚才在别处说过]"
_SEPARATOR = "｜"
_CONTINUATION = "> "


def _previous_reply_time(home: Path, session_id: str, platform: str = "") -> float | None:
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND
    from hermes_state_registry import acquire, release_or_close

    def _latest_in_session(db, sid: str) -> float | None:
        page_size, offset = 16, 0
        while page := db.get_messages(sid, latest=True, limit=page_size, offset=offset):
            for msg in reversed(page):
                metadata = msg.get("display_metadata") or {}
                if (msg.get("role") == "assistant" and msg.get("content") and not msg.get("tool_calls")
                        and msg.get("display_kind") not in {"hidden", FAILED_TURN_DISPLAY_KIND}
                        and not msg.get("_compressed_summary")
                        and metadata.get("qq_delivery") in (None, "COMPLETED")
                        and type(msg.get("timestamp")) in (int, float)):
                    return msg["timestamp"]
            if len(page) < page_size:
                break
            offset += len(page)
        return None

    path = home / "state.db"
    if not path.is_file():
        return None
    db = acquire(path)
    try:
        ts = _latest_in_session(db, session_id)
        if ts is not None:
            return ts
        conn = getattr(db, "_conn", None)
        if platform and conn is not None:
            try:
                rows = conn.execute(
                    "SELECT id FROM sessions WHERE source = ? AND id != ? ORDER BY updated_at DESC LIMIT 3",
                    (platform, session_id),
                ).fetchall()
            except Exception:
                rows = []
            for row in rows:
                prev_sid = str(row[0] if not isinstance(row, dict) else row.get("id") or "")
                if prev_sid:
                    ts = _latest_in_session(db, prev_sid)
                    if ts is not None:
                        return ts
    finally:
        release_or_close(db)
    return None


def _private_receipt_route(owner_key: str, session_id: str, home=None):
    from agent.delegation_context import is_delegated_child_context
    from gateway.session_context import (
        _SESSION_CHAT_ID, _SESSION_CHAT_TYPE, _SESSION_ID, _SESSION_PLATFORM, _SESSION_THREAD_ID,
        _SESSION_USER_ID, _SESSION_USER_ID_ALT, _UNSET,
    )
    from hermes_constants import get_hermes_home

    def bound(var):
        value = var.get()
        return "" if value is _UNSET else str(value or "")

    if (not owner_key or not session_id or bound(_SESSION_ID) != session_id
            or bound(_SESSION_CHAT_TYPE) not in {"dm", "direct", "private"}
            or is_delegated_child_context() or in_public_interaction()):
        return None
    active_home = get_hermes_home().resolve()
    if home is not None and Path(home).resolve() != active_home:
        return None
    platform, sender_id = bound(_SESSION_PLATFORM), bound(_SESSION_USER_ID)
    chat_id, thread_id = bound(_SESSION_CHAT_ID), bound(_SESSION_THREAD_ID)
    if (not platform or not chat_id
            or owner_key != _owner_key(platform, user_id=sender_id, user_id_alt=bound(_SESSION_USER_ID_ALT))):
        return None
    return active_home, platform, sender_id, chat_id, thread_id


def recent_self_utterances(
    owner_key: str, session_id: str, *, home=None, since=None, unconsumed_only: bool = False,
) -> str:
    """Read receipts only in the current bound, private main turn; never append a chat row."""
    from hermes_time import now

    route = _private_receipt_route(owner_key, session_id, home)
    if route is None:
        return ""
    clock = now()
    window_floor = clock.timestamp() - 12 * 60 * 60
    if since is None and unconsumed_only:
        try:
            since = _previous_reply_time(route[0], session_id, route[1])
        except Exception as exc:
            logger.warning("Could not read previous reply time (%s)", type(exc).__name__)
    lower = window_floor if since is None else max(window_floor, since)
    rows = _self_utterance_rows(owner_key, session_id, route, since=lower, through=clock.timestamp())
    if not rows:
        return ""
    return "\n".join([HEADER, *_format_experience_rows(rows, clock.tzinfo)])


def _self_utterance_rows(owner_key, session_id, route, *, since, through, strict=False):
    active_home, platform, sender_id, chat_id, thread_id = route
    rows = []
    try:
        from cron.executions import recent_sent_text

        rows.extend((stamp, "定时主动发言", text) for stamp, text in recent_sent_text(
            platform=platform, chat_id=chat_id, thread_id=thread_id, owner_key=owner_key,
            since=since, home=active_home))
    except Exception as exc:
        if strict:
            raise
        logger.warning("Could not read cron send receipts (%s)", type(exc).__name__)
    try:
        from hermes_cli.plugins import get_plugin_manager

        runtime = getattr(get_plugin_manager(), "_bilibili_public_runtime", None)
        if runtime:
            rows.extend(runtime[0].recent_own_replies(
                session_id=session_id, platform=platform, sender_id=sender_id, since=since))
    except Exception as exc:
        if strict:
            raise
        logger.warning("Could not read Bilibili send receipts (%s)", type(exc).__name__)
    return [(stamp, label, text) for stamp, label, text in rows if since < stamp <= through]


def _format_experience_rows(rows, tzinfo):
    for stamp, label, text in sorted(rows):
        timestamp = datetime.fromtimestamp(stamp, tz=tzinfo).strftime("%Y-%m-%d %H:%M")
        first, *tail = text.split("\n")
        yield _SEPARATOR.join((timestamp, label, first))
        yield from (_CONTINUATION + line for line in tail)


def parse_self_utterance_context(block: str) -> list[str]:
    """Recover verbatim outbound text, including line breaks, for attribution checks."""
    header, _, rest = block.partition("\n")
    if header != HEADER:
        return []
    result = []
    for line in rest.split("\n"):
        if line.startswith(_CONTINUATION) and result:
            result[-1] += "\n" + line[len(_CONTINUATION):]
            continue
        pieces = line.split(_SEPARATOR, 2)
        if len(pieces) == 3:
            result.append(pieces[2])
    return result
