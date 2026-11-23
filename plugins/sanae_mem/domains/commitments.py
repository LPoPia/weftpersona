from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..infra.defaults import DEFAULT_PERSONA_NAME
from ..governance.group_scope import GroupGrant, exclude_group_rows, is_group_owner
from ..governance.source_context import PROCESSING_SOURCE_KEYS, parse_source, source_json
from ..storage.sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)


DEFAULT_MASKED_OCCUPATION = "还答应了另一位朋友一件事待办"

COMMITMENT_STATUS_COMPONENT = "commitments"
COMMITMENT_ARCHIVE_NAME = "commitments-archived.jsonl"

_COMMITMENT_ARCHIVE_LOCKS_GUARD = threading.Lock()
_COMMITMENT_ARCHIVE_LOCKS: Dict[str, threading.Lock] = {}


def _archive_lock_key(archive_path: Path) -> str:
    raw = os.path.abspath(os.fspath(archive_path))
    if raw.startswith("\\\\?\\UNC\\"):
        raw = "\\\\" + raw[8:]
    elif raw.startswith("\\\\?\\"):
        raw = raw[4:]
    return os.path.normcase(os.path.normpath(raw))


def _commitment_archive_lock(archive_path: Path) -> threading.Lock:
    key = _archive_lock_key(archive_path)
    with _COMMITMENT_ARCHIVE_LOCKS_GUARD:
        lock = _COMMITMENT_ARCHIVE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _COMMITMENT_ARCHIVE_LOCKS[key] = lock
        return lock

_TOOL_TEXT_KEYS = ("content", "output", "result", "data", "message", "text")
_TOOL_FAILURE_STATES = frozenset({
    "error", "failed", "failure", "false", "denied", "timeout", "timed_out",
    "cancelled", "canceled", "aborted", "rejected", "unavailable", "blocked",
    "not_found", "no_results",
})
_TOOL_UNFINISHED_STATES = frozenset({
    "pending", "queued", "staged", "accepted", "running", "in_progress",
    "processing", "started", "submitted", "partial", "incomplete", "not_started",
    "waiting",
})
_TOOL_ERROR_PREFIXES = ("Error:", "Exception:", "Traceback", "Failed", "error:")
_RECEIPT_KEYS = frozenset({
    "ok", "success", "failed", "is_error", "error", "errors", "exception",
    "traceback", "status", "state", "message", "detail", "details", "code",
    "reason", "retryable", *_TOOL_TEXT_KEYS,
})


def _coerce_now(value: datetime | str | int | float | None) -> datetime:
    if value is None:
        return datetime.now().astimezone()
    if isinstance(value, bool):
        return datetime.now().astimezone()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).astimezone()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now().astimezone()
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.astimezone()
    return datetime.now().astimezone()


def _quote_in_text(quote: str, text: str) -> bool:
    q = re.sub(r"\s+", " ", str(quote or "")).strip()
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return bool(q) and q in t


def _user_quote_evidence(user_text: str, user_quote: str) -> tuple:
    quote = re.sub(r"\s+", " ", str(user_quote or "")).strip()
    if not quote or not _quote_in_text(quote, user_text):
        return False, "quote_not_in_current_user_text"
    return True, ""


def _contains_identity_token(summary: str, token: str) -> bool:
    token = str(token or "").strip()
    if not token:
        return False
    if not re.fullmatch(r"[0-9a-f]{16}", token):
        for run in re.findall(r"\d{4,}", token):
            if run in summary:
                return True
    if re.fullmatch(r"[\w.@:-]+", token, flags=re.ASCII):
        pattern = r"(?<![0-9A-Za-z])" + re.escape(token) + r"(?![0-9A-Za-z])"
        return re.search(pattern, summary, flags=re.IGNORECASE) is not None
    return token.lower() in summary.lower()


_QUOTED_SPAN = re.compile(r"[「『“‘\"']([^」』”’\"']{2,})[」』”’\"']")


def _safe_masked_text(value: Any, *, owner_key: str, source_quote: str = "",
                      text: str = "", partner_name: str = "",
                      identity_tokens: Sequence[str] = ()) -> str:
    summary = re.sub(r"\s+", " ", str(value or "")).strip()
    if not summary:
        return ""

    for anchor in (partner_name, owner_key, *identity_tokens):
        if _contains_identity_token(summary, anchor):
            return ""
    if re.search(r"(?:@|https?://|www\.)\S", summary):
        return ""

    private_text = re.sub(r"\s+", " ", str(text or "")).strip()
    quote = re.sub(r"\s+", " ", str(source_quote or "")).strip()
    for span in _QUOTED_SPAN.findall(summary):
        if _quote_in_text(span, quote) or _quote_in_text(span, private_text):
            return ""
    if quote and quote in summary:
        return ""
    return summary


def _embedded_json(value: str) -> Any:
    text = str(value or "").strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _is_receipt_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(str(key) in _RECEIPT_KEYS for key in value)
    )


def _receipt_failure(receipt: Dict[str, Any]) -> str:
    if receipt.get("is_error") is True:
        return "is_error=true"
    if receipt.get("ok") is False:
        return "ok=false"
    if receipt.get("success") is False:
        return "success=false"
    if receipt.get("failed") is True:
        return "failed=true"
    for key in ("error", "errors", "exception", "traceback"):
        if receipt.get(key) not in (None, "", False, [], {}):
            return key
    for key in ("status", "state"):
        status = receipt.get(key)
        if isinstance(status, str):
            token = status.strip().lower()
            if token in _TOOL_FAILURE_STATES:
                return f"{key}={status}"
            if token in _TOOL_UNFINISHED_STATES:
                return f"unfinished:{key}={status}"
        elif status is False:
            return f"{key}=false"
    return ""


def _error_text(text: str) -> str:
    if any(text.startswith(prefix) for prefix in _TOOL_ERROR_PREFIXES):
        return f"error_text={text[:120]}"
    return ""


def _tool_failure_reason(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        parsed = _embedded_json(text)
        if isinstance(parsed, dict):
            return _tool_failure_reason(parsed)
        return _error_text(text)

    if not isinstance(value, dict):
        return ""
    reason = _receipt_failure(value)
    if reason:
        return reason

    for key in _TOOL_TEXT_KEYS:
        if key not in value:
            continue
        child = value[key]
        if isinstance(child, str):
            text = child.strip()
            parsed = _embedded_json(text)
            if isinstance(parsed, dict):
                if not _is_receipt_dict(parsed):
                    continue
                reason = _receipt_failure(parsed)
            else:
                reason = _error_text(text)
            if reason:
                return f"{key}.{reason}"
        elif isinstance(child, dict) and _is_receipt_dict(child):
            reason = _receipt_failure(child)
            if reason:
                return f"{key}.{reason}"
    return ""


def _tool_receipt_fields(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    fields: Dict[str, Any] = {}
    for key in ("ok", "success", "status", "state", "is_error", "error"):
        if key not in value:
            continue
        item = value[key]
        if item in (None, ""):
            continue
        fields[key] = item[:120] if isinstance(item, str) else item
    return fields


def _tool_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in _TOOL_TEXT_KEYS:
            child = value.get(key)
            if isinstance(child, str) and child.strip():
                return child.strip()
            if isinstance(child, (dict, list)) and child:
                return json.dumps(child, ensure_ascii=False, default=str)
        body = {
            key: item for key, item in value.items()
            if key not in ("name", "tool_name", "tool_call_id", "call_id", "turn_id")
        }
        return json.dumps(body or value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def _usable_tool_evidence(tool_results: Sequence[object], turn_id: str) -> tuple:
    usable: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    for index, tr in enumerate(tool_results or ()):
        if isinstance(tr, dict):
            ref = str(tr.get("tool_call_id") or tr.get("call_id") or "").strip()
            name = str(tr.get("name") or tr.get("tool_name") or "").strip()
            stamp = str(tr.get("turn_id") or "").strip()
        else:
            ref = name = stamp = ""
        ref = ref or f"tool-{index}"

        reason = _tool_failure_reason(tr)
        if reason:
            rejected.append({"tool_call_id": ref, "name": name, "reason": reason})
            continue
        if stamp and turn_id and stamp != turn_id:
            rejected.append({
                "tool_call_id": ref, "name": name,
                "reason": f"stale_turn={stamp}",
            })
            continue
        text = _tool_text(tr)
        if not text:
            rejected.append({"tool_call_id": ref, "name": name, "reason": "empty"})
            continue
        usable.append({
            "tool_call_id": ref,
            "name": name,
            "receipt": _tool_receipt_fields(tr),
            "summary": text[:1200],
        })
    return usable, rejected


@dataclass(frozen=True)
class CommitmentContext:
    home: Path
    session_id: str
    platform: str
    partner_name: str
    persona_name: str
    identity_tokens: tuple[str, ...]


@dataclass(frozen=True)
class CommitmentGroup:
    active: bool
    grant: GroupGrant | None


@dataclass(frozen=True)
class CommitmentPrompt:
    system: str
    user: str


class FulfillmentWriter(Protocol):
    def __call__(
        self, text: str, session_id: str, *, importance: float, owner_key: str,
        source: str, source_context: Mapping[str, str], experience_subject: str,
        fact_subject: str, involved_owner_keys: str, read_scope: str,
        fulfillment_commitment_id: int | None = None,
    ) -> int | None: ...


class CommitmentStatusWriter(Protocol):
    def __call__(
        self, *, ok: bool, stage: str, error: str = "", pending: int | None = None,
    ) -> None: ...


def parse_aux_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


class Commitments:
    def __init__(
        self, sqlite: SqliteFacade, *,
        current_context: Callable[[], CommitmentContext],
        current_group: Callable[[], CommitmentGroup],
        read_owner_keys: Callable[[str], Sequence[str]],
        write_fulfillment: FulfillmentWriter,
        has_failure: Callable[[], bool],
        write_status: CommitmentStatusWriter,
        complete: Callable[[CommitmentPrompt], str | None],
    ) -> None:
        self._sqlite = sqlite
        self._current_context = current_context
        self._current_group = current_group
        self._read_owner_keys = read_owner_keys
        self._write_fulfillment = write_fulfillment
        self._has_failure = has_failure
        self._write_status = write_status
        self._complete = complete

    @staticmethod
    def _commitment_owner_clause(owners: Tuple[str, ...], column: str = "owner_key") -> tuple[str, list[str]]:
        """``AND`` clause matching exactly these owners; no owner matches nothing at all."""
        if not owners:
            return ("0", [])
        if len(owners) == 1:
            return (f"{column}=?", [owners[0]])
        return (f"{column} IN ({','.join('?' for _ in owners)})", list(owners))

    def _commitment_resolution_owners(self, owner_key: str) -> Sequence[str]:
        group = self._current_group()
        if group.active:
            return group.grant.resolution_owners if group.grant is not None else ()
        return tuple(key for key in self._read_owner_keys(owner_key) if not is_group_owner(key))

    def record_commitment(
        self,
        owner_key: str,
        text: str,
        *,
        source_quote: str = "",
        turn_id: str = "",
        now: datetime | str | int | float | None = None,
        masked_text: str = "",
        source_context: str = "",
    ) -> Optional[int]:
        """Record a personal future commitment for a trusted owner.

        Idempotent per source, not per words forever: retrying the same complete
        delivered turn returns its own row in any status, while a later
        independent turn that really promises the same words again records a new
        pending item. Migration sources keep deduping by owner and original
        words, so a carried-forward source neither duplicates nor revives a
        resolved row. A short identity-free ``masked_text`` from the same
        auxiliary model is kept for other people's view; anything unsafe falls
        back to the generic occupation. ``source_context`` is the canonical
        provenance JSON of that same turn; empty keeps the private call unchanged.

        A group turn files the promise under its one write owner and records that scene's own
        provenance; a private turn never files a group promise, and a switched-off group records
        nothing.
        """
        if not isinstance(owner_key, str) or not owner_key.strip():
            return None
        owner_key = owner_key.strip()

        if not isinstance(text, str) or not text.strip():
            return None
        text = text.strip()

        group = self._current_group()
        if group.active:
            grant = group.grant
            if grant is None or not grant.write_owner:
                return None
            owner_key = grant.write_owner
            # Only the live grant supplies scene facts. A delivery receipt can add processing metadata.
            source = dict(grant.source)
            provided = parse_source(source_context) or {}
            source.update({key: provided.get(key) or "" for key in PROCESSING_SOURCE_KEYS})
            source_context = source_json(source)
        elif is_group_owner(owner_key):
            return None

        source_quote = str(source_quote or "").strip()
        turn_id = str(turn_id or "").strip()
        now_iso = _coerce_now(now).isoformat()
        context = self._current_context()
        masked = _safe_masked_text(
            masked_text, owner_key=owner_key, source_quote=source_quote,
            text=text, partner_name=context.partner_name,
            identity_tokens=context.identity_tokens,
        ) or DEFAULT_MASKED_OCCUPATION

        with self._sqlite.transaction() as conn:
            if source_quote:
                if turn_id.startswith("legacy_"):
                    existing = conn.execute(
                        "SELECT id FROM commitments WHERE owner_key = ? AND source_quote = ? LIMIT 1",
                        (owner_key, source_quote),
                    ).fetchone()
                else:
                    existing = conn.execute(
                        "SELECT id FROM commitments WHERE owner_key = ? AND source_quote = ? "
                        "AND turn_id = ? LIMIT 1",
                        (owner_key, source_quote, turn_id),
                    ).fetchone()
                if existing is not None:
                    return int(existing["id"])

            existing_text = conn.execute(
                "SELECT id FROM commitments WHERE owner_key = ? AND text = ? AND status = 'pending' LIMIT 1",
                (owner_key, text),
            ).fetchone()
            if existing_text is not None:
                return int(existing_text["id"])

            cursor = conn.execute(
                """
                INSERT INTO commitments (
                    owner_key, text, masked_text, source_quote, turn_id, status,
                    version, created_at, updated_at, source_context
                ) VALUES (?, ?, ?, ?, ?, 'pending', 1, ?, ?, ?)
                """,
                (owner_key, text, masked, source_quote, turn_id, now_iso, now_iso,
                 str(source_context or "")),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def render_pending_commitments(self, current_owner: str) -> str:
        """Render pending commitments: full for current_owner, generic masked for all others.

        Never expires over time. A private entry's SQL never reaches a ``group:`` promise, so
        it cannot even show the generic "someone else's promise" line for another scene; a
        group turn reads only its granted owners and shows them in full.
        """
        if not self._sqlite.is_open:
            return ""

        current_owner = str(current_owner or "").strip()
        query = ("SELECT id, owner_key, text, masked_text, source_quote FROM commitments "
                 "WHERE status = 'pending'")
        params: List[Any] = []
        group = self._current_group()
        if group.active:
            granted = group.grant.read_owners if group.grant is not None else ()
            keys = ([key for key in self._read_owner_keys(current_owner) if key in granted]
                    if current_owner else list(granted))
            clause, clause_params = self._commitment_owner_clause(tuple(keys))
            query += " AND " + clause
            params.extend(clause_params)
        else:
            query += " AND " + exclude_group_rows()
        query += " ORDER BY id ASC"

        rows = self._sqlite.fetchall(query, params)

        if not rows:
            return ""

        read_owners = set(self._read_owner_keys(current_owner)) if current_owner else set()
        lines = []
        context = self._current_context()
        for row in rows:
            if group.active:
                # Every row here already belongs to this scene's grant.
                t = str(row["text"] or "").strip()
                lines.append(t if t.startswith("- ") else f"- {t}")
                continue
            row_owner = str(row["owner_key"] or "").strip()
            if current_owner and row_owner in read_owners:
                t = str(row["text"] or "").strip()
                lines.append(t if t.startswith("- ") else f"- {t}")
            else:
                safe = _safe_masked_text(
                    row["masked_text"], owner_key=row_owner,
                    source_quote=str(row["source_quote"] or ""),
                    text=str(row["text"] or ""), partner_name=context.partner_name,
                    identity_tokens=context.identity_tokens,
                )
                if safe and safe != DEFAULT_MASKED_OCCUPATION:
                    lines.append(f"- {DEFAULT_MASKED_OCCUPATION}（{safe}）")
                else:
                    lines.append(f"- {DEFAULT_MASKED_OCCUPATION}")

        if not lines:
            return ""
        return "[我答应过还没办完的事]\n" + "\n".join(lines)

    def update_commitments_after_delivery(
        self,
        user_text: str,
        assistant_text: str,
        *,
        owner_key: str,
        turn_id: str,
        tool_results: Sequence[object] = (),
    ) -> bool:
        """Add or resolve commitments using this delivered turn only.

        The auxiliary model owns the semantic mapping; this method only proves
        the cited evidence is a usable current-round tool receipt or the
        person's own words this turn, against the exact pending id/version.
        When the model is unavailable the component records
        ``auxiliary_unavailable`` (or the parse failure) and nothing is queued:
        the turn is not replayed and later turns stay independent.
        """
        if not isinstance(owner_key, str) or not owner_key.strip():
            return False
        owner_key = owner_key.strip()

        if not self._sqlite.is_open:
            return False

        user_text = str(user_text or "").strip()
        assistant_text = str(assistant_text or "").strip()
        turn_id = str(turn_id or "").strip()

        # The bound person's older accounts keep their rows; this turn may resolve them too.
        # The snapshot carries no owner key: the auxiliary model only sees id/text/quote/version.
        # A group turn takes only its writable owner, so a narrow thread's material never closes
        # the parent's shared promise.
        read_owners = self._commitment_resolution_owners(owner_key)
        if not read_owners:
            return False
        pending_rows = self._sqlite.fetchall(
            """
            SELECT id, text, source_quote, version
              FROM commitments
             WHERE owner_key IN (%s) AND status = 'pending'
             ORDER BY id ASC
            """ % ",".join("?" for _ in read_owners),
            tuple(read_owners),
        )

        pending_items = [
            {"id": int(r["id"]), "text": str(r["text"]), "quote": str(r["source_quote"]), "version": int(r["version"])}
            for r in pending_rows
        ]

        extracted, failure = self._call_commitments_llm(
            user_text=user_text, assistant_text=assistant_text,
            pending_items=pending_items, tool_results=tool_results, turn_id=turn_id,
        )

        if isinstance(extracted, dict):
            ok = bool(
                self._apply_commitment_updates(
                    owner_key, turn_id, assistant_text=assistant_text, user_text=user_text,
                    tool_results=tool_results,
                    new_commitments=extracted.get("new_commitments") or [],
                    resolutions=extracted.get("resolutions") or [],
                    pending_items=pending_items,
                )
            )
            if ok and self._has_failure():
                self._write_status(ok=True, stage="retry")
            return ok

        self._write_status(
            ok=False, stage=failure, error=failure, pending=len(pending_items))
        logger.warning(
            "sanae_mem: commitments %s on delivered turn %s; "
            "nothing queued, later turns stay independent", failure, turn_id)
        return False

    def _call_commitments_llm(
        self,
        *,
        user_text: str,
        assistant_text: str,
        pending_items: List[Dict[str, Any]],
        tool_results: Sequence[object],
        turn_id: str,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        usable_tools, rejected_tools = _usable_tool_evidence(tool_results, turn_id)
        tools_str = json.dumps(
            {"successful": usable_tools, "not_usable": rejected_tools},
            ensure_ascii=False,
        ) if (usable_tools or rejected_tools) else "无"
        pending_str = json.dumps(pending_items, ensure_ascii=False) if pending_items else "无"

        persona_name = self._current_context().persona_name.strip() or DEFAULT_PERSONA_NAME
        system_prompt = (
            f"你负责维护{persona_name}与眼前对话者的待办约定与承诺状态。只处理眼前对话者本人，绝不关联其他人的事项。\n"
            f"【新增承诺】仅当{persona_name}在本轮送达的回复里亲口答应对方之后要做的具体未来事项时输出；"
            "text 简明写事项，quote 必须从本轮回复逐字摘录；"
            "masked_text 是给其他人看的第三人称事项简述（如「帮看一份技术日志」），"
            "只写事项本身；不要真实姓名、账号/ID、网址或带引号的原文，正常事项词可以保留，拿不准就留空；"
            "随口寒暄、哄劝安抚、玩笑、一般看法、只是准备/打算但还没明确答应的，不要输出；"
            "没有具体新承诺就输出空数组。\n"
            "【销账】程序只按本轮工具回执的顶层分流：successful 表示这次调用本身没有报错/未完成，"
            "not_usable 表示调用失败、排队（queued/staged）/未完成或旧轮结果，绝不能当作履约证据；"
            "successful 返回的材料里，被查询对象自身的 pending/running/failed 只是数据状态，"
            "不代表这次调用失败；但如果整次调用只是排队/受理，仍不能销账。\n"
            "办完（action='fulfilled'）必须满足其一："
            "a) tool_call_id 指向 successful 中确实办成该具体事项的一项；"
            "b) user_quote 逐字摘录本轮用户消息中明确确认已完成/已收到的原话。"
            f"{persona_name}只说准备去办、调用排队/受理中、成功但无关、旧轮结果，都不能销账。\n"
            "取消（action='cancelled'）必须 user_quote 逐字摘录本轮用户消息中明确取消或"
            "不再需要该具体事项的原话；问句、条件、反话/玩笑、未来计划、含糊或礼貌应答都不算。"
            "同一用户多件待办互不连坐，只销有确凿对应证据的一项；没达到证据条件的一律保持 pending。\n"
            "只输出合法 JSON，不要任何 Markdown 标记或多余文字：\n"
            f'{{"new_commitments":[{{"text":"事项描述","quote":"{persona_name}原话片段",'
            '"masked_text":"给外人看的无身份简述或空串"}],'
            '"resolutions":[{"id":123,"action":"fulfilled|cancelled","evidence":"依据说明",'
            '"tool_call_id":"本轮 successful 的 tool_call_id；本人确认时留空",'
            '"user_quote":"本人原话逐字片段；工具证据时留空"}]}'
        )

        user_prompt = (
            f"【本轮用户消息（当前对话者原话）】\n{user_text}\n\n"
            f"【{persona_name}本轮送达的回复】\n{assistant_text}\n\n"
            f"【本轮工具证据】\n{tools_str}\n\n"
            f"【当前该用户名下尚未办完的待办承诺】\n{pending_str}\n"
        )

        raw = self._complete(CommitmentPrompt(system=system_prompt, user=user_prompt))
        if raw is None:
            return None, "auxiliary_unavailable"
        parsed = parse_aux_json(raw)
        return parsed, "" if parsed is not None else "invalid_json"

    def _apply_commitment_updates(
        self,
        owner_key: str,
        turn_id: str,
        *,
        assistant_text: str,
        user_text: str,
        tool_results: Sequence[object],
        new_commitments: List[Dict[str, Any]],
        resolutions: List[Dict[str, Any]],
        pending_items: List[Dict[str, Any]],
        now: datetime | str | int | float | None = None,
    ) -> bool:
        version_map = {item["id"]: item["version"] for item in pending_items}
        item_map = {item["id"]: item for item in pending_items}
        usable_tools, rejected_tools = _usable_tool_evidence(tool_results, turn_id)
        usable_by_ref = {item["tool_call_id"]: item for item in usable_tools}
        rejected_by_ref = {item["tool_call_id"]: item for item in rejected_tools}

        occurred_dt = _coerce_now(now)
        now_iso = occurred_dt.isoformat()
        failure = ""
        had_rejected = False
        if not self._sqlite.is_open:
            return False

        for res in resolutions:
            if not isinstance(res, dict):
                had_rejected = True
                continue
            cid = res.get("id")
            action = res.get("action")
            evidence = str(res.get("evidence") or "").strip()
            tool_ref = str(res.get("tool_call_id") or "").strip()
            user_quote = str(res.get("user_quote") or evidence).strip()
            if not isinstance(cid, int) or action not in ("fulfilled", "cancelled"):
                had_rejected = True
                continue
            expected_ver = version_map.get(cid)
            target_item = item_map.get(cid)
            if expected_ver is None or target_item is None:
                had_rejected = True
                continue

            valid_resolution = False
            verified_evidence = ""

            if action == "cancelled":
                quote_ok, quote_reason = _user_quote_evidence(
                    user_text, user_quote)
                if not quote_ok:
                    logger.warning(
                        "sanae_mem: rejected cancellation quote for %d (%s)",
                        cid, quote_reason)
                else:
                    valid_resolution = True
                    verified_evidence = f"user_cancellation:{user_quote[:120]}"

            elif action == "fulfilled":
                if tool_ref:
                    tool = usable_by_ref.get(tool_ref)
                    if tool is not None:
                        valid_resolution = True
                        verified_evidence = f"tool_result:{tool_ref}:{tool['summary'][:120]}"
                    else:
                        rejected = rejected_by_ref.get(tool_ref)
                        if rejected is not None:
                            logger.warning(
                                "sanae_mem: rejected fulfillment citing unusable tool result %s (%s) for %d",
                                tool_ref, rejected["reason"], cid)
                        else:
                            logger.warning(
                                "sanae_mem: rejected fulfillment citing unknown tool result %s for %d",
                                tool_ref, cid)

                if not valid_resolution and user_quote:
                    quote_ok, quote_reason = _user_quote_evidence(
                        user_text, user_quote)
                    if quote_ok:
                        valid_resolution = True
                        verified_evidence = f"user_confirmation:{user_quote[:120]}"
                    else:
                        logger.warning(
                            "sanae_mem: rejected fulfillment quote for %d (%s)",
                            cid, quote_reason)

                if not valid_resolution:
                    logger.warning("sanae_mem: rejected fulfillment without valid evidence for %d", cid)

            if not valid_resolution:
                had_rejected = True
                continue

            owner_clause, owner_params = self._commitment_owner_clause(
                tuple(self._commitment_resolution_owners(owner_key)))
            archive_path = self._commitment_archive_path()
            with _commitment_archive_lock(archive_path):
                row = self._sqlite.fetchone(
                    f"SELECT * FROM commitments WHERE id = ? AND status = 'pending' AND {owner_clause}",
                    (cid, *owner_params),
                )
                if row is None or int(row["version"]) != expected_ver:
                    continue
                try:
                    resolved = self._archive_and_resolve(
                        row, archive_path, action=action,
                        evidence=verified_evidence or evidence, now_iso=now_iso,
                    )
                except OSError as exc:
                    logger.error(
                        "sanae_mem: commitment %d archive failed; kept pending (%s)",
                        cid, type(exc).__name__)
                    failure = failure or "archive_failed"
                    continue
                except sqlite3.Error as exc:
                    logger.error(
                        "sanae_mem: commitment %d resolve failed (%s)",
                        cid, type(exc).__name__)
                    failure = failure or "resolve_failed"
                    continue

            if not resolved:
                continue

            if action == "fulfilled":
                try:
                    row_owner = str(row["owner_key"])
                    event_id = self.record_fulfillment_event(
                        row_owner if is_group_owner(row_owner) else owner_key,
                        str(row["text"] or ""), cid,
                    )
                except (sqlite3.Error, OSError) as exc:
                    logger.error(
                        "sanae_mem: commitment %d fulfillment event failed (%s)",
                        cid, type(exc).__name__)
                    event_id = None
                if event_id is None:
                    failure = failure or "event_write_failed"

        for nc in new_commitments:
            if not isinstance(nc, dict):
                had_rejected = True
                continue
            text = str(nc.get("text") or "").strip()
            quote = str(nc.get("quote") or "").strip()
            if not quote or quote not in assistant_text:
                logger.warning("sanae_mem: rejected commitment with fabricated or missing quote: %r", quote)
                had_rejected = True
                continue
            if not text:
                had_rejected = True
                continue
            try:
                recorded_id = self.record_commitment(
                    owner_key, text, source_quote=quote, turn_id=turn_id,
                    now=occurred_dt, masked_text=str(nc.get("masked_text") or ""),
                )
            except (sqlite3.Error, OSError) as exc:
                logger.error(
                    "sanae_mem: commitment record failed (%s)",
                    type(exc).__name__,
                )
                failure = failure or "record_failed"
                continue
            if recorded_id is None:
                had_rejected = True

        if failure:
            stage = ("archive" if failure == "archive_failed"
                     else "event" if failure == "event_write_failed"
                     else "record" if failure == "record_failed"
                     else "resolve")
            self._write_status(ok=False, stage=stage, error=failure)
        return not failure and not had_rejected

    def _archive_and_resolve(
        self, row: sqlite3.Row, archive_path: Path, *, action: str, evidence: str, now_iso: str,
    ) -> bool:
        from ..infra.archive import _append_fsynced_jsonl

        reason = "commitment_cancelled" if action == "cancelled" else "commitment_fulfilled"
        entry = {
            **dict(row),
            "archived_at": datetime.now().astimezone().isoformat(),
            "archive_reason": reason,
            "resolution": {"action": action, "evidence": str(evidence or "")[:200]},
        }
        _append_fsynced_jsonl(archive_path, (json.dumps(entry, ensure_ascii=False, default=str),))

        column = "fulfilled_at" if action == "fulfilled" else "cancelled_at"
        with self._sqlite.transaction() as conn:
            cursor = conn.execute(
                f"""
                UPDATE commitments
                   SET status = ?,
                       version = version + 1,
                       updated_at = ?,
                       {column} = ?,
                       resolution_evidence = ?
                 WHERE id = ? AND version = ? AND status = 'pending' AND owner_key = ?
                """,
                (action, now_iso, now_iso, evidence, row["id"], row["version"], row["owner_key"]),
            )
            return cursor.rowcount > 0

    def _commitment_archive_path(self) -> Path:
        return self._current_context().home / "memory-archive" / COMMITMENT_ARCHIVE_NAME

    def record_fulfillment_event(
        self, owner_key: str, text: str, commitment_id: int,
    ) -> Optional[int]:
        from ..governance.source_context import SELF_SUBJECT, self_attribution

        context = self._current_context()
        content = f"我已兑现此前答应的事：{text}"
        if is_group_owner(owner_key):
            # A group promise's completion belongs to that scene's own events row; it is never
            # filed as a private "self" event. The row keeps the promise's own owner: an
            # inherited thread closes the exact promise it filed while its audience was
            # ``unknown`` and records the completion under that same exact owner, so nothing is
            # pushed into the shared parent row. A narrow thread's parent promise is not in its
            # resolution owners and writes nothing here.
            grant = self._current_group().grant
            owner = str(owner_key)
            if grant is None or owner not in grant.resolution_owners:
                return None
            source = dict(grant.source)
            source["ref"] = f"commitment-fulfilled:{int(commitment_id)}"
            source["commitment_id"] = str(int(commitment_id))
            involved = (json.dumps([grant.person_key], ensure_ascii=False)
                        if grant.person_key else "")
            return self._write_fulfillment(
                content, context.session_id,
                importance=0.6, owner_key=owner, source="explicit",
                source_context=source,
                experience_subject=SELF_SUBJECT,
                fact_subject=f"owner:{grant.person_key}" if grant.person_key else "",
                involved_owner_keys=involved,
                read_scope=json.dumps({"kind": "group", "owner": owner},
                                      ensure_ascii=False, sort_keys=True),
                fulfillment_commitment_id=int(commitment_id),
            )
        subject, fact_subject, involved, scope = self_attribution(owner_key)
        return self._write_fulfillment(
            content, context.session_id,
            importance=0.6, owner_key=owner_key, source="explicit",
            source_context={
                "kind": "self",
                "platform": context.platform,
                "visibility": "private",
                "certainty": "observed",
                "ref": f"commitment-fulfilled:{int(commitment_id)}",
            },
            experience_subject=subject, fact_subject=fact_subject,
            involved_owner_keys=involved, read_scope=scope,
        )

    def list_commitments(
        self,
        owner_key: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List promises this call may see.

        A group turn is limited to the owners its current grant allows (an explicit owner never
        widens it); a private entry's SQL never includes a ``group:`` promise at all.
        """
        if not self._sqlite.is_open:
            return []
        query = "SELECT * FROM commitments WHERE 1=1"
        params: List[Any] = []
        group = self._current_group()
        if group.active:
            granted = group.grant.read_owners if group.grant is not None else ()
            keys = [key for key in self._read_owner_keys(owner_key) if key in granted] \
                if owner_key else list(granted)
            clause, clause_params = self._commitment_owner_clause(tuple(keys))
            query += " AND " + clause
            params.extend(clause_params)
            owner_key = None
        else:
            query += " AND " + exclude_group_rows()
        if owner_key:
            # The bound person's older accounts keep their rows; the reader may see them.
            keys = self._read_owner_keys(owner_key)
            if len(keys) > 1:
                query += " AND owner_key IN (%s)" % ",".join("?" for _ in keys)
                params.extend(keys)
            else:
                query += " AND owner_key = ?"
                params.append(keys[0] if keys else owner_key.strip())
        if status:
            query += " AND status = ?"
            params.append(status.strip())
        query += " ORDER BY id ASC"

        rows = self._sqlite.fetchall(query, params)
        return [dict(r) for r in rows]

    def resolve_commitment_with_reason(
        self,
        commitment_id: int,
        action: str,
        evidence: str = "",
        expected_version: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Resolve a commitment explicitly with CAS, returning ``(resolved, reason)``.

        In a group turn only the owners the live grant derives for resolution may be resolved
        with CAS (an inherited thread may close the exact promise it wrote while its audience was
        ``unknown``; a narrow thread never closes the parent's shared promise). A private entry
        never resolves a group promise.
        """
        if not self._sqlite.is_open or action not in ("fulfilled", "cancelled"):
            return False, "not_resolved"
        group = self._current_group()
        if group.active:
            owners = group.grant.resolution_owners if group.grant is not None else ()
            owner_clause, owner_params = self._commitment_owner_clause(owners)
        else:
            owner_clause, owner_params = exclude_group_rows(), []
        now_iso = datetime.now().astimezone().isoformat()
        archive_path = self._commitment_archive_path()
        with _commitment_archive_lock(archive_path):
            row = self._sqlite.fetchone(
                f"SELECT * FROM commitments WHERE id = ? AND {owner_clause}",
                (commitment_id, *owner_params),
            )
            if row is None or row["status"] != "pending":
                return False, "not_resolved"
            cur_ver = int(row["version"])
            if expected_version is not None and cur_ver != expected_version:
                return False, "not_resolved"

            try:
                resolved = self._archive_and_resolve(
                    row, archive_path, action=action, evidence=evidence, now_iso=now_iso,
                )
            except OSError as exc:
                logger.error(
                    "sanae_mem: commitment %d archive failed; kept pending (%s)",
                    commitment_id, type(exc).__name__)
                self._write_status(ok=False, stage="archive", error="archive_failed")
                return False, "archive_failed"

            return resolved, "" if resolved else "not_resolved"

    def resolve_commitment(
        self,
        commitment_id: int,
        action: str,
        evidence: str = "",
        expected_version: Optional[int] = None,
    ) -> bool:
        """Resolve a commitment explicitly with CAS."""
        resolved, _ = self.resolve_commitment_with_reason(
            commitment_id,
            action,
            evidence=evidence,
            expected_version=expected_version,
        )
        return resolved
