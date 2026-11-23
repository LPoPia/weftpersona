from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple

from ..infra import auxiliary, host_bridge
from ..domains.commitments import _quote_in_text, _usable_tool_evidence, parse_aux_json
from ..cognitive.daily_inputs import _VISION_IMAGE_NOTE, _split_known_meme_notes, flatten_user_text
from ..cognitive.daily_rules import _FACE_TAG, _GATEWAY_ENVELOPE, _HOST_PREFIX
from .group_memory import member_impression_read_owners
from ..cognitive.impression import Impression, ImpressionTarget, apply_impression, read_impression
from ..governance.group_scope import GroupGrant
from ..governance.source_context import SELF_SUBJECT, parse_source, source_json
from ..infra.textutil import _embed, _pack_vector, event_keywords

if TYPE_CHECKING:
    from ..cognitive.inner_wiring import InnerWiring
    from ..governance.llm_guard import AuxiliaryModel
    from ..storage.sqlite_facade import SqliteFacade
    from .ingest import Ingest

logger = logging.getLogger(__name__)

PROCESSING = "group_receipt_ingest"
PROCESSING_VERSION = "1"

_EMPTY_TEXT_PLACEHOLDER = "(The user sent a message with no text content)"
_DISCORD_TRIGGER_NOTE_RE = re.compile(r"^\[Triggering message id: [^\n\]]+\](?:\n\n|\s*)")
_SLACK_APP_CONTEXT_RE = re.compile(r"^\[Slack app context: user is viewing channel [^\n\]]+\](?:\n\n|\s*)")
_SLACK_ATTACHMENT_NOTICE_RE = re.compile(r"^\[Slack attachment notice\]\n(?:- [^\n]*\n)*(?:\n|$)")
_REPLY_ID_BLOCK_RE = re.compile(
    r"^\[Replying to(?: your previous message)?: message ID [^\]\n]+\](?:\n\n|\n|$)"
)
_NATIVE_IMAGE_HINT_RE = re.compile(r"^\[Image attached(?: at)?: [^\n]*\](?:\n\n|\n|$)")
_VOICE_FAIL_NOTE_RE = re.compile(
    r"^\[voice message could not be transcribed automatically; the audio is available at: [^\n\]]+\](?:\n\n|\n|$)"
)
_VOICE_EMPTY_NOTE_RE = re.compile(
    r"^\[The user sent a voice message but it came through empty or inaudible[^\]]*\](?:\n\n|\n|$)"
)
_MEDIA_FILE_NOTE_RE = re.compile(
    r"^\[The user sent (?:(?:a|an) )?(?:audio file attachment|video attachment|text document|document): [^\]]+\](?:\n\n|\n|$)"
)
_QQ_QUOTED_LEN_RE = re.compile(r"^\[Quoted message; length=(\d+)\]:\n")
_BACKFILL_HEADERS = (
    "[Recent channel messages]",
    "[Messages prefixed with [unverified]",
    "[Context around the replied-to message]",
    "[Thread context]",
    "[Slack thread context",
)
_BACKFILL_NEW_MSG_RE = re.compile(r"(?:^|\n\n)\[New message\]\n")
_PLATFORM_SENDER_SUFFIX_RE = re.compile(
    r"(?:\|\s*|\A)(?:discord|slack|qqbot|qq|telegram|whatsapp|matrix|feishu|lark|wecom|dingtalk|signal|email|sms|cli|api)\s+user\s+(?:\S+|<@[^>]+>)\s*$",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "你在整理{persona}在一个群聊里刚刚发生的一轮往返。只依据下面给出的真实材料，"
    "不补时间、地点、动机、结果或人物关系。\n"
    "【events】只写本轮群成员原话里确实说到的具体事情。quote 必须从【本轮群成员消息】逐字摘一句；"
    "text 是你对这一句的简明概括，不要写进 quote，也不要写成我说的话。"
    "他转述别人的话时，只当作他本人的转述，被转述的人不是本轮发言人。"
    "如果这条事件是在转述/回应【场景】里给出的 reply_to_message_id 所指的那条消息，"
    "可在该 event 上附带 \"paraphrase_ref\"（只能原样填 reply_to_message_id，没有或不确定就省略或填空串）。\n"
    "【impression】只写这位成员在这个群里表现出的相对稳定的样子。quote 逐字出自"
    "【本轮群成员消息】；没有足够的新证据就输出 null。\n"
    "【new_commitments】仅当【我这条已送达的回复】里确实答应了这个群一件将来要做的具体事时输出；"
    "quote 必须逐字摘自【我这条已送达的回复】。群成员单方面提出请求、我还没答应的，不是我的承诺。\n"
    "【commitment_updates】只处理【这个群当前还没办完的约定】里出现过的 id，用它的 version 认同一件。"
    "action=fulfilled 必须满足其一：basis=delivered_reply，这条已送达的回复本身就是答应交付的东西，"
    "quote 逐字摘自【我这条已送达的回复】；basis=user_confirmed，本轮群成员原话确认已经完成或收到，"
    "quote 逐字摘自【本轮群成员消息】；basis=tool_result，本轮有真实工具成功回执，quote 摘自本轮真实消息。"
    "我只是一句话声称做完了外部动作、却既没有真实工具成功也没有本人确认的，不算兑现；拿不准就不要输出。"
    "action=cancelled 只能因为本轮真实消息明确取消或不再需要，quote 逐字摘自那条消息。"
    "没有变化就输出空数组。\n"
    "只输出合法 JSON，不要 Markdown 代码块或多余文字：\n"
    '{{"events":[{{"text":"概括","quote":"他的原话","paraphrase_ref":"仅当确属转述该回复消息时原样填 reply_to_message_id，否则省略或空串"}}],'
    '"impression":null 或 {{"speaker":"user","text":"我的理解","quote":"他的原话"}},'
    '"new_commitments":[{{"text":"我答应的具体事项","quote":"我回复里的原话",'
    '"masked_text":"给外人看的无身份简述或空串"}}],'
    '"commitment_updates":[{{"id":1,"action":"fulfilled|cancelled",'
    '"basis":"delivered_reply|user_confirmed|tool_result","quote":"原话"}}]}}'
)

_SUMMARY_SYSTEM_PROMPT = (
    "你在整理{persona}所在的一个群聊里留下的一条事件。只依据下面给出的本条原文和来源，"
    "不补时间、地点、动机、结果或人物关系，不读私人档案、用户档案、内心状态、其它群或其它线程。"
    "summary 只概括这条原文里确实说到的事，不改写成人物的私人自述。"
    "只输出合法 JSON，不要 Markdown 代码块或多余文字：{{\"summary\":\"这条群事件的简明概括\"}}"
)

_SUMMARY_SOURCE_KEYS = (
    "platform", "chat_type", "chat_id", "thread_id", "parent_chat_id",
    "thread_visibility", "ref", "time", "occurred_at", "author_id",
)


@dataclass(frozen=True)
class GroupDeliveryContext:
    grant: GroupGrant | None
    session_id: str
    persona_name: str


class GroupDeliveryWorker(Protocol):
    @property
    def ingest(self) -> Ingest: ...

    @property
    def inner_wiring(self) -> InnerWiring: ...

    def group_delivery_context(self) -> GroupDeliveryContext: ...
    def runtime_llm(self) -> AuxiliaryModel | None: ...
    def list_commitments(
        self, owner_key: str | None = None, status: str | None = None,
    ) -> list[dict[str, Any]]: ...
    def record_commitment(
        self, owner_key: str, text: str, *, source_quote: str = ..., turn_id: str = ...,
        now: datetime | str | int | float | None = ..., masked_text: str = ...,
        source_context: str = ...,
    ) -> int | None: ...
    def resolve_commitment_with_reason(
        self, commitment_id: int, action: str, evidence: str = ...,
        expected_version: int | None = ...,
    ) -> tuple[bool, str]: ...
    def write_group_delivery_event(
        self, grant: GroupGrant, *, quote: str, summary: str, delivered_at: object,
        delivery_ref: str, paraphrase_ref: str,
    ) -> int | None: ...


def _processing_source(
    grant: Any, *, delivery_ref: str = "", commitment_id: str = "", paraphrase_ref: str = "",
) -> dict:
    source = dict(grant.source)
    source["processing"] = PROCESSING
    source["processing_version"] = PROCESSING_VERSION
    source["delivery_ref"] = str(delivery_ref or "")
    source["commitment_id"] = str(commitment_id or "")
    if paraphrase_ref:
        source["paraphrase_ref"] = paraphrase_ref
    return source


def _strip_replying_to_block(text: str) -> str:
    m_id = _REPLY_ID_BLOCK_RE.match(text)
    if m_id:
        return text[m_id.end():]
    m_quote = re.match(r'^\[Replying to(?: your previous message)?: "', text)
    if not m_quote:
        return text
    rest = text[m_quote.end():]
    close_idx = rest.find('"]\n\n')
    if close_idx != -1:
        return rest[close_idx + 4:]
    close_single = rest.find('"]\n')
    if close_single != -1:
        return rest[close_single + 3:]
    if rest.endswith('"]'):
        return ""
    return text


def _strip_platform_sender_label(
    text: str, *, platform: str = "", author_id: str = "", sender_name: str = "",
    allow_plain_label: bool = False,
) -> str:
    if not text.startswith("["):
        return text
    depth = 0
    end_idx = -1
    for idx, ch in enumerate(text):
        if ch in "\r\n":
            return text
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break
    if end_idx <= 1:
        return text
    inner = text[1:end_idx].strip()
    is_label = bool(_PLATFORM_SENDER_SUFFIX_RE.search(inner))
    if not is_label and author_id:
        is_label = inner.endswith(f"Slack user <@{author_id}>") or (
            bool(platform) and inner.endswith(f"{platform} user {author_id}")
        )
    if not is_label and sender_name and inner == sender_name:
        is_label = True
    if not is_label and allow_plain_label:
        is_label = True
    if not is_label:
        return text
    tail = text[end_idx + 1:]
    if tail.startswith(" ") or tail.startswith("\t"):
        return tail[1:]
    if tail.startswith("\n"):
        return tail[1:]
    if tail == "":
        return ""
    return text


def _strip_qq_quoted_block(text: str) -> Tuple[str, bool]:
    m_len = _QQ_QUOTED_LEN_RE.match(text)
    if m_len:
        end = m_len.end() + int(m_len.group(1))
        if end == len(text):
            return "", True
        if text[end:end + 2] != "\n\n":
            return "", True
        return text[end + 2:], True
    if text.startswith("[Quoted message]:"):
        return "", True
    return text, False


def authored_group_user_text(
    raw: Any, *, authored: bool = False, sender_name: str = "",
    platform: str = "", author_id: str = "",
) -> str:
    if authored:
        if isinstance(raw, str):
            text = raw.strip()
        else:
            text = flatten_user_text(raw).strip()
        return "" if text == _EMPTY_TEXT_PLACEHOLDER else text

    if isinstance(raw, str):
        text = raw
    else:
        text = flatten_user_text(raw)
    text, _ = _split_known_meme_notes(text)
    text = text.lstrip("\r\n")
    after_new_msg = False
    for _ in range(16):
        prev = text
        text = _DISCORD_TRIGGER_NOTE_RE.sub("", text)
        text = _HOST_PREFIX.sub("", text)
        text = _SLACK_APP_CONTEXT_RE.sub("", text)
        text = _SLACK_ATTACHMENT_NOTICE_RE.sub("", text)
        text = _NATIVE_IMAGE_HINT_RE.sub("", text)
        text = _VOICE_FAIL_NOTE_RE.sub("", text)
        text = _VOICE_EMPTY_NOTE_RE.sub("", text)
        text = _MEDIA_FILE_NOTE_RE.sub("", text)
        m_vis = _VISION_IMAGE_NOTE.match(text)
        if m_vis:
            text = text[m_vis.end():].lstrip("\n")
        text = _strip_replying_to_block(text)
        if any(text.startswith(hdr) for hdr in _BACKFILL_HEADERS):
            m_new = _BACKFILL_NEW_MSG_RE.search(text)
            if not m_new:
                return ""
            text = text[m_new.end():]
            after_new_msg = True
        stripped_label = _strip_platform_sender_label(
            text, platform=platform, author_id=author_id, sender_name=sender_name,
            allow_plain_label=after_new_msg,
        )
        if stripped_label != text:
            text = stripped_label
            after_new_msg = False
        text, qq_handled = _strip_qq_quoted_block(text)
        if qq_handled and not text:
            return ""
        text = text.lstrip("\r\n")
        if text == prev:
            break
    text = _GATEWAY_ENVELOPE.sub(" ", text)
    text = _FACE_TAG.sub("", text).strip()
    if text == _EMPTY_TEXT_PLACEHOLDER:
        return ""
    return text


def _valid_paraphrase_ref(grant: Any, candidate: Any, *, delivery_ref: str = "") -> str:
    value = str(candidate or "").strip()
    if not value:
        return ""
    reply_ref = str(grant.source.get("reply_to_message_id") or "").strip()
    self_ref = str(grant.source.get("ref") or "").strip()
    delivery = str(delivery_ref or "").strip()
    if not reply_ref or value != reply_ref:
        return ""
    if value == self_ref or (delivery and value == delivery):
        return ""
    return value


def _now_iso(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value)).astimezone().isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    return datetime.now().astimezone().isoformat()


def _group_read_scope(grant: Any) -> str:
    return json.dumps({"kind": "group", "owner": grant.write_owner},
                      ensure_ascii=False, sort_keys=True)


def _involved(grant: Any) -> str:
    return json.dumps([grant.person_key], ensure_ascii=False) if grant.person_key else ""


@dataclass(frozen=True)
class _ImpressionSnapshot:
    target: ImpressionTarget | None
    base: Impression | None
    reference: str


def _impression_snapshot(worker: GroupDeliveryWorker, grant: GroupGrant) -> _ImpressionSnapshot:
    target = worker.inner_wiring.group_impression_target(grant)
    if target is None:
        return _ImpressionSnapshot(None, None, "")
    base = read_impression(target)
    if base is not None:
        return _ImpressionSnapshot(target, base, base.text)
    for owner in member_impression_read_owners(grant)[1:]:
        reference = read_impression((target[0], target[1], owner))
        if reference is not None:
            return _ImpressionSnapshot(target, None, reference.text)
    return _ImpressionSnapshot(target, None, "")


def _pending_snapshot(worker: GroupDeliveryWorker) -> list:
    grant = worker.group_delivery_context().grant
    owners = set(grant.resolution_owners) if grant is not None else set()
    if not owners:
        return []
    snapshot = []
    for row in worker.list_commitments(status="pending"):
        if not isinstance(row, Mapping) or row.get("id") is None:
            continue
        owner = str(row.get("owner_key") or "")
        if owner not in owners:
            continue
        snapshot.append({
            "id": int(row["id"]),
            "text": str(row.get("text") or ""),
            "quote": str(row.get("source_quote") or ""),
            "version": int(row["version"]),
            "owner": owner,
        })
    return snapshot


def _model_pending(pending: list) -> list:
    return [
        {key: item[key] for key in ("id", "text", "quote", "version")}
        for item in pending
    ]


def _call_model(worker: GroupDeliveryWorker, grant: GroupGrant, *, persona_name: str,
                user_text: str, assistant_text: str,
                impression_text: str, pending: list, speaker: str,
                diagnostics: list) -> Optional[dict]:
    persona = persona_name.strip() or "我"
    scene = {
        "platform": grant.source.get("platform", ""),
        "chat_type": grant.source.get("chat_type", ""),
        "chat_id": grant.source.get("chat_id", ""),
        "thread_id": grant.source.get("thread_id", ""),
        "parent_chat_id": grant.source.get("parent_chat_id", ""),
        "message_ref": grant.source.get("ref", ""),
        "reply_to_message_id": grant.source.get("reply_to_message_id", ""),
        "time": grant.source.get("time", ""),
        "speaker": speaker,
    }
    user_prompt = (
        "【本轮群成员消息】\n" + user_text + "\n\n"
        "【我这条已送达的回复】\n" + assistant_text + "\n\n"
        "【场景】\n" + json.dumps(scene, ensure_ascii=False) + "\n\n"
        "【我对这位成员在这个群里的印象（旧版，可为空）】\n" + (impression_text or "无") + "\n\n"
        "【这个群当前还没办完的约定】\n"
        + (json.dumps(_model_pending(pending), ensure_ascii=False) if pending else "无") + "\n"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(persona=persona)},
        {"role": "user", "content": user_prompt},
    ]
    raw = auxiliary.complete_simple(
        worker.runtime_llm(), messages, temperature=0.0, max_tokens=900,
        timeout=25.0, purpose="group_ingest",
        failure_message="group ingest auxiliary update failed",
    )
    if not raw:
        diagnostics.append("auxiliary_unavailable")
        return None
    parsed = parse_aux_json(raw if isinstance(raw, str) else raw.text)
    if parsed is None:
        diagnostics.append("invalid_json")
    return parsed


class _WriteGroupQuote(Protocol):
    def __call__(
        self, text: str, session_id: str, /, *, importance: float, owner_key: str,
        source: str, source_context: Mapping[str, object], experience_subject: str,
        fact_subject: str, involved_owner_keys: str, read_scope: str,
    ) -> int | None: ...


def write_group_delivery_event(
    grant: GroupGrant, *, quote: str, summary: str, delivered_at: object, delivery_ref: str,
    paraphrase_ref: str, session_id: str, data_dir: Path, sqlite: SqliteFacade,
    write_quote: _WriteGroupQuote,
    promote_summary: Callable[[int, str, str, bytes, str], int],
    upsert_vector: Callable[[int, str, str, float, str, list[float]], None],
) -> int | None:
    event_id = write_quote(
        quote, session_id, importance=0.6,
        owner_key=grant.write_owner, source="explicit",
        source_context=_processing_source(
            grant, delivery_ref=delivery_ref, paraphrase_ref=paraphrase_ref),
        experience_subject=SELF_SUBJECT,
        fact_subject=f"owner:{grant.person_key}" if grant.person_key else "",
        involved_owner_keys=_involved(grant), read_scope=_group_read_scope(grant),
    )
    if event_id is None:
        return None
    summary_vector = _embed(summary, data_dir=data_dir)
    if not promote_summary(
            int(event_id), summary, event_keywords(summary),
            _pack_vector(summary_vector), _now_iso(delivered_at)):
        logger.warning("sanae_mem: group event %s kept its raw text (summary not distilled)",
                       event_id)
    else:
        row = sqlite.fetchone(
            "SELECT topic, importance, ts FROM events WHERE id=?",
            (int(event_id),),
        )
        if row is not None:
            upsert_vector(
                int(event_id), summary, str(row["topic"] or ""),
                float(row["importance"] or 0.0), str(row["ts"] or ""), summary_vector,
            )
    return int(event_id)


def _write_fulfillment_event(worker: GroupDeliveryWorker, grant: GroupGrant, *, text: str,
                             commitment_id: int, owner: str, delivery_ref: str,
                             session_id: str) -> Optional[int]:
    content = f"我已兑现此前答应的事：{text}"
    record_owner = str(owner or "")
    return worker.ingest.write_event(
        content, session_id, importance=0.6,
        owner_key=record_owner, source="explicit",
        source_context=_processing_source(
            grant, delivery_ref=delivery_ref, commitment_id=str(int(commitment_id))),
        experience_subject=SELF_SUBJECT,
        fact_subject=f"owner:{grant.person_key}" if grant.person_key else "",
        involved_owner_keys=_involved(grant),
        read_scope=json.dumps({"kind": "group", "owner": record_owner},
                              ensure_ascii=False, sort_keys=True),
        fulfillment_commitment_id=int(commitment_id),
    )


def summarize_group_event(
    row: sqlite3.Row,
    now: datetime,
    *,
    current_persona_name: Callable[[], str],
    current_model: Callable[[], AuxiliaryModel | None],
    get_event: Callable[[int, str], sqlite3.Row | None],
    parse_json: Callable[[str], dict[str, object] | None],
    submit_summary: Callable[[int, str, datetime], bool],
) -> Tuple[str, str]:
    event_id = int(row["id"])
    if get_event(event_id, now.isoformat()) is None:
        return "", "write_failed"
    source = parse_source(row["source_context"]) or {}
    material = {key: source[key] for key in _SUMMARY_SOURCE_KEYS if source.get(key)}
    material["ts"] = str(row["ts"] or "")
    persona = current_persona_name().strip() or "我"
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT.format(persona=persona)},
        {"role": "user", "content": (
            "【本条群事件原文】\n" + str(row["raw"] or "") + "\n\n"
            "【来源】\n" + json.dumps(material, ensure_ascii=False)
        )},
    ]
    raw = auxiliary.complete_simple(
        current_model(), messages, temperature=0.0, max_tokens=300,
        timeout=25.0, purpose="group_distill",
        failure_message="group maintenance summary failed",
    )
    if not raw:
        return "", "auxiliary_unavailable"
    parsed = parse_json(raw if isinstance(raw, str) else raw.text)
    if not isinstance(parsed, dict):
        return "", "parse_failed"
    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        return "", "empty_text"
    if not submit_summary(event_id, summary, now):
        return "", "write_failed"
    return summary, ""


def _basis_evidence(action: str, basis: str, quote: str, *, user_text: str,
                    assistant_text: str, usable_tools: Sequence[Mapping[str, Any]]) -> str:
    if action == "cancelled":
        if _quote_in_text(quote, user_text):
            return f"user_cancellation:{quote[:120]}"
        if _quote_in_text(quote, assistant_text):
            return f"delivered_reply_cancellation:{quote[:120]}"
        return ""
    if action != "fulfilled":
        return ""
    if basis == "delivered_reply":
        return f"delivered_reply:{quote[:120]}" if _quote_in_text(quote, assistant_text) else ""
    if basis == "user_confirmed":
        return f"user_confirmation:{quote[:120]}" if _quote_in_text(quote, user_text) else ""
    if basis == "tool_result":
        if not usable_tools:
            return ""
        if not (_quote_in_text(quote, user_text) or _quote_in_text(quote, assistant_text)):
            return ""
        tool = usable_tools[0]
        return f"tool_result:{tool['tool_call_id']}:{tool['summary'][:120]}"
    return ""


def _apply_updates(worker: GroupDeliveryWorker, grant: GroupGrant, updates: Any, *, pending: list,
                   user_text: str, assistant_text: str, transcript: Sequence[Any],
                   turn_id: str, delivery_ref: str, result: dict,
                   session_id: str = "") -> None:
    snapshot = {item["id"]: item for item in pending}
    usable, rejected = _usable_tool_evidence(transcript, turn_id)
    if rejected:
        logger.warning("sanae_mem: group receipt had %d unusable tool results", len(rejected))
    for update in updates or []:
        if not isinstance(update, Mapping):
            continue
        commitment_id = update.get("id")
        action = str(update.get("action") or "")
        basis = str(update.get("basis") or "")
        quote = str(update.get("quote") or "").strip()
        if isinstance(commitment_id, bool) or not isinstance(commitment_id, int):
            continue
        row = snapshot.get(commitment_id)
        if row is None:
            logger.warning(
                "sanae_mem: group receipt update names %r, not this scene's pending promise",
                commitment_id)
            continue
        evidence = _basis_evidence(
            action, basis, quote, user_text=user_text, assistant_text=assistant_text,
            usable_tools=usable)
        if not evidence:
            logger.warning(
                "sanae_mem: group receipt update for %d rejected (%s/%s)",
                commitment_id, action or "no-action", basis or "no-basis")
            continue
        resolved, resolve_reason = worker.resolve_commitment_with_reason(
            commitment_id, action, evidence=evidence,
            expected_version=int(row["version"]),
        )
        if not resolved:
            if resolve_reason == "archive_failed":
                result["ok"] = False
                result["reason"] = result["reason"] or "archive_failed"
            continue
        result["updates"].append({"id": commitment_id, "action": action})
        if action != "fulfilled":
            continue
        try:
            active_session = session_id or worker.group_delivery_context().session_id
            if _write_fulfillment_event(
                    worker, grant, text=str(row["text"] or ""),
                    commitment_id=commitment_id, owner=str(row["owner"]),
                    delivery_ref=delivery_ref, session_id=active_session) is None:
                result["ok"] = False
                result["reason"] = result["reason"] or "event_write_failed"
        except Exception as exc:
            logger.error("sanae_mem: group fulfillment event failed (%s)", type(exc).__name__)
            result["ok"] = False
            result["reason"] = result["reason"] or "event_write_failed"


def apply_group_delivery(
    worker: GroupDeliveryWorker,
    *,
    user_message: Any,
    assistant_response: Any,
    turn_id: str = "",
    delivered_at: Any = None,
    delivery_ref: str = "",
    transcript: Sequence[Any] = (),
    user_message_authored: bool = False,
) -> dict:
    result: dict = {
        "ok": False, "reason": "group_switched_off", "events": [], "impression": None,
        "commitments": [], "updates": [],
    }
    ctx = worker.group_delivery_context()
    grant = ctx.grant
    if grant is None:
        return result
    env = (
        host_bridge.current_turn_envelope()
        if hasattr(host_bridge, "current_turn_envelope")
        else None
    )
    sender_name = str(getattr(env, "sender_name", "") or "").strip() if env is not None else ""
    user_text = authored_group_user_text(
        user_message,
        authored=bool(user_message_authored),
        sender_name=sender_name,
        platform=str(grant.source.get("platform") or ""),
        author_id=str(grant.source.get("author_id") or ""),
    )
    assistant_text = str(assistant_response or "").strip()
    if not user_text or not assistant_text or not grant.person_key:
        result["reason"] = "missing_turn_material"
        return result

    pending = _pending_snapshot(worker)
    impression = _impression_snapshot(worker, grant)
    diagnostics: list = []
    parsed = _call_model(
        worker, grant, persona_name=ctx.persona_name,
        user_text=user_text, assistant_text=assistant_text,
        impression_text=impression.reference, pending=pending,
        speaker=grant.person_key, diagnostics=diagnostics,
    )
    if not isinstance(parsed, dict):
        failure = diagnostics[0] if diagnostics else "invalid_json"
        result["reason"] = failure
        logger.warning(
            "sanae_mem: group ingest %s on delivered turn %s; nothing queued", failure, turn_id)
        return result

    ctx = worker.group_delivery_context()
    grant = ctx.grant
    if grant is None:
        return result
    result["ok"] = True
    result["reason"] = ""
    now = float(delivered_at or datetime.now().timestamp())

    for item in parsed.get("events") or []:
        if not isinstance(item, Mapping):
            continue
        quote = str(item.get("quote") or "").strip()
        summary = str(item.get("text") or "").strip()
        if not quote or not summary or not _quote_in_text(quote, user_text):
            logger.warning("sanae_mem: group event rejected (quote not in this member's text)")
            continue
        paraphrase_ref = _valid_paraphrase_ref(
            grant, item.get("paraphrase_ref"), delivery_ref=delivery_ref)
        event_id = worker.write_group_delivery_event(
            grant, quote=quote, summary=summary,
            delivered_at=delivered_at, delivery_ref=delivery_ref,
            paraphrase_ref=paraphrase_ref)
        if event_id is not None:
            result["events"].append(event_id)

    candidate = parsed.get("impression")
    if isinstance(candidate, Mapping) and impression.target is not None:
        current = worker.group_delivery_context().grant
        target_now = worker.inner_wiring.group_impression_target(current) if current is not None else None
        if target_now is not None and target_now == impression.target:
            if apply_impression(
                    impression.target, impression.base, dict(candidate), turn_id=turn_id,
                    now=now, user=user_text, assistant=assistant_text,
                    source_context=source_json(_processing_source(
                        current, delivery_ref=delivery_ref)),
                    user_authored=True):
                result["impression"] = str(candidate.get("text") or "")
        else:
            logger.warning("sanae_mem: group impression dropped (grant or target changed)")

    for item in parsed.get("new_commitments") or []:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if not text or not quote or not _quote_in_text(quote, assistant_text):
            logger.warning(
                "sanae_mem: group promise rejected (not stated in the delivered reply)")
            continue
        grant = worker.group_delivery_context().grant
        if grant is None:
            break
        commitment_id = worker.record_commitment(
            grant.write_owner, text, source_quote=quote, turn_id=turn_id,
            now=now, masked_text=str(item.get("masked_text") or ""),
            source_context=source_json(_processing_source(
                grant, delivery_ref=delivery_ref)),
        )
        if commitment_id is not None:
            result["commitments"].append(int(commitment_id))

    if parsed.get("commitment_updates"):
        ctx = worker.group_delivery_context()
        grant = ctx.grant
        if grant is None:
            result["ok"] = False
            result["reason"] = "group_switched_off"
            return result
        _apply_updates(
            worker, grant, parsed.get("commitment_updates"), pending=pending,
            user_text=user_text, assistant_text=assistant_text,
            transcript=tuple(transcript or ()),
            turn_id=turn_id, delivery_ref=delivery_ref, result=result,
            session_id=ctx.session_id)
    return result
