"""First-person evidence for USER.md; legacy snapshots stay readable."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from ..infra.chat_scope import ChatScope
from ..infra.contracts import freeze_mapping, freeze_value
from ..infra.host_bridge import current_turn_envelope, is_delegated_child_turn
from ..infra.source_descriptors import is_non_conversation_source
from .source_context import (
    ChatIdentity, current_source_record, is_maintainer_source, parse_source,
)


_NORMALIZATION_RULES: Tuple[Tuple[str, str], ...] = (
    ("的意思就是", "="),
    ("的意思是", "="),
    ("意思是", "="),
    ("代表着", "="),
    ("代表", "="),
    ("就是", "="),
    ("means", "="),
)


def normalize_user_text(text: str) -> str:
    """有限规范化：固定连接词 → "="，再把空白压成单空格。不猜语义、不加词。"""
    out = text or ""
    for old, new in _NORMALIZATION_RULES:
        if old in out:
            out = out.replace(old, new)
    return " ".join(out.split()).strip()


_UNTRUSTED_ORIGINS = frozenset({"background_review"})

_MAX_EVIDENCE_MESSAGES = 16
_MAX_EVIDENCE_CHARS = 6000


@dataclass(frozen=True)
class EvidenceMessage:
    """有界窗口里的一条消息；``index`` 是它在实时会话消息表里的位置（用于判顺序）。"""

    index: int
    role: str
    text: str


@dataclass(frozen=True)
class UserEvidence:
    """宿主构造的可信用户证据。``available=False`` ⇒ 无法证明任何用户亲口内容。

    除 ``available`` 之外还携带**原 build 时的上下文判据**（``origin``/``unattended``/``platform``）
    与**受过滤前的真实源输入**（``source_messages``），好让重放侧能**重跑**
    ``build_user_evidence`` 的拒绝判据（cron 会话前缀／后台 origin／unattended／不可信 platform／
    合成/工具消息），而不是采信这里存档的 ``available`` 布尔标签。
    """

    available: bool
    session: str = ""
    user: str = ""
    turn: str = ""
    total_messages: int = 0
    messages: Tuple[EvidenceMessage, ...] = ()
    reason: str = ""
    origin: str = ""
    unattended: bool = False
    platform: str = ""
    source_messages: Tuple[Mapping[str, Any], ...] = field(default=(), repr=False, compare=False)
    source_context: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_messages",
            tuple(freeze_value(msg) for msg in (self.source_messages or ())),
        )
        if self.source_context is not None:
            object.__setattr__(self, "source_context", freeze_mapping(self.source_context))

    @classmethod
    def unavailable(cls, reason: str, *, session: str = "", user: str = "", turn: str = "") -> "UserEvidence":
        return cls(False, session, user, turn, 0, (), reason)

    @property
    def user_messages(self) -> List[EvidenceMessage]:
        return [m for m in self.messages if m.role == "user"]


def capture_private_turn(
    text: str, *, current_message: dict[str, object] | None, session_id: str,
    platform: str, scope: ChatScope, identity: ChatIdentity, author_id: str,
) -> Optional[UserEvidence]:
    """Bind original turn text to the actual private gateway message and resolved identity."""
    envelope = current_turn_envelope()
    if (not identity.private_owner_key or not identity.maintainer or scope.platform != platform
            or not session_id or scope.session_id != session_id
            or not scope.session_key or envelope is None
            or str(getattr(envelope, "session_key", "") or "") != scope.session_key
            or is_delegated_child_turn()
            or str(getattr(envelope, "write_origin", "") or "") != "foreground"):
        return None
    if author_id and author_id not in (scope.actor, scope.actor_alt):
        return None
    if not isinstance(text, str) or not text.strip() or len(text) > _MAX_EVIDENCE_CHARS:
        return None
    if (not isinstance(current_message, dict) or current_message.get("role") != "user"
            or current_message.get("display_kind") or current_message.get("tool_call_id")
            or current_message.get("content") != text):
        return None
    # Hermes stamps the persisted turn with the actual inbound id, even for a queued
    # follow-up whose route still carries the opening message's id.
    message_id = str(current_message.get("platform_message_id") or "").strip()
    source = current_source_record(maintainer=True, scope=scope)
    if not message_id or source is None:
        return None
    source["ref"] = message_id
    source["message_id"] = message_id
    evidence = build_user_evidence(
        [{"role": "user", "content": text}], session=session_id,
        user=scope.actor_alt or scope.actor, platform=platform, origin="foreground",
        turn=message_id, source_context=source,
    )
    return evidence if evidence.available else None


def _resolve_summary_predicate():
    """Recognize the host's durable summary markers without importing its compressor internals."""
    def is_summary(message: Any) -> bool:
        if not isinstance(message, dict):
            return False
        if (message.get("_compressed_summary")
                or message.get("display_kind") in {"hidden", "failed_turn", "summary"}):
            return True
        text = _message_text(message.get("content")).lstrip()
        if text.startswith(("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]")):
            return True
        delimiter = "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]"
        if delimiter in text:
            tail = text.split(delimiter, 1)[1].lstrip()
            return tail.startswith(("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]"))
        return False

    return is_summary


def _message_text(content: Any) -> str:
    """纯文本视图；列表型 content 只取 text 部件，不拼 tool/image 载荷。"""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = [p if isinstance(p, str) else p.get("text", "")
                 for p in content
                 if isinstance(p, str) or (isinstance(p, Mapping) and p.get("type") in ("text", "input_text"))]
        return "\n".join(part for part in parts if isinstance(part, str))
    return ""


def build_user_evidence(
    messages: Any,
    *,
    session: str = "",
    user: str = "",
    platform: str = "",
    origin: str = "foreground",
    unattended: bool = False,
    turn: str = "",
    max_messages: int = _MAX_EVIDENCE_MESSAGES,
    max_chars: int = _MAX_EVIDENCE_CHARS,
    is_synthetic=None,
    source_context: Optional[Mapping[str, str]] = None,
) -> UserEvidence:
    """From the live conversation messages, build a bounded, host-trusted user-source window.

    Only ``role == "user"`` rows count as the user's own words, and only when the row is not a
    tool result and not a compaction summary. Assistant rows are kept solely so the host can check
    that a candidate was presented before a confirmation; they are never evidence of a user fact.
    A ``cron_``-prefixed ``session`` is refused outright: the per-run session id is the stable
    evidence of a cron turn, while the instance ``platform`` may be empty or lag behind (雷区43).
    """
    total = len(messages) if isinstance(messages, list) else 0
    # 原 build 时的上下文判据，随证据一起保留（重放据此重跑判据，不信存档 available 布尔）。
    if source_context is not None:
        raw_kind = str(source_context.get("material_kind") or "") if isinstance(source_context, Mapping) else ""
        if raw_kind == "processed":
            return UserEvidence.unavailable("processed material is never verbatim user speech", session=session, user=user, turn=turn)
        source_context = parse_source(source_context)
        if (
            not source_context
            or source_context.get("certainty") == "derived_processed"
            or str(source_context.get("kind") or "").startswith("external_")
            or not is_maintainer_source(source_context)
        ):
            return UserEvidence.unavailable("source is not the maintainer", session=session, user=user, turn=turn)
    _ctx_flags = {"origin": str(origin or ""), "unattended": bool(unattended),
                  "platform": str(platform or ""), "source_context": source_context}

    def _refuse(reason: str) -> UserEvidence:
        return UserEvidence(False, session, user, turn, total, (), reason, **_ctx_flags)

    if source_context is not None and (
        source_context["kind"] != "maintainer" or not user
        or source_context["author_id"] != user or source_context["visibility"] == "public"
        or (source_context["platform"] and source_context["platform"] != platform)
    ):
        return _refuse("source actor has no maintainer confirmation authority")

    if unattended or (origin or "foreground") in _UNTRUSTED_ORIGINS:
        return _refuse("unattended/background review source is not user speech")
    # 雷区43：cron 的稳定证据是**当次 session_id 的 ``cron_`` 前缀**，不是实例 platform
    # （可能为空或滞后）。真实 cron 运行由 cron/scheduler 构造为 ``cron_<job>_<ts>``，故此处
    # 决断；反之普通会话即使 platform 滞后为 'cron' 也不得被误伤（platform 只作佐证）。
    if str(session or "").strip().startswith("cron_"):
        return _refuse("a cron session's text is not an interactive user turn")
    plat = (platform or "").strip().lower()
    if is_non_conversation_source(plat):
        return _refuse(f"session source '{plat}' is not an interactive user turn")
    if not isinstance(messages, list) or not messages:
        return _refuse("no conversation messages available")

    if is_synthetic is None:
        try:
            is_synthetic = _resolve_summary_predicate()
        except Exception as e:  # cannot prove a row is not a summary ⇒ do not trust any of them
            return _refuse(f"summary guard unavailable: {type(e).__name__}: {e}")

    kept: List[EvidenceMessage] = []
    budget = max_chars
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        if msg.get("tool_call_id"):
            continue
        if role == "user":
            if msg.get("_material_kind") == "processed" or msg.get("material_kind") == "processed":
                continue
            meta = msg.get("display_metadata")
            row_source = msg.get("_source_context", meta.get("source_context") if isinstance(meta, dict) else None)
            if row_source is not None:
                if isinstance(row_source, dict) and str(row_source.get("material_kind") or "") == "processed":
                    continue
                row_source = parse_source(row_source)
                if (not row_source or row_source.get("certainty") == "derived_processed"
                        or str(row_source.get("kind") or "").startswith("external_")
                        or not is_maintainer_source(row_source) or row_source["author_id"] != user
                        or row_source["visibility"] == "public"
                        or (row_source["platform"] and row_source["platform"] != platform)):
                    continue
            author = msg.get("turn_author")
            if isinstance(author, dict) and (author.get("is_bot") or (
                author.get("id") and user and str(author["id"]) != user
            )):
                continue
        text = _message_text(msg.get("content"))
        if not text.strip() or len(text) > max_chars:
            continue
        try:
            if is_synthetic(msg):
                continue
        except Exception:
            continue
        kept.append(EvidenceMessage(idx, role, text))
        budget -= len(text)
        if len(kept) >= max_messages or budget <= 0:
            break
    kept.reverse()
    if not any(m.role == "user" for m in kept):
        return _refuse("no user-authored message in the trusted window")
    return UserEvidence(True, session, user, turn, total, tuple(kept), "", **_ctx_flags,
                        source_messages=tuple(messages) if isinstance(messages, list) else ())


SOURCE_SNAPSHOT_VERSION = 2


def _json_safe(value: Any) -> Any:
    """把任意值投影成 JSON 可序列化形态（保留 dict/list/标量，其余转 str），只用于快照落库。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def snapshot_source_evidence(evidence: Optional["UserEvidence"]) -> Dict[str, Any]:
    """把宿主可信来源序列化成**可重跑**的规范化快照（仅后台字段，不进自然正文）。

    ``available`` 只作诊断；重放侧以 `restore_source_evidence` 重跑 `build_user_evidence` 为准。
    ``evidence=None`` ⇒ 返回空的拒绝快照（重放据此拒给 grounded 资格）。
    """
    if evidence is None:
        return {"version": SOURCE_SNAPSHOT_VERSION, "available": False,
                "session": "", "user": "", "turn": "", "total_messages": 0,
                "origin": "", "unattended": False, "platform": "",
                "reason": "no host-built source context", "source_messages": []}
    return {
        "version": SOURCE_SNAPSHOT_VERSION,
        "available": bool(getattr(evidence, "available", False)),  # 仅诊断
        "session": str(getattr(evidence, "session", "") or ""),
        "user": str(getattr(evidence, "user", "") or ""),
        "turn": str(getattr(evidence, "turn", "") or ""),
        "total_messages": int(getattr(evidence, "total_messages", 0) or 0),
        "origin": str(getattr(evidence, "origin", "") or ""),
        "unattended": bool(getattr(evidence, "unattended", False)),
        "platform": str(getattr(evidence, "platform", "") or ""),
        "source_context": _json_safe(getattr(evidence, "source_context", None)),
        "reason": str(getattr(evidence, "reason", "") or ""),
        "source_messages": [_json_safe(m) for m in (getattr(evidence, "source_messages", ()) or ())],
    }


def restore_source_evidence(snapshot: Dict[str, Any]) -> Optional["UserEvidence"]:
    """从持久快照**重跑** `build_user_evidence`；拒绝 ⇒ 返回 None。

    不采信存档的 ``available`` 布尔：手工 `UserEvidence(True...)`（无原始源输入）／cron 会话前缀／
    后台 origin／unattended／不可信 platform／合成或工具消息，都在重跑 `build_user_evidence` 时被拒。
    """
    if not isinstance(snapshot, dict):
        return None
    raw = snapshot.get("source_messages")
    if not isinstance(raw, list):
        return None
    evidence = build_user_evidence(
        raw,
        session=str(snapshot.get("session") or ""),
        user=str(snapshot.get("user") or ""),
        platform=str(snapshot.get("platform") or ""),
        origin=str(snapshot.get("origin") or ""),
        unattended=bool(snapshot.get("unattended") or False),
        turn=str(snapshot.get("turn") or ""),
        source_context=snapshot.get("source_context"),
    )
    if not evidence.available:
        return None
    return evidence


USER_EXTRACTED = "USER_EXTRACTED"
USER_UNPROVEN = "USER_UNPROVEN"      # A2：有原文，但周边框架不在有限正向许可内
MODEL_INFERENCE = "MODEL_INFERENCE"


@dataclass(frozen=True)
class SourceVerdict:
    """来源文字与肯定资格分开记（A2）：``extracted``＝正文确实在某条用户消息里；
    ``affirmed``＝那条消息里它的周边框架**能证明**是肯定。``allowed`` 只在两者都为真时为真。"""

    allowed: bool
    status: str
    reason: str = ""
    extracted: bool = False
    affirmed: bool = False


# 保守分句：只按**强标点与换行**切句（含英文句点）。分句让「不是 X，是 Y」这类纠正里，
# 被否定的旧值与新说的值落在不同句子里，可以分开判；不按空格/词切，也不做词性分析。
# 问号不在这里：切掉它会把「X = habit?」变成肯定句。
_CLAUSE_DELIMITERS = "。！!；;，,\n\r."

# 周边框架只允许第一人称、语气填充和肯定包装。时间、频率、否定、
# 疑问和引用的标记不能省略，否则抽取的片段可能改变原话。
_POSITIVE_FRAME_TOKENS: Tuple[str, ...] = (
    "我们", "咱们", "本人", "我",
    "其实", "对了", "另外", "还有", "嗯", "哦", "啊", "呃", "那个",
    "而是", "才是", "是对的", "才对", "正确", "没错", "对的", "是",
    "帮我记住", "你要记住", "你记着", "记一下", "记住", "记下", "记得", "备注一下", "备注",
    "is correct", "are correct", "was correct", "correct", "is", "are", "was", "were",
)
_POSITIVE_FRAME_TOKENS_ORDERED: Tuple[str, ...] = tuple(
    sorted(_POSITIVE_FRAME_TOKENS, key=len, reverse=True))

# 框架里允许剥掉的**非语义**标点/空白（只在判框架时剥掉，不改正文；`=` 也在这里剥）。
# 刻意**不含**：引号类与块引用 `>`（**引用不是肯定**）、问号（**疑问不是肯定**）——
# 这些都是语义标记，剥掉就等于制造肯定。
_FRAME_PUNCTUATION = " \t\r\n，。．.,!！；;：:、（）()【】[]《》~～—_*＝="


def _frame_is_permitted(residual: str) -> bool:
    """框架必须**能被正向词表逐段吃干净**；吃不动任何一段 ⇒ 不算肯定（默认拒绝）。"""
    text = "".join(ch for ch in residual if ch not in _FRAME_PUNCTUATION).lower()
    i = 0
    while i < len(text):
        for token in _POSITIVE_FRAME_TOKENS_ORDERED:
            if text.startswith(token, i):
                i += len(token)
                break
        else:
            return False
    return True


def _clause_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _CLAUSE_DELIMITERS:
            spans.append((start, i))
            start = i + 1
    spans.append((start, len(text)))
    return spans


def _permitted_frame(message: str, body: str) -> Optional[Tuple[bool, str]]:
    """正文**最后一次**出现在这条消息里时，它的周边框架是否落在有限正向许可内。

    ``None`` ＝ 这条消息根本没有该正文；否则 ``(是否许可, 框架原文或 "")``。
    取最后一次出现＝同一句里「先说 X 对、后说 X 不对」以最后的为准。正文跨句时，逐句
    扣掉正文覆盖的那一段再判剩余框架（用户自己那整句仍算一条完整抽取）。
    """
    pos = message.rfind(body)
    if pos < 0:
        return None
    # 疑问句不是事实陈述：正文本身带着问号（整句照抄一个问句）也不算肯定。
    if body.rstrip().endswith(("?", "？")):
        return (False, body.rstrip()[-1])
    # An exact quotation is still a quotation when the proposed body includes its delimiters.
    if body.lstrip().startswith((">", "「", "『", "“", "‘", '"')):
        return (False, body.lstrip()[0])
    end = pos + len(body)
    for start, stop in _clause_spans(message):
        if stop <= pos or start >= end:
            continue
        covered = message[max(pos, start):min(end, stop)]
        residual = message[start:stop].replace(covered, "", 1)
        if not _frame_is_permitted(residual):
            return (False, residual.strip())
    return (True, "")


_DISCORD_TRIGGER_NOTE_RE = re.compile(r"^\[Triggering message id: [^\n\]]+\](?:\n\n|\s*)")
_REPLY_ID_BLOCK_RE = re.compile(
    r"^\[Replying to(?: your previous message)?: message ID [^\]\n]+\](?:\n\n|\n|$)"
)
_REPLY_QUOTED_OPEN_RE = re.compile(r'^\[Replying to(?: your previous message)?: "')
_QQ_QUOTED_LEN_RE = re.compile(r"^\[Quoted message; length=(\d+)\]:\n")


def strip_inbound_quote_headers(text: str) -> str:
    """确定性剥离宿主拼入的回复/引用头，仅保留用户本轮真实写下的正文。

    严格基于宿主固定语法标记（`[Triggering message id: ...]`、`[Replying to ...]`、
    `[Quoted message; length=N]:`）、长度前缀与引号/括号配对结构，不做任何词汇匹配。
    """
    current = str(text or "").lstrip("\r\n")
    for _ in range(8):
        previous = current
        m_trigger = _DISCORD_TRIGGER_NOTE_RE.match(current)
        if m_trigger:
            current = current[m_trigger.end():].lstrip("\r\n")
            continue

        m_reply_id = _REPLY_ID_BLOCK_RE.match(current)
        if m_reply_id:
            current = current[m_reply_id.end():].lstrip("\r\n")
            continue

        m_reply_quote = _REPLY_QUOTED_OPEN_RE.match(current)
        if m_reply_quote:
            rest = current[m_reply_quote.end():]
            close_double = rest.rfind('"]\n\n')
            if close_double != -1:
                current = rest[close_double + 4:].lstrip("\r\n")
                continue
            close_single = rest.rfind('"]\n')
            if close_single != -1:
                current = rest[close_single + 3:].lstrip("\r\n")
                continue
            if rest.endswith('"]'):
                return ""
            # 未闭合的引用块可能已被截断，整段不视作用户亲口正文。
            return ""

        m_qq_len = _QQ_QUOTED_LEN_RE.match(current)
        if m_qq_len:
            end = m_qq_len.end() + int(m_qq_len.group(1))
            if end >= len(current) or current[end:end + 2] != "\n\n":
                return ""
            current = current[end + 2:].lstrip("\r\n")
            continue

        if current.startswith("[Quoted message]:"):
            return ""

        if current == previous:
            break
    return current


def qualify_user_text(content: str, evidence: Optional[UserEvidence]) -> SourceVerdict:
    """整条正文既要是可信用户消息的合法抽取，也要是用户**肯定**的陈述。

    抽取资格（来源文字）＝规范化后正文整体落在某条可信用户消息里（⇒「原话＋追加推断/
    第二个映射」这类超串不放行）。肯定资格＝**最新一条提到它的**用户消息里，它的周边
    框架落在 `_POSITIVE_FRAME_TOKENS` 的有限正向许可内；旧的肯定被后来的否定覆盖，证不了
    就当没肯定（⇒ 暂存待确认）。两者分开记在 verdict 上。
    """
    body = normalize_user_text(content or "")
    if not body:
        return SourceVerdict(False, MODEL_INFERENCE, "empty content")
    if evidence is None or not evidence.available:
        reason = evidence.reason if evidence is not None else "no source context"
        return SourceVerdict(False, MODEL_INFERENCE, f"no trusted user source ({reason})")
    # 从最新一条用户消息往前找第一条**提到**该正文的：它说了算。
    for msg in reversed(evidence.user_messages):
        authored_text = strip_inbound_quote_headers(msg.text)
        source = normalize_user_text(authored_text)
        if not source:
            continue
        found = _permitted_frame(source, body)
        if found is None:
            continue
        permitted, frame = found
        if permitted:
            return SourceVerdict(True, USER_EXTRACTED,
                                 f"a complete, affirmatively-framed statement in user message #{msg.index}",
                                 extracted=True, affirmed=True)
        return SourceVerdict(
            False, USER_UNPROVEN,
            f"the text occurs in user message #{msg.index} but its surrounding frame "
            f"({frame!r}) is not in the host's positive-permission list, so the host cannot prove "
            f"the user stated it as a fact", extracted=True, affirmed=False)
    return SourceVerdict(False, MODEL_INFERENCE,
                         "the whole entry is not a legal extraction of a user-authored message")
