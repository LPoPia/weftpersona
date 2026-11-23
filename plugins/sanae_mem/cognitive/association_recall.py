"""Association recall, candidate review, and context rendering."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from time import monotonic
from typing import Any, Callable, List, Optional, Protocol, Tuple

from ..infra import auxiliary
from .association_history import (
    aux_history_messages,
)
from ..infra.clock import sleep_day_iso
from ..storage.event_store import gist_provenance
from ..infra.source_descriptors import source_label, video_origin
from ..governance.shared_memory import SHARED_TOPIC_LABEL, is_other_person_source
from ..governance.source_context import (
    parse_source, public_source_view, source_view,
)
from ..infra.textutil import _tokens
from ..infra.words import _EXPLICIT_INJECT_MIN_SEMANTIC, _INJECTION_RECALL_WINDOW_DAYS


_RELATIONSHIPS = frozenset({"same_event", "related"})
_STATES = frozenset({"history", "ongoing", "ended", "unknown"})
_ASSOCIATION_HEADER = "## 小事记忆\n"
_ASSOCIATION_GAP_HEADER = "## 小事记忆\n## 上次聊到的（自然接话线索，仅作参考，可不提）\n"
_ASSOCIATION_PUBLIC_HEADER = "## 小事记忆\n"
_DAILY_SOURCE_KINDS = frozenset({"statement"})

_RETREAT_SELF_QUOTE_RE = re.compile(
    r"(?:(?:这句|这话|刚才那句)?我?(?:先)?(?:收回|当没说|算我说错)"
    r"|我(?:先)?不(?:跟你|同你)?(?:争|辩|吵|计较|较真|理论)"
    r"|我先不(?:动|改|提|问|碰)"
    r"|我(?:又)?(?:没|不)(?:踩|贬低|否定|怪|凶)(?:我)?自己"
    r"|(?:算|当)成[「『“\"][^」』”\"]{1,8}[」』”\"]"
    r"|(?:不用|不必|别)(?:自己)?(?:一个人|独自|为了(?:把)?我)"
    r"|(?:旧|原)(?:内容|文本|记录)被(?:替换|覆盖|改写|删))"
)
_HER_RETREAT_TAIL_RE = re.compile(
    r"[，；]\s*(?:我便没再追问|我就没再追问|我没再追问|我没有再追问|我便没有再追问|我便不再追问)(?=[。.]?$)"
)

_SYSTEM = (
    "你在核对候选事件是否真的支撑一次情境联想，并在主动生成/发送前判断是否安全。"
    "只能依据给出的候选 summary（整理后的展示文本，联想正文会用它）、raw（原始证据）、"
    "时间、来源、既有补充记录和用途；summary 与 raw 不一致时，summary 是当前理解，"
    "raw 只是历史原话，不能把旧原文当成现状；没有 summary 时以 raw 为准。"
    "不得新增候选 id，不得把仅相似的事说成同一件。"
    "source_kind 只是提取类型，不是作者或真实性的证明；quote 是证据原话。"
    "区分本人陈述、转述/引用、玩闹：引用和玩闹不能当客观事实，互动经历可保留但不要把玩闹当现实。"
    "历史不代表现状：旧的工作/考试/进行状态不能当成今天的现状；daily 的 active/closed/期限只是记录生命周期，不是现实当前状态。"
    "主动交流不要复问已聊过或已结束的事项。"
    "同一事件=同一件事的继续/同一作品的不同版本/同一 thread 的新进展；"
    "相关=主题或情境相连但不是同一件；仅相似=只是像但没有证据相连，应丢弃。"
    "状态只能是 history|ongoing|ended|unknown：已发生过/已聊过=history，"
    "明确收束=ended，不能判断=unknown；不因没有后续就认定事情仍未结束。"
    "只有候选内容本身显示这件事现在仍在延续（明确的进行中计划、尚未完成的当前约束，"
    "且没有收束证据）才判 ongoing；过去写过“还没完成”但没有后续证据的旧事只能取 history "
    "或 unknown，不能当成今天仍未完成。time 是记录时间，可与 now 比较先后。"
    "顶层 valid 必须是布尔：草稿要复问已聊过/已结束事项、或把昨日状态当今天 ⇒ false；"
    "正常没有相关或可安全提及 ⇒ true。items 仍逐条给出。"
    "只输出 JSON："
    '{"valid": true, "items":[{"id":1,"relationship":"same_event|related|drop",'
    '"state":"history|ongoing|ended|unknown"}]}'
)


def assistant_statement(item: dict) -> bool:
    source = item.get("source_context") or item.get("source") or {}
    return (item.get("attribution") == "self" and source.get("kind") == "self"
            and str(source.get("ref") or "").startswith("assistant-statement:"))


def strip_her_retreat_tail(text: str) -> str:
    return _HER_RETREAT_TAIL_RE.sub("", text or "")


def clip(body: str, budget: Optional[int]) -> str:
    if budget is None:
        return body
    return body[:max(0, int(budget))]


def review_source(source: Any) -> dict:
    record = source if isinstance(source, dict) else {}
    return {key: str(record.get(key) or "")
            for key in ("kind", "platform", "certainty", "visibility")
            if record.get(key)}


_LONG_TIERS = frozenset({"long", "important", "distilled"})


def prefers_summary(item: dict) -> bool:
    tier = str(item.get("tier") or "")
    source = str(item.get("summary_source") or "")
    return tier in _LONG_TIERS or (bool(source) and source != "original")


_SUPPLEMENT_RE = re.compile(r"\[补充\s+([^\]]+)\]\s*")
_ANCHOR_RE = re.compile(r"^\[20\d\d-\d\d-\d\d\s+\d\d:\d\d\|[^\]]+\]\s*")


def timeline_entries(item: dict) -> List[Tuple[str, str]]:
    raw = _ANCHOR_RE.sub("", str(item.get("text") or "")).strip()
    entries: List[Tuple[str, str]] = []
    matches = list(_SUPPLEMENT_RE.finditer(raw))
    if matches:
        first = raw[:matches[0].start()].strip()
        if first:
            entries.append((str(item.get("time") or ""), first))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            body = raw[match.end():end].strip()
            if body:
                entries.append((match.group(1).strip(), body))
    elif raw:
        entries.append((str(item.get("time") or ""), raw))
    seen = {body for _when, body in entries}
    for occurrence in item.get("occurrences") or []:
        body = str(occurrence.get("text") or "").strip()
        if body and body not in seen:
            entries.append((str(occurrence.get("time") or ""), body))
            seen.add(body)
    entries.sort(key=lambda pair: time_key(pair[0]))
    return entries


def entry_text(entry: Tuple[str, str]) -> str:
    when, body = entry
    label = ""
    try:
        label = relative(when)
    except Exception:
        label = ""
    return ("%s：" % label if label else "") + body


def timeline(item: dict) -> str:
    entries = timeline_entries(item)
    if len(entries) == 1:
        return entries[0][1]
    return "；".join(entry_text(entry) for entry in entries)


def compact_timeline(item: dict, lead: str, suffix: str, budget: int) -> str:
    entries = timeline_entries(item)
    fixed = lead + suffix
    room = int(budget) - len(fixed)
    if room < 8:
        suffix = ""
        fixed = lead
        room = int(budget) - len(fixed)
    if not entries or room < 8:
        return clip(fixed + (entries[-1][1] if entries else ""), budget)

    latest = entry_text(entries[-1])
    if len(latest) > room:
        latest = latest[:room]
    selected = [latest]
    used = len(latest)
    for entry in reversed(entries[:-1]):
        part = entry_text(entry)
        if used + 1 + len(part) <= room:
            selected.insert(0, part)
            used += 1 + len(part)
    return clip(lead + "；".join(selected) + suffix, budget)


def time_key(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed.timestamp()
    except (TypeError, ValueError):
        return float("inf")


def conservative_item(item: dict) -> dict:
    if item.get("verified_same_event"):
        return dict(item, relationship="same_event", state=str(item.get("state") or "unknown"))
    state = str(item.get("state") or "unknown")
    if state not in {"history", "unknown"}:
        state = "unknown"
    return dict(item, relationship="related", state=state)


def conservative(items: List[dict]) -> List[dict]:
    return [conservative_item(item) for item in items]


def relative(value: str) -> str:
    if not value:
        return ""
    created = datetime.fromisoformat(value)
    now = datetime.now().astimezone()
    if created.tzinfo is None:
        created = created.replace(tzinfo=now.tzinfo)
    days = max(0, (now.date() - created.astimezone(now.tzinfo).date()).days)
    if days == 0:
        return "今天"
    if days == 1:
        return "昨天"
    if days < 7:
        return "%d天前" % days
    return created.strftime("%m月%d日")


def strip_lead(text: str) -> str:
    text = re.sub(r"^\[[^\]]+\]\s*", "", str(text or ""))
    text = re.sub(r"^(?:我想|我想说|我|她|他)\s*", "", text)
    return text.strip()[:70]


def parse_review(text: str) -> Tuple[List[dict], Optional[bool], bool]:
    from ..infra.model_output import _load_braced_json

    raw = str(text or "")
    try:
        value = _load_braced_json(raw)
    except (TypeError, ValueError):
        return [], None, False
    if not isinstance(value, dict):
        return [], None, False
    valid = value.get("valid") if isinstance(value.get("valid"), bool) else None
    items = value.get("items")
    out = []
    for entry in items if isinstance(items, list) else []:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("id")
        if isinstance(identifier, bool) or identifier is None:
            continue
        if not isinstance(identifier, (int, str)):
            continue
        if isinstance(identifier, str) and not identifier.strip():
            continue
        out.append({"id": identifier,
                    "relationship": str(entry.get("relationship") or ""),
                    "state": str(entry.get("state") or "")})
    return out, valid, True










class AssociationSearch(Protocol):
    def __call__(
        self, args: dict[str, Any], /, *, for_injection: bool, explicit_history: bool,
        with_relevance: bool, with_source: bool, with_recall_meta: bool,
    ) -> list[dict]: ...


class DailyRecallReader(Protocol):
    def __call__(self, *, owner_key: str, profile_key: str, since: str) -> list[sqlite3.Row]: ...


def retrieve(
    search_query: str,
    *,
    gap_continuation: bool,
    proactive: bool,
    search: AssociationSearch,
    list_daily_entries_for_recall: DailyRecallReader,
    association_read_owner: Callable[[], str],
    daily_owner_key: str,
    daily_profile_key: str,
    daily_owner_platform: str,
    agent_context: str,
    exclude_ids: Optional[set] = None,
    explicit_history: bool = False,
) -> List[dict]:
    excluded = exclude_ids or set()
    raw_items = search(
        {"query": search_query, "top_k": 10,
         **({"gap_continue": True} if gap_continuation else {})},
        for_injection=bool(gap_continuation), explicit_history=explicit_history,
        with_relevance=True, with_source=True, with_recall_meta=True,
    )

    event_embedding_real = any(item.get("embedding_real", False) for item in raw_items)
    if not gap_continuation and not proactive and not event_embedding_real:
        raw_items = [item for item in raw_items if assistant_statement(item)
                     or item.get("shared")
                     or (explicit_history and item.get("tier") == "dormant")]

    query_tokens = set(_tokens(search_query))
    items: List[dict] = []
    for item in raw_items:
        if item.get("id") in excluded:
            continue
        if assistant_statement(item) and _RETREAT_SELF_QUOTE_RE.search(str(item.get("raw") or "")):
            continue
        relationship, credible = association_relationship(item, query_tokens, top=not items)
        if not credible:
            continue
        items.append(make_association_item(item, relationship, public=False))
        if len(items) >= 3:
            break

    daily_items = daily_items_for_recall(
        search_query, exclude_ids=excluded,
        list_daily_entries_for_recall=list_daily_entries_for_recall,
        association_read_owner=association_read_owner,
        daily_owner_key=daily_owner_key, daily_profile_key=daily_profile_key,
        daily_owner_platform=daily_owner_platform, agent_context=agent_context,
    )
    if daily_items:
        items.extend(daily_items)
        items = [item for item in items if item["id"] not in excluded]
        items.sort(key=lambda item: float(item.get("relevance") or 0.0), reverse=True)
        items = items[:4]

    if gap_continuation:
        items = items[:1]
    return items


def daily_items_for_recall(
    search_query: str,
    *,
    list_daily_entries_for_recall: DailyRecallReader,
    association_read_owner: Callable[[], str],
    daily_owner_key: str,
    daily_profile_key: str,
    daily_owner_platform: str,
    agent_context: str,
    exclude_ids: Optional[set] = None,
) -> List[dict]:
    excluded = exclude_ids or set()
    owner = str(daily_owner_key or "")
    profile = str(daily_profile_key or "")
    if not (owner and profile):
        return []
    try:
        read_owner = association_read_owner()
    except PermissionError:
        return []
    if str(read_owner or "") != owner:
        if not (agent_context == "cron" and not str(read_owner or "")):
            return []
    from .daily_state import _view_noise as daily_view_noise
    since = (datetime.now().astimezone()
             - timedelta(days=_INJECTION_RECALL_WINDOW_DAYS)).isoformat()
    rows = list_daily_entries_for_recall(owner_key=owner, profile_key=profile, since=since)
    query_tokens = set(_tokens(search_query))
    if not query_tokens:
        return []
    today = sleep_day_iso()
    items: List[dict] = []
    for row in rows:
        if str(row["kind"] or "") == "OPEN_THREAD":
            continue
        if str(row["source_kind"] or "") not in _DAILY_SOURCE_KINDS:
            continue
        text = str(row["text"] or "").strip()
        quote = str(row["source_quote"] or "").strip()
        if not text or not quote:
            continue
        if daily_view_noise(text) or daily_view_noise(quote):
            continue
        overlap = len(query_tokens & set(_tokens(text + " " + quote)))
        if overlap < 2:
            continue
        opened = str(row["opened_on"] or "")
        status = str(row["status"] or "")
        if opened and opened < today or status in {"closed", "expired"}:
            state = "history"
        else:
            state = "unknown"
        entry_id = int(row["entry_id"])
        if "daily:%d" % entry_id in excluded:
            continue
        when = str(row["created_at"] or opened or "")
        author_kind = str(row["author_kind"] or "unknown")
        maintainer = author_kind == "maintainer"
        fact_subject = f"owner:{owner}" if maintainer else ""
        items.append({
            "id": "daily:%d" % entry_id,
            "entry_id": entry_id,
            "source_table": "daily",
            "time": when,
            "opened_on": opened,
            "expires_on": str(row["expires_on"] or ""),
            "source": {
                "kind": "maintainer" if maintainer else "unknown",
                "author_id": "",
                "platform": daily_owner_platform,
                "ref": "daily:%d" % entry_id,
                "time": when,
                "certainty": "observed" if maintainer else "unverified",
                "visibility": "private",
            },
            "relationship": "related",
            "state": state,
            "text": text,
            "summary": text,
            "topic": "",
            "experience_subject": f"owner:{owner}" if maintainer else "",
            "fact_subject": fact_subject,
            "attribution": "user_fact" if maintainer else "unknown",
            "occurrences": [],
            "asked_at": "",
            "discussed": False,
            "daily_kind": str(row["kind"] or ""),
            "slot": str(row["slot"] or ""),
            "status": status,
            "source_kind": str(row["source_kind"] or ""),
            "source_turn": str(row["source_turn"] or ""),
            "quote": quote,
            "relevance": overlap / max(1, len(query_tokens)),
        })
    items.sort(key=lambda item: float(item.get("relevance") or 0.0), reverse=True)
    return items[:2]


def association_relationship(item: dict, query_tokens: set, *, top: bool) -> Tuple[str, bool]:
    if item.get("shared"):
        text = str(item.get("shared_view") or "")
    else:
        raw = re.sub(r"^\[[^\]]+\]\s*", "", str(item.get("raw") or ""))
        summary = str(item.get("summary") or "")
        text = " ".join([raw[:240], summary[:240],
                         str(item.get("keywords") or "")])
        for occurrence in item.get("occurrences") or []:
            text += " " + str(occurrence.get("text") or "")[:120]
        if assistant_statement(item):
            text = raw.removeprefix("我说过：")[:240]
    overlap_count = len(query_tokens & set(_tokens(text)))
    semantic = float(item.get("semantic") or 0.0)
    real = bool(item.get("embedding_real"))

    if real:
        credible = (overlap_count >= 2 or semantic >= _EXPLICIT_INJECT_MIN_SEMANTIC)
    else:
        credible = overlap_count >= 2
    if not credible:
        return "related", False
    return "related", True


def make_association_item(item: dict, relationship: str, *, public: bool) -> dict:
    shared = bool(item.get("shared")) or item.get("attribution") == "shared_experience"
    source = item.get("source_context")
    if not isinstance(source, dict):
        source = public_source_view(None) if public else source_view(parse_source(None))
    raw_id = item["id"]
    event_id = raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) else 0
    return {
        "id": raw_id,
        "event_id": event_id,
        "source_table": str(item.get("source_table") or "event"),
        "time": str(item.get("created_at") or item.get("ts") or ""),
        "source": source if not shared else {},
        "relationship": relationship,
        "_shared_view": shared,
        "state": association_state(item),
        "text": str(item.get("shared_view") or "") if shared
                else str(item.get("raw") or item.get("summary") or ""),
        "raw": "" if shared else str(item.get("raw") or ""),
        "summary": "" if shared else str(item.get("summary") or ""),
        "tier": str(item.get("tier") or ""),
        "summary_source": str(item.get("summary_source") or ""),
        "topic": "" if shared else str(item.get("topic") or ""),
        "attribution": "shared_experience" if shared
                       else str(item.get("attribution") or "unknown"),
        "fact_subject": "" if shared else str(item.get("fact_subject") or ""),
        "experience_subject": "" if shared else str(item.get("experience_subject") or ""),
        "occurrences": [] if shared else list(item.get("occurrences") or []),
        "asked_at": str(item.get("asked_at") or ""),
        "discussed": bool(item.get("asked") or item.get("asked_at")),
        "relevance": float(item.get("relevance") or 0.0),
        "semantic": float(item.get("semantic") or 0.0),
    }


def association_state(item: dict) -> str:
    state = "unknown"
    created = str(item.get("created_at") or "")
    try:
        if created and datetime.fromisoformat(created).date() < datetime.now().astimezone().date():
            state = "history"
    except (TypeError, ValueError):
        pass
    if item.get("asked") or item.get("asked_at"):
        state = "history"
    return state if state in _STATES else "unknown"


def aux_review(
    items: List[dict],
    query: str,
    conversation_history: Optional[List[Any]],
    *,
    public: bool,
    purpose: str,
    llm: Any,
    deadline: Optional[float] = None,
) -> Tuple[List[dict], bool, Optional[bool]]:
    needs_verdict = purpose in {"proactive", "delivery_check"}
    if llm is None:
        if needs_verdict:
            return [], True, None
        return conservative(items), False, True
    payload = {
        "purpose": purpose,
        "query": query,
        "now": datetime.now().astimezone().isoformat(),
        "recent_conversation": aux_history_messages(
            query, conversation_history, purpose),
        "candidates": [
            {
                "id": item["id"], "time": item["time"], "relationship": item["relationship"],
                "state": item["state"], "source": review_source(item.get("source")),
                "attribution": item.get("attribution", "unknown"),
                "experience_subject": "" if item.get("_shared_view")
                                      else item.get("experience_subject", ""),
                "fact_subject": "" if item.get("_shared_view")
                                else item.get("fact_subject", ""),
                "source_table": item.get("source_table", "event"),
                "source_kind": item.get("source_kind", ""),
                "source_turn": item.get("source_turn", ""),
                "quote": item.get("quote", ""),
                "summary": str(item.get("summary") or item.get("text") or ""),
                "raw": "" if item.get("_shared_view") else (item.get("raw") or item["text"]),
                "occurrences": [] if item.get("_shared_view") else item["occurrences"],
            }
            for item in items
        ],
    }
    timeout = 30 if deadline is None else deadline - monotonic()
    if timeout <= 0:
        return [], True, None
    result = auxiliary.complete_complex(
        llm,
        [{"role": "system", "content": _SYSTEM},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.0, max_tokens=480, timeout=timeout,
        failure_message="association review unavailable",
    )
    if result is None:
        return [], True, None
    parsed, valid, decoded = parse_review(getattr(result, "text", ""))
    if not decoded or (needs_verdict and valid is None):
        return [], True, valid
    verdicts = {entry["id"]: entry for entry in parsed if isinstance(entry, dict)}
    out: List[dict] = []
    for item in items:
        verdict = verdicts.get(item["id"])
        if not verdict:
            out.append(conservative_item(item))
            continue
        relationship = str(verdict.get("relationship") or "")
        if relationship == "drop":
            continue
        if relationship in _RELATIONSHIPS:
            item = dict(item, relationship=relationship)
        state = str(verdict.get("state") or "")
        if state in _STATES:
            item = dict(item, state=state)
        out.append(item)
    return out, False, valid


def render_association(
    items: List[dict],
    *,
    max_chars: Optional[int],
    gap: bool,
    public: bool,
    inject_max_chars: int,
    relative_date: Callable[[str], str],
    association_read_owner: Callable[[], str],
    owner_key: str,
    read_owners: Tuple[str, ...] = (),
) -> Tuple[str, List[dict]]:
    header = _ASSOCIATION_GAP_HEADER if gap else (
        _ASSOCIATION_PUBLIC_HEADER if public else _ASSOCIATION_HEADER)
    budget = max(1, int(max_chars or inject_max_chars))
    used = len(header)
    lines: List[str] = []
    selected: List[dict] = []
    for item in items:
        provenance = association_provenance(
            item, public=public, relative_date=relative_date,
            association_read_owner=association_read_owner, owner_key=owner_key,
            read_owners=read_owners,
        )
        remaining = budget - used
        if remaining <= len(provenance) + 12:
            continue
        body_budget = remaining - len(provenance) - 2
        body = association_body(item, gap=gap, public=public, budget=body_budget)
        if not body:
            continue
        line = "- %s%s" % (provenance, body)
        if len(line) > remaining:
            line = line[:remaining]
        if len(line) <= len(provenance) + 4:
            continue
        lines.append(line)
        selected.append(item)
        used += len(line) + 1
        if len(selected) >= 3:
            break
    if not selected:
        return "", []
    return header + "\n".join(lines), selected


def association_provenance(
    item: dict,
    *,
    public: bool,
    relative_date: Callable[[str], str],
    association_read_owner: Callable[[], str],
    owner_key: str,
    read_owners: Tuple[str, ...] = (),
) -> str:
    when = ""
    try:
        when = relative_date(str(item.get("time") or ""))
    except Exception:
        when = ""
    source = parse_source(item.get("source")) or {}
    kind = str(source.get("kind") or "unknown")
    certainty = str(source.get("certainty") or "unverified")
    origin = video_origin(source, watch_row=item.get("source_table") == "video")
    if public:
        label = f"公开，{origin.label}" if origin else "公开"
        bits = [label] + ([when] if when else [])
        if origin and origin.ref:
            bits.append(origin.ref)
        return "〔%s〕" % "，".join(bits)
    if item.get("_shared_view") or item.get("attribution") == "shared_experience":
        return SHARED_TOPIC_LABEL
    try:
        current_owner = association_read_owner()
    except PermissionError:
        current_owner = str(owner_key or "")
    fact_subject = str(item.get("fact_subject") or "")
    if is_other_person_source(source, fact_subject, current_owner, read_owners=read_owners):
        return SHARED_TOPIC_LABEL
    attribution = str(item.get("attribution") or "")
    if attribution in {"user_statement", "user_fact"}:
        label = "他说过的" if attribution == "user_statement" else "他的事"
    elif attribution == "self":
        label = "我经历过"
    elif kind == "video":
        label = "视频里看到的"
    elif kind == "maintainer":
        label = "你提过"
    elif kind == "self":
        label = "我经历过"
    elif kind == "system":
        label = "系统"
    else:
        label = "来源不明"
    platform = origin.label if origin else source_label(str(source.get("platform") or ""))
    if platform and label:
        label = f"{label}·{platform}"
    bits = [part for part in (when, label) if part]
    if origin and origin.ref:
        bits.append(origin.ref)
    if certainty == "unverified" and kind not in {"maintainer", "self"}:
        bits.append("未核实")
    faded = gist_provenance(item, bits)
    if faded:
        return faded
    if not bits:
        return ""
    return "〔我记着的，%s〕" % "，".join(bits)


def association_body(
    item: dict,
    *,
    gap: bool,
    public: bool,
    budget: Optional[int] = None,
) -> str:
    text = str(item.get("text") or "")
    if not text:
        return ""
    if not public and item.get("_shared_view"):
        return clip(text, budget)
    if not public and assistant_statement(item):
        statement = re.sub(r"^\[[^\]]+\]\s*", "", text)
        return clip(statement[:70] if gap else statement, budget)
    relationship = str(item.get("relationship") or "related")
    state = str(item.get("state") or "unknown")
    if gap:
        source_text = (str(item.get("summary") or text)
                       if prefers_summary(item) else text)
        snippet = strip_lead(source_text)
        topic = str(item.get("topic") or "").strip()
        body = ("%s方面：" % topic if topic and topic != "其他" else "") + snippet
        return clip(body[:70], budget)
    if public:
        source = parse_source(item.get("source")) or {}
        is_video = video_origin(source, watch_row=item.get("source_table") == "video") is not None
        if relationship == "same_event":
            lead = "同一个视频：" if is_video else "同一件事："
        else:
            lead = "相关："
        return clip(lead + text, budget)
    lead = "同一件事：" if relationship == "same_event" else "相关："
    if state == "history":
        suffix = "（已聊过）" if item.get("discussed") else "（过去的事）"
    else:
        suffix = {"ongoing": "（进行中）", "ended": "（已结束）"}.get(state, "")
    if prefers_summary(item):
        summary = strip_her_retreat_tail(str(item.get("summary") or text))
        return clip(lead + summary + suffix, budget)
    rendered_timeline = strip_her_retreat_tail(timeline(item))
    body = lead + rendered_timeline + suffix
    if budget is None or len(body) <= budget:
        return body
    return compact_timeline(item, lead, suffix, budget)
