from __future__ import annotations

import re
from collections import namedtuple
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence

try:
    from . import daily_inputs
    from .daily_rules import THREAD_TTL_DAYS, _CHAT_META, _ROUTINE
except ImportError:
    import daily_inputs
    from daily_rules import THREAD_TTL_DAYS, _CHAT_META, _ROUTINE

KIND_FACT = "DAILY_FACT"
KIND_CONSTRAINT = "ACTIVE_CONSTRAINT"
KIND_THREAD = "OPEN_THREAD"
_KINDS = frozenset((KIND_FACT, KIND_CONSTRAINT, KIND_THREAD))
_ACTIONS = frozenset(("set", "update", "close"))
MAX_ITEMS = 4
MAX_TEXT_CHARS = 110
MAX_EVENT_KEY = 64

_FUTURE = {
    "大后天": 3, "后天": 2, "明天": 1, "明日": 1, "明早": 1, "明晚": 1,
    "下周": 7, "下个星期": 7, "下个月": 30,
}
_FUTURE_DAY = re.compile("|".join(_FUTURE))
_DATE_PREFIX = re.compile(r"^\((\d{4}-\d{2}-\d{2})\)\s*")
_SPECULATIVE = re.compile(
    r"八成|估计|可能|大概|好像|说不定|没准|"
    r"(?:肯定|应该)[^，。！？;；\n]*(?:已经|早就|了(?:吧|啊|呀|呢)?$)"
)
_AMBIGUOUS_MODAL = re.compile(r"肯定|应该")
_PLAN_WORDING = re.compile(r"计划|打算|准备|需要")
_DESIGN_INTENT_CLAUSE = re.compile(
    r"(?:把|让)(?:你|我|早苗)(?:改|磨|调|训|塑|捏|更)|"
    r"(?:改动|修改|调整)的?(?:目的|目标|意图)是|"
    r"AI\s*味|ai\s*味|人机味|模板味|"
    r"(?:模型|提示词|prompt)[^，,。；;\n]{0,16}(?:难调|好调|调教|微调)|"
    r"(?:难调|调教)[^，,。；;\n]{0,16}(?:模型|提示词|prompt)|"
    r"(?:塑造成|调成|改成|磨出|养出|留点|长出)[^，,。；;\n]{0,10}(?:性格|人设|语气|风格|腔调|脾气|活人感)|"
    r"调教目标|设计意图"
)


def _strip_design_intent(text: str) -> str:
    parts = [p.strip() for p in re.split(r"[，,；;。]+", str(text or "")) if p.strip()]
    kept = [p for p in parts if not _DESIGN_INTENT_CLAUSE.search(p)]
    return "，".join(kept)


_SELF_RULE_OR_META_CLAUSE = re.compile(
    r"(?:以后|下次|往后|今后)[^，,。；;\n]{0,12}(?:我|自己)?[^，,。；;\n]{0,8}"
    r"(?:绝对不能|绝不(?:再|能)?|不能再?|别再|不要再|必须|一定|要先|得先|切记|记住别)|"
    r"(?:我|自己)(?:以后|下次|往后|今后)(?:绝对不能|绝不(?:再|能)?|不能再?|别再|不要再|必须|一定|要先|得先)|"
    r"记忆(?:遗忘)?机制|遗忘(?:曲线|机制|规则)|系统提示词|行为层(?:条文|规范|规则)|常驻记忆(?:规则|机制)"
)
_DANGLING_RULE_OR_COND_CLAUSE = re.compile(
    r"^(?:要是|如果|若|当|一旦|每当|凡是|遇到|碰到|面对|除非)|"
    r"(?:不由我|由他(?:设定|决定)|我有权|允许我|不准我|切记|切忌|严禁)"
)


def strip_self_rule_and_design_intent(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if not _DESIGN_INTENT_CLAUSE.search(raw) and not _SELF_RULE_OR_META_CLAUSE.search(raw):
        return raw
    parts = [p.strip() for p in re.split(r"[，,；;。！!\n]+", raw) if p.strip()]
    kept = [
        p for p in parts
        if not _DESIGN_INTENT_CLAUSE.search(p)
        and not _SELF_RULE_OR_META_CLAUSE.search(p)
        and not _DANGLING_RULE_OR_COND_CLAUSE.search(p)
    ]
    return "，".join(kept)

Sources = namedtuple(
    "Sources",
    "user assistant known message known_map known_slots recent_user preceding_user known_dates",
    defaults=("", "", "", "", {}, {}, "", "", {}),
)
SemanticItem = namedtuple(
    "SemanticItem",
    "kind action target_slot event_key text quote uncertain feeling expires_on",
    defaults=(False, "", None),
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _safe_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if daily_inputs._is_system_injection(value) or daily_inputs._looks_secret(value):
        return False
    if value.startswith(("[Quoted message]", ">")):
        return False
    return True


def _strip_copied_target_date(text: str, target_slot: str, pools: Sources) -> str:
    target_date = str((pools.known_dates or {}).get(target_slot) or "").strip() if target_slot else ""
    target_text = str((pools.known_slots or {}).get(target_slot) or "")
    if not target_date or target_date in pools.message or target_date in target_text:
        return text
    prefix = f"({target_date}) "
    if text.startswith(prefix) and text[len(prefix):].strip():
        return text[len(prefix):].lstrip()
    return text


def _has_unverified_date_prefix(text: str, pools: Sources, target_text: str) -> bool:
    match = _DATE_PREFIX.match(text)
    if match is None:
        return False
    day = match.group(1)
    return day not in str(pools.message or "") and day not in str(target_text or "")


def _label_plan(text: str, today: str):
    days = [_FUTURE[m.group()] for m in _FUTURE_DAY.finditer(text)]
    if not days or not today or re.match(r"^（\d{2}-\d{2} 说）", text):
        return text, None
    expiry = date.fromisoformat(today) + timedelta(days=min(days) + 2)
    return f"（{today[5:]} 说）{text}", expiry.isoformat()


def validate_items(
    raw_items: Sequence[dict], *, message: str, sources: Any,
    known_map: Dict[str, str], time_label: str = "",
    max_items: int = MAX_ITEMS, today: str = "",
    recent_user: str = "",
) -> List[SemanticItem]:
    if not _safe_text(message):
        return []
    pools = (sources if isinstance(sources, Sources) else Sources())._replace(
        message=message, known_map=known_map,
    )
    picked: List[SemanticItem] = []
    seen: set = set()
    for raw in raw_items:
        if len(picked) >= max_items:
            break
        item = _validate_one(raw, pools=pools, today=today)
        key = (item.kind, item.action, item.target_slot or _norm(item.text)) if item else None
        if not item or key in seen:
            continue
        seen.add(key)
        picked.append(item)
    return picked


def _validate_one(raw: dict, *, pools: Sources, today: str) -> Optional[SemanticItem]:
    message = str(pools.message or "")
    kind = str(raw.get("kind") or "").strip().upper()
    action = str(raw.get("action") or "set").strip().lower()
    quote = str(raw.get("quote") or "").strip()
    target_slot = str(raw.get("target_slot") or "").strip()
    event_key = re.sub(r"\s+", "", str(raw.get("event_key") or ""))
    if (kind not in _KINDS or action not in _ACTIONS or not quote or quote not in message
            or len(event_key) > MAX_EVENT_KEY or raw.get("uncertain") is True):
        return None
    if action == "set":
        target_slot = ""
    target = (pools.known_slots or {}).get(target_slot, "") if target_slot else ""
    if action != "set" and (not target_slot or not target or pools.known_map.get(target_slot) != kind):
        return None

    text = _strip_copied_target_date(str(raw.get("text") or "").strip(), target_slot, pools)
    text = _strip_design_intent(text)
    if kind == KIND_THREAD and (
        _ROUTINE.search(text) or _ROUTINE.search(quote)
        or _CHAT_META.fullmatch(text)
    ):
        return None
    if _has_unverified_date_prefix(text, pools, target):
        return None
    evidence = daily_inputs._quoted_clauses(message, quote)
    if any(_SPECULATIVE.search(clause) for clause in evidence):
        return None
    if any(_AMBIGUOUS_MODAL.search(clause) for clause in evidence) and not (
        kind == KIND_FACT and action != "close" and raw.get("plan") is True
        and _PLAN_WORDING.search(text)
    ):
        return None
    if action != "close":
        if not text or not _safe_text(text):
            return None
        text, expires_on = _label_plan(text, today)
        if kind == KIND_THREAD and expires_on is None and today:
            base_day = (
                str((pools.known_dates or {}).get(target_slot) or "").strip()
                if action == "update" and target_slot else ""
            ) or today
            try:
                expires_on = (
                    date.fromisoformat(base_day) + timedelta(days=THREAD_TTL_DAYS)
                ).isoformat()
            except ValueError:
                expires_on = None
        if len(text) > MAX_TEXT_CHARS:
            return None
    else:
        expires_on = None
        if text and (not _safe_text(text) or len(text) > MAX_TEXT_CHARS):
            text = ""
    return SemanticItem(
        kind, action, target_slot, event_key, text, quote, False, "", expires_on,
    )
