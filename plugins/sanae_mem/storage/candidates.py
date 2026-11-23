from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

from ..infra.model_output import _load_braced_json

logger = logging.getLogger(__name__)

DUTY_TARGETS = ("user", "memory")
CANDIDATE_DUTIES = ("user", "memory", "none")
SOURCE_KINDS = ("user_affirmed", "event_promotion", "user_statement", "unknown")
ACTIONS = ("none", "merge_evidence", "propose_new")
DEMOTE_KINDS = (
    "outdated_recent_status",
    "core_trait_or_identity",
    "long_term_habit_or_preference",
    "active_or_uncertain",
)


def _clean(value: Any, limit: int = 200) -> str:
    return str(value or "").strip()[:limit]


def parse_skill_mutation_result(raw: Any) -> Optional[Dict[str, Any]]:
    """Accept only a complete, self-consistent skill-domain verdict."""
    if isinstance(raw, dict):
        item = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            item = _load_braced_json(text)
        except (TypeError, ValueError):
            return None
    if not isinstance(item, dict) or not isinstance(item.get("allow"), bool):
        return None
    domain = ("objective_tool_or_technical" if item["allow"]
              else "protected_persona_or_interpersonal")
    if item.get("domain") != domain or not isinstance(item.get("reason"), str):
        return None
    return {"allow": item["allow"], "domain": domain, "reason": _clean(item.get("reason"), 240)}


def parse_candidate_result(raw: Any) -> Optional[Dict[str, Any]]:
    """保守解析辅助槽返回的候选结果；任何不确定 ⇒ None（**不写长期**）。"""
    if isinstance(raw, dict):
        data = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            data = _load_braced_json(text)
        except (ValueError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    summary = _clean(data.get("summary"), 150)
    duty = _clean(data.get("duty"), 16).lower()
    source_kind = _clean(data.get("source_kind"), 32).lower()
    long_term = data.get("long_term")
    covered_by = _clean(data.get("covered_by"), 6000)
    significance = _clean(data.get("significance"), 120)
    if duty not in CANDIDATE_DUTIES or not isinstance(long_term, bool):
        return None
    if duty == "none":
        if long_term or covered_by:
            return None
    elif not summary:
        return None
    if source_kind not in SOURCE_KINDS:
        source_kind = "unknown"
    return {
        "summary": summary, "duty": duty, "source_kind": source_kind,
        "long_term": long_term, "covered_by": covered_by, "significance": significance,
    }


def parse_grounded_user_result(raw: Any) -> Optional[Dict[str, Any]]:
    """Strict scoped response: 'none' is a valid refusal, not a malformed model result.

    Do not truncate covered_by: doing so silently changes an exact existing entry
    reference into a substring and breaks USER's non-duplication/revision contract.
    """
    if isinstance(raw, dict):
        item = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            item = _load_braced_json(text)
        except (TypeError, ValueError):
            return None
    if not isinstance(item, dict):
        return None
    if (not isinstance(item.get("summary"), str)
            or not isinstance(item.get("covered_by"), str)
            or not isinstance(item.get("significance"), str)
            or not isinstance(item.get("long_term"), bool)
            or item.get("duty") not in ("user", "none")
            or item.get("source_kind") != "user_statement"):
        return None
    if (len(item["summary"]) > 6000 or len(item["covered_by"]) > 6000
            or len(item["significance"]) > 600):
        return None
    if item["duty"] == "none":
        if item["long_term"] or item["covered_by"]:
            return None
    elif not item["summary"].strip() or not item["long_term"]:
        return None
    return {key: item[key] for key in
            ("summary", "duty", "source_kind", "long_term", "covered_by", "significance")}


def review(event: Dict[str, Any], candidate: Optional[Dict[str, Any]], *,
           baseline_sha256: str = "", review_sha256: str = "") -> Dict[str, Any]:
    """将事件原文与语义审核结果转换为提案，不直接写长期记忆。

    来源资格以宿主传入的 event["source_kind"] 为准，不采信模型自报。后台证据与自然正文分开，
    并绑定事件全文、目标、常驻 baseline 和后端审核记录；重放时任一不匹配都拒写并重新审核。"""
    event_id = event.get("id")
    raw = str(event.get("raw") or "")
    ts = _clean(event.get("ts"), 64)
    if not raw or event_id is None:
        return {"action": "none", "reason": "no_event_evidence",
                "background": {"event_id": event_id}}
    evidence_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    background = {"event_id": int(event_id), "event_sha256": evidence_hash, "ts": ts,
                  "event_chars": len(raw),
                  "source_kind": str(event.get("source_kind") or "unknown"),
                  "baseline_sha256": str(baseline_sha256 or ""),
                  "review_sha256": str(review_sha256 or "")}
    if not candidate:
        return {"action": "none", "reason": "candidate_parse_failed", "background": background}
    if not candidate.get("long_term"):
        return {"action": "none", "reason": "not_long_term", "background": background,
                "duty": candidate.get("duty")}
    body = candidate["summary"]
    if candidate.get("covered_by"):
        return {"action": "merge_evidence", "target": candidate["duty"], "text": body,
                "covered_by": candidate["covered_by"], "background": background,
                "significance": candidate.get("significance", "")}
    return {"action": "propose_new", "target": candidate["duty"], "text": body,
            "background": background, "significance": candidate.get("significance", "")}


def build_model_context(*, event_raw: str, source_kind: str, resident_user: str = "",
                        resident_memory: str = "", summaries: Optional[List[str]] = None,
                        budget_chars: int = 12000) -> Dict[str, Any]:
    """组装辅助槽上下文，原文与常驻正文均由调用方实读传入。

    事件原文完整保留；预算只截辅助材料。截断时返回 truncated=True 和 truncated_parts，
    调用方不得将其视为完整审核。"""
    event_body = str(event_raw or "")
    head = [f"【来源】{source_kind or 'unknown'}", f"【事件原文】{event_body}"]
    tail_parts: List[str] = []
    if resident_user:
        tail_parts.append(f"【当前 USER 常驻】{resident_user}")
    if resident_memory:
        tail_parts.append(f"【当前 MEMORY 常驻】{resident_memory}")
    for summary in (summaries or []):
        if str(summary).strip():
            tail_parts.append(f"【更高层概括】{str(summary).strip()}")
    head_text = "\n".join(head)
    room = max(0, int(budget_chars) - len(head_text))
    kept: List[str] = []
    truncated_parts: List[str] = []
    for part in tail_parts:
        label = part.split("】", 1)[0] + "】"
        if room <= 0:
            truncated_parts.append(label)
            continue
        if len(part) <= room:
            kept.append(part)
            room -= len(part) + 1
        else:
            kept.append(part[:room])
            truncated_parts.append(label)
            room = 0
    text = "\n".join(head + kept)
    return {"text": text, "truncated": bool(truncated_parts), "truncated_parts": truncated_parts,
            "event_chars": len(event_body), "source_kind": source_kind,
            "budget_chars": int(budget_chars)}


def parse_user_demote_result(
    raw: Any, *, valid_indices: Set[int],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Parse strict JSON decisions for USER.md demotion candidates (fail-closed on any ambiguity)."""
    if not valid_indices:
        return {}
    if isinstance(raw, dict):
        payload = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            payload = _load_braced_json(text)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(valid_indices):
        return None
    parsed: Dict[int, Dict[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            return None
        idx = item.get("index")
        allow_demote = item.get("allow_demote")
        kind = _clean(item.get("kind"), 64).lower()
        reason = _clean(item.get("reason"), 240)
        if isinstance(idx, bool) or not isinstance(idx, int) or idx not in valid_indices or idx in parsed:
            return None
        if not isinstance(allow_demote, bool) or kind not in DEMOTE_KINDS:
            return None
        if allow_demote and kind != "outdated_recent_status":
            return None
        if not allow_demote and kind == "outdated_recent_status":
            return None
        parsed[idx] = {
            "index": idx,
            "allow_demote": allow_demote,
            "kind": kind,
            "reason": reason,
        }
    if set(parsed.keys()) != set(valid_indices):
        return None
    return parsed
