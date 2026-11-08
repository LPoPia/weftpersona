"""在既有蒸馏额度内，为原文事件生成长期记忆候选。

候选只产出提案或证据合并，不直接写 USER/MEMORY。解析失败返回 action="none"，原文保留。
已被常驻内容覆盖时合并证据；其余提案走既有 write_approval，并绑定 baseline/CAS。
自然正文不包含 id、hash、CAS 或版本；这些证据只保存在后台。"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DUTY_TARGETS = ("user", "memory")
SOURCE_KINDS = ("user_affirmed", "event_promotion", "user_statement", "unknown")
ACTIONS = ("none", "merge_evidence", "propose_new")


def _clean(value: Any, limit: int = 200) -> str:
    return str(value or "").strip()[:limit]


def parse_candidate_result(raw: Any) -> Optional[Dict[str, Any]]:
    """保守解析辅助槽返回的候选结果；任何不确定 ⇒ None（**不写长期**）。"""
    if isinstance(raw, dict):
        data = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except (ValueError, TypeError):
            return None
    if not isinstance(data, dict):
        return None
    summary = _clean(data.get("summary"), 150)
    duty = _clean(data.get("duty"), 16).lower()
    source_kind = _clean(data.get("source_kind"), 32).lower()
    long_term = data.get("long_term")
    covered_by = _clean(data.get("covered_by"), 120)
    significance = _clean(data.get("significance"), 120)
    if not summary or duty not in DUTY_TARGETS:
        return None
    if not isinstance(long_term, bool):
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
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            item = json.loads(text[start:end + 1])
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


def stage_proposal(home: Any, proposal: Dict[str, Any], *, store: Any = None,
                   event_version: str = "") -> Dict[str, Any]:
    """把 `propose_new` 变成**可见提案**（复用既有 write_approval 暂存/基线；不写目标文件）。

    暂存面不可用或失败 ⇒ 如实返回失败（不写长期、不静默降级成直接写入）。

    载荷里的 `classification` 是**后端审核结论的引用**（不是自报证据）：`review_sha256`＋
    `baseline_sha256`＋`event_sha256` 三项都在重放时回读后端记录逐项核对；缺记录或对不上 ⇒ 拒写。
    """
    if proposal.get("action") != "propose_new":
        return {"staged": False, "reason": "not_a_new_proposal", "proposal": proposal}
    background = proposal.get("background") or {}
    payload = {
        "action": "add", "target": proposal["target"], "content": proposal["text"],
        "classification": {
            "duty": proposal["target"],
            "source_kind": str(background.get("source_kind") or "unknown"),
            "event_id": background.get("event_id"),
            "event_sha256": background.get("event_sha256"),
            "event_version": str(event_version or background.get("event_version") or ""),
            "review_sha256": str(background.get("review_sha256") or ""),
            "baseline_sha256": str(background.get("baseline_sha256") or ""),
            "content_sha256": hashlib.sha256(str(proposal["text"]).encode("utf-8")).hexdigest(),
            "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    }
    if store is not None:
        try:
            from tools.memory_tool_store import target_baseline_hash
            payload["baseline"] = {"target_hash": target_baseline_hash(store._entries_for(proposal["target"]))}
        except Exception as exc:
            return {"staged": False, "reason": f"baseline_unavailable:{type(exc).__name__}",
                    "proposal": proposal}
    try:
        from tools import write_approval as wa
        record = wa.stage_write(wa.MEMORY, payload,
                                summary=f"长记忆候选（{proposal['target']}）：{proposal['text'][:60]}",
                                origin=wa.current_origin())
    except Exception as exc:
        return {"staged": False, "reason": f"approval_unavailable:{type(exc).__name__}",
                "proposal": proposal}
    if not record.get("staged"):
        return {"staged": False, "reason": record.get("stage_error", "stage_failed"), "proposal": proposal}
    return {"staged": True, "pending_id": record.get("id", ""), "proposal": proposal}
