from __future__ import annotations

import logging
import math
import os
import re
import site
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .words import *
    from .textutil import *
    from .schemas import *
except ImportError:
    from words import *
    from textutil import *
    from schemas import *

try:
    from .promotion_review import PromotionReviewMixin
    from .grounded_review import GroundedReviewMixin
    from .source_context import source_identity
except ImportError:
    from promotion_review import PromotionReviewMixin
    from grounded_review import GroundedReviewMixin
    from source_context import source_identity

logger = logging.getLogger(__name__)

_PROTECTED_MERGE_TIERS = ("long", "distilled", "important", "pending")


class LifecycleMixin(PromotionReviewMixin, GroundedReviewMixin):
        def _update_cross_session_hits(self, now: datetime) -> None:
            assert self._conn is not None
            recent = (now - timedelta(hours=24)).isoformat()
            oldest = (now - timedelta(days=14)).isoformat()
            self.update_cross_session_hits(oldest, recent)

        @staticmethod
        def _candidate_text(raw: str) -> str:
            return re.sub(r"^\[[^]]+\]\s*", "", raw).strip()

        def _promotion_candidate(self, row: sqlite3.Row, now: datetime) -> Optional[dict]:
            text = self._candidate_text(str(row["raw"]))
            if any(word in text for word in _CONFLICT_WORDS):
                return None
            signal = next((word for word in _PROMOTION_WORDS if word in text), None)
            if signal:
                scheduled = signal == "约好了" and (
                    any(word in text for word in _SCHEDULE_WORDS) or re.search(r"\d{1,2}\s*号", text)
                )
                stable_fact = signal == "记住" and any(
                    word in text for word in ("喜欢", "不喜欢", "偏好", "习惯")
                )
                suggest_tier = "important"
                pending_to = "short-lifecycle" if scheduled else "long" if stable_fact else "important"
                reason = "signal"
            elif float(row["importance"]) >= 0.7 and any(
                word in text for word in _EMOTION_WORDS + _IMPACT_WORDS
            ):
                suggest_tier = "long"
                pending_to = "long"
                reason = "importance"
            else:
                return None
            return {
                "id": int(row["id"]),
                "raw": text[:80],
                "keywords": str(row["keywords"] or ""),
                "importance": float(row["importance"]),
                "confidence": float(row["confidence"]),
                "topic": str(row["topic"] or ""),
                "ts": str(row["ts"]),
                "suggest_tier": suggest_tier,
                "reason": reason,
                "pending_to": pending_to,
            }

        def marked_promoted(self, ids: List[int], to: str) -> dict:
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            if to not in {"long", "important"}:
                raise ValueError("to must be long or important")
            unique_ids = sorted({int(event_id) for event_id in ids})
            if not unique_ids:
                return {"ids": [], "to": to, "updated": 0}
            now = datetime.now().astimezone().isoformat()
            updated = self.promote_events(unique_ids, to, now)
            return {"ids": unique_ids, "to": to, "updated": updated}

        def _distill_candidates(self, now: datetime) -> List[dict]:
            assert self._conn is not None
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = start + timedelta(days=1)
            already = self.count_distilled_between(start.isoformat(), tomorrow.isoformat())
            remaining = max(0, self._distill_daily_cap - already)
            if not remaining:
                return []
            cutoff = (now - timedelta(hours=24)).isoformat()
            rows = self.list_distill_candidates(cutoff, now.isoformat())
            candidates = []
            for row in rows:
                text = self._candidate_text(str(row["raw"]))
                topic = str(row["topic"] or "")
                if any(word in text for word in _CONFLICT_WORDS) or _has_system_meta(text) or _noise_shape(
                    text, relax_deictic=True
                ):
                    continue
                importance = float(row["importance"])
                emotional = any(word in text for word in _EMOTION_WORDS + _RELATION_WORDS)
                if importance < self._distill_threshold and not emotional and topic not in _DISTILL_TOPICS:
                    continue
                reason = "importance" if importance >= self._distill_threshold else "emotion" if emotional else "topic"
                session_id = str(row["session_id"] or "") if "session_id" in row.keys() else ""
                ts_value = str(row["ts"] or "") if "ts" in row.keys() else ""
                candidates.append({
                    "id": int(row["id"]),
                    "raw": str(row["raw"]),
                    "text": text,
                    "ts": ts_value,
                    "session_id": session_id,
                    "source_kind": "unknown",
                    "importance": importance,
                    "topic": topic,
                    "reason": reason,
                })
                if len(candidates) >= remaining:
                    break
            return candidates

        def distill(self, now: Optional[datetime] = None) -> dict:
            """在既有蒸馏额度内，对每个候选调用一次辅助槽，生成摘要与语义候选。

            上下文保留事件全文，辅助材料截断会显式标记；解析失败只保留事件，不写长期记忆。"""
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            items = []
            reasons: Dict[str, int] = {}
            candidates = self._distill_candidates(now)
            for candidate in candidates:
                review = self.review_promotion_candidate(int(candidate["id"]), now)
                proposal = review.get("proposal") or {}
                parsed = bool(review.get("parsed"))
                distilled = str(proposal.get("text") or "") if parsed else ""
                if not parsed:
                    reason = str(review.get("review_skip_reason") or proposal.get("reason") or "parse_failed")
                    if reason == "candidate_parse_failed":
                        reason = "parse_failed"
                    reasons[reason] = reasons.get(reason, 0) + 1
                elif not distilled.strip():
                    reasons["empty_text"] = reasons.get("empty_text", 0) + 1
                elif self.distill_promote(candidate["id"], distilled, now):
                    items.append({
                        "id": candidate["id"],
                        "text": distilled,
                        "topic": candidate["topic"],
                        "importance": candidate["importance"],
                        "candidate_action": proposal.get("action"),
                        "candidate_duty": proposal.get("target", ""),
                        "review_sha256": review.get("review_sha256", ""),
                        "context_truncated": review.get("context_truncated", False),
                        "resident_available": review.get("resident_available", False),
                        "event_chars": len(str(candidate.get("raw") or "")),
                        "source_kind": candidate.get("source_kind"),
                    })
                else:
                    reasons["write_failed"] = reasons.get("write_failed", 0) + 1
            result = {"items": items, "cap": self._distill_daily_cap,
                      "candidates": len(candidates), "zero_output_reasons": reasons}
            status = self._read_status()
            status["last_distill_result"] = {
                "candidates": len(candidates), "written": len(items),
                "zero_output_reasons": reasons,
            }
            self._write_status_payload(status)
            return result

        _MIGRATION_BINDING_EVENT_ID = -1

        def distill_promote(self, event_id: int, text: str, now: Optional[datetime] = None) -> bool:
            """将摘要写入 summary/distilled_at，不改 raw 或生命周期。失败返回 False。"""
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            distilled = re.sub(r"\s+", " ", text).strip()[:120]
            if not distilled:
                return False
            row = self.get_distillable_event(int(event_id), now.isoformat())
            if row is None:
                return False
            vector = _embed(distilled)
            updated = self.promote_distilled_event(
                int(event_id), distilled, self._keywords(distilled), _pack_vector(vector), now.isoformat()
            )
            if updated != 1:
                return False
            self._upsert_chroma(
                int(event_id), distilled, str(row["topic"] or ""), float(row["importance"]), str(row["ts"]), vector
            )
            return True

        def compact(self, now: Optional[datetime] = None) -> dict:
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            expired = self.expire_to_review(now.isoformat())
            forgotten_ids = self.review_expiring(
                now.isoformat(), self._expiry_review_days, _WRITE_IMPORTANCE_FLOOR
            )
            forgotten = self._purge_events(
                forgotten_ids, tombstone=True, protect_grounded=True)
            curve = self.collapse_forgetting_curve(now)
            natural_ids = self.review_expiring(
                now.isoformat(), self._expiry_review_days, _WRITE_IMPORTANCE_FLOOR, natural=True)
            forgotten += self._purge_events(natural_ids, tombstone=False, protect_grounded=True)
    
            cutoff = (now - timedelta(hours=24)).isoformat()
            rows = self.recent_merge_rows(cutoff)
            groups: Dict[tuple, List[sqlite3.Row]] = {}
            for row in rows:
                if row["keywords"]:
                    groups.setdefault(
                        (row["owner_key"], row["keywords"], source_identity(row["source_context"]),
                         row["experience_subject"], row["fact_subject"], row["involved_owner_keys"]),
                        []).append(row)
            grounded_pins = self._grounded_protected_ids()
            merged = 0
            merge_groups_skipped = 0
            for group in groups.values():
                if len(group) < 2 or any(r["id"] in grounded_pins for r in group):
                    continue
                protected = [
                    row for row in group if str(row["tier"] or "") in _PROTECTED_MERGE_TIERS
                    or int(row["id"]) in grounded_pins
                ]
                if len(protected) >= 2:
                    merge_groups_skipped += 1
                    continue
                if protected:
                    survivor = protected[0]
                else:
                    survivor = max(group, key=lambda row: (float(row["importance"]), int(row["id"])))
                duplicates = [int(row["id"]) for row in group if int(row["id"]) != int(survivor["id"])]
                merged += self.merge_duplicate_events(
                    int(survivor["id"]), duplicates,
                    max(float(row["importance"]) for row in group),
                    max(float(row["confidence"]) for row in group), now.isoformat())
    
            self._update_cross_session_hits(now)
            candidate_rows = self.list_short_candidates()
            candidates = []
            for row in candidate_rows:
                candidate = self._promotion_candidate(row, now)
                if candidate is not None:
                    candidates.append(candidate)
            new_candidates = list(candidates)
            for row in self.list_pending_candidates():
                candidates.append({
                    "id": int(row["id"]),
                    "raw": self._candidate_text(str(row["raw"]))[:80],
                    "keywords": str(row["keywords"] or ""),
                    "importance": float(row["importance"]),
                    "confidence": float(row["confidence"]),
                    "topic": str(row["topic"] or ""),
                    "ts": str(row["ts"]),
                    "suggest_tier": "important" if row["candidate_reason"] == "signal" else "long",
                    "reason": str(row["candidate_reason"] or ""),
                    "pending_to": str(row["pending_to"] or ""),
                })
            candidates.sort(key=lambda item: item["id"])
            if new_candidates:
                pending_by_id = {item["id"]: item["pending_to"] for item in new_candidates}
                candidate_reasons = {item["id"]: item["reason"] for item in new_candidates}
                self.mark_pending(pending_by_id, now.isoformat(), candidate_reasons)
    
            distill_candidates = self._distill_candidates(now)
            resident = self._resident_cold_report(now)
    
            status = self._read_status()
            base_status = dict(status)
            today = now.date().isoformat()
            same_day = status.get("date") == today
            prior_runs = int(status.get("daily_compactor_runs", 0)) if same_day else 0
            decayed = 0
            importance_decayed = 0
            if status.get("last_compactor_date") != today:
                self._daily_rollover(today)
                status = self._read_status()
                base_status = dict(status)
                old_cutoff = (now - timedelta(days=_CONFIDENCE_DECAY_AFTER_DAYS)).isoformat()
                decayed = self.decay_confidence(old_cutoff, 0.9, 0.1, now.isoformat())
                importance_decayed = self.decay_expiring_importance(
                    self._expiring_importance_decay, self._expiring_importance_floor, now.isoformat()
                )
            if status.get("date") != today:
                status["date"] = today
                status["daily_writes"] = 0
                status["daily_scheduled_asks"] = 0
                status["daily_scheduled_attempts"] = 0
                status["distill_errors_today"] = 0
            status["last_compactor_date"] = today
            status["last_compactor_at"] = now.isoformat()
            status["daily_compactor_runs"] = prior_runs + 1
            result = {
                "expired": expired,
                "forgotten": forgotten,
                "merged": merged,
                "merge_groups_skipped": merge_groups_skipped,
                "decayed": decayed,
                "importance_decayed": importance_decayed,
                "faded": curve["faded"],
                "dormant": curve["dormant"],
                "candidates": candidates,
                "distill_candidates": distill_candidates,
                "resident_capacity": resident["capacity"],
                "cold_resident_entries": resident["cold_resident_entries"],
            }
            try:
                self.distill()
                status["last_distill_ok_at"] = now.isoformat()
            except Exception as exc:
                logger.warning("sanae_mem: compact distill failed (%s)", type(exc).__name__)
                status["distill_errors_today"] = int(status.get("distill_errors_today", 0)) + 1
                status["last_distill_error_at"] = now.isoformat()
                status["last_distill_error"] = (f"{type(exc).__name__}: {exc}")[:120]
            status["last_compactor"] = result
            self._write_status_payload(status, base=base_status)
            return result

        def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
            """记录压缩次数，按宿主钩子契约返回空串。

            已识别的测试会话不写 status；其余只更新压缩观测字段，不改跨日复位所用的 date。
            """
            del messages
            now = datetime.now().astimezone()
            today = now.date().isoformat()
            status = self._read_status()
            previous = int(status.get("pre_compress_count_today", 0)) if status.get("pre_compress_date") == today else 0
            status["pre_compress_date"] = today
            status["last_pre_compress_at"] = now.isoformat()
            status["pre_compress_count_today"] = previous + 1
            self._write_status_payload(status)
            return ""
