from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Sequence

from ..domains.commitments import parse_aux_json
from ..infra.host_bridge import is_test_isolated_turn
from ..infra.status_store import StatusStore
from ..infra.words import (
    _CONFIDENCE_DECAY_AFTER_DAYS,
    _CONFLICT_WORDS,
    _DISTILL_TOPICS,
    _EMOTION_WORDS,
    _RELATION_WORDS,
    _WRITE_IMPORTANCE_FLOOR,
    _has_system_meta,
    _noise_shape,
)
from ..infra.textutil import _candidate_text, _embed, _pack_vector

from . import group_ingest
from ..governance.source_context import source_identity

if TYPE_CHECKING:
    from ..governance.llm_guard import AuxiliaryModel
    from ..storage.sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)

_PROTECTED_MERGE_TIERS = ("long", "distilled", "important", "pending")


class _ReviewExpiring(Protocol):
    def __call__(self, now: str, window_days: int, importance: float, /, *,
                 natural: bool = False) -> list[int]: ...


class _PurgeEvents(Protocol):
    def __call__(self, ids: list[int], /, *, tombstone: bool, protect_grounded: bool) -> int: ...


@dataclass(frozen=True)
class LifecycleDependencies:
    current_data_dir: Callable[[], Path]
    current_session_id: Callable[[], str]
    current_persona_name: Callable[[], str]
    current_model: Callable[[], AuxiliaryModel | None]
    current_promotion_owner: Callable[[], str]
    is_primary: Callable[[], bool]
    group_scene_active: Callable[[], bool]
    private_chat_owner: Callable[[], str]
    group_maintenance_owners: Callable[[], Sequence[str]]
    distill_daily_cap: Callable[[], int]
    promote_explicit_save: Callable[[int, str, datetime], dict]
    promote_events: Callable[[list[int], str, str], int]
    count_distilled_between: Callable[[str, str], int]
    list_distill_candidates: Callable[[str, str], list[sqlite3.Row]]
    review_maintenance_candidate: Callable[[int, datetime], dict]
    can_review_candidate: Callable[[datetime], bool]
    note_candidate_call: Callable[[datetime], None]
    get_distillable_event: Callable[[int, str], sqlite3.Row | None]
    promote_distilled_event: Callable[[int, str, str, bytes, str], int]
    keywords: Callable[[str], str]
    upsert_chroma: Callable[[int, str, str, float, str, list[float]], None]
    expire_to_review: Callable[[str], int]
    review_expiring: _ReviewExpiring
    purge_events: _PurgeEvents
    collapse_forgetting_curve: Callable[[datetime], dict[str, int]]
    recent_merge_rows: Callable[[str], list[sqlite3.Row]]
    grounded_protected_ids: Callable[[], set[int]]
    merge_duplicate_events: Callable[[int, list[int], float, float, str], int]
    resident_cold_report: Callable[[datetime], dict]
    daily_rollover: Callable[[str], None]
    decay_confidence: Callable[[str, float, float, str], int]
    decay_expiring_importance: Callable[[float, float, str], int]
    status: StatusStore


class Lifecycle:
        def __init__(
            self, sqlite: SqliteFacade, dependencies: LifecycleDependencies, *,
            distill_threshold: float, expiry_review_days: int,
            expiring_importance_decay: float, expiring_importance_floor: float,
        ) -> None:
            self._sqlite = sqlite
            self._deps = dependencies
            self._distill_threshold = distill_threshold
            self._expiry_review_days = expiry_review_days
            self._expiring_importance_decay = expiring_importance_decay
            self._expiring_importance_floor = expiring_importance_floor

        def reconfigure(
            self, *, distill_threshold: float, expiry_review_days: int,
            expiring_importance_decay: float, expiring_importance_floor: float,
        ) -> None:
            self._distill_threshold = distill_threshold
            self._expiry_review_days = expiry_review_days
            self._expiring_importance_decay = expiring_importance_decay
            self._expiring_importance_floor = expiring_importance_floor

        def marked_promoted(self, ids: List[int], to: str) -> dict:
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            if to not in {"long", "important"}:
                raise ValueError("to must be long or important")
            unique_ids = sorted({int(event_id) for event_id in ids})
            if not unique_ids:
                return {"ids": [], "to": to, "updated": 0}
            if self._deps.group_scene_active() or (
                self._deps.is_primary() and not self._deps.private_chat_owner()
            ):
                return {"ids": unique_ids, "to": to, "updated": 0}
            now = datetime.now().astimezone()
            owner = self._deps.current_promotion_owner()
            if to == "long" and owner:
                updated = 0
                for event_id in unique_ids:
                    if self._deps.promote_explicit_save(event_id, owner, now).get("ok"):
                        updated += 1
                return {"ids": unique_ids, "to": to, "updated": updated,
                        "channel": "explicit_save"}
            updated = self._deps.promote_events(unique_ids, to, now.isoformat())
            return {"ids": unique_ids, "to": to, "updated": updated}

        def _distill_candidates(self, now: datetime) -> List[dict]:
            assert self._sqlite.is_open
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = start + timedelta(days=1)
            already = self._deps.count_distilled_between(start.isoformat(), tomorrow.isoformat())
            remaining = max(0, self._deps.distill_daily_cap() - already)
            if not remaining:
                return []
            cutoff = (now - timedelta(hours=24)).isoformat()
            rows = self._deps.list_distill_candidates(cutoff, now.isoformat())
            candidates = []
            for row in rows:
                text = _candidate_text(str(row["raw"]))
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

        def _group_distill_candidates(self, now: datetime, limit: int) -> List[sqlite3.Row]:
            if limit <= 0:
                return []
            owners = [str(owner) for owner in self._deps.group_maintenance_owners() if owner]
            if not owners:
                return []
            cutoff = (now - timedelta(hours=24)).isoformat()
            marks = ",".join("?" for _ in owners)
            return self._sqlite.fetchall(
                f"""SELECT id, raw, owner_key, source_context, importance, topic, ts
                   FROM events
                   WHERE created_at>=? AND blacklist=0 AND distilled_at IS NULL
                     AND (tier='short' OR (tier IN ('long','important') AND length(raw)>150))
                     AND COALESCE(summary_source,'') NOT IN ('gist_faded', 'distilled')
                     AND expired_at IS NULL
                     AND (ttl_expire IS NULL OR ttl_expire>?)
                     AND owner_key IN ({marks})
                   ORDER BY importance DESC, id DESC LIMIT ?""",
                (cutoff, now.isoformat(), *owners, int(limit)),
            )

        def distill(self, now: Optional[datetime] = None) -> dict:
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            items = []
            reasons: Dict[str, int] = {}
            candidates = self._distill_candidates(now)
            for candidate in candidates:
                review = self._deps.review_maintenance_candidate(int(candidate["id"]), now)
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

            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = start + timedelta(days=1)
            remaining = max(0, self._deps.distill_daily_cap() - self._deps.count_distilled_between(
                start.isoformat(), tomorrow.isoformat()))
            group_candidates = self._group_distill_candidates(now, remaining)
            for row in group_candidates:
                if not self._deps.can_review_candidate(now):
                    reasons["quota_exhausted"] = reasons.get("quota_exhausted", 0) + 1
                    continue
                if self._deps.get_distillable_event(
                        int(row["id"]), now.isoformat()) is None:
                    reasons["write_failed"] = reasons.get("write_failed", 0) + 1
                    continue
                self._deps.note_candidate_call(now)
                summary, reason = group_ingest.summarize_group_event(
                    row, now,
                    current_persona_name=self._deps.current_persona_name,
                    current_model=self._deps.current_model,
                    get_event=self._deps.get_distillable_event,
                    parse_json=parse_aux_json,
                    submit_summary=self.distill_promote,
                )
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                items.append({
                    "id": int(row["id"]),
                    "text": summary,
                    "topic": str(row["topic"] or ""),
                    "importance": float(row["importance"] or 0),
                    "candidate_action": "summary",
                    "candidate_duty": "group",
                    "review_sha256": "",
                    "context_truncated": False,
                    "resident_available": False,
                    "event_chars": len(str(row["raw"] or "")),
                    "source_kind": "group",
                })

            candidate_count = len(candidates) + len(group_candidates)
            result = {"items": items, "cap": self._deps.distill_daily_cap(),
                      "candidates": candidate_count, "zero_output_reasons": reasons}

            def record(status: dict) -> dict:
                status["last_distill_result"] = {
                    "candidates": candidate_count, "written": len(items),
                    "zero_output_reasons": reasons,
                }
                return status

            self._deps.status.mutate(record)
            return result

        def distill_promote(self, event_id: int, text: str, now: Optional[datetime] = None) -> bool:
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            distilled = re.sub(r"\s+", " ", text).strip()[:120]
            if not distilled:
                return False
            row = self._deps.get_distillable_event(int(event_id), now.isoformat())
            if row is None:
                return False
            vector = _embed(distilled, data_dir=self._deps.current_data_dir())
            updated = self._deps.promote_distilled_event(
                int(event_id), distilled, self._deps.keywords(distilled), _pack_vector(vector), now.isoformat()
            )
            if updated != 1:
                return False
            self._deps.upsert_chroma(
                int(event_id), distilled, str(row["topic"] or ""), float(row["importance"]), str(row["ts"]), vector
            )
            return True

        def compact(self, now: Optional[datetime] = None) -> dict:
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            expired = self._deps.expire_to_review(now.isoformat())
            forgotten_ids = self._deps.review_expiring(
                now.isoformat(), self._expiry_review_days, _WRITE_IMPORTANCE_FLOOR
            )
            forgotten = self._deps.purge_events(
                forgotten_ids, tombstone=True, protect_grounded=True)
            curve = self._deps.collapse_forgetting_curve(now)
            natural_ids = self._deps.review_expiring(
                now.isoformat(), self._expiry_review_days, _WRITE_IMPORTANCE_FLOOR, natural=True)
            forgotten += self._deps.purge_events(natural_ids, tombstone=False, protect_grounded=True)
    
            cutoff = (now - timedelta(hours=24)).isoformat()
            rows = self._deps.recent_merge_rows(cutoff)
            groups: Dict[tuple, List[sqlite3.Row]] = {}
            for row in rows:
                if row["keywords"]:
                    groups.setdefault(
                        (row["owner_key"], row["keywords"], source_identity(row["source_context"]),
                         row["experience_subject"], row["fact_subject"], row["involved_owner_keys"]),
                        []).append(row)
            grounded_pins = self._deps.grounded_protected_ids()
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
                merged += self._deps.merge_duplicate_events(
                    int(survivor["id"]), duplicates,
                    max(float(row["importance"]) for row in group),
                    max(float(row["confidence"]) for row in group), now.isoformat())
    
            distill_candidates = self._distill_candidates(now)
            resident = self._deps.resident_cold_report(now)
    
            status = self._deps.status.read()
            base_status = dict(status)
            today = now.date().isoformat()
            decayed = 0
            importance_decayed = 0
            if status.get("last_compactor_date") != today:
                self._deps.daily_rollover(today)
                status = self._deps.status.read()
                base_status = dict(status)
                old_cutoff = (now - timedelta(days=_CONFIDENCE_DECAY_AFTER_DAYS)).isoformat()
                decayed = self._deps.decay_confidence(old_cutoff, 0.9, 0.1, now.isoformat())
                importance_decayed = self._deps.decay_expiring_importance(
                    self._expiring_importance_decay, self._expiring_importance_floor, now.isoformat()
                )
            result = {
                "expired": expired,
                "forgotten": forgotten,
                "merged": merged,
                "merge_groups_skipped": merge_groups_skipped,
                "decayed": decayed,
                "importance_decayed": importance_decayed,
                "faded": curve["faded"],
                "dormant": curve["dormant"],
                "distill_candidates": distill_candidates,
                "resident_capacity": resident["capacity"],
                "cold_resident_entries": resident["cold_resident_entries"],
            }
            distill_error = None
            try:
                self.distill(now=now)
            except Exception as exc:
                logger.warning("sanae_mem: compact distill failed (%s)", type(exc).__name__)
                distill_error = f"{type(exc).__name__}: {exc}"[:120]
            status["last_compactor"] = result
            self._deps.status.write_diff(status, base=base_status)

            def record_compaction(status: dict) -> dict:
                self._deps.status.roll_date(status, today)
                prior_runs = int(status.get("daily_compactor_runs", 0))
                status["last_compactor_date"] = today
                status["last_compactor_at"] = now.isoformat()
                status["daily_compactor_runs"] = prior_runs + 1
                if distill_error is None:
                    status["last_distill_ok_at"] = now.isoformat()
                else:
                    status["distill_errors_today"] = int(
                        status.get("distill_errors_today", 0)) + 1
                    status["last_distill_error_at"] = now.isoformat()
                    status["last_distill_error"] = distill_error
                return status

            self._deps.status.mutate(record_compaction)
            return result

        def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
            if is_test_isolated_turn(self._deps.current_session_id()):
                return ""
            del messages
            now = datetime.now().astimezone()
            today = now.date().isoformat()

            def record(status: dict) -> dict:
                previous = (int(status.get("pre_compress_count_today", 0))
                            if status.get("pre_compress_date") == today else 0)
                status["pre_compress_date"] = today
                status["last_pre_compress_at"] = now.isoformat()
                status["pre_compress_count_today"] = previous + 1
                return status

            self._deps.status.mutate(record)
            return ""
