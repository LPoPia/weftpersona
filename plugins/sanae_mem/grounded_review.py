from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .source_context import is_maintainer_source, parse_source
except ImportError:
    from source_context import is_maintainer_source, parse_source

logger = logging.getLogger(__name__)


class GroundedReviewMixin:

        def review_grounded_user(self, *, text: str, evidence_ids: List[int],
                                 replaces: str = "", memory_dir: Optional[Path] = None,
                                 now: Optional[datetime] = None) -> dict:
            """Review a USER candidate against persisted, first-person evidence."""
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            now = now or datetime.now().astimezone()
            body = str(text or "")
            if not body.strip():
                return {"ok": False, "reason": "empty_text"}
            if not isinstance(evidence_ids, list) or not (2 <= len(evidence_ids) <= 3):
                return {"ok": False, "reason": "representative_evidence_requires_two_or_three"}
            if any(isinstance(eid, bool) or not isinstance(eid, int) or eid <= 0
                   for eid in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
                return {"ok": False, "reason": "evidence_ids_must_be_distinct"}
            try:
                from tools import memory_source as _ms
            except Exception:
                return {"ok": False, "reason": "source_qualifier_unavailable"}
            resident_user, resident_memory = self._resident_texts(memory_dir)
            if not getattr(self, "_resident_read_ok", False):
                return {"ok": False, "reason": "resident_unavailable"}
            entries = self._resident_entries(resident_user)
            replaces_text = str(replaces or "").strip()
            if replaces_text and replaces_text not in entries:
                return {"ok": False, "reason": "replaced_recognition_missing"}
            items: List[Dict[str, Any]] = []
            evidence_texts: List[str] = []
            speaker_id: Optional[str] = None
            seen_facts = set()
            for eid in sorted(evidence_ids):
                row = self.get_grounded_occurrence(eid)
                if row is None:
                    return {"ok": False, "reason": "evidence_event_missing", "event_id": eid}
                owner = str(getattr(self, "_owner_key", "") or "")
                if owner and str(row["owner_key"] or "") != owner:
                    return {"ok": False, "reason": "evidence_owner_mismatch", "event_id": eid}
                raw = str(row["raw"] or "")
                version = str(row["version"] or "")
                stored_snap = str(row["source_snapshot"] or "")
                stored_sha = str(row["source_snapshot_sha256"] or "")
                if not stored_snap or not stored_sha:
                    return {"ok": False, "reason": "evidence_source_snapshot_missing", "event_id": eid}
                stored_source = str(row.get("source_context") or "")
                if stored_source and not is_maintainer_source(stored_source):
                    return {"ok": False, "reason": "evidence_source_not_maintainer", "event_id": eid,
                            "source_kind": str((parse_source(stored_source) or {}).get("kind") or "")}
                try:
                    snapshot = json.loads(stored_snap)
                except (ValueError, TypeError):
                    return {"ok": False, "reason": "evidence_source_snapshot_unreadable", "event_id": eid}
                restored = _ms.restore_source_evidence(snapshot)
                if restored is None:
                    return {"ok": False, "reason": "evidence_source_snapshot_unqualified", "event_id": eid}
                current_speaker = str(restored.user or "")
                if speaker_id is not None and current_speaker != speaker_id:
                    return {"ok": False, "reason": "mixed_evidence_speakers", "event_id": eid}
                speaker_id = current_speaker
                fact_text = self._candidate_text(raw)
                verdict = _ms.qualify_user_text(fact_text, restored)
                if not (getattr(verdict, "allowed", False) and getattr(verdict, "extracted", False)
                        and getattr(verdict, "affirmed", False)):
                    return {"ok": False, "reason": "evidence_source_unqualified", "event_id": eid}
                fact_key = _ms.normalize_user_text(fact_text)
                if not fact_key or fact_key in seen_facts:
                    return {"ok": False, "reason": "duplicate_source_fact", "event_id": eid}
                seen_facts.add(fact_key)
                items.append({
                    "event_id": eid,
                    "event_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "event_version": version,
                    "source_snapshot_sha256": stored_sha,
                })
                evidence_texts.append(fact_text)
            evidence_group = json.dumps(items, ensure_ascii=False)
            evidence_block = "\n".join(f"【事实证据{i + 1}】{t}" for i, t in enumerate(evidence_texts))
            context_text = "\n".join([
                f"【认识正文】{body}",
                f"【拟修正的旧 USER 认识】{str(replaces or '').strip()}",
                evidence_block,
                f"【当前 USER 常驻】{resident_user}",
                f"【当前 MEMORY 常驻】{resident_memory}",
            ])
            candidate = None
            quota_ok = self._llm_calls_today(now) < max(0, int(self._distill_daily_cap))
            if quota_ok:
                if self._llm is not None:
                    self._note_llm_call(now)
                try:
                    candidate = self._maybe_llm_grounded_review(context_text)
                except Exception as exc:
                    logger.warning("sanae_mem: grounded user review failed (%s)", type(exc).__name__)
            if candidate is None:
                return {"ok": False, "reason": "grounded_review_unavailable", "target": "user"}
            if str(candidate.get("duty") or "") != "user" or not candidate.get("long_term"):
                return {"ok": False, "reason": "not_a_long_term_user_candidate", "target": "user",
                        "duty": str(candidate.get("duty") or "")}
            if " ".join(str(candidate.get("summary") or "").split()) != " ".join(body.split()):
                return {"ok": False, "reason": "recognition_review_text_mismatch"}
            anchor_id = int(items[0]["event_id"])
            covered = str(candidate.get("covered_by") or "")
            if covered and covered not in entries:
                return {"ok": False, "reason": "covered_user_recognition_not_found"}
            if replaces_text and covered != replaces_text:
                return {"ok": False, "reason": "revision_relationship_unverified"}
            content_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            baseline_sha = self._resident_baseline_sha256(resident_user, resident_memory)
            significance = str(candidate.get("significance") or "")
            review_sha = self.put_event_review(
                anchor_id, target="user", duty="user", content_sha256=content_sha,
                event_sha256=str(items[0]["event_sha256"]),
                event_version=str(items[0]["event_version"]),
                baseline_sha256=baseline_sha, source_kind="user_grounded",
                significance=significance, reviewed_text=body, now=now.isoformat(),
                evidence_group=evidence_group)
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO recognition_review_targets"
                    "(review_sha256,covered_sha256) VALUES(?,?)",
                    (review_sha, hashlib.sha256(covered.encode("utf-8")).hexdigest() if covered else ""))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                return {"ok": False, "reason": "review_coverage_binding_failed"}
            return {"ok": True, "review_sha256": review_sha, "baseline_sha256": baseline_sha,
                    "content_sha256": content_sha, "target": "user",
                    "evidence_group": evidence_group, "evidence_group_ids": sorted(
                        {int(item["event_id"]) for item in items}),
                    "anchor_event_id": anchor_id, "significance": significance,
                    "review_action": "merge_evidence" if covered else "propose_new",
                    "covered_by": covered}

        def grounded_binding(self, declared: Dict[str, Any]) -> Optional[dict]:
            """核对审核正文和原始证据；常驻基线由宿主读取实际写入目录来核对。"""
            declared = declared or {}
            anchor = declared.get("event_id")
            review_sha = str(declared.get("review_sha256") or "")
            if isinstance(anchor, bool) or anchor is None or not review_sha:
                return None
            try:
                event_id = int(anchor)
            except (TypeError, ValueError):
                return None
            row = self.get_event_review(event_id, review_sha)
            if row is None:
                return None
            if row["target"] != "user" or row["source_kind"] != "user_grounded":
                return None
            try:
                group = json.loads(str(row["evidence_group"] or ""))
            except (ValueError, TypeError):
                return None
            if not isinstance(group, list) or not 2 <= len(group) <= 3:
                return None
            ordered_ids: List[int] = []
            owner = str(getattr(self, "_owner_key", "") or "")
            for item in group:
                if not isinstance(item, dict):
                    return None
                eid = item.get("event_id")
                if type(eid) is not int or eid <= 0:
                    return None
                if (not ordered_ids and eid != event_id) or (ordered_ids and eid <= ordered_ids[-1]):
                    return None
                ordered_ids.append(eid)
                ev = self.get_grounded_occurrence(eid)
                if ev is None:
                    return None
                if owner and str(ev["owner_key"] or "") != owner:
                    return None
                raw = str(ev["raw"] or "")
                if hashlib.sha256(raw.encode("utf-8")).hexdigest() != str(item.get("event_sha256") or ""):
                    return None
                if str(ev["version"] or "") != str(item.get("event_version") or ""):
                    return None
                snapshot_sha = str(item.get("source_snapshot_sha256") or "")
                if not snapshot_sha or str(ev["source_snapshot_sha256"] or "") != snapshot_sha:
                    return None
                stored_source = str(ev.get("source_context") or "")
                if stored_source and not is_maintainer_source(stored_source):
                    return None
            if declared.get("evidence_group_ids", ordered_ids) != ordered_ids:
                return None
            out = {key: row[key] for key in row.keys()}
            out["evidence_group_ids"] = ordered_ids
            out["verified"] = True
            return out
