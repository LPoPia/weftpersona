from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict

from . import memory_evidence as ms
from .account_binding import current_profile
from .resident_store import joint_baseline, parse_entries
from ..infra.host_bridge import is_test_isolated_turn
from .source_context import is_maintainer_source, parse_source

if TYPE_CHECKING:
    from ..storage.event_store import GroundedOccurrence
    from ..storage.sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundedReviewContext:
    home: Path
    session_id: str
    owner: str
    maintainer_owner: str
    maintainer_platform: str
    maintainer_ids: tuple[str, ...]


class _WriteReview(Protocol):
    def __call__(
        self, event_id: int, *, target: str, duty: str, content_sha256: str,
        event_sha256: str, event_version: str, baseline_sha256: str, source_kind: str,
        significance: str, reviewed_text: str, now: str, evidence_group: str,
    ) -> str: ...


class _EvidenceRef(TypedDict):
    event_id: int
    event_sha256: str
    event_version: str
    source_snapshot_sha256: str


class GroundedReview:
        def __init__(
            self,
            sqlite: SqliteFacade,
            *,
            current_context: Callable[[], GroundedReviewContext],
            read_resident_texts: Callable[[Path | None], tuple[str, str] | None],
            read_owner_keys: Callable[[str], Collection[str]],
            get_occurrence: Callable[[int], GroundedOccurrence | None],
            get_review: Callable[[int, str], sqlite3.Row | None],
            write_review: _WriteReview,
            candidate_text: Callable[[str], str],
            is_bound_speaker: Callable[[str, str, str, str, str], bool],
            review_candidate: Callable[[datetime, str], Mapping[str, object] | None],
        ) -> None:
            self._sqlite = sqlite
            self._current_context = current_context
            self._read_resident_texts = read_resident_texts
            self._read_owner_keys = read_owner_keys
            self._get_occurrence = get_occurrence
            self._get_review = get_review
            self._write_review = write_review
            self._candidate_text = candidate_text
            self._is_bound_speaker = is_bound_speaker
            self._review_candidate = review_candidate

        def review_grounded_user(self, *, text: str, evidence_ids: list[int],
                                 replaces: str = "", memory_dir: Path | None = None,
                                 now: datetime | None = None) -> dict[str, object]:
            """Review a USER candidate against persisted, first-person evidence."""
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            context = self._current_context()
            if is_test_isolated_turn(context.session_id):
                return {"ok": False, "reason": "test_session_isolated"}
            now = now or datetime.now().astimezone()
            body = str(text or "")
            if not body.strip():
                return {"ok": False, "reason": "empty_text"}
            if not isinstance(evidence_ids, list) or not (1 <= len(evidence_ids) <= 3):
                return {"ok": False, "reason": "evidence_ids_requires_one_to_three"}
            if any(isinstance(eid, bool) or not isinstance(eid, int) or eid <= 0
                   for eid in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
                return {"ok": False, "reason": "evidence_ids_must_be_distinct"}
            resident = self._read_resident_texts(memory_dir)
            if resident is None:
                return {"ok": False, "reason": "resident_unavailable"}
            resident_user, resident_memory = resident
            entries = parse_entries(resident_user)
            replaces_text = str(replaces or "").strip()
            if replaces_text and replaces_text not in entries:
                return {"ok": False, "reason": "replaced_recognition_missing"}
            if not context.owner:
                return {"ok": False, "reason": "evidence_owner_unavailable"}
            allowed_owners = self._read_owner_keys(context.owner)
            items: list[_EvidenceRef] = []
            evidence_texts: list[str] = []
            speaker_id: str | None = None
            seen_facts: set[str] = set()
            for eid in sorted(evidence_ids):
                row = self._get_occurrence(eid)
                if row is None:
                    return {"ok": False, "reason": "evidence_event_missing", "event_id": eid}
                if str(row["owner_key"] or "") not in allowed_owners:
                    return {"ok": False, "reason": "evidence_owner_mismatch", "event_id": eid}
                raw = str(row["raw"] or "")
                version = str(row["version"] or "")
                stored_sha = str(row["source_snapshot_sha256"] or "")
                current_speaker, fact_text, reason = self._qualified_occurrence(row, context)
                if current_speaker is None:
                    return {"ok": False, "reason": reason, "event_id": eid}
                if speaker_id is not None and current_speaker != speaker_id:
                    return {"ok": False, "reason": "mixed_evidence_speakers", "event_id": eid}
                speaker_id = current_speaker
                fact_key = ms.normalize_user_text(fact_text)
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
            candidate = self._review_candidate(now, context_text)
            if candidate is None:
                return {"ok": False, "reason": "grounded_review_unavailable", "target": "user"}
            if str(candidate.get("duty") or "") != "user" or not candidate.get("long_term"):
                return {"ok": False, "reason": "not_a_long_term_user_candidate", "target": "user",
                        "duty": str(candidate.get("duty") or "")}
            if " ".join(str(candidate.get("summary") or "").split()) != " ".join(body.split()):
                return {"ok": False, "reason": "recognition_review_text_mismatch"}
            anchor_id = items[0]["event_id"]
            covered = str(candidate.get("covered_by") or "")
            if covered and covered not in entries:
                return {"ok": False, "reason": "covered_user_recognition_not_found"}
            if replaces_text and covered != replaces_text:
                return {"ok": False, "reason": "revision_relationship_unverified"}
            content_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            baseline_sha = joint_baseline(resident_user, resident_memory)
            significance = str(candidate.get("significance") or "")
            try:
                with self._sqlite.transaction() as conn:
                    review_sha = self._write_review(
                        anchor_id, target="user", duty="user", content_sha256=content_sha,
                        event_sha256=items[0]["event_sha256"],
                        event_version=items[0]["event_version"],
                        baseline_sha256=baseline_sha, source_kind="user_grounded",
                        significance=significance, reviewed_text=body, now=now.isoformat(),
                        evidence_group=evidence_group)
                    conn.execute(
                        "INSERT OR REPLACE INTO recognition_review_targets"
                        "(review_sha256,covered_sha256) VALUES(?,?)",
                        (review_sha, hashlib.sha256(covered.encode("utf-8")).hexdigest() if covered else ""))
            except sqlite3.Error as exc:
                logger.warning("sanae_mem: grounded review storage failed (%s)", type(exc).__name__)
                return {"ok": False, "reason": "review_coverage_binding_failed"}
            return {"ok": True, "review_sha256": review_sha, "baseline_sha256": baseline_sha,
                    "content_sha256": content_sha, "target": "user",
                    "evidence_group": evidence_group, "evidence_group_ids": [item["event_id"] for item in items],
                    "anchor_event_id": anchor_id, "significance": significance,
                    "review_action": "merge_evidence" if covered else "propose_new",
                    "covered_by": covered}

        def _occurrence_speaker_key(
            self, row_owner: str, restored: ms.UserEvidence, source: Mapping[str, str] | None,
            context: GroundedReviewContext,
        ) -> str | None:
            anchor = context.maintainer_owner
            if anchor:
                snap_source = source or parse_source(restored.source_context)
                profile = (snap_source or {}).get("profile") or current_profile(context.home)
                scope_id = (snap_source or {}).get("scope_id") or ""
                if self._is_bound_speaker(profile, restored.platform, restored.user, scope_id, anchor):
                    return anchor
                if not source and not scope_id:
                    if row_owner and row_owner in self._read_owner_keys(anchor):
                        identities = context.maintainer_ids
                        platform = context.maintainer_platform
                        if ((not identities or restored.user in identities)
                                and (not platform or restored.platform == platform)):
                            return anchor
                return None
            identities = context.maintainer_ids
            platform = context.maintainer_platform
            if ((identities and restored.user not in identities)
                    or (platform and restored.platform != platform)):
                return None
            return restored.user

        def qualifies_occurrence(self, row: GroundedOccurrence) -> bool:
            speaker, _, _ = self._qualified_occurrence(row, self._current_context())
            return speaker is not None

        def _qualified_occurrence(
            self, row: GroundedOccurrence, context: GroundedReviewContext,
        ) -> tuple[str | None, str, str]:
            stored_source = row["source_context"]
            source = parse_source(stored_source)
            if source and (
                source["certainty"] == "derived_processed"
                or source["kind"].startswith("external_")
            ):
                return None, "", "evidence_source_processed_or_external"
            if stored_source and not is_maintainer_source(stored_source):
                return None, "", "evidence_source_not_maintainer"
            try:
                snapshot = json.loads(str(row["source_snapshot"] or ""))
            except (ValueError, TypeError):
                return None, "", "evidence_source_snapshot_unreadable"
            restored = ms.restore_source_evidence(snapshot)
            if restored is None or not restored.user:
                return None, "", "evidence_source_snapshot_unqualified"
            if source and (source["author_id"] != restored.user or source["visibility"] == "public"
                           or (source["platform"] and source["platform"] != restored.platform)):
                return None, "", "evidence_source_snapshot_unqualified"
            speaker = self._occurrence_speaker_key(row["owner_key"], restored, source, context)
            if not speaker:
                return None, "", "evidence_source_not_maintainer"
            fact_text = self._candidate_text(str(row["raw"] or ""))
            if not ms.qualify_user_text(fact_text, restored).allowed:
                return None, "", "evidence_source_unqualified"
            return speaker, fact_text, ""

        def grounded_binding(self, declared: Mapping[str, object]) -> dict[str, object] | None:
            """Recheck the exact reviewed text and original evidence before resident CAS."""
            declared = declared or {}
            anchor = declared.get("event_id")
            review_sha = str(declared.get("review_sha256") or "")
            if isinstance(anchor, bool) or not isinstance(anchor, (int, float, str, bytes, bytearray)) or not review_sha:
                return None
            try:
                event_id = int(anchor)
            except (TypeError, ValueError):
                return None
            row = self._get_review(event_id, review_sha)
            if row is None:
                return None
            if (row["target"] != "user" or row["duty"] != "user"
                    or row["source_kind"] != "user_grounded"):
                return None
            content = declared.get("content")
            if not isinstance(content, str) or content != row["reviewed_text"]:
                return None
            content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if not content or row["content_sha256"] != content_sha:
                return None
            for key in ("content_sha256", "event_sha256", "event_version", "baseline_sha256"):
                if not row[key] or declared.get(key) != row[key]:
                    return None
            try:
                group = json.loads(str(row["evidence_group"] or ""))
            except (ValueError, TypeError):
                return None
            if not isinstance(group, list) or not 1 <= len(group) <= 3:
                return None
            ordered_ids: list[int] = []
            context = self._current_context()
            if not context.owner:
                return None
            allowed_owners = self._read_owner_keys(context.owner)
            if not isinstance(group[0], dict) or any(
                group[0].get(key) != row[key] for key in ("event_sha256", "event_version")
            ):
                return None
            speakers: set[str] = set()
            facts: set[str] = set()
            for item in group:
                if not isinstance(item, dict):
                    return None
                eid = item.get("event_id")
                if type(eid) is not int or eid <= 0:
                    return None
                if (not ordered_ids and eid != event_id) or (ordered_ids and eid <= ordered_ids[-1]):
                    return None
                ordered_ids.append(eid)
                ev = self._get_occurrence(eid)
                if ev is None:
                    return None
                if str(ev["owner_key"] or "") not in allowed_owners:
                    return None
                raw = str(ev["raw"] or "")
                if hashlib.sha256(raw.encode("utf-8")).hexdigest() != str(item.get("event_sha256") or ""):
                    return None
                if str(ev["version"] or "") != str(item.get("event_version") or ""):
                    return None
                snapshot_sha = str(item.get("source_snapshot_sha256") or "")
                if not snapshot_sha or str(ev["source_snapshot_sha256"] or "") != snapshot_sha:
                    return None
                speaker, fact_text, _ = self._qualified_occurrence(ev, context)
                if speaker is None:
                    return None
                speakers.add(speaker)
                facts.add(ms.normalize_user_text(fact_text))
            if len(speakers) != 1 or len(facts) != len(group):
                return None
            if declared.get("evidence_group_ids") != ordered_ids:
                return None
            out = {key: row[key] for key in row.keys()}
            out["evidence_group_ids"] = ordered_ids
            out["verified"] = True
            return out
