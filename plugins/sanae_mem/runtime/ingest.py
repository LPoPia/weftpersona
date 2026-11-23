from __future__ import annotations

import json
import hashlib
import logging
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol

from ..infra.chat_scope import ChatScope
from ..infra.status_store import StatusStore
from ..infra.words import (
    _APPRAISE_CONFLICT_WORDS,
    _APPRAISE_NEGATIVE_WORDS,
    _APPRAISE_POSITIVE_WORDS,
    _APPRAISE_STRONG_AFFECT_WORDS,
    _COMMAND_PREFIXES,
    _DEFAULT_EVENT_TTL_DAYS,
    _DUPLICATE_IMPORTANCE_INCREMENT,
    _EMOTION_WORDS,
    _IMPORTANCE_CEILING,
    _IMPORTANT_WORDS,
    _RELATION_WORDS,
    _SCHEDULE_TTL_DAYS,
    _SCHEDULE_WORDS,
    _TECH_WORDS,
    _TOPIC_WORDS,
    _TRUST_WORDS,
    _WRITE_IMPORTANCE_FLOOR,
    _has_system_meta,
    _noise_shape,
)
from ..infra.textutil import _embed, _fingerprint, _pack_vector, event_keywords
from ..governance import memory_evidence as ms
from ..governance.llm_guard import AuxiliaryModel
from ..storage.event_store import HISTORICAL_LIMIT_NOTICE, SUMMARY_SOURCE_HISTORICAL
from ..storage.curve import CURVE_BASE_DAYS, bounded_half_life, initial_half_life
from ..storage.sqlite_facade import SqliteFacade
from ..cognitive.daily_semantic_validation import strip_self_rule_and_design_intent
from ..governance.source_context import (
    ChatIdentity, current_chat_scope, current_source_record, in_public_interaction, self_attribution,
    source_identity, source_json, user_fact_attribution,
)
from ..governance.shared_memory import (
    SHARED_VIEW_STATUS_COMPONENT, project_shared_view, should_project,
)

logger = logging.getLogger(__name__)


def event_topic(text: str) -> str:
    for topic, words in _TOPIC_WORDS.items():
        if any(word in text for word in words):
            return topic
    return "其他"

def appraise_clause(text: str) -> dict[str, int]:
    def level(groups: Mapping[int, tuple[str, ...]]) -> int:
        return max(
            (value for value, words in groups.items() if any(word in text for word in words)),
            default=0,
        )

    conflict = level(_APPRAISE_CONFLICT_WORDS)

    has_negative = any(word in text for word in _APPRAISE_NEGATIVE_WORDS)
    has_positive = any(word in text for word in _APPRAISE_POSITIVE_WORDS)
    affect = 2 if has_positive else 1 if has_negative else 0
    if affect and (
        any(word in text for word in _APPRAISE_STRONG_AFFECT_WORDS)
        or re.search(r"[！!？?]{2,}", text)
    ):
        affect = 3

    return {
        "a_conflict": conflict,
        "affect": affect,
    }

def event_time_anchor(now: datetime) -> str:
    hour = now.hour
    period = "凌晨" if hour < 6 else "早上" if hour < 9 else "上午" if hour < 12 else "下午" if hour < 18 else "晚上"
    return f"{now:%Y-%m-%d %H:%M}|今天{period}"

class _InsertEventRow(Protocol):
    def __call__(
        self, fingerprint: str, values: tuple, appraisal: dict | None, owner_key: str, *,
        summary: str, source_snapshot: str, source_snapshot_sha256: str, source_context: str,
        experience_subject: str, fact_subject: str, involved_owner_keys: str, read_scope: str,
        half_life_days: float, curve_ttl: str, fulfillment_commitment_id: int | None,
    ) -> int | None: ...


class _DuplicateCandidates(Protocol):
    def __call__(
        self, session_id: str, cutoff: str, *, unscoped: bool, owner_key: str,
        fingerprint: str, body: str, experience_subject: str, fact_subject: str,
        involved_owner_keys: str, read_scope: str,
    ) -> list[sqlite3.Row]: ...


class _RecordOccurrence(Protocol):
    def __call__(
        self, event_id: int, raw: str, observed_at: str,
        source_snapshot: str, source_snapshot_sha256: str, *, preserve_previous: bool = False,
    ) -> bool: ...


class _ReinforceRecall(Protocol):
    def __call__(
        self, ids: list[int], *, via: str, now: datetime, session_id: str,
    ) -> int: ...


class _SetSharedView(Protocol):
    def __call__(self, event_id: int, text: str, *, touch: bool) -> bool: ...


@dataclass(frozen=True)
class IngestDependencies:
    current_data_dir: Callable[[], Path]
    emotion_ttl_days: int
    current_model: Callable[[], AuxiliaryModel | None]
    status: StatusStore
    write_sample: Callable[[str], None]
    record_user_active: Callable[[], None]
    upsert_vector: Callable[[int, str, str, float, str, list[float]], None]
    current_insert_event_row: Callable[[], _InsertEventRow]
    current_find_duplicate_candidates: Callable[[], _DuplicateCandidates]
    current_record_event_occurrence: Callable[[], _RecordOccurrence]
    grounded_protected_ids: Callable[[], set[int]]
    append_event_evidence: Callable[[int, str, str], int]
    reinforce_event: Callable[[int, float, str], sqlite3.Row | None]
    reinforce_expiring: Callable[[int, str, str], int]
    reinforce_on_recall: _ReinforceRecall
    set_shared_view: _SetSharedView
    current_session_id: Callable[[], str]
    group_grant: Callable[[], object | None]
    group_scene_active: Callable[[], bool]
    write_owner: Callable[[], str]
    is_configured_private_owner: Callable[[], bool]


class Ingest:
        def __init__(self, sqlite: SqliteFacade, dependencies: IngestDependencies) -> None:
            self._sqlite = sqlite
            self._deps = dependencies

        def reconfigure(self, *, emotion_ttl_days: int) -> None:
            self._deps = replace(self._deps, emotion_ttl_days=emotion_ttl_days)

        def record_private_turn(
            self, user_text: str, *, current_message: dict[str, object] | None,
            session_id: str, platform: str, author_id: str,
            scope: ChatScope, identity: ChatIdentity,
        ) -> tuple[list[int], bool]:
            turn_source: Mapping[str, str] | None = current_source_record(
                maintainer=identity.maintainer, scope=scope,
            )
            evidence = ms.capture_private_turn(
                user_text, current_message=current_message, session_id=session_id, platform=platform,
                scope=scope, identity=identity, author_id=author_id,
            )
            if evidence is not None:
                turn_source = evidence.source_context
            touch_resident = turn_source is None or turn_source.get("kind") == "maintainer"
            if touch_resident:
                self._deps.record_user_active()
            event_ids: list[int] = []
            for candidate in self._candidate_clauses(str(user_text or "").strip()):
                if not self._is_candidate(candidate):
                    continue
                self._deps.write_sample(candidate)
                event_id = self._sync_candidate(
                    candidate, session_id, appraise_clause(candidate), owner_key=identity.private_owner_key,
                    source_evidence=evidence, source_context=turn_source,
                )
                if isinstance(event_id, int) and not isinstance(event_id, bool):
                    event_ids.append(event_id)
            self.project_shared_views(event_ids, identity.private_owner_key)
            return event_ids, touch_resident

        def project_shared_views(self, event_ids: List[int], owner_key: str) -> None:
            owner = str(owner_key or "")
            if not owner or not event_ids or not self._sqlite.is_open:
                return
            llm = self._deps.current_model()
            attempted = 0
            failed = 0
            for event_id in sorted({int(item) for item in event_ids
                                    if isinstance(item, int) and not isinstance(item, bool)}):
                row = self._sqlite.fetchone(
                    "SELECT raw, summary, shared_view, experience_subject, "
                    "fact_subject, read_scope FROM events "
                    "WHERE id=? AND owner_key=? AND blacklist=0",
                    (event_id, owner),
                )
                if row is None:
                    continue
                if not should_project(
                        owner_key=owner,
                        experience_subject=str(row["experience_subject"] or ""),
                        fact_subject=str(row["fact_subject"] or ""),
                        read_scope=row["read_scope"]):
                    continue
                text = re.sub(r"^\[[^\]]+\]\s*", "", str(row["raw"] or "")).strip()
                text = text or str(row["summary"] or "").strip()
                if not text:
                    continue
                attempted += 1
                projection = project_shared_view(llm, text)
                if not projection.ok:
                    failed += 1
                    self._record_shared_view_failure(projection.error)
                    continue
                if projection.view != str(row["shared_view"] or ""):
                    self._deps.set_shared_view(event_id, projection.view, touch=False)
            if attempted and not failed:
                self._clear_shared_view_failure()

        def _shared_view_status(self) -> dict:
            components = self._deps.status.read().get("components")
            current = (components.get(SHARED_VIEW_STATUS_COMPONENT)
                       if isinstance(components, dict) else None)
            return current if isinstance(current, dict) else {}

        def _record_shared_view_failure(self, error: str) -> None:
            count = self._shared_view_status().get("failures")
            self._deps.status.write_maintenance(
                SHARED_VIEW_STATUS_COMPONENT, ok=False, stage="projection",
                error=error, failures=(count if type(count) is int else 0) + 1)

        def _clear_shared_view_failure(self) -> None:
            current = self._shared_view_status()
            if current.get("ok") is not False:
                return
            count = current.get("failures")
            self._deps.status.write_maintenance(
                SHARED_VIEW_STATUS_COMPONENT, ok=True, stage="projection",
                failures=count if type(count) is int else 0)

        def _is_candidate(self, text: str) -> bool:
            length = len(text)
            if length < 8 or length > 80:
                return False
            if any(text.startswith(prefix) for prefix in _COMMAND_PREFIXES):
                return False
            if _has_system_meta(text):
                return False
            from ..cognitive.daily_semantic_validation import _DESIGN_INTENT_CLAUSE
            if _DESIGN_INTENT_CLAUSE.search(text):
                return False
            if _noise_shape(text):
                return False
            has_signal = any(word in text for word in _TRUST_WORDS + _EMOTION_WORDS + _RELATION_WORDS)
            if any(word in text for word in _TECH_WORDS) and not has_signal:
                return False
            return True

        def _importance(self, text: str) -> float:
            score = 0.2
            if any(word in text for word in _TRUST_WORDS) or any(word in text for word in _EMOTION_WORDS):
                score += 0.3
            if any(word in text for word in _IMPORTANT_WORDS) or any(
                    word in text for word in ("答应", "履行", "按时完成", "按时交付")):
                score += 0.4
            if any(word in text for word in _RELATION_WORDS):
                score += 0.4
            return min(_IMPORTANCE_CEILING, score)

        def _candidate_clauses(self, text: str) -> List[str]:
            normalized = re.sub(r"[\t\r ]+", " ", text).strip()
            if not normalized:
                return []
            from ..cognitive.daily_semantic_validation import _DESIGN_INTENT_CLAUSE
            sentences = [normalized] if len(normalized) <= 80 else [
                part.strip() for part in re.split(r"[。！？!?\n]+", normalized) if part.strip()
            ]
            out: List[str] = []
            for sent in sentences:
                if _DESIGN_INTENT_CLAUSE.search(sent):
                    sub = [
                        p.strip() for p in re.split(r"[，,；;]+", sent)
                        if p.strip() and not _DESIGN_INTENT_CLAUSE.search(p)
                    ]
                    out.extend(sub)
                else:
                    out.append(sent)
            return out

        def _sync_candidate(
            self, text: str, session_id: str, appraisal: dict, owner_key: str = "",
            source_evidence: ms.UserEvidence | None = None, source_context: Optional[Mapping[str, object]] = None,
        ) -> Optional[int]:
            fact_subject = f"owner:{owner_key}" if owner_key else ""
            subject, fact, involved, scope = self_attribution(
                owner_key, fact_subject=fact_subject)
            return self.write_event(
                text, session_id, importance=None, ttl_days=None, appraisal=appraisal,
                owner_key=owner_key, source="auto", source_evidence=source_evidence,
                source_context=source_context, experience_subject=subject,
                fact_subject=fact, involved_owner_keys=involved, read_scope=scope,
            )

        def _ttl_days_for(self, text: str, ttl_days: Any = None) -> int:
            topic = event_topic(text)
            if any(word in text for word in _SCHEDULE_WORDS):
                default_days = _SCHEDULE_TTL_DAYS
            elif topic in {"情绪", "关系"}:
                default_days = self._deps.emotion_ttl_days
            else:
                default_days = _DEFAULT_EVENT_TTL_DAYS
            if ttl_days is None:
                return default_days
            try:
                requested = int(ttl_days)
            except (TypeError, ValueError):
                return default_days
            return max(1, min(requested, default_days))

        def curve_base_days(self, text: str) -> float:
            if any(word in text for word in _SCHEDULE_WORDS):
                return float(_SCHEDULE_TTL_DAYS)
            if event_topic(text) in {"情绪", "关系"}:
                return float(self._deps.emotion_ttl_days)
            return CURVE_BASE_DAYS

        def _curve_write_timing(
            self, text: str, now: datetime, importance: float,
            appraisal: Optional[dict], ttl_days: Any = None,
            ttl_days_exact: Optional[int] = None,
        ) -> tuple:
            affect = int((appraisal or {}).get("affect", 0))
            requested: Optional[int]
            try:
                requested = None if ttl_days is None else int(ttl_days)
            except (TypeError, ValueError):
                requested = None
            if ttl_days_exact is not None:
                days, explicit = int(ttl_days_exact), True
            else:
                days, explicit = self._ttl_days_for(text, requested), requested is not None
            if explicit:
                half_life = bounded_half_life(float(days), "short")
            else:
                half_life = bounded_half_life(
                    initial_half_life(self.curve_base_days(text), importance, affect),
                    "short",
                )
            expire = now + timedelta(days=days)
            return expire, "" if explicit else expire.isoformat(), half_life

        def tool_add(self, args: Mapping[str, object]) -> dict[str, object]:
            content = str(args.get("content") or "").strip()
            if not content:
                raise ValueError("content is required")
            content = strip_self_rule_and_design_intent(content)
            if not content:
                return {"id": None, "stored": False, "reason": "self_rule_or_meta_filtered"}
            session_id = self._deps.current_session_id()
            requested = args.get("importance")
            importance = 0.7 if requested is None else requested
            if (
                self._deps.group_grant() is not None
                or self._deps.group_scene_active()
                or in_public_interaction()
            ):
                raise PermissionError("group chat cannot write memories in-turn")
            write_owner = self._deps.write_owner()
            subject, fact, involved, scope = self_attribution(write_owner)
            event_id = self.write_event(
                content, session_id, importance=importance, ttl_days=args.get("ttl_days"),
                appraisal=None, owner_key=write_owner, source="explicit",
                source_context=current_source_record(
                    maintainer=self._deps.is_configured_private_owner(),
                ),
                experience_subject=subject, fact_subject=fact,
                involved_owner_keys=involved, read_scope=scope,
            )
            if event_id is not None:
                self.project_shared_views([event_id], write_owner)
            return {
                "id": event_id,
                "stored": event_id is not None,
                "reason": "stored" if event_id is not None else "duplicate_below_floor",
            }

        def write_event(
            self, text: str, session_id: str, *, importance: Any = None, ttl_days: Any = None,
            ttl_days_exact: Optional[int] = None,
            appraisal: Optional[dict] = None, owner_key: str = "", source: str = "auto",
            source_evidence: ms.UserEvidence | None = None, source_context: Optional[Mapping[str, object]] = None,
            experience_subject: str = "", fact_subject: str = "",
            involved_owner_keys: str = "", read_scope: str = "",
            fulfillment_commitment_id: Optional[int] = None,
            defer_post_commit: Optional[List[Callable[[], None]]] = None,
        ) -> Optional[int]:
            assert self._sqlite.is_open
            original = str(text or "").strip()
            body = re.sub(r"\s+", " ", original).strip()
            if not body:
                return None
            source_record_json = source_json(source_context)
            if importance is None:
                score = self._importance(body)
            else:
                try:
                    score = float(importance)
                except (TypeError, ValueError):
                    score = self._importance(body)
            score = max(0.0, min(_IMPORTANCE_CEILING, score))
            if source == "assistant_statement":
                score = max(score, _WRITE_IMPORTANCE_FLOOR)
            now = datetime.now().astimezone()
            keywords = event_keywords(body)
            snapshot_json = ""
            snapshot_sha = ""
            if source == "auto" and isinstance(source_evidence, ms.UserEvidence):
                stored = ms.snapshot_source_evidence(source_evidence)
                restored = ms.restore_source_evidence(stored)
                if ms.qualify_user_text(original, restored).allowed:
                    snapshot_json = json.dumps(stored, ensure_ascii=False, sort_keys=True)
                    snapshot_sha = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
            below_floor = score < _WRITE_IMPORTANCE_FLOOR
            topic = event_topic(body)
            ttl, curve_ttl, half_life = self._curve_write_timing(
                body, now, score, appraisal, ttl_days, ttl_days_exact)
            precomputed_vector = (
                _embed(original, data_dir=self._deps.current_data_dir())
                if (not below_floor or source == "explicit")
                else None
            )
            matched_duplicate: Optional[tuple[int, float]] = None
            created_event_id: Optional[int] = None
            created_anchor = ""
            with self._sqlite.transaction():
                duplicate = self._find_duplicate(
                    keywords, session_id, now, source_record_json, owner_key=owner_key, body=body,
                    experience_subject=experience_subject, fact_subject=fact_subject,
                    involved_owner_keys=involved_owner_keys, read_scope=read_scope)
                if duplicate:
                    is_expiring = duplicate["expired_at"] is not None
                    bumped = score if importance is not None else min(
                        _IMPORTANCE_CEILING, score + _DUPLICATE_IMPORTANCE_INCREMENT)
                    if bumped < _WRITE_IMPORTANCE_FLOOR and not is_expiring:
                        return None
                    bumped = min(_IMPORTANCE_CEILING, max(bumped, float(duplicate["importance"]) + 0.1))
                    existing = self._sqlite.fetchone(
                        "SELECT raw,summary_source FROM events WHERE id=?", (int(duplicate["id"]),)
                    )
                    existing_raw = str(existing["raw"]) if existing else ""
                    faded = existing and existing["summary_source"] == "gist_faded"
                    if not faded and not self._deps.current_record_event_occurrence()(
                            int(duplicate["id"]), original, now.isoformat(), snapshot_json, snapshot_sha,
                            preserve_previous=True):
                        return None
                    if not faded and body and body not in existing_raw and duplicate["id"] not in self._deps.grounded_protected_ids():
                        self._deps.append_event_evidence(int(duplicate["id"]), original, now.isoformat())
                    matched_duplicate = (int(duplicate["id"]), bumped)
                else:
                    if below_floor and source != "explicit":
                        return None
                    created_anchor = event_time_anchor(now)
                    created_event_id = self._insert_event(
                        original, topic, score, 0.6, ttl, now, session_id,
                        appraisal if source == "auto" else None, owner_key,
                        source_snapshot=snapshot_json, source_snapshot_sha256=snapshot_sha,
                        source_context=source_record_json,
                        experience_subject=experience_subject, fact_subject=fact_subject,
                        involved_owner_keys=involved_owner_keys, read_scope=read_scope,
                        half_life_days=half_life, curve_ttl=curve_ttl,
                        fulfillment_commitment_id=fulfillment_commitment_id,
                        precomputed_vector=precomputed_vector,
                        upsert_chroma=False,
                    )
                    if created_event_id is not None:
                        self._deps.current_record_event_occurrence()(
                            int(created_event_id), original, now.isoformat(), snapshot_json, snapshot_sha)
            def _run_or_defer(fn: Callable[[], None]) -> None:
                if defer_post_commit is not None:
                    defer_post_commit.append(fn)
                else:
                    fn()

            def _record_write_status(ts: datetime = now) -> None:
                self._deps.status.record_write(ts)

            if matched_duplicate is not None:
                dup_id, bumped = matched_duplicate
                self._reinforce(
                    dup_id, body, bumped, now, session_id,
                    defer_post_commit=defer_post_commit,
                )
                _run_or_defer(_record_write_status)
                return dup_id
            if created_event_id is not None:
                if precomputed_vector is not None:
                    _run_or_defer(
                        lambda eid=created_event_id, doc=body[:150], t=topic, s=score, anchor=created_anchor, vec=precomputed_vector: (
                            self._deps.upsert_vector(eid, doc, t, s, anchor, vec)
                        )
                    )
                if below_floor:
                    _run_or_defer(lambda ts=now: self._count_below_floor(ts))
                else:
                    _run_or_defer(_record_write_status)
            return created_event_id

        def _count_below_floor(self, now: datetime) -> None:
            try:
                def record(status: dict) -> dict:
                    status["events_kept_below_floor"] = int(
                        status.get("events_kept_below_floor", 0)) + 1
                    status["last_below_floor_at"] = now.isoformat()
                    return status

                self._deps.status.mutate(record)
            except (OSError, ValueError, TypeError, OverflowError) as exc:
                logger.warning("sanae_mem: below-floor counter failed (%s)", type(exc).__name__)

        @staticmethod
        def display_text(row_or_item: sqlite3.Row | Mapping[str, object]) -> str:
            historical = False
            try:
                historical = str(row_or_item["summary_source"] or "") == SUMMARY_SOURCE_HISTORICAL
            except (IndexError, KeyError):
                historical = False
            try:
                value = row_or_item["summary"]
            except (IndexError, KeyError):
                value = None
            if value:
                return f"{HISTORICAL_LIMIT_NOTICE} {value}" if historical else str(value)
            try:
                raw = str(row_or_item["raw"])
                return f"{HISTORICAL_LIMIT_NOTICE} {raw}" if historical else raw
            except (IndexError, KeyError):
                return ""


        def _insert_event(
            self,
            content: str,
            topic: str,
            importance: float,
            confidence: float,
            ttl: Optional[datetime],
            now: datetime,
            session_id: str,
            appraisal: Optional[dict] = None,
            owner_key: str = "",
            source_snapshot: str = "",
            source_snapshot_sha256: str = "",
            source_context: str = "",
            experience_subject: str = "",
            fact_subject: str = "",
            involved_owner_keys: str = "",
            read_scope: str = "",
            half_life_days: Optional[float] = None,
            curve_ttl: Optional[str] = None,
            fulfillment_commitment_id: Optional[int] = None,
            precomputed_vector: Optional[List[float]] = None,
            upsert_chroma: bool = True,
        ) -> Optional[int]:
            assert self._sqlite.is_open
            original = str(content or "").strip()
            body = re.sub(r"\s+", " ", original).strip()
            if not body:
                return None
            anchor = event_time_anchor(now)
            raw = f"[{anchor}] {original}"
            summary = body[:150]
            fingerprint = _fingerprint(body)
            vector = (
                precomputed_vector
                if precomputed_vector is not None
                else _embed(original, data_dir=self._deps.current_data_dir())
            )
            if half_life_days is None:
                half_life_days = bounded_half_life(
                    initial_half_life(
                        self.curve_base_days(body), importance,
                        int((appraisal or {}).get("affect", 0))),
                    "short",
                )
            else:
                half_life_days = bounded_half_life(float(half_life_days), "short")
            if curve_ttl is None:
                curve_ttl = ttl.isoformat() if ttl else ""
            insert_event_row = self._deps.current_insert_event_row()
            event_id = insert_event_row(
                fingerprint,
                (
                    anchor, raw, event_keywords(body), importance, confidence, topic,
                    ttl.isoformat() if ttl else None, now.isoformat(), _pack_vector(vector), session_id,
                ),
                appraisal,
                owner_key,
                summary=summary,
                source_snapshot=source_snapshot,
                source_snapshot_sha256=source_snapshot_sha256,
                source_context=source_context,
                experience_subject=experience_subject, fact_subject=fact_subject,
                involved_owner_keys=involved_owner_keys,
                read_scope=read_scope,
                half_life_days=half_life_days,
                curve_ttl=curve_ttl,
                fulfillment_commitment_id=fulfillment_commitment_id,
            )
            if event_id is None:
                return None
            if upsert_chroma:
                self._deps.upsert_vector(event_id, summary, topic, importance, anchor, vector)
            return event_id

        def ingest_grounded_fact(
            self, content: str, *, source_evidence: object, session_id: str,
            owner_key: str, is_maintainer: Callable[[], bool],
        ) -> dict:
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            original = str(content or "").strip()
            body = re.sub(r"\s+", " ", original).strip()
            if not body:
                return {"ok": False, "reason": "empty_content"}
            if not isinstance(source_evidence, ms.UserEvidence):
                return {"ok": False, "reason": "trusted_source_evidence_required"}
            active_session = session_id
            scope = current_chat_scope()
            if (not is_maintainer() or scope.session_id != active_session
                    or not scope.message_id or source_evidence.turn != scope.message_id
                    or source_evidence.user != (scope.actor_alt or scope.actor)
                    or source_evidence.session != active_session):
                return {"ok": False, "reason": "trusted_source_evidence_required"}
            snapshot = ms.snapshot_source_evidence(source_evidence)
            restored = ms.restore_source_evidence(snapshot)
            verdict = ms.qualify_user_text(original, restored)
            if not verdict.allowed:
                return {"ok": False, "reason": "source_unqualified", "detail": verdict.reason[:200]}
            snapshot_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
            snapshot_sha = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
            now = datetime.now().astimezone()
            subject, fact, involved, scope = user_fact_attribution(owner_key)
            importance = self._importance(body)
            ttl, curve_ttl, half_life = self._curve_write_timing(body, now, importance, None)
            event_id = self._insert_event(
                original, event_topic(body), importance, 0.6, ttl, now,
                active_session, None, owner_key,
                source_snapshot=snapshot_json, source_snapshot_sha256=snapshot_sha,
                source_context=source_json(restored.source_context if restored is not None else None),
                experience_subject=subject, fact_subject=fact,
                involved_owner_keys=involved, read_scope=scope,
                half_life_days=half_life, curve_ttl=curve_ttl,
            )
            if event_id is None:
                return {"ok": False, "reason": "event_write_failed"}
            self.project_shared_views([event_id], owner_key)
            return {"ok": True, "id": event_id, "source_snapshot_sha256": snapshot_sha}

        def _find_duplicate(self, keywords: str, session_id: str, now: datetime,
                            source_value: str = "", *, owner_key: str = "", body: str = "",
                            experience_subject: str = "", fact_subject: str = "",
                            involved_owner_keys: str = "", read_scope: str = ""):
            assert self._sqlite.is_open
            terms = set(keywords.split())
            if not terms:
                return None
            cutoff = (now - timedelta(hours=24)).isoformat()
            find_duplicate_candidates = self._deps.current_find_duplicate_candidates()
            rows = find_duplicate_candidates(
                session_id, cutoff, unscoped=False, owner_key=owner_key,
                fingerprint=_fingerprint(body) if body else '', body=body,
                experience_subject=experience_subject, fact_subject=fact_subject,
                involved_owner_keys=involved_owner_keys, read_scope=read_scope,
            )
            wanted = source_identity(source_value)
            for row in rows:
                if source_identity(row["source_context"]) != wanted:
                    continue
                other = set(row["keywords"].split())
                if len(terms & other) >= max(1, min(2, len(terms))):
                    return row
            return None

        def _reinforce(self, event_id: int, content: str, importance: float, now: datetime,
                       session_id: str = "",
                       defer_post_commit: Optional[List[Callable[[], None]]] = None) -> None:
            assert self._sqlite.is_open
            row = self._deps.reinforce_event(event_id, importance, now.isoformat())
            if not row:
                return
            if row["expired_at"] is not None:
                ttl_days = (
                    _SCHEDULE_TTL_DAYS
                    if any(word in content for word in _SCHEDULE_WORDS)
                    else self._deps.emotion_ttl_days
                    if row["topic"] in {"情绪", "关系"}
                    else _DEFAULT_EVENT_TTL_DAYS
                )
                self._deps.reinforce_expiring(
                    event_id, now.isoformat(), (now + timedelta(days=ttl_days)).isoformat()
                )
            self._deps.reinforce_on_recall([event_id], via="chat", now=now, session_id=session_id)
            if defer_post_commit is not None:
                summary_text = self.display_text(row)
                topic_val = row["topic"]
                ts_val = row["ts"]
                defer_post_commit.append(
                    lambda eid=event_id, doc=summary_text, t=topic_val, imp=importance, ts=ts_val, text=content: (
                        self._deps.upsert_vector(
                            eid, doc, t, imp, ts,
                            _embed(text, data_dir=self._deps.current_data_dir()),
                        )
                    )
                )
            else:
                self._deps.upsert_vector(
                    event_id, self.display_text(row), row["topic"], importance, row["ts"],
                    _embed(content, data_dir=self._deps.current_data_dir()),
                )
