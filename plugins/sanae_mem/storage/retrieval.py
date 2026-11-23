from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Protocol, TypedDict

from ..infra.words import (
    _CHROMA_BONUS_MIN_COSINE,
    _EXPLICIT_INJECT_MIN_SEMANTIC,
    _GAP_RECALL_WINDOW_DAYS,
    _INJECTION_RECALL_WINDOW_DAYS,
    _RECALL_CHROMA_BONUS,
    _RECALL_LEXICAL_WEIGHT,
    _RECALL_RECENCY_DECAY_DAYS,
    _RECALL_RECENCY_WEIGHT,
    _RECALL_SEMANTIC_WEIGHT,
)
from ..infra.textutil import _bool, _cosine, _embed_text, _tokens, _unpack_vector
from ..infra.event_fields import SELF_SUBJECT, parse_source, public_source_view, source_view
from ..infra.chat_scope import current_chat_scope, in_public_interaction

from .event_store import HISTORICAL_LIMIT_NOTICE, SUMMARY_SOURCE_HISTORICAL, GIST_FADED_NOTICE, SUMMARY_SOURCE_GIST_FADED, gist_provenance
from .storage import chroma_backend_errors, is_chroma_backend_failure
from .sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)


class _GroupGrant(Protocol):
    @property
    def person_key(self) -> str: ...
    @property
    def write_owner(self) -> str: ...
    @property
    def source(self) -> Mapping[str, str]: ...


class _GroupEventWriter(Protocol):
    def __call__(
        self, text: str, session_id: str, *, importance: object, ttl_days: object,
        appraisal: None, owner_key: str, source: str, source_context: Mapping[str, str],
        experience_subject: str, fact_subject: str, involved_owner_keys: str, read_scope: str,
    ) -> int | None: ...


class _EventQuery(Protocol):
    def __call__(
        self, now: str, topic: str, created_after: str | None, /, *,
        include_expiring: bool, dormant_owner: str | None, unscoped: bool,
        owner_key: str | None, read_context: bool,
        experience_subject: str | None, fact_subject: str | None,
    ) -> list[sqlite3.Row]: ...


class _TierWindowQuery(Protocol):
    def __call__(
        self, now: str, window_start: str, topic: str, created_after: str | None, /, *,
        unscoped: bool, owner_key: str | None, read_context: bool,
        experience_subject: str | None, fact_subject: str | None,
    ) -> list[sqlite3.Row]: ...


class _SharedViewQuery(Protocol):
    def __call__(
        self, now: str, owner_key: str | None, /, *, window_start: str | None,
        created_after: str | None, topic: str, asked_before: str | None,
    ) -> list[sqlite3.Row]: ...


class _ReinforceRecall(Protocol):
    def __call__(self, ids: list[int], /, *, via: str, session_id: str) -> int: ...


class _VideoQuery(Protocol):
    def __call__(self, query: str, /, *, read_owner: str, limit: int) -> list[dict[str, Any]]: ...


class _VectorMatches(TypedDict, total=False):
    ids: list[list[str]] | None


class _VectorCollection(Protocol):
    def query(
        self, *, query_embeddings: list[list[float]], n_results: int,
        where: dict[str, dict[str, list[int]]],
    ) -> _VectorMatches: ...


@dataclass(frozen=True, kw_only=True)
class RetrievalDependencies:
    sqlite: SqliteFacade
    current_data_dir: Callable[[], Path]
    current_collection: Callable[[], _VectorCollection | None]
    current_session_id: Callable[[], str]
    current_owner: Callable[[], str]
    is_primary: Callable[[], bool]
    shadow_enabled: Callable[[], bool]
    group_grant: Callable[[], _GroupGrant | None]
    group_scene_active: Callable[[], bool]
    private_chat_owner: Callable[[], str]
    free_think_owner: Callable[[], str]
    association_read_owner: Callable[[], str]
    association_read_valid: Callable[[], bool]
    write_owner: Callable[[], str]
    scheduled_allowed: Callable[[], bool]
    mark_asked: Callable[[list[dict], str, str], None]
    record_scheduled_ask: Callable[[], None]
    relative_date: Callable[[str], str]
    shared_topic_label: str
    keywords: Callable[[str], str]
    display_text: Callable[[sqlite3.Row], str]
    query_events: _EventQuery
    query_tier_window: _TierWindowQuery
    query_shared_views: _SharedViewQuery
    query_public_projections: Callable[[str], list[sqlite3.Row]]
    query_event_occurrences: Callable[[list[int]], dict[int, list[dict]]]
    writable_event_ids: Callable[[list[int], str], list[int]]
    delete_events: Callable[[list[int]], int]
    reinforce_on_recall: _ReinforceRecall
    video_memory_items: _VideoQuery
    write_group_event: _GroupEventWriter


class Retrieval:
        def __init__(self, deps: RetrievalDependencies) -> None:
            self._deps = deps

        def _forget_owner(self) -> str:
            try:
                owner = str(self._deps.write_owner() or "")
            except PermissionError:
                owner = ""
            if not owner:
                raise PermissionError("forget requires a verified current owner")
            return owner

        def tool_forget(self, args: Mapping[str, Any]) -> dict:
            target = str(args.get("target") or "")
            if target and target not in {"user", "memory"}:
                raise ValueError("target must be user or memory")
            owner = self._forget_owner()
            event_id = args.get("id")
            query = str(args.get("query") or "").strip()
            if event_id is not None:
                ids = [int(event_id)]
            elif query:
                ids = [item["id"] for item in self.search(
                    {"query": query, "top_k": 10}, internal=True)
                    if isinstance(item.get("id"), int) and not isinstance(item.get("id"), bool)]
            else:
                raise ValueError("id or query is required")
            assert self._deps.sqlite.is_open
            unique_ids = sorted(set(ids))
            db_ids = self._deps.writable_event_ids(unique_ids, owner)
            deleted = self._deps.delete_events(db_ids)
            return {
                "deleted": deleted,
                "ids": db_ids,
                "db_ids": db_ids,
                "chroma_ids": [f"evt-{event_id}" for event_id in db_ids],
                "blacklisted": deleted,
                "target": target or None,
            }

        def tool_search(self, args: Mapping[str, Any], *, session_id: str = "",
                        free_think_owner: str = "") -> List[dict]:
            if self._deps.shadow_enabled():
                return []
            group_grant = self._deps.group_grant()
            if in_public_interaction() and not free_think_owner and group_grant is None:
                results = self.public_search(args)
                for item in results:
                    item.pop("_semantic", None)
                    item.pop("_embedding_real", None)
                    item.pop("created_at", None)
                return results
            scheduled = (
                False
                if (group_grant is not None or self._deps.group_scene_active() or in_public_interaction())
                else _bool(args.get("scheduled"), False)
            )
            if scheduled and not self._deps.scheduled_allowed():
                return []
            if scheduled:
                results = self.search(
                    args, for_injection=True, with_source=True,
                    with_evidence_limit=True, scoped_owner=free_think_owner or self._deps.current_owner() or None)
            else:
                results = self.search(
                    args, for_injection=False, with_source=True,
                    with_evidence_limit=True, explicit_history=True,
                    scoped_owner=free_think_owner or self._deps.current_owner() or None)
                dormant_ids = [r["id"] for r in results if r.get("tier") == "dormant"]
                if dormant_ids:
                    self._deps.reinforce_on_recall(dormant_ids, via="tool_search", session_id=session_id)
            if scheduled and results:
                results = sorted(results, key=lambda item: item["importance"], reverse=True)[:1]
                self._deps.mark_asked(results, "scheduled", self._deps.current_session_id())
                self._deps.record_scheduled_ask()
            for item in results:
                item.pop("summary", None)
                if item.get("shared"):
                    view = str(item.get("shared_view") or "")
                    item["raw"] = self._deps.shared_topic_label + view
                    item["attribution"] = "shared_experience"
                    item.pop("source_context", None)
                    item.pop("occurrences", None)
                    item.pop("fact_subject", None)
                    item.pop("experience_subject", None)
                    item.pop("session_id", None)
                    item.pop("historical_summary_only", None)
                    continue
                if item.pop("historical_summary_only", False):
                    item["evidence_limit"] = HISTORICAL_LIMIT_NOTICE
                if item.get("tier") == "dormant":
                    when = self._deps.relative_date(item["created_at"])
                    item["provenance"] = gist_provenance(item, [when] if when else [])
                if item.get("summary_source") == SUMMARY_SOURCE_GIST_FADED:
                    item["evidence_limit"] = GIST_FADED_NOTICE
            return results

        def public_search(self, args: Mapping[str, Any]) -> List[dict]:
            if current_chat_scope().visibility != "public":
                return []
            assert self._deps.sqlite.is_open
            query = str(args.get("query") or "").strip()
            if not query:
                raise ValueError("query is required")
            top_k = max(1, min(10, int(args.get("top_k", 5))))
            now = datetime.now().astimezone()
            rows = self._deps.query_public_projections(now.isoformat())
            query_tokens = set(_tokens(query) + self._deps.keywords(query).split())
            query_vector, embedding_real = _embed_text(query, query=True, data_dir=self._deps.current_data_dir())
            scored = []
            for row in rows:
                excerpt = str(row["public_excerpt"] or "")
                if not excerpt:
                    continue
                excerpt_tokens = set(_tokens(excerpt))
                overlap = len(query_tokens & excerpt_tokens) / max(1, len(query_tokens))
                semantic = _cosine(query_vector, _embed_text(excerpt, data_dir=self._deps.current_data_dir())[0])
                if embedding_real:
                    credible = overlap > 0 or semantic >= _EXPLICIT_INJECT_MIN_SEMANTIC
                else:
                    credible = overlap > 0
                if not credible:
                    continue
                created = datetime.fromisoformat(row["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=now.tzinfo)
                age_days = max(0.0, (now - created).total_seconds() / 86400)
                recency = math.exp(-age_days / _RECALL_RECENCY_DECAY_DAYS)
                score = (_RECALL_LEXICAL_WEIGHT * overlap
                         + _RECALL_SEMANTIC_WEIGHT * max(0.0, semantic)
                         + _RECALL_RECENCY_WEIGHT * recency)
                scored.append(((score, overlap, recency, int(row["id"])), row, excerpt,
                               max(0.0, semantic)))
            scored.sort(key=lambda entry: entry[0], reverse=True)
            return [
                {
                    "id": int(row["id"]),
                    "excerpt": excerpt,
                    "created_at": str(row["created_at"] or ""),
                    "source": public_source_view(parse_source(row["source_context"])),
                    "_semantic": semantic,
                    "_embedding_real": bool(embedding_real),
                }
                for _, row, excerpt, semantic in scored[:top_k]
            ]

        @staticmethod
        def _event_result_id(item: dict) -> Optional[int]:
            if item.get("shared") or str(item.get("source_table") or "event") != "event":
                return None
            value = item.get("id")
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            return value

        @staticmethod
        def _attribution_for_row(row: sqlite3.Row) -> str:
            experience = str(row["experience_subject"] or "")
            fact = str(row["fact_subject"] or "")
            if experience == SELF_SUBJECT and fact == SELF_SUBJECT:
                return "self"
            if experience == SELF_SUBJECT and fact.startswith("owner:"):
                return "user_statement"
            if experience.startswith("owner:") and fact == experience:
                return "user_fact"
            if experience.startswith("owner:") or fact.startswith("owner:"):
                return "user_fact"
            return "unknown"

        def add_group_event(self, text: str, *, importance: object = 0.5, ttl_days: object = None,
                            source: str = "explicit") -> Optional[int]:
            """Store one event inside the current group scene; ``None`` when not authorized.

            The row keeps the group owner from this turn's grant and the group provenance of the
            utterance. It never becomes a shared view, a public excerpt or a private row.
            """
            grant = self._deps.group_grant()
            if grant is None:
                return None
            involved = json.dumps([grant.person_key], ensure_ascii=False) if grant.person_key else ""
            fact = f"owner:{grant.person_key}" if grant.person_key else ""
            read_scope = json.dumps({"kind": "group", "owner": grant.write_owner},
                                    ensure_ascii=False, sort_keys=True)
            return self._deps.write_group_event(
                text, self._deps.current_session_id(), importance=importance, ttl_days=ttl_days,
                appraisal=None, owner_key=grant.write_owner, source=source,
                source_context=dict(grant.source),
                experience_subject=SELF_SUBJECT, fact_subject=fact,
                involved_owner_keys=involved, read_scope=read_scope,
            )

        def search(
            self,
            args: Mapping[str, Any],
            internal: bool = False,
            for_injection: bool = False,
            with_relevance: bool = False,
            with_source: bool = False,
            with_recall_meta: bool = False,
            with_evidence_limit: bool = False,
            owner_key: Optional[str] = None,
            scoped: Optional[bool] = None,
            experience_subject: Optional[str] = None,
            fact_subject: Optional[str] = None,
            writable: bool = False,
            explicit_history: bool = False,
            scoped_owner: Optional[str] = None,
        ) -> List[dict]:
            group_grant = self._deps.group_grant()
            if in_public_interaction() and group_grant is None and not (
                scoped_owner and scoped_owner == self._deps.free_think_owner()
            ):
                return self.public_search(args)
            if current_chat_scope().visibility == "private" and not self._deps.private_chat_owner():
                return []
            if self._deps.shadow_enabled() and not internal:
                return []
            assert self._deps.sqlite.is_open
            query = str(args.get("query") or "").strip()
            if not query:
                raise ValueError("query is required")
            top_k = max(1, min(10, int(args.get("top_k", 5))))
            topic = str(args.get("topic") or "")
            days = args.get("days")
            gap_continue = _bool(args.get("gap_continue"), False)
            read_owner: str | None
            if group_grant is not None:
                # A group scene reads its own owners: not the private owner set, not the public
                # excerpt projection, and never another group or a sibling thread.
                read_owner = None
            elif writable:
                try:
                    read_owner = (self._deps.association_read_owner()
                                  if owner_key is None else owner_key)
                except PermissionError:
                    return []
            elif scoped_owner is not None:
                read_owner = str(scoped_owner or "")
            else:
                try:
                    read_owner = self._deps.association_read_owner()
                except PermissionError:
                    return []
            now = datetime.now().astimezone()
            now_iso = now.isoformat()
            created_after = (
                (now - timedelta(days=int(days))).isoformat()
                if days is not None else None
            )
            asked_before = (
                (now - timedelta(hours=24)).isoformat()
                if (gap_continue or for_injection) else None
            )
            window_start: Optional[str] = None
            if gap_continue:
                window_start = (now - timedelta(days=_GAP_RECALL_WINDOW_DAYS)).isoformat()
                rows = self._deps.query_tier_window(
                    now_iso, window_start, topic, created_after,
                    unscoped=False, owner_key=read_owner,
                    read_context=True,
                    experience_subject=experience_subject,
                    fact_subject=fact_subject,
                )
            elif for_injection:
                window_start = (now - timedelta(days=_INJECTION_RECALL_WINDOW_DAYS)).isoformat()
                rows = self._deps.query_tier_window(
                    now_iso, window_start, topic, created_after,
                    unscoped=False, owner_key=read_owner,
                    read_context=True,
                    experience_subject=experience_subject,
                    fact_subject=fact_subject,
                )
            else:
                dormant_owner = (self._deps.association_read_owner()
                                 if group_grant is None and explicit_history
                                 and self._deps.is_primary()
                                 and self._deps.association_read_valid() else None)
                rows = self._deps.query_events(
                    now_iso, topic, created_after, include_expiring=internal,
                    dormant_owner=dormant_owner or None,
                    unscoped=False, owner_key=read_owner,
                    read_context=True,
                    experience_subject=experience_subject,
                    fact_subject=fact_subject,
                )
            if group_grant is not None or self._deps.group_scene_active():
                shared_rows = []
            else:
                shared_rows = self._deps.query_shared_views(
                    now_iso, read_owner, window_start=window_start,
                    created_after=created_after, topic=topic,
                    asked_before=asked_before,
                )
            query_tokens = set(_tokens(query) + self._deps.keywords(query).split())
            query_vector, embedding_real = _embed_text(query, query=True, data_dir=self._deps.current_data_dir())
            own_ids = {int(row["id"]) for row in rows}
            chroma_ids = set()
            collection = self._deps.current_collection()
            if collection is not None and own_ids:
                try:
                    result = collection.query(
                        query_embeddings=[query_vector],
                        n_results=min(20, max(top_k, 1)),
                        where={"event_id": {"$in": sorted(own_ids)}},
                    )
                except chroma_backend_errors() as exc:
                    if not is_chroma_backend_failure(exc):
                        raise
                    logger.warning("sanae_mem: Chroma query failed; using SQLite vectors (%s)", type(exc).__name__)
                else:
                    chroma_ids = {
                        int(item.removeprefix("evt-"))
                        for item in (result.get("ids") or [[]])[0]
                    } & own_ids
            scored = []
            for row in rows:
                row_tokens = set(_tokens(row["raw"]) + row["keywords"].split())
                try:
                    row_tokens |= set(_tokens(row["summary"] or ""))
                except Exception:
                    pass
                lexical = len(query_tokens & row_tokens) / max(1, len(query_tokens))
                semantic = _cosine(query_vector, _unpack_vector(row["vec"]))
                if row["tier"] == "dormant":
                    gist_words = set(_tokens(row["summary"] or ""))
                    matches = len(set(_tokens(query)) & gist_words)
                    if matches < (2 if embedding_real else 3) or (
                            embedding_real and semantic < _EXPLICIT_INJECT_MIN_SEMANTIC):
                        continue
                active = datetime.fromisoformat(row["last_active_at"] or row["updated_at"] or row["created_at"])
                if active.tzinfo is None:
                    active = active.replace(tzinfo=now.tzinfo)
                idle_days = max(0.0, (now - active).total_seconds() / 86400)
                recency = math.exp(-idle_days / max(_RECALL_RECENCY_DECAY_DAYS, row["half_life_days"]))
                chroma_bonus = (
                    _RECALL_CHROMA_BONUS
                    if (int(row["id"]) in chroma_ids and semantic >= _CHROMA_BONUS_MIN_COSINE)
                    else 0.0
                )
                relevance = _RECALL_LEXICAL_WEIGHT * lexical + _RECALL_SEMANTIC_WEIGHT * max(0.0, semantic) + chroma_bonus
                score = relevance + _RECALL_RECENCY_WEIGHT * recency
                if gap_continue or relevance > 0:
                    tier_priority = int(idle_days <= 30 and row["tier"] in {"important", "long", "distilled"})
                    key = (tier_priority, score, float(row["importance"])) if gap_continue else score
                    scored.append((key, row, relevance, max(0.0, float(semantic)), False))
            for row in shared_rows:
                view = str(row["shared_view"] or "")
                if not view:
                    continue
                row_tokens = set(_tokens(view))
                lexical = len(query_tokens & row_tokens) / max(1, len(query_tokens))
                semantic = _cosine(query_vector, _embed_text(view, data_dir=self._deps.current_data_dir())[0])
                created = datetime.fromisoformat(row["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=now.tzinfo)
                idle_days = max(0.0, (now - created).total_seconds() / 86400)
                recency = math.exp(-idle_days / _RECALL_RECENCY_DECAY_DAYS)
                relevance = (_RECALL_LEXICAL_WEIGHT * lexical
                             + _RECALL_SEMANTIC_WEIGHT * max(0.0, semantic))
                score = relevance + _RECALL_RECENCY_WEIGHT * recency
                if gap_continue or relevance > 0:
                    key = (0, score, 0.0) if gap_continue else score
                    scored.append((key, row, relevance, max(0.0, float(semantic)), True))
            scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
            results = []
            for _, row, relevance, semantic, shared in scored[:top_k]:
                if shared:
                    view = str(row["shared_view"] or "")
                    item = {
                        "id": int(row["id"]),
                        "raw": "",
                        "summary": "",
                        "topic": "",
                        "importance": 0.0,
                        "ts": "",
                        "created_at": str(row["created_at"] or ""),
                        "session_id": "",
                        "experience_subject": "",
                        "fact_subject": "",
                        "attribution": "shared_experience",
                        "shared": True,
                        "shared_view": view,
                    }
                    if with_relevance:
                        item["relevance"] = float(relevance)
                        item["semantic"] = float(semantic)
                        item["embedding_real"] = bool(embedding_real)
                    if with_recall_meta:
                        item.update(asked=0, asked_at="", updated_at="", keywords="",
                                    tier="", summary_source="", occurrences=[])
                    if with_evidence_limit:
                        item["historical_summary_only"] = False
                    results.append(item)
                    continue
                attribution = self._attribution_for_row(row)
                item = {
                    "id": int(row["id"]),
                    "raw": row["raw"],
                    "summary": self._deps.display_text(row),
                    "topic": row["topic"],
                    "importance": float(row["importance"]),
                    "ts": row["ts"],
                    "created_at": row["created_at"],
                    "session_id": str(row["session_id"] or ""),
                    "experience_subject": str(row["experience_subject"] or ""),
                    "fact_subject": str(row["fact_subject"] or ""),
                    "attribution": attribution,
                }
                if row["summary_source"] == SUMMARY_SOURCE_GIST_FADED:
                    item["summary_source"] = SUMMARY_SOURCE_GIST_FADED
                    item["tier"] = row["tier"]
                if with_source:
                    item["source_context"] = source_view(
                        parse_source(row["source_context"]))
                if gap_continue:
                    item.update(tier=row["tier"])
                if with_relevance:
                    item["relevance"] = float(relevance)
                    item["semantic"] = float(semantic)
                    item["embedding_real"] = bool(embedding_real)
                if with_recall_meta:
                    keys = row.keys()
                    item["asked"] = int(row["asked"] or 0) if "asked" in keys else 0
                    item["asked_at"] = str(row["asked_at"] or "") if "asked_at" in keys else ""
                    item["updated_at"] = str(row["updated_at"] or "") if "updated_at" in keys else ""
                    item["keywords"] = str(row["keywords"] or "")
                    item["tier"] = str(row["tier"] or "") if "tier" in keys else ""
                    item["summary_source"] = (
                        str(row["summary_source"] or "") if "summary_source" in keys else "")
                if with_evidence_limit:
                    keys = row.keys()
                    item["historical_summary_only"] = (
                        "summary_source" in keys
                        and str(row["summary_source"] or "") == SUMMARY_SOURCE_HISTORICAL)
                results.append(item)
            if len(results) < top_k and read_owner is not None and not self._deps.group_scene_active():
                try:
                    video_items = self._deps.video_memory_items(
                        query, read_owner=read_owner, limit=top_k - len(results))
                except Exception as exc:
                    logger.warning("sanae_mem: video recall skipped (%s)", type(exc).__name__)
                    video_items = []
                results.extend(video_items)
            if with_recall_meta and results:
                event_ids = [event_id for item in results
                             if (event_id := self._event_result_id(item)) is not None]
                grouped = self._deps.query_event_occurrences(event_ids)
                for item in results:
                    event_id = self._event_result_id(item)
                    item["occurrences"] = (
                        grouped.get(event_id, []) if event_id is not None else [])
            return results
