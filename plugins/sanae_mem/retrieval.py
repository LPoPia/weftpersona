from __future__ import annotations

import json
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
    from .source_context import (
        SELF_SUBJECT, current_source_record, in_public_interaction, parse_source,
        public_source_view, self_attribution, source_view,
    )
except ImportError:
    from words import *
    from textutil import *
    from schemas import *
    from source_context import (
        SELF_SUBJECT, current_source_record, in_public_interaction, parse_source,
        public_source_view, self_attribution, source_view,
    )

try:
    from .event_store import HISTORICAL_LIMIT_NOTICE, SUMMARY_SOURCE_HISTORICAL, GIST_FADED_NOTICE, SUMMARY_SOURCE_GIST_FADED, gist_provenance
except ImportError:
    from event_store import HISTORICAL_LIMIT_NOTICE, SUMMARY_SOURCE_HISTORICAL, GIST_FADED_NOTICE, SUMMARY_SOURCE_GIST_FADED, gist_provenance

logger = logging.getLogger(__name__)


class RetrievalMixin:
        def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
            try:
                if in_public_interaction() and tool_name not in {"sanae_mem_add", "sanae_mem_search"}:
                    raise PermissionError("tool is unavailable in public interaction")
                if self._conn is None:
                    raise RuntimeError("provider is not initialized")
                if tool_name == "sanae_mem_add":
                    result = self._tool_add(args)
                elif tool_name == "sanae_mem_search":
                    result = self._tool_search(args, session_id=str(kwargs.get("session_id") or self._session_id))
                elif tool_name == "sanae_mem_forget":
                    result = self._tool_forget(args)
                elif tool_name == "sanae_mem_compact":
                    result = self._tool_compact(args)
                elif tool_name == "sanae_mem_marked_promoted":
                    result = self.marked_promoted(args.get("ids") or [], str(args.get("to") or ""))
                elif tool_name == "sanae_mem_distill":
                    result = self.distill()
                elif tool_name == "sanae_mem_followup":
                    result = self._tool_followup(args)
                elif tool_name == "sanae_mem_review":
                    result = self._tool_review(args)
                elif tool_name == "sanae_mem_report_false_trigger":
                    result = {"weekly_count": self.record_false_trigger(), "shadow": self._shadow}
                elif tool_name == "sanae_inner_state":
                    result = self._tool_inner_state(args, **kwargs)
                elif tool_name == "sanae_impression":
                    result = self._tool_impression(args, **kwargs)
                elif tool_name == "sanae_meme_save":
                    result = self._tool_meme_save(args)
                elif tool_name == "sanae_meme_search":
                    result = self._tool_meme_search(args)
                elif tool_name == "sanae_meme_update":
                    result = self._tool_meme_update(args)
                elif tool_name == "sanae_meme_delete":
                    result = self._tool_meme_delete(args)
                else:
                    raise ValueError(f"unknown tool: {tool_name}")
                return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

        def _tool_followup(self, args: Dict[str, Any]) -> dict:
            return {"ok": False, "reason": "followup_retired",
                    "message": "未结与关怀队列已撤掉；经历按普通记忆保存与召回，不再建项或销账。"}

        def _tool_review(self, args: Dict[str, Any]) -> dict:
            mode = str(args.get("mode") or "").strip()
            if mode == "migration":
                text = str(args.get("text") or "")
                target = str(args.get("target") or "")
                if not text.strip() or target not in ("memory", "user"):
                    return {"ok": False, "reason": "migration_review_needs_text_and_target"}
                return self.review_migration_text(text, target)
            raw_id = args.get("event_id")
            if isinstance(raw_id, bool) or raw_id is None:
                return {"ok": False, "reason": "event_id_required"}
            try:
                event_id = int(raw_id)
            except (TypeError, ValueError):
                return {"ok": False, "reason": "event_id_required"}
            return self.review_outcome(event_id)

        def _tool_add(self, args: Dict[str, Any]) -> dict:
            content = str(args.get("content") or "").strip()
            if not content:
                raise ValueError("content is required")
            try:
                from .daily_semantic_validation import strip_self_rule_and_design_intent
            except ImportError:
                from daily_semantic_validation import strip_self_rule_and_design_intent
            content = strip_self_rule_and_design_intent(content)
            if not content:
                return {"id": None, "stored": False, "reason": "self_rule_or_meta_filtered"}
            session_id = self._session_id
            requested = args.get("importance")
            importance = 0.7 if requested is None else requested
            write_owner = self._write_owner()
            subject, fact, involved, scope = self_attribution(write_owner)
            event_id = self._ingest_shared(
                content, session_id, importance=importance, ttl_days=args.get("ttl_days"),
                appraisal=None, owner_key=write_owner, source="explicit",
                source_context=current_source_record(),
                experience_subject=subject, fact_subject=fact,
                involved_owner_keys=involved, read_scope=scope,
            )
            if event_id is not None and in_public_interaction():
                from agent.interaction_context import current_interaction
                review = current_interaction().review_excerpt
                try:
                    if current_interaction().audience == "public" and callable(review) and review(content) is True:
                        self.set_public_excerpt(event_id, content)
                except Exception:
                    pass
            return {
                "id": event_id,
                "stored": event_id is not None,
                "reason": "stored" if event_id is not None else "duplicate_below_floor",
            }

        def _tool_forget(self, args: Dict[str, Any]) -> dict:
            target = str(args.get("target") or "")
            if target and target not in {"user", "memory"}:
                raise ValueError("target must be user or memory")
            event_id = args.get("id")
            query = str(args.get("query") or "").strip()
            if event_id is not None:
                ids = [int(event_id)]
            elif query:
                ids = [item["id"] for item in self._search(
                    {"query": query, "top_k": 10}, internal=True)]
            else:
                raise ValueError("id or query is required")
            assert self._conn is not None
            unique_ids = sorted(set(ids))
            db_ids = self.event_ids_existing(unique_ids)
            deleted = self._delete_events(db_ids)
            return {
                "deleted": deleted,
                "ids": db_ids,
                "db_ids": db_ids,
                "chroma_ids": [f"evt-{event_id}" for event_id in db_ids],
                "blacklisted": deleted,
                "target": target or None,
            }

        def _tool_search(self, args: Dict[str, Any], *, session_id: str = "") -> List[dict]:
            if self._shadow:
                return []
            if in_public_interaction():
                results = self._public_search(args)
                for item in results:
                    item.pop("_semantic", None)
                    item.pop("_embedding_real", None)
                    item.pop("created_at", None)
                return results
            scheduled = _bool(args.get("scheduled"), False)
            if scheduled and (self._read_gate != "scheduled" or not self._scheduled_allowed()):
                return []
            if scheduled:
                results = self._search(
                    args, for_injection=True, with_source=True,
                    with_evidence_limit=True, scoped_owner=self._owner_key or None)
            else:
                results = self._search(
                    args, for_injection=False, with_source=True,
                    with_evidence_limit=True, explicit_history=True,
                    scoped_owner=self._owner_key or None)
                dormant_ids = [r["id"] for r in results if r.get("tier") == "dormant"]
                if dormant_ids:
                    self.reinforce_on_recall(dormant_ids, via="tool_search", session_id=session_id)
            if scheduled and results:
                results = sorted(results, key=lambda item: item["importance"], reverse=True)[:1]
                self._mark_asked(results, "scheduled", self._session_id)
                self._record_scheduled_ask()
            for item in results:
                item.pop("summary", None)
                if item.pop("historical_summary_only", False):
                    item["evidence_limit"] = HISTORICAL_LIMIT_NOTICE
                if item.get("tier") == "dormant":
                    when = self._relative_date(item["created_at"])
                    item["provenance"] = gist_provenance(item, [when] if when else [])
                if item.get("summary_source") == SUMMARY_SOURCE_GIST_FADED:
                    item["evidence_limit"] = GIST_FADED_NOTICE
            return results

        def _public_search(self, args: Dict[str, Any], *,
                           owner_key: Optional[str] = None,
                           scoped: Optional[bool] = None) -> List[dict]:
            assert self._conn is not None
            query = str(args.get("query") or "").strip()
            if not query:
                raise ValueError("query is required")
            top_k = max(1, min(10, int(args.get("top_k", 5))))
            now = datetime.now().astimezone()
            read_scoped = self._owner_scoped if scoped is None else bool(scoped)
            read_owner = self._owner_key if owner_key is None else owner_key
            rows = self.query_public_projections(
                now.isoformat(), unscoped=not read_scoped, owner_key=read_owner,
                cross_owner_self=True)
            from agent.interaction_context import current_interaction
            context = current_interaction()
            if context and context.audience.startswith("dm:"):
                rows = list(rows) + self.query_private_peer_events(
                    now.isoformat(), context.audience[3:], owner_key=read_owner)
            query_tokens = set(_tokens(query) + self._keywords(query).split())
            query_vector, embedding_real = _embed_text(query, query=True)
            scored = []
            for row in rows:
                excerpt = str(row["public_excerpt"] or "")
                if not excerpt:
                    continue
                excerpt_tokens = set(_tokens(excerpt))
                overlap = len(query_tokens & excerpt_tokens) / max(1, len(query_tokens))
                semantic = _cosine(query_vector, _embed_text(excerpt)[0])
                if embedding_real:
                    credible = overlap > 0 or semantic >= _EXPLICIT_INJECT_MIN_SEMANTIC
                else:
                    credible = overlap > 0
                if not credible:
                    continue
                created = datetime.fromisoformat(row["created_at"])
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
                    "source": (source_view if context and context.audience.startswith("dm:") else public_source_view)(parse_source(row["source_context"])),
                    "_semantic": semantic,
                    "_embedding_real": bool(embedding_real),
                }
                for _, row, excerpt, semantic in scored[:top_k]
            ]

        @staticmethod
        def _attribution_for_row(row: Any) -> str:
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

        def _search(
            self,
            args: Dict[str, Any],
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
            if in_public_interaction():
                return self._public_search(args, owner_key=owner_key, scoped=scoped)
            if self._shadow and not internal:
                return []
            assert self._conn is not None
            query = str(args.get("query") or "").strip()
            if not query:
                raise ValueError("query is required")
            top_k = max(1, min(10, int(args.get("top_k", 5))))
            topic = str(args.get("topic") or "")
            days = args.get("days")
            gap_continue = _bool(args.get("gap_continue"), False)
            read_owner = ""
            if writable:
                try:
                    read_owner = (self._association_read_owner()
                                  if owner_key is None else owner_key)
                except PermissionError:
                    return []
            elif scoped_owner is not None:
                read_owner = str(scoped_owner or "")
            now_iso = datetime.now().astimezone().isoformat()
            created_after = (
                (datetime.now().astimezone() - timedelta(days=int(days))).isoformat()
                if days is not None else None
            )
            if gap_continue:
                window_start = (
                    datetime.now().astimezone() - timedelta(days=_GAP_RECALL_WINDOW_DAYS)
                ).isoformat()
                rows = self.query_tier_window(
                    now_iso, window_start, topic, created_after,
                    unscoped=not writable, owner_key=read_owner,
                    read_context=bool(read_owner), writable=writable,
                    experience_subject=experience_subject,
                    fact_subject=fact_subject,
                )
            elif for_injection:
                window_start = (
                    datetime.now().astimezone() - timedelta(days=_INJECTION_RECALL_WINDOW_DAYS)
                ).isoformat()
                rows = self.query_tier_window(
                    now_iso, window_start, topic, created_after,
                    unscoped=not writable, owner_key=read_owner,
                    read_context=bool(read_owner), writable=writable,
                    experience_subject=experience_subject,
                    fact_subject=fact_subject,
                )
            else:
                dormant_owner = (self._association_read_owner()
                                 if explicit_history and self._agent_context == "primary"
                                 and self._association_read_valid() else None)
                rows = self.query_events(
                    now_iso, topic, created_after, include_expiring=internal,
                    dormant_owner=dormant_owner or None,
                    unscoped=not writable, owner_key=read_owner,
                    read_context=bool(read_owner), writable=writable,
                    experience_subject=experience_subject,
                    fact_subject=fact_subject,
                )
            query_tokens = set(_tokens(query) + self._keywords(query).split())
            query_vector, embedding_real = _embed_text(query, query=True)
            chroma_ids = set()
            if self._collection is not None:
                try:
                    result = self._collection.query(query_embeddings=[query_vector], n_results=min(20, max(top_k, 1)))
                    chroma_ids = {int(item.removeprefix("evt-")) for item in (result.get("ids") or [[]])[0]}
                except Exception as exc:
                    logger.warning("sanae_mem: Chroma query failed; using SQLite vectors (%s)", type(exc).__name__)
            now = datetime.now().astimezone()
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
                    scored.append((key, row, relevance, max(0.0, float(semantic))))
            scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
            results = []
            for _, row, relevance, semantic in scored[:top_k]:
                attribution = self._attribution_for_row(row)
                item = {
                    "id": int(row["id"]),
                    "raw": row["raw"],
                    "summary": self._display_text(row),
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
            if with_recall_meta and results:
                grouped = self.query_event_occurrences([item["id"] for item in results])
                for item in results:
                    item["occurrences"] = grouped.get(item["id"], [])
            return results
