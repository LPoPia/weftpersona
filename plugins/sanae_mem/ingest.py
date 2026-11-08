from __future__ import annotations

import json
import hashlib
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
    from .storage import HISTORICAL_LIMIT_NOTICE, SUMMARY_SOURCE_HISTORICAL
    from .curve import CURVE_BASE_DAYS, bounded_half_life, initial_half_life
    from .daily_inputs import _split_known_meme_notes, flatten_user_text
    from .source_context import (
        current_source_record, self_attribution,
        source_identity, source_json, user_fact_attribution,
    )
except ImportError:
    from words import *
    from textutil import *
    from schemas import *
    from storage import HISTORICAL_LIMIT_NOTICE, SUMMARY_SOURCE_HISTORICAL
    from curve import CURVE_BASE_DAYS, bounded_half_life, initial_half_life
    from daily_inputs import _split_known_meme_notes, flatten_user_text
    from source_context import (
        current_source_record, self_attribution,
        source_identity, source_json, user_fact_attribution,
    )

logger = logging.getLogger(__name__)


class IngestMixin:
        def sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            *,
            session_id: str = "",
            messages: Optional[List[Dict[str, Any]]] = None,
            turn_author: Optional[Dict[str, Any]] = None,
            turn_context: Optional[Dict[str, Any]] = None,
        ) -> None:
            try:
                active_session = session_id or self._session_id
                if (self._conn is not None
                        and (str(self._agent_context) == "cron" or self._is_system_turn(active_session))
                        and str(assistant_content or "").strip()
                        and str(assistant_content or "").strip() not in {"HEARTBEAT_OK", "[SILENT]"}):
                    try:
                        self.resolve_explored_curiosity_threads(str(assistant_content or ""))
                    except Exception as exc:
                        logger.warning(
                            "sanae_mem: curiosity thread resolve failed (%s)", type(exc).__name__)
                if self._agent_context == "cron" and self._conn is not None:
                    synthetic = bool(
                        isinstance(turn_author, dict) and (turn_author.get("is_bot") or turn_author.get("synthetic"))
                        or isinstance(turn_context, dict) and turn_context.get("synthetic")
                    )
                    self._on_cron_heartbeat(session_id=active_session, assistant_content=assistant_content,
                                            synthetic=synthetic)
                if self._agent_context != "primary" or self._conn is None:
                    return
                if self._is_system_turn(session_id or self._session_id):
                    return
                if isinstance(turn_author, dict) and turn_author.get("is_bot"):
                    return
                if not str(assistant_content or "").strip():
                    return
                text = str(user_content or "").strip()
                if turn_context is not None:
                    _, meme_notes = _split_known_meme_notes(
                        flatten_user_text(turn_context.get("user_message")))
                    if meme_notes:
                        text, _ = _split_known_meme_notes(text)
                turn_source = current_source_record()
                author = turn_author if isinstance(turn_author, dict) else {}
                actor = str(author.get("id") or "")
                maintainer = str(getattr(self, "_maintainer_id", "") or "")
                if turn_source is None and (author.get("is_bot") or (actor and maintainer and actor != maintainer)):
                    from agent.interaction_context import source_record
                    turn_source = source_record({"kind": "external_user", "author_id": actor,
                                                 "platform": self._platform})
                source_evidence = self._trusted_turn_source(
                    text, session_id or self._session_id, messages, turn_author, turn_source)
                if turn_source is None or turn_source.get("kind") == "maintainer":
                    self._record_user_active()
                for candidate in self._candidate_clauses(text):
                    if not self._is_candidate(candidate):
                        continue
                    self._write_sample(candidate)
                    appraisal = self._appraise_clause(candidate)
                    self._sync_candidate(
                        candidate, session_id or self._session_id, appraisal,
                        owner_key=self._owner_key, source_evidence=source_evidence,
                        source_context=turn_source,
                    )
                if turn_context is not None:
                    self._record_completed_daily_turn(
                        active_session, assistant_content, messages, turn_context, sender_id=actor)
                if turn_source is None or turn_source.get("kind") == "maintainer":
                    self._update_resident_touch(text, assistant_content, session_id=active_session)
            except Exception as exc:
                logger.warning("sanae_mem: sync_turn failed (%s)", type(exc).__name__)

        def _store_assistant_statement(self, quote: str, session_id: str,
                                       owner_key: str, platform: str) -> Optional[int]:
            ref = "assistant-statement:" + _fingerprint(quote)
            if self._conn is not None:
                row = self._conn.execute(
                    "SELECT id FROM events WHERE owner_key=? AND source_context LIKE ? LIMIT 1",
                    (owner_key, f'%"{ref}"%'),
                ).fetchone()
                if row is not None:
                    return int(row["id"])
            subject, fact, involved, scope = self_attribution(owner_key)
            return self._ingest_shared(
                f"我说过：{quote}", session_id, owner_key=owner_key, source="assistant_statement",
                source_context={"kind": "self", "platform": platform, "visibility": "private",
                                "certainty": "observed", "ref": ref},
                experience_subject=subject, fact_subject=fact,
                involved_owner_keys=involved, read_scope=scope,
            )

        def _trusted_turn_source(self, text: str, session_id: str,
                                 messages: Optional[List[Dict[str, Any]]],
                                 turn_author: Optional[Dict[str, Any]],
                                 source_context: Optional[Dict[str, Any]] = None) -> Any:
            if not isinstance(messages, list) or not messages:
                return None
            if isinstance(turn_author, dict) and turn_author.get("is_bot"):
                return None
            actor = str((turn_author or {}).get("id") or "")
            maintainer = str(getattr(self, "_maintainer_id", "") or "")
            if maintainer and actor and actor != maintainer:
                return None
            if not isinstance(turn_author, dict) and str(getattr(self, "_platform", "")).lower() not in (
                    "cli", "terminal", "desktop"):
                return None
            current = next((m for m in reversed(messages) if isinstance(m, dict)
                            and m.get("role") == "user"), None)
            if current is None or current.get("tool_call_id"):
                return None
            try:
                from tools import memory_source as ms
                source_text = ms._message_text(current.get("content")).strip()
                if not source_text or source_text != str(text or "").strip():
                    return None
                ev = ms.build_user_evidence(
                    [current], session=session_id, user=str((turn_author or {}).get("id") or ""),
                    platform=str(getattr(self, "_platform", "") or ""),
                    origin="foreground", unattended=False, source_context=source_context)
                return ev if ev.available else None
            except Exception:
                return None

        @staticmethod
        def _is_system_turn(session_id: str) -> bool:
            return str(session_id or "").startswith("cron")

        def _is_candidate(self, text: str) -> bool:
            length = len(text)
            if length < 8 or length > 80:
                return False
            if any(text.startswith(prefix) for prefix in _COMMAND_PREFIXES):
                return False
            if _has_system_meta(text):
                return False
            try:
                from .daily_semantic_validation import _DESIGN_INTENT_CLAUSE
            except ImportError:
                from daily_semantic_validation import _DESIGN_INTENT_CLAUSE
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

        def _keywords(self, text: str) -> str:
            values: List[str] = []
            for word in _TRUST_WORDS + _EMOTION_WORDS + _RELATION_WORDS + _IMPORTANT_WORDS + tuple(
                item for group in _TOPIC_WORDS.values() for item in group
            ):
                if word in text and word not in values:
                    values.append(word)
            values.extend(token for token in _tokens(text) if token not in values)
            return " ".join(values[:16])

        def _topic(self, text: str) -> str:
            for topic, words in _TOPIC_WORDS.items():
                if any(word in text for word in words):
                    return topic
            return "其他"

        def _candidate_clauses(self, text: str) -> List[str]:
            normalized = re.sub(r"[\t\r ]+", " ", text).strip()
            if not normalized:
                return []
            try:
                from .daily_semantic_validation import _DESIGN_INTENT_CLAUSE
            except ImportError:
                from daily_semantic_validation import _DESIGN_INTENT_CLAUSE
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

        def _appraise_clause(self, text: str) -> dict:
            def level(groups: dict) -> int:
                return max(
                    (value for value, words in groups.items() if any(word in text for word in words)),
                    default=0,
                )

            intent = level(_APPRAISE_INTENT_WORDS)
            relation = level(_APPRAISE_REL_WORDS)
            self_relevance = level(_APPRAISE_SELF_WORDS)
            if self_relevance < 2 and "你" in text:
                self_relevance = 2
            conflict = level(_APPRAISE_CONFLICT_WORDS)

            has_negative = any(word in text for word in _APPRAISE_NEGATIVE_WORDS)
            has_positive = any(word in text for word in _APPRAISE_POSITIVE_WORDS)
            affect = 2 if has_positive else 1 if has_negative else 0
            if affect and (
                any(word in text for word in _APPRAISE_STRONG_AFFECT_WORDS)
                or re.search(r"[！!？?]{2,}", text)
            ):
                affect = 3

            serious = level(_APPRAISE_SERIOUS_WORDS)
            if (intent or relation or affect) and any(mark in text for mark in "？?"):
                serious = max(serious, 2)
            if len(text) >= 24 and any(word in text for word in _APPRAISE_SERIOUS_CONTEXT):
                serious = max(serious, 3)

            return {
                "a_serious": serious,
                "a_intent": intent,
                "a_rel": relation,
                "a_self": self_relevance,
                "a_conflict": conflict,
                "affect": affect,
            }

        def _sync_candidate(
            self, text: str, session_id: str, appraisal: dict, owner_key: str = "",
            source_evidence: Any = None, source_context: Optional[Dict[str, Any]] = None,
        ) -> Optional[int]:
            fact_subject = f"owner:{owner_key}" if owner_key else ""
            subject, fact, involved, scope = self_attribution(
                owner_key, fact_subject=fact_subject)
            return self._ingest_shared(
                text, session_id, importance=None, ttl_days=None, appraisal=appraisal,
                owner_key=owner_key, source="auto", source_evidence=source_evidence,
                source_context=source_context, experience_subject=subject,
                fact_subject=fact, involved_owner_keys=involved, read_scope=scope,
            )

        def _ttl_days_for(self, text: str, ttl_days: Any = None) -> int:
            topic = self._topic(text)
            if any(word in text for word in _SCHEDULE_WORDS):
                default_days = _SCHEDULE_TTL_DAYS
            elif topic in {"情绪", "关系"}:
                default_days = self._emotion_ttl_days
            else:
                default_days = _DEFAULT_EVENT_TTL_DAYS
            if ttl_days is None:
                return default_days
            try:
                requested = int(ttl_days)
            except (TypeError, ValueError):
                return default_days
            return max(1, min(requested, default_days))

        def _curve_base_days(self, text: str) -> float:
            if any(word in text for word in _SCHEDULE_WORDS):
                return float(_SCHEDULE_TTL_DAYS)
            if self._topic(text) in {"情绪", "关系"}:
                return float(self._emotion_ttl_days)
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
                # 显式 TTL 保持原语义：寿命字段跟明确的到期天数走，不进曲线范围钳制。
                half_life = float(days)
            else:
                half_life = bounded_half_life(
                    initial_half_life(self._curve_base_days(text), importance, affect), "short")
            expire = now + timedelta(days=days)
            return expire, "" if explicit else expire.isoformat(), half_life

        def _ingest_shared(
            self, text: str, session_id: str, *, importance: Any = None, ttl_days: Any = None,
            ttl_days_exact: Optional[int] = None,
            appraisal: Optional[dict] = None, owner_key: str = "", source: str = "auto",
            source_evidence: Any = None, source_context: Optional[Dict[str, Any]] = None,
            experience_subject: str = "", fact_subject: str = "",
            involved_owner_keys: str = "", read_scope: str = "",
        ) -> Optional[int]:
            assert self._conn is not None
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
            keywords = self._keywords(body)
            snapshot_json = ""
            snapshot_sha = ""
            if source == "auto" and source_evidence is not None:
                try:
                    from tools import memory_source as ms
                    if isinstance(source_evidence, ms.UserEvidence):
                        stored = ms.snapshot_source_evidence(source_evidence)
                        restored = ms.restore_source_evidence(stored)
                        verdict = ms.qualify_user_text(original, restored)
                        if restored is not None and verdict.allowed and verdict.extracted and verdict.affirmed:
                            snapshot_json = json.dumps(stored, ensure_ascii=False, sort_keys=True)
                            snapshot_sha = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
                except Exception:
                    pass
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
                existing = self._conn.execute(
                    "SELECT raw,summary_source FROM events WHERE id=?", (int(duplicate["id"]),)
                ).fetchone()
                existing_raw = str(existing["raw"]) if existing else ""
                faded = existing and existing["summary_source"] == "gist_faded"
                if not faded and not self.record_event_occurrence(
                        int(duplicate["id"]), original, now.isoformat(), snapshot_json, snapshot_sha,
                        preserve_previous=True):
                    return None
                if not faded and body and body not in existing_raw and duplicate["id"] not in self._grounded_protected_ids():
                    self.append_event_evidence(int(duplicate["id"]), original, now.isoformat())
                self._reinforce(int(duplicate["id"]), body, bumped, now, session_id)
                self._write_status(now)
                return int(duplicate["id"])
            below_floor = score < _WRITE_IMPORTANCE_FLOOR
            if below_floor and source != "explicit":
                return None
            ttl, curve_ttl, half_life = self._curve_write_timing(
                body, now, score, appraisal, ttl_days, ttl_days_exact)
            event_id = self._insert_event(
                original, self._topic(body), score, 0.6, ttl, now, session_id,
                appraisal if source == "auto" else None, owner_key,
                source_snapshot=snapshot_json, source_snapshot_sha256=snapshot_sha,
                source_context=source_record_json,
                experience_subject=experience_subject, fact_subject=fact_subject,
                involved_owner_keys=involved_owner_keys, read_scope=read_scope,
                half_life_days=half_life, curve_ttl=curve_ttl,
            )
            if event_id is not None:
                self.record_event_occurrence(
                    int(event_id), original, now.isoformat(), snapshot_json, snapshot_sha)
                if below_floor:
                    self._count_below_floor(now)
                else:
                    self._write_status(now, appraised=bool(appraisal and any(appraisal.values())))
            return event_id

        def _count_below_floor(self, now: datetime) -> None:
            try:
                status = self._read_status()
                status["events_kept_below_floor"] = int(status.get("events_kept_below_floor", 0)) + 1
                status["last_below_floor_at"] = now.isoformat()
                self._write_status_payload(status)
            except Exception as exc:
                logger.warning("sanae_mem: below-floor counter failed (%s)", type(exc).__name__)

        @staticmethod
        def _display_text(row_or_item: Any) -> str:
            historical = False
            try:
                historical = str(row_or_item["summary_source"] or "") == SUMMARY_SOURCE_HISTORICAL
            except Exception:
                historical = False
            try:
                value = row_or_item["summary"]
            except Exception:
                value = None
            if value:
                return f"{HISTORICAL_LIMIT_NOTICE} {value}" if historical else str(value)
            try:
                raw = str(row_or_item["raw"])
                return f"{HISTORICAL_LIMIT_NOTICE} {raw}" if historical else raw
            except Exception:
                return ""


        def _time_anchor(self, now: datetime) -> str:
            hour = now.hour
            period = "凌晨" if hour < 6 else "早上" if hour < 9 else "上午" if hour < 12 else "下午" if hour < 18 else "晚上"
            return f"{now:%Y-%m-%d %H:%M}|今天{period}"

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
        ) -> Optional[int]:
            assert self._conn is not None
            original = str(content or "").strip()
            body = re.sub(r"\s+", " ", original).strip()
            if not body:
                return None
            anchor = self._time_anchor(now)
            raw = f"[{anchor}] {original}"
            summary = body[:150]
            fingerprint = _fingerprint(body)
            vector = _embed(original)
            if half_life_days is None:
                half_life_days = bounded_half_life(initial_half_life(
                    self._curve_base_days(body), importance,
                    int((appraisal or {}).get("affect", 0))), "short")
            if curve_ttl is None:
                curve_ttl = ttl.isoformat() if ttl else ""
            event_id = self.insert_event_row(
                fingerprint,
                (
                    anchor, raw, self._keywords(body), importance, confidence, topic,
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
            )
            if event_id is None:
                return None
            self._upsert_chroma(event_id, summary, topic, importance, anchor, vector)
            return event_id

        def ingest_grounded_fact(self, content: str, *, source_evidence, session_id: str = "") -> dict:
            """**内部**（非工具面）grounded 事实写：创建事件的同时验证并持久化来源快照。

            只由宿主 `PluginEventTarget.write(..., source_evidence=...)` 调用；公开
            `sanae_mem_add` 不经过这里（保持 unknown）。`source_evidence` 必须是真正的
            `tools.memory_source.UserEvidence`（isinstance 判定，dict/SimpleNamespace/伪类型一律拒）；
            来源在这里被**重跑** `build_user_evidence`（拒 cron 会话前缀／后台 origin／unattended／
            不可信 platform／合成或工具消息／非维护者的可信出处）再 `qualify_user_text`（整条正文须是用户肯定原话），
            全过才在同一行事件里持久化 source snapshot（绑定 raw 全文 hash/version）。
            写完的 unknown 永不补资格。
            """
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            original = str(content or "").strip()
            body = re.sub(r"\s+", " ", original).strip()
            if not body:
                return {"ok": False, "reason": "empty_content"}
            try:
                from tools import memory_source as _ms
            except Exception:
                return {"ok": False, "reason": "source_qualifier_unavailable"}
            if not isinstance(source_evidence, _ms.UserEvidence):
                return {"ok": False, "reason": "trusted_source_evidence_required"}
            restored = _ms.build_user_evidence(
                list(getattr(source_evidence, "source_messages", ()) or ()),
                session=str(getattr(source_evidence, "session", "") or ""),
                user=str(getattr(source_evidence, "user", "") or ""),
                platform=str(getattr(source_evidence, "platform", "") or ""),
                origin=str(getattr(source_evidence, "origin", "") or ""),
                unattended=bool(getattr(source_evidence, "unattended", False)),
                turn=str(getattr(source_evidence, "turn", "") or ""),
                source_context=getattr(source_evidence, "source_context", None),
            )
            if not getattr(restored, "available", False):
                return {"ok": False, "reason": "source_unqualified",
                        "detail": str(getattr(restored, "reason", "") or "")}
            verdict = _ms.qualify_user_text(original, restored)
            if not (getattr(verdict, "allowed", False) and getattr(verdict, "extracted", False)
                    and getattr(verdict, "affirmed", False)):
                return {"ok": False, "reason": "source_unqualified",
                        "detail": str(getattr(verdict, "reason", "") or "")[:200]}
            snapshot = _ms.snapshot_source_evidence(restored)
            snapshot_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
            snapshot_sha = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
            now = datetime.now().astimezone()
            subject, fact, involved, scope = user_fact_attribution(self._owner_key)
            importance = self._importance(body)
            ttl, curve_ttl, half_life = self._curve_write_timing(body, now, importance, None)
            event_id = self._insert_event(
                original, self._topic(body), importance, 0.6, ttl, now,
                active_session, None, self._owner_key,
                source_snapshot=snapshot_json, source_snapshot_sha256=snapshot_sha,
                source_context=source_json(getattr(restored, "source_context", None)),
                experience_subject=subject, fact_subject=fact,
                involved_owner_keys=involved, read_scope=scope,
                half_life_days=half_life, curve_ttl=curve_ttl,
            )
            if event_id is None:
                return {"ok": False, "reason": "event_write_failed"}
            return {"ok": True, "id": event_id, "source_snapshot_sha256": snapshot_sha}

        def _find_duplicate(self, keywords: str, session_id: str, now: datetime,
                            source_value: Any = "", *, owner_key: str = "", body: str = "",
                            experience_subject: str = "", fact_subject: str = "",
                            involved_owner_keys: str = "", read_scope: str = ""):
            assert self._conn is not None
            terms = set(keywords.split())
            if not terms:
                return None
            cutoff = (now - timedelta(hours=24)).isoformat()
            rows = self.find_duplicate_candidates(
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
                       session_id: str = "") -> None:
            assert self._conn is not None
            row = self.reinforce_event(event_id, importance, now.isoformat())
            if not row:
                return
            if row["expired_at"] is not None:
                ttl_days = (
                    _SCHEDULE_TTL_DAYS
                    if any(word in content for word in _SCHEDULE_WORDS)
                    else self._emotion_ttl_days
                    if row["topic"] in {"情绪", "关系"}
                    else _DEFAULT_EVENT_TTL_DAYS
                )
                self.reinforce_expiring(
                    event_id, now.isoformat(), (now + timedelta(days=ttl_days)).isoformat()
                )
            self.reinforce_on_recall([event_id], via="chat", now=now, session_id=session_id)
            self._upsert_chroma(event_id, self._display_text(row), row["topic"], importance, row["ts"], _embed(content))
