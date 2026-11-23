from __future__ import annotations

import json
import logging
import math
import re
import sqlite3

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from ..infra.archive import _append_fsynced_jsonl
from .curve import HALF_LIFE_MIN_DAYS, half_life_ceiling
from .event_store import SUMMARY_SOURCE_DISTILLED, SUMMARY_SOURCE_GIST_FADED
from .sqlite_facade import SqliteFacade
from .storage import chroma_backend_errors, is_chroma_backend_failure
from ..infra.event_fields import exclude_group_rows
from ..infra.textutil import _candidate_text, _embed, _pack_vector, _fingerprint, event_keywords

logger = logging.getLogger(__name__)

_CURVE_VIVID_FLOOR = 0.35
_CURVE_DORMANT_FLOOR = 0.15
_CURVE_DORMANT_IMPORTANCE = 0.6
_CURVE_DEFAULT_HALF_LIFE_DAYS = 14.0

class _EventScopes(Protocol):
    def private_owner_write_clause(self) -> Optional[tuple[str, list[str]]]: ...
    def read_scope_clause(self) -> tuple[str, list[str]]: ...
    def group_write_clause(self) -> tuple[str, list[str]]: ...
    def group_maintenance_clause(self) -> tuple[str, list[str]]: ...


class _EventVectorCollection(Protocol):
    def delete(self, *, ids: list[str]) -> None: ...


class EventMaintenance:
        def __init__(
            self, sqlite: SqliteFacade, *, scopes: _EventScopes,
            archive_path: Callable[[], Path], current_data_dir: Callable[[], Path],
            has_primary_private_owner: Callable[[], bool],
            upsert_chroma: Callable[[int, str, str, float, str, list[float]], None],
            current_collection: Callable[[], _EventVectorCollection | None],
        ) -> None:
            self._sqlite = sqlite
            self._scopes = scopes
            self._archive_path = archive_path
            self._current_data_dir = current_data_dir
            self._has_primary_private_owner = has_primary_private_owner
            self._upsert_chroma = upsert_chroma
            self._current_collection = current_collection

        def delete_events(self, ids: List[int]) -> int:
            return self.purge_events(ids, tombstone=True)

        def _archive_events(self, rows: Sequence[sqlite3.Row], reason: str = "") -> int:
            if not rows:
                return 0
            path = self._archive_path()
            stamp = datetime.now().astimezone().isoformat()
            n = 0

            def archive_lines():
                nonlocal n
                for row in rows:
                    source = self._sqlite.fetchone(
                        "SELECT source_context, owner_key, experience_subject, fact_subject, "
                        "involved_owner_keys, read_scope, shared_view FROM events WHERE id=?",
                        (int(row["id"]),))
                    entry = {"archived_at": stamp, "id": int(row["id"]), "raw": row["raw"]}
                    for key in ("source_snapshot", "source_snapshot_sha256"):
                        if key in row.keys() and row[key]:
                            entry[key] = row[key]
                    if "observed_at" in row.keys():
                        entry["observed_at"] = str(row["observed_at"] or "")
                    entry.update(dict(source) if source else {"source_context": ""})
                    if reason:
                        entry["archive_reason"] = reason
                    n += 1
                    yield json.dumps(entry, ensure_ascii=False)

            _append_fsynced_jsonl(path, archive_lines())
            return n

        def purge_events(self, ids: List[int], *, tombstone: bool,
                         protect_grounded: bool = False) -> int:
            assert self._sqlite.is_open
            unique_ids = sorted(set(ids))
            if not unique_ids:
                return 0
            with self._sqlite.transaction() as conn:
                if protect_grounded:
                    pinned = self.grounded_protected_ids()
                    unique_ids = [eid for eid in unique_ids if eid not in pinned]
                if not unique_ids:
                    return 0
                marks = ",".join("?" for _ in unique_ids)
                guard = " AND expired_at IS NOT NULL" if protect_grounded else ""
                rows = conn.execute(
                    f"SELECT id,raw,original_fingerprint,source_snapshot,source_snapshot_sha256 "
                    f"FROM events WHERE id IN ({marks})" + guard, unique_ids).fetchall()
                unique_ids = [row["id"] for row in rows]
                if not unique_ids:
                    return 0
                placeholders = ",".join("?" for _ in unique_ids)
                occurrences = conn.execute(
                    f"SELECT event_id AS id,raw,observed_at,source_snapshot,source_snapshot_sha256 FROM event_occurrences WHERE event_id IN ({placeholders})",
                    unique_ids).fetchall()
                self._archive_events([*rows, *occurrences])
                if tombstone:
                    now = datetime.now().astimezone().isoformat()
                    for row in rows:
                        content = re.sub(r"^\[[^]]+\]\s*", "", row["raw"])
                        conn.execute(
                            "INSERT OR REPLACE INTO forgotten(fingerprint, created_at) VALUES (?, ?)",
                            (row["original_fingerprint"] or _fingerprint(content), now),
                        )
                conn.execute(f"UPDATE events SET blacklist=1 WHERE id IN ({placeholders})", unique_ids)
                conn.execute(f"DELETE FROM events WHERE id IN ({placeholders})", unique_ids)
            collection = self._current_collection()
            if collection is not None and rows:
                vector_ids = [f"evt-{int(row['id'])}" for row in rows]
                try:
                    collection.delete(ids=vector_ids)
                except chroma_backend_errors() as exc:
                    if not is_chroma_backend_failure(exc):
                        raise
                    logger.warning("sanae_mem: Chroma delete failed (%s)", type(exc).__name__)
            return len(rows)

        def promote_events(self, ids: List[int], to: str, now: str) -> int:
            assert self._sqlite.is_open
            if not ids:
                return 0
            scope = self._scopes.private_owner_write_clause()
            if scope is None:
                return 0
            scope_clause, scope_params = scope
            placeholders = ",".join("?" for _ in ids)
            ceiling = half_life_ceiling(to)
            cursor = self._sqlite.execute(
                f"UPDATE events SET tier=?, promoted_at=?, last_active_at=?, "
                "recall_count=MAX(recall_count,3), curve_ttl='', "
                "half_life_days=MIN(?, MAX(?, COALESCE(half_life_days, ?))), "
                "promoted_reason=COALESCE(NULLIF(candidate_reason,''),'manual'), "
                f"pending_to='', ttl_expire=NULL, expired_at=NULL "
                f"WHERE id IN ({placeholders}) AND blacklist=0 AND {scope_clause}",
                [to, now, now, ceiling, HALF_LIFE_MIN_DAYS, HALF_LIFE_MIN_DAYS, *ids, *scope_params],
            )
            return max(0, int(cursor.rowcount))

        def expire_to_review(self, now: str) -> int:
            assert self._sqlite.is_open
            pinned = sorted(self.grounded_protected_ids())
            skip = (" AND id NOT IN (" + ",".join("?" for _ in pinned) + ")") if pinned else ""
            visible, visible_params = self._scopes.group_maintenance_clause()
            cursor = self._sqlite.execute(
                f"""UPDATE events
                   SET tier='short', pending_to='', expired_at=?, ttl_expire=NULL,
                       asked=0, updated_at=?
                   WHERE tier IN ('short', 'pending') AND expired_at IS NULL
                     AND ttl_expire IS NOT NULL AND ttl_expire<? AND ttl_expire<>curve_ttl
                     AND {visible}""" + skip,
                (now, now, now, *visible_params, *pinned),
            )
            return max(0, int(cursor.rowcount))

        def review_expiring(self, now: str, window_days: int, importance_threshold: float, *, natural: bool = False) -> List[int]:
            assert self._sqlite.is_open
            current = datetime.fromisoformat(now)
            cutoff = (current - timedelta(days=window_days)).isoformat()
            accelerated_cutoff = (current - timedelta(days=1)).isoformat()
            visible, visible_params = self._scopes.group_maintenance_clause()
            return [
                int(row["id"])
                for row in self._sqlite.fetchall(
                    f"""SELECT id FROM events
                       WHERE tier='short' AND blacklist=0
                         AND expired_at IS NOT NULL AND (COALESCE(ttl_expire=curve_ttl,0)=?)
                         AND (expired_at<=? OR (?=0 AND importance<? AND expired_at<=?))
                         AND {visible}""",
                    (int(natural), cutoff, int(natural), importance_threshold,
                     accelerated_cutoff, *visible_params),
                )
            ]

        def grounded_protected_ids(self) -> set[int]:
            assert self._sqlite.is_open
            return {
                int(row[0]) for row in self._sqlite.fetchall(
                    "SELECT DISTINCT event_id FROM recognition_evidence")
            }

        def _curve_candidates(self, now: datetime) -> List[sqlite3.Row]:
            assert self._sqlite.is_open
            visible, visible_params = self._scopes.group_maintenance_clause()
            return self._sqlite.fetchall(
                f"""SELECT id, ts, raw, summary, summary_source, topic, importance, tier,
                          recall_count, last_active_at, half_life_days, created_at, updated_at,
                          ttl_expire,curve_ttl,source_snapshot,source_snapshot_sha256
                   FROM events
                   WHERE blacklist=0 AND expired_at IS NULL AND (
                       (tier IN ('short','pending') AND ttl_expire=curve_ttl)
                       OR (tier IN ('important','long','distilled') AND curve_ttl='' AND trim(COALESCE(summary,''))<>''
                           AND julianday(?) - julianday(COALESCE(NULLIF(last_active_at,''),
                               NULLIF(updated_at,''),created_at)) > 45))
                     AND {visible}
                   ORDER BY id""", (now.isoformat(), *visible_params)
            )

        @staticmethod
        def _curve_idle_and_half_life(row: sqlite3.Row, now: datetime) -> tuple[float, float]:
            anchor = datetime.fromisoformat(
                str(row["last_active_at"] or "").strip()
                or str(row["updated_at"] or "").strip()
                or str(row["created_at"]))
            if anchor.tzinfo is None:
                anchor = anchor.replace(tzinfo=now.tzinfo)
            idle_days = max(0.0, (now - anchor).total_seconds() / 86400)
            half_life = float(row["half_life_days"] or 0) or _CURVE_DEFAULT_HALF_LIFE_DAYS
            return idle_days, half_life

        @classmethod
        def _curve_activation(cls, row: sqlite3.Row, now: datetime) -> float:
            idle_days, half_life = cls._curve_idle_and_half_life(row, now)
            return max(0.0, min(1.0, float(row["importance"] or 0))) * 2 ** (-idle_days / half_life)

        @staticmethod
        def _curve_forgets_quietly(row: sqlite3.Row) -> bool:
            return (int(row["recall_count"] or 0) == 0
                    and float(row["importance"] or 0) < _CURVE_DORMANT_IMPORTANCE
                    and str(row["tier"] or "") == "short")

        def collapse_forgetting_curve(self, now: datetime) -> Dict[str, int]:
            """每晚推进三阶模糊痕迹坍缩：鲜活 → 大概印象 → 深层休眠／自然淡忘。

            受证据锁保护的行原样保留；已坍缩过或已休眠的行不再二次处理，保证幂等。
            """
            result = {"faded": 0, "dormant": 0, "forgotten": 0}
            protected = self.grounded_protected_ids()
            for row in self._curve_candidates(now):
                event_id = row["id"]
                body = re.sub(r"\s+", " ", _candidate_text(row["raw"])).strip()
                if (row["tier"] in ("long", "important", "distilled")
                        and row["summary_source"] in ("original", "", None)
                        and len(body) > 150
                        and row["summary"].strip() == body[:150]):
                    continue
                idle_days, half_life = self._curve_idle_and_half_life(row, now)
                activation = self._curve_activation(row, now)
                if event_id in protected or idle_days < half_life or activation >= _CURVE_VIVID_FLOOR:
                    continue
                dormant = activation < _CURVE_DORMANT_FLOOR
                quiet = dormant and self._curve_forgets_quietly(row)
                already_faded = row["summary_source"] == SUMMARY_SOURCE_GIST_FADED
                if already_faded and not dormant:
                    continue
                gist = (str(row["summary"] or "").strip()
                        or _candidate_text(row["raw"])[:80]
                        or str(row["raw"] or "").strip()[:80])
                vector = None if quiet or already_faded else _embed(gist, data_dir=self._current_data_dir())
                with self._sqlite.transaction() as conn:
                    current = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
                    if (not current or any(current[k] != row[k] for k in row.keys())
                            or current["expired_at"] or current["blacklist"]
                            or event_id in self.grounded_protected_ids()):
                        continue
                    if quiet:
                        conn.execute("UPDATE events SET expired_at=? WHERE id=?",
                                     (now.isoformat(), event_id))
                        continue
                    if vector is not None:
                        occurrences = conn.execute(
                            "SELECT event_id AS id,raw,observed_at,source_snapshot,source_snapshot_sha256 FROM event_occurrences WHERE event_id=? ORDER BY id",
                            (event_id,)).fetchall()
                        self._archive_events([row, *occurrences], reason="gist_collapse")
                        conn.execute("DELETE FROM event_occurrences WHERE event_id=?", (event_id,))
                        original = re.sub(r"^\[[^]]*\]\s*", "", row["raw"]).split("\n[补充 ", 1)[0]
                        conn.execute(
                            "UPDATE events SET raw=?,summary=?,summary_source=?,keywords=?,vec=?,"
                            "original_fingerprint=?,source_snapshot='',source_snapshot_sha256='' WHERE id=?",
                            (f"[{row['ts']}] {gist}", gist, SUMMARY_SOURCE_GIST_FADED,
                             event_keywords(gist), _pack_vector(vector), _fingerprint(original), event_id))
                    if not dormant and row["ttl_expire"] and row["ttl_expire"] == row["curve_ttl"]:
                        anchor = datetime.fromisoformat(row["last_active_at"] or row["updated_at"] or row["created_at"])
                        days = row["half_life_days"] * math.log2(row["importance"] / _CURVE_DORMANT_FLOOR)
                        until = (anchor + timedelta(days=days)).isoformat()
                        conn.execute("UPDATE events SET ttl_expire=?,curve_ttl=? WHERE id=?",
                                     (until, until, event_id))
                    if dormant:
                        conn.execute(
                            "UPDATE events SET tier='dormant',ttl_expire=NULL,curve_ttl='',pending_to='' WHERE id=?",
                            (event_id,))
                if vector is not None:
                    self._upsert_chroma(event_id, gist, row["topic"], row["importance"], row["ts"], vector)
                    result["faded"] += 1
                result["dormant"] += int(dormant)
            return result

        def decay_expiring_importance(self, factor: float, floor: float, now: str) -> int:
            assert self._sqlite.is_open
            visible, visible_params = self._scopes.group_maintenance_clause()
            cursor = self._sqlite.execute(
                f"""UPDATE events
                   SET importance=MAX(?, importance * ?), updated_at=?
                   WHERE tier='short' AND blacklist=0 AND expired_at IS NOT NULL
                     AND {visible}""",
                (floor, factor, now, *visible_params),
            )
            return max(0, int(cursor.rowcount))

        def decay_confidence(self, older_than: str, factor: float, floor: float, now: str) -> int:
            assert self._sqlite.is_open
            visible, visible_params = self._scopes.group_maintenance_clause()
            cursor = self._sqlite.execute(
                "UPDATE events SET confidence=MAX(?, confidence * ?), updated_at=? "
                f"WHERE created_at<? AND asked=0 AND blacklist=0 AND {visible}",
                (floor, factor, now, older_than, *visible_params),
            )
            return max(0, int(cursor.rowcount))

        def count_distilled_between(self, start: str, end: str) -> int:
            assert self._sqlite.is_open
            row = self._sqlite.fetchone(
                "SELECT COUNT(*) FROM events WHERE distilled_at>=? AND distilled_at<? "
                "AND COALESCE(CASE WHEN json_valid(source_context) "
                "THEN json_extract(source_context,'$.processing') END, '')<>'group_receipt_ingest'",
                (start, end),
            )
            assert row is not None
            return int(row[0])

        def list_distill_candidates(self, cutoff: str, now: str) -> List[sqlite3.Row]:
            assert self._sqlite.is_open
            # Group rows are not distilled in this round: the summary reviewer carries a private
            # resident baseline, so a group summary needs its own entry before it may run.
            return self._sqlite.fetchall(
                f"""SELECT id, raw, importance, topic, ts, session_id FROM events
                   WHERE created_at>=? AND blacklist=0 AND distilled_at IS NULL
                     AND (tier='short' OR (tier IN ('long','important') AND length(raw)>150))
                     AND COALESCE(summary_source,'') NOT IN ('gist_faded', 'distilled')
                     AND expired_at IS NULL
                     AND (ttl_expire IS NULL OR ttl_expire>?)
                     AND {exclude_group_rows()}
                   ORDER BY importance DESC, id DESC""",
                (cutoff, now),
            )

        def _summary_write_clause(self) -> tuple[str, list[str]]:
            """Rows an id-based summary write may touch in this runtime.

            A trusted private turn keeps the whole readable history of its bound person, so an old
            row under a previously bound owner is still summarized by its own id (its owner and
            source are never rewritten). A group turn is limited to its one write owner and
            trusted cron to the groups the current config still enables. A caller-supplied
            parameter never widens any of them.
            """
            if self._has_primary_private_owner():
                return self._scopes.read_scope_clause()
            return self._scopes.group_write_clause()

        def get_distillable_event(self, event_id: int, now: str) -> Optional[sqlite3.Row]:
            assert self._sqlite.is_open
            visible, visible_params = self._summary_write_clause()
            return self._sqlite.fetchone(
                f"""SELECT ts, topic, importance FROM events
                   WHERE id=? AND (tier='short' OR (tier IN ('long','important') AND length(raw)>150))
                     AND blacklist=0 AND distilled_at IS NULL
                     AND expired_at IS NULL
                     AND (ttl_expire IS NULL OR ttl_expire>?)
                     AND COALESCE(summary_source,'') NOT IN (?, ?)
                     AND {visible}""",
                (event_id, now, SUMMARY_SOURCE_GIST_FADED, SUMMARY_SOURCE_DISTILLED,
                 *visible_params),
            )

        def promote_distilled_event(
            self, event_id: int, summary: str, keywords: str, vector: bytes, now: str,
            *, record_distilled_at: bool = True,
        ) -> int:
            """写入 summary、summary_source=distilled 和可选 distilled_at，返回变更行数。

            不改 raw、TTL、tier 或晋升字段。返回 0 表示前置条件不满足，调用方保留原事件。
            已坍缩成大概印象或已蒸馏成摘要的行不再覆盖。"""
            assert self._sqlite.is_open
            visible, visible_params = self._summary_write_clause()
            distilled_value = now if record_distilled_at else None
            cursor = self._sqlite.execute(
                f"""UPDATE events
                   SET summary=?, summary_source=?, keywords=?, vec=?,
                       distilled_at=COALESCE(?, distilled_at), updated_at=?
                   WHERE id=? AND (tier='short' OR (tier IN ('long','important') AND length(raw)>150))
                     AND blacklist=0 AND distilled_at IS NULL
                     AND expired_at IS NULL AND (ttl_expire IS NULL OR ttl_expire>?)
                     AND COALESCE(summary_source,'') NOT IN (?, ?)
                     AND {visible}""",
                (summary, SUMMARY_SOURCE_DISTILLED, keywords, vector, distilled_value, now,
                 event_id, now, SUMMARY_SOURCE_GIST_FADED, SUMMARY_SOURCE_DISTILLED,
                 *visible_params),
            )
            return max(0, int(cursor.rowcount))

        def recent_merge_rows(self, cutoff: str) -> List[sqlite3.Row]:
            """返回 24 小时内可参与去重的行，包含层级、可信出处和人物主体字段。

            合并方据此保护长期保留行，并按出处及主体分组；深层休眠的行不参与合并，
            否则一次近重复合并就会把休眠事件翻出来或直接删掉。"""
            assert self._sqlite.is_open
            visible, visible_params = self._scopes.group_maintenance_clause()
            return self._sqlite.fetchall(
                f"""SELECT id, keywords, importance, confidence, tier, owner_key, source_context,
                          experience_subject, fact_subject, involved_owner_keys
                   FROM events
                   WHERE created_at>=? AND blacklist=0 AND expired_at IS NULL
                     AND tier<>'dormant' AND COALESCE(summary_source,'')<>'gist_faded'
                     AND {visible} ORDER BY id""",
                (cutoff, *visible_params),
            )

        def merge_duplicate_events(self, survivor_id: int, duplicates: List[int],
                                   importance: float, confidence: float, now: str) -> int:
            """Lossless compact: archive, move each occurrence, then delete duplicate IDs in one DB txn."""
            assert self._sqlite.is_open
            ids = sorted(set(int(i) for i in duplicates) - {int(survivor_id)})
            ids = [i for i in ids if i not in self.grounded_protected_ids()]
            if not ids:
                return 0
            visible, visible_params = self._summary_write_clause()
            scoped_ids = [int(survivor_id), *ids]
            marks = ",".join("?" for _ in scoped_ids)
            allowed = self._sqlite.fetchall(
                f"SELECT id FROM events WHERE id IN ({marks}) AND {visible}",
                (*scoped_ids, *visible_params),
            )
            if len(allowed) != len(scoped_ids):
                return 0
            with self._sqlite.transaction() as conn:
                keep = conn.execute(
                    "SELECT raw,topic,ts,owner_key,source_snapshot,source_snapshot_sha256,created_at,source_context, "
                    "experience_subject,fact_subject,involved_owner_keys "
                    "FROM events WHERE id=?", (int(survivor_id),)).fetchone()
                marks = ",".join("?" for _ in ids)
                rows = conn.execute(
                    f"SELECT id,raw,created_at,owner_key,source_snapshot,source_snapshot_sha256,source_context, "
                    f"experience_subject,fact_subject,involved_owner_keys "
                    f"FROM events WHERE id IN ({marks}) ORDER BY id", ids).fetchall()
                if keep is None or len(rows) != len(ids) or any(
                        r["owner_key"] != keep["owner_key"] for r in rows):
                    raise ValueError("merge_owner_or_row_drift")
                if any(r["source_context"] != keep["source_context"] for r in rows):
                    raise ValueError("merge_source_drift")
                if any((r["experience_subject"], r["fact_subject"], r["involved_owner_keys"])
                       != (keep["experience_subject"], keep["fact_subject"],
                           keep["involved_owner_keys"]) for r in rows):
                    raise ValueError("merge_subject_drift")
                self._archive_events(rows)
                if keep["source_snapshot"] and not conn.execute(
                        "SELECT 1 FROM event_occurrences WHERE event_id=? LIMIT 1",
                        (int(survivor_id),)).fetchone():
                    conn.execute(
                        "INSERT INTO event_occurrences"
                        "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                        " VALUES(?,?,?,?,?)",
                        (int(survivor_id), keep["raw"], keep["created_at"],
                         keep["source_snapshot"], keep["source_snapshot_sha256"]))
                text = str(keep["raw"] or "")
                for row in rows:
                    source_count = conn.execute(
                        "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                        (row["id"],)).fetchone()[0]
                    if not source_count:
                        conn.execute(
                            "INSERT INTO event_occurrences"
                            "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                            " VALUES(?,?,?,?,?)",
                            (int(survivor_id), row["raw"], row["created_at"],
                             row["source_snapshot"], row["source_snapshot_sha256"]))
                    else:
                        conn.execute(
                            "UPDATE event_occurrences SET event_id=? WHERE event_id=?",
                            (int(survivor_id), int(row["id"])))
                    text += f"\n[补充 {row['created_at']}] {row['raw']}"
                vector = _embed(text, data_dir=self._current_data_dir())
                summary = text[-150:]
                conn.execute(
                    "UPDATE events SET raw=?, summary=?, vec=?, importance=?, confidence=?, updated_at=? "
                    "WHERE id=?", (text, summary, _pack_vector(vector), importance,
                                   confidence, now, int(survivor_id)))
                conn.execute(f"DELETE FROM events WHERE id IN ({marks})", ids)
            self._upsert_chroma(survivor_id, summary, keep["topic"], importance, keep["ts"], vector)
            collection = self._current_collection()
            if collection is not None:
                vector_ids = [f"evt-{i}" for i in ids]
                try:
                    collection.delete(ids=vector_ids)
                except chroma_backend_errors() as exc:
                    if not is_chroma_backend_failure(exc):
                        raise
                    logger.warning("sanae_mem: Chroma delete after merge failed (%s)", type(exc).__name__)
            return len(ids)

        def update_merged_survivor(self, event_id: int, importance: float, confidence: float, now: str) -> None:
            assert self._sqlite.is_open
            visible, visible_params = self._summary_write_clause()
            self._sqlite.execute(
                f"UPDATE events SET importance=?, confidence=?, updated_at=? "
                f"WHERE id=? AND {visible}",
                (importance, confidence, now, event_id, *visible_params),
            )

        def list_short_candidates(self) -> List[sqlite3.Row]:
            assert self._sqlite.is_open
            visible, visible_params = self._scopes.group_maintenance_clause()
            return self._sqlite.fetchall(
                f"""SELECT id, raw, keywords, importance, confidence, topic, ts
                   FROM events WHERE blacklist=0 AND tier='short' AND expired_at IS NULL
                     AND COALESCE(summary_source,'')<>'gist_faded'
                     AND {visible} ORDER BY id""", visible_params
            )

        def list_pending_candidates(self) -> List[sqlite3.Row]:
            assert self._sqlite.is_open
            visible, visible_params = self._scopes.group_maintenance_clause()
            return self._sqlite.fetchall(
                f"""SELECT id, raw, keywords, importance, confidence, topic, ts,
                          pending_to, candidate_reason
                   FROM events WHERE blacklist=0 AND tier='pending' AND expired_at IS NULL
                     AND {visible}
                   ORDER BY id""", visible_params
            )

        def mark_pending(self, pending_by_id: Dict[int, str], now: str, reasons: Optional[Dict[int, str]] = None) -> None:
            """进入待确认候选池，写 staged_at 和 candidate_reason，不确认晋升。

            不更新 promoted_at；旧 pending 行的该字段可能保留历史值。"""
            assert self._sqlite.is_open
            reason_map = reasons or {}
            with self._sqlite.transaction() as conn:
                for event_id, pending_to in pending_by_id.items():
                    conn.execute(
                        "UPDATE events SET tier='pending', pending_to=?, staged_at=?, "
                        "candidate_reason=? WHERE id=?",
                        (pending_to, now, str(reason_map.get(event_id, "") or ""), event_id),
                    )
