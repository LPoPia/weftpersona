from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    from .curve import half_life_ceiling
    from .event_store import SUMMARY_SOURCE_DISTILLED, SUMMARY_SOURCE_GIST_FADED
    from .textutil import _embed, _pack_vector, _fingerprint, provider_home
except ImportError:
    from curve import half_life_ceiling
    from event_store import SUMMARY_SOURCE_DISTILLED, SUMMARY_SOURCE_GIST_FADED
    from textutil import _embed, _pack_vector, _fingerprint, provider_home

logger = logging.getLogger(__name__)

_CURVE_VIVID_FLOOR = 0.35
_CURVE_DORMANT_FLOOR = 0.15
_CURVE_DORMANT_IMPORTANCE = 0.6
_CURVE_DEFAULT_HALF_LIFE_DAYS = 14.0

class EventMaintenanceMixin:
        def _delete_events(self, ids: List[int]) -> int:
            return self._purge_events(ids, tombstone=True)

        def _archive_events(self, rows, reason: str = "") -> int:
            if not rows:
                return 0
            archive_dir = os.path.join(provider_home(self), "memory-archive")
            os.makedirs(archive_dir, exist_ok=True)
            path = os.path.join(archive_dir, "events-archived.jsonl")
            stamp = datetime.now().astimezone().isoformat()
            n = 0
            with open(path, "a", encoding="utf-8") as fh:
                for row in rows:
                    source = self._conn.execute(
                        "SELECT source_context, owner_key, experience_subject, fact_subject, "
                        "involved_owner_keys, read_scope, shared_view FROM events WHERE id=?",
                        (int(row["id"]),)).fetchone()
                    entry = {"archived_at": stamp, "id": int(row["id"]), "raw": row["raw"]}
                    for key in ("source_snapshot", "source_snapshot_sha256"):
                        if key in row.keys() and row[key]:
                            entry[key] = row[key]
                    if "observed_at" in row.keys():
                        entry["observed_at"] = str(row["observed_at"] or "")
                    entry.update(dict(source) if source else {"source_context": ""})
                    if reason:
                        entry["archive_reason"] = reason
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    n += 1
            return n

        def _purge_events(self, ids: List[int], *, tombstone: bool,
                          protect_grounded: bool = False) -> int:
            assert self._conn is not None
            unique_ids = sorted(set(ids))
            if not unique_ids:
                return 0
            with self._lock, self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                if protect_grounded:
                    pinned = self._grounded_protected_ids()
                    unique_ids = [eid for eid in unique_ids if eid not in pinned]
                if not unique_ids:
                    return 0
                marks = ",".join("?" for _ in unique_ids)
                guard = " AND expired_at IS NOT NULL" if protect_grounded else ""
                rows = self._conn.execute(
                    f"SELECT id,raw,original_fingerprint,source_snapshot,source_snapshot_sha256 "
                    f"FROM events WHERE id IN ({marks})" + guard, unique_ids).fetchall()
                unique_ids = [row["id"] for row in rows]
                if not unique_ids:
                    return 0
                placeholders = ",".join("?" for _ in unique_ids)
                occurrences = self._conn.execute(
                    f"SELECT event_id AS id,raw,observed_at,source_snapshot,source_snapshot_sha256 FROM event_occurrences WHERE event_id IN ({placeholders})",
                    unique_ids).fetchall()
                self._archive_events([*rows, *occurrences])
                if tombstone:
                    now = datetime.now().astimezone().isoformat()
                    for row in rows:
                        content = re.sub(r"^\[[^]]+\]\s*", "", row["raw"])
                        self._conn.execute(
                            "INSERT OR REPLACE INTO forgotten(fingerprint, created_at) VALUES (?, ?)",
                            (row["original_fingerprint"] or _fingerprint(content), now),
                        )
                self._conn.execute(f"UPDATE events SET blacklist=1 WHERE id IN ({placeholders})", unique_ids)
                self._conn.execute(f"DELETE FROM events WHERE id IN ({placeholders})", unique_ids)
                self._conn.commit()
            if self._collection is not None and rows:
                try:
                    self._collection.delete(ids=[f"evt-{int(row['id'])}" for row in rows])
                except Exception as exc:
                    logger.warning("sanae_mem: Chroma delete failed (%s)", type(exc).__name__)
            return len(rows)

        def promote_events(self, ids: List[int], to: str, now: str) -> int:
            assert self._conn is not None
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            with self._lock:
                cursor = self._conn.execute(
                    f"UPDATE events SET tier=?, promoted_at=?, last_active_at=?, "
                    "recall_count=MAX(recall_count,3), curve_ttl='', "
                    "half_life_days=MIN(half_life_days, ?), "
                    "promoted_reason=COALESCE(NULLIF(candidate_reason,''),'manual'), "
                    f"pending_to='', ttl_expire=NULL, expired_at=NULL WHERE id IN ({placeholders}) AND blacklist=0",
                    [to, now, now, half_life_ceiling(to), *ids])
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def expire_to_review(self, now: str) -> int:
            assert self._conn is not None
            pinned = sorted(self._grounded_protected_ids())
            skip = (" AND id NOT IN (" + ",".join("?" for _ in pinned) + ")") if pinned else ""
            with self._lock:
                cursor = self._conn.execute(
                    """UPDATE events
                       SET tier='short', pending_to='', expired_at=?, ttl_expire=NULL,
                           asked=0, updated_at=?
                       WHERE tier IN ('short', 'pending') AND expired_at IS NULL
                         AND ttl_expire IS NOT NULL AND ttl_expire<? AND ttl_expire<>curve_ttl""" + skip,
                    (now, now, now, *pinned),
                )
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def review_expiring(self, now: str, window_days: int, importance_threshold: float, *, natural: bool = False) -> List[int]:
            assert self._conn is not None
            current = datetime.fromisoformat(now)
            cutoff = (current - timedelta(days=window_days)).isoformat()
            accelerated_cutoff = (current - timedelta(days=1)).isoformat()
            return [
                int(row["id"])
                for row in self._conn.execute(
                    """SELECT id FROM events
                       WHERE tier='short' AND blacklist=0
                         AND expired_at IS NOT NULL AND (COALESCE(ttl_expire=curve_ttl,0)=?)
                         AND (expired_at<=? OR (?=0 AND importance<? AND expired_at<=?))""",
                    (int(natural), cutoff, int(natural), importance_threshold, accelerated_cutoff),
                ).fetchall()
            ]

        def _grounded_protected_ids(self) -> set:
            assert self._conn is not None
            protected = set(self.pending_grounded_evidence_ids())
            protected.update(
                int(row[0]) for row in self._conn.execute(
                    "SELECT DISTINCT event_id FROM recognition_evidence"))
            return protected

        def _curve_candidates(self, now: datetime) -> List[sqlite3.Row]:
            assert self._conn is not None
            return self._conn.execute(
                """SELECT id, ts, raw, summary, summary_source, topic, importance, tier,
                          recall_count, last_active_at, half_life_days, created_at, updated_at,
                          ttl_expire,curve_ttl,source_snapshot,source_snapshot_sha256
                   FROM events
                   WHERE blacklist=0 AND expired_at IS NULL AND (
                       (tier IN ('short','pending') AND ttl_expire=curve_ttl)
                       OR (tier IN ('important','long','distilled') AND curve_ttl='' AND trim(COALESCE(summary,''))<>''
                           AND julianday(?) - julianday(COALESCE(NULLIF(last_active_at,''),
                               NULLIF(updated_at,''),created_at)) > 45))
                   ORDER BY id""", (now.isoformat(),)
            ).fetchall()

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
            protected = self._grounded_protected_ids()
            for row in self._curve_candidates(now):
                event_id = row["id"]
                body = re.sub(r"\s+", " ", self._candidate_text(row["raw"])).strip()
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
                        or self._candidate_text(row["raw"])[:80]
                        or str(row["raw"] or "").strip()[:80])
                vector = None if quiet or already_faded else _embed(gist)
                with self._lock, self._conn:
                    self._conn.execute("BEGIN IMMEDIATE")
                    current = self._conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
                    if (not current or any(current[k] != row[k] for k in row.keys())
                            or current["expired_at"] or current["blacklist"]
                            or event_id in self._grounded_protected_ids()):
                        continue
                    if quiet:
                        self._conn.execute("UPDATE events SET expired_at=? WHERE id=?",
                                           (now.isoformat(), event_id))
                        continue
                    if not already_faded:
                        occurrences = self._conn.execute(
                            "SELECT event_id AS id,raw,observed_at,source_snapshot,source_snapshot_sha256 FROM event_occurrences WHERE event_id=? ORDER BY id",
                            (event_id,)).fetchall()
                        self._archive_events([row, *occurrences], reason="gist_collapse")
                        self._conn.execute("DELETE FROM event_occurrences WHERE event_id=?", (event_id,))
                        original = re.sub(r"^\[[^]]*\]\s*", "", row["raw"]).split("\n[补充 ", 1)[0]
                        self._conn.execute(
                            "UPDATE events SET raw=?,summary=?,summary_source=?,keywords=?,vec=?,"
                            "original_fingerprint=?,source_snapshot='',source_snapshot_sha256='' WHERE id=?",
                            (f"[{row['ts']}] {gist}", gist, SUMMARY_SOURCE_GIST_FADED,
                             self._keywords(gist), _pack_vector(vector), _fingerprint(original), event_id))
                    if not dormant and row["ttl_expire"] and row["ttl_expire"] == row["curve_ttl"]:
                        anchor = datetime.fromisoformat(row["last_active_at"] or row["updated_at"] or row["created_at"])
                        days = row["half_life_days"] * math.log2(row["importance"] / _CURVE_DORMANT_FLOOR)
                        until = (anchor + timedelta(days=days)).isoformat()
                        self._conn.execute("UPDATE events SET ttl_expire=?,curve_ttl=? WHERE id=?",
                                           (until, until, event_id))
                    if dormant:
                        self._conn.execute(
                            "UPDATE events SET tier='dormant',ttl_expire=NULL,curve_ttl='',pending_to='' WHERE id=?",
                            (event_id,))
                if vector is not None:
                    self._upsert_chroma(event_id, gist, row["topic"], row["importance"], row["ts"], vector)
                    result["faded"] += 1
                result["dormant"] += int(dormant)
            return result

        def decay_expiring_importance(self, factor: float, floor: float, now: str) -> int:
            assert self._conn is not None
            with self._lock:
                cursor = self._conn.execute(
                    """UPDATE events
                       SET importance=MAX(?, importance * ?), updated_at=?
                       WHERE tier='short' AND blacklist=0 AND expired_at IS NOT NULL""",
                    (floor, factor, now),
                )
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def decay_confidence(self, older_than: str, factor: float, floor: float, now: str) -> int:
            assert self._conn is not None
            with self._lock:
                cursor = self._conn.execute(
                    "UPDATE events SET confidence=MAX(?, confidence * ?), updated_at=? "
                    "WHERE created_at<? AND asked=0 AND blacklist=0",
                    (floor, factor, now, older_than),
                )
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def count_distilled_between(self, start: str, end: str) -> int:
            assert self._conn is not None
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE distilled_at>=? AND distilled_at<?", (start, end)
            ).fetchone()[0])

        def list_distill_candidates(self, cutoff: str, now: str) -> List[sqlite3.Row]:
            assert self._conn is not None
            return self._conn.execute(
                """SELECT id, raw, importance, topic, ts, session_id FROM events
                   WHERE created_at>=? AND blacklist=0 AND distilled_at IS NULL
                     AND (tier='short' OR (tier IN ('long','important') AND length(raw)>150))
                     AND COALESCE(summary_source,'')<>'gist_faded'
                     AND expired_at IS NULL
                     AND (ttl_expire IS NULL OR ttl_expire>?)
                   ORDER BY importance DESC, id DESC""",
                (cutoff, now),
            ).fetchall()

        def get_distillable_event(self, event_id: int, now: str) -> Optional[sqlite3.Row]:
            assert self._conn is not None
            return self._conn.execute(
                """SELECT ts, topic, importance FROM events
                   WHERE id=? AND (tier='short' OR (tier IN ('long','important') AND length(raw)>150))
                     AND blacklist=0 AND distilled_at IS NULL
                     AND expired_at IS NULL
                     AND (ttl_expire IS NULL OR ttl_expire>?)
                     AND COALESCE(summary_source,'')<>?""",
                (event_id, now, SUMMARY_SOURCE_GIST_FADED),
            ).fetchone()

        def promote_distilled_event(self, event_id: int, summary: str, keywords: str, vector: bytes, now: str) -> int:
            """写入 summary、summary_source=distilled 和 distilled_at，返回变更行数。

            不改 raw、TTL、tier 或晋升字段。返回 0 表示前置条件不满足，调用方保留原事件。
            已坍缩成大概印象的行不再蒸馏：再摘一次只会把印象冒充成新鲜原话。"""
            assert self._conn is not None
            with self._lock:
                cursor = self._conn.execute(
                    """UPDATE events
                       SET summary=?, summary_source=?, keywords=?, vec=?, distilled_at=?, updated_at=?
                       WHERE id=? AND (tier='short' OR (tier IN ('long','important') AND length(raw)>150))
                         AND blacklist=0 AND distilled_at IS NULL
                         AND expired_at IS NULL AND (ttl_expire IS NULL OR ttl_expire>?)
                         AND COALESCE(summary_source,'')<>?""",
                    (summary, SUMMARY_SOURCE_DISTILLED, keywords, vector, now, now, event_id, now,
                     SUMMARY_SOURCE_GIST_FADED),
                )
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def recent_merge_rows(self, cutoff: str) -> List[sqlite3.Row]:
            """返回 24 小时内可参与去重的行，包含层级、可信出处和人物主体字段。

            合并方据此保护长期保留行，并按出处及主体分组；深层休眠的行不参与合并，
            否则一次近重复合并就会把休眠事件翻出来或直接删掉。"""
            assert self._conn is not None
            return self._conn.execute(
                """SELECT id, keywords, importance, confidence, tier, owner_key, source_context,
                          experience_subject, fact_subject, involved_owner_keys
                   FROM events
                   WHERE created_at>=? AND blacklist=0 AND expired_at IS NULL
                     AND tier<>'dormant' AND COALESCE(summary_source,'')<>'gist_faded' ORDER BY id""",
                (cutoff,),
            ).fetchall()

        def merge_duplicate_events(self, survivor_id: int, duplicates: List[int],
                                   importance: float, confidence: float, now: str) -> int:
            """Lossless compact: archive, move each occurrence, then delete duplicate IDs in one DB txn."""
            assert self._conn is not None
            ids = sorted(set(int(i) for i in duplicates) - {int(survivor_id)})
            ids = [i for i in ids if i not in self._grounded_protected_ids()]
            if not ids:
                return 0
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    keep = self._conn.execute(
                        "SELECT raw,topic,ts,owner_key,source_snapshot,source_snapshot_sha256,created_at,source_context, "
                        "experience_subject,fact_subject,involved_owner_keys "
                        "FROM events WHERE id=?", (int(survivor_id),)).fetchone()
                    marks = ",".join("?" for _ in ids)
                    rows = self._conn.execute(
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
                    if keep["source_snapshot"] and not self._conn.execute(
                            "SELECT 1 FROM event_occurrences WHERE event_id=? LIMIT 1",
                            (int(survivor_id),)).fetchone():
                        self._conn.execute(
                            "INSERT INTO event_occurrences"
                            "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                            " VALUES(?,?,?,?,?)",
                            (int(survivor_id), keep["raw"], keep["created_at"],
                             keep["source_snapshot"], keep["source_snapshot_sha256"]))
                    self._archive_events(rows)
                    text = str(keep["raw"] or "")
                    for row in rows:
                        source_count = self._conn.execute(
                            "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                            (row["id"],)).fetchone()[0]
                        if not source_count:
                            self._conn.execute(
                                "INSERT INTO event_occurrences"
                                "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                                " VALUES(?,?,?,?,?)",
                                (int(survivor_id), row["raw"], row["created_at"],
                                 row["source_snapshot"], row["source_snapshot_sha256"]))
                        else:
                            self._conn.execute(
                                "UPDATE event_occurrences SET event_id=? WHERE event_id=?",
                                (int(survivor_id), int(row["id"])))
                        text += f"\n[补充 {row['created_at']}] {row['raw']}"
                    vector = _embed(text)
                    summary = text[-150:]
                    self._conn.execute(
                        "UPDATE events SET raw=?, summary=?, vec=?, importance=?, confidence=?, updated_at=? "
                        "WHERE id=?", (text, summary, _pack_vector(vector), importance,
                                       confidence, now, int(survivor_id)))
                    self._conn.execute(f"DELETE FROM events WHERE id IN ({marks})", ids)
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise
            self._upsert_chroma(survivor_id, summary, keep["topic"], importance, keep["ts"], vector)
            if self._collection is not None:
                try:
                    self._collection.delete(ids=[f"evt-{i}" for i in ids])
                except Exception as exc:
                    logger.warning("sanae_mem: Chroma delete after merge failed (%s)", type(exc).__name__)
            return len(ids)

        def update_merged_survivor(self, event_id: int, importance: float, confidence: float, now: str) -> None:
            assert self._conn is not None
            with self._lock:
                self._conn.execute(
                    "UPDATE events SET importance=?, confidence=?, updated_at=? WHERE id=?",
                    (importance, confidence, now, event_id),
                )
                self._conn.commit()

        def list_short_candidates(self) -> List[sqlite3.Row]:
            assert self._conn is not None
            return self._conn.execute(
                """SELECT id, raw, keywords, importance, confidence, topic, ts, cross_session_hits
                   FROM events WHERE blacklist=0 AND tier='short' AND expired_at IS NULL
                     AND COALESCE(summary_source,'')<>'gist_faded' ORDER BY id"""
            ).fetchall()

        def list_pending_candidates(self) -> List[sqlite3.Row]:
            assert self._conn is not None
            return self._conn.execute(
                """SELECT id, raw, keywords, importance, confidence, topic, ts,
                          pending_to, candidate_reason
                   FROM events WHERE blacklist=0 AND tier='pending' AND expired_at IS NULL
                   ORDER BY id"""
            ).fetchall()

        def mark_pending(self, pending_by_id: Dict[int, str], now: str, reasons: Optional[Dict[int, str]] = None) -> None:
            """进入待确认候选池，写 staged_at 和 candidate_reason，不确认晋升。

            不更新 promoted_at；旧 pending 行的该字段可能保留历史值。"""
            assert self._conn is not None
            reason_map = reasons or {}
            with self._lock:
                for event_id, pending_to in pending_by_id.items():
                    self._conn.execute(
                        "UPDATE events SET tier='pending', pending_to=?, staged_at=?, "
                        "candidate_reason=? WHERE id=?",
                        (pending_to, now, str(reason_map.get(event_id, "") or ""), event_id),
                    )
                self._conn.commit()
