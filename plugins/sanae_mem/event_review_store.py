from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

from datetime import datetime
from typing import Optional

class EventReviewStoreMixin:
        def _migrate_event_reviews_schema(self) -> None:
            assert self._conn is not None
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(event_reviews)")}
            additions = {
                "reviewed_text": "TEXT NOT NULL DEFAULT ''",
                "evidence_group": "TEXT NOT NULL DEFAULT ''",
            }
            with self._lock:
                for name, definition in additions.items():
                    if name not in columns:
                        self._conn.execute(
                            f"ALTER TABLE event_reviews ADD COLUMN {name} {definition}")
                        self._conn.commit()

        def commit_recognition_support(self, recognition: str, reviewed: dict) -> dict:
            """Persist support links for one host-verified grounded recognition.

            The host already checked the exact entry in the real files; this only compares
            the recognition body with the verified record and, when the covered entry differs
            from the reviewed text, its coverage target. Evidence retention is still checked
            inside the same SQLite transaction.
            """
            assert self._conn is not None
            body = str(recognition or "").strip()
            record = reviewed if isinstance(reviewed, dict) else {}
            ids = record.get("evidence_group_ids")
            review_sha = str(record.get("review_sha256") or "")
            if not body or not review_sha:
                return {"ok": False, "reason": "review_missing"}
            if (not isinstance(ids, list) or not (2 <= len(ids) <= 3)
                    or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in ids)
                    or len(set(ids)) != len(ids) or ids != sorted(ids)):
                return {"ok": False, "reason": "invalid_support_group"}
            body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if body_hash != str(record.get("content_sha256") or ""):
                coverage = self._conn.execute(
                    "SELECT covered_sha256 FROM recognition_review_targets WHERE review_sha256=?",
                    (review_sha,)).fetchone()
                if coverage is None or str(coverage["covered_sha256"]) != body_hash:
                    return {"ok": False, "reason": "covered_recognition_unbound"}
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    tiers = self._conn.execute(
                        "SELECT id,tier FROM events WHERE blacklist=0 AND id IN (" +
                        ",".join("?" for _ in ids) + ")", tuple(ids)).fetchall()
                    if len(tiers) != len(ids) or any(str(t["tier"]) != "long" for t in tiers):
                        self._conn.rollback()
                        return {"ok": False, "reason": "support_not_kept_long"}
                    self._conn.execute(
                        "DELETE FROM recognition_evidence WHERE recognition_sha256=?", (body_hash,))
                    for eid in ids:
                        self._conn.execute(
                            "INSERT INTO recognition_evidence"
                            "(recognition_sha256,event_id,review_sha256,linked_at)"
                            " VALUES(?,?,?,?)",
                            (body_hash, eid, review_sha, datetime.now().astimezone().isoformat()))
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    return {"ok": False, "reason": "support_commit_failed"}
            return {"ok": True, "recognition_sha256": body_hash, "evidence_ids": ids,
                    "evidence_count": len(ids)}

        def pending_grounded_evidence_ids(self) -> set[int]:
            """Pin only evidence referenced by an actually staged USER proposal.

            Reviews alone are not permanent retention instructions. Explicit user forgetting
            still bypasses the automatic compact protection.
            """
            assert self._conn is not None
            try:
                from tools import write_approval as wa
                proposals = wa.list_pending(wa.MEMORY)
            except Exception:
                proposals = None
            if proposals is None:
                rows = self._conn.execute(
                    "SELECT evidence_group FROM event_reviews WHERE source_kind='user_grounded'"
                ).fetchall()
            else:
                rows = []
                for proposal in proposals:
                    cls = (proposal.get("payload") or {}).get("classification") or {}
                    if cls.get("source_kind") != "user_grounded":
                        continue
                    row = self._conn.execute(
                        "SELECT evidence_group FROM event_reviews "
                        "WHERE event_id=? AND review_sha256=? AND source_kind='user_grounded'",
                        (cls.get("event_id"), cls.get("review_sha256"))).fetchone()
                    if row is not None:
                        rows.append(row)
            ids: set[int] = set()
            for row in rows:
                try:
                    group = json.loads(str(row["evidence_group"] or ""))
                    if isinstance(group, list) and (2 <= len(group) <= 3):
                        items = [int(x["event_id"]) for x in group]
                        if all(i > 0 for i in items) and len(set(items)) == len(items):
                            ids.update(items)
                except (TypeError, ValueError, KeyError):
                    continue
            return ids

        def put_event_review(self, event_id: int, *, target: str, duty: str, content_sha256: str,
                             event_sha256: str, event_version: str = "",
                             baseline_sha256: str = "", source_kind: str = "unknown",
                             significance: str = "", reviewed_text: str = "", now: str = "",
                             evidence_group: str = "") -> str:
            """保全后端生成的语义审核记录，返回**不透明记录 id**。

            记录 id 只是回读键；重放时按落库正文 hash、目标、证据组（事件全文／版本／来源快照
            hash）与常驻 baseline 逐项核对，不再重算拼接摘要。"""
            assert self._conn is not None
            significance = str(significance or "")[:200]
            review_sha = uuid.uuid4().hex
            with self._lock:
                self._conn.execute(
                    """INSERT INTO event_reviews
                       (event_id, review_sha256, target, duty, content_sha256, event_sha256,
                        event_version, baseline_sha256, source_kind, significance, reviewed_text,
                        evidence_group, reviewed_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (int(event_id), review_sha, str(target), str(duty), str(content_sha256),
                     str(event_sha256), str(event_version or ""), str(baseline_sha256 or ""),
                     str(source_kind or "unknown"), significance,
                     str(reviewed_text or ""),
                     str(evidence_group or ""),
                     now or datetime.now().astimezone().isoformat(timespec="seconds")),
                )
                self._conn.commit()
            return review_sha

        def get_event_review(self, event_id: int, review_sha256: str = "") -> Optional[sqlite3.Row]:
            """按（事件 id，可选记录 id）**纯读回**审核记录；取不到返回 None。

            · 给了 `review_sha256` ⇒ 只认那一条（调用方声明的审核结论必须可回读）；
            · 没给 ⇒ 回该事件**最近一次**审核记录。
            """
            assert self._conn is not None
            if isinstance(event_id, bool) or not isinstance(event_id, int):
                return None
            try:
                if review_sha256:
                    return self._conn.execute(
                        "SELECT * FROM event_reviews WHERE event_id=? AND review_sha256=?",
                        (int(event_id), str(review_sha256)),
                    ).fetchone()
                return self._conn.execute(
                    "SELECT * FROM event_reviews WHERE event_id=? "
                    "ORDER BY reviewed_at DESC, rowid DESC LIMIT 1",
                    (int(event_id),),
                ).fetchone()
            except Exception:
                return None

        def find_event_review(self, content_sha256: str, target: str) -> Optional[sqlite3.Row]:
            """按（候选正文 hash，目标档）**纯读回**后端审核记录；取不到返回 None。

            为什么需要它：宿主真实入口（`agent/inline_tool_executors.py::_memory`）把实际
            `UserEvidence`／`PluginEventTarget` 传进 memory 工具，但**不透传 classification**；
            重放一条来源已确认的提案时，只能按**确切候选正文＋目标档**回读后端是否真的保全过
            这条候选的语义审核记录。这里只读、不生成、不新增调用；查到也**不等于**授权——
            调用方还要逐项核验事件全文／版本／常驻 baseline（见宿主 `_long_term_evidence_guard`）。

            同一（正文，目标）有多条时取**最近一次**审核；旧基线的那条会在常驻 baseline 核验里
            被挡下并要求重新审核。
            """
            assert self._conn is not None
            body = str(content_sha256 or "").strip()
            tgt = str(target or "").strip()
            if not body or not tgt:
                return None
            try:
                return self._conn.execute(
                    "SELECT * FROM event_reviews WHERE content_sha256=? AND target=? "
                    "ORDER BY reviewed_at DESC, rowid DESC LIMIT 1",
                    (body, tgt),
                ).fetchone()
            except Exception:
                return None
