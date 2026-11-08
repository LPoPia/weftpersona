from __future__ import annotations

import sqlite3

from typing import Any, Dict, List, Optional, Tuple

class StateStoreMixin:
        def _migrate_daily_context_schema(self) -> None:
            assert self._conn is not None
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(daily_context)")}
            if "source_kind" not in columns:
                with self._lock:
                    self._conn.execute(
                        "ALTER TABLE daily_context ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'statement'")
                    self._conn.commit()
            if "author_kind" not in columns:
                with self._lock:
                    self._conn.execute(
                        "ALTER TABLE daily_context ADD COLUMN author_kind TEXT NOT NULL DEFAULT 'unknown'")
                    self._conn.commit()


        def open_daily_entry(
            self, *, owner_key: str, profile_key: str, kind: str, slot: str, text: str,
            opened_on: str, now: str, expires_on: Optional[str] = None,
            source_quote: str = "", source_turn: str = "", source_kind: str = "statement",
            author_kind: str = "unknown",
        ) -> Optional[int]:
            """插入一条新的 active 行；作用域缺失返回 None；同槽已 active 会抛 IntegrityError。"""
            assert self._conn is not None
            if not (owner_key and profile_key):
                return None
            with self._lock:
                cursor = self._conn.execute(
                    "INSERT INTO daily_context "
                    "(owner_key, profile_key, kind, slot, text, opened_on, expires_on, "
                    " source_quote, source_turn, source_kind, author_kind, "
                    " status, version, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,'active',1,?,?)",
                    (owner_key, profile_key, kind, slot, text, opened_on, expires_on,
                     source_quote, source_turn, source_kind, author_kind, now, now),
                )
                self._conn.commit()
                return int(cursor.lastrowid)

        def find_active_daily_entry(
            self, *, owner_key: str, profile_key: str, kind: str, slot: str,
        ) -> Optional[sqlite3.Row]:
            assert self._conn is not None
            if not (owner_key and profile_key):
                return None
            return self._conn.execute(
                "SELECT * FROM daily_context WHERE owner_key=? AND profile_key=? "
                "AND kind=? AND slot=? AND status='active'",
                (owner_key, profile_key, kind, slot),
            ).fetchone()

        def list_active_daily_entries(
            self, *, owner_key: str, profile_key: str, today: str,
        ) -> List[sqlite3.Row]:
            """本作用域的有效行；DAILY_FACT 只留当天（即使惰性清理还没跑到）。"""
            assert self._conn is not None
            if not (owner_key and profile_key):
                return []
            return self._conn.execute(
                "SELECT * FROM daily_context WHERE owner_key=? AND profile_key=? AND status='active' "
                "AND (kind<>'DAILY_FACT' OR opened_on=?) "
                "ORDER BY CASE kind WHEN 'ACTIVE_CONSTRAINT' THEN 0 "
                "WHEN 'OPEN_THREAD' THEN 1 ELSE 2 END, entry_id",
                (owner_key, profile_key, today),
            ).fetchall()

        def list_daily_entries_for_day(
            self, *, owner_key: str, profile_key: str, day: str, kind: Optional[str] = None,
            include_statuses: Tuple[str, ...] = ("active",), source_kind: Optional[str] = None,
        ) -> List[sqlite3.Row]:
            """按 `opened_on` 取某一天的行（昨天概况用）。

            **不删不改**：`include_statuses` 只决定读哪些状态（跨日之后 DAILY_FACT 已是
            `expired`，记录仍在，所以昨天概况要连 `expired` 一起读）。
            """
            assert self._conn is not None
            if not (owner_key and profile_key and day):
                return []
            clauses = ["owner_key=?", "profile_key=?", "opened_on=?"]
            params: List[Any] = [owner_key, profile_key, day]
            if kind:
                clauses.append("kind=?")
                params.append(kind)
            statuses = tuple(str(value) for value in (include_statuses or ()) if value)
            if statuses:
                clauses.append("status IN (%s)" % ",".join("?" * len(statuses)))
                params.extend(statuses)
            if source_kind:
                clauses.append("source_kind=?")
                params.append(source_kind)
            return self._conn.execute(
                "SELECT * FROM daily_context WHERE " + " AND ".join(clauses)
                + " ORDER BY entry_id",
                params,
            ).fetchall()

        def list_daily_entries_for_recall(
            self, *, owner_key: str, profile_key: str, since: str,
        ) -> List[sqlite3.Row]:
            """Read-only daily rows for association; scoped owner+profile required.

            Includes active/closed/expired rows (not the superseded old versions) so a
            closed exam or an old work state can still be understood as history. It
            never writes and never treats a daily row as a real-world ongoing state.
            `OPEN_THREAD` is her own first-person curiosity, not a fact about him, so it
            never enters association recall (read it via `active_curiosity_threads`).
            """
            assert self._conn is not None
            if not (owner_key and profile_key and since):
                return []
            since_date = str(since)[:10]
            return self._conn.execute(
                "SELECT * FROM daily_context WHERE owner_key=? AND profile_key=? "
                "AND status<>'superseded' AND kind<>'OPEN_THREAD' "
                "AND (created_at>=? OR opened_on>=?) "
                "ORDER BY COALESCE(updated_at, created_at, opened_on) DESC, entry_id DESC",
                (owner_key, profile_key, str(since), since_date),
            ).fetchall()

        def daily_active_stats(
            self, *, owner_key: str, profile_key: str, exclude_entry_id: Optional[int] = None,
        ) -> dict:
            """准入用的有效条数/文本预算（可按 entry_id 排除一条，供原位更新重算）。"""
            assert self._conn is not None
            if not (owner_key and profile_key):
                return {"counts": {}, "chars": 0}
            clauses = ["owner_key=?", "profile_key=?", "status='active'"]
            params: List[Any] = [owner_key, profile_key]
            if exclude_entry_id is not None:
                clauses.append("entry_id<>?")
                params.append(int(exclude_entry_id))
            rows = self._conn.execute(
                "SELECT kind, text FROM daily_context WHERE " + " AND ".join(clauses), params
            ).fetchall()
            counts: Dict[str, int] = {}
            chars = 0
            for row in rows:
                counts[str(row["kind"])] = counts.get(str(row["kind"]), 0) + 1
                chars += len(str(row["text"] or ""))
            return {"counts": counts, "chars": chars}

        def cas_supersede_daily_entry(
            self, *, owner_key: str, profile_key: str, entry_id: int, version: int, status: str, now: str,
        ) -> int:
            """版本条件关闭（superseded/closed/expired）：rowcount==0 ⇒ 目标已变，调用方重读再决。"""
            assert self._conn is not None
            if not (owner_key and profile_key):
                return 0
            with self._lock:
                cursor = self._conn.execute(
                    "UPDATE daily_context SET status=?, version=version+1, updated_at=? "
                    "WHERE entry_id=? AND owner_key=? AND profile_key=? AND version=? AND status='active'",
                    (status, now, int(entry_id), owner_key, profile_key, int(version)),
                )
                self._conn.commit()
                return max(0, int(cursor.rowcount))

        def cas_update_daily_entry(
            self, *, owner_key: str, profile_key: str, entry_id: int, version: int, text: str,
            opened_on: str, expires_on: Optional[str], source_quote: str, source_turn: str, now: str,
            source_kind: str = "statement",
        ) -> int:
            """同槽原位更新：先把**旧版本**留在本表的 superseded 行，再 CAS 更新当前行。

            旧行的 text/引文/来源轮次/创建时间原样复制，因此更新不会丢原始证据；
            快照与 CAS 更新在同一事务：版本冲突（rowcount==0）整体回滚，不留伪历史。
            返回 1＝更新成功，0＝版本已变/行不 active（调用方按冲突处理）。
            """
            assert self._conn is not None
            if not (owner_key and profile_key):
                return 0
            with self._lock:
                try:
                    row = self._conn.execute(
                        "SELECT * FROM daily_context WHERE entry_id=? AND owner_key=? "
                        "AND profile_key=? AND version=? AND status='active'",
                        (int(entry_id), owner_key, profile_key, int(version)),
                    ).fetchone()
                    if row is None:
                        return 0
                    self._conn.execute(
                        "INSERT INTO daily_context "
                        "(owner_key, profile_key, kind, slot, text, opened_on, expires_on, "
                        " source_quote, source_turn, source_kind, author_kind, "
                        " status, version, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,'superseded',?,?,?)",
                        (row["owner_key"], row["profile_key"], row["kind"], row["slot"],
                         row["text"], row["opened_on"], row["expires_on"], row["source_quote"],
                         row["source_turn"], row["source_kind"], row["author_kind"],
                         int(row["version"]), row["created_at"], row["updated_at"]),
                    )
                    cursor = self._conn.execute(
                        "UPDATE daily_context SET text=?, opened_on=?, expires_on=?, source_quote=?, "
                        "source_turn=?, source_kind=?, author_kind=?, version=version+1, updated_at=? "
                        "WHERE entry_id=? AND owner_key=? AND profile_key=? AND version=? "
                        "AND status='active'",
                        (text, opened_on, expires_on, source_quote, source_turn, source_kind,
                         row["author_kind"], now,
                         int(entry_id), owner_key, profile_key, int(version)),
                    )
                    if max(0, int(cursor.rowcount)) != 1:
                        self._conn.rollback()
                        return 0
                    self._conn.commit()
                    return 1
                except Exception:
                    self._conn.rollback()
                    raise

        def expire_stale_daily_facts(self, *, owner_key: str, profile_key: str, today: str, now: str) -> int:
            """本表内的惰性清理：① 跨日的 DAILY_FACT ② 用户明确给过且已过期的行。

            只看这一张表；不动 events、不删行（改 status='expired'）。
            """
            assert self._conn is not None
            if not (owner_key and profile_key):
                return 0
            with self._lock:
                cursor = self._conn.execute(
                    "UPDATE daily_context SET status='expired', version=version+1, updated_at=? "
                    "WHERE owner_key=? AND profile_key=? AND status='active' "
                    "AND ( (kind='DAILY_FACT' AND opened_on<>?) "
                    "      OR (expires_on IS NOT NULL AND expires_on<>'' AND expires_on<?) )",
                    (now, owner_key, profile_key, today, today),
                )
                self._conn.commit()
                return max(0, int(cursor.rowcount))
