from __future__ import annotations

import hashlib
import sqlite3

from datetime import datetime, timedelta
from typing import Any, List, Optional

try:
    from .curve import (
        CURVE_COUNTED_VIAS, CURVE_MIN_GAP, CURVE_PROMOTE_RECALLS, CURVE_WAKE_VIAS,
        bounded_half_life, strengthened_half_life, terminal_ttl,
    )
except ImportError:
    from curve import (
        CURVE_COUNTED_VIAS, CURVE_MIN_GAP, CURVE_PROMOTE_RECALLS, CURVE_WAKE_VIAS,
        bounded_half_life, strengthened_half_life, terminal_ttl,
    )

SUMMARY_SOURCE_HISTORICAL = "historical_summary_only"
SUMMARY_SOURCE_DISTILLED = "distilled"
SUMMARY_SOURCE_ORIGINAL = "original"
SUMMARY_SOURCE_GIST_FADED = "gist_faded"
GIST_FADED_NOTICE = "（大概印象，具体原话记不清了）"

_VISIBLE_TTL = """(ttl_expire IS NULL OR julianday(?) < CASE
    WHEN ttl_expire=curve_ttl THEN MAX(julianday(ttl_expire),
        julianday(COALESCE(NULLIF(last_active_at,''),NULLIF(updated_at,''),created_at))
        + half_life_days * log2(MAX(0.15,importance)/0.15))
    ELSE julianday(ttl_expire) END)"""


def gist_provenance(item: dict, bits: list[str], *, compact: bool = False) -> str:
    if item.get("tier") == "dormant":
        return "〔" + "，".join(["久远记忆｜被他提醒才想起来的大概印象", *bits]) + "〕"
    if item.get("summary_source") == SUMMARY_SOURCE_GIST_FADED:
        separator = "｜" if compact and bits else "，"
        return "〔我记着的大概印象" + separator + "，".join([*bits, "具体原话记不清了"]) + "〕"
    return ""


HISTORICAL_LIMIT_NOTICE = "（历史概括，原话未单独留存）"

_PERMANENT_TIERS = ("long", "important", "distilled")

class EventStoreMixin:
        def mark_asked(self, ids: List[int], via: str = "", session_id: str = "") -> int:
            """标记这些行已被她读到。

            顺带落账：`asked_via` 记通道码，`asked_session` 记会话 id 的 sha256 前 16 位
            （不落明文）。两列只作观测：不进上下文、不参与排序与判据。
            通道码语汇：`scheduled`／`explicit_past`／`gap`／`tool_search`。
            """
            assert self._conn is not None
            if not ids:
                return 0
            now = datetime.now().astimezone().isoformat()
            session_digest = (
                hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16] if session_id else ""
            )
            placeholders = ",".join("?" for _ in ids)
            with self._lock:
                cursor = self._conn.execute(
                    f"UPDATE events SET asked=1, asked_at=?, asked_via=?, asked_session=? "
                    f"WHERE id IN ({placeholders})",
                    [now, str(via or ""), session_digest, *ids],
                )
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def reset_asked_all(self) -> int:
            assert self._conn is not None
            with self._lock:
                cursor = self._conn.execute(
                    "UPDATE events SET asked=0, asked_via='', asked_session=''"
                )
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def reinforce_on_recall(self, ids: List[int], via: str = "",
                               now: Optional[datetime] = None, session_id: str = "") -> int:
            """跨日有效提取强化（间隔效应）：被真正聊起一次，就把这条记忆拉长一点。

            只有 `curve.CURVE_COUNTED_VIAS`（私聊联想 chat／明确追问 explicit_past）且距上次
            活跃满 12 小时才累加一次 `recall_count`：同一晚反复追问不算间隔重提，隔夜自动塞入的
            `gap`／`scheduled` 线索也不算用户主动想起。`tool_search` 只用来把 `dormant` 的深层
            记忆翻出来（唤醒成 `long`），不给活事件刷分。

            职责边界：只改 `recall_count / half_life_days / last_active_at / ttl_expire /
            curve_ttl` 与晋升字段，**绝不清零 `asked`**（那是 `mark_asked` 的职责）。永久层级
            （long/important/distilled/dormant）强化后不重新挂 TTL。公开回合、非 primary 上下文、
            `T:` 测试会话，以及不属于当前 owner 的行一律不写，返回实际强化行数。
            """
            assert self._conn is not None
            if not ids or str(via or "") not in CURVE_WAKE_VIAS:
                return 0
            if (self._agent_context != "primary" or self._is_system_turn(session_id)
                    or self._is_system_turn(self._session_id)):
                return 0
            try:
                from .source_context import in_public_interaction
            except ImportError:
                from source_context import in_public_interaction
            if in_public_interaction():
                return 0
            current = self._as_local_time(now)
            unique = sorted({int(item) for item in ids
                             if isinstance(item, int) and not isinstance(item, bool)})
            if not unique:
                return 0
            placeholders = ",".join("?" for _ in unique)
            reinforced = 0
            with self._lock, self._conn:
                self._conn.execute('BEGIN IMMEDIATE')
                rows = self._conn.execute(
                    f"SELECT id,tier,recall_count,half_life_days,last_active_at,updated_at,created_at,ttl_expire "
                    f"FROM events WHERE id IN ({placeholders}) AND blacklist=0 AND owner_key=?",
                    [*unique, str(self._owner_key)],
                ).fetchall()
                for row in rows:
                    tier = str(row["tier"] or "")
                    dormant = tier == "dormant"
                    if ((tier in ("short", "pending") and row["ttl_expire"] is None)
                                or (dormant and via not in {"explicit_past", "tool_search"})
                                or (not dormant and via not in CURVE_COUNTED_VIAS)):
                        continue
                    if current - self._row_last_active(row) < CURVE_MIN_GAP:
                        continue
                    count = int(row["recall_count"] or 0) + 1
                    next_tier = tier
                    reason = ""
                    if dormant:
                        next_tier, reason = "long", "dormant_wake"
                    elif tier in ("short", "pending") and count >= CURVE_PROMOTE_RECALLS:
                        next_tier, reason = "long", "spaced_recall"
                    # 强化按**最终层级**取上界：留在短期/待定按 60，晋升/唤醒进长期按 120。
                    half_life = bounded_half_life(
                        strengthened_half_life(row["half_life_days"], row["recall_count"]),
                        next_tier)
                    tier = next_tier
                    expire = (None if tier in _PERMANENT_TIERS
                              else terminal_ttl(current, half_life).isoformat())
                    fields = ("recall_count=?, half_life_days=?, last_active_at=?, ttl_expire=?, "
                              "curve_ttl=?, expired_at=NULL, tier=?")
                    params: List[Any] = [count, half_life, current.isoformat(), expire,
                                         expire or "", tier]
                    if reason:
                        fields += ", promoted_at=?, promoted_reason=?, pending_to=''"
                        params.extend([current.isoformat(), reason])
                    params.append(int(row["id"]))
                    self._conn.execute(f"UPDATE events SET {fields} WHERE id=?", params)
                    reinforced += 1
                if reinforced:
                    self._conn.commit()
            return reinforced

        @staticmethod
        def _as_local_time(value: Any) -> datetime:
            current = value
            if isinstance(value, str) and value:
                try:
                    current = datetime.fromisoformat(value)
                except ValueError:
                    current = None
            if not isinstance(current, datetime):
                current = datetime.now().astimezone()
            return current if current.tzinfo else current.replace(
                tzinfo=datetime.now().astimezone().tzinfo)

        @classmethod
        def _row_last_active(cls, row: sqlite3.Row) -> datetime:
            anchor = str(row["last_active_at"] or "") or str(row["updated_at"] or row["created_at"] or "")
            return cls._as_local_time(anchor)

        @property
        def _owner_scoped(self) -> bool:
            return bool(self._owner_key) and self._agent_context == "primary"

        @staticmethod
        def _read_scope_clause(owner_key: Optional[str], *, include_shared: bool = True):
            owner = str(owner_key or "")
            self_clause = (
                "(experience_subject='sanae' AND fact_subject='sanae' "
                "AND json_valid(read_scope) "
                "AND json_extract(read_scope,'$.kind')='self')")
            if not include_shared:
                return ((f"(owner_key=? OR {self_clause})", [owner]) if owner
                        else (self_clause, []))
            shared_clause = "(owner_key<>? AND shared_view<>'')"
            if owner:
                return (f"(owner_key=? OR {self_clause} OR {shared_clause})",
                        [owner, owner])
            return (f"({self_clause} OR {shared_clause})", [""])

        def find_duplicate_candidates(
            self, session_id: str, cutoff: str, unscoped: bool = False,
            owner_key: Optional[str] = None,
            experience_subject: Optional[str] = None,
            fact_subject: Optional[str] = None,
            involved_owner_keys: Optional[str] = None,
            read_scope: Optional[str] = None, fingerprint: str = '', body: str = '',
        ) -> List[sqlite3.Row]:
            assert self._conn is not None
            clauses = [
                "blacklist=0", "tier<>'dormant'",
                "((session_id=? AND created_at>=?) OR (tier='short' AND expired_at IS NOT NULL)"
                " OR (?>'' AND (original_fingerprint=? OR (original_fingerprint='' "
                "AND substr(raw,instr(raw,'] ')+2)=?))))",
            ]
            params: List[Any] = [session_id, cutoff, fingerprint, fingerprint, body]
            if not unscoped:
                clauses.append("owner_key=?")
                params.append(self._owner_key if owner_key is None else owner_key)
            for column, value in (
                    ("experience_subject", experience_subject),
                    ("fact_subject", fact_subject),
                    ("involved_owner_keys", involved_owner_keys),
                    ("read_scope", read_scope)):
                if value is not None:
                    clauses.append(f"{column}=?")
                    params.append(value)
            return self._conn.execute(
                "SELECT id, keywords, importance, expired_at, source_context FROM events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY (original_fingerprint=? AND original_fingerprint<>'') DESC, "
                "(expired_at IS NOT NULL) DESC, id DESC LIMIT 20",
                [*params, fingerprint],
            ).fetchall()

        def query_public_projections(self, now: str, unscoped: bool = False,
                                     owner_key: Optional[str] = None,
                                     cross_owner_self: bool = False
                                     ) -> List[sqlite3.Row]:
            """公开面唯一可读的行：**只有**已被可信层审阅出公开投影的事件。

            ``public_excerpt`` 为空（含全部旧行）⇒ 公开检索看不见这一行。查询**只**取投影与
            出处元数据，不取 ``raw``／``summary``／``vec`` —— 公开结果不得从私人原文衍生。
            """
            assert self._conn is not None
            clauses = [
                "blacklist=0", "tier<>'dormant'", "public_excerpt<>''", "expired_at IS NULL",
                _VISIBLE_TTL,
            ]
            params: List[Any] = [now]
            if not unscoped:
                owner = self._owner_key if owner_key is None else owner_key
                if cross_owner_self:
                    clauses.append(
                        "(owner_key=? OR (json_valid(source_context) "
                        "AND json_extract(source_context,'$.kind')='self'))")
                else:
                    clauses.append("owner_key=?")
                params.append(owner)
            return self._conn.execute(
                "SELECT id, public_excerpt, source_context, created_at FROM events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY id DESC LIMIT 500",
                params,
            ).fetchall()

        def query_private_peer_events(self, now: str, peer: str,
                                      owner_key: Optional[str] = None) -> List[sqlite3.Row]:
            """Read only this peer's attributed DM events; never load other private rows."""
            assert self._conn is not None
            return self._conn.execute(
                "SELECT id, raw AS public_excerpt, source_context, created_at FROM events "
                "WHERE owner_key=? AND blacklist=0 AND expired_at IS NULL "
                f"AND tier<>'dormant' AND {_VISIBLE_TTL} "
                "AND json_valid(source_context) "
                "AND json_extract(source_context,'$.visibility')='private' "
                "AND json_extract(source_context,'$.platform')='bilibili' "
                "AND json_extract(source_context,'$.kind')='external_user' "
                "AND json_extract(source_context,'$.author_id')=? "
                "AND json_extract(source_context,'$.ref') LIKE 'bilibili:dm:%' "
                "ORDER BY id DESC LIMIT 500",
                (self._owner_key if owner_key is None else owner_key, now, peer),
            ).fetchall()

        def set_public_excerpt(self, event_id: int, excerpt: str) -> bool:
            """**host-only**：给某条事件设置／清空公开投影（`excerpt` 空串＝撤销公开视图）。

            模型面到不了这里：它既不在 `get_tool_schemas()` 里，也不在 `handle_tool_call()` 的
            分派表里。设置本身＝可信层显式审阅这一事实，不是模型自标 ``public=true``；写入只动
            ``public_excerpt`` 一列，原文、摘要与出处一字不改。返回是否命中一行。
            """
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT 1 FROM events WHERE id=? AND blacklist=0", (int(event_id),)).fetchone()
            if row is None:
                return False
            with self._lock:
                self._conn.execute(
                    "UPDATE events SET public_excerpt=?, updated_at=? WHERE id=? AND blacklist=0",
                    (str(excerpt or ""), datetime.now().astimezone().isoformat(), int(event_id)),
                )
                self._conn.commit()
            return True

        def set_shared_view(self, event_id: int, text: str) -> bool:
            """Host-only neutral view for another user's shared experience.

            It updates only ``shared_view``; raw/summary/source/vector stay untouched.
            The caller is the trusted host, never a model tool; the text must be
            neutral and must not be derived from the other user's private content.
            """
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT 1 FROM events WHERE id=? AND blacklist=0", (int(event_id),)).fetchone()
            if row is None:
                return False
            with self._lock:
                self._conn.execute(
                    "UPDATE events SET shared_view=?, updated_at=? WHERE id=? AND blacklist=0",
                    (str(text or ""), datetime.now().astimezone().isoformat(), int(event_id)),
                )
                self._conn.commit()
            return True

        def event_source_context(self, event_id: int) -> Optional[dict]:
            try:
                from .source_context import parse_source
            except ImportError:
                from source_context import parse_source
            row = self._conn.execute("SELECT source_context FROM events WHERE id=?", (int(event_id),)).fetchone()
            return parse_source(row["source_context"]) if row else None

        def query_tier_window(
            self, now: str, window_start: str, topic: str = "", created_after: Optional[str] = None,
            unscoped: bool = False, owner_key: Optional[str] = None,
            experience_subject: Optional[str] = None,
            fact_subject: Optional[str] = None, read_context: bool = False,
            writable: bool = False,
        ) -> List[sqlite3.Row]:
            assert self._conn is not None
            asked_before = (datetime.fromisoformat(now) - timedelta(hours=24)).isoformat()
            clauses = [
                "blacklist=0", "expired_at IS NULL", _VISIBLE_TTL,
                "(asked=0 OR asked_at='' OR asked_at<?)",
                "tier<>'dormant'",
                "(tier IN ('important', 'long', 'distilled') OR "
                "CASE WHEN recall_count>0 THEN COALESCE(NULLIF(last_active_at,''),NULLIF(updated_at,''),created_at) "
                "ELSE created_at END>=?)",
            ]
            params: List[Any] = [now, asked_before, window_start]
            if read_context or writable:
                scope_clause, scope_params = self._read_scope_clause(
                    owner_key, include_shared=not writable)
                clauses.append(scope_clause)
                params.extend(scope_params)
            elif not unscoped:
                clauses.append("owner_key=?")
                params.append(self._owner_key if owner_key is None else owner_key)
            if experience_subject is not None:
                clauses.append("experience_subject=?")
                params.append(experience_subject)
            if fact_subject is not None:
                clauses.append("fact_subject=?")
                params.append(fact_subject)
            if topic:
                clauses.append("topic=?")
                params.append(topic)
            if created_after is not None:
                clauses.append("created_at>=?")
                params.append(created_after)
            return self._conn.execute(
                "SELECT * FROM events WHERE " + " AND ".join(clauses) + " ORDER BY id DESC LIMIT 500", params
            ).fetchall()

        def query_events(
            self, now: str, topic: str = "", created_after: Optional[str] = None,
            include_expiring: bool = False, unscoped: bool = False,
            owner_key: Optional[str] = None,
            experience_subject: Optional[str] = None,
            fact_subject: Optional[str] = None, read_context: bool = False,
            writable: bool = False, dormant_owner: Optional[str] = None,
        ) -> List[sqlite3.Row]:
            assert self._conn is not None
            clauses = ["blacklist=0", _VISIBLE_TTL]
            if dormant_owner is None:
                clauses.append("tier<>'dormant'")
            else:
                clauses.append("(tier<>'dormant' OR owner_key=?)")
            if not include_expiring:
                clauses.append("expired_at IS NULL")
            params: List[Any] = [now] if dormant_owner is None else [now, dormant_owner]
            if read_context or writable:
                scope_clause, scope_params = self._read_scope_clause(
                    owner_key, include_shared=not writable)
                clauses.append(scope_clause)
                params.extend(scope_params)
            elif not unscoped:
                clauses.append("owner_key=?")
                params.append(self._owner_key if owner_key is None else owner_key)
            if experience_subject is not None:
                clauses.append("experience_subject=?")
                params.append(experience_subject)
            if fact_subject is not None:
                clauses.append("fact_subject=?")
                params.append(fact_subject)
            if topic:
                clauses.append("topic=?")
                params.append(topic)
            if created_after is not None:
                clauses.append("created_at>=?")
                params.append(created_after)
            return self._conn.execute(
                "SELECT * FROM events WHERE " + " AND ".join(clauses) + " ORDER BY id DESC LIMIT 500", params
            ).fetchall()

        def insert_event_row(
            self, fingerprint: str, values: tuple, appraisal: Optional[dict] = None, owner_key: str = "",
            summary: str = "", summary_source: str = SUMMARY_SOURCE_ORIGINAL,
            source_snapshot: str = "", source_snapshot_sha256: str = "",
            source_context: str = "", experience_subject: str = "",
            fact_subject: str = "", involved_owner_keys: str = "", read_scope: str = "",
            half_life_days: float = 14.0, last_active_at: str = "", curve_ttl: str = "",
        ) -> Optional[int]:
            """插入一行事件。

            `values` 是 `(ts, raw, keywords, importance, confidence, topic, ttl_expire,
            created_at, vec, session_id)`；`summary` 只落展示／检索摘要列，raw 保持原文全文。
            缺省空串 ⇒ 读面回退 raw，旧调用方行为不变。
            `summary_source` 标注摘要来源（新写默认 original）。
            `source_snapshot`／`source_snapshot_sha256` 只在内部 grounded 事实写入口绑定；
            公开 sanae_mem_add 不传 ⇒ 留空＝unknown、永不补资格。
            `source_context` 是这条事件当时的可信出处（由入库入口从 ContextVar 取，
            不采信调用参数／模型参数）；空串＝旧行 unknown。
            `half_life_days` 是初始半衰期 S0；`curve_ttl` 只在自然写入时镜像 `ttl_expire`，
            显式改期传空串（见 curve.py）。`last_active_at` 缺省＝插入时刻。
            """
            assert self._conn is not None
            created_at = str(values[7] or "")
            with self._lock:
                if self._conn.execute("SELECT 1 FROM forgotten WHERE fingerprint=?", (fingerprint,)).fetchone():
                    return None
                cursor = self._conn.execute(
                    """INSERT INTO events
                       (ts, raw, summary, summary_source, keywords, importance, confidence, topic, ttl_expire,
                        created_at, vec, session_id, a_serious, a_intent, a_rel, a_self, a_conflict, affect,
                        owner_key, source_snapshot, source_snapshot_sha256, source_context,
                        experience_subject, fact_subject, involved_owner_keys, read_scope,
                        last_active_at, half_life_days, curve_ttl, original_fingerprint)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?)""",
                    tuple(values[:2]) + (str(summary or ""), str(summary_source or ""))
                    + tuple(values[2:]) + tuple(
                        int((appraisal or {}).get(name, 0))
                        for name in ("a_serious", "a_intent", "a_rel", "a_self", "a_conflict", "affect")
                    ) + (str(owner_key or ""), str(source_snapshot or ""),
                         str(source_snapshot_sha256 or ""), str(source_context or ""),
                         str(experience_subject or ""), str(fact_subject or ""),
                         str(involved_owner_keys or ""), str(read_scope or ""),
                         str(last_active_at or created_at), float(half_life_days),
                         str(curve_ttl or ""), fingerprint),
                )
                event_id = int(cursor.lastrowid)
                self._conn.commit()
            return event_id

        def event_ids_existing(self, ids: List[int],
                               owner_key: Optional[str] = None,
                               experience_subject: Optional[str] = None,
                               fact_subject: Optional[str] = None,
                               writable: bool = False) -> List[int]:
            assert self._conn is not None
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            sql = f"SELECT id FROM events WHERE id IN ({placeholders})"
            params: List[Any] = list(ids)
            if writable:
                clause, scope_params = self._read_scope_clause(
                    owner_key, include_shared=False)
                sql += " AND " + clause
                params.extend(scope_params)
            elif owner_key is not None:
                sql += " AND owner_key=?"
                params.append(owner_key)
            if experience_subject is not None:
                sql += " AND experience_subject=?"
                params.append(experience_subject)
            if fact_subject is not None:
                sql += " AND fact_subject=?"
                params.append(fact_subject)
            rows = self._conn.execute(sql, params).fetchall()
            return [int(row["id"]) for row in rows]

        def get_event_row_by_id(self, event_id: int) -> Optional[sqlite3.Row]:
            """按真实事件 id 纯读取一行（受控迁移的目标精确回读）。

            与 ``reinforce_event`` 不同，它**不改 importance、不写 updated_at、不标记 asked**，
            只回 ``raw``/``topic`` 等原值；行不存在返回 ``None``。宿主用它验证「目标确实按 id
            写进去了」，不接受工具返回里的 ``stored: true`` 当证据。
            """
            assert self._conn is not None
            if isinstance(event_id, bool) or not isinstance(event_id, int):
                return None
            try:
                return self._conn.execute(
                    "SELECT id, raw, topic, importance, ts, source_context, "
                    "experience_subject, fact_subject FROM events WHERE id=?", (int(event_id),)
                ).fetchone()
            except Exception:
                return None

        def record_event_occurrence(self, event_id: int, raw: str, observed_at: str,
                                    source_snapshot: str = "",
                                    source_snapshot_sha256: str = "",
                                    preserve_previous: bool = False) -> bool:
            """Record a separate utterance under one searchable event, never infer source proof."""
            assert self._conn is not None
            if not raw:
                return False
            with self._lock:
                previous = self._conn.execute(
                    "SELECT raw,created_at,source_snapshot,source_snapshot_sha256 "
                    "FROM events WHERE id=? AND blacklist=0", (int(event_id),)).fetchone()
                if previous is None:
                    return False
                if (preserve_previous
                        and not self._conn.execute(
                            "SELECT 1 FROM event_occurrences WHERE event_id=? LIMIT 1",
                            (int(event_id),)).fetchone()):
                    self._conn.execute(
                        "INSERT INTO event_occurrences"
                        "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                        " VALUES(?,?,?,?,?)",
                        (int(event_id), previous["raw"], previous["created_at"],
                         previous["source_snapshot"], previous["source_snapshot_sha256"]))
                self._conn.execute(
                    "INSERT INTO event_occurrences"
                    "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                    " VALUES(?,?,?,?,?)",
                    (int(event_id), str(raw), str(observed_at),
                     str(source_snapshot or ""), str(source_snapshot_sha256 or "")))
                self._conn.commit()
            return True

        def query_event_occurrences(self, event_ids: List[int]) -> dict:
            """Recall-only read of the existing occurrence side records, oldest first.

            The event row keeps the first utterance; later utterances live in
            ``event_occurrences``. Retrieval must see the whole timeline instead of
            only the latest summary. This method never writes or backfills.
            """
            assert self._conn is not None
            ids = sorted({int(item) for item in event_ids if not isinstance(item, bool)})
            if not ids:
                return {}
            placeholders = ",".join("?" for _ in ids)
            rows = self._conn.execute(
                "SELECT event_id, raw, observed_at FROM event_occurrences "
                f"WHERE event_id IN ({placeholders}) ORDER BY observed_at, id",
                ids,
            ).fetchall()
            grouped: dict = {}
            for row in rows:
                grouped.setdefault(int(row["event_id"]), []).append(
                    {"time": str(row["observed_at"] or ""), "text": str(row["raw"] or "")}
                )
            return grouped

        def get_grounded_occurrence(self, event_id: int) -> Optional[dict]:
            """Return earliest captured, source-backed utterance; merged text is never evidence."""
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT raw, created_at, source_snapshot, source_snapshot_sha256, owner_key, "
                "source_context FROM events WHERE id=? AND blacklist=0", (int(event_id),)).fetchone()
            if row is None:
                return None
            original = str(row["raw"] or "")
            occurrences = self._conn.execute(
                "SELECT raw,observed_at,source_snapshot,source_snapshot_sha256 "
                "FROM event_occurrences WHERE event_id=? AND source_snapshot<>'' "
                "ORDER BY id", (int(event_id),)).fetchall()
            for item in occurrences:
                raw = str(item["raw"] or "")
                if (not raw or raw not in original or
                        hashlib.sha256(str(item["source_snapshot"]).encode("utf-8")).hexdigest()
                        != str(item["source_snapshot_sha256"] or "")):
                    return None
                return {
                    "raw": raw, "version": str(item["observed_at"] or ""),
                    "source_snapshot": str(item["source_snapshot"]),
                    "source_snapshot_sha256": str(item["source_snapshot_sha256"]),
                    "owner_key": str(row["owner_key"] or ""),
                    "source_context": str(row["source_context"] or "")}
            if not row["source_snapshot"]:
                return None
            snap = str(row["source_snapshot"])
            if hashlib.sha256(snap.encode("utf-8")).hexdigest() != str(
                    row["source_snapshot_sha256"] or ""):
                return None
            return {"raw": original, "version": str(row["created_at"] or ""),
                    "source_snapshot": snap,
                    "source_snapshot_sha256": str(row["source_snapshot_sha256"]),
                    "owner_key": str(row["owner_key"] or ""),
                    "source_context": str(row["source_context"] or "")}

        def append_event_evidence(self, event_id: int, fragment: str, now: str) -> int:
            """把新一次原文追加进既有事件行（近重复＝增证据/合并），不新建可召回项。

            `raw` 保留第一次原文；后续每次新说法以 `[补充 <ts>] <原文>` 追加在同一行内，
            `summary` 刷新为最新说法的摘要。返回 1／0。
            """
            assert self._conn is not None
            body = str(fragment or "").strip()
            if not body:
                return 0
            row = self._conn.execute("SELECT raw FROM events WHERE id=?", (int(event_id),)).fetchone()
            if row is None:
                return 0
            merged = f"{str(row['raw'] or '')}\n[补充 {now}] {body}"
            with self._lock:
                cursor = self._conn.execute(
                    "UPDATE events SET raw=?, summary=?, updated_at=? WHERE id=?",
                    (merged, body[:150], now, int(event_id)),
                )
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def reinforce_event(self, event_id: int, importance: float, now: str) -> Optional[sqlite3.Row]:
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT raw, summary, topic, ts, expired_at FROM events WHERE id=?", (event_id,)
            ).fetchone()
            if not row:
                return None
            with self._lock:
                self._conn.execute(
                    "UPDATE events SET importance=?, updated_at=? WHERE id=?", (importance, now, event_id)
                )
                self._conn.commit()
            return row

        def reinforce_expiring(self, event_id: int, now: str, ttl_expire: str) -> int:
            assert self._conn is not None
            with self._lock:
                cursor = self._conn.execute(
                    """UPDATE events
                       SET expired_at=NULL, ttl_expire=?, curve_ttl=?, asked=0, updated_at=?
                       WHERE id=? AND tier='short' AND blacklist=0 AND expired_at IS NOT NULL""",
                    (ttl_expire, ttl_expire, now, event_id),
                )
                self._conn.commit()
            return max(0, int(cursor.rowcount))

        def update_cross_session_hits(self, oldest: str, recent: str) -> None:
            assert self._conn is not None
            with self._lock:
                self._conn.execute(
                    """UPDATE events AS current
                       SET cross_session_hits=(
                         SELECT COUNT(DISTINCT previous.session_id)
                         FROM events AS previous
                         WHERE previous.keywords=current.keywords
                           AND previous.keywords<>''
                           AND previous.session_id<>current.session_id
                           AND previous.created_at>=?
                           AND previous.created_at<?
                            AND previous.blacklist=0
                            AND previous.expired_at IS NULL
                        )
                       WHERE current.blacklist=0 AND current.expired_at IS NULL""",
                    (oldest, recent),
                )
                self._conn.commit()
