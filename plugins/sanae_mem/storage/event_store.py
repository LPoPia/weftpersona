from __future__ import annotations

import hashlib
import json
import sqlite3

from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, List, Optional, TypedDict

from .curve import (
    CURVE_COUNTED_VIAS, CURVE_MIN_GAP, CURVE_WAKE_VIAS,
    bounded_half_life, strengthened_half_life, terminal_ttl,
)
from .sqlite_facade import SqliteFacade
from ..infra.source_descriptors import private_peer_route
from ..infra.event_fields import exclude_group_rows, is_group_owner, parse_source
from ..infra.host_bridge import get_hermes_home
from ..infra.chat_scope import current_chat_scope, in_public_interaction

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


class GroundedOccurrence(TypedDict):
    raw: str
    version: str
    source_snapshot: str
    source_snapshot_sha256: str
    owner_key: str
    source_context: str


class EventStore:
        """Event rows and occurrences on the shared SQLite facade with per-call authority."""

        def __init__(
            self, sqlite: SqliteFacade, *,
            current_session_id: Callable[[], str],
            current_agent_context: Callable[[], str],
            current_home: Callable[[], str],
            current_owner: Callable[[], str],
            current_read_owners: Callable[[], tuple[str, ...]],
            private_chat_owner: Callable[[], str],
            is_system_turn: Callable[[str], bool],
            group_scene_active: Callable[[], bool],
            group_read_owners: Callable[[], list[str]],
            group_write_owner_key: Callable[[], str],
            group_maintenance_owners: Callable[[], list[str]],
            group_resolution_owners: Callable[[], tuple[str, ...]],
        ) -> None:
            self._sqlite = sqlite
            self._current_session_id = current_session_id
            self._current_agent_context = current_agent_context
            self._current_home = current_home
            self._current_owner = current_owner
            self._current_read_owners = current_read_owners
            self._private_chat_owner = private_chat_owner
            self._is_system_turn = is_system_turn
            self._group_scene_active = group_scene_active
            self._group_read_owners = group_read_owners
            self._group_write_owner_key = group_write_owner_key
            self._group_maintenance_owners = group_maintenance_owners
            self._group_resolution_owners = group_resolution_owners

        def _cron_maintenance_authorized(self) -> bool:
            """True only for a trusted local/cron maintenance runtime in its own HOME."""
            if self._current_agent_context() != "cron":
                return False
            if in_public_interaction() or current_chat_scope().visibility != "local":
                return False
            configured_home = self._current_home()
            if configured_home:
                try:
                    if Path(get_hermes_home()).resolve() != Path(configured_home).resolve():
                        return False
                except Exception:
                    return False
            return True

        def private_owner_write_clause(self) -> Optional[tuple[str, list[str]]]:
            """Scope private lifecycle updates (asked/reset/promote) to actual private ownership.

            Never touches group rows; returns ``None`` when this turn holds no private authority.
            """
            if self._group_scene_active() or in_public_interaction():
                return None
            agent_context = self._current_agent_context()
            owner_key = self._current_owner()
            if agent_context == "primary":
                scope = current_chat_scope()
                if scope.visibility == "private" and not self._private_chat_owner():
                    return None
                owners = [
                    key for key in self.read_owner_keys_for()
                    if key and not is_group_owner(key)
                ]
                if not owners:
                    return None
                if len(owners) == 1:
                    return (f"owner_key=? AND {exclude_group_rows()}", [owners[0]])
                marks = ",".join("?" for _ in owners)
                return (f"owner_key IN ({marks}) AND {exclude_group_rows()}", owners)
            if agent_context == "cron":
                if not self._cron_maintenance_authorized():
                    return None
                if owner_key:
                    if is_group_owner(owner_key):
                        return None
                    return (f"owner_key=? AND {exclude_group_rows()}", [owner_key])
                return (exclude_group_rows(), [])
            return None

        def mark_asked(self, ids: List[int], via: str = "", session_id: str = "") -> int:
            assert self._sqlite.is_open
            if not ids:
                return 0
            scope = self.private_owner_write_clause()
            if scope is None:
                return 0
            scope_clause, scope_params = scope
            now = datetime.now().astimezone().isoformat()
            session_digest = (
                hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16] if session_id else ""
            )
            placeholders = ",".join("?" for _ in ids)
            with self._sqlite.transaction() as conn:
                cursor = conn.execute(
                    f"UPDATE events SET asked=1, asked_at=?, asked_via=?, asked_session=? "
                    f"WHERE id IN ({placeholders}) AND {scope_clause}",
                    [now, str(via or ""), session_digest, *ids, *scope_params],
                )
            return max(0, int(cursor.rowcount))

        @staticmethod
        def _asked_at_before_date(value: Any, before_date: str, local_tz: Any) -> bool:
            raw = str(value or "").strip()
            if not raw:
                return True
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                return True
            if parsed.tzinfo is not None and local_tz is not None:
                parsed = parsed.astimezone(local_tz)
            return parsed.date().isoformat() < before_date

        def reset_asked_all(self, *, before_date: Optional[str] = None) -> int:
            assert self._sqlite.is_open
            scope = self.private_owner_write_clause()
            if scope is None:
                return 0
            scope_clause, scope_params = scope
            with self._sqlite.transaction() as conn:
                if before_date is None:
                    cursor = conn.execute(
                        f"UPDATE events SET asked=0, asked_via='', asked_session='' WHERE {scope_clause}",
                        scope_params,
                    )
                    return max(0, int(cursor.rowcount))
                rows = conn.execute(
                    f"SELECT id, asked_at FROM events WHERE asked=1 AND {scope_clause}",
                    scope_params,
                ).fetchall()
                local_tz = datetime.now().astimezone().tzinfo
                stale_ids = [
                    int(row["id"]) for row in rows
                    if self._asked_at_before_date(row["asked_at"], before_date, local_tz)
                ]
                if not stale_ids:
                    return 0
                updated = 0
                for start in range(0, len(stale_ids), 500):
                    batch = stale_ids[start:start + 500]
                    marks = ",".join("?" for _ in batch)
                    cursor = conn.execute(
                        f"UPDATE events SET asked=0, asked_via='', asked_session='' "
                        f"WHERE id IN ({marks}) AND {scope_clause}",
                        [*batch, *scope_params],
                    )
                    updated += max(0, int(cursor.rowcount))
                return updated

        def reinforce_on_recall(self, ids: List[int], via: str = "",
                               now: Optional[datetime] = None, session_id: str = "") -> int:
            """跨日有效提取强化（间隔效应）：被真正聊起一次，就把这条记忆拉长一点。

            只有 `curve.CURVE_COUNTED_VIAS`（私聊联想 chat／明确追问 explicit_past）且距上次
            活跃满 12 小时才累加一次 `recall_count`：同一晚反复追问不算间隔重提，隔夜自动塞入的
            `gap`／`scheduled` 线索也不算用户主动想起。`tool_search` 同样不刷分。

            职责边界：
            **绝不改 `tier` 或 `promoted_*`**：长期晋升只走记忆主轴的三条真实通道
            （完整送达后的跨睡眠日再提及、inner_state 实际 CAS 后的强震动、她亲手珍藏），
            检索次数本身不再是一条晋升通道。`dormant` 行只读，不在这里被唤醒。公开回合、
            非 primary 上下文，以及不属于当前 owner 的行一律不写。
            """
            assert self._sqlite.is_open
            if not ids or str(via or "") not in CURVE_WAKE_VIAS:
                return 0
            if (self._current_agent_context() != "primary" or self._is_system_turn(session_id)
                    or self._is_system_turn(self._current_session_id())):
                return 0
            if in_public_interaction():
                return 0
            current = self._as_local_time(now)
            unique = sorted({int(item) for item in ids
                             if isinstance(item, int) and not isinstance(item, bool)})
            if not unique:
                return 0
            placeholders = ",".join("?" for _ in unique)
            reinforced = 0
            with self._sqlite.transaction() as conn:
                scope_clause, scope_params = self.read_scope_clause(self._current_owner())
                rows = conn.execute(
                    f"SELECT id,tier,recall_count,half_life_days,last_active_at,updated_at,created_at,ttl_expire "
                    f"FROM events WHERE id IN ({placeholders}) AND blacklist=0 AND {scope_clause}",
                    [*unique, *scope_params],
                ).fetchall()
                for row in rows:
                    tier = str(row["tier"] or "")
                    if tier == "dormant" or via not in CURVE_COUNTED_VIAS:
                        continue
                    if tier in ("short", "pending") and row["ttl_expire"] is None:
                        continue
                    if current - self._row_last_active(row) < CURVE_MIN_GAP:
                        continue
                    count = int(row["recall_count"] or 0) + 1
                    half_life = bounded_half_life(
                        strengthened_half_life(row["half_life_days"], row["recall_count"]),
                        tier,
                    )
                    expire = (None if tier in _PERMANENT_TIERS
                              else terminal_ttl(current, half_life).isoformat())
                    conn.execute(
                        "UPDATE events SET recall_count=?, half_life_days=?, last_active_at=?, "
                        "ttl_expire=?, curve_ttl=?, expired_at=NULL WHERE id=?",
                        (count, half_life, current.isoformat(), expire, expire or "",
                         int(row["id"])),
                    )
                    reinforced += 1
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

        def _group_scope_clause(self, column: str = "owner_key") -> tuple[str, list[str]]:
            """Owner-less reads: only the group owners this scene was granted, nothing wider.

            A private/public reader names no ``group:`` owner and gets ``[]`` here, so it can
            neither read a group row nor have a group row read on its behalf. A managed group turn
            without a grant (switched off, unknown domain) reads no row at all — not the private
            rows it could otherwise reach by an empty owner.
            """
            granted = self._group_read_owners()
            owners = [str(owner) for owner in (granted or ()) if owner]
            if not owners:
                return ("0", []) if self._group_scene_active() else (exclude_group_rows(column), [])
            placeholders = ",".join("?" for _ in owners)
            return (f"{column} IN ({placeholders})", owners)

        def group_write_clause(self, column: str = "owner_key") -> tuple[str, list[str]]:
            """Rows this context may change the body of: the one write owner, never the read set.

            A thread that may *read* its parent group must not push its own raw text into the
            shared parent row, so this clause names ``_group_write_owner_key()`` only. A group
            turn with no grant writes nothing; a background maintenance pass may write only the
            groups the current config still enables.
            """
            owner = self._group_write_owner_key()
            if owner:
                return (f"{column}=?", [str(owner)])
            if self._group_scene_active():
                return ("0", [])
            return self.group_maintenance_clause(column)

        def group_maintenance_clause(self, column: str = "owner_key") -> tuple[str, list[str]]:
            """Background rows: a disabled group is neither updated nor deleted."""
            owners = [str(owner) for owner in (self._group_maintenance_owners() or ()) if owner]
            if not owners:
                return (exclude_group_rows(column), [])
            placeholders = ",".join("?" for _ in owners)
            return (f"({exclude_group_rows(column)} OR {column} IN ({placeholders}))", owners)

        def read_owner_keys_for(self, owner_key: Optional[str] = None) -> list[str]:
            """Readable owners for this query: the bound private owner plus its legacy accounts.

            The union only widens the *reader*: every row keeps its own owner and source facts.
            """
            owner = str(self._current_owner() if owner_key is None else owner_key or "")
            if not owner:
                return []
            if owner != str(self._current_owner() or ""):
                return [owner]
            history = (str(key) for key in self._current_read_owners())
            return sorted({owner, *(key for key in history if key)})

        def read_scope_clause(self, owner_key: Optional[str] = None) -> tuple[str, list[str]]:
            keys = self.read_owner_keys_for(owner_key)
            allowed_group = [str(owner) for owner in (self._group_read_owners() or ()) if owner]
            if self._group_scene_active():
                # This turn is a managed group scene. Only this group's granted owners are
                # readable: no explicit private owner, group owner or sibling widening applies.
                if owner_key is None or not str(owner_key):
                    keys = sorted(allowed_group)
                else:
                    keys = sorted(key for key in keys if key in allowed_group)
                if not keys:
                    return ("0", [])
                if len(keys) == 1:
                    return ("owner_key=?", [keys[0]])
                return ("owner_key IN (%s)" % ",".join("?" for _ in keys), keys)
            if owner_key is None:
                # The current scene: the private owner batch plus this scene's group owners.
                keys = sorted({*keys, *allowed_group})
            elif any(is_group_owner(key) for key in keys):
                # A caller-supplied owner/id never widens the group grant of this turn.
                keys = sorted(key for key in keys if key in allowed_group)
                if not keys:
                    return ("0", [])
            if len(keys) <= 1:
                return ("owner_key=?", [keys[0] if keys else ""])
            return ("owner_key IN (%s)" % ",".join("?" for _ in keys), keys)

        def query_shared_views(
            self, now: str, owner_key: Optional[str] = None, *,
            window_start: Optional[str] = None,
            created_after: Optional[str] = None,
            topic: str = "",
            asked_before: Optional[str] = None,
        ) -> List[sqlite3.Row]:
            """Cross-owner rows, projected columns only.

            This is the only query allowed to read another owner's row. It never
            selects ``raw``/``summary``/``vec``/``keywords``/``source_context`` and
            never joins ``event_occurrences``; callers can only render the neutral
            view the host wrote after semantic projection.
            """
            assert self._sqlite.is_open
            if self._group_scene_active():
                return []
            owner = str(self._current_owner() if owner_key is None else owner_key)
            own_keys = self.read_owner_keys_for(owner) if owner else [owner]
            if not own_keys:
                own_keys = [""]
            own_placeholders = ",".join("?" for _ in own_keys)
            clauses = [
                "blacklist=0", "tier<>'dormant'", "expired_at IS NULL",
                "shared_view<>''", f"owner_key NOT IN ({own_placeholders})", _VISIBLE_TTL, exclude_group_rows(),
            ]
            params: List[Any] = [*own_keys, now]
            if asked_before is not None:
                clauses.append("(asked=0 OR asked_at='' OR asked_at<?)")
                params.append(asked_before)
            if window_start is not None:
                clauses.append(
                    "(tier IN ('important', 'long', 'distilled') OR "
                    "CASE WHEN recall_count>0 THEN COALESCE(NULLIF(last_active_at,''),"
                    "NULLIF(updated_at,''),created_at) ELSE created_at END>=?)")
                params.append(window_start)
            if created_after is not None:
                clauses.append("created_at>=?")
                params.append(created_after)
            if topic:
                clauses.append("topic=?")
                params.append(topic)
            return self._sqlite.fetchall(
                "SELECT id, shared_view, created_at FROM events WHERE "
                + " AND ".join(clauses) + " ORDER BY id DESC LIMIT 500", params
            )

        def find_duplicate_candidates(
            self, session_id: str, cutoff: str, unscoped: bool = False,
            owner_key: Optional[str] = None,
            experience_subject: Optional[str] = None,
            fact_subject: Optional[str] = None,
            involved_owner_keys: Optional[str] = None,
            read_scope: Optional[str] = None, fingerprint: str = '', body: str = '',
        ) -> List[sqlite3.Row]:
            assert self._sqlite.is_open
            clauses = [
                "blacklist=0", "tier<>'dormant'",
                "((session_id=? AND created_at>=?) OR (tier='short' AND expired_at IS NOT NULL)"
                " OR (?>'' AND (original_fingerprint=? OR (original_fingerprint='' "
                "AND substr(raw,instr(raw,'] ')+2)=?))))",
            ]
            params: List[Any] = [session_id, cutoff, fingerprint, fingerprint, body]
            if not unscoped:
                clauses.append("owner_key=?")
                params.append(self._current_owner() if owner_key is None else owner_key)
            # Duplicate detection reads only owners this turn was granted; a group row of another
            # group is never a merge candidate here.
            duplicate_group_clause, duplicate_group_params = self._group_scope_clause()
            clauses.append(duplicate_group_clause)
            params.extend(duplicate_group_params)
            for column, value in (
                    ("experience_subject", experience_subject),
                    ("fact_subject", fact_subject),
                    ("involved_owner_keys", involved_owner_keys),
                    ("read_scope", read_scope)):
                if value is not None:
                    clauses.append(f"{column}=?")
                    params.append(value)
            return self._sqlite.fetchall(
                "SELECT id, keywords, importance, expired_at, source_context FROM events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY (original_fingerprint=? AND original_fingerprint<>'') DESC, "
                "(expired_at IS NOT NULL) DESC, id DESC LIMIT 20",
                [*params, fingerprint],
            )

        def query_public_projections(self, now: str, unscoped: bool = False,
                                     owner_key: Optional[str] = None,
                                     cross_owner_self: bool = False
                                     ) -> List[sqlite3.Row]:
            """公开面唯一可读的行：**只有**已被可信层审阅出公开投影的事件。

            ``public_excerpt`` 为空（含全部旧行）⇒ 公开检索看不见这一行。查询**只**取投影与
            出处元数据，不取 ``raw``／``summary``／``vec`` —— 公开结果不得从私人原文衍生。
            """
            assert self._sqlite.is_open
            clauses = [
                "blacklist=0", "tier<>'dormant'", "public_excerpt<>''", "expired_at IS NULL",
                _VISIBLE_TTL, exclude_group_rows(),
            ]
            params: List[Any] = [now]
            if not unscoped:
                owner = self._current_owner() if owner_key is None else owner_key
                if cross_owner_self:
                    clauses.append(
                        "(owner_key=? OR (json_valid(source_context) "
                        "AND json_extract(source_context,'$.kind')='self'))")
                else:
                    clauses.append("owner_key=?")
                params.append(owner)
            return self._sqlite.fetchall(
                "SELECT id, public_excerpt, source_context, created_at FROM events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY id DESC LIMIT 500",
                params,
            )

        def query_private_peer_events(self, now: str, peer: str,
                                      owner_key: Optional[str] = None, *,
                                      platform: Optional[str] = None) -> List[sqlite3.Row]:
            """Read the owner's private inbox for a peer on a supported source route."""
            assert self._sqlite.is_open
            route = private_peer_route(platform)
            owner = self._current_owner() if owner_key is None else owner_key
            if route is None or not owner or not peer:
                return []

            def _query(conn: sqlite3.Connection) -> List[sqlite3.Row]:
                result: list[sqlite3.Row] = []
                with closing(conn.execute(
                    "SELECT id, raw AS public_excerpt, source_context, created_at FROM events"
                    " WHERE owner_key=? AND blacklist=0 AND expired_at IS NULL"
                    f" AND tier<>'dormant' AND {_VISIBLE_TTL}"
                    " AND json_valid(source_context)"
                    " AND json_extract(source_context,'$.visibility')='private'"
                    " AND json_extract(source_context,'$.platform')=?"
                    " AND json_extract(source_context,'$.kind')='external_user'"
                    " AND json_extract(source_context,'$.author_id')=?"
                    " ORDER BY id DESC",
                    (owner, now, route.platform, peer),
                )) as rows:
                    for row in rows:
                        source = parse_source(row["source_context"])
                        if source is not None and route.matches_ref(source["ref"], owner_key=owner, peer=peer):
                            result.append(row)
                            if len(result) == 500:
                                break
                return result

            return self._sqlite.read(_query)

        def set_public_excerpt(self, event_id: int, excerpt: str) -> bool:
            """**host-only**：给某条事件设置／清空公开投影（`excerpt` 空串＝撤销公开视图）。

            模型面到不了这里：它既不在 `get_tool_schemas()` 里，也不在 `handle_tool_call()` 的
            分派表里。设置本身＝可信层显式审阅这一事实，不是模型自标 ``public=true``；写入只动
            ``public_excerpt`` 一列，原文、摘要与出处一字不改。返回是否命中一行。
            """
            assert self._sqlite.is_open
            with self._sqlite.transaction() as conn:
                row = conn.execute(
                    f"SELECT 1 FROM events WHERE id=? AND blacklist=0 AND {exclude_group_rows()}",
                    (int(event_id),)).fetchone()
                if row is None:
                    return False
                conn.execute(
                    "UPDATE events SET public_excerpt=?, updated_at=? WHERE id=? AND blacklist=0",
                    (str(excerpt or ""), datetime.now().astimezone().isoformat(), int(event_id)),
                )
            return True

        def set_shared_view(self, event_id: int, text: str, *, touch: bool = True) -> bool:
            """Host-only neutral view for another user's shared experience.

            It updates only ``shared_view``; raw/summary/source/vector stay untouched.
            The caller is the trusted host, never a model tool; the text must be
            neutral and must not be derived from the other user's private content.
            ``touch=False`` keeps an internal projection from changing recency.
            """
            assert self._sqlite.is_open
            with self._sqlite.transaction() as conn:
                row = conn.execute(
                    f"SELECT 1 FROM events WHERE id=? AND blacklist=0 AND {exclude_group_rows()}",
                    (int(event_id),)).fetchone()
                if row is None:
                    return False
                if touch:
                    conn.execute(
                        "UPDATE events SET shared_view=?, updated_at=? WHERE id=? AND blacklist=0",
                        (str(text or ""), datetime.now().astimezone().isoformat(), int(event_id)),
                    )
                else:
                    conn.execute(
                        "UPDATE events SET shared_view=? WHERE id=? AND blacklist=0",
                        (str(text or ""), int(event_id)),
                    )
            return True

        def event_source_context(self, event_id: int) -> Optional[dict]:
            visible, visible_params = self._group_scope_clause()
            row = self._sqlite.fetchone(
                f"SELECT source_context FROM events WHERE id=? AND {visible}",
                (int(event_id), *visible_params))
            return parse_source(row["source_context"]) if row else None

        def query_tier_window(
            self, now: str, window_start: str, topic: str = "", created_after: Optional[str] = None,
            unscoped: bool = False, owner_key: Optional[str] = None,
            experience_subject: Optional[str] = None,
            fact_subject: Optional[str] = None, read_context: bool = False,
            writable: bool = False,
        ) -> List[sqlite3.Row]:
            assert self._sqlite.is_open
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
                scope_clause, scope_params = self.read_scope_clause(owner_key)
                clauses.append(scope_clause)
                params.extend(scope_params)
            elif not unscoped:
                clauses.append("owner_key=?")
                params.append(self._current_owner() if owner_key is None else owner_key)
            else:
                # A whole-table window still only reaches groups the config still enables.
                group_clause, group_params = self.group_maintenance_clause()
                clauses.append(group_clause)
                params.extend(group_params)
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
            return self._sqlite.fetchall(
                "SELECT * FROM events WHERE " + " AND ".join(clauses) + " ORDER BY id DESC LIMIT 500", params
            )

        def query_events(
            self, now: str, topic: str = "", created_after: Optional[str] = None,
            include_expiring: bool = False, unscoped: bool = False,
            owner_key: Optional[str] = None,
            experience_subject: Optional[str] = None,
            fact_subject: Optional[str] = None, read_context: bool = False,
            writable: bool = False, dormant_owner: Optional[str] = None,
        ) -> List[sqlite3.Row]:
            assert self._sqlite.is_open
            clauses = ["blacklist=0", _VISIBLE_TTL]
            params: List[Any] = [now]
            if dormant_owner is None:
                clauses.append("tier<>'dormant'")
            else:
                dormant_keys = self.read_owner_keys_for(dormant_owner)
                if dormant_keys:
                    d_placeholders = ",".join("?" for _ in dormant_keys)
                    clauses.append(f"(tier<>'dormant' OR owner_key IN ({d_placeholders}))")
                    params.extend(dormant_keys)
                else:
                    clauses.append("tier<>'dormant'")
            if not include_expiring:
                clauses.append("expired_at IS NULL")
            if read_context or writable:
                scope_clause, scope_params = self.read_scope_clause(owner_key)
                clauses.append(scope_clause)
                params.extend(scope_params)
            elif not unscoped:
                clauses.append("owner_key=?")
                params.append(self._current_owner() if owner_key is None else owner_key)
            else:
                # A whole-table window still only reaches groups the config still enables.
                group_clause, group_params = self.group_maintenance_clause()
                clauses.append(group_clause)
                params.extend(group_params)
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
            return self._sqlite.fetchall(
                "SELECT * FROM events WHERE " + " AND ".join(clauses) + " ORDER BY id DESC LIMIT 500", params
            )

        def _commitment_continuation_owner(
            self, owner: str, commitment_id: Optional[int], values: tuple, *,
            source_context: str, read_scope: str,
        ) -> bool:
            """True only for the completion record of a promise this scene may still continue.

            An ``inherited`` thread keeps the exact owner it wrote under while its audience was
            ``unknown``, so that promise's completion event stays there instead of moving to the
            shared owner. Every condition is read from the stored row and this call's own
            material: the row must be that very promise, already fulfilled, under the requested
            owner, the current grant must still derive that owner for resolution, the event must
            be its completion wording, and its link and scope must name the same promise and
            owner. A plain new event names no commitment and never passes; nothing here widens a
            write to another owner, another scene or the shared scope.
            """
            if isinstance(commitment_id, bool) or not isinstance(commitment_id, int):
                return False
            if not owner or owner not in self._group_resolution_owners():
                return False
            row = self._sqlite.fetchone(
                "SELECT owner_key, text, status FROM commitments WHERE id=?",
                (int(commitment_id),))
            if row is None:
                return False
            if str(row["owner_key"] or "") != owner or str(row["status"] or "") != "fulfilled":
                return False
            promised = str(row["text"] or "").strip()
            raw = str(values[1] or "") if isinstance(values, tuple) and len(values) > 1 else ""
            if not promised or promised not in raw:
                return False
            origin = parse_source(source_context) or {}
            if str(origin.get("commitment_id") or "") != str(int(commitment_id)):
                return False
            try:
                scope = json.loads(read_scope) if read_scope else None
            except (TypeError, ValueError):
                scope = None
            return isinstance(scope, dict) and str(scope.get("owner") or "") == owner

        def insert_event_row(
            self, fingerprint: str, values: tuple, appraisal: Optional[dict] = None, owner_key: str = "",
            summary: str = "", summary_source: str = SUMMARY_SOURCE_ORIGINAL,
            source_snapshot: str = "", source_snapshot_sha256: str = "",
            source_context: str = "", experience_subject: str = "",
            fact_subject: str = "", involved_owner_keys: str = "", read_scope: str = "",
            half_life_days: float = 14.0, last_active_at: str = "", curve_ttl: str = "",
            fulfillment_commitment_id: Optional[int] = None,
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
            `fulfillment_commitment_id` 只由内部兑现入口传：一条已履行约定的完成记录可以留在
            该约定原来的 exact owner 上（见 `_commitment_continuation_owner`），普通新材料不传
            ⇒ 仍只许本场景的 `write_owner`。
            """
            assert self._sqlite.is_open
            owner = str(owner_key or "")
            allowed_write = self._group_write_owner_key()
            continuation = False
            if self._group_scene_active():
                # This turn *is* a managed group scene: only its one write owner may take a row.
                # A switched-off group writes nothing, and an explicit private or other-group
                # owner is not an authorization this turn ever gains. The one exception is the
                # completion record of a promise this scene may still continue.
                if not allowed_write:
                    return None
                if owner != str(allowed_write):
                    if not self._commitment_continuation_owner(
                            owner, fulfillment_commitment_id, values,
                            source_context=source_context, read_scope=read_scope):
                        return None
                    continuation = True
            elif is_group_owner(owner) and owner != str(allowed_write):
                # Only this turn's own group may take a new row; a caller-supplied owner cannot.
                return None
            created_at = str(values[7] or "")
            clamped_half_life = bounded_half_life(float(half_life_days), "short")
            with self._sqlite.transaction() as conn:
                if conn.execute("SELECT 1 FROM forgotten WHERE fingerprint=?", (fingerprint,)).fetchone():
                    return None
                cursor = conn.execute(
                    """INSERT INTO events
                       (ts, raw, summary, summary_source, keywords, importance, confidence, topic, ttl_expire,
                        created_at, vec, session_id, a_conflict, affect, owner_key,
                        source_snapshot, source_snapshot_sha256, source_context,
                        experience_subject, fact_subject, involved_owner_keys, read_scope,
                        last_active_at, half_life_days, curve_ttl, original_fingerprint)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    tuple(values[:2]) + (str(summary or ""), str(summary_source or ""))
                    + tuple(values[2:]) + tuple(
                        int((appraisal or {}).get(name, 0))
                        for name in ("a_conflict", "affect")
                    ) + (str(owner_key or ""), str(source_snapshot or ""),
                         str(source_snapshot_sha256 or ""), str(source_context or ""),
                         str(experience_subject or ""), str(fact_subject or ""),
                         str(involved_owner_keys or ""), str(read_scope or ""),
                         str(last_active_at or created_at), clamped_half_life,
                         str(curve_ttl or ""), fingerprint),
                )
                assert cursor.lastrowid is not None
                event_id = int(cursor.lastrowid)
                if continuation:
                    # The completion record keeps its first utterance from this same authorized
                    # insert. Later appends still go through the ordinary write-owner gate.
                    conn.execute(
                        "INSERT INTO event_occurrences"
                        "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                        " VALUES(?,?,?,?,?)",
                        (event_id, str(values[1] or ""), created_at,
                         str(source_snapshot or ""), str(source_snapshot_sha256 or "")))
            return event_id

        def event_ids_existing(self, ids: List[int],
                               owner_key: Optional[str] = None,
                               experience_subject: Optional[str] = None,
                               fact_subject: Optional[str] = None,
                               writable: bool = False) -> List[int]:
            assert self._sqlite.is_open
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            sql = f"SELECT id FROM events WHERE id IN ({placeholders})"
            params: List[Any] = list(ids)
            if writable:
                clause, scope_params = self.read_scope_clause(owner_key)
                sql += " AND " + clause
                params.extend(scope_params)
            elif owner_key is not None:
                granted = [str(item) for item in (self._group_read_owners() or ())]
                if self._group_scene_active() and str(owner_key) not in granted:
                    # In a managed group scene an explicit owner is not a read grant at all.
                    return []
                if is_group_owner(owner_key) and str(owner_key) not in granted:
                    # A group owner named by the caller is not a read grant for this turn.
                    return []
                sql += " AND owner_key=?"
                params.append(owner_key)
            else:
                clause, scope_params = self._group_scope_clause()
                sql += " AND " + clause
                params.extend(scope_params)
            if experience_subject is not None:
                sql += " AND experience_subject=?"
                params.append(experience_subject)
            if fact_subject is not None:
                sql += " AND fact_subject=?"
                params.append(fact_subject)
            rows = self._sqlite.fetchall(sql, params)
            return [int(row["id"]) for row in rows]

        def get_event_row_by_id(self, event_id: int) -> Optional[sqlite3.Row]:
            """按真实事件 id 纯读取一行（受控迁移的目标精确回读）。

            与 ``reinforce_event`` 不同，它**不改 importance、不写 updated_at、不标记 asked**，
            只回 ``raw``/``topic`` 等原值；行不存在返回 ``None``。宿主用它验证「目标确实按 id
            写进去了」，不接受工具返回里的 ``stored: true`` 当证据。
            绑定 owner 时，别人的行 fail-closed（不先取原文再遮罩）；无 owner 的维护上下文保持原行为。
            """
            assert self._sqlite.is_open
            if isinstance(event_id, bool) or not isinstance(event_id, int):
                return None
            sql = ("SELECT id, raw, topic, importance, ts, source_context, "
                   "experience_subject, fact_subject FROM events WHERE id=?")
            params: List[Any] = [int(event_id)]
            if str(self._current_owner() or ""):
                clause, scope_params = self.read_scope_clause(self._current_owner())
                sql += " AND " + clause
                params.extend(scope_params)
            else:
                # An owner-less context reads a group row only through this turn's own grant.
                clause, scope_params = self._group_scope_clause()
                sql += " AND " + clause
                params.extend(scope_params)
            try:
                return self._sqlite.fetchone(sql, params)
            except Exception:
                return None

        def record_event_occurrence(self, event_id: int, raw: str, observed_at: str,
                                    source_snapshot: str = "",
                                    source_snapshot_sha256: str = "",
                                    preserve_previous: bool = False) -> bool:
            """Record a separate utterance under one searchable event, never infer source proof."""
            assert self._sqlite.is_open
            if not raw:
                return False
            with self._sqlite.transaction() as conn:
                writable, writable_params = self.group_write_clause()
                previous = conn.execute(
                    "SELECT raw,created_at,source_snapshot,source_snapshot_sha256 "
                    f"FROM events WHERE id=? AND blacklist=0 AND {writable}",
                    (int(event_id), *writable_params)).fetchone()
                if previous is None:
                    return False
                if (preserve_previous
                        and not conn.execute(
                            "SELECT 1 FROM event_occurrences WHERE event_id=? LIMIT 1",
                            (int(event_id),)).fetchone()):
                    conn.execute(
                        "INSERT INTO event_occurrences"
                        "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                        " VALUES(?,?,?,?,?)",
                        (int(event_id), previous["raw"], previous["created_at"],
                         previous["source_snapshot"], previous["source_snapshot_sha256"]))
                conn.execute(
                    "INSERT INTO event_occurrences"
                    "(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256)"
                    " VALUES(?,?,?,?,?)",
                    (int(event_id), str(raw), str(observed_at),
                     str(source_snapshot or ""), str(source_snapshot_sha256 or "")))
            return True

        def query_event_occurrences(self, event_ids: List[int]) -> dict:
            """Recall-only read of the existing occurrence side records, oldest first.

            The event row keeps the first utterance; later utterances live in
            ``event_occurrences``. Retrieval must see the whole timeline instead of
            only the latest summary. This method never writes or backfills.
            """
            assert self._sqlite.is_open
            ids = sorted({int(item) for item in event_ids if not isinstance(item, bool)})
            if not ids:
                return {}
            placeholders = ",".join("?" for _ in ids)
            rows = self._sqlite.fetchall(
                "SELECT event_id, raw, observed_at FROM event_occurrences "
                f"WHERE event_id IN ({placeholders}) ORDER BY observed_at, id",
                ids,
            )
            grouped: dict = {}
            for row in rows:
                grouped.setdefault(int(row["event_id"]), []).append(
                    {"time": str(row["observed_at"] or ""), "text": str(row["raw"] or "")}
                )
            return grouped

        def get_grounded_occurrence(self, event_id: int) -> GroundedOccurrence | None:
            """Return earliest captured, source-backed utterance; merged text is never evidence."""
            assert self._sqlite.is_open
            sql = ("SELECT raw, created_at, source_snapshot, source_snapshot_sha256, owner_key, "
                   "source_context FROM events WHERE id=? AND blacklist=0")
            params: List[Any] = [int(event_id)]
            if str(self._current_owner() or ""):
                clause, scope_params = self.read_scope_clause(self._current_owner())
                sql += " AND " + clause
                params.extend(scope_params)
            else:
                clause, scope_params = self._group_scope_clause()
                sql += " AND " + clause
                params.extend(scope_params)

            def _read(conn: sqlite3.Connection) -> tuple[Optional[sqlite3.Row], List[sqlite3.Row]]:
                row = conn.execute(sql, params).fetchone()
                if row is None:
                    return None, []
                occurrences = conn.execute(
                    "SELECT raw,observed_at,source_snapshot,source_snapshot_sha256 "
                    "FROM event_occurrences WHERE event_id=? AND source_snapshot<>'' "
                    "ORDER BY id", (int(event_id),)).fetchall()
                return row, occurrences

            row, occurrences = self._sqlite.read(_read)
            if row is None:
                return None
            original = str(row["raw"] or "")
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
            assert self._sqlite.is_open
            body = str(fragment or "").strip()
            if not body:
                return 0
            writable, writable_params = self.group_write_clause()
            with self._sqlite.transaction() as conn:
                row = conn.execute(
                    f"SELECT raw FROM events WHERE id=? AND {writable}",
                    (int(event_id), *writable_params)).fetchone()
                if row is None:
                    return 0
                merged = f"{str(row['raw'] or '')}\n[补充 {now}] {body}"
                cursor = conn.execute(
                    "UPDATE events SET raw=?, summary=?, updated_at=? WHERE id=?",
                    (merged, body[:150], now, int(event_id)),
                )
            return max(0, int(cursor.rowcount))

        def reinforce_event(self, event_id: int, importance: float, now: str) -> Optional[sqlite3.Row]:
            assert self._sqlite.is_open
            visible, visible_params = self._group_scope_clause()
            with self._sqlite.transaction() as conn:
                row = conn.execute(
                    f"SELECT raw, summary, topic, ts, expired_at FROM events WHERE id=? AND {visible}",
                    (event_id, *visible_params)
                ).fetchone()
                if not row:
                    return None
                conn.execute(
                    "UPDATE events SET importance=?, updated_at=? WHERE id=?", (importance, now, event_id)
                )
            return row

        def reinforce_expiring(self, event_id: int, now: str, ttl_expire: str) -> int:
            assert self._sqlite.is_open
            visible, visible_params = self._group_scope_clause()
            with self._sqlite.transaction() as conn:
                cursor = conn.execute(
                    """UPDATE events
                       SET expired_at=NULL, ttl_expire=?, curve_ttl=?, asked=0, updated_at=?
                       WHERE id=? AND tier='short' AND blacklist=0 AND expired_at IS NOT NULL"""
                    + f" AND {visible}",
                    (ttl_expire, ttl_expire, now, event_id, *visible_params),
                )
            return max(0, int(cursor.rowcount))
