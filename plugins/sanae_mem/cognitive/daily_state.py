from __future__ import annotations

import logging
import re
import sqlite3
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..storage.sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)

from .daily_inputs import (
    _is_question_clause, _is_second_person_clause,
    _looks_secret, _scrub_face_tags, _split_clauses,
)
from .daily_rules import (
    KIND_CONSTRAINT, KIND_FACT, KIND_THREAD,
    MAX_ACTIVE_PER_KIND, MAX_BASIS_CHARS, MAX_BLOCK_TEXT_CHARS,
    MAX_QUOTE_CHARS, MAX_TEXT_CHARS,
    SOURCE_INFERENCE, SOURCE_STATEMENT, STALE_TTL_DAYS, STALE_TTL_MAX_DAYS, THREAD_TTL_DAYS,
    _CHAT_META, _Decision,
    _ROUTINE, _SUSTAINED_HORIZON,
    _TRANSIENT_STATE,
    _norm_text, _now_iso,
)
from ..infra.textutil import _tokens


def _row_source(row: Any) -> str:
    try:
        value = str(row["source_kind"] or "").strip()
    except (IndexError, KeyError):
        return SOURCE_STATEMENT
    return value or SOURCE_STATEMENT


def _sustained_horizon(text: str) -> bool:
    return _SUSTAINED_HORIZON.search(str(text or "")) is not None


def _default_ttl(today: str, sustained: bool = False) -> Optional[str]:
    days = STALE_TTL_MAX_DAYS if sustained else STALE_TTL_DAYS
    try:
        return (date.fromisoformat(today) + timedelta(days=days)).isoformat()
    except ValueError:
        return None


def _shares_content(left: str, right: str, min_len: int = 3) -> bool:
    a, b = _norm_text(left), _norm_text(right)
    if len(a) < min_len or len(b) < min_len:
        return False
    return any(a[index:index + min_len] in b for index in range(len(a) - min_len + 1))


def _is_transient_state(text: str) -> bool:
    return _TRANSIENT_STATE.search(str(text or "")) is not None


_THREAD_COMMON_WORDS = (
    "好奇", "随口", "提到", "提起", "那个", "什么", "怎么", "到底", "为什么", "一下",
    "了解", "琢磨", "查查", "看看", "自己", "今天", "最近", "觉得", "打算",
    "回头", "细聊", "说的",
)


def _thread_core_text(text: str) -> str:
    value = str(text or "")
    for word in _THREAD_COMMON_WORDS:
        value = value.replace(word, " ")
    return value


def _thread_core_tokens(text: str) -> set[str]:
    value = _thread_core_text(text)
    tokens = set(_tokens(value))
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        tokens.update(run[index:index + 2] for index in range(len(run) - 1))
        tokens.update(run[index:index + 3] for index in range(len(run) - 2))
    return {token for token in tokens if len(token) >= 2}


def _thread_same_topic(thread_text: str, target_text: str) -> bool:
    left_text, right_text = _thread_core_text(thread_text), _thread_core_text(target_text)
    shared = _thread_core_tokens(thread_text) & _thread_core_tokens(target_text)
    if not shared:
        return False
    if any(token.isascii() for token in shared):
        return True
    if any(len(token) >= 3 and token in right_text for token in shared):
        return True
    return sum(1 for token in shared if len(token) == 2) >= 2


_THREAD_ACTIVITY_HEAD_RE = re.compile(r"定时(?:自主)?活动完成：[^。！\n]+[。！]?")
_THREAD_SEGMENT_RE = re.compile(r"[。！？!?\n]")
_THREAD_UNEXPLORED_RE = re.compile(
    r"没查|还没|未查|没看|还没看|没去查|没搜|没空|没来得及|没弄清|没搞懂|没弄明白|"
    r"不知道|不清楚|还不懂|没细看|先不查|先不看|以后再|改天再|下次再|晚点再|有空再|"
    r"回头再|等有空|改天看|以后有空|打算去查|想找时间查|还在好奇|还是好奇"
)
_THREAD_EXPLORED_RE = re.compile(
    r"B站观看|收到该视频的抽样画面分析|阅读了该视频的字幕|查了|查到|查清|搜了|搜到|"
    r"翻了|翻到|看了|看到|读了|读到|找到|了解了|去了解|研究了一下|琢磨明白|"
    r"弄明白|搞明白|搞懂|弄清|看懂|原来|原理是|其实是|靠的是|核心是|机制是|结构是|是因为"
)


_THREAD_RECORD_SPLIT_RE = re.compile(r"\n(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*[|｜]|\s*-\s)")
_VIDEO_WATCH_RECORD_RE = re.compile(r"B站观看|收到该视频的抽样画面分析|阅读了该视频的字幕")
_BOOK_TITLE_RE = re.compile(r"《[^》]{2,80}》")


def _thread_exploration_completed(thread_text: str, activity_text: str) -> bool:
    raw = str(activity_text or "").strip()
    if not raw:
        return False
    thread_titles = set(_BOOK_TITLE_RE.findall(str(thread_text or "")))
    records = _THREAD_RECORD_SPLIT_RE.split(raw)
    for record in records:
        body = _THREAD_ACTIVITY_HEAD_RE.sub("", record)
        if _THREAD_UNEXPLORED_RE.search(body):
            continue
        if _VIDEO_WATCH_RECORD_RE.search(record):
            rec_titles = set(_BOOK_TITLE_RE.findall(record))
            if thread_titles and rec_titles and (thread_titles & rec_titles):
                continue
        matched = [
            piece for segment in _THREAD_SEGMENT_RE.split(body)
            if (piece := segment.strip()) and _thread_same_topic(thread_text, piece)
        ]
        if matched and any(_THREAD_EXPLORED_RE.search(piece) for piece in matched):
            return True
    return False


def _resolve_daily_expiry(decision: "_Decision", row: Any, today: str) -> Optional[str]:
    existing = ""
    stored = ""
    if row is not None:
        try:
            existing = str(row["expires_on"] or "")
            stored = str(row["text"] or "")
        except (IndexError, KeyError):
            existing, stored = "", ""
    if str(decision.kind) == KIND_THREAD:
        if existing:
            return existing
        if decision.expires_on:
            return decision.expires_on
        try:
            return (date.fromisoformat(today) + timedelta(days=THREAD_TTL_DAYS)).isoformat()
        except ValueError:
            return None
    if decision.expires_on:
        return decision.expires_on
    if str(decision.kind) != KIND_CONSTRAINT:
        return existing or None
    quote = str(decision.quote or "")
    text = str(decision.text or "")
    if _is_transient_state(text or quote):
        return today
    if existing and existing > today and not _shares_content(quote, stored):
        return existing
    return _default_ttl(today, _sustained_horizon(quote or text or stored))


_QUOTED_ROW = re.compile(r'^[“"「『].*[”"」』]$', re.S)


def _view_noise(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if _QUOTED_ROW.match(value):
        return True
    clauses = _split_clauses(value) or [value]
    return (
        any(_CHAT_META.fullmatch(clause) for clause in clauses)
        or any(_is_question_clause(clause) for clause in clauses)
        or any(_is_second_person_clause(clause) for clause in clauses)
    )


def _fact_rank(row: Any) -> Tuple[int, int]:
    text = str(row["text"] or "")
    return (1 if _ROUTINE.search(text) else 0, -int(row["entry_id"]))


class _DailyWriteRejected(RuntimeError):
    pass


class DailyState:
    def __init__(self, sqlite: SqliteFacade, *, current_author_kind: Callable[[], str]) -> None:
        self._sqlite = sqlite
        self._current_author_kind = current_author_kind

    def apply(
        self, scope: Tuple[str, str], decisions: List[_Decision], today: str, session_id: str,
    ) -> bool:
        owner, profile = scope
        now = _now_iso()
        written = False
        try:
            self.expire_stale_daily_facts(
                owner_key=owner, profile_key=profile, today=today, now=now,
            )
            for decision in decisions:
                if self._apply_one(owner, profile, decision, today, now, session_id):
                    written = True
        except (_DailyWriteRejected, sqlite3.OperationalError) as exc:
            logger.warning("sanae_mem: daily_context write failed (%s)", type(exc).__name__)
            return False
        return written

    def _apply_one(
        self, owner: str, profile: str, decision: _Decision, today: str, now: str, session_id: str,
    ) -> bool:
        kind, slot, action = decision.kind, decision.slot, decision.action
        if kind == KIND_THREAD and (action == "clear" or not decision.text):
            if action != "clear":
                return False
        if action == "clear":
            row = self.find_active_daily_entry(
                owner_key=owner, profile_key=profile, kind=kind, slot=slot,
            )
            if row is None:
                return False
            for _attempt in (0, 1):
                if self.cas_supersede_daily_entry(
                    owner_key=owner, profile_key=profile, entry_id=int(row["entry_id"]),
                    version=int(row["version"]), status="closed", now=now,
                ):
                    return True
                row = self.find_active_daily_entry(
                    owner_key=owner, profile_key=profile, kind=kind, slot=slot,
                )
                if row is None:
                    return False
            raise _DailyWriteRejected("daily_context clear conflict")

        text = _scrub_face_tags(str(decision.text or "")).strip()
        quote = _scrub_face_tags(str(decision.quote or ""))[:MAX_QUOTE_CHARS]
        if kind == KIND_THREAD and _ROUTINE.search(text):
            return False
        if _looks_secret(text) or _looks_secret(quote):
            logger.info("sanae_mem: daily_context rejected (secret)")
            return False
        row = self.find_active_daily_entry(
            owner_key=owner, profile_key=profile, kind=kind, slot=slot,
        )
        if kind == KIND_THREAD and row is None:
            for candidate in self.list_active_daily_entries(
                    owner_key=owner, profile_key=profile, today=today):
                if str(candidate["kind"]) != KIND_THREAD:
                    continue
                candidate_text = str(candidate["text"] or "")
                candidate_quote = str(candidate["source_quote"] or "")
                if (_shares_content(text, candidate_text)
                        or _shares_content(quote, candidate_quote)
                        or _thread_same_topic(
                            f"{text} {quote}", f"{candidate_text} {candidate_quote}")):
                    row = candidate
                    slot = str(candidate["slot"])
                    break
        expires = _resolve_daily_expiry(decision, row, today)
        opened_on = str(row["opened_on"] or today) if (kind == KIND_THREAD and row is not None) else today
        if row is not None:
            if not text or len(text) > MAX_TEXT_CHARS:
                raise _DailyWriteRejected("daily_context update not storable")
            if (str(row["text"] or "") == text
                    and str(row["expires_on"] or "") == str(expires or "")
                    and _row_source(row) == decision.source_kind):
                return False
            if not self.cas_update_daily_entry(
                owner_key=owner, profile_key=profile, entry_id=int(row["entry_id"]),
                version=int(row["version"]), text=text, opened_on=opened_on,
                expires_on=expires, source_quote=quote,
                source_turn=session_id, now=now, source_kind=decision.source_kind,
            ):
                raise _DailyWriteRejected("daily_context update conflict")
            return True

        if not text or len(text) > MAX_TEXT_CHARS:
            return False
        if kind != KIND_FACT:
            stats = self.daily_active_stats(owner_key=owner, profile_key=profile)
            if stats["counts"].get(kind, 0) >= MAX_ACTIVE_PER_KIND.get(kind, 0):
                logger.info("sanae_mem: daily_context rejected (%s quota)", kind)
                return False
        author_kind = self._current_author_kind()
        try:
            entry_id = self.open_daily_entry(
                owner_key=owner, profile_key=profile, kind=kind, slot=slot, text=text,
                opened_on=today, now=now, expires_on=expires,
                source_quote=quote, source_turn=session_id,
                source_kind=decision.source_kind,
                author_kind=author_kind,
            )
            return entry_id is not None
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorcode != sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                raise
            row = self.find_active_daily_entry(
                owner_key=owner, profile_key=profile, kind=kind, slot=slot,
            )
            if row is None:
                raise
            expires = _resolve_daily_expiry(decision, row, today)
            if kind == KIND_THREAD:
                opened_on = str(row["opened_on"] or today)
        if not self.cas_update_daily_entry(
            owner_key=owner, profile_key=profile, entry_id=int(row["entry_id"]),
            version=int(row["version"]), text=text, opened_on=opened_on,
            expires_on=expires, source_quote=quote,
            source_turn=session_id, now=now, source_kind=decision.source_kind,
        ):
            raise _DailyWriteRejected("daily_context update conflict")
        return True

    def render(self, scope: Tuple[str, str], today: str) -> str:
        owner, profile = scope
        try:
            rows = self.list_active_daily_entries(
                owner_key=owner, profile_key=profile, today=today,
            )
        except sqlite3.OperationalError as exc:
            logger.warning("sanae_mem: daily_context read failed (%s)", type(exc).__name__)
            return ""

        def opened_today(row: Any) -> bool:
            opened = str(row["opened_on"] if "opened_on" in row.keys() else today or today)
            return opened == today

        lines: List[str] = []
        used = 0
        for row in rows:
            if str(row["kind"]) != KIND_CONSTRAINT or not opened_today(row):
                continue
            text = str(row["text"] or "").strip()
            if not text or _view_noise(text):
                continue
            lines.append(f"· 眼下：{text}")
        fact_rows = [
            row for row in rows
            if str(row["kind"]) == KIND_FACT and _row_source(row) != SOURCE_INFERENCE
            and opened_today(row)
        ]
        fact_rows.sort(key=_fact_rank)
        for row in fact_rows:
            text = str(row["text"] or "").strip()
            if not text or _view_noise(text):
                continue
            if used + len(text) > MAX_BLOCK_TEXT_CHARS:
                continue
            used += len(text)
            lines.append(f"· {text}")
        for row in rows:
            if str(row["kind"]) != KIND_FACT or _row_source(row) != SOURCE_INFERENCE:
                continue
            if not opened_today(row):
                continue
            text = str(row["text"] or "").strip()
            if not text:
                continue
            basis = str(row["source_quote"] or "").strip()[:MAX_BASIS_CHARS]
            if basis:
                lines.append(f"· {text}（未确认；依据「{basis}」）")
            else:
                lines.append(f"· {text}（未确认）")
        return "\n".join(lines)

    def open_daily_entry(
        self, *, owner_key: str, profile_key: str, kind: str, slot: str, text: str,
        opened_on: str, now: str, expires_on: Optional[str] = None,
        source_quote: str = "", source_turn: str = "", source_kind: str = "statement",
        author_kind: str = "unknown",
    ) -> Optional[int]:
        """插入一条新的 active 行；作用域缺失返回 None；同槽已 active 会抛 IntegrityError。"""
        assert self._sqlite.is_open
        if not (owner_key and profile_key):
            return None
        with self._sqlite.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO daily_context "
                "(owner_key, profile_key, kind, slot, text, opened_on, expires_on, "
                " source_quote, source_turn, source_kind, author_kind, "
                " status, version, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,'active',1,?,?)",
                (owner_key, profile_key, kind, slot, text, opened_on, expires_on,
                 source_quote, source_turn, source_kind, author_kind, now, now),
            )
            return int(cursor.lastrowid) if cursor.lastrowid is not None else None

    def find_active_daily_entry(
        self, *, owner_key: str, profile_key: str, kind: str, slot: str,
    ) -> Optional[sqlite3.Row]:
        assert self._sqlite.is_open
        if not (owner_key and profile_key):
            return None
        return self._sqlite.fetchone(
            "SELECT * FROM daily_context WHERE owner_key=? AND profile_key=? "
            "AND kind=? AND slot=? AND status='active'",
            (owner_key, profile_key, kind, slot),
        )

    def list_active_daily_entries(
        self, *, owner_key: str, profile_key: str, today: str,
    ) -> List[sqlite3.Row]:
        """本作用域的有效行；DAILY_FACT 只留当天（即使惰性清理还没跑到）。"""
        assert self._sqlite.is_open
        if not (owner_key and profile_key):
            return []
        return self._sqlite.fetchall(
            "SELECT * FROM daily_context WHERE owner_key=? AND profile_key=? AND status='active' "
            "AND (kind<>'DAILY_FACT' OR opened_on=?) "
            "ORDER BY CASE kind WHEN 'ACTIVE_CONSTRAINT' THEN 0 "
            "WHEN 'OPEN_THREAD' THEN 1 ELSE 2 END, entry_id",
            (owner_key, profile_key, today),
        )

    def list_daily_entries_for_day(
        self, *, owner_key: str, profile_key: str, day: str, kind: Optional[str] = None,
        include_statuses: Tuple[str, ...] = ("active",), source_kind: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """按 `opened_on` 取某一天的行（昨天概况用）。

        **不删不改**：`include_statuses` 只决定读哪些状态（跨日之后 DAILY_FACT 已是
        `expired`，记录仍在，所以昨天概况要连 `expired` 一起读）。
        """
        assert self._sqlite.is_open
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
        return self._sqlite.fetchall(
            "SELECT * FROM daily_context WHERE " + " AND ".join(clauses)
            + " ORDER BY entry_id",
            params,
        )

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
        assert self._sqlite.is_open
        if not (owner_key and profile_key and since):
            return []
        since_date = str(since)[:10]
        return self._sqlite.fetchall(
            "SELECT * FROM daily_context WHERE owner_key=? AND profile_key=? "
            "AND status<>'superseded' AND kind<>'OPEN_THREAD' "
            "AND (created_at>=? OR opened_on>=?) "
            "ORDER BY COALESCE(updated_at, created_at, opened_on) DESC, entry_id DESC",
            (owner_key, profile_key, str(since), since_date),
        )

    def daily_active_stats(
        self, *, owner_key: str, profile_key: str, exclude_entry_id: Optional[int] = None,
    ) -> dict:
        """准入用的有效条数/文本预算（可按 entry_id 排除一条，供原位更新重算）。"""
        assert self._sqlite.is_open
        if not (owner_key and profile_key):
            return {"counts": {}, "chars": 0}
        clauses = ["owner_key=?", "profile_key=?", "status='active'"]
        params: List[Any] = [owner_key, profile_key]
        if exclude_entry_id is not None:
            clauses.append("entry_id<>?")
            params.append(int(exclude_entry_id))
        rows = self._sqlite.fetchall(
            "SELECT kind, text FROM daily_context WHERE " + " AND ".join(clauses), params
        )
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
        assert self._sqlite.is_open
        if not (owner_key and profile_key):
            return 0
        with self._sqlite.transaction() as conn:
            cursor = conn.execute(
                "UPDATE daily_context SET status=?, version=version+1, updated_at=? "
                "WHERE entry_id=? AND owner_key=? AND profile_key=? AND version=? AND status='active'",
                (status, now, int(entry_id), owner_key, profile_key, int(version)),
            )
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
        assert self._sqlite.is_open
        if not (owner_key and profile_key):
            return 0
        with self._sqlite.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM daily_context WHERE entry_id=? AND owner_key=? "
                "AND profile_key=? AND version=? AND status='active'",
                (int(entry_id), owner_key, profile_key, int(version)),
            ).fetchone()
            if row is None:
                return 0
            conn.execute(
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
            cursor = conn.execute(
                "UPDATE daily_context SET text=?, opened_on=?, expires_on=?, source_quote=?, "
                "source_turn=?, source_kind=?, author_kind=?, version=version+1, updated_at=? "
                "WHERE entry_id=? AND owner_key=? AND profile_key=? AND version=? "
                "AND status='active'",
                (text, opened_on, expires_on, source_quote, source_turn, source_kind,
                 row["author_kind"], now,
                 int(entry_id), owner_key, profile_key, int(version)),
            )
            if max(0, int(cursor.rowcount)) != 1:
                raise _DailyWriteRejected("daily_context cas_update conflict")
            return 1

    def expire_stale_daily_facts(self, *, owner_key: str, profile_key: str, today: str, now: str) -> int:
        """本表内的惰性清理：① 跨日的 DAILY_FACT ② 用户明确给过且已过期的行。

        只看这一张表；不动 events、不删行（改 status='expired'）。
        """
        assert self._sqlite.is_open
        if not (owner_key and profile_key):
            return 0
        with self._sqlite.transaction() as conn:
            cursor = conn.execute(
                "UPDATE daily_context SET status='expired', version=version+1, updated_at=? "
                "WHERE owner_key=? AND profile_key=? AND status='active' "
                "AND ( (kind='DAILY_FACT' AND opened_on<>?) "
                "      OR (expires_on IS NOT NULL AND expires_on<>'' AND expires_on<?) )",
                (now, owner_key, profile_key, today, today),
            )
            return max(0, int(cursor.rowcount))
