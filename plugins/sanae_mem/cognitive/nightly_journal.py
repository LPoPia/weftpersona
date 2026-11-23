from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol

from ..infra import auxiliary
from ..infra.archive import _append_fsynced_jsonl
from ..infra.atomic_io import file_lock
from ..infra.clock import SHANGHAI_TZ, shanghai_datetime, sleep_day_iso, sleep_day_window
from ..infra.contracts import ViveMindError
from ..storage.curve import bounded_half_life, initial_half_life, terminal_ttl
from ..governance.resident import COLD_DAYS as JOURNAL_SINK_DAYS
from ..governance.source_context import in_public_interaction, self_attribution
from ..infra.textutil import (
    _embed, _fingerprint, _pack_vector,
)

logger = logging.getLogger(__name__)

JOURNAL_PREFIX = "【"
JOURNAL_MARK = " · 今日小记】"
JOURNAL_TARGET = "memory"
_QUIET_DAY = "今天没有什么要记下。"
_MAX_ENTRY_CHARS = 160
_MAX_SOURCE_ITEMS = 24
_DAILY_CLOSE_HOUR = 23
_INTERNAL_MARK = re.compile(r"\[(?:SILENT|HEARTBEAT_OK)\]")
_LEADING_SYSTEM = re.compile(r"^\s*(?:系统提示|系统|system\s*prompt)\s*[：:]\s*", re.I)
_LEADING_LABEL = re.compile(r"^\s*(?:\[[^\]]{1,30}\]|【[^】]{1,30}】)\s*")


class _JournalRunWriteError(ViveMindError, RuntimeError):

    def __init__(self, stage: str, cause: BaseException):
        super().__init__("nightly journal run table write failed")
        self.stage = stage
        self.cause = cause


def journal_entry_for(day: str, body: str) -> str:
    text = re.sub(r"\s+", " ", str(body or "")).strip()
    return f"{JOURNAL_PREFIX}{day}{JOURNAL_MARK}{text}"


def is_journal_entry(entry: str) -> bool:
    value = str(entry or "").strip()
    return value.startswith(JOURNAL_PREFIX) and JOURNAL_MARK in value


def journal_entry_day(entry: str) -> str:
    value = str(entry or "").strip()
    if not is_journal_entry(value):
        return ""
    head = value[len(JOURNAL_PREFIX):value.index(JOURNAL_MARK)]
    return head.strip()


def _journal_body(entry: str) -> str:
    return str(entry or "").split(JOURNAL_MARK, 1)[-1].strip()


def _entry_sha(entry: str) -> str:
    return hashlib.sha256(str(entry or "").encode("utf-8")).hexdigest()


def _journal_source_text(text: str) -> str:
    value = _INTERNAL_MARK.sub("", str(text or ""))
    value = _LEADING_SYSTEM.sub("", value)
    value = _LEADING_LABEL.sub("", value)
    return re.sub(r"\s+", " ", value).strip()[:280]


def _journal_activity_text(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("定时活动完成："):
        value = value.split("：", 1)[1]
        if "活动结果：" not in value:
            return ""
        value = value.split("活动结果：", 1)[1]
    elif "活动结果：" in value:
        value = value.split("活动结果：", 1)[1]
    return _journal_source_text(value)


if TYPE_CHECKING:
    from ..governance.resident_store import ResidentStore
    from ..storage.sqlite_facade import SqliteFacade
    from .inner_state_types import InnerState


@dataclass(frozen=True)
class NightlyJournalTarget:
    home: Path
    data_dir: Path
    persona_name: str
    agent_context: str


class _JournalReply(Protocol):
    @property
    def text(self) -> str: ...


class NightlyJournalModel(Protocol):
    def complete(
        self, messages: list[dict[str, str]], *, task: str, temperature: float,
        max_tokens: int, timeout: float, purpose: str,
    ) -> _JournalReply | dict[str, object] | str | None: ...


class _ReadSharedInnerState(Protocol):
    def __call__(
        self, *, owner_key: str, now: float, since: float, session_id: str,
    ) -> InnerState | None: ...


class _WriteJournalStatus(Protocol):
    def __call__(
        self, *, ok: bool, stage: str, day: str, error: str, now: datetime, data_dir: Path,
    ) -> None: ...


class NightlyJournal:
    def __init__(
        self, sqlite: SqliteFacade, target: NightlyJournalTarget, *,
        model: NightlyJournalModel | None,
        current_session_id: Callable[[], str],
        shadow_enabled: Callable[[], bool],
        trusted_owner: Callable[[str | None], str],
        read_shared_inner_state: _ReadSharedInnerState,
        write_status: _WriteJournalStatus,
        curve_base_days: Callable[[str], float],
        keywords: Callable[[str], str],
        upsert_vector: Callable[[int, str, str, float, str, List[float]], None],
    ) -> None:
        self._sqlite = sqlite
        self._target = target
        self._model = model
        self._current_session_id = current_session_id
        self._shadow_enabled = shadow_enabled
        self._trusted_owner = trusted_owner
        self._read_shared_inner_state = read_shared_inner_state
        self._write_status = write_status
        self._curve_base_days = curve_base_days
        self._keywords = keywords
        self._upsert_vector = upsert_vector

    def _nightly_journal_archive(self, records: List[dict]) -> bool:
        if not records:
            return True
        try:
            path = self._target.home / "memory-archive" / "journal.jsonl"
            _append_fsynced_jsonl(
                path, (json.dumps(record, ensure_ascii=False) for record in records),
            )
        except (OSError, UnicodeError) as exc:
            logger.warning("sanae_mem: nightly journal archive failed (%s)",
                           type(exc).__name__)
            return False
        return True

    def _journal_run_row(self, owner_key: str, day: str) -> Optional[dict]:
        assert self._sqlite.is_open
        row = self._sqlite.fetchone(
            "SELECT * FROM nightly_journal_runs WHERE owner_key=? AND day=?",
            (owner_key, day),
        )
        return dict(row) if row is not None else None

    def _journal_status(self, *, ok: bool, stage: str, day: str,
                        error: str = "", now: datetime) -> None:
        self._write_status(
            ok=ok, stage=stage, error=error, day=day, now=now, data_dir=self._target.data_dir)

    def _journal_mark_run(self, owner_key: str, day: str, status: str,
                          *, entry: str = "", error: str = "", now: datetime,
                          stage: str = "journal") -> None:
        assert self._sqlite.is_open
        stamp = now.isoformat(timespec="seconds")
        try:
            self._sqlite.execute(
                """INSERT INTO nightly_journal_runs
                   (owner_key, day, status, entry_sha256, attempts, last_error,
                    completed_at, updated_at)
                   VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                   ON CONFLICT(owner_key, day) DO UPDATE SET
                     status=excluded.status,
                     entry_sha256=CASE WHEN excluded.entry_sha256<>'' THEN excluded.entry_sha256
                                       ELSE nightly_journal_runs.entry_sha256 END,
                     attempts=nightly_journal_runs.attempts + 1,
                     last_error=excluded.last_error,
                     completed_at=CASE WHEN excluded.status='completed' THEN excluded.completed_at
                                       ELSE nightly_journal_runs.completed_at END,
                     updated_at=excluded.updated_at""",
                (owner_key, day, status, _entry_sha(entry) if entry else "",
                 str(error or "")[:240], stamp if status == "completed" else "", stamp),
            )
        except sqlite3.Error as exc:
            self._journal_status(ok=False, stage=stage, day=day,
                                 error="journal_run_write_failed", now=now)
            raise _JournalRunWriteError(stage, exc) from exc
        if status == "completed":
            self._journal_status(ok=True, stage=stage, day=day, now=now)
        elif status == "failed":
            self._journal_status(ok=False, stage=stage, day=day, error=error, now=now)

    def _journal_source_items(self, owner_key: str, now: datetime) -> List[dict]:
        assert self._sqlite.is_open
        _, start, end = sleep_day_window(now)
        through = min(end, shanghai_datetime(now))
        since_epoch, through_epoch = start.timestamp(), through.timestamp()

        from .life_context import (
            _cron_rows, _extract_brief_summary, _verified_public_video,
        )
        from ..domains.video_memory import public_watch_projection_rows

        def _load_db_sources(conn: sqlite3.Connection) -> List[dict]:
            db_items: List[dict] = []
            rows = conn.execute(
                """SELECT summary, raw, created_at FROM events
                   WHERE owner_key=? AND blacklist=0
                     AND experience_subject='sanae' AND fact_subject='sanae'
                     AND created_at>=? AND created_at<?
                   ORDER BY id""",
                (owner_key, start.isoformat(), through.isoformat()),
            ).fetchall()
            for row in rows:
                text = re.sub(r"^\[[^\]]*\]\s*", "", str(row["summary"] or row["raw"] or "")).strip()
                text = re.sub(r"^我说过：", "", text).strip()
                if text:
                    db_items.append({"kind": "经历", "state": "done",
                                     "at": str(row["created_at"] or ""),
                                     "text": text[:220]})

            has_video = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='video_watches'"
            ).fetchone()
            if has_video is not None:
                videos = conn.execute(
                    """SELECT title, summary, watched_at, source_context, owner_key, bvid
                       FROM video_watches
                       WHERE owner_key = ?
                         AND (read_scope='' OR json_extract(read_scope, '$.kind')='self')
                         AND CAST(strftime('%s', watched_at) AS INTEGER) BETWEEN ? AND ?
                       ORDER BY watched_at""",
                    (owner_key, int(since_epoch), int(through_epoch) + 1),
                ).fetchall()
                seen_bvids = set()
                for row in videos:
                    title = str(row["title"] or "").strip()
                    summary = str(row["summary"] or "").strip()
                    bvid = str(row["bvid"] or "")
                    if title and _verified_public_video(
                            row["source_context"], row_owner=str(row["owner_key"] or ""),
                            owner_key=owner_key):
                        brief = _extract_brief_summary(summary) if summary else ""
                        text = f"观看了《{title}》" + (f"：{brief}" if brief else "")
                    else:
                        text = "看了一个视频"
                    if bvid:
                        seen_bvids.add(bvid)
                    db_items.append({"kind": "视频", "state": "done",
                                     "at": str(row["watched_at"] or ""),
                                     "text": text[:260]})
                for row in public_watch_projection_rows(
                        conn, owner_key=owner_key,
                        since=since_epoch, through=through_epoch):
                    bvid = str(row.get("bvid") or "")
                    title = str(row.get("title") or "").strip()
                    if not title or (bvid and bvid in seen_bvids):
                        continue
                    brief = _extract_brief_summary(str(row.get("public_excerpt") or ""))
                    text = f"观看了《{title}》" + (f"：{brief}" if brief else "")
                    if bvid:
                        seen_bvids.add(bvid)
                    db_items.append({"kind": "视频", "state": "done",
                                     "at": str(row.get("watched_at") or ""),
                                     "text": text[:260]})
            return db_items

        items: List[dict] = self._sqlite.read(_load_db_sources)

        for stamp, kind, text in _cron_rows(
            self._target.home, owner_key, since_epoch, through_epoch, data_dir=self._target.data_dir,
        ):
            thought = kind == "自由放空"
            cleaned = _journal_source_text(text) if thought else _journal_activity_text(text)
            if cleaned:
                items.append({"kind": kind, "state": "thought" if thought else "done",
                              "at": str(stamp), "text": cleaned})
                # `_cron_rows` returns newest-first; take only the latest valid entry for the day.
                break

        mood = self._read_shared_inner_state(
            owner_key=owner_key, now=through_epoch, since=since_epoch,
            session_id=self._current_session_id())
        if mood is not None and str(mood.text or "").strip():
            items.append({"kind": "心境", "state": "thought",
                          "at": str(mood.updated_at),
                          "text": str(mood.text).strip()[:240]})

        deduped: List[dict] = []
        seen = set()
        for item in items:
            text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append({**item, "text": text})
            if len(deduped) >= _MAX_SOURCE_ITEMS:
                break
        return deduped

    @staticmethod
    def _journal_flash_text(raw: _JournalReply | dict[str, object] | str) -> str:
        if isinstance(raw, dict):
            text = raw.get("text") or raw.get("journal") or raw.get("content") or ""
        else:
            text = raw if isinstance(raw, str) else raw.text
        text = str(text or "").strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                text = str(parsed.get("text") or parsed.get("journal") or parsed.get("content") or "")
        text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip().strip('"“”')
        text = re.sub(r"^(?:今日小记|小记)\s*[：:]\s*", "", text)
        if (not text or len(text) > _MAX_ENTRY_CHARS
                or any(mark in text for mark in ("BEHAVIOR", "记忆系统", "插件", "定时任务"))):
            return ""
        return text.rstrip("；;，, ") + (
            "" if text.endswith(("。", "！", "？", "!", "?")) else "。")

    def _journal_compose(self, owner_key: str, day: str, now: datetime) -> dict:
        sources = self._journal_source_items(owner_key, now)
        if not sources:
            return {"ok": True, "body": _QUIET_DAY, "sources": 0, "via": "quiet_day"}
        if self._model is None:
            return {"ok": False, "reason": "flash_unavailable", "sources": len(sources)}
        source_text = json.dumps(sources, ensure_ascii=False)
        year = day.split("-", 1)[0] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) else ""
        date_label = f"{day}（{year}年，今天）" if year else day
        persona_name = self._target.persona_name.strip() or "AI伙伴"
        prompt = (
            f"你是{persona_name}的今日小记整理器。下面每条当天片段都属于{persona_name}自己，"
            "带 kind、state、at、text：state=done 是已经发生的事，"
            "state=thought 是当时的想法或感受。\n"
            "只根据这些片段，用第一人称写一段自然的小记：写她实际做过/看过的事、"
            "独自想过的事，以及由这些事自然引出的自己的心情。\n"
            "硬边界：\n"
            "1) 只写片段能支持的事实；片段没有的动作、地点、人物、天气、工具、"
            "具体时刻和先后顺序一律不补。\n"
            "2) state=thought 的内容不能写成已经做了；片段里说打算、晚点、想要、"
            "如果，且没有明确完成表述的，也不能写成已完成；片段明确说"
            "“我已兑现/已经做完/完成了”的，按已完成写。\n"
            "3) 事实、完成状态和时间以片段为准；at 是素材时间，不要另编时间。\n"
            "4) 日期是小记所属的那一天；片段里写明的年份和日期是事实。"
            "需要换成相对说法时，以这个日期为准：同一年是今年，下一年是明年，"
            "上一年是去年；不要凭猜想的日期把来源年份说错。\n"
            "5) 片段少就只写这一两件事，宁可短，不要补成整天的剧情；平淡就平淡。\n"
            "只写她自己的经历，不写别人的私事，不写系统、记忆、插件、定时任务或"
            "工具机制，不写行动清单、解释或建议。只输出正文，不要标题、编号、"
            "引号或代码块，不超过160字。"
        )
        raw: _JournalReply | dict[str, object] | str | None = auxiliary.complete_simple(
            self._model,
            [{"role": "system", "content": prompt},
             {"role": "user",
              "content": f"日期：{date_label}\n当天片段：{source_text}"}],
            temperature=0.2, max_tokens=260, timeout=25,
            purpose="nightly_journal", failure_message="nightly journal Flash unavailable",
        )
        if raw is None:
            return {"ok": False, "reason": "flash_unavailable", "sources": len(sources)}
        body = self._journal_flash_text(raw)
        if not body:
            return {"ok": False, "reason": "flash_invalid", "sources": len(sources)}
        return {"ok": True, "body": body, "sources": len(sources), "via": "flash"}

    def _journal_sink_event(self, owner_key: str, entry: str, *, now: datetime,
                            reason: str) -> Optional[int]:
        assert self._sqlite.is_open
        fingerprint = _fingerprint(_journal_body(entry))
        existing = self._sqlite.fetchone(
            """SELECT id FROM events
               WHERE owner_key=? AND promoted_reason=? AND original_fingerprint=?
                 AND blacklist=0 ORDER BY id DESC LIMIT 1""",
            (owner_key, "nightly_journal_sink", fingerprint),
        )
        if existing is not None:
            return int(existing["id"])
        stamp = now.isoformat(timespec="seconds")
        day = journal_entry_day(entry) or sleep_day_iso(now)
        ts = f"{day} 23:00"
        body = _journal_body(entry)
        summary = body[:150]
        vector = _embed(body, data_dir=self._target.data_dir)
        importance = 0.7
        half_life = bounded_half_life(
            initial_half_life(self._curve_base_days(body), importance), "short")
        try:
            anchor = datetime.fromisoformat(f"{day}T23:00:00").replace(tzinfo=SHANGHAI_TZ)
        except (TypeError, ValueError):
            anchor = now
        expire = terminal_ttl(anchor, half_life)
        subject, fact, involved, scope = self_attribution(owner_key, fact_subject="sanae")
        values = {
            "ts": ts, "raw": entry, "summary": summary,
            "summary_source": "original", "keywords": self._keywords(body),
            "importance": importance, "confidence": 0.6, "topic": "生活",
            "ttl_expire": expire.isoformat(), "created_at": stamp, "updated_at": stamp,
            "vec": _pack_vector(vector), "session_id": self._current_session_id(),
            "tier": "short", "promoted_at": "",
            "promoted_reason": "nightly_journal_sink",
            "owner_key": owner_key,
            "experience_subject": subject, "fact_subject": fact,
            "involved_owner_keys": involved, "read_scope": scope,
            "last_active_at": stamp, "recall_count": 0,
            "half_life_days": half_life, "curve_ttl": expire.isoformat(),
            "original_fingerprint": fingerprint,
            "candidate_reason": reason,
        }
        columns = list(values)
        cursor = self._sqlite.execute(
            "INSERT INTO events (%s) VALUES (%s)" % (
                ", ".join(columns), ", ".join("?" for _ in columns)),
            [values[name] for name in columns],
        )
        assert cursor.lastrowid is not None
        event_id = int(cursor.lastrowid)
        self._upsert_vector(event_id, body, "生活", importance, ts, vector)
        return event_id

    def _journal_sink(self, store: ResidentStore, entries: List[str], owner_key: str, *,
                      now: datetime, reason: str) -> dict:
        clean = [str(entry) for entry in entries if is_journal_entry(entry)]
        if not clean:
            return {"ok": True, "removed": 0}
        records = [{
            "archived_at": now.isoformat(timespec="seconds"),
            "reason": reason,
            "owner_key": owner_key,
            "entry": entry,
            "entry_sha256": _entry_sha(entry),
        } for entry in clean]
        if not self._nightly_journal_archive(records):
            return {"ok": False, "reason": "archive_failed", "removed": 0}
        event_ids = []
        try:
            for entry in clean:
                event_id = self._journal_sink_event(owner_key, entry, now=now, reason=reason)
                if event_id is None:
                    return {"ok": False, "reason": "event_write_failed",
                            "removed": 0, "event_ids": event_ids}
                event_ids.append(event_id)
        except (sqlite3.Error, OSError, UnicodeError, ViveMindError) as exc:
            logger.warning("sanae_mem: nightly journal sink event failed (%s)",
                           type(exc).__name__)
            return {"ok": False, "reason": f"event_write_failed:{type(exc).__name__}",
                    "removed": 0, "event_ids": event_ids}
        removed = 0
        for entry in clean:
            try:
                result = store.remove(JOURNAL_TARGET, entry)
            except (OSError, UnicodeError, ViveMindError) as exc:
                logger.warning("sanae_mem: nightly journal MEMORY remove failed (%s)",
                               type(exc).__name__)
                return {"ok": False, "reason": "memory_remove_failed",
                        "removed": removed, "event_ids": event_ids}
            if not result.get("success"):
                return {"ok": False, "reason": "memory_remove_failed",
                        "removed": removed, "event_ids": event_ids}
            removed += 1
        return {"ok": True, "removed": removed, "event_ids": event_ids}

    def _journal_collapse(self, store: ResidentStore, owner_key: str, *, now: datetime) -> dict:
        entries = list(store.entries(JOURNAL_TARGET))
        by_day: Dict[str, List[str]] = {}
        for entry in entries:
            if is_journal_entry(entry):
                day = journal_entry_day(entry)
                if day:
                    by_day.setdefault(day, []).append(entry)
        duplicates: List[str] = []
        for day in sorted(by_day):
            group = by_day[day]
            if len(group) > 1:
                duplicates.extend(group[:-1])
        if not duplicates:
            return {"ok": True, "removed": 0}
        return self._journal_sink(store, duplicates, owner_key, now=now,
                                  reason="journal_collapse")

    def _journal_expire(self, store: ResidentStore, owner_key: str, *, now: datetime) -> dict:
        cutoff = sleep_day_iso(shanghai_datetime(now) - timedelta(days=JOURNAL_SINK_DAYS))
        entries = [entry for entry in store.entries(JOURNAL_TARGET)
                   if is_journal_entry(entry) and journal_entry_day(entry)
                   and journal_entry_day(entry) < cutoff]
        if not entries:
            return {"ok": True, "removed": 0}
        return self._journal_sink(store, entries, owner_key, now=now,
                                  reason="journal_window")

    def _journal_make_room(self, store: ResidentStore, owner: str, needed: int, *,
                           now: datetime) -> dict:
        entries = list(store.entries(JOURNAL_TARGET))
        journal_entries = [entry for entry in entries if is_journal_entry(entry)]
        journal_entries.sort(key=lambda entry: (journal_entry_day(entry), entries.index(entry)))
        selected: List[str] = []
        current = len("\n§\n".join(entries))
        limit = int(store.char_limit(JOURNAL_TARGET))
        for entry in journal_entries:
            if current + needed <= limit:
                break
            selected.append(entry)
            current -= len(entry) + 3
        if current + needed > limit:
            return {"ok": False, "reason": "memory_full"}
        return self._journal_sink(store, selected, owner, now=now, reason="journal_overflow")

    def run(self, *, now: datetime | None, owner_key: str | None, force: bool) -> dict:
        now = shanghai_datetime(now)
        day = sleep_day_iso(now)
        if not self._sqlite.is_open:
            self._journal_status(ok=False, stage="init", day=day,
                                 error="not_initialized", now=now)
            return {"ok": False, "reason": "not_initialized"}
        if self._target.agent_context not in {"primary", "cron"}:
            return {"ok": False, "reason": "unsupported_context"}
        if not force and self._shadow_enabled():
            return {"ok": False, "reason": "shadow"}
        if in_public_interaction():
            return {"ok": False, "reason": "public_interaction"}
        owner = self._trusted_owner(owner_key)
        if not owner:
            self._journal_status(ok=False, stage="identity", day=day,
                                 error="untrusted_owner", now=now)
            return {"ok": False, "reason": "untrusted_owner"}
        if now.hour < _DAILY_CLOSE_HOUR and not force:
            return {"ok": False, "reason": "before_close"}
        try:
            from ..governance.resident_store import ResidentStore
        except ModuleNotFoundError as exc:
            store_module = f"{__package__}".removesuffix(".cognitive") + ".governance.resident_store"
            if exc.name != store_module:
                raise
            self._journal_status(ok=False, stage="memory_store", day=day,
                                 error=f"memory_store_unavailable:{type(exc).__name__}",
                                 now=now)
            return {"ok": False, "reason": f"memory_store_unavailable:{type(exc).__name__}"}
        with file_lock(self._target.data_dir / "nightly_journal.lock"):
            try:
                run = self._journal_run_row(owner, day)
                if run and run.get("status") == "completed":
                    self._journal_status(ok=True, stage="already", day=day, now=now)
                    return {"ok": True, "already": True, "day": day,
                            "reason": "already_completed"}
                store = ResidentStore(self._target.home / "memories")
                # ResidentStore.entries("memory") reads ONLY MEMORY.md and never touches USER.md.
                entries = list(store.entries(JOURNAL_TARGET))
                existing = next((entry for entry in entries
                                 if is_journal_entry(entry) and journal_entry_day(entry) == day), "")
                if existing:
                    self._journal_mark_run(owner, day, "completed", entry=existing,
                                           now=now, stage="already")
                    return {"ok": True, "already": True, "day": day,
                            "entry": existing, "reason": "entry_already_present"}

                self._journal_mark_run(owner, day, "running", now=now, stage="compose")
                composed = self._journal_compose(owner, day, now)
                if not composed.get("ok"):
                    self._journal_mark_run(
                        owner, day, "failed",
                        error=str(composed.get("reason") or "journal_compose_failed"),
                        now=now, stage="compose")
                    return {"ok": False, "reason": str(composed.get("reason")),
                            "day": day, "sources": composed.get("sources", 0)}
                collapsed = self._journal_collapse(store, owner, now=now)
                if not collapsed.get("ok"):
                    self._journal_mark_run(owner, day, "failed",
                                           error=str(collapsed.get("reason")),
                                           now=now, stage="collapse")
                    return {"ok": False, "reason": str(collapsed.get("reason")),
                            "day": day}
                expired = self._journal_expire(store, owner, now=now)
                if not expired.get("ok"):
                    self._journal_mark_run(owner, day, "failed",
                                           error=str(expired.get("reason")),
                                           now=now, stage="expire")
                    return {"ok": False, "reason": str(expired.get("reason")),
                            "day": day}
                entry = journal_entry_for(day, str(composed["body"]))
                room = self._journal_make_room(
                    store, owner, len(entry) + 3, now=now)
                if not room.get("ok"):
                    self._journal_mark_run(owner, day, "failed",
                                           error=str(room.get("reason")),
                                           now=now, stage="capacity")
                    return {"ok": False, "reason": str(room.get("reason")), "day": day}
                result = store.add(JOURNAL_TARGET, entry)
                if not result.get("success"):
                    self._journal_mark_run(
                        owner, day, "failed",
                        error=str(result.get("error") or "memory_add_failed"),
                        now=now, stage="write")
                    return {"ok": False, "reason": "memory_add_failed",
                            "error": str(result.get("error") or ""), "day": day}
                self._journal_mark_run(owner, day, "completed", entry=entry,
                                       now=now, stage="write")
                return {"ok": True, "already": False, "day": day, "entry": entry,
                        "collapsed": collapsed.get("removed", 0),
                        "expired": expired.get("removed", 0),
                        "sunk": room.get("removed", 0),
                        "sources": composed.get("sources", 0),
                        "via": composed.get("via", "")}
            except _JournalRunWriteError as exc:
                logger.warning("sanae_mem: nightly journal run state not written at %s (%s)",
                               exc.stage, type(exc.cause).__name__)
                return {"ok": False, "reason": "journal_run_write_failed", "day": day}
            except (sqlite3.Error, OSError, UnicodeError, ViveMindError) as exc:
                logger.warning("sanae_mem: nightly journal failed (%s)", type(exc).__name__)
                try:
                    self._journal_mark_run(owner, day, "failed",
                                           error=f"{type(exc).__name__}:{exc}",
                                           now=now, stage="run")
                except _JournalRunWriteError as mark_exc:
                    logger.warning("sanae_mem: nightly journal run state not written (%s)",
                                   type(mark_exc.cause).__name__)
                return {"ok": False, "reason": f"{type(exc).__name__}:{exc}",
                        "day": day}
