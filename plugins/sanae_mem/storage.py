from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import site
import sqlite3
import sys
from threading import Lock
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List

from hermes_cli.active_sessions import _FileLock
from utils import atomic_json_write

try:
    from .curve import (
        HALF_LIFE_MAX_LONG_DAYS, HALF_LIFE_MAX_SHORT_DAYS, HALF_LIFE_MIN_DAYS,
    )
    from .words import *
    from .textutil import *
    from .source_context import in_public_interaction
    from .textutil import hermes_home, provider_home
    from .schemas import *
except ImportError:
    from curve import (
        HALF_LIFE_MAX_LONG_DAYS, HALF_LIFE_MAX_SHORT_DAYS, HALF_LIFE_MIN_DAYS,
    )
    from words import *
    from textutil import *
    from source_context import in_public_interaction
    from textutil import hermes_home, provider_home
    from schemas import *

logger = logging.getLogger(__name__)
_STATUS_WRITER = Lock()

_CURVE_COLUMNS = frozenset({"recall_count", "last_active_at", "half_life_days", "curve_ttl"})
_CURVE_BACKFILL_SQL = (
    """UPDATE events SET last_active_at = COALESCE(
         NULLIF(asked_at, ''), NULLIF(updated_at, ''), created_at)""",
    """UPDATE events SET recall_count = CASE
         WHEN COALESCE(tier, '') IN ('long', 'important', 'distilled')
           THEN MAX(3, COALESCE(cross_session_hits, 0))
         ELSE MAX(0, COALESCE(cross_session_hits, 0)) END""",
    f"""UPDATE events SET half_life_days = MIN(
         CASE WHEN COALESCE(tier, '') IN ('long', 'important', 'distilled')
           THEN {HALF_LIFE_MAX_LONG_DAYS} ELSE {HALF_LIFE_MAX_SHORT_DAYS} END,
         MAX(
           CASE WHEN COALESCE(tier, '') IN ('long', 'important', 'distilled')
             THEN {HALF_LIFE_MAX_SHORT_DAYS} ELSE {HALF_LIFE_MIN_DAYS} END,
           (CASE
              WHEN NULLIF(ttl_expire, '') IS NOT NULL
                AND julianday(ttl_expire) > julianday(created_at)
                THEN MIN({HALF_LIFE_MAX_LONG_DAYS}, MAX(1.0, julianday(ttl_expire) - julianday(created_at)))
              WHEN COALESCE(topic, '') IN ('情绪', '关系') THEN 90.0
              ELSE 6.0 END)
           * (1.0 + 2.5 * COALESCE(importance, 0.5))
           * (1.0 + 0.5 * MIN(1.0, MAX(0.0, COALESCE(affect, 0) / 3.0)))))""",
    "UPDATE events SET curve_ttl = COALESCE(ttl_expire, '')",
)


try:
    from .event_store import (
        EventStoreMixin, HISTORICAL_LIMIT_NOTICE, SUMMARY_SOURCE_DISTILLED,
        SUMMARY_SOURCE_HISTORICAL, SUMMARY_SOURCE_ORIGINAL,
    )
    from .event_review_store import EventReviewStoreMixin
    from .event_maintenance import EventMaintenanceMixin
    from .impression import IMPRESSION_DDL
    from .inner_state import InnerStateMixin
    from .state_store import StateStoreMixin
except ImportError:
    from event_store import (
        EventStoreMixin, HISTORICAL_LIMIT_NOTICE, SUMMARY_SOURCE_DISTILLED,
        SUMMARY_SOURCE_HISTORICAL, SUMMARY_SOURCE_ORIGINAL,
    )
    from event_review_store import EventReviewStoreMixin
    from event_maintenance import EventMaintenanceMixin
    from impression import IMPRESSION_DDL
    from inner_state import InnerStateMixin
    from state_store import StateStoreMixin
class StorageMixin(EventStoreMixin, EventReviewStoreMixin, EventMaintenanceMixin, StateStoreMixin, InnerStateMixin):
        def _open_sqlite(self) -> None:
            db_path = self._data_dir / "sanae.db"
            self._conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT NOT NULL,
                  raw TEXT NOT NULL,
                  -- 2026-10-09 统一重构准备：raw＝原文全文（不截断）；summary＝展示／检索摘要。
                  summary TEXT DEFAULT '',
                  -- 2026-10-10：summary 的**来源**（后台限度标识；不进注入/展示）。
                  summary_source TEXT DEFAULT '',
                  keywords TEXT DEFAULT '',
                  importance REAL DEFAULT 0.5,
                  confidence REAL DEFAULT 0.6,
                  topic TEXT DEFAULT '',
                  ttl_expire TEXT,
                  asked INTEGER DEFAULT 0,
                  asked_at TEXT NOT NULL DEFAULT '',
                  blacklist INTEGER DEFAULT 0,
                  created_at TEXT NOT NULL,
                  updated_at TEXT,
                  vec BLOB,
                  session_id TEXT DEFAULT '',
                  tier TEXT DEFAULT 'short',
                  -- ⚠️ promoted_at 的**双语义**（2026-09-17 C 批收窄，历史不可分、不回填）：
                  --   历史语义＝「进过候选池」（mark_pending 也写它）⇒ 生产上存在
                  --   tier 仍为 'pending' 而 promoted_at 非空的行；
                  --   自 2026-09-17 起新写**只表示「确认晋升」**（long／important／distilled），
                  --   mark_pending 改写 staged_at。
                  --   ⇒ **不能再用 promoted_at 非空单独推出 tier**；判「是否确认晋升」必须同时看 tier。
                  promoted_at TEXT,
                  cross_session_hits INTEGER DEFAULT 0,
                  pending_to TEXT DEFAULT '',
                  distilled_at TEXT,
                  expired_at TEXT,
                  -- 间隔重激活半衰期：跨日有效召回次数、最后一次活跃时间戳、当前半衰期（天）。
                  recall_count INTEGER NOT NULL DEFAULT 0,
                  last_active_at TEXT NOT NULL DEFAULT '',
                  half_life_days REAL NOT NULL DEFAULT 14.0,
                  -- 曲线自己算出的 ttl_expire 的镜像。两者不一致＝显式参数或人工改期，
                  -- 读侧据此走旧过期路径（判定见 curve.CURVE_TTL_NATURAL_SQL）。
                  curve_ttl TEXT NOT NULL DEFAULT '',
                  original_fingerprint TEXT NOT NULL DEFAULT '',
                  owner_key TEXT NOT NULL DEFAULT '',
                  candidate_reason TEXT DEFAULT '',
                  staged_at TEXT,
                  promoted_reason TEXT DEFAULT '',
                  asked_via TEXT DEFAULT '',
                  asked_session TEXT DEFAULT '',
                  -- 公开交流批次（2026-10-13）：这条事件**当时的可信出处**（宿主接入层字段的
                  -- canonical JSON：kind/author_id/platform/ref/time/certainty/visibility）。
                  -- 旧行留空＝unknown，不批量猜填；它不是 USER 证言，也不由模型改。
                  source_context TEXT NOT NULL DEFAULT '',
                  -- 已由**可信层显式审阅**的公开投影（host-only 写点 `set_public_excerpt`）。
                  -- 空＝没有公开视图 ⇒ 公开检索看不见这一行；公开检索只返回这一列，不返回
                  -- raw/summary 或任何后台快照。
                  public_excerpt TEXT NOT NULL DEFAULT '',
                  -- 个人经历主体／涉及对象／读取范围（2026-10-20 多用户第一批）。
                  -- 旧行全部留空＝unknown，不批量回填；owner_key 仍是用户隔离键，不承担这三者。
                  experience_subject TEXT NOT NULL DEFAULT '',
                  fact_subject TEXT NOT NULL DEFAULT '',
                  involved_owner_keys TEXT NOT NULL DEFAULT '',
                  read_scope TEXT NOT NULL DEFAULT '',
                  -- Host-only neutral safe view for another user's shared
                  -- interaction; never derived from that user's raw/summary/vec.
                  shared_view TEXT NOT NULL DEFAULT ''
                );
                -- Every original utterance in a near-duplicate group keeps its own immutable
                -- text, time and *optional* source proof; one event ID remains the recall unit.
                CREATE TABLE IF NOT EXISTS event_occurrences (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                  raw TEXT NOT NULL,
                  observed_at TEXT NOT NULL,
                  source_snapshot TEXT NOT NULL DEFAULT '',
                  source_snapshot_sha256 TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_event_occurrences_event
                  ON event_occurrences(event_id, id);
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
                  USING fts5(raw, keywords, content='events', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
                  INSERT INTO events_fts(rowid, raw, keywords) VALUES (new.id, new.raw, new.keywords);
                END;
                CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
                  INSERT INTO events_fts(events_fts, rowid, raw, keywords)
                    VALUES ('delete', old.id, old.raw, old.keywords);
                END;
                CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
                  INSERT INTO events_fts(events_fts, rowid, raw, keywords)
                    VALUES ('delete', old.id, old.raw, old.keywords);
                  INSERT INTO events_fts(rowid, raw, keywords) VALUES (new.id, new.raw, new.keywords);
                END;
                -- 梗与表情记忆区：独立于普通 events。文字/解释/别名/来源在这里，
                -- 素材原文件在 <data_dir>/memes/，同图按 sha256 去重；不建图片向量。
                CREATE TABLE IF NOT EXISTS meme_records (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL DEFAULT '',
                  meaning TEXT NOT NULL DEFAULT '',
                  aliases TEXT NOT NULL DEFAULT '[]',
                  source TEXT NOT NULL DEFAULT '',
                  source_ref TEXT NOT NULL DEFAULT '',
                  source_time TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  vec BLOB,
                  owner_key TEXT NOT NULL DEFAULT '',
                  session_id TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS meme_assets (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  record_id INTEGER NOT NULL REFERENCES meme_records(id) ON DELETE CASCADE,
                  path TEXT NOT NULL,
                  sha256 TEXT NOT NULL UNIQUE,
                  mime TEXT NOT NULL,
                  bytes INTEGER NOT NULL DEFAULT 0,
                  width INTEGER NOT NULL DEFAULT 0,
                  height INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_meme_assets_record
                  ON meme_assets(record_id);
                CREATE TABLE IF NOT EXISTS meme_forgotten (
                  fingerprint TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS meme_fts
                  USING fts5(title, meaning, aliases, content='meme_records', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS meme_ai AFTER INSERT ON meme_records BEGIN
                  INSERT INTO meme_fts(rowid, title, meaning, aliases)
                    VALUES (new.id, new.title, new.meaning, new.aliases);
                END;
                CREATE TRIGGER IF NOT EXISTS meme_ad AFTER DELETE ON meme_records BEGIN
                  INSERT INTO meme_fts(meme_fts, rowid, title, meaning, aliases)
                    VALUES ('delete', old.id, old.title, old.meaning, old.aliases);
                END;
                CREATE TRIGGER IF NOT EXISTS meme_au AFTER UPDATE ON meme_records BEGIN
                  INSERT INTO meme_fts(meme_fts, rowid, title, meaning, aliases)
                    VALUES ('delete', old.id, old.title, old.meaning, old.aliases);
                  INSERT INTO meme_fts(rowid, title, meaning, aliases)
                    VALUES (new.id, new.title, new.meaning, new.aliases);
                END;
                CREATE TABLE IF NOT EXISTS forgotten (
                  fingerprint TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL
                );
                -- 核心①（2026-10-09 返工）：长期候选的**语义审核结果**由后端真实生成并保全。
                -- 它把「实际候选正文 hash ＋ 固定目标 ＋ 事件 id/全文 hash/版本 ＋ 审核当时的常驻
                -- baseline」绑在一起；审批重放只能按这张记录核对，**不接受调用方自报 duty/hash**。
                -- 换正文／换目标／换事件／常驻变了 ⇒ 记录对不上 ⇒ 重放拒写并要求重新审核。
                CREATE TABLE IF NOT EXISTS event_reviews (
                  event_id INTEGER NOT NULL,
                  review_sha256 TEXT NOT NULL,
                  target TEXT NOT NULL,
                  duty TEXT NOT NULL,
                  content_sha256 TEXT NOT NULL,
                  event_sha256 TEXT NOT NULL,
                  event_version TEXT NOT NULL DEFAULT '',
                  baseline_sha256 TEXT NOT NULL DEFAULT '',
                  source_kind TEXT NOT NULL DEFAULT 'unknown',
                  significance TEXT NOT NULL DEFAULT '',
                  -- 审核过的**候选正文**（插件内部绑定/复用；不进注入、不进报告、面板不出）。
                  -- 它只用于「同一事件同一版本不重复花一次辅助调用」与提案文本的忠实回放。
                  reviewed_text TEXT NOT NULL DEFAULT '',
                  -- Evidence IDs and fingerprints of the event text, version and source snapshot.
                  evidence_group TEXT NOT NULL DEFAULT '',
                  reviewed_at TEXT NOT NULL,
                  PRIMARY KEY (event_id, review_sha256)
                );
                -- Audit-only support links; USER prose stays short and is never auto-edited.
                -- A recognition has at most three CURRENT representative event references.
                CREATE TABLE IF NOT EXISTS recognition_review_targets (
                  review_sha256 TEXT PRIMARY KEY,
                  covered_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recognition_evidence (
                  recognition_sha256 TEXT NOT NULL,
                  event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE RESTRICT,
                  review_sha256 TEXT NOT NULL,
                  linked_at TEXT NOT NULL,
                  PRIMARY KEY (recognition_sha256, event_id)
                );
                CREATE TABLE IF NOT EXISTS mood_state (
                  channel TEXT PRIMARY KEY CHECK(channel IN ('negative', 'warm')),
                  by_topic TEXT NOT NULL DEFAULT '[]',
                  cause_event_id INTEGER,
                  cause_topic TEXT NOT NULL,
                  cause_text TEXT NOT NULL,
                  intensity REAL NOT NULL CHECK(intensity >= 0 AND intensity <= 1),
                  turns_remaining INTEGER NOT NULL DEFAULT 0,
                  started_at TEXT NOT NULL,
                  last_reinforced_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  expires_at TEXT,
                  FOREIGN KEY(cause_event_id) REFERENCES events(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS inner_state_revisions (
                  profile_key TEXT NOT NULL CHECK(profile_key <> ''),
                  revision INTEGER NOT NULL CHECK(revision > 0),
                  text TEXT NOT NULL,
                  recent_words TEXT NOT NULL,
                  -- 2026-11-06: the moment the current words were picked up verbatim.
                  -- Rows written before the split keep NULL and read as felt_at.
                  recent_words_at REAL,
                  variables TEXT NOT NULL,
                  felt_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  processed_at REAL,
                  source TEXT NOT NULL CHECK(source IN ('manual','flash')),
                  thought_source TEXT NOT NULL CHECK(thought_source IN ('manual','flash')),
                  turn_id TEXT NOT NULL,
                  changed INTEGER NOT NULL CHECK(changed IN (0,1)),
                  PRIMARY KEY(profile_key, revision)
                );
                -- B2（2026-10-05）当日上下文独立小表：不扩 events、不进召回/晋升/蒸馏。
                -- 作用域与 events 不同：owner_key/profile_key 必须非空（缺一个就 fail-closed，
                -- 绝不落到 events 允许的空 owner 共享桶）。同槽同时只允许一条 active（部分唯一索引）。
                CREATE TABLE IF NOT EXISTS daily_context (
                  entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  owner_key TEXT NOT NULL CHECK(owner_key <> ''),
                  profile_key TEXT NOT NULL CHECK(profile_key <> ''),
                  kind TEXT NOT NULL CHECK(kind IN ('DAILY_FACT','ACTIVE_CONSTRAINT','OPEN_THREAD')),
                  slot TEXT NOT NULL CHECK(slot <> ''),
                  text TEXT NOT NULL,
                  opened_on TEXT NOT NULL,
                  expires_on TEXT,
                  source_quote TEXT NOT NULL DEFAULT '',
                  source_turn TEXT NOT NULL DEFAULT '',
                  -- B2+（2026-10-06）：'statement'＝用户本轮明确陈述；'inference'＝由本轮证据
                  -- 推出的短期推测（渲染必须与事实分开、并标"未确认"）。老行一律默认
                  -- 'statement' —— 迁移前能进这张表的都来自用户明确陈述，语义有依据。
                  source_kind TEXT NOT NULL DEFAULT 'statement',
                  -- Who authored the daily statement. Old rows stay unknown; only a
                  -- verified fixed maintainer write records maintainer here.
                  author_kind TEXT NOT NULL DEFAULT 'unknown',
                  status TEXT NOT NULL CHECK(status IN ('active','superseded','closed','expired')),
                  version INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_context_active_slot
                  ON daily_context(owner_key, profile_key, kind, slot) WHERE status='active';
                CREATE INDEX IF NOT EXISTS idx_daily_context_scope
                  ON daily_context(owner_key, profile_key, status);
                CREATE INDEX IF NOT EXISTS idx_daily_context_day
                  ON daily_context(owner_key, profile_key, opened_on);
                """
            )
            self._migrate_events_schema(db_path)
            self._migrate_daily_context_schema()
            self._migrate_event_reviews_schema()
            self._migrate_inner_state_schema()
            self._conn.executescript(IMPRESSION_DDL)
            self._conn.commit()


        def _migrate_events_schema(self, db_path: Path) -> None:
            assert self._conn is not None
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
            additions = {
                "tier": "TEXT DEFAULT 'short'",
                "summary": "TEXT DEFAULT ''",
                "summary_source": "TEXT DEFAULT ''",
                "promoted_at": "TEXT",
                "cross_session_hits": "INTEGER DEFAULT 0",
                "pending_to": "TEXT DEFAULT ''",
                "distilled_at": "TEXT",
                "expired_at": "TEXT",
                "recall_count": "INTEGER NOT NULL DEFAULT 0",
                "last_active_at": "TEXT NOT NULL DEFAULT ''",
                "half_life_days": "REAL NOT NULL DEFAULT 14.0",
                "curve_ttl": "TEXT NOT NULL DEFAULT ''",
                "original_fingerprint": "TEXT NOT NULL DEFAULT ''",
                "asked_at": "TEXT NOT NULL DEFAULT ''",
                "a_serious": "INTEGER NOT NULL DEFAULT 0",
                "a_intent": "INTEGER NOT NULL DEFAULT 0",
                "a_rel": "INTEGER NOT NULL DEFAULT 0",
                "a_self": "INTEGER NOT NULL DEFAULT 0",
                "a_conflict": "INTEGER NOT NULL DEFAULT 0",
                "affect": "INTEGER NOT NULL DEFAULT 0",
                "owner_key": "TEXT NOT NULL DEFAULT ''",
                "candidate_reason": "TEXT DEFAULT ''",
                "staged_at": "TEXT",
                "promoted_reason": "TEXT DEFAULT ''",
                "asked_via": "TEXT DEFAULT ''",
                "asked_session": "TEXT DEFAULT ''",
                "source_snapshot": "TEXT NOT NULL DEFAULT ''",
                "source_snapshot_sha256": "TEXT NOT NULL DEFAULT ''",
                "source_context": "TEXT NOT NULL DEFAULT ''",
                "public_excerpt": "TEXT NOT NULL DEFAULT ''",
                "experience_subject": "TEXT NOT NULL DEFAULT ''",
                "fact_subject": "TEXT NOT NULL DEFAULT ''",
                "involved_owner_keys": "TEXT NOT NULL DEFAULT ''",
                "read_scope": "TEXT NOT NULL DEFAULT ''",
                "shared_view": "TEXT NOT NULL DEFAULT ''",
            }
            missing = [(name, definition) for name, definition in additions.items() if name not in columns]
            if missing:
                stamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
                backup_path = db_path.with_name(f"{db_path.name}.bak-p6-{stamp}")
                _prune_sibling_backups(db_path, "p6")
                destination = sqlite3.connect(backup_path)
                try:
                    self._conn.backup(destination)
                finally:
                    destination.close()
                with self._lock:
                    for name, definition in missing:
                        self._conn.execute(f"ALTER TABLE events ADD COLUMN {name} {definition}")
                    if {name for name, _ in missing} & _CURVE_COLUMNS:
                        # 回填是 UPDATE，会触发 events_au 的 fts5 'delete'；旧备份或外部工具写进来的
                        # 行本不在外部内容索引里，缺这一步会当场报 "database disk image is malformed"。
                        self._conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
                        for statement in _CURVE_BACKFILL_SQL:
                            self._conn.execute(statement)
                    self._conn.commit()
            self._repair_curve_backfill()

            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_owner_key ON events(owner_key)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_experience_subject "
                "ON events(experience_subject)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_self_attribution "
                "ON events(owner_key, experience_subject, fact_subject)")
            self._conn.commit()

        def _repair_curve_backfill(self) -> None:
            assert self._conn is not None
            over_cap = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE half_life_days > ? "
                "OR (COALESCE(tier, 'short') IN ('short', 'pending') AND half_life_days > ?)",
                (HALF_LIFE_MAX_LONG_DAYS, HALF_LIFE_MAX_SHORT_DAYS),
            ).fetchone()[0]
            empty_sum_rows = self._conn.execute(
                "SELECT id, raw, tier, summary_source FROM events "
                "WHERE trim(COALESCE(summary, '')) = '' AND trim(COALESCE(raw, '')) <> ''"
            ).fetchall()
            if not (over_cap or empty_sum_rows):
                return
            with self._lock:
                self._conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
                if over_cap:
                    # 只压上界、不抬下界：存量修复不得让旧行重新获得寿命。
                    self._conn.execute(
                        "UPDATE events SET half_life_days = ? WHERE half_life_days > ?",
                        (HALF_LIFE_MAX_LONG_DAYS, HALF_LIFE_MAX_LONG_DAYS))
                    self._conn.execute(
                        "UPDATE events SET half_life_days = ? "
                        "WHERE COALESCE(tier, 'short') IN ('short', 'pending') AND half_life_days > ?",
                        (HALF_LIFE_MAX_SHORT_DAYS, HALF_LIFE_MAX_SHORT_DAYS))
                for row in empty_sum_rows:
                    cleaned = re.sub(r"^\[[^]]+\]\s*", "", str(row["raw"] or "")).strip()
                    text = re.sub(r"\s+", " ", cleaned or str(row["raw"] or "").strip())[:150]
                    if text:
                        src = str(row["summary_source"] or "")
                        if not src and str(row["tier"] or "") in ("long", "important", "distilled"):
                            src = SUMMARY_SOURCE_HISTORICAL
                        self._conn.execute(
                            "UPDATE events SET summary=?, summary_source=? WHERE id=?",
                            (text, src, int(row["id"])),
                        )
                self._conn.commit()

        def _read_status(self) -> dict:
            return self._read_status_file(self._data_dir / "status.json")

        @staticmethod
        def _read_status_file(path: Path) -> dict:
            try:
                value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                return value if isinstance(value, dict) else {}
            except Exception:
                return {}

        def _mutate_status(
            self, change: Callable[[dict], dict], *, session_id: str = "",
            data_dir: Optional[Path] = None,
        ) -> None:
            if in_public_interaction():
                return
            path = (data_dir if data_dir is not None else self._data_dir) / "status.json"
            with _STATUS_WRITER, _FileLock(path.with_suffix(".lock")):
                atomic_json_write(path, change(self._read_status_file(path)))

        def _write_status_payload(self, payload: dict, base: Optional[dict] = None) -> None:
            def merge(latest: dict) -> dict:
                if base is None:
                    merged = dict(payload)
                    if "inner_state" in latest:
                        merged["inner_state"] = latest["inner_state"]
                    return merged
                for k, v in payload.items():
                    if k != "inner_state" and (k not in base or base.get(k) != v):
                        latest[k] = v
                for k in base.keys() - payload.keys():
                    if k != "inner_state":
                        latest.pop(k, None)
                return latest

            self._mutate_status(merge)


        def _add_chroma_site_packages(self) -> None:
            configured = self._config.get("chroma_site_packages")
            candidates = [Path(configured).expanduser()] if configured else []
            venv = self._data_dir / "venv"
            candidates.extend(
                [
                    venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
                    venv / "lib64" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
                ]
            )
            for candidate in candidates:
                if candidate.is_dir():
                    site.addsitedir(str(candidate))

        def _open_chroma(self) -> None:
            try:
                self._add_chroma_site_packages()
                import chromadb
    
                chroma_path = self._data_dir / "chroma"
                chroma_path.mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=str(chroma_path))
                self._collection = self._chroma_client.get_or_create_collection(
                    name=_COLLECTION, metadata={"hnsw:space": "cosine"}
                )
            except Exception as exc:
                self._collection = None
                self._chroma_client = None
                logger.warning("sanae_mem: Chroma unavailable; using SQLite vectors (%s)", type(exc).__name__)

        def _upsert_chroma(
            self, event_id: int, raw: str, topic: str, importance: float, ts: str, vector: List[float]
        ) -> None:
            if self._collection is None:
                return
            try:
                self._collection.upsert(
                    ids=[f"evt-{event_id}"],
                    documents=[raw],
                    embeddings=[vector],
                    metadatas=[{"event_id": event_id, "ts": ts, "topic": topic, "importance": importance}],
                )
            except Exception as exc:
                logger.warning("sanae_mem: Chroma write failed; SQLite retained event (%s)", type(exc).__name__)


        def _write_status(self, now: datetime, *, appraised: bool = False) -> None:
            date = now.date().isoformat()

            def record(payload: dict) -> dict:
                count = int(payload.get("daily_writes", 0)) if payload.get("date") == date else 0
                if payload.get("date") != date:
                    payload["daily_scheduled_asks"] = 0
                    payload["daily_compactor_runs"] = 0
                payload.setdefault("appraise_unconsumed_since", date)
                payload.setdefault("appraise_rows_written", 0)
                if appraised:
                    payload["appraise_rows_written"] = int(payload["appraise_rows_written"]) + 1
                payload.update({"date": date, "daily_writes": count + 1, "last_write_at": now.isoformat()})
                return payload

            self._mutate_status(record)

        def _write_sample(self, text: str) -> None:
            now = datetime.now().astimezone()
            date = now.date().isoformat()
            path = self._data_dir / "sample.log"
            count = 0
            if path.exists():
                try:
                    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(date))
                except Exception:
                    return
            if count < 3:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{date}\t{text[:80]}\n")

        def backup_paths(self) -> List[str]:
            return [str(self._data_dir / "sanae.db"), str(self._data_dir / "chroma"),
                    str(self._data_dir / "memes")]

        def shutdown(self) -> None:
            with self._lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
                self._collection = None
                self._chroma_client = None


_BACKUP_KEEP = 15


def _prune_sibling_backups(db_path, prefix: str, keep: int = _BACKUP_KEEP) -> None:
    try:
        import glob as _glob
        pattern = str(db_path.with_name(db_path.name + ".bak-" + prefix + "-*"))
        files = [p for p in _glob.glob(pattern) if p != str(db_path)]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for old in files[keep:]:
            try:
                os.remove(old)
            except OSError:
                pass
    except Exception:
        pass
