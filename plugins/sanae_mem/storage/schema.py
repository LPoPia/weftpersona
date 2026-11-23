from __future__ import annotations

import json
import re
import sqlite3

CURRENT_SCHEMA_VERSION = 6

CURRENT_SCHEMA_SQL = r"""
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
                  a_conflict INTEGER NOT NULL DEFAULT 0,
                  affect INTEGER NOT NULL DEFAULT 0,
                  owner_key TEXT NOT NULL DEFAULT '',
                  source_snapshot TEXT NOT NULL DEFAULT '',
                  source_snapshot_sha256 TEXT NOT NULL DEFAULT '',
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
                  session_id TEXT NOT NULL DEFAULT '',
                  tier TEXT NOT NULL DEFAULT 'common',
                  visibility_scope TEXT NOT NULL DEFAULT 'private',
                  evidence_status TEXT NOT NULL DEFAULT 'uncertain',
                  source_material_id INTEGER
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
                  last_spark_at REAL NOT NULL DEFAULT 0.0,
                  -- Author of text/recent_words. Empty = pre-owner legacy row; it keeps
                  -- its verbatim text and slow variables but is never claimed or shown.
                  owner_key TEXT NOT NULL DEFAULT '',
                  -- 'private' rows belong to one verified owner. 'internal' rows are the
                  -- trusted autonomous heart, unbound to any person: never rendered into
                  -- a conversation, world block, shared projection or journal.
                  visibility TEXT NOT NULL DEFAULT 'private',
                  PRIMARY KEY(profile_key, revision)
                );
                -- Host-only common view for people who are not the author of the
                -- current private revision. Written only by the independent shared
                -- call whose payload has no private inputs.
                CREATE TABLE IF NOT EXISTS inner_state_shared (
                  profile_key TEXT PRIMARY KEY CHECK(profile_key <> ''),
                  text TEXT NOT NULL,
                  variables TEXT NOT NULL,
                  felt_at REAL NOT NULL,
                  updated_at REAL NOT NULL,
                  source_revision INTEGER NOT NULL DEFAULT 0
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

CREATE TABLE IF NOT EXISTS impression_revisions (
    profile_key TEXT NOT NULL,
    owner_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT NOT NULL,
    evidence TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    updated_at REAL NOT NULL,
    recorded_at REAL NOT NULL,
    baseline TEXT,
    recent TEXT,
    -- v5: the same canonical provenance JSON as events.source_context, so a group-scoped
    -- impression keeps its real speaker, chat/thread and message. Empty = old private row
    -- whose provenance was never recorded (unknown, never backfilled).
    source_context TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(profile_key, owner_key, revision)
);

CREATE TABLE IF NOT EXISTS commitments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  owner_key TEXT NOT NULL CHECK(owner_key <> ''),
  text TEXT NOT NULL CHECK(text <> ''),
  masked_text TEXT NOT NULL DEFAULT '',
  source_quote TEXT NOT NULL DEFAULT '',
  turn_id TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK(status IN ('pending', 'fulfilled', 'cancelled')) DEFAULT 'pending',
  version INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  fulfilled_at TEXT,
  cancelled_at TEXT,
  resolution_evidence TEXT NOT NULL DEFAULT '',
  -- v5: canonical provenance JSON of the turn that made the promise. Empty = old private row.
  source_context TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_commitments_owner_status
  ON commitments(owner_key, status);
CREATE INDEX IF NOT EXISTS idx_commitments_quote
  ON commitments(owner_key, source_quote);

CREATE TABLE IF NOT EXISTS video_watches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  bvid TEXT NOT NULL,
  platform TEXT NOT NULL DEFAULT 'bilibili',
  source_object_id TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  cid TEXT NOT NULL DEFAULT '',
  mode TEXT NOT NULL DEFAULT '',
  analysis TEXT NOT NULL DEFAULT '',
  captions TEXT NOT NULL DEFAULT '',
  summary TEXT NOT NULL DEFAULT '',
  source_context TEXT NOT NULL DEFAULT '',
  raw_observation TEXT NOT NULL DEFAULT '',
  read_scope TEXT NOT NULL DEFAULT '',
  public_excerpt TEXT NOT NULL DEFAULT '',
  owner_key TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  watched_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS video_fts
  USING fts5(bvid, title, summary, analysis, content='video_watches', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS video_ai AFTER INSERT ON video_watches BEGIN
  INSERT INTO video_fts(rowid, bvid, title, summary, analysis)
    VALUES (new.id, new.bvid, new.title, new.summary, new.analysis);
END;
CREATE TRIGGER IF NOT EXISTS video_ad AFTER DELETE ON video_watches BEGIN
  INSERT INTO video_fts(video_fts, rowid, bvid, title, summary, analysis)
    VALUES ('delete', old.id, old.bvid, old.title, old.summary, old.analysis);
END;
CREATE TRIGGER IF NOT EXISTS video_au AFTER UPDATE ON video_watches BEGIN
  INSERT INTO video_fts(video_fts, rowid, bvid, title, summary, analysis)
    VALUES ('delete', old.id, old.bvid, old.title, old.summary, old.analysis);
  INSERT INTO video_fts(rowid, bvid, title, summary, analysis)
    VALUES (new.id, new.bvid, new.title, new.summary, new.analysis);
END;

CREATE TABLE IF NOT EXISTS memory_axis_mentions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  owner_key TEXT NOT NULL CHECK(owner_key <> ''),
                  turn_id TEXT NOT NULL CHECK(turn_id <> ''),
                  event_id INTEGER,
                  channel TEXT NOT NULL CHECK(channel IN
                    ('delivered_remention','inner_state_flash','explicit_save')),
                  sleep_day TEXT NOT NULL,
                  observed_at TEXT NOT NULL,
                  delivered_at TEXT NOT NULL DEFAULT '',
                  user_quote TEXT NOT NULL,
                  assistant_quote TEXT NOT NULL,
                  user_sha256 TEXT NOT NULL,
                  assistant_sha256 TEXT NOT NULL,
                  stage TEXT NOT NULL CHECK(stage IN ('instant','medium','long')),
                  evidence_json TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  UNIQUE(owner_key, turn_id, channel)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_axis_one_event_day
                  ON memory_axis_mentions(owner_key, event_id, channel, sleep_day)
                  WHERE event_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_memory_axis_owner_day
                  ON memory_axis_mentions(owner_key, sleep_day, id);
                CREATE TABLE IF NOT EXISTS memory_axis_promotions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  owner_key TEXT NOT NULL CHECK(owner_key <> ''),
                  event_id INTEGER NOT NULL,
                  channel TEXT NOT NULL CHECK(channel IN
                    ('delivered_remention','inner_state_flash','explicit_save')),
                  mention_id INTEGER,
                  previous_tier TEXT NOT NULL DEFAULT '',
                  promoted_at TEXT NOT NULL,
                  UNIQUE(owner_key, event_id, channel)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_axis_promotions_event
                  ON memory_axis_promotions(owner_key, event_id);
                CREATE TABLE IF NOT EXISTS memory_axis_watermarks (
                  owner_key TEXT NOT NULL CHECK(owner_key <> ''),
                  channel TEXT NOT NULL CHECK(channel IN
                    ('delivered_remention','inner_state_flash','explicit_save')),
                  last_delivered_at TEXT NOT NULL DEFAULT '',
                  last_turn_id TEXT NOT NULL DEFAULT '',
                  last_sleep_day TEXT NOT NULL DEFAULT '',
                  last_error TEXT NOT NULL DEFAULT '',
                  failed_count INTEGER NOT NULL DEFAULT 0,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(owner_key, channel)
                );
                CREATE TABLE IF NOT EXISTS nightly_journal_runs (
                  owner_key TEXT NOT NULL CHECK(owner_key <> ''),
                  day TEXT NOT NULL CHECK(day <> ''),
                  status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
                  entry_sha256 TEXT NOT NULL DEFAULT '',
                  attempts INTEGER NOT NULL DEFAULT 0,
                  last_error TEXT NOT NULL DEFAULT '',
                  completed_at TEXT NOT NULL DEFAULT '',
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(owner_key, day)
                );
                CREATE TABLE IF NOT EXISTS source_materials (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  owner_key TEXT NOT NULL DEFAULT '',
                  session_id TEXT NOT NULL DEFAULT '',
                  source_plugin TEXT NOT NULL,
                  platform TEXT NOT NULL,
                  source_object_id TEXT NOT NULL DEFAULT '',
                  material_kind TEXT NOT NULL CHECK(material_kind IN ('raw', 'processed')),
                  author TEXT,
                  occurred_at TEXT,
                  collected_at TEXT,
                  privacy_scope TEXT NOT NULL CHECK(privacy_scope IN ('private', 'restricted', 'public')),
                  processing_type TEXT,
                  processing_version TEXT,
                  raw_locator TEXT,
                  credibility TEXT NOT NULL CHECK(credibility IN ('verified_raw', 'derived_processed', 'unverified_external')),
                  user_md_eligible INTEGER NOT NULL DEFAULT 0 CHECK(user_md_eligible = 0),
                  verbatim_eligible INTEGER NOT NULL DEFAULT 0 CHECK(verbatim_eligible IN (0, 1)),
                  title TEXT NOT NULL DEFAULT '',
                  content TEXT NOT NULL,
                  content_sha256 TEXT NOT NULL,
                  extra_metadata TEXT NOT NULL DEFAULT '{}',
                  linked_event_id INTEGER,
                  ingested_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_source_materials_dedup
                  ON source_materials(
                    owner_key,
                    source_plugin,
                    platform,
                    source_object_id,
                    material_kind,
                    COALESCE(processing_type, ''),
                    content_sha256
                  );
                CREATE INDEX IF NOT EXISTS idx_source_materials_lookup
                  ON source_materials(owner_key, platform, source_object_id, material_kind);
"""

_VIDEO_TABLE_START = "CREATE TABLE IF NOT EXISTS video_watches ("
_VIDEO_AUX_START = "CREATE VIRTUAL TABLE IF NOT EXISTS video_fts"
_AXIS_TABLE_START = "CREATE TABLE IF NOT EXISTS memory_axis_mentions"
_video_table_start = CURRENT_SCHEMA_SQL.index(_VIDEO_TABLE_START)
_video_aux_start = CURRENT_SCHEMA_SQL.index(_VIDEO_AUX_START, _video_table_start)
_axis_table_start = CURRENT_SCHEMA_SQL.index(_AXIS_TABLE_START, _video_aux_start)
VIDEO_WATCHES_TABLE_SQL = CURRENT_SCHEMA_SQL[_video_table_start:_video_aux_start].strip()
VIDEO_WATCHES_AUX_SQL = CURRENT_SCHEMA_SQL[_video_aux_start:_axis_table_start].strip()


RECEIPT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS free_think_reflections (
    owner_key TEXT NOT NULL CHECK(owner_key <> ''),
    execution_id TEXT NOT NULL CHECK(execution_id <> ''),
    job_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    finished_at REAL NOT NULL,
    reflection TEXT NOT NULL CHECK(reflection <> ''),
    created_at TEXT NOT NULL,
    PRIMARY KEY(owner_key, execution_id)
);
CREATE INDEX IF NOT EXISTS idx_free_think_reflections_owner_time
    ON free_think_reflections(owner_key, finished_at DESC);
CREATE TABLE IF NOT EXISTS session_reply_watermarks (
    owner_key TEXT NOT NULL CHECK(owner_key <> ''),
    session_id TEXT NOT NULL CHECK(session_id <> ''),
    platform TEXT NOT NULL DEFAULT '',
    delivered_at REAL NOT NULL,
    turn_id TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(owner_key, session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_reply_watermarks_owner_plat
    ON session_reply_watermarks(owner_key, platform, delivered_at DESC);
"""

CURRENT_SCHEMA_SQL += RECEIPT_SCHEMA_SQL

# v4：维护者账号绑定（一人多平台账号的显式确认映射）。
# 一行＝「某平台范围里的某个账号」已确认属于 `person_owner_key` 这个既有归属锚。
# 只由 `memory_identity` 锚所在的人写入；平台账号在一个范围内只能属于一个人（主键即冲突面）。
MAINTAINER_BINDING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS maintainer_account_bindings (
    -- Host profile is part of the account identity; an absent profile is normalized to 'default'.
    profile TEXT NOT NULL DEFAULT 'default' CHECK(profile <> ''),
    platform TEXT NOT NULL CHECK(platform <> ''),
    account_id TEXT NOT NULL CHECK(account_id <> ''),
    -- Slack workspace / QQ Bot app ID. Empty only for Discord, whose user snowflake is global
    -- (the conversation guild is not part of the user's identity).

    scope_id TEXT NOT NULL DEFAULT '',
    person_owner_key TEXT NOT NULL CHECK(person_owner_key <> ''),
    created_at TEXT NOT NULL,
    PRIMARY KEY(profile, platform, account_id, scope_id)
);
CREATE INDEX IF NOT EXISTS idx_maintainer_bindings_person
    ON maintainer_account_bindings(person_owner_key, profile, platform);
"""

CURRENT_SCHEMA_SQL += MAINTAINER_BINDING_SCHEMA_SQL


def _create_video_indexes(
    conn: sqlite3.Connection, *, ensure_scope_unique: bool = False,
) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_video_watches_bvid ON video_watches(bvid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_video_watches_identity ON video_watches(platform, source_object_id, cid, owner_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_video_watches_watched ON video_watches(watched_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_video_watches_owner ON video_watches(owner_key)")
    if ensure_scope_unique:
        for row in conn.execute("PRAGMA index_list(video_watches)"):
            if not int(row[2]):
                continue
            index = str(row[1]).replace('"', '""')
            columns = [
                str(info[2]) for info in conn.execute(f'PRAGMA index_info("{index}")')
            ]
            if columns == ["platform", "source_object_id", "cid", "owner_key", "read_scope"]:
                break
        else:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_video_watches_scope "
                "ON video_watches(platform, source_object_id, cid, owner_key, read_scope) "
                "WHERE source_object_id <> ''"
            )


def _mark_ambiguous_legacy_video_rows(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, mode, cid, analysis, raw_observation, summary FROM video_watches").fetchall()
    for row in rows:
        row_id, row_mode, row_cid, analysis, raw_value, summary_value = row
        raw_text = str(raw_value or "")
        try:
            raw = json.loads(raw_text) if raw_text else {}
        except (TypeError, ValueError):
            raw = {}
        if isinstance(raw, dict) and raw.get("_video_version_ambiguous") is True:
            continue
        observed_cid = str(raw.get("cid") or "").strip() if isinstance(raw, dict) else ""
        raw_analysis = str(raw.get("analysis") or "") if isinstance(raw, dict) else ""
        summary_cids = [match.strip() for match in re.findall(r"\bCID\s+([^).，\s]+)", str(summary_value or ""))
                        if match.strip().lower() not in {"未知", "unknown"}]
        modes = set(str(row_mode or "").split("+"))
        # Pre-v3 rows only kept the most recently observed CID but merged analysis
        # and captions without a version boundary. Do not offer that blend for reuse.
        known_cids = set(summary_cids)
        if observed_cid:
            known_cids.add(observed_cid)
        row_cid = str(row_cid or "").strip()
        stored_segments = {part for part in str(analysis or "").split("\n\n") if part.strip()}
        observed_segments = {part for part in raw_analysis.split("\n\n") if part.strip()}
        unversioned_analysis_mismatch = (
            not known_cids and "video_analysis" in modes
            and stored_segments != observed_segments
        )
        if ((row_cid and any(value != row_cid for value in known_cids))
                or (not known_cids and len(modes) > 1)
                or unversioned_analysis_mismatch):
            wrapped: dict[str, object] = (
                raw if isinstance(raw, dict) else {"legacy_raw_observation": raw_text}
            )
            wrapped["_video_version_ambiguous"] = True
            conn.execute("UPDATE video_watches SET raw_observation=? WHERE id=?",
                         (json.dumps(wrapped, ensure_ascii=False), int(row_id)))
