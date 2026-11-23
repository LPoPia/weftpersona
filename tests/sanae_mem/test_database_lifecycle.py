from __future__ import annotations

import base64
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem.provider import SanaeMemProvider
from sanae_mem.storage import database, legacy_upgrade, schema


@pytest.mark.parametrize("version", [2, 3])
def test_meme_upgrade_keeps_existing_video_schema(tmp_path, version):
    path = tmp_path / "sanae.db"
    with closing(database.open_database(path)) as conn, conn:
        conn.execute("ALTER TABLE video_watches ADD COLUMN sibling_task_video_field TEXT")
        video_schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='video_watches'"
        ).fetchone()[0]
        for column in ("tier", "visibility_scope", "evidence_status", "source_material_id"):
            conn.execute(f"ALTER TABLE meme_records DROP COLUMN {column}")
        conn.execute(
            "INSERT INTO meme_records(title,meaning,created_at,updated_at) "
            "VALUES ('legacy local entry','existing meaning','2026-01-01','2026-01-01')"
        )
        conn.execute(f"PRAGMA user_version={version}")

    for _ in range(2):
        with closing(database.open_database(path)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
            assert conn.execute("SELECT sql FROM sqlite_master WHERE name='video_watches'").fetchone()[0] == video_schema
            assert tuple(conn.execute(
                "SELECT title,tier,visibility_scope,evidence_status FROM meme_records"
            ).fetchone()) == ("legacy local entry", "common", "private", "uncertain")


def _legacy_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY, ts TEXT NOT NULL, raw TEXT NOT NULL,
          keywords TEXT DEFAULT '', importance REAL DEFAULT 0.5,
          confidence REAL DEFAULT 0.6, topic TEXT DEFAULT '', ttl_expire TEXT,
          asked INTEGER DEFAULT 0, blacklist INTEGER DEFAULT 0,
          created_at TEXT NOT NULL, updated_at TEXT, vec BLOB,
          session_id TEXT DEFAULT '', a_serious INTEGER DEFAULT 0,
          a_intent INTEGER DEFAULT 0, a_rel INTEGER DEFAULT 0, a_self INTEGER DEFAULT 0,
          a_conflict INTEGER DEFAULT 0, affect INTEGER DEFAULT 0
        );
        CREATE TABLE event_occurrences (
          id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL,
          raw TEXT NOT NULL, observed_at TEXT NOT NULL, source_snapshot BLOB,
          source_snapshot_sha256 TEXT
        );
        CREATE TABLE mood_state(channel TEXT PRIMARY KEY, cause_text TEXT);
        CREATE TABLE impression_revisions (
          profile_key TEXT NOT NULL, owner_key TEXT NOT NULL, revision INTEGER NOT NULL,
          text TEXT NOT NULL, speaker TEXT NOT NULL, evidence TEXT NOT NULL,
          turn_id TEXT NOT NULL, updated_at REAL NOT NULL, recorded_at REAL NOT NULL,
          PRIMARY KEY(profile_key, owner_key, revision)
        );
        CREATE TABLE impression_session_views (
          profile_key TEXT NOT NULL, owner_key TEXT NOT NULL,
          session_id TEXT NOT NULL, revision INTEGER NOT NULL,
          PRIMARY KEY(profile_key, owner_key, session_id)
        );
        CREATE TABLE commitments (
          id INTEGER PRIMARY KEY AUTOINCREMENT, owner_key TEXT NOT NULL,
          text TEXT NOT NULL, masked_text TEXT NOT NULL DEFAULT '',
          source_quote TEXT NOT NULL DEFAULT '', turn_id TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'pending', version INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL, fulfilled_at TEXT,
          cancelled_at TEXT, resolution_evidence TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE commitment_legacy_queue (
          source_kind TEXT NOT NULL, source_id TEXT NOT NULL, owner_key TEXT NOT NULL,
          source_text TEXT NOT NULL DEFAULT '', source_at REAL NOT NULL DEFAULT 0,
          status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
          last_error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL, PRIMARY KEY(source_kind, source_id)
        );
        CREATE TABLE inner_state_revisions (
          profile_key TEXT NOT NULL, revision INTEGER NOT NULL, text TEXT NOT NULL,
          recent_words TEXT NOT NULL, variables TEXT NOT NULL, felt_at REAL NOT NULL,
          updated_at REAL NOT NULL, processed_at REAL,
          source TEXT NOT NULL, thought_source TEXT NOT NULL, turn_id TEXT NOT NULL,
          changed INTEGER NOT NULL, owner_key TEXT NOT NULL DEFAULT '',
          PRIMARY KEY(profile_key, revision)
        );
        CREATE TABLE daily_context (
          entry_id INTEGER PRIMARY KEY AUTOINCREMENT, owner_key TEXT NOT NULL,
          profile_key TEXT NOT NULL, kind TEXT NOT NULL, slot TEXT NOT NULL,
          text TEXT NOT NULL, opened_on TEXT NOT NULL, expires_on TEXT,
          source_quote TEXT NOT NULL DEFAULT '', source_turn TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL, version INTEGER NOT NULL, created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE event_reviews (
          event_id INTEGER NOT NULL, review_sha256 TEXT NOT NULL, target TEXT NOT NULL,
          duty TEXT NOT NULL, content_sha256 TEXT NOT NULL, event_sha256 TEXT NOT NULL,
          event_version TEXT NOT NULL DEFAULT '', baseline_sha256 TEXT NOT NULL DEFAULT '',
          source_kind TEXT NOT NULL DEFAULT 'unknown', significance TEXT NOT NULL DEFAULT '',
          reviewed_at TEXT NOT NULL, PRIMARY KEY(event_id, review_sha256)
        );
        CREATE TABLE video_watches (
          id INTEGER PRIMARY KEY AUTOINCREMENT, bvid TEXT NOT NULL,
          title TEXT NOT NULL DEFAULT '', cid TEXT NOT NULL DEFAULT '', mode TEXT NOT NULL DEFAULT '',
          analysis TEXT NOT NULL DEFAULT '', captions TEXT NOT NULL DEFAULT '',
          summary TEXT NOT NULL DEFAULT '', source_context TEXT NOT NULL DEFAULT '',
          raw_observation TEXT NOT NULL DEFAULT '', public_excerpt TEXT NOT NULL DEFAULT '',
          session_id TEXT NOT NULL DEFAULT '', watched_at TEXT NOT NULL,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          UNIQUE(bvid)
        );
        """
    )
    conn.execute(
        "INSERT INTO events(ts,raw,keywords,topic,created_at,vec,a_serious,a_intent,a_rel,a_self,a_conflict,affect) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        ("t", "早苗在B站阅读了该视频的字幕：《SYNTHETIC-VIDEO-TITLE》（BV1234567890，CID 7）。SYNTHETIC-RAW-RETAINED",
         "video", "视频", "2026-01-01", sqlite3.Binary(b"\x00\xffvec"), 3, 2, 1, 3, 2, 3),
    )
    event_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO event_occurrences(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256) "
        "VALUES(?,?,?,?,?)",
        (event_id, "SYNTHETIC-OCCURRENCE-RAW", "2026-01-02", sqlite3.Binary(b"\x00\x01source"), "sha"),
    )
    conn.execute(
        "INSERT INTO impression_revisions VALUES('profile','owner',1,'SYNTHETIC-IMPRESSION','','','turn',1,1)"
    )
    conn.execute(
        "INSERT INTO inner_state_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("profile", 1, "SYNTHETIC-INNER-STATE-RECORD",
         "", "{}", 100, 100, None, "flash", "flash", "turn", 1, "owner"),
    )
    for index in range(5):
        conn.execute(
            "INSERT INTO impression_session_views VALUES('profile','owner',?,?)",
            (f"session-{index}", index + 1),
        )
    for index in range(10):
        status = "migrated" if index < 8 else "skipped"
        conn.execute(
            "INSERT INTO commitment_legacy_queue VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("event", str(index), "owner", "", 1, status, 0, "", "created", "updated"),
        )
    vec_blob = base64.b64encode(b"\x09\x08\x07").decode("ascii")
    conn.execute(
        "INSERT INTO video_watches(bvid,raw_observation,watched_at,created_at,updated_at) "
        "VALUES(?,?,?,?,?)",
        ("BV0987654321", json.dumps([{"event": {"id": 9, "vec": {"__sqlite_blob_base64__": vec_blob}}}]),
         "2026-01-03", "2026-01-03", "2026-01-03"),
    )
    conn.execute("PRAGMA user_version=0")
    conn.commit()
    conn.close()


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_provider_rebinds_database_and_backup_paths_to_current_home(tmp_path, monkeypatch):
    first_home = tmp_path / "first-home"
    second_home = tmp_path / "second-home"
    first_data = first_home / "sanae-mem"
    second_data = second_home / "sanae-mem"
    monkeypatch.setenv("HERMES_HOME", str(first_home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    provider = SanaeMemProvider({"semantic_backend": "sqlite"})
    sqlite = provider.sqlite
    try:
        assert provider.backup_paths() == [
            str(first_data / "sanae.db"), str(first_data / "chroma"), str(first_data / "memes"),
        ]
        assert not first_data.exists()
        assert sqlite.connection is None
        provider.initialize("first", hermes_home=str(first_home))
        old_connection = sqlite.connection
        assert old_connection is not None
        assert sqlite.db_path == first_data / "sanae.db"

        provider.initialize("second", hermes_home=str(second_home))
        assert provider.sqlite is sqlite
        assert sqlite.db_path == second_data / "sanae.db"
        assert (second_data / "sanae.db").is_file()
        assert sqlite.read(lambda conn: conn.execute("PRAGMA user_version").fetchone()[0]) == (
            database.CURRENT_SCHEMA_VERSION
        )
        assert provider.backup_paths() == [
            str(second_data / "sanae.db"), str(second_data / "chroma"), str(second_data / "memes"),
        ]
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            old_connection.execute("SELECT 1")
    finally:
        provider.shutdown()


def test_provider_migrates_external_database_with_archive_in_active_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient-home"))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    home = tmp_path / "active-home"
    data_dir = tmp_path / "external" / "data"
    db_path = data_dir / "sanae.db"
    _legacy_database(db_path)
    provider = SanaeMemProvider({"semantic_backend": "sqlite", "data_dir": str(data_dir)})
    try:
        provider.initialize("legacy", hermes_home=str(home))
        assert provider.sqlite.db_path == db_path
        assert provider.backup_paths() == [
            str(db_path), str(data_dir / "chroma"), str(data_dir / "memes"),
        ]
        archive = home / "memory-archive" / "events-archived.jsonl"
        assert "SYNTHETIC-RAW-RETAINED" in archive.read_text(encoding="utf-8")
        assert db_path.with_name("sanae.db.bak.v0.sqlite").is_file()
    finally:
        provider.shutdown()


def test_new_database_installs_only_current_schema_and_live_event_fields(tmp_path):
    path = tmp_path / "sanae.db"
    conn = database.open_database(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    tables = _tables(conn)
    assert not ({"mood_state", "impression_session_views", "commitment_legacy_queue"} & tables)
    event_columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    assert {"a_conflict", "affect"} <= event_columns
    assert not ({"a_serious", "a_intent", "a_rel", "a_self"} & event_columns)
    conn.execute("INSERT INTO events(ts,raw,keywords,created_at) VALUES('now','current text','current','now')")
    assert conn.execute("SELECT count(*) FROM events_fts WHERE events_fts MATCH 'current'").fetchone()[0] == 1
    conn.close()
    assert not list(tmp_path.glob("*.bak.*.sqlite"))


def test_schema_v1_backup_precedes_source_materials_and_video_v3_upgrade(tmp_path):
    path = tmp_path / "sanae.db"
    conn = database.open_database(path)
    conn.execute(
        "INSERT INTO events(ts,raw,keywords,created_at) VALUES('now','kept event','current','now')"
    )
    conn.execute("DROP TABLE source_materials")
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()

    upgraded = database.open_database(path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    assert upgraded.execute("SELECT raw FROM events").fetchone()[0] == "kept event"
    assert upgraded.execute("SELECT count(*) FROM source_materials").fetchone()[0] == 0
    indexes = {row[1] for row in upgraded.execute("PRAGMA index_list(source_materials)")}
    assert {"idx_source_materials_dedup", "idx_source_materials_lookup"} <= indexes
    upgraded.close()

    backup = path.with_name("sanae.db.bak.v1.sqlite")
    with sqlite3.connect(backup) as saved:
        assert saved.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert saved.execute("PRAGMA user_version").fetchone()[0] == 1
        assert saved.execute(
            "SELECT raw FROM events WHERE raw='kept event'"
        ).fetchone()[0] == "kept event"
        assert saved.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='source_materials'"
        ).fetchone()[0] == 0


def test_v2_to_v3_video_upgrade_does_not_rerun_legacy_event_repairs(tmp_path):
    path = tmp_path / "sanae.db"
    conn = database.open_database(path)
    conn.execute("INSERT INTO events(ts,raw,keywords,created_at) VALUES('now','kept event','current','now')")
    conn.execute("PRAGMA user_version=2")
    conn.commit()
    conn.close()

    upgraded = database.open_database(path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    assert upgraded.execute("SELECT summary FROM events WHERE raw='kept event'").fetchone()[0] == ""
    assert upgraded.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    upgraded.close()
    with sqlite3.connect(path.with_name("sanae.db.bak.v2.sqlite")) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 2


def test_v5_upgrade_clamps_over_cap_half_lives_with_backup_and_preserves_low_values(tmp_path):
    path = tmp_path / "sanae.db"
    conn = database.open_database(path)
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t1','long over cap','','k1','2026-09-11','long',4,140.2872671733932)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t2','short over cap','short summary','k2','2026-11-21','short',0,270.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t3','pending over cap','pending summary','k3','2026-11-22','pending',0,225.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t4','short normal','short normal','k4','2026-11-22','short',0,12.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t5','short explicit low','short low','k5','2026-11-22','short',0,3.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t6','long normal','long normal','k6','2026-11-22','long',3,60.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t7','null tier over cap','null summary','k7','2026-11-22',NULL,0,90.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t8','empty tier over cap','empty summary','k8','2026-11-22','',0,95.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t9','blank tier over cap','blank summary','k9','2026-11-22','   ',0,90.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t10','padded short over cap','padded summary','k10','2026-11-22',' short ',0,95.0)"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t11','vtab tier over cap','vtab summary','k11','2026-11-22',?,0,90.0)",
        ("\x0b",),
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t12','nbsp short over cap','nbsp summary','k12','2026-11-22',?,0,90.0)",
        ("\u00a0short\u00a0",),
    )
    conn.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t13','ideographic pending over cap','ideo summary','k13','2026-11-22',?,0,90.0)",
        ("\u3000pending\u3000",),
    )
    conn.execute("CREATE TABLE update_audit(ts TEXT NOT NULL)")
    conn.execute(
        "CREATE TRIGGER audit_half_life_updates AFTER UPDATE ON events BEGIN "
        "INSERT INTO update_audit(ts) VALUES (new.ts); END"
    )
    conn.execute("PRAGMA user_version=5")
    conn.commit()
    conn.close()

    upgraded = database.open_database(path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    assert upgraded.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    rows = dict(upgraded.execute("SELECT ts, half_life_days FROM events").fetchall())
    assert rows["t1"] == pytest.approx(120.0)
    assert rows["t2"] == pytest.approx(60.0)
    assert rows["t3"] == pytest.approx(60.0)
    assert rows["t4"] == pytest.approx(12.0)
    assert rows["t5"] == pytest.approx(3.0)
    assert rows["t6"] == pytest.approx(60.0)
    assert rows["t7"] == pytest.approx(60.0)
    assert rows["t8"] == pytest.approx(60.0)
    assert rows["t9"] == pytest.approx(60.0)
    assert rows["t10"] == pytest.approx(60.0)
    assert rows["t11"] == pytest.approx(60.0)
    assert rows["t12"] == pytest.approx(60.0)
    assert rows["t13"] == pytest.approx(60.0)
    assert upgraded.execute("SELECT summary FROM events WHERE ts='t1'").fetchone()[0] == ""
    audit_counts = upgraded.execute(
        "SELECT ts, COUNT(*) FROM update_audit GROUP BY ts ORDER BY ts"
    ).fetchall()
    assert [(row[0], row[1]) for row in audit_counts] == [
        ("t1", 1), ("t10", 1), ("t11", 1), ("t12", 1), ("t13", 1),
        ("t2", 1), ("t3", 1), ("t7", 1), ("t8", 1), ("t9", 1),
    ]
    upgraded.execute("DROP TRIGGER audit_half_life_updates")
    upgraded.execute("DROP TABLE update_audit")
    upgraded.commit()
    upgraded.close()

    backup = path.with_name("sanae.db.bak.v5.sqlite")
    with sqlite3.connect(backup) as saved:
        assert saved.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert saved.execute("PRAGMA user_version").fetchone()[0] == 5
        backup_rows = dict(saved.execute("SELECT ts, half_life_days FROM events").fetchall())
        assert backup_rows["t1"] == pytest.approx(140.2872671733932)
        assert backup_rows["t2"] == pytest.approx(270.0)
        assert backup_rows["t3"] == pytest.approx(225.0)
        assert backup_rows["t7"] == pytest.approx(90.0)
        assert backup_rows["t8"] == pytest.approx(95.0)
        assert backup_rows["t9"] == pytest.approx(90.0)
        assert backup_rows["t10"] == pytest.approx(95.0)
        assert backup_rows["t11"] == pytest.approx(90.0)
        assert backup_rows["t12"] == pytest.approx(90.0)
        assert backup_rows["t13"] == pytest.approx(90.0)
    backup_bytes = backup.read_bytes()

    reopened = database.open_database(path)
    assert reopened.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    reopened.close()
    assert backup.read_bytes() == backup_bytes


def test_empty_db_concurrent_v5_init_does_not_skip_v6_upgrade(tmp_path):
    import threading
    import time

    path = tmp_path / "sanae.db"
    # Pre-initialize an empty WAL file so readers can observe empty schema without blocking while
    # writer B holds an uncommitted BEGIN IMMEDIATE creation of v5 + 140.287 row.
    with closing(sqlite3.connect(path)) as init_conn:
        init_conn.execute("PRAGMA journal_mode=WAL")

    writer = sqlite3.connect(path, timeout=5.0)
    writer.execute("BEGIN IMMEDIATE")
    statement = ""
    for line in schema.CURRENT_SCHEMA_SQL.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                writer.execute(statement)
            statement = ""
    writer.execute(
        "INSERT INTO events(ts,raw,summary,keywords,created_at,tier,recall_count,half_life_days) "
        "VALUES('t1','race event','race summary','k1','2026-09-11','long',4,140.2872671733932)"
    )
    writer.execute("PRAGMA user_version=5")

    opened: list[sqlite3.Connection] = []
    errors: list[BaseException] = []
    started = threading.Event()

    def run_open() -> None:
        started.set()
        try:
            opened.append(database.open_database(path))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_open)
    thread.start()
    assert started.wait(timeout=2.0)
    # Give open_database time to pass pre-lock inspection and park on BEGIN IMMEDIATE.
    time.sleep(0.15)
    writer.commit()
    writer.close()
    thread.join(timeout=10.0)

    assert not errors, errors
    assert len(opened) == 1
    try:
        conn = opened[0]
        assert conn.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
        assert conn.execute("SELECT half_life_days FROM events WHERE ts='t1'").fetchone()[0] == pytest.approx(120.0)
    finally:
        opened[0].close()

    backup = path.with_name("sanae.db.bak.v5.sqlite")
    assert backup.is_file()
    with closing(sqlite3.connect(backup)) as saved:
        assert saved.execute("PRAGMA user_version").fetchone()[0] == 5
        assert saved.execute("SELECT half_life_days FROM events WHERE ts='t1'").fetchone()[0] == pytest.approx(
            140.2872671733932
        )


def test_v1_partial_curve_columns_preserve_existing_half_life(tmp_path):
    path = tmp_path / "sanae.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY, ts TEXT NOT NULL, raw TEXT NOT NULL,
          summary TEXT DEFAULT '', summary_source TEXT DEFAULT '',
          keywords TEXT DEFAULT '', importance REAL DEFAULT 0.5,
          confidence REAL DEFAULT 0.6, topic TEXT DEFAULT '', ttl_expire TEXT,
          asked INTEGER DEFAULT 0, blacklist INTEGER DEFAULT 0,
          created_at TEXT NOT NULL, updated_at TEXT, vec BLOB,
          session_id TEXT DEFAULT '', tier TEXT DEFAULT 'short',
          recall_count INTEGER NOT NULL DEFAULT 0,
          half_life_days REAL NOT NULL DEFAULT 14.0,
          curve_ttl TEXT NOT NULL DEFAULT ''
        );
        CREATE VIRTUAL TABLE events_fts USING fts5(raw, keywords, content='events', content_rowid='id');
        INSERT INTO events(ts,raw,summary,keywords,created_at,tier,half_life_days)
          VALUES('t1','partial v1','partial v1','k1','2026-09-01','short',3.0);
        PRAGMA user_version=1;
        """
    )
    conn.commit()
    conn.close()

    with closing(database.open_database(path, archive_dir=tmp_path / "memory-archive")) as upgraded:
        assert upgraded.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
        row = upgraded.execute("SELECT half_life_days, last_active_at FROM events WHERE ts='t1'").fetchone()
        assert row[0] == pytest.approx(3.0)
        assert row[1] == "2026-09-01"


def test_legacy_upgrade_archives_before_retirement_and_preserves_live_behavior(tmp_path, monkeypatch):
    path = tmp_path / "data" / "sanae.db"
    _legacy_database(path)
    (path.parent / "status.json").write_text(json.dumps({
        "unrelated": {"keep": True},
        "inner_state": {"failure_count": 2, "spark_at_by_profile": {
            "profile|owner": 2000.0, "profile": 9000.0, "missing|owner": 3000.0,
        }},
    }), encoding="utf-8")
    conn = database.open_database(path, archive_dir=tmp_path / "memory-archive")

    assert conn.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    tables = _tables(conn)
    assert not ({"mood_state", "impression_session_views", "commitment_legacy_queue"} & tables)
    event_columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    assert {"a_conflict", "affect"} <= event_columns
    assert not ({"a_serious", "a_intent", "a_rel", "a_self"} & event_columns)
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    video = conn.execute("SELECT * FROM video_watches WHERE bvid='BV1234567890'").fetchone()
    assert video is not None and video["owner_key"] == "" and video["read_scope"] == ""
    assert video["title"] == "SYNTHETIC-VIDEO-TITLE" and "SYNTHETIC-RAW-RETAINED" in video["raw_observation"]
    assert conn.execute("SELECT count(*) FROM video_fts WHERE video_fts MATCH 'SYNTHETIC'").fetchone()[0] == 1
    conn.execute(
        "INSERT INTO video_watches(bvid,owner_key,read_scope,watched_at,created_at,updated_at) "
        "VALUES('BV1234567890','another-owner','{\"kind\":\"self\"}',?,?,?)",
        ("2026-02-01", "2026-02-01", "2026-02-01"),
    )
    assert conn.execute(
        "SELECT count(*) FROM video_watches WHERE bvid='BV1234567890'"
    ).fetchone()[0] == 2
    assert tuple(conn.execute(
        "SELECT baseline,recent FROM impression_revisions WHERE profile_key='profile'"
    ).fetchone()) == (None, None)
    assert conn.execute(
        "SELECT last_spark_at FROM inner_state_revisions WHERE profile_key='profile' AND owner_key='owner'"
    ).fetchone()[0] == 2000.0
    conn.close()

    backup = path.with_name("sanae.db.bak.v0.sqlite")
    with sqlite3.connect(backup) as saved:
        assert saved.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert saved.execute("SELECT count(*) FROM mood_state").fetchone()[0] == 0
        assert saved.execute("SELECT count(*) FROM impression_session_views").fetchone()[0] == 5
        assert saved.execute("SELECT count(*) FROM commitment_legacy_queue").fetchone()[0] == 10
        saved_event = saved.execute("SELECT a_serious,a_intent,a_rel,a_self,vec FROM events").fetchone()
        assert saved_event[:4] == (3, 2, 1, 3) and saved_event[4] == b"\x00\xffvec"
        occurrence = saved.execute("SELECT raw,source_snapshot FROM event_occurrences").fetchone()
        assert occurrence == ("SYNTHETIC-OCCURRENCE-RAW", b"\x00\x01source")

    archive = (tmp_path / "memory-archive" / "events-archived.jsonl").read_text(encoding="utf-8")
    event_record = json.loads(archive.splitlines()[0])
    assert event_record["raw"] == "早苗在B站阅读了该视频的字幕：《SYNTHETIC-VIDEO-TITLE》（BV1234567890，CID 7）。SYNTHETIC-RAW-RETAINED"
    assert event_record["event"]["vec"] == {"__sqlite_blob_base64__": base64.b64encode(b"\x00\xffvec").decode("ascii")}
    assert event_record["occurrences"][0]["source_snapshot"] == {
        "__sqlite_blob_base64__": base64.b64encode(b"\x00\x01source").decode("ascii")
    }
    raw_archive = (tmp_path / "memory-archive" / "video-raw-observation-archived.jsonl").read_text(encoding="utf-8")
    archived_provenance = json.loads(raw_archive.splitlines()[0])["raw_observation"]
    assert "vec" in archived_provenance
    spark_archive = (tmp_path / "memory-archive" / "inner-state-spark-clock-archived.jsonl").read_text(encoding="utf-8")
    unknown_clock = json.loads(spark_archive.splitlines()[0])["spark_at_by_profile"]
    assert unknown_clock == {"profile": 9000.0, "missing|owner": 3000.0}
    old_status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
    assert old_status["unrelated"] == {"keep": True}
    assert old_status["inner_state"]["spark_at_by_profile"]["profile|owner"] == 2000.0
    def unexpected_upgrade(*_args, **_kwargs):
        raise AssertionError("completed migrations must not run again")

    monkeypatch.setattr(database, "_backup_before_upgrade", unexpected_upgrade)
    monkeypatch.setattr(legacy_upgrade, "upgrade_legacy_database", unexpected_upgrade)
    repeated = database.open_database(path, archive_dir=tmp_path / "memory-archive")
    assert repeated.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    assert repeated.execute("SELECT count(*) FROM video_watches WHERE bvid='BV1234567890'").fetchone()[0] == 1
    repeated.close()


def test_archive_failure_keeps_old_rows_and_does_not_advance_version(tmp_path, monkeypatch):
    path = tmp_path / "data" / "sanae.db"
    _legacy_database(path)
    original = legacy_upgrade._append_fsynced_jsonl

    def fail_archive(*_args, **_kwargs):
        raise OSError("fsync denied")

    monkeypatch.setattr(legacy_upgrade, "_append_fsynced_jsonl", fail_archive)
    with pytest.raises(RuntimeError, match="upgrade from version 0 failed") as error:
        database.open_database(path, archive_dir=tmp_path / "memory-archive")
    assert isinstance(error.value.__cause__, OSError)
    assert "fsync denied" in str(error.value.__cause__)
    with sqlite3.connect(path) as source:
        assert source.execute("PRAGMA user_version").fetchone()[0] == 0
        assert source.execute("SELECT count(*) FROM events").fetchone()[0] == 1
        assert source.execute("SELECT count(*) FROM event_occurrences").fetchone()[0] == 1
    backup = path.with_name("sanae.db.bak.v0.sqlite")
    assert backup.is_file()
    with sqlite3.connect(path) as old_program:
        old_program.execute(
            "INSERT INTO events(ts,raw,keywords,created_at) VALUES('after-failure','written by old version','old','later')"
        )
    monkeypatch.setattr(legacy_upgrade, "_append_fsynced_jsonl", original)
    retry = database.open_database(path, archive_dir=tmp_path / "memory-archive")
    assert retry.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    retry.close()
    with sqlite3.connect(backup) as refreshed_backup:
        assert refreshed_backup.execute(
            "SELECT count(*) FROM events WHERE raw='written by old version'"
        ).fetchone()[0] == 1


def test_sql_failure_after_durable_archive_keeps_source_and_retries(tmp_path):
    path = tmp_path / "data" / "sanae.db"
    _legacy_database(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TRIGGER block_video_upgrade_delete BEFORE DELETE ON events "
        "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="upgrade from version 0 failed") as error:
        database.open_database(path, archive_dir=tmp_path / "memory-archive")
    assert isinstance(error.value.__cause__, sqlite3.IntegrityError)
    with sqlite3.connect(path) as source:
        assert source.execute("PRAGMA user_version").fetchone()[0] == 0
        assert source.execute("SELECT count(*) FROM events").fetchone()[0] == 1
        assert source.execute("SELECT count(*) FROM event_occurrences").fetchone()[0] == 1
        assert source.execute("SELECT count(*) FROM commitment_legacy_queue").fetchone()[0] == 10
    archive_path = tmp_path / "memory-archive" / "events-archived.jsonl"
    assert archive_path.is_file() and "SYNTHETIC-VIDEO-TITLE" in archive_path.read_text(encoding="utf-8")
    assert len(archive_path.read_text(encoding="utf-8").splitlines()) == 1
    conn = sqlite3.connect(path)
    conn.execute("DROP TRIGGER block_video_upgrade_delete")
    conn.commit()
    conn.close()
    retry = database.open_database(path, archive_dir=tmp_path / "memory-archive")
    assert retry.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    assert retry.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    assert len(archive_path.read_text(encoding="utf-8").splitlines()) == 1
    retry.close()


def test_v0_video_archive_retry_handles_u2028_blob_new_occurrence_and_fsync(tmp_path, monkeypatch):
    import os

    # 1. U+2028 in raw + new occurrence added after SQL rollback must not duplicate unchanged entries
    #    and must archive the new occurrence on retry.
    path_u = tmp_path / "u2028" / "data" / "sanae.db"
    _legacy_database(path_u)
    with closing(sqlite3.connect(path_u)) as conn:
        conn.execute(
            "UPDATE events SET raw=? WHERE id=1",
            ("早苗在B站阅读了该视频的字幕：《SYNTHETIC-VIDEO-TITLE》（BV1234567890，CID 7）。第一段\u2028第二段",),
        )
        conn.execute(
            "CREATE TRIGGER block_u2028_delete BEFORE DELETE ON events "
            "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
        )
        conn.commit()
    with pytest.raises(RuntimeError, match="upgrade from version 0 failed"):
        database.open_database(path_u, archive_dir=tmp_path / "u2028" / "memory-archive")

    archive_u = tmp_path / "u2028" / "memory-archive" / "events-archived.jsonl"
    assert len([line for line in archive_u.read_text(encoding="utf-8").split("\n") if line.strip()]) == 1

    with closing(sqlite3.connect(path_u)) as conn:
        conn.execute("DROP TRIGGER block_u2028_delete")
        conn.commit()
    with closing(database.open_database(path_u, archive_dir=tmp_path / "u2028" / "memory-archive")) as retried:
        assert retried.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    lines_after_identical_retry = [
        line for line in archive_u.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
    assert len(lines_after_identical_retry) == 1

    # 1b. New occurrence appended after SQL failure must be archived on retry.
    path_occ = tmp_path / "occ" / "data" / "sanae.db"
    _legacy_database(path_occ)
    with closing(sqlite3.connect(path_occ)) as conn:
        conn.execute(
            "CREATE TRIGGER block_occ_delete BEFORE DELETE ON events "
            "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
        )
        conn.commit()
    with pytest.raises(RuntimeError, match="upgrade from version 0 failed"):
        database.open_database(path_occ, archive_dir=tmp_path / "occ" / "memory-archive")
    with closing(sqlite3.connect(path_occ)) as conn:
        conn.execute("DROP TRIGGER block_occ_delete")
        conn.execute(
            "INSERT INTO event_occurrences(event_id,raw,observed_at,source_snapshot,source_snapshot_sha256) "
            "VALUES(1, 'SYNTHETIC-OCCURRENCE-SECOND', '2026-01-03', ?, 'sha2')",
            (sqlite3.Binary(b"\x00\x02source"),),
        )
        conn.commit()
    with closing(database.open_database(path_occ, archive_dir=tmp_path / "occ" / "memory-archive")) as retried_occ:
        assert retried_occ.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
    occ_lines = [
        json.loads(line)
        for line in (tmp_path / "occ" / "memory-archive" / "events-archived.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    assert len(occ_lines) == 2
    assert len(occ_lines[-1]["occurrences"]) == 2
    assert occ_lines[-1]["occurrences"][1]["raw"] == "SYNTHETIC-OCCURRENCE-SECOND"

    # 2. BLOB raw in video event must encode and deduplicate without TypeError.
    path_blob = tmp_path / "blob" / "data" / "sanae.db"
    _legacy_database(path_blob)
    with closing(sqlite3.connect(path_blob)) as conn:
        conn.execute("ALTER TABLE events ADD COLUMN source_context TEXT NOT NULL DEFAULT ''")
        conn.execute(
            "UPDATE events SET raw=?, source_context=? WHERE id=1",
            (
                sqlite3.Binary(b"\x80\xffBLOB-RAW"),
                json.dumps({"platform": "bilibili", "kind": "self", "ref": "https://www.bilibili.com/video/BV1234567890/"}),
            ),
        )
        conn.commit()
    with closing(database.open_database(path_blob, archive_dir=tmp_path / "blob" / "memory-archive")) as upgraded_blob:
        assert upgraded_blob.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
        assert upgraded_blob.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    blob_entry = json.loads(
        (tmp_path / "blob" / "memory-archive" / "events-archived.jsonl").read_text(encoding="utf-8").split("\n")[0]
    )
    assert blob_entry["raw"] == {"__sqlite_blob_base64__": base64.b64encode(b"\x80\xffBLOB-RAW").decode("ascii")}

    # 3. os.fsync failure after writing archive file must still fsync on retry before deleting source rows.
    path_fs = tmp_path / "fsync" / "data" / "sanae.db"
    _legacy_database(path_fs)
    archive_fs = tmp_path / "fsync" / "memory-archive" / "events-archived.jsonl"
    real_fsync = os.fsync
    fail_next_archive_fsync = True
    archive_fsync_calls = 0

    def guarded_fsync(fd: int) -> None:
        nonlocal fail_next_archive_fsync, archive_fsync_calls
        if archive_fs.is_file() and archive_fs.stat().st_size > 0:
            archive_fsync_calls += 1
            if fail_next_archive_fsync:
                fail_next_archive_fsync = False
                raise OSError("simulated fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", guarded_fsync)
    with pytest.raises(RuntimeError, match="upgrade from version 0 failed") as err:
        database.open_database(path_fs, archive_dir=tmp_path / "fsync" / "memory-archive")
    assert isinstance(err.value.__cause__, OSError)
    assert archive_fs.is_file() and archive_fs.stat().st_size > 0
    with closing(sqlite3.connect(path_fs)) as source:
        assert source.execute("PRAGMA user_version").fetchone()[0] == 0
        assert source.execute("SELECT count(*) FROM events").fetchone()[0] == 1

    calls_before_retry = archive_fsync_calls
    with closing(database.open_database(path_fs, archive_dir=tmp_path / "fsync" / "memory-archive")) as retried_fs:
        assert retried_fs.execute("PRAGMA user_version").fetchone()[0] == database.CURRENT_SCHEMA_VERSION
        assert retried_fs.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    assert archive_fsync_calls > calls_before_retry
    assert len([line for line in archive_fs.read_text(encoding="utf-8").split("\n") if line.strip()]) == 1
