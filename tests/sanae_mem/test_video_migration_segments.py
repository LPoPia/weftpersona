"""Synthetic checks for dedicated video records and FTS fallback."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("VIVEMIND_HERMES_SOURCE"):
    HERMES_ROOT = Path(os.environ["VIVEMIND_HERMES_SOURCE"]).expanduser().resolve()
else:
    _hermes_spec = importlib.util.find_spec("hermes_constants")
    if _hermes_spec is not None and _hermes_spec.origin:
        HERMES_ROOT = Path(_hermes_spec.origin).resolve().parent
    else:
        HERMES_ROOT = next(
            (path for path in (
                REPO_ROOT.parent / "hermes-minimal-host",
                REPO_ROOT.parent / "hermes-minimal-host-20261112",
            ) if path.is_dir()),
            Path(tempfile.gettempdir()) / "vivemind-hermes-6093627",
        )
DEPENDENCIES = HERMES_ROOT.with_name(HERMES_ROOT.name + "-deps")
PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
if not HERMES_ROOT.is_dir():
    pytest.skip("pinned Hermes source is unavailable", allow_module_level=True)
for import_root in (DEPENDENCIES, HERMES_ROOT, PLUGIN_ROOT):
    if import_root.is_dir():
        sys.path.insert(0, str(import_root))

from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from hermes_cli.plugins import PluginManager
from tools.registry import registry

from sanae_mem.plugin_entry import SanaeMemProvider
import sanae_mem.domains.video_memory as video_memory_module
from sanae_mem.integrations import register_video_memory
from test_integrations import VIDEO_TOOL, _load_bilibili_addon


@contextmanager
def _private_route():
    reset_session_vars()
    tokens = set_session_vars(
        platform="bilibili", chat_type="dm", chat_id="fixture-user",
        user_id="fixture-user", session_id="fixture-video-session",
        session_key="agent:main:bilibili:dm:fixture-user",
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    instance = SanaeMemProvider({
        "data_dir": str(tmp_path / "sanae-mem"),
        "semantic_backend": "sqlite",
    })
    with _private_route():
        instance.initialize(
            "fixture-video-session", hermes_home=str(home),
            platform="bilibili", agent_context="primary",
        )
    yield instance
    instance.shutdown()


@pytest.fixture
def addon(provider, tmp_path):
    manager = PluginManager(scope_key=provider._hermes_home)
    try:
        module = _load_bilibili_addon(manager, tmp_path)
        module.providers[provider._session_id] = provider
        yield module.bind_memory(provider), manager.scope_key
    finally:
        manager.unload()


def _stamp() -> str:
    return datetime.now().astimezone().isoformat()


def _insert_video_row(provider, bvid: str, title: str, summary: str, owner_key: str) -> int:
    stamp = _stamp()
    with provider._lock:
        cursor = provider._conn.execute(
            """INSERT INTO video_watches
               (bvid, title, mode, analysis, captions, summary, read_scope,
                owner_key, watched_at, created_at, updated_at)
               VALUES (?, ?, 'video_analysis', '', '', ?, '{"kind":"self"}', ?, ?, ?, ?)""",
            (bvid, title, summary, owner_key, stamp, stamp, stamp),
        )
        provider._conn.commit()
    return int(cursor.lastrowid)


def test_record_merges_analysis_and_captions_for_same_bv(provider, addon):
    client, scope_key = addon
    bvid = "BV1RecordMx1"
    with _private_route():
        assert client.record_video_watch({
            "status": "ok", "mode": "video_analysis", "bvid": bvid,
            "title": "SYNTHETIC-VIDEO-TITLE", "analysis": "SYNTHETIC-VIDEO-ANALYSIS",
        }, delivered=True, public=True, _watch_receipt="0123456789abcdef0123456789abcdef")
        assert client.record_video_watch({
            "status": "ok", "mode": "captions_only", "bvid": bvid,
            "title": "SYNTHETIC-VIDEO-TITLE",
            "lines": [{"text": "SYNTHETIC-CAPTION-ALPHA"}, {"text": "SYNTHETIC-CAPTION-BETA"}],
        }, delivered=True, public=True, _watch_receipt="1123456789abcdef0123456789abcdef")

        rows = provider._conn.execute(
            "SELECT * FROM video_watches WHERE bvid=?", (bvid,)
        ).fetchall()
        assert len(rows) == 1
        detail = json.loads(registry.dispatch(VIDEO_TOOL, {"bvid": bvid}, scope=scope_key))["result"]
        assert detail["found"] is True
        assert "SYNTHETIC-VIDEO-ANALYSIS" in detail["analysis"]
        texts = [item.get("text") for item in detail["captions"]]
        assert "SYNTHETIC-CAPTION-ALPHA" in texts
        assert "SYNTHETIC-CAPTION-BETA" in texts


def test_platform_and_cid_are_independent_video_identities_and_lookup_is_version_exact(provider, addon):
    client, _scope_key = addon
    with _private_route():
        assert client.record_video_watch({
            "status": "ok", "mode": "video_analysis", "platform": "bilibili",
            "source_object_id": "BV1RecordMx1", "cid": "cid-old",
            "analysis": "SYNTHETIC-OLD-CID-ANALYSIS", "title": "same title",
        }, delivered=True, public=True, _watch_receipt="2123456789abcdef0123456789abcdef")
        assert client.record_video_watch({
            "status": "ok", "mode": "captions_only", "platform": "bilibili",
            "source_object_id": "BV1RecordMx1", "cid": "cid-new",
            "lines": [{"text": "SYNTHETIC-NEW-CID-CAPTION"}], "title": "same title",
        }, delivered=True, public=True, _watch_receipt="3123456789abcdef0123456789abcdef")
        simulated_source = SimpleNamespace(plugin_id="douyin_source", has_plugin=lambda name: True)
        source = register_video_memory(simulated_source).for_session("fixture-video-session")
        assert source.record_video_watch({
            "status": "ok", "mode": "video_analysis",
            "analysis": "SYNTHETIC-DOUYIN-ANALYSIS", "title": "same title",
        }, platform="douyin", source_object_id="BV1RecordMx1", cid="cid-old",
            settled=True, _watch_receipt="4123456789abcdef0123456789abcdef")

        broad = client.lookup_video_watch("bilibili", "BV1RecordMx1")
        assert set(broad) == {"record_id", "watched_at", "summary", "title"}
        assert client.lookup_video_watch("bilibili", "BV1RecordMx1", cid="cid-new") is None
        old = client.lookup_video_watch("bilibili", "BV1RecordMx1", cid="cid-old")
        assert old["analysis"] == "SYNTHETIC-OLD-CID-ANALYSIS"
        assert old["captions"] == []
        douyin = source.lookup_video_watch("douyin", "BV1RecordMx1", cid="cid-old")
        assert douyin["analysis"] == "SYNTHETIC-DOUYIN-ANALYSIS"
        assert provider._conn.execute(
            "SELECT count(*) FROM video_watches WHERE source_object_id='BV1RecordMx1'"
        ).fetchone()[0] == 3


def test_multisegment_video_replay_does_not_duplicate_existing_blocks(provider, addon):
    client, _scope_key = addon
    analysis = "SCENE-A: a kettle turns on.\n\nSCENE-B: steam rises."
    with _private_route():
        for receipt, body in (("0123456789abcdef0123456789abc001", analysis),
                              ("0123456789abcdef0123456789abc002", "SCENE-C: timer rings."),
                              ("0123456789abcdef0123456789abc003", analysis)):
            assert client.record_video_watch({
                "status": "ok", "mode": "video_analysis", "platform": "bilibili",
                "source_object_id": "BV1RecordMx1", "cid": "cid-015",
                "analysis": body, "title": "multisegment fixture",
            }, delivered=True, public=True, _watch_receipt=receipt)
        cached = client.lookup_video_watch("bilibili", "BV1RecordMx1", cid="cid-015")
        assert client.record_video_watch({
            "status": "ok", "mode": "video_analysis", "bvid": "BV1RecordMx1", "cid": "cid-015",
            "analysis": cached["analysis"], "title": "multisegment fixture",
            "reused_video_watch": {"record_id": cached["record_id"]},
        }, delivered=True, public=True, _watch_receipt="0123456789abcdef0123456789abc004")
        saved = provider._conn.execute(
            "SELECT analysis, summary FROM video_watches WHERE source_object_id='BV1RecordMx1' AND cid='cid-015'"
        ).fetchone()
        assert saved["analysis"] == analysis + "\n\nSCENE-C: timer rings."
        assert saved["summary"].count("SCENE-A: a kettle turns on.") == 1
        assert saved["summary"].count("SCENE-B: steam rises.") == 1
        assert saved["summary"].count("SCENE-C: timer rings.") == 1


def test_replaying_receipt_is_idempotent_across_later_delivery(tmp_path, monkeypatch):
    import sqlite3
    from contextlib import closing

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    db_path = tmp_path / "sanae-mem" / "sanae.db"
    local_provider = SanaeMemProvider({
        "data_dir": str(tmp_path / "sanae-mem"),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "bilibili", "user_id": "fixture-user"},
    })
    stamps = iter((
        "2026-01-01T00:00:01+09:00",
        "2026-01-01T00:00:02+09:00",
        "2026-01-01T00:00:03+09:00",
        "2026-01-01T00:00:04+09:00",
    ))

    class FixedClock:
        @classmethod
        def now(cls):
            return datetime.fromisoformat(next(stamps))

    monkeypatch.setattr(video_memory_module, "datetime", FixedClock)
    llm_calls = []

    class _FakeCuriosityLlm:
        def complete(self, messages, **kwargs):
            idx = len(llm_calls) + 1
            llm_calls.append(messages)
            user_prompt = str(messages[-1]["content"])
            if "seed-kettle" in user_prompt:
                text, quote = "好奇 kettle 怎么烧水的", "kettle boiling process"
            elif "seed-second" in user_prompt:
                text, quote = "好奇 second 怎么处理的", "second video details"
            elif "看了 second actual observation" in user_prompt:
                text, quote = f"好奇视频后续细节 {idx}", "看了 second actual observation"
            else:
                text, quote = f"好奇视频后续细节 {idx}", "看了 same kettle observation"
            return SimpleNamespace(text=json.dumps(
                {"items": [{"kind": "OPEN_THREAD", "action": "set", "text": text, "quote": quote}]},
                ensure_ascii=False,
            ))

    manager = PluginManager(scope_key=str(home))
    try:
        with _private_route():
            local_provider.initialize(
                "fixture-video-session", hermes_home=str(home),
                platform="bilibili", agent_context="primary",
            )
            local_provider.attach_runtime_llm(_FakeCuriosityLlm())
        addon_dir = tmp_path / "addon"
        addon_dir.mkdir()
        module = _load_bilibili_addon(manager, addon_dir)
        client = module.bind_memory(local_provider)

        def deliver(receipt, text):
            return client.record_video_watch({
                "status": "ok", "mode": "video_analysis", "bvid": "BV1RecordMx1",
                "title": "receipt fixture", "analysis": text,
            }, delivered=True, public=True, _watch_receipt=receipt)

        def watched_at_db():
            with closing(sqlite3.connect(db_path)) as db:
                return db.execute(
                    "SELECT watched_at FROM video_watches WHERE bvid='BV1RecordMx1'"
                ).fetchone()[0]

        def thread_statuses():
            with closing(sqlite3.connect(db_path)) as db:
                return {
                    row[0]: row[1]
                    for row in db.execute("SELECT text, status FROM daily_context WHERE kind='OPEN_THREAD'")
                }

        with _private_route():
            local_provider.daily_context.record_video_curiosity("seed-kettle", "kettle boiling process")
            assert thread_statuses() == {"好奇 kettle 怎么烧水的": "active"}
            receipt_a = "7123456789abcdef0123456789abcdef"
            receipt_b = "8123456789abcdef0123456789abcdef"
            assert deliver(receipt_a, "看了 same kettle observation")
            first = watched_at_db()
            statuses_after_a = thread_statuses()
            assert statuses_after_a["好奇 kettle 怎么烧水的"] == "closed"
            assert len(statuses_after_a) == 2 and sum(v == "active" for v in statuses_after_a.values()) == 1
            assert len(llm_calls) == 2

            assert deliver(receipt_a, "看了 same kettle observation")
            assert watched_at_db() == first
            assert thread_statuses() == statuses_after_a
            assert len(llm_calls) == 2

            local_provider.daily_context.record_video_curiosity("seed-second", "second video details")
            assert thread_statuses()["好奇 second 怎么处理的"] == "active"
            assert deliver(receipt_b, "看了 second actual observation")
            second = watched_at_db()
            assert datetime.fromisoformat(second) == datetime.fromisoformat("2026-01-01T00:00:03+09:00")
            assert datetime.fromisoformat(second) > datetime.fromisoformat(first)
            statuses_after_b = thread_statuses()
            assert statuses_after_b["好奇 second 怎么处理的"] == "closed"
            assert statuses_after_b.get("好奇视频后续细节 4") == "active"
            assert len(statuses_after_b) == 4 and sum(v == "active" for v in statuses_after_b.values()) == 1
            assert len(llm_calls) == 4

            assert deliver(receipt_a, "看了 same kettle observation")
            assert watched_at_db() == second
            assert thread_statuses() == statuses_after_b
            assert len(llm_calls) == 4
    finally:
        manager.unload()
        local_provider.shutdown()


def test_video_detail_uses_most_recent_cid_metadata_and_keeps_cross_version_content(provider):
    bvid = "BV1RecordMx1"
    owner = provider._owner_key
    base = {
        "bvid": bvid, "platform": "bilibili", "source_object_id": bvid,
        "mode": "video_analysis+captions_only", "read_scope": '{"kind":"self"}',
        "owner_key": owner, "session_id": "fixture-video-session",
        "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
        "raw_observation": "{}",
    }
    with provider._conn:
        provider._conn.execute(
            """INSERT INTO video_watches
               (bvid, platform, source_object_id, title, cid, mode, analysis, captions, summary,
                source_context, raw_observation, read_scope, owner_key, session_id, watched_at,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bvid, "bilibili", bvid, "older A", "cid-A", base["mode"], "ANALYSIS-A",
             '[{"text":"CAPTION-A"}]', "old summary A",
             json.dumps({"kind":"self", "platform":"bilibili", "ref":"old-A"}),
             "{}", base["read_scope"], owner, base["session_id"], "2026-01-03T00:00:00+00:00",
             base["created_at"], base["updated_at"]),
        )
        provider._conn.execute(
            """INSERT INTO video_watches
               (bvid, platform, source_object_id, title, cid, mode, analysis, captions, summary,
                source_context, raw_observation, read_scope, owner_key, session_id, watched_at,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bvid, "bilibili", bvid, "middle B", "cid-B", base["mode"], "ANALYSIS-B",
             '[{"text":"CAPTION-B"}]', "summary B",
             json.dumps({"kind":"self", "platform":"bilibili", "ref":"middle-B"}),
             "{}", base["read_scope"], owner, base["session_id"], "2026-01-02T00:00:00+00:00",
             base["created_at"], base["updated_at"]),
        )
        provider._conn.execute(
            """UPDATE video_watches SET title='latest A', summary='latest summary A',
               source_context=?, watched_at='2026-01-04T00:00:00+00:00'
               WHERE source_object_id=? AND cid='cid-A'""",
            (json.dumps({"kind":"self", "platform":"bilibili", "ref":"latest-A"}), bvid),
        )

    result = provider.video_memory.get_video_watch_detail(bvid, owner)
    assert result["cid"] == "cid-A"
    assert result["title"] == "latest A"
    assert result["summary"] == "latest summary A"
    assert json.loads(result["source_context"])["ref"] == "latest-A"
    assert result["watched_at"] == "2026-01-04T00:00:00+00:00"
    assert result["analysis"] == "ANALYSIS-A\n\nANALYSIS-B"
    assert {row["text"] for row in json.loads(result["captions"])} == {"CAPTION-A", "CAPTION-B"}


def test_lookup_can_read_only_trusted_persona_owned_legacy_self_rows(provider, addon):
    client, _scope_key = addon
    bvid = "BV1RecordMx1"
    with provider._conn:
        provider._conn.execute(
            """INSERT INTO video_watches
               (bvid, platform, source_object_id, title, cid, mode, analysis, captions, summary,
                source_context, raw_observation, read_scope, owner_key, session_id, watched_at,
                created_at, updated_at)
               VALUES (?, 'bilibili', ?, 'legacy video', 'legacy-cid', 'video_analysis',
                'LEGACY-PRIVATE-ANALYSIS', '[]', 'legacy summary', ?, '{}', ?, '', '',
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')""",
            (bvid, bvid,
             json.dumps({"kind":"self", "platform":"bilibili", "certainty":"observed",
                         "visibility":"public", "ref":f"https://www.bilibili.com/video/{bvid}/"}),
             '{"kind":"self"}'),
        )
        provider._conn.execute(
            """INSERT INTO video_watches
               (bvid, platform, source_object_id, title, cid, mode, analysis, captions, summary,
                source_context, raw_observation, read_scope, owner_key, session_id, watched_at,
                created_at, updated_at)
               VALUES (?, 'bilibili', ?, 'another owner', 'other-cid', 'video_analysis',
                'DO-NOT-READ', '[]', 'other summary', ?, '{}', ?, 'someone-else', '',
                '2026-01-02T00:00:00+00:00', '2026-01-02T00:00:00+00:00', '2026-01-02T00:00:00+00:00')""",
            ("BV1OtherMx01", "BV1OtherMx01",
             json.dumps({"kind":"self", "platform":"bilibili", "certainty":"observed",
                         "visibility":"public", "ref":"https://www.bilibili.com/video/BV1OtherMx01/"}),
             '{"kind":"self"}'),
        )
    with _private_route():
        legacy = client.lookup_video_watch("bilibili", bvid, cid="legacy-cid")
        assert legacy["analysis"] == "LEGACY-PRIVATE-ANALYSIS"
        assert client.lookup_video_watch("bilibili", "BV1OtherMx01", cid="other-cid") is None


def test_ambiguous_legacy_cross_cid_analysis_is_not_reused_until_current_cid_is_reobserved(provider, addon):
    client, _scope_key = addon
    with _private_route():
        assert client.record_video_watch({
            "status": "ok", "mode": "video_analysis", "platform": "bilibili",
            "source_object_id": "BV1RecordMx1", "cid": "cid-b",
            "analysis": "SCENE-A: old source.\n\nSCENE-B: current source.",
            "title": "cross-CID legacy fixture",
        }, delivered=True, public=True, _watch_receipt="5123456789abcdef0123456789abcdef")
        provider._conn.execute(
            "UPDATE video_watches SET summary=?, raw_observation=? WHERE source_object_id=? AND cid=?",
            ("CID cid-a scene; CID cid-b scene", json.dumps({"cid": "cid-b", "analysis": "SCENE-B: current source."}),
             "BV1RecordMx1", "cid-b"),
        )
        from sanae_mem.storage.database import _mark_ambiguous_legacy_video_rows
        _mark_ambiguous_legacy_video_rows(provider._conn)
        assert client.lookup_video_watch("bilibili", "BV1RecordMx1", cid="cid-b") is None

        assert client.record_video_watch({
            "status": "ok", "mode": "video_analysis", "platform": "bilibili",
            "source_object_id": "BV1RecordMx1", "cid": "cid-b",
            "analysis": "SCENE-B: current source.", "title": "cross-CID legacy fixture",
        }, delivered=True, public=True, _watch_receipt="6123456789abcdef0123456789abcdef")
        exact = client.lookup_video_watch("bilibili", "BV1RecordMx1", cid="cid-b")
        assert exact["analysis"] == "SCENE-B: current source."
        archived = provider._conn.execute(
            "SELECT analysis, cid FROM video_watches WHERE source_object_id=? AND cid LIKE 'legacy-ambiguous:%'",
            ("BV1RecordMx1",),
        ).fetchone()
        assert archived is not None and "SCENE-A: old source." in archived["analysis"]


def test_video_fts_real_error_is_logged_to_status_then_falls_back(provider, caplog):
    bvid = "BV1FtsFail01"
    _insert_video_row(
        provider, bvid, "fixture windchime clip", "fixture windchime plays; SYNTHETIC-STATUS-EXCLUDED-MARKER",
        provider._owner_key,
    )
    with provider._lock:
        provider._conn.execute("DROP TABLE video_fts")
        provider._conn.commit()

    caplog.set_level(logging.WARNING)
    with _private_route():
        hits = provider.video_memory.search_video_watches("windchime", owner_key=provider._owner_key)
        assert hits and hits[0]["bvid"] == bvid
        state = provider.status_store.read()["video_fts"]
        assert state["failure_count"] == 1
        assert state["last_failure"] == "OperationalError"
        assert state["last_failure_at"]
        serialized = json.dumps(provider.status_store.read(), ensure_ascii=False)
    assert "windchime" not in serialized
    assert "SYNTHETIC-STATUS-EXCLUDED-MARKER" not in serialized
    assert any("video FTS query failed" in record.message for record in caplog.records)
