"""Real SQLite history and public reads with only the memory plugin available."""

from contextlib import closing, contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote

import pytest

pytest.importorskip("hermes_host_contracts")

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_host_contracts import TurnEnvelope, turn_envelope_scope
from sanae_mem import SanaeMemProvider, submit_source_material
from sanae_mem.bilibili_history import BILIBILI_SOURCE
from sanae_mem.integrations import register_video_memory
from sanae_mem.cognitive.life_context import collect_recent_activities
from sanae_mem.governance.memory_evidence import build_user_evidence
from sanae_mem.cognitive.self_utterances import recent_self_utterances
from sanae_mem.governance.shared_memory import is_other_person_source
from sanae_mem.governance.source_context import current_chat_scope, current_source_record
from sanae_mem.infra.source_descriptors import PrivatePeerRoute, SourceDescriptor, install_builtin_sources


def _private_turn(actor: str, platform: str = "telegram") -> TurnEnvelope:
    return TurnEnvelope(platform=platform, chat_type="dm", chat_id=actor, sender_id=actor,
                        session_id=actor, session_key=f"{platform}:dm:{actor}")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path):
    token = set_hermes_home_override(str(tmp_path))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@contextmanager
def closing_memory(provider: SanaeMemProvider):
    try:
        yield provider
    finally:
        provider.shutdown()


def _open_memory(home: Path, actor: str = "reader") -> SanaeMemProvider:
    provider = SanaeMemProvider({
        "semantic_backend": "sqlite", "memory_identity": {"platform": "telegram", "user_id": actor},
    })
    with turn_envelope_scope(_private_turn(actor)):
        provider.initialize(actor, hermes_home=str(home), platform="telegram", agent_context="primary")
    return provider


def test_public_activity_preserves_both_origins_and_hides_other_owners_bodies(tmp_path):
    with closing_memory(_open_memory(tmp_path)):
        data = tmp_path / "sanae-mem"
        at = datetime.now().astimezone()
        iso = at.isoformat()
        object_id = "BV1234567890"
        legacy_ref = f"https://www.bilibili.com/video/{object_id}/"
        other_ref = f"https://video.example.test/{object_id}"
        with closing(sqlite3.connect(data / "sanae.db")) as db, db:
            for platform, stored, canonical_id, title, source, mode, excerpt, scope in (
                ("", object_id, "", "同名视频", {"kind": "self", "ref": legacy_ref},
                 "captions_only", "", "self"),
                ("douyin", "douyin:" + object_id, object_id, "同名视频",
                 {"kind": "video", "platform": "douyin", "ref": other_ref},
                 "video_analysis", "经审核的公开视频简介", "self"),
                ("douyin", "douyin:private", "private", "外人的私人标题",
                 {"kind": "video", "platform": "douyin", "ref": "https://video.example.test/private"},
                 "video_analysis", "", "owner"),
            ):
                source.update(visibility="public", certainty="observed")
                db.execute(
                    """INSERT INTO video_watches
                       (platform,bvid,source_object_id,title,source_context,mode,public_excerpt,
                        read_scope,owner_key,summary,raw_observation,analysis,captions,
                        watched_at,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (platform, stored, canonical_id, title, json.dumps(source), mode, excerpt,
                     json.dumps({"kind": scope}), "other-owner", "PRIVATE-SUMMARY",
                     "PRIVATE-OBSERVATION", "PRIVATE-ANALYSIS", "PRIVATE-CAPTION", iso, iso, iso),
                )
        with turn_envelope_scope(_private_turn("reader")):
            result = collect_recent_activities(
                home=tmp_path, data_dir=data, owner_key=current_chat_scope().owner,
                session_id="reader", since=at.timestamp()-60, through=at.timestamp()+1,
                private_experience=False,
            )
        assert result.count("同名视频") == 2
        assert "B站" in result and "字幕，未观看画面" in result and legacy_ref in result
        assert "douyin" in result and "抽样画面分析" in result and other_ref in result
        assert "经审核的公开视频简介" in result
        assert "PRIVATE-" not in result and "外人的私人标题" not in result


def test_cold_receipt_is_read_only_by_its_owner(tmp_path):
    data = tmp_path / "sanae-mem"
    data.mkdir()
    with turn_envelope_scope(_private_turn("reader")):
        reader = current_chat_scope().owner
    with closing(sqlite3.connect(data / "sanae.db")) as db, db:
        db.execute("""CREATE TABLE bilibili_reply_receipts
                      (owner_key TEXT, receipt_id TEXT, at REAL, text TEXT, visibility TEXT)""")
        db.executemany("INSERT INTO bilibili_reply_receipts VALUES (?, ?, ?, ?, ?)", [
            (reader, "one", time.time()-2, "自己的冷启动历史回执", "private"),
            ("foreign", "one", time.time()-2, "别人的历史回执", "private"),
        ])
    with turn_envelope_scope(_private_turn("reader")):
        text = recent_self_utterances(reader, "reader", home=tmp_path, data_dir=data)
    assert "B站私信" in text and "自己的冷启动历史回执" in text
    assert "别人的历史回执" not in text


def test_public_submit_accepts_independent_evidence_and_checks_its_original(tmp_path, monkeypatch):
    addon = tmp_path / "plugins" / "other_source"
    addon.mkdir(parents=True)
    (addon / "plugin.yaml").write_text("name: other_source\n", encoding="utf-8")
    (addon / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [other_source]\n", encoding="utf-8")
    empty_bundled = tmp_path / "empty-bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    with closing_memory(_open_memory(tmp_path)) as provider:
        term = "彩纸开会"
        quote = f"{term}也就是聚成一团"
        with turn_envelope_scope(_private_turn("reader")):
            evidence = {
                "source_plugin": "bilibili_public", "platform": "bilibili",
                "material_kind": "processed", "privacy_scope": "public",
                "processing_type": "video_meme_evidence", "processing_version": "2",
                "source_object_id": "BV1234567890", "title": term, "content": term,
                "raw_locator": "https://www.bilibili.com/video/BV1234567890",
                "extra_metadata": {
                    "evidence_status": "confirmed", "occurrence_kind": "comment",
                    "occurrence_quote": term, "meaning": "聚成一团", "verification_quote": quote,
                    "verification_text": quote, "verification_url": "https://proof.example.test/meme",
                },
            }
            manager = PluginManager(scope_key=str(tmp_path))
            try:
                manager.discover_and_load()
                context = PluginContext(PluginManifest(name="other_source"), manager)
                assert context.has_plugin("other_source")
                foreign_binding = register_video_memory(context)(provider)
                verdict = foreign_binding.ingest_source_material(evidence)
                assert not verdict["ok"] and verdict["reason"] == "untrusted_video_meme_route"
            finally:
                manager.unload()
            for mismatch in (
                {"raw_locator": "https://www.bilibili.com/video/BV0987654321"},
                {"source_plugin": "Bilibili_Public"},
                {"material_kind": "raw"},
            ):
                verdict = submit_source_material({**evidence, **mismatch}, session_id="reader")
                assert not verdict["ok"] and verdict["reason"] == "untrusted_video_meme_route"

            bad_quote = {**evidence, "extra_metadata": {
                **evidence["extra_metadata"], "verification_text": f"没有逐字引文：{term}",
            }}
            assert submit_source_material(bad_quote, session_id="reader")["ok"]
            assert provider.known_meme_terms([term]) == []
            assert submit_source_material(evidence, session_id="reader")["ok"]
            assert provider.known_meme_terms([term]) == [term]
        with closing(sqlite3.connect(tmp_path / "sanae-mem" / "sanae.db")) as db:
            assert db.execute("SELECT COUNT(*) FROM video_watches").fetchone() == (0,)
            assert db.execute("SELECT COUNT(*) FROM source_materials").fetchone() == (2,)
            assert db.execute("SELECT COUNT(*) FROM meme_records").fetchone() == (1,)


def test_plugin_chat_transport_keeps_original_speaker_and_private_owner():
    platform = "test-extra-transport"
    with turn_envelope_scope(_private_turn("reader", platform)):
        source = current_source_record(maintainer=True)
        assert source is not None and source["platform"] == platform
        scope = current_chat_scope()
        evidence = build_user_evidence(
            [{"role": "user", "content": "自己的原始发言"}],
            session="reader", user="reader", turn="turn", platform=platform,
            origin="foreground", source_context=source,
        )
        assert evidence.available and evidence.user_messages[0].text == "自己的原始发言"
        external_user = dict(source, kind="external_user")
        assert not is_other_person_source(external_user, f"owner:{scope.owner}", scope.owner)
        assert is_other_person_source(dict(external_user, author_id="someone-else"), "", scope.owner)


def test_unstructured_old_occurrences_preserve_each_owner_observation(tmp_path):
    with closing_memory(_open_memory(tmp_path)):
        data = tmp_path / "sanae-mem"
        now = datetime.now().astimezone()
        first, later = (now - timedelta(seconds=delta) for delta in (50, 20))
        prefix = "我在B站阅读了该视频的字幕，未观看画面："
        first_raw, later_raw = prefix + "第一次字幕。后半句也保留。", prefix + "第二次字幕。"
        foreign_raw = prefix + "另一个账号的旧字幕。"
        source = json.dumps({
            "kind": "self", "platform": "bilibili", "visibility": "public", "certainty": "observed",
            "ref": "https://www.bilibili.com/video/BV1234567890/",
        })
        with turn_envelope_scope(_private_turn("reader")):
            owner = current_chat_scope().owner
            with closing(sqlite3.connect(data / "sanae.db")) as db, db:
                for saved_owner, raw, occurrences in (
                    (owner, first_raw, [(first_raw, first), (later_raw, later)]),
                    ("other-owner", foreign_raw, [(foreign_raw, first)]),
                ):
                    event = db.execute(
                        """INSERT INTO events
                           (ts,raw,owner_key,read_scope,experience_subject,fact_subject,source_context,created_at)
                           VALUES (?,?,?,'{"kind":"self"}','sanae','sanae',?,?)""",
                        (first.isoformat(), raw, saved_owner, source, first.isoformat()),
                    ).lastrowid
                    db.executemany("INSERT INTO event_occurrences(event_id,raw,observed_at) VALUES(?,?,?)",
                                   [(event, occurrence, at.isoformat()) for occurrence, at in occurrences])
            def read_activities(*, private_experience: bool) -> str:
                return collect_recent_activities(
                    home=tmp_path, data_dir=data, owner_key=owner, session_id="reader",
                    since=now.timestamp()-60, through=now.timestamp(),
                    private_experience=private_experience,
                )

            private = read_activities(private_experience=True)
            assert first_raw in private and later_raw in private and foreign_raw not in private
            assert not read_activities(private_experience=False)


def test_private_peer_query_checks_source_route_and_caps_results(tmp_path):
    with closing_memory(_open_memory(tmp_path)) as provider:
        with turn_envelope_scope(_private_turn("reader")):
            owner = current_chat_scope().owner
        stamp = datetime.now().astimezone().isoformat()
        data = tmp_path / "sanae-mem" / "sanae.db"

        def record(platform, author, visibility, ref):
            return json.dumps(dict(kind="external_user", platform=platform, author_id=author,
                                   visibility=visibility, ref=ref))

        with closing(sqlite3.connect(data)) as db, db:
            rows = [
                (stamp, f"message-{i}", owner,
                 record("bilibili", "7", "private", f"bilibili:dm:42:7:{i}"), stamp)
                for i in range(505)
            ]
            rows.extend((stamp, "foreign", saved_owner, source, stamp) for saved_owner, source in (
                ("other-owner", record("bilibili", "7", "private", "bilibili:dm:42:7:1")),
                (owner, record("telegram", "7", "private", "bilibili:dm:42:7:1")),
                (owner, record("bilibili", "8", "private", "bilibili:dm:42:7:1")),
                (owner, record("bilibili", "7", "private", "bilibili:dm:42:8:1")),
                (owner, record("bilibili", "7", "public", "bilibili:dm:42:7:1")),
                (owner, "{invalid"),
            ))
            db.executemany(
                "INSERT INTO events(ts,raw,owner_key,source_context,created_at) VALUES(?,?,?,?,?)", rows,
            )
        result = provider.event_store.query_private_peer_events(stamp, "7", owner)
        assert [row["public_excerpt"] for row in result] == [f"message-{i}" for i in range(504, 4, -1)]
        assert provider.event_store.query_private_peer_events(stamp, "7", owner, platform="telegram") == []


def test_private_peer_query_uses_the_registered_source_ref_format(tmp_path):
    platform, peer = "extra-inbox", "team:ops[*]?%_"

    def matches_ref(ref: str, *, owner_key: str, peer: str) -> bool:
        prefix = f"inbox/{owner_key}/{quote(peer, safe='')}/"
        return ref.startswith(prefix) and ref[len(prefix):].isdecimal()

    source = SourceDescriptor(platform, "Extra inbox", PrivatePeerRoute(platform, matches_ref))
    with closing_memory(_open_memory(tmp_path)) as provider:
        with turn_envelope_scope(_private_turn("reader")):
            owner = current_chat_scope().owner
        stamp = datetime.now().astimezone().isoformat()
        sender = quote(peer, safe="")
        with closing(sqlite3.connect(tmp_path / "sanae-mem" / "sanae.db")) as db, db:
            db.executemany(
                "INSERT INTO events(ts,raw,owner_key,source_context,created_at) VALUES(?,?,?,?,?)",
                [(stamp, text, saved_owner, json.dumps(dict(
                    kind="external_user", platform=platform, author_id=peer, visibility="private", ref=ref,
                )), stamp) for text, saved_owner, ref in (
                    ("older", owner, f"inbox/{owner}/{sender}/1"),
                    ("newer", owner, f"inbox/{owner}/{sender}/2"),
                    ("wrong account", owner, f"inbox/other-account/{sender}/3"),
                    ("wrong owner", "other-owner", f"inbox/{owner}/{sender}/4"),
                    ("wrong peer", owner, f"inbox/{owner}/team/5"),
                )],
            )
        install_builtin_sources(BILIBILI_SOURCE, source)
        try:
            result = provider.event_store.query_private_peer_events(stamp, peer, owner, platform=platform)
        finally:
            install_builtin_sources(BILIBILI_SOURCE)

        assert [row["public_excerpt"] for row in result] == ["newer", "older"]
        assert result[0].keys() == ["id", "public_excerpt", "source_context", "created_at"]
