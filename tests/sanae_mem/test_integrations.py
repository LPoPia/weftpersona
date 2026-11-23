"""Exercise the optional adapter in the pinned original plugin loader and tool registry."""

from __future__ import annotations

import base64
import json
import shutil
import sqlite3
import time
from contextlib import ExitStack, closing
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
from test_original_hermes import HERMES_ROOT, _route

from agent.memory_manager import MemoryManager
from agent.delegation_context import delegated_child_context
from hermes_cli.plugins import PluginManager, PluginManifest
from plugins.memory import import_provider_module, load_memory_provider
from tools.registry import registry, session_tool_scope

from sanae_mem import SanaeMemProvider
from sanae_mem.cognitive.self_utterances import recent_self_utterances
from sanae_mem.governance.source_context import current_chat_scope


VIDEO_TOOL = "sanae_video_detail"
OBSERVATION = {"status": "ok", "bvid": "BV1234567890", "mode": "captions_only",
               "lines": [{"text": "SYNTHETIC-CAPTION"}]}
LEGACY_BVID = "BV0987654321"


def _load_bilibili_addon(manager, root, *, resolver=True):
    addon = root / "bilibili_public"
    addon.mkdir()
    setup = (
        "from sanae_mem.integrations import register_bilibili_memory\n"
        if resolver else
        "from plugins.memory import import_provider_module\n"
        'register_bilibili_memory = import_provider_module("sanae_mem", "integrations").register_bilibili_memory\n'
    )
    binding = "register_bilibili_memory(ctx, providers.get)" if resolver else "register_bilibili_memory(ctx)"
    (addon / "__init__.py").write_text(
        setup +
        "providers = {}\n"
        "def register(ctx):\n"
        "    global bind_memory\n"
        f"    bind_memory = {binding}\n", encoding="utf-8",
    )
    manager._load_plugin(PluginManifest(name="bilibili_public", source="user", path=str(addon)))
    loaded = manager._plugins["bilibili_public"]
    assert loaded.enabled, loaded.error
    return loaded.module


def test_optional_bilibili_is_owned_by_the_adapter_and_preserves_storage(tmp_path, monkeypatch):
    import hermes_cli.plugins

    assert HERMES_ROOT.resolve() in Path(hermes_cli.plugins.__file__).resolve().parents
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  isolation: in_process\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    data = tmp_path / "explicit-data-dir"
    config = {
        "data_dir": str(data), "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "account-A"},
    }
    manager = PluginManager(scope_key=str(home))
    memory_managers = []

    def make_provider(actor, session):
        provider = SanaeMemProvider(config)
        memory_manager = MemoryManager()
        with _route(actor=actor, session=session):
            memory_manager.add_provider(provider)
            memory_manager.initialize_all(session_id=session, hermes_home=str(home),
                                          platform="telegram", agent_context="primary")
        memory_managers.append(memory_manager)
        return provider, memory_manager

    def tool(bvid, session=""):
        return json.loads(registry.dispatch(VIDEO_TOOL, {"bvid": bvid},
                                            scope=manager.scope_key, session_id=session))

    try:
        provider_a, mem_a = make_provider("account-A", "private-A")
        provider_b, _ = make_provider("account-B", "private-B")
        with _route(actor="account-A", session="private-A"):
            owner_a = current_chat_scope().owner
            assert mem_a.has_tool("sanae_meme_search")
            assert not mem_a.has_tool(VIDEO_TOOL)
            assert registry.get_entry(VIDEO_TOOL, scope=manager.scope_key) is None
            assert json.loads(provider_a.handle_tool_call(VIDEO_TOOL, {"bvid": LEGACY_BVID}))["ok"] is False
            image = BytesIO()
            Image.new("RGB", (2, 2)).save(image, format="PNG")
            source = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")
            saved = json.loads(mem_a.handle_tool_call("sanae_meme_save", {
                "title": "月亮唱歌", "meaning": "SYNTHETIC-PRIVATE-MEANING", "images": [source],
            }))
            assert saved["ok"] is True, saved
            assert provider_a._conn.execute("SELECT count(*) FROM meme_assets").fetchone()[0] == 1
        with provider_a._conn:
            provider_a._conn.execute(
                """INSERT INTO video_watches (bvid, owner_key, read_scope, analysis, watched_at, created_at, updated_at)
                   VALUES (?, ?, ?, 'OLD-PRIVATE-DETAIL', '2026-01-01', '2026-01-01', '2026-01-01')""",
                (LEGACY_BVID, owner_a, json.dumps({"kind": "owner", "owner": owner_a})),
            )

        addon = _load_bilibili_addon(manager, tmp_path)
        with session_tool_scope(manager.scope_key), _route(actor="account-A", session="private-A"):
            # Stock Hermes builds the tool catalog before initializing its memory providers.
            assert registry.get_definitions({VIDEO_TOOL})[0]["function"]["name"] == VIDEO_TOOL
            assert tool(LEGACY_BVID)["ok"] is False
        addon.providers.update({"private-A": provider_a, "private-B": provider_b})
        client_a = addon.bind_memory(provider_a)
        stamp = time.time() - 10
        reply = dict(receipt_id="example-receipt-1", at=stamp, text="SYNTHETIC-PRIVATE-REPLY", visibility="private")

        with session_tool_scope(manager.scope_key), _route(actor="account-A", session="private-A"):
            assert tool(LEGACY_BVID)["result"]["analysis"] == "OLD-PRIVATE-DETAIL"
            assert not client_a.record_video_watch(OBSERVATION, delivered=False, public=True)
            assert not client_a.record_video_watch(OBSERVATION, delivered=True, public=False)
            assert client_a.record_video_watch(OBSERVATION, delivered=True, public=True,
                                               _watch_receipt="0123456789abcdef0123456789abcdef",
                                               review_public_excerpt=lambda _: False)
            assert client_a.get_public_video_watch(OBSERVATION["bvid"]) == ""
            assert client_a.record_video_watch(OBSERVATION, delivered=True, public=True,
                                               _watch_receipt="1123456789abcdef0123456789abcdef",
                                               review_public_excerpt=lambda _: True)
            assert "SYNTHETIC-CAPTION" in client_a.get_public_video_watch(OBSERVATION["bvid"])
            assert client_a.known_meme_terms(["看，月亮唱歌"]) == []
            assert client_a.public_meme_senses(["看，月亮唱歌"]) == []
            assert "SYNTHETIC-PRIVATE-MEANING" in client_a.meme_context_for_public_material(["看，月亮唱歌"])
            assert not client_a.meme_context_for_public_material(["看，月亮唱歌"], public_only=True)
            evidence = {
                "source_plugin": "bilibili_public", "platform": "bilibili",
                "source_object_id": LEGACY_BVID, "material_kind": "processed",
                "privacy_scope": "public", "processing_type": "video_meme_evidence",
                "raw_locator": f"https://www.bilibili.com/video/{LEGACY_BVID}",
                "title": "彩纸开会", "content": "评论说彩纸开会",
                "extra_metadata": {
                    "meaning": "SYNTHETIC-PUBLIC-MEANING",
                    "occurrence_kind": "analysis", "occurrence_quote": "评论说彩纸开会",
                    "verification_url": "https://example.test/meme",
                    "verification_text": "彩纸开会：SYNTHETIC-PUBLIC-MEANING",
                    "verification_quote": "彩纸开会：SYNTHETIC-PUBLIC-MEANING",
                },
            }
            assert client_a.ingest_source_material(evidence)["ok"]
            assert client_a.known_meme_terms(["彩纸开会"]) == []
            evidence["extra_metadata"]["occurrence_kind"] = "comment"
            wrong_sense = {**evidence, "extra_metadata": {
                **evidence["extra_metadata"], "meaning": "UNSUPPORTED-ROLE-MEANING",
            }}
            assert client_a.ingest_source_material(wrong_sense)["ok"]
            assert client_a.known_meme_terms(["彩纸开会"]) == []
            assert client_a.ingest_source_material(evidence)["ok"]
            assert client_a.known_meme_terms(["彩纸开会"]) == ["彩纸开会"]
            public_context = client_a.meme_context_for_public_material(
                ["彩纸开会", "月亮唱歌"], public_only=True,
            )
            assert "SYNTHETIC-PUBLIC-MEANING" in public_context
            assert "https://example.test/meme" in public_context
            assert "SYNTHETIC-PRIVATE-MEANING" not in public_context
            sense, = client_a.public_meme_senses(["彩纸开会", "月亮唱歌"])
            assert sense["term"] == "彩纸开会" and sense["source"] == "https://example.test/meme"
            with delegated_child_context("private-A"):
                assert client_a.meme_context_for_public_material(["看，月亮唱歌"]) == ""
                assert client_a.public_meme_senses(["彩纸开会"]) == []
            assert not client_a.record_accepted_reply(**reply, accepted=False)
            assert not client_a.record_accepted_reply(**reply, accepted=True, test_reply=True)
            assert client_a.record_accepted_reply(**reply, accepted=True)
            assert not client_a.record_accepted_reply(**reply, accepted=True)
            assert "SYNTHETIC-PRIVATE-REPLY" in recent_self_utterances(owner_a, "private-A", data_dir=data)
            assert tool(LEGACY_BVID, session="private-B")["ok"] is False

        with session_tool_scope(manager.scope_key), _route(actor="account-B", session="private-B"):
            owner_b = current_chat_scope().owner
            assert tool(LEGACY_BVID)["result"]["found"] is False
            # Cross-owner reads only get sanitized public_excerpt projections, never another owner's full detail.
            assert tool(OBSERVATION["bvid"])["result"]["found"] is False
            assert client_a.get_public_video_watch(OBSERVATION["bvid"]) == ""
            assert client_a.meme_context_for_public_material(["看，月亮唱歌"]) == ""
            assert recent_self_utterances(owner_b, "private-B", data_dir=data) == ""

        with session_tool_scope(manager.scope_key), _route(actor="account-A", session="private-A", chat_type="group"):
            # Availability is static; the handler enforces the current private route.
            assert registry.get_definitions({VIDEO_TOOL})[0]["function"]["name"] == VIDEO_TOOL
            assert tool(OBSERVATION["bvid"])["ok"] is False
            assert client_a.meme_context_for_public_material(["看，月亮唱歌"]) == ""
            public_turn = provider_a.handle_pre_llm_call(session_id="private-A", user_message="彩纸开会，月亮唱歌")
            assert "SYNTHETIC-PUBLIC-MEANING" in public_turn.get("context", "")
            assert "SYNTHETIC-PRIVATE-MEANING" not in public_turn["context"]
            assert not client_a.record_video_watch(OBSERVATION, delivered=True, public=True)
            assert recent_self_utterances(owner_a, "private-A", data_dir=data) == ""

        assert manager.unload("bilibili_public")
        assert registry.get_entry(VIDEO_TOOL, scope=manager.scope_key) is None
        with _route(actor="account-A", session="private-A"):
            assert client_a.get_public_video_watch(OBSERVATION["bvid"]) == ""
            assert not client_a.record_video_watch(OBSERVATION, delivered=True, public=True)
            assert mem_a.has_tool("sanae_meme_search")
        with closing(sqlite3.connect(data / "sanae.db")) as conn:
            assert set(conn.execute("SELECT bvid, owner_key FROM video_watches")) == {
                (LEGACY_BVID, owner_a), (OBSERVATION["bvid"], owner_a),
            }
    finally:
        manager.unload()
        for memory_manager in memory_managers:
            memory_manager.shutdown_all()


def test_video_shaped_source_material_does_not_register_a_watch(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  isolation: in_process\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    manager = PluginManager(scope_key=str(home))
    provider = SanaeMemProvider({
        "data_dir": str(tmp_path / "data"), "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "account-A"},
    })
    try:
        with _route(actor="account-A", session="private-A"):
            manager_memory = MemoryManager()
            manager_memory.add_provider(provider)
            manager_memory.initialize_all(session_id="private-A", hermes_home=str(home),
                                          platform="telegram", agent_context="primary")
            addon = _load_bilibili_addon(manager, tmp_path)
            addon.providers["private-A"] = provider
            client = addon.bind_memory(provider)
            verdict = client.ingest_source_material({
                "source_plugin": "douyin_source", "platform": "douyin",
                "source_object_id": "dy-video-42", "material_kind": "processed",
                "processing_type": "video_watch_delivery", "privacy_scope": "private",
                "title": "synthetic clip", "content": "synthetic processed captions",
                "extra_metadata": {"cid": "cid-7", "delivery_id": "delivery-7",
                                   "reused_video_watch": {"record_id": 1}},
            })
            assert verdict["ok"] is True
            retry = client.ingest_source_material({
                "source_plugin": "douyin_source", "platform": "douyin",
                "source_object_id": "dy-video-42", "material_kind": "processed",
                "processing_type": "video_watch_delivery", "privacy_scope": "private",
                "title": "synthetic clip", "content": "synthetic processed captions",
                "extra_metadata": {"cid": "cid-7", "delivery_id": "delivery-7",
                                   "reused_video_watch": {"record_id": 1}},
            })
            next_delivery = client.ingest_source_material({
                "source_plugin": "douyin_source", "platform": "douyin",
                "source_object_id": "dy-video-42", "material_kind": "processed",
                "processing_type": "video_watch_delivery", "privacy_scope": "private",
                "title": "synthetic clip", "content": "synthetic processed captions",
                "extra_metadata": {"cid": "cid-7", "delivery_id": "delivery-8",
                                   "reused_video_watch": {"record_id": 1}},
            })
            assert retry["deduplicated"] is True
            assert next_delivery["ok"] is True and next_delivery["deduplicated"] is True
            assert verdict["material_id"] == retry["material_id"] == next_delivery["material_id"]
            assert provider._conn.execute("SELECT count(*) FROM video_watches").fetchone()[0] == 0
            assert provider._conn.execute("SELECT count(*) FROM source_materials").fetchone()[0] == 1
            metadata = json.loads(provider._conn.execute(
                "SELECT extra_metadata FROM source_materials"
            ).fetchone()[0])
            assert {item["delivery_id"] for item in metadata["video_deliveries"]} == {
                "delivery-7", "delivery-8",
            }
            from sanae_mem.integrations import register_video_memory
            source_ctx = SimpleNamespace(plugin_id="douyin_source", has_plugin=lambda _name: True)
            source_binding = register_video_memory(
                source_ctx, resolve_provider=lambda session: provider if session == "private-A" else None,
            )
            video_memory = source_binding.for_session("private-A")
            assert video_memory is not None
            observation = {"status": "ok", "mode": "video_analysis",
                           "title": "synthetic clip", "analysis": "settled visual content"}
            token = "3123456789abcdef0123456789abcdef"
            assert not video_memory.record_video_watch(
                observation, platform="douyin", source_object_id="dy-video-42", cid="v2",
                settled=False, _watch_receipt=token,
            )
            assert video_memory.record_video_watch(
                observation, platform="douyin", source_object_id="dy-video-42", cid="v2",
                settled=True, _watch_receipt=token,
            )
            assert video_memory.lookup_video_watch("douyin", "dy-video-42") is not None
            exact = video_memory.lookup_video_watch("douyin", "dy-video-42", cid="v2")
            assert exact["analysis"] == "settled visual content"
            manager_memory.shutdown_all()
    finally:
        manager.unload()
        provider.shutdown()


def test_default_binder_tracks_stock_loaded_providers_across_session_lifecycle(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    installed = home / "plugins" / "sanae_mem"
    shutil.copytree(Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem", installed,
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    config = {"data_dir": str(tmp_path / "sanae-mem"), "semantic_backend": "sqlite",
              "memory_identity": {"platform": "telegram", "user_id": "account-A"}}
    (home / "config.yaml").write_text(json.dumps({
        "memory": {"provider": "sanae_mem"},
        "plugins": {"enabled": ["sanae_mem"], "isolation": "in_process", "sanae_mem": config},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    with ExitStack() as cleanup:
        plugin_manager = PluginManager(scope_key=str(home))
        cleanup.callback(plugin_manager.unload)
        addon = _load_bilibili_addon(plugin_manager, tmp_path, resolver=False)
        for_session = addon.bind_memory.for_session
        assert all(for_session(value) is None for value in ("", None, 42, "private-A"))

        provider = load_memory_provider("sanae_mem", register_skills=False)
        assert provider is not None
        entry = import_provider_module("sanae_mem", "plugin_entry")
        assert type(provider) is entry.SanaeMemProvider
        assert for_session("private-A") is None

        manager = MemoryManager()
        cleanup.callback(manager.shutdown_all)
        options = dict(hermes_home=str(home), platform="telegram", agent_context="primary")

        def tool(session_id):
            return json.loads(registry.dispatch(
                VIDEO_TOOL, {"bvid": OBSERVATION["bvid"]},
                scope=plugin_manager.scope_key, session_id=session_id,
            ))

        with session_tool_scope(plugin_manager.scope_key), _route(actor="account-A", session="private-A"):
            manager.add_provider(provider)
            manager.initialize_all(session_id="private-A", **options)
            client = for_session("private-A")
            assert client is not None and client._provider is provider
            assert entry.resolve_active_provider(tmp_path / "other-profile", "private-A") is None
            assert client.record_video_watch(
                OBSERVATION, delivered=True, public=True, review_public_excerpt=lambda _: True,
                _watch_receipt="2123456789abcdef0123456789abcdef",
            )
            assert tool("private-A")["result"]["found"] is True
            assert tool("private-B")["ok"] is False

            saved = json.loads(manager.handle_tool_call("sanae_meme_save", {
                "title": "水晶月亮", "meaning": "SYNTHETIC-BOUND-MEME-MEANING",
            }))
            assert saved["ok"] is True, saved
            parts = ["看，水晶月亮"]
            assert "SYNTHETIC-BOUND-MEME-MEANING" in client.meme_context_for_public_material(parts)
            with delegated_child_context("private-A"):
                assert client.meme_context_for_public_material(parts) == ""

            # The main and both hook instances are live together; neither hook may replace the main.
            with ExitStack() as workers:
                for label in ("pre", "post"):
                    worker = entry.SanaeMemProvider(config)
                    workers.callback(worker.shutdown)
                    worker.initialize("private-A", hook_worker=True, **options)
                    assert for_session("private-A")._provider is provider
                    worker.on_session_switch(f"hook-{label}")
                    assert for_session(f"hook-{label}") is None
                shadow = entry.SanaeMemProvider({**config, "shadow": True})
                workers.callback(shadow.shutdown)
                shadow.initialize("private-A", **options)
                assert for_session("private-A")._provider is provider
            assert for_session("private-A")._provider is provider

            replacement = entry.SanaeMemProvider(config)
            cleanup.callback(replacement.shutdown)
            replacement.initialize("private-A", **options)
            assert for_session("private-A")._provider is replacement
            replacement.on_session_switch("private-B")
            assert for_session("private-A")._provider is provider
            assert for_session("private-B")._provider is replacement

        for actor, session, chat_type in (
            ("account-A", "private-A", "group"),
            ("account-A", "private-A", "channel"),
            ("account-A", "private-B", "dm"),
            ("account-B", "private-A", "dm"),
        ):
            with _route(actor=actor, session=session, chat_type=chat_type):
                assert client.meme_context_for_public_material(parts) == ""

        with _route(actor="account-A", session="private-A"):
            manager.on_session_switch("private-B")
        assert for_session("private-A") is None
        assert for_session("private-B")._provider is provider
        replacement.shutdown()
        assert for_session("private-B")._provider is provider

        with _route(actor="account-A", session="private-B"):
            assert tool("private-B")["result"]["found"] is True
            assert "SYNTHETIC-BOUND-MEME-MEANING" in client.meme_context_for_public_material(parts)
            manager.shutdown_all()
            assert for_session("private-B") is None
            assert client.meme_context_for_public_material(parts) == ""


def test_bilibili_session_lookup_surfaces_unexpected_resolver_errors(tmp_path, monkeypatch):
    from sanae_mem.integrations import register_bilibili_memory

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    class Context:
        plugin_id = "bilibili_public"

        def has_plugin(self, _plugin_id):
            return True

        def current_cron_execution(self):
            return None

        def register_tool(self, **_):
            return SimpleNamespace(active=True)

    def broken_resolver(_session_id):
        raise RuntimeError("resolver programming error")

    injected = register_bilibili_memory(Context(), broken_resolver)
    with _route(actor="account-A", session="private-A"), pytest.raises(
            RuntimeError, match="resolver programming error"):
        injected.for_session("private-A")
