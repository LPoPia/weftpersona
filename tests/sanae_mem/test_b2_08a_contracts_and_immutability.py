"""Unit and public-contract integration tests for B2-08A."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from inspect import Parameter, signature
from pathlib import Path
from types import MappingProxyType
from typing import Any, Tuple, get_type_hints

import pytest
import yaml

from sanae_mem.cognitive import daily_semantic_validation, inner_state_store, nightly_journal, self_utterances
from sanae_mem.cognitive.inner_state_types import InnerState, InnerStateView
from sanae_mem.domains import commitments, meme_assets
from sanae_mem.governance import account_binding, llm_guard, memory_evidence, resident, source_context, source_intake
from sanae_mem.infra import atomic_io, auxiliary, chat_scope, contracts, event_fields, host_bridge, textutil
from sanae_mem.provider import SanaeMemProvider
from sanae_mem.runtime import delivery_receipts, group_memory, private_delivery_settler
from sanae_mem.storage import database


def test_vivemind_error_hierarchy_and_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    journal_err_cls = vars(nightly_journal)["_JournalRunWriteError"]
    assert issubclass(contracts.ViveMindError, Exception)
    for exc_cls in (
        journal_err_cls,
        meme_assets.MemeChatCopyError,
        account_binding.AccountBindingConflict,
        textutil.HermesHomeConflict,
        host_bridge.HostConfigParseError,
    ):
        assert issubclass(exc_cls, contracts.ViveMindError)

    assert issubclass(host_bridge.HostConfigParseError, ValueError)
    for runtime_cls in (
        journal_err_cls,
        meme_assets.MemeChatCopyError,
        account_binding.AccountBindingConflict,
        textutil.HermesHomeConflict,
    ):
        assert issubclass(runtime_cls, RuntimeError)

    bad_home = tmp_path / "bad_yaml_home"
    bad_home.mkdir()
    (bad_home / "config.yaml").write_text("foo: [unclosed\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(bad_home) == {}
    assert any("HostConfigParseError" in rec.getMessage() for rec in caplog.records)

    monkeypatch.setenv(textutil.HOME_ENV, str(tmp_path / "home_a"))
    monkeypatch.setenv(textutil.HOME_OVERRIDE_ENV, str(tmp_path / "home_b"))
    with pytest.raises(contracts.ViveMindError):
        textutil.hermes_home()

    with pytest.raises(contracts.ViveMindError):
        meme_assets.chat_target(tmp_path, {"path": "bad.txt", "sha256": "abc"})

    db_path = tmp_path / "bindings.db"
    with closing(database.open_database(db_path)) as conn:
        cfg_one = {
            "memory_identity": {"platform": "discord", "user_id": "owner-1"},
            "maintainer_account_bindings": [{"platform": "discord", "user_id": "alt-1"}],
        }
        account_binding.confirm_bindings(conn, cfg_one, profile="default")
        cfg_conflict = {
            "memory_identity": {"platform": "discord", "user_id": "owner-2"},
            "maintainer_account_bindings": [{"platform": "discord", "user_id": "alt-1"}],
        }
        with pytest.raises(contracts.ViveMindError):
            account_binding.confirm_bindings(conn, cfg_conflict, profile="default")


def test_dead_protocol_removed_and_private_delivery_protocol_slimmed() -> None:
    assert not hasattr(llm_guard, "ResidentRouteMessage")
    protocol = private_delivery_settler.PrivateDeliveryProvider
    annotations = get_type_hints(protocol)
    assert "_owner_key" not in annotations
    assert "_owner_key" not in vars(protocol)
    assert "_config" not in annotations and "_config" not in vars(protocol)
    assert "_sync_candidate" not in annotations and "_sync_candidate" not in vars(protocol)
    assert [
        name for name in vars(protocol)
        if name.startswith("_") and not name.startswith("__")
        and name not in {"_is_protocol", "_abc_impl", "_is_runtime_protocol"}
    ] == []
    assert "resolve_private_delivery_route" in vars(protocol)
    assert get_type_hints(resident.ResidentMemory.update_touch)["return"] is int

    record = protocol.record_private_turn_events
    parameters = signature(record).parameters
    required_keywords = ("current_message", "session_id", "platform", "author_id")
    assert tuple(parameters) == ("self", "user_text", *required_keywords)
    assert parameters["user_text"].kind == Parameter.POSITIONAL_OR_KEYWORD
    assert all(parameters[name].kind == Parameter.KEYWORD_ONLY for name in required_keywords)
    assert all(parameter.default is Parameter.empty for parameter in parameters.values())
    assert get_type_hints(record) == {
        "user_text": str,
        "current_message": dict[str, object] | None,
        "session_id": str,
        "platform": str,
        "author_id": str,
        "return": tuple[list[int], bool],
    }

    capture = memory_evidence.capture_private_turn
    parameters = signature(capture).parameters
    required_keywords = ("current_message", "session_id", "platform", "scope", "identity", "author_id")
    assert tuple(parameters) == ("text", *required_keywords)
    assert parameters["text"].kind == Parameter.POSITIONAL_OR_KEYWORD
    assert all(parameters[name].kind == Parameter.KEYWORD_ONLY for name in required_keywords)
    assert all(parameter.default is Parameter.empty for parameter in parameters.values())
    assert get_type_hints(capture) == {
        "text": str,
        "current_message": dict[str, object] | None,
        "session_id": str,
        "platform": str,
        "scope": chat_scope.ChatScope,
        "identity": source_context.ChatIdentity,
        "author_id": str,
        "return": memory_evidence.UserEvidence | None,
    }


def test_source_material_extra_metadata_is_truly_immutable_and_json_serializable() -> None:
    nested = {"nested": {"k": "v"}, "items": [1, 2]}
    material = source_intake.SourceMaterial(
        source_plugin="bilibili_public",
        platform="bilibili",
        material_kind="raw",
        content="hello",
        extra_metadata=nested,
    )
    nested["nested"]["k"] = "mutated"
    nested["items"].append(3)

    assert material.extra_metadata["nested"]["k"] == "v"
    assert material.extra_metadata["items"] == (1, 2)
    with pytest.raises(TypeError):
        material.extra_metadata["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        material.extra_metadata["nested"]["k"] = "x"

    from_map = source_intake.SourceMaterial.from_mapping({
        "source_plugin": "bilibili_public",
        "platform": "bilibili",
        "material_kind": "raw",
        "content": "world",
        "extra_metadata": {"a": {"b": [10]}},
    })
    with pytest.raises(TypeError):
        from_map.extra_metadata["a"]["b"] = 2

    thawed = contracts.thaw_json(from_map.extra_metadata)
    assert json.loads(json.dumps(thawed)) == {"a": {"b": [10]}}


def test_user_evidence_is_immutable_and_roundtrips_snapshot_and_source_json() -> None:
    msgs = [{"role": "user", "content": [{"type": "text", "text": "我喜欢喝乌龙茶"}]}]
    ctx = {
        "kind": "maintainer",
        "author_id": "u1",
        "platform": "telegram",
        "visibility": "private",
        "certainty": "verified",
        "ref": "m1",
    }
    ev = memory_evidence.build_user_evidence(
        msgs,
        session="s1",
        user="u1",
        platform="telegram",
        turn="m1",
        source_context=ctx,
    )
    assert ev.available is True
    msgs[0]["content"][0]["text"] = "hacked"  # type: ignore[index]
    ctx["kind"] = "external_user"

    assert ev.source_messages[0]["content"][0]["text"] == "我喜欢喝乌龙茶"
    assert ev.source_context is not None
    assert ev.source_context["kind"] == "maintainer"

    with pytest.raises(TypeError):
        ev.source_context["kind"] = "external_user"  # type: ignore[index]
    with pytest.raises(TypeError):
        ev.source_messages[0]["role"] = "assistant"  # type: ignore[index]

    snap = memory_evidence.snapshot_source_evidence(ev)
    encoded = json.dumps(snap, ensure_ascii=False)
    restored = memory_evidence.restore_source_evidence(json.loads(encoded))
    assert restored is not None and restored.available is True
    assert event_fields.source_json(restored.source_context) != ""


def test_inner_state_variables_immutable_and_store_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    data_dir = home / "sanae-mem"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    raw_vars = {"tension": 0.4, "closeness": 0.6, "energy": 0.1, "mood": 0.3}
    prev_state = InnerState(
        revision=1,
        text="刚才有些安静",
        recent_words="",
        recent_words_at=None,
        variables={"tension": 0.0, "closeness": 0.1, "energy": 0.1, "mood": 0.0},
        felt_at=900.0,
        updated_at=900.0,
        processed_at=None,
        source="flash",
        thought_source="flash",
        turn_id="t0",
        changed=True,
        owner_key="owner1",
        text_kind="private",
    )
    state = InnerState(
        revision=2,
        text="心里很踏实",
        recent_words="",
        recent_words_at=None,
        variables=raw_vars,
        felt_at=1000.0,
        updated_at=1000.0,
        processed_at=None,
        source="flash",
        thought_source="flash",
        turn_id="t1",
        changed=True,
        owner_key="owner1",
        text_kind="private",
    )
    raw_vars["tension"] = 0.99
    assert state.variables["tension"] == 0.4
    with pytest.raises(TypeError):
        state.variables["tension"] = 0.9  # type: ignore[index]

    view = InnerStateView(head=state, private=state)
    with pytest.raises(TypeError):
        view.variables["tension"] = 0.9  # type: ignore[index]

    owner_key = textutil.hermes_home(str(home))
    assert owner_key
    provider = SanaeMemProvider({
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "1001"},
    })
    provider.initialize("s-flash", platform="telegram", hermes_home=str(home))
    try:
        bound_owner = account_binding.maintainer_anchor({
            "memory_identity": {"platform": "telegram", "user_id": "1001"},
        })
        flash_res = provider.memory_axis.record_inner_state_flash(
            owner_key=bound_owner,
            user_text="今天天气真好",
            assistant_text="是呀，挺舒服的。",
            turn_id="t1",
            previous_state=prev_state,
            current_state=state,
            now=1000.0,
        )
        assert flash_res["ok"] is True
        assert flash_res["delta"] == 0.5
    finally:
        provider.shutdown()

    db_path = data_dir / "sanae.db"
    assert inner_state_store.append_state(db_path, (data_dir, "default"), prev_state) is True
    assert inner_state_store.append_state(db_path, (data_dir, "default"), state) is True
    assert inner_state_store.write_shared_state(
        db_path, "default", text="心情平和", variables=state.variables, now=1001.0, source_revision=2,
    ) is True
    inspected = inner_state_store.inspect_inner_state(db_path, "default", recent=2, owner_key="owner1")
    assert json.loads(json.dumps(inspected))["current"]["variables"]["tension"] == 0.4


def test_binding_map_group_grant_and_sources_are_immutable() -> None:
    confirmed_input = {("default", "discord", "u1", ""): "owner-a"}
    declared_input = {("default", "discord", "u1", "")}
    bm = account_binding.BindingMap(confirmed_input, declared_input)
    confirmed_input[("default", "discord", "u2", "")] = "owner-b"
    declared_input.add(("default", "discord", "u2", ""))

    assert ("default", "discord", "u2", "") not in bm.confirmed
    assert ("default", "discord", "u2", "") not in bm.declared
    with pytest.raises(TypeError):
        bm.confirmed[("default", "discord", "u3", "")] = "owner-c"  # type: ignore[index]
    with pytest.raises(AttributeError):
        bm.declared.add(("default", "discord", "u3", ""))  # type: ignore[attr-defined]

    grant_source = {"platform": "discord", "chat_id": "c1"}
    grant = group_memory.GroupGrant(
        scope_key="discord:c1",
        conversation_key="discord:c1",
        shared_owner="group:discord:c1",
        exact_owner="group:discord:c1",
        write_owner="group:discord:c1",
        read_owners=("group:discord:c1",),
        person_key="person:abc",
        source=grant_source,
    )
    grant_source["chat_id"] = "hacked"
    assert grant.source["chat_id"] == "c1"
    with pytest.raises(TypeError):
        grant.source["chat_id"] = "hacked"  # type: ignore[index]

    s1 = daily_semantic_validation.Sources()
    s2 = daily_semantic_validation.Sources()
    assert isinstance(s1.known_map, MappingProxyType)
    assert isinstance(s1.known_slots, MappingProxyType)
    assert isinstance(s1.known_dates, MappingProxyType)
    with pytest.raises(TypeError):
        s1.known_map["k"] = "v"  # type: ignore[index]
    with pytest.raises(TypeError):
        s2.known_slots["k"] = "v"  # type: ignore[index]
    with pytest.raises(TypeError):
        s2.known_dates["k"] = "v"  # type: ignore[index]

    replaced = replace(s1, known_map={"slot1": "DAILY_FACT"})
    with pytest.raises(TypeError):
        replaced.known_map["slot1"] = "OTHER"  # type: ignore[index]

    validated = daily_semantic_validation.validate_items(
        [
            {
                "kind": "DAILY_FACT",
                "action": "set",
                "text": "他今天下午去图书馆看书",
                "quote": "今天下午去图书馆看书",
            }
        ],
        message="我今天下午去图书馆看书",
        sources=s1,
        known_map={},
        today="2026-11-21",
    )
    assert len(validated) == 1


def test_verified_private_delivery_transcript_frozen_and_settles_via_public_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts_mod = pytest.importorskip("hermes_host_contracts")
    hermes_plugins = pytest.importorskip("hermes_cli.plugins")

    raw_transcript = [
        {"role": "user", "content": "帮我查一下", "platform_message_id": "in-0"},
        {"role": "assistant", "content": "已经查到了。"},
        {"role": "user", "content": "我今天下午去图书馆看书", "platform_message_id": "in-1"},
        {
            "role": "tool",
            "tool_call_id": "tc-1",
            "name": "lookup",
            "content": [{"type": "text", "text": "查到了"}],
            "error": {"code": "none", "details": [1, 2]},
        },
        {"role": "assistant", "content": "好的，祝你看书顺利。"},
    ]
    delivery = private_delivery_settler.VerifiedPrivateDelivery(
        session_id="s1",
        turn_id="t1",
        platform="telegram",
        chat_id="1001",
        chat_type="dm",
        sender_id="1001",
        user_message="我今天下午去图书馆看书",
        assistant_response="好的，祝你看书顺利。",
        delivered_at=1700000000.0,
        incoming_message_id="in-1",
        message_id="out-1",
        session_key="agent:main:telegram:dm:1001",
        transcript=tuple(raw_transcript),
    )
    raw_transcript[0]["content"] = "mutated"
    raw_transcript[3]["error"]["details"].append(99)  # type: ignore[index]

    assert delivery.transcript[0]["content"] == "帮我查一下"
    assert delivery.transcript[3]["error"]["details"] == (1, 2)
    with pytest.raises(TypeError):
        delivery.transcript[0]["content"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        delivery.transcript[3]["error"]["code"] = "x"

    tools = private_delivery_settler.extract_delivered_tool_results(delivery.transcript, turn_id="t1")
    assert len(tools) == 1
    assert json.loads(json.dumps(list(tools)))[0]["error"] == {"code": "none", "details": [1, 2]}

    daily_user_prompts: list[str] = []

    class _LoopbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages") or []
            sys_text = ""
            user_text = ""
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "")
                content = str(msg.get("content") or "")
                if role in ("system", "developer"):
                    sys_text += content
                elif role == "user":
                    user_text = content
            if "DAILY_FACT" in sys_text:
                daily_user_prompts.append(user_text)
                payload: dict[str, Any] = {
                    "items": [
                        {
                            "kind": "DAILY_FACT",
                            "action": "set",
                            "target_slot": "",
                            "event_key": "library_trip",
                            "text": "他今天下午去图书馆看书",
                            "quote": "今天下午去图书馆看书",
                            "uncertain": False,
                            "plan": False,
                        }
                    ]
                }
            elif "new_commitments" in sys_text:
                payload = {"new_commitments": [], "resolutions": []}
            else:
                payload = {
                    "sentences": ["记着他下午要去图书馆看书。"],
                    "variables": {"tension": 0.1, "closeness": 0.5, "energy": 0.2},
                    "shared_sentences": None,
                }
            resp = json.dumps({
                "id": "chatcmpl-b208a",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{port}/v1"

    home = tmp_path / "hermes_settle"
    home.mkdir()
    data_dir = home / "sanae-mem"
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "memory": {"provider": "sanae_mem"},
            "auxiliary": {
                "sanae_daily_flash": {
                    "provider": "custom",
                    "model": "test-flash",
                    "base_url": base_url,
                    "api_key": "dummy-key",
                },
                "sanae_extract": {
                    "provider": "custom",
                    "model": "test-flash",
                    "base_url": base_url,
                    "api_key": "dummy-key",
                },
            },
            "plugins": {
                "enabled": ["sanae_mem"],
                "sanae_mem": {
                    "data_dir": str(data_dir),
                    "semantic_backend": "sqlite",
                    "memory_identity": {"platform": "telegram", "user_id": "1001"},
                },
            },
        }, allow_unicode=True),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    manager = hermes_plugins.get_plugin_manager()
    manager.discover_and_load()
    ctx = hermes_plugins.PluginContext(
        hermes_plugins.PluginManifest(name="sanae_mem", source="user"),
        manager,
    )
    runtime_llm = auxiliary.register(ctx)
    auxiliary.register_daily(ctx)
    provider = SanaeMemProvider({
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "1001"},
    })
    provider.attach_runtime_llm(runtime_llm)
    env = contracts_mod.TurnEnvelope(
        platform="telegram",
        chat_type="dm",
        chat_id="1001",
        sender_id="1001",
        session_id="s1",
        session_key="agent:main:telegram:dm:1001",
        message_id="in-1",
        turn_id="t1",
        is_messaging_surface=True,
        write_origin="foreground",
    )
    try:
        with contracts_mod.turn_envelope_scope(env):
            provider.initialize("s1", platform="telegram", hermes_home=str(home), agent_context="primary")
            outcome = private_delivery_settler.settle_private_delivery(provider, delivery)
        assert outcome.settled is True
        assert outcome.daily_recorded is True
        assert outcome.commitments_updated is True
        assert outcome.inner_state_updated is True
        assert daily_user_prompts and "帮我查一下" in daily_user_prompts[0]
        with closing(sqlite3.connect(data_dir / "sanae.db")) as conn:
            daily_count = conn.execute("SELECT count(*) FROM daily_context").fetchone()[0]
            assert daily_count == 1
    finally:
        provider.shutdown()
        manager.unload()
        server.shutdown()
        server.server_close()


def test_type_hints_delivery_receipts_and_atomic_write_creates_parents(tmp_path: Path) -> None:
    assert vars(self_utterances)["Any"] is Any
    assert "session_id" in get_type_hints(self_utterances.recent_self_utterances)
    assert vars(commitments)["Tuple"] is Tuple
    assert "owner_key" in get_type_hints(commitments.Commitments.record_commitment)

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "display:\n  runtime_footer:\n    enabled: true\n    fields: [model, served_model]\n",
        encoding="utf-8",
    )
    pat = delivery_receipts.runtime_footer_pattern(home, "telegram", model="")
    assert pat is not None
    assert pat.fullmatch("gpt-5") is not None

    nested_file = tmp_path / "a" / "b" / "c" / "note.txt"
    atomic_io.atomic_write_text(nested_file, "hello")
    assert nested_file.read_text(encoding="utf-8") == "hello"
