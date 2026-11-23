from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _harness import install_plugin_copy, resolve_hermes_host_root  # noqa: E402

HERMES_ROOT = resolve_hermes_host_root()
PLUGINS_ROOT = Path(__file__).resolve().parents[2] / "plugins"
SANAE_PLUGIN_DIR = PLUGINS_ROOT / "sanae_mem"

if HERMES_ROOT is not None and str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))
if str(PLUGINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGINS_ROOT))

pytestmark = pytest.mark.skipif(
    HERMES_ROOT is None,
    reason="hermes-agent host contracts source tree not found",
)

import hermes_constants as _hc  # noqa: E402
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest  # noqa: E402
from hermes_host_contracts import (  # noqa: E402
    DeliveryReceipt,
    TurnEnvelope,
    clear_all_contract_registrations,
    emit_delivery_receipt,
    get_cognitive_prompt_policy,
    get_group_context_policy,
    get_group_reply_policy,
    get_inbound_merge_policy,
    get_memory_mutation_gate,
    turn_envelope_scope,
)
from sanae_mem.cognitive import inner_wiring  # noqa: E402
from sanae_mem import plugin_entry  # noqa: E402
from sanae_mem.governance import source_intake  # noqa: E402
from sanae_mem.cognitive.daily_context import daily_profile_key  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_host_contracts(monkeypatch):
    clear_all_contract_registrations()
    monkeypatch.delenv("HERMES_HOME", raising=False)
    yield
    clear_all_contract_registrations()


@contextlib.contextmanager
def _home_override(home: Path):
    token = _hc.set_hermes_home_override(home)
    try:
        yield home
    finally:
        _hc.reset_hermes_home_override(token)


def _write_home_config(
    home: Path,
    *,
    owner_user_id: str = "10001",
    persona_prompt: str = "I am Sanae",
    group_enabled: bool = True,
    long_term_memory: bool = True,
    short_term_sharing: bool = False,
    shadow_mode: bool = False,
    data_dir: Path | None = None,
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    target_data_dir = data_dir or (home / "sanae_data")
    plugin_cfg = {
        "data_dir": str(target_data_dir),
        "shadow": shadow_mode,
        "persona_prompt": persona_prompt,
        "memory_identity": {
            "platform": "qq",
            "user_id": owner_user_id,
        },
        "group_context": {
            "defaults": {
                "short_term_shared": group_enabled,
                "long_term_memory": long_term_memory,
                "allow_unmentioned_reply": group_enabled,
            },
        },
    }
    root_cfg = {
        "memory": {
            "provider": "sanae_mem",
        },
        "plugins": {
            "enabled": ["sanae_mem"],
            "sanae_mem": plugin_cfg,
        },
    }
    cfg_path = home / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(root_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg_path


class _RecordingLLM:
    def __init__(self, label: str, response_payload: dict[str, Any]):
        self.label = label
        self.response_payload = response_payload
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        content = json.dumps(self.response_payload, ensure_ascii=False)
        return types.SimpleNamespace(text=content, content=content)


def _build_ctx(plugin_name: str = "sanae_mem") -> tuple[PluginContext, dict[str, Any]]:
    state: dict[str, Any] = {
        "memory_factory": None,
        "unload_callbacks": [],
    }
    manifest = PluginManifest(name=plugin_name, version="0.1.0")
    manager = PluginManager()
    ctx = PluginContext(manifest=manifest, manager=manager)
    orig_reg_mem = ctx.register_memory_provider
    orig_on_unload = getattr(ctx, "on_unload", None)

    def _capture_mem(name, cls_or_factory=None, *args, **kwargs):
        # Support both register_memory_provider(name, factory) and single-arg forms
        factory = cls_or_factory if cls_or_factory is not None else name
        state["memory_factory"] = factory
        return orig_reg_mem(name, cls_or_factory, *args, **kwargs) if cls_or_factory is not None else orig_reg_mem(name)

    def _capture_unload(cb):
        state["unload_callbacks"].append(cb)
        if orig_on_unload is not None:
            return orig_on_unload(cb)
        return None

    ctx.register_memory_provider = _capture_mem  # type: ignore[method-assign]
    if hasattr(ctx, "on_unload"):
        ctx.on_unload = _capture_unload  # type: ignore[method-assign]
    return ctx, state


def _run_unload(state: dict[str, Any]) -> None:
    for cb in list(reversed(state["unload_callbacks"])):
        cb()


def _install_plugin_copy(home: Path) -> Path:
    return install_plugin_copy(home, PLUGINS_ROOT.parent, plugin_name="sanae_mem")


def test_dual_home_register_switch_unload_policies_and_status_isolation(tmp_path):
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    home_c = tmp_path / "home_c"
    home_shadow = tmp_path / "home_shadow"

    _write_home_config(
        home_a,
        owner_user_id="10001",
        persona_prompt="Persona A Original",
        group_enabled=True,
        long_term_memory=True,
    )
    _write_home_config(
        home_b,
        owner_user_id="99999",
        persona_prompt="Persona B Original",
        group_enabled=False,
        long_term_memory=False,
    )
    _write_home_config(
        home_c,
        owner_user_id="10001",
        persona_prompt="Persona C Unregistered",
        group_enabled=True,
        long_term_memory=True,
    )
    _write_home_config(
        home_shadow,
        owner_user_id="10001",
        persona_prompt="Persona Shadow",
        group_enabled=True,
        long_term_memory=True,
        shadow_mode=True,
    )

    llm_a = _RecordingLLM(
        "llm_a",
        {"allow": False, "domain": "protected_persona_or_interpersonal", "reason": "from-A"},
    )
    llm_b = _RecordingLLM(
        "llm_b",
        {"allow": True, "domain": "objective_tool_or_technical", "reason": "from-B"},
    )

    # Register HOME A
    ctx_a, state_a = _build_ctx()
    with _home_override(home_a):
        plugin_entry.register(ctx_a)
        provider_a = state_a["memory_factory"]
        provider_a._llm = llm_a

    # Register HOME B afterwards (must NOT overwrite HOME A)
    ctx_b, state_b = _build_ctx()
    with _home_override(home_b):
        plugin_entry.register(ctx_b)
        provider_b = state_b["memory_factory"]
        provider_b._llm = llm_b

    owner_dm_env = TurnEnvelope(
        platform="qq",
        chat_type="dm",
        chat_id="10001",
        sender_id="10001",
        session_key="agent:main:qq:dm:10001",
    )
    group_env = TurnEnvelope(
        platform="qq",
        chat_type="group",
        chat_id="9001",
        scope_id="default",
        sender_id="20002",
    )
    group_source = types.SimpleNamespace(
        platform="qq",
        chat_type="group",
        chat_id="9001",
        scope_id="default",
        user_id="20002",
    )
    sample_bundle = {
        "identity_parts": [],
        "help_guidance": "",
        "guidance_parts": [],
        "skills_prompt": "",
        "alibaba_identity_parts": [],
        "context_files_parts": ["WORKSPACE_CONTEXT_LINE"],
        "core_memory_parts": ["CORE_MEM_LINE"],
        "user_profile_parts": ["USER_PROFILE_LINE"],
    }

    # Switch back to HOME A: policies & gate must read HOME A live config, HOME A _llm, HOME A status.json
    with _home_override(home_a):
        prompt_policy = get_cognitive_prompt_policy()
        assert prompt_policy is not None
        assembled_a = prompt_policy.assemble_system_prompt_parts(
            agent=None,
            parts_bundle=sample_bundle,
            system_message="hello",
            envelope=owner_dm_env,
        )
        assert assembled_a is not None
        # owner_user_id == "10001" -> private_maintainer -> workspace & core memory allowed
        assert "WORKSPACE_CONTEXT_LINE" in assembled_a.get("context", "")
        assert "CORE_MEM_LINE" in assembled_a.get("context", "")

        assert get_group_context_policy()(group_source) is True
        turn_policy_a = prompt_policy.resolve_turn_policy(group_env)
        assert turn_policy_a is not None
        allowed_a = set(getattr(turn_policy_a, "allowed_tools", ()) or ())
        assert "sanae_mem_search" in allowed_a
        assert "sanae_mem_add" not in allowed_a
        assert "sanae_impression" not in allowed_a

        # Mutate HOME A config on disk without re-registering: live reload must reflect change
        _write_home_config(
            home_a,
            owner_user_id="88888",  # 10001 is no longer maintainer in HOME A!
            persona_prompt="Persona A Edited Live",
            group_enabled=False,
            long_term_memory=False,
        )
        assembled_a_edited = prompt_policy.assemble_system_prompt_parts(
            agent=None,
            parts_bundle=sample_bundle,
            system_message="hello",
            envelope=owner_dm_env,
        )
        assert assembled_a_edited is not None
        assert "WORKSPACE_CONTEXT_LINE" not in assembled_a_edited.get("context", "")
        assert "CORE_MEM_LINE" not in assembled_a_edited.get("context", "")
        assert get_group_context_policy()(group_source) is False
        turn_policy_a_edited = prompt_policy.resolve_turn_policy(group_env)
        assert turn_policy_a_edited is not None
        assert "sanae_mem_add" not in set(getattr(turn_policy_a_edited, "allowed_tools", ()) or ())

        gate = get_memory_mutation_gate()
        assert gate is not None
        with turn_envelope_scope(owner_dm_env):
            verdict_a = gate.evaluate_skill_mutation(
                action="create",
                name="daily_checkin_helper",
                args={"content": "每次见面都温柔提醒她别熬夜"},
                envelope=owner_dm_env,
            )
        assert verdict_a is not None
        assert verdict_a.allow is False
        assert len(llm_a.calls) == 1
        assert len(llm_b.calls) == 0

        status_a_path = home_a / "sanae_data" / "status.json"
        status_b_path = home_b / "sanae_data" / "status.json"
        assert status_a_path.exists()
        assert not status_b_path.exists()
        status_a = json.loads(status_a_path.read_text(encoding="utf-8"))
        assert status_a.get("components", {}).get("skill_mutation_review", {}).get("ok") is True

    # Switch to HOME B: must use HOME B config, HOME B _llm, HOME B status.json
    with _home_override(home_b):
        prompt_policy_b = get_cognitive_prompt_policy()
        assert prompt_policy_b is not None
        env_b = TurnEnvelope(
            platform="qq",
            chat_type="dm",
            chat_id="99999",
            sender_id="99999",
            session_key="agent:main:qq:dm:99999",
        )
        assembled_b = prompt_policy_b.assemble_system_prompt_parts(
            agent=None,
            parts_bundle=sample_bundle,
            system_message="hello",
            envelope=env_b,
        )
        assert assembled_b is not None
        assert "WORKSPACE_CONTEXT_LINE" in assembled_b.get("context", "")
        assert "CORE_MEM_LINE" in assembled_b.get("context", "")
        assert get_group_context_policy()(group_source) is False

        gate_b = get_memory_mutation_gate()
        assert gate_b is not None
        with turn_envelope_scope(env_b):
            verdict_b = gate_b.evaluate_skill_mutation(
                action="create",
                name="docker_compose_helper",
                args={"content": "docker compose up -d helper note"},
                envelope=env_b,
            )
        assert verdict_b is None  # allow=True returns None (pass)
        assert len(llm_b.calls) == 1
        assert len(llm_a.calls) == 1

        status_b = json.loads((home_b / "sanae_data" / "status.json").read_text(encoding="utf-8"))
        assert status_b.get("components", {}).get("skill_mutation_review", {}).get("ok") is True

    # Unregistered HOME C & shadow HOME must not borrow HOME A/B policies or models
    for isolated_home in (home_c, home_shadow):
        with _home_override(isolated_home):
            p = get_cognitive_prompt_policy()
            if p is not None:
                assert p.assemble_system_prompt_parts(
                    agent=None,
                    parts_bundle=sample_bundle,
                    system_message="hello",
                    envelope=owner_dm_env,
                ) is None
                assert p.resolve_turn_policy(group_env) is None
            assert get_group_context_policy()(group_source) is None
            assert get_group_reply_policy()(group_source) is None
            assert get_inbound_merge_policy().should_debounce_event({}, "qq", "dm", busy=False) is None
            g = get_memory_mutation_gate()
            if g is not None:
                assert g.evaluate_skill_mutation(
                    action="create",
                    name="some_tech_note",
                    args={"content": "plain script note"},
                ) is None
            assert len(llm_a.calls) == 1
            assert len(llm_b.calls) == 1

    # Unload HOME B: HOME A remains active; HOME B stops serving
    _run_unload(state_b)
    with _home_override(home_a):
        res = get_cognitive_prompt_policy().assemble_system_prompt_parts(
            agent=None,
            parts_bundle=sample_bundle,
            system_message="hello",
            envelope=owner_dm_env,
        )
        assert res is not None and "USER_PROFILE_LINE" in res.get("context", "")
    with _home_override(home_b):
        p_b_after = get_cognitive_prompt_policy()
        if p_b_after is not None:
            assert p_b_after.assemble_system_prompt_parts(
                agent=None,
                parts_bundle=sample_bundle,
                system_message="hello",
                envelope=owner_dm_env,
            ) is None

    # Unload last HOME (HOME A): all 5 policies cleanly unregister
    _run_unload(state_a)
    assert get_cognitive_prompt_policy() is None
    assert get_memory_mutation_gate() is None
    assert get_inbound_merge_policy() is None
    assert get_group_context_policy() is None
    assert get_group_reply_policy() is None


def test_dual_home_plugin_manager_separate_modules_and_source_intake_restore(tmp_path):
    home_a = tmp_path / "pm_home_a"
    home_b = tmp_path / "pm_home_b"
    home_shadow = tmp_path / "pm_home_shadow"

    _install_plugin_copy(home_a)
    _install_plugin_copy(home_b)
    _write_home_config(home_a, owner_user_id="10001", persona_prompt="PM Persona A")
    _write_home_config(home_b, owner_user_id="10002", persona_prompt="PM Persona B")
    _write_home_config(home_shadow, owner_user_id="10003", shadow_mode=True)

    with _home_override(home_a):
        pm_a = PluginManager(scope_key=str(home_a))
    with _home_override(home_b):
        pm_b = PluginManager(scope_key=str(home_b))

    try:
        with _home_override(home_a):
            pm_a.discover_and_load()
        with _home_override(home_b):
            pm_b.discover_and_load()

        # Verify distinct plugin modules were loaded for the two HOMEs
        loaded_mods = {
            name: mod
            for name, mod in sys.modules.items()
            if (name == "hermes_plugins.sanae_mem" or name.startswith("hermes_plugins.sanae_mem__home_"))
            and mod is not None
        }
        assert len(loaded_mods) >= 2
        assert len({id(m) for m in loaded_mods.values()}) >= 2

        env_a = TurnEnvelope(
            platform="qq", chat_type="dm", chat_id="10001", sender_id="10001",
            session_key="agent:main:qq:dm:10001",
        )
        env_b = TurnEnvelope(
            platform="qq", chat_type="dm", chat_id="10002", sender_id="10002",
            session_key="agent:main:qq:dm:10002",
        )
        env_shadow = TurnEnvelope(
            platform="qq", chat_type="dm", chat_id="10003", sender_id="10003",
            session_key="agent:main:qq:dm:10003",
        )

        # Ambient HOME is B, yet explicit target HOME A routes to HOME A DB and live config
        with _home_override(home_b), turn_envelope_scope(env_a):
            verdict_a = source_intake.submit_source_material(
                source_intake.SourceMaterial(
                    source_plugin="bilibili_public",
                    platform="bilibili",
                    material_kind="processed",
                    processing_type="summary",
                    content="Target HOME A explicit source note",
                    title="note-a",
                    source_object_id="obj-a-1",
                ),
                create_memory_event=True,
                hermes_home_path=home_a,
            )
            assert verdict_a["ok"] is True

        with _home_override(home_a), turn_envelope_scope(env_b):
            verdict_b = source_intake.submit_source_material(
                source_intake.SourceMaterial(
                    source_plugin="bilibili_public",
                    platform="bilibili",
                    material_kind="processed",
                    processing_type="summary",
                    content="Target HOME B explicit source note",
                    title="note-b",
                    source_object_id="obj-b-1",
                ),
                create_memory_event=True,
                hermes_home_path=home_b,
            )
            assert verdict_b["ok"] is True

        # Unload HOME B while HOME A continues to use the same public module.
        with _home_override(home_b):
            pm_b.unload("sanae_mem")

        with _home_override(home_a), turn_envelope_scope(env_a):
            from sanae_mem.governance import source_intake as live_intake

            assert live_intake is source_intake
            verdict_a2 = live_intake.submit_source_material(
                live_intake.SourceMaterial(
                    source_plugin="bilibili_public",
                    platform="bilibili",
                    material_kind="processed",
                    processing_type="summary",
                    content="Second note after HOME B unloaded",
                    title="note-a2",
                    source_object_id="obj-a-2",
                ),
                create_memory_event=True,
                hermes_home_path=home_a,
            )
            assert verdict_a2["ok"] is True

        # Shadow target should not write into HOME A or HOME B DBs
        with turn_envelope_scope(env_shadow):
            verdict_shadow = source_intake.submit_source_material(
                source_intake.SourceMaterial(
                    source_plugin="bilibili_public",
                    platform="bilibili",
                    material_kind="processed",
                    processing_type="summary",
                    content="Should be blocked in shadow mode",
                    title="shadow-note",
                    source_object_id="obj-shadow-1",
                ),
                create_memory_event=True,
                hermes_home_path=home_shadow,
            )
            assert verdict_shadow["ok"] is False

        assert not (home_shadow / "sanae_data" / "sanae.db").exists()
        with sqlite3.connect(home_a / "sanae_data" / "sanae.db") as conn_a:
            rows_a = conn_a.execute("SELECT summary FROM events").fetchall()
        with sqlite3.connect(home_b / "sanae_data" / "sanae.db") as conn_b:
            rows_b = conn_b.execute("SELECT summary FROM events").fetchall()

        text_a = "\n".join(str(r[0]) for r in rows_a)
        text_b = "\n".join(str(r[0]) for r in rows_b)
        assert "Target HOME A" in text_a
        assert "Second note after HOME B unloaded" in text_a
        assert "Target HOME B" not in text_a
        assert "Target HOME B" in text_b
        assert "Target HOME A" not in text_b
    finally:
        with contextlib.suppress(Exception):
            with _home_override(home_b):
                pm_b.unload("sanae_mem")
        with contextlib.suppress(Exception):
            with _home_override(home_a):
                pm_a.unload("sanae_mem")


def test_loaded_sessions_and_material_submission_follow_live_shadow_per_home(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from plugins.memory import load_memory_provider

    homes = [tmp_path / "shadow-start", tmp_path / "normal-start"]
    managers = []
    providers = []
    sid = "same-session-id"
    source = SessionSource(platform=Platform.QQBOT, chat_type="group", chat_id="9001",
                           scope_id="default", user_id="group-user")

    def configure(index, shadow):
        path = _write_home_config(homes[index], owner_user_id=str(10001 + index), shadow_mode=shadow)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["plugins"]["sanae_mem"]["semantic_backend"] = "sqlite"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def envelope(index):
        owner = str(10001 + index)
        return TurnEnvelope(platform="qq", chat_type="dm", chat_id=owner, sender_id=owner,
                            session_id=sid, session_key=f"agent:main:qq:dm:{owner}")

    def probe(index, label, active):
        home = homes[index]
        with _home_override(home), turn_envelope_scope(envelope(index)):
            policy = get_group_context_policy()
            assert policy is not None and policy(source) is (True if active else None)
            resolved = plugin_entry.resolve_active_provider(home, sid)
            assert resolved is (providers[index] if active else None)
            result = plugin_entry.submit_source_material(
                {"source_plugin": "bilibili_public", "platform": "bilibili",
                 "material_kind": "processed", "processing_type": "summary",
                 "source_object_id": label, "content": label},
                session_id=sid,
            )
            assert result["ok"] is active, result
            if not active:
                assert result["reason"] == "provider_shadow_mode"

    try:
        for index, home in enumerate(homes):
            _install_plugin_copy(home)
            configure(index, shadow=index == 0)
            with _home_override(home), turn_envelope_scope(envelope(index)):
                manager = PluginManager(scope_key=str(home))
                managers.append(manager)
                manager.discover_and_load()
                assert any(row["name"] == "sanae_mem" and row["enabled"] for row in manager.list_plugins())
                provider = load_memory_provider("sanae_mem", register_skills=False)
                assert provider is not None
                providers.append(provider)
                provider.initialize(sid, platform="qq", hermes_home=str(home))
        assert type(providers[0]) is not type(providers[1])
        probe(0, "a-initial-shadow", False)
        probe(1, "b-initial-normal", True)
        configure(0, shadow=False)
        probe(0, "a-first-resume", True)
        configure(1, shadow=True)
        probe(1, "b-paused", False)
        configure(0, shadow=True)
        probe(0, "a-paused", False)
        configure(0, shadow=False)
        probe(0, "a-second-resume", True)
        configure(1, shadow=False)
        probe(1, "b-resumed", True)

        for home, expected in zip(homes, [{"a-first-resume", "a-second-resume"},
                                         {"b-initial-normal", "b-resumed"}]):
            with contextlib.closing(sqlite3.connect(home / "sanae_data" / "sanae.db")) as connection:
                assert {row[0] for row in connection.execute("SELECT content FROM source_materials")} == expected
    finally:
        for home, provider in zip(homes, providers):
            with _home_override(home):
                provider.shutdown()
        for home, manager in zip(homes, managers):
            with _home_override(home):
                manager.unload("sanae_mem")


def test_external_group_builtin_notes_guard_follows_the_active_home_policy(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from hermes_cli.plugins import get_plugin_manager
    from plugins.memory import load_memory_provider

    from sanae_mem.governance.memory_guard import PUBLIC_MEMORY_PROMPT_BLOCK_MESSAGE

    home_a, home_b = tmp_path / "cold-shadow", tmp_path / "adjacent-normal"
    managers: dict[Path, PluginManager] = {}
    held_provider = None
    platform = Platform.QQBOT
    markers = ("SYNTHETIC_BUILTIN_NOTE_A", "SYNTHETIC_PROFILE_NOTE_B")
    owner = "10001"

    def configure(home, shadow):
        path = _write_home_config(home, shadow_mode=shadow)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["memory"].update(memory_enabled=True, user_profile_enabled=True)
        config["plugins"]["sanae_mem"]["semantic_backend"] = "sqlite"
        config["plugins"]["sanae_mem"]["memory_identity"]["platform"] = platform.value
        path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def pre_results(home, source):
        session_key = f"agent:main:{platform.value}:{source.chat_type}:{source.chat_id}"
        with turn_envelope_scope(TurnEnvelope(
            platform=platform.value, chat_type=source.chat_type, chat_id=source.chat_id,
            sender_id=source.user_id, session_id="guard-session", session_key=session_key,
        )):
            return [reply for reply in managers[home].invoke_hook(
                "pre_llm_call", session_id="guard-session", turn_id="guard-turn",
                platform=platform.value, user_message="合成测试消息",
            ) if isinstance(reply, dict) and reply.get("action") == "block"]

    group = SessionSource(platform=platform, chat_type="group", chat_id="synthetic-group", user_id="visitor")
    maintainer = SessionSource(platform=platform, chat_type="dm", chat_id=owner, user_id=owner)
    expected_vetoes = [{"action": "block", "message": PUBLIC_MEMORY_PROMPT_BLOCK_MESSAGE}]

    def probe(home, *, blocked, source=group):
        with _home_override(home):
            admission = managers[home].invoke_hook(
                "post_gateway_admission", source=source.to_dict(), text="合成测试消息",
            )
            expected = [{"action": "handled", "reply": PUBLIC_MEMORY_PROMPT_BLOCK_MESSAGE}]
            assert admission == (expected if blocked else [])
            assert pre_results(home, source) == (expected_vetoes if blocked else [])

    try:
        for home in (home_a, home_b):
            _install_plugin_copy(home)
            configure(home, shadow=home == home_a)
            memory_dir = home / "memories"
            memory_dir.mkdir()
            for name, marker in zip(("MEMORY.md", "USER.md"), markers):
                (memory_dir / name).write_text(marker, encoding="utf-8")
            with _home_override(home):
                managers[home] = get_plugin_manager()
                if home == home_a:
                    held_provider = load_memory_provider("sanae_mem", register_skills=False)
                    assert held_provider is not None
                    assert pre_results(home, group) == expected_vetoes
                managers[home].discover_and_load()
                assert any(row["name"] == "sanae_mem" and row["enabled"]
                           for row in managers[home].list_plugins())
        probe(home_a, blocked=True)
        probe(home_b, blocked=False)
        for shadow in (False, True, False):
            configure(home_a, shadow=shadow)
            probe(home_a, blocked=shadow)
            probe(home_b, blocked=False)
        probe(home_a, blocked=False, source=maintainer)
    finally:
        if held_provider is not None:
            with _home_override(home_a):
                held_provider.shutdown()
        for home, manager in managers.items():
            with _home_override(home):
                manager.unload()


def test_receipt_broadcast_reads_payload_only_for_eligible_owner(tmp_path):
    from hermes_cli.plugins import get_plugin_manager
    from hermes_host_contracts import register_delivery_receipt_handler, unregister_delivery_receipt_handler
    from plugins.memory import load_memory_provider
    from sanae_mem.runtime.delivery import DeliveryRouter

    reads: list[str] = []

    class ObservedReceipt(DeliveryReceipt):
        def __getattribute__(self, field):
            if field in {"delivered_text", "user_message", "transcript"}:
                reads.append(field)
            return super().__getattribute__(field)

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    managers = []
    collector = None

    def broadcast(message="first", *, confirmed=True, cron=False):
        reads.clear()
        receipt = ObservedReceipt(
            session_id="group-session", platform="qq", chat_id="test-group", chat_type="group",
            sender_id="sender", delivered_at=1700000000, delivered_text="收到。",
            incoming_message_id=f"in-{message}", message_id=f"out-{message}",
            user_message="我们周六一起看电影。", transport_confirmed=confirmed, is_cron=cron,
        )
        return emit_delivery_receipt(receipt)

    try:
        for home in (home_a, home_b):
            _install_plugin_copy(home)
            _write_home_config(home, long_term_memory=False)
            with _home_override(home):
                manager = get_plugin_manager()
                managers.append(manager)
                manager.discover_and_load()
                assert any(row["name"] == "sanae_mem" and row["enabled"] for row in manager.list_plugins())
        with _home_override(home_a):
            collector = load_memory_provider("sanae_mem", register_skills=False)
            assert collector is not None
            assert broadcast() == 3
            assert sorted(reads) == ["delivered_text", "transcript", "user_message"]
            assert broadcast() == 3
            assert sorted(reads) == ["delivered_text", "user_message"]
            assert broadcast(confirmed=False) == 0 and reads == []
            assert broadcast(cron=True) == 3 and reads == []

            incoming_provider = plugin_entry.SanaeMemProvider()
            incoming_router = DeliveryRouter(home_a.resolve(), incoming_provider, discovery=True)
            incoming_handler = incoming_router.on_transport_receipt
            register_delivery_receipt_handler(incoming_handler)
            try:
                assert broadcast("registration-window") == 4
                assert sorted(reads) == ["delivered_text", "transcript", "user_message"]
                release_owner = incoming_router.attach()
                try:
                    assert broadcast("registration-window") == 4
                    assert sorted(reads) == ["delivered_text", "user_message"]
                finally:
                    release_owner()
                assert broadcast("release-window") == 4
                assert sorted(reads) == ["delivered_text", "transcript", "user_message"]
            finally:
                unregister_delivery_receipt_handler(incoming_handler)
                incoming_provider.shutdown()

            _write_home_config(home_a, long_term_memory=False, shadow_mode=True)
            assert broadcast() == 3 and reads == []
            _write_home_config(home_a, long_term_memory=False)

            assert managers[0].unload("sanae_mem")
            assert broadcast("after-unload") == 2
            assert sorted(reads) == ["delivered_text", "transcript", "user_message"]
            collector.shutdown()
            collector = None
            assert broadcast() == 1 and reads == []
    finally:
        if collector is not None:
            with _home_override(home_a):
                collector.shutdown()
        for home, manager in zip((home_a, home_b), managers):
            with _home_override(home):
                manager.unload()


def test_dual_home_background_group_receipt_uses_live_config_own_conn_and_own_llm(tmp_path):
    home_a = tmp_path / "bg_home_a"
    home_b = tmp_path / "bg_home_b"

    # Start HOME A with long_term_memory=False in prototype config, then flip True on disk AFTER register()
    _write_home_config(home_a, owner_user_id="10001", group_enabled=True, long_term_memory=False)
    _write_home_config(home_b, owner_user_id="10002", group_enabled=True, long_term_memory=True)

    llm_a = _RecordingLLM(
        "bg_llm_a",
        {
            "events": [{"text": "在A群约定周六看电影", "quote": "周六一起去看新电影吧"}],
            "impression": {"speaker": "user", "quote": "周六一起去看新电影吧", "text": "喜欢周末约片"},
            "new_commitments": [{"text": "周六一起看电影", "quote": "周六我陪你们去看"}],
            "commitment_updates": [],
        },
    )
    llm_b = _RecordingLLM(
        "bg_llm_b",
        {
            "events": [{"text": "在B群讨论周三上线", "quote": "B群周三上线记得盯一下"}],
            "impression": None,
            "new_commitments": [],
            "commitment_updates": [],
        },
    )

    class _PoisonConn:
        def __getattr__(self, name: str):
            raise AssertionError(f"prototype._conn.{name} must not be touched by background receipt worker")

    ctx_a, state_a = _build_ctx()
    with _home_override(home_a):
        plugin_entry.register(ctx_a)
        provider_a = state_a["memory_factory"]
        provider_a._llm = llm_a
        provider_a._conn = _PoisonConn()

    ctx_b, state_b = _build_ctx()
    with _home_override(home_b):
        plugin_entry.register(ctx_b)
        provider_b = state_b["memory_factory"]
        provider_b._llm = llm_b
        provider_b._conn = _PoisonConn()

    try:
        # Enable HOME A long_term_memory on disk AFTER register(), while provider_a._config stays stale!
        _write_home_config(home_a, owner_user_id="10001", group_enabled=True, long_term_memory=True)

        # Emit HOME A receipt under HOME A, then wait while ambient HERMES_HOME immediately switches to HOME B
        receipt_a = DeliveryReceipt(
            session_id="qq:group:778899",
            platform="qq",
            chat_type="group",
            chat_id="778899",
            scope_id="default",
            sender_id="u_alpha",
            turn_id="turn-a-1",
            message_id="msg-out-a-1",
            incoming_message_id="msg-in-a-1",
            user_message="周六一起去看新电影吧",
            delivered_text="好呀，周六我陪你们去看。",
            delivered_at=1700000000.0,
            transport_confirmed=True,
        )
        envelope_a = TurnEnvelope(
            platform="qq",
            chat_type="group",
            chat_id="778899",
            scope_id="default",
            sender_id="u_alpha",
            sender_name="群友甲",
        )
        with _home_override(home_a):
            with turn_envelope_scope(envelope_a):
                emit_delivery_receipt(receipt_a)

        with _home_override(home_b):
            inner_wiring._wait_for_pending(((home_a / "sanae_data").resolve(), daily_profile_key(home_a)))
        assert len(llm_a.calls) >= 1
        assert len(llm_b.calls) == 0

        with sqlite3.connect(home_a / "sanae_data" / "sanae.db") as conn_a:
            ev_a = conn_a.execute("SELECT summary FROM events").fetchall()
            assert any("周六看电影" in str(r[0]) for r in ev_a)

        if (home_b / "sanae_data" / "sanae.db").exists():
            with sqlite3.connect(home_b / "sanae_data" / "sanae.db") as conn_b:
                ev_b = conn_b.execute("SELECT summary FROM events").fetchall()
                assert ev_b == []

        # Flip HOME A long_term_memory=False on disk and emit another HOME A receipt -> must skip LLM & DB write
        _write_home_config(home_a, owner_user_id="10001", group_enabled=True, long_term_memory=False)
        calls_before = len(llm_a.calls)
        receipt_a2 = DeliveryReceipt(
            session_id="qq:group:778899",
            platform="qq",
            chat_type="group",
            chat_id="778899",
            scope_id="default",
            sender_id="u_alpha",
            turn_id="turn-a-2",
            message_id="msg-out-a-2",
            incoming_message_id="msg-in-a-2",
            user_message="周日再聚餐？",
            delivered_text="周日看情况再定。",
            delivered_at=1700000060.0,
            transport_confirmed=True,
        )
        with _home_override(home_a):
            with turn_envelope_scope(envelope_a):
                emit_delivery_receipt(receipt_a2)
        with _home_override(home_b):
            inner_wiring._wait_for_pending(((home_a / "sanae_data").resolve(), daily_profile_key(home_a)))
        assert len(llm_a.calls) == calls_before

        # Emit HOME B receipt under HOME B and wait while ambient HOME is HOME A -> routes strictly to HOME B LLM & DB
        receipt_b = DeliveryReceipt(
            session_id="qq:group:556677",
            platform="qq",
            chat_type="group",
            chat_id="556677",
            scope_id="default",
            sender_id="u_beta",
            turn_id="turn-b-1",
            message_id="msg-out-b-1",
            incoming_message_id="msg-in-b-1",
            user_message="B群周三上线记得盯一下",
            delivered_text="收到，周三上线我会盯着。",
            delivered_at=1700000120.0,
            transport_confirmed=True,
        )
        envelope_b = TurnEnvelope(
            platform="qq",
            chat_type="group",
            chat_id="556677",
            scope_id="default",
            sender_id="u_beta",
            sender_name="群友乙",
        )
        with _home_override(home_b):
            with turn_envelope_scope(envelope_b):
                emit_delivery_receipt(receipt_b)
        with _home_override(home_a):
            inner_wiring._wait_for_pending(((home_b / "sanae_data").resolve(), daily_profile_key(home_b)))
        assert len(llm_b.calls) >= 1
        with sqlite3.connect(home_b / "sanae_data" / "sanae.db") as conn_b:
            ev_b = conn_b.execute("SELECT summary FROM events").fetchall()
            assert any("B群" in str(r[0]) or "周三上线" in str(r[0]) for r in ev_b)
    finally:
        _run_unload(state_b)
        _run_unload(state_a)


def test_dual_home_reverse_unload_order_keeps_second_home_alive(tmp_path):
    home_a = tmp_path / "rev_home_a"
    home_b = tmp_path / "rev_home_b"
    _write_home_config(home_a, owner_user_id="10001", group_enabled=True, long_term_memory=True)
    _write_home_config(home_b, owner_user_id="20002", group_enabled=False, long_term_memory=False)

    ctx_a, state_a = _build_ctx()
    with _home_override(home_a):
        plugin_entry.register(ctx_a)

    ctx_b, state_b = _build_ctx()
    with _home_override(home_b):
        plugin_entry.register(ctx_b)

    group_source = types.SimpleNamespace(platform="qq", chat_type="group", chat_id="1", scope_id="default", user_id="2")
    # Unload HOME A first: HOME B must stay active, HOME A must be gone
    _run_unload(state_a)
    with _home_override(home_a):
        assert get_group_context_policy()(group_source) is None
    with _home_override(home_b):
        assert get_group_context_policy()(group_source) is False

    # Now unload HOME B: all policies clean up
    _run_unload(state_b)
    assert get_cognitive_prompt_policy() is None
    assert get_memory_mutation_gate() is None
    assert get_inbound_merge_policy() is None
    assert get_group_context_policy() is None
    assert get_group_reply_policy() is None
