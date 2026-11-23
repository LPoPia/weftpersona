"""Integration checks against the pinned Hermes session context and memory hooks."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from contextlib import contextmanager
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
PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
if HERMES_ROOT.is_dir():
    sys.path.insert(0, str(HERMES_ROOT))
    sys.path.insert(0, str(PLUGIN_ROOT))

if not HERMES_ROOT.is_dir():
    pytest.skip("pinned Hermes source vivemind-hermes-6093627 is unavailable", allow_module_level=True)

from gateway import session_context as _session_context
from agent.memory_manager import MemoryManager
from hermes_cli import config as hermes_config
from hermes_cli import plugins as hermes_plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from sanae_mem import plugin_entry
from sanae_mem.governance import memory_guard
from sanae_mem.runtime import hooks as runtime_hooks
from sanae_mem.governance.memory_guard import (
    LOCAL_REFERENCE_BLOCK_MESSAGE,
    MIXED_SESSION_SEARCH_BLOCK_MESSAGE,
    PUBLIC_MEMORY_PROMPT_BLOCK_MESSAGE,
    PUBLIC_MEMORY_TOOL_BLOCK_MESSAGE,
)
from sanae_mem.governance.source_context import current_chat_scope

assert Path(_session_context.__file__).resolve().is_relative_to(HERMES_ROOT.resolve())
clear_session_vars = _session_context.clear_session_vars
reset_session_vars = _session_context.reset_session_vars
set_session_vars = _session_context.set_session_vars


@contextmanager
def _route(*, actor: str, session: str, chat_type: str = "dm", chat_id: str | None = None):
    reset_session_vars()
    tokens = set_session_vars(
        platform="telegram",
        chat_id=chat_id or f"chat-{session}",
        chat_type=chat_type,
        user_id=actor,
        session_id=session,
        session_key=f"agent:main:telegram:{chat_type}:{session}",
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    memories = home / "memories"
    memories.mkdir(parents=True)
    (memories / "USER.md").write_text("A_PROFILE_PRIVATE_NOTE", encoding="utf-8")
    (memories / "MEMORY.md").write_text("PLUGIN_MEMORY_PRIVATE_NOTE", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    data_dir = tmp_path / "sanae-mem"
    config = {
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "account-A"},
        "partner_name": "测试账号 A",
    }
    (home / "config.yaml").write_text(json.dumps({
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "plugins": {"sanae_mem": config},
    }), encoding="utf-8")
    provider_type = plugin_entry.SanaeMemProvider
    monkeypatch.setattr(hermes_config, "load_config", lambda: {"memory": {}})

    def make_provider(actor, session):
        provider = provider_type(config)
        manager = MemoryManager()
        with _route(actor=actor, session=session):
            manager.add_provider(provider)
            assert manager.has_tool("sanae_mem_add")
            assert manager.has_tool("sanae_mem_search")
            manager.initialize_all(
                session_id=session,
                hermes_home=str(home),
                agent_context="primary",
                platform="telegram",
            )
            assert manager.has_tool("sanae_mem_add")
            assert manager.has_tool("sanae_mem_search")
        return provider, manager

    provider_a, memory_manager_a = make_provider("account-A", "private-A")
    hook_manager = PluginManager(scope_key=str(home))
    plugin_context = PluginContext(PluginManifest(name="sanae_mem", source="user"), hook_manager)
    monkeypatch.setattr(plugin_entry, "SanaeMemProvider", lambda: provider_a)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register_daily", lambda _ctx: None)
    plugin_entry.register(plugin_context)
    monkeypatch.setattr(plugin_entry, "SanaeMemProvider", provider_type)

    providers = [provider_a]
    memory_managers = [memory_manager_a]
    result = {
        "home": home,
        "provider_a": provider_a,
        "memory_manager_a": memory_manager_a,
        "make_provider": make_provider,
        "providers": providers,
        "memory_managers": memory_managers,
        "plugin_context": plugin_context,
        "hook_manager": hook_manager,
    }
    yield result
    for provider in providers:
        if provider._conn is not None:
            provider._conn.close()
            provider._conn = None


def _tool(provider, name, args, memory_manager=None):
    result = (memory_manager.handle_tool_call(name, args) if memory_manager is not None
              else provider.handle_tool_call(name, args))
    return json.loads(result)


def test_private_group_unknown_routes_are_isolated_in_plugin_sql_and_context(world):
    provider_a = world["provider_a"]
    with _route(actor="account-A", session="private-A"):
        assert current_chat_scope().visibility == "private"
        assert {schema["name"] for schema in provider_a.get_tool_schemas()} >= {
            "sanae_mem_add", "sanae_mem_search",
        }
        a_write = _tool(provider_a, "sanae_mem_add", {
            "content": "A's private honeycrisp compass is stored in locker nine.",
        }, world["memory_manager_a"])
        assert a_write["ok"] and a_write["result"]["stored"]
        a_prompt = provider_a.system_prompt_block()
        assert "A_PROFILE_PRIVATE_NOTE" in a_prompt
        assert "PLUGIN_MEMORY_PRIVATE_NOTE" in a_prompt
        a_search = _tool(provider_a, "sanae_mem_search", {"query": "honeycrisp compass"}, world["memory_manager_a"])
        assert "honeycrisp compass" in json.dumps(a_search, ensure_ascii=False)

    public_id = provider_a.ingest.write_event(
        "PRIVATE_ORIGINAL_RAW_COPPER_RIVER",
        "reviewed-public-event",
        importance=0.9,
        source="explicit",
        source_context={"kind": "self", "platform": "telegram", "visibility": "public"},
        experience_subject="sanae",
        fact_subject="sanae",
        involved_owner_keys="[]",
        read_scope='{"kind":"self"}',
    )
    assert public_id is not None
    assert provider_a.event_store.set_public_excerpt(public_id, "copper river public approved excerpt")

    before_group_write = provider_a._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    with _route(actor="account-A", session="group-1", chat_type="group", chat_id="group-1"):
        assert current_chat_scope().visibility == "public"
        assert "A_PROFILE_PRIVATE_NOTE" not in provider_a.system_prompt_block()
        assert {schema["name"] for schema in provider_a.get_tool_schemas()} == {"sanae_mem_search"}

        group_search = _tool(provider_a, "sanae_mem_search", {"query": "copper river"}, world["memory_manager_a"])
        rendered = json.dumps(group_search, ensure_ascii=False)
        assert "copper river public approved excerpt" in rendered
        assert "PRIVATE_ORIGINAL_RAW_COPPER_RIVER" not in rendered
        assert "honeycrisp compass" not in rendered
        assert "violet bicycle" not in rendered

        group_prefetch = provider_a.prefetch("copper river", session_id="group-1")
        assert "copper river public approved excerpt" in group_prefetch
        assert "PRIVATE_ORIGINAL_RAW_COPPER_RIVER" not in group_prefetch

        rejected = _tool(provider_a, "sanae_mem_add", {"content": "group must not write memory"}, world["memory_manager_a"])
        assert not rejected["ok"]
        assert not _tool(
            provider_a, "sanae_meme_search", {"query": "group-private-meme"}, world["memory_manager_a"]
        )["ok"]
        assert not _tool(provider_a, "sanae_inner_state", {}, world["memory_manager_a"])["ok"]
    after_group_write = provider_a._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert after_group_write == before_group_write

    with _route(actor="account-A", session="private-A"):
        a_search = _tool(provider_a, "sanae_mem_search", {"query": "honeycrisp compass"}, world["memory_manager_a"])
        assert "honeycrisp compass" in json.dumps(a_search, ensure_ascii=False)

    provider_b, memory_manager_b = world["make_provider"]("account-B", "private-B")
    world["providers"].append(provider_b)
    world["memory_managers"].append(memory_manager_b)
    with _route(actor="account-B", session="private-B"):
        assert current_chat_scope().visibility == "private"
        b_write = _tool(provider_b, "sanae_mem_add", {
            "content": "B's private violet bicycle is kept in room four.",
        }, memory_manager_b)
        assert b_write["ok"] and b_write["result"]["stored"]
        b_prompt = provider_b.system_prompt_block()
        assert "A_PROFILE_PRIVATE_NOTE" not in b_prompt
        assert "PLUGIN_MEMORY_PRIVATE_NOTE" not in b_prompt
        b_search = _tool(provider_b, "sanae_mem_search", {"query": "violet bicycle"}, memory_manager_b)
        assert "violet bicycle" in json.dumps(b_search, ensure_ascii=False)
        assert "honeycrisp compass" not in json.dumps(b_search, ensure_ascii=False)

    with _route(actor="account-A", session="private-A"):
        a_search = _tool(provider_a, "sanae_mem_search", {"query": "honeycrisp compass"}, world["memory_manager_a"])
        assert "honeycrisp compass" in json.dumps(a_search, ensure_ascii=False)
        assert "violet bicycle" not in json.dumps(a_search, ensure_ascii=False)

    rows = provider_a._conn.execute(
        "SELECT owner_key, raw, public_excerpt FROM events ORDER BY id"
    ).fetchall()
    owner_rows = {row["owner_key"]: row["raw"] for row in rows if row["owner_key"]}
    assert owner_rows[provider_a._owner_key].find("honeycrisp compass") >= 0
    assert owner_rows[provider_b._owner_key].find("violet bicycle") >= 0
    assert provider_a._owner_key != provider_b._owner_key

    with _route(actor="account-A", session="unknown-1", chat_type="unrecognized"):
        assert current_chat_scope().visibility == "unknown"
        assert provider_a.system_prompt_block() == ""
        assert provider_a.get_tool_schemas() == []
        assert provider_a.prefetch("honeycrisp compass", session_id="unknown-1") == ""
        unknown_search = _tool(
            provider_a, "sanae_mem_search", {"query": "copper river"}, world["memory_manager_a"]
        )
        assert not unknown_search["ok"]


def test_registered_native_memory_guards_block_external_chat(world, monkeypatch):
    # Exercise the legacy prompt veto when the host has no active cognitive policy.
    import hermes_host_contracts as hc
    monkeypatch.setattr(hc, "get_cognitive_prompt_policy", lambda: None)
    hook_manager = world["hook_manager"]
    hooks = hook_manager._hooks
    assert any(hook.__name__ == "guard_public_memory_tool" for hook in hooks["pre_tool_call"])
    assert any(hook.__name__ == "guard_builtin_memory_prompt" for hook in hooks["pre_llm_call"])
    assert memory_guard.builtin_memory_flags() == (False, False)
    monkeypatch.setattr(memory_guard, "builtin_memory_flags", lambda: (True, True))
    monkeypatch.setattr(hermes_plugins, "_delivery_manager", lambda: hook_manager)

    with _route(actor="account-A", session="private-A"):
        assert hermes_plugins.get_pre_tool_call_block_message("memory", {}) is None
        assert hermes_plugins.get_pre_tool_call_block_message(
            "read_file", {"path": str(world["home"] / "memories" / "USER.md")},
        ) is None
        assert hermes_plugins.get_pre_tool_call_block_message("session_search", {}) == MIXED_SESSION_SEARCH_BLOCK_MESSAGE
        results = hermes_plugins.invoke_hook("pre_llm_call")
        assert not any(isinstance(result, dict) and result.get("action") == "block"
                       for result in results)

    for chat_type, actor in (("group", "account-A"), ("unrecognized", "account-A"), ("dm", "account-B")):
        with _route(actor=actor, session=f"restricted-{chat_type}", chat_type=chat_type):
            assert hermes_plugins.get_pre_tool_call_block_message(
                "memory", {"action": "add", "target": "memory", "content": "B cannot edit the maintainer's notes"},
            ) == PUBLIC_MEMORY_TOOL_BLOCK_MESSAGE
            assert hermes_plugins.get_pre_tool_call_block_message("session_search", {}) == MIXED_SESSION_SEARCH_BLOCK_MESSAGE
            assert hermes_plugins.get_pre_tool_call_block_message(
                "read_file", {"path": str(world["home"] / "memories" / "USER.md")}
            ) == PUBLIC_MEMORY_TOOL_BLOCK_MESSAGE
            assert {"action": "block", "message": PUBLIC_MEMORY_PROMPT_BLOCK_MESSAGE} in (
                hermes_plugins.invoke_hook("pre_llm_call")
            )


def test_external_file_tools_block_profile_paths_in_native_dispatch(world, tmp_path, monkeypatch):
    import hermes_host_contracts as hc
    monkeypatch.setattr(hc, "get_cognitive_prompt_policy", lambda: None)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from model_tools import handle_function_call
    from tools.terminal_tool import clear_session_cwd, record_session_cwd

    monkeypatch.setattr(hermes_plugins, "_delivery_manager", lambda: world["hook_manager"])
    home = world["home"]
    sessions = home / "sessions"
    sessions.mkdir(exist_ok=True)
    (sessions / "private.jsonl").write_text("A_PRIVATE_SESSION", encoding="utf-8")
    override = tmp_path / "other-profile"
    override.mkdir()
    (override / "state.db").write_text("A_PROFILE_STATE", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(override))
    public_dir = tmp_path / "workspace"
    public_dir.mkdir()
    (public_dir / "notes.txt").write_text("SHARED_FILE", encoding="utf-8")
    task_id = "private-B-file-dispatch"

    def call(tool, **args):
        return json.loads(handle_function_call(tool, args, task_id=task_id, session_id="private-B"))

    def blocked(tool, **args):
        assert call(tool, **args) == {"error": PUBLIC_MEMORY_TOOL_BLOCK_MESSAGE}

    with _route(actor="account-B", session="private-B"):
        record_session_cwd(task_id, str(home))
        try:
            blocked("read_file", path="sessions/private.jsonl")
            blocked("search_files", path=None, pattern="A_PRIVATE_SESSION")
            blocked("search_files", path=str(tmp_path), pattern="A_PRIVATE_SESSION")
            blocked("read_file", path=str(override / "state.db"))
            if os.name != "nt":
                (public_dir / "alias").symlink_to(sessions, target_is_directory=True)
                blocked("search_files", path=str(public_dir / "alias"), pattern="A_PRIVATE_SESSION")
            public_file = call("read_file", path=str(public_dir / "notes.txt"))
            assert "SHARED_FILE" in public_file.get("content", ""), public_file
            monkeypatch.delenv("HERMES_HOME_OVERRIDE")
            token = set_hermes_home_override(str(override))
            try:
                blocked("read_file", path=str(override / "state.db"))
            finally:
                reset_hermes_home_override(token)
        finally:
            clear_session_cwd(task_id)


def test_original_session_search_is_blocked_across_private_users(world, monkeypatch):
    from hermes_state import SessionDB
    from tools.session_search_tool import session_search

    marker = "SYNTHETIC_ACCOUNT_A_SESSION_ONLY"
    db = SessionDB(db_path=world["home"] / "state.db")
    try:
        db.create_session("synthetic-private-A", source="telegram")
        db.append_message("synthetic-private-A", role="user", content=marker)
        db.create_session("synthetic-private-B", source="telegram")
        db.append_message("synthetic-private-B", role="user", content="SYNTHETIC_ACCOUNT_B_SESSION")

        with _route(actor="account-B", session="private-B"):
            # The pinned host tool itself scopes only to this profile database; it can
            # read A's session by ID, which is why the plugin must veto it before dispatch.
            raw_host_result = json.loads(session_search(
                session_id="synthetic-private-A", current_session_id="synthetic-private-B", db=db,
            ))
            assert raw_host_result["success"] is True
            assert marker in json.dumps(raw_host_result, ensure_ascii=False)

            monkeypatch.setattr(hermes_plugins, "_delivery_manager", lambda: world["hook_manager"])
            block = hermes_plugins.get_pre_tool_call_block_message(
                "session_search", {"session_id": "synthetic-private-A"}, session_id="synthetic-private-B",
            )
            assert block == MIXED_SESSION_SEARCH_BLOCK_MESSAGE
    finally:
        db.close()


def test_gateway_admission_keeps_external_references_out_of_local_files(world, monkeypatch):
    # Exercise the legacy prompt veto when the host has no active cognitive policy.
    import hermes_host_contracts as hc
    monkeypatch.setattr(hc, "get_cognitive_prompt_policy", lambda: None)
    from gateway.config import Platform
    from gateway.run_inbound_consumer import run_post_admission_hook
    from gateway.session import SessionSource

    monkeypatch.setattr(hermes_plugins, "_delivery_manager", lambda: world["hook_manager"])
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))

    def admit(chat_type, text, actor="account-A", platform=Platform.TELEGRAM):
        source = SessionSource(platform=platform, chat_id="chat", chat_type=chat_type, user_id=actor)
        event = SimpleNamespace(message_id="incoming-1", text=text)
        return asyncio.run(run_post_admission_hook(runner, event, source, "session-1"))

    blocked = (True, LOCAL_REFERENCE_BLOCK_MESSAGE)
    assert admit("group", "@file:notes.txt") == blocked
    assert admit("dm", "@file:notes.txt", actor="account-B") == blocked
    assert admit("dm", "@folder:notes", actor="account-B") == blocked
    assert admit("dm", "@file:notes.txt", platform=Platform.DISCORD) == blocked
    assert admit("group", "hello @someone") == (False, None)
    assert admit("dm", "hello @someone", actor="account-B") == (False, None)
    assert admit("dm", "@file:notes.txt") == (False, None)

    monkeypatch.setattr(memory_guard, "builtin_memory_flags", lambda: (True, True))
    blocked_prompt = (True, PUBLIC_MEMORY_PROMPT_BLOCK_MESSAGE)
    assert admit("dm", "hello", actor="account-B") == blocked_prompt
    assert admit("group", "hello") == blocked_prompt
    assert admit("dm", "hello") == (False, None)
