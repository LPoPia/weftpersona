"""Regression tests for Task 06: host_bridge silent-swallow removal and fail-fast registration."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from typing import Any, Callable

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem.infra import host_bridge  # type: ignore[import-not-found]
from sanae_mem import plugin_entry  # type: ignore[import-not-found]


from sanae_mem import provider as provider_module
from sanae_mem.runtime import hooks as runtime_hooks


class _FakePluginContext:
    """Minimal PluginContext stub for testing plugin_entry.register() boundary behavior."""

    def __init__(
        self,
        *,
        fail_hook_name: str | None = None,
        fail_callback_name: str | None = None,
    ) -> None:
        self.fail_hook_name = fail_hook_name
        self.fail_callback_name = fail_callback_name
        self.provider: Any = None
        self.hooks: list[tuple[str, Any]] = []
        self.unload_callbacks: list[Callable[[], None]] = []

    def register_memory_provider(self, provider: Any) -> None:
        self.provider = provider

    def register_hook(self, name: str, callback: Any) -> None:
        cb_name = getattr(callback, "__name__", "")
        if self.fail_hook_name is not None and name == self.fail_hook_name:
            if self.fail_callback_name is None or cb_name == self.fail_callback_name:
                raise RuntimeError(f"hook registration failed: {name}/{cb_name}")
        self.hooks.append((name, callback))

    def register_command(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def on_unload(self, callback: Callable[[], None]) -> None:
        self.unload_callbacks.append(callback)


def _make_full_host_contracts(
    home: Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> tuple[SimpleNamespace, dict[str, Any]]:
    """Build an in-memory host contracts namespace matching public hermes_host_contracts APIs."""
    state: dict[str, Any] = {
        "cognitive_policy": None,
        "inbound_merge_policy": None,
        "group_context_policy": None,
        "group_reply_policy": None,
        "memory_mutation_gate": None,
        "delivery_handlers": [],
        "cron_preparers": [],
        "cron_gates": [],
        "cron_completion_handlers": [],
    }

    def _set(key: str) -> Callable[[Any], None]:
        def _setter(value: Any) -> None:
            state[key] = value

        return _setter

    def _clear(key: str) -> Callable[[Any], None]:
        def _clearer(value: Any = None) -> None:
            if value is None or state[key] is value:
                state[key] = None

        return _clearer

    def _append(key: str) -> Callable[[Any], None]:
        def _appender(value: Any) -> None:
            if value not in state[key]:
                state[key].append(value)

        return _appender

    def _remove(key: str) -> Callable[[Any], None]:
        def _remover(value: Any) -> None:
            if value in state[key]:
                state[key].remove(value)

        return _remover

    attrs: dict[str, Any] = {
        "get_hermes_home": lambda: home,
        "get_process_hermes_home": lambda: home,
        "get_cognitive_prompt_policy": lambda: state["cognitive_policy"],
        "register_cognitive_prompt_policy": _set("cognitive_policy"),
        "unregister_cognitive_prompt_policy": _clear("cognitive_policy"),
        "get_inbound_merge_policy": lambda: state["inbound_merge_policy"],
        "register_inbound_merge_policy": _set("inbound_merge_policy"),
        "unregister_inbound_merge_policy": _clear("inbound_merge_policy"),
        "get_group_context_policy": lambda: state["group_context_policy"],
        "register_group_context_policy": _set("group_context_policy"),
        "unregister_group_context_policy": _clear("group_context_policy"),
        "get_group_reply_policy": lambda: state["group_reply_policy"],
        "register_group_reply_policy": _set("group_reply_policy"),
        "unregister_group_reply_policy": _clear("group_reply_policy"),
        "GroupReplyDecision": lambda **fields: SimpleNamespace(**fields),
        "get_memory_mutation_gate": lambda: state["memory_mutation_gate"],
        "register_memory_mutation_gate": _set("memory_mutation_gate"),
        "unregister_memory_mutation_gate": _clear("memory_mutation_gate"),
        "register_delivery_receipt_handler": _append("delivery_handlers"),
        "unregister_delivery_receipt_handler": _remove("delivery_handlers"),
        "register_cron_prompt_preparer": _append("cron_preparers"),
        "unregister_cron_prompt_preparer": _remove("cron_preparers"),
        "register_cron_pre_delivery_gate": _append("cron_gates"),
        "unregister_cron_pre_delivery_gate": _remove("cron_gates"),
        "register_cron_completion_handler": _append("cron_completion_handlers"),
        "unregister_cron_completion_handler": _remove("cron_completion_handlers"),
        "list_plugin_cron_jobs": lambda **_kwargs: [],
    }
    if overrides:
        attrs.update(overrides)
    return SimpleNamespace(**attrs), state


def test_host_bridge_contains_zero_suppress_calls_or_silent_pass() -> None:
    source = (PLUGIN_ROOT / "sanae_mem" / "infra" / "host_bridge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "contextlib":
            assert all(alias.name != "suppress" for alias in node.names)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id != "suppress"
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr != "suppress"
        if isinstance(node, ast.ExceptHandler):
            assert not (len(node.body) == 1 and isinstance(node.body[0], ast.Pass))


def test_standalone_and_old_host_missing_seams_degrade_without_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 1. Pure standalone: _hc is None and _hti is None
    monkeypatch.setattr(host_bridge, "_hc", None)
    monkeypatch.setattr(host_bridge, "_hti", None)
    assert host_bridge.register_delivery_receipt_handler(lambda _r: None) is False
    assert host_bridge.register_cognitive_prompt_policy(object()) is False
    assert host_bridge.register_inbound_merge_policy(object()) is False
    assert host_bridge.register_memory_mutation_gate(object()) is False
    assert host_bridge.register_group_context_policy(object()) is False
    assert host_bridge.register_group_reply_policy(object()) is False
    assert host_bridge.register_cron_completion_handler(lambda _e: None) is False
    assert host_bridge.register_cron_hooks(
        prompt_preparer=lambda _j, _p, _e: None,
        pre_delivery_gate=lambda _j, _t, _e: None,
    ) is False
    host_bridge.unregister_delivery_receipt_handler(lambda _r: None)
    host_bridge.unregister_cognitive_prompt_policy(object())
    host_bridge.unregister_inbound_merge_policy(object())
    host_bridge.unregister_memory_mutation_gate(object())
    host_bridge.unregister_group_context_policy(object())
    host_bridge.unregister_group_reply_policy(object())
    host_bridge.unregister_cron_completion_handler(lambda _e: None)
    host_bridge.unregister_cron_hooks(
        prompt_preparer=lambda _j, _p, _e: None,
        pre_delivery_gate=lambda _j, _t, _e: None,
    )
    assert host_bridge.current_turn_envelope() is None
    assert host_bridge.is_test_isolated_turn("normal-sess") is False
    assert host_bridge.is_test_isolated_turn("T:isolated") is True
    assert host_bridge.current_cron_binding() is None
    assert host_bridge.verified_cron_envelope() is None
    assert host_bridge.query_verified_cron_sends(bound_owner="owner") == []
    assert host_bridge.resolve_task_cwd() == [Path.cwd()]

    # 2. Old host: _hc exists but omits newer policy/hook seams
    monkeypatch.setattr(host_bridge, "_hc", SimpleNamespace(get_hermes_home=lambda: tmp_path))
    monkeypatch.setattr(runtime_hooks.auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register_daily", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)
    ctx = _FakePluginContext()
    with caplog.at_level(logging.WARNING):
        plugin_entry.register(ctx)
    try:
        assert ctx.provider is not None
        assert ctx.provider.name == "sanae_mem"
        assert "delivery receipt hook skipped" in caplog.text
        assert "cron hooks skipped" in caplog.text
        assert "registration seam; skipped" in caplog.text
    finally:
        for unload in reversed(ctx.unload_callbacks):
            unload()


@pytest.mark.parametrize(
    ("bridge_call", "contract_name"),
    [
        (lambda: host_bridge.register_delivery_receipt_handler(lambda _r: None), "register_delivery_receipt_handler"),
        (lambda: host_bridge.register_cognitive_prompt_policy(object()), "register_cognitive_prompt_policy"),
        (lambda: host_bridge.register_inbound_merge_policy(object()), "register_inbound_merge_policy"),
        (lambda: host_bridge.register_memory_mutation_gate(object()), "register_memory_mutation_gate"),
        (lambda: host_bridge.register_group_context_policy(object()), "register_group_context_policy"),
        (lambda: host_bridge.register_group_reply_policy(object()), "register_group_reply_policy"),
        (lambda: host_bridge.register_cron_completion_handler(lambda _e: None), "register_cron_completion_handler"),
        (
            lambda: host_bridge.register_cron_hooks(prompt_preparer=lambda _j, _p, _e: None),
            "register_cron_prompt_preparer",
        ),
        (
            lambda: host_bridge.register_cron_hooks(pre_delivery_gate=lambda _j, _t, _e: None),
            "register_cron_pre_delivery_gate",
        ),
    ],
)
def test_host_bridge_registration_fails_fast_and_logs_error_when_seam_raises(
    bridge_call: Callable[[], bool],
    contract_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(f"{contract_name} exploded")

    monkeypatch.setattr(host_bridge, "_hc", SimpleNamespace(**{contract_name: _boom}))
    with caplog.at_level(logging.ERROR, logger=host_bridge.logger.name):
        with pytest.raises(RuntimeError, match=f"{contract_name} exploded"):
            bridge_call()
    assert any(
        rec.levelno >= logging.ERROR and contract_name in rec.getMessage()
        for rec in caplog.records
    )


def test_register_cron_hooks_rolls_back_preparer_when_gate_registration_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparers: list[Any] = []

    def _add_preparer(fn: Any) -> None:
        preparers.append(fn)

    def _remove_preparer(fn: Any) -> None:
        preparers.remove(fn)

    def _broken_gate(_fn: Any) -> None:
        raise RuntimeError("gate registry broken")

    monkeypatch.setattr(
        host_bridge,
        "_hc",
        SimpleNamespace(
            register_cron_prompt_preparer=_add_preparer,
            unregister_cron_prompt_preparer=_remove_preparer,
            register_cron_pre_delivery_gate=_broken_gate,
        ),
    )
    with pytest.raises(RuntimeError, match="gate registry broken"):
        host_bridge.register_cron_hooks(
            prompt_preparer=lambda _j, _p, _e: None,
            pre_delivery_gate=lambda _j, _t, _e: None,
        )
    assert preparers == []


@pytest.mark.parametrize(
    ("hook_name", "callback_name"),
    [
        ("post_gateway_admission", "guard_public_local_references"),
        ("pre_llm_call", "guard_builtin_memory_prompt"),
        ("pre_tool_call", "guard_public_memory_tool"),
        ("pre_tool_call", "guard_memory_write"),
        ("pre_tool_call", "guard_controlled_memory_write"),
        ("transform_tool_result", "remind_memory_result"),
        ("pre_llm_call", "handle_pre_llm_call"),
    ],
)
def test_plugin_entry_register_fails_fast_on_core_hook_failure(
    hook_name: str,
    callback_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hc_stub, _state = _make_full_host_contracts(tmp_path)
    monkeypatch.setattr(host_bridge, "_hc", hc_stub)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register_daily", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)

    ctx = _FakePluginContext(fail_hook_name=hook_name, fail_callback_name=callback_name)
    with caplog.at_level(logging.ERROR, logger=runtime_hooks.logger.name):
        with pytest.raises(RuntimeError, match=f"hook registration failed: {hook_name}/{callback_name}"):
            plugin_entry.register(ctx)
    assert any(
        rec.levelno >= logging.ERROR and callback_name in rec.getMessage()
        for rec in caplog.records
    )


def test_plugin_entry_register_keeps_auxiliary_and_daily_flash_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hc_stub, _state = _make_full_host_contracts(tmp_path)
    monkeypatch.setattr(host_bridge, "_hc", hc_stub)
    monkeypatch.setattr(
        runtime_hooks.auxiliary,
        "register",
        lambda _ctx: (_ for _ in ()).throw(RuntimeError("aux down")),
    )
    monkeypatch.setattr(
        runtime_hooks.auxiliary,
        "register_daily",
        lambda _ctx: (_ for _ in ()).throw(RuntimeError("daily down")),
    )
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)

    ctx = _FakePluginContext()
    with caplog.at_level(logging.WARNING):
        plugin_entry.register(ctx)
    try:
        assert ctx.provider is not None
        assert "auxiliary registration skipped" in caplog.text
        assert "daily Flash registration skipped" in caplog.text
    finally:
        for unload in reversed(ctx.unload_callbacks):
            unload()


def _probe_policy_active(state: dict[str, Any], policy_key: str, home: Path) -> bool:
    """Check via public host contract interfaces whether *policy_key* serves *home*."""
    policy = state[policy_key]
    if policy is None:
        return False
    group_source = SimpleNamespace(platform="qqbot", chat_type="group", chat_id="g1", scope_id="app1")
    with host_bridge.hermes_home_scope(home):
        if policy_key == "cognitive_policy":
            env = SimpleNamespace(
                session_context_engaged=True,
                platform="qqbot",
                chat_type="group",
                chat_id="g1",
                sender_id="u1",
                is_messaging_surface=True,
                restricted_audience="",
                is_cron=False,
                is_delegated_child=False,
                session_id="s1",
            )
            return policy.resolve_turn_policy(env) is not None
        if policy_key == "inbound_merge_policy":
            return policy.debounce_window_seconds("qqbot", "dm", busy=False) == 5.0
        if policy_key == "group_context_policy":
            return policy(group_source) is not None
        if policy_key == "group_reply_policy":
            return policy(group_source) is not None
        if policy_key == "memory_mutation_gate":
            verdict = policy.evaluate_memory_mutation(action="add", target="user", content="probe")
            return verdict is not None and bool(verdict.blocked)
    raise ValueError(f"Unknown policy_key: {policy_key}")


@pytest.mark.parametrize(
    "failing_contract",
    [
        "register_delivery_receipt_handler",
        "register_cognitive_prompt_policy",
        "register_inbound_merge_policy",
        "register_group_context_policy",
        "register_group_reply_policy",
        "register_memory_mutation_gate",
        "register_cron_prompt_preparer",
        "register_cron_pre_delivery_gate",
    ],
)
def test_plugin_entry_register_fails_fast_when_host_policy_or_hook_contract_raises(
    failing_contract: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    home_a.mkdir()
    home_b.mkdir()

    hc_stub, state = _make_full_host_contracts(home_a)
    monkeypatch.setattr(host_bridge, "_hc", hc_stub)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register_daily", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)

    # First register home_a cleanly so shared home-routed policy registries already exist.
    ctx_a = _FakePluginContext()
    with host_bridge.hermes_home_scope(home_a):
        plugin_entry.register(ctx_a)

    # Now inject a failure on the target host contract and attempt to register home_b.
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(f"contract failure: {failing_contract}")

    monkeypatch.setattr(hc_stub, failing_contract, _boom)
    ctx_b = _FakePluginContext()
    try:
        with caplog.at_level(logging.ERROR):
            with host_bridge.hermes_home_scope(home_b):
                with pytest.raises(RuntimeError, match=f"contract failure: {failing_contract}"):
                    plugin_entry.register(ctx_b)
        assert any(
            rec.levelno >= logging.ERROR and failing_contract in rec.getMessage()
            for rec in caplog.records
        )
        # Verify via public policy interfaces that home_a stays active and failed home_b was rolled back
        # across every home-routed policy, not just the failing one.
        for policy_key in (
            "cognitive_policy",
            "inbound_merge_policy",
            "group_context_policy",
            "group_reply_policy",
            "memory_mutation_gate",
        ):
            assert _probe_policy_active(state, policy_key, home_a) is True
            assert _probe_policy_active(state, policy_key, home_b) is False
    finally:
        for unload in reversed(ctx_b.unload_callbacks):
            unload()
        for unload in reversed(ctx_a.unload_callbacks):
            unload()


def test_runtime_queries_and_unregisters_log_warnings_instead_of_silent_suppress(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom(label: str) -> Callable[..., Any]:
        def _raiser(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(f"{label} crashed")

        return _raiser

    # 1. is_test_isolated_turn: _hti fails (logged), falls through to _hc envelope (True)
    monkeypatch.setattr(host_bridge, "_hti", SimpleNamespace(is_test_session=_boom("hti")))
    monkeypatch.setattr(
        host_bridge,
        "_hc",
        SimpleNamespace(current_turn_envelope=lambda: SimpleNamespace(is_test_isolated=True)),
    )
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.is_test_isolated_turn("sess-1") is True
    assert "is_test_session failed" in caplog.text

    # 2. All runtime queries raising on _hc must log WARNING and return safe defaults
    caplog.clear()
    monkeypatch.setattr(host_bridge, "_hti", None)
    monkeypatch.setattr(
        host_bridge,
        "_hc",
        SimpleNamespace(
            current_turn_envelope=_boom("turn_env"),
            current_cron_envelope=_boom("cron_env"),
            resolve_task_cwd=_boom("task_cwd"),
            query_verified_cron_sends=_boom("cron_sends"),
        ),
    )
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.is_test_isolated_turn("sess-2") is False
        assert host_bridge.current_turn_envelope() is None
        assert host_bridge.verified_cron_envelope() is None
        assert host_bridge.current_cron_binding() is None
        assert host_bridge.resolve_task_cwd("t-1") == [Path.cwd()]
        assert host_bridge.query_verified_cron_sends(bound_owner="owner-1") == []
    for marker in (
        "current_turn_envelope failed",
        "current_cron_envelope failed",
        "resolve_task_cwd failed",
        "query_verified_cron_sends failed",
    ):
        assert marker in caplog.text

    # 3. Malformed cron envelope and malformed cron send row log WARNING cleanly
    caplog.clear()
    monkeypatch.setattr(
        host_bridge,
        "_hc",
        SimpleNamespace(
            current_cron_envelope=lambda **_kwargs: SimpleNamespace(job_id="j1"),
            query_verified_cron_sends=lambda **_kwargs: [
                {"bound_owner": "owner-1", "delivered_at": "not-a-float", "delivered_text": "bad"},
                {"bound_owner": "owner-1", "delivered_at": 100.0, "delivered_text": "good"},
            ],
        ),
    )
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.current_cron_binding() is None
        assert host_bridge.query_verified_cron_sends(bound_owner="owner-1") == [(100.0, "good")]
    assert "malformed cron envelope" in caplog.text
    assert "invalid delivered_at" in caplog.text

    # 4. Unregister paths log WARNING and unregister_cron_hooks isolates both calls
    caplog.clear()
    gate_unregistered: list[Any] = []
    monkeypatch.setattr(
        host_bridge,
        "_hc",
        SimpleNamespace(
            unregister_delivery_receipt_handler=_boom("unreg_receipt"),
            unregister_cognitive_prompt_policy=_boom("unreg_cog"),
            unregister_inbound_merge_policy=_boom("unreg_merge"),
            unregister_memory_mutation_gate=_boom("unreg_gate"),
            unregister_group_context_policy=_boom("unreg_gctx"),
            unregister_group_reply_policy=_boom("unreg_greply"),
            unregister_cron_completion_handler=_boom("unreg_cron_done"),
            unregister_cron_prompt_preparer=_boom("unreg_cron_prep"),
            unregister_cron_pre_delivery_gate=lambda fn: gate_unregistered.append(fn),
        ),
    )
    gate_fn = lambda _j, _t, _e: None
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        host_bridge.unregister_delivery_receipt_handler(lambda _r: None)
        host_bridge.unregister_cognitive_prompt_policy(object())
        host_bridge.unregister_inbound_merge_policy(object())
        host_bridge.unregister_memory_mutation_gate(object())
        host_bridge.unregister_group_context_policy(object())
        host_bridge.unregister_group_reply_policy(object())
        host_bridge.unregister_cron_completion_handler(lambda _e: None)
        host_bridge.unregister_cron_hooks(
            prompt_preparer=lambda _j, _p, _e: None,
            pre_delivery_gate=gate_fn,
        )
    assert gate_unregistered == [gate_fn]
    for marker in (
        "unregister_delivery_receipt_handler failed",
        "unregister_cognitive_prompt_policy failed",
        "unregister_inbound_merge_policy failed",
        "unregister_memory_mutation_gate failed",
        "unregister_group_context_policy failed",
        "unregister_group_reply_policy failed",
        "unregister_cron_completion_handler failed",
        "unregister_cron_prompt_preparer failed",
    ):
        assert marker in caplog.text


def test_load_host_config_distinguishes_missing_vs_corrupt_and_unreadable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 1. Missing config.yaml -> FileNotFoundError -> quiet {} (no warning)
    missing_home = tmp_path / "missing"
    missing_home.mkdir()
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(missing_home) == {}
    assert caplog.records == []

    # 2. Empty config.yaml -> quiet {} (no warning)
    empty_home = tmp_path / "empty"
    empty_home.mkdir()
    (empty_home / "config.yaml").write_text("   \n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(empty_home) == {}
    assert caplog.records == []

    # 3. Corrupt YAML -> logs WARNING and returns {}
    corrupt_home = tmp_path / "corrupt"
    corrupt_home.mkdir()
    (corrupt_home / "config.yaml").write_text("plugins:\n  sanae_mem: [unterminated\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(corrupt_home) == {}
    assert any(
        rec.levelno == logging.WARNING and "failed to parse host config" in rec.getMessage()
        for rec in caplog.records
    )

    # 4. Non-mapping YAML root -> logs WARNING and returns {}
    caplog.clear()
    list_home = tmp_path / "list_root"
    list_home.mkdir()
    (list_home / "config.yaml").write_text("- item1\n- item2\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(list_home) == {}
    assert any(
        rec.levelno == logging.WARNING and "root must be a mapping" in rec.getMessage()
        for rec in caplog.records
    )

    # 5. Unreadable config.yaml (directory instead of file triggers real OSError/PermissionError)
    caplog.clear()
    unreadable_home = tmp_path / "unreadable"
    (unreadable_home / "config.yaml").mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(unreadable_home) == {}
    assert any(
        rec.levelno == logging.WARNING and "failed to read host config" in rec.getMessage()
        for rec in caplog.records
    )


def test_same_home_reregistration_failure_preserves_existing_policy_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home = tmp_path / "same_home"
    home.mkdir()
    hc_stub, state = _make_full_host_contracts(home)
    monkeypatch.setattr(host_bridge, "_hc", hc_stub)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register_daily", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)

    ctx_first = _FakePluginContext()
    with host_bridge.hermes_home_scope(home):
        plugin_entry.register(ctx_first)

    gate_before = state["memory_mutation_gate"]
    assert gate_before is not None
    with host_bridge.hermes_home_scope(home):
        verdict_before = gate_before.evaluate_memory_mutation(
            action="add", target="user", content="prior profile rule",
        )
    assert verdict_before is not None and verdict_before.blocked is True

    # 1. Re-registering the SAME home when register_memory_mutation_gate raises must restore prior entry
    def _boom(_gate: Any) -> None:
        raise RuntimeError("transient host gate error")

    monkeypatch.setattr(hc_stub, "register_memory_mutation_gate", _boom)
    ctx_second = _FakePluginContext()
    try:
        with caplog.at_level(logging.ERROR):
            with host_bridge.hermes_home_scope(home):
                with pytest.raises(RuntimeError, match="transient host gate error"):
                    plugin_entry.register(ctx_second)
        assert state["memory_mutation_gate"] is gate_before
        with host_bridge.hermes_home_scope(home):
            verdict_after_err = gate_before.evaluate_memory_mutation(
                action="add", target="user", content="profile rule after error",
            )
        assert verdict_after_err is not None and verdict_after_err.blocked is True

        # 2. Re-registering the SAME home when seam returns False (absent seam) must also preserve prior entry
        caplog.clear()
        monkeypatch.setattr(hc_stub, "register_memory_mutation_gate", None)
        ctx_third = _FakePluginContext()
        try:
            with caplog.at_level(logging.WARNING):
                with host_bridge.hermes_home_scope(home):
                    plugin_entry.register(ctx_third)
            assert "registration seam; skipped" in caplog.text
            with host_bridge.hermes_home_scope(home):
                verdict_after_skip = gate_before.evaluate_memory_mutation(
                    action="add", target="user", content="profile rule after skip",
                )
            assert verdict_after_skip is not None and verdict_after_skip.blocked is True
        finally:
            for unload in reversed(ctx_third.unload_callbacks):
                unload()
    finally:
        for unload in reversed(ctx_second.unload_callbacks):
            unload()
        for unload in reversed(ctx_first.unload_callbacks):
            unload()


@pytest.mark.parametrize(
    "failing_contract",
    [
        "register_memory_mutation_gate",
        "register_cron_prompt_preparer",
        "register_cron_pre_delivery_gate",
    ],
)
def test_same_home_reregistration_failure_unloads_cleanly_preserving_prior_policies(
    failing_contract: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "same_home_r2"
    home.mkdir()
    hc_stub, state = _make_full_host_contracts(home)
    monkeypatch.setattr(host_bridge, "_hc", hc_stub)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register_daily", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)

    policy_keys = (
        "cognitive_policy",
        "inbound_merge_policy",
        "group_context_policy",
        "group_reply_policy",
        "memory_mutation_gate",
    )

    ctx_a = _FakePluginContext()
    with host_bridge.hermes_home_scope(home):
        plugin_entry.register(ctx_a)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(f"same-home reload failed at {failing_contract}")

    monkeypatch.setattr(hc_stub, failing_contract, _boom)
    ctx_b = _FakePluginContext()
    with host_bridge.hermes_home_scope(home):
        with pytest.raises(RuntimeError, match=f"same-home reload failed at {failing_contract}"):
            plugin_entry.register(ctx_b)

    # Before running ctx_b's on_unload callbacks, all policies from ctx_a must still be active.
    with host_bridge.hermes_home_scope(home):
        assert state["inbound_merge_policy"] is not None
        assert state["inbound_merge_policy"].debounce_window_seconds("qqbot", "dm", busy=False) == 5.0
        verdict_before = state["memory_mutation_gate"].evaluate_memory_mutation(
            action="add", target="user", content="profile fact before unload",
        )
        assert verdict_before is not None and verdict_before.blocked is True
    for key in policy_keys:
        assert _probe_policy_active(state, key, home) is True

    # Run ALL of ctx_b's on_unload callbacks (just as host teardown does after failed register).
    for unload in reversed(ctx_b.unload_callbacks):
        unload()

    # After ctx_b teardown, ctx_a's policies and mutation gate must remain completely intact.
    with host_bridge.hermes_home_scope(home):
        assert state["inbound_merge_policy"] is not None
        assert state["inbound_merge_policy"].debounce_window_seconds("qqbot", "dm", busy=False) == 5.0
        verdict_after = state["memory_mutation_gate"].evaluate_memory_mutation(
            action="add", target="user", content="profile fact after ctx_b unload",
        )
        assert verdict_after is not None and verdict_after.blocked is True
    for key in policy_keys:
        assert _probe_policy_active(state, key, home) is True

    # Finally unload ctx_a: every policy must cleanly unregister from the host.
    for unload in reversed(ctx_a.unload_callbacks):
        unload()
    for key in policy_keys:
        assert state[key] is None


def test_same_home_dual_registration_unloads_in_fifo_order_without_dropping_live_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "same_home_fifo"
    home.mkdir()
    hc_stub, state = _make_full_host_contracts(home)
    monkeypatch.setattr(host_bridge, "_hc", hc_stub)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks.auxiliary, "register_daily", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)

    policy_keys = (
        "cognitive_policy",
        "inbound_merge_policy",
        "group_context_policy",
        "group_reply_policy",
        "memory_mutation_gate",
    )
    ctx_a = _FakePluginContext()
    ctx_b = _FakePluginContext()
    with host_bridge.hermes_home_scope(home):
        plugin_entry.register(ctx_a)
        plugin_entry.register(ctx_b)

    # Unload older ctx_a first -> newer ctx_b must keep every policy active.
    for unload in reversed(ctx_a.unload_callbacks):
        unload()
    with host_bridge.hermes_home_scope(home):
        assert state["inbound_merge_policy"].debounce_window_seconds("qqbot", "dm", busy=False) == 5.0
        verdict = state["memory_mutation_gate"].evaluate_memory_mutation(
            action="add", target="user", content="profile fact after ctx_a unload",
        )
        assert verdict is not None and verdict.blocked is True
    for key in policy_keys:
        assert _probe_policy_active(state, key, home) is True

    # Now unload ctx_b -> every policy must cleanly unregister.
    for unload in reversed(ctx_b.unload_callbacks):
        unload()
    for key in policy_keys:
        assert state[key] is None


@pytest.mark.parametrize(
    "raw_yaml",
    [
        "plugins: {sanae_mem: !!bool maybe}\n",
        "plugins: {sanae_mem: !!timestamp bad}\n",
        "plugins: {sanae_mem: !!int not_an_int}\n",
        "plugins: {sanae_mem: !!int ''}\n",
        "plugins: {sanae_mem: !!float ''}\n",
    ],
)
@pytest.mark.parametrize("backend", ["pyyaml", "ruamel"])
def test_load_host_config_handles_malformed_yaml_tags_for_pyyaml_and_ruamel(
    raw_yaml: str,
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if backend == "ruamel":
        monkeypatch.setitem(sys.modules, "yaml", None)
    cfg_home = tmp_path / f"bad_tag_{backend}"
    cfg_home.mkdir()
    (cfg_home / "config.yaml").write_text(raw_yaml, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(cfg_home) == {}
    assert any(
        rec.levelno == logging.WARNING
        and "failed to parse host config" in rec.getMessage()
        and "HostConfigParseError" in rec.getMessage()
        for rec in caplog.records
    )


def test_on_session_switch_rebinds_cleanly_and_raises_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    data_dir = tmp_path / "sanae-mem"
    config = {
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "owner-1"},
    }
    provider = plugin_entry.SanaeMemProvider(config)
    with host_bridge.turn_envelope_scope(
        platform="telegram",
        chat_type="dm",
        chat_id="owner-1",
        sender_id="owner-1",
        session_id="sess-old",
        session_key="agent:main:telegram:dm:owner-1",
        is_messaging_surface=True,
    ):
        provider.initialize("sess-old", hermes_home=str(home), platform="telegram", agent_context="primary")
    try:
        assert plugin_entry.resolve_active_provider(home, "sess-old") is provider

        # Normal switch rebinds active provider registry from old -> new
        provider.on_session_switch("sess-new")
        assert plugin_entry.resolve_active_provider(home, "sess-old") is None
        assert plugin_entry.resolve_active_provider(home, "sess-new") is provider

        # Empty session_id is a no-op
        provider.on_session_switch("")
        assert plugin_entry.resolve_active_provider(home, "sess-new") is provider

        # Failure during active-provider rebind must log ERROR and propagate,
        # while STILL having switched the provider's session_id so subsequent writes
        # cannot silently land in the old session if the host catches the exception.
        orig_resolve = Path.resolve

        def _failing_resolve(path_self: Path, strict: bool = False) -> Path:
            if str(path_self) == str(home):
                raise OSError("stale mount during rebind")
            return orig_resolve(path_self, strict=strict)

        monkeypatch.setattr(Path, "resolve", _failing_resolve)
        try:
            with caplog.at_level(logging.ERROR, logger=provider_module.logger.name):
                with pytest.raises(OSError, match="stale mount during rebind"):
                    provider.on_session_switch("sess-after-err")
        finally:
            monkeypatch.setattr(Path, "resolve", orig_resolve)

        assert any(
            rec.levelno >= logging.ERROR and "session switch rebind failed" in rec.getMessage()
            for rec in caplog.records
        )
        assert plugin_entry.resolve_active_provider(home, "sess-new") is None

        # Simulate host continuing after catching on_session_switch error: new write must
        # carry the new session_id ("sess-after-err"), never the old "sess-new".
        with host_bridge.turn_envelope_scope(
            platform="telegram",
            chat_type="dm",
            chat_id="owner-1",
            sender_id="owner-1",
            session_id="sess-after-err",
            session_key="agent:main:telegram:dm:owner-1",
            is_messaging_surface=True,
        ):
            add_res = json.loads(
                provider.handle_tool_call(
                    "sanae_mem_add",
                    {"content": "下午在街角咖啡馆一起喝了手冲耶加雪菲。"},
                )
            )
        assert add_res["ok"] is True and add_res["result"]["stored"] is True, add_res
        with sqlite3.connect(str(data_dir / "sanae.db")) as conn:
            session_ids = [row[0] for row in conn.execute("SELECT session_id FROM events").fetchall()]
        assert session_ids == ["sess-after-err"]
    finally:
        provider.shutdown()
