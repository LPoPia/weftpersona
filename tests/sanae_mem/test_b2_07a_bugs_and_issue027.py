"""Regression and concurrency tests for B2-07A bug fixes and Issue 027."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import contextlib
from contextvars import copy_context
from datetime import datetime, timedelta
import importlib.util
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import threading
from time import monotonic
from typing import Any, Callable, Iterator, cast

import pytest
import yaml

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
SANAE_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from _harness import (  # noqa: E402
    LoopbackLlmServer as _LoopbackLLMServer,
    install_plugin_copy,
    resolve_hermes_host_root,
)

HERMES_ROOT = resolve_hermes_host_root()
if HERMES_ROOT is not None and str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from sanae_mem import plugin_entry  # type: ignore[import-not-found]
from sanae_mem.infra import host_bridge  # type: ignore[import-not-found]


def _base_config(data_dir: Path, **extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "data_dir": str(data_dir),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "owner-1"},
    }
    config.update(extra)
    return config


def _private_scope(session_id: str) -> Any:
    return host_bridge.turn_envelope_scope(
        platform="telegram",
        chat_type="dm",
        chat_id="owner-1",
        sender_id="owner-1",
        session_id=session_id,
        session_key="agent:main:telegram:dm:owner-1",
        is_messaging_surface=True,
    )


def _public_scope(session_id: str, chat_id: str = "public-room") -> Any:
    return host_bridge.turn_envelope_scope(
        platform="telegram",
        chat_type="group",
        chat_id=chat_id,
        sender_id="visitor-1",
        session_id=session_id,
        session_key=f"agent:main:telegram:group:{chat_id}",
        is_messaging_surface=True,
    )


def _install_plugin_for_home(
    home: Path,
    *,
    base_url: str,
    plugin_extra: dict[str, Any] | None = None,
    transient_retries: int | None = None,
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "memories").mkdir(exist_ok=True)
    plugin_dest = home / "plugins" / "sanae_mem"
    if not plugin_dest.exists():
        install_plugin_copy(home, SANAE_PLUGIN_DIR.parents[1], plugin_name="sanae_mem")
    data_dir = (home / "sanae-mem").resolve()
    (home / ".env").write_text("OPENAI_API_KEY=loopback-fixture\n", encoding="utf-8")
    plugin_cfg = _base_config(data_dir, **(plugin_extra or {}))
    auxiliary_cfg: dict[str, Any] = {
        "sanae_extract": {
            "provider": "custom",
            "model": "vm-extract",
            "base_url": base_url,
            "api_mode": "chat_completions",
        },
        "sanae_daily_flash": {
            "provider": "custom",
            "model": "vm-flash",
            "base_url": base_url,
            "api_mode": "chat_completions",
        },
    }
    if transient_retries is not None:
        auxiliary_cfg["transient_retries"] = int(transient_retries)
    root_cfg = {
        "auxiliary": auxiliary_cfg,
        "memory": {
            "provider": "sanae_mem",
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "plugins": {
            "enabled": ["sanae_mem"],
            "sanae_mem": plugin_cfg,
        },
    }
    (home / "config.yaml").write_text(
        yaml.safe_dump(root_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return data_dir


@contextlib.contextmanager
def _host_home_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import hermes_constants as hc  # pyright: ignore[reportMissingImports]
    from hermes_host_contracts import clear_all_contract_registrations  # pyright: ignore[reportMissingImports]

    clear_all_contract_registrations()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("OPENAI_API_KEY", "loopback-fixture")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    token = hc.set_hermes_home_override(home)
    try:
        yield
    finally:
        hc.reset_hermes_home_override(token)
        clear_all_contract_registrations()


@pytest.mark.skipif(HERMES_ROOT is None, reason="hermes-agent host source tree not found")
def test_compact_passes_now_to_distill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug 1: compact(now=...) forwards now to distill(now=now) through real host PluginLlm HTTP."""
    from hermes_cli.plugins import get_plugin_manager  # pyright: ignore[reportMissingImports]
    from plugins.memory import load_memory_provider  # pyright: ignore[reportMissingImports]

    home = tmp_path / "home"
    target_now = datetime.now().astimezone() + timedelta(days=10)

    def _respond(_payload: dict[str, Any], _idx: int) -> dict[str, Any]:
        return {
            "summary": "周末去郊外公园野餐并带了手冲咖啡",
            "duty": "memory",
            "source_kind": "event_promotion",
            "long_term": True,
            "covered_by": "",
            "significance": "记录周末休闲习惯",
        }

    with _LoopbackLLMServer(_respond) as llm_server:
        data_dir = _install_plugin_for_home(
            home,
            base_url=llm_server.base_url,
            plugin_extra={"distill_daily_cap": 2, "distill_threshold": 0.5},
        )
        with _host_home_env(home, monkeypatch):
            manager = get_plugin_manager()
            loaded = load_memory_provider("sanae_mem", register_skills=False)
            assert loaded is not None
            provider = cast(plugin_entry.SanaeMemProvider, loaded)
            try:
                with _private_scope("sess-compact"):
                    provider.initialize(
                        "sess-compact",
                        hermes_home=str(home),
                        platform="telegram",
                        agent_context="primary",
                    )
                    add_res = json.loads(
                        provider.handle_tool_call(
                            "sanae_mem_add",
                            {
                                "content": "周末去郊外公园野餐，带了手冲壶和耶加雪菲咖啡豆。",
                                "importance": 0.9,
                            },
                        )
                    )
                assert add_res["ok"] is True and add_res["result"]["stored"] is True
                event_id = int(add_res["result"]["id"])

                created_iso = (target_now - timedelta(hours=2)).isoformat()
                with sqlite3.connect(str(data_dir / "sanae.db")) as conn:
                    conn.execute(
                        "UPDATE events SET created_at=?, ts=? WHERE id=?",
                        (created_iso, created_iso, event_id),
                    )
                    conn.commit()

                requests_before = len(llm_server.requests)
                with _private_scope("sess-compact"):
                    result = provider.lifecycle.compact(now=target_now)
                assert len(result["distill_candidates"]) == 1
                assert len(llm_server.requests) > requests_before

                with sqlite3.connect(str(data_dir / "sanae.db")) as conn:
                    row = conn.execute(
                        "SELECT distilled_at, summary FROM events WHERE id=?",
                        (event_id,),
                    ).fetchone()
                assert row is not None
                assert row[0] == target_now.isoformat()
                assert "郊外公园野餐" in str(row[1])

                status = json.loads((data_dir / "status.json").read_text(encoding="utf-8"))
                assert status["last_distill_result"]["written"] == 1
            finally:
                provider.shutdown()
                manager.unload("sanae_mem")


def test_public_search_handles_naive_created_at_without_type_error(tmp_path: Path) -> None:
    """Bug 2: _public_search normalizes naive created_at timestamps before subtracting from now."""
    home = tmp_path / "home"
    home.mkdir()
    data_dir = tmp_path / "sanae-mem"
    provider = plugin_entry.SanaeMemProvider(_base_config(data_dir))

    with _private_scope("sess-seed"):
        provider.initialize(
            "sess-seed",
            hermes_home=str(home),
            platform="telegram",
            agent_context="primary",
        )
    try:
        with _private_scope("sess-seed"):
            add_res = json.loads(
                provider.handle_tool_call(
                    "sanae_mem_add",
                    {"content": "平时常去街角那家咖啡馆喝手冲耶加雪菲。"},
                )
            )
        assert add_res["ok"] is True and add_res["result"]["stored"] is True
        event_id = int(add_res["result"]["id"])

        with sqlite3.connect(str(data_dir / "sanae.db")) as conn:
            conn.execute(
                "UPDATE events SET public_excerpt=?, created_at=? WHERE id=?",
                ("喜欢街角咖啡馆的手冲耶加雪菲", "2026-10-01T12:00:00", event_id),
            )
            conn.commit()

        with _public_scope("sess-pub"):
            search_res = json.loads(
                provider.handle_tool_call(
                    "sanae_mem_search",
                    {"query": "手冲耶加雪菲", "top_k": 5},
                )
            )
        assert search_res["ok"] is True, search_res
        items = search_res["result"]
        assert len(items) == 1
        assert items[0]["id"] == event_id
        assert "手冲耶加雪菲" in items[0]["excerpt"]
    finally:
        provider.shutdown()


def test_read_gate_reads_behavior_from_provider_home_not_ambient_home(tmp_path: Path) -> None:
    """Bug 3: _behavior_alphabet and _read_behavior_zone read from provider_home(self)."""
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    home_a.mkdir()
    home_b.mkdir()

    (home_a / "BEHAVIOR.md").write_text(
        "**编号字母表**：①②③\n"
        "<!-- RENDER:system_prompt_block:BEGIN -->\n"
        "①② HOME_A_BEHAVIOR_MARKER\n"
        "<!-- RENDER:system_prompt_block:END -->\n",
        encoding="utf-8",
    )
    (home_b / "BEHAVIOR.md").write_text(
        "**编号字母表**：④⑤⑥\n"
        "<!-- RENDER:system_prompt_block:BEGIN -->\n"
        "④⑤ HOME_B_BEHAVIOR_MARKER\n"
        "<!-- RENDER:system_prompt_block:END -->\n",
        encoding="utf-8",
    )

    data_a = home_a / "sanae-mem"
    provider_a = plugin_entry.SanaeMemProvider(_base_config(data_a))
    with _private_scope("sess-home-a"):
        provider_a.initialize(
            "sess-home-a",
            hermes_home=str(home_a),
            platform="telegram",
            agent_context="primary",
        )
    try:
        with host_bridge.hermes_home_scope(home_b), _private_scope("sess-home-a"):
            prompt = provider_a.system_prompt_block()
        assert "HOME_A_BEHAVIOR_MARKER" in prompt
        assert "HOME_B_BEHAVIOR_MARKER" not in prompt
        status = json.loads((data_a / "status.json").read_text(encoding="utf-8"))
        assert status["behavior_block_ok"] is True
        assert status["behavior_block_chars"] == len("①② HOME_A_BEHAVIOR_MARKER")

        with host_bridge.hermes_home_scope(home_b), _public_scope("sess-home-a"):
            public_prompt = provider_a.system_prompt_block()
        assert "HOME_A_BEHAVIOR_MARKER" in public_prompt
        assert "HOME_B_BEHAVIOR_MARKER" not in public_prompt
    finally:
        provider_a.shutdown()


def test_read_gate_does_not_fallback_to_data_dir_parent_when_home_behavior_missing(
    tmp_path: Path,
) -> None:
    """Bug 3 boundary: missing BEHAVIOR.md in provider_home(self) must never fall back to data_dir.parent."""
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    home_a.mkdir()
    home_b.mkdir()

    (home_b / "BEHAVIOR.md").write_text(
        "**编号字母表**：①②③\n"
        "<!-- RENDER:system_prompt_block:BEGIN -->\n"
        "①② B_ONLY_BEHAVIOR_MARKER\n"
        "<!-- RENDER:system_prompt_block:END -->\n",
        encoding="utf-8",
    )

    data_in_b = home_b / "db-only"
    provider_a = plugin_entry.SanaeMemProvider(_base_config(data_in_b))
    with _private_scope("sess-no-leak"):
        provider_a.initialize(
            "sess-no-leak",
            hermes_home=str(home_a),
            platform="telegram",
            agent_context="primary",
        )
    try:
        with _private_scope("sess-no-leak"):
            prompt = provider_a.system_prompt_block()
        assert "B_ONLY_BEHAVIOR_MARKER" not in prompt
        status = json.loads((data_in_b / "status.json").read_text(encoding="utf-8"))
        assert status["behavior_block_ok"] is False
        assert status["behavior_block_reason"] == "missing_file"
    finally:
        provider_a.shutdown()


_ASSOCIATION_HISTORY = [{"role": "user", "content": "明天早晨有什么安排吗？"}]


@pytest.mark.skipif(HERMES_ROOT is None, reason="hermes-agent host source tree not found")
@pytest.mark.parametrize(
    ("scenario", "conversation_history"),
    [
        pytest.param("followup", None, id="followup-review"),
        pytest.param("followup", _ASSOCIATION_HISTORY, id="followup-planning"),
        pytest.param("first_reply", None, id="first-reply-and-healthy-busy"),
        pytest.param("switch_session", _ASSOCIATION_HISTORY, id="switch-session"),
        pytest.param("replace_owner", _ASSOCIATION_HISTORY, id="replace-owner"),
        pytest.param("revoke_owner", _ASSOCIATION_HISTORY, id="revoke-owner"),
    ],
)
def test_proactive_association_timeout_does_not_issue_second_model_request(
    scenario: str,
    conversation_history: list[dict[str, str]] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """The proactive hot path keeps its model budget and drops an invalidated handoff."""
    from hermes_cli.plugins import get_plugin_manager  # pyright: ignore[reportMissingImports]
    from plugins.memory import load_memory_provider  # pyright: ignore[reportMissingImports]

    home = tmp_path / "home"
    session_id = "sess-proactive"
    query = "去公园慢跑吃早饭"
    stale_query = "旧的公园慢跑吃早饭安排"
    measuring_association = threading.Event()
    first_request_entered = threading.Event()
    release_first_request = threading.Event()
    waiter_started = threading.Event()
    event_id = 0

    def _slow_then_normal(payload: dict[str, Any], call_index: int) -> dict[str, Any]:
        if call_index == 1 and measuring_association.is_set():
            first_request_entered.set()
            delay = {"first_reply": 2.70, "followup": 0.52}.get(scenario, 0.95)
            release_first_request.wait(timeout=delay)
        if "needs_history" in payload["messages"][0]["content"]:
            return {"needs_history": True, "query": "公园慢跑早饭"}
        return {"valid": True, "items": [{"id": event_id, "relationship": "related", "state": "history"}]}

    def _hold_first_body(call_index: int) -> None:
        if scenario != "first_reply" and call_index == 1 and measuring_association.is_set():
            release_first_request.wait(timeout=8.0)

    with (
        _LoopbackLLMServer(_slow_then_normal, before_response_body=_hold_first_body) as llm_server,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        # PluginLlm.complete() has no per-call retry toggle; auxiliary.transient_retries=0
        # disables host transport timeout retries so the server strictly counts logical calls.
        data_dir = _install_plugin_for_home(home, base_url=llm_server.base_url, transient_retries=0)
        with _host_home_env(home, monkeypatch):
            manager = get_plugin_manager()
            loaded = load_memory_provider("sanae_mem", register_skills=False)
            assert loaded is not None
            provider = cast(plugin_entry.SanaeMemProvider, loaded)
            try:
                with _private_scope(session_id):
                    provider.initialize(
                        session_id,
                        hermes_home=str(home),
                        platform="telegram",
                        agent_context="primary",
                    )
                    add_res = json.loads(
                        provider.handle_tool_call(
                            "sanae_mem_add",
                            {"content": "明天早晨打算去公园慢跑一圈再吃早饭。"},
                        )
                    )
                    assert add_res["ok"] is True and add_res["result"]["stored"] is True
                    event_id = int(add_res["result"]["id"])

                with sqlite3.connect(data_dir / "sanae.db") as conn:
                    owner, asked = conn.execute(
                        "SELECT owner_key, asked FROM events WHERE id=?", (event_id,),
                    ).fetchone()
                assert isinstance(owner, str) and owner and asked == 0
                current_owner: str | None = owner
                if scenario == "revoke_owner":
                    provider.association_state.bind_read_owner(owner, resolver=lambda: current_owner)

                def _assoc(
                    text: str,
                    sid: str,
                    *,
                    history: list[dict[str, str]] | None = None,
                    hot_path: bool = True,
                ) -> dict:
                    AssociationTurn = importlib.import_module(
                        "sanae_mem.cognitive.association"
                    ).AssociationTurn
                    with provider.association_state.lock:
                        rev = provider.association_state.revision
                    return provider.association.associate_context(
                        text,
                        conversation_history=history,
                        purpose="proactive",
                        session_id=sid,
                        hot_path=hot_path,
                        turn=AssociationTurn(
                            owner_key=owner,
                            daily_owner_key="",
                            daily_profile_key="",
                            daily_owner_platform="telegram",
                            agent_context="primary",
                            inject_max_chars=400,
                            llm=provider.runtime_llm(),
                        ),
                        revision=rev,
                    )

                def run_hot(text: str, sid: str, *, waiting: bool = False) -> tuple[dict, float]:
                    with _private_scope(sid):
                        started = monotonic()
                        if waiting:
                            waiter_started.set()
                        result = _assoc(text, sid, history=conversation_history, hot_path=True)
                        return result, monotonic() - started

                llm_server.requests.clear()
                measuring_association.set()
                first = pool.submit(copy_context().run, run_hot, query, session_id)
                assert first_request_entered.wait(2.0)
                if scenario == "first_reply":
                    busy, busy_elapsed = run_hot(query, session_id)
                    record_property("healthy_busy_seconds", busy_elapsed)
                    assert busy["status"] == "unavailable" and busy_elapsed < 0.40
                    assert not first.done()
                result, elapsed = first.result(timeout=3.3)
                record_property("first_seconds", elapsed)
                with llm_server.lock:
                    assert len(llm_server.requests) == 1
                if scenario == "first_reply":
                    assert 2.70 <= elapsed < 3.0
                    assert result["status"] == "ok" and result["valid"] is True
                    assert [item["id"] for item in result["items"]] == [event_id]
                    return
                assert result["status"] == "unavailable" and elapsed < 3.3

                if scenario == "followup":
                    # Late headers leave the host reading the first body after the foreground
                    # timeout. Keep that body held while a fresh call waits for worker cleanup.
                    threading.Event().wait(0.4)
                    with llm_server.lock:
                        requests_before_followup = list(llm_server.requests)
                    assert len(requests_before_followup) == 1
                    if conversation_history is not None:
                        assert not any(
                            "你在核对候选事件是否真的支撑一次情境联想"
                            in json.dumps(req, ensure_ascii=False)
                            for req in requests_before_followup
                        )

                else:
                    pending = pool.submit(copy_context().run, run_hot, stale_query, session_id, waiting=True)
                    assert waiter_started.wait(1.0)
                    # The first body is still held. A healthy-busy fallback would already return.
                    with pytest.raises(FutureTimeout):
                        pending.result(timeout=0.15)
                    with llm_server.lock:
                        assert len(llm_server.requests) == 1

                    invalidated_at = monotonic()
                    if scenario == "switch_session":
                        session_id = "sess-resumed"
                        provider.on_session_switch(session_id)
                    elif scenario == "replace_owner":
                        provider.association_state.bind_read_owner("person:replacement", resolver=None)
                    else:
                        current_owner = None
                        with _private_scope(session_id):
                            denied = _assoc(query, session_id, hot_path=False)
                        assert denied["status"] == "unavailable"
                    stale_result, waiting_elapsed = pending.result(timeout=0.30)
                    invalidation_elapsed = monotonic() - invalidated_at
                    record_property("handoff_seconds", waiting_elapsed)
                    record_property("after_invalidation_seconds", invalidation_elapsed)
                    assert invalidation_elapsed < 0.30
                    assert stale_result["status"] == "unavailable" and not stale_result["items"]
                    with llm_server.lock:
                        assert len(llm_server.requests) == 1
                    release_first_request.set()
                    if scenario != "switch_session":
                        provider.association_state.bind_read_owner(owner, resolver=None)

                followup, followup_elapsed = run_hot(query, session_id)
                if scenario != "followup":
                    # The cancelled waiter already claimed the handoff. Observe the old
                    # transport drain through public calls that make no POST while it is busy.
                    recover_by = monotonic() + 1.0
                    while followup["status"] == "unavailable" and monotonic() < recover_by:
                        threading.Event().wait(0.02)
                        followup, followup_elapsed = run_hot(query, session_id)
                record_property("followup_seconds", followup_elapsed)
                assert followup["status"] == "ok" and followup["valid"] is True
                assert [item["id"] for item in followup["items"]] == [event_id]
                assert followup_elapsed < 2.0
                with llm_server.lock:
                    expected_posts = 2 if conversation_history is None else 3
                    assert len(llm_server.requests) == expected_posts
                    assert all(stale_query not in json.dumps(req, ensure_ascii=False) for req in llm_server.requests)
                with sqlite3.connect(data_dir / "sanae.db") as conn:
                    assert conn.execute("SELECT owner_key, asked FROM events WHERE id=?", (event_id,)).fetchone() == (
                        owner, 0,
                    )
            finally:
                release_first_request.set()
                provider.shutdown()
                manager.unload("sanae_mem")


@pytest.mark.parametrize("backend", ["pyyaml", "ruamel"])
def test_load_host_config_sanitizes_yaml_error_logs_without_leaking_secrets(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue 027 #1: load_host_config logs only exception classes and positions, never secret values."""
    if backend == "ruamel":
        monkeypatch.setitem(sys.modules, "yaml", None)

    secret_val = "TOP_SECRET_API_KEY_987654"
    secret_other = "ANOTHER_SENSITIVE_VALUE_123"

    # 1. Explicit YAML tag value conversion error (ValueError carrying raw value in str(exc))
    tag_home = tmp_path / f"tag_{backend}"
    tag_home.mkdir()
    (tag_home / "config.yaml").write_text(
        f"api_key: !!int '{secret_val}'\n",
        encoding="utf-8",
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(tag_home) == {}
    assert secret_val not in caplog.text
    assert "api_key" not in caplog.text
    assert "invalid literal" not in caplog.text
    assert any(
        rec.levelno == logging.WARNING
        and "failed to parse host config" in rec.getMessage()
        and "HostConfigParseError" in rec.getMessage()
        and "ValueError" in rec.getMessage()
        for rec in caplog.records
    )

    # 2. Malformed YAML syntax with line/column mark (MarkedYAMLError carrying raw line snippet)
    syntax_home = tmp_path / f"syntax_{backend}"
    syntax_home.mkdir()
    (syntax_home / "config.yaml").write_text(
        f"secret_token: {secret_other}\nbroken: [unterminated\n",
        encoding="utf-8",
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(syntax_home) == {}
    assert secret_other not in caplog.text
    assert "unterminated" not in caplog.text
    assert any(
        rec.levelno == logging.WARNING
        and "failed to parse host config" in rec.getMessage()
        and "HostConfigParseError" in rec.getMessage()
        and "line " in rec.getMessage()
        for rec in caplog.records
    )

    # 3. Valid UTF-8 containing NUL byte -> ReaderError with position (never leaking secret)
    reader_home = tmp_path / f"reader_{backend}"
    reader_home.mkdir()
    (reader_home / "config.yaml").write_text(
        f"secret: {secret_val}\nbad: \x00hidden\n",
        encoding="utf-8",
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(reader_home) == {}
    assert secret_val not in caplog.text
    assert "hidden" not in caplog.text
    assert any(
        rec.levelno == logging.WARNING
        and "failed to parse host config" in rec.getMessage()
        and "HostConfigParseError" in rec.getMessage()
        and "ReaderError" in rec.getMessage()
        and "position " in rec.getMessage()
        for rec in caplog.records
    )

    # 4. Invalid UTF-8 bytes (UnicodeDecodeError)
    bad_utf8_home = tmp_path / f"utf8_{backend}"
    bad_utf8_home.mkdir()
    (bad_utf8_home / "config.yaml").write_bytes(
        b"secret: \xff\xfe" + secret_val.encode("ascii")
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=host_bridge.logger.name):
        assert host_bridge.load_host_config(bad_utf8_home) == {}
    assert secret_val not in caplog.text
    assert any(
        rec.levelno == logging.WARNING
        and "failed to read host config" in rec.getMessage()
        and "UnicodeDecodeError" in rec.getMessage()
        and "at byte " in rec.getMessage()
        for rec in caplog.records
    )


def test_on_session_switch_retries_rebind_after_transient_error_and_respects_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue 027 #2: on_session_switch rebinds on retry after transient error, but not after shutdown."""
    home = tmp_path / "home"
    home.mkdir()
    data_dir = tmp_path / "sanae-mem"
    provider = plugin_entry.SanaeMemProvider(_base_config(data_dir))
    with _private_scope("sess-1"):
        provider.initialize(
            "sess-1",
            hermes_home=str(home),
            platform="telegram",
            agent_context="primary",
        )
    try:
        assert plugin_entry.resolve_active_provider(home, "sess-1") is provider

        # Inside provider.on_session_switch(), _unbind_active_provider() is pure in-memory dict cleanup
        # and _bind_active_provider() -> register_active_provider() calls Path(self._hermes_home).resolve()
        # (strict=False), which is the sole OS I/O boundary (and strict=False swallows ELOOP symlink loops),
        # so transient stale-mount OSError is injected at the stdlib Path.resolve boundary.
        orig_resolve = Path.resolve

        def _failing_resolve(path_self: Path, strict: bool = False) -> Path:
            if str(path_self) == str(home):
                raise OSError("transient stale mount")
            return orig_resolve(path_self, strict=strict)

        monkeypatch.setattr(Path, "resolve", _failing_resolve)
        try:
            with pytest.raises(OSError, match="transient stale mount"):
                provider.on_session_switch("sess-2")
        finally:
            monkeypatch.setattr(Path, "resolve", orig_resolve)

        assert plugin_entry.resolve_active_provider(home, "sess-1") is None
        assert plugin_entry.resolve_active_provider(home, "sess-2") is None

        # Retry after transient failure must restore active provider binding
        provider.on_session_switch("sess-2")
        assert plugin_entry.resolve_active_provider(home, "sess-2") is provider

        # Hook workers must never bind on session switch
        worker = plugin_entry.SanaeMemProvider(_base_config(data_dir))
        with _private_scope("sess-hw"):
            worker.initialize(
                "sess-hw",
                hermes_home=str(home),
                platform="telegram",
                agent_context="primary",
                hook_worker=True,
            )
        try:
            worker.on_session_switch("sess-hw-2")
            assert plugin_entry.resolve_active_provider(home, "sess-hw-2") is None
        finally:
            worker.shutdown()
    finally:
        provider.shutdown()

    assert plugin_entry.resolve_active_provider(home, "sess-2") is None
    # Switching after shutdown must not resurrect active registration
    provider.on_session_switch("sess-after-shutdown")
    assert plugin_entry.resolve_active_provider(home, "sess-after-shutdown") is None


@pytest.mark.skipif(HERMES_ROOT is None, reason="hermes-agent host source tree not found")
def test_memory_manager_dispatched_on_session_switch_retries_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue 027 #2 (host dispatch): MemoryManager.on_session_switch failure then retry rebinds."""
    from agent.memory_manager import MemoryManager  # pyright: ignore[reportMissingImports]
    from hermes_cli.plugins import get_plugin_manager  # pyright: ignore[reportMissingImports]
    from plugins.memory import load_memory_provider  # pyright: ignore[reportMissingImports]

    home = tmp_path / "home"
    data_dir = _install_plugin_for_home(home, base_url="http://127.0.0.1:1/v1")
    with _host_home_env(home, monkeypatch):
        plugin_mgr = get_plugin_manager()
        mem_mgr = MemoryManager()
        loaded = load_memory_provider("sanae_mem", register_skills=False)
        assert loaded is not None
        try:
            with _private_scope("mgr-sess-1"):
                mem_mgr.add_provider(loaded)
                mem_mgr.initialize_all(
                    session_id="mgr-sess-1",
                    hermes_home=str(home),
                    platform="telegram",
                    agent_context="primary",
                )
                assert plugin_entry.resolve_active_provider(home, "mgr-sess-1") is loaded

                orig_resolve = Path.resolve

                def _failing_resolve(path_self: Path, strict: bool = False) -> Path:
                    if str(path_self) == str(home):
                        raise OSError("transient host mount error")
                    return orig_resolve(path_self, strict=strict)

                monkeypatch.setattr(Path, "resolve", _failing_resolve)
                try:
                    # MemoryManager swallows provider exceptions in _each_provider after unbind
                    mem_mgr.on_session_switch("mgr-sess-2")
                finally:
                    monkeypatch.setattr(Path, "resolve", orig_resolve)

                assert plugin_entry.resolve_active_provider(home, "mgr-sess-1") is None
                assert plugin_entry.resolve_active_provider(home, "mgr-sess-2") is None

                # Second host-dispatched switch must re-register the active provider
                mem_mgr.on_session_switch("mgr-sess-2")
                assert plugin_entry.resolve_active_provider(home, "mgr-sess-2") is loaded

                with _private_scope("mgr-sess-2"):
                    add_res = json.loads(
                        mem_mgr.handle_tool_call(
                            "sanae_mem_add",
                            {"content": "傍晚沿着河边慢跑了三公里。"},
                        )
                    )
                assert add_res["ok"] is True and add_res["result"]["stored"] is True
                with sqlite3.connect(str(data_dir / "sanae.db")) as conn:
                    sessions = [
                        row[0] for row in conn.execute("SELECT session_id FROM events").fetchall()
                    ]
                assert sessions == ["mgr-sess-2"]
        finally:
            mem_mgr.shutdown_all()
            plugin_mgr.unload("sanae_mem")

        assert plugin_entry.resolve_active_provider(home, "mgr-sess-2") is None
        mem_mgr.on_session_switch("mgr-sess-after-shutdown")
        assert plugin_entry.resolve_active_provider(home, "mgr-sess-after-shutdown") is None


def test_concurrent_multi_home_read_gate_isolation_stress(tmp_path: Path) -> None:
    """Stress: concurrent multi-HOME system_prompt_block() calls never leak cross-HOME behaviors."""
    home_count = 4
    homes: list[Path] = []
    markers: list[str] = []
    providers: list[plugin_entry.SanaeMemProvider] = []

    for idx in range(home_count):
        home = tmp_path / f"home_{idx}"
        home.mkdir()
        marker = f"ISOLATED_BEHAVIOR_HOME_{idx}_XYZ"
        (home / "BEHAVIOR.md").write_text(
            f"**编号字母表**：①②③\n"
            f"<!-- RENDER:system_prompt_block:BEGIN -->\n"
            f"① {marker}\n"
            f"<!-- RENDER:system_prompt_block:END -->\n",
            encoding="utf-8",
        )
        provider = plugin_entry.SanaeMemProvider(_base_config(home / "sanae-mem"))
        with _private_scope(f"sess-{idx}"):
            provider.initialize(
                f"sess-{idx}",
                hermes_home=str(home),
                platform="telegram",
                agent_context="primary",
            )
        homes.append(home)
        markers.append(marker)
        providers.append(provider)

    try:
        num_threads = 8
        iterations = 30
        barrier = threading.Barrier(num_threads)

        def _worker(tid: int) -> None:
            barrier.wait()
            own_idx = tid % home_count
            provider = providers[own_idx]
            own_marker = markers[own_idx]
            for step in range(iterations):
                distractor_home = homes[(own_idx + 1 + step) % home_count]
                with host_bridge.hermes_home_scope(distractor_home), _public_scope(f"sess-{own_idx}"):
                    prompt = provider.system_prompt_block()
                assert own_marker in prompt
                for other_idx, other_marker in enumerate(markers):
                    if other_idx != own_idx:
                        assert other_marker not in prompt

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_worker, tid) for tid in range(num_threads)]
            for fut in futures:
                fut.result()
    finally:
        for provider in providers:
            provider.shutdown()


def test_concurrent_on_session_switch_fault_recovery_stress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stress: verify mid-flight fault/retry rebind invariants and shared-provider contention."""
    shared_home = tmp_path / "shared_home"
    shared_home.mkdir()
    lane_count = 6
    lane_homes: list[Path] = []
    lane_providers: list[plugin_entry.SanaeMemProvider] = []
    for tid in range(lane_count):
        lane_home = tmp_path / f"lane_home_{tid}"
        lane_home.mkdir()
        lane_provider = plugin_entry.SanaeMemProvider(_base_config(lane_home / "sanae-mem"))
        with _private_scope(f"lane-{tid}-init"):
            lane_provider.initialize(
                f"lane-{tid}-init",
                hermes_home=str(lane_home),
                platform="telegram",
                agent_context="primary",
            )
        lane_homes.append(lane_home)
        lane_providers.append(lane_provider)

    shared_provider = plugin_entry.SanaeMemProvider(_base_config(tmp_path / "shared-sanae-mem"))
    with _private_scope("shared-init"):
        shared_provider.initialize(
            "shared-init",
            hermes_home=str(shared_home),
            platform="telegram",
            agent_context="primary",
        )

    orig_resolve = Path.resolve
    fail_once_targets: dict[int, str] = {}
    fail_lock = threading.Lock()
    shared_seen_sessions: list[str] = ["shared-init"]
    seen_lock = threading.Lock()

    def _flaky_resolve(path_self: Path, strict: bool = False) -> Path:
        tid = threading.get_ident()
        with fail_lock:
            target = fail_once_targets.get(tid)
            if target is not None and str(path_self) == target:
                fail_once_targets.pop(tid, None)
                should_fail = True
            else:
                should_fail = False
        if should_fail:
            raise OSError("intermittent mount failure")
        return orig_resolve(path_self, strict=strict)

    monkeypatch.setattr(Path, "resolve", _flaky_resolve)
    try:
        # Phase 1: 6 concurrent threads each validate fail -> unbound -> retry -> immediate rebind
        steps = 25
        phase1_barrier = threading.Barrier(lane_count)

        def _lane_worker(tid: int) -> None:
            phase1_barrier.wait()
            ident = threading.get_ident()
            home = lane_homes[tid]
            provider = lane_providers[tid]
            prev_sid = f"lane-{tid}-init"
            assert plugin_entry.resolve_active_provider(home, prev_sid) is provider
            for step in range(steps):
                sid = f"lane-{tid}-step-{step}"
                if (tid + step) % 2 == 0:
                    with fail_lock:
                        fail_once_targets[ident] = str(home)
                    with pytest.raises(OSError, match="intermittent mount failure"):
                        provider.on_session_switch(sid)
                    assert plugin_entry.resolve_active_provider(home, prev_sid) is None
                    assert plugin_entry.resolve_active_provider(home, sid) is None
                provider.on_session_switch(sid)
                assert plugin_entry.resolve_active_provider(home, sid) is provider
                assert plugin_entry.resolve_active_provider(home, prev_sid) is None
                prev_sid = sid

        with ThreadPoolExecutor(max_workers=lane_count) as pool:
            futs = [pool.submit(_lane_worker, tid) for tid in range(lane_count)]
            for fut in futs:
                fut.result()

        # Phase 2: 6 concurrent switchers + 2 readers on ONE shared provider; verify mid-flight
        # uniqueness across ALL registered sessions (multi-writer seqlock scans and safe checkpoints).
        num_switchers = 6
        num_readers = 2
        phase2_barrier = threading.Barrier(num_switchers + num_readers)
        checkpoint_barrier = threading.Barrier(num_switchers)
        stop_readers = threading.Event()
        active_writers = [0]
        write_seq = [0]

        def _shared_switcher(tid: int) -> None:
            phase2_barrier.wait(timeout=10.0)
            ident = threading.get_ident()
            for step in range(steps):
                sid = f"shared-t{tid}-{step}"
                with seen_lock:
                    shared_seen_sessions.append(sid)
                if (tid + step) % 3 == 0:
                    with fail_lock:
                        fail_once_targets[ident] = str(shared_home)
                    with seen_lock:
                        active_writers[0] += 1
                        write_seq[0] += 1
                    try:
                        with pytest.raises(OSError, match="intermittent mount failure"):
                            shared_provider.on_session_switch(sid)
                    finally:
                        with seen_lock:
                            active_writers[0] -= 1
                            write_seq[0] += 1
                with seen_lock:
                    active_writers[0] += 1
                    write_seq[0] += 1
                try:
                    shared_provider.on_session_switch(sid)
                finally:
                    with seen_lock:
                        active_writers[0] -= 1
                        write_seq[0] += 1
                if step % 5 == 4:
                    idx = checkpoint_barrier.wait(timeout=10.0)
                    try:
                        if idx == 0:
                            with seen_lock:
                                snapshot = list(shared_seen_sessions)
                            mid_winners = [
                                s
                                for s in snapshot
                                if plugin_entry.resolve_active_provider(shared_home, s)
                                is shared_provider
                            ]
                            assert len(mid_winners) == 1
                    except BaseException:
                        checkpoint_barrier.abort()
                        raise
                    checkpoint_barrier.wait(timeout=10.0)

        def _shared_reader() -> None:
            phase2_barrier.wait(timeout=10.0)
            while not stop_readers.is_set():
                with seen_lock:
                    w1 = active_writers[0]
                    s1 = write_seq[0]
                    snapshot = list(shared_seen_sessions)
                if w1 != 0:
                    continue
                hits = [
                    s
                    for s in snapshot
                    if plugin_entry.resolve_active_provider(shared_home, s) is shared_provider
                ]
                with seen_lock:
                    w2 = active_writers[0]
                    s2 = write_seq[0]
                if w2 == 0 and s1 == s2:
                    assert len(hits) <= 1

        with ThreadPoolExecutor(max_workers=num_switchers + num_readers) as pool:
            reader_futs = [pool.submit(_shared_reader) for _ in range(num_readers)]
            switcher_futs = [pool.submit(_shared_switcher, tid) for tid in range(num_switchers)]
            try:
                for fut in switcher_futs:
                    fut.result()
            finally:
                stop_readers.set()
            for fut in reader_futs:
                fut.result()

        active_winners = [
            sid
            for sid in shared_seen_sessions
            if plugin_entry.resolve_active_provider(shared_home, sid) is shared_provider
        ]
        assert len(active_winners) == 1
        for sid in shared_seen_sessions:
            if sid != active_winners[0]:
                assert plugin_entry.resolve_active_provider(shared_home, sid) is None
    finally:
        monkeypatch.setattr(Path, "resolve", orig_resolve)
        for lane_provider in lane_providers:
            lane_provider.shutdown()
        shared_provider.shutdown()

    for sid in shared_seen_sessions:
        assert plugin_entry.resolve_active_provider(shared_home, sid) is None
    shared_provider.on_session_switch("shared-post-shutdown")
    assert plugin_entry.resolve_active_provider(shared_home, "shared-post-shutdown") is None
