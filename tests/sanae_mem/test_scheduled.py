"""Run the installed private journal through the pinned original Hermes scheduler."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import yaml


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
if not HERMES_ROOT.is_dir():
    pytest.skip("pinned original Hermes source is unavailable", allow_module_level=True)
DEPENDENCIES = HERMES_ROOT.with_name(HERMES_ROOT.name + "-deps")
if DEPENDENCIES.is_dir():
    sys.path.insert(0, str(DEPENDENCIES))
sys.path.insert(0, str(HERMES_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins"))

from cron import jobs, scheduler
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from sanae_mem.infra.clock import sleep_day_iso
from sanae_mem.governance.memory_guard import builtin_memory_flags
from sanae_mem.cognitive.nightly_journal import is_journal_entry, journal_entry_day, journal_entry_for
from sanae_mem.plugin_entry import SanaeMemProvider
from sanae_mem.runtime.scheduled import run_installed_nightly_journal, sync_scheduled_jobs


assert Path(scheduler.__file__).resolve().is_relative_to(HERMES_ROOT)


@pytest.fixture
def free_think_model():
    prompts: list[str] = []

    class ModelIO(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompts.append("\n".join(str(message["content"]) for message in request["messages"]))
            reply = {"role": "assistant", "content": "[SILENT]"}
            if request.get("stream"):
                event = {"choices": [{"index": 0, "delta": reply, "finish_reason": "stop"}]}
                body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps({"id": "free-think-test", "model": request["model"],
                                   "choices": [{"index": 0, "message": reply, "finish_reason": "stop"}]}).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), ModelIO) as model:
        thread = threading.Thread(target=model.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{model.server_port}/v1", prompts
        finally:
            model.shutdown()
            thread.join()


@pytest.mark.parametrize("native_hook", [False, True, None], ids=["fallback", "native-hook", "installed-host"])
def test_free_think_context_uses_native_only_for_an_active_session(tmp_path, monkeypatch, native_hook, free_think_model):
    from agent.memory_manager import MemoryManager
    from cron.executions import create_execution, mark_execution_running
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, get_plugin_manager
    from hermes_host_contracts import CronExecutionEnvelope, TurnEnvelope, cron_envelope_scope, turn_envelope_scope
    from plugins.memory import load_memory_provider
    from run_agent import AIAgent
    from sanae_mem.infra import host_bridge
    from sanae_mem.governance.account_binding import maintainer_anchor
    from sanae_mem.plugin_entry import resolve_active_provider
    from sanae_mem.runtime.scheduled import register_free_think_hooks

    installed = native_hook is None
    native = host_bridge.has_native_turn_context() if native_hook is None else native_hook
    base_url, prompts = free_think_model
    home = tmp_path / "hermes"
    shutil.copytree(Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem", home / "plugins" / "sanae_mem",
                    ignore=shutil.ignore_patterns("tests", "__pycache__"))
    config = {"free_think": True, "semantic_backend": "sqlite",
              "memory_identity": {"platform": "telegram", "user_id": "reflection-owner"}}
    def set_shadow(shadow):
        (home / "config.yaml").write_text(yaml.safe_dump({
            "memory": {"provider": "sanae_mem", "memory_enabled": False, "user_profile_enabled": False},
            "model": {"default": "vm-free-think", "provider": "custom", "base_url": base_url,
                      "context_length": 64000},
            "plugins": {"enabled": ["sanae_mem"], "sanae_mem": {**config, "shadow": shadow}},
        }), encoding="utf-8")

    set_shadow(installed)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    token = set_hermes_home_override(str(home))
    manager = get_plugin_manager() if installed else PluginManager(scope_key=str(home))
    memory = MemoryManager()
    provider = None
    agent = None
    try:
        if installed:
            manager.discover_and_load()
            assert any(row["name"] == "sanae_mem" and row["enabled"] for row in manager.list_plugins())
            assert jobs.list_jobs() == []
            provider = load_memory_provider("sanae_mem", register_skills=False)
            assert provider is not None
            set_shadow(False)
            with turn_envelope_scope(TurnEnvelope(
                platform="telegram", chat_type="dm", chat_id="reflection-owner", sender_id="reflection-owner",
                session_id="schedule-resume", message_id="resume", session_key="agent:main:telegram:dm:reflection-owner",
            )):
                manager.invoke_hook("pre_llm_call", session_id="schedule-resume",
                                    turn_id="resume", platform="telegram", user_message="")
        else:
            ctx = PluginContext(PluginManifest(name="sanae_mem"), manager)
            ctx.on_unload(register_free_think_hooks(ctx, SanaeMemProvider(config), native_turn_context=native))
            sync_scheduled_jobs(config)
        job, = jobs.list_jobs()
        job_prompt = job["prompt"]
        assert isinstance(job_prompt, str)
        set_shadow(False)
        execution = create_execution(job["id"], source="scheduled")
        mark_execution_running(execution["id"])
        owner = maintainer_anchor(config)
        if provider is None:
            provider = load_memory_provider("sanae_mem", register_skills=False)
        assert provider is not None
        assert type(provider).__module__.startswith("_hermes_user_memory.")
        memory.add_provider(provider)
        with cron_envelope_scope(CronExecutionEnvelope(
            job_id=job["id"], job_name=job["name"], execution_id=execution["id"],
            session_id="free-think", profile="default", profile_home=str(home),
            bound_owner=owner,
        )):
            marker = "我还要给对方留一本书"
            seed = SanaeMemProvider(config)
            seed.initialize("seed-cron", platform="cron", agent_context="cron",
                            hermes_home=home, hook_worker=True)
            try:
                assert seed.record_commitment(owner, marker) is not None
            finally:
                seed.shutdown()
            def emitted_context(session_id: str = "free-think"):
                replies = manager.invoke_hook(
                    "pre_llm_call", session_id=session_id, turn_id="turn-1", platform="cron",
                    user_message="", sender_id="",
                )
                return [result for result in replies
                        if isinstance(result, str) or isinstance(result, dict) and "context" in result]

            def make_agent(session_id: str, memory_manager: MemoryManager | None = None) -> AIAgent:
                return AIAgent(
                    model="vm-free-think", provider="custom", base_url=base_url, api_key="loopback-fixture",
                    platform="cron", session_id=session_id, memory_manager=memory_manager,
                    enabled_toolsets=[], quiet_mode=True, skip_background_review=True,
                )

            def wire_context_count(turn_agent: AIAgent, session_id: str, *, visitor: bool = False) -> int:
                tokens = set_session_vars(
                    platform="telegram" if visitor else "cron", session_id=session_id,
                    chat_type="dm" if visitor else "", chat_id="visitor" if visitor else "",
                    user_id="visitor" if visitor else "",
                    session_key="agent:main:telegram:dm:visitor" if visitor else "",
                )
                try:
                    prompts.clear()
                    result = turn_agent.run_conversation(job_prompt, conversation_history=[])
                    assert result["final_response"] == "[SILENT]"
                    prompt, = prompts
                    assert prompt.count("[我内心的感受]") == prompt.count(marker)
                    assert prompt.count("[我眼中的世界]") == prompt.count(marker)
                    return prompt.count(marker)
                finally:
                    clear_session_vars(tokens)

            assert resolve_active_provider(home, "free-think") is None
            fallback, = emitted_context()
            assert isinstance(fallback, dict)
            assert marker in fallback["context"]
            assert fallback == {"context": fallback["context"], "ephemeral": True}
            set_shadow(True)
            memory.initialize_all(session_id="free-think", platform="cron", agent_context="cron",
                                  hermes_home=home)
            assert emitted_context() == []
            if installed:
                agent = make_agent("free-think", memory)
                assert wire_context_count(agent, "free-think") == 0
            set_shadow(False)
            active_provider = resolve_active_provider(home, "free-think")
            assert active_provider is not None and active_provider is provider
            active_hook_context = [] if native else [fallback]
            assert emitted_context() == active_hook_context
            if agent is not None:
                assert wire_context_count(agent, "free-think") == 1
            assert emitted_context("other-session") == []
            if agent is not None:
                other = make_agent("other-session")
                try:
                    assert wire_context_count(other, "other-session") == 0
                finally:
                    other.close()
            set_shadow(True)
            assert resolve_active_provider(home, "free-think") is None
            assert emitted_context() == []
            if agent is not None:
                assert wire_context_count(agent, "free-think") == 0
            set_shadow(False)
            assert resolve_active_provider(home, "free-think") is active_provider
            assert emitted_context() == active_hook_context
            if agent is not None:
                assert wire_context_count(agent, "free-think") == 1
        with turn_envelope_scope(TurnEnvelope(
            platform="telegram", chat_type="dm", chat_id="visitor", sender_id="visitor",
            session_id="free-think", session_key="agent:main:telegram:dm:visitor",
        )):
            assert emitted_context() == []
            if agent is not None:
                assert wire_context_count(agent, "free-think", visitor=True) == 0
    finally:
        if agent is None:
            memory.shutdown_all()
        else:
            agent.close()
        manager.unload()
        reset_hermes_home_override(token)


def test_installed_cron_jobs_pause_resume_and_retry_after_shadow_changes(tmp_path, monkeypatch):
    from hermes_cli.plugins import get_plugin_manager
    from hermes_host_contracts import TurnEnvelope, turn_envelope_scope
    from plugins.memory import load_memory_provider

    home = tmp_path / "hermes"
    shutil.copytree(Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem", home / "plugins" / "sanae_mem",
                    ignore=shutil.ignore_patterns("tests", "__pycache__"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(str(home))
    manager = get_plugin_manager()

    def set_shadow(shadow, *, free_think=True):
        (home / "config.yaml").write_text(yaml.safe_dump({
            "memory": {"provider": "sanae_mem"},
            "plugins": {"enabled": ["sanae_mem"], "sanae_mem": {
                "semantic_backend": "sqlite", "nightly_journal": True, "free_think": free_think, "shadow": shadow,
                "memory_identity": {"platform": "telegram", "user_id": "4242"},
            }},
        }), encoding="utf-8")

    def next_turn(callback=None):
        with turn_envelope_scope(TurnEnvelope(
            platform="telegram", chat_type="dm", chat_id="4242", sender_id="4242",
            session_id="existing-session", message_id="message", session_key="agent:main:telegram:dm:4242",
        )):
            turn = dict(session_id="existing-session", turn_id="turn", platform="telegram", user_message="")
            if callback is None:
                manager.invoke_hook("pre_llm_call", **turn)
            else:
                callback(**turn)
        return [job for job in jobs.list_jobs(include_disabled=True) if job["name"].startswith("ViveMind:")]

    try:
        set_shadow(True)
        manager.discover_and_load()
        assert next_turn() == []
        launcher = home / "scripts" / "vivemind-nightly.py"
        assert not launcher.exists()
        launcher.parent.mkdir(exist_ok=True)
        launcher.write_text("occupied path", encoding="utf-8")
        set_shadow(False)
        assert next_turn() == []
        launcher.unlink()

        normal = next_turn()
        assert {job["script"] for job in normal} == {None, "vivemind-nightly.py"}
        assert all(job["enabled"] for job in normal)
        task_ids = {job["id"] for job in normal}
        set_shadow(True)
        paused = next_turn()
        assert {job["id"] for job in paused} == task_ids
        assert all(not job["enabled"] for job in paused)
        set_shadow(False)
        resumed = next_turn()
        assert {job["id"] for job in resumed} == task_ids
        assert all(job["enabled"] for job in resumed)

        held_provider = load_memory_provider("sanae_mem", register_skills=False)
        assert held_provider is not None
        try:
            set_shadow(False, free_think=False)
            before_reload = next_turn()
            assert {job["id"] for job in before_reload} == task_ids
            assert all(job["enabled"] for job in before_reload)
            manager.discover_and_load(force=True)
            after_reload = jobs.list_jobs(include_disabled=True)
            assert {job["id"] for job in after_reload} == task_ids
            assert [job["script"] for job in after_reload if job["enabled"]] == ["vivemind-nightly.py"]

            set_shadow(False)
            assert manager.unload("sanae_mem")
            held_hook = getattr(held_provider, "handle_pre_llm_call", None)
            assert callable(held_hook)
            restored = next_turn(held_hook)
            assert {job["id"] for job in restored} == task_ids
            assert all(job["enabled"] for job in restored)
            manager.discover_and_load(force=True)
        finally:
            held_provider.shutdown()

        free_think, = [job for job in resumed if not job["no_agent"]]
        jobs.pause_job(free_think["id"], reason="operator maintenance")
        set_shadow(True)
        next_turn()
        set_shadow(False)
        enabled, = [job for job in next_turn() if job["enabled"]]
        assert enabled["script"] == "vivemind-nightly.py"
        operator_paused = jobs.get_job(free_think["id"])
        assert operator_paused is not None and operator_paused["paused_reason"] == "operator maintenance"
    finally:
        manager.unload()
        reset_hermes_home_override(token)


def test_free_think_delivers_only_to_bound_private_target(tmp_path, monkeypatch):
    from functools import partial

    from cron.scheduler_delivery import _deliver_result
    from hermes_host_contracts import (
        query_verified_cron_sends, register_cron_pre_delivery_gate, unregister_cron_pre_delivery_gate,
    )
    from sanae_mem.governance.cognitive_context import gate_proactive_cron_delivery
    from tools import send_message_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:loopback-test-token")
    config = {"free_think": True, "memory_identity": {"platform": "telegram", "user_id": "4242"}}
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"sanae_mem": config}}), encoding="utf-8",
    )
    token = set_hermes_home_override(str(tmp_path))
    gate = partial(gate_proactive_cron_delivery, config=config)
    register_cron_pre_delivery_gate(gate)
    sends = []

    async def send(platform, _config, target, message, **_):
        sends.append((platform.value, target))
        if "send failed" in message:
            return {"error": "fixture transport refused the send"}
        return {"success": True, "message_id": "fixture-telegram-message"}

    monkeypatch.setattr(send_message_tool, "_send_to_platform", send)
    try:
        sync_scheduled_jobs(config)
        job = jobs.list_jobs(include_disabled=True)[0]
        assert _deliver_result(dict(job, execution_id="rebound", deliver="telegram:4646"), "wrong route") is None
        assert _deliver_result(dict(job, execution_id="silent"), "[SILENT]") is None
        assert sends == []
        assert query_verified_cron_sends() == []

        assert _deliver_result(dict(job, execution_id="sent"), "a private note") is None
        assert sends == [("telegram", "4242")]
        receipts = query_verified_cron_sends()
        assert [(row["execution_id"], row["message_id"]) for row in receipts] == [
            ("sent", "fixture-telegram-message"),
        ]

        assert _deliver_result(dict(job, execution_id="send-refused"), "send failed")
        assert sends == [("telegram", "4242"), ("telegram", "4242")]
        assert query_verified_cron_sends() == receipts
    finally:
        unregister_cron_pre_delivery_gate(gate)
        reset_hermes_home_override(token)


def test_installed_nightly_script_and_same_day_restart_preserve_one_private_journal(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "plugins").mkdir(parents=True)
    (home / "memories").mkdir()
    plugin = Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem"
    shutil.copytree(plugin, home / "plugins" / "sanae_mem",
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    config = {
        "semantic_backend": "sqlite",
        "nightly_journal": True,
        "memory_identity": {"platform": "telegram", "user_id": "journal-owner"},
    }
    (home / "config.yaml").write_text(yaml.safe_dump({
        "memory": {"provider": "sanae_mem", "memory_enabled": False,
                   "user_profile_enabled": False},
        "plugins": {"enabled": ["sanae_mem"], "sanae_mem": config},
    }), encoding="utf-8")
    private_user = "PRIVATE_USER_NOTE_DO_NOT_READ_OR_REWRITE"
    (home / "memories" / "USER.md").write_text(private_user, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    token = set_hermes_home_override(str(home))
    try:
        assert builtin_memory_flags() == (False, False)
        sync_scheduled_jobs(config)
        sync_scheduled_jobs(config)
        installed = jobs.list_jobs(include_disabled=True)
        assert len(installed) == 1
        job = installed[0]
        assert job["no_agent"] is True
        assert job["deliver"] == job["failure_deliver"] == "local"
        assert job["timezone"] == "Asia/Shanghai"
        assert job["schedule"]["expr"] == "0 23 * * *"
        launcher = home / "scripts" / job["script"]
        assert launcher.is_file()

        # This executes the installed script in a genuine child Python process.
        # Before the closing hour it must refuse; after it, it may close the same day.
        success, output, final_response, error = scheduler.run_job(job)
        assert private_user not in output + final_response
        if not success:
            assert "before_close" in str(error), output
        else:
            assert "ViveMind nightly journal: complete" in final_response
        from cron.scheduler_delivery import _resolve_delivery_targets
        assert _resolve_delivery_targets(job) == []
        assert _resolve_delivery_targets(job, for_failure=True) == []
        assert scheduler._deliver_result(job, final_response) is None
        assert scheduler._deliver_result(job, "test failure notice", for_failure=True) is None

        from tools.memory_tool import MemoryStore
        read_file = MemoryStore._read_file

        def read_memory_only(path):
            assert Path(path).name != "USER.md", "nightly must not load the disabled user notebook"
            return read_file(path)

        monkeypatch.setattr(MemoryStore, "_read_file", staticmethod(read_memory_only))
        now = datetime.now(ZoneInfo("Asia/Shanghai")).replace(
            hour=23, minute=1, second=0, microsecond=0)
        first = run_installed_nightly_journal(SanaeMemProvider(config), now=now)
        assert first["ok"], first
        second = run_installed_nightly_journal(SanaeMemProvider(config), now=now)
        assert second["ok"] and second["already"], second
        sync_scheduled_jobs(config)
        assert len(jobs.list_jobs(include_disabled=True)) == 1

        entries = MemoryStore._read_file(home / "memories" / "MEMORY.md")
        journals = [entry for entry in entries if is_journal_entry(entry)]
        assert len(journals) == 1
        assert journal_entry_day(journals[0]) == sleep_day_iso(now)
        assert private_user not in journals[0]
        assert "sources" not in first or first["sources"] == 0
        assert first.get("via", "quiet_day") == "quiet_day"
        assert (home / "memories" / "USER.md").read_text(encoding="utf-8") == private_user
        assert builtin_memory_flags() == (False, False)
        with sqlite3.connect(home / "sanae-mem" / "sanae.db") as db:
            rows = db.execute("SELECT day, status FROM nightly_journal_runs").fetchall()
        assert rows == [(sleep_day_iso(now), "completed")]
    finally:
        reset_hermes_home_override(token)


def test_nightly_archive_failure_preserves_resident_entry_and_retry_sinks_it(tmp_path, monkeypatch):
    from sanae_mem.governance.account_binding import maintainer_anchor
    from sanae_mem.infra.host_bridge import turn_envelope_scope

    home = tmp_path / "hermes"
    memory = home / "memories" / "MEMORY.md"
    archive = home / "memory-archive" / "journal.jsonl"
    memory.parent.mkdir(parents=True)
    archive.mkdir(parents=True)
    old_entry = journal_entry_for("2025-01-01", "这是用于归档测试的旧小记。")
    original = old_entry.encode("utf-8")
    memory.write_bytes(original)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    config = {"semantic_backend": "sqlite", "persona_name": "夜记测试",
              "memory_identity": {"platform": "telegram", "user_id": "archive-test-owner"}}
    provider = SanaeMemProvider(config)
    owner = maintainer_anchor(config)
    now = datetime(2026, 11, 10, 23, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    provider.initialize("archive-test", hermes_home=home, hook_worker=True)
    try:
        with turn_envelope_scope(platform="cron", session_id="archive-test"):
            failed = provider.run_nightly_journal(now=now)
        assert failed == {"ok": False, "reason": "archive_failed", "day": sleep_day_iso(now)}
        assert memory.read_bytes() == original
        assert provider.sqlite.fetchone(
            "SELECT id FROM events WHERE promoted_reason='nightly_journal_sink'",
        ) is None
        attempt = provider.sqlite.fetchone("SELECT status, last_error FROM nightly_journal_runs")
        assert attempt is not None and tuple(attempt) == ("failed", "archive_failed")

        archive.rmdir()
        with turn_envelope_scope(platform="cron", session_id="archive-test"):
            complete = provider.run_nightly_journal(now=now)
        assert complete["ok"] and complete["expired"] == 1
        archived, = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
        assert (archived["entry"], archived["owner_key"], archived["reason"]) == (
            old_entry, owner, "journal_window",
        )
        assert memory.read_text(encoding="utf-8") == complete["entry"]
        sunk = provider.sqlite.fetchone(
            "SELECT raw, owner_key, session_id FROM events WHERE promoted_reason='nightly_journal_sink'",
        )
        assert sunk is not None and tuple(sunk) == (old_entry, owner, "archive-test")
        attempt = provider.sqlite.fetchone("SELECT status, attempts FROM nightly_journal_runs")
        assert attempt is not None and tuple(attempt) == ("completed", 4)
    finally:
        provider.shutdown()


def test_original_cron_free_think_upgrades_and_keeps_diary_private(tmp_path, monkeypatch):
    from cron.executions import list_executions
    from gateway import session_cron
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from hermes_state_registry import acquire, release_or_close
    from hermes_state_runtime import begin_runtime_epoch
    from plugins.memory import load_memory_provider
    from sanae_mem.cognitive.daily_context import daily_profile_key
    from sanae_mem.cognitive.daily_rules import KIND_THREAD
    from sanae_mem.cognitive.inner_state_store import append_state, read_state
    from sanae_mem.cognitive.life_context import collect_life_experiences, collect_recent_activities
    from sanae_mem.cognitive.inner_state_types import InnerState
    from sanae_mem.runtime.scheduled import _FREE_THINK_PROMPT, _FREE_THINK_TOOLSETS, _LEGACY_FREE_THINK_PROMPT

    diary_note = "我想到窗台上的光会顺着书脊慢慢挪过去。"
    saved_event = "我把有折角的那本书放到了窗边。"
    rejected = "这次没有完成，不能当成已经发生的事。"
    thought = "窗台上的那点光还在心里。我想慢一点再翻页。"
    journal = "我想到窗台上的光会顺着书脊慢慢挪过去，想慢一点再翻页。"
    replies = deque([None, "diary", "[SILENT]", "[CRON_FAILURE]\n" + rejected, "[SILENT]"])

    main_requests, heart_requests, journal_requests = [], [], []

    class ModelIO(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.respond({"data": [{"id": "vm-free-think"}, {"id": "vm-heart"}]})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "messages" not in request:
                self.respond({"name": "vm-free-think"})
                return
            model = request.get("model")
            tool_calls = []
            if model == "vm-heart" and "当天片段：" in request["messages"][-1]["content"]:
                journal_requests.append(request)
                content = journal
            elif model == "vm-heart":
                heart_requests.append(request)
                content = json.dumps({"sentences": [thought], "shared_sentences": None,
                                      "variables": {"tension": 0.1, "closeness": 0.2, "energy": 0.3}},
                                     ensure_ascii=False)
            else:
                main_requests.append(request)
                content = replies.popleft() if replies else "[SILENT]"
                if content is None:
                    tool_calls = [{"id": "private-memory-check", "type": "function",
                                   "function": {"name": "sanae_mem_search",
                                                "arguments": '{"query":"折角的那本书"}'}}]
                if content == "diary":
                    content = None
                    tool_calls = [{"id": "private-diary-check", "type": "function",
                                   "function": {"name": "sanae_free_think_diary",
                                                "arguments": json.dumps({"text": diary_note})}}]
            finish = "tool_calls" if tool_calls else "stop"
            message = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            if request.get("stream"):
                delta = dict(message)
                if tool_calls:
                    delta["tool_calls"] = [{"index": 0, **call} for call in tool_calls]
                chunks = [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}]
                data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                self.respond(data, "text/event-stream")
            else:
                self.respond({"id": "fixture-completion", "object": "chat.completion", "model": model,
                              "created": 1, "choices": [{"index": 0, "finish_reason": finish,
                                                        "message": message}],
                              "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})

        def respond(self, payload, content_type="application/json"):
            body = (payload if isinstance(payload, str) else json.dumps(payload)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    home = tmp_path / "hermes"
    plugin = Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem"
    shutil.copytree(plugin, home / "plugins" / "sanae_mem",
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    config = {"semantic_backend": "sqlite", "free_think": True, "nightly_journal": True,
              "memory_identity": {"platform": "telegram", "user_id": "reflection-owner"}}
    (home / "memories").mkdir()
    memory_note = "本次实测的常驻笔记：我记着一本书。"
    user_note = "本次实测的私有资料：对方喜欢纸质书。"
    (home / "memories" / "MEMORY.md").write_text(memory_note, encoding="utf-8")
    (home / "memories" / "USER.md").write_text(user_note, encoding="utf-8")
    curiosity = "想顺手了解纸张为什么会泛黄。"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("OPENAI_API_KEY", "loopback-fixture")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:loopback-test-token")
    token = set_hermes_home_override(str(home))
    with ThreadingHTTPServer(("127.0.0.1", 0), ModelIO) as model:
        thread = threading.Thread(target=model.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{model.server_port}/v1"
        (home / ".env").write_text("OPENAI_API_KEY=loopback-fixture\nTAVILY_API_KEY=fixture-unused\n",
                                  encoding="utf-8")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "model": {"default": "vm-free-think", "provider": "custom", "base_url": url,
                      "context_length": 64000},
            "auxiliary": {"sanae_daily_flash": {"provider": "custom", "model": "vm-heart",
                                                "base_url": url, "api_mode": "chat_completions"}},
            "web": {"search_backend": "tavily"},
            "tools": {"tool_search": {"enabled": "off"}},
            "memory": {"provider": "sanae_mem", "memory_enabled": False,
                       "user_profile_enabled": False},
            "plugins": {"enabled": ["sanae_mem"], "sanae_mem": config},
        }), encoding="utf-8")
        try:
            discover_plugins()
            loaded = next(plugin for plugin in get_plugin_manager()._plugins.values()
                          if plugin.manifest.name == "sanae_mem")
            wait_for_reflection = importlib.import_module(loaded.module.__name__ + ".cognitive.inner_wiring")._wait_for_pending
            installed = [job for job in jobs.list_jobs(include_disabled=True) if not job["no_agent"]]
            assert len(installed) == 1
            job = installed[0]
            toolsets = _FREE_THINK_TOOLSETS
            assert not job["no_agent"] and job["enabled_toolsets"] == toolsets
            assert job["deliver"] == "telegram:reflection-owner"
            assert job["failure_deliver"] == "local"
            assert job["timezone"] == "Asia/Shanghai" and job["schedule"]["expr"] == "0 8,16,22 * * *"
            old_job = jobs.update_job(job["id"], {"schedule": "30 22 * * *", "enabled_toolsets": [],
                                                  "prompt": _LEGACY_FREE_THINK_PROMPT, "deliver": "local",
                                                  "bound_owner": None, "activity_class": None})
            assert old_job is not None and old_job["enabled_toolsets"] == []

            provider = SanaeMemProvider(config)
            provider.initialize("setup", agent_context="cron", platform="cron",
                                hermes_home=home, hook_worker=True)
            assert provider.system_prompt_block() == ""
            assert provider.get_tool_schemas() == []
            assert json.loads(provider.handle_tool_call(
                "sanae_mem_search", {"query": "书"}, session_id="setup",
            ))["ok"] is False
            owner = provider._daily_owner_key
            target = (home / "sanae-mem", daily_profile_key(home))
            database = target[0] / "sanae.db"
            prior = time.time() - 60
            seed = SanaeMemProvider(config)
            seed_tokens = set_session_vars(
                platform="telegram", chat_type="dm", chat_id="fixture-chat",
                user_id="reflection-owner", session_id="earlier-private-chat",
                session_key="agent:main:telegram:dm:fixture-chat",
            )
            try:
                seed.initialize("earlier-private-chat", platform="telegram",
                                hermes_home=home, hook_worker=True)
                seeded = json.loads(seed.handle_tool_call(
                    "sanae_mem_add", {"content": saved_event, "importance": 0.9},
                ))
                assert seeded["ok"] and seeded["result"]["stored"], seeded
            finally:
                seed.shutdown()
                clear_session_vars(seed_tokens)
            assert provider.open_daily_entry(
                owner_key=owner, profile_key=target[1], kind=KIND_THREAD, slot="fixture-curiosity",
                text=curiosity, opened_on=sleep_day_iso(), now=datetime.now().astimezone().isoformat(),
            )
            assert append_state(database, target, InnerState(
                revision=1, text="桌上的书还没读完。", recent_words="", recent_words_at=None,
                variables={"tension": 0.0, "closeness": 0.0, "energy": 0.0},
                felt_at=prior, updated_at=prior, processed_at=prior, source="flash", thought_source="flash",
                turn_id="earlier-private-delivery", changed=True, owner_key=owner, text_kind="private",
            ))

            async def fire_jobs():
                db = acquire(home / "state.db")
                try:
                    gateway_config = GatewayConfig()
                    runner = SimpleNamespace(_draining=False, config=gateway_config, adapters={},
                                             session_store=SessionStore(home / "sessions", gateway_config))
                    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
                    authority = SessionAuthority(
                        runner, profile_id=str(home), instance_id="free-think-fixture", db=db,
                        epoch=begin_runtime_epoch(db, instance_id="free-think-fixture"),
                    )
                    runner.session_authority = authority
                    session_cron.bind_owner(authority)
                    try:
                        sync_scheduled_jobs(config)
                        sync_scheduled_jobs(config)
                        assert len([j for j in jobs.list_jobs(include_disabled=True) if not j["no_agent"]]) == 1
                        upgraded = jobs.resolve_job_ref(job["name"])
                        assert upgraded["id"] == job["id"]
                        assert upgraded["enabled_toolsets"] == toolsets
                        assert upgraded["schedule"]["expr"] == "0 8,16,22 * * *"
                        assert upgraded["deliver"] == "telegram:reflection-owner"
                        assert upgraded["failure_deliver"] == "local"
                        assert upgraded["prompt"] == _LEGACY_FREE_THINK_PROMPT
                        next_run = datetime.fromisoformat(upgraded["next_run_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
                        assert next_run.hour in {8, 16, 22} and next_run.minute == 0
                        for expected_status in ("completed", "failed", "completed"):
                            await asyncio.wait_for(asyncio.to_thread(scheduler._run_one_job_body, upgraded), 20)
                            record = list_executions(job_id=job["id"], limit=1)[0]
                            assert record["status"] == expected_status, record
                            assert record["delivery_outcome"] == "suppressed", record
                            await asyncio.to_thread(wait_for_reflection, target)
                    finally:
                        session_cron.unbind_owner(authority)
                finally:
                    release_or_close(db)

            try:
                asyncio.run(fire_jobs())
                view = read_state(database, target[1], internal=True)
                assert view is not None and view.private is not None
                assert view.revision == 3 and view.processed_at == prior
                assert thought in view.private.text
                assert len(main_requests) == 5
                first_exploration = main_requests[0]
                names = {tool["function"]["name"] for tool in first_exploration["tools"]}
                assert {"sanae_mem_search", "web_search", "session_search", "sanae_free_think_diary"} <= names
                assert names.isdisjoint({"send_message", "terminal", "write_file", "patch",
                                         "cronjob_manage", "browser_navigate", "read_file"})
                assert not any(name.startswith("mcp_") for name in names)
                prompt = "\n".join(str(msg["content"]) for msg in first_exploration["messages"]
                                   if msg["role"] == "system")
                assert all(marker in prompt for marker in ("MEMORY.md", memory_note, user_note,
                                                           "[我留着心眼想顺手了解的事]", curiosity))
                rendered_messages = "\n".join(
                    str(msg.get("content") or "") for msg in first_exploration["messages"]
                )
                assert _FREE_THINK_PROMPT in rendered_messages
                assert _LEGACY_FREE_THINK_PROMPT not in rendered_messages
                next_cycle = "\n".join(str(msg.get("content") or "") for msg in main_requests[-1]["messages"])
                assert next_cycle.count("[我内心的感受]") == 1
                assert thought in next_cycle
                result = next(msg for msg in main_requests[1]["messages"]
                              if msg.get("tool_call_id") == "private-memory-check")
                assert json.loads(result["content"])["ok"] is True
                assert saved_event in result["content"]
                assert heart_requests
                for request in heart_requests:
                    heartbeat = json.loads(request["messages"][-1]["content"])
                    assert heartbeat["delivered_reply"] == heartbeat["lived_experiences"] == ""
                assert rejected not in json.dumps(heart_requests, ensure_ascii=False)
                assert diary_note not in json.dumps(heart_requests, ensure_ascii=False)
                diary_entries = list((home / "diary").glob("free-think-*.md"))
                assert len(diary_entries) == 1
                assert diary_note in diary_entries[0].read_text(encoding="utf-8")
                stale = "这是一份旧记录，改动时间不能充当本次完成回执。"
                legacy = home / "sanae_free_think.md"
                legacy.write_text(f"# 旧的自由放空\n\n{stale}\n", encoding="utf-8")
                original_legacy = legacy.read_bytes()
                through = datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(seconds=1)
                assert prior < legacy.stat().st_mtime < through.timestamp()
                tokens = set_session_vars(platform="telegram", chat_type="dm", chat_id="fixture-chat",
                                          user_id="reflection-owner", session_id="owner-check",
                                          session_key="agent:main:telegram:dm:fixture-chat")
                try:
                    for collect in (collect_life_experiences, collect_recent_activities):
                        facts = collect(home=home, data_dir=target[0], owner_key=owner, session_id="owner-check",
                                        since=prior, through=through.timestamp())
                        assert diary_note not in facts
                        assert rejected not in facts
                        assert stale not in facts
                finally:
                    clear_session_vars(tokens)

                # Use the real nightly entry point on the same profile; force only the clock gate.
                nightly = load_memory_provider("sanae_mem")
                tokens = set_session_vars(platform="cron", session_id="nightly-replay")
                try:
                    nightly.initialize("nightly-replay", agent_context="cron", platform="cron",
                                       hermes_home=home, hook_worker=True)
                    result = nightly.run_nightly_journal(now=through, force=True)
                    assert result["ok"], result
                finally:
                    nightly.shutdown()
                    clear_session_vars(tokens)
                assert len(journal_requests) == 1
                _, source_json = journal_requests[0]["messages"][-1]["content"].split("当天片段：", 1)
                assert not any(item["kind"] == "自由放空" for item in json.loads(source_json))
                assert diary_note not in json.dumps(journal_requests[0], ensure_ascii=False)
                assert stale not in json.dumps(journal_requests[0], ensure_ascii=False)
                assert legacy.read_bytes() == original_legacy
                assert journal in (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
            finally:
                provider.shutdown()
        finally:
            model.shutdown()
            thread.join()
            reset_hermes_home_override(token)


def test_discovered_plugin_and_memory_provider_share_schedule_registry_across_shadow_hot_toggle(tmp_path, monkeypatch):
    from typing import cast
    from hermes_cli.plugins import PluginManager
    from hermes_host_contracts import TurnEnvelope, turn_envelope_scope
    from plugins.memory import load_memory_provider
    from sanae_mem.governance import runtime_registry
    from sanae_mem.runtime import scheduled as public_scheduled

    home = tmp_path / "hermes"
    shutil.copytree(Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem", home / "plugins" / "sanae_mem",
                    ignore=shutil.ignore_patterns("tests", "__pycache__"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(str(home))
    manager = PluginManager(scope_key=str(home))
    provider: SanaeMemProvider | None = None

    def write_config(shadow: bool) -> None:
        (home / "config.yaml").write_text(yaml.safe_dump({
            "memory": {"provider": "sanae_mem"},
            "plugins": {"enabled": ["sanae_mem"], "sanae_mem": {
                "semantic_backend": "sqlite",
                "nightly_journal": True,
                "free_think": True,
                "shadow": shadow,
                "memory_identity": {"platform": "telegram", "user_id": "4242"},
            }},
        }), encoding="utf-8")

    def vivemind_jobs() -> list[dict]:
        return [job for job in jobs.list_jobs(include_disabled=True) if job["name"].startswith("ViveMind:")]

    try:
        write_config(True)
        manager.discover_and_load()
        assert any(row["name"] == "sanae_mem" and row["enabled"] for row in manager.list_plugins())
        assert vivemind_jobs() == []

        discovered_scheduled = importlib.import_module("hermes_plugins.sanae_mem.runtime.scheduled")
        assert discovered_scheduled.register_schedule is public_scheduled.register_schedule
        assert discovered_scheduled.register_schedule is runtime_registry.register_schedule
        assert discovered_scheduled.reconcile_registered_schedule is public_scheduled.reconcile_registered_schedule
        assert discovered_scheduled.reconcile_registered_schedule is runtime_registry.reconcile_registered_schedule

        loaded_provider = load_memory_provider("sanae_mem", register_skills=False)
        assert loaded_provider is not None
        assert type(loaded_provider).__module__.startswith("_hermes_user_memory.")
        provider = cast(SanaeMemProvider, loaded_provider)
        provider.initialize("hot-resume", platform="telegram", hermes_home=home, agent_context="primary")

        envelope = TurnEnvelope(
            platform="telegram", chat_type="dm", chat_id="4242", sender_id="4242",
            session_id="hot-resume", message_id="m-1", session_key="agent:main:telegram:dm:4242",
        )
        write_config(False)
        with turn_envelope_scope(envelope):
            provider.handle_pre_llm_call(session_id="hot-resume", turn_id="t-1", platform="telegram", user_message="")

        restored = vivemind_jobs()
        assert {job["script"] for job in restored} == {None, "vivemind-nightly.py"}
        assert all(job["enabled"] for job in restored)

        write_config(True)
        with turn_envelope_scope(envelope):
            provider.handle_pre_llm_call(session_id="hot-resume", turn_id="t-2", platform="telegram", user_message="")

        paused = vivemind_jobs()
        assert {job["script"] for job in paused} == {None, "vivemind-nightly.py"}
        assert not any(job["enabled"] for job in paused)

        write_config(False)
        with turn_envelope_scope(envelope):
            provider.handle_pre_llm_call(session_id="hot-resume", turn_id="t-3", platform="telegram", user_message="")

        resumed = vivemind_jobs()
        assert {job["script"] for job in resumed} == {None, "vivemind-nightly.py"}
        assert all(job["enabled"] for job in resumed)
    finally:
        if provider is not None:
            provider.shutdown()
        manager.unload()
        reset_hermes_home_override(token)
