"""Regression tests for B2-09 audit closeout fixes (P1-1, P1-4, P2-1, P2-2, P2-3)."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import yaml

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem import SanaeMemProvider  # noqa: E402
from sanae_mem.governance.association_state import (  # noqa: E402
    AssociationState,
    HotAssociation,
)
from sanae_mem.governance.read_gate import HookRecall  # noqa: E402
from sanae_mem.infra import host_bridge  # noqa: E402
from sanae_mem.infra.textutil import _owner_key  # noqa: E402

AssociationTurn = importlib.import_module("sanae_mem.cognitive.association").AssociationTurn
from sanae_mem.storage.background_tasks import BackgroundTasks  # noqa: E402
from sanae_mem.storage.database import open_database  # noqa: E402
from sanae_mem.storage.event_maintenance import EventMaintenance  # noqa: E402
from sanae_mem.storage.event_store import EventStore  # noqa: E402
from sanae_mem.storage.sqlite_facade import SqliteFacade  # noqa: E402

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _write_home_config(
    home: Path,
    *,
    data_dir: Path | None = None,
    persona_name: str = "早苗",
    partner_name: str = "阿泽",
    partner_role: str = "挚友",
    partner_pronoun: str = "他",
    read_gate: str = "explicit",
    inject_max_chars: int = 400,
    daily_scheduled_asks: int = 2,
    emotion_ttl_days: int = 90,
    distill_daily_cap: int = 2,
    platform: str = "telegram",
    user_id: str = "owner-1",
    nightly_journal: bool = False,
    free_think: bool = False,
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "memories").mkdir(parents=True, exist_ok=True)
    plugin_cfg: dict[str, Any] = {
        "semantic_backend": "sqlite",
        "persona_name": persona_name,
        "partner_name": partner_name,
        "partner_role": partner_role,
        "partner_pronoun": partner_pronoun,
        "read_gate": read_gate,
        "inject_max_chars": inject_max_chars,
        "daily_scheduled_asks": daily_scheduled_asks,
        "emotion_ttl_days": emotion_ttl_days,
        "distill_daily_cap": distill_daily_cap,
        "memory_identity": {
            "platform": platform,
            "user_id": user_id,
        },
        "nightly_journal": nightly_journal,
        "free_think": free_think,
    }
    if data_dir is not None:
        plugin_cfg["data_dir"] = str(data_dir)
    root_cfg = {
        "memory": {"provider": "sanae_mem"},
        "plugins": {
            "enabled": ["sanae_mem"],
            "sanae_mem": plugin_cfg,
        },
    }
    cfg_path = home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(root_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return cfg_path


def _private_scope(
    *,
    platform: str = "telegram",
    user_id: str = "owner-1",
    session_id: str = "sess-1",
    turn_id: str = "turn-1",
    message_id: str = "msg-1",
):
    return host_bridge.turn_envelope_scope(
        platform=platform,
        chat_type="direct",
        chat_id=f"dm-{user_id}",
        sender_id=user_id,
        session_id=session_id,
        session_key=f"agent:main:{platform}:dm:{user_id}",
        turn_id=turn_id,
        message_id=message_id,
    )


def _setup_p1_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: str = "p1-session",
) -> tuple[SanaeMemProvider, str]:
    home = tmp_path / "home"
    data_dir = tmp_path / "data"
    _write_home_config(home, data_dir=data_dir, platform="telegram", user_id="owner-1")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    provider = SanaeMemProvider()
    with _private_scope(session_id=session_id):
        provider.initialize(session_id, hermes_home=str(home), platform="telegram")
        seeded = json.loads(
            provider.handle_tool_call("sanae_mem_add", {"content": "平时偶尔散步看云"})
        )
    assert seeded["ok"] is True
    event_id = int(seeded["result"]["id"])
    row = provider.sqlite.fetchone("SELECT owner_key FROM events WHERE id=?", (event_id,))
    assert row is not None
    owner = str(row["owner_key"])
    with provider.sqlite.transaction():
        provider.sqlite.execute("DELETE FROM events WHERE id=?", (event_id,))
    return provider, owner


def test_p1_1_check_prefetch_hook_recall_releases_lock_for_hot_worker_and_switch_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, owner = _setup_p1_provider(tmp_path, monkeypatch, session_id="p1-sess")
    state = provider.association_state
    query = "上次聊过的项目进展怎么样"
    session_id = "p1-sess"
    try:
        entered = threading.Event()
        release = threading.Event()
        gate_active = threading.Event()

        def resolver() -> str:
            if gate_active.is_set():
                entered.set()
                assert release.wait(timeout=5.0)
            return owner

        provider.association_state.bind_read_owner(owner, resolver=resolver)
        with _private_scope(session_id=session_id):
            hook_out = provider.handle_pre_llm_call(
                user_message=query,
                session_id=session_id,
                platform="telegram",
                sender_id="owner-1",
            )
        assert isinstance(hook_out, dict)
        assert state.is_hook_delivered() is True

        # Hold hot_lock so thread 2 exercises finish_hot_worker() + Condition.notify_all().
        hot_call = HotAssociation()
        hook_req = state.begin_hook_recall(session_id, session_id, query)
        rev_before = hook_req.revision
        assert state.try_start_hot(
            hot_call,
            revision=rev_before,
            hook_request=None,
            deadline=10**12,
            handoff_deadline=None,
        )
        state.clear_hook_request()

        gate_active.set()
        thread2_results: dict[str, Any] = {}

        def thread2_actor() -> None:
            assert entered.wait(timeout=5.0)
            # If resolver() held state.lock, all three calls below would deadlock.
            thread2_results["current_before"] = state.request_current(rev_before, None)
            state.finish_hot_worker()
            state.switch_session()
            thread2_results["current_after"] = state.request_current(rev_before, None)
            release.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_check = pool.submit(
                state.consume_prefetch_hook_recall,
                query,
                session_id,
                False,
                validate_read=lambda: state.validate_read(owner),
            )
            fut_actor = pool.submit(thread2_actor)
            fut_actor.result(timeout=5.0)
            recall = fut_check.result(timeout=5.0)

        assert thread2_results["current_before"] is True
        assert thread2_results["current_after"] is False
        assert recall is HookRecall.REVOKED
        cache_after, _ = state.current_recall_snapshot()
        assert cache_after is None
    finally:
        provider.shutdown()


def test_p1_1_check_prefetch_hook_recall_preserves_newer_cache_on_invalid_old_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, owner = _setup_p1_provider(tmp_path, monkeypatch, session_id="p1-aba")
    state = provider.association_state
    query = "上次聊过的项目进展怎么样"
    session_id = "p1-aba"
    try:
        entered = threading.Event()
        release = threading.Event()
        gate_active = threading.Event()

        def failing_after_pause() -> str:
            if gate_active.is_set():
                entered.set()
                assert release.wait(timeout=5.0)
                return "someone-else"
            return owner

        def run_sync_assoc() -> dict:
            with state.lock:
                rev = state.revision
            return provider.association.associate_context(
                query,
                session_id=session_id,
                hot_path=False,
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

        provider.association_state.bind_read_owner(owner, resolver=failing_after_pause)
        with _private_scope(session_id=session_id):
            run_sync_assoc()
        old_cache, _ = state.current_recall_snapshot()
        assert old_cache is not None

        gate_active.set()

        def replace_with_fresh_cache() -> None:
            assert entered.wait(timeout=5.0)
            provider.association_state.bind_read_owner(owner, resolver=None)
            with _private_scope(session_id=session_id):
                run_sync_assoc()
            release.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_check = pool.submit(
                state.consume_prefetch_hook_recall,
                query,
                session_id,
                False,
                validate_read=lambda: state.validate_read(owner),
            )
            fut_rebind = pool.submit(replace_with_fresh_cache)
            fut_rebind.result(timeout=5.0)
            recall = fut_check.result(timeout=5.0)

        assert recall is HookRecall.REVOKED
        new_cache, _ = state.current_recall_snapshot()
        assert new_cache is not None
        assert new_cache is not old_cache
    finally:
        provider.shutdown()


def test_p1_1_prefetch_public_api_completes_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, owner = _setup_p1_provider(tmp_path, monkeypatch, session_id="p1-e2e")
    state = provider.association_state
    query = "上次聊过的项目进展怎么样"
    session_id = "p1-e2e"
    try:
        entered = threading.Event()
        release = threading.Event()
        gate_active = threading.Event()

        def resolver() -> str:
            if gate_active.is_set():
                entered.set()
                assert release.wait(timeout=5.0)
            return owner

        provider.association_state.bind_read_owner(owner, resolver=resolver)
        with _private_scope(session_id=session_id):
            with state.lock:
                rev_init = state.revision
            provider.association.associate_context(
                query,
                session_id=session_id,
                hot_path=False,
                turn=AssociationTurn(
                    owner_key=owner,
                    daily_owner_key="",
                    daily_profile_key="",
                    daily_owner_platform="telegram",
                    agent_context="primary",
                    inject_max_chars=400,
                    llm=provider.runtime_llm(),
                ),
                revision=rev_init,
            )
        hot_call = HotAssociation()
        with state.lock:
            rev = state.revision
        assert state.try_start_hot(
            hot_call, revision=rev, hook_request=None, deadline=10**12, handoff_deadline=None,
        )

        gate_active.set()

        def actor() -> None:
            assert entered.wait(timeout=5.0)
            assert state.request_current(rev, None) is True
            state.finish_hot_worker()
            state.switch_session()
            release.set()

        def run_prefetch() -> str:
            with _private_scope(session_id=session_id):
                return provider.prefetch(query, session_id=session_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_prefetch = pool.submit(run_prefetch)
            fut_actor = pool.submit(actor)
            fut_actor.result(timeout=5.0)
            result = fut_prefetch.result(timeout=5.0)

        assert result == ""
    finally:
        provider.shutdown()


def test_issue032_3_association_state_machine_and_lock_free_interleavings() -> None:
    # 1. Hook returns first: delivered once; first prefetch cancels hook request, second sees DELIVERED.
    state = AssociationState()
    key = ("静音键盘", "sess-1", "chat", False, 400, "")
    req = state.begin_hook_recall("sess-1", "sess-1", "静音键盘")
    result = {"status": "ok", "context": "## 上次聊到的\n- 静音键盘", "items": [{"id": 1}]}
    assert state.store_recall(key=key, result=result, revision=req.revision, hook_request=req) is True
    assert state.mark_hook_delivered(req, result, binding_session_id="sess-1") is True
    assert state.current_recall() == result
    validate_calls = 0

    def counting_valid() -> bool:
        nonlocal validate_calls
        validate_calls += 1
        return True

    assert state.consume_prefetch_hook_recall("静音键盘", "sess-1", False, validate_read=counting_valid) is HookRecall.CANCELLED
    assert validate_calls == 0
    assert state.consume_prefetch_hook_recall("静音键盘", "sess-1", False, validate_read=counting_valid) is HookRecall.DELIVERED
    assert validate_calls == 1

    # 2. Prefetch cancels in-flight hook: late store_recall and mark_hook_delivered both fail.
    state2 = AssociationState()
    req2 = state2.begin_hook_recall("sess-2", "sess-2", "海边鹦鹉")
    assert state2.consume_prefetch_hook_recall("海边鹦鹉", "sess-2", False, validate_read=lambda: True) is HookRecall.CANCELLED
    late_res = {"status": "ok", "context": "late", "items": [{"id": 2}]}
    assert state2.store_recall(
        key=("海边鹦鹉", "sess-2", "chat", False, 400, ""),
        result=late_res,
        revision=req2.revision,
        hook_request=req2,
    ) is False
    assert state2.mark_hook_delivered(req2, late_res, binding_session_id="sess-2") is False
    assert state2.current_recall() is None

    # 3. Session switch or owner revocation while in flight invalidates stale outcome.
    state3 = AssociationState()
    req3 = state3.begin_hook_recall("sess-3", "sess-3", "咖啡")
    state3.switch_session()
    assert state3.store_recall(
        key=("咖啡", "sess-3", "chat", False, 400, ""),
        result=late_res,
        revision=req3.revision,
        hook_request=req3,
    ) is False
    assert state3.mark_hook_delivered(req3, late_res, binding_session_id="sess-3") is False
    assert state3.current_recall() is None

    # 4. Lock-free validate_read allows concurrent bind_read_owner + finish_hot_worker without deadlock.
    state4 = AssociationState()
    entered = threading.Event()
    release = threading.Event()
    state4.bind_read_owner("owner-1", lambda: (entered.set(), release.wait(timeout=5.0), "owner-1")[2])
    hot_call = HotAssociation()
    rev4 = state4.revision
    assert state4.try_start_hot(hot_call, revision=rev4, hook_request=None, deadline=10**12, handoff_deadline=None)
    assert state4.store_recall(
        key=("画展", "sess-4", "chat", False, 400, ""),
        result=result,
        revision=rev4,
        hook_request=None,
    ) is True

    def concurrent_rebind() -> None:
        assert entered.wait(timeout=5.0)
        state4.bind_read_owner("owner-2", None)
        state4.finish_hot_worker()
        release.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_check = pool.submit(
            state4.consume_prefetch_hook_recall,
            "画展",
            "sess-4",
            False,
            validate_read=lambda: state4.validate_read("owner-1"),
        )
        fut_rebind = pool.submit(concurrent_rebind)
        fut_rebind.result(timeout=5.0)
        assert fut_check.result(timeout=5.0) is HookRecall.REVOKED

    # 5. Session switch after store_recall and before mark_hook_delivered aborts delivery atomically.
    state5 = AssociationState()
    req5 = state5.begin_hook_recall("sess-5", "sess-5", "京都红叶")
    res5 = {"status": "ok", "context": "## 上次聊到的\n- 京都红叶", "items": [{"id": 5}]}
    assert state5.store_recall(
        key=("京都红叶", "sess-5", "chat", False, 400, ""),
        result=res5,
        revision=req5.revision,
        hook_request=req5,
    ) is True
    state5.switch_session()
    delivered_5: list[str] = []
    assert state5.mark_hook_delivered(
        req5,
        res5,
        binding_session_id="sess-5",
        on_deliver=lambda: delivered_5.append("delivered"),
    ) is False
    assert delivered_5 == []
    assert state5.is_hook_delivered() is False

    # 6. mark_hook_delivered holds state.lock across on_deliver so concurrent switch_session cannot interleave.
    state6 = AssociationState()
    req6 = state6.begin_hook_recall("sess-6", "sess-6", "银座咖啡")
    res6 = {"status": "ok", "context": "## 上次聊到的\n- 银座咖啡", "items": [{"id": 6}]}
    assert state6.store_recall(
        key=("银座咖啡", "sess-6", "chat", False, 400, ""),
        result=res6,
        revision=req6.revision,
        hook_request=req6,
    ) is True
    deliver_entered = threading.Event()
    deliver_release = threading.Event()
    timeline: list[str] = []

    def slow_deliver() -> None:
        deliver_entered.set()
        assert deliver_release.wait(timeout=5.0)
        timeline.append("deliver_done")

    def concurrent_switch() -> None:
        assert deliver_entered.wait(timeout=5.0)
        state6.switch_session()
        timeline.append("switch_done")

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_deliver = pool.submit(
            state6.mark_hook_delivered,
            req6,
            res6,
            binding_session_id="sess-6",
            on_deliver=slow_deliver,
        )
        fut_switch = pool.submit(concurrent_switch)
        assert deliver_entered.wait(timeout=5.0)
        time.sleep(0.05)
        assert timeline == []
        deliver_release.set()
        assert fut_deliver.result(timeout=5.0) is True
        fut_switch.result(timeout=5.0)
    assert timeline == ["deliver_done", "switch_done"]
    assert state6.current_recall() is None


def test_issue032_3_real_host_turn_delivers_hook_recall_once_on_wire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("hermes_constants") is None:
        pytest.skip("hermes-agent host not on PYTHONPATH")

    repo_root = Path(__file__).resolve().parents[2]
    MemoryManager = importlib.import_module("agent.memory_manager").MemoryManager
    session_ctx_mod = importlib.import_module("gateway.session_context")
    clear_session_vars = session_ctx_mod.clear_session_vars
    set_session_vars = session_ctx_mod.set_session_vars
    get_plugin_manager = importlib.import_module("hermes_cli.plugins").get_plugin_manager
    constants_mod = importlib.import_module("hermes_constants")
    reset_hermes_home_override = constants_mod.reset_hermes_home_override
    set_hermes_home_override = constants_mod.set_hermes_home_override
    contracts_mod = importlib.import_module("hermes_host_contracts")
    TurnEnvelope = contracts_mod.TurnEnvelope
    turn_envelope_scope = contracts_mod.turn_envelope_scope
    load_memory_provider = importlib.import_module("plugins.memory").load_memory_provider
    AIAgent = importlib.import_module("run_agent").AIAgent

    wire_requests: list[list[dict[str, Any]]] = []

    class _LoopbackModel(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length))
            wire_requests.append(list(req.get("messages") or []))
            reply = {"role": "assistant", "content": "好呀，我一直记得京都红叶那事。"}
            if req.get("stream"):
                event = {"choices": [{"index": 0, "delta": reply, "finish_reason": "stop"}]}
                body = f"data: {json.dumps(event, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode("utf-8")
                content_type = "text/event-stream"
            else:
                body = json.dumps(
                    {
                        "id": "wire-test",
                        "model": req.get("model", "vm-wire-test"),
                        "choices": [{"index": 0, "message": reply, "finish_reason": "stop"}],
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    home = tmp_path / "hermes"
    shutil.copytree(
        repo_root / "plugins" / "sanae_mem",
        home / "plugins" / "sanae_mem",
        ignore=shutil.ignore_patterns("tests", "__pycache__"),
    )
    plugin_cfg: dict[str, Any] = {
        "semantic_backend": "sqlite",
        "read_gate": "normal",
        "gap_continue": True,
        "gap_threshold_hours": 0,
        "nightly_journal": False,
        "free_think": False,
        "memory_identity": {"platform": "telegram", "user_id": "wire-owner"},
    }
    marker = "我想买一把静音键盘"

    with ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackModel) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "memory": {"provider": "sanae_mem", "memory_enabled": False, "user_profile_enabled": False},
                    "model": {
                        "default": "vm-wire-test",
                        "provider": "custom",
                        "base_url": base_url,
                        "context_length": 64000,
                    },
                    "plugins": {"enabled": ["sanae_mem"], "sanae_mem": plugin_cfg},
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        token = set_hermes_home_override(str(home))
        manager = get_plugin_manager()
        memory = MemoryManager()
        agent: Any = None
        seed = SanaeMemProvider(plugin_cfg)
        seed_open = True
        try:
            with turn_envelope_scope(
                TurnEnvelope(
                    platform="telegram",
                    chat_type="dm",
                    chat_id="wire-owner",
                    sender_id="wire-owner",
                    session_id="wire-seed",
                    message_id="seed-1",
                    session_key="agent:main:telegram:dm:wire-owner",
                )
            ):
                seed.initialize("wire-seed", platform="telegram", hermes_home=home)
                seeded = json.loads(
                    seed.handle_tool_call(
                        "sanae_mem_add",
                        {"content": marker, "importance": 0.9},
                    )
                )
                assert seeded["ok"] is True
                seeded_id = int(seeded["result"]["id"])
            seed.shutdown()
            seed_open = False

            manager.discover_and_load()
            provider = load_memory_provider("sanae_mem", register_skills=False)
            assert provider is not None
            memory.add_provider(provider)
            memory.initialize_all(
                session_id="wire-sess",
                platform="telegram",
                hermes_home=home,
            )
            agent = AIAgent(
                model="vm-wire-test",
                provider="custom",
                base_url=base_url,
                api_key="loopback-fixture",
                platform="telegram",
                session_id="wire-sess",
                memory_manager=memory,
                enabled_toolsets=[],
                quiet_mode=True,
                skip_background_review=True,
            )
            session_tokens = set_session_vars(
                platform="telegram",
                session_id="wire-sess",
                chat_type="dm",
                chat_id="wire-owner",
                user_id="wire-owner",
                session_key="agent:main:telegram:dm:wire-owner",
            )
            try:
                with turn_envelope_scope(
                    TurnEnvelope(
                        platform="telegram",
                        chat_type="dm",
                        chat_id="wire-owner",
                        sender_id="wire-owner",
                        session_id="wire-sess",
                        turn_id="wire-turn-1",
                        message_id="wire-msg-1",
                        session_key="agent:main:telegram:dm:wire-owner",
                    )
                ):
                    out = agent.run_conversation(
                        "静音键盘的事",
                        conversation_history=[],
                    )
            finally:
                clear_session_vars(session_tokens)

            assert out["final_response"]
            assert len(wire_requests) >= 1
            wire_text = "\n".join(str(msg.get("content") or "") for msg in wire_requests[0])
            assert wire_text.count("买一把静音键盘") == 1

            with closing(sqlite3.connect(home / "sanae-mem" / "sanae.db")) as conn:
                row = conn.execute(
                    "SELECT asked, asked_session FROM events WHERE id = ?",
                    (seeded_id,),
                ).fetchone()
            assert row is not None
            assert int(row[0]) == 1
            assert row[1]
        finally:
            if seed_open:
                seed.shutdown()
            if agent is None:
                memory.shutdown_all()
            else:
                agent.close()
            manager.unload()
            reset_hermes_home_override(token)
            server.shutdown()
            thread.join()


def test_p2_1_event_maintenance_chroma_operational_error_warns_and_commits_sqlite(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    archive_dir = tmp_path / "archive"
    archive_file = archive_dir / "events-archived.jsonl"
    conn = open_database(data_dir / "sanae.db", archive_dir=archive_dir)
    lock = threading.RLock()
    current_conn: sqlite3.Connection | None = conn

    def get_conn() -> sqlite3.Connection | None:
        return current_conn

    def set_conn(new_conn: sqlite3.Connection | None) -> None:
        nonlocal current_conn
        current_conn = new_conn

    facade = SqliteFacade(lock=lock, conn_getter=get_conn, conn_setter=set_conn)
    facade.attach(conn, db_path=data_dir / "sanae.db")
    owner = "owner-p2-1"

    class _FailingChromaCollection:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc
            self.deleted_batches: list[list[str]] = []

        def delete(self, *, ids: list[str]) -> None:
            self.deleted_batches.append(list(ids))
            raise self.exc

    scopes = EventStore(
        facade,
        current_session_id=lambda: "sess-p2-1",
        current_agent_context=lambda: "primary",
        current_home=lambda: str(tmp_path),
        current_owner=lambda: owner,
        current_read_owners=lambda: (owner,),
        private_chat_owner=lambda: owner,
        is_system_turn=lambda _sid: False,
        group_scene_active=lambda: False,
        group_read_owners=lambda: [],
        group_write_owner_key=lambda: "",
        group_maintenance_owners=lambda: [],
        group_resolution_owners=lambda: (),
    )
    failing_collection = _FailingChromaCollection(sqlite3.OperationalError("chroma sqlite busy"))
    maintenance = EventMaintenance(
        facade,
        scopes=scopes,
        archive_path=lambda: archive_file,
        current_data_dir=lambda: data_dir,
        has_primary_private_owner=lambda: True,
        upsert_chroma=lambda *_args: None,
        current_collection=lambda: failing_collection,
    )

    try:
        now = "2026-11-22T12:00:00+08:00"
        with facade.transaction() as tx:
            tx.execute(
                """INSERT INTO events(ts, raw, summary, topic, keywords, importance, confidence,
                                      owner_key, experience_subject, fact_subject, created_at, updated_at)
                   VALUES (?, ?, ?, '日常', '咖啡', 0.6, 0.7, ?, ?, ?, ?, ?)""",
                ("2026-11-22 10:00", "第一条要删掉的记忆", "第一条要删掉的记忆", owner, owner, owner, now, now),
            )
            purge_id = int(tx.execute("SELECT last_insert_rowid()").fetchone()[0])
            tx.execute(
                """INSERT INTO events(ts, raw, summary, topic, keywords, importance, confidence,
                                      owner_key, experience_subject, fact_subject, created_at, updated_at)
                   VALUES (?, ?, ?, '日常', '散步', 0.7, 0.7, ?, ?, ?, ?, ?)""",
                ("2026-11-22 10:05", "主事件：晚上去河边散步", "晚上去河边散步", owner, owner, owner, now, now),
            )
            survivor_id = int(tx.execute("SELECT last_insert_rowid()").fetchone()[0])
            tx.execute(
                """INSERT INTO events(ts, raw, summary, topic, keywords, importance, confidence,
                                      owner_key, experience_subject, fact_subject, created_at, updated_at)
                   VALUES (?, ?, ?, '日常', '散步', 0.5, 0.6, ?, ?, ?, ?, ?)""",
                ("2026-11-22 10:10", "重复事件：河边散步吹风", "河边散步吹风", owner, owner, owner, now, now),
            )
            dup_id = int(tx.execute("SELECT last_insert_rowid()").fetchone()[0])

        with caplog.at_level(logging.WARNING, logger="sanae_mem.storage.event_maintenance"):
            purged = maintenance.purge_events([purge_id], tombstone=True)
            merged = maintenance.merge_duplicate_events(survivor_id, [dup_id], 0.85, 0.9, now)

        assert purged == 1
        assert merged == 1
        assert failing_collection.deleted_batches == [[f"evt-{purge_id}"], [f"evt-{dup_id}"]]
        assert "sanae_mem: Chroma delete failed (OperationalError)" in caplog.text
        assert "sanae_mem: Chroma delete after merge failed (OperationalError)" in caplog.text

        with closing(sqlite3.connect(data_dir / "sanae.db")) as check_db:
            assert check_db.execute("SELECT id FROM events WHERE id=?", (purge_id,)).fetchone() is None
            assert check_db.execute("SELECT id FROM events WHERE id=?", (dup_id,)).fetchone() is None
            survivor_row = check_db.execute(
                "SELECT raw, importance, confidence FROM events WHERE id=?",
                (survivor_id,),
            ).fetchone()
            assert survivor_row is not None
            assert "主事件：晚上去河边散步" in survivor_row[0]
            assert "重复事件：河边散步吹风" in survivor_row[0]
            assert survivor_row[1] == pytest.approx(0.85)
            assert survivor_row[2] == pytest.approx(0.9)
            assert check_db.execute("SELECT COUNT(*) FROM forgotten").fetchone()[0] == 1
            assert (
                check_db.execute(
                    "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                    (survivor_id,),
                ).fetchone()[0]
                >= 1
            )
        assert archive_file.is_file()
        archived_ids = [
            json.loads(line)["id"]
            for line in archive_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert purge_id in archived_ids and dup_id in archived_ids

        # Non-backend errors must propagate rather than being swallowed.
        failing_collection.exc = ValueError("unexpected programming bug")
        with facade.transaction() as tx:
            tx.execute(
                """INSERT INTO events(ts, raw, summary, topic, keywords, importance, confidence,
                                      owner_key, experience_subject, fact_subject, created_at, updated_at)
                   VALUES (?, ?, ?, '日常', '测试', 0.6, 0.7, ?, ?, ?, ?, ?)""",
                ("2026-11-22 10:20", "非法异常不该被吞", "非法异常不该被吞", owner, owner, owner, now, now),
            )
            bad_id = int(tx.execute("SELECT last_insert_rowid()").fetchone()[0])
        with pytest.raises(ValueError, match="unexpected programming bug"):
            maintenance.purge_events([bad_id], tombstone=True)
    finally:
        facade.close()


def test_p2_2_background_tasks_pause_atomicity_and_unseal() -> None:
    tasks = BackgroundTasks()
    assert tasks.try_register() is True

    entered_pause = threading.Event()

    def pause_worker() -> None:
        with tasks.pause():
            entered_pause.set()

    thread = threading.Thread(target=pause_worker)
    thread.start()
    # pause() must wait for active == 0 before yielding.
    assert not entered_pause.wait(timeout=0.05)
    tasks.complete()
    assert entered_pause.wait(timeout=2.0)
    thread.join(timeout=2.0)
    # Restored to unsealed.
    assert tasks.try_register() is True
    tasks.complete()

    tasks.seal_and_wait()
    assert tasks.try_register() is False

    # Plain pause() keeps sealed=True when already sealed.
    with tasks.pause():
        assert tasks.try_register() is False
    assert tasks.try_register() is False

    # Failed pause(unseal_on_exit=True) keeps sealed=True.
    with pytest.raises(RuntimeError, match="boom"):
        with tasks.pause(unseal_on_exit=True):
            raise RuntimeError("boom")
    assert tasks.try_register() is False

    # Successful pause(unseal_on_exit=True) unseals.
    with tasks.pause(unseal_on_exit=True):
        assert tasks.try_register() is False
    assert tasks.try_register() is True
    tasks.complete()

    tasks.seal_and_wait()
    assert tasks.try_register() is False
    tasks.unseal()
    assert tasks.try_register() is True
    tasks.complete()


def test_p2_2_and_p2_3_provider_shutdown_then_initialize_unseals_and_failure_keeps_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data_dir = tmp_path / "data"
    _write_home_config(
        home,
        data_dir=data_dir,
        platform="telegram",
        user_id="owner-1",
        nightly_journal=True,
    )
    bad_home = tmp_path / "bad-home"
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("occupied", encoding="utf-8")
    _write_home_config(bad_home, data_dir=blocking_file, platform="telegram", user_id="owner-1")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))

    provider = SanaeMemProvider()
    now = datetime(2026, 11, 22, 23, 30, tzinfo=_SHANGHAI)
    try:
        with _private_scope(session_id="s-init-1"):
            provider.initialize("s-init-1", hermes_home=str(home), platform="telegram")
        assert isinstance(provider.background_tasks, BackgroundTasks)
        assert provider.background_tasks.try_register() is True
        provider.background_tasks.complete()

        provider.shutdown()
        assert provider.background_tasks.try_register() is False
        assert provider.run_nightly_journal(now=now) == {"ok": False, "reason": "shutdown"}
        assert provider.submit_free_think_heartbeat(data_dir, "default", lambda: None) is False

        # A failing initialize() must keep BackgroundTasks sealed.
        with pytest.raises(OSError):
            provider.initialize("s-fail", hermes_home=str(bad_home), platform="telegram")
        assert not provider.sqlite.is_open
        assert provider.background_tasks.try_register() is False
        assert provider.run_nightly_journal(now=now) == {"ok": False, "reason": "shutdown"}

        # A subsequent successful initialize() opens SQLite and unseals BackgroundTasks.
        with _private_scope(session_id="s-init-2"):
            provider.initialize("s-init-2", hermes_home=str(home), platform="telegram")
        assert provider.sqlite.is_open
        assert provider.background_tasks.try_register() is True
        provider.background_tasks.complete()

        ran = threading.Event()
        assert provider.submit_free_think_heartbeat(data_dir, "default", ran.set) is True
        assert ran.wait(timeout=5.0)

        with host_bridge.turn_envelope_scope(platform="cron", session_id="s-init-2"):
            journal_res = provider.run_nightly_journal(now=now)
        assert journal_res.get("reason") != "shutdown"
        assert journal_res.get("ok") is True
    finally:
        provider.shutdown()


def test_p2_3_delivery_uses_public_background_tasks_property() -> None:
    delivery_path = PLUGIN_ROOT / "sanae_mem" / "runtime" / "delivery.py"
    tree = ast.parse(delivery_path.read_text(encoding="utf-8"), filename=str(delivery_path))
    private_refs = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_background_tasks"
    ]
    assert private_refs == []


def test_p1_4_same_instance_initialize_home_switch_refreshes_config_and_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_a = tmp_path / "home-a"
    data_a = tmp_path / "data-a"
    home_b = tmp_path / "home-b"
    data_b = tmp_path / "data-b"

    _write_home_config(
        home_a,
        data_dir=data_a,
        persona_name="PersonaA",
        partner_name="PartnerA",
        partner_role="朋友",
        partner_pronoun="他",
        read_gate="explicit",
        inject_max_chars=120,
        daily_scheduled_asks=1,
        emotion_ttl_days=30,
        distill_daily_cap=1,
        platform="telegram",
        user_id="owner-a",
    )
    _write_home_config(
        home_b,
        data_dir=data_b,
        persona_name="PersonaB",
        partner_name="PartnerB",
        partner_role="搭档",
        partner_pronoun="她",
        read_gate="scheduled",
        inject_max_chars=260,
        daily_scheduled_asks=3,
        emotion_ttl_days=60,
        distill_daily_cap=4,
        platform="discord",
        user_id="owner-b",
    )

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home_a))

    captured_prompts: list[tuple[str, list[dict[str, str]]]] = []

    class _RecordingLlm:
        def complete(
            self,
            messages: list[dict[str, str]],
            *,
            task: str = "",
            temperature: float = 0.0,
            max_tokens: int = 320,
            timeout: float = 30.0,
            purpose: str | None = None,
        ) -> Any:
            captured_prompts.append((str(purpose or task), messages))
            if purpose == "daily_context_semantic":
                return SimpleNamespace(text=json.dumps({"items": []}, ensure_ascii=False))
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "route": "inner_state",
                        "inner_thought": "心里记下了。",
                        "event_fact": None,
                        "reason": "temp",
                    },
                    ensure_ascii=False,
                )
            )

    provider = SanaeMemProvider()
    provider.attach_runtime_llm(_RecordingLlm())

    try:
        with _private_scope(platform="telegram", user_id="owner-a", session_id="sess-a", turn_id="t-a"):
            provider.initialize("sess-a", hermes_home=str(home_a), platform="telegram")
            assert provider._config["persona_name"] == "PersonaA"
            assert Path(provider.backup_paths()[0]).parent.resolve() == data_a.resolve()
            assert provider.ingest.curve_base_days("今天好开心") == 30.0

            ctx_a = provider.current_turn_context(
                session_id="sess-a",
                turn_id="t-a",
                sender_id="owner-a",
            )
            assert "PartnerA，我的朋友" in ctx_a

            captured_prompts.clear()
            provider.route_resident_write(
                "memory",
                {"action": "add", "content": "刚才有点小开心"},
                session_id="sess-a",
                turn_id="t-a",
            )
            assert any("PersonaA" in msg["content"] for _, msgs in captured_prompts for msg in msgs)

            # Explicit mode must not perform scheduled prefetch.
            owner_a = _owner_key("telegram", user_id="owner-a")
            provider.sqlite.execute(
                """INSERT INTO events(ts, raw, summary, topic, keywords, importance, confidence,
                                      owner_key, experience_subject, fact_subject, created_at, updated_at)
                   VALUES ('2026-11-22 09:00', 'A库里的咖啡约定', 'A库里的咖啡约定', '日常', '咖啡',
                           0.9, 0.9, ?, ?, ?, '2026-11-22T09:00:00+08:00', '2026-11-22T09:00:00+08:00')""",
                (owner_a, owner_a, owner_a),
            )
            assert provider.prefetch("普通问候", session_id="sess-a", scheduled=True) == ""

        # Switch the SAME instance to home_b.
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home_b))
        with _private_scope(platform="discord", user_id="owner-b", session_id="sess-b", turn_id="t-b"):
            provider.initialize("sess-b", hermes_home=str(home_b), platform="discord")
            assert provider._config["persona_name"] == "PersonaB"
            assert provider._config["partner_name"] == "PartnerB"
            assert provider._config["read_gate"] == "scheduled"
            assert Path(provider.backup_paths()[0]).parent.resolve() == data_b.resolve()
            assert (data_b / "sanae.db").is_file()
            assert provider.ingest.curve_base_days("今天好开心") == 60.0

            ctx_b = provider.current_turn_context(
                session_id="sess-b",
                turn_id="t-b",
                sender_id="owner-b",
            )
            assert "PartnerB，我的搭档" in ctx_b
            assert "PartnerA" not in ctx_b

            captured_prompts.clear()
            provider.route_resident_write(
                "memory",
                {"action": "add", "content": "刚才有点小期待"},
                session_id="sess-b",
                turn_id="t-b",
            )
            guard_msgs = [msg["content"] for _, msgs in captured_prompts for msg in msgs]
            assert any("PersonaB" in text for text in guard_msgs)
            assert not any("PersonaA" in text for text in guard_msgs)

            captured_prompts.clear()
            provider.daily_context.record_completed_turn(
                "sess-b",
                "好的，记着了。",
                [{"role": "user", "content": "我今天下午要去美术馆看展"}],
                sender_id="owner-b",
                platform="discord",
                user_message="我今天下午要去美术馆看展",
                turn_id="t-daily-b",
            )
            daily_msgs = [msg["content"] for _, msgs in captured_prompts for msg in msgs]
            assert any("PersonaB" in text for text in daily_msgs)
            assert not any("PersonaA" in text for text in daily_msgs)

            # ReadGate refreshed to scheduled mode with home_b data.
            owner_b = _owner_key("discord", user_id="owner-b")
            provider.sqlite.execute(
                """INSERT INTO events(ts, raw, summary, topic, keywords, importance, confidence,
                                      owner_key, experience_subject, fact_subject, created_at, updated_at)
                   VALUES ('2026-11-22 09:00', 'B库里关于画展的约定', 'B库里关于画展的约定', '日常', '画展',
                           0.9, 0.9, ?, ?, ?, '2026-11-22T09:00:00+08:00', '2026-11-22T09:00:00+08:00')""",
                (owner_b, owner_b, owner_b),
            )
            prefetched = provider.prefetch("下午去画展看看", session_id="sess-b", scheduled=True)
            assert "B库里关于画展的约定" in prefetched
            assert provider._read_gate._options.partner_name == "PartnerB"
            assert provider._read_gate._options.inject_max_chars == 260
    finally:
        provider.shutdown()


def test_p1_4_explicit_config_preserved_across_home_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    explicit_data = tmp_path / "explicit-data"

    _write_home_config(
        home_a,
        data_dir=tmp_path / "data-a",
        persona_name="HomePersonaA",
        partner_name="HomePartnerA",
        platform="telegram",
        user_id="owner-a",
    )
    _write_home_config(
        home_b,
        data_dir=tmp_path / "data-b",
        persona_name="HomePersonaB",
        partner_name="HomePartnerB",
        platform="telegram",
        user_id="owner-a",
    )
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home_a))

    explicit_cfg = {
        "data_dir": str(explicit_data),
        "semantic_backend": "sqlite",
        "persona_name": "ExplicitPersona",
        "partner_name": "ExplicitPartner",
        "partner_role": "老友",
        "read_gate": "explicit",
        "emotion_ttl_days": 45,
        "memory_identity": {"platform": "telegram", "user_id": "owner-a"},
    }
    provider = SanaeMemProvider(explicit_cfg)
    try:
        with _private_scope(platform="telegram", user_id="owner-a", session_id="s-exp-1", turn_id="t1"):
            provider.initialize("s-exp-1", hermes_home=str(home_a), platform="telegram")
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home_b))
        with _private_scope(platform="telegram", user_id="owner-a", session_id="s-exp-2", turn_id="t2"):
            provider.initialize("s-exp-2", hermes_home=str(home_b), platform="telegram")
            assert provider._config["persona_name"] == "ExplicitPersona"
            assert provider._config["partner_name"] == "ExplicitPartner"
            assert Path(provider.backup_paths()[0]).parent.resolve() == explicit_data.resolve()
            assert provider.ingest.curve_base_days("今天好开心") == 45.0
            ctx = provider.current_turn_context(
                session_id="s-exp-2",
                turn_id="t2",
                sender_id="owner-a",
            )
            assert "ExplicitPartner，我的老友" in ctx
    finally:
        provider.shutdown()
