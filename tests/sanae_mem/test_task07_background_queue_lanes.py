"""Real-transport integration tests for Task 07: background shutdown barrier, lane isolation, and failure status."""

from __future__ import annotations

import contextlib
from contextlib import closing
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any, Callable, Iterator, cast

import pytest
import yaml

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _harness import LoopbackLlmServer, resolve_hermes_host_root  # noqa: E402

HERMES_ROOT = resolve_hermes_host_root()
PLUGINS_ROOT = Path(__file__).resolve().parents[2] / "plugins"

if HERMES_ROOT is not None and str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))
if str(PLUGINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGINS_ROOT))

pytestmark = pytest.mark.skipif(
    HERMES_ROOT is None,
    reason="hermes-agent host contracts source tree not found",
)

import hermes_constants as _hc  # noqa: E402  # pyright: ignore[reportMissingImports]
from cron import jobs as cron_jobs  # noqa: E402  # pyright: ignore[reportMissingImports]
from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager  # noqa: E402  # pyright: ignore[reportMissingImports]
from hermes_host_contracts import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    CronCompletionEvent,
    DeliveryReceipt,
    TurnEnvelope,
    clear_all_contract_registrations,
    emit_cron_completion,
    emit_delivery_receipt,
    turn_envelope_scope,
)
from sanae_mem.cognitive import inner_wiring  # noqa: E402
from sanae_mem import plugin_entry  # noqa: E402
from sanae_mem.cognitive.daily_context import daily_profile_key  # noqa: E402
from sanae_mem.plugin_entry import SanaeMemProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_host_contracts(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    clear_all_contract_registrations()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("OPENAI_API_KEY", "loopback-fixture")
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.delenv("HERMES_HOME", raising=False)
    yield
    inner_wiring.wait_for_pending_updates(timeout=5.0)
    clear_all_contract_registrations()


@contextlib.contextmanager
def _home_override(home: Path) -> Iterator[Path]:
    token = _hc.set_hermes_home_override(home)
    try:
        yield home
    finally:
        _hc.reset_hermes_home_override(token)


class _LoopbackLLMServer(LoopbackLlmServer):
    """Real loopback OpenAI-compatible HTTP server (only the outermost HTTP boundary is stubbed)."""

    def __init__(self, responder: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        super().__init__(lambda payload, _idx: responder(payload))


def _write_home_config(
    home: Path,
    *,
    base_url: str,
    owner_user_id: str = "10001",
    free_think: bool = False,
    backend: str = "sqlite",
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    data_dir = (home / "sanae-mem").resolve()
    (home / ".env").write_text("OPENAI_API_KEY=loopback-fixture\n", encoding="utf-8")
    root_cfg: dict[str, Any] = {
        "auxiliary": {
            "sanae_daily_flash": {
                "provider": "custom",
                "model": "vm-flash",
                "base_url": base_url,
                "api_mode": "chat_completions",
            },
            "sanae_extract": {
                "provider": "custom",
                "model": "vm-extract",
                "base_url": base_url,
                "api_mode": "chat_completions",
            },
        },
        "memory": {
            "provider": "sanae_mem",
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "plugins": {
            "enabled": ["sanae_mem"],
            "sanae_mem": {
                "data_dir": str(data_dir),
                "semantic_backend": backend,
                "free_think": free_think,
                "memory_identity": {
                    "platform": "qq",
                    "user_id": owner_user_id,
                },
                "group_context": {
                    "defaults": {
                        "short_term_shared": True,
                        "long_term_memory": True,
                        "allow_unmentioned_reply": True,
                    },
                },
            },
        },
    }
    (home / "config.yaml").write_text(
        yaml.safe_dump(root_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return data_dir


def _register_plugin_in_home(
    home: Path,
) -> tuple[SanaeMemProvider, list[Callable[[], None]]]:
    manifest = PluginManifest(name="sanae_mem", version="0.1.0", source="user")
    captured: list[SanaeMemProvider] = []
    unload_callbacks: list[Callable[[], None]] = []

    with _home_override(home):
        manager = get_plugin_manager()
        ctx = PluginContext(manifest=manifest, manager=manager)
        orig_register_mem = ctx.register_memory_provider
        orig_on_unload = ctx.on_unload

        def _capture_memory_provider(provider: Any) -> Any:
            captured.append(provider)
            return orig_register_mem(provider)

        def _capture_unload(callback: Callable[[], None]) -> Any:
            unload_callbacks.append(callback)
            return orig_on_unload(callback)

        ctx.register_memory_provider = _capture_memory_provider  # type: ignore[method-assign]
        ctx.on_unload = _capture_unload  # type: ignore[method-assign]
        plugin_entry.register(ctx)

    assert len(captured) == 1, "register(ctx) must register one SanaeMemProvider prototype"
    return captured[0], unload_callbacks


def _run_unloads(callbacks: list[Callable[[], None]]) -> None:
    for callback in reversed(callbacks):
        callback()


def _query_rows(data_dir: Path, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    db_path = data_dir / "sanae.db"
    if not db_path.is_file():
        return []
    with closing(sqlite3.connect(str(db_path))) as conn:
        return list(conn.execute(sql, params).fetchall())


def test_group_slow_flash_does_not_block_private_lane_or_hot_path(tmp_path: Path) -> None:
    """Blocking the group lane >1.0s must not delay private lane settlement or current_turn_context."""
    home = tmp_path / "lane-isolation"
    group_entered = threading.Event()
    release_group = threading.Event()
    group_quote = "GROUP_SLOW_FLASH_MARKER 周六下午一起去河边野餐吧"
    private_thought = "窗边起风了，今天聊得挺轻松。"

    def respond(payload: dict[str, Any]) -> dict[str, Any]:
        last_content = str((payload.get("messages") or [{}])[-1].get("content") or "")
        if "GROUP_SLOW_FLASH_MARKER" in last_content:
            group_entered.set()
            release_group.wait(15.0)
            return {
                "events": [{"text": "周六群里约去河边野餐", "quote": group_quote}],
                "impression": None,
                "new_commitments": [],
                "commitment_updates": [],
            }
        if '"delivered_reply"' in last_content:
            return {
                "sentences": [private_thought],
                "shared_sentences": None,
                "variables": {"tension": 0.1, "closeness": 0.4, "energy": 0.2},
            }
        return {"add": [], "update": [], "resolve": [], "new_commitments": [], "updates": []}

    with _LoopbackLLMServer(respond) as llm_server:
        data_dir = _write_home_config(home, base_url=llm_server.base_url)
        profile_key = daily_profile_key(home)
        prototype, unloads = _register_plugin_in_home(home)
        try:
            group_receipt = DeliveryReceipt(
                session_id="qq:group:990011",
                platform="qq",
                chat_type="group",
                chat_id="990011",
                scope_id="default",
                sender_id="u_friend",
                turn_id="turn-group-slow-1",
                message_id="out-group-1",
                incoming_message_id="in-group-1",
                session_key="agent:main:qq:group:990011",
                user_message=group_quote,
                delivered_text="好呀，周六我带水果过去。",
                clean_text="好呀，周六我带水果过去。",
                delivered_at=time.time(),
                transport_confirmed=True,
            )
            with _home_override(home):
                emit_delivery_receipt(group_receipt)

            assert group_entered.wait(15.0), "group background worker never reached Flash HTTP"
            block_started = time.monotonic()

            # Group lane is now genuinely blocked. Submit a private DM delivery on the same data_dir.
            private_receipt = DeliveryReceipt(
                session_id="qq:dm:10001",
                platform="qq",
                chat_type="dm",
                chat_id="10001",
                scope_id="default",
                sender_id="10001",
                turn_id="turn-private-fast-1",
                message_id="out-dm-1",
                incoming_message_id="in-dm-1",
                session_key="agent:main:qq:dm:10001",
                user_message="PRIVATE_FAST_MARKER 今天天气真好呀。",
                delivered_text="嗯，风吹过来挺舒服的。",
                clean_text="嗯，风吹过来挺舒服的。",
                delivered_at=time.time(),
                transport_confirmed=True,
                is_bot=False,
            )
            with _home_override(home):
                emit_delivery_receipt(private_receipt)

            # Waiting on the private lane must finish quickly while group remains blocked.
            t_wait = time.monotonic()
            assert inner_wiring.wait_for_pending_updates(
                (data_dir, profile_key, "private"), timeout=5.0
            )
            private_wait_elapsed = time.monotonic() - t_wait
            assert private_wait_elapsed < 1.0, f"private lane wait took {private_wait_elapsed:.3f}s"

            watermarks = _query_rows(
                data_dir,
                "SELECT turn_id FROM session_reply_watermarks WHERE session_id=?",
                ("qq:dm:10001",),
            )
            assert watermarks == [("turn-private-fast-1",)]
            assert _query_rows(data_dir, "SELECT summary FROM events WHERE summary LIKE '%野餐%'") == []

            # Private hot path current_turn_context must also finish immediately without waiting on group.
            private_env = TurnEnvelope(
                platform="qq",
                chat_type="dm",
                chat_id="10001",
                scope_id="default",
                sender_id="10001",
                session_id="qq:dm:10001",
                session_key="agent:main:qq:dm:10001",
                message_id="in-dm-2",
            )
            with _home_override(home), turn_envelope_scope(private_env):
                reader = SanaeMemProvider()
                try:
                    reader.initialize("qq:dm:10001", platform="qq", hermes_home=home, hook_worker=True)
                    t_ctx = time.monotonic()
                    rendered = reader.current_turn_context(
                        session_id="qq:dm:10001",
                        turn_id="turn-private-next",
                        platform="qq",
                        sender_id="10001",
                    )
                    ctx_elapsed = time.monotonic() - t_ctx
                finally:
                    reader.shutdown()

            assert ctx_elapsed < 0.8, f"current_turn_context was blocked by group lane ({ctx_elapsed:.3f}s)"
            assert private_thought in rendered

            # Keep group blocked past 1.1s total to prove private finished well before group unblocked.
            remaining_block = 1.1 - (time.monotonic() - block_started)
            if remaining_block > 0:
                time.sleep(remaining_block)
            assert _query_rows(data_dir, "SELECT summary FROM events WHERE summary LIKE '%野餐%'") == []

            # Now release group and confirm 2-tuple legacy wait waits for the group lane as well.
            release_group.set()
            assert inner_wiring.wait_for_pending_updates((data_dir, profile_key), timeout=10.0)
            group_events = _query_rows(data_dir, "SELECT summary FROM events")
            assert any("野餐" in str(row[0]) for row in group_events), group_events
        finally:
            release_group.set()
            _run_unloads(unloads)
            prototype.shutdown()


def test_prototype_shutdown_waits_for_in_flight_delivery_and_rejects_new_submissions(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """prototype.shutdown() must drain in-flight deliveries before returning and reject new ones."""
    home = tmp_path / "shutdown-barrier"
    in_flight_entered = threading.Event()
    release_in_flight = threading.Event()
    turn1_thought = "关机前这一条心里话必须稳稳落盘。"

    def respond(payload: dict[str, Any]) -> dict[str, Any]:
        last_content = str((payload.get("messages") or [{}])[-1].get("content") or "")
        if '"delivered_reply"' in last_content:
            if "SHUTDOWN_TURN_1_MARKER" in last_content:
                in_flight_entered.set()
                release_in_flight.wait(15.0)
            return {
                "sentences": [turn1_thought],
                "shared_sentences": None,
                "variables": {"tension": 0.1, "closeness": 0.3, "energy": 0.2},
            }
        return {"add": [], "update": [], "resolve": [], "new_commitments": [], "updates": []}

    with _LoopbackLLMServer(respond) as llm_server:
        data_dir = _write_home_config(home, base_url=llm_server.base_url)
        prototype, unloads = _register_plugin_in_home(home)
        shutdown_done = threading.Event()
        try:
            with caplog.at_level(logging.WARNING):
                turn1 = DeliveryReceipt(
                    session_id="qq:dm:10001",
                    platform="qq",
                    chat_type="dm",
                    chat_id="10001",
                    scope_id="default",
                    sender_id="10001",
                    turn_id="turn-barrier-1",
                    message_id="out-b-1",
                    incoming_message_id="in-b-1",
                    session_key="agent:main:qq:dm:10001",
                    user_message="SHUTDOWN_TURN_1_MARKER 晚上散步去了。",
                    delivered_text="SHUTDOWN_TURN_1_MARKER 慢点走挺好。",
                    clean_text="SHUTDOWN_TURN_1_MARKER 慢点走挺好。",
                    delivered_at=time.time(),
                    transport_confirmed=True,
                    is_bot=False,
                )
                with _home_override(home):
                    emit_delivery_receipt(turn1)

                assert in_flight_entered.wait(15.0), "in-flight delivery never reached Flash HTTP"

                def _do_shutdown() -> None:
                    prototype.shutdown()
                    shutdown_done.set()

                shutdown_thread = threading.Thread(target=_do_shutdown, daemon=True)
                shutdown_thread.start()

                # shutdown() must remain blocked until Turn 1 finishes.
                assert not shutdown_done.wait(0.25), "shutdown() returned before in-flight job completed"

                # A new receipt arriving while sealing must be rejected.
                turn2 = DeliveryReceipt(
                    session_id="qq:dm:10001",
                    platform="qq",
                    chat_type="dm",
                    chat_id="10001",
                    scope_id="default",
                    sender_id="10001",
                    turn_id="turn-barrier-2-sealing",
                    message_id="out-b-2",
                    incoming_message_id="in-b-2",
                    session_key="agent:main:qq:dm:10001",
                    user_message="SHUTDOWN_TURN_2_REJECTED",
                    delivered_text="这条在停机中应被拒收。",
                    clean_text="这条在停机中应被拒收。",
                    delivered_at=time.time() + 1.0,
                    transport_confirmed=True,
                    is_bot=False,
                )
                with _home_override(home):
                    emit_delivery_receipt(turn2)

                release_in_flight.set()
                assert shutdown_done.wait(15.0), "shutdown() deadlocked instead of finishing after drain"
                shutdown_thread.join(timeout=5.0)

                # Immediately after shutdown() returns, Turn 1 is already committed on disk.
                watermarks = _query_rows(
                    data_dir,
                    "SELECT turn_id FROM session_reply_watermarks WHERE session_id=?",
                    ("qq:dm:10001",),
                )
                assert watermarks == [("turn-barrier-1",)]
                revisions = _query_rows(data_dir, "SELECT turn_id, text FROM inner_state_revisions")
                assert len(revisions) == 1
                assert revisions[0][0] == "turn-barrier-1"
                assert turn1_thought in str(revisions[0][1])

                # A third receipt after shutdown() is also rejected.
                requests_after_shutdown = len(llm_server.requests)
                turn3 = DeliveryReceipt(
                    session_id="qq:dm:10001",
                    platform="qq",
                    chat_type="dm",
                    chat_id="10001",
                    scope_id="default",
                    sender_id="10001",
                    turn_id="turn-barrier-3-after",
                    message_id="out-b-3",
                    incoming_message_id="in-b-3",
                    session_key="agent:main:qq:dm:10001",
                    user_message="SHUTDOWN_TURN_3_REJECTED",
                    delivered_text="停机后新单也应被拒收。",
                    clean_text="停机后新单也应被拒收。",
                    delivered_at=time.time() + 2.0,
                    transport_confirmed=True,
                    is_bot=False,
                )
                with _home_override(home):
                    emit_delivery_receipt(turn3)
                assert inner_wiring.wait_for_pending_updates(timeout=2.0)
                assert len(llm_server.requests) == requests_after_shutdown
                assert len(_query_rows(data_dir, "SELECT turn_id FROM inner_state_revisions")) == 1
                assert "rejected background job after provider shutdown" in caplog.text
        finally:
            release_in_flight.set()
            _run_unloads(unloads)
            prototype.shutdown()


def test_cron_completion_binds_to_prototype_shutdown_barrier_and_decouples_from_group_lane(
    tmp_path: Path,
) -> None:
    """Cron heartbeat runs on its own 'cron' lane (unblocked by 'group') and holds prototype.shutdown()."""
    home = tmp_path / "cron-barrier"
    group_entered = threading.Event()
    release_group = threading.Event()
    cron_entered = threading.Event()
    release_cron = threading.Event()
    group_quote = "CRON_ISOLATION_GROUP_MARKER 明天一起看展览吧"
    cron_thought = "自由放空的时候听见了窗外的雨声。"

    def respond(payload: dict[str, Any]) -> dict[str, Any]:
        last_content = str((payload.get("messages") or [{}])[-1].get("content") or "")
        if "CRON_ISOLATION_GROUP_MARKER" in last_content:
            group_entered.set()
            release_group.wait(15.0)
            return {
                "events": [{"text": "明天群里约看展览", "quote": group_quote}],
                "impression": None,
                "new_commitments": [],
                "commitment_updates": [],
            }
        if '"delivered_reply": ""' in last_content and '"lived_experiences": ""' in last_content:
            cron_entered.set()
            release_cron.wait(15.0)
            return {
                "sentences": [cron_thought],
                "shared_sentences": None,
                "variables": {"tension": 0.0, "closeness": 0.2, "energy": 0.3},
            }
        return {"add": [], "update": [], "resolve": [], "new_commitments": [], "updates": []}

    with _LoopbackLLMServer(respond) as llm_server:
        data_dir = _write_home_config(home, base_url=llm_server.base_url, free_think=True)
        prototype, unloads = _register_plugin_in_home(home)
        shutdown_done = threading.Event()
        try:
            with _home_override(home):
                installed_jobs = [
                    job for job in cron_jobs.list_jobs(include_disabled=True) if not job.get("no_agent")
                ]
            assert len(installed_jobs) == 1, f"expected free-think cron job, got {installed_jobs}"
            job_spec = dict(installed_jobs[0])

            # 1. Block the group lane first.
            group_receipt = DeliveryReceipt(
                session_id="qq:group:880022",
                platform="qq",
                chat_type="group",
                chat_id="880022",
                scope_id="default",
                sender_id="u_group_member",
                turn_id="turn-cron-group-1",
                message_id="out-cg-1",
                incoming_message_id="in-cg-1",
                session_key="agent:main:qq:group:880022",
                user_message=group_quote,
                delivered_text="好呀，明天一起去。",
                clean_text="好呀，明天一起去。",
                delivered_at=time.time(),
                transport_confirmed=True,
            )
            with _home_override(home):
                emit_delivery_receipt(group_receipt)
            assert group_entered.wait(15.0), "group lane did not enter Flash"

            # 2. Emit cron completion while group is blocked -> cron lane must run immediately in parallel.
            cron_event = CronCompletionEvent(
                job_id=str(job_spec["id"]),
                job_name=str(job_spec["name"]),
                execution_id="exec-cron-1",
                session_id="cron:free-think:1",
                profile_home=str(home),
                status="completed",
                delivery_outcome="suppressed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                job_spec=job_spec,
            )
            with _home_override(home):
                emit_cron_completion(cron_event)
            assert cron_entered.wait(15.0), "cron lane was blocked by in-flight group lane"

            # 3. Start prototype.shutdown() -> must wait for BOTH cron and group lanes.
            def _do_shutdown() -> None:
                prototype.shutdown()
                shutdown_done.set()

            shutdown_thread = threading.Thread(target=_do_shutdown, daemon=True)
            shutdown_thread.start()
            assert not shutdown_done.wait(0.25), "shutdown() returned while cron and group were active"

            release_cron.set()
            assert not shutdown_done.wait(0.2), "shutdown() returned while group lane was still active"

            release_group.set()
            assert shutdown_done.wait(15.0), "shutdown() did not complete after both lanes drained"
            shutdown_thread.join(timeout=5.0)

            # Both cron heartbeat and group event are persisted immediately after shutdown().
            revisions = _query_rows(data_dir, "SELECT turn_id, text FROM inner_state_revisions")
            assert len(revisions) == 1
            assert revisions[0][0] == f"cron:{job_spec['id']}:exec-cron-1"
            assert cron_thought in str(revisions[0][1])
            group_events = _query_rows(data_dir, "SELECT summary FROM events")
            assert any("展览" in str(row[0]) for row in group_events), group_events

            # Post-shutdown cron completion must be rejected.
            post_shutdown_event = CronCompletionEvent(
                job_id=str(job_spec["id"]),
                job_name=str(job_spec["name"]),
                execution_id="exec-cron-2-rejected",
                session_id="cron:free-think:2",
                profile_home=str(home),
                status="completed",
                delivery_outcome="suppressed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                job_spec=job_spec,
            )
            with _home_override(home):
                emit_cron_completion(post_shutdown_event)
            assert inner_wiring.wait_for_pending_updates(timeout=2.0)
            assert len(_query_rows(data_dir, "SELECT turn_id FROM inner_state_revisions")) == 1
        finally:
            release_cron.set()
            release_group.set()
            _run_unloads(unloads)
            prototype.shutdown()


def test_drain_exception_records_atomic_status_without_leaking_chat_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unhandled exceptions inside _drain must atomically write status.json with zero chat text."""
    home = tmp_path / "drain-failure-status"
    data_dir = _write_home_config(home, base_url="http://127.0.0.1:9/v1")
    profile_key = daily_profile_key(home)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Seed an unrelated status component to verify atomic merge preserves existing sections.
    status_path = data_dir / "status.json"
    status_path.write_text(
        json.dumps({"components": {"memory_axis": {"ok": True, "stage": "private"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Trigger a genuine unhandled RuntimeError inside open_database() without mocking internals.
    with closing(sqlite3.connect(str(data_dir / "sanae.db"))) as conn:
        conn.execute("PRAGMA user_version = 999")
        conn.commit()

    prototype, unloads = _register_plugin_in_home(home)
    private_canary = "CANARY_PRIVATE_SECRET_TEXT_42"
    assistant_canary = "CANARY_ASSISTANT_SECRET_TEXT_99"
    group_canary = "CANARY_GROUP_SECRET_TEXT_77"
    try:
        with caplog.at_level(logging.WARNING):
            private_receipt = DeliveryReceipt(
                session_id="qq:dm:10001",
                platform="qq",
                chat_type="dm",
                chat_id="10001",
                scope_id="default",
                sender_id="10001",
                turn_id="turn-fail-private-1",
                message_id="out-fail-1",
                incoming_message_id="in-fail-1",
                session_key="agent:main:qq:dm:10001",
                user_message=f"{private_canary} 这句私聊正文绝不能进状态文件",
                delivered_text=f"{assistant_canary} 这句回复正文也绝不能进状态文件",
                clean_text=f"{assistant_canary} 这句回复正文也绝不能进状态文件",
                delivered_at=time.time(),
                transport_confirmed=True,
                is_bot=False,
            )
            with _home_override(home):
                emit_delivery_receipt(private_receipt)

            assert inner_wiring.wait_for_pending_updates((data_dir, profile_key, "private"), timeout=5.0)

            raw_status_1 = status_path.read_text(encoding="utf-8")
            status_1 = json.loads(raw_status_1)
            assert status_1["components"]["memory_axis"] == {"ok": True, "stage": "private"}
            bw_1 = status_1["components"]["background_worker"]
            assert bw_1["ok"] is False
            assert bw_1["stage"] == "drain"
            assert bw_1["lane"] == "private"
            assert bw_1["error"] == "RuntimeError"
            assert bw_1["failure_count"] == 1
            parsed_at = datetime.fromisoformat(bw_1["at"])
            assert parsed_at.tzinfo is not None
            assert private_canary not in raw_status_1
            assert assistant_canary not in raw_status_1

            # Also verify a group lane exception increments failure_count and keeps zero chat text.
            group_receipt = DeliveryReceipt(
                session_id="qq:group:770033",
                platform="qq",
                chat_type="group",
                chat_id="770033",
                scope_id="default",
                sender_id="u_group_err",
                turn_id="turn-fail-group-1",
                message_id="out-fail-g-1",
                incoming_message_id="in-fail-g-1",
                session_key="agent:main:qq:group:770033",
                user_message=f"{group_canary} 群聊正文同样绝不能进状态文件",
                delivered_text="收到。",
                clean_text="收到。",
                delivered_at=time.time() + 1.0,
                transport_confirmed=True,
            )
            with _home_override(home):
                emit_delivery_receipt(group_receipt)

            assert inner_wiring.wait_for_pending_updates((data_dir, profile_key, "group"), timeout=5.0)

            raw_status_2 = status_path.read_text(encoding="utf-8")
            status_2 = json.loads(raw_status_2)
            bw_2 = status_2["components"]["background_worker"]
            assert bw_2["ok"] is False
            assert bw_2["lane"] == "group"
            assert bw_2["error"] == "RuntimeError"
            assert bw_2["failure_count"] == 2
            assert group_canary not in raw_status_2
            assert private_canary not in caplog.text
            assert assistant_canary not in caplog.text
            assert group_canary not in caplog.text
            assert "background worker failed" in caplog.text
    finally:
        _run_unloads(unloads)
        # Barrier must have been released in finally even after unhandled job exceptions.
        prototype.shutdown()


def test_wait_for_pending_updates_target_and_lane_validation(tmp_path: Path) -> None:
    """Target/lane contracts accept 2-tuple, 3-tuple, and lane filters, and reject conflicts/invalid lanes."""
    data_dir = (tmp_path / "data").resolve()
    profile_key = "test-profile"

    assert inner_wiring.wait_for_pending_updates((data_dir, profile_key), timeout=1.0) is True
    assert inner_wiring.wait_for_pending_updates((data_dir, profile_key, "private"), timeout=1.0) is True
    assert inner_wiring.wait_for_pending_updates((data_dir, profile_key), lane="group", timeout=1.0) is True
    assert inner_wiring.wait_for_pending_updates(lane="cron", timeout=1.0) is True

    with pytest.raises(ValueError, match="conflicting lanes"):
        inner_wiring.wait_for_pending_updates((data_dir, profile_key, "private"), lane="group", timeout=1.0)

    invalid_lane = cast(inner_wiring.BackgroundLane, "bogus")
    with pytest.raises(ValueError, match="unsupported background queue lane"):
        inner_wiring.wait_for_pending_updates(lane=invalid_lane, timeout=1.0)


@pytest.mark.parametrize("backend", ["sqlite", "chroma"])
def test_group_delivery_shutdown_barrier_drains_and_rejects_after_shutdown(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    backend: str,
) -> None:
    """Group receipts must hold prototype.shutdown() until persisted and reject post-shutdown receipts."""
    from sanae_mem.infra.textutil import _embed, _pack_vector
    from sanae_mem.infra.words import _COLLECTION

    chromadb = pytest.importorskip("chromadb") if backend == "chroma" else None
    home = tmp_path / "group-barrier"
    group_entered = threading.Event()
    release_group = threading.Event()
    group_quote_1 = "GROUP_BARRIER_TURN_1 周末去海边看日出"
    summary = "周末群里约去海边看日出"

    def respond(payload: dict[str, Any]) -> dict[str, Any]:
        last_content = str((payload.get("messages") or [{}])[-1].get("content") or "")
        if "GROUP_BARRIER_TURN_1" in last_content:
            group_entered.set()
            release_group.wait(15.0)
            return {
                "events": [{"text": summary, "quote": group_quote_1}],
                "impression": None,
                "new_commitments": [],
                "commitment_updates": [],
            }
        return {"events": [], "impression": None, "new_commitments": [], "commitment_updates": []}

    with _LoopbackLLMServer(respond) as llm_server:
        data_dir = _write_home_config(home, base_url=llm_server.base_url, backend=backend)
        prototype, unloads = _register_plugin_in_home(home)
        shutdown_done = threading.Event()
        try:
            with caplog.at_level(logging.WARNING):
                receipt_1 = DeliveryReceipt(
                    session_id="qq:group:660044",
                    platform="qq",
                    chat_type="group",
                    chat_id="660044",
                    scope_id="default",
                    sender_id="u_member_1",
                    turn_id="turn-group-barrier-1",
                    message_id="out-gb-1",
                    incoming_message_id="in-gb-1",
                    session_key="agent:main:qq:group:660044",
                    user_message=group_quote_1,
                    delivered_text="好呀，我定闹钟。",
                    clean_text="好呀，我定闹钟。",
                    delivered_at=time.time(),
                    transport_confirmed=True,
                )
                with _home_override(home):
                    emit_delivery_receipt(receipt_1)
                assert group_entered.wait(15.0), "group receipt did not reach Flash HTTP"

                def _do_shutdown() -> None:
                    prototype.shutdown()
                    shutdown_done.set()

                shutdown_thread = threading.Thread(target=_do_shutdown, daemon=True)
                shutdown_thread.start()
                assert not shutdown_done.wait(0.25), "shutdown() returned before group receipt drained"

                release_group.set()
                assert shutdown_done.wait(15.0), "shutdown() did not complete after group receipt drained"
                shutdown_thread.join(timeout=5.0)

                events_after_drain = _query_rows(
                    data_dir, "SELECT id, raw, summary, vec, distilled_at FROM events",
                )
                assert len(events_after_drain) == 1
                event_id, raw, indexed_summary, vector, distilled_at = events_after_drain[0]
                expected_vector = _embed(summary, data_dir=data_dir)
                assert group_quote_1 in raw and indexed_summary == summary
                assert vector == _pack_vector(expected_vector) and distilled_at is None
                if chromadb is not None:
                    with chromadb.PersistentClient(path=str(data_dir / "chroma")) as client:
                        indexed = client.get_collection(_COLLECTION).get(
                            ids=[f"evt-{event_id}"], include=["documents", "embeddings"],
                        )
                        assert indexed["ids"] == [f"evt-{event_id}"]
                        assert indexed["documents"] == [summary]
                        assert indexed["embeddings"][0].tolist() == pytest.approx(expected_vector)

                requests_after_shutdown = len(llm_server.requests)
                receipt_2 = DeliveryReceipt(
                    session_id="qq:group:660044",
                    platform="qq",
                    chat_type="group",
                    chat_id="660044",
                    scope_id="default",
                    sender_id="u_member_1",
                    turn_id="turn-group-barrier-2-rejected",
                    message_id="out-gb-2",
                    incoming_message_id="in-gb-2",
                    session_key="agent:main:qq:group:660044",
                    user_message="GROUP_BARRIER_TURN_2 停机后的群聊不应入队",
                    delivered_text="收到。",
                    clean_text="收到。",
                    delivered_at=time.time() + 1.0,
                    transport_confirmed=True,
                )
                with _home_override(home):
                    emit_delivery_receipt(receipt_2)
                assert inner_wiring.wait_for_pending_updates(timeout=2.0)
                assert len(llm_server.requests) == requests_after_shutdown
                assert "rejected background job after provider shutdown (lane=group)" in caplog.text
        finally:
            release_group.set()
            _run_unloads(unloads)
            prototype.shutdown()

