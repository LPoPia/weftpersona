"""Regression tests for B2-08B-1: read-only render_inner_state & current_turn_context.

All checks use public host/plugin interfaces, real host plugin loading + an
outermost local HTTP server for OpenAI-compatible completions, and direct SQLite
row/data_version assertions — no private ``._xxx`` backdoors and no internal
function/hook monkeypatching.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import closing, contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PARENT = REPO_ROOT / "plugins"
if str(PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_PARENT))

if os.environ.get("VIVEMIND_HERMES_SOURCE"):
    HERMES_ROOT = Path(os.environ["VIVEMIND_HERMES_SOURCE"]).expanduser().resolve()
else:
    _hermes_spec = importlib.util.find_spec("hermes_constants")
    if _hermes_spec is not None and _hermes_spec.origin:
        HERMES_ROOT = Path(_hermes_spec.origin).resolve().parent
    else:
        HERMES_ROOT = next(
            (
                path
                for path in (
                    REPO_ROOT.parent / "hermes-agent",
                    REPO_ROOT.parent / "hermes-minimal-host",
                    REPO_ROOT.parent / "hermes-minimal-host-20261112",
                )
                if path.is_dir()
            ),
            Path(tempfile.gettempdir()) / "vivemind-hermes-6093627",
        )
if HERMES_ROOT.is_dir() and str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

hc = pytest.importorskip("hermes_host_contracts")
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars  # noqa: E402
from hermes_cli.plugins import discover_plugins, get_plugin_manager  # noqa: E402

from sanae_mem.cognitive.daily_context import daily_profile_key  # noqa: E402
from sanae_mem.cognitive.daily_rules import KIND_THREAD  # noqa: E402
from sanae_mem.cognitive.inner_wiring import wait_for_pending_updates  # noqa: E402
from sanae_mem.infra.clock import sleep_day_iso  # noqa: E402
from sanae_mem.infra.textutil import _owner_key  # noqa: E402
from sanae_mem.integrations import register_video_memory  # noqa: E402
from sanae_mem.plugin_entry import SanaeMemProvider  # noqa: E402
from sanae_mem.runtime.scheduled import sync_scheduled_jobs  # noqa: E402


def _mount_sanae_mem_plugin(home: Path) -> None:
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    link = plugins_dir / "sanae_mem"
    if not link.exists():
        link.symlink_to(REPO_ROOT / "plugins" / "sanae_mem", target_is_directory=True)


def _wait_all_pending_updates(
    target: tuple[Path, str],
    *,
    lane: Literal["private", "group", "cron"] | None,
    timeout: float = 5.0,
) -> bool:
    deadline = time.monotonic() + timeout
    ok = wait_for_pending_updates(target, lane=lane, timeout=timeout)
    for name, mod in list(sys.modules.items()):
        if name.endswith(".cognitive.inner_wiring"):
            wait_fn = getattr(mod, "wait_for_pending_updates", None)
            if callable(wait_fn) and wait_fn is not wait_for_pending_updates:
                remaining = max(0.1, deadline - time.monotonic())
                ok = bool(wait_fn(target, lane=lane, timeout=remaining)) and ok
    return ok


def _write_home_config(
    home: Path,
    *,
    user_id: str = "1001",
    free_think: bool = False,
    llm_base_url: str = "",
) -> dict[str, object]:
    config: dict[str, object] = {
        "data_dir": str(home / "sanae-mem"),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": user_id},
        "free_think": free_think,
    }
    raw_cfg: dict[str, object] = {
        "model": (
            {
                "provider": "custom",
                "default": "test-main-model",
                "base_url": llm_base_url,
                "api_key": "test-key",
            }
            if llm_base_url
            else "test-model"
        ),
        "memory": {"provider": "sanae_mem"},
        "plugins": {
            "enabled": ["sanae_mem", "bilibili_explore"],
            "sanae_mem": config,
        },
    }
    if llm_base_url:
        raw_cfg["auxiliary"] = {
            "sanae_daily_flash": {
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": llm_base_url,
                "api_key": "test-key",
            }
        }
    (home / "config.yaml").write_text(
        yaml.safe_dump(raw_cfg, allow_unicode=True),
        encoding="utf-8",
    )
    return config


@contextmanager
def _private_turn_scope(
    *,
    platform: str = "telegram",
    user_id: str = "1001",
    session_id: str = "dm-readonly-render",
    message_id: str = "msg-1",
):
    reset_session_vars()
    tokens = set_session_vars(
        platform=platform,
        chat_type="dm",
        chat_id=user_id,
        user_id=user_id,
        session_id=session_id,
        session_key=f"agent:main:{platform}:dm:{user_id}",
        message_id=message_id,
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


def _snapshot_db(conn: sqlite3.Connection) -> tuple[int, list[tuple[object, ...]], list[tuple[object, ...]]]:
    data_version = int(conn.execute("PRAGMA data_version").fetchone()[0])
    inner_rows = [
        tuple(row)
        for row in conn.execute(
            "SELECT profile_key, revision, text, recent_words, recent_words_at, variables, "
            "felt_at, updated_at, processed_at, source, thought_source, turn_id, changed, "
            "owner_key, visibility, last_spark_at "
            "FROM inner_state_revisions ORDER BY revision ASC"
        ).fetchall()
    ]
    daily_rows = [
        tuple(row)
        for row in conn.execute(
            "SELECT entry_id, version, kind, slot, text, status, opened_on, expires_on, updated_at "
            "FROM daily_context ORDER BY entry_id ASC"
        ).fetchall()
    ]
    return data_version, inner_rows, daily_rows


@contextmanager
def _local_openai_server(
    *,
    spark_on_turn: dict[str, str | None],
    fail_inner_turns: set[str] | None = None,
    daily_items_by_turn: dict[str, list[dict[str, object]]] | None = None,
):
    turn_counter = {"n": 0}
    fail_set = fail_inner_turns or set()
    daily_map = daily_items_by_turn or {}
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
            messages = body.get("messages") or []
            system = str(messages[0].get("content") or "") if messages else ""
            user_payload = str(messages[-1].get("content") or "") if messages else ""

            status = 200
            content = "{}"
            if "daily_context" in system or ("items" in system and "new_commitments" not in system and "sentences" not in system):
                items: list[dict[str, object]] = []
                for marker, configured in daily_map.items():
                    if marker in user_payload:
                        items = configured
                        break
                content = json.dumps({"items": items}, ensure_ascii=False)
            elif "new_commitments" in system:
                content = '{"new_commitments": [], "resolutions": []}'
            elif "sentences" in system and "variables" in system:
                payload = json.loads(user_payload)
                user_text = str(payload.get("user") or "")
                with lock:
                    turn_counter["n"] += 1
                    turn_no = turn_counter["n"]
                if any(marker in user_text for marker in fail_set):
                    status = 500
                else:
                    spark = None
                    for marker, candidate in spark_on_turn.items():
                        if marker in user_text:
                            spark = candidate
                            break
                    content = json.dumps(
                        {
                            "sentences": [f"我记着这一阵的对话（第{turn_no}轮）。"],
                            "variables": {"tension": 0.0, "closeness": 0.3, "energy": 0.1},
                            "recent_words": None,
                            "spark": spark,
                        },
                        ensure_ascii=False,
                    )

            if status != 200:
                resp = json.dumps({"error": {"message": "simulated failure"}}).encode("utf-8")
            else:
                resp = json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": str(body.get("model") or "test-flash-model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_readonly_render_preserves_spark_until_verified_delivery_and_handles_failed_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    spark_text = "看着这漫出来的水，想到屋檐下那只胖鸽子"
    with _local_openai_server(
        spark_on_turn={"鲸鱼": spark_text},
        fail_inner_turns={"触发失败"},
    ) as llm_base_url:
        config = _write_home_config(home, llm_base_url=llm_base_url)
        _mount_sanae_mem_plugin(home)
        discover_plugins(force=True)
        provider = SanaeMemProvider(config)

        owner = _owner_key("telegram", user_id="1001")
        profile_key = daily_profile_key(home)
        target = (home / "sanae-mem", profile_key)
        db_path = target[0] / "sanae.db"
        now_ts = time.time()

        try:
            with _private_turn_scope(message_id="seed-1"):
                provider.initialize("dm-readonly-render", hermes_home=str(home), platform="telegram")
                assert json.loads(
                    provider.handle_tool_call(
                        "sanae_mem_add",
                        {"content": "守矢神社参道傍晚的夕照把石阶染成橘黄色", "importance": 0.9},
                    )
                )["ok"]
                assert json.loads(
                    provider.handle_tool_call(
                        "sanae_mem_add",
                        {"content": "智能咖啡论文里的温度闭环控制原理", "importance": 0.9},
                    )
                )["ok"]

            # Turn 1 delivery seeds a pending spark in inner_state_revisions.
            hc.emit_delivery_receipt(
                hc.DeliveryReceipt(
                    transport_confirmed=True,
                    session_id="dm-readonly-render",
                    turn_id="turn-1",
                    platform="telegram",
                    chat_type="dm",
                    chat_id="1001",
                    sender_id="1001",
                    user_message="刚看完鲸鱼跃出水面的纪录片段",
                    delivered_text="那画面听着就够静的。",
                    clean_text="那画面听着就够静的。",
                    delivered_at=now_ts - 60.0,
                    incoming_message_id="in-1",
                    message_id="out-1",
                    session_key="agent:main:telegram:dm:1001",
                    is_bot=False,
                )
            )
            assert _wait_all_pending_updates(target, lane="private", timeout=15.0)

            with closing(sqlite3.connect(db_path)) as observer:
                before_ver, before_inner, before_daily = _snapshot_db(observer)
                assert any(spark_text in str(row[2]) for row in before_inner)

                with _private_turn_scope(message_id="read-2"):
                    provider.initialize("dm-readonly-render", hermes_home=str(home), platform="telegram")
                    rendered_1 = provider.inner_state.render_inner_state(owner_key=owner, session_id="dm-readonly-render")
                    rendered_2 = provider.inner_state.render_inner_state(owner_key=owner, session_id="dm-readonly-render")
                    ctx_1 = provider.current_turn_context(
                        session_id="dm-readonly-render",
                        turn_id="turn-2",
                        platform="telegram",
                        sender_id="1001",
                        user_message="说正事吧",
                    )
                    ctx_2 = provider.current_turn_context(
                        session_id="dm-readonly-render",
                        turn_id="turn-2",
                        platform="telegram",
                        sender_id="1001",
                        user_message="说正事吧",
                    )
                    tool_read = json.loads(
                        provider.handle_tool_call(
                            "sanae_inner_state",
                            {"operation": "read"},
                            session_id="dm-readonly-render",
                            turn_id="turn-2",
                        )
                    )

                assert spark_text in rendered_1 and rendered_1 == rendered_2
                assert spark_text in ctx_1 and ctx_1 == ctx_2
                assert tool_read["ok"] is True and spark_text in tool_read["result"]["text"]

                after_ver, after_inner, after_daily = _snapshot_db(observer)
                assert after_ver == before_ver, "read-only renders must not write SQLite"
                assert after_inner == before_inner
                assert after_daily == before_daily

                # Turn 2 delivery: inner_state_update intentionally fails (returns None),
                # yet the exposed spark MUST be consumed once and last_spark_at preserved.
                hc.emit_delivery_receipt(
                    hc.DeliveryReceipt(
                        transport_confirmed=True,
                        session_id="dm-readonly-render",
                        turn_id="turn-2",
                        platform="telegram",
                        chat_type="dm",
                        chat_id="1001",
                        sender_id="1001",
                        user_message="这轮触发失败一下看看",
                        delivered_text="好，我记着。",
                        clean_text="好，我记着。",
                        delivered_at=now_ts - 30.0,
                        incoming_message_id="in-2",
                        message_id="out-2",
                        session_key="agent:main:telegram:dm:1001",
                        is_bot=False,
                    )
                )
                assert _wait_all_pending_updates(target, lane="private", timeout=15.0)

                post_ver, post_inner, _ = _snapshot_db(observer)
                assert post_ver != before_ver
                assert len(post_inner) == len(before_inner)
                assert all(spark_text not in str(row[2]) for row in post_inner)
                last_spark = post_inner[-1][-1]
                assert isinstance(last_spark, float) and last_spark == before_inner[-1][-1] and last_spark > 0.0

            with _private_turn_scope(message_id="read-3"):
                provider.initialize("dm-readonly-render", hermes_home=str(home), platform="telegram")
                ctx_3 = provider.current_turn_context(
                    session_id="dm-readonly-render",
                    turn_id="turn-3",
                    platform="telegram",
                    sender_id="1001",
                    user_message="下一轮",
                )
            assert spark_text not in ctx_3
        finally:
            get_plugin_manager().unload()
            provider.shutdown()


def test_cross_restart_old_receipt_replay_does_not_consume_newer_turn_spark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    spark_text = "想起屋檐下晒太阳的那只胖鸽子"
    owner = _owner_key("telegram", user_id="1001")
    profile_key = daily_profile_key(home)
    target = (home / "sanae-mem", profile_key)
    now_ts = time.time()

    old_receipt = hc.DeliveryReceipt(
        transport_confirmed=True,
        session_id="dm-readonly-render",
        turn_id="turn-old",
        platform="telegram",
        chat_type="dm",
        chat_id="1001",
        sender_id="1001",
        user_message="先聊点日常",
        delivered_text="嗯，听着呢。",
        clean_text="嗯，听着呢。",
        delivered_at=now_ts - 120.0,
        incoming_message_id="in-old",
        message_id="out-old",
        session_key="agent:main:telegram:dm:1001",
        is_bot=False,
    )

    with _local_openai_server(spark_on_turn={"鲸鱼": spark_text}) as llm_base_url:
        config = _write_home_config(home, llm_base_url=llm_base_url)
        _mount_sanae_mem_plugin(home)
        discover_plugins(force=True)
        provider_1 = SanaeMemProvider(config)

        try:
            with _private_turn_scope(message_id="seed-1"):
                provider_1.initialize("dm-readonly-render", hermes_home=str(home), platform="telegram")
                assert json.loads(
                    provider_1.handle_tool_call(
                        "sanae_mem_add",
                        {"content": "守矢神社参道傍晚的夕照把石阶染成橘黄色", "importance": 0.9},
                    )
                )["ok"]
                assert json.loads(
                    provider_1.handle_tool_call(
                        "sanae_mem_add",
                        {"content": "智能咖啡论文里的温度闭环控制原理", "importance": 0.9},
                    )
                )["ok"]

            hc.emit_delivery_receipt(old_receipt)
            assert _wait_all_pending_updates(target, lane="private", timeout=15.0)

            hc.emit_delivery_receipt(
                hc.DeliveryReceipt(
                    transport_confirmed=True,
                    session_id="dm-readonly-render",
                    turn_id="turn-newer",
                    platform="telegram",
                    chat_type="dm",
                    chat_id="1001",
                    sender_id="1001",
                    user_message="刚看完鲸鱼跃出水面的纪录片段",
                    delivered_text="那画面听着就够静的。",
                    clean_text="那画面听着就够静的。",
                    delivered_at=now_ts - 10.0,
                    incoming_message_id="in-newer",
                    message_id="out-newer",
                    session_key="agent:main:telegram:dm:1001",
                    is_bot=False,
                )
            )
            assert _wait_all_pending_updates(target, lane="private", timeout=15.0)

            state_before = provider_1.inner_state.read_inner_state(owner_key=owner)
            assert state_before is not None and spark_text in state_before.text
        finally:
            get_plugin_manager().unload()
            provider_1.shutdown()

        # Simulate process restart / plugin reload with empty in-memory DeliveryHistory.
        discover_plugins(force=True)
        provider_2 = SanaeMemProvider(config)
        try:
            hc.emit_delivery_receipt(old_receipt)
            assert _wait_all_pending_updates(target, lane="private", timeout=15.0)

            state_after = provider_2.inner_state.read_inner_state(target=target, owner_key=owner)
            assert state_after is not None and spark_text in state_after.text, (
                "replaying an already-settled old receipt after restart must not eat turn-newer's spark"
            )
        finally:
            get_plugin_manager().unload()
            provider_2.shutdown()


def test_spark_consume_sqlite_error_records_status_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    spark_text = "看着这漫出来的水，想到屋檐下那只胖鸽子"
    with _local_openai_server(spark_on_turn={"鲸鱼": spark_text}) as llm_base_url:
        config = _write_home_config(home, llm_base_url=llm_base_url)
        _mount_sanae_mem_plugin(home)
        discover_plugins(force=True)
        provider = SanaeMemProvider(config)

        profile_key = daily_profile_key(home)
        target = (home / "sanae-mem", profile_key)
        db_path = target[0] / "sanae.db"
        now_ts = time.time()

        try:
            with _private_turn_scope(message_id="seed-1"):
                provider.initialize("dm-readonly-render", hermes_home=str(home), platform="telegram")
                assert json.loads(
                    provider.handle_tool_call(
                        "sanae_mem_add",
                        {"content": "守矢神社参道傍晚的夕照把石阶染成橘黄色", "importance": 0.9},
                    )
                )["ok"]
                assert json.loads(
                    provider.handle_tool_call(
                        "sanae_mem_add",
                        {"content": "智能咖啡论文里的温度闭环控制原理", "importance": 0.9},
                    )
                )["ok"]

            hc.emit_delivery_receipt(
                hc.DeliveryReceipt(
                    transport_confirmed=True,
                    session_id="dm-readonly-render",
                    turn_id="turn-1",
                    platform="telegram",
                    chat_type="dm",
                    chat_id="1001",
                    sender_id="1001",
                    user_message="刚看完鲸鱼跃出水面的纪录片段",
                    delivered_text="那画面听着就够静的。",
                    clean_text="那画面听着就够静的。",
                    delivered_at=now_ts - 60.0,
                    incoming_message_id="in-1",
                    message_id="out-1",
                    session_key="agent:main:telegram:dm:1001",
                    is_bot=False,
                )
            )
            assert _wait_all_pending_updates(target, lane="private", timeout=15.0)

            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute(
                    "CREATE TRIGGER fail_spark_update BEFORE UPDATE ON inner_state_revisions "
                    "BEGIN SELECT RAISE(ABORT, 'simulated_spark_update_error'); END;"
                )

            hc.emit_delivery_receipt(
                hc.DeliveryReceipt(
                    transport_confirmed=True,
                    session_id="dm-readonly-render",
                    turn_id="turn-2",
                    platform="telegram",
                    chat_type="dm",
                    chat_id="1001",
                    sender_id="1001",
                    user_message="第二轮",
                    delivered_text="好的。",
                    clean_text="好的。",
                    delivered_at=now_ts - 10.0,
                    incoming_message_id="in-2",
                    message_id="out-2",
                    session_key="agent:main:telegram:dm:1001",
                    is_bot=False,
                )
            )
            assert _wait_all_pending_updates(target, lane="private", timeout=15.0)

            status_path = target[0] / "status.json"
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
            assert status_data["inner_state"]["last_failure"] == "spark_consume_failed"
            assert status_data["inner_state"]["failure_count"] >= 1
        finally:
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute("DROP TRIGGER IF EXISTS fail_spark_update")
            get_plugin_manager().unload()
            provider.shutdown()


def _install_bilibili_plugin(home: Path) -> Path:
    plugin_dir = home / "plugins" / "bilibili_explore"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: bilibili_explore\nversion: 0.1.0\ndescription: test bilibili plugin\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "CAPTURED_CTX = None\n\ndef register(ctx):\n    global CAPTURED_CTX\n    CAPTURED_CTX = ctx\n",
        encoding="utf-8",
    )
    return plugin_dir


def _find_captured_bilibili_ctx(plugin_dir: Path):
    target_suffix = str(plugin_dir / "__init__.py")
    for mod in list(sys.modules.values()):
        if (getattr(mod, "__file__", None) or "").endswith(target_suffix):
            return mod.CAPTURED_CTX
    raise AssertionError("bilibili_explore plugin was not loaded by discover_plugins")


def test_recent_activities_is_readonly_and_curiosity_closes_on_non_readonly_lifecycle_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with _local_openai_server(spark_on_turn={}) as llm_base_url:
        config = _write_home_config(home, free_think=True, llm_base_url=llm_base_url)
        _mount_sanae_mem_plugin(home)
        bili_dir = _install_bilibili_plugin(home)
        discover_plugins(force=True)
        source_ctx = _find_captured_bilibili_ctx(bili_dir)
        provider = SanaeMemProvider(config)

        owner = _owner_key("telegram", user_id="1001")
        profile_key = daily_profile_key(home)
        target = (home / "sanae-mem", profile_key)
        db_path = target[0] / "sanae.db"
        now_ts = time.time()

        try:
            with _private_turn_scope(message_id="init-1"):
                provider.initialize("dm-readonly-render", hermes_home=str(home), platform="telegram")
                assert provider.open_daily_entry(
                    owner_key=owner,
                    profile_key=profile_key,
                    kind=KIND_THREAD,
                    slot="paper-yellowing",
                    text="想顺手了解纸张为什么会泛黄。",
                    opened_on=sleep_day_iso(),
                    now=datetime.now().astimezone().isoformat(),
                )
                assert provider.daily_context.active_curiosity_threads() == ["想顺手了解纸张为什么会泛黄。"]

            # Seed a completed free-think reflection row so _recent_activities renders it.
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute(
                    "INSERT INTO free_think_reflections "
                    "(owner_key, job_id, execution_id, session_id, finished_at, reflection, created_at) "
                    "VALUES (?, 'job-1', 'exec-1', 'cron-1', ?, ?, ?)",
                    (
                        owner,
                        now_ts - 60.0,
                        "查了纸张为什么会泛黄，原来是因为纸里的木质素见光氧化。",
                        datetime.now().astimezone().isoformat(),
                    ),
                )

            with closing(sqlite3.connect(db_path)) as observer:
                before_ver, before_inner, before_daily = _snapshot_db(observer)

                with _private_turn_scope(message_id="read-ctx"):
                    provider.initialize("dm-readonly-render", hermes_home=str(home), platform="telegram")
                    ctx_1 = provider.current_turn_context(
                        session_id="dm-readonly-render",
                        turn_id="turn-curiosity-1",
                        platform="telegram",
                        sender_id="1001",
                        user_message="今天过得怎么样",
                    )
                    ctx_2 = provider.current_turn_context(
                        session_id="dm-readonly-render",
                        turn_id="turn-curiosity-1",
                        platform="telegram",
                        sender_id="1001",
                        user_message="今天过得怎么样",
                    )
                assert "木质素见光氧化" in ctx_1 and ctx_1 == ctx_2
                assert provider.daily_context.active_curiosity_threads() == ["想顺手了解纸张为什么会泛黄。"]

                after_ver, after_inner, after_daily = _snapshot_db(observer)
                assert after_ver == before_ver, "current_turn_context must not close curiosity threads in SQLite"
                assert after_inner == before_inner
                assert after_daily == before_daily

                # Now settle real private delivery -> _apply_verified_delivery closes the explored thread.
                hc.emit_delivery_receipt(
                    hc.DeliveryReceipt(
                        transport_confirmed=True,
                        session_id="dm-readonly-render",
                        turn_id="turn-curiosity-1",
                        platform="telegram",
                        chat_type="dm",
                        chat_id="1001",
                        sender_id="1001",
                        user_message="今天过得怎么样",
                        delivered_text="今天还挺充实的。",
                        clean_text="今天还挺充实的。",
                        delivered_at=now_ts - 10.0,
                        incoming_message_id="in-c1",
                        message_id="out-c1",
                        session_key="agent:main:telegram:dm:1001",
                        is_bot=False,
                    )
                )
                assert _wait_all_pending_updates(target, lane="private", timeout=15.0)
                assert provider.daily_context.active_curiosity_threads() == []

                # Non-read-only video watch write path closes matching curiosity threads.
                assert provider.open_daily_entry(
                    owner_key=owner,
                    profile_key=profile_key,
                    kind=KIND_THREAD,
                    slot="coffee-pid",
                    text="好奇咖啡机温控闭环怎么稳定水温。",
                    opened_on=sleep_day_iso(),
                    now=datetime.now().astimezone().isoformat(),
                )
                with _private_turn_scope(message_id="vid-1"):
                    provider.initialize("dm-readonly-render", hermes_home=str(home), platform="telegram")
                    video_client = register_video_memory(source_ctx)(provider)
                    assert video_client.record_video_watch(
                        {
                            "status": "ok",
                            "mode": "video_analysis",
                            "title": "咖啡机温控闭环解析",
                            "analysis": "看了咖啡机温控闭环解析，原来是用PID算法稳定萃取水温。",
                        },
                        platform="bilibili",
                        source_object_id="BV1TestPid99",
                        cid="cid-pid-1",
                        settled=True,
                        _watch_receipt="3123456789abcdef0123456789abcdef",
                    )
                assert "好奇咖啡机温控闭环怎么稳定水温。" not in provider.daily_context.active_curiosity_threads()

                # Non-read-only free-think cron completion with [SILENT] + diary entry closes matching curiosity threads.
                sync_scheduled_jobs(config)
                from cron import jobs as cron_jobs  # noqa: E402

                ft_job = next(j for j in cron_jobs.list_jobs(include_disabled=True) if not j["no_agent"])
                assert provider.open_daily_entry(
                    owner_key=owner,
                    profile_key=profile_key,
                    kind=KIND_THREAD,
                    slot="tea-oxidation",
                    text="想顺手查查乌龙茶发酵为什么叶子边缘会变红。",
                    opened_on=sleep_day_iso(),
                    now=datetime.now().astimezone().isoformat(),
                )
                diary_dir = home / "diary"
                diary_dir.mkdir(exist_ok=True)
                diary_body = "刚才搜了乌龙茶发酵的机理，原来是做青揉捻后叶缘细胞破损引发多酚氧化变红。"
                exec_id = "exec-curiosity-silent"
                entry_hash = hashlib.sha256(f"{owner}:{exec_id}:{diary_body}".encode()).hexdigest()
                (diary_dir / f"free-think-{entry_hash}.md").write_text(
                    f"# {datetime.now().astimezone().isoformat(timespec='minutes')}\n\n{diary_body}\n",
                    encoding="utf-8",
                )
                hc.emit_cron_completion(
                    hc.CronCompletionEvent(
                        job_id=ft_job["id"],
                        job_name=ft_job["name"],
                        execution_id=exec_id,
                        session_id="cron-curiosity-silent",
                        profile_home=str(home),
                        status="completed",
                        delivery_outcome="suppressed",
                        final_response="[SILENT]",
                        finished_at=datetime.now().astimezone().isoformat(),
                        job_spec=dict(ft_job),
                    )
                )
                assert _wait_all_pending_updates(target, lane="cron", timeout=15.0)
                assert "想顺手查查乌龙茶发酵为什么叶子边缘会变红。" not in provider.daily_context.active_curiosity_threads()

                # Cross-run isolation: a new [SILENT] execution with no diary/reflection of its own
                # must NOT pull in an older execution's reflection to close an active curiosity thread.
                assert provider.open_daily_entry(
                    owner_key=owner,
                    profile_key=profile_key,
                    kind=KIND_THREAD,
                    slot="aurora-colors",
                    text="想顺手了解极光为什么会有绿色和紫色。",
                    opened_on=sleep_day_iso(),
                    now=datetime.now().astimezone().isoformat(),
                )
                with closing(sqlite3.connect(db_path)) as conn, conn:
                    conn.execute(
                        "INSERT INTO free_think_reflections "
                        "(owner_key, job_id, execution_id, session_id, finished_at, reflection, created_at) "
                        "VALUES (?, ?, 'older-run', 'cron-older', ?, ?, ?)",
                        (
                            owner,
                            ft_job["id"],
                            time.time() - 120.0,
                            "查了极光颜色的成因，是太阳风高能粒子撞击高层大气氧原子和氮分子激发发光。",
                            datetime.now().astimezone().isoformat(),
                        ),
                    )
                hc.emit_cron_completion(
                    hc.CronCompletionEvent(
                        job_id=ft_job["id"],
                        job_name=ft_job["name"],
                        execution_id="new-run",
                        session_id="cron-new-run",
                        profile_home=str(home),
                        status="completed",
                        delivery_outcome="suppressed",
                        final_response="[SILENT]",
                        finished_at=datetime.now().astimezone().isoformat(),
                        job_spec=dict(ft_job),
                    )
                )
                assert _wait_all_pending_updates(target, lane="cron", timeout=15.0)
                assert "想顺手了解极光为什么会有绿色和紫色。" in provider.daily_context.active_curiosity_threads()
        finally:
            get_plugin_manager().unload()
            provider.shutdown()
