"""Multi-threaded concurrency stress tests for B2-06B-3 cognitive & integrations SqliteFacade migration."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, cast
from zoneinfo import ZoneInfo

import pytest
import yaml

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from _harness import http_json_llm_server as _http_llm_server, install_plugin_copy  # noqa: E402

from agent.memory_manager import MemoryManager  # noqa: E402
from hermes_cli.plugins import (  # noqa: E402
    discover_plugins,
    get_plugin_manager,
    invoke_hook,
)
from hermes_host_contracts import DeliveryReceipt, emit_delivery_receipt  # noqa: E402
import plugins.memory as host_memory  # noqa: E402
from tools.registry import registry  # noqa: E402

from sanae_mem import SanaeMemProvider  # noqa: E402
from sanae_mem.cognitive.nightly_journal import journal_entry_for  # noqa: E402
from sanae_mem.governance.account_binding import maintainer_anchor  # noqa: E402
from sanae_mem.governance.source_context import current_chat_scope, resolve_chat_identity  # noqa: E402
from sanae_mem.infra import host_bridge  # noqa: E402


def _setup_home_with_real_plugins(home: Path, data_dir: Path, base_url: str) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "memories").mkdir(parents=True, exist_ok=True)
    (home / "memories" / "MEMORY.md").write_text(
        "【2025-01-01 · 今日小记】旧日小记归档验证条目。\n",
        encoding="utf-8",
    )

    install_plugin_copy(home, PLUGIN_ROOT.parent, plugin_name="sanae_mem")
    plugins_dir = home / "plugins"

    addon = plugins_dir / "bilibili_public"
    addon.mkdir(parents=True, exist_ok=True)
    (addon / "plugin.yaml").write_text(
        "name: bilibili_public\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    addon_init = addon / "__init__.py"
    addon_init.write_text(
        "from __future__ import annotations\n"
        "from plugins.memory import import_provider_module\n\n"
        "bind_memory = None\n\n"
        "def register(ctx):\n"
        "    global bind_memory\n"
        "    integrations = import_provider_module('sanae_mem', 'integrations')\n"
        "    bind_memory = integrations.register_bilibili_memory(ctx)\n",
        encoding="utf-8",
    )

    cfg: dict[str, Any] = {
        "memory": {"provider": "sanae_mem"},
        "model": {
            "default": "test-main-model",
            "provider": "custom",
            "base_url": base_url,
            "api_key": "dummy-key",
        },
        "auxiliary": {
            "sanae_daily_flash": {
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": base_url,
                "api_key": "dummy-key",
            },
            "sanae_extract": {
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": base_url,
                "api_key": "dummy-key",
            },
        },
        "plugins": {
            "isolation": "in_process",
            "enabled": ["sanae_mem", "bilibili_public"],
            "sanae_mem": {
                "data_dir": str(data_dir),
                "semantic_backend": "sqlite",
                "shadow": False,
                "memory_identity": {"platform": "telegram", "user_id": "1001"},
                "partner_name": "PartnerA",
            },
        },
    }
    (home / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True),
        encoding="utf-8",
    )
    return addon_init.resolve()


def _find_loaded_binder(addon_init: Path) -> Any:
    for mod in list(sys.modules.values()):
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        try:
            if Path(mod_file).resolve() == addon_init:
                return getattr(mod, "bind_memory", None)
        except OSError:
            continue
    return None


def _wait_all_inner_wiring(timeout: float = 30.0) -> None:
    for mod in list(sys.modules.values()):
        if isinstance(mod, types.ModuleType) and mod.__name__.endswith(".cognitive.inner_wiring"):
            wait_fn = getattr(mod, "wait_for_pending_updates", None)
            if callable(wait_fn):
                assert wait_fn(timeout=timeout)


@pytest.mark.parametrize("rebind", [False, True], ids=["shutdown", "rebind"])
def test_nightly_journal_drains_before_provider_changes_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebind: bool,
) -> None:
    home = tmp_path / "old-home"
    next_home = tmp_path / "next-home"
    data_dir = home / "sanae-mem"
    config = {"semantic_backend": "sqlite", "persona_name": "夜记测试",
              "memory_identity": {"platform": "telegram", "user_id": "1001"}}
    owner = maintainer_anchor(config)
    now = datetime(2026, 11, 10, 23, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    model_entered = threading.Event()
    model_release = threading.Event()
    requests: list[dict[str, Any]] = []

    def respond(body: dict[str, Any]) -> dict[str, str]:
        requests.append(body)
        model_entered.set()
        assert model_release.wait(10)
        return {"text": "我把书放回架上。"}

    with _http_llm_server(respond) as base_url:
        _setup_home_with_real_plugins(home, data_dir, base_url)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        discover_plugins(force=True)
        manager = get_plugin_manager()
        loaded = host_memory.load_memory_provider("sanae_mem", register_skills=False)
        assert loaded is not None
        provider = SanaeMemProvider(config)
        provider.attach_runtime_llm(cast(SanaeMemProvider, loaded).runtime_llm())
        provider.initialize("old-session", hermes_home=home, hook_worker=True)

        def run_journal(at: datetime, session_id: str = "old-session") -> dict:
            with host_bridge.turn_envelope_scope(platform="cron", session_id=session_id):
                return provider.run_nightly_journal(now=at)

        try:
            provider.sqlite.execute(
                """INSERT INTO events (ts, raw, summary, owner_key, created_at,
                                       experience_subject, fact_subject)
                   VALUES (?, ?, ?, ?, ?, 'sanae', 'sanae')""",
                ("2026-11-10 23:29", "把书放回架上。", "把书放回架上。", owner,
                 (now - timedelta(minutes=1)).isoformat()),
            )
            old_memory = home / "memories" / "MEMORY.md"
            original_old_memory = old_memory.read_bytes()
            next_memory = next_home / "memories" / "MEMORY.md"
            next_memory.parent.mkdir(parents=True)
            next_entry = journal_entry_for("2025-01-02", "这条小记只属于后续目录。")
            next_memory.write_text(next_entry, encoding="utf-8")
            next_db = next_home / "sanae-mem" / "sanae.db"

            with ThreadPoolExecutor(max_workers=2) as threads:
                journal = threads.submit(run_journal, now)
                try:
                    assert model_entered.wait(10)
                    if rebind:
                        transition = threads.submit(
                            provider.initialize, "next-session", hermes_home=next_home, hook_worker=True,
                        )
                    else:
                        transition = threads.submit(provider.shutdown)
                    before_close = now.replace(hour=22)
                    for _ in range(100):
                        rejected = run_journal(before_close)
                        if rejected == {"ok": False, "reason": "shutdown"}:
                            break
                        assert rejected["reason"] == "before_close"
                        time.sleep(0.01)
                    else:
                        pytest.fail("Journal submissions did not seal during the lifecycle transition")
                    assert not transition.done() and not next_db.exists()
                    assert provider.sqlite.is_open and old_memory.read_bytes() == original_old_memory
                finally:
                    model_release.set()
                assert journal.result(timeout=10)["ok"]
                transition.result(timeout=10)

            assert len(requests) == 1
            assert "你是夜记测试的今日小记整理器" in requests[0]["messages"][0]["content"]
            assert "把书放回架上" in requests[0]["messages"][-1]["content"]
            assert "我把书放回架上。" in old_memory.read_text(encoding="utf-8")
            assert next_memory.read_text(encoding="utf-8") == next_entry
            assert not (next_home / "memory-archive").exists()
            with closing(sqlite3.connect(data_dir / "sanae.db")) as db:
                assert db.execute("SELECT status FROM nightly_journal_runs").fetchall() == [("completed",)]
                assert db.execute(
                    "SELECT session_id, owner_key FROM events WHERE promoted_reason='nightly_journal_sink'",
                ).fetchall() == [("old-session", owner)]
            if rebind:
                assert provider.sqlite.fetchone("SELECT status FROM nightly_journal_runs") is None
                assert run_journal(now, "next-session")["ok"]
                sunk = provider.sqlite.fetchone(
                    "SELECT raw, session_id, owner_key FROM events WHERE promoted_reason='nightly_journal_sink'",
                )
                assert sunk is not None and tuple(sunk) == (next_entry, "next-session", owner)
            else:
                assert not provider.sqlite.is_open
                assert provider.run_nightly_journal(now=now)["reason"] == "shutdown"
        finally:
            provider.shutdown()
            loaded.shutdown()
            manager.unload()


def test_b2_06b3_concurrent_cognitive_nightly_journal_and_bilibili_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes-home"
    data_dir = tmp_path / "sanae-mem"
    session_id = "b2-06b3-conc-session"

    def llm_responder(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages") or []
        sys_prompt = str(messages[0].get("content") or "") if messages else ""
        user_prompt = str(messages[-1].get("content") or "") if messages else ""
        if "new_commitments" in sys_prompt or "待办约定与承诺" in sys_prompt:
            return {"new_commitments": [], "resolutions": []}
        if "DAILY_FACT" in sys_prompt or "OPEN_THREAD" in sys_prompt:
            match = re.search(r"我今天要去(\d+)号馆复查", user_prompt)
            if match:
                tag = match.group(1)
                quote = f"我今天要去{tag}号馆复查"
                return {
                    "items": [
                        {
                            "kind": "DAILY_FACT",
                            "slot": f"visit_hall_{tag}",
                            "action": "set",
                            "text": f"维护者今天要去{tag}号馆复查",
                            "quote": quote,
                        }
                    ]
                }
            return {"items": []}
        if "今日小记" in sys_prompt:
            return {"text": "今天安静整理了一些日常片段，心里很踏实。"}
        return {
            "sentences": ["记着今天和他聊了一会儿，心里踏实。"],
            "variables": {"tension": 0.1, "closeness": 0.7, "energy": 0.3},
            "shared_sentences": None,
        }

    def _private_scope(msg_id: str):
        return host_bridge.turn_envelope_scope(
            platform="telegram",
            chat_type="dm",
            chat_id="1001",
            sender_id="1001",
            session_id=session_id,
            session_key="agent:main:telegram:dm:1001",
            message_id=msg_id,
        )

    with _http_llm_server(llm_responder) as base_url:
        addon_init = _setup_home_with_real_plugins(home, data_dir, base_url)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
        monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)

        memory_manager = MemoryManager()
        with host_bridge.hermes_home_scope(home):
            discover_plugins(force=True)
            plugin_manager = get_plugin_manager()
            try:
                loaded_plugins = plugin_manager.list_plugins()
                enabled_names = {
                    str(p.get("name")) for p in loaded_plugins if p.get("enabled") is True
                }
                assert "sanae_mem" in enabled_names, loaded_plugins
                assert "bilibili_public" in enabled_names, loaded_plugins

                loaded_provider = host_memory.load_memory_provider("sanae_mem", register_skills=False)
                assert loaded_provider is not None
                provider = cast(SanaeMemProvider, loaded_provider)

                binder = _find_loaded_binder(addon_init)
                assert binder is not None, "bilibili_public register(ctx) did not expose bind_memory"

                with _private_scope("init-msg"):
                    memory_manager.add_provider(provider)
                    memory_manager.initialize_all(
                        session_id=session_id,
                        hermes_home=str(home),
                        platform="telegram",
                        agent_context="primary",
                    )
                    identity = resolve_chat_identity(current_chat_scope(), {
                        "memory_identity": {"platform": "telegram", "user_id": "1001"},
                    })
                    owner_key = identity.private_owner_key
                    assert owner_key

                errors: List[BaseException] = []
                errors_lock = threading.Lock()
                state_lock = threading.Lock()
                expected_receipts: set[str] = set()
                expected_bvids: set[str] = set()
                journal_days_ok: set[str] = set()
                expected_daily_quotes: set[str] = set()

                num_threads = 10
                barrier = threading.Barrier(num_threads)

                def _record_error(exc: BaseException) -> None:
                    with errors_lock:
                        errors.append(exc)

                def bilibili_receipt_worker(worker_idx: int) -> None:
                    try:
                        barrier.wait(timeout=10)
                        base_ts = time.time()
                        for i in range(15):
                            receipt_id = f"bili-rcpt-{worker_idx}-{i}"
                            bvid = f"BV1{worker_idx:02d}{i:02d}56789"
                            watch_receipt = hashlib.md5(f"{worker_idx}-{i}".encode("utf-8")).hexdigest()
                            with host_bridge.hermes_home_scope(home), _private_scope(f"bili-{worker_idx}-{i}"):
                                client = binder.for_session(session_id)
                                assert client is not None
                                ok = client.record_accepted_reply(
                                    receipt_id=receipt_id,
                                    at=base_ts + i,
                                    text=f"B站并发回执内容 {worker_idx}-{i}",
                                    visibility="private",
                                    accepted=True,
                                )
                                assert ok is True, f"record_accepted_reply failed: {receipt_id}"
                                # Duplicate receipt_id must remain idempotent and not raise.
                                dup_ok = client.record_accepted_reply(
                                    receipt_id=receipt_id,
                                    at=base_ts + i,
                                    text=f"B站并发回执内容 {worker_idx}-{i}",
                                    visibility="private",
                                    accepted=True,
                                )
                                assert dup_ok is False, f"duplicate receipt should return False: {receipt_id}"

                                watched = client.record_video_watch(
                                    {
                                        "status": "ok",
                                        "bvid": bvid,
                                        "title": f"并发视频标题{worker_idx}_{i}",
                                        "mode": "captions_only",
                                        "lines": [{"text": f"并发字幕内容 {worker_idx}-{i}"}],
                                    },
                                    delivered=True,
                                    public=True,
                                    _watch_receipt=watch_receipt,
                                )
                                assert watched is True, f"record_video_watch failed: {bvid}"

                                raw_detail = registry.dispatch(
                                    "sanae_video_detail",
                                    {"bvid": bvid},
                                    scope=plugin_manager.scope_key,
                                    session_id=session_id,
                                )
                                detail_resp = json.loads(raw_detail) if isinstance(raw_detail, str) else raw_detail
                                assert detail_resp["ok"] is True, detail_resp
                                assert detail_resp["result"]["found"] is True, detail_resp
                            with state_lock:
                                expected_receipts.add(receipt_id)
                                expected_bvids.add(bvid)
                    except BaseException as exc:
                        _record_error(exc)

                def nightly_journal_worker(worker_idx: int) -> None:
                    try:
                        barrier.wait(timeout=10)
                        tz = ZoneInfo("Asia/Shanghai")
                        base_day = datetime(2026, 11, 10, 23, 30, 0, tzinfo=tz)
                        for offset in (0, worker_idx + 1):
                            target_dt = base_day + timedelta(days=offset)
                            day_str = target_dt.strftime("%Y-%m-%d")
                            with host_bridge.hermes_home_scope(home), _private_scope(f"nj-{worker_idx}-{offset}"):
                                res = provider.run_nightly_journal(now=target_dt, force=True)
                            assert res.get("ok") is True, res
                            with state_lock:
                                journal_days_ok.add(day_str)
                    except BaseException as exc:
                        _record_error(exc)

                def cognitive_delivery_and_hook_worker(worker_idx: int) -> None:
                    try:
                        barrier.wait(timeout=10)
                        for i in range(4):
                            tag = f"{worker_idx}{i}"
                            user_msg = f"我今天要去{tag}号馆复查，晚点再聊。"
                            reply_msg = f"好的，去{tag}号馆复查路上注意安全。"
                            turn_id = f"dlv-turn-{worker_idx}-{i}"
                            in_id = f"dlv-in-{worker_idx}-{i}"
                            out_id = f"dlv-out-{worker_idx}-{i}"

                            with host_bridge.hermes_home_scope(home), _private_scope(in_id):
                                hook_results = invoke_hook(
                                    "pre_llm_call",
                                    session_id=session_id,
                                    turn_id=turn_id,
                                    platform="telegram",
                                    sender_id="1001",
                                    user_message=user_msg,
                                    conversation_history=[],
                                )
                                context_blocks = [
                                    item["context"].strip()
                                    for item in hook_results
                                    if isinstance(item, dict)
                                    and isinstance(item.get("context"), str)
                                    and item["context"].strip()
                                ]
                                assert context_blocks, f"pre_llm_call returned no context: {hook_results!r}"
                                assert any("[我眼中的世界]" in block for block in context_blocks), context_blocks

                                receipt = DeliveryReceipt(
                                    session_id=session_id,
                                    platform="telegram",
                                    chat_id="1001",
                                    chat_type="dm",
                                    sender_id="1001",
                                    turn_id=turn_id,
                                    message_id=out_id,
                                    incoming_message_id=in_id,
                                    session_key="agent:main:telegram:dm:1001",
                                    user_message=user_msg,
                                    delivered_text=reply_msg,
                                    clean_text=reply_msg,
                                    delivered_at=time.time() + worker_idx * 10 + i,
                                    is_bot=False,
                                    transport_confirmed=True,
                                    transcript=(
                                        {
                                            "role": "user",
                                            "content": user_msg,
                                            "platform_message_id": in_id,
                                        },
                                        {"role": "assistant", "content": reply_msg},
                                    ),
                                )
                                dispatched = emit_delivery_receipt(receipt)
                                assert dispatched >= 1

                                threads_list = provider.daily_context.active_curiosity_threads()
                                assert isinstance(threads_list, list)
                                closed = provider.resolve_explored_curiosity_threads(
                                    f"查完了无关内容 {worker_idx}_{i}",
                                )
                                assert isinstance(closed, int)
                            with state_lock:
                                expected_daily_quotes.add(f"我今天要去{tag}号馆复查")
                    except BaseException as exc:
                        _record_error(exc)

                def event_save_worker(worker_idx: int) -> None:
                    try:
                        barrier.wait(timeout=10)
                        for i in range(10):
                            with host_bridge.hermes_home_scope(home), _private_scope(f"ev-{worker_idx}-{i}"):
                                raw = provider.handle_tool_call("sanae_meme_save", {
                                    "title": f"并发认知梗词{worker_idx}_{i}",
                                    "meaning": f"并发认知梗解释 {worker_idx}-{i}",
                                    "aliases": [f"并发别名{worker_idx}_{i}"],
                                }, session_id=session_id)
                            resp = json.loads(raw)
                            assert resp["ok"] is True, resp
                    except BaseException as exc:
                        _record_error(exc)

                threads = [
                    threading.Thread(target=bilibili_receipt_worker, args=(0,)),
                    threading.Thread(target=bilibili_receipt_worker, args=(1,)),
                    threading.Thread(target=bilibili_receipt_worker, args=(2,)),
                    threading.Thread(target=nightly_journal_worker, args=(0,)),
                    threading.Thread(target=nightly_journal_worker, args=(1,)),
                    threading.Thread(target=nightly_journal_worker, args=(2,)),
                    threading.Thread(target=cognitive_delivery_and_hook_worker, args=(0,)),
                    threading.Thread(target=cognitive_delivery_and_hook_worker, args=(1,)),
                    threading.Thread(target=event_save_worker, args=(0,)),
                    threading.Thread(target=event_save_worker, args=(1,)),
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=60)
                    assert not t.is_alive(), "worker thread timed out"

                _wait_all_inner_wiring(timeout=30.0)
                assert not errors, f"Concurrent errors: {errors!r}"

                # Verify SQLite file integrity & complete persistence with a fresh connection
                db_path = data_dir / "sanae.db"
                assert db_path.is_file()
                check_conn = sqlite3.connect(str(db_path))
                check_conn.row_factory = sqlite3.Row
                try:
                    # 1. Bilibili receipts: exact count, no duplicates
                    bili_rows = check_conn.execute(
                        "SELECT receipt_id FROM bilibili_reply_receipts WHERE owner_key=?",
                        (owner_key,),
                    ).fetchall()
                    stored_receipts = {str(r["receipt_id"]) for r in bili_rows}
                    assert stored_receipts == expected_receipts
                    assert len(bili_rows) == len(expected_receipts) == 45

                    # 2. Bilibili video watches: all 45 persisted cleanly
                    video_rows = check_conn.execute(
                        "SELECT bvid FROM video_watches WHERE owner_key=?",
                        (owner_key,),
                    ).fetchall()
                    stored_bvids = {str(r["bvid"]) for r in video_rows}
                    assert expected_bvids.issubset(stored_bvids)
                    assert len(expected_bvids) == 45

                    # 3. Nightly journal runs: all days completed
                    nj_rows = check_conn.execute(
                        "SELECT day, status FROM nightly_journal_runs WHERE owner_key=?",
                        (owner_key,),
                    ).fetchall()
                    completed_days = {str(r["day"]) for r in nj_rows if str(r["status"]) == "completed"}
                    assert journal_days_ok.issubset(completed_days)

                    # 4. Expired 2025-01-01 journal entry was archived to journal.jsonl and sunk to events
                    archive_file = home / "memory-archive" / "journal.jsonl"
                    assert archive_file.is_file()
                    archive_text = archive_file.read_text(encoding="utf-8")
                    assert "旧日小记归档验证条目" in archive_text
                    sunk_row = check_conn.execute(
                        "SELECT id FROM events WHERE owner_key=? AND promoted_reason='nightly_journal_sink'",
                        (owner_key,),
                    ).fetchone()
                    assert sunk_row is not None

                    # 5. Daily context entries settled via real emit_delivery_receipt path
                    daily_rows = check_conn.execute(
                        "SELECT kind, slot, text, source_quote FROM daily_context "
                        "WHERE owner_key=? AND status='active' AND kind='DAILY_FACT'",
                        (owner_key,),
                    ).fetchall()
                    assert len(daily_rows) >= 1, "expected DAILY_FACT rows settled via emit_delivery_receipt"
                    stored_quotes = {str(r["source_quote"]) for r in daily_rows}
                    assert stored_quotes & expected_daily_quotes, (stored_quotes, expected_daily_quotes)
                finally:
                    check_conn.close()
            finally:
                memory_manager.shutdown_all()
                with host_bridge.hermes_home_scope(home):
                    plugin_manager.unload()
