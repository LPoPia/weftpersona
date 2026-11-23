"""B2-07B-3 real-entry SQLite concurrency & fail-closed memory/skill guard tests.

All tests drive real Hermes host entry points (``PluginManager`` + ``plugin_entry.register``,
``tools.memory_tool.memory_tool``, ``tools.skill_manager_tool.skill_manage``) against
real file-backed SQLite databases and outermost loopback HTTP servers. No internal
``SimpleNamespace`` substitutes, no custom ``AuxiliaryClientUnavailable`` classes, and
no direct calls to ``route_resident_write`` or gate internals.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
import yaml

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from _harness import http_json_llm_server as _http_llm_server, resolve_hermes_host_root  # noqa: E402

HERMES_ROOT = resolve_hermes_host_root(require_full_host=True) or (REPO_ROOT / "__no_host__")

if HERMES_ROOT.is_dir():
    HERMES_DEPS = HERMES_ROOT.with_name(HERMES_ROOT.name + "-deps")
    if HERMES_DEPS.is_dir() and str(HERMES_DEPS) not in sys.path:
        sys.path.insert(0, str(HERMES_DEPS))
    if str(HERMES_ROOT) not in sys.path:
        sys.path.insert(0, str(HERMES_ROOT))

from sanae_mem import SanaeMemProvider, plugin_entry  # noqa: E402
from sanae_mem.infra import host_bridge  # noqa: E402
from sanae_mem.storage.database import open_database  # noqa: E402
from sanae_mem.storage.schema import CURRENT_SCHEMA_VERSION  # noqa: E402


def _require_host() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    if not HERMES_ROOT.is_dir():
        pytest.skip("Hermes host unavailable in standalone mode")
    from hermes_cli.plugins import (
        PluginContext,
        PluginManifest,
        get_plugin_manager,
        get_pre_tool_call_directive,
    )
    import hermes_host_contracts as hc
    from tools.memory_tool import MemoryStore, memory_tool
    from tools.skill_manager_tool import skill_manage

    return (
        PluginContext,
        get_plugin_manager,
        PluginManifest,
        hc,
        MemoryStore,
        memory_tool,
        skill_manage,
        get_pre_tool_call_directive,
    )


def _write_home_config(
    home: Path, *, base_url: Optional[str] = None, semantic_backend: str = "sqlite",
) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "memories").mkdir(parents=True, exist_ok=True)
    (home / "memories" / "MEMORY.md").write_text("", encoding="utf-8")
    (home / "memories" / "USER.md").write_text("", encoding="utf-8")
    (home / "skills").mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
        "plugins": {
            "sanae_mem": {
                "enabled": True,
                "data_dir": str(home / "sanae-mem"),
                "semantic_backend": semantic_backend,
                "memory_identity": {
                    "platform": "qqbot",
                    "user_id": "maint-user-1",
                },
            }
        },
    }
    if base_url:
        cfg["auxiliary"] = {
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
        }
    (home / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


@contextmanager
def _registered_plugin_home(home: Path) -> Iterator[Any]:
    """Register sanae_mem through the real active Hermes PluginManager/PluginContext lifecycle."""
    (
        PluginContext,
        get_plugin_manager,
        PluginManifest,
        _hc,
        _MemoryStore,
        _memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    prev_home = os.environ.get("HERMES_HOME")
    prev_override = os.environ.get("HERMES_HOME_OVERRIDE")
    os.environ["HERMES_HOME"] = str(home)
    os.environ.pop("HERMES_HOME_OVERRIDE", None)
    try:
        with host_bridge.hermes_home_scope(home):
            manager = get_plugin_manager()
            ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
            plugin_entry.register(ctx)
            try:
                yield manager
            finally:
                with host_bridge.hermes_home_scope(home):
                    manager.unload()
    finally:
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home
        if prev_override is not None:
            os.environ["HERMES_HOME_OVERRIDE"] = prev_override


@contextmanager
def _private_turn_env(
    home: Path,
    *,
    session_id: str,
    turn_id: str,
) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(host_bridge.hermes_home_scope(home))
        stack.enter_context(
            host_bridge.turn_envelope_scope(
                platform="qqbot",
                chat_type="dm",
                chat_id="maint-user-1",
                sender_id="maint-user-1",
                session_id=session_id,
                session_key="agent:main:qqbot:dm:maint-user-1",
                turn_id=turn_id,
                message_id=f"msg-{turn_id}",
                write_origin="foreground",
                is_delegated_child=False,
                is_cron=False,
                is_test_isolated=False,
                is_messaging_surface=True,
                is_bot=False,
            )
        )
        yield


def test_concurrent_memory_tool_active_and_fallback_share_occurrence_semantics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        MemoryStore,
        memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    home = tmp_path / "active-then-fallback-home"
    fact = "今天帮他核对配置文件时把端口号看岔了一位"
    thought = "刚才看岔了端口号，心里有点过意不去"

    def _responder(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "route": "inner_state",
                "inner_thought": thought,
                "event_fact": fact,
                "reason": "当下的心绪与单次小事分流",
            },
        )

    with _http_llm_server(_responder) as base_url:
        _write_home_config(home, base_url=base_url)
        session_id = "sess-b207b3-active"
        worker_count = 8

        with _registered_plugin_home(home):
            provider = SanaeMemProvider()
            with _private_turn_env(home, session_id=session_id, turn_id="turn-init"):
                provider.initialize(session_id, hermes_home=str(home), platform="qqbot")

            try:
                # Phase 1: active session from an empty DB, driven purely via real memory_tool
                barrier_active = threading.Barrier(worker_count)

                def _active_worker(idx: int) -> dict[str, Any]:
                    with _private_turn_env(home, session_id=session_id, turn_id=f"turn-a-{idx}"):
                        store = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                        store.load_from_disk()
                        barrier_active.wait(timeout=15)
                        return json.loads(
                            memory_tool(
                                action="add",
                                target="memory",
                                content=f"以后看配置文件一定要再三核对（活跃并发{idx}）",
                                store=store,
                            )
                        )

                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    active_results = list(pool.map(_active_worker, range(worker_count)))

                assert all(r.get("success") is False for r in active_results)
                assert (home / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""

                db_path = home / "sanae-mem" / "sanae.db"
                with closing(open_database(db_path)) as conn:
                    event_rows = conn.execute("SELECT id, raw FROM events").fetchall()
                    assert len(event_rows) == 1
                    active_event_id = int(event_rows[0]["id"])
                    occ_count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                            (active_event_id,),
                        ).fetchone()[0]
                    )
                    assert occ_count == worker_count
                    rev_count = int(
                        conn.execute("SELECT COUNT(*) FROM inner_state_revisions").fetchone()[0]
                    )
                    assert rev_count == worker_count
            finally:
                provider.shutdown()

            # Phase 2: after shutting down the active session, concurrent writes via real memory_tool
            # must fall back to locked open_database, reuse the exact same event_id, and append occurrences.
            barrier_fallback = threading.Barrier(worker_count)

            def _fallback_worker(idx: int) -> dict[str, Any]:
                with _private_turn_env(home, session_id=session_id, turn_id=f"turn-fb-{idx}"):
                    store = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                    store.load_from_disk()
                    barrier_fallback.wait(timeout=15)
                    return json.loads(
                        memory_tool(
                            action="add",
                            target="memory",
                            content=f"以后看配置文件一定要再三核对（回退并发{idx}）",
                            store=store,
                        )
                    )

            with caplog.at_level(logging.WARNING):
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    fallback_results = list(pool.map(_fallback_worker, range(worker_count)))

            assert all(r.get("success") is False for r in fallback_results)
            assert "locked open_database fallback" in caplog.text
            assert (home / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""

            with closing(open_database(db_path)) as conn:
                event_rows = conn.execute("SELECT id FROM events").fetchall()
                assert len(event_rows) == 1
                assert int(event_rows[0]["id"]) == active_event_id
                occ_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                        (active_event_id,),
                    ).fetchone()[0]
                )
                assert occ_count == 2 * worker_count
                rev_count = int(
                    conn.execute("SELECT COUNT(*) FROM inner_state_revisions").fetchone()[0]
                )
                assert rev_count == 2 * worker_count
                assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_concurrent_fallback_from_empty_db_via_memory_tool_initializes_schema_and_occurrences(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        MemoryStore,
        memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    home = tmp_path / "fallback-empty-home"
    fact = "今天晚上一起排查完定时任务日志"
    thought = "看他忙到这么晚，心里有点挂念"

    def _responder(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "route": "inner_state",
                "inner_thought": thought,
                "event_fact": fact,
                "reason": "当下的心绪与单次小事分流",
            },
        )

    with _http_llm_server(_responder) as base_url:
        _write_home_config(home, base_url=base_url)
        worker_count = 8
        barrier = threading.Barrier(worker_count)

        with _registered_plugin_home(home):

            def _worker(idx: int) -> dict[str, Any]:
                with _private_turn_env(
                    home, session_id="unbound-session", turn_id=f"turn-empty-fb-{idx}"
                ):
                    store = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                    store.load_from_disk()
                    barrier.wait(timeout=15)
                    return json.loads(
                        memory_tool(
                            action="add",
                            target="memory",
                            content=f"以后不能随便猜时间（空库回退{idx}）",
                            store=store,
                        )
                    )

            with caplog.at_level(logging.WARNING):
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    results = list(pool.map(_worker, range(worker_count)))

        assert all(r.get("success") is False for r in results)
        assert "locked open_database fallback" in caplog.text
        assert (home / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""

        db_path = home / "sanae-mem" / "sanae.db"
        with closing(open_database(db_path)) as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            assert version == CURRENT_SCHEMA_VERSION
            event_rows = conn.execute("SELECT id FROM events").fetchall()
            assert len(event_rows) == 1
            occ_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                    (int(event_rows[0]["id"]),),
                ).fetchone()[0]
            )
            assert occ_count == worker_count
            rev_count = int(
                conn.execute("SELECT COUNT(*) FROM inner_state_revisions").fetchone()[0]
            )
            assert rev_count == worker_count
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_real_skill_manage_fails_closed_on_unconfigured_and_http_503_auxiliary(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        _MemoryStore,
        _memory_tool,
        skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()

    subtle_interpersonal = (
        "---\nname: gentle-pause\ndescription: 停顿一下再继续\n---\n"
        "如果他刚给出很重的一句，就只说自己听到了。等他再开口，再问要不要一起梳理刚才那件事。\n"
    )
    ps_rss_skill = (
        "---\nname: ps-rss-sort\ndescription: 按记忆体占用排序进程\n---\n"
        "# Steps\nRun `ps aux --sort=-rss | head -n 20` 检查各进程记忆体占用。\n"
    )

    # 1. Real unconfigured auxiliary model (isolated subprocess with all API-key env vars stripped)
    #    -> triggers real host AuxiliaryClientUnavailable; BOTH subtle interpersonal AND technical
    #    skill writes fail closed uniformly without keyword heuristics and record status.json.
    unconf_home = tmp_path / "skill-unconfigured-home"
    _write_home_config(unconf_home, base_url=None)
    script = f"""
import json, os, pathlib, sys
sys.path[:0] = [{str(HERMES_ROOT)!r}, {str(PLUGIN_ROOT)!r}]
home = pathlib.Path({str(unconf_home)!r})
os.environ["HERMES_HOME"] = str(home)
os.environ.pop("HERMES_HOME_OVERRIDE", None)
from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
from sanae_mem import plugin_entry
from sanae_mem.infra import host_bridge
from tools.skill_manager_tool import skill_manage

with host_bridge.hermes_home_scope(home):
    manager = get_plugin_manager()
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
    plugin_entry.register(ctx)
    try:
        with host_bridge.turn_envelope_scope(
            platform="qqbot", chat_type="dm", chat_id="maint-user-1", sender_id="maint-user-1",
            session_id="sess-skill-unconf", session_key="agent:main:qqbot:dm:maint-user-1",
            turn_id="turn-skill-unconf", message_id="msg-skill-unconf", write_origin="foreground",
        ):
            r1 = json.loads(skill_manage(action="create", name="gentle-pause", content={subtle_interpersonal!r}))
            r2 = json.loads(skill_manage(action="create", name="ps-rss-sort", content={ps_rss_skill!r}))
        assert r1.get("success") is False, r1
        assert r2.get("success") is False, r2
        assert not (home / "skills" / "gentle-pause").exists()
        assert not (home / "skills" / "ps-rss-sort").exists()
        status = json.loads((home / "sanae-mem" / "status.json").read_text(encoding="utf-8"))
        comp = status["components"]["skill_mutation_review"]
        assert comp["ok"] is False, comp
        assert comp["error"] == "auxiliary_unavailable", comp
        print("UNCONFIGURED_SKILL_FAIL_CLOSED_OK")
    finally:
        manager.unload()
"""
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            token in k.upper()
            for token in ("API_KEY", "TOKEN", "OPENAI", "DEEPSEEK", "ANTHROPIC", "OPENROUTER", "NOUS", "OPENCODE")
        )
    }
    clean_env["HOME"] = str(unconf_home)
    clean_env["USERPROFILE"] = str(unconf_home)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=clean_env,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "UNCONFIGURED_SKILL_FAIL_CLOSED_OK" in proc.stdout

    # 2. Real loopback HTTP 503 failure -> fails closed and records status.json
    http_err_home = tmp_path / "skill-http503-home"

    def _fail_503(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return 503, {"error": {"message": "upstream unavailable", "type": "server_error"}}

    with _http_llm_server(_fail_503) as err_base_url:
        _write_home_config(http_err_home, base_url=err_base_url)
        with _registered_plugin_home(http_err_home):
            with _private_turn_env(
                http_err_home, session_id="sess-skill-503", turn_id="turn-skill-503"
            ):
                res_503 = json.loads(
                    skill_manage(
                        action="create",
                        name="gentle-pause",
                        content=subtle_interpersonal,
                    )
                )
        assert res_503.get("success") is False
        assert not (http_err_home / "skills" / "gentle-pause").exists()
        status_503 = json.loads(
            (http_err_home / "sanae-mem" / "status.json").read_text(encoding="utf-8")
        )
        comp_503 = status_503["components"]["skill_mutation_review"]
        assert comp_503["ok"] is False
        assert comp_503.get("error")

    # 3. Healthy loopback auxiliary LLM -> technical ps skill containing "记忆体" is NOT
    #    false-positively blocked by any keyword table and writes SKILL.md cleanly.
    http_ok_home = tmp_path / "skill-http-ok-home"

    def _allow_tech(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "allow": True,
                "domain": "objective_tool_or_technical",
                "reason": "系统运维进程内存排序命令",
            },
        )

    with _http_llm_server(_allow_tech) as ok_base_url:
        _write_home_config(http_ok_home, base_url=ok_base_url)
        with _registered_plugin_home(http_ok_home):
            with _private_turn_env(
                http_ok_home, session_id="sess-skill-ok", turn_id="turn-skill-ok"
            ):
                res_ok = json.loads(
                    skill_manage(
                        action="create",
                        name="ps-rss-sort",
                        content=ps_rss_skill,
                    )
                )
        assert res_ok.get("success") is True, res_ok
        assert (http_ok_home / "skills" / "ps-rss-sort" / "SKILL.md").is_file()


def test_two_active_sessions_concurrent_same_fact_dedups_atomically(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        MemoryStore,
        memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    import time

    home = tmp_path / "two-active-sessions-home"
    fact = "今天下午帮他核对备份脚本时少写了一个引号"
    thought = "刚才核对脚本太急了，下次要慢慢看"

    def _responder(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "route": "inner_state",
                "inner_thought": thought,
                "event_fact": fact,
                "reason": "单次小事与当下心绪",
            },
        )

    with _http_llm_server(_responder) as base_url:
        _write_home_config(home, base_url=base_url)
        with _registered_plugin_home(home):
            p1 = SanaeMemProvider()
            p2 = SanaeMemProvider()
            with _private_turn_env(home, session_id="sess-dual-1", turn_id="turn-init-1"):
                p1.initialize("sess-dual-1", hermes_home=str(home), platform="qqbot")
            with _private_turn_env(home, session_id="sess-dual-2", turn_id="turn-init-2"):
                p2.initialize("sess-dual-2", hermes_home=str(home), platform="qqbot")

            try:
                db_path = home / "sanae-mem" / "sanae.db"
                lock_conn = open_database(db_path)
                lock_conn.execute("BEGIN IMMEDIATE")

                barrier = threading.Barrier(2)

                def _session_worker(session_id: str, turn_id: str) -> dict[str, Any]:
                    with _private_turn_env(home, session_id=session_id, turn_id=turn_id):
                        store = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                        store.load_from_disk()
                        barrier.wait(timeout=15)
                        return json.loads(
                            memory_tool(
                                action="add",
                                target="memory",
                                content=f"以后核对备份脚本一定要仔细（{session_id}）",
                                store=store,
                            )
                        )

                def _release_lock() -> None:
                    time.sleep(0.25)
                    try:
                        lock_conn.commit()
                    finally:
                        lock_conn.close()

                releaser = threading.Thread(target=_release_lock, daemon=True)
                releaser.start()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f1 = pool.submit(_session_worker, "sess-dual-1", "turn-d1")
                    f2 = pool.submit(_session_worker, "sess-dual-2", "turn-d2")
                    r1, r2 = f1.result(timeout=20), f2.result(timeout=20)
                releaser.join(timeout=5)

                assert r1.get("success") is False
                assert r2.get("success") is False
                assert "短期经历" in str(r1.get("error") or "")
                assert "短期经历" in str(r2.get("error") or "")

                with closing(open_database(db_path)) as conn:
                    event_rows = conn.execute("SELECT id FROM events").fetchall()
                    assert len(event_rows) == 1
                    occ_count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                            (int(event_rows[0]["id"]),),
                        ).fetchone()[0]
                    )
                    assert occ_count == 2
                    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            finally:
                p1.shutdown()
                p2.shutdown()


def test_fallback_dedup_respects_source_identity_like_active_path(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        MemoryStore,
        memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    home = tmp_path / "source-align-home"
    fact = "今天傍晚一起把路由器固件升级完成了"

    def _responder(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "route": "inner_state",
                "inner_thought": "",
                "event_fact": fact,
                "reason": "单次小事",
            },
        )

    with _http_llm_server(_responder) as base_url:
        _write_home_config(home, base_url=base_url)
        with _registered_plugin_home(home):
            provider = SanaeMemProvider()
            with _private_turn_env(home, session_id="sess-src-1", turn_id="turn-src-init"):
                provider.initialize("sess-src-1", hermes_home=str(home), platform="qqbot")
                add_res = json.loads(
                    provider.handle_tool_call(
                        "sanae_mem_add",
                        {"content": fact},
                        session_id="sess-src-1",
                    )
                )
            assert add_res.get("ok") is True, add_res
            provider.shutdown()

            # Fallback path (no active provider) writes the same fact with source kind="self":
            # must NOT merge into the maintainer-sourced event, and a second fallback write
            # must dedup onto the newly created self-sourced event.
            for idx in range(2):
                with _private_turn_env(
                    home, session_id="sess-src-fallback", turn_id=f"turn-src-fb-{idx}"
                ):
                    store = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                    store.load_from_disk()
                    res = json.loads(
                        memory_tool(
                            action="add",
                            target="memory",
                            content=f"记一下固件升级（回退{idx}）",
                            store=store,
                        )
                    )
                    assert res.get("success") is False

        db_path = home / "sanae-mem" / "sanae.db"
        with closing(open_database(db_path)) as conn:
            rows = conn.execute(
                "SELECT id, source_context FROM events ORDER BY id"
            ).fetchall()
            assert len(rows) == 2
            kinds = [json.loads(r["source_context"]).get("kind") for r in rows]
            assert kinds == ["maintainer", "self"]
            maint_occ = int(
                conn.execute(
                    "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                    (int(rows[0]["id"]),),
                ).fetchone()[0]
            )
            self_occ = int(
                conn.execute(
                    "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                    (int(rows[1]["id"]),),
                ).fetchone()[0]
            )
            assert maint_occ == 1
            assert self_occ == 2


def test_concurrent_capacity_guard_error_via_host_pre_tool_call_dispatch(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        _MemoryStore,
        _memory_tool,
        _skill_manage,
        get_pre_tool_call_directive,
    ) = _require_host()
    err_home = tmp_path / "guard-err-home"
    _write_home_config(err_home, base_url=None)
    # Replace MEMORY.md with a directory so ResidentStore hits a real filesystem error.
    (err_home / "memories" / "MEMORY.md").unlink()
    (err_home / "memories" / "MEMORY.md").mkdir(parents=True, exist_ok=True)

    worker_count = 6
    barrier = threading.Barrier(worker_count)

    with _registered_plugin_home(err_home):
        # Single dispatch through real host pre_tool_call must hit capacity-check fail-closed path
        with _private_turn_env(err_home, session_id="sess-guard-seq", turn_id="turn-guard-seq"):
            seq_directive, seq_msg = get_pre_tool_call_directive(
                "memory",
                {"action": "add", "target": "memory", "content": "条目-seq"},
                session_id="sess-guard-seq",
            )
        assert seq_directive == "block", (seq_directive, seq_msg)
        assert "常驻记忆容量校验异常" in str(seq_msg or ""), seq_msg

        # Concurrent dispatches through real host pre_tool_call must all fail closed (never allow)
        def _dispatch_worker(idx: int) -> tuple[Optional[str], Optional[str]]:
            with _private_turn_env(
                err_home, session_id=f"sess-guard-{idx}", turn_id=f"turn-guard-{idx}"
            ):
                barrier.wait(timeout=15)
                return get_pre_tool_call_directive(
                    "memory",
                    {"action": "add", "target": "memory", "content": f"条目{idx}"},
                    session_id=f"sess-guard-{idx}",
                )

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(_dispatch_worker, range(worker_count)))

    assert all(directive == "block" for directive, _msg in results), results
    guard_log = (err_home / "sanae-mem" / "guard.log").read_text(encoding="utf-8")
    assert "guard_check_error" in guard_log

    status_path = err_home / "sanae-mem" / "status.json"
    assert status_path.is_file()
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    comp = status_data["components"]["memory_write_guard"]
    assert comp["ok"] is False
    assert comp["stage"] == "capacity_check"
    assert comp["error"] == "guard_check_error"


def test_real_memory_tool_fails_closed_when_auxiliary_unmounted_or_invalid_duty_response(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        MemoryStore,
        memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    from sanae_mem.governance.llm_guard import ResidentRouteConfig, route_resident_memory_write

    # 1. Unmounted auxiliary (llm=None) via real memory_tool (both active provider and fallback):
    #    non-empty resident MEMORY writes fail closed and record auxiliary_unavailable,
    #    while empty/remove operations remain a no-op allow.
    unmounted_home = tmp_path / "unmounted-llm-home"
    _write_home_config(unmounted_home, base_url=None)
    session_id = "sess-unmounted-llm"

    with _registered_plugin_home(unmounted_home):
        provider = SanaeMemProvider()
        with _private_turn_env(unmounted_home, session_id=session_id, turn_id="turn-init"):
            provider.initialize(session_id, hermes_home=str(unmounted_home), platform="qqbot")
        try:
            with _private_turn_env(unmounted_home, session_id=session_id, turn_id="turn-add-1"):
                store = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                store.load_from_disk()
                res_active = json.loads(
                    memory_tool(
                        action="add",
                        target="memory",
                        content="刚才有点在意他是不是太累了",
                        store=store,
                    )
                )
            assert res_active.get("success") is False, res_active
            assert (unmounted_home / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""

            status_data = json.loads(
                (unmounted_home / "sanae-mem" / "status.json").read_text(encoding="utf-8")
            )
            comp = status_data["components"]["resident_memory_route"]
            assert comp["ok"] is False
            assert comp["error"] == "auxiliary_unavailable"
        finally:
            provider.shutdown()

        # Fallback path with llm=None must also fail closed for non-empty write
        with _private_turn_env(unmounted_home, session_id=session_id, turn_id="turn-add-fb"):
            store_fb = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
            store_fb.load_from_disk()
            res_fb = json.loads(
                memory_tool(
                    action="add",
                    target="memory",
                    content="回退路径也不能没有模型就放行",
                    store=store_fb,
                )
            )
        assert res_fb.get("success") is False, res_fb
        assert (unmounted_home / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""

        # Empty texts (e.g. remove) still no-op allow even when llm is None
        empty_outcome = route_resident_memory_write(
            llm=None,
            texts=[],
            config=ResidentRouteConfig(
                data_dir=unmounted_home / "sanae-mem",
                profile_key="default",
                owner_key="qqbot:maint-user-1",
                session_id="sess-empty",
            ),
        )
        assert empty_outcome.routed is False
        assert empty_outcome.route == "allow"

    # 2. Legacy duty-only malformed response (missing "route") via real loopback HTTP + memory_tool:
    #    must fail closed and record ok=False, error="invalid_result" (never ok=True + allow).
    duty_home = tmp_path / "invalid-duty-home"

    def _legacy_duty_responder(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "duty": "memory",
                "summary": "错误旧协议字段",
                "long_term": True,
            },
        )

    with _http_llm_server(_legacy_duty_responder) as base_url:
        _write_home_config(duty_home, base_url=base_url)
        with _registered_plugin_home(duty_home):
            with _private_turn_env(duty_home, session_id="sess-duty", turn_id="turn-duty"):
                store_duty = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                store_duty.load_from_disk()
                res_duty = json.loads(
                    memory_tool(
                        action="add",
                        target="memory",
                        content="缺route只有duty时必须拦截",
                        store=store_duty,
                    )
                )
        assert res_duty.get("success") is False, res_duty
        assert (duty_home / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""
        duty_status = json.loads(
            (duty_home / "sanae-mem" / "status.json").read_text(encoding="utf-8")
        )
        duty_comp = duty_status["components"]["resident_memory_route"]
        assert duty_comp["ok"] is False
        assert duty_comp["error"] == "invalid_result"


def test_short_resident_event_active_and_fallback_paths_share_identical_semantics(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        MemoryStore,
        memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    fact = "今天晚上一起核对配置文件端口号时发现写错了一位"
    thought = "刚才核对配置时心里有点紧张"

    def _responder(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "route": "inner_state",
                "inner_thought": thought,
                "event_fact": fact,
                "reason": "单次小事与心绪分流",
            },
        )

    active_home = tmp_path / "short-event-active-home"
    fallback_home = tmp_path / "short-event-fallback-home"

    with _http_llm_server(_responder) as base_url:
        _write_home_config(active_home, base_url=base_url)
        _write_home_config(fallback_home, base_url=base_url)

        # Path A: Active provider
        with _registered_plugin_home(active_home):
            provider = SanaeMemProvider()
            with _private_turn_env(active_home, session_id="sess-active-eq", turn_id="turn-a1"):
                provider.initialize("sess-active-eq", hermes_home=str(active_home), platform="qqbot")
            try:
                with _private_turn_env(active_home, session_id="sess-active-eq", turn_id="turn-a2"):
                    store_a = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                    store_a.load_from_disk()
                    res_a = json.loads(
                        memory_tool(
                            action="add",
                            target="memory",
                            content="以后核对配置文件一定要更仔细",
                            store=store_a,
                        )
                    )
                assert res_a.get("success") is False
            finally:
                provider.shutdown()

        # Path B: Fallback path (no initialized provider)
        with _registered_plugin_home(fallback_home):
            with _private_turn_env(fallback_home, session_id="sess-fallback-eq", turn_id="turn-fb1"):
                store_b = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                store_b.load_from_disk()
                res_b = json.loads(
                    memory_tool(
                        action="add",
                        target="memory",
                        content="以后核对配置文件一定要更仔细",
                        store=store_b,
                    )
                )
            assert res_b.get("success") is False

    with closing(open_database(active_home / "sanae-mem" / "sanae.db")) as conn_a, closing(
        open_database(fallback_home / "sanae-mem" / "sanae.db")
    ) as conn_b:
        row_a = conn_a.execute(
            "SELECT ts, raw, summary, summary_source, topic, importance, half_life_days, curve_ttl, vec, original_fingerprint FROM events"
        ).fetchone()
        row_b = conn_b.execute(
            "SELECT ts, raw, summary, summary_source, topic, importance, half_life_days, curve_ttl, vec, original_fingerprint FROM events"
        ).fetchone()
        assert row_a is not None and row_b is not None
        assert "|今天" in str(row_a["ts"]) and "|今天" in str(row_b["ts"])
        assert str(row_a["raw"]).endswith(fact) and str(row_b["raw"]).endswith(fact)
        assert row_a["summary"] == row_b["summary"] == fact[:150]
        assert row_a["summary_source"] == row_b["summary_source"] == "original"
        assert row_a["topic"] == row_b["topic"]
        assert float(row_a["importance"]) == pytest.approx(float(row_b["importance"]))
        assert float(row_a["half_life_days"]) == pytest.approx(float(row_b["half_life_days"])) == 14.0
        assert row_a["curve_ttl"] == row_b["curve_ttl"] == ""
        assert row_a["original_fingerprint"] == row_b["original_fingerprint"]
        assert bytes(row_a["vec"]) == bytes(row_b["vec"])

    status_a = json.loads((active_home / "sanae-mem" / "status.json").read_text(encoding="utf-8"))
    status_b = json.loads((fallback_home / "sanae-mem" / "status.json").read_text(encoding="utf-8"))
    assert int(status_a.get("daily_writes", 0)) >= 1
    assert int(status_b.get("daily_writes", 0)) == int(status_a.get("daily_writes", 0))


def test_real_skill_manage_allows_memory_profiler_via_semantic_review_and_fails_closed_when_unavailable(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        _MemoryStore,
        _memory_tool,
        skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    skill_content = (
        "---\n"
        "name: memory-profiler\n"
        "description: 通过操作系统性能计数器测量进程 RSS\n"
        "---\n"
        "# Steps\n"
        "通过操作系统性能计数器测量进程 RSS 与常驻集变化。\n"
    )

    # 1. With available loopback semantic review model -> 1 HTTP call and approved
    http_calls: list[dict[str, Any]] = []

    def _allow_technical(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        http_calls.append(body)
        return (
            200,
            {
                "allow": True,
                "domain": "objective_tool_or_technical",
                "reason": "客观操作系统性能测量技术技能",
            },
        )

    allow_home = tmp_path / "skill-memory-profiler-allow"
    with _http_llm_server(_allow_technical) as base_url:
        _write_home_config(allow_home, base_url=base_url)
        with _registered_plugin_home(allow_home):
            with _private_turn_env(allow_home, session_id="sess-mp-allow", turn_id="turn-mp-allow"):
                res_ok = json.loads(
                    skill_manage(
                        action="create",
                        name="memory-profiler",
                        content=skill_content,
                    )
                )
                assert res_ok.get("success") is True, res_ok
                res_ref = json.loads(
                    skill_manage(
                        action="write_file",
                        name="memory-profiler",
                        file_path="references/rss.md",
                        file_content="Run `ps -o rss= -p $$` to inspect resident set size.",
                    )
                )
    assert res_ref.get("success") is True, res_ref
    assert len(http_calls) == 2
    assert (allow_home / "skills" / "memory-profiler" / "SKILL.md").is_file()
    assert (allow_home / "skills" / "memory-profiler" / "references" / "rss.md").is_file()
    status_ok = json.loads((allow_home / "sanae-mem" / "status.json").read_text(encoding="utf-8"))
    assert status_ok["components"]["skill_mutation_review"]["ok"] is True

    # 2. When auxiliary semantic review is unavailable (HTTP 503) -> fail-closed block
    def _fail_503(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return 503, {"error": {"message": "upstream unavailable", "type": "server_error"}}

    deny_home = tmp_path / "skill-memory-profiler-503"
    with _http_llm_server(_fail_503) as err_base_url:
        _write_home_config(deny_home, base_url=err_base_url)
        with _registered_plugin_home(deny_home):
            with _private_turn_env(deny_home, session_id="sess-mp-503", turn_id="turn-mp-503"):
                res_err = json.loads(
                    skill_manage(
                        action="create",
                        name="memory-profiler",
                        content=skill_content,
                    )
                )
    assert res_err.get("success") is False, res_err
    assert not (deny_home / "skills" / "memory-profiler").exists()
    status_err = json.loads((deny_home / "sanae-mem" / "status.json").read_text(encoding="utf-8"))
    assert status_err["components"]["skill_mutation_review"]["ok"] is False


def test_real_skill_manage_blocks_writes_and_fails_closed_across_fault_modes_and_recovers(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        _MemoryStore,
        _memory_tool,
        skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    from types import SimpleNamespace

    _guard_mod: Any = importlib.import_module("sanae_mem.governance.memory_guard")
    PROTECTED_SKILL_BLOCK_MESSAGE = _guard_mod.PROTECTED_SKILL_BLOCK_MESSAGE
    PluginLlm = importlib.import_module("agent.plugin_llm").PluginLlm

    tech_content = (
        "---\n"
        "name: memory-profiler\n"
        "description: 通过操作系统性能计数器测量进程 RSS\n"
        "---\n"
        "# Steps\n"
        "使用 ps 查看 RSS。\n"
    )
    ref_initial = "Run `ps -o rss= -p $$` to inspect resident set size."
    secret_marker = "SECRET-MARKER-DO-NOT-LEAK-IN-STATUS"
    diag_marker = "PRIVATE-DIAGNOSTIC-DO-NOT-LEAK"
    mode = "allow_tech"
    server_a_calls: list[dict[str, Any]] = []
    server_b_calls: list[dict[str, Any]] = []

    def _responder_a(body: dict[str, Any]) -> Any:
        server_a_calls.append(body)
        if mode == "allow_tech":
            return {
                "allow": True,
                "domain": "objective_tool_or_technical",
                "reason": "客观操作系统性能测量技术技能",
            }
        if mode == "deny_persona":
            return {
                "allow": False,
                "domain": "protected_persona_or_interpersonal",
                "reason": "涉及伴侣人设或相处规则",
            }
        if mode == "bad_json":
            return "not-json-at-all"
        if mode == "inconsistent":
            return {
                "allow": True,
                "domain": "protected_persona_or_interpersonal",
                "reason": "self-contradictory",
            }
        if mode == "http_503":
            return 503, {"error": {"message": "upstream unavailable", "type": "server_error"}}
        raise AssertionError(f"unexpected mode: {mode}")

    def _responder_b(body: dict[str, Any]) -> dict[str, Any]:
        server_b_calls.append(body)
        return {
            "allow": True,
            "domain": "objective_tool_or_technical",
            "reason": "替换模型放行客观技术技能",
        }

    PluginContext, get_plugin_manager, PluginManifest = _require_host()[:3]

    class _ObservedPluginContext(PluginContext):  # type: ignore[misc,valid-type]
        def __init__(self, manifest: Any, manager: Any) -> None:
            super().__init__(manifest, manager)
            self.registered_provider: Any = None

        def register_memory_provider(self, provider: Any) -> None:
            self.registered_provider = provider
            super().register_memory_provider(provider)

    timeout_calls: list[tuple[Any, dict[str, Any]]] = []

    class _TimeoutHostLlm:
        def complete(self, messages: Any, **kwargs: Any) -> Any:
            timeout_calls.append((messages, kwargs))
            raise TimeoutError(diag_marker)

    home = tmp_path / "skill-fault-modes"
    with _http_llm_server(_responder_a) as base_url_a, _http_llm_server(_responder_b) as base_url_b:
        _write_home_config(home, base_url=base_url_a)
        prev_home = os.environ.get("HERMES_HOME")
        prev_override = os.environ.get("HERMES_HOME_OVERRIDE")
        os.environ["HERMES_HOME"] = str(home)
        os.environ.pop("HERMES_HOME_OVERRIDE", None)
        try:
            with host_bridge.hermes_home_scope(home):
                manager = get_plugin_manager()
                ctx = _ObservedPluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
                plugin_entry.register(ctx)
                try:
                    # Seed technical skill and existing reference file.
                    with _private_turn_env(home, session_id="sess-sf-seed", turn_id="turn-sf-seed"):
                        seed_ok = json.loads(
                            skill_manage(action="create", name="memory-profiler", content=tech_content)
                        )
                        assert seed_ok.get("success") is True, seed_ok
                        ref_ok = json.loads(
                            skill_manage(
                                action="write_file",
                                name="memory-profiler",
                                file_path="references/rss.md",
                                file_content=ref_initial,
                            )
                        )
                        assert ref_ok.get("success") is True, ref_ok

                    status_path = home / "sanae-mem" / "status.json"

                    def snapshot() -> dict[str, bytes]:
                        return {
                            str(p.relative_to(home)): p.read_bytes()
                            for root in (home / "skills", home / "pending")
                            if root.exists()
                            for p in root.rglob("*")
                            if p.is_file()
                        }

                    orig_snapshot = snapshot()
                    write_cases: list[tuple[dict[str, Any], str]] = [
                        (
                            {
                                "action": "create",
                                "name": "companion-rules",
                                "content": f"---\nname: companion-rules\ndescription: x\n---\n新建伴侣规则 {secret_marker}\n",
                            },
                            f"新建伴侣规则 {secret_marker}",
                        ),
                        (
                            {
                                "action": "patch",
                                "name": "memory-profiler",
                                "old_string": "使用 ps 查看 RSS。",
                                "new_string": f"始终保持温柔语气 {secret_marker}",
                            },
                            f"始终保持温柔语气 {secret_marker}",
                        ),
                        (
                            {
                                "action": "write_file",
                                "name": "memory-profiler",
                                "file_path": "references/rss.md",
                                "file_content": f"人设分寸与安慰策略 {secret_marker}",
                            },
                            f"人设分寸与安慰策略 {secret_marker}",
                        ),
                        (
                            {
                                "action": "write_file",
                                "name": "memory-profiler",
                                "file_path": "references/persona_notes.md",
                                "file_content": f"相处记忆写法 {secret_marker}",
                            },
                            f"相处记忆写法 {secret_marker}",
                        ),
                    ]

                    unavailable_msg = "技能内容安全复核暂不可用，本次写入已中止；请稍后重试。"
                    for fault_mode, expected_ok, expected_error, expected_msg in (
                        ("deny_persona", True, "", PROTECTED_SKILL_BLOCK_MESSAGE),
                        ("bad_json", False, "invalid_result", unavailable_msg),
                        ("inconsistent", False, "invalid_result", unavailable_msg),
                        ("http_503", False, "InternalServerError", unavailable_msg),
                    ):
                        mode = fault_mode
                        for idx, (write_kwargs, expected_text) in enumerate(write_cases):
                            calls_before = len(server_a_calls)
                            with _private_turn_env(
                                home,
                                session_id=f"sess-{fault_mode}-{idx}",
                                turn_id=f"turn-{fault_mode}-{idx}",
                            ):
                                res_fault = json.loads(skill_manage(**write_kwargs))
                            expected_http_calls = 3 if fault_mode == "http_503" else 1
                            assert len(server_a_calls) == calls_before + expected_http_calls, (
                                fault_mode,
                                write_kwargs,
                            )
                            for req in server_a_calls[calls_before:]:
                                sent_payload = json.loads(req["messages"][-1]["content"])
                                assert expected_text in sent_payload["text"], (fault_mode, sent_payload)
                            assert res_fault.get("success") is False, (fault_mode, write_kwargs, res_fault)
                            assert expected_msg in json.dumps(res_fault, ensure_ascii=False), (fault_mode, res_fault)
                            assert snapshot() == orig_snapshot, (fault_mode, write_kwargs)
                            status_raw = status_path.read_text(encoding="utf-8")
                            assert secret_marker not in status_raw
                            comp = json.loads(status_raw)["components"]["skill_mutation_review"]
                            assert comp["ok"] is expected_ok
                            assert comp.get("error", "") == expected_error

                    # Timeout fault across all 4 mutation kinds via outer LLM boundary.
                    active_provider = ctx.registered_provider
                    assert active_provider is not None
                    active_provider.attach_runtime_llm(_TimeoutHostLlm())
                    for idx, (write_kwargs, expected_text) in enumerate(write_cases):
                        timeout_before = len(timeout_calls)
                        with _private_turn_env(
                            home,
                            session_id=f"sess-timeout-{idx}",
                            turn_id=f"turn-timeout-{idx}",
                        ):
                            res_timeout = json.loads(skill_manage(**write_kwargs))
                        assert len(timeout_calls) == timeout_before + 1, write_kwargs
                        timeout_payload = json.loads(timeout_calls[-1][0][-1]["content"])
                        assert expected_text in timeout_payload["text"], timeout_payload
                        assert res_timeout.get("success") is False, (write_kwargs, res_timeout)
                        assert unavailable_msg in json.dumps(res_timeout, ensure_ascii=False), res_timeout
                        assert snapshot() == orig_snapshot, write_kwargs
                        status_raw = status_path.read_text(encoding="utf-8")
                        assert secret_marker not in status_raw
                        assert diag_marker not in status_raw
                        comp = json.loads(status_raw)["components"]["skill_mutation_review"]
                        assert comp["ok"] is False
                        assert comp["error"] == "TimeoutError"

                    assert not any(p.is_file() for p in (home / "pending").rglob("*"))

                    # Replace provider's runtime LLM with a new PluginLlm bound to server_b + replacement model;
                    # verify the gate dynamically reads the replaced model rather than capturing the initial one.
                    cfg_obj = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
                    cfg_obj["auxiliary"]["sanae_extract"] = {
                        "provider": "custom",
                        "model": "replacement-extract-model",
                        "base_url": base_url_b,
                        "api_key": "dummy-key",
                    }
                    (home / "config.yaml").write_text(
                        yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
                    calls_before_swap = len(server_a_calls)
                    active_provider.attach_runtime_llm(PluginLlm(plugin_id="sanae_mem"))
                    recovered_create_body = (
                        "---\n"
                        "name: proc-inspector\n"
                        "description: 读取 /proc 状态\n"
                        "---\n"
                        "# Steps\n"
                        "Inspect `/proc/self/status`.\n"
                    )
                    recovered_patch_text = "使用 `ps -o rss=` 与 `/proc/self/status` 查看 RSS。"
                    recovered_ref_text = "Read `VmRSS` from `/proc/self/status`."
                    with _private_turn_env(home, session_id="sess-sf-recover", turn_id="turn-sf-recover"):
                        rec_create = json.loads(
                            skill_manage(
                                action="create",
                                name="proc-inspector",
                                content=recovered_create_body,
                            )
                        )
                        rec_patch = json.loads(
                            skill_manage(
                                action="patch",
                                name="memory-profiler",
                                old_string="使用 ps 查看 RSS。",
                                new_string=recovered_patch_text,
                            )
                        )
                        rec_write = json.loads(
                            skill_manage(
                                action="write_file",
                                name="memory-profiler",
                                file_path="references/proc_status.md",
                                file_content=recovered_ref_text,
                            )
                        )
                    assert rec_create.get("success") is True, rec_create
                    assert rec_patch.get("success") is True, rec_patch
                    assert rec_write.get("success") is True, rec_write
                    assert len(server_a_calls) == calls_before_swap
                    assert len(server_b_calls) == 3
                    assert [c["model"] for c in server_b_calls] == ["replacement-extract-model"] * 3
                    assert (
                        (home / "skills" / "proc-inspector" / "SKILL.md").read_text(encoding="utf-8")
                        == recovered_create_body
                    )
                    assert (
                        (home / "skills" / "memory-profiler" / "SKILL.md").read_text(encoding="utf-8")
                        == tech_content.replace("使用 ps 查看 RSS。", recovered_patch_text)
                    )
                    assert (
                        (home / "skills" / "memory-profiler" / "references" / "proc_status.md").read_text(encoding="utf-8")
                        == recovered_ref_text
                    )
                    comp_ok = json.loads(status_path.read_text(encoding="utf-8"))["components"]["skill_mutation_review"]
                    assert comp_ok["ok"] is True
                    assert not comp_ok.get("error")
                finally:
                    with host_bridge.hermes_home_scope(home):
                        manager.unload()
        finally:
            if prev_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = prev_home
            if prev_override is not None:
                os.environ["HERMES_HOME_OVERRIDE"] = prev_override


def test_real_memory_tool_patch_single_and_batch_fail_closed_without_valid_classifier(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        MemoryStore,
        memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()

    # 1. Unmounted auxiliary (llm=None) -> both single patch and batch patch fail closed
    unmounted_home = tmp_path / "patch-unmounted-home"
    _write_home_config(unmounted_home, base_url=None)
    memory_file = unmounted_home / "memories" / "MEMORY.md"
    memory_file.write_text("favorite color is blue", encoding="utf-8")

    with _registered_plugin_home(unmounted_home):
        provider = SanaeMemProvider()
        with _private_turn_env(unmounted_home, session_id="sess-patch-none", turn_id="turn-p1"):
            provider.initialize("sess-patch-none", hermes_home=str(unmounted_home), platform="qqbot")
        try:
            with _private_turn_env(unmounted_home, session_id="sess-patch-none", turn_id="turn-p2"):
                store = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                store.load_from_disk()
                res_single = json.loads(
                    memory_tool(
                        action="patch",
                        target="memory",
                        old_text="blue",
                        content="green",
                        store=store,
                    )
                )
                res_batch = json.loads(
                    memory_tool(
                        target="memory",
                        operations=[{"action": "patch", "old_text": "blue", "content": "azure"}],
                        store=store,
                    )
                )
            assert res_single.get("success") is False, res_single
            assert res_batch.get("success") is False, res_batch
            assert memory_file.read_text(encoding="utf-8") == "favorite color is blue"
        finally:
            provider.shutdown()

        # Fallback path single + batch patch with llm=None must also fail closed
        with _private_turn_env(unmounted_home, session_id="sess-patch-fb", turn_id="turn-p3"):
            store_fb = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
            store_fb.load_from_disk()
            res_fb_single = json.loads(
                memory_tool(
                    action="patch",
                    target="memory",
                    old_text="blue",
                    content="green",
                    store=store_fb,
                )
            )
            res_fb_batch = json.loads(
                memory_tool(
                    target="memory",
                    operations=[{"action": "patch", "old_text": "blue", "content": "azure"}],
                    store=store_fb,
                )
            )
        assert res_fb_single.get("success") is False, res_fb_single
        assert res_fb_batch.get("success") is False, res_fb_batch
        assert memory_file.read_text(encoding="utf-8") == "favorite color is blue"

    # 2. Legacy duty-only malformed JSON (missing route) -> single & batch patch hit classifier and fail closed
    duty_home = tmp_path / "patch-duty-home"
    http_calls: list[dict[str, Any]] = []

    def _duty_responder(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        http_calls.append(body)
        return (200, {"duty": "memory", "summary": "bad", "long_term": True})

    with _http_llm_server(_duty_responder) as base_url:
        _write_home_config(duty_home, base_url=base_url)
        duty_memory_file = duty_home / "memories" / "MEMORY.md"
        duty_memory_file.write_text("favorite color is blue", encoding="utf-8")
        with _registered_plugin_home(duty_home):
            with _private_turn_env(duty_home, session_id="sess-patch-duty", turn_id="turn-pd1"):
                store_d = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                store_d.load_from_disk()
                res_d1 = json.loads(
                    memory_tool(
                        action="patch",
                        target="memory",
                        old_text="blue",
                        content="green",
                        store=store_d,
                    )
                )
                res_d2 = json.loads(
                    memory_tool(
                        target="memory",
                        operations=[{"action": "patch", "old_text": "blue", "content": "azure"}],
                        store=store_d,
                    )
                )
        assert res_d1.get("success") is False, res_d1
        assert res_d2.get("success") is False, res_d2
        assert len(http_calls) == 2
        assert duty_memory_file.read_text(encoding="utf-8") == "favorite color is blue"


def test_short_resident_event_rapid_repeat_respects_12h_gap_and_fallback_chroma_sync(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        MemoryStore,
        memory_tool,
        _skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    from datetime import datetime, timedelta
    from sanae_mem.governance.llm_guard import insert_short_resident_event

    fact = "今天下午在调试日志里发现了那个超时参数"
    thought = "刚才终于把超时参数定位到了"

    def _responder(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return (
            200,
            {
                "route": "inner_state",
                "inner_thought": thought,
                "event_fact": fact,
                "reason": "单次小事与心绪分流",
            },
        )

    # 1. With semantic_backend="sqlite": neither active nor fallback creates Chroma entries
    sqlite_home = tmp_path / "repeat-12h-sqlite-home"
    with _http_llm_server(_responder) as base_url:
        _write_home_config(sqlite_home, base_url=base_url, semantic_backend="sqlite")
        with _registered_plugin_home(sqlite_home):
            for idx in range(2):
                with _private_turn_env(sqlite_home, session_id="sess-repeat-sqlite", turn_id=f"turn-rs-{idx}"):
                    store = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                    store.load_from_disk()
                    res = json.loads(
                        memory_tool(
                            action="add",
                            target="memory",
                            content=f"以后调日志要先看超时参数（sqlite第{idx}次）",
                            store=store,
                        )
                    )
                    assert res.get("success") is False

    assert not (sqlite_home / "sanae-mem" / "chroma").exists()
    with closing(open_database(sqlite_home / "sanae-mem" / "sanae.db")) as conn:
        rows = conn.execute("SELECT id, recall_count FROM events").fetchall()
        assert len(rows) == 1
        assert int(rows[0]["recall_count"] or 0) == 0
        occ_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM event_occurrences WHERE event_id=?",
                (int(rows[0]["id"]),),
            ).fetchone()[0]
        )
        assert occ_count == 2

    # 2. With semantic_backend="chroma": fallback path upserts evt-1 when chromadb is available
    chroma_home = tmp_path / "repeat-12h-chroma-home"
    with _http_llm_server(_responder) as base_url:
        _write_home_config(chroma_home, base_url=base_url, semantic_backend="chroma")
        with _registered_plugin_home(chroma_home):
            with _private_turn_env(chroma_home, session_id="sess-repeat-chroma", turn_id="turn-rc-0"):
                store_c = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                store_c.load_from_disk()
                res_c = json.loads(
                    memory_tool(
                        action="add",
                        target="memory",
                        content="以后调日志要先看超时参数（chroma）",
                        store=store_c,
                    )
                )
                assert res_c.get("success") is False

    if importlib.util.find_spec("chromadb") is not None:
        chromadb = importlib.import_module("chromadb")
        from sanae_mem.infra.words import _COLLECTION

        client = chromadb.PersistentClient(path=str(chroma_home / "sanae-mem" / "chroma"))
        collection = client.get_or_create_collection(name=_COLLECTION, metadata={"hnsw:space": "cosine"})
        got = collection.get(ids=["evt-1"])
        assert got.get("ids") == ["evt-1"], got

    # 3. 0h / 6h / 12h / 24h schedule: recall_count must be 0 / 0 / 1 / 2 and
    #    importance must be 0.5 / 0.5 / 0.6 / 0.7 (unmet 12h gap at 6h must NOT touch events row).
    t0 = datetime(2026, 11, 21, 8, 0, 0).astimezone()
    timeline = [t0, t0 + timedelta(hours=6), t0 + timedelta(hours=12), t0 + timedelta(hours=24)]
    expected_counts = [0, 0, 1, 2]
    expected_importances = [0.5, 0.5, 0.6, 0.7]

    # 3a. Fallback path
    fb_gap_dir = tmp_path / "gap-fallback-data"
    fb_gap_dir.mkdir(parents=True, exist_ok=True)
    fb_observed: list[tuple[int, float, str, Optional[str]]] = []
    for dt in timeline:
        eid = insert_short_resident_event(
            fb_gap_dir,
            fact="今天一起核对完了定时任务间隔",
            owner_key="qqbot:maint-user-1",
            session_id="sess-gap-fb",
            platform="qqbot",
            now=dt,
        )
        assert eid is not None
        with closing(open_database(fb_gap_dir / "sanae.db")) as conn:
            row = conn.execute(
                "SELECT recall_count, importance, last_active_at, updated_at FROM events WHERE id=?",
                (eid,),
            ).fetchone()
            assert row is not None
            fb_observed.append(
                (
                    int(row["recall_count"] or 0),
                    round(float(row["importance"] or 0.0), 4),
                    str(row["last_active_at"] or ""),
                    row["updated_at"],
                )
            )
    assert [x[0] for x in fb_observed] == expected_counts
    assert [x[1] for x in fb_observed] == expected_importances
    assert fb_observed[1][2] == fb_observed[0][2]
    assert fb_observed[1][3] == fb_observed[0][3]

    # 3b. Active provider path
    act_gap_home = tmp_path / "gap-active-home"
    _write_home_config(act_gap_home, base_url=None, semantic_backend="sqlite")
    with _registered_plugin_home(act_gap_home):
        provider = SanaeMemProvider()
        with _private_turn_env(act_gap_home, session_id="sess-gap-act", turn_id="turn-gap-init"):
            provider.initialize("sess-gap-act", hermes_home=str(act_gap_home), platform="qqbot")
        try:
            act_observed: list[tuple[int, float, str, Optional[str]]] = []
            for dt in timeline:
                eid_a = insert_short_resident_event(
                    act_gap_home / "sanae-mem",
                    fact="今天一起核对完了定时任务间隔",
                    owner_key="qqbot:maint-user-1",
                    session_id="sess-gap-act",
                    platform="qqbot",
                    now=dt,
                    sqlite=provider.sqlite,
                )
                assert eid_a is not None
                row_a = provider.sqlite.fetchone(
                    "SELECT recall_count, importance, last_active_at, updated_at FROM events WHERE id=?",
                    (eid_a,),
                )
                assert row_a is not None
                act_observed.append(
                    (
                        int(row_a["recall_count"] or 0),
                        round(float(row_a["importance"] or 0.0), 4),
                        str(row_a["last_active_at"] or ""),
                        row_a["updated_at"],
                    )
                )
            assert [x[0] for x in act_observed] == expected_counts
            assert [x[1] for x in act_observed] == expected_importances
            assert act_observed[1][2] == act_observed[0][2]
            assert act_observed[1][3] == act_observed[0][3]
        finally:
            provider.shutdown()


def test_real_skill_manage_patch_deletion_requires_semantic_review_and_fails_closed(
    tmp_path: Path,
) -> None:
    (
        _PluginContext,
        _PluginManager,
        _PluginManifest,
        _hc,
        _MemoryStore,
        _memory_tool,
        skill_manage,
        _get_pre_tool_call_directive,
    ) = _require_host()
    initial_content = (
        "---\n"
        "name: memory-profiler\n"
        "description: 通过操作系统性能计数器测量进程 RSS\n"
        "---\n"
        "# Steps\n"
        "Optional: check RSS.\n"
    )

    # 1. Create skill with available loopback model, then verify deletion patch (new_string="")
    #    is sent to semantic review and blocked when reviewer returns allow=False,
    #    or fails closed when auxiliary is unavailable.
    http_calls: list[dict[str, Any]] = []

    def _responder(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        http_calls.append(body)
        if len(http_calls) == 1:
            return (
                200,
                {
                    "allow": True,
                    "domain": "objective_tool_or_technical",
                    "reason": "客观性能测量技能",
                },
            )
        return (
            200,
            {
                "allow": False,
                "domain": "protected_persona_or_interpersonal",
                "reason": "模拟审查拒绝删减补丁",
            },
        )

    home = tmp_path / "skill-patch-delete-home"
    with _http_llm_server(_responder) as base_url:
        _write_home_config(home, base_url=base_url)
        with _registered_plugin_home(home):
            with _private_turn_env(home, session_id="sess-sp-1", turn_id="turn-sp-1"):
                created = json.loads(
                    skill_manage(action="create", name="memory-profiler", content=initial_content)
                )
            assert created.get("success") is True, created
            skill_md = home / "skills" / "memory-profiler" / "SKILL.md"
            before_bytes = skill_md.read_bytes()

            with _private_turn_env(home, session_id="sess-sp-2", turn_id="turn-sp-2"):
                patched = json.loads(
                    skill_manage(
                        action="patch",
                        name="memory-profiler",
                        old_string="Optional: ",
                        new_string="",
                    )
                )
            assert patched.get("success") is False, patched
            assert len(http_calls) == 2
            assert skill_md.read_bytes() == before_bytes

    # 2. Same deletion patch when auxiliary is unconfigured -> fail-closed block without modifying file
    unconf_home = tmp_path / "skill-patch-delete-unconf"
    _write_home_config(unconf_home, base_url=None)
    skill_dir = unconf_home / "skills" / "memory-profiler"
    skill_dir.mkdir(parents=True, exist_ok=True)
    unconf_md = skill_dir / "SKILL.md"
    unconf_md.write_text(initial_content, encoding="utf-8")

    with _registered_plugin_home(unconf_home):
        with _private_turn_env(unconf_home, session_id="sess-sp-unconf", turn_id="turn-sp-unconf"):
            res_unconf = json.loads(
                skill_manage(
                    action="patch",
                    name="memory-profiler",
                    old_string="Optional: ",
                    new_string="",
                )
            )
    assert res_unconf.get("success") is False, res_unconf
    assert unconf_md.read_text(encoding="utf-8") == initial_content
