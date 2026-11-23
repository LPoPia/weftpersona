"""Multithreaded CAS concurrency, backoff, and exhaustion tests for B2-07B-1."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest import TestCase

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
PLUGIN_ROOT = REPO_ROOT / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from _harness import (  # noqa: E402
    http_json_llm_server,
    install_plugin_copy,
    write_home_config,
)
from sanae_mem import SanaeMemProvider  # noqa: E402
from sanae_mem.cognitive.daily_context import daily_profile_key  # noqa: E402
from sanae_mem.cognitive.impression import read_impression, write_impression  # noqa: E402
from sanae_mem.cognitive.inner_state_store import latest_state  # noqa: E402
from sanae_mem.governance.llm_guard import (  # noqa: E402
    ResidentRouteConfig,
    route_resident_memory_write,
    write_short_inner_state,
)
from sanae_mem.infra import host_bridge  # noqa: E402
from sanae_mem.infra.contracts import CasConflictError, ViveMindError  # noqa: E402
from sanae_mem.infra.status_store import read_status_file  # noqa: E402
from sanae_mem.storage.sqlite_facade import connect_readonly, connect_readwrite  # noqa: E402


def _init_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch | None = None) -> tuple[SanaeMemProvider, Path, str]:
    home = tmp_path / "hermes-home"
    data_dir = tmp_path / "sanae-data"
    home.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if monkeypatch is not None:
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    else:
        os.environ["HERMES_HOME"] = str(home)
        os.environ["HERMES_HOME_OVERRIDE"] = str(home)
    with host_bridge.turn_envelope_scope(
        platform="discord",
        chat_type="dm",
        chat_id="dm-owner",
        sender_id="owner-1",
        session_id="sess-init",
        session_key="sess-init",
        message_id="msg-init",
    ):
        provider = SanaeMemProvider({
            "data_dir": str(data_dir),
            "semantic_backend": "sqlite",
            "shadow": False,
        })
        provider.initialize("sess-init", hermes_home=str(home), platform="discord")
    return provider, data_dir, "owner-1"


def test_cas_conflict_error_hierarchy_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert issubclass(CasConflictError, ViveMindError)
    assert issubclass(CasConflictError, RuntimeError)

    provider, data_dir, owner_key = _init_provider(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError):
            provider.inner_state.write_inner_state("此刻心里有点乱", owner_key=owner_key, max_retries=-1)
        with pytest.raises(ValueError):
            write_short_inner_state(data_dir, "default", "此刻心里有点乱", owner_key=owner_key, max_retries=-1)
        with pytest.raises(ValueError):
            write_impression((data_dir, "default", owner_key), "稳重可靠", turn_id="t-1", max_retries=-1)
    finally:
        provider.shutdown()


def test_write_inner_state_multithreaded_monotonic_revisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, data_dir, owner_key = _init_provider(tmp_path, monkeypatch)
    threads = 8
    writes_per_thread = 6
    barrier = threading.Barrier(threads)
    revisions: list[int] = []
    rev_lock = threading.Lock()
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(writes_per_thread):
                state = provider.inner_state.write_inner_state(
                    f"心里话 w={worker_id} s={step}",
                    owner_key=owner_key,
                    turn_id=f"turn-{worker_id}-{step}",
                )
                with rev_lock:
                    revisions.append(state.revision)
        except BaseException as exc:
            errors.append(exc)

    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futs = [pool.submit(worker, i) for i in range(threads)]
            for fut in futs:
                fut.result()

        assert not errors
        expected_total = threads * writes_per_thread
        assert sorted(revisions) == list(range(1, expected_total + 1))

        _, profile_key = provider.inner_state.inner_state_target()
        head = latest_state(data_dir / "sanae.db", profile_key)
        assert head is not None
        assert head.revision == expected_total
        with closing(connect_readonly(data_dir / "sanae.db")) as conn:
            db_revs = [
                int(row[0])
                for row in conn.execute(
                    "SELECT revision FROM inner_state_revisions WHERE profile_key=? ORDER BY revision ASC",
                    (profile_key,),
                ).fetchall()
            ]
        assert db_revs == list(range(1, expected_total + 1))
    finally:
        provider.shutdown()


def test_write_short_inner_state_multithreaded_monotonic_revisions(tmp_path: Path) -> None:
    data_dir = tmp_path / "sanae-data"
    profile_key = "profile-short"
    owner_key = "owner-short"
    threads = 8
    writes_per_thread = 6
    barrier = threading.Barrier(threads)
    revisions: list[int] = []
    rev_lock = threading.Lock()
    errors: list[BaseException] = []

    first = write_short_inner_state(
        data_dir, profile_key, "初始心里话", owner_key=owner_key, turn_id="turn-0",
    )
    assert first.revision == 1

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(writes_per_thread):
                state = write_short_inner_state(
                    data_dir,
                    profile_key,
                    f"短期心里话 w={worker_id} s={step}",
                    owner_key=owner_key,
                    turn_id=f"turn-{worker_id}-{step}",
                )
                with rev_lock:
                    revisions.append(state.revision)
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = [pool.submit(worker, i) for i in range(threads)]
        for fut in futs:
            fut.result()

    assert not errors
    expected_total = 1 + threads * writes_per_thread
    assert sorted(revisions) == list(range(2, expected_total + 1))
    head = latest_state(data_dir / "sanae.db", profile_key)
    assert head is not None
    assert head.revision == expected_total


def test_write_impression_multithreaded_monotonic_revisions(tmp_path: Path) -> None:
    data_dir = tmp_path / "sanae-data"
    target = (data_dir, "profile-imp", "owner-imp")
    threads = 8
    writes_per_thread = 6
    barrier = threading.Barrier(threads)
    revisions: list[int] = []
    rev_lock = threading.Lock()
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(writes_per_thread):
                state = write_impression(
                    target,
                    f"【长期底色】稳重\n【阶段近况】并发更新 w={worker_id} s={step}",
                    turn_id=f"turn-{worker_id}-{step}",
                )
                with rev_lock:
                    revisions.append(state.revision)
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = [pool.submit(worker, i) for i in range(threads)]
        for fut in futs:
            fut.result()

    assert not errors
    expected_total = threads * writes_per_thread
    assert sorted(revisions) == list(range(1, expected_total + 1))
    latest = read_impression(target)
    assert latest is not None
    assert latest.revision == expected_total


@contextmanager
def _compete_at_wall_clock(
    db_path: Path,
    insert_sql: str,
    make_params: Callable[[int], tuple[Any, ...]],
    *,
    start_revision: int,
) -> Iterator[list[int]]:
    """Advance SQLite revisions on a second real connection when the outer wall clock is read."""
    host_thread = threading.get_ident()
    competing_changes: list[int] = []
    real_time = time.time

    with (
        ThreadPoolExecutor(max_workers=1) as writer,
        closing(writer.submit(connect_readwrite, db_path, create=False).result(timeout=10.0)) as race_conn,
    ):
        def advance(revision: int) -> int:
            with race_conn:
                return race_conn.execute(insert_sql, make_params(revision)).rowcount

        def clocked_time() -> float:
            if threading.get_ident() == host_thread:
                revision = start_revision + len(competing_changes)
                competing_changes.append(writer.submit(advance, revision).result(timeout=10.0))
            return real_time()

        time.time = clocked_time
        try:
            yield competing_changes
        finally:
            time.time = real_time


@contextmanager
def _compete_at_wal_write_connections(
    db_path: Path,
    insert_sql: str,
    make_params: Callable[[int], tuple[Any, ...]],
    *,
    start_revision: int,
    initial_advance: bool = False,
) -> Iterator[list[int]]:
    """Advance SQLite revisions on a second real connection at every rw connect boundary."""
    write_uri = db_path.resolve().as_uri() + "?mode=rw"
    host_thread = threading.get_ident()
    competing_changes: list[int] = []
    active = True

    with (
        ThreadPoolExecutor(max_workers=1) as writer,
        closing(writer.submit(connect_readwrite, db_path, create=False).result(timeout=10.0)) as race_conn,
    ):
        def advance(revision: int) -> int:
            with race_conn:
                return race_conn.execute(insert_sql, make_params(revision)).rowcount

        def compete() -> None:
            revision = start_revision + len(competing_changes)
            competing_changes.append(writer.submit(advance, revision).result(timeout=10.0))

        def audit(event: str, args: tuple[object, ...]) -> None:
            if (
                active
                and event == "sqlite3.connect"
                and args[0] == write_uri
                and threading.get_ident() == host_thread
            ):
                # Each CAS attempt reads the head before opening this write connection.
                compete()

        if initial_advance:
            compete()  # The writer goes idle after the first commit, until the host requests another.
        sys.addaudithook(audit)  # The hook lives only in this scenario's subprocess.
        try:
            yield competing_changes
        finally:
            active = False


def _run_cas_subprocess(tmp_path: Path, mode: str, extra_arg: str = "") -> None:
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "hermes-home")}
    env.pop("HERMES_HOME_OVERRIDE", None)
    env.pop("HERMES_SAFE_MODE", None)
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), mode, str(tmp_path)]
    if extra_arg:
        cmd.append(extra_arg)
    result = subprocess.run(cmd, env=env, capture_output=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cas_exhaustion_after_backoff_retries_for_all_three_writers(tmp_path: Path) -> None:
    _run_cas_subprocess(tmp_path, "exhaustion")


def _run_cas_exhaustion_scenario(tmp_path: Path) -> None:
    provider, data_dir, owner_key = _init_provider(tmp_path)
    try:
        _, profile_key = provider.inner_state.inner_state_target()
        db_path = data_dir / "sanae.db"

        # 1. write_inner_state exhaustion after 2 backoff retries (3 attempts total)
        base_state = provider.inner_state.write_inner_state("基线心里话", owner_key=owner_key, turn_id="base-1")
        vars_json = json.dumps(dict(base_state.variables))
        inner_sql = (
            "INSERT INTO inner_state_revisions "
            "(profile_key, revision, text, recent_words, recent_words_at, variables, "
            "felt_at, updated_at, processed_at, source, thought_source, turn_id, changed, "
            "owner_key, visibility, last_spark_at) "
            "VALUES (?, ?, '抢先写入', '', NULL, ?, 1.0, 1.0, NULL, 'manual', 'manual', ?, 1, ?, 'private', 0.0)"
        )
        with _compete_at_wal_write_connections(
            db_path,
            inner_sql,
            lambda rev: (profile_key, rev, vars_json, f"race-inner-{rev}", owner_key),
            start_revision=base_state.revision + 1,
        ) as wins_1:
            with pytest.raises(CasConflictError, match="after 2 retries"):
                provider.inner_state.write_inner_state(
                    "被抢先的心里话", owner_key=owner_key, turn_id="lose-1", max_retries=2,
                )
        assert wins_1 == [1, 1, 1]
        head_after_1 = latest_state(db_path, profile_key)
        assert head_after_1 is not None
        assert head_after_1.revision == base_state.revision + 3

        # 2. write_short_inner_state exhaustion after 2 backoff retries (3 attempts total)
        start_rev_2 = head_after_1.revision + 1
        with _compete_at_wal_write_connections(
            db_path,
            inner_sql,
            lambda rev: (profile_key, rev, vars_json, f"race-short-{rev}", owner_key),
            start_revision=start_rev_2,
        ) as wins_2:
            with pytest.raises(CasConflictError, match="after 2 retries"):
                write_short_inner_state(
                    data_dir, profile_key, "被抢先的短期心里话",
                    owner_key=owner_key, turn_id="lose-2", max_retries=2,
                )
        assert wins_2 == [1, 1, 1]
        head_after_2 = latest_state(db_path, profile_key)
        assert head_after_2 is not None
        assert head_after_2.revision == start_rev_2 + 2

        # 3. write_impression exhaustion after 2 backoff retries (3 attempts total)
        imp_target = (data_dir, profile_key, owner_key)
        imp_seed = write_impression(imp_target, "基线印象-1", turn_id="imp-seed")
        assert imp_seed.revision == 1
        # Without a concurrent writer advancing the revision, write_impression commits cleanly.
        imp_base = write_impression(imp_target, "基线印象-2", turn_id="imp-base")
        assert imp_base.revision == 2
        imp_sql = (
            "INSERT INTO impression_revisions "
            "(profile_key, owner_key, revision, text, speaker, evidence, turn_id, updated_at, "
            "recorded_at, baseline, recent, source_context) "
            "VALUES (?, ?, ?, '抢先印象', '', '', ?, 1.0, 1.0, '抢先印象', '', '')"
        )
        with _compete_at_wall_clock(
            db_path,
            imp_sql,
            lambda rev: (profile_key, owner_key, rev, f"race-imp-{rev}"),
            start_revision=imp_base.revision + 1,
        ) as imp_wins:
            with pytest.raises(CasConflictError, match="after 2 retries"):
                write_impression(
                    imp_target, "被抢先的印象", turn_id="lose-imp", max_retries=2,
                )
        assert imp_wins == [1, 1, 1]
        imp_after = read_impression(imp_target)
        assert imp_after is not None
        assert imp_after.revision == imp_base.revision + 3
    finally:
        provider.shutdown()


def test_route_resident_memory_write_distinguishes_cas_conflict_with_real_plugin_llm(
    tmp_path: Path,
) -> None:
    pytest.importorskip("agent.plugin_llm")
    pytest.importorskip("hermes_cli.plugins")
    _run_cas_subprocess(tmp_path, "route")


def _run_route_resident_cas_scenario(tmp_path: Path) -> None:
    PluginLlm = importlib.import_module("agent.plugin_llm").PluginLlm
    plugins_mod = importlib.import_module("hermes_cli.plugins")
    discover_plugins = plugins_mod.discover_plugins
    get_plugin_manager = plugins_mod.get_plugin_manager

    home = tmp_path / "hermes-home"
    data_dir = tmp_path / "sanae-data"
    home.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    install_plugin_copy(home, REPO_ROOT, plugin_name="sanae_mem")

    profile_key = "profile-route"
    owner_key = "owner-route"

    base_state = write_short_inner_state(
        data_dir, profile_key, "基线心里话", owner_key=owner_key, turn_id="base",
    )
    vars_json = json.dumps(dict(base_state.variables))
    db_path = data_dir / "sanae.db"

    responses: list[dict[str, Any]] = [
        {"route": "inner_state", "inner_thought": "心里有点在意这件事", "event_fact": None},
        {"route": "inner_state", "inner_thought": "", "event_fact": None},
    ]
    http_calls: list[dict[str, Any]] = []

    def responder(body: dict[str, Any]) -> dict[str, Any]:
        http_calls.append(body)
        return responses[min(len(http_calls) - 1, len(responses) - 1)]

    with http_json_llm_server(responder, chat_id="chatcmpl-b207b1") as base_url, ExitStack() as stack:
        cfg_yaml = {
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
            },
            "plugins": {
                "enabled": ["sanae_mem"],
                "sanae_mem": {
                    "data_dir": str(data_dir),
                    "semantic_backend": "sqlite",
                    "shadow": False,
                    "memory_identity": {"platform": "discord", "user_id": owner_key},
                },
            },
        }
        write_home_config(home, cfg_yaml)
        os.environ["HERMES_HOME"] = str(home)
        os.environ.pop("HERMES_HOME_OVERRIDE", None)
        os.environ.pop("HERMES_SAFE_MODE", None)

        discover_plugins(force=True)
        manager = get_plugin_manager()
        stack.callback(manager.unload)

        llm = PluginLlm(plugin_id="sanae_mem")
        route_cfg = ResidentRouteConfig(
            data_dir=data_dir,
            profile_key=profile_key,
            owner_key=owner_key,
            session_id="sess-route",
            turn_id="turn-route",
        )
        inner_sql = (
            "INSERT INTO inner_state_revisions "
            "(profile_key, revision, text, recent_words, recent_words_at, variables, "
            "felt_at, updated_at, processed_at, source, thought_source, turn_id, changed, "
            "owner_key, visibility, last_spark_at) "
            "VALUES (?, ?, '抢先写入', '', NULL, ?, 1.0, 1.0, NULL, 'manual', 'manual', ?, 1, ?, 'private', 0.0)"
        )

        # Case A: deterministic dual-connection CAS conflict -> explicit cas_conflict outcome + warning + status.json
        conflict_box = []
        with host_bridge.turn_envelope_scope(
            platform="discord",
            chat_type="dm",
            chat_id="dm-owner",
            sender_id=owner_key,
            session_id="sess-route",
            session_key="sess-route",
            message_id="msg-route",
        ), TestCase().assertLogs(level=logging.WARNING) as captured:
            with _compete_at_wal_write_connections(
                db_path,
                inner_sql,
                lambda rev: (profile_key, rev, vars_json, f"race-r-{rev}", owner_key),
                start_revision=base_state.revision + 1,
            ) as competing_changes:
                conflict_box.append(
                    route_resident_memory_write(
                        llm=llm,
                        texts=["刚才有点在意"],
                        config=route_cfg,
                        write_inner_state_fn=lambda thought: write_short_inner_state(
                            data_dir, profile_key, thought, owner_key=owner_key,
                            turn_id="turn-route", max_retries=2,
                        ),
                    )
                )

        assert len(http_calls) == 1
        assert competing_changes == [1, 1, 1]
        assert len(conflict_box) == 1
        conflict_outcome = conflict_box[0]
        assert conflict_outcome.routed is True
        assert conflict_outcome.route == "inner_state_cas_conflict"
        assert conflict_outcome.error == "cas_conflict"
        assert conflict_outcome.inner_state_text == ""
        assert "cas_conflict" in conflict_outcome.message
        assert any("CAS conflict" in rec.getMessage() for rec in captured.records)

        head = latest_state(db_path, profile_key)
        assert head is not None
        assert head.revision == base_state.revision + 3

        status = read_status_file(data_dir / "status.json")
        comp = status.get("components", {}).get("resident_memory_route", {})
        assert comp.get("ok") is False
        assert comp.get("error") == "cas_conflict"

        # Case B: normal policy skip (inner_thought="") -> distinct route="inner_state", error=""
        with host_bridge.turn_envelope_scope(
            platform="discord",
            chat_type="dm",
            chat_id="dm-owner",
            sender_id=owner_key,
            session_id="sess-route",
            session_key="sess-route",
            message_id="msg-route-2",
        ):
            skip_outcome = route_resident_memory_write(
                llm=llm,
                texts=["普通教条总结"],
                config=route_cfg,
            )

        assert len(http_calls) == 2
        assert skip_outcome.routed is True
        assert skip_outcome.route == "inner_state"
        assert skip_outcome.error == ""
        assert skip_outcome.inner_state_text == ""
        assert "cas_conflict" not in skip_outcome.message


@pytest.mark.parametrize("initial_advance", [False, True], ids=["normal", "pause-after-first-commit"])
def test_host_memory_tool_dispatches_cas_conflict_and_policy_skip(
    tmp_path: Path, initial_advance: bool,
) -> None:
    pytest.importorskip("hermes_cli")
    _run_cas_subprocess(tmp_path, "host", str(int(initial_advance)))


def _run_host_memory_tool_cas_scenario(tmp_path: Path, initial_advance: bool) -> None:
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from tools.memory_tool import load_on_disk_store, memory_tool

    home = tmp_path / "hermes-home"
    data_dir = tmp_path / "sanae-data"
    memories_dir = home / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    memory_md = memories_dir / "MEMORY.md"
    memory_md.write_text("", encoding="utf-8")
    install_plugin_copy(home, REPO_ROOT, plugin_name="sanae_mem")

    profile_key = daily_profile_key(home.resolve())
    owner_user_id = "owner-host-tool"
    owner_key = f"discord:{owner_user_id}"

    base_state = write_short_inner_state(
        data_dir, profile_key, "基线心里话", owner_key=owner_key, turn_id="host-base",
    )
    vars_json = json.dumps(dict(base_state.variables))
    db_path = data_dir / "sanae.db"

    def read_revisions() -> list[sqlite3.Row]:
        with closing(connect_readonly(db_path)) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            return conn.execute("SELECT * FROM inner_state_revisions ORDER BY profile_key, revision").fetchall()

    responses: list[dict[str, Any]] = [
        {"route": "inner_state", "inner_thought": "心里有点在意这件事", "event_fact": None},
        {"route": "inner_state", "inner_thought": "", "event_fact": None},
    ]
    http_calls: list[dict[str, Any]] = []

    def responder(body: dict[str, Any]) -> dict[str, Any]:
        http_calls.append(body)
        return responses[min(len(http_calls) - 1, len(responses) - 1)]

    with http_json_llm_server(responder, chat_id="chatcmpl-b207b1") as base_url, ExitStack() as stack:
        cfg_yaml = {
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
            },
            "memory": {
                "memory_enabled": True,
                "user_profile_enabled": False,
                "memory_char_limit": 8000,
                "user_char_limit": 6000,
            },
            "plugins": {
                "enabled": ["sanae_mem"],
                "sanae_mem": {
                    "data_dir": str(data_dir),
                    "semantic_backend": "sqlite",
                    "shadow": False,
                    "memory_identity": {"platform": "discord", "user_id": owner_user_id},
                },
            },
        }
        write_home_config(home, cfg_yaml)

        discover_plugins(force=True)
        manager = get_plugin_manager()
        stack.callback(manager.unload)

        store = load_on_disk_store()
        inner_sql = (
            "INSERT INTO inner_state_revisions "
            "(profile_key, revision, text, recent_words, recent_words_at, variables, "
            "felt_at, updated_at, processed_at, source, thought_source, turn_id, changed, "
            "owner_key, visibility, last_spark_at) "
            "VALUES (?, ?, '抢先写入', '', NULL, ?, 1.0, 1.0, NULL, 'manual', 'manual', ?, 1, ?, 'private', 0.0)"
        )

        # Case A: real host memory_tool -> ViveMindMemoryMutationGate -> default 32-retry CAS exhaustion
        revisions_before = read_revisions()
        first_race_revision = base_state.revision + 1
        conflict_results: list[str] = []
        with host_bridge.turn_envelope_scope(
            platform="discord",
            chat_type="dm",
            chat_id="dm-owner",
            sender_id=owner_user_id,
            session_id="sess-host-tool",
            session_key="agent:main:discord:dm:dm-owner",
            message_id="msg-host-1",
        ), TestCase().assertLogs(level=logging.WARNING) as captured:
            with _compete_at_wal_write_connections(
                db_path,
                inner_sql,
                lambda rev: (profile_key, rev, vars_json, f"race-host-{rev}", owner_key),
                start_revision=first_race_revision,
                initial_advance=initial_advance,
            ) as competing_changes:
                conflict_results.append(
                    memory_tool(
                        action="add",
                        target="memory",
                        content="刚才有点在意这件事",
                        store=store,
                    )
                )

        assert len(http_calls) == 1
        assert len(conflict_results) == 1
        conflict_payload = json.loads(conflict_results[0])
        assert conflict_payload["success"] is False
        assert "cas_conflict" in conflict_payload["error"]
        assert "并发写入冲突" in conflict_payload["error"]
        assert "本次未能落盘" in conflict_payload["error"]
        assert memory_md.read_text(encoding="utf-8") == ""
        assert any(
            "CAS conflict" in rec.getMessage()
            and isinstance(rec.args, tuple)
            and len(rec.args) == 1
            and isinstance(rec.args[0], CasConflictError)
            and str(rec.args[0]).endswith("after 32 retries")
            for rec in captured.records
        )

        status_after_conflict = read_status_file(data_dir / "status.json")
        comp_conflict = status_after_conflict.get("components", {}).get("resident_memory_route", {})
        assert comp_conflict.get("ok") is False
        assert comp_conflict.get("error") == "cas_conflict"

        # One committed competing row per host CAS connection, plus the optional advance.
        expected_wins = 33 + int(initial_advance)
        assert competing_changes == [1] * expected_wins
        revisions_after_conflict = read_revisions()
        assert revisions_after_conflict[:len(revisions_before)] == revisions_before
        assert [
            (row["profile_key"], row["revision"], row["turn_id"], row["text"], row["owner_key"])
            for row in revisions_after_conflict[len(revisions_before):]
        ] == [
            (profile_key, rev, f"race-host-{rev}", "抢先写入", owner_key)
            for rev in range(first_race_revision, first_race_revision + expected_wins)
        ]

        # Case B: real host memory_tool -> normal policy skip (inner_thought="")
        with host_bridge.turn_envelope_scope(
            platform="discord",
            chat_type="dm",
            chat_id="dm-owner",
            sender_id=owner_user_id,
            session_id="sess-host-tool",
            session_key="agent:main:discord:dm:dm-owner",
            message_id="msg-host-2",
        ):
            skip_raw = memory_tool(
                action="add",
                target="memory",
                content="普通教条总结",
                store=store,
            )

        assert len(http_calls) == 2
        assert [call["model"] for call in http_calls] == ["test-flash-model", "test-flash-model"]
        skip_payload = json.loads(skip_raw)
        assert skip_payload["success"] is False
        assert "不写入长期常驻文件" in skip_payload["error"]
        assert "cas_conflict" not in skip_payload["error"]
        assert "并发写入冲突" not in skip_payload["error"]
        assert memory_md.read_text(encoding="utf-8") == ""
        assert read_revisions() == revisions_after_conflict

        status_after_skip = read_status_file(data_dir / "status.json")
        comp_skip = status_after_skip.get("components", {}).get("resident_memory_route", {})
        assert comp_skip.get("ok") is True


if __name__ == "__main__":
    _mode = sys.argv[1]
    if _mode == "exhaustion":
        _run_cas_exhaustion_scenario(Path(sys.argv[2]))
    elif _mode == "route":
        _run_route_resident_cas_scenario(Path(sys.argv[2]))
    elif _mode == "host":
        _run_host_memory_tool_cas_scenario(Path(sys.argv[2]), sys.argv[3] == "1")
    else:
        raise ValueError(f"Unknown CAS scenario mode: {_mode!r}")
