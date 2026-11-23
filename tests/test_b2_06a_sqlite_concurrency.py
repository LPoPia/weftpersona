"""B2-06A real-file SQLite concurrency, migration & facade invariant tests."""

from __future__ import annotations

import importlib
import json
import shutil
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
import types
from typing import Any, Callable, Iterator, cast

import pytest
import yaml

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))
PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from _harness import http_json_llm_server as _http_llm_server  # noqa: E402
from sanae_mem import SanaeMemProvider  # noqa: E402
def _associate(
    provider: SanaeMemProvider,
    query: str,
    *,
    owner_key: str,
    conversation_history: list[Any] | None = None,
    purpose: str = "chat",
    session_id: str = "",
    gap_continuation: bool = False,
    hot_path: bool = False,
) -> dict:
    AssociationTurn = importlib.import_module("sanae_mem.cognitive.association").AssociationTurn
    with provider.association_state.lock:
        rev = provider.association_state.revision
    return provider.association.associate_context(
        query,
        conversation_history=conversation_history,
        purpose=purpose,
        session_id=session_id,
        gap_continuation=gap_continuation,
        hot_path=hot_path,
        turn=AssociationTurn(
            owner_key=owner_key,
            daily_owner_key="",
            daily_profile_key="",
            daily_owner_platform="telegram",
            agent_context="primary",
            inject_max_chars=400,
            llm=provider.runtime_llm(),
        ),
        revision=rev,
    )


from sanae_mem.cognitive.impression import (  # noqa: E402
    apply_impression,
    read_impression,
    write_impression,
)
from sanae_mem.storage.database import open_database  # noqa: E402
from sanae_mem.storage.schema import CURRENT_SCHEMA_VERSION  # noqa: E402
from sanae_mem.storage.sqlite_facade import (  # noqa: E402
    SqliteFacade,
    connect_readonly,
    connect_readwrite,
    open_sqlite_connection,
)


@contextmanager
def _private_turn(
    *,
    session_id: str = "sess-b206a",
    turn_id: str = "turn-b206a",
    platform: str = "telegram",
    sender_id: str = "1001",
    message_id: str = "msg-1",
) -> Iterator[None]:
    from hermes_host_contracts import TurnEnvelope, turn_envelope_scope

    envelope = TurnEnvelope(
        platform=platform,
        chat_type="dm",
        chat_id=sender_id,
        sender_id=sender_id,
        sender_id_alt="",
        session_id=session_id,
        session_key=f"agent:main:{platform}:dm:{sender_id}",
        message_id=message_id,
        turn_id=turn_id,
        occurred_at="2026-11-20T18:00:00+08:00",
        is_bot=False,
        is_messaging_surface=True,
        write_origin="foreground",
    )
    with turn_envelope_scope(envelope):
        yield


def _open_host_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stack: ExitStack,
    *,
    session_id: str = "sess-b206a",
    base_url: str = "",
) -> SanaeMemProvider:
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    import plugins.memory as host_memory

    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        PLUGIN_ROOT / "sanae_mem",
        plugins_dir / "sanae_mem",
        ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"),
    )
    plugin_cfg: dict[str, Any] = {
        "data_dir": str(home / "sanae-mem"),
        "semantic_backend": "sqlite",
        "shadow": False,
        "memory_identity": {"platform": "telegram", "user_id": "1001"},
    }
    cfg: dict[str, Any] = {
        "memory": {"provider": "sanae_mem"},
        "plugins": {
            "enabled": ["sanae_mem"],
            "sanae_mem": plugin_cfg,
        },
    }
    if base_url:
        cfg["model"] = {
            "default": "test-main-model",
            "provider": "custom",
            "base_url": base_url,
            "api_key": "dummy-key",
        }
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
    (home / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)

    discover_plugins()
    manager = get_plugin_manager()
    loaded = host_memory.load_memory_provider("sanae_mem", register_skills=False)
    assert loaded is not None
    provider = cast(SanaeMemProvider, loaded)
    stack.callback(manager.unload)
    stack.callback(provider.shutdown)
    stack.enter_context(_private_turn(session_id=session_id))
    provider.initialize(
        session_id,
        hermes_home=str(home),
        platform="telegram",
        agent_context="primary",
    )
    return provider


def _count(facade: SqliteFacade, sql: str) -> int:
    row = facade.fetchone(sql)
    assert row is not None
    return int(row[0])


def test_connection_helpers_apply_standard_pragmas_and_ro_guards(tmp_path: Path) -> None:
    db_path = tmp_path / "pragmas.db"
    with closing(open_sqlite_connection(db_path, mode="rwc")) as rw:
        assert int(rw.execute("PRAGMA busy_timeout").fetchone()[0]) == 10000
        assert str(rw.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        assert int(rw.execute("PRAGMA foreign_keys").fetchone()[0]) == 1
        rw.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT NOT NULL)")
        rw.execute("INSERT INTO items (val) VALUES ('ok')")
        rw.commit()

    with closing(connect_readonly(db_path)) as ro:
        assert int(ro.execute("PRAGMA busy_timeout").fetchone()[0]) == 10000
        assert int(ro.execute("PRAGMA foreign_keys").fetchone()[0]) == 1
        assert int(ro.execute("PRAGMA query_only").fetchone()[0]) == 1
        assert ro.execute("SELECT val FROM items").fetchone()[0] == "ok"
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("INSERT INTO items (val) VALUES ('blocked')")

    with closing(connect_readwrite(db_path, create=False)) as rw2:
        facade = SqliteFacade(conn=rw2, db_path=db_path)
        with facade.thread_readonly():
            row = facade.fetchone("SELECT val FROM items")
            assert row is not None and row[0] == "ok"
            with pytest.raises(RuntimeError, match="read-only"):
                with facade.transaction():
                    pass


def test_top_level_commit_failure_rolls_back_and_leaves_clean_connection(tmp_path: Path) -> None:
    db_path = tmp_path / "deferred_fk.db"
    with closing(open_database(db_path)) as conn:
        facade = SqliteFacade(conn=conn, db_path=db_path)
        facade.execute("CREATE TABLE parents (id INTEGER PRIMARY KEY)")
        facade.execute(
            "CREATE TABLE children ("
            "id INTEGER PRIMARY KEY, "
            "parent_id INTEGER NOT NULL REFERENCES parents(id) DEFERRABLE INITIALLY DEFERRED)"
        )

        # Deferred foreign key fails only at COMMIT time.
        with pytest.raises(sqlite3.IntegrityError):
            with facade.transaction() as tx:
                tx.execute("INSERT INTO children (id, parent_id) VALUES (1, 999)")

        assert facade.tx_depth == 0
        assert not conn.in_transaction

        # Subsequent top-level write must run normally and be visible to an independent RO connection.
        facade.execute("INSERT INTO parents (id) VALUES (10)")
        facade.execute("INSERT INTO children (id, parent_id) VALUES (2, 10)")
        with closing(connect_readonly(db_path)) as ro:
            assert [tuple(r) for r in ro.execute("SELECT id FROM parents").fetchall()] == [(10,)]
            assert [tuple(r) for r in ro.execute("SELECT id, parent_id FROM children").fetchall()] == [(2, 10)]


def test_nested_transactions_and_concurrent_readers_preserve_rollback_atomicity(tmp_path: Path) -> None:
    db_path = tmp_path / "nested.db"
    with closing(open_database(db_path)) as conn:
        facade = SqliteFacade(conn=conn, db_path=db_path)
        facade.execute("CREATE TABLE ledger (id INTEGER PRIMARY KEY AUTOINCREMENT, tag TEXT NOT NULL, amount INTEGER NOT NULL)")

        # 1. Outer rollback must undo inner nested savepoint writes (no premature inner commit).
        with pytest.raises(RuntimeError, match="abort outer"):
            with facade.transaction() as outer_conn:
                outer_conn.execute("INSERT INTO ledger (tag, amount) VALUES ('outer-1', 10)")
                with facade.transaction() as inner_conn:
                    inner_conn.execute("INSERT INTO ledger (tag, amount) VALUES ('inner-1', 20)")
                raise RuntimeError("abort outer")

        assert _count(facade, "SELECT COUNT(*) FROM ledger") == 0
        assert facade.tx_depth == 0
        assert not conn.in_transaction

        # 2. Inner rollback isolated by SAVEPOINT leaves earlier outer writes intact.
        with facade.transaction() as outer_conn:
            outer_conn.execute("INSERT INTO ledger (tag, amount) VALUES ('outer-keep', 50)")
            with pytest.raises(ValueError, match="abort inner"):
                with facade.transaction() as inner_conn:
                    inner_conn.execute("INSERT INTO ledger (tag, amount) VALUES ('inner-drop', 99)")
                    raise ValueError("abort inner")

        rows = facade.fetchall("SELECT tag, amount FROM ledger ORDER BY id")
        assert [(r["tag"], r["amount"]) for r in rows] == [("outer-keep", 50)]
        assert facade.tx_depth == 0
        assert not conn.in_transaction

        # 3. Multi-threaded nested commits/rollbacks + concurrent RO readers.
        errors: list[BaseException] = []
        stop_readers = threading.Event()
        committed_batches = 0
        batch_lock = threading.Lock()

        def writer(worker_id: int) -> None:
            nonlocal committed_batches
            try:
                for step in range(20):
                    should_abort_outer = (step % 2 == 1)
                    if should_abort_outer:
                        try:
                            with facade.transaction() as c_out:
                                c_out.execute(
                                    "INSERT INTO ledger (tag, amount) VALUES (?, ?)",
                                    (f"abort-{worker_id}-{step}-a", 100),
                                )
                                with facade.transaction() as c_in:
                                    c_in.execute(
                                        "INSERT INTO ledger (tag, amount) VALUES (?, ?)",
                                        (f"abort-{worker_id}-{step}-b", 200),
                                    )
                                raise RuntimeError("rollback whole batch")
                        except RuntimeError:
                            pass
                    else:
                        with facade.transaction() as c_out:
                            c_out.execute(
                                "INSERT INTO ledger (tag, amount) VALUES (?, ?)",
                                (f"ok-{worker_id}-{step}-a", 1),
                            )
                            with facade.transaction() as c_in:
                                c_in.execute(
                                    "INSERT INTO ledger (tag, amount) VALUES (?, ?)",
                                    (f"ok-{worker_id}-{step}-b", 2),
                                )
                        with batch_lock:
                            committed_batches += 1
            except BaseException as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                while not stop_readers.is_set():
                    aborted_rows = _count(
                        facade, "SELECT COUNT(*) FROM ledger WHERE tag LIKE 'abort-%'"
                    )
                    assert aborted_rows == 0
                    with closing(connect_readonly(db_path)) as ro:
                        ro_aborted = ro.execute(
                            "SELECT COUNT(*) FROM ledger WHERE tag LIKE 'abort-%'"
                        ).fetchone()[0]
                        assert ro_aborted == 0
                        total = ro.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
                        assert (total - 1) % 2 == 0
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=10) as pool:
            reader_futures = [pool.submit(reader) for _ in range(2)]
            writer_futures = [pool.submit(writer, i) for i in range(8)]
            for fut in writer_futures:
                fut.result()
            stop_readers.set()
            for fut in reader_futures:
                fut.result()

        assert not errors
        assert committed_batches == 8 * 10
        assert _count(facade, "SELECT COUNT(*) FROM ledger WHERE tag LIKE 'abort-%'") == 0
        assert _count(facade, "SELECT COUNT(*) FROM ledger") == 1 + committed_batches * 2
        assert facade.tx_depth == 0


def test_association_timeout_fallback_and_worker_concurrent_sqlite_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    pytest.importorskip("hermes_cli.plugins")

    worker_in_http = threading.Event()
    release_worker = threading.Event()
    http_calls: list[str] = []

    def llm_responder(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages") or []
        sys_prompt = str(messages[0].get("content") or "") if messages else ""
        http_calls.append(sys_prompt[:40])
        if "needs_history" in sys_prompt:
            worker_in_http.set()
            release_worker.wait(timeout=5.0)
            return {"needs_history": True, "query": "河边鹅卵石"}
        return {"valid": True, "items": [{"id": 1, "relationship": "related", "state": "history"}]}

    with _http_llm_server(llm_responder) as base_url, ExitStack() as stack:
        provider = _open_host_provider(tmp_path, monkeypatch, stack, base_url=base_url)
        added = json.loads(
            provider.handle_tool_call(
                "sanae_mem_add",
                {"content": "上次去河边捡到一块白色鹅卵石，说好下次还去河边。"},
            )
        )
        assert added["ok"] is True
        assert added["result"]["stored"] is True
        owner_row = provider.sqlite.fetchone("SELECT owner_key FROM events ORDER BY id DESC LIMIT 1")
        assert owner_row is not None
        owner_key = str(owner_row["owner_key"])

        lock_acquired = threading.Event()
        lock_errors: list[BaseException] = []

        def long_write_lock_holder() -> None:
            try:
                with provider.sqlite.transaction():
                    lock_acquired.set()
                    time.sleep(4.2)
            except BaseException as exc:
                lock_errors.append(exc)

        holder = threading.Thread(target=long_write_lock_holder)
        holder.start()
        assert lock_acquired.wait(timeout=2.0)

        # End-to-end budget check: even while another thread holds the SQLite write lock for 4.2s,
        # hot-path association stays within the 3.0s budget via read-only fallback without blocking on 4.2s.
        t_start = time.monotonic()
        res = _associate(
            provider,
            "河边捡白色鹅卵石",
            owner_key=owner_key,
            conversation_history=[{"role": "user", "content": "河边捡白色鹅卵石"}],
            purpose="chat",
            session_id="sess-b206a",
            gap_continuation=True,
            hot_path=True,
        )
        hot_elapsed = time.monotonic() - t_start
        assert worker_in_http.is_set()
        assert res["status"] == "ok"
        assert "鹅卵石" in res["context"]
        assert hot_elapsed < 3.0, f"hot_elapsed={hot_elapsed:.3f}s exceeded 3.0s budget"

        # A second call while the write lock is still held and worker is still in flight
        # must immediately take the read-only fallback without blocking on either lock.
        t_second = time.monotonic()
        second_res = _associate(
            provider,
            "河边捡白色鹅卵石",
            owner_key=owner_key,
            purpose="chat",
            session_id="sess-b206a",
            gap_continuation=True,
            hot_path=True,
        )
        second_elapsed = time.monotonic() - t_second
        assert second_res["status"] == "ok"
        assert second_elapsed < 0.5, f"second_elapsed={second_elapsed:.3f}s blocked on lock"

        # Wake the timed-out HTTP call so the background worker proceeds to real SQLite recall
        # concurrently with the write lock holder finishing and subsequent writers/readers.
        release_worker.set()
        holder.join(timeout=6.0)
        assert not holder.is_alive()
        assert not lock_errors

        distinct_notes = (
            ("alpha-钢琴曲谱需要重新抄写。", "alpha2-图书馆借阅的建筑史下周到期。"),
            ("beta-望远镜镜片已经擦拭干净。", "beta2-阳台的迷迭香今天抽了新芽。"),
            ("gamma-烘焙咖啡豆闻起来很香。", "gamma2-修理好的胶片相机装上了新电池。"),
            ("delta-银杏植物标本夹在书页里。", "delta2-周末准备熬一锅番茄牛腩汤。"),
        )
        errors: list[BaseException] = []

        def concurrent_reader_and_writer(idx: int) -> None:
            try:
                for round_idx, note in enumerate(distinct_notes[idx]):
                    with _private_turn(turn_id=f"conc-{idx}-{round_idx}"):
                        read_res = _associate(
                            provider,
                            "河边捡白色鹅卵石",
                            owner_key=owner_key,
                            purpose="chat",
                            session_id="sess-b206a",
                            gap_continuation=True,
                            hot_path=True,
                        )
                        assert read_res["status"] in {"ok", "empty", "unavailable"}
                        add_res = json.loads(
                            provider.handle_tool_call(
                                "sanae_mem_add",
                                {"content": note},
                            )
                        )
                        assert add_res["ok"] is True, f"idx={idx}: {add_res}"
                        assert add_res["result"]["stored"] is True, f"idx={idx}: {add_res}"
            except BaseException as exc:
                errors.append(RuntimeError(f"idx={idx}: {type(exc).__name__}: {exc}"))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(concurrent_reader_and_writer, i) for i in range(4)]
            for fut in futs:
                fut.result()

        assert not errors
        assert len(http_calls) >= 1
        assert _count(provider.sqlite, "SELECT COUNT(*) FROM events") == 9
        assert "timed-out association worker failed" not in caplog.text


@pytest.mark.parametrize("expire_by", ["prefetch", "session_switch"])
def test_expired_association_hook_does_not_deliver_or_mark_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expire_by: str,
) -> None:
    pytest.importorskip("hermes_cli.plugins")

    session_id = "sess-b206a"
    query = "河边捡白色鹅卵石"
    planning_started = threading.Event()
    release_plan = threading.Event()
    planned_queries: list[str] = []

    def llm_responder(body: dict[str, Any]) -> dict[str, Any]:
        messages = body["messages"]
        if "needs_history" not in messages[0]["content"]:
            return {}
        planned_queries.append(json.loads(messages[-1]["content"])["current_user"])
        planning_started.set()
        release_plan.wait(timeout=5.0)
        return {"needs_history": True, "query": query}

    with _http_llm_server(llm_responder) as base_url, ExitStack() as stack:
        provider = _open_host_provider(tmp_path, monkeypatch, stack, base_url=base_url)
        added = json.loads(provider.handle_tool_call(
            "sanae_mem_add", {"content": "上次去河边捡到一块白色鹅卵石，说好下次还去河边。"},
        ))
        assert added["ok"] and added["result"]["stored"]

        def run_hook(sid: str, turn_id: str) -> dict:
            with _private_turn(session_id=sid, turn_id=turn_id):
                return provider.handle_pre_llm_call(
                    user_message=query, session_id=sid, turn_id=turn_id, platform="telegram",
                )

        with ThreadPoolExecutor(max_workers=1) as pool:
            old_hook = pool.submit(run_hook, session_id, "old-turn")
            try:
                assert planning_started.wait(timeout=2.0)
                if expire_by == "prefetch":
                    assert provider.prefetch(query, session_id=session_id) == ""
                else:
                    session_id = "resumed-session"
                    provider.on_session_switch(session_id)
            finally:
                release_plan.set()
            assert "鹅卵石" not in old_hook.result(timeout=4.0).get("context", "")

        assert planned_queries == [query]
        assert provider.recall_status() is None
        assert _count(provider.sqlite, "SELECT COUNT(*) FROM events WHERE asked=1") == 0

        if expire_by == "prefetch":
            session_id = "resumed-session"
            provider.on_session_switch(session_id)
        next_hook = run_hook(session_id, "new-turn")
        assert "鹅卵石" in next_hook["context"]
        with _private_turn(session_id=session_id, turn_id="new-turn"):
            assert provider.prefetch(query, session_id=session_id) == ""
        assert provider.recall_status() is not None
        assert _count(provider.sqlite, "SELECT COUNT(*) FROM events WHERE asked=1") == 1


def test_association_hot_path_500_memories_stays_under_3s_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("hermes_cli.plugins")

    worker_in_http = threading.Event()
    release_worker = threading.Event()

    def llm_responder(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages") or []
        sys_prompt = str(messages[0].get("content") or "") if messages else ""
        if "needs_history" in sys_prompt:
            worker_in_http.set()
            release_worker.wait(timeout=5.0)
            return {"needs_history": True, "query": "河边鹅卵石"}
        return {"valid": True, "items": [{"id": 1, "relationship": "related", "state": "history"}]}

    with _http_llm_server(llm_responder) as base_url, ExitStack() as stack:
        provider = _open_host_provider(tmp_path, monkeypatch, stack, base_url=base_url)
        seed_added = json.loads(
            provider.handle_tool_call(
                "sanae_mem_add",
                {"content": "合成初始化种子：今天整理了一下桌面。"},
            )
        )
        assert seed_added["ok"] is True
        assert seed_added["result"]["stored"] is True

        seed_row = provider.sqlite.fetchone(
            "SELECT ts, keywords, importance, confidence, topic, ttl_expire, created_at, vec, "
            "session_id, owner_key, source_context, experience_subject, fact_subject, "
            "involved_owner_keys, read_scope, last_active_at, half_life_days, curve_ttl "
            "FROM events ORDER BY id ASC LIMIT 1"
        )
        assert seed_row is not None
        filler_raw = "合成背景样本段落：" + ("今天去公园散步顺便整理笔记。" * 140)
        filler_summary = filler_raw[:150]
        with provider.sqlite.transaction() as conn:
            conn.executemany(
                """INSERT INTO events
                   (ts, raw, summary, summary_source, keywords, importance, confidence, topic,
                    ttl_expire, created_at, vec, session_id, a_conflict, affect, owner_key,
                    source_snapshot, source_snapshot_sha256, source_context,
                    experience_subject, fact_subject, involved_owner_keys, read_scope,
                    last_active_at, half_life_days, curve_ttl, original_fingerprint)
                   VALUES (?, ?, ?, 'original', ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, '', '', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        seed_row["ts"],
                        f"{filler_raw}-{i}",
                        f"{filler_summary}-{i}",
                        f"背景 散步 笔记 样本{i}",
                        0.45,
                        float(seed_row["confidence"]),
                        "日常",
                        seed_row["ttl_expire"],
                        seed_row["created_at"],
                        seed_row["vec"],
                        seed_row["session_id"],
                        seed_row["owner_key"],
                        seed_row["source_context"],
                        seed_row["experience_subject"],
                        seed_row["fact_subject"],
                        seed_row["involved_owner_keys"],
                        seed_row["read_scope"],
                        seed_row["last_active_at"],
                        float(seed_row["half_life_days"]),
                        seed_row["curve_ttl"],
                        f"fp-synthetic-{i}",
                    )
                    for i in range(499)
                ],
            )
        added = json.loads(
            provider.handle_tool_call(
                "sanae_mem_add",
                {"content": "上次去河边捡到一块白色鹅卵石，说好下次还去河边。"},
            )
        )
        assert added["ok"] is True
        assert added["result"]["stored"] is True
        assert _count(provider.sqlite, "SELECT COUNT(*) FROM events") == 501
        owner_key = str(seed_row["owner_key"])

        lock_acquired = threading.Event()
        lock_errors: list[BaseException] = []

        def long_write_lock_holder() -> None:
            try:
                with provider.sqlite.transaction():
                    lock_acquired.set()
                    time.sleep(4.2)
            except BaseException as exc:
                lock_errors.append(exc)

        holder = threading.Thread(target=long_write_lock_holder)
        holder.start()
        assert lock_acquired.wait(timeout=2.0)

        try:
            t_start = time.monotonic()
            res = _associate(
                provider,
                "河边捡白色鹅卵石",
                owner_key=owner_key,
                conversation_history=[{"role": "user", "content": "河边捡白色鹅卵石"}],
                purpose="chat",
                session_id="sess-b206a",
                gap_continuation=True,
                hot_path=True,
            )
            elapsed = time.monotonic() - t_start
            assert worker_in_http.is_set()
            assert res["status"] == "ok"
            assert "鹅卵石" in res["context"]
            assert elapsed < 3.0, f"500-memory hot_path elapsed={elapsed:.3f}s exceeded 3.0s budget"
        finally:
            release_worker.set()
            holder.join(timeout=6.0)
        assert not holder.is_alive()
        assert not lock_errors


def test_association_hot_path_without_llm_bypasses_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("hermes_host_contracts")
    home = tmp_path / "no_llm_home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)

    provider = SanaeMemProvider(
        {
            "data_dir": str(home / "sanae-mem"),
            "semantic_backend": "sqlite",
            "shadow": False,
            "memory_identity": {"platform": "telegram", "user_id": "1001"},
        }
    )
    with ExitStack() as stack:
        stack.callback(provider.shutdown)
        stack.enter_context(_private_turn(session_id="sess-no-llm"))
        provider.initialize(
            "sess-no-llm",
            hermes_home=str(home),
            platform="telegram",
            agent_context="primary",
        )
        added = json.loads(
            provider.handle_tool_call(
                "sanae_mem_add",
                {"content": "上次去河边捡到一块白色鹅卵石，说好下次还去河边。"},
            )
        )
        assert added["ok"] is True
        assert added["result"]["stored"] is True
        owner_row = provider.sqlite.fetchone("SELECT owner_key FROM events ORDER BY id DESC LIMIT 1")
        assert owner_row is not None
        owner_key = str(owner_row["owner_key"])

        lock_acquired = threading.Event()
        lock_errors: list[BaseException] = []

        def long_write_lock_holder() -> None:
            try:
                with provider.sqlite.transaction():
                    lock_acquired.set()
                    time.sleep(4.2)
            except BaseException as exc:
                lock_errors.append(exc)

        holder = threading.Thread(target=long_write_lock_holder)
        holder.start()
        assert lock_acquired.wait(timeout=2.0)

        t_start = time.monotonic()
        res = _associate(
            provider,
            "河边捡白色鹅卵石",
            owner_key=owner_key,
            purpose="chat",
            session_id="sess-no-llm",
            gap_continuation=True,
            hot_path=True,
        )
        elapsed = time.monotonic() - t_start
        assert res["status"] == "ok"
        assert "鹅卵石" in res["context"]
        assert elapsed < 1.0, f"no-LLM hot_path blocked on write lock ({elapsed:.3f}s)"

        holder.join(timeout=6.0)
        assert not holder.is_alive()
        assert not lock_errors


def test_impression_multithreaded_concurrent_writes_without_operational_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "sanae-mem"
    target = (data_dir, "profile-a", "owner-a")
    errors: list[BaseException] = []
    written_revisions: list[int] = []
    rev_lock = threading.Lock()
    stop_readers = threading.Event()
    threads = 10
    writes_per_thread = 6

    def writer(worker_id: int) -> None:
        try:
            for step in range(writes_per_thread):
                state = write_impression(
                    target,
                    f"【长期底色】稳重可靠\n【阶段近况】并发更新 w={worker_id} s={step}",
                    turn_id=f"turn-{worker_id}-{step}",
                )
                with rev_lock:
                    written_revisions.append(state.revision)
        except BaseException as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            prev_rev = 0
            while not stop_readers.is_set():
                cur = read_impression(target)
                if cur is not None:
                    assert cur.revision >= prev_rev
                    prev_rev = cur.revision
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=12) as pool:
        reader_futs = [pool.submit(reader) for _ in range(2)]
        writer_futs = [pool.submit(writer, i) for i in range(threads)]
        for fut in writer_futs:
            fut.result()
        stop_readers.set()
        for fut in reader_futs:
            fut.result()

    assert not errors
    expected_total = threads * writes_per_thread
    assert sorted(written_revisions) == list(range(1, expected_total + 1))

    latest = read_impression(target)
    assert latest is not None
    assert latest.revision == expected_total
    with closing(connect_readonly(data_dir / "sanae.db")) as conn:
        revs = [
            int(row[0])
            for row in conn.execute(
                "SELECT revision FROM impression_revisions "
                "WHERE profile_key=? AND owner_key=? ORDER BY revision ASC",
                ("profile-a", "owner-a"),
            ).fetchall()
        ]
        assert revs == list(range(1, expected_total + 1))


def test_apply_impression_upgrades_v4_legacy_db_with_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))

    data_dir = home / "sanae-mem"
    db_path = data_dir / "sanae.db"
    with closing(open_database(db_path)) as seed_conn:
        # Downgrade to v4 shape (impression_revisions & commitments without source_context).
        seed_conn.executescript(
            """
            CREATE TABLE impression_revisions_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_key TEXT NOT NULL,
                owner_key TEXT NOT NULL,
                revision INTEGER NOT NULL,
                text TEXT NOT NULL,
                speaker TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                turn_id TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL,
                recorded_at REAL NOT NULL,
                baseline TEXT NOT NULL DEFAULT '',
                recent TEXT NOT NULL DEFAULT '',
                UNIQUE(profile_key, owner_key, revision)
            );
            INSERT INTO impression_revisions_v4
                (profile_key, owner_key, revision, text, speaker, evidence, turn_id, updated_at, recorded_at)
            VALUES ('profile-v4', 'owner-v4', 1, '【长期底色】旧版印象', 'user', '旧原话', 't-old', 100.0, 100.0);
            DROP TABLE impression_revisions;
            ALTER TABLE impression_revisions_v4 RENAME TO impression_revisions;

            CREATE TABLE commitments_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_key TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                masked_text TEXT NOT NULL DEFAULT '',
                source_quote TEXT NOT NULL DEFAULT '',
                turn_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                fulfilled_at TEXT,
                cancelled_at TEXT,
                resolution_evidence TEXT NOT NULL DEFAULT ''
            );
            DROP TABLE commitments;
            ALTER TABLE commitments_v4 RENAME TO commitments;
            PRAGMA user_version=4;
            """
        )

    target = (data_dir, "profile-v4", "owner-v4")
    base = read_impression(target)
    assert base is not None and base.revision == 1

    ok = apply_impression(
        target,
        base,
        {
            "speaker": "user",
            "quote": "这周开始每天早起跑步",
            "text": "【长期底色】旧版印象\n【阶段近况】这周开始每天早起跑步",
        },
        turn_id="t-v5",
        now=200.0,
        user="我这周开始每天早起跑步，感觉精神好多了。",
        assistant="听起来不错呀。",
        source_context='{"kind":"maintainer","platform":"telegram"}',
    )
    assert ok is True

    backup_path = db_path.with_name("sanae.db.bak.v4.sqlite")
    assert backup_path.is_file()
    with closing(connect_readonly(backup_path)) as bak_conn:
        assert int(bak_conn.execute("PRAGMA user_version").fetchone()[0]) == 4

    with closing(connect_readonly(db_path)) as cur_conn:
        assert int(cur_conn.execute("PRAGMA user_version").fetchone()[0]) == CURRENT_SCHEMA_VERSION
    upgraded = read_impression(target)
    assert upgraded is not None
    assert upgraded.revision == 2
    assert "早起跑步" in upgraded.recent
    assert upgraded.source_context == '{"kind":"maintainer","platform":"telegram"}'


def test_session_reply_watermarks_concurrent_with_locked_write_transactions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    pytest.importorskip("hermes_cli.plugins")
    from hermes_host_contracts import DeliveryReceipt, emit_delivery_receipt

    seq_lock = threading.Lock()
    seq = 0

    def llm_responder(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal seq
        messages = body.get("messages") or []
        sys_prompt = str(messages[0].get("content") or "") if messages else ""
        if "new_commitments" in sys_prompt or "待办约定与承诺" in sys_prompt:
            return {"new_commitments": [], "resolutions": []}
        if "DAILY_FACT" in sys_prompt or "OPEN_THREAD" in sys_prompt:
            return {"items": []}
        with seq_lock:
            seq += 1
            cur = seq
        return {
            "sentences": [f"记着今天第{cur}次和他聊了一会儿，心里踏实。"],
            "variables": {"tension": 0.1, "closeness": 0.7, "energy": 0.3},
            "shared_sentences": None,
        }

    with _http_llm_server(llm_responder) as base_url, ExitStack() as stack:
        provider = _open_host_provider(tmp_path, monkeypatch, stack, base_url=base_url)
        provider.sqlite.execute(
            "CREATE TABLE IF NOT EXISTS tx_probe (id INTEGER PRIMARY KEY AUTOINCREMENT, marker TEXT NOT NULL)"
        )

        errors: list[BaseException] = []
        committed_probes = 0
        probe_lock = threading.Lock()
        deliveries_done = threading.Event()
        barrier = threading.Barrier(6)

        def tx_worker(worker_id: int) -> None:
            nonlocal committed_probes
            try:
                barrier.wait(timeout=5.0)
                step = 0
                while not deliveries_done.is_set() or step < 4:
                    if step % 2 == 0:
                        try:
                            with provider.sqlite.transaction() as conn:
                                conn.execute(
                                    "INSERT INTO tx_probe (marker) VALUES (?)",
                                    (f"rolled-back-{worker_id}-{step}",),
                                )
                                time.sleep(0.01)
                                raise RuntimeError("intentional rollback")
                        except RuntimeError:
                            pass
                    else:
                        with provider.sqlite.transaction() as conn:
                            conn.execute(
                                "INSERT INTO tx_probe (marker) VALUES (?)",
                                (f"committed-{worker_id}-{step}",),
                            )
                            time.sleep(0.01)
                        with probe_lock:
                            committed_probes += 1
                    step += 1
                    time.sleep(0.04)
            except BaseException as exc:
                errors.append(exc)

        def delivery_worker(worker_id: int) -> None:
            try:
                barrier.wait(timeout=5.0)
                for step in range(3):
                    turn_id = f"wm-turn-{worker_id}-{step}"
                    delivered_at = 1700000000.0 + worker_id * 100.0 + step
                    in_id = f"in-{worker_id}-{step}"
                    out_id = f"msg-{worker_id}-{step}"
                    receipt = DeliveryReceipt(
                        session_id="sess-b206a",
                        platform="telegram",
                        chat_id="1001",
                        chat_type="dm",
                        sender_id="1001",
                        turn_id=turn_id,
                        message_id=out_id,
                        incoming_message_id=in_id,
                        session_key="agent:main:telegram:dm:1001",
                        user_message=f"第{step}轮问候",
                        delivered_text=f"我收到第{step}轮啦",
                        clean_text=f"我收到第{step}轮啦",
                        delivered_at=delivered_at,
                        is_bot=False,
                        transport_confirmed=True,
                        transcript=(
                            {
                                "role": "user",
                                "content": f"第{step}轮问候",
                                "platform_message_id": in_id,
                            },
                            {"role": "assistant", "content": f"我收到第{step}轮啦"},
                        ),
                    )
                    dispatched = emit_delivery_receipt(receipt)
                    assert dispatched >= 1
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=6) as pool:
            tx_futs = [pool.submit(tx_worker, i) for i in range(2)]
            dlv_futs = [pool.submit(delivery_worker, i) for i in range(4)]
            for fut in dlv_futs:
                fut.result()
            try:
                for mod in list(sys.modules.values()):
                    if isinstance(mod, types.ModuleType) and mod.__name__.endswith(
                        ".cognitive.inner_wiring"
                    ):
                        assert mod.wait_for_pending_updates(timeout=30.0)
            finally:
                deliveries_done.set()
            for fut in tx_futs:
                fut.result()

        assert not errors
        assert "OperationalError" not in caplog.text
        assert seq >= 1
        assert committed_probes >= 2 * 2
        assert _count(provider.sqlite, "SELECT COUNT(*) FROM tx_probe WHERE marker LIKE 'rolled-back-%'") == 0
        assert _count(provider.sqlite, "SELECT COUNT(*) FROM tx_probe WHERE marker LIKE 'committed-%'") == committed_probes
        assert _count(provider.sqlite, "SELECT COUNT(*) FROM inner_state_revisions") >= 1
        wm_rows = provider.sqlite.fetchall(
            "SELECT session_id, delivered_at, turn_id FROM session_reply_watermarks "
            "WHERE session_id='sess-b206a'"
        )
        assert len(wm_rows) == 1
        assert wm_rows[0]["turn_id"] == "wm-turn-3-2"
        assert float(wm_rows[0]["delivered_at"]) == pytest.approx(1700000000.0 + 3 * 100.0 + 2.0)
