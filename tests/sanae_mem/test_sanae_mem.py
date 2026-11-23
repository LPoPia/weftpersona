import atexit
import importlib.util
import itertools
import json
import hashlib
import os as _os
import tempfile as _tempfile
import logging
import math
import re
import shutil
import sqlite3
import sys
import threading
import types
from weakref import WeakKeyDictionary
from concurrent.futures import Future
from contextlib import ExitStack
from contextvars import copy_context
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


_TEST_ENV_NAMES = ("HERMES_HOME", "HERMES_HOME_OVERRIDE")
_REAL_PRODUCTION_ROOT = (Path.home() / ".hermes").resolve()


def _production_root() -> Path:
    return _REAL_PRODUCTION_ROOT


def _assert_isolated(root) -> None:
    production = _production_root()
    resolved = Path(root).resolve()
    assert resolved != production, (
        "R1 测试隔离失败：测试根就是生产根（%s）；拒绝运行用例。" % production
    )
    assert not str(resolved).startswith(str(production) + _os.sep), (
        "R1 测试隔离失败：测试根落在生产根之下（%s）；拒绝运行用例。" % resolved
    )


def _ensure_isolated_root() -> Path:
    if not (_os.environ.get("HERMES_HOME") or _os.environ.get("HERMES_HOME_OVERRIDE")):
        root = Path(_tempfile.mkdtemp(prefix="sanae-test-root-"))
        for name in _TEST_ENV_NAMES:
            _os.environ[name] = str(root)
        _assert_isolated(root)
        return root
    roots = []
    for name in _TEST_ENV_NAMES:
        value = _os.environ.get(name)
        if value:
            roots.append(Path(value).resolve())
    for root in roots:
        _assert_isolated(root)
    if len(roots) == 2 and roots[0] != roots[1]:
        raise AssertionError(
            "R1 测试隔离失败：HERMES_HOME=%s 与 HERMES_HOME_OVERRIDE=%s 不一致；拒绝运行用例。"
            % (roots[0], roots[1])
        )
    return roots[0] if roots else _production_root()


@pytest.fixture(scope="session", autouse=True)
def _isolated_session_root():
    return _ensure_isolated_root()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch, _isolated_session_root):
    root = tmp_path_factory.mktemp("home")
    _assert_isolated(root)
    for name in _TEST_ENV_NAMES:
        monkeypatch.setenv(name, str(root))
    return root


def set_test_home(monkeypatch, path) -> Path:
    root = Path(path)
    _assert_isolated(root)
    for name in _TEST_ENV_NAMES:
        monkeypatch.setenv(name, str(root))
    return root


for path in (Path(__file__).resolve().parents[2] / "plugins", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sanae_mem as module
_sanae_mod: Any = module
SanaeMemProvider = _sanae_mod.SanaeMemProvider
register_plugin = _sanae_mod.register
from sanae_mem.infra import host_bridge
from sanae_mem.cognitive import inner_state
from sanae_mem.governance import memory_guard as memory_guard_mod, source_context
from sanae_mem.governance.resident_store import joint_baseline
from sanae_mem.infra import textutil, words
from sanae_mem.runtime.ingest import appraise_clause, event_time_anchor, event_topic
from sanae_mem.storage.curve import initial_half_life


_turn_stack = None


@pytest.fixture(autouse=True)
def _trusted_turns():
    global _turn_stack
    with ExitStack() as stack:
        _turn_stack = stack
        stack.enter_context(host_bridge.turn_envelope_scope(platform="cli", session_id="fixture"))
        try:
            yield
        finally:
            _turn_stack = None


def _enter_test_turn(session_id, identity, context="primary"):
    assert _turn_stack is not None
    fields = {"platform": "test", "chat_type": "dm", "chat_id": "fixture-chat",
              "sender_id": identity.get("user_id", ""),
              "sender_id_alt": identity.get("user_id_alt", ""),
              "session_id": session_id, "session_key": "test:" + session_id,
              "message_id": "fixture-message",
              "is_messaging_surface": False}
    if context != "primary":
        fields = {"platform": "cron", "session_id": session_id}
    _turn_stack.enter_context(host_bridge.turn_envelope_scope(**fields))


_PROVIDER_META: WeakKeyDictionary[Any, tuple[Path, str, str]] = WeakKeyDictionary()
_SEED_SCRATCH_PROVIDERS: dict[str, tuple[Any, Path]] = {}
_SEED_SEQ = itertools.count(1)


@atexit.register
def _cleanup_seed_scratch_providers() -> None:
    for scratch, scratch_home in list(_SEED_SCRATCH_PROVIDERS.values()):
        try:
            scratch.shutdown()
        finally:
            shutil.rmtree(scratch_home, ignore_errors=True)
    _SEED_SCRATCH_PROVIDERS.clear()


def _seed_fingerprint_and_vec(data_dir: Path, text: str) -> tuple[str, bytes]:
    models_src = data_dir / "models"
    cache_key = str(models_src.resolve()) if models_src.exists() else ""

    def _run_in_isolated_ctx() -> tuple[str, bytes]:
        entry = _SEED_SCRATCH_PROVIDERS.get(cache_key)
        if entry is None:
            scratch_home = Path(_tempfile.mkdtemp(prefix="sanae-seed-"))
            scratch_data = scratch_home / "data"
            scratch_data.mkdir(parents=True, exist_ok=True)
            if models_src.exists():
                (scratch_data / "models").symlink_to(models_src)
            with host_bridge.turn_envelope_scope(
                platform="test", chat_type="dm", chat_id="seed-chat",
                sender_id="seed-owner", session_id="seed-sess",
                session_key="test:seed-sess", message_id="seed-init",
                is_messaging_surface=False,
            ):
                scratch = SanaeMemProvider({"data_dir": str(scratch_data), "semantic_backend": "sqlite", "shadow": False})
                scratch.initialize(
                    "seed-sess", hermes_home=str(scratch_home), platform="test",
                    agent_context="primary", user_id="seed-owner",
                )
            _SEED_SCRATCH_PROVIDERS[cache_key] = (scratch, scratch_home)
        else:
            scratch, _ = entry
        seq = next(_SEED_SEQ)
        with host_bridge.turn_envelope_scope(
            platform="test", chat_type="dm", chat_id="seed-chat",
            sender_id="seed-owner", session_id="seed-sess",
            session_key="test:seed-sess", message_id=f"seed-msg-{seq}",
            is_messaging_surface=False,
        ):
            res = json.loads(scratch.handle_tool_call("sanae_mem_add", {"content": text, "importance": 0.9}))
        assert res["ok"] and res["result"]["stored"], res
        eid = res["result"]["id"]
        row = scratch.sqlite.fetchone("SELECT vec FROM events WHERE id=?", (eid,))
        assert row is not None
        vec = bytes(row["vec"])
        with scratch.sqlite.transaction() as conn:
            conn.execute("DELETE FROM event_occurrences WHERE event_id=?", (eid,))
            conn.execute("DELETE FROM events WHERE id=?", (eid,))
        normalized = re.sub(r"\W+", "", text, flags=re.UNICODE).lower()
        fp = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return fp, vec

    return copy_context().run(_run_in_isolated_ctx)


def _provider_owner(provider: Any) -> str:
    meta = _PROVIDER_META.get(provider)
    if meta is not None:
        return meta[1]
    return source_context.current_chat_scope().owner


def make_provider(tmp_path, backend="sqlite", shadow=False, context="primary", **config):
    data_dir = Path(config.get("data_dir") or (tmp_path / "data"))
    values = {"data_dir": str(data_dir), "semantic_backend": backend, "shadow": shadow}
    values.update(config)
    identity = {"user_id": "fixture-owner"}
    for key in ("user_id", "user_id_alt"):
        if key in config:
            identity[key] = config.pop(key)
    if "memory_identity" not in values and identity:
        values["memory_identity"] = {"platform": "test", **identity}
    _enter_test_turn("session-a", identity, context)
    provider = SanaeMemProvider(values)
    provider.initialize(
        "session-a", hermes_home=str(tmp_path), platform="test", agent_context=context, **identity
    )
    owner_key = (
        source_context.scope_account_key(
            source_context.ChatScope(
                "private",
                platform="test",
                actor=str(identity.get("user_id") or ""),
                actor_alt=str(identity.get("user_id_alt") or ""),
            )
        )
        if context == "primary"
        else ""
    )
    _PROVIDER_META[provider] = (data_dir, owner_key, "session-a")
    return provider


def _set_status_fields(provider, **fields):
    def update(status):
        status.update(fields)
        return status

    provider.status_store.mutate(update)


def seed_row(provider, text, importance, *, session_id, now, owner_key="", topic=None,
             ttl=None, source_attribution=("", "", "", "")):
    # These tests require independent rows, including duplicates and pre-existing unowned rows.
    subject, fact, involved, scope = source_attribution
    anchor = event_time_anchor(now)
    meta = _PROVIDER_META.get(provider)
    data_dir = meta[0] if meta is not None else Path(provider.runtime_config()["data_dir"])
    embed_state = textutil.get_embedding_state(data_dir)
    if not embed_state.failed and embed_state.session is None:
        provider.retrieval.search({"query": "warmup"}, internal=True)
    fingerprint, vec = _seed_fingerprint_and_vec(data_dir, text)
    expires = ttl.isoformat() if ttl else ""
    event_id = provider.event_store.insert_event_row(
        fingerprint,
        (anchor, f"[{anchor}] {text}", textutil.event_keywords(text), importance, 0.9,
         topic or event_topic(text), expires or None, now.isoformat(),
         vec, session_id),
        owner_key=owner_key, summary=text[:150], experience_subject=subject, fact_subject=fact,
        involved_owner_keys=involved, read_scope=scope, curve_ttl=expires,
        half_life_days=initial_half_life(provider.ingest.curve_base_days(text), importance),
    )
    assert event_id is not None
    return event_id


def seed_event(provider, content, importance=0.7, topic=None):
    now = datetime.now().astimezone()
    _, owner, session_id = _PROVIDER_META[provider]
    return seed_row(
        provider, content, importance, topic=topic, now=now, session_id=session_id,
        owner_key=owner, ttl=now + timedelta(days=14),
        source_attribution=source_context.self_attribution(owner),
    )


def call(provider, tool, args):
    return json.loads(provider.handle_tool_call(tool, args))


def test_only_explicit_home_routes_every_path(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    root = tmp_path / "explicit-root"
    provider = SanaeMemProvider({"semantic_backend": "sqlite", "shadow": False})
    try:
        with host_bridge.turn_envelope_scope(
            platform="test", chat_type="dm", chat_id="explicit-chat",
            sender_id="explicit-owner", session_id="s-explicit",
            session_key="test:s-explicit", message_id="explicit-msg",
            is_messaging_surface=False,
        ):
            provider.initialize("s-explicit", hermes_home=str(root), platform="test", agent_context="primary")
            added = call(provider, "sanae_mem_add", {"content": "记住只落在显式根目录的测试记录"})
            assert added["ok"] and added["result"]["stored"]
        assert (root / "sanae-mem" / "sanae.db").is_file()
        assert (root / "sanae-mem" / "status.json").is_file()
        assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 1
    finally:
        provider.shutdown()


def test_only_hermes_home_routes_every_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    assert textutil.hermes_home() == str(tmp_path)


def test_only_hermes_home_override_routes_every_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert textutil.hermes_home() == str(tmp_path)


def test_two_root_variables_agreeing_are_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(tmp_path))
    assert textutil.hermes_home() == str(tmp_path)


def test_two_root_variables_disagreeing_fail_fast(tmp_path, monkeypatch):
    left = tmp_path / "root-left"
    right = tmp_path / "root-right"
    monkeypatch.setenv("HERMES_HOME", str(left))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(right))
    with pytest.raises(textutil.HermesHomeConflict) as caught:
        textutil.hermes_home()
    message = str(caught.value)
    assert "HERMES_HOME" in message and "HERMES_HOME_OVERRIDE" in message
    assert str(left) in message and str(right) in message
    with pytest.raises(textutil.HermesHomeConflict):
        SanaeMemProvider({"semantic_backend": "sqlite", "shadow": False})


def test_no_variables_falls_back_to_home_and_stays_off_production(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    expected = fake_home / ".hermes"
    assert textutil.hermes_home() == str(expected), "回退面必须是本轮临时 HOME，不是生产 HOME"
    _assert_isolated(expected)


def test_explicit_host_home_beats_environment_and_records_one_diagnostic(
    tmp_path, monkeypatch, caplog
):
    env_root = tmp_path / "env-root"
    explicit = tmp_path / "host-root"
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(env_root))
    assert textutil.hermes_home(str(explicit)) == str(explicit)

    textutil.reset_home_diagnostics()
    first = textutil.record_home_resolution(str(explicit), str(explicit), log_dir=tmp_path)
    second = textutil.record_home_resolution(str(explicit), str(explicit), log_dir=tmp_path)
    assert first and second == "", "每进程只记一次"
    assert str(explicit) in first and str(env_root) in first
    assert "source=explicit" in first
    assert len(list(tmp_path.glob("home-resolution.log"))) == 1


def test_archive_events_lands_in_the_provider_home_not_production(tmp_path, monkeypatch):
    ambient = set_test_home(monkeypatch, tmp_path / "ambient")
    initial = tmp_path / "initial"
    provider = make_provider(initial)
    added = call(provider, "sanae_mem_add", {"content": "记住上周买的静音键盘"})
    assert added["ok"] and added["result"]["stored"]
    event_id = added["result"]["id"]
    row = provider.sqlite.fetchone("SELECT * FROM events WHERE id=?", (event_id,))
    assert row is not None
    before = dict(row)
    occurrences = [dict(row) for row in provider.sqlite.fetchall(
        "SELECT * FROM event_occurrences WHERE event_id=?", (event_id,),
    )]
    assert occurrences

    rebound = tmp_path / "rebound"
    provider.initialize("session-b", hermes_home=str(rebound), platform="test")
    archive = rebound / "memory-archive" / "events-archived.jsonl"
    archive.mkdir(parents=True)
    failed = call(provider, "sanae_mem_forget", {"id": event_id})
    assert failed["ok"] is False and archive.name in failed["error"]
    row = provider.sqlite.fetchone("SELECT * FROM events WHERE id=?", (event_id,))
    assert row is not None and dict(row) == before
    assert [dict(row) for row in provider.sqlite.fetchall(
        "SELECT * FROM event_occurrences WHERE event_id=?", (event_id,),
    )] == occurrences
    assert provider.sqlite.fetchall("SELECT * FROM forgotten") == []

    archive.rmdir()
    forgotten = call(provider, "sanae_mem_forget", {"id": event_id})
    assert forgotten["ok"] and forgotten["result"]["deleted"] == 1
    assert provider.sqlite.fetchone("SELECT id FROM events WHERE id=?", (event_id,)) is None
    assert provider.sqlite.fetchall("SELECT * FROM event_occurrences WHERE event_id=?", (event_id,)) == []
    assert len(provider.sqlite.fetchall("SELECT * FROM forgotten")) == 1
    entries = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    assert [(entry["raw"], entry.get("observed_at")) for entry in entries] == [
        (before["raw"], None), *((item["raw"], item["observed_at"]) for item in occurrences),
    ]
    assert all((entry["id"], entry["owner_key"], entry["source_context"])
               == (event_id, before["owner_key"], before["source_context"]) for entry in entries)
    for other_home in (ambient, initial, initial / "data"):
        assert not (other_home / "memory-archive" / archive.name).exists()
    provider.shutdown()


_BEHAVIOR_BEGIN = "<!-- RENDER:system_prompt_block:BEGIN -->"
_BEHAVIOR_END = "<!-- RENDER:system_prompt_block:END -->"
_BEHAVIOR_FALLBACK_PREFIX = "（行为纪律块本次未加载——"


def _write_behavior(home: Path, body: str) -> None:
    (home / "BEHAVIOR.md").write_text(
        f"before\n{_BEHAVIOR_BEGIN}\n{body}\n{_BEHAVIOR_END}\nafter", encoding="utf-8",
    )


def test_behavior_block_success_records_probe_without_changing_output(tmp_path, monkeypatch):
    provider = make_provider(tmp_path)
    set_test_home(monkeypatch, tmp_path)
    body = "## 常驻纪律\n- 保持诚实"
    _write_behavior(tmp_path, body)

    expected = (
        "我记着的小事：需要回想过去或自然提起时可用 sanae_mem_search；"
        "顺着话头自然带出，查到了也不必硬提，每轮最多带一处；可用 sanae_mem_forget 忘掉。\n\n" + body
    )
    assert provider.system_prompt_block() == expected
    status = provider.status_store.read()
    assert status["behavior_block_ok"] is True
    assert status["behavior_block_chars"] == len(body)
    assert status["behavior_block_reason"] == "ok"
    datetime.fromisoformat(status["behavior_block_at"])


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("missing", "missing_file"),
        ("marker", "missing_marker"),
        ("directory", "read_error"),
    ],
)
def test_behavior_block_failures_record_reason_and_use_fallback(tmp_path, monkeypatch, setup, reason):
    provider = make_provider(tmp_path)
    set_test_home(monkeypatch, tmp_path)
    path = tmp_path / "BEHAVIOR.md"
    if setup == "marker":
        path.write_text("没有渲染标记", encoding="utf-8")
    elif setup == "directory":
        path.mkdir()

    prompt = provider.system_prompt_block()
    assert "\n\n" + _BEHAVIOR_FALLBACK_PREFIX in prompt
    status = provider.status_store.read()
    assert status["behavior_block_ok"] is False
    assert status["behavior_block_chars"] == 0
    assert status["behavior_block_reason"] == reason
    datetime.fromisoformat(status["behavior_block_at"])


def test_behavior_block_repeated_reads_stay_ok(tmp_path, monkeypatch):
    provider = make_provider(tmp_path)
    set_test_home(monkeypatch, tmp_path)
    body = "## 常驻纪律\n- ①保持诚实\n- ②不出口内部词"
    _write_behavior(tmp_path, body)
    first = provider.system_prompt_block()
    assert body in first
    assert _BEHAVIOR_FALLBACK_PREFIX not in first
    second = provider.system_prompt_block()
    status = provider.status_store.read()
    assert status["behavior_block_ok"] is True, status["behavior_block_reason"]
    assert status["behavior_block_reason"] == "ok"
    assert body in second
    assert _BEHAVIOR_FALLBACK_PREFIX not in second


def test_behavior_zone_shrink_reports_shrunk_not_read_error(tmp_path, monkeypatch):
    provider = make_provider(tmp_path)
    set_test_home(monkeypatch, tmp_path)
    body = "①" + "字" * 4999
    _write_behavior(tmp_path, body)
    assert provider.system_prompt_block().endswith(body)

    _write_behavior(tmp_path, "①短")
    assert _BEHAVIOR_FALLBACK_PREFIX in provider.system_prompt_block()
    status = json.loads((tmp_path / "data" / "status.json").read_text(encoding="utf-8"))
    assert status["behavior_block_ok"] is False
    assert status["behavior_block_reason"].startswith("shrunk:")


def test_behavior_zone_small_but_deliberate_edit_is_not_shrunk(tmp_path, monkeypatch):
    provider = make_provider(tmp_path)
    set_test_home(monkeypatch, tmp_path)
    long_body = "①" + "字" * 4999
    _write_behavior(tmp_path, long_body)
    assert provider.system_prompt_block().endswith(long_body)

    body = "①" + "字" * 1999
    _write_behavior(tmp_path, body)
    assert provider.system_prompt_block().endswith(body)
    status = json.loads((tmp_path / "data" / "status.json").read_text(encoding="utf-8"))
    assert status["behavior_block_reason"] == "ok"


def test_behavior_probe_uses_new_home_after_provider_reinitialize(tmp_path, monkeypatch):
    before = tmp_path / "before"
    set_test_home(monkeypatch, before)
    provider = SanaeMemProvider({"semantic_backend": "sqlite", "shadow": False})
    provider.initialize("session-a", hermes_home=str(before), platform="test")
    long_body = "①" + "字" * 4999
    _write_behavior(before, long_body)
    assert provider.system_prompt_block().endswith(long_body)
    assert provider.prefetch("你好") == ""
    provider.shutdown()

    after = tmp_path / "after"
    after.mkdir()
    set_test_home(monkeypatch, after)
    _write_behavior(after, "①新目录纪律")
    provider.initialize("session-a", hermes_home=str(after), platform="test")
    assert provider.prefetch("你好") == ""
    status = json.loads((after / "sanae-mem" / "status.json").read_text(encoding="utf-8"))
    assert status["behavior_block_ok"] is True
    assert provider.system_prompt_block().endswith("①新目录纪律")
    provider.shutdown()


class GuardStore:
    def __init__(self, entries, limit=100):
        self._entries = entries
        self.limit = limit

    def entries(self, target):
        return self._entries

    def char_limit(self, target):
        return self.limit


def install_guard_store(monkeypatch, store_or_loader):
    def load(_memory_dir):
        return store_or_loader() if callable(store_or_loader) else store_or_loader

    monkeypatch.setattr(memory_guard_mod, "ResidentStore", load)


def test_memory_guard_allows_write_below_threshold(monkeypatch):
    install_guard_store(monkeypatch, GuardStore(["a" * 40]))
    guard = memory_guard_mod
    assert guard.guard_memory_write("memory", {"action": "add", "content": "b" * 20}) is None


def test_memory_guard_blocks_batch_above_threshold(monkeypatch):
    install_guard_store(monkeypatch, GuardStore(["a" * 90]))
    guard = memory_guard_mod
    result = guard.guard_memory_write(
        "memory", {"target": "memory", "operations": [{"action": "add", "content": "b" * 10}]}
    )
    assert result == {"action": "block", "message": guard._BLOCK_MESSAGE}


def test_memory_guard_replace_accounts_for_replaced_entry(monkeypatch):
    install_guard_store(monkeypatch, GuardStore(["old " + "a" * 80]))
    guard = memory_guard_mod
    assert guard.guard_memory_write(
        "memory", {"action": "replace", "old_text": "old", "new_text": "short"}
    ) is None


def test_memory_guard_ignores_non_memory_tools(monkeypatch):
    def fail_if_called():
        raise AssertionError("loader should not run")

    install_guard_store(monkeypatch, fail_if_called)
    guard = memory_guard_mod
    assert guard.guard_memory_write("write_file", {"path": "x", "content": "y"}) is None


def test_sqlite_schema_and_fts_triggers(tmp_path):
    provider = make_provider(tmp_path)
    added = call(provider, "sanae_mem_add", {"content": "我下周准备参加考试", "topic": "学习"})
    event_id = added["result"]["id"]
    assert provider.sqlite.fetchone("SELECT count(*) FROM events_fts")[0] == 1
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET raw='考试安排已更新' WHERE id=?", (event_id,))
    assert provider.sqlite.fetchone("SELECT raw FROM events_fts WHERE rowid=?", (event_id,))[0] == "考试安排已更新"
    deleted = call(provider, "sanae_mem_forget", {"id": event_id})
    assert deleted["ok"] and deleted["result"]["deleted"] == 1
    assert provider.sqlite.fetchone("SELECT count(*) FROM events_fts")[0] == 0


def test_schema_migration_is_idempotent_and_preserves_old_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = data_dir / "sanae.db"
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TABLE events (
             id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, raw TEXT NOT NULL,
             keywords TEXT DEFAULT '', importance REAL DEFAULT 0.5, confidence REAL DEFAULT 0.6,
             topic TEXT DEFAULT '', ttl_expire TEXT, asked INTEGER DEFAULT 0,
             blacklist INTEGER DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT,
             vec BLOB, session_id TEXT DEFAULT ''
           )"""
    )
    connection.execute(
        "INSERT INTO events (ts, raw, created_at) VALUES ('old-ts', 'old-event', '2026-01-01T00:00:00+08:00')"
    )
    connection.commit()
    connection.close()

    provider = make_provider(tmp_path)
    columns = {row[1] for row in provider.sqlite.fetchall("PRAGMA table_info(events)")}
    assert {
        "tier", "promoted_at", "cross_session_hits", "pending_to", "distilled_at", "expired_at", "asked_at"
    } <= columns
    assert tuple(provider.sqlite.fetchone("SELECT raw, tier FROM events")) == (
        "old-event",
        "short",
    )
    provider.shutdown()
    backups = list(data_dir.glob("sanae.db.bak.v0.sqlite"))
    assert len(backups) == 1

    restarted = make_provider(tmp_path)
    assert restarted.sqlite.fetchone("SELECT count(*) FROM events")[0] == 1
    restarted.shutdown()
    assert list(data_dir.glob("sanae.db.bak.v0.sqlite")) == backups


def test_current_events_keep_only_live_appraisal_fields(tmp_path):
    provider = make_provider(tmp_path)
    columns = {row[1] for row in provider.sqlite.fetchall("PRAGMA table_info(events)")}
    assert {"a_conflict", "affect"} <= columns
    assert not ({"a_serious", "a_intent", "a_rel", "a_self"} & columns)

    added = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘"})
    values = provider.sqlite.fetchone(
        "SELECT a_conflict, affect FROM events WHERE id=?",
        (added["result"]["id"],),
    )
    assert tuple(values) == (0, 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "我很信任你，这件重要的事就托付给你了！",
            {"a_conflict": 0, "affect": 2},
        ),
        (
            "哈哈我就是逗你玩的，别当真",
            {"a_conflict": 0, "affect": 0},
        ),
        (
            "我真的很难过，能不能陪陪我？",
            {"a_conflict": 0, "affect": 1},
        ),
        (
            "你错了，我完全不同意，别再这样了！",
            {"a_conflict": 2, "affect": 0},
        ),
        (
            "这是一条普通但足够长的描述",
            {"a_conflict": 0, "affect": 0},
        ),
    ],
)
def test_appraise_clause_keeps_only_conflict_and_affect(text, expected):
    assert appraise_clause(text) == expected


def test_appraisal_is_not_counted_or_exposed_by_search_or_injection(tmp_path):
    provider = make_provider(tmp_path)
    provider.sync_turn("我很信任你，这件重要的事就托付给你了！", "嗯", session_id="session-a")

    status = provider.status_store.read()
    assert "appraise_unconsumed_since" not in status
    assert "appraise_rows_written" not in status
    result = provider.retrieval.search({"query": "信任 托付"}, internal=True)[0]
    assert {"id", "raw", "topic", "importance", "ts"} <= set(result)
    block = provider.prefetch("还记得我托付给你的那件重要的事吗")
    assert all(name not in block for name in ("a_serious", "a_intent", "a_rel", "a_self", "a_conflict", "affect"))


def test_hash_embedding_is_deterministic_normalized_and_matches_dims():
    first = module._hash_embed("想买一把新键盘 hello")
    second = module._hash_embed("想买一把新键盘 hello")
    assert first == second
    assert len(first) == words._DIMS
    assert math.isclose(math.sqrt(sum(item * item for item in first)), 1.0, rel_tol=1e-6)


@pytest.mark.parametrize("backend", ["sqlite", "chroma"])
def test_add_search_forget_end_to_end(tmp_path, backend):
    provider = make_provider(tmp_path, backend=backend, user_id="u1")
    added = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "topic": "购物"})
    assert added["ok"] and added["result"]["stored"]
    event_id = added["result"]["id"]
    found = call(provider, "sanae_mem_search", {"query": "静音键盘"})
    assert found["ok"] and found["result"][0]["id"] == event_id
    forgotten = call(provider, "sanae_mem_forget", {"id": event_id})
    assert forgotten["result"]["deleted"] == 1
    assert provider.sqlite.fetchone("SELECT count(*) FROM events WHERE id=?", (event_id,))[0] == 0
    assert provider.sqlite.fetchone("SELECT count(*) FROM forgotten")[0] == 1
    collection = provider.storage.current_collection()
    if collection is not None:
        assert collection.get(ids=[f"evt-{event_id}"])["ids"] == []


def test_sync_turn_filters_deduplicates_and_updates_status(tmp_path):
    provider = make_provider(tmp_path)
    provider.sync_turn("今天有点焦虑，记住我下周有一场重要考试", "会加油的", session_id="session-a")
    provider.sync_turn("今天有点焦虑，记住我下周有一场重要考试", "会加油的", session_id="session-a")
    assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 1
    row = provider.sqlite.fetchone("SELECT importance FROM events")
    assert row[0] == pytest.approx(0.95)
    status = json.loads((tmp_path / "data" / "status.json").read_text(encoding="utf-8"))
    assert status["daily_writes"] == 2
    provider.sync_turn("查一下今天的服务器运行日志", "嗯", session_id="session-a")
    provider.sync_turn("这是一条普通但足够长且没有任何信号的日常描述", "嗯", session_id="session-a")
    assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 1


def test_unreadable_sample_does_not_prevent_private_event_write(tmp_path, caplog, request):
    provider = make_provider(tmp_path)
    request.addfinalizer(provider.shutdown)
    sample = tmp_path / "data" / "sample.log"
    sample.write_bytes(b"\xff")
    provider.sync_turn("记住我下周有一场重要考试", "会加油的", session_id="session-a")
    assert provider.sqlite.fetchone("SELECT raw FROM events WHERE raw LIKE '%重要考试%'") is not None
    assert sample.read_bytes() == b"\xff"
    assert "sample log read failed (UnicodeDecodeError)" in caplog.text


def test_chroma_open_io_failure_preserves_sqlite_vectors(tmp_path, caplog, request):
    pytest.importorskip("chromadb")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "chroma").touch()
    provider = make_provider(tmp_path, backend="chroma")
    request.addfinalizer(provider.shutdown)
    assert provider.storage.current_collection() is None
    assert "Chroma unavailable; using SQLite vectors (FileExistsError)" in caplog.text
    event_id = _r5_add(provider, "周末去书店买植物百科图鉴")
    stored = provider.sqlite.fetchone("SELECT vec FROM events WHERE id=?", (event_id,))
    assert stored is not None and stored["vec"]
    hits = call(provider, "sanae_mem_search", {"query": "植物百科图鉴"})["result"]
    assert any(hit["id"] == event_id for hit in hits)


def test_chroma_incompatible_persisted_dimensions_preserve_sqlite_write_and_search(tmp_path, caplog, request):
    chromadb = pytest.importorskip("chromadb")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with chromadb.PersistentClient(path=str(data_dir / "chroma")) as client:
        collection = client.get_or_create_collection(words._COLLECTION, metadata={"hnsw:space": "cosine"})
        collection.upsert(ids=["evt-old"], embeddings=[[0.5, 0.5]])
    provider = make_provider(tmp_path, backend="chroma")
    request.addfinalizer(provider.shutdown)
    assert provider.storage.current_collection() is not None
    event_id = _r5_add(provider, "周末去书店买植物百科图鉴")
    hits = call(provider, "sanae_mem_search", {"query": "植物百科图鉴"})["result"]
    assert any(hit["id"] == event_id for hit in hits)
    assert "Chroma write failed; SQLite retained event" in caplog.text
    assert "Chroma query failed; using SQLite vectors" in caplog.text


def test_chroma_upsert_propagates_invalid_input(tmp_path, request):
    pytest.importorskip("chromadb")
    provider = make_provider(tmp_path, backend="chroma")
    request.addfinalizer(provider.shutdown)
    assert provider.storage.current_collection() is not None
    with pytest.raises((IndexError, ValueError)):
        provider.storage.upsert_vector(1, "invalid embedding", "test", 0.5, "2026-11-22", [])


def test_sync_turn_persists_last_user_active_at(tmp_path):
    provider = make_provider(tmp_path)
    provider.sync_turn("这是一条不需要写成事件的普通消息", "嗯", session_id="session-a")
    status_path = tmp_path / "data" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    first_active = datetime.fromisoformat(status["last_user_active_at"])

    restarted = make_provider(tmp_path)
    assert restarted.status_store.read()["last_user_active_at"] == first_active.isoformat()
    restarted.sync_turn("另一条普通消息", "嗯", session_id="session-a")
    refreshed = datetime.fromisoformat(restarted.status_store.read()["last_user_active_at"])
    assert refreshed >= first_active


def test_importance_threshold_and_agent_context_guard(tmp_path):
    primary = make_provider(tmp_path / "primary")
    primary.sync_turn("这是一条长度足够但只是普通日常描述的话", "嗯")
    assert primary.sqlite.fetchone("SELECT count(*) FROM events")[0] == 0
    subagent = make_provider(tmp_path / "subagent", context="subagent")
    subagent.sync_turn("请记住我明天有一场非常重要的考试安排", "嗯")
    assert subagent.sqlite.fetchone("SELECT count(*) FROM events")[0] == 0


def test_low_importance_repeat_does_not_reinforce(tmp_path):
    provider = make_provider(tmp_path)
    provider.sync_turn("记住我周末想买一把静音键盘吧", "嗯", session_id="session-a")
    before = provider.sqlite.fetchone("SELECT importance FROM events")[0]
    provider.sync_turn("周末那把静音键盘目前还是原来的情况呀", "嗯", session_id="session-a")
    after = provider.sqlite.fetchone("SELECT importance FROM events")[0]
    assert before == after == pytest.approx(0.5)


def test_explicit_importance_overrides_the_word_list_and_still_clips(tmp_path):
    provider = make_provider(tmp_path)
    explicit = call(
        provider, "sanae_mem_add", {"content": "把那盆绿萝挪到窗边", "importance": 0.62}
    )["result"]
    assert explicit["stored"] is True
    assert provider.sqlite.fetchone(
        "SELECT importance FROM events WHERE id=?", (explicit["id"],)
    )[0] == pytest.approx(0.62)

    clipped = call(
        provider, "sanae_mem_add", {"content": "给窗台的绿萝换了水", "importance": 3}
    )["result"]
    assert provider.sqlite.fetchone(
        "SELECT importance FROM events WHERE id=?", (clipped["id"],)
    )[0] == pytest.approx(0.95)

    before = provider.sqlite.fetchone("SELECT count(*) FROM events")
    assert before is not None
    provider.sync_turn("窗台上多了几粒灰尘", "嗯", session_id="session-a")
    after = provider.sqlite.fetchone("SELECT count(*) FROM events")
    assert after is not None and after[0] == before[0]


def test_schedule_events_keep_the_short_default_ttl(tmp_path):
    provider = make_provider(tmp_path)
    added = call(provider, "sanae_mem_add", {"content": "记住下周三的面试安排"})["result"]
    row = provider.sqlite.fetchone(
        "SELECT created_at, ttl_expire FROM events WHERE id=?", (added["id"],)
    )
    days = (datetime.fromisoformat(row["ttl_expire"])
            - datetime.fromisoformat(row["created_at"])).total_seconds() / 86400
    assert 6.9 < days < 7.1


def test_forgotten_candidate_is_not_reingested_or_counted(tmp_path):
    provider = make_provider(tmp_path, user_id="forget-user")
    added = call(provider, "sanae_mem_add", {"content": "用户托付给我的那件重要的事，我一定替他记住"})
    call(provider, "sanae_mem_forget", {"id": added["result"]["id"]})
    before_writes = provider.status_store.read().get("daily_writes", 0)
    provider.sync_turn("用户托付给我的那件重要的事，我一定替他记住", "嗯", session_id="session-a")
    assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 0
    status = json.loads((tmp_path / "data" / "status.json").read_text(encoding="utf-8"))
    assert "last_user_active_at" in status
    assert status.get("daily_writes", 0) == before_writes


def test_time_anchor_is_stored_and_never_recomputed(tmp_path):
    provider = make_provider(tmp_path)
    text = "记住周末准备买拖鞋"
    added = call(provider, "sanae_mem_add", {"content": text})
    event_id = added["result"]["id"]
    written = provider.sqlite.fetchone("SELECT ts, raw, created_at FROM events WHERE id=?", (event_id,))
    assert written is not None
    assert written["ts"] == event_time_anchor(datetime.fromisoformat(written["created_at"]))
    assert written["raw"] == f"[{written['ts']}] {text}"

    stored_anchor = "2025-09-01 00:00|那天晚上"
    stored_raw = f"[{stored_anchor}] {text}"
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET ts=?,raw=? WHERE id=?", (stored_anchor, stored_raw, event_id))
    result = provider.retrieval.search({"query": "拖鞋"}, internal=True)[0]
    assert result["ts"] == stored_anchor
    assert result["raw"] == stored_raw


def test_shadow_suppresses_reads_but_keeps_writes(tmp_path):
    provider = make_provider(tmp_path, shadow=True, user_id="u1")
    added = call(provider, "sanae_mem_add", {"content": "我想买一双新拖鞋"})
    assert added["result"]["stored"]
    assert call(provider, "sanae_mem_search", {"query": "拖鞋"})["result"] == []
    assert provider.prefetch("拖鞋") == ""
    forgotten = call(provider, "sanae_mem_forget", {"query": "拖鞋"})
    assert forgotten["result"]["deleted"] == 1


def test_long_relation_turn_is_split_and_calibrated(tmp_path):
    provider = make_provider(tmp_path)
    samples = [
        "这段铺垫只是为了模拟一条明显超过长度门的输入，其中有很多不需要保存的普通描述和语气词，也不会单独成为候选。"
        "我希望你能慢慢走进我的内心，成为我愿意信任和依靠的重要的人。",
        "今天只是聊了些没有记忆信号的普通生活安排。",
        "这份庇护和约定对我很重要，也请你珍惜它。",
    ]
    for sample in samples:
        provider.sync_turn(sample, "嗯", session_id="session-a")
    rows = provider.sqlite.fetchall("SELECT topic, importance, ttl_expire, created_at FROM events")
    assert len(rows) >= 2
    assert all(row["topic"] == "关系" for row in rows)
    assert all(float(row["importance"]) >= 0.45 for row in rows)
    assert all(
        (datetime.fromisoformat(row["ttl_expire"]) - datetime.fromisoformat(row["created_at"])).days
        >= 29
        for row in rows
    )


def test_read_gate_filters_marks_asked_and_caps_injection(tmp_path):
    provider = make_provider(tmp_path, read_gate="explicit", inject_max_chars=80, gap_continue=False)
    call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    assert provider.prefetch("静音键盘") == ""
    block = provider.prefetch("还记得我说过的静音键盘吗")
    if _fallback_fail_closed():
        assert block == "", "散列回退不得自动注入；实际 %r" % (block,)
        assert provider.sqlite.fetchone("SELECT asked FROM events")[0] == 0, \
            "没进块的行不得被标记"
        return
    assert block.startswith("## 小事记忆\n-")
    assert len(block) <= 80
    assert provider.sqlite.fetchone("SELECT asked FROM events")[0] == 1
    assert provider.prefetch("还记得我说过的静音键盘吗").startswith("## 小事记忆\n-")


def test_gap_threshold_controls_continuation(tmp_path):
    provider = make_provider(tmp_path, gap_threshold_hours=4)
    call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    provider.prefetch("你好", session_id="session-a")
    now = datetime.now().astimezone()

    _set_status_fields(provider, last_user_active_at=(now - timedelta(hours=3)).isoformat())
    assert provider.prefetch("今天想随便聊点什么", session_id="session-a") == ""

    _set_status_fields(provider, last_user_active_at=(now - timedelta(hours=5)).isoformat())
    assert provider.prefetch("今天想随便聊点什么", session_id="session-a") == ""
    assert provider.prefetch("静音键盘的事", session_id="session-a").startswith("## 小事记忆\n")
    assert provider.status_store.read().get("daily_scheduled_asks", 0) == 0


def test_fresh_session_continues_once_and_same_session_does_not_repeat(tmp_path):
    provider = make_provider(tmp_path)
    call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})

    first = provider.prefetch("静音键盘的事", session_id="session-new")
    assert "## 上次聊到的" in first
    assert provider.prefetch("还可以再聊聊别的吗", session_id="session-new") == ""
    assert provider.prefetch("静音键盘的事", session_id="session-new") == ""


def test_prefetch_reuses_private_hook_outcome_and_rechecks_owner(tmp_path, monkeypatch):
    set_test_home(monkeypatch, tmp_path)
    provider = make_provider(tmp_path, gap_threshold_hours=0)
    query = "静音键盘的事"
    added = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    event_id = added["result"]["id"]
    with sqlite3.connect(tmp_path / "data" / "sanae.db") as conn:
        owner = conn.execute("SELECT owner_key FROM events WHERE id=?", (event_id,)).fetchone()[0]

    current_owner = owner
    checks = 0

    def resolve_owner():
        nonlocal checks
        checks += 1
        return current_owner

    provider.association_state.bind_read_owner(owner, resolver=resolve_owner)
    hook = provider.handle_pre_llm_call(user_message=query, session_id="session-a")
    assert "## 上次聊到的" in hook["context"]
    assert provider.recall_status() is not None
    before = checks
    assert provider.prefetch(query, session_id="session-a") == ""
    assert checks == before
    assert provider.prefetch(query, session_id="session-a") == ""
    assert checks == before + 1

    current_owner = "person:rebound"
    assert provider.prefetch(query, session_id="session-a") == ""
    assert provider.recall_status() is None

    current_owner = owner
    provider.association_state.bind_read_owner(owner, resolver=resolve_owner)
    missing = "海边鹦鹉的事"
    hook = provider.handle_pre_llm_call(user_message=missing, session_id="session-a")
    assert "## 小事记忆" not in hook.get("context", "")
    assert provider.recall_status() is None
    before = checks
    assert provider.prefetch(missing, session_id="session-a") == ""
    assert checks == before
    assert provider.prefetch(missing, session_id="session-a") == ""
    assert checks == before + 1
    provider.shutdown()


@pytest.mark.parametrize("same_query", [True, False], ids=["same-query", "different-query"])
def test_prefetch_cancels_in_flight_hook_without_late_injection(tmp_path, monkeypatch, same_query):
    set_test_home(monkeypatch, tmp_path)
    provider = make_provider(tmp_path)
    query = "静音键盘的事"
    added = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    event_id = added["result"]["id"]
    with sqlite3.connect(tmp_path / "data" / "sanae.db") as conn:
        owner = conn.execute("SELECT owner_key FROM events WHERE id=?", (event_id,)).fetchone()[0]

    started = threading.Event()
    release = threading.Event()
    prefetch_checked_owner = threading.Event()
    hook_reply: Future[dict] = Future()

    def resolve_owner():
        if started.is_set():
            prefetch_checked_owner.set()
        else:
            started.set()
            if not release.wait(5):
                raise TimeoutError("hook read did not resume")
        return owner

    def invoke_hook():
        try:
            hook_reply.set_result(provider.handle_pre_llm_call(user_message=query, session_id="session-a"))
        except BaseException as exc:
            hook_reply.set_exception(exc)

    provider.association_state.bind_read_owner(owner, resolver=resolve_owner)
    hook = host_bridge.spawn_context_thread(invoke_hook, name="read-gate-test-hook")
    hook.start()
    try:
        assert started.wait(3)
        prefetch_query = query if same_query else "还记得海边的鹦鹉吗"
        assert provider.prefetch(prefetch_query, session_id="session-a") == ""
        assert prefetch_checked_owner.is_set() is not same_query
    finally:
        release.set()
        hook.join(6)

    assert "静音键盘" not in hook_reply.result(timeout=1).get("context", "")
    assert provider.recall_status() is None
    with sqlite3.connect(tmp_path / "data" / "sanae.db") as conn:
        assert conn.execute("SELECT asked FROM events WHERE id=?", (event_id,)).fetchone() == (0,)

    provider.on_session_switch("session-b")
    _enter_test_turn("session-b", {"user_id": "fixture-owner"})
    provider.on_turn_start(1, query)
    assert "## 上次聊到的" in provider.prefetch(query, session_id="session-b")
    provider.shutdown()


def test_fresh_session_skips_trivial_and_busy_context(tmp_path):
    provider = make_provider(tmp_path)
    call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})

    assert provider.prefetch("你好", session_id="trivial-session") == ""
    assert provider.prefetch("我正在忙工作，晚点再聊", session_id="busy-session") == ""
    assert provider.sqlite.fetchone("SELECT asked FROM events")[0] == 0


def test_continuation_prefers_promoted_tier_and_is_natural_and_bounded(tmp_path):
    provider = make_provider(tmp_path)
    short_id = call(
        provider, "sanae_mem_add", {"content": "记住我明天准备复习课程", "importance": 0.95}
    )["result"]["id"]
    long_id = call(
        provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.7}
    )["result"]["id"]
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET tier='short' WHERE id=?", (short_id,))
        conn.execute("UPDATE events SET tier='long', ttl_expire=NULL WHERE id=?", (long_id,))

    block = provider.prefetch("静音键盘的事", session_id="session-new")
    assert "购物" in block
    assert "键盘" in block
    assert "学习" not in block
    assert "我想" not in block
    assert "我想买一把静音键盘" not in block
    assert len(block) <= 100
    assert block.count("\n- ") == 1
    assert all(word not in block for word in ("自动续接", "检索", "记忆库"))


def test_gap_continue_config_defaults_and_switch(tmp_path):
    provider = make_provider(tmp_path)
    schema = {item["key"]: item for item in provider.get_config_schema()}
    assert schema["gap_continue"]["default"] is True
    assert schema["gap_threshold_hours"]["default"] == 2

    disabled = make_provider(tmp_path / "disabled", gap_continue=False)
    call(disabled, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    assert disabled.prefetch("今天想随便聊点什么", session_id="fresh-session") == ""


def test_scheduled_gate_and_daily_limit(tmp_path):
    provider = make_provider(tmp_path, read_gate="scheduled", daily_scheduled_asks=1)
    first = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    second = call(provider, "sanae_mem_add", {"content": "我明天准备复习课程", "importance": 0.8})
    assert provider.prefetch("键盘", scheduled=True)
    assert provider.prefetch("课程", scheduled=True) == ""
    assert provider.sqlite.fetchone("SELECT asked FROM events WHERE id=?", (first["result"]["id"],))[0] == 1
    assert provider.sqlite.fetchone("SELECT asked FROM events WHERE id=?", (second["result"]["id"],))[0] == 0


def test_injection_excludes_old_asked_and_blacklisted_rows(tmp_path):
    provider = make_provider(tmp_path)
    ids = [
        seed_event(provider, content=f"记住静音键盘方案{i}")
        for i in range(4)
    ]
    old = (datetime.now().astimezone() - timedelta(days=31)).isoformat()
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET created_at=? WHERE id=?", (old, ids[1]))
        conn.execute(
            "UPDATE events SET asked=1, asked_at=? WHERE id=?",
            (datetime.now().astimezone().isoformat(), ids[2]),
        )
        conn.execute("UPDATE events SET blacklist=1 WHERE id=?", (ids[3],))
    results = provider.retrieval.search({"query": "静音键盘方案", "top_k": 10}, for_injection=True)
    recalled = "\n".join(item["raw"] for item in results)
    assert f"方案0" in recalled
    assert all(f"方案{index}" not in recalled for index in (1, 2, 3))


def test_false_trigger_twice_is_counted_without_auto_shadow(tmp_path):
    provider = make_provider(tmp_path)
    event_id = call(
        provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9}
    )["result"]["id"]
    first = call(provider, "sanae_mem_report_false_trigger", {})
    assert first["result"] == {"weekly_count": 1, "shadow": False}
    assert not provider.shadow_enabled()
    second = call(provider, "sanae_mem_report_false_trigger", {})
    assert second["result"] == {"weekly_count": 2, "shadow": False}
    assert not provider.shadow_enabled()

    status = json.loads((tmp_path / "data" / "status.json").read_text(encoding="utf-8"))
    assert status["weekly_false_triggers"] == 2
    assert status["false_trigger_week"]
    assert status["last_false_trigger_at"]
    assert "auto_shadow" not in status and "auto_shadow_at" not in status
    log = (tmp_path / "data" / "safety.log").read_text(encoding="utf-8")
    assert "false-trigger=2 action=counted" in log
    assert "auto-shadow" not in log

    found = call(provider, "sanae_mem_search", {"query": "静音键盘"})
    assert [item["id"] for item in found["result"]] == [event_id]
    assert provider.association_state.current_recall() is None
    provider.prefetch("还记得我说过的静音键盘吗", session_id="session-a")
    recall_state = provider.association_state.current_recall()
    assert recall_state is not None and recall_state.get("status") in {"ok", "empty"}, \
        "两次上报后 prefetch 仍应进入联想／检索"

    _set_status_fields(
        provider, auto_shadow=True,
        auto_shadow_at=datetime.now().astimezone().isoformat())
    restarted = make_provider(tmp_path)
    assert not restarted.shadow_enabled()


def test_manual_shadow_survives_false_trigger_reports(tmp_path):
    provider = make_provider(tmp_path, shadow=True, user_id="u1")
    added = call(provider, "sanae_mem_add", {"content": "我想买一双新拖鞋"})
    assert added["result"]["stored"]
    assert call(provider, "sanae_mem_report_false_trigger", {})["result"] == {
        "weekly_count": 1, "shadow": True}
    assert call(provider, "sanae_mem_report_false_trigger", {})["result"] == {
        "weekly_count": 2, "shadow": True}
    assert provider.shadow_enabled(), "shadow 只来自显式配置；误触发上报既不开也不关它"
    assert call(provider, "sanae_mem_search", {"query": "拖鞋"})["result"] == []
    assert provider.prefetch("拖鞋") == ""
    assert call(provider, "sanae_mem_forget", {"query": "拖鞋"})["result"]["deleted"] == 1


def test_search_recall_meta_exposes_tier_and_summary_source(tmp_path):
    provider = make_provider(tmp_path)
    short_id = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘"})["result"]["id"]
    long_id = call(provider, "sanae_mem_add", {"content": "我明天准备复习课程"})["result"]["id"]
    with provider.sqlite.transaction() as conn:
        conn.execute(
            "UPDATE events SET tier='long', summary=?, summary_source='distilled' WHERE id=?",
            ("整理后的长期摘要", long_id),
        )

    results = provider.retrieval.search(
        {"query": "静音键盘 复习课程", "top_k": 10}, internal=True, with_recall_meta=True)
    by_id = {item["id"]: item for item in results}
    assert by_id[long_id]["tier"] == "long"
    assert by_id[long_id]["summary_source"] == "distilled"
    assert by_id[short_id]["tier"] == "short"
    assert by_id[short_id]["summary_source"] == "original"


def test_compactor_expires_merges_decays_and_updates_status(tmp_path):
    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()
    expired = seed_event(provider, content='记住已经过期的考试')
    keep = seed_event(provider, content='记住重复的静音键盘', importance=0.8)
    duplicate = seed_event(provider, content='记住另一个重复事件', importance=0.9)
    old = seed_event(provider, content='记住很久以前的课程')
    keywords = provider.sqlite.fetchone("SELECT keywords FROM events WHERE id=?", (keep,))[0]
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET ttl_expire=? WHERE id=?", ((now - timedelta(days=1)).isoformat(), expired))
        conn.execute("UPDATE events SET keywords=? WHERE id=?", (keywords, duplicate))
        conn.execute(
            "UPDATE events SET created_at=?, confidence=0.8, asked=0 WHERE id=?",
            ((now - timedelta(days=31)).isoformat(), old),
        )
    result = provider.lifecycle.compact(now)
    assert {key: result[key] for key in ("expired", "merged", "decayed")} == {
        "expired": 1,
        "merged": 1,
        "decayed": 1,
    }
    assert "candidates" not in result
    expired_row = provider.sqlite.fetchone(
        "SELECT tier, ttl_expire, expired_at FROM events WHERE id=?", (expired,)
    )
    assert tuple(expired_row[:2]) == ("short", None)
    assert expired_row["expired_at"] == now.isoformat()
    assert provider.sqlite.fetchone("SELECT count(*) FROM forgotten")[0] == 0
    assert provider.sqlite.fetchone("SELECT count(*) FROM events WHERE id IN (?,?)", (keep, duplicate))[0] == 1
    assert provider.sqlite.fetchone("SELECT confidence FROM events WHERE id=?", (old,))[0] == pytest.approx(0.72)
    status = json.loads((tmp_path / "data" / "status.json").read_text(encoding="utf-8"))
    assert status["daily_compactor_runs"] == 1
    assert status["last_compactor"] == result
    assert provider.lifecycle.compact(now)["decayed"] == 0


def test_expired_short_is_hidden_then_reinforced_with_fresh_ttl(tmp_path):
    provider = make_provider(tmp_path, user_id="reinforcement-user")
    now = datetime.now().astimezone()
    content = "和你一起挑选的静音键盘对我很重要，我一定要记住这个约定"
    provider.sync_turn(content, "嗯", session_id="session-a")
    event_id = provider.sqlite.fetchone("SELECT id FROM events")[0]
    with provider.sqlite.transaction() as conn:
        conn.execute(
            "UPDATE events SET ttl_expire=? WHERE id=?",
            ((now - timedelta(seconds=1)).isoformat(), event_id),
        )

    result = provider.lifecycle.compact(now)
    assert result["expired"] == 1
    assert result["distill_candidates"] == []
    assert call(provider, "sanae_mem_search", {"query": "静音键盘"})["result"] == []

    provider.sync_turn(content, "嗯", session_id="session-new")
    row = provider.sqlite.fetchone(
        "SELECT tier, expired_at, ttl_expire, importance FROM events WHERE id=?", (event_id,)
    )
    assert row["tier"] == "short"
    assert row["expired_at"] is None
    assert 13 < (
        datetime.fromisoformat(row["ttl_expire"]) - datetime.now().astimezone()
    ).total_seconds() / 86400 < 15
    assert row["importance"] > 0.7
    assert call(provider, "sanae_mem_search", {"query": "静音键盘"})["result"][0]["id"] == event_id


def test_expiring_event_is_tombstoned_after_review_window(tmp_path):
    provider = make_provider(tmp_path, expiry_review_days=30)
    now = datetime.now().astimezone()
    content = "记住已经过期的考试安排"
    event_id = call(provider, "sanae_mem_add", {"content": content})["result"]["id"]
    with provider.sqlite.transaction() as conn:
        conn.execute(
            "UPDATE events SET ttl_expire=? WHERE id=?",
            ((now - timedelta(seconds=1)).isoformat(), event_id),
        )

    provider.lifecycle.compact(now)
    result = provider.lifecycle.compact(now + timedelta(days=31))
    assert result["forgotten"] == 1
    assert provider.sqlite.fetchone("SELECT count(*) FROM events WHERE id=?", (event_id,))[0] == 0
    assert provider.sqlite.fetchone("SELECT count(*) FROM forgotten")[0] == 1
    assert call(provider, "sanae_mem_add", {"content": content})["result"]["stored"] is False


def test_expiring_importance_decays_once_daily_to_configured_floor(tmp_path):
    provider = make_provider(
        tmp_path, expiring_importance_decay=0.5, expiring_importance_floor=0.4
    )
    now = datetime.now().astimezone()
    event_id = call(
        provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.7}
    )["result"]["id"]
    with provider.sqlite.transaction() as conn:
        conn.execute(
            "UPDATE events SET ttl_expire=? WHERE id=?",
            ((now - timedelta(seconds=1)).isoformat(), event_id),
        )

    first = provider.lifecycle.compact(now)
    assert first["importance_decayed"] == 1
    assert provider.sqlite.fetchone(
        "SELECT importance FROM events WHERE id=?", (event_id,)
    )[0] == pytest.approx(0.4)
    assert provider.lifecycle.compact(now)["importance_decayed"] == 0

    schema = {item["key"]: item for item in provider.get_config_schema()}
    assert schema["expiry_review_days"]["default"] == 30
    assert schema["expiring_importance_decay"]["default"] == 0.9
    assert schema["expiring_importance_floor"]["default"] == 0.4


def test_low_importance_expiring_event_accelerates_forgetting_after_one_day(tmp_path):
    provider = make_provider(tmp_path, expiry_review_days=30)
    now = datetime.now().astimezone()
    event_id = call(
        provider, "sanae_mem_add", {"content": "我想买一把静音键盘"}
    )["result"]["id"]
    with provider.sqlite.transaction() as conn:
        conn.execute(
            """UPDATE events
               SET ttl_expire=NULL, expired_at=?, importance=0.44
               WHERE id=?""",
            ((now - timedelta(days=2)).isoformat(), event_id),
        )

    result = provider.lifecycle.compact(now)
    assert result["forgotten"] == 1
    assert provider.sqlite.fetchone("SELECT count(*) FROM events WHERE id=?", (event_id,))[0] == 0
    assert provider.sqlite.fetchone("SELECT count(*) FROM forgotten")[0] == 1


def test_active_forget_still_deletes_expiring_event_immediately(tmp_path):
    provider = make_provider(tmp_path, user_id="u1")
    now = datetime.now().astimezone()
    event_id = call(
        provider, "sanae_mem_add", {"content": "我想买一把静音键盘"}
    )["result"]["id"]
    with provider.sqlite.transaction() as conn:
        conn.execute(
            "UPDATE events SET ttl_expire=? WHERE id=?",
            ((now - timedelta(seconds=1)).isoformat(), event_id),
        )
    provider.lifecycle.compact(now)

    forgotten = call(provider, "sanae_mem_forget", {"query": "静音键盘"})
    assert forgotten["result"]["ids"] == [event_id]
    assert forgotten["result"]["deleted"] == 1
    assert provider.sqlite.fetchone("SELECT count(*) FROM forgotten")[0] == 1


def test_permanent_tiers_do_not_enter_expiry_review(tmp_path):
    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()
    ids = [
        seed_event(provider, content=f"值得长期保留的事项{index}")
        for index in range(3)
    ]
    with provider.sqlite.transaction() as conn:
        for event_id, tier in zip(ids, ("long", "important", "distilled")):
            conn.execute(
                "UPDATE events SET tier=?, ttl_expire=NULL WHERE id=?", (tier, event_id)
            )

    result = provider.lifecycle.compact(now + timedelta(days=365))
    rows = provider.sqlite.fetchall(
        "SELECT id, tier, expired_at FROM events WHERE id IN (?,?,?) ORDER BY id", ids
    )
    assert result["expired"] == 0
    assert [(row["id"], row["tier"], row["expired_at"]) for row in rows] == list(
        zip(ids, ("long", "important", "distilled"), (None, None, None))
    )


def test_compact_records_distill_failure_but_still_completes(tmp_path, caplog):
    class DistillCandidateLlm:
        def complete(self, messages, **kwargs):
            return types.SimpleNamespace(text=json.dumps({
                "summary": "今天决定去报古琴班",
                "duty": "memory",
                "source_kind": "user_statement",
                "long_term": True,
                "covered_by": "",
                "significance": "长期兴趣决定",
            }, ensure_ascii=False))

    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()
    (tmp_path / "memories").mkdir()
    for name in ("USER.md", "MEMORY.md"):
        (tmp_path / "memories" / name).write_text("", encoding="utf-8")
    provider.lifecycle.compact(now)
    status = json.loads((tmp_path / "data" / "status.json").read_text(encoding="utf-8"))
    assert status["last_distill_ok_at"] == now.isoformat()
    assert status.get("distill_errors_today", 0) == 0

    provider.attach_runtime_llm(DistillCandidateLlm())
    seed_event(provider, content="我想学古琴很多年了，今天终于决定去报班。", importance=0.8)
    with provider.sqlite.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER test_distill_outage BEFORE UPDATE OF distilled_at ON events "
            "WHEN NEW.distilled_at IS NOT NULL "
            "BEGIN SELECT RAISE(ABORT, 'test distill outage'); END"
        )
    try:
        with caplog.at_level(logging.WARNING, logger="sanae_mem_tested"):
            result = provider.lifecycle.compact(now)
    finally:
        with provider.sqlite.transaction() as conn:
            conn.execute("DROP TRIGGER IF EXISTS test_distill_outage")
    assert {"expired", "merged", "decayed", "distill_candidates"} <= set(result)
    status = json.loads((tmp_path / "data" / "status.json").read_text(encoding="utf-8"))
    assert status["daily_compactor_runs"] == 2
    assert status["distill_errors_today"] == 1
    assert status["last_distill_error_at"] == now.isoformat()
    assert "IntegrityError" in status["last_distill_error"]
    assert "test distill outage" in status["last_distill_error"]
    assert status["last_compactor"] == result
    assert "compact distill failed" in caplog.text


def test_compact_records_zero_distill_output_when_review_quota_is_exhausted(tmp_path):
    provider = make_provider(tmp_path, distill_daily_cap=2)
    now = datetime.now().astimezone()
    provider.lifecycle.compact(now)
    (tmp_path / "memories").mkdir()
    for name in ("USER.md", "MEMORY.md"):
        (tmp_path / "memories" / name).write_text("", encoding="utf-8")
    seed_event(provider, content="值得长期记得的第一件事", importance=0.8)
    seed_event(provider, content="值得长期记得的第二件事", importance=0.7)
    _set_status_fields(provider, date=now.date().isoformat(), daily_candidate_reviews=2)

    provider.lifecycle.compact(now)

    status = provider.status_store.read()
    assert status["last_distill_result"] == {
        "candidates": 2, "written": 0,
        "zero_output_reasons": {"quota_exhausted": 2},
    }


def test_compact_records_zero_distill_output_on_parse_failure(tmp_path):
    class InvalidLlm:
        def complete(self, messages, **kwargs):
            return types.SimpleNamespace(text="invalid candidate")

    provider = make_provider(tmp_path, distill_daily_cap=2)
    now = datetime.now().astimezone()
    (tmp_path / "memories").mkdir()
    for name in ("USER.md", "MEMORY.md"):
        (tmp_path / "memories" / name).write_text("", encoding="utf-8")
    provider.attach_runtime_llm(InvalidLlm())
    seed_event(provider, content="值得长期记得的一件事", importance=0.8)

    provider.lifecycle.compact(now)

    status = provider.status_store.read()
    assert status["last_distill_result"] == {
        "candidates": 1, "written": 0,
        "zero_output_reasons": {"parse_failed": 1},
    }


def test_skipped_review_retries_legacy_empty_cache_when_quota_returns(tmp_path, monkeypatch):
    provider = make_provider(tmp_path, distill_daily_cap=2)
    now = datetime.now().astimezone()
    (tmp_path / "memories").mkdir()
    for name in ("USER.md", "MEMORY.md"):
        (tmp_path / "memories" / name).write_text("", encoding="utf-8")
    event_id = seed_event(provider, content="值得长期记得的一件事", importance=0.8)
    row = provider.sqlite.fetchone(
        "SELECT raw, updated_at, created_at FROM events WHERE id=?", (event_id,)
    )
    provider.event_review_store.put_event_review(
        event_id, target="", duty="", content_sha256=hashlib.sha256(b"").hexdigest(),
        event_sha256=hashlib.sha256(str(row["raw"]).encode("utf-8")).hexdigest(),
        event_version=str(row["updated_at"] or row["created_at"] or ""),
        baseline_sha256=joint_baseline("", ""),
        reviewed_text="", now=now.isoformat(),
    )
    calls = []

    class _ValidCandidateLlm:
        def complete(self, messages, **kwargs):
            calls.append(event_id)
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "summary": "这件事值得长期记得",
                        "duty": "memory",
                        "source_kind": "unknown",
                        "long_term": True,
                        "covered_by": "",
                        "significance": "重要经历",
                    },
                    ensure_ascii=False,
                )
            )

    provider.attach_runtime_llm(_ValidCandidateLlm())
    result = provider.lifecycle.distill(now)
    assert [item["id"] for item in result["items"]] == [event_id]
    assert calls == [event_id]
    assert provider.status_store.read()["last_distill_result"] == {
        "candidates": 1, "written": 1, "zero_output_reasons": {},
    }


def test_valid_negative_review_is_reused_without_new_model_call(tmp_path, monkeypatch):
    provider = make_provider(tmp_path, distill_daily_cap=2)
    (tmp_path / "memories").mkdir()
    for name in ("USER.md", "MEMORY.md"):
        (tmp_path / "memories" / name).write_text("", encoding="utf-8")
    event_id = seed_event(provider, content="今天普通的一件事")
    calls = []

    class _NegativeCandidateLlm:
        def complete(self, messages, **kwargs):
            calls.append(event_id)
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "summary": "普通的一件事",
                        "duty": "memory",
                        "source_kind": "unknown",
                        "long_term": False,
                        "covered_by": "",
                        "significance": "",
                    },
                    ensure_ascii=False,
                )
            )

    provider.attach_runtime_llm(_NegativeCandidateLlm())
    first = provider.promotion_review.review_promotion_candidate(event_id)
    second = provider.promotion_review.review_promotion_candidate(event_id)
    assert first["action"] == "none"
    assert second["reused"] is True
    assert calls == [event_id]


def test_marked_promoted_to_important_clears_the_expiry(tmp_path):
    provider = make_provider(tmp_path)
    event_id = call(provider, "sanae_mem_add", {"content": "记住我最喜欢静音键盘"})["result"]["id"]
    ttl_expire = (datetime.now().astimezone() + timedelta(hours=2)).isoformat()
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET ttl_expire=? WHERE id=?", (ttl_expire, event_id))
    result = call(
        provider,
        "sanae_mem_marked_promoted",
        {"ids": [event_id], "to": "important"},
    )["result"]
    assert result == {"ids": [event_id], "to": "important", "updated": 1}
    row = provider.sqlite.fetchone(
        "SELECT tier, promoted_at, pending_to, ttl_expire FROM events WHERE id=?", (event_id,)
    )
    assert row[0] == "important"
    assert row[1]
    assert row[2] == ""
    assert row[3] is None


def test_distill_candidate_filters_threshold_emotion_topic_and_exclusions(tmp_path):
    provider = make_provider(tmp_path, distill_daily_cap=10)
    now = datetime.now().astimezone()
    threshold = seed_event(provider, content='一次很有意义的尝试', importance=0.6)
    emotion = seed_event(provider, content='最近有些焦虑', importance=0.5)
    identity = seed_event(provider, content='对自我定位的新理解', topic='身份', importance=0.5)
    blacklisted = seed_event(provider, content='不该留下的高分事项', importance=0.9)
    done = seed_event(provider, content='已经蒸馏过的事项', importance=0.95)
    system = seed_event(provider, content='服务器部署已经完成', importance=0.99)
    conflict = seed_event(provider, content='这次吵架很重要', importance=0.98)
    expired = seed_event(provider, content='已经过期的高分事项', importance=0.97)
    pending = seed_event(provider, content='等待P4处理的事项', importance=0.96)
    permanent = seed_event(provider, content='已有永久层级的事项', importance=0.94)
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET blacklist=1 WHERE id=?", (blacklisted,))
        conn.execute("UPDATE events SET distilled_at=? WHERE id=?", (now.isoformat(), done))
        conn.execute("UPDATE events SET ttl_expire=? WHERE id=?", ((now - timedelta(seconds=1)).isoformat(), expired))
        conn.execute("UPDATE events SET tier='pending' WHERE id=?", (pending,))
        conn.execute("UPDATE events SET tier='long' WHERE id=?", (permanent,))

    ids = {item["id"] for item in provider.lifecycle.compact(now)["distill_candidates"]}
    assert ids == {threshold, emotion, identity}
    assert not ids & {blacklisted, done, system, conflict, expired, pending, permanent}


def test_distill_daily_cap_and_importance_order(tmp_path):
    provider = make_provider(tmp_path, distill_daily_cap=2)
    tool_names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert "sanae_mem_distill" in tool_names
    assert "sanae_mem_marked_distilled" not in tool_names
    ids = [
        seed_event(provider, content=f"高价值事项{index}", importance=importance)
        for index, importance in enumerate((0.7, 0.95, 0.8))
    ]
    candidates = provider.lifecycle.compact()["distill_candidates"]
    assert [item["id"] for item in candidates] == [ids[1], ids[2]]


@pytest.mark.parametrize("backend", ["sqlite", "chroma"])
def test_distill_updates_summary_indexes_and_preserves_raw_and_lifecycle(tmp_path, backend, request):
    if backend == "chroma":
        pytest.importorskip("chromadb")
    provider = make_provider(tmp_path, backend=backend)
    request.addfinalizer(provider.shutdown)
    collection = provider.storage.current_collection()
    assert (collection is not None) == (backend == "chroma")
    now = datetime.now().astimezone()
    event_id = seed_event(provider, content='原始措辞只用于替换验证', importance=0.8)
    before = provider.sqlite.fetchone("SELECT raw, vec, tier, ttl_expire, promoted_at, pending_to FROM events WHERE id=?", (event_id,))

    assert provider.lifecycle.distill_promote(event_id, "这次表达承载着一项值得长期记得的选择。", now)
    row = provider.sqlite.fetchone(
        "SELECT raw, summary, keywords, vec, tier, ttl_expire, distilled_at, promoted_at, pending_to FROM events WHERE id=?",
        (event_id,),
    )
    assert provider.sqlite.fetchone("SELECT COUNT(*) FROM events WHERE id=?", (event_id,))[0] == 1
    assert row["summary"] == "这次表达承载着一项值得长期记得的选择。"
    assert row["raw"] == before["raw"]
    assert "原始措辞只用于替换验证" in row["raw"]
    assert row["vec"] != before["vec"]
    assert tuple(row[key] for key in ("tier", "ttl_expire", "promoted_at", "pending_to")) == tuple(before[key] for key in ("tier", "ttl_expire", "promoted_at", "pending_to"))
    assert row["distilled_at"]
    assert provider.sqlite.fetchone("SELECT raw FROM events_fts WHERE rowid=?", (event_id,))[0] == row["raw"]
    if collection is not None:
        indexed = collection.get(ids=[f"evt-{event_id}"])
        assert indexed["ids"] == [f"evt-{event_id}"]
        assert indexed["documents"] == [row["summary"]]
    assert provider.lifecycle.distill_promote(event_id, "不应再次覆盖", now) is False


def test_distill_is_end_to_end_idempotent_and_enforces_daily_cap(tmp_path):
    class Llm:
        calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            return types.SimpleNamespace(text=json.dumps({"summary": f"值得长期记得的事件概括{self.calls}", "duty": "memory", "source_kind": "unknown", "long_term": True, "covered_by": "", "significance": "重要经历"}, ensure_ascii=False))

    provider = make_provider(tmp_path, distill_daily_cap=2)
    llm = Llm()
    memories = tmp_path / "memories"
    memories.mkdir()
    for name in ("USER.md", "MEMORY.md"):
        (memories / name).write_text("", encoding="utf-8")
    provider.attach_runtime_llm(llm)
    ids = [
        seed_event(provider, content=f"高价值单次表达{index}", importance=importance)
        for index, importance in enumerate((0.7, 0.95, 0.8))
    ]
    first = provider.lifecycle.distill()
    assert [item["id"] for item in first["items"]] == [ids[1], ids[2]]
    assert llm.calls == 2
    assert provider.lifecycle.distill()["items"] == []
    assert llm.calls == 2
    assert provider.sqlite.fetchone("SELECT COUNT(*) FROM events WHERE distilled_at IS NOT NULL")[0] == 2


@pytest.mark.parametrize("backend", ["sqlite", "chroma"])
def test_distilled_event_is_searchable_gap_eligible_and_forgettable(tmp_path, backend, request):
    if backend == "chroma":
        pytest.importorskip("chromadb")
    provider = make_provider(tmp_path, backend=backend, user_id="u1")
    request.addfinalizer(provider.shutdown)
    collection = provider.storage.current_collection()
    assert (collection is not None) == (backend == "chroma")
    event_id = seed_event(provider, content='原始事件内容', importance=0.8)
    distilled = "想长期坚持一项带来安定感的选择"
    assert provider.lifecycle.distill_promote(event_id, distilled)
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET tier='long', ttl_expire=NULL WHERE id=?", (event_id,))
        conn.execute(
            "UPDATE events SET created_at=? WHERE id=?",
            ((datetime.now().astimezone() - timedelta(days=60)).isoformat(), event_id),
        )

    assert call(provider, "sanae_mem_search", {"query": "安定感"})["result"][0]["id"] == event_id
    assert provider.retrieval.search({"query": "无关查询", "top_k": 1, "gap_continue": True}, internal=True)[0]["tier"] == "long"
    if collection is not None:
        assert collection.get(ids=[f"evt-{event_id}"])["ids"] == [f"evt-{event_id}"]
    assert call(provider, "sanae_mem_forget", {"id": event_id})["result"]["deleted"] == 1
    if collection is not None:
        assert collection.get(ids=[f"evt-{event_id}"])["ids"] == []
    assert call(provider, "sanae_mem_add", {"content": "原始事件内容"})["result"]["stored"] is False


def test_distill_does_not_consume_scheduled_quota_or_change_read_gate(tmp_path):
    class Llm:
        def complete(self, messages, **kwargs):
            return types.SimpleNamespace(text=json.dumps({"summary": "用户希望长期坚持这项选择。", "duty": "memory", "source_kind": "unknown", "long_term": True, "covered_by": "", "significance": "重要经历"}, ensure_ascii=False))

    provider = make_provider(tmp_path, read_gate="scheduled", daily_scheduled_asks=1)
    memories = tmp_path / "memories"
    memories.mkdir()
    for name in ("USER.md", "MEMORY.md"):
        (memories / name).write_text("", encoding="utf-8")
    provider.attach_runtime_llm(Llm())
    seed_event(provider, content="想长期坚持这项选择", importance=0.8)
    before = provider.status_store.read()
    result = call(provider, "sanae_mem_distill", {})
    after = provider.status_store.read()
    assert result["ok"] and len(result["result"]["items"]) == 1
    assert provider.sqlite.fetchone("SELECT tier FROM events")[0] == "short"
    assert after.get("daily_scheduled_asks", 0) == before.get("daily_scheduled_asks", 0)
    assert not any((tmp_path / "data" / name).exists() for name in ("USER.md", "MEMORY.md"))
    assert provider.prefetch("长期坚持", scheduled=True).startswith("## 小事记忆")


@pytest.mark.parametrize("backend", ["sqlite", "chroma"])
def test_forget_returns_db_chroma_blacklist_and_parsed_target(tmp_path, backend, request):
    if backend == "chroma":
        pytest.importorskip("chromadb")
    provider = make_provider(tmp_path, backend=backend, user_id="u1")
    request.addfinalizer(provider.shutdown)
    collection = provider.storage.current_collection()
    assert (collection is not None) == (backend == "chroma")
    event_id = call(provider, "sanae_mem_add", {"content": "记住一件待删除的小事"})["result"]["id"]
    if collection is not None:
        assert collection.get(ids=[f"evt-{event_id}"])["ids"] == [f"evt-{event_id}"]
    result = call(
        provider,
        "sanae_mem_forget",
        {"id": event_id, "target": "user"},
    )["result"]
    assert result == {
        "deleted": 1,
        "ids": [event_id],
        "db_ids": [event_id],
        "chroma_ids": [f"evt-{event_id}"],
        "blacklisted": 1,
        "target": "user",
    }
    if collection is not None:
        assert collection.get(ids=[f"evt-{event_id}"])["ids"] == []
    assert provider.sqlite.fetchone("SELECT count(*) FROM forgotten")[0] == 1


def test_backup_paths(tmp_path):
    provider = make_provider(tmp_path)
    assert provider.backup_paths() == [
        str(tmp_path / "data" / "sanae.db"),
        str(tmp_path / "data" / "chroma"),
        str(tmp_path / "data" / "memes"),
    ]


def test_scheduled_tool_search_honors_gate_and_daily_cap(tmp_path):
    world = make_provider(tmp_path / "asked", read_gate="scheduled", daily_scheduled_asks=2)
    keyboard = call(world, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})["result"]["id"]
    first = call(world, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})
    assert first["ok"] and [item["id"] for item in first["result"]] == [keyboard]
    assert world.status_store.read()["daily_scheduled_asks"] == 1
    assert world.sqlite.fetchone("SELECT asked FROM events WHERE id=?", (keyboard,))[0] == 1

    again = call(world, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})
    assert again["ok"] and again["result"] == []
    assert world.status_store.read()["daily_scheduled_asks"] == 1

    capped = make_provider(tmp_path / "capped", read_gate="scheduled", daily_scheduled_asks=1)
    dental = call(capped, "sanae_mem_add", {"content": "这周预约了牙医复查", "importance": 0.8})["result"]["id"]
    hit = call(capped, "sanae_mem_search", {"query": "牙医复查", "scheduled": True})
    assert hit["ok"] and [item["id"] for item in hit["result"]] == [dental]
    assert capped.status_store.read()["daily_scheduled_asks"] == 1
    denied = call(capped, "sanae_mem_search", {"query": "牙医复查", "scheduled": True})
    assert denied["ok"] and denied["result"] == []
    assert capped.status_store.read()["daily_scheduled_asks"] == 1

    blocked = make_provider(tmp_path / "explicit-gate", read_gate="explicit")
    call(blocked, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    assert call(blocked, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})["result"] == []
    assert blocked.status_store.read().get("daily_scheduled_asks", 0) == 0


def test_explicit_paths_unchanged_under_scheduled_gate(tmp_path):
    provider = make_provider(tmp_path, read_gate="scheduled")
    keyboard = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})["result"]["id"]
    dental = call(provider, "sanae_mem_add", {"content": "这周预约了牙医复查", "importance": 0.8})["result"]["id"]

    found = call(provider, "sanae_mem_search", {"query": "静音键盘"})
    assert found["ok"] and found["result"][0]["id"] == keyboard
    assert provider.status_store.read().get("daily_scheduled_asks", 0) == 0

    block = provider.prefetch("还记得我说过的静音键盘吗")
    if _fallback_fail_closed():
        assert block == "", "散列回退不得自动注入；实际 %r" % (block,)
    else:
        assert block.startswith("## 小事记忆")
    assert provider.status_store.read().get("daily_scheduled_asks", 0) == 0

    scheduled_block = provider.prefetch("牙医复查", scheduled=True)
    assert scheduled_block.startswith("## 小事记忆")
    assert provider.status_store.read()["daily_scheduled_asks"] == 1
    rows = provider.sqlite.fetchall(
        "SELECT id, importance, asked, asked_via FROM events WHERE asked_via='scheduled'")
    assert len(rows) == 1, "定时轮只应带出一条，并且只给它记 scheduled 账：%s" % (rows,)
    picked_id, picked_importance, asked_flag, via = rows[0]
    assert asked_flag == 1 and via == "scheduled"
    best_importance = max(
        row[0] for row in provider.sqlite.fetchall("SELECT importance FROM events"))
    assert picked_importance == best_importance, "定时轮应在候选里取 importance 最高的那条"
    _ = dental

    provider.event_store.mark_asked([row[0] for row in
                         provider.sqlite.fetchall("SELECT id FROM events")])
    assert provider.prefetch("牙医复查", scheduled=True) == "", \
        "所有候选都已被读过 ⇒ 定时轮不得重复带出"

    capped = make_provider(tmp_path / "capped", read_gate="scheduled", daily_scheduled_asks=1)
    call(capped, "sanae_mem_add", {"content": "这周预约了牙医复查", "importance": 0.8})
    assert capped.prefetch("牙医复查", scheduled=True).startswith("## 小事记忆")
    assert capped.prefetch("牙医复查", scheduled=True) == "", "配额已用尽 ⇒ 定时轮不得再出"
    _ = (keyboard, dental)


def test_scheduled_first_ask_keeps_asked_mark(tmp_path):
    provider = make_provider(tmp_path / "same-day", read_gate="scheduled", daily_scheduled_asks=2)
    keyboard = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})["result"]["id"]
    hit = call(provider, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})
    assert hit["ok"] and [item["id"] for item in hit["result"]] == [keyboard]
    assert provider.sqlite.fetchone("SELECT asked FROM events WHERE id=?", (keyboard,))[0] == 1
    again = call(provider, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})
    assert again["ok"] and again["result"] == []
    assert provider.status_store.read()["daily_scheduled_asks"] == 1


def test_scheduled_cross_day_resets_asked_before_first_ask(tmp_path):
    provider = make_provider(tmp_path / "cross-day", read_gate="scheduled", daily_scheduled_asks=2)
    keyboard = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})["result"]["id"]
    hit = call(provider, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})
    assert hit["ok"] and [item["id"] for item in hit["result"]] == [keyboard]
    assert provider.sqlite.fetchone("SELECT asked FROM events WHERE id=?", (keyboard,))[0] == 1
    yesterday = (datetime.now().astimezone() - timedelta(days=1)).date().isoformat()
    _set_status_fields(provider, date=yesterday, last_asked_reset_date=yesterday)
    next_day = call(provider, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})
    assert next_day["ok"] and [item["id"] for item in next_day["result"]] == [keyboard]
    assert provider.sqlite.fetchone("SELECT asked FROM events WHERE id=?", (keyboard,))[0] == 1
    assert provider.status_store.read()["daily_scheduled_asks"] == 1


def test_compact_resets_asked_without_scheduled_prefetch(tmp_path):
    provider = make_provider(tmp_path, read_gate="scheduled")
    event_id = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘"})["result"]["id"]
    provider.event_store.mark_asked([event_id])
    now = datetime.now().astimezone()
    yesterday = (now - timedelta(days=1)).date().isoformat()
    _set_status_fields(
        provider, last_compactor_date=yesterday, last_asked_reset_date=yesterday)

    provider.lifecycle.compact(now)

    assert provider.sqlite.fetchone("SELECT asked FROM events WHERE id=?", (event_id,))[0] == 0
    assert provider.status_store.read()["last_asked_reset_date"] == now.date().isoformat()


def test_injection_recalls_event_asked_more_than_24_hours_ago(tmp_path):
    provider = make_provider(tmp_path)
    event_id = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘"})["result"]["id"]
    old_asked_at = (datetime.now().astimezone() - timedelta(hours=25)).isoformat()
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET asked=1, asked_at=? WHERE id=?", (old_asked_at, event_id))

    results = provider.retrieval.search({"query": "静音键盘"}, for_injection=True)

    assert [item["id"] for item in results] == [event_id]


def test_explicit_past_bypasses_recent_asked_filter(tmp_path):
    provider = make_provider(tmp_path, read_gate="explicit")
    event_id = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘"})["result"]["id"]
    provider.event_store.mark_asked([event_id])

    block = provider.prefetch("还记得我说过的静音键盘吗")

    if _fallback_fail_closed():
        assert block == "", "散列回退不得自动注入；实际 %r" % (block,)
    else:
        assert "静音键盘" in block


def test_scheduled_prefetch_records_attempt_without_success(tmp_path):
    provider = make_provider(tmp_path, read_gate="scheduled")

    assert provider.prefetch("不存在的旧事", scheduled=True) == ""

    status = provider.status_store.read()
    assert status["daily_scheduled_attempts"] == 1
    assert status.get("daily_scheduled_asks", 0) == 0


def test_emotion_relation_ttl_is_90_days_schedule_and_trivial_unchanged(tmp_path):
    scenarios = [
        ("emotion", "最近感到有些焦虑", "情绪", 90),
        ("relation", "最近更信任和依靠家人了", "关系", 90),
        ("schedule-first", "明天陪妈妈复查，我更信任她了", "关系", 7),
        ("trivial", "我想买一把静音键盘", "购物", 14),
    ]
    for name, text, expected_topic, expected_days in scenarios:
        provider = make_provider(tmp_path / name)
        provider.sync_turn(text, "嗯嗯", session_id="session-a")
        row = provider.sqlite.fetchone("SELECT topic, ttl_expire FROM events")
        assert row is not None, name
        assert row[0] == expected_topic, (name, row[0])
        approx_days = (datetime.fromisoformat(row[1]) - datetime.now().astimezone()).total_seconds() / 86400
        assert expected_days - 1 < approx_days < expected_days + 1, (name, approx_days)


def test_emotion_ttl_days_configurable_and_default_90(tmp_path):
    provider = make_provider(tmp_path, emotion_ttl_days=45)
    provider.sync_turn("最近感到有些焦虑", "嗯嗯", session_id="session-a")
    expire = provider.sqlite.fetchone("SELECT ttl_expire FROM events")[0]
    approx_days = (datetime.fromisoformat(expire) - datetime.now().astimezone()).total_seconds() / 86400
    assert 44 < approx_days < 46

    schema = {item["key"]: item for item in provider.get_config_schema()}
    assert schema["emotion_ttl_days"]["default"] == 90
    assert schema["emotion_ttl_days"]["minimum"] == 1


def test_scheduled_search_recalls_summarized_long_and_important_but_not_old_short(tmp_path):
    important = make_provider(tmp_path / "important", read_gate="scheduled", daily_scheduled_asks=5)
    iid = seed_event(important, content='珍视这段来之不易的信任', importance=0.9)
    with important.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET tier='important', ttl_expire=NULL WHERE id=?", (iid,))
    hit = call(important, "sanae_mem_search", {"query": "来之不易的信任", "scheduled": True})
    assert hit["ok"] and [item["id"] for item in hit["result"]] == [iid]

    long_ev = make_provider(tmp_path / "long", read_gate="scheduled", daily_scheduled_asks=5)
    lid = seed_event(long_ev, content='想长期坚持早起冥想', importance=0.7)
    with long_ev.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET tier='long', ttl_expire=NULL WHERE id=?", (lid,))
    hit = call(long_ev, "sanae_mem_search", {"query": "早起冥想", "scheduled": True})
    assert hit["ok"] and [item["id"] for item in hit["result"]] == [lid]

    distilled = make_provider(tmp_path / "distilled", read_gate="scheduled", daily_scheduled_asks=5)
    did = seed_event(distilled, content='想长期坚持早起冥想', importance=0.8)
    assert distilled.lifecycle.distill_promote(did, "长期坚持早起冥想的习惯")
    with distilled.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET tier='long', ttl_expire=NULL WHERE id=?", (did,))
        conn.execute(
            "UPDATE events SET created_at=? WHERE id=?",
            ((datetime.now().astimezone() - timedelta(days=60)).isoformat(), did),
        )
    assert distilled.sqlite.fetchone("SELECT tier FROM events WHERE id=?", (did,))[0] == "long"
    hit = call(distilled, "sanae_mem_search", {"query": "冥想习惯", "scheduled": True})
    assert hit["ok"] and [item["id"] for item in hit["result"]] == [did]

    old_short = make_provider(tmp_path / "old-short", read_gate="scheduled")
    oid = seed_event(old_short, content='记住一个很久以前的小事', importance=0.6)
    past = (datetime.now().astimezone() - timedelta(days=60)).isoformat()
    with old_short.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET created_at=?, ttl_expire=? WHERE id=?", (past, past, oid))
    hit = call(old_short, "sanae_mem_search", {"query": "很久以前", "scheduled": True})
    assert hit["ok"] and hit["result"] == []
    assert old_short.status_store.read().get("daily_scheduled_asks", 0) == 0


_SYSTEM_META_SAMPLES = (
    "帮我查一下 deepseek harness 有没有评估 skill",
    "hermes gateway 的日志要重定向到哪个文件",
    "我要给你增添情绪",
    "帮早苗加一个记忆温度模块吧",
    "原理：情绪=会被记住、因此影响后续言行的事件",
    "先整理今日份的事件流，把今天的小事按重要度收拢",
    "早苗的记忆系统要不要再调一下蒸馏层的参数",
    "整理一下怎么这么多？USER.md 不是记录用户最重要的事情吗",
    "user 是重要记忆吧，我记得",
    "MEMORY.md 里的内容也要一起注入吗",
)


@pytest.mark.parametrize("text", _SYSTEM_META_SAMPLES)
def test_sync_turn_rejects_system_meta_talk(tmp_path, text):
    provider = make_provider(tmp_path / "meta")
    provider.sync_turn(text, "嗯嗯", session_id="session-a")
    assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 0
    assert not (tmp_path / "data" / "sample.log").exists() or "harness" not in (tmp_path / "data" / "sample.log").read_text(
        encoding="utf-8"
    )


_LIFE_WORK_KEEP_SAMPLES = (
    "记住：周末去超市买镂空速干拖鞋，汗脚要透气防臭",
    "最近有点焦虑，下周要交那批零件的图纸",
    "帮我设个提醒：每天凌晨一点叫我睡觉",
    "帮我评估一下这套夹具的装夹方案行不行，下周要加工那批轴",
    "今天把轴径公差又超了0.02，明天得返工，帮我记一下",
    "你感性那层其实早沉了，不过没关系，看到你开心我也开心，这两年还是很喜欢有你在",
)


@pytest.mark.parametrize("text", _LIFE_WORK_KEEP_SAMPLES)
def test_sync_turn_keeps_life_and_mechanical_work_talk(tmp_path, text):
    provider = make_provider(tmp_path / "keep")
    provider.sync_turn(text, "嗯嗯", session_id="session-a")
    rows = provider.sqlite.fetchall("SELECT topic, importance FROM events")
    assert len(rows) == 1, text
    assert float(rows[0]["importance"]) >= 0.45


def test_distill_candidate_skips_new_meta_phrases_and_english_tokens(tmp_path):
    provider = make_provider(tmp_path, distill_daily_cap=10)
    good = seed_event(provider, content='一次值得记住的生活小事', importance=0.7)
    meta_phrase = seed_event(provider, content='USER.md 不是用来记录用户最重要的事情吗', importance=0.99)
    meta_token = seed_event(provider, content='deepseek harness 有没有评估 skill', importance=0.99)
    meta_principle = seed_event(provider, content='原理：情绪=会被记住、因此影响后续言行的事件', importance=0.99)
    ids = {item["id"] for item in provider.lifecycle.compact()["distill_candidates"]}
    assert good in ids
    assert not ids & {meta_phrase, meta_token, meta_principle}


_NOISE_GATE1_SAMPLES = (
    "用户提及早苗正为其按摩，语气中带着意外与确认",
    "她转述时语气里带着一丝迟疑",
    "对方回应时语气透着一点无奈",
    "这条属于人格/时段理解常识，待 teaching.md 落地后分流进教学区",
    "本条属于日常分类，等 schema.yaml 落地后分流到配置区",
    "对比一下 glm-5.3-flash 和 deepseek v4-flash 的价格",
    "麻烦比较一下这两款显示器的性价比",
    "帮我看看这把静音键盘多少钱",
    "请查一下这家快递的收费标准",
    "获取当前时间，好烦啊",
    "先查询一下系统时间，再去跑脚本",
    "毕竟这种还是很重要的，必须得每次留档",
    "这种事很重要，一定要存档记录在案",
    "总之这类记录必须一律备案",
)


@pytest.mark.parametrize("text", _NOISE_GATE1_SAMPLES)
def test_sync_turn_rejects_six_noise_shapes_at_write_gate(tmp_path, text):
    provider = make_provider(tmp_path / "noise")
    provider.sync_turn(text, "嗯嗯", session_id="session-a")
    assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 0
    assert not (tmp_path / "data" / "sample.log").exists()


_KEEP_WITH_NEAR_MISS_WORDS = (
    "这份图纸要留档，帮我记住一下",
    "老板要求客户每次对接完都要留档，觉得有点烦",
    "帮我记一下，那键盘的价格在五百以内",
    "明天帮我查一下去杭州的火车票",
    "我想买的静音键盘今天对比了半天，终于下单了",
    "帮我写个脚本获取当前时间，顺便记一下用法",
)


@pytest.mark.parametrize("text", _KEEP_WITH_NEAR_MISS_WORDS)
def test_sync_turn_keeps_near_miss_neutral_expressions(tmp_path, text):
    provider = make_provider(tmp_path / "keep")
    provider.sync_turn(text, "嗯嗯", session_id="session-a")
    rows = provider.sqlite.fetchall("SELECT topic, importance FROM events")
    assert len(rows) == 1, text
    assert float(rows[0]["importance"]) >= 0.45


def test_distill_candidates_skip_six_noise_rows_but_keep_good(tmp_path):
    provider = make_provider(tmp_path, distill_daily_cap=10)
    good = seed_event(provider, content='一次值得记住的生活小事', importance=0.7)
    footnote = seed_event(provider, content='毕竟这种还是很重要的，必须得每次留档', importance=0.99)
    no_deictic = seed_event(provider, content='重要文件每次都要留档', importance=0.99)
    ask = seed_event(provider, content='帮我看看这把静音键盘多少钱', importance=0.99)
    observe = seed_event(provider, content='用户提及早苗正为其按摩，语气中带着意外与确认', importance=0.99)
    ids = {item["id"] for item in provider.lifecycle.compact()["distill_candidates"]}
    assert good in ids
    assert not ids & {footnote, no_deictic, ask, observe}


def test_register_resolves_host_llm_through_collector(tmp_path):
    calls = []

    candidate_json = json.dumps({
        "summary": "她把这次选择视为长期方向",
        "duty": "memory",
        "source_kind": "event_promotion",
        "long_term": True,
        "covered_by": "",
        "significance": "这是一次长期方向的选择",
    }, ensure_ascii=False)

    class FakeLlm:
        def complete(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return types.SimpleNamespace(text=candidate_json)

    class HostCtx:
        llm = FakeLlm()

    class Collector:
        provider = None

        def __init__(self):
            self._context = HostCtx()
            self.aux_task = None

        def register_command(self, *args):
            pass

        def register_memory_provider(self, provider):
            self.provider = provider

        def register_auxiliary_task(self, *args, **kwargs):
            self.aux_task = (args, kwargs)

        @property
        def llm(self):
            return self._context.llm

        def register_hook(self, *args):
            pass

    collector = Collector()
    register_plugin(collector)
    provider = collector.provider
    assert provider is not None
    assert provider.name == "sanae_mem"
    assert collector.aux_task is not None
    assert isinstance(provider.runtime_llm(), FakeLlm)
    provider.initialize("session-x", hermes_home=str(tmp_path), platform="test")
    assert provider.llm_guard.review_candidate("想认真坚持这条路") == {
        "summary": "她把这次选择视为长期方向",
        "duty": "memory",
        "source_kind": "event_promotion",
        "long_term": True,
        "covered_by": "",
        "significance": "这是一次长期方向的选择",
    }
    assert calls and calls[0][1]["task"] == "sanae_extract"
    provider.shutdown()


def test_register_registers_provider_before_auxiliary_failure():
    class Collector:
        provider = None
        hooks = None

        def register_command(self, *args):
            pass

        def register_memory_provider(self, provider):
            self.provider = provider

        def register_hook(self, name, callback):
            self.hooks = (self.hooks or []) + [(name, callback)]

        def register_auxiliary_task(self, *args, **kwargs):
            raise RuntimeError("not available in test")

    collector = Collector()
    register_plugin(collector)
    assert collector.provider is not None
    try:
        assert collector.provider.name == "sanae_mem"
        assert collector.hooks is not None
        assert [name for name, _cb in collector.hooks] == [
            "post_gateway_admission", "pre_llm_call", "pre_tool_call", "pre_tool_call",
            "pre_tool_call", "transform_tool_result", "pre_llm_call", "pre_llm_call", "post_llm_call",
        ]
        assert [cb.__name__ for _, cb in collector.hooks] == [
            "guard_public_local_references", "guard_builtin_memory_prompt", "guard_public_memory_tool",
            "guard_memory_write", "guard_controlled_memory_write", "remind_memory_result",
            "handle_pre_llm_call", "append_private_heart", "observe_final_text",
        ]
        assert collector.hooks[6][1].__self__ is collector.provider
    finally:
        collector.provider.shutdown()


def test_register_llm_resolution_failure_stays_fail_open():
    class Collector:
        provider = None

        def register_command(self, *args):
            pass

        def register_memory_provider(self, provider):
            self.provider = provider

        def register_auxiliary_task(self, *args, **kwargs):
            pass

        @property
        def llm(self):
            raise RuntimeError("host LLM context unavailable")

        def register_hook(self, *args):
            pass

    collector = Collector()
    register_plugin(collector)
    assert collector.provider is not None
    try:
        assert collector.provider.name == "sanae_mem"
        assert collector.provider.runtime_llm() is None
        assert collector.provider.llm_guard.review_candidate("值得记住的事") is None
    finally:
        collector.provider.shutdown()


_SECRET_A = "用户甲托付给我的那件重要的事，我一定替他记住"
_SECRET_B = "用户乙托付给我的另一件重要的事，只跟他有关"


def test_owner_key_is_stable_prefixed_and_hashed_not_raw():
    def _acct(platform: str, *, user_id: str = "", user_id_alt: str = "") -> str:
        return source_context.scope_account_key(
            source_context.ChatScope("private", platform=platform, actor=user_id, actor_alt=user_id_alt)
        )

    key = _acct("qqbot", user_id="openid-abc")
    assert key == _acct("qqbot", user_id="openid-abc")
    assert len(key) == 16 and all(character in "0123456789abcdef" for character in key)
    assert "openid-abc" not in key
    assert _acct("qqbot", user_id="openid-abc") != _acct("telegram", user_id="openid-abc")
    assert _acct("qqbot", user_id="primary", user_id_alt="alternate") == _acct(
        "qqbot", user_id="alternate"
    )
    assert _acct("cron") == ""
    assert _acct("cli", user_id="", user_id_alt="") == ""


def test_sync_turn_stores_hashed_owner_and_never_exposes_it(tmp_path):
    provider = make_provider(tmp_path, user_id="openid-aaa")
    provider.sync_turn(_SECRET_A, "嗯", session_id="session-a")
    stored = provider.sqlite.fetchall("SELECT owner_key FROM events")
    assert len(stored) == 1
    expected_owner = source_context.scope_account_key(
        source_context.ChatScope("private", platform="test", actor="openid-aaa")
    )
    assert stored[0]["owner_key"] == expected_owner
    assert expected_owner not in provider.prefetch("那件重要的事")
    assert expected_owner not in provider.system_prompt_block()


def test_second_owner_cannot_read_first_owners_private_memory(tmp_path):
    owner_a = make_provider(tmp_path, user_id="openid-aaa")
    owner_a.sync_turn(_SECRET_A, "嗯", session_id="session-a")
    event_id = owner_a.sqlite.fetchone(
        "SELECT id FROM events ORDER BY id DESC LIMIT 1")["id"]
    owner_b = make_provider(tmp_path, user_id="openid-bbb")

    assert _provider_owner(owner_b) and _provider_owner(owner_b) != _provider_owner(owner_a)
    assert owner_b.retrieval.search({"query": "托付 重要的事"}, for_injection=True) == []
    assert owner_b.retrieval.search({"query": "托付 重要的事", "gap_continue": True}) == []
    assert "替他记住" not in owner_b.prefetch("那件重要的事")
    _enter_test_turn("session-a", {"user_id": "openid-aaa"})
    assert owner_a.retrieval.search({"query": "托付 重要的事"}, for_injection=False)[0]["raw"].endswith("替他记住")

    owner_a.event_store.set_shared_view(event_id, "之前和人聊过托付一件重要的事")
    _enter_test_turn("session-a", {"user_id": "openid-bbb"})
    shared = owner_b.retrieval.search({"query": "托付 重要的事"}, for_injection=True)
    assert shared and shared[0]["shared"] is True
    assert shared[0]["raw"] == ""
    assert shared[0]["shared_view"] == "之前和人聊过托付一件重要的事"


def test_same_owner_still_recalls_across_sessions(tmp_path):
    provider = make_provider(tmp_path, user_id="openid-aaa")
    provider.sync_turn(_SECRET_A, "嗯", session_id="session-a")
    restarted = make_provider(tmp_path, user_id="openid-aaa")
    restarted.initialize("session-fresh", hermes_home=str(tmp_path), platform="test", user_id="openid-aaa")
    found = restarted.retrieval.search({"query": "托付 重要的事"}, for_injection=True)
    assert found and found[0]["raw"].endswith("替他记住")


def test_unowned_rows_are_not_claimed_by_a_bound_owner(tmp_path):
    provider = make_provider(tmp_path, user_id="openid-aaa")
    provider.sync_turn(_SECRET_A, "嗯", session_id="session-a")
    unowned = seed_row(provider, "系统级维护记录", 0.9, session_id="cron_x",
                       now=datetime.now().astimezone())
    assert unowned is not None
    assert provider.sqlite.fetchone(
        "SELECT owner_key FROM events WHERE id=?", (unowned,)
    )["owner_key"] == ""

    scoped = provider.retrieval.search({"query": "托付 重要的事"}, for_injection=True)
    assert {item["id"] for item in scoped} != {row[0] for row in provider.sqlite.fetchall("SELECT id FROM events")}
    assert unowned not in {item["id"] for item in scoped}
    assert provider.sqlite.fetchone("SELECT owner_key FROM events WHERE id=?", (unowned,))[0] == ""
    assert len(provider.query_events(datetime.now().astimezone().isoformat())) == 1
    assert len(
        provider.query_events(datetime.now().astimezone().isoformat(), unscoped=True)
    ) == 2


def test_cron_does_not_read_other_owner_private_but_can_write_system_level(tmp_path):
    owner = make_provider(tmp_path, user_id="openid-aaa")
    owner.sync_turn(_SECRET_A, "嗯", session_id="session-a")
    now = datetime.now().astimezone()
    assert owner.query_events(now.isoformat(), read_context=True)[0]["raw"].endswith(_SECRET_A)

    cron = make_provider(tmp_path, context="cron")
    assert cron.query_events(now.isoformat(), read_context=True) == []
    assert cron.retrieval.search({"query": "托付 重要的事", "gap_continue": True}) == []
    system_level = cron.ingest.write_event(
        "夜里整理出来的一条系统级小事记录", "cron_job_20260910",
        importance=0.9, source="explicit",
    )
    assert system_level is not None
    written = cron.sqlite.fetchone("SELECT owner_key FROM events WHERE id=?", (system_level,))
    assert written["owner_key"] == ""
    assert cron.retrieval.search({"query": "系统级"}, for_injection=True)
    owned_ids = {item["id"] for item in owner.retrieval.search({"query": "系统级"}, for_injection=True)}
    assert system_level not in owned_ids
    assert cron.sqlite.fetchone("SELECT owner_key FROM events WHERE id=?", (system_level,))[0] == ""


def test_subagent_context_never_claims_ownership(tmp_path):
    provider = make_provider(tmp_path, user_id="openid-aaa")
    provider.sync_turn(_SECRET_A, "嗯", session_id="session-a")
    now = datetime.now().astimezone().isoformat()
    assert provider.query_events(now, read_context=True)[0]["raw"].endswith(_SECRET_A)

    provider.initialize(
        "session-a", hermes_home=str(tmp_path), platform="test",
        agent_context="subagent", user_id="openid-aaa",
    )
    assert provider.query_events(now, read_context=True) == []


def test_legacy_unknown_rows_stay_unclaimed_and_private(tmp_path):
    legacy = make_provider(tmp_path)
    seed_row(legacy, _SECRET_A, 0.9, topic="其他", session_id="legacy-session",
             now=datetime.now().astimezone())
    legacy.shutdown()

    provider = make_provider(tmp_path, user_id="openid-aaa")
    now = datetime.now().astimezone().isoformat()
    assert provider.query_events(now, read_context=True) == []
    assert provider.sqlite.fetchone(
        "SELECT owner_key FROM events")["owner_key"] == ""
    assert provider.retrieval.search({"query": "托付 重要的事"}, for_injection=True) == []
    assert provider.sqlite.fetchone("SELECT owner_key FROM events")[0] == ""
    assert provider.query_events(now, unscoped=True)

    other = make_provider(tmp_path, user_id="openid-bbb")
    assert other.retrieval.search({"query": "托付 重要的事"}, for_injection=True) == []
    assert other.sqlite.fetchone("SELECT owner_key FROM events")[0] == ""
    other.sync_turn(_SECRET_B, "嗯", session_id="session-a")
    visible = other.query_events(now, read_context=True)
    assert len(visible) == 1
    assert visible[0]["raw"].endswith(_SECRET_B)

    _enter_test_turn("session-a", {"user_id": "openid-aaa"})
    provider.on_turn_start(1, "托付 重要的事")
    assert provider.query_events(now, read_context=True) == []


def test_owner_key_column_and_index_share_the_migration_path(tmp_path):
    data_dir = tmp_path / "data"
    provider = make_provider(tmp_path)
    schema = {row[1]: row for row in provider.sqlite.fetchall("PRAGMA table_info(events)")}
    assert "owner_key" in schema
    assert schema["owner_key"][3] == 1 and schema["owner_key"][4] == "''"
    indexes = {row[1] for row in provider.sqlite.fetchall("PRAGMA index_list(events)")}
    assert "idx_events_owner_key" in indexes

    provider.shutdown()
    backups_after_first_open = sorted(p.name for p in data_dir.glob("sanae.db.bak-p6-*"))

    restarted = make_provider(tmp_path)
    assert {row[1] for row in restarted.sqlite.fetchall("PRAGMA index_list(events)")} >= {"idx_events_owner_key"}
    assert sorted(p.name for p in data_dir.glob("sanae.db.bak-p6-*")) == backups_after_first_open






def test_owner_key_is_platform_scoped_and_nickname_independent():
    def _from_envelope(platform: str, user_id: str, user_name: str = "") -> str:
        return source_context.chat_scope_for(
            types.SimpleNamespace(
                platform=platform,
                chat_type="dm",
                chat_id="c1",
                session_key="s1",
                sender_id=user_id,
                sender_name=user_name,
            )
        ).owner

    assert _from_envelope("qqbot", "same-id") != _from_envelope("bilibili", "same-id")
    assert _from_envelope("qqbot", "a") != _from_envelope("qqbot", "b")
    assert _from_envelope("qqbot", "same-id", user_name="任何昵称") == _from_envelope("qqbot", "same-id")








def test_duplicate_reinforcement_is_scoped_to_the_current_owner(tmp_path):
    owner_a = make_provider(tmp_path, user_id="openid-aaa")
    owner_a.sync_turn("我很信任你，请记住这件重要的事", "嗯", session_id="shared-session")
    row_a = owner_a.sqlite.fetchone(
        "SELECT id, importance, updated_at FROM events WHERE owner_key=?", (_provider_owner(owner_a),)
    )

    owner_b = make_provider(tmp_path, user_id="openid-bbb")
    owner_b.sync_turn("我也信任你，请记住这件重要的安排", "嗯", session_id="shared-session")
    rows = owner_b.sqlite.fetchall(
        "SELECT id, owner_key, importance, updated_at FROM events ORDER BY id"
    )

    assert len(rows) == 2
    assert {row["owner_key"] for row in rows} == {_provider_owner(owner_a), _provider_owner(owner_b)}
    unchanged_a = next(row for row in rows if row["id"] == row_a["id"])
    assert (unchanged_a["importance"], unchanged_a["updated_at"]) == (
        row_a["importance"], row_a["updated_at"]
    )
    cutoff = (datetime.now().astimezone() - timedelta(hours=24)).isoformat()
    assert {row["id"] for row in owner_b.event_store.find_duplicate_candidates("shared-session", cutoff)} == {
        next(row["id"] for row in rows if row["owner_key"] == _provider_owner(owner_b))
    }
    assert {row["id"] for row in owner_b.event_store.find_duplicate_candidates(
        "shared-session", cutoff, unscoped=True
    )} == {row["id"] for row in rows}




def test_tool_add_keeps_attribution_and_other_owner_needs_shared_view(tmp_path):
    owner = make_provider(tmp_path, user_id="openid-aaa")
    added = call(owner, "sanae_mem_add", {"content": _SECRET_A, "importance": 0.9})
    event_id = added["result"]["id"]

    row = owner.sqlite.fetchone("SELECT owner_key FROM events WHERE id=?", (event_id,))
    assert row["owner_key"] == _provider_owner(owner)
    assert [item["id"] for item in owner.retrieval.search(
        {"query": "托付 重要的事"}, for_injection=True
    )] == [event_id]

    other = make_provider(tmp_path, user_id="openid-bbb")
    assert other.retrieval.search({"query": "托付 重要的事"}, for_injection=True) == []
    assert [row["id"] for row in other.query_events(
        datetime.now().astimezone().isoformat(), unscoped=True
    )] == [event_id]

    owner.event_store.set_shared_view(event_id, "之前和人聊过托付一件重要的事")
    hits = other.retrieval.search({"query": "托付 重要的事"}, for_injection=True)
    assert [item["id"] for item in hits] == [event_id]
    assert hits[0]["attribution"] == "shared_experience"
    assert hits[0]["raw"] == ""






def test_pre_compress_does_not_run_maintenance(tmp_path):
    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()

    expired = call(provider, "sanae_mem_add", {"content": "记住已经过期的考试"})["result"]["id"]
    keep = call(provider, "sanae_mem_add", {"content": "记住重复的静音键盘", "importance": 0.8})["result"]["id"]
    duplicate = call(provider, "sanae_mem_add", {"content": "记住另一个重复事件", "importance": 0.9})["result"]["id"]
    old = call(provider, "sanae_mem_add", {"content": "记住很久以前的课程"})["result"]["id"]

    keywords = provider.sqlite.fetchone("SELECT keywords FROM events WHERE id=?", (keep,))[0]
    with provider.sqlite.transaction() as conn:
        conn.execute(
            "UPDATE events SET ttl_expire=? WHERE id=?",
            ((now - timedelta(days=1)).isoformat(), expired),
        )
        conn.execute("UPDATE events SET keywords=? WHERE id=?", (keywords, duplicate))
        conn.execute(
            "UPDATE events SET created_at=?, confidence=0.8, asked=1 WHERE id=?",
            ((now - timedelta(days=31)).isoformat(), old),
        )
    yesterday = (now - timedelta(days=1)).date().isoformat()
    _set_status_fields(
        provider, date=now.date().isoformat(), last_compactor_date=yesterday,
        last_asked_reset_date=yesterday, daily_compactor_runs=3)

    def snapshot():
        rows = [tuple(r) for r in provider.sqlite.fetchall(
            "SELECT id, tier, expired_at, confidence, asked, blacklist, owner_key, "
            "cross_session_hits, pending_to FROM events ORDER BY id")]
        return {
            "rows": rows,
            "asked": dict(provider.sqlite.fetchall("SELECT asked, count(*) FROM events GROUP BY 1")),
            "tier": dict(provider.sqlite.fetchall("SELECT tier, count(*) FROM events GROUP BY 1")),
        }

    before = snapshot()
    status_before = provider.status_store.read()

    assert provider.on_pre_compress([]) == ""

    after = snapshot()
    assert after["rows"] == before["rows"], "events row set/columns must be untouched"
    assert after["asked"] == before["asked"], "asked must NOT be reset by the compress path"
    assert after["tier"] == before["tier"]
    status_after = provider.status_store.read()
    assert status_after.get("last_compactor_at") == status_before.get("last_compactor_at")
    assert status_after.get("daily_compactor_runs") == status_before.get("daily_compactor_runs")
    assert status_after.get("last_compactor_date") == status_before.get("last_compactor_date")
    assert status_after["last_asked_reset_date"] == yesterday, "no _daily_rollover from compress"
    assert provider.sqlite.fetchone("SELECT count(*) FROM forgotten")[0] == 0
    assert provider.sqlite.fetchone(
        "SELECT expired_at FROM events WHERE id=?", (expired,)
    )[0] is None


def test_pre_compress_records_observation_key(tmp_path):
    provider = make_provider(tmp_path)
    today = datetime.now().astimezone().date().isoformat()

    assert provider.on_pre_compress([{"role": "user", "content": "irrelevant"}]) == ""
    status = provider.status_store.read()
    assert status["pre_compress_count_today"] == 1
    assert status["pre_compress_date"] == today
    datetime.fromisoformat(status["last_pre_compress_at"])
    first_at = status["last_pre_compress_at"]

    assert provider.on_pre_compress([]) == ""
    status = provider.status_store.read()
    assert status["pre_compress_count_today"] == 2
    assert status["last_pre_compress_at"] >= first_at

    _set_status_fields(
        provider, pre_compress_date=(datetime.now().astimezone()
                                     - timedelta(days=1)).date().isoformat())
    assert provider.on_pre_compress([]) == ""
    status = provider.status_store.read()
    assert status["pre_compress_count_today"] == 1
    assert "last_compactor_at" not in status
    assert "last_compactor_date" not in status


def test_pre_compress_does_not_disable_daily_reset_gates(tmp_path):
    provider = make_provider(tmp_path, read_gate="scheduled")
    now = datetime.now().astimezone()
    yesterday = (now - timedelta(days=1)).date().isoformat()

    _set_status_fields(
        provider, date=yesterday, daily_writes=7, daily_scheduled_asks=2,
        daily_scheduled_attempts=2, distill_errors_today=1, daily_compactor_runs=3,
        last_compactor_date=yesterday)

    assert provider.on_pre_compress([]) == ""

    s = provider.status_store.read()
    assert s["date"] == yesterday, "compress must NOT stamp today's date"
    assert s["pre_compress_date"] == now.date().isoformat()
    assert s["daily_scheduled_asks"] == 2
    assert s["daily_writes"] == 7

    provider.lifecycle.compact(now)
    s = provider.status_store.read()
    assert s["daily_scheduled_asks"] == 0, "compact() day-change reset must still fire"
    assert s["daily_writes"] == 0
    assert s["distill_errors_today"] == 0
    assert provider.read_gate.scheduled_allowed() is True, "08:00 scheduled recall must not be blocked"

    event_id = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘"})["result"]["id"]
    hits = provider.retrieval.search({"query": "静音键盘", "scheduled": True, "top_k": 5}, for_injection=True)
    assert event_id in [item["id"] for item in hits], "scheduled path must still surface 存量记忆"


_CRON_SESSION = "cron_f325f4857f91_20260917_230000"
_SIGNAL_TURN = "今天有点焦虑，记住我下周有一场重要考试"


def test_cron_turn_ingests_nothing_and_records_no_user_activity(tmp_path):
    provider = make_provider(tmp_path)
    status_path = tmp_path / "data" / "status.json"

    provider.sync_turn(_SIGNAL_TURN, "嗯", session_id=_CRON_SESSION)
    assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 0
    assert not status_path.exists(), "cron 轮不得走到 _record_user_active()"

    provider.sync_turn(_SIGNAL_TURN, "嗯", session_id="session-a")
    assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 1
    assert "last_user_active_at" in json.loads(status_path.read_text(encoding="utf-8"))


def test_cron_turn_still_honours_her_explicit_memory_tools(tmp_path):
    provider = make_provider(tmp_path)
    provider.on_session_switch(_CRON_SESSION)

    added = call(provider, "sanae_mem_add", {"content": "她在整理夜自己决定留下的一条"})
    assert added["ok"] and added["result"]["id"]
    assert provider.sqlite.fetchone("SELECT count(*) FROM events")[0] == 1


def test_cron_gate_does_not_close_the_distill_path(tmp_path):
    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()
    event_id = seed_event(provider, content='一次很有意义的尝试', importance=0.8)

    assert provider.lifecycle.distill_promote(event_id, "这次尝试值得长期记得。", now)
    row = provider.sqlite.fetchone(
        "SELECT tier, promoted_reason, summary, distilled_at FROM events WHERE id=?", (event_id,)
    )
    assert (row["tier"], row["promoted_reason"]) == ("short", "")
    assert row["summary"] == "这次尝试值得长期记得。" and row["distilled_at"]


def _force_same_keywords(provider, keyword_source_id, *target_ids):
    keywords = provider.sqlite.fetchone(
        "SELECT keywords FROM events WHERE id=?", (keyword_source_id,)
    )[0]
    assert keywords
    with provider.sqlite.transaction() as conn:
        for target in target_ids:
            conn.execute("UPDATE events SET keywords=? WHERE id=?", (keywords, target))
    return keywords


def test_compact_never_deletes_a_long_row_as_a_duplicate(tmp_path):
    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()
    keeper = seed_event(provider, content='记住我有一把静音键盘', importance=0.55)
    duplicate = seed_event(provider, content='又提起那把键盘的事情', importance=0.9)
    with provider.sqlite.transaction() as conn:
        conn.execute(
            "UPDATE events SET tier='long', ttl_expire=NULL WHERE id=?", (keeper,)
        )
    _force_same_keywords(provider, keeper, duplicate)

    result = provider.lifecycle.compact(now)

    remaining = {int(row["id"]) for row in provider.sqlite.fetchall("SELECT id FROM events")}
    assert keeper in remaining
    assert duplicate not in remaining
    row = provider.sqlite.fetchone(
        "SELECT tier, importance FROM events WHERE id=?", (keeper,)
    )
    assert row["tier"] == "long"
    assert float(row["importance"]) == pytest.approx(0.9)
    assert (result["merged"], result["merge_groups_skipped"]) == (1, 0)


def test_compact_skips_a_group_holding_two_long_lived_rows(tmp_path):
    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()
    long_id = seed_event(provider, content='记住我有一把静音键盘', importance=0.5)
    pending_id = seed_event(provider, content='另一条候选里的事', importance=0.6)
    plain_id = seed_event(provider, content='今天路上看到一只猫', importance=0.5)
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET tier='long', ttl_expire=NULL WHERE id=?", (long_id,))
        conn.execute(
            "UPDATE events SET tier='pending', ttl_expire=NULL WHERE id=?", (pending_id,)
        )
    _force_same_keywords(provider, long_id, pending_id, plain_id)

    result = provider.lifecycle.compact(now)

    remaining = {int(row["id"]) for row in provider.sqlite.fetchall("SELECT id FROM events")}
    assert {long_id, pending_id, plain_id} <= remaining
    assert (result["merged"], result["merge_groups_skipped"]) == (0, 1)


def test_compact_still_merges_plain_groups(tmp_path):
    provider = make_provider(tmp_path)
    now = datetime.now().astimezone()
    low = seed_event(provider, content='记住重复的静音键盘', importance=0.8)
    high = seed_event(provider, content='记住另一个重复事件', importance=0.9)
    _force_same_keywords(provider, low, high)
    before = [dict(row) for row in provider.sqlite.fetchall("SELECT * FROM events ORDER BY id")]
    archive = tmp_path / "memory-archive" / "events-archived.jsonl"

    conflicting_source = json.dumps({**json.loads(before[0]["source_context"] or "{}"), "legacy": "metadata"})
    provider.sqlite.execute("UPDATE events SET source_context=? WHERE id=?", (conflicting_source, low))
    with pytest.raises(ValueError, match="merge_source_drift"):
        provider.lifecycle.compact(now)
    assert not archive.exists()
    assert [dict(row) for row in provider.sqlite.fetchall("SELECT * FROM events ORDER BY id")] == [
        {**before[0], "source_context": conflicting_source}, before[1],
    ]
    provider.sqlite.execute("UPDATE events SET source_context=? WHERE id=?", (before[0]["source_context"], low))

    archive.mkdir(parents=True)
    with pytest.raises(OSError):
        provider.lifecycle.compact(now)
    assert [dict(row) for row in provider.sqlite.fetchall("SELECT * FROM events ORDER BY id")] == before
    assert provider.sqlite.fetchall("SELECT * FROM event_occurrences") == []
    archive.rmdir()

    result = provider.lifecycle.compact(now)

    remaining = {int(row["id"]) for row in provider.sqlite.fetchall("SELECT id FROM events")}
    assert remaining == {high}
    assert (result["merged"], result["merge_groups_skipped"]) == (1, 0)
    entries = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    assert [(entry["id"], entry["raw"]) for entry in entries] == [(low, before[0]["raw"])]
    assert [tuple(row) for row in provider.sqlite.fetchall("SELECT event_id,raw FROM event_occurrences")] == [
        (high, before[0]["raw"]),
    ]


_ACCOUNTING_COLUMNS = {
    "candidate_reason", "staged_at", "promoted_reason", "asked_via", "asked_session",
}


def test_promotion_accounting_columns_exist_and_current_schema_reopens_without_migration(tmp_path):
    provider = make_provider(tmp_path)
    schema = {row[1] for row in provider.sqlite.fetchall("PRAGMA table_info(events)")}
    assert _ACCOUNTING_COLUMNS <= schema

    event_id = call(provider, "sanae_mem_add", {"content": "记住我喜欢静音键盘"})["result"]["id"]
    provider.shutdown()
    backups = list((tmp_path / "data").glob("sanae.db.pre-upgrade-v*.sqlite"))
    assert backups == []

    restarted = make_provider(tmp_path)
    assert {row[1] for row in restarted.sqlite.fetchall("PRAGMA table_info(events)")} == schema
    assert restarted.sqlite.fetchone("SELECT raw FROM events WHERE id=?", (event_id,))[0]
    restarted.shutdown()
    assert list((tmp_path / "data").glob("sanae.db.pre-upgrade-v*.sqlite")) == backups


def test_marked_promoted_keeps_a_legacy_staged_reason_or_falls_back_to_manual(tmp_path):
    provider = make_provider(tmp_path)
    staged = call(provider, "sanae_mem_add", {"content": "记住我喜欢静音键盘"})["result"]["id"]
    with provider.sqlite.transaction() as conn:
        conn.execute("UPDATE events SET candidate_reason='signal' WHERE id=?", (staged,))
    call(provider, "sanae_mem_marked_promoted", {"ids": [staged], "to": "long"})
    row = provider.sqlite.fetchone(
        "SELECT tier, promoted_at, promoted_reason FROM events WHERE id=?", (staged,)
    )
    assert row["tier"] == "long" and row["promoted_at"]
    assert row["promoted_reason"] == "signal", "旧候选行的原因列仍被沿用"

    manual = call(provider, "sanae_mem_add", {"content": "她觉得该留下的一件事"})["result"]["id"]
    call(provider, "sanae_mem_marked_promoted", {"ids": [manual], "to": "important"})
    assert provider.sqlite.fetchone(
        "SELECT promoted_reason FROM events WHERE id=?", (manual,)
    )[0] == "manual"


def test_scheduled_ask_records_channel_and_hashed_session(tmp_path):
    expected = hashlib.sha256(b"session-a").hexdigest()[:16]
    provider = make_provider(tmp_path, read_gate="scheduled", daily_scheduled_asks=2)
    keyboard = call(
        provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9}
    )["result"]["id"]

    hit = call(provider, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})

    assert [item["id"] for item in hit["result"]] == [keyboard]
    row = provider.sqlite.fetchone(
        "SELECT asked, asked_via, asked_session FROM events WHERE id=?", (keyboard,)
    )
    assert tuple(row) == (1, "scheduled", expected)


def test_explicit_past_and_gap_asks_record_their_own_channel(tmp_path):
    expected = hashlib.sha256(b"session-a").hexdigest()[:16]
    explicit = make_provider(tmp_path / "explicit", read_gate="explicit")
    recalled = call(explicit, "sanae_mem_add", {"content": "我想买一把静音键盘"})["result"]["id"]
    explicit_block = explicit.prefetch("还记得我说过的静音键盘吗")
    row = explicit.sqlite.fetchone(
        "SELECT asked_via, asked_session FROM events WHERE id=?", (recalled,)
    )
    if _fallback_fail_closed():
        assert explicit_block == "", "散列回退不得自动注入；实际 %r" % (explicit_block,)
        assert tuple(row) == ("", ""), "没进块的行不得被标记；实际 %s" % (tuple(row),)
    else:
        assert "静音键盘" in explicit_block
        assert tuple(row) == ("explicit_past", expected)

    gap = make_provider(tmp_path / "gap", gap_threshold_hours=4)
    other = call(gap, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})["result"]["id"]
    now = datetime.now().astimezone()
    gap.prefetch("你好", session_id="session-a")
    _set_status_fields(gap, last_user_active_at=(now - timedelta(hours=5)).isoformat())
    assert gap.prefetch("静音键盘的事", session_id="session-a").startswith("## 小事记忆\n")
    assert gap.sqlite.fetchone(
        "SELECT asked_via FROM events WHERE id=?", (other,)
    )[0] == "gap"


def test_cross_day_reset_clears_the_ask_provenance_columns(tmp_path):
    provider = make_provider(tmp_path, read_gate="scheduled", daily_scheduled_asks=2)
    keyboard = call(
        provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9}
    )["result"]["id"]
    call(provider, "sanae_mem_search", {"query": "静音键盘", "scheduled": True})
    assert provider.sqlite.fetchone(
        "SELECT asked_via FROM events WHERE id=?", (keyboard,)
    )[0] == "scheduled"

    assert provider.event_store.reset_asked_all() == 1

    row = provider.sqlite.fetchone(
        "SELECT asked, asked_via, asked_session FROM events WHERE id=?", (keyboard,)
    )
    assert tuple(row) == (0, "", "")


def _plugin_module(name):
    return importlib.import_module("sanae_mem." + name)


TEST_SID = "T:session-t"
TEST_TURN_TEXT = "我们今天吵架了，我很难过"


def make_test_provider(tmp_path, monkeypatch, **config):
    set_test_home(monkeypatch, tmp_path)
    _enter_test_turn(TEST_SID, {"user_id": "fixture-owner"})
    values = {"data_dir": str(tmp_path / "data"), "semantic_backend": "sqlite", "shadow": False}
    values.update(config)
    provider = SanaeMemProvider(values)
    provider.initialize(TEST_SID, hermes_home=str(tmp_path), platform="test",
                        agent_context="primary")
    return provider


@pytest.mark.parametrize("surface", [
    "sanae_mem_add", "sanae_mem_forget", "sanae_mem_compact",
    "sanae_mem_marked_promoted", "sanae_mem_distill", "sync_turn",
    "on_turn_start", "on_pre_compress", "sanae_mem_report_false_trigger",
])
def test_isolated_session_leaves_events_and_life_state_untouched(tmp_path, monkeypatch, surface):
    normal = make_provider(tmp_path)
    written = call(normal, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    assert written["ok"] and written["result"]["stored"]
    event_id = written["result"]["id"]
    normal.shutdown()
    provider = make_test_provider(tmp_path, monkeypatch)
    before = [tuple(row) for row in provider.sqlite.fetchall("SELECT * FROM events ORDER BY id")]
    status_path = tmp_path / "data" / "status.json"
    status_before = status_path.read_bytes()
    writes_before = provider.sqlite.read(lambda conn: conn.total_changes)
    _os.utime(status_path, (1_000_000_000, 1_000_000_000))
    status_mtime_before = status_path.stat().st_mtime_ns
    try:
        if surface == "sync_turn":
            provider.sync_turn(TEST_TURN_TEXT, "我记住了。", session_id=TEST_SID)
        elif surface == "on_turn_start":
            provider.on_turn_start(1, TEST_TURN_TEXT)
        elif surface == "on_pre_compress":
            assert provider.on_pre_compress([{"role": "user", "content": TEST_TURN_TEXT}]) == ""
        else:
            args = {
                "sanae_mem_add": {"content": "我想买两把静音键盘", "importance": 0.9},
                "sanae_mem_forget": {"query": "静音键盘"},
                "sanae_mem_marked_promoted": {"ids": [event_id], "to": "long"},
            }.get(surface, {})
            payload = call(provider, surface, args)
            assert payload["ok"] is True
        assert [tuple(row) for row in provider.sqlite.fetchall("SELECT * FROM events ORDER BY id")] == before
        assert provider.sqlite.read(lambda conn: conn.total_changes) == writes_before, "测试会话不得写入生活数据库"
        assert status_path.read_bytes() == status_before
        assert status_path.stat().st_mtime_ns == status_mtime_before, "测试会话不得重写 status"
        assert not (tmp_path / "data" / "safety.log").exists()
    finally:
        provider.shutdown()


@pytest.mark.parametrize("action", ["add", "replace", "merge", "remove"])
def test_isolated_session_blocks_all_host_memory_actions(tmp_path, monkeypatch, action):
    provider = make_test_provider(tmp_path, monkeypatch)
    try:
        result = memory_guard_mod.guard_memory_write(
            "memory", {"action": action, "target": "memory", "content": "测试内容",
                       "old_text": "旧内容"}, TEST_SID)
        assert result is not None and result["action"] == "block"
    finally:
        provider.shutdown()


def test_isolated_session_can_read_existing_events(tmp_path, monkeypatch):
    normal = make_provider(tmp_path)
    event_id = seed_event(normal, "我想买一把静音键盘", importance=0.9)
    normal.shutdown()
    provider = make_test_provider(tmp_path, monkeypatch)
    try:
        found = call(provider, "sanae_mem_search", {"query": "静音键盘"})
        assert found["ok"] is True
        assert event_id in [row["id"] for row in found["result"]]
        assert provider.sqlite.fetchone("SELECT COUNT(*) FROM events")[0] == 1
    finally:
        provider.shutdown()


def test_normal_session_keeps_event_and_status_writes(tmp_path):
    provider = make_provider(tmp_path)
    provider.on_session_switch("session-normal")
    status_path = tmp_path / "data" / "status.json"
    before = status_path.read_bytes() if status_path.exists() else None
    try:
        assert memory_guard_mod.guard_memory_write(
            "memory", {"action": "add", "target": "memory", "content": "正常会话"},
            "session-normal") is None
        added = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
        assert added["result"]["stored"] is True
        assert provider.sqlite.fetchone("SELECT COUNT(*) FROM events")[0] == 1
        provider.on_pre_compress([{"role": "user", "content": "随便说点什么"}])
        assert status_path.exists() and status_path.read_bytes() != before
        assert provider.status_store.read()["pre_compress_count_today"] == 1
    finally:
        provider.shutdown()


_R4_QUERY = "还记得上次我和小林淘到的那本植物图鉴吗"
_R4_RELEVANT = "八月和小林在城南旧书店淘到一本绝版的植物图鉴"
_R4_NOISE = (
    "明天上午九点要陪客户开季度预算评审会",
    "护照快到期了得找时间去出入境大厅换新证",
    "周五晚上陪妈妈去医院做一次复查",
    "阳台那根漏水的水管约了师傅上门修理",
    "给妹妹准备的生日礼物已经寄出去了",
    "周三下午要交这个季度的所得税申报表",
)
_R4_ITEM_KEYS = {
    "id", "raw", "summary", "topic", "importance", "ts", "created_at", "session_id",
    "experience_subject", "fact_subject", "attribution",
}
_TOOL_ITEM_KEYS = (_R4_ITEM_KEYS - {"summary"}) | {"source_context"}


def _r4_add(provider, text, importance, days_ago=0, topic=None, tier=None):
    now = datetime.now().astimezone() - timedelta(days=days_ago)
    event_id = seed_row(provider, text, importance, topic=topic, now=now,
                        session_id="session-a", owner_key=_provider_owner(provider))
    if tier:
        with provider.sqlite.transaction() as conn:
            conn.execute("UPDATE events SET tier=? WHERE id=?", (tier, event_id))
    return event_id


def _fallback_fail_closed() -> bool:
    return textutil.get_embedding_state().session is None


def _r4_inject(provider, query, session_id="session-a"):
    before = {
        int(row[0]): int(row[1] or 0)
        for row in provider.sqlite.fetchall("SELECT id, asked FROM events")
    }
    block = provider.prefetch(query, session_id=session_id)
    after = {
        int(row[0]): int(row[1] or 0)
        for row in provider.sqlite.fetchall("SELECT id, asked FROM events")
    }
    marked = sorted(eid for eid, count in after.items() if count > before.get(eid, 0))
    recall = provider.association_state.current_recall()
    if not block or not recall or recall.get("status") != "ok":
        return block, marked
    marked_set = set(marked)
    ranked = [
        int(item["id"])
        for item in (recall.get("items") or [])
        if int(item["id"]) in marked_set
    ]
    extra = [eid for eid in marked if eid not in ranked]
    return block, ranked + extra


def test_r4_relevant_event_survives_high_importance_noise(tmp_path):
    provider = make_provider(tmp_path)
    relevant = _r4_add(provider, _R4_RELEVANT, 0.5)
    noise = [_r4_add(provider, text, 0.95) for text in _R4_NOISE]

    block, injected = _r4_inject(provider, _R4_QUERY)

    if _fallback_fail_closed():
        assert injected == [], "散列回退不得自动注入；实际 %s" % (injected,)
        return
    assert injected, "显式追问必须至少带出一条（相关行就在库里）"
    assert injected[0] == relevant, (
        "相关行必须排第一；实际注入顺序 id=%s（相关行 id=%s）" % (injected, relevant)
    )
    assert relevant in injected
    assert "植物图鉴" in block
    for order, event_id in enumerate(injected):
        if event_id != relevant:
            assert event_id in noise and order > 0


def test_r4_closer_match_outranks_less_close_one(tmp_path):
    provider = make_provider(tmp_path)
    strong = _r4_add(provider, _R4_RELEVANT, 0.3)
    weak = _r4_add(provider, "之前好像和小林提过一句城南旧书店的植物图鉴", 0.95)

    _, injected = _r4_inject(provider, _R4_QUERY)

    if _fallback_fail_closed():
        assert injected == [], "散列回退不得自动注入；实际 %s" % (injected,)
        return
    assert injected[:2] == [strong, weak], (
        "更相关者必须在前；实际 %s（strong=%s weak=%s）" % (injected, strong, weak)
    )


def test_r4_importance_still_breaks_relevance_ties(tmp_path):
    provider = make_provider(tmp_path)
    low = _r4_add(provider, _R4_RELEVANT, 0.35)
    high = _r4_add(provider, _R4_RELEVANT, 0.9)

    _, injected = _r4_inject(provider, _R4_QUERY)

    if _fallback_fail_closed():
        assert injected == [], "散列回退不得自动注入；实际 %s" % (injected,)
        return
    assert set(injected) == {low, high}
    assert injected[0] == high, "同分时 importance 高者在前（实际 %s）" % (injected,)


def test_r4_long_tier_age_exemption_survives(tmp_path):
    provider = make_provider(tmp_path)
    relevant = _r4_add(provider, _R4_RELEVANT, 0.5, days_ago=60, topic="生活", tier="long")
    for text in _R4_NOISE:
        _r4_add(provider, text, 0.95)

    _, injected = _r4_inject(provider, _R4_QUERY)

    if _fallback_fail_closed():
        assert injected == [], "散列回退不得自动注入；实际 %s" % (injected,)
        return
    assert injected and injected[0] == relevant, (
        "60 天前的 long 事件必须仍被显式追问召回；实际 %s（相关行 %s）" % (injected, relevant)
    )


def test_r4_no_match_does_not_pad_with_unrelated_old_events(tmp_path):
    provider = make_provider(tmp_path)
    for text in _R4_NOISE:
        _r4_add(provider, text, 0.95)

    _, injected = _r4_inject(provider, "还记得上次我们一起去看的那场极光吗")

    assert injected == [], (
        "没有任何够格候选（真嵌入看 0.34／0.60，散列回退 fail closed）"
        "⇒ 注入块必须为空；实际 %s" % (injected,)
    )


def test_r4_plain_and_gap_paths_are_untouched(tmp_path):
    provider = make_provider(tmp_path, gap_threshold_hours=4)
    call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})
    provider.prefetch("你好", session_id="session-a")
    now = datetime.now().astimezone()

    plain = provider.retrieval.search({"query": _R4_QUERY, "top_k": 10}, for_injection=True)
    assert plain and all(set(item) == _R4_ITEM_KEYS for item in plain)

    _set_status_fields(provider, last_user_active_at=(now - timedelta(hours=3)).isoformat())
    assert provider.prefetch("今天想随便聊点什么", session_id="session-a") == ""
    _set_status_fields(provider, last_user_active_at=(now - timedelta(hours=5)).isoformat())
    block = provider.prefetch("静音键盘的事", session_id="session-a")
    assert block.startswith("## 小事记忆")
    assert block.count("\n- ") == 1


def test_r4_tool_search_shape_is_unchanged(tmp_path):
    provider = make_provider(tmp_path)
    keyboard = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘", "importance": 0.9})["result"]["id"]
    call(provider, "sanae_mem_add", {"content": "这周预约了牙医复查", "importance": 0.8})

    hit = call(provider, "sanae_mem_search", {"query": "静音键盘", "top_k": 10})
    assert hit["ok"]
    assert [item["id"] for item in hit["result"]][0] == keyboard
    assert all(set(item) == _TOOL_ITEM_KEYS for item in hit["result"]), \
        "工具面保留来源/归属，但不得多出 relevance 之类的内部字段"
    assert provider.sqlite.fetchone(
        "SELECT asked FROM events WHERE id=?", (keyboard,))[0] == 0


def test_tool_search_scopes_to_calling_owner_and_does_not_claim_unowned_self_rows(tmp_path):
    owner = make_provider(tmp_path, user_id="owner-a")
    private = call(owner, "sanae_mem_add", {
        "content": "测试共同词 医院复查 只属于本人", "importance": 0.8})["result"]["id"]
    now = datetime.now().astimezone()
    self_row = seed_row(
        owner, "测试共同词 B站视频画面分析，属于早苗自己看过的内容", 0.7, topic="bilibili",
        now=now, session_id="session-a", ttl=now + timedelta(days=14),
        source_attribution=("sanae", "sanae", "", '{"kind": "self"}'),
    )
    other = make_provider(tmp_path, user_id="owner-b")
    other_hits = call(other, "sanae_mem_search", {"query": "测试共同词", "top_k": 10})["result"]
    other_ids = [item["id"] for item in other_hits]
    assert private not in other_ids, "他人不得检索到本人归属事件"
    assert self_row not in other_ids, "空 owner 的 self 行不因体验主体是早苗就被他人认领"
    _enter_test_turn("session-a", {"user_id": "owner-a"})
    own_hits = call(owner, "sanae_mem_search", {"query": "测试共同词", "top_k": 10})["result"]
    assert private in [item["id"] for item in own_hits], "本人可读自己的事件"
    assert self_row not in [item["id"] for item in own_hits], "空 owner 旧行不认领给当前 owner"


def test_r4_scheduled_path_still_ranks_by_importance(tmp_path):
    provider = make_provider(tmp_path, read_gate="scheduled", daily_scheduled_asks=2)
    low = seed_event(provider, content='牙医复查排在周四', importance=0.5)
    high = seed_event(provider, content='牙医复查要带上次的片子', importance=0.95)

    block = provider.prefetch("牙医复查", scheduled=True)

    assert block.startswith("## 小事记忆")
    assert block.count("\n- ") == 1
    row = provider.sqlite.fetchone(
        "SELECT asked_via FROM events WHERE id=?", (high,))
    assert row is not None and row[0] == "scheduled"
    assert provider.sqlite.fetchone(
        "SELECT asked FROM events WHERE id=?", (low,))[0] == 0


def test_r4_asked_provenance_marks_only_injected_rows(tmp_path):
    provider = make_provider(tmp_path)
    relevant = _r4_add(provider, _R4_RELEVANT, 0.5)
    noise = [_r4_add(provider, text, 0.95) for text in _R4_NOISE]

    _, injected = _r4_inject(provider, _R4_QUERY)
    expected_session = hashlib.sha256(b"session-a").hexdigest()[:16]

    if _fallback_fail_closed():
        assert injected == [], "散列回退不得自动注入；实际 %s" % (injected,)
        for event_id in noise:
            assert tuple(provider.sqlite.fetchone(
                "SELECT asked, asked_via, asked_session FROM events WHERE id=?",
                (event_id,))) == (0, "", ""), "没进块的行不得被标记"
        return
    assert relevant in injected
    for event_id in injected:
        row = provider.sqlite.fetchone(
            "SELECT asked, asked_at, asked_via, asked_session FROM events WHERE id=?",
            (event_id,))
        assert row[0] == 1 and row[1] and row[2] == "explicit_past"
        assert row[3] == expected_session
    for event_id in set(noise) - set(injected):
        row = provider.sqlite.fetchone(
            "SELECT asked, asked_via, asked_session FROM events WHERE id=?",
            (event_id,))
        assert tuple(row) == (0, "", ""), "没进块的候选不得被标记"


def test_r4_unrelated_second_owner_memory_never_enters_the_block(tmp_path):
    first = make_provider(tmp_path, user_id="owner-one")
    second = make_provider(tmp_path, user_id="owner-two")
    assert _provider_owner(second) and _provider_owner(second) != _provider_owner(first)

    mine = _r4_add(first, _R4_RELEVANT, 0.5)
    theirs = seed_row(
        second, "第二 owner 的植物图鉴在城北旧书店", 0.95, topic="生活",
        now=datetime.now().astimezone(), session_id="session-b",
        owner_key=_provider_owner(second),
    )

    _, injected = _r4_inject(first, _R4_QUERY)

    if _fallback_fail_closed():
        assert injected == [], "散列回退不得自动注入；实际 %s" % (injected,)
        assert theirs not in injected, "另一人的无关事件不得被注入"
        return
    assert mine in injected
    assert theirs not in injected, "另一人的无关事件不得被注入"


def test_r4_scenario_writes_only_inside_the_isolated_root(tmp_path):
    production = _production_root()

    def snapshot():
        out = {}
        for name in ("memory-archive/events-archived.jsonl", "sanae-mem/sanae.db",
                     "memories/MEMORY.md", "memories/USER.md",
                     "sanae-mem/status.json"):
            path = production / name
            if path.is_file():
                out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out

    before = snapshot()
    provider = make_provider(tmp_path)
    _r4_add(provider, _R4_RELEVANT, 0.5)
    for text in _R4_NOISE:
        _r4_add(provider, text, 0.95)
    _r4_inject(provider, _R4_QUERY)

    assert snapshot() == before, "R4 场景不得改动生产归档／库／记忆文件"




_R5_RELEVANT_PAIRS = (
    ("还记得上次我和小林淘到的那本书吗", "八月和小林在城南旧书店淘到一本绝版的植物图鉴"),
    ("上次说的那个换护照的事办了吗", "护照快到期了得找时间去出入境大厅换新证"),
    ("还记得我说过要陪妈妈复查吗", "周五晚上陪妈妈去医院做一次复查"),
    ("之前那根漏水的水管修好了吗", "阳台那根漏水的水管约了师傅上门修理"),
    ("我上次是不是跟你提过给妹妹寄礼物", "给妹妹准备的生日礼物已经寄出去了"),
    ("那个季度所得税申报表交了吗", "周三下午要交这个季度的所得税申报表"),
    ("还记得上次面试的事吗", "下周二上午去城西那家公司面试产品岗位"),
    ("之前说要买的那把静音键盘下单了吗", "想买一把静音键盘，已经加了购物车还没下单"),
    ("上次体检的结果怎么样", "上周三在市医院做了年度体检，报告说胆固醇偏高"),
    ("还记得我说过要学着做饭吗", "报了个周末的烹饪班，从下周开始学做家常菜"),
)

_R5_NEGATIVE_EVENTS = (
    "把阳台那台旧风扇擦干净收进了柜子",
    "楼道里的声控灯坏了得找物业来换",
    "冰箱冷冻层的抽屉卡住了要修",
    "楼下花坛的月季该剪枝了",
    "厨房抽油烟机的滤网该换了",
    "客厅那盏台灯接触不良要修",
    "阳台的晾衣架螺丝松了要拧紧",
    "卫生间的花洒出水变小了要通",
    "书房的椅子坐垫塌了要换新的",
    "院子里的桂花今年开得晚",
    "电动车的后刹车有点松要调",
    "卧室的窗帘轨道掉了一颗螺丝",
    "鱼缸的过滤器该清洗了",
    "阳台的花盆底下有积水要倒",
    "厨房水槽的下水变慢了要通",
    "门口的鞋架有点晃要加固",
    "书桌抽屉的滑轨坏了要换",
    "客厅的空调滤网该洗了",
    "楼上的水管有响声要问一下",
    "餐椅的一条腿有点不稳要垫",
)


def _r5_add(provider, text, importance=0.7):
    result = call(provider, "sanae_mem_add", {"content": text, "importance": importance})["result"]
    assert result["stored"]
    return result["id"]


def _r5_chroma_ids(provider, query):
    collection = provider.storage.current_collection()
    assert collection is not None
    fetched = collection.query(
        query_embeddings=[textutil._embed_query(query)], n_results=10)
    return {int(x.removeprefix("evt-")) for x in (fetched.get("ids") or [[]])[0]}


def _r5_lexical_semantic(provider, query, event_id):
    query_tokens = set(textutil._tokens(query) + textutil.event_keywords(query).split())
    row = provider.sqlite.fetchone(
        "SELECT raw, keywords, vec FROM events WHERE id=?", (event_id,))
    row_tokens = set(textutil._tokens(row["raw"]) + row["keywords"].split())
    lexical = len(query_tokens & row_tokens) / max(1, len(query_tokens))
    semantic = textutil._cosine(textutil._embed_query(query), textutil._unpack_vector(row["vec"]))
    return lexical, semantic


def test_r5_chroma_bonus_only_for_rows_that_are_actually_close(tmp_path):
    pytest.importorskip("chromadb", reason="optional Chroma integration dependency is not installed")
    provider = make_provider(tmp_path, backend="chroma")
    assert provider.storage.current_collection() is not None, "本用例只覆盖 chroma 路径"
    _r5_add(provider, _R4_RELEVANT, 0.5)
    for text in _R4_NOISE:
        _r5_add(provider, text, 0.95)

    chroma_ids = _r5_chroma_ids(provider, _R4_QUERY)
    assert chroma_ids, "chroma 必须真的召回了行，否则这条断言是无的放矢"

    hits = provider.retrieval.search({"query": _R4_QUERY, "top_k": 10}, for_injection=True,
                            with_relevance=True)
    assert hits
    granted, withheld = [], []
    for item in hits:
        lexical, semantic = _r5_lexical_semantic(provider, _R4_QUERY, item["id"])
        expected = (words._RECALL_LEXICAL_WEIGHT * lexical
                    + words._RECALL_SEMANTIC_WEIGHT * max(0.0, semantic))
        close = item["id"] in chroma_ids and semantic >= words._CHROMA_BONUS_MIN_COSINE
        if close:
            expected += words._RECALL_CHROMA_BONUS
            granted.append(item["id"])
        elif item["id"] in chroma_ids:
            withheld.append(item["id"])
        assert item["relevance"] == pytest.approx(expected, abs=1e-9), (
            "id=%s 的 relevance 与「够近才加分」的期望不符" % item["id"]
        )
    assert granted, "至少要有一行真的够近 —— 否则加分分支没被走到，这条用例等于没测"
    assert withheld, "至少要有一行被扣下加分 —— 否则这条用例是同义反复"


def test_r5_chroma_pool_keeps_the_real_target(tmp_path):
    pytest.importorskip("chromadb", reason="optional Chroma integration dependency is not installed")
    provider = make_provider(tmp_path, backend="chroma")
    assert provider.storage.current_collection() is not None, "本用例只覆盖 chroma 路径"
    targets = [(_r5_add(provider, event, 0.5), query)
               for query, event in _R5_RELEVANT_PAIRS]
    for event in _R5_NEGATIVE_EVENTS:
        _r5_add(provider, event, 0.95)

    kept = []
    for target_id, query in targets:
        hits = provider.retrieval.search({"query": query, "top_k": 10}, for_injection=True)
        if any(item["id"] == target_id for item in hits):
            kept.append(query)

    assert len(kept) >= 9, (
        "候选池把真目标挤掉了：只有 %d/10 被检索返回；未返回的是 %s"
        % (len(kept), [q for _i, q in targets if q not in kept])
    )


def test_r5_sqlite_backend_is_unaffected(tmp_path):
    provider = make_provider(tmp_path)
    assert provider.storage.current_collection() is None
    relevant = _r5_add(provider, _R4_RELEVANT, 0.5)
    for text in _R4_NOISE:
        _r5_add(provider, text, 0.95)

    block, injected = _r4_inject(provider, _R4_QUERY)

    if _fallback_fail_closed():
        assert injected == [], "散列回退不得自动注入；实际 %s" % (injected,)
        return
    assert injected and injected[0] == relevant
    assert "植物图鉴" in block


def test_r5_tool_face_still_leaks_nothing(tmp_path):
    provider = make_provider(tmp_path, backend="chroma")
    keyboard = call(provider, "sanae_mem_add",
                    {"content": "我想买一把静音键盘", "importance": 0.9})["result"]["id"]
    hit = call(provider, "sanae_mem_search", {"query": "静音键盘", "top_k": 10})
    assert hit["ok"]
    assert [item["id"] for item in hit["result"]][0] == keyboard
    assert all(set(item) == _TOOL_ITEM_KEYS for item in hit["result"])


_R6_MODEL_SRC = _production_root() / "sanae-mem" / "models"
_R6_HAS_MODEL = (_R6_MODEL_SRC / "bge-small-zh-v1.5" / "model.onnx").is_file()

_R6_PARAPHRASE_ONLY = "之前我妈那次检查没什么事吧"
_R6_PARAPHRASE_EVENT = "周五晚上陪妈妈去医院做一次复查"


def _r6_provider(tmp_path, *, with_model, **config):
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    if with_model and _R6_HAS_MODEL:
        (data / "models").symlink_to(_R6_MODEL_SRC)
    _enter_test_turn("session-a", {"user_id": "u1"})
    provider = SanaeMemProvider(
        dict({"data_dir": str(data), "semantic_backend": "sqlite", "shadow": False}, **config))
    provider.initialize("session-a", hermes_home=str(tmp_path), platform="test",
                        agent_context="primary", user_id="u1")
    owner_key = source_context.scope_account_key(
        source_context.ChatScope("private", platform="test", actor="u1")
    )
    _PROVIDER_META[provider] = (data, owner_key, "session-a")
    return provider


def _r6_reset_embedder():
    textutil.reset_embedding_state()


@pytest.mark.skipif(not _R6_HAS_MODEL, reason="本机没有本地嵌入模型（静态资产），跳过真路径")
def test_r6_real_embedding_reaches_a_paraphrase_the_hash_never_could(tmp_path):
    _r6_reset_embedder()
    provider = _r6_provider(tmp_path, with_model=True, read_gate="explicit")
    event_id = _r4_add(provider, _R6_PARAPHRASE_EVENT, 0.5)

    state = textutil.get_embedding_state()
    assert state.session is not None, "模型必须真的被加载（否则这条是同义反复）"
    assert state.failed is False

    block, injected = _r4_inject(provider, _R6_PARAPHRASE_ONLY)

    assert event_id in injected, (
        "真嵌入必须能捞回这条改写；实际注入 %s（事件 id=%s）" % (injected, event_id)
    )
    assert "复查" in block


@pytest.mark.skipif(not _R6_HAS_MODEL, reason="本机没有本地嵌入模型（静态资产），跳过真路径")
def test_r6_gate_stops_padding_when_nobody_remembers(tmp_path):
    _r6_reset_embedder()
    provider = _r6_provider(tmp_path, with_model=True, read_gate="explicit")
    for text in _R4_NOISE:
        _r4_add(provider, text, 0.95)

    _, injected = _r4_inject(provider, "还记得上次我们一起去看的那场极光吗")

    assert injected == [], "没有够得上证据门的候选时，注入块必须是空的；实际 %s" % (injected,)


def test_r6_missing_model_falls_back_to_hash_without_auto_injection(tmp_path):
    _r6_reset_embedder()
    provider = _r6_provider(tmp_path, with_model=False, read_gate="explicit")
    event_id = _r4_add(provider, "我想买一把静音键盘", 0.9)

    state = textutil.get_embedding_state()
    assert state.session is None
    assert state.failed is True, "缺模型必须被记成 failed（而不是每次重试）"

    vector = textutil._embed("随便一句话")
    assert len(vector) == words._DIMS, "回退路径也必须给出 _DIMS 维向量（否则与库里的形状不一致）"

    block, injected = _r4_inject(provider, "还记得我说过的静音键盘吗")

    assert injected == [], "散列回退不得自动注入（fail closed）；实际 %s" % (injected,)
    assert "小事记忆" not in block

    hit = call(provider, "sanae_mem_search", {"query": "静音键盘", "top_k": 10})
    assert hit["ok"] and [item["id"] for item in hit["result"]][0] == event_id
    _r6_reset_embedder()


def test_r6_query_prefix_is_only_applied_to_queries(tmp_path, monkeypatch):
    seen = []

    def spy(texts):
        seen.append(list(texts))
        return None

    monkeypatch.setattr(textutil, "_onnx_embed", spy)
    textutil._embed("文档侧文本")
    textutil._embed_query("问句侧文本")

    assert seen[0] == ["文档侧文本"], "文档侧不得加前缀"
    assert seen[1] == [words._EMBED_QUERY_PREFIX + "问句侧文本"], "问句侧必须加前缀"


def test_dormant_search_shows_relative_date_yesterday(tmp_path):
    provider = make_provider(tmp_path / "dormant-rel")
    event_id = call(provider, "sanae_mem_add", {"content": "我想买一把静音键盘"})["result"]["id"]
    yesterday_iso = (datetime.now().astimezone() - timedelta(days=1)).isoformat()
    provider.sqlite.execute(
        "UPDATE events SET tier='dormant', summary_source='gist_faded', created_at=? WHERE id=?",
        (yesterday_iso, event_id),
    )
    hit = call(provider, "sanae_mem_search", {"query": "静音键盘"})
    assert hit["ok"] and len(hit["result"]) == 1
    assert hit["result"][0]["id"] == event_id
    assert "昨天" in hit["result"][0]["provenance"]


def test_compact_write_status_diff_preserves_concurrent_fields(tmp_path):
    provider = make_provider(tmp_path / "compact-diff", distill_daily_cap=2)
    now = datetime.now().astimezone()
    provider.lifecycle.compact(now)
    (tmp_path / "compact-diff" / "memories").mkdir(exist_ok=True)
    for name in ("USER.md", "MEMORY.md"):
        (tmp_path / "compact-diff" / "memories" / name).write_text("", encoding="utf-8")
    seeded = call(provider, "sanae_mem_add", {"content": "值得长期记得的重要约定", "importance": 0.8})
    assert seeded["ok"] and seeded["result"]["stored"] is True

    class ConcurrentWriteLlm:
        def __init__(self):
            self.triggered = False

        def complete(self, messages, **kwargs):
            if not self.triggered:
                self.triggered = True
                added = call(provider, "sanae_mem_add", {"content": "并发期间新增的另一条记忆"})
                assert added["ok"] and added["result"]["stored"] is True
            return types.SimpleNamespace(text=json.dumps({
                "action": "discard", "target": "memory", "text": "", "reason": "none",
            }))

    provider.attach_runtime_llm(ConcurrentWriteLlm())
    result = provider.lifecycle.compact(now)
    status = json.loads((tmp_path / "compact-diff" / "data" / "status.json").read_text(encoding="utf-8"))
    assert status["daily_writes"] == 2
    assert status["last_compactor"] == result


def test_runtime_half_life_bounds_across_ingest_recall_and_promotions(tmp_path):
    provider = make_provider(tmp_path / "hl-bounds")
    try:
        # 1. Natural ingest with emotion_ttl_days=90 must clamp short half_life_days to <= 60.0
        #    while preserving ttl_expire around +90d.
        provider.sync_turn("今天真的好开心，谢谢你一直信任我！", "嗯嗯", session_id="session-a")
        emotion_row = provider.sqlite.fetchone(
            "SELECT id, tier, half_life_days, ttl_expire, curve_ttl, owner_key FROM events ORDER BY id DESC LIMIT 1"
        )
        assert emotion_row is not None
        owner_key = str(emotion_row["owner_key"])
        assert owner_key
        assert emotion_row["tier"] == "short"
        assert emotion_row["half_life_days"] == pytest.approx(60.0)
        ttl_days = (
            datetime.fromisoformat(emotion_row["ttl_expire"]) - datetime.now().astimezone()
        ).total_seconds() / 86400
        assert 89 < ttl_days < 91

        # 2. Natural ingest with importance=0.0 must clamp half_life_days up to floor >= 9.0.
        low_res = call(provider, "sanae_mem_add", {
            "content": "顺手在便利店买了一瓶矿泉水", "importance": 0.0,
        })
        assert low_res["ok"] and low_res["result"]["stored"] is True
        low_row = provider.sqlite.fetchone(
            "SELECT half_life_days FROM events WHERE id=?", (low_res["result"]["id"],)
        )
        assert low_row is not None and low_row["half_life_days"] == pytest.approx(9.0)

        # 3. Explicit ttl_days preserves shorter ttl_expire (+3d) while clamping half_life_days into [9.0, 60.0].
        short_exp = call(provider, "sanae_mem_add", {
            "content": "明天去楼下取一下快递包裹", "ttl_days": 3,
        })
        assert short_exp["ok"] and short_exp["result"]["stored"] is True
        short_exp_row = provider.sqlite.fetchone(
            "SELECT half_life_days, ttl_expire FROM events WHERE id=?", (short_exp["result"]["id"],)
        )
        assert short_exp_row["half_life_days"] == pytest.approx(9.0)
        short_exp_days = (
            datetime.fromisoformat(short_exp_row["ttl_expire"]) - datetime.now().astimezone()
        ).total_seconds() / 86400
        assert 2 < short_exp_days < 4

        long_exp = call(provider, "sanae_mem_add", {
            "content": "最近常常惦记这段珍贵的陪伴和信任", "ttl_days": 90,
        })
        assert long_exp["ok"] and long_exp["result"]["stored"] is True
        assert provider.sqlite.fetchone(
            "SELECT half_life_days FROM events WHERE id=?", (long_exp["result"]["id"],)
        )["half_life_days"] == pytest.approx(60.0)

        # Direct insert_event_row also enforces [9.0, 60.0] bounds on new runtime rows.
        now_iso = datetime.now().astimezone().isoformat()
        ttl_iso = (datetime.now().astimezone() + timedelta(days=3)).isoformat()
        direct_inserted = provider.event_store.insert_event_row(
            "fp-direct-low-test",
            (
                event_time_anchor(datetime.now().astimezone()),
                "[test] 直接落库新行下限测试",
                textutil.event_keywords("直接落库新行下限测试"),
                0.5, 0.9, "其他", ttl_iso, now_iso, None, "session-a",
            ),
            owner_key=owner_key,
            summary="直接落库新行下限测试",
            half_life_days=3.0,
        )
        assert direct_inserted is not None
        assert provider.sqlite.fetchone(
            "SELECT half_life_days FROM events WHERE id=?", (direct_inserted,)
        )["half_life_days"] == pytest.approx(9.0)

        # 4. reinforce_on_recall clamps short<=60, NULL/empty tier<=60 (with or without TTL), and long<=120.
        later = datetime.now().astimezone() + timedelta(hours=13)
        assert provider.event_store.reinforce_on_recall(
            [int(emotion_row["id"])], via="chat", now=later, session_id="session-a"
        ) == 1
        assert provider.sqlite.fetchone(
            "SELECT half_life_days, recall_count FROM events WHERE id=?", (int(emotion_row["id"]),)
        )[:] == (pytest.approx(60.0), 1)

        null_tier_id = call(provider, "sanae_mem_add", {
            "content": "旧库空层级事件召回上限一致性验证", "importance": 0.6,
        })["result"]["id"]
        provider.sqlite.execute(
            "UPDATE events SET tier=NULL, half_life_days=60.0, recall_count=0, last_active_at=? WHERE id=?",
            ((later - timedelta(days=1)).isoformat(), null_tier_id),
        )
        assert provider.event_store.reinforce_on_recall(
            [int(null_tier_id)], via="chat", now=later, session_id="session-a"
        ) == 1
        assert provider.sqlite.fetchone(
            "SELECT tier, half_life_days, recall_count FROM events WHERE id=?", (null_tier_id,)
        )[:] == (None, pytest.approx(60.0), 1)

        null_tier_no_ttl_id = call(provider, "sanae_mem_add", {
            "content": "旧库空层级且无TTL事件保留召回资格并钳在60天", "importance": 0.6,
        })["result"]["id"]
        provider.sqlite.execute(
            "UPDATE events SET tier='', ttl_expire=NULL, curve_ttl='', half_life_days=60.0, "
            "recall_count=0, last_active_at=? WHERE id=?",
            ((later - timedelta(days=1)).isoformat(), null_tier_no_ttl_id),
        )
        assert provider.event_store.reinforce_on_recall(
            [int(null_tier_no_ttl_id)], via="chat", now=later, session_id="session-a"
        ) == 1
        assert provider.sqlite.fetchone(
            "SELECT tier, half_life_days, recall_count FROM events WHERE id=?", (null_tier_no_ttl_id,)
        )[:] == ("", pytest.approx(60.0), 1)

        long_id = call(provider, "sanae_mem_add", {
            "content": "长期测试事件用于验证召回半衰期上限", "importance": 0.8,
        })["result"]["id"]
        provider.sqlite.execute(
            "UPDATE events SET tier='long', recall_count=3, half_life_days=120.0, "
            "ttl_expire=NULL, curve_ttl='', last_active_at=? WHERE id=?",
            ((later - timedelta(days=1)).isoformat(), long_id),
        )
        assert provider.event_store.reinforce_on_recall(
            [int(long_id)], via="chat", now=later, session_id="session-a"
        ) == 1
        recalled_long = provider.sqlite.fetchone(
            "SELECT half_life_days, recall_count FROM events WHERE id=?", (long_id,)
        )
        assert recalled_long["recall_count"] == 4
        assert recalled_long["half_life_days"] == pytest.approx(120.0)

        # 5. memory_axis explicit_save promotion caps over-cap rows at 120.0 instead of keeping >120.
        axis_id = call(provider, "sanae_mem_add", {
            "content": "那张手写书签一直夹在旧诗集第一页里", "importance": 0.7,
        })["result"]["id"]
        provider.sqlite.execute("UPDATE events SET half_life_days=150.0 WHERE id=?", (axis_id,))
        marked_long = provider.lifecycle.marked_promoted([int(axis_id)], "long")
        assert marked_long["updated"] == 1
        assert marked_long.get("channel") == "explicit_save"
        axis_row = provider.sqlite.fetchone(
            "SELECT tier, half_life_days FROM events WHERE id=?", (axis_id,)
        )
        assert axis_row[:] == ("long", pytest.approx(120.0))
        promo_row = provider.sqlite.fetchone(
            "SELECT channel FROM memory_axis_promotions WHERE event_id=?", (axis_id,)
        )
        assert promo_row is not None and promo_row["channel"] == "explicit_save"

        # 6. event_maintenance.promote_events clamps both floor (3->9) and ceiling (150->120).
        imp_over = call(provider, "sanae_mem_add", {
            "content": "天文馆望远镜镜片校准日程", "importance": 0.7,
        })["result"]["id"]
        imp_under = call(provider, "sanae_mem_add", {
            "content": "图书馆旧报纸微缩胶卷清单", "importance": 0.7,
        })["result"]["id"]
        assert imp_over != imp_under
        provider.sqlite.execute("UPDATE events SET half_life_days=150.0 WHERE id=?", (imp_over,))
        provider.sqlite.execute("UPDATE events SET half_life_days=3.0 WHERE id=?", (imp_under,))
        marked_imp = provider.lifecycle.marked_promoted([int(imp_over), int(imp_under)], "important")
        assert marked_imp["updated"] == 2
        assert "channel" not in marked_imp
        assert provider.sqlite.fetchone(
            "SELECT tier, half_life_days FROM events WHERE id=?", (imp_over,)
        )[:] == ("important", pytest.approx(120.0))
        assert provider.sqlite.fetchone(
            "SELECT tier, half_life_days FROM events WHERE id=?", (imp_under,)
        )[:] == ("important", pytest.approx(9.0))
    finally:
        provider.shutdown()
