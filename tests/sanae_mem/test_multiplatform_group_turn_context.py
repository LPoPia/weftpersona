"""Task 04: the group's own commitments and its automatic turn context.

Everything here goes through the real ``SanaeMemProvider`` over a temporary HOME and its real
SQLite database: the same grant resolution, the same search/SQL paths and the same
``handle_pre_llm_call`` / ``prefetch`` entry points the live turn uses. No authorization or
read/write function is stubbed out.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem import SanaeMemProvider  # noqa: E402
from sanae_mem.runtime import group_memory  # noqa: E402
from sanae_mem.infra import host_bridge  # noqa: E402
from sanae_mem.governance.source_context import current_chat_scope, parse_source  # noqa: E402

GROUP_A = "default/discord/dom-a/guild-a"
GROUP_B = "default/discord/dom-b/guild-b"
OWNER_A = group_memory.GROUP_OWNER_PREFIX + GROUP_A
OWNER_B = group_memory.GROUP_OWNER_PREFIX + GROUP_B
PRIVATE_OWNER = "person:private-a"

EVENT_A = "A群哨兵：周五去河边捡石头"
IMPRESSION_A = "他在这群里说话很稳，从不抢话"
PROMISE_A = "答应给这个群带一袋橘子"
IMPRESSION_THREAD = "线程里才看得出来他其实在犹豫"
PROMISE_THREAD = "线程里答应先替他保密"
IMPRESSION_B = "B群里他只聊工作"
IMPRESSION_PRIVATE = "私聊里他其实很累"


def _config(overrides: dict, defaults: bool = False) -> str:
    lines = ["plugins:", "  sanae_mem:", "    group_context:", "      defaults:",
             f"        long_term_memory: {str(defaults).lower()}",
             f"        short_term_shared: {str(defaults).lower()}", "      overrides:"]
    for key, value in overrides.items():
        lines.append(f'        "{key}":')
        lines.append(f"          long_term_memory: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def _write_home(home: Path, overrides: dict, defaults: bool = False) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(_config(overrides, defaults), encoding="utf-8")
    return home


def _use_home(monkeypatch, home: Path) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    return home


@contextmanager
def _turn(*, chat_id: str = "guild-a", platform: str = "discord", scope_id: str = "dom-a",
          chat_type: str = "group", thread_id: str = "", parent_chat_id: str = "",
          thread_visibility: str = "", synthetic: bool = False, sender: str = "account-a",
          session_key: str = "session-a", session_id: str = "session-a",
          message_id: str = "m-1"):
    with host_bridge.turn_envelope_scope(
        platform=platform, chat_type=chat_type, chat_id=chat_id, sender_id=sender,
        sender_id_alt="", session_id=session_id, session_key=session_key,
        message_id=message_id, scope_id=scope_id, thread_id=thread_id,
        parent_chat_id=parent_chat_id, thread_visibility=thread_visibility,
        synthetic_reply_thread=synthetic, occurred_at="2026-05-01T10:00:00+08:00",
        is_bot=False,
    ):
        yield


def _provider(home: Path, stack: ExitStack, *, session_id: str = "session-a",
              context: str = "primary") -> SanaeMemProvider:
    stack.enter_context(host_bridge.turn_envelope_scope(
        platform="cli", session_id="fixture"))
    provider = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
    })
    provider.initialize(session_id, hermes_home=str(home), platform="discord",
                        agent_context=context)
    return provider


def _tool(provider, name: str, args: dict, **kwargs) -> dict:
    return json.loads(provider.handle_tool_call(name, args, **kwargs))


def _add(provider, text: str) -> int:
    event_id = provider.add_group_event(text, importance=0.8)
    assert isinstance(event_id, int)
    return event_id


def _impression(provider, operation: str, text: str = "", *, turn_id: str,
                session_id: str = "") -> dict:
    args = {"operation": operation, **({"text": text} if text else {})}
    if group_memory.is_managed_group_scene(current_chat_scope()):
        return provider.inner_wiring.tool_impression(
            operation=operation, text=text, turn_id=turn_id, session_id=session_id, synthetic=False,
        )
    payload = _tool(provider, "sanae_impression", args, turn_id=turn_id,
                    session_id=session_id)
    assert payload["ok"], payload
    return payload["result"]


def _pre_llm(provider, message: str, session_id: str) -> str:
    payload = provider.handle_pre_llm_call(
        user_message=message, session_id=session_id, platform="discord",
        sender_id="account-a", turn_id="turn-ctx")
    assert isinstance(payload, dict)
    return str(payload.get("context") or "")


def _counts(home: Path) -> tuple:
    with sqlite3.connect(home / "data" / "sanae.db") as conn:
        return (
            conn.execute("SELECT count(*) FROM events").fetchone()[0],
            conn.execute("SELECT count(*) FROM commitments").fetchone()[0],
            conn.execute("SELECT count(*) FROM impression_revisions").fetchone()[0],
        )


@pytest.fixture
def stack():
    with ExitStack() as entered:
        yield entered


def _seed_group_a(home: Path, stack: ExitStack) -> None:
    """One group turn writes its event, this member's impression and a promise."""
    provider = _provider(home, stack)
    with _turn():
        _add(provider, EVENT_A)
        _impression(provider, "write", IMPRESSION_A, turn_id="turn-a")
        assert provider.record_commitment(
            OWNER_A, PROMISE_A, source_quote="带一袋橘子", turn_id="turn-a") is not None
    provider.shutdown()


def test_group_materials_reach_the_real_pre_llm_and_prefetch_after_a_restart(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    _seed_group_a(home, stack)

    # A brand-new provider and a brand-new session still continue the same group.
    hook = _provider(home, stack, session_id="session-b")
    with _turn(session_id="session-b", session_key="session-b"):
        context = _pre_llm(hook, EVENT_A, "session-b")
        assert EVENT_A in context
        assert IMPRESSION_A in context
        assert PROMISE_A in context
    hook.shutdown()

    recall = _provider(home, stack, session_id="session-b")
    with _turn(session_id="session-b", session_key="session-b"):
        context = recall.prefetch(EVENT_A, session_id="session-b")
        assert EVENT_A in context
        assert IMPRESSION_A in context
        assert PROMISE_A in context
        # The same turn is not delivered twice by the hook and the prefetch.
        assert _pre_llm(recall, EVENT_A, "session-b") == ""
    recall.shutdown()


def test_other_group_private_and_plain_public_never_read_group_a(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch,
                     _write_home(tmp_path / "home", {GROUP_A: True, GROUP_B: True}))
    _seed_group_a(home, stack)

    other_group = _provider(home, stack, session_id="session-b")
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b", session_id="session-b", message_id="m-b"):
        assert EVENT_A not in other_group.prefetch(EVENT_A, session_id="session-b")
        assert _pre_llm(other_group, EVENT_A, "session-b") == ""
        assert other_group.list_commitments() == []
    other_group.shutdown()

    private = _provider(home, stack, session_id="session-dm")
    with _turn(chat_type="dm", chat_id="dm-1", scope_id="dom-a",
               session_key="session-dm", session_id="session-dm", message_id="m-dm"):
        private.on_turn_start(1, "")
        assert EVENT_A not in private.prefetch(EVENT_A, session_id="session-dm")
        assert private.list_commitments() == []
        assert private.commitments.render_pending_commitments(private._owner_key) == ""
    private.shutdown()

    # An ordinary public scene keeps the approved-excerpt rule and never sees the group row.
    plain = _provider(home, stack)
    with _turn(platform="telegram", chat_id="chan-1", scope_id="", session_key="telegram-1"):
        assert plain._group_scope_grant() is None
        assert EVENT_A not in plain.prefetch(EVENT_A, session_id="session-a")
    plain.shutdown()


def test_a_group_turn_loads_no_private_profile_or_other_session(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    (home / "memories").mkdir(parents=True, exist_ok=True)
    (home / "memories" / "USER.md").write_text("PRIVATE-USER-DRAWER", encoding="utf-8")
    (home / "memories" / "MEMORY.md").write_text("PRIVATE-MEMORY-NOTEBOOK", encoding="utf-8")
    _seed_group_a(home, stack)

    provider = _provider(home, stack)
    with _turn():
        context = provider.prefetch(EVENT_A, session_id="session-a")
        assert EVENT_A in context
        assert "PRIVATE-USER-DRAWER" not in context
        assert "PRIVATE-MEMORY-NOTEBOOK" not in context
        # A recall for another session is not this turn's group context (and not public either).
        assert provider.prefetch(EVENT_A, session_id="another-session") == ""
        hook = provider.handle_pre_llm_call(
            user_message=EVENT_A, session_id="another-session", platform="discord",
            sender_id="account-a", turn_id="turn-other")
        assert "context" not in hook
    provider.shutdown()


def test_group_fulfillment_event_lands_in_the_same_group(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)

    with _turn():
        commitment_id = provider.record_commitment(
            OWNER_A, PROMISE_A, source_quote="橘子", turn_id="turn-a")
        assert provider.commitments.resolve_commitment(
            commitment_id, "fulfilled", evidence="已带") is True
        event_id = provider.commitments.record_fulfillment_event(OWNER_A, PROMISE_A, commitment_id)
        assert isinstance(event_id, int)
        row = provider._conn.execute(
            "SELECT owner_key, source_context, read_scope FROM events WHERE id=?",
            (event_id,)).fetchone()
        assert row["owner_key"] == OWNER_A
        source = parse_source(row["source_context"])
        assert source["visibility"] == "group"
        assert source["chat_id"] == "guild-a"
        assert source["author_id"] == "account-a"
        assert source["ref"] == "commitment-fulfilled:%d" % commitment_id
        assert json.loads(row["read_scope"])["kind"] == "group"
        assert "兑现" in provider.prefetch(PROMISE_A, session_id="session-a")

    thread_kwargs = dict(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
                         parent_chat_id="guild-a", thread_visibility="narrower",
                         session_key="session-t1", message_id="m-t1")
    with _turn(**thread_kwargs):
        # The narrow thread never files the parent's shared promise as its own event.
        assert provider.commitments.record_fulfillment_event(OWNER_A, PROMISE_A, commitment_id) is None
    provider.shutdown()


def test_one_person_keeps_separate_impressions_per_group_and_private(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch,
                     _write_home(tmp_path / "home", {GROUP_A: True, GROUP_B: True}))
    provider = _provider(home, stack)

    with _turn():
        _impression(provider, "write", IMPRESSION_A, turn_id="turn-a")
        target_a = provider._group_impression_target()
    with _turn(chat_id="guild-b", scope_id="dom-b", session_key="session-b",
               message_id="m-b"):
        _impression(provider, "write", IMPRESSION_B, turn_id="turn-b")
        assert provider._group_impression_target() != target_a
    with _turn():
        assert _impression(provider, "read", turn_id="turn-r")["text"] == IMPRESSION_A
    with _turn(chat_id="guild-b", scope_id="dom-b", session_key="session-b",
               message_id="m-b"):
        assert _impression(provider, "read", turn_id="turn-r")["text"] == IMPRESSION_B
    provider.shutdown()

    private = _provider(home, stack, session_id="session-dm")
    with _turn(chat_type="dm", chat_id="dm-1", scope_id="dom-a",
               session_key="session-dm", session_id="session-dm", message_id="m-dm"):
        private.on_turn_start(1, "")
        _impression(private, "write", IMPRESSION_PRIVATE, turn_id="turn-dm",
                    session_id="session-dm")
        assert _impression(private, "read", turn_id="turn-dm",
                           session_id="session-dm")["text"] == IMPRESSION_PRIVATE
        private_owner = private._owner_key
    rows = private._conn.execute(
        "SELECT owner_key, text FROM impression_revisions ORDER BY owner_key").fetchall()
    stored = {str(row["owner_key"]): str(row["text"]) for row in rows}
    assert stored[target_a[2]] == IMPRESSION_A
    assert stored[private_owner] == IMPRESSION_PRIVATE
    assert IMPRESSION_B in stored.values()
    assert len(stored) == 3
    private.shutdown()


def test_narrow_thread_reads_the_parent_impression_but_writes_only_its_own_row(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)

    with _turn():
        _impression(provider, "write", IMPRESSION_A, turn_id="turn-a")
        parent_commitment = provider.record_commitment(
            OWNER_A, PROMISE_A, source_quote="橘子", turn_id="turn-a")
        assert isinstance(parent_commitment, int)
        shared_target = provider._group_impression_target()

    thread_kwargs = dict(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
                         parent_chat_id="guild-a", thread_visibility="narrower",
                         session_key="session-t1", message_id="m-t1")
    with _turn(**thread_kwargs):
        grant = provider._group_scope_grant()
        assert grant.write_owner == OWNER_A + "/threads/thread-1"
        # The parent's row about the same member is readable here.
        assert _impression(provider, "read", turn_id="turn-t")["text"] == IMPRESSION_A
        _impression(provider, "write", IMPRESSION_THREAD, turn_id="turn-t")
        # An explicit parent owner never widens the narrow thread's write.
        thread_commitment = provider.record_commitment(
            OWNER_A, PROMISE_THREAD, source_quote="保密", turn_id="turn-t")
        # The parent's shared promise stays readable here, but the new one is this thread's own.
        assert {row["id"] for row in provider.list_commitments()} == {
            parent_commitment, thread_commitment}
        assert provider._conn.execute(
            "SELECT owner_key FROM commitments WHERE id=?",
            (thread_commitment,)).fetchone()["owner_key"] == grant.write_owner
        # The parent's shared promise is not this thread's to close.
        assert provider.commitments.resolve_commitment(parent_commitment, "cancelled") is False

    parent_row = provider._conn.execute(
        "SELECT text FROM impression_revisions WHERE owner_key=? ORDER BY revision DESC",
        (shared_target[2],)).fetchone()
    assert parent_row["text"] == IMPRESSION_A
    pending_parent = provider._conn.execute(
        "SELECT status, version FROM commitments WHERE id=?", (parent_commitment,)).fetchone()
    assert (pending_parent["status"], pending_parent["version"]) == ("pending", 1)

    with _turn():
        assert _impression(provider, "read", turn_id="turn-p")["text"] == IMPRESSION_A
        assert [row["text"] for row in provider.list_commitments()] == [PROMISE_A]
        context = provider.prefetch(PROMISE_THREAD, session_id="session-a")
        assert IMPRESSION_THREAD not in context
        assert PROMISE_THREAD not in context
        assert provider.commitments.resolve_commitment(thread_commitment, "cancelled") is False

    sibling = dict(thread_kwargs, chat_id="thread-2", thread_id="thread-2",
                   session_key="session-t2", message_id="m-t2",
                   thread_visibility="unknown")
    with _turn(**sibling):
        # An unknown thread audience never reads the parent's shared row: the parent id alone is
        # not an audience statement.
        assert _impression(provider, "read", turn_id="turn-s")["text"] == ""
        assert provider.list_commitments() == []
    inherited = dict(thread_kwargs, chat_id="thread-3", thread_id="thread-3",
                     session_key="session-t3", message_id="m-t3",
                     thread_visibility="inherited")
    with _turn(**inherited):
        # A platform-stated inherited audience is the whole group, so the parent rows are read.
        assert _impression(provider, "read", turn_id="turn-i")["text"] == IMPRESSION_A
        assert [row["text"] for row in provider.list_commitments()] == [PROMISE_A]
    provider.shutdown()


def test_inherited_thread_keeps_its_own_unknown_era_impression_only(
        tmp_path, monkeypatch, stack):
    """The exact row written while the audience was unknown survives the later ``inherited``."""
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)
    thread = dict(chat_type="thread", chat_id="thread-3", thread_id="thread-3",
                  parent_chat_id="guild-a", session_key="session-t3", message_id="m-t3")

    with _turn(**dict(thread, thread_visibility="unknown")):
        assert provider._group_scope_grant().read_owners == (OWNER_A + "/threads/thread-3",)
        _impression(provider, "write", IMPRESSION_THREAD, turn_id="turn-t3")
        exact_target = provider._group_impression_target()

    with _turn(**dict(thread, thread_visibility="inherited", message_id="m-t3b")):
        grant = provider._group_scope_grant()
        assert grant.write_owner == OWNER_A
        # The same person row of the write owner comes first, then the granted exact thread row.
        assert exact_target[2] != OWNER_A + "#" + grant.person_key, \
            "the row under the old unknown-era exact owner is a different row"
        assert group_memory.member_impression_read_owners(grant) == (
            OWNER_A + "#" + grant.person_key, exact_target[2])
        assert _impression(provider, "read", turn_id="turn-t4")["text"] == IMPRESSION_THREAD
        assert IMPRESSION_THREAD in provider.prefetch("线程里的旧事", session_id="session-a")

    sibling = dict(thread, chat_id="thread-4", thread_id="thread-4",
                   session_key="session-t4", message_id="m-t4",
                   thread_visibility="unknown")
    with _turn(**sibling):
        # A sibling thread is outside this grant: its own row is all it can read.
        assert _impression(provider, "read", turn_id="turn-s")["text"] == ""
    provider.shutdown()

    parent = _provider(home, stack)
    with _turn():
        assert _impression(parent, "read", turn_id="turn-p")["text"] == ""
        assert IMPRESSION_THREAD not in parent.prefetch("线程里的旧事", session_id="session-a")
    parent.shutdown()

    private = _provider(home, stack, session_id="session-dm")
    with _turn(chat_type="dm", chat_id="dm-1", scope_id="dom-a",
               session_key="session-dm", session_id="session-dm", message_id="m-dm"):
        private.on_turn_start(1, "")
        assert _impression(private, "read", turn_id="turn-dm",
                           session_id="session-dm")["text"] == ""
    private.shutdown()


def test_switching_the_group_off_stops_reads_and_writes_but_keeps_the_rows(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    _seed_group_a(home, stack)
    rows_before = _counts(home)

    _write_home(home, {GROUP_A: False})
    closed = _provider(home, stack, session_id="session-b")
    with _turn(session_id="session-b", session_key="session-b"):
        assert closed._group_scope_grant() is None
        assert closed.add_group_event("关闭后不许再写") is None
        assert closed.record_commitment(OWNER_A, "关闭后不许再记") is None
        assert closed._group_impression_target() is None
        assert EVENT_A not in _pre_llm(closed, EVENT_A, "session-b")
        assert EVENT_A not in closed.prefetch(EVENT_A, session_id="session-b")
        assert closed.list_commitments() == []
        assert closed.commitments.render_pending_commitments(OWNER_A) == ""
    assert _counts(home) == rows_before
    closed.shutdown()

    _write_home(home, {GROUP_A: True})
    reopened = _provider(home, stack, session_id="session-c")
    with _turn(session_id="session-c", session_key="session-c"):
        assert EVENT_A in _pre_llm(reopened, EVENT_A, "session-c")
    reopened.shutdown()


def test_group_provenance_and_cas_survive_the_group_entry(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)

    with _turn():
        event_id = _add(provider, EVENT_A)
        source = provider.event_store.event_source_context(event_id)
        assert source["author_id"] == "account-a"
        assert source["chat_id"] == "guild-a"
        assert source["ref"] == "m-1"
        assert source["time"] == "2026-05-01T10:00:00+08:00"
        assert source["visibility"] == "group"
        commitment_id = provider.record_commitment(
            OWNER_A, PROMISE_A, source_quote="带一袋橘子", turn_id="turn-a")
        row = provider._conn.execute(
            "SELECT owner_key, source_quote, source_context, version FROM commitments "
            "WHERE id=?", (commitment_id,)).fetchone()
        assert row["owner_key"] == OWNER_A
        assert row["source_quote"] == "带一袋橘子"
        stored = parse_source(row["source_context"])
        assert (stored["chat_id"], stored["ref"], stored["time"]) == (
            "guild-a", "m-1", "2026-05-01T10:00:00+08:00")
        # A wrong revision never overwrites the row.
        assert provider.commitments.resolve_commitment(
            commitment_id, "cancelled", evidence="stale", expected_version=7) is False
        current = provider._conn.execute(
            "SELECT status, version FROM commitments WHERE id=?", (commitment_id,)).fetchone()
        assert (current["status"], current["version"]) == ("pending", 1)
        assert provider.commitments.resolve_commitment(
            commitment_id, "cancelled", evidence="cancelled", expected_version=1) is True
        current = provider._conn.execute(
            "SELECT status, version, source_context FROM commitments WHERE id=?",
            (commitment_id,)).fetchone()
        assert (current["status"], current["version"]) == ("cancelled", 2)
        assert parse_source(current["source_context"])["chat_id"] == "guild-a"

        _impression(provider, "write", IMPRESSION_A, turn_id="turn-a")
        target = provider._group_impression_target()
        impression_row = provider._conn.execute(
            "SELECT revision, source_context FROM impression_revisions "
            "WHERE owner_key=? ORDER BY revision DESC", (target[2],)).fetchone()
        assert impression_row["revision"] == 1
        impression_source = parse_source(impression_row["source_context"])
        assert (impression_source["kind"], impression_source["chat_id"],
                impression_source["ref"]) == ("group_member", "guild-a", "m-1")

    thread_kwargs = dict(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
                         parent_chat_id="guild-a", thread_visibility="narrower",
                         session_key="session-t1", message_id="m-t1")
    with _turn(**thread_kwargs):
        _impression(provider, "write", IMPRESSION_THREAD, turn_id="turn-t")
        target = provider._group_impression_target()
        grant = provider._group_scope_grant()
        assert target[2] == OWNER_A + "/threads/thread-1#" + grant.person_key
        source = parse_source(provider._conn.execute(
            "SELECT source_context FROM impression_revisions WHERE owner_key=? "
            "ORDER BY revision DESC", (target[2],)).fetchone()["source_context"])
        assert (source["chat_id"], source["thread_id"], source["parent_chat_id"],
                source["thread_visibility"], source["ref"], source["time"]) == (
            "thread-1", "thread-1", "guild-a", "narrower", "m-t1",
            "2026-05-01T10:00:00+08:00")
    provider.shutdown()


def test_a_group_turn_only_ever_writes_its_own_scene_owner(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch,
                     _write_home(tmp_path / "home", {GROUP_A: True, GROUP_B: True}))
    provider = _provider(home, stack)
    values = ("2026-05-01", "guarded row", "kw", 0.8, 0.6, "topic", None,
              "2026-05-01", None, "session-a")

    with _turn():
        assert provider.event_store.insert_event_row(
            "forged-private", values, owner_key=PRIVATE_OWNER) is None
        assert provider.event_store.insert_event_row("forged-other-group", values, owner_key=OWNER_B) is None
        assert isinstance(provider.event_store.insert_event_row(
            "own-group", values, owner_key=OWNER_A), int)
    assert provider._conn.execute(
        "SELECT count(*) FROM events WHERE owner_key=?", (PRIVATE_OWNER,)).fetchone()[0] == 0

    _write_home(home, {GROUP_A: False})
    with _turn():
        assert provider.event_store.insert_event_row("closed-group", values, owner_key=OWNER_A) is None
    assert provider._conn.execute(
        "SELECT count(*) FROM events WHERE owner_key=?",
        (OWNER_A,)).fetchone()[0] == 1
    provider.shutdown()


@runtime_checkable
class _GroupWriteProvider(Protocol):
    def add_group_event(self, text: str) -> int | None: ...

    def record_commitment(self, owner_key: str, text: str, *, turn_id: str) -> int | None: ...


def test_group_context_dedups_across_split_namespaces_handles_pure_image_and_shows_provenance(
        tmp_path, monkeypatch, stack):
    import shutil
    import yaml
    from hermes_cli.plugins import PluginManager
    from plugins.memory import load_memory_provider

    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    path = home / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["plugins"]["enabled"] = ["sanae_mem"]
    config["plugins"]["sanae_mem"].update(data_dir=str(home / "data"), semantic_backend="sqlite")
    config["memory"] = {"provider": "sanae_mem"}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    shutil.copytree(Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem", home / "plugins" / "sanae_mem",
                    ignore=shutil.ignore_patterns("tests", "__pycache__"))
    manager = PluginManager(scope_key=str(home))
    stack.callback(manager.unload)
    manager.discover_and_load()
    assert any(row["name"] == "sanae_mem" and row["enabled"] for row in manager.list_plugins())
    provider = load_memory_provider("sanae_mem", register_skills=False)
    assert provider is not None
    assert type(provider).__module__.startswith("_hermes_user_memory.")
    assert isinstance(provider, _GroupWriteProvider)
    provider.initialize("session-split", hermes_home=str(home), platform="discord",
                        agent_context="primary")
    shadow = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": True,
    })
    shadow.initialize("session-split", hermes_home=str(home), platform="discord",
                      agent_context="primary")

    def invoke_group_hook(**turn):
        results = manager.invoke_hook("pre_llm_call", **turn)
        payload, = [result for result in results
                    if isinstance(result, dict) and result.get("ephemeral") and result.get("context")]
        return payload

    try:
        with _turn(
            session_id="session-split",
            session_key="session-split",
            thread_id="thread-7",
            parent_chat_id="guild-a",
            chat_type="thread",
            thread_visibility="narrower",
            synthetic=False,
        ):
            from sanae_mem.governance.source_context import current_chat_scope, group_source_record, source_json

            event_id = provider.add_group_event(EVENT_A)
            assert isinstance(event_id, int)
            relayed_src = source_json(
                group_source_record(current_chat_scope(), paraphrase_ref="msg-orig-9"))
            with closing(sqlite3.connect(home / "data" / "sanae.db")) as db, db:
                db.execute("UPDATE events SET source_context=? WHERE id=?", (relayed_src, event_id))
            _impression(provider, "write", IMPRESSION_A, turn_id="turn-imp")
            provider.record_commitment(OWNER_A, PROMISE_A, turn_id="turn-promise")

            # Correct turn order: on_turn_start -> pre_llm_call (duplicate namespace) -> prefetch
            provider.on_turn_start(1, "橘子", author_id="account-a")
            hook_payload = invoke_group_hook(
                session_id="session-split",
                turn_id="turn-1",
                user_message="带一袋橘子怎么样",
            )
            assert hook_payload.get("ephemeral") is True
            ctx = hook_payload.get("context") or ""
            assert EVENT_A in ctx
            assert IMPRESSION_A in ctx
            assert PROMISE_A in ctx
            assert "发言人:account-a" in ctx
            assert "线程:thread-7" in ctx
            assert "转述自:msg-orig-9" in ctx
            assert "非维护者私域" not in ctx
            # Same-turn prefetch must dedup cleanly without duplicating group context.
            assert provider.prefetch("带一袋橘子怎么样", session_id="session-split") == ""
            # Shadow provider with the same session must still stay silent.
            assert shadow.handle_pre_llm_call(
                session_id="session-split",
                turn_id="turn-1",
                user_message="带一袋橘子怎么样",
            ) == {}

            # Next turn with a pure-image payload (empty text query): impression & commitments still load.
            provider.on_turn_start(2, "", author_id="account-a")
            image_payload = invoke_group_hook(
                session_id="session-split",
                turn_id="turn-2",
                user_message=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
            )
            assert image_payload.get("ephemeral") is True
            image_ctx = image_payload.get("context") or ""
            assert IMPRESSION_A in image_ctx
            assert PROMISE_A in image_ctx
            assert EVENT_A not in image_ctx
            assert provider.prefetch("", session_id="session-split") == ""

            # Turn 3 without on_turn_start still starts fresh when turn_id changes, and
            # empty-query prefetch delivers when no hook already sent it.
            provider.on_turn_start(3, "", author_id="account-a")
            direct_prefetch = provider.prefetch("", session_id="session-split")
            assert IMPRESSION_A in direct_prefetch
            assert PROMISE_A in direct_prefetch
    finally:
        shadow.shutdown()
        provider.shutdown()
