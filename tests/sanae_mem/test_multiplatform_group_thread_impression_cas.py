"""Task 04: the automatic group impression keeps the CAS base of its own target.

Every case runs through the real ``SanaeMemProvider`` over a temporary HOME and its real SQLite
database: the same grant resolution, the same ``read_impression``/``apply_impression`` CAS, the
same ``apply_group_delivery`` worker and the same ``sanae_impression`` tool. Only the auxiliary
model is doubled, with the same stand-in shape the receipt suite uses. The "concurrent" writer
inside those model calls is real too: production SQL, production grant, no stubbed gate.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem import SanaeMemProvider  # noqa: E402
from sanae_mem.runtime import group_ingest, group_memory  # noqa: E402
from sanae_mem.infra import host_bridge  # noqa: E402
from sanae_mem.governance.source_context import parse_source  # noqa: E402

GROUP_A = "default/discord/dom-a/guild-a"
OWNER_A = group_memory.GROUP_OWNER_PREFIX + GROUP_A

USER_TEXT = "周六下午我去河边捡石头，顺便把上次借你的那本书带过去。"
REPLY_TEXT = "好，那周六见。"
QUOTE = "周六下午我去河边捡石头"
PARENT_IMPRESSION = "他在大群里说话很稳，从不抢话"
PARENT_SECOND = "他这周在大群里开始主动问时间"
OLD_THREAD_IMPRESSION = "他在线程里其实一直在犹豫"
CONCURRENT_IMPRESSION = "他在线程里已经把东西都收拾好了"
NEW_IMPRESSION = "他一直在张罗出门捡石头这件事"
WRONG_QUOTE = "他从来没在这个群里说过话"

THREAD_1_TURN = dict(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
                     parent_chat_id="guild-a", thread_visibility="narrower",
                     session_key="session-t1", session_id="session-t1", message_id="m-t1")


def _config(overrides: dict) -> str:
    lines = ["plugins:", "  sanae_mem:", "    group_context:", "      defaults:",
             "        long_term_memory: false", "      overrides:"]
    for key, value in overrides.items():
        lines.append(f'        "{key}":')
        lines.append(f"          long_term_memory: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def _write_home(home: Path, overrides: dict) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(_config(overrides), encoding="utf-8")
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


def _provider(home: Path, stack: ExitStack,
              *, session_id: str = "session-a") -> SanaeMemProvider:
    stack.enter_context(host_bridge.turn_envelope_scope(
        platform="cli", session_id="fixture"))
    provider = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
    })
    provider.initialize(session_id, hermes_home=str(home), platform="discord")
    return provider


def _tool(provider, name: str, args: dict, **kwargs) -> dict:
    return json.loads(provider.handle_tool_call(name, args, **kwargs))


def _impression(provider, operation: str, text: str = "", *, turn_id: str) -> dict:
    return provider.inner_wiring.tool_impression(
        operation=operation, text=text, session_id="", turn_id=turn_id, synthetic=False,
    )


class _Flash:
    """The auxiliary model stand-in; ``on_call`` runs while the model boundary is open."""

    def __init__(self, payload: dict, on_call=None):
        self.payload = payload
        self.on_call = on_call
        self.requests = []
        self.llm = SimpleNamespace(complete=self.complete)

    def complete(self, messages, **kwargs):
        self.requests.append({"messages": messages, **kwargs})
        if self.on_call is not None:
            self.on_call()
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


def _payload(*, text: str = NEW_IMPRESSION, quote: str = QUOTE) -> dict:
    return {
        "events": [], "impression": {"speaker": "user", "text": text, "quote": quote},
        "new_commitments": [], "commitment_updates": [],
    }


def _deliver(provider, flash: _Flash) -> dict:
    provider._llm = flash.llm
    return group_ingest.apply_group_delivery(
        provider, user_message=USER_TEXT, assistant_response=REPLY_TEXT,
        turn_id="turn-t1", delivered_at=time.time() + 60, delivery_ref="out-t1")


def _impression_rows(home: Path, owner: str) -> list:
    return _conn_rows(
        home, "SELECT revision, text, evidence, source_context FROM impression_revisions "
              "WHERE owner_key=? ORDER BY revision", (owner,))


def _conn_row(home: Path, sql: str, params=()):
    with sqlite3.connect(home / "data" / "sanae.db") as conn:
        return conn.execute(sql, params).fetchone()


def _conn_rows(home: Path, sql: str, params=()):
    with sqlite3.connect(home / "data" / "sanae.db") as conn:
        return conn.execute(sql, params).fetchall()


@pytest.fixture
def stack():
    with ExitStack() as entered:
        yield entered


def _seed_parent(home: Path, stack: ExitStack) -> str:
    """An ordinary group turn already wrote this member's shared impression (revision 1)."""
    seed = _provider(home, stack)
    with _turn():
        _impression(seed, "write", PARENT_IMPRESSION, turn_id="turn-a")
        owner = seed._group_impression_target()[2]
    seed.shutdown()
    return owner


def test_narrow_thread_model_reads_the_parent_text_and_writes_its_own_revision_one(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    parent_owner = _seed_parent(home, stack)
    parent_before = _impression_rows(home, parent_owner)
    assert [(row[0], row[1]) for row in parent_before] == [(1, PARENT_IMPRESSION)]

    provider = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        grant = provider._group_scope_grant()
        target = provider._group_impression_target()
        assert target[2] == OWNER_A + "/threads/thread-1#" + grant.person_key
        assert _impression_rows(home, target[2]) == [], "the exact target starts empty"
        flash = _Flash(_payload())
        result = _deliver(provider, flash)
        assert result["ok"] is True and result["impression"] == NEW_IMPRESSION
        # The model may see the parent's row as reference; it is not this target's base.
        assert PARENT_IMPRESSION in flash.requests[0]["messages"][1]["content"]
    provider.shutdown()

    rows = _impression_rows(home, target[2])
    assert len(rows) == 1
    revision, text, evidence, source_json_text = rows[0]
    assert (revision, text, evidence) == (1, NEW_IMPRESSION, QUOTE)
    source = parse_source(source_json_text)
    assert (source["ref"], source["thread_id"], source["visibility"],
            source["processing"]) == ("m-t1", "thread-1", "group", "group_receipt_ingest")
    # The parent's row keeps its own text, revision 1 and provenance.
    assert _impression_rows(home, parent_owner) == parent_before
    # No other scene took a row from this turn.
    assert _conn_row(home, "SELECT count(*) FROM impression_revisions")[0] == 2

    reader = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        assert _impression(reader, "read", turn_id="turn-read")["text"] == NEW_IMPRESSION
    reader.shutdown()

    parent = _provider(home, stack)
    with _turn():
        assert _impression(parent, "read", turn_id="turn-read-parent")["text"] == PARENT_IMPRESSION
    parent.shutdown()

    sibling = _provider(home, stack, session_id="session-t4")
    with _turn(chat_id="thread-4", chat_type="thread", thread_id="thread-4",
               parent_chat_id="guild-a", thread_visibility="unknown",
               session_key="session-t4", session_id="session-t4", message_id="m-t4"):
        assert _impression(sibling, "read", turn_id="turn-sibling")["text"] == ""
    sibling.shutdown()


def test_an_empty_target_with_no_parent_row_still_starts_at_revision_one(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        target = provider._group_impression_target()
        flash = _Flash(_payload())
        result = _deliver(provider, flash)
        assert result["ok"] is True and result["impression"] == NEW_IMPRESSION
        assert "（旧版，可为空）】\n无" in flash.requests[0]["messages"][1]["content"]
    provider.shutdown()

    rows = _impression_rows(home, target[2])
    assert [(row[0], row[1], row[2]) for row in rows] == [(1, NEW_IMPRESSION, QUOTE)]


@pytest.mark.parametrize("existing", [False, True])
def test_a_concurrent_write_on_the_same_target_is_never_overwritten(
        tmp_path, monkeypatch, stack, existing):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    seed = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        if existing:
            _impression(seed, "write", OLD_THREAD_IMPRESSION, turn_id="turn-old")
        target = seed._group_impression_target()
    seed.shutdown()

    concurrent = _provider(home, stack, session_id="session-t1")

    def during_model():
        # A real impression write lands on this same target while the model is out.
        _impression(concurrent, "write", CONCURRENT_IMPRESSION, turn_id="turn-concurrent")

    provider = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        flash = _Flash(_payload(), on_call=during_model)
        result = _deliver(provider, flash)
        assert len(flash.requests) == 1
        assert result["impression"] is None, "the stale candidate must not be applied"
    provider.shutdown()
    concurrent.shutdown()

    rows = _impression_rows(home, target[2])
    expected = [(1, OLD_THREAD_IMPRESSION), (2, CONCURRENT_IMPRESSION)] if existing \
        else [(1, CONCURRENT_IMPRESSION)]
    assert [(row[0], row[1]) for row in rows] == expected


def test_a_parent_write_during_the_boundary_never_becomes_the_target_version(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    parent_owner = _seed_parent(home, stack)
    parent_before = _impression_rows(home, parent_owner)
    parent_writer = _provider(home, stack)

    def during_model():
        with _turn():
            _impression(parent_writer, "write", PARENT_SECOND, turn_id="turn-parent-2")

    provider = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        target = provider._group_impression_target()
        flash = _Flash(_payload(), on_call=during_model)
        result = _deliver(provider, flash)
        assert result["impression"] == NEW_IMPRESSION
    provider.shutdown()
    parent_writer.shutdown()

    rows = _impression_rows(home, target[2])
    assert [(row[0], row[1]) for row in rows] == [(1, NEW_IMPRESSION)]
    parent_rows = _impression_rows(home, parent_owner)
    assert [(row[0], row[1]) for row in parent_rows] == [
        (1, PARENT_IMPRESSION), (2, PARENT_SECOND)]
    assert parent_rows[0] == parent_before[0], "the parent's own revision 1 stays as it was"


def test_wrong_quote_and_a_closed_or_foreign_home_write_no_impression(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        flash = _Flash(_payload(quote=WRONG_QUOTE))
        result = _deliver(provider, flash)
        assert result["ok"] is True and result["impression"] is None
    provider.shutdown()
    assert _conn_row(home, "SELECT count(*) FROM impression_revisions")[0] == 0

    _write_home(home, {GROUP_A: False})
    closed = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        flash = _Flash(_payload())
        result = _deliver(closed, flash)
        assert result["ok"] is False and result["reason"] == "group_switched_off"
        assert flash.requests == [], "a switched-off group never reaches the model"
    closed.shutdown()
    assert _conn_row(home, "SELECT count(*) FROM impression_revisions")[0] == 0

    _write_home(home, {GROUP_A: True})
    foreign = _provider(home, stack, session_id="session-t1")
    outside = _write_home(tmp_path / "foreign", {GROUP_A: True})
    monkeypatch.setenv("HERMES_HOME", str(outside))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(outside))
    with _turn(**THREAD_1_TURN):
        flash = _Flash(_payload())
        result = _deliver(foreign, flash)
        assert result["ok"] is False and result["reason"] == "group_switched_off"
        assert flash.requests == [], "another HOME never authorizes this worker's group"
    foreign.shutdown()
    assert _conn_row(home, "SELECT count(*) FROM impression_revisions")[0] == 0
