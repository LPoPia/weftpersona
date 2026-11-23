"""Task 04: thread promises — who reads them, who may close them, where the record lands.

Every case runs through the real ``SanaeMemProvider`` over a temporary HOME and its real SQLite
database: the same grant resolution, the same ``list_commitments``/``resolve_commitment`` SQL and
CAS, the same ``apply_group_delivery`` worker and the same ``insert_event_row`` gate. Only the
auxiliary model is doubled, with the same stand-in shape the receipt suite uses. No grant, SQL
predicate or ownership check is stubbed out.
"""

from __future__ import annotations

import json
import sqlite3
import sys
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
GROUP_B = "default/discord/dom-b/guild-b"
OWNER_A = group_memory.GROUP_OWNER_PREFIX + GROUP_A
OWNER_B = group_memory.GROUP_OWNER_PREFIX + GROUP_B
THREAD_1 = OWNER_A + "/threads/thread-1"
THREAD_3 = OWNER_A + "/threads/thread-3"
THREAD_4 = OWNER_A + "/threads/thread-4"

PROMISE_A = "答应给这个群带一袋橘子"
PROMISE_T = "答应在这个线程里先留一封短信"
PROMISE_T4 = "线程四里答应替他留一份旧名单"
PROMISE_B = "答应给那个群整理一份工作清单"

CONFIRM_USER = "已经收到了，短信我看过了"
CONFIRM_REPLY = "好，那这件事就完成了"
COMPLETION = "我已兑现此前答应的事"


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


def _provider(home: Path, stack: ExitStack, *, session_id: str = "session-a") -> SanaeMemProvider:
    stack.enter_context(host_bridge.turn_envelope_scope(
        platform="cli", session_id="fixture"))
    provider = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
    })
    provider.initialize(session_id, hermes_home=str(home), platform="discord")
    return provider


def _pre_llm(provider, message: str, session_id: str) -> str:
    payload = provider.handle_pre_llm_call(
        user_message=message, session_id=session_id, platform="discord",
        sender_id="account-a", turn_id="turn-ctx")
    assert isinstance(payload, dict)
    return str(payload.get("context") or "")


class _Flash:
    """The auxiliary model stand-in: one fixed answer, every real request recorded."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.requests = []
        self.llm = SimpleNamespace(complete=self.complete)

    def complete(self, messages, **kwargs):
        self.requests.append({"messages": messages, **kwargs})
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


def _row(provider, sql: str, params=()):
    return provider._conn.execute(sql, params).fetchone()


def _event_values(raw: str) -> tuple:
    return ("2026-05-01T10:00:00+08:00", raw, "kw", 0.8, 0.6, "其他", None,
            "2026-05-01T10:00:00+08:00", None, "session-a")


def _result() -> dict:
    return {"ok": True, "reason": "", "events": [], "impression": None,
            "commitments": [], "updates": []}


THREAD_1_TURN = dict(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
                     parent_chat_id="guild-a", thread_visibility="narrower",
                     session_key="session-t1", session_id="session-t1", message_id="m-t1")
THREAD_3_TURN = dict(chat_type="thread", chat_id="thread-3", thread_id="thread-3",
                     parent_chat_id="guild-a", thread_visibility="unknown",
                     session_key="session-t3", session_id="session-t3", message_id="m-t3")


@pytest.fixture
def stack():
    with ExitStack() as entered:
        yield entered


def _seed_unknown_promise(home: Path, stack: ExitStack) -> tuple:
    """The promise this thread filed while its audience was still ``unknown``."""
    seed = _provider(home, stack, session_id="session-t3")
    with _turn(**THREAD_3_TURN):
        grant = seed._group_scope_grant()
        assert grant.write_owner == THREAD_3
        promise_id = seed.record_commitment(
            OWNER_A, PROMISE_T, source_quote="留一封短信", turn_id="old-exact")
        assert isinstance(promise_id, int)
        source = _row(seed, "SELECT source_context FROM commitments WHERE id=?",
                      (promise_id,))["source_context"]
    seed.shutdown()
    return promise_id, source


def test_narrow_thread_reads_the_parent_promise_in_both_entries_but_cannot_close_it(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    seed = _provider(home, stack)
    with _turn():
        parent_id = seed.record_commitment(
            OWNER_A, PROMISE_A, source_quote="橘子", turn_id="turn-a")
        assert isinstance(parent_id, int)
    seed.shutdown()

    hook = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        grant = hook._group_scope_grant()
        assert (grant.shared_owner, grant.exact_owner) == (OWNER_A, THREAD_1)
        assert grant.write_owner == THREAD_1
        # The parent's shared promise is readable here, but it is not this thread's to resolve.
        assert grant.resolution_owners == (THREAD_1,)
        assert PROMISE_A in _pre_llm(hook, PROMISE_A, "session-t1")
    hook.shutdown()

    recall = _provider(home, stack, session_id="session-t1")
    with _turn(**THREAD_1_TURN):
        assert PROMISE_A in recall.prefetch(PROMISE_A, session_id="session-t1")
        assert recall.commitments.resolve_commitment(parent_id, "cancelled", evidence="不算") is False

        snapshot = group_ingest._pending_snapshot(recall)
        assert [item["id"] for item in snapshot] == []
        flash = _Flash({
            "events": [], "impression": None, "new_commitments": [],
            "commitment_updates": [{
                "id": parent_id, "action": "fulfilled", "basis": "user_confirmed",
                "quote": CONFIRM_USER,
            }],
        })
        recall._llm = flash.llm
        result = group_ingest.apply_group_delivery(
            recall, user_message=CONFIRM_USER, assistant_response=CONFIRM_REPLY,
            turn_id="turn-t1", delivered_at=1794946929.0, delivery_ref="out-t1")
        assert result["updates"] == []
        # The parent's promise is never offered to the model as this thread's pending item.
        assert PROMISE_A not in flash.requests[0]["messages"][1]["content"]
    recall.shutdown()

    parent = _conn_row(home, "SELECT status, version FROM commitments WHERE id=?", (parent_id,))
    assert parent == ("pending", 1)
    assert _conn_row(home, "SELECT count(*) FROM events")[0] == 0


def test_inherited_thread_continues_the_exact_promise_it_filed_while_unknown(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    promise_id, original_source = _seed_unknown_promise(home, stack)
    inherited = dict(THREAD_3_TURN, thread_visibility="inherited", message_id="m-i1")

    hook = _provider(home, stack, session_id="session-t3")
    with _turn(**inherited):
        grant = hook._group_scope_grant()
        assert grant.write_owner == OWNER_A
        assert set(grant.resolution_owners) == {OWNER_A, THREAD_3}
        assert len(grant.resolution_owners) == 2
        # The exact-era promise is in the first automatic entry.
        assert PROMISE_T in _pre_llm(hook, PROMISE_T, "session-t3")
        snapshot = group_ingest._pending_snapshot(hook)
        assert [(item["id"], item["version"], item["owner"]) for item in snapshot] == [
            (promise_id, 1, THREAD_3)]
    hook.shutdown()

    recall = _provider(home, stack, session_id="session-t3")
    with _turn(**inherited):
        flash = _Flash({
            "events": [], "impression": None, "new_commitments": [],
            "commitment_updates": [{
                "id": promise_id, "action": "fulfilled", "basis": "user_confirmed",
                "quote": CONFIRM_USER,
            }],
        })
        recall._llm = flash.llm
        result = group_ingest.apply_group_delivery(
            recall, user_message=CONFIRM_USER, assistant_response=CONFIRM_REPLY,
            turn_id="turn-inherited", delivered_at=1794946929.0, delivery_ref="out-t3")
        assert result["ok"] is True and result["reason"] == ""
        assert result["updates"] == [{"id": promise_id, "action": "fulfilled"}]
        # The model saw this promise with its real id and version, and the real confirmation.
        prompt = flash.requests[0]["messages"][1]["content"]
        assert f'"id": {promise_id}, "text": "{PROMISE_T}", "quote": "留一封短信", "version": 1' \
            in prompt
        assert CONFIRM_USER in prompt
        assert OWNER_A not in prompt and THREAD_3 not in prompt, \
            "owner authorization never leaves this process"
        assert PROMISE_T in recall.prefetch(PROMISE_T, session_id="session-t3")
    recall.shutdown()

    resolved = _conn_row(
        home, "SELECT owner_key, source_context, status, version, resolution_evidence, "
              "fulfilled_at FROM commitments WHERE id=?", (promise_id,))
    owner, source_context, status, version, evidence, fulfilled_at = resolved
    assert owner == THREAD_3, "the promise keeps its original exact owner"
    assert source_context == original_source, "the promise's own provenance is untouched"
    assert (status, version) == ("fulfilled", 2)
    assert evidence.startswith("user_confirmation:") and CONFIRM_USER in evidence
    assert fulfilled_at

    events = _conn_rows(
        home, "SELECT id, owner_key, raw, source_context, read_scope FROM events")
    assert len(events) == 1
    event_id, event_owner, raw, event_source, read_scope = events[0]
    assert event_owner == THREAD_3, "the completion record stays on the exact owner"
    assert COMPLETION in raw and PROMISE_T in raw
    source = parse_source(event_source)
    assert source["ref"] == "m-i1", "the real incoming ref of this turn is kept"
    assert source["commitment_id"] == str(promise_id)
    assert source["delivery_ref"] == "out-t3"
    assert source["thread_visibility"] == "inherited"
    assert json.loads(read_scope) == {"kind": "group", "owner": THREAD_3}

    # This new completion event kept its first utterance from the same authorized insert.
    occurrences = _conn_rows(
        home, "SELECT raw, observed_at, source_snapshot FROM event_occurrences WHERE event_id=?",
        (event_id,))
    assert len(occurrences) == 1
    occurrence_raw, observed_at, snapshot = occurrences[0]
    assert COMPLETION in occurrence_raw and PROMISE_T in occurrence_raw
    assert occurrence_raw in raw
    assert observed_at and snapshot == ""

    # This thread reads its own completion; the parent, a sibling and a private chat do not.
    reader = _provider(home, stack, session_id="session-t3")
    with _turn(**inherited):
        context = reader.prefetch(PROMISE_T, session_id="session-t3")
        assert COMPLETION in context and PROMISE_T in context
        assert reader.event_store.get_event_row_by_id(event_id) is not None
        assert len(reader.event_store.query_event_occurrences([event_id]).get(event_id, [])) == 1
    reader.shutdown()

    parent = _provider(home, stack)
    with _turn():
        context = parent.prefetch(PROMISE_T, session_id="session-a")
        assert COMPLETION not in context and PROMISE_T not in context
        assert parent.list_commitments() == []
        # The parent may neither read nor append to the exact-era completion record.
        assert parent.event_store.get_event_row_by_id(event_id) is None
        assert parent.event_store.record_event_occurrence(
            event_id, "parent-append", "2026-05-02T10:00:00+08:00") is False
    parent.shutdown()

    sibling = _provider(home, stack, session_id="session-t4")
    with _turn(chat_id="thread-4", chat_type="thread", thread_id="thread-4",
               parent_chat_id="guild-a", thread_visibility="unknown",
               session_key="session-t4", session_id="session-t4", message_id="m-t4"):
        context = sibling.prefetch(PROMISE_T, session_id="session-t4")
        assert COMPLETION not in context and PROMISE_T not in context
        assert sibling.event_store.get_event_row_by_id(event_id) is None
        assert sibling.event_store.record_event_occurrence(
            event_id, "sibling-append", "2026-05-02T10:00:00+08:00") is False
    sibling.shutdown()
    assert _conn_rows(
        home, "SELECT raw FROM event_occurrences WHERE event_id=?", (event_id,)) == [
        (occurrence_raw,)]

    private = _provider(home, stack, session_id="session-dm")
    with _turn(chat_type="dm", chat_id="dm-1", scope_id="dom-a", session_key="session-dm",
               session_id="session-dm", message_id="m-dm"):
        private.on_turn_start(1, "")
        context = private.prefetch(PROMISE_T, session_id="session-dm")
        assert COMPLETION not in context and PROMISE_T not in context
        assert private.list_commitments() == []
    private.shutdown()


def test_a_plain_group_completion_keeps_the_shared_owner(tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)
    with _turn():
        grant = provider._group_scope_grant()
        assert grant.resolution_owners == (OWNER_A,)
        promise_id = provider.record_commitment(
            OWNER_A, PROMISE_A, source_quote="橘子", turn_id="turn-a")
        assert provider.commitments.resolve_commitment(promise_id, "fulfilled", evidence="已带") is True
        event_id = provider.commitments.record_fulfillment_event(OWNER_A, PROMISE_A, promise_id)
        assert isinstance(event_id, int)
        row = _row(provider, "SELECT owner_key, read_scope, source_context FROM events WHERE id=?",
                   (event_id,))
        assert row["owner_key"] == OWNER_A
        assert json.loads(row["read_scope"]) == {"kind": "group", "owner": OWNER_A}
        assert parse_source(row["source_context"])["commitment_id"] == str(promise_id)
        assert PROMISE_A in provider.prefetch(PROMISE_A, session_id="session-a")
    provider.shutdown()


def test_foreign_sibling_private_and_wrong_quotes_never_update_a_promise(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch,
                     _write_home(tmp_path / "home", {GROUP_A: True, GROUP_B: True}))
    seed = _provider(home, stack)
    with _turn():
        promise_a = seed.record_commitment(
            OWNER_A, PROMISE_A, source_quote="橘子", turn_id="turn-a")
    with _turn(chat_id="thread-4", chat_type="thread", thread_id="thread-4",
               parent_chat_id="guild-a", thread_visibility="unknown",
               session_key="session-t4", message_id="m-t4"):
        promise_t4 = seed.record_commitment(
            OWNER_A, PROMISE_T4, source_quote="旧名单", turn_id="turn-t4")
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b", message_id="m-b"):
        promise_b = seed.record_commitment(
            OWNER_B, PROMISE_B, source_quote="工作清单", turn_id="turn-b")
    seed.shutdown()
    assert (_commitment_owner(home, promise_t4), _commitment_owner(home, promise_b)) == (
        THREAD_4, OWNER_B)

    provider = _provider(home, stack)
    with _turn():
        snapshot = group_ingest._pending_snapshot(provider)
        assert [item["id"] for item in snapshot] == [promise_a], \
            "a sibling thread's and another group's promise are not this scene's"
        flash = _Flash({
            "events": [], "impression": None, "new_commitments": [],
            "commitment_updates": [
                {"id": promise_t4, "action": "fulfilled", "basis": "user_confirmed",
                 "quote": CONFIRM_USER},
                {"id": promise_b, "action": "cancelled", "basis": "user_confirmed",
                 "quote": "不用了"},
                # A real id of this scene, but this turn never said this.
                {"id": promise_a, "action": "fulfilled", "basis": "user_confirmed",
                 "quote": "橘子已经给你了"},
            ],
        })
        provider._llm = flash.llm
        result = group_ingest.apply_group_delivery(
            provider, user_message="今天不下雨了，橘子先放我这儿", assistant_response="好",
            turn_id="turn-foreign", delivered_at=1794946929.0, delivery_ref="out-a")
        assert result["updates"] == []
    provider.shutdown()

    assert [_commitment_state(home, cid) for cid in (promise_a, promise_t4, promise_b)] == [
        ("pending", 1), ("pending", 1), ("pending", 1)]
    assert _conn_row(home, "SELECT count(*) FROM events")[0] == 0


def test_closed_home_updates_nothing_and_keeps_the_rows(tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    promise_id, _ = _seed_unknown_promise(home, stack)
    inherited = dict(THREAD_3_TURN, thread_visibility="inherited", message_id="m-i1")

    _write_home(home, {GROUP_A: False})
    closed = _provider(home, stack, session_id="session-t3")
    flash = _Flash({
        "events": [], "impression": None, "new_commitments": [],
        "commitment_updates": [{"id": promise_id, "action": "fulfilled",
                                "basis": "user_confirmed", "quote": CONFIRM_USER}],
    })
    closed._llm = flash.llm
    with _turn(**inherited):
        assert closed._group_scope_grant() is None
        result = group_ingest.apply_group_delivery(
            closed, user_message=CONFIRM_USER, assistant_response=CONFIRM_REPLY,
            turn_id="turn-closed", delivered_at=1794946929.0, delivery_ref="out-closed")
        assert result["ok"] is False and result["reason"] == "group_switched_off"
    closed.shutdown()

    assert flash.requests == [], "a switched-off group never reaches the model"
    assert _commitment_state(home, promise_id) == ("pending", 1)
    assert _conn_row(home, "SELECT count(*) FROM events")[0] == 0


def test_a_stale_snapshot_never_overwrites_and_a_resolved_row_writes_once(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    promise_id, _ = _seed_unknown_promise(home, stack)
    inherited = dict(THREAD_3_TURN, thread_visibility="inherited", message_id="m-i1")
    updates = [{"id": promise_id, "action": "fulfilled", "basis": "user_confirmed",
                "quote": CONFIRM_USER}]

    provider = _provider(home, stack, session_id="session-t3")
    with _turn(**inherited):
        stale = group_ingest._pending_snapshot(provider)
        # A concurrent writer bumps the still-pending row while the model is running.
        provider._conn.execute(
            "UPDATE commitments SET version=version+1, updated_at=? WHERE id=?",
            ("2026-05-02T09:00:00+08:00", promise_id))
        provider._conn.commit()
        grant = provider._group_scope_grant()
        stale_result = _result()
        group_ingest._apply_updates(
            provider, grant, updates, pending=stale, user_text=CONFIRM_USER,
            assistant_text=CONFIRM_REPLY, transcript=(), turn_id="turn-stale",
            delivery_ref="out-stale", result=stale_result)
        assert stale_result["updates"] == []
        assert _commitment_state(home, promise_id) == ("pending", 2)
        assert _conn_row(home, "SELECT count(*) FROM events")[0] == 0

        fresh = group_ingest._pending_snapshot(provider)
        assert [(item["id"], item["version"]) for item in fresh] == [(promise_id, 2)]
        result = _result()
        group_ingest._apply_updates(
            provider, grant, updates, pending=fresh, user_text=CONFIRM_USER,
            assistant_text=CONFIRM_REPLY, transcript=(), turn_id="turn-fresh",
            delivery_ref="out-ok", result=result)
        assert result["updates"] == [{"id": promise_id, "action": "fulfilled"}]
        assert _commitment_state(home, promise_id) == ("fulfilled", 3)
        assert _conn_row(home, "SELECT count(*) FROM events")[0] == 1

        # The same snapshot again: the row is no longer pending, so no second completion event.
        again = _result()
        group_ingest._apply_updates(
            provider, grant, updates, pending=fresh, user_text=CONFIRM_USER,
            assistant_text=CONFIRM_REPLY, transcript=(), turn_id="turn-again",
            delivery_ref="out-again", result=again)
        assert again["updates"] == []
        assert _conn_row(home, "SELECT count(*) FROM events")[0] == 1
    provider.shutdown()


def test_a_plain_new_event_cannot_borrow_the_resolution_owner(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    promise_id, _ = _seed_unknown_promise(home, stack)
    inherited = dict(THREAD_3_TURN, thread_visibility="inherited", message_id="m-i1")
    seed = _provider(home, stack, session_id="session-t3")
    with _turn(**inherited):
        # Still pending: the completion seam never accepts it.
        assert seed.event_store.insert_event_row(
            "pending-completion", _event_values(f"[x] {COMPLETION}：{PROMISE_T}"),
            owner_key=THREAD_3, fulfillment_commitment_id=promise_id,
            source_context=json.dumps({"commitment_id": str(promise_id)}),
            read_scope=json.dumps({"kind": "group", "owner": THREAD_3}),
        ) is None
        assert seed.commitments.resolve_commitment(promise_id, "fulfilled", evidence="已收到") is True
        link = json.dumps({"commitment_id": str(promise_id), "ref": "m-i1"})
        own_scope = json.dumps({"kind": "group", "owner": THREAD_3})
        # A plain new event with the exact owner and no commitment is still refused.
        assert seed.event_store.insert_event_row(
            "plain-exact", _event_values("[x] 随便记一条新的事"), owner_key=THREAD_3) is None
        # With the real id, the text must still be that promise's completion record.
        assert seed.event_store.insert_event_row(
            "wrong-text", _event_values("[x] 随便记一条新的事"), owner_key=THREAD_3,
            fulfillment_commitment_id=promise_id, source_context=link,
            read_scope=own_scope) is None
        # ... and neither the link nor the read scope may name another scene.
        assert seed.event_store.insert_event_row(
            "wrong-link", _event_values(f"[x] {COMPLETION}：{PROMISE_T}"), owner_key=THREAD_3,
            fulfillment_commitment_id=promise_id,
            source_context=json.dumps({"commitment_id": "999"}), read_scope=own_scope) is None
        assert seed.event_store.insert_event_row(
            "wrong-scope", _event_values(f"[x] {COMPLETION}：{PROMISE_T}"), owner_key=THREAD_3,
            fulfillment_commitment_id=promise_id, source_context=link,
            read_scope=json.dumps({"kind": "group", "owner": OWNER_A})) is None
        # The exact completion record of this fulfilled promise is the one row allowed there.
        assert isinstance(seed.event_store.insert_event_row(
            "real-completion", _event_values(f"[x] {COMPLETION}：{PROMISE_T}"),
            owner_key=THREAD_3, fulfillment_commitment_id=promise_id, source_context=link,
            read_scope=own_scope), int)
        assert _conn_row(home, "SELECT count(*) FROM events")[0] == 1
    seed.shutdown()


def _conn_row(home: Path, sql: str, params=()):
    with sqlite3.connect(home / "data" / "sanae.db") as conn:
        return conn.execute(sql, params).fetchone()


def _conn_rows(home: Path, sql: str, params=()):
    with sqlite3.connect(home / "data" / "sanae.db") as conn:
        return conn.execute(sql, params).fetchall()


def _commitment_state(home: Path, commitment_id: int) -> tuple:
    return _conn_row(home, "SELECT status, version FROM commitments WHERE id=?",
                     (commitment_id,))


def _commitment_owner(home: Path, commitment_id: int) -> str:
    return _conn_row(home, "SELECT owner_key FROM commitments WHERE id=?",
                     (commitment_id,))[0]
