"""Task 04: one real transport receipt is the only thing that writes a group turn.

Every case here goes through the host's real Slack/QQ intake types, its real final-send receipt
path and the plugin's actually registered receipt handler, which opens the real group worker over
a temporary HOME's SQLite database. Only the platform ``send`` and the auxiliary model are
doubled; the grant, the writes, the dedup and the commitment CAS are the production ones.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("VIVEMIND_HERMES_SOURCE"):
    HERMES_ROOT = Path(os.environ["VIVEMIND_HERMES_SOURCE"]).expanduser().resolve()
else:
    _hermes_spec = importlib.util.find_spec("hermes_constants")
    if _hermes_spec is not None and _hermes_spec.origin:
        HERMES_ROOT = Path(_hermes_spec.origin).resolve().parent
    else:
        HERMES_ROOT = next(
            (path for path in (
                REPO_ROOT.parent / "hermes-minimal-host",
                REPO_ROOT.parent / "hermes-minimal-host-20261112",
            ) if path.is_dir()),
            Path(tempfile.gettempdir()) / "vivemind-hermes-6093627",
        )
if not HERMES_ROOT.is_dir():
    pytest.skip("pinned Hermes source vivemind-hermes-6093627 is unavailable", allow_module_level=True)
sys.path.insert(0, str(HERMES_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins"))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_context import (
    clear_session_vars, reset_session_vars, set_session_vars,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from sanae_mem.runtime import group_memory
from sanae_mem.infra import host_bridge
from sanae_mem.runtime import hooks as runtime_hooks
from sanae_mem.cognitive.daily_context import daily_profile_key
from sanae_mem.cognitive.inner_wiring import _wait_for_pending
from sanae_mem.plugin_entry import SanaeMemProvider
from sanae_mem.runtime.hooks import register_private_heart_hooks
from sanae_mem.governance.source_context import current_chat_scope, parse_source

assert Path(sys.modules[BasePlatformAdapter.__module__].__file__).resolve().is_relative_to(HERMES_ROOT)

GROUP_QQ = "default/qqbot/102000001/G_QQ"
GROUP_QQ_THREAD = GROUP_QQ + "/threads/T_1"
GROUP_OTHER = "default/qqbot/102000002/G_OTHER"
OWNER_QQ = group_memory.GROUP_OWNER_PREFIX + GROUP_QQ
OWNER_THREAD = group_memory.GROUP_OWNER_PREFIX + GROUP_QQ_THREAD

USER_TEXT = "周六下午我去河边捡石头，顺便把上次借你的那本书带过去。你帮我想想要不要叫上阿哲。"
REPLY_TEXT = "想叫就叫上阿哲吧。我答应你，周六上午先帮你列一份带的东西清单。"
OLD_IMPRESSION = "他在这群里说话很稳，从不抢话"
OLD_PROMISE = "答应给这个群带一袋橘子"
EVENT_SUMMARY = "他周六下午要去河边捡石头，还要把借的书带过去"
NEW_IMPRESSION = "他在这个群里会主动张罗出门的事"
NEW_PROMISE = "周六上午帮他列一份带的东西清单"
THREAD_REPLY = "名单我周六上午再补给你。"
THREAD_USER = "清单你直接发我。"


class _Adapter(BasePlatformAdapter):
    def __init__(self, result: SendResult, platform: Platform = Platform.QQBOT):
        super().__init__(PlatformConfig(enabled=True), platform)
        self.result = result
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return self.result


class _Flash:
    """The auxiliary model stand-in: records the exact request for a later real-Flash replay."""

    def __init__(self, payload):
        self.payload = payload
        self.requests = []
        self.llm = SimpleNamespace(complete=self.complete)

    def complete(self, messages, **kwargs):
        self.requests.append({"messages": messages, **kwargs})
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


def _write_home(home: Path, overrides: dict, data_dir: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    lines = [
        "plugins:", "  sanae_mem:", f"    data_dir: {data_dir}",
        "    semantic_backend: sqlite", "    group_context:", "      defaults:",
        "        long_term_memory: false", "      overrides:",
    ]
    for key, enabled in overrides.items():
        lines += [f'        "{key}":', f"          long_term_memory: {str(enabled).lower()}"]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home


def _use_home(monkeypatch, home: Path) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    return home


def _provider(home: Path, data_dir: Path, *, session_id: str = "qq-group-session"):
    provider = SanaeMemProvider({
        "data_dir": str(data_dir), "semantic_backend": "sqlite", "shadow": False,
    })
    provider.initialize(session_id, hermes_home=str(home), platform="qqbot")
    return provider


def _context(home: Path):
    manager = PluginManager(scope_key=str(home))
    return PluginContext(PluginManifest(name="sanae_mem", source="user"), manager), manager


@contextmanager
def _turn(*, chat_id="G_QQ", platform="qqbot", scope_id="102000001", chat_type="group",
          thread_id="", parent_chat_id="", thread_visibility="", synthetic=False,
          sender="U_MEMBER", session_key="qqbot-1", session_id="qq-group-session",
          message_id="in-qq-1"):
    with host_bridge.turn_envelope_scope(
        platform=platform, chat_type=chat_type, chat_id=chat_id, sender_id=sender,
        sender_id_alt="", session_id=session_id, session_key=session_key,
        message_id=message_id, scope_id=scope_id, thread_id=thread_id,
        parent_chat_id=parent_chat_id, thread_visibility=thread_visibility,
        synthetic_reply_thread=synthetic, occurred_at="2026-05-01T10:00:00+08:00",
        is_bot=False,
    ):
        yield current_chat_scope()


def _qq_source(*, chat_id="G_QQ", chat_type="group", message_id="in-qq-1", scope_id="102000001",
               thread_id="", parent_chat_id="", thread_visibility="unknown"):
    return SessionSource(
        platform=Platform.QQBOT, chat_id=chat_id, chat_type=chat_type, user_id="U_MEMBER",
        message_id=message_id, scope_id=scope_id, occurred_at="2026-05-01T10:00:00+08:00",
        is_bot=False, is_bot_known=True, inbound_coverage="mention_only",
        thread_id=thread_id or None, parent_chat_id=parent_chat_id or None,
        thread_visibility=thread_visibility,
    )


def _send(adapter, source, *, session_id, turn_id, session_key, user_text, answer):
    """One real final send: the host builds and emits the receipt from this message event."""
    event = MessageEvent(text=user_text, message_type=MessageType.TEXT,
                         message_id=str(source.message_id or ""), source=source)
    event._resolved_session_id = session_id
    event._turn_id = turn_id
    event._model = "provider/test-model"
    delivered = []
    asyncio.run(adapter._send_final_text(
        event, session_key, answer, {}, False, 0, delivered.append))
    return delivered


@contextmanager
def _registered(home: Path, provider, ctx, monkeypatch):
    """Register the plugin's real handler through the host registry, then remove it."""
    from hermes_host_contracts import register_delivery_receipt_handler

    captured = []

    def capture_registration(handler):
        captured.append(handler)
        return True

    with monkeypatch.context() as registration:
        registration.setenv("HERMES_HOME", str(home))
        registration.setattr(runtime_hooks, "register_delivery_receipt_handler", capture_registration)
        release = register_private_heart_hooks(ctx, provider)
    assert len(captured) == 1, "the plugin registered its delivery handler"
    register_delivery_receipt_handler(captured[0])
    try:
        yield captured[0]
    finally:
        release()


def _rows(data_dir: Path, sql: str, params=()):
    database = data_dir / "sanae.db"
    if not database.is_file():
        return []
    with sqlite3.connect(database) as conn:
        return conn.execute(sql, params).fetchall()


def _wait_rows(data_dir: Path, sql: str, params=(), timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = _rows(data_dir, sql, params)
        if rows:
            return rows
        time.sleep(0.02)
    return []


def _counts(data_dir: Path) -> tuple:
    def count(table):
        rows = _rows(data_dir, f"SELECT count(*) FROM {table}")
        return rows[0][0] if rows else 0

    return count("events"), count("commitments"), count("impression_revisions")


def _seed_group(home: Path, data_dir: Path):
    """A previous, ordinary group turn: its impression and one still-pending promise."""
    provider = _provider(home, data_dir)
    with _turn():
        provider.inner_wiring.tool_impression(
            operation="write", text=OLD_IMPRESSION, session_id="", turn_id="old", synthetic=False,
        )
        commitment_id = provider.record_commitment(
            OWNER_QQ, OLD_PROMISE, source_quote="橘子", turn_id="old")
        assert isinstance(commitment_id, int)
    provider.shutdown()
    return commitment_id


def _bring_up(home: Path, data_dir: Path, flash: _Flash):
    provider = _provider(home, data_dir)
    provider._llm = flash.llm
    ctx, manager = _context(home)
    return provider, ctx, manager


def _full_payload() -> dict:
    return {
        "events": [{"text": EVENT_SUMMARY, "quote": "周六下午我去河边捡石头"}],
        "impression": {"speaker": "user", "text": NEW_IMPRESSION, "quote": "周六下午我去河边捡石头"},
        "new_commitments": [{
            "text": NEW_PROMISE,
            "quote": "我答应你，周六上午先帮你列一份带的东西清单",
            "masked_text": "帮他列一份出门清单",
        }],
        "commitment_updates": [],
    }


def _deliver_qq(home: Path, provider, ctx, monkeypatch, flash: _Flash, *,
                tmp_path: Path | None = None):
    """Bring up the registered handler, send one QQ group turn, and wait for its rows."""
    data_dir = Path(provider._config["data_dir"])
    with _registered(home, provider, ctx, monkeypatch):
        reset_session_vars()
        assert current_chat_scope().visibility == "unknown", "no inbound turn is left bound"
        adapter = _Adapter(SendResult(success=True, message_id="qq-out-1"))
        delivered = _send(
            adapter, _qq_source(), session_id="qq-group-session", turn_id="qq-group-turn",
            session_key="qqbot:group:G_QQ", user_text=USER_TEXT, answer=REPLY_TEXT)
        assert delivered[0].success is True
        _wait_rows(data_dir, "SELECT id FROM commitments WHERE source_quote=?",
                   ("我答应你，周六上午先帮你列一份带的东西清单",))
        # The same native receipt delivered twice still lands once.
        _send(adapter, _qq_source(), session_id="qq-group-session", turn_id="qq-group-turn",
              session_key="qqbot:group:G_QQ", user_text=USER_TEXT, answer=REPLY_TEXT)
    if tmp_path is not None:
        (tmp_path / "group-flash-request.json").write_text(
            json.dumps(flash.requests, ensure_ascii=False, indent=2), encoding="utf-8")
    return data_dir


def test_qq_group_receipt_writes_its_own_member_events_impression_and_promise(
        tmp_path, monkeypatch):
    home = _write_home(tmp_path / "hermes", {GROUP_QQ: True},
                       tmp_path / "hermes" / "sanae-mem")
    _use_home(monkeypatch, home)
    (home / "memories").mkdir(parents=True, exist_ok=True)
    (home / "memories" / "USER.md").write_text("PRIVATE-USER-DRAWER", encoding="utf-8")
    (home / "memories" / "MEMORY.md").write_text("PRIVATE-MEMORY-NOTEBOOK", encoding="utf-8")
    old_commitment = _seed_group(home, home / "sanae-mem")

    flash = _Flash(_full_payload())
    provider, ctx, manager = _bring_up(home, home / "sanae-mem", flash)
    try:
        data_dir = _deliver_qq(home, provider, ctx, monkeypatch, flash, tmp_path=tmp_path)
    finally:
        provider.shutdown()

    # The request is the complete material a real Flash replay needs: this turn's real member
    # message, the delivered reply, the scene facts, this member's old impression and the pending
    # promise. No private drawer and no other scene is loaded.
    assert len(flash.requests) == 1, "one delivered turn asked the model exactly once"
    request = flash.requests[0]
    assert request["purpose"] == "group_ingest"
    assert request["task"] == "sanae_daily_flash", "the existing Flash auxiliary slot is reused"
    system_prompt = request["messages"][0]["content"]
    user_prompt = request["messages"][1]["content"]
    assert "events" in system_prompt and "commitment_updates" in system_prompt
    assert USER_TEXT in user_prompt and REPLY_TEXT in user_prompt
    assert OLD_IMPRESSION in user_prompt and OLD_PROMISE in user_prompt
    assert "PRIVATE-USER-DRAWER" not in user_prompt
    assert "PRIVATE-MEMORY-NOTEBOOK" not in user_prompt
    captured = json.loads((tmp_path / "group-flash-request.json").read_text(encoding="utf-8"))
    assert captured[0]["messages"] == request["messages"]

    event_rows = _rows(
        data_dir, "SELECT raw, summary, summary_source, owner_key, source_context "
                  "FROM events WHERE owner_key=?", (OWNER_QQ,))
    assert len(event_rows) == 1
    raw, summary, summary_source, owner, source_json_text = event_rows[0]
    assert "周六下午我去河边捡石头" in raw, "raw keeps the member's verbatim quote"
    assert summary == EVENT_SUMMARY and summary_source == "distilled"
    source = parse_source(source_json_text)
    assert (source["visibility"], source["author_id"], source["chat_id"], source["ref"],
            source["time"]) == ("group", "U_MEMBER", "G_QQ", "in-qq-1",
                                "2026-05-01T10:00:00+08:00")
    assert source["processing"] == "group_receipt_ingest"
    assert source["processing_version"] and source["delivery_ref"] == "qq-out-1"

    impression_rows = _rows(
        data_dir, "SELECT owner_key, text, evidence, source_context FROM impression_revisions "
                  "ORDER BY revision")
    assert [row[1] for row in impression_rows] == [OLD_IMPRESSION, NEW_IMPRESSION]
    member_owner = impression_rows[-1][0]
    assert member_owner.startswith(OWNER_QQ + "#")
    assert impression_rows[-1][2] == "周六下午我去河边捡石头"
    impression_source = parse_source(impression_rows[-1][3])
    assert (impression_source["ref"], impression_source["author_id"],
            impression_source["processing"]) == ("in-qq-1", "U_MEMBER", "group_receipt_ingest")

    promise = _rows(data_dir, "SELECT owner_key, text, source_quote, status, source_context "
                              "FROM commitments WHERE text=?", (NEW_PROMISE,))
    assert len(promise) == 1, "the promise quoted from the delivered reply is filed once"
    assert promise[0][0] == OWNER_QQ and promise[0][3] == "pending"
    assert promise[0][2] == "我答应你，周六上午先帮你列一份带的东西清单"
    promise_source = parse_source(promise[0][4])
    assert (promise_source["ref"], promise_source["delivery_ref"], promise_source["processing"]) \
        == ("in-qq-1", "qq-out-1", "group_receipt_ingest")
    # The older, unrelated promise is untouched.
    assert _rows(data_dir, "SELECT status, version FROM commitments WHERE id=?",
                 (old_commitment,)) == [("pending", 1)]
    assert _counts(data_dir) == (1, 2, 2), "the duplicate receipt wrote nothing"


def test_narrow_thread_keeps_its_own_scene_and_cannot_close_the_parent_promise(
        tmp_path, monkeypatch):
    home = _write_home(tmp_path / "hermes", {GROUP_QQ: True},
                       tmp_path / "hermes" / "sanae-mem")
    _use_home(monkeypatch, home)
    parent_commitment = _seed_group(home, home / "sanae-mem")
    parent_events = _counts(home / "sanae-mem")[0]

    thread_seed = _provider(home, home / "sanae-mem")
    with _turn(chat_id="T_1", chat_type="thread", thread_id="T_1", parent_chat_id="G_QQ",
               thread_visibility="narrower", session_key="qq-thread-old",
               session_id="qq-thread-old", message_id="in-t-old"):
        thread_commitment = thread_seed.record_commitment(
            OWNER_QQ, "线程里答应先替他保密", source_quote="保密", turn_id="thread-old")
    thread_seed.shutdown()
    assert isinstance(thread_commitment, int)

    payload = {
        "events": [{"text": "他要我把清单直接发给他", "quote": THREAD_USER}],
        "impression": None,
        "new_commitments": [{"text": "周六上午把名单补给他", "quote": THREAD_REPLY}],
        "commitment_updates": [
            # The parent's shared promise is not this thread's to close, and a mere assistant
            # claim of an external action has no real tool success behind it.
            {"id": parent_commitment, "action": "fulfilled", "basis": "delivered_reply",
             "quote": THREAD_REPLY},
            {"id": thread_commitment, "action": "fulfilled", "basis": "tool_result",
             "quote": THREAD_REPLY},
        ],
    }
    flash = _Flash(payload)
    provider, ctx, manager = _bring_up(home, home / "sanae-mem", flash)
    data_dir = home / "sanae-mem"
    try:
        with _registered(home, provider, ctx, monkeypatch):
            reset_session_vars()
            adapter = _Adapter(SendResult(success=True, message_id="thread-out-1"))
            _send(
                adapter,
                _qq_source(chat_id="T_1", chat_type="thread", message_id="in-t1",
                           thread_id="T_1", parent_chat_id="G_QQ",
                           thread_visibility="narrower"),
                session_id="qq-thread-session", turn_id="qq-thread-turn",
                session_key="qqbot:thread:T_1", user_text=THREAD_USER, answer=THREAD_REPLY)
            _wait_rows(data_dir, "SELECT id FROM events WHERE owner_key=?", (OWNER_THREAD,))
        assert len(flash.requests) == 1
        user_prompt = flash.requests[0]["messages"][1]["content"]
        assert THREAD_USER in user_prompt and THREAD_REPLY in user_prompt
        assert OLD_PROMISE not in user_prompt, "only this thread's write owner's promises are shown"
    finally:
        provider.shutdown()

    event_row = _rows(data_dir, "SELECT raw, owner_key, source_context FROM events WHERE owner_key=?",
                      (OWNER_THREAD,))
    assert len(event_row) == 1
    raw, owner, source_json_text = event_row[0]
    assert THREAD_USER in raw
    source = parse_source(source_json_text)
    assert (source["chat_id"], source["thread_id"], source["parent_chat_id"],
            source["thread_visibility"], source["ref"], source["time"], source["author_id"]) == (
        "T_1", "T_1", "G_QQ", "narrower", "in-t1", "2026-05-01T10:00:00+08:00", "U_MEMBER")
    # The promise written from the delivered reply belongs to the thread, not the parent.
    assert _rows(data_dir, "SELECT owner_key, source_context FROM commitments WHERE source_quote=?",
                 (THREAD_REPLY,))[0][0] == OWNER_THREAD
    assert len(_rows(data_dir, "SELECT id FROM events")) == parent_events + 1
    # Neither the parent's shared promise nor the thread's was closed by a bare claim.
    assert _rows(data_dir, "SELECT status, version FROM commitments WHERE id=?",
                 (parent_commitment,)) == [("pending", 1)]
    assert _rows(data_dir, "SELECT status, version FROM commitments WHERE id=?",
                 (thread_commitment,)) == [("pending", 1)]


def test_the_delivered_reply_itself_can_fulfil_and_keeps_the_real_incoming_ref(
        tmp_path, monkeypatch):
    home = _write_home(tmp_path / "hermes", {GROUP_QQ: True},
                       tmp_path / "hermes" / "sanae-mem")
    _use_home(monkeypatch, home)
    data_dir = home / "sanae-mem"
    seed = _provider(home, data_dir)
    with _turn():
        promise_id = seed.record_commitment(
            OWNER_QQ, "帮他想好要不要叫上阿哲", source_quote="你帮我想想要不要叫上阿哲",
            turn_id="old")
    assert isinstance(promise_id, int)
    seed.shutdown()

    payload = {
        "events": [],
        "impression": None,
        "new_commitments": [],
        "commitment_updates": [{
            "id": promise_id, "action": "fulfilled", "basis": "delivered_reply",
            "quote": "想叫就叫上阿哲吧",
        }],
    }
    flash = _Flash(payload)
    provider, ctx, manager = _bring_up(home, data_dir, flash)
    try:
        with _registered(home, provider, ctx, monkeypatch):
            reset_session_vars()
            adapter = _Adapter(SendResult(success=True, message_id="qq-out-1"))
            _send(adapter, _qq_source(), session_id="qq-group-session",
                  turn_id="qq-group-turn", session_key="qqbot:group:G_QQ",
                  user_text=USER_TEXT, answer=REPLY_TEXT)
            _wait_rows(data_dir, "SELECT id FROM events WHERE raw LIKE ?",
                       ("%我已兑现此前答应的事%",))
            _send(adapter, _qq_source(), session_id="qq-group-session",
                  turn_id="qq-group-turn", session_key="qqbot:group:G_QQ",
                  user_text=USER_TEXT, answer=REPLY_TEXT)
        assert len(flash.requests) == 1
    finally:
        provider.shutdown()

    assert _rows(data_dir, "SELECT status, version FROM commitments WHERE id=?",
                 (promise_id,)) == [("fulfilled", 2)]
    rows = _rows(data_dir, "SELECT owner_key, raw, source_context FROM events")
    assert len(rows) == 1, "the duplicate receipt did not write a second completion"
    owner, raw, source_json_text = rows[0]
    assert owner == OWNER_QQ and "我已兑现此前答应的事" in raw
    source = parse_source(source_json_text)
    assert source["ref"] == "in-qq-1", "the real incoming ref is never replaced by a marker"
    assert source["commitment_id"] == str(promise_id), "the promise is linked by its own field"
    assert source["delivery_ref"] == "qq-out-1"


def test_closed_group_and_foreign_home_write_nothing(tmp_path, monkeypatch):
    closed_home = _write_home(tmp_path / "closed", {GROUP_QQ: False},
                              tmp_path / "closed" / "sanae-mem")
    _use_home(monkeypatch, closed_home)
    flash = _Flash(_full_payload())
    provider, ctx, manager = _bring_up(closed_home, closed_home / "sanae-mem", flash)
    try:
        with _registered(closed_home, provider, ctx, monkeypatch):
            reset_session_vars()
            adapter = _Adapter(SendResult(success=True, message_id="qq-out-1"))
            _send(adapter, _qq_source(), session_id="qq-group-session",
                  turn_id="qq-closed-turn", session_key="qqbot:group:G_QQ",
                  user_text=USER_TEXT, answer=REPLY_TEXT)
            _wait_for_pending((Path(provider._config["data_dir"]).resolve(),
                               daily_profile_key(closed_home)))
        assert flash.requests == [], "a switched-off group never reaches the model"
        assert _counts(closed_home / "sanae-mem") == (0, 0, 0)
    finally:
        provider.shutdown()

    live_home = _write_home(tmp_path / "live", {GROUP_QQ: True},
                            tmp_path / "live" / "sanae-mem")
    _use_home(monkeypatch, live_home)
    flash = _Flash(_full_payload())
    provider, ctx, manager = _bring_up(live_home, live_home / "sanae-mem", flash)
    try:
        with _registered(live_home, provider, ctx, monkeypatch):
            foreign = tmp_path / "foreign"
            foreign.mkdir()
            monkeypatch.setenv("HERMES_HOME", str(foreign))
            monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(foreign))
            reset_session_vars()
            adapter = _Adapter(SendResult(success=True, message_id="qq-out-1"))
            _send(adapter, _qq_source(), session_id="qq-group-session",
                  turn_id="qq-foreign-turn", session_key="qqbot:group:G_QQ",
                  user_text=USER_TEXT, answer=REPLY_TEXT)
        assert flash.requests == [], "the handler registered in another HOME owns nothing here"
        assert _counts(live_home / "sanae-mem") == (0, 0, 0)
    finally:
        provider.shutdown()


def test_failed_send_and_an_undelivered_draft_write_no_group_promise(tmp_path, monkeypatch):
    home = _write_home(tmp_path / "hermes", {GROUP_QQ: True},
                       tmp_path / "hermes" / "sanae-mem")
    _use_home(monkeypatch, home)
    data_dir = home / "sanae-mem"
    flash = _Flash(_full_payload())
    provider, ctx, manager = _bring_up(home, data_dir, flash)
    try:
        with _registered(home, provider, ctx, monkeypatch):
            reset_session_vars()
            failed = _Adapter(SendResult(success=False, error="chat_not_found"))
            delivered = _send(
                failed, _qq_source(), session_id="qq-group-session", turn_id="qq-failed-turn",
                session_key="qqbot:group:G_QQ", user_text=USER_TEXT, answer=REPLY_TEXT)
            assert delivered[0].success is False

            # The model produced a draft, but nothing was said: the post_llm hook only observes.
            tokens = set_session_vars(
                platform="qqbot", chat_type="group", chat_id="G_QQ", user_id="U_MEMBER",
                session_id="qq-group-session", session_key="qqbot:group:G_QQ",
                message_id="in-qq-draft",
            )
            try:
                manager.invoke_hook(
                    "post_llm_call", session_id="qq-group-session", turn_id="qq-draft-turn",
                    platform="qqbot", user_message=USER_TEXT,
                    assistant_response="我答应你，周六上午先帮你列一份带的东西清单。",
                    model="provider/test-model",
                )
            finally:
                clear_session_vars(tokens)
        assert flash.requests == []
        assert _counts(data_dir) == (0, 0, 0)
    finally:
        provider.shutdown()


def test_a_new_session_reads_the_receipt_written_promise_only_inside_that_group(
        tmp_path, monkeypatch):
    home = _write_home(tmp_path / "hermes", {GROUP_QQ: True, GROUP_OTHER: True},
                       tmp_path / "hermes" / "sanae-mem")
    _use_home(monkeypatch, home)
    data_dir = home / "sanae-mem"
    flash = _Flash(_full_payload())
    provider, ctx, manager = _bring_up(home, data_dir, flash)
    try:
        _deliver_qq(home, provider, ctx, monkeypatch, flash)
    finally:
        provider.shutdown()

    # A brand-new session of the same group still gets this turn's own materials.
    reader = _provider(home, data_dir, session_id="new-session")
    with _turn(session_id="new-session", session_key="qqbot-new"):
        context = reader.prefetch(EVENT_SUMMARY, session_id="new-session")
        assert EVENT_SUMMARY in context and NEW_PROMISE in context
    reader.shutdown()

    other = _provider(home, data_dir, session_id="other-session")
    with _turn(chat_id="G_OTHER", scope_id="102000002", sender="U_OTHER",
               session_key="qq-other", session_id="other-session", message_id="in-other"):
        assert NEW_PROMISE not in other.prefetch(NEW_PROMISE, session_id="other-session")
        assert EVENT_SUMMARY not in other.prefetch(EVENT_SUMMARY, session_id="other-session")
    other.shutdown()

    private = _provider(home, data_dir, session_id="dm-session")
    with _turn(chat_type="dm", chat_id="U_MEMBER", sender="U_MEMBER",
               session_key="qq-dm", session_id="dm-session", message_id="in-dm"):
        private.on_turn_start(1, "")
        assert NEW_PROMISE not in private.prefetch(NEW_PROMISE, session_id="dm-session")
        assert EVENT_SUMMARY not in private.prefetch(EVENT_SUMMARY, session_id="dm-session")
        assert private.list_commitments() == []
    private.shutdown()

    sibling = _provider(home, data_dir, session_id="sibling-session")
    with _turn(chat_id="T_9", chat_type="thread", thread_id="T_9", parent_chat_id="G_QQ",
               thread_visibility="unknown", sender="U_MEMBER", session_key="qq-t9",
               session_id="sibling-session", message_id="in-t9"):
        assert NEW_PROMISE not in sibling.prefetch(NEW_PROMISE, session_id="sibling-session")
        assert EVENT_SUMMARY not in sibling.prefetch(EVENT_SUMMARY, session_id="sibling-session")
    sibling.shutdown()
