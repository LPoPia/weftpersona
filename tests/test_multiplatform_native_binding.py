"""Cross-host replay: real adapter capture → real admission/claim → real ViveMind turn.

Each case drives the production chain end to end: the platform adapter's own entry point, the
durable FIFO seam (``prepare_native`` → the real ``_admit_native_write`` → ``claim_session_input``
→ ``restore_native``), the host's own turn binder (``GatewayRunner._set_session_env``), and then
ViveMind's real tool entry while that binding is live. The admission is settled through the real
store and read back at its own identity, so a regression in the adapter, the codec, the queue or
the ownership resolver fails here. Only the platform SDKs are stubbed.

    PYTHONPATH=<vivemind>/plugins/sanae_mem:<vivemind>/plugins:<hermes> python -m pytest \
        tests/test_multiplatform_native_binding.py
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("gateway.run")
pytest.importorskip("hermes_host_contracts")
pytest.importorskip("tests.gateway.test_multiplatform_scene_scope")

_PLUGINS = Path(__file__).resolve().parents[1] / "plugins"
if str(_PLUGINS) not in sys.path:
    sys.path.insert(0, str(_PLUGINS))

import gateway.run as gateway_run  # noqa: E402
import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from gateway.session import SessionContext  # noqa: E402
from gateway.session_context import clear_session_vars  # noqa: E402
from gateway.session_envelope import prepare_native, restore_native  # noqa: E402
from hermes_state_runtime import (claim_session_input, find_session_admission,  # noqa: E402
                                 settle_session_input)  # noqa: E402
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest  # noqa: E402
from tests.gateway.conftest import owner  # noqa: E402,F401 — the host's canonical authority fixture
from tests.gateway.test_multiplatform_scene_scope import (  # noqa: E402
    _FakeDMChannel, _FakeThread, _discord_adapter, _discord_message as _host_discord_message,
    _qq_adapter, _slack_capture, _slack_event, slack_adapter,
)

from sanae_mem import plugin_entry as entry  # noqa: E402
from sanae_mem.infra import auxiliary  # noqa: E402
from sanae_mem.runtime import hooks as runtime_hooks  # noqa: E402
from sanae_mem.infra.clock import sleep_day_iso  # noqa: E402
from sanae_mem.governance.cognitive_context import classify_turn_surface  # noqa: E402
from sanae_mem.cognitive.daily_rules import KIND_FACT, SOURCE_STATEMENT  # noqa: E402
from sanae_mem.runtime.scheduled import sync_scheduled_jobs  # noqa: E402
from sanae_mem.infra.textutil import _owner_key  # noqa: E402

QQ_APP = "102000001"
OTHER_APP = "999000002"
ANCHOR = _owner_key("qqbot", user_id="U_OPENID")
SLACK_LEGACY = _owner_key("slack", user_id="U_MAINT", scope_id="T_TEAM")
CONFIG = {
    "memory_identity": {"platform": "qqbot", "user_id": "U_OPENID", "scope_id": QQ_APP},
    "maintainer_account_bindings": [
        {"platform": "slack", "user_id": "U_MAINT", "scope_id": "T_TEAM"},
        {"platform": "discord", "user_id": "42"},
        {"platform": "qqbot", "user_id": "U_OPENID", "scope_id": QQ_APP},
    ],
    "semantic_backend": "sqlite",
    "shadow": False,
}


@pytest.fixture(autouse=True)
def _hermetic_home(tmp_path, monkeypatch):
    """The host's live-state guard: the authority opens this test's own profile, never the real one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _runner() -> SimpleNamespace:
    """The admission seam only: authorization verdict + platform lookup, nothing else."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(multiplex_profiles=False)
    adapter = SimpleNamespace(platform="platform-seam")
    runner._adapter_for_source = lambda source: adapter
    runner._adapter_profile_for_source = lambda source: None
    runner._adapter_flag = lambda platform, flag, profile=None: False
    runner._is_user_authorized_for_source = lambda source, allow_adapter_delegation=False: True
    runner.session_store = SimpleNamespace(
        _generate_session_key=lambda source, key_source=None, *, group_policy=True: (
            f"agent:main:{source.platform.value}:{source.chat_type}:{source.chat_id}"))
    return runner


def _admit(authority, payload: dict, request_id: str, session_id: str) -> dict:
    """The real native write seam: reconcile against the committed row, then the SQLite digest."""
    return authority._admit_native_write(
        principal_id="messaging:test", session_id=session_id, request_id=request_id,
        payload=payload, authorize=None, raw=None)


async def _deliver(authority, event, *, session_id: str, source: str, turn):
    """capture → durable row → FIFO claim → restore → host binding → ViveMind → settle."""
    runner = _runner()
    payload = json.loads(json.dumps(await prepare_native(runner, event)))
    authority.db.create_session(session_id, source=source)
    admitted = await asyncio.to_thread(_admit, authority, payload, event.message_id, session_id)
    assert admitted["status"] == "queued", "the inbound message is the queued input"

    claimed = claim_session_input(authority.db, epoch=authority.epoch, session_id=session_id)
    assert claimed["admission_id"] == admitted["admission_id"]
    assert claimed["payload"]["native_text_v1"]["source"] == payload["native_text_v1"]["source"]

    restored = restore_native(claimed["payload"], runner)
    assert restored.source.to_dict() == event.source.to_dict(), "the recovered turn is the captured one"

    # The turn body runs INSIDE the host's own binding: the ContextVar is cleared on exit, so a
    # read or write after this block would not be the live turn's.
    binding = object.__new__(gateway_run.GatewayRunner)
    binding.adapters = {}
    context = SessionContext(source=restored.source, connected_platforms=[], home_channels={},
                             session_key=payload["native_text_v1"]["route"], session_id=session_id)
    tokens = binding._set_session_env(context)
    try:
        result = turn(restored)
        if inspect.isawaitable(result):
            # Await INSIDE the binding: the callback's own transport calls still see the turn.
            result = await result
    finally:
        clear_session_vars(tokens)

    settled = settle_session_input(
        authority.db, epoch=authority.epoch, admission_id=claimed["admission_id"],
        generation=claimed["generation"], outcome="completed")
    assert (settled["status"], settled["outcome"]) == ("terminal", "completed")
    same = find_session_admission(authority.db, principal_id="messaging:test",
                                  session_id=session_id, request_id=event.message_id)
    assert same["admission_id"] == claimed["admission_id"], "one row, one execution"
    assert same["payload"]["text"] == payload["text"], "the settled input is the admitted message"
    return result


def _provider(tmp_path, platform: str) -> "entry.SanaeMemProvider":
    provider = entry.SanaeMemProvider({"data_dir": str(tmp_path / "data"), **CONFIG})
    provider.initialize(f"agent:main:{platform}:dm", hermes_home=str(tmp_path),
                        platform=platform, agent_context="primary")
    return provider


def _add(provider, content: str) -> int:
    """ViveMind's own tool entry, not a direct row insert: scope, owner and guard all apply."""
    result = json.loads(provider.handle_tool_call("sanae_mem_add", {"content": content}))
    assert result["ok"] is True, result
    assert result["result"]["stored"] is True, result
    return result["result"]["id"]


# ── the three-platform chain ─────────────────────────────────────────────────────────────────

async def _qq_message(app_id: str = QQ_APP):
    adapter = _qq_adapter()
    adapter._app_id = app_id
    await adapter._handle_c2c_message(
        {"content": "hi"}, "qq-m1", "hi", {"user_openid": "U_OPENID"},
        "2024-05-01T00:00:00+00:00")
    return adapter.handle_message.await_args.args[0]


async def _slack_message(slack_adapter, user: str = "U_MAINT", channel: str = "D_CHAN",
                         text: str = "hello"):
    return await _slack_capture(slack_adapter, _slack_event(
        channel, "im", "1700000000.000111", user=user, text=text))


async def _discord_message(monkeypatch):
    if discord_platform.discord is None:
        monkeypatch.setattr(discord_platform, "discord",
                            SimpleNamespace(DMChannel=_FakeDMChannel, Thread=_FakeThread),
                            raising=False)
    else:
        monkeypatch.setattr(discord_platform.discord, "DMChannel", _FakeDMChannel, raising=False)
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
    adapter = _discord_adapter()
    assert await adapter._handle_message(_host_discord_message(_FakeDMChannel(55), 201))
    return adapter, adapter.handle_message.await_args.args[0]


class _RecordingChannel:
    """Discord channel stand-in: records the wire call a reply would make."""

    def __init__(self, channel_id: int) -> None:
        self.id = channel_id
        self.sent: list = []

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(id=5001)


@pytest.mark.asyncio
async def test_real_platform_chain_reads_other_platform_history_and_writes_one_owner(
    tmp_path, owner, slack_adapter, monkeypatch,
):
    # QQ Bot: the app ID the adapter states is the binding scope.
    qq = await _qq_message()

    def qq_turn(restored):
        provider = _provider(tmp_path, "qqbot")
        try:
            assert provider._owner_key == ANCHOR
            new_id = _add(provider, "the qq turn")
            rows = {row["id"]: row["owner_key"] for row in provider._conn.execute(
                "SELECT id, owner_key FROM events").fetchall()}
            assert rows[new_id] == ANCHOR
        finally:
            provider.shutdown()
        return new_id

    qq_row = await _deliver(owner, qq, session_id="qq", source="qqbot", turn=qq_turn)

    # A row the same person wrote on the old Slack account, before any binding existed.
    slack = await _slack_message(slack_adapter)

    def slack_turn(restored):
        provider = _provider(tmp_path, "slack")
        try:
            provider._conn.execute(
                "INSERT INTO events (ts,raw,created_at,owner_key,source_context)"
                " VALUES ('2026-01-01','old slack line','2026-01-01',?,?)",
                (SLACK_LEGACY, '{"kind":"maintainer","platform":"slack"}'))
            provider._conn.commit()
            legacy_id = provider._conn.execute(
                "SELECT id FROM events WHERE owner_key=?", (SLACK_LEGACY,)).fetchone()["id"]
            assert provider._owner_key == ANCHOR
            assert provider.event_store.get_event_row_by_id(qq_row) is not None, "the QQ turn is readable"
            assert provider.event_store.get_event_row_by_id(legacy_id) is not None, "its own old row too"
            own_id = _add(provider, "the slack turn")
        finally:
            provider.shutdown()
        return own_id

    slack_row = await _deliver(owner, slack, session_id="slack", source="slack", turn=slack_turn)

    # Slack answers on the conversation the message came from, through the real adapter.send.
    slack_adapter._app.client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1700000000.000999"})
    sent = await slack_adapter.send(slack.source.chat_id, "reply",
                                    metadata={"team_id": slack.source.scope_id})
    assert sent.success
    wire = slack_adapter._app.client.chat_postMessage.await_args.kwargs
    assert wire["channel"] == "D_CHAN", "the reply lands on the inbound DM"
    assert wire.get("thread_ts") is None

    discord, discord_msg = await _discord_message(monkeypatch)

    def discord_turn(restored):
        provider = _provider(tmp_path, "discord")
        try:
            assert provider._owner_key == ANCHOR, "snowflakes are global, the guild is not an account"
            for row in (qq_row, slack_row):
                assert provider.event_store.get_event_row_by_id(row) is not None
            own_id = _add(provider, "the discord turn")
            owners = {row[0] for row in provider._conn.execute(
                "SELECT DISTINCT owner_key FROM events WHERE id IN (?,?,?)",
                (qq_row, slack_row, own_id))}
            assert owners == {ANCHOR}, "one owner, three transports"
        finally:
            provider.shutdown()
        return own_id

    await _deliver(owner, discord_msg, session_id="discord", source="discord", turn=discord_turn)

    # Discord answers in its own DM channel, never another conversation.
    reply_channel = _RecordingChannel(55)
    discord._client.get_channel = lambda cid: (
        reply_channel if int(cid) == 55
        else pytest.fail(f"Discord reply left the inbound DM: {cid}"))
    discord._ensure_connected = AsyncMock(return_value=True)
    discord._outbound_blocked = lambda *a, **k: None
    await discord.send("55", "reply")
    assert reply_channel.sent, "the inbound DM itself received the reply"


@pytest.mark.asyncio
async def test_qq_actor_from_another_app_gets_no_private_grant(tmp_path, owner):
    """The same openid under a different app is a different account: no anchor, no history."""

    def bound_turn(restored):
        provider = _provider(tmp_path, "qqbot")
        try:
            assert provider._owner_key == ANCHOR
            return _add(provider, "the bound app's line")
        finally:
            provider.shutdown()

    anchor_row = await _deliver(owner, await _qq_message(), session_id="qq-app",
                                source="qqbot", turn=bound_turn)

    def other_turn(restored):
        provider = _provider(tmp_path, "qqbot")
        try:
            initial_owner = provider._owner_key
            assert initial_owner not in {"", ANCHOR}, "another app's ID is another account"
            assert provider.event_store.get_event_row_by_id(anchor_row) is None, "the anchor's history is closed"
            assert provider.resolve_private_delivery_route().owner_key == initial_owner
            own_row = _add(provider, "the other app's line")
            owner_of = dict(provider._conn.execute(
                "SELECT id, owner_key FROM events").fetchall())
            assert owner_of[own_row] == provider._owner_key
            assert owner_of[anchor_row] == ANCHOR, "the anchor's row keeps its own owner"
        finally:
            provider.shutdown()

    await _deliver(owner, await _qq_message(OTHER_APP), session_id="qq-other",
                   source="qqbot", turn=other_turn)


def test_proactive_target_stays_the_configured_qq_identity(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr("sanae_mem.runtime.scheduled.host_bridge.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("sanae_mem.runtime.scheduled.host_bridge.hermes_source_root", lambda: tmp_path)
    monkeypatch.setattr("sanae_mem.runtime.scheduled.host_bridge.list_plugin_cron_jobs",
                        lambda include_disabled=True: [])
    monkeypatch.setattr("sanae_mem.runtime.scheduled.host_bridge.sync_plugin_cron_job",
                        lambda **kwargs: calls.append(kwargs))
    sync_scheduled_jobs({**CONFIG, "free_think": True})
    free_think = [call for call in calls if call["prompt"] and "deliver" in call]
    assert free_think and free_think[0]["deliver"] == "qqbot:U_OPENID"


# ── one ID shape, two Slack workspaces, one old maintainer activity ───────────────────────────

SHARED_USER = "U_SAME"                 # the same ID shape in both workspaces
WORKSPACE_ONE = "T_WORKSPACE_ONE"
WORKSPACE_TWO = "T_WORKSPACE_TWO"
# The old anchor: memory_identity with no scope_id, exactly the shape the collision review named.
LEGACY_ANCHOR = _owner_key("slack", user_id=SHARED_USER)
SENTINEL = "旧维护者今天把书架整理了一遍。"


def _collision_config(*bindings: dict) -> dict:
    return {
        "memory_identity": {"platform": "slack", "user_id": SHARED_USER},
        "maintainer_account_bindings": list(bindings),
        "semantic_backend": "sqlite",
        "shadow": False,
    }


def _turn_provider(tmp_path, session_id: str, config: dict) -> "entry.SanaeMemProvider":
    """A real provider on the turn's own live scope; one shared ``sanae.db`` under tmp."""
    provider = entry.SanaeMemProvider({"data_dir": str(tmp_path / "data"), **config})
    provider.initialize(session_id, hermes_home=str(tmp_path), platform="slack",
                        agent_context="primary")
    return provider


def _turn_reads(provider, session_id: str, tag: str) -> list:
    """The real entry, twice, with the arguments a private turn passes."""
    outs = [provider.current_turn_context(
        session_id=session_id, turn_id=f"{tag}-{round_no}", user_message="今天怎么样")
        for round_no in (1, 2)]
    for out in outs:
        assert out.strip(), "the private turn renders its own context"
    return outs


def _assert_no_maintainer_activity(outs: list) -> None:
    """Neither site that could carry his day: the activity block and the daily block."""
    for out in outs:
        assert SENTINEL not in out, out
        assert "今日活动" not in out, "no maintainer daily block for someone else:\n" + out
        assert "是他自己的事" not in out, "not the configured private person:\n" + out


async def _workspace_message(slack_adapter, team: str, channel: str, ts: str):
    return await _slack_capture(slack_adapter, _slack_event(
        channel, "im", ts, user=SHARED_USER, team=team, text="今天怎么样"))


@pytest.mark.asyncio
async def test_same_shaped_slack_id_in_another_workspace_never_gets_the_maintainer_activity(
    tmp_path, owner, slack_adapter,
):
    """Two workspaces state the same user ID; only an explicitly bound one may reach the anchor.

    The old maintainer activity is seeded once, in the real daily-state format, under the
    unscoped ``memory_identity`` anchor. Until a workspace is bound, neither private entry reads
    it; after binding the first workspace, only that workspace carries it.
    """
    unbound = _collision_config()
    owners: dict = {}

    def workspace_one_unbound(restored):
        provider = _turn_provider(tmp_path, "slack-one-a", unbound)
        try:
            assert provider._owner_key not in {"", LEGACY_ANCHOR}, "an unbound ID is not the anchor"
            assert classify_turn_surface(config=unbound) == "private_external"
            seeded = provider.open_daily_entry(
                owner_key=LEGACY_ANCHOR, profile_key=provider._daily_profile_key,
                kind=KIND_FACT, slot="fixture-old-maintainer-activity", text=SENTINEL,
                opened_on=sleep_day_iso(), now=datetime.now().astimezone().isoformat(),
                source_quote="", source_turn="fixture", source_kind=SOURCE_STATEMENT,
                author_kind="maintainer")
            assert seeded, "the old maintainer activity is stored once, for today"
            _assert_no_maintainer_activity(_turn_reads(provider, "slack-one-a", "one-unbound"))
            owners["one"] = provider._owner_key
        finally:
            provider.shutdown()

    await _deliver(owner, await _workspace_message(
        slack_adapter, WORKSPACE_ONE, "D_ONE", "1700000001.000101"),
        session_id="slack-one-a", source="slack", turn=workspace_one_unbound)

    def workspace_two_unbound(restored):
        provider = _turn_provider(tmp_path, "slack-two-a", unbound)
        try:
            assert provider._owner_key not in {"", LEGACY_ANCHOR}
            assert classify_turn_surface(config=unbound) == "private_external"
            _assert_no_maintainer_activity(_turn_reads(provider, "slack-two-a", "two-unbound"))
            owners["two"] = provider._owner_key
        finally:
            provider.shutdown()

    await _deliver(owner, await _workspace_message(
        slack_adapter, WORKSPACE_TWO, "D_TWO", "1700000002.000202"),
        session_id="slack-two-a", source="slack", turn=workspace_two_unbound)
    assert owners["one"] and owners["two"] and owners["one"] != owners["two"], (
        "same-shaped IDs in different workspaces are different private owners")

    # Only the first workspace is declared; the second one shares the ID shape, not the person.
    bound = _collision_config(
        {"platform": "slack", "user_id": SHARED_USER, "scope_id": WORKSPACE_ONE})

    def workspace_one_bound(restored):
        provider = _turn_provider(tmp_path, "slack-one-b", bound)
        try:
            assert provider._owner_key == LEGACY_ANCHOR, "the bound workspace reaches the anchor"
            assert classify_turn_surface(config=bound) == "private_maintainer"
            outs = _turn_reads(provider, "slack-one-b", "one-bound")
            assert all(SENTINEL in out for out in outs), "the bound workspace keeps the old activity"
            assert all("今日活动" in out for out in outs), "it arrives as today's activity block"
        finally:
            provider.shutdown()

    await _deliver(owner, await _workspace_message(
        slack_adapter, WORKSPACE_ONE, "D_ONE", "1700000003.000301"),
        session_id="slack-one-b", source="slack", turn=workspace_one_bound)

    def workspace_two_bound(restored):
        provider = _turn_provider(tmp_path, "slack-two-b", bound)
        try:
            assert provider._owner_key == owners["two"], "the binding did not move its neighbour"
            assert classify_turn_surface(config=bound) == "private_external"
            _assert_no_maintainer_activity(_turn_reads(provider, "slack-two-b", "two-bound"))
        finally:
            provider.shutdown()

    await _deliver(owner, await _workspace_message(
        slack_adapter, WORKSPACE_TWO, "D_TWO", "1700000004.000402"),
        session_id="slack-two-b", source="slack", turn=workspace_two_bound)


# ── one delivered reply resolves the same person's old alias commitments ──────────────────────

BOOK_QUOTE = "Remember to return two library books."
REPORT_QUOTE = "Remember to send the quarterly report."
OTHER_QUOTE = "Water the neighbour's plants."
TURN_USER = ("The two library books are already returned and the quarterly report went out "
             "this morning, so cancel both reminders.")
TURN_REPLY = "好，这两件事的提醒我都不再记着了。"


@pytest.mark.asyncio
async def test_delivered_reply_resolves_the_bound_alias_commitments_too(
    tmp_path, owner, slack_adapter,
):
    """A bound person's older Slack commitment is in this turn's snapshot, not only the anchor's.

    The person's own words travel through the adapter's inbound message and come back out of the
    SQLite queue as the restored text. Only a confirmed send on the restored address hands that
    text to ``sync_turn``, so one quote that names both items resolves exactly those two rows.
    """
    ids: dict = {}
    seen: list = []
    recovered: dict = {}
    slack_adapter._app.client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1700000006.000601"})

    def complete(messages, **_):
        prompt = str(messages[-1].get("content") or "")
        if "【当前该用户名下尚未办完的待办承诺】" not in prompt:
            return None
        seen.append(prompt)
        return SimpleNamespace(text=json.dumps({
            "new_commitments": [],
            "resolutions": [
                {"id": ids[SLACK_LEGACY], "action": "cancelled",
                 "evidence": "本人说已还", "user_quote": TURN_USER},
                {"id": ids[ANCHOR], "action": "cancelled",
                 "evidence": "本人说已还", "user_quote": TURN_USER},
                # A cited id this turn never offered: the external row must stay pending.
                {"id": ids["external-owner"], "action": "cancelled",
                 "evidence": "本人说已还", "user_quote": TURN_USER},
            ],
        }, ensure_ascii=False))

    async def delivered_turn(restored):
        provider = _provider(tmp_path, "slack")
        provider._llm = SimpleNamespace(complete=complete)
        try:
            assert provider._owner_key == ANCHOR
            for owner_key, quote in ((SLACK_LEGACY, BOOK_QUOTE),
                                     (ANCHOR, REPORT_QUOTE),
                                     ("external-owner", OTHER_QUOTE)):
                cursor = provider._conn.execute(
                    "INSERT INTO commitments (owner_key, text, masked_text, source_quote,"
                    " turn_id, status, version, created_at, updated_at)"
                    " VALUES (?,?,?,?,'seed','pending',1,'2026-01-01','2026-01-01')",
                    (owner_key, quote, "", quote))
                ids[owner_key] = int(cursor.lastrowid)
            provider._conn.commit()
            # The reply leaves on the conversation the recovered turn actually came from.
            sent = await slack_adapter.send(
                restored.source.chat_id, TURN_REPLY,
                metadata={"team_id": restored.source.scope_id})
            assert sent.success, sent.error
            recovered["text"] = restored.text
            from sanae_mem.runtime.private_delivery_settler import (
                VerifiedPrivateDelivery, settle_private_delivery,
            )

            provider.on_session_switch("slack-commitment")
            settle_private_delivery(
                provider,
                VerifiedPrivateDelivery(
                    session_id="slack-commitment",
                    turn_id="slack-delivered-1",
                    platform="slack",
                    chat_id=restored.source.chat_id,
                    chat_type="dm",
                    sender_id="U_MAINT",
                    user_message=restored.text,
                    assistant_response=TURN_REPLY,
                    delivered_at=1700000000.0,
                    incoming_message_id=restored.message_id,
                    scope_id=restored.source.scope_id,
                    session_key=restored.source.chat_id,
                    is_bot=False,
                ),
            )
        finally:
            provider.shutdown()

    await _deliver(owner, await _slack_message(slack_adapter, text=TURN_USER),
                   session_id="slack-commitment", source="slack", turn=delivered_turn)
    assert recovered["text"] == TURN_USER, "sync_turn reads the text the queue restored"
    wire = slack_adapter._app.client.chat_postMessage.await_args.kwargs
    assert wire["channel"] == "D_CHAN", "the reply lands on the recovered inbound DM"

    snapshot = json.loads(seen[-1].rsplit("【当前该用户名下尚未办完的待办承诺】\n", 1)[1])
    assert [item["id"] for item in snapshot] == [ids[SLACK_LEGACY], ids[ANCHOR]]
    assert SLACK_LEGACY not in seen[-1] and ANCHOR not in seen[-1], "no owner internals in the prompt"

    with closing(sqlite3.connect(tmp_path / "data" / "sanae.db")) as conn:
        conn.row_factory = sqlite3.Row
        rows = {row["id"]: row for row in conn.execute("SELECT * FROM commitments")}

    for owner_key, quote in ((SLACK_LEGACY, BOOK_QUOTE), (ANCHOR, REPORT_QUOTE)):
        row = rows[ids[owner_key]]
        assert (row["status"], row["version"]) == ("cancelled", 2), row
        assert row["resolution_evidence"].startswith("user_cancellation:")
        assert TURN_USER in row["resolution_evidence"]
        assert (row["owner_key"], row["source_quote"], row["text"]) == (owner_key, quote, quote)

    external = rows[ids["external-owner"]]
    assert (external["status"], external["version"]) == ("pending", 1)
    assert not external["resolution_evidence"]
    assert (external["owner_key"], external["source_quote"]) == ("external-owner", OTHER_QUOTE)


# ── a memory binding never unlocks the host profile-file entries ──────────────────────────────

def _write_plugin_config(home: Path, bindings: list) -> None:
    lines = [
        "plugins:",
        "  sanae_mem:",
        f"    data_dir: {json.dumps(str(home / 'data'))}",
        "    semantic_backend: sqlite",
        "    shadow: false",
        "    memory_identity:",
        "      platform: qqbot",
        "      user_id: U_OPENID",
    ]
    if bindings:
        lines.append("    maintainer_account_bindings:")
        lines.extend("      - " + json.dumps(item) for item in bindings)
    else:
        lines.append("    maintainer_account_bindings: []")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_memory_binding_keeps_host_profile_files_closed_and_memory_working(
    tmp_path, owner, slack_adapter, monkeypatch,
):
    """Registering the real hooks: only the fixed account keeps host file rights."""
    profile_file = tmp_path / "operator-test.txt"
    profile_file.write_text("operator profile note", encoding="utf-8")
    _write_plugin_config(tmp_path, [])

    manager = PluginManager(scope_key=str(tmp_path))
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
    monkeypatch.setattr(auxiliary, "register", lambda _ctx: None)
    monkeypatch.setattr(auxiliary, "register_daily", lambda _ctx: None)
    monkeypatch.setattr(runtime_hooks, "sync_scheduled_jobs", lambda _cfg: None)
    entry.register(ctx)

    def read_file(source, session_id):
        return manager.invoke_hook("pre_tool_call", tool_name="read_file",
                                   args={"path": str(profile_file)}, session_id=session_id)

    def at_file(source, platform, session_key):
        return manager.invoke_hook(
            "post_gateway_admission", session_key=session_key, platform=platform,
            source=source, message_id="synthetic", text="@file:operator-test.txt")

    before: dict = {}

    def unbound(restored):
        source = restored.source.to_dict()
        before["read_file"] = read_file(source, "slack-gate")
        before["at_file"] = at_file(source, "slack", "agent:main:slack:dm:D_CHAN")

    await _deliver(owner, await _slack_message(slack_adapter), session_id="slack-before",
                   source="slack", turn=unbound)
    assert [item["action"] for item in before["read_file"]] == ["block"]
    assert [item["action"] for item in before["at_file"]] == ["handled"]

    # The only configuration change: the same Slack account joins the memory binding.
    _write_plugin_config(tmp_path, [
        {"platform": "slack", "user_id": "U_MAINT", "scope_id": "T_TEAM"}])
    after: dict = {}

    def bound(restored):
        source = restored.source.to_dict()
        after["read_file"] = read_file(source, "slack-gate-2")
        after["at_file"] = at_file(source, "slack", "agent:main:slack:dm:D_CHAN")
        provider = _provider(tmp_path, "slack")
        try:
            assert provider._owner_key == ANCHOR
            added = _add(provider, "记住窗台上的绿萝")
            found = json.loads(provider.handle_tool_call(
                "sanae_mem_search", {"query": "绿萝"}))["result"]
            assert [row["id"] for row in found] == [added], "the bound chat keeps its memory tools"
        finally:
            provider.shutdown()

    await _deliver(owner,
                   await _slack_capture(slack_adapter, _slack_event(
                       "D_CHAN", "im", "1700000000.000222", user="U_MAINT")),
                   session_id="slack-after", source="slack", turn=bound)
    assert [item["action"] for item in after["read_file"]] == ["block"], "a binding is not host authz"
    assert [item["action"] for item in after["at_file"]] == ["handled"]

    qq: dict = {}

    def qq_operator(restored):
        source = restored.source.to_dict()
        qq["read_file"] = read_file(source, "qq-gate")
        qq["at_file"] = at_file(source, "qqbot", "agent:main:qqbot:dm:U_OPENID")

    await _deliver(owner, await _qq_message(), session_id="qq-gate", source="qqbot",
                   turn=qq_operator)
    assert qq["read_file"] == [], "the fixed account keeps the host file entry it always had"
    assert qq["at_file"] == []

    # The legacy scope-less ``memory_identity``, with the main QQ now also listed explicitly in
    # its own app: the same account still keeps the host file entry it always had.
    _write_plugin_config(tmp_path, [
        {"platform": "slack", "user_id": "U_MAINT", "scope_id": "T_TEAM"},
        {"platform": "qqbot", "user_id": "U_OPENID", "scope_id": "102000001"},
    ])
    listed: dict = {}

    def qq_listed(restored):
        source = restored.source.to_dict()
        listed["read_file"] = read_file(source, "qq-listed")
        listed["at_file"] = at_file(source, "qqbot", "agent:main:qqbot:dm:U_OPENID")

    await _deliver(owner, await _qq_message(), session_id="qq-listed", source="qqbot",
                   turn=qq_listed)
    assert listed["read_file"] == [], "an explicit row for the fixed account adds nothing"
    assert listed["at_file"] == []
