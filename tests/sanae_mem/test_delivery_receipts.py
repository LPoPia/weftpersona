"""Exercise the read path against the unmodified Hermes final send."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import closing, contextmanager
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

from agent.memory_manager import MemoryManager
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.runtime_footer import build_footer_line
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from sanae_mem.cognitive import inner_wiring
from sanae_mem.runtime import delivery as runtime_delivery, hooks as runtime_hooks
from sanae_mem.cognitive.daily_context import daily_profile_key
from sanae_mem.runtime.delivery_receipts import clean_receipt_text, runtime_footer_pattern
from sanae_mem.infra.host_bridge import get_hermes_home
from sanae_mem.cognitive.inner_state_store import owner_state
from sanae_mem.plugin_entry import SanaeMemProvider
from sanae_mem.runtime.hooks import register_private_heart_hooks
from sanae_mem.governance.source_context import current_chat_scope, resolve_chat_identity
from sanae_mem.infra.textutil import _owner_key

assert Path(sys.modules[BasePlatformAdapter.__module__].__file__).resolve().is_relative_to(HERMES_ROOT)


class _Adapter(BasePlatformAdapter):
    def __init__(self, result: SendResult, while_sending):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.result = result
        self.while_sending = while_sending
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        self.while_sending()
        return self.result


@pytest.fixture
def receipt_query(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("gateway:\n  delivery_ledger: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return {
        "ledger": home / "state.db",
        "session_key": "agent:main:telegram:dm:100",
        "message_id": "200",
        "platform": "telegram",
        "chat_id": "100",
        "thread_id": "",
        "text": "这是这次私聊的回复。",
        "not_before": time.time(),
        "timeout": 0,
    }


def _enable_runtime_footer(home):
    with (home / "config.yaml").open("a", encoding="utf-8") as handle:
        handle.write("display:\n  runtime_footer:\n    enabled: true\n"
                     "    fields: [model, context_pct, latency]\n")
    return build_footer_line(
        user_config={"display": {"runtime_footer": {
            "enabled": True, "fields": ["model", "context_pct", "latency"],
        }}}, platform_key="telegram", model="provider/test-model", context_tokens=10,
        context_length=100, turn_seconds=1.2,
    )


def test_exact_model_footer_preserves_non_footer_paragraph(receipt_query):
    home = receipt_query["ledger"].parent
    footer = _enable_runtime_footer(home)
    pattern = runtime_footer_pattern(home, "telegram", "provider/test-model")
    assert pattern is not None and pattern.fullmatch(footer)
    assert clean_receipt_text("正文\n\n" + footer, footer_pattern=pattern) == "正文"
    assert clean_receipt_text("正文\n\n普通段落", footer_pattern=pattern) == "正文\n\n普通段落"
    assert clean_receipt_text("正文\n\nother-model · 10% · 1s", footer_pattern=pattern).endswith("other-model · 10% · 1s")


def test_missing_receipt_model_uses_generic_model_footer_pattern(receipt_query):
    home = receipt_query["ledger"].parent
    footer = _enable_runtime_footer(home)
    pattern = runtime_footer_pattern(home, "telegram", "")
    assert pattern is not None and pattern.fullmatch(footer)
    assert clean_receipt_text("正文\n\n" + footer, footer_pattern=pattern) == "正文"
    assert clean_receipt_text("正文\n\n普通段落", footer_pattern=pattern) == "正文\n\n普通段落"
    assert clean_receipt_text("wire reply", "host-clean reply", pattern) == "host-clean reply"


def test_delivery_receipts_queue_by_timestamp_after_same_owner_watches_resolve(tmp_path):
    from sanae_mem.cognitive import inner_wiring
    from sanae_mem.storage.background_tasks import BackgroundTasks

    target = (tmp_path / "sanae.db", "profile")
    applied = []
    background_tasks = BackgroundTasks()
    first_probe = threading.Event()
    release_probe = threading.Event()

    def probe_newer() -> tuple[bool, float | None]:
        if not first_probe.is_set():
            # The first poll only sees the newer watch. Both are ready on the next poll.
            first_probe.set()
            release_probe.wait(5.0)
            return False, None
        return True, 101.0

    inner_wiring._watch_delivery(
        target, probe_newer, lambda _stamp: applied.append("newer"),
        background_tasks=background_tasks,
    )
    try:
        assert first_probe.wait(5.0)
        inner_wiring._watch_delivery(
            target, lambda: (True, 100.0), lambda _stamp: applied.append("older"),
            background_tasks=background_tasks,
        )
    finally:
        release_probe.set()

    assert inner_wiring.wait_for_pending_updates(target, timeout=5.0)
    assert applied == ["older", "newer"]


def test_original_hermes_mood_advances_only_on_delivery_and_reaches_next_turn(
        receipt_query, request, monkeypatch):
    home = receipt_query["ledger"].parent
    footer = _enable_runtime_footer(home)
    data_dir = home / "sanae-mem"
    assert SanaeMemProvider({"semantic_backend": "sqlite"})._partner_role == "对话伙伴"
    provider = SanaeMemProvider({
        "data_dir": str(data_dir), "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "100"},
        "partner_name": "测试对象",
        "partner_role": "项目搭档",
    })
    requests = []
    private_thought = "我还想在窗边坐一会儿。"

    def complete(messages, *, purpose, **_):
        assert purpose == "inner_state_update"
        assert "测试对象" in messages[0]["content"]
        assert "项目搭档" in messages[0]["content"]
        requests.append(json.loads(messages[-1]["content"]))
        return SimpleNamespace(text=json.dumps({
            "sentences": [private_thought, "等天暗下来再把那本书翻开。"],
            "variables": {"tension": 0.1, "closeness": 0.4, "energy": -0.3},
            "shared_sentences": None,
        }, ensure_ascii=False))

    provider._llm = SimpleNamespace(complete=complete)
    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
    from hermes_host_contracts import register_delivery_receipt_handler
    captured = []

    def capture_registration(handler):
        captured.append(handler)
        return True

    with monkeypatch.context() as registration:
        registration.setattr(runtime_hooks, "register_delivery_receipt_handler", capture_registration)
        release = register_private_heart_hooks(ctx, provider)
    register_delivery_receipt_handler(captured[0])
    request.addfinalizer(release)

    @contextmanager
    def bind(message_id, actor="100"):
        reset_session_vars()
        tokens = set_session_vars(
            platform="telegram", chat_type="dm", chat_id=actor, user_id=actor,
            session_id="private-session" if actor == "100" else "other-private-session",
            session_key=f"agent:main:telegram:dm:{actor}",
            message_id=message_id,
        )
        try:
            yield current_chat_scope()
        finally:
            clear_session_vars(tokens)

    def send_turn(message_id, turn_id, answer, result, check_while_sending):
        event = MessageEvent(
            text="我刚把书桌挪到窗边。", message_type=MessageType.TEXT, message_id=message_id,
            source=SessionSource(
                platform=Platform.TELEGRAM, chat_id="100", chat_type="dm", user_id="100",
                message_id=message_id,
            ),
        )
        adapter = _Adapter(result, check_while_sending)
        with bind(message_id):
            manager.invoke_hook(
                "post_llm_call", session_id="private-session", turn_id=turn_id, platform="telegram",
                user_message=event.text, assistant_response=answer, model="provider/test-model",
            )
            check_while_sending()
            event._resolved_session_id = "private-session"
            event._model = "provider/test-model"
            # No receipt turn_id: exercise the post_llm_call text-to-turn binding.
            delivered = []
            asyncio.run(adapter._send_final_text(
                event, receipt_query["session_key"], f"{answer}\n\n{footer}", {},
                False, 0, delivered.append,
            ))
        assert delivered[0].success is result.success

    native_memory = None
    if callable(getattr(MemoryManager, "current_turn_context", None)):
        native_memory = MemoryManager()
        request.addfinalizer(native_memory.shutdown_all)
        with bind("setup"):
            native_memory.add_provider(provider)
            native_memory.initialize_all(
                session_id="private-session", hermes_home=str(home),
                agent_context="primary", platform="telegram",
            )

    def next_turn(message_id):
        turn = {
            "session_id": "private-session", "turn_id": f"after-{message_id}",
            "platform": "telegram", "user_message": "窗外下雨了。",
        }
        with bind(message_id):
            replies = manager.invoke_hook("pre_llm_call", **turn)
            fallback = "\n".join(reply.get("context", "") for reply in replies)
            if native_memory is None:
                return fallback
            assert not fallback, "the fallback must not duplicate native Tier 2 context"
            return native_memory.current_turn_context(sender_id="100", synthetic=False, **turn)

    with bind("200") as scope:
        owner = scope.owner
    profile = daily_profile_key(home)

    def saved():
        database = data_dir / "sanae.db"
        return owner_state(database, profile, owner) if database.is_file() else None

    def no_state_before_ack():
        assert not requests
        assert saved() is None

    # Both another owner's missing reply and this owner's older missing reply
    # must stay outside the serialized update queue for the delivered turn.
    with bind("190", actor="999") as scope:
        other_owner = scope.owner
        manager.invoke_hook(
            "post_llm_call", session_id="other-private-session", turn_id="other-missing", platform="telegram",
            user_message="另一位对话者的话。", assistant_response="另一位没收到的回复。", model="provider/test-model",
        )
    with bind("191"):
        manager.invoke_hook(
            "post_llm_call", session_id="private-session", turn_id="own-missing", platform="telegram",
            user_message="较早的一句话。", assistant_response="较早没收到的回复。", model="provider/test-model",
        )
    no_state_before_ack()

    send_turn("200", "delivered-turn", "窗边光线好，坐过去试试。", SendResult(success=True, message_id="201"),
              no_state_before_ack)
    assert private_thought in next_turn("202")
    confirmed = saved()
    assert confirmed and confirmed.turn_id == "delivered-turn" and confirmed.revision == 1
    assert owner_state(data_dir / "sanae.db", profile, other_owner) is None
    assert [request["delivered_reply"] for request in requests] == ["窗边光线好，坐过去试试。"]

    def same_state_before_failure():
        assert saved() == confirmed
        assert len(requests) == 1

    send_turn("203", "failed-turn", "没送到的另一句话。", SendResult(success=False, error="chat_not_found"),
              same_state_before_failure)
    assert private_thought in next_turn("204")
    assert saved() == confirmed
    assert len(requests) == 1
    def watermark():
        with closing(sqlite3.connect(data_dir / "sanae.db")) as conn:
            return conn.execute(
                "SELECT delivered_at, turn_id FROM session_reply_watermarks WHERE owner_key=? AND session_id=?",
                (owner, "private-session"),
            ).fetchone()

    assert watermark() == (confirmed.processed_at, "delivered-turn")

    for message_id, turn_id, answer in (
        ("205", "second-delivered", "我刚看见窗外有一盏路灯。"),
        ("207", "third-delivered", "路灯照亮了湿漉漉的树叶。"),
    ):
        send_turn(message_id, turn_id, answer, SendResult(success=True, message_id=message_id + "-out"),
                  lambda: None)
    assert private_thought in next_turn("209")
    latest = saved()
    assert latest.turn_id == "third-delivered" and latest.revision == 3
    assert [request["delivered_reply"] for request in requests] == [
        "窗边光线好，坐过去试试。", "我刚看见窗外有一盏路灯。", "路灯照亮了湿漉漉的树叶。",
    ]

    # A duplicate native receipt can carry a later timestamp.
    # The same turn still must not advance mood or its processing watermark twice.
    send_turn("207", "third-delivered", "路灯照亮了湿漉漉的树叶。",
              SendResult(success=True, message_id="207-duplicate"), lambda: None)
    assert private_thought in next_turn("210")
    assert saved() == latest and len(requests) == 3
    assert watermark() == (latest.processed_at, "third-delivered")


# ── receipt provenance: the dispatch-time scene/key, never ambient session state ─────────────


class _ReceiptRecorder(SanaeMemProvider):
    """Record the verified delivery and its scope after real event settlement."""

    def __init__(self):
        super().__init__({"semantic_backend": "sqlite"})
        self.opened = []
        self.applied = []
        self.done = threading.Event()
        self._inner_payloads = []
        self._active_session_id = ""
        self.attach_runtime_llm(SimpleNamespace(complete=self._complete))

    def _complete(self, messages, *, purpose="", **_):
        if purpose == "inner_state_update":
            for message in reversed(messages):
                content = message.get("content") if isinstance(message, dict) else ""
                if isinstance(content, str) and content.lstrip().startswith("{"):
                    try:
                        self._inner_payloads.append(json.loads(content))
                    except ValueError:
                        pass
                    break
            return SimpleNamespace(text=json.dumps({
                "sentences": ["还想在窗边多坐一会儿。"],
                "variables": {"tension": 0.1, "closeness": 0.4, "energy": -0.3},
                "shared_sentences": None,
            }, ensure_ascii=False))
        return SimpleNamespace(text="{}")

    def initialize(self, session_id, **kwargs):
        self._active_session_id = session_id
        self.opened.append((session_id, kwargs))
        super().initialize(session_id, **kwargs)

    def shutdown(self):
        try:
            if self.sqlite.is_open and self._active_session_id:
                row = self.sqlite.fetchone(
                    "SELECT session_id, turn_id, delivered_at "
                    "FROM session_reply_watermarks WHERE session_id=?",
                    (self._active_session_id,),
                )
                if row is not None:
                    payload = self._inner_payloads[-1] if self._inner_payloads else {}
                    self.applied.append((
                        {
                            "session_id": row["session_id"],
                            "turn_id": row["turn_id"],
                            "user_message": payload.get("user", ""),
                            "assistant_response": payload.get("delivered_reply", ""),
                            "delivered_at": float(row["delivered_at"]),
                        },
                        current_chat_scope(),
                    ))
                    self.done.set()
        finally:
            super().shutdown()


@contextmanager
def _registered_receipt_handler(ctx, provider, monkeypatch, home=None):
    """Register the plugin's real handler through the host registry, then remove it.

    *home* is the Hermes HOME the provider is registered for; the process is back on its launch
    HOME before the test sends. None registers in whatever HOME the caller already bound."""
    from hermes_host_contracts import register_delivery_receipt_handler

    captured = []

    def capture_registration(handler):
        captured.append(handler)
        return True

    if home is None:
        monkeypatch.setattr(runtime_hooks, "register_delivery_receipt_handler", capture_registration)
        release = register_private_heart_hooks(ctx, provider)
    else:
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


def _recording_worker(recorder):
    """The production open hand-off, recorded: the worker sees the scope's own session/platform."""
    def open_worker(_prototype, scope, home):
        recorder.initialize(scope.session_id, platform=scope.platform, hermes_home=home, hook_worker=True)
        return recorder

    return open_worker


def _receipt_provider(home):
    provider = SanaeMemProvider({
        "data_dir": str(home / "sanae-mem"), "semantic_backend": "sqlite",
        "memory_identity": {"platform": "slack", "user_id": "U_SAME"},
    })
    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
    return provider, ctx


def _slack_source(*, chat_id="D_CHAN", chat_type="dm", scene="dm", scope_id="T_ONE"):
    return SessionSource(
        platform=Platform.SLACK, chat_id=chat_id, chat_type=chat_type, user_id="U_SAME",
        message_id="201", scope_id=scope_id, scene_chat_type=scene,
    )


def _send_final(adapter, source, *, session_id, turn_id, session_key,
                answer, text="我刚把书桌挪到窗边。"):
    event = MessageEvent(text=text, message_type=MessageType.TEXT, message_id="201",
                         source=source)
    event._resolved_session_id = session_id
    event._turn_id = turn_id
    event._model = "provider/test-model"
    asyncio.run(adapter._send_final_text(
        event, session_key, answer, {}, False, 0, lambda _result: None))


def test_slack_dm_receipt_rebuilds_the_private_scope_once(receipt_query, monkeypatch):
    home = receipt_query["ledger"].parent
    provider, ctx = _receipt_provider(home)
    recorder = _ReceiptRecorder()
    user_text = "我打算把书桌挪到窗边。"
    with _registered_receipt_handler(ctx, provider, monkeypatch):
        monkeypatch.setattr(runtime_delivery, "_open_private_hook_provider",
                            _recording_worker(recorder))
        reset_session_vars()
        assert current_chat_scope().visibility == "unknown", "no inbound turn is left bound"
        adapter = _Adapter(SendResult(success=True, message_id="301"), lambda: None)
        source = _slack_source()
        _send_final(adapter, source, session_id="slack-private-session",
                    turn_id="slack-turn-1", session_key="slack:T_ONE:dm:D_CHAN",
                    answer="窗边光线好，坐过去试试。", text=user_text)
        assert recorder.done.wait(10), "the production queue never applied the delivery"

        # The same native receipt delivered twice still updates once.
        _send_final(adapter, source, session_id="slack-private-session",
                    turn_id="slack-turn-1", session_key="slack:T_ONE:dm:D_CHAN",
                    answer="窗边光线好，坐过去试试。", text=user_text)
        assert len(recorder.applied) == 1
        assert len(recorder.opened) == 1

    kwargs, scope = recorder.applied[0]
    assert kwargs["session_id"] == "slack-private-session"
    assert kwargs["turn_id"] == "slack-turn-1"
    assert kwargs["user_message"] == user_text
    assert kwargs["assistant_response"] == "窗边光线好，坐过去试试。"
    assert isinstance(kwargs["delivered_at"], float)
    assert scope.visibility == "private"
    assert scope.session_key == "slack:T_ONE:dm:D_CHAN"
    assert (scope.scope_id, scope.thread_id) == ("T_ONE", "")
    assert scope.owner == _owner_key("slack", user_id="U_SAME", scope_id="T_ONE")
    assert scope.profile == "default", \
        "the standalone host home is its own profile; no cleared inbound profile was inherited"
    assert recorder.opened[0][0] == "slack-private-session"
    assert recorder.opened[0][1]["platform"] == "slack"
    with closing(sqlite3.connect(home / "sanae-mem" / "sanae.db")) as conn:
        rows = conn.execute(
            "SELECT owner_key, source_snapshot, source_context FROM events WHERE session_id=?",
            ("slack-private-session",),
        ).fetchall()
    assert len(rows) == 1
    owner_key, snapshot, source = rows[0]
    assert owner_key == scope.owner and snapshot == ""
    assert json.loads(source)["kind"] == "external_user"


def test_named_profile_receipt_binds_the_worker_envelope_and_home(receipt_query, monkeypatch):
    """A receipt from a named runtime profile reaches the queue worker as that profile.

    The source states no profile and the dispatch-time key is the historical ``agent:main``
    namespace: only the host's pinned ``RoutingIdentity.runtime_profile`` names ``named-ops``, so
    the worker must never re-derive a profile from the key or fall back to ``default``. The
    inbound context is cleared first, so the binding is rebuilt, not inherited. The provider is
    registered while the process sits in the named HOME, as the real load does; the send then
    runs on the launch HOME and only the pinned identity names the named one."""
    from gateway.session_identity import RoutingIdentity

    home = receipt_query["ledger"].parent / "profiles" / "named-ops"
    home.mkdir(parents=True)
    data_dir = home / "sanae-mem"
    config = {"data_dir": str(data_dir), "semantic_backend": "sqlite",
              "memory_identity": {"platform": "slack", "user_id": "U_SAME"},
              # The one trusted declaration: this workspace, in this profile only.
              "maintainer_account_bindings": [
                  {"profile": "named-ops", "platform": "slack", "user_id": "U_SAME",
                   "scope_id": "T_ONE"},
              ]}
    provider = SanaeMemProvider(config)
    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
    observed = SimpleNamespace(scope=None, home=None, identity=None, written_owner=None,
                              read_owners=(), live_owner=None)
    completed = []

    def complete(messages, *, purpose, **_):
        completed.append(purpose)
        return SimpleNamespace(text=json.dumps({
            "sentences": ["还想在窗边多坐一会儿。"],
            "variables": {"tension": 0.1, "closeness": 0.4, "energy": -0.3},
            "shared_sentences": None,
        }, ensure_ascii=False))

    def open_real_worker(_prototype, scope, worker_home):
        observed.scope, observed.home = scope, get_hermes_home()
        worker = SanaeMemProvider(config)
        worker._llm = SimpleNamespace(complete=complete)
        worker.initialize(scope.session_id, platform=scope.platform,
                          hermes_home=worker_home, hook_worker=True)
        # Read on the worker's own connection, from the same scope the queue rebuilt.
        observed.identity = resolve_chat_identity(scope, config, connection=worker._conn)
        observed.written_owner = worker._owner_key
        observed.read_owners = tuple(worker._read_owner_keys)
        observed.live_owner = worker.resolve_private_delivery_route().owner_key
        return worker

    def watermark():
        if not (data_dir / "sanae.db").is_file():
            return None
        with closing(sqlite3.connect(data_dir / "sanae.db")) as conn:
            return conn.execute(
                "SELECT turn_id FROM session_reply_watermarks WHERE session_id=?",
                ("slack-named-session",),
            ).fetchone()

    try:
        with _registered_receipt_handler(ctx, provider, monkeypatch, home=home):
            monkeypatch.setattr(runtime_delivery, "_open_private_hook_provider", open_real_worker)
            reset_session_vars()
            assert current_chat_scope().visibility == "unknown", "no inbound turn is left bound"
            assert get_hermes_home() == receipt_query["ledger"].parent, \
                "registration ran in the named HOME; the send stays on the launch home"
            source = _slack_source()
            source.profile = None
            # The host pinned this routing decision at ingress; the receipt names it.
            source._identity = RoutingIdentity(
                transport_profile="default", runtime_profile="named-ops",
                authorization_home=home.parent.parent, runtime_home=home, multiplexed=False)
            adapter = _Adapter(SendResult(success=True, message_id="601"), lambda: None)
            _send_final(adapter, source, session_id="slack-named-session",
                        turn_id="slack-named-turn", session_key="agent:main:slack:dm:D_CHAN",
                        answer="这一条从命名档案里发出。")

            deadline = time.monotonic() + 10
            while watermark() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert watermark() == ("slack-named-turn",), "the real worker never applied it"

            # A second copy of the same native receipt must not reach the queue again.
            _send_final(adapter, source, session_id="slack-named-session",
                        turn_id="slack-named-turn", session_key="agent:main:slack:dm:D_CHAN",
                        answer="这一条从命名档案里发出。")
            inner_wiring._wait_for_pending((data_dir.resolve(), daily_profile_key(home)))
            with closing(sqlite3.connect(data_dir / "sanae.db")) as conn:
                assert conn.execute(
                    "SELECT COUNT(*) FROM session_reply_watermarks").fetchone() == (1,)
            assert completed.count("inner_state_update") == 1, \
                "the worker distilled one inner state for the applied receipt"
    finally:
        provider.shutdown()

    anchor = _owner_key("slack", user_id="U_SAME")
    workspace_owner = _owner_key("slack", user_id="U_SAME", scope_id="T_ONE")
    assert observed.scope.profile == "named-ops"
    assert observed.scope.session_key == "agent:main:slack:dm:D_CHAN"
    assert observed.home == home
    assert observed.identity.maintainer is True, "the declared workspace is this person"
    assert observed.identity.private_owner_key == anchor, "written under the existing anchor"
    assert observed.written_owner == anchor
    assert observed.live_owner == anchor, "the live worker resolves that same anchor"
    assert set(observed.read_owners) >= {anchor, workspace_owner}, \
        "the workspace row stays readable, at its own historical owner"


def test_slack_receipts_without_workspace_scene_or_key_do_not_update(receipt_query, monkeypatch):
    home = receipt_query["ledger"].parent
    provider, ctx = _receipt_provider(home)
    with _registered_receipt_handler(ctx, provider, monkeypatch):
        reset_session_vars()
        adapter = _Adapter(SendResult(success=True, message_id="401"), lambda: None)
        refused = (
            # multi-person DM: the scene is public
            (_slack_source(chat_id="G_MPIM", chat_type="group", scene="group"),
             "slack:T_ONE:group:G_MPIM:U_SAME"),
            # a confirmed 1:1 IM with no workspace
            (_slack_source(scope_id=None), "slack:T_ONE:dm:D_CHAN"),
            # transport dm but no trusted scene
            (_slack_source(scene=None), "slack:T_ONE:dm:D_CHAN"),
            # a Slack receipt with no dispatch-time key
            (_slack_source(), ""),
        )
        for index, (source, session_key) in enumerate(refused):
            recorder = _ReceiptRecorder()
            monkeypatch.setattr(runtime_delivery, "_open_private_hook_provider",
                                _recording_worker(recorder))
            _send_final(adapter, source, session_id=f"slack-refused-{index}",
                        turn_id=f"slack-refused-turn-{index}", session_key=session_key,
                        answer="不该落库的回复。")
            assert recorder.applied == [] and recorder.opened == []

        control = _ReceiptRecorder()
        monkeypatch.setattr(runtime_delivery, "_open_private_hook_provider",
                            _recording_worker(control))
        _send_final(adapter, _slack_source(), session_id="slack-private-control",
                    turn_id="slack-control-turn", session_key="slack:T_ONE:dm:D_CHAN",
                    answer="这一条才该落库。")
        assert control.done.wait(10)

    _, scope = control.applied[0]
    assert scope.visibility == "private"
    assert scope.session_key == "slack:T_ONE:dm:D_CHAN"


def test_legacy_receipt_without_actual_key_keeps_the_non_slack_fallback(receipt_query, monkeypatch):
    """An older host states no dispatch-time key: other platforms keep the synthesized one."""
    from hermes_host_contracts import DeliveryReceipt, emit_delivery_receipt

    home = receipt_query["ledger"].parent
    provider, ctx = _receipt_provider(home)
    with _registered_receipt_handler(ctx, provider, monkeypatch):
        reset_session_vars()
        telegram = _ReceiptRecorder()
        monkeypatch.setattr(runtime_delivery, "_open_private_hook_provider",
                            _recording_worker(telegram))
        emit_delivery_receipt(DeliveryReceipt(
            session_id="legacy-telegram-session", platform="telegram", chat_id="100",
            delivered_text="旧式回复。", delivered_at=time.time(), transport_confirmed=True,
            chat_type="dm", sender_id="100", turn_id="legacy-turn", message_id="legacy-msg",
            user_message="旧式一句话。",
        ))
        assert telegram.done.wait(10)
        _, scope = telegram.applied[0]
        assert scope.session_key == "agent:main:telegram:dm:100"
        assert scope.visibility == "private"
        assert scope.owner == _owner_key("telegram", user_id="100")

        slack = _ReceiptRecorder()
        monkeypatch.setattr(runtime_delivery, "_open_private_hook_provider",
                            _recording_worker(slack))
        emit_delivery_receipt(DeliveryReceipt(
            session_id="legacy-slack-session", platform="slack", chat_id="D_CHAN",
            delivered_text="旧式回复。", delivered_at=time.time(), transport_confirmed=True,
            chat_type="dm", sender_id="U_SAME", turn_id="legacy-slack-turn",
            message_id="legacy-slack-msg", user_message="旧式一句话。",
        ))
        assert slack.applied == [] and slack.opened == []


def test_qq_private_receipt_keeps_its_existing_owner_update(receipt_query, monkeypatch):
    home = receipt_query["ledger"].parent
    provider, ctx = _receipt_provider(home)
    recorder = _ReceiptRecorder()
    with _registered_receipt_handler(ctx, provider, monkeypatch):
        monkeypatch.setattr(runtime_delivery, "_open_private_hook_provider",
                            _recording_worker(recorder))
        reset_session_vars()
        source = SessionSource(
            platform=Platform.QQBOT, chat_id="U_OPENID", chat_type="dm", user_id="U_OPENID",
            message_id="204", scope_id="102000001",
        )
        adapter = _Adapter(SendResult(success=True, message_id="501"), lambda: None)
        _send_final(adapter, source, session_id="qq-private-session",
                    turn_id="qq-turn-1", session_key="qqbot:dm:U_OPENID",
                    answer="窗边光线好，坐过去试试。")
        assert recorder.done.wait(10)

    _, scope = recorder.applied[0]
    assert scope.visibility == "private"
    assert scope.session_key == "qqbot:dm:U_OPENID"
    assert scope.scope_id == "102000001"
    assert scope.owner == _owner_key("qqbot", user_id="U_OPENID")


# ── two HOME-bound providers, one dispatch: the receipt belongs to the sending HOME only ─────


class _InnerStateCounter:
    """A provider's model stand-in: counts the purposes that provider actually asked for."""

    def __init__(self):
        self.purposes = []
        self.llm = SimpleNamespace(complete=self.complete)

    def complete(self, messages, *, purpose, **_):
        self.purposes.append(purpose)
        return SimpleNamespace(text=json.dumps({
            "sentences": ["还想在窗边多坐一会儿。"],
            "variables": {"tension": 0.1, "closeness": 0.4, "energy": -0.3},
            "shared_sentences": None,
        }, ensure_ascii=False))


def _watermark_turns(data_dir, session_id):
    database = data_dir / "sanae.db"
    if not database.is_file():
        return []
    with closing(sqlite3.connect(database)) as conn:
        return [row[0] for row in conn.execute(
            "SELECT turn_id FROM session_reply_watermarks WHERE session_id=?", (session_id,))]


def test_receipt_applies_only_in_the_provider_registered_for_the_dispatch_home(tmp_path, monkeypatch):
    """Two providers actually loaded from two HOMEs; one dispatched receipt reaches one database.

    Each HOME activates ``memory.provider=sanae_mem`` with its own custom ``data_dir`` and loads
    the plugin through the ordinary ``plugins.memory.load_memory_provider``, so both register their
    own process-global receipt handler. A real Slack DM is taken in through the production
    ``SlackAdapter._handle_slack_message`` and its ``MessageEvent`` is sent back out through the
    real final send. The host pins ``RoutingIdentity.runtime_home`` around the synchronous emit,
    which is the HOME the receipt was sent from: only that provider may queue the delivery, or the
    same turn lands in both databases.

    Only the named profile's config declares the workspace ``maintainer_account_bindings`` entry,
    so the one real worker the filtered handler opens is observed as that profile: it writes under
    the existing ``memory_identity`` anchor, still reads the workspace's own historical owner, and
    gains no host profile-file right from the new binding. The unbound-identity negative case stays
    in ``tests/test_multiplatform_account_binding.py``."""
    import importlib
    from unittest.mock import AsyncMock, patch

    import hermes_host_contracts
    import plugins.memory as host_memory
    from gateway.session_identity import RoutingIdentity
    from plugins.platforms.slack.adapter import SlackAdapter

    unnamed_home = tmp_path / "hermes"
    named_home = unnamed_home / "profiles" / "named-review"
    data_dirs = {}
    for home in (unnamed_home, named_home):
        home.mkdir(parents=True)
        data_dir = home / f"{home.name}-memory"
        data_dirs[home] = data_dir
        binding = (
            # The one trusted declaration: this workspace account, in this profile only.
            "    maintainer_account_bindings:\n"
            "      - profile: named-review\n"
            "        platform: slack\n"
            "        scope_id: T_TEAM\n"
            "        user_id: U_SAME\n"
            if home == named_home else ""
        )
        (home / "config.yaml").write_text(
            "memory:\n  provider: sanae_mem\n"
            "plugins:\n  sanae_mem:\n"
            f"    data_dir: {data_dir}\n"
            "    semantic_backend: sqlite\n"
            "    memory_identity:\n      platform: slack\n      user_id: U_SAME\n"
            + binding,
            encoding="utf-8",
        )
    assert data_dirs[unnamed_home] != data_dirs[named_home]

    plugin_source = Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem"
    assert (plugin_source / "plugin_entry.py").is_file()
    on_disk_find = host_memory.find_provider_dir
    monkeypatch.setattr(
        host_memory, "find_provider_dir",
        lambda name: plugin_source if name == "sanae_mem" else on_disk_find(name),
    )

    registered = []
    real_register = hermes_host_contracts.register_delivery_receipt_handler

    def record(handler):
        registered.append(handler)
        real_register(handler)

    monkeypatch.setattr(hermes_host_contracts, "register_delivery_receipt_handler", record)
    providers, models = {}, {}
    try:
        for home in (unnamed_home, named_home):
            with monkeypatch.context() as bound_home:
                bound_home.setenv("HERMES_HOME", str(home))
                provider = host_memory.load_memory_provider("sanae_mem")
            assert provider is not None, "the ordinary loader found the configured provider"
            models[home] = _InnerStateCounter()
            provider._llm = models[home].llm
            providers[home] = provider
            assert Path(provider._config["data_dir"]).resolve() == data_dirs[home].resolve(), \
                "each provider kept the data_dir its own HOME configured"
        assert len(registered) == 2, "each loaded provider registered one receipt handler"
        assert providers[named_home]._config["maintainer_account_bindings"] == [{
            "profile": "named-review", "platform": "slack", "scope_id": "T_TEAM",
            "user_id": "U_SAME",
        }]
        assert not providers[unnamed_home]._config.get("maintainer_account_bindings"), \
            "the default profile declares no account binding"

        module_name = type(providers[named_home]).__module__
        package = (
            module_name.rsplit(".", 2)[0]
            if module_name.endswith(".runtime.provider")
            else module_name.rsplit(".", 1)[0]
        )
        wiring = importlib.import_module(f"{package}.cognitive.inner_wiring")
        daily = importlib.import_module(f"{package}.cognitive.daily_context")
        targets = {
            home: (data_dirs[home].resolve(), daily.daily_profile_key(home)) for home in providers
        }
        # The real worker the plugin's own handler opens for the applied receipt; wrapped, not
        # replaced, so the actual provider, extraction and queue stay in charge.
        entry = importlib.import_module(f"{package}.runtime.delivery")
        source_context = importlib.import_module(f"{package}.governance.source_context")
        real_open_worker = entry._open_private_hook_provider
        workers = []

        def observe_worker(prototype, scope, home):
            worker = real_open_worker(prototype, scope, home)
            workers.append((Path(home).resolve(), SimpleNamespace(
                scope=scope, worker=worker, owner=worker.resolve_private_delivery_route().owner_key,
                identity=source_context.resolve_chat_identity(
                    scope, worker._config, connection=worker._conn),
            )))
            return worker

        monkeypatch.setattr(entry, "_open_private_hook_provider", observe_worker)

        observed = []

        def observe_home(receipt):
            observed.append((receipt, Path(get_hermes_home()).resolve()))

        hermes_host_contracts.register_delivery_receipt_handler(observe_home)
        try:
            monkeypatch.setenv("HERMES_HOME", str(unnamed_home))
            reset_session_vars()
            assert current_chat_scope().visibility == "unknown", "no inbound turn is left bound"
            assert get_hermes_home() == unnamed_home, "the process runs on the unnamed home"

            # Real Slack intake: the production handler parses the wire event and dispatches one
            # MessageEvent. Only the Slack SDK and the peripheral lookups are doubled.
            slack = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
            slack._app = SimpleNamespace(client=AsyncMock())
            slack._bot_user_id = "U_BOT"
            slack._running = True
            inbound = []
            slack.handle_message = AsyncMock(side_effect=lambda event: inbound.append(event))
            wire_event = {
                "channel": "D_REVIEW", "channel_type": "im", "team": "T_TEAM",
                "user": "U_SAME", "text": "我刚把书桌挪到窗边。",
                "ts": "1700000000.000701", "thread_ts": "1700000000.000701",
            }
            with patch.object(slack, "_resolve_user_name", new=AsyncMock(return_value="Test User")), \
                 patch.object(slack, "_resolve_channel_name", new=AsyncMock(return_value="review")), \
                 patch.object(slack, "_set_assistant_thread_title", new=AsyncMock()), \
                 patch.object(slack, "_hydrate_thread_context",
                              new=AsyncMock(return_value=("", [], []))), \
                 patch.object(slack, "_collect_inbound_media",
                              new=AsyncMock(return_value=([], [], [], wire_event["text"]))), \
                 patch.object(slack, "_channel_gate_allows", new=AsyncMock(return_value=True)), \
                 patch.object(slack, "_peer_bot_drop", new=AsyncMock(return_value=False)), \
                 patch.object(slack, "_drop_bot_sender", new=AsyncMock(return_value=False)):
                asyncio.run(slack._handle_slack_message(wire_event))
            assert len(inbound) == 1, "the real Slack intake dispatched exactly one MessageEvent"
            message = inbound[0]
            reset_session_vars()  # the inbound turn is over; the receipt must rebuild the scope
            assert current_chat_scope().visibility == "unknown", "no inbound turn is left bound"
            assert (message.source.chat_id, message.source.chat_type) == ("D_REVIEW", "dm")
            assert message.source.scope_id == "T_TEAM"
            assert message.source.thread_id == wire_event["ts"]
            assert (message.source.user_id, message.text) == ("U_SAME", wire_event["text"])

            message.source.profile = None
            # The host pinned this routing decision at ingress; the receipt names it.
            message.source._identity = RoutingIdentity(
                transport_profile="default", runtime_profile="named-review",
                authorization_home=unnamed_home, runtime_home=named_home, multiplexed=True)
            # The named dispatch key for this Slack DM; the receipt must name exactly this route.
            session_key = "agent:named-review:slack:dm:D_REVIEW"

            # The same inbound MessageEvent goes through the real SlackAdapter final send; only
            # the SDK transport is doubled, and it acks with a real-shaped ts.
            slack.send = AsyncMock(
                return_value=SendResult(success=True, message_id="1700000000.000799"))
            message._resolved_session_id = "slack-two-home-session"
            message._turn_id = "slack-two-home-turn"
            message._model = "provider/test-model"
            delivered = []
            for _ in range(2):
                asyncio.run(slack._send_final_text(
                    message, session_key, "这一条从命名档案里发出。", {}, False, 0,
                    delivered.append))
            assert [result.message_id for result in delivered] == \
                ["1700000000.000799"] * 2
            sent = slack.send.await_args.kwargs
            assert sent["chat_id"] == message.source.chat_id == "D_REVIEW"
            assert sent["reply_to"] == message.message_id == wire_event["ts"]
            assert sent["content"] == "这一条从命名档案里发出。"
            assert get_hermes_home() == unnamed_home, "the send returns to the launch home"
            reset_session_vars()
            assert current_chat_scope().visibility == "unknown", "the send left no turn bound"

            deadline = time.monotonic() + 10
            while (_watermark_turns(data_dirs[named_home], "slack-two-home-session") == []
                   and time.monotonic() < deadline):
                time.sleep(0.02)
            for target in targets.values():
                wiring._wait_for_pending(target)
        finally:
            hermes_host_contracts.unregister_delivery_receipt_handler(observe_home)

        assert [home for _, home in observed] == [named_home.resolve()] * 2, \
            "the host pinned the dispatch HOME around both emits"
        receipts = [receipt for receipt, _ in observed]
        assert [receipt.message_id for receipt in receipts] == ["1700000000.000799"] * 2
        assert {(receipt.chat_id, receipt.scope_id, receipt.thread_id, receipt.profile,
                 receipt.session_key, receipt.sender_id) for receipt in receipts} == {
            ("D_REVIEW", "T_TEAM", wire_event["ts"], "named-review", session_key, "U_SAME")}
        assert [receipt.user_message for receipt in receipts] == [wire_event["text"]] * 2

        assert _watermark_turns(data_dirs[named_home], "slack-two-home-session") == \
            ["slack-two-home-turn"]
        assert _watermark_turns(data_dirs[unnamed_home], "slack-two-home-session") == [], \
            "the provider loaded for the other HOME never queued this delivery"
        assert "inner_state_update" in models[named_home].purposes
        assert "daily_context_extract" in models[named_home].purposes
        assert models[unnamed_home].purposes == []

        # The one real worker, read outside its thread: it ran as the named profile and resolved
        # the declared workspace account through this instance's own connection.
        assert [home for home, _ in workers] == [named_home.resolve()], \
            "only the filtered-in provider opened a worker; the default one opened none"
        view = workers[0][1]
        anchor = _owner_key("slack", user_id="U_SAME")
        workspace_owner = _owner_key("slack", user_id="U_SAME", scope_id="T_TEAM")
        assert view.scope.profile == "named-review"
        assert view.scope.session_key == session_key
        assert view.worker._data_dir.resolve() == data_dirs[named_home].resolve()
        assert view.identity.private_owner_key == anchor == view.worker._owner_key, \
            "the bound workspace account writes under the existing memory_identity anchor"
        assert view.owner == anchor, "the live worker resolves that same anchor"
        assert set(view.identity.private_read_owner_keys) >= {anchor, workspace_owner}, \
            "the workspace's own historical owner stays readable"
        assert view.identity.maintainer is True
        assert view.identity.host_operator is False, \
            "the new memory binding never carries host profile-file or workspace rights"
    finally:
        for handler in registered:
            hermes_host_contracts.unregister_delivery_receipt_handler(handler)
        for provider in providers.values():
            provider.shutdown()
