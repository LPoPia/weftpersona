"""Clean-subprocess real-gateway end-to-end deterministic integration tests.

Covers Task §③ items 1–3:
1. Cold-start (no pre-imported ``agent.memory_manager``) two-turn DM via real
   ``discover_plugins()`` + ``GatewayRunner`` + ``AIAgent`` + HTTP LLM wire:
   - Both turns' HTTP wire requests contain latest ``[我眼中的世界]`` & ``[我内心的感受]``.
   - Turn 2's HTTP wire request history for Turn 1 and ``state.db`` ``api_content``
     have zero stale dynamic blocks.
2. Strict "no DB write before real DeliveryReceipt" + all 3 DM delivery paths:
   - Platform send failure -> ``delivery_ledger``: ``events``, ``daily_context``,
     ``commitments``, ``inner_state_revisions`` all stay at 0 new rows.
   - ``delivery_ledger`` startup redelivery (``_redeliver_claimed_obligations``),
     DM ``/queue`` queued reply (``_send_queued_final_text``), and normal DM direct
     send: each emits a single idempotent ``DeliveryReceipt`` and persists all
     target tables (including commitment creation & fulfillment).
3. Multi-sentence quote evidence gate + resident ``MEMORY.md`` routing:
   - ``[Replying to ...]`` & QQ ``[Quoted message; length=N]:`` middle sentences
     strictly return ``allowed=False`` (``USER_UNPROVEN``), while real trailing
     user text returns ``USER_EXTRACTED``.
   - Real ``tools.memory_tool.memory_tool`` with ``memory_enabled: true`` routes
     self-critique / single trivia to ``sanae.db`` (``MEMORY.md`` untouched) and
     allows ``route="allow"`` milestones into ``MEMORY.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _harness import resolve_vivemind_and_hermes_roots, run_clean_subprocess  # noqa: E402


def _resolve_roots() -> tuple[Path, Path]:
    return resolve_vivemind_and_hermes_roots()


def _run_clean_subprocess(
    script: str, *, vivemind_root: Path, hermes_root: Path, home: Path, result_file: Path | None = None,
) -> dict:
    return run_clean_subprocess(
        script, vivemind_root=vivemind_root, hermes_root=hermes_root, home=home, result_file=result_file,
    )


def test_e2e_cold_start_wire_and_three_dm_delivery_paths(tmp_path: Path) -> None:
    vivemind_root, hermes_root = _resolve_roots()
    home = tmp_path / "hermes_e2e"
    home.mkdir(parents=True)
    result_file = home / "wire-result.json"

    script = f"""
import asyncio
import json
import logging
import os
import shutil
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# 1. Cold-start invariant: agent.memory_manager MUST NOT be pre-imported.
assert "agent.memory_manager" not in sys.modules, "must start cold without agent.memory_manager"
os.environ["QQ_ALLOWED_USERS"] = "u-dm-1"

home = Path({str(home)!r})
vivemind_root = Path({str(vivemind_root)!r})
plugins_dir = home / "plugins"
plugins_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(vivemind_root / "plugins" / "sanae_mem", plugins_dir / "sanae_mem")

# 2. Outermost HTTP LLM server stub (main brain + auxiliary Flash).
wire_requests = []
flash_requests = []
commitment_id_holder = {{"id": None}}
on_main_request_hook = {{"fn": None}}
hold_background_for = {{"marker": ""}}
background_entered = threading.Event()
background_release = threading.Event()

class _LLMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        model = str(body.get("model") or "")
        messages = body.get("messages") or body.get("input") or []
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    last_user = c
                elif isinstance(c, list):
                    last_user = "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
                else:
                    last_user = json.dumps(c, ensure_ascii=False)
                break
        sys_text = str(body.get("instructions") or "")
        for m in messages:
            if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                c = m.get("content")
                sys_text += (c if isinstance(c, str) else "")

        if model == "test-flash-model":
            flash_requests.append(body)
            marker = hold_background_for["marker"]
            if marker and marker in last_user:
                background_entered.set()
                background_release.wait(15)
            if "【本轮群成员消息】" in last_user:
                quote = last_user.split("【本轮群成员消息】", 1)[1].strip().splitlines()[0]
                reply_obj = {{"events": [{{"text": quote, "quote": quote}}]}}
            elif last_user.lstrip().startswith('{{"previous":') or "state_variables" in sys_text or "variables" in sys_text:
                reply_obj = {{
                    "sentences": ["我现在心情挺安稳的。", "想把那份书单整理好。"],
                    "variables": {{"tension": 0.1, "closeness": 0.4, "energy": -0.1}},
                    "shared_sentences": None,
                }}
            elif "【本轮用户消息】" in last_user or "DAILY_FACT" in sys_text:
                if "今天下午我要去图书馆看书" in last_user:
                    reply_obj = {{
                        "items": [
                            {{"kind": "DAILY_FACT", "action": "set", "target_slot": "", "event_key": "lib", "text": "他今天下午要去图书馆看书", "quote": "今天下午我要去图书馆看书", "uncertain": False, "plan": False}},
                            {{"kind": "OPEN_THREAD", "action": "set", "target_slot": "", "event_key": "book_chat", "text": "想回头问问他今晚打算看哪本书", "quote": "晚上再聊", "uncertain": False, "plan": False}},
                        ]
                    }}
                else:
                    reply_obj = {{"items": []}}
            elif "【本轮用户消息（当前对话者原话）】" in last_user or "new_commitments" in sys_text:
                if "书单我收到啦" in last_user and commitment_id_holder["id"] is not None:
                    reply_obj = {{
                        "new_commitments": [],
                        "resolutions": [{{"id": commitment_id_holder["id"], "action": "fulfilled", "evidence": "对方确认收到书单", "tool_call_id": "", "user_quote": "书单我收到啦"}}],
                    }}
                elif "我答应你明天上午把书单发给你" in last_user:
                    reply_obj = {{
                        "new_commitments": [{{"text": "明天上午把书单发给他", "quote": "我答应你明天上午把书单发给你", "masked_text": "帮他整理一份书单"}}],
                        "resolutions": [],
                    }}
                else:
                    reply_obj = {{"new_commitments": [], "resolutions": []}}
            elif "【事件原文】" in last_user:
                reply_obj = {{"summary": "他下午要去图书馆看书", "shareable": True}}
            else:
                reply_obj = {{
                    "events": [{{"summary": "他今天下午要去图书馆看书，约定晚上聊书单", "importance": 0.6, "kind": "episodic"}}],
                    "facts": [],
                }}
            content = json.dumps(reply_obj, ensure_ascii=False)
        else:
            wire_requests.append(body)
            hook_fn = on_main_request_hook.get("fn")
            if callable(hook_fn):
                hook_fn(last_user)
            if "BUILTIN_GUARD_INPUT_" in last_user:
                content = "收到这条合成测试消息。"
            elif "书单我收到啦" in last_user:
                content = "太好啦，收到就好，慢慢看。"
            elif "排队补发" in last_user or "热可可" in last_user:
                content = "热可可暖暖的，很舒服呀。"
            else:
                content = "好呀，我答应你明天上午把书单发给你。"

        if body.get("stream"):
            chunk1 = json.dumps({{
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [{{"index": 0, "delta": {{"role": "assistant", "content": content}}, "finish_reason": None}}],
            }}, ensure_ascii=False)
            chunk2 = json.dumps({{
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "choices": [{{"index": 0, "delta": {{}}, "finish_reason": "stop"}}],
                "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
            }}, ensure_ascii=False)
            sse = f"data: {{chunk1}}\\n\\ndata: {{chunk2}}\\n\\ndata: [DONE]\\n\\n".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)
            return

        resp = json.dumps({{
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{{"index": 0, "message": {{"role": "assistant", "content": content}}, "finish_reason": "stop"}}],
            "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
        }}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

server = ThreadingHTTPServer(("127.0.0.1", 0), _LLMHandler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()

base_url = f"http://127.0.0.1:{{port}}/v1"
(home / "config.yaml").write_text(
    yaml.safe_dump({{
        "model": {{
            "default": "test-main-model",
            "provider": "custom",
            "base_url": base_url,
            "api_key": "dummy-key",
        }},
        "auxiliary": {{
            "sanae_daily_flash": {{"provider": "custom", "model": "test-flash-model", "base_url": base_url, "api_key": "dummy-key"}},
            "sanae_extract": {{"provider": "custom", "model": "test-flash-model", "base_url": base_url, "api_key": "dummy-key"}},
        }},
        "gateway": {{"delivery_ledger": True}},
        "plugins": {{
            "enabled": ["sanae_mem"],
            "sanae_mem": {{
                "data_dir": str(home / "sanae-mem"),
                "semantic_backend": "sqlite",
                "memory_identity": {{"platform": "qqbot", "user_id": "u-dm-1"}},
                "group_context": {{"defaults": {{"long_term_memory": True}}}},
            }},
        }},
    }}, allow_unicode=True),
    encoding="utf-8",
)

from gateway.config import Platform, PlatformConfig
from gateway.delivery_ledger import mark_delivered, sweep_recoverable
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.plugins import discover_plugins
from hermes_host_contracts import emit_delivery_receipt, register_delivery_receipt_handler
from sanae_mem.cognitive.inner_wiring import wait_for_pending_updates

# Discover plugins cold (gateway startup order)
discover_plugins()

receipts = []
register_delivery_receipt_handler(receipts.append)

class _StubQQAdapter(BasePlatformAdapter):
    def __init__(self, runner):
        super().__init__(PlatformConfig(enabled=True), Platform.QQBOT)
        self.gateway_runner = runner
        self.fail_next = False
        self.sent = []
        self._msg_seq = 0

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id: str):
        return {{"name": chat_id, "type": "dm"}}

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        self._msg_seq += 1
        if self.fail_next:
            return SendResult(success=False, error="network_timeout")
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id=f"out-{{self._msg_seq}}")

runner = GatewayRunner()
adapter = _StubQQAdapter(runner)
runner.adapters[Platform.QQBOT] = adapter
adapter._message_handler = runner._handle_message

async def _ok_adapter(row):
    return adapter
runner._obligation_adapter = _ok_adapter

session_key = "agent:main:qqbot:dm:u-dm-1"
source = SessionSource(
    platform=Platform.QQBOT,
    chat_id="u-dm-1",
    chat_type="dm",
    user_id="u-dm-1",
    message_id="in-1",
)

def _table_counts():
    db = home / "sanae-mem" / "sanae.db"
    if not db.is_file():
        return {{"events": 0, "daily_context": 0, "commitments": 0, "inner_state_revisions": 0}}
    with sqlite3.connect(str(db)) as conn:
        out = {{}}
        for tbl in ("events", "daily_context", "commitments", "inner_state_revisions"):
            out[tbl] = conn.execute(f"SELECT COUNT(*) FROM {{tbl}}").fetchone()[0]
        return out

def _wait_all_inner():
    for mod in list(sys.modules.values()):
        fn = getattr(mod, "wait_for_pending_updates", None) if mod is not None else None
        if callable(fn):
            fn(timeout=10.0)

async def _run_all():
    # Step A: Turn 1 with platform send FAILURE -> delivery_ledger, ZERO DB writes.
    adapter.fail_next = True
    ev1 = MessageEvent(
        text="今天下午我要去图书馆看书，晚上再聊。",
        source=source,
        message_id="in-1",
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(ev1, session_key)
    _wait_all_inner()
    counts_after_fail = _table_counts()
    receipts_after_fail = len(receipts)

    # Step B: Startup delivery_ledger redelivery succeeds (Path 1).
    adapter.fail_next = False
    with sqlite3.connect(str(home / "state.db")) as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid = 999999999, owner_started_at = 1")
        conn.commit()
    claimed = sweep_recoverable()
    redelivered = await runner._redeliver_claimed_obligations(claimed)
    second_mark = mark_delivered(claimed[0]["obligation_id"])
    _wait_all_inner()
    counts_after_redeliver = _table_counts()
    receipts_after_redeliver = len(receipts)

    # Capture pending commitment id for Turn 2 fulfillment
    with sqlite3.connect(str(home / "sanae-mem" / "sanae.db")) as conn:
        row = conn.execute("SELECT id, status FROM commitments ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            commitment_id_holder["id"] = row[0]

    # Step C: Turn 2 normal DM direct send succeeds (Path 2) + fulfills commitment.
    src2 = SessionSource(
        platform=Platform.QQBOT,
        chat_id="u-dm-1",
        chat_type="dm",
        user_id="u-dm-1",
        message_id="in-2",
    )
    ev2 = MessageEvent(
        text="书单我收到啦，晚上再聊。",
        source=src2,
        message_id="in-2",
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(ev2, session_key)
    _wait_all_inner()
    counts_after_turn2 = _table_counts()
    receipts_after_turn2 = len(receipts)

    # Step D: DM /queue queued reply via real GatewayRunner._run_agent_queued_followup -> _send_queued_final_text (Path 3).
    src3 = SessionSource(
        platform=Platform.QQBOT,
        chat_id="u-dm-1",
        chat_type="dm",
        user_id="u-dm-1",
        message_id="in-3",
    )
    ev3 = MessageEvent(text="我刚才去楼下买了一杯热可可，顺便取了个快递。", source=src3, message_id="in-3")
    src4 = SessionSource(
        platform=Platform.QQBOT,
        chat_id="u-dm-1",
        chat_type="dm",
        user_id="u-dm-1",
        message_id="in-4",
    )
    ev4 = MessageEvent(text="排队补发测试：我打算看刚买的新杂志。", source=src4, message_id="in-4")

    def _queue_while_in_flight(_user_text):
        on_main_request_hook["fn"] = None
        adapter._pending_messages[session_key] = ev4

    on_main_request_hook["fn"] = _queue_while_in_flight
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(ev3, session_key)
    # Give any spawned drain task a chance to finish if _process_message_background handed off.
    for _ in range(40):
        if len(receipts) >= 4:
            break
        await asyncio.sleep(0.1)
    _wait_all_inner()
    counts_after_queue = _table_counts()
    receipts_after_queue = len(receipts)
    return (
        counts_after_fail, receipts_after_fail,
        redelivered, second_mark, counts_after_redeliver, receipts_after_redeliver,
        counts_after_turn2, receipts_after_turn2,
        counts_after_queue, receipts_after_queue,
    )

(
    counts_after_fail, receipts_after_fail,
    redelivered, second_mark, counts_after_redeliver, receipts_after_redeliver,
    counts_after_turn2, receipts_after_turn2,
    counts_after_queue, receipts_after_queue,
) = asyncio.run(_run_all())

# Inspect wire requests (Turn 1 and Turn 2) & state.db api_content
def _extract_user_texts(req):
    items = req.get("messages") or req.get("input") or []
    out = []
    for m in items:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                out.append("".join(str(p.get("text") or "") for p in c if isinstance(p, dict)))
            else:
                out.append(json.dumps(c, ensure_ascii=False))
    return out

main_wire = [r for r in wire_requests if _extract_user_texts(r)]
assert len(main_wire) >= 2, f"expected >=2 main wire requests with user msgs, got {{len(main_wire)}}"
turn1_reqs = [r for r in main_wire if any("今天下午我要去图书馆看书" in u for u in _extract_user_texts(r))]
turn2_reqs = [r for r in main_wire if any("书单我收到啦" in u for u in _extract_user_texts(r))]
assert len(turn1_reqs) >= 1 and len(turn2_reqs) >= 1, f"missing turn reqs: main_wire_count={{len(main_wire)}}, users={{[_extract_user_texts(r)[-1][:40] for r in main_wire]}}"
turn1_user_wire = _extract_user_texts(turn1_reqs[-1])[-1]
turn2_user_msgs = _extract_user_texts(turn2_reqs[-1])
turn2_hist_turn1_user = turn2_user_msgs[0]
turn2_curr_user_wire = turn2_user_msgs[-1]

with sqlite3.connect(str(home / "state.db")) as conn:
    api_contents = [r[0] or "" for r in conn.execute("SELECT api_content FROM messages").fetchall()]
    persisted_contents = [r[0] or "" for r in conn.execute("SELECT content FROM messages").fetchall()]

with sqlite3.connect(str(home / "sanae-mem" / "sanae.db")) as conn:
    commit_statuses = [r[0] for r in conn.execute("SELECT status FROM commitments").fetchall()]
    daily_kinds = [r[0] for r in conn.execute("SELECT kind FROM daily_context").fetchall()]
    for incoming_id, text in (
        ("in-1", "今天下午我要去图书馆看书，晚上再聊。"),
        ("in-4", "排队补发测试：我打算看刚买的新杂志。"),
    ):
        rows = conn.execute(
            "SELECT source_snapshot FROM events WHERE json_extract(source_context, '$.ref')=?",
            (incoming_id,),
        ).fetchall()
        assert len(rows) == 1, (incoming_id, rows)
        evidence = json.loads(rows[0][0])
        assert evidence["source_context"]["kind"] == "maintainer"
        assert evidence["turn"] == incoming_id
        assert evidence["source_messages"] == [{{"role": "user", "content": text}}]

def change_shadow(shadow):
    path = home / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["plugins"]["sanae_mem"]["shadow"] = shadow
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

async def deliver_probe(scene, message_id, marker):
    chat = "u-dm-1" if scene == "dm" else "group-shadow"
    probe_source = SessionSource(
        platform=Platform.QQBOT, chat_id=chat, chat_type=scene, user_id="u-dm-1",
        message_id=message_id, scope_id="default",
    )
    event = MessageEvent(text=marker, source=probe_source, message_id=message_id, reply_expected=True)
    key = f"agent:main:qqbot:{{scene}}:{{chat}}"
    adapter._active_sessions[key] = asyncio.Event()
    await adapter._process_message_background(event, key)
    receipt = next(receipt for receipt in reversed(receipts) if receipt.incoming_message_id == message_id)
    assert receipt.chat_type == scene and receipt.transport_confirmed
    return receipt

async def check_shadow_between_enqueue_and_execution(scene):
    blocker_marker = f"今早在图书馆遇到朋友，{{scene}}后台占位。"
    queued_marker = f"刚搬到新住处，{{scene}}排队时关闭写入。"
    resumed_marker = f"重新开始整理书单，{{scene}}恢复送达。"
    change_shadow(True)
    before = len(flash_requests)
    blocker = await deliver_probe(scene, f"shadow-{{scene}}-1", blocker_marker)
    queued = await deliver_probe(scene, f"shadow-{{scene}}-2", queued_marker)
    _wait_all_inner()
    assert len(flash_requests) == before

    background_entered.clear()
    background_release.clear()
    hold_background_for["marker"] = blocker_marker
    change_shadow(False)
    try:
        emit_delivery_receipt(blocker)
        assert background_entered.wait(10), (scene, "the first delivered turn did not reach the model")
        emit_delivery_receipt(queued)
        change_shadow(True)
    finally:
        background_release.set()
        _wait_all_inner()
        hold_background_for["marker"] = ""
    assert not any(queued_marker in json.dumps(request, ensure_ascii=False)
                   for request in flash_requests[before:]), (scene, "queued turn reached the model in shadow")

    change_shadow(False)
    resumed = await deliver_probe(scene, f"shadow-{{scene}}-3", resumed_marker)
    _wait_all_inner()
    assert any(resumed_marker in json.dumps(request, ensure_ascii=False)
               for request in flash_requests[before:]), (scene, "did not resume settlement")

    requests_after_resume = len(flash_requests)
    emit_delivery_receipt(resumed)
    _wait_all_inner()
    assert len(flash_requests) == requests_after_resume, (scene, "duplicate settlement")
    with sqlite3.connect(home / "sanae-mem" / "sanae.db") as conn:
        if scene == "dm":
            assert not conn.execute("SELECT 1 FROM inner_state_revisions WHERE turn_id=?", (queued.turn_id,)).fetchone()
            assert conn.execute("SELECT 1 FROM inner_state_revisions WHERE turn_id=?", (resumed.turn_id,)).fetchone()
        else:
            assert conn.execute("SELECT summary FROM events WHERE summary IN (?, ?)",
                                (queued_marker, resumed_marker)).fetchall() == [(resumed_marker,)]

asyncio.run(check_shadow_between_enqueue_and_execution("dm"))
asyncio.run(check_shadow_between_enqueue_and_execution("group"))

async def check_builtin_notes_on_the_gateway_wire(result):
    from gateway.config import GatewayConfig
    from gateway.platform_registry import PlatformEntry, platform_registry
    from sanae_mem.governance.memory_guard import PUBLIC_MEMORY_PROMPT_BLOCK_MESSAGE

    markers = ("SYNTHETIC_OPERATOR_MEMORY_431", "SYNTHETIC_OPERATOR_PROFILE_862")
    notes = home / "memories"
    notes.mkdir(exist_ok=True)
    for filename, marker in zip(("MEMORY.md", "USER.md"), markers):
        (notes / filename).write_text(marker, encoding="utf-8")
    path = home / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.setdefault("memory", {{}}).update(memory_enabled=True, user_profile_enabled=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    os.environ["QQ_ALLOWED_USERS"] = "u-dm-1,u-group-guest"
    platform_scope = platform_registry.current_scope_key()
    platform_registry.register(PlatformEntry(
        name=Platform.QQBOT.value, label="QQ test transport",
        adapter_factory=lambda config: _StubQQAdapter(None), check_fn=lambda: True,
        allowed_users_env="QQ_ALLOWED_USERS",
    ), scope=platform_scope)
    gateway = GatewayRunner(GatewayConfig(
        platforms={{Platform.QQBOT: PlatformConfig(enabled=True)}}, multiplex_profiles=False,
    ))
    try:
        assert await gateway.start()
        transport = gateway.adapters[Platform.QQBOT]

        async def send_probe(label, *, scene="group", blocked=False):
            chat = "builtin-guard-group" if scene == "group" else "u-dm-1"
            sender = "u-group-guest" if scene == "group" else "u-dm-1"
            message_id = f"builtin-guard-{{label}}"
            text = f"BUILTIN_GUARD_INPUT_{{label}}"
            source = SessionSource(
                platform=Platform.QQBOT, chat_id=chat, chat_type=scene, user_id=sender,
                message_id=message_id, scope_id="default",
            )
            if scene == "dm":
                await gateway.async_session_store.get_or_create_session(source, force_new=True)
            sent_before, requests_before, flash_before = len(transport.sent), len(wire_requests), len(flash_requests)
            await transport.handle_message(MessageEvent(
                text=text, source=source, message_id=message_id, reply_expected=True,
            ))
            expected = PUBLIC_MEMORY_PROMPT_BLOCK_MESSAGE if blocked else "收到这条合成测试消息。"
            for _ in range(240):
                if any(peer == chat and expected in reply for peer, reply in transport.sent[sent_before:]):
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError((label, "expected gateway reply not sent", transport.sent[sent_before:]))
            requests = wire_requests[requests_before:]
            carrying_input = [request for request in requests
                              if any(text in content for content in _extract_user_texts(request))]
            if blocked:
                assert not carrying_input and not any(request.get("stream") for request in requests), (
                    label, "blocked group turn reached the model",
                )
                assert not any(text in item for request in flash_requests[flash_before:]
                               for item in _extract_user_texts(request)), label
                return
            matching = [request for request in carrying_input if request.get("stream")]
            assert matching, (label, "no model request carried the current inbound message")
            for request in matching:
                system = "\\n".join([
                    str(request.get("instructions") or ""),
                    *(str(message.get("content") or "") for message in request.get("messages", [])
                      if message.get("role") in ("system", "developer")),
                ])
                present = tuple(marker in system for marker in markers)
                assert present == (scene == "dm",) * len(markers), (label, present)

        await send_probe("normal")
        change_shadow(True)
        await send_probe("shadow", blocked=True)
        change_shadow(False)
        await send_probe("restored")
        await send_probe("maintainer", scene="dm")
        Path({str(result_file)!r}).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logging.exception("Built-in notes gateway probe failed")
        raise
    finally:
        await gateway.stop()
        platform_registry.unregister(Platform.QQBOT.value, scope=platform_scope)

result = {{
    "counts_after_fail": counts_after_fail,
    "receipts_after_fail": receipts_after_fail,
    "redelivered": redelivered,
    "second_mark": second_mark,
    "counts_after_redeliver": counts_after_redeliver,
    "receipts_after_redeliver": receipts_after_redeliver,
    "counts_after_turn2": counts_after_turn2,
    "receipts_after_turn2": receipts_after_turn2,
    "counts_after_queue": counts_after_queue,
    "receipts_after_queue": receipts_after_queue,
    "turn1_has_world": "[我眼中的世界]" in turn1_user_wire,
    "turn1_has_heart": "[我内心的感受]" in turn1_user_wire,
    "turn2_has_world": "[我眼中的世界]" in turn2_curr_user_wire,
    "turn2_has_heart": "[我内心的感受]" in turn2_curr_user_wire,
    "turn2_hist_stale_world": "[我眼中的世界]" in turn2_hist_turn1_user,
    "turn2_hist_stale_heart": "[我内心的感受]" in turn2_hist_turn1_user,
    "state_db_stale_world": any("[我眼中的世界]" in c for c in api_contents + persisted_contents),
    "state_db_stale_heart": any("[我内心的感受]" in c for c in api_contents + persisted_contents),
    "commit_statuses": commit_statuses,
    "daily_kinds": daily_kinds,
    "flash_req_count": len(flash_requests),
    "flash_snippets": [str((r.get("messages") or [{{}}])[-1].get("content") or "")[:60] for r in flash_requests],
    "status_json": json.loads((home / "sanae-mem" / "status.json").read_text(encoding="utf-8")) if (home / "sanae-mem" / "status.json").is_file() else None,
    "receipt_summaries": [
        {{
            "session_id": r.session_id,
            "turn_id": r.turn_id,
            "platform": r.platform,
            "chat_id": r.chat_id,
            "chat_type": r.chat_type,
            "sender_id": r.sender_id,
            "user_message": r.user_message[:30],
            "clean_text": r.clean_text[:30],
            "transport_confirmed": r.transport_confirmed,
            "is_cron": r.is_cron,
        }}
        for r in receipts
    ],
}}
asyncio.run(check_builtin_notes_on_the_gateway_wire(result))
server.shutdown()
"""
    res = _run_clean_subprocess(
        script, vivemind_root=vivemind_root, hermes_root=hermes_root, home=home, result_file=result_file,
    )

    # 1. Wire & state.db invariants
    assert res["turn1_has_world"] is True
    assert res["turn1_has_heart"] is True
    assert res["turn2_has_world"] is True
    assert res["turn2_has_heart"] is True
    assert res["turn2_hist_stale_world"] is False
    assert res["turn2_hist_stale_heart"] is False
    assert res["state_db_stale_world"] is False
    assert res["state_db_stale_heart"] is False

    # 2. Zero writes before delivery receipt
    assert res["receipts_after_fail"] == 0
    assert res["counts_after_fail"] == {
        "events": 0,
        "daily_context": 0,
        "commitments": 0,
        "inner_state_revisions": 0,
    }

    # 3. Redelivery + idempotency
    assert res["redelivered"] == 1
    assert res["second_mark"] is False
    assert res["receipts_after_redeliver"] == 1
    assert res["counts_after_redeliver"]["events"] >= 1, f"res={res}"
    assert res["counts_after_redeliver"]["daily_context"] >= 1, f"res={res}"
    assert res["counts_after_redeliver"]["commitments"] >= 1, f"res={res}"
    assert res["counts_after_redeliver"]["inner_state_revisions"] >= 1, f"res={res}"

    # 4. Turn 2 direct send + commitment resolution
    assert res["receipts_after_turn2"] == 2
    assert res["counts_after_turn2"]["events"] > res["counts_after_redeliver"]["events"]
    assert "fulfilled" in res["commit_statuses"] or "resolved" in res["commit_statuses"]
    assert any(k in ("DAILY_FACT", "OPEN_THREAD") for k in res["daily_kinds"])

    # 5. DM /queue queued reply (via real _run_agent_queued_followup -> _send_queued_final_text)
    assert res["receipts_after_queue"] >= 3
    assert res["counts_after_queue"]["inner_state_revisions"] > res["counts_after_turn2"]["inner_state_revisions"], (
        f"turn2={res['counts_after_turn2']}, queue={res['counts_after_queue']}"
    )


def test_e2e_quote_evidence_and_resident_memory_routing_subprocess(tmp_path: Path) -> None:
    vivemind_root, hermes_root = _resolve_roots()
    home = tmp_path / "hermes_guard"
    home.mkdir(parents=True)

    script = f"""
import json
import shutil
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

assert "agent.memory_manager" not in sys.modules, "must start cold without agent.memory_manager"

home = Path({str(home)!r})
vivemind_root = Path({str(vivemind_root)!r})
plugins_dir = home / "plugins"
plugins_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(vivemind_root / "plugins" / "sanae_mem", plugins_dir / "sanae_mem")

class _RouteHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = body.get("messages") or []
        last_user = str(messages[-1].get("content") or "") if messages else ""
        if "检讨自己" in last_user:
            payload = {{"route": "inner_state", "reason": "self_critique"}}
        elif "冰美式" in last_user:
            payload = {{"route": "inner_state", "inner_thought": "", "event_fact": "今天中午去楼下买了杯冰美式。", "reason": "single_trivia"}}
        else:
            payload = {{"route": "allow", "reason": "cross_period_milestone"}}
        resp = json.dumps({{
            "id": "chatcmpl-guard",
            "object": "chat.completion",
            "choices": [{{"index": 0, "message": {{"role": "assistant", "content": json.dumps(payload)}}, "finish_reason": "stop"}}],
            "usage": {{"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}},
        }}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

server = ThreadingHTTPServer(("127.0.0.1", 0), _RouteHandler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{{port}}/v1"

(home / "config.yaml").write_text(
    yaml.safe_dump({{
        "memory": {{"memory_enabled": True}},
        "auxiliary": {{
            "sanae_daily_flash": {{"provider": "custom", "model": "test-flash-model", "base_url": base_url, "api_key": "dummy-key"}},
            "sanae_extract": {{"provider": "custom", "model": "test-flash-model", "base_url": base_url, "api_key": "dummy-key"}},
        }},
        "plugins": {{
            "enabled": ["sanae_mem"],
            "sanae_mem": {{
                "data_dir": str(home / "sanae-mem"),
                "semantic_backend": "sqlite",
                "memory_identity": {{"platform": "qqbot", "user_id": "u-guard-1"}},
            }},
        }},
    }}, allow_unicode=True),
    encoding="utf-8",
)

from hermes_cli.plugins import discover_plugins
from hermes_host_contracts import TurnEnvelope, turn_envelope_scope
from sanae_mem import SanaeMemProvider
from sanae_mem.governance.memory_evidence import capture_private_turn, qualify_user_text
from sanae_mem.governance.source_context import current_chat_scope, resolve_chat_identity
from tools.memory_tool import MemoryStore, memory_tool

discover_plugins()

# Part 1: Multi-sentence quote evidence gate
cfg = {{"data_dir": str(home / "sanae-mem"), "semantic_backend": "sqlite", "memory_identity": {{"platform": "qqbot", "user_id": "u-guard-1"}}}}
provider = SanaeMemProvider(cfg)
env_q1 = TurnEnvelope(
    platform="qqbot", chat_type="dm", chat_id="u-guard-1", sender_id="u-guard-1",
    scope_id="default", session_id="s-quote-1", session_key="agent:main:qqbot:dm:u-guard-1",
    message_id="m-q1", turn_id="t-q1", is_messaging_surface=True, write_origin="foreground",
)
with turn_envelope_scope(env_q1):
    provider.initialize("s-quote-1", hermes_home=str(home), platform="qqbot", agent_context="primary")
    scope = current_chat_scope()
    identity = provider.sqlite.read(lambda conn: resolve_chat_identity(scope, cfg, connection=conn))
    reply_msg = '[Replying to your previous message: "句1。其实我最喜欢抹茶拿铁。句3"]\\n\\n好的，我记下了。我平时习惯喝美式'
    ev_reply = capture_private_turn(
        reply_msg,
        current_message={{"role": "user", "content": reply_msg, "platform_message_id": "m-q1"}},
        session_id="s-quote-1", platform="qqbot", scope=scope, identity=identity,
        author_id="u-guard-1",
    )
    v_reply_quoted = qualify_user_text("其实我最喜欢抹茶拿铁", ev_reply)
    v_reply_real = qualify_user_text("我平时习惯喝美式", ev_reply)

    quoted_body = "句1。其实我最喜欢乌龙茶。句3"
    qq_msg = f"[Quoted message; length={{len(quoted_body)}}]:\\n{{quoted_body}}\\n\\n真的吗，我周末常去图书馆"
    ev_qq = capture_private_turn(
        qq_msg,
        current_message={{"role": "user", "content": qq_msg, "platform_message_id": "m-q2"}},
        session_id="s-quote-1", platform="qqbot", scope=scope, identity=identity,
        author_id="u-guard-1",
    )
    v_qq_quoted = qualify_user_text("其实我最喜欢乌龙茶", ev_qq)
    v_qq_real = qualify_user_text("我周末常去图书馆", ev_qq)
provider.shutdown()

# Part 2: Resident MEMORY.md routing via real memory_tool
memories_dir = home / "memories"
memories_dir.mkdir(parents=True, exist_ok=True)
store = MemoryStore()
store.load_from_disk()
mem_file = memories_dir / "MEMORY.md"

env = TurnEnvelope(
    platform="qqbot",
    chat_type="dm",
    chat_id="u-guard-1",
    sender_id="u-guard-1",
    session_id="sess-guard-1",
    session_key="agent:main:qqbot:dm:u-guard-1",
    message_id="m-guard-1",
    turn_id="t-guard-1",
    is_messaging_surface=True,
    write_origin="foreground",
)
with turn_envelope_scope(env):
    r_critique = json.loads(memory_tool(action="add", target="memory", content="我以后不能再这样粗心了，要检讨自己。", store=store))
    mem_after_critique = mem_file.read_text(encoding="utf-8") if mem_file.is_file() else ""

    r_trivia = json.loads(memory_tool(action="add", target="memory", content="今天中午去楼下买了杯冰美式。", store=store))
    mem_after_trivia = mem_file.read_text(encoding="utf-8") if mem_file.is_file() else ""

    r_allow = json.loads(memory_tool(action="add", target="memory", content="我们约定每年秋天都一起去京都看红叶。", store=store))
    mem_after_allow = mem_file.read_text(encoding="utf-8") if mem_file.is_file() else ""

with sqlite3.connect(str(home / "sanae-mem" / "sanae.db")) as conn:
    inner_cnt = conn.execute("SELECT COUNT(*) FROM inner_state_revisions").fetchone()[0]
    short_events = conn.execute("SELECT raw, ttl_expire FROM events WHERE ttl_expire IS NOT NULL AND ttl_expire != ''").fetchall()

server.shutdown()
print(json.dumps({{
    "reply_quoted": {{"allowed": v_reply_quoted.allowed, "status": v_reply_quoted.status}},
    "reply_real": {{"allowed": v_reply_real.allowed, "status": v_reply_real.status}},
    "qq_quoted": {{"allowed": v_qq_quoted.allowed, "status": v_qq_quoted.status}},
    "qq_real": {{"allowed": v_qq_real.allowed, "status": v_qq_real.status}},
    "mem_after_critique": mem_after_critique,
    "mem_after_trivia": mem_after_trivia,
    "mem_after_allow": mem_after_allow,
    "inner_cnt": inner_cnt,
    "short_event_count": len(short_events),
    "r_allow_success": r_allow.get("success"),
}}, ensure_ascii=False))
"""
    res = _run_clean_subprocess(script, vivemind_root=vivemind_root, hermes_root=hermes_root, home=home)

    assert res["reply_quoted"]["allowed"] is False
    assert res["reply_real"] == {"allowed": True, "status": "USER_EXTRACTED"}
    assert res["qq_quoted"]["allowed"] is False
    assert res["qq_real"] == {"allowed": True, "status": "USER_EXTRACTED"}

    assert res["mem_after_critique"] == ""
    assert res["mem_after_trivia"] == ""
    assert "我们约定每年秋天都一起去京都看红叶。" in res["mem_after_allow"]
    assert res["inner_cnt"] >= 1
    assert res["short_event_count"] >= 1
    assert res["r_allow_success"] is True
