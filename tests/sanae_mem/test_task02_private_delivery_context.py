"""Task 02 regression tests: plugin_user_context propagation and truthful settlement results."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import threading
from typing import Any

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _harness import install_plugin_copy, resolve_vivemind_and_hermes_roots, run_clean_subprocess  # noqa: E402
from sanae_mem.cognitive.daily_inputs import (
    MEME_CONTEXT_MARKER,
    SELF_UTTERANCE_MARKER,
    turn_context_block,
)
from sanae_mem.storage.database import open_database
from sanae_mem.domains.meme_context import context_for_turn
from sanae_mem.runtime.private_delivery_settler import TurnUserContextCache


def _resolve_roots() -> tuple[Path, Path]:
    return resolve_vivemind_and_hermes_roots()


def _run_clean_subprocess(
    script: str,
    *,
    vivemind_root: Path,
    hermes_root: Path,
    home: Path,
) -> dict[str, Any]:
    install_plugin_copy(home, vivemind_root)
    return run_clean_subprocess(
        script, vivemind_root=vivemind_root, hermes_root=hermes_root, home=home,
    )


def test_meme_context_and_turn_context_block_coexistence(tmp_path: Path) -> None:
    """[常玩梗短提醒] and [梗与表情记忆] coexist cleanly and remain reliably extractable."""
    conn = open_database(tmp_path / "sanae.db")
    try:
        now = datetime.now().astimezone().isoformat()
        conn.execute(
            """INSERT INTO meme_records
               (title, meaning, aliases, source, source_ref, source_time, created_at,
                updated_at, owner_key, session_id, tier, visibility_scope, evidence_status)
               VALUES (?, ?, ?, '', '', '', ?, ?, ?, '', 'common', 'private', 'confirmed')""",
            (
                "贴贴猫猫",
                "表达撒娇和想贴近的心情",
                json.dumps(["猫猫贴贴"], ensure_ascii=False),
                now,
                now,
                "owner-A",
            ),
        )
        conn.commit()

        rendered = context_for_turn(
            conn,
            "今天也要猫猫贴贴",
            shadow=False,
            public=False,
            owner_key="owner-A",
        )
        assert "[常玩梗短提醒]" in rendered
        assert MEME_CONTEXT_MARKER in rendered
        assert "\n\n" + MEME_CONTEXT_MARKER in rendered

        extracted = turn_context_block(rendered, MEME_CONTEXT_MARKER)
        assert extracted.startswith(MEME_CONTEXT_MARKER)
        assert "贴贴猫猫" in extracted
        assert "表达撒娇和想贴近的心情" in extracted
        assert "[常玩梗短提醒]" not in extracted

        # Backward compatibility: single-newline legacy concatenation still extracts accurately.
        single_newline_combined = (
            "[常玩梗短提醒] 贴贴猫猫：表达撒娇和想贴近的心情\n"
            "[梗与表情记忆] 内部参考：\n"
            "- 记录#1「贴贴猫猫」：表达撒娇和想贴近的心情\n"
            "[我刚才在别处说过]\n"
            "2026-11-19 09:00｜定时主动发言｜早上记得吃早饭。"
        )
        meme_part = turn_context_block(single_newline_combined, MEME_CONTEXT_MARKER)
        elsewhere_part = turn_context_block(single_newline_combined, SELF_UTTERANCE_MARKER)
        assert meme_part.startswith(MEME_CONTEXT_MARKER)
        assert "记录#1「贴贴猫猫」" in meme_part
        assert SELF_UTTERANCE_MARKER not in meme_part
        assert elsewhere_part.startswith(SELF_UTTERANCE_MARKER)
        assert "早上记得吃早饭。" in elsewhere_part
        assert MEME_CONTEXT_MARKER not in elsewhere_part
    finally:
        conn.close()


def test_turn_user_context_cache_isolation_merge_and_ttl(tmp_path: Path) -> None:
    """TurnUserContextCache merges blocks per (home, session, turn), isolates scopes, and expires by TTL."""
    cache = TurnUserContextCache(ttl_seconds=300.0)
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()

    self_block = (
        "[我刚才在别处说过]\n"
        "2026-11-19 08:30｜定时主动发言｜上午出门别忘带伞。"
    )
    meme_with_reminder = (
        "[常玩梗短提醒] 贴贴猫猫：表达撒娇和想贴近的心情\n\n"
        "[梗与表情记忆] 内部参考：\n"
        "- 记录#7「贴贴猫猫」：表达撒娇和想贴近的心情"
    )
    reminder_only = "[常玩梗短提醒] 贴贴猫猫：表达撒娇和想贴近的心情"

    # Reminder-only without matched [梗与表情记忆] must not populate cache.
    cache.record(home_a, "sess-1", "turn-empty", meme_context=reminder_only, now=1000.0)
    assert cache.read(home_a, "sess-1", "turn-empty", now=1001.0) == ""

    # Blank key parts never store or leak.
    cache.record(home_a, "sess-1", "", self_utterances=self_block, now=1000.0)
    assert cache.read(home_a, "sess-1", "", now=1001.0) == ""

    # Order-independent merge of self_utterances + meme_context.
    cache.record(home_a, "sess-1", "turn-1", meme_context=meme_with_reminder, now=1000.0)
    cache.record(home_a, "sess-1", "turn-1", self_utterances=self_block, now=1005.0)
    merged = cache.read(home_a, "sess-1", "turn-1", now=1010.0)
    assert turn_context_block(merged, SELF_UTTERANCE_MARKER) == self_block
    assert "贴贴猫猫" in turn_context_block(merged, MEME_CONTEXT_MARKER)
    assert "[常玩梗短提醒]" not in merged

    # Strict isolation across profile home, session_id, and turn_id.
    assert cache.read(home_b, "sess-1", "turn-1", now=1010.0) == ""
    assert cache.read(home_a, "sess-2", "turn-1", now=1010.0) == ""
    assert cache.read(home_a, "sess-1", "turn-2", now=1010.0) == ""

    # TTL eviction at 300s boundary.
    assert cache.read(home_a, "sess-1", "turn-1", now=1305.0) != ""
    assert cache.read(home_a, "sess-1", "turn-1", now=1306.0) == ""

    # Concurrent multi-thread writes and reads stay isolated and error-free.
    errors: list[BaseException] = []

    def _worker(idx: int) -> None:
        try:
            turn = f"turn-concurrent-{idx}"
            s_text = f"[我刚才在别处说过]\n2026-11-19 09:00｜定时主动发言｜并发消息-{idx}"
            m_text = f"[梗与表情记忆] 内部参考：\n- 记录#{idx}「梗-{idx}」：含义-{idx}"
            cache.record(home_a, f"sess-{idx}", turn, self_utterances=s_text, now=2000.0)
            cache.record(home_a, f"sess-{idx}", turn, meme_context=m_text, now=2000.0)
            got = cache.read(home_a, f"sess-{idx}", turn, now=2001.0)
            assert f"并发消息-{idx}" in got
            assert f"含义-{idx}" in got
        except BaseException as exc:  # pragma: no cover - asserted empty below
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_e2e_dual_load_private_delivery_propagates_elsewhere_and_meme_context_subprocess(
    tmp_path: Path,
) -> None:
    """Real dual-load cold-start (discover_plugins + load_memory_provider) emits exactly one
    Step 2 request per turn carrying [我刚才在别处说过] and [梗与表情记忆], supports FIFO
    multi-turn same-clean_reply omitted-turn_id receipts without cross-talk, honors explicit
    receipt.plugin_user_context, and isolates cross-session / cross-profile turns.
    """
    vivemind_root, hermes_root = _resolve_roots()
    home = tmp_path / "hermes_task02_ctx"
    other_home = tmp_path / "hermes_task02_other"
    home.mkdir(parents=True)
    other_home.mkdir(parents=True)

    script = f"""
from dataclasses import dataclass as _dataclass
import json
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

assert "agent.memory_manager" not in sys.modules, "must start cold without agent.memory_manager"

home = Path({str(home)!r})
other_home = Path({str(other_home)!r})

daily_prompts = []
flash_lock = threading.Lock()


class _FlashHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = body.get("messages") or []
        sys_text = ""
        user_text = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "")
            if role in ("system", "developer"):
                sys_text += content
            elif role == "user":
                user_text = content

        if "DAILY_FACT" in sys_text and "【本轮用户消息】" in user_text:
            with flash_lock:
                daily_prompts.append(user_text)
            if "下午去图书馆" in user_text:
                payload = {{
                    "items": [
                        {{
                            "kind": "DAILY_FACT",
                            "action": "set",
                            "target_slot": "",
                            "event_key": "lib_t1",
                            "text": "他下午要去图书馆",
                            "quote": "下午去图书馆",
                            "uncertain": False,
                            "plan": False,
                        }}
                    ]
                }}
            elif "晚上去吃拉面" in user_text:
                payload = {{
                    "items": [
                        {{
                            "kind": "DAILY_FACT",
                            "action": "set",
                            "target_slot": "",
                            "event_key": "ramen_t2",
                            "text": "他晚上要去吃拉面",
                            "quote": "晚上去吃拉面",
                            "uncertain": False,
                            "plan": False,
                        }}
                    ]
                }}
            elif "明早坐高铁出差" in user_text:
                payload = {{
                    "items": [
                        {{
                            "kind": "DAILY_FACT",
                            "action": "set",
                            "target_slot": "",
                            "event_key": "train_t3",
                            "text": "他明早坐高铁出差",
                            "quote": "明早坐高铁出差",
                            "uncertain": False,
                            "plan": False,
                        }}
                    ]
                }}
            else:
                payload = {{"items": []}}
        elif "new_commitments" in sys_text:
            payload = {{"new_commitments": [], "resolutions": []}}
        else:
            payload = {{
                "sentences": ["记着他今天的行程，心里踏实了不少。", "一会儿再关心一下他。"],
                "variables": {{"tension": 0.1, "closeness": 0.6, "energy": 0.1}},
                "shared_sentences": None,
            }}

        resp = json.dumps({{
            "id": "chatcmpl-task02",
            "object": "chat.completion",
            "choices": [{{
                "index": 0,
                "message": {{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}},
                "finish_reason": "stop",
            }}],
            "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
        }}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


server = ThreadingHTTPServer(("127.0.0.1", 0), _FlashHandler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{{port}}/v1"

(home / "config.yaml").write_text(
    yaml.safe_dump({{
        "memory": {{"provider": "sanae_mem"}},
        "auxiliary": {{
            "sanae_daily_flash": {{
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": base_url,
                "api_key": "dummy-key",
            }},
            "sanae_extract": {{
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": base_url,
                "api_key": "dummy-key",
            }},
        }},
        "plugins": {{
            "enabled": ["sanae_mem"],
            "sanae_mem": {{
                "data_dir": str(home / "sanae-mem"),
                "semantic_backend": "sqlite",
                "memory_identity": {{"platform": "telegram", "user_id": "1001"}},
            }},
        }},
    }}, allow_unicode=True),
    encoding="utf-8",
)

from hermes_cli.plugins import discover_plugins, get_plugin_manager
from hermes_host_contracts import (
    DeliveryReceipt,
    TurnEnvelope,
    emit_delivery_receipt,
    record_verified_cron_send,
    turn_envelope_scope,
)
import plugins.memory as host_memory

# Cold dual-load: discover_plugins() (M1) + load_memory_provider() (M2)
discover_plugins()
manager = get_plugin_manager()
provider = host_memory.load_memory_provider("sanae_mem")
assert provider is not None

env_t1 = TurnEnvelope(
    platform="telegram",
    chat_type="dm",
    chat_id="1001",
    sender_id="1001",
    session_id="dm-s1",
    session_key="agent:main:telegram:dm:1001",
    message_id="in-1",
    turn_id="turn-1",
    is_messaging_surface=True,
    write_origin="foreground",
)

with turn_envelope_scope(env_t1):
    provider.initialize("dm-s1", platform="telegram", hermes_home=str(home), agent_context="primary")
    for title, alias, meaning in (
        ("摸鱼小猫", "猫猫摸鱼", "用来打趣对方又在偷懒摸鱼"),
        ("橘猫打滚", "橘猫滚一圈", "表示今天心情懒洋洋想撒娇"),
        ("柴犬歪头", "柴犬疑惑", "表示对眼前的事情有点摸不着头脑"),
    ):
        save_res = json.loads(provider.handle_tool_call("sanae_meme_save", {{
            "title": title,
            "aliases": [alias],
            "meaning": meaning,
        }}))
        assert save_res["ok"] is True and save_res["result"]["stored"] is True

    from sanae_mem.governance.source_context import current_chat_scope
    owner = current_chat_scope().owner
    record_verified_cron_send(
        execution_id="cron-exec-1",
        job_id="morning-ping",
        platform="telegram",
        chat_id="1001",
        thread_id="",
        bound_owner=owner,
        delivered_at=time.time() - 5.0,
        delivered_text="早上我顺手泡了杯红茶，提醒你带伞。",
    )

    pre_results = manager.invoke_hook(
        "pre_llm_call",
        session_id="dm-s1",
        turn_id="turn-1",
        platform="telegram",
        sender_id="1001",
        user_message="今天又猫猫摸鱼了，下午去图书馆看书",
        conversation_history=[],
    )

assert any("[梗与表情记忆]" in str(r.get("context") or "") for r in pre_results if isinstance(r, dict))
assert any("[常玩梗短提醒]" in str(r.get("context") or "") for r in pre_results if isinstance(r, dict))
assert any("早上我顺手泡了杯红茶" in str(r.get("context") or "") for r in pre_results if isinstance(r, dict))

# Strict invariant: zero DB writes before real DeliveryReceipt
with sqlite3.connect(str(home / "sanae-mem" / "sanae.db")) as conn:
    counts_before_receipt = {{
        "daily_context": conn.execute("SELECT COUNT(*) FROM daily_context").fetchone()[0],
        "commitments": conn.execute("SELECT COUNT(*) FROM commitments").fetchone()[0],
        "inner_state_revisions": conn.execute("SELECT COUNT(*) FROM inner_state_revisions").fetchone()[0],
    }}


def _wait_all_inner() -> None:
    for mod in list(sys.modules.values()):
        fn = getattr(mod, "wait_for_pending_updates", None) if mod is not None else None
        if callable(fn):
            fn(timeout=10.0)


# Turn 1: DeliveryReceipt WITH explicit turn_id
emit_delivery_receipt(DeliveryReceipt(
    session_id="dm-s1",
    platform="telegram",
    chat_id="1001",
    chat_type="dm",
    sender_id="1001",
    turn_id="turn-1",
    message_id="out-1",
    incoming_message_id="in-1",
    session_key="agent:main:telegram:dm:1001",
    user_message="今天又猫猫摸鱼了，下午去图书馆看书",
    delivered_text="去图书馆刚好能换个安静环境。",
    clean_text="去图书馆刚好能换个安静环境。",
    delivered_at=time.time(),
    is_bot=False,
    transcript=(
        {{"role": "user", "content": "今天又猫猫摸鱼了，下午去图书馆看书", "platform_message_id": "in-1"}},
        {{"role": "assistant", "content": "去图书馆刚好能换个安静环境。"}},
    ),
))
_wait_all_inner()

# Turn 2: DeliveryReceipt WITHOUT turn_id -> observe_final_text clean_reply -> turn_id fallback
env_t2 = TurnEnvelope(
    platform="telegram",
    chat_type="dm",
    chat_id="1001",
    sender_id="1001",
    session_id="dm-s1",
    session_key="agent:main:telegram:dm:1001",
    message_id="in-2",
    turn_id="turn-2",
    is_messaging_surface=True,
    write_origin="foreground",
)
with turn_envelope_scope(env_t2):
    record_verified_cron_send(
        execution_id="cron-exec-2",
        job_id="evening-ping",
        platform="telegram",
        chat_id="1001",
        thread_id="",
        bound_owner=owner,
        delivered_at=time.time(),
        delivered_text="傍晚刚在巷口看到一家新开的拉面馆。",
    )
    time.sleep(0.05)
    manager.invoke_hook(
        "pre_llm_call",
        session_id="dm-s1",
        turn_id="turn-2",
        platform="telegram",
        sender_id="1001",
        user_message="哈哈又猫猫摸鱼啦，晚上去吃拉面",
        conversation_history=[],
    )
    manager.invoke_hook(
        "post_llm_call",
        session_id="dm-s1",
        turn_id="turn-2",
        platform="telegram",
        user_message="哈哈又猫猫摸鱼啦，晚上去吃拉面",
        assistant_response="那家豚骨汤头肯定很暖和。",
        model="test-main",
    )

emit_delivery_receipt(DeliveryReceipt(
    session_id="dm-s1",
    platform="telegram",
    chat_id="1001",
    chat_type="dm",
    sender_id="1001",
    turn_id="",
    message_id="out-2",
    incoming_message_id="in-2",
    session_key="agent:main:telegram:dm:1001",
    user_message="哈哈又猫猫摸鱼啦，晚上去吃拉面",
    delivered_text="那家豚骨汤头肯定很暖和。",
    clean_text="那家豚骨汤头肯定很暖和。",
    delivered_at=time.time(),
    is_bot=False,
    transcript=(
        {{"role": "user", "content": "哈哈又猫猫摸鱼啦，晚上去吃拉面", "platform_message_id": "in-2"}},
        {{"role": "assistant", "content": "那家豚骨汤头肯定很暖和。"}},
    ),
))
_wait_all_inner()

# Turn 3: Explicit receipt.plugin_user_context overrides non-empty cached context via public emit_delivery_receipt
env_t3 = TurnEnvelope(
    platform="telegram",
    chat_type="dm",
    chat_id="1001",
    sender_id="1001",
    session_id="dm-s1",
    session_key="agent:main:telegram:dm:1001",
    message_id="in-3",
    turn_id="turn-3",
    is_messaging_surface=True,
    write_origin="foreground",
)
with turn_envelope_scope(env_t3):
    manager.invoke_hook(
        "pre_llm_call",
        session_id="dm-s1",
        turn_id="turn-3",
        platform="telegram",
        sender_id="1001",
        user_message="明早坐高铁出差，顺便猫猫摸鱼一下",
        conversation_history=[],
    )
    # Also observe final text with same reply as later FIFO test to verify explicit turn_id removes its slot
    manager.invoke_hook(
        "post_llm_call",
        session_id="dm-s1",
        turn_id="turn-3",
        platform="telegram",
        user_message="明早坐高铁出差，顺便猫猫摸鱼一下",
        assistant_response="好呀，我知道啦。",
        model="test-main",
    )

@_dataclass(frozen=True)
class _ReceiptWithContext(DeliveryReceipt):
    plugin_user_context: str = ""

emit_delivery_receipt(_ReceiptWithContext(
    session_id="dm-s1",
    platform="telegram",
    chat_id="1001",
    chat_type="dm",
    sender_id="1001",
    turn_id="turn-3",
    message_id="out-3",
    incoming_message_id="in-3",
    session_key="agent:main:telegram:dm:1001",
    user_message="明早坐高铁出差，记得叫我",
    delivered_text="好呀，我知道啦。",
    clean_text="好呀，我知道啦。",
    delivered_at=time.time(),
    is_bot=False,
    transcript=(
        {{"role": "user", "content": "明早坐高铁出差，记得叫我", "platform_message_id": "in-3"}},
        {{"role": "assistant", "content": "好呀，我知道啦。"}},
    ),
    plugin_user_context=(
        "[我刚才在别处说过]\\n2026-11-19 20:00｜定时主动发言｜显式回执里的别处发言。\\n\\n"
        "[梗与表情记忆] 内部参考：\\n- 记录#99「显式回执梗」：优先使用回执自带的上下文"
    ),
))
_wait_all_inner()

# Turns 5A & 5B: Same session, TWO turns producing the EXACT SAME clean_reply ("好呀，我知道啦。")
# with DIFFERENT memes before either receipt arrives -> omitted-turn_id receipts must FIFO match 1->A, 2->B!
env_5a = TurnEnvelope(
    platform="telegram",
    chat_type="dm",
    chat_id="1001",
    sender_id="1001",
    session_id="dm-s1",
    session_key="agent:main:telegram:dm:1001",
    message_id="in-5a",
    turn_id="turn-same-reply-a",
    is_messaging_surface=True,
    write_origin="foreground",
)
with turn_envelope_scope(env_5a):
    manager.invoke_hook(
        "pre_llm_call",
        session_id="dm-s1",
        turn_id="turn-same-reply-a",
        platform="telegram",
        sender_id="1001",
        user_message="中午吃完饭准备橘猫滚一圈休息会儿",
        conversation_history=[],
    )
    manager.invoke_hook(
        "post_llm_call",
        session_id="dm-s1",
        turn_id="turn-same-reply-a",
        platform="telegram",
        user_message="中午吃完饭准备橘猫滚一圈休息会儿",
        assistant_response="好呀，我知道啦。",
        model="test-main",
    )

env_5b = TurnEnvelope(
    platform="telegram",
    chat_type="dm",
    chat_id="1001",
    sender_id="1001",
    session_id="dm-s1",
    session_key="agent:main:telegram:dm:1001",
    message_id="in-5b",
    turn_id="turn-same-reply-b",
    is_messaging_surface=True,
    write_origin="foreground",
)
with turn_envelope_scope(env_5b):
    manager.invoke_hook(
        "pre_llm_call",
        session_id="dm-s1",
        turn_id="turn-same-reply-b",
        platform="telegram",
        sender_id="1001",
        user_message="刚开完需求会有点柴犬疑惑在复盘",
        conversation_history=[],
    )
    manager.invoke_hook(
        "post_llm_call",
        session_id="dm-s1",
        turn_id="turn-same-reply-b",
        platform="telegram",
        user_message="刚开完需求会有点柴犬疑惑在复盘",
        assistant_response="好呀，我知道啦。",
        model="test-main",
    )

receipt_5a = DeliveryReceipt(
    session_id="dm-s1",
    platform="telegram",
    chat_id="1001",
    chat_type="dm",
    sender_id="1001",
    turn_id="",
    message_id="out-5a",
    incoming_message_id="in-5a",
    session_key="agent:main:telegram:dm:1001",
    user_message="中午吃完饭准备橘猫滚一圈休息会儿",
    delivered_text="好呀，我知道啦。",
    clean_text="好呀，我知道啦。",
    delivered_at=time.time(),
    is_bot=False,
    transcript=(
        {{"role": "user", "content": "中午吃完饭准备橘猫滚一圈休息会儿", "platform_message_id": "in-5a"}},
        {{"role": "assistant", "content": "好呀，我知道啦。"}},
    ),
)
emit_delivery_receipt(receipt_5a)
_wait_all_inner()

# Verbatim replay of omitted-turn_id out-5a BEFORE out-5b arrives:
# must be deduplicated cleanly WITHOUT popping out-5b's FIFO turn_id or inflating inner_state_revisions.
emit_delivery_receipt(receipt_5a)
_wait_all_inner()

emit_delivery_receipt(DeliveryReceipt(
    session_id="dm-s1",
    platform="telegram",
    chat_id="1001",
    chat_type="dm",
    sender_id="1001",
    turn_id="",
    message_id="out-5b",
    incoming_message_id="in-5b",
    session_key="agent:main:telegram:dm:1001",
    user_message="刚开完需求会有点柴犬疑惑在复盘",
    delivered_text="好呀，我知道啦。",
    clean_text="好呀，我知道啦。",
    delivered_at=time.time() + 0.01,
    is_bot=False,
    transcript=(
        {{"role": "user", "content": "刚开完需求会有点柴犬疑惑在复盘", "platform_message_id": "in-5b"}},
        {{"role": "assistant", "content": "好呀，我知道啦。"}},
    ),
))
_wait_all_inner()

# Turn 4: Different session/turn must NOT inherit any cached context from dm-s1
emit_delivery_receipt(DeliveryReceipt(
    session_id="dm-other-session",
    platform="telegram",
    chat_id="1001",
    chat_type="dm",
    sender_id="1001",
    turn_id="turn-1",
    message_id="out-4",
    incoming_message_id="in-4",
    session_key="agent:main:telegram:dm:1001",
    user_message="我刚才喝了一杯热茶，感觉清醒多了",
    delivered_text="喝点热的确实舒服。",
    clean_text="喝点热的确实舒服。",
    delivered_at=time.time() + 0.02,
    is_bot=False,
    transcript=(
        {{"role": "user", "content": "我刚才喝了一杯热茶，感觉清醒多了", "platform_message_id": "in-4"}},
        {{"role": "assistant", "content": "喝点热的确实舒服。"}},
    ),
))
_wait_all_inner()

from sanae_mem.runtime.private_delivery_settler import read_turn_user_context
other_home_read = read_turn_user_context(other_home, "dm-s1", "turn-1")

with sqlite3.connect(str(home / "sanae-mem" / "sanae.db")) as conn:
    counts_after = {{
        "daily_context": conn.execute("SELECT COUNT(*) FROM daily_context").fetchone()[0],
        "inner_state_revisions": conn.execute("SELECT COUNT(*) FROM inner_state_revisions").fetchone()[0],
    }}

provider.shutdown()
server.shutdown()

print(json.dumps({{
    "counts_before_receipt": counts_before_receipt,
    "counts_after": counts_after,
    "daily_prompts": daily_prompts,
    "other_home_read": other_home_read,
}}, ensure_ascii=False))
"""
    res = _run_clean_subprocess(script, vivemind_root=vivemind_root, hermes_root=hermes_root, home=home)

    # Zero writes before real delivery receipt
    assert res["counts_before_receipt"] == {
        "daily_context": 0,
        "commitments": 0,
        "inner_state_revisions": 0,
    }
    assert res["counts_after"]["daily_context"] >= 3
    assert res["counts_after"]["inner_state_revisions"] == 6
    assert res["other_home_read"] == ""

    prompts: list[str] = res["daily_prompts"]
    assert len(prompts) == 6, f"expected exactly 6 Step 2 requests across 6 distinct turns (no replay inflation), got {len(prompts)}"

    # Strict assertions: EVERY Step 2 request for each target turn must carry its full expected context blocks.
    def _prompts_for(user_snippet: str) -> list[str]:
        return [
            p for p in prompts
            if user_snippet in p.split("【最近对话】", 1)[0]
        ]

    t1_prompts = _prompts_for("下午去图书馆")
    assert len(t1_prompts) == 1, f"expected 1 Step 2 request for Turn 1, got {len(t1_prompts)}: {t1_prompts!r}"
    assert all(
        "早上我顺手泡了杯红茶" in p and "摸鱼小猫" in p and "用来打趣对方又在偷懒摸鱼" in p
        for p in t1_prompts
    ), f"Turn 1 lost elsewhere or meme_block: {t1_prompts!r}"

    t2_prompts = _prompts_for("晚上去吃拉面")
    assert len(t2_prompts) == 1, f"expected 1 Step 2 request for Turn 2, got {len(t2_prompts)}: {t2_prompts!r}"
    assert all(
        "傍晚刚在巷口看到一家新开的拉面馆" in p and "摸鱼小猫" in p
        for p in t2_prompts
    ), f"Turn 2 lost elsewhere or meme_block on clean_reply fallback: {t2_prompts!r}"

    t3_prompts = _prompts_for("明早坐高铁出差")
    assert len(t3_prompts) == 1, f"expected 1 Step 2 request for Turn 3, got {len(t3_prompts)}: {t3_prompts!r}"
    assert all(
        "显式回执里的别处发言" in p and "显式回执梗" in p and "摸鱼小猫" not in p
        for p in t3_prompts
    ), f"Turn 3 did not honor explicit receipt.plugin_user_context: {t3_prompts!r}"

    # Two same-clean_reply turns without turn_id: strict FIFO match with zero cross-talk
    t5a_prompts = _prompts_for("中午吃完饭准备橘猫滚一圈休息会儿")
    assert len(t5a_prompts) == 1, f"expected 1 Step 2 request for Turn 5A, got {len(t5a_prompts)}: {t5a_prompts!r}"
    assert all(
        "橘猫打滚" in p and "表示今天心情懒洋洋想撒娇" in p and "柴犬歪头" not in p and "显式回执" not in p
        for p in t5a_prompts
    ), f"Turn 5A mismatched meme block: {t5a_prompts!r}"

    t5b_prompts = _prompts_for("刚开完需求会有点柴犬疑惑在复盘")
    assert len(t5b_prompts) == 1, f"expected 1 Step 2 request for Turn 5B, got {len(t5b_prompts)}: {t5b_prompts!r}"
    assert all(
        "柴犬歪头" in p and "表示对眼前的事情有点摸不着头脑" in p and "橘猫打滚" not in p and "显式回执" not in p
        for p in t5b_prompts
    ), f"Turn 5B mismatched meme block: {t5b_prompts!r}"

    # Other session must NOT inherit any cached context
    t4_prompts = _prompts_for("喝了一杯热茶")
    assert len(t4_prompts) == 1, f"expected 1 Step 2 request for Turn 4, got {len(t4_prompts)}: {t4_prompts!r}"
    for p in t4_prompts:
        assert "早上我顺手泡了杯红茶" not in p
        assert "摸鱼小猫" not in p
        assert "橘猫打滚" not in p
        assert "柴犬歪头" not in p
        assert "显式回执" not in p


def test_settle_private_delivery_truthful_step_results_and_failures_subprocess(
    tmp_path: Path,
) -> None:
    """Step 2/4/5 return exact bools across full success, skips, quota/noop, rejected commitments,
    and real SQLite write failures (commitments trigger, rementions trigger, watermark trigger).
    """
    vivemind_root, hermes_root = _resolve_roots()
    home = tmp_path / "hermes_task02_results"
    home.mkdir(parents=True)

    script = f"""
import importlib
import json
import logging
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

home = Path({str(home)!r})

fault_mode = {{"mode": "ok", "pending_id": None}}
warnings_captured = []


class _ListHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING:
            warnings_captured.append(record.getMessage())


logging.getLogger().addHandler(_ListHandler())


class _FaultFlashHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = body.get("messages") or []
        sys_text = ""
        user_text = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "")
            if role in ("system", "developer"):
                sys_text += content
            elif role == "user":
                user_text = content

        mode = fault_mode["mode"]
        is_daily = "DAILY_FACT" in sys_text and "【本轮用户消息】" in user_text
        is_commitments = "new_commitments" in sys_text
        is_inner = not is_daily and not is_commitments

        if (
            mode == "all_fail"
            or (mode == "daily_fail" and is_daily)
            or (mode == "inner_fail" and is_inner)
        ):
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"upstream error")
            return

        if mode == "commitments_invalid_json" and is_commitments:
            content = "not-valid-json-{{{{"
        elif is_daily:
            if mode == "daily_thread_1":
                items = [{{
                    "kind": "OPEN_THREAD", "action": "set", "target_slot": "",
                    "event_key": "astro_film",
                    "text": "好奇他提到的那部冷门天文纪录片",
                    "quote": "天文纪录片",
                    "uncertain": False, "plan": False,
                }}]
            elif mode == "daily_thread_2":
                items = [{{
                    "kind": "OPEN_THREAD", "action": "set", "target_slot": "",
                    "event_key": "pottery_glaze",
                    "text": "想回头查查他说的柴烧陶釉色工艺",
                    "quote": "柴烧陶釉色",
                    "uncertain": False, "plan": False,
                }}]
            elif mode == "daily_thread_quota":
                items = [{{
                    "kind": "OPEN_THREAD", "action": "set", "target_slot": "",
                    "event_key": "bamboo_flute",
                    "text": "想了解一下他刚提的竹笛指法门派",
                    "quote": "竹笛指法",
                    "uncertain": False, "plan": False,
                }}]
            elif mode == "daily_noop":
                items = [{{
                    "kind": "DAILY_FACT", "action": "set", "target_slot": "",
                    "event_key": "walk_park",
                    "text": "他今天要去公园散步",
                    "quote": "今天要去公园散步",
                    "uncertain": False, "plan": False,
                }}]
            else:
                items = [{{
                    "kind": "DAILY_FACT", "action": "set", "target_slot": "",
                    "event_key": "walk_park" if mode == "ok" else f"fact_{{mode}}",
                    "text": "他今天要去公园散步" if mode == "ok" else f"他今天要去公园散步（{{mode}}）",
                    "quote": "今天要去公园散步",
                    "uncertain": False, "plan": False,
                }}]
            content = json.dumps({{"items": items}}, ensure_ascii=False)
        elif is_commitments:
            if mode == "commitments_fabricated_quote":
                content = json.dumps({{
                    "new_commitments": [{{
                        "text": "明天帮他买咖啡",
                        "quote": "一段回复里根本不存在的伪造原话",
                        "masked_text": "帮买咖啡",
                    }}],
                    "resolutions": [],
                }}, ensure_ascii=False)
            elif mode == "commitments_create_real":
                content = json.dumps({{
                    "new_commitments": [{{
                        "text": "明天把整理好的清单发给他",
                        "quote": "去公园走走挺舒服的。",
                        "masked_text": "整理一份清单",
                    }}],
                    "resolutions": [],
                }}, ensure_ascii=False)
            elif mode == "commitments_rejected_resolution":
                content = json.dumps({{
                    "new_commitments": [],
                    "resolutions": [{{
                        "id": fault_mode["pending_id"],
                        "action": "fulfilled",
                        "evidence": "伪造确认",
                        "tool_call_id": "",
                        "user_quote": "用户根本没说过的确认原话",
                    }}],
                }}, ensure_ascii=False)
            elif mode == "commitments_db_error":
                content = json.dumps({{
                    "new_commitments": [{{
                        "text": "晚上帮他核对航班信息",
                        "quote": "去公园走走挺舒服的。",
                        "masked_text": "核对行程信息",
                    }}],
                    "resolutions": [],
                }}, ensure_ascii=False)
            else:
                content = json.dumps({{"new_commitments": [], "resolutions": []}}, ensure_ascii=False)
        else:
            content = json.dumps({{
                "sentences": ["听他说去公园散步，心里放松了一点。", "希望今晚风别太大。"],
                "variables": {{"tension": 0.1, "closeness": 0.6, "energy": 0.1}},
                "shared_sentences": None,
            }}, ensure_ascii=False)

        resp = json.dumps({{
            "id": "chatcmpl-fault",
            "object": "chat.completion",
            "choices": [{{
                "index": 0,
                "message": {{"role": "assistant", "content": content}},
                "finish_reason": "stop",
            }}],
            "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
        }}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


server = ThreadingHTTPServer(("127.0.0.1", 0), _FaultFlashHandler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{{port}}/v1"

(home / "config.yaml").write_text(
    yaml.safe_dump({{
        "memory": {{"provider": "sanae_mem"}},
        "auxiliary": {{
            "sanae_daily_flash": {{
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": base_url,
                "api_key": "dummy-key",
            }},
            "sanae_extract": {{
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": base_url,
                "api_key": "dummy-key",
            }},
        }},
        "plugins": {{
            "enabled": ["sanae_mem"],
            "sanae_mem": {{
                "data_dir": str(home / "sanae-mem"),
                "semantic_backend": "sqlite",
                "memory_identity": {{"platform": "telegram", "user_id": "1001"}},
            }},
        }},
    }}, allow_unicode=True),
    encoding="utf-8",
)

import plugins.memory as host_memory
from hermes_host_contracts import TurnEnvelope, turn_envelope_scope

provider = host_memory.load_memory_provider("sanae_mem")
assert provider is not None
provider_module = type(provider).__module__
pkg = (
    provider_module.rsplit(".", 2)[0]
    if provider_module.endswith(".runtime.provider")
    else provider_module.rsplit(".", 1)[0]
)
settler = importlib.import_module(f"{{pkg}}.runtime.private_delivery_settler")
VerifiedPrivateDelivery = settler.VerifiedPrivateDelivery
settle_private_delivery = settler.settle_private_delivery
db_path = home / "sanae-mem" / "sanae.db"


def _run_case(
    mode: str,
    *,
    turn_id: str,
    sender_id: str = "1001",
    session_id: str = "dm-truth",
    user_message: str = "今天要去公园散步",
    delivered_at: float,
) -> dict:
    fault_mode["mode"] = mode
    env = TurnEnvelope(
        platform="telegram",
        chat_type="dm",
        chat_id=sender_id,
        sender_id=sender_id,
        session_id=session_id,
        session_key=f"agent:main:telegram:dm:{{sender_id}}",
        message_id=f"in-{{turn_id}}",
        turn_id=turn_id,
        is_messaging_surface=True,
        write_origin="foreground",
    )
    with turn_envelope_scope(env):
        provider.initialize(session_id, platform="telegram", hermes_home=str(home), agent_context="primary")
        delivery = VerifiedPrivateDelivery(
            session_id=session_id,
            turn_id=turn_id,
            platform="telegram",
            chat_id=sender_id,
            chat_type="dm",
            sender_id=sender_id,
            user_message=user_message,
            assistant_response="去公园走走挺舒服的。",
            delivered_at=delivered_at,
            incoming_message_id=f"in-{{turn_id}}",
            message_id=f"out-{{turn_id}}",
            session_key=f"agent:main:telegram:dm:{{sender_id}}",
            is_bot=False,
            transcript=(
                {{"role": "user", "content": user_message, "platform_message_id": f"in-{{turn_id}}"}},
                {{"role": "assistant", "content": "去公园走走挺舒服的。"}},
            ),
        )
        res = settle_private_delivery(provider, delivery)
        return {{
            "settled": res.settled,
            "daily_recorded": res.daily_recorded,
            "commitments_updated": res.commitments_updated,
            "inner_state_updated": res.inner_state_updated,
            "reason": res.reason,
        }}


base_ts = time.time() - 60.0
outcomes = {{}}
outcomes["all_ok"] = _run_case("ok", turn_id="t-ok", delivered_at=base_ts + 1.0)
# Noop same DAILY_FACT text/slot -> no DB write -> daily_recorded=False
outcomes["daily_noop"] = _run_case("daily_noop", turn_id="t-daily-noop", delivered_at=base_ts + 2.0)

# Fill OPEN_THREAD quota (MAX_ACTIVE_PER_KIND[KIND_THREAD] == 2), then 3rd thread must be rejected with daily_recorded=False
outcomes["thread_1"] = _run_case(
    "daily_thread_1", turn_id="t-th-1", user_message="刚听说有部冷门天文纪录片很棒", delivered_at=base_ts + 3.0,
)
outcomes["thread_2"] = _run_case(
    "daily_thread_2", turn_id="t-th-2", user_message="还有人研究柴烧陶釉色工艺", delivered_at=base_ts + 4.0,
)
outcomes["daily_quota_full"] = _run_case(
    "daily_thread_quota", turn_id="t-th-quota", user_message="我还听说竹笛指法有不少门派", delivered_at=base_ts + 5.0,
)

outcomes["non_maintainer"] = _run_case(
    "ok", turn_id="t-other-user", sender_id="2002", session_id="dm-other-user", delivered_at=base_ts + 6.0,
)
outcomes["daily_fail"] = _run_case("daily_fail", turn_id="t-daily-fail", delivered_at=base_ts + 7.0)
outcomes["commitments_fail"] = _run_case(
    "commitments_invalid_json", turn_id="t-commit-fail", delivered_at=base_ts + 8.0,
)
outcomes["commitments_fabricated"] = _run_case(
    "commitments_fabricated_quote", turn_id="t-commit-fab", delivered_at=base_ts + 9.0,
)

# Create one real pending commitment, then reject a resolution with fabricated user_quote
outcomes["commitments_created"] = _run_case(
    "commitments_create_real", turn_id="t-commit-create", delivered_at=base_ts + 10.0,
)
with sqlite3.connect(str(db_path)) as conn:
    pending_row = conn.execute("SELECT id FROM commitments WHERE status='pending' ORDER BY id DESC LIMIT 1").fetchone()
    fault_mode["pending_id"] = int(pending_row[0]) if pending_row else None
outcomes["commitments_rejected_res"] = _run_case(
    "commitments_rejected_resolution", turn_id="t-commit-rej-res", delivered_at=base_ts + 11.0,
)

# SQLite IntegrityError trigger on commitments INSERT -> must NOT crash settle_private_delivery; returns False + status.json
with sqlite3.connect(str(db_path)) as conn:
    conn.execute(
        "CREATE TRIGGER block_commitment_insert BEFORE INSERT ON commitments "
        "BEGIN SELECT RAISE(ABORT, 'simulated_commitment_constraint'); END"
    )
    conn.commit()
try:
    outcomes["commitments_db_error"] = _run_case(
        "commitments_db_error", turn_id="t-commit-db-err", delivered_at=base_ts + 12.0,
    )
finally:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DROP TRIGGER IF EXISTS block_commitment_insert")
        conn.commit()

status_after_commit_err = json.loads((home / "sanae-mem" / "status.json").read_text(encoding="utf-8"))

outcomes["inner_fail"] = _run_case("inner_fail", turn_id="t-inner-fail", delivered_at=base_ts + 13.0)

# SQLite trigger on memory_axis_mentions INSERT -> record_delivered_rementions returns ok=False -> inner_state_updated=False
with sqlite3.connect(str(db_path)) as conn:
    conn.execute(
        "CREATE TRIGGER block_remention_insert BEFORE INSERT ON memory_axis_mentions "
        "BEGIN SELECT RAISE(ABORT, 'simulated_remention_constraint'); END"
    )
    conn.commit()
try:
    outcomes["remention_db_error"] = _run_case(
        "ok_remention_err", turn_id="t-remention-err", delivered_at=base_ts + 14.0,
    )
finally:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DROP TRIGGER IF EXISTS block_remention_insert")
        conn.commit()

# SQLite trigger on session_reply_watermarks INSERT/UPDATE -> _apply_verified_delivery catches sqlite3.Error -> False
with sqlite3.connect(str(db_path)) as conn:
    conn.execute(
        "CREATE TRIGGER block_watermark_update BEFORE UPDATE ON session_reply_watermarks "
        "BEGIN SELECT RAISE(ABORT, 'simulated_watermark_err'); END"
    )
    conn.commit()
try:
    outcomes["watermark_db_error"] = _run_case(
        "ok_watermark_err", turn_id="t-watermark-err", delivered_at=base_ts + 15.0,
    )
finally:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DROP TRIGGER IF EXISTS block_watermark_update")
        conn.commit()

outcomes["all_fail"] = _run_case("all_fail", turn_id="t-all-fail", delivered_at=base_ts + 16.0)

with sqlite3.connect(str(db_path)) as conn:
    pending_status = conn.execute(
        "SELECT status FROM commitments WHERE id=?", (fault_mode["pending_id"],)
    ).fetchone()[0]

provider.shutdown()
server.shutdown()

print(json.dumps({{
    "outcomes": outcomes,
    "commit_status": (status_after_commit_err.get("components") or {{}}).get("commitments"),
    "pending_status": pending_status,
    "incomplete_warnings": [
        w for w in warnings_captured if "private delivery settlement incomplete" in w
    ],
    "remention_warnings": [
        w for w in warnings_captured if "delivered remention failed" in w
    ],
}}, ensure_ascii=False))
"""
    res = _run_clean_subprocess(script, vivemind_root=vivemind_root, hermes_root=hermes_root, home=home)
    outcomes = res["outcomes"]

    assert outcomes["all_ok"] == {
        "settled": True,
        "daily_recorded": True,
        "commitments_updated": True,
        "inner_state_updated": True,
        "reason": "",
    }

    # Item 2: noop & quota-full daily decisions must truthfully report daily_recorded=False
    assert outcomes["daily_noop"]["settled"] is True
    assert outcomes["daily_noop"]["daily_recorded"] is False
    assert "daily_not_recorded" in outcomes["daily_noop"]["reason"]
    assert outcomes["thread_1"]["daily_recorded"] is True
    assert outcomes["thread_2"]["daily_recorded"] is True
    assert outcomes["daily_quota_full"]["settled"] is True
    assert outcomes["daily_quota_full"]["daily_recorded"] is False
    assert "daily_not_recorded" in outcomes["daily_quota_full"]["reason"]

    # Non-maintainer skip
    assert outcomes["non_maintainer"]["settled"] is True
    assert outcomes["non_maintainer"]["daily_recorded"] is False
    assert outcomes["non_maintainer"]["commitments_updated"] is True
    assert outcomes["non_maintainer"]["inner_state_updated"] is True
    assert "daily_skipped" in outcomes["non_maintainer"]["reason"]

    # Daily LLM 500 failure
    assert outcomes["daily_fail"]["settled"] is True
    assert outcomes["daily_fail"]["daily_recorded"] is False
    assert outcomes["daily_fail"]["commitments_updated"] is True
    assert outcomes["daily_fail"]["inner_state_updated"] is True
    assert "daily_not_recorded" in outcomes["daily_fail"]["reason"]

    # Item 3: commitments invalid JSON, fabricated quote, rejected resolution, and SQLite INSERT failure
    assert outcomes["commitments_fail"]["settled"] is True
    assert outcomes["commitments_fail"]["daily_recorded"] is True
    assert outcomes["commitments_fail"]["commitments_updated"] is False
    assert outcomes["commitments_fail"]["inner_state_updated"] is True
    assert "commitments_not_updated" in outcomes["commitments_fail"]["reason"]

    assert outcomes["commitments_fabricated"]["settled"] is True
    assert outcomes["commitments_fabricated"]["commitments_updated"] is False
    assert "commitments_not_updated" in outcomes["commitments_fabricated"]["reason"]

    assert outcomes["commitments_created"]["commitments_updated"] is True
    assert outcomes["commitments_rejected_res"]["settled"] is True
    assert outcomes["commitments_rejected_res"]["commitments_updated"] is False
    assert "commitments_not_updated" in outcomes["commitments_rejected_res"]["reason"]
    assert res["pending_status"] == "pending"

    assert outcomes["commitments_db_error"]["settled"] is True
    assert outcomes["commitments_db_error"]["commitments_updated"] is False
    assert outcomes["commitments_db_error"]["inner_state_updated"] is True
    assert "commitments_not_updated" in outcomes["commitments_db_error"]["reason"]
    assert res["commit_status"]["ok"] is False
    assert res["commit_status"]["stage"] == "record"
    assert res["commit_status"]["error"] == "record_failed"

    # Item 4: Step 5 inner Flash failure, remention SQLite failure, and watermark SQLite failure
    assert outcomes["inner_fail"]["settled"] is True
    assert outcomes["inner_fail"]["daily_recorded"] is True
    assert outcomes["inner_fail"]["commitments_updated"] is True
    assert outcomes["inner_fail"]["inner_state_updated"] is False
    assert "inner_state_not_updated" in outcomes["inner_fail"]["reason"]

    assert outcomes["remention_db_error"]["settled"] is True
    assert outcomes["remention_db_error"]["inner_state_updated"] is False
    assert "inner_state_not_updated" in outcomes["remention_db_error"]["reason"]
    assert len(res["remention_warnings"]) >= 1

    assert outcomes["watermark_db_error"]["settled"] is True
    assert outcomes["watermark_db_error"]["inner_state_updated"] is False
    assert "inner_state_not_updated" in outcomes["watermark_db_error"]["reason"]

    assert outcomes["all_fail"]["settled"] is True
    assert outcomes["all_fail"]["daily_recorded"] is False
    assert outcomes["all_fail"]["commitments_updated"] is False
    assert outcomes["all_fail"]["inner_state_updated"] is False
    assert "daily_not_recorded" in outcomes["all_fail"]["reason"]
    assert "commitments_not_updated" in outcomes["all_fail"]["reason"]
    assert "inner_state_not_updated" in outcomes["all_fail"]["reason"]

    assert len(res["incomplete_warnings"]) >= 10


def _run_same_reply_omitted_turn_id_scenario(
    tmp_path: Path,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Run a clean dual-load subprocess with two same-reply omitted-turn_id turns (in-5a, in-5b)."""
    vivemind_root, hermes_root = _resolve_roots()
    home = tmp_path / f"hermes_task02_{scenario}"
    home.mkdir(parents=True)

    script = f"""
import json
import sqlite3
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

home = Path({str(home)!r})
scenario = {scenario!r}

daily_prompts = []
flash_lock = threading.Lock()


class _FlashHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = body.get("messages") or []
        sys_text = ""
        user_text = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "")
            if role in ("system", "developer"):
                sys_text += content
            elif role == "user":
                user_text = content

        if "DAILY_FACT" in sys_text and "【本轮用户消息】" in user_text:
            with flash_lock:
                daily_prompts.append(user_text)
            payload = {{"items": []}}
        elif "new_commitments" in sys_text:
            payload = {{"new_commitments": [], "resolutions": []}}
        else:
            payload = {{
                "sentences": ["记着他刚才说的话，心里踏实了一些。", "一会儿再接着聊。"],
                "variables": {{"tension": 0.1, "closeness": 0.6, "energy": 0.1}},
                "shared_sentences": None,
            }}

        resp = json.dumps({{
            "id": "chatcmpl-task02-order",
            "object": "chat.completion",
            "choices": [{{
                "index": 0,
                "message": {{"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}},
                "finish_reason": "stop",
            }}],
            "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
        }}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


server = ThreadingHTTPServer(("127.0.0.1", 0), _FlashHandler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
base_url = f"http://127.0.0.1:{{port}}/v1"

(home / "config.yaml").write_text(
    yaml.safe_dump({{
        "memory": {{"provider": "sanae_mem"}},
        "auxiliary": {{
            "sanae_daily_flash": {{
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": base_url,
                "api_key": "dummy-key",
            }},
            "sanae_extract": {{
                "provider": "custom",
                "model": "test-flash-model",
                "base_url": base_url,
                "api_key": "dummy-key",
            }},
        }},
        "plugins": {{
            "enabled": ["sanae_mem"],
            "sanae_mem": {{
                "data_dir": str(home / "sanae-mem"),
                "semantic_backend": "sqlite",
                "memory_identity": {{"platform": "telegram", "user_id": "1001"}},
            }},
        }},
    }}, allow_unicode=True),
    encoding="utf-8",
)

import hermes_cli.plugins as host_plugins
import plugins.memory as host_memory
from hermes_host_contracts import (
    DeliveryReceipt,
    TurnEnvelope,
    emit_delivery_receipt,
    record_verified_cron_send,
    turn_envelope_scope,
)

host_plugins.discover_plugins(force=True)
provider = host_memory.load_memory_provider("sanae_mem")
assert provider is not None
manager = host_plugins.get_plugin_manager()
shared_reply = "好呀，我知道啦。"


def _wait_all_inner() -> None:
    for mod in list(sys.modules.values()):
        fn = getattr(mod, "wait_for_pending_updates", None) if mod is not None else None
        if callable(fn):
            fn(timeout=10.0)


# Turn 5A (in-5a): 窗边有一盆薄荷 + 橘猫打滚
env_5a = TurnEnvelope(
    platform="telegram",
    chat_type="dm",
    chat_id="1001",
    sender_id="1001",
    session_id="dm-order",
    session_key="agent:main:telegram:dm:1001",
    message_id="in-5a",
    turn_id="turn-5a",
    is_messaging_surface=True,
    write_origin="foreground",
)
with turn_envelope_scope(env_5a):
    provider.initialize("dm-order", platform="telegram", hermes_home=str(home), agent_context="primary")
    for title, alias, meaning in (
        ("橘猫打滚", "橘猫滚一圈", "表示今天心情懒洋洋想撒娇"),
        ("柴犬歪头", "柴犬疑惑", "表示对眼前的事情有点摸不着头脑"),
    ):
        save_res = json.loads(provider.handle_tool_call("sanae_meme_save", {{
            "title": title,
            "aliases": [alias],
            "meaning": meaning,
        }}))
        assert save_res["ok"] is True and save_res["result"]["stored"] is True

    from sanae_mem.governance.source_context import current_chat_scope
    owner = current_chat_scope().owner
    record_verified_cron_send(
        execution_id="cron-5a",
        job_id="window-note",
        platform="telegram",
        chat_id="1001",
        thread_id="",
        bound_owner=owner,
        delivered_at=time.time() - 2.0,
        delivered_text="窗边有一盆薄荷，阳光照着挺精神的。",
    )
    manager.invoke_hook(
        "pre_llm_call",
        session_id="dm-order",
        turn_id="turn-5a",
        platform="telegram",
        sender_id="1001",
        user_message="中午吃完饭准备橘猫滚一圈休息会儿",
        conversation_history=[],
    )
    manager.invoke_hook(
        "post_llm_call",
        session_id="dm-order",
        turn_id="turn-5a",
        platform="telegram",
        user_message="中午吃完饭准备橘猫滚一圈休息会儿",
        assistant_response=shared_reply,
        model="test-main",
    )

# Turn 5B (in-5b): 桌上有一盏纸灯 + 柴犬歪头
env_5b = TurnEnvelope(
    platform="telegram",
    chat_type="dm",
    chat_id="1001",
    sender_id="1001",
    session_id="dm-order",
    session_key="agent:main:telegram:dm:1001",
    message_id="in-5b",
    turn_id="turn-5b",
    is_messaging_surface=True,
    write_origin="foreground",
)
with turn_envelope_scope(env_5b):
    record_verified_cron_send(
        execution_id="cron-5b",
        job_id="desk-note",
        platform="telegram",
        chat_id="1001",
        thread_id="",
        bound_owner=owner,
        delivered_at=time.time() - 1.0,
        delivered_text="桌上有一盏纸灯，傍晚点起来很柔和。",
    )
    manager.invoke_hook(
        "pre_llm_call",
        session_id="dm-order",
        turn_id="turn-5b",
        platform="telegram",
        sender_id="1001",
        user_message="刚开完需求会有点柴犬疑惑在复盘",
        conversation_history=[],
    )
    manager.invoke_hook(
        "post_llm_call",
        session_id="dm-order",
        turn_id="turn-5b",
        platform="telegram",
        user_message="刚开完需求会有点柴犬疑惑在复盘",
        assistant_response=shared_reply,
        model="test-main",
    )

now_ts = time.time()
if scenario == "empty_outbound_replay":
    receipt_5a_empty = DeliveryReceipt(
        session_id="dm-order",
        platform="telegram",
        chat_id="1001",
        chat_type="dm",
        sender_id="1001",
        turn_id="",
        message_id="",
        incoming_message_id="in-5a",
        session_key="agent:main:telegram:dm:1001",
        user_message="中午吃完饭准备橘猫滚一圈休息会儿",
        delivered_text=shared_reply,
        clean_text=shared_reply,
        delivered_at=now_ts,
        is_bot=False,
        transcript=(
            {{"role": "user", "content": "中午吃完饭准备橘猫滚一圈休息会儿", "platform_message_id": "in-5a"}},
            {{"role": "assistant", "content": shared_reply}},
        ),
    )
    receipt_5b = DeliveryReceipt(
        session_id="dm-order",
        platform="telegram",
        chat_id="1001",
        chat_type="dm",
        sender_id="1001",
        turn_id="",
        message_id="out-5b",
        incoming_message_id="in-5b",
        session_key="agent:main:telegram:dm:1001",
        user_message="刚开完需求会有点柴犬疑惑在复盘",
        delivered_text=shared_reply,
        clean_text=shared_reply,
        delivered_at=now_ts + 0.05,
        is_bot=False,
        transcript=(
            {{"role": "user", "content": "刚开完需求会有点柴犬疑惑在复盘", "platform_message_id": "in-5b"}},
            {{"role": "assistant", "content": shared_reply}},
        ),
    )
    emit_delivery_receipt(receipt_5a_empty)
    _wait_all_inner()
    # Verbatim replay with empty outbound message_id must NOT pop in-5b's queued turn_id
    emit_delivery_receipt(receipt_5a_empty)
    _wait_all_inner()
    emit_delivery_receipt(receipt_5b)
    _wait_all_inner()
elif scenario == "out_of_order_replay":
    receipt_5b = DeliveryReceipt(
        session_id="dm-order",
        platform="telegram",
        chat_id="1001",
        chat_type="dm",
        sender_id="1001",
        turn_id="",
        message_id="out-5b",
        incoming_message_id="in-5b",
        session_key="agent:main:telegram:dm:1001",
        user_message="刚开完需求会有点柴犬疑惑在复盘",
        delivered_text=shared_reply,
        clean_text=shared_reply,
        delivered_at=now_ts,
        is_bot=False,
        transcript=(
            {{"role": "user", "content": "刚开完需求会有点柴犬疑惑在复盘", "platform_message_id": "in-5b"}},
            {{"role": "assistant", "content": shared_reply}},
        ),
    )
    receipt_5a = DeliveryReceipt(
        session_id="dm-order",
        platform="telegram",
        chat_id="1001",
        chat_type="dm",
        sender_id="1001",
        turn_id="",
        message_id="out-5a",
        incoming_message_id="in-5a",
        session_key="agent:main:telegram:dm:1001",
        user_message="中午吃完饭准备橘猫滚一圈休息会儿",
        delivered_text=shared_reply,
        clean_text=shared_reply,
        delivered_at=now_ts + 0.05,
        is_bot=False,
        transcript=(
            {{"role": "user", "content": "中午吃完饭准备橘猫滚一圈休息会儿", "platform_message_id": "in-5a"}},
            {{"role": "assistant", "content": shared_reply}},
        ),
    )
    # Out-of-order: out-5b arrives FIRST, then out-5a, then verbatim replay of out-5a
    emit_delivery_receipt(receipt_5b)
    _wait_all_inner()
    emit_delivery_receipt(receipt_5a)
    _wait_all_inner()
    emit_delivery_receipt(receipt_5a)
    _wait_all_inner()

with sqlite3.connect(str(home / "sanae-mem" / "sanae.db")) as conn:
    inner_revisions = conn.execute("SELECT COUNT(*) FROM inner_state_revisions").fetchone()[0]

provider.shutdown()
server.shutdown()
print(json.dumps({{"daily_prompts": daily_prompts, "inner_revisions": inner_revisions}}, ensure_ascii=False))
"""
    return _run_clean_subprocess(script, vivemind_root=vivemind_root, hermes_root=hermes_root, home=home)


def test_e2e_empty_outbound_message_id_replay_preserves_second_turn_subprocess(
    tmp_path: Path,
) -> None:
    """Replaying an omitted-turn_id receipt with empty message_id must not steal the next turn's context."""
    res = _run_same_reply_omitted_turn_id_scenario(tmp_path, scenario="empty_outbound_replay")
    prompts: list[str] = res["daily_prompts"]
    assert len(prompts) == 2, f"expected exactly 2 Step 2 prompts (replay deduped), got {len(prompts)}"
    assert res["inner_revisions"] == 2

    p_5a, p_5b = prompts[0], prompts[1]
    assert "中午吃完饭准备橘猫滚一圈休息会儿" in p_5a.split("【最近对话】", 1)[0]
    assert "窗边有一盆薄荷" in p_5a
    assert "橘猫打滚" in p_5a
    assert "桌上有一盏纸灯" not in p_5a
    assert "柴犬歪头" not in p_5a

    assert "刚开完需求会有点柴犬疑惑在复盘" in p_5b.split("【最近对话】", 1)[0]
    assert "桌上有一盏纸灯" in p_5b
    assert "柴犬歪头" in p_5b
    assert "橘猫打滚" not in p_5b


def test_e2e_omitted_turn_id_out_of_order_delivery_and_replay_subprocess(
    tmp_path: Path,
) -> None:
    """Out-of-order omitted-turn_id receipts (out-5b before out-5a, plus replay) match strictly by incoming_message_id."""
    res = _run_same_reply_omitted_turn_id_scenario(tmp_path, scenario="out_of_order_replay")
    prompts: list[str] = res["daily_prompts"]
    assert len(prompts) == 2, f"expected exactly 2 Step 2 prompts (replay deduped), got {len(prompts)}"
    assert res["inner_revisions"] == 2

    p_first_5b, p_second_5a = prompts[0], prompts[1]
    assert "刚开完需求会有点柴犬疑惑在复盘" in p_first_5b.split("【最近对话】", 1)[0]
    assert "桌上有一盏纸灯" in p_first_5b
    assert "柴犬歪头" in p_first_5b
    assert "橘猫打滚" not in p_first_5b

    assert "中午吃完饭准备橘猫滚一圈休息会儿" in p_second_5a.split("【最近对话】", 1)[0]
    assert "窗边有一盆薄荷" in p_second_5a
    assert "橘猫打滚" in p_second_5a
    assert "桌上有一盏纸灯" not in p_second_5a
    assert "柴犬歪头" not in p_second_5a
