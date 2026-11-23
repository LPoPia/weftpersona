"""Real cold-start & private delivery settlement integration tests."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _harness import (  # noqa: E402
    resolve_vivemind_and_hermes_roots,
    run_clean_subprocess,
    write_home_config,
)


def _write_config(home: Path, *, user_id: str = "1001", model: str = "test-model") -> None:
    write_home_config(
        home,
        {
            "model": model,
            "plugins": {
                "enabled": ["sanae_mem"],
                "sanae_mem": {
                    "data_dir": str(home / "sanae-mem"),
                    "semantic_backend": "sqlite",
                    "memory_identity": {"platform": "telegram", "user_id": user_id},
                },
            },
        },
    )


def test_cold_start_discover_plugins_mounts_ephemeral_private_heart_in_fresh_subprocess(tmp_path):
    """Fresh subprocess with no pre-imported agent.memory_manager mounts append_private_heart as ephemeral."""
    home = tmp_path / "hermes"
    home.mkdir()
    _write_config(home)
    vivemind_root, hermes_root = resolve_vivemind_and_hermes_roots()

    script = f"""
import json, os, sys
from pathlib import Path

assert "agent.memory_manager" not in sys.modules, "must start cold without agent.memory_manager"
os.environ["HERMES_HOME"] = {str(home)!r}
from _harness import install_plugin_copy
install_plugin_copy(Path({str(home)!r}), Path({str(vivemind_root)!r}), plugin_name="sanae_mem")

from hermes_cli.plugins import discover_plugins, get_plugin_manager
from hermes_host_contracts import TurnEnvelope, turn_envelope_scope
from agent.turn_context import build_api_messages, compose_user_api_content

discover_plugins()
manager = get_plugin_manager()

envelope = TurnEnvelope(
    platform="telegram",
    chat_type="dm",
    chat_id="1001",
    sender_id="1001",
    session_id="cold-dm-1",
    session_key="agent:main:telegram:dm:1001",
    message_id="msg-1",
    turn_id="turn-1",
    is_messaging_surface=True,
    write_origin="foreground",
)
with turn_envelope_scope(envelope):
    results = manager.invoke_hook(
        "pre_llm_call",
        session_id="cold-dm-1",
        turn_id="turn-1",
        platform="telegram",
        user_message="今天下午我要去图书馆看书",
    )

ephemeral_blocks = [
    r for r in results
    if isinstance(r, dict) and r.get("ephemeral") is True and "[我眼中的世界]" in str(r.get("context") or "")
]
assert len(ephemeral_blocks) == 1, f"expected 1 ephemeral private heart block, got: {{results!r}}"
heart = ephemeral_blocks[0]["context"]
assert "[我内心的感受]" in heart

# Verify wire vs durable api_content separation across turns
durable_sidecar = compose_user_api_content("今天下午我要去图书馆看书", "", "")
assert durable_sidecar is None, "ephemeral heart must not land in api_content sidecar"

class _FakeAgent:
    _current_turn_timestamp = 1700000000.0
    ephemeral_system_prompt = ""
    model = "test-model"
    provider = "custom"
    base_url = ""
    def _copy_reasoning_content_for_api(self, msg, api_msg): pass
    def _should_sanitize_tool_calls(self): return False

turn1_msg = {{"role": "user", "content": "今天下午我要去图书馆看书"}}
wire1, _ = build_api_messages(
    _FakeAgent(),
    [turn1_msg],
    current_turn_user_idx=0,
    ext_prefetch_cache="",
    plugin_user_context="",
    moa_config=None,
    active_system_prompt="SYS",
    ephemeral_user_context=heart,
)
assert "[我眼中的世界]" in wire1[1]["content"] and "[我内心的感受]" in wire1[1]["content"]
assert "[我眼中的世界]" not in turn1_msg["content"], "persisted history must stay clean"

turn2_msg = {{"role": "user", "content": "晚上再聊"}}
wire2, _ = build_api_messages(
    _FakeAgent(),
    [turn1_msg, {{"role": "assistant", "content": "好呀，祝你看书开心"}}, turn2_msg],
    current_turn_user_idx=2,
    ext_prefetch_cache="",
    plugin_user_context="",
    moa_config=None,
    active_system_prompt="SYS",
    ephemeral_user_context=heart,
)
assert "[我眼中的世界]" not in wire2[1]["content"], "turn 1 history in turn 2 wire must have zero stale block"
assert "[我眼中的世界]" in wire2[3]["content"], "turn 2 current message in wire must have live block"
print(json.dumps({{"ok": True}}))
"""
    assert run_clean_subprocess(
        script, vivemind_root=vivemind_root, hermes_root=hermes_root, home=home, timeout=30,
    ) == {"ok": True}


def test_verified_private_delivery_validation():
    from sanae_mem.runtime.private_delivery_settler import VerifiedPrivateDelivery

    with pytest.raises(ValueError):
        VerifiedPrivateDelivery(
            session_id="",
            turn_id="t1",
            platform="telegram",
            chat_id="1001",
            chat_type="dm",
            sender_id="1001",
            user_message="hi",
            assistant_response="hello",
            delivered_at=time.time(),
        )
    with pytest.raises(ValueError):
        VerifiedPrivateDelivery(
            session_id="s1",
            turn_id="t1",
            platform="telegram",
            chat_id="1001",
            chat_type="dm",
            sender_id="1001",
            user_message="hi",
            assistant_response="hello",
            delivered_at=0.0,
        )


def test_messaging_sync_turn_blocked_and_receipt_settles_all_tables_idempotently(tmp_path, monkeypatch):
    """Messaging surface sync_turn writes 0 rows; real DeliveryReceipt settles events, daily_context, commitments, inner_state."""
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_host_contracts import (
        DeliveryReceipt,
        TurnEnvelope,
        emit_delivery_receipt,
        turn_envelope_scope,
    )
    from sanae_mem.cognitive import inner_wiring
    from sanae_mem import plugin_entry
    from sanae_mem.runtime import hooks as runtime_hooks

    home = tmp_path / "hermes"
    home.mkdir()
    _write_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    provider = plugin_entry.SanaeMemProvider(
        {
            "data_dir": str(home / "sanae-mem"),
            "semantic_backend": "sqlite",
            "memory_identity": {"platform": "telegram", "user_id": "1001"},
        }
    )

    def fake_complete(messages, *, purpose="", **kwargs):
        system = str(messages[0].get("content") or "") if messages else ""
        if purpose == "commitments_update" or "待办约定与承诺" in system:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "new_commitments": [
                            {
                                "text": "晚上帮忙查资料",
                                "quote": "我晚上帮你查一下资料。",
                                "masked_text": "帮忙查资料",
                            }
                        ],
                        "resolutions": [],
                    },
                    ensure_ascii=False,
                )
            )
        if purpose == "daily_context_extract":
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "items": [
                            {
                                "kind": "DAILY_FACT",
                                "slot": "afternoon_meeting",
                                "action": "set",
                                "text": "维护者今天下午要开会",
                                "quote": "我今天下午要去开会",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        return SimpleNamespace(
            text=json.dumps(
                {
                    "sentences": [
                        "记住他今天下午要去开会，心里记挂着这事。",
                        "晚上记得帮他把资料查好。",
                    ],
                    "variables": {"tension": 0.1, "closeness": 0.7, "energy": 0.2},
                    "shared_sentences": None,
                },
                ensure_ascii=False,
            )
        )

    provider._llm = SimpleNamespace(complete=fake_complete)

    envelope = TurnEnvelope(
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
    with turn_envelope_scope(envelope):
        provider.initialize("dm-s1", platform="telegram", hermes_home=str(home))
        # Even if sync_turn is called pre-delivery on a messaging surface, it must not write
        provider.sync_turn(
            "我很信任你，记住我今天下午要去开会！",
            "好的，我晚上帮你查一下资料。",
            session_id="dm-s1",
        )

    db_path = home / "sanae-mem" / "sanae.db"
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM daily_context").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM commitments").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM inner_state_revisions").fetchone()[0] == 0

    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)
    captured: list = []

    def capture_registration(handler):
        captured.append(handler)
        return True

    monkeypatch.setattr(runtime_hooks, "register_delivery_receipt_handler", capture_registration)
    release = runtime_hooks.register_private_heart_hooks(ctx, provider)
    assert len(captured) == 1
    handler = captured[0]

    receipt = DeliveryReceipt(
        session_id="dm-s1",
        platform="telegram",
        chat_id="1001",
        chat_type="dm",
        sender_id="1001",
        turn_id="turn-1",
        message_id="out-1",
        incoming_message_id="in-1",
        session_key="agent:main:telegram:dm:1001",
        user_message="我很信任你，记住我今天下午要去开会！",
        delivered_text="好的，我晚上帮你查一下资料。",
        clean_text="好的，我晚上帮你查一下资料。",
        delivered_at=time.time(),
        is_bot=False,
        transcript=(
            {"role": "user", "content": "我很信任你，记住我今天下午要去开会！", "platform_message_id": "in-1"},
            {"role": "assistant", "content": "好的，我晚上帮你查一下资料。"},
        ),
    )

    try:
        # Emit twice to verify idempotency
        handler(receipt)
        handler(receipt)
        assert inner_wiring.wait_for_pending_updates(timeout=10.0)
    finally:
        release()
        provider.shutdown()

    with closing(sqlite3.connect(db_path)) as conn:
        events_count = conn.execute("SELECT count(*) FROM events").fetchone()[0]
        daily_count = conn.execute("SELECT count(*) FROM daily_context").fetchone()[0]
        commitments_count = conn.execute("SELECT count(*) FROM commitments").fetchone()[0]
        inner_count = conn.execute("SELECT count(*) FROM inner_state_revisions").fetchone()[0]
        assert events_count == 1, f"expected 1 event, got {events_count}"
        assert daily_count == 1, f"expected 1 daily_context, got {daily_count}"
        assert commitments_count == 1, f"expected 1 commitment, got {commitments_count}"
        assert inner_count == 1, f"expected 1 inner_state_revision, got {inner_count}"
