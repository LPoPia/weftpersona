"""Regression tests for group authored-evidence isolation and paraphrase_ref storage.

Covers legacy undecorated/decorated receipts, authored pre-stripped receipts (including user-typed
bracketed text), quote isolation for events/impressions/commitments, paraphrase_ref strict
platform validation, and cross-repo real Hermes -> DeliveryReceipt -> group_ingest storage.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("VIVEMIND_HERMES_SOURCE"):
    HERMES_ROOT = Path(os.environ["VIVEMIND_HERMES_SOURCE"]).expanduser().resolve()
else:
    _spec = importlib.util.find_spec("hermes_constants")
    HERMES_ROOT = Path(_spec.origin).resolve().parent if (_spec and _spec.origin) else Path()

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins"))
if HERMES_ROOT.is_dir() and str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from sanae_mem.runtime import group_ingest, group_memory
from sanae_mem.infra import host_bridge
from sanae_mem.runtime import hooks as runtime_hooks
from sanae_mem.plugin_entry import SanaeMemProvider
from sanae_mem.runtime.hooks import register_private_heart_hooks

GROUP_QQ = "default/qqbot/102000001/G_QQ"
OWNER_QQ = group_memory.GROUP_OWNER_PREFIX + GROUP_QQ


class _Flash:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []
        self.llm = SimpleNamespace(complete=self.complete)

    def complete(self, messages, **_kwargs):
        self.calls.append({"messages": messages})
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


def _write_home(home: Path, overrides: dict, data_dir: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    lines = [
        "plugins:",
        "  sanae_mem:",
        f"    data_dir: {data_dir}",
        "    semantic_backend: sqlite",
        "    group_context:",
        "      defaults:",
        "        long_term_memory: false",
        "      overrides:",
    ]
    for key, enabled in overrides.items():
        lines += [f'        "{key}":', f"          long_term_memory: {str(enabled).lower()}"]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home


def _use_home(monkeypatch, home: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))


def _open_provider(home: Path, data_dir: Path, llm: _Flash, *, session_id: str = "s-grp"):
    provider = SanaeMemProvider(
        {"data_dir": str(data_dir), "semantic_backend": "sqlite", "shadow": False}
    )
    provider.initialize(session_id, hermes_home=str(home), platform="qqbot")
    provider._llm = llm.llm
    return provider


@contextmanager
def _scope_vars(
    *,
    platform: str = "qqbot",
    chat_id: str = "G_QQ",
    scope_id: str = "102000001",
    sender: str = "U_BOB",
    sender_name: str = "Bob",
    message_id: str = "msg-self",
    reply_to: str = "msg-quoted",
):
    with host_bridge.turn_envelope_scope(
        platform=platform,
        chat_type="group",
        chat_id=chat_id,
        sender_id=sender,
        sender_name=sender_name,
        sender_id_alt="",
        session_id="s-grp",
        session_key="qqbot:group:G_QQ",
        message_id=message_id,
        reply_to_message_id=reply_to,
        scope_id=scope_id,
        thread_id="",
        parent_chat_id="",
        thread_visibility="unknown",
        occurred_at="2026-11-18T10:05:00Z",
        is_bot=False,
    ):
        yield


def test_legacy_decorated_receipt_strips_prefixes_and_blocks_quoted_speaker_quotes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_dir = home / "sanae-mem"
    _write_home(home, {GROUP_QQ: True}, data_dir)
    _use_home(monkeypatch, home)

    quoted_qq = "Alice说不去"
    decorated_user = (
        "[Triggering message id: `msg-self` — use as `message_id` for reply/react/pin via the discord tools.]\n\n"
        "[Wed 2026-11-18 10:05 Asia/Tokyo]\n\n"
        "[Replying to: \"Alice说她明天不去加班\"]\n\n"
        "[Recent channel messages]\nAlice: 我明天不去加班，顺便带了蛋糕。\n\n[New message]\n"
        f"[Bob [Team | Alice] | qqbot user U_BOB] [Quoted message; length={len(quoted_qq)}]:\n{quoted_qq}\n\n"
        "收到，那明天蛋糕放我桌子上吧。"
    )
    flash = _Flash(
        {
            "events": [
                {"text": "Alice不去加班", "quote": "我明天不去加班，顺便带了蛋糕。", "paraphrase_ref": "msg-quoted"},
                {"text": "让明天把蛋糕放他桌子上", "quote": "那明天蛋糕放我桌子上吧。", "paraphrase_ref": "msg-quoted"},
                {"text": "坏转述编号", "quote": "收到，那明天蛋糕放我桌子上吧。", "paraphrase_ref": "msg-invented"},
                {"text": "引用了自己", "quote": "收到，那明天蛋糕放我桌子上吧。", "paraphrase_ref": "msg-self"},
                {"text": "引用了回复", "quote": "收到，那明天蛋糕放我桌子上吧。", "paraphrase_ref": "out-deliv-1"},
            ],
            "impression": {
                "speaker": "user",
                "text": "他不加班还总带蛋糕",
                "quote": "我明天不去加班",
            },
            "new_commitments": [],
            "commitment_updates": [],
        }
    )
    provider = _open_provider(home, data_dir, flash)
    try:
        commitment_id = provider.record_commitment(
            OWNER_QQ, "帮Bob盯一下明天蛋糕", source_quote="我明天帮Bob盯着蛋糕",
            turn_id="t0", now=100.0,
        )
        flash.payload["commitment_updates"] = [
            {"id": commitment_id, "action": "fulfilled", "basis": "user_confirmed", "quote": "我明天不去加班"},
        ]
        with _scope_vars(message_id="msg-self", reply_to="msg-quoted"):
            res = group_ingest.apply_group_delivery(
                provider,
                user_message=decorated_user,
                assistant_response="好的，帮你放好。",
                turn_id="t1",
                delivered_at=200.0,
                delivery_ref="out-deliv-1",
                user_message_authored=False,
            )
    finally:
        provider.shutdown()

    assert res["ok"] is True
    assert len(flash.calls) == 1
    prompt_user = flash.calls[0]["messages"][1]["content"]
    assert "收到，那明天蛋糕放我桌子上吧。" in prompt_user
    assert "Alice说她明天不去加班" not in prompt_user
    assert "我明天不去加班" not in prompt_user
    assert '"reply_to_message_id": "msg-quoted"' in prompt_user

    assert len(res["events"]) == 4  # Alice's quote rejected; 4 Bob-quoted events remain
    assert res["impression"] is None  # Quoted Alice line rejected for impression
    assert res["updates"] == []  # Quoted Alice line rejected for user_confirmed

    conn = sqlite3.connect(str(data_dir / "sanae.db"))
    rows = conn.execute(
        "SELECT id, raw, summary, source_context, fact_subject FROM events ORDER BY id ASC"
    ).fetchall()
    conn.close()
    by_id = {row[0]: row for row in rows}
    ev_valid = by_id[res["events"][0]]
    assert json.loads(ev_valid[3])["paraphrase_ref"] == "msg-quoted"
    assert ev_valid[4].startswith("owner:person:")

    for eid in res["events"][1:]:
        src = json.loads(by_id[eid][3])
        assert src.get("paraphrase_ref", "") == ""


def test_pure_quote_empty_authored_text_skips_model_and_writes_nothing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_dir = home / "sanae-mem"
    _write_home(home, {GROUP_QQ: True}, data_dir)
    _use_home(monkeypatch, home)

    flash = _Flash({"events": [], "impression": None, "new_commitments": [], "commitment_updates": []})
    provider = _open_provider(home, data_dir, flash)
    try:
        with _scope_vars():
            res_authored = group_ingest.apply_group_delivery(
                provider, user_message="   ", assistant_response="收到", turn_id="t-empty-1",
                user_message_authored=True,
            )
            res_legacy_quote = group_ingest.apply_group_delivery(
                provider, user_message="[Quoted message]:\n别人说的一大段话", assistant_response="收到",
                turn_id="t-empty-2", user_message_authored=False,
            )
            res_legacy_mismatch_len = group_ingest.apply_group_delivery(
                provider, user_message="[Quoted message; length=999]:\n短", assistant_response="收到",
                turn_id="t-empty-3", user_message_authored=False,
            )
    finally:
        provider.shutdown()

    assert res_authored["reason"] == "missing_turn_material"
    assert res_legacy_quote["reason"] == "missing_turn_material"
    assert res_legacy_mismatch_len["reason"] == "missing_turn_material"
    assert flash.calls == []


def test_authored_user_message_preserves_literal_brackets_verbatim(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data_dir = home / "sanae-mem"
    _write_home(home, {GROUP_QQ: True}, data_dir)
    _use_home(monkeypatch, home)

    literal = "[活动] 周六去爬山，集合点在东门"
    flash = _Flash(
        {
            "events": [{"text": "计划周六去爬山", "quote": literal}],
            "impression": {"speaker": "user", "text": "喜欢周末组织户外活动", "quote": literal},
            "new_commitments": [],
            "commitment_updates": [],
        }
    )
    provider = _open_provider(home, data_dir, flash)
    try:
        with _scope_vars(reply_to=""):
            res = group_ingest.apply_group_delivery(
                provider, user_message=literal, assistant_response="好，我也去！",
                turn_id="t-lit", delivered_at=300.0, delivery_ref="out-lit",
                user_message_authored=True,
            )
    finally:
        provider.shutdown()

    assert res["ok"] is True
    assert len(res["events"]) == 1
    assert res["impression"] == "喜欢周末组织户外活动"

    conn = sqlite3.connect(str(data_dir / "sanae.db"))
    raw = conn.execute("SELECT raw FROM events WHERE id = ?", (res["events"][0],)).fetchone()[0]
    conn.close()
    assert literal in raw


@pytest.mark.skipif(not HERMES_ROOT.is_dir(), reason="hermes worktree unavailable")
def test_cross_repo_qq_reply_receipt_writes_only_speaker_text_with_paraphrase_ref(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from hermes_host_contracts import (
        clear_all_contract_registrations,
        register_delivery_receipt_handler,
    )
    from plugins.platforms.qqbot.adapter import QQAdapter

    clear_all_contract_registrations()
    home = tmp_path / "home"
    data_dir = home / "sanae-mem"
    _write_home(home, {GROUP_QQ: True}, data_dir)
    _use_home(monkeypatch, home)

    flash = _Flash(
        {
            "events": [{"text": "确认周六下午带相机", "quote": "我周六一定带相机过去", "paraphrase_ref": "qq-quoted-99"}],
            "impression": None,
            "new_commitments": [],
            "commitment_updates": [],
        }
    )
    provider = _open_provider(home, data_dir, flash, session_id="s-e2e")
    manager = PluginManager(scope_key=str(home))
    ctx = PluginContext(PluginManifest(name="sanae_mem", source="user"), manager)

    captured_handlers = []

    def capture_registration(handler):
        captured_handlers.append(handler)
        return True

    with monkeypatch.context() as reg:
        reg.setenv("HERMES_HOME", str(home))
        reg.setattr(runtime_hooks, "register_delivery_receipt_handler", capture_registration)
        release = register_private_heart_hooks(ctx, provider)
    assert len(captured_handlers) == 1
    register_delivery_receipt_handler(captured_handlers[0])

    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "102000001"}))
    adapter._bot_openid = "BOT_QQ"

    async def _run():
        captured = []

        async def _fake_quote(_d):
            quoted = "Alice: 带上你的相机"
            return {
                "quote_block": f"[Quoted message; length={len(quoted)}]:\n{quoted}",
                "image_urls": [],
                "image_media_types": [],
            }

        async def _capture(ev):
            captured.append(ev)

        adapter._process_quoted_context = _fake_quote
        adapter.handle_message = _capture
        await adapter._ingest(
            {"message_reference": {"message_id": "qq-quoted-99"}},
            "qq-in-100",
            "我周六一定带相机过去",
            [],
            "2026-11-18T11:00:00Z",
            chat_id="G_QQ",
            qq_chat_type="group",
            chat_type="group",
            user_id="U_BOB",
            user_name="Bob",
        )
        ev = captured[0]
        runner = object.__new__(GatewayRunner)
        runner.config = SimpleNamespace(stt_enabled=False, image_vision_enabled=False)
        runner.adapters = {}
        runner._pending_native_image_paths = {}
        runner._should_echo_stt_transcripts = lambda: False
        runner._delivery_adapter_for = lambda _s: None
        ev._persist_user_message = await runner._prepare_inbound_message_text(
            event=ev, source=ev.source, history=[], session_key="agent:main:qqbot:group:G_QQ",
        )
        ev._clean_final_response = "好嘞，到时候见。"
        adapter._emit_final_delivery_receipt(
            ev, "agent:main:qqbot:group:G_QQ", "好嘞，到时候见。",
            SendResult(success=True, message_id="qq-out-200"),
        )

    try:
        asyncio.run(_run())
        deadline = time.monotonic() + 8.0
        rows = []
        while time.monotonic() < deadline:
            if (data_dir / "sanae.db").is_file():
                with sqlite3.connect(data_dir / "sanae.db") as conn:
                    rows = conn.execute("SELECT raw, source_context, fact_subject FROM events").fetchall()
                if rows:
                    break
            time.sleep(0.05)
    finally:
        release()
        provider.shutdown()
        clear_all_contract_registrations()

    assert len(flash.calls) == 1
    assert "带上你的相机" not in flash.calls[0]["messages"][1]["content"]
    assert "我周六一定带相机过去" in flash.calls[0]["messages"][1]["content"]
    assert len(rows) == 1
    assert "我周六一定带相机过去" in rows[0][0]
    assert "带上你的相机" not in rows[0][0]
    src = json.loads(rows[0][1])
    assert src["paraphrase_ref"] == "qq-quoted-99"
    assert src["ref"] == "qq-in-100"
    assert rows[0][2].startswith("owner:person:")
