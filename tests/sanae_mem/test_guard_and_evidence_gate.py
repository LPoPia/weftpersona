"""Real-entry integration tests for Issue 021 (quote-header stripping) and Issue 023 (resident memory routing)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from typing import Any
import yaml

import pytest

_HERMES_ROOT = Path(__file__).resolve().parents[3] / "multiplatform-batch-hermes"
if _HERMES_ROOT.is_dir() and str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

import hermes_constants as _hc
from hermes_host_contracts import (
    TurnEnvelope,
    get_memory_mutation_gate,
    turn_envelope_scope,
)


@contextmanager
def hermes_home_scope(home: Path):
    token = _hc.set_hermes_home_override(home)
    try:
        yield home
    finally:
        _hc.reset_hermes_home_override(token)
from tools.memory_tool import MemoryStore, memory_tool

from sanae_mem import plugin_entry
from sanae_mem.cognitive.daily_context import daily_profile_key
from sanae_mem.cognitive.inner_state_store import owner_state
from sanae_mem.governance.memory_evidence import (
    MODEL_INFERENCE,
    USER_EXTRACTED,
    capture_private_turn,
    qualify_user_text,
)
from sanae_mem.governance.source_context import current_chat_scope, resolve_chat_identity


def _write_home_config(home: Path, *, owner_id: str = "10001") -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "memories").mkdir(parents=True, exist_ok=True)
    (home / "memories" / "MEMORY.md").write_text("", encoding="utf-8")
    (home / "memories" / "USER.md").write_text("", encoding="utf-8")
    cfg = {
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
        "plugins": {
            "sanae_mem": {
                "enabled": True,
                "persona_name": "早苗",
                "memory_identity": {
                    "platform": "qqbot",
                    "user_id": owner_id,
                    "scope_id": "default",
                },
            }
        },
    }
    (home / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")


@contextmanager
def _fake_plugin_ctx(home: Path):
    hooks: dict[str, list[Any]] = {}
    unload: list[Any] = []
    state: dict[str, Any] = {"unload": unload}

    class _Ctx:
        def get_config(self):
            return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["plugins"]["sanae_mem"]

        def register_hook(self, name, fn):
            hooks.setdefault(name, []).append(fn)

        def register_tool(self, **_kwargs):
            return None

        def register_memory_provider(self, provider):
            state["provider"] = provider

        def on_unload(self, fn):
            unload.append(fn)

    with hermes_home_scope(home):
        plugin_entry.register(_Ctx())
    try:
        yield state
    finally:
        for fn in reversed(unload):
            fn()


def _private_envelope(session_id: str, *, owner_id: str = "10001", message_id: str = "m-1") -> TurnEnvelope:
    return TurnEnvelope(
        platform="qqbot",
        chat_type="dm",
        chat_id=owner_id,
        sender_id=owner_id,
        scope_id="default",
        session_id=session_id,
        session_key=f"agent:main:qqbot:dm:{owner_id}",
        message_id=message_id,
        turn_id=f"turn-{message_id}",
        write_origin="foreground",
        is_messaging_surface=True,
    )


def test_evidence_gate_strips_multi_clause_reply_and_qq_quote_headers_via_capture_turn(tmp_path: Path):
    home = tmp_path / "home"
    _write_home_config(home, owner_id="10001")
    cfg = {"memory_identity": {"platform": "qqbot", "user_id": "10001", "scope_id": "default"}}

    with _fake_plugin_ctx(home) as state:
        provider = state["provider"]
        env = _private_envelope("s-quote-1", owner_id="10001", message_id="msg-1")
        with hermes_home_scope(home), turn_envelope_scope(env):
            provider.initialize("s-quote-1", hermes_home=str(home), platform="qqbot", agent_context="primary")
        try:
            def capture(text: str, session_id: str, message_id: str):
                scope = current_chat_scope()
                identity = provider.sqlite.read(
                    lambda conn: resolve_chat_identity(scope, cfg, connection=conn)
                )
                return capture_private_turn(
                    text, current_message={"role": "user", "content": text, "platform_message_id": message_id},
                    session_id=session_id, platform="qqbot", scope=scope, identity=identity,
                    author_id="10001",
                )

            # 1. [Replying to your previous message: "..."] with 3+ clauses
            reply_msg = '[Replying to your previous message: "第一句无关。其实我最喜欢抹茶拿铁。第三句无关"]\n\n好的。我平时习惯喝美式'
            with hermes_home_scope(home), turn_envelope_scope(env):
                evidence = capture(reply_msg, "s-quote-1", "msg-1")
            assert evidence is not None and evidence.available is True
            quoted_verdict = qualify_user_text("其实我最喜欢抹茶拿铁", evidence)
            assert quoted_verdict.allowed is False
            assert quoted_verdict.status == MODEL_INFERENCE

            authored_verdict = qualify_user_text("我平时习惯喝美式", evidence)
            assert authored_verdict.allowed is True
            assert authored_verdict.status == USER_EXTRACTED

            # 2. QQ [Quoted message; length=N]:\n...
            quoted_body = "第一句无关。其实我最喜欢抹茶拿铁。第三句无关"
            qq_msg = f"[Quoted message; length={len(quoted_body)}]:\n{quoted_body}\n\n我周末常去图书馆"
            env_qq = _private_envelope("s-quote-2", owner_id="10001", message_id="msg-2")
            with hermes_home_scope(home), turn_envelope_scope(env_qq):
                provider.initialize("s-quote-2", hermes_home=str(home), platform="qqbot", agent_context="primary")
                evidence_qq = capture(qq_msg, "s-quote-2", "msg-2")
            assert evidence_qq is not None and evidence_qq.available is True
            assert qualify_user_text("其实我最喜欢抹茶拿铁", evidence_qq).allowed is False
            qq_authored = qualify_user_text("我周末常去图书馆", evidence_qq)
            assert qq_authored.allowed is True
            assert qq_authored.status == USER_EXTRACTED

            # 3. Nested '"]\n\n' inside quoted reply text must not leak inner quoted clause
            nested_msg = '[Replying to your previous message: "示例["a"]\n\n其实我最喜欢抹茶拿铁"]\n\n好的。我平时习惯喝美式'
            env_nested = _private_envelope("s-nested-1", owner_id="10001", message_id="msg-nested")
            with hermes_home_scope(home), turn_envelope_scope(env_nested):
                provider.initialize("s-nested-1", hermes_home=str(home), platform="qqbot", agent_context="primary")
                ev_nested = capture(nested_msg, "s-nested-1", "msg-nested")
            assert ev_nested is not None and ev_nested.available is True
            assert qualify_user_text("其实我最喜欢抹茶拿铁", ev_nested).allowed is False
            nested_authored = qualify_user_text("我平时习惯喝美式", ev_nested)
            assert nested_authored.allowed is True
            assert nested_authored.status == USER_EXTRACTED

            # 4. Malformed / unclosed or length-overflow quote headers fail closed
            malformed_cases = [
                '[Replying to your previous message: "第一句。其实我最喜欢抹茶拿铁。第三句',
                "[Quoted message; length=999]:\n第一句。其实我最喜欢抹茶拿铁",
                "[Quoted message]:\n其实我最喜欢抹茶拿铁",
            ]
            for idx, bad_text in enumerate(malformed_cases, start=10):
                env_bad = _private_envelope(f"s-bad-{idx}", owner_id="10001", message_id=f"msg-{idx}")
                with hermes_home_scope(home), turn_envelope_scope(env_bad):
                    provider.initialize(f"s-bad-{idx}", hermes_home=str(home), platform="qqbot", agent_context="primary")
                    ev_bad = capture(bad_text, f"s-bad-{idx}", f"msg-{idx}")
                assert ev_bad is not None
                assert qualify_user_text("其实我最喜欢抹茶拿铁", ev_bad).allowed is False
        finally:
            provider.shutdown()


def test_memory_tool_routes_self_review_and_trivia_to_sanae_db_with_multi_home_isolation(tmp_path: Path):
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    _write_home_config(home_a, owner_id="10001")
    _write_home_config(home_b, owner_id="20002")

    def fake_llm(messages, **_kwargs):
        user_payload = json.loads(messages[-1]["content"])
        proposed = " ".join(user_payload.get("proposed_entries") or [])
        if "以后不能" in proposed or "说话规矩" in proposed:
            return SimpleNamespace(text=json.dumps({
                "route": "inner_state",
                "inner_thought": "刚才有点急着接话，心里还是有点过意不去。",
                "event_fact": None,
                "reason": "自我检讨与说话规矩",
            }, ensure_ascii=False))
        if "看课表" in proposed:
            return SimpleNamespace(text=json.dumps({
                "route": "inner_state",
                "inner_thought": "把课表看岔了，下次得更仔细。",
                "event_fact": "今天帮他看课表时，我把周四的课看成了周三",
                "reason": "单次日常小事",
            }, ensure_ascii=False))
        if "午饭" in proposed:
            return SimpleNamespace(text=json.dumps({
                "route": "inner_state",
                "inner_thought": "",
                "event_fact": "今天中午他告诉我午饭吃了番茄牛腩面",
                "reason": "纯单次日常事实，无特殊心绪",
            }, ensure_ascii=False))
        return SimpleNamespace(text=json.dumps({
            "route": "allow",
            "inner_thought": "",
            "event_fact": None,
            "reason": "长期关系里程碑",
        }, ensure_ascii=False))

    with _fake_plugin_ctx(home_a) as state_a, _fake_plugin_ctx(home_b) as state_b:
        provider_a = state_a["provider"]
        provider_b = state_b["provider"]
        provider_a.initialize("sess-a", hermes_home=str(home_a), platform="qqbot", agent_context="primary")
        provider_b.initialize("sess-b", hermes_home=str(home_b), platform="qqbot", agent_context="primary")
        provider_a._llm = SimpleNamespace(complete=fake_llm)
        provider_b._llm = SimpleNamespace(complete=fake_llm)

        try:
            env_a = _private_envelope("sess-a", owner_id="10001", message_id="m-a1")
            with hermes_home_scope(home_a), turn_envelope_scope(env_a):
                assert get_memory_mutation_gate() is not None
                store_a = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                store_a.load_from_disk()

                # 1. Self-criticism -> blocked from MEMORY.md, written to inner_state_revisions
                res_crit = json.loads(memory_tool(
                    action="add",
                    target="memory",
                    content="以后不能再顺着猜他的意思了，这是说话规矩。",
                    store=store_a,
                ))
                assert res_crit["success"] is False
                assert "不写入长期常驻文件" in res_crit["error"]
                assert (home_a / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""

                db_a = home_a / "sanae-mem" / "sanae.db"
                with sqlite3.connect(str(db_a)) as conn:
                    owner_in_db = conn.execute(
                        "SELECT owner_key FROM inner_state_revisions WHERE turn_id='turn-m-a1'"
                    ).fetchone()[0]
                st_a = owner_state(db_a, daily_profile_key(home_a), owner_in_db)
                assert st_a is not None, f"res_crit={res_crit}"
                assert st_a.text == "刚才有点急着接话，心里还是有点过意不去。", f"res_crit={res_crit}, st_a={st_a}"

                # 2. Single-turn trivia -> blocked from MEMORY.md, short 14-day event written to events
                res_trivia = json.loads(memory_tool(
                    action="add",
                    target="memory",
                    content="今天帮他看课表时把周四的课看成周三了。",
                    store=store_a,
                ))
                assert res_trivia["success"] is False
                assert "短期经历" in res_trivia["error"]
                assert (home_a / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""

                with sqlite3.connect(str(db_a)) as conn:
                    rows = conn.execute(
                        "SELECT raw, importance, ttl_expire, owner_key FROM events"
                    ).fetchall()
                assert len(rows) == 1
                raw_ev, importance, ttl_expire, owner_key = rows[0]
                assert "周四的课看成了周三" in raw_ev
                assert importance == pytest.approx(0.5)
                assert owner_key == owner_in_db
                ttl_days = (datetime.fromisoformat(ttl_expire) - datetime.now().astimezone()).total_seconds() / 86400
                assert 13 < ttl_days < 15

                # 2b. Single-turn trivia with empty inner_thought -> inner_state unchanged, short event still written
                with sqlite3.connect(str(db_a)) as conn:
                    rev_before = conn.execute("SELECT MAX(revision) FROM inner_state_revisions").fetchone()[0]
                res_trivia_only = json.loads(memory_tool(
                    action="add",
                    target="memory",
                    content="他今天午饭吃了番茄牛腩面。",
                    store=store_a,
                ))
                assert res_trivia_only["success"] is False
                assert "短期经历" in res_trivia_only["error"]
                assert (home_a / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""
                with sqlite3.connect(str(db_a)) as conn:
                    rev_after = conn.execute("SELECT MAX(revision) FROM inner_state_revisions").fetchone()[0]
                    ev_rows = conn.execute("SELECT raw FROM events ORDER BY id").fetchall()
                assert rev_after == rev_before
                assert len(ev_rows) == 2
                assert ev_rows[0][0].endswith("今天帮他看课表时，我把周四的课看成了周三")
                assert ev_rows[1][0].endswith("今天中午他告诉我午饭吃了番茄牛腩面")

                # 3. Milestone (route="allow") -> written to MEMORY.md
                res_allow = json.loads(memory_tool(
                    action="add",
                    target="memory",
                    content="我们约定好每年秋天一起去看银杏。",
                    store=store_a,
                ))
                assert res_allow["success"] is True
                assert "我们约定好每年秋天一起去看银杏。" in (home_a / "memories" / "MEMORY.md").read_text(encoding="utf-8")

            # 4. Multi-HOME isolation: switch to home_b, route failure must write home_b status.json only
            provider_b._llm = SimpleNamespace(complete=lambda *_, **__: (_ for _ in ()).throw(RuntimeError("flash down")))
            env_b = _private_envelope("sess-b", owner_id="20002", message_id="m-b1")
            with hermes_home_scope(home_b), turn_envelope_scope(env_b):
                store_b = MemoryStore(memory_char_limit=2200, user_char_limit=1375)
                store_b.load_from_disk()
                res_b = json.loads(memory_tool(
                    action="add",
                    target="memory",
                    content="以后不能再多嘴了。",
                    store=store_b,
                ))
                assert res_b["success"] is False
                assert (home_b / "memories" / "MEMORY.md").read_text(encoding="utf-8") == ""
                status_a = json.loads((home_a / "sanae-mem" / "status.json").read_text(encoding="utf-8"))
                status_b = json.loads((home_b / "sanae-mem" / "status.json").read_text(encoding="utf-8"))
                assert status_a["components"]["resident_memory_route"]["ok"] is True
                assert status_b["components"]["resident_memory_route"]["ok"] is False
                with sqlite3.connect(str(home_b / "sanae-mem" / "sanae.db")) as conn_b:
                    assert conn_b.execute("SELECT count(*) FROM events").fetchone()[0] == 0
        finally:
            provider_a.shutdown()
            provider_b.shutdown()
