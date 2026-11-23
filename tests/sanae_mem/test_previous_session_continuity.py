"""Tests for cross-session first-turn [我眼中的世界] continuity and QQ DM prompt denoise."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_ROOT = Path(__file__).resolve().parents[2] / "plugins"
for _p in (str(TESTS_ROOT), str(PLUGINS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _harness import resolve_vivemind_and_hermes_roots, run_clean_subprocess  # noqa: E402
from sanae_mem.cognitive.previous_session import PreviousSessionQuery, read_previous_session_tail  # noqa: E402
from sanae_mem.infra.chat_scope import ChatScope  # noqa: E402
from sanae_mem.infra.textutil import _owner_key  # noqa: E402


def _create_state_db(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                origin_json TEXT,
                title TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                last_activity_at REAL,
                hidden INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                _compressed_summary INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                api_content TEXT,
                display_kind TEXT,
                timestamp REAL NOT NULL
            );
            """
        )


def _insert_session(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    source: str = "qqbot",
    user_id: str = "u-owner",
    chat_id: str = "u-owner",
    chat_type: str = "dm",
    session_key: str | None = None,
    origin_json: str | None = None,
    title: str | None = None,
    parent_session_id: str | None = None,
    started_at: float,
    ended_at: float | None,
    end_reason: str | None = "daily",
    last_activity_at: float | None = None,
) -> None:
    if session_key is None:
        session_key = f"agent:main:{source}:{chat_type}:{chat_id}"
    if origin_json is None:
        origin_json = json.dumps(
            {
                "platform": source,
                "chat_type": chat_type,
                "chat_id": chat_id,
                "user_id": user_id,
            }
        )
    conn.execute(
        """
        INSERT INTO sessions(
            id, source, user_id, session_key, chat_id, chat_type, origin_json, title,
            parent_session_id, started_at, ended_at, end_reason, last_activity_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            source,
            user_id,
            session_key,
            chat_id,
            chat_type,
            origin_json,
            title,
            parent_session_id,
            started_at,
            ended_at,
            end_reason,
            last_activity_at,
        ),
    )


def _insert_message(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    role: str,
    content: str,
    timestamp: float,
    tool_call_id: str | None = None,
    tool_calls: str | None = None,
    compressed_summary: int = 0,
    display_kind: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO messages(
            session_id, role, content, tool_call_id, tool_calls,
            _compressed_summary, display_kind, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            tool_call_id,
            tool_calls,
            compressed_summary,
            display_kind,
            timestamp,
        ),
    )


def _private_scope(
    *,
    platform: str = "qqbot",
    actor: str = "u-owner",
    chat_id: str = "u-owner",
    session_id: str = "sess-current",
) -> ChatScope:
    return ChatScope(
        visibility="private",
        platform=platform,
        actor=actor,
        chat_type="dm",
        chat_id=chat_id,
        session_id=session_id,
        session_key=f"agent:main:{platform}:dm:{chat_id}",
    )


def test_previous_session_tail_24h_hit_filtering_and_turn2_skip(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    home = tmp_path / "home"
    home.mkdir()
    db_path = home / "state.db"
    _create_state_db(db_path)

    owner = _owner_key("qqbot", user_id="u-owner")
    with sqlite3.connect(str(db_path)) as conn:
        _insert_session(
            conn,
            session_id="sess-ancient",
            started_at=now - 50 * 3600,
            ended_at=now - 48 * 3600,
            last_activity_at=now - 48 * 3600,
        )
        # Daily-reset predecessor has parent_session_id pointing to sess-ancient!
        _insert_session(
            conn,
            session_id="sess-prev",
            parent_session_id="sess-ancient",
            started_at=now - 10 * 3600,
            ended_at=now - 2 * 3600,
            last_activity_at=now - 3 * 3600,
        )
        _insert_session(
            conn,
            session_id="sess-current",
            parent_session_id="sess-prev",
            started_at=now - 10,
            ended_at=None,
            end_reason=None,
        )
        _insert_message(conn, session_id="sess-prev", role="user", content="更早的第一轮", timestamp=now - 300)
        _insert_message(conn, session_id="sess-prev", role="assistant", content="更早的回复", timestamp=now - 290)
        _insert_message(
            conn,
            session_id="sess-prev",
            role="user",
            content="[时间：2026-11-23 周一 23:00]\n今晚打算几点睡？\n\n[我眼中的世界]\n旧世界\n\n[我内心的感受]\n旧感受",
            timestamp=now - 200,
        )
        _insert_message(
            conn,
            session_id="sess-prev",
            role="assistant",
            content="",
            tool_calls='[{"id": "call_1"}]',
            timestamp=now - 195,
        )
        _insert_message(
            conn,
            session_id="sess-prev",
            role="tool",
            content="tool output",
            tool_call_id="call_1",
            timestamp=now - 194,
        )
        _insert_message(
            conn,
            session_id="sess-prev",
            role="assistant",
            content="十二点前睡，明天还要早起。",
            timestamp=now - 190,
        )
        _insert_message(
            conn,
            session_id="sess-prev",
            role="assistant",
            content="压缩摘要",
            compressed_summary=1,
            timestamp=now - 185,
        )
        _insert_message(
            conn,
            session_id="sess-prev",
            role="assistant",
            content="隐藏消息",
            display_kind="hidden",
            timestamp=now - 184,
        )
        _insert_message(
            conn,
            session_id="sess-prev",
            role="assistant",
            content="失败消息",
            display_kind="failed_turn",
            timestamp=now - 183,
        )
        _insert_message(
            conn,
            session_id="sess-prev",
            role="user",
            content="那我明天早上叫你",
            timestamp=now - 180,
        )
        _insert_message(
            conn,
            session_id="sess-prev",
            role="assistant",
            content="好呀，明天见。",
            timestamp=now - 170,
        )

    query = PreviousSessionQuery(
        home=home,
        current_session_id="sess-current",
        current_owner=owner,
        read_owners=(owner,),
        scope=_private_scope(session_id="sess-current"),
        is_maintainer=True,
        partner_name="测试伴侣",
        now_ts=now,
    )
    block = read_previous_session_tail(query)
    assert block.startswith("〔上回聊到的末尾〕\n")
    assert "今晚打算几点睡？" in block
    assert "十二点前睡，明天还要早起。" in block
    assert "那我明天早上叫你" in block
    assert "好呀，明天见。" in block
    assert "更早的第一轮" not in block
    for leaked in ("旧世界", "旧感受", "tool output", "压缩摘要", "隐藏消息", "失败消息", "[时间："):
        assert leaked not in block

    # Turn 2: once current session has a completed assistant reply, do not re-inject.
    with sqlite3.connect(str(db_path)) as conn:
        _insert_message(conn, session_id="sess-current", role="user", content="早安", timestamp=now - 5)
        _insert_message(conn, session_id="sess-current", role="assistant", content="早呀", timestamp=now - 4)
    assert read_previous_session_tail(query) == ""


def test_previous_session_tail_window_owner_and_surface_isolation(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    now = 1_800_000_000.0
    home = tmp_path / "home"
    home.mkdir()

    # Missing state.db returns empty cleanly without error.
    owner_a = _owner_key("qqbot", user_id="user-a")
    scope_a = _private_scope(actor="user-a", chat_id="user-a", session_id="sess-cur")
    assert (
        read_previous_session_tail(
            PreviousSessionQuery(
                home=home,
                current_session_id="sess-cur",
                current_owner=owner_a,
                read_owners=(owner_a,),
                scope=scope_a,
                is_maintainer=False,
                partner_name="",
                now_ts=now,
            )
        )
        == ""
    )

    db_path = home / "state.db"
    # Empty/missing-table DB logs warning and returns "".
    sqlite3.connect(str(db_path)).close()
    with caplog.at_level(logging.WARNING):
        assert (
            read_previous_session_tail(
                PreviousSessionQuery(
                    home=home,
                    current_session_id="sess-cur",
                    current_owner=owner_a,
                    read_owners=(owner_a,),
                    scope=scope_a,
                    is_maintainer=False,
                    partner_name="",
                    now_ts=now,
                )
            )
            == ""
        )
    assert "previous session tail read failed" in caplog.text

    db_path.unlink()
    _create_state_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        # 1) > 24h expired session for user-a
        _insert_session(
            conn,
            session_id="sess-expired",
            user_id="user-a",
            chat_id="user-a",
            started_at=now - 30 * 3600,
            ended_at=now - 25 * 3600,
            last_activity_at=now - 25 * 3600,
        )
        _insert_message(conn, session_id="sess-expired", role="user", content="超时对话", timestamp=now - 25 * 3600)
        # 2) Cross-owner session (user-b) within 24h
        _insert_session(
            conn,
            session_id="sess-other-user",
            user_id="user-b",
            chat_id="user-b",
            started_at=now - 3600,
            ended_at=now - 1800,
            last_activity_at=now - 1800,
        )
        _insert_message(conn, session_id="sess-other-user", role="user", content="别人的秘密", timestamp=now - 1800)
        # 3) Group chat session within 24h
        _insert_session(
            conn,
            session_id="sess-group",
            user_id="user-a",
            chat_id="group-1",
            chat_type="group",
            started_at=now - 3600,
            ended_at=now - 1200,
            last_activity_at=now - 1200,
        )
        _insert_message(conn, session_id="sess-group", role="user", content="群聊内容", timestamp=now - 1200)
        # 4) Cron session within 24h
        _insert_session(
            conn,
            session_id="sess-cron",
            source="cron",
            user_id="user-a",
            chat_id="user-a",
            started_at=now - 3600,
            ended_at=now - 900,
            last_activity_at=now - 900,
        )
        _insert_message(conn, session_id="sess-cron", role="assistant", content="定时自言自语", timestamp=now - 900)
        # 5) T: isolated test session within 24h (by id, by title, and by T:-only user messages)
        _insert_session(
            conn,
            session_id="T:probe-1",
            user_id="user-a",
            chat_id="user-a",
            started_at=now - 3600,
            ended_at=now - 600,
            last_activity_at=now - 600,
        )
        _insert_message(conn, session_id="T:probe-1", role="user", content="T: 测试探针消息", timestamp=now - 600)
        _insert_session(
            conn,
            session_id="20261124_010000_abcd",
            user_id="user-a",
            chat_id="user-a",
            title="T: 线上测试",
            started_at=now - 3500,
            ended_at=now - 500,
            last_activity_at=now - 500,
        )
        _insert_message(conn, session_id="20261124_010000_abcd", role="user", content="普通探针问题", timestamp=now - 510)
        _insert_message(conn, session_id="20261124_010000_abcd", role="assistant", content="测试会话的回复不应透出", timestamp=now - 505)
        _insert_session(
            conn,
            session_id="20261124_005500_efgh",
            user_id="user-a",
            chat_id="user-a",
            started_at=now - 3400,
            ended_at=now - 450,
            last_activity_at=now - 450,
        )
        _insert_message(conn, session_id="20261124_005500_efgh", role="user", content="[时间：2026-11-24 周二 00:55]\nT: 无标题测试问题", timestamp=now - 460)
        _insert_message(conn, session_id="20261124_005500_efgh", role="assistant", content="孤立测试回复也不应透出", timestamp=now - 455)

    assert (
        read_previous_session_tail(
            PreviousSessionQuery(
                home=home,
                current_session_id="sess-cur",
                current_owner=owner_a,
                read_owners=(owner_a,),
                scope=scope_a,
                is_maintainer=False,
                partner_name="",
                now_ts=now,
            )
        )
        == ""
    )

    # Bound read_owners account on another platform IS allowed when explicitly in read_owners.
    bound_tg_owner = _owner_key("telegram", user_id="tg-a")
    with sqlite3.connect(str(db_path)) as conn:
        _insert_session(
            conn,
            session_id="sess-tg-bound",
            source="telegram",
            user_id="tg-a",
            chat_id="tg-a",
            started_at=now - 3600,
            ended_at=now - 300,
            last_activity_at=now - 300,
        )
        _insert_message(conn, session_id="sess-tg-bound", role="user", content="我刚下班到家", timestamp=now - 350)
        _insert_message(conn, session_id="sess-tg-bound", role="assistant", content="快去歇会儿吧", timestamp=now - 340)

    bound_tail = read_previous_session_tail(
        PreviousSessionQuery(
            home=home,
            current_session_id="sess-cur",
            current_owner=owner_a,
            read_owners=(owner_a, bound_tg_owner),
            scope=scope_a,
            is_maintainer=True,
            partner_name="测试伴侣",
            now_ts=now,
        )
    )
    assert "我刚下班到家" in bound_tail
    assert "快去歇会儿吧" in bound_tail


def test_previous_session_tail_not_crowded_out_by_other_owners(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    home = tmp_path / "home"
    home.mkdir()
    db_path = home / "state.db"
    _create_state_db(db_path)

    owner = _owner_key("qqbot", user_id="u-owner")
    scope = _private_scope(actor="u-owner", chat_id="u-owner", session_id="sess-current")
    with sqlite3.connect(str(db_path)) as conn:
        # 1 older ended DM for the current owner within 24h
        _insert_session(
            conn,
            session_id="sess-owner-old",
            user_id="u-owner",
            chat_id="u-owner",
            started_at=now - 7200,
            ended_at=now - 3600,
            last_activity_at=now - 3600,
        )
        _insert_message(conn, session_id="sess-owner-old", role="user", content="我刚煮了一锅罗宋汤", timestamp=now - 3620)
        _insert_message(conn, session_id="sess-owner-old", role="assistant", content="记得趁热喝呀。", timestamp=now - 3610)

        # 40 newer ended DMs from unrelated users (> _MAX_CANDIDATE_SESSIONS=32)
        for idx in range(40):
            other_uid = f"u-other-{idx}"
            other_sid = f"sess-other-{idx}"
            ts = now - 1800 + idx * 10
            _insert_session(
                conn,
                session_id=other_sid,
                user_id=other_uid,
                chat_id=other_uid,
                started_at=ts - 60,
                ended_at=ts,
                last_activity_at=ts,
            )
            _insert_message(conn, session_id=other_sid, role="user", content=f"陌生人消息 {idx}", timestamp=ts - 20)
            _insert_message(conn, session_id=other_sid, role="assistant", content=f"陌生人回复 {idx}", timestamp=ts - 10)

    tail = read_previous_session_tail(
        PreviousSessionQuery(
            home=home,
            current_session_id="sess-current",
            current_owner=owner,
            read_owners=(owner,),
            scope=scope,
            is_maintainer=True,
            partner_name="测试伴侣",
            now_ts=now,
        )
    )
    assert tail.startswith("〔上回聊到的末尾〕\n")
    assert "我刚煮了一锅罗宋汤" in tail
    assert "记得趁热喝呀。" in tail
    assert "陌生人消息" not in tail

    # Bound cross-platform read_owners account must also survive >32 newer unrelated sessions.
    bound_tg_owner = _owner_key("telegram", user_id="tg-bound")
    with sqlite3.connect(str(db_path)) as conn:
        _insert_session(
            conn,
            session_id="sess-tg-bound-old",
            source="telegram",
            user_id="tg-bound",
            chat_id="tg-bound",
            started_at=now - 3000,
            ended_at=now - 2400,
            last_activity_at=now - 2400,
        )
        _insert_message(conn, session_id="sess-tg-bound-old", role="user", content="地铁刚出站", timestamp=now - 2420)
        _insert_message(conn, session_id="sess-tg-bound-old", role="assistant", content="路上注意安全。", timestamp=now - 2410)

    bound_tail = read_previous_session_tail(
        PreviousSessionQuery(
            home=home,
            current_session_id="sess-current",
            current_owner=owner,
            read_owners=(owner, bound_tg_owner),
            scope=scope,
            is_maintainer=True,
            partner_name="测试伴侣",
            now_ts=now,
        )
    )
    assert bound_tail.startswith("〔上回聊到的末尾〕\n")
    assert "地铁刚出站" in bound_tail
    assert "路上注意安全。" in bound_tail


def test_test_session_registry_and_test_log_fail_closed_on_real_filesystem(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = 1_800_000_000.0
    home = tmp_path / "home"
    home.mkdir()
    db_path = home / "state.db"
    _create_state_db(db_path)
    owner = _owner_key("qqbot", user_id="u-owner")
    query = PreviousSessionQuery(
        home=home,
        current_session_id="sess-current",
        current_owner=owner,
        read_owners=(owner,),
        scope=_private_scope(session_id="sess-current"),
        is_maintainer=True,
        partner_name="测试伴侣",
        now_ts=now,
    )
    with sqlite3.connect(str(db_path)) as conn:
        _insert_session(
            conn,
            session_id="sess-ordinary-prev",
            started_at=now - 3600,
            ended_at=now - 600,
            last_activity_at=now - 600,
        )
        _insert_message(conn, session_id="sess-ordinary-prev", role="user", content="不应泄漏的测试提问", timestamp=now - 610)
        _insert_message(conn, session_id="sess-ordinary-prev", role="assistant", content="不应泄漏的测试回复", timestamp=now - 605)
        _insert_session(
            conn,
            session_id="sess-other-normal",
            started_at=now - 5000,
            ended_at=now - 1200,
            last_activity_at=now - 1200,
        )
        _insert_message(conn, session_id="sess-other-normal", role="user", content="另一段普通会话提问", timestamp=now - 1210)
        _insert_message(conn, session_id="sess-other-normal", role="assistant", content="另一段普通会话回复", timestamp=now - 1205)

    # 1) Registry replaced by a directory -> real OSError -> fail closed (abort entire lookup).
    state_dir = home / "state"
    state_dir.mkdir(parents=True)
    registry_file = state_dir / "test_isolated_sessions.json"
    registry_file.mkdir()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert read_previous_session_tail(query) == ""
    assert "failing closed" in caplog.text
    registry_file.rmdir()

    # 2) Corrupted JSON -> real ValueError -> fail closed.
    registry_file.write_text("{corrupted json", encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert read_previous_session_tail(query) == ""
    assert "failing closed" in caplog.text

    # 3) Wrong schema (non-dict top level / non-list-or-dict sessions) -> fail closed.
    for bad_payload in ("[]", '{"sessions": "not-a-list"}', '{"sessions": null}'):
        registry_file.write_text(bad_payload, encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert read_previous_session_tail(query) == ""
        assert "failing closed" in caplog.text

    # 4) Valid primary registry + corrupted legacy registry -> still fail closed.
    registry_file.write_text('{"version": 1, "sessions": []}', encoding="utf-8")
    legacy_dir = home / "sanae-mem"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "test_sessions.json"
    legacy_file.write_text("{bad legacy json", encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert read_previous_session_tail(query) == ""
    assert "failing closed" in caplog.text

    # 5) Armed via test_isolated_sessions.json, legacy test_sessions.json, and test-log/<sid>.jsonl.
    with sqlite3.connect(str(db_path)) as conn:
        _insert_session(
            conn,
            session_id="sess-legacy-armed",
            started_at=now - 3000,
            ended_at=now - 400,
            last_activity_at=now - 400,
        )
        _insert_message(conn, session_id="sess-legacy-armed", role="user", content="旧登记表测试提问", timestamp=now - 410)
        _insert_message(conn, session_id="sess-legacy-armed", role="assistant", content="旧登记表测试回复", timestamp=now - 405)
        _insert_session(
            conn,
            session_id="sess-logged-test",
            started_at=now - 2500,
            ended_at=now - 300,
            last_activity_at=now - 300,
        )
        _insert_message(conn, session_id="sess-logged-test", role="user", content="日志测试提问", timestamp=now - 310)
        _insert_message(conn, session_id="sess-logged-test", role="assistant", content="日志测试回复", timestamp=now - 305)

    registry_file.write_text(
        json.dumps({"version": 1, "sessions": ["sess-ordinary-prev"]}),
        encoding="utf-8",
    )
    legacy_file.write_text(
        json.dumps({"sessions": {"sess-legacy-armed": {"armed": True}}}),
        encoding="utf-8",
    )
    test_log_dir = home / "test-log"
    test_log_dir.mkdir(parents=True)
    (test_log_dir / "sess-logged-test.jsonl").write_text('{"role":"user"}\n', encoding="utf-8")

    # All 3 test candidates are skipped; only the real older session survives.
    tail = read_previous_session_tail(query)
    assert "另一段普通会话提问" in tail
    assert "另一段普通会话回复" in tail
    for leaked in ("不应泄漏的测试", "旧登记表测试", "日志测试"):
        assert leaked not in tail

    # 6) Clean subprocess check with real host_bridge.is_test_session_id & broken directory registry.
    vivemind_root, hermes_root = resolve_vivemind_and_hermes_roots()
    sub_home = tmp_path / "sub_home"
    sub_home.mkdir()
    _create_state_db(sub_home / "state.db")
    with sqlite3.connect(str(sub_home / "state.db")) as conn:
        _insert_session(
            conn,
            session_id="sess-sub-armed",
            started_at=now - 3600,
            ended_at=now - 600,
            last_activity_at=now - 600,
        )
        _insert_message(conn, session_id="sess-sub-armed", role="user", content="子进程不应泄漏提问", timestamp=now - 610)
        _insert_message(conn, session_id="sess-sub-armed", role="assistant", content="子进程不应泄漏回复", timestamp=now - 605)

    (sub_home / "state").mkdir(parents=True)
    (sub_home / "state" / "test_isolated_sessions.json").write_text(
        json.dumps({"version": 1, "sessions": ["sess-sub-armed"]}),
        encoding="utf-8",
    )
    res_armed_file = sub_home / "sub-armed.json"
    sub_script_armed = f"""
import json
from pathlib import Path
from sanae_mem.cognitive.previous_session import PreviousSessionQuery, read_previous_session_tail
from sanae_mem.infra import host_bridge
from sanae_mem.infra.chat_scope import ChatScope
from sanae_mem.infra.textutil import _owner_key

home = Path({str(sub_home)!r})
owner = _owner_key("qqbot", user_id="u-owner")
tail = read_previous_session_tail(
    PreviousSessionQuery(
        home=home,
        current_session_id="sess-current",
        current_owner=owner,
        read_owners=(owner,),
        scope=ChatScope(visibility="private", platform="qqbot", actor="u-owner", chat_type="dm", chat_id="u-owner", session_id="sess-current", session_key="agent:main:qqbot:dm:u-owner"),
        is_maintainer=True,
        partner_name="测试伴侣",
        now_ts={now!r},
    )
)
Path({str(res_armed_file)!r}).write_text(json.dumps({{
    "armed_is_test": host_bridge.is_test_session_id("sess-sub-armed"),
    "other_is_test": host_bridge.is_test_session_id("sess-unarmed"),
    "tail": tail,
}}), encoding="utf-8")
"""
    res_armed = run_clean_subprocess(
        sub_script_armed,
        vivemind_root=vivemind_root,
        hermes_root=hermes_root,
        home=sub_home,
        result_file=res_armed_file,
    )
    assert res_armed["armed_is_test"] is True
    assert res_armed["other_is_test"] is False
    assert res_armed["tail"] == ""

    # Now replace test_isolated_sessions.json with a directory in a fresh subprocess.
    (sub_home / "state" / "test_isolated_sessions.json").unlink()
    (sub_home / "state" / "test_isolated_sessions.json").mkdir()
    res_dir_file = sub_home / "sub-dir.json"
    sub_script_dir = f"""
import io
import json
import logging
from pathlib import Path
from sanae_mem.cognitive.previous_session import PreviousSessionQuery, read_previous_session_tail
from sanae_mem.infra import host_bridge
from sanae_mem.infra.chat_scope import ChatScope
from sanae_mem.infra.textutil import _owner_key

log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.WARNING)

home = Path({str(sub_home)!r})
owner = _owner_key("qqbot", user_id="u-owner")
tail = read_previous_session_tail(
    PreviousSessionQuery(
        home=home,
        current_session_id="sess-current",
        current_owner=owner,
        read_owners=(owner,),
        scope=ChatScope(visibility="private", platform="qqbot", actor="u-owner", chat_type="dm", chat_id="u-owner", session_id="sess-current", session_key="agent:main:qqbot:dm:u-owner"),
        is_maintainer=True,
        partner_name="测试伴侣",
        now_ts={now!r},
    )
)
Path({str(res_dir_file)!r}).write_text(json.dumps({{
    "host_is_test_on_dir": host_bridge.is_test_session_id("sess-sub-armed"),
    "tail": tail,
    "logs": log_stream.getvalue(),
}}), encoding="utf-8")
"""
    res_dir = run_clean_subprocess(
        sub_script_dir,
        vivemind_root=vivemind_root,
        hermes_root=hermes_root,
        home=sub_home,
        result_file=res_dir_file,
    )
    assert res_dir["tail"] == ""
    assert "failing closed" in res_dir["logs"]
    assert res_dir["host_is_test_on_dir"] is True


def test_test_session_registry_invalid_utf8_fails_closed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = 1_800_000_000.0
    home = tmp_path / "home"
    home.mkdir()
    db_path = home / "state.db"
    _create_state_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        _insert_session(
            conn,
            session_id="sess-normal-prev",
            started_at=now - 3600,
            ended_at=now - 600,
            last_activity_at=now - 600,
        )
        _insert_message(conn, session_id="sess-normal-prev", role="user", content="普通对话提问", timestamp=now - 610)
        _insert_message(conn, session_id="sess-normal-prev", role="assistant", content="普通对话回复", timestamp=now - 605)

    state_dir = home / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "test_isolated_sessions.json").write_bytes(b"\xff")

    owner = _owner_key("qqbot", user_id="u-owner")
    query = PreviousSessionQuery(
        home=home,
        current_session_id="sess-current",
        current_owner=owner,
        read_owners=(owner,),
        scope=_private_scope(session_id="sess-current"),
        is_maintainer=True,
        partner_name="测试伴侣",
        now_ts=now,
    )
    with caplog.at_level(logging.WARNING):
        assert read_previous_session_tail(query) == ""
    assert "failing closed" in caplog.text
    assert "UnicodeDecodeError" in caplog.text


def test_previous_session_tail_not_crowded_out_by_registered_test_sessions(tmp_path: Path) -> None:
    now = 1_800_000_000.0
    home = tmp_path / "home"
    home.mkdir()
    db_path = home / "state.db"
    _create_state_db(db_path)

    owner = _owner_key("qqbot", user_id="u-owner")
    scope = _private_scope(actor="u-owner", chat_id="u-owner", session_id="sess-current")
    registered_ids: list[str] = []
    with sqlite3.connect(str(db_path)) as conn:
        _insert_session(
            conn,
            session_id="sess-owner-real",
            user_id="u-owner",
            chat_id="u-owner",
            started_at=now - 7200,
            ended_at=now - 3600,
            last_activity_at=now - 3600,
        )
        _insert_message(conn, session_id="sess-owner-real", role="user", content="昨晚散步看到一只橘猫", timestamp=now - 3620)
        _insert_message(conn, session_id="sess-owner-real", role="assistant", content="它有没有过来蹭你？", timestamp=now - 3610)

        # 35 newer registered test sessions for the SAME owner (> _MAX_CANDIDATE_SESSIONS=32)
        for idx in range(35):
            test_sid = f"20261124_0200{idx:02d}_reg"
            registered_ids.append(test_sid)
            ts = now - 1800 + idx * 10
            _insert_session(
                conn,
                session_id=test_sid,
                user_id="u-owner",
                chat_id="u-owner",
                started_at=ts - 60,
                ended_at=ts,
                last_activity_at=ts,
            )
            _insert_message(conn, session_id=test_sid, role="user", content=f"已登记测试提问 {idx}", timestamp=ts - 20)
            _insert_message(conn, session_id=test_sid, role="assistant", content=f"已登记测试回复 {idx}", timestamp=ts - 10)

    state_dir = home / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "test_isolated_sessions.json").write_text(
        json.dumps({"version": 1, "sessions": registered_ids}),
        encoding="utf-8",
    )

    tail = read_previous_session_tail(
        PreviousSessionQuery(
            home=home,
            current_session_id="sess-current",
            current_owner=owner,
            read_owners=(owner,),
            scope=scope,
            is_maintainer=True,
            partner_name="测试伴侣",
            now_ts=now,
        )
    )
    assert tail.startswith("〔上回聊到的末尾〕\n")
    assert "昨晚散步看到一只橘猫" in tail
    assert "它有没有过来蹭你？" in tail
    assert "已登记测试" not in tail


def test_e2e_cold_start_qq_dm_denoise_and_cross_session_continuity(tmp_path: Path) -> None:
    vivemind_root, hermes_root = resolve_vivemind_and_hermes_roots()
    home = tmp_path / "hermes_dm_continuity_e2e"
    home.mkdir(parents=True)
    result_file = home / "e2e-result.json"

    script = f"""
import asyncio
import json
import logging
import os
import shutil
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

logging.basicConfig(level=logging.WARNING)
assert "agent.memory_manager" not in sys.modules, "must start cold without agent.memory_manager"
os.environ["QQ_ALLOWED_USERS"] = "u-dm-1"

home = Path({str(home)!r})
vivemind_root = Path({str(vivemind_root)!r})
plugins_dir = home / "plugins"
plugins_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(vivemind_root / "plugins" / "sanae_mem", plugins_dir / "sanae_mem")

# Stage real-shaped skills (Chinese daily/devops + English coding/office) in home/skills.
skills_root = home / "skills"
for rel, name, desc in [
    ("software-development/systematic-debugging", "systematic-debugging", "4-phase root cause debugging."),
    ("creative/claude-design", "claude-design", "Design one-off HTML artifacts."),
    ("productivity/pdf", "pdf", "PDF files: create, read, merge, fill, OCR."),
    ("devops/goat-usage-query", "goat-usage-query", "查询 GOAT / Command Code 与 Kimi Coding 的额度用量。"),
    ("social-media/sanae-companion", "sanae-companion", "查询东方Project原作设定、同人曲目详情或向 QQ 发送文件与语音时查阅。"),
]:
    d = skills_root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\\nname: {{name}}\\ndescription: {{desc}}\\n---\\nBody.\\n", encoding="utf-8")

wire_requests = []

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
                last_user = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                break
        if model == "test-flash-model":
            if last_user.lstrip().startswith('{{"previous":') or "state_variables" in str(messages):
                reply_obj = {{"sentences": ["我刚醒还有点迷糊。"], "variables": {{"tension": 0.0, "closeness": 0.3, "energy": 0.1}}, "shared_sentences": None}}
            elif "【本轮用户消息】" in last_user:
                reply_obj = {{"items": []}}
            elif "【本轮用户消息（当前对话者原话）】" in last_user:
                reply_obj = {{"new_commitments": [], "resolutions": []}}
            elif "【事件原文】" in last_user:
                reply_obj = {{"summary": "昨晚约好今天去散步", "shareable": True}}
            else:
                reply_obj = {{"events": [{{"summary": "对方刚发来早安问候", "importance": 0.5, "kind": "episodic"}}], "facts": []}}
            content = json.dumps(reply_obj, ensure_ascii=False)
        else:
            wire_requests.append(body)
            has_tool_results = any(isinstance(m, dict) and m.get("role") == "tool" for m in messages)
            if "早安，今天天气不错" in last_user and not has_tool_results:
                tool_calls_payload = [
                    {{"id": "call_sl", "type": "function", "function": {{"name": "skills_list", "arguments": "{{}}"}}}},
                    {{"id": "call_sv_dbg", "type": "function", "function": {{"name": "skill_view", "arguments": json.dumps({{"name": "systematic-debugging"}})}}}},
                    {{"id": "call_sv_sanae", "type": "function", "function": {{"name": "skill_view", "arguments": json.dumps({{"name": "sanae-companion"}})}}}},
                ]
                if body.get("stream"):
                    tc_deltas = [
                        {{"index": idx, "id": tc["id"], "type": "function", "function": tc["function"]}}
                        for idx, tc in enumerate(tool_calls_payload)
                    ]
                    c1 = json.dumps({{
                        "id": "chatcmpl-e2e-tc",
                        "object": "chat.completion.chunk",
                        "choices": [{{"index": 0, "delta": {{"role": "assistant", "content": None, "tool_calls": tc_deltas}}, "finish_reason": None}}],
                    }}, ensure_ascii=False)
                    c2 = json.dumps({{
                        "id": "chatcmpl-e2e-tc",
                        "object": "chat.completion.chunk",
                        "choices": [{{"index": 0, "delta": {{}}, "finish_reason": "tool_calls"}}],
                        "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
                    }}, ensure_ascii=False)
                    sse = f"data: {{c1}}\\n\\ndata: {{c2}}\\n\\ndata: [DONE]\\n\\n".encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(sse)))
                    self.end_headers()
                    self.wfile.write(sse)
                    return
                resp = json.dumps({{
                    "id": "chatcmpl-e2e-tc",
                    "object": "chat.completion",
                    "choices": [{{"index": 0, "message": {{"role": "assistant", "content": None, "tool_calls": tool_calls_payload}}, "finish_reason": "tool_calls"}}],
                    "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
                }}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return
            content = "早呀，今天风挺舒服的。"

        if body.get("stream"):
            c1 = json.dumps({{
                "id": "chatcmpl-e2e",
                "object": "chat.completion.chunk",
                "choices": [{{"index": 0, "delta": {{"role": "assistant", "content": content}}, "finish_reason": None}}],
            }}, ensure_ascii=False)
            c2 = json.dumps({{
                "id": "chatcmpl-e2e",
                "object": "chat.completion.chunk",
                "choices": [{{"index": 0, "delta": {{}}, "finish_reason": "stop"}}],
                "usage": {{"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}},
            }}, ensure_ascii=False)
            sse = f"data: {{c1}}\\n\\ndata: {{c2}}\\n\\ndata: [DONE]\\n\\n".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)
            return

        resp = json.dumps({{
            "id": "chatcmpl-e2e",
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
            "default": "kimi-coding/k3.1-pro",
            "provider": "custom",
            "base_url": base_url,
            "api_key": "dummy-key",
        }},
        "auxiliary": {{
            "sanae_daily_flash": {{"provider": "custom", "model": "test-flash-model", "base_url": base_url, "api_key": "dummy-key"}},
            "sanae_extract": {{"provider": "custom", "model": "test-flash-model", "base_url": base_url, "api_key": "dummy-key"}},
        }},
        "platforms": {{
            "qqbot": {{"enabled": True, "token": "stub-token"}},
        }},
        "plugins": {{
            "enabled": ["sanae_mem"],
            "sanae_mem": {{
                "data_dir": str(home / "sanae-mem"),
                "semantic_backend": "sqlite",
                "memory_identity": {{"platform": "qqbot", "user_id": "u-dm-1"}},
            }},
        }},
    }}, allow_unicode=True),
    encoding="utf-8",
)

from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, ProcessingOutcome, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.plugins import discover_plugins
from hermes_host_contracts import TurnEnvelope, get_cognitive_prompt_policy
from hermes_state import SessionDB
from sanae_mem.cognitive.inner_wiring import wait_for_pending_updates

discover_plugins()

# Seed a previous ended QQ DM session for u-dm-1 within 24h + an unrelated user's session.
now = time.time()
sdb = SessionDB(home / "state.db")
sdb.create_session(
    "sess-prev-dm",
    source="qqbot",
    user_id="u-dm-1",
    chat_id="u-dm-1",
    chat_type="dm",
    session_key="agent:main:qqbot:dm:u-dm-1",
    origin_json=json.dumps({{"platform": "qqbot", "chat_type": "dm", "chat_id": "u-dm-1", "user_id": "u-dm-1"}}),
)
sdb.append_message("sess-prev-dm", role="user", content="昨晚我煮了番茄牛腩面", timestamp=now - 3600)
sdb.append_message("sess-prev-dm", role="assistant", content="听着好香啊，我也想吃一口。", timestamp=now - 3590)
sdb.end_session("sess-prev-dm", "daily")

sdb.create_session(
    "sess-other-dm",
    source="qqbot",
    user_id="u-other",
    chat_id="u-other",
    chat_type="dm",
    session_key="agent:main:qqbot:dm:u-other",
    origin_json=json.dumps({{"platform": "qqbot", "chat_type": "dm", "chat_id": "u-other", "user_id": "u-other"}}),
)
sdb.append_message("sess-other-dm", role="user", content="别人的隐私对话", timestamp=now - 1800)
sdb.end_session("sess-other-dm", "daily")
sdb.close()

class _StubQQAdapter(BasePlatformAdapter):
    def __init__(self, platform_config: PlatformConfig):
        super().__init__(platform_config, Platform.QQBOT)
        self.sent = []
        self.seq = 0
        self.turn_done = None
        self.completed_tasks = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id: str):
        return {{"name": chat_id, "type": "dm"}}

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        self.seq += 1
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id=f"out-{{self.seq}}")

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        task = asyncio.current_task()
        if task is not None:
            self.completed_tasks.append(task)
        if self.turn_done is not None:
            self.turn_done.set()

platform_registry.register(
    PlatformEntry(
        name="qqbot",
        label="Stub QQ",
        adapter_factory=_StubQQAdapter,
        check_fn=lambda: True,
        validate_config=lambda _cfg: True,
        source="builtin",
        allowed_users_env="QQ_ALLOWED_USERS",
    ),
    scope=platform_registry.current_scope_key(),
)

src1 = SessionSource(platform=Platform.QQBOT, chat_id="u-dm-1", chat_type="dm", user_id="u-dm-1", message_id="m-1")
src2 = SessionSource(platform=Platform.QQBOT, chat_id="u-dm-1", chat_type="dm", user_id="u-dm-1", message_id="m-2")

def _wait_all_inner():
    wait_for_pending_updates(timeout=10.0)

async def _run():
    runner = GatewayRunner()
    started = await runner.start()
    assert started is True
    try:
        adapter = runner.adapters[Platform.QQBOT]
        adapter.turn_done = asyncio.Event()
        await adapter.handle_message(MessageEvent(text="早安，今天天气不错", source=src1, message_id="m-1"))
        await asyncio.wait_for(adapter.turn_done.wait(), timeout=20.0)
        await adapter.completed_tasks[-1]
        _wait_all_inner()

        with sqlite3.connect(str(home / "state.db")) as conn:
            t1_db_rows = [
                (r[0] or "", r[1] or "")
                for r in conn.execute("SELECT content, api_content FROM messages WHERE session_id != 'sess-prev-dm'").fetchall()
            ]

        adapter.turn_done.clear()
        await adapter.handle_message(MessageEvent(text="中午一起出去走走吗", source=src2, message_id="m-2"))
        await asyncio.wait_for(adapter.turn_done.wait(), timeout=20.0)
        await adapter.completed_tasks[-1]
        _wait_all_inner()
        return t1_db_rows
    finally:
        await runner.stop()

try:
    t1_db_rows = asyncio.run(_run())
finally:
    server.shutdown()

def _sys(req):
    parts = [str(req.get("instructions") or "")]
    for m in req.get("messages") or req.get("input") or []:
        if isinstance(m, dict) and m.get("role") in ("system", "developer"):
            parts.append(str(m.get("content") or ""))
    return "\\n".join(parts)

def _users(req):
    out = []
    for m in req.get("messages") or req.get("input") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            out.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
    return out

conv_reqs = [r for r in wire_requests if _users(r) and _sys(r)]
t1_sys = _sys(conv_reqs[0])
t1_users = _users(conv_reqs[0])
t1_user = t1_users[-1]
t2_users = _users(conv_reqs[-1])

tool_results = {{}}
for req in conv_reqs:
    for m in req.get("messages") or req.get("input") or []:
        if isinstance(m, dict) and m.get("role") == "tool":
            tcid = str(m.get("tool_call_id") or "")
            if tcid:
                tool_results[tcid] = json.loads(str(m.get("content") or "{{}}"))

skills_list_res = tool_results.get("call_sl") or {{}}
skills_list_names = {{s.get("name") for s in (skills_list_res.get("skills") or []) if isinstance(s, dict)}}
sv_dbg_res = tool_results.get("call_sv_dbg") or {{}}
sv_sanae_res = tool_results.get("call_sv_sanae") or {{}}

with sqlite3.connect(str(home / "state.db")) as conn:
    db_rows = [
        (r[0] or "", r[1] or "")
        for r in conn.execute("SELECT content, api_content FROM messages WHERE session_id != 'sess-prev-dm'").fetchall()
    ]

policy = get_cognitive_prompt_policy()
qq_dm_env = TurnEnvelope(platform="qqbot", chat_id="u-dm-1", chat_type="dm", sender_id="u-dm-1", session_id="sess-check")
policy_active_blocked = policy.is_skill_allowed("systematic-debugging", "software-development", "4-phase root cause debugging.", qq_dm_env)
cfg_path = home / "config.yaml"
cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
cfg_data["plugins"]["sanae_mem"]["shadow"] = True
cfg_path.write_text(yaml.safe_dump(cfg_data, allow_unicode=True), encoding="utf-8")
policy_shadow_allowed = policy.is_skill_allowed("systematic-debugging", "software-development", "4-phase root cause debugging.", qq_dm_env)
cfg_data["plugins"]["sanae_mem"]["shadow"] = False
cfg_path.write_text(yaml.safe_dump(cfg_data, allow_unicode=True), encoding="utf-8")

Path({str(result_file)!r}).write_text(
    json.dumps({{
        "sys_has_help": "For help with Hermes Agent itself" in t1_sys,
        "sys_has_steer": "## Mid-turn user steering" in t1_sys,
        "sys_has_exec_disc": "# Execution discipline" in t1_sys,
        "sys_has_debug_skill": "systematic-debugging" in t1_sys,
        "sys_has_design_skill": "claude-design" in t1_sys,
        "sys_has_pdf_skill": "pdf:" in t1_sys,
        "sys_has_sanae_skill": "sanae-companion" in t1_sys,
        "sys_has_goat_skill": "goat-usage-query" in t1_sys,
        "skills_list_ok": bool(skills_list_res.get("success")),
        "skills_list_names": sorted(n for n in skills_list_names if isinstance(n, str)),
        "sv_dbg_res": sv_dbg_res,
        "sv_sanae_ok": bool(sv_sanae_res.get("success")),
        "policy_active_blocked": policy_active_blocked,
        "policy_shadow_allowed": policy_shadow_allowed,
        "t1_has_world_and_prev_tail": "[我眼中的世界]" in t1_user and "〔上回聊到的末尾〕" in t1_user and "番茄牛腩面" in t1_user,
        "t1_hist_has_api_residual": any("[我眼中的世界]" in c or "[我内心的感受]" in c or "〔上回聊到的末尾〕" in c or "[我眼中的世界]" in a or "[我内心的感受]" in a or "〔上回聊到的末尾〕" in a for c, a in t1_db_rows),
        "t1_has_other_user": "别人的隐私对话" in t1_user,
        "t2_current_has_prev_tail": "〔上回聊到的末尾〕" in t2_users[-1],
        "t2_hist_has_transient": any("〔上回聊到的末尾〕" in u or "[我眼中的世界]" in u or "[我内心的感受]" in u for u in t2_users[:-1]),
        "db_has_transient": any("〔上回聊到的末尾〕" in c or "〔上回聊到的末尾〕" in a or "[我眼中的世界]" in a or "[我内心的感受]" in a for c, a in db_rows),
    }}, ensure_ascii=False),
    encoding="utf-8",
)
"""
    res = run_clean_subprocess(
        script,
        vivemind_root=vivemind_root,
        hermes_root=hermes_root,
        home=home,
        result_file=result_file,
    )
    assert res["sys_has_help"] is False
    assert res["sys_has_steer"] is False
    assert res["sys_has_exec_disc"] is False
    assert res["sys_has_debug_skill"] is False
    assert res["sys_has_design_skill"] is False
    assert res["sys_has_pdf_skill"] is False
    assert res["sys_has_sanae_skill"] is True
    assert res["sys_has_goat_skill"] is True
    assert res["skills_list_ok"] is True
    assert "sanae-companion" in res["skills_list_names"]
    assert "goat-usage-query" in res["skills_list_names"]
    assert "systematic-debugging" not in res["skills_list_names"]
    assert "claude-design" not in res["skills_list_names"]
    assert "pdf" not in res["skills_list_names"]
    assert res["sv_dbg_res"].get("success") is False
    assert "not available in this chat surface" in str(res["sv_dbg_res"].get("error") or "")
    assert res["sv_sanae_ok"] is True
    assert res["policy_active_blocked"] is False
    assert res["policy_shadow_allowed"] is True
    assert res["t1_has_world_and_prev_tail"] is True
    assert res["t1_hist_has_api_residual"] is False
    assert res["t1_has_other_user"] is False
    assert res["t2_current_has_prev_tail"] is False
    assert res["t2_hist_has_transient"] is False
    assert res["db_has_transient"] is False
