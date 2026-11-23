"""Task 06 cross-platform maintainer binding end-to-end chain tests.

Covers:
1. ``confirm_bindings`` filtering by active profile + ``read_owners`` isolation for non-anchors.
2. Slack primary ``memory_identity`` anchor with ``scope_id`` & cross-workspace collision blocking.
3. Bound Slack/Discord DM wiring into ``daily_context``, ``grounded_review`` / ``USER.md`` /
   ``promotion_review`` / ``memory_axis``, ``reinforce_on_recall`` / ``dormant_owner`` /
   ``sanae_mem_forget``, ``self_utterances``, ``life_context``, Bilibili/video lookup & ``source_intake``,
   ``known_inbound_images``, ``shared_view`` exclusion, and ``is_other_person_source``.
4. Distinction between ``host_operator`` (fixed account only) and bound DM private rights.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
import time
from collections import deque
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PLUGINS = _ROOT / "plugins"
_MEM = _PLUGINS / "sanae_mem"
for path in (_MEM, _PLUGINS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hermes_host_contracts as hc  # noqa: E402
from gateway.session_context import clear_session_vars, set_session_vars  # noqa: E402

from sanae_mem.governance import account_binding  # noqa: E402
from sanae_mem.storage import database  # noqa: E402
from sanae_mem import integrations  # noqa: E402
from sanae_mem.cognitive import life_context  # noqa: E402
from sanae_mem.domains import meme_assets, memes  # noqa: E402
from sanae_mem import plugin_entry as entry  # noqa: E402
from sanae_mem.runtime import scheduled  # noqa: E402
from sanae_mem.runtime.private_delivery_settler import (  # noqa: E402
    VerifiedPrivateDelivery,
    settle_private_delivery,
)
from sanae_mem.cognitive import self_utterances  # noqa: E402
from sanae_mem.governance import shared_memory, source_intake  # noqa: E402
from sanae_mem.governance.source_context import (  # noqa: E402
    ChatScope,
    current_chat_scope,
    resolve_chat_identity,
    source_view,
)
from sanae_mem.infra.textutil import _owner_key  # noqa: E402

QQ_PRIMARY = {"platform": "qqbot", "user_id": "qq-openid-1"}
QQ_ANCHOR = _owner_key("qqbot", user_id="qq-openid-1")
SLACK_LEGACY = _owner_key("slack", user_id="U_MAINT", scope_id="T_WORK")
DISCORD_LEGACY = _owner_key("discord", user_id="4242")
MULTI_CONFIG = {
    "memory_identity": dict(QQ_PRIMARY),
    "maintainer_account_bindings": [
        {"platform": "slack", "user_id": "U_MAINT", "scope_id": "T_WORK"},
        {"platform": "discord", "user_id": "4242"},
    ],
}

SLACK_PRIMARY_CFG = {
    "memory_identity": {"platform": "slack", "user_id": "U_MAINT", "scope_id": "T_MAIN"},
    "maintainer_account_bindings": [
        {"platform": "qqbot", "user_id": "qq-openid-1", "scope_id": "102000001"},
        {"platform": "discord", "user_id": "4242"},
    ],
}
SLACK_ANCHOR = _owner_key("slack", user_id="U_MAINT", scope_id="T_MAIN")


@pytest.fixture
def bind():
    tokens: list = []

    def _bind(**facts) -> None:
        tokens.extend(set_session_vars(**facts))

    yield _bind
    if tokens:
        clear_session_vars(tokens)


def _write_plugin_config(home: Path, config: dict) -> None:
    cfg_path = home / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({"plugins": {"sanae_mem": config}}, ensure_ascii=False), encoding="utf-8")


def _provider(tmp_path: Path, config: dict, *, bind, **facts) -> entry.SanaeMemProvider:
    _write_plugin_config(tmp_path, {**config, "data_dir": str(tmp_path / "sanae-mem")})
    bind(**facts)
    provider = entry.SanaeMemProvider({
        "data_dir": str(tmp_path / "sanae-mem"),
        "semantic_backend": "sqlite",
        "shadow": False,
        **config,
    })
    provider.initialize(
        facts["session_id"],
        hermes_home=str(tmp_path),
        platform=facts["platform"],
        agent_context="primary",
    )
    return provider


def _settle_daily_image(
    provider: entry.SanaeMemProvider, scope: ChatScope, home: Path, *, expected_owner: str | None,
) -> None:
    turn_id = f"turn-{scope.message_id}"
    with hc.turn_envelope_scope(hc.TurnEnvelope(
        session_id=scope.session_id, session_key=scope.session_key, platform=scope.platform,
        chat_id=scope.chat_id, chat_type=scope.chat_type, sender_id=scope.actor,
        scope_id=scope.scope_id, message_id=scope.message_id, turn_id=turn_id,
        is_messaging_surface=True, write_origin="foreground",
    )):
        result = settle_private_delivery(provider, VerifiedPrivateDelivery(
            session_id=scope.session_id, session_key=scope.session_key, platform=scope.platform,
            chat_id=scope.chat_id, chat_type=scope.chat_type, sender_id=scope.actor,
            scope_id=scope.scope_id, incoming_message_id=scope.message_id, turn_id=turn_id,
            user_message="给你看张图\n[已知梗图] 一只猫", assistant_response="看到了，是只猫。",
            delivered_at=time.time(), has_native_image=True,
        ))
        if result.daily_recorded:
            rendered = provider.current_turn_context(
                session_id=scope.session_id, turn_id=turn_id,
                platform=scope.platform, sender_id=scope.actor,
            )
            assert "发了一张图片" in rendered
    assert result.settled, result.reason
    assert result.daily_recorded == (expected_owner is not None)
    with closing(sqlite3.connect(home / "sanae-mem" / "sanae.db")) as conn:
        rows = conn.execute(
            "SELECT owner_key, text FROM daily_context WHERE source_turn=?", (scope.session_id,),
        ).fetchall()
    assert rows == ([] if expected_owner is None else [(expected_owner, "发了一张图片")])


# ── 1. confirm_bindings profile filter + Slack anchor + QQ unscoped read_owners ─

def test_confirm_bindings_filters_by_active_profile_and_read_owners_isolates_stranger(tmp_path):
    db_path = tmp_path / "sanae-mem" / "sanae.db"
    cfg = {
        "memory_identity": dict(QQ_PRIMARY),
        "maintainer_account_bindings": [
            {"platform": "slack", "user_id": "U_MAINT", "scope_id": "T_WORK"},
            {"profile": "work", "platform": "discord", "user_id": "9999"},
        ],
    }
    with closing(database.open_database(db_path)) as conn:
        default_res = account_binding.confirm_bindings(conn, cfg)
        assert (default_res["confirmed"], default_res["existing"]) == (1, 0)
        rows = [
            tuple(r)
            for r in conn.execute(
                "SELECT profile, platform, account_id FROM maintainer_account_bindings"
            ).fetchall()
        ]
        assert rows == [("default", "slack", "U_MAINT")]

        work_res = account_binding.confirm_bindings(conn, cfg, profile="work")
        assert work_res["confirmed"] == 1
        stranger = _owner_key("slack", user_id="U_OTHER", scope_id="T_OTHER")
        assert account_binding.read_owners(stranger, config=cfg, connection=conn) == (stranger,)


def test_slack_primary_anchor_blocks_cross_workspace_and_reads_legacy_qq(tmp_path, monkeypatch, bind):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert account_binding.maintainer_anchor(SLACK_PRIMARY_CFG) == SLACK_ANCHOR
    assert scheduled._schedule_owner(SLACK_PRIMARY_CFG) == SLACK_ANCHOR

    provider = _provider(
        tmp_path,
        SLACK_PRIMARY_CFG,
        bind=bind,
        platform="slack",
        chat_type="dm",
        chat_id="D_MAIN",
        user_id="U_MAINT",
        scope_id="T_MAIN",
        session_id="sess-slack-main",
        session_key="slack:T_MAIN:dm:D_MAIN",
        message_id="m-main",
    )
    assert provider._daily_owner_key == SLACK_ANCHOR
    assert provider._owner_key == SLACK_ANCHOR
    assert QQ_ANCHOR in provider._read_owner_keys  # unscoped legacy QQ key readable

    main_scope = current_chat_scope()
    main_identity = resolve_chat_identity(main_scope, SLACK_PRIMARY_CFG, connection=provider._conn)
    assert (main_identity.maintainer, main_identity.host_operator) == (True, True)

    # Bound QQ DM gets memory rights, NOT host_operator rights
    clear_session_vars([])
    bind(
        platform="qqbot",
        chat_type="dm",
        chat_id="qq-dm",
        user_id="qq-openid-1",
        scope_id="102000001",
        session_id="sess-qq-bound",
        session_key="qqbot:dm:qq-dm",
        message_id="m-qq",
    )
    qq_identity = resolve_chat_identity(current_chat_scope(), SLACK_PRIMARY_CFG, connection=provider._conn)
    assert (qq_identity.maintainer, qq_identity.host_operator) == (True, False)
    assert qq_identity.private_owner_key == SLACK_ANCHOR

    # Same Slack user_id in a DIFFERENT workspace T_OTHER must be strictly rejected
    clear_session_vars([])
    bind(
        platform="slack",
        chat_type="dm",
        chat_id="D_OTHER",
        user_id="U_MAINT",
        scope_id="T_OTHER",
        session_id="sess-slack-other",
        session_key="slack:T_OTHER:dm:D_OTHER",
        message_id="m-other",
    )
    other_identity = resolve_chat_identity(current_chat_scope(), SLACK_PRIMARY_CFG, connection=provider._conn)
    assert (other_identity.maintainer, other_identity.host_operator) == (False, False)
    assert other_identity.private_owner_key != SLACK_ANCHOR
    provider.on_session_switch("sess-slack-other")
    _settle_daily_image(provider, current_chat_scope(), tmp_path, expected_owner=None)
    provider.on_session_switch("sess-slack-main")
    _settle_daily_image(provider, main_scope, tmp_path, expected_owner=SLACK_ANCHOR)
    provider.shutdown()


# ── 2. daily_context works in bound Slack/Discord DMs ───────────────────────────

def test_daily_context_allows_bound_slack_and_discord_dms(tmp_path, monkeypatch, bind):
    for plat, user, scope, skey in (
        ("slack", "U_MAINT", "T_WORK", "slack:T_WORK:dm:D1"),
        ("discord", "4242", "", "discord:dm:55:4242"),
    ):
        clear_session_vars([])
        home = tmp_path / plat
        monkeypatch.setenv("HERMES_HOME", str(home))
        provider = _provider(
            home,
            MULTI_CONFIG,
            bind=bind,
            platform=plat,
            chat_type="dm",
            chat_id="C1",
            user_id=user,
            scope_id=scope,
            session_id=f"sess-{plat}",
            session_key=skey,
            message_id=f"m-{plat}",
        )
        _settle_daily_image(provider, current_chat_scope(), home, expected_owner=QQ_ANCHOR)
        provider.shutdown()


# ── 3. grounded_review / USER.md / promotion_review / memory_axis ───────────────

def _seed_grounded_event(
    provider: entry.SanaeMemProvider,
    *,
    raw: str,
    owner_key: str,
    platform: str,
    user: str,
    scope_id: str = "",
) -> int:
    source = source_view({
        "kind": "maintainer",
        "author_id": user,
        "platform": platform,
        "ref": f"msg-{user}-{platform}",
        "certainty": "verified",
        "visibility": "private",
        "scope_id": scope_id,
        "profile": "default",
    })
    now_iso = datetime.now().astimezone().isoformat()
    snapshot = {
        "version": 2,
        "available": True,
        "session": "old-sess",
        "user": user,
        "turn": "msg-1",
        "total_messages": 1,
        "origin": "foreground",
        "unattended": False,
        "platform": platform,
        "source_context": source,
        "reason": "",
        "source_messages": [{"role": "user", "content": raw.split("] ", 1)[-1]}],
    }
    snap_json = json.dumps(snapshot, ensure_ascii=False)
    snap_sha = hashlib.sha256(snap_json.encode("utf-8")).hexdigest()
    eid = provider.event_store.insert_event_row(
        f"fp-{platform}-{raw}",
        (now_iso, raw, raw.split("] ", 1)[-1], 0.5, 0.8, "", None, now_iso, None, "life"),
        owner_key=owner_key,
        source_context=json.dumps(source, ensure_ascii=False),
    )
    provider._conn.execute(
        "INSERT INTO event_occurrences(event_id, raw, observed_at, source_snapshot, source_snapshot_sha256) "
        "VALUES (?, ?, ?, ?, ?)",
        (eid, raw, now_iso, snap_json, snap_sha),
    )
    provider._conn.commit()
    return int(eid)


def test_grounded_review_and_promotion_accept_bound_and_prebinding_events(tmp_path, monkeypatch, bind, request):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.memory import import_provider_module, load_memory_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(str(tmp_path))
    request.addfinalizer(lambda: reset_hermes_home_override(token))
    shutil.copytree(_MEM, tmp_path / "plugins" / "sanae_mem",
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    base_url = "https://review-fixture.invalid/v1"
    model_config = {
        "provider": "custom", "model": "review-fixture", "base_url": base_url,
        "api_key": "fixture-key", "api_mode": "chat_completions",
    }
    (tmp_path / "config.yaml").write_text(json.dumps({
        "memory": {"provider": "sanae_mem"},
        "auxiliary": {"sanae_extract": model_config},
        "plugins": {"sanae_mem": {**MULTI_CONFIG, "semantic_backend": "sqlite", "shadow": False}},
    }), encoding="utf-8")
    replies = deque([
        {
            "duty": "user", "long_term": True, "summary": "他周末最爱去静安寺喝手冲咖啡",
            "source_kind": "user_statement", "covered_by": "", "significance": "偏好咖啡",
        },
        {
            "duty": "memory", "long_term": True, "summary": "他偏爱手冲深烘",
            "source_kind": "event_promotion", "covered_by": "", "significance": "偏好咖啡",
        },
    ])

    def model_response(_client: httpx.Client, request: httpx.Request, **_kwargs: Any) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{base_url}/chat/completions"
        return httpx.Response(200, request=request, json={
            "id": "review-fixture", "object": "chat.completion", "model": "review-fixture", "created": 1,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(replies.popleft(), ensure_ascii=False),
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })

    monkeypatch.setattr(httpx.Client, "send", model_response)
    bind(
        platform="slack",
        chat_type="dm",
        chat_id="D_WORK",
        user_id="U_MAINT",
        scope_id="T_WORK",
        session_id="sess-slack-review",
        session_key="slack:T_WORK:dm:D_WORK",
        message_id="m-review",
    )
    provider = load_memory_provider("sanae_mem", register_skills=False)
    installed_module = import_provider_module("sanae_mem")
    assert installed_module is not None
    provider_type: type[entry.SanaeMemProvider] = installed_module.SanaeMemProvider
    assert isinstance(provider, provider_type)
    request.addfinalizer(provider.shutdown)
    provider.initialize("sess-slack-review", hermes_home=str(tmp_path), platform="slack",
                        agent_context="primary")
    # Mix a pre-binding Slack event (stored under SLACK_LEGACY) and a current QQ event (stored under QQ_ANCHOR)
    old_slack_eid = _seed_grounded_event(
        provider,
        raw="[2026-01-01 10:00|今天上午] 其实我周末最爱去静安寺喝手冲咖啡",
        owner_key=SLACK_LEGACY,
        platform="slack",
        user="U_MAINT",
        scope_id="T_WORK",
    )
    qq_eid = _seed_grounded_event(
        provider,
        raw="[2026-01-02 10:00|今天上午] 我只喝深度烘焙黑咖啡",
        owner_key=QQ_ANCHOR,
        platform="qqbot",
        user="qq-openid-1",
        scope_id="102000001",
    )
    with hc.turn_envelope_scope(hc.TurnEnvelope(
        platform="slack", chat_type="dm", chat_id="D_WORK", sender_id="U_MAINT",
        scope_id="T_WORK", session_id="sess-slack-review", session_key="slack:T_WORK:dm:D_WORK",
        message_id="m-review", write_origin="foreground",
    )):
        saved = provider.resident.save_grounded_user(
            text="他周末最爱去静安寺喝手冲咖啡",
            evidence_ids=[old_slack_eid, qq_eid],
        )
    assert saved["ok"] is True and saved["written"] is True
    assert "他周末最爱去静安寺喝手冲咖啡" in (tmp_path / "memories" / "USER.md").read_text(encoding="utf-8")

    # Promotion review & memory_axis promote on pre-binding event
    promo = provider.promotion_review.review_promotion_candidate(old_slack_eid, enforce_daily_cap=False)
    assert promo["action"] == "propose_new" and promo["proposal"]["background"]["event_id"] == old_slack_eid
    assert promo["proposal"]["target"] == "memory"

    provider.sqlite.execute("UPDATE events SET tier='short' WHERE id=?", (old_slack_eid,))
    with hc.turn_envelope_scope(hc.TurnEnvelope(
        platform="slack", chat_type="dm", chat_id="D_WORK", sender_id="U_MAINT",
        scope_id="T_WORK", session_id="sess-slack-review", session_key="slack:T_WORK:dm:D_WORK",
        message_id="m-review", write_origin="foreground",
    )):
        axis_res = provider.lifecycle.marked_promoted([old_slack_eid], "long")
    assert axis_res == {"ids": [old_slack_eid], "to": "long", "updated": 1, "channel": "explicit_save"}
    event = provider.sqlite.fetchone("SELECT tier, owner_key FROM events WHERE id=?", (old_slack_eid,))
    assert event is not None and tuple(event) == ("long", SLACK_LEGACY)
    promotion = provider.sqlite.fetchone(
        "SELECT channel, previous_tier, owner_key FROM memory_axis_promotions WHERE event_id=?",
        (old_slack_eid,),
    )
    assert promotion is not None and tuple(promotion) == ("explicit_save", "short", QQ_ANCHOR)


# ── 4. recall_count, dormant wakeup, forget, shared_view exclusion, is_other_person_source ─

def test_recall_dormant_forget_shared_views_and_attribution_on_legacy_rows(tmp_path, monkeypatch, bind):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider(
        tmp_path,
        MULTI_CONFIG,
        bind=bind,
        platform="slack",
        chat_type="dm",
        chat_id="D_WORK",
        user_id="U_MAINT",
        scope_id="T_WORK",
        session_id="sess-slack",
        session_key="slack:T_WORK:dm:D_WORK",
        message_id="m-1",
    )
    past = (datetime.now().astimezone() - timedelta(days=2)).isoformat()
    legacy_id = provider.event_store.insert_event_row(
        "fp-old-slack",
        (past, "[2026-01-01 10:00|今天上午] 在旧Slack账号聊过的老乐队", "在旧Slack账号聊过的老乐队",
         0.6, 0.9, "", None, past, past, "music"),
        owner_key=SLACK_LEGACY,
    )
    dormant_id = provider.event_store.insert_event_row(
        "fp-dormant-slack",
        (past, "[2025-12-01 10:00|今天上午] 旧Slack账号的休眠记忆", "旧Slack账号的休眠记忆",
         0.6, 0.9, "", None, past, past, "life"),
        owner_key=SLACK_LEGACY,
    )
    with provider._conn:
        provider._conn.execute(
            "UPDATE events SET tier='short', half_life_days=3.0, ttl_expire=?, shared_view=? WHERE id=?",
            ((datetime.now().astimezone() + timedelta(days=5)).isoformat(), "之前聊过一个老乐队", legacy_id),
        )
        provider._conn.execute(
            "UPDATE events SET tier='dormant', half_life_days=3.0 WHERE id=?",
            (dormant_id,),
        )

    # reinforce_on_recall updates recall_count on SLACK_LEGACY event
    assert provider.event_store.reinforce_on_recall([legacy_id], via="chat") == 1
    assert provider._conn.execute("SELECT recall_count FROM events WHERE id=?", (legacy_id,)).fetchone()[0] == 1

    # query_shared_views excludes own legacy rows so it never duplicates or leaks as someone else's topic
    shared_rows = provider.event_store.query_shared_views(now=datetime.now().astimezone().isoformat(), owner_key=QQ_ANCHOR)
    assert all(int(r["id"]) != legacy_id for r in shared_rows)

    # dormant_owner finds old-account dormant events
    ev_rows = provider.query_events(
        datetime.now().astimezone().isoformat(), read_context=True, dormant_owner=QQ_ANCHOR,
    )
    assert any(int(r["id"]) == dormant_id for r in ev_rows)

    # is_other_person_source treats pre-binding private external_user of bound Slack as current owner, not stranger
    pre_binding_src = {
        "kind": "external_user", "platform": "slack", "author_id": "U_MAINT",
        "scope_id": "T_WORK", "visibility": "private",
    }
    assert shared_memory.is_other_person_source(
        pre_binding_src, f"owner:{SLACK_LEGACY}", QQ_ANCHOR, read_owners=provider._read_owner_keys,
    ) is False
    # Different Slack workspace & Bilibili DM still count as other person
    assert shared_memory.is_other_person_source(
        {**pre_binding_src, "scope_id": "T_OTHER"}, f"owner:{SLACK_LEGACY}", QQ_ANCHOR,
        read_owners=provider._read_owner_keys,
    ) is True
    assert shared_memory.is_other_person_source(
        {"kind": "external_user", "platform": "bilibili", "author_id": "bilibili:dm:77", "visibility": "private"},
        "", QQ_ANCHOR, read_owners=provider._read_owner_keys,
    ) is True

    # The forget tool can delete the recalled legacy row.
    forgot = json.loads(provider.handle_tool_call("sanae_mem_forget", {"id": legacy_id}))
    assert forgot["ok"] is True
    assert legacy_id in forgot["result"]["ids"]
    assert provider.sqlite.fetchone("SELECT id FROM events WHERE id=?", (legacy_id,)) is None


# ── 5. self_utterances, life_context, integrations, source_intake, memes ────────

def test_self_utterances_life_context_video_intake_and_memes_in_bound_dm(tmp_path, monkeypatch, bind):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    data_dir = tmp_path / "sanae-mem"
    provider = _provider(
        tmp_path,
        MULTI_CONFIG,
        bind=bind,
        platform="slack",
        chat_type="dm",
        chat_id="D_WORK",
        user_id="U_MAINT",
        scope_id="T_WORK",
        session_id="sess-slack-bound",
        session_key="slack:T_WORK:dm:D_WORK",
        message_id="m-bound",
    )
    now_ts = time.time()
    # Seed historical records under SLACK_LEGACY (pre-binding) and QQ_ANCHOR (post-binding)
    watch_ts = (datetime.now().astimezone() - timedelta(seconds=30)).isoformat()
    with provider._conn:
        provider._conn.execute(
            "CREATE TABLE IF NOT EXISTS bilibili_reply_receipts(at REAL, visibility TEXT, text TEXT, owner_key TEXT)"
        )
        provider._conn.execute(
            "INSERT INTO bilibili_reply_receipts VALUES (?, 'private', '旧账号发的B站回复', ?)",
            (now_ts - 60, SLACK_LEGACY),
        )
        provider._conn.execute(
            "INSERT INTO video_watches (bvid,platform,source_object_id,title,summary,read_scope,owner_key,"
            "watched_at,created_at,updated_at) VALUES ('BV1ab4y1c7DE','bilibili','BV1ab4y1c7DE','绑定前看的视频',"
            "'详细总结内容','{\"kind\":\"self\"}',?,?,?,?)",
            (SLACK_LEGACY, watch_ts, watch_ts, watch_ts),
        )
        provider._conn.execute(
            "INSERT INTO free_think_reflections "
            "(owner_key, execution_id, job_id, session_id, finished_at, reflection, created_at) "
            "VALUES (?, 'exec-1', 'job-1', 'cron_1', ?, '自由放空时想的小事', ?)",
            (SLACK_LEGACY, now_ts - 50, datetime.now().astimezone().isoformat()),
        )

    # self_utterances works inside bound Slack DM and surfaces pre-binding Bilibili reply
    utter = self_utterances.recent_self_utterances(
        QQ_ANCHOR, "sess-slack-bound", home=tmp_path, data_dir=data_dir,
    )
    assert "旧账号发的B站回复" in utter

    # life_context reads old-account watches & free_think reflections for bound Slack DM
    life = life_context.collect_recent_activities(
        home=tmp_path, data_dir=data_dir, owner_key=QQ_ANCHOR,
        session_id="sess-slack-bound", since=now_ts - 3600, through=now_ts,
        private_experience=True,
    )
    assert "绑定前看的视频" in life and "自由放空时想的小事" in life

    # integrations + video lookup + source_intake accept bound Slack DM
    ctx = SimpleNamespace(has_plugin=lambda name: True, current_cron_execution=lambda: None)
    reg = SimpleNamespace(active=True)
    bili = integrations._BilibiliMemory(ctx, provider, home=tmp_path, registration=reg)
    assert bili._owner(write=False) == QQ_ANCHOR
    assert provider.video_memory.lookup_video_watch("bilibili", "BV1ab4y1c7DE", owner_key=QQ_ANCHOR)["title"] == "绑定前看的视频"

    with hc.turn_envelope_scope(hc.TurnEnvelope(
        platform="slack", chat_type="dm", chat_id="D_WORK", sender_id="U_MAINT",
        scope_id="T_WORK", session_id="sess-slack-bound", session_key="slack:T_WORK:dm:D_WORK",
        message_id="m-bound", write_origin="foreground",
    )):
        assert source_intake._resolve_intake_owner(provider) == QQ_ANCHOR

    # memes.known_inbound_images finds memes stored under QQ_ANCHOR, SLACK_LEGACY, or legacy ''
    img = tmp_path / "cat.png"
    img.write_bytes(b"fake-png-bytes-123")
    digest = meme_assets.file_sha256(img)
    with provider._conn:
        cur = provider._conn.execute(
            "INSERT INTO meme_records (title,meaning,created_at,updated_at,owner_key,tier,"
            "visibility_scope,evidence_status) VALUES ('猫猫图','开心表情','2026-01-01','2026-01-01',?,'common','private','confirmed')",
            (QQ_ANCHOR,),
        )
        provider._conn.execute(
            "INSERT INTO meme_assets (record_id, path, sha256, mime, created_at) VALUES (?, ?, ?, 'image/png', '2026-01-01')",
            (int(cur.lastrowid), str(img), digest),
        )
    notes = memes.known_inbound_images([str(img)])
    assert str(img) in notes and "猫猫图" in notes[str(img)]

    legacy_img = tmp_path / "old-slack.png"
    legacy_img.write_bytes(b"fake-old-slack-png-456")
    legacy_digest = meme_assets.file_sha256(legacy_img)
    with provider._conn:
        cur2 = provider._conn.execute(
            "INSERT INTO meme_records (title,meaning,created_at,updated_at,owner_key,tier,"
            "visibility_scope,evidence_status) VALUES ('老图','旧账号收藏','2026-01-01','2026-01-01',?,'common','private','confirmed')",
            (SLACK_LEGACY,),
        )
        provider._conn.execute(
            "INSERT INTO meme_assets (record_id, path, sha256, mime, created_at) VALUES (?, ?, ?, 'image/png', '2026-01-01')",
            (int(cur2.lastrowid), str(legacy_img), legacy_digest),
        )
    notes2 = memes.known_inbound_images([str(legacy_img)])
    assert str(legacy_img) in notes2 and "老图" in notes2[str(legacy_img)]

    # Stranger DM in another workspace cannot see maintainer private memes
    clear_session_vars([])
    bind(
        platform="slack",
        chat_type="dm",
        chat_id="D_STRANGER",
        user_id="U_STRANGER",
        scope_id="T_OTHER",
        session_id="sess-stranger",
        session_key="slack:T_OTHER:dm:D_STRANGER",
        message_id="m-stranger",
    )
    assert memes.known_inbound_images([str(img), str(legacy_img)]) == {}
