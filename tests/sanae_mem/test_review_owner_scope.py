"""Exercise USER evidence on the pinned Hermes provider boundary."""

from contextlib import contextmanager
import json
from types import SimpleNamespace

from test_original_hermes import clear_session_vars, set_session_vars, world

from sanae_mem.infra import host_bridge
from sanae_mem.governance.memory_evidence import restore_source_evidence


@contextmanager
def _private_turn(message_id, *, actor="account-A", session="private-A", chat_type="dm"):
    tokens = set_session_vars(
        platform="telegram", chat_id="chat-test", chat_type=chat_type, user_id=actor,
        session_id=session, session_key="agent:main:telegram:dm:test", message_id=message_id,
    )
    try:
        with host_bridge.turn_envelope_scope(is_messaging_surface=False):
            yield
    finally:
        clear_session_vars(tokens)


def test_original_hermes_sync_binds_only_the_current_user_words(world, monkeypatch):
    provider = world["provider_a"]
    manager = world["memory_manager_a"]
    raw_facts = (
        "我每周都会沿着河边骑一趟自行车，这是我很重要的锻炼习惯",
        "我每个星期六都会去天文馆做志愿讲解，这是我答应过的事",
    )
    history = [
        {"role": "user", "content": "我每周三都去参加攀岩俱乐部"},
        {"role": "assistant", "content": "我会试着从公开消息里总结别人的偏好"},
    ]
    def sync(raw, *, message_id=None, **metadata):
        turn = [{"role": "user", "content": raw, "platform_message_id": message_id, **metadata}]
        manager.sync_all(raw, "收到。", session_id="private-A", messages=[*history, *turn])
        manager.flush_pending(timeout=5)

    for index, raw in enumerate(raw_facts, start=1):
        with _private_turn("opening-route"):
            # Original Hermes has no turn_author on an ordinary human turn. The
            # user row carries the actual inbound id, even on queued follow-ups.
            sync(raw, message_id=f"physical-{index}")
    with _private_turn("group", chat_type="group"):
        sync("我最喜欢在周末制作木雕，这是我很重要的爱好", message_id="group")
    with _private_turn("synthetic"):
        sync("我最喜欢每天坐渡轮，这是我很重要的习惯", message_id="synthetic", display_kind="notification")
    with _private_turn("no-id"):
        sync("我经常用自己的零花钱买书，这项习惯对我很重要")
    with _private_turn("no-source-quote"):
        result = json.loads(manager.handle_tool_call(
            "sanae_mem_add", {"content": "我每年会参加一次桥牌比赛"}))
        assert result["ok"] and result["result"]["id"]

    with _private_turn("review"):
        ids = [row[0] for row in provider._conn.execute(
            "SELECT id FROM events WHERE source_snapshot<>'' ORDER BY id"
        )]
        assert len(ids) == 2
        for index, (eid, raw) in enumerate(zip(ids, raw_facts), start=1):
            row = provider.event_store.get_grounded_occurrence(eid)
            assert row["raw"] == raw
            snapshot = restore_source_evidence(json.loads(row["source_snapshot"]))
            assert snapshot.user == "account-A"
            assert snapshot.turn == f"physical-{index}"
            assert snapshot.source_context["ref"] == snapshot.turn
            assert [item.text for item in snapshot.user_messages] == [raw]

        recognition = "重视长期坚持自己的兴趣活动"
        contexts = []

        class _FakeGroundedLlm:
            def complete(self, messages, **kwargs):
                contexts.append(str(messages[-1]["content"]))
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "duty": "user",
                            "long_term": True,
                            "summary": recognition,
                            "covered_by": "",
                            "significance": "用户长期坚持的兴趣活动",
                            "source_kind": "user_statement",
                        },
                        ensure_ascii=False,
                    )
                )

        provider.attach_runtime_llm(_FakeGroundedLlm())
        assert not provider.grounded_review.review_grounded_user(text=recognition, evidence_ids=[])["ok"]
        reviewed = provider.grounded_review.review_grounded_user(text=recognition, evidence_ids=ids[:1])
        assert reviewed["ok"], reviewed
        assert len(contexts) == 1
        assert raw_facts[0] in contexts[0]
        assert raw_facts[1] not in contexts[0]
        assert history[0]["content"] not in contexts[0]
        anchor = json.loads(reviewed["evidence_group"])[0]
        declared = {
            "content": recognition, "review_sha256": reviewed["review_sha256"],
            "content_sha256": reviewed["content_sha256"],
            "baseline_sha256": reviewed["baseline_sha256"],
            "evidence_group_ids": ids[:1], "event_id": anchor["event_id"],
            "event_sha256": anchor["event_sha256"], "event_version": anchor["event_version"],
        }
        assert provider.grounded_review.grounded_binding(declared)["verified"]
        assert provider.grounded_review.grounded_binding({**declared, "content": recognition + "。"}) is None
        provider._conn.execute("UPDATE events SET blacklist=1 WHERE id=?", (ids[0],))
        provider._conn.commit()
        assert provider.grounded_review.grounded_binding(declared) is None
