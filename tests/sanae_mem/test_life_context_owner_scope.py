"""Scheduled speech requires durable text for the same private owner and route."""

from contextlib import contextmanager
import sqlite3
import time

from test_original_hermes import clear_session_vars, set_session_vars

from cron import delivery_queue, executions
from hermes_host_contracts import record_verified_cron_send
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from sanae_mem.cognitive.self_utterances import recent_self_utterances
from sanae_mem.governance.source_context import current_chat_scope


@contextmanager
def _route(*, actor="owner-a", chat_type="dm"):
    tokens = set_session_vars(
        platform="telegram", chat_type=chat_type, chat_id="shared-route", thread_id="topic-a",
        user_id=actor, session_id="next-turn", session_key="next-key",
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


def test_original_outcome_does_not_prove_sent_text_for_the_private_owner(tmp_path):
    token = set_hermes_home_override(str(tmp_path))
    try:
        job = {"id": "test-job", "origin": {"platform": "telegram", "chat_id": "shared-route",
                                           "thread_id": "topic-a", "user_id": "owner-a"}}
        execution = executions.create_execution(job["id"], source="fixture")
        eid = execution["id"]
        sent = "这段正文仅在队列发送时存在。"
        delivery_queue.enqueue(eid, job, sent)
        observed = []
        assert delivery_queue.drain(lambda _, content, __: observed.append(content)) == 1
        executions.finish_execution(eid, success=True, delivery_outcome="delivered")
        assert observed == [sent]
        assert executions.get_execution(eid)["delivery_outcome"] == "delivered"
        with sqlite3.connect(delivery_queue.queue_path(tmp_path)) as db:
            assert db.execute("SELECT status, job_json, content FROM deliveries").fetchone() == (
                "delivered", "{}", "")
        with _route():
            owner = current_chat_scope().owner
            assert recent_self_utterances(owner, "next-turn", home=tmp_path) == ""
    finally:
        reset_hermes_home_override(token)


def test_legacy_receipts_require_the_exact_owner_and_private_route(tmp_path):
    token = set_hermes_home_override(str(tmp_path))
    try:
        with _route(actor="owner-b"):
            other_owner = current_chat_scope().owner
        with _route():
            owner = current_chat_scope().owner
            confirmed = "这是已证实的私聊发言。"
            now = time.time()
            for eid, platform, chat, thread, bound, stamp, text in [
                ("ok", "telegram", "shared-route", "topic-a", owner, now - 1, confirmed),
                ("other-owner", "telegram", "shared-route", "topic-a", other_owner, now - 1, "其他人"),
                ("blank-owner", "telegram", "shared-route", "topic-a", "", now - 1, "无身份"),
                ("other-topic", "telegram", "shared-route", "topic-b", owner, now - 1, "其他话题"),
                ("other-chat", "telegram", "other-route", "topic-a", owner, now - 1, "其他聊天"),
                ("other-platform", "discord", "shared-route", "topic-a", owner, now - 1, "其他平台"),
                ("future", "telegram", "shared-route", "topic-a", owner, now + 60, "未来消息"),
                ("expired", "telegram", "shared-route", "topic-a", owner, now - 13 * 3600, "过期消息"),
            ]:
                record_verified_cron_send(execution_id=eid, job_id="confirmed-job", platform=platform,
                                          chat_id=chat, thread_id=thread, bound_owner=bound,
                                          delivered_at=stamp, delivered_text=text)
            result = recent_self_utterances(owner, "next-turn", home=tmp_path)
            assert confirmed in result
            assert all(text not in result for text in (
                "其他人", "无身份", "其他话题", "其他聊天", "其他平台", "未来消息", "过期消息"))
            assert recent_self_utterances(other_owner, "next-turn", home=tmp_path) == ""
        with _route(chat_type="group"):
            assert recent_self_utterances(owner, "next-turn", home=tmp_path) == ""
    finally:
        reset_hermes_home_override(token)
