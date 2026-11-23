from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from contextvars import copy_context
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import time
from threading import Event
from types import SimpleNamespace

import pytest

from test_sanae_mem import make_provider, _trusted_turns
from sanae_mem.cognitive import inner_state
from sanae_mem.cognitive.inner_state import inspect_inner_state
from sanae_mem.infra.status_store import StatusStore


START = 1_700_000_000.0
HOUR = 3600
SHANGHAI = timezone(timedelta(hours=8))
VARS = {"tension": 0.8, "closeness": -0.4, "energy": 0.2}
FIRST = ["我现在很不服。", "我不想就这么算了。"]


def own(provider):
    return str(provider._daily_owner_key or provider._owner_key)


def read_own(provider, **kwargs):
    return provider.inner_state.read_inner_state(owner_key=own(provider), **kwargs)


def write_own(provider, text, **kwargs):
    return provider.inner_state.write_inner_state(text, owner_key=own(provider), **kwargs)


def update(provider, **kwargs):
    return provider.update_inner_state(owner_key=own(provider), **kwargs)


def render_own(provider, state=None, **kwargs):
    return provider.inner_state.render_inner_state(state, owner_key=own(provider), **kwargs)


def at(day, hour, minute=0):
    return datetime(2023, 11, day, hour, minute, tzinfo=SHANGHAI).timestamp()


def answer(sentences=None, recent_words=None, variables=VARS, shared_sentences=None):
    return SimpleNamespace(text=json.dumps({
        "sentences": sentences,
        "variables": variables if sentences else None,
        "recent_words": recent_words,
        "shared_sentences": shared_sentences,
    }, ensure_ascii=False))


def test_inner_state_survives_restart_and_overnight_changes_the_injected_text(tmp_path):
    first = make_provider(tmp_path, user_id="maintainer")
    prompts = []
    commitment = "我答应明天提醒你。"

    def complete(messages, **kwargs):
        prompts.append(json.loads(messages[-1]["content"]))
        return answer(FIRST, commitment) if len(prompts) == 1 else answer()

    first._llm = SimpleNamespace(complete=complete)
    reasoning = "仅本轮主脑可见的临时思考，我有一点不服。"
    assert update(first,
        turn_id="first", user_text="这样说不对", assistant_text=commitment,
        reasoning=reasoning, experiences="刚看完一段短片", now=START,
    )
    assert prompts[0]["reasoning"] == reasoning and prompts[0]["lived_experiences"] == "刚看完一段短片"
    current = read_own(first)
    assert current.text == "".join(FIRST) and current.recent_words == commitment
    assert current.felt_at == current.processed_at == START
    assert first._conn.execute("SELECT name FROM sqlite_master WHERE name='mood_state'").fetchone() is None
    first.shutdown()

    next_chat = make_provider(tmp_path, user_id="maintainer")
    next_chat.on_session_switch("another-session")
    next_chat._llm = SimpleNamespace(complete=complete)
    recent = render_own(next_chat, now=START + 60)
    assert recent.startswith("[我此刻心里]") and "".join(FIRST) in recent
    # 新铁律：消除 recent_words 作为反复尾注的旧活跃职责，心境只留消化后感受
    assert "不必说出来" in recent and "我最近说过" not in recent
    assert read_own(next_chat).recent_words == commitment
    assert "0.8" not in recent and reasoning not in recent

    assert update(next_chat, turn_id="quiet", user_text="早", assistant_text="早。",
                                        now=START + 8 * HOUR)
    assert prompts[-1]["reasoning"] is None
    assert "余味" in prompts[-1]["previous"]["thought"]
    assert prompts[-1]["previous"]["earlier_text"] == "".join(FIRST)
    assert prompts[-1]["previous"]["seconds_since_feeling"] == 8 * HOUR
    overnight = render_own(next_chat, now=START + 8 * HOUR)
    assert "余味" in overnight and FIRST[0] not in overnight and FIRST[1] not in overnight
    assert commitment not in overnight
    assert read_own(next_chat).recent_words == commitment
    quiet = read_own(next_chat)
    assert quiet.felt_at == START and quiet.processed_at == START + 8 * HOUR
    assert abs(quiet.variables["tension"]) < 0.2

    assert update(next_chat, turn_id="another-quiet", user_text="嗯", assistant_text="嗯。",
                                        now=START + 9 * HOUR)
    directory, key = next_chat.inner_state.inner_state_target()
    inspected = inspect_inner_state(directory / "sanae.db", key, recent=5, owner_key=own(next_chat))
    assert inspected["current"]["processed_at"] == START + 9 * HOUR
    assert [row["turn_id"] for row in inspected["recent"]] == ["first"]
    assert reasoning not in json.dumps(inspected, ensure_ascii=False)
    assert read_own(next_chat).revision == inspected["current"]["revision"]


def test_late_result_cannot_overwrite_a_manual_edit_even_across_connections(tmp_path):
    old = make_provider(tmp_path, user_id="maintainer")
    new = make_provider(tmp_path, user_id="maintainer")
    target = old.inner_state.inner_state_target()
    write_own(old, "我还有点想解释。", turn_id="before", now=START)
    entered, release = Event(), Event()
    calls = []

    def slow(messages, **kwargs):
        calls.append(json.loads(messages[-1]["content"]))
        entered.set()
        assert release.wait(4)
        return answer(FIRST)

    old._llm = SimpleNamespace(complete=slow)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(old.update_inner_state, turn_id="late", target=target,
                             user_text="上一轮的话", assistant_text="上一轮回复", now=START + 1,
                             owner_key=own(old))
        try:
            assert entered.wait(2)
            manual = write_own(new, "现在不想聊这个", turn_id="manual", now=START + 2)
            old.shutdown()
        finally:
            release.set()
        assert result.result(timeout=2) is False

    assert read_own(new).private == manual
    assert manual.processed_at is None
    assert "现在不想聊这个" in render_own(new, now=START + 3)
    new._llm = SimpleNamespace(complete=slow)
    assert not update(new, turn_id="queued-before-manual", user_text="旧轮", assistant_text="旧回复",
                                      now=START + 1)
    assert len(calls) == 1
    assert update(new, turn_id="manual", user_text="同轮", assistant_text="同轮回复",
                                  now=START + 3)
    settled = read_own(new)
    assert settled.text == manual.text and settled.thought_source == "manual"
    assert settled.variables == manual.variables and settled.felt_at == manual.felt_at
    assert settled.updated_at == manual.updated_at and settled.processed_at == START + 3
    assert not update(new, turn_id="manual", user_text="重复通知", assistant_text="同轮回复",
                                      now=START + 3.5)
    assert len(calls) == 2

    def from_manual(messages, **kwargs):
        calls.append(json.loads(messages[-1]["content"]))
        return answer()

    new._llm = SimpleNamespace(complete=from_manual)
    assert update(new, turn_id="next", user_text="下一轮", assistant_text="下一轮回复",
                                  now=START + 4)
    assert calls[-1]["previous"]["earlier_text"] == "现在不想聊这个"
    assert calls[-1]["previous"]["source"] == "manual"
    current = read_own(new)
    assert current.text == manual.text and current.thought_source == "manual"
    assert current.processed_at == START + 4 and current.felt_at == START + 2


def test_background_commit_uses_captured_profile_after_provider_shutdown(tmp_path):
    original = make_provider(tmp_path, user_id="maintainer")
    directory, key = target = original.inner_state.inner_state_target()
    different = make_provider(tmp_path / "other", user_id="maintainer", data_dir=str(directory))
    write_own(different, "这是另一个配置里的想法", now=START)
    original._llm = SimpleNamespace(complete=lambda *_a, **_k: answer(FIRST, "这是模型编的承诺"))
    original.shutdown()
    assert update(original,
        target=target, turn_id="old-provider", user_text="想说什么？",
        assistant_text="这件事还没有讲完。",
        now=START + 1,
    )
    assert inspect_inner_state(
        directory / "sanae.db", key, recent=2, owner_key=own(original))["current"]["text"] == "".join(FIRST)
    assert read_own(original, target=target).recent_words == ""
    assert read_own(different).text == "这是另一个配置里的想法"


def test_failed_updates_keep_the_previous_state_and_status_merges_across_providers(tmp_path):
    writer = make_provider(tmp_path, user_id="maintainer")
    monitor = make_provider(tmp_path, user_id="maintainer")
    baseline = write_own(writer, "现在不想聊这个", turn_id="failure", now=START)
    writer.status_store.mutate(lambda status: {**status, "behavior_ok": True})
    writer._llm = SimpleNamespace(complete=lambda *_a, **_k: SimpleNamespace(
        text='{"text":"我以为他这样。","quote":"这一轮"}',
    ))
    assert not update(writer,
        turn_id="failure", user_text="这一轮", assistant_text="收到", now=START + 1,
    )
    assert read_own(writer).private == baseline
    assert writer.status_store.read()["inner_state"]["last_failure"] == "invalid_flash_output"
    monitor.status_store.mutate(lambda status: {**status, "monitor_seen": True})
    assert monitor.status_store.read()["inner_state"]["failure_count"] == 1

    def write_failure(_):
        writer.inner_state.record_inner_state_failure("flash_unavailable", now=START + 2)

    data_dir, _ = monitor.inner_state.inner_state_target()
    status_store = StatusStore(
        current_data_dir=lambda: data_dir, current_session_id=lambda: "session-a",
    )

    def write_daily(_):
        status_store.record_write(datetime.fromtimestamp(START + 2, timezone.utc))

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(copy_context().run, work, i) for i in range(4) for work in (write_failure, write_daily)]
        for job in jobs:
            job.result(timeout=3)
    status = writer.status_store.read()
    assert status["behavior_ok"] and status["monitor_seen"]
    assert status["daily_writes"] == 4
    assert monitor.status_store.read() == status_store.read() == status
    assert read_own(writer).private == baseline
    assert "appraise_rows_written" not in status
    assert status["inner_state"]["failure_count"] == 5
    assert status["inner_state"]["last_failure"] == "flash_unavailable"


def test_sleep_settles_a_strong_feeling_into_afterglow(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    prompts = []
    promise = "我今晚不想理你了。"

    def complete(messages, **kwargs):
        prompts.append(json.loads(messages[-1]["content"]))
        return answer(FIRST, promise) if len(prompts) == 1 else answer()

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    felt = at(15, 1, 30)
    assert update(p, turn_id="upset", user_text="我忘了回你消息",
                                assistant_text=promise, now=felt)
    assert "".join(FIRST) in render_own(p, now=felt + 60)

    woke = render_own(p, now=at(15, 8, 30))
    assert "余味" in woke and "睡了一觉" in woke
    assert FIRST[0] not in woke and FIRST[1] not in woke
    assert not any(char.isdigit() for char in woke)
    for slot in ("凌晨", "早上", "昨天", "昨晚", "今早", "夜里"):
        assert slot not in woke

    assert update(p, turn_id="morning", user_text="早", assistant_text="早。",
                                now=at(15, 8, 30))
    assert "余味" in prompts[-1]["previous"]["thought"]
    assert prompts[-1]["previous"]["earlier_text"] == ""
    assert "".join(FIRST) not in json.dumps(prompts[-1]["previous"], ensure_ascii=False)
    assert prompts[-1]["previous"]["recent_words"] == promise
    assert prompts[-1]["previous"]["seconds_since_feeling"] == 7 * HOUR
    state = read_own(p)
    assert state.text == "".join(FIRST) and state.felt_at == felt

    later = render_own(p, now=at(15, 9, 30))
    assert "余味" in later and promise not in later


def test_sleep_clears_a_light_feeling(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    light = {"tension": 0.2, "closeness": 0.1, "energy": 0.1}
    prompts = []

    def complete(messages, **kwargs):
        prompts.append(json.loads(messages[-1]["content"]))
        return answer(["我有点累了。", "不想多说了。"], None, light) if len(prompts) == 1 else answer()

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    felt = at(15, 1, 30)
    assert update(p, turn_id="tired", user_text="我先去睡了",
                                assistant_text="去睡吧。", now=felt)
    assert "我有点累了。" in render_own(p, now=felt + 60)
    assert render_own(p, now=at(15, 8, 30)) == ""

    assert update(p, turn_id="morning", user_text="早", assistant_text="早。",
                                now=at(15, 8, 31))
    assert prompts[-1]["previous"]["thought"] == ""
    assert prompts[-1]["previous"]["earlier_text"] == ""
    assert prompts[-1]["previous"]["seconds_since_feeling"] is None
    state = read_own(p)
    assert state.text == "" and state.recent_words == ""
    assert render_own(p, now=at(15, 9)) == ""


def test_same_day_feeling_and_words_expire_and_do_not_revive(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    commitment = "我答应明天提醒你。"
    prompts = []

    def complete(messages, **kwargs):
        prompts.append(json.loads(messages[-1]["content"]))
        return answer(FIRST, commitment)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    assert update(p, turn_id="first", user_text="这样说不对",
                                assistant_text=commitment, now=START)
    assert commitment not in render_own(p, now=START + 8 * HOUR)
    assert read_own(p).recent_words == commitment
    words_gone = render_own(p, now=START + 12 * HOUR)
    assert commitment not in words_gone and "余味" in words_gone
    assert render_own(p, now=START + 16 * HOUR) == ""

    assert update(p, turn_id="late", user_text="还在吗", assistant_text="在。",
                                now=START + 16 * HOUR + 60)
    assert prompts[-1]["previous"]["thought"] == ""
    assert prompts[-1]["previous"]["earlier_text"] == ""
    assert prompts[-1]["previous"]["recent_words"] == ""
    state = read_own(p)
    assert state.text == "" and state.recent_words == ""
    assert render_own(p, now=START + 18 * HOUR) == ""


def test_quiet_turns_never_extend_a_feeling_or_its_words(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    commitment = "我答应明天提醒你。"
    calls = 0

    def complete(messages, **kwargs):
        nonlocal calls
        calls += 1
        return answer(FIRST, commitment) if calls == 1 else answer()

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    assert update(p, turn_id="first", user_text="这样说不对",
                                assistant_text=commitment, now=START)
    for hour in (1, 2, 4, 6, 8, 10):
        assert update(p, turn_id=f"quiet-{hour}", user_text="嗯",
                                    assistant_text="嗯。", now=START + hour * HOUR)
        assert read_own(p).felt_at == START
    assert commitment not in render_own(p, now=START + 10 * HOUR)
    assert read_own(p).recent_words == commitment

    assert update(p, turn_id="quiet-12", user_text="嗯", assistant_text="嗯。",
                                now=START + 12 * HOUR)
    assert read_own(p).recent_words == ""
    assert read_own(p).recent_words_at is None
    assert update(p, turn_id="quiet-16", user_text="嗯", assistant_text="嗯。",
                                now=START + 16 * HOUR)
    assert read_own(p).text == ""
    assert render_own(p, now=START + 16 * HOUR + 60) == ""


def test_sleep_clears_negative_energy_and_direction_colors_the_afterglow(tmp_path):
    felt, wake = at(15, 1, 30), at(15, 8, 30)

    def night(tmp_path, variables):
        p = make_provider(tmp_path, user_id="maintainer")
        replies = [answer(FIRST, None, variables), answer()]
        p._llm = SimpleNamespace(complete=lambda *_a, **_k: replies.pop(0))
        assert update(p, turn_id="upset", user_text="我睡不着",
                                    assistant_text="我陪你。", now=felt)
        assert read_own(p).variables["energy"] == variables["energy"]
        assert update(p, turn_id="morning", user_text="早",
                                    assistant_text="早。", now=wake)
        assert read_own(p).variables["energy"] == 0.0
        return render_own(p, now=wake + 60)

    tense = night(tmp_path / "tense", {"tension": 0.9, "closeness": -0.4, "energy": -0.6})
    close = night(tmp_path / "close", {"tension": 0.1, "closeness": 0.9, "energy": -0.6})
    assert "余味" in tense and "绷" in tense
    assert "余味" in close and "靠近" in close
    assert tense != close
    for text in (tense, close):
        assert not any(char.isdigit() for char in text)


def test_manual_write_after_sleep_drops_expired_words_and_keeps_energy_at_zero(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    promise = "我答应明天提醒你。"
    p._llm = SimpleNamespace(complete=lambda *_a, **_k: answer(
        FIRST, promise, {"tension": 0.8, "closeness": -0.4, "energy": -0.5}))
    assert update(p, turn_id="first", user_text="这样说不对",
                                assistant_text=promise, now=at(15, 1, 30))

    manual = write_own(p, "我先睡一会儿。", turn_id="nap", now=at(15, 9, 30))
    assert manual.text == "我先睡一会儿。" and manual.felt_at == at(15, 9, 30)
    assert manual.recent_words == "" and manual.variables["energy"] == 0.0
    rendered = render_own(p, now=at(15, 9, 31))
    assert "我先睡一会儿。" in rendered and promise not in rendered


def test_update_prompt_guides_unfinished_threads_and_softening(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    calls = []
    replies = [
        answer(["我心里还惦记着未完的约定。", "我想等他回来接着说。"],
               "那我等你回来说。", {"tension": 0.3, "closeness": 0.4, "energy": 0.1}),
        answer(),
    ]

    def complete(messages, **kwargs):
        calls.append(messages)
        return replies.pop(0)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    assert update(p, turn_id="thread", user_text="我晚点回来跟你说。",
                                assistant_text="那我等你回来说。", reasoning="   ", now=START)
    payload = json.loads(calls[-1][1]["content"])
    assert payload["previous"] is None and payload["reasoning"] is None
    prompt = calls[-1][0]["content"]
    assert "此刻仍在持续的自身感受与关系取向" in prompt
    assert "不把一方的处境或感受算到另一方头上" in prompt
    assert "previous 的文字不是句式模板" in prompt
    assert "没有有据的新变化时，sentences 返回 null" in prompt
    assert "不将对外建议、认错和接话安排改写成内心待办" in prompt
    assert "描述双方话语或记忆去留只用平实字面动词" in prompt

    assert update(p, turn_id="next", user_text="我回来了。", assistant_text="回来啦。",
                                now=START + 2 * HOUR)
    previous = json.loads(calls[-1][1]["content"])["previous"]
    assert previous["thought"] == "我心里还惦记着未完的约定。我想等他回来接着说。"
    assert previous["earlier_text"] == previous["thought"]
    assert previous["variables"]["tension"] > 0 and previous["variables"]["closeness"] > 0
    assert previous["seconds_since_feeling"] == 2 * HOUR

    p._llm = SimpleNamespace(complete=lambda *_a, **_k: answer(FIRST, "我随口编的一句"))
    assert update(p, turn_id="made-up", user_text="嗯", assistant_text="嗯。",
                                now=START + 3 * HOUR)
    state = read_own(p)
    assert state.recent_words == "那我等你回来说。"
    assert "随口编" not in state.recent_words


def test_the_common_view_rides_the_same_answer_and_shares_one_head(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    calls, payloads = [], []
    replies = [
        answer(FIRST, "我答应明天提醒你。", VARS,
               shared_sentences="我心里还惦记着院子里那阵风。"),
        answer(None, None, None, shared_sentences="我心里还惦记着院子里那阵风。"),
    ]

    def complete(messages, **kwargs):
        calls.append(kwargs.get("purpose"))
        payloads.append(json.loads(messages[-1]["content"]))
        return replies.pop(0)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    try:
        assert update(p, turn_id="common-1", user_text="在吗",
                      assistant_text="我答应明天提醒你。", now=START)
        common = p.inner_state.read_inner_state()  # shared view only
        own = read_own(p)
        assert common is not None and common.shared.text == "我心里还惦记着院子里那阵风。"
        assert own is not None and own.private is not None and own.private.text == "".join(FIRST)
        # One profile chain: the common view and the private fragment share it.
        assert common.head.revision == own.head.revision
        assert common.head.variables == own.head.variables
        assert common.head.processed_at == own.head.processed_at == START

        assert update(p, turn_id="common-2", user_text="嗯", assistant_text="嗯。",
                      now=START + 60)
        assert payloads[-1]["previous"]["shared"]["text"] == "我心里还惦记着院子里那阵风。"
        again = p.inner_state.read_inner_state()
        assert again.shared.text == "我心里还惦记着院子里那阵风。"
        assert again.shared.felt_at == START, "同一句不是新变化，不刷新时钟"
        assert calls == ["inner_state_update"] * 2, "每轮一次心境 Flash，不另开共享调用"
    finally:
        p.shutdown()


def test_an_internal_mood_syncs_itself_and_never_spells_the_lovers_name(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer", partner_name="阿泽")
    replies = [
        answer(["我心里还留着那阵风，惦记着院子里的树。"]),
        answer(["我想把今天发现的那点新鲜事慢慢琢磨清楚。"]),
        answer(["我心里安静了一点。", "我心里还惦记着阿泽那件旧外套。"]),
    ]

    def complete(messages, **kwargs):
        return replies.pop(0)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    try:
        # An autonomous own-activity answer carries no separate shared field; its
        # own sentence is safe and becomes the common view in the same row.
        assert update(p, turn_id="cron:1", user_text="", assistant_text="",
                      experiences="我去院子看了秋天的树。", now=START,
                      preserve_processed_at=True, internal=True)
        common = p.inner_state.read_inner_state()
        assert common is not None and common.text == "我心里还留着那阵风，惦记着院子里的树。"
        assert common.felt_at == START

        # Wording overlap with her own activity is the situation flowing into the
        # mood, not a leak: the same sentence still reaches the common view.
        assert update(p, turn_id="cron:2", user_text="", assistant_text="",
                      experiences="我把今天发现的那点新鲜事记在了本子上。", now=START + 60,
                      preserve_processed_at=True, internal=True)
        grown = p.inner_state.read_inner_state()
        assert grown is not None and grown.text == "我想把今天发现的那点新鲜事慢慢琢磨清楚。"
        assert grown.felt_at == START + 60

        # A sentence that spells the lover's real name stays in her own internal
        # row: only the safe sentence of the same mood becomes the common view.
        assert update(p, turn_id="cron:3", user_text="", assistant_text="",
                      experiences="我把那件旧外套收好了。", now=START + 120,
                      preserve_processed_at=True, internal=True)
        internal = p.inner_state.read_inner_state(internal=True)
        assert internal is not None and "阿泽" in internal.text
        kept = p.inner_state.read_inner_state()
        assert kept is not None and kept.text == "我心里安静了一点。"
        assert kept.felt_at == START + 120 and "阿泽" not in kept.text
    finally:
        p.shutdown()


def test_new_feeling_does_not_renew_the_old_words_clock(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    promise = "我答应明天提醒你。"
    replies = [
        answer(FIRST, promise),
        answer(["我想先安静一会儿。", "我不想再解释了。"], None,
               {"tension": 0.5, "closeness": 0.2, "energy": 0.1}),
    ]
    p._llm = SimpleNamespace(complete=lambda *_a, **_k: replies.pop(0))
    assert update(p, turn_id="t0", user_text="这样说不对",
                                assistant_text=promise, now=START)
    assert read_own(p).recent_words_at == START

    assert update(p, turn_id="t10", user_text="还在吗", assistant_text="在。",
                                now=START + 10 * HOUR)
    state = read_own(p)
    assert state.felt_at == START + 10 * HOUR and state.text == "我想先安静一会儿。我不想再解释了。"
    assert state.recent_words == promise and state.recent_words_at == START

    early = render_own(p, now=START + 11 * HOUR)
    assert promise not in early and "我想先安静一会儿。" in early
    assert read_own(p).recent_words == promise
    late = render_own(p, now=START + 13 * HOUR)
    assert promise not in late and "我想先安静一会儿。" in late


def test_new_words_after_the_old_feeling_expired_show_at_once(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    promise = "我答应明天提醒你。"
    replies = [answer(FIRST), answer(None, promise)]
    p._llm = SimpleNamespace(complete=lambda *_a, **_k: replies.pop(0))
    assert update(p, turn_id="t0", user_text="这样说不对",
                                assistant_text="我听到了。", now=START)
    assert render_own(p, now=START + 16 * HOUR) == ""

    picked = START + 16 * HOUR + 60
    assert update(p, turn_id="late", user_text="在吗", assistant_text=promise, now=picked)
    state = read_own(p)
    assert state.text == "" and state.felt_at == START
    assert state.recent_words == promise and state.recent_words_at == picked

    shown = render_own(p, now=picked + 60)
    assert promise not in shown and "余味" not in shown and FIRST[0] not in shown
    assert read_own(p).recent_words == promise


def test_manual_write_keeps_the_old_words_clock(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    promise = "我答应明天提醒你。"
    p._llm = SimpleNamespace(complete=lambda *_a, **_k: answer(None, promise))
    assert update(p, turn_id="t0", user_text="我出门了", assistant_text=promise, now=START)
    assert read_own(p).recent_words_at == START

    manual = write_own(p, "我想歇一会儿。", turn_id="nap", now=START + 6 * HOUR)
    assert manual.felt_at == START + 6 * HOUR
    assert manual.recent_words == promise and manual.recent_words_at == START
    assert promise not in render_own(p, now=START + 7 * HOUR)
    assert read_own(p).recent_words == promise
    assert promise not in render_own(p, now=START + 13 * HOUR)


def test_same_words_repeated_verbatim_start_a_new_clock(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    promise = "我答应明天提醒你。"
    replies = [answer(FIRST, promise), answer(None, promise), answer(None, "我随口编的一句")]
    p._llm = SimpleNamespace(complete=lambda *_a, **_k: replies.pop(0))
    assert update(p, turn_id="t0", user_text="这样说不对",
                                assistant_text=promise, now=START)

    assert update(p, turn_id="again", user_text="记得吗", assistant_text=promise,
                                now=START + 10 * HOUR)
    state = read_own(p)
    assert state.recent_words == promise and state.recent_words_at == START + 10 * HOUR
    assert promise not in render_own(p, now=START + 14 * HOUR)
    assert read_own(p).recent_words == promise

    assert update(p, turn_id="made-up", user_text="嗯", assistant_text="嗯。",
                                now=START + 16 * HOUR)
    state = read_own(p)
    assert state.recent_words == promise and state.recent_words_at == START + 10 * HOUR
    assert promise not in render_own(p, now=START + 23 * HOUR)


def test_same_turn_lock_refreshes_only_the_words_clock(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    promise = "我答应明天提醒你。"
    manual = write_own(p, "我刚才有点急。", turn_id="same", now=START)
    calls = []

    def complete(messages, **kwargs):
        calls.append(messages)
        return answer(None, promise)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    assert update(p, turn_id="same", user_text="我在听", assistant_text=promise,
                                now=START + 60)
    state = read_own(p)
    assert state.text == manual.text and state.felt_at == manual.felt_at
    assert state.recent_words == promise and state.recent_words_at == START + 60
    assert state.processed_at == START + 60

    assert not update(p, turn_id="same", user_text="重复通知",
                                    assistant_text=promise, now=START + 120)
    assert len(calls) == 1


def test_legacy_inner_state_columns_upgrade_once_and_keep_felt_at(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    promise = "我答应明天提醒你。"
    p._llm = SimpleNamespace(complete=lambda *_a, **_k: answer(FIRST, promise))
    assert update(p, turn_id="old", user_text="这样说不对",
                                assistant_text=promise, now=START)
    directory, key = p.inner_state.inner_state_target()
    p.shutdown()
    database = directory / "sanae.db"

    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute("ALTER TABLE inner_state_revisions DROP COLUMN recent_words_at")

    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute("PRAGMA user_version=0")

    reopened = make_provider(tmp_path, user_id="maintainer")
    state = read_own(reopened)
    assert state.recent_words_at == START
    inspected = inspect_inner_state(database, key, recent=2, owner_key=own(reopened))
    assert inspected["current"]["recent_words"] == promise
    assert inspected["current"]["recent_words_at"] == START
    with closing(sqlite3.connect(database)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(inner_state_revisions)")}
        stored = conn.execute(
            "SELECT recent_words_at FROM inner_state_revisions").fetchone()[0]
    assert "recent_words_at" in columns and stored is None
    assert promise not in render_own(reopened, now=START + 10 * HOUR)
    assert read_own(reopened).recent_words == promise
    assert promise not in render_own(reopened, now=START + 13 * HOUR)
    reopened.shutdown()


def test_affect_and_unspoken_link_have_separate_lifetimes(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    calls = []
    replies = [
        SimpleNamespace(text=json.dumps({
            "sentences": ["我心里挺好奇他看到的是哪一句。", "我还想和他聊下去。"],
            "variables": {"tension": 0.2, "closeness": 0.4, "energy": 0.0},
            "recent_words": None,
            "unspoken_link": "把他的“想不出来”连到了：早上改记忆卡住的事",
        }, ensure_ascii=False)),
        answer(),
    ]

    def complete(messages, **kwargs):
        calls.append(json.loads(messages[-1]["content"]))
        return replies.pop(0)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    try:
        assert update(p,
            turn_id="t1", user_text="想不出来",
            assistant_text="等你愿意说卡住的是哪一点，我听着。",
            reasoning="他是不是还在琢磨早上改记忆卡住的事？",
            now=START,
        )
        rendered_next = render_own(p, now=START + 30)
        assert "我心里挺好奇他看到的是哪一句。" in rendered_next
        assert "（刚才我心里把这句话和「早上改记忆卡住的事」连到了一起，嘴上没有点名。）" in rendered_next

        assert update(p,
            turn_id="t2", user_text="你不理解我，我只是发表情包逗你玩",
            assistant_text="好嘛，我还以为你卡在早上改记忆那件事上呢。",
            now=START + 60,
        )
        assert "连到了一起，但嘴上没点名" not in calls[-1]["previous"]["thought"]
        assert "连到了一起，但嘴上没点名" not in calls[-1]["previous"]["earlier_text"]
        rendered_after = render_own(p, now=START + 90)
        assert "连到了一起，但嘴上没点名" not in rendered_after
        assert "我心里挺好奇他看到的是哪一句。" in rendered_after
    finally:
        p.shutdown()


def test_tool_add_filters_self_rules_and_meta_and_keeps_concrete_fact(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    try:
        pure_rule = json.loads(
            p.handle_tool_call("sanae_mem_add", {"content": "以后他说忙的时候我绝对不能再多嘴追问。"})
        )["result"]
        assert pure_rule["stored"] is False and pure_rule["reason"] == "self_rule_or_meta_filtered"

        cond_rule = json.loads(
            p.handle_tool_call("sanae_mem_add", {"content": "要是他说忙，我以后绝不再多嘴追问。"})
        )["result"]
        assert cond_rule["stored"] is False and cond_rule["reason"] == "self_rule_or_meta_filtered"

        meta_rule = json.loads(p.handle_tool_call("sanae_mem_add", {
            "content": "记忆遗忘机制由他设定，不由我全权决定；若涉及重要记忆，我有权表达保留意愿并拒绝删除。"
        }))["result"]
        assert meta_rule["stored"] is False and meta_rule["reason"] == "self_rule_or_meta_filtered"

        mixed = json.loads(p.handle_tool_call("sanae_mem_add", {
            "content": "今天帮他看课表时把周四的课看成周三了，下次一定看准日期。"
        }))["result"]
        assert mixed["stored"] is True and mixed["id"]
        raw = p._conn.execute("SELECT raw FROM events WHERE id = ?", (mixed["id"],)).fetchone()[0]
        assert raw.endswith("今天帮他看课表时把周四的课看成周三了") and "下次一定" not in raw
    finally:
        p.shutdown()


def seed_distant_event(provider, content, *, tier="long", platform="", topic="", a_conflict=0):
    stamp = datetime.now().astimezone().isoformat()
    source = json.dumps({"platform": platform}, ensure_ascii=False) if platform else ""
    provider._conn.execute(
        "INSERT INTO events (ts, raw, summary, created_at, updated_at, owner_key, tier, topic, a_conflict, source_context)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (stamp, content, content, stamp, stamp, provider._owner_key, tier, topic, a_conflict, source),
    )
    provider._conn.commit()


def spark_answer(sentences, variables, spark=None, unspoken=None):
    return SimpleNamespace(text=json.dumps({
        "sentences": sentences,
        "variables": variables if sentences else None,
        "recent_words": None,
        "unspoken_link": unspoken,
        "spark": spark,
    }, ensure_ascii=False))


def test_distant_memories_skip_neighbours_and_stay_deterministic(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    prompts, payloads = [], []

    def complete(messages, **kwargs):
        prompts.append(messages[0]["content"])
        payloads.append(json.loads(messages[-1]["content"]))
        return spark_answer(FIRST, {"tension": 0.0, "closeness": 0.2, "energy": 0.0})

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    user_text = "刚看完鲸鱼跃出水面的纪录片段"
    assistant_text = "那画面听着就够静的。"
    neighbour = "[今天] 鲸鱼跃出水面那一段看得我出神"
    seed_distant_event(p, neighbour, tier="long")
    seed_distant_event(p, "守矢神社参道傍晚的夕照把石阶染成橘色", tier="important")
    seed_distant_event(p, "猫娘在直播间跟观众讨论咖啡机的控温曲线", tier="distilled")
    seed_distant_event(p, "解析智能咖啡论文里的温度闭环控制", platform="bilibili")
    seed_distant_event(p, "随手记的一件短事", tier="short")
    seed_distant_event(p, "不过天天这样生气也很心累的", tier="long", topic="情绪", a_conflict=1)
    seed_distant_event(p, "我说过：明天早上叫他起床", tier="important")
    try:
        assert update(p,
            turn_id="t1", user_text=user_text, assistant_text=assistant_text, now=START)
        [payload] = payloads
        distant = payload["distant_memories"]
        assert len(distant) == 2
        assert not any("鲸鱼" in value for value in distant), "词面重合的近邻事件不进远距抽样"
        assert "随手记的一件短事" not in distant, "short 非 bilibili 事件不在候选池"
        assert "不过天天这样生气也很心累的" not in distant, "争执/情绪事件不进远距闪念候选池"
        assert not any("我说过：" in value for value in distant), "承诺原话不进远距闪念候选池"
        assert "自然的跨主题关联" in prompts[0] and "distant_memories" in prompts[0]

        directory, profile_key = p.inner_state.inner_state_target()
        again = inner_state._sample_distant_memories(
            directory / "sanae.db", owner_key=p._owner_key, profile_key=profile_key,
            user_text=user_text, assistant_text=assistant_text, turn_id="t1", now=START + 5,
        )
        assert again == distant, "同一 turn_id 的抽样必须可复现"
    finally:
        p.shutdown()


def test_spark_note_shows_once_then_is_stripped_and_cools_down(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    prompts, payloads = [], []
    spark = "看着这漫出来的水冷不丁想起神社檐下那只胖鸽子"
    note = ("（刚才脑子里闪过一个念头：「%s」，"
            "只是过去的一点联想，不是这轮的事实，可以不用管。）" % spark)
    replies = [
        spark_answer(
            FIRST, {"tension": 0.0, "closeness": 0.3, "energy": 0.0}, spark,
            unspoken="把他的“漫出来”连到了：早上一起看鲸鱼视频的事",
        ),
        spark_answer(["换个话题也好。", "我听着呢。"], {"tension": 0.0, "closeness": 0.3, "energy": 0.1}),
        spark_answer(None, None),
    ]

    def complete(messages, **kwargs):
        prompts.append(messages[0]["content"])
        payloads.append(json.loads(messages[-1]["content"]))
        return replies.pop(0)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    seed_distant_event(p, "守矢神社参道傍晚的夕照把石阶染成橘色", tier="important")
    seed_distant_event(p, "解析智能咖啡论文里的温度闭环控制", platform="bilibili")
    try:
        assert update(p,
            turn_id="t1", user_text="刚看完鲸鱼跃出水面的纪录片段",
            assistant_text="那画面听着就够静的。",
            reasoning="他说漫出来，我想到早上一起看鲸鱼视频的事。", now=START,
        )
        assert "distant_memories" in payloads[0]
        text = read_own(p).text
        assert text.endswith(note)
        assert "连到了一起" in text, "unspoken 尾注与 spark 尾注并存"
        rendered = render_own(p, now=START + 30)
        assert note in rendered

        assert update(p,
            turn_id="t2", user_text="换个话题聊点别的吧", assistant_text="行，聊点别的。",
            now=START + 600,
        )
        assert "distant_memories" not in payloads[1], "2 小时冷却内不再注入远距记忆"
        assert "冷不丁" not in payloads[1]["previous"]["thought"]
        assert "冷不丁" not in read_own(p).text
        assert "冷不丁" not in render_own(p, now=START + 660)

        assert update(p,
            turn_id="t3", user_text="那篇论文还在看吗", assistant_text="还在看呢。",
            now=START + 2 * HOUR + 100,
        )
        assert "distant_memories" in payloads[2], "冷却期满后可再次抽样"
    finally:
        p.shutdown()


def test_conflict_or_action_plan_spark_never_leaves_a_note(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    payloads = []
    replies = [
        spark_answer(FIRST, VARS, "这跟檐下那只胖鸽子简直一个路数"),
        spark_answer(None, None, "先听他慢慢说完再说"),
        spark_answer(["换个话题也好。", "我听着呢。"],
                     {"tension": 0.0, "closeness": 0.1, "energy": 0.0}, "先听他慢慢说完再说"),
    ]

    def complete(messages, **kwargs):
        payloads.append(json.loads(messages[-1]["content"]))
        return replies.pop(0)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    seed_distant_event(p, "守矢神社参道傍晚的夕照把石阶染成橘黄色", tier="important")
    seed_distant_event(p, "猫娘在直播间跟观众讨论咖啡机的控温曲线", tier="distilled")
    try:
        assert update(p,
            turn_id="t1", user_text="我今天真的有点委屈，你根本不听我说",
            assistant_text="我在听。", now=START,
        )
        assert "冷不丁" not in read_own(p).text, "本轮紧绷态不得挂出闪念尾注"

        assert update(p,
            turn_id="t2", user_text="刚看完鲸鱼跃出水面的纪录片段",
            assistant_text="那画面听着就够静的。", now=START + 60,
        )
        assert "distant_memories" not in payloads[1], "上一版处于紧绷态时不抽样远距记忆"
        assert "冷不丁" not in read_own(p).text

        assert update(p,
            turn_id="t3", user_text="那个视频里的水看着还挺有意思的",
            assistant_text="嗯，我也觉得。", now=START + 120,
        )
        state = read_own(p)
        assert "冷不丁" not in state.text, "行为计划类闪念直接丢弃"
        assert "先听他慢慢说完再说" not in state.text
        assert "冷不丁" not in render_own(p, now=START + 150)
    finally:
        p.shutdown()



def seed_labeled_memory(provider, content, *, tier="long", topic="", experience="",
                        fact="", source=None):
    stamp = datetime.now().astimezone().isoformat()
    provider._conn.execute(
        "INSERT INTO events (ts, raw, summary, created_at, updated_at, owner_key, tier, topic,"
        " experience_subject, fact_subject, source_context) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (stamp, content, content, stamp, stamp, provider._owner_key, tier, topic,
         experience, fact, json.dumps(source, ensure_ascii=False) if source else ""),
    )
    provider._conn.commit()


def test_distant_spark_uses_video_attribution_and_skips_current_title(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    try:
        stamp = datetime.now().astimezone().isoformat()
        p._conn.execute(
            """INSERT INTO video_watches
               (bvid, platform, source_object_id, title, cid, mode, analysis, summary, captions,
                source_context, read_scope, owner_key, watched_at, created_at, updated_at)
               VALUES ('douyin:dy-9','douyin','dy-9','雨夜实验', 'v1','video_analysis', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("人物把伞沿抬高，路灯倒影在积水里摇晃。",
             "抽样画面分析：雨夜实验出现路灯倒影。",
             json.dumps([{"text": "雨还没有停"}]),
             json.dumps({"kind": "video", "certainty": "observed"}),
             json.dumps({"kind": "owner", "owner": p._owner_key}), p._owner_key,
             stamp, stamp, stamp),
        )
        p._conn.commit()
        directory, profile_key = p.inner_state.inner_state_target()
        sampled = inner_state._sample_distant_memories(
            directory / "sanae.db", owner_key=p._owner_key, profile_key=profile_key,
            user_text="我们改天去夜市看看吧", assistant_text="好啊。",
            turn_id="douyin-spark", now=START + 10 * HOUR,
        )
        assert sampled and "雨夜实验" in sampled[0] and "路灯倒影" in sampled[0]
        assert "〔我在douyin看过的〕" in sampled[0]
        assert "雨还没有停" in sampled[0]
        assert inner_state._sample_distant_memories(
            directory / "sanae.db", owner_key=p._owner_key, profile_key=profile_key,
            user_text="刚好在聊雨夜实验这个片段", assistant_text="嗯。",
            turn_id="douyin-spark-same-title", now=START + 10 * HOUR,
        ) == []
    finally:
        p.shutdown()


from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from typing import Literal


@contextmanager
def _host_plugin_llm_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, responder):
    hc = pytest.importorskip("hermes_host_contracts")
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
    import yaml
    from sanae_mem.cognitive.daily_context import daily_profile_key
    from sanae_mem.cognitive.inner_wiring import wait_for_pending_updates
    from sanae_mem.plugin_entry import SanaeMemProvider

    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOME_OVERRIDE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            messages = body.get("messages") or []
            system = str(messages[0].get("content") or "") if messages else ""
            user_payload = str(messages[-1].get("content") or "") if messages else ""
            content = responder(system, user_payload)
            if content is None:
                resp = json.dumps({"error": {"message": "simulated failure"}}).encode("utf-8")
                status = 500
            else:
                resp = json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": str(body.get("model") or "test-flash-model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                status = 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    llm_base_url = f"http://127.0.0.1:{server.server_port}/v1"

    config: dict[str, object] = {
        "data_dir": str(home / "sanae-mem"),
        "semantic_backend": "sqlite",
        "memory_identity": {"platform": "telegram", "user_id": "maintainer"},
    }
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "provider": "custom",
                    "default": "test-main-model",
                    "base_url": llm_base_url,
                    "api_key": "test-key",
                },
                "auxiliary": {
                    "sanae_daily_flash": {
                        "provider": "custom",
                        "model": "test-flash-model",
                        "base_url": llm_base_url,
                        "api_key": "test-key",
                    }
                },
                "memory": {"provider": "sanae_mem"},
                "plugins": {"enabled": ["sanae_mem"], "sanae_mem": config},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    link = plugins_dir / "sanae_mem"
    if not link.exists():
        link.symlink_to(Path(__file__).resolve().parents[2] / "plugins" / "sanae_mem", target_is_directory=True)

    discover_plugins(force=True)
    p = SanaeMemProvider(config)
    profile_key = daily_profile_key(home)
    target = (home / "sanae-mem", profile_key)

    def wait_all(*, lane: Literal["private", "group", "cron"] | None = "private", timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        ok = wait_for_pending_updates(target, lane=lane, timeout=timeout)
        for name, mod in list(sys.modules.items()):
            if name.endswith(".cognitive.inner_wiring"):
                wait_fn = getattr(mod, "wait_for_pending_updates", None)
                if callable(wait_fn) and wait_fn is not wait_for_pending_updates:
                    remaining = max(0.1, deadline - time.monotonic())
                    ok = bool(wait_fn(target, lane=lane, timeout=remaining)) and ok
        return ok

    @contextmanager
    def turn_scope(message_id: str = "msg-1"):
        reset_session_vars()
        tokens = set_session_vars(
            platform="telegram",
            chat_type="dm",
            chat_id="maintainer",
            user_id="maintainer",
            session_id="session-a",
            session_key="agent:main:telegram:dm:maintainer",
            message_id=message_id,
        )
        try:
            with hc.turn_envelope_scope():
                yield
        finally:
            clear_session_vars(tokens)

    try:
        with turn_scope("init"):
            p.initialize("session-a", hermes_home=str(home), platform="telegram")
        yield SimpleNamespace(
            hc=hc,
            home=home,
            config=config,
            provider=p,
            target=target,
            wait_all=wait_all,
            turn_scope=turn_scope,
        )
    finally:
        get_plugin_manager().unload()
        p.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_spark_is_not_consumed_on_read_and_cooldown_survives_failed_delivery_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    spark = "看着这漫出来的水冷不丁想起神社檐下那只胖鸽子"
    replies = [
        spark_answer(FIRST, {"tension": 0.0, "closeness": 0.3, "energy": 0.0}, spark).text,
        '{"text":"这条不是 JSON 态","quote":"这轮"}',
    ]

    def responder(system: str, _user_payload: str) -> str:
        if "sentences" in system and "variables" in system:
            return replies.pop(0)
        if "new_commitments" in system:
            return '{"new_commitments": [], "resolutions": []}'
        return '{"items": []}'

    now_ts = time.time()
    with _host_plugin_llm_home(tmp_path, monkeypatch, responder) as env:
        p = env.provider
        target = env.target
        seed_distant_event(p, "守矢神社参道傍晚的夕照把石阶染成橘黄色", tier="important")
        seed_distant_event(p, "解析智能咖啡论文里的温度闭环控制", platform="bilibili")

        env.hc.emit_delivery_receipt(
            env.hc.DeliveryReceipt(
                transport_confirmed=True,
                session_id="session-a",
                turn_id="t1",
                platform="telegram",
                chat_type="dm",
                chat_id="maintainer",
                sender_id="maintainer",
                user_message="刚看完鲸鱼跃出水面的纪录片段",
                delivered_text="那画面听着就够静的。",
                clean_text="那画面听着就够静的。",
                delivered_at=now_ts - 60.0,
                incoming_message_id="in-1",
                message_id="out-1",
                session_key="agent:main:telegram:dm:maintainer",
                is_bot=False,
            )
        )
        assert env.wait_all(lane="private", timeout=15.0)

        initial = read_own(p, target=target)
        assert initial is not None and "冷不丁" in initial.text

        first = render_own(p, target=target, session_id="session-a", now=now_ts - 30.0)
        second = render_own(p, target=target, session_id="session-a", now=now_ts - 25.0)
        assert "冷不丁" in first and first == second
        stored = read_own(p, target=target)
        assert stored is not None
        assert "冷不丁" in stored.text, "只读渲染不得改写数据库消费闪念尾注"
        assert "".join(FIRST) in stored.text

        env.hc.emit_delivery_receipt(
            env.hc.DeliveryReceipt(
                transport_confirmed=True,
                session_id="session-a",
                turn_id="t2",
                platform="telegram",
                chat_type="dm",
                chat_id="maintainer",
                sender_id="maintainer",
                user_message="你刚说那个水花挺好看",
                delivered_text="嗯。",
                clean_text="嗯。",
                delivered_at=now_ts - 10.0,
                incoming_message_id="in-2",
                message_id="out-2",
                session_key="agent:main:telegram:dm:maintainer",
                is_bot=False,
            )
        )
        assert env.wait_all(lane="private", timeout=15.0)

        after_settle = read_own(p, target=target)
        assert after_settle is not None
        assert "冷不丁" not in after_settle.text, "送达结算后即使心境更新失败也必须剥离已曝光闪念"
        assert "冷不丁" not in render_own(
            p, target=target, session_id="session-a", now=now_ts,
        ), "后台失败后绝不二次曝光"

        row = p.sqlite.fetchone(
            "SELECT last_spark_at FROM inner_state_revisions "
            "WHERE profile_key=? AND owner_key=? ORDER BY revision DESC LIMIT 1",
            (target[1], own(p)),
        )
        assert row is not None and isinstance(row["last_spark_at"], float)
        last_spark_at = row["last_spark_at"]
        assert inner_state._sample_distant_memories(
            target[0] / "sanae.db", owner_key=own(p), profile_key=target[1],
            user_text="你刚说那个水花挺好看", assistant_text="嗯。", turn_id="t3",
            now=last_spark_at + 2 * HOUR - 1,
        ) == [], "尾注已被剥离，但 last_spark_at 保全的 2 小时冷却仍然生效"


def test_spark_clock_survives_restart_after_the_note_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from sanae_mem.cognitive.inner_state_spark import consume_pending_spark
    from sanae_mem.plugin_entry import SanaeMemProvider

    spark = "看着这漫出来的水冷不丁想起神社檐下那只胖鸽子"
    reply_text = spark_answer(
        FIRST, {"tension": 0.0, "closeness": 0.3, "energy": 0.0}, spark,
    ).text

    def responder(system: str, _user_payload: str) -> str:
        if "sentences" in system and "variables" in system:
            return reply_text
        if "new_commitments" in system:
            return '{"new_commitments": [], "resolutions": []}'
        return '{"items": []}'

    now_ts = time.time()
    with _host_plugin_llm_home(tmp_path, monkeypatch, responder) as env:
        p = env.provider
        target = env.target
        owner_key = own(p)
        seed_distant_event(p, "守矢神社参道傍晚的夕照把石阶染成橘黄色", tier="important")
        seed_distant_event(p, "解析智能咖啡论文里的温度闭环控制", platform="bilibili")

        env.hc.emit_delivery_receipt(
            env.hc.DeliveryReceipt(
                transport_confirmed=True,
                session_id="session-a",
                turn_id="persist-clock",
                platform="telegram",
                chat_type="dm",
                chat_id="maintainer",
                sender_id="maintainer",
                user_message="刚看完鲸鱼跃出水面的纪录片段",
                delivered_text="那画面听着就够静的。",
                clean_text="那画面听着就够静的。",
                delivered_at=now_ts - 30.0,
                incoming_message_id="in-clock-1",
                message_id="out-clock-1",
                session_key="agent:main:telegram:dm:maintainer",
                is_bot=False,
            )
        )
        assert env.wait_all(lane="private", timeout=15.0)
        assert "冷不丁" in render_own(p, target=target, session_id="session-a", now=now_ts - 10.0)
        before_consume = read_own(p, target=target)
        assert before_consume is not None and "冷不丁" in before_consume.text
        assert consume_pending_spark(
            target[0] / "sanae.db", target, owner_key=owner_key, exclude_turn_id="next-turn",
        ) is True
        after_consume = read_own(p, target=target)
        assert after_consume is not None and "冷不丁" not in after_consume.text
        row = p.sqlite.fetchone(
            "SELECT last_spark_at FROM inner_state_revisions "
            "WHERE profile_key=? AND owner_key=? ORDER BY revision DESC LIMIT 1",
            (target[1], owner_key),
        )
        assert row is not None and isinstance(row["last_spark_at"], float)
        last_spark_at = row["last_spark_at"]
        saved_config = dict(env.config)
        saved_home = env.home

    reopened = SanaeMemProvider(saved_config)
    try:
        reopened.initialize("session-a", hermes_home=str(saved_home), platform="telegram")
        assert inner_state._sample_distant_memories(
            target[0] / "sanae.db", owner_key=owner_key,
            profile_key=target[1], user_text="刚才的水花很特别",
            assistant_text="确实。", turn_id="after-restart", now=last_spark_at + 2 * HOUR - 1,
        ) == []
        assert inner_state._sample_distant_memories(
            target[0] / "sanae.db", owner_key=owner_key,
            profile_key=target[1], user_text="刚才的水花很特别",
            assistant_text="确实。", turn_id="after-cooldown", now=last_spark_at + 2 * HOUR + 1,
        )
    finally:
        reopened.shutdown()


def test_conflict_variables_without_sentences_still_land_and_block_the_spark(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")

    def complete(*args, **kwargs):
        return SimpleNamespace(text=json.dumps({
            "sentences": None,
            "variables": {"tension": 0.9, "closeness": -0.2, "energy": 0.0},
            "recent_words": None,
            "spark": "这跟檐下那只胖鸽子简直一个路数",
        }, ensure_ascii=False))

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    try:
        assert update(p,
            turn_id="t1", user_text="刚看完鲸鱼跃出水面的纪录片段",
            assistant_text="那画面听着就够静的。", now=START,
        )
        state = read_own(p)
        assert abs(state.variables["tension"] - 0.9) < 1e-6, "sentences=null 时新紧绷度必须生效"
        assert abs(state.variables["closeness"] + 0.2) < 1e-6
        assert state.changed is True
        assert "冷不丁" not in state.text
        assert "一个路数" not in state.text
    finally:
        p.shutdown()


def test_next_turn_serious_message_shows_then_settles_the_pending_spark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    spark = "这跟檐下那只胖鸽子简直一个路数"
    replies = [
        spark_answer(FIRST, {"tension": 0.0, "closeness": 0.2, "energy": 0.0}, spark).text,
        spark_answer(FIRST, {"tension": 0.0, "closeness": 0.2, "energy": 0.0}).text,
        spark_answer(FIRST, {"tension": 0.0, "closeness": 0.2, "energy": 0.0}, spark).text,
    ]

    def responder(system: str, _user_payload: str) -> str:
        if "sentences" in system and "variables" in system:
            return replies.pop(0)
        if "new_commitments" in system:
            return '{"new_commitments": [], "resolutions": []}'
        return '{"items": []}'

    real = time.time()
    turn = dict(session_id="session-a", turn_id="t2", platform="telegram",
                sender_id="maintainer", sender_id_alt="", chat_type="dm")
    with _host_plugin_llm_home(tmp_path, monkeypatch, responder) as env:
        p = env.provider
        target = env.target
        seed_distant_event(p, "守矢神社参道傍晚的夕照把石阶染成橘黄色", tier="important")
        seed_distant_event(p, "解析智能咖啡论文里的温度闭环控制", platform="bilibili")

        env.hc.emit_delivery_receipt(
            env.hc.DeliveryReceipt(
                transport_confirmed=True,
                session_id="session-a",
                turn_id="t1",
                platform="telegram",
                chat_type="dm",
                chat_id="maintainer",
                sender_id="maintainer",
                user_message="刚看完鲸鱼跃出水面的纪录片段",
                delivered_text="那画面听着就够静的。",
                clean_text="那画面听着就够静的。",
                delivered_at=real - 60.0,
                incoming_message_id="in-s1",
                message_id="out-s1",
                session_key="agent:main:telegram:dm:maintainer",
                is_bot=False,
            )
        )
        assert env.wait_all(lane="private", timeout=15.0)
        state_1 = read_own(p, target=target)
        assert state_1 is not None and "闪过一个念头" in state_1.text

        with env.turn_scope("read-s2"):
            block = p.current_turn_context(**dict(turn, user_message="我现在很忙，说正事"))
            again = p.current_turn_context(**dict(turn, user_message="我现在很忙，说正事"))
        assert "闪过一个念头" in block, "取消严肃词表后不再在主脑前硬性剥离，由主脑判断是否使用"
        assert block == again, "送达前重复调用 current_turn_context 必须幂等"
        state_after_read = read_own(p, target=target)
        assert state_after_read is not None and "闪过一个念头" in state_after_read.text, "只读渲染不得消费闪念"
        assert "".join(FIRST) in block

        env.hc.emit_delivery_receipt(
            env.hc.DeliveryReceipt(
                transport_confirmed=True,
                session_id="session-a",
                turn_id="t3",
                platform="telegram",
                chat_type="dm",
                chat_id="maintainer",
                sender_id="maintainer",
                user_message="刚看到鲸鱼跃出水面的片段",
                delivered_text="那画面听着就够静的。",
                clean_text="那画面听着就够静的。",
                delivered_at=real - 20.0,
                incoming_message_id="in-s3",
                message_id="out-s3",
                session_key="agent:main:telegram:dm:maintainer",
                is_bot=False,
            )
        )
        assert env.wait_all(lane="private", timeout=15.0)
        state_after_settle = read_own(p, target=target)
        assert state_after_settle is not None and "闪过一个念头" not in state_after_settle.text
        with env.turn_scope("read-s4"):
            angry = p.current_turn_context(**dict(turn, turn_id="t4", user_message="你根本没听我说，我很委屈"))
        assert "闪过一个念头" not in angry

        env.hc.emit_delivery_receipt(
            env.hc.DeliveryReceipt(
                transport_confirmed=True,
                session_id="session-a",
                turn_id="t5",
                platform="telegram",
                chat_type="dm",
                chat_id="maintainer",
                sender_id="maintainer",
                user_message="刚看到鲸鱼跃出水面的片段",
                delivered_text="那画面听着就够静的。",
                clean_text="那画面听着就够静的。",
                delivered_at=real + 7260.0,
                incoming_message_id="in-s5",
                message_id="out-s5",
                session_key="agent:main:telegram:dm:maintainer",
                is_bot=False,
            )
        )
        assert env.wait_all(lane="private", timeout=15.0)
        state_5 = read_own(p, target=target)
        assert state_5 is not None and "闪过一个念头" in state_5.text
        with env.turn_scope("read-s6"):
            setback = p.current_turn_context(**dict(
                turn, turn_id="t6",
                user_message="因为这个事笑不出来，刚忙活了一下午的方案全白费了，全得推倒重来",
            ))
        assert "闪过一个念头" in setback, "取消严肃词表后由主脑决定是否忽略"
        state_6 = read_own(p, target=target)
        assert state_6 is not None and "闪过一个念头" in state_6.text, "未送达前保持只读"


def test_distant_memories_keep_attribution_and_stay_cross_topic(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    owner = p._owner_key
    expected = {}
    seed_labeled_memory(p, "守矢神社参道傍晚的夕照把石阶染成橘黄色", topic="风景",
                        experience="sanae", fact="sanae")
    expected["〔我自己的经历〕守矢神社参道傍晚的夕照把石阶染成橘黄色"] = "风景"
    seed_labeled_memory(p, "解析智能咖啡论文里的温度闭环控制", topic="技术",
                        source={"platform": "bilibili", "kind": "self",
                                "certainty": "observed"})
    expected["〔我在B站看过的〕解析智能咖啡论文里的温度闭环控制"] = "技术"
    seed_labeled_memory(p, "小时候在河边捞鱼的事", topic="生活",
                        experience="sanae", fact=f"owner:{owner}",
                        source={"kind": "maintainer"})
    expected["〔他说过的事〕小时候在河边捞鱼的事"] = "生活"
    seed_labeled_memory(p, "老家早餐爱吃肠粉和云吞", topic="饮食",
                        experience=f"owner:{owner}", fact=f"owner:{owner}")
    expected["〔他的事〕老家早餐爱吃肠粉和云吞"] = "饮食"
    try:
        directory, profile_key = p.inner_state.inner_state_target()
        with sqlite3.connect(directory / "sanae.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT summary, raw, topic, experience_subject, fact_subject, source_context "
                "FROM events ORDER BY id"
            ).fetchall()
        labels = {inner_state._distant_memory_attribution(row) for row in rows}
        assert labels == {"〔我自己的经历〕", "〔我在B站看过的〕", "〔他说过的事〕", "〔他的事〕"}
        sampled = inner_state._sample_distant_memories(
            directory / "sanae.db", owner_key=owner, profile_key=profile_key,
            user_text="刚看到一个讲鲸鱼跃出水面的片段", assistant_text="那画面听着就够静的。",
            turn_id="pair-1", now=START,
        )
        assert len(sampled) == 2
        assert all(item in expected for item in sampled)
        assert expected[sampled[0]] != expected[sampled[1]], "抽出的两条必须彼此跨主题"
        bodies = [item[item.index("〕") + 1:] for item in sampled]
        assert len(inner_state._body_tokens(bodies[0])
                   & inner_state._body_tokens(bodies[1])) < 2, "两条不得词面近邻"
    finally:
        p.shutdown()

    dense = make_provider(tmp_path / "dense", user_id="maintainer")
    seed_labeled_memory(dense, "守矢神社参道傍晚的夕照把石阶染成橘黄色", topic="风景",
                        experience="sanae", fact="sanae")
    seed_labeled_memory(dense, "守矢神社参道夕照里石阶和人影都染成橘黄色", topic="风景",
                        experience="sanae", fact="sanae")
    try:
        directory, profile_key = dense.inner_state.inner_state_target()
        only = inner_state._sample_distant_memories(
            directory / "sanae.db", owner_key=dense._owner_key, profile_key=profile_key,
            user_text="刚看到一个讲鲸鱼跃出水面的片段", assistant_text="那画面听着就够静的。",
            turn_id="pair-2", now=START,
        )
        assert len(only) == 1, "候选彼此词面高度重合时只取 1 条"
    finally:
        dense.shutdown()


def test_attribution_keeps_subject_ownership_ahead_of_the_bilibili_platform():
    owner = "owner:abc123"

    def row(experience="", fact="", source=""):
        return {"experience_subject": experience, "fact_subject": fact,
                "source_context": json.dumps(source, ensure_ascii=False) if source else ""}

    assert inner_state._distant_memory_attribution(row(
        "sanae", owner, {"kind": "maintainer", "platform": "bilibili"},
    )) == "〔他说过的事·bilibili〕"
    assert inner_state._distant_memory_attribution(row(
        owner, owner, {"platform": "bilibili"},
    )) == "〔他的事·bilibili〕"
    assert inner_state._distant_memory_attribution(row(
        "sanae", "sanae", {"kind": "self", "platform": "bilibili", "certainty": "observed"},
    )) == "〔我在B站看过的〕"
    assert inner_state._distant_memory_attribution(row(
        "sanae", "sanae", {"platform": "bilibili", "certainty": "unverified"},
    )) == "〔我在B站看过的（未核实）〕"
    assert inner_state._distant_memory_attribution(row(
        "", "", {"kind": "video", "certainty": "unverified"},
    )) == "〔B站视频内容（未核实）〕"
    assert inner_state._distant_memory_attribution(row(
        "", "", {"kind": "external_user", "certainty": "unverified"},
    )) == "〔别人说的事（未核实）〕"
    assert inner_state._distant_memory_attribution(row()) == "〔过往记忆（来源未细分）〕"


def test_spark_prompt_keeps_the_vocabulary_discipline():
    prompt = inner_state._SPARK_PROMPT
    for banned in ("委屈", "难受", "生气", "看着", "冷不丁想起", "简直一个路数"):
        assert banned not in prompt
    assert "归属标签" in prompt and "variables.tension > 0.2" in prompt


@pytest.mark.parametrize("closeness, expected", [(0.65, True), (0.30, True), (0.2, False), (0.1, False)])
def test_long_sleep_15h_preserves_strong_closeness_afterglow(tmp_path, closeness, expected):
    p = make_provider(tmp_path, user_id="maintainer")
    p._llm = SimpleNamespace(complete=lambda *_a, **_k: answer(
        ["我还有很多话想慢慢聊。", "今天做过的事一直记在心里。"],
        variables={"tension": 0.0, "closeness": closeness, "energy": 0.0}))
    start = at(8, 2, 10)
    try:
        assert update(p, turn_id="night", user_text="晚安", assistant_text="晚安。", now=start)
        block = render_own(p, now=at(8, 17, 1))
        assert bool(block) == expected
        if expected:
            assert "睡了一觉" in block and "靠近" in block
        assert render_own(p, now=start + 24 * HOUR) == ""
        assert render_own(p, now=start + 365 * 24 * HOUR) == ""
    finally:
        p.shutdown()


def test_update_inner_state_rejects_reply_parroting_without_refreshing_the_feeling(tmp_path):
    p = make_provider(tmp_path, user_id="maintainer")
    base = write_own(p, "我想慢慢把今天的事情想明白。", now=START)
    reply = "今天我会把神社院子里的落叶扫干净，再去厨房看看。"
    p._llm = SimpleNamespace(complete=lambda *_a, **_k: answer([reply, "他说今天还有很多事情。"],
        variables={"tension": 0.0, "closeness": 0.4, "energy": 0.0}))
    try:
        assert update(p, turn_id="parrot", user_text="今天呢？", assistant_text=reply, now=START + 1)
        state = read_own(p)
        assert state.text == base.text and state.felt_at == base.felt_at
        assert inner_state._reject_reply_parrot(["听他说这些", "他提到工作", "我想去看看屋后的树。"], reply) == ["我想去看看屋后的树。"]
        assert inner_state._reject_reply_parrot(["干净落叶神社院子今天厨房再去"], reply) == []
        for prompt in (inner_state._UPDATE_PROMPT, inner_state._SPARK_PROMPT, inner_state._CRON_HEARTBEAT_PROMPT):
            assert not any(word in prompt for word in ("委屈", "难受", "生气", "懵", "拿不准", "冷不丁想起"))
        p._llm = SimpleNamespace(complete=lambda *_a, **_k: answer([reply, "他说还有一件事。"],
            variables={"tension": 0.0, "closeness": 0.0, "energy": 0.0}))
        assert update(p, turn_id="expired", user_text="后来呢？", assistant_text=reply, now=START + 25 * HOUR)
        assert read_own(p).text == ""
    finally:
        p.shutdown()


def test_compact_distant_memory_text_strips_video_disclaimer_and_keeps_quotes():
    raw = (
        "[2026-11-04 22:28|今天晚上] 早苗在B站收到该视频的抽样画面分析，音轨未核验：《有一个灵梦前来买瓜》"
        "（BV1NCDHBcEpr，CID 37356503506）。范围：sampled_visual_analysis。来源：video_analysis；"
        "内容理解未经独立核实：以下按约 **00:00–01:29.8** 的画面顺序描述，时间均为近似值。人物以外观区分。\n\n"
        "## 一、可见事件与时间顺序\n"
        "- **00:02.6–00:07.9**：画面切到林间西瓜摊。绿发女孩戴着青蛙状发饰，手持记事本。\n"
        "## 二、画面内可见文字\n"
        "- **00:08.6–00:17.2**：“苗姐这瓜多少钱一斤”“1000日元一斤”“这瓜保熟嘛？”“要是不熟我就把青蛙卖给你”"
    )
    summary = raw[:150]
    compacted = inner_state._compact_distant_memory_text({"summary": summary, "raw": raw})
    assert "《有一个灵梦前来买瓜》" in compacted
    assert "林间西瓜摊" in compacted
    assert "这瓜保熟嘛？" in compacted
    assert "要是不熟我就把青蛙卖给你" in compacted
    assert "sampled_visual_analysis" not in compacted



def test_persona_and_partner_prompts_are_configurable_and_free_of_private_nickname(tmp_path):
    from sanae_mem.cognitive.impression import IMPRESSION_PROMPT
    from sanae_mem.governance.llm_guard import _RESIDENT_ROUTE_SYSTEM_TEMPLATE

    assert "user 是对方本轮说的话" in inner_state._UPDATE_PROMPT
    assert "对方（user）是我的挚友" in IMPRESSION_PROMPT
    assert "用户（对方）是{persona_name}的挚友" in _RESIDENT_ROUTE_SYSTEM_TEMPLATE

    p = make_provider(
        tmp_path,
        persona_name="阿燐",
        partner_name="觉大人",
        partner_role="挚友",
        partner_pronoun="她",
        user_id="maintainer",
    )
    calls = []

    def complete(messages, **kwargs):
        calls.append(messages)
        return answer(FIRST)

    p.attach_runtime_llm(SimpleNamespace(complete=complete))
    try:
        imp_target = (p.inner_state.inner_state_target()[0], p._daily_profile_key, p._daily_owner_key)
        assert update(p,
            turn_id="t-custom", user_text="今天地灵殿很安静。", assistant_text="嗯，我也觉得。",
            now=START, impression_target=imp_target,
        )
        system_prompt = calls[-1][0]["content"]
        assert "你就是阿燐" in system_prompt
        assert "user 是觉大人" in system_prompt
        assert "觉大人（user）是我的挚友" in system_prompt
        assert "（称呼她一律用「她」，不用「他」）" in system_prompt
        assert "男性" not in system_prompt and "女性" not in system_prompt
        assert "我最近眼里的她" in system_prompt
    finally:
        p.shutdown()
