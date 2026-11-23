from __future__ import annotations

import json
import sys
import types
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem import SanaeMemProvider  # noqa: E402
from sanae_mem.runtime import group_memory  # noqa: E402
from sanae_mem.infra import host_bridge  # noqa: E402

GROUP_A = "default/discord/dom-a/guild-a"
GROUP_B = "default/discord/dom-b/guild-b"
OWNER_A = group_memory.GROUP_OWNER_PREFIX + GROUP_A
OWNER_B = group_memory.GROUP_OWNER_PREFIX + GROUP_B
PRIVATE_OWNER = "person:private-a"


def _config(defaults: bool = False, overrides: dict | None = None) -> str:
    lines = ["plugins:", "  sanae_mem:", "    group_context:", "      defaults:",
             f"        long_term_memory: {str(defaults).lower()}",
             f"        short_term_shared: {str(defaults).lower()}", "      overrides:"]
    for key, value in (overrides or {}).items():
        lines.append(f'        "{key}":')
        lines.append(f"          long_term_memory: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def _home(tmp_path: Path, *, defaults: bool = False, overrides: dict | None = None) -> Path:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(_config(defaults, overrides), encoding="utf-8")
    return home


def _use_home(monkeypatch, home: Path) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    return home


@contextmanager
def _turn(*, chat_id: str = "guild-a", platform: str = "discord",
          scope_id: str = "dom-a", chat_type: str = "group", thread_id: str = "",
          parent_chat_id: str = "", thread_visibility: str = "",
          synthetic: bool = False, sender: str = "account-a",
          session_key: str = "session-a", message_id: str = "m-1"):
    with host_bridge.turn_envelope_scope(
        platform=platform, chat_type=chat_type, chat_id=chat_id, sender_id=sender,
        sender_id_alt="", session_id="session-a", session_key=session_key,
        message_id=message_id, scope_id=scope_id, thread_id=thread_id,
        parent_chat_id=parent_chat_id, thread_visibility=thread_visibility,
        synthetic_reply_thread=synthetic, occurred_at="2026-05-01T10:00:00+08:00",
        is_bot=False,
    ):
        yield


def _provider(home: Path, stack: ExitStack, *, context: str = "primary") -> SanaeMemProvider:
    stack.enter_context(host_bridge.turn_envelope_scope(
        platform="cli", session_id="fixture"))
    provider = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
        "distill_daily_cap": 10,
    })
    provider.initialize("session-a", hermes_home=str(home), platform="discord",
                        agent_context=context)
    return provider


def _add(provider: SanaeMemProvider, text: str, importance: float = 0.8) -> int:
    event_id = provider.add_group_event(text, importance=importance)
    assert isinstance(event_id, int)
    return event_id


def _row(provider: SanaeMemProvider, event_id: int):
    return provider._conn.execute(
        "SELECT raw, summary, summary_source, distilled_at, owner_key, source_context, "
        "topic, importance, ts, vec FROM events WHERE id=?", (int(event_id),)).fetchone()


class _CaptureLlm:
    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.on_call is not None:
            self.on_call()
        return types.SimpleNamespace(
            text=json.dumps({"summary": f"摘要-{len(self.calls)}"}, ensure_ascii=False))


@pytest.fixture
def stack():
    with ExitStack() as entered:
        yield entered


def test_enabled_group_maintenance_summarizes_only_each_own_row(tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _home(
        tmp_path, overrides={GROUP_A: True, GROUP_B: True}))
    provider = _provider(home, stack)
    now = datetime.now().astimezone()
    private_id = provider.event_store.insert_event_row(
        "private-sentinel", (now.isoformat(), "PRIVATE-SENTINEL", "kw", 0.3, 0.6,
                             "topic", None, now.isoformat(), None, "session-p"),
        owner_key=PRIVATE_OWNER)
    assert isinstance(private_id, int)

    with _turn():
        a_id = _add(provider, "ALPHA-SENTINEL only group a", importance=0.9)
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b", message_id="m-b"):
        b_id = _add(provider, "BETA-SENTINEL only group b", importance=0.8)
    with _turn(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
               parent_chat_id="guild-a", thread_visibility="narrower",
               session_key="session-t1", message_id="m-t1"):
        sibling_id = _add(provider, "SIBLING-SENTINEL only narrow thread", importance=0.7)

    before = {event_id: dict(_row(provider, event_id)) for event_id in
              (private_id, a_id, b_id, sibling_id)}
    background = _provider(home, stack, context="cron")
    llm = _CaptureLlm()
    background._llm = llm
    result = background.lifecycle.distill(now)

    group_calls = [call for call in llm.calls if call[1].get("purpose") == "group_distill"]
    assert len(group_calls) == 3
    sentinels = ("ALPHA-SENTINEL", "BETA-SENTINEL", "SIBLING-SENTINEL")
    for messages, _ in group_calls:
        prompt = messages[-1]["content"]
        assert sum(sentinel in prompt for sentinel in sentinels) == 1
        assert "PRIVATE-SENTINEL" not in prompt
        assert "USER.md" not in prompt and "MEMORY.md" not in prompt
    alpha_prompt = next(messages[-1]["content"] for messages, _ in group_calls
                        if "ALPHA-SENTINEL" in messages[-1]["content"])
    assert "BETA-SENTINEL" not in alpha_prompt
    assert "SIBLING-SENTINEL" not in alpha_prompt
    assert '"chat_id": "guild-a"' in alpha_prompt

    for event_id, owner in ((a_id, OWNER_A), (b_id, OWNER_B),
                            (sibling_id, OWNER_A + "/threads/thread-1")):
        row = _row(background, event_id)
        assert row["raw"] == before[event_id]["raw"]
        assert row["source_context"] == before[event_id]["source_context"]
        assert row["owner_key"] == owner
        assert row["summary"].startswith("摘要-")
        assert row["summary_source"] == "distilled"
        assert row["distilled_at"]
    assert _row(background, private_id)["summary"] == before[private_id]["summary"]
    assert result["items"] and result["candidates"] == 3

    llm.calls.clear()
    again = background.lifecycle.distill(now)
    assert again["items"] == []
    assert not [call for call in llm.calls if call[1].get("purpose") == "group_distill"]


def test_group_maintenance_uses_the_one_existing_daily_cap(tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _home(tmp_path, overrides={GROUP_A: True}))
    provider = _provider(home, stack)
    now = datetime.now().astimezone()
    with _turn():
        _add(provider, "CAP-FIRST", importance=0.9)
        _add(provider, "CAP-SECOND", importance=0.8)

    background = _provider(home, stack, context="cron")
    background._distill_daily_cap = 1
    llm = _CaptureLlm()
    background._llm = llm
    result = background.lifecycle.distill(now)

    group_calls = [call for call in llm.calls if call[1].get("purpose") == "group_distill"]
    assert len(group_calls) == 1
    assert len(result["items"]) == 1


def test_global_true_and_explicit_override_off_follow_current_config(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _home(tmp_path, defaults=True))
    provider = _provider(home, stack)
    now = datetime.now().astimezone()
    with _turn():
        a_id = _add(provider, "ALPHA-GLOBAL-SENTINEL", importance=0.9)
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b", message_id="m-b"):
        b_id = _add(provider, "BETA-GLOBAL-SENTINEL", importance=0.8)

    # A explicit false wins over the true global default; B keeps the global default.
    (home / "config.yaml").write_text(
        _config(True, {GROUP_A: False}), encoding="utf-8")
    background = _provider(home, stack, context="cron")
    llm = _CaptureLlm()
    background._llm = llm
    background.lifecycle.distill(now)

    prompts = [messages[-1]["content"] for messages, kwargs in llm.calls
               if kwargs.get("purpose") == "group_distill"]
    assert len(prompts) == 1 and "BETA-GLOBAL-SENTINEL" in prompts[0]
    assert "ALPHA-GLOBAL-SENTINEL" not in prompts[0]
    assert _row(background, a_id)["distilled_at"] is None
    assert _row(background, b_id)["distilled_at"]


def test_group_disabled_while_model_runs_is_not_updated(tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _home(tmp_path, overrides={GROUP_A: True}))
    provider = _provider(home, stack)
    now = datetime.now().astimezone()
    with _turn():
        event_id = _add(provider, "CLOSE-DURING-CALL-SENTINEL", importance=0.9)

    background = _provider(home, stack, context="cron")
    before = background.sqlite.fetchone("SELECT * FROM events WHERE id=?", (event_id,))
    assert before is not None
    llm = _CaptureLlm(
        on_call=lambda: (home / "config.yaml").write_text(
            _config(False, {GROUP_A: False}), encoding="utf-8"))
    background._llm = llm
    result = background.lifecycle.distill(now)

    assert len(llm.calls) == 1
    after = background.sqlite.fetchone("SELECT * FROM events WHERE id=?", (event_id,))
    assert after is not None and tuple(after) == tuple(before)
    assert result["items"] == []
    assert result["zero_output_reasons"].get("write_failed") == 1


def test_id_summary_entries_reject_other_group_and_parent_thread(tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _home(
        tmp_path, overrides={GROUP_A: True, GROUP_B: True}))
    provider = _provider(home, stack)
    now = datetime.now().astimezone()
    with _turn():
        a_id = _add(provider, "alpha id scope", importance=0.9)
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b", message_id="m-b"):
        b_id = _add(provider, "beta id scope", importance=0.8)
    b_before = _row(provider, b_id)

    with _turn():
        assert provider.event_maintenance.get_distillable_event(a_id, now.isoformat()) is not None
        assert provider.event_maintenance.get_distillable_event(b_id, now.isoformat()) is None
        assert provider.event_maintenance.promote_distilled_event(
            b_id, "must not write", "", b"", now.isoformat()) == 0
    assert _row(provider, b_id)["summary"] == b_before["summary"]

    with _turn(chat_type="dm", chat_id="dm-a"):
        provider.on_turn_start(1, "")
        added = json.loads(provider.handle_tool_call("sanae_mem_add", {"content": "记住私聊里提起的原话"}))
        assert added["ok"] and added["result"]["stored"]

    with _turn(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
               parent_chat_id="guild-a", thread_visibility="narrower",
               session_key="session-t1", message_id="m-t1"):
        narrow_id = _add(provider, "narrow id scope", importance=0.7)
        assert provider.event_maintenance.get_distillable_event(added["result"]["id"], now.isoformat()) is None
        assert provider.event_maintenance.get_distillable_event(a_id, now.isoformat()) is None
        assert provider.event_maintenance.promote_distilled_event(
            a_id, "must not touch parent", "", b"", now.isoformat()) == 0
        assert provider.event_maintenance.get_distillable_event(narrow_id, now.isoformat()) is not None


def test_bound_person_distills_its_old_private_owner_row_by_id(
        tmp_path, monkeypatch, stack):
    """A bound account on another platform summarizes its own pre-binding row by that old id.

    The historical private owner is the Slack account key this person wrote under before the
    declaration; the real resolver puts it into ``private_read_owner_keys``. The row keeps its
    owner and source, and another person's old owner is still refused.
    """
    from sanae_mem.governance.account_binding import maintainer_anchor
    from sanae_mem.infra.textutil import _owner_key

    home = _use_home(monkeypatch, _home(tmp_path))
    config = {
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
        "distill_daily_cap": 10,
        "memory_identity": {"platform": "discord", "user_id": "anchor-1"},
        "maintainer_account_bindings": [{
            "profile": "default", "platform": "slack", "user_id": "U_SAME",
            "scope_id": "T_TEAM",
        }],
    }
    stack.enter_context(host_bridge.turn_envelope_scope(
        platform="cli", session_id="fixture"))
    provider = SanaeMemProvider(config)
    provider.initialize("session-a", hermes_home=str(home), platform="slack",
                        agent_context="primary")
    now = datetime.now().astimezone()
    anchor = maintainer_anchor(config)
    old_owner = _owner_key("slack", user_id="U_SAME", scope_id="T_TEAM")
    other_owner = _owner_key("slack", user_id="U_OTHER", scope_id="T_TEAM")
    old_source = json.dumps({"kind": "external_user", "visibility": "private",
                             "chat_id": "D_OLD"}, ensure_ascii=False)
    old_id = provider.event_store.insert_event_row(
        "old-own-row", (now.isoformat(), "OLD-OWN-ROW", "kw", 0.3, 0.6, "topic", None,
                        now.isoformat(), None, "session-old"),
        owner_key=old_owner, source_context=old_source)
    other_id = provider.event_store.insert_event_row(
        "old-other-row", (now.isoformat(), "OLD-OTHER-ROW", "kw", 0.3, 0.6, "topic", None,
                          now.isoformat(), None, "session-other"),
        owner_key=other_owner, source_context=old_source)
    assert isinstance(old_id, int) and isinstance(other_id, int)

    with _turn(platform="slack", chat_type="dm", chat_id="D_SAME", scope_id="T_TEAM",
               sender="U_SAME", session_key="slack-dm", message_id="m-dm"):
        provider.on_turn_start(1, "")
        assert provider._owner_key == anchor
        assert old_owner != anchor, "the old row lives under the pre-binding account owner"
        assert old_owner in provider._read_owner_keys, "Task 02 keeps the old owner readable"
        assert provider.event_maintenance.get_distillable_event(old_id, now.isoformat()) is not None
        assert provider.lifecycle.distill_promote(old_id, "旧事的一句话摘要", now) is True
        row = _row(provider, old_id)
        assert (row["owner_key"], row["source_context"]) == (old_owner, old_source)
        assert row["raw"] == "OLD-OWN-ROW"
        assert row["summary_source"] == "distilled" and row["distilled_at"]

        # Another person's row under that platform's own owner is not this person's history.
        assert provider.event_maintenance.get_distillable_event(other_id, now.isoformat()) is None
        assert provider.lifecycle.distill_promote(other_id, "别人的旧事", now) is False
    other = _row(provider, other_id)
    assert other["summary_source"] != "distilled" and other["distilled_at"] is None
    assert other["owner_key"] == other_owner
    provider.shutdown()


def test_inherited_thread_reads_its_old_exact_row_but_parent_does_not(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _home(tmp_path, overrides={GROUP_A: True}))
    provider = _provider(home, stack)
    with _turn(chat_type="thread", chat_id="thread-3", thread_id="thread-3",
               parent_chat_id="guild-a", thread_visibility="narrower",
               session_key="session-t3", message_id="m-t3"):
        exact_id = _add(provider, "legacy exact row", importance=0.7)
        source = json.loads(_row(provider, exact_id)["source_context"])
    source["thread_visibility"] = "inherited"
    provider._conn.execute(
        "UPDATE events SET source_context=? WHERE id=?",
        (json.dumps(source, ensure_ascii=False, sort_keys=True), exact_id))
    provider._conn.commit()

    with _turn(chat_type="thread", chat_id="thread-3", thread_id="thread-3",
               parent_chat_id="guild-a", thread_visibility="inherited",
               session_key="session-t3", message_id="m-t3b"):
        assert provider._group_scope_grant().read_owners == (
            OWNER_A, OWNER_A + "/threads/thread-3")
        assert provider.event_store.event_ids_existing([exact_id]) == [exact_id]

    with _turn():
        assert provider.event_store.event_ids_existing([exact_id]) == []


@pytest.mark.parametrize("backend", ["sqlite", "chroma"])
def test_group_receipt_summary_does_not_consume_daily_cap_and_updates_chroma(
        tmp_path, monkeypatch, stack, backend):
    from sanae_mem.runtime import group_ingest
    from sanae_mem.infra.textutil import _embed, _pack_vector
    from sanae_mem.infra.words import _COLLECTION

    chromadb = pytest.importorskip("chromadb") if backend == "chroma" else None

    home = _use_home(monkeypatch, _home(tmp_path, overrides={GROUP_A: True}))
    stack.enter_context(host_bridge.turn_envelope_scope(platform="cli", session_id="fixture"))
    provider = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": backend, "shadow": False,
        "distill_daily_cap": 2,
    })
    provider.initialize("session-a", hermes_home=str(home), platform="discord")
    stack.callback(provider.shutdown)
    collection = None
    if chromadb is not None:
        client = chromadb.PersistentClient(path=str(home / "data" / "chroma"))
        stack.callback(client.close)
        collection = client.get_collection(_COLLECTION)

    now = datetime.now().astimezone()
    p1 = provider.event_store.insert_event_row(
        "priv-1", (now.isoformat(), "这是一件值得记下来的私人重要经历", "重要", 0.9, 0.6,
                   "生活", None, now.isoformat(), None, "session-p"),
        owner_key=PRIVATE_OWNER,
    )
    p2 = provider.event_store.insert_event_row(
        "priv-2", (now.isoformat(), "这是另一件值得记住的私人开心决定", "开心", 0.85, 0.6,
                   "生活", None, now.isoformat(), None, "session-p"),
        owner_key=PRIVATE_OWNER,
    )
    assert isinstance(p1, int) and isinstance(p2, int)

    group_event_ids = []
    expected_summaries = []
    for idx in range(3):
        quote = f"周六去公园散步第{idx}次"
        summary = f"摘要结果第{idx}条"
        expected_summaries.append(summary)
        payload = {
            "events": [{"text": summary, "quote": quote}],
            "impression": None,
            "new_commitments": [],
            "commitment_updates": [],
        }
        provider._llm = types.SimpleNamespace(
            complete=lambda _msgs, p=payload, **_kw: types.SimpleNamespace(
                text=json.dumps(p, ensure_ascii=False)))
        with _turn(message_id=f"msg-{idx}"):
            res = group_ingest.apply_group_delivery(
                provider,
                user_message=f"群友原话：{quote}，顺便聊聊别的",
                assistant_response="好的知道了",
                turn_id=f"turn-{idx}",
                delivered_at=now.timestamp() + idx,
                delivery_ref=f"out-{idx}",
            )
            assert res["ok"] is True and len(res["events"]) == 1
            group_event_ids.append(res["events"][0])

    for idx, (eid, summary) in enumerate(zip(group_event_ids, expected_summaries)):
        row = _row(provider, eid)
        assert f"周六去公园散步第{idx}次" in row["raw"]
        assert row["summary"] == summary
        assert row["summary_source"] == "distilled"
        assert row["distilled_at"] is None
        assert row["vec"] == _pack_vector(_embed(summary, data_dir=home / "data"))
        if collection is not None:
            indexed = collection.get(ids=[f"evt-{eid}"], include=["documents", "embeddings"])
            assert indexed["ids"] == [f"evt-{eid}"]
            assert indexed["documents"] == [summary]
            assert indexed["embeddings"][0].tolist() == pytest.approx(_embed(summary, data_dir=home / "data"))

    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    day_end = (now.replace(hour=0, minute=0, second=0, microsecond=0)
               + __import__("datetime").timedelta(days=1)).isoformat()
    assert provider.event_maintenance.count_distilled_between(day_start, day_end) == 0

    # Private daily distill still keeps its full cap=2 and does not re-summarize group receipts.
    memories = home / "memories"
    memories.mkdir(parents=True, exist_ok=True)
    for name in ("USER.md", "MEMORY.md"):
        (memories / name).write_text("", encoding="utf-8")

    class _DistillLlm:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            return types.SimpleNamespace(text=json.dumps({
                "summary": f"私人摘要-{self.calls}",
                "duty": "memory", "source_kind": "unknown",
                "long_term": True, "covered_by": "", "significance": "重要经历",
            }, ensure_ascii=False))

    provider._llm = _DistillLlm()
    distill_res = provider.lifecycle.distill(now)
    assert [item["id"] for item in distill_res["items"]] == [p1, p2]
    assert not any(item["id"] in group_event_ids for item in distill_res["items"])
    provider.shutdown()
    if chromadb is None:
        return

    # A receipt worker initialized with hook_worker=True and semantic_backend='chroma'
    # lazily opens Chroma before refreshing the summary vector.
    hook_data = home / "hook-data"
    hook_worker = SanaeMemProvider({
        "data_dir": str(hook_data), "semantic_backend": "chroma", "shadow": False,
    })
    hook_worker.initialize("session-h", hermes_home=str(home), platform="discord",
                           hook_worker=True)
    stack.callback(hook_worker.shutdown)
    assert hook_worker.storage.current_collection() is None
    assert not (hook_data / "chroma").exists()

    hook_payload = {
        "events": [{"text": "钩子线程摘要结果", "quote": "周六钩子线程去海边"}],
        "impression": None, "new_commitments": [], "commitment_updates": [],
    }
    hook_worker._llm = types.SimpleNamespace(
        complete=lambda _msgs, **_kw: types.SimpleNamespace(
            text=json.dumps(hook_payload, ensure_ascii=False)))
    with _turn(thread_id="reply-hook", parent_chat_id="guild-a", synthetic=True,
               message_id="msg-hook"):
        hook_res = group_ingest.apply_group_delivery(
            hook_worker,
            user_message="群友：周六钩子线程去海边",
            assistant_response="收到",
            turn_id="turn-hook",
            delivered_at=now.timestamp() + 10,
            delivery_ref="out-hook",
        )
        assert hook_res["ok"] is True and len(hook_res["events"]) == 1
        hook_eid = hook_res["events"][0]
    hook_row = _row(hook_worker, hook_eid)
    assert hook_row["owner_key"] == OWNER_A
    assert hook_row["summary"] == "钩子线程摘要结果" and hook_row["distilled_at"] is None
    assert hook_worker.storage.current_collection() is not None
    assert (hook_data / "chroma" / "chroma.sqlite3").is_file()
    hook_client = chromadb.PersistentClient(path=str(hook_data / "chroma"))
    stack.callback(hook_client.close)
    indexed = hook_client.get_collection(_COLLECTION).get(
        ids=[f"evt-{hook_eid}"], include=["documents", "embeddings"],
    )
    assert indexed["ids"] == [f"evt-{hook_eid}"]
    assert indexed["documents"] == ["钩子线程摘要结果"]
    assert indexed["embeddings"][0].tolist() == pytest.approx(_embed("钩子线程摘要结果", data_dir=hook_data))
    hook_worker.shutdown()
    assert hook_worker.storage.current_collection() is None


def test_count_distilled_between_handles_legacy_json_and_nightly_group_distill(
        tmp_path, monkeypatch, stack):
    from datetime import timedelta

    home = _use_home(monkeypatch, _home(tmp_path, overrides={GROUP_A: True}))
    stack.enter_context(host_bridge.turn_envelope_scope(platform="cli", session_id="fixture"))
    provider = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
        "distill_daily_cap": 6,
    })
    provider.initialize("session-a", hermes_home=str(home), platform="discord")

    now = datetime.now().astimezone()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    day_end = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).isoformat()

    legacy_contexts = [
        "",
        "{}",
        json.dumps({"kind": "maintainer", "platform": "qq", "visibility": "private"}),
        json.dumps({"platform": "slack", "processing": None}),
        "{broken-json",
    ]
    for idx, ctx in enumerate(legacy_contexts):
        eid = provider.event_store.insert_event_row(
            f"legacy-{idx}",
            (now.isoformat(), f"legacy distilled row {idx}", "kw", 0.9, 0.6, "生活",
             None, now.isoformat(), None, f"s-{idx}"),
            owner_key=PRIVATE_OWNER,
            source_context=ctx,
        )
        assert isinstance(eid, int)
        provider._conn.execute(
            "UPDATE events SET summary='摘要', summary_source='distilled', distilled_at=? WHERE id=?",
            (now.isoformat(), eid),
        )

    with _turn():
        receipt_eid = provider.event_store.insert_event_row(
            "receipt-with-stamp",
            (now.isoformat(), "receipt row", "kw", 0.6, 0.6, "生活",
             None, now.isoformat(), None, "s-receipt"),
            owner_key=OWNER_A,
            source_context=json.dumps({
                "kind": "group_member", "platform": "discord", "visibility": "group",
                "processing": "group_receipt_ingest",
            }),
        )
        assert isinstance(receipt_eid, int)
    provider._conn.execute(
        "UPDATE events SET summary='即时摘要', summary_source='distilled', distilled_at=? WHERE id=?",
        (now.isoformat(), receipt_eid),
    )
    provider._conn.commit()

    # All 5 legacy/valid-JSON private rows count; the group receipt row does not.
    assert provider.event_maintenance.count_distilled_between(day_start, day_end) == 5

    # One nightly group distillation consumes the 6th and final slot of distill_daily_cap=6.
    with _turn():
        g_nightly = _add(provider, "ALPHA-ZEBRA-MOUNTAIN-BIKE-EVENT", importance=0.9)
        g_extra = _add(provider, "OMEGA-DOLPHIN-PIANO-CONCERT-EVENT", importance=0.8)
    assert g_extra != g_nightly
    provider.shutdown()

    background = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
        "distill_daily_cap": 6,
    })
    background.initialize("cron-1", hermes_home=str(home), platform="cron", agent_context="cron")
    llm = _CaptureLlm()
    background._llm = llm
    try:
        res = background.lifecycle.distill(now)
    finally:
        background.shutdown()

    assert [item["id"] for item in res["items"]] == [g_nightly]
    check = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
    })
    check.initialize("check-1", hermes_home=str(home), platform="cron", agent_context="cron")
    try:
        assert check.event_maintenance.count_distilled_between(day_start, day_end) == 6
        assert _row(check, g_nightly)["distilled_at"]
        assert _row(check, g_extra)["distilled_at"] is None
    finally:
        check.shutdown()
