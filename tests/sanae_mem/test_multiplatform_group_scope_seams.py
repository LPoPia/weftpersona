from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("VIVEMIND_HERMES_SOURCE"):
    HERMES_ROOT = Path(os.environ["VIVEMIND_HERMES_SOURCE"]).expanduser().resolve()
else:
    _hermes_spec = importlib.util.find_spec("hermes_constants")
    if _hermes_spec is not None and _hermes_spec.origin:
        HERMES_ROOT = Path(_hermes_spec.origin).resolve().parent
    else:
        HERMES_ROOT = next(
            (path for path in (
                REPO_ROOT.parent / "hermes-minimal-host",
                REPO_ROOT.parent / "hermes-minimal-host-20261112",
            ) if path.is_dir()),
            Path(tempfile.gettempdir()) / "vivemind-hermes-6093627",
        )
if not HERMES_ROOT.is_dir():
    pytest.skip("pinned Hermes source is unavailable", allow_module_level=True)
sys.path.insert(0, str(HERMES_ROOT))

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem import SanaeMemProvider  # noqa: E402
from sanae_mem.runtime import group_memory  # noqa: E402
from sanae_mem.infra import host_bridge  # noqa: E402

GROUP_A = "default/discord/dom-a/guild-a"
OWNER_A = group_memory.GROUP_OWNER_PREFIX + GROUP_A
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


@contextmanager
def _turn(*, chat_id: str = "guild-a", platform: str = "discord", scope_id: str = "dom-a",
          chat_type: str = "group", thread_id: str = "", parent_chat_id: str = "",
          thread_visibility: str = "", synthetic: bool = False, sender: str = "account-a",
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
    })
    provider.initialize("session-a", hermes_home=str(home), platform="discord",
                        agent_context=context)
    return provider


def _values(text: str) -> tuple:
    return ("2026-05-01", text, "kw", 0.8, 0.6, "topic", None, "2026-05-01", None, "session-a")


@pytest.fixture
def stack():
    with ExitStack() as entered:
        yield entered


def test_closed_group_cannot_reach_private_or_other_group_rows(tmp_path, monkeypatch, stack):
    home = _home(tmp_path, overrides={GROUP_A: True})
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    provider = _provider(home, stack)

    private_id = provider.event_store.insert_event_row(
        "private-fingerprint", _values("private diary line"), owner_key=PRIVATE_OWNER)
    assert isinstance(private_id, int)
    with _turn():
        group_id = provider.add_group_event("group line of guild a")
        assert isinstance(group_id, int)

    # Turning the switch off never turns the scene into a private reader.
    (home / "config.yaml").write_text(_config(False, {GROUP_A: False}), encoding="utf-8")
    with _turn():
        assert provider._group_scope_grant() is None
        assert provider.event_store.get_event_row_by_id(private_id) is None
        assert provider.event_store.get_event_row_by_id(group_id) is None
        assert provider.event_store.event_ids_existing([private_id, group_id]) == []
        assert provider.event_store.get_grounded_occurrence(private_id) is None
        # An explicit private owner is not a read grant inside a managed group scene either.
        assert provider.event_store.event_ids_existing([private_id], owner_key=PRIVATE_OWNER) == []
        assert provider.query_events(
            "2100-01-01T00:00:00+08:00", read_context=True, owner_key=PRIVATE_OWNER) == []

    # A re-enabled group reads its own row only, never the private one.
    (home / "config.yaml").write_text(_config(False, {GROUP_A: True}), encoding="utf-8")
    with _turn():
        assert provider.event_store.get_event_row_by_id(group_id) is not None
        assert provider.event_store.get_event_row_by_id(private_id) is None
        assert provider.event_store.event_ids_existing([private_id], owner_key=PRIVATE_OWNER) == []


def test_narrow_thread_cannot_push_its_text_into_the_shared_parent_row(
        tmp_path, monkeypatch, stack):
    home = _home(tmp_path, overrides={GROUP_A: True})
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    provider = _provider(home, stack)

    with _turn():
        shared_id = provider.add_group_event("shared parent row")
    occurrences_before = provider._conn.execute(
        "SELECT count(*) FROM event_occurrences WHERE event_id=?", (shared_id,)).fetchone()[0]
    with _turn(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
               parent_chat_id="guild-a", thread_visibility="narrower",
               session_key="session-t1", message_id="m-t1"):
        narrow_id = provider.add_group_event("narrow row")
        # Reads reach the parent group; writes of the body do not.
        assert provider.event_store.get_event_row_by_id(shared_id) is not None
        assert provider.event_store.append_event_evidence(shared_id, "NARROW-SECRET", "2026-05-02") == 0
        assert provider.event_store.record_event_occurrence(
            shared_id, "NARROW-SECRET", "2026-05-02", preserve_previous=True) is False
        assert provider.event_store.append_event_evidence(narrow_id, "narrow follow-up", "2026-05-02") == 1

    row = provider._conn.execute(
        "SELECT raw FROM events WHERE id=?", (shared_id,)).fetchone()
    assert "NARROW-SECRET" not in row["raw"]
    assert provider._conn.execute(
        "SELECT count(*) FROM event_occurrences WHERE event_id=?", (shared_id,)).fetchone()[0] \
        == occurrences_before


def test_enabled_group_owners_follow_the_global_default_and_inheritance(
        tmp_path, monkeypatch, stack):
    home = _home(tmp_path, defaults=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    provider = _provider(home, stack)

    from types import SimpleNamespace

    with _turn():
        plain_id = provider.add_group_event("plain group under a true default")
        assert plain_id is not None
    with _turn(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
               parent_chat_id="guild-a", thread_visibility="narrower",
               session_key="session-t1", message_id="m-t1"):
        thread_id = provider.add_group_event("narrow thread row under a true default")
        assert thread_id is not None

    class _GroupSummaryLlm:
        def complete(self, messages, **kwargs):
            return SimpleNamespace(text='{"summary": "群内事件摘要"}')

    background = _provider(home, stack, context="cron")
    background.attach_runtime_llm(_GroupSummaryLlm())

    (home / "config.yaml").write_text(_config(defaults=False), encoding="utf-8")
    assert background.lifecycle.distill()["candidates"] == 0

    # No explicit per-group true exists anywhere; the global default is what enables these rows.
    (home / "config.yaml").write_text(_config(defaults=True), encoding="utf-8")
    distilled = background.lifecycle.distill()
    assert distilled["candidates"] == 2
    assert {int(item["id"]) for item in distilled["items"]} == {plain_id, thread_id}


def test_group_impression_rows_stay_per_group_and_member(tmp_path, monkeypatch, stack):
    home = _home(tmp_path, overrides={GROUP_A: True})
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    provider = _provider(home, stack)

    with _turn(sender="account-a"):
        target_a = provider._group_impression_target()
        assert target_a is not None and target_a[2].startswith(OWNER_A + "#")
        written = provider.inner_wiring.tool_impression(
            operation="write", text="他在这群里说话很稳。", session_id="", turn_id="t-1", synthetic=False)
        assert written["text"]
    with _turn(sender="account-b", session_key="session-b"):
        target_b = provider._group_impression_target()
        assert target_b[2] != target_a[2]
        assert provider.inner_wiring.tool_impression(
            operation="read", text=None, session_id="", turn_id="t-2", synthetic=False)["text"] == ""
    with _turn(chat_type="dm", chat_id="dm-1"):
        assert provider._group_impression_target() is None


def test_group_tool_surface_is_the_same_in_schema_gate_and_prompt_policy(
        tmp_path, monkeypatch, stack):
    from types import SimpleNamespace

    from sanae_mem.governance.cognitive_context import ViveMindCognitivePromptPolicy

    home = _home(tmp_path, overrides={GROUP_A: True})
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME_OVERRIDE", str(home))
    provider = _provider(home, stack)
    expected = {"sanae_mem_search"}

    def envelope() -> SimpleNamespace:
        return SimpleNamespace(
            platform="discord", source="discord", chat_type="group", chat_id="guild-a",
            sender_id="account-a", sender_id_alt="", scope_id="dom-a",
            session_key="session-a", session_id="session-a", is_cron=False,
            restricted_audience="", is_delegated_child=False, is_messaging_surface=True,
            session_context_engaged=True, thread_id="", profile="",
        )

    with _turn():
        schemas = {item["name"] for item in provider.get_tool_schemas()}
        assert schemas == expected
        policy = ViveMindCognitivePromptPolicy().resolve_turn_policy(envelope())
        assert set(policy.allowed_tools) == expected
        # The gate accepts search only and refuses in-turn write tools.
        assert json.loads(provider.handle_tool_call(
            "sanae_mem_add", {"content": "group write"}))["ok"] is False
        assert json.loads(provider.handle_tool_call(
            "sanae_impression", {"operation": "write", "text": "x"}))["ok"] is False
        assert json.loads(provider.handle_tool_call(
            "sanae_mem_forget", {"ids": [1]}))["ok"] is False

    (home / "config.yaml").write_text(_config(False), encoding="utf-8")
    with _turn():
        names = {item["name"] for item in provider.get_tool_schemas()}
        assert names == {"sanae_mem_search"}
        assert json.loads(provider.handle_tool_call(
            "sanae_mem_add", {"content": "closed group write"}))["ok"] is False
