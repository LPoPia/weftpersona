from __future__ import annotations

import json
import sys
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


def _config(overrides: dict, defaults: bool = False) -> str:
    lines = ["plugins:", "  sanae_mem:", "    group_context:", "      defaults:",
             f"        long_term_memory: {str(defaults).lower()}",
             f"        short_term_shared: {str(defaults).lower()}", "      overrides:"]
    for key, value in overrides.items():
        lines.append(f'        "{key}":')
        lines.append(f"          long_term_memory: {str(value).lower()}")
    return "\n".join(lines) + "\n"


def _write_home(home: Path, overrides: dict, defaults: bool = False) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(_config(overrides, defaults), encoding="utf-8")
    return home


@pytest.fixture
def stack():
    with ExitStack() as entered:
        yield entered


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


def _provider(home: Path, stack: ExitStack, *, data: str = "data", context: str = "primary"):
    stack.enter_context(host_bridge.turn_envelope_scope(
        platform="cli", session_id="fixture"))
    provider = SanaeMemProvider({
        "data_dir": str(home / data), "semantic_backend": "sqlite", "shadow": False,
    })
    provider.initialize("session-a", hermes_home=str(home), platform="discord",
                        agent_context=context)
    return provider


def _tool(provider, name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(name, args))


def _ids(payload) -> list:
    assert payload["ok"], payload
    return [item["id"] for item in payload["result"]]


def _add(provider, text: str) -> int:
    event_id = provider.add_group_event(text, importance=0.8)
    assert isinstance(event_id, int)
    return event_id


def _search_ids(provider, query: str) -> list:
    return _ids(_tool(provider, "sanae_mem_search", {"query": query, "top_k": 10}))


def _row(provider, event_id: int):
    return provider._conn.execute(
        "SELECT owner_key, source_context, shared_view, public_excerpt FROM events WHERE id=?",
        (int(event_id),)).fetchone()


def test_grant_follows_current_scene_and_live_config(tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)

    with _turn():
        grant = provider._group_scope_grant()
        assert grant is not None
        assert grant.write_owner == OWNER_A
        assert grant.read_owners == (OWNER_A,)
        assert grant.source["chat_id"] == "guild-a"
        assert grant.source["visibility"] == "group"
        assert grant.person_key

    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b"):
        # Group B is not enabled: no owner, no read set, no source.
        assert provider._group_scope_grant() is None
        assert provider.add_group_event("group b blocked") is None
        with pytest.raises(PermissionError):
            provider.ingest.tool_add({"content": "group b blocked"})

    with _turn(chat_type="dm", chat_id="dm-1", scope_id="dom-a"):
        assert provider._group_scope_grant() is None

    with _turn(platform="telegram"):
        assert provider._group_scope_grant() is None

    # The switch is re-read from the live HOME config, not a copy taken at init.
    _write_home(home, {GROUP_A: False})
    with _turn():
        assert provider._group_scope_grant() is None
    _write_home(home, {GROUP_A: True})
    with _turn():
        assert provider._group_scope_grant() is not None


def test_search_and_read_stay_inside_the_current_group(tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch,
                     _write_home(tmp_path / "home", {GROUP_A: True, GROUP_B: True}))
    provider = _provider(home, stack)

    with _turn():
        event_a = _add(provider, "alpha marker from guild a")
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b"):
        event_b = _add(provider, "beta marker from guild b")

    for event_id, owner in ((event_a, OWNER_A), (event_b, OWNER_B)):
        row = _row(provider, event_id)
        assert row["owner_key"] == owner
        assert json.loads(row["source_context"])["visibility"] == "group"
        assert row["shared_view"] == "" and row["public_excerpt"] == ""

    with _turn():
        found = _search_ids(provider, "alpha")
        assert event_a in found and event_b not in found
        assert event_b not in _search_ids(provider, "alpha")
        assert provider.event_store.event_ids_existing([event_b]) == []
        assert provider.event_store.get_event_row_by_id(event_b) is None
        assert provider.event_store.event_source_context(event_b) is None

    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b"):
        found = _search_ids(provider, "beta")
        assert event_b in found and event_a not in found
        assert provider.event_store.event_ids_existing([event_a]) == []

    # A private scene reads neither group, and a public scene keeps the approved-only path.
    with _turn(chat_type="dm", chat_id="dm-1", scope_id="dom-a"):
        assert provider.event_store.event_ids_existing([event_a, event_b]) == []
        assert provider.event_store.get_event_row_by_id(event_a) is None
    with _turn(platform="telegram"):
        assert provider._group_scope_grant() is None
        assert provider.event_store.event_ids_existing([event_a, event_b]) == []


def test_narrow_thread_writes_exact_owner_and_does_not_leak_to_parent_or_sibling(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)

    with _turn():
        shared_id = _add(provider, "shared gamma in the group")
    with _turn(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
               parent_chat_id="guild-a", thread_visibility="narrower",
               session_key="session-t1", message_id="m-t1"):
        narrow_id = _add(provider, "delta only inside the narrow thread")
        grant = provider._group_scope_grant()
        assert grant.write_owner == OWNER_A + "/threads/thread-1"
        assert set(grant.read_owners) == {OWNER_A, OWNER_A + "/threads/thread-1"}
        found = _search_ids(provider, "delta")
        assert narrow_id in found
        assert shared_id in _search_ids(provider, "gamma")

    with _turn(chat_type="thread", chat_id="thread-2", thread_id="thread-2",
               parent_chat_id="guild-a", thread_visibility="unknown",
               session_key="session-t2", message_id="m-t2"):
        assert narrow_id not in _search_ids(provider, "delta")
        assert provider.event_store.get_event_row_by_id(narrow_id) is None

    with _turn():
        # The parent group sees the shared row only; the narrow thread row never flows back.
        assert shared_id in _search_ids(provider, "gamma")
        assert narrow_id not in _search_ids(provider, "delta")
        assert provider.event_store.event_ids_existing([narrow_id]) == []
        assert json.loads(_row(provider, narrow_id)["source_context"])["thread_visibility"] == \
            "narrower"


def test_inherited_and_synthetic_threads_write_the_shared_group_owner(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)

    with _turn(chat_type="thread", chat_id="thread-3", thread_id="thread-3",
               parent_chat_id="guild-a", thread_visibility="inherited",
               session_key="session-t3", message_id="m-t3"):
        inherited_id = _add(provider, "epsilon inherited audience")
        assert provider._group_scope_grant().write_owner == OWNER_A
    with _turn(chat_id="guild-a", thread_id="reply-9", parent_chat_id="guild-a",
               synthetic=True, message_id="m-r9"):
        synthetic_id = _add(provider, "zeta synthetic reply landing")
        assert provider._group_scope_grant().write_owner == OWNER_A

    with _turn():
        found = _search_ids(provider, "epsilon")
        assert inherited_id in found
        assert synthetic_id in _search_ids(provider, "zeta")
        assert json.loads(_row(provider, inherited_id)["source_context"])["thread_id"] == \
            "thread-3"


def test_forged_owner_id_and_maintenance_arguments_do_not_widen_the_grant(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch,
                     _write_home(tmp_path / "home", {GROUP_A: True, GROUP_B: True}))
    provider = _provider(home, stack)
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b"):
        event_b = _add(provider, "eta row of group b")

    with _turn():
        values = ("2026-05-01", "theta forged write", "forged", 0.9, 0.6, "topic", None,
                  "2026-05-01", None, "session-a")
        assert provider.event_store.insert_event_row("forged-fingerprint", values, owner_key=OWNER_B) is None
        assert provider.event_store.event_ids_existing([event_b], owner_key=OWNER_B) == []
        assert provider.query_events(
            "2100-01-01T00:00:00+08:00", read_context=True, owner_key=OWNER_B) == []
        assert provider.event_store.query_tier_window(
            "2100-01-01T00:00:00+08:00", "2000-01-01T00:00:00+08:00",
            read_context=True, owner_key=OWNER_B) == []
        assert provider.event_store.reinforce_event(event_b, 1.0, "2026-05-02T00:00:00+08:00") is None
        assert provider.event_store.append_event_evidence(event_b, "injected", "2026-05-02") == 0
        assert provider.event_store.get_grounded_occurrence(event_b) is None


def test_disabled_group_keeps_its_rows_and_stops_read_write_and_cleanup(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)
    with _turn():
        event_a = _add(provider, "iota retained group row")
    background = _provider(home, stack, context="cron")
    curve_now = datetime.fromisoformat("2100-01-01T00:00:00+08:00")

    _write_home(home, {GROUP_A: False})
    with _turn():
        assert provider._group_scope_grant() is None
        assert provider.add_group_event("kappa blocked write") is None
        assert _search_ids(provider, "iota") == []
        assert provider.event_store.event_ids_existing([event_a]) == []
        assert provider.event_store.get_event_row_by_id(event_a) is None
        # The background candidate sets no longer offer the row for cleanup.
        assert event_a not in {int(row["id"]) for row in provider.event_maintenance.list_short_candidates()}
        assert provider.event_maintenance.expire_to_review("2100-01-01T00:00:00+08:00") == 0
    assert background.event_maintenance.collapse_forgetting_curve(curve_now) == {"faded": 0, "dormant": 0, "forgotten": 0}
    assert provider._conn.execute(
        "SELECT count(*) FROM events WHERE id=?", (event_a,)).fetchone()[0] == 1
    assert provider._conn.execute(
        "SELECT expired_at FROM events WHERE id=?", (event_a,)).fetchone()[0] is None

    _write_home(home, {GROUP_A: True})
    with _turn():
        assert event_a in _search_ids(provider, "iota")
        assert provider.event_store.get_event_row_by_id(event_a) is not None
    assert background.event_maintenance.collapse_forgetting_curve(curve_now) == {"faded": 1, "dormant": 1, "forgotten": 0}


def test_background_context_maintains_only_the_groups_the_config_enables(
        tmp_path, monkeypatch, stack):
    from types import SimpleNamespace

    home = _use_home(monkeypatch,
                     _write_home(tmp_path / "home", {GROUP_A: True, GROUP_B: True}))
    provider = _provider(home, stack, data="foreground")
    with _turn():
        event_a = _add(provider, "lambda enabled group")
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b"):
        event_b = _add(provider, "mu disabled later group")

    _write_home(home, {GROUP_A: True, GROUP_B: False})
    with _turn(chat_id="guild-b", scope_id="dom-b", sender="account-b",
               session_key="session-b"):
        assert provider.add_group_event("mu disabled group") is None

    class _GroupSummaryLlm:
        def complete(self, messages, **kwargs):
            return SimpleNamespace(text='{"summary": "群内事件摘要"}')

    background = _provider(home, stack, data="foreground", context="cron")
    candidates = {int(row["id"]) for row in background.event_maintenance.list_short_candidates()}
    assert event_a in candidates and event_b not in candidates
    background.attach_runtime_llm(_GroupSummaryLlm())
    distilled = background.lifecycle.distill()
    assert distilled["candidates"] == 1
    assert [int(item["id"]) for item in distilled["items"]] == [event_a]
    row_b = background.sqlite.fetchone("SELECT distilled_at FROM events WHERE id=?", (event_b,))
    assert row_b is not None and row_b["distilled_at"] is None


def test_two_homes_in_one_process_and_a_foreign_home_grant_nothing(
        tmp_path, monkeypatch, stack):
    home_a = _write_home(tmp_path / "home-a", {GROUP_A: True})
    home_b = _write_home(tmp_path / "home-b", {GROUP_A: False})
    _use_home(monkeypatch, home_a)
    provider_a = _provider(home_a, stack)
    with _turn():
        assert provider_a._group_scope_grant() is not None
        event_a = _add(provider_a, "lambda foreign home group row")
        assert _search_ids(provider_a, "lambda") == [event_a]

    _use_home(monkeypatch, home_b)
    provider_b = _provider(home_b, stack, data="b")
    with _turn():
        assert provider_b._group_scope_grant() is None
        # provider_a was built for another HOME: its static config copy grants nothing.
        assert provider_a._group_scope_grant() is None
        assert _search_ids(provider_a, "lambda") == []
        assert provider_a.query_events("2100-01-01T00:00:00+08:00", owner_key=OWNER_A) == []
        assert provider_a.group_turn_context("lambda") == ""


def test_group_rows_never_become_shared_or_public_projections(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)
    with _turn():
        event_a = _add(provider, "nu group row that must stay group only")
        assert provider.event_store.set_shared_view(event_a, "SYNTHETIC-SHARED") is False
        assert provider.event_store.set_public_excerpt(event_a, "SYNTHETIC-PUBLIC") is False

        # Even a wrongly polluted projection row is invisible to the private/public readers.
        provider._conn.execute(
            "UPDATE events SET shared_view='POLLUTED', public_excerpt='POLLUTED' WHERE id=?",
            (event_a,))
        provider._conn.commit()
        now = "2100-01-01T00:00:00+08:00"
        assert event_a not in {int(row["id"]) for row in provider.event_store.query_shared_views(now)}
        assert event_a not in {int(row["id"]) for row in provider.event_store.query_public_projections(
            now, unscoped=True)}

    with _turn(platform="telegram"):
        assert _search_ids(provider, "nu") == []


def test_group_tool_surface_is_search_only_and_forbids_in_turn_writes(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)
    assert group_memory.GROUP_TOOL_NAMES == frozenset({"sanae_mem_search"})

    with _turn():
        schemas = provider.get_tool_schemas()
        assert [item["name"] for item in schemas] == ["sanae_mem_search"]
        assert "scheduled" not in schemas[0]["parameters"]["properties"]

        for tool_name, args in (
            ("sanae_mem_add", {"content": "should not write in turn"}),
            ("sanae_impression", {"operation": "read"}),
            ("sanae_impression", {"operation": "write", "text": "should not write"}),
        ):
            payload = _tool(provider, tool_name, args)
            assert payload["ok"] is False

        with pytest.raises(PermissionError):
            provider.ingest.tool_add({"content": "direct _tool_add blocked"})

    assert provider._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert provider._conn.execute("SELECT COUNT(*) FROM impression_revisions").fetchone()[0] == 0


@pytest.mark.parametrize("daily_scheduled_asks", [0, 2])
def test_group_search_ignores_scheduled_and_has_no_write_side_effects(
        tmp_path, monkeypatch, stack, daily_scheduled_asks):
    from datetime import timedelta

    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    stack.enter_context(host_bridge.turn_envelope_scope(platform="cli", session_id="fixture"))
    provider = SanaeMemProvider({
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
        "read_gate": "scheduled", "daily_scheduled_asks": daily_scheduled_asks,
    })
    provider.initialize("session-a", hermes_home=str(home), platform="discord")

    now = datetime.now().astimezone()
    yesterday = (now - timedelta(days=1)).date().isoformat()
    private_id = provider.event_store.insert_event_row(
        "private-asked-row",
        (now.isoformat(), "private scheduled marker", "marker", 0.9, 0.6, "topic",
         None, now.isoformat(), None, "session-p"),
        owner_key="person:private-a",
    )
    provider._conn.execute(
        "UPDATE events SET asked=1, asked_via='scheduled', asked_at=? WHERE id=?",
        (now.isoformat(), private_id),
    )
    provider.status_store.mutate(lambda st: {**st, "date": yesterday, "last_asked_reset_date": yesterday})
    provider._conn.commit()

    with _turn():
        g1 = _add(provider, "group scheduled marker high", )
        g2 = _add(provider, "group scheduled marker low")
        res = _tool(provider, "sanae_mem_search", {
            "query": "marker", "scheduled": True, "top_k": 10,
        })
        assert res["ok"] is True
        assert {item["id"] for item in res["result"]} == {g1, g2}

    for gid in (g1, g2):
        row = provider._conn.execute(
            "SELECT asked, asked_via, asked_at FROM events WHERE id=?", (gid,),
        ).fetchone()
        assert tuple(row) == (0, "", "")
    prow = provider._conn.execute(
        "SELECT asked, asked_via FROM events WHERE id=?", (private_id,),
    ).fetchone()
    assert tuple(prow) == (1, "scheduled")
    status = provider.status_store.read()
    assert int(status.get("daily_scheduled_asks", 0)) == 0
    assert status.get("last_asked_reset_date") == yesterday


def test_group_search_never_queries_private_shared_views_regardless_of_top_k(
        tmp_path, monkeypatch, stack):
    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    provider = _provider(home, stack)
    now = datetime.now().astimezone().isoformat()
    for idx in range(3):
        pid = provider.event_store.insert_event_row(
            f"private-shared-{idx}",
            (now, f"PRIVATE-ALPHA secret raw {idx}", "alpha", 0.9, 0.6, "topic",
             None, now, None, f"session-p-{idx}"),
            owner_key=f"person:private-{idx}",
        )
        assert isinstance(pid, int)
        assert provider.event_store.set_shared_view(pid, f"SHARED-ALPHA-VIEW-{idx}") is True

    with _turn():
        group_ids = {_add(provider, f"group alpha event {idx}") for idx in range(2)}
        for top_k in (1, 5, 10):
            payload = _tool(provider, "sanae_mem_search", {"query": "alpha", "top_k": top_k})
            assert payload["ok"] is True
            hits = payload["result"]
            assert hits and all(item["id"] in group_ids for item in hits)
            assert all(not item.get("shared") for item in hits)
            assert "SHARED-ALPHA-VIEW" not in json.dumps(hits, ensure_ascii=False)
        prefetched = provider.prefetch("alpha", session_id="session-a")
        assert "SHARED-ALPHA-VIEW" not in prefetched
        assert provider.event_store.query_shared_views("2100-01-01T00:00:00+08:00") == []


def test_private_reset_asked_and_marked_promoted_stay_within_private_ownership(
        tmp_path, monkeypatch, stack):
    from datetime import timedelta
    from sanae_mem.governance.account_binding import maintainer_anchor
    from sanae_mem.infra.textutil import _owner_key

    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    config = {
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
        "read_gate": "scheduled", "daily_scheduled_asks": 5,
        "memory_identity": {"platform": "discord", "user_id": "anchor-1"},
        "maintainer_account_bindings": [{
            "profile": "default", "platform": "slack", "user_id": "U_BOUND",
            "scope_id": "T_1",
        }],
    }
    stack.enter_context(host_bridge.turn_envelope_scope(platform="cli", session_id="fixture"))
    provider = SanaeMemProvider(config)
    provider.initialize("session-a", hermes_home=str(home), platform="slack")

    now = datetime.now().astimezone()
    future = (now + timedelta(days=7)).isoformat()
    anchor = maintainer_anchor(config)
    bound_old = _owner_key("slack", user_id="U_BOUND", scope_id="T_1")
    other_owner = _owner_key("discord", user_id="other-1")

    with _turn():
        group_id = _add(provider, "group row protected from private reset and promote")

    own_id = provider.event_store.insert_event_row(
        "own-row", (now.isoformat(), "own private row", "kw", 0.8, 0.6, "topic",
                    future, now.isoformat(), None, "s-own"), owner_key=anchor)
    bound_old_id = provider.event_store.insert_event_row(
        "bound-old-row", (now.isoformat(), "bound old private row", "kw", 0.8, 0.6, "topic",
                          future, now.isoformat(), None, "s-old"), owner_key=bound_old)
    other_id = provider.event_store.insert_event_row(
        "other-row", (now.isoformat(), "other user private row", "kw", 0.8, 0.6, "topic",
                      future, now.isoformat(), None, "s-other"), owner_key=other_owner)

    for eid in (group_id, own_id, bound_old_id, other_id):
        provider._conn.execute(
            "UPDATE events SET asked=1, asked_via='scheduled', asked_at=?, ttl_expire=? WHERE id=?",
            (now.isoformat(), future, eid),
        )
    provider._conn.commit()

    with _turn():
        assert provider.event_store.reset_asked_all() == 0
        assert provider.lifecycle.marked_promoted([group_id], "important")["updated"] == 0

    with _turn(platform="slack", chat_type="dm", chat_id="D_1", scope_id="T_1",
               sender="U_BOUND", session_key="slack:dm:D_1", message_id="m-dm"):
        provider.on_turn_start(1, "")
        promoted = _tool(provider, "sanae_mem_marked_promoted", {
            "ids": [group_id, other_id, bound_old_id, own_id], "to": "important",
        })
        assert promoted["ok"] is True
        assert promoted["result"]["updated"] == 2

        long_attempt = _tool(provider, "sanae_mem_marked_promoted", {
            "ids": [group_id, other_id], "to": "long",
        })
        assert long_attempt["ok"] is True
        assert long_attempt["result"]["updated"] == 0

        assert provider.event_store.reset_asked_all() == 2

    for eid, exp_tier, exp_asked in (
        (own_id, "important", 0),
        (bound_old_id, "important", 0),
        (group_id, "short", 1),
        (other_id, "short", 1),
    ):
        row = provider._conn.execute(
            "SELECT tier, asked, ttl_expire FROM events WHERE id=?", (eid,),
        ).fetchone()
        assert (row["tier"], row["asked"]) == (exp_tier, exp_asked)
        if exp_tier == "short":
            assert row["ttl_expire"] == future
        else:
            assert row["ttl_expire"] is None

    # Nightly cron compact resets remaining private asked marks without touching group rows.
    yesterday = (now - timedelta(days=1)).date().isoformat()
    provider.status_store.mutate(lambda st: {**st, "last_compactor_date": yesterday,
                                        "last_asked_reset_date": yesterday})
    cron = SanaeMemProvider(config)
    cron.initialize("cron-1", hermes_home=str(home), platform="cron", agent_context="cron")
    try:
        cron.lifecycle.compact(now)
    finally:
        cron.shutdown()
    assert provider._conn.execute("SELECT asked FROM events WHERE id=?", (other_id,)).fetchone()[0] == 0
    assert provider._conn.execute("SELECT asked FROM events WHERE id=?", (group_id,)).fetchone()[0] == 1
    provider.shutdown()


def test_multi_owner_daily_rollover_separates_same_day_repeat_missed_days_and_groups(
        tmp_path, monkeypatch, stack):
    from datetime import timedelta
    from sanae_mem.infra.textutil import _owner_key

    home = _use_home(monkeypatch, _write_home(tmp_path / "home", {GROUP_A: True}))
    config = {
        "data_dir": str(home / "data"), "semantic_backend": "sqlite", "shadow": False,
        "read_gate": "scheduled", "daily_scheduled_asks": 5,
    }
    stack.enter_context(host_bridge.turn_envelope_scope(platform="cli", session_id="fixture"))
    provider = SanaeMemProvider(config)
    provider.initialize("session-a", hermes_home=str(home), platform="discord")

    now = datetime.now().astimezone()
    today = now.date().isoformat()
    yesterday_dt = now - timedelta(days=1)
    three_days_ago_dt = now - timedelta(days=3)
    yesterday = yesterday_dt.date().isoformat()

    owner_a = _owner_key("discord", user_id="user-a")
    owner_b = _owner_key("discord", user_id="user-b")

    with _turn():
        group_id = _add(provider, "group asked marker")

    a_id = provider.event_store.insert_event_row(
        "a-yesterday", (now.isoformat(), "我想买一把静音键盘", "静音键盘", 0.9, 0.6, "topic",
                        None, now.isoformat(), None, "s-a"), owner_key=owner_a)
    b_yest_id = provider.event_store.insert_event_row(
        "b-yesterday", (now.isoformat(), "这周预约了牙医复查", "牙医复查", 0.9, 0.6, "topic",
                        None, now.isoformat(), None, "s-b1"), owner_key=owner_b)
    b_old_id = provider.event_store.insert_event_row(
        "b-missed", (now.isoformat(), "记得去图书馆还书", "图书馆还书", 0.5, 0.6, "topic",
                     None, now.isoformat(), None, "s-b2"), owner_key=owner_b)

    for eid, stamp in (
        (group_id, yesterday_dt.isoformat()),
        (a_id, yesterday_dt.isoformat()),
        (b_yest_id, yesterday_dt.isoformat()),
        (b_old_id, three_days_ago_dt.isoformat()),
    ):
        provider._conn.execute(
            "UPDATE events SET asked=1, asked_via='scheduled', asked_at=? WHERE id=?",
            (stamp, eid),
        )
    provider._conn.commit()
    provider.status_store.mutate(lambda st: {**st, "date": yesterday, "last_asked_reset_date": yesterday})

    # Owner A runs first today: resets only Owner A and stamps last_asked_reset_date=today.
    with _turn(platform="discord", chat_type="dm", chat_id="dm-a", scope_id="",
               sender="user-a", session_key="discord:dm:dm-a", message_id="m-a1"):
        provider.on_turn_start(1, "")
        first_a = _tool(provider, "sanae_mem_search", {
            "query": "静音键盘", "scheduled": True, "top_k": 5,
        })
        assert first_a["ok"] and [item["id"] for item in first_a["result"]] == [a_id]
        # Same-day repeat for Owner A must not re-reset today's ask.
        repeat_a = _tool(provider, "sanae_mem_search", {
            "query": "静音键盘", "scheduled": True, "top_k": 5,
        })
        assert repeat_a["ok"] and repeat_a["result"] == []

    assert provider.status_store.read()["last_asked_reset_date"] == today
    assert provider._conn.execute("SELECT asked FROM events WHERE id=?", (b_yest_id,)).fetchone()[0] == 1
    assert provider._conn.execute("SELECT asked FROM events WHERE id=?", (b_old_id,)).fetchone()[0] == 1
    assert provider._conn.execute("SELECT asked FROM events WHERE id=?", (group_id,)).fetchone()[0] == 1

    # Owner B runs later on the SAME day: must still reset yesterday & multi-day missed rows.
    with _turn(platform="discord", chat_type="dm", chat_id="dm-b", scope_id="",
               sender="user-b", session_key="discord:dm:dm-b", message_id="m-b1"):
        provider.on_turn_start(1, "")
        first_b = _tool(provider, "sanae_mem_search", {
            "query": "牙医复查", "scheduled": True, "top_k": 5,
        })
        assert first_b["ok"] and [item["id"] for item in first_b["result"]] == [b_yest_id]
        assert provider._conn.execute("SELECT asked FROM events WHERE id=?", (b_old_id,)).fetchone()[0] == 0
        # Same-day repeat for Owner B must not re-reset b_yest_id just asked today.
        repeat_b = _tool(provider, "sanae_mem_search", {
            "query": "牙医复查", "scheduled": True, "top_k": 5,
        })
        assert repeat_b["ok"] and repeat_b["result"] == []

    assert provider._conn.execute("SELECT asked FROM events WHERE id=?", (a_id,)).fetchone()[0] == 1
    assert provider._conn.execute("SELECT asked FROM events WHERE id=?", (group_id,)).fetchone()[0] == 1
    provider.shutdown()


def test_maintenance_rollover_requires_authority_and_preserves_same_day_asks(
        tmp_path, monkeypatch, stack):
    from datetime import timedelta
    from sanae_mem.infra.textutil import _owner_key

    home_a = _write_home(tmp_path / "home-a", {GROUP_A: True})
    home_b = _write_home(tmp_path / "home-b", {GROUP_A: False})
    _use_home(monkeypatch, home_a)
    config = {
        "data_dir": str(home_a / "data"), "semantic_backend": "sqlite", "shadow": False,
        "read_gate": "scheduled", "daily_scheduled_asks": 5,
    }
    stack.enter_context(host_bridge.turn_envelope_scope(platform="cli", session_id="fixture"))
    primary = SanaeMemProvider(config)
    primary.initialize("session-a", hermes_home=str(home_a), platform="discord")

    now = datetime.now().astimezone()
    today = now.date().isoformat()
    yesterday_dt = now - timedelta(days=1)
    yesterday = yesterday_dt.date().isoformat()
    owner_a = _owner_key("discord", user_id="user-a")
    owner_b = _owner_key("discord", user_id="user-b")

    with _turn():
        group_id = _add(primary, "group row protected in maintenance")
    today_id = primary.event_store.insert_event_row(
        "today-row", (now.isoformat(), "asked today", "kw", 0.8, 0.6, "topic",
                      None, now.isoformat(), None, "s-1"), owner_key=owner_a)
    missed_id = primary.event_store.insert_event_row(
        "missed-row", (now.isoformat(), "missed yesterday", "kw", 0.8, 0.6, "topic",
                       None, now.isoformat(), None, "s-2"), owner_key=owner_b)

    for eid, stamp in (
        (group_id, yesterday_dt.isoformat()),
        (today_id, now.isoformat()),
        (missed_id, yesterday_dt.isoformat()),
    ):
        primary._conn.execute(
            "UPDATE events SET asked=1, asked_via='scheduled', asked_at=? WHERE id=?",
            (stamp, eid),
        )
    primary._conn.commit()
    primary.status_store.mutate(lambda st: {**st, "date": yesterday, "last_asked_reset_date": yesterday})
    primary.shutdown()

    cron = SanaeMemProvider(config)
    cron.initialize("cron-1", hermes_home=str(home_a), platform="cron", agent_context="cron")
    subagent = SanaeMemProvider(config)
    subagent.initialize("sub-1", hermes_home=str(home_a), platform="cli", agent_context="subagent")
    try:
        # 1. Subagent has no maintenance write authority.
        subagent.read_gate.daily_rollover(today)
        assert subagent.event_store.reset_asked_all() == 0

        # 2. Cron inside public/group, private, or delegated-child scope has no authority.
        with _turn():
            cron.read_gate.daily_rollover(today)
            assert cron.event_store.reset_asked_all() == 0
        with host_bridge.turn_envelope_scope(platform="cron", session_id="cron-1",
                                            is_delegated_child=True):
            cron.read_gate.daily_rollover(today)
            assert cron.event_store.reset_asked_all() == 0
        with _turn(platform="discord", chat_type="dm", chat_id="dm-a", scope_id="",
                   sender="user-a", session_key="discord:dm:dm-a", message_id="m-priv"):
            cron.read_gate.daily_rollover(today)
            assert cron.event_store.reset_asked_all() == 0

        # 3. Cron belonging to home_a cannot mutate while active HOME is home_b.
        _use_home(monkeypatch, home_b)
        cron.read_gate.daily_rollover(today)
        assert cron.event_store.reset_asked_all() == 0

        _use_home(monkeypatch, home_a)
        assert cron.status_store.read().get("last_asked_reset_date") == yesterday
        for eid in (group_id, today_id, missed_id):
            assert cron._conn.execute("SELECT asked FROM events WHERE id=?", (eid,)).fetchone()[0] == 1

        # 4. Authorized cron on a same-day status preserves today_id and resets missed_id only.
        cron.status_store.mutate(lambda st: {**st, "date": today, "last_asked_reset_date": today})
        cron.read_gate.daily_rollover(today)
        assert cron._conn.execute("SELECT asked FROM events WHERE id=?", (today_id,)).fetchone()[0] == 1
        assert cron._conn.execute("SELECT asked FROM events WHERE id=?", (missed_id,)).fetchone()[0] == 0
        assert cron._conn.execute("SELECT asked FROM events WHERE id=?", (group_id,)).fetchone()[0] == 1

        # 5. Authorized cron on a cross-day rollover refreshes status and keeps group untouched.
        cron._conn.execute(
            "UPDATE events SET asked=1, asked_at=? WHERE id=?",
            (yesterday_dt.isoformat(), missed_id),
        )
        cron._conn.commit()
        cron.status_store.mutate(lambda st: {**st, "date": yesterday, "last_asked_reset_date": yesterday})
        cron.lifecycle.compact(now)
        assert cron.status_store.read()["last_asked_reset_date"] == today
        assert cron._conn.execute("SELECT asked FROM events WHERE id=?", (today_id,)).fetchone()[0] == 0
        assert cron._conn.execute("SELECT asked FROM events WHERE id=?", (missed_id,)).fetchone()[0] == 0
        assert cron._conn.execute("SELECT asked FROM events WHERE id=?", (group_id,)).fetchone()[0] == 1
    finally:
        subagent.shutdown()
        cron.shutdown()
