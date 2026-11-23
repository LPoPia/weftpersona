"""Task 02 acceptance: one maintainer person across bound platform accounts, schema v4.

The turn facts come from the host's real session binding (``gateway.session_context``) and the
memory writes/reads go through the real provider and its SQLite CRUD; no final ``ChatScope`` is
built by hand here.

    PYTHONPATH=<vivemind>/plugins/sanae_mem:<vivemind>/plugins:<hermes> \
      python -m pytest tests/test_multiplatform_account_binding.py
"""
from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

pytest.importorskip("hermes_host_contracts")
pytest.importorskip("gateway.session_context")

_PLUGINS = Path(__file__).resolve().parents[1] / "plugins"
if str(_PLUGINS) not in sys.path:
    sys.path.insert(0, str(_PLUGINS))

from gateway.session_context import clear_session_vars, set_session_vars  # noqa: E402

from sanae_mem.governance import account_binding  # noqa: E402
from sanae_mem.storage import database, schema  # noqa: E402
from sanae_mem import plugin_entry as entry  # noqa: E402
from sanae_mem.governance.source_context import current_chat_scope, resolve_chat_identity  # noqa: E402
from sanae_mem.infra.textutil import _owner_key  # noqa: E402

QQ = {"platform": "qqbot", "user_id": "qq-openid-1"}
ANCHOR = _owner_key("qqbot", user_id="qq-openid-1")
SLACK_LEGACY = _owner_key("slack", user_id="U_MAINT", scope_id="T_WORK")
DISCORD_LEGACY = _owner_key("discord", user_id="4242")
CONFIG = {
    "memory_identity": dict(QQ),
    "maintainer_account_bindings": [
        {"platform": "slack", "user_id": "U_MAINT", "scope_id": "T_WORK"},
        {"platform": "discord", "user_id": "4242"},
    ],
}


@pytest.fixture
def bind():
    tokens: list = []

    def _bind(**facts) -> None:
        tokens.extend(set_session_vars(**facts))

    yield _bind
    if tokens:
        clear_session_vars(tokens)


def _slack(user: str = "U_MAINT", scope: str = "T_WORK", chat_type: str = "dm",
           is_bot=None) -> dict:
    facts = dict(platform="slack", chat_type=chat_type, chat_id="D_CHAN", user_id=user,
                 scope_id=scope, session_key=f"slack:{scope}:{chat_type}:D_CHAN",
                 message_id="m1")
    if is_bot is not None:
        facts["is_bot"] = is_bot
    return facts


def _identity():
    return resolve_chat_identity(current_chat_scope(), CONFIG)


# ── person / private owner / addressing ──────────────────────────────────────────

def test_bound_slack_dm_reads_and_writes_the_qq_anchor(bind):
    bind(**_slack())
    scope = current_chat_scope()
    identity = _identity()

    assert scope.visibility == "private"
    assert identity.maintainer is True
    assert identity.person_key == account_binding.person_key(ANCHOR)
    assert identity.private_owner_key == ANCHOR, "new memory lands on the historical owner"
    assert SLACK_LEGACY in identity.private_read_owner_keys
    assert ANCHOR in identity.private_read_owner_keys
    # The transport address of this turn is untouched: replies still go to the Slack DM.
    assert (scope.platform, scope.chat_id) == ("slack", "D_CHAN")


def test_bound_discord_dm_shares_the_same_person(bind):
    bind(platform="discord", chat_type="dm", chat_id="55", user_id="4242",
         session_key="discord:dm:55:4242", message_id="m2")
    identity = _identity()

    assert identity.maintainer is True
    assert identity.private_owner_key == ANCHOR
    assert DISCORD_LEGACY in identity.private_read_owner_keys


def test_same_account_in_a_group_only_yields_the_person(bind):
    bind(**_slack(chat_type="group"))
    scope = current_chat_scope()
    identity = _identity()

    assert scope.visibility == "public"
    assert identity.person_key == account_binding.person_key(ANCHOR)
    assert (identity.private_owner_key, identity.private_read_owner_keys) == ("", ())


def test_qq_same_actor_in_another_app_does_not_get_the_old_anchor(bind):
    config = {
        "memory_identity": {"platform": "qqbot", "user_id": "qq-openid-1",
                            "scope_id": "102000001"},
        "maintainer_account_bindings": [
            {"platform": "qqbot", "user_id": "qq-openid-1", "scope_id": "102000001"},
        ],
    }
    bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
         scope_id="102000001", session_key="qq:app-a", message_id="q1")
    own_app = resolve_chat_identity(current_chat_scope(), config)
    assert own_app.maintainer is True and own_app.private_owner_key == ANCHOR

    clear_session_vars([])
    bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
         scope_id="999000002", session_key="qq:app-b", message_id="q2")
    other_app = resolve_chat_identity(current_chat_scope(), config)
    assert other_app.maintainer is False
    assert other_app.private_owner_key != ANCHOR


def test_qq_private_chat_without_any_binding_config_keeps_its_old_anchor(tmp_path, bind):
    """An install that never declared a binding keeps working, app fact or not, and keeps its rows."""
    config = {"memory_identity": {"platform": "qqbot", "user_id": "qq-openid-1"},
              "semantic_backend": "sqlite", "shadow": False,
              "data_dir": str(tmp_path / "data")}
    bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
         scope_id="102000001", session_key="qqbot:dm:qq-dm", message_id="q1")
    provider = entry.SanaeMemProvider(dict(config))
    provider.initialize("qqbot:dm:qq-dm", hermes_home=str(tmp_path), platform="qqbot",
                        agent_context="primary")
    assert provider._owner_key == ANCHOR
    old_id = provider.event_store.insert_event_row(
        "fp-qq-old", ("2026-01-01T00:00:00", "old qq line", "", 0.5, 0.6, "", None,
                      "2026-01-01T00:00:00", None, "qq"),
        owner_key=ANCHOR)

    # The same person's next private line reads that row and writes the same owner, even when the
    # host states no app scope at all.
    bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
         session_key="qqbot:dm:qq-dm-no-app", message_id="q2")
    provider.on_turn_start(1, "")
    assert provider._owner_key == ANCHOR, "the original anchor survives a scope-less host"
    assert provider.event_store.get_event_row_by_id(old_id) is not None
    new_id = provider.event_store.insert_event_row(
        "fp-qq-new", ("2026-02-01T00:00:00", "new qq line", "", 0.5, 0.6, "", None,
                      "2026-02-01T00:00:00", None, "qq"),
        owner_key=provider._owner_key)
    assert dict(provider._conn.execute(
        "SELECT id, owner_key FROM events").fetchall()) == {old_id: ANCHOR, new_id: ANCHOR}


def test_both_explicit_account_spellings_bind_and_unsupported_platforms_are_ignored():
    keys = account_binding.configured_bindings({"maintainer_account_bindings": [
        {"platform": "qqbot", "user_id": "c2c-openid", "user_id_alt": "group-openid",
         "scope_id": "app-a"},
        {"platform": "telegram", "user_id": "not-in-first-batch"},
    ]})
    assert keys == [
        ("default", "qqbot", "c2c-openid", "app-a"),
        ("default", "qqbot", "group-openid", "app-a"),
    ]


def test_confirmed_qq_row_for_another_owner_blocks_the_legacy_anchor(bind, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {**CONFIG, "data_dir": str(data_dir), "maintainer_account_bindings": [
        *CONFIG["maintainer_account_bindings"],
        {"platform": "qqbot", "user_id": "qq-openid-1", "scope_id": "102000001"},
    ]}
    with closing(database.open_database(data_dir / "sanae.db")) as conn:
        account_binding.confirm_bindings(conn, config)
        conn.execute("UPDATE maintainer_account_bindings SET person_owner_key='someone-else' "
                     "WHERE platform='qqbot'")
        conn.commit()
    bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
         scope_id="102000001", session_key="qq:app-a", message_id="q1")
    identity = resolve_chat_identity(current_chat_scope(), config)
    assert identity.maintainer is False and identity.private_owner_key != ANCHOR


def test_profile_is_part_of_the_binding_identity(bind):
    config = {
        "memory_identity": dict(QQ),
        "maintainer_account_bindings": [
            {"profile": "work", "platform": "discord", "user_id": "4242"},
        ],
    }
    bind(platform="discord", profile="work", chat_type="dm", chat_id="55", user_id="4242",
         session_key="discord:work", message_id="w1")
    work = resolve_chat_identity(current_chat_scope(), config)
    assert work.maintainer is True and work.private_owner_key == ANCHOR

    clear_session_vars([])
    bind(platform="discord", profile="default", chat_type="dm", chat_id="55", user_id="4242",
         session_key="discord:default", message_id="d1")
    default = resolve_chat_identity(current_chat_scope(), config)
    assert default.maintainer is False and default.private_owner_key != ANCHOR


def test_same_shaped_id_in_another_workspace_is_not_the_maintainer(bind):
    bind(**_slack(scope="T_OTHER"))
    identity = _identity()

    assert identity.maintainer is False
    assert identity.private_owner_key not in {"", ANCHOR}
    assert ANCHOR not in identity.private_read_owner_keys
    assert identity.person_key != account_binding.person_key(ANCHOR)


def test_unbound_dm_and_bot_dm_get_no_private_history(bind):
    bind(**_slack(user="U_STRANGER"))
    stranger = _identity()
    assert stranger.private_owner_key == _owner_key(
        "slack", user_id="U_STRANGER", scope_id="T_WORK")
    assert stranger.maintainer is False

    clear_session_vars([])
    bind(**_slack(is_bot=True))
    bot = _identity()
    assert (bot.person_key, bot.private_owner_key) == ("", "")


def test_unknown_scene_and_empty_config_never_grant_private_ownership(bind):
    bind(platform="slack", chat_type="unknown", chat_id="G_OLD", user_id="U_MAINT",
         scope_id="T_WORK", session_key="slack:T_WORK:old:G_OLD", message_id="m3")
    unknown = resolve_chat_identity(current_chat_scope(), CONFIG)
    assert (unknown.person_key, unknown.private_owner_key,
            unknown.private_read_owner_keys) == ("", "", ())

    bind(**_slack())
    no_anchor = resolve_chat_identity(current_chat_scope(), {})
    assert no_anchor.maintainer is False
    assert no_anchor.private_owner_key == SLACK_LEGACY


def test_current_source_record_keeps_the_real_account_scope_profile_and_time(bind):
    from sanae_mem.governance.source_context import current_source_record, parse_source

    bind(platform="slack", profile="work", chat_type="dm", chat_id="D_CHAN",
         user_id="U_MAINT", scope_id="T_WORK", session_key="slack:T_WORK:dm:D_CHAN",
         message_id="m-source", occurred_at="2026-01-02T03:04:05+00:00")
    record = current_source_record(maintainer=True)
    assert record is not None
    assert (record["author_id"], record["platform"], record["ref"]) == (
        "U_MAINT", "slack", "m-source")
    assert (record["scope_id"], record["profile"], record["occurred_at"]) == (
        "T_WORK", "work", "2026-01-02T03:04:05+00:00")
    # Rows written before these appended fields decode with unknown values, unchanged.
    old = parse_source('{"kind":"maintainer","author_id":"U_OLD","platform":"slack"}')
    assert old["author_id"] == "U_OLD"
    assert (old["scope_id"], old["profile"], old["occurred_at"]) == ("", "", "")


def test_guard_paths_read_the_same_confirmed_mapping_without_a_provider_connection(bind, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {**CONFIG, "data_dir": str(data_dir)}
    with closing(database.open_database(data_dir / "sanae.db")) as conn:
        account_binding.confirm_bindings(conn, config)
        conn.execute("UPDATE maintainer_account_bindings SET person_owner_key='someone-else' "
                     "WHERE platform='slack'")
        conn.commit()

    bind(**_slack())
    identity = resolve_chat_identity(current_chat_scope(), config)  # no connection argument
    assert identity.maintainer is False
    assert ANCHOR not in identity.private_read_owner_keys


def test_guard_path_reads_the_home_database_read_only(tmp_path, monkeypatch, bind):
    """Without a provider connection the same profile's Home/sanae-mem file is the mapping."""
    home = tmp_path / "home"
    data_dir = home / "sanae-mem"
    data_dir.mkdir(parents=True)
    config = {**CONFIG, "data_dir": str(data_dir), "maintainer_account_bindings": [
        *CONFIG["maintainer_account_bindings"],
        {"platform": "qqbot", "user_id": "qq-openid-1", "scope_id": "102000001"},
    ]}
    with closing(database.open_database(data_dir / "sanae.db")) as conn:
        account_binding.confirm_bindings(conn, config)
        conn.execute("UPDATE maintainer_account_bindings SET person_owner_key='someone-else' "
                     "WHERE platform='qqbot'")
        conn.commit()
        monkeypatch.setattr("sanae_mem.infra.host_bridge.get_hermes_home", lambda: home)
        bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
             scope_id="102000001", session_key="qq:app-a", message_id="q1")
        # The config here carries no data_dir: the path is the profile's Home default.
        identity = resolve_chat_identity(current_chat_scope(), {
            "memory_identity": dict(QQ),
            "maintainer_account_bindings": config["maintainer_account_bindings"],
        })
    assert identity.maintainer is False and identity.private_owner_key != ANCHOR


def _conflicted_db(tmp_path, config, where="1=1", *, profile=None):
    """The same profile's temporary ``sanae.db`` with one row owned by somebody else."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {**config, "data_dir": str(data_dir)}
    with closing(database.open_database(data_dir / "sanae.db")) as conn:
        account_binding.confirm_bindings(conn, config, profile=profile)
        conn.execute("UPDATE maintainer_account_bindings SET person_owner_key='someone-else' "
                     f"WHERE {where}")
        conn.commit()
    return config


def test_no_app_qq_host_never_falls_back_to_the_raw_account_of_the_anchor(tmp_path, bind):
    """A blocked account whose raw key equals the anchor gets no private owner and no history."""
    config = _conflicted_db(tmp_path, {
        "memory_identity": dict(QQ), "semantic_backend": "sqlite", "shadow": False,
        "maintainer_account_bindings": [
            {"platform": "qqbot", "user_id": "qq-openid-1", "scope_id": "102000001"}],
    })
    bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
         session_key="qqbot:dm:qq-dm", message_id="q1")  # old host states no app at all
    scope = current_chat_scope()
    assert scope.visibility == "private"

    guard = resolve_chat_identity(scope, config)  # the same profile's file, read-only
    assert (guard.maintainer, guard.private_owner_key, guard.private_read_owner_keys) == (
        False, "", ())

    provider = entry.SanaeMemProvider({**config, "maintainer_account_bindings": []})
    provider.initialize("qqbot:dm:qq-dm", hermes_home=str(tmp_path), platform="qqbot",
                        agent_context="primary")
    assert (provider._owner_key, provider._read_owner_keys) == ("", ())
    assert provider.resolve_private_delivery_route().owner_key == ""
    names = {tool.get("name") or tool.get("function", {}).get("name")
             for tool in provider.get_tool_schemas()}
    assert "sanae_mem_add" not in names


def test_no_app_qq_host_without_an_app_fact_uses_no_binding(tmp_path, bind):
    """Declared app, or a same-owner app binding, both mean an app-less turn is that account nowhere."""
    declared = {**CONFIG, "data_dir": str(tmp_path / "data"), "maintainer_account_bindings": [
        *CONFIG["maintainer_account_bindings"],
        {"platform": "qqbot", "user_id": "qq-openid-1", "scope_id": "102000001"}]}
    bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
         session_key="qqbot:dm:qq-dm", message_id="q1")
    bound = resolve_chat_identity(current_chat_scope(), declared)
    assert (bound.maintainer, bound.private_owner_key, bound.private_read_owner_keys) == (
        False, "", ())

    clear_session_vars([])
    fixed_app = {**CONFIG, "data_dir": str(tmp_path / "data"),
                 "memory_identity": {**QQ, "scope_id": "102000001"}}
    bind(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
         session_key="qqbot:dm:qq-dm-no-app", message_id="q2")
    operator_app = resolve_chat_identity(current_chat_scope(), fixed_app)
    assert (operator_app.maintainer, operator_app.private_owner_key) == (False, "")


def test_blocked_discord_account_gets_the_person_but_no_private_owner(tmp_path, bind):
    """Discord's global snowflake can equal the anchor; a conflict still yields no private read."""
    config = _conflicted_db(tmp_path, {
        "memory_identity": {"platform": "discord", "user_id": "4242"},
        "maintainer_account_bindings": [{"platform": "discord", "user_id": "4242"}],
    })
    bind(platform="discord", chat_type="dm", chat_id="55", user_id="4242",
         session_key="discord:dm:55:4242", message_id="d1")
    blocked = resolve_chat_identity(current_chat_scope(), config)
    assert (blocked.maintainer, blocked.private_owner_key, blocked.private_read_owner_keys) == (
        False, "", ())

    clear_session_vars([])
    bind(platform="discord", chat_type="group", chat_id="77", user_id="4242",
         session_key="discord:group:77", message_id="d2")
    group = resolve_chat_identity(current_chat_scope(), config)
    assert group.maintainer is False and group.private_owner_key == ""
    assert group.person_key == account_binding.person_key(DISCORD_LEGACY)


def test_another_profile_never_lends_its_explicitly_bound_owner(tmp_path, bind):
    """The fixed memory_identity hint stays inside its own profile's mapping."""
    other_owner = "someone-else"
    config = _conflicted_db(tmp_path, {
        "memory_identity": dict(QQ),
        "maintainer_account_bindings": [
            {"profile": "work", "platform": "qqbot", "user_id": "qq-openid-1",
             "scope_id": "102000001"}],
    }, where="profile='work'", profile="work")
    with closing(sqlite3.connect(str(Path(config["data_dir"]) / "sanae.db"))) as rows:
        assert rows.execute("SELECT person_owner_key FROM maintainer_account_bindings"
                            ).fetchone()[0] == other_owner

    bind(platform="qqbot", profile="work", chat_type="dm", chat_id="qq-dm",
         user_id="qq-openid-1", scope_id="102000001", session_key="qq:work", message_id="q1")
    work = resolve_chat_identity(current_chat_scope(), config)
    assert work.private_owner_key not in {ANCHOR, other_owner}
    assert other_owner not in work.private_read_owner_keys

    clear_session_vars([])
    bind(platform="qqbot", profile="default", chat_type="dm", chat_id="qq-dm",
         user_id="qq-openid-1", scope_id="102000001", session_key="qq:default", message_id="q2")
    default = resolve_chat_identity(current_chat_scope(), config)
    assert other_owner not in default.private_read_owner_keys
    # This profile declares nothing, so its own old anchor keeps working exactly as before.
    assert default.private_owner_key == ANCHOR


def test_private_surface_classification_uses_the_same_resolver(bind):
    from sanae_mem.governance import cognitive_context as cognitive

    bind(**_slack())
    assert cognitive.classify_turn_surface(None, CONFIG) == "private_maintainer"

    clear_session_vars([])
    bind(**_slack(chat_type="group"))
    assert cognitive.classify_turn_surface(None, CONFIG) == "restricted_public"

    clear_session_vars([])
    bind(**_slack(user="U_STRANGER"))
    assert cognitive.classify_turn_surface(None, CONFIG) == "private_external"


# ── schema v4 ───────────────────────────────────────────────────────────────────

def _v3_database(path: Path) -> None:
    with closing(database.open_database(path)) as conn:
        conn.execute("INSERT INTO events (ts,raw,created_at,owner_key,source_context) "
                     "VALUES ('2026-01-01','old qq line','2026-01-01',?,?)",
                     (ANCHOR, '{"kind":"maintainer","platform":"qqbot"}'))
        conn.execute("INSERT INTO video_watches (bvid,owner_key,watched_at,created_at,updated_at) "
                     "VALUES ('BV1','" + SLACK_LEGACY + "','2026-01-01','2026-01-01','2026-01-01')")
        conn.execute("INSERT INTO meme_records (title,created_at,updated_at,owner_key) "
                     "VALUES ('old meme','2026-01-01','2026-01-01',?)", (ANCHOR,))
        conn.execute("INSERT INTO impression_revisions"
                     "(profile_key,owner_key,revision,text,speaker,evidence,turn_id,"
                     "updated_at,recorded_at) VALUES ('p',?,1,'old impression','u','e','t',1.0,1.0)",
                     (ANCHOR,))
        conn.execute("INSERT INTO commitments (owner_key,text,created_at,updated_at) "
                     "VALUES (?,'old promise','2026-01-01','2026-01-01')", (DISCORD_LEGACY,))
        conn.execute("DROP TABLE maintainer_account_bindings")
        conn.execute("PRAGMA user_version=3")
        conn.commit()


def test_v3_copy_upgrades_to_v4_without_touching_existing_rows(tmp_path):
    path = tmp_path / "sanae.db"
    _v3_database(path)

    with closing(database.open_database(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.CURRENT_SCHEMA_VERSION
        assert conn.execute("SELECT raw, owner_key, source_context FROM events").fetchone()[:2] == (
            "old qq line", ANCHOR)
        assert conn.execute("SELECT owner_key FROM video_watches").fetchone()[0] == SLACK_LEGACY
        assert conn.execute("SELECT title FROM meme_records").fetchone()[0] == "old meme"
        assert conn.execute("SELECT text FROM impression_revisions").fetchone()[0] == "old impression"
        assert conn.execute("SELECT text FROM commitments").fetchone()[0] == "old promise"
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='maintainer_account_bindings'"
                            ).fetchone() is not None

        backup = path.with_name(f"{path.name}.bak.v3.sqlite")
        assert backup.is_file()
        with closing(sqlite3.connect(str(backup))) as copy:
            assert copy.execute("SELECT raw FROM events").fetchone()[0] == "old qq line"
            assert copy.execute("PRAGMA user_version").fetchone()[0] == 3

    # Reopening a v4 database is a no-op, not another upgrade.
    with closing(database.open_database(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.CURRENT_SCHEMA_VERSION
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_v3_failure_after_binding_table_rolls_back_and_restart_finishes_v4(tmp_path, monkeypatch):
    path = tmp_path / "sanae.db"
    _v3_database(path)
    from sanae_mem.storage import legacy_upgrade

    real = legacy_upgrade._execute_schema_script
    calls = []

    def fail_after_create(conn, script):
        calls.append(script)
        real(conn, script)
        if "maintainer_account_bindings" in script:
            raise RuntimeError("injected after binding table creation")

    monkeypatch.setattr(legacy_upgrade, "_execute_schema_script", fail_after_create)
    with pytest.raises(RuntimeError, match="injected after binding table creation"):
        database.open_database(path)
    assert calls, "the failure point was reached"

    with closing(sqlite3.connect(str(path))) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        assert conn.execute("SELECT raw FROM events").fetchone()[0] == "old qq line"
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master "
                            "WHERE name='maintainer_account_bindings'").fetchone()[0] == 0
    backup = path.with_name(f"{path.name}.bak.v3.sqlite")
    assert backup.is_file()

    monkeypatch.setattr(legacy_upgrade, "_execute_schema_script", real)
    with closing(database.open_database(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.CURRENT_SCHEMA_VERSION
        assert conn.execute("SELECT raw FROM events").fetchone()[0] == "old qq line"
        assert conn.execute("SELECT COUNT(*) FROM maintainer_account_bindings").fetchone()[0] == 0


def test_bindings_are_idempotent_and_a_conflict_rolls_back(tmp_path):
    path = tmp_path / "sanae.db"
    with closing(database.open_database(path)) as conn:
        first = account_binding.confirm_bindings(conn, CONFIG)
        assert first["confirmed"] == 2
        again = account_binding.confirm_bindings(conn, CONFIG)
        assert (again["confirmed"], again["existing"]) == (0, 2)

        conn.execute("UPDATE maintainer_account_bindings SET person_owner_key='someone-else' "
                     "WHERE platform='slack'")
        conn.commit()
        conn.execute("DELETE FROM maintainer_account_bindings WHERE platform='discord'")
        conn.commit()
        with pytest.raises(account_binding.AccountBindingConflict):
            account_binding.confirm_bindings(conn, CONFIG)
        rows = dict(conn.execute(
            "SELECT platform, person_owner_key FROM maintainer_account_bindings").fetchall())
        assert rows == {"slack": "someone-else"}, "the failed batch left nothing behind"
        assert conn.in_transaction is False


def test_empty_binding_list_is_a_deterministic_no_op(tmp_path):
    path = tmp_path / "sanae.db"
    with closing(database.open_database(path)) as conn:
        result = account_binding.confirm_bindings(conn, {"memory_identity": dict(QQ)})
        assert result == {"anchor": ANCHOR, "confirmed": 0, "existing": 0, "ignored": 0}
        assert conn.execute("SELECT COUNT(*) FROM maintainer_account_bindings").fetchone()[0] == 0


# ── provider read/write across platforms ────────────────────────────────────────

def _provider(tmp_path, bind, **facts):
    bind(**facts)
    provider = entry.SanaeMemProvider({
        "data_dir": str(tmp_path / "data"), "semantic_backend": "sqlite", "shadow": False,
        **CONFIG,
    })
    provider.initialize(facts["session_key"], hermes_home=str(tmp_path),
                        platform=facts["platform"], agent_context="primary")
    return provider


def test_new_turn_writes_to_the_anchor_and_old_history_stays_readable(tmp_path, bind):
    qq_facts = dict(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
                    scope_id="102000001", session_key="qqbot:dm:qq-dm", message_id="q1")
    provider = _provider(tmp_path, bind, **qq_facts)
    assert provider._owner_key == ANCHOR

    # A row written by the old Slack account before the binding existed.
    legacy_id = provider.event_store.insert_event_row(
        "fp-legacy", ("2026-01-01T00:00:00", "old slack line", "", 0.5, 0.6, "", None,
                      "2026-01-01T00:00:00", None, "slack-session"),
        owner_key=SLACK_LEGACY, source_context='{"kind":"maintainer","platform":"slack"}')

    bind(**{**_slack(), "session_key": "slack:T_WORK:dm:D_CHAN"})
    provider._platform = "slack"  # a live cached provider belongs to this platform route
    provider.on_turn_start(1, "")
    assert provider._owner_key == ANCHOR
    new_id = provider.event_store.insert_event_row(
        "fp-new", ("2026-02-01T00:00:00", "new slack line", "", 0.5, 0.6, "", None,
                   "2026-02-01T00:00:00", None, "slack-session"),
        owner_key=provider._owner_key, source_context='{"kind":"maintainer","platform":"slack"}')

    rows = {row["id"]: row for row in provider._conn.execute(
        "SELECT id, owner_key, source_context FROM events").fetchall()}
    assert rows[new_id]["owner_key"] == ANCHOR, "the new turn joined the historical owner"
    assert rows[legacy_id]["owner_key"] == SLACK_LEGACY, "the old row kept its own owner"
    assert "slack" in rows[legacy_id]["source_context"]

    assert provider.event_store.get_event_row_by_id(legacy_id) is not None, "old history is still readable"
    assert provider.event_store.get_event_row_by_id(new_id) is not None

    # The same person's older impression and promise stay readable under their own owner keys.
    from sanae_mem.cognitive.impression import read_impression

    provider._conn.execute(
        "INSERT INTO impression_revisions"
        "(profile_key,owner_key,revision,text,speaker,evidence,turn_id,updated_at,recorded_at)"
        " VALUES (?,?,1,'old slack impression','u','e','t',1.0,1.0)",
        (provider._daily_profile_key, SLACK_LEGACY))
    provider._conn.execute(
        "INSERT INTO commitments (owner_key,text,created_at,updated_at)"
        " VALUES (?,'old slack promise','2026-01-01','2026-01-01')", (SLACK_LEGACY,))
    provider._conn.commit()

    impression = read_impression(
        (provider._data_dir, provider._daily_profile_key, provider._owner_key),
        owners=provider._read_owner_keys)
    assert impression is not None and impression.text == "old slack impression"
    texts = {row["text"] for row in provider.list_commitments(provider._owner_key)}
    assert "old slack promise" in texts

    # A different person's private route never borrows either row.
    bind(**_slack(user="U_STRANGER"))
    provider.on_turn_start(1, "")
    assert provider._owner_key not in {"", ANCHOR}
    assert provider.event_store.get_event_row_by_id(legacy_id) is None
    assert provider.event_store.get_event_row_by_id(new_id) is None


def test_bound_private_chat_reads_old_video_and_meme_history_but_a_group_cannot(tmp_path, bind):
    qq_facts = dict(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
                    scope_id="102000001", session_key="qqbot:dm:qq-dm", message_id="q1")
    provider = _provider(tmp_path, bind, **qq_facts)
    provider._conn.execute(
        "INSERT INTO video_watches (bvid,platform,title,summary,read_scope,owner_key,"
        "watched_at,created_at,updated_at) VALUES ('BV1xx411c7mD','bilibili','old talk',"
        "'bound video history','{\"kind\":\"owner\"}',?,'2026-01-01','2026-01-01','2026-01-01')",
        (SLACK_LEGACY,))
    provider._conn.execute(
        "INSERT INTO meme_records (title,meaning,created_at,updated_at,owner_key,tier,"
        "visibility_scope,evidence_status) VALUES ('old meme','bound meme history',"
        "'2026-01-01','2026-01-01',?,'common','private','confirmed')", (SLACK_LEGACY,))
    provider._conn.commit()

    bind(**{**_slack(), "session_key": "slack:T_WORK:dm:D_CHAN"})
    provider._platform = "slack"
    provider.on_turn_start(1, "")
    assert provider.video_memory.list_video_watches(owner_key=provider._owner_key)[0]["bvid"] == "BV1xx411c7mD"
    assert provider.video_memory.get_video_watch_detail("BV1xx411c7mD", provider._owner_key)["title"] == "old talk"
    assert "bound meme history" in provider.meme_context_for_turn("old meme")

    bind(**_slack(chat_type="group"))
    provider.on_turn_start(1, "")
    assert provider._owner_key == ""
    assert provider.video_memory.list_video_watches(owner_key=provider._owner_key) == []
    assert provider.meme_context_for_turn("old meme") == ""


def test_group_recognition_opens_no_private_read_through_the_provider(tmp_path, bind):
    qq_facts = dict(platform="qqbot", chat_type="dm", chat_id="qq-dm", user_id="qq-openid-1",
                    session_key="qqbot:dm:qq-dm", message_id="q1")
    provider = _provider(tmp_path, bind, **qq_facts)
    provider.event_store.insert_event_row(
        "fp-private", ("2026-01-01T00:00:00", "private line", "", 0.5, 0.6, "", None,
                       "2026-01-01T00:00:00", None, "qq"),
        owner_key=ANCHOR)

    bind(**_slack(chat_type="group"))
    provider.on_turn_start(1, "")
    identity = _identity()
    assert identity.person_key == account_binding.person_key(ANCHOR)
    assert provider._owner_key == ""
    assert provider.resolve_private_delivery_route().owner_key == ""
    names = {tool.get("name") or tool.get("function", {}).get("name")
             for tool in provider.get_tool_schemas()}
    assert "sanae_mem_add" not in names
