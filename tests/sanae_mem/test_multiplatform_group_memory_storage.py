from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import ExitStack, closing
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from sanae_mem.governance import account_binding
from sanae_mem.storage import database
from sanae_mem.cognitive import impression
from sanae_mem.storage import schema
from sanae_mem import SanaeMemProvider
from sanae_mem.infra.host_bridge import hermes_home_scope, turn_envelope_scope
from sanae_mem.governance.source_context import (
    chat_scope_for, current_source_record, group_source_record, parse_source, source_json,
)

PROVENANCE_TABLES = ("impression_revisions", "commitments")


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}


def _bindings(conn: sqlite3.Connection) -> list[tuple] | None:
    """The stored account bindings, or ``None`` when the file has no such table (pre-v4)."""
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='maintainer_account_bindings'"
    ).fetchone() is None:
        return None
    return [tuple(row) for row in conn.execute(
        "SELECT profile,platform,account_id,person_owner_key FROM maintainer_account_bindings")]


def _envelope(**overrides: object) -> SimpleNamespace:
    facts = dict(
        platform="discord", chat_type="group", chat_id="guild-a", sender_id="account-a",
        sender_id_alt="", session_key="session-a", message_id="message-a-1", scope_id="",
        parent_chat_id="", reply_to_message_id="", occurred_at="2026-05-01T10:00:00+08:00",
        inbound_coverage="full", is_bot=False, thread_visibility="unknown", profile="default",
        thread_id="", session_id="sid-a", source="", is_delegated_child=False,
        restricted_audience="",
    )
    facts.update(overrides)
    return SimpleNamespace(**facts)


def _drop_provenance_columns(path: Path, *, drop_bindings: bool = False) -> None:
    conn = sqlite3.connect(path)
    for table in PROVENANCE_TABLES:
        conn.execute(f"ALTER TABLE {table} DROP COLUMN source_context")
    if drop_bindings:
        conn.execute("DROP TABLE maintainer_account_bindings")
    conn.commit()
    conn.close()


def _seed_private_rows(path: Path, *, version: int, binding: bool) -> None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO impression_revisions"
        "(profile_key,owner_key,revision,text,speaker,evidence,turn_id,updated_at,recorded_at,"
        "baseline,recent) VALUES('profile','owner',1,'SYNTHETIC-IMPRESSION','user',"
        "'SYNTHETIC-QUOTE','turn-1',1.0,1.0,'SYNTHETIC-BASELINE','SYNTHETIC-RECENT')"
    )
    conn.execute(
        "INSERT INTO commitments"
        "(owner_key,text,masked_text,source_quote,turn_id,status,version,created_at,updated_at) "
        "VALUES('owner','SYNTHETIC-COMMITMENT','masked','SYNTHETIC-QUOTE','turn-1','pending',1,"
        "'2026-01-01','2026-01-01')"
    )
    conn.execute(
        "INSERT INTO events(ts,raw,keywords,created_at,owner_key,read_scope,source_context) "
        "VALUES('now','SYNTHETIC-PRIVATE-RAW','current','now','owner',"
        "'{\"kind\": \"owner\", \"owner\": \"owner\"}',"
        "'{\"kind\": \"external_user\", \"visibility\": \"private\"}')"
    )
    if binding:
        conn.execute(
            "INSERT INTO maintainer_account_bindings"
            "(profile,platform,account_id,scope_id,person_owner_key,created_at) "
            "VALUES('default','discord','account-a','','person:a','2026-01-01')"
        )
    conn.execute(f"PRAGMA user_version={version}")
    conn.commit()
    conn.close()


def _prepared_database(path: Path, *, version: int) -> None:
    """The real shape of ``version``: v3 has no binding table, neither version has v5 columns."""
    with closing(database.open_database(path)):
        pass
    _drop_provenance_columns(path, drop_bindings=version < 4)
    _seed_private_rows(path, version=version, binding=version >= 4)


def test_new_database_declares_current_version_and_provenance_columns(tmp_path):
    path = tmp_path / "sanae.db"
    with closing(database.open_database(path)) as conn:
        assert schema.CURRENT_SCHEMA_VERSION >= 6
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.CURRENT_SCHEMA_VERSION
        for table in PROVENANCE_TABLES:
            assert "source_context" in _columns(conn, table)
    assert not list(tmp_path.glob("*.bak.*.sqlite"))


@pytest.mark.parametrize("version", [3, 4])
def test_upgrade_from_v3_and_v4_appends_source_context_and_keeps_private_rows(tmp_path, version):
    path = tmp_path / "sanae.db"
    _prepared_database(path, version=version)

    # The seed is the version it claims to be before the upgrade runs: v3 never had the v4
    # binding table, and neither version had the two v5 provenance columns.
    with closing(sqlite3.connect(path)) as seeded:
        assert seeded.execute("PRAGMA user_version").fetchone()[0] == version
        for table in PROVENANCE_TABLES:
            assert "source_context" not in _columns(seeded, table)
        assert _bindings(seeded) == (
            None if version < 4 else [("default", "discord", "account-a", "person:a")])

    for _ in range(2):  # open twice: the second open is the restart and must not upgrade again
        with closing(database.open_database(path)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.CURRENT_SCHEMA_VERSION
            for table in PROVENANCE_TABLES:
                assert "source_context" in _columns(conn, table)
            row = conn.execute(
                "SELECT profile_key,owner_key,revision,text,speaker,evidence,turn_id,"
                "baseline,recent,source_context FROM impression_revisions"
            ).fetchone()
            assert tuple(row) == ("profile", "owner", 1, "SYNTHETIC-IMPRESSION", "user",
                                  "SYNTHETIC-QUOTE", "turn-1", "SYNTHETIC-BASELINE",
                                  "SYNTHETIC-RECENT", "")
            commitment = conn.execute(
                "SELECT owner_key,text,source_quote,turn_id,status,version,source_context "
                "FROM commitments"
            ).fetchone()
            assert tuple(commitment) == ("owner", "SYNTHETIC-COMMITMENT", "SYNTHETIC-QUOTE",
                                         "turn-1", "pending", 1, "")
            event = conn.execute(
                "SELECT raw,owner_key,read_scope,source_context FROM events"
            ).fetchone()
            assert tuple(event) == (
                "SYNTHETIC-PRIVATE-RAW", "owner",
                '{"kind": "owner", "owner": "owner"}',
                '{"kind": "external_user", "visibility": "private"}',
            )
            if version >= 4:
                # The confirmed row keeps its owner; the upgrade adds nothing to it.
                assert _bindings(conn) == [("default", "discord", "account-a", "person:a")]
            else:
                # v3 first meets the table here: created empty, and the confirm path can write it.
                assert _bindings(conn) == []
                conn.execute("SAVEPOINT binding_probe")
                conn.execute(
                    "INSERT INTO maintainer_account_bindings"
                    "(profile,platform,account_id,scope_id,person_owner_key,created_at) "
                    "VALUES('default','discord','account-a','','person:a','2026-01-01')"
                )
                assert account_binding.stored_bindings(conn) == {
                    ("default", "discord", "account-a", ""): "person:a"}
                conn.execute("ROLLBACK TO binding_probe")
                conn.execute("RELEASE binding_probe")

    # Neither upgrade leaves a table behind or misses one the current schema declares.
    with closing(database.open_database(tmp_path / "fresh.db")) as fresh:
        fresh_tables = _tables(fresh)
    with closing(sqlite3.connect(path)) as upgraded:
        assert _tables(upgraded) == fresh_tables

    backup = path.with_name(f"sanae.db.bak.v{version}.sqlite")
    with closing(sqlite3.connect(backup)) as saved:
        assert saved.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert saved.execute("PRAGMA user_version").fetchone()[0] == version
        for table in PROVENANCE_TABLES:
            assert "source_context" not in _columns(saved, table)
        assert _bindings(saved) == (
            None if version < 4 else [("default", "discord", "account-a", "person:a")])
        assert saved.execute("SELECT text FROM impression_revisions").fetchone()[0] == \
            "SYNTHETIC-IMPRESSION"
    assert not path.with_name(f"sanae.db.bak.v{schema.CURRENT_SCHEMA_VERSION}.sqlite").exists()


def test_failed_v5_upgrade_rolls_back_and_recovers_on_reopen(tmp_path):
    path = tmp_path / "sanae.db"
    _prepared_database(path, version=4)
    # A view named like the table makes the real ADD COLUMN fail after the first table was
    # already altered, so the rollback of that DDL is what this test observes.
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE commitments RENAME TO commitments_row")
    conn.execute("CREATE VIEW commitments AS SELECT * FROM commitments_row")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="upgrade from version 4 failed") as error:
        database.open_database(path)
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)

    with closing(sqlite3.connect(path)) as source:
        assert source.execute("PRAGMA user_version").fetchone()[0] == 4
        assert "source_context" not in _columns(source, "impression_revisions")
        assert "source_context" not in _columns(source, "commitments_row")
        assert source.execute("SELECT text FROM impression_revisions").fetchone()[0] == \
            "SYNTHETIC-IMPRESSION"
        assert source.execute("SELECT text FROM commitments_row").fetchone()[0] == \
            "SYNTHETIC-COMMITMENT"

    backup = path.with_name("sanae.db.bak.v4.sqlite")
    with closing(sqlite3.connect(backup)) as saved:
        assert saved.execute("PRAGMA user_version").fetchone()[0] == 4
        assert saved.execute("SELECT text FROM commitments").fetchone()[0] == \
            "SYNTHETIC-COMMITMENT"

    conn = sqlite3.connect(path)
    conn.execute("DROP VIEW commitments")
    conn.execute("ALTER TABLE commitments_row RENAME TO commitments")
    conn.commit()
    conn.close()

    with closing(database.open_database(path)) as retried:
        assert retried.execute("PRAGMA user_version").fetchone()[0] == schema.CURRENT_SCHEMA_VERSION
        for table in PROVENANCE_TABLES:
            assert "source_context" in _columns(retried, table)
        assert retried.execute("SELECT text FROM commitments").fetchone()[0] == \
            "SYNTHETIC-COMMITMENT"
        assert retried.execute(
            "SELECT person_owner_key FROM maintainer_account_bindings"
        ).fetchone()[0] == "person:a"


def test_group_source_keeps_chat_and_thread_locations_without_becoming_public():
    group_a = chat_scope_for(_envelope())
    group_b = chat_scope_for(_envelope(
        chat_id="guild-b", session_key="session-b", message_id="message-b-1",
        sender_id="account-b",
    ))
    narrow_thread = chat_scope_for(_envelope(
        chat_type="thread", chat_id="thread-1", thread_id="thread-1",
        parent_chat_id="guild-a", thread_visibility="narrower",
        session_key="session-thread", message_id="message-t-1",
    ))
    private = chat_scope_for(_envelope(
        chat_type="dm", chat_id="dm-1", session_key="session-dm",
        message_id="message-dm-1", thread_id="",
    ))

    record_a = group_source_record(group_a)
    record_b = group_source_record(group_b)
    record_thread = group_source_record(narrow_thread)

    assert record_a["visibility"] == "group"
    assert record_a["kind"] == "group_member"
    assert record_a["author_id"] == "account-a"
    assert record_a["ref"] == "message-a-1"
    assert record_a["chat_id"] == "guild-a"
    assert record_a["session_key"] == "session-a"
    assert record_a["thread_id"] == "" and record_a["paraphrase_ref"] == ""
    assert record_b["chat_id"] == "guild-b"
    assert record_thread["chat_id"] == "thread-1"
    assert record_thread["parent_chat_id"] == "guild-a"
    assert record_thread["thread_visibility"] == "narrower"
    assert record_a != record_b and record_a != record_thread

    # Nothing is dropped between the JSON column and the parsed record.
    stored_a = source_json(record_a)
    read_back = parse_source(stored_a)
    for key in ("chat_id", "thread_id", "parent_chat_id", "thread_visibility", "session_key",
                "paraphrase_ref", "visibility", "author_id", "ref"):
        assert read_back[key] == record_a[key]
    assert parse_source(source_json({"chat_id": "guild-a"}))["chat_id"] == "guild-a"

    relay = group_source_record(group_a, paraphrase_ref="message-b-9")
    assert relay["ref"] == "message-a-1" and relay["paraphrase_ref"] == "message-b-9"

    assert group_source_record(private) is None
    assert group_source_record(chat_scope_for(None)) is None
    assert current_source_record(maintainer=True, scope=private)["visibility"] == "private"
    assert current_source_record(maintainer=True, scope=private)["chat_id"] == ""
    assert current_source_record(maintainer=True, scope=group_a) is None


def test_group_provenance_round_trips_through_events_impressions_and_commitments(tmp_path):
    path = tmp_path / "data" / "sanae.db"
    envelopes = {
        "a": _envelope(scope_id="dom-a"),
        "b": _envelope(chat_id="guild-b", scope_id="dom-b", session_key="session-b",
                       message_id="message-b-1", sender_id="account-b"),
        "thread": _envelope(chat_type="thread", chat_id="thread-1", thread_id="thread-1",
                            scope_id="dom-a", parent_chat_id="guild-a", thread_visibility="narrower",
                            session_key="session-thread", message_id="message-t-1"),
    }
    owners = {
        "a": "group:default/discord/dom-a/guild-a",
        "b": "group:default/discord/dom-b/guild-b",
        "thread": "group:default/discord/dom-a/guild-a/threads/thread-1",
    }

    with ExitStack() as stack:
        stack.enter_context(hermes_home_scope(tmp_path))
        store = SanaeMemProvider({
            "data_dir": str(path.parent),
            "semantic_backend": "sqlite",
            "group_context": {"defaults": {"long_term_memory": True}},
        })
        stack.callback(store.shutdown)
        store.initialize("sid-a", hermes_home=str(tmp_path), platform="discord", hook_worker=True)
        conn = stack.enter_context(closing(sqlite3.connect(path)))
        conn.row_factory = sqlite3.Row
        for name, envelope in envelopes.items():
            with turn_envelope_scope(inherit_context=False, **vars(envelope)):
                store.event_store.insert_event_row(
                    f"fingerprint-{name}",
                    ("2026-05-01", f"SYNTHETIC-EVENT-{name}", "group", 0.6, 0.6, "群聊",
                     None, "2026-05-01", None, "session-" + name),
                    owner_key=owners[name],
                    source_context=source_json(group_source_record(chat_scope_for(envelope))),
                    read_scope=json.dumps({"kind": "owner", "owner": owners[name]}),
                )
        rows = conn.execute(
            "SELECT raw,owner_key,source_context FROM events ORDER BY id"
        ).fetchall()
        assert [json.loads(row["source_context"])["chat_id"] for row in rows] == \
            ["guild-a", "guild-b", "thread-1"]
        assert [row["owner_key"] for row in rows] == [
            owners["a"], owners["b"], owners["thread"]
        ]
        assert json.loads(rows[2]["source_context"])["thread_visibility"] == "narrower"
        assert conn.execute("SELECT count(*) FROM events WHERE raw LIKE '%EVENT-a%'").fetchone()[0] == 1

        for name, envelope in envelopes.items():
            impression.write_impression(
                (path.parent, "profile", owners[name]),
                f"SYNTHETIC-IMPRESSION-{name}",
                turn_id=f"turn-{name}",
                source_context=source_json(group_source_record(chat_scope_for(envelope))),
            )
        for name in ("a", "b", "thread"):
            read = impression.read_impression((path.parent, "profile", owners[name]))
            assert read is not None
            source = parse_source(read.source_context)
            assert source is not None
            assert read.text == f"SYNTHETIC-IMPRESSION-{name}"
            assert source["chat_id"] == {
                "a": "guild-a", "b": "guild-b", "thread": "thread-1",
            }[name]
            assert source["visibility"] == "group"
        # The private call keeps its empty provenance and its old read/write shape.
        private_write = impression.write_impression(
            (path.parent, "profile", "private-owner"), "SYNTHETIC-PRIVATE", turn_id="turn-p",
        )
        assert private_write.source_context == ""
        stored_private_impression = conn.execute(
            "SELECT source_context FROM impression_revisions WHERE owner_key='private-owner'"
        ).fetchone()
        assert stored_private_impression is not None and stored_private_impression[0] == ""

        commitment_ids = {}
        for name, envelope in envelopes.items():
            with turn_envelope_scope(inherit_context=False, **vars(envelope)):
                commitment_id = store.record_commitment(
                    owners[name], f"SYNTHETIC-PROMISE-{name}", turn_id=f"turn-{name}",
                    source_context=source_json(group_source_record(chat_scope_for(envelope))),
                )
                assert commitment_id is not None
                commitment_ids[name] = commitment_id
        for name, envelope in envelopes.items():
            with turn_envelope_scope(inherit_context=False, **vars(envelope)):
                listed = store.list_commitments(owners[name])
                assert len(listed) == 1
                source = parse_source(listed[0]["source_context"])
                assert source is not None
                assert source["chat_id"] == {
                    "a": "guild-a", "b": "guild-b", "thread": "thread-1",
                }[name]
        private = _envelope(chat_type="dm", chat_id="dm-1", thread_id="")
        with turn_envelope_scope(inherit_context=False, **vars(private)):
            assert store.record_commitment(
                "private-owner", "SYNTHETIC-PRIVATE-PROMISE", turn_id="turn-p",
            )
            assert store.list_commitments(owners["a"]) == []
            assert not store.commitments.resolve_commitment(commitment_ids["a"], "cancelled")
        stored_private_commitment = conn.execute(
            "SELECT source_context FROM commitments WHERE owner_key='private-owner'"
        ).fetchone()
        assert stored_private_commitment is not None and stored_private_commitment[0] == ""

        archive = tmp_path / "memory-archive" / "commitments-archived.jsonl"
        archive.mkdir(parents=True, exist_ok=True)
        with turn_envelope_scope(inherit_context=False, **vars(envelopes["a"])):
            assert store.resolve_commitment_with_reason(
                commitment_ids["a"], "fulfilled", expected_version=1,
            ) == (False, "archive_failed")
        assert tuple(conn.execute(
            "SELECT status, version FROM commitments WHERE id=?", (commitment_ids["a"],),
        ).fetchone()) == ("pending", 1)
        event_count = conn.execute("SELECT count(*) FROM events").fetchone()
        assert event_count is not None and event_count[0] == 3

        # Resolving keeps the original provenance and only moves status/version.
        archive.rmdir()
        with turn_envelope_scope(inherit_context=False, **vars(envelopes["a"])):
            assert store.commitments.resolve_commitment(commitment_ids["a"], "cancelled", evidence="cancelled")
        archived, = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
        assert (archived["id"], archived["status"], archived["version"]) == (
            commitment_ids["a"], "pending", 1,
        )
        resolved = conn.execute(
            "SELECT status,version,source_context FROM commitments WHERE id=?",
            (commitment_ids["a"],),
        ).fetchone()
        assert resolved is not None
        assert (resolved["status"], resolved["version"]) == ("cancelled", 2)
        assert json.loads(resolved["source_context"])["chat_id"] == "guild-a"
        with turn_envelope_scope(inherit_context=False, **vars(envelopes["a"])):
            assert store.list_commitments(owners["a"])[0]["status"] == "cancelled"
        with turn_envelope_scope(inherit_context=False, **vars(envelopes["b"])):
            source = parse_source(store.list_commitments(owners["b"])[0]["source_context"])
            assert source is not None and source["chat_id"] == "guild-b"
