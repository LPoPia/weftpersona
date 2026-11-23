"""SQLite persistence and value types for inner state revisions and shared view."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, fields
import json
from pathlib import Path
import sqlite3
from typing import Any

from ..infra.contracts import freeze_mapping
from .sqlite_facade import open_sqlite_connection

InnerStateTarget = tuple[Path, str]


@dataclass(frozen=True)
class InnerState:
    revision: int
    text: str
    recent_words: str
    recent_words_at: float | None
    variables: Mapping[str, float]
    felt_at: float
    updated_at: float
    processed_at: float | None
    source: str
    thought_source: str
    turn_id: str
    changed: bool
    owner_key: str = ""
    text_kind: str = "legacy"
    last_spark_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", freeze_mapping(self.variables))


@dataclass(frozen=True)
class InnerStateView:
    """One current common head plus the two independently-timed visible segments.

    ``head`` is always the newest revision of the single profile chain: its
    revision, slow variables, clocks and delivery watermark are the only common
    metadata. ``shared`` is the current safe common view and ``private`` is the
    latest revision authored by the requesting owner (or, for an internal read,
    the latest autonomous internal revision). Neither segment replaces the
    other; each keeps its own ``felt_at`` and TTL.
    """

    head: InnerState
    shared: InnerState | None = None
    private: InnerState | None = None

    @property
    def segment(self) -> InnerState | None:
        return self.private or self.shared

    @property
    def revision(self) -> int:
        return self.head.revision

    @property
    def variables(self) -> Mapping[str, float]:
        return self.head.variables

    @property
    def processed_at(self) -> float | None:
        return self.head.processed_at

    @property
    def updated_at(self) -> float:
        return self.head.updated_at

    @property
    def text(self) -> str:
        """Both visible raw segments, shared first; neither hides the other."""
        parts: list[str] = []
        if self.shared is not None and self.shared.text:
            parts.append(self.shared.text)
        if self.private is not None and self.private.text and self.private.text not in parts:
            parts.append(self.private.text)
        return "\n".join(parts)

    @property
    def recent_words(self) -> str:
        return self.private.recent_words if self.private is not None else ""

    @property
    def recent_words_at(self) -> float | None:
        return self.private.recent_words_at if self.private is not None else None

    @property
    def felt_at(self) -> float:
        return self.segment.felt_at if self.segment is not None else self.head.felt_at

    @property
    def source(self) -> str:
        return self.segment.source if self.segment is not None else self.head.source

    @property
    def thought_source(self) -> str:
        return self.segment.thought_source if self.segment is not None else self.head.thought_source

    @property
    def turn_id(self) -> str:
        return self.segment.turn_id if self.segment is not None else self.head.turn_id

    @property
    def changed(self) -> bool:
        return self.segment.changed if self.segment is not None else self.head.changed

    @property
    def owner_key(self) -> str:
        return self.private.owner_key if self.private is not None else ""

    @property
    def text_kind(self) -> str:
        return self.segment.text_kind if self.segment is not None else "shared"


def _state_to_dict(state: InnerState) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in fields(state):
        value = getattr(state, spec.name)
        out[spec.name] = dict(value) if isinstance(value, Mapping) else value
    return out


def _connect(database: Path, mode: str) -> sqlite3.Connection:
    conn_mode = "ro" if mode == "ro" else "rw"
    return open_sqlite_connection(database, mode=conn_mode)


def _visible_kind(visibility: str, owner_key: str) -> str:
    if str(visibility or "private") == "internal":
        return "internal"
    return "private" if owner_key else "legacy"


def _row_visibility(state: InnerState) -> str:
    return "internal" if state.text_kind == "internal" else "private"


def _state_from_row(row: sqlite3.Row | dict) -> InnerState:
    values = dict(row)
    if values["recent_words_at"] is None and values["recent_words"]:
        values["recent_words_at"] = values["felt_at"]
    values["variables"] = json.loads(values["variables"])
    values["changed"] = bool(values["changed"])
    values["last_spark_at"] = float(values.get("last_spark_at") or 0.0)
    owner = str(values.get("owner_key") or "")
    values["owner_key"] = owner
    values["text_kind"] = _visible_kind(values.pop("visibility", "private"), owner)
    return InnerState(**values)


def _states(conn: sqlite3.Connection, profile_key: str, limit: int, *,
            before: int | None = None, owner_key: str | None = None,
            visibility: str | None = None) -> list[InnerState]:
    sql = (
        "SELECT revision, text, recent_words, recent_words_at, variables, felt_at, updated_at, "
        "processed_at, source, thought_source, turn_id, changed, owner_key, visibility, last_spark_at "
        "FROM inner_state_revisions WHERE profile_key=?"
    )
    params: list = [profile_key]
    if owner_key is not None:
        sql += " AND owner_key=?"
        params.append(owner_key)
    if visibility is not None:
        sql += " AND visibility=?"
        params.append(visibility)
    if before is not None:
        sql += " AND revision<? AND changed=1"
        params.append(before)
    sql += " ORDER BY revision DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_state_from_row(row) for row in rows]


def _latest_state(conn: sqlite3.Connection, profile_key: str, *,
                  owner_key: str | None = None,
                  visibility: str | None = None) -> InnerState | None:
    rows = _states(conn, profile_key, 1, owner_key=owner_key, visibility=visibility)
    return rows[0] if rows else None


def _clock_state(conn: sqlite3.Connection, profile_key: str) -> InnerState | None:
    row = conn.execute(
        "SELECT revision, variables, felt_at, updated_at, processed_at, source, turn_id, last_spark_at "
        "FROM inner_state_revisions WHERE profile_key=? ORDER BY revision DESC LIMIT 1",
        (profile_key,),
    ).fetchone()
    if row is None:
        return None
    return InnerState(
        revision=int(row["revision"]), text="", recent_words="", recent_words_at=None,
        variables=json.loads(row["variables"]), felt_at=float(row["felt_at"]),
        updated_at=float(row["updated_at"]), processed_at=row["processed_at"],
        source=str(row["source"]), thought_source=str(row["source"]),
        turn_id=str(row["turn_id"] or ""), changed=False,
        owner_key="", text_kind="legacy", last_spark_at=float(row["last_spark_at"] or 0.0),
    )


def _shared_table_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='inner_state_shared'"
    ).fetchone() is not None


def _shared_state(conn: sqlite3.Connection, profile_key: str) -> InnerState | None:
    if not profile_key or not _shared_table_exists(conn):
        return None
    row = conn.execute(
        "SELECT text, variables, felt_at, updated_at, source_revision "
        "FROM inner_state_shared WHERE profile_key=?",
        (profile_key,),
    ).fetchone()
    if row is None or not str(row["text"] or "").strip():
        return None
    return InnerState(
        revision=int(row["source_revision"] or 0), text=str(row["text"] or ""),
        recent_words="", recent_words_at=None, variables=json.loads(row["variables"]),
        felt_at=float(row["felt_at"]), updated_at=float(row["updated_at"]),
        processed_at=None, source="flash", thought_source="flash", turn_id="",
        changed=True, owner_key="", text_kind="shared",
    )


def inspect_inner_state(
    database: Path, profile_key: str, *, recent: int, owner_key: str = "",
) -> dict:
    """Inspect one authorized scope without opening a writer.

    ``owner_key`` selects that interlocutor's private revisions; without it only
    the safe shared view is returned (never an unbound legacy row).
    """
    if recent < 0:
        raise ValueError("recent must be nonnegative")
    with closing(_connect(database, "ro")) as conn:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='inner_state_revisions'"
        ).fetchone() is None:
            return {"current": None, "recent": []}
        if not owner_key:
            shared = _shared_state(conn, profile_key)
            return {"current": _state_to_dict(shared) if shared is not None else None, "recent": []}
        states = _states(conn, profile_key, 1, owner_key=owner_key, visibility="private")
        if not states:
            return {"current": None, "recent": []}
        current = states[0]
        changes = _states(conn, profile_key, recent, before=current.revision,
                          owner_key=owner_key, visibility="private")
        return {"current": _state_to_dict(current), "recent": [_state_to_dict(state) for state in changes]}


def latest_state(database: Path, profile_key: str) -> InnerState | None:
    if not profile_key:
        return None
    with closing(_connect(database, "ro")) as conn:
        return _clock_state(conn, profile_key)


def owner_state(database: Path, profile_key: str, owner_key: str) -> InnerState | None:
    if not owner_key:
        return None
    with closing(_connect(database, "ro")) as conn:
        return _latest_state(conn, profile_key, owner_key=owner_key, visibility="private")


def read_state(
    database: Path, profile_key: str, *, owner_key: str = "",
    shared_only: bool = False, internal: bool = False,
) -> InnerStateView | None:
    if not profile_key:
        return None
    with closing(_connect(database, "ro")) as conn:
        head = _clock_state(conn, profile_key)
        if head is None:
            return None
        shared = _shared_state(conn, profile_key)
        private = None
        if internal:
            private = _latest_state(conn, profile_key, owner_key="", visibility="internal")
        elif not shared_only and owner_key:
            private = _latest_state(conn, profile_key, owner_key=owner_key, visibility="private")
        if shared is None and private is None:
            return None
        return InnerStateView(head=head, shared=shared, private=private)


def append_state(database: Path, target: InnerStateTarget, state: InnerState) -> bool:
    data_dir, profile_key = target
    if not state.owner_key and state.text_kind != "internal":
        return False
    with closing(_connect(database, "rw")) as conn, conn:
        cursor = conn.execute(
            """INSERT INTO inner_state_revisions
               (profile_key, revision, text, recent_words, recent_words_at, variables,
                felt_at, updated_at, processed_at, source, thought_source, turn_id, changed,
                owner_key, visibility, last_spark_at)
               SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
               WHERE COALESCE((SELECT MAX(revision) FROM inner_state_revisions
                               WHERE profile_key=?), 0)=?""",
            (profile_key, state.revision, state.text, state.recent_words, state.recent_words_at,
             json.dumps(dict(state.variables)), state.felt_at, state.updated_at, state.processed_at,
             state.source, state.thought_source, state.turn_id, state.changed, state.owner_key,
             _row_visibility(state), float(state.last_spark_at), profile_key, state.revision - 1),
        )
        return cursor.rowcount == 1


def replace_state_text(
    database: Path, target: InnerStateTarget, state: InnerState, *,
    owner_key: str, new_text: str,
) -> bool:
    data_dir, profile_key = target
    if not owner_key or state.owner_key != owner_key:
        return False
    with closing(_connect(database, "rw")) as conn, conn:
        cursor = conn.execute(
            "UPDATE inner_state_revisions SET text=? "
            "WHERE profile_key=? AND owner_key=? AND revision=? AND text=?",
            (new_text, profile_key, owner_key, state.revision, state.text),
        )
        return cursor.rowcount == 1


def read_shared_state(database: Path, profile_key: str) -> InnerState | None:
    if not profile_key:
        return None
    with closing(_connect(database, "ro")) as conn:
        return _shared_state(conn, profile_key)


def write_shared_state(
    database: Path, profile_key: str, *, text: str, variables: Mapping[str, float],
    now: float, source_revision: int,
) -> bool:
    with closing(_connect(database, "rw")) as conn, conn:
        cursor = conn.execute(
            """INSERT INTO inner_state_shared
               (profile_key, text, variables, felt_at, updated_at, source_revision)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(profile_key) DO UPDATE SET
                 text=excluded.text, variables=excluded.variables,
                 felt_at=excluded.felt_at, updated_at=excluded.updated_at,
                 source_revision=excluded.source_revision
               WHERE excluded.updated_at >= inner_state_shared.updated_at
                 AND excluded.text <> inner_state_shared.text""",
            (profile_key, text, json.dumps(dict(variables)), now, now, int(source_revision)),
        )
        return cursor.rowcount == 1
