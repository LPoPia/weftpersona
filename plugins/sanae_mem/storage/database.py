from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from ..infra.atomic_io import fsync_directory
from .curve import clamp_stored_half_life_ceilings
from .schema import (
    CURRENT_SCHEMA_SQL, CURRENT_SCHEMA_VERSION, MAINTAINER_BINDING_SCHEMA_SQL,
    RECEIPT_SCHEMA_SQL, VIDEO_WATCHES_AUX_SQL, VIDEO_WATCHES_TABLE_SQL,
    _create_video_indexes, _mark_ambiguous_legacy_video_rows,
)
from .sqlite_facade import connect_readonly, connect_readwrite


def open_database(
    path: Path | str, *, archive_dir: Path | str | None = None,
) -> sqlite3.Connection:
    database_path = Path(path)
    conn = connect_readwrite(database_path)
    try:
        initialize_database(conn, database_path, archive_dir=archive_dir)
    except BaseException:
        conn.close()
        raise
    return conn


def _schema_flags(conn: sqlite3.Connection) -> tuple[bool, bool, bool, bool]:
    meme_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(meme_records)")}
    meme_current = {"tier", "visibility_scope", "evidence_status", "source_material_id"} <= meme_columns
    video_current = _video_identity_schema_current(conn)
    binding_current = _account_binding_schema_current(conn)
    source_current = _group_source_schema_current(conn)
    return meme_current, video_current, binding_current, source_current


def _upgrade_v2_plus_locked(conn: sqlite3.Connection, database_path: Path) -> None:
    from .legacy_upgrade import _execute_schema_script, upgrade_meme_schema

    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"sanae_mem database version {version} is newer than supported "
            f"version {CURRENT_SCHEMA_VERSION}"
        )
    meme_current, video_current, binding_current, source_current = _schema_flags(conn)
    if (version == CURRENT_SCHEMA_VERSION and meme_current and video_current
            and binding_current and source_current):
        _execute_schema_script(conn, RECEIPT_SCHEMA_SQL)
        conn.commit()
        return
    backup_path: Path | None = None
    try:
        backup_path = _backup_before_upgrade(conn, database_path, version)
        _execute_schema_script(conn, RECEIPT_SCHEMA_SQL)
        if not video_current:
            _ensure_video_identity_columns(conn)
            _create_video_indexes(conn, ensure_scope_unique=True)
        if not meme_current:
            upgrade_meme_schema(conn)
        # v4: the binding table is additive; existing rows, owners and source facts are untouched.
        if not binding_current:
            _upgrade_account_binding_schema(conn)
        _execute_schema_script(conn, MAINTAINER_BINDING_SCHEMA_SQL)
        # v5: only the missing provenance column is appended; every stored row keeps its text,
        # owner, evidence and subject and reads the new column as the empty unknown.
        if not source_current:
            _upgrade_group_source_columns(conn)
        # v6: clamp any stored events whose half_life_days drifted above tier ceilings
        # (short/pending/empty <= 60, others <= 120) without raising lower values.
        if version < 6:
            clamp_stored_half_life_ceilings(conn)
        conn.execute(f"PRAGMA user_version={CURRENT_SCHEMA_VERSION}")
        conn.commit()
    except BaseException as exc:
        if conn.in_transaction:
            conn.rollback()
        backup_note = f" and backup is available at {backup_path}" if backup_path is not None else ""
        raise RuntimeError(
            f"sanae_mem schema upgrade from version {version} failed; source is preserved"
            f"{backup_note}: {exc}"
        ) from exc


def initialize_database(
    conn: sqlite3.Connection,
    path: Path | str,
    *,
    archive_dir: Path | str | None = None,
) -> None:
    database_path = Path(path)
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"sanae_mem database version {version} is newer than supported "
            f"version {CURRENT_SCHEMA_VERSION}"
        )
    if version >= 2:
        meme_current, video_current, binding_current, source_current = _schema_flags(conn)
        if (version == CURRENT_SCHEMA_VERSION and meme_current and video_current
                and binding_current and source_current):
            conn.executescript(RECEIPT_SCHEMA_SQL)
            return

        conn.execute("BEGIN IMMEDIATE")
        try:
            _upgrade_v2_plus_locked(conn, database_path)
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        return

    from .legacy_upgrade import _execute_schema_script, upgrade_legacy_database

    backup_path: Path | None = None
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version >= 2:
            _upgrade_v2_plus_locked(conn, database_path)
            return

        has_schema = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','index','trigger','view') "
            "AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone() is not None
        if not has_schema:
            _execute_schema_script(conn, CURRENT_SCHEMA_SQL)
            _create_event_indexes(conn)
            _create_video_indexes(conn, ensure_scope_unique=True)
            conn.execute(f"PRAGMA user_version={CURRENT_SCHEMA_VERSION}")
            conn.commit()
            return

        backup_path = _backup_before_upgrade(conn, database_path, version)
        _execute_schema_script(conn, CURRENT_SCHEMA_SQL)
        upgrade_legacy_database(
            conn,
            database_path.parent,
            Path(archive_dir) if archive_dir is not None else database_path.parent / "memory-archive",
        )
        # The pre-v2 script only creates missing tables; an existing legacy row table needs the
        # v5 provenance column appended the same way as the v2+ chain.
        _upgrade_group_source_columns(conn)
        conn.execute(f"PRAGMA user_version={CURRENT_SCHEMA_VERSION}")
        conn.commit()
    except BaseException as exc:
        if conn.in_transaction:
            conn.rollback()
        if isinstance(exc, RuntimeError) and str(exc).startswith("sanae_mem schema upgrade from version "):
            raise
        backup_note = f" and backup is available at {backup_path}" if backup_path is not None else ""
        if backup_path is None:
            raise
        raise RuntimeError(
            f"sanae_mem database upgrade from version {version} failed; "
            f"source is preserved{backup_note}: {exc}"
        ) from exc


def _create_event_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_owner_key ON events(owner_key)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_experience_subject ON events(experience_subject)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_self_attribution "
        "ON events(owner_key, experience_subject, fact_subject)"
    )


def _ensure_video_identity_columns(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(video_watches)")}
    unique_keys = []
    for row in conn.execute("PRAGMA index_list(video_watches)"):
        if int(row[2]):
            name = str(row[1]).replace('"', '""')
            unique_keys.append([str(info[2]) for info in conn.execute(f'PRAGMA index_info("{name}")')])
    identity_key = ["platform", "source_object_id", "cid", "owner_key", "read_scope"]
    if {"platform", "source_object_id"} <= columns and identity_key in unique_keys:
        return
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        _migrate_video_identity_locked(conn)
        if owns_transaction:
            conn.commit()
    except BaseException:
        if owns_transaction and conn.in_transaction:
            conn.rollback()
        raise


def _migrate_video_identity_locked(conn: sqlite3.Connection) -> None:
    from .legacy_upgrade import _execute_schema_script
    conn.execute("DROP TRIGGER IF EXISTS video_ai")
    conn.execute("DROP TRIGGER IF EXISTS video_ad")
    conn.execute("DROP TRIGGER IF EXISTS video_au")
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(video_watches)")}
    if "platform" not in columns:
        conn.execute("ALTER TABLE video_watches ADD COLUMN platform TEXT NOT NULL DEFAULT 'bilibili'")
    if "source_object_id" not in columns:
        conn.execute("ALTER TABLE video_watches ADD COLUMN source_object_id TEXT NOT NULL DEFAULT ''")
    conn.execute("UPDATE video_watches SET platform='bilibili' WHERE platform='' OR platform IS NULL")
    conn.execute("UPDATE video_watches SET source_object_id=bvid WHERE source_object_id='' OR source_object_id IS NULL")
    _mark_ambiguous_legacy_video_rows(conn)
    # Legacy v2 has UNIQUE(bvid, owner_key, read_scope), which prevents keeping
    # historical CID versions. Rebuild only this table to replace that key.
    unique_keys = []
    for row in conn.execute("PRAGMA index_list(video_watches)"):
        if int(row[2]):
            name = str(row[1]).replace('"', '""')
            unique_keys.append([str(info[2]) for info in conn.execute(f'PRAGMA index_info("{name}")')])
    if any("bvid" in columns for columns in unique_keys):
        conn.execute(VIDEO_WATCHES_TABLE_SQL.replace(
            "CREATE TABLE IF NOT EXISTS video_watches", "CREATE TABLE video_watches_upgrade", 1))
        columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(video_watches_upgrade)")]
        old_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(video_watches)")}
        common = [c for c in columns if c in old_columns]
        conn.execute(f"INSERT INTO video_watches_upgrade ({','.join(common)}) SELECT {','.join(common)} FROM video_watches")
        conn.execute("DROP TABLE video_watches")
        conn.execute("ALTER TABLE video_watches_upgrade RENAME TO video_watches")
    _execute_schema_script(conn, VIDEO_WATCHES_AUX_SQL)
    conn.execute("INSERT INTO video_fts(video_fts) VALUES('rebuild')")


def _video_identity_schema_current(conn: sqlite3.Connection) -> bool:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='video_watches'").fetchone() is None:
        return False
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(video_watches)")}
    if not {"platform", "source_object_id"} <= columns:
        return False
    for row in conn.execute("PRAGMA index_list(video_watches)"):
        if not int(row[2]):
            continue
        name = str(row[1]).replace('"', '""')
        cols = [str(info[2]) for info in conn.execute(f'PRAGMA index_info("{name}")')]
        if cols == ["platform", "source_object_id", "cid", "owner_key", "read_scope"]:
            return True
    return False


def _account_binding_schema_current(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA table_info(maintainer_account_bindings)").fetchall()
    if not rows:
        return False
    columns = {str(row[1]): int(row[5]) for row in rows}
    # Profile is part of the primary key, in this order. A table created by the first draft
    # lacks it; rebuild that additive table without touching any other table.
    return [name for name, position in sorted(columns.items(), key=lambda item: item[1])
            if position] == ["profile", "platform", "account_id", "scope_id"]


def _group_source_schema_current(conn: sqlite3.Connection) -> bool:
    for table in ("impression_revisions", "commitments"):
        columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
        if "source_context" not in columns:
            return False
    return True


def _upgrade_group_source_columns(conn: sqlite3.Connection) -> None:
    """Append the v5 provenance column where it is missing; no stored value is rewritten."""
    for table in ("impression_revisions", "commitments"):
        columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
        if "source_context" not in columns:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN source_context TEXT NOT NULL DEFAULT ''"
            )


def _upgrade_account_binding_schema(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='maintainer_account_bindings'"
    ).fetchone() is not None
    if not exists:
        return
    columns = {str(row[1]) for row in conn.execute(
        "PRAGMA table_info(maintainer_account_bindings)").fetchall()}
    profile_expr = "COALESCE(NULLIF(profile,''),'default')" if "profile" in columns else "'default'"
    conn.execute("ALTER TABLE maintainer_account_bindings RENAME TO _maintainer_bindings_v3")
    conn.execute("DROP INDEX IF EXISTS idx_maintainer_bindings_person")
    from .schema import MAINTAINER_BINDING_SCHEMA_SQL as binding_schema
    from .legacy_upgrade import _execute_schema_script
    _execute_schema_script(conn, binding_schema)
    conn.execute(
        "INSERT OR IGNORE INTO maintainer_account_bindings"
        "(profile, platform, account_id, scope_id, person_owner_key, created_at) "
        f"SELECT {profile_expr}, platform, account_id, scope_id, person_owner_key, created_at "
        "FROM _maintainer_bindings_v3"
    )
    conn.execute("DROP TABLE _maintainer_bindings_v3")


def _backup_before_upgrade(
    conn: sqlite3.Connection, database_path: Path, version: int,
) -> Path:
    backup_path = database_path.with_name(
        f"{database_path.name}.bak.v{version}.sqlite"
    ).resolve()
    descriptor, temporary_name = tempfile.mkstemp(dir=backup_path.parent, prefix=".sanae-db-backup-")
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        snapshot = connect_readonly(database_path) if conn.in_transaction else conn
        try:
            destination = sqlite3.connect(temporary_path)
            try:
                snapshot.backup(destination)
                destination.commit()
            finally:
                destination.close()
        finally:
            if snapshot is not conn:
                snapshot.close()
        backup_fd = os.open(temporary_path, os.O_RDWR)
        try:
            os.fsync(backup_fd)
        finally:
            os.close(backup_fd)
        _verify_backup(temporary_path, version)
        os.replace(temporary_path, backup_path)
        fsync_directory(backup_path.parent)
        return backup_path
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _verify_backup(path: Path, expected_version: int) -> None:
    conn = connect_readonly(path)
    try:
        check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()
    if check != "ok" or version != expected_version:
        raise RuntimeError(f"invalid pre-upgrade database backup: {path}")
