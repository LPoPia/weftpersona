from __future__ import annotations

import json
import base64
import math
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

from ..infra.archive import _append_fsynced_jsonl
from .curve import (
    HALF_LIFE_MAX_LONG_DAYS, HALF_LIFE_MAX_SHORT_DAYS, HALF_LIFE_MIN_DAYS,
    clamp_stored_half_life_ceilings, has_over_cap_half_life,
)
from .event_store import SUMMARY_SOURCE_HISTORICAL
from .schema import (
    VIDEO_WATCHES_AUX_SQL, VIDEO_WATCHES_TABLE_SQL, _create_video_indexes,
    _mark_ambiguous_legacy_video_rows,
)
from .sqlite_facade import SqliteFacade
from ..infra.video_record_values import VIDEO_MODE_MIXED, video_caption_items, video_scope_value

_BLOB_TAG = "__sqlite_blob_base64__"


def _encode_sqlite_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _encode_sqlite_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_sqlite_value(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {_BLOB_TAG: base64.b64encode(bytes(value)).decode("ascii")}
    return value


def _provenance_without_vectors(entries: Any) -> tuple[list[Any], bool]:
    cleaned: list[Any] = []
    dropped = False
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict):
            event = entry.get("event")
            if isinstance(event, dict) and "vec" in event:
                entry = {**entry, "event": {key: value for key, value in event.items() if key != "vec"}}
                dropped = True
        cleaned.append(entry)
    return cleaned, dropped

_EVENT_COLUMNS = {
    "tier": "TEXT DEFAULT 'short'",
    "summary": "TEXT DEFAULT ''",
    "summary_source": "TEXT DEFAULT ''",
    "promoted_at": "TEXT",
    "cross_session_hits": "INTEGER DEFAULT 0",
    "pending_to": "TEXT DEFAULT ''",
    "distilled_at": "TEXT",
    "expired_at": "TEXT",
    "recall_count": "INTEGER NOT NULL DEFAULT 0",
    "last_active_at": "TEXT NOT NULL DEFAULT ''",
    "half_life_days": "REAL NOT NULL DEFAULT 14.0",
    "curve_ttl": "TEXT NOT NULL DEFAULT ''",
    "original_fingerprint": "TEXT NOT NULL DEFAULT ''",
    "asked_at": "TEXT NOT NULL DEFAULT ''",
    "a_conflict": "INTEGER NOT NULL DEFAULT 0",
    "affect": "INTEGER NOT NULL DEFAULT 0",
    "owner_key": "TEXT NOT NULL DEFAULT ''",
    "candidate_reason": "TEXT DEFAULT ''",
    "staged_at": "TEXT",
    "promoted_reason": "TEXT DEFAULT ''",
    "asked_via": "TEXT DEFAULT ''",
    "asked_session": "TEXT DEFAULT ''",
    "source_snapshot": "TEXT NOT NULL DEFAULT ''",
    "source_snapshot_sha256": "TEXT NOT NULL DEFAULT ''",
    "source_context": "TEXT NOT NULL DEFAULT ''",
    "public_excerpt": "TEXT NOT NULL DEFAULT ''",
    "experience_subject": "TEXT NOT NULL DEFAULT ''",
    "fact_subject": "TEXT NOT NULL DEFAULT ''",
    "involved_owner_keys": "TEXT NOT NULL DEFAULT ''",
    "read_scope": "TEXT NOT NULL DEFAULT ''",
    "shared_view": "TEXT NOT NULL DEFAULT ''",
}

_CURVE_BACKFILL_BY_COLUMN: tuple[tuple[str, str], ...] = (
    (
        "last_active_at",
        """UPDATE events SET last_active_at = COALESCE(
             NULLIF(asked_at, ''), NULLIF(updated_at, ''), created_at)""",
    ),
    (
        "recall_count",
        """UPDATE events SET recall_count = CASE
             WHEN COALESCE(tier, '') IN ('long', 'important', 'distilled')
               THEN MAX(3, COALESCE(cross_session_hits, 0))
             ELSE MAX(0, COALESCE(cross_session_hits, 0)) END""",
    ),
    (
        "half_life_days",
        f"""UPDATE events SET half_life_days = MIN(
             CASE WHEN COALESCE(tier, '') IN ('long', 'important', 'distilled')
               THEN {HALF_LIFE_MAX_LONG_DAYS} ELSE {HALF_LIFE_MAX_SHORT_DAYS} END,
             MAX(
               CASE WHEN COALESCE(tier, '') IN ('long', 'important', 'distilled')
                 THEN {HALF_LIFE_MAX_SHORT_DAYS} ELSE {HALF_LIFE_MIN_DAYS} END,
               (CASE
                  WHEN NULLIF(ttl_expire, '') IS NOT NULL
                    AND julianday(ttl_expire) > julianday(created_at)
                    THEN MIN({HALF_LIFE_MAX_LONG_DAYS}, MAX(1.0, julianday(ttl_expire) - julianday(created_at)))
                  WHEN COALESCE(topic, '') IN ('情绪', '关系') THEN 90.0
                  ELSE 6.0 END)
               * (1.0 + 2.5 * COALESCE(importance, 0.5))
               * (1.0 + 0.5 * MIN(1.0, MAX(0.0, COALESCE(affect, 0) / 3.0)))))""",
    ),
    (
        "curve_ttl",
        "UPDATE events SET curve_ttl = COALESCE(ttl_expire, '')",
    ),
)
_CURVE_COLUMNS = {column for column, _statement in _CURVE_BACKFILL_BY_COLUMN}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_columns(conn: sqlite3.Connection, table: str, additions: dict[str, str]) -> set[str]:
    columns = _columns(conn, table)
    added = set()
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            added.add(name)
    return added


def _execute_schema_script(conn: sqlite3.Connection, script: str) -> None:
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                conn.execute(statement)
            statement = ""
    if statement.strip():
        raise RuntimeError("incomplete video schema SQL")


def _has_video_scope_unique_index(conn: sqlite3.Connection) -> bool:
    for row in conn.execute("PRAGMA index_list(video_watches)"):
        if not int(row[2]):
            continue
        index = str(row[1]).replace('"', '""')
        columns = [str(info[2]) for info in conn.execute(f'PRAGMA index_info("{index}")')]
        if columns == ["platform", "source_object_id", "cid", "owner_key", "read_scope"]:
            return True
    return False


def _rebuild_video_scope_schema(conn: sqlite3.Connection) -> None:
    if _has_video_scope_unique_index(conn):
        return
    duplicates = conn.execute(
        "SELECT 1 FROM video_watches GROUP BY platform,source_object_id,cid,owner_key,read_scope "
        "HAVING COUNT(*)>1 LIMIT 1"
    ).fetchone()
    if duplicates is not None:
        raise RuntimeError("legacy video rows collide under owner/read-scope identity")

    table_sql = VIDEO_WATCHES_TABLE_SQL.replace(
        "CREATE TABLE IF NOT EXISTS video_watches",
        "CREATE TABLE video_watches_upgrade",
        1,
    )
    conn.execute("DROP TRIGGER IF EXISTS video_ai")
    conn.execute("DROP TRIGGER IF EXISTS video_ad")
    conn.execute("DROP TRIGGER IF EXISTS video_au")
    conn.execute(table_sql)
    columns = (
        "id,bvid,platform,source_object_id,title,cid,mode,analysis,captions,summary,source_context,raw_observation,"
        "read_scope,public_excerpt,owner_key,session_id,watched_at,created_at,updated_at"
    )
    conn.execute(
        f"INSERT INTO video_watches_upgrade ({columns}) "
        f"SELECT {columns} FROM video_watches"
    )
    conn.execute("DROP TABLE video_watches")
    conn.execute("ALTER TABLE video_watches_upgrade RENAME TO video_watches")
    _execute_schema_script(conn, VIDEO_WATCHES_AUX_SQL)


def _repair_event_curve(conn: sqlite3.Connection) -> None:
    over_cap = has_over_cap_half_life(conn)
    empty_summaries = conn.execute(
        "SELECT id, raw, tier, summary_source FROM events "
        "WHERE trim(COALESCE(summary, '')) = '' AND trim(COALESCE(raw, '')) <> ''"
    ).fetchall()
    if not (over_cap or empty_summaries):
        return
    conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
    if over_cap:
        clamp_stored_half_life_ceilings(conn)
    import re
    for row in empty_summaries:
        raw = str(row["raw"] or "")
        cleaned = re.sub(r"^\[[^]]+\]\s*", "", raw).strip()
        summary = re.sub(r"\s+", " ", cleaned or raw.strip())[:150]
        if not summary:
            continue
        source = str(row["summary_source"] or "")
        if not source and str(row["tier"] or "") in ("long", "important", "distilled"):
            source = SUMMARY_SOURCE_HISTORICAL
        conn.execute(
            "UPDATE events SET summary=?, summary_source=? WHERE id=?",
            (summary, source, int(row["id"])),
        )


def _normalize_video_observations(conn: sqlite3.Connection, archive_dir: Path) -> int:
    rows = conn.execute(
        "SELECT id, raw_observation FROM video_watches "
        "WHERE raw_observation LIKE '%\"vec\"%'"
    ).fetchall()
    pending = []
    for row in rows:
        original = str(row["raw_observation"] or "")
        try:
            parsed = json.loads(original)
        except (TypeError, ValueError):
            continue
        cleaned, dropped = _provenance_without_vectors(parsed)
        if dropped:
            pending.append((int(row["id"]), original, json.dumps(cleaned, ensure_ascii=False)))
    if not pending:
        return 0
    stamp = datetime.now().astimezone().isoformat()
    _append_fsynced_jsonl(
        archive_dir / "video-raw-observation-archived.jsonl",
        (json.dumps({
            "archived_at": stamp,
            "archive_reason": "strip_duplicated_event_vector",
            "id": row_id,
            "raw_observation": original,
        }, ensure_ascii=False) for row_id, original, _cleaned in pending),
    )
    for row_id, _original, cleaned in pending:
        conn.execute("UPDATE video_watches SET raw_observation=? WHERE id=?", (cleaned, row_id))
    return len(pending)


def _video_caption_items(value: object) -> list[dict[str, object]]:
    items = video_caption_items(value)
    return [
        {key: item for key, item in caption.items()
         if key not in {"source_snapshot", "source_snapshot_sha256"}}
        for caption in items
    ]


def upgrade_meme_schema(conn: sqlite3.Connection) -> None:
    _ensure_columns(conn, "meme_records", {
        "tier": "TEXT NOT NULL DEFAULT 'common'",
        "visibility_scope": "TEXT NOT NULL DEFAULT 'private'",
        "evidence_status": "TEXT NOT NULL DEFAULT 'uncertain'",
        "source_material_id": "INTEGER",
    })


def upgrade_legacy_database(
    conn: sqlite3.Connection,
    database_dir: Path,
    archive_dir: Path,
) -> None:
    database_dir = Path(database_dir)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    added = _ensure_columns(conn, "events", _EVENT_COLUMNS)
    if added & _CURVE_COLUMNS:
        conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
        for column, statement in _CURVE_BACKFILL_BY_COLUMN:
            if column in added:
                conn.execute(statement)
    _repair_event_curve(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_owner_key ON events(owner_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_experience_subject ON events(experience_subject)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_self_attribution "
        "ON events(owner_key, experience_subject, fact_subject)"
    )

    _ensure_columns(conn, "event_occurrences", {
        "source_snapshot": "TEXT NOT NULL DEFAULT ''",
        "source_snapshot_sha256": "TEXT NOT NULL DEFAULT ''",
    })
    _ensure_columns(conn, "event_reviews", {
        "reviewed_text": "TEXT NOT NULL DEFAULT ''",
        "evidence_group": "TEXT NOT NULL DEFAULT ''",
    })
    _ensure_columns(conn, "daily_context", {
        "source_kind": "TEXT NOT NULL DEFAULT 'statement'",
        "author_kind": "TEXT NOT NULL DEFAULT 'unknown'",
    })
    _ensure_columns(conn, "inner_state_revisions", {
        "recent_words_at": "REAL",
        "owner_key": "TEXT NOT NULL DEFAULT ''",
        "visibility": "TEXT NOT NULL DEFAULT 'private'",
        "last_spark_at": "REAL NOT NULL DEFAULT 0.0",
    })
    _ensure_columns(conn, "impression_revisions", {
        "baseline": "TEXT",
        "recent": "TEXT",
    })
    _ensure_columns(conn, "video_watches", {
        "platform": "TEXT NOT NULL DEFAULT 'bilibili'",
        "source_object_id": "TEXT NOT NULL DEFAULT ''",
        "title": "TEXT NOT NULL DEFAULT ''",
        "cid": "TEXT NOT NULL DEFAULT ''",
        "mode": "TEXT NOT NULL DEFAULT ''",
        "analysis": "TEXT NOT NULL DEFAULT ''",
        "captions": "TEXT NOT NULL DEFAULT ''",
        "summary": "TEXT NOT NULL DEFAULT ''",
        "source_context": "TEXT NOT NULL DEFAULT ''",
        "raw_observation": "TEXT NOT NULL DEFAULT ''",
        "read_scope": "TEXT NOT NULL DEFAULT ''",
        "public_excerpt": "TEXT NOT NULL DEFAULT ''",
        "owner_key": "TEXT NOT NULL DEFAULT ''",
        "session_id": "TEXT NOT NULL DEFAULT ''",
        "watched_at": "TEXT NOT NULL DEFAULT ''",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    })
    conn.execute("DROP TRIGGER IF EXISTS video_ai")
    conn.execute("DROP TRIGGER IF EXISTS video_ad")
    conn.execute("DROP TRIGGER IF EXISTS video_au")
    conn.execute("UPDATE video_watches SET platform='bilibili' WHERE platform='' OR platform IS NULL")
    conn.execute("UPDATE video_watches SET source_object_id=bvid WHERE source_object_id='' OR source_object_id IS NULL")
    _mark_ambiguous_legacy_video_rows(conn)
    upgrade_meme_schema(conn)
    _rebuild_video_scope_schema(conn)
    _create_video_indexes(conn, ensure_scope_unique=True)
    _migrate_spark_clock(conn, database_dir, archive_dir)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inner_state_owner "
        "ON inner_state_revisions(profile_key, owner_key, revision)"
    )

    conn.execute("INSERT INTO video_fts(video_fts) VALUES('rebuild')")
    _normalize_video_observations(conn, archive_dir)
    video = _LegacyVideoMigration(conn, archive_dir)
    video._migrate_events()

    for table in ("mood_state", "impression_session_views", "commitment_legacy_queue"):
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            conn.execute(f"DROP TABLE {table}")

    event_columns = _columns(conn, "events")
    for column in ("a_serious", "a_intent", "a_rel", "a_self"):
        if column in event_columns:
            conn.execute(f"ALTER TABLE events DROP COLUMN {column}")


def _migrate_spark_clock(
    conn: sqlite3.Connection,
    database_dir: Path,
    archive_dir: Path,
) -> None:
    by_owner: dict[tuple[str, str], float] = {}
    marked_rows = conn.execute(
        "SELECT profile_key, owner_key, MAX(updated_at) AS spark_at "
        "FROM inner_state_revisions "
        "WHERE owner_key<>'' AND text LIKE '%（刚才脑子里闪过一个念头：「%' "
        "GROUP BY profile_key, owner_key"
    ).fetchall()
    for row in marked_rows:
        key = (str(row["profile_key"] or ""), str(row["owner_key"] or ""))
        if all(key):
            by_owner[key] = max(by_owner.get(key, 0.0), float(row["spark_at"] or 0.0))

    status_path = Path(database_dir) / "status.json"
    unknown: dict[str, Any] = {}
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("cannot read legacy spark clock status") from exc
        inner = payload.get("inner_state") if isinstance(payload, dict) else None
        marks = inner.get("spark_at_by_profile") if isinstance(inner, dict) else None
        if isinstance(marks, dict):
            for raw_key, raw_value in marks.items():
                key = str(raw_key)
                if "|" not in key:
                    unknown[key] = raw_value
                    continue
                profile_key, owner_key = key.split("|", 1)
                if not profile_key or not owner_key:
                    unknown[key] = raw_value
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    unknown[key] = raw_value
                    continue
                if not math.isfinite(value) or value < 0:
                    unknown[key] = raw_value
                    continue
                exists = conn.execute(
                    "SELECT 1 FROM inner_state_revisions "
                    "WHERE profile_key=? AND owner_key=? LIMIT 1",
                    (profile_key, owner_key),
                ).fetchone()
                if exists is None:
                    unknown[key] = raw_value
                    continue
                owner = (profile_key, owner_key)
                by_owner[owner] = max(by_owner.get(owner, 0.0), value)
        elif marks not in (None, {}):
            unknown["unparsed"] = marks

    if unknown:
        _append_fsynced_jsonl(
            Path(archive_dir) / "inner-state-spark-clock-archived.jsonl",
            (json.dumps({
                "archived_at": datetime.now().astimezone().isoformat(),
                "archive_reason": "unclaimed_legacy_spark_clock",
                "spark_at_by_profile": unknown,
            }, ensure_ascii=False),),
        )

    for (profile_key, owner_key), spark_at in by_owner.items():
        conn.execute(
            "UPDATE inner_state_revisions SET last_spark_at=MAX(COALESCE(last_spark_at,0),?) "
            "WHERE profile_key=? AND owner_key=? "
            "AND revision=(SELECT MAX(revision) FROM inner_state_revisions "
            "WHERE profile_key=? AND owner_key=?)",
            (spark_at, profile_key, owner_key, profile_key, owner_key),
        )


class _LegacyVideoMigration:
    def __init__(self, conn: sqlite3.Connection, archive_dir: Path):
        self._sqlite = SqliteFacade(conn=conn)
        self._archive_dir = archive_dir

    def _migrate_events(self) -> Dict[str, Any]:
        """Move legacy video rows only inside a versioned schema upgrade."""
        if not self._sqlite.is_open:
            return {"migrated": 0, "archived": 0}

        archive_dir = self._archive_dir
        archive_path = archive_dir / "events-archived.jsonl"

        def _load_legacy_rows(conn: sqlite3.Connection):
            evt_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if evt_table is None:
                return None, False, []
            rows = conn.execute(
                """
                SELECT *
                  FROM events
                 WHERE (
                    (json_valid(source_context) AND json_extract(source_context, '$.platform') = 'bilibili'
                     AND json_extract(source_context, '$.ref') LIKE 'https://www.bilibili.com/video/BV%/'
                     AND json_extract(source_context, '$.kind') = 'self')
                    OR raw LIKE '%在B站收到该视频的抽样画面分析%'
                    OR raw LIKE '%在B站阅读了该视频的字幕%'
                 )
                """
            ).fetchall()
            if not rows:
                return [], False, []
            has_occ = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_occurrences'"
            ).fetchone() is not None
            entries = []
            for row in rows:
                eid = int(row["id"])
                occ_rows = []
                if has_occ:
                    occ_cur = conn.execute(
                        "SELECT * FROM event_occurrences WHERE event_id = ?",
                        (eid,),
                    ).fetchall()
                    occ_rows = [dict(o) for o in occ_cur]
                entries.append({
                    "event": dict(row),
                    "occurrences": occ_rows,
                })
            return rows, has_occ, entries

        rows, has_occ, event_entries = self._sqlite.read(_load_legacy_rows)
        if not rows:
            return {"migrated": 0, "archived": 0}

        def _canonical_archive_key(entry: Mapping[str, Any]) -> str:
            return json.dumps(
                {k: v for k, v in entry.items() if k != "archived_at"},
                ensure_ascii=False,
                sort_keys=True,
            )

        already_archived: set[str] = set()
        if archive_path.is_file():
            for line in archive_path.read_text(encoding="utf-8").split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    existing = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if (isinstance(existing, dict)
                        and existing.get("archive_reason") == "migrate_video_to_dedicated_store"):
                    already_archived.add(_canonical_archive_key(existing))

        stamp = datetime.now().astimezone().isoformat()

        pending_archive_lines: list[str] = []
        for item in event_entries:
            ev = item["event"]
            entry = {
                "archived_at": stamp,
                "archive_reason": "migrate_video_to_dedicated_store",
                "id": int(ev["id"]),
                "event": _encode_sqlite_value(ev),
                "occurrences": _encode_sqlite_value(item["occurrences"]),
                "raw": _encode_sqlite_value(ev.get("raw") or ""),
                "summary": _encode_sqlite_value(ev.get("summary") or ""),
                "source_context": _encode_sqlite_value(
                    ev.get("source_context") or ""),
                "owner_key": _encode_sqlite_value(ev.get("owner_key") or ""),
                "read_scope": _encode_sqlite_value(ev.get("read_scope") or ""),
                "public_excerpt": _encode_sqlite_value(
                    ev.get("public_excerpt") or ""),
            }
            canonical = _canonical_archive_key(entry)
            if canonical in already_archived:
                continue
            already_archived.add(canonical)
            pending_archive_lines.append(json.dumps(entry, ensure_ascii=False))

        if pending_archive_lines or archive_path.is_file():
            _append_fsynced_jsonl(archive_path, pending_archive_lines)

        groups: Dict[tuple, dict] = {}
        migrated_ids = []
        for item in event_entries:
            row = item["event"]
            raw_text = str(row.get("raw") or "")
            summary_text = str(row.get("summary") or "")
            source_ctx_str = str(row.get("source_context") or "")

            bv_match = re.search(r"BV[0-9A-Za-z]{10}", source_ctx_str) or re.search(r"BV[0-9A-Za-z]{10}", raw_text)
            if not bv_match:
                continue
            bvid = bv_match.group(0)

            title_match = re.search(r"《(.*?)》", raw_text) or re.search(r"《(.*?)》", summary_text)
            title = title_match.group(1) if title_match else bvid

            cid_match = re.search(r"CID\s*([0-9]+)", raw_text) or re.search(r"CID\s*([0-9]+)", summary_text)
            cid = cid_match.group(1) if cid_match else ""

            mode = "video_analysis" if "抽样画面分析" in raw_text else "captions_only"

            analysis = ""
            if mode == "video_analysis":
                if "内容理解未经独立核实：" in raw_text:
                    analysis = raw_text.split("内容理解未经独立核实：", 1)[1]
                else:
                    analysis = raw_text

            caption_items = _video_caption_items(item["occurrences"])
            if mode == "captions_only":
                tail = (
                    raw_text.split("内容理解未经独立核实：", 1)[1]
                    if "内容理解未经独立核实：" in raw_text else "")
                for part in re.split(r"[；\n]", tail):
                    text = part.strip()
                    if text:
                        caption_items.append({"text": text})

            owner = str(row.get("owner_key") or "").strip()
            read_scope = video_scope_value(row.get("read_scope"))
            watched_at = str(row.get("created_at") or stamp)
            key = (bvid, owner, read_scope)
            group = groups.get(key)
            if group is None:
                group = {
                    "bvid": bvid, "owner": owner, "read_scope": read_scope,
                    "title": title, "cid": cid, "session_id": "",
                    "modes": [], "analyses": [], "captions": [], "caption_keys": set(),
                    "summaries": [], "source_contexts": [], "public_excerpts": [],
                    "watched_at": [], "created_at": [], "provenance": [],
                }
                groups[key] = group
            if not group["title"] or group["title"] == bvid:
                group["title"] = title
            if not group["cid"]:
                group["cid"] = cid
            if not group["session_id"]:
                group["session_id"] = str(row.get("session_id") or "")
            if mode not in group["modes"]:
                group["modes"].append(mode)
            if analysis and analysis not in group["analyses"]:
                group["analyses"].append(analysis)
            for caption in caption_items:
                caption_key = json.dumps(caption, sort_keys=True, ensure_ascii=False)
                if caption_key not in group["caption_keys"]:
                    group["caption_keys"].add(caption_key)
                    group["captions"].append(caption)
            if summary_text and summary_text not in group["summaries"]:
                group["summaries"].append(summary_text)
            if source_ctx_str and source_ctx_str not in group["source_contexts"]:
                group["source_contexts"].append(source_ctx_str)
            public_excerpt = str(row.get("public_excerpt") or "")
            if public_excerpt and public_excerpt not in group["public_excerpts"]:
                group["public_excerpts"].append(public_excerpt)
            group["watched_at"].append(watched_at)
            group["created_at"].append(str(row.get("created_at") or stamp))
            group["provenance"].append({
                "event": dict(row), "occurrences": item["occurrences"]})
            migrated_ids.append(int(row["id"]))

        with self._sqlite.transaction() as conn:
            for key, group in groups.items():
                bvid, owner, read_scope = key
                analyses = list(group["analyses"])
                captions = list(group["captions"])
                caption_keys = set(group["caption_keys"])
                summaries = list(group["summaries"])
                source_contexts = list(group["source_contexts"])
                public_excerpts = list(group["public_excerpts"])
                modes = list(group["modes"])
                provenance = list(group["provenance"])

                existing = conn.execute(
                    "SELECT mode, analysis, captions, summary, source_context, "
                    "public_excerpt, raw_observation, watched_at, created_at "
                    "FROM video_watches WHERE bvid=? AND owner_key=? AND read_scope=?",
                    (bvid, owner, read_scope),
                ).fetchone()
                if existing is not None:
                    old_analysis = str(existing["analysis"] or "")
                    old_parts = [part for part in old_analysis.split("\n\n")
                                 if part.strip()]
                    for part in reversed(old_parts):
                        if part not in analyses:
                            analyses.insert(0, part)
                    for caption in _video_caption_items(existing["captions"]):
                        caption_key = json.dumps(caption, sort_keys=True, ensure_ascii=False)
                        if caption_key not in caption_keys:
                            caption_keys.add(caption_key)
                            captions.append(caption)
                    old_summary = str(existing["summary"] or "")
                    old_summary_parts = [part for part in old_summary.splitlines()
                                         if part.strip()]
                    for part in reversed(old_summary_parts):
                        if part not in summaries:
                            summaries.insert(0, part)
                    old_source = str(existing["source_context"] or "").strip()
                    if old_source and old_source not in source_contexts:
                        source_contexts.insert(0, old_source)
                    old_excerpt = str(existing["public_excerpt"] or "").strip()
                    if old_excerpt and old_excerpt not in public_excerpts:
                        public_excerpts.insert(0, old_excerpt)
                    old_mode = str(existing["mode"] or "").strip()
                    for old_part in old_mode.split("+"):
                        if old_part and old_part not in modes:
                            modes.append(old_part)
                    try:
                        old_provenance = json.loads(existing["raw_observation"] or "[]")
                    except (TypeError, ValueError):
                        old_provenance = []
                    if isinstance(old_provenance, list):
                        provenance = old_provenance + provenance
                    group["watched_at"].append(str(existing["watched_at"] or ""))
                    group["created_at"].append(str(existing["created_at"] or ""))

                merged_mode = (
                    VIDEO_MODE_MIXED
                    if {"video_analysis", "captions_only"} <= set(modes)
                    else "+".join(sorted(m for m in modes if m)))
                watched = [item for item in group["watched_at"] if item]
                created = [item for item in group["created_at"] if item]
                provenance_rows = []
                provenance_ids = set()
                for entry in provenance:
                    event = entry.get("event") if isinstance(entry, dict) else None
                    event_id = event.get("id") if isinstance(event, dict) else None
                    if event_id is not None:
                        if event_id in provenance_ids:
                            continue
                        provenance_ids.add(event_id)
                    provenance_rows.append(entry)
                provenance_rows, _vectors_dropped = _provenance_without_vectors(
                    provenance_rows)

                conn.execute(
                    """
                    INSERT INTO video_watches (
                        bvid, platform, source_object_id, title, cid, mode, analysis, captions, summary,
                        source_context, raw_observation, read_scope, public_excerpt,
                        owner_key, session_id, watched_at, created_at, updated_at
                    ) VALUES (?, 'bilibili', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(platform, source_object_id, cid, owner_key, read_scope)
                    WHERE source_object_id <> '' DO UPDATE SET
                        title = excluded.title,
                        cid = excluded.cid,
                        mode = excluded.mode,
                        analysis = excluded.analysis,
                        captions = excluded.captions,
                        summary = excluded.summary,
                        source_context = excluded.source_context,
                        raw_observation = excluded.raw_observation,
                        read_scope = excluded.read_scope,
                        public_excerpt = excluded.public_excerpt,
                        owner_key = excluded.owner_key,
                        session_id = excluded.session_id,
                        watched_at = excluded.watched_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        bvid, bvid, group["title"], group["cid"], merged_mode,
                        "\n\n".join(analyses),
                        json.dumps(captions, ensure_ascii=False) if captions else "",
                        "\n".join(summaries),
                        source_contexts[0] if source_contexts else "",
                        json.dumps(_encode_sqlite_value(provenance_rows),
                                   ensure_ascii=False),
                        read_scope,
                        public_excerpts[0] if public_excerpts else "",
                        owner, group["session_id"],
                        min(watched) if watched else stamp,
                        min(created) if created else stamp,
                        stamp,
                    ),
                )

            if migrated_ids:
                placeholders = ",".join("?" for _ in migrated_ids)
                if has_occ:
                    conn.execute(
                        f"DELETE FROM event_occurrences WHERE event_id IN ({placeholders})",
                        migrated_ids,
                    )
                conn.execute(
                    f"DELETE FROM events WHERE id IN ({placeholders})",
                    migrated_ids,
                )

        return {"migrated": len(migrated_ids), "archived": len(rows)}
