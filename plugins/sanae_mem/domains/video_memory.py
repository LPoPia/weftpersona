from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..storage.sqlite_facade import SqliteFacade

from ..infra.defaults import DEFAULT_PERSONA_NAME
from ..infra.event_fields import parse_source
from ..infra.source_descriptors import (
    SourceDescriptor, VideoOrigin, legacy_video_source, public_video_origin, source_descriptor,
    video_origin, video_storage_key,
    watch_row_origin,
)
from ..infra.video_record_values import (
    VIDEO_MODE_MIXED as VIDEO_MODE_MIXED,
    video_caption_items,
    video_playback_label,
    video_scope_value,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublicVideoWatch:
    watched_at: str
    title: str
    storage_key: str
    mode: str
    public_excerpt: str
    origin: VideoOrigin


def read_public_video_watches(
    conn: sqlite3.Connection, *, owners: tuple[str, ...], since: float, through: float,
) -> list[PublicVideoWatch]:
    """Read only public columns across owners, including title-only unreviewed watches."""
    if not owners:
        return []
    placeholders = ",".join("?" for _ in owners)
    rows = conn.execute(
        f"""
        SELECT watched_at, title, bvid, mode, COALESCE(public_excerpt, ''),
               COALESCE(platform, ''), COALESCE(source_object_id, ''), source_context
          FROM video_watches
         WHERE (owner_key = '' OR owner_key NOT IN ({placeholders}))
           AND json_valid(source_context)
           AND CASE WHEN read_scope = '' THEN 1 WHEN json_valid(read_scope)
                    THEN json_extract(read_scope, '$.kind') = 'self' ELSE 0 END
           AND json_extract(source_context, '$.certainty') = 'observed'
           AND json_extract(source_context, '$.visibility') = 'public'
           AND CAST(strftime('%s', watched_at) AS INTEGER) BETWEEN ? AND ?
         ORDER BY watched_at ASC
        """,
        (*owners, int(since), int(through) + 1),
    )
    result: list[PublicVideoWatch] = []
    for watched_at, title, storage_key, mode, excerpt, platform, object_id, source_json in rows:
        source = parse_source(source_json) or {}
        origin = watch_row_origin(source_json, platform, object_id, storage_key)
        if origin is not None and public_video_origin(source, origin):
            result.append(PublicVideoWatch(watched_at, title, storage_key, mode, excerpt, origin))
    return result


def public_watch_projection_rows(
    conn: sqlite3.Connection | None, *, owner_key: str, since: float, through: float,
    exclude_owners: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    """Compatibility view of the public projection."""
    if conn is None or not str(owner_key or "").strip():
        return []
    owners = tuple(sorted({str(item).strip() for item in (owner_key, *(exclude_owners or ()))
                           if str(item).strip()}))
    return [
        {"watched_at": row.watched_at, "title": row.title, "bvid": row.storage_key,
         "mode": row.mode, "public_excerpt": row.public_excerpt}
        for row in read_public_video_watches(conn, owners=owners, since=since, through=through)
    ]


def _unowned_legacy_watch(source: SourceDescriptor) -> tuple[str, tuple[str | int, ...]]:
    """Only an exact legacy storage identifier and canonical locator unlock unowned detail."""
    return (
        "owner_key = '' AND platform = ? "
        "AND CASE WHEN json_valid(read_scope) THEN json_extract(read_scope, '$.kind')='self' ELSE 0 END "
        "AND CASE WHEN json_valid(source_context) THEN "
        "     json_extract(source_context, '$.kind')='self' "
        " AND json_extract(source_context, '$.certainty')='observed' "
        " AND json_extract(source_context, '$.visibility')='public' "
        " AND CAST(json_extract(source_context, '$.ref') AS BLOB)=CAST((? || bvid || '/') AS BLOB) "
        " ELSE 0 END "
        "AND typeof(bvid)='text' AND length(CAST(bvid AS BLOB))=? AND bvid GLOB ?",
        (source.platform, source.legacy_video_ref_prefix,
         source.legacy_storage_id_length, source.legacy_storage_id_glob),
    )


@dataclass(frozen=True)
class VideoMemoryDependencies:
    sqlite: SqliteFacade
    current_session_id: Callable[[], str]
    current_persona_name: Callable[[], str]
    read_owner_keys: Callable[[str], list[str]]
    mutate_status: Callable[[Callable[[dict[str, Any]], dict[str, Any]]], None]
    resolve_explored_curiosity: Callable[[str], int]
    record_curiosity: Callable[[str, str, str], object]


class VideoMemory:
    def __init__(self, dependencies: VideoMemoryDependencies) -> None:
        self._dependencies = dependencies
        self._sqlite = dependencies.sqlite

    @staticmethod
    def _video_dedupe_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        seen = set()
        unique = []
        for row in rows:
            origin = watch_row_origin("", row["platform"], row["source_object_id"], row["bvid"])
            if origin is not None and origin.object_id:
                if origin.key in seen:
                    continue
                seen.add(origin.key)
            unique.append(row)
        return unique

    def _record_video_fts_failure(self, exc: sqlite3.OperationalError) -> None:
        logger.warning(
            "sanae_mem: video FTS query failed; using LIKE fallback (%s)",
            type(exc).__name__)
        stamp = datetime.now().astimezone().isoformat()

        def record(payload: dict[str, Any]) -> dict[str, Any]:
            state = payload.get("video_fts")
            state = dict(state) if isinstance(state, dict) else {}
            count = state.get("failure_count")
            state.update({
                "last_failure": type(exc).__name__,
                "last_failure_at": stamp,
                "failure_count": (count if type(count) is int else 0) + 1,
            })
            payload["video_fts"] = state
            return payload

        try:
            self._dependencies.mutate_status(record)
        except OSError as status_exc:
            logger.warning(
                "sanae_mem: video FTS status write failed (%s)",
                type(status_exc).__name__)


    def record_video_watch(
        self, observation: dict[str, Any], *, owner_key: str, public_observation: bool,
        review_public_excerpt: Callable[[str], bool] | None = None,
    ) -> bool:
        """Record a delivered public observation authorized by the integration.

        Stores full analysis, captions, source identity, raw observation and viewing scope
        in the dedicated video_watches table without TTL expiration.
        """

        session_id = str(self._dependencies.current_session_id() or "")

        write_owner = owner_key
        if not self._sqlite.is_open or not isinstance(observation, dict):
            return False

        origin = video_origin({"ref": str(observation.get("raw_locator") or "")},
                              platform=str(observation.get("platform") or ""),
                              object_id=str(observation.get("source_object_id") or ""),
                              legacy_id=str(observation.get("bvid") or ""), watch_row=True)
        mode = observation.get("mode")
        if (origin is None or not origin.platform or not origin.object_id or len(origin.object_id) > 200
                or observation.get("status") != "ok" or mode not in {"video_analysis", "captions_only"}):
            return False
        platform, object_id = origin.key
        descriptor = source_descriptor(platform)
        if (descriptor and descriptor.legacy_video_id
                and descriptor.legacy_video_id(object_id, "") != object_id):
            return False
        storage_key = video_storage_key(origin)

        full_analysis = str(observation.get("analysis") or "")

        lines = observation.get("lines") or (observation.get("captions") or {}).get("lines") or []
        captions_json = json.dumps(lines, ensure_ascii=False) if lines else ""

        if mode == "video_analysis":
            content = full_analysis[:1600]
        else:
            content = "；".join(str(x.get("text") or "") for x in lines[:24] if isinstance(x, dict))[:1300]

        if not content and not full_analysis:
            return False

        url = origin.ref or object_id
        title = str(observation.get("title") or object_id).replace("\n", " ")[:140]
        cid = str(observation.get("cid") or "")
        coverage = observation.get("coverage", mode)

        actor = str(self._dependencies.current_persona_name() or "").strip() or DEFAULT_PERSONA_NAME
        platform_name = origin.label
        experience = video_playback_label(mode)
        summary_prefix = (
            f"{actor}在{platform_name}{experience}：《{title}》（{object_id}，CID {cid or '未知'}）。"
            f"范围：{coverage}。来源：{mode}；内容理解未经独立核实："
        )
        summary = summary_prefix + content

        now_dt = datetime.now().astimezone()
        now_iso = now_dt.isoformat()
        receipt = observation.get("_watch_receipt")
        watched_at = (
            now_iso if isinstance(receipt, str) and receipt
            else str(observation.get("watched_at") or now_iso)
        )

        source = {
            "kind": "self" if descriptor and descriptor.legacy_unscoped_video else "video",
            "platform": platform,
            "ref": url,
            "time": now_iso,
            "certainty": "observed",
            "visibility": "public",
        }
        if source["kind"] == "video":
            source["source_object_id"] = object_id
        is_public = public_observation and public_video_origin(source, origin)
        if not is_public:
            source["visibility"] = "private"
        source_context_json = json.dumps(source, ensure_ascii=False)

        read_scope = {"kind": "self"} if is_public else {"kind": "owner", "owner": write_owner}
        read_scope_json = video_scope_value(read_scope)

        with self._sqlite.transaction() as conn:
            previous = conn.execute(
                "SELECT id, mode, analysis, captions, summary, raw_observation FROM video_watches "
                "WHERE platform=? AND source_object_id=? AND cid=? AND owner_key=? AND read_scope=?",
                (platform, object_id, cid, write_owner, read_scope_json),
            ).fetchone()
            original = dict(observation)
            receipts = []
            if previous is not None:
                try:
                    previous_observation = json.loads(previous["raw_observation"] or "{}")
                except (ValueError, TypeError):
                    previous_observation = {}
                if isinstance(previous_observation, dict):
                    receipts = list(previous_observation.get("_watch_receipts") or [])
                    original_receipt = previous_observation.get("_watch_receipt")
                    if original_receipt and original_receipt not in receipts:
                        receipts.append(original_receipt)
                if receipt and receipt in receipts:
                    return True
                if (isinstance(previous_observation, dict)
                        and previous_observation.get("_video_version_ambiguous") is True):
                    legacy_cid = f"legacy-ambiguous:{cid}:{int(previous['id'])}"
                    conn.execute(
                        "UPDATE video_watches SET cid=? WHERE id=?",
                        (legacy_cid, int(previous["id"])),
                    )
                    previous = None
                    receipts = []
                elif previous_observation:
                    original = (previous_observation if isinstance(previous_observation, dict)
                                else {"legacy_raw_observation": previous_observation})
            if isinstance(receipt, str) and receipt:
                original["_watch_receipts"] = [*receipts, receipt]
            raw_observation_json = json.dumps(original, ensure_ascii=False)
            if previous is None:
                conn.execute(
                    """
                    INSERT INTO video_watches (
                        bvid, platform, source_object_id, title, cid, mode, analysis, captions, summary,
                        source_context, raw_observation, read_scope, public_excerpt,
                        owner_key, session_id, watched_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        storage_key, platform, object_id, title, cid, mode, full_analysis, captions_json, summary,
                        source_context_json, raw_observation_json, read_scope_json, "",
                        write_owner, session_id, watched_at, now_iso, now_iso,
                    ),
                )
            else:
                analyses = [
                    part
                    for part in str(previous["analysis"] or "").split("\n\n")
                    if part.strip()
                ]
                added_analysis = []
                for part in (piece for piece in full_analysis.split("\n\n") if piece.strip()):
                    if part not in analyses:
                        analyses.append(part)
                        added_analysis.append(part)
                captions = video_caption_items(captions_json)
                caption_keys = {
                    json.dumps(item, sort_keys=True, ensure_ascii=False)
                    for item in captions
                }
                for item in video_caption_items(previous["captions"]):
                    key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if key not in caption_keys:
                        caption_keys.add(key)
                        captions.append(item)
                old_summary = str(previous["summary"] or "")
                summary_update = summary
                if old_summary and mode == "video_analysis":
                    new_content = "\n\n".join(added_analysis)[:1600]
                    summary_update = summary_prefix + new_content if new_content else ""
                summary_value = old_summary or summary_update
                if old_summary and summary_update and summary_update not in old_summary:
                    summary_value = f"{old_summary}\n{summary_update}"
                modes = {
                    part for part in str(previous["mode"] or "").split("+")
                    if part
                }
                modes.add(mode)
                merged_mode = (
                    VIDEO_MODE_MIXED
                    if {"video_analysis", "captions_only"} <= modes
                    else "+".join(sorted(modes)))
                conn.execute(
                    """
                    UPDATE video_watches SET
                        title=?, cid=?, mode=?, analysis=?, captions=?, summary=?,
                        source_context=?, raw_observation=?, session_id=?, watched_at=?, updated_at=?
                     WHERE platform=? AND source_object_id=? AND cid=? AND owner_key=? AND read_scope=?
                    """,
                    (
                        title, cid, merged_mode,
                        "\n\n".join(analyses),
                        json.dumps(captions, ensure_ascii=False) if captions else "",
                        summary_value,
                        source_context_json, raw_observation_json, session_id, watched_at, now_iso,
                        platform, object_id, cid, write_owner, read_scope_json,
                    ),
                )
        public_excerpt = ""
        if callable(review_public_excerpt) and review_public_excerpt(summary[:900]) is True:
            public_excerpt = summary[:900]
        if public_excerpt:
            self._sqlite.execute(
                "UPDATE video_watches SET public_excerpt=?, updated_at=? "
                "WHERE platform=? AND source_object_id=? AND cid=? AND owner_key=? AND read_scope=?",
                (public_excerpt, now_iso, platform, object_id, cid, write_owner, read_scope_json),
            )

        self._dependencies.resolve_explored_curiosity(summary)
        self._dependencies.record_curiosity(title, content, session_id)

        return True

    def get_public_video_watch(self, bvid: str, *, owner_key: str) -> str:
        """Reviewed active recollection; technical caches cannot revive forgotten events."""
        source = legacy_video_source()
        if not (self._sqlite.is_open and isinstance(bvid, str) and source.legacy_video_id
                and source.legacy_video_id(bvid, "")):
            return ""

        owners = self._video_read_owners(owner_key)
        owner_clause, owner_params = (
            self._video_owner_clause("owner_key", owners) if owners else ("0", [])
        )
        row = self._sqlite.fetchone(
            f"""
            SELECT public_excerpt FROM video_watches
             WHERE bvid = ? AND public_excerpt <> ''
               AND platform=?
               AND (read_scope = '' OR json_extract(read_scope, '$.kind') = 'self' OR ({owner_clause} AND owner_key <> ''))
             ORDER BY id DESC LIMIT 1
            """,
            (bvid, source.platform, *owner_params),
        )
        return str(row["public_excerpt"] or "")[:900] if row is not None else ""

    def lookup_video_watch(self, platform: str, source_object_id: str, *, owner_key: str,
                           cid: str | None = None) -> Optional[Dict[str, Any]]:
        owners = self._video_read_owners(owner_key)
        if not self._sqlite.is_open or not owners:
            return None
        platform = str(platform or "").strip().lower()
        object_id = str(source_object_id or "").strip()
        if not platform or not object_id or len(object_id) > 200:
            return None
        if cid is not None and not str(cid).strip():
            return None
        owner_clause, owner_params = self._video_owner_clause("owner_key", owners)
        params: list[Any] = [platform, object_id, *owner_params]
        query = ("SELECT id, watched_at, summary, title, cid, mode, analysis, captions, source_context, raw_observation "
                 f"FROM video_watches WHERE platform=? AND source_object_id=? AND {owner_clause}")
        if cid is not None:
            query += " AND cid=?"
            params.append(str(cid))
        query += " ORDER BY COALESCE(julianday(watched_at), 0) DESC, id DESC LIMIT 1"

        def _read_row(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
            row = conn.execute(query, params).fetchone()
            source = source_descriptor(platform)
            if row is None and source and source.legacy_video_url:
                legacy_params: list[Any] = [platform, object_id]
                legacy_query = (
                    "SELECT id, watched_at, summary, title, cid, mode, analysis, captions, source_context, raw_observation "
                    "FROM video_watches WHERE platform=? AND source_object_id=? AND owner_key='' "
                    "AND json_valid(read_scope) AND json_extract(read_scope, '$.kind')='self' "
                    "AND json_valid(source_context) AND json_extract(source_context, '$.kind')='self' "
                    "AND json_extract(source_context, '$.certainty')='observed' "
                    "AND json_extract(source_context, '$.visibility')='public' "
                    "AND json_extract(source_context, '$.ref')=?")
                legacy_params.append(source.legacy_video_url(object_id))
                if cid is not None:
                    legacy_query += " AND cid=?"
                    legacy_params.append(str(cid))
                legacy_query += " ORDER BY COALESCE(julianday(watched_at), 0) DESC, id DESC LIMIT 1"
                row = conn.execute(legacy_query, legacy_params).fetchone()
            return row

        row = self._sqlite.read(_read_row)
        if row is None:
            return None
        result: Dict[str, Any] = {
            "record_id": int(row["id"]), "watched_at": str(row["watched_at"] or ""),
            "summary": str(row["summary"] or ""),
        }
        if row["title"]:
            result["title"] = str(row["title"])
        if cid is not None:
            if "video_analysis" not in str(row["mode"] or "").split("+"):
                return None
            try:
                raw_observation = json.loads(row["raw_observation"] or "{}")
            except (ValueError, TypeError):
                raw_observation = {}
            if isinstance(raw_observation, dict) and raw_observation.get("_video_version_ambiguous") is True:
                return None
            analysis = str(row["analysis"] or "")
            if not analysis:
                return None
            try:
                captions = video_caption_items(row["captions"])
            except (ValueError, TypeError):
                captions = []
            try:
                source = json.loads(row["source_context"] or "{}")
            except (ValueError, TypeError):
                source = {}
            result.update({"analysis": analysis,
                           "captions": captions,
                           "source_context": source if isinstance(source, dict) else {}})
        return result

    def _video_read_owners(self, owner_key: Optional[str]) -> List[str]:
        owner = str(owner_key or "").strip()
        if not owner:
            return []
        return list(self._dependencies.read_owner_keys(owner))

    @staticmethod
    def _video_owner_clause(column: str, owners: List[str]) -> tuple[str, List[str]]:
        if len(owners) == 1:
            return f"{column} = ?", [owners[0]]
        return f"{column} IN (%s)" % ",".join("?" for _ in owners), list(owners)

    def list_video_watches(
        self,
        limit: int = 20,
        offset: int = 0,
        owner_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self._sqlite.is_open:
            return []
        query = (
            "SELECT id, bvid, platform, source_object_id, title, cid, mode, summary, watched_at, owner_key, public_excerpt "
            "FROM video_watches WHERE 1=1"
        )
        owners = self._video_read_owners(owner_key)
        if not owners:
            return []
        clause, params = self._video_owner_clause("owner_key", owners)
        query += " AND " + clause
        query += " ORDER BY watched_at DESC, id DESC"

        rows = self._sqlite.fetchall(query, params)
        rows = self._video_dedupe_rows(rows)
        start = max(0, int(offset))
        return [dict(r) for r in rows[start:start + max(1, int(limit))]]

    def search_video_watches(
        self,
        query: str,
        limit: int = 10,
        owner_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self._sqlite.is_open or not str(query or "").strip():
            return []
        owners = self._video_read_owners(owner_key)
        if not owners:
            return []
        term = query.strip()
        limit_val = max(1, int(limit))

        def _search_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
            clean_term = re.sub(r'["\'*]', '', term)
            owner_clause, owner_params = self._video_owner_clause("w.owner_key", owners)
            try:
                fts_query = (
                    "SELECT w.id, w.bvid, w.platform, w.source_object_id, w.title, w.cid, w.mode, w.summary, w.watched_at, w.owner_key "
                    "FROM video_fts f JOIN video_watches w ON w.id = f.rowid "
                    "WHERE video_fts MATCH ? "
                )
                params: List[Any] = [f'"{clean_term}"' if clean_term else term]
                fts_query += " AND " + owner_clause
                params.extend(owner_params)
                fts_query += " ORDER BY rank"
                rows = conn.execute(fts_query, params).fetchall()
                if rows:
                    rows = self._video_dedupe_rows(rows)
                    return [dict(r) for r in rows[:limit_val]]
            except sqlite3.OperationalError as exc:
                self._record_video_fts_failure(exc)

            like_query = (
                "SELECT id, bvid, platform, source_object_id, title, cid, mode, summary, watched_at, owner_key "
                "FROM video_watches WHERE (title LIKE ? OR summary LIKE ? OR bvid LIKE ?) "
            )
            like_owner_clause, like_owner_params = self._video_owner_clause("owner_key", owners)
            params = [f"%{term}%", f"%{term}%", f"%{term}%"]
            like_query += " AND " + like_owner_clause
            params.extend(like_owner_params)
            like_query += " ORDER BY watched_at DESC, id DESC"
            rows = conn.execute(like_query, params).fetchall()
            rows = self._video_dedupe_rows(rows)
            return [dict(r) for r in rows[:limit_val]]

        return self._sqlite.read(_search_rows)

    def get_video_watch_detail(
        self,
        bvid: str,
        owner_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch full video record on-demand (complete analysis, captions, raw observation).

        An empty owner never means "every owner": it refuses the read. A bound
        owner sees its own rows plus unowned (owner_key='') legacy
        self-scoped rows carrying complete observed public evidence; another
        specific owner's full analysis/captions/summary never crosses owners.
        """
        if not self._sqlite.is_open or not isinstance(bvid, str) or not bvid.strip():
            return None
        owners = self._video_read_owners(owner_key)
        if not owners:
            return None
        bv = bvid.strip()

        owner_clause, owner_params = self._video_owner_clause("owner_key", owners)
        source = legacy_video_source()
        legacy_clause, legacy_params = _unowned_legacy_watch(source)
        rows = self._sqlite.fetchall(
            f"SELECT * FROM video_watches WHERE bvid=? "
            f"AND ((platform IN (?, '') AND {owner_clause}) OR ({legacy_clause})) ORDER BY id ASC",
            (bv, source.platform, *owner_params, *legacy_params),
        )
        if not rows:
            return None
        analyses: List[str] = []
        captions: List[dict[str, object]] = []
        caption_keys = set()
        for row in rows:
            text = str(row["analysis"] or "").strip()
            if text and text not in analyses:
                analyses.append(text)
            for item in video_caption_items(row["captions"]):
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if key not in caption_keys:
                    caption_keys.add(key)
                    captions.append(item)

        def _watch_time(row: sqlite3.Row) -> float:
            try:
                return datetime.fromisoformat(str(row["watched_at"] or "")).timestamp()
            except (TypeError, ValueError, OverflowError):
                return 0.0

        latest_row = max(rows, key=lambda row: (_watch_time(row), int(row["id"])))
        merged = dict(latest_row)
        merged["analysis"] = "\n\n".join(analyses)
        merged["captions"] = (
            json.dumps(captions, ensure_ascii=False) if captions else "")
        for field in ("title", "cid", "summary", "source_context"):
            if str(merged.get(field) or "").strip():
                continue
            for row in reversed(rows):
                value = str(row[field] or "").strip()
                if value:
                    merged[field] = row[field]
                    break
        merged["watched_at"] = str(latest_row["watched_at"] or "")
        modes = set()
        for row in rows:
            modes.update(
                part for part in str(row["mode"] or "").split("+") if part)
        if len(modes) > 1:
            merged["mode"] = (
                VIDEO_MODE_MIXED
                if {"video_analysis", "captions_only"} <= modes
                else "+".join(sorted(modes)))
        return merged

    def tool_video_detail(self, args: Dict[str, Any], *, owner_key: str) -> dict[str, Any]:
        bvid = str(args.get("bvid") or "").strip()
        accepts = legacy_video_source().legacy_video_id
        if accepts is None or not accepts(bvid, ""):
            raise ValueError("bvid is required")
        row = self.get_video_watch_detail(bvid, owner_key)
        if row is None:
            return {"found": False, "bvid": bvid}
        try:
            captions = json.loads(row.get("captions") or "[]")
        except (TypeError, ValueError):
            captions = []
        source = {}
        try:
            source = json.loads(row.get("source_context") or "{}")
        except (TypeError, ValueError):
            source = {}
        return {
            "found": True,
            "bvid": bvid,
            "title": str(row.get("title") or ""),
            "summary": str(row.get("summary") or ""),
            "analysis": str(row.get("analysis") or ""),
            "captions": captions if isinstance(captions, list) else [],
            "watched_at": str(row.get("watched_at") or ""),
            "source": source if isinstance(source, dict) else {},
        }

    @staticmethod
    def _video_brief(summary: str) -> str:
        text = str(summary or "")
        if "内容理解未经独立核实：" in text:
            text = text.split("内容理解未经独立核实：", 1)[1]
        elif "未观看画面：" in text:
            text = text.split("未观看画面：", 1)[1]
        first = re.split(r"[。！？\n]", text.strip())[0].strip()
        return (first or text.strip())[:80]

    def _video_search_rows(self, query: str, *, read_owner: str,
                           limit: int = 3) -> List[Dict[str, Any]]:
        if not self._sqlite.is_open:
            return []
        text = str(query or "").strip()
        if not text:
            return []
        legacy_source = legacy_video_source()
        find_legacy_id = legacy_source.legacy_video_in_text
        legacy_id = find_legacy_id(text) if find_legacy_id else ""
        owners = self._video_read_owners(read_owner)
        if not owners:
            return []
        owner_clause, owner_params = self._video_owner_clause("owner_key", owners)
        legacy_clause, legacy_params = _unowned_legacy_watch(legacy_source)
        rows = self._sqlite.fetchall(
            f"""SELECT id, bvid, platform, source_object_id, title,
                       CASE WHEN owner_key = '' THEN public_excerpt ELSE summary END AS summary,
                       watched_at, source_context, mode, owner_key, read_scope, public_excerpt
                  FROM video_watches WHERE ({owner_clause} OR ({legacy_clause}))
                 ORDER BY watched_at DESC LIMIT 200""",
            (*owner_params, *legacy_params),
        )
        from ..infra.textutil import _tokens
        query_tokens = set(_tokens(text))
        hits: List[tuple] = []
        for row in rows:
            origin = watch_row_origin(row["source_context"], row["platform"], row["source_object_id"],
                                      row["bvid"])
            exact_id = bool(origin and origin.object_id and (
                origin.object_id == legacy_id if legacy_id else origin.object_id in text))
            if legacy_id or exact_id:
                if not exact_id:
                    continue
                score = 1.0
            else:
                haystack = f"{row['title'] or ''} {row['summary'] or ''}"
                row_tokens = set(_tokens(haystack))
                shared = len(query_tokens & row_tokens)
                if shared < 2 or not query_tokens:
                    continue
                score = shared / len(query_tokens)
            hits.append((score, dict(row)))
        hits.sort(key=lambda entry: (entry[0], entry[1]["watched_at"]), reverse=True)
        results = []
        seen = set()
        for _score, row in hits:
            origin = watch_row_origin(row["source_context"], row["platform"], row["source_object_id"],
                                      row["bvid"])
            if origin is not None and origin.object_id:
                if origin.key in seen:
                    continue
                seen.add(origin.key)
            results.append(dict(row))
            if len(results) >= max(1, int(limit)):
                break
        return results

    def video_memory_items(self, query: str, *, read_owner: str,
                           limit: int = 3) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for row in self._video_search_rows(query, read_owner=read_owner, limit=limit):
            title = str(row.get("title") or "").strip()
            brief = self._video_brief(str(row.get("summary") or ""))
            experience = video_playback_label(str(row.get("mode") or "")) or "观看了"
            body = f"{experience}：《{title}》" if title else ""
            body = "：".join(part for part in (body, brief) if part)
            source = parse_source(row.get("source_context")) or {}
            origin = watch_row_origin(row.get("source_context"), row.get("platform"),
                                      row.get("source_object_id"), row.get("bvid"))
            if origin is not None:
                source.update(kind="video", platform=origin.platform,
                              source_object_id=origin.object_id, ref=origin.ref)
            items.append({
                "id": "video:%d" % int(row["id"]),
                "raw": body,
                "summary": body,
                "topic": "视频",
                "importance": 0.7,
                "ts": str(row.get("watched_at") or ""),
                "created_at": str(row.get("watched_at") or ""),
                "session_id": "",
                "experience_subject": "sanae",
                "fact_subject": "sanae",
                "attribution": "self",
                "source_context": source if isinstance(source, dict) else {},
                "source_table": "video",
                "bvid": str(row.get("bvid") or ""),
                "occurrences": [],
                "keywords": "",
                "asked": 0,
                "asked_at": "",
                "tier": "",
                "summary_source": "",
                "relevance": 1.0,
                "semantic": 0.0,
                "embedding_real": False,
            })
        return items

    def video_public_rows(self, query: str, *, read_owner: str,
                          limit: int = 3) -> List[Dict[str, Any]]:
        if not self._sqlite.is_open:
            return []
        text = str(query or "").strip()
        if not text:
            return []
        legacy_source = legacy_video_source()
        find_legacy_id = legacy_source.legacy_video_in_text
        legacy_id = find_legacy_id(text) if find_legacy_id else ""
        owners = self._video_read_owners(read_owner)
        owner_values = ["", *owners]
        owner_placeholders = ",".join("?" for _ in owner_values)
        rows = self._sqlite.fetchall(
            f"""SELECT id, bvid, public_excerpt, watched_at, source_context, platform, source_object_id
                 FROM video_watches
                WHERE public_excerpt<>''
                  AND (owner_key IN ({owner_placeholders})
                       OR json_extract(NULLIF(read_scope, ''), '$.kind')='self')
                ORDER BY watched_at DESC LIMIT 200""",
            owner_values,
        )
        from ..infra.textutil import _tokens
        query_tokens = set(_tokens(text))
        hits: List[tuple] = []
        seen = set()
        for row in rows:
            excerpt = str(row["public_excerpt"] or "").strip()
            if not excerpt:
                continue
            source = parse_source(row["source_context"]) or {}
            origin = watch_row_origin(source, row["platform"], row["source_object_id"], row["bvid"])
            if not origin or origin.key in seen or not public_video_origin(source, origin):
                continue
            exact_id = bool(origin.object_id and (
                origin.object_id == legacy_id if legacy_id else origin.object_id in text))
            if legacy_id or exact_id:
                if not exact_id:
                    continue
                score = 1.0
            else:
                shared = len(query_tokens & set(_tokens(excerpt)))
                if shared < 2 or not query_tokens:
                    continue
                score = shared / len(query_tokens)
            seen.add(origin.key)
            source.update(kind="video", platform=origin.platform,
                          source_object_id=origin.object_id, ref=origin.ref)
            hits.append((score, {
                "id": int(row["id"]),
                "public_excerpt": excerpt[:900],
                "created_at": str(row["watched_at"] or ""),
                "source_context": json.dumps(source, ensure_ascii=False),
                "source_table": "video",
            }))
        hits.sort(key=lambda entry: (entry[0], entry[1]["created_at"]), reverse=True)
        return [item for _score, item in hits[:max(1, int(limit))]]
