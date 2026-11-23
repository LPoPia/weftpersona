from __future__ import annotations

from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3
import re

from ..infra.defaults import DEFAULT_PERSONA_NAME
from .self_utterances import (
    _format_experience_rows, _owner_sql, _private_receipt_route, _readable_owners,
    _self_utterance_rows,
)
from ..infra.event_fields import SELF_SUBJECT, parse_source
from ..infra.textutil import _load_plugin_config
from ..domains.video_memory import read_public_video_watches
from ..infra.video_record_values import video_playback_label
from ..infra.source_descriptors import (
    VideoOrigin, legacy_playback_mode, public_video_origin, video_origin, watch_row_origin,
)
from ..storage.sqlite_facade import connect_readonly


def collect_life_experiences(*, home, data_dir, owner_key, session_id, since, through,
                             allow_trusted_cron=False, private_experience=True,
                             read_owners=()) -> str:
    """Read facts in (since, through], scoped to the delivered private turn's profile.

    ``private_experience=False`` is the person-attributed path for someone other
    than the configured owner: it never reads her private cron
    activity. Only proved public videos (title and one sentence of
    gist) and her own delivered words to this same counterpart stay visible.
    """
    route = _private_receipt_route(owner_key, session_id, home)
    cron_home = _trusted_cron_home(home, owner_key, session_id) if route is None and allow_trusted_cron else None
    if route is None and cron_home is None:
        return ""
    lower = through - 24 * 60 * 60 if since is None else since
    if lower >= through:
        return ""
    resolved_home = route[0] if route else cron_home
    owners = _readable_owners(read_owners or owner_key, data_dir=data_dir, home=resolved_home)
    rows = (_self_utterance_rows(owner_key, route, data_dir=data_dir, since=lower, through=through,
                                 strict=True, read_owners=owners)
            if route is not None else [])
    rows.extend(_watch_rows(Path(data_dir).resolve() / "sanae.db", owners, lower, through,
                            public_brief=not private_experience))
    if private_experience:
        rows.extend(_cron_rows(resolved_home, owners, lower, through, data_dir=data_dir))
    return "\n".join(_format_experience_rows(rows, datetime.now().astimezone().tzinfo))


def collect_recent_activities(*, home, data_dir, owner_key, session_id, since, through,
                              allow_trusted_cron=False, private_experience=True,
                              read_owners=()) -> str:
    """Read facts in (since, through] for the next private turn: only what she did alone.

    Outbound words (scheduled proactive messages, source replies) belong to the receipt block
    `[我刚才在别处说过]` and are never repeated here. The caller's visibility decides what is
    read at all: an outsider gets proved public videos only, never her private cron activity.
    """
    route = _private_receipt_route(owner_key, session_id, home)
    cron_home = _trusted_cron_home(home, owner_key, session_id) if route is None and allow_trusted_cron else None
    if route is None and cron_home is None:
        return ""
    lower = through - 24 * 60 * 60 if since is None else since
    if lower >= through:
        return ""
    resolved_home = route[0] if route else cron_home
    owners = _readable_owners(read_owners or owner_key, data_dir=data_dir, home=resolved_home)
    rows = _watch_rows(Path(data_dir).resolve() / "sanae.db", owners, lower, through,
                       public_brief=not private_experience)
    if private_experience:
        rows.extend(_cron_rows(resolved_home, owners, lower, through, data_dir=data_dir))
    return "\n".join(_format_experience_rows(rows, datetime.now().astimezone().tzinfo))


def _verified_public_video(source_context, *, row_owner="", owner_key="") -> bool:
    source = parse_source(source_context) or {}
    return (public_video_origin(source, video_origin(source))
            or (bool(owner_key) and str(row_owner or "") == str(owner_key)))


def _trusted_cron_home(home, owner_key, session_id):
    from ..infra.host_bridge import get_hermes_home
    from ..governance.free_think_state import current_free_think
    from ..infra.textutil import _load_plugin_config

    active = get_hermes_home().resolve()
    current = current_free_think(_load_plugin_config(), session_id)
    return active if current and current[0] == owner_key and Path(home).resolve() == active else None


def _stamp(value):
    try:
        date = datetime.fromisoformat(value)
        return date.timestamp() if date.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def _extract_brief_summary(summary: str) -> str:
    if not summary:
        return ""
    if "内容理解未经独立核实：" in summary:
        part = summary.split("内容理解未经独立核实：", 1)[1].strip()
    elif "未观看画面：" in summary:
        part = summary.split("未观看画面：", 1)[1].strip()
    else:
        part = summary.strip()
    first_sentence = re.split(r"[。！？\n]", part)[0].strip()
    return first_sentence[:80] if first_sentence else part[:80]


def _configured_persona_name() -> str:
    try:
        cfg = _load_plugin_config()
    except Exception:
        cfg = {}
    name = str((cfg or {}).get("persona_name") or "").strip() if isinstance(cfg, dict) else ""
    return name or DEFAULT_PERSONA_NAME


def _watch_text(origin: VideoOrigin, title: str, mode: str, brief: str, *, actor: str) -> str:
    identity = " · ".join(dict.fromkeys(part for part in (origin.object_id, origin.ref) if part))
    experience = video_playback_label(mode)
    lead = f"{actor}在{origin.label}{experience}：" if experience else ""
    text = f"{lead}《{title}》" + (f"（{identity}）" if identity else "")
    return f"{text}。{brief}" if brief else text


def _watch_rows(path: Path, owners: tuple[str, ...], since: float, through: float,
                *, public_brief: bool) -> list[tuple[float, str, str]]:
    if not path.is_file() or not owners:
        return []
    vw_clause, vw_params = _owner_sql("owner_key", owners)
    ev_clause, ev_params = _owner_sql("e.owner_key", owners)
    actor = _configured_persona_name()
    rows: list[tuple[float, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(stamp: float | None, origin: VideoOrigin, title: str, mode: str, brief: str) -> None:
        if not title or stamp is None or not since < stamp <= through or origin.key in seen:
            return
        seen.add(origin.key)
        label = origin.label + ("视频" if public_brief else "观看")
        rows.append((stamp, label, _watch_text(origin, title, mode, brief, actor=actor)))

    with closing(connect_readonly(path)) as conn:
        has_video_watches = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='video_watches'"
        ).fetchone() is not None
        if has_video_watches:
            db_rows = conn.execute(f"""
                SELECT watched_at, title, summary, bvid, mode, source_context,
                       platform, source_object_id, public_excerpt,
                       CASE WHEN json_valid(raw_observation)
                            THEN json_extract(raw_observation, '$.platform') END,
                       CASE WHEN json_valid(raw_observation)
                            THEN json_extract(raw_observation, '$.source_object_id') END,
                       CASE WHEN json_valid(raw_observation)
                            THEN json_extract(raw_observation, '$.mode') END
                  FROM video_watches
                 WHERE {vw_clause}
                   AND CASE WHEN read_scope = '' THEN 1 WHEN json_valid(read_scope)
                       THEN json_extract(read_scope, '$.kind') IN ('self', 'owner')
                       ELSE 0 END
                   AND CAST(strftime('%s', watched_at) AS INTEGER) BETWEEN ? AND ?
                 ORDER BY watched_at ASC
            """, (*vw_params, int(since), int(through) + 1))
            for (watched_at, title, summary, old_id, mode, source_json, platform, object_id,
                 excerpt, raw_platform, raw_id, raw_mode) in db_rows:
                source = parse_source(source_json) or {}
                origin = watch_row_origin(source_json, platform or raw_platform, object_id or raw_id, old_id)
                if origin is None or (public_brief and not public_video_origin(source, origin)):
                    continue
                body = excerpt if public_brief else summary
                add(_stamp(watched_at), origin, title, mode or raw_mode or "",
                    _extract_brief_summary(body))

        if not public_brief and int(conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('events','event_occurrences')"
        ).fetchone()[0]) == 2:
            legacy_rows = conn.execute(f"""
                SELECT o.observed_at, o.raw, e.source_context
                  FROM events AS e JOIN event_occurrences AS o ON o.event_id=e.id
                 WHERE {ev_clause} AND e.experience_subject=? AND e.fact_subject=?
                   AND e.blacklist=0 AND e.expired_at IS NULL AND e.source_context<>''
                   AND CASE WHEN json_valid(e.read_scope)
                            THEN json_extract(e.read_scope, '$.kind') = 'self' ELSE 0 END
                   AND CAST(strftime('%s', o.observed_at) AS INTEGER) BETWEEN ? AND ?
            """, (*ev_params, SELF_SUBJECT, SELF_SUBJECT, int(since), int(through) + 1))
            for observed_at, raw, source_json in legacy_rows:
                source = parse_source(source_json) or {}
                origin = video_origin(source)
                stamp = _stamp(observed_at)
                if (origin is not None and source.get("kind") == "self" and public_video_origin(source, origin)
                        and stamp is not None and since < stamp <= through and legacy_playback_mode(origin, raw)):
                    rows.append((stamp, origin.label + "观看", raw))

        if public_brief and has_video_watches:
            for row in read_public_video_watches(conn, owners=owners, since=since, through=through):
                add(_stamp(row.watched_at), row.origin, row.title, row.mode,
                    _extract_brief_summary(row.public_excerpt))
    return rows


def _cron_rows(home, owner_key, since, through, *, data_dir=None):
    from ..governance.free_think_state import completed_free_think_rows

    return completed_free_think_rows(Path(home), owner_key, since, through, data_dir=data_dir)
