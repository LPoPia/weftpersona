from __future__ import annotations

from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3
import re

try:
    from .self_utterances import _format_experience_rows, _private_receipt_route, _self_utterance_rows
    from .source_context import SELF_SUBJECT
except ImportError:
    from self_utterances import _format_experience_rows, _private_receipt_route, _self_utterance_rows
    from source_context import SELF_SUBJECT

def collect_life_experiences(*, home, data_dir, owner_key, session_id, since, through,
                             allow_trusted_cron=False) -> str:
    """Read facts in (since, through], scoped to the delivered private turn's profile."""
    from hermes_time import now

    route = _private_receipt_route(owner_key, session_id, home)
    cron_home = _trusted_cron_home(home, owner_key, session_id) if allow_trusted_cron else None
    if route is None and cron_home is None:
        return ""
    lower = through - 24 * 60 * 60 if since is None else since
    if lower >= through:
        return ""
    rows = (_self_utterance_rows(owner_key, session_id, route, since=lower, through=through, strict=True)
            if route is not None else [])
    rows.extend(_watch_rows(Path(data_dir).resolve() / "sanae.db", owner_key, lower, through))
    rows.extend(_cron_rows(route[0] if route else cron_home, owner_key, lower, through))
    rows.extend(_free_think_rows(Path(home), lower, through))
    return "\n".join(_format_experience_rows(rows, now().tzinfo))


def collect_recent_activities(*, home, data_dir, owner_key, session_id, since, through,
                              allow_trusted_cron=False) -> str:
    """Read facts in (since, through] for the next private turn: only what she did alone.

    Outbound words (scheduled proactive messages, Bilibili replies) belong to the receipt block
    `[我刚才在别处说过]` and are never repeated here.
    """
    from hermes_time import now

    route = _private_receipt_route(owner_key, session_id, home)
    cron_home = _trusted_cron_home(home, owner_key, session_id) if allow_trusted_cron else None
    if route is None and cron_home is None:
        return ""
    lower = through - 24 * 60 * 60 if since is None else since
    if lower >= through:
        return ""
    rows = _watch_rows(Path(data_dir).resolve() / "sanae.db", owner_key, lower, through)
    rows.extend(_cron_rows(route[0] if route else cron_home, owner_key, lower, through))
    rows.extend(_free_think_rows(Path(home), lower, through))
    return "\n".join(_format_experience_rows(rows, now().tzinfo))


def _trusted_cron_home(home, owner_key, session_id):
    from cron.jobs import get_job
    from cron.proactive import owner_of, trusted_personal_action
    from gateway.session_cron import current_execution
    from hermes_constants import get_hermes_home
    from agent.delegation_context import is_delegated_child_context
    from agent.interaction_context import is_public_interaction

    binding = current_execution()
    if (not binding or not owner_key
            or is_delegated_child_context() or is_public_interaction()
            or Path(home).resolve() != get_hermes_home().resolve()):
        return None
    _, _, job_id, execution_id = binding
    if not trusted_personal_action(session_id, f"cron:{job_id}:{execution_id}"):
        return None
    job = get_job(job_id)
    if owner_of(job) != owner_key:
        return None
    return Path(home).resolve()


def _free_think_rows(home, since, through):
    try:
        path = home / "sanae_free_think.md"
        stamp = path.stat().st_mtime
        if not ((since is None or stamp > since) and stamp <= through):
            return []
        blocks = re.split(r"(?m)^#{1,3}\s+[^\n]*$|^---+\s*$", path.read_text(encoding="utf-8", errors="replace"))
        for block in reversed(blocks):
            for paragraph in reversed(re.split(r"\n\s*\n", block)):
                lines = []
                for line in paragraph.splitlines():
                    line = line.strip()
                    bare = line.strip("*_` ")
                    if (not bare or re.fullmatch(r"[—–]+\s*\S{1,20}", bare)
                            or re.fullmatch(r"[\d\s:/T+Z.年月日时分秒()（）\-]+", bare)
                            or re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?:\s*[（(].*[）)])?$", bare)):
                        continue
                    lines.append(line)
                text = " ".join(lines)[:280]
                if text:
                    return [(stamp, "自由放空", text)]
    except OSError:
        pass
    return []


def _stamp(value):
    try:
        date = datetime.fromisoformat(value)
        return date.timestamp() if date.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def _watch_rows(path, owner_key, since, through):
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_occurrences'").fetchone():
            return []
        rows = conn.execute("""
            SELECT o.observed_at, o.raw
              FROM events AS e JOIN event_occurrences AS o ON o.event_id=e.id
             WHERE e.owner_key IN ('', ?) AND e.experience_subject=? AND e.fact_subject=?
               AND e.blacklist=0 AND e.expired_at IS NULL
               AND json_extract(NULLIF(e.read_scope, ''), '$.kind')='self' AND e.source_context<>''
               AND json_extract(e.source_context, '$.kind')='self'
               AND json_extract(e.source_context, '$.platform')='bilibili'
               AND json_extract(e.source_context, '$.certainty')='observed'
               AND json_extract(e.source_context, '$.visibility')='public'
               AND json_extract(e.source_context, '$.ref') LIKE 'https://www.bilibili.com/video/BV%/'
               AND (o.raw LIKE '早苗在B站收到该视频的抽样画面分析，音轨未核验：%'
                 OR o.raw LIKE '早苗在B站阅读了该视频的字幕，未观看画面：%')
               AND CAST(strftime('%s', o.observed_at) AS INTEGER) BETWEEN ? AND ?
        """, (owner_key, SELF_SUBJECT, SELF_SUBJECT, int(since), int(through) + 1)).fetchall()
    return [(stamp, "B站观看", text) for observed_at, text in rows
            if (stamp := _stamp(observed_at)) is not None and since < stamp <= through]


def _cron_rows(home, owner_key, since, through):
    from cron.executions import _execution_path
    from gateway.response_filters import is_autonomous_silence_response
    from hermes_state_keys import CRON_PRINCIPAL_ID, cron_request_id
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_terminal import RESULT_PREFIX

    execution_path, state_path = _execution_path(home), home / "state.db"
    if not (execution_path.is_file() and state_path.is_file()):
        return []
    with closing(sqlite3.connect(execution_path.as_uri() + "?mode=ro", uri=True)) as executions:
        completed = executions.execute("""
            SELECT id, job_id, finished_at, delivery_outcome FROM executions
             WHERE status='completed' AND delivery_outcome IN ('suppressed','not_configured','delivered')
               AND CAST(strftime('%s', finished_at) AS INTEGER) BETWEEN ? AND ?
        """, (int(since), int(through) + 1)).fetchall()
    rows = []
    with closing(sqlite3.connect(state_path.as_uri() + "?mode=ro", uri=True)) as state:
        for execution_id, job_id, finished_at, outcome in completed:
            stamp = _stamp(finished_at)
            if stamp is None or not since < stamp <= through:
                continue
            action = state.execute("""
                SELECT a.target_session_id,
                       json_extract(json_extract(p.value, '$.policy.request_json'), '$.cron_job.name'),
                       json_extract(json_extract(p.value, '$.policy.request_json'), '$.cron_job.memory_read_session_id'),
                       CASE WHEN ?='suppressed'
                            AND json_extract(json_extract(p.value, '$.policy.request_json'), '$.cron_job.deliver')='local'
                            AND json_extract(json_extract(p.value, '$.policy.request_json'),
                                             '$.cron_job.memory_read_owner')=?
                            THEN json_extract(r.value, '$.result.cron_result[2]') END
                  FROM session_admissions AS a
                  JOIN state_meta AS p ON p.key=? || a.target_session_id
                  JOIN state_meta AS r ON r.key=? || a.admission_id
                 WHERE a.principal_id=? AND a.request_id=? AND a.status='terminal' AND a.outcome='completed'
                   AND json_extract(p.value, '$.policy.source')='cron'
                   AND json_extract(json_extract(p.value, '$.policy.request_json'), '$.cron_job.activity_class')='action'
                   AND COALESCE(json_extract(json_extract(p.value, '$.policy.request_json'),
                                             '$.cron_job.memory_read_owner'), '') IN ('', ?)
                   AND json_extract(r.value, '$.result.cron_result[0]')=1
            """, (outcome, owner_key, POLICY_PREFIX, RESULT_PREFIX, CRON_PRINCIPAL_ID,
                  cron_request_id(job_id, execution_id), owner_key)).fetchone()
            if action is None:
                continue
            run_session, name, origin_session, result = action
            text = f"定时活动完成：{name or '未命名活动'}"
            if isinstance(result, str) and result.strip() and not is_autonomous_silence_response(result):
                text += f"。活动结果：{result}"
            rows.append((stamp, "定时自主活动", text))
    return rows
