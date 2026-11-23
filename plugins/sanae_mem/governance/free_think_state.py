"""Pure free-think state, metadata, and read-only reflection queries."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from ..infra import host_bridge
from ..infra.textutil import _bool, _load_plugin_config
from ..storage.sqlite_facade import connect_readonly
from .account_binding import maintainer_anchor
from .identity_config import daily_memory_identity

_SHANGHAI = "Asia/Shanghai"
_FREE_THINK_PREFIX = "ViveMind: free think "
_FREE_THINK_SCHEDULE = "0 8,16,22 * * *"
_DIARY_TOOL = "sanae_free_think_diary"
_FREE_THINK_TOOLSETS = ["web", "memory", "session_search", "bilibili_explore", _DIARY_TOOL, "no_mcp"]
_LEGACY_FREE_THINK_PROMPT = (
    '这段时间只属于你自己。顺着刚才留在心里的事随便想一会儿，不必回复任何人。'
    '确实想过、想留下的一点，用第一人称写在 JSON 对象的 reflection 字段里。'
    '想到的事只当作想法，别写成已经做过；如果没有想留下的内容，只回 [SILENT]。'
)
_FREE_THINK_PROMPT = (
    "这段空闲时间你自己安排，想怎么过都行。可以随手逛逛 B 站，也可以顺着最近想弄明白的事"
    "上网查一查，或者干脆什么都不做。有些念头只想留给自己，就写进日记。"
    "真有话想私下跟对方说，最后的回复就是你想发的那句话。没有要发的话，最后只回 [SILENT]。"
    "这一轮里实际做过的事才算经历，只是想过的依然是念头。"
)


def _schedule_owner(config: dict) -> str:
    return maintainer_anchor(config)


def _free_think_enabled(config: dict) -> bool:
    return _bool(config.get("free_think"), False) and not _bool(config.get("shadow"), False)


def _owned_free_think_job(job: dict, owner: str | None = None) -> bool:
    name = str(job.get("name") or "")
    return (job.get("prompt") in (_LEGACY_FREE_THINK_PROMPT, _FREE_THINK_PROMPT) and
            (name == _FREE_THINK_PREFIX + owner if owner else name.startswith(_FREE_THINK_PREFIX)))


def _private_free_think_target(config: dict) -> str:
    platform, primary, alternate = daily_memory_identity(config)
    recipient = primary or alternate
    if (not platform or not recipient or any(c in recipient for c in ",:#")
            or any(c.isspace() for c in platform + recipient)
            or not platform.replace("_", "").isalnum()):
        return ""
    return f"{platform}:{recipient}"


def _private_free_think_job(job, owner: str, target: str) -> bool:
    tools = job.get("enabled_toolsets")
    return (
        bool(target) and _owned_free_think_job(job, owner)
        and job.get("bound_owner") == owner and job.get("activity_class") == "action"
        and not job.get("no_agent") and not job.get("script")
        and job.get("deliver") == target and job.get("failure_deliver") == "local"
        and isinstance(tools, list) and "no_mcp" in tools
        and all(name in _FREE_THINK_TOOLSETS for name in tools)
    )


def current_free_think(config: dict, session_id: str):
    """Authorize only the canonical private run installed for this owner."""
    if not _free_think_enabled(config) or not session_id:
        return None
    owner = _schedule_owner(config)
    env = host_bridge.verified_cron_envelope()
    if (not owner or env is None or not env.verified or env.cancelled or not env.is_runnable
            or env.bound_owner != owner or not env.job_id or not env.execution_id
            or str(env.session_id or "") != session_id
            or host_bridge.is_test_isolated_turn(session_id)):
        return None
    job = next((j for j in host_bridge.list_plugin_cron_jobs(include_disabled=True)
                if j.get("id") == env.job_id), None)
    target = _private_free_think_target(config)
    if job and job.get("enabled") and _private_free_think_job(job, owner, target):
        return owner, env
    return None


def free_think_can_deliver(job, config) -> bool:
    owner = _schedule_owner(config)
    target = _private_free_think_target(config)
    if (not owner or not _free_think_enabled(config) or not job.get("id") or not job.get("execution_id")
            or not _private_free_think_job(job, owner, target)):
        return False
    return any(
        live.get("id") == job["id"] and live.get("enabled") and not live.get("paused")
        and _private_free_think_job(live, owner, target)
        for live in host_bridge.list_plugin_cron_jobs(include_disabled=True)
    )


def completed_free_think_rows(home: Path, owner: object, since: float, through: float, *,
                              data_dir: Path | None = None, execution_id: str = ""):
    """Read historical private reflections saved under the previous schedule."""
    directory = Path(data_dir if data_dir is not None else
                     (_load_plugin_config().get("data_dir") or Path(home) / "sanae-mem"))
    path = directory.expanduser().resolve() / "sanae.db"
    owners = (
        tuple(sorted({str(k).strip() for k in owner if str(k).strip()}))
        if isinstance(owner, (list, tuple, set, frozenset))
        else ((str(owner).strip(),) if str(owner or "").strip() else ())
    )
    if not owners or not path.is_file():
        return []
    clause = "owner_key=?" if len(owners) == 1 else "owner_key IN (%s)" % ",".join("?" for _ in owners)
    exec_id = str(execution_id or "").strip()
    params: list[object] = [*owners, since, through]
    exec_clause = ""
    if exec_id:
        exec_clause = " AND execution_id=?"
        params.append(exec_id)
    try:
        with closing(connect_readonly(path)) as conn:
            rows = conn.execute(
                f"SELECT finished_at, reflection FROM free_think_reflections "
                f"WHERE {clause} AND finished_at>? AND finished_at<=?{exec_clause} "
                f"ORDER BY finished_at DESC",
                params,
            ).fetchall()
        return [(float(ts), "自由放空", str(text)) for ts, text in rows]
    except sqlite3.OperationalError:
        return []
