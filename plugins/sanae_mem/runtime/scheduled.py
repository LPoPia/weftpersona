"""Install and run ViveMind's private tasks through the original Hermes scheduler."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from sanae_mem.governance.runtime_registry import (
    reconcile_registered_schedule as reconcile_registered_schedule,
    register_schedule as register_schedule,
    resolve_active_provider,
)

from ..infra.atomic_io import atomic_write_text, file_lock
from ..governance.free_think_state import (
    _DIARY_TOOL as _DIARY_TOOL,
    _FREE_THINK_PREFIX as _FREE_THINK_PREFIX,
    _FREE_THINK_PROMPT as _FREE_THINK_PROMPT,
    _FREE_THINK_SCHEDULE as _FREE_THINK_SCHEDULE,
    _FREE_THINK_TOOLSETS as _FREE_THINK_TOOLSETS,
    _LEGACY_FREE_THINK_PROMPT as _LEGACY_FREE_THINK_PROMPT,
    _SHANGHAI as _SHANGHAI,
    _free_think_enabled as _free_think_enabled,
    _owned_free_think_job as _owned_free_think_job,
    _private_free_think_job as _private_free_think_job,
    _private_free_think_target as _private_free_think_target,
    _schedule_owner as _schedule_owner,
    completed_free_think_rows as completed_free_think_rows,
    current_free_think as current_free_think,
    free_think_can_deliver as free_think_can_deliver,
)
from ..governance.source_context import daily_profile_key
from ..infra.textutil import _bool, _owner_key, _load_plugin_config
from ..infra import host_bridge
from ..infra.host_bridge import CronCompletion, PluginRegistration, get_hermes_home

if TYPE_CHECKING:
    from .provider import SanaeMemProvider

logger = logging.getLogger(__name__)

_NIGHTLY_NAME = "ViveMind: nightly journal"
_NIGHTLY_SCRIPT = "vivemind-nightly.py"
_PAUSED_BY_PLUGIN = "Disabled in ViveMind configuration"


def _sync_job(*, enabled, name, belongs_to_plugin, refresh=False, deliver="local", **settings):
    if refresh and enabled:
        named = [j for j in host_bridge.list_plugin_cron_jobs(include_disabled=True) if j.get("name") == name]
        if named and not any(belongs_to_plugin(j) for j in named):
            raise RuntimeError(f"ViveMind cron job name is already occupied: {name}")
    return host_bridge.sync_plugin_cron_job(
        name=name, enabled=enabled, match_fn=belongs_to_plugin, refresh=refresh,
        paused_reason=_PAUSED_BY_PLUGIN, timezone=_SHANGHAI, deliver=deliver,
        failure_deliver="local", **settings,
    )


def _nightly_launcher(hermes_source: Path) -> str:
    return f"""# Installed by ViveMind; implementation lives in the memory provider.
import os
import sys
from pathlib import Path

os.environ["HERMES_HOME"] = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, {str(hermes_source)!r})
from plugins.memory import load_memory_provider

provider = load_memory_provider("sanae_mem")
if provider is None:
    raise SystemExit("ViveMind nightly journal: memory provider unavailable")
result = provider.run_installed_nightly_journal()
if not result.get("ok"):
    raise SystemExit("ViveMind nightly journal: " + str(result.get("reason", "failed")))
print("ViveMind nightly journal: complete")
"""


def sync_scheduled_jobs(config: dict) -> None:
    from ..infra.host_bridge import get_hermes_home

    home = get_hermes_home()
    launcher = home / "scripts" / _NIGHTLY_SCRIPT
    owner = _schedule_owner(config)
    target = _private_free_think_target(config)
    nightly = (_bool(config.get("nightly_journal"), False)
               and not _bool(config.get("shadow"), False) and bool(owner))
    free_think = bool(owner and target and _free_think_enabled(config))
    if not (nightly or free_think or launcher.is_file() or (home / "cron").is_dir()):
        return

    source_root = host_bridge.hermes_source_root()
    if source_root is None:
        return
    with file_lock(home / "sanae-mem" / "scheduled-jobs.lock"):
        if nightly:
            launcher.parent.mkdir(parents=True, exist_ok=True)
            expected = _nightly_launcher(source_root)
            if not launcher.exists() and not launcher.is_symlink():
                atomic_write_text(launcher, expected)
            elif launcher.read_text(encoding="utf-8") != expected:
                raise RuntimeError("ViveMind nightly launcher path is already occupied")
        _sync_job(
            enabled=nightly, name=_NIGHTLY_NAME,
            belongs_to_plugin=lambda job: job.get("no_agent") and job.get("script") == _NIGHTLY_SCRIPT,
            prompt=None, script=_NIGHTLY_SCRIPT, no_agent=True, schedule="0 23 * * *",
        )
        for job in host_bridge.list_plugin_cron_jobs(include_disabled=True):
            if (_owned_free_think_job(job) and job.get("enabled")
                    and (not free_think or not _owned_free_think_job(job, owner))):
                host_bridge.pause_plugin_cron_job(job["id"], reason=_PAUSED_BY_PLUGIN)
        if owner:
            _sync_job(
                enabled=free_think, name=_FREE_THINK_PREFIX + owner,
                belongs_to_plugin=lambda job: _owned_free_think_job(job, owner),
                refresh=True, prompt=_FREE_THINK_PROMPT, no_agent=False, script=None,
                deliver=target or "local", bound_owner=owner, activity_class="action",
                enabled_toolsets=_FREE_THINK_TOOLSETS, schedule=_FREE_THINK_SCHEDULE,
            )


def run_installed_nightly_journal(provider: SanaeMemProvider, *, now: datetime | None = None) -> dict:
    home = get_hermes_home().resolve()
    config = provider.runtime_config(home)
    if _bool(config.get("shadow"), False):
        return {"ok": False, "reason": "shadow"}
    if not _bool(config.get("nightly_journal"), False):
        return {"ok": False, "reason": "disabled"}
    session_id = "vivemind:nightly"
    with host_bridge.turn_envelope_scope(platform="cron", session_id=session_id):
        try:
            provider.initialize(session_id, agent_context="cron", platform="cron",
                                hermes_home=home, hook_worker=True)
            return provider.run_nightly_journal(now=now)
        finally:
            provider.shutdown()


def _execution_diary_texts(home: Path, owner: str, execution_id: str) -> list[str]:
    diary = home / "diary"
    if not diary.is_dir() or diary.is_symlink() or not owner or not execution_id:
        return []
    entries: list[str] = []
    for path in sorted(diary.glob("free-think-*.md")):
        if path.is_symlink() or not path.is_file():
            continue
        expected_hash = path.stem.removeprefix("free-think-")
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        body = raw.split("\n\n", 1)[1].strip() if "\n\n" in raw else raw.strip()
        if not body:
            continue
        actual_hash = hashlib.sha256(f"{owner}:{execution_id}:{body}".encode()).hexdigest()
        if actual_hash == expected_hash:
            entries.append(body)
    return entries


def register_free_think_hooks(
    ctx: PluginRegistration[SanaeMemProvider], prototype: SanaeMemProvider, *, native_turn_context: bool,
) -> Callable[[], None]:
    registered_home = get_hermes_home().resolve()
    if not _bool(prototype.runtime_config(registered_home).get("free_think"), False):
        return lambda: None

    def append_context(*, session_id="", **turn):
        home = get_hermes_home().resolve()
        if home != registered_home:
            return None
        config = prototype.runtime_config(home)
        if current_free_think(config, session_id) is None:
            return None
        # Native Tier 2 runs after pre_llm_call, and needs an initialized cron provider.
        if native_turn_context:
            active = resolve_active_provider(home, session_id)
            if active is not None and active.is_cron_context:
                return None
        worker = prototype.open_runtime_worker(
            home, config, session_id=session_id, platform="cron", agent_context="cron",
        )
        try:
            block = worker.current_turn_context(session_id=session_id, synthetic=False, **turn)
            return {"context": block, "ephemeral": True} if block else None
        finally:
            worker.shutdown()

    def append_diary(args, *, session_id="", **_):
        if get_hermes_home().resolve() != registered_home:
            return json.dumps({"ok": False, "error": "Profile changed."})
        current = current_free_think(prototype.runtime_config(registered_home), session_id)
        text = args.get("text") if isinstance(args, dict) else None
        if current is None or not isinstance(text, str) or not text.strip():
            return json.dumps({"ok": False, "error": "A verified free-think run and nonempty text are required."})
        owner, env = current
        diary = registered_home / "diary"
        if diary.resolve() != diary:
            return json.dumps({"ok": False, "error": "Diary folder is outside the profile."})
        diary.mkdir(mode=0o700, exist_ok=True)
        text = text.strip()
        entry = hashlib.sha256(f"{owner}:{env.execution_id}:{text}".encode()).hexdigest()
        path = diary / f"free-think-{entry}.md"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return json.dumps({"ok": True, "already_written": True})
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"# {datetime.now().astimezone().isoformat(timespec='minutes')}\n\n{text}\n")
        except (OSError, UnicodeError):
            path.unlink(missing_ok=True)
            raise
        return json.dumps({"ok": True})

    def on_cron_completion(event: CronCompletion) -> None:
        if event.status != "completed":
            return
        home = host_bridge.cron_profile_home(event)
        if home != registered_home or home != get_hermes_home().resolve():
            return
        config = prototype.runtime_config(home)
        if not _free_think_enabled(config):
            return
        owner = _schedule_owner(config)
        frozen_job = host_bridge.cron_completion_job_spec(event)
        if (not owner or not event.execution_id or frozen_job.get("id") != event.job_id
                or not _private_free_think_job(frozen_job, owner, _private_free_think_target(config))):
            return
        try:
            stamp = datetime.fromisoformat(event.finished_at)
            stamp_ts = stamp.timestamp() if stamp.tzinfo is not None else time.time()
        except (TypeError, ValueError):
            stamp_ts = time.time()
        data_dir = Path(config.get("data_dir") or home / "sanae-mem").expanduser().resolve()
        profile_key = daily_profile_key(home)

        def apply_completed_heartbeat() -> None:
            current_config = prototype.runtime_config(home)
            if (not _free_think_enabled(current_config) or _schedule_owner(current_config) != owner
                    or not _private_free_think_job(frozen_job, owner, _private_free_think_target(current_config))):
                return
            worker = prototype.open_runtime_worker(
                home, current_config, session_id=event.session_id, platform="cron", agent_context="cron",
            )
            try:
                exploration_parts: list[str] = []
                final_response = str(event.final_response or "").strip()
                if final_response and final_response not in {"HEARTBEAT_OK", "[SILENT]"}:
                    exploration_parts.append(final_response)
                exploration_parts.extend(_execution_diary_texts(home, owner, event.execution_id))
                reflection_rows = completed_free_think_rows(
                    home, owner, stamp_ts - 24 * 3600, stamp_ts + 1.0,
                    data_dir=data_dir, execution_id=event.execution_id,
                )
                exploration_parts.extend(text for _ts, _label, text in reflection_rows if text)
                if exploration_parts:
                    try:
                        worker.resolve_explored_curiosity_threads("\n".join(exploration_parts))
                    except sqlite3.Error as exc:
                        logger.warning(
                            "sanae_mem: curiosity thread resolve failed during cron completion (%s)",
                            type(exc).__name__,
                        )
                worker.update_inner_state(
                    turn_id=f"cron:{event.job_id}:{event.execution_id}",
                    user_text="", assistant_text="",
                    session_id=event.session_id, now=stamp_ts, owner_key=owner,
                    preserve_processed_at=True, internal=True,
                )
            finally:
                worker.shutdown()

        prototype.submit_free_think_heartbeat(data_dir, profile_key, apply_completed_heartbeat)

    def diary_available() -> bool:
        if get_hermes_home().resolve() != registered_home:
            return False
        envelope = host_bridge.verified_cron_envelope()
        return envelope is not None and current_free_think(
            prototype.runtime_config(registered_home), envelope.session_id,
        ) is not None

    ctx.register_hook("pre_llm_call", append_context)
    host_bridge.register_plugin_tool(
        ctx,
        name=_DIARY_TOOL, toolset=_DIARY_TOOL, handler=append_diary,
        schema={
            "name": _DIARY_TOOL,
            "description": "自由放空时留一段给自己的日记。",
            "parameters": {
                "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
            },
        },
        check_fn=diary_available,
    )
    if host_bridge.register_cron_completion_handler(on_cron_completion):
        return lambda: host_bridge.unregister_cron_completion_handler(on_cron_completion)
    return lambda: None
