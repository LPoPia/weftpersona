from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from .atomic_io import atomic_json_write, file_lock
from .clock import shanghai_datetime
from .host_bridge import is_test_isolated_turn
from .chat_scope import in_public_interaction

logger = logging.getLogger(__name__)
_STATUS_WRITER = RLock()
StatusPayload = dict[str, Any]
StatusMutation = Callable[[StatusPayload], StatusPayload]


def read_status_file(path: Path) -> StatusPayload:
    with _STATUS_WRITER:
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, RecursionError):
            return {}


def _component_error_code(value: object) -> str:
    head = re.split(r"[:：]", str(value or "").strip(), maxsplit=1)[0].strip()
    return (head or "unknown")[:80]


def write_component_status(
    data_dir: Path,
    component: str,
    *,
    ok: bool,
    stage: str,
    error: str = "",
    session_id: str = "",
    now: Any = None,
    **extra: Any,
) -> None:
    """Persist a single component health record into *data_dir*/status.json."""
    if in_public_interaction() or is_test_isolated_turn(session_id):
        return
    record: dict[str, Any] = {
        "ok": bool(ok),
        "stage": str(stage or "unknown"),
        "at": shanghai_datetime(now).isoformat(timespec="seconds"),
    }
    if error:
        record["error"] = _component_error_code(error)
    record.update({key: value for key, value in extra.items() if value not in (None, "")})
    path = Path(data_dir).resolve() / "status.json"
    try:
        with _STATUS_WRITER, file_lock(path.with_suffix(".lock")):
            latest = read_status_file(path)
            components = latest.get("components")
            if not isinstance(components, dict):
                components = {}
            components[str(component)] = record
            latest["components"] = components
            atomic_json_write(path, latest)
    except OSError as exc:
        logger.warning("sanae_mem: %s status write failed (%s)", component, type(exc).__name__)


class StatusStore:

    _STATUS_DIFF_EXCLUDED = frozenset({"components", "inner_state"})

    def __init__(
        self, *, current_data_dir: Callable[[], Path], current_session_id: Callable[[], str],
    ) -> None:
        self._current_data_dir = current_data_dir
        self._current_session_id = current_session_id

    def read(self) -> StatusPayload:
        return read_status_file(self._current_data_dir() / "status.json")

    @staticmethod
    def roll_date(payload: StatusPayload, today: str) -> None:
        """Reset the counters owned by the shared natural-day ``date`` field."""
        if payload.get("date") == today:
            return
        payload["date"] = today
        for key in (
            "daily_writes", "daily_scheduled_asks", "daily_scheduled_attempts",
            "daily_candidate_reviews", "daily_resident_reviews", "daily_compactor_runs",
            "distill_errors_today",
        ):
            payload[key] = 0

    def mutate(
        self, change: StatusMutation, *, session_id: str = "", data_dir: Path | None = None,
    ) -> None:
        if in_public_interaction():
            return
        if is_test_isolated_turn(session_id or self._current_session_id()):
            return
        path = (data_dir if data_dir is not None else self._current_data_dir()) / "status.json"
        with _STATUS_WRITER, file_lock(path.with_suffix(".lock")):
            latest = read_status_file(path)
            updated = change(latest)
            if not isinstance(updated, dict):
                raise TypeError("status mutation must return a dict")
            atomic_json_write(path, updated)

    def record_write(self, now: datetime) -> None:
        date = now.date().isoformat()

        def record(payload: StatusPayload) -> StatusPayload:
            self.roll_date(payload, date)
            count = int(payload.get("daily_writes", 0))
            payload.update({"daily_writes": count + 1, "last_write_at": now.isoformat()})
            return payload

        self.mutate(record)

    def write_diff(
        self, payload: StatusPayload, *, base: StatusPayload, session_id: str = "",
        data_dir: Path | None = None,
    ) -> None:
        """Apply only business fields changed from a previously read status snapshot."""
        changed = {
            key: value for key, value in payload.items()
            if key not in self._STATUS_DIFF_EXCLUDED
            and (key not in base or base[key] != value)
        }
        removed = (set(base) - set(payload)) - self._STATUS_DIFF_EXCLUDED

        def apply(latest: StatusPayload) -> StatusPayload:
            latest.update(changed)
            for key in removed:
                latest.pop(key, None)
            return latest

        self.mutate(apply, session_id=session_id, data_dir=data_dir)

    def write_maintenance(
        self, component: str, *, ok: bool, stage: str, error: str = "",
        now: datetime | float | None = None, data_dir: Path | None = None, **extra: object,
    ) -> None:
        record: dict[str, object] = {
            "ok": bool(ok), "stage": str(stage or "unknown"),
            "at": shanghai_datetime(now).isoformat(timespec="seconds"),
        }
        if error:
            record["error"] = _component_error_code(error)
        record.update({key: value for key, value in extra.items() if value not in (None, "")})

        def merge(payload: StatusPayload) -> StatusPayload:
            components = payload.get("components")
            if not isinstance(components, dict):
                components = {}
            components[str(component)] = record
            payload["components"] = components
            return payload

        try:
            self.mutate(merge, data_dir=data_dir)
        except OSError as exc:
            logger.warning("sanae_mem: %s status not written (%s)",
                           component, type(exc).__name__)
