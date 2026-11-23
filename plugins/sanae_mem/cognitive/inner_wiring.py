from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..governance.group_scope import GroupGrant, member_impression_owner, member_impression_read_owners
from ..governance.source_context import current_chat_scope, source_json
from ..infra.atomic_io import atomic_json_write, file_lock
from ..infra.clock import shanghai_datetime
from ..infra.contracts import CasConflictError
from ..infra.host_bridge import (
    ctx_bound,
    current_cron_binding,
    is_delegated_child_turn,
    is_test_isolated_turn,
    spawn_context_thread,
)
from ..infra.status_store import _STATUS_WRITER, _component_error_code, read_status_file
from ..storage.background_tasks import BackgroundTasks
from ..storage.sqlite_facade import SqliteFacade
from .daily_inputs import delivered_user_input
from .impression import read_impression, render_impression, write_impression
from .inner_state_life import INNER_STATE_MARKER
from .inner_state_spark import consume_pending_spark
from .inner_state_types import InnerState, InnerStateTarget
from .life_context import collect_life_experiences, collect_recent_activities
from .previous_session import PreviousSessionQuery, read_previous_session_tail
from .self_utterances import HEADER as _SELF_HEADER, recent_self_utterances
from .turn_user_context import record_turn_user_context

logger = logging.getLogger(__name__)

_ACTIVITY_LEAD = (
    "（以下是我自己的空闲经历；顺口提起时只先丢半句引子或一句简单印象，留白看他接不接茬；"
    "哪怕正聊在共同兴头上，也只用一两句日常大白话带过核心意思，绝不照抄下面的书面术语或分步说明）\n我这阵子刚做完的事："
)
_WAIT_SECONDS = 3.0
_RECEIPT_WAIT_SECONDS = 30.0
_RECEIPT_POLL_SECONDS = 0.1
BackgroundLane = Literal["private", "group", "cron"]
_LegacyTarget = tuple[Path, str]
_QueueTarget = tuple[Path, str, BackgroundLane]
_Target = _QueueTarget
_TargetLike = _QueueTarget | _LegacyTarget


INNER_STATE_SCHEMA = {
    "name": "sanae_inner_state",
    "description": (
        "读写我此刻的心里话。read 是看看自己现在心里怎么想；"
        "write 是我想亲手记下自己的当下感受，用第一人称写两到四句话。"
        "心里话是给我自己的，不必告诉对方。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["read", "write"]},
            "text": {"type": "string", "description": "write 时的心里话"},
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class _QueuedJob:
    run: Callable[[], None]
    finished: threading.Event
    background_tasks: BackgroundTasks
    test_isolated: bool

    def __post_init__(self) -> None:
        if self.background_tasks is None:
            raise ValueError("background job requires a background task registry")


@dataclass
class _Updates:
    queue: deque[_QueuedJob]
    latest: threading.Event


_updates_lock = threading.Lock()
_updates: dict[_QueueTarget, _Updates] = {}


@dataclass(eq=False)
class _ReceiptWatch:
    target: _TargetLike
    probe: Callable[[], tuple[bool, float | None]]
    apply: Callable[[float], None]
    expires: float
    background_tasks: BackgroundTasks

    def __post_init__(self) -> None:
        if self.background_tasks is None:
            raise ValueError("background delivery watch requires a background task registry")


_receipts_lock = threading.Lock()
_receipt_watches: dict[_ReceiptWatch, None] = {}
_receipt_worker_running = False


def _normalize_lane(lane: str) -> BackgroundLane:
    if lane == "private":
        return "private"
    if lane == "group":
        return "group"
    if lane == "cron":
        return "cron"
    raise ValueError(f"unsupported background queue lane: {lane!r}")


def _normalize_queue_target(
    target: _TargetLike,
    *,
    default_lane: BackgroundLane = "private",
) -> _QueueTarget:
    if len(target) == 3:
        data_dir, profile_key, lane = target
        return (Path(data_dir).expanduser().resolve(), str(profile_key), _normalize_lane(lane))
    if len(target) == 2:
        data_dir, profile_key = target
        return (Path(data_dir).expanduser().resolve(), str(profile_key), default_lane)
    raise ValueError(f"expected (data_dir, profile_key[, lane]) target, got {target!r}")


def _parse_target_filter(
    target: _TargetLike | None,
    lane: BackgroundLane | None,
) -> tuple[tuple[Path, str] | None, BackgroundLane | None]:
    normalized_lane = _normalize_lane(lane) if lane is not None else None
    if target is None:
        return None, normalized_lane
    if len(target) == 3:
        data_dir, profile_key, target_lane = target
        resolved_lane = _normalize_lane(target_lane)
        if normalized_lane is not None and normalized_lane != resolved_lane:
            raise ValueError(
                f"conflicting lanes: target has {resolved_lane!r}, lane argument has {normalized_lane!r}"
            )
        return (Path(data_dir).expanduser().resolve(), str(profile_key)), resolved_lane
    if len(target) == 2:
        data_dir, profile_key = target
        return (Path(data_dir).expanduser().resolve(), str(profile_key)), normalized_lane
    raise ValueError(f"expected (data_dir, profile_key[, lane]) target, got {target!r}")


def _matches_queue_target(
    queue_target: _QueueTarget,
    owner_filter: tuple[Path, str] | None,
    lane_filter: BackgroundLane | None,
) -> bool:
    if owner_filter is not None and (queue_target[0], queue_target[1]) != owner_filter:
        return False
    if lane_filter is not None and queue_target[2] != lane_filter:
        return False
    return True


def _record_background_failure(target: _QueueTarget, exc: BaseException) -> None:
    """Persist background worker failure metadata to *data_dir*/status.json without chat text."""
    data_dir, _profile_key, lane = target
    path = Path(data_dir).expanduser().resolve() / "status.json"
    error_code = _component_error_code(type(exc).__name__)
    recorded_at = shanghai_datetime(None).isoformat(timespec="seconds")
    try:
        with _STATUS_WRITER, file_lock(path.with_suffix(".lock")):
            latest = read_status_file(path)
            raw_components = latest.get("components")
            components: dict[str, object] = (
                dict(raw_components) if isinstance(raw_components, dict) else {}
            )
            previous = components.get("background_worker")
            prev_count = previous.get("failure_count") if isinstance(previous, dict) else 0
            components["background_worker"] = {
                "ok": False,
                "stage": "drain",
                "lane": lane,
                "error": error_code,
                "at": recorded_at,
                "failure_count": (prev_count if type(prev_count) is int else 0) + 1,
            }
            latest["components"] = components
            atomic_json_write(path, latest)
    except OSError as status_exc:
        logger.warning(
            "sanae_mem: background worker status write failed (lane=%s, error=%s)",
            lane,
            type(status_exc).__name__,
        )


def _collect_deliveries(
    target: _TargetLike | None = None,
    *,
    lane: BackgroundLane | None = None,
) -> None:
    """Probe without blocking; confirmed turns queue chronologically per owner lane."""
    owner_filter, lane_filter = _parse_target_filter(target, lane)
    with _receipts_lock:
        watches = [
            watch
            for watch in _receipt_watches
            if _matches_queue_target(_normalize_queue_target(watch.target), owner_filter, lane_filter)
        ]
    ready: list[tuple[_ReceiptWatch, float | None]] = []
    for watch in watches:
        try:
            terminal, delivered_at = watch.probe()
        except Exception as exc:
            logger.warning("sanae_mem: delivery receipt read failed (%s)", type(exc).__name__)
            terminal, delivered_at = True, None
        if terminal or time.monotonic() >= watch.expires:
            ready.append((watch, delivered_at))
    with _receipts_lock:
        ready_by_target: dict[_QueueTarget, list[tuple[_ReceiptWatch, float]]] = {}
        for watch, delivered_at in ready:
            if watch not in _receipt_watches:
                continue
            del _receipt_watches[watch]
            if delivered_at is not None:
                queue_target = _normalize_queue_target(watch.target)
                ready_by_target.setdefault(queue_target, []).append((watch, delivered_at))

        for queue_target, queued in ready_by_target.items():
            for watch, delivered_at in sorted(queued, key=lambda item: item[1]):
                _submit(
                    watch.background_tasks,
                    queue_target,
                    lambda w=watch, at=delivered_at: w.apply(at),
                )


def _poll_delivery_receipts() -> None:
    global _receipt_worker_running
    while True:
        _collect_deliveries()
        with _receipts_lock:
            if not _receipt_watches:
                _receipt_worker_running = False
                return
        time.sleep(_RECEIPT_POLL_SECONDS)


def _watch_delivery(
    target: _TargetLike,
    probe: Callable[[], tuple[bool, float | None]],
    apply: Callable[[float], None],
    *,
    background_tasks: BackgroundTasks,
) -> None:
    global _receipt_worker_running
    if background_tasks is None:
        raise ValueError("background delivery watch requires a background task registry")
    queue_target = _normalize_queue_target(target)
    watch = _ReceiptWatch(
        queue_target,
        probe,
        ctx_bound(apply),
        time.monotonic() + _RECEIPT_WAIT_SECONDS,
        background_tasks=background_tasks,
    )
    with _receipts_lock:
        _receipt_watches[watch] = None
        if not _receipt_worker_running:
            _receipt_worker_running = True
            try:
                spawn_context_thread(_poll_delivery_receipts, name="sanae-delivery-receipts").start()
            except Exception:
                del _receipt_watches[watch]
                _receipt_worker_running = False
                raise


def _drain(target: _QueueTarget, updates: _Updates) -> None:
    while True:
        with _updates_lock:
            if not updates.queue:
                del _updates[target]
                return
            queued = updates.queue.popleft()
        try:
            queued.run()
        except Exception as exc:
            logger.warning(
                "sanae_mem: background worker failed (lane=%s, error=%s)",
                target[2],
                type(exc).__name__,
            )
            if not queued.test_isolated:
                _record_background_failure(target, exc)
        finally:
            try:
                queued.background_tasks.complete()
            finally:
                queued.finished.set()


def _submit(
    background_tasks: BackgroundTasks,
    target: _TargetLike,
    job: Callable[[], None],
) -> bool:
    if background_tasks is None:
        raise ValueError("background queue submission requires a background task registry")
    queue_target = _normalize_queue_target(target)
    if not background_tasks.try_register():
        logger.warning(
            "sanae_mem: rejected background job after provider shutdown (lane=%s)",
            queue_target[2],
        )
        return False
    try:
        finished = threading.Event()
        queued = _QueuedJob(
            run=ctx_bound(job),
            finished=finished,
            background_tasks=background_tasks,
            test_isolated=is_test_isolated_turn(),
        )
        with _updates_lock:
            updates = _updates.get(queue_target)
            if updates is not None:
                updates.queue.append(queued)
                updates.latest = finished
                return True
            updates = _updates[queue_target] = _Updates(deque([queued]), finished)
            try:
                spawn_context_thread(
                    _drain,
                    name=f"sanae-bg-{queue_target[2]}",
                    args=(queue_target, updates),
                ).start()
            except BaseException:
                del _updates[queue_target]
                raise
    except BaseException:
        background_tasks.complete()
        raise
    return True


def _wait_for_pending(
    target: _TargetLike,
    *,
    lane: BackgroundLane | None = None,
    timeout: float = _WAIT_SECONDS,
) -> None:
    # The next turn can start before the poller's next tick. Recheck actual receipts now.
    wait_for_pending_updates(target, lane=lane, timeout=timeout)


def wait_for_pending_updates(
    target: _TargetLike | None = None,
    *,
    lane: BackgroundLane | None = None,
    timeout: float = _WAIT_SECONDS,
) -> bool:
    """Wait deterministically for queued background updates matching *target* and *lane*."""
    owner_filter, lane_filter = _parse_target_filter(target, lane)
    _collect_deliveries(target, lane=lane)
    with _updates_lock:
        events = [
            updates.latest
            for queue_target, updates in _updates.items()
            if _matches_queue_target(queue_target, owner_filter, lane_filter)
        ]
    if not events:
        return True
    deadline = time.monotonic() + max(0.0, timeout)
    for event in events:
        remaining = max(0.0, deadline - time.monotonic())
        if not event.wait(remaining):
            return False
    return True


class _RenderInnerState(Protocol):
    def __call__(
        self, *, target: InnerStateTarget, owner_key: str, session_id: str, internal: bool,
    ) -> str: ...


class _WriteInnerState(Protocol):
    def __call__(self, text: str, /, *, turn_id: str, owner_key: str, internal: bool) -> InnerState: ...


class _UpdateInnerState(Protocol):
    def __call__(
        self, *, turn_id: str, user_text: str, assistant_text: str, experiences: str,
        image_context: str, session_id: str, now: float, target: InnerStateTarget,
        impression_target: tuple[Path, str, str], owner_key: str,
    ) -> bool: ...


class _RecordInnerStateFailure(Protocol):
    def __call__(self, reason: str, /, *, session_id: str, target: InnerStateTarget) -> None: ...


class _RecordDeliveredRementions(Protocol):
    def __call__(
        self, *, owner_key: str, user_text: str, assistant_text: str, turn_id: str, delivered_at: float,
    ) -> Mapping[str, object]: ...


class InnerWiring:
    def __init__(
        self,
        sqlite: SqliteFacade,
        *,
        current_home: Callable[[], Path],
        current_data_dir: Callable[[], Path],
        current_profile: Callable[[], str],
        read_owners: Callable[[], tuple[str, ...]],
        partner_name: str,
        partner_role: str,
        private_turn_owner: Callable[[str], str],
        authorized_cron_owner: Callable[[str], str],
        is_maintainer: Callable[[], bool],
        group_grant: Callable[[], GroupGrant | None],
        render_maintainer_daily_facts: Callable[[], str],
        render_commitments: Callable[[str], str],
        inner_state_target: Callable[[], InnerStateTarget],
        latest_inner_state: Callable[[InnerStateTarget], InnerState | None],
        render_inner_state: _RenderInnerState,
        write_inner_state: _WriteInnerState,
        update_inner_state: _UpdateInnerState,
        record_inner_state_failure: _RecordInnerStateFailure,
        record_delivered_rementions: _RecordDeliveredRementions,
        resolve_explored_curiosity: Callable[[str], int],
    ) -> None:
        self._sqlite = sqlite
        self._current_home = current_home
        self._current_data_dir = current_data_dir
        self._current_profile = current_profile
        self._read_owners = read_owners
        self._partner_name = partner_name
        self._partner_role = partner_role
        self._private_turn_owner = private_turn_owner
        self._authorized_cron_owner = authorized_cron_owner
        self._is_maintainer = is_maintainer
        self._group_grant = group_grant
        self._render_maintainer_daily_facts = render_maintainer_daily_facts
        self._render_commitments = render_commitments
        self._inner_state_target = inner_state_target
        self._latest_inner_state = latest_inner_state
        self._render_inner_state = render_inner_state
        self._write_inner_state = write_inner_state
        self._update_inner_state = update_inner_state
        self._record_inner_state_failure = record_inner_state_failure
        self._record_delivered_rementions = record_delivered_rementions
        self._resolve_explored_curiosity = resolve_explored_curiosity

    def reconfigure(self, *, partner_name: str, partner_role: str) -> None:
        self._partner_name = partner_name
        self._partner_role = partner_role

    def cron_inner_scope(
        self, *, session_id: str, synthetic: bool,
    ) -> tuple[str, str] | None:
        if synthetic or not session_id:
            return None
        owner = self._authorized_cron_owner(session_id)
        return (owner, self._current_profile()) if owner else None

    def inner_turn_scope(
        self, *, session_id: str, turn_id: str, synthetic: bool = False,
    ) -> tuple[str, str] | None:
        if synthetic or not turn_id or turn_id == "system-prompt" or not session_id:
            return None
        owner = self._private_turn_owner(session_id)
        return (owner, self._current_profile()) if owner else None

    def current_turn_context(
        self, *, session_id: str, turn_id: str, synthetic: bool, sender_id: str,
        sender_id_alt: str, durable_native_thread: bool,
    ) -> str:
        person_scope = self.inner_turn_scope(session_id=session_id, turn_id=turn_id, synthetic=synthetic)
        cron_scope = None if person_scope is not None else self.cron_inner_scope(
            session_id=session_id, synthetic=synthetic,
        )
        scope = person_scope or cron_scope
        if scope is None:
            return ""
        target = self._inner_state_target()
        _wait_for_pending((target[0], target[1], "cron" if cron_scope is not None else "private"))
        person_scope = self.inner_turn_scope(session_id=session_id, turn_id=turn_id, synthetic=synthetic)
        cron_scope = None if person_scope is not None else self.cron_inner_scope(
            session_id=session_id, synthetic=synthetic,
        )
        scope = person_scope or cron_scope
        if scope is None:
            return ""
        is_cron = cron_scope is not None
        current_owner = scope[0]
        is_maintainer = not is_cron and self._is_maintainer()

        state = self._latest_inner_state(target)
        block = self._render_inner_state(
            target=target, owner_key=current_owner, session_id=session_id,
            internal=is_cron,
        )

        partner_name = self._partner_name
        partner_role = self._partner_role

        world_items: list[str] = []

        if not is_cron:
            sender = str(sender_id or sender_id_alt or current_owner[:8])
            if is_maintainer:
                partner_name = partner_name or sender
                identity_line = (
                    f"现在跟我说话的是{partner_name}，我的{partner_role}。"
                    "下面他今天的活动，是他自己的事。"
                )
            else:
                identity_line = f"现在跟我说话的是{sender}。"
            world_items.append(f"〔现在跟我说话的人〕\n{identity_line}")

        facts = self._recent_activities(
            current_owner, state, session_id=session_id, allow_trusted_cron=is_cron,
            private_experience=is_cron or is_maintainer,
        )
        raw_outbound = ""
        outbound = ""
        if not is_cron:
            home_path = self._current_home()
            try:
                raw_outbound = recent_self_utterances(
                    current_owner, session_id, home=home_path, data_dir=self._current_data_dir(),
                    unconsumed_only=True,
                    read_owners=self._read_owners(),
                )
                if raw_outbound:
                    outbound = "\n".join(
                        line for line in raw_outbound.splitlines()
                        if line.strip() and line.strip() != _SELF_HEADER)
            except (OSError, sqlite3.Error) as exc:
                logger.warning("sanae_mem: recent self utterances read failed (%s)", type(exc).__name__)
                raw_outbound = ""
                outbound = ""
            if raw_outbound:
                record_turn_user_context(
                    home_path,
                    session_id,
                    turn_id,
                    self_utterances=raw_outbound,
                )
        clean_facts = facts.strip()
        clean_outbound = outbound.strip()
        merged_activities = []
        if clean_facts:
            merged_activities.append(clean_facts)
        if clean_outbound and clean_outbound not in clean_facts:
            merged_activities.append(clean_outbound)
        if merged_activities:
            world_items.append("\n".join((_ACTIVITY_LEAD, "\n".join(merged_activities))))

        try:
            commitments = self._render_commitments(current_owner).strip()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("sanae_mem: render_pending_commitments failed (%s)", type(exc).__name__)
            commitments = ""
        if commitments:
            world_items.append(f"〔我答应过还没办完的事〕\n{commitments}")

        if is_maintainer:
            maintainer_facts = self._render_maintainer_daily_facts()
            if maintainer_facts:
                world_items.append(f"〔{partner_name}的今日活动〕\n{maintainer_facts}")

        if not is_cron and not is_delegated_child_turn():
            previous_tail = read_previous_session_tail(
                PreviousSessionQuery(
                    home=self._current_home(),
                    current_session_id=session_id,
                    current_owner=current_owner,
                    read_owners=self._read_owners(),
                    scope=current_chat_scope(),
                    is_maintainer=is_maintainer,
                    partner_name=partner_name,
                )
            )
            if previous_tail:
                world_items.append(previous_tail)

        if not is_cron and not durable_native_thread:
            imp = read_impression((target[0], scope[1], current_owner), owners=self._read_owners())
            if imp and (imp.text or imp.baseline or imp.recent):
                rendered_imp = render_impression(
                    imp, is_maintainer=is_maintainer, partner_name=partner_name, partner_role=partner_role,
                )
                if rendered_imp:
                    world_items.append(rendered_imp)

        if not world_items and not block:
            return ""
        outer = "[我眼中的世界]\n" + "\n\n".join(world_items) if world_items else ""
        inner = "[我内心的感受]\n" + (block or INNER_STATE_MARKER)
        return "\n\n".join(part for part in (outer, inner) if part)

    def _recent_activities(
        self, owner: str, state: InnerState | None, *, session_id: str,
        allow_trusted_cron: bool, private_experience: bool,
    ) -> str:
        since = state.processed_at if state is not None else None
        try:
            return collect_recent_activities(
                home=self._current_home(), data_dir=self._current_data_dir(), owner_key=owner,
                session_id=session_id, since=since, through=time.time(),
                allow_trusted_cron=allow_trusted_cron,
                private_experience=private_experience,
                read_owners=self._read_owners(),
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("sanae_mem: recent activity read failed (%s)", type(exc).__name__)
            return ""

    def group_impression_target(self, grant: GroupGrant | None) -> tuple[Path, str, str] | None:
        """The member impression row of this group turn, or ``None`` when not authorized.

        It lives under the exact group/thread owner plus the trusted person label, so the same
        person in two groups, in a sibling thread or in a private chat keeps separate rows, and a
        group row never becomes a private read.
        """
        owner = member_impression_owner(grant)
        if not owner:
            return None
        return (self._current_data_dir(), self._current_profile(), owner)

    def tool_impression(
        self, *, operation: object, text: object, session_id: str, turn_id: str, synthetic: bool,
    ) -> dict[str, str]:
        grant = self._group_grant()
        group_target = self.group_impression_target(grant)
        if grant is not None and group_target is not None:
            if operation == "read":
                # A real thread also reads the shared parent's row about this same member.
                state = read_impression(
                    group_target, owners=member_impression_read_owners(grant)[1:])
                return {"text": state.text if state is not None else ""}
            if operation == "write":
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("请写下我对他的印象。")
                # The row only ever grows under this scene's own write owner, and it keeps the
                # real speaker/chat/thread/message of this turn as its provenance.
                try:
                    state = write_impression(
                        group_target, text.strip(), turn_id=turn_id,
                        source_context=source_json(grant.source))
                except CasConflictError as exc:
                    logger.warning("sanae_mem: group impression CAS conflict (%s)", exc)
                    self._record_inner_state_failure(
                        "impression_cas_conflict",
                        session_id=session_id,
                        target=self._inner_state_target(),
                    )
                    raise
                return {"text": state.text, "note": "我已记下新版，下一轮就能带上。"}
            raise ValueError("请选择 read 或 write。")
        scope = self.inner_turn_scope(session_id=session_id, turn_id=turn_id, synthetic=synthetic)
        if scope is None:
            raise PermissionError("对他的印象只在本人的私聊中使用。")
        owner, profile = scope
        target = (self._inner_state_target()[0], profile, owner)
        if operation == "read":
            state = read_impression(target, owners=self._read_owners())
            return {"text": state.text if state is not None else ""}
        if operation == "write":
            if not isinstance(text, str) or not text.strip():
                raise ValueError("请写下我对他的印象。")
            try:
                state = write_impression(target, text.strip(), turn_id=turn_id)
            except CasConflictError as exc:
                logger.warning("sanae_mem: private impression CAS conflict (%s)", exc)
                self._record_inner_state_failure(
                    "impression_cas_conflict",
                    session_id=session_id,
                    target=self._inner_state_target(),
                )
                raise
            return {"text": state.text, "note": "我已记下新版，下一轮就能带上。"}
        raise ValueError("请选择 read 或 write。")

    def tool_inner_state(
        self, *, operation: object, text: object, session_id: str, turn_id: str, synthetic: bool,
    ) -> dict[str, str]:
        person_scope = self.inner_turn_scope(session_id=session_id, turn_id=turn_id, synthetic=synthetic)
        cron_scope = None if person_scope is not None else self.cron_inner_scope(
            session_id=session_id, synthetic=synthetic,
        )
        scope = person_scope or cron_scope
        if scope is None:
            raise PermissionError("心里话只在本人的私聊中使用。")
        internal = cron_scope is not None
        if operation == "read":
            return {"text": self._render_inner_state(
                target=self._inner_state_target(), owner_key=scope[0], session_id=session_id,
                internal=internal,
            )}
        if operation == "write":
            if not isinstance(text, str) or not text.strip():
                raise ValueError("请写下我此刻的心里话。")
            if internal:
                binding = current_cron_binding()
                if not binding:
                    raise PermissionError("当前自由放空执行身份未通过核验。")
                _, job_id, execution_id = binding
                turn_id = f"cron:{job_id}:{execution_id}"
            try:
                state = self._write_inner_state(
                    text, turn_id=turn_id, owner_key=scope[0], internal=internal)
            except CasConflictError as exc:
                logger.warning("sanae_mem: inner_state CAS conflict (%s)", exc)
                self._record_inner_state_failure(
                    "inner_state_cas_conflict",
                    session_id=session_id,
                    target=self._inner_state_target(),
                )
                raise
            return {"text": state.text}
        raise ValueError("请选择 read 或 write。")

    def apply_verified_delivery(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message: str,
        assistant_response: str,
        delivered_at: float,
        has_native_image: bool = False,
    ) -> bool:
        scope = self.inner_turn_scope(session_id=session_id, turn_id=turn_id)
        if scope is None:
            return False
        target = self._inner_state_target()
        record_failure = self._record_inner_state_failure
        try:
            previous = self._latest_inner_state(target)
            watermark = self._sqlite.fetchone(
                "SELECT delivered_at, turn_id FROM session_reply_watermarks "
                "WHERE owner_key=? AND session_id=?",
                (scope[0], session_id),
            )
            stale_receipt = (
                (previous is not None and previous.updated_at > delivered_at)
                or (
                    watermark is not None
                    and (
                        float(watermark["delivered_at"] or 0.0) > delivered_at
                        or (bool(turn_id) and str(watermark["turn_id"] or "") == turn_id)
                    )
                )
            )
            if not stale_receipt:
                consume_pending_spark(
                    target[0] / "sanae.db",
                    target,
                    owner_key=scope[0],
                    exclude_turn_id=turn_id,
                    delivered_at=delivered_at,
                )
        except sqlite3.Error as exc:
            logger.warning("sanae_mem: pending spark consume failed (%s)", type(exc).__name__)
            record_failure("spark_consume_failed", session_id=session_id, target=target)
            return False
        private_experience = self._is_maintainer()
        user_text, image_context = delivered_user_input(
            user_message, has_native_image=has_native_image,
        )
        remention_ok = True
        if user_text.strip():
            try:
                remention_res = self._record_delivered_rementions(
                    owner_key=scope[0], user_text=user_text, assistant_text=assistant_response,
                    turn_id=turn_id, delivered_at=delivered_at,
                )
            except (OSError, sqlite3.Error) as exc:
                remention_ok = False
                logger.warning("sanae_mem: delivered remention skipped (%s)", type(exc).__name__)
            else:
                if remention_res.get("ok") is False:
                    remention_ok = False
                    logger.warning(
                        "sanae_mem: delivered remention failed (%s)",
                        str(remention_res.get("reason") or "unknown"),
                    )
        try:
            experiences = collect_life_experiences(
                home=self._current_home(), data_dir=target[0], owner_key=scope[0], session_id=session_id,
                since=previous.processed_at if previous is not None else None,
                through=delivered_at, private_experience=private_experience,
                read_owners=self._read_owners(),
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("sanae_mem: life context failed during verified delivery (%s)", type(exc).__name__)
            record_failure("life_context_failed", session_id=session_id, target=target)
            return False
        if experiences:
            try:
                self._resolve_explored_curiosity(experiences)
            except (OSError, sqlite3.Error) as exc:
                logger.warning("sanae_mem: curiosity thread resolve failed (%s)", type(exc).__name__)
        try:
            self._sqlite.execute(
                "INSERT INTO session_reply_watermarks(owner_key, session_id, platform, delivered_at, turn_id) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(owner_key, session_id) DO UPDATE SET "
                "platform=excluded.platform, "
                "turn_id=CASE WHEN excluded.delivered_at >= session_reply_watermarks.delivered_at "
                "THEN excluded.turn_id ELSE session_reply_watermarks.turn_id END, "
                "delivered_at=MAX(session_reply_watermarks.delivered_at, excluded.delivered_at)",
                (scope[0], session_id, current_chat_scope().platform, delivered_at, turn_id),
            )
        except sqlite3.Error as exc:
            logger.warning("sanae_mem: session reply watermark write failed (%s)", type(exc).__name__)
            record_failure("watermark_write_failed", session_id=session_id, target=target)
            return False
        inner_state_ok = self._update_inner_state(
            turn_id=turn_id, user_text=user_text, assistant_text=assistant_response,
            experiences=experiences, image_context=image_context,
            session_id=session_id, now=delivered_at, target=target,
            impression_target=(target[0], scope[1], scope[0]), owner_key=scope[0],
        )
        return inner_state_ok and remention_ok
