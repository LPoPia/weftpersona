"""Explicit runtime bindings shared across ViveMind layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import logging
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Generic, Mapping, Protocol, TypeVar
import weakref

from ..infra.host_bridge import get_host_runtime_home
from ..infra.textutil import _bool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .source_intake import SourceIntakeVerdict, SourceMaterial
    from ..runtime.delivery import DeliveryHistory
    from ..runtime.provider import SanaeMemProvider


class SourceIntakeSink(Protocol):
    def __call__(
        self,
        material: SourceMaterial | Mapping[str, Any],
        *,
        session_id: str,
        create_memory_event: bool,
        importance: float | None,
    ) -> dict[str, Any]: ...


class RegisteredIntake(Protocol):
    def __call__(
        self,
        provider: SanaeMemProvider,
        material: SourceMaterial | Mapping[str, Any],
        /,
        *,
        session_id: str,
        create_memory_event: bool,
        importance: float | None,
    ) -> dict[str, Any]: ...


class LegacySourceIntakeSink(Protocol):
    def __call__(
        self,
        material: SourceMaterial | Mapping[str, Any],
        *,
        session_id: str,
        create_memory_event: bool,
        importance: float | None,
    ) -> SourceIntakeVerdict: ...


@dataclass(frozen=True)
class _IntakeRegistration:
    provider: weakref.ReferenceType[SanaeMemProvider]
    intake: RegisteredIntake


@dataclass(frozen=True)
class _DeliveryRegistration:
    owner: object
    discovery: bool
    history: DeliveryHistory


Key = TypeVar("Key")
Value = TypeVar("Value")


@dataclass(frozen=True, eq=False)
class _Registration(Generic[Value]):
    value: Value


class _ScopedRegistrations(Generic[Key, Value]):
    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[Key, list[_Registration[Value]]] = {}

    def add(self, key: Key, value: Value) -> Callable[[], None]:
        registration = _Registration(value)
        with self._lock:
            self._entries.setdefault(key, []).append(registration)

        def release() -> None:
            with self._lock:
                stack = self._entries.get(key)
                if stack is not None and registration in stack:
                    stack.remove(registration)
                    if not stack:
                        del self._entries[key]

        return release

    def values(self, key: Key) -> tuple[Value, ...]:
        with self._lock:
            return tuple(entry.value for entry in reversed(self._entries.get(key, ())))


_providers = _ScopedRegistrations[tuple[Path, str], weakref.ReferenceType["SanaeMemProvider"]]()
_sinks = _ScopedRegistrations[Path, _IntakeRegistration]()
_explicit_sinks: dict[Path, LegacySourceIntakeSink] = {}
_explicit_sink_lock = RLock()
_deliveries = _ScopedRegistrations[Path, _DeliveryRegistration]()
_delivery_lock = RLock()


def register_active_provider(
    home: Path, session_id: str, provider: SanaeMemProvider,
) -> Callable[[], None]:
    release = _providers.add((home.resolve(), session_id), weakref.ref(provider))
    return weakref.finalize(provider, release)


def resolve_active_provider(home: Path | str, session_id: str) -> SanaeMemProvider | None:
    for reference in _providers.values((Path(home).resolve(), session_id)):
        provider = reference()
        if provider is not None and provider.is_active_session(session_id) and not provider.shadow_enabled():
            return provider
    return None


def register_delivery_owner(
    home: Path, owner: object, discovery: bool, history: DeliveryHistory,
) -> tuple[DeliveryHistory, Callable[[], None]]:
    with _delivery_lock:
        entries = _deliveries.values(home)
        if entries:
            history = entries[0].history
        release_registration = _deliveries.add(home, _DeliveryRegistration(owner, discovery, history))

    def release() -> None:
        with _delivery_lock:
            release_registration()

    return history, release


def owns_delivery(home: Path, owner: object) -> bool:
    entries = _deliveries.values(home)
    if not entries:
        return False
    selected = next((entry for entry in entries if entry.discovery), entries[0])
    return selected.owner is owner


def register_source_intake_sink(
    home: Path, provider: SanaeMemProvider, sink: RegisteredIntake,
) -> Callable[[], None]:
    release = _sinks.add(home.resolve(), _IntakeRegistration(weakref.ref(provider), sink))
    return weakref.finalize(provider, release)


def resolve_source_intake_sink(home: Path) -> SourceIntakeSink | None:
    with _explicit_sink_lock:
        sink = _explicit_sinks.get(home.resolve())
    if sink is not None:
        def submit(
            material: SourceMaterial | Mapping[str, Any],
            *,
            session_id: str,
            create_memory_event: bool,
            importance: float | None,
        ) -> dict[str, Any]:
            return sink(material, session_id=session_id, create_memory_event=create_memory_event,
                        importance=importance).to_dict()

        return submit
    for entry in _sinks.values(home.resolve()):
        provider = entry.provider()
        if provider is not None:
            return partial(entry.intake, provider)
    return None


def register_explicit_source_sink(home: Path, sink: LegacySourceIntakeSink) -> None:
    with _explicit_sink_lock:
        _explicit_sinks[home.resolve()] = sink


def unregister_explicit_source_sink(home: Path, sink: LegacySourceIntakeSink | None) -> None:
    with _explicit_sink_lock:
        key = home.resolve()
        if sink is None or _explicit_sinks.get(key) is sink:
            _explicit_sinks.pop(key, None)


@dataclass(frozen=True, eq=False)
class _ScheduleBinding:
    read_config: Callable[[Path], dict[str, Any]]
    sync: Callable[[dict[str, Any]], None]


@dataclass
class _RegisteredSchedule:
    bindings: list[_ScheduleBinding] = field(default_factory=list)
    applied_shadow: bool | None = None
    syncing: bool = False


_registered_schedules: dict[Path, _RegisteredSchedule] = {}
_schedule_lock = RLock()


def register_schedule(
    home: Path, read_config: Callable[[Path], dict[str, Any]], sync: Callable[[dict[str, Any]], None],
) -> Callable[[], None]:
    resolved_home = home.resolve()
    binding = _ScheduleBinding(read_config, sync)
    with _schedule_lock:
        schedule = _registered_schedules.setdefault(resolved_home, _RegisteredSchedule())
        schedule.bindings.append(binding)
        schedule.applied_shadow = None

    def release() -> None:
        with _schedule_lock:
            if binding in schedule.bindings:
                if schedule.bindings[-1] is binding:
                    schedule.applied_shadow = None
                schedule.bindings.remove(binding)
                if not schedule.bindings:
                    _registered_schedules.pop(resolved_home, None)

    return release


def reconcile_registered_schedule(home: Path) -> None:
    resolved_home = home.resolve()
    if get_host_runtime_home().resolve() != resolved_home:
        return
    with _schedule_lock:
        schedule = _registered_schedules.get(resolved_home)
        if schedule is None or schedule.syncing:
            return
        binding = schedule.bindings[-1]
        config = binding.read_config(resolved_home)
        shadow = _bool(config.get("shadow"), False)
        if schedule.applied_shadow == shadow:
            return
        schedule.syncing = True
        try:
            binding.sync(config)
        except Exception:
            schedule.applied_shadow = None
            logger.exception("sanae_mem: could not synchronize scheduled tasks")
        else:
            if schedule.bindings and schedule.bindings[-1] is binding:
                schedule.applied_shadow = shadow
        finally:
            schedule.syncing = False
