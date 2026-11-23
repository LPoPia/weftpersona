"""Lightweight shared contracts, base exceptions, and immutability helpers for ViveMind."""

from __future__ import annotations

from collections.abc import Mapping
import random
import time
from types import MappingProxyType
from typing import Any, Optional, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")

DEFAULT_CAS_MAX_RETRIES = 32


class ViveMindError(Exception):
    """Base exception for explicit ViveMind domain and runtime errors."""


class CasConflictError(ViveMindError, RuntimeError):
    """Raised when optimistic CAS retries are exhausted under concurrent contention."""


def cas_backoff_sleep(
    attempt: int,
    *,
    base_delay: float = 0.002,
    max_delay: float = 0.05,
) -> float:
    """Sleep for an exponentially backed-off, equal-jitter duration and return seconds slept."""
    cap = min(max_delay, base_delay * (2 ** max(0, int(attempt))))
    delay = random.uniform(cap * 0.5, cap)
    if delay > 0.0:
        time.sleep(delay)
    return delay


EMPTY_MAPPING: Mapping[Any, Any] = MappingProxyType({})


def empty_mapping() -> Mapping[Any, Any]:
    """Return the shared read-only empty mapping (safe as a dataclass default_factory)."""
    return EMPTY_MAPPING


def freeze_value(value: Any) -> Any:
    """Recursively copy and freeze mutable containers into read-only equivalents."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    return value


def freeze_mapping(value: Optional[Mapping[_K, _V]]) -> Mapping[_K, _V]:
    """Return an isolated read-only view of *value* (recursive for nested containers)."""
    if value is None:
        return EMPTY_MAPPING
    if not isinstance(value, Mapping):
        raise TypeError(f"expected Mapping, got {type(value).__name__}")
    return MappingProxyType({key: freeze_value(item) for key, item in value.items()})


def thaw_json(value: Any) -> Any:
    """Convert frozen Mapping/tuple/set containers back into plain JSON-serializable values."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [thaw_json(item) for item in value]
    return value
