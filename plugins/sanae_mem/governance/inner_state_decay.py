"""Pure inner-state slow-variable decay and recent-words TTL helpers."""

from __future__ import annotations

from ..infra.clock import sleep_day_date
from ..storage.inner_state_store import InnerState

_HOUR = 3600

_HALF_LIVES = {"tension": 3 * _HOUR, "closeness": 5 * _HOUR, "energy": 6 * _HOUR}

_MIN_SLEEP = 5 * _HOUR

_WORDS_TTL = 12 * _HOUR

_WORDS_AFTER_SLEEP_TTL = 8 * _HOUR


def _slept_over(earlier: float, now: float) -> bool:
    return now - earlier >= _MIN_SLEEP and sleep_day_date(now) > sleep_day_date(earlier)


def _slow_variables(state: InnerState | None, now: float) -> dict[str, float]:
    if state is None:
        return {key: 0.0 for key in _HALF_LIVES}
    elapsed = max(0.0, now - state.updated_at)
    values = {
        key: state.variables[key] * 2 ** (-elapsed / half_life)
        for key, half_life in _HALF_LIVES.items()
    }
    if values["energy"] < 0 and _slept_over(state.felt_at, now):
        values["energy"] = 0.0
    return values


def _live_words(state: InnerState | None, now: float) -> str:
    if state is None or not state.recent_words:
        return ""
    said = state.recent_words_at if state.recent_words_at is not None else state.felt_at
    elapsed = max(0.0, now - said)
    limit = _WORDS_AFTER_SLEEP_TTL if _slept_over(said, now) else _WORDS_TTL
    return state.recent_words if elapsed < limit else ""


def _kept_words(state: InnerState | None, now: float) -> tuple[str, float | None]:
    words = _live_words(state, now)
    if state is None or not words:
        return "", None
    said = state.recent_words_at if state.recent_words_at is not None else state.felt_at
    return words, said
