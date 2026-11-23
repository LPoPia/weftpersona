from __future__ import annotations

import time
from pathlib import Path

from ..governance.inner_state_decay import (
    _HALF_LIVES as _HALF_LIVES,
    _HOUR as _HOUR,
    _MIN_SLEEP as _MIN_SLEEP,
    _WORDS_AFTER_SLEEP_TTL as _WORDS_AFTER_SLEEP_TTL,
    _WORDS_TTL as _WORDS_TTL,
    _kept_words as _kept_words,
    _live_words as _live_words,
    _slept_over as _slept_over,
    _slow_variables as _slow_variables,
)
from .inner_state_spark import _strip_unspoken_note
from .inner_state_store import read_state
from .inner_state_types import InnerState, InnerStateTarget, InnerStateView

INNER_STATE_MARKER = "[我此刻心里]"

_SLEEP_ECHO = 0.10

_SLEEP_SETTLE_FLOOR = 0.30

_SLEEP_DIRECTION_RETENTION = 0.50

_SLEEP_THOUGHT_TTL = 24 * _HOUR

_ECHO = 0.15

_THOUGHT_TTL = 16 * _HOUR


def _afterglow(state: InnerState, now: float, *, slept: bool) -> str:
    view = _slow_variables(state, now)
    if slept:
        view = {key: max((value, state.variables[key] * _SLEEP_DIRECTION_RETENTION), key=abs)
                for key, value in view.items()}
    tension, closeness = view["tension"], view["closeness"]
    if tension >= _ECHO and tension >= closeness:
        return ("睡了一觉，那股绷着的劲总算松下去了，心里还留着一点没散尽的余味。" if slept
                else "先前那阵心思淡下去了，绷着的那点劲还没全松，只留一点余味。")
    if closeness >= _ECHO:
        return ("睡了一觉，先前那些情绪已经平下去，心里还温着一点想靠近他的余味。" if slept
                else "先前那阵心思淡下去了，心里还留着一点想靠近他的温软余味。")
    if closeness <= -_ECHO:
        return ("睡了一觉，气消下去一些，心里那点还没顺过来的余味还留着。" if slept
                else "先前那阵心思淡下去了，心里那点没顺过来的余味还留着。")
    return ("睡了一觉，先前那阵心思已经平下去，只留一点余味。" if slept
            else "先前那阵心思已经淡下去了，只留一点余味。")

def _thought_at(state: InnerState, now: float) -> str:
    if not state.text:
        return ""
    elapsed = max(0.0, now - state.felt_at)
    slept = _slept_over(state.felt_at, now)
    if elapsed >= (_SLEEP_THOUGHT_TTL if slept else _THOUGHT_TTL):
        return ""
    if slept:
        view = _slow_variables(state, now)
        peak = max(abs(state.variables["tension"]), abs(state.variables["closeness"]))
        if (peak < _SLEEP_SETTLE_FLOOR
                and max(abs(view["tension"]), abs(view["closeness"])) < _SLEEP_ECHO):
            return ""
        return _afterglow(state, now, slept=True)
    strength = max(abs(value) for value in state.variables.values())
    half_life = (2 + 2 * strength) * _HOUR
    if elapsed < half_life:
        return state.text
    clean = _strip_unspoken_note(state.text)
    if not clean:
        return ""
    if elapsed < 2 * half_life:
        return f"早先我心里还转着这个念头：「{clean}」过了这一阵，这股劲已经淡些了。"
    return _afterglow(state, now, slept=False)

def _visible_thoughts(view: InnerStateView, now: float) -> list[tuple[InnerState, str]]:
    segments: list[tuple[InnerState, str]] = []
    if view.shared is not None:
        shared = _thought_at(view.shared, now)
        if shared:
            segments.append((view.shared, shared))
    if view.private is not None:
        private = _thought_at(view.private, now)
        if private and all(private != text for _state, text in segments):
            segments.append((view.private, private))
    return segments


def render_inner_state(
    database: Path, target: InnerStateTarget, state: InnerState | None = None, *,
    now: float, session_id: str = "", owner_key: str = "",
    internal: bool = False,
) -> str:
    if state is not None:
        thought = _thought_at(state, now)
        segments = [(state, thought)] if thought else []
    else:
        view = read_state(
            database, target[1], owner_key=owner_key, internal=internal)
        segments = _visible_thoughts(view, now) if view is not None else []
    if not segments:
        return ""
    rendered = [text for _segment, text in segments if text]
    if not rendered:
        return ""
    return "\n".join(part for part in (
        INNER_STATE_MARKER,
        "这是我上一轮过后心里留着的一点底色（不是他这轮新发的图或刚说的话），可以影响我留意什么、怎么说，不必说出来。",
        "\n".join(rendered),
    ) if part)
