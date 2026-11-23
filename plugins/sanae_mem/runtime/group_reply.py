"""Non-addressed group participation: the switch and the per-group start cadence.

The settings live in the same ``group_context`` mapping as the short-term context switches, but
``allow_unmentioned_reply`` resolves on its own (an explicit per-group boolean, else ``defaults``,
else false). ``unmentioned_interval_seconds`` is a plain positive number of seconds — 0 means "no
wait" — with 30 as the default; per-group values win over defaults.

The host owns what this policy cannot see: whether the transport actually addressed this bot, the
trusted message order, the start quota and the send. This module only answers for verified group
scenes it governs; anything else returns None and keeps the host's own behaviour.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .group_context import (
    GROUP_FEATURES,
    _platform,
    _scope_from_source,
    conversation_scope_key,
    group_feature_enabled,
    group_scope_key,
)

FEATURE = "allow_unmentioned_reply"
INTERVAL_KEY = "unmentioned_interval_seconds"
DEFAULT_INTERVAL_SECONDS = 30.0
_GROUP_CONFIG_KEY = "group_context"


def _interval(value: Any) -> float:
    """A finite non-negative number of seconds; anything else takes the default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_INTERVAL_SECONDS
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        return DEFAULT_INTERVAL_SECONDS
    return seconds


def unmentioned_interval_seconds(config: Mapping[str, Any] | None, scope: Any) -> float:
    """This group's non-addressed start interval: per-group value, else the default, else 30s."""
    key = group_scope_key(scope)
    mapping = config.get(_GROUP_CONFIG_KEY) if isinstance(config, Mapping) else None
    if key is None or not isinstance(mapping, Mapping):
        return DEFAULT_INTERVAL_SECONDS
    overrides = mapping.get("overrides")
    if isinstance(overrides, Mapping):
        group = overrides.get(key)
        if isinstance(group, Mapping) and INTERVAL_KEY in group:
            return _interval(group.get(INTERVAL_KEY))
    defaults = mapping.get("defaults")
    if isinstance(defaults, Mapping) and INTERVAL_KEY in defaults:
        return _interval(defaults.get(INTERVAL_KEY))
    return DEFAULT_INTERVAL_SECONDS


class ViveMindGroupReplyPolicy:
    """Answer the host's managed-group participation question for one verified group scene.

    ``None`` declines platforms/scenes the plugin does not govern, so the host keeps its own
    mention rule there.
    """

    def __init__(self, config_getter, decision_factory):
        self._config_getter = config_getter
        self._decision_factory = decision_factory

    def __call__(self, source: Any, *, profile: str | None = None) -> Any:
        scope = _scope_from_source(source, profile)
        if scope.visibility != "public" or not _platform(scope.platform):
            return None
        group_key = group_scope_key(scope)
        conversation_key = conversation_scope_key(scope)
        if group_key is None or conversation_key is None:
            return None
        config = self._config_getter()
        config = config if isinstance(config, Mapping) else {}
        return self._decision_factory(
            group_key=group_key,
            conversation_key=conversation_key,
            allow_unmentioned_reply=group_feature_enabled(config, scope, FEATURE),
            unmentioned_interval_seconds=unmentioned_interval_seconds(config, scope),
        )
