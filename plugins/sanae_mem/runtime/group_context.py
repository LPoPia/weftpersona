"""Per-group short-term context settings and the host group-context policy.

The three entries (``short_term_shared``, ``long_term_memory``,
``allow_unmentioned_reply``) resolve independently: an explicit per-group boolean
wins, missing/null/``inherit`` follows ``defaults``, and a missing default is false.
Only platforms whose scenes Task 01 proved trustworthy (QQ, Slack, Discord) take
part; private chats and unknown scenes never do.
"""

from __future__ import annotations

from typing import Any, Mapping
from ..governance.group_scope import (
    GROUP_FEATURES as GROUP_FEATURES,
    _GROUP_CONFIG_KEY as _GROUP_CONFIG_KEY,
    _MANAGED_PLATFORMS as _MANAGED_PLATFORMS,
    _base_key as _base_key,
    _configured as _configured,
    _platform as _platform,
    _scope_chat_id as _scope_chat_id,
    _segment as _segment,
    conversation_scope_key as conversation_scope_key,
    group_feature_enabled as group_feature_enabled,
    group_scope_key as group_scope_key,
)
from ..governance.source_context import ChatScope, chat_scope_for


class _SourceEnvelope:
    """The host ``SessionSource`` reshaped as a trusted turn envelope for ``chat_scope_for``.

    Scene normalization lives in one place: the host states ``scene_chat_type`` when its transport
    field was ambiguous (Slack), and that overrides ``chat_type`` exactly as it does for the turn
    envelope. Ids are never parsed from text or names.
    """

    def __init__(self, source: Any, profile: str | None) -> None:
        self.platform = str(
            getattr(getattr(source, "platform", ""), "value", getattr(source, "platform", "")) or "")
        self.chat_type = str(
            getattr(source, "scene_chat_type", "") or getattr(source, "chat_type", "") or "")
        self.chat_id = str(getattr(source, "chat_id", "") or "")
        self.thread_id = str(getattr(source, "thread_id", "") or "")
        self.sender_id = str(getattr(source, "user_id", "") or "")
        self.sender_id_alt = str(getattr(source, "user_id_alt", "") or "")
        self.message_id = str(getattr(source, "message_id", "") or "")
        self.session_key = ""
        self.session_id = ""
        self.scope_id = str(getattr(source, "scope_id", "") or "")
        self.parent_chat_id = str(getattr(source, "parent_chat_id", "") or "")
        self.reply_to_message_id = str(getattr(source, "reply_to_message_id", "") or "")
        self.occurred_at = str(getattr(source, "occurred_at", "") or "")
        self.inbound_coverage = str(getattr(source, "inbound_coverage", "") or "unknown")
        self.is_bot = getattr(source, "is_bot", None)
        self.thread_visibility = str(getattr(source, "thread_visibility", "") or "unknown")
        self.profile = str(profile or getattr(source, "profile", "") or "")
        self.synthetic_reply_thread = getattr(source, "synthetic_reply_thread", False) is True


def _scope_from_source(source: Any, profile: str | None) -> ChatScope:
    return chat_scope_for(_SourceEnvelope(source, profile))


class ViveMindGroupContextPolicy:
    """Answer the host's shared-vs-per-member short-term context question.

    ``True`` shares the group/thread history, ``False`` keeps member history, and
    ``None`` declines platforms/scenes the plugin does not govern.
    """

    def __init__(self, config_getter):
        self._config_getter = config_getter

    def __call__(self, source: Any, *, profile: str | None = None) -> bool | None:
        scope = _scope_from_source(source, profile)
        if scope.visibility != "public" or not _platform(scope.platform):
            return None
        key = group_scope_key(scope)
        if key is None:
            # A group on a managed platform whose domain/parent we cannot verify: isolate members
            # rather than fall back to a shared default we cannot scope.
            return False
        config = self._config_getter()
        return group_feature_enabled(config if isinstance(config, Mapping) else {}, scope, "short_term_shared")
