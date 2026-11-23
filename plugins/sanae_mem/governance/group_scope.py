"""Pure group scope, feature switch, and member impression owner helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from urllib.parse import quote

from ..infra.contracts import freeze_mapping
from ..infra.event_fields import (
    GROUP_OWNER_PREFIX as GROUP_OWNER_PREFIX,
    exclude_group_rows as exclude_group_rows,
    is_group_owner as is_group_owner,
)
from .source_context import PUBLIC_CHAT_TYPES, ChatScope

GROUP_FEATURES = ("short_term_shared", "long_term_memory", "allow_unmentioned_reply")
# Each managed platform's domain comes from a platform-stated fact: QQ's app id, Slack's workspace,
# Discord's guild. Without it a group id is only unique inside a domain we cannot see.
_MANAGED_PLATFORMS = {"qq": "qqbot", "qqbot": "qqbot", "slack": "slack", "discord": "discord"}
_GROUP_CONFIG_KEY = "group_context"

LONG_TERM_FEATURE = "long_term_memory"
# The one tool surface a switched-on trusted group opens in-turn: search only.
# Writes happen off-turn from confirmed delivery receipts.
# The tool gate, the registered schemas and the prompt policy all read this same set.
GROUP_TOOL_NAMES = frozenset({"sanae_mem_search"})


def _segment(value: Any) -> str:
    return quote(str(value or "").strip(), safe="")


def _platform(value: str) -> str:
    return _MANAGED_PLATFORMS.get(str(value or "").strip().lower(), "")


def _scope_chat_id(scope: ChatScope) -> str:
    """The group a scope belongs to: its parent channel for a real thread, the channel/chat
    otherwise, and the parent channel again when the thread is only the host's reply landing spot."""
    if scope.thread_id or scope.synthetic_reply_thread:
        return scope.parent_chat_id
    return scope.chat_id


def _base_key(scope: ChatScope) -> tuple[str, str] | None:
    """``(key, platform)`` for a verified group scene; None for anything else (private, unknown,
    unmanaged platform, or a missing domain/parent)."""
    if scope.visibility != "public":
        return None
    if str(scope.chat_type or "").strip().lower() not in PUBLIC_CHAT_TYPES:
        return None
    platform = _platform(scope.platform)
    chat_id = str(_scope_chat_id(scope) or "").strip()
    if not platform or not chat_id:
        return None
    if not str(scope.scope_id or "").strip():
        return None
    if (scope.thread_id or scope.synthetic_reply_thread) and not str(scope.parent_chat_id or "").strip():
        return None
    profile = str(scope.profile or "").strip() or "default"
    return (
        "/".join((_segment(profile), _segment(platform), _segment(scope.scope_id), _segment(chat_id))),
        platform,
    )


def group_scope_key(scope: ChatScope) -> str | None:
    """Settings scope for one group/channel; None when the domain is not trusted."""
    base = _base_key(scope)
    return base[0] if base else None


def conversation_scope_key(scope: ChatScope) -> str | None:
    """Exact group or real-thread scope: the group key plus ``/threads/<id>``.

    A thread the host synthesized for its own reply is not a conversation scope of its own: it
    resolves to the group it was opened in."""
    base = _base_key(scope)
    if base is None:
        return None
    if scope.thread_id and not scope.synthetic_reply_thread:
        return f"{base[0]}/threads/{_segment(scope.thread_id)}"
    return base[0]


def _configured(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def group_feature_enabled(
    config: Mapping[str, Any] | None, scope: ChatScope, feature: str,
) -> bool:
    """Resolve one group feature; per-group boolean wins, else defaults, else false."""
    if feature not in GROUP_FEATURES:
        return False
    key = group_scope_key(scope)
    if key is None:
        return False
    mapping = config.get(_GROUP_CONFIG_KEY) if isinstance(config, Mapping) else None
    if not isinstance(mapping, Mapping):
        return False
    overrides = mapping.get("overrides")
    if isinstance(overrides, Mapping):
        explicit = _configured((overrides.get(key) or {}).get(feature)) if isinstance(overrides.get(key), Mapping) else None
        if explicit is not None:
            return explicit
    defaults = mapping.get("defaults")
    default = _configured(defaults.get(feature)) if isinstance(defaults, Mapping) else None
    return bool(default)


@dataclass(frozen=True)
class GroupGrant:
    """What the current trusted runtime may do for one scene, resolved per call."""

    scope_key: str
    conversation_key: str
    shared_owner: str
    exact_owner: str
    write_owner: str
    read_owners: Tuple[str, ...]
    person_key: str
    source: Mapping[str, str]
    thread_visibility: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "read_owners", tuple(self.read_owners))
        object.__setattr__(self, "source", freeze_mapping(self.source))

    @property
    def shared_scope(self) -> bool:
        """True when this scene's material belongs to the whole group, not only one thread."""
        return self.write_owner == self.shared_owner

    @property
    def resolution_owners(self) -> Tuple[str, ...]:
        """Owners of promises already recorded here that this scene may fulfil or cancel.

        A plain group and a thread the host only synthesized for its reply own the shared
        scene. An ``inherited`` thread continues the promise it filed under its own exact key
        while its audience was still ``unknown``, so both owners stay resolvable there. A
        ``narrower`` or ``unknown`` thread may only touch its own exact one, which keeps the
        parent's shared promise readable but not writable.

        This is not a read grant and it never widens new material: :attr:`write_owner` still
        decides where a new event or a new promise goes.
        """
        if self.exact_owner == self.shared_owner:
            return (self.shared_owner,)
        if self.thread_visibility == "unknown":
            return (self.exact_owner,)
        if self.thread_visibility == "inherited":
            return tuple(sorted({self.shared_owner, self.exact_owner}))
        return (self.exact_owner,)


def member_impression_owner(grant: Optional[GroupGrant], owner: str = "") -> str:
    """The impression row of one member inside one scene: ``<group owner>#<person>``.

    The person label is Task 03's trusted ``person_key``; the private chat keeps its own row
    under the private owner, so the same person in two groups or in a private chat stays apart.
    """
    if grant is None or not grant.person_key:
        return ""
    return f"{owner or grant.write_owner}#{grant.person_key}"


def member_impression_read_owners(grant: Optional[GroupGrant]) -> Tuple[str, ...]:
    """Rows of *that same member* this scene may read, first the one it writes.

    Every owner this scene may already read maps to that same member's row, so the write owner's
    row comes first and the granted read owners follow, deduplicated. A thread that reads its
    parent also sees the parent group's row about that person, and an ``inherited`` thread still
    finds the exact row it wrote while its audience was ``unknown`` or ``narrower``. No owner
    outside ``grant.read_owners`` is added: another member, a sibling thread, a private chat or a
    disabled parent stays unreadable.
    """
    if grant is None or not grant.person_key:
        return ()
    owners = [member_impression_owner(grant, grant.write_owner)]
    for owner in grant.read_owners:
        row = member_impression_owner(grant, owner)
        if row and row not in owners:
            owners.append(row)
    return tuple(owners)
