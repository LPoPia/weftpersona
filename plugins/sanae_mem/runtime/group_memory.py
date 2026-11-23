"""Long-term memory scope for one trusted multi-person scene.

A group row is an ordinary ``events`` row whose ``owner_key`` starts with ``group:``. The owner
namespace is the only new fact: the key itself is built from Task 03's ``group_scope_key`` /
``conversation_scope_key`` (no second parser, no duplicated column), and the switch is Task 03's
``group_feature_enabled(..., 'long_term_memory')`` read from the trusted runtime config.

Ownership rules, all derived from the already-resolved ``ChatScope``:

* a plain group, and a thread the host only synthesized for its own reply, write the shared group
  owner ``group:<group key>``;
* a real thread with an explicit ``inherited`` audience also writes that shared owner;
* any other real thread (``narrower``/``unknown``) writes its exact ``group:<conversation key>``.

Reading is never wider than the scene: a plain group and a synthesized reply thread read the
shared owner; a real ``inherited`` or ``narrower`` thread reads the shared owner plus its own
exact thread owner (so an inherited thread keeps rows it previously wrote under the exact key);
an ``unknown`` thread reads only its exact owner. Nothing guesses a thread's audience from the
existence of a parent channel, and a narrow thread never flows back into the group, a sibling
thread, a private chat or another group. ``source`` always keeps the original chat/thread ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Tuple

from ..governance.group_scope import (
    GROUP_OWNER_PREFIX as GROUP_OWNER_PREFIX,
    GROUP_TOOL_NAMES as GROUP_TOOL_NAMES,
    GroupGrant as GroupGrant,
    LONG_TERM_FEATURE as LONG_TERM_FEATURE,
    conversation_scope_key,
    exclude_group_rows as exclude_group_rows,
    group_feature_enabled,
    group_scope_key,
    is_group_owner as is_group_owner,
    member_impression_owner as member_impression_owner,
    member_impression_read_owners as member_impression_read_owners,
)
from ..governance.source_context import (
    PUBLIC_CHAT_TYPES, ChatScope, group_source_record, parse_source, resolve_chat_identity,
)


def _managed_scene(scope: Any) -> bool:
    if not isinstance(scope, ChatScope):
        return False
    if scope.visibility != "public":
        return False
    return str(scope.chat_type or "").strip().lower() in PUBLIC_CHAT_TYPES


def group_grant_for(
    scope: Any, config: Optional[Mapping[str, Any]], *, connection: Any = None,
) -> Optional[GroupGrant]:
    """Resolve one grant, or ``None`` when this scene may not use group long-term memory.

    The audience comes from the platform-stated scope only; ``config`` is the live plugin config
    of the current trusted runtime, never a copy frozen at enqueue time.
    """
    if not _managed_scene(scope):
        return None
    scope_key = group_scope_key(scope)
    conversation_key = conversation_scope_key(scope)
    if not scope_key or not conversation_key:
        return None
    if not group_feature_enabled(config, scope, LONG_TERM_FEATURE):
        return None
    shared_owner = GROUP_OWNER_PREFIX + scope_key
    exact_owner = GROUP_OWNER_PREFIX + conversation_key
    real_thread = bool(scope.thread_id) and not scope.synthetic_reply_thread
    inherited = real_thread and scope.thread_visibility == "inherited"
    write_owner = exact_owner if real_thread and not inherited else shared_owner
    # Only a thread whose platform stated an audience reads the parent's shared row: an
    # ``inherited`` thread is the whole group, and a ``narrower`` one was explicitly cut from it.
    # An ``unknown`` thread reads just its exact owner; the parent id alone is not a read grant.
    if not real_thread:
        read_owners = (shared_owner,)
    elif scope.thread_visibility == "unknown":
        read_owners = (exact_owner,)
    else:
        read_owners = tuple(sorted({shared_owner, exact_owner}))
    identity = resolve_chat_identity(scope, config, connection=connection)
    return GroupGrant(
        scope_key=scope_key,
        conversation_key=conversation_key,
        shared_owner=shared_owner,
        exact_owner=exact_owner,
        write_owner=write_owner,
        read_owners=read_owners,
        person_key=identity.person_key,
        source=group_source_record(scope) or {},
        thread_visibility=str(scope.thread_visibility or ""),
    )


def current_group_grant(
    config: Optional[Mapping[str, Any]], scope: Any = None, *, connection: Any = None,
) -> Optional[GroupGrant]:
    """Grant for the scene of *this* turn; ``scope`` may be the one the caller already resolved."""
    if scope is None:
        from ..governance.source_context import current_chat_scope
        scope = current_chat_scope()
    return group_grant_for(scope, config, connection=connection)


def is_managed_group_scene(scope: Any) -> bool:
    """True for a public scene whose group key is provable, enabled or not.

    The provider uses this to tell "this turn is a group scene and the switch is off" (no private
    read at all) from "this provider has no group wiring" (a bare storage caller).
    """
    return _managed_scene(scope) and group_scope_key(scope) is not None


def group_read_clause(grant: Optional[GroupGrant], column: str = "owner_key") -> Tuple[str, list]:
    """SQL predicate limiting a read to the owners this scene may see."""
    owners = list(grant.read_owners) if grant else []
    if not owners:
        return (exclude_group_rows(column), [])
    return ("(%s IN (%s))" % (column, ",".join("?" for _ in owners)), owners)


def group_write_owner(grant: Optional[GroupGrant]) -> str:
    return grant.write_owner if grant else ""


def _stored_group_scope(record: Any) -> Optional[ChatScope]:
    """Rebuild the scene a stored group row was written in, from its own source JSON only."""
    if not isinstance(record, dict):
        return None
    platform = str(record.get("platform") or "")
    chat_type = str(record.get("chat_type") or "")
    if not platform or not chat_type:
        # No stated scene: the switch cannot be re-resolved, so the row stays untouched.
        return None
    return ChatScope(
        "public", platform,
        chat_type=chat_type,
        chat_id=str(record.get("chat_id") or ""),
        thread_id=str(record.get("thread_id") or ""),
        scope_id=str(record.get("scope_id") or ""),
        parent_chat_id=str(record.get("parent_chat_id") or ""),
        thread_visibility=str(record.get("thread_visibility") or "unknown"),
        profile=str(record.get("profile") or "default"),
        synthetic_reply_thread=str(record.get("synthetic_reply_thread") or "") == "true",
    )


def enabled_group_owners(
    config: Optional[Mapping[str, Any]], scope_key: str = "", *, connection: Any = None,
) -> Tuple[str, ...]:
    """Group owners the current trusted config still lets background work maintain.

    Candidates come from the group rows already stored in this database (their owner plus the
    scene facts in their source column). Each candidate is then re-resolved through the same
    ``group_scope_key`` + ``group_feature_enabled`` used by the turn, so a global default, an
    inherited thread and a per-group override all follow the current HOME's config. A disabled
    group yields no owner and its rows are neither updated nor deleted; no registry table exists
    and no second precedence parser is written here.
    """
    if not isinstance(config, Mapping) or connection is None:
        return ()
    try:
        rows = connection.execute(
            "SELECT DISTINCT owner_key, source_context FROM events "
            "WHERE owner_key LIKE ?", (GROUP_OWNER_PREFIX + "%",),
        ).fetchall()
    except Exception:
        return ()
    owners = set()
    for row in rows:
        owner = str(row[0] or "")
        if not owner:
            continue
        scope = _stored_group_scope(parse_source(row[1]))
        if scope is None:
            continue
        key = group_scope_key(scope)
        if not key or (scope_key and key != scope_key):
            continue
        if not group_feature_enabled(config, scope, LONG_TERM_FEATURE):
            continue
        allowed = {GROUP_OWNER_PREFIX + key}
        conversation = conversation_scope_key(scope)
        if conversation:
            allowed.add(GROUP_OWNER_PREFIX + conversation)
        if owner in allowed:
            owners.add(owner)
    return tuple(sorted(owners))
