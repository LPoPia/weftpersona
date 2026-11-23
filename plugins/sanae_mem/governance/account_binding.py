"""Maintainer account bindings: one person, several platform accounts, one owner anchor.

A binding is a *declared* account inside the platform fact the adapter actually stated:
``(profile, platform, scope, account)``. Nothing here reads a display name, a chat id or
message prose: an account only joins a person when an operator wrote it into
``maintainer_account_bindings`` (or into the same trusted config list).

Confirmed bindings are persisted (schema v4) so the same declaration can be replayed and the
association survives without re-reading the config on every turn.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from .identity_config import read_memory_identity
from ..infra.chat_scope import ChatScope, current_chat_scope
from ..infra.contracts import ViveMindError, freeze_mapping
from ..infra import host_bridge

# The first batch of platforms with a clear account namespace. Unknown platforms are ignored
# rather than guessed: a new platform needs its own stable scope fact before it can bind.
SUPPORTED_PLATFORMS = frozenset({"slack", "discord", "qq", "qqbot"})
# A platform whose account IDs are only unique inside a stated scope must state that scope.
SCOPED_PLATFORMS = frozenset({"slack", "qq", "qqbot"})
DEFAULT_PROFILE = "default"

BindingKey = Tuple[str, str, str, str]  # profile, platform, account_id, scope_id


class AccountBindingConflict(ViveMindError, RuntimeError):
    """An account is already confirmed for another owner; nothing in this batch is applied."""


class Binding(NamedTuple):
    profile: str
    platform: str
    account_id: str
    scope_id: str
    person_owner_key: str

    @property
    def key(self) -> BindingKey:
        return (self.profile, self.platform, self.account_id, self.scope_id)


def normalize_profile(value: Any) -> str:
    """A missing profile is the trusted active/default profile, never a wildcard."""
    return str(value or "").strip() or DEFAULT_PROFILE


def account_scope(platform: Any, scope_id: Any) -> Optional[str]:
    """The platform fact an account ID lives inside, or ``None`` when a required one is absent.

    Discord user snowflakes are global and deliberately do not split by guild; Slack team IDs and
    QQ Bot app IDs are part of the account identity.
    """
    name = str(platform or "").strip().lower()
    scope = str(scope_id or "").strip()
    if name in SCOPED_PLATFORMS:
        return scope or None
    return ""


def _account_key(platform: str, user_id: str, user_id_alt: str = "") -> str:
    """The platform's own unscoped account key, or ``""`` when it states no account."""
    from ..infra.textutil import _owner_key
    return _owner_key(platform, user_id=user_id, user_id_alt=user_id_alt)


def person_key(owner_key: Any) -> str:
    """Stable person label for an owner anchor; a separate namespace from the storage owner."""
    value = str(owner_key or "").strip()
    if not value:
        return ""
    return "person:" + hashlib.sha256(("sanae-person:" + value).encode("utf-8")).hexdigest()[:16]


def current_profile(hermes_home: str | Path | None = None, scope: ChatScope | None = None) -> str:
    """Resolve the active profile name from scope, hermes_home path, or default."""
    if scope is None:
        scope = current_chat_scope()
    scoped = str(scope.profile or "").strip()
    if scoped:
        return normalize_profile(scoped)
    if hermes_home is not None:
        home = Path(hermes_home)
    else:
        home = host_bridge.get_hermes_home()
    if home.parent.name == "profiles" and home.name:
        return normalize_profile(home.name)
    return DEFAULT_PROFILE


def _binding_keys(raw: Any) -> List[BindingKey]:
    if not isinstance(raw, dict):
        return []
    platform = str(raw.get("platform") or "").strip().lower()
    primary = str(raw.get("user_id") or "").strip()
    alternate = str(raw.get("user_id_alt") or "").strip()
    scope_id = str(raw.get("scope_id") or raw.get("app_id") or "").strip()
    if platform not in SUPPORTED_PLATFORMS or not (primary or alternate):
        return []
    scope = account_scope(platform, scope_id)
    if scope is None:
        # The same ID exists in every workspace/app: without the stated scope this is not an account.
        return []
    profile = normalize_profile(raw.get("profile"))
    # Both spellings are explicit stable facts for the same adapter. Either name is usable, but
    # nothing is inferred when the operator supplied only one of them.
    return [(profile, platform, account_id, scope)
            for account_id in sorted({primary, alternate} - {""})]


def configured_bindings(config: Any, *, profile: Optional[str] = None) -> List[BindingKey]:
    """Declared account keys; duplicates collapse to one deterministic entry.

    A malformed or unsupported entry is dropped instead of guessed: it grants nothing.
    """
    raw = (config or {}).get("maintainer_account_bindings")
    if not isinstance(raw, (list, tuple)):
        return []
    keys = {key for item in raw for key in _binding_keys(item)}
    if profile is not None:
        keys = {key for key in keys if key[0] == normalize_profile(profile)}
    return sorted(keys)


def database_path(config: Any = None) -> Path:
    """The same profile's ``sanae.db``: an explicit ``data_dir`` override, else the Home default."""
    configured = str((config or {}).get("data_dir") or "").strip()
    if configured:
        return Path(configured).expanduser() / "sanae.db"
    return host_bridge.get_hermes_home() / "sanae-mem" / "sanae.db"


def stored_bindings(
    connection: Optional[sqlite3.Connection], *, profile: Optional[str] = None,
) -> Optional[Dict[BindingKey, str]]:
    """Confirmed rows and their anchors; ``None`` means the read failed, not "no bindings".

    A file older than v4 simply has no table: nothing was ever confirmed there. Any other read
    error stays distinguishable so no caller can mistake it for an empty mapping and grant.
    """
    if connection is None:
        return {}
    try:
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='maintainer_account_bindings'"
        ).fetchone() is None:
            return {}
        rows = connection.execute(
            "SELECT profile, platform, account_id, scope_id, person_owner_key "
            "FROM maintainer_account_bindings"
        ).fetchall()
    except sqlite3.Error:
        return None
    found = {(str(row[0]), str(row[1]), str(row[2]), str(row[3])): str(row[4])
             for row in rows}
    if profile is not None:
        wanted = normalize_profile(profile)
        found = {key: owner for key, owner in found.items() if key[0] == wanted}
    return found


@dataclass(frozen=True, init=False)
class BindingMap:
    """One read of the effective mapping: confirmed rows plus the same profile's declarations."""

    confirmed: Mapping[BindingKey, str]
    declared: frozenset[BindingKey]

    def __init__(
        self,
        confirmed: Mapping[BindingKey, str],
        declared: Iterable[BindingKey],
    ) -> None:
        object.__setattr__(self, "confirmed", freeze_mapping(confirmed))
        object.__setattr__(self, "declared", frozenset(declared))


def read_mapping(
    config: Any = None, *, connection: Optional[sqlite3.Connection] = None,
    profile: Optional[str] = None,
) -> Optional[BindingMap]:
    """The mapping every ownership branch decides from; ``None`` when the stored side is unreadable.

    Without a provider connection the same profile's database is opened read-only. A file that does
    not exist yet holds no confirmed row; a failing query is reported as such, never as empty.
    """
    declared = frozenset(configured_bindings(config, profile=profile))
    if connection is not None:
        confirmed = stored_bindings(connection, profile=profile)
        return None if confirmed is None else BindingMap(confirmed, declared)
    path = database_path(config)
    if not path.is_file():
        return BindingMap({}, declared)
    try:
        own: Optional[sqlite3.Connection] = sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro", uri=True)
    except (OSError, sqlite3.Error):
        return None
    try:
        with closing(own):
            confirmed = stored_bindings(own, profile=profile)
    except sqlite3.Error:
        return None
    return None if confirmed is None else BindingMap(confirmed, declared)


def _configured_identity(config: Mapping[str, object] | None) -> Tuple[str, str, str, str]:
    """(platform, primary, alternate, explicitly configured scope) of memory_identity."""
    identity = read_memory_identity(config, os.environ)
    return identity.platform.lower(), identity.user_id, identity.user_id_alt, identity.scope_id


class OwnerDecision(NamedTuple):
    """One ownership answer for one trusted scope.

    ``owner`` is the storage owner this account may use. ``blocked`` records that the effective
    mapping names this exact account for someone else: no branch may then fall back to the old
    unscoped anchor, and no other declaration may widen this account's grant. ``fixed`` marks the
    operator's configured ``memory_identity`` account itself: host profile files and the runtime
    workspace follow that marker, never a newly bound memory account.
    """

    owner: str = ""
    source: str = ""  # "binding" (declared/confirmed) or "legacy" (the old unscoped anchor)
    blocked: bool = False
    fixed: bool = False


DECLINED = OwnerDecision()
BLOCKED = OwnerDecision(blocked=True)


def _is_configured_fixed(
    platform: str, actors: Any, scope: str, config: Any, *, trusted_scopes: Any = (),
) -> bool:
    """Whether this stated account is the operator's fixed ``memory_identity`` account.

    The same scope rules ``owner_decision`` applies: an operator-stated app/workspace is the only
    one the fixed account answers for. Slack always needs that stated workspace (a bare ID is an
    account nowhere there). An older QQ ``memory_identity`` without one still names this same
    person, but only inside an app the account was explicitly declared/confirmed for: the same
    openid in any other app is another account. Discord snowflakes are global.
    """
    fixed_platform, fixed_primary, fixed_alternate, fixed_scope = _configured_identity(config)
    if platform != fixed_platform or not set(actors) & ({fixed_primary, fixed_alternate} - {""}):
        return False
    fixed = str(fixed_scope or "").strip()
    if platform == "slack":
        return bool(fixed) and scope == fixed
    if platform in {"qq", "qqbot"}:
        return scope == fixed if fixed else scope in set(trusted_scopes)
    return True


def _unscoped_legacy(
    profile: str, name: str, primary: str, alternate: str, config: Any,
    connection: Optional[sqlite3.Connection], owner: str,
) -> OwnerDecision:
    """The old anchor of a scope-stating platform whose host stated no scope.

    No binding can be looked up without the scope it is keyed by, and none is invented. The
    historical unscoped anchor still names this person; a confirmed row for another owner under
    any app refuses it. Once the account lives inside a declared app, an app-less turn is that
    account nowhere: it gets neither the anchor nor its own raw key as a private owner.
    """
    fixed_platform, fixed_primary, fixed_alternate, fixed_scope = _configured_identity(config)
    actors = {value for value in (primary, alternate) if value}
    if name != fixed_platform or not actors & ({fixed_primary, fixed_alternate} - {""}):
        return DECLINED
    if name == "slack":
        # A workspace-less Slack ID is not this person's account anywhere: no anchor, no guess.
        return DECLINED
    mapping = read_mapping(config, connection=connection, profile=profile)
    if mapping is None:
        return BLOCKED
    bound = {key for key in set(mapping.confirmed) | mapping.declared
             if key[0] == profile and key[1] == name and key[2] in actors}
    if any(mapping.confirmed.get(key, owner) != owner for key in bound):
        return BLOCKED
    if name in {"qq", "qqbot"} and (bound or str(fixed_scope or "").strip()):
        return BLOCKED
    return OwnerDecision(owner, "legacy", False, True)


def owner_decision(
    profile: Any, platform: Any, actor: Any, actor_alt: Any = "", scope_id: Any = "", *,
    config: Any = None, connection: Optional[sqlite3.Connection] = None, anchor: Any = "",
) -> OwnerDecision:
    """The single ownership decision: person anchor of this account, or nothing.

    Profile, subject IDs and platform scope are normalized once and every branch reads the same
    effective mapping. An existing row for another owner blocks the whole decision *before* the
    legacy rule is considered, so ``account == anchor`` can never route around a confirmed
    conflict, and a declaration in another profile/scope grants nothing here.
    """
    owner = str(anchor or "").strip()
    name = str(platform or "").strip().lower()
    if not owner or not name:
        return DECLINED
    profile = normalize_profile(profile)
    primary = str(actor or "").strip()
    alternate = str(actor_alt or "").strip()
    actors = sorted({value for value in (primary, alternate) if value})
    if not actors:
        return DECLINED
    if name not in SUPPORTED_PLATFORMS:
        # No account namespace exists on this platform, so nothing can be declared or confirmed
        # there and nothing can conflict. The historical unscoped comparison is the whole rule.
        account = _account_key(name, primary, alternate)
        return DECLINED if account != owner else OwnerDecision(owner, "legacy", False, True)
    scope = account_scope(name, scope_id)
    if scope is None:
        # The host states no scope on a platform whose accounts are only unique inside one:
        # nothing can be looked up or declared, so only the old unscoped anchor can apply.
        return _unscoped_legacy(profile, name, primary, alternate, config, connection, owner)
    mapping = read_mapping(config, connection=connection, profile=profile)
    if mapping is None:
        # An unreadable mapping is not an empty one: nothing is granted from it.
        return BLOCKED
    same_platform = lambda key: key[0] == profile and key[1] == name  # noqa: E731
    confirmed = {(key[2], key[3]): value
                 for key, value in mapping.confirmed.items() if same_platform(key)}
    declared = {(key[2], key[3]) for key in mapping.declared if same_platform(key)}
    exact = [(account_id, scope) for account_id in actors]
    # ``actor``/``actor_alt`` name the same account. One confirmed row for either spelling is
    # authoritative, and a row for another owner refuses every spelling.
    if any(confirmed.get(key, owner) != owner for key in exact):
        return BLOCKED
    if any(confirmed.get(key) == owner for key in exact) or any(key in declared for key in exact):
        # A declared/confirmed account that is also the operator's fixed identity keeps its
        # existing host rights; any other new binding adds memory ownership only. The scopes this
        # same person's account is declared/confirmed for are the only ones a legacy (scope-less)
        # fixed ``memory_identity`` may answer for.
        trusted = {key_scope for (account_id, key_scope), stored in confirmed.items()
                   if account_id in actors and stored == owner}
        trusted |= {key_scope for account_id, key_scope in declared if account_id in actors}
        return OwnerDecision(owner, "binding", False,
                             _is_configured_fixed(name, actors, scope, config,
                                                  trusted_scopes=trusted))

    fixed_platform, fixed_primary, fixed_alternate, fixed_scope = _configured_identity(config)
    if name != fixed_platform or not set(actors) & ({fixed_primary, fixed_alternate} - {""}):
        return DECLINED
    if name in {"qq", "qqbot"}:
        fixed = str(fixed_scope or "").strip()
        if fixed:
            # An operator-stated app is the only app this old anchor answers for.
            return OwnerDecision(owner, "legacy", False, True) if scope == fixed else DECLINED
        # The same ID exists in every app: once any app-scoped declaration exists for it, only
        # that exact scope can reach the anchor, and a scope-less turn cannot use a new binding.
        bound_scopes = {key[1] for key in set(confirmed) | declared
                        if key[0] in actors and key[1]}
        return DECLINED if bound_scopes else OwnerDecision(owner, "legacy", False, True)
    if name == "slack":
        fixed = str(fixed_scope or "").strip()
        return (OwnerDecision(owner, "legacy", False, True)
                if fixed and scope == fixed else DECLINED)
    return OwnerDecision(owner, "legacy", False, True)  # Discord snowflakes are global


def read_owners(
    anchor: str, *, config: Any = None, connection: Optional[sqlite3.Connection] = None,
    profile: Optional[str] = None,
) -> Tuple[str, ...]:
    """Every historical owner key of this person's clearly bound accounts.

    Old rows keep their own ``owner_key``: the read side may look at them, but nothing is
    rewritten. A declaration that conflicts with an already-confirmed row is not merged, and an
    unreadable mapping adds nothing beyond the anchor itself.
    """
    owner = str(anchor or "").strip()
    if not owner:
        return ()
    profile = normalize_profile(profile) if profile is not None else None
    keys = {owner}
    mapping = read_mapping(config, connection=connection, profile=profile)
    if mapping is None:
        return (owner,)
    anchor = maintainer_anchor(config) if config is not None else ""
    declared = set(mapping.declared) if anchor and owner == anchor else set()
    for key, stored_owner in mapping.confirmed.items():
        if stored_owner == owner:
            declared.add(key)
    declared = {key for key in declared
                if not mapping.confirmed.get(key) or mapping.confirmed.get(key) == owner}
    from ..infra.textutil import _owner_key
    for _profile, platform, account_id, scope_id in declared:
        legacy = _owner_key(platform, user_id=account_id, scope_id=scope_id)
        if legacy:
            keys.add(legacy)
        if platform in {"qq", "qqbot"} and scope_id and not any(
            key[:3] == (_profile, platform, account_id) and stored_owner != owner
            for key, stored_owner in mapping.confirmed.items()
        ):
            unscoped = _owner_key(platform, user_id=account_id)
            if unscoped:
                keys.add(unscoped)
    return tuple(sorted(keys))


def confirm_bindings(
    connection: sqlite3.Connection, config: Any, *, now: Optional[str] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist the declared bindings in one transaction; a conflicting owner rolls all back.

    Replaying the same declaration is stable (0 new rows). A row that already belongs to another
    owner raises :class:`AccountBindingConflict` *before* anything is committed, so every live
    binding stays intact.
    """
    anchor = maintainer_anchor(config)
    active_profile = normalize_profile(profile) if profile is not None else current_profile()
    entries = configured_bindings(config, profile=active_profile)
    result: Dict[str, Any] = {"anchor": anchor, "confirmed": 0, "existing": 0, "ignored": 0}
    if not anchor or not entries:
        result["ignored"] = len(entries)
        return result
    stamp = str(now or datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    owns_transaction = not connection.in_transaction
    if owns_transaction:
        connection.execute("BEGIN IMMEDIATE")
    try:
        for profile, platform, account_id, scope_id in entries:
            row = connection.execute(
                "SELECT person_owner_key FROM maintainer_account_bindings "
                "WHERE profile=? AND platform=? AND account_id=? AND scope_id=?",
                (profile, platform, account_id, scope_id),
            ).fetchone()
            if row is not None:
                if str(row[0]) != anchor:
                    raise AccountBindingConflict(
                        f"{platform} account is already bound to another owner; "
                        "no binding was changed"
                    )
                result["existing"] += 1
                continue
            connection.execute(
                "INSERT INTO maintainer_account_bindings"
                "(profile, platform, account_id, scope_id, person_owner_key, created_at)"
                " VALUES(?,?,?,?,?,?)",
                (profile, platform, account_id, scope_id, anchor, stamp),
            )
            result["confirmed"] += 1
        if owns_transaction:
            connection.commit()
    except BaseException:
        if owns_transaction and connection.in_transaction:
            connection.rollback()
        raise
    return result


def maintainer_anchor(config: Any) -> str:
    """The existing ``memory_identity`` owner anchor; empty means no private person is configured."""
    from ..infra.textutil import _owner_key
    platform, user_id, alternate, scope_id = _configured_identity(config)
    if not platform or not (user_id or alternate):
        return ""
    scope = scope_id if platform == "slack" else ""
    return _owner_key(platform, user_id=user_id, user_id_alt=alternate, scope_id=scope)
