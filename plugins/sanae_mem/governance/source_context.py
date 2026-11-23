from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

from ..infra.chat_scope import (
    ChatScope as ChatScope,
    PRIVATE_CHAT_TYPES as PRIVATE_CHAT_TYPES,
    PUBLIC_CHAT_TYPES as PUBLIC_CHAT_TYPES,
    chat_scope_for as chat_scope_for,
    current_chat_scope as current_chat_scope,
    in_public_interaction as in_public_interaction,
    scope_from_source_facts as scope_from_source_facts,
)
from ..infra.textutil import _owner_key

from ..infra.event_fields import (
    PROCESSING_SOURCE_KEYS as PROCESSING_SOURCE_KEYS,
    SELF_SUBJECT as SELF_SUBJECT,
    parse_source as parse_source,
    public_source_view as public_source_view,
    source_identity as source_identity,
    source_json as source_json,
    source_view as source_view,
)

# A multi-person scope keeps its own audience: not "public" (an approved open excerpt) and not
# "private" (one verified owner).
GROUP_SOURCE_VISIBILITY = "group"


class ChatIdentity(NamedTuple):
    """Who this scope shows, and what private memory that person may read or write.

    ``person_key`` is a person label: it may be used in a group, and it is empty when the scope
    states no trustworthy account. ``private_owner_key`` is the storage owner the private chat
    writes to; it is only set for a trusted one-to-one scope. ``private_read_owner_keys`` lists the
    historical owners of the same person's clearly bound accounts, so old rows stay readable
    where they are. ``maintainer`` says the private person is the configured maintainer anchor.
    ``host_operator`` says this scope is the operator's own fixed account: only that account
    (or an already-local/cron turn) carries host profile-file and workspace rights; a new
    memory binding never adds them.
    """

    person_key: str = ""
    private_owner_key: str = ""
    private_read_owner_keys: Tuple[str, ...] = ()
    maintainer: bool = False
    account_key: str = ""
    host_operator: bool = False


def scope_account_key(scope: ChatScope) -> str:
    """The platform account of this scope, keyed inside its own platform scope.

    Slack must state its workspace. A stated QQ app ID is part of the account; an older
    QQ row without one keeps the historical unscoped key only through the guarded legacy
    rule in ``resolve_chat_identity``. Discord's snowflake is global.
    """
    if scope is None or not scope.platform or not (scope.actor or scope.actor_alt):
        return ""
    if scope.platform == "slack" and not scope.scope_id:
        return ""
    if scope.is_bot is True:
        return ""
    scoped = scope.platform in {"slack", "qq", "qqbot"}
    return _owner_key(scope.platform, user_id=scope.actor, user_id_alt=scope.actor_alt,
                      scope_id=scope.scope_id if scoped else "")


def resolve_chat_identity(
    scope: Any, config: Any = None, *, connection: Any = None,
) -> ChatIdentity:
    """Split person, private memory ownership and transport addressing for one trusted scope.

    The same account in a group yields only ``person_key``; private reading/writing needs a
    verified one-to-one scope *and* an account that is either the owner anchor or explicitly
    bound to it. Unknown/restricted scopes, bots and same-shaped IDs from another workspace get
    nothing. A conflict or an unresolvable account scope yields a person label only: no private
    owner and no readable history in any private chat.
    """
    if not isinstance(scope, ChatScope) or scope.visibility not in {"private", "public"}:
        return ChatIdentity()
    account = scope_account_key(scope)
    if not account:
        return ChatIdentity()
    from .account_binding import maintainer_anchor, owner_decision, person_key, read_owners
    anchor = maintainer_anchor(config)
    # One decision for the whole turn: the same profile, actor pair and platform scope, read from
    # one effective mapping. A confirmed row for another owner returns no owner at all, so neither
    # ``account == anchor`` nor the legacy rule can hand this account the anchor back.
    decision = owner_decision(scope.profile, scope.platform, scope.actor, scope.actor_alt,
                              scope.scope_id, config=config, connection=connection, anchor=anchor)
    bound = decision.owner
    if decision.blocked:
        # The mapping names this account for someone else, or an app-scoped account states no app:
        # no branch may fall back to the raw account key, even when that key equals the anchor.
        return ChatIdentity(person_key(account), "", (), False, account)
    if bound:
        person = person_key(bound)
        if scope.visibility != "private":
            # Recognized in a group: the person is known, no private text or write right travels.
            return ChatIdentity(person, "", (), True, account)
        owners = read_owners(bound, config=config, connection=connection, profile=scope.profile)
        return ChatIdentity(person, bound, tuple(sorted({bound, account, *owners})), True,
                            account, bool(decision.fixed))
    person = person_key(account)
    if scope.visibility != "private":
        return ChatIdentity(person, "", (), False, account)
    return ChatIdentity(person, account, (account,), False, account)



def participant_code(owner_key: str) -> str:
    """Stable, non-identifying code for the same trusted owner."""
    value = str(owner_key or "")
    if not value:
        return ""
    digest = hashlib.sha256(("sanae-shared:" + value).encode("utf-8")).hexdigest()
    return "参与者-" + digest[:6].upper()


def is_self_scope(value: Any) -> bool:
    if isinstance(value, dict):
        scope = value
    elif value:
        try:
            scope = json.loads(str(value))
        except (TypeError, ValueError):
            return False
    else:
        return False
    return isinstance(scope, dict) and str(scope.get("kind") or "") == "self"


def self_attribution(owner_key: str = "", *, fact_subject: Optional[str] = None
                     ) -> Tuple[str, str, str, str]:
    """(experience_subject, fact_subject, involved_owner_keys, read_scope).

    A conversation or action is experienced by Sanae; a bound owner is an involved
    person, not the subject. ``fact_subject`` says whose first-person fact the raw
    text states: Sanae for her own experience, or ``owner:<key>`` for an automatic
    user statement. The two are separate so the later read batch never renders a
    user's "I could not sleep" as Sanae's own insomnia.

    The raw scope stays private: self-only when no person is involved, or that
    involved owner's context only when the event carries the person's private
    context. A safe cross-user view is a later-batch read view, never this raw row.
    """
    owner = str(owner_key or "")
    fact = SELF_SUBJECT if fact_subject is None else str(fact_subject or "")
    involved = [owner] if owner else []
    scope = ({"kind": "owner", "owner": owner} if owner
             else {"kind": "self"})
    return (SELF_SUBJECT, fact,
            json.dumps(involved, ensure_ascii=False, sort_keys=True),
            json.dumps(scope, ensure_ascii=False, sort_keys=True))


def user_fact_attribution(owner_key: str) -> Tuple[str, str, str, str]:
    """A user's own grounded fact never becomes Sanae's personal experience."""
    owner = str(owner_key or "")
    if not owner:
        return "", "", "", ""
    return (f"owner:{owner}", f"owner:{owner}",
            json.dumps([owner], ensure_ascii=False, sort_keys=True),
            json.dumps({"kind": "owner", "owner": owner},
                       ensure_ascii=False, sort_keys=True))


def current_source_record(
    *, maintainer: Optional[bool] = None, scope: Optional[ChatScope] = None,
) -> Optional[Dict[str, str]]:
    """The actual account/scope/profile behind this private turn.

    The caller may pass the scope it already resolved from the trusted envelope; otherwise the
    one current host envelope is read. ``author_id`` is the platform actor, never a person label.
    """
    current = current_chat_scope() if scope is None else scope
    if maintainer is None or not isinstance(current, ChatScope) or not current.owner:
        return None
    return source_view({
        "kind": "maintainer" if maintainer else "external_user",
        "author_id": current.actor_alt or current.actor,
        "platform": current.platform,
        "ref": current.message_id,
        "time": current.occurred_at,
        "certainty": "verified",
        "visibility": "private",
        "scope_id": current.scope_id,
        "profile": current.profile or "default",
        "occurred_at": current.occurred_at,
    })


def group_source_record(
    scope: ChatScope, *, paraphrase_ref: str = "",
) -> Optional[Dict[str, str]]:
    """The complete provenance of one utterance in a trusted multi-person scope.

    The scene comes only from the already-resolved ``ChatScope`` (host envelope or admission
    facts); prose, model output or a bare ID prefix never decides it. A one-to-one, unknown or
    isolated scope returns ``None`` so no group provenance is invented. ``ref`` stays the message
    that was said here, and ``paraphrase_ref`` locates the original utterance when this speaker is
    only relaying it — a relay never becomes the quoted person's own statement.
    """
    if not isinstance(scope, ChatScope) or scope.visibility != "public":
        return None
    if not scope.platform or not scope.chat_id:
        return None
    return source_view({
        "kind": "group_member",
        "author_id": scope.actor_alt or scope.actor,
        "platform": scope.platform,
        "ref": scope.message_id,
        "time": scope.occurred_at,
        "certainty": "verified",
        "visibility": GROUP_SOURCE_VISIBILITY,
        "scope_id": scope.scope_id,
        "profile": scope.profile or "default",
        "occurred_at": scope.occurred_at,
        "chat_id": scope.chat_id,
        "thread_id": scope.thread_id,
        "parent_chat_id": scope.parent_chat_id,
        "thread_visibility": scope.thread_visibility,
        "session_key": scope.session_key,
        "paraphrase_ref": str(paraphrase_ref or ""),
        "chat_type": scope.chat_type,
        "reply_to_message_id": scope.reply_to_message_id,
        "synthetic_reply_thread": "true" if scope.synthetic_reply_thread else "false",
    })


def is_maintainer_source(value: Any) -> bool:
    """落库来源是否**明确**为维护者本人；空／unknown 一律 ``False``（不追认）。"""
    record = parse_source(value)
    return bool(record and record["kind"] == "maintainer")


def source_notice(source: Dict[str, Any]) -> str:
    record = source_view(source)
    for key in ("experience_subject", "fact_subject"):
        if source.get(key):
            record[key] = str(source[key])
    if record["kind"] in {"external_user", "video", "unknown"}:
        record["certainty"] = "unverified"
    if record["visibility"] != "public" and in_public_interaction():
        record = public_source_view(record)
    encoded = json.dumps({k: v for k, v in record.items() if v}, ensure_ascii=False, sort_keys=True)
    return "[Source provenance: " + encoded.replace("<", "\\u003c").replace(">", "\\u003e") + "]"


def source_notices(text: str) -> list[str]:
    prefix = "[Source provenance: "
    result, offset = [], 0
    while (start := text.find(prefix, offset)) >= 0:
        offset = start + len(prefix)
        try:
            value, length = json.JSONDecoder().raw_decode(text[offset:])
        except ValueError:
            continue
        end = offset + length
        if isinstance(value, dict) and text[end:end + 1] == "]":
            result.append(text[start:end + 1])
            offset = end + 1
    return result


def daily_profile_key(hermes_home: Any) -> str:
    """已解析数据根（宿主显式 hermes_home）的稳定标签，不新建 profile 系统。"""
    try:
        value = str(Path(hermes_home))
    except Exception:
        value = str(hermes_home or "")
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
