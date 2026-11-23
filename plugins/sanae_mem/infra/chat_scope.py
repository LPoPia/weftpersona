"""Raw conversation scope derived from the trusted host turn or admission facts."""

from __future__ import annotations

from typing import NamedTuple, Optional

from .host_bridge import current_turn_envelope, host_chat_fields
from .textutil import _owner_key

PRIVATE_CHAT_TYPES = frozenset({"dm", "private", "direct"})
# Platform-stated multi-person scenes (Task 01). A group-context scope must be one of these.
PUBLIC_CHAT_TYPES = frozenset({"group", "channel", "thread", "guild", "public"})


class ChatScope(NamedTuple):
    visibility: str
    platform: str = ""
    actor: str = ""
    actor_alt: str = ""
    message_id: str = ""
    chat_type: str = ""
    chat_id: str = ""
    thread_id: str = ""
    session_id: str = ""
    session_key: str = ""
    # Platform facts carried alongside the scene (appended: positional callers keep working).
    scope_id: str = ""
    parent_chat_id: str = ""
    reply_to_message_id: str = ""
    occurred_at: str = ""
    inbound_coverage: str = "unknown"
    is_bot: Optional[bool] = None
    thread_visibility: str = "unknown"
    profile: str = ""
    # The thread id is where the host's REPLY lands, not an inbound thread (no thread scope).
    synthetic_reply_thread: bool = False

    @property
    def owner(self) -> str:
        if self.visibility != "private":
            return ""
        # Keep the historical QQ owner namespace (Task 03's added facts contract). The account
        # resolver applies the app scope when granting a configured/bound anchor.
        scope = self.scope_id if self.platform == "slack" else ""
        return _owner_key(self.platform, user_id=self.actor, user_id_alt=self.actor_alt,
                          scope_id=scope)


def _visibility(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"narrower", "inherited"} else "unknown"


def chat_scope_for(envelope: object | None) -> ChatScope:
    """Resolve a ChatScope from one trusted turn envelope.

    The scene comes from the envelope's platform-stated ``chat_type``; the id prefix is never a
    substitute. Facts travel on every non-isolated scope (speaker, address, time, thread visibility)
    while ``owner`` stays limited to a private scene.
    """
    if envelope is None:
        return ChatScope("unknown")
    fields = host_chat_fields(envelope)
    platform, chat_type, chat_id = fields.platform, fields.chat_type, fields.chat_id
    sender_id, sender_alt = fields.sender_id, fields.sender_id_alt
    scope = ChatScope(
        "unknown", platform, sender_id, sender_alt, fields.message_id, chat_type, chat_id,
        session_key=fields.session_key,
        scope_id=fields.scope_id,
        parent_chat_id=fields.parent_chat_id,
        reply_to_message_id=fields.reply_to_message_id,
        occurred_at=fields.occurred_at,
        inbound_coverage=fields.inbound_coverage,
        is_bot=fields.is_bot,
        thread_visibility=_visibility(fields.thread_visibility),
        profile=fields.profile,
        thread_id=fields.thread_id,
        session_id=fields.session_id,
        synthetic_reply_thread=fields.synthetic_reply_thread,
    )
    if fields.restricted:
        # Isolation stays absolute: no speaker, address or time crosses this boundary.
        return ChatScope("unknown", platform)
    if chat_type in PUBLIC_CHAT_TYPES:
        # The speaker travels with a multi-person scope (later group tasks need "who"), but the
        # scope holds no private owner and grants none.
        return scope._replace(visibility="public")
    if chat_type in {*PRIVATE_CHAT_TYPES, "c2c"}:
        actor = sender_id or sender_alt
        alternate = sender_alt if sender_id and actor == sender_id else ""
        if platform and chat_id and actor and fields.session_key:
            # Slack states the private scope as a workspace; without it the private owner would key
            # on an ID that is only unique inside a workspace, so the scope stays unknown.
            visibility = "unknown" if platform == "slack" and not scope.scope_id else "private"
            return scope._replace(visibility=visibility, actor=actor, actor_alt=alternate)
    if not chat_type and not chat_id and (platform or fields.source) in {"cli", "tui", "desktop", "cron"}:
        return ChatScope("local", platform)
    # Unknown scene: no private owner, but the trusted routing facts still describe the event.
    return scope


def current_chat_scope() -> ChatScope:
    """Resolve identity from the trusted host envelope, never ambient environment."""
    return chat_scope_for(current_turn_envelope())


def scope_from_source_facts(source: object) -> ChatScope:
    """The same scene reading for a trusted admission fact dict (no turn envelope at hand).

    Only a platform-stated one-to-one scene with an address and an actor becomes private; any
    other shape keeps the facts but grants no owner.
    """
    if not isinstance(source, dict):
        return ChatScope("unknown")
    platform = str(source.get("platform") or "").strip().lower()
    chat_type = str(source.get("chat_type") or "").strip().lower()
    chat_id = str(source.get("chat_id") or "").strip()
    actor = str(source.get("user_id") or "").strip()
    actor_alt = str(source.get("user_id_alt") or "").strip()
    scope_id = str(source.get("scope_id") or "").strip()
    is_bot = source.get("is_bot")
    profile = str(source.get("profile") or "").strip()
    if source.get("restricted_audience"):
        return ChatScope("unknown", platform, scope_id=scope_id, is_bot=is_bot, profile=profile)
    visibility = "unknown"
    if chat_type in PUBLIC_CHAT_TYPES:
        visibility = "public"
    elif (chat_type in {*PRIVATE_CHAT_TYPES, "c2c"} and platform and chat_id and (actor or actor_alt)
          and (platform != "slack" or scope_id)):
        visibility = "private"
    return ChatScope(
        visibility, platform, actor, actor_alt, chat_type=chat_type, chat_id=chat_id,
        scope_id=scope_id, is_bot=is_bot, profile=profile,
    )


def in_public_interaction() -> bool:
    """Unknown messaging contexts have the same restrictions as public chats."""
    return current_chat_scope().visibility in {"public", "unknown"}


def is_system_turn(session_id: str) -> bool:
    return str(session_id or "").startswith("cron")
