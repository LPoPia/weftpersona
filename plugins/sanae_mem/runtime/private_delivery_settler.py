from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, Sequence

from ..cognitive.daily_inputs import delivered_user_input
from ..cognitive.turn_user_context import (
    TURN_USER_CONTEXT_TTL_SECONDS as TURN_USER_CONTEXT_TTL_SECONDS,
    CachedTurnUserContext as CachedTurnUserContext,
    TurnUserContextCache as TurnUserContextCache,
    _TURN_USER_CONTEXT_CACHE as _TURN_USER_CONTEXT_CACHE,
    _TurnUserContextKey as _TurnUserContextKey,
    clear_turn_user_context as clear_turn_user_context,
    prune_turn_user_context as prune_turn_user_context,
    read_turn_user_context as read_turn_user_context,
    record_turn_user_context as record_turn_user_context,
)
from ..infra.contracts import freeze_value, thaw_json
from ..infra.host_bridge import is_test_isolated_turn
from ..governance.source_context import in_public_interaction
from ..storage.sqlite_facade import SqliteFacade

if TYPE_CHECKING:
    from ..cognitive.daily_context import DailyContext
    from ..cognitive.inner_wiring import InnerWiring
    from ..governance.resident import ResidentMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifiedPrivateDelivery:
    session_id: str
    turn_id: str
    platform: str
    chat_id: str
    chat_type: str
    sender_id: str
    user_message: str
    assistant_response: str
    delivered_at: float
    sender_id_alt: str = ""
    message_id: str = ""
    incoming_message_id: str = ""
    has_native_image: bool = False
    transcript: tuple[Mapping[str, Any], ...] = ()
    user_message_authored: bool = False
    profile: str = ""
    scope_id: str = ""
    session_key: str = ""
    thread_id: str = ""
    occurred_at: str = ""
    reply_to_message_id: str = ""
    is_bot: Optional[bool] = None
    plugin_user_context: str = ""

    def __post_init__(self) -> None:
        required_text = (
            ("session_id", self.session_id),
            ("turn_id", self.turn_id),
            ("platform", self.platform),
            ("chat_id", self.chat_id),
            ("chat_type", self.chat_type),
            ("sender_id", self.sender_id),
            ("user_message", self.user_message),
            ("assistant_response", self.assistant_response),
        )
        for field_name, value in required_text:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"VerifiedPrivateDelivery.{field_name} must be a non-empty string")
        if isinstance(self.delivered_at, bool) or not isinstance(self.delivered_at, (int, float)):
            raise TypeError("VerifiedPrivateDelivery.delivered_at must be a number")
        if float(self.delivered_at) <= 0.0:
            raise ValueError("VerifiedPrivateDelivery.delivered_at must be positive")
        object.__setattr__(
            self,
            "transcript",
            tuple(freeze_value(item) for item in (self.transcript or ())),
        )


@dataclass(frozen=True)
class PrivateDeliverySettlementResult:
    settled: bool
    owner_key: str = ""
    event_ids: tuple[int, ...] = ()
    daily_recorded: bool = False
    commitments_updated: bool = False
    inner_state_updated: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PrivateDeliveryRoute:
    is_primary: bool
    owner_key: str
    is_configured_private_owner: bool


class PrivateDeliveryProvider(Protocol):
    @property
    def sqlite(self) -> SqliteFacade: ...

    @property
    def inner_wiring(self) -> InnerWiring: ...

    @property
    def daily_context(self) -> DailyContext: ...

    @property
    def resident(self) -> ResidentMemory: ...

    def resolve_private_delivery_route(self) -> PrivateDeliveryRoute: ...
    def record_private_turn_events(
        self, user_text: str, *, current_message: dict[str, object] | None,
        session_id: str, platform: str, author_id: str,
    ) -> tuple[list[int], bool]: ...
    def update_commitments_after_delivery(
        self,
        user_text: str,
        assistant_text: str,
        *,
        owner_key: str,
        turn_id: str,
        tool_results: Sequence[Any] = (),
    ) -> bool: ...


def extract_delivered_tool_results(
    messages: Optional[Sequence[Mapping[str, Any]]],
    *,
    turn_id: str = "",
) -> tuple[dict[str, Any], ...]:
    if not messages:
        return ()
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, Mapping) and message.get("role") == "user":
            start = index + 1
            break
    results: list[dict[str, Any]] = []
    for message in messages[start:]:
        if not isinstance(message, Mapping) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, (list, tuple)):
            content = " ".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, Mapping) and part.get("type") in {"text", "input_text"}
            )
        text = str(content or "")[:8000]
        if not text:
            continue
        result: dict[str, Any] = {
            "name": str(message.get("name") or message.get("tool_name") or ""),
            "tool_call_id": str(message.get("tool_call_id") or ""),
            "content": text,
            "turn_id": str(turn_id or ""),
        }
        for key in ("status", "is_error", "ok", "success", "error"):
            if key in message:
                result[key] = thaw_json(message[key])
        results.append(result)
    return tuple(results)


def _resolve_current_user_message(
    delivery: VerifiedPrivateDelivery,
    user_text: str,
) -> Optional[dict[str, object]]:
    fallback_msg_id = (delivery.incoming_message_id or delivery.message_id).strip()
    for msg in reversed(delivery.transcript):
        if not isinstance(msg, Mapping):
            continue
        if (
            msg.get("role") == "user"
            and not msg.get("display_kind")
            and not msg.get("tool_call_id")
        ):
            resolved = dict(msg)
            resolved["content"] = user_text
            if not str(resolved.get("platform_message_id") or "").strip() and fallback_msg_id:
                resolved["platform_message_id"] = fallback_msg_id
            return resolved
    if not fallback_msg_id:
        return None
    return {
        "role": "user",
        "content": user_text,
        "platform_message_id": fallback_msg_id,
    }


def settle_private_delivery(
    provider: PrivateDeliveryProvider,
    delivery: VerifiedPrivateDelivery,
) -> PrivateDeliverySettlementResult:
    if not provider.sqlite.is_open:
        raise RuntimeError("PrivateDeliveryProvider is not initialized")
    if is_test_isolated_turn(delivery.session_id):
        return PrivateDeliverySettlementResult(settled=False, reason="test_isolated")
    if in_public_interaction():
        return PrivateDeliverySettlementResult(settled=False, reason="public_interaction")
    if delivery.is_bot is True:
        return PrivateDeliverySettlementResult(settled=False, reason="bot_sender")

    route = provider.resolve_private_delivery_route()
    if not route.is_primary:
        return PrivateDeliverySettlementResult(settled=False, reason="non_primary_context")
    owner = route.owner_key
    if not owner:
        return PrivateDeliverySettlementResult(settled=False, reason="no_private_owner")
    scope = provider.inner_wiring.inner_turn_scope(
        session_id=delivery.session_id,
        turn_id=delivery.turn_id,
    )
    if scope is None:
        return PrivateDeliverySettlementResult(settled=False, reason="scope_rejected")

    user_text, _ = delivered_user_input(
        delivery.user_message,
        has_native_image=delivery.has_native_image,
    )
    clean_user_text = user_text.strip()
    is_maintainer = route.is_configured_private_owner

    candidate_ids, touch_resident = provider.record_private_turn_events(
        clean_user_text,
        current_message=_resolve_current_user_message(delivery, clean_user_text),
        session_id=delivery.session_id,
        platform=delivery.platform,
        author_id=delivery.sender_id,
    )

    daily_recorded = bool(
        provider.daily_context.record_completed_turn(
            delivery.session_id,
            delivery.assistant_response,
            list(delivery.transcript),
            sender_id=delivery.sender_id,
            platform=delivery.platform,
            user_message=delivery.user_message,
            turn_id=delivery.turn_id,
            plugin_user_context=delivery.plugin_user_context,
        )
    )

    if touch_resident:
        delivered_dt = datetime.fromtimestamp(float(delivery.delivered_at)).astimezone()
        provider.resident.update_touch(
            clean_user_text,
            delivery.assistant_response,
            session_id=delivery.session_id,
            now=delivered_dt,
        )

    tool_results = extract_delivered_tool_results(delivery.transcript, turn_id=delivery.turn_id)
    commitments_updated = bool(
        provider.update_commitments_after_delivery(
            clean_user_text,
            delivery.assistant_response,
            owner_key=owner,
            turn_id=delivery.turn_id,
            tool_results=tool_results,
        )
    )

    inner_state_updated = bool(
        provider.inner_wiring.apply_verified_delivery(
            session_id=delivery.session_id,
            turn_id=delivery.turn_id,
            user_message=delivery.user_message,
            assistant_response=delivery.assistant_response,
            delivered_at=delivery.delivered_at,
            has_native_image=delivery.has_native_image,
        )
    )

    incomplete_steps: list[str] = []
    if not daily_recorded:
        incomplete_steps.append("daily_skipped" if not is_maintainer else "daily_not_recorded")
    if not commitments_updated:
        incomplete_steps.append("commitments_not_updated")
    if not inner_state_updated:
        incomplete_steps.append("inner_state_not_updated")
    reason = ",".join(incomplete_steps)
    if (is_maintainer and not daily_recorded) or not commitments_updated or not inner_state_updated:
        logger.warning(
            "sanae_mem: private delivery settlement incomplete for session=%s turn=%s (%s)",
            delivery.session_id,
            delivery.turn_id,
            reason,
        )

    return PrivateDeliverySettlementResult(
        settled=True,
        owner_key=owner,
        event_ids=tuple(candidate_ids),
        daily_recorded=daily_recorded,
        commitments_updated=commitments_updated,
        inner_state_updated=inner_state_updated,
        reason=reason,
    )
