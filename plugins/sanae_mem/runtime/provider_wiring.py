from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from sanae_mem.governance.runtime_registry import (
    reconcile_registered_schedule,
    register_active_provider,
    resolve_active_provider,
)

from ..cognitive.association import Association, AssociationDependencies, AssociationTurn
from ..cognitive.daily_context import DailyBinding, DailyContext
from ..cognitive.daily_inputs import _without_known_meme_notes
from ..cognitive.daily_state import DailyState
from ..cognitive.inner_state import (
    InnerStateDependencies,
    InnerStatePersona,
    InnerStateService,
)
from ..cognitive.inner_wiring import InnerWiring
from ..cognitive.nightly_journal import NightlyJournal, NightlyJournalTarget
from ..domains.commitments import (
    COMMITMENT_STATUS_COMPONENT,
    CommitmentContext,
    CommitmentGroup,
    CommitmentPrompt,
    Commitments,
)
from ..domains.memes import MemeRuntime, MemeStore
from ..domains.video_memory import VideoMemory, VideoMemoryDependencies
from ..governance.account_binding import owner_decision
from ..governance.association_state import AssociationHookRequest, AssociationState
from ..governance.grounded_review import GroundedReview, GroundedReviewContext
from ..governance.group_scope import GroupGrant
from ..governance.llm_guard import LlmGuard
from ..governance.memory_axis import MemoryAxis
from ..governance.promotion_review import PromotionReview, PromotionReviewDependencies
from ..governance.read_gate import (
    ReadGate,
    ReadGateDependencies,
    ReadGateOptions,
    RecallPurpose,
    RecallResult,
)
from ..governance.resident import ResidentDependencies, ResidentMemory
from ..governance.shared_memory import SHARED_TOPIC_LABEL
from ..governance.source_context import (
    current_chat_scope,
    in_public_interaction,
    resolve_chat_identity,
)
from ..governance.source_intake import SourceIntakeContext
from ..infra import auxiliary
from ..infra.chat_scope import is_system_turn
from ..infra.contracts import DEFAULT_CAS_MAX_RETRIES
from ..infra.defaults import DEFAULT_PARTNER_ROLE, DEFAULT_PERSONA_NAME
from ..infra.host_bridge import (
    current_turn_is_messaging_surface,
    get_hermes_home,
    get_host_runtime_home,
    hermes_home_scope,
    is_delegated_child_turn,
    is_test_isolated_turn,
)
from ..infra.status_store import StatusStore
from ..infra.textutil import _bool, _candidate_text, event_keywords
from ..infra.words import _GAP_INJECT_MAX_CHARS
from ..storage.event_maintenance import EventMaintenance
from ..storage.event_review_store import EventReviewStore
from ..storage.event_store import EventStore
from ..storage.retrieval import Retrieval, RetrievalDependencies
from ..storage.storage import Storage
from .group_ingest import GroupDeliveryContext
from .group_memory import (
    current_group_grant,
    enabled_group_owners,
    group_write_owner,
    is_managed_group_scene,
)
from .ingest import Ingest, IngestDependencies, event_topic
from .lifecycle import Lifecycle, LifecycleDependencies
from .private_delivery_settler import PrivateDeliveryRoute, record_turn_user_context
from .scheduled import current_free_think
from .tool_router import ToolRouter

if TYPE_CHECKING:
    from .provider import SanaeMemProvider

logger = logging.getLogger(__name__)


class _AuxiliaryReply(Protocol):
    @property
    def text(self) -> str: ...


def aux_reply_text(raw: _AuxiliaryReply | str) -> str:
    return raw if isinstance(raw, str) else raw.text


@dataclass(frozen=True)
class _ProviderSettings:
    read_gate_options: ReadGateOptions
    inject_max_chars: int
    distill_threshold: float
    distill_daily_cap: int
    resident_review_daily_cap: int
    emotion_ttl_days: int
    expiry_review_days: int
    expiring_importance_decay: float
    expiring_importance_floor: float
    chroma_enabled: bool
    chroma_site_packages: str
    persona_name: str
    partner_name: str
    partner_role: str
    partner_pronoun: str


def _parse_provider_settings(config: Mapping[str, Any]) -> _ProviderSettings:
    gate = str(config.get("read_gate", "explicit")).lower()
    inject_max_chars = max(1, min(400, int(config.get("inject_max_chars", 400))))
    partner_name = str(config.get("partner_name") or "").strip()
    pronoun = str(config.get("partner_pronoun") or "他").strip()
    requested = str(config.get("semantic_backend", "chroma")).lower()
    return _ProviderSettings(
        read_gate_options=ReadGateOptions(
            scheduled=gate == "scheduled",
            inject_max_chars=inject_max_chars,
            daily_scheduled_asks=max(0, int(config.get("daily_scheduled_asks", 2))),
            gap_continue=_bool(config.get("gap_continue"), True),
            gap_threshold_hours=max(0.0, float(config.get("gap_threshold_hours", 2))),
            partner_name=partner_name,
        ),
        inject_max_chars=inject_max_chars,
        distill_threshold=max(0.0, min(1.0, float(config.get("distill_threshold", 0.6)))),
        distill_daily_cap=max(0, int(config.get("distill_daily_cap", 2))),
        resident_review_daily_cap=max(0, int(config.get("resident_review_daily_cap", 60))),
        emotion_ttl_days=max(1, int(config.get("emotion_ttl_days", 90))),
        expiry_review_days=max(1, int(config.get("expiry_review_days", 30))),
        expiring_importance_decay=max(
            0.0, min(1.0, float(config.get("expiring_importance_decay", 0.9)))
        ),
        expiring_importance_floor=max(
            0.0, min(1.0, float(config.get("expiring_importance_floor", 0.4)))
        ),
        chroma_enabled=requested != "sqlite",
        chroma_site_packages=str(config.get("chroma_site_packages") or ""),
        persona_name=(
            str(config.get("persona_name") or DEFAULT_PERSONA_NAME).strip()
            or DEFAULT_PERSONA_NAME
        ),
        partner_name=partner_name,
        partner_role=str(config.get("partner_role") or "").strip() or DEFAULT_PARTNER_ROLE,
        partner_pronoun=pronoun if pronoun in {"他", "她"} else "他",
    )


def assign_provider_settings(provider: SanaeMemProvider, settings: _ProviderSettings) -> None:
    provider._inject_max_chars = settings.inject_max_chars
    provider._distill_daily_cap = settings.distill_daily_cap
    provider._resident_review_daily_cap = settings.resident_review_daily_cap
    provider._emotion_ttl_days = settings.emotion_ttl_days
    provider._persona_name = settings.persona_name
    provider._partner_name = settings.partner_name
    provider._partner_role = settings.partner_role
    provider._partner_pronoun = settings.partner_pronoun


def refresh_provider_config(provider: SanaeMemProvider, new_config: Mapping[str, Any]) -> None:
    provider._config.clear()
    provider._config.update(new_config)
    settings = _parse_provider_settings(provider._config)
    assign_provider_settings(provider, settings)
    provider._storage.reconfigure(
        chroma_enabled=settings.chroma_enabled,
        chroma_site_packages=settings.chroma_site_packages,
    )
    provider.ingest.reconfigure(emotion_ttl_days=settings.emotion_ttl_days)
    provider._llm_guard = replace(
        provider._llm_guard,
        persona_name=settings.persona_name,
        partner_name=settings.partner_name,
        partner_role=settings.partner_role,
        partner_pronoun=settings.partner_pronoun,
    )
    provider._daily_context.reconfigure(
        config=provider._config,
        persona_name=settings.persona_name,
    )
    provider._read_gate.reconfigure(settings.read_gate_options)
    provider._inner_wiring.reconfigure(
        partner_name=settings.partner_name,
        partner_role=settings.partner_role,
    )
    provider._lifecycle.reconfigure(
        distill_threshold=settings.distill_threshold,
        expiry_review_days=settings.expiry_review_days,
        expiring_importance_decay=settings.expiring_importance_decay,
        expiring_importance_floor=settings.expiring_importance_floor,
    )


def _grounded_review_context(provider: SanaeMemProvider) -> GroundedReviewContext:
    return GroundedReviewContext(
        home=Path(provider._hermes_home),
        session_id=provider._session_id,
        owner=provider._owner_key,
        maintainer_owner=provider._daily_owner_key,
        maintainer_platform=provider._daily_owner_platform,
        maintainer_ids=provider._daily_identity,
    )


def _is_grounded_evidence_owner(
    provider: SanaeMemProvider,
    profile: str,
    platform: str,
    actor: str,
    scope_id: str,
    anchor: str,
) -> bool:
    decision = provider._sqlite.read_optional(
        lambda conn: owner_decision(
            profile, platform, actor, "", scope_id,
            config=provider._config, connection=conn, anchor=anchor,
        )
    )
    return decision.owner == anchor and not decision.blocked


def _review_grounded_candidate(
    provider: SanaeMemProvider, now: datetime, context: str,
) -> Mapping[str, object] | None:
    candidate = None
    if not provider._promotion_review.can_review_resident(now):
        error = "quota_exhausted"
    elif provider._llm is None:
        error = "llm_unavailable"
    else:
        error = provider._promotion_review.reserve_resident_call(now)
        if not error:
            try:
                candidate = _review_grounded_llm(provider, context)
            except OSError as exc:
                logger.warning("sanae_mem: grounded user review failed (%s)", type(exc).__name__)
                error = "review_error"
    if candidate is None:
        write_maintenance_status(
            provider, "resident_review", ok=False, stage="grounded_user",
            error=error or "grounded_review_unavailable", now=now,
        )
    elif provider._status_store.read().get("components", {}).get("resident_review", {}).get("ok") is False:
        write_maintenance_status(
            provider, "resident_review", ok=True, stage="grounded_user", now=now,
        )
    return candidate


def _review_grounded_llm(provider: SanaeMemProvider, context: str) -> Mapping[str, object] | None:
    return provider._llm_guard.review_grounded_user(context)


def write_maintenance_status(
    provider: SanaeMemProvider,
    component: str,
    *,
    ok: bool,
    stage: str,
    error: str = "",
    now: datetime | float | None = None,
    data_dir: Path | None = None,
    **extra: object,
) -> None:
    provider._status_store.write_maintenance(
        component, ok=ok, stage=stage, error=error, now=now, data_dir=data_dir, **extra,
    )


def write_owner(provider: SanaeMemProvider) -> str:
    return provider._association_state.write_owner(provider._owner_key)


def association_read_owner(provider: SanaeMemProvider) -> str:
    return provider._association_state.read_owner(provider._owner_key)


def association_read_valid(provider: SanaeMemProvider) -> bool:
    return provider._association_state.validate_read(provider._owner_key)


def apply_private_identity(provider: SanaeMemProvider, scope: Any = None) -> None:
    current = current_chat_scope() if scope is None else scope
    if provider._agent_context != "primary":
        return
    provider._read_owner_keys = ()
    if current.platform and provider._platform and current.platform != provider._platform:
        provider._owner_key = ""
        return
    identity = provider._sqlite.read_optional(
        lambda conn: resolve_chat_identity(current, provider._config, connection=conn)
    )
    provider._owner_key = identity.private_owner_key
    provider._read_owner_keys = identity.private_read_owner_keys


def private_chat_owner(provider: SanaeMemProvider) -> str:
    scope = current_chat_scope()
    if (
        scope.visibility != "private"
        or not provider._platform
        or scope.platform != provider._platform
    ):
        return ""
    identity = provider._sqlite.read_optional(
        lambda conn: resolve_chat_identity(scope, provider._config, connection=conn)
    )
    owner = identity.private_owner_key
    return owner if owner and owner == provider._owner_key else ""


def is_configured_private_owner(provider: SanaeMemProvider) -> bool:
    scope = current_chat_scope()
    identity = provider._sqlite.read_optional(
        lambda conn: resolve_chat_identity(scope, provider._config, connection=conn)
    )
    return bool(identity.maintainer and private_chat_owner(provider))


def live_group_config(provider: SanaeMemProvider) -> dict[str, Any] | None:
    if get_hermes_home().resolve() != Path(provider._hermes_home).resolve():
        return None
    return provider.runtime_config()


def group_scope_grant(provider: SanaeMemProvider) -> GroupGrant | None:
    if provider._agent_context != "primary" or provider.shadow_enabled():
        return None
    config = live_group_config(provider)
    if config is None:
        return None
    return provider._sqlite.read_optional(
        lambda conn: current_group_grant(config, connection=conn)
    )


def group_resolution_owners(provider: SanaeMemProvider) -> tuple[str, ...]:
    grant = group_scope_grant(provider)
    return grant.resolution_owners if grant is not None else ()


def authorized_free_think_owner(provider: SanaeMemProvider, session_id: str = "") -> str:
    if provider._agent_context != "cron" or not provider._daily_owner_key:
        return ""
    session_id = session_id or provider._session_id
    if session_id != provider._session_id:
        return ""
    if get_host_runtime_home().resolve() != Path(provider._hermes_home).resolve():
        return ""
    active = current_free_think(provider.runtime_config(), session_id)
    return active[0] if active and active[0] == provider._daily_owner_key else ""


def bind_active_provider(provider: SanaeMemProvider) -> None:
    if provider._session_id and provider._sqlite.is_open:
        provider._release_active_provider = register_active_provider(
            Path(provider._hermes_home), provider._session_id, provider,
        )


def unbind_active_provider(provider: SanaeMemProvider) -> None:
    release = provider._release_active_provider
    provider._release_active_provider = None
    if release is not None:
        release()


def current_recall_event_ids(provider: SanaeMemProvider) -> tuple[int, ...]:
    state = provider._association_state.current_recall()
    if state is None or state.get("status") != "ok":
        return ()
    ids: list[int] = []
    for item in state.get("items") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_table") or "event") != "event":
            continue
        value = item.get("id")
        if isinstance(value, int) and not isinstance(value, bool):
            ids.append(int(value))
    return tuple(dict.fromkeys(ids))


def association_query_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(
            str(part.get("text") or "") for part in value
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
        ).strip()
    return ""


def build_source_intake_context(provider: SanaeMemProvider) -> SourceIntakeContext:
    return SourceIntakeContext(
        session_id=provider._session_id,
        agent_context=provider._agent_context,
        shadow=provider.shadow_enabled(),
        daily_owner_key=provider._daily_owner_key,
        daily_owner_platform=provider._daily_owner_platform,
        daily_profile_key=provider._daily_profile_key,
        private_chat_owner=private_chat_owner(provider),
        is_configured_private_owner=is_configured_private_owner(provider),
        open_daily_entry=provider.open_daily_entry,
    )


def resolve_integration_owner(
    provider: SanaeMemProvider,
    *,
    home: Path,
    allow_cron: bool,
    write: bool,
    cron_active: bool,
) -> str | None:
    if not provider._session_id or Path(provider._hermes_home).resolve() != home:
        return None
    scope = current_chat_scope()
    priv = private_chat_owner(provider) if provider._agent_context == "primary" else ""
    if (
        provider._agent_context == "primary"
        and scope.visibility == "private"
        and scope.session_id == provider._session_id
        and priv
    ):
        expected: str | None = priv
    elif allow_cron and provider._agent_context == "cron" and cron_active:
        expected = None
    else:
        return None
    try:
        owner = write_owner(provider) if write else association_read_owner(provider)
    except PermissionError:
        return None
    return owner if expected is None or owner == expected else None


def resolve_private_delivery_route(provider: SanaeMemProvider) -> PrivateDeliveryRoute:
    if provider._agent_context != "primary":
        return PrivateDeliveryRoute(
            is_primary=False,
            owner_key="",
            is_configured_private_owner=False,
        )
    apply_private_identity(provider)
    owner = private_chat_owner(provider)
    if not owner:
        return PrivateDeliveryRoute(
            is_primary=True,
            owner_key="",
            is_configured_private_owner=False,
        )
    return PrivateDeliveryRoute(
        is_primary=True,
        owner_key=owner,
        is_configured_private_owner=is_configured_private_owner(provider),
    )


def build_group_delivery_context(provider: SanaeMemProvider) -> GroupDeliveryContext:
    return GroupDeliveryContext(
        grant=group_scope_grant(provider),
        session_id=provider._session_id,
        persona_name=provider._persona_name,
    )


def run_pre_llm_call(provider: SanaeMemProvider, *args: Any, **kwargs: Any) -> dict:
    home = get_hermes_home().resolve()
    if home != Path(provider._hermes_home).resolve():
        return {}
    session_id = str(kwargs.get("session_id") or provider._session_id)
    scope = current_chat_scope()
    if session_id and scope.session_id == session_id and not is_delegated_child_turn():
        reconcile_registered_schedule(home)
    if not provider.shadow_enabled() and (
        not provider._sqlite.is_open or (session_id and provider._session_id != session_id)
    ):
        active = resolve_active_provider(home, session_id) if session_id else None
        if active is not None and active is not provider:
            return active.handle_pre_llm_call(*args, **kwargs)
    apply_private_identity(provider, scope)
    if (
        provider._agent_context == "primary"
        and not provider.shadow_enabled()
        and scope.visibility == "public"
    ):
        if scope.session_id and scope.session_id == session_id == provider._session_id:
            authored = _without_known_meme_notes(kwargs.get("user_message"))
            turn_id = str(kwargs.get("turn_id") or "")
            parts = []
            group_context = provider.group_turn_context(
                association_query_text(authored),
                session_id=session_id,
                turn_id=turn_id,
            )
            if group_context:
                parts.append(group_context)
            meme_context = provider.meme_context_for_turn(authored)
            if meme_context:
                parts.append(meme_context)
            context = "\n\n".join(parts)
            if not context:
                return {}
            if group_context:
                return {"context": context, "ephemeral": True}
            return {"context": context}
        return {}
    if (
        not private_chat_owner(provider)
        or provider._agent_context != "primary"
        or provider.shadow_enabled()
    ):
        return {}
    authored_message = _without_known_meme_notes(kwargs.get("user_message"))
    query = association_query_text(authored_message)
    state = provider._association_state
    request = state.begin_hook_recall(session_id, provider._session_id, query)
    provider._daily_context.check_pre_llm_identity(
        session_id=str(kwargs.get("session_id") or ""),
        platform=str(kwargs.get("platform") or ""),
        sender_id=str(kwargs.get("sender_id") or ""),
    )
    base: dict[str, str] = {}
    if not state.hook_active(request, binding_session_id=provider._session_id):
        return base
    if query:
        session_id = str(kwargs.get("session_id") or provider._session_id)
        gap = provider._read_gate.begin_hook_recall(query, session_id)
        explicit = ReadGate.triggers_explicit_past(query)
        result = associate_context(
            provider,
            query,
            conversation_history=kwargs.get("conversation_history"),
            purpose="chat",
            session_id=session_id,
            max_chars=_GAP_INJECT_MAX_CHARS if gap else provider._inject_max_chars,
            gap_continuation=gap,
            hot_path=True,
            _hook_request=request,
        )

        def _apply_hook_delivery() -> None:
            if result.get("status") == "ok" and result.get("context"):
                via = "gap" if gap else "explicit_past" if explicit else "chat"
                provider._read_gate.mark_asked(result["items"], via, session_id)
                if gap:
                    provider._read_gate.record_gap_delivery(session_id)
                prior = str(base.get("context") or "")
                base["context"] = (prior + "\n\n" + result["context"]).strip()

        if not state.mark_hook_delivered(
            request,
            result,
            binding_session_id=provider._session_id,
            on_deliver=_apply_hook_delivery,
        ):
            return base
    meme_context = ""
    try:
        meme_context = provider.meme_context_for_turn(
            authored_message, known_context=kwargs.get("user_message"),
        )
        if meme_context:
            prior = str(base.get("context") or "")
            base["context"] = (prior + "\n\n" + meme_context).strip()
    except Exception:
        logger.debug("sanae_mem: meme context lookup skipped", exc_info=True)
    if meme_context:
        record_turn_user_context(
            provider._hermes_home,
            session_id,
            str(kwargs.get("turn_id") or ""),
            meme_context=meme_context,
        )
    return base


def sync_provider_turn(
    provider: SanaeMemProvider,
    user_content: str,
    assistant_content: str,
    *,
    session_id: str = "",
    messages: list[dict[str, Any]] | None = None,
    turn_author: dict[str, Any] | None = None,
) -> None:
    if is_test_isolated_turn(session_id or provider._session_id):
        return
    active_session = session_id or provider._session_id
    if (
        provider._sqlite.is_open
        and is_system_turn(active_session)
        and str(assistant_content or "").strip()
        and str(assistant_content or "").strip() not in {"HEARTBEAT_OK", "[SILENT]"}
    ):
        provider._daily_context.resolve_explored_curiosity_threads(
            str(assistant_content or ""), propagate_read_error=True,
        )
    if current_turn_is_messaging_surface():
        return
    if in_public_interaction() or (
        provider._agent_context == "primary" and not private_chat_owner(provider)
    ):
        return
    if provider._agent_context != "primary" or not provider._sqlite.is_open:
        return
    apply_private_identity(provider)
    if is_system_turn(active_session):
        return
    author = turn_author if isinstance(turn_author, dict) else {}
    if author.get("is_bot"):
        return
    if not str(assistant_content or "").strip():
        return
    current_message = next(
        (msg for msg in reversed(messages or []) if isinstance(msg, dict) and msg.get("role") == "user"),
        None,
    )
    _, touch_resident = provider.record_private_turn_events(
        user_content,
        current_message=current_message,
        session_id=active_session,
        platform=provider._platform,
        author_id=str(author.get("id") or ""),
    )
    if touch_resident:
        provider._resident.update_touch(
            str(user_content or "").strip(),
            assistant_content,
            session_id=active_session,
        )


def run_provider_nightly_journal(
    provider: SanaeMemProvider,
    *,
    now: datetime | None = None,
    owner_key: str | None = None,
    force: bool = False,
) -> dict:
    with provider._background_tasks.track() as active:
        if not active:
            return {"ok": False, "reason": "shutdown"}
        target = NightlyJournalTarget(
            home=Path(provider._hermes_home),
            data_dir=provider._data_dir,
            persona_name=provider._persona_name,
            agent_context=provider._agent_context,
        )
        journal = NightlyJournal(
            provider._sqlite,
            target,
            model=provider._llm,
            current_session_id=lambda: provider._session_id,
            shadow_enabled=provider.shadow_enabled,
            trusted_owner=lambda requested: provider._memory_axis.trusted_owner(
                requested or provider._daily_owner_key or provider._owner_key
            ),
            read_shared_inner_state=provider._inner_state.ensure_shared_inner_state,
            write_status=lambda *, ok, stage, day, error, now, data_dir: write_maintenance_status(
                provider, "nightly_journal", ok=ok, stage=stage, day=day,
                error=error, now=now, data_dir=data_dir,
            ),
            curve_base_days=provider.ingest.curve_base_days,
            keywords=event_keywords,
            upsert_vector=provider._storage.upsert_vector,
        )
        with hermes_home_scope(target.home):
            return journal.run(now=now, owner_key=owner_key, force=force)


def group_scene_active(provider: SanaeMemProvider) -> bool:
    if provider._agent_context != "primary" or provider.shadow_enabled():
        return False
    return is_managed_group_scene(current_chat_scope())


def group_read_owners(provider: SanaeMemProvider) -> list[str]:
    grant = group_scope_grant(provider)
    return list(grant.read_owners) if grant else []


def group_write_owner_key(provider: SanaeMemProvider) -> str:
    return group_write_owner(group_scope_grant(provider))


def group_maintenance_owners(provider: SanaeMemProvider) -> list[str]:
    if provider._agent_context != "cron":
        return []
    config = live_group_config(provider)
    if config is None:
        return []
    return provider._sqlite.read_optional(
        lambda conn: list(enabled_group_owners(config, connection=conn))
    )


def associate_context(
    provider: SanaeMemProvider,
    query: str,
    *,
    conversation_history: list[Any] | None = None,
    purpose: str = "chat",
    session_id: str = "",
    max_chars: int | None = None,
    gap_continuation: bool = False,
    hot_path: bool = False,
    _hook_request: AssociationHookRequest | None = None,
) -> dict:
    state = provider._association_state
    with state.lock:
        revision = state.revision
        turn = AssociationTurn(
            owner_key=provider._owner_key,
            daily_owner_key=provider._daily_owner_key,
            daily_profile_key=provider._daily_profile_key,
            daily_owner_platform=provider._daily_owner_platform,
            agent_context=provider._agent_context,
            inject_max_chars=provider._inject_max_chars,
            llm=provider._llm,
        )
    return provider._association.associate_context(
        query, conversation_history=conversation_history, purpose=purpose,
        session_id=session_id, max_chars=max_chars, gap_continuation=gap_continuation,
        hot_path=hot_path, turn=turn, revision=revision, hook_request=_hook_request,
    )


def _daily_binding(provider: SanaeMemProvider) -> DailyBinding:
    return DailyBinding(
        session_id=provider._session_id,
        agent_context=provider._agent_context,
        platform=provider._platform,
        owner_key=provider._daily_owner_key,
        profile_key=provider._daily_profile_key,
        owner_platform=provider._daily_owner_platform,
        owner_accounts=provider._daily_identity,
    )


def _private_inner_turn_owner(provider: SanaeMemProvider, session_id: str) -> str:
    scope = current_chat_scope()
    if (
        provider.shadow_enabled()
        or provider._agent_context != "primary"
        or session_id != scope.session_id
        or session_id != provider._session_id
    ):
        return ""
    return private_chat_owner(provider)


def _commitment_context(provider: SanaeMemProvider) -> CommitmentContext:
    raw_config = provider._config.get("memory_identity")
    configured_ids = (
        tuple(str(raw_config.get(key) or "").strip() for key in ("user_id", "user_id_alt"))
        if isinstance(raw_config, dict) else ()
    )
    return CommitmentContext(
        home=Path(provider._hermes_home),
        session_id=provider._session_id,
        platform=provider._platform,
        partner_name=provider._partner_name,
        persona_name=provider._persona_name,
        identity_tokens=(
            provider._owner_key,
            provider._daily_owner_key,
            provider._maintainer_id,
            *provider._daily_identity,
            *configured_ids,
        ),
    )


def _commitment_group(provider: SanaeMemProvider) -> CommitmentGroup:
    active = group_scene_active(provider)
    return CommitmentGroup(active=active, grant=group_scope_grant(provider) if active else None)


def _has_commitment_failure(provider: SanaeMemProvider) -> bool:
    components = provider._status_store.read().get("components")
    status = components.get(COMMITMENT_STATUS_COMPONENT) if isinstance(components, dict) else None
    return isinstance(status, dict) and status.get("ok") is False


def _complete_commitment_prompt(provider: SanaeMemProvider, prompt: CommitmentPrompt) -> str | None:
    if provider._llm is None:
        return None
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    raw = auxiliary.complete_simple(
        provider._llm, messages, temperature=0.0, max_tokens=600, timeout=25.0,
        failure_message="commitments auxiliary update failed",
    )
    return aux_reply_text(raw) if raw else None


def _read_gate_provenance_owner(provider: SanaeMemProvider) -> str:
    try:
        return association_read_owner(provider)
    except PermissionError:
        return provider._owner_key


def _associate_for_prefetch(
    provider: SanaeMemProvider,
    query: str,
    purpose: RecallPurpose,
    session_id: str,
    max_chars: int,
    gap_continuation: bool,
) -> RecallResult:
    result = associate_context(
        provider,
        query,
        purpose=purpose,
        session_id=session_id,
        max_chars=max_chars,
        gap_continuation=gap_continuation,
        hot_path=True,
    )
    return RecallResult(result["status"], result["context"], result["items"])


def wire_provider_components(provider: SanaeMemProvider, settings: _ProviderSettings) -> None:
    sqlite = provider._sqlite
    lock = provider._lock

    event_review_store = EventReviewStore(sqlite)
    provider._event_review_store = event_review_store
    provider._session_id = ""
    provider._agent_context = "primary"
    provider._owner_key = ""
    provider._read_owner_keys = ()

    event_store = EventStore(
        sqlite,
        current_session_id=lambda: provider._session_id,
        current_agent_context=lambda: provider._agent_context,
        current_home=lambda: provider._hermes_home,
        current_owner=lambda: provider._owner_key,
        current_read_owners=lambda: provider._read_owner_keys,
        private_chat_owner=lambda: private_chat_owner(provider),
        is_system_turn=is_system_turn,
        group_scene_active=lambda: group_scene_active(provider),
        group_read_owners=lambda: group_read_owners(provider),
        group_write_owner_key=lambda: group_write_owner_key(provider),
        group_maintenance_owners=lambda: group_maintenance_owners(provider),
        group_resolution_owners=lambda: group_resolution_owners(provider),
    )
    provider._event_store = event_store

    video_memory = VideoMemory(VideoMemoryDependencies(
        sqlite=sqlite,
        current_session_id=lambda: provider._session_id,
        current_persona_name=lambda: provider._persona_name,
        read_owner_keys=event_store.read_owner_keys_for,
        mutate_status=lambda change: provider._status_store.mutate(change),
        resolve_explored_curiosity=lambda summary: provider._daily_context.resolve_explored_curiosity_threads(summary),
        record_curiosity=lambda title, content, session_id: provider._daily_context.record_video_curiosity(
            title, content, session_id=session_id,
        ),
    ))
    provider._video_memory = video_memory

    provider._release_active_provider = None
    provider._active_provider_desired = False
    provider._release_source_sink = None
    provider._agent_identity = ""
    provider._platform = ""
    provider._llm = None

    configured_dir = provider._config.get("data_dir")
    provider._data_dir = (
        Path(configured_dir).expanduser()
        if configured_dir
        else Path(provider._hermes_home) / "sanae-mem"
    )

    storage = Storage(
        sqlite,
        lock,
        current_data_dir=lambda: provider._data_dir,
        chroma_enabled=settings.chroma_enabled,
        chroma_site_packages=settings.chroma_site_packages,
    )
    provider._storage = storage

    event_maintenance = EventMaintenance(
        sqlite,
        scopes=event_store,
        archive_path=lambda: Path(provider._hermes_home) / "memory-archive" / "events-archived.jsonl",
        current_data_dir=lambda: provider._data_dir,
        has_primary_private_owner=lambda: (
            provider._agent_context == "primary"
            and bool(provider._owner_key)
            and not group_scene_active(provider)
        ),
        upsert_chroma=storage.upsert_vector,
        current_collection=storage.current_collection,
    )
    provider._event_maintenance = event_maintenance

    status_store = StatusStore(
        current_data_dir=lambda: provider._data_dir,
        current_session_id=lambda: provider._session_id,
    )
    provider._status_store = status_store

    ingest = Ingest(sqlite, IngestDependencies(
        current_data_dir=lambda: provider._data_dir,
        emotion_ttl_days=provider._emotion_ttl_days,
        current_model=provider.runtime_llm,
        status=status_store,
        write_sample=storage.write_sample,
        record_user_active=lambda: provider._read_gate.record_user_active(),
        upsert_vector=storage.upsert_vector,
        current_insert_event_row=lambda: event_store.insert_event_row,
        current_find_duplicate_candidates=lambda: event_store.find_duplicate_candidates,
        current_record_event_occurrence=lambda: event_store.record_event_occurrence,
        grounded_protected_ids=event_maintenance.grounded_protected_ids,
        append_event_evidence=event_store.append_event_evidence,
        reinforce_event=event_store.reinforce_event,
        reinforce_expiring=event_store.reinforce_expiring,
        reinforce_on_recall=lambda ids, via, now, session_id: event_store.reinforce_on_recall(
            ids, via=via, now=now, session_id=session_id,
        ),
        set_shared_view=lambda event_id, text, touch: event_store.set_shared_view(event_id, text, touch=touch),
        current_session_id=lambda: provider._session_id,
        group_grant=lambda: group_scope_grant(provider),
        group_scene_active=lambda: group_scene_active(provider),
        write_owner=lambda: write_owner(provider),
        is_configured_private_owner=lambda: is_configured_private_owner(provider),
    ))
    provider.ingest = ingest

    promotion_review = PromotionReview(PromotionReviewDependencies(
        sqlite=sqlite,
        current_session_id=lambda: provider._session_id,
        memory_dir=lambda: Path(provider._hermes_home) / "memories",
        write_owner=lambda: write_owner(provider),
        read_scope_clause=event_store.read_scope_clause,
        event_source_context=event_store.event_source_context,
        get_event_review=event_review_store.get_event_review,
        put_event_review=event_review_store.put_event_review,
        review_candidate=lambda text: provider._llm_guard.review_candidate(text, full_context=True),
        model_available=lambda: provider._llm is not None,
        distill_daily_cap=lambda: provider._distill_daily_cap,
        resident_review_daily_cap=lambda: provider._resident_review_daily_cap,
        read_status=status_store.read,
        mutate_status=status_store.mutate,
        roll_status_date=status_store.roll_date,
        write_candidate_status=lambda ok, now, error: write_maintenance_status(
            provider, "resident_review", ok=ok, stage="promotion_candidate", error=error, now=now,
        ),
    ))
    provider._promotion_review = promotion_review

    provider._daily_profile_key = ""
    provider._llm_guard = LlmGuard(
        sqlite=sqlite,
        persona_name=provider._persona_name,
        partner_name=provider._partner_name,
        partner_role=provider._partner_role,
        partner_pronoun=provider._partner_pronoun,
        current_llm=provider.runtime_llm,
        current_session_id=lambda: provider._session_id,
        current_data_dir=lambda: provider._data_dir,
        current_profile_key=lambda: provider._daily_profile_key,
        current_platform=lambda: provider._platform,
        current_context=lambda: provider._agent_context,
        authorized_cron_owner=lambda session_id="": authorized_free_think_owner(provider, session_id),
        write_owner=lambda: write_owner(provider),
        apply_private_identity=lambda scope=None: apply_private_identity(provider, scope),
        write_inner_state=lambda text, *, owner_key, turn_id="", now=None, internal=False, max_retries=DEFAULT_CAS_MAX_RETRIES: (
            provider._inner_state.write_inner_state(
                text, owner_key=owner_key, turn_id=turn_id, now=now,
                internal=internal, max_retries=max_retries,
            )
        ),
        candidate_text=_candidate_text,
        upsert_chroma=storage.upsert_vector,
        write_status=status_store.record_write,
    )

    provider._daily_owner_key = ""
    provider._daily_owner_platform = ""
    provider._daily_identity = ()
    provider._maintainer_id = ""
    daily_state = DailyState(
        sqlite,
        current_author_kind=lambda: "maintainer" if provider._daily_owner_key else "unknown",
    )
    provider._daily_state = daily_state
    daily_context = DailyContext(
        sqlite,
        daily_state,
        config=provider._config,
        current_binding=lambda: _daily_binding(provider),
        current_model=lambda: provider._llm,
        persona_name=provider._persona_name,
    )
    provider._daily_context = daily_context

    provider._association_state = AssociationState()
    memory_axis = MemoryAxis(
        sqlite,
        current_owner=lambda: provider._owner_key,
        daily_owner=lambda: provider._daily_owner_key,
        write_owner=lambda: write_owner(provider),
        read_owner_keys=event_store.read_owner_keys_for,
        current_recall_event_ids=lambda: current_recall_event_ids(provider),
        read_status=status_store.read,
        write_maintenance_status=lambda component, *, ok, stage, error="", now=None, data_dir=None, **extra: (
            write_maintenance_status(
                provider, component, ok=ok, stage=stage, error=error, now=now, data_dir=data_dir, **extra,
            )
        ),
    )
    provider._memory_axis = memory_axis
    inner_state = InnerStateService(InnerStateDependencies(
        current_target=lambda: (provider._data_dir.resolve(), provider._daily_profile_key),
        current_session_id=lambda: provider._session_id,
        current_read_owners=lambda: provider._read_owner_keys,
        current_maintainer=lambda: provider._daily_owner_key,
        current_persona=lambda: InnerStatePersona(
            provider._persona_name,
            provider._partner_name,
            provider._partner_role,
            provider._partner_pronoun,
        ),
        current_model=provider.runtime_llm,
        mutate_status=status_store.mutate,
        record_flash=memory_axis.record_inner_state_flash,
    ))
    provider._inner_state = inner_state

    provider._association = Association(
        sqlite,
        provider._association_state,
        AssociationDependencies(
            search=lambda args, **kwargs: provider._retrieval.search(args, **kwargs),
            public_search=lambda args: provider._retrieval.public_search(args),
            daily_recall=daily_state.list_daily_entries_for_recall,
            read_owner=lambda: association_read_owner(provider),
            read_owner_keys=event_store.read_owner_keys_for,
            shadow_enabled=provider.shadow_enabled,
            relative_date=ReadGate.relative_date,
            triggers_explicit_past=ReadGate.triggers_explicit_past,
            is_trivial_message=ReadGate.is_trivial_message,
            is_busy_context=ReadGate.is_busy_context,
        ),
    )

    memes = MemeStore(
        sqlite,
        MemeRuntime(
            data_dir=lambda: provider._data_dir,
            archive_home=lambda: Path(provider._hermes_home),
            session_id=lambda: provider._session_id,
            shadow_enabled=provider.shadow_enabled,
            private_chat_owner=lambda: private_chat_owner(provider),
            association_read_owner=lambda: association_read_owner(provider),
            write_owner=lambda: write_owner(provider),
            read_owner_keys=event_store.read_owner_keys_for,
            is_configured_private_owner=lambda: is_configured_private_owner(provider),
        ),
    )
    provider._memes = memes

    commitments = Commitments(
        sqlite,
        current_context=lambda: _commitment_context(provider),
        current_group=lambda: _commitment_group(provider),
        read_owner_keys=event_store.read_owner_keys_for,
        write_fulfillment=ingest.write_event,
        has_failure=lambda: _has_commitment_failure(provider),
        write_status=lambda *, ok, stage, error="", pending=None: write_maintenance_status(
            provider, COMMITMENT_STATUS_COMPONENT, ok=ok, stage=stage, error=error, pending=pending,
        ),
        complete=lambda prompt: _complete_commitment_prompt(provider, prompt),
    )
    provider._commitments = commitments

    read_gate = ReadGate(
        sqlite,
        status_store,
        settings.read_gate_options,
        ReadGateDependencies(
            home=lambda: Path(provider._hermes_home),
            data_dir=lambda: provider._data_dir,
            profile_key=lambda: provider._daily_profile_key,
            session_id=lambda: provider._session_id,
            shadow_enabled=provider.shadow_enabled,
            is_cron=lambda: provider.is_cron_context,
            in_group_scene=lambda: group_scene_active(provider),
            group_grant=lambda: group_scope_grant(provider),
            private_chat_owner=lambda: private_chat_owner(provider),
            free_think_owner=lambda session_id="": authorized_free_think_owner(provider, session_id),
            is_configured_private_owner=lambda: is_configured_private_owner(provider),
            can_write_private_events=lambda: event_store.private_owner_write_clause() is not None,
            provenance_owner=lambda: _read_gate_provenance_owner(provider),
            read_owner_keys=event_store.read_owner_keys_for,
            group_events=lambda query: provider._retrieval.search({"query": query}, with_source=True),
            scheduled_events=lambda query: provider._retrieval.search(
                {"query": query, "top_k": 10}, for_injection=True, with_source=True,
            ),
            reset_asked=lambda before_date: event_store.reset_asked_all(before_date=before_date),
            mark_asked=lambda ids, via, session_id: event_store.mark_asked(ids, via, session_id),
            reinforce_recall=lambda ids, via, session_id: event_store.reinforce_on_recall(
                ids, via=via, session_id=session_id,
            ),
            resident_capacity=lambda: provider._resident.store(),
            curiosity_threads=daily_context.active_curiosity_threads,
            pending_group_commitments=lambda: commitments.render_pending_commitments(""),
            associate=lambda query, purpose, session_id, max_chars, gap_continuation: (
                _associate_for_prefetch(
                    provider, query, purpose, session_id, max_chars, gap_continuation,
                )
            ),
        ),
    )
    provider._read_gate = read_gate

    grounded_review = GroundedReview(
        sqlite,
        current_context=lambda: _grounded_review_context(provider),
        read_resident_texts=promotion_review.read_resident_texts,
        read_owner_keys=event_store.read_owner_keys_for,
        get_occurrence=event_store.get_grounded_occurrence,
        get_review=event_review_store.get_event_review,
        write_review=event_review_store.put_event_review,
        candidate_text=_candidate_text,
        is_bound_speaker=lambda profile, platform, actor, scope_id, anchor: (
            _is_grounded_evidence_owner(provider, profile, platform, actor, scope_id, anchor)
        ),
        review_candidate=lambda now, context: _review_grounded_candidate(provider, now, context),
    )
    provider._grounded_review = grounded_review

    resident = ResidentMemory(ResidentDependencies(
        sqlite=sqlite,
        home=lambda: Path(provider._hermes_home),
        data_dir=lambda: provider._data_dir,
        session_id=lambda: provider._session_id,
        owner_key=lambda: provider._owner_key,
        daily_owner_key=lambda: provider._daily_owner_key,
        agent_context=lambda: provider._agent_context,
        shadow_enabled=provider.shadow_enabled,
        is_configured_private_owner=lambda: is_configured_private_owner(provider),
        is_system_turn=is_system_turn,
        read_owner_keys=event_store.read_owner_keys_for,
        review_grounded_user=grounded_review.review_grounded_user,
        grounded_binding=grounded_review.grounded_binding,
        commit_recognition_support=event_review_store.commit_recognition_support,
        get_grounded_occurrence=event_store.get_grounded_occurrence,
        is_qualified_occurrence=grounded_review.qualifies_occurrence,
        review_promotion_candidate=promotion_review.review_promotion_candidate,
        get_event_review=event_review_store.get_event_review,
        event_exists=lambda event_id: event_store.get_event_row_by_id(event_id) is not None,
        promote_events=event_maintenance.promote_events,
        review_user_demotion=lambda candidates, *, resident_user="", now=None: (
            provider._llm_guard.review_user_demotion(
                candidates, resident_user=resident_user, now=now,
            )
        ),
        memory_capacity_snapshot=read_gate.memory_capacity_snapshot,
        write_maintenance_status=lambda component, *, ok, stage, error="", now=None, data_dir=None, **extra: (
            write_maintenance_status(
                provider, component, ok=ok, stage=stage, error=error, now=now, data_dir=data_dir, **extra,
            )
        ),
        topic=event_topic,
        keywords=event_keywords,
        upsert_vector=storage.upsert_vector,
    ))
    provider._resident = resident

    inner_wiring = InnerWiring(
        sqlite,
        current_home=lambda: Path(provider._hermes_home),
        current_data_dir=lambda: provider._data_dir,
        current_profile=lambda: provider._daily_profile_key,
        read_owners=lambda: provider._read_owner_keys,
        partner_name=provider._partner_name,
        partner_role=provider._partner_role,
        private_turn_owner=lambda session_id: _private_inner_turn_owner(provider, session_id),
        authorized_cron_owner=lambda session_id="": authorized_free_think_owner(provider, session_id),
        is_maintainer=lambda: is_configured_private_owner(provider),
        group_grant=lambda: group_scope_grant(provider),
        render_maintainer_daily_facts=daily_context.render_maintainer_daily_facts,
        render_commitments=commitments.render_pending_commitments,
        inner_state_target=inner_state.inner_state_target,
        latest_inner_state=lambda target: inner_state.latest_inner_state(target=target),
        render_inner_state=inner_state.render_inner_state,
        write_inner_state=inner_state.write_inner_state,
        update_inner_state=inner_state.update_inner_state,
        record_inner_state_failure=inner_state.record_inner_state_failure,
        record_delivered_rementions=memory_axis.record_delivered_rementions,
        resolve_explored_curiosity=daily_context.resolve_explored_curiosity_threads,
    )
    provider._inner_wiring = inner_wiring

    retrieval = Retrieval(RetrievalDependencies(
        sqlite=sqlite,
        current_data_dir=lambda: provider._data_dir,
        current_collection=storage.current_collection,
        current_session_id=lambda: provider._session_id,
        current_owner=lambda: provider._owner_key,
        is_primary=lambda: provider._agent_context == "primary",
        shadow_enabled=provider.shadow_enabled,
        group_grant=lambda: group_scope_grant(provider),
        group_scene_active=lambda: group_scene_active(provider),
        private_chat_owner=lambda: private_chat_owner(provider),
        free_think_owner=lambda session_id="": authorized_free_think_owner(provider, session_id),
        association_read_owner=lambda: association_read_owner(provider),
        association_read_valid=lambda: association_read_valid(provider),
        write_owner=lambda: write_owner(provider),
        scheduled_allowed=read_gate.scheduled_allowed,
        mark_asked=read_gate.mark_asked,
        record_scheduled_ask=read_gate.record_scheduled_ask,
        relative_date=ReadGate.relative_date,
        shared_topic_label=SHARED_TOPIC_LABEL,
        keywords=event_keywords,
        display_text=ingest.display_text,
        query_events=event_store.query_events,
        query_tier_window=event_store.query_tier_window,
        query_shared_views=event_store.query_shared_views,
        query_public_projections=lambda now: event_store.query_public_projections(now, unscoped=True),
        query_event_occurrences=event_store.query_event_occurrences,
        writable_event_ids=lambda ids, owner: event_store.event_ids_existing(ids, owner_key=owner, writable=True),
        delete_events=event_maintenance.delete_events,
        reinforce_on_recall=event_store.reinforce_on_recall,
        video_memory_items=video_memory.video_memory_items,
        write_group_event=ingest.write_event,
    ))
    provider._retrieval = retrieval

    lifecycle = Lifecycle(
        sqlite,
        LifecycleDependencies(
            current_data_dir=lambda: provider._data_dir,
            current_session_id=lambda: provider._session_id,
            current_persona_name=lambda: provider._persona_name,
            current_model=provider.runtime_llm,
            current_promotion_owner=lambda: provider._owner_key or provider._daily_owner_key,
            is_primary=lambda: provider._agent_context == "primary",
            group_scene_active=lambda: group_scene_active(provider),
            private_chat_owner=lambda: private_chat_owner(provider),
            group_maintenance_owners=lambda: group_maintenance_owners(provider),
            distill_daily_cap=lambda: provider._distill_daily_cap,
            promote_explicit_save=lambda event_id, owner, now: memory_axis.promote_explicit_save(
                event_id, owner_key=owner, now=now,
            ),
            promote_events=event_maintenance.promote_events,
            count_distilled_between=event_maintenance.count_distilled_between,
            list_distill_candidates=event_maintenance.list_distill_candidates,
            review_maintenance_candidate=lambda event_id, now: promotion_review.review_promotion_candidate(
                event_id, now, maintenance=True,
            ),
            can_review_candidate=promotion_review.can_review_candidate,
            note_candidate_call=promotion_review.note_candidate_call,
            get_distillable_event=event_maintenance.get_distillable_event,
            promote_distilled_event=event_maintenance.promote_distilled_event,
            keywords=event_keywords,
            upsert_chroma=storage.upsert_vector,
            expire_to_review=event_maintenance.expire_to_review,
            review_expiring=event_maintenance.review_expiring,
            purge_events=event_maintenance.purge_events,
            collapse_forgetting_curve=event_maintenance.collapse_forgetting_curve,
            recent_merge_rows=event_maintenance.recent_merge_rows,
            grounded_protected_ids=event_maintenance.grounded_protected_ids,
            merge_duplicate_events=event_maintenance.merge_duplicate_events,
            resident_cold_report=resident.cold_report,
            daily_rollover=read_gate.daily_rollover,
            decay_confidence=event_maintenance.decay_confidence,
            decay_expiring_importance=event_maintenance.decay_expiring_importance,
            status=status_store,
        ),
        distill_threshold=settings.distill_threshold,
        expiry_review_days=settings.expiry_review_days,
        expiring_importance_decay=settings.expiring_importance_decay,
        expiring_importance_floor=settings.expiring_importance_floor,
    )
    provider._lifecycle = lifecycle

    def _tool_inner_state(
        args: Mapping[str, object],
        *,
        session_id: str = "",
        turn_id: str = "",
        synthetic: bool = False,
        **_: object,
    ) -> dict[str, str]:
        return inner_wiring.tool_inner_state(
            operation=args.get("operation"),
            text=args.get("text"),
            session_id=session_id,
            turn_id=turn_id,
            synthetic=synthetic,
        )

    def _tool_impression(
        args: Mapping[str, object],
        *,
        session_id: str = "",
        turn_id: str = "",
        synthetic: bool = False,
        **_: object,
    ) -> dict[str, str]:
        return inner_wiring.tool_impression(
            operation=args.get("operation"),
            text=args.get("text"),
            session_id=session_id,
            turn_id=turn_id,
            synthetic=synthetic,
        )

    provider._tool_router = ToolRouter(
        sqlite=sqlite,
        memes=memes,
        resident=resident,
        current_session_id=lambda: provider._session_id,
        authorized_free_think_owner=lambda session_id="": authorized_free_think_owner(provider, session_id),
        private_chat_owner=lambda: private_chat_owner(provider),
        has_group_grant=lambda: group_scope_grant(provider) is not None,
        add=ingest.tool_add,
        search=retrieval.tool_search,
        forget=retrieval.tool_forget,
        compact=lifecycle.compact,
        marked_promoted=lifecycle.marked_promoted,
        distill=lifecycle.distill,
        review_migration_text=promotion_review.review_migration_text,
        review_outcome=provider.review_outcome,
        record_false_trigger=read_gate.record_false_trigger,
        shadow_enabled=provider.shadow_enabled,
        runtime_config=provider.runtime_config,
        plugin_config=provider._config,
        inner_state=_tool_inner_state,
        impression=_tool_impression,
    )
