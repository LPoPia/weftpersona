from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sanae_mem.governance.runtime_registry import (
    reconcile_registered_schedule,
    register_schedule,
    register_source_intake_sink,
    resolve_active_provider,
)
from sanae_mem.runtime.home_policies import register_home_policies

from ..infra import auxiliary
from ..governance.cognitive_context import gate_proactive_cron_delivery, prepare_proactive_cron_prompt
from ..cognitive.daily_inputs import _without_known_meme_notes
from ..infra.host_bridge import (
    PluginRegistration, cron_profile_home, get_hermes_home, has_native_turn_context,
    hermes_home_scope, is_delegated_child_turn, plugin_unload_registrar,
    register_cron_hooks, register_delivery_receipt_handler, unregister_cron_hooks,
    unregister_delivery_receipt_handler,
)
from ..governance.memory_guard import (
    guard_builtin_memory_prompt, guard_controlled_memory_write, guard_memory_write,
    guard_public_local_references, guard_public_memory_tool, remind_memory_result,
)
from .private_delivery_settler import record_turn_user_context
from .provider import SanaeMemProvider
from .scheduled import _owned_free_think_job, register_free_think_hooks, sync_scheduled_jobs
from ..governance.source_context import ChatScope, current_chat_scope
from ..governance.source_intake import SourceIntakeVerdict, SourceMaterial, ingest_source_material
from ..infra.textutil import _bool
from .delivery import DeliveryRouter, _open_private_hook_provider

logger = logging.getLogger(__name__)


def _private_hook_scope(session_id: str, turn_id: str, platform: str) -> ChatScope | None:
    scope = current_chat_scope()
    if (scope.owner and scope.session_id and scope.session_id == session_id and turn_id
            and scope.platform == platform and not is_delegated_child_turn()):
        return scope
    return None


def register_private_heart_hooks(
    ctx: PluginRegistration[SanaeMemProvider], prototype: SanaeMemProvider,
) -> Callable[[], None]:
    registered_home = get_hermes_home().resolve()
    router = DeliveryRouter(registered_home, prototype, discovery=plugin_unload_registrar(ctx) is not None)

    def append_private_heart(*, session_id="", turn_id="", platform="", user_message=None, **_):
        home = get_hermes_home().resolve()
        if Path(home).resolve() != registered_home:
            return None
        scope = _private_hook_scope(session_id, turn_id, platform)
        if scope is None or _bool(prototype.runtime_config(home).get("shadow"), False):
            return None
        worker = _open_private_hook_provider(prototype, scope, home)
        try:
            block = worker.current_turn_context(
                session_id=session_id, turn_id=turn_id, platform=platform, sender_id=scope.actor,
                sender_id_alt=scope.actor_alt, user_message=user_message,
            )
            if user_message is not None:
                active = resolve_active_provider(home, session_id)
                if active is not None and type(active) is not SanaeMemProvider:
                    authored = _without_known_meme_notes(user_message)
                    try:
                        meme_ctx = worker.meme_context_for_turn(authored, known_context=user_message)
                    except (sqlite3.Error, OSError) as exc:
                        logger.warning(
                            "sanae_mem: fallback meme context read failed (%s)",
                            type(exc).__name__,
                        )
                        meme_ctx = ""
                    if meme_ctx:
                        record_turn_user_context(home, session_id, turn_id, meme_context=meme_ctx)
            return {"context": block, "ephemeral": True} if block else None
        finally:
            worker.shutdown()

    def observe_final_text(
        *, session_id="", turn_id="", platform="", user_message=None, assistant_response=None, model="", **_,
    ):
        home = get_hermes_home()
        if Path(home).resolve() != registered_home:
            return
        scope = _private_hook_scope(session_id, turn_id, platform)
        has_user_message = (
            (isinstance(user_message, str) and bool(user_message.strip()))
            or (isinstance(user_message, (list, tuple)) and bool(user_message))
        )
        if (scope is None or _bool(prototype.runtime_config(home).get("shadow"), False)
                or not scope.message_id or not scope.session_key
                or not has_user_message
                or not isinstance(assistant_response, str) or not assistant_response.strip()):
            return
        router.observe(session_id, turn_id, scope.message_id, assistant_response)

    if not has_native_turn_context():
        ctx.register_hook("pre_llm_call", append_private_heart)
    ctx.register_hook("post_llm_call", observe_final_text)
    handler = router.on_transport_receipt
    if register_delivery_receipt_handler(handler):
        release_owner = router.attach()

        def release() -> None:
            release_owner()
            unregister_delivery_receipt_handler(handler)

        return release
    logger.warning("sanae_mem: host does not expose register_delivery_receipt_handler; delivery receipt hook skipped")
    return lambda: None


def _register_cron(home: Path, provider: SanaeMemProvider) -> Callable[[], None]:
    def prepare(job: Mapping[str, Any], prompt: str, envelope: object = None) -> str | None:
        if cron_profile_home(envelope) != home:
            return None
        config = provider.runtime_config(home)
        if _bool(config.get("shadow"), False):
            return None
        return prepare_proactive_cron_prompt(job, prompt, envelope, config=config)

    def gate(job: Mapping[str, Any], response_text: str, envelope: object = None) -> dict[str, Any] | bool | None:
        if cron_profile_home(envelope) != home:
            return None
        config = provider.runtime_config(home)
        if _bool(config.get("shadow"), False) and not _owned_free_think_job(dict(job)):
            return None
        return gate_proactive_cron_delivery(job, response_text, envelope, config=config)

    def release() -> None:
        unregister_cron_hooks(prompt_preparer=prepare, pre_delivery_gate=gate)

    try:
        if register_cron_hooks(prompt_preparer=prepare, pre_delivery_gate=gate):
            return release
    except BaseException:
        release()
        raise
    logger.warning("sanae_mem: host does not expose cron hook registration; cron hooks skipped")
    return lambda: None


def _register_intake(home: Path, provider: SanaeMemProvider) -> Callable[[], None]:
    def intake(
        prototype: SanaeMemProvider, material: SourceMaterial | Mapping[str, Any], *,
        session_id: str, create_memory_event: bool, importance: float | None,
    ) -> dict[str, Any]:
        with hermes_home_scope(home):
            config = prototype.runtime_config(home)
            if _bool(config.get("shadow"), False):
                return SourceIntakeVerdict(False, reason="provider_shadow_mode").to_dict()
            scope = current_chat_scope()
            worker = prototype.open_runtime_worker(
                home, config, session_id=session_id or "source-intake", platform=scope.platform or "cron",
                agent_context="primary" if scope.owner else "cron",
            )
            try:
                return ingest_source_material(
                    worker, material, create_memory_event=create_memory_event, importance=importance,
                    allow_configured_owner_fallback=True,
                ).to_dict()
            finally:
                worker.shutdown()

    return register_source_intake_sink(home, provider, intake)


def register_runtime_hooks(ctx: PluginRegistration[SanaeMemProvider], provider: SanaeMemProvider) -> None:
    home = get_hermes_home().resolve()
    for hook, callback in (
        ("post_gateway_admission", guard_public_local_references),
        ("pre_llm_call", guard_builtin_memory_prompt),
        ("pre_tool_call", guard_public_memory_tool),
        ("pre_tool_call", guard_memory_write),
        ("pre_tool_call", guard_controlled_memory_write),
        ("transform_tool_result", remind_memory_result),
        ("pre_llm_call", provider.handle_pre_llm_call),
    ):
        try:
            ctx.register_hook(hook, callback)
        except Exception:
            logger.exception("sanae_mem: failed to register core hook %s (%s)", hook, callback.__name__)
            raise
    try:
        provider.attach_runtime_llm(auxiliary.register(ctx))
    except Exception as exc:
        logger.warning("sanae_mem: auxiliary registration skipped (%s)", type(exc).__name__)
    try:
        auxiliary.register_daily(ctx)
    except Exception as exc:
        logger.warning("sanae_mem: daily Flash registration skipped (%s)", type(exc).__name__)

    releases: list[Callable[[], None]] = []

    def release_all() -> None:
        for release in reversed(releases):
            release()
        releases.clear()

    try:
        releases.append(register_private_heart_hooks(ctx, provider))
        releases.append(register_free_think_hooks(ctx, provider, native_turn_context=has_native_turn_context()))
        releases.append(register_home_policies(home, provider))
        releases.append(_register_cron(home, provider))
        releases.append(_register_intake(home, provider))
        releases.append(register_schedule(home, provider.runtime_config, sync_scheduled_jobs))
        on_unload = plugin_unload_registrar(ctx)
        if on_unload is None:
            provider.own_runtime_registration(release_all)
        else:
            on_unload(release_all)
    except BaseException:
        release_all()
        raise
    reconcile_registered_schedule(home)
