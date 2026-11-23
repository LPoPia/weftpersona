from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional, Protocol, TypeVar

from sanae_mem.governance.runtime_registry import resolve_active_provider
from ..governance.cognitive_context import ViveMindCognitivePromptPolicy, ViveMindInboundMergePolicy
from ..governance.llm_guard import ResidentRouteOutcome
from .group_context import ViveMindGroupContextPolicy
from .group_reply import ViveMindGroupReplyPolicy
from ..infra.host_bridge import (
    current_cron_binding, get_hermes_home, group_reply_decision,
    get_cognitive_prompt_policy, register_cognitive_prompt_policy, unregister_cognitive_prompt_policy,
    get_group_context_policy, register_group_context_policy, unregister_group_context_policy,
    get_group_reply_policy, register_group_reply_policy, unregister_group_reply_policy,
    get_inbound_merge_policy, register_inbound_merge_policy, unregister_inbound_merge_policy,
    get_memory_mutation_gate, register_memory_mutation_gate, unregister_memory_mutation_gate,
)
from ..governance.memory_guard import ViveMindMemoryMutationGate
from .provider import SanaeMemProvider
from .scheduled import current_free_think
from ..infra.textutil import _bool

logger = logging.getLogger(__name__)
_lock = threading.RLock()


class _HomeRegistry:
    def __init__(self) -> None:
        self.stacks: dict[Path, list[SanaeMemProvider]] = {}

    def current(self, home: Path) -> SanaeMemProvider | None:
        with _lock:
            stack = self.stacks.get(home)
            return stack[-1] if stack else None

    def config(self) -> dict[str, Any]:
        home = get_hermes_home().resolve()
        provider = self.current(home)
        return provider.runtime_config(home) if provider is not None else {}

    def active(self) -> SanaeMemProvider | None:
        home = get_hermes_home().resolve()
        provider = self.current(home)
        if provider is not None and not _bool(provider.runtime_config(home).get("shadow"), False):
            return provider
        return None


class _HomeRoutedCognitivePromptPolicy(ViveMindCognitivePromptPolicy):
    def __init__(self, registry: _HomeRegistry):
        self._home_registry = registry
        super().__init__(self._load_current_config)

    def _active(self) -> SanaeMemProvider | None:
        return self._home_registry.active()

    def covers_builtin_memory_for_current_home(self) -> bool:
        return self._active() is not None

    def _load_current_config(self) -> dict[str, Any]:
        return self._home_registry.config()

    def resolve_turn_policy(self, envelope: Any) -> Any:
        return None if self._active() is None else super().resolve_turn_policy(envelope)

    def assemble_system_prompt_parts(
        self, agent: Any, parts_bundle: Any, system_message: Any, envelope: Any,
    ):
        if not self.covers_builtin_memory_for_current_home():
            return None
        return super().assemble_system_prompt_parts(agent, parts_bundle, system_message, envelope)

    def is_skill_allowed(
        self,
        skill_name: str,
        category: str,
        description: str,
        envelope: Any = None,
    ) -> bool:
        if self._active() is None:
            return True
        return super().is_skill_allowed(
            skill_name, category, description, envelope=envelope,
        )

    def format_skills_index_prompt(
        self, default_rendered: str, skills_by_category: Any,
        available_tools: Any, envelope: Any, **kwargs: Any,
    ):
        if self._active() is None:
            return None
        return super().format_skills_index_prompt(
            default_rendered, skills_by_category, available_tools, envelope, **kwargs,
        )

    def format_inbound_turn_timestamp(
        self, clean_content: str, event_epoch: Any, envelope: Any, tz: Any, **kwargs: Any,
    ):
        if self._active() is None:
            return None
        return super().format_inbound_turn_timestamp(
            clean_content, event_epoch, envelope, tz, **kwargs,
        )

    def compaction_config_overrides(
        self, session_id: str, platform: str, envelope: Any, **kwargs: Any,
    ):
        if self._active() is None:
            return None
        return super().compaction_config_overrides(session_id, platform, envelope, **kwargs)

    def compaction_summary_prompt(
        self, default_prompt: str, session_id: str, platform: str, envelope: Any, **kwargs: Any,
    ):
        if self._active() is None:
            return None
        return super().compaction_summary_prompt(
            default_prompt, session_id, platform, envelope, **kwargs,
        )


class _HomeRoutedMemoryMutationGate(ViveMindMemoryMutationGate):
    def __init__(self, registry: _HomeRegistry):
        self._home_registry = registry
        super().__init__(
            llm_getter=self._resolve_llm,
            on_review_status=self._write_status,
            resident_write_router=self._route_resident_write,
            cron_free_think_authorizer=self._is_authorized_free_think,
        )

    def _active(self) -> SanaeMemProvider | None:
        return self._home_registry.active()

    def _resolve_llm(self) -> Any:
        entry = self._active()
        return entry.runtime_llm() if entry is not None else None

    def _is_authorized_free_think(self, session_id: str = "") -> bool:
        binding = current_cron_binding()
        if not binding:
            return False
        return bool(current_free_think(self._home_registry.config(), session_id or binding[0]))

    def _route_resident_write(
        self,
        *,
        action: str,
        target: str,
        content: Any = None,
        old_text: Any = None,
        operations: Any = None,
        envelope: Any = None,
    ) -> Optional[ResidentRouteOutcome]:
        entry = self._active()
        if entry is None:
            return None
        home = get_hermes_home().resolve()
        session_id = str(envelope.session_id or "") if envelope is not None else ""
        turn_id = str(envelope.turn_id or "") if envelope is not None else ""
        active_provider = resolve_active_provider(home, session_id) if session_id else None
        if active_provider is None:
            logger.warning(
                "sanae_mem: no active session provider for resident memory gate "
                "(home=%s session_id=%s); using locked open_database fallback",
                home,
                session_id or "<empty>",
            )
            return None
        payload: dict[str, Any] = {"action": action}
        if content is not None:
            payload["content"] = content
        if old_text is not None:
            payload["old_text"] = old_text
        if operations is not None:
            payload["operations"] = operations
        llm_override = None if active_provider.runtime_llm() is not None else entry.runtime_llm()
        routed_json = active_provider.route_resident_write(
            target or "memory",
            payload,
            session_id=session_id,
            turn_id=turn_id,
            llm=llm_override,
        )
        if routed_json is None:
            return ResidentRouteOutcome(routed=False, route="allow")
        routed_data = json.loads(routed_json)
        return ResidentRouteOutcome(
            routed=True,
            route=str(routed_data.get("reason") or routed_data.get("routed_to") or "inner_state"),
            message=str(routed_data.get("message") or routed_json),
            inner_state_text=str(routed_data.get("inner_state_text") or ""),
            event_id=routed_data.get("event_id"),
        )

    def _write_status(self, **status: Any) -> None:
        entry = self._active()
        if entry is None:
            return
        home = get_hermes_home().resolve()
        cfg = entry.runtime_config(home)
        configured_dir = cfg.get("data_dir")
        data_dir = (
            Path(configured_dir).expanduser()
            if configured_dir
            else home / "sanae-mem"
        )
        entry.report_skill_mutation(data_dir, **status)

    def evaluate_memory_mutation(self, *args: Any, **kwargs: Any):
        if self._active() is None:
            return None
        return super().evaluate_memory_mutation(*args, **kwargs)

    def evaluate_skill_mutation(self, *args: Any, **kwargs: Any):
        if self._active() is None:
            return None
        return super().evaluate_skill_mutation(*args, **kwargs)


class _HomeRoutedInboundMergePolicy(ViveMindInboundMergePolicy):
    def __init__(self, registry: _HomeRegistry):
        self._home_registry = registry
        super().__init__()

    def _active(self) -> SanaeMemProvider | None:
        return self._home_registry.active()

    def debounce_window_seconds(self, platform: str, chat_type: str, *, busy: bool):
        if self._active() is None:
            return None
        return super().debounce_window_seconds(platform, chat_type, busy=busy)

    def should_debounce_event(self, event: Any, platform: str, chat_type: str, *, busy: bool):
        if self._active() is None:
            return None
        return super().should_debounce_event(event, platform, chat_type, busy=busy)

    def can_merge_events(
        self, existing: Any, incoming: Any, platform: str, chat_type: str, *, busy: bool,
    ):
        if self._active() is None:
            return None
        return super().can_merge_events(existing, incoming, platform, chat_type, busy=busy)

    def merge_texts(self, texts: Any, platform: str, chat_type: str):
        if self._active() is None:
            return None
        return super().merge_texts(texts, platform, chat_type)


class _HomeRoutedGroupContextPolicy(ViveMindGroupContextPolicy):
    def __init__(self, registry: _HomeRegistry):
        self._home_registry = registry
        super().__init__(self._load_current_config)

    def _load_current_config(self) -> dict[str, Any]:
        return self._home_registry.config()

    def __call__(self, source: Any, *, profile: Optional[str] = None) -> Optional[bool]:
        if self._home_registry.active() is None:
            return None
        return super().__call__(source, profile=profile)


class _HomeRoutedGroupReplyPolicy(ViveMindGroupReplyPolicy):
    def __init__(self, registry: _HomeRegistry):
        self._home_registry = registry
        super().__init__(self._load_current_config, group_reply_decision)

    def _load_current_config(self) -> dict[str, Any]:
        return self._home_registry.config()

    def __call__(self, source: Any, *, profile: Optional[str] = None) -> Any:
        if self._home_registry.active() is None:
            return None
        return super().__call__(source, profile=profile)


class _RoutedPolicy(Protocol):
    def __init__(self, registry: _HomeRegistry) -> None: ...
    @property
    def _home_registry(self) -> _HomeRegistry: ...


_Policy = TypeVar("_Policy", bound=_RoutedPolicy)


def _register_home_policy(
    home: Path, provider: SanaeMemProvider, policy_type: type[_Policy],
    getter: Callable[[], object], register: Callable[[_Policy], bool],
    unregister: Callable[[_Policy], None],
) -> Callable[[], None] | None:
    with _lock:
        existing = getter()
        policy = existing if isinstance(existing, policy_type) else policy_type(_HomeRegistry())
        registry = policy._home_registry
        registry.stacks.setdefault(home, []).append(provider)

        def release() -> None:
            with _lock:
                stack = registry.stacks.get(home)
                if stack is None or provider not in stack:
                    return
                stack.remove(provider)
                if not stack:
                    del registry.stacks[home]
                if not registry.stacks and getter() is policy:
                    unregister(policy)

        try:
            if register(policy):
                return release
        except BaseException:
            release()
            raise
        release()
        logger.warning("sanae_mem: host does not expose %s registration seam; skipped", policy_type.__name__)
        return None


def register_home_policies(home: Path, provider: SanaeMemProvider) -> Callable[[], None]:
    releases: list[Callable[[], None]] = []

    def release_all() -> None:
        for release in reversed(releases):
            release()

    def add(release: Callable[[], None] | None) -> None:
        if release is not None:
            releases.append(release)

    try:
        add(_register_home_policy(home, provider, _HomeRoutedCognitivePromptPolicy,
                                 get_cognitive_prompt_policy, register_cognitive_prompt_policy, unregister_cognitive_prompt_policy))
        add(_register_home_policy(home, provider, _HomeRoutedInboundMergePolicy,
                                 get_inbound_merge_policy, register_inbound_merge_policy, unregister_inbound_merge_policy))
        add(_register_home_policy(home, provider, _HomeRoutedGroupContextPolicy,
                                 get_group_context_policy, register_group_context_policy, unregister_group_context_policy))
        add(_register_home_policy(home, provider, _HomeRoutedGroupReplyPolicy,
                                 get_group_reply_policy, register_group_reply_policy, unregister_group_reply_policy))
        add(_register_home_policy(home, provider, _HomeRoutedMemoryMutationGate,
                                 get_memory_mutation_gate, register_memory_mutation_gate, unregister_memory_mutation_gate))
    except BaseException:
        release_all()
        raise
    return release_all
