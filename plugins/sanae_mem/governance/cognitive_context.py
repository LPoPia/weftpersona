"""ViveMind Cognitive Context, Prompt Assembly, Timing, Compaction & Inbound Merge Policies.

Centralizes all cognitive state-machine and surface-boundary policies owned by ViveMind
(``sanae_mem``), plugging into ``hermes_host_contracts`` when available and remaining
importable and testable standalone without custom Hermes fork internals:

1. ``Tier 0 / Tier 1 / Tier 2`` anti-recency system prompt assembly:
   - Places ``BEHAVIOR.md`` (``provider_memory_parts``) at the start of ``Tier 2``
     (``volatile``) followed by unchanged ``SOUL.md`` (``identity_parts``) immediately
     before conversation history so recency bias preserves authentic persona voice
     without altering ``SOUL.md`` / ``BEHAVIOR.md`` text;
   - Strictly gates ``USER.md``, ``MEMORY.md``, ``context_files_parts``, and host
     runtime environment blocks by verified interaction surface (private maintainer DM,
     external user DM, group/public social, and bound personal cron).
2. Surface-switch & restricted public social policy (``build_public_social_turn_policy``).
3. Private chat continuity summary prompt (``## What I remember``,
   ``## Work I may need to pick up``) and verbatim-user-section suppression
   (``omit_verbatim_user_section = True``).
4. ``04:00`` Asia/Shanghai sleep-day & Chinese speaker time-gap perception
   (``[时间：...；距对方上次发言...；距我上次发言...]``), incorporating verified cron sends.
5. Chinese Skill index guidance with available-context boundary awareness
   (``## 可用技能索引``).
6. QQ consecutive message burst-merge judgment (``ViveMindInboundMergePolicy``):
   5.0s debounce window for private DM, compatible reply-target check (merging plain +
   single quote while separating distinct quote targets), slash-command & ``T:`` bypass.
7. Cron proactive silence gate & prompt context preparer.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from .account_binding import maintainer_anchor
from ..infra.clock import SHANGHAI_TZ, sleep_day_iso
from ..infra.defaults import DEFAULT_PARTNER_ROLE, DEFAULT_PERSONA_NAME
from .identity_config import daily_memory_identity
from .group_scope import GROUP_TOOL_NAMES, group_feature_enabled
from ..infra.host_bridge import (
    current_cron_binding,
    current_turn_envelope,
    host_chat_fields,
    is_delegated_child_turn,
    is_test_isolated_turn,
    query_verified_cron_sends,
)
from .free_think_state import _FREE_THINK_PROMPT, _owned_free_think_job, free_think_can_deliver
from .source_context import (
    PRIVATE_CHAT_TYPES, chat_scope_for, current_chat_scope, resolve_chat_identity,
    scope_from_source_facts,
)
from ..infra.textutil import _load_plugin_config, _owner_key

try:
    import hermes_host_contracts as _hc
except ImportError:
    _hc = None  # type: ignore[assignment]

_PromptBase = _hc.CognitivePromptPolicy if _hc is not None else object
_MergeBase = _hc.InboundMergePolicy if _hc is not None else object

_WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
_QQ_PLATFORMS = frozenset({"qqbot", "qq"})
# Platforms whose private-chat lane this policy may merge on. Slack/Discord join QQ: the turn
# itself is still the host's, the judgment of "same thought split over messages" is ours.
_DM_MERGE_PLATFORMS = _QQ_PLATFORMS | frozenset({"slack", "discord"})
_QQ_DM_TYPES = frozenset({"dm", "private", "direct", "c2c"})
_SILENT_TOKENS = frozenset({"[SILENT]", "SILENT", "HEARTBEAT_OK", "[HEARTBEAT_OK]"})
_PRIVATE_DM_SURFACES = frozenset({"private_maintainer", "private_external"})
_PRIVATE_DM_STRIPPED_GUIDANCE_PREFIXES = (
    "## Mid-turn user steering",
    "# Finishing the job",
    "# Parallel tool calls",
    "# Tool-use enforcement",
    "# Google model operational directives",
    "# Execution discipline",
)
_QQ_DM_BLOCKED_SKILL_CATEGORIES = frozenset({
    "software-development",
    "autonomous-ai-agents",
    "apple",
    "email",
    "note-taking",
    "creative",
})

DEFAULT_PUBLIC_ALLOWED_TOOLS: frozenset[str] = frozenset({
    "sanae_mem_search",
    "sanae_meme_search",
})

RESTRICTED_TOOL_NAMES: frozenset[str] = frozenset({
    "terminal",
    "execute_code",
    "write_file",
    "patch",
    "read_file",
    "search_files",
    "skill_manage",
    "delegate_task",
    "memory",
    "session_search",
    "cronjob",
    "process",
    "computer_use",
})


def _restricted_tool_name(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    return normalized in RESTRICTED_TOOL_NAMES or normalized.startswith("browser_")


def _restricted_allowed_tools(extra_allowed_tools: Sequence[str] = ()) -> frozenset[str]:
    allowed = set(DEFAULT_PUBLIC_ALLOWED_TOOLS)
    allowed.update(
        str(item).strip()
        for item in extra_allowed_tools
        if item and str(item).strip() and not _restricted_tool_name(str(item))
    )
    return frozenset(allowed)


def _envelope_surface_flags(envelope: Any) -> tuple[str, bool]:
    """Return ``(platform, is_delegated_child)`` from *envelope* or active scope."""
    if envelope is not None:
        fields = host_chat_fields(envelope)
        return fields.platform, fields.is_delegated_child
    scope = current_chat_scope()
    return scope.platform, is_delegated_child_turn()


def _filter_private_dm_guidance(guidance_parts: Sequence[Any]) -> list[str]:
    """Strip host-level English operational directives while retaining memory/skill rules."""
    kept: list[str] = []
    for part in guidance_parts:
        if not isinstance(part, str) or not part.strip():
            continue
        if part.lstrip().startswith(_PRIVATE_DM_STRIPPED_GUIDANCE_PREFIXES):
            continue
        kept.append(part)
    return kept


def _has_chinese_skill_description(description: str) -> bool:
    """Return True when a skill description carries a Chinese main description."""
    cleaned = str(description or "").strip()
    if cleaned.startswith("[project]"):
        cleaned = cleaned[len("[project]"):].strip()
    if not cleaned:
        return False
    cjk_count = sum(1 for ch in cleaned if "\u4e00" <= ch <= "\u9fff")
    if cjk_count < 2:
        return False
    ascii_alpha = sum(1 for ch in cleaned if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
    first_alpha = next((ch for ch in cleaned if ch.isalpha()), "")
    starts_cjk = bool(first_alpha and "\u4e00" <= first_alpha <= "\u9fff")
    return starts_cjk or cjk_count * 4 >= ascii_alpha


def _skill_entry_pair(entry: Any) -> tuple[str, str]:
    if isinstance(entry, Mapping):
        name = str(entry.get("load_name") or entry.get("name") or "").strip()
        desc = str(entry.get("description") or entry.get("desc") or "").strip()
        return name, desc
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) >= 2:
        return str(entry[0] or "").strip(), str(entry[1] or "").strip()
    return "", ""


def _is_qq_main_dm(envelope: Any, config: Mapping[str, Any]) -> bool:
    """Return True iff the active turn is a non-delegated QQ private DM."""
    surface = classify_turn_surface(envelope, config)
    plat, is_delegated_child = _envelope_surface_flags(envelope)
    return (
        surface in _PRIVATE_DM_SURFACES
        and plat in _QQ_PLATFORMS
        and not is_delegated_child
    )


def _is_qq_dm_skill_allowed(category: str, description: str) -> bool:
    """Return True iff a skill passes QQ main-DM category & Chinese description filters."""
    top_cat = str(category or "").strip().split("/", 1)[0].lower()
    if top_cat in _QQ_DM_BLOCKED_SKILL_CATEGORIES:
        return False
    return _has_chinese_skill_description(description)


def _filter_qq_dm_skills(
    skills_by_category: Mapping[str, Sequence[Any]],
) -> dict[str, list[tuple[str, str]]]:
    """Filter out pure-English engineering/office/design skills for QQ main DMs."""
    filtered: dict[str, list[tuple[str, str]]] = {}
    for category, entries in (skills_by_category or {}).items():
        cat_name = str(category or "").strip()
        top_cat = cat_name.split("/", 1)[0].lower()
        if top_cat in _QQ_DM_BLOCKED_SKILL_CATEGORIES:
            continue
        kept_entries: list[tuple[str, str]] = []
        for entry in entries or ():
            name, desc = _skill_entry_pair(entry)
            if not name:
                continue
            if _is_qq_dm_skill_allowed(cat_name, desc):
                kept_entries.append((name, desc))
        if kept_entries:
            filtered[cat_name] = kept_entries
    return filtered


def _demoted_categories_from_index_lines(index_lines: Sequence[str]) -> set[str]:
    demoted: set[str] = set()
    marker = " [names only]:"
    for line in index_lines:
        if not isinstance(line, str):
            continue
        stripped = line.strip()
        if marker in stripped:
            cat = stripped.split(marker, 1)[0].strip()
            if cat:
                demoted.add(cat)
    return demoted


def _render_filtered_skill_index_lines(
    skills_by_category: Mapping[str, Sequence[tuple[str, str]]],
    demoted: set[str],
) -> list[str]:
    index_lines: list[str] = []
    for category in sorted(skills_by_category):
        entries = skills_by_category[category]
        if category in demoted:
            names = ", ".join(sorted({name for name, _ in entries if name}))
            index_lines.append(f"  {category} [names only]: {names}")
            continue
        rows: dict[str, str] = {}
        for name, desc in sorted(entries, key=lambda item: item[0]):
            rows.setdefault(name, desc)
        indent = "  " if len(rows) == 1 else "    "
        if len(rows) > 1:
            index_lines.append(f"  {category}:")
        index_lines.extend(
            f"{indent}- {name}: {desc}" if desc else f"{indent}- {name}"
            for name, desc in rows.items()
        )
    return index_lines


def _clean_filtered_hidden_note(
    hidden_note: str,
    kept_categories: set[str],
    demoted_categories: set[str],
) -> str:
    if not hidden_note:
        return ""
    has_kept_demoted = bool(kept_categories & demoted_categories)
    kept_top_cats = {cat.split("/", 1)[0].lower() for cat in kept_categories}
    kept_lines: list[str] = []
    for raw_line in hidden_note.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "[names only]" in line and not has_kept_demoted:
            continue
        lowered = line.lower()
        if any(
            blocked in lowered and blocked not in kept_top_cats
            for blocked in _QQ_DM_BLOCKED_SKILL_CATEGORIES
        ):
            continue
        kept_lines.append(raw_line)
    return ("\n" + "\n".join(line.lstrip("\n") for line in kept_lines)) if kept_lines else ""


def _render_chinese_skills_prompt(index_lines: Sequence[str], hidden_note: str) -> str:
    return (
        "## 可用技能索引\n"
        "回复前先浏览下方技能目录。若某项技能与当前事项或话题有关联，先调用 skill_view(name) 查看完整说明再处理。\n"
        "使用技能时请结合当前对话场合实际可得的上下文与工具范围行事：私聊、定时活动与公开场合各自可见的记忆、"
        "本地文件及可用工具不同；若某项技能提到当前场合未开放的工具或上下文，不必向对方解释机制，顺着当前可得条件自然处理即可。\n"
        "发现技能步骤有误或遗漏时可用 skill_manage(action='patch') 更新；跑通复杂新流程后也可整理保存为技能。\n\n"
        "<available_skills>\n"
        + "\n".join(index_lines)
        + "\n</available_skills>\n\n"
        "仅在确认下方没有任何相关技能时，才直接处理。"
        + hidden_note
    )


def _join_non_empty(parts: Sequence[str]) -> str:
    return "\n\n".join(str(p).strip() for p in parts if isinstance(p, str) and str(p).strip())


def format_duration_zh(seconds: float) -> str:
    """Render elapsed seconds into natural Chinese conversational duration."""
    delta = max(0, int(round(seconds)))
    if delta < 60:
        return "刚刚"
    minutes, _ = divmod(delta, 60)
    if minutes < 60:
        return f"{minutes}分钟"
    hours, rem_min = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}小时{rem_min}分钟" if rem_min else f"{hours}小时"
    days, rem_hours = divmod(hours, 24)
    return f"{days}天{rem_hours}小时" if rem_hours else f"{days}天"


def _sleep_day_gap_days(prev_dt: datetime, curr_dt: datetime) -> int:
    try:
        d_prev = datetime.fromisoformat(sleep_day_iso(prev_dt)).date()
        d_curr = datetime.fromisoformat(sleep_day_iso(curr_dt)).date()
        return max(0, (d_curr - d_prev).days)
    except Exception:
        return 0


def _extract_history_timestamps(
    history: Sequence[Any],
    *,
    before_epoch: float,
) -> tuple[float | None, float | None]:
    """Return ``(last_user_ts, last_self_ts)`` strictly preceding ``before_epoch``."""
    last_user: float | None = None
    last_self: float | None = None
    for item in history or ():
        if not isinstance(item, Mapping):
            continue
        if item.get("_compressed_summary") or item.get("tool_call_id"):
            continue
        display_kind = str(item.get("display_kind") or "").strip()
        if display_kind in {"hidden", "failed_turn", "heartbeat", "system_notice"}:
            continue
        ts_raw = item.get("timestamp")
        if isinstance(ts_raw, bool) or not isinstance(ts_raw, (int, float)):
            continue
        ts = float(ts_raw)
        if ts <= 0 or ts >= before_epoch:
            continue
        role = str(item.get("role") or "")
        if role == "user":
            if last_user is None or ts > last_user:
                last_user = ts
        elif role == "assistant" and not item.get("tool_calls"):
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                if last_self is None or ts > last_self:
                    last_self = ts
    return last_user, last_self


def _verified_private_scope(envelope: Any):
    """The verified one-to-one scope of a DM envelope, or ``None`` when it is not private.

    Surface classification does not need a live session key; the platform-stated scene still
    lets the same identity resolver decide this DM.
    """
    scope = chat_scope_for(envelope) if envelope is not None else current_chat_scope()
    if scope.visibility != "private" and envelope is not None:
        scope = scope_from_source_facts({
            "platform": str(getattr(envelope, "platform", "") or ""),
            "chat_type": str(getattr(envelope, "chat_type", "") or ""),
            "chat_id": str(getattr(envelope, "chat_id", "") or ""),
            "user_id": str(getattr(envelope, "sender_id", "") or ""),
            "user_id_alt": str(getattr(envelope, "sender_id_alt", "") or ""),
            "scope_id": str(getattr(envelope, "scope_id", "") or ""),
            "profile": str(getattr(envelope, "profile", "") or ""),
            "is_bot": getattr(envelope, "is_bot", None),
        })
    return scope if scope.visibility == "private" else None


def classify_turn_surface(
    envelope: Any = None,
    config: Mapping[str, Any] | None = None,
    agent: Any = None,
) -> str:
    """Classify the active turn surface under ViveMind's authority.

    Returns one of:
    - ``"restricted_public"``: explicit restricted/public audience, group/channel, or unknown remote surface
    - ``"private_maintainer"``: verified interactive DM with configured ``memory_identity``
    - ``"private_external"``: verified interactive DM with a non-maintainer person
    - ``"personal_cron"``: verified cron run owned by configured ``memory_identity``
    - ``"local_operator"``: local CLI / TUI / operator console turn
    """
    cfg = dict(_load_plugin_config() if config is None else config)
    fixed_plat, fixed_uid, fixed_alt = daily_memory_identity(cfg)
    maintainer_ids = {v for v in (fixed_uid, fixed_alt) if v}

    restricted_aud = str(getattr(envelope, "restricted_audience", "") or "").strip()
    if restricted_aud:
        return "restricted_public"

    plat = str(getattr(envelope, "platform", "") or "").strip().lower()
    src = str(getattr(envelope, "source", "") or "").strip().lower()
    chat_type = str(getattr(envelope, "chat_type", "") or "").strip().lower()
    chat_id = str(getattr(envelope, "chat_id", "") or "").strip()
    sender_id = str(getattr(envelope, "sender_id", "") or "").strip()
    sender_alt = str(getattr(envelope, "sender_id_alt", "") or "").strip()
    agent_plat = str(getattr(agent, "platform", "") or "").strip().lower()
    is_cron = bool(
        getattr(envelope, "is_cron", False)
        or plat == "cron"
        or src == "cron"
        or agent_plat == "cron"
        or getattr(agent, "cron_session", False)
    )

    scope_visibility = None
    if not plat and not chat_type and envelope is None:
        scope = current_chat_scope()
        plat = scope.platform
        chat_type = scope.chat_type
        chat_id = scope.chat_id
        sender_id = scope.actor
        sender_alt = scope.actor_alt
        # The live ChatScope already decided the scene (platform + speaker + session key); its
        # routing fields describe the event but must not upgrade an unknown scene to a private one.
        scope_visibility = scope.visibility

    if is_cron:
        cron_binding = current_cron_binding()
        if cron_binding is not None and fixed_plat and maintainer_ids:
            return "personal_cron"
        return "restricted_public"

    if chat_type in {"group", "channel", "thread", "guild", "public"}:
        return "restricted_public"

    if chat_type in PRIVATE_CHAT_TYPES or chat_type == "c2c":
        if not plat or not chat_id or not (sender_id or sender_alt):
            return "restricted_public"
        if scope_visibility is not None and scope_visibility != "private":
            # The envelope was omitted, so the live ChatScope already decided this scene; its
            # routing fields describe the event and must not upgrade an unknown scene to private.
            return "restricted_public"
        # Who this DM belongs to comes from the one identity resolver (account binding included);
        # the transport facts above stay untouched.
        scope = _verified_private_scope(envelope)
        if scope is None:
            return "restricted_public"
        identity = resolve_chat_identity(scope, cfg)
        return "private_maintainer" if identity.maintainer else "private_external"

    if not chat_type and not chat_id and (
        (plat or src) in {"cli", "tui", "desktop", "local"}
        or (not plat and not src and agent_plat in {"cli", "tui", "desktop", "local"})
    ):
        return "local_operator"

    return "restricted_public"


def build_public_social_turn_policy(
    *,
    config: Mapping[str, Any] | None = None,
    audience: str = "public_social",
    extra_allowed_tools: Sequence[str] = (),
    platform_context: str = "",
    action_tool_name: str = "",
    action_tool_schema: Mapping[str, Any] | None = None,
    action_dispatcher: Callable[[dict[str, Any], str | None], str | dict[str, Any]] | None = None,
) -> Any:
    """Build ViveMind's hard pre-LLM ``RestrictedTurnPolicy`` for public/social interactions."""
    cfg = dict(_load_plugin_config() if config is None else config)
    persona_name = str(cfg.get("persona_name") or DEFAULT_PERSONA_NAME).strip() or DEFAULT_PERSONA_NAME
    allowed = set(_restricted_allowed_tools(extra_allowed_tools))
    if action_tool_name and str(action_tool_name).strip() and not _restricted_tool_name(action_tool_name):
        allowed.add(str(action_tool_name).strip())

    prompt_lines = [
        f"你是{persona_name}，正在公开社交场合自然发言或回复。",
        "严守公开场合边界：只依据眼前公开可见的内容与经审核可公开的自身记忆摘录回应，"
        "绝不提及任何私聊对话、私人约定、本地文件路径、账号配置或系统内部细节。",
        "语气保持自然、简明、平实，有话则说，不必客套铺垫。",
    ]
    if platform_context and str(platform_context).strip():
        prompt_lines.append(str(platform_context).strip())
    system_prompt = "\n\n".join(prompt_lines)

    if _hc is not None:
        return _hc.RestrictedTurnPolicy(
            audience=audience,
            system_prompt="",
            allowed_tools=frozenset(allowed),
            override_system_prompt=False,
            isolate_history=True,
            skip_background_review=True,
            skip_context_files=False,
            action_tool_name=action_tool_name,
            action_tool_schema=action_tool_schema,
            action_dispatcher=action_dispatcher,
            metadata={"prompt_context": system_prompt},
        )
    return {
        "audience": audience,
        "system_prompt": "",
        "allowed_tools": frozenset(allowed),
        "override_system_prompt": False,
        "isolate_history": True,
        "skip_background_review": True,
        "skip_context_files": False,
        "action_tool_name": action_tool_name,
        "metadata": {"prompt_context": system_prompt},
    }


def _is_unbound_local_probe(envelope: Any) -> bool:
    """True only before any session, messaging, cron, child, or restricted context is bound."""
    if envelope is None:
        return False
    if (
        getattr(envelope, "restricted_audience", "")
        or getattr(envelope, "is_cron", False)
        or getattr(envelope, "is_delegated_child", False)
        or getattr(envelope, "is_messaging_surface", False)
        or getattr(envelope, "session_context_engaged", False)
    ):
        return False
    return not any(
        str(getattr(envelope, attr, "") or "").strip()
        for attr in (
            "platform", "source", "chat_type", "chat_id",
            "sender_id", "sender_id_alt", "session_id", "session_key",
        )
    )


class ViveMindCognitivePromptPolicy(_PromptBase):
    """ViveMind's unified prompt-tiering, timing, skill-index, and compaction policy."""

    def __init__(self, config_loader: Callable[[], Mapping[str, Any]] | None = None):
        self._config_loader = config_loader or _load_plugin_config

    def _config(self) -> dict[str, Any]:
        return dict(self._config_loader() or {})

    def resolve_turn_policy(self, envelope: Any) -> Any:
        """Keep normal prompt assembly while hard-gating tools on non-maintainer surfaces."""
        if _hc is None or _is_unbound_local_probe(envelope):
            return None
        surface = classify_turn_surface(envelope, self._config())
        if surface not in {"restricted_public", "private_external"}:
            return None
        allowed = _restricted_allowed_tools()
        if surface == "restricted_public":
            # The same group gate the tool schemas and tool call use: only a switched-on,
            # trustworthy group adds its own add/search/impression tools.
            config = self._config()
            scope = chat_scope_for(envelope)
            if (scope.visibility == "public" and group_feature_enabled(
                    config, scope, "long_term_memory")):
                # Exactly the schemas the group branch registers and the gate accepts.
                allowed = frozenset(GROUP_TOOL_NAMES)
        return _hc.RestrictedTurnPolicy(
            audience=surface,
            allowed_tools=allowed,
            override_system_prompt=False,
            isolate_history=True,
            skip_background_review=True,
            skip_context_files=False,
        )

    def assemble_system_prompt_parts(
        self,
        agent: Any,
        parts_bundle: Mapping[str, Any],
        system_message: str | None,
        envelope: Any,
    ) -> dict[str, str] | None:
        """Assemble ``Tier 0 (stable) -> Tier 1 (context) -> Tier 2 (volatile)``.

        Operational guidance and the host skills index stay in the stable prefix.
        Provider behavior guidance starts the volatile tail, followed by unchanged
        ``SOUL.md`` identity and configured persona immediately before conversation history.
        """
        surface = classify_turn_surface(envelope, self._config(), agent=agent)

        identity_parts = list(parts_bundle.get("identity_parts") or [])
        help_guidance = str(parts_bundle.get("help_guidance") or "")
        guidance_parts = list(parts_bundle.get("guidance_parts") or [])
        skills_prompt = str(parts_bundle.get("skills_prompt") or "")
        alibaba_parts = list(parts_bundle.get("alibaba_identity_parts") or [])
        auto_load_parts = list(parts_bundle.get("auto_load_parts") or [])
        coding_prefix_parts = list(parts_bundle.get("coding_prefix_parts") or [])
        coding_workspace_parts = list(parts_bundle.get("coding_workspace_parts") or [])
        coding_trailing_parts = list(parts_bundle.get("coding_trailing_parts") or [])
        post_workspace_parts = list(parts_bundle.get("post_workspace_parts") or [])
        context_files_parts = list(parts_bundle.get("context_files_parts") or [])
        core_memory_parts = list(parts_bundle.get("core_memory_parts") or [])
        user_profile_parts = list(parts_bundle.get("user_profile_parts") or [])
        plugin_sections = list(parts_bundle.get("plugin_sections_after_memory") or [])
        active_profile_line = str(parts_bundle.get("active_profile_line") or "")
        timestamp_line = str(parts_bundle.get("timestamp_line") or "")
        runtime_env_block = str(parts_bundle.get("runtime_environment_block") or "")

        # ── Tier 0: Stable Prefix ──
        _, is_delegated_child = _envelope_surface_flags(envelope)
        strip_host_dm_guidance = surface in _PRIVATE_DM_SURFACES and not is_delegated_child
        tier0_operational: list[str] = []
        if help_guidance and not strip_host_dm_guidance:
            tier0_operational.append(help_guidance)
        tier0_operational.extend(
            _filter_private_dm_guidance(guidance_parts) if strip_host_dm_guidance else guidance_parts
        )
        tier0_operational.extend(alibaba_parts)
        tier0_operational.extend(auto_load_parts)
        tier0_operational.extend(coding_prefix_parts)
        if skills_prompt and surface != "restricted_public":
            tier0_operational.append(skills_prompt)
        if not coding_workspace_parts:
            tier0_operational.extend(coding_trailing_parts)
            tier0_operational.extend(post_workspace_parts)

        # ── Tier 1: Conversation Context (Surface-gated files, skills & resident memory) ──
        tier1_parts: list[str] = []

        # Host context files and the runtime environment stay with the operator's fixed account
        # (or an already-local/cron turn); a cross-platform memory binding adds none of them.
        allow_private_workspace = surface in {"local_operator", "personal_cron"}
        if surface == "private_maintainer":
            scope = _verified_private_scope(envelope)
            allow_private_workspace = bool(
                scope is not None
                and resolve_chat_identity(scope, self._config()).host_operator)
        allow_user_profile = surface in {
            "private_maintainer", "private_external", "local_operator", "personal_cron"
        }
        allow_core_memory = surface in {"private_maintainer", "local_operator", "personal_cron"}

        if allow_private_workspace:
            tier1_parts.extend(context_files_parts)
            if coding_workspace_parts:
                tier1_parts.extend([*coding_workspace_parts, *coding_trailing_parts, *post_workspace_parts])

        if allow_core_memory:
            if "builtin_memory_parts" in parts_bundle:
                tier1_parts.extend(parts_bundle.get("builtin_memory_parts") or [])
            else:
                tier1_parts.extend(core_memory_parts)
        if allow_user_profile:
            tier1_parts.extend(user_profile_parts)

        prompt_context = ""
        if _hc is not None:
            with suppress(Exception):
                policy = _hc.current_restricted_turn_policy()
                prompt_context = str((getattr(policy, "metadata", {}) or {}).get("prompt_context") or "")
        if surface != "restricted_public":
            tier1_parts.extend(plugin_sections)
        if prompt_context:
            tier1_parts.append(prompt_context)
        if allow_private_workspace and active_profile_line:
            tier1_parts.append(active_profile_line)
        if allow_private_workspace and runtime_env_block:
            tier1_parts.append(runtime_env_block)
        if timestamp_line:
            tier1_parts.append(timestamp_line)

        # ── Tier 2: Behavior first; SOUL + persona immediately before history ──
        tier2_parts = list(parts_bundle.get("provider_memory_parts") or [])
        tier2_parts.extend(identity_parts)
        if system_message:
            tier2_parts.append(system_message)

        return {
            "stable": _join_non_empty(tier0_operational),
            "context": _join_non_empty(tier1_parts),
            "volatile": _join_non_empty(tier2_parts),
        }

    def is_skill_allowed(
        self,
        skill_name: str,
        category: str,
        description: str,
        envelope: Any = None,
    ) -> bool:
        resolved_envelope = envelope if envelope is not None else current_turn_envelope()
        if not _is_qq_main_dm(resolved_envelope, self._config()):
            return True
        return _is_qq_dm_skill_allowed(category, description)

    def format_skills_index_prompt(
        self,
        default_rendered: str,
        skills_by_category: Mapping[str, Sequence[Any]],
        available_tools: set[str] | None,
        envelope: Any,
        **kwargs: Any,
    ) -> str | None:
        is_qq_main_dm = _is_qq_main_dm(envelope, self._config())
        index_lines = list(kwargs.get("index_lines") or [])
        hidden_note = str(kwargs.get("hidden_note") or "")
        if not is_qq_main_dm:
            if not index_lines:
                return default_rendered
            return _render_chinese_skills_prompt(index_lines, hidden_note)
        if not skills_by_category and not index_lines:
            return default_rendered
        demoted = _demoted_categories_from_index_lines(index_lines)
        filtered_categories = _filter_qq_dm_skills(skills_by_category)
        filtered_lines = _render_filtered_skill_index_lines(filtered_categories, demoted)
        cleaned_note = _clean_filtered_hidden_note(
            hidden_note, set(filtered_categories), demoted,
        )
        if not filtered_lines:
            return "## 可用技能索引\n当前私聊场景无可加载的日常技能。"
        return _render_chinese_skills_prompt(filtered_lines, cleaned_note)

    def format_inbound_turn_timestamp(
        self,
        clean_content: str,
        event_epoch: float | None,
        envelope: Any,
        tz: Any,
        **kwargs: Any,
    ) -> str | None:
        """Render 04:00 Asia/Shanghai sleep-day & Chinese speaker-gap header."""
        if kwargs.get("internal") or event_epoch is None:
            return None
        try:
            epoch = float(event_epoch)
        except (TypeError, ValueError):
            return None
        if epoch <= 0:
            return None

        target_tz = SHANGHAI_TZ or tz
        now_dt = datetime.fromtimestamp(epoch, tz=target_tz)
        weekday = _WEEKDAY_ZH[now_dt.weekday()]
        clock_str = f"{now_dt.strftime('%Y-%m-%d')} {weekday} {now_dt.strftime('%H:%M')}"
        if now_dt.hour < 4:
            sday = sleep_day_iso(now_dt)
            clock_str += f"（按04:00作息归属{sday[5:]}深夜）"

        history = kwargs.get("history") or ()
        last_user_ts, last_self_ts = _extract_history_timestamps(history, before_epoch=epoch)

        # Also check verified proactive cron deliveries so self-gap reflects real outbound speech!
        cfg = self._config()
        plat = str(getattr(envelope, "platform", "") or "").strip().lower()
        chat_id = str(getattr(envelope, "chat_id", "") or "").strip()
        owner_key = maintainer_anchor(cfg)
        if owner_key:
            cron_sends = query_verified_cron_sends(
                bound_owner=owner_key,
                platform=plat,
                chat_id=chat_id,
                until_ts=epoch,
            )
            if cron_sends:
                latest_cron_ts = cron_sends[-1][0]
                if latest_cron_ts < epoch and (last_self_ts is None or latest_cron_ts > last_self_ts):
                    last_self_ts = latest_cron_ts

        segments = [f"时间：{clock_str}"]
        if last_user_ts is not None and epoch >= last_user_ts:
            user_gap = format_duration_zh(epoch - last_user_ts)
            prev_user_dt = datetime.fromtimestamp(last_user_ts, tz=target_tz)
            day_diff = _sleep_day_gap_days(prev_user_dt, now_dt)
            if day_diff >= 1:
                user_gap += f"（跨{day_diff}个作息日）"
            segments.append(f"距对方上次发言{user_gap}")
        if last_self_ts is not None and epoch >= last_self_ts:
            self_gap = format_duration_zh(epoch - last_self_ts)
            segments.append(f"距我上次发言{self_gap}")

        header = f"[{'；'.join(segments)}]"
        return f"{header}\n{clean_content}" if clean_content else header

    def _is_private_conversational_compaction(self, platform: str, envelope: Any) -> bool:
        surface = classify_turn_surface(envelope, self._config())
        if surface in {"private_maintainer", "private_external"}:
            return True
        plat = str(platform or getattr(envelope, "platform", "") or "").strip().lower()
        chat_type = str(getattr(envelope, "chat_type", "") or "").strip().lower()
        return plat in {"qqbot", "qq", "telegram", "discord"} and chat_type in (PRIVATE_CHAT_TYPES | {"c2c", ""})

    def compaction_config_overrides(
        self,
        session_id: str,
        platform: str,
        envelope: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if not self._is_private_conversational_compaction(platform, envelope):
            return None
        overrides: dict[str, Any] = {"omit_verbatim_user_section": True}
        if "turns_count" in kwargs:
            heading = str(kwargs.get("historical_task_heading") or "## Historical Task Snapshot")
            active_task = str(kwargs.get("active_task") or "None")
            prev_note = str(kwargs.get("previous_summary_note") or "")
            reason_text = str(kwargs.get("reason_text") or "")
            turns_count = int(kwargs.get("turns_count") or 0)
            overrides["fallback_summary_body"] = (
                f"{heading}\n{active_task}\n\n"
                "## What I remember\n"
                f"- 早前 {turns_count} 轮对话已完成本地保底折叠，摘要模型暂时不可用。{prev_note}\n"
                "- 优先顺着保留在尾部的最近几轮对话与当前记忆继续自然承接，不编造未核实的细节。\n\n"
                "## Work I may need to pick up\n"
                f"- 若仍有未办完的事项，先核对最新文件、记忆或工具状态后再继续。{reason_text}"
            )
        return overrides

    def compaction_summary_prompt(
        self,
        default_prompt: str,
        session_id: str,
        platform: str,
        envelope: Any,
        **kwargs: Any,
    ) -> str | None:
        if not self._is_private_conversational_compaction(platform, envelope):
            return None
        content_to_summarize = str(kwargs.get("content_to_summarize") or "")
        previous_summary = str(kwargs.get("previous_summary") or "")
        memory_section = str(kwargs.get("memory_section") or "")
        lang_rule = str(kwargs.get("language_and_provenance_rule") or "")
        hist_heading = str(kwargs.get("historical_task_heading") or "## Historical Task Snapshot")
        pruned_heading = str(kwargs.get("pruned_skills_heading") or "## Pruned Skills")
        focus_topic = str(kwargs.get("focus_topic") or "").strip()

        prev_block = f"\nPREVIOUS SUMMARY:\n{previous_summary}\n" if previous_summary else ""
        focus_block = (
            f'\nFOCUS TOPIC: "{focus_topic}"\n优先保留与 "{focus_topic}" 相关的事实、结论与上下文细节。'
            if focus_topic
            else ""
        )
        return (
            "请以第一人称（「我」）将较早的私聊对话整理为连贯的记忆摘要，供后续对话自然衔接。"
            "只输出结构化摘要正文，不要寒暄或额外前缀。"
            f"{lang_rule}"
            "严禁保留任何 API Key、Token、密码或连接凭据，遇到一律替换为 [REDACTED]。\n"
            f"{prev_block}\n"
            f"TURNS TO SUMMARIZE:\n{content_to_summarize}{memory_section}\n\n"
            "请严格使用以下章节标题整理（用中文自然表述事实与脉络，区分对方亲口说的事、我答应的事与已办完的事）：\n\n"
            f"{hist_heading}\n"
            "（若对方最后一句提问、请求或话头尚未回应完，简明写在这里；若已回应完写 None）\n\n"
            "## What I remember\n"
            "- 刚才聊过的关键事实、对方的近况与处境、我们达成的共识或已经办完的事项（带清楚时间与归属，不把推测写成事实）\n\n"
            "## Work I may need to pick up\n"
            "- 我答应过还没办完的事、待继续跟进的文件/问题或未收尾的话头（没有写 None）\n\n"
            f"{pruned_heading}\n"
            "- 原样保留输入中出现过的 `[SKILL_PRUNED: ...]` 标记（若无写 None）"
            f"{focus_block}"
        )


class ViveMindInboundMergePolicy(_MergeBase):
    """ViveMind's consecutive-message burst merge judgment for private chat."""

    def __init__(self, *, qq_dm_window_seconds: float = 5.0):
        self._qq_dm_window = float(qq_dm_window_seconds)

    @staticmethod
    def _route_ok(platform: str, chat_type: str) -> bool:
        """The host routed this input as a 1:1 scene on a platform we merge on."""
        return (
            str(platform or "").strip().lower() in _DM_MERGE_PLATFORMS
            and str(chat_type or "").strip().lower() in _QQ_DM_TYPES
        )

    @classmethod
    def _eligible(cls, event: Any, platform: str, chat_type: str) -> bool:
        """One judgment for every decision below: an ordinary short message from ONE person in a
        private scene.

        Controls (``/…``, ``T:``), internal wakes, anything declaring a file, and every scene the
        platform did not state as private are excluded. Slack's transport type cannot tell a 1:1
        from an mpim, so Slack must state the private scene itself together with the workspace it
        happened in; a peer bot is not the single person the window belongs to.
        """
        if not cls._route_ok(platform, chat_type):
            return False
        if getattr(event, "internal", False):
            return False
        get_cmd = getattr(event, "get_command", None)
        if callable(get_cmd) and get_cmd():
            return False
        text = cls._event_text(event).strip()
        if not text or text.startswith("/") or text.startswith("T:"):
            return False
        msg_type = getattr(event, "message_type", None)
        if str(getattr(msg_type, "value", msg_type) or "text").strip().lower() not in {"text", ""}:
            return False
        # A file rides along even when the message type still reads TEXT.
        if getattr(event, "media_urls", None) or getattr(event, "media_types", None):
            return False
        source = getattr(event, "source", None)
        if source is None:
            return True  # a hand-built event with no adapter envelope: the route is all there is
        if str(getattr(source, "chat_type", "") or "").strip().lower() not in _QQ_DM_TYPES:
            return False
        if not (str(getattr(source, "chat_id", "") or "").strip()
                and (getattr(source, "user_id", None) or getattr(source, "user_id_alt", None))):
            return False
        if getattr(source, "is_bot", False):
            return False
        scene = str(getattr(source, "scene_chat_type", "") or "").strip().lower()
        if scene and scene not in _QQ_DM_TYPES:
            return False
        if str(platform or "").strip().lower() == "slack" and not (
            scene in _QQ_DM_TYPES and str(getattr(source, "scope_id", "") or "").strip()
        ):
            return False
        return True

    @staticmethod
    def _event_text(event: Any) -> str:
        text = getattr(event, "text", "")
        return text if isinstance(text, str) else ""

    @staticmethod
    def _reply_signature(event: Any) -> tuple[str, str] | None:
        # Task 01 carries the platform's explicit quote target on the source; older shapes put it
        # on the event itself.
        source = getattr(event, "source", None)
        reply_id = str(getattr(event, "reply_to_message_id", None)
                       or getattr(source, "reply_to_message_id", None) or "").strip()
        reply_txt = str(getattr(event, "reply_to_text", "") or "").strip()
        if not reply_id and not reply_txt:
            return None
        return (reply_id, reply_txt)

    def debounce_window_seconds(self, platform: str, chat_type: str, *, busy: bool) -> float | None:
        if self._route_ok(platform, chat_type):
            return self._qq_dm_window
        return None

    def should_debounce_event(self, event: Any, platform: str, chat_type: str, *, busy: bool) -> bool | None:
        if not self._route_ok(platform, chat_type):
            return None
        return self._eligible(event, platform, chat_type)

    def can_merge_events(
        self,
        existing: Any,
        incoming: Any,
        platform: str,
        chat_type: str,
        *,
        busy: bool,
    ) -> bool | None:
        """Judge whether two consecutive inputs of one private chat belong to the same turn.

        - Slash commands, internal wakes, ``T:`` test turns, files and non-text never merge.
        - Plain + plain, plain + quote, and quote + plain merge into one turn.
        - Quote + quote merges ONLY when both quote the exact same target message; two different
          quote targets stay two distinct turns.
        """
        if not self._route_ok(platform, chat_type):
            return None
        if not (self._eligible(existing, platform, chat_type)
                and self._eligible(incoming, platform, chat_type)):
            return False
        sig_a = self._reply_signature(existing)
        sig_b = self._reply_signature(incoming)
        if sig_a is not None and sig_b is not None:
            id_a, _ = sig_a
            id_b, _ = sig_b
            if id_a and id_b:
                return id_a == id_b
            return sig_a == sig_b
        return True

    def merge_texts(self, texts: Sequence[str], platform: str, chat_type: str) -> str | None:
        if not self._route_ok(platform, chat_type):
            return None
        cleaned = [str(t).strip() for t in texts if isinstance(t, str) and str(t).strip()]
        return "\n".join(cleaned)


def prepare_proactive_cron_prompt(
    job: Mapping[str, Any],
    prompt: str,
    envelope: Any = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> str | None:
    """Enrich ViveMind-governed proactive cron runs with non-intrusive silence permission."""
    if not isinstance(job, Mapping) or not isinstance(prompt, str) or not prompt.strip():
        return None
    if _owned_free_think_job(dict(job)):
        if (getattr(envelope, "verified", False)
                and not getattr(envelope, "cancelled", True)
                and getattr(envelope, "job_id", "") == job.get("id")
                and free_think_can_deliver(
                    {**job, "execution_id": envelope.execution_id},
                    _load_plugin_config() if config is None else config,
                )):
            return _FREE_THINK_PROMPT
        return None
    meta = job.get("job_metadata") if isinstance(job.get("job_metadata"), Mapping) else {}
    activity_class = str(
        job.get("activity_class")
        or meta.get("activity_class")
        or getattr(envelope, "activity_class", "")
        or ""
    ).strip().lower()
    is_proactive = bool(
        activity_class in {"proactive", "care", "greeting", "share", "vivemind_proactive"}
        or meta.get("vivemind_proactive")
    )
    if not is_proactive:
        return None
    silence_hint = "（若结合当下时间与最近互动觉得没必要打扰，直接只回复 [SILENT]，不必勉强开口。）"
    if "[SILENT]" in prompt:
        return None
    return f"{prompt.rstrip()}\n\n{silence_hint}"


def gate_proactive_cron_delivery(
    job: Mapping[str, Any],
    response_text: str,
    envelope: Any = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | bool | None:
    """Suppress silence and enforce the private destination of free-think runs."""
    if is_test_isolated_turn(getattr(envelope, "session_id", None)):
        return {"allow": False, "text": response_text, "reason": "test_isolated"}
    stripped = str(response_text or "").strip()
    if not stripped or stripped.upper() in _SILENT_TOKENS:
        return {"allow": False, "text": "", "reason": "silent_token"}
    if isinstance(job, Mapping) and _owned_free_think_job(dict(job)):
        if not free_think_can_deliver(job, _load_plugin_config() if config is None else config):
            return {"allow": False, "text": response_text, "reason": "vivemind_free_think_unverified"}
    return None
