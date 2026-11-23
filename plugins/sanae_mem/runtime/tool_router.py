from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, cast

from ..cognitive.impression import IMPRESSION_SCHEMA
from ..cognitive.inner_wiring import INNER_STATE_SCHEMA
from ..domains.meme_schemas import (
    MEME_DELETE_SCHEMA, MEME_SAVE_SCHEMA, MEME_SEARCH_SCHEMA, MEME_UPDATE_SCHEMA,
)
from ..governance.group_scope import GROUP_TOOL_NAMES
from ..governance.source_context import resolve_chat_identity
from ..infra.chat_scope import current_chat_scope
from ..infra.host_bridge import current_cron_binding, is_test_isolated_turn
from ..storage.sqlite_facade import SqliteFacade
from .scheduled import current_free_think
from .schemas import (
    ADD_SCHEMA, COMPACT_SCHEMA, DISTILL_SCHEMA, FALSE_TRIGGER_SCHEMA,
    FORGET_SCHEMA, MARKED_PROMOTED_SCHEMA, SEARCH_SCHEMA, _PRIVATE_REVIEW_SCHEMA,
)


class MemeTools(Protocol):
    def save(self, args: dict[str, object]) -> dict[str, object]: ...
    def search(self, args: dict[str, object]) -> list[dict[str, object]]: ...
    def update(self, args: dict[str, object]) -> dict[str, object]: ...
    def delete(self, args: dict[str, object]) -> dict[str, object]: ...
    def available_sticker_summary(self) -> str: ...


class ResidentTools(Protocol):
    def can_review_private_user(self) -> bool: ...
    def save_grounded_user(
        self, *, text: str, evidence_ids: list[int], replaces: str,
    ) -> dict[str, object]: ...
    def promote_reviewed_event(self, event_id: int) -> dict[str, object]: ...
    def demote_resident_entry(self, target: str, entry_text: str) -> dict[str, object]: ...


class SearchTool(Protocol):
    def __call__(
        self, args: dict[str, object], *, session_id: str, free_think_owner: str,
    ) -> list[dict[str, object]]: ...


class TurnTool(Protocol):
    def __call__(
        self, args: dict[str, object], *, session_id: str, turn_id: str, synthetic: bool,
    ) -> Mapping[str, object]: ...


class ToolRouter:
    def __init__(
        self,
        *,
        sqlite: SqliteFacade,
        memes: MemeTools,
        resident: ResidentTools,
        current_session_id: Callable[[], str],
        authorized_free_think_owner: Callable[[str], str],
        private_chat_owner: Callable[[], str],
        has_group_grant: Callable[[], bool],
        add: Callable[[dict[str, object]], dict[str, object]],
        search: SearchTool,
        forget: Callable[[dict[str, object]], dict[str, object]],
        compact: Callable[[], dict[str, object]],
        marked_promoted: Callable[[list[int], str], dict[str, object]],
        distill: Callable[[], dict[str, object]],
        review_migration_text: Callable[[str, str], dict[str, object]],
        review_outcome: Callable[[int], dict[str, object]],
        record_false_trigger: Callable[[], int],
        shadow_enabled: Callable[[], bool],
        runtime_config: Callable[[], dict[str, Any]],
        plugin_config: dict[str, Any],
        inner_state: TurnTool,
        impression: TurnTool,
    ) -> None:
        self._sqlite = sqlite
        self._memes = memes
        self._resident = resident
        self._current_session_id = current_session_id
        self._authorized_free_think_owner = authorized_free_think_owner
        self._private_chat_owner = private_chat_owner
        self._has_group_grant = has_group_grant
        self._add = add
        self._search = search
        self._forget = forget
        self._compact = compact
        self._marked_promoted = marked_promoted
        self._distill = distill
        self._review_migration_text = review_migration_text
        self._review_outcome = review_outcome
        self._record_false_trigger = record_false_trigger
        self._shadow_enabled = shadow_enabled
        self._runtime_config = runtime_config
        self._plugin_config = plugin_config
        self._inner_state = inner_state
        self._impression = impression

    def tool_schemas(self) -> list[dict[str, Any]]:
        session_id = self._current_session_id()
        if self._authorized_free_think_owner(""):
            return [SEARCH_SCHEMA, MEME_SEARCH_SCHEMA]
        if not session_id and not self._shadow_enabled():
            # Hermes registers tool names before it initializes the provider.
            binding = current_cron_binding()
            if binding and current_free_think(self._runtime_config(), binding[0]):
                return [SEARCH_SCHEMA, MEME_SEARCH_SCHEMA]
        scope = current_chat_scope()
        visibility = scope.visibility
        if visibility == "public":
            if self._has_group_grant():
                return [{
                    **SEARCH_SCHEMA,
                    "parameters": {
                        **SEARCH_SCHEMA["parameters"],
                        "properties": {
                            key: value
                            for key, value in SEARCH_SCHEMA["parameters"]["properties"].items()
                            if key != "scheduled"
                        },
                    },
                }]
            return [{**SEARCH_SCHEMA, "description": "搜索经审核可公开的记忆摘录。"}]
        if visibility == "unknown" or (
            visibility == "private" and session_id and not self._private_chat_owner()
        ):
            return []
        meme_search_schema = dict(MEME_SEARCH_SCHEMA)
        try:
            summary = self._memes.available_sticker_summary()
        except Exception:
            summary = ""
        if summary:
            meme_search_schema["description"] = f"{MEME_SEARCH_SCHEMA['description']} {summary}"
        identity = self._sqlite.read_optional(
            lambda conn: resolve_chat_identity(scope, self._plugin_config, connection=conn)
        )
        review_available = bool(identity.maintainer and identity.private_owner_key)
        return [
            ADD_SCHEMA,
            SEARCH_SCHEMA,
            FORGET_SCHEMA,
            COMPACT_SCHEMA,
            MARKED_PROMOTED_SCHEMA,
            DISTILL_SCHEMA,
            FALSE_TRIGGER_SCHEMA,
            INNER_STATE_SCHEMA,
            IMPRESSION_SCHEMA,
            MEME_SAVE_SCHEMA,
            meme_search_schema,
            MEME_UPDATE_SCHEMA,
            MEME_DELETE_SCHEMA,
            *([_PRIVATE_REVIEW_SCHEMA] if review_available else []),
        ]

    def dispatch(
        self, tool_name: str, args: dict[str, object], *,
        session_id: str, turn_id: str, synthetic: bool,
    ) -> Mapping[str, object] | list[dict[str, object]]:
        scope = current_chat_scope()
        effective_session = str(session_id or self._current_session_id())
        free_think_owner = (
            self._authorized_free_think_owner(effective_session)
            if effective_session == self._current_session_id() else ""
        )
        if not free_think_owner and (scope.visibility == "unknown" or (
            scope.visibility == "private" and not self._private_chat_owner()
        )):
            raise PermissionError("tool requires a verified conversation scope")
        if free_think_owner and tool_name not in {"sanae_mem_search", "sanae_meme_search"}:
            raise PermissionError("private free think can only search memories")
        has_group_grant = self._has_group_grant()
        if has_group_grant and tool_name not in GROUP_TOOL_NAMES:
            raise PermissionError("this group chat can only search group memories")
        if not has_group_grant and scope.visibility == "public" and tool_name != "sanae_mem_search":
            raise PermissionError("only approved public memories can be searched in this chat")
        if not self._sqlite.is_open:
            raise RuntimeError("provider is not initialized")
        if is_test_isolated_turn(effective_session) and tool_name in {
            "sanae_mem_add", "sanae_mem_forget", "sanae_mem_compact",
            "sanae_mem_marked_promoted", "sanae_mem_distill", "sanae_mem_review",
            "sanae_mem_report_false_trigger",
        }:
            return {"stored": False, "reason": "test_session_isolated"}
        if tool_name == "sanae_mem_add":
            return self._add(args)
        if tool_name == "sanae_mem_search":
            return self._search(args, session_id=effective_session, free_think_owner=free_think_owner)
        if tool_name == "sanae_mem_forget":
            return self._forget(args)
        if tool_name == "sanae_mem_compact":
            return self._tool_compact(args)
        if tool_name == "sanae_mem_marked_promoted":
            return self._marked_promoted(cast(list[int], args.get("ids") or []), str(args.get("to") or ""))
        if tool_name == "sanae_mem_distill":
            return self._distill()
        if tool_name == "sanae_mem_review":
            return self._tool_review(args)
        if tool_name == "sanae_mem_report_false_trigger":
            return {"weekly_count": self._record_false_trigger(), "shadow": self._shadow_enabled()}
        if tool_name == "sanae_inner_state":
            return self._inner_state(args, session_id=session_id, turn_id=turn_id, synthetic=synthetic)
        if tool_name == "sanae_impression":
            return self._impression(args, session_id=session_id, turn_id=turn_id, synthetic=synthetic)
        if tool_name == "sanae_meme_save":
            return self._memes.save(args)
        if tool_name == "sanae_meme_search":
            return self._memes.search(args)
        if tool_name == "sanae_meme_update":
            return self._memes.update(args)
        if tool_name == "sanae_meme_delete":
            return self._memes.delete(args)
        raise ValueError(f"unknown tool: {tool_name}")

    def _tool_compact(self, args: dict[str, object]) -> dict[str, object]:
        target = str(args.get("demote_target") or "").strip()
        if not target:
            return self._compact()
        entry = str(args.get("demote_entry") or "").strip()
        if not entry:
            return {"ok": False, "demoted": False, "reason": "demote_entry_required"}
        return self._resident.demote_resident_entry(target, entry)

    def _tool_review(self, args: dict[str, object]) -> dict[str, object]:
        mode = str(args.get("mode") or "").strip()
        if mode in {"grounded_user", "stage"} and not self._resident.can_review_private_user():
            return {"ok": False, "reason": "verified_maintainer_private_chat_required"}
        if mode == "grounded_user":
            return self._resident.save_grounded_user(
                text=str(args.get("text") or ""),
                evidence_ids=cast(list[int], args.get("evidence_ids")),
                replaces=str(args.get("replaces") or ""),
            )
        if mode == "migration":
            text = str(args.get("text") or "")
            target = str(args.get("target") or "")
            if not text.strip() or target not in ("memory", "user"):
                return {"ok": False, "reason": "migration_review_needs_text_and_target"}
            return self._review_migration_text(text, target)
        raw_id = args.get("event_id")
        if isinstance(raw_id, bool) or raw_id is None:
            return {"ok": False, "reason": "event_id_required"}
        try:
            event_id = int(cast(int | str, raw_id))
        except (TypeError, ValueError):
            return {"ok": False, "reason": "event_id_required"}
        if mode == "stage":
            return self._resident.promote_reviewed_event(event_id)
        return self._review_outcome(event_id)
