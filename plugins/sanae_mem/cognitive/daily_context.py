from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

from .daily_inputs import (
    MEME_CONTEXT_MARKER, SELF_UTTERANCE_MARKER, _assistant_quote_safe,
    _history_pools, _image_parts, _is_second_person_clause,
    _is_system_injection, _known_item_lines,
    _looks_secret, _model_safe_text, _normalize_input,
    _preceding_user_pool, _recent_dialogue, _recent_user_pool,
    _row_has, _semantic_eligible, _split_known_meme_notes, _turn_context_block,
    flatten_user_text,
)
from .daily_rules import (
    DAILY_SEMANTIC_MAX_TOKENS, DAILY_SEMANTIC_SEEN_MAX,
    DAILY_SEMANTIC_TIMEOUT, KIND_CONSTRAINT, KIND_FACT, KIND_THREAD,
    MAX_ACTIVE_PER_KIND, MAX_TEXT_CHARS,
    SOURCE_STATEMENT,
    _CHAT_META, _Decision, _EVENT_SLOT_PREFIX,
    _IMAGE_ONLY_TEXT, _RECORD_SUPPRESSION, _life_fragment_slot,
    _norm_text, _now_iso,
)
from ..infra.clock import sleep_day_iso
from ..governance.identity_config import daily_memory_identity as daily_memory_identity
from .daily_semantic_validation import SemanticItem
from .daily_state import DailyState, _row_source, _thread_exploration_completed
from ..storage.sqlite_facade import SqliteFacade
from ..governance.source_context import (
    ChatIdentity,
    ChatScope,
    current_chat_scope,
    current_source_record,
    daily_profile_key as daily_profile_key,
    in_public_interaction,
    resolve_chat_identity,
)

from . import daily_semantics

_ASCII_SLOT_RE = re.compile(r"^[A-Za-z0-9_:-]{1,32}$")


def _event_slot(key: str) -> str:
    normalized = re.sub(r"\s+", "", str(key or ""))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return _EVENT_SLOT_PREFIX + digest


def _semantic_slot(item: SemanticItem, text: str) -> str:
    event_key = re.sub(r"\s+", "", item.event_key)
    if item.kind == KIND_CONSTRAINT:
        if event_key and _ASCII_SLOT_RE.fullmatch(event_key):
            return event_key
        seed = event_key or _norm_text(text)
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return "constraint:" + digest
    if item.kind == KIND_THREAD:
        seed = event_key or _norm_text(text)
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return "thread:" + digest
    if event_key:
        return _event_slot(event_key)
    return _life_fragment_slot(text)


def identity_matches(cached: tuple[str, ...], sender_id: str) -> bool:
    sender = str(sender_id or "").strip()
    return bool(sender) and sender in cached


@dataclass(frozen=True)
class DailyBinding:
    session_id: str
    agent_context: str
    platform: str
    owner_key: str
    profile_key: str
    owner_platform: str
    owner_accounts: tuple[str, ...]

    @property
    def scope(self) -> tuple[str, str] | None:
        if not self.owner_key or not self.profile_key:
            return None
        return self.owner_key, self.profile_key


class DailyModel(Protocol):
    def complete(
        self, messages: list[dict[str, str]], *, task: str, temperature: float,
        max_tokens: int, timeout: float, purpose: str,
    ) -> object: ...


class DailyContext:
    def __init__(
        self, sqlite: SqliteFacade, daily_state: DailyState, *,
        config: Mapping[str, object], current_binding: Callable[[], DailyBinding],
        current_model: Callable[[], DailyModel | None], persona_name: str,
    ) -> None:
        self._sqlite = sqlite
        self._daily_state = daily_state
        self._config = config
        self._current_binding = current_binding
        self._current_model = current_model
        self._persona_name = persona_name.strip()
        self._daily_semantic_seen: dict[tuple[str, str], None] = {}
        self._daily_semantic_lock = threading.Lock()

    def reconfigure(self, *, config: Mapping[str, object], persona_name: str) -> None:
        self._config = config
        self._persona_name = persona_name.strip()
        with self._daily_semantic_lock:
            self._daily_semantic_seen.clear()

    def check_pre_llm_identity(self, *, session_id: str, platform: str, sender_id: str) -> None:
        # Diagnose identity failures before generation. Daily writes use verified delivery.
        self._daily_turn_scope(session_id, platform, sender_id, require_current=True)

    def _resolve_daily_identity(self) -> tuple[ChatScope, ChatIdentity] | None:
        scope = current_chat_scope()
        if not isinstance(scope, ChatScope):
            raise TypeError(
                f"current_chat_scope returned {type(scope).__name__}, expected ChatScope"
            )
        if scope.visibility == "unknown":
            return None
        if self._sqlite.is_open:
            identity = self._sqlite.read(
                lambda conn: resolve_chat_identity(scope, self._config, connection=conn)
            )
        else:
            identity = resolve_chat_identity(scope, self._config, connection=None)
        if not isinstance(identity, ChatIdentity):
            raise TypeError(
                f"resolve_chat_identity returned invalid identity {type(identity).__name__}"
            )
        return scope, identity

    def _daily_turn_scope(
        self, session_id: str, platform: str, sender_id: str, *, require_current: bool = False,
    ) -> tuple[str, str] | None:
        if not session_id or in_public_interaction():
            return None
        binding = self._current_binding()
        if binding.agent_context != "primary":
            return None
        if require_current and session_id != binding.session_id:
            return None
        daily_scope = binding.scope
        if daily_scope is None:
            return None
        try:
            resolved = self._resolve_daily_identity()
        except Exception as exc:
            logger.warning(
                "sanae_mem: daily identity resolution failed (%s)", type(exc).__name__
            )
            return None
        if resolved is not None:
            scope, identity = resolved
            if scope.visibility != "private" or not identity.maintainer:
                return None
            if identity.private_owner_key != daily_scope[0]:
                return None
            turn_platform = str(platform or scope.platform or "").strip().lower()
            if scope.platform and turn_platform and turn_platform != scope.platform:
                return None
            if binding.platform and scope.platform and scope.platform != binding.platform:
                return None
            if sender_id and not identity_matches((scope.actor, scope.actor_alt), sender_id):
                return None
            return daily_scope
        if platform != binding.platform or platform != binding.owner_platform:
            return None
        if platform == "slack":
            return None
        if not identity_matches(binding.owner_accounts, sender_id):
            return None
        return daily_scope

    def record_completed_turn(
        self,
        session_id: str,
        assistant_content: str,
        history: object,
        *,
        sender_id: str,
        platform: str,
        user_message: object,
        turn_id: str,
        plugin_user_context: str = "",
    ) -> bool:
        sender_id = str(sender_id or "")
        platform = str(platform or "")
        scope = self._daily_turn_scope(session_id, platform, sender_id)
        if scope is None or not self._daily_source_allows(sender_id):
            return False
        user_text, meme_notes = _split_known_meme_notes(flatten_user_text(user_message))
        if "[OUT-OF-BAND]【B站收件箱内部唤醒】" in user_text:
            return False
        today = sleep_day_iso()
        context = str(plugin_user_context or "")
        elsewhere = _turn_context_block(context, SELF_UTTERANCE_MARKER)
        meme_block = _turn_context_block(context, MEME_CONTEXT_MARKER)
        meme = "\n".join(part for part in (meme_notes, meme_block) if part)
        decisions = self._daily_semantic_decisions(
            user_text, history, session_id, today, scope=scope, assistant=assistant_content,
            elsewhere=elsewhere, meme=meme, turn_key=turn_id,
        )
        if _image_parts(user_message) or meme_notes:
            decisions.insert(0, _Decision(
                KIND_FACT, "media:image:" + today, "set", _IMAGE_ONLY_TEXT, "", None,
            ))
        with self._sqlite.transaction():
            applied = bool(self._daily_state.apply(scope, decisions, today, session_id))
        return applied

    def _daily_semantic_decisions(
        self, user_text: str, conversation_history: object, session_id: str, today: str,
        *, scope: tuple[str, str], assistant: str, elsewhere: str = "", meme: str = "",
        turn_key: str = "",
    ) -> list[_Decision]:
        model = self._current_model()
        if model is None:
            return []
        model_text = _model_safe_text(_normalize_input(user_text))
        if not _semantic_eligible(model_text):
            return []
        assistant = assistant.strip()
        assistant_safe = assistant if _assistant_quote_safe(assistant) else ""
        if turn_key:
            key = (session_id, str(turn_key))
            with self._daily_semantic_lock:
                seen = self._daily_semantic_seen
                if key in seen:
                    return []
                seen[key] = None
                while len(seen) > DAILY_SEMANTIC_SEEN_MAX:
                    seen.pop(next(iter(seen)))
        owner, profile = scope
        try:
            rows = self._sqlite.read(
                lambda _conn: self._daily_state.list_active_daily_entries(
                    owner_key=owner, profile_key=profile, today=today,
                )
            )
        except (sqlite3.Error, RuntimeError) as exc:
            logger.warning("sanae_mem: daily semantics state read failed (%s)", type(exc).__name__)
            return []
        known = [
            row for row in rows
            if str(row["kind"]) != KIND_FACT or _row_source(row) == SOURCE_STATEMENT
        ]
        if isinstance(conversation_history, (list, tuple)):
            for index in range(len(conversation_history) - 1, -1, -1):
                entry = conversation_history[index]
                if isinstance(entry, Mapping) and entry.get("role") == "user":
                    conversation_history = conversation_history[:index + 1]
                    break
        persona_name = self._persona_name
        recent = (
            _recent_dialogue(conversation_history, persona_name=persona_name)
            if persona_name else _recent_dialogue(conversation_history)
        )
        history_user, history_assistant = _history_pools(conversation_history, model_text)
        known_lines, known_map, known_text, known_slots = _known_item_lines(known)
        known_dates = {}
        for row in known:
            slot = str(row["slot"] or "")
            if slot not in known_slots:
                continue
            opened = str(row["opened_on"] or "").strip() if _row_has(row, "opened_on") else ""
            if opened:
                known_dates[slot] = opened
        try:
            raw = daily_semantics.call_llm(
                model, message=model_text, recent=recent, known_items=known_lines,
                timeout=DAILY_SEMANTIC_TIMEOUT, max_tokens=DAILY_SEMANTIC_MAX_TOKENS,
                assistant=assistant_safe, elsewhere=elsewhere, meme=meme,
                persona_name=persona_name,
            )
            items = daily_semantics.parse_items(raw) or []
            sources = daily_semantics.Sources(
                user="\n".join(part for part in (model_text, history_user) if part),
                assistant=history_assistant,
                known=known_text,
                message=model_text,
                known_map=known_map,
                known_slots=known_slots,
                recent_user=_recent_user_pool(conversation_history, model_text),
                preceding_user=_preceding_user_pool(conversation_history, model_text),
                known_dates=known_dates,
            )
            validated = daily_semantics.validate_items(
                items, message=model_text, sources=sources, known_map=known_map,
                today=today,
            )
        except Exception as exc:
            logger.warning("sanae_mem: daily semantics skipped (%s)", type(exc).__name__)
            return []
        return self._semantic_items_to_decisions(validated, known_map)

    @staticmethod
    def _semantic_items_to_decisions(
        items: Sequence[SemanticItem], known_map: dict[str, str],
    ) -> list[_Decision]:
        out: list[_Decision] = []
        for item in items:
            text = str(item.text or "").strip()
            expires = item.expires_on
            if item.action == "close":
                kind = known_map.get(item.target_slot, item.kind)
                out.append(_Decision(kind, item.target_slot, "clear", "", item.quote))
                continue
            if item.kind == KIND_FACT and (
                _CHAT_META.fullmatch(text) or _is_second_person_clause(text)
            ):
                continue
            if not text or len(text) > MAX_TEXT_CHARS:
                continue
            if _looks_secret(text) or _is_system_injection(text):
                continue
            if item.action == "update":
                kind = known_map.get(item.target_slot, item.kind)
                out.append(_Decision(kind, item.target_slot, "set", text, item.quote, expires))
                continue
            slot = _semantic_slot(item, text)
            out.append(_Decision(item.kind, slot, "set", text, item.quote, expires))
        return out

    def _daily_source_allows(self, sender_id: str) -> bool:
        try:
            scope = current_chat_scope()
            if not isinstance(scope, ChatScope) or scope.visibility != "private":
                visibility = (
                    scope.visibility
                    if isinstance(scope, ChatScope)
                    and scope.visibility in {"unknown", "public", "local"}
                    else "invalid"
                )
                logger.warning("sanae_mem: daily source rejected (scope_visibility=%s)", visibility)
                return False
            identity = self._sqlite.read_optional(
                lambda conn: resolve_chat_identity(scope, self._config, connection=conn)
            )
            if not isinstance(identity, ChatIdentity) or not identity.maintainer:
                logger.warning("sanae_mem: daily source rejected (non_maintainer_identity)")
                return False
            daily_owner_key = self._current_binding().owner_key
            if not daily_owner_key or identity.private_owner_key != daily_owner_key:
                logger.warning("sanae_mem: daily source rejected (owner_key_mismatch)")
                return False
            source = current_source_record(maintainer=identity.maintainer, scope=scope)
            if not isinstance(source, dict) or str(source.get("kind") or "") != "maintainer":
                if source is None:
                    source_kind = "missing"
                elif isinstance(source, dict):
                    raw_kind = str(source.get("kind") or "")
                    source_kind = (
                        raw_kind
                        if raw_kind in {"external_user", "group_member", "self", "video", "unknown"}
                        else "invalid"
                    )
                else:
                    source_kind = "invalid"
                logger.warning("sanae_mem: daily source rejected (source_kind=%s)", source_kind)
                return False
            if sender_id and not identity_matches((scope.actor, scope.actor_alt), sender_id):
                logger.warning("sanae_mem: daily source rejected (sender_mismatch)")
                return False
            author = str(source.get("author_id") or "").strip()
            if not author or not identity_matches((scope.actor, scope.actor_alt), author):
                logger.warning("sanae_mem: daily source rejected (author_mismatch)")
                return False
            scope_platform = str(scope.platform or "").strip().lower()
            source_platform = str(source.get("platform") or "").strip().lower()
            if not scope_platform or not source_platform or source_platform != scope_platform:
                logger.warning("sanae_mem: daily source rejected (platform_mismatch)")
                return False
        except Exception as exc:
            logger.warning("sanae_mem: daily source check failed (%s)", type(exc).__name__)
            return False
        return True

    def render_maintainer_daily_facts(self) -> str:
        scope = self._current_binding().scope
        if scope is None or not self._sqlite.is_open:
            return ""
        try:
            return self._daily_state.render(scope, sleep_day_iso())
        except (sqlite3.Error, RuntimeError) as exc:
            logger.warning("sanae_mem: daily_context render failed (%s)", type(exc).__name__)
            return ""

    def active_curiosity_threads(self, *, today: str | None = None) -> list[str]:
        """她自己留着待空闲时查一查的好奇心小尾巴（不进每轮聊天尾注）。"""
        scope = self._current_binding().scope
        if scope is None:
            return []
        day = today or sleep_day_iso()
        rows = self._daily_state.list_active_daily_entries(
            owner_key=scope[0], profile_key=scope[1], today=day,
        )
        return [
            str(row["text"] or "").strip() for row in rows
            if str(row["kind"]) == KIND_THREAD and str(row["text"] or "").strip()
            and (not row["expires_on"] or str(row["expires_on"]) >= day)
        ]

    def resolve_explored_curiosity_threads(
        self, activity_text: str, *, today: str | None = None, propagate_read_error: bool = False,
    ) -> int:
        """Close curiosity threads her own free-time activity already explored."""
        if not str(activity_text or "").strip() or not self._sqlite.is_open:
            return 0
        scope = self._current_binding().scope
        if scope is None:
            return 0
        day = today or sleep_day_iso()
        try:
            rows = self._daily_state.list_active_daily_entries(
                owner_key=scope[0], profile_key=scope[1], today=day,
            )
        except (sqlite3.Error, RuntimeError) as exc:
            if propagate_read_error:
                raise
            logger.warning("sanae_mem: curiosity state read failed (%s)", type(exc).__name__)
            return 0
        closed = 0
        for row in rows:
            if str(row["kind"]) != KIND_THREAD:
                continue
            thread_text = f"{row['text'] or ''} {row['source_quote'] or ''}"
            if not _thread_exploration_completed(thread_text, str(activity_text)):
                continue
            if self._daily_state.cas_supersede_daily_entry(
                owner_key=scope[0], profile_key=scope[1], entry_id=int(row["entry_id"]),
                version=int(row["version"]), status="closed", now=_now_iso(),
            ):
                closed += 1
        return closed

    def record_video_curiosity(
        self, title: str, content: str, *, session_id: str = "", today: str | None = None,
    ) -> list[_Decision]:
        """Extract at most one OPEN_THREAD curiosity when she finishes watching a video."""
        model = self._current_model()
        if model is None or not self._sqlite.is_open:
            return []
        binding = self._current_binding()
        active_session = session_id or binding.session_id
        scope = binding.scope
        if scope is None:
            return []
        day = today or sleep_day_iso()
        material = _model_safe_text(_normalize_input(f"《{title}》：{content}"))
        if not material or _RECORD_SUPPRESSION.search(material):
            return []
        owner, profile = scope
        try:
            def _load_thread_snapshot(_conn: sqlite3.Connection) -> tuple[bool, list[sqlite3.Row]]:
                stats = self._daily_state.daily_active_stats(owner_key=owner, profile_key=profile)
                if stats["counts"].get(KIND_THREAD, 0) >= MAX_ACTIVE_PER_KIND.get(KIND_THREAD, 0):
                    return True, []
                return False, self._daily_state.list_active_daily_entries(
                    owner_key=owner, profile_key=profile, today=day,
                )

            at_limit, rows = self._sqlite.read(_load_thread_snapshot)
            if at_limit:
                return []
        except (sqlite3.Error, RuntimeError) as exc:
            logger.warning("sanae_mem: video curiosity state read failed (%s)", type(exc).__name__)
            return []
        known = [row for row in rows if str(row["kind"]) == KIND_THREAD]
        known_lines, known_map, known_text, known_slots = _known_item_lines(known)
        try:
            raw = daily_semantics.call_video_curiosity_llm(
                model, material=material, known_items=known_lines,
                timeout=DAILY_SEMANTIC_TIMEOUT, max_tokens=240,
            )
            items = [
                item for item in (daily_semantics.parse_items(raw) or [])
                if str(item.get("kind") or "").upper() == KIND_THREAD
                and str(item.get("action") or "set").lower() == "set"
            ]
            sources = daily_semantics.Sources(
                user=material, message=material, known=known_text,
                known_map=known_map, known_slots=known_slots,
            )
            validated = daily_semantics.validate_items(
                items, message=material, sources=sources, known_map=known_map,
                max_items=1, today=day,
            )
        except Exception as exc:
            logger.warning("sanae_mem: video curiosity skipped (%s)", type(exc).__name__)
            return []
        book_title = f"《{str(title or '').strip('《》 ')}》" if str(title or "").strip("《》 ") else ""
        decisions = []
        for d in self._semantic_items_to_decisions(validated, known_map):
            if d.kind == KIND_THREAD and d.action == "set":
                quote = d.quote or ""
                if book_title and book_title not in quote and book_title not in (d.text or ""):
                    quote = f"{book_title} {quote}".strip()
                decisions.append(_Decision(d.kind, d.slot, d.action, d.text, quote, d.expires_on))
                break
        if decisions:
            with self._sqlite.transaction():
                self._daily_state.apply(scope, decisions, day, active_session)
        return decisions
