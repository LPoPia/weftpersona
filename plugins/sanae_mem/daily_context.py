from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
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
        BLOCK_MARKER, DAILY_SEMANTIC_MAX_TOKENS, DAILY_SEMANTIC_SEEN_MAX,
        DAILY_SEMANTIC_TIMEOUT, KIND_CONSTRAINT, KIND_FACT, KIND_THREAD,
        MAX_ACTIVE_PER_KIND, MAX_BLOCK_CHARS, MAX_BLOCK_TEXT_CHARS,
        MAX_QUOTE_CHARS, MAX_TERMINAL_ROWS, MAX_TEXT_CHARS,
        SOURCE_INFERENCE, SOURCE_STATEMENT, STALE_TTL_DAYS, STALE_TTL_MAX_DAYS,
        _BLOCK_LABELS, _BLOCK_NOTE, _CHAT_META, _Decision, _EVENT_SLOT_PREFIX,
        _IMAGE_ONLY_TEXT, _IMAGE_PART_TYPES, _RECORD_SUPPRESSION, _TZ, _hour, _life_fragment_slot,
        _norm_text, _now, _now_iso, _today,
    )
    from .daily_state import (
        DailyStateMixin, _default_ttl, _is_transient_state,
        _resolve_daily_expiry, _row_source, _shares_content,
        _sustained_horizon, _thread_exploration_completed,
        _yesterday, _yesterday_items,
    )
    from .source_context import in_public_interaction
except ImportError:
    from daily_inputs import (
        MEME_CONTEXT_MARKER, SELF_UTTERANCE_MARKER, _assistant_quote_safe,
        _history_pools, _image_parts, _is_second_person_clause,
        _is_system_injection, _known_item_lines,
        _looks_secret, _model_safe_text, _normalize_input,
        _preceding_user_pool, _recent_dialogue, _recent_user_pool,
        _row_has, _semantic_eligible, _split_known_meme_notes, _turn_context_block,
        flatten_user_text,
    )
    from daily_rules import (
        BLOCK_MARKER, DAILY_SEMANTIC_MAX_TOKENS, DAILY_SEMANTIC_SEEN_MAX,
        DAILY_SEMANTIC_TIMEOUT, KIND_CONSTRAINT, KIND_FACT, KIND_THREAD,
        MAX_ACTIVE_PER_KIND, MAX_BLOCK_CHARS, MAX_BLOCK_TEXT_CHARS,
        MAX_QUOTE_CHARS, MAX_TERMINAL_ROWS, MAX_TEXT_CHARS,
        SOURCE_INFERENCE, SOURCE_STATEMENT, STALE_TTL_DAYS, STALE_TTL_MAX_DAYS,
        _BLOCK_LABELS, _BLOCK_NOTE, _CHAT_META, _Decision, _EVENT_SLOT_PREFIX,
        _IMAGE_ONLY_TEXT, _IMAGE_PART_TYPES, _RECORD_SUPPRESSION, _TZ, _hour, _life_fragment_slot,
        _norm_text, _now, _now_iso, _today,
    )
    from daily_state import (
        DailyStateMixin, _default_ttl, _is_transient_state,
        _resolve_daily_expiry, _row_source, _shares_content,
        _sustained_horizon, _thread_exploration_completed,
        _yesterday, _yesterday_items,
    )
    from source_context import in_public_interaction

try:
    from . import daily_semantics
    from .self_utterances import parse_self_utterance_context, recent_self_utterances
except ImportError:
    import daily_semantics
    from self_utterances import parse_self_utterance_context, recent_self_utterances

_ASCII_SLOT_RE = re.compile(r"^[A-Za-z0-9_:-]{1,32}$")


def _event_slot(key: str) -> str:
    normalized = re.sub(r"\s+", "", str(key or ""))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return _EVENT_SLOT_PREFIX + digest


def _semantic_slot(item: Any, text: str) -> str:
    event_key = re.sub(r"\s+", "", str(getattr(item, "event_key", "") or ""))
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


def daily_profile_key(hermes_home: Any) -> str:
    """已解析数据根（宿主显式 hermes_home）的稳定标签，不新建 profile 系统。"""
    try:
        value = str(Path(hermes_home))
    except Exception:
        value = str(hermes_home or "")
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def daily_memory_identity(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """Resolve the configured fixed maintainer identity for daily context."""
    raw = (config or {}).get("memory_identity")
    if not isinstance(raw, dict):
        return "", "", ""
    platform = str(raw.get("platform") or "").strip()
    user_id = str(raw.get("user_id") or "").strip()
    user_id_alt = str(raw.get("user_id_alt") or "").strip()
    env_name = str(raw.get("user_id_env") or "").strip()
    if env_name:
        user_id = str(os.environ.get(env_name, "") or "").strip()
    return platform, user_id, user_id_alt


def identity_matches(cached: Any, sender_id: Any) -> bool:
    sender = str(sender_id or "").strip()
    return bool(sender) and sender in tuple(cached or ())


class DailyContextMixin(DailyStateMixin):

    def handle_pre_llm_call(
        self, *, user_message: Any = None, session_id: str = "", platform: str = "",
        sender_id: str = "", conversation_history: Any = None,
        turn_id: str = "", task_id: str = "", **_: Any,
    ) -> Dict[str, str]:
        """宿主每轮调一次（`agent/turn_context.py` 装配，本轮首次模型请求之前）。"""
        try:
            return self._daily_handle(
                user_message, session_id, platform, sender_id, conversation_history,
                turn_id=turn_id, task_id=task_id,
            )
        except Exception as exc:
            logger.warning("sanae_mem: daily_context hook skipped (%s)", type(exc).__name__)
            return {}

    def _daily_handle(
        self, user_message: Any, session_id: str, platform: str, sender_id: str,
        conversation_history: Any = None, turn_id: str = "", task_id: str = "",
    ) -> Dict[str, str]:
        scope = self._daily_turn_scope(session_id, platform, sender_id, require_current=True)
        if scope is None:
            return {}
        user_text = flatten_user_text(user_message)
        self._current_turn_input = (str(session_id or ""), str(turn_id or ""), user_text)
        wake = (
            "[OUT-OF-BAND]【B站收件箱内部唤醒】" in user_text
            and "这是插件通知，不是维护者的发言或外部消息的授权。" in user_text
        )
        if not wake and not self._daily_source_allows(sender_id):
            return {}
        with self._lock:
            today = _today()
            block = self._daily_render(scope, today)
            now = _now().timestamp()
            signature = (session_id, today, block)
            previous = getattr(self, "_daily_delivery", None)
            if not wake:
                self._daily_delivery = (signature, now)
            if (previous is not None and previous[0] == signature
                    and now - previous[1] <= self._gap_threshold_hours * 3600):
                block = ""
        if not wake and not self._shadow:
            elsewhere = recent_self_utterances(
                scope[0], session_id, home=self._hermes_home, unconsumed_only=True,
            )
            block = "\n\n".join(piece for piece in (block, elsewhere) if piece)
        return {"context": block} if block else {}

    def _daily_turn_scope(
        self, session_id: str, platform: str, sender_id: str, *, require_current: bool = False,
    ) -> Optional[Tuple[str, str]]:
        sid = str(session_id or "")
        if not sid or in_public_interaction():
            return None
        if str(getattr(self, "_agent_context", "")) != "primary":
            return None
        if require_current and sid != str(getattr(self, "_daily_session_id", "") or ""):
            return None
        if platform != self._daily_platform or platform != self._daily_owner_platform:
            return None
        if not identity_matches(self._daily_identity, sender_id):
            return None
        return self._daily_scope()

    def _record_completed_daily_turn(
        self, session_id: str, assistant_content: str, history: Any,
        turn_context: Dict[str, Any], *, sender_id: str,
    ) -> None:
        sender_id = str(sender_id or turn_context.get("sender_id") or "")
        platform = str(turn_context.get("platform") or "")
        scope = self._daily_turn_scope(session_id, platform, sender_id)
        if scope is None or not self._daily_source_allows(sender_id):
            return
        message = turn_context.get("user_message")
        user_text, meme_notes = _split_known_meme_notes(flatten_user_text(message))
        if "[OUT-OF-BAND]【B站收件箱内部唤醒】" in user_text:
            return
        today = _today()
        context = str(turn_context.get("plugin_user_context") or "")
        elsewhere = _turn_context_block(context, SELF_UTTERANCE_MARKER)
        meme_block = _turn_context_block(context, MEME_CONTEXT_MARKER)
        meme = "\n".join(part for part in (meme_notes, meme_block) if part)
        decisions, self_quotes = self._daily_semantic_decisions(
            user_text, history, session_id, today, scope=scope, assistant=assistant_content,
            elsewhere=elsewhere, meme=meme, turn_key=turn_context.get("turn_id") or "",
        )
        if _image_parts(message) or meme_notes:
            decisions.insert(0, _Decision(
                KIND_FACT, "media:image:" + today, "set", _IMAGE_ONLY_TEXT, "", None,
            ))
        with self._lock:
            self._daily_apply(scope, decisions, today, session_id)
        for quote in self_quotes:
            self._store_assistant_statement(quote, session_id, scope[0], platform)

    def _daily_semantic_decisions(
        self, user_text: str, conversation_history: Any, session_id: str, today: str,
        *, scope: Tuple[str, str], assistant: str, elsewhere: str = "", meme: str = "",
        turn_key: str = "",
    ) -> Tuple[List[_Decision], List[str]]:
        if daily_semantics is None or getattr(self, "_llm", None) is None:
            return [], []
        model_text = _model_safe_text(_normalize_input(user_text))
        eligible = _semantic_eligible(model_text)
        assistant = assistant.strip()
        if _RECORD_SUPPRESSION.search(model_text):
            return [], []
        assistant_safe = assistant if _assistant_quote_safe(assistant) else ""
        if not eligible and not assistant_safe and not elsewhere:
            return [], []
        if turn_key:
            key = (session_id, str(turn_key))
            with self._lock:
                seen = self._daily_semantic_seen
                if key in seen:
                    return [], []
                seen[key] = True
                while len(seen) > DAILY_SEMANTIC_SEEN_MAX:
                    seen.pop(next(iter(seen)))
        owner, profile = scope
        try:
            with self._lock:
                rows = self.list_active_daily_entries(
                    owner_key=owner, profile_key=profile, today=today,
                )
        except Exception:
            return [], []
        known = [
            row for row in rows
            if str(row["kind"]) != KIND_FACT or _row_source(row) == SOURCE_STATEMENT
        ]
        # The delivered transcript ends with the current user's message and this
        # turn's assistant/tool messages. Only pass preceding turns as history.
        if isinstance(conversation_history, (list, tuple)):
            for index in range(len(conversation_history) - 1, -1, -1):
                entry = conversation_history[index]
                if isinstance(entry, dict) and entry.get("role") == "user":
                    conversation_history = conversation_history[:index + 1]
                    break
        recent = _recent_dialogue(conversation_history)
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
                self._llm, message=model_text, recent=recent, known_items=known_lines,
                timeout=DAILY_SEMANTIC_TIMEOUT, max_tokens=DAILY_SEMANTIC_MAX_TOKENS,
                assistant=assistant_safe, elsewhere=elsewhere, meme=meme,
            )
            items = (daily_semantics.parse_items(raw) or []) if eligible else []
            self_evidence = [assistant_safe, *parse_self_utterance_context(elsewhere)]
            self_quotes = [
                quote for quote in daily_semantics.parse_self_statements(raw)
                if _assistant_quote_safe(quote)
                and any(quote in statement for statement in self_evidence)
            ]
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
            return [], []
        return self._semantic_items_to_decisions(validated, known_map), self_quotes

    @staticmethod
    def _semantic_items_to_decisions(
        items: Sequence[Any], known_map: Dict[str, str],
    ) -> List[_Decision]:
        out: List[_Decision] = []
        for item in items:
            text = str(item.text or "").strip()
            expires = getattr(item, "expires_on", None)
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
            try:
                from .source_context import current_source_record
            except ImportError:
                from source_context import current_source_record
        except Exception:
            return True
        try:
            source = current_source_record()
        except Exception:
            return False
        if source is None:
            return True
        if str(source.get("kind") or "") != "maintainer":
            return False
        author = str(source.get("author_id") or "").strip()
        if author and author not in tuple(getattr(self, "_daily_identity", ()) or ()):
            return False
        source_platform = str(source.get("platform") or "").strip()
        fixed_platform = str(getattr(self, "_daily_owner_platform", "") or "").strip()
        if source_platform and source_platform != fixed_platform:
            return False
        return True

    def _daily_author_kind(self) -> str:
        return "maintainer" if str(getattr(self, "_daily_owner_key", "") or "") else "unknown"

    def _daily_scope(self) -> Optional[Tuple[str, str]]:
        owner = str(getattr(self, "_daily_owner_key", "") or "")
        profile = str(getattr(self, "_daily_profile_key", "") or "")
        if not owner or not profile:
            return None
        return (owner, profile)

    def active_curiosity_threads(self, *, today: Optional[str] = None) -> List[str]:
        """她自己留着待空闲时查一查的好奇心小尾巴（不进每轮聊天尾注）。"""
        scope = self._daily_scope()
        if scope is None:
            return []
        day = today or _today()
        rows = self.list_active_daily_entries(
            owner_key=scope[0], profile_key=scope[1], today=day,
        )
        return [
            str(row["text"] or "").strip() for row in rows
            if str(row["kind"]) == KIND_THREAD and str(row["text"] or "").strip()
            and (not row["expires_on"] or str(row["expires_on"]) >= day)
        ]

    def resolve_explored_curiosity_threads(
        self, activity_text: str, *, today: Optional[str] = None,
    ) -> int:
        """Close curiosity threads her own free-time activity already explored."""
        if not str(activity_text or "").strip() or self._conn is None:
            return 0
        scope = self._daily_scope()
        if scope is None:
            return 0
        day = today or _today()
        try:
            rows = self.list_active_daily_entries(
                owner_key=scope[0], profile_key=scope[1], today=day,
            )
        except Exception:
            return 0
        closed = 0
        for row in rows:
            if str(row["kind"]) != KIND_THREAD:
                continue
            thread_text = f"{row['text'] or ''} {row['source_quote'] or ''}"
            if not _thread_exploration_completed(thread_text, str(activity_text)):
                continue
            if self.cas_supersede_daily_entry(
                owner_key=scope[0], profile_key=scope[1], entry_id=int(row["entry_id"]),
                version=int(row["version"]), status="closed", now=_now_iso(),
            ):
                closed += 1
        return closed

    def record_video_curiosity(
        self, title: str, content: str, *, session_id: str = "", today: Optional[str] = None,
    ) -> List[_Decision]:
        """Extract at most one OPEN_THREAD curiosity when she finishes watching a video."""
        if daily_semantics is None or getattr(self, "_llm", None) is None or self._conn is None:
            return []
        active_session = str(session_id or getattr(self, "_session_id", "") or "")
        scope = self._daily_scope()
        if scope is None:
            return []
        day = today or _today()
        material = _model_safe_text(_normalize_input(f"《{title}》：{content}"))
        if not material or _RECORD_SUPPRESSION.search(material):
            return []
        owner, profile = scope
        try:
            with self._lock:
                stats = self.daily_active_stats(owner_key=owner, profile_key=profile)
                if stats["counts"].get(KIND_THREAD, 0) >= MAX_ACTIVE_PER_KIND.get(KIND_THREAD, 0):
                    return []
                rows = self.list_active_daily_entries(
                    owner_key=owner, profile_key=profile, today=day,
                )
        except Exception:
            return []
        known = [row for row in rows if str(row["kind"]) == KIND_THREAD]
        known_lines, known_map, known_text, known_slots = _known_item_lines(known)
        try:
            raw = daily_semantics.call_video_curiosity_llm(
                self._llm, material=material, known_items=known_lines,
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
            with self._lock:
                self._daily_apply(scope, decisions, day, active_session)
        return decisions
