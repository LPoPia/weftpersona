from __future__ import annotations

import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

try:
    from . import auxiliary
    from .association import AssociationMixin
    from .config_schema import CONFIG_SCHEMA
    from .daily_inputs import _without_known_meme_notes
    from .daily_context import (
        DailyContextMixin, daily_memory_identity, daily_profile_key,
    )
    from .impression import IMPRESSION_SCHEMA
    from .ingest import IngestMixin
    from .inner_wiring import INNER_STATE_SCHEMA, InnerWiringMixin
    from .lifecycle import LifecycleMixin
    from .llm_guard import LlmGuardMixin
    from .meme_schemas import (
        MEME_DELETE_SCHEMA, MEME_SAVE_SCHEMA, MEME_SEARCH_SCHEMA, MEME_UPDATE_SCHEMA,
    )
    from .memes import MemesMixin
    from .memory_guard import guard_memory_write
    from .read_gate import ReadGateMixin
    from .resident import ResidentMixin
    from .retrieval import RetrievalMixin
    from .schemas import *
    from .source_context import self_attribution
    from .storage import StorageMixin
    from .textutil import (
        _bool, _configure_embedding, _load_plugin_config, _owner_key,
        hermes_home, record_home_resolution,
    )
    from .words import *
except ImportError:
    import auxiliary
    from association import AssociationMixin
    from config_schema import CONFIG_SCHEMA
    from daily_inputs import _without_known_meme_notes
    from daily_context import (
        DailyContextMixin, daily_memory_identity, daily_profile_key,
    )
    from impression import IMPRESSION_SCHEMA
    from ingest import IngestMixin
    from inner_wiring import INNER_STATE_SCHEMA, InnerWiringMixin
    from lifecycle import LifecycleMixin
    from llm_guard import LlmGuardMixin
    from meme_schemas import (
        MEME_DELETE_SCHEMA, MEME_SAVE_SCHEMA, MEME_SEARCH_SCHEMA, MEME_UPDATE_SCHEMA,
    )
    from memes import MemesMixin
    from memory_guard import guard_memory_write
    from read_gate import ReadGateMixin
    from resident import ResidentMixin
    from retrieval import RetrievalMixin
    from schemas import *
    from source_context import self_attribution
    from storage import StorageMixin
    from textutil import (
        _bool, _configure_embedding, _load_plugin_config, _owner_key,
        hermes_home, record_home_resolution,
    )
    from words import *

logger = logging.getLogger(__name__)


class SanaeMemProvider(
    AssociationMixin, RetrievalMixin, ReadGateMixin, ResidentMixin, IngestMixin, LifecycleMixin,
    InnerWiringMixin, LlmGuardMixin, DailyContextMixin, MemesMixin, StorageMixin, MemoryProvider
):
        sync_after_delivery = True

        def __init__(self, config: Optional[dict] = None):
            self._config = dict(_load_plugin_config() if config is None else config)
            self._shadow = _bool(self._config.get("shadow"), False)
            gate = str(self._config.get("read_gate", "explicit")).lower()
            self._read_gate = gate if gate in {"explicit", "scheduled"} else "explicit"
            self._inject_max_chars = max(1, min(400, int(self._config.get("inject_max_chars", 400))))
            self._daily_scheduled_asks = max(0, int(self._config.get("daily_scheduled_asks", 2)))
            self._gap_continue = _bool(self._config.get("gap_continue"), True)
            self._gap_threshold_hours = max(0.0, float(self._config.get("gap_threshold_hours", 2)))
            self._distill_threshold = max(0.0, min(1.0, float(self._config.get("distill_threshold", 0.6))))
            self._distill_daily_cap = max(0, int(self._config.get("distill_daily_cap", 2)))
            self._emotion_ttl_days = max(1, int(self._config.get("emotion_ttl_days", 90)))
            self._expiry_review_days = max(1, int(self._config.get("expiry_review_days", 30)))
            self._expiring_importance_decay = max(
                0.0, min(1.0, float(self._config.get("expiring_importance_decay", 0.9)))
            )
            self._expiring_importance_floor = max(
                0.0, min(1.0, float(self._config.get("expiring_importance_floor", 0.4)))
            )
            requested = str(self._config.get("semantic_backend", "chroma")).lower()
            self._semantic_backend = requested if requested in {"chroma", "sqlite"} else "chroma"
            self._persona_name = str(self._config.get("persona_name") or "早苗").strip() or "早苗"
            self._partner_name = str(self._config.get("partner_name") or "").strip()
            self._partner_role = str(self._config.get("partner_role") or "挚友").strip() or "挚友"
            pronoun = str(self._config.get("partner_pronoun") or "他").strip()
            self._partner_pronoun = pronoun if pronoun in {"他", "她"} else "他"
            self._conn: Optional[sqlite3.Connection] = None
            self._chroma_client = None
            self._collection = None
            self._lock = threading.RLock()
            self._session_id = ""
            self._seen_session_ids: set[str] = set()
            self._resumed_session_ids: set[str] = set()
            self._agent_context = "primary"
            self._platform = ""
            self._owner_key = ""
            self._llm = None
            self._hermes_home = hermes_home()
            self._data_dir = Path(self._config.get("data_dir", Path(self._hermes_home) / "sanae-mem")).expanduser()
            self._daily_profile_key = ""
            self._daily_session_id = ""
            self._daily_platform = ""
            self._daily_owner_key = ""
            self._daily_owner_platform = ""
            self._daily_identity: tuple = ()
            self._daily_semantic_seen: dict = {}
            self._last_recall: Optional[dict] = None
            self._association_cache: Optional[dict] = None
            self._association_hook_registered = False
            self._association_hook_request = None
            self._association_hot_lock = threading.Lock()
            self._association_owner_key = ""
            self._association_owner_resolver = None
            self._association_read_invalid = False
            self._write_owner_key = ""
            self._write_owner_resolver = None

        @property
        def name(self) -> str:
            return "sanae_mem"

        def is_available(self) -> bool:
            return True

        def initialize(self, session_id: str, **kwargs) -> None:
            self._hermes_home = hermes_home(kwargs.get("hermes_home"))
            configured_dir = self._config.get("data_dir")
            self._data_dir = Path(configured_dir).expanduser() if configured_dir else Path(self._hermes_home) / "sanae-mem"
            self._session_id = session_id
            self._agent_context = str(kwargs.get("agent_context") or "primary")
            self._platform = str(kwargs.get("platform") or "")
            self._maintainer_id = str(kwargs.get("user_id") or kwargs.get("user_id_alt") or "")
            self._owner_key = (
                _owner_key(
                    self._platform,
                    **{key: kwargs.get(key) for key in ("user_id", "user_id_alt")},
                )
                if self._agent_context == "primary"
                else ""
            )
            self._association_owner_key = ""
            self._association_owner_resolver = None
            self._association_read_invalid = False
            self._write_owner_key = ""
            self._write_owner_resolver = None
            self._association_cache = None
            self._association_hook_request = None
            self._last_recall = None
            fixed_platform, fixed_user, fixed_alt = daily_memory_identity(self._config)
            self._daily_owner_key = (
                _owner_key(fixed_platform, user_id=fixed_user, user_id_alt=fixed_alt)
                if fixed_platform and (fixed_user or fixed_alt) else ""
            )
            self._daily_owner_platform = fixed_platform if self._daily_owner_key else ""
            self._daily_identity = tuple(
                value for value in (fixed_user, fixed_alt) if value
            ) if self._daily_owner_key else ()
            self._daily_profile_key = daily_profile_key(self._hermes_home)
            self._daily_session_id = str(session_id or "")
            self._daily_platform = self._platform
            self._data_dir.mkdir(parents=True, exist_ok=True)
            _configure_embedding(self._data_dir)
            self._open_sqlite()
            if self._semantic_backend == "chroma":
                self._open_chroma()
            record_home_resolution(
                kwargs.get("hermes_home"), self._hermes_home, log_dir=self._data_dir
            )

        def on_session_switch(
            self,
            new_session_id: str,
            *,
            parent_session_id: str = "",
            reset: bool = False,
            rewound: bool = False,
            **kwargs: Any,
        ) -> None:
            """会话中途改绑（`/new`、`/resume`、`/branch`、压缩）：把 `_session_id` 跟过去。

            宿主不重建 provider 就改 session_id。测试隔离的写入闸按 session_id 判定，
            不跟过去就会拿旧 id 判定；`_tool_add` 也会把事件写到旧会话 id。

            `parent_session_id` / `reset` / `rewound` 本插件暂不使用（保留形参以对齐契约）。
            """
            try:
                if new_session_id:
                    self._session_id = str(new_session_id)
                    self._daily_session_id = str(new_session_id)
                    self._association_cache = None
                    self._association_hook_request = None
                    self._last_recall = None
                    self._write_owner_key = ""
                    self._write_owner_resolver = None
            except Exception:
                pass

        @staticmethod
        def _association_query_text(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return " ".join(
                    str(part.get("text") or "") for part in value
                    if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
                ).strip()
            return ""

        def _association_gap_likely(self, query: str, session_id: str) -> bool:
            if not self._gap_continue or not query:
                return False
            if session_id and session_id in self._resumed_session_ids:
                return False
            if self._triggers_explicit_past(query) or self._is_trivial_message(query) \
                    or self._is_busy_context(query):
                return False
            fresh = bool(session_id) and session_id not in self._seen_session_ids
            return bool(fresh or self._gap_detected())

        def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
            pass

        def handle_pre_llm_call(self, *args: Any, **kwargs: Any) -> dict:
            """Existing host hook: deliver shared association before memory prefetch.

            The hook is the only place that sees this turn's conversation history, so
            the association read happens here for private chat. ``prefetch`` sees the
            cache and avoids a second same-turn copy.
            The independent meme/expression zone is also read here: text aliases and
            exact image hashes can add a short explanation without touching events.
            Public turns are handled by ``prefetch`` under the reviewed projection.
            """
            session_id = str(kwargs.get("session_id") or self._session_id)
            authored_message = _without_known_meme_notes(kwargs.get("user_message"))
            query = self._association_query_text(authored_message)
            request = {"session_id": session_id, "binding_session_id": self._session_id,
                       "query": query, "closed": False}
            with self._lock:
                self._association_hook_request = request
            base = super().handle_pre_llm_call(*args, **kwargs)
            if not isinstance(base, dict):
                base = {}
            if not self._association_hook_request_active(request):
                return base
            try:
                from .source_context import in_public_interaction
            except ImportError:
                from source_context import in_public_interaction
            if in_public_interaction() or self._agent_context != "primary" or self._shadow:
                return base
            if query:
                session_id = str(kwargs.get("session_id") or self._session_id)
                gap = self._association_gap_likely(query, session_id)
                explicit = self._triggers_explicit_past(query)
                if session_id:
                    self._seen_session_ids.add(session_id)
                result = self.associate_context(
                    query,
                    conversation_history=kwargs.get("conversation_history"),
                    purpose="chat",
                    session_id=session_id,
                    max_chars=_GAP_INJECT_MAX_CHARS if gap else self._inject_max_chars,
                    gap_continuation=gap,
                    hot_path=True,
                    _hook_request=request,
                )
                with self._lock:
                    if not self._association_hook_request_active(request):
                        return base
                    if self._association_cache is not None:
                        self._association_cache["delivered_by_hook"] = True
                    if result.get("status") == "ok" and result.get("context"):
                        via = "gap" if gap else "explicit_past" if explicit else "chat"
                        self._mark_asked(result["items"], via, session_id)
                        if gap:
                            self._resumed_session_ids.add(session_id)
                        prior = str(base.get("context") or "")
                        base["context"] = (prior + "\n\n" + result["context"]).strip()
            try:
                meme_context = self.meme_context_for_turn(
                    authored_message, known_context=kwargs.get("user_message"))
                if meme_context:
                    prior = str(base.get("context") or "")
                    base["context"] = (prior + "\n\n" + meme_context).strip()
            except Exception:
                logger.debug("sanae_mem: meme context lookup skipped", exc_info=True)
            return base

        def _association_hook_request_active(self, request) -> bool:
            return (self._association_hook_request is request
                    and not request["closed"]
                    and request["binding_session_id"] == self._session_id)

        def recall_status(self):
            """Host recall indicator: only a delivered ``ok`` association counts."""
            state = self._last_recall
            if not state or state.get("status") != "ok" or not state.get("items"):
                return None
            try:
                from agent.memory_provider import RecallStatus
            except Exception:
                return None
            return RecallStatus("sanae_mem", len(state["items"]))

        def record_video_watch(self, observation: dict) -> bool:
            """Host-only record after a selected result was delivered to the main turn."""
            from agent.interaction_context import current_interaction
            context = current_interaction()
            write_owner = self._write_owner()
            if not (context and context.public and context.audience == "public"
                    and context.source.get("platform") == "bilibili"
                    and (self._owner_scoped or bool(write_owner)
                         or self._agent_context == "cron")
                    and self._conn is not None and isinstance(observation, dict)):
                return False
            bv, mode = observation.get("bvid"), observation.get("mode")
            if (not isinstance(bv, str) or not re.fullmatch(r"BV[0-9A-Za-z]{10}", bv)
                    or observation.get("status") != "ok" or mode not in {"video_analysis", "captions_only"}):
                return False
            if mode == "video_analysis":
                content = str(observation.get("analysis") or "")[:1600]
                experience = "收到该视频的抽样画面分析，音轨未核验"
            else:
                lines = observation.get("lines") or (observation.get("captions") or {}).get("lines") or []
                content = "；".join(str(x.get("text") or "") for x in lines[:24] if isinstance(x, dict))[:1300]
                experience = "阅读了该视频的字幕，未观看画面"
            if not content:
                return False
            url = f"https://www.bilibili.com/video/{bv}/"
            title = str(observation.get("title") or bv).replace("\n", " ")[:140]
            summary = (f"早苗在B站{experience}：《{title}》（{bv}，CID {observation.get('cid', '未知')}）。"
                       f"范围：{observation.get('coverage', mode)}。来源：{mode}；内容理解未经独立核实：{content}")
            source = {"kind": "self", "platform": "bilibili", "ref": url,
                      "time": datetime.now().astimezone().isoformat(),
                      "certainty": "observed", "visibility": "public"}
            subject, fact, involved, scope = self_attribution("")
            with self._lock:
                event_id = self._ingest_shared(
                    summary, self._session_id, owner_key=write_owner,
                    source="explicit", source_context=source,
                    experience_subject=subject, fact_subject=fact,
                    involved_owner_keys=involved, read_scope=scope)
                if event_id is None:
                    return False
                self._conn.execute(
                    "UPDATE events SET ttl_expire=NULL WHERE id=? AND owner_key=? "
                    "AND blacklist=0 AND expired_at IS NULL", (event_id, write_owner))
                self._conn.commit()
                if callable(context.review_excerpt) and context.review_excerpt(summary[:900]) is True:
                    self.set_public_excerpt(event_id, summary[:900])
            try:
                self.resolve_explored_curiosity_threads(summary)
                self.record_video_curiosity(title, content, session_id=self._session_id)
            except Exception as exc:
                logger.warning("sanae_mem: video curiosity hook skipped (%s)", type(exc).__name__)
            return True

        def get_public_video_watch(self, bvid: str) -> str:
            """Reviewed active recollection; technical caches cannot revive forgotten events."""
            from agent.interaction_context import current_interaction
            context = current_interaction()
            try:
                read_owner = self._association_read_owner()
            except PermissionError:
                return ""
            if not (context and context.public and context.audience == "public"
                    and context.source.get("platform") == "bilibili"
                    and self._conn is not None and isinstance(bvid, str)
                    and re.fullmatch(r"BV[0-9A-Za-z]{10}", bvid)):
                return ""
            with self._lock:
                row = self._conn.execute(
                    "SELECT public_excerpt FROM events WHERE blacklist=0 "
                    "AND expired_at IS NULL AND (ttl_expire IS NULL OR ttl_expire>?) "
                    "AND public_excerpt<>'' AND json_valid(source_context) "
                    "AND json_extract(source_context,'$.platform')='bilibili' "
                    "AND json_extract(source_context,'$.kind')='self' "
                    "AND json_extract(source_context,'$.visibility')='public' "
                    "AND ((experience_subject='sanae' AND fact_subject='sanae' "
                    "AND json_valid(read_scope) "
                    "AND json_extract(read_scope,'$.kind')='self') "
                    "OR (owner_key=? AND owner_key<>'')) "
                    "AND json_extract(source_context,'$.ref')=? ORDER BY id DESC LIMIT 1",
                    (datetime.now().astimezone().isoformat(), read_owner,
                     f"https://www.bilibili.com/video/{bvid}/")).fetchone()
            return str(row["public_excerpt"] or "")[:900] if row else ""

        def get_tool_schemas(self) -> List[Dict[str, Any]]:
            meme_search_schema = dict(MEME_SEARCH_SCHEMA)
            try:
                summary = self._meme_available_sticker_summary()
            except Exception:
                summary = ""
            if summary:
                meme_search_schema["description"] = f"{MEME_SEARCH_SCHEMA['description']} {summary}"
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
            ]

        def get_config_schema(self) -> List[Dict[str, Any]]:
            return CONFIG_SCHEMA


def register(ctx) -> None:
    """Register the provider first so auxiliary registration can fail independently."""
    provider = SanaeMemProvider()
    ctx.register_memory_provider(provider)
    try:
        ctx.register_hook("pre_tool_call", guard_memory_write)
    except Exception as exc:
        logger.warning("sanae_mem: memory write guard registration skipped (%s)", type(exc).__name__)
    try:
        from .memory_guard import guard_controlled_memory_write
    except ImportError:
        from memory_guard import guard_controlled_memory_write
    try:
        ctx.register_hook("pre_tool_call", guard_controlled_memory_write)
    except Exception as exc:
        logger.warning("sanae_mem: controlled memory write gate registration skipped (%s)", type(exc).__name__)
    try:
        from .memory_guard import remind_memory_result
    except ImportError:
        from memory_guard import remind_memory_result
    try:
        ctx.register_hook("transform_tool_result", remind_memory_result)
    except Exception as exc:
        logger.warning("sanae_mem: memory reminder hook registration skipped (%s)", type(exc).__name__)
    try:
        ctx.register_hook("pre_llm_call", provider.handle_pre_llm_call)
        provider._association_hook_registered = True
    except Exception as exc:
        logger.warning("sanae_mem: daily-context hook registration skipped (%s)", type(exc).__name__)
    try:
        provider._llm = auxiliary.register(ctx)
    except Exception as exc:
        logger.warning("sanae_mem: auxiliary registration skipped (%s)", type(exc).__name__)
    try:
        auxiliary.register_daily(ctx)
    except Exception as exc:
        logger.warning("sanae_mem: daily Flash registration skipped (%s)", type(exc).__name__)
