from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..cognitive.association import Association
from ..cognitive.daily_context import DailyContext, daily_memory_identity, daily_profile_key
from ..cognitive.daily_state import DailyState
from ..cognitive.inner_state import InnerStateService
from ..cognitive.inner_state_types import InnerStateTarget
from ..cognitive.inner_wiring import InnerWiring, _submit
from ..domains.commitments import Commitments
from ..domains.memes import MemeStore
from ..domains.video_memory import VideoMemory
from ..governance.account_binding import confirm_bindings, current_profile, maintainer_anchor, read_owners
from ..governance.association_state import AssociationState
from ..governance.grounded_review import GroundedReview
from ..governance.group_scope import GroupGrant
from ..governance.llm_guard import AuxiliaryModel, LlmGuard
from ..governance.memory_axis import MemoryAxis
from ..governance.memory_evidence import UserEvidence
from ..governance.promotion_review import PromotionReview
from ..governance.read_gate import ReadGate
from ..governance.resident import ResidentMemory
from ..governance.source_context import current_chat_scope, resolve_chat_identity
from ..governance.source_intake import SourceIntakeContext
from ..infra.host_bridge import MemoryProvider, RecallStatus, hermes_home_scope
from ..infra.status_store import StatusStore
from ..infra.textutil import _bool, _configure_embedding, hermes_home, record_home_resolution
from ..storage.background_tasks import BackgroundTasks
from ..storage.event_maintenance import EventMaintenance
from ..storage.event_review_store import EventReviewStore
from ..storage.event_store import EventStore
from ..storage.retrieval import Retrieval
from ..storage.sqlite_facade import SqliteFacade
from ..storage.storage import Storage
from .config import live_plugin_config
from .config_schema import CONFIG_SCHEMA
from .group_ingest import GroupDeliveryContext, write_group_delivery_event
from .ingest import Ingest
from .lifecycle import Lifecycle
from .private_delivery_settler import PrivateDeliveryRoute
from .provider_wiring import (
    _parse_provider_settings,
    apply_private_identity,
    assign_provider_settings,
    authorized_free_think_owner,
    bind_active_provider,
    build_group_delivery_context,
    build_source_intake_context,
    group_scope_grant,
    refresh_provider_config,
    resolve_integration_owner,
    resolve_private_delivery_route,
    run_pre_llm_call,
    run_provider_nightly_journal,
    sync_provider_turn,
    unbind_active_provider,
    wire_provider_components,
    write_maintenance_status,
    write_owner,
)
from .scheduled import run_installed_nightly_journal
from .schemas import _PRIVATE_REVIEW_SCHEMA as _PRIVATE_REVIEW_SCHEMA
from .tool_router import ToolRouter

logger = logging.getLogger(__name__)
_active_provider_lock = threading.RLock()


class SanaeMemProvider(MemoryProvider):
        sync_after_delivery = True

        def __init__(self, config: Optional[dict] = None):
            self._explicit_config = config is not None
            self._hermes_home = hermes_home()
            self._config = live_plugin_config(Path(self._hermes_home)) if config is None else dict(config)
            settings = _parse_provider_settings(self._config)
            assign_provider_settings(self, settings)
            self._conn: Optional[sqlite3.Connection] = None
            self._lock: threading.RLock = threading.RLock()
            self._lifecycle_lock = threading.Lock()
            self._background_tasks = BackgroundTasks()
            self._sqlite: SqliteFacade = SqliteFacade(
                lock=self._lock,
                conn_getter=lambda: self._conn,
                conn_setter=self._set_sqlite_conn,
            )
            self._inject_max_chars: int
            self._distill_daily_cap: int
            self._resident_review_daily_cap: int
            self._emotion_ttl_days: int
            self._persona_name: str
            self._partner_name: str
            self._partner_role: str
            self._partner_pronoun: str
            self._event_review_store: EventReviewStore
            self._session_id: str
            self._agent_context: str
            self._owner_key: str
            self._read_owner_keys: tuple[str, ...]
            self._event_store: EventStore
            self._video_memory: VideoMemory
            self._release_active_provider: Callable[[], None] | None
            self._active_provider_desired: bool
            self._release_source_sink: Callable[[], None] | None
            self._agent_identity: str
            self._platform: str
            self._llm: Any
            self._data_dir: Path
            self._storage: Storage
            self._event_maintenance: EventMaintenance
            self._status_store: StatusStore
            self.ingest: Ingest
            self._promotion_review: PromotionReview
            self._daily_profile_key: str
            self._llm_guard: LlmGuard
            self._daily_owner_key: str
            self._daily_owner_platform: str
            self._daily_identity: tuple[str, ...]
            self._maintainer_id: str
            self._daily_state: DailyState
            self._daily_context: DailyContext
            self._association_state: AssociationState
            self._memory_axis: MemoryAxis
            self._inner_state: InnerStateService
            self._association: Association
            self._memes: MemeStore
            self._commitments: Commitments
            self._read_gate: ReadGate
            self._grounded_review: GroundedReview
            self._resident: ResidentMemory
            self._inner_wiring: InnerWiring
            self._retrieval: Retrieval
            self._lifecycle: Lifecycle
            self._tool_router: ToolRouter
            wire_provider_components(self, settings)

        def query_events(
            self, now: str, topic: str = "", created_after: Optional[str] = None,
            include_expiring: bool = False, unscoped: bool = False,
            owner_key: Optional[str] = None,
            experience_subject: Optional[str] = None,
            fact_subject: Optional[str] = None, read_context: bool = False,
            writable: bool = False, dormant_owner: Optional[str] = None,
        ) -> List[sqlite3.Row]:
            return self._event_store.query_events(
                now, topic, created_after, include_expiring, unscoped, owner_key,
                experience_subject, fact_subject, read_context, writable, dormant_owner,
            )

        def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
            return self._lifecycle.on_pre_compress(messages)

        def current_turn_context(
            self, *, session_id: str = "", turn_id: str = "", synthetic: bool = False,
            sender_id: str = "", sender_id_alt: str = "", durable_native_thread: bool = False,
            **_: object,
        ) -> str:
            return self._inner_wiring.current_turn_context(
                session_id=session_id, turn_id=turn_id, synthetic=synthetic, sender_id=sender_id,
                sender_id_alt=sender_id_alt, durable_native_thread=durable_native_thread,
            )

        def _group_scope_grant(self) -> GroupGrant | None:
            return group_scope_grant(self)

        def _group_impression_target(self, grant: GroupGrant | None = None) -> tuple[Path, str, str] | None:
            return self._inner_wiring.group_impression_target(
                group_scope_grant(self) if grant is None else grant,
            )

        def record_commitment(
            self, owner_key: str, text: str, *, source_quote: str = "", turn_id: str = "",
            now: datetime | str | int | float | None = None, masked_text: str = "",
            source_context: str = "",
        ) -> int | None:
            return self._commitments.record_commitment(
                owner_key, text, source_quote=source_quote, turn_id=turn_id, now=now,
                masked_text=masked_text, source_context=source_context,
            )

        def list_commitments(
            self, owner_key: str | None = None, status: str | None = None,
        ) -> list[dict[str, Any]]:
            return self._commitments.list_commitments(owner_key, status)

        def update_commitments_after_delivery(
            self, user_text: str, assistant_text: str, *, owner_key: str, turn_id: str,
            tool_results: Sequence[object] = (),
        ) -> bool:
            return self._commitments.update_commitments_after_delivery(
                user_text, assistant_text, owner_key=owner_key, turn_id=turn_id,
                tool_results=tool_results,
            )

        def resolve_commitment_with_reason(
            self, commitment_id: int, action: str, evidence: str = "",
            expected_version: int | None = None,
        ) -> tuple[bool, str]:
            return self._commitments.resolve_commitment_with_reason(
                commitment_id, action, evidence=evidence, expected_version=expected_version,
            )

        def review_outcome(self, event_id: int, now: datetime | None = None) -> dict:
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            review = self._promotion_review.review_promotion_candidate(
                int(event_id), now, owner_key=str(write_owner(self) or ""))
            review["ok"] = bool(review.get("resident_available", True))
            return review

        def _set_sqlite_conn(self, conn: Optional[sqlite3.Connection]) -> None:
            self._conn = conn

        def route_resident_write(
            self, target: str, payload: dict[str, object], *,
            source_ctx: UserEvidence | None = None, session_id: str = "",
            turn_id: str = "", llm: AuxiliaryModel | None = None,
        ) -> Optional[str]:
            return self._llm_guard.route_resident_write(
                target, payload, source_ctx=source_ctx, session_id=session_id, turn_id=turn_id, llm=llm,
            )

        def update_inner_state(
            self, *, turn_id: str, user_text: str, assistant_text: str, reasoning: str = "",
            experiences: str = "", image_context: str = "", session_id: str = "", now: float | None = None,
            target: InnerStateTarget | None = None,
            impression_target: tuple[Path, str, str] | None = None,
            preserve_processed_at: bool = False, owner_key: str, internal: bool = False,
        ) -> bool:
            return self._inner_state.update_inner_state(
                turn_id=turn_id, user_text=user_text, assistant_text=assistant_text,
                reasoning=reasoning, experiences=experiences, image_context=image_context,
                session_id=session_id, now=now, target=target, impression_target=impression_target,
                preserve_processed_at=preserve_processed_at, owner_key=owner_key, internal=internal,
            )

        def open_daily_entry(
            self, *, owner_key: str, profile_key: str, kind: str, slot: str, text: str,
            opened_on: str, now: str, expires_on: Optional[str] = None,
            source_quote: str = "", source_turn: str = "", source_kind: str = "statement",
            author_kind: str = "unknown",
        ) -> Optional[int]:
            return self._daily_state.open_daily_entry(
                owner_key=owner_key, profile_key=profile_key, kind=kind, slot=slot, text=text,
                opened_on=opened_on, now=now, expires_on=expires_on, source_quote=source_quote,
                source_turn=source_turn, source_kind=source_kind, author_kind=author_kind,
            )

        def resolve_explored_curiosity_threads(
            self, activity_text: str, *, today: str | None = None,
        ) -> int:
            return self._daily_context.resolve_explored_curiosity_threads(activity_text, today=today)

        def runtime_config(self, home: Path | None = None) -> dict[str, Any]:
            return live_plugin_config(
                Path(self._hermes_home) if home is None else home,
                defaults=self._config if self._explicit_config else None,
            )

        def shadow_enabled(self) -> bool:
            return _bool(self.runtime_config().get("shadow"), False)

        def attach_runtime_llm(self, llm: Any) -> None:
            self._llm = llm

        def runtime_llm(self) -> Any:
            return self._llm

        def own_runtime_registration(self, release: Callable[[], None]) -> None:
            self._release_source_sink = release

        def open_runtime_worker(
            self, home: Path, config: dict[str, Any], *, session_id: str,
            platform: str, agent_context: str = "primary",
        ) -> SanaeMemProvider:
            with hermes_home_scope(home):
                worker = SanaeMemProvider(config=config)
                worker.attach_runtime_llm(self.runtime_llm())
                try:
                    worker.initialize(session_id, hermes_home=str(home), platform=platform,
                                      agent_context=agent_context, hook_worker=True)
                except BaseException:
                    worker.shutdown()
                    raise
                return worker

        def report_skill_mutation(self, data_dir: Path, **status: Any) -> None:
            write_maintenance_status(self, "skill_mutation_review", data_dir=data_dir, **status)

        def permits_group_delivery(self) -> bool:
            return group_scope_grant(self) is not None

        def source_intake_context(self) -> SourceIntakeContext:
            return build_source_intake_context(self)

        def integration_owner(
            self, *, home: Path, allow_cron: bool, write: bool, cron_active: bool,
        ) -> str | None:
            return resolve_integration_owner(
                self, home=home, allow_cron=allow_cron, write=write, cron_active=cron_active,
            )

        def resolve_private_delivery_route(self) -> PrivateDeliveryRoute:
            return resolve_private_delivery_route(self)

        def group_delivery_context(self) -> GroupDeliveryContext:
            return build_group_delivery_context(self)

        def write_group_delivery_event(
            self, grant: GroupGrant, *, quote: str, summary: str, delivered_at: object,
            delivery_ref: str, paraphrase_ref: str,
        ) -> int | None:
            self._storage.ensure_vectors()
            return write_group_delivery_event(
                grant, quote=quote, summary=summary, delivered_at=delivered_at,
                delivery_ref=delivery_ref, paraphrase_ref=paraphrase_ref,
                session_id=self._session_id, data_dir=self._data_dir, sqlite=self._sqlite,
                write_quote=self.ingest.write_event,
                promote_summary=lambda event_id, text, keywords, vector, now: (
                    self._event_maintenance.promote_distilled_event(
                        event_id, text, keywords, vector, now, record_distilled_at=False,
                    )
                ),
                upsert_vector=self._storage.upsert_vector,
            )

        @property
        def name(self) -> str:
            return "sanae_mem"

        @property
        def sqlite(self) -> SqliteFacade:
            return self._sqlite

        @property
        def storage(self) -> Storage:
            return self._storage

        @property
        def background_tasks(self) -> BackgroundTasks:
            return self._background_tasks

        @property
        def association_state(self) -> AssociationState:
            return self._association_state

        @property
        def event_store(self) -> EventStore:
            return self._event_store

        @property
        def event_maintenance(self) -> EventMaintenance:
            return self._event_maintenance

        @property
        def event_review_store(self) -> EventReviewStore:
            return self._event_review_store

        @property
        def grounded_review(self) -> GroundedReview:
            return self._grounded_review

        @property
        def promotion_review(self) -> PromotionReview:
            return self._promotion_review

        @property
        def resident(self) -> ResidentMemory:
            return self._resident

        @property
        def commitments(self) -> Commitments:
            return self._commitments

        @property
        def inner_state(self) -> InnerStateService:
            return self._inner_state

        @property
        def inner_wiring(self) -> InnerWiring:
            return self._inner_wiring

        @property
        def video_memory(self) -> VideoMemory:
            return self._video_memory

        @property
        def daily_context(self) -> DailyContext:
            return self._daily_context

        @property
        def memory_axis(self) -> MemoryAxis:
            return self._memory_axis

        @property
        def lifecycle(self) -> Lifecycle:
            return self._lifecycle

        @property
        def read_gate(self) -> ReadGate:
            return self._read_gate

        @property
        def association(self) -> Association:
            return self._association

        @property
        def status_store(self) -> StatusStore:
            return self._status_store

        @property
        def llm_guard(self) -> LlmGuard:
            return self._llm_guard

        @property
        def retrieval(self) -> Retrieval:
            return self._retrieval

        @property
        def memes(self) -> MemeStore:
            return self._memes

        def backup_paths(self) -> List[str]:
            return self._storage.backup_paths(self._data_dir)

        def is_available(self) -> bool:
            return True

        def initialize(self, session_id: str, **kwargs) -> None:
            with self._lifecycle_lock, self._background_tasks.pause(unseal_on_exit=True):
                self._initialize(session_id, **kwargs)

        def _initialize(self, session_id: str, **kwargs) -> None:
            with _active_provider_lock:
                self._active_provider_desired = False
                unbind_active_provider(self)
            if self._sqlite.is_open:
                self._storage.close()
            home = hermes_home(kwargs.get("hermes_home"))
            home_changed = Path(home).resolve() != Path(self._hermes_home).resolve()
            if home_changed:
                self._read_gate.reset_home()
            elif str(session_id or "") != self._session_id:
                self._read_gate.reset_group_delivery()
            self._hermes_home = home
            if home_changed and not self._explicit_config:
                refresh_provider_config(self, live_plugin_config(Path(home)))
            configured_dir = self._config.get("data_dir")
            self._data_dir = Path(configured_dir).expanduser() if configured_dir else Path(self._hermes_home) / "sanae-mem"
            self._session_id = str(session_id or "")
            self._agent_context = str(kwargs.get("agent_context") or "primary")
            self._agent_identity = str(kwargs.get("agent_identity") or "")
            self._platform = str(kwargs.get("platform") or "").lower()
            scope = current_chat_scope()
            self._owner_key = ""
            self._read_owner_keys = ()
            self._association_state.reset()
            fixed_platform, fixed_user, fixed_alt = daily_memory_identity(self._config)
            self._maintainer_id = fixed_user or fixed_alt
            self._daily_owner_key = maintainer_anchor(self._config)
            if self._agent_context == "cron":
                self._owner_key = authorized_free_think_owner(self)
            self._daily_owner_platform = fixed_platform if self._daily_owner_key else ""
            self._daily_identity = tuple(
                value for value in (fixed_user, fixed_alt) if value
            ) if self._daily_owner_key else ()
            self._daily_profile_key = daily_profile_key(self._hermes_home)
            self._data_dir.mkdir(parents=True, exist_ok=True)
            _configure_embedding(self._data_dir)
            self._storage.open_sqlite(
                self._data_dir, archive_dir=Path(self._hermes_home) / "memory-archive",
            )
            if not self._sqlite.is_open:
                raise RuntimeError("ViveMind SQLite connection was not opened")
            active_profile = current_profile(self._hermes_home, scope)
            with self._sqlite.transaction() as connection:
                confirm_bindings(connection, self._config, profile=active_profile)
            apply_private_identity(self, scope)
            if self._agent_context == "cron" and self._owner_key:
                self._read_owner_keys = self._sqlite.read(
                    lambda conn: read_owners(
                        self._owner_key, config=self._config, connection=conn,
                        profile=active_profile,
                    )
                )
            if not kwargs.get("hook_worker"):
                self._storage.ensure_vectors()
            record_home_resolution(
                kwargs.get("hermes_home"), self._hermes_home, log_dir=self._data_dir
            )
            if not kwargs.get("hook_worker"):
                with _active_provider_lock:
                    if self._session_id and self._sqlite.is_open:
                        self._active_provider_desired = True
                        bind_active_provider(self)

        def is_active_session(self, session_id: str) -> bool:
            return self._sqlite.is_open and self._session_id == session_id

        @property
        def is_cron_context(self) -> bool:
            return self._agent_context == "cron"

        def shutdown(self) -> None:
            with self._lifecycle_lock:
                with _active_provider_lock:
                    self._active_provider_desired = False
                    unbind_active_provider(self)
                    if self._release_source_sink is not None:
                        self._release_source_sink()
                        self._release_source_sink = None
                self._background_tasks.seal_and_wait()
                self._storage.close()

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

            宿主不重建 provider 就改 session_id；不跟过去，工具写入会落到旧会话 id。

            `parent_session_id` / `reset` / `rewound` 本插件暂不使用（保留形参以对齐契约）。
            """
            if not new_session_id:
                return
            old_session_id = self._session_id
            target_session_id = str(new_session_id)
            try:
                with _active_provider_lock:
                    old_session_id = self._session_id
                    if target_session_id != old_session_id:
                        self._read_gate.reset_group_delivery()
                    with self._association_state.lock:
                        self._session_id = target_session_id
                        self._association_state.switch_session()
                    unbind_active_provider(self)
                    if self._active_provider_desired:
                        bind_active_provider(self)
            except Exception:
                logger.exception(
                    "sanae_mem: session switch rebind failed (old=%r, new=%r)",
                    old_session_id,
                    target_session_id,
                )
                raise

        def system_prompt_block(self) -> str:
            return self._read_gate.system_prompt_block()

        def group_turn_context(self, query: str, *, session_id: str = "", turn_id: str = "") -> str:
            return self._read_gate.group_turn_context(query, session_id=session_id, turn_id=turn_id)

        def prefetch(self, query: str, *, session_id: str = "", scheduled: bool = False) -> str:
            return self._read_gate.prefetch(
                query,
                check_hook_recall=lambda q, sid, pub: self._association_state.consume_prefetch_hook_recall(
                    q, sid, pub, validate_read=lambda: self._association_state.validate_read(self._owner_key),
                ),
                session_id=session_id,
                scheduled=scheduled,
            )

        def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
            apply_private_identity(self)
            self._read_gate.reset_group_delivery()

        def record_private_turn_events(
            self, user_text: str, *, current_message: dict[str, object] | None,
            session_id: str, platform: str, author_id: str,
        ) -> tuple[list[int], bool]:
            scope = current_chat_scope()
            identity = self._sqlite.read(
                lambda conn: resolve_chat_identity(scope, self._config, connection=conn)
            )
            if (not identity.private_owner_key or identity.private_owner_key != self._owner_key
                    or scope.platform != self._platform):
                return [], False
            return self.ingest.record_private_turn(
                user_text, current_message=current_message, session_id=session_id,
                platform=platform, author_id=author_id, scope=scope, identity=identity,
            )

        def known_meme_terms(self, parts: List[str]) -> List[str]:
            return self._memes.known_meme_terms(parts)

        def public_meme_senses(self, parts: List[str]) -> List[Dict[str, Any]]:
            return self._memes.public_meme_senses(parts)

        def meme_context_for_public_material(self, parts: List[str], *, public_only: bool = False) -> str:
            return self._memes.meme_context_for_public_material(parts, public_only=public_only)

        def meme_context_for_turn(self, user_message: Any, *, known_context: Any = None) -> str:
            return self._memes.meme_context_for_turn(user_message, known_context=known_context)

        def handle_pre_llm_call(self, *args: Any, **kwargs: Any) -> dict:
            """Existing host hook: deliver shared association before memory prefetch."""
            return run_pre_llm_call(self, *args, **kwargs)

        def recall_status(self):
            """Host recall indicator: only a delivered ``ok`` association counts."""
            state = self._association_state.current_recall()
            if not state or state.get("status") != "ok" or not state.get("items"):
                return None
            return RecallStatus("sanae_mem", len(state["items"]))

        def add_group_event(self, text: str, *, importance: object = 0.5, ttl_days: object = None,
                            source: str = "explicit") -> int | None:
            return self._retrieval.add_group_event(
                text, importance=importance, ttl_days=ttl_days, source=source,
            )

        def sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            *,
            session_id: str = "",
            messages: Optional[List[Dict[str, Any]]] = None,
            turn_author: Optional[Dict[str, Any]] = None,
        ) -> None:
            sync_provider_turn(
                self, user_content, assistant_content,
                session_id=session_id, messages=messages, turn_author=turn_author,
            )

        def submit_free_think_heartbeat(
            self, data_dir: Path, profile_key: str, job: Callable[[], None],
        ) -> bool:
            return _submit(self._background_tasks, (data_dir, profile_key, "cron"), job)

        def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
            try:
                result = self._tool_router.dispatch(
                    tool_name, args, session_id=kwargs.get("session_id", ""),
                    turn_id=kwargs.get("turn_id", ""), synthetic=kwargs.get("synthetic", False),
                )
                return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

        def get_tool_schemas(self) -> List[Dict[str, Any]]:
            return self._tool_router.tool_schemas()

        def get_config_schema(self) -> List[Dict[str, Any]]:
            return CONFIG_SCHEMA

        def run_nightly_journal(
            self, *, now: datetime | None = None, owner_key: str | None = None,
            force: bool = False,
        ) -> dict:
            return run_provider_nightly_journal(self, now=now, owner_key=owner_key, force=force)

        def run_installed_nightly_journal(self, *, now: datetime | None = None) -> dict:
            return run_installed_nightly_journal(self, now=now)
