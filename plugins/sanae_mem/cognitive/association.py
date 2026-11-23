from __future__ import annotations

import logging
import re
import sqlite3
from concurrent.futures import Future, TimeoutError, wait
from dataclasses import dataclass
from functools import partial
from time import monotonic
from typing import Any, Callable, List, Optional, Protocol, Tuple

from ..infra.host_bridge import spawn_context_thread

from . import association_history as _history
from . import association_recall as _recall
from ..governance.association_state import AssociationHookRequest, AssociationState, HotAssociation
from ..governance.source_context import in_public_interaction, public_source_view
from ..infra.textutil import _tokens
from ..infra.words import _EXPLICIT_INJECT_MIN_SEMANTIC
from ..storage.sqlite_facade import SqliteFacade


logger = logging.getLogger(__name__)
_HOT_ASSOCIATION_SECONDS = 3.0
_HOT_FALLBACK_RESERVE_SECONDS = 0.65
_HOT_HANDOFF_SECONDS = 0.75


class AssociationLlm(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], *, task: str, temperature: float,
        max_tokens: int, timeout: float, purpose: str | None = None,
    ) -> object: ...


@dataclass(frozen=True)
class AssociationTurn:
    owner_key: str
    daily_owner_key: str
    daily_profile_key: str
    daily_owner_platform: str
    agent_context: str
    inject_max_chars: int
    llm: AssociationLlm | None


@dataclass(frozen=True)
class AssociationDependencies:
    search: _recall.AssociationSearch
    public_search: Callable[[dict[str, Any]], list[dict]]
    daily_recall: _recall.DailyRecallReader
    read_owner: Callable[[], str]
    read_owner_keys: Callable[[str], list[str]]
    shadow_enabled: Callable[[], bool]
    relative_date: Callable[[str], str]
    triggers_explicit_past: Callable[[str], bool]
    is_trivial_message: Callable[[str], bool]
    is_busy_context: Callable[[str], bool]


class AssociationRead(Protocol):
    def __call__(self, *, allow_model: bool = True, deadline: float | None = None) -> dict: ...


class Association:
    def __init__(
        self, sqlite: SqliteFacade, state: AssociationState, dependencies: AssociationDependencies,
    ) -> None:
        self._sqlite = sqlite
        self._state = state
        self._dependencies = dependencies

    def _association_need_decision(self, query: str,
                                   conversation_history: Optional[List[Any]],
                                   *, llm: AssociationLlm | None, purpose: str = "chat",
                                   deadline: Optional[float] = None) -> dict:
        return _history.need_decision(
            query, conversation_history, purpose=purpose, llm=llm,
            deadline=deadline, unavailable=self._association_unavailable,
        )

    def associate_context(
        self,
        query: str,
        *,
        conversation_history: Optional[List[Any]] = None,
        purpose: str = "chat",
        session_id: str = "",
        max_chars: Optional[int] = None,
        gap_continuation: bool = False,
        hot_path: bool = False,
        turn: AssociationTurn,
        revision: int,
        hook_request: AssociationHookRequest | None = None,
    ) -> dict:
        history_digest = ""
        if _history.aux_uses_history(query, conversation_history, purpose):
            history_digest = _history.history_digest(conversation_history)
        key = (str(query or ""), str(session_id or ""), str(purpose or "chat"),
               bool(gap_continuation), max_chars, history_digest)
        read = partial(
            self._associate_context, query, turn=turn,
            conversation_history=conversation_history, purpose=purpose,
            max_chars=max_chars, gap_continuation=gap_continuation,
        )
        hot_call = HotAssociation() if hot_path and turn.llm is not None else None
        try:
            result = (
                self._associate_on_hot_path(
                    read, purpose=purpose, hot_call=hot_call, revision=revision, hook_request=hook_request,
                )
                if hot_path else read()
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("sanae_mem: association failed (%s)", type(exc).__name__)
            result = self._association_unavailable()
        if str(purpose or "chat") in {"proactive", "delivery_check"} \
                and result.get("status") == "empty":
            result["valid"] = True
        stored = self._state.store_recall(
            key=key,
            result=result,
            revision=revision,
            hook_request=hook_request,
            hot_call=hot_call,
        )
        return result if stored else self._association_unavailable()

    def _associate_on_hot_path(
        self, recall: AssociationRead, *, purpose: str, hot_call: HotAssociation | None,
        revision: int, hook_request: AssociationHookRequest | None,
    ) -> dict:
        if not self._sqlite.is_open:
            return self._association_unavailable()
        started = monotonic()
        deadline = started + _HOT_ASSOCIATION_SECONDS
        proactive = str(purpose or "chat") in {"proactive", "delivery_check"}
        state = self._state
        if hot_call is not None and state.try_start_hot(
            hot_call, revision=revision, hook_request=hook_request, deadline=deadline,
            handoff_deadline=started + _HOT_HANDOFF_SECONDS if proactive else None,
        ):
            future: Future[dict] = Future()

            def read() -> None:
                try:
                    if (
                        monotonic() < deadline and self._sqlite.is_open
                        and state.request_current(revision, hook_request)
                    ):
                        with self._sqlite.thread_readonly(deadline=deadline):
                            result = recall(deadline=deadline)
                    else:
                        result = self._association_unavailable()
                except Exception as exc:
                    if monotonic() >= deadline:
                        logger.warning("sanae_mem: timed-out association worker failed (%s)", type(exc).__name__)
                    future.set_exception(exc)
                else:
                    future.set_result(result)
                finally:
                    state.finish_hot_worker()

            try:
                spawn_context_thread(read, name="sanae-association").start()
            except Exception:
                state.finish_hot_worker()
                raise
            try:
                reserve = 0.05 if proactive else _HOT_FALLBACK_RESERVE_SECONDS
                wait_timeout = max(0.0, deadline - reserve - monotonic())
                if wait((future,), timeout=wait_timeout).done:
                    result = future.result()
                    if result.get("status") != "unavailable":
                        return result
                else:
                    hot_call.foreground_timed_out = True
            except TimeoutError:
                pass
        if not state.request_current(revision, hook_request):
            return self._association_unavailable()
        try:
            with self._sqlite.thread_readonly(deadline=deadline):
                return recall(allow_model=hot_call is None, deadline=deadline)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower() and monotonic() >= deadline:
                logger.warning("sanae_mem: hot-path association fallback exceeded deadline")
                return self._association_unavailable()
            raise

    @staticmethod
    def _association_empty() -> dict:
        return {"status": "empty", "context": "", "items": []}

    @staticmethod
    def _association_unavailable() -> dict:
        return {"status": "unavailable", "context": "", "items": []}

    def _associate_context(
        self, query: str, *, turn: AssociationTurn,
        conversation_history: Optional[List[Any]], purpose: str, max_chars: Optional[int],
        gap_continuation: bool,
        allow_model: bool = True, deadline: Optional[float] = None,
    ) -> dict:
        query = str(query or "").strip()
        purpose = str(purpose or "chat").strip() or "chat"
        if not query:
            return self._association_empty()
        if self._dependencies.shadow_enabled():
            return self._association_empty()
        if not self._sqlite.is_open:
            return self._association_unavailable()

        public = in_public_interaction() or purpose == "public"
        if purpose == "public" and not in_public_interaction():
            return self._association_unavailable()

        if public:
            return self._associate_public(query, turn=turn, max_chars=max_chars,
                                          allow_model=allow_model, deadline=deadline)

        explicit = bool(self._dependencies.triggers_explicit_past(query))
        if purpose in {"chat", "action"} and not explicit:
            if self._dependencies.is_busy_context(query):
                return self._association_empty()
            if self._dependencies.is_trivial_message(query):
                allow_short = bool(conversation_history) \
                    and not _history.exact_trivial(query)
                if not allow_short:
                    return self._association_empty()

        proactive = purpose in {"proactive", "delivery_check"}
        has_history = bool(_history.history_messages(conversation_history, limit=1))
        items: List[dict] = []
        if explicit and not proactive:
            items = self._association_retrieve(
                _history.no_llm_query(query, conversation_history, True),
                turn=turn, gap_continuation=False, proactive=False, explicit_history=True)
            if items and allow_model and turn.llm is not None:
                reviewed, unavailable, _valid = self._association_aux_review(
                    items, query, conversation_history, llm=turn.llm, public=False, purpose="chat", deadline=deadline)
                if unavailable:
                    items = _recall.conservative(items)
                else:
                    rejected = {item["id"] for item in items} - {
                        item["id"] for item in reviewed}
                    items = reviewed
                    if not items:
                        decision = self._association_need_decision(
                            query, conversation_history, llm=turn.llm, purpose=purpose, deadline=deadline)
                        if decision.get("status") == "ok":
                            planned = str(decision.get("query") or "").strip()
                            if _history.plan_query_ok(planned):
                                items = self._association_retrieve(
                                    planned, turn=turn, gap_continuation=False, proactive=False,
                                    exclude_ids=rejected, explicit_history=True)
            elif not items and allow_model and turn.llm is not None:
                decision = self._association_need_decision(
                    query, conversation_history, llm=turn.llm, purpose=purpose, deadline=deadline)
                if decision.get("status") == "ok":
                    planned = str(decision.get("query") or "").strip()
                    if _history.plan_query_ok(planned):
                        items = self._association_retrieve(
                            planned, turn=turn, gap_continuation=False, proactive=False, explicit_history=True)
        else:
            search_query = query
            if purpose == "chat":
                if allow_model and turn.llm is not None:
                    decision = self._association_need_decision(
                        query, conversation_history, llm=turn.llm, purpose="chat", deadline=deadline)
                    if decision.get("status") != "ok":
                        return self._association_unavailable()
                    if not decision.get("needs"):
                        return self._association_empty()
                    planned = str(decision.get("query") or "").strip()
                    if not _history.plan_query_ok(planned):
                        return self._association_unavailable()
                    search_query = planned
                elif turn.llm is not None and not gap_continuation:
                    return self._association_unavailable()
                else:
                    search_query = _history.no_llm_query(
                        query, conversation_history, False)
            elif purpose in {"action", "proactive"} and allow_model and turn.llm is not None and has_history:
                decision = self._association_need_decision(
                    query, conversation_history, llm=turn.llm, purpose=purpose, deadline=deadline)
                if decision.get("status") != "ok":
                    return self._association_unavailable()
                if not decision.get("needs"):
                    return self._association_empty()
                planned = str(decision.get("query") or "").strip()
                if not _history.plan_query_ok(planned):
                    return self._association_unavailable()
                search_query = planned
            items = self._association_retrieve(
                search_query, turn=turn, gap_continuation=gap_continuation, proactive=proactive)

        if not items:
            return self._association_empty()

        if proactive:
            if not allow_model or turn.llm is None:
                return self._association_unavailable()
            items, unavailable, valid = self._association_aux_review(
                items, query, conversation_history, llm=turn.llm, public=False, purpose=purpose,
                deadline=deadline)
            if unavailable:
                return self._association_unavailable()
            if valid is False:
                return {"status": "ok", "context": "", "items": items, "valid": False}
            if not items:
                return self._association_empty()

        context, selected = self._render_association(
            items, turn=turn, max_chars=max_chars, gap=gap_continuation, public=False)
        if not selected or not context:
            return self._association_empty()
        result = {"status": "ok", "context": context, "items": selected}
        if proactive:
            result["valid"] = True
        return result

    def _associate_public(self, query: str, *, turn: AssociationTurn, max_chars: Optional[int],
                          allow_model: bool = True,
                          deadline: Optional[float] = None) -> dict:
        query = str(query or "").strip()
        if not query or self._dependencies.is_trivial_message(query) or self._dependencies.is_busy_context(query):
            return self._association_empty()
        try:
            rows = self._dependencies.public_search({"query": query, "top_k": 10})
        except (OSError, sqlite3.Error) as exc:
            logger.warning("sanae_mem: public association failed (%s)", type(exc).__name__)
            return self._association_unavailable()
        if not rows:
            return self._association_empty()

        query_tokens = set(_tokens(query))
        query_bvids = set(re.findall(r"BV[0-9A-Za-z]{10}", query))
        items: List[dict] = []
        for row in rows:
            excerpt = str(row.get("excerpt") or "")
            if not excerpt:
                continue
            raw_source = row.get("source")
            source = raw_source if isinstance(raw_source, dict) else public_source_view(None)
            ref_bvids = set(re.findall(r"BV[0-9A-Za-z]{10}", str(source.get("ref") or "")))
            exact_video = bool(query_bvids & ref_bvids)
            overlap = len(query_tokens & set(_tokens(excerpt)))
            semantic = float(row.get("_semantic") or 0.0)
            real = bool(row.get("_embedding_real"))
            if not exact_video and overlap < 2 \
                    and not (real and semantic >= _EXPLICIT_INJECT_MIN_SEMANTIC):
                continue
            relationship = "same_event" if exact_video else "related"
            items.append({
                "id": int(row["id"]),
                "time": str(row.get("created_at") or ""),
                "source": source,
                "relationship": relationship,
                "state": "unknown",
                "text": excerpt,
                "occurrences": [],
                "asked_at": "",
                "visibility": "public",
                "verified_same_event": bool(exact_video),
            })
            if len(items) >= 3:
                break
        if not items:
            return self._association_empty()
        if allow_model and turn.llm is not None and any(not item.get("verified_same_event") for item in items):
            items, unavailable, _valid = self._association_aux_review(
                items, query, None, llm=turn.llm, public=True, purpose="public", deadline=deadline)
            if unavailable:
                return self._association_unavailable()
        elif not allow_model:
            items = [item for item in items if item.get("verified_same_event")]
        elif turn.llm is None:
            items = _recall.conservative(items)
        if not items:
            return self._association_empty()
        context, selected = self._render_association(
            items, turn=turn, max_chars=max_chars, gap=False, public=True)
        if not selected or not context:
            return self._association_empty()
        return {"status": "ok", "context": context, "items": selected}

    def _association_retrieve(
        self, search_query: str, *, turn: AssociationTurn, gap_continuation: bool, proactive: bool,
        exclude_ids: Optional[set] = None, explicit_history: bool = False,
    ) -> List[dict]:
        return _recall.retrieve(
            search_query, gap_continuation=gap_continuation, proactive=proactive,
            exclude_ids=exclude_ids, explicit_history=explicit_history,
            search=self._dependencies.search,
            list_daily_entries_for_recall=self._dependencies.daily_recall,
            association_read_owner=self._dependencies.read_owner,
            daily_owner_key=turn.daily_owner_key,
            daily_profile_key=turn.daily_profile_key,
            daily_owner_platform=turn.daily_owner_platform,
            agent_context=turn.agent_context,
        )

    def _association_aux_review(
        self, items: List[dict], query: str,
        conversation_history: Optional[List[Any]], *, llm: AssociationLlm | None, public: bool,
        purpose: str = "chat", deadline: Optional[float] = None,
    ) -> Tuple[List[dict], bool, Optional[bool]]:
        return _recall.aux_review(
            items, query, conversation_history, public=public, purpose=purpose,
            deadline=deadline, llm=llm,
        )

    def _render_association(
        self, items: List[dict], *, turn: AssociationTurn, max_chars: Optional[int], gap: bool, public: bool,
    ) -> Tuple[str, List[dict]]:
        return _recall.render_association(
            items, max_chars=max_chars, gap=gap, public=public,
            inject_max_chars=turn.inject_max_chars,
            relative_date=self._dependencies.relative_date,
            association_read_owner=self._dependencies.read_owner,
            owner_key=turn.owner_key,
            read_owners=tuple(self._dependencies.read_owner_keys(turn.owner_key)),
        )
