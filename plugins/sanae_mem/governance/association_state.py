from __future__ import annotations

import threading
from dataclasses import dataclass
from time import monotonic
from typing import Callable

from .read_gate import HookRecall


OwnerResolver = Callable[[], str | None]


@dataclass(frozen=True)
class _OwnerBinding:
    owner: str = ""
    resolver: OwnerResolver | None = None


@dataclass(frozen=True)
class AssociationHookRequest:
    session_id: str
    binding_session_id: str
    query: str
    revision: int


@dataclass
class AssociationCache:
    key: tuple[str, str, str, bool, int | None, str]
    delivered_by_hook: bool = False


@dataclass(eq=False)
class HotAssociation:
    foreground_timed_out: bool = False


class AssociationState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._hot_lock = threading.Lock()
        self._hot_finished = threading.Condition(self.lock)
        self._hot_call: HotAssociation | None = None
        self._hot_handoff: HotAssociation | None = None
        self.revision = 0
        self._hook_request: AssociationHookRequest | None = None
        self._cache: AssociationCache | None = None
        self._last_recall: dict | None = None
        self._read_binding = _OwnerBinding()
        self._write_binding = _OwnerBinding()
        self._read_invalid = False

    def request_current(self, revision: int, hook_request: AssociationHookRequest | None) -> bool:
        with self.lock:
            return revision == self.revision and (
                hook_request is None or self.hook_active(hook_request)
            )

    def try_start_hot(
        self, call: HotAssociation, *, revision: int, hook_request: AssociationHookRequest | None,
        deadline: float, handoff_deadline: float | None,
    ) -> bool:
        with self._hot_finished:
            if not self.request_current(revision, hook_request):
                return False
            previous = self._hot_call
            if self._hot_lock.locked():
                if handoff_deadline is None or self._hot_handoff is None:
                    return False
                self._hot_handoff = None
                finished = self._hot_finished.wait_for(
                    lambda: (
                        not self.request_current(revision, hook_request)
                        or not self._hot_lock.locked()
                        or self._hot_call is not previous
                    ),
                    timeout=max(0.0, handoff_deadline - monotonic()),
                )
                if not finished or self._hot_call is not previous:
                    return False
                if not self.request_current(revision, hook_request):
                    return False
            if monotonic() >= deadline or not self._hot_lock.acquire(blocking=False):
                return False
            self._hot_call = call
            return True

    def finish_hot_foreground(self, call: HotAssociation) -> None:
        with self.lock:
            if call.foreground_timed_out and self._hot_call is call and self._hot_lock.locked():
                self._hot_handoff = call

    def finish_hot_worker(self) -> None:
        with self._hot_finished:
            self._hot_lock.release()
            self._hot_handoff = None
            self._hot_finished.notify_all()

    def reset(self) -> None:
        with self.lock:
            self._read_binding = _OwnerBinding()
            self._read_invalid = False
            self.switch_session()

    def switch_session(self) -> None:
        with self.lock:
            self.revision += 1
            self._cache = None
            self._last_recall = None
            self._hook_request = None
            self._write_binding = _OwnerBinding()
            self._hot_finished.notify_all()

    def bind_read_owner(self, owner_key: str, resolver: OwnerResolver | None) -> None:
        with self.lock:
            self.revision += 1
            self._read_binding = _OwnerBinding(str(owner_key or ""), resolver)
            self._read_invalid = False
            self._cache = None
            self._hot_finished.notify_all()

    def bind_write_owner(self, owner_key: str, resolver: OwnerResolver | None) -> None:
        with self.lock:
            self._write_binding = _OwnerBinding(str(owner_key or ""), resolver)

    def write_owner(self, default_owner: str) -> str:
        with self.lock:
            binding = self._write_binding
        if binding.resolver is None:
            return binding.owner or default_owner
        owner = str(binding.resolver() or "")
        with self.lock:
            if binding is not self._write_binding or owner != binding.owner:
                raise PermissionError("memory identity changed during the action")
        return owner

    def read_owner(self, default_owner: str) -> str:
        with self.lock:
            if self._read_invalid:
                raise PermissionError("memory identity changed during the action")
            binding = self._read_binding
            revision = self.revision
        if binding.resolver is None:
            return binding.owner or default_owner
        try:
            current = str(binding.resolver() or "")
        except Exception as exc:
            with self.lock:
                if revision == self.revision:
                    self._invalidate_read()
            raise PermissionError("memory identity could not be verified") from exc
        with self.lock:
            if revision != self.revision:
                raise PermissionError("memory identity changed during the action")
            if current != binding.owner:
                self._invalidate_read()
                raise PermissionError("memory identity changed during the action")
        return current

    def _invalidate_read(self) -> None:
        with self._hot_finished:
            self._read_invalid = True
            self.revision += 1
            self._cache = None
            self._last_recall = None
            self._hot_finished.notify_all()

    def validate_read(self, default_owner: str) -> bool:
        try:
            self.read_owner(default_owner)
            return True
        except PermissionError:
            return False

    def hook_active(
        self,
        request: AssociationHookRequest,
        *,
        binding_session_id: str | None = None,
    ) -> bool:
        with self.lock:
            if binding_session_id is not None and request.binding_session_id != binding_session_id:
                return False
            return self._hook_request is request and request.revision == self.revision

    def begin_hook_recall(
        self,
        session_id: str,
        binding_session_id: str,
        query: str,
    ) -> AssociationHookRequest:
        with self.lock:
            request = AssociationHookRequest(session_id, binding_session_id, query, self.revision)
            self._hook_request = request
            return request

    def clear_hook_request(self) -> None:
        with self.lock:
            self._hook_request = None

    def store_recall(
        self,
        *,
        key: tuple[str, str, str, bool, int | None, str],
        result: dict,
        revision: int,
        hook_request: AssociationHookRequest | None,
        hot_call: HotAssociation | None = None,
    ) -> bool:
        with self.lock:
            current = self.request_current(revision, hook_request)
            if current:
                self._last_recall = result
                self._cache = AssociationCache(key)
            if hot_call is not None:
                self.finish_hot_foreground(hot_call)
            return current

    def mark_hook_delivered(
        self,
        request: AssociationHookRequest,
        result: dict,
        *,
        binding_session_id: str | None = None,
        on_deliver: Callable[[], None] | None = None,
    ) -> bool:
        with self.lock:
            if not self.hook_active(request, binding_session_id=binding_session_id):
                return False
            if self._cache is not None and self._last_recall is result:
                self._cache.delivered_by_hook = True
            if on_deliver is not None:
                on_deliver()
            return True

    def is_hook_delivered(self) -> bool:
        with self.lock:
            return bool(self._cache is not None and self._cache.delivered_by_hook)

    def _cancel_pending_hook_locked(self, query: str, session_id: str) -> bool:
        request = self._hook_request
        if request is None:
            return False
        self._hook_request = None
        return request.session_id == session_id and request.query == query

    def consume_prefetch_hook_recall(
        self,
        query: str,
        session_id: str,
        public: bool,
        *,
        validate_read: Callable[[], bool],
    ) -> HookRecall:
        with self.lock:
            if not public and self._cancel_pending_hook_locked(query, session_id):
                return HookRecall.CANCELLED

            cache = self._cache
            revision = self.revision
            if cache is None:
                return HookRecall.MISS

        valid = validate_read()

        with self.lock:
            if not valid:
                if self._cache is cache:
                    self._cache = None
                    self._last_recall = None
                return HookRecall.REVOKED
            if self.revision != revision:
                return HookRecall.REVOKED
            if self._cache is not cache:
                return HookRecall.MISS
            if not public and self._cancel_pending_hook_locked(query, session_id):
                return HookRecall.CANCELLED
            result = self._last_recall
            if not public and cache.delivered_by_hook and result is not None:
                if cache.key[:2] == (query, str(session_id)):
                    status = result["status"]
                    if status == "empty":
                        return HookRecall.EMPTY
                    if status == "unavailable":
                        return HookRecall.UNAVAILABLE
                    return HookRecall.DELIVERED
            return HookRecall.MISS

    def current_recall(self) -> dict | None:
        with self.lock:
            return self._last_recall

    def current_recall_snapshot(self) -> tuple[AssociationCache | None, dict | None]:
        with self.lock:
            return self._cache, self._last_recall
