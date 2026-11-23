from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Protocol

from .infra.host_bridge import get_hermes_home
def _always_enabled() -> bool:
    return True

from .infra.host_bridge import is_delegated_child_turn
from .schemas import VIDEO_DETAIL_SCHEMA
from .bilibili_history import BILIBILI_PLUGIN as _BILIBILI_PLUGIN, record_bilibili_reply
from .governance.source_context import current_chat_scope


class _SqliteOpen(Protocol):
    @property
    def is_open(self) -> bool: ...


class _TrustedOwnerProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def sqlite(self) -> _SqliteOpen: ...

    def shadow_enabled(self) -> bool: ...

    def integration_owner(
        self, *, home: Path, allow_cron: bool, write: bool, cron_active: bool,
    ) -> str | None: ...


def _resolve_trusted_owner(
    provider: _TrustedOwnerProvider | None,
    *,
    home: Path,
    cron: object,
    current_cron: object,
    allow_cron: bool,
    write: bool,
) -> str | None:
    if (
        provider is None
        or provider.name != "sanae_mem"
        or provider.shadow_enabled()
        or not provider.sqlite.is_open
        or is_delegated_child_turn()
        or get_hermes_home().resolve() != home
    ):
        return None
    cron_active = cron is not None and current_cron == cron
    return provider.integration_owner(
        home=home,
        allow_cron=allow_cron,
        write=write,
        cron_active=cron_active,
    )


class _BilibiliMemory:
    def __init__(self, ctx, provider, home, registration):
        self._ctx = ctx
        self._provider = provider
        self._home = home
        self._registration = registration
        self._cron = ctx.current_cron_execution()

    def _owner(self, *, allow_cron=False, write=False):
        if not self._registration.active or not self._ctx.has_plugin(_BILIBILI_PLUGIN):
            return None
        return _resolve_trusted_owner(
            self._provider,
            home=self._home,
            cron=self._cron,
            current_cron=self._ctx.current_cron_execution(),
            allow_cron=allow_cron,
            write=write,
        )

    def record_video_watch(self, observation, *, delivered, public, platform="bilibili",
                           source_object_id=None, cid=None, _watch_receipt=None,
                           reused_video_watch=None, review_public_excerpt=None) -> bool:
        receipt = _watch_receipt
        if receipt is None and isinstance(observation, dict):
            receipt = observation.get("_watch_receipt")
        if (delivered is not True or public is not True
                or not isinstance(receipt, str)
                or not re.fullmatch(r"[0-9a-f]{32}", receipt)):
            return False
        owner = self._owner(allow_cron=True, write=True)
        if owner is None:
            return False
        canonical = dict(observation) if isinstance(observation, dict) else observation
        if isinstance(canonical, dict):
            canonical.update({"platform": platform,
                              "source_object_id": (source_object_id or canonical.get("source_object_id")
                                                   or canonical.get("bvid")),
                              "cid": cid if cid is not None else canonical.get("cid")})
            canonical["_watch_receipt"] = receipt
            if reused_video_watch is not None:
                canonical["reused_video_watch"] = reused_video_watch
        return self._provider.video_memory.record_video_watch(
            canonical, owner_key=owner, public_observation=True,
            review_public_excerpt=review_public_excerpt,
        )

    def get_public_video_watch(self, bvid: str) -> str:
        owner = self._owner(allow_cron=True)
        return self._provider.video_memory.get_public_video_watch(bvid, owner_key=owner) if owner is not None else ""

    def lookup_video_watch(self, platform, source_object_id, *, cid=None) -> dict | None:
        owner = self._owner(allow_cron=True)
        if owner is None:
            return None
        return self._provider.video_memory.lookup_video_watch(
            platform, source_object_id, owner_key=owner, cid=cid,
        )

    def known_meme_terms(self, parts: list[str]) -> list[str]:
        return self._provider.known_meme_terms(parts) if self._owner(allow_cron=True) is not None else []

    def public_meme_senses(self, parts: list[str]) -> list[dict]:
        if not self._owner(allow_cron=True):
            return []
        return self._provider.public_meme_senses(parts)

    def meme_context_for_public_material(self, parts: list[str], *, public_only: bool = False) -> str:
        if not self._owner(allow_cron=public_only):
            return ""
        return self._provider.meme_context_for_public_material(
            list(parts), public_only=public_only,
        )

    def record_accepted_reply(self, *, receipt_id, at, text, visibility, accepted, test_reply=False) -> bool:
        owner = self._owner(allow_cron=True, write=True)
        if (not owner or accepted is not True or test_reply
                or not isinstance(receipt_id, str) or not receipt_id.strip()
                or type(at) not in (int, float) or not math.isfinite(at) or at <= 0
                or not isinstance(text, str) or not text.strip()
                or visibility not in ("private", "public")):
            return False
        with self._provider.sqlite.transaction() as conn:
            return record_bilibili_reply(
                conn, owner=owner, receipt_id=receipt_id, at=at,
                text=text, visibility=visibility,
            )

    def ingest_source_material(self, payload, *, create_memory_event: bool = False, importance: float | None = None) -> dict:
        owner = self._owner(allow_cron=True, write=True)
        if owner is None:
            return {"ok": False, "reason": "untrusted_intake_route"}
        from .governance.source_intake import ingest_source_material as _ingest_material
        verdict = _ingest_material(
            self._provider,
            payload,
            create_memory_event=create_memory_event,
            importance=importance,
            allow_configured_owner_fallback=True,
            authorized_source_plugin=_BILIBILI_PLUGIN,
        )
        return verdict.to_dict()


class _SourceVideoMemory:
    def __init__(self, ctx, provider):
        self._ctx = ctx
        self._provider = provider
        self._plugin_id = str(getattr(ctx, "plugin_id", "") or "").strip()
        self._home = get_hermes_home().resolve()
        current_cron = getattr(ctx, "current_cron_execution", None)
        self._cron = current_cron() if callable(current_cron) else None

    def _owner(self, *, write=False):
        if not self._plugin_id:
            return None
        if callable(getattr(self._ctx, "has_plugin", None)) and not self._ctx.has_plugin(self._plugin_id):
            return None
        current_cron = self._ctx.current_cron_execution() if self._cron is not None else None
        return _resolve_trusted_owner(
            self._provider,
            home=self._home,
            cron=self._cron,
            current_cron=current_cron,
            allow_cron=True,
            write=write,
        )

    def lookup_video_watch(self, platform, source_object_id, *, cid=None):
        owner = self._owner()
        if not owner:
            return None
        return self._provider.video_memory.lookup_video_watch(
            platform, source_object_id, owner_key=owner, cid=cid,
        )

    def record_video_watch(self, observation, *, platform, source_object_id, cid,
                           settled, _watch_receipt, reused_video_watch=None,
                           review_public_excerpt=None):
        owner = self._owner(write=True)
        receipt = _watch_receipt
        if receipt is None and isinstance(observation, dict):
            receipt = observation.get("_watch_receipt")
        if (not owner or settled is not True or not isinstance(receipt, str)
                or not re.fullmatch(r"[0-9a-f]{32}", receipt)):
            return False
        canonical = dict(observation) if isinstance(observation, dict) else observation
        if not isinstance(canonical, dict):
            return False
        canonical.update({"platform": platform, "source_object_id": source_object_id,
                          "cid": cid, "_watch_receipt": receipt})
        if reused_video_watch is not None:
            canonical["reused_video_watch"] = reused_video_watch
        return self._provider.video_memory.record_video_watch(
            canonical, owner_key=owner,
            public_observation=canonical.get("privacy_scope") == "public" or canonical.get("public") is True,
            review_public_excerpt=review_public_excerpt,
        )

    def ingest_source_material(self, payload, *, create_memory_event=False, importance=None):
        owner = self._owner(write=True)
        if not owner:
            return {"ok": False, "reason": "untrusted_intake_route"}
        from .governance.source_intake import ingest_source_material
        return ingest_source_material(
            self._provider, payload, create_memory_event=create_memory_event,
            importance=importance, allow_configured_owner_fallback=True,
            authorized_source_plugin=self._plugin_id,
        ).to_dict()


def register_video_memory(ctx, resolve_provider=None):
    """Bind neutral video lookup/settlement to a real source plugin context."""
    plugin_id = str(getattr(ctx, "plugin_id", "") or "").strip()
    if not plugin_id or plugin_id == "sanae_mem":
        raise PermissionError("video memory requires a source plugin context")
    home = get_hermes_home().resolve()
    if resolve_provider is None:
        from sanae_mem.governance.runtime_registry import resolve_active_provider
        resolve_provider = lambda session_id: resolve_active_provider(home, session_id)

    def bind(provider):
        return _SourceVideoMemory(ctx, provider)

    def for_session(session_id):
        if not isinstance(session_id, str) or not session_id:
            return None
        provider = resolve_provider(session_id)
        return bind(provider) if provider is not None else None

    bind.for_session = for_session

    return bind


def register_bilibili_memory(ctx, resolve_provider=None):
    """Return a binder for the adapter's active memory provider.

    Call from the adapter's ``register(ctx)``. The returned callable keeps the
    existing ``bind(provider)`` contract for platform operations. Its
    ``for_session(session_id)`` method finds the active provider in this profile
    unless an explicit resolver was supplied. The client checks the live route
    before each operation.
    """
    if ctx.plugin_id != _BILIBILI_PLUGIN:
        raise PermissionError("Bilibili video detail requires its own plugin context")
    home = get_hermes_home().resolve()
    if resolve_provider is None:
        from sanae_mem.governance.runtime_registry import resolve_active_provider
        resolve_provider = lambda session_id: resolve_active_provider(home, session_id)
    registration = None

    def bind(provider):
        return _BilibiliMemory(ctx, provider, home, registration)

    def for_session(session_id):
        if not isinstance(session_id, str) or not session_id:
            return None
        provider = resolve_provider(session_id)
        return bind(provider) if provider is not None else None

    bind.for_session = for_session

    def available():
        scope = current_chat_scope()
        return bool(registration and registration.active and ctx.has_plugin(_BILIBILI_PLUGIN)
                    and get_hermes_home().resolve() == home and scope.owner and scope.session_id
                    and not is_delegated_child_turn())

    def handle(args, *, session_id=""):
        scope = current_chat_scope()
        if available() and (not session_id or session_id == scope.session_id):
            client = for_session(scope.session_id)
            owner = client._owner() if client is not None else None
            if client is not None and owner is not None:
                return json.dumps(
                    {"ok": True, "result": client._provider.video_memory.tool_video_detail(args, owner_key=owner)},
                    ensure_ascii=False,
                )
        return json.dumps({"ok": False, "error": "video detail requires a verified private session"})

    registration = ctx.register_tool(name=VIDEO_DETAIL_SCHEMA["name"], toolset=_BILIBILI_PLUGIN,
                                     schema=VIDEO_DETAIL_SCHEMA, handler=handle, check_fn=_always_enabled)
    if registration is None:
        raise RuntimeError("Bilibili video detail registration failed")
    return bind
