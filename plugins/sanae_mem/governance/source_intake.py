from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, NamedTuple, Protocol, cast

if TYPE_CHECKING:
    from ..storage.sqlite_facade import SqliteFacade

from sanae_mem.governance.runtime_registry import (
    LegacySourceIntakeSink,
    register_explicit_source_sink,
    resolve_active_provider,
    resolve_source_intake_sink,
    unregister_explicit_source_sink,
)

from ..infra.atomic_io import file_lock
from ..infra.contracts import EMPTY_MAPPING, empty_mapping, freeze_mapping, thaw_json
from ..infra.host_bridge import get_hermes_home, hermes_home_scope

from ..infra.host_bridge import (
    current_cron_binding,
    current_turn_envelope,
    is_test_isolated_turn,
    verified_cron_envelope,
)
from ..infra.source_descriptors import source_descriptor, web_locator
from .source_context import current_chat_scope, self_attribution
from ..infra.textutil import _bool, _load_plugin_config, hermes_home

logger = logging.getLogger(__name__)

class _CrossProcessIntakeLock:
    def __init__(self, lock_file: Path | None) -> None:
        self._thread_lock = threading.RLock()
        self._lock_file = lock_file
        self._depth = 0
        self._file_cm: contextlib.AbstractContextManager[Any] | None = None

    def __enter__(self) -> _CrossProcessIntakeLock:
        self._thread_lock.acquire()
        if self._depth == 0 and self._lock_file is not None:
            cm = file_lock(self._lock_file)
            try:
                cm.__enter__()
            except BaseException:
                self._thread_lock.release()
                raise
            self._file_cm = cm
        self._depth += 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._depth -= 1
        try:
            if self._depth == 0 and self._file_cm is not None:
                cm = self._file_cm
                self._file_cm = None
                cm.__exit__(exc_type, exc, tb)
        finally:
            self._thread_lock.release()


_INTAKE_LOCKS_GUARD = threading.Lock()
_INTAKE_LOCKS: dict[str, _CrossProcessIntakeLock] = {}


def _intake_lock_key(db_path: Path | str) -> str:
    raw = os.path.abspath(os.fspath(db_path))
    if raw.startswith("\\\\?\\UNC\\"):
        raw = "\\\\" + raw[8:]
    elif raw.startswith("\\\\?\\"):
        raw = raw[4:]
    return os.path.normcase(os.path.normpath(raw))


def _intake_lock_for_db(db_path: Path | str | None) -> contextlib.AbstractContextManager[Any]:
    if db_path is None:
        return contextlib.nullcontext()
    raw_str = os.fspath(db_path)
    if str(raw_str).strip() == ":memory:":
        key = ":memory:"
        lock_file: Path | None = None
    else:
        key = _intake_lock_key(db_path)
        resolved = Path(key)
        lock_file = resolved.parent / f".{resolved.name}.intake.lock"
    with _INTAKE_LOCKS_GUARD:
        lock = _INTAKE_LOCKS.get(key)
        if lock is None:
            lock = _CrossProcessIntakeLock(lock_file)
            _INTAKE_LOCKS[key] = lock
        return lock

VALID_MATERIAL_KINDS = frozenset({"raw", "processed"})
VALID_PRIVACY_SCOPES = frozenset({"private", "restricted", "public"})
MaterialKind = Literal["raw", "processed"]
PrivacyScope = Literal["private", "restricted", "public"]


class CredibilityDecision(NamedTuple):
    credibility: str
    user_md_eligible: bool
    verbatim_eligible: bool

def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_timestamp_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return None
        return str(value)
    text = str(value).strip()
    return text if text else None


@dataclass(frozen=True)
class SourceMaterial:
    source_plugin: str
    platform: str
    material_kind: MaterialKind
    content: str
    privacy_scope: PrivacyScope = "public"
    source_object_id: str = ""
    author: str | None = None
    occurred_at: str | float | None = None
    collected_at: str | float | None = None
    processing_type: str | None = None
    processing_version: str | None = None
    raw_locator: str | None = None
    title: str = ""
    extra_metadata: Mapping[str, Any] = field(default_factory=empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_metadata", freeze_mapping(self.extra_metadata))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceMaterial":
        if not isinstance(payload, Mapping):
            raise TypeError("SourceMaterial payload must be a mapping")
        extra_raw = payload.get("extra_metadata")
        if extra_raw is not None and not isinstance(extra_raw, Mapping):
            raise TypeError("SourceMaterial extra_metadata must be a mapping")
        return cls(
            source_plugin=str(payload.get("source_plugin") or "").strip(),
            platform=str(payload.get("platform") or "").strip().lower(),
            material_kind=cast(MaterialKind, str(payload.get("material_kind") or "").strip().lower()),
            content=str(payload.get("content") or ""),
            privacy_scope=cast(PrivacyScope, str(payload.get("privacy_scope") or "public").strip().lower()),
            source_object_id=str(payload.get("source_object_id") or "").strip(),
            author=_optional_str(payload.get("author")),
            occurred_at=payload.get("occurred_at"),
            collected_at=payload.get("collected_at"),
            processing_type=_optional_str(payload.get("processing_type")),
            processing_version=_optional_str(payload.get("processing_version")),
            raw_locator=_optional_str(payload.get("raw_locator")),
            title=str(payload.get("title") or "").strip(),
            extra_metadata=extra_raw if extra_raw is not None else EMPTY_MAPPING,
        )

    def validate(self) -> tuple[bool, str]:
        if not self.source_plugin or not str(self.source_plugin).strip():
            return False, "missing_source_plugin"
        if not self.platform or not str(self.platform).strip():
            return False, "missing_platform"
        if self.material_kind not in VALID_MATERIAL_KINDS:
            return False, "invalid_material_kind"
        if self.privacy_scope not in VALID_PRIVACY_SCOPES:
            return False, "invalid_privacy_scope"
        if not isinstance(self.content, str) or not self.content.strip():
            return False, "empty_content"
        if self.material_kind == "processed" and not _optional_str(self.processing_type):
            return False, "processed_requires_processing_type"
        return True, ""


@dataclass(frozen=True)
class SourceIntakeVerdict:
    ok: bool
    reason: str = ""
    material_id: int | None = None
    deduplicated: bool = False
    credibility: str = ""
    user_md_eligible: bool = False
    verbatim_eligible: bool = False
    linked_event_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "material_id": self.material_id,
            "deduplicated": self.deduplicated,
            "credibility": self.credibility,
            "user_md_eligible": self.user_md_eligible,
            "verbatim_eligible": self.verbatim_eligible,
            "linked_event_id": self.linked_event_id,
        }


def classify_material_credibility(material: SourceMaterial) -> CredibilityDecision:
    """Return ``(credibility, user_md_eligible, verbatim_eligible)``.

    External source materials NEVER qualify for ``USER.md`` (which requires 2-3
    verified maintainer interactive DM raw utterances). Preliminary ``processed``
    summaries / subtitle slices / digests are also NEVER eligible as verbatim quotes.
    """
    if material.material_kind == "processed":
        return CredibilityDecision("derived_processed", False, False)
    if material.privacy_scope == "public" or material.source_object_id:
        return CredibilityDecision("verified_raw", False, True)
    return CredibilityDecision("unverified_external", False, False)


def _video_meme_evidence(material: SourceMaterial) -> bool:
    if material.processing_type != "video_meme_evidence":
        return False
    descriptor = source_descriptor(material.platform)
    route = descriptor.meme_evidence if descriptor else None
    if (route is None or material.source_plugin != route.plugin
            or material.material_kind != "processed" or material.privacy_scope != "public"
            or not material.title.strip()
            or not route.matches_original(material.source_object_id, material.raw_locator or "")):
        raise ValueError("untrusted_video_meme_route")
    meta = material.extra_metadata or {}
    kind = str(meta.get("occurrence_kind") or "").strip().lower()
    occurrence_quote = str(meta.get("occurrence_quote") or "")
    meaning = str(meta.get("meaning") or "").strip()
    verification_url = str(meta.get("verification_url") or "").strip()
    verification_text = str(meta.get("verification_text") or "")
    verification_quote = str(meta.get("verification_quote") or "")
    occurrence_supported = material.title in occurrence_quote and occurrence_quote in material.content
    valid_url = web_locator(verification_url)
    if material.processing_version == "2":
        definition_supplied = material.title != verification_quote.strip() and len(verification_quote) <= 240
    else:
        definition_supplied = meaning in verification_quote
    verification_supported = bool(valid_url and verification_quote
                                  and material.title in verification_quote
                                  and definition_supplied
                                  and verification_quote in verification_text)
    return bool(kind in route.occurrence_kinds and meaning and occurrence_supported and verification_supported
                and not str(meta.get("uncertainty") or "").strip())


def build_material_source_context(material: SourceMaterial) -> dict[str, str]:
    """Project ``SourceMaterial`` into ViveMind's canonical ``events.source_context`` dict.

    Preserves empty strings when optional fields are absent — never fabricates timestamps,
    authors, or refs.
    """
    credibility, _, _ = classify_material_credibility(material)
    certainty = "derived_processed" if material.material_kind == "processed" else (
        "observed" if credibility == "verified_raw" else "unverified"
    )
    visibility = "public" if material.privacy_scope == "public" else "private"
    return {
        "kind": f"external_{material.material_kind}",
        "author_id": _optional_str(material.author) or "",
        "platform": str(material.platform).strip().lower(),
        "ref": _optional_str(material.raw_locator) or str(material.source_object_id or "").strip(),
        "time": _optional_timestamp_str(material.occurred_at) or "",
        "certainty": certainty,
        "visibility": visibility,
        "source_plugin": str(material.source_plugin).strip(),
        "source_object_id": material.source_object_id.strip(),
        "material_kind": material.material_kind,
        "processing_type": _optional_str(material.processing_type) or "",
        "processing_version": _optional_str(material.processing_version) or "",
    }


@dataclass(frozen=True)
class SourceIntakeContext:
    session_id: str
    agent_context: str
    shadow: bool
    daily_owner_key: str
    daily_owner_platform: str
    daily_profile_key: str
    private_chat_owner: str
    is_configured_private_owner: bool
    open_daily_entry: Callable[..., int | None] | None = None


class SourceEventWriter(Protocol):
    def write_event(
        self,
        raw: str,
        session_id: str = ...,
        *,
        importance: float = ...,
        owner_key: str | None = ...,
        source: str = ...,
        source_evidence: object = ...,
        source_context: Mapping[str, object] | None = ...,
        experience_subject: str = ...,
        fact_subject: str = ...,
        involved_owner_keys: str = ...,
        read_scope: str = ...,
        defer_post_commit: list[Callable[[], None]] | None = ...,
    ) -> int | None: ...


class SourceIntakeProvider(Protocol):
    @property
    def sqlite(self) -> SqliteFacade: ...

    @property
    def ingest(self) -> SourceEventWriter: ...

    def source_intake_context(self) -> SourceIntakeContext: ...


def _resolve_intake_owner(
    provider: SourceIntakeProvider | None,
    *,
    allow_configured_owner_fallback: bool = False,
) -> str:
    if provider is None or not provider.sqlite.is_open:
        return ""
    ctx = provider.source_intake_context()
    if ctx.shadow:
        return ""
    envelope = current_turn_envelope()
    if envelope is not None and (
        getattr(envelope, "is_delegated_child", False)
        or getattr(envelope, "restricted_audience", "")
        or getattr(envelope, "is_test_isolated", False)
    ):
        return ""
    scope = current_chat_scope()
    configured_owner = ctx.daily_owner_key
    configured_platform = ctx.daily_owner_platform.strip().lower()
    owner_matches_scope = bool(
        configured_owner
        and scope.visibility == "private"
        and scope.owner == configured_owner
        and scope.platform == configured_platform
    )
    if ctx.agent_context == "primary" and scope.owner:
        priv = ctx.private_chat_owner
        return (
            priv
            if configured_owner and scope.visibility == "private"
            and priv == configured_owner and ctx.is_configured_private_owner
            else ""
        )
    if ctx.agent_context == "cron":
        cron = verified_cron_envelope()
        if (
            current_cron_binding() is not None
            and cron is not None
            and getattr(cron, "verified", False)
            and not getattr(cron, "cancelled", True)
            and str(getattr(cron, "bound_owner", "") or "") == configured_owner
        ):
            return configured_owner
    if allow_configured_owner_fallback and owner_matches_scope:
        return configured_owner
    return ""


def ingest_source_material(
    provider: SourceIntakeProvider | None,
    material: SourceMaterial | Mapping[str, Any],
    *,
    create_memory_event: bool = False,
    importance: float | None = None,
    allow_configured_owner_fallback: bool = False,
    authorized_source_plugin: str | None = None,
) -> SourceIntakeVerdict:
    """Validate, govern, and persist one ``SourceMaterial`` inside ``provider``."""
    if isinstance(material, Mapping):
        try:
            material = SourceMaterial.from_mapping(material)
        except Exception as exc:
            return SourceIntakeVerdict(False, reason=f"invalid_payload:{type(exc).__name__}")
    elif not isinstance(material, SourceMaterial):
        if callable(getattr(material, "to_dict", None)):
            try:
                material = SourceMaterial.from_mapping(material.to_dict())
            except Exception as exc:
                return SourceIntakeVerdict(False, reason=f"invalid_payload:{type(exc).__name__}")
        elif type(material).__name__ == "SourceMaterial" and hasattr(material, "source_plugin"):
            try:
                material = SourceMaterial.from_mapping({
                    field_name: getattr(material, field_name, None)
                    for field_name in (
                        "source_plugin", "platform", "material_kind", "content",
                        "privacy_scope", "source_object_id", "author", "occurred_at",
                        "collected_at", "processing_type", "processing_version",
                        "raw_locator", "title", "extra_metadata",
                    )
                })
            except Exception as exc:
                return SourceIntakeVerdict(False, reason=f"invalid_payload:{type(exc).__name__}")
        else:
            return SourceIntakeVerdict(False, reason="invalid_material_type")

    valid, err = material.validate()
    if not valid:
        return SourceIntakeVerdict(False, reason=err)

    if provider is None:
        if is_test_isolated_turn(""):
            return SourceIntakeVerdict(False, reason="test_session_isolated")
        return SourceIntakeVerdict(False, reason="provider_unavailable")
    ctx = provider.source_intake_context()
    session_id = ctx.session_id
    if is_test_isolated_turn(session_id):
        return SourceIntakeVerdict(False, reason="test_session_isolated")
    if not provider.sqlite.is_open:
        return SourceIntakeVerdict(False, reason="provider_unavailable")
    if ctx.shadow:
        return SourceIntakeVerdict(False, reason="provider_shadow_mode")

    owner_key = _resolve_intake_owner(
        provider,
        allow_configured_owner_fallback=allow_configured_owner_fallback,
    )
    if not owner_key:
        return SourceIntakeVerdict(False, reason="untrusted_intake_route")

    try:
        confirmed_meme = _video_meme_evidence(material)
    except ValueError as exc:
        return SourceIntakeVerdict(False, reason=str(exc))

    if (authorized_source_plugin is not None and material.processing_type == "video_meme_evidence"
            and material.source_plugin != authorized_source_plugin):
        return SourceIntakeVerdict(False, reason="untrusted_video_meme_route")

    credibility, user_md_eligible, verbatim_eligible = classify_material_credibility(material)
    assert user_md_eligible is False

    clean_content = material.content.strip()
    extras = thaw_json(material.extra_metadata) if material.extra_metadata else {}
    identity = clean_content
    if material.processing_type == "video_meme_evidence":
        create_memory_event = False
        extras["evidence_status"] = "confirmed" if confirmed_meme else "uncertain"
        identity = json.dumps([material.title, clean_content, extras], ensure_ascii=False, sort_keys=True)
    content_sha = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    occurred_at_str = _optional_timestamp_str(material.occurred_at)
    collected_at_str = _optional_timestamp_str(material.collected_at)
    proc_type = _optional_str(material.processing_type) if material.material_kind == "processed" else _optional_str(material.processing_type)
    proc_ver = _optional_str(material.processing_version)
    raw_loc = _optional_str(material.raw_locator)
    author_str = _optional_str(material.author)
    ingested_at = datetime.now().astimezone().isoformat()
    video_key = ""
    if extras.get("delivery_id") and (extras.get("cid") or extras.get("reused_video_watch")):
        video_key = json.dumps({key: extras.get(key) for key in
                                ("cid", "delivery_id", "reused_video_watch")},
                               ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    extra_json = json.dumps(extras, ensure_ascii=False, sort_keys=True)

    linked_event_id: int | None = None
    existing_sql = """SELECT id, linked_event_id FROM source_materials
               WHERE owner_key = ?
                 AND source_plugin = ?
                 AND platform = ?
                 AND source_object_id = ?
                 AND material_kind = ?
                 AND COALESCE(processing_type, '') = ?
                 AND content_sha256 = ?"""
    existing_params = (
        owner_key,
        material.source_plugin.strip(),
        material.platform.strip().lower(),
        material.source_object_id.strip(),
        material.material_kind,
        proc_type or "",
        content_sha,
    )

    def _already_ingested_verdict(conn: sqlite3.Connection, existing_row: Any) -> SourceIntakeVerdict:
        if video_key:
            try:
                old_metadata = json.loads(conn.execute(
                    "SELECT extra_metadata FROM source_materials WHERE id=?", (int(existing_row[0]),)
                ).fetchone()[0] or "{}")
            except (TypeError, ValueError, IndexError):
                old_metadata = {}
            deliveries = list(old_metadata.get("video_deliveries") or []) if isinstance(old_metadata, dict) else []
            if isinstance(old_metadata, dict) and old_metadata.get("delivery_id"):
                initial = {key: old_metadata.get(key) for key in
                           ("cid", "delivery_id", "reused_video_watch")}
                if initial not in deliveries:
                    deliveries.append(initial)
            marker = json.loads(video_key)
            if marker not in deliveries:
                deliveries.append(marker)
                old_metadata = dict(old_metadata) if isinstance(old_metadata, dict) else {}
                old_metadata["video_deliveries"] = deliveries
                conn.execute("UPDATE source_materials SET extra_metadata=? WHERE id=?",
                             (json.dumps(old_metadata, ensure_ascii=False, sort_keys=True), int(existing_row[0])))
        return SourceIntakeVerdict(
            ok=True,
            reason="already_ingested",
            material_id=int(existing_row[0]),
            deduplicated=True,
            credibility=credibility,
            user_md_eligible=False,
            verbatim_eligible=verbatim_eligible,
            linked_event_id=int(existing_row[1]) if existing_row[1] is not None else None,
        )

    post_commit_actions: list[Callable[[], None]] = []
    with _intake_lock_for_db(provider.sqlite.db_path), provider.sqlite.locked():
        with provider.sqlite.transaction() as conn:
            existing = conn.execute(existing_sql, existing_params).fetchone()
            if existing is not None:
                return _already_ingested_verdict(conn, existing)

            if create_memory_event:
                src_ctx = build_material_source_context(material)
                fact_subject = f"external:{material.platform.strip().lower()}"
                exp_subj, fact_subj, involved, read_scope = self_attribution(
                    owner_key if material.privacy_scope != "public" else "",
                    fact_subject=fact_subject,
                )
                prefix = (
                    f"[初步加工·{proc_type}] "
                    if material.material_kind == "processed"
                    else f"[原始材料·{material.platform}] "
                )
                title_head = f"《{material.title}》：" if material.title else ""
                event_body = f"{prefix}{title_head}{clean_content}"[:600]
                linked_event_id = provider.ingest.write_event(
                    event_body,
                    session_id or "source_intake",
                    importance=importance if importance is not None else 0.55,
                    owner_key=owner_key,
                    source="explicit",
                    source_evidence=None,
                    source_context=src_ctx,
                    experience_subject=exp_subj,
                    fact_subject=fact_subj,
                    involved_owner_keys=involved,
                    read_scope=read_scope,
                    defer_post_commit=post_commit_actions,
                )

            cursor = conn.execute(
                """INSERT INTO source_materials (
                    owner_key, session_id, source_plugin, platform, source_object_id,
                    material_kind, author, occurred_at, collected_at, privacy_scope,
                    processing_type, processing_version, raw_locator, credibility,
                    user_md_eligible, verbatim_eligible, title, content, content_sha256,
                    extra_metadata, linked_event_id, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    owner_key,
                    session_id,
                    material.source_plugin.strip(),
                    material.platform.strip().lower(),
                    material.source_object_id.strip(),
                    material.material_kind,
                    author_str,
                    occurred_at_str,
                    collected_at_str,
                    material.privacy_scope,
                    proc_type,
                    proc_ver,
                    raw_loc,
                    credibility,
                    1 if verbatim_eligible else 0,
                    material.title,
                    clean_content,
                    content_sha,
                    extra_json,
                    linked_event_id,
                    ingested_at,
                ),
            )
            assert cursor.lastrowid is not None
            material_id = int(cursor.lastrowid)
            if confirmed_meme:
                meaning = str(extras.get("meaning") or "").strip()
                aliases = json.dumps([], ensure_ascii=False)
                duplicate = conn.execute(
                    "SELECT id FROM meme_records WHERE title=? AND meaning=? "
                    "AND tier='eye_familiar' AND visibility_scope='public' LIMIT 1",
                    (material.title, meaning),
                ).fetchone()
                if duplicate is None:
                    conn.execute(
                        """INSERT INTO meme_records
                           (title, meaning, aliases, source, source_ref, source_time,
                            created_at, updated_at, owner_key, session_id, tier,
                            visibility_scope, evidence_status, source_material_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                   'eye_familiar', 'public', 'confirmed', ?)""",
                        (material.title, meaning, aliases, material.source_plugin, raw_loc or "",
                         occurred_at_str or "", ingested_at, ingested_at, owner_key,
                         session_id, material_id),
                    )
        for action in post_commit_actions:
            action()

    curiosity_hint = str((material.extra_metadata or {}).get("curiosity_candidate") or "").strip()
    if curiosity_hint and owner_key == ctx.daily_owner_key:
        open_daily = ctx.open_daily_entry
        profile_key = ctx.daily_profile_key
        if open_daily is not None and profile_key:
            try:
                from ..infra.clock import sleep_day_iso
                from .read_helpers import KIND_THREAD
                slot_seed = material.source_object_id or content_sha[:12]
                open_daily(
                    owner_key=owner_key,
                    profile_key=profile_key,
                    kind=KIND_THREAD,
                    slot=f"intake:{material.platform}:{slot_seed[:24]}",
                    text=curiosity_hint[:120],
                    opened_on=sleep_day_iso(),
                    now=ingested_at,
                )
            except Exception:
                logger.debug("sanae_mem: optional curiosity intake skipped", exc_info=True)

    return SourceIntakeVerdict(
        ok=True,
        reason="stored",
        material_id=material_id,
        deduplicated=False,
        credibility=credibility,
        user_md_eligible=False,
        verbatim_eligible=verbatim_eligible,
        linked_event_id=linked_event_id,
    )


class _SourceIntakeWorker(SourceIntakeProvider, Protocol):
    def initialize(
        self, session_id: str, *, hermes_home: str, platform: str,
        agent_context: str, hook_worker: bool,
    ) -> None: ...

    def shutdown(self) -> None: ...


_source_intake_worker_factory: Callable[[dict[str, object]], _SourceIntakeWorker]


def _bind_source_intake_worker(
    factory: Callable[[dict[str, object]], _SourceIntakeWorker],
) -> None:
    global _source_intake_worker_factory
    _source_intake_worker_factory = factory


def register_source_intake_sink(home: Path, sink: LegacySourceIntakeSink) -> None:
    register_explicit_source_sink(Path(home), sink)


def unregister_source_intake_sink(
    home: Path, sink: LegacySourceIntakeSink | None = None,
) -> None:
    unregister_explicit_source_sink(Path(home), sink)


def submit_source_material(
    material: SourceMaterial | Mapping[str, Any],
    *,
    session_id: str = "",
    create_memory_event: bool = False,
    importance: float | None = None,
    hermes_home_path: Path | str | None = None,
) -> dict[str, Any]:
    """Public entrypoint for external source plugins.

    If ViveMind is active for the current profile/session, routes into ViveMind's
    unified intake pipeline. If ViveMind is not installed or inactive, returns a
    clean ``{"ok": False, "reason": "vivemind_unavailable"}`` dict without raising.
    """
    if is_test_isolated_turn(session_id or None):
        return SourceIntakeVerdict(False, reason="test_session_isolated").to_dict()

    resolved_home = Path(hermes_home_path or get_hermes_home()).resolve()

    with hermes_home_scope(resolved_home):
        active_sid = str(session_id or current_chat_scope().session_id or "").strip()
        if active_sid:
            provider = resolve_active_provider(resolved_home, active_sid)
            if provider is not None:
                return ingest_source_material(
                    provider,
                    material,
                    create_memory_event=create_memory_event,
                    importance=importance,
                ).to_dict()

        sink = resolve_source_intake_sink(resolved_home)
        if sink is not None:
            return sink(
                material,
                session_id=active_sid,
                create_memory_event=create_memory_event,
                importance=importance,
            )

        cfg = _load_plugin_config()
        if not cfg:
            return SourceIntakeVerdict(False, reason="vivemind_unavailable").to_dict()
        if _bool(cfg.get("shadow"), False):
            return SourceIntakeVerdict(False, reason="provider_shadow_mode").to_dict()
        worker = _source_intake_worker_factory(cfg)
        try:
            worker.initialize(
                active_sid or "source-intake",
                hermes_home=str(resolved_home),
                platform=current_chat_scope().platform or "cron",
                agent_context="primary" if current_chat_scope().owner else "cron",
                hook_worker=True,
            )
            verdict = ingest_source_material(
                worker,
                material,
                create_memory_event=create_memory_event,
                importance=importance,
                allow_configured_owner_fallback=True,
            )
            return verdict.to_dict()
        finally:
            worker.shutdown()
