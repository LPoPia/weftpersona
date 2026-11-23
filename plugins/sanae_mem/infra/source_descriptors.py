"""External source capabilities shared by intake, history, and provenance rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol
from urllib.parse import urlsplit

from .event_fields import parse_source


class PrivatePeerRefMatcher(Protocol):
    def __call__(self, ref: str, *, owner_key: str, peer: str) -> bool: ...


@dataclass(frozen=True)
class PrivatePeerRoute:
    platform: str
    matches_ref: PrivatePeerRefMatcher


@dataclass(frozen=True)
class ReceiptTable:
    table: str
    private_label: str
    public_label: str


@dataclass(frozen=True)
class MemeEvidenceRoute:
    plugin: str
    occurrence_kinds: frozenset[str]
    matches_original: Callable[[str, str], bool]


@dataclass(frozen=True)
class SourceDescriptor:
    platform: str
    label: str
    private_peer: PrivatePeerRoute | None = None
    receipts: ReceiptTable | None = None
    meme_evidence: MemeEvidenceRoute | None = None
    legacy_video_id: Callable[[str, str], str] | None = None
    legacy_video_mode: Callable[[str], str] | None = None
    legacy_video_in_text: Callable[[str], str] | None = None
    legacy_video_url: Callable[[str], str] | None = None
    legacy_video_kind: Callable[[str], bool] | None = None
    legacy_unscoped_video: bool = False
    legacy_video_storage_key: Callable[[str], str] | None = None
    legacy_video_ref_prefix: str = ""
    legacy_storage_id_glob: str = ""
    legacy_storage_id_length: int = 0


_sources: tuple[SourceDescriptor, ...] = ()


def install_builtin_sources(*sources: SourceDescriptor) -> None:
    global _sources
    _sources = sources


def source_descriptor(platform: str) -> SourceDescriptor | None:
    return next((source for source in _sources if source.platform == platform), None)


def legacy_video_source() -> SourceDescriptor:
    return next(source for source in _sources if source.legacy_unscoped_video)


def source_label(platform: str) -> str:
    source = source_descriptor(platform)
    return source.label if source else platform


def private_peer_route(platform: str | None) -> PrivatePeerRoute | None:
    if platform is not None:
        source = source_descriptor(platform)
        return source.private_peer if source else None
    # The old API had no platform argument and served the earliest inbox.
    return next((source.private_peer for source in _sources if source.private_peer), None)


def receipt_tables() -> tuple[ReceiptTable, ...]:
    return tuple(source.receipts for source in _sources if source.receipts is not None)


_NON_CONVERSATION_SOURCES = frozenset({"subagent", "tool", "system", "background_review"})


def is_non_conversation_source(platform: str) -> bool:
    """Recognize reserved host origins and registered external sources."""
    return platform in _NON_CONVERSATION_SOURCES or source_descriptor(platform) is not None


def web_locator(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and parsed.username is None
    except ValueError:
        return False


@dataclass(frozen=True)
class VideoOrigin:
    platform: str
    object_id: str
    ref: str

    @property
    def key(self) -> tuple[str, str]:
        return self.platform, self.object_id

    @property
    def label(self) -> str:
        return source_label(self.platform) or "平台"


def video_origin(source: Mapping[str, str], *, platform: str = "", object_id: str = "",
                 legacy_id: str = "", watch_row: bool = False) -> VideoOrigin | None:
    """Decode a typed video row, explicit video provenance, or a supported legacy source."""
    stored_platform = platform.strip().lower()
    stated_platform = source.get("platform", "").strip().lower()
    if stored_platform and stated_platform and stored_platform != stated_platform:
        return None
    platform = stated_platform or stored_platform
    ref = source.get("ref", "")
    kind = source.get("kind", "")
    descriptor = source_descriptor(platform)
    if not platform and (watch_row or kind == "video"):
        descriptor = legacy_video_source()
        if descriptor and (kind == "video" or (descriptor.legacy_video_id
                                               and descriptor.legacy_video_id(legacy_id or object_id, ref))):
            platform = descriptor.platform
    if not platform and not (object_id or ref):
        return None
    if not (watch_row or kind == "video" or (
            descriptor and descriptor.legacy_video_kind and descriptor.legacy_video_kind(kind))):
        return None

    object_id = object_id.strip() or source.get("source_object_id", "")
    if not object_id and platform and legacy_id.startswith(platform + ":"):
        object_id = legacy_id.removeprefix(platform + ":")
    if descriptor and descriptor.legacy_video_id:
        object_id = object_id or descriptor.legacy_video_id(legacy_id, ref)
        if not ref and object_id and descriptor.legacy_video_url:
            ref = descriptor.legacy_video_url(object_id)
    return VideoOrigin(platform, object_id, ref)


def public_video_origin(source: Mapping[str, str], origin: VideoOrigin | None) -> bool:
    if (origin is None or not origin.platform or not origin.object_id
            or source.get("visibility") != "public" or source.get("certainty") != "observed"):
        return False
    ref = source.get("ref", "")
    descriptor = source_descriptor(origin.platform)
    if descriptor and descriptor.legacy_video_id:
        return (source.get("kind") in {"", "self"}
                and descriptor.legacy_video_id("", ref) == origin.object_id)
    return (source.get("kind") == "video" and source.get("platform") == origin.platform
            and bool(ref) and web_locator(ref))


def watch_row_origin(source_context: object, platform: object, object_id: object,
                     legacy_id: object) -> VideoOrigin | None:
    return video_origin(parse_source(source_context) or {}, platform=str(platform or ""),
                        object_id=str(object_id or ""), legacy_id=str(legacy_id or ""), watch_row=True)


def video_storage_key(origin: VideoOrigin) -> str:
    descriptor = source_descriptor(origin.platform)
    if descriptor and descriptor.legacy_video_storage_key:
        return descriptor.legacy_video_storage_key(origin.object_id)
    return f"{origin.platform}:{origin.object_id}"


def legacy_event_origin(source: Mapping[str, str], raw: str) -> VideoOrigin | None:
    origin = video_origin(source)
    if origin and origin.object_id:
        return origin
    if source.get("platform") or source.get("kind") not in (None, "", "self"):
        return None
    for descriptor in _sources:
        if (descriptor.legacy_video_mode and descriptor.legacy_video_mode(raw)
                and descriptor.legacy_video_in_text):
            object_id = descriptor.legacy_video_in_text(raw)
            if object_id:
                return video_origin(source, platform=descriptor.platform, object_id=object_id, watch_row=True)
    return None


def legacy_playback_mode(origin: VideoOrigin, raw: str) -> str:
    descriptor = source_descriptor(origin.platform)
    if descriptor and descriptor.legacy_video_mode:
        return descriptor.legacy_video_mode(raw)
    return ""
