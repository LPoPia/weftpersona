from __future__ import annotations

import json
from typing import Dict, Mapping, Optional, Tuple

_SOURCE_KEYS = (
    "kind", "author_id", "platform", "ref", "time", "certainty", "visibility",
    # Appended platform facts. Old JSON without them decodes as unknown ("") and is not rewritten.
    "scope_id", "profile", "occurred_at",
    # v5 group scope: the chat/thread the utterance really happened in, plus the locator of the
    # original utterance when this speaker is only relaying someone else. Still empty for private
    # rows and never guessed from prose.
    "chat_id", "thread_id", "parent_chat_id", "thread_visibility", "session_key",
    "paraphrase_ref",
    # The scene and reply facts an off-turn group worker needs; missing JSON stays unknown.
    "chat_type", "reply_to_message_id", "synthetic_reply_thread",
    # A record distilled from one real delivery receipt states how it was produced and which
    # outbound reply it belongs to; a row written straight from a turn keeps these empty.
    "processing", "processing_version", "delivery_ref", "commitment_id",
    "source_plugin", "source_object_id", "material_kind", "processing_type",
)

# The processing metadata a model-distilled group record must keep from its producer.
PROCESSING_SOURCE_KEYS = ("processing", "processing_version", "delivery_ref", "commitment_id")

SELF_SUBJECT = "sanae"
GROUP_OWNER_PREFIX = "group:"


def is_group_owner(owner: object) -> bool:
    return str(owner or "").startswith(GROUP_OWNER_PREFIX)


def exclude_group_rows(column: str = "owner_key") -> str:
    """SQL predicate for the private/public projections: they only ever see person owners."""
    return f"COALESCE({column},'') NOT LIKE '{GROUP_OWNER_PREFIX}%'"


def source_json(source: Optional[Mapping[str, object]]) -> str:
    """把来源记录序列化成落库列（canonical JSON）；没有来源 ⇒ 空串（＝旧行 unknown）。"""
    record = _project(source)
    if record is None or not any(record.values()):
        return ""
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def parse_source(value: object) -> Optional[Dict[str, str]]:
    """读回落库的来源列；没有／读不动／空 ⇒ ``None``（unknown，不推测、不追认）。"""
    if isinstance(value, Mapping):
        return _project(value)
    if not value:
        return None
    try:
        loaded = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    return _project(loaded)


def source_identity(value: object) -> Tuple[str, ...]:
    """来源身份（查重／合并分组用）：出处不同 ⇒ 身份不同；缺来源＝全空元组（旧行）。"""
    record = parse_source(value)
    if record is None:
        return tuple("" for _ in _SOURCE_KEYS)
    return tuple(record[key] for key in _SOURCE_KEYS)


def public_source_view(record: Optional[Mapping[str, object]]) -> Dict[str, str]:
    """公开出处保留完整字段；私人出处只给出类别与确定性。"""
    source = record if isinstance(record, Mapping) else {}
    if source.get("visibility") == "public":
        return source_view(source)
    return {
        "kind": str(source.get("kind") or "unknown"),
        "certainty": str(source.get("certainty") or "unverified"),
    }


def source_view(record: Optional[Mapping[str, object]]) -> Dict[str, str]:
    """注入／检索面携带的结构来源：缺来源时**明确未知**（不冒充满分来源）。"""
    source = record if isinstance(record, Mapping) else {}
    return {
        "kind": str(source.get("kind") or "unknown"),
        "author_id": str(source.get("author_id") or ""),
        "platform": str(source.get("platform") or ""),
        "ref": str(source.get("ref") or ""),
        "time": str(source.get("time") or ""),
        "certainty": str(source.get("certainty") or "unverified"),
        "visibility": str(source.get("visibility") or "private"),
        "scope_id": str(source.get("scope_id") or ""),
        "profile": str(source.get("profile") or ""),
        "occurred_at": str(source.get("occurred_at") or ""),
        "chat_id": str(source.get("chat_id") or ""),
        "thread_id": str(source.get("thread_id") or ""),
        "parent_chat_id": str(source.get("parent_chat_id") or ""),
        "thread_visibility": str(source.get("thread_visibility") or ""),
        "session_key": str(source.get("session_key") or ""),
        "paraphrase_ref": str(source.get("paraphrase_ref") or ""),
        "chat_type": str(source.get("chat_type") or ""),
        "reply_to_message_id": str(source.get("reply_to_message_id") or ""),
        "synthetic_reply_thread": str(source.get("synthetic_reply_thread") or ""),
        "processing": str(source.get("processing") or ""),
        "processing_version": str(source.get("processing_version") or ""),
        "delivery_ref": str(source.get("delivery_ref") or ""),
        "commitment_id": str(source.get("commitment_id") or ""),
        **{key: str(source[key]) for key in
           ("source_plugin", "source_object_id", "material_kind", "processing_type")
           if source.get(key)},
    }


def _project(value: object) -> Optional[Dict[str, str]]:
    if not isinstance(value, Mapping):
        return None
    return {key: str(value.get(key) or "") for key in _SOURCE_KEYS}
