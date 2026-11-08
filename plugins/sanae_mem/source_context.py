"""可信来源的只读视图（宿主 ``InteractionContext`` 的规范化投影）。

来源**只能**来自宿主接入层：``agent.interaction_context.InteractionContext`` 由派发方构造，
经既有 ContextVar／``ctx_bound`` 工作线程传递；消息正文与工具参数都建立不了它
（``source_context`` 不是 USER 证言，只是一段出处记录）。

本模块不缓存、不落任何进程态：每个调用点当场读，读不到就给 ``None``／``False``——
缺少新元数据时明确未知，不推断作者或追认维护者资格。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

_SOURCE_KEYS = ("kind", "author_id", "platform", "ref", "time", "certainty", "visibility")

SELF_SUBJECT = "sanae"


def participant_code(owner_key: str) -> str:
    """Stable, non-identifying code for the same trusted owner."""
    value = str(owner_key or "")
    if not value:
        return ""
    digest = hashlib.sha256(("sanae-shared:" + value).encode("utf-8")).hexdigest()
    return "参与者-" + digest[:6].upper()


def is_self_scope(value: Any) -> bool:
    if isinstance(value, dict):
        scope = value
    elif value:
        try:
            scope = json.loads(str(value))
        except (TypeError, ValueError):
            return False
    else:
        return False
    return isinstance(scope, dict) and str(scope.get("kind") or "") == "self"


def self_attribution(owner_key: str = "", *, fact_subject: Optional[str] = None
                     ) -> Tuple[str, str, str, str]:
    """(experience_subject, fact_subject, involved_owner_keys, read_scope).

    A conversation or action is experienced by Sanae; a bound owner is an involved
    person, not the subject. ``fact_subject`` says whose first-person fact the raw
    text states: Sanae for her own experience, or ``owner:<key>`` for an automatic
    user statement. The two are separate so the later read batch never renders a
    user's "I could not sleep" as Sanae's own insomnia.

    The raw scope stays private: self-only when no person is involved, or that
    involved owner's context only when the event carries the person's private
    context. A safe cross-user view is a later-batch read view, never this raw row.
    """
    owner = str(owner_key or "")
    fact = SELF_SUBJECT if fact_subject is None else str(fact_subject or "")
    involved = [owner] if owner else []
    scope = ({"kind": "owner", "owner": owner} if owner
             else {"kind": "self"})
    return (SELF_SUBJECT, fact,
            json.dumps(involved, ensure_ascii=False, sort_keys=True),
            json.dumps(scope, ensure_ascii=False, sort_keys=True))


def user_fact_attribution(owner_key: str) -> Tuple[str, str, str, str]:
    """A user's own grounded fact never becomes Sanae's personal experience."""
    owner = str(owner_key or "")
    if not owner:
        return "", "", "", ""
    return (f"owner:{owner}", f"owner:{owner}",
            json.dumps([owner], ensure_ascii=False, sort_keys=True),
            json.dumps({"kind": "owner", "owner": owner},
                       ensure_ascii=False, sort_keys=True))


def current_source_record() -> Optional[Dict[str, str]]:
    """当前这一轮的可信来源（规范化记录）；没有可信上下文 ⇒ ``None``（旧行为）。"""
    try:
        from agent.interaction_context import current_interaction, source_record
    except Exception:
        return None
    try:
        interaction = current_interaction()
        if interaction is None:
            return None
        return source_record(getattr(interaction, "source", None), public=bool(interaction.public))
    except Exception:
        return None


def in_public_interaction() -> bool:
    """公开交流场景（宿主派发方声明）。读不到上下文 ⇒ ``False``（单用户旧行为）。"""
    try:
        from agent.interaction_context import current_interaction
    except Exception:
        return False
    try:
        interaction = current_interaction()
        return bool(interaction is not None and interaction.public)
    except Exception:
        return False


def source_json(source: Optional[Dict[str, Any]]) -> str:
    """把来源记录序列化成落库列（canonical JSON）；没有来源 ⇒ 空串（＝旧行 unknown）。"""
    record = _project(source)
    if record is None or not any(record.values()):
        return ""
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def parse_source(value: Any) -> Optional[Dict[str, str]]:
    """读回落库的来源列；没有／读不动／空 ⇒ ``None``（unknown，不推测、不追认）。"""
    if isinstance(value, dict):
        return _project(value)
    if not value:
        return None
    try:
        loaded = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    return _project(loaded)


def source_identity(value: Any) -> Tuple[str, ...]:
    """来源身份（查重／合并分组用）：出处不同 ⇒ 身份不同；缺来源＝全空元组（旧行）。"""
    record = parse_source(value)
    if record is None:
        return tuple("" for _ in _SOURCE_KEYS)
    return tuple(record[key] for key in _SOURCE_KEYS)


def is_maintainer_source(value: Any) -> bool:
    """落库来源是否**明确**为维护者本人；空／unknown 一律 ``False``（不追认）。"""
    record = parse_source(value)
    return bool(record and record["kind"] == "maintainer")


def public_source_view(record: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """公开出处保留完整字段；私人出处只给出类别与确定性。"""
    source = record if isinstance(record, dict) else {}
    if source.get("visibility") == "public":
        return source_view(source)
    return {
        "kind": str(source.get("kind") or "unknown"),
        "certainty": str(source.get("certainty") or "unverified"),
    }


def source_view(record: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """注入／检索面携带的结构来源：缺来源时**明确未知**（不冒充满分来源）。"""
    source = record if isinstance(record, dict) else {}
    return {
        "kind": str(source.get("kind") or "unknown"),
        "author_id": str(source.get("author_id") or ""),
        "platform": str(source.get("platform") or ""),
        "ref": str(source.get("ref") or ""),
        "time": str(source.get("time") or ""),
        "certainty": str(source.get("certainty") or "unverified"),
        "visibility": str(source.get("visibility") or "private"),
    }


def _project(value: Any) -> Optional[Dict[str, str]]:
    if not isinstance(value, dict):
        return None
    return {key: str(value.get(key) or "") for key in _SOURCE_KEYS}
