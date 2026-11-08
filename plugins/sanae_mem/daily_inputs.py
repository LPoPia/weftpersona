from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

try:
    from .daily_rules import (
        BLOCK_MARKER, _CLAUSE_SPLIT, _CLOCK_TIME, _FACE_TAG,
        _FORGET_OBJECT, _FORGET_VERB, _GATEWAY_ENVELOPE, _HOST_PREFIX,
        _IMAGE_PART_TYPES, _KNOWN_ITEM_CHARS, _KNOWN_ITEM_MAX,
        _QUESTION_MARKS, _QUESTION_PARTICLE, _QUESTION_TAIL, _QUESTION_WORDS,
        _RECENT_DIALOGUE_CHARS, _RECENT_DIALOGUE_MAX, _RECORD_SUPPRESSION,
        _REPORTED_INQUIRY, _SECOND_PERSON_HEAD, _SECRET_PATTERNS,
        _SELF_PREFIX_BY_LEN, _SELF_PRONOUN_ITER, _SELF_PRONOUN_OBJECT_MARK,
        _SELF_SUBJECT_MARK, _SUBJ_GAP, _SYSTEM_INJECTION,
    )
except ImportError:
    from daily_rules import (
        BLOCK_MARKER, _CLAUSE_SPLIT, _CLOCK_TIME, _FACE_TAG,
        _FORGET_OBJECT, _FORGET_VERB, _GATEWAY_ENVELOPE, _HOST_PREFIX,
        _IMAGE_PART_TYPES, _KNOWN_ITEM_CHARS, _KNOWN_ITEM_MAX,
        _QUESTION_MARKS, _QUESTION_PARTICLE, _QUESTION_TAIL, _QUESTION_WORDS,
        _RECENT_DIALOGUE_CHARS, _RECENT_DIALOGUE_MAX, _RECORD_SUPPRESSION,
        _REPORTED_INQUIRY, _SECOND_PERSON_HEAD, _SECRET_PATTERNS,
        _SELF_PREFIX_BY_LEN, _SELF_PRONOUN_ITER, _SELF_PRONOUN_OBJECT_MARK,
        _SELF_SUBJECT_MARK, _SUBJ_GAP, _SYSTEM_INJECTION,
    )

_TRIVIAL_ACK = re.compile(
    r"^(?:嗯+|哦+|噢+|啊+|呃+|好(?:的|呀|吧|嘞|呢)?|行(?:吧|啊)?|可以|收到|明白|知道了|"
    r"在吗|在不在|你好|您好|早安|午安|晚安|早上好|晚上好|再见|哈+|嘿+|唔+|ok|okay)+[。！!？?～~…\s]*$",
    re.IGNORECASE,
)
KNOWN_MEME_MARKER = "[已知梗图]"
SELF_UTTERANCE_MARKER = "[我刚才在别处说过]"
MEME_CONTEXT_MARKER = "[梗与表情记忆]"
_VISION_IMAGE_NOTE = re.compile(
    r"^\[The user sent an image~ Here's what I can see:\n(?P<observation>[\s\S]*?)\]\n"
    r"\[If you need a closer look, use vision_analyze with image_url: [^\n]* ~\]$"
    r"|^\[The user sent an image but (?:I couldn't quite see it|something went wrong when I)[^\n]*\]$",
    re.MULTILINE,
)
_NATIVE_IMAGE_HINT = re.compile(r"^\[Image attached(?: at)?: [^\n]*\]$", re.MULTILINE)


def _quoted_clauses(message: str, quote: str) -> List[str]:
    fragments = _split_clauses(quote)
    return [quote, *(clause for clause in _split_clauses(message)
                     if any(fragment in clause for fragment in fragments))]


def _turn_context_block(context: str, marker: str) -> str:
    return next((block for block in str(context or "").split("\n\n")
                 if block.startswith(marker)), "")


def _split_known_meme_notes(message: str) -> Tuple[str, str]:
    lines = str(message or "").splitlines()
    return ("\n".join(line for line in lines if not line.startswith(KNOWN_MEME_MARKER)),
            "\n".join(line for line in lines if line.startswith(KNOWN_MEME_MARKER)))


def _without_known_meme_notes(message: Any) -> Any:
    if isinstance(message, str):
        text, notes = _split_known_meme_notes(message)
        return text.strip() if notes else message
    if isinstance(message, list):
        return [
            dict(part, text=_without_known_meme_notes(part.get("text")))
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"} else part
            for part in message
        ]
    return message


def _assistant_quote_safe(text: str) -> bool:
    return bool(text and not _TRIVIAL_ACK.fullmatch(text)
                and not _looks_secret(text) and not _is_system_injection(text))


def _is_forget_instruction(clause: str) -> bool:
    value = str(clause or "")
    return bool(_FORGET_VERB.search(value)) and bool(_FORGET_OBJECT.search(value))


def _scrub_face_tags(text: str) -> str:
    return _FACE_TAG.sub("", str(text or ""))


def _looks_secret(text: str) -> bool:
    value = str(text or "")
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


def _redact_secrets(text: str) -> str:
    value = str(text or "")
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[已隐去]", value)
    return value


def _normalize_input(text: str) -> str:
    value = _HOST_PREFIX.sub("", str(text or ""))
    quoted = re.match(r"^\[Quoted message; length=(\d+)\]:\n", value)
    if quoted:
        end = quoted.end() + int(quoted.group(1))
        if value[end:end + 2] != "\n\n":
            return ""
        value = value[end + 2:]
    elif value.startswith("[Quoted message]:"):
        return ""
    return _strip_gateway_envelope(_scrub_face_tags(_strip_injected_block(value)))


def _strip_gateway_envelope(text: str) -> str:
    return _GATEWAY_ENVELOPE.sub(" ", str(text or ""))


def _image_parts(message: Any) -> int:
    if not isinstance(message, (list, tuple)):
        return 0
    return sum(
        1 for item in message
        if isinstance(item, dict) and str(item.get("type") or "") in _IMAGE_PART_TYPES
    )


def _model_safe_text(text: str) -> str:
    parts: List[str] = []
    for clause in _split_clauses(_normalize_input(text)):
        if _looks_secret(clause):
            continue
        parts.append(_redact_secrets(clause))
    return "，".join(part for part in parts if part).strip()


def _semantic_eligible(text: str) -> bool:
    value = str(text or "").strip()
    if not value or not _has_cjk(value):
        return False
    if _is_system_injection(value) or _looks_secret(value):
        return False
    if _RECORD_SUPPRESSION.search(value) or _is_forget_instruction(value):
        return False
    if _TRIVIAL_ACK.match(value):
        return False
    clauses = _split_clauses(value)
    if not clauses:
        return False
    if all(_is_question_clause(clause) for clause in clauses):
        return False
    return True


def _recent_dialogue(history: Any) -> str:
    if not isinstance(history, (list, tuple)):
        return ""
    entries = [entry for entry in list(history)[:-1] if isinstance(entry, dict)]
    lines: List[str] = []
    for entry in entries[-_RECENT_DIALOGUE_MAX:]:
        role = str(entry.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        body = _model_safe_text(flatten_user_text(entry.get("content")))[:_RECENT_DIALOGUE_CHARS]
        if body:
            lines.append("%s：%s" % ("他" if role == "user" else "早苗", body))
    return "\n".join(lines)


def _history_pools(history: Any, current: str = "") -> Tuple[str, str]:
    if not isinstance(history, (list, tuple)):
        return ("", "")
    entries = [item for item in list(history) if isinstance(item, dict)]
    if entries:
        last = entries[-1]
        body = _model_safe_text(flatten_user_text(last.get("content")))[:_RECENT_DIALOGUE_CHARS]
        key = re.sub(r"\s+", "", str(current or ""))
        if (str(last.get("role") or "") == "user" and key
                and re.sub(r"\s+", "", body) == key):
            entries = entries[:-1]
    user_parts: List[str] = []
    assistant_parts: List[str] = []
    for entry in entries:
        role = str(entry.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        body = _model_safe_text(flatten_user_text(entry.get("content")))[:_RECENT_DIALOGUE_CHARS]
        if not body:
            continue
        (user_parts if role == "user" else assistant_parts).append(body)
    return ("\n".join(user_parts), "\n".join(assistant_parts))


def _recent_user_pool(history: Any, current: str = "") -> str:
    if not isinstance(history, (list, tuple)):
        return ""
    key = re.sub(r"\s+", "", str(current or "") or "")
    lines: List[str] = []
    for entry in list(history)[-_RECENT_DIALOGUE_MAX:]:
        if not isinstance(entry, dict) or str(entry.get("role") or "") != "user":
            continue
        body = _model_safe_text(flatten_user_text(entry.get("content")))[:_RECENT_DIALOGUE_CHARS]
        if not body:
            continue
        if key and re.sub(r"\s+", "", body) == key:
            continue
        lines.append(body)
    return "\n".join(lines)


def _preceding_user_pool(history: Any, current: str = "") -> str:
    if not isinstance(history, (list, tuple)):
        return ""
    key = re.sub(r"\s+", "", str(current or ""))
    window = list(history)[-_RECENT_DIALOGUE_MAX:]
    if window:
        last = window[-1]
        if isinstance(last, dict) and str(last.get("role") or "") == "user":
            last_body = _model_safe_text(flatten_user_text(last.get("content")))
            if key and re.sub(r"\s+", "", last_body) == key:
                window = window[:-1]
    preceding = ""
    for entry in window:
        if not isinstance(entry, dict) or str(entry.get("role") or "") != "user":
            continue
        body = _model_safe_text(flatten_user_text(entry.get("content")))[:_RECENT_DIALOGUE_CHARS]
        if not body:
            continue
        preceding = body
    return preceding


def _known_item_lines(rows: Sequence[Any]) -> Tuple[List[str], Dict[str, str], str, Dict[str, str]]:
    lines: List[str] = []
    mapping: Dict[str, str] = {}
    slots: Dict[str, str] = {}
    texts: List[str] = []
    for row in rows:
        if len(lines) >= _KNOWN_ITEM_MAX:
            break
        slot = str(row["slot"] or "")
        text = str(row["text"] or "").strip()[:_KNOWN_ITEM_CHARS]
        if not slot or not text:
            continue
        mapping[slot] = str(row["kind"] or "")
        slots[slot] = text
        opened = str(row["opened_on"] or "").strip() if _row_has(row, "opened_on") else ""
        dated = ("(%s) " % opened) if opened else ""
        lines.append("- slot=%s | %s | %s%s" % (slot, mapping[slot], dated, text))
        texts.append(text)
    return lines, mapping, "\n".join(texts), slots


def _row_has(row: Any, key: str) -> bool:
    try:
        row[key]
    except Exception:
        return False
    return True


def _is_system_injection(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if _SYSTEM_INJECTION.search(value):
        return True
    if value.startswith("{") and value.endswith("}"):
        try:
            payload = json.loads(value)
        except Exception:
            return False
        if isinstance(payload, dict) and any(
                key in payload for key in ("role", "content", "type")):
            return True
    return False


def _self_subject_predicate(clause: str) -> bool:
    value = str(clause or "")
    for match in _SELF_PRONOUN_ITER.finditer(value):
        if match.group(0) == "我的":
            continue
        if match.start() and value[match.start() - 1] in _SELF_PRONOUN_OBJECT_MARK:
            continue
        tail = _SUBJ_GAP.sub("", value[match.end():]).lstrip()
        if not tail or tail[0] in "的你您他她它们这那":
            continue
        mark = _SELF_SUBJECT_MARK.search(tail)
        if mark is not None and mark.start() <= 3:
            return True
    return False


def _is_second_person_clause(clause: str) -> bool:
    head = _strip_function_edges(str(clause or ""))
    if not _SECOND_PERSON_HEAD.match(head):
        return False
    return not _self_subject_predicate(clause)


def flatten_user_text(message: Any) -> str:
    """只取文本：多模态里的图片/OCR/附加记忆都不是用户原始陈述。"""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, (list, tuple)):
        parts: List[str] = []
        for item in message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type") or "text") == "text":
                value = item.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(part for part in parts if part)
    if isinstance(message, dict) and str(message.get("type") or "text") == "text":
        value = message.get("text")
        return value if isinstance(value, str) else ""
    return ""


def delivered_user_input(message: Any, *, has_native_image: bool = False) -> Tuple[str, str]:
    """Separate authored text from the gateway's known-image and visual context."""
    body, memes = _split_known_meme_notes(flatten_user_text(message))
    native = has_native_image or bool(_image_parts(message))
    vision = list(_VISION_IMAGE_NOTE.finditer(body))
    if not (native or memes or vision):
        return body, ""
    images = ["对方发了图片。"]
    if memes:
        images.append(memes)
    for note in vision:
        if observation := (note.group("observation") or "").strip():
            images.append("视觉观察：" + observation)
    body = _HOST_PREFIX.sub("", _VISION_IMAGE_NOTE.sub("", body)).strip()
    if native and _NATIVE_IMAGE_HINT.search(body):
        body = _NATIVE_IMAGE_HINT.sub("", body).strip()
        if body == "What do you see in this image?":
            body = ""
    return body, "\n".join(images)


def _strip_injected_block(text: str) -> str:
    index = text.find(BLOCK_MARKER)
    if index < 0:
        return text
    return text[:index]


def _split_clauses(text: str) -> List[str]:
    parts: List[str] = []
    for chunk in _CLAUSE_SPLIT.split(text):
        value = str(chunk or "").strip()
        if not value:
            continue
        if all(char in "，,。！!；;\n\r\t？? " for char in value):
            if (parts and any(mark in value for mark in _QUESTION_MARKS)
                    and not parts[-1].endswith(_QUESTION_MARKS)):
                parts[-1] = parts[-1] + "？"
            continue
        parts.append(value)
    return parts


def _starts_with_interrogative(clause: str) -> bool:
    head = str(clause or "").strip()
    while head:
        for token in _SELF_PREFIX_BY_LEN:
            if head.startswith(token):
                head = head[len(token):]
                break
        else:
            break
    match = _QUESTION_WORDS.match(head)
    if match is None:
        return False
    return head[match.end():match.end() + 1] not in ("都", "也")


def _is_question_clause(clause: str) -> bool:
    value = str(clause or "").strip()
    if not value:
        return False
    if value.endswith(_QUESTION_MARKS):
        return True
    if _QUESTION_PARTICLE.search(value):
        return True
    if _REPORTED_INQUIRY.search(value):
        return False
    if _QUESTION_TAIL.search(value):
        return True
    return _starts_with_interrogative(value)


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def _strip_function_edges(text: str) -> str:
    value = str(text or "")
    while value:
        for token in _SELF_PREFIX_BY_LEN:
            if value.startswith(token):
                value = value[len(token):]
                break
        else:
            match = _CLOCK_TIME.match(value)
            if match is not None and match.end() > 0:
                value = value[match.end():]
                continue
            break
    while value:
        for token in _SELF_PREFIX_BY_LEN:
            if value.endswith(token):
                value = value[:-len(token)]
                break
        else:
            break
    return value.strip()
