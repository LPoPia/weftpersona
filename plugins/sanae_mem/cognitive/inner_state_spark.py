from __future__ import annotations

from contextlib import closing
import hashlib
import re
from pathlib import Path
import sqlite3

from .inner_state_store import _connect, owner_state, replace_state_text
from ..governance.source_context import SELF_SUBJECT, in_public_interaction, parse_source
from ..infra.textutil import _tokens
from .inner_state_types import InnerState, InnerStateTarget
from ..infra.source_descriptors import legacy_event_origin, legacy_playback_mode, video_origin, watch_row_origin
from ..infra.video_record_values import video_caption_items, video_playback_label

_HOUR = 3600
_SPARK_COOLDOWN = 2 * _HOUR

_UNSPOKEN_NOTE_RE = re.compile(r"\s*（刚才我心里把这句话和「[^」]+」连到了一起，[^）]+）\s*$")

_SPARK_NOTE_RE = re.compile(r"\s*（刚才脑子里闪过一个念头：「[^」]+」[^）]+）\s*$")

_CN_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")

def _strip_unspoken_note(text: str) -> str:
    value = text or ""
    while True:
        stripped = _SPARK_NOTE_RE.sub("", _UNSPOKEN_NOTE_RE.sub("", value))
        if stripped == value:
            return stripped
        value = stripped

def _clean_unspoken_link(link: str) -> str:
    return re.sub(r"^把他的.*?(?:连到了|连到)[：:]\s*", "", (link or "").strip()).strip("「」\"'。！？ ")

def _format_unspoken_note(
    link: str, assistant_text: str = "", *, user_text: str = "", reasoning: str = ""
) -> str:
    cleaned = _clean_unspoken_link(link)
    if not cleaned or not (reasoning or "").strip():
        return ""
    if (assistant_text and cleaned in assistant_text) or (user_text and cleaned in user_text):
        return ""
    return (f"（刚才我心里把这句话和「{cleaned}」连到了一起，嘴上没有点名。）")

def _clean_spark(spark: str) -> str:
    cleaned = (spark or "").strip().strip("「」\"'")
    if len(cleaned) > 75 or len(cleaned) < 4:
        return ""
    return cleaned

def _format_spark_note(spark: str, assistant_text: str = "") -> str:
    cleaned = _clean_spark(spark)
    if not cleaned or (assistant_text and cleaned in assistant_text):
        return ""
    return (
        f"（刚才脑子里闪过一个念头：「{cleaned}」，"
        f"只是过去的一点联想，不是这轮的事实，可以不用管。）"
    )

def _distant_memory_attribution(row: sqlite3.Row | dict) -> str:
    def _get(key: str) -> str:
        try:
            return str(row[key] or "").strip()
        except (IndexError, KeyError, TypeError):
            return ""

    experience, fact = _get("experience_subject"), _get("fact_subject")
    source = parse_source(row["source_context"] if _get("source_context") else "") or {}
    kind = str(source.get("kind") or "")
    platform = _get("platform") or str(source.get("platform") or "")
    certainty = str(source.get("certainty") or "")
    origin = video_origin(source, platform=platform, watch_row=_get("source_table") == "video")
    if origin is None:
        origin = legacy_event_origin(source, _get("raw"))
    video = origin is not None
    video_label = origin.label if origin else "平台"
    suffix = f"·{platform}" if video and platform else ""
    if (experience == SELF_SUBJECT and fact.startswith("owner:")) or kind == "maintainer":
        return f"〔他说过的事{suffix}〕"
    if experience.startswith("owner:") or fact.startswith("owner:"):
        return f"〔他的事{suffix}〕"
    if kind == "external_user":
        if certainty == "unverified":
            return "〔别人说的事（未核实）〕"
        return "〔别人说的事〕"
    if (experience == SELF_SUBJECT and fact == SELF_SUBJECT) or (
            kind == "self" and not fact.startswith("owner:")):
        if video:
            if certainty == "unverified":
                return f"〔我在{video_label}看过的（未核实）〕"
            return f"〔我在{video_label}看过的〕"
        return "〔我自己的经历〕"
    if video:
        if certainty == "unverified":
            return f"〔{video_label}视频内容（未核实）〕"
        return f"〔{video_label}视频内容〕"
    return "〔过往记忆（来源未细分）〕"

def _body_tokens(text: str) -> set[str]:
    tokens = set(_tokens(str(text or "")))
    for run in _CN_RUN_RE.findall(str(text or "")):
        tokens.update(run[index:index + 2] for index in range(len(run) - 1))
    return tokens

def _compact_distant_memory_text(row: sqlite3.Row | dict) -> str:
    raw = re.sub(r"^\[[^\]]+\]\s*", "", str(row["raw"] or "").strip()).strip()
    summary = re.sub(r"^\[[^\]]+\]\s*", "", str(row["summary"] or "").strip()).strip()
    base = raw if ("抽样画面分析" in raw or "video_analysis" in raw or len(raw) > len(summary)) else (summary or raw)
    if not base:
        return ""
    if "抽样画面分析" in base or "video_analysis" in base:
        title_match = re.search(r"《[^》]+》", base)
        title = title_match.group(0) if title_match else ""
        body = re.sub(
            r"^.*?内容理解未经独立核实[：:]\s*", "", base, count=1, flags=re.S
        )
        body = re.sub(
            r"(?:^|\n)(?:##+|###+|\*\*[^*\n]+\*\*)[^\n]*\n?", "\n", body
        )
        body = re.sub(
            r"(?:以下(?:按|依据|是针对|是对该视频)[^\n。]*[。：:]?\s*|当前(?:材料|未提供)[^\n。]*[。]\s*|"
            r"人物以外观区分[^\n。]*[。]\s*|以上按画面原文摘录[^\n]*$)",
            "",
            body,
            flags=re.S,
        )
        quotes = [q.strip() for q in re.findall(r"[“\"]([^”\"\n]{2,28})[”\"]", body)]
        seen_q: list[str] = []
        for q in quotes:
            if (q not in seen_q
                    and not re.search(r"小云雀|DeepSeek|探索未至|预览版|标准模式|好嘞|嘿嘿|啊啊", q, re.I)):
                seen_q.append(q)
        clean_lines: list[str] = []
        for line in body.splitlines():
            s = re.sub(r"^[\s*|-]+|\*\*|\d{2}:\d{2}(?:\.\d+)?[–-]\d{2}:\d{2}(?:\.\d+)?[：:\s｜|]*", "", line).strip()
            if not s or s.startswith("---") or "画面内可见文字" in s or "按时间顺序" in s:
                continue
            clean_lines.append(s)
        scene_part = " ".join(clean_lines)[:120].strip()
        quote_part = ("；台词：" + " / ".join(f"「{q}」" for q in seen_q[:12])) if seen_q else ""
        combined = f"{title}：{scene_part}{quote_part}".strip("： ")
        return combined[:340]
    return (summary or raw)[:150]

def _sample_distant_memories(
    database: Path, *, owner_key: str, profile_key: str, user_text: str,
    assistant_text: str, turn_id: str, now: float,
) -> list[str]:
    if len((user_text or "").strip()) < 6 or not database.exists():
        return []
    if not owner_key:
        return []
    with closing(_connect(database, "ro")) as conn:
        if profile_key:
            row = conn.execute(
                "SELECT last_spark_at FROM inner_state_revisions "
                "WHERE profile_key=? AND owner_key=? "
                "ORDER BY revision DESC LIMIT 1",
                (profile_key, owner_key),
            ).fetchone()
            last_spark_at = float(row["last_spark_at"] or 0.0) if row is not None else 0.0
            if now - last_spark_at < _SPARK_COOLDOWN:
                return []
        rows = conn.execute(
            "SELECT id, summary, raw, topic, experience_subject, fact_subject, source_context "
            "FROM events "
            "WHERE owner_key=? AND blacklist=0 AND expired_at IS NULL "
            "AND COALESCE(a_conflict, 0) = 0 AND COALESCE(topic, '') <> '情绪' "
            "AND raw NOT LIKE '我说过：%' "
            "AND (tier IN ('important', 'long', 'distilled') OR (json_valid(source_context) "
            "     AND json_extract(source_context, '$.certainty')='observed' "
            "     AND json_extract(source_context, '$.visibility')='public')) "
            "ORDER BY id DESC LIMIT 60",
            (owner_key,),
        ).fetchall()
        video_rows = conn.execute(
            "SELECT platform, source_object_id, bvid, title, summary, analysis, captions, source_context, mode "
            "FROM video_watches WHERE owner_key=? ORDER BY watched_at DESC LIMIT 60",
            (owner_key,),
        ).fetchall()
    seen: set[tuple[str, str]] = set()
    for video in video_rows:
        origin = watch_row_origin(video["source_context"], video["platform"],
                                  video["source_object_id"], video["bvid"])
        if origin is None or (origin.object_id and origin.key in seen):
            continue
        seen.add(origin.key)
        title = str(video["title"] or "").strip()
        raw = video["summary"]
        if video["analysis"]:
            captions = video_caption_items(video["captions"])
            lines = "\n".join(f"“{str(item['text']).strip()[:28]}”" for item in captions[:6])
            raw = f"《{title}》：抽样画面分析内容理解未经独立核实：{video['analysis']}\n{lines}"
        source = parse_source(video["source_context"]) or {}
        source.update(kind="video", platform=origin.platform, source_object_id=origin.object_id, ref=origin.ref)
        rows.append({"title": title, "platform": origin.platform, "summary": video["summary"],
                     "raw": raw, "topic": "视频", "experience_subject": SELF_SUBJECT,
                     "fact_subject": SELF_SUBJECT, "source_context": source, "source_table": "video",
                     "mode": video["mode"]})
    dialogue = f"{user_text or ''} {assistant_text or ''}"
    dialogue_tokens = set(_tokens(dialogue))
    candidates: list[tuple[str, str, set[str]]] = []
    for row in rows:
        head = re.sub(r"^\[[^\]]+\]\s*", "", str(row["summary"] or row["raw"] or "").strip())[:140]
        text = _compact_distant_memory_text(row)
        if isinstance(row, dict):
            title = row["title"]
            if title and title.casefold() in dialogue.casefold():
                continue
            head = text
        if not text or len(set(_tokens(head)) & dialogue_tokens) >= 2:
            continue
        label = _distant_memory_attribution(row)
        source = parse_source(row["source_context"]) or {}
        origin = video_origin(source) or legacy_event_origin(source, str(row["raw"] or ""))
        topic = str(row["topic"] or "").strip() or (origin.platform if origin else "")
        if origin:
            mode = str(row.get("mode") or "") if isinstance(row, dict) else (
                legacy_playback_mode(origin, str(row["raw"] or "")))
            if playback := video_playback_label(mode):
                if playback not in text:
                    text = f"{playback}：{text}"
            if origin.ref and origin.ref not in text:
                text += f"（{origin.ref}）"
        candidates.append((f"{label}{text}", topic, _body_tokens(head)))
    if not candidates:
        return []
    start = int(hashlib.sha256(str(turn_id).encode("utf-8")).hexdigest(), 16) % len(candidates)
    first_text, first_topic, first_tokens = candidates[start]
    second_text = ""
    same_topic_fallback = ""
    for offset in range(1, len(candidates)):
        _text, topic, tokens = candidates[(start + offset) % len(candidates)]
        if len(first_tokens & tokens) >= 2:
            continue
        if not first_topic or not topic or first_topic != topic:
            second_text = _text
            break
        if not same_topic_fallback:
            same_topic_fallback = _text
    second_text = second_text or same_topic_fallback
    return [first_text, second_text] if second_text else [first_text]


def consume_spark(
    database: Path, target: InnerStateTarget, state: InnerState, *, owner_key: str,
) -> bool:
    if not owner_key or state.owner_key != owner_key:
        return False
    if not _SPARK_NOTE_RE.search(state.text):
        return False
    stripped = _SPARK_NOTE_RE.sub("", state.text)
    return replace_state_text(
        database, target, state, owner_key=owner_key, new_text=stripped)


def consume_pending_spark(
    database: Path,
    target: InnerStateTarget,
    *,
    owner_key: str,
    exclude_turn_id: str = "",
    delivered_at: float | None = None,
) -> bool:
    """Atomically strip any pending spark note from *owner_key*'s latest revision."""
    owner = str(owner_key or "").strip()
    if not owner or in_public_interaction():
        return False
    _data_dir, profile_key = target
    state = owner_state(database, profile_key, owner)
    if state is None:
        return False
    if exclude_turn_id and state.turn_id == exclude_turn_id:
        return False
    if delivered_at is not None and state.updated_at > float(delivered_at):
        return False
    return consume_spark(database, target, state, owner_key=owner)
