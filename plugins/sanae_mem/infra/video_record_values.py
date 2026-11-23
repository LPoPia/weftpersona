"""Shared encoding of stored video scope and caption values."""

from __future__ import annotations

import json

VIDEO_MODE_MIXED = "video_analysis+captions_only"

_PLAYBACK_LABELS = {
    "video_analysis": "收到该视频的抽样画面分析，音轨未核验",
    "captions_only": "阅读了该视频的字幕，未观看画面",
    VIDEO_MODE_MIXED: "收到该视频的抽样画面分析并阅读了字幕",
}


def video_playback_label(mode: str) -> str:
    return _PLAYBACK_LABELS.get(mode, "")


def video_scope_value(value: object) -> str:
    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return text
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    return str(value or "").strip()


def video_caption_items(value: object) -> list[dict[str, object]]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "[]")
        except (TypeError, ValueError):
            return []
    else:
        parsed = value
    if isinstance(parsed, dict):
        parsed = parsed.get("lines") or []
    if not isinstance(parsed, list):
        return []
    items: list[dict[str, object]] = []
    for entry in parsed:
        if isinstance(entry, str):
            item: dict[str, object] = {"text": entry}
        elif isinstance(entry, dict):
            item = dict(entry)
            if not str(item.get("text") or "").strip() and item.get("raw"):
                item["text"] = str(item.get("raw") or "")
        else:
            continue
        if str(item.get("text") or "").strip():
            items.append(item)
    return items
