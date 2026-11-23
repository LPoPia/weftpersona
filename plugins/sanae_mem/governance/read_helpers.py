"""Shared governance-layer constants and pure text filters for daily rules and read gating."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Tuple

KIND_THREAD = "OPEN_THREAD"

KNOWN_MEME_MARKER = "[已知梗图]"


def _split_known_meme_notes(message: str) -> Tuple[str, str]:
    lines = str(message or "").splitlines()
    return (
        "\n".join(line for line in lines if not line.startswith(KNOWN_MEME_MARKER)),
        "\n".join(line for line in lines if line.startswith(KNOWN_MEME_MARKER)),
    )


def _without_known_meme_notes(message: Any) -> Any:
    if isinstance(message, str):
        text, notes = _split_known_meme_notes(message)
        return text.strip() if notes else message
    if isinstance(message, (list, tuple)):
        return [
            dict(part, text=_without_known_meme_notes(part.get("text")))
            if isinstance(part, Mapping) and part.get("type") in {"text", "input_text"}
            else part
            for part in message
        ]
    return message
