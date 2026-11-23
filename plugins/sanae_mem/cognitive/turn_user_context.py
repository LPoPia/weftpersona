"""Turn-scoped plugin user context cache for pre-LLM assembly and post-delivery settlement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Optional

from .daily_inputs import (
    MEME_CONTEXT_MARKER,
    SELF_UTTERANCE_MARKER,
    turn_context_block,
)

TURN_USER_CONTEXT_TTL_SECONDS = 300.0
_TurnUserContextKey = tuple[str, str, str]


@dataclass(frozen=True)
class CachedTurnUserContext:
    """Turn-scoped plugin user context blocks captured during pre-LLM assembly."""

    self_utterances: str = ""
    meme_context: str = ""
    updated_at: float = 0.0

    def render(self) -> str:
        parts = [
            part.strip()
            for part in (self.self_utterances, self.meme_context)
            if part and part.strip()
        ]
        return "\n\n".join(parts)


class TurnUserContextCache:
    """Thread-safe, TTL-evicted per-(resolved_home, session_id, turn_id) context cache."""

    def __init__(self, *, ttl_seconds: float = TURN_USER_CONTEXT_TTL_SECONDS) -> None:
        if ttl_seconds <= 0.0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = float(ttl_seconds)
        self._lock = threading.Lock()
        self._entries: dict[_TurnUserContextKey, CachedTurnUserContext] = {}

    @staticmethod
    def _make_key(home: Path | str, session_id: str, turn_id: str) -> Optional[_TurnUserContextKey]:
        raw_home = str(home or "").strip()
        sid = str(session_id or "").strip()
        tid = str(turn_id or "").strip()
        if not raw_home or not sid or not tid:
            return None
        return (str(Path(raw_home).resolve()), sid, tid)

    def _prune_locked(self, now_ts: float) -> None:
        cutoff = now_ts - self._ttl_seconds
        for key, entry in list(self._entries.items()):
            if entry.updated_at < cutoff:
                self._entries.pop(key, None)

    def record(
        self,
        home: Path | str,
        session_id: str,
        turn_id: str,
        *,
        self_utterances: str = "",
        meme_context: str = "",
        now: Optional[float] = None,
    ) -> None:
        key = self._make_key(home, session_id, turn_id)
        if key is None:
            return
        now_ts = time.time() if now is None else float(now)
        self_block = turn_context_block(self_utterances, SELF_UTTERANCE_MARKER).strip()
        meme_block = turn_context_block(meme_context, MEME_CONTEXT_MARKER).strip()
        with self._lock:
            self._prune_locked(now_ts)
            if not self_block and not meme_block:
                return
            previous = self._entries.get(key)
            merged_self = self_block or (previous.self_utterances if previous is not None else "")
            merged_meme = meme_block or (previous.meme_context if previous is not None else "")
            self._entries[key] = CachedTurnUserContext(
                self_utterances=merged_self,
                meme_context=merged_meme,
                updated_at=now_ts,
            )

    def read(
        self,
        home: Path | str,
        session_id: str,
        turn_id: str,
        *,
        now: Optional[float] = None,
    ) -> str:
        key = self._make_key(home, session_id, turn_id)
        if key is None:
            return ""
        now_ts = time.time() if now is None else float(now)
        with self._lock:
            self._prune_locked(now_ts)
            entry = self._entries.get(key)
        return entry.render() if entry is not None else ""

    def prune(self, now: Optional[float] = None) -> None:
        now_ts = time.time() if now is None else float(now)
        with self._lock:
            self._prune_locked(now_ts)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_TURN_USER_CONTEXT_CACHE = TurnUserContextCache(ttl_seconds=TURN_USER_CONTEXT_TTL_SECONDS)


def record_turn_user_context(
    home: Path | str,
    session_id: str,
    turn_id: str,
    *,
    self_utterances: str = "",
    meme_context: str = "",
    now: Optional[float] = None,
) -> None:
    _TURN_USER_CONTEXT_CACHE.record(
        home,
        session_id,
        turn_id,
        self_utterances=self_utterances,
        meme_context=meme_context,
        now=now,
    )


def read_turn_user_context(
    home: Path | str,
    session_id: str,
    turn_id: str,
    *,
    now: Optional[float] = None,
) -> str:
    return _TURN_USER_CONTEXT_CACHE.read(home, session_id, turn_id, now=now)


def prune_turn_user_context(now: Optional[float] = None) -> None:
    _TURN_USER_CONTEXT_CACHE.prune(now=now)


def clear_turn_user_context() -> None:
    _TURN_USER_CONTEXT_CACHE.clear()
