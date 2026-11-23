from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path

from ..infra.atomic_io import atomic_write_text, file_lock
from .source_context import source_notices

ENTRY_DELIMITER = "\n§\n"
RETENTION_MARK = "【保留·不可删】"
RETENTION_MARKS = (RETENTION_MARK, "【常驻保留】")
_FILES = {"user": "USER.md", "memory": "MEMORY.md"}
_PUNCTUATION = re.compile(
    "[\\s!-/:-@\\[-`{-~\\u00b7\\u2013\\u2014\\u2018-\\u201d\\u2026"
    "\\u3000-\\u303f\\uff01-\\uff0f\\uff1a-\\uff20\\uff3b-\\uff40\\uff5b-\\uff65]+"
)


def has_retention_mark(text: str) -> bool:
    value = str(text or "")
    return any(mark in value for mark in RETENTION_MARKS)


def entry_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_entries(text: str) -> list[str]:
    return list(dict.fromkeys(entry.strip() for entry in text.split(ENTRY_DELIMITER) if entry.strip()))


def target_baseline_hash(entries: list[str]) -> str:
    return entry_hash(ENTRY_DELIMITER.join(dict.fromkeys(entries)))


def joint_baseline(user: str, memory: str) -> str:
    return entry_hash(json.dumps([user, memory], ensure_ascii=False, separators=(",", ":")))


def _normalized_locator(value: str) -> str:
    return _PUNCTUATION.sub("", value).casefold()


def _fuzzy_match(needle: str, value: str, grams: Counter) -> bool:
    pairs = Counter(zip(value, value[1:]))
    if sum((grams & pairs).values()) / (len(needle) - 1) >= .75:
        return True
    match = SequenceMatcher(None, needle, value, autojunk=False)
    if len(value) <= len(needle) * 5 // 4:
        return match.ratio() >= .75
    blocks = [block for block in match.get_matching_blocks() if block.size]
    if not blocks:
        return False
    start = blocks[0].b
    end = blocks[-1].b + blocks[-1].size
    start = max(0, min(start, end - len(needle)))
    return SequenceMatcher(None, needle, value[start:start + len(needle)], autojunk=False).ratio() >= .75


def locate_entry(entries: list[str], locator: str) -> tuple[int | None, bool]:
    if not locator.strip():
        return None, False
    if locator in entries:
        return entries.index(locator), False
    matches = [i for i, text in enumerate(entries) if locator in text]
    if not matches:
        needle = _normalized_locator(locator)
        if len(needle) >= 8:
            grams = Counter(zip(needle, needle[1:]))
            for i, text in enumerate(entries):
                value = _normalized_locator(text)
                if _fuzzy_match(needle, value, grams):
                    matches.append(i)
    return ((matches[0], False) if len({entries[i] for i in matches}) == 1
            else (None, bool(matches)))


class ResidentStore:
    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)

    def _path(self, target: str) -> Path:
        return self.memory_dir / _FILES[target]

    def target_enabled(self, target: str) -> bool:
        return target in _FILES

    def entries(self, target: str) -> list[str]:
        try:
            return parse_entries(self._path(target).read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return []

    @staticmethod
    def char_limit(target: str) -> int:
        from ..infra.host_bridge import load_host_config

        limits = (load_host_config() or {}).get("memory") or {}
        return int(limits.get(target + "_char_limit", 1375 if target == "user" else 2200))

    @contextmanager
    def locked(self):
        # The pin uses the same adjacent lock files for native writers.
        with file_lock(self._path("memory").with_suffix(".md.lock")):
            with file_lock(self._path("user").with_suffix(".md.lock")):
                yield

    def baseline(self) -> str:
        return joint_baseline(*(ENTRY_DELIMITER.join(self.entries(target))
                                for target in ("user", "memory")))

    def add(self, target: str, content: str) -> dict:
        with self.locked():
            res = self.apply_locked(target, content=content)
            if not res.get("success") and "error" not in res:
                res = {**res, "error": str(res.get("reason") or "resident_add_failed")}
            return res

    def remove(self, target: str, entry: str, *, expected_hash: str = "") -> dict:
        with self.locked():
            return self.apply_locked(
                target,
                replaces=entry,
                expected_entry=expected_hash or entry_hash(entry),
            )

    def apply_locked(self, target: str, *, content: str = "", replaces: str = "",
                     expected_entry: str = "") -> dict:
        from ..infra.threat_scan import scan_for_threats

        entries = self.entries(target)
        if replaces and (replaces not in entries or entry_hash(replaces) != expected_entry):
            return {"success": False, "reason": "entry_baseline_changed"}
        if replaces and has_retention_mark(replaces):
            return {"success": False, "reason": "retention_protected"}
        if replaces and content and any(notice not in content for notice in source_notices(replaces)):
            return {"success": False, "reason": "source_provenance_required"}
        if content and (ENTRY_DELIMITER in content or scan_for_threats(content, scope="strict")):
            return {"success": False, "reason": "unsafe_resident_entry"}
        if replaces:
            index = entries.index(replaces)
            entries[index:index + 1] = [content] if content else []
        elif content and content not in entries:
            entries.append(content)
        body = ENTRY_DELIMITER.join(dict.fromkeys(entries))
        if content and len(body) > self.char_limit(target):
            return {"success": False, "reason": "resident_capacity_exceeded"}
        atomic_write_text(self._path(target), body)
        if self.entries(target) != parse_entries(body):
            return {"success": False, "reason": "resident_readback_failed"}
        return {"success": True}
