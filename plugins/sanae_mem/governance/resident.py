from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple

from ..infra.host_bridge import is_delegated_child_turn, is_test_isolated_turn
from ..infra.archive import _append_fsynced_jsonl
from ..infra.atomic_io import atomic_json_write, file_lock
from .resident_store import (
    ENTRY_DELIMITER, ResidentStore, entry_hash, has_retention_mark, locate_entry,
)
from ..infra.textutil import _embed, _fingerprint, _pack_vector, _tokens

from .source_context import (
    current_chat_scope, in_public_interaction, self_attribution, user_fact_attribution,
)

from ..storage.event_store import SUMMARY_SOURCE_GIST_FADED

if TYPE_CHECKING:
    from ..storage.event_store import GroundedOccurrence
    from ..storage.sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)
_RESIDENT_TOUCH_LOCK = threading.RLock()

TOUCH_FILE = "resident_touch.json"
TOUCH_MIN_OVERLAP = 3
TOUCH_SHINGLE_RATIO = 0.25
CAPACITY_PCT = 85.0
COLD_DAYS = 45
DEMOTED_TIER = "long"
DEMOTED_IMPORTANCE = 0.75
DEMOTED_RECALL_COUNT = 3
DEMOTED_HALF_LIFE_DAYS = 60.0
DEMOTED_PROMOTED_REASON = "resident_demote"
DEMOTE_SOURCE_KIND = "resident_demote"


def _entry_sha(entry: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", entry).strip().encode("utf-8")).hexdigest()[:16]


def _entry_key(target: str, entry: str) -> str:
    return f"{target}:{_entry_sha(entry)}"


def _days_since(stamp: Any, now: datetime) -> Optional[float]:
    try:
        value = datetime.fromisoformat(str(stamp or ""))
    except (TypeError, ValueError):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=now.tzinfo)
    return max(0.0, (now - value).total_seconds() / 86400)


def _blocked(reason: str = "resident_write_blocked") -> dict:
    return {"ok": False, "demoted": False, "reason": reason}


class _GroundedReview(Protocol):
    def __call__(
        self, *, text: str, evidence_ids: list[int], replaces: str, memory_dir: Path,
    ) -> dict: ...


class _PromotionReview(Protocol):
    def __call__(self, event_id: int, /, *, enforce_daily_cap: bool) -> dict: ...


class _UserDemoteReview(Protocol):
    def __call__(
        self, candidates: list[tuple[int, str]], /, *, resident_user: str, now: datetime,
    ) -> dict[int, dict] | None: ...


class _MaintenanceStatus(Protocol):
    def __call__(
        self, component: str, /, *, ok: bool, stage: str, now: datetime, error: str = "",
    ) -> None: ...


@dataclass(frozen=True)
class ResidentDependencies:
    sqlite: SqliteFacade
    home: Callable[[], Path]
    data_dir: Callable[[], Path]
    session_id: Callable[[], str]
    owner_key: Callable[[], str]
    daily_owner_key: Callable[[], str]
    agent_context: Callable[[], str]
    shadow_enabled: Callable[[], bool]
    is_configured_private_owner: Callable[[], bool]
    is_system_turn: Callable[[str], bool]
    read_owner_keys: Callable[[str], list[str]]
    review_grounded_user: _GroundedReview
    grounded_binding: Callable[[dict], dict | None]
    commit_recognition_support: Callable[[str, dict], dict]
    get_grounded_occurrence: Callable[[int], GroundedOccurrence | None]
    is_qualified_occurrence: Callable[[GroundedOccurrence], bool]
    review_promotion_candidate: _PromotionReview
    get_event_review: Callable[[int, str], sqlite3.Row | None]
    event_exists: Callable[[int], bool]
    promote_events: Callable[[list[int], str, str], int]
    review_user_demotion: _UserDemoteReview
    memory_capacity_snapshot: Callable[[], dict[str, int | float]]
    write_maintenance_status: _MaintenanceStatus
    topic: Callable[[str], str]
    keywords: Callable[[str], str]
    upsert_vector: Callable[[int, str, str, float, str, list[float]], None]


class ResidentMemory:
        def __init__(self, dependencies: ResidentDependencies) -> None:
            self._deps = dependencies
            self._sqlite = dependencies.sqlite

        def store(self) -> ResidentStore:
            return ResidentStore(self._deps.home() / "memories")

        def can_review_private_user(self, *, require_message: bool = True) -> bool:
            scope = current_chat_scope()
            return bool(not self._deps.shadow_enabled() and self._deps.is_configured_private_owner()
                        and scope.session_id == self._deps.session_id() and (scope.message_id or not require_message)
                        and not is_delegated_child_turn() and not self._resident_writes_blocked())

        def save_grounded_user(self, *, text: str, evidence_ids: list[int], replaces: str = "") -> dict:
            if not self.can_review_private_user():
                return _blocked("verified_maintainer_private_chat_required")
            body = text.strip()
            replaces = replaces.strip()
            store = self.store()
            reviewed = self._deps.review_grounded_user(
                text=body, evidence_ids=evidence_ids, replaces=replaces, memory_dir=store.memory_dir)
            if not reviewed.get("ok"):
                return reviewed
            original_user = list(store.entries("user"))
            original_memory = list(store.entries("memory"))
            covered = str(reviewed.get("covered_by") or "")
            evicted: List[str] = []
            if not covered or replaces:
                projected = list(original_user)
                if replaces:
                    projected[projected.index(replaces)] = body
                elif body not in projected:
                    projected.append(body)
                if len(ENTRY_DELIMITER.join(dict.fromkeys(projected))) > store.char_limit("user"):
                    cap = self._ensure_user_capacity(
                        store, incoming_content=body, replaces=replaces,
                        now=datetime.now().astimezone())
                    if not cap.get("ok"):
                        return _blocked(str(cap.get("reason") or "resident_capacity_exceeded"))
                    evicted = list(cap.get("evicted") or [])
            anchor = json.loads(reviewed["evidence_group"])[0]
            declared = dict(reviewed, **anchor, content=body)
            return self._apply_grounded_user(
                declared, replaces=replaces, covered=covered, evicted=evicted,
                original_user=original_user, original_memory=original_memory)

        def _apply_grounded_user(self, declared: dict, *, replaces: str, covered: str,
                                 evicted: Optional[List[str]] = None,
                                 original_user: Optional[List[str]] = None,
                                 original_memory: Optional[List[str]] = None) -> dict:
            if not self.can_review_private_user():
                return _blocked("verified_maintainer_private_chat_required")
            store = self.store()
            try:
                with store.locked(), self._sqlite.transaction() as conn:
                    evicted = list(evicted or [])
                    if evicted:
                        expected_user = [e for e in (original_user or []) if e not in evicted]
                        if (store.entries("memory") != list(original_memory or [])
                                or store.entries("user") != expected_user):
                            return _blocked("resident_baseline_changed")
                        baseline = store.baseline()
                        conn.execute(
                            "UPDATE event_reviews SET baseline_sha256=? WHERE review_sha256=?",
                            (baseline, str(declared.get("review_sha256") or "")))
                        declared["baseline_sha256"] = baseline
                    binding = self._deps.grounded_binding(declared)
                    if binding is None:
                        return _blocked("review_binding_invalid")
                    if binding["baseline_sha256"] != store.baseline():
                        return _blocked("resident_baseline_changed")
                    target = conn.execute(
                        "SELECT covered_sha256 FROM recognition_review_targets WHERE review_sha256=?",
                        (binding["review_sha256"],)).fetchone()
                    if (target is None or target["covered_sha256"] != (entry_hash(covered) if covered else "")
                            or (replaces and replaces != covered)
                            or (covered and covered not in store.entries("user"))):
                        return _blocked("review_coverage_changed")
                    body = binding["reviewed_text"]
                    if not covered or replaces:
                        written = store.apply_locked("user", content=body, replaces=replaces,
                                                     expected_entry=entry_hash(replaces))
                        if not written.get("success"):
                            return _blocked(written["reason"])
                    ids = binding["evidence_group_ids"]
                    self._deps.promote_events(ids, "long", datetime.now().astimezone().isoformat())
                    support = self._deps.commit_recognition_support(covered if covered and not replaces else body,
                                                              binding)
            except (OSError, ValueError):
                return _blocked("resident_store_unavailable")
            return {"ok": True, "target": "user", "written": bool(not covered or replaces),
                    "support_linked": support["ok"], "support": support,
                    "evicted_count": len(evicted or [])}

        @staticmethod
        def _project_user_size(entries: List[str], incoming_content: str, replaces: str) -> int:
            projected = list(entries)
            if replaces and replaces in projected:
                projected[projected.index(replaces)] = incoming_content
            elif incoming_content and incoming_content not in projected:
                projected.append(incoming_content)
            return len(ENTRY_DELIMITER.join(dict.fromkeys(projected)))

        def _ensure_user_capacity(self, store: ResidentStore, *, incoming_content: str,
                                  replaces: str, now: datetime) -> dict:
            entries = list(store.entries("user"))
            candidates = [(idx, entry) for idx, entry in enumerate(entries)
                          if not has_retention_mark(entry) and entry != replaces]
            if not candidates:
                return {"ok": False, "reason": "resident_capacity_exceeded", "evicted": []}
            evicted: List[str] = []
            for offset in range(0, len(candidates), 10):
                batch = candidates[offset:offset + 10]
                try:
                    decisions = self._deps.review_user_demotion(
                        batch, resident_user=ENTRY_DELIMITER.join(entries), now=now)
                except TimeoutError as exc:
                    logger.warning("sanae_mem: capacity demote review failed (%s)", type(exc).__name__)
                    decisions = None
                if decisions is None:
                    self._deps.write_maintenance_status(
                        "resident_demote", ok=False, stage="user_demote_review",
                        error="demote_review_unavailable", now=now)
                    return {"ok": False, "reason": "demote_review_unavailable", "evicted": evicted}
                for idx, entry in batch:
                    decision = decisions.get(idx) if isinstance(decisions, dict) else None
                    if not isinstance(decision, dict) or not decision.get("allow_demote"):
                        continue
                    result = self.demote_resident_entry(
                        "user", entry, now=now, _preverified_decision=decision)
                    if result.get("reason") in {
                        "archive_failed", "event_write_failed", "archived_destination_unavailable",
                    }:
                        return {"ok": False, "reason": result["reason"], "evicted": evicted}
                    if result.get("demoted"):
                        evicted.append(entry)
                        if self._project_user_size(
                                list(store.entries("user")), incoming_content, replaces) <= store.char_limit("user"):
                            return {"ok": True, "evicted": evicted}
            return {"ok": False, "reason": "resident_capacity_exceeded", "evicted": evicted}

        def _resident_touch_path(self) -> Path:
            return self._deps.data_dir() / TOUCH_FILE

        def _read_resident_touch(self) -> dict:
            path = self._resident_touch_path()
            try:
                value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, ValueError):
                value = {}
            if not isinstance(value, dict):
                value = {}
            for key in ("entries", "demoted"):
                if not isinstance(value.get(key), dict):
                    value[key] = {}
            return value

        def _mutate_resident_touch(self, change: Callable[[dict], bool]) -> Optional[dict]:
            path = self._resident_touch_path()
            with _RESIDENT_TOUCH_LOCK, file_lock(path.with_suffix(".lock")):
                state = self._read_resident_touch()
                try:
                    if not change(state):
                        return state
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json_write(path, state)
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "sanae_mem: resident_touch write failed (%s)", type(exc).__name__)
                    return None
                return state

        def _resident_writes_blocked(self, session_id: str = "", *, maintenance: bool = False) -> str:
            nightly = maintenance and self._deps.agent_context() == "cron" and bool(self._deps.daily_owner_key())
            if self._deps.agent_context() != "primary" and not nightly:
                return "non_primary_context"
            if in_public_interaction():
                return "public_interaction"
            bound = self._deps.session_id()
            session = str(session_id or bound)
            if is_test_isolated_turn(session) or is_test_isolated_turn(bound):
                return "test_session_isolated"
            if not nightly and (self._deps.is_system_turn(session) or self._deps.is_system_turn(bound)):
                return "system_turn"
            if not nightly and self._deps.daily_owner_key() and self._deps.daily_owner_key() != self._deps.owner_key():
                return "different_owner"
            if not session:
                return ""
            return ""

        @staticmethod
        def _resident_match(turn_tokens: set[str], compact_turn: str, entry: str) -> Tuple[bool, int]:
            overlap = len(turn_tokens & set(_tokens(entry)))
            if overlap >= TOUCH_MIN_OVERLAP:
                return True, overlap
            compact_entry = re.sub(r"\s+", "", str(entry or ""))
            if len(compact_entry) < 3:
                return False, overlap
            shingles = {compact_entry[i:i + 3] for i in range(len(compact_entry) - 2)}
            hit = sum(1 for shingle in shingles if shingle in compact_turn)
            return hit / len(shingles) >= TOUCH_SHINGLE_RATIO, overlap

        def update_touch(
            self, user_content: str, assistant_content: str, *,
            session_id: str = "", now: Optional[datetime] = None,
        ) -> int:
            try:
                if self._resident_writes_blocked(session_id):
                    return 0
                store = self.store()
                text = f"{user_content or ''} {assistant_content or ''}"
                turn_tokens = set(_tokens(text))
                now = now or datetime.now().astimezone()
                compact_turn = re.sub(r"\s+", "", text)
                return self._reconcile_resident_entries(store, now, turn_tokens, compact_turn)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "sanae_mem: resident touch update failed (%s)", type(exc).__name__)
                return 0

        def _reconcile_resident_entries(
            self, store: ResidentStore, now: datetime, tokens: set[str] | None = None, turn: str = "",
        ) -> int:
            stamp = now.isoformat()
            matched = set()
            current = {}
            for target in ("memory", "user"):
                if not store.target_enabled(target):
                    continue
                for entry in store.entries(target):
                    key = _entry_key(target, entry)
                    current[key] = (target, entry[:40])
                    if tokens and self._resident_match(tokens, turn, entry)[0]:
                        matched.add(key)

            def reconcile(state: dict) -> bool:
                old = state["entries"]
                if not matched and current.keys() == old.keys():
                    return False
                fresh = {}
                for key, (target, head) in current.items():
                    record = old.get(key) or dict(target=target, head=head, first_seen_at=stamp,
                                                  last_matched_at=stamp, match_count=0)
                    if key in matched:
                        record["last_matched_at"] = stamp
                        record["match_count"] = int(record.get("match_count", 0)) + 1
                    fresh[key] = record
                state["entries"] = fresh
                return True

            return len(matched) if self._mutate_resident_touch(reconcile) is not None else 0

        def cold_report(self, now: datetime) -> dict:
            capacity = self._deps.memory_capacity_snapshot()
            hot = [
                target for target in ("memory",)
                if float(capacity.get(target + "_pct") or 0.0) >= CAPACITY_PCT
            ]
            cold: List[dict] = []
            store = self.store()
            if not self._resident_writes_blocked(maintenance=True):
                self._reconcile_resident_entries(store, now)
            if hot:
                state = self._read_resident_touch()
                for target in hot:
                    if not store.target_enabled(target):
                        continue
                    for entry in store.entries(target):
                        digest = _entry_sha(entry)
                        record = state["entries"].get(f"{target}:{digest}") or {}
                        unmatched = _days_since(record.get("last_matched_at"), now)
                        seen = _days_since(record.get("first_seen_at"), now)
                        if unmatched is None or seen is None:
                            continue
                        if unmatched < COLD_DAYS or seen < COLD_DAYS:
                            continue
                        cold.append({
                            "target": target, "entry_sha": digest, "head": entry[:40],
                            "days_unmatched": int(unmatched), "chars": len(entry),
                        })
            cold.sort(key=lambda item: (-item["days_unmatched"], item["target"]))
            return {"capacity": capacity, "cold_resident_entries": cold}

        def _archive_demoted_entry(self, target: str, entry: str, sha: str,
                                   now: datetime) -> bool:
            path = self._deps.home() / "memory-archive" / f"{target}-demoted.jsonl"
            line = json.dumps({
                "archived_at": now.isoformat(), "target": target, "entry_sha": sha,
                "reason": DEMOTE_SOURCE_KIND, "entry": entry,
            }, ensure_ascii=False)
            try:
                _append_fsynced_jsonl(path, (line,))
            except OSError as exc:
                logger.warning(
                    "sanae_mem: resident demote archive failed (%s)", type(exc).__name__)
                return False
            return True

        def _insert_demoted_event(self, target: str, entry: str, sha: str,
                                  last_active: str, now: datetime) -> Optional[int]:
            assert self._sqlite.is_open
            body = re.sub(r"\s+", " ", entry).strip()
            if not body:
                return None
            attribution = user_fact_attribution if target == "user" else self_attribution
            owner = self._deps.owner_key() or self._deps.daily_owner_key()
            subject, fact, involved, scope = attribution(owner)
            topic = self._deps.topic(body)
            ts = now.strftime("%Y-%m-%d %H:%M")
            stamp = now.isoformat()
            summary = body[:150]
            vector = _embed(body, data_dir=self._deps.data_dir())
            values: Dict[str, Any] = {
                "ts": ts, "raw": entry if target == "user" else f"[{ts}] {entry}",
                "summary": summary,
                "summary_source": SUMMARY_SOURCE_GIST_FADED,
                "keywords": self._deps.keywords(body), "importance": DEMOTED_IMPORTANCE,
                "confidence": 0.6, "topic": topic, "ttl_expire": None,
                "created_at": stamp, "updated_at": stamp, "vec": _pack_vector(vector),
                "session_id": self._deps.session_id(), "tier": DEMOTED_TIER,
                "promoted_at": stamp, "promoted_reason": DEMOTED_PROMOTED_REASON,
                "asked": 1, "asked_at": stamp, "owner_key": owner,
                "experience_subject": subject, "fact_subject": fact,
                "involved_owner_keys": involved, "read_scope": scope,
            }
            values.update(last_active_at=last_active or stamp, recall_count=DEMOTED_RECALL_COUNT,
                          half_life_days=DEMOTED_HALF_LIFE_DAYS, original_fingerprint=_fingerprint(body))
            columns = list(values)
            with self._sqlite.transaction() as conn:
                if conn.execute(
                        "SELECT 1 FROM forgotten WHERE fingerprint=?", (_fingerprint(body),)).fetchone():
                    return None
                cursor = conn.execute(
                    "INSERT INTO events (%s) VALUES (%s)" % (
                        ", ".join(columns), ", ".join("?" for _ in columns)),
                    [values[name] for name in columns],
                )
                assert cursor.lastrowid is not None
                event_id = int(cursor.lastrowid)
            self._deps.upsert_vector(event_id, summary, topic, DEMOTED_IMPORTANCE, ts, vector)
            return event_id

        @staticmethod
        def _locate_resident_entry(entries: List[str], locator: str):
            return locate_entry(entries, str(locator or ""))

        def _remove_resident_entry(self, store: ResidentStore, target: str, entry: str) -> dict:
            try:
                result = store.remove(target, entry, expected_hash=entry_hash(entry))
            except (OSError, ValueError) as exc:
                return {"success": False, "reason": type(exc).__name__}
            return result if isinstance(result, dict) else {"success": False, "reason": "remove_failed"}

        def demote_resident_entry(
            self, target: str, entry_text: str, now: Optional[datetime] = None,
            *, _preverified_decision: Optional[dict] = None,
        ) -> dict:
            """把一条常驻条目退居成事件库里的长期印象，再从常驻文件移除。

            顺序固定：核对条目在册 → 原文归档 → 写长期印象事件 → 才动常驻文件。
            MEMORY 在事件落库后按 CAS 直接移除；USER 由插件内判断模型确认是过时
            阶段近况后按 CAS 直接移除。两者都不生成 pending 人工提案。
            """
            if not self._sqlite.is_open:
                raise RuntimeError("provider is not initialized")
            target = str(target or "").strip().lower()
            if target not in ("memory", "user"):
                raise ValueError("target must be memory or user")
            blocked = self._resident_writes_blocked(maintenance=True)
            if blocked:
                return _blocked(blocked)
            with file_lock(self._resident_touch_path().with_suffix(".demote.lock")):
                return self._demote_locked(
                    target, entry_text, now or datetime.now().astimezone(),
                    _preverified_decision=_preverified_decision)

        def _demote_locked(self, target: str, entry_text: str, now: datetime, *,
                           _preverified_decision: Optional[dict] = None) -> dict:
            store = self.store()
            if not store.target_enabled(target):
                return _blocked("target_disabled")
            entries = list(store.entries(target))
            idx, ambiguous = self._locate_resident_entry(entries, entry_text)
            if idx is None:
                return {"ok": False, "demoted": False,
                        "reason": "entry_ambiguous" if ambiguous else "entry_not_found"}
            entry = entries[idx]
            if has_retention_mark(entry):
                return {"ok": False, "demoted": False, "reason": "retention_protected"}

            sha = _entry_sha(entry)
            key = f"{target}:{sha}"
            state = self._read_resident_touch()
            record = state["entries"].get(key) or {}
            receipt = dict(state["demoted"].get(key) or {})
            last_active = str(record.get("last_matched_at") or "") or now.isoformat()

            event_id = receipt.get("event_id")
            if not isinstance(event_id, int) or not self._deps.event_exists(event_id):
                event_id = None
            if event_id is None:
                owner = self._deps.owner_key() or self._deps.daily_owner_key()
                existing = self._sqlite.fetchone(
                    "SELECT id FROM events WHERE promoted_reason=? AND owner_key=? AND original_fingerprint=? "
                    "AND fact_subject=? AND blacklist=0 ORDER BY id DESC LIMIT 1",
                    (DEMOTED_PROMOTED_REASON, owner, _fingerprint(entry),
                     "owner:" + owner if target == "user" else "sanae"),
                )
                event_id = int(existing[0]) if existing else None
            if event_id is None:
                if not self._archive_demoted_entry(target, entry, sha, now):
                    return {"ok": False, "demoted": False, "reason": "archive_failed"}
                try:
                    event_id = self._insert_demoted_event(target, entry, sha, last_active, now)
                except (OSError, ValueError, sqlite3.Error) as exc:
                    logger.warning(
                        "sanae_mem: resident demote event write failed (%s)",
                        type(exc).__name__)
                    event_id = None
                if event_id is None:
                    return {"ok": False, "demoted": False, "reason": "event_write_failed"}
                receipt = {
                    "target": target, "event_id": event_id, "entry_sha": sha,
                    "demoted_at": now.isoformat(),
                }

            def save_receipt(**fields: Any) -> None:
                def change(current: dict) -> bool:
                    merged = dict(receipt)
                    merged.update(fields)
                    current["demoted"][key] = merged
                    return True
                self._mutate_resident_touch(change)

            if target == "user":
                decision = _preverified_decision
                if decision is None:
                    try:
                        decisions = self._deps.review_user_demotion(
                            [(0, entry)], resident_user=ENTRY_DELIMITER.join(entries), now=now)
                    except TimeoutError as exc:
                        logger.warning("sanae_mem: user demote review failed (%s)", type(exc).__name__)
                        decisions = None
                    decision = decisions.get(0) if isinstance(decisions, dict) else None
                if decision is None:
                    self._deps.write_maintenance_status(
                        "resident_demote", ok=False, stage="user_demote_review",
                        error="demote_review_unavailable", now=now)
                    save_receipt(removed=False, review_error="demote_review_unavailable")
                    return {"ok": False, "demoted": False, "staged": False,
                            "target": "user", "event_id": event_id,
                            "reason": "demote_review_unavailable"}
                kind = str(decision.get("kind") or "")
                self._deps.write_maintenance_status(
                    "resident_demote", ok=True, stage="user_demote_review", now=now)
                if not decision.get("allow_demote"):
                    save_receipt(removed=False, review_kind=kind)
                    return {"ok": False, "demoted": False, "staged": False,
                            "target": "user", "event_id": event_id,
                            "reason": "user_entry_not_outdated", "kind": kind}
                archived = self._sqlite.fetchone(
                    "SELECT 1 FROM events WHERE id=? AND owner_key=? AND blacklist=0 "
                    "AND tier='long' AND promoted_reason=?",
                    (event_id, self._deps.owner_key() or self._deps.daily_owner_key(),
                     DEMOTED_PROMOTED_REASON),
                )
                if archived is None:
                    save_receipt(removed=False)
                    return {"ok": False, "demoted": False, "staged": False,
                            "target": "user", "event_id": event_id,
                            "reason": "archived_destination_unavailable"}

            removed = self._remove_resident_entry(store, target, entry)
            if not removed.get("success"):
                save_receipt(removed=False)
                label = "USER.md" if target == "user" else "MEMORY.md"
                return {
                    "ok": False, "demoted": False, "staged": False, "target": target,
                    "event_id": event_id, "reason": "source_retained",
                    "message": ("事件已写好（id=%d），但 %s 这条没能移除（%s）；"
                                "重跑一次只补删这一步，不会重复写事件。"
                                % (event_id, label, str(removed.get("reason")
                                                        or removed.get("error") or "remove_failed")[:80])),
                }

            def drop_entry(current: dict) -> bool:
                current["demoted"][key] = {**receipt, "removed": True}
                current["entries"].pop(key, None)
                return True
            self._mutate_resident_touch(drop_entry)
            return {
                "ok": True, "demoted": True, "staged": False,
                "target": target, "event_id": event_id,
                "removed": True, "chars": len(entry),
                "message": ("这条过时近况已归档并退居长期事件库，已从 USER.md 移除。"
                            if target == "user" else
                            "这条已退居成事件库里的长期印象，常驻文件里删掉了。"),
            }

        def _link_promoted_user_event(self, content: str, event_id: int,
                                      review_sha: str, now_iso: str) -> None:
            if not review_sha:
                return
            self._sqlite.execute(
                "INSERT OR REPLACE INTO recognition_evidence"
                "(recognition_sha256,event_id,review_sha256,linked_at) VALUES(?,?,?,?)",
                (entry_hash(content), event_id, review_sha, now_iso),
            )

        def promote_reviewed_event(self, event_id: int) -> dict:
            if not self.can_review_private_user():
                return {"ok": False, "promoted": False, "staged": False,
                        "reason": "verified_maintainer_private_chat_required"}
            allowed_owners = self._deps.read_owner_keys(self._deps.owner_key())
            row = self._deps.get_grounded_occurrence(event_id)
            if (row is None or str(row["owner_key"] or "") not in allowed_owners
                    or not self._deps.is_qualified_occurrence(row)):
                return {"ok": False, "promoted": False, "staged": False,
                        "reason": "evidence_source_unqualified"}
            review = self._deps.review_promotion_candidate(event_id, enforce_daily_cap=False)
            proposal = review.get("proposal") or {}
            action = str(proposal.get("action") or "")
            target = str(proposal.get("target") or "")
            now = datetime.now().astimezone()
            now_iso = now.isoformat()
            store = self.store()
            if action == "merge_evidence" and target in ("user", "memory"):
                covered_by = str(proposal.get("covered_by") or "")
                with store.locked(), self._sqlite.transaction():
                    if covered_by not in store.entries(target):
                        return {"ok": False, "promoted": False, "staged": False,
                                "reason": "review_coverage_changed", "proposal": proposal}
                    self._deps.promote_events([event_id], "long", now_iso)
                    if target == "user":
                        self._link_promoted_user_event(
                            covered_by, event_id, str(review.get("review_sha256") or ""), now_iso)
                return {"ok": True, "promoted": True, "written": False, "staged": False,
                        "target": target, "event_id": event_id, "covered_by": covered_by,
                        "proposal": proposal}
            review_sha = str(review.get("review_sha256") or "")
            if action == "propose_new" and target in ("user", "memory") and review_sha:
                content = str(proposal.get("text") or "")
                original_user = list(store.entries("user"))
                original_memory = list(store.entries("memory"))
                evicted: List[str] = []
                if (target == "user"
                        and self._project_user_size(original_user, content, "") > store.char_limit("user")):
                    cap = self._ensure_user_capacity(
                        store, incoming_content=content, replaces="", now=now)
                    if not cap.get("ok"):
                        return {"ok": False, "promoted": False, "staged": False,
                                "reason": str(cap.get("reason") or "resident_capacity_exceeded"),
                                "proposal": proposal}
                    evicted = list(cap.get("evicted") or [])
                with store.locked(), self._sqlite.transaction() as conn:
                    if evicted:
                        expected_user = [entry for entry in original_user if entry not in evicted]
                        if (store.entries("memory") != original_memory
                                or store.entries("user") != expected_user):
                            return {"ok": False, "promoted": False, "staged": False,
                                    "reason": "resident_baseline_changed", "proposal": proposal}
                        conn.execute(
                            "UPDATE event_reviews SET baseline_sha256=? WHERE review_sha256=?",
                            (store.baseline(), review_sha))
                    saved = self._deps.get_event_review(event_id, review_sha)
                    event = conn.execute(
                        "SELECT owner_key,raw,updated_at,created_at,blacklist FROM events "
                        "WHERE id=?", (event_id,)).fetchone()
                    if (saved is None or event is None
                            or str(event["owner_key"] or "") not in allowed_owners
                            or int(event["blacklist"] or 0) != 0
                            or saved["target"] != target or saved["duty"] != target
                            or saved["reviewed_text"] != content
                            or saved["content_sha256"] != entry_hash(content)
                            or saved["event_sha256"] != entry_hash(str(event["raw"] or ""))
                            or saved["event_version"] != (event["updated_at"] or event["created_at"])
                            or saved["baseline_sha256"] != store.baseline()):
                        return {"ok": False, "promoted": False, "staged": False,
                                "reason": "review_binding_invalid", "proposal": proposal}
                    if target == "user":
                        occurrence = self._deps.get_grounded_occurrence(event_id)
                        if occurrence is None or not self._deps.is_qualified_occurrence(occurrence):
                            return {"ok": False, "promoted": False, "staged": False,
                                    "reason": "evidence_source_unqualified", "proposal": proposal}
                    written = store.apply_locked(target, content=content)
                    if not written.get("success"):
                        return {"ok": False, "promoted": False, "staged": False,
                                "reason": written.get("reason"), "proposal": proposal}
                    self._deps.promote_events([event_id], "long", now_iso)
                    if target == "user":
                        self._link_promoted_user_event(content, event_id, review_sha, now_iso)
                return {"ok": True, "promoted": True, "written": True, "staged": False,
                        "target": target, "event_id": event_id, "proposal": proposal}
            return {"ok": False, "promoted": False, "staged": False,
                    "reason": str(review.get("review_skip_reason") or proposal.get("reason")
                                  or "not_long_term_candidate"), "proposal": proposal}
