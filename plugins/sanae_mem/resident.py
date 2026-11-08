from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hermes_cli.active_sessions import _FileLock
from utils import atomic_json_write

try:
    from .textutil import _embed, _fingerprint, _pack_vector, _tokens, provider_home
except ImportError:
    from textutil import _embed, _fingerprint, _pack_vector, _tokens, provider_home

try:
    from .event_store import SUMMARY_SOURCE_GIST_FADED
except ImportError:
    from event_store import SUMMARY_SOURCE_GIST_FADED

logger = logging.getLogger(__name__)

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


def _load_resident_store():
    try:
        from tools.memory_tool import load_on_disk_store
    except Exception:
        return None
    try:
        return load_on_disk_store()
    except Exception:
        return None


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


class ResidentMixin:
        def _resident_touch_path(self) -> Path:
            return Path(self._data_dir) / TOUCH_FILE

        def _read_resident_touch(self) -> dict:
            path = self._resident_touch_path()
            try:
                value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except Exception:
                value = {}
            if not isinstance(value, dict):
                value = {}
            for key in ("entries", "demoted"):
                if not isinstance(value.get(key), dict):
                    value[key] = {}
            return value

        def _mutate_resident_touch(self, change: Callable[[dict], bool]) -> Optional[dict]:
            path = self._resident_touch_path()
            with self._lock:
                with _FileLock(path.with_suffix(".lock")):
                    state = self._read_resident_touch()
                    try:
                        if not change(state):
                            return state
                        path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_json_write(path, state)
                    except Exception as exc:
                        logger.warning(
                            "sanae_mem: resident_touch write failed (%s)", type(exc).__name__)
                        return None
                    return state

        def _resident_writes_blocked(self, session_id: str = "", *, maintenance: bool = False) -> str:
            nightly = maintenance and self._agent_context == "cron" and bool(self._daily_owner_key)
            if self._agent_context != "primary" and not nightly:
                return "non_primary_context"
            try:
                from .source_context import in_public_interaction
            except ImportError:
                from source_context import in_public_interaction
            if in_public_interaction():
                return "public_interaction"
            bound = str(getattr(self, "_session_id", "") or "")
            session = str(session_id or bound)
            if not nightly and (self._is_system_turn(session) or self._is_system_turn(bound)):
                return "system_turn"
            if not nightly and self._daily_owner_key and self._daily_owner_key != self._owner_key:
                return "different_owner"
            return ""

        @staticmethod
        def _resident_match(turn_tokens: set, compact_turn: str, entry: str) -> Tuple[bool, int]:
            overlap = len(turn_tokens & set(_tokens(entry)))
            if overlap >= TOUCH_MIN_OVERLAP:
                return True, overlap
            compact_entry = re.sub(r"\s+", "", str(entry or ""))
            if len(compact_entry) < 3:
                return False, overlap
            shingles = {compact_entry[i:i + 3] for i in range(len(compact_entry) - 2)}
            hit = sum(1 for shingle in shingles if shingle in compact_turn)
            return hit / len(shingles) >= TOUCH_SHINGLE_RATIO, overlap

        def _update_resident_touch(
            self, user_content: str, assistant_content: str, *,
            session_id: str = "", now: Optional[datetime] = None,
        ) -> int:
            try:
                if self._resident_writes_blocked(session_id):
                    return 0
                store = _load_resident_store()
                if store is None:
                    return 0
                text = f"{user_content or ''} {assistant_content or ''}"
                turn_tokens = set(_tokens(text))
                now = now or datetime.now().astimezone()
                compact_turn = re.sub(r"\s+", "", text)
                return self._reconcile_resident_entries(store, now, turn_tokens, compact_turn)
            except Exception as exc:
                logger.warning(
                    "sanae_mem: resident touch update failed (%s)", type(exc).__name__)
                return 0

        def _reconcile_resident_entries(self, store, now, tokens=(), turn="") -> int:
            stamp = now.isoformat()
            matched = set()
            current = {}
            for target in ("memory", "user"):
                if not store.target_enabled(target):
                    continue
                for entry in store._entries_for(target):
                    key = _entry_key(target, entry)
                    current[key] = (target, entry[:40])
                    if tokens and self._resident_match(tokens, turn, entry)[0]:
                        matched.add(key)

            def reconcile(state):
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

        def _pending_demote_receipts(self) -> set:
            try:
                from tools import write_approval as wa
                records = wa.list_pending(wa.MEMORY)
            except Exception:
                return set()
            waiting = set()
            for record in records:
                payload = record.get("payload") or {}
                source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
                if str(source.get("kind") or "") != DEMOTE_SOURCE_KIND:
                    continue
                digest = str(source.get("entry_sha") or "")
                if digest:
                    waiting.add(digest)
            return waiting

        def _resident_cold_report(self, now: datetime) -> dict:
            capacity = self._memory_capacity_snapshot()
            hot = [
                target for target in ("memory", "user")
                if float(capacity.get(target + "_pct") or 0.0) >= CAPACITY_PCT
            ]
            cold: List[dict] = []
            store = _load_resident_store()
            if store is not None and not self._resident_writes_blocked(maintenance=True):
                self._reconcile_resident_entries(store, now)
            if hot:
                state = self._read_resident_touch()
                waiting = self._pending_demote_receipts()
                for target in hot:
                    if store is None or not store.target_enabled(target):
                        continue
                    for entry in store._entries_for(target):
                        digest = _entry_sha(entry)
                        if digest in waiting:
                            continue
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
            archive_dir = Path(provider_home(self)) / "memory-archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            path = archive_dir / f"{target}-demoted.jsonl"
            line = json.dumps({
                "archived_at": now.isoformat(), "target": target, "entry_sha": sha,
                "reason": DEMOTE_SOURCE_KIND, "entry": entry,
            }, ensure_ascii=False)
            try:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                logger.warning(
                    "sanae_mem: resident demote archive failed (%s)", type(exc).__name__)
                return False
            return True

        def _insert_demoted_event(self, target: str, entry: str, sha: str,
                                  last_active: str, now: datetime) -> Optional[int]:
            assert self._conn is not None
            body = re.sub(r"\s+", " ", entry).strip()
            if not body:
                return None
            try:
                from .source_context import self_attribution, user_fact_attribution
            except ImportError:
                from source_context import self_attribution, user_fact_attribution
            attribution = user_fact_attribution if target == "user" else self_attribution
            owner = self._owner_key or self._daily_owner_key
            subject, fact, involved, scope = attribution(owner)
            topic = self._topic(body)
            ts = now.strftime("%Y-%m-%d %H:%M")
            stamp = now.isoformat()
            summary = body[:150]
            vector = _embed(body)
            values: Dict[str, Any] = {
                "ts": ts, "raw": entry if target == "user" else f"[{ts}] {entry}", "summary": summary,
                "summary_source": SUMMARY_SOURCE_GIST_FADED,
                "keywords": self._keywords(body), "importance": DEMOTED_IMPORTANCE,
                "confidence": 0.6, "topic": topic, "ttl_expire": None,
                "created_at": stamp, "updated_at": stamp, "vec": _pack_vector(vector),
                "session_id": self._session_id, "tier": DEMOTED_TIER,
                "promoted_at": stamp, "promoted_reason": DEMOTED_PROMOTED_REASON,
                "asked": 1, "asked_at": stamp, "owner_key": owner,
                "experience_subject": subject, "fact_subject": fact,
                "involved_owner_keys": involved, "read_scope": scope,
            }
            values.update(last_active_at=last_active or stamp, recall_count=DEMOTED_RECALL_COUNT,
                          half_life_days=DEMOTED_HALF_LIFE_DAYS, original_fingerprint=_fingerprint(body))
            columns = list(values)
            with self._lock:
                if self._conn.execute(
                        "SELECT 1 FROM forgotten WHERE fingerprint=?", (_fingerprint(body),)).fetchone():
                    return None
                cursor = self._conn.execute(
                    "INSERT INTO events (%s) VALUES (%s)" % (
                        ", ".join(columns), ", ".join("?" for _ in columns)),
                    [values[name] for name in columns],
                )
                event_id = int(cursor.lastrowid)
                self._conn.commit()
            self._upsert_chroma(event_id, summary, topic, DEMOTED_IMPORTANCE, ts, vector)
            return event_id

        @staticmethod
        def _locate_resident_entry(entries: List[str], locator: str):
            from tools.memory_tool_store import _locate_entry

            idx, ambiguous, _ = _locate_entry(list(entries), str(locator or ""))
            return idx, ambiguous

        def _remove_resident_entry(self, store, target: str, entry: str) -> dict:
            try:
                from tools.memory_tool_store import ExpectedBaseline, entry_hash
            except Exception:
                return {"success": False, "reason": "memory_store_unavailable"}
            try:
                result = store.remove(
                    target, entry, expect=ExpectedBaseline(entry_hash=entry_hash(entry)))
            except Exception as exc:
                return {"success": False, "reason": type(exc).__name__}
            return result if isinstance(result, dict) else {"success": False, "reason": "remove_failed"}

        def _stage_user_demote(self, target: str, entry: str, sha: str, event_id: int,
                               receipt: dict, now: datetime) -> dict:
            from tools import write_approval as wa
            from tools.memory_tool_store import entry_hash

            pending_id = str(receipt.get("proposal_id") or "")
            if not pending_id:
                for pending in wa.list_pending(wa.MEMORY):
                    payload = pending.get("payload") or {}
                    if payload.get("demoted_event_id") == event_id and payload.get("content") == entry:
                        pending_id = pending["id"]
                        break
            if pending_id and wa.get_pending(wa.MEMORY, pending_id):
                return {
                    "ok": True, "demoted": False, "staged": True, "already_staged": True,
                    "target": target, "event_id": event_id, "proposal_id": pending_id,
                    "reason": "proposal_pending",
                    "message": "这条已经出过提案，还在等确认；没有重复写事件。",
                }
            payload = {
                "action": "migrate", "target": target, "destination": "event",
                "content": entry, "authorization": "migration",
                "baseline": {"entry_hash": entry_hash(entry)},
                "target_id": event_id, "demoted_event_id": event_id,
                "source": {
                    "reason": "migration_required", "kind": DEMOTE_SOURCE_KIND,
                    "entry_sha": sha, "demoted_event_id": event_id,
                    "session": str(getattr(self, "_session_id", "") or ""),
                    "user": str(getattr(self, "_maintainer_id", "") or ""),
                    "staged_seq": 0, "destination": "event", "destination_qualified": False,
                    "removed_entries": [entry],
                    "staged_at": now.isoformat(),
                },
            }
            record = wa.stage_write(
                wa.MEMORY, payload,
                summary=("退居一条长期没被聊到的 USER 条目到事件库：%s" % entry[:60]),
                origin=wa.current_origin())
            if not record.get("staged"):
                return {
                    "ok": False, "demoted": False, "staged": False, "target": target,
                    "event_id": event_id, "reason": "proposal_stage_failed",
                    "message": "事件已落库留底，但退居提案没能写入待审批目录；USER.md 未动。",
                }
            return {
                "ok": True, "demoted": False, "staged": True, "target": target,
                "event_id": event_id, "proposal_id": str(record.get("id") or ""),
                "reason": "migration_required",
                "message": ("USER.md 的条目受损耗保护，只出了退居提案：事件已在库里留底，"
                            "确认后才会从 USER.md 移除（/memory pending 查看）。"),
            }

        def demote_resident_entry(
            self, target: str, entry_text: str, now: Optional[datetime] = None,
        ) -> dict:
            """把一条常驻条目退居成事件库里的长期印象，再从常驻文件移除。

            顺序固定：核对条目在册 → 原文归档 → 写长期印象事件 → 才动常驻文件。
            MEMORY 由她自己维护，事件落库后按 CAS 直接移除；USER 受损耗保护，
            只落一条可审批的迁移提案，绝不硬删。
            """
            if self._conn is None:
                raise RuntimeError("provider is not initialized")
            target = str(target or "").strip().lower()
            if target not in ("memory", "user"):
                raise ValueError("target must be memory or user")
            blocked = self._resident_writes_blocked(maintenance=True)
            if blocked:
                return _blocked(blocked)
            with _FileLock(self._resident_touch_path().with_suffix(".demote.lock")):
                return self._demote_locked(target, entry_text, now or datetime.now().astimezone())

        def _demote_locked(self, target: str, entry_text: str, now: datetime) -> dict:
            store = _load_resident_store()
            if store is None:
                return _blocked("memory_store_unavailable")
            if not store.target_enabled(target):
                return _blocked("target_disabled")
            entries = list(store._entries_for(target))
            idx, ambiguous = self._locate_resident_entry(entries, entry_text)
            if idx is None:
                return {"ok": False, "demoted": False,
                        "reason": "entry_ambiguous" if ambiguous else "entry_not_found"}
            entry = entries[idx]
            try:
                from tools.memory_tool_store import RETENTION_MARK
            except Exception:
                RETENTION_MARK = ""
            if RETENTION_MARK and RETENTION_MARK in entry:
                return {"ok": False, "demoted": False, "reason": "retention_protected"}

            sha = _entry_sha(entry)
            key = f"{target}:{sha}"
            state = self._read_resident_touch()
            record = state["entries"].get(key) or {}
            receipt = dict(state["demoted"].get(key) or {})
            last_active = str(record.get("last_matched_at") or "") or now.isoformat()

            event_id = receipt.get("event_id")
            if not isinstance(event_id, int) or self.get_event_row_by_id(event_id) is None:
                event_id = None
            if event_id is None:
                owner = self._owner_key or self._daily_owner_key
                existing = self._conn.execute(
                    "SELECT id FROM events WHERE promoted_reason=? AND owner_key=? AND original_fingerprint=? "
                    "AND fact_subject=? AND blacklist=0 ORDER BY id DESC LIMIT 1",
                    (DEMOTED_PROMOTED_REASON, owner, _fingerprint(entry),
                     "owner:" + owner if target == "user" else "sanae")).fetchone()
                event_id = int(existing[0]) if existing else None
            if event_id is None:
                if not self._archive_demoted_entry(target, entry, sha, now):
                    return {"ok": False, "demoted": False, "reason": "archive_failed"}
                event_id = self._insert_demoted_event(target, entry, sha, last_active, now)
                if event_id is None:
                    return {"ok": False, "demoted": False, "reason": "event_write_failed"}
                receipt = {
                    "target": target, "event_id": event_id, "entry_sha": sha,
                    "demoted_at": now.isoformat(), "proposal_id": "",
                }

            def save_receipt(**fields: Any) -> None:
                def change(current: dict) -> bool:
                    merged = dict(receipt)
                    merged.update(fields)
                    current["demoted"][key] = merged
                    return True
                self._mutate_resident_touch(change)

            if target == "user":
                staged = self._stage_user_demote(target, entry, sha, event_id, receipt, now)
                save_receipt(
                    proposal_id=str(staged.get("proposal_id") or receipt.get("proposal_id") or ""),
                    staged_at=now.isoformat())
                return staged

            removed = self._remove_resident_entry(store, target, entry)
            if not removed.get("success"):
                save_receipt(removed=False)
                return {
                    "ok": False, "demoted": False, "target": target, "event_id": event_id,
                    "reason": "source_retained",
                    "message": ("事件已写好（id=%d），但 MEMORY.md 这条没能移除（%s）；"
                                "重跑一次只补删这一步，不会重复写事件。"
                                % (event_id, str(removed.get("reason")
                                                 or removed.get("error") or "remove_failed")[:80])),
                }

            def drop_entry(current: dict) -> bool:
                current["demoted"][key] = {**receipt, "removed": True}
                current["entries"].pop(key, None)
                return True
            self._mutate_resident_touch(drop_entry)
            return {
                "ok": True, "demoted": True, "target": target, "event_id": event_id,
                "removed": True, "chars": len(entry),
                "message": "这条已退居成事件库里的长期印象，常驻文件里删掉了。",
            }

        def _tool_compact(self, args: Dict[str, Any]) -> dict:
            target = str(args.get("demote_target") or "").strip()
            if not target:
                return self.compact()
            entry = str(args.get("demote_entry") or "").strip()
            if not entry:
                return {"ok": False, "demoted": False, "reason": "demote_entry_required"}
            return self.demote_resident_entry(target, entry)
