from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Tuple

from ..infra.clock import shanghai_datetime, sleep_day_iso
from ..storage.curve import HALF_LIFE_MAX_LONG_DAYS

if TYPE_CHECKING:
    from ..storage.sqlite_facade import SqliteFacade

logger = logging.getLogger(__name__)


class _AxisTxAbort(Exception):
    """Internal sentinel to abort a SqliteFacade transaction cleanly."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__()
        self.payload = payload

AXIS_CHANNELS = (
    "delivered_remention",
    "inner_state_flash",
    "explicit_save",
)
_MIN_TEXT = 8
_MIN_SHARED_TRIGRAMS = 4
_MIN_MATCH_RATIO = 0.20
_FLASH_DELTA = 0.45
_FLASH_STRENGTH_FLOOR = 0.55
_AFTERGLOW_STRENGTH = 0.30
_CONFLICT_TENSION = 0.20
_QUOTE_LIMIT = 500
_AXIS_CANDIDATE_LIMIT = 120
# Cap prefilter trigrams so `2 * len(terms)` OR-ed LIKE clauses stay <= 256,
# well below SQLite's default SQLITE_MAX_EXPR_DEPTH (1000).
_MAX_PREFILTER_TERMS: int = 128
_AXIS_EVENT_COLUMNS = ("id, owner_key, raw, summary, tier, created_at, updated_at, "
                       "promoted_reason, shared_view, a_conflict")
_STATE_DIMENSIONS = ("tension", "closeness", "energy", "mood")


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _trigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[index:index + 3] for index in range(max(0, len(compact) - 2))}


def _match_score(query: str, candidate: str) -> Tuple[int, float]:
    left, right = _normalize(query), _normalize(candidate)
    if len(left) < _MIN_TEXT or len(right) < _MIN_TEXT:
        return 0, 0.0
    if left in right or right in left:
        return len(left), 1.0
    left_grams = _trigrams(left)
    if not left_grams:
        return 0, 0.0
    shared = len(left_grams & _trigrams(right))
    return shared, shared / len(left_grams)


def _ordered_trigrams(text: str) -> Tuple[str, ...]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 3:
        return ()
    return tuple(dict.fromkeys(
        compact[index:index + 3] for index in range(len(compact) - 2)
    ))


def _sample_terms(terms: Tuple[str, ...], max_terms: int) -> Tuple[str, ...]:
    if max_terms <= 0:
        raise ValueError("max_terms must be positive")
    total = len(terms)
    if total <= max_terms:
        return terms
    if max_terms == 1:
        return (terms[0],)
    span = total - 1
    denom = max_terms - 1
    return tuple(terms[(index * span) // denom] for index in range(max_terms))


def _prefilter_terms(
    text: str,
    *,
    max_terms: int = _MAX_PREFILTER_TERMS,
) -> Tuple[str, ...]:
    if max_terms <= 0:
        raise ValueError("max_terms must be positive")
    normalized = _normalize(text)
    if len(normalized) < _MIN_TEXT:
        return ()
    ordered = _ordered_trigrams(normalized)
    selected = _sample_terms(ordered, max_terms)
    return tuple(sorted(selected))


def _like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _quote(value: Any) -> str:
    return _normalize(value)[:_QUOTE_LIMIT]


def _quote_sha(value: Any) -> str:
    return hashlib.sha256(_quote(value).encode("utf-8")).hexdigest()


class _StateSnapshot(Protocol):
    @property
    def variables(self) -> Mapping[str, float]: ...

    @property
    def felt_at(self) -> float: ...


AxisState = _StateSnapshot | Mapping[str, object]


class _WriteMaintenanceStatus(Protocol):
    def __call__(
        self, component: str, *, ok: bool, stage: str, now: datetime, error: str = "",
    ) -> None: ...


def _state_variables(state: AxisState | None) -> Optional[Dict[str, float]]:
    if state is None:
        return None
    values = state.get("variables") if isinstance(state, Mapping) else state.variables
    if values is None and isinstance(state, Mapping):
        values = state
    if not isinstance(values, Mapping):
        return None
    out: Dict[str, float] = {}
    for key, value in values.items():
        if key in {"variables", "felt_at", "updated_at"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[str(key)] = float(value)
    if not out:
        return None
    for key in _STATE_DIMENSIONS:
        out.setdefault(key, 0.0)
    return out


def _state_felt_at(state: AxisState | None, fallback: datetime) -> datetime:
    if state is None:
        return fallback
    if isinstance(state, Mapping):
        value = state.get("felt_at") if isinstance(state, dict) else None
    else:
        value = state.felt_at
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    try:
        return shanghai_datetime(float(value))
    except (OverflowError, OSError, ValueError):
        return fallback


class MemoryAxis:
    """Record memory mentions and promotions against the current turn's scope.

    Public worker contracts::

        record_delivered_rementions(*, owner_key, user_text, assistant_text,
                                    turn_id, delivered_at)
        record_inner_state_flash(*, owner_key, user_text, assistant_text,
                                 turn_id, previous_state, current_state, now)

    Both return a JSON-serializable dict.  ``ok=False`` is a real failure: the
    corresponding watermark was not advanced.  ``recorded=False`` with ``ok=True``
    means this turn was already recorded (idempotent replay) or did not deepen.
    """

    def __init__(
        self,
        sqlite: SqliteFacade,
        *,
        current_owner: Callable[[], str],
        daily_owner: Callable[[], str],
        write_owner: Callable[[], str],
        read_owner_keys: Callable[[str], Sequence[str]],
        current_recall_event_ids: Callable[[], tuple[int, ...]],
        read_status: Callable[[], Mapping[str, object]],
        write_maintenance_status: _WriteMaintenanceStatus,
    ) -> None:
        self._sqlite = sqlite
        self._current_owner = current_owner
        self._daily_owner = daily_owner
        self._write_owner = write_owner
        self._read_owner_keys = read_owner_keys
        self._current_recall_event_ids = current_recall_event_ids
        self._read_status = read_status
        self._write_maintenance_status = write_maintenance_status

    def trusted_owner(self, owner_key: str) -> str:
        owner = str(owner_key or "").strip()
        if not owner:
            return ""
        trusted = {self._current_owner(), self._daily_owner()}
        try:
            write_owner = self._write_owner()
        except PermissionError:
            write_owner = ""
        if write_owner:
            trusted.add(write_owner)
        return owner if owner in trusted else ""

    def _axis_note_success(self, channel: str, now: datetime) -> None:
        components = self._read_status().get("components")
        current = components.get("memory_axis") if isinstance(components, dict) else None
        if (isinstance(current, dict) and current.get("ok") is False
                and current.get("stage") == channel):
            self._write_maintenance_status("memory_axis", ok=True, stage=channel, now=now)

    def _axis_note_failure(self, owner_key: str, channel: str, reason: str,
                           now: datetime) -> None:
        self._write_maintenance_status("memory_axis", ok=False, stage=channel or "unknown",
                                       error=reason, now=now)
        if not self._sqlite.is_open or not owner_key or channel not in AXIS_CHANNELS:
            return
        try:
            self._sqlite.execute(
                """INSERT INTO memory_axis_watermarks
                   (owner_key, channel, last_error, failed_count, updated_at)
                   VALUES (?, ?, ?, 1, ?)
                   ON CONFLICT(owner_key, channel) DO UPDATE SET
                     last_error=excluded.last_error,
                     failed_count=memory_axis_watermarks.failed_count + 1,
                     updated_at=excluded.updated_at""",
                (owner_key, channel, str(reason or "write_failed")[:240],
                 now.isoformat(timespec="seconds")),
            )
        except (sqlite3.Error, RuntimeError) as exc:
            logger.warning("sanae_mem: memory-axis failure state not written (%s)",
                           type(exc).__name__)

    @staticmethod
    def _axis_event_score(text: str, row: sqlite3.Row) -> Tuple[int, float]:
        return max(
            _match_score(text, str(row["raw"] or "")),
            _match_score(text, str(row["summary"] or "")),
        )

    def _axis_candidate_rows(
        self,
        owner_key: str,
        text: str,
        earliest: str,
    ) -> list[sqlite3.Row]:
        assert self._sqlite.is_open
        allowed = self._read_owner_keys(owner_key)
        placeholders = ",".join("?" for _ in allowed)
        recall_ids = self._current_recall_event_ids()
        terms: Tuple[str, ...] = _sample_terms(
            _prefilter_terms(text, max_terms=_MAX_PREFILTER_TERMS),
            _MAX_PREFILTER_TERMS,
        )

        def _load(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            rows: list[sqlite3.Row] = []
            seen: set[int] = set()
            for event_id in recall_ids:
                row = conn.execute(
                    f"SELECT {_AXIS_EVENT_COLUMNS} FROM events "
                    f"WHERE id=? AND owner_key IN ({placeholders}) AND blacklist=0",
                    (event_id, *allowed),
                ).fetchone()
                if row is not None and int(row["id"]) not in seen:
                    seen.add(int(row["id"]))
                    rows.append(row)
            if terms:
                pattern = " OR ".join(
                    ["raw LIKE ? ESCAPE '\\'", "summary LIKE ? ESCAPE '\\'"] * len(terms)
                )
                params: list[Any] = [*allowed, earliest]
                for term in terms:
                    params.extend([_like_pattern(term)] * 2)
                params.append(_AXIS_CANDIDATE_LIMIT)
                for row in conn.execute(
                    f"SELECT {_AXIS_EVENT_COLUMNS} FROM events "
                    f"WHERE owner_key IN ({placeholders}) AND blacklist=0 AND created_at>=? "
                    f"AND ({pattern}) ORDER BY id DESC LIMIT ?",
                    params,
                ).fetchall():
                    if int(row["id"]) not in seen:
                        seen.add(int(row["id"]))
                        rows.append(row)
            return rows

        return self._sqlite.read(_load)

    def _axis_find_event(self, owner_key: str, user_text: str, assistant_text: str,
                         now: datetime) -> Optional[sqlite3.Row]:
        assert self._sqlite.is_open
        today = sleep_day_iso(now)
        earliest = (shanghai_datetime(now) - timedelta(days=366)).isoformat()
        current_start = shanghai_datetime(now) - timedelta(hours=6)
        prior: Optional[sqlite3.Row] = None
        prior_score = 0.0
        current: Optional[sqlite3.Row] = None
        current_score = 0.0
        for row in self._axis_candidate_rows(owner_key, _normalize(user_text), earliest):
            created = str(row["created_at"] or "")
            try:
                created_at = shanghai_datetime(datetime.fromisoformat(created))
            except (TypeError, ValueError):
                continue
            shared, ratio = self._axis_event_score(user_text, row)
            matched = shared >= _MIN_SHARED_TRIGRAMS and ratio >= _MIN_MATCH_RATIO
            if created_at < current_start:
                if sleep_day_iso(created_at) >= today or not matched:
                    continue
                if ratio > prior_score or (ratio == prior_score and prior is not None
                                           and int(row["id"]) > int(prior["id"])):
                    prior, prior_score = row, ratio
            elif matched:
                if ratio > current_score or (ratio == current_score and current is not None
                                             and int(row["id"]) > int(current["id"])):
                    current, current_score = row, ratio
        if prior is not None and prior_score >= max(_MIN_MATCH_RATIO, current_score * 0.8):
            return prior
        return current if current is not None else prior

    @staticmethod
    def _axis_stage(owner_key: str, event_id: int, channel: str,
                    current_day: str, conn: sqlite3.Connection) -> str:
        if channel != "delivered_remention":
            return "instant"
        row = conn.execute(
            """SELECT COUNT(DISTINCT sleep_day) AS days,
                      MIN(sleep_day) AS first_day, MAX(sleep_day) AS last_day
               FROM memory_axis_mentions
               WHERE owner_key=? AND event_id=? AND channel=?""",
            (owner_key, event_id, channel),
        ).fetchone()
        days = int(row["days"] or 0)
        if current_day not in {str(row["first_day"] or ""), str(row["last_day"] or "")}:
            existing = conn.execute(
                """SELECT 1 FROM memory_axis_mentions
                   WHERE owner_key=? AND event_id=? AND channel=? AND sleep_day=?
                   LIMIT 1""",
                (owner_key, event_id, channel, current_day),
            ).fetchone()
            if existing is None:
                days += 1
        if days >= 3:
            return "long"
        if days >= 2:
            return "medium"
        return "instant"

    def _axis_promote_locked(self, owner_key: str, event_id: int, channel: str,
                             mention_id: Optional[int], now: datetime,
                             conn: sqlite3.Connection, reason: str = "") -> Dict[str, Any]:
        row = conn.execute(
            """SELECT id, owner_key, tier, updated_at, promoted_reason, candidate_reason
               FROM events WHERE id=? AND blacklist=0""",
            (int(event_id),),
        ).fetchone()
        if row is None:
            return {"ok": False, "reason": "event_not_found"}
        row_owner = str(row["owner_key"] or "")
        if row_owner not in self._read_owner_keys(owner_key):
            return {"ok": False, "reason": "event_owner_mismatch"}
        if str(row["tier"] or "") == "long":
            return {"ok": True, "promoted": False, "reason": "already_long"}
        version = row["updated_at"]
        stamp = now.isoformat(timespec="seconds")
        if reason:
            promoted_reason = reason
        elif channel == "explicit_save":
            promoted_reason = str(row["candidate_reason"] or "") or "manual"
        else:
            promoted_reason = f"memory_axis:{channel}"
        cursor = conn.execute(
            """UPDATE events SET tier='long', promoted_at=?, last_active_at=?,
                      recall_count=MAX(recall_count, 3),
                      half_life_days=?,
                      curve_ttl='', pending_to='', ttl_expire=NULL, expired_at=NULL,
                      promoted_reason=?
               WHERE id=? AND owner_key=? AND blacklist=0 AND tier<>'long'
                 AND (updated_at IS ? )""",
            (stamp, stamp, float(HALF_LIFE_MAX_LONG_DAYS),
             promoted_reason, int(event_id), row_owner, version),
        )
        if cursor.rowcount != 1:
            return {"ok": False, "reason": "event_version_changed"}
        conn.execute(
            """INSERT OR IGNORE INTO memory_axis_promotions
               (owner_key, event_id, channel, mention_id, previous_tier, promoted_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (owner_key, int(event_id), channel, mention_id, str(row["tier"] or ""), stamp),
        )
        return {"ok": True, "promoted": True, "reason": "promoted"}

    @staticmethod
    def _axis_touch_watermark(conn: sqlite3.Connection, owner_key: str, channel: str,
                              delivered_iso: str, turn_id: str, observed_at: datetime) -> None:
        conn.execute(
            """INSERT INTO memory_axis_watermarks
               (owner_key, channel, last_delivered_at, last_turn_id, last_sleep_day,
                last_error, failed_count, updated_at)
               VALUES (?, ?, ?, ?, ?, '', 0, ?)
               ON CONFLICT(owner_key, channel) DO UPDATE SET
                 last_delivered_at=excluded.last_delivered_at,
                 last_turn_id=excluded.last_turn_id,
                 last_sleep_day=excluded.last_sleep_day,
                 last_error='',
                 updated_at=excluded.updated_at""",
            (owner_key, channel, delivered_iso, turn_id, sleep_day_iso(observed_at),
             observed_at.isoformat(timespec="seconds")),
        )

    def _axis_commit_mention(self, *, owner_key: str, turn_id: str, channel: str,
                             event: Optional[sqlite3.Row], user_text: str,
                             assistant_text: str, delivered_at: Optional[datetime],
                             observed_at: datetime, evidence: Dict[str, Any],
                             stage: str) -> Dict[str, Any]:
        assert self._sqlite.is_open
        event_id = int(event["id"]) if event is not None else None
        now_iso = observed_at.isoformat(timespec="seconds")
        delivered_iso = delivered_at.isoformat(timespec="seconds") if delivered_at else now_iso
        user_quote = _quote(user_text)
        assistant_quote = _quote(assistant_text)
        try:
            with self._sqlite.transaction() as conn:
                existing = conn.execute(
                    """SELECT id, event_id, stage FROM memory_axis_mentions
                       WHERE owner_key=? AND turn_id=? AND channel=?""",
                    (owner_key, turn_id, channel),
                ).fetchone()
                if existing is not None:
                    self._axis_touch_watermark(
                        conn, owner_key, channel, delivered_iso, turn_id, observed_at)
                    return {"ok": True, "recorded": False, "reason": "already_recorded",
                            "mention_id": int(existing["id"]),
                            "event_id": int(existing["event_id"]) if existing["event_id"] else None,
                            "stage": str(existing["stage"] or "instant")}
                if event_id is not None:
                    same_day = conn.execute(
                        """SELECT id, stage FROM memory_axis_mentions
                           WHERE owner_key=? AND event_id=? AND channel=? AND sleep_day=?
                           LIMIT 1""",
                        (owner_key, event_id, channel, sleep_day_iso(observed_at)),
                    ).fetchone()
                    if same_day is not None:
                        self._axis_touch_watermark(
                            conn, owner_key, channel, delivered_iso, turn_id, observed_at)
                        return {"ok": True, "recorded": False, "reason": "same_sleep_day",
                                "mention_id": int(same_day["id"]), "event_id": event_id,
                                "stage": str(same_day["stage"] or "instant")}
                cursor = conn.execute(
                    """INSERT INTO memory_axis_mentions
                       (owner_key, turn_id, event_id, channel, sleep_day, observed_at,
                        delivered_at, user_quote, assistant_quote, user_sha256,
                        assistant_sha256, stage, evidence_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (owner_key, turn_id, event_id, channel, sleep_day_iso(observed_at),
                     now_iso, delivered_iso, user_quote, assistant_quote,
                     _quote_sha(user_text), _quote_sha(assistant_text), stage,
                     json.dumps(evidence, ensure_ascii=False, sort_keys=True), now_iso),
                )
                assert cursor.lastrowid is not None
                mention_id = int(cursor.lastrowid)
                promotion: Dict[str, Any] = {"ok": True, "promoted": False, "reason": "not_due"}
                if event_id is not None and (stage == "long" or channel == "inner_state_flash"):
                    promotion = self._axis_promote_locked(
                        owner_key, event_id, channel, mention_id, observed_at, conn)
                    if not promotion.get("ok"):
                        raise _AxisTxAbort({
                            "ok": False,
                            "recorded": False,
                            "reason": str(promotion.get("reason") or "promotion_failed"),
                        })
                self._axis_touch_watermark(
                    conn, owner_key, channel, delivered_iso, turn_id, observed_at)
        except _AxisTxAbort as abort:
            return abort.payload
        return {
            "ok": True, "recorded": True, "mention_id": mention_id,
            "event_id": event_id, "stage": stage,
            "promoted": bool(promotion.get("promoted")),
            "promotion_reason": str(promotion.get("reason") or ""),
        }

    def record_delivered_rementions(self, *, owner_key: str, user_text: str,
                                    assistant_text: str, turn_id: str,
                                    delivered_at: datetime | str | int | float) -> Dict[str, Any]:
        """Record one fully delivered private turn and promote on the third sleep day.

        ``delivered_at`` is mandatory (epoch seconds or ISO-8601).  Missing delivery,
        empty text, an untrusted owner or an unreadable turn id is a real failure and
        leaves the watermark untouched.
        """
        channel = "delivered_remention"
        owner = self.trusted_owner(owner_key)
        text = _normalize(user_text)
        answer = _normalize(assistant_text)
        turn = _normalize(turn_id)
        now = shanghai_datetime()
        if not self._sqlite.is_open or not owner or not text or not answer or not turn:
            reason = ("not_initialized" if not self._sqlite.is_open else
                      "untrusted_owner" if not owner else
                      "empty_text" if not text or not answer else "turn_id_required")
            self._axis_note_failure(owner, channel, reason, now)
            return {"ok": False, "recorded": False, "reason": reason}
        try:
            if isinstance(delivered_at, datetime):
                delivered = shanghai_datetime(delivered_at)
            elif isinstance(delivered_at, (int, float)) and not isinstance(delivered_at, bool):
                delivered = shanghai_datetime(float(delivered_at))
            elif isinstance(delivered_at, str) and delivered_at.strip():
                delivered = shanghai_datetime(datetime.fromisoformat(delivered_at.strip()))
            else:
                raise ValueError("delivery_time_required")
            if delivered > now + timedelta(minutes=5):
                raise ValueError("delivery_time_in_future")
            event = self._axis_find_event(owner, text, answer, delivered)
            event_id = int(event["id"]) if event is not None else None
            stage = "instant"
            if event_id is not None:
                stage = self._sqlite.read(
                    lambda conn: self._axis_stage(
                        owner, event_id, channel, sleep_day_iso(delivered), conn
                    )
                )
            result = self._axis_commit_mention(
                owner_key=owner, turn_id=turn, channel=channel, event=event,
                user_text=text, assistant_text=answer, delivered_at=delivered,
                observed_at=delivered,
                evidence={"kind": "delivered_turn", "event_match": bool(event)},
                stage=stage,
            )
            if not result.get("ok"):
                self._axis_note_failure(owner, channel, str(result.get("reason")), now)
            else:
                self._axis_note_success(channel, now)
            return result
        except (sqlite3.Error, RuntimeError, ValueError, OverflowError, OSError) as exc:
            logger.warning("sanae_mem: delivered remention write failed (%s)",
                           type(exc).__name__)
            self._axis_note_failure(owner, channel, f"{type(exc).__name__}:{exc}", now)
            return {"ok": False, "recorded": False,
                    "reason": f"{type(exc).__name__}:{exc}"}

    def record_inner_state_flash(self, *, owner_key: str, user_text: str,
                                 assistant_text: str, turn_id: str,
                                 previous_state: AxisState | None, current_state: AxisState | None,
                                 now: datetime | int | float | None) -> Dict[str, Any]:
        """Promote on a real inner-state shift, never on a fight or a word list.

        A flash qualifies when an emotional direction moves strongly, or when a strong
        emotional direction survives into a later sleep day above the existing afterglow
        floor.  ``energy`` is physiological and never participates in either decision.
        A strong negative mood is not a quarrel: only a real event appraisal carrying
        conflict evidence plus active tension is filtered as a transient episode.
        """
        channel = "inner_state_flash"
        owner = self.trusted_owner(owner_key)
        text = _normalize(user_text)
        answer = _normalize(assistant_text)
        turn = _normalize(turn_id)
        try:
            current_at = shanghai_datetime(
                now if isinstance(now, datetime)
                or (isinstance(now, (int, float)) and not isinstance(now, bool))
                else None
            )
        except (TypeError, ValueError, OverflowError, OSError):
            current_at = shanghai_datetime()
        if not self._sqlite.is_open or not owner or not turn or not (text or answer):
            reason = ("not_initialized" if not self._sqlite.is_open else
                      "untrusted_owner" if not owner else
                      "turn_id_required" if not turn else "empty_text")
            self._axis_note_failure(owner, channel, reason, current_at)
            return {"ok": False, "recorded": False, "reason": reason}
        previous_vars = _state_variables(previous_state)
        current_vars = _state_variables(current_state)
        if previous_vars is None or current_vars is None:
            self._axis_note_failure(owner, channel, "state_variables_required", current_at)
            return {"ok": False, "recorded": False, "reason": "state_variables_required"}
        previous_at = _state_felt_at(previous_state, current_at)
        current_at = _state_felt_at(current_state, current_at)
        if current_at < previous_at:
            self._axis_note_failure(owner, channel, "state_time_reversed", current_at)
            return {"ok": False, "recorded": False, "reason": "state_time_reversed"}
        keys = set(previous_vars) | set(current_vars)
        deltas = {key: current_vars.get(key, 0.0) - previous_vars.get(key, 0.0)
                  for key in keys}
        emotional_keys = {key for key in keys if key != "energy"}
        peak_delta = max((abs(deltas[key]) for key in emotional_keys), default=0.0)
        strength = max((abs(current_vars.get(key, 0.0)) for key in emotional_keys), default=0.0)
        slept = sleep_day_iso(current_at) > sleep_day_iso(previous_at)
        strong_shift = peak_delta >= _FLASH_DELTA and strength >= _FLASH_STRENGTH_FLOOR
        afterglow = False
        if slept:
            direction_keys = [
                key for key in emotional_keys
                if abs(previous_vars.get(key, 0.0)) >= _FLASH_STRENGTH_FLOOR
                and _AFTERGLOW_STRENGTH <= abs(current_vars.get(key, 0.0))
                < abs(previous_vars.get(key, 0.0))
                and previous_vars.get(key, 0.0) * current_vars.get(key, 0.0) > 0
            ]
            afterglow = bool(direction_keys) and strength >= _AFTERGLOW_STRENGTH
        if not (strong_shift or afterglow):
            self._axis_note_success(channel, current_at)
            return {"ok": True, "recorded": False, "reason": "no_strong_shift",
                    "delta": round(peak_delta, 4), "deltas": {
                        key: round(value, 4) for key, value in deltas.items()},
                    "slept": slept}
        try:
            query = text or answer
            event = self._axis_find_event(owner, query, answer, current_at)
            if event is None:
                self._axis_note_success(channel, current_at)
                return {"ok": True, "recorded": False, "reason": "no_event_evidence",
                        "delta": round(peak_delta, 4), "slept": slept}
            try:
                event_day = sleep_day_iso(datetime.fromisoformat(str(event["created_at"] or "")))
            except (TypeError, ValueError):
                event_day = sleep_day_iso(current_at)
            if afterglow and not strong_shift and event_day >= sleep_day_iso(current_at):
                self._axis_note_success(channel, current_at)
                return {"ok": True, "recorded": False, "reason": "afterglow_without_prior_event",
                        "delta": round(peak_delta, 4), "slept": slept}
            conflict_evidence = int(event["a_conflict"] or 0) >= 2
            if conflict_evidence and current_vars.get("tension", 0.0) > _CONFLICT_TENSION:
                self._axis_note_success(channel, current_at)
                return {"ok": True, "recorded": False, "reason": "conflict_episode",
                        "delta": round(peak_delta, 4), "slept": slept}
            result = self._axis_commit_mention(
                owner_key=owner, turn_id=turn, channel=channel, event=event,
                user_text=text, assistant_text=answer, delivered_at=None,
                observed_at=current_at,
                evidence={"kind": "inner_state_flash", "delta": round(peak_delta, 4),
                          "deltas": {key: round(value, 4) for key, value in deltas.items()},
                          "slept": slept, "previous": previous_vars, "current": current_vars,
                          "conflict_evidence": conflict_evidence},
                stage="instant",
            )
            if not result.get("ok"):
                self._axis_note_failure(owner, channel, str(result.get("reason")), current_at)
            else:
                self._axis_note_success(channel, current_at)
                result["delta"] = round(peak_delta, 4)
                result["deltas"] = {key: round(value, 4) for key, value in deltas.items()}
                result["slept"] = slept
            return result
        except (sqlite3.Error, RuntimeError, ValueError, OverflowError, OSError) as exc:
            logger.warning("sanae_mem: inner-state flash write failed (%s)",
                           type(exc).__name__)
            self._axis_note_failure(owner, channel, f"{type(exc).__name__}:{exc}", current_at)
            return {"ok": False, "recorded": False,
                    "reason": f"{type(exc).__name__}:{exc}"}

    def promote_explicit_save(self, event_id: int, *, owner_key: Optional[str] = None,
                              now: Optional[datetime] = None) -> Dict[str, Any]:
        """Channel 3: the persona's own save tool; only extends the event's lifetime."""
        if not self._sqlite.is_open:
            self._axis_note_failure("", "explicit_save", "not_initialized", shanghai_datetime(now))
            return {"ok": False, "promoted": False, "reason": "not_initialized"}
        owner = self.trusted_owner(owner_key if owner_key is not None else self._current_owner())
        if not owner:
            self._axis_note_failure("", "explicit_save", "untrusted_owner", shanghai_datetime(now))
            return {"ok": False, "promoted": False, "reason": "untrusted_owner"}
        stamp = shanghai_datetime(now)
        try:
            try:
                with self._sqlite.transaction() as conn:
                    result = self._axis_promote_locked(
                        owner, int(event_id), "explicit_save", None, stamp, conn)
                    if not result.get("ok"):
                        raise _AxisTxAbort(result)
                    conn.execute(
                        """INSERT INTO memory_axis_watermarks
                           (owner_key, channel, last_delivered_at, last_turn_id,
                            last_sleep_day, last_error, failed_count, updated_at)
                           VALUES (?, 'explicit_save', ?, '', ?, '', 0, ?)
                           ON CONFLICT(owner_key, channel) DO UPDATE SET
                             last_delivered_at=excluded.last_delivered_at,
                             last_sleep_day=excluded.last_sleep_day,
                             last_error='', updated_at=excluded.updated_at""",
                        (owner, stamp.isoformat(timespec="seconds"),
                         sleep_day_iso(stamp), stamp.isoformat(timespec="seconds")),
                    )
            except _AxisTxAbort as abort:
                result = abort.payload
            if not result.get("ok"):
                self._axis_note_failure(owner, "explicit_save",
                                        str(result.get("reason")), stamp)
                return {**result, "promoted": False}
            self._axis_note_success("explicit_save", stamp)
            return result
        except (sqlite3.Error, RuntimeError, TypeError, ValueError, OverflowError, OSError) as exc:
            logger.warning("sanae_mem: explicit-save promotion failed (%s)",
                           type(exc).__name__)
            self._axis_note_failure(owner, "explicit_save", f"{type(exc).__name__}:{exc}", stamp)
            return {"ok": False, "promoted": False,
                    "reason": f"{type(exc).__name__}:{exc}"}
