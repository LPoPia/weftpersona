from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Iterator, Optional

CURVE_BASE_DAYS = 6.0
CURVE_GROWTH_EXPONENT = 0.7
CURVE_TTL_SPAN = 2.0
CURVE_MIN_GAP = timedelta(hours=12)
CURVE_COUNTED_VIAS = frozenset({"chat", "explicit_past"})
CURVE_WAKE_VIAS = frozenset({"chat", "explicit_past", "tool_search"})

CURVE_TTL_NATURAL_SQL = "curve_ttl<>'' AND curve_ttl=ttl_expire AND expired_at IS NULL"

HALF_LIFE_MIN_DAYS = 9.0
HALF_LIFE_MAX_LONG_DAYS = 120.0
HALF_LIFE_MAX_SHORT_DAYS = 60.0
SHORT_TIERS = frozenset({"short", "pending"})


def half_life_ceiling(tier: Optional[str]) -> float:
    """该层级允许的半衰期上界（短期/待定及未设层级均按短期 60，其余长期层为 120）。"""
    normalized = str(tier or "").strip() or "short"
    return HALF_LIFE_MAX_SHORT_DAYS if normalized in SHORT_TIERS else HALF_LIFE_MAX_LONG_DAYS


def bounded_half_life(value: float, tier: Optional[str], *, floor: bool = True) -> float:
    """将半衰期钳制到层级允许范围（短期/待定 <= 60 天，其余 <= 120 天；floor=True 时下限 9 天）。

    ``floor=False`` 只压上界（存量修复或保留更短显式寿命时使用：不抬高已有低值）。
    """
    current = min(float(value or 0.0), half_life_ceiling(tier))
    if floor:
        current = max(HALF_LIFE_MIN_DAYS, current)
    return current


def _iter_over_cap_rows(conn: sqlite3.Connection) -> Iterator[tuple[int, float]]:
    for event_id, tier, half_life in conn.execute(
        "SELECT id, tier, half_life_days FROM events WHERE half_life_days > ?",
        (HALF_LIFE_MAX_SHORT_DAYS,),
    ):
        ceiling = half_life_ceiling(tier)
        if float(half_life) > ceiling:
            yield int(event_id), ceiling


def has_over_cap_half_life(conn: sqlite3.Connection) -> bool:
    """Return whether ``events`` contains any row exceeding its tier half-life ceiling."""
    return next(_iter_over_cap_rows(conn), None) is not None


def clamp_stored_half_life_ceilings(conn: sqlite3.Connection) -> int:
    """Clamp stored ``half_life_days`` ceilings (``floor=False``) in a single pass per row."""
    updates = [(ceiling, event_id) for event_id, ceiling in _iter_over_cap_rows(conn)]
    if not updates:
        return 0
    conn.executemany("UPDATE events SET half_life_days = ? WHERE id = ?", updates)
    return len(updates)


def initial_half_life(base_days: float, importance: float, affect: int = 0) -> float:
    """S0 = d_base * (1 + 2.5*importance) * (1 + 0.5*affect_norm)。"""
    affect_norm = min(1.0, max(0.0, float(affect or 0) / 3.0))
    value = float(base_days) * (1.0 + 2.5 * float(importance)) * (1.0 + 0.5 * affect_norm)
    return max(1.0, value)


def strengthened_half_life(half_life_days: float, recall_count: int) -> float:
    """从存量 S_{n-1} 推到 S_n = S0 * (1+n)^0.7，不需要另存 S0。"""
    previous = max(0, int(recall_count or 0))
    current = float(half_life_days or 0.0)
    if current <= 0:
        current = initial_half_life(CURVE_BASE_DAYS, 0.5)
    base = current / ((1.0 + previous) ** CURVE_GROWTH_EXPONENT)
    return base * ((1.0 + previous + 1) ** CURVE_GROWTH_EXPONENT)


def terminal_ttl(now: datetime, half_life_days: float) -> datetime:
    return now + timedelta(days=CURVE_TTL_SPAN * max(0.0, float(half_life_days or 0.0)))


def is_curve_natural(ttl_expire: Optional[str], curve_ttl: Optional[str],
                     expired_at: Optional[str]) -> bool:
    """行级判定，与 `CURVE_TTL_NATURAL_SQL` 同义。"""
    ttl = str(ttl_expire or "")
    return bool(ttl) and ttl == str(curve_ttl or "") and not expired_at
