from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

CURVE_BASE_DAYS = 6.0
CURVE_GROWTH_EXPONENT = 0.7
CURVE_TTL_SPAN = 2.0
CURVE_MIN_GAP = timedelta(hours=12)
CURVE_PROMOTE_RECALLS = 3
CURVE_COUNTED_VIAS = frozenset({"chat", "explicit_past"})
CURVE_WAKE_VIAS = frozenset({"chat", "explicit_past", "tool_search"})

CURVE_TTL_NATURAL_SQL = "curve_ttl<>'' AND curve_ttl=ttl_expire AND expired_at IS NULL"

# 曲线管理记忆的既定寿命范围：长期层 9～120 天，短期/待定层不超过 60 天。
# 写侧、强化、晋升与存量修复都从这里取同一组上界；存量修复只压不抬（见 bounded_half_life）。
HALF_LIFE_MIN_DAYS = 9.0
HALF_LIFE_MAX_LONG_DAYS = 120.0
HALF_LIFE_MAX_SHORT_DAYS = 60.0
SHORT_TIERS = frozenset({"short", "pending"})


def half_life_ceiling(tier: Optional[str]) -> float:
    """该层级允许的半衰期上界（短期/待定为 60，其余长期层为 120）。"""
    return HALF_LIFE_MAX_SHORT_DAYS if str(tier or "") in SHORT_TIERS else HALF_LIFE_MAX_LONG_DAYS


def bounded_half_life(value: float, tier: Optional[str], *, floor: bool = True) -> float:
    """把半衰期钳进既定范围。``floor=False`` 只压上界（存量修复用：不抬高旧行寿命）。"""
    current = min(float(value or 0.0), half_life_ceiling(tier))
    if floor:
        current = max(HALF_LIFE_MIN_DAYS, current)
    return current


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
