"""NFLX-specific metric functions + re-exports of shared technicals.

Everything price/technical is shared (ck_trading.monitoring.technicals);
this module only owns the NFLX fundamentals validator, whose field list and
sign rules differ from the other monitors.
"""

from __future__ import annotations

from ck_trading.monitoring.quarters import (  # noqa: F401
    EMPTY_SERIES_SCHEMA,
    QUARTER_RE,
    quarter_period_start,
)
from ck_trading.monitoring.technicals import (  # noqa: F401
    daily_indicators,
    normalize_daily,
    normalized_performance,
    weekly_close_series,
    weekly_sample,
)

# revenue growth CAN go negative (that is exactly the bear signal).
_SIGNED_FIELDS = ("revenue_growth_pct", "next_q_guide_growth_pct")
_NON_NEGATIVE_FIELDS = ("fcf_billions", "buyback_billions")
_PCT_RANGE_FIELDS = ("op_margin_pct",)
_BINARY_FIELDS = ("ads_on_track",)


def validate_fundamentals(quarters: list[dict]) -> list[str]:
    """Return a list of human-readable problems (empty = valid)."""
    errors: list[str] = []
    seen: set[str] = set()
    for i, q in enumerate(quarters):
        label = str(q.get("quarter", "")).strip()
        if not QUARTER_RE.match(label):
            errors.append(f"第{i + 1}行: 季度标签 {label!r} 不符合 YYYYQ1-4 格式")
            continue
        if label in seen:
            errors.append(f"季度 {label} 重复")
        seen.add(label)

        all_fields = (
            _SIGNED_FIELDS + _NON_NEGATIVE_FIELDS
            + _PCT_RANGE_FIELDS + _BINARY_FIELDS
        )
        for fld in all_fields:
            val = q.get(fld)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                errors.append(f"{label}: {fld} 不是数字 ({val!r})")
                continue
            if fld in _NON_NEGATIVE_FIELDS and v < 0:
                errors.append(f"{label}: {fld} 为负数 ({v})")
            if fld in _PCT_RANGE_FIELDS and not 0 <= v <= 100:
                errors.append(f"{label}: {fld} 超出 [0,100] ({v})")
            if fld in _BINARY_FIELDS and v not in (0.0, 1.0):
                errors.append(f"{label}: {fld} 只能是 0 或 1 ({v})")
    return errors
