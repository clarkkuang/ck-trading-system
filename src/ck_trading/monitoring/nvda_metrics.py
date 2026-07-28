"""NVDA-specific metric functions + re-exports of shared technicals.

The generic price/technical/P-E functions moved to
ck_trading.monitoring.technicals when the NFLX monitor became the second
consumer. Re-exported here so existing imports and tests stay valid.
"""

from __future__ import annotations

from ck_trading.monitoring.quarters import (  # noqa: F401
    EMPTY_SERIES_SCHEMA,
    QUARTER_RE,
    quarter_period_start,
)
from ck_trading.monitoring.technicals import (  # noqa: F401
    daily_indicators,
    forward_pe_series,
    latest_forward_eps,
    normalize_daily,
    normalized_performance,
    weekly_close_series,
    weekly_sample,
)

# Fields where a NEGATIVE value is meaningful signal, not bad data.
# exposure ratio deliberately NOT range-capped: guarantees can exceed 100%
# of TTM revenue (the OpenAI $250B talks alone imply ~77%).
_SIGNED_FIELDS = ("dc_qoq_growth_pct", "guide_vs_consensus_pct")
_NON_NEGATIVE_FIELDS = (
    "dc_revenue_billions", "total_revenue_billions", "forward_eps_consensus",
    "ar_days", "customer_financing_exposure_pct",
)
_PCT_RANGE_FIELDS = ("gross_margin_pct", "asic_server_share_pct")


# ---------------------------------------------------------------------------
# Fundamentals validation (NVDA fields; negative growth values ALLOWED)
# ---------------------------------------------------------------------------
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

        all_fields = _SIGNED_FIELDS + _NON_NEGATIVE_FIELDS + _PCT_RANGE_FIELDS
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
            # _SIGNED_FIELDS: negatives are legitimate signal (guide miss /
            # revenue decline) — no sign check.
    return errors
