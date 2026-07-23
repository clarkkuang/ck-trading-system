"""INTC-specific metric functions + re-exports of shared technicals.

Everything price/technical is shared (ck_trading.monitoring.technicals). This
module owns (1) the Intel fundamentals validator, whose field list is unique
to the foundry-turnaround thesis, and (2) pb_ratio_series — Intel is valued on
Price/Book, not forward P/E (GAAP is distorted by the CHIPS escrow
mark-to-market; non-GAAP forward P/E is ~60x). P/B mirrors the forward_pe
computation but divides weekly close by book_value_per_share.
"""

from __future__ import annotations

import polars as pl

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

# op-loss / growth / fcf / eps CAN go negative (that IS the signal for some);
# external rev / customer count / dcai rev / amd share / book value cannot.
_SIGNED_FIELDS = (
    "foundry_op_loss_billions", "dcai_yoy_growth_pct",
    "adjusted_fcf_billions", "non_gaap_eps",
)
_NON_NEGATIVE_FIELDS = (
    "foundry_external_rev_millions", "foundry_14a_firm_customers",
    "dcai_revenue_billions", "amd_server_rev_share_pct",
    "book_value_per_share",
)
_PCT_RANGE_FIELDS = ("yield_18a_profitable_pct",)


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
    return errors


def _bvps_entries(quarters: list[dict]) -> list[tuple]:
    """Usable (period_start, quarter_label, bvps>0) entries, ascending."""
    out = []
    for q in quarters:
        bvps = q.get("book_value_per_share")
        label = str(q.get("quarter", ""))
        if bvps is None:
            continue
        try:
            bvps_f = float(bvps)
            start = quarter_period_start(label)
        except (TypeError, ValueError):
            continue
        if bvps_f <= 0:
            continue
        out.append((start, label, bvps_f))
    out.sort()
    return out


def pb_ratio_series(
    prices: pl.DataFrame, ticker: str, quarters: list[dict]
) -> pl.DataFrame:
    """Weekly Price/Book: weekly close / latest entered book_value_per_share.

    Backward as-of join of BVPS entries onto weekly closes by period_start.
    Weeks before the first BVPS entry are DROPPED (no fabricated history).
    Empty when no usable BVPS exists => P/B rules report insufficient_data.
    """
    entries = _bvps_entries(quarters)
    if not entries:
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    closes = weekly_close_series(prices, ticker)
    if closes.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)

    rows = []
    for row in closes.iter_rows(named=True):
        bvps = None
        for start, _label, b in entries:
            if start <= row["period_start"]:
                bvps = b
            else:
                break
        if bvps is None:
            continue  # week precedes the first BVPS entry
        rows.append({
            "period_key": row["period_key"],
            "period_start": row["period_start"],
            "value": row["value"] / bvps,
        })
    if not rows:
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    return pl.DataFrame(rows, schema=EMPTY_SERIES_SCHEMA)


def latest_pb_ratio(prices: pl.DataFrame, ticker: str, quarters: list[dict]):
    """(pb, bvps) at the newest week, or None."""
    series = pb_ratio_series(prices, ticker, quarters)
    if series.is_empty():
        return None
    entries = _bvps_entries(quarters)
    latest = series.sort("period_start").tail(1).row(0, named=True)
    return latest["value"], entries[-1][2]
