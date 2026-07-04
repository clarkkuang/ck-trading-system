"""Shared quarter / scenario helpers for monitor instances.

Promoted from att_metrics.py when the third monitor instance (NVDA) arrived
— rule of three. att_metrics re-exports these names so existing imports and
tests keep working unchanged.

Everything here is field-agnostic: quarter labels ("2026Q3"), generic
fundamentals series extraction, and valuation-scenario math/validation.
Monitor-specific validators (field names, sign constraints) stay in each
monitor's own metrics module.
"""

from __future__ import annotations

import re
from datetime import date

import polars as pl

QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")

EMPTY_SERIES_SCHEMA = {
    "period_key": pl.Utf8,
    "period_start": pl.Date,
    "value": pl.Float64,
}


def quarter_period_start(quarter: str) -> date:
    """'2026Q3' -> date(2026, 7, 1). Raises ValueError on a bad label."""
    m = QUARTER_RE.match(quarter.strip())
    if not m:
        raise ValueError(f"Bad quarter label {quarter!r}; expected YYYYQ[1-4]")
    year, q = int(m.group(1)), int(m.group(2))
    return date(year, 3 * (q - 1) + 1, 1)


def fundamentals_series(quarters: list[dict], field: str) -> pl.DataFrame:
    """[period_key, period_start, value] for one fundamentals field.

    Rows whose field is null/missing or whose quarter label is malformed
    are dropped. Sorted ascending by derived period_start.
    """
    rows = []
    for q in quarters:
        val = q.get(field)
        label = q.get("quarter", "")
        if val is None:
            continue
        try:
            start = quarter_period_start(label)
        except ValueError:
            continue
        rows.append({
            "period_key": label,
            "period_start": start,
            "value": float(val),
        })
    if not rows:
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    return pl.DataFrame(rows, schema=EMPTY_SERIES_SCHEMA).sort("period_start")


def weighted_fair_value(scenarios: list[dict]) -> float:
    """Probability-weighted implied price: Σ p/100 × price."""
    return sum(
        float(s.get("probability_pct", 0)) / 100.0 * float(s.get("implied_price", 0))
        for s in scenarios
    )


def validate_scenarios(scenarios: list[dict]) -> list[str]:
    """Return a list of human-readable problems (empty = valid)."""
    errors: list[str] = []
    ids: set[str] = set()
    total = 0.0
    for s in scenarios:
        sid = str(s.get("id", "")).strip()
        if not sid:
            errors.append("有情景缺少 id")
        elif sid in ids:
            errors.append(f"情景 id {sid!r} 重复")
        ids.add(sid)

        try:
            p = float(s.get("probability_pct", 0))
        except (TypeError, ValueError):
            errors.append(f"{sid}: 概率不是数字")
            continue
        if not 0 <= p <= 100:
            errors.append(f"{sid}: 概率 {p} 超出 [0,100]")
        total += p

        try:
            low = float(s.get("price_low", 0))
            implied = float(s.get("implied_price", 0))
            high = float(s.get("price_high", 0))
        except (TypeError, ValueError):
            errors.append(f"{sid}: 价格字段不是数字")
            continue
        if not (0 < low <= implied <= high):
            errors.append(
                f"{sid}: 需满足 0 < low({low}) ≤ implied({implied}) ≤ high({high})"
            )
    if abs(total - 100.0) >= 0.01:
        errors.append(f"概率之和为 {total:.2f}%,必须等于 100%")
    return errors
