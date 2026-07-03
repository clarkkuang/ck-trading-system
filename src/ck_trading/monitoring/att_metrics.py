"""Pure metric functions for the AT&T vs SpaceX monitor. No I/O."""

from __future__ import annotations

import datetime as dt
import re
from datetime import date
from typing import Sequence

import polars as pl

_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")

_EMPTY_SERIES_SCHEMA = {
    "period_key": pl.Utf8,
    "period_start": pl.Date,
    "value": pl.Float64,
}


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Weekly prices
# ---------------------------------------------------------------------------
def compute_weekly_closes(daily: pl.DataFrame) -> pl.DataFrame:
    """Daily OHLCV rows -> one row per (ticker, iso_week): the last close.

    Input needs [ticker, date, close]; output:
    [ticker, iso_week, week_start, price_date, close, collected_at].
    A partial current week produces a provisional row that later runs
    replace via the store's keep-last dedupe.
    """
    empty = pl.DataFrame(schema={
        "ticker": pl.Utf8,
        "iso_week": pl.Utf8,
        "week_start": pl.Date,
        "price_date": pl.Date,
        "close": pl.Float64,
        "collected_at": pl.Datetime("us"),
    })
    if daily.is_empty():
        return empty

    df = daily.filter(
        pl.col("close").is_not_null() & pl.col("close").is_not_nan()
    )
    if df.is_empty():
        return empty

    df = df.with_columns(
        (
            pl.col("date").dt.iso_year().cast(pl.Utf8)
            + pl.lit("-W")
            + pl.col("date").dt.week().cast(pl.Utf8).str.zfill(2)
        ).alias("iso_week"),
        (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday() - 1))
        .alias("week_start"),
    )
    weekly = (
        df.sort("date")
        .group_by(["ticker", "iso_week", "week_start"])
        .agg(
            pl.col("date").last().alias("price_date"),
            pl.col("close").last().cast(pl.Float64),
        )
        .with_columns(
            pl.lit(_utcnow()).cast(pl.Datetime("us")).alias("collected_at")
        )
        .select("ticker", "iso_week", "week_start", "price_date", "close",
                "collected_at")
        .sort(["ticker", "week_start"])
    )
    return weekly


def weekly_close_series(prices: pl.DataFrame, ticker: str) -> pl.DataFrame:
    """[period_key, period_start, value] for one ticker's weekly closes."""
    if prices.is_empty():
        return pl.DataFrame(schema=_EMPTY_SERIES_SCHEMA)
    sub = prices.filter(pl.col("ticker") == ticker)
    if sub.is_empty():
        return pl.DataFrame(schema=_EMPTY_SERIES_SCHEMA)
    return (
        sub.select(
            pl.col("iso_week").alias("period_key"),
            pl.col("week_start").alias("period_start"),
            pl.col("close").alias("value"),
        )
        .unique(subset=["period_key"], keep="last")
        .sort("period_start")
    )


def normalized_performance(
    prices: pl.DataFrame,
    tickers: Sequence[str],
    lookback_weeks: int | None = None,
) -> pl.DataFrame:
    """Weekly closes normalized to 100 at each ticker's first week in window.

    Returns [ticker, week_start, norm]. Tickers with shorter history (e.g. a
    recent IPO) align from their own first available week.
    """
    empty = pl.DataFrame(schema={
        "ticker": pl.Utf8, "week_start": pl.Date, "norm": pl.Float64,
    })
    if prices.is_empty():
        return empty

    sub = prices.filter(pl.col("ticker").is_in(list(tickers)))
    if sub.is_empty():
        return empty

    if lookback_weeks is not None:
        cutoff = sub["week_start"].max() - dt.timedelta(weeks=lookback_weeks)
        sub = sub.filter(pl.col("week_start") >= cutoff)
        if sub.is_empty():
            return empty

    return (
        sub.sort(["ticker", "week_start"])
        .with_columns(
            (pl.col("close") / pl.col("close").first().over("ticker") * 100)
            .alias("norm")
        )
        .select("ticker", "week_start", "norm")
    )


# ---------------------------------------------------------------------------
# Quarterly fundamentals
# ---------------------------------------------------------------------------
def quarter_period_start(quarter: str) -> date:
    """'2026Q3' -> date(2026, 7, 1). Raises ValueError on a bad label."""
    m = _QUARTER_RE.match(quarter.strip())
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
        return pl.DataFrame(schema=_EMPTY_SERIES_SCHEMA)
    return pl.DataFrame(rows, schema=_EMPTY_SERIES_SCHEMA).sort("period_start")


def validate_fundamentals(quarters: list[dict]) -> list[str]:
    """Return a list of human-readable problems (empty = valid)."""
    errors: list[str] = []
    seen: set[str] = set()
    for i, q in enumerate(quarters):
        label = str(q.get("quarter", "")).strip()
        if not _QUARTER_RE.match(label):
            errors.append(f"第{i + 1}行: 季度标签 {label!r} 不符合 YYYYQ1-4 格式")
            continue
        if label in seen:
            errors.append(f"季度 {label} 重复")
        seen.add(label)
        for fld in ("churn_pct", "convergence_pct", "fcf_billions",
                    "capex_billions", "net_debt_ebitda"):
            val = q.get(fld)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                errors.append(f"{label}: {fld} 不是数字 ({val!r})")
                continue
            if v < 0:
                errors.append(f"{label}: {fld} 为负数 ({v})")
    return errors


# ---------------------------------------------------------------------------
# Valuation scenarios
# ---------------------------------------------------------------------------
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
