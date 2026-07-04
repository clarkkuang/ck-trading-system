"""Pure metric functions for the AT&T vs SpaceX monitor. No I/O.

Generic quarter/scenario helpers now live in ck_trading.monitoring.quarters
(promoted when the third monitor instance arrived); they are re-exported
here so existing imports keep working.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Sequence

import polars as pl

from ck_trading.monitoring.quarters import (  # noqa: F401 — re-exports
    EMPTY_SERIES_SCHEMA as _EMPTY_SERIES_SCHEMA,
    QUARTER_RE as _QUARTER_RE,
    fundamentals_series,
    quarter_period_start,
    validate_scenarios,
    weighted_fair_value,
)


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
