"""Shared technical/price metric functions for monitors. No I/O.

Promoted from nvda_metrics when the NFLX monitor became the second
consumer (rule of three, same pattern as quarters.py). Daily price
storage + weekly sampling: rolling windows use TRADING-DAY rows;
rule evaluation happens on weekly series so the shared engine's
max_gap_days=14 semantics carry over unchanged.
"""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import polars as pl

from ck_trading.monitoring.quarters import (
    EMPTY_SERIES_SCHEMA,
    QUARTER_RE,
    quarter_period_start,
)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Daily prices
# ---------------------------------------------------------------------------
def normalize_daily(daily: pl.DataFrame) -> pl.DataFrame:
    """Raw collector rows -> [ticker, date, close, collected_at].

    Drops null/NaN closes, dedupes (ticker, date) keep-last, sorts.
    """
    empty = pl.DataFrame(schema={
        "ticker": pl.Utf8,
        "date": pl.Date,
        "close": pl.Float64,
        "collected_at": pl.Datetime("us"),
    })
    if daily.is_empty() or "close" not in daily.columns:
        return empty
    df = daily.filter(
        pl.col("close").is_not_null() & pl.col("close").is_not_nan()
    )
    if df.is_empty():
        return empty
    return (
        df.select(
            pl.col("ticker"),
            pl.col("date"),
            pl.col("close").cast(pl.Float64),
        )
        .unique(subset=["ticker", "date"], keep="last")
        .with_columns(
            pl.lit(_utcnow()).cast(pl.Datetime("us")).alias("collected_at")
        )
        .sort(["ticker", "date"])
    )


def daily_indicators(
    prices: pl.DataFrame,
    ticker: str,
    *,
    dip_lookback: int = 60,
    sma_period: int = 200,
) -> pl.DataFrame:
    """Technical indicator table for one ticker (trading-day row windows).

    Returns [date, close, high_60d, dip_pct, sma_200, pct_vs_200dma,
    gated_dip] sorted by date. Warmup rows (insufficient window) are NULL —
    not 0.0 — so they drop out of weekly sampling instead of reading as
    "no dip"/"on trend".

    gated_dip = dip_pct if close > sma_200 else 0.0 (strict >). Null while
    sma_200 is null.
    """
    empty = pl.DataFrame(schema={
        "date": pl.Date, "close": pl.Float64, "high_60d": pl.Float64,
        "dip_pct": pl.Float64, "sma_200": pl.Float64,
        "pct_vs_200dma": pl.Float64, "gated_dip": pl.Float64,
    })
    if prices.is_empty():
        return empty
    sub = (
        prices.filter(pl.col("ticker") == ticker)
        .unique(subset=["date"], keep="last")
        .sort("date")
    )
    if sub.is_empty():
        return empty

    out = sub.select("date", "close").with_columns(
        pl.col("close").rolling_max(dip_lookback, min_samples=dip_lookback)
        .alias("high_60d"),
        pl.col("close").rolling_mean(sma_period, min_samples=sma_period)
        .alias("sma_200"),
    ).with_columns(
        ((pl.col("high_60d") - pl.col("close")) / pl.col("high_60d"))
        .alias("dip_pct"),
        (pl.col("close") / pl.col("sma_200") - 1).alias("pct_vs_200dma"),
    ).with_columns(
        pl.when(pl.col("sma_200").is_null())
        .then(None)
        .when(pl.col("close") > pl.col("sma_200"))
        .then(pl.col("dip_pct"))
        .otherwise(0.0)
        .alias("gated_dip")
    )
    return out.select(
        "date", "close", "high_60d", "dip_pct", "sma_200",
        "pct_vs_200dma", "gated_dip",
    )


def weekly_sample(
    df: pl.DataFrame, value_col: str, *, date_col: str = "date"
) -> pl.DataFrame:
    """Daily rows -> weekly series [period_key, period_start, value].

    Takes the LAST non-null row per ISO week. period_key format matches the
    att monitor byte-for-byte ("2026-W27"); period_start = Monday.
    """
    if df.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    sub = df.filter(
        pl.col(value_col).is_not_null() & pl.col(value_col).is_not_nan()
    )
    if sub.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    return (
        sub.sort(date_col)
        .with_columns(
            (
                pl.col(date_col).dt.iso_year().cast(pl.Utf8)
                + pl.lit("-W")
                + pl.col(date_col).dt.week().cast(pl.Utf8).str.zfill(2)
            ).alias("period_key"),
            (pl.col(date_col)
             - pl.duration(days=pl.col(date_col).dt.weekday() - 1))
            .alias("period_start"),
        )
        .group_by("period_key", "period_start")
        .agg(pl.col(value_col).last().cast(pl.Float64).alias("value"))
        .sort("period_start")
        .select("period_key", "period_start", "value")
    )


def weekly_close_series(prices: pl.DataFrame, ticker: str) -> pl.DataFrame:
    """Weekly close series for one ticker (charts + P/E base)."""
    if prices.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    sub = prices.filter(pl.col("ticker") == ticker)
    if sub.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    return weekly_sample(sub, "close")


def normalized_performance(
    prices: pl.DataFrame,
    tickers: Sequence[str],
    lookback_weeks: int | None = None,
) -> pl.DataFrame:
    """Weekly closes normalized to 100 at each ticker's first week in window.

    Returns [ticker, week_start, norm].
    """
    empty = pl.DataFrame(schema={
        "ticker": pl.Utf8, "week_start": pl.Date, "norm": pl.Float64,
    })
    frames = []
    for t in tickers:
        s = weekly_close_series(prices, t)
        if s.is_empty():
            continue
        frames.append(s.with_columns(pl.lit(t).alias("ticker")))
    if not frames:
        return empty
    allw = pl.concat(frames)
    if lookback_weeks is not None:
        cutoff = allw["period_start"].max() - dt.timedelta(weeks=lookback_weeks)
        allw = allw.filter(pl.col("period_start") >= cutoff)
        if allw.is_empty():
            return empty
    return (
        allw.sort(["ticker", "period_start"])
        .with_columns(
            (pl.col("value") / pl.col("value").first().over("ticker") * 100)
            .alias("norm")
        )
        .select("ticker", pl.col("period_start").alias("week_start"), "norm")
    )


# ---------------------------------------------------------------------------
# Forward P/E (the semi-auto valuation layer)
# ---------------------------------------------------------------------------
def _eps_entries(quarters: list[dict]) -> list[tuple[dt.date, str, float]]:
    """Usable (period_start, quarter_label, eps) entries, ascending."""
    out = []
    for q in quarters:
        eps = q.get("forward_eps_consensus")
        label = str(q.get("quarter", ""))
        if eps is None:
            continue
        try:
            eps_f = float(eps)
            start = quarter_period_start(label)
        except (TypeError, ValueError):
            continue
        if eps_f <= 0:
            continue
        out.append((start, label, eps_f))
    out.sort()
    return out


def forward_pe_series(
    prices: pl.DataFrame, ticker: str, quarters: list[dict]
) -> pl.DataFrame:
    """Weekly forward P/E: weekly close / latest entered forward EPS.

    Backward as-of join of EPS entries onto weekly closes by period_start.
    Weeks before the first EPS entry are DROPPED (no fabricated history).
    Empty when no usable EPS exists => P/E rules report insufficient_data.
    """
    entries = _eps_entries(quarters)
    if not entries:
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    closes = weekly_close_series(prices, ticker)
    if closes.is_empty():
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)

    rows = []
    for row in closes.iter_rows(named=True):
        eps = None
        for start, _label, e in entries:
            if start <= row["period_start"]:
                eps = e
            else:
                break
        if eps is None:
            continue  # week precedes the first EPS entry
        rows.append({
            "period_key": row["period_key"],
            "period_start": row["period_start"],
            "value": row["value"] / eps,
        })
    if not rows:
        return pl.DataFrame(schema=EMPTY_SERIES_SCHEMA)
    return pl.DataFrame(rows, schema=EMPTY_SERIES_SCHEMA)


def latest_forward_eps(quarters: list[dict]) -> tuple[str, float] | None:
    """(quarter_label, eps) of the newest usable EPS entry, or None."""
    entries = _eps_entries(quarters)
    if not entries:
        return None
    _start, label, eps = entries[-1]
    return label, eps


