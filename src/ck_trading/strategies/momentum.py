"""Momentum Strategy.

Classic cross-sectional momentum: buy recent winners, sell recent losers.
Uses a lookback window with a skip period to avoid short-term reversal effects.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class MomentumStrategy(Strategy):
    """Cross-sectional momentum: long winners, short losers."""

    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        top_pct: float = 0.20,
        bottom_pct: float = 0.20,
        min_history_days: int = 200,
    ):
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.min_history_days = min_history_days

    @property
    def name(self) -> str:
        return "Momentum"

    @property
    def description(self) -> str:
        return (
            f"Cross-sectional momentum: {self.lookback_months}M lookback with "
            f"{self.skip_months}M skip period. Long top {self.top_pct:.0%}, short bottom "
            f"{self.bottom_pct:.0%} of universe. Data: price only."
        )

    @property
    def rebalance_frequency(self) -> str:
        return "monthly"

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if prices.is_empty():
            return empty_screen_result()

        # Compute window boundaries
        lookback_start = _subtract_months(as_of_date, self.lookback_months)
        skip_end = _subtract_months(as_of_date, self.skip_months)

        # Filter prices to the lookback window (excluding the skip period)
        window = prices.filter(
            (pl.col("date") >= lookback_start) & (pl.col("date") <= skip_end)
        )

        if window.is_empty():
            return empty_screen_result()

        # For each ticker: get first and last close in the window
        per_ticker = (
            window.sort("date")
            .group_by("ticker")
            .agg(
                pl.col("close").first().alias("first_close"),
                pl.col("close").last().alias("last_close"),
                pl.col("date").count().alias("num_days"),
            )
        )

        # Filter tickers with insufficient history
        per_ticker = per_ticker.filter(pl.col("num_days") >= self.min_history_days)

        if per_ticker.is_empty():
            return empty_screen_result()

        # Compute return
        per_ticker = per_ticker.with_columns(
            ((pl.col("last_close") - pl.col("first_close")) / pl.col("first_close"))
            .alias("momentum_return")
        )

        # Rank
        n = per_ticker.height
        per_ticker = per_ticker.with_columns(
            pl.col("momentum_return").rank("ordinal").alias("rank")
        )

        top_cutoff = max(1, int(n * (1 - self.top_pct)))
        bottom_cutoff = max(1, int(n * self.bottom_pct))

        # Assign signals
        per_ticker = per_ticker.with_columns(
            pl.when(pl.col("rank") > top_cutoff)
            .then(pl.lit("BUY"))
            .when(pl.col("rank") <= bottom_cutoff)
            .then(pl.lit("SELL"))
            .otherwise(pl.lit("HOLD"))
            .alias("signal_type")
        )

        # Filter to only BUY / SELL
        result = per_ticker.filter(pl.col("signal_type") != "HOLD")

        if result.is_empty():
            return empty_screen_result()

        # Score = momentum return (higher = better for BUY; for SELL, we still
        # use the return so that the most negative are "strongest" sells)
        result = result.with_columns(
            pl.col("momentum_return").alias("score"),
            (
                pl.lit(f"{self.lookback_months}M Return=")
                + (pl.col("momentum_return") * 100).round(1).cast(pl.Utf8)
                + pl.lit("% (Rank ")
                + pl.col("rank").cast(pl.Utf8)
                + pl.lit("/")
                + pl.lit(str(n))
                + pl.lit(")")
            ).alias("rationale"),
        )

        return result.select(["ticker", "score", "signal_type", "rationale"])


def _subtract_months(d: date, months: int) -> date:
    """Subtract *months* from *d*, clamping the day to fit the new month."""
    import calendar

    month = d.month - months
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)
