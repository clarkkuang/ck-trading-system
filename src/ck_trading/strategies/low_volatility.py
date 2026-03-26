"""Low Volatility Strategy.

Buy lowest-volatility stocks (low-vol anomaly).
Academic evidence shows low-vol stocks outperform on a risk-adjusted basis.
Uses daily return standard deviation over a lookback period.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class LowVolatilityStrategy(Strategy):
    """Buy stocks with lowest historical volatility."""

    def __init__(
        self,
        lookback_days: int = 252,
        top_n: int = 20,
    ):
        self.lookback_days = lookback_days
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Low Volatility"

    @property
    def description(self) -> str:
        return (
            "Buy stocks with lowest historical volatility. Academic evidence shows "
            "low-vol stocks outperform on a risk-adjusted basis. Uses daily return "
            "standard deviation over lookback period. Data: price only."
        )

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if prices.is_empty():
            return empty_screen_result()

        cutoff = as_of_date - timedelta(days=self.lookback_days)

        window = prices.filter(
            (pl.col("date") >= cutoff) & (pl.col("date") <= as_of_date)
        )

        if window.is_empty():
            return empty_screen_result()

        # Compute daily returns and annualized volatility per ticker
        returns = (
            window.sort(["ticker", "date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1)
                .alias("daily_return")
            )
            .filter(pl.col("daily_return").is_not_null())
        )

        vol = (
            returns.group_by("ticker")
            .agg(
                pl.col("daily_return").std().alias("daily_std"),
                pl.col("daily_return").count().alias("num_days"),
            )
            .filter(
                (pl.col("daily_std").is_not_null())
                & (pl.col("daily_std") > 0)
                & (pl.col("num_days") >= 20)  # require at least 20 trading days
            )
        )

        if vol.is_empty():
            return empty_screen_result()

        # Annualize and score = 1 / volatility (lower vol = higher score)
        vol = vol.with_columns(
            (pl.col("daily_std") * (252 ** 0.5)).alias("ann_vol"),
        ).with_columns(
            (1.0 / pl.col("ann_vol")).alias("score"),
            pl.lit("BUY").alias("signal_type"),
            (
                pl.lit("Ann. Vol=")
                + (pl.col("ann_vol") * 100).round(1).cast(pl.Utf8)
                + pl.lit("%")
            ).alias("rationale"),
        )

        result = vol.sort("score", descending=True).head(self.top_n)
        return result.select(["ticker", "score", "signal_type", "rationale"])
