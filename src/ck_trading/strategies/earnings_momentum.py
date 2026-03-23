"""Earnings Momentum strategy.

Buys stocks with recent positive earnings surprises.
Requires extra_data["earnings"] with columns:
  ticker, report_date, surprise_pct
"""

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.base import Strategy
from ck_trading.strategies.registry import register


@register
class EarningsMomentumStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Earnings Momentum"

    @property
    def description(self) -> str:
        return "Buy stocks with recent positive earnings surprises"

    @property
    def rebalance_frequency(self) -> str:
        return "monthly"

    def __init__(
        self,
        surprise_threshold: float = 0.05,
        lookback_days: int = 60,
        top_n: int = 20,
    ):
        self.surprise_threshold = surprise_threshold
        self.lookback_days = lookback_days
        self.top_n = top_n

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if extra_data is None or "earnings" not in extra_data:
            return _empty_result()

        earnings = extra_data["earnings"]
        if earnings.is_empty():
            return _empty_result()

        required = {"ticker", "report_date", "surprise_pct"}
        if not required.issubset(set(earnings.columns)):
            return _empty_result()

        cutoff_date = as_of_date - timedelta(days=self.lookback_days)

        # Filter to recent earnings within lookback window
        recent = earnings.filter(
            (pl.col("report_date") >= cutoff_date)
            & (pl.col("report_date") <= as_of_date)
            & (pl.col("surprise_pct").is_not_null())
            & (pl.col("surprise_pct") > self.surprise_threshold)
        )

        if recent.is_empty():
            return _empty_result()

        # Take latest surprise per ticker
        latest = (
            recent.sort("report_date", descending=True)
            .group_by("ticker")
            .first()
        )

        latest = latest.with_columns(
            pl.col("surprise_pct").alias("score"),
            pl.lit("BUY").alias("signal_type"),
            (
                pl.lit("Earnings surprise: +")
                + (pl.col("surprise_pct") * 100).round(1).cast(pl.Utf8)
                + pl.lit("%")
            ).alias("rationale"),
        )

        result = latest.sort("score", descending=True).head(self.top_n)
        return result.select(["ticker", "score", "signal_type", "rationale"])


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "score": pl.Float64,
            "signal_type": pl.Utf8,
            "rationale": pl.Utf8,
        }
    )
