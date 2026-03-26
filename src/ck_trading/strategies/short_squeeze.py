"""Short Squeeze Strategy.

Detect potential short squeezes: high short interest combined with recent
upward momentum. Targets stocks where short sellers may be forced to cover.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class ShortSqueezeStrategy(Strategy):
    """High short interest + recent price momentum = squeeze candidate."""

    def __init__(
        self,
        min_short_ratio: float = 0.15,
        min_momentum_pct: float = 0.05,
        lookback_days: int = 5,
        top_n: int = 10,
    ):
        self.min_short_ratio = min_short_ratio
        self.min_momentum_pct = min_momentum_pct
        self.lookback_days = lookback_days
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Short Squeeze"

    @property
    def description(self) -> str:
        return (
            "Detect potential short squeezes: high short interest + recent upward "
            "momentum. Targets stocks where short sellers may be forced to cover. "
            "Data: FINRA short interest + price."
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
        if extra_data is None or "short_interest" not in extra_data:
            return empty_screen_result()

        si = extra_data["short_interest"]
        if si.is_empty() or prices.is_empty():
            return empty_screen_result()

        # Determine the short_ratio column name
        if "short_ratio" not in si.columns:
            if "short_volume" in si.columns and "total_volume" in si.columns:
                si = si.with_columns(
                    (pl.col("short_volume") / pl.col("total_volume")).alias("short_ratio")
                )
            else:
                return empty_screen_result()

        if "ticker" not in si.columns or "date" not in si.columns:
            return empty_screen_result()

        # Get latest short_ratio per ticker
        latest_si = (
            si.filter(pl.col("date") <= as_of_date)
            .sort("date", descending=True)
            .group_by("ticker")
            .first()
            .filter(pl.col("short_ratio") >= self.min_short_ratio)
        )

        if latest_si.is_empty():
            return empty_screen_result()

        # Compute N-day price momentum
        cutoff = as_of_date - timedelta(days=self.lookback_days * 2)  # buffer
        window = prices.filter(
            (pl.col("date") >= cutoff) & (pl.col("date") <= as_of_date)
        )

        if window.is_empty():
            return empty_screen_result()

        # Get first and last close over lookback to compute momentum
        momentum = (
            window.sort(["ticker", "date"])
            .group_by("ticker")
            .agg(
                pl.col("close").last().alias("last_close"),
                pl.col("close").first().alias("first_close"),
            )
            .with_columns(
                ((pl.col("last_close") - pl.col("first_close")) / pl.col("first_close"))
                .alias("momentum")
            )
            .filter(pl.col("momentum") >= self.min_momentum_pct)
        )

        if momentum.is_empty():
            return empty_screen_result()

        # Join short interest with momentum
        combined = latest_si.select(["ticker", "short_ratio"]).join(
            momentum.select(["ticker", "momentum"]),
            on="ticker",
            how="inner",
        )

        if combined.is_empty():
            return empty_screen_result()

        combined = combined.with_columns(
            (pl.col("short_ratio") * pl.col("momentum")).alias("score"),
            pl.lit("BUY").alias("signal_type"),
            (
                pl.lit("Short ratio=")
                + (pl.col("short_ratio") * 100).round(1).cast(pl.Utf8)
                + pl.lit("%, ")
                + pl.lit("Momentum=")
                + (pl.col("momentum") * 100).round(1).cast(pl.Utf8)
                + pl.lit("%")
            ).alias("rationale"),
        )

        result = combined.sort("score", descending=True).head(self.top_n)
        return result.select(["ticker", "score", "signal_type", "rationale"])
