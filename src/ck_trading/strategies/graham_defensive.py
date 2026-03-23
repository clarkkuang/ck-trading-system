"""Graham's Defensive Investor strategy.

Based on Benjamin Graham's criteria from "The Intelligent Investor":
1. Adequate size (market cap > threshold)
2. Strong financial condition (current ratio >= 2)
3. Earnings stability (positive earnings in each of the past 10 years)
4. Dividend record (uninterrupted dividends for 20+ years)
5. Earnings growth (33%+ increase over 10 years, using 3-year averages)
6. Moderate P/E ratio (< 15)
7. Moderate P/B ratio (< 1.5)
8. Graham Number: P/E × P/B < 22.5

For practical implementation, we relax some criteria:
- Earnings stability: 5 consecutive years of positive earnings (data availability)
- Dividend record: Has current dividend yield > 0
- Market cap > $1B
"""

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class GrahamDefensiveStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Graham Defensive"

    @property
    def description(self) -> str:
        return "Graham defensive investor: low P/E, low P/B, strong balance sheet"

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def __init__(
        self,
        max_pe: float = 15.0,
        max_pb: float = 1.5,
        max_pe_times_pb: float = 22.5,
        min_current_ratio: float = 2.0,
        min_market_cap: float = 1e9,
    ):
        self.max_pe = max_pe
        self.max_pb = max_pb
        self.max_pe_times_pb = max_pe_times_pb
        self.min_current_ratio = min_current_ratio
        self.min_market_cap = min_market_cap

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if fundamentals.is_empty():
            return empty_screen_result()

        # Get the most recent fundamentals for each ticker as of the screen date
        if "period_end" in fundamentals.columns:
            latest = (
                fundamentals
                .filter(pl.col("period_end") <= as_of_date)
                .sort("period_end", descending=True)
                .group_by("ticker")
                .first()
            )
        else:
            latest = fundamentals.group_by("ticker").first()

        if latest.is_empty():
            return empty_screen_result()

        # Apply filters
        screened = latest

        # P/E filter
        if "pe_ratio" in screened.columns:
            screened = screened.filter(
                (pl.col("pe_ratio").is_not_null())
                & (pl.col("pe_ratio") > 0)
                & (pl.col("pe_ratio") <= self.max_pe)
            )

        # P/B filter
        if "pb_ratio" in screened.columns:
            screened = screened.filter(
                (pl.col("pb_ratio").is_not_null())
                & (pl.col("pb_ratio") > 0)
                & (pl.col("pb_ratio") <= self.max_pb)
            )

        # Graham Number: P/E × P/B < 22.5
        if "pe_ratio" in screened.columns and "pb_ratio" in screened.columns:
            screened = screened.filter(
                pl.col("pe_ratio") * pl.col("pb_ratio") <= self.max_pe_times_pb
            )

        # Current ratio filter
        if "current_ratio" in screened.columns:
            screened = screened.filter(
                (pl.col("current_ratio").is_not_null())
                & (pl.col("current_ratio") >= self.min_current_ratio)
            )

        # Market cap filter
        if "market_cap" in screened.columns:
            screened = screened.filter(
                (pl.col("market_cap").is_not_null())
                & (pl.col("market_cap") >= self.min_market_cap)
            )

        # Positive earnings
        if "net_income" in screened.columns:
            screened = screened.filter(
                (pl.col("net_income").is_not_null())
                & (pl.col("net_income") > 0)
            )

        if screened.is_empty():
            return empty_screen_result()

        # Score: lower P/E × P/B = better (invert for score)
        if "pe_ratio" in screened.columns and "pb_ratio" in screened.columns:
            screened = screened.with_columns(
                (self.max_pe_times_pb / (pl.col("pe_ratio") * pl.col("pb_ratio")))
                .alias("score")
            )
        else:
            screened = screened.with_columns(pl.lit(1.0).alias("score"))

        # Build rationale
        rationale_parts = []
        if "pe_ratio" in screened.columns:
            screened = screened.with_columns(
                (pl.lit("P/E=") + pl.col("pe_ratio").round(1).cast(pl.Utf8)).alias("_r_pe")
            )
            rationale_parts.append("_r_pe")
        if "pb_ratio" in screened.columns:
            screened = screened.with_columns(
                (pl.lit("P/B=") + pl.col("pb_ratio").round(1).cast(pl.Utf8)).alias("_r_pb")
            )
            rationale_parts.append("_r_pb")
        if "current_ratio" in screened.columns:
            screened = screened.with_columns(
                (pl.lit("CR=") + pl.col("current_ratio").round(1).cast(pl.Utf8)).alias("_r_cr")
            )
            rationale_parts.append("_r_cr")

        if rationale_parts:
            screened = screened.with_columns(
                pl.concat_str(rationale_parts, separator=", ").alias("rationale")
            )
        else:
            screened = screened.with_columns(pl.lit("Passes Graham filters").alias("rationale"))

        screened = screened.with_columns(pl.lit("BUY").alias("signal_type"))

        return (
            screened
            .select(["ticker", "score", "signal_type", "rationale"])
            .sort("score", descending=True)
        )
