"""Composite Value Score strategy.

Multi-factor scoring that combines multiple value signals:
- Low P/E (earnings cheapness)
- Low P/B (asset cheapness)
- Low P/S (revenue cheapness)
- High ROE (profitability)
- High current ratio (financial strength)
- Low debt/equity (leverage safety)
- Positive earnings growth

Each factor is ranked and z-scored within the universe,
then combined with configurable weights.
"""

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class CompositeValueStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Composite Value"

    @property
    def description(self) -> str:
        return "Multi-factor value scoring: cheapness + quality + safety"

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        top_n: int = 20,
        min_market_cap: float = 5e8,
    ):
        self.weights = weights or {
            "pe_score": 0.20,       # Earnings cheapness
            "pb_score": 0.15,       # Asset cheapness
            "ps_score": 0.10,       # Revenue cheapness
            "roe_score": 0.20,      # Profitability
            "cr_score": 0.10,       # Financial strength
            "de_score": 0.10,       # Leverage safety
            "margin_score": 0.15,   # Operating quality
        }
        self.top_n = top_n
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

        # Get latest fundamentals
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

        if latest.is_empty() or latest.height < 5:
            return empty_screen_result()

        # Market cap filter
        if "market_cap" in latest.columns:
            latest = latest.filter(
                (pl.col("market_cap").is_not_null())
                & (pl.col("market_cap") >= self.min_market_cap)
            )

        # Filter: need positive earnings
        if "pe_ratio" in latest.columns:
            latest = latest.filter(
                (pl.col("pe_ratio").is_not_null()) & (pl.col("pe_ratio") > 0)
            )

        if latest.is_empty() or latest.height < 3:
            return empty_screen_result()

        # Score each factor using percentile rank (0 to 1)
        scored = latest

        # Lower P/E = better (invert rank)
        if "pe_ratio" in scored.columns:
            scored = scored.with_columns(
                (1 - pl.col("pe_ratio").rank("ordinal") / pl.col("pe_ratio").count())
                .alias("pe_score")
            )
        else:
            scored = scored.with_columns(pl.lit(0.5).alias("pe_score"))

        # Lower P/B = better
        if "pb_ratio" in scored.columns:
            scored = scored.with_columns(
                (1 - pl.col("pb_ratio").fill_null(pl.col("pb_ratio").median())
                 .rank("ordinal") / scored.height)
                .alias("pb_score")
            )
        else:
            scored = scored.with_columns(pl.lit(0.5).alias("pb_score"))

        # Lower P/S = better
        if "ps_ratio" in scored.columns:
            scored = scored.with_columns(
                (1 - pl.col("ps_ratio").fill_null(pl.col("ps_ratio").median())
                 .rank("ordinal") / scored.height)
                .alias("ps_score")
            )
        else:
            scored = scored.with_columns(pl.lit(0.5).alias("ps_score"))

        # Higher ROE = better
        if "roe" in scored.columns:
            scored = scored.with_columns(
                (pl.col("roe").fill_null(0).rank("ordinal") / scored.height)
                .alias("roe_score")
            )
        else:
            scored = scored.with_columns(pl.lit(0.5).alias("roe_score"))

        # Higher current ratio = better (capped benefit)
        if "current_ratio" in scored.columns:
            scored = scored.with_columns(
                (pl.col("current_ratio").fill_null(1.0).clip(0, 5)
                 .rank("ordinal") / scored.height)
                .alias("cr_score")
            )
        else:
            scored = scored.with_columns(pl.lit(0.5).alias("cr_score"))

        # Lower debt/equity = better
        if "debt_to_equity" in scored.columns:
            scored = scored.with_columns(
                (1 - pl.col("debt_to_equity").fill_null(pl.col("debt_to_equity").median())
                 .clip(0, 10).rank("ordinal") / scored.height)
                .alias("de_score")
            )
        else:
            scored = scored.with_columns(pl.lit(0.5).alias("de_score"))

        # Higher operating margin = better
        if "operating_margin" in scored.columns:
            scored = scored.with_columns(
                (pl.col("operating_margin").fill_null(0).rank("ordinal") / scored.height)
                .alias("margin_score")
            )
        else:
            scored = scored.with_columns(pl.lit(0.5).alias("margin_score"))

        # Weighted composite score
        score_expr = pl.lit(0.0)
        for factor, weight in self.weights.items():
            if factor in scored.columns:
                score_expr = score_expr + pl.col(factor) * weight

        scored = scored.with_columns(score_expr.alias("score"))

        # Take top N
        top = scored.sort("score", descending=True).head(self.top_n)

        # Build rationale
        ratio_cols = []
        for col_name, _label in [
            ("pe_ratio", "P/E"), ("pb_ratio", "P/B"), ("roe", "ROE"),
        ]:
            if col_name in top.columns:
                ratio_cols.append(col_name)

        if ratio_cols:
            parts = []
            for col_name in ratio_cols:
                label = {"pe_ratio": "P/E", "pb_ratio": "P/B", "roe": "ROE"}.get(col_name, col_name)
                top = top.with_columns(
                    (pl.lit(f"{label}=") + pl.col(col_name).round(1).cast(pl.Utf8))
                    .alias(f"_r_{col_name}")
                )
                parts.append(f"_r_{col_name}")
            top = top.with_columns(
                pl.concat_str(parts, separator=", ", ignore_nulls=True)
                .fill_null("Composite value score")
                .alias("rationale")
            )
        else:
            top = top.with_columns(pl.lit("Composite value score").alias("rationale"))

        top = top.with_columns(pl.lit("BUY").alias("signal_type"))

        return top.select(["ticker", "score", "signal_type", "rationale"])
