"""Greenblatt's Magic Formula strategy.

From "The Little Book That Beats the Market":
1. Rank all stocks by Earnings Yield = EBIT / Enterprise Value (higher is better)
2. Rank by Return on Capital = EBIT / (Net Fixed Assets + Working Capital)
3. Combine ranks (sum of both ranks, lower combined rank is better)
4. Buy top 20-30 stocks, rebalance annually

Simplified implementation:
- Earnings Yield = EBIT / EV (or use E/P if EBIT unavailable)
- ROC = EBIT / (Total Assets - Current Liabilities - Cash)
- Exclude financials and utilities (different capital structures)
"""

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class MagicFormulaStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Magic Formula"

    @property
    def description(self) -> str:
        return "Greenblatt's Magic Formula: high earnings yield + high return on capital"

    @property
    def rebalance_frequency(self) -> str:
        return "annual"

    def __init__(self, top_n: int = 30, min_market_cap: float = 1e8):
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

        if latest.is_empty():
            return empty_screen_result()

        # Filter: need EBIT and EV data
        available = set(latest.columns)

        # Compute Earnings Yield
        if "ebit" in available and "enterprise_value" in available:
            latest = latest.filter(
                (pl.col("ebit").is_not_null())
                & (pl.col("enterprise_value").is_not_null())
                & (pl.col("enterprise_value") > 0)
                & (pl.col("ebit") > 0)
            )
            latest = latest.with_columns(
                (pl.col("ebit") / pl.col("enterprise_value")).alias("earnings_yield")
            )
        elif "pe_ratio" in available:
            # Fallback: E/P ratio
            latest = latest.filter(
                (pl.col("pe_ratio").is_not_null()) & (pl.col("pe_ratio") > 0)
            )
            latest = latest.with_columns(
                (1.0 / pl.col("pe_ratio")).alias("earnings_yield")
            )
        else:
            return empty_screen_result()

        # Compute Return on Capital
        if all(c in available for c in ["ebit", "total_assets", "current_liabilities"]):
            cash_col = "cash_and_equivalents" if "cash_and_equivalents" in available else None

            latest = latest.with_columns(
                pl.col("total_assets").alias("_ta"),
                pl.col("current_liabilities").fill_null(0).alias("_cl"),
            )
            if cash_col:
                latest = latest.with_columns(
                    pl.col(cash_col).fill_null(0).alias("_cash")
                )
            else:
                latest = latest.with_columns(pl.lit(0.0).alias("_cash"))

            latest = latest.with_columns(
                (pl.col("_ta") - pl.col("_cl") - pl.col("_cash")).alias("_invested_capital")
            )
            latest = latest.filter(pl.col("_invested_capital") > 0)
            latest = latest.with_columns(
                (pl.col("ebit") / pl.col("_invested_capital")).alias("return_on_capital")
            )
        elif "roe" in available:
            latest = latest.filter(pl.col("roe").is_not_null())
            latest = latest.with_columns(pl.col("roe").alias("return_on_capital"))
        else:
            return empty_screen_result()

        if latest.is_empty():
            return empty_screen_result()

        # Market cap filter
        if "market_cap" in available:
            latest = latest.filter(
                (pl.col("market_cap").is_not_null())
                & (pl.col("market_cap") >= self.min_market_cap)
            )

        if latest.is_empty():
            return empty_screen_result()

        # Rank by both factors (lower rank = better)
        latest = latest.with_columns(
            pl.col("earnings_yield")
            .rank("ordinal", descending=True)
            .alias("ey_rank"),
            pl.col("return_on_capital")
            .rank("ordinal", descending=True)
            .alias("roc_rank"),
        )

        # Combined rank (lower = better)
        latest = latest.with_columns(
            (pl.col("ey_rank") + pl.col("roc_rank")).alias("combined_rank")
        )

        # Take top N
        top = latest.sort("combined_rank").head(self.top_n)

        # Score: invert combined rank so higher = better
        max_rank = top["combined_rank"].max()
        top = top.with_columns(
            ((max_rank - pl.col("combined_rank") + 1) / max_rank).alias("score")
        )

        # Rationale
        top = top.with_columns(
            (
                pl.lit("EY=")
                + (pl.col("earnings_yield") * 100).round(1).cast(pl.Utf8)
                + pl.lit("% ROC=")
                + (pl.col("return_on_capital") * 100).round(1).cast(pl.Utf8)
                + pl.lit("% Rank=")
                + pl.col("combined_rank").cast(pl.Utf8)
            ).alias("rationale")
        )

        top = top.with_columns(pl.lit("BUY").alias("signal_type"))

        return top.select(["ticker", "score", "signal_type", "rationale"])
