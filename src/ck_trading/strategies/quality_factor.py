"""Quality Factor strategy.

Selects stocks with high profitability, low leverage, and stable earnings.
Composite scoring based on:
- ROE (higher = better)
- Debt-to-Equity (lower = better)
- Earnings stability across periods (consistent positive growth)
"""

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy

try:
    from ck_trading.strategies.registry import register
except ImportError:
    def register(cls):
        return cls


@register
class QualityFactorStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Quality Factor"

    @property
    def description(self) -> str:
        return "High ROE, low leverage, and stable earnings growth"

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def __init__(
        self,
        min_roe: float = 0.15,
        max_debt_to_equity: float = 1.0,
        min_periods: int = 2,
        top_n: int = 20,
        min_market_cap: float = 5e8,
    ):
        self.min_roe = min_roe
        self.max_debt_to_equity = max_debt_to_equity
        self.min_periods = min_periods
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
            return _empty_result()

        if "period_end" not in fundamentals.columns:
            return _empty_result()

        fund = fundamentals.filter(pl.col("period_end") <= as_of_date)
        if fund.is_empty():
            return _empty_result()

        # Get latest fundamentals per ticker
        latest = (
            fund.sort("period_end", descending=True)
            .group_by("ticker")
            .first()
        )

        # Filter by ROE
        if "roe" in latest.columns:
            latest = latest.filter(
                (pl.col("roe").is_not_null()) & (pl.col("roe") >= self.min_roe)
            )
        else:
            return _empty_result()

        if latest.is_empty():
            return _empty_result()

        # Filter by debt-to-equity
        if "debt_to_equity" in latest.columns:
            latest = latest.filter(
                (pl.col("debt_to_equity").is_not_null())
                & (pl.col("debt_to_equity") <= self.max_debt_to_equity)
            )

        if latest.is_empty():
            return _empty_result()

        # Filter by market cap
        if "market_cap" in latest.columns:
            latest = latest.filter(
                (pl.col("market_cap").is_not_null())
                & (pl.col("market_cap") >= self.min_market_cap)
            )

        if latest.is_empty():
            return _empty_result()

        # Compute earnings stability per ticker
        # Need multiple periods with net_income
        qualifying_tickers = latest["ticker"].to_list()

        stability_records = []
        for ticker in qualifying_tickers:
            ticker_fund = (
                fund.filter(pl.col("ticker") == ticker)
                .sort("period_end")
            )

            if ticker_fund.height < self.min_periods:
                # Not enough periods for stability; exclude
                continue

            net_incomes = ticker_fund["net_income"].drop_nulls().to_list()
            if len(net_incomes) < self.min_periods:
                continue

            # Compute growth rates between consecutive periods
            growth_rates = []
            for i in range(1, len(net_incomes)):
                if net_incomes[i - 1] == 0:
                    continue
                gr = (net_incomes[i] - net_incomes[i - 1]) / abs(net_incomes[i - 1])
                growth_rates.append(gr)

            if not growth_rates:
                stability = 0.0
            else:
                all_positive = all(g > 0 for g in growth_rates)
                mean_gr = sum(growth_rates) / len(growth_rates)
                if mean_gr == 0:
                    cv = float("inf")
                else:
                    std_gr = (
                        sum((g - mean_gr) ** 2 for g in growth_rates) / len(growth_rates)
                    ) ** 0.5
                    cv = abs(std_gr / mean_gr)

                # Stability score: 1.0 if all positive and low CV, decreasing otherwise
                if all_positive:
                    stability = max(0.0, 1.0 - cv)
                else:
                    stability = max(0.0, 0.5 - cv)

            stability_records.append({"ticker": ticker, "stability": stability})

        if not stability_records:
            return _empty_result()

        stability_df = pl.DataFrame(stability_records)

        # Join stability back to latest fundamentals
        scored = latest.join(stability_df, on="ticker", how="inner")

        if scored.is_empty():
            return _empty_result()

        n = scored.height

        # Rank components
        # ROE rank: higher = better
        scored = scored.with_columns(
            (pl.col("roe").rank("ordinal") / n).alias("roe_rank")
        )

        # D/E rank: lower = better (invert)
        scored = scored.with_columns(
            (1.0 - pl.col("debt_to_equity").rank("ordinal") / n).alias("de_rank")
        )

        # Stability rank: higher = better
        scored = scored.with_columns(
            (pl.col("stability").rank("ordinal") / n).alias("stability_rank")
        )

        # Composite score (equal weight)
        scored = scored.with_columns(
            ((pl.col("roe_rank") + pl.col("de_rank") + pl.col("stability_rank")) / 3.0)
            .alias("score")
        )

        # Top N
        top = scored.sort("score", descending=True).head(self.top_n)

        # Build rationale
        top = top.with_columns(
            (
                pl.lit("ROE=")
                + (pl.col("roe") * 100).round(1).cast(pl.Utf8)
                + pl.lit("% D/E=")
                + pl.col("debt_to_equity").round(2).cast(pl.Utf8)
                + pl.lit(" Stability=")
                + pl.col("stability").round(2).cast(pl.Utf8)
            ).alias("rationale")
        )

        top = top.with_columns(pl.lit("BUY").alias("signal_type"))

        return top.select(["ticker", "score", "signal_type", "rationale"])


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "score": pl.Float64,
            "signal_type": pl.Utf8,
            "rationale": pl.Utf8,
        }
    )
