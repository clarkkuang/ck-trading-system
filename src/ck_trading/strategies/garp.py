"""GARP (Growth at Reasonable Price) strategy.

Identifies stocks with strong earnings growth trading at reasonable
valuations, using the PEG ratio as the core metric.

PEG = PE ratio / earnings growth rate (%)
Lower PEG indicates better value for growth received.
"""

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class GARPStrategy(Strategy):
    @property
    def name(self) -> str:
        return "GARP"

    @property
    def description(self) -> str:
        return "Growth at Reasonable Price: low PEG ratio screening"

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def __init__(
        self,
        max_peg: float = 1.5,
        min_growth: float = 0.10,
        top_n: int = 20,
        min_market_cap: float = 5e8,
    ):
        self.max_peg = max_peg
        self.min_growth = min_growth
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

        if "period_end" not in fundamentals.columns:
            return empty_screen_result()

        fund = fundamentals.filter(pl.col("period_end") <= as_of_date)
        if fund.is_empty():
            return empty_screen_result()

        # Get the two most recent periods per ticker
        ranked = (
            fund.sort(["ticker", "period_end"], descending=[False, True])
            .with_columns(
                pl.col("period_end")
                .rank("ordinal", descending=True)
                .over("ticker")
                .alias("period_rank")
            )
            .filter(pl.col("period_rank") <= 2)
        )

        # Only keep tickers that have exactly 2 periods
        ticker_counts = (
            ranked.group_by("ticker")
            .agg(pl.len().alias("n_periods"))
            .filter(pl.col("n_periods") >= 2)
        )

        if ticker_counts.is_empty():
            return empty_screen_result()

        valid_tickers = ticker_counts["ticker"].to_list()

        results = []
        for ticker in valid_tickers:
            ticker_data = (
                ranked.filter(pl.col("ticker") == ticker)
                .sort("period_end", descending=True)
            )

            latest = ticker_data.row(0, named=True)
            prior = ticker_data.row(1, named=True)

            # Market cap filter
            market_cap = latest.get("market_cap")
            if market_cap is None or market_cap < self.min_market_cap:
                continue

            # Get PE ratio from latest
            pe_ratio = latest.get("pe_ratio")
            if pe_ratio is None or pe_ratio <= 0:
                continue

            # Compute earnings growth
            latest_ni = latest.get("net_income")
            prior_ni = prior.get("net_income")

            if latest_ni is None or prior_ni is None:
                continue
            if prior_ni == 0:
                continue  # Can't compute growth from zero

            earnings_growth = (latest_ni - prior_ni) / abs(prior_ni)

            # Exclude negative growth (PEG meaningless)
            if earnings_growth <= 0:
                continue

            # Check minimum growth threshold
            if earnings_growth < self.min_growth:
                continue

            # Compute PEG
            earnings_growth_pct = earnings_growth * 100
            if earnings_growth_pct == 0:
                continue

            peg = pe_ratio / earnings_growth_pct

            # Filter by PEG
            if peg <= 0 or peg > self.max_peg:
                continue

            # Score = 1 / PEG (lower PEG = higher score)
            score = 1.0 / peg

            results.append({
                "ticker": ticker,
                "score": score,
                "signal_type": "BUY",
                "rationale": (
                    f"PE={pe_ratio:.1f} "
                    f"Growth={earnings_growth * 100:.1f}% "
                    f"PEG={peg:.2f}"
                ),
            })

        if not results:
            return empty_screen_result()

        result_df = pl.DataFrame(results).sort("score", descending=True)
        return result_df.head(self.top_n)
