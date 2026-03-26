"""DCF Intrinsic Value strategy.

Estimates intrinsic value using a simplified Discounted Cash Flow model:
1. Project free cash flows for N years using historical growth rate
2. Calculate terminal value using perpetuity growth model
3. Discount all cash flows to present value
4. Compare to current market price
5. Signal BUY if market price < intrinsic value * (1 - margin_of_safety)
"""

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class DCFIntrinsicStrategy(Strategy):
    @property
    def name(self) -> str:
        return "DCF Intrinsic Value"

    @property
    def description(self) -> str:
        return (
            "Discounted Cash Flow intrinsic value: estimate fair value from projected free cash "
            "flows and buy stocks trading below DCF value with a margin of safety. "
            "Data: fundamentals (cash flow statement + balance sheet) + price."
        )

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def __init__(
        self,
        discount_rate: float = 0.10,
        terminal_growth: float = 0.03,
        projection_years: int = 10,
        margin_of_safety: float = 0.30,
    ):
        self.discount_rate = discount_rate
        self.terminal_growth = terminal_growth
        self.projection_years = projection_years
        self.margin_of_safety = margin_of_safety

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if fundamentals.is_empty():
            return empty_screen_result()

        # Get fundamentals with at least 2 periods for growth estimation
        if "period_end" in fundamentals.columns:
            fund = fundamentals.filter(pl.col("period_end") <= as_of_date)
        else:
            fund = fundamentals

        if "free_cash_flow" not in fund.columns and "operating_cash_flow" not in fund.columns:
            return empty_screen_result()

        tickers = fund["ticker"].unique().to_list()
        results = []

        for ticker in tickers:
            ticker_data = fund.filter(pl.col("ticker") == ticker).sort("period_end")
            intrinsic = self._estimate_intrinsic_value(ticker_data)
            if intrinsic is None:
                continue

            # Get current price
            latest_row = (
                ticker_data.sort("period_end", descending=True)
                .row(0, named=True)
            )
            market_cap = latest_row.get("market_cap")
            if not market_cap or market_cap <= 0:
                continue

            # Get shares outstanding estimate
            bvps = latest_row.get("book_value_per_share")
            equity = latest_row.get("total_equity")
            if bvps and bvps > 0 and equity:
                shares = equity / bvps
                intrinsic_per_share = intrinsic / shares
                price_per_share = market_cap / shares
            else:
                intrinsic_per_share = intrinsic
                price_per_share = market_cap

            buy_below = intrinsic_per_share * (1 - self.margin_of_safety)
            if price_per_share <= buy_below and intrinsic_per_share > 0:
                upside = (intrinsic_per_share - price_per_share) / price_per_share
                rationale = (
                    f"Intrinsic=${intrinsic_per_share:,.0f} "
                    f"vs Price=${price_per_share:,.0f} "
                    f"({upside:.0%} upside)"
                )
                results.append({
                    "ticker": ticker,
                    "score": upside,
                    "signal_type": "BUY",
                    "rationale": rationale,
                })

        if not results:
            return empty_screen_result()

        return pl.DataFrame(results).sort("score", descending=True)

    def _estimate_intrinsic_value(self, ticker_data: pl.DataFrame) -> float | None:
        """Estimate intrinsic value using DCF."""
        # Get free cash flow (or operating cash flow - capex)
        fcf_col = None
        if "free_cash_flow" in ticker_data.columns:
            fcf_col = "free_cash_flow"
        elif "operating_cash_flow" in ticker_data.columns and "capex" in ticker_data.columns:
            ticker_data = ticker_data.with_columns(
                (pl.col("operating_cash_flow") + pl.col("capex").fill_null(0)).alias("_fcf")
            )
            fcf_col = "_fcf"
        elif "operating_cash_flow" in ticker_data.columns:
            fcf_col = "operating_cash_flow"
        else:
            return None

        fcf_values = (
            ticker_data
            .filter(pl.col(fcf_col).is_not_null())
            .sort("period_end")
            [fcf_col]
            .to_list()
        )

        if not fcf_values or len(fcf_values) < 1:
            return None

        latest_fcf = fcf_values[-1]
        if latest_fcf <= 0:
            return None

        # Estimate growth rate from historical data
        if len(fcf_values) >= 2:
            first_positive = next((v for v in fcf_values if v > 0), None)
            if first_positive and first_positive > 0:
                years = len(fcf_values) - 1
                growth = (latest_fcf / first_positive) ** (1 / max(years, 1)) - 1
                growth = max(min(growth, 0.25), 0.0)  # Cap at 25%, floor at 0%
            else:
                growth = 0.05
        else:
            growth = 0.05

        # Project FCFs
        total_pv = 0.0
        projected_fcf = latest_fcf
        for year in range(1, self.projection_years + 1):
            projected_fcf *= (1 + growth)
            pv = projected_fcf / (1 + self.discount_rate) ** year
            total_pv += pv

        # Terminal value
        if self.discount_rate <= self.terminal_growth:
            # Cannot compute terminal value when discount rate <= growth rate
            return None
        terminal_fcf = projected_fcf * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (self.discount_rate - self.terminal_growth)
        terminal_pv = terminal_value / (1 + self.discount_rate) ** self.projection_years
        total_pv += terminal_pv

        return total_pv
