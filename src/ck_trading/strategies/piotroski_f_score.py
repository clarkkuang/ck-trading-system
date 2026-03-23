"""Piotroski F-Score strategy.

8-point scoring system evaluating three areas:
1. Profitability (4 points):
   - Positive ROA
   - Positive operating cash flow
   - ROA improvement (year over year)
   - Accruals: operating cash flow > net income (quality of earnings)

2. Leverage/Liquidity (2 points):
   - Decrease in long-term debt ratio
   - Increase in current ratio

3. Operating Efficiency (2 points):
   - Increase in gross margin
   - Increase in asset turnover

Note: The original Piotroski F-Score includes a 9th point for no share dilution,
but this implementation omits it because reliable shares-outstanding data is not
always available. Scores range from 0-8.

Stocks scoring 7-8 are strong buys; 0-2 are potential shorts.
"""

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class PiotroskiFScoreStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Piotroski F-Score"

    @property
    def description(self) -> str:
        return "8-factor scoring: profitability, leverage, and operating efficiency"

    @property
    def rebalance_frequency(self) -> str:
        return "quarterly"

    def __init__(self, min_score: int = 6):
        self.min_score = min_score

    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        if fundamentals.is_empty():
            return empty_screen_result()

        # Need at least 2 periods to compute year-over-year changes
        if "period_end" not in fundamentals.columns:
            return empty_screen_result()

        fund = fundamentals.filter(pl.col("period_end") <= as_of_date)
        if fund.is_empty():
            return empty_screen_result()

        # Get latest two periods per ticker
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

        tickers = ranked.filter(pl.col("period_rank") == 1)["ticker"].unique().to_list()
        results = []

        for ticker in tickers:
            ticker_data = (
                ranked.filter(pl.col("ticker") == ticker)
                .sort("period_end", descending=True)
            )
            if ticker_data.height < 1:
                continue

            current = ticker_data.row(0, named=True)
            prior = ticker_data.row(1, named=True) if ticker_data.height >= 2 else None

            score, details = self._compute_f_score(current, prior)
            if score >= self.min_score:
                results.append({
                    "ticker": ticker,
                    "score": float(score),
                    "signal_type": "BUY",
                    "rationale": f"F-Score={score}/8 ({', '.join(details)})",
                })

        if not results:
            return empty_screen_result()

        return pl.DataFrame(results).sort("score", descending=True)

    def _compute_f_score(self, current: dict, prior: dict | None) -> tuple[int, list[str]]:
        """Compute 8-point Piotroski F-Score."""
        score = 0
        details = []

        # --- Profitability (4 points) ---

        # 1. Positive ROA
        roa = current.get("roa")
        ni = current.get("net_income")
        ta = current.get("total_assets")
        if roa and roa > 0 or ni and ta and ta > 0 and ni / ta > 0:
            score += 1
            details.append("ROA+")

        # 2. Positive operating cash flow
        ocf = current.get("operating_cash_flow")
        if ocf and ocf > 0:
            score += 1
            details.append("OCF+")

        # 3. ROA improvement
        if prior:
            prev_roa = prior.get("roa")
            prev_ni = prior.get("net_income")
            prev_ta = prior.get("total_assets")
            curr_roa = roa or (ni / ta if ni and ta and ta > 0 else None)
            if prev_roa is None and prev_ni and prev_ta and prev_ta > 0:
                prev_roa = prev_ni / prev_ta
            if curr_roa and prev_roa and curr_roa > prev_roa:
                score += 1
                details.append("dROA+")

        # 4. Accruals: OCF > Net Income (quality of earnings)
        if ocf and ni and ocf > ni:
            score += 1
            details.append("Accruals+")

        # --- Leverage/Liquidity (3 points) ---

        # 5. Decrease in debt ratio
        if prior:
            curr_debt = current.get("debt_to_equity")
            prev_debt = prior.get("debt_to_equity")
            if curr_debt is not None and prev_debt is not None and curr_debt < prev_debt:
                score += 1
                details.append("Debt-")

        # 6. Increase in current ratio
        if prior:
            curr_cr = current.get("current_ratio")
            prev_cr = prior.get("current_ratio")
            if curr_cr and prev_cr and curr_cr > prev_cr:
                score += 1
                details.append("CR+")

        # --- Operating Efficiency (2 points) ---

        # 7. Gross margin improvement
        if prior:
            curr_gm = current.get("gross_margin")
            prev_gm = prior.get("gross_margin")
            if curr_gm and prev_gm and curr_gm > prev_gm:
                score += 1
                details.append("GM+")

        # 8. Asset turnover improvement
        if prior:
            curr_at = current.get("asset_turnover")
            prev_at = prior.get("asset_turnover")
            if curr_at and prev_at and curr_at > prev_at:
                score += 1
                details.append("AT+")

        return score, details
