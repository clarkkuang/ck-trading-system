"""Ten Bagger Strategy.

Screen for stocks with 10x return potential by combining:
- Small/micro-cap filter ($50M–$2B market cap)
- High revenue growth (>20% YoY)
- High gross margin (>30%)
- Healthy balance sheet (D/E < 1.0)
- High ROE (>15%)
- Positive price momentum (6-month return)
- Optional insider buying boost (from extra_data)

Multi-factor composite score with growth-weighted ranking.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class TenBaggerStrategy(Strategy):
    """Screen for stocks with 10x return potential."""

    def __init__(
        self,
        min_market_cap: float = 50e6,
        max_market_cap: float = 2e9,
        min_revenue_growth: float = 0.20,
        min_gross_margin: float = 0.30,
        max_debt_to_equity: float = 1.0,
        min_roe: float = 0.15,
        momentum_days: int = 126,
        top_n: int = 20,
    ):
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.min_revenue_growth = min_revenue_growth
        self.min_gross_margin = min_gross_margin
        self.max_debt_to_equity = max_debt_to_equity
        self.min_roe = min_roe
        self.momentum_days = momentum_days
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Ten Bagger"

    @property
    def description(self) -> str:
        return (
            "Ten Bagger screens for stocks with 10x return potential: "
            "small/micro-cap ($50M–$2B), high revenue growth (>20% YoY), "
            "strong gross margin (>30%), healthy balance sheet (D/E<1), "
            "high ROE (>15%), and positive 6-month momentum. "
            "Composite score weights growth 30%, margin 20%, ROE 20%, "
            "momentum 15%, free cash flow 15%. Insider buying adds bonus. "
            "Data: prices + fundamentals + optional insider trades."
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
        if fundamentals.is_empty() or "period_end" not in fundamentals.columns:
            return empty_screen_result()

        # --- Step 1: get latest two periods per ticker ---
        fund = fundamentals.filter(pl.col("period_end") <= as_of_date)
        if fund.is_empty():
            return empty_screen_result()

        fund = fund.sort(["ticker", "period_end"], descending=[False, True])
        latest = fund.group_by("ticker").first()

        # --- Step 2: small/micro-cap filter ---
        if "market_cap" not in latest.columns:
            return empty_screen_result()

        latest = latest.filter(
            (pl.col("market_cap").is_not_null())
            & (pl.col("market_cap") >= self.min_market_cap)
            & (pl.col("market_cap") <= self.max_market_cap)
        )
        if latest.is_empty():
            return empty_screen_result()

        # --- Step 3: gross margin filter ---
        if "gross_margin" in latest.columns:
            latest = latest.filter(
                (pl.col("gross_margin").is_not_null())
                & (pl.col("gross_margin") >= self.min_gross_margin)
            )

        if latest.is_empty():
            return empty_screen_result()

        # --- Step 4: debt-to-equity filter ---
        if "debt_to_equity" in latest.columns:
            latest = latest.filter(
                (pl.col("debt_to_equity").is_not_null())
                & (pl.col("debt_to_equity") <= self.max_debt_to_equity)
            )

        if latest.is_empty():
            return empty_screen_result()

        # --- Step 5: ROE filter ---
        if "roe" in latest.columns:
            latest = latest.filter(
                (pl.col("roe").is_not_null())
                & (pl.col("roe") >= self.min_roe)
            )

        if latest.is_empty():
            return empty_screen_result()

        # --- Step 6: compute revenue growth vs. prior period ---
        valid_tickers = latest["ticker"].to_list()

        growth_records = []
        for ticker in valid_tickers:
            ticker_fund = (
                fund.filter(pl.col("ticker") == ticker)
                .sort("period_end", descending=True)
            )
            if ticker_fund.height < 2:
                continue

            latest_row = ticker_fund.row(0, named=True)
            prior_row = ticker_fund.row(1, named=True)

            rev_latest = latest_row.get("revenue")
            rev_prior = prior_row.get("revenue")

            if rev_latest is None or rev_prior is None or rev_prior == 0:
                continue

            growth = (rev_latest - rev_prior) / abs(rev_prior)
            if growth < self.min_revenue_growth:
                continue

            growth_records.append({"ticker": ticker, "revenue_growth": growth})

        if not growth_records:
            return empty_screen_result()

        growth_df = pl.DataFrame(growth_records)
        candidates = latest.join(growth_df, on="ticker", how="inner")

        if candidates.is_empty():
            return empty_screen_result()

        # --- Step 7: price momentum ---
        momentum_map: dict[str, float] = {}
        if not prices.is_empty():
            cutoff = as_of_date - timedelta(days=self.momentum_days)
            price_window = prices.filter(
                (pl.col("date") >= cutoff) & (pl.col("date") <= as_of_date)
            )
            for ticker in candidates["ticker"].to_list():
                t_prices = price_window.filter(pl.col("ticker") == ticker).sort("date")
                if t_prices.height >= 2:
                    start_p = t_prices["close"][0]
                    end_p = t_prices["close"][-1]
                    if start_p and start_p > 0:
                        momentum_map[ticker] = float(end_p / start_p - 1)

        candidates = candidates.with_columns(
            pl.col("ticker")
            .map_elements(lambda t: momentum_map.get(t, 0.0), return_dtype=pl.Float64)
            .alias("momentum")
        )

        # --- Step 8: insider buying bonus ---
        insider_map: dict[str, float] = {}
        if extra_data is not None and "insider_trades" in extra_data:
            trades = extra_data["insider_trades"]
            if not trades.is_empty() and {"ticker", "date", "transaction_type", "value"}.issubset(set(trades.columns)):
                cutoff90 = as_of_date - timedelta(days=90)
                recent = trades.filter(
                    (pl.col("date") >= cutoff90) & (pl.col("date") <= as_of_date)
                )
                if not recent.is_empty():
                    net = (
                        recent.with_columns(
                            pl.when(pl.col("transaction_type") == "P")
                            .then(pl.col("value"))
                            .otherwise(-pl.col("value"))
                            .alias("signed_value")
                        )
                        .group_by("ticker")
                        .agg(pl.col("signed_value").sum().alias("net_buy"))
                    )
                    for row in net.iter_rows(named=True):
                        if row["net_buy"] > 0:
                            insider_map[row["ticker"]] = float(row["net_buy"])

        candidates = candidates.with_columns(
            pl.col("ticker")
            .map_elements(lambda t: insider_map.get(t, 0.0), return_dtype=pl.Float64)
            .alias("insider_net_buy")
        )

        # --- Step 9: rank and composite score ---
        n = candidates.height
        if n == 0:
            return empty_screen_result()

        def pct_rank(col: str, descending: bool = True) -> pl.Expr:
            """Return percentile rank [0,1], higher = better. Ties share the same rank."""
            if descending:
                return pl.col(col).rank("average") / n
            else:
                return (1.0 - pl.col(col).rank("average") / n)

        candidates = candidates.with_columns(
            pct_rank("revenue_growth").alias("r_growth"),
            pct_rank("gross_margin").alias("r_margin"),
            pct_rank("roe").alias("r_roe"),
            pct_rank("momentum").alias("r_momentum"),
        )

        # FCF rank (optional column)
        if "free_cash_flow" in candidates.columns:
            candidates = candidates.with_columns(
                pct_rank("free_cash_flow").alias("r_fcf")
            )
        else:
            candidates = candidates.with_columns(pl.lit(0.5).alias("r_fcf"))

        # Insider bonus: add up to 0.1 to final score if net buying > 0
        candidates = candidates.with_columns(
            pl.when(pl.col("insider_net_buy") > 0)
            .then(0.1)
            .otherwise(0.0)
            .alias("insider_bonus")
        )

        candidates = candidates.with_columns(
            (
                0.30 * pl.col("r_growth")
                + 0.20 * pl.col("r_margin")
                + 0.20 * pl.col("r_roe")
                + 0.15 * pl.col("r_momentum")
                + 0.15 * pl.col("r_fcf")
                + pl.col("insider_bonus")
            ).alias("score")
        )

        # --- Step 10: build rationale and output ---
        candidates = candidates.with_columns(
            (
                pl.lit("RevGrowth=")
                + (pl.col("revenue_growth") * 100).round(1).cast(pl.Utf8)
                + pl.lit("% GM=")
                + (pl.col("gross_margin") * 100).round(1).cast(pl.Utf8)
                + pl.lit("% ROE=")
                + (pl.col("roe") * 100).round(1).cast(pl.Utf8)
                + pl.lit("% Mom=")
                + (pl.col("momentum") * 100).round(1).cast(pl.Utf8)
                + pl.lit("%")
            ).alias("rationale"),
            pl.lit("BUY").alias("signal_type"),
        )

        result = candidates.sort("score", descending=True).head(self.top_n)
        return result.select(["ticker", "score", "signal_type", "rationale"])
