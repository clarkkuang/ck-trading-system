"""Mean Reversion Strategy.

Uses RSI (Relative Strength Index) to identify oversold and overbought
conditions.  Buys oversold tickers and sells overbought ones.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Compute RSI from a list of close prices.

    Uses the standard Wilder smoothing method:
    1. First average is a simple mean of the first *period* changes.
    2. Subsequent averages use exponential smoothing:
       avg = (prev_avg * (period - 1) + current) / period

    Returns None if there are fewer than period + 1 prices.
    """
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Seed with simple average of first *period* deltas
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [max(-d, 0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing for the rest
    for d in deltas[period:]:
        gain = max(d, 0)
        loss = max(-d, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


@register
class MeanReversionStrategy(Strategy):
    """RSI-based mean reversion: buy oversold, sell overbought."""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        top_n: int = 20,
    ):
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Mean Reversion"

    @property
    def description(self) -> str:
        return (
            f"RSI({self.rsi_period}) mean reversion: "
            f"buy < {self.oversold_threshold}, sell > {self.overbought_threshold}"
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
        if prices.is_empty():
            return empty_screen_result()

        # Use prices up to as_of_date
        hist = prices.filter(pl.col("date") <= as_of_date)
        if hist.is_empty():
            return empty_screen_result()

        tickers = hist["ticker"].unique().to_list()
        rows: list[dict] = []

        for ticker in tickers:
            ticker_prices = (
                hist.filter(pl.col("ticker") == ticker)
                .sort("date")
            )
            closes = ticker_prices["close"].to_list()
            rsi = compute_rsi(closes, self.rsi_period)
            if rsi is None:
                continue

            if rsi < self.oversold_threshold:
                score = (self.oversold_threshold - rsi) / self.oversold_threshold
                rows.append({
                    "ticker": ticker,
                    "score": score,
                    "signal_type": "BUY",
                    "rationale": f"RSI={rsi:.1f} (oversold)",
                })
            elif rsi > self.overbought_threshold:
                score = (rsi - self.overbought_threshold) / (100.0 - self.overbought_threshold)
                rows.append({
                    "ticker": ticker,
                    "score": score,
                    "signal_type": "SELL",
                    "rationale": f"RSI={rsi:.1f} (overbought)",
                })

        if not rows:
            return empty_screen_result()

        result = pl.DataFrame(rows)

        # Limit to top_n per signal type, sorted by score descending
        buys = (
            result.filter(pl.col("signal_type") == "BUY")
            .sort("score", descending=True)
            .head(self.top_n)
        )
        sells = (
            result.filter(pl.col("signal_type") == "SELL")
            .sort("score", descending=True)
            .head(self.top_n)
        )

        combined = pl.concat([buys, sells], how="diagonal")
        if combined.is_empty():
            return empty_screen_result()

        return combined.select(["ticker", "score", "signal_type", "rationale"])
