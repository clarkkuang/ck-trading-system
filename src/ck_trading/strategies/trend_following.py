"""Trend Following Strategy.

Dual moving-average crossover (SMA50/SMA200) with ATR-based stop loss.
Buys confirmed uptrends, sells confirmed downtrends.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from ck_trading.strategies.base import Strategy, empty_screen_result
from ck_trading.strategies.registry import register


@register
class TrendFollowingStrategy(Strategy):
    """SMA crossover trend following with ATR volatility filter."""

    def __init__(
        self,
        fast_sma: int = 50,
        slow_sma: int = 200,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        top_n: int = 20,
    ):
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "Trend Following"

    @property
    def description(self) -> str:
        return (
            f"SMA({self.fast_sma}/{self.slow_sma}) trend following "
            f"with ATR({self.atr_period}) stop"
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

        hist = prices.filter(pl.col("date") <= as_of_date)
        if hist.is_empty():
            return empty_screen_result()

        tickers = hist["ticker"].unique().to_list()
        rows: list[dict] = []

        for ticker in tickers:
            tp = hist.filter(pl.col("ticker") == ticker).sort("date")
            n = tp.height

            # Need at least slow_sma data points
            if n < self.slow_sma:
                continue

            closes = tp["close"].to_list()
            highs = tp["high"].to_list()
            lows = tp["low"].to_list()

            current_close = closes[-1]
            sma_fast = sum(closes[-self.fast_sma:]) / self.fast_sma
            sma_slow = sum(closes[-self.slow_sma:]) / self.slow_sma

            # Compute ATR
            atr = self._compute_atr(closes, highs, lows, self.atr_period)
            if atr is None or atr == 0:
                continue

            stop_loss = current_close - self.atr_stop_multiplier * atr

            # Determine signal
            if current_close > sma_fast > sma_slow:
                signal = "BUY"
                score = (current_close - sma_slow) / atr
            elif current_close < sma_fast < sma_slow:
                signal = "SELL"
                score = (sma_slow - current_close) / atr
            else:
                continue

            rationale = (
                f"SMA{self.fast_sma}={sma_fast:.2f} "
                f"SMA{self.slow_sma}={sma_slow:.2f} "
                f"ATR={atr:.2f} "
                f"StopLoss={stop_loss:.2f}"
            )

            rows.append({
                "ticker": ticker,
                "score": score,
                "signal_type": signal,
                "rationale": rationale,
            })

        if not rows:
            return empty_screen_result()

        result = pl.DataFrame(rows)

        # Take top_n by score for each signal type
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

    @staticmethod
    def _compute_atr(
        closes: list[float],
        highs: list[float],
        lows: list[float],
        period: int,
    ) -> float | None:
        """Compute Average True Range over the last *period* bars."""
        n = len(closes)
        if n < period + 1:
            return None

        true_ranges: list[float] = []
        for i in range(n - period, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        return sum(true_ranges) / period
