"""Signal generator - runs strategies and produces signals."""

from datetime import date, datetime

import polars as pl

from ck_trading.models.signals import Signal, SignalType
from ck_trading.strategies.base import Strategy


class SignalGenerator:
    """Run strategies against current data to generate trading signals."""

    def __init__(self, strategies: list[Strategy] | None = None):
        self.strategies = strategies or []

    def add_strategy(self, strategy: Strategy) -> None:
        self.strategies.append(strategy)

    def generate(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date | None = None,
    ) -> list[Signal]:
        """Run all strategies and collect signals."""
        if as_of_date is None:
            as_of_date = date.today()

        signals = []
        for strategy in self.strategies:
            try:
                result = strategy.screen(prices, fundamentals, as_of_date)
                if result.is_empty():
                    continue

                for row in result.iter_rows(named=True):
                    # Get current price for the ticker
                    price = None
                    if not prices.is_empty():
                        price_row = prices.filter(
                            pl.col("ticker") == row["ticker"]
                        ).sort("date", descending=True).head(1)
                        if not price_row.is_empty():
                            price = price_row["close"][0]

                    signal = Signal(
                        ticker=row["ticker"],
                        signal_type=SignalType(row["signal_type"]),
                        strategy_name=strategy.name,
                        score=row["score"],
                        rationale=row.get("rationale", ""),
                        generated_at=datetime.now(),
                        price_at_signal=price,
                    )
                    signals.append(signal)
            except Exception as e:
                print(f"Error running strategy {strategy.name}: {e}")
                continue

        # Sort by score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
