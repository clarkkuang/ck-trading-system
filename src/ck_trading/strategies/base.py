"""Base strategy protocol for value investing strategies."""

from abc import ABC, abstractmethod
from datetime import date

import polars as pl


class Strategy(ABC):
    """Abstract base for all value investing strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable strategy description."""
        ...

    @property
    def rebalance_frequency(self) -> str:
        """How often the strategy rebalances: monthly, quarterly, annual."""
        return "quarterly"

    @abstractmethod
    def screen(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        as_of_date: date,
    ) -> pl.DataFrame:
        """Screen stocks and return ranked results.

        Args:
            prices: DataFrame with columns [ticker, date, open, high, low, close, volume]
            fundamentals: DataFrame with financial data
            as_of_date: Date to evaluate as-of (for point-in-time correctness)

        Returns:
            DataFrame with columns: [ticker, score, signal_type, rationale]
            - score: float, higher = stronger signal
            - signal_type: "BUY", "SELL", or "HOLD"
            - rationale: human-readable explanation
        """
        ...

    def generate_signals(
        self,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        start_date: date,
        end_date: date,
        rebalance_dates: list[date] | None = None,
    ) -> pl.DataFrame:
        """Generate signals across a date range (for backtesting).

        Default implementation calls screen() at each rebalance date.
        Override for more efficient vectorized implementations.
        """
        if rebalance_dates is None:
            rebalance_dates = _generate_rebalance_dates(
                start_date, end_date, self.rebalance_frequency
            )

        frames = []
        for rebal_date in rebalance_dates:
            result = self.screen(prices, fundamentals, rebal_date)
            if not result.is_empty():
                result = result.with_columns(pl.lit(rebal_date).alias("date"))
                frames.append(result)

        if not frames:
            return pl.DataFrame(
                schema={"ticker": pl.Utf8, "date": pl.Date, "score": pl.Float64,
                        "signal_type": pl.Utf8, "rationale": pl.Utf8}
            )

        return pl.concat(frames)


def _generate_rebalance_dates(
    start: date, end: date, freq: str
) -> list[date]:
    """Generate rebalance dates between start and end."""

    dates = []
    current = start

    if freq == "monthly":
        delta_months = 1
    elif freq == "quarterly":
        delta_months = 3
    elif freq == "annual":
        delta_months = 12
    else:
        delta_months = 3

    while current <= end:
        dates.append(current)
        # Advance by N months
        month = current.month + delta_months
        year = current.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = min(current.day, _days_in_month(year, month))
        current = date(year, month, day)

    return dates


def _days_in_month(year: int, month: int) -> int:
    """Return number of days in a month."""
    import calendar
    return calendar.monthrange(year, month)[1]
