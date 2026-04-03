"""Dividend tracking for portfolio positions."""

from dataclasses import dataclass
from datetime import date


@dataclass
class DividendRecord:
    """A single dividend payment record."""

    ticker: str
    ex_date: date
    pay_date: date
    amount_per_share: float
    shares_held: float

    @property
    def total_amount(self) -> float:
        return self.amount_per_share * self.shares_held


class DividendTracker:
    """Track dividend income across portfolio positions."""

    def __init__(self) -> None:
        self._records: list[DividendRecord] = []

    def add_dividend(
        self,
        ticker: str,
        ex_date: date,
        pay_date: date,
        amount_per_share: float,
        shares_held: float,
    ) -> None:
        self._records.append(DividendRecord(
            ticker=ticker,
            ex_date=ex_date,
            pay_date=pay_date,
            amount_per_share=amount_per_share,
            shares_held=shares_held,
        ))

    def get_dividends(self, ticker: str) -> list[DividendRecord]:
        return [r for r in self._records if r.ticker == ticker]

    def total_income(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> float:
        """Total dividend income, optionally filtered by pay_date range."""
        records = self._records
        if start_date:
            records = [r for r in records if r.pay_date >= start_date]
        if end_date:
            records = [r for r in records if r.pay_date <= end_date]
        return sum(r.total_amount for r in records)

    def income_by_ticker(self) -> dict[str, float]:
        """Total dividend income grouped by ticker."""
        result: dict[str, float] = {}
        for r in self._records:
            result[r.ticker] = result.get(r.ticker, 0) + r.total_amount
        return result

    def yield_on_cost(
        self, ticker: str, avg_cost: float, year: int
    ) -> float:
        """Annual dividend yield on cost basis.

        Sum of per-share dividends in the given year / avg_cost.
        """
        if avg_cost <= 0:
            return 0.0
        annual_div = sum(
            r.amount_per_share
            for r in self._records
            if r.ticker == ticker and r.pay_date.year == year
        )
        return annual_div / avg_cost

    def projected_annual_income(
        self, ticker: str, shares: float, frequency: int = 4
    ) -> float:
        """Project annual income based on most recent dividend rate.

        Args:
            ticker: stock ticker
            shares: current shares held
            frequency: expected payments per year (4=quarterly, 12=monthly)
        """
        ticker_divs = sorted(
            (r for r in self._records if r.ticker == ticker),
            key=lambda r: r.ex_date,
        )
        if not ticker_divs:
            return 0.0
        latest_rate = ticker_divs[-1].amount_per_share
        return latest_rate * shares * frequency
