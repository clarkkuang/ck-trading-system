"""Tests for dividend tracking."""

from datetime import date

import pytest

from ck_trading.portfolio.dividends import (
    DividendRecord,
    DividendTracker,
)


@pytest.fixture
def tracker():
    return DividendTracker()


class TestDividendRecord:
    def test_create_record(self):
        rec = DividendRecord(
            ticker="AAPL",
            ex_date=date(2024, 2, 9),
            pay_date=date(2024, 2, 15),
            amount_per_share=0.24,
            shares_held=100,
        )
        assert rec.total_amount == pytest.approx(24.0)

    def test_total_amount_calculation(self):
        rec = DividendRecord(
            ticker="MSFT",
            ex_date=date(2024, 3, 14),
            pay_date=date(2024, 3, 21),
            amount_per_share=0.75,
            shares_held=200,
        )
        assert rec.total_amount == pytest.approx(150.0)


class TestDividendTracker:
    def test_add_and_get_dividends(self, tracker):
        tracker.add_dividend(
            ticker="AAPL", ex_date=date(2024, 2, 9),
            pay_date=date(2024, 2, 15),
            amount_per_share=0.24, shares_held=100,
        )
        divs = tracker.get_dividends("AAPL")
        assert len(divs) == 1
        assert divs[0].amount_per_share == 0.24

    def test_total_income(self, tracker):
        tracker.add_dividend(
            ticker="AAPL", ex_date=date(2024, 2, 9),
            pay_date=date(2024, 2, 15),
            amount_per_share=0.24, shares_held=100,
        )
        tracker.add_dividend(
            ticker="MSFT", ex_date=date(2024, 3, 14),
            pay_date=date(2024, 3, 21),
            amount_per_share=0.75, shares_held=200,
        )
        total = tracker.total_income()
        assert total == pytest.approx(24.0 + 150.0)

    def test_total_income_date_range(self, tracker):
        tracker.add_dividend(
            ticker="AAPL", ex_date=date(2024, 1, 10),
            pay_date=date(2024, 1, 15),
            amount_per_share=0.24, shares_held=100,
        )
        tracker.add_dividend(
            ticker="AAPL", ex_date=date(2024, 4, 10),
            pay_date=date(2024, 4, 15),
            amount_per_share=0.24, shares_held=100,
        )
        total = tracker.total_income(
            start_date=date(2024, 3, 1), end_date=date(2024, 5, 1)
        )
        assert total == pytest.approx(24.0)  # only April

    def test_total_income_empty(self, tracker):
        assert tracker.total_income() == 0.0

    def test_yield_on_cost(self, tracker):
        """Annual dividend / cost basis."""
        # 4 quarterly dividends of $0.24 on 100 shares at $150 cost
        for month in [2, 5, 8, 11]:
            tracker.add_dividend(
                ticker="AAPL", ex_date=date(2024, month, 10),
                pay_date=date(2024, month, 15),
                amount_per_share=0.24, shares_held=100,
            )
        yoc = tracker.yield_on_cost("AAPL", avg_cost=150.0, year=2024)
        # Annual div = 4 * 0.24 = 0.96 per share, cost = 150
        assert yoc == pytest.approx(0.96 / 150.0)

    def test_yield_on_cost_no_dividends(self, tracker):
        yoc = tracker.yield_on_cost("AAPL", avg_cost=150.0, year=2024)
        assert yoc == 0.0

    def test_projected_annual_income(self, tracker):
        """Project income based on most recent dividend rate."""
        # 2 quarterly dividends so far
        tracker.add_dividend(
            ticker="AAPL", ex_date=date(2024, 2, 10),
            pay_date=date(2024, 2, 15),
            amount_per_share=0.24, shares_held=100,
        )
        tracker.add_dividend(
            ticker="AAPL", ex_date=date(2024, 5, 10),
            pay_date=date(2024, 5, 15),
            amount_per_share=0.25, shares_held=100,
        )
        projected = tracker.projected_annual_income(
            "AAPL", shares=100, frequency=4
        )
        # Uses most recent rate: 0.25 * 100 * 4 = 100
        assert projected == pytest.approx(100.0)

    def test_projected_annual_income_no_data(self, tracker):
        assert tracker.projected_annual_income("AAPL", shares=100) == 0.0

    def test_income_by_ticker(self, tracker):
        tracker.add_dividend(
            ticker="AAPL", ex_date=date(2024, 2, 9),
            pay_date=date(2024, 2, 15),
            amount_per_share=0.24, shares_held=100,
        )
        tracker.add_dividend(
            ticker="MSFT", ex_date=date(2024, 3, 14),
            pay_date=date(2024, 3, 21),
            amount_per_share=0.75, shares_held=200,
        )
        by_ticker = tracker.income_by_ticker()
        assert by_ticker["AAPL"] == pytest.approx(24.0)
        assert by_ticker["MSFT"] == pytest.approx(150.0)
