"""Tests for base strategy utilities."""

from datetime import date

import polars as pl

from ck_trading.strategies.base import (
    Strategy,
    _generate_rebalance_dates,
    empty_screen_result,
)


def test_generate_rebalance_dates_quarterly():
    dates = _generate_rebalance_dates(date(2024, 1, 1), date(2024, 12, 31), "quarterly")
    assert dates[0] == date(2024, 1, 1)
    assert dates[1] == date(2024, 4, 1)
    assert dates[2] == date(2024, 7, 1)
    assert dates[3] == date(2024, 10, 1)
    assert len(dates) == 4


def test_generate_rebalance_dates_monthly():
    dates = _generate_rebalance_dates(date(2024, 1, 1), date(2024, 6, 30), "monthly")
    assert len(dates) == 6
    assert dates[1] == date(2024, 2, 1)


def test_generate_rebalance_dates_annual():
    dates = _generate_rebalance_dates(date(2020, 1, 1), date(2024, 12, 31), "annual")
    assert len(dates) == 5
    assert dates[0] == date(2020, 1, 1)
    assert dates[1] == date(2021, 1, 1)


def test_generate_rebalance_dates_unknown_freq_defaults_quarterly():
    dates = _generate_rebalance_dates(date(2024, 1, 1), date(2024, 12, 31), "unknown")
    assert len(dates) == 4  # Same as quarterly


def test_generate_rebalance_dates_end_of_month():
    # Jan 31 + 1 month = Feb 28/29
    dates = _generate_rebalance_dates(date(2024, 1, 31), date(2024, 6, 30), "monthly")
    assert dates[0] == date(2024, 1, 31)
    assert dates[1] == date(2024, 2, 29)  # 2024 is leap year


def test_empty_screen_result():
    result = empty_screen_result()
    assert result.is_empty()
    assert "ticker" in result.columns
    assert "score" in result.columns
    assert "signal_type" in result.columns
    assert "rationale" in result.columns


class ConcreteStrategy(Strategy):
    """Minimal concrete strategy for testing the base class."""

    @property
    def name(self) -> str:
        return "Test"

    @property
    def description(self) -> str:
        return "Test strategy"

    def screen(self, prices, fundamentals, as_of_date, extra_data=None):
        return pl.DataFrame({
            "ticker": ["AAPL"],
            "score": [0.9],
            "signal_type": ["BUY"],
            "rationale": ["test"],
        })


def test_strategy_default_rebalance_frequency():
    s = ConcreteStrategy()
    assert s.rebalance_frequency == "quarterly"


def test_generate_signals():
    s = ConcreteStrategy()
    prices = pl.DataFrame()
    fund = pl.DataFrame()
    result = s.generate_signals(
        prices, fund,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )
    assert not result.is_empty()
    assert "date" in result.columns
    assert "ticker" in result.columns


def test_generate_signals_with_custom_rebalance_dates():
    s = ConcreteStrategy()
    result = s.generate_signals(
        pl.DataFrame(), pl.DataFrame(),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        rebalance_dates=[date(2024, 3, 1), date(2024, 9, 1)],
    )
    dates = result["date"].to_list()
    assert date(2024, 3, 1) in dates
    assert date(2024, 9, 1) in dates
