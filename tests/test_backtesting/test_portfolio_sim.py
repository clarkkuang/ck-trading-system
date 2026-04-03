"""Tests for portfolio simulation utilities."""

from datetime import date

from ck_trading.backtesting.portfolio_sim import (
    equal_weight_allocation,
    generate_rebalance_schedule,
    score_weighted_allocation,
)


class TestEqualWeightAllocation:
    def test_basic(self):
        alloc = equal_weight_allocation(
            capital=100_000,
            tickers=["AAPL", "MSFT"],
            prices={"AAPL": 150.0, "MSFT": 370.0},
        )
        assert "AAPL" in alloc
        assert "MSFT" in alloc
        # 50k / 150 = 333 shares for AAPL
        assert alloc["AAPL"] == 333
        # 50k / 370 = 135 shares for MSFT
        assert alloc["MSFT"] == 135

    def test_empty_tickers(self):
        alloc = equal_weight_allocation(100_000, [], {"AAPL": 150.0})
        assert alloc == {}

    def test_max_positions_limit(self):
        tickers = [f"T{i}" for i in range(30)]
        prices = {t: 100.0 for t in tickers}
        alloc = equal_weight_allocation(100_000, tickers, prices, max_positions=5)
        assert len(alloc) == 5

    def test_zero_price_excluded(self):
        alloc = equal_weight_allocation(
            100_000, ["AAPL", "BAD"], {"AAPL": 150.0, "BAD": 0.0}
        )
        assert "AAPL" in alloc
        assert "BAD" not in alloc

    def test_missing_price_excluded(self):
        alloc = equal_weight_allocation(100_000, ["AAPL", "MISSING"], {"AAPL": 150.0})
        assert "AAPL" in alloc
        assert "MISSING" not in alloc


class TestScoreWeightedAllocation:
    def test_basic(self):
        alloc = score_weighted_allocation(
            capital=100_000,
            ticker_scores={"AAPL": 0.8, "MSFT": 0.2},
            prices={"AAPL": 150.0, "MSFT": 370.0},
        )
        assert "AAPL" in alloc
        assert "MSFT" in alloc
        # AAPL gets 80% of capital: 80k/150 = 533
        assert alloc["AAPL"] == 533
        # MSFT gets 20% of capital: 20k/370 = 54
        assert alloc["MSFT"] == 54

    def test_empty_scores(self):
        alloc = score_weighted_allocation(100_000, {}, {"AAPL": 150.0})
        assert alloc == {}

    def test_zero_scores_fallback_to_equal_weight(self):
        alloc = score_weighted_allocation(
            100_000,
            {"AAPL": 0.0, "MSFT": 0.0},
            {"AAPL": 150.0, "MSFT": 370.0},
        )
        # Falls back to equal weight
        assert "AAPL" in alloc
        assert "MSFT" in alloc

    def test_max_positions(self):
        scores = {f"T{i}": float(i) for i in range(1, 11)}
        prices = {f"T{i}": 100.0 for i in range(1, 11)}
        alloc = score_weighted_allocation(100_000, scores, prices, max_positions=3)
        assert len(alloc) == 3
        # Top 3 by score: T10, T9, T8
        assert "T10" in alloc
        assert "T9" in alloc
        assert "T8" in alloc


class TestGenerateRebalanceSchedule:
    def test_quarterly(self):
        trading_dates = [
            date(2024, 1, 2), date(2024, 1, 3),
            date(2024, 4, 1), date(2024, 4, 2),
            date(2024, 7, 1), date(2024, 7, 2),
            date(2024, 10, 1), date(2024, 10, 2),
        ]
        schedule = generate_rebalance_schedule(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            trading_dates=trading_dates,
            frequency="quarterly",
        )
        assert len(schedule) > 0
        # All rebalance dates should be in trading_dates
        for d in schedule:
            assert d in trading_dates

    def test_empty_trading_dates(self):
        schedule = generate_rebalance_schedule(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            trading_dates=[],
            frequency="quarterly",
        )
        assert schedule == []
