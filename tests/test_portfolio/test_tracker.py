"""Tests for PortfolioTracker."""

from datetime import date

import pytest

from ck_trading.models.market_data import Market
from ck_trading.models.portfolio import Trade, TradeAction
from ck_trading.portfolio.tracker import PortfolioTracker
from ck_trading.storage.metadata_store import MetadataStore


@pytest.fixture
def tracker(tmp_path):
    store = MetadataStore(db_path=tmp_path / "test.db")
    return PortfolioTracker(metadata_store=store)


def test_buy_creates_position(tracker):
    trade = Trade(
        ticker="AAPL", market=Market.US, action=TradeAction.BUY,
        shares=100, price=150.0, date=date(2024, 1, 15),
    )
    tracker.add_trade(trade)
    positions = tracker.get_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "AAPL"
    assert positions[0]["shares"] == 100


def test_buy_averages_cost(tracker):
    tracker.add_trade(Trade(
        ticker="AAPL", market=Market.US, action=TradeAction.BUY,
        shares=100, price=150.0, date=date(2024, 1, 15),
    ))
    tracker.add_trade(Trade(
        ticker="AAPL", market=Market.US, action=TradeAction.BUY,
        shares=100, price=200.0, date=date(2024, 2, 15),
    ))
    positions = tracker.get_positions()
    assert len(positions) == 1
    assert positions[0]["shares"] == 200
    assert positions[0]["avg_cost"] == pytest.approx(175.0)


def test_sell_reduces_shares(tracker):
    tracker.add_trade(Trade(
        ticker="AAPL", market=Market.US, action=TradeAction.BUY,
        shares=100, price=150.0, date=date(2024, 1, 15),
    ))
    tracker.add_trade(Trade(
        ticker="AAPL", market=Market.US, action=TradeAction.SELL,
        shares=40, price=180.0, date=date(2024, 3, 15),
    ))
    positions = tracker.get_positions()
    assert len(positions) == 1
    assert positions[0]["shares"] == 60


def test_sell_all_closes_position(tracker):
    tracker.add_trade(Trade(
        ticker="AAPL", market=Market.US, action=TradeAction.BUY,
        shares=100, price=150.0, date=date(2024, 1, 15),
    ))
    tracker.add_trade(Trade(
        ticker="AAPL", market=Market.US, action=TradeAction.SELL,
        shares=100, price=180.0, date=date(2024, 3, 15),
    ))
    positions = tracker.get_positions()
    assert len(positions) == 0


def test_calculate_portfolio_value(tracker):
    tracker.add_trade(Trade(
        ticker="AAPL", market=Market.US, action=TradeAction.BUY,
        shares=100, price=150.0, date=date(2024, 1, 15),
    ))
    tracker.add_trade(Trade(
        ticker="MSFT", market=Market.US, action=TradeAction.BUY,
        shares=50, price=370.0, date=date(2024, 1, 15),
    ))
    result = tracker.calculate_portfolio_value({"AAPL": 180.0, "MSFT": 400.0})
    assert result["total_cost"] == pytest.approx(100 * 150 + 50 * 370)
    assert result["total_value"] == pytest.approx(100 * 180 + 50 * 400)
    assert result["total_pnl"] > 0


def test_calculate_portfolio_value_empty(tracker):
    result = tracker.calculate_portfolio_value({"AAPL": 180.0})
    assert result["total_cost"] == 0.0
    assert result["total_value"] == 0.0
    assert result["total_pnl"] == 0.0
