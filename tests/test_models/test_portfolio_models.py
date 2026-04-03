"""Tests for portfolio and position models."""

from datetime import date, datetime

from ck_trading.models.market_data import Market
from ck_trading.models.portfolio import (
    PortfolioSnapshot,
    Position,
    Trade,
    TradeAction,
)


def test_trade_action_enum():
    assert TradeAction.BUY == "BUY"
    assert TradeAction.SELL == "SELL"


def test_position_cost_basis():
    pos = Position(
        ticker="AAPL",
        market=Market.US,
        shares=100,
        avg_cost=150.0,
        date_acquired=date(2024, 1, 15),
    )
    assert pos.cost_basis == 15_000.0


def test_position_market_value():
    pos = Position(
        ticker="AAPL",
        market=Market.US,
        shares=100,
        avg_cost=150.0,
        date_acquired=date(2024, 1, 15),
        current_price=180.0,
    )
    assert pos.market_value == 18_000.0


def test_position_market_value_none_when_no_price():
    pos = Position(
        ticker="AAPL",
        market=Market.US,
        shares=100,
        avg_cost=150.0,
        date_acquired=date(2024, 1, 15),
    )
    assert pos.market_value is None


def test_position_unrealized_pnl():
    pos = Position(
        ticker="AAPL",
        market=Market.US,
        shares=100,
        avg_cost=150.0,
        date_acquired=date(2024, 1, 15),
        current_price=180.0,
    )
    assert pos.unrealized_pnl == 3_000.0
    assert pos.unrealized_pnl_pct == 3_000.0 / 15_000.0


def test_position_unrealized_pnl_none_when_no_price():
    pos = Position(
        ticker="AAPL",
        market=Market.US,
        shares=100,
        avg_cost=150.0,
        date_acquired=date(2024, 1, 15),
    )
    assert pos.unrealized_pnl is None
    assert pos.unrealized_pnl_pct is None


def test_position_negative_pnl():
    pos = Position(
        ticker="AAPL",
        market=Market.US,
        shares=50,
        avg_cost=200.0,
        date_acquired=date(2024, 1, 15),
        current_price=180.0,
    )
    assert pos.unrealized_pnl == -1_000.0
    assert pos.unrealized_pnl_pct < 0


def test_trade_defaults():
    trade = Trade(
        ticker="AAPL",
        market=Market.US,
        action=TradeAction.BUY,
        shares=100,
        price=150.0,
        date=date(2024, 1, 15),
    )
    assert trade.fees == 0.0
    assert trade.notes == ""


def test_portfolio_snapshot():
    pos = Position(
        ticker="AAPL",
        market=Market.US,
        shares=100,
        avg_cost=150.0,
        date_acquired=date(2024, 1, 15),
        current_price=180.0,
    )
    snap = PortfolioSnapshot(
        timestamp=datetime(2024, 3, 15, 16, 0),
        positions=[pos],
        cash=50_000.0,
        total_value=68_000.0,
    )
    assert len(snap.positions) == 1
    assert snap.cash == 50_000.0
    assert snap.daily_pnl is None
