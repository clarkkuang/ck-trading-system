"""Tests for MetadataStore."""

from datetime import date, datetime

import pytest

from ck_trading.models.market_data import Market
from ck_trading.models.portfolio import Position, Trade, TradeAction
from ck_trading.models.signals import Signal, SignalType
from ck_trading.storage.metadata_store import MetadataStore


@pytest.fixture
def store(tmp_path):
    return MetadataStore(db_path=tmp_path / "test_metadata.db")


class TestSignals:
    def test_save_and_retrieve_signal(self, store):
        sig = Signal(
            ticker="AAPL",
            signal_type=SignalType.BUY,
            strategy_name="Graham",
            score=0.85,
            generated_at=datetime(2024, 1, 15, 10, 30),
            price_at_signal=185.0,
        )
        row_id = store.save_signal(sig)
        assert row_id > 0

        signals = store.get_recent_signals(limit=10)
        assert len(signals) == 1
        assert signals[0]["ticker"] == "AAPL"
        assert signals[0]["strategy_name"] == "Graham"

    def test_recent_signals_ordering(self, store):
        for i, ticker in enumerate(["AAPL", "MSFT", "GOOG"]):
            store.save_signal(Signal(
                ticker=ticker,
                signal_type=SignalType.BUY,
                strategy_name="Test",
                score=0.5 + i * 0.1,
                generated_at=datetime(2024, 1, 15 + i, 10, 0),
            ))
        signals = store.get_recent_signals(limit=2)
        assert len(signals) == 2
        # Most recent first
        assert signals[0]["ticker"] == "GOOG"


class TestPositions:
    def test_save_and_get_position(self, store):
        pos = Position(
            ticker="AAPL",
            market=Market.US,
            shares=100,
            avg_cost=150.0,
            date_acquired=date(2024, 1, 15),
        )
        row_id = store.save_position(pos)
        assert row_id > 0

        positions = store.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["ticker"] == "AAPL"
        assert positions[0]["shares"] == 100


class TestTrades:
    def test_save_trade(self, store):
        trade = Trade(
            ticker="AAPL",
            market=Market.US,
            action=TradeAction.BUY,
            shares=100,
            price=150.0,
            date=date(2024, 1, 15),
        )
        row_id = store.save_trade(trade)
        assert row_id > 0


def _ticker_entry(ticker, market="US", name="", sector="Tech", industry=""):
    return {
        "ticker": ticker, "market": market, "name": name,
        "sector": sector, "industry": industry,
    }


class TestUniverse:
    def test_save_and_get_universe(self, store):
        tickers = [
            _ticker_entry("AAPL", name="Apple", industry="Electronics"),
            _ticker_entry("MSFT", name="Microsoft", industry="Software"),
        ]
        store.save_universe(tickers)
        universe = store.get_universe()
        assert len(universe) == 2

    def test_get_universe_by_market(self, store):
        tickers = [
            _ticker_entry("AAPL", name="Apple", industry="Electronics"),
            _ticker_entry("0700.HK", market="HK", name="Tencent", industry="Internet"),
        ]
        store.save_universe(tickers)
        us = store.get_universe(market="US")
        assert len(us) == 1
        assert us[0]["ticker"] == "AAPL"

    def test_upsert_universe(self, store):
        tickers = [_ticker_entry("AAPL", name="Apple", industry="Electronics")]
        store.save_universe(tickers)
        tickers[0]["name"] = "Apple Inc."
        store.save_universe(tickers)
        universe = store.get_universe()
        assert len(universe) == 1
        assert universe[0]["name"] == "Apple Inc."


class TestWatchlist:
    def test_add_and_get(self, store):
        store.add_to_watchlist("AAPL", market="US", notes="Value play")
        wl = store.get_watchlist()
        assert len(wl) == 1
        assert wl[0]["ticker"] == "AAPL"

    def test_ignore_duplicates(self, store):
        store.add_to_watchlist("AAPL")
        store.add_to_watchlist("AAPL")
        wl = store.get_watchlist()
        assert len(wl) == 1


class TestJobRuns:
    def test_log_job(self, store):
        job_id = store.log_job_start("daily_prices")
        assert job_id > 0
        store.log_job_end(job_id, status="success", result="OK")


class TestContextManager:
    def test_context_manager(self, tmp_path):
        with MetadataStore(db_path=tmp_path / "ctx.db") as store:
            store.add_to_watchlist("AAPL")
            wl = store.get_watchlist()
            assert len(wl) == 1
