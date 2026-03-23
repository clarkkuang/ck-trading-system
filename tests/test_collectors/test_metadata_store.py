"""Tests for MetadataStore – watchlist, positions, and universe methods."""

from datetime import date

import pytest

from ck_trading.models.market_data import Market
from ck_trading.models.portfolio import Position
from ck_trading.storage.metadata_store import MetadataStore


@pytest.fixture
def meta(tmp_path):
    db_path = tmp_path / "test_metadata.db"
    store = MetadataStore(str(db_path))
    yield store
    store.close()


# --- Watchlist ---


def test_add_to_watchlist_with_source(meta: MetadataStore):
    """Adding a ticker with an explicit source stores that source."""
    meta.add_to_watchlist("AAPL", market="US", source="portfolio")
    rows = meta.get_watchlist()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["source"] == "portfolio"


def test_add_to_watchlist_default_source(meta: MetadataStore):
    """Default source should be 'manual'."""
    meta.add_to_watchlist("MSFT")
    rows = meta.get_watchlist()
    assert len(rows) == 1
    assert rows[0]["source"] == "manual"


def test_add_to_watchlist_duplicate_idempotent(meta: MetadataStore):
    """Adding the same ticker twice should not crash or create duplicates."""
    meta.add_to_watchlist("GOOG", market="US", source="manual")
    meta.add_to_watchlist("GOOG", market="US", source="manual")
    rows = meta.get_watchlist()
    assert len(rows) == 1


def test_add_to_watchlist_source_field_in_result(meta: MetadataStore):
    """get_watchlist() entries must contain the 'source' key."""
    meta.add_to_watchlist("TSLA", source="screening")
    meta.add_to_watchlist("NVDA", source="manual")
    rows = meta.get_watchlist()
    assert all("source" in r for r in rows)
    sources = {r["ticker"]: r["source"] for r in rows}
    assert sources["TSLA"] == "screening"
    assert sources["NVDA"] == "manual"


# --- Positions ---


def test_save_position_basic(meta: MetadataStore):
    """Save a position and retrieve it via get_open_positions."""
    pos = Position(
        ticker="AAPL",
        market=Market.US,
        shares=100,
        avg_cost=150.0,
        date_acquired=date(2025, 1, 15),
    )
    row_id = meta.save_position(pos)
    assert isinstance(row_id, int)

    positions = meta.get_open_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "AAPL"
    assert positions[0]["shares"] == 100
    assert positions[0]["avg_cost"] == 150.0


def test_get_open_positions_excludes_closed(meta: MetadataStore):
    """get_open_positions returns only non-closed positions."""
    pos1 = Position(
        ticker="AAPL",
        market=Market.US,
        shares=50,
        avg_cost=140.0,
        date_acquired=date(2025, 1, 1),
    )
    pos2 = Position(
        ticker="GOOG",
        market=Market.US,
        shares=20,
        avg_cost=2800.0,
        date_acquired=date(2025, 2, 1),
    )
    meta.save_position(pos1)
    pos2_id = meta.save_position(pos2)

    # Close one position directly via SQL
    meta.conn.execute(
        "UPDATE positions SET is_closed = 1 WHERE id = ?", (pos2_id,)
    )
    meta.conn.commit()

    open_positions = meta.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0]["ticker"] == "AAPL"


# --- Universe ---


def test_get_universe(meta: MetadataStore):
    """save_universe + get_universe round-trip works."""
    tickers = [
        {"ticker": "AAPL", "market": "US", "name": "Apple", "sector": "Tech", "industry": "Consumer Electronics"},
        {"ticker": "0700.HK", "market": "HK", "name": "Tencent", "sector": "Tech", "industry": "Internet"},
    ]
    meta.save_universe(tickers)

    result = meta.get_universe()
    assert len(result) == 2
    ticker_set = {r["ticker"] for r in result}
    assert ticker_set == {"AAPL", "0700.HK"}


def test_get_universe_filter_by_market(meta: MetadataStore):
    """get_universe(market=...) filters correctly."""
    tickers = [
        {"ticker": "AAPL", "market": "US", "name": "Apple", "sector": "Tech", "industry": "Consumer Electronics"},
        {"ticker": "MSFT", "market": "US", "name": "Microsoft", "sector": "Tech", "industry": "Software"},
        {"ticker": "0700.HK", "market": "HK", "name": "Tencent", "sector": "Tech", "industry": "Internet"},
    ]
    meta.save_universe(tickers)

    us = meta.get_universe(market="US")
    assert len(us) == 2
    assert all(r["market"] == "US" for r in us)

    hk = meta.get_universe(market="HK")
    assert len(hk) == 1
    assert hk[0]["ticker"] == "0700.HK"
