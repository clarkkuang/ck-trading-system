"""Tests for stock_pool_selector underlying logic.

The widget itself uses Streamlit UI (st.selectbox, st.multiselect) which
cannot be unit-tested easily.  We verify the data-fetching paths that the
widget relies on and the module-level constants.
"""

from datetime import date

import pytest

from ck_trading.dashboard.widgets.stock_pool_selector import POOL_OPTIONS
from ck_trading.models.market_data import Market
from ck_trading.models.portfolio import Position
from ck_trading.storage.metadata_store import MetadataStore


@pytest.fixture
def meta(tmp_path):
    db_path = tmp_path / "test_metadata.db"
    store = MetadataStore(str(db_path))
    yield store
    store.close()


def test_pool_options_constant():
    """POOL_OPTIONS should contain the four expected pool names."""
    assert "Full Universe" in POOL_OPTIONS
    assert "Watchlist" in POOL_OPTIONS
    assert "Portfolio Holdings" in POOL_OPTIONS
    assert "Custom" in POOL_OPTIONS


def test_full_universe_path(meta: MetadataStore):
    """'Full Universe' path: get_universe returns all tickers."""
    meta.save_universe([
        {"ticker": "AAPL", "market": "US", "name": "Apple", "sector": "Tech", "industry": "CE"},
        {"ticker": "MSFT", "market": "US", "name": "Microsoft", "sector": "Tech", "industry": "SW"},
    ])
    tickers = [u["ticker"] for u in meta.get_universe()]
    assert set(tickers) == {"AAPL", "MSFT"}


def test_watchlist_path(meta: MetadataStore):
    """'Watchlist' path: get_watchlist returns added tickers."""
    meta.add_to_watchlist("GOOG")
    meta.add_to_watchlist("TSLA")
    tickers = [w["ticker"] for w in meta.get_watchlist()]
    assert set(tickers) == {"GOOG", "TSLA"}


def test_portfolio_holdings_path(meta: MetadataStore):
    """'Portfolio Holdings' path: get_open_positions returns unique tickers."""
    meta.save_position(Position(
        ticker="AAPL", market=Market.US, shares=10, avg_cost=150.0, date_acquired=date(2025, 1, 1),
    ))
    meta.save_position(Position(
        ticker="AAPL", market=Market.US, shares=20, avg_cost=155.0, date_acquired=date(2025, 2, 1),
    ))
    positions = meta.get_open_positions()
    tickers = list({p["ticker"] for p in positions})
    assert tickers == ["AAPL"]


def test_empty_watchlist_returns_empty(meta: MetadataStore):
    """Watchlist path returns empty list when nothing is added."""
    assert meta.get_watchlist() == []


def test_empty_positions_returns_empty(meta: MetadataStore):
    """Portfolio path returns empty list when no positions exist."""
    assert meta.get_open_positions() == []
