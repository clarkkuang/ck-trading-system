"""Tests for DuckDBStore."""

from datetime import date

import polars as pl
import pytest

from ck_trading.storage.duckdb_store import DuckDBStore
from ck_trading.storage.parquet_store import ParquetStore


@pytest.fixture
def data_dir(tmp_path):
    """Set up a tmp data dir with parquet files for DuckDB to query."""
    pstore = ParquetStore(base_dir=tmp_path)
    prices = pl.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
        "open": [185.0, 186.0, 370.0],
        "high": [186.0, 187.0, 372.0],
        "low": [184.0, 185.0, 369.0],
        "close": [185.5, 186.5, 371.0],
        "volume": [50_000_000, 48_000_000, 25_000_000],
        "adj_close": [185.5, 186.5, 371.0],
    })
    pstore.save_prices(prices, market="us")
    return tmp_path


@pytest.fixture
def duckdb_store(data_dir):
    db_path = data_dir / "test.duckdb"
    store = DuckDBStore(db_path=db_path, data_dir=data_dir)
    yield store
    store.close()


def test_query_prices(duckdb_store):
    result = duckdb_store.get_prices(
        tickers=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        market="us",
    )
    assert result.shape[0] == 2
    assert result["ticker"].unique().to_list() == ["AAPL"]


def test_query_all_tickers(duckdb_store):
    result = duckdb_store.query("SELECT DISTINCT ticker FROM us_prices ORDER BY ticker")
    tickers = result["ticker"].to_list()
    assert tickers == ["AAPL", "MSFT"]


def test_get_latest_prices(duckdb_store):
    result = duckdb_store.get_latest_prices(market="us")
    assert result.shape[0] == 2
    # AAPL latest should be Jan 3
    aapl = result.filter(pl.col("ticker") == "AAPL")
    assert aapl["date"].to_list()[0] == date(2024, 1, 3)


def test_screen_query(duckdb_store):
    result = duckdb_store.screen(
        "SELECT ticker, close FROM us_prices WHERE close > 300"
    )
    assert all(t == "MSFT" for t in result["ticker"].to_list())


def test_context_manager(data_dir):
    db_path = data_dir / "ctx.duckdb"
    with DuckDBStore(db_path=db_path, data_dir=data_dir) as store:
        result = store.query("SELECT COUNT(*) AS cnt FROM us_prices")
        assert result["cnt"].to_list()[0] == 3


def test_refresh_views(duckdb_store, data_dir):
    # Add more data and refresh
    pstore = ParquetStore(base_dir=data_dir)
    new_prices = pl.DataFrame({
        "ticker": ["GOOG"],
        "date": [date(2024, 1, 2)],
        "open": [140.0],
        "high": [142.0],
        "low": [139.0],
        "close": [141.0],
        "volume": [20_000_000],
        "adj_close": [141.0],
    })
    pstore.save_prices(new_prices, market="us")
    duckdb_store.refresh_views()
    result = duckdb_store.query("SELECT DISTINCT ticker FROM us_prices ORDER BY ticker")
    assert "GOOG" in result["ticker"].to_list()
