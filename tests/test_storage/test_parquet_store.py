"""Tests for ParquetStore."""

from datetime import date

import polars as pl
import pytest

from ck_trading.storage.parquet_store import ParquetStore


@pytest.fixture
def store(tmp_path):
    return ParquetStore(base_dir=tmp_path)


@pytest.fixture
def price_df():
    return pl.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2), date(2024, 1, 3)],
        "open": [185.0, 186.0, 370.0, 372.0],
        "high": [186.0, 187.0, 372.0, 374.0],
        "low": [184.0, 185.0, 369.0, 371.0],
        "close": [185.5, 186.5, 371.0, 373.0],
        "volume": [50_000_000, 48_000_000, 25_000_000, 26_000_000],
        "adj_close": [185.5, 186.5, 371.0, 373.0],
    })


class TestSaveAndLoadPrices:
    def test_save_and_load_all(self, store, price_df):
        store.save_prices(price_df, market="us")
        loaded = store.load_prices(market="us")
        assert loaded.shape[0] == 4
        assert set(loaded["ticker"].unique().to_list()) == {"AAPL", "MSFT"}

    def test_load_specific_tickers(self, store, price_df):
        store.save_prices(price_df, market="us")
        loaded = store.load_prices(market="us", tickers=["AAPL"])
        assert loaded.shape[0] == 2
        assert loaded["ticker"].unique().to_list() == ["AAPL"]

    def test_load_nonexistent_market(self, store):
        loaded = store.load_prices(market="jp")
        assert loaded.is_empty()

    def test_save_empty_df(self, store):
        store.save_prices(pl.DataFrame(), market="us")
        loaded = store.load_prices(market="us")
        assert loaded.is_empty()

    def test_merge_deduplicates(self, store):
        df1 = pl.DataFrame({
            "ticker": ["AAPL", "AAPL"],
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "open": [185.0, 186.0],
            "high": [186.0, 187.0],
            "low": [184.0, 185.0],
            "close": [185.5, 186.5],
            "volume": [50_000_000, 48_000_000],
            "adj_close": [185.5, 186.5],
        })
        df2 = pl.DataFrame({
            "ticker": ["AAPL", "AAPL"],
            "date": [date(2024, 1, 3), date(2024, 1, 4)],
            "open": [186.0, 187.0],
            "high": [187.0, 188.0],
            "low": [185.0, 186.0],
            "close": [186.8, 187.5],
            "volume": [49_000_000, 47_000_000],
            "adj_close": [186.8, 187.5],
        })
        store.save_prices(df1, market="us")
        store.save_prices(df2, market="us")
        loaded = store.load_prices(market="us", tickers=["AAPL"])
        assert loaded.shape[0] == 3  # Jan 2, 3, 4 — Jan 3 deduplicated


class TestSaveAndLoadFundamentals:
    def test_save_and_load(self, store, sample_fundamentals):
        store.save_fundamentals(sample_fundamentals, market="us")
        loaded = store.load_fundamentals(market="us")
        assert not loaded.is_empty()
        assert "AAPL" in loaded["ticker"].to_list()

    def test_load_specific_tickers(self, store, sample_fundamentals):
        store.save_fundamentals(sample_fundamentals, market="us")
        loaded = store.load_fundamentals(market="us", tickers=["MSFT"])
        assert loaded["ticker"].unique().to_list() == ["MSFT"]

    def test_load_nonexistent_market(self, store):
        loaded = store.load_fundamentals(market="jp")
        assert loaded.is_empty()

    def test_save_empty_df(self, store):
        store.save_fundamentals(pl.DataFrame(), market="us")
        loaded = store.load_fundamentals(market="us")
        assert loaded.is_empty()


class TestSaveAndLoadMacro:
    def test_save_and_load(self, store):
        macro = pl.DataFrame({
            "series_id": ["GDP", "GDP", "CPI", "CPI"],
            "date": [date(2024, 1, 1), date(2024, 4, 1), date(2024, 1, 1), date(2024, 4, 1)],
            "value": [27.5, 28.0, 308.0, 310.0],
        })
        store.save_macro(macro)
        loaded = store.load_macro()
        assert loaded.shape[0] == 4

    def test_load_nonexistent(self, store):
        loaded = store.load_macro(name="nonexistent")
        assert loaded.is_empty()


class TestSaveAndLoadAlternative:
    def test_save_and_load(self, store):
        alt = pl.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "insider_buy_count": [5, 3],
        })
        store.save_alternative(alt, source="insider", name="trades")
        loaded = store.load_alternative(source="insider", name="trades")
        assert loaded.shape[0] == 2

    def test_load_filtered_by_ticker(self, store):
        alt = pl.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "value": [1, 2, 3],
        })
        store.save_alternative(alt, source="test", name="data")
        loaded = store.load_alternative(source="test", name="data", tickers=["AAPL", "GOOG"])
        assert loaded.shape[0] == 2

    def test_load_nonexistent(self, store):
        loaded = store.load_alternative(source="none", name="none")
        assert loaded.is_empty()
