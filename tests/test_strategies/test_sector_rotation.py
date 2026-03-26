"""Tests for Sector Rotation Strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.sector_rotation import SectorRotationStrategy
from ck_trading.strategies.base import empty_screen_result


EMPTY_FUND = pl.DataFrame()
AS_OF = date(2022, 1, 1)


def make_prices_trend(
    ticker: str,
    start: date,
    days: int,
    start_price: float,
    daily_change: float,
) -> pl.DataFrame:
    """Generate synthetic price data with a constant daily growth rate."""
    dates = [start + timedelta(days=i) for i in range(days)]
    prices_list = [start_price * (1 + daily_change) ** i for i in range(days)]
    return pl.DataFrame({
        "ticker": [ticker] * days,
        "date": dates,
        "open": prices_list,
        "high": [p * 1.01 for p in prices_list],
        "low": [p * 0.99 for p in prices_list],
        "close": prices_list,
        "volume": [1_000_000] * days,
    })


def make_macro(records: list[dict]) -> pl.DataFrame:
    """Build a macro DataFrame with columns [date, value, series_id]."""
    return pl.DataFrame(records).with_columns(pl.col("date").cast(pl.Date))


def make_etf_prices(etfs: list[str], daily_changes: list[float]) -> pl.DataFrame:
    """Create price data for a list of ETFs starting 60 days before AS_OF."""
    start = AS_OF - timedelta(days=60)
    frames = []
    for etf, dc in zip(etfs, daily_changes):
        frames.append(make_prices_trend(etf, start, 60, 100.0, dc))
    return pl.concat(frames)


class TestSectorRotationStrategy:
    def test_properties(self):
        s = SectorRotationStrategy()
        assert s.name == "Sector Rotation"
        assert s.rebalance_frequency == "monthly"
        assert s.description

    def test_inverted_yield_defensive(self):
        macro = make_macro([
            {"date": date(2021, 12, 30), "value": -0.5, "series_id": "T10Y2Y"},
            {"date": date(2021, 12, 30), "value": 20.0, "series_id": "VIXCLS"},
        ])
        prices = make_etf_prices(["QQQ", "ARKG", "TLT", "VTV"],
                                  [0.003, 0.002, 0.001, 0.0015])

        s = SectorRotationStrategy()
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"macro": macro})

        assert not result.is_empty()
        tickers = result["ticker"].to_list()
        # Inverted yield curve -> risk-off -> defensive
        assert "TLT" in tickers or "VTV" in tickers
        assert "QQQ" not in tickers
        assert "ARKG" not in tickers

    def test_high_vix_defensive(self):
        macro = make_macro([
            {"date": date(2021, 12, 30), "value": 1.0, "series_id": "T10Y2Y"},
            {"date": date(2021, 12, 30), "value": 40.0, "series_id": "VIXCLS"},
        ])
        prices = make_etf_prices(["QQQ", "ARKG", "TLT", "VTV"],
                                  [0.003, 0.002, 0.001, 0.0015])

        s = SectorRotationStrategy(vix_threshold=30)
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"macro": macro})

        assert not result.is_empty()
        tickers = result["ticker"].to_list()
        assert "TLT" in tickers or "VTV" in tickers
        assert "QQQ" not in tickers

    def test_normal_offensive(self):
        macro = make_macro([
            {"date": date(2021, 12, 30), "value": 1.5, "series_id": "T10Y2Y"},
            {"date": date(2021, 12, 30), "value": 15.0, "series_id": "VIXCLS"},
        ])
        prices = make_etf_prices(["QQQ", "ARKG", "TLT", "VTV"],
                                  [0.003, 0.002, 0.001, 0.0015])

        s = SectorRotationStrategy(vix_threshold=30)
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"macro": macro})

        assert not result.is_empty()
        tickers = result["ticker"].to_list()
        assert "QQQ" in tickers or "ARKG" in tickers
        assert "TLT" not in tickers

    def test_no_macro_returns_empty(self):
        s = SectorRotationStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUND, AS_OF, extra_data=None)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_etf_not_in_prices_skipped(self):
        macro = make_macro([
            {"date": date(2021, 12, 30), "value": 1.5, "series_id": "T10Y2Y"},
            {"date": date(2021, 12, 30), "value": 15.0, "series_id": "VIXCLS"},
        ])
        # Only QQQ in prices, ARKG missing
        prices = make_etf_prices(["QQQ"], [0.003])

        s = SectorRotationStrategy()
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"macro": macro})

        # Should work with just QQQ, not crash
        assert not result.is_empty()
        assert "ARKG" not in result["ticker"].to_list()
        assert "QQQ" in result["ticker"].to_list()

    def test_all_buy_signals(self):
        macro = make_macro([
            {"date": date(2021, 12, 30), "value": 1.5, "series_id": "T10Y2Y"},
            {"date": date(2021, 12, 30), "value": 15.0, "series_id": "VIXCLS"},
        ])
        prices = make_etf_prices(["QQQ", "ARKG"], [0.003, 0.002])

        s = SectorRotationStrategy()
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"macro": macro})

        assert not result.is_empty()
        signals = result["signal_type"].to_list()
        assert all(sig == "BUY" for sig in signals)

    def test_score_by_momentum(self):
        macro = make_macro([
            {"date": date(2021, 12, 30), "value": 1.5, "series_id": "T10Y2Y"},
            {"date": date(2021, 12, 30), "value": 15.0, "series_id": "VIXCLS"},
        ])
        # QQQ has higher momentum than ARKG
        prices = make_etf_prices(["QQQ", "ARKG"], [0.005, 0.001])

        s = SectorRotationStrategy()
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"macro": macro})

        assert len(result) == 2
        scores = dict(zip(result["ticker"].to_list(), result["score"].to_list()))
        assert scores["QQQ"] > scores["ARKG"]

    def test_custom_etf_mapping(self):
        macro = make_macro([
            {"date": date(2021, 12, 30), "value": -0.5, "series_id": "T10Y2Y"},
            {"date": date(2021, 12, 30), "value": 20.0, "series_id": "VIXCLS"},
        ])
        # Defensive regime (inverted yield), custom ETFs
        prices = make_etf_prices(["SPY", "AGG"], [0.002, 0.001])

        s = SectorRotationStrategy(offensive_etfs=["SPY"], defensive_etfs=["AGG"])
        result = s.screen(prices, EMPTY_FUND, AS_OF,
                          extra_data={"macro": macro})

        assert not result.is_empty()
        tickers = result["ticker"].to_list()
        # Risk-off -> defensive -> AGG selected
        assert "AGG" in tickers
        assert "SPY" not in tickers
