"""Tests for Insider Following Strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.insider_following import InsiderFollowingStrategy
from ck_trading.strategies.base import empty_screen_result


EMPTY_PRICES = pl.DataFrame()
EMPTY_FUND = pl.DataFrame()
AS_OF = date(2022, 1, 1)


def make_insider_trades(records: list[dict]) -> pl.DataFrame:
    """Build an insider_trades DataFrame from a list of dicts."""
    return pl.DataFrame(records).with_columns(
        pl.col("date").cast(pl.Date),
    )


class TestInsiderFollowingStrategy:
    def test_properties(self):
        s = InsiderFollowingStrategy()
        assert s.name == "Insider Following"
        assert s.rebalance_frequency == "monthly"
        assert s.description

    def test_net_buyer_selected(self):
        trades = make_insider_trades([
            {"ticker": "AAPL", "date": date(2021, 11, 15), "transaction_type": "P",
             "shares": 1000, "price": 100.0, "value": 100_000.0},
        ])
        s = InsiderFollowingStrategy(lookback_days=90, min_buy_value=50_000)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"insider_trades": trades})

        assert not result.is_empty()
        assert "AAPL" in result["ticker"].to_list()

    def test_net_seller_excluded(self):
        trades = make_insider_trades([
            {"ticker": "MSFT", "date": date(2021, 11, 15), "transaction_type": "S",
             "shares": 2000, "price": 100.0, "value": 200_000.0},
        ])
        s = InsiderFollowingStrategy(lookback_days=90, min_buy_value=50_000)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"insider_trades": trades})

        assert result.is_empty()

    def test_small_buy_filtered(self):
        trades = make_insider_trades([
            {"ticker": "TSLA", "date": date(2021, 12, 1), "transaction_type": "P",
             "shares": 100, "price": 100.0, "value": 10_000.0},
        ])
        s = InsiderFollowingStrategy(lookback_days=90, min_buy_value=50_000)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"insider_trades": trades})

        assert result.is_empty()

    def test_no_extra_data_returns_empty(self):
        s = InsiderFollowingStrategy()
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF, extra_data=None)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_wrong_key_returns_empty(self):
        df = pl.DataFrame({"x": [1]})
        s = InsiderFollowingStrategy()
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"wrong_key": df})
        assert result.is_empty()

    def test_lookback_filter(self):
        trades = make_insider_trades([
            # 200 days ago - outside 90-day lookback
            {"ticker": "OLD", "date": date(2021, 6, 15), "transaction_type": "P",
             "shares": 1000, "price": 100.0, "value": 100_000.0},
            # 30 days ago - inside lookback
            {"ticker": "NEW", "date": date(2021, 12, 2), "transaction_type": "P",
             "shares": 1000, "price": 100.0, "value": 100_000.0},
        ])
        s = InsiderFollowingStrategy(lookback_days=90, min_buy_value=50_000)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"insider_trades": trades})

        tickers = result["ticker"].to_list()
        assert "OLD" not in tickers
        assert "NEW" in tickers

    def test_multiple_insiders_aggregated(self):
        trades = make_insider_trades([
            {"ticker": "GOOG", "date": date(2021, 11, 10), "transaction_type": "P",
             "shares": 500, "price": 100.0, "value": 50_000.0},
            {"ticker": "GOOG", "date": date(2021, 11, 20), "transaction_type": "P",
             "shares": 500, "price": 100.0, "value": 50_000.0},
        ])
        s = InsiderFollowingStrategy(lookback_days=90, min_buy_value=50_000)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"insider_trades": trades})

        assert not result.is_empty()
        assert "GOOG" in result["ticker"].to_list()
        # Aggregated value should be 100k
        scores = dict(zip(result["ticker"].to_list(), result["score"].to_list()))
        assert scores["GOOG"] == pytest.approx(100_000.0)

    def test_ranking_by_net_value(self):
        trades = make_insider_trades([
            {"ticker": "AAPL", "date": date(2021, 12, 1), "transaction_type": "P",
             "shares": 2000, "price": 100.0, "value": 200_000.0},
            {"ticker": "MSFT", "date": date(2021, 12, 1), "transaction_type": "P",
             "shares": 1000, "price": 100.0, "value": 100_000.0},
        ])
        s = InsiderFollowingStrategy(lookback_days=90, min_buy_value=50_000)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"insider_trades": trades})

        assert len(result) == 2
        tickers = result["ticker"].to_list()
        assert tickers[0] == "AAPL"
        assert tickers[1] == "MSFT"
