"""Tests for Earnings Surprise Strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.earnings_surprise import EarningsSurpriseStrategy
from ck_trading.strategies.base import empty_screen_result


EMPTY_PRICES = pl.DataFrame()
EMPTY_FUND = pl.DataFrame()
AS_OF = date(2022, 1, 1)


def make_earnings(records: list[dict]) -> pl.DataFrame:
    """Build an earnings DataFrame from a list of dicts."""
    return pl.DataFrame(records).with_columns(pl.col("date").cast(pl.Date))


class TestEarningsSurpriseStrategy:
    def test_properties(self):
        s = EarningsSurpriseStrategy()
        assert s.name == "Earnings Surprise"
        assert s.rebalance_frequency == "quarterly"
        assert s.description

    def test_consecutive_beats_selected(self):
        earnings = make_earnings([
            {"ticker": "AAPL", "date": date(2021, 4, 1), "surprise_pct": 0.05},
            {"ticker": "AAPL", "date": date(2021, 7, 1), "surprise_pct": 0.06},
            {"ticker": "AAPL", "date": date(2021, 10, 1), "surprise_pct": 0.07},
        ])
        s = EarningsSurpriseStrategy(min_consecutive_beats=2, min_surprise_pct=0.03)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"earnings": earnings})

        assert not result.is_empty()
        assert "AAPL" in result["ticker"].to_list()

    def test_single_beat_excluded(self):
        earnings = make_earnings([
            {"ticker": "TSLA", "date": date(2021, 10, 1), "surprise_pct": 0.08},
        ])
        s = EarningsSurpriseStrategy(min_consecutive_beats=2, min_surprise_pct=0.03)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"earnings": earnings})

        assert result.is_empty()

    def test_negative_surprise_breaks_streak(self):
        earnings = make_earnings([
            {"ticker": "MSFT", "date": date(2021, 4, 1), "surprise_pct": 0.05},
            {"ticker": "MSFT", "date": date(2021, 7, 1), "surprise_pct": -0.02},  # miss
            {"ticker": "MSFT", "date": date(2021, 10, 1), "surprise_pct": 0.06},
        ])
        # Most recent sorted desc: 2021-10 (beat), 2021-07 (miss) -> streak=1
        s = EarningsSurpriseStrategy(min_consecutive_beats=2, min_surprise_pct=0.03)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"earnings": earnings})

        assert result.is_empty()

    def test_no_extra_data_returns_empty(self):
        s = EarningsSurpriseStrategy()
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF, extra_data=None)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_ranking_by_avg_surprise(self):
        earnings = make_earnings([
            {"ticker": "AAPL", "date": date(2021, 7, 1), "surprise_pct": 0.10},
            {"ticker": "AAPL", "date": date(2021, 10, 1), "surprise_pct": 0.10},
            {"ticker": "MSFT", "date": date(2021, 7, 1), "surprise_pct": 0.05},
            {"ticker": "MSFT", "date": date(2021, 10, 1), "surprise_pct": 0.05},
        ])
        s = EarningsSurpriseStrategy(min_consecutive_beats=2, min_surprise_pct=0.03)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"earnings": earnings})

        assert len(result) == 2
        tickers = result["ticker"].to_list()
        assert tickers[0] == "AAPL"
        assert tickers[1] == "MSFT"
        scores = result["score"].to_list()
        assert scores[0] > scores[1]

    def test_small_surprise_filtered(self):
        earnings = make_earnings([
            {"ticker": "NFLX", "date": date(2021, 7, 1), "surprise_pct": 0.01},
            {"ticker": "NFLX", "date": date(2021, 10, 1), "surprise_pct": 0.02},
        ])
        # Both surprises < min_surprise_pct of 3%, so no beats counted
        s = EarningsSurpriseStrategy(min_consecutive_beats=2, min_surprise_pct=0.03)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"earnings": earnings})

        assert result.is_empty()

    def test_empty_earnings_returns_empty(self):
        earnings = pl.DataFrame(schema={"ticker": pl.Utf8, "date": pl.Date,
                                         "surprise_pct": pl.Float64})
        s = EarningsSurpriseStrategy()
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"earnings": earnings})

        assert result.is_empty()

    def test_top_n_respected(self):
        records = []
        for i in range(5):
            ticker = f"T{i}"
            records.extend([
                {"ticker": ticker, "date": date(2021, 7, 1),
                 "surprise_pct": 0.05 + i * 0.01},
                {"ticker": ticker, "date": date(2021, 10, 1),
                 "surprise_pct": 0.05 + i * 0.01},
            ])
        earnings = make_earnings(records)

        s = EarningsSurpriseStrategy(min_consecutive_beats=2, min_surprise_pct=0.03,
                                      top_n=2)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"earnings": earnings})

        assert len(result) == 2

    def test_rationale_contains_streak_info(self):
        earnings = make_earnings([
            {"ticker": "AAPL", "date": date(2021, 7, 1), "surprise_pct": 0.05},
            {"ticker": "AAPL", "date": date(2021, 10, 1), "surprise_pct": 0.06},
        ])
        s = EarningsSurpriseStrategy(min_consecutive_beats=2, min_surprise_pct=0.03)
        result = s.screen(EMPTY_PRICES, EMPTY_FUND, AS_OF,
                          extra_data={"earnings": earnings})

        assert not result.is_empty()
        rationale = result["rationale"][0]
        # Should mention streak count and/or avg surprise
        assert "consecutive" in rationale.lower() or "beat" in rationale.lower()
        assert "%" in rationale
