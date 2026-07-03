"""Tests for AT&T monitor pure metric functions."""

import datetime as dt
from datetime import date

import polars as pl
import pytest

from ck_trading.monitoring.att_config import DEFAULT_SCENARIOS
from ck_trading.monitoring.att_metrics import (
    compute_weekly_closes,
    fundamentals_series,
    normalized_performance,
    quarter_period_start,
    validate_fundamentals,
    validate_scenarios,
    weekly_close_series,
    weighted_fair_value,
)


def _daily(rows):
    return pl.DataFrame(
        rows,
        schema={"ticker": pl.Utf8, "date": pl.Date, "close": pl.Float64},
    )


class TestWeeklyCloses:
    def test_last_close_per_week(self):
        # 2026-06-29 is a Monday
        daily = _daily([
            {"ticker": "T", "date": date(2026, 6, 29), "close": 20.5},
            {"ticker": "T", "date": date(2026, 6, 30), "close": 20.7},
            {"ticker": "T", "date": date(2026, 7, 2), "close": 20.9},
        ])
        out = compute_weekly_closes(daily)
        assert out.height == 1
        row = out.row(0, named=True)
        assert row["close"] == 20.9  # last close of the week
        assert row["price_date"] == date(2026, 7, 2)
        assert row["iso_week"] == "2026-W27"
        assert row["week_start"] == date(2026, 6, 29)

    def test_multiple_weeks_and_tickers(self):
        daily = _daily([
            {"ticker": "T", "date": date(2026, 6, 22), "close": 21.0},
            {"ticker": "T", "date": date(2026, 6, 29), "close": 20.7},
            {"ticker": "VZ", "date": date(2026, 6, 29), "close": 40.0},
        ])
        out = compute_weekly_closes(daily)
        assert out.height == 3
        assert set(out["iso_week"].to_list()) == {"2026-W26", "2026-W27"}

    def test_null_close_dropped(self):
        daily = pl.DataFrame({
            "ticker": ["T", "T"],
            "date": [date(2026, 6, 29), date(2026, 6, 30)],
            "close": [None, 20.7],
        })
        out = compute_weekly_closes(daily)
        assert out.height == 1
        assert out["close"][0] == 20.7

    def test_empty(self):
        assert compute_weekly_closes(pl.DataFrame()).is_empty()


class TestWeeklySeries:
    def test_series_extraction(self):
        weekly = compute_weekly_closes(_daily([
            {"ticker": "T", "date": date(2026, 6, 22), "close": 21.0},
            {"ticker": "T", "date": date(2026, 6, 29), "close": 20.7},
            {"ticker": "VZ", "date": date(2026, 6, 29), "close": 40.0},
        ]))
        s = weekly_close_series(weekly, "T")
        assert s["value"].to_list() == [21.0, 20.7]
        assert s["period_key"].to_list() == ["2026-W26", "2026-W27"]

    def test_missing_ticker_empty(self):
        assert weekly_close_series(pl.DataFrame(), "T").is_empty()


class TestNormalizedPerformance:
    def test_base_100_at_first_week(self):
        weekly = compute_weekly_closes(_daily([
            {"ticker": "T", "date": date(2026, 6, 22), "close": 20.0},
            {"ticker": "T", "date": date(2026, 6, 29), "close": 22.0},
            {"ticker": "SPCX", "date": date(2026, 6, 29), "close": 162.0},
        ]))
        out = normalized_performance(weekly, ["T", "SPCX"])
        t = out.filter(pl.col("ticker") == "T")["norm"].to_list()
        assert t[0] == pytest.approx(100.0)
        assert t[1] == pytest.approx(110.0)
        # SPCX aligns from its own first week
        spcx = out.filter(pl.col("ticker") == "SPCX")["norm"].to_list()
        assert spcx == [pytest.approx(100.0)]


class TestQuarterMapping:
    def test_all_quarters(self):
        assert quarter_period_start("2026Q1") == date(2026, 1, 1)
        assert quarter_period_start("2026Q2") == date(2026, 4, 1)
        assert quarter_period_start("2026Q3") == date(2026, 7, 1)
        assert quarter_period_start("2026Q4") == date(2026, 10, 1)

    def test_bad_labels_raise(self):
        for bad in ("2026Q5", "26Q1", "2026-Q1", "Q1", ""):
            with pytest.raises(ValueError):
                quarter_period_start(bad)


class TestFundamentalsSeries:
    QUARTERS = [
        {"quarter": "2025Q4", "churn_pct": 0.80, "fcf_billions": 5.0},
        {"quarter": "2026Q1", "churn_pct": 0.83, "fcf_billions": None},
        {"quarter": "bogus", "churn_pct": 9.9},
    ]

    def test_null_and_malformed_dropped(self):
        churn = fundamentals_series(self.QUARTERS, "churn_pct")
        assert churn["value"].to_list() == [0.80, 0.83]  # bogus dropped
        fcf = fundamentals_series(self.QUARTERS, "fcf_billions")
        assert fcf["value"].to_list() == [5.0]  # null dropped

    def test_sorted_by_period(self):
        out = fundamentals_series(
            [{"quarter": "2026Q1", "churn_pct": 2.0},
             {"quarter": "2025Q3", "churn_pct": 1.0}],
            "churn_pct",
        )
        assert out["period_key"].to_list() == ["2025Q3", "2026Q1"]

    def test_empty(self):
        assert fundamentals_series([], "churn_pct").is_empty()


class TestValidateFundamentals:
    def test_valid(self):
        assert validate_fundamentals([
            {"quarter": "2026Q1", "churn_pct": 0.83, "convergence_pct": 45.0},
        ]) == []

    def test_bad_label(self):
        errs = validate_fundamentals([{"quarter": "2026-1"}])
        assert any("YYYYQ1-4" in e for e in errs)

    def test_duplicate(self):
        errs = validate_fundamentals([
            {"quarter": "2026Q1"}, {"quarter": "2026Q1"},
        ])
        assert any("重复" in e for e in errs)

    def test_negative_and_non_numeric(self):
        errs = validate_fundamentals([
            {"quarter": "2026Q1", "churn_pct": -1, "fcf_billions": "abc"},
        ])
        assert any("负数" in e for e in errs)
        assert any("不是数字" in e for e in errs)


class TestScenarios:
    def test_default_weighted_fair(self):
        assert weighted_fair_value(list(DEFAULT_SCENARIOS)) == pytest.approx(23.30)

    def test_defaults_valid(self):
        assert validate_scenarios(list(DEFAULT_SCENARIOS)) == []

    def test_sum_not_100_fails(self):
        s = [dict(x) for x in DEFAULT_SCENARIOS]
        s[0]["probability_pct"] = 44.0
        errs = validate_scenarios(s)
        assert any("100%" in e for e in errs)

    def test_low_above_implied_fails(self):
        s = [dict(x) for x in DEFAULT_SCENARIOS]
        s[0]["price_low"] = 29.0  # > implied 28
        errs = validate_scenarios(s)
        assert any("low" in e for e in errs)

    def test_duplicate_id_fails(self):
        s = [dict(x) for x in DEFAULT_SCENARIOS]
        s[1]["id"] = s[0]["id"]
        errs = validate_scenarios(s)
        assert any("重复" in e for e in errs)
