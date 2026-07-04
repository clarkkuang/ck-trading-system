"""Tests for NVDA framework pure metric functions."""

import datetime as dt
from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.monitoring.nvda_config import DEFAULT_SCENARIOS
from ck_trading.monitoring.nvda_metrics import (
    daily_indicators,
    forward_pe_series,
    latest_forward_eps,
    normalize_daily,
    normalized_performance,
    validate_fundamentals,
    weekly_close_series,
    weekly_sample,
)
from ck_trading.monitoring.quarters import (
    validate_scenarios,
    weighted_fair_value,
)


def _daily(ticker: str, closes: list[float], start: date = date(2025, 1, 6)) -> pl.DataFrame:
    """Consecutive weekdays (skip weekends) starting at a Monday."""
    rows = []
    d = start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        rows.append({"ticker": ticker, "date": d, "close": c})
        d += timedelta(days=1)
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "date": pl.Date, "close": pl.Float64}
    )


class TestNormalizeDaily:
    def test_drops_null_and_dedupes(self):
        df = pl.DataFrame({
            "ticker": ["NVDA", "NVDA", "NVDA"],
            "date": [date(2026, 6, 1), date(2026, 6, 1), date(2026, 6, 2)],
            "close": [None, 100.0, 101.0],
        })
        out = normalize_daily(df)
        assert out.height == 2
        assert out["close"].to_list() == [100.0, 101.0]

    def test_empty(self):
        assert normalize_daily(pl.DataFrame()).is_empty()


class TestDailyIndicators:
    def test_rolling_high_and_dip(self):
        # 250 flat @100, then ramp to 200 over 60d, then drop to 170
        closes = [100.0] * 250 + [100 + (i + 1) * (100 / 60) for i in range(60)] + [170.0]
        df = _daily("NVDA", closes)
        out = daily_indicators(df, "NVDA")
        last = out.tail(1).row(0, named=True)
        assert last["high_60d"] == pytest.approx(200.0)
        assert last["dip_pct"] == pytest.approx((200 - 170) / 200)  # 15%

    def test_warmup_rows_null_not_zero(self):
        df = _daily("NVDA", [100.0] * 50)  # < 60 rows
        out = daily_indicators(df, "NVDA")
        assert out["high_60d"].null_count() == 50
        assert out["gated_dip"].null_count() == 50  # null, NOT 0.0

    def test_gate_open_equals_dip(self):
        # long uptrend keeps close > sma_200; dip = gated_dip
        closes = [100 + i * 0.5 for i in range(260)]
        closes[-1] = closes[-2] * 0.90  # 10% single-day drop, still above sma
        df = _daily("NVDA", closes)
        out = daily_indicators(df, "NVDA").tail(1).row(0, named=True)
        assert out["pct_vs_200dma"] > 0
        assert out["gated_dip"] == pytest.approx(out["dip_pct"])

    def test_gate_closed_zero(self):
        # long downtrend: close < sma_200 -> gated_dip == 0.0
        closes = [300 - i * 0.5 for i in range(260)]
        df = _daily("NVDA", closes)
        out = daily_indicators(df, "NVDA").tail(1).row(0, named=True)
        assert out["pct_vs_200dma"] < 0
        assert out["dip_pct"] > 0
        assert out["gated_dip"] == 0.0

    def test_duplicate_dates_deduped(self):
        df = pl.DataFrame({
            "ticker": ["NVDA"] * 2,
            "date": [date(2026, 6, 1)] * 2,
            "close": [100.0, 105.0],
        })
        out = daily_indicators(df, "NVDA")
        assert out.height == 1
        assert out["close"][0] == 105.0


class TestWeeklySample:
    def test_last_per_iso_week_and_key_format(self):
        # Mon 2026-06-29 .. Thu 2026-07-02 => one week "2026-W27"
        df = pl.DataFrame({
            "date": [date(2026, 6, 29), date(2026, 6, 30), date(2026, 7, 2)],
            "v": [1.0, 2.0, 3.0],
        })
        out = weekly_sample(df, "v")
        assert out.height == 1
        row = out.row(0, named=True)
        assert row["period_key"] == "2026-W27"
        assert row["period_start"] == date(2026, 6, 29)
        assert row["value"] == 3.0

    def test_null_rows_excluded(self):
        df = pl.DataFrame({
            "date": [date(2026, 6, 29), date(2026, 7, 2)],
            "v": [None, 3.0],
        })
        out = weekly_sample(df, "v")
        assert out["value"].to_list() == [3.0]

    def test_empty(self):
        assert weekly_sample(pl.DataFrame(schema={"date": pl.Date, "v": pl.Float64}), "v").is_empty()


class TestForwardPE:
    QUARTERS = [
        {"quarter": "2026Q2", "forward_eps_consensus": 7.0},
    ]

    def _prices(self):
        # weeks around the 2026Q2 period start (2026-04-01)
        return pl.DataFrame({
            "ticker": ["NVDA"] * 3,
            "date": [date(2026, 3, 23), date(2026, 4, 6), date(2026, 4, 13)],
            "close": [190.0, 210.0, 217.0],
        })

    def test_weeks_before_first_eps_dropped(self):
        s = forward_pe_series(self._prices(), "NVDA", self.QUARTERS)
        # 2026-03-23 week precedes 2026Q2 period_start (04-01) -> dropped
        assert s.height == 2
        assert s["value"].to_list() == [pytest.approx(30.0), pytest.approx(31.0)]

    def test_asof_switch_on_second_entry(self):
        quarters = [
            {"quarter": "2026Q2", "forward_eps_consensus": 7.0},
            {"quarter": "2026Q3", "forward_eps_consensus": 8.0},
        ]
        prices = pl.DataFrame({
            "ticker": ["NVDA"] * 2,
            "date": [date(2026, 6, 22), date(2026, 7, 6)],  # before/after 07-01
            "close": [210.0, 216.0],
        })
        s = forward_pe_series(prices, "NVDA", quarters)
        assert s["value"].to_list() == [pytest.approx(30.0), pytest.approx(27.0)]

    def test_no_eps_empty(self):
        assert forward_pe_series(self._prices(), "NVDA", []).is_empty()
        assert forward_pe_series(
            self._prices(), "NVDA",
            [{"quarter": "2026Q2", "forward_eps_consensus": None}],
        ).is_empty()
        assert forward_pe_series(
            self._prices(), "NVDA",
            [{"quarter": "2026Q2", "forward_eps_consensus": 0}],
        ).is_empty()

    def test_latest_forward_eps(self):
        quarters = [
            {"quarter": "2026Q3", "forward_eps_consensus": 8.0},
            {"quarter": "2026Q2", "forward_eps_consensus": 7.0},
            {"quarter": "2026Q4", "forward_eps_consensus": None},
        ]
        assert latest_forward_eps(quarters) == ("2026Q3", 8.0)
        assert latest_forward_eps([]) is None


class TestValidateFundamentals:
    def test_negative_growth_allowed(self):
        errs = validate_fundamentals([{
            "quarter": "2026Q3",
            "dc_qoq_growth_pct": -5.0,
            "guide_vs_consensus_pct": -2.0,
        }])
        assert errs == []

    def test_negative_revenue_rejected(self):
        errs = validate_fundamentals([{
            "quarter": "2026Q3", "dc_revenue_billions": -1.0,
        }])
        assert any("负数" in e for e in errs)

    def test_margin_range(self):
        errs = validate_fundamentals([{
            "quarter": "2026Q3", "gross_margin_pct": 105.0,
        }])
        assert any("[0,100]" in e for e in errs)

    def test_label_and_dup(self):
        errs = validate_fundamentals([
            {"quarter": "FY27Q2"},
            {"quarter": "2026Q3"}, {"quarter": "2026Q3"},
        ])
        assert any("YYYYQ1-4" in e for e in errs)
        assert any("重复" in e for e in errs)


class TestScenarios:
    def test_weighted_fair(self):
        assert weighted_fair_value(list(DEFAULT_SCENARIOS)) == pytest.approx(223.75)

    def test_defaults_valid(self):
        assert validate_scenarios(list(DEFAULT_SCENARIOS)) == []


class TestNormalizedPerformance:
    def test_base_100(self):
        prices = pl.concat([
            _daily("NVDA", [200.0, 210.0, 220.0, 230.0, 240.0, 250.0]),
            _daily("SMH", [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        ])
        out = normalized_performance(prices, ["NVDA", "SMH"])
        nvda = out.filter(pl.col("ticker") == "NVDA")["norm"].to_list()
        assert nvda[0] == pytest.approx(100.0)


class TestSharedCompat:
    def test_att_metrics_reexport_still_works(self):
        from ck_trading.monitoring.att_metrics import quarter_period_start as a
        from ck_trading.monitoring.quarters import quarter_period_start as b
        assert a is b
