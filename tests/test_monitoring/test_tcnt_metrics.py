"""Tests for the Tencent exit monitor's pure metric functions."""

from datetime import date

import polars as pl
import pytest

from ck_trading.monitoring.tcnt_metrics import (
    exit_band,
    latest_trailing_eps_hkd,
    trailing_pe_series,
    validate_fundamentals,
)


def _prices(closes, ticker="0700.HK", start=date(2026, 4, 6)):
    from datetime import timedelta
    rows, d = [], start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        rows.append({"ticker": ticker, "date": d, "close": float(c)})
        d += timedelta(days=7)
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "date": pl.Date, "close": pl.Float64}
    )


class TestValidateFundamentals:
    def test_negative_fcf_allowed(self):
        # 2026Q2 was the first negative FCF since 2005 — it must be storable
        errs = validate_fundamentals([
            {"quarter": "2026Q2", "free_cash_flow_billions": -13.8}
        ])
        assert errs == []

    def test_negative_profit_growth_allowed(self):
        errs = validate_fundamentals([
            {"quarter": "2026Q3", "non_ifrs_profit_yoy_pct": -4.0}
        ])
        assert errs == []

    def test_negative_buyback_rejected(self):
        errs = validate_fundamentals([
            {"quarter": "2026Q3", "buyback_hkd_billions": -1.0}
        ])
        assert any("buyback" in e for e in errs)

    def test_label_and_dup(self):
        errs = validate_fundamentals([
            {"quarter": "26Q2"}, {"quarter": "2026Q3"}, {"quarter": "2026Q3"},
        ])
        assert any("YYYYQ1-4" in e for e in errs)
        assert any("重复" in e for e in errs)


class TestTrailingEps:
    def test_single_quarter_annualised(self):
        got = latest_trailing_eps_hkd([
            {"quarter": "2026Q2", "non_ifrs_profit_billions": 68.4}
        ])
        assert got is not None
        q, eps = got
        assert q == "2026Q2"
        # 68.4 * 4 * 1.09 / 9.126 ~= 32.7 — the design-date anchor
        assert eps == pytest.approx(32.7, abs=0.2)

    def test_four_quarters_summed_not_annualised(self):
        qs = [
            {"quarter": "2025Q3", "non_ifrs_profit_billions": 60.0},
            {"quarter": "2025Q4", "non_ifrs_profit_billions": 62.0},
            {"quarter": "2026Q1", "non_ifrs_profit_billions": 65.0},
            {"quarter": "2026Q2", "non_ifrs_profit_billions": 68.4},
        ]
        _, eps = latest_trailing_eps_hkd(qs)
        assert eps == pytest.approx(255.4 * 1.09 / 9.126, rel=1e-6)

    def test_empty(self):
        assert latest_trailing_eps_hkd([]) is None
        assert latest_trailing_eps_hkd(
            [{"quarter": "2026Q2", "non_ifrs_profit_billions": None}]
        ) is None


class TestTrailingPeSeries:
    QUARTERS = [{"quarter": "2026Q2", "non_ifrs_profit_billions": 68.4}]

    def test_pe_matches_price_over_eps(self):
        s = trailing_pe_series(_prices([446.6]), "0700.HK", self.QUARTERS)
        assert s.height == 1
        assert s["value"][0] == pytest.approx(446.6 / 32.7, abs=0.05)

    def test_weeks_before_first_quarter_dropped(self):
        # weeks of 03-23 and 03-30 precede the 2026Q2 period start (04-01);
        # only the 04-06 week survives rather than being valued on a
        # fabricated EPS
        s = trailing_pe_series(
            _prices([390.0, 400.0, 446.6], start=date(2026, 3, 23)),
            "0700.HK", self.QUARTERS,
        )
        assert s.height == 1
        assert s["value"][0] == pytest.approx(446.6 / 32.7, abs=0.05)

    def test_no_eps_empty(self):
        assert trailing_pe_series(_prices([446.6]), "0700.HK", []).is_empty()
        assert trailing_pe_series(
            _prices([446.6]), "0700.HK",
            [{"quarter": "2026Q2", "non_ifrs_profit_billions": 0}],
        ).is_empty()

    def test_missing_ticker_empty(self):
        assert trailing_pe_series(
            _prices([446.6]), "9988.HK", self.QUARTERS
        ).is_empty()


class TestExitBand:
    def _series(self, lo=12.6, hi=20.9):
        vals = [lo, (lo + hi) / 2, hi] * 20
        return pl.DataFrame({
            "period_key": [f"w{i}" for i in range(len(vals))],
            "period_start": [date(2026, 1, 5)] * len(vals),
            "value": vals,
        })

    def test_design_date_spot_lands_in_lower_third(self):
        """13.7x in a 12.6-20.9x range must be a HALF tranche, not baseline.

        This is the recalibration that was frozen on 2026-08-16.
        """
        got = exit_band(13.7, self._series())
        assert got["multiplier"] == 0.5
        assert got["tranche_pct"] == pytest.approx(2.5)
        assert "下1/3" in got["band"]

    def test_middle_third_is_baseline(self):
        got = exit_band(17.0, self._series())
        assert got["multiplier"] == 1.0
        assert got["tranche_pct"] == pytest.approx(5.0)

    def test_upper_third_accelerates(self):
        got = exit_band(20.0, self._series())
        assert got["multiplier"] == 2.0
        assert got["tranche_pct"] == pytest.approx(10.0)

    def test_clear_half_and_all_override_bands(self):
        assert "清半" in exit_band(23.0, self._series())["band"]
        cleared = exit_band(26.5, self._series())
        assert "清空" in cleared["band"]
        assert cleared["tranche_pct"] == 100.0

    def test_deep_brake_overrides_and_zeroes_tranche(self):
        got = exit_band(9.5, self._series())
        assert got["multiplier"] == 0.0
        assert got["tranche_pct"] == 0.0
        assert "预算" in got["note"]

    def test_no_series_falls_back_to_baseline_not_crash(self):
        got = exit_band(13.7, None)
        assert got["multiplier"] == 1.0
        assert got["tranche_pct"] == pytest.approx(5.0)

    def test_bad_pe_returns_no_data(self):
        assert exit_band(0.0, self._series())["tranche_pct"] is None
        assert exit_band(None, self._series())["tranche_pct"] is None

    def test_band_moves_with_the_range_not_with_price(self):
        """The point of percentile bands: same P/E, different regime, different action."""
        low_regime = exit_band(17.0, self._series(lo=10.0, hi=18.0))
        high_regime = exit_band(17.0, self._series(lo=16.0, hi=30.0))
        assert low_regime["multiplier"] == 2.0    # 17x is near the top of 10-18
        assert high_regime["multiplier"] == 0.5   # 17x is near the bottom of 16-30


class TestRelativeStrength:
    def _pair(self, sub_closes, bmk_closes):
        from datetime import timedelta
        rows, d = [], date(2026, 1, 5)
        for s, b in zip(sub_closes, bmk_closes):
            rows.append({"ticker": "0700.HK", "date": d, "close": float(s)})
            rows.append({"ticker": "KWEB", "date": d, "close": float(b)})
            d += timedelta(days=7)
        return pl.DataFrame(
            rows, schema={"ticker": pl.Utf8, "date": pl.Date, "close": pl.Float64}
        )

    def test_excess_return_in_points(self):
        from ck_trading.monitoring.tcnt_metrics import relative_strength_series
        # subject -20%, benchmark -10% over 4 weeks -> -10pp excess
        sub = [100, 100, 100, 100, 80]
        bmk = [100, 100, 100, 100, 90]
        s = relative_strength_series(self._pair(sub, bmk), "0700.HK", "KWEB", 4)
        assert s.height == 1
        assert s["value"][0] == pytest.approx(-10.0, abs=0.01)

    def test_falling_with_the_sector_is_not_underperformance(self):
        """The Feb-2026 lesson: -14.5% vs a -23% sector must NOT look bad."""
        from ck_trading.monitoring.tcnt_metrics import relative_strength_series
        sub = [100, 100, 100, 100, 85.5]
        bmk = [100, 100, 100, 100, 77.0]
        s = relative_strength_series(self._pair(sub, bmk), "0700.HK", "KWEB", 4)
        assert s["value"][0] > 0   # outperformed while falling

    def test_missing_benchmark_is_empty_not_error(self):
        from ck_trading.monitoring.tcnt_metrics import relative_strength_series
        assert relative_strength_series(
            self._pair([100] * 5, [100] * 5), "0700.HK", "NOPE", 4
        ).is_empty()

    def test_window_longer_than_history_is_empty(self):
        from ck_trading.monitoring.tcnt_metrics import relative_strength_series
        assert relative_strength_series(
            self._pair([100] * 5, [100] * 5), "0700.HK", "KWEB", 52
        ).is_empty()


class TestBuybackBlackout:
    CAL = (("2026-03-18", "FY2025 annual"), ("2026-08-12", "2026Q2 interim"))

    def test_inside_annual_window(self):
        from ck_trading.monitoring.tcnt_metrics import buyback_blackout
        # the real event: buybacks stopped 2026-01-16 for 2026-03-18 results
        got = buyback_blackout(date(2026, 2, 10), self.CAL, 61, 30)
        assert got["in_blackout"] is True
        assert got["results_date"] == date(2026, 3, 18)
        assert got["days_remaining"] == 36

    def test_just_outside_window(self):
        from ck_trading.monitoring.tcnt_metrics import buyback_blackout
        got = buyback_blackout(date(2026, 1, 10), self.CAL, 61, 30)
        assert got["in_blackout"] is False
        assert got["next_window_start"] == date(2026, 1, 16)

    def test_interim_window_is_shorter(self):
        from ck_trading.monitoring.tcnt_metrics import buyback_blackout
        assert buyback_blackout(date(2026, 7, 20), self.CAL, 61, 30)["in_blackout"] is True
        assert buyback_blackout(date(2026, 7, 1), self.CAL, 61, 30)["in_blackout"] is False

    def test_calendar_exhausted_is_reported(self):
        from ck_trading.monitoring.tcnt_metrics import buyback_blackout
        got = buyback_blackout(date(2030, 1, 1), self.CAL, 61, 30)
        assert got["in_blackout"] is False
        assert "补录" in got["note"]
