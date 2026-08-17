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
