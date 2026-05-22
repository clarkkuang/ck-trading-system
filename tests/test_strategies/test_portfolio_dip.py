"""Tests for the Portfolio-Tailored Buy the Dip Strategy."""

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.portfolio_dip import (
    DEFAULT_PORTFOLIO_THRESHOLDS,
    PortfolioDipStrategy,
)


def _make_prices(ticker: str, start: date, closes: list[float]) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(len(closes))]
    return pl.DataFrame({
        "ticker": [ticker] * len(closes),
        "date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000] * len(closes),
        "adj_close": closes,
    })


def _dip_series(peak: float = 200.0, dip_pct: float = 0.10) -> list[float]:
    """100 flat days → 150-day rally to peak → 10-day pullback → 1-day bounce.

    The long rally ensures the 200-day SMA is well below current price
    even after the pullback, so the uptrend filter passes.
    """
    flat = [100.0] * 100
    rally = [100.0 + (peak - 100.0) * i / 150 for i in range(1, 151)]
    trough = peak * (1 - dip_pct)
    pullback = [peak - (peak - trough) * j / 10 for j in range(1, 11)]
    bounce = [pullback[-1] * 1.01]
    return flat + rally + pullback + bounce


EMPTY_FUNDAMENTALS = pl.DataFrame()
AS_OF = date(2024, 1, 1)


class TestPortfolioDipStrategy:
    def test_properties(self):
        s = PortfolioDipStrategy()
        assert s.name == "Portfolio Dip"
        assert s.rebalance_frequency == "monthly"
        assert "Per-ticker" in s.description

    def test_registered(self):
        from ck_trading.strategies.registry import get_strategy
        s = get_strategy("Portfolio Dip")
        assert isinstance(s, PortfolioDipStrategy)

    def test_default_thresholds_loaded(self):
        s = PortfolioDipStrategy()
        # Spot-check a few calibrated values
        assert s.threshold_for("VOO") == DEFAULT_PORTFOLIO_THRESHOLDS["VOO"]
        assert s.threshold_for("TSLA") == DEFAULT_PORTFOLIO_THRESHOLDS["TSLA"]

    def test_unknown_ticker_falls_back_to_default(self):
        s = PortfolioDipStrategy(default_min_dip_pct=0.12)
        assert s.threshold_for("UNKNOWN_TICKER") == 0.12

    def test_per_ticker_override(self):
        s = PortfolioDipStrategy(dip_thresholds={"VOO": 0.04})
        assert s.threshold_for("VOO") == 0.04

    def test_low_vol_ticker_triggers_on_small_dip(self):
        """VOO at 8% dip should trigger (threshold 5.5%).

        Note: the 1-day bounce reduces effective dip-from-peak by ~1%,
        so we leave a safety margin above the 5.5% threshold.
        """
        closes = _dip_series(peak=200.0, dip_pct=0.08)
        prices = _make_prices("VOO", date(2023, 1, 1), closes)
        s = PortfolioDipStrategy()
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert not result.is_empty()
        assert result["ticker"][0] == "VOO"

    def test_high_vol_ticker_needs_deeper_dip(self):
        """TSLA at 6% dip should NOT trigger (threshold 23.3%)."""
        closes = _dip_series(peak=110.0, dip_pct=0.06)
        prices = _make_prices("TSLA", date(2023, 1, 1), closes)
        s = PortfolioDipStrategy()
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()

    def test_high_vol_ticker_triggers_on_deep_dip(self):
        """TSLA at 25% dip exceeds 23.3% threshold."""
        closes = _dip_series(peak=200.0, dip_pct=0.25)
        prices = _make_prices("TSLA", date(2023, 1, 1), closes)
        s = PortfolioDipStrategy()
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert not result.is_empty()

    def test_cross_ticker_scoring_normalized(self):
        """A ticker hitting 2× its threshold should outrank one hitting 1.2×."""
        # Both built with the long-rally helper so they pass the SMA200 filter.
        # VOO at 12% (≈2.18× its 5.5% threshold)
        voo = _make_prices("VOO", date(2023, 1, 1), _dip_series(peak=200, dip_pct=0.12))
        # NVDA at 16% (≈1.26× its 12.7% threshold)
        nvda = _make_prices("NVDA", date(2023, 1, 1), _dip_series(peak=200, dip_pct=0.16))
        prices = pl.concat([voo, nvda])

        s = PortfolioDipStrategy()
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.height == 2
        # VOO should rank higher because it's deeper relative to its own threshold
        assert result["ticker"][0] == "VOO"

    def test_falling_knife_blocked_by_reversal(self):
        """Still making new lows — reversal filter blocks the signal."""
        uptrend = [100.0 * (1.001 ** i) for i in range(240)]
        # Every day lower than the last for 12 days
        pullback = [uptrend[-1] * (1 - 0.02 * i) for i in range(1, 12)]
        closes = uptrend + pullback
        prices = _make_prices("VOO", date(2023, 1, 1), closes)
        s = PortfolioDipStrategy(reversal_window=5)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()

    def test_top_n_limit(self):
        frames = []
        for i, dip in enumerate([0.08, 0.10, 0.12, 0.15, 0.20]):
            closes = _dip_series(peak=200.0, dip_pct=dip)
            frames.append(_make_prices(f"T{i}", date(2023, 1, 1), closes))
        prices = pl.concat(frames)
        # Use a uniform low threshold so all 5 trigger
        s = PortfolioDipStrategy(
            dip_thresholds={f"T{i}": 0.05 for i in range(5)},
            top_n=2,
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.height == 2

    def test_empty_prices(self):
        s = PortfolioDipStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_uptrend_filter_optional(self):
        """When require_uptrend=False, sideways/downtrend dips still trigger."""
        # 250 days drifting down then a sharp 12% pullback off the recent high
        downtrend = [100.0 * (0.999 ** i) for i in range(250)]
        pullback = [downtrend[-1] * (1 - 0.04 * i) for i in range(1, 4)]
        bounce = [pullback[-1] * 1.01]
        closes = downtrend + pullback + bounce

        prices = _make_prices("VOO", date(2023, 1, 1), closes)
        # Override threshold so we don't have to hit 5.5% which may be borderline
        s = PortfolioDipStrategy(
            dip_thresholds={"VOO": 0.08},
            require_uptrend=False,
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        # Should trigger since uptrend not required
        assert not result.is_empty()

    def test_uptrend_filter_when_enabled(self):
        """With require_uptrend=True, prices below SMA200 are blocked."""
        # Same downtrend series
        downtrend = [100.0 * (0.999 ** i) for i in range(250)]
        pullback = [downtrend[-1] * (1 - 0.04 * i) for i in range(1, 4)]
        bounce = [pullback[-1] * 1.01]
        closes = downtrend + pullback + bounce
        prices = _make_prices("VOO", date(2023, 1, 1), closes)
        s = PortfolioDipStrategy(
            dip_thresholds={"VOO": 0.08},
            require_uptrend=True,
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        # Drifting downward means current < SMA200 → filtered
        assert result.is_empty()
