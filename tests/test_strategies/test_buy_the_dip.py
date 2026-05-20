"""Tests for Buy the Dip Strategy."""

from datetime import date, timedelta

import polars as pl

from ck_trading.strategies.buy_the_dip import BuyTheDipStrategy, compute_dip_signal


def make_prices(ticker: str, start: date, closes: list[float]) -> pl.DataFrame:
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


EMPTY_FUNDAMENTALS = pl.DataFrame()
AS_OF = date(2024, 1, 1)


class TestBuyTheDipStrategy:
    def test_properties(self):
        s = BuyTheDipStrategy()
        assert s.name == "Buy the Dip"
        assert s.rebalance_frequency == "monthly"
        assert "Buy" in s.description

    def _dip_series(self, peak: float, dip_pct: float) -> list[float]:
        """Build a price series: 220 flat days at 100, 30-day rally to peak,
        a 10-day pullback to (1 - dip_pct) * peak, and a small bounce."""
        flat = [100.0] * 220
        rally = [100.0 + (peak - 100.0) * i / 30 for i in range(1, 31)]
        trough = peak * (1 - dip_pct)
        pullback = [peak - (peak - trough) * j / 10 for j in range(1, 11)]
        bounce = [pullback[-1] * 1.01]
        return flat + rally + pullback + bounce

    def test_uptrend_with_pullback_triggers_buy(self):
        """Rally to a peak then a 12% pullback with a 1-day bounce → BUY."""
        closes = self._dip_series(peak=120.0, dip_pct=0.12)
        prices = make_prices("UPTREND", date(2023, 1, 1), closes)
        s = BuyTheDipStrategy(
            lookback=60, min_dip_pct=0.10, trend_period=200, reversal_window=3
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert not result.is_empty()
        assert result["signal_type"][0] == "BUY"
        assert "from" in result["rationale"][0]

    def test_no_pullback_no_signal(self):
        """A steadily rising stock with no dip should produce no signal."""
        closes = [100.0 * (1.002 ** i) for i in range(260)]
        prices = make_prices("RISE", date(2023, 1, 1), closes)
        s = BuyTheDipStrategy(lookback=60, min_dip_pct=0.10)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()

    def test_downtrend_filtered_out(self):
        """A persistent downtrend has a dip from peak but fails the SMA filter."""
        closes = [100.0 * (0.99 ** i) for i in range(260)]
        prices = make_prices("DOWN", date(2023, 1, 1), closes)
        s = BuyTheDipStrategy(
            lookback=60, min_dip_pct=0.10, trend_period=200, require_uptrend=True
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()

    def test_falling_knife_filtered_by_reversal(self):
        """Still making new lows: reversal window blocks the BUY."""
        uptrend = [100.0 * (1.001 ** i) for i in range(240)]
        # Falling every day, last day is the new low
        pullback = [uptrend[-1] * (1 - 0.02 * i) for i in range(1, 12)]
        closes = uptrend + pullback

        prices = make_prices("KNIFE", date(2023, 1, 1), closes)
        s = BuyTheDipStrategy(
            lookback=60, min_dip_pct=0.10, trend_period=200, reversal_window=5
        )
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()

    def test_top_n_limits_results(self):
        """Only top_n candidates are returned, sorted by dip depth."""
        frames = []
        for i, dip in enumerate([0.11, 0.13, 0.15, 0.17, 0.19]):
            closes = self._dip_series(peak=150.0, dip_pct=dip)
            frames.append(make_prices(f"T{i}", date(2023, 1, 1), closes))

        prices = pl.concat(frames)
        s = BuyTheDipStrategy(top_n=3, min_dip_pct=0.10)
        result = s.screen(prices, EMPTY_FUNDAMENTALS, AS_OF)

        assert result.height == 3
        scores = result["score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_empty_prices(self):
        s = BuyTheDipStrategy()
        result = s.screen(pl.DataFrame(), EMPTY_FUNDAMENTALS, AS_OF)
        assert result.is_empty()
        assert set(result.columns) == {"ticker", "score", "signal_type", "rationale"}

    def test_insufficient_history_returns_none(self):
        """Fewer than `needed` closes yields no signal."""
        closes = [100.0] * 20
        result = compute_dip_signal(
            closes,
            lookback=60,
            min_dip_pct=0.10,
            trend_period=200,
            reversal_window=5,
            require_uptrend=True,
        )
        assert result is None

    def test_compute_dip_signal_basic(self):
        # 200 flat closes around 100 (so SMA200 ~ 100), then a clean -15% dip,
        # then a small bounce; current > min(last 3 closes) for reversal.
        closes = [100.0] * 200 + [85.0, 84.0, 86.0]
        result = compute_dip_signal(
            closes,
            lookback=60,
            min_dip_pct=0.10,
            trend_period=200,
            reversal_window=3,
            # SMA200 would be ~100, current 86 is below, so disable uptrend filter
            # for this micro-test focused on dip math.
            require_uptrend=False,
        )
        assert result is not None
        assert result["peak"] == 100.0
        assert result["current"] == 86.0
        assert abs(result["dip_pct"] - 0.14) < 1e-9
