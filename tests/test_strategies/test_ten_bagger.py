"""Tests for TenBagger Strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from ck_trading.strategies.ten_bagger import TenBaggerStrategy


AS_OF = date(2023, 6, 30)


def make_fund(
    ticker: str,
    market_cap: float = 500e6,
    revenue: float = 100e6,
    gross_margin: float = 0.45,
    debt_to_equity: float = 0.5,
    roe: float = 0.20,
    free_cash_flow: float = 10e6,
    period_end: date | None = None,
) -> dict:
    if period_end is None:
        period_end = AS_OF - timedelta(days=30)
    return {
        "ticker": ticker,
        "market_cap": market_cap,
        "revenue": revenue,
        "gross_profit": revenue * gross_margin,
        "gross_margin": gross_margin,
        "debt_to_equity": debt_to_equity,
        "roe": roe,
        "net_income": revenue * roe * 0.1,
        "free_cash_flow": free_cash_flow,
        "period_end": period_end,
    }


def make_prices(
    ticker: str,
    start_price: float = 10.0,
    end_price: float = 12.0,
    days: int = 130,
    as_of: date = AS_OF,
) -> pl.DataFrame:
    """Generate simple linear prices from start to end."""
    start = as_of - timedelta(days=days)
    dates = [start + timedelta(days=i) for i in range(days + 1)]
    prices = [
        start_price + (end_price - start_price) * i / days
        for i in range(days + 1)
    ]
    return pl.DataFrame({
        "ticker": [ticker] * len(dates),
        "date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * len(dates),
        "adj_close": prices,
    })


def two_period_fund(ticker: str, **kwargs) -> pl.DataFrame:
    """Create two periods: latest with good metrics, prior with lower revenue."""
    latest = make_fund(ticker, **kwargs)
    prior = make_fund(
        ticker,
        period_end=latest["period_end"] - timedelta(days=365),
        revenue=latest["revenue"] / 1.5,  # prior revenue = 33% less → 50% growth
        **{k: v for k, v in kwargs.items() if k not in ("revenue", "period_end")},
    )
    return pl.DataFrame([prior, latest])


EMPTY_PRICES = pl.DataFrame(
    schema={"ticker": pl.Utf8, "date": pl.Date, "open": pl.Float64,
            "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Int64, "adj_close": pl.Float64}
)


class TestTenBaggerStrategy:
    def test_properties(self):
        s = TenBaggerStrategy()
        assert s.name == "Ten Bagger"
        assert "10x" in s.description.lower() or "ten bagger" in s.description.lower()
        assert s.rebalance_frequency == "quarterly"

    def test_basic_candidate_selected(self):
        """A stock meeting all criteria should be selected."""
        fund = two_period_fund("GROW", market_cap=500e6, revenue=100e6,
                               gross_margin=0.45, debt_to_equity=0.5, roe=0.20)
        prices = make_prices("GROW", start_price=10.0, end_price=13.0)
        s = TenBaggerStrategy(top_n=10)
        result = s.screen(prices, fund, AS_OF)
        assert "GROW" in result["ticker"].to_list()

    def test_large_cap_excluded(self):
        """Market cap above max_market_cap should be filtered out."""
        fund = two_period_fund("BIGCO", market_cap=50e9)
        prices = make_prices("BIGCO", start_price=100.0, end_price=120.0)
        s = TenBaggerStrategy(max_market_cap=2e9, top_n=10)
        result = s.screen(prices, fund, AS_OF)
        assert "BIGCO" not in result["ticker"].to_list()

    def test_tiny_cap_excluded(self):
        """Market cap below min_market_cap should be filtered out."""
        fund = two_period_fund("TINY", market_cap=10e6)
        prices = make_prices("TINY")
        s = TenBaggerStrategy(min_market_cap=50e6, top_n=10)
        result = s.screen(prices, fund, AS_OF)
        assert "TINY" not in result["ticker"].to_list()

    def test_low_revenue_growth_excluded(self):
        """Revenue growth below threshold should be excluded."""
        latest = make_fund("SLOW", revenue=100e6)
        prior = make_fund("SLOW", revenue=95e6, period_end=AS_OF - timedelta(days=400))
        fund = pl.DataFrame([prior, latest])
        prices = make_prices("SLOW")
        s = TenBaggerStrategy(min_revenue_growth=0.20, top_n=10)
        result = s.screen(prices, fund, AS_OF)
        assert "SLOW" not in result["ticker"].to_list()

    def test_low_gross_margin_excluded(self):
        """Gross margin below threshold should be excluded."""
        fund = two_period_fund("LOWGM", gross_margin=0.10)
        prices = make_prices("LOWGM")
        s = TenBaggerStrategy(min_gross_margin=0.30, top_n=10)
        result = s.screen(prices, fund, AS_OF)
        assert "LOWGM" not in result["ticker"].to_list()

    def test_high_debt_excluded(self):
        """High debt-to-equity should be excluded."""
        fund = two_period_fund("DEBT", debt_to_equity=3.0)
        prices = make_prices("DEBT")
        s = TenBaggerStrategy(max_debt_to_equity=1.0, top_n=10)
        result = s.screen(prices, fund, AS_OF)
        assert "DEBT" not in result["ticker"].to_list()

    def test_low_roe_excluded(self):
        """Low ROE should be excluded."""
        fund = two_period_fund("LOWROE", roe=0.05)
        prices = make_prices("LOWROE")
        s = TenBaggerStrategy(min_roe=0.15, top_n=10)
        result = s.screen(prices, fund, AS_OF)
        assert "LOWROE" not in result["ticker"].to_list()

    def test_top_n_respected(self):
        """top_n limits the number of results."""
        frames = []
        for i in range(10):
            frames.append(two_period_fund(f"T{i:02d}", market_cap=500e6))
        fund = pl.concat(frames, how="diagonal")
        prices = pl.concat(
            [make_prices(f"T{i:02d}") for i in range(10)], how="diagonal"
        )
        s = TenBaggerStrategy(top_n=3)
        result = s.screen(prices, fund, AS_OF)
        assert result.height <= 3

    def test_all_buy_signals(self):
        """All outputs should be BUY."""
        fund = two_period_fund("BUY1")
        prices = make_prices("BUY1")
        s = TenBaggerStrategy(top_n=10)
        result = s.screen(prices, fund, AS_OF)
        if not result.is_empty():
            assert all(sig == "BUY" for sig in result["signal_type"].to_list())

    def test_higher_growth_ranks_first(self):
        """Stock with higher revenue growth should score higher."""
        fund_hi = two_period_fund("HIGRW", revenue=100e6)  # 50% growth implicit
        # Low growth: prior revenue is 90% of latest (11% growth)
        latest_lo = make_fund("LOGRW", revenue=100e6)
        prior_lo = make_fund("LOGRW", revenue=90e6, period_end=AS_OF - timedelta(days=400))
        fund_lo = pl.DataFrame([prior_lo, latest_lo])
        fund = pl.concat([fund_hi, fund_lo], how="diagonal")
        prices = pl.concat(
            [make_prices("HIGRW"), make_prices("LOGRW")], how="diagonal"
        )
        s = TenBaggerStrategy(min_revenue_growth=0.10, top_n=10)
        result = s.screen(prices, fund, AS_OF)
        tickers = result["ticker"].to_list()
        if "HIGRW" in tickers and "LOGRW" in tickers:
            assert tickers.index("HIGRW") < tickers.index("LOGRW")

    def test_empty_fundamentals_returns_empty(self):
        """Empty fundamentals should return empty result."""
        s = TenBaggerStrategy()
        result = s.screen(EMPTY_PRICES, pl.DataFrame(), AS_OF)
        assert result.is_empty()

    def test_rationale_contains_key_metrics(self):
        """Rationale should include RevGrowth, GM, ROE, Mom."""
        fund = two_period_fund("CHECK")
        prices = make_prices("CHECK")
        s = TenBaggerStrategy(top_n=10)
        result = s.screen(prices, fund, AS_OF)
        if not result.is_empty():
            rationale = result["rationale"][0]
            assert "RevGrowth=" in rationale
            assert "GM=" in rationale
            assert "ROE=" in rationale
            assert "Mom=" in rationale

    def test_insider_buying_boosts_score(self):
        """Stock with insider buying should score higher than identical one without."""
        fund_a = two_period_fund("INS_A")
        fund_b = two_period_fund("INS_B")
        fund = pl.concat([fund_a, fund_b], how="diagonal")
        prices = pl.concat(
            [make_prices("INS_A"), make_prices("INS_B")], how="diagonal"
        )
        # INS_A has insider buying, INS_B does not
        insider_trades = pl.DataFrame({
            "ticker": ["INS_A"],
            "date": [AS_OF - timedelta(days=10)],
            "transaction_type": ["P"],
            "value": [100_000.0],
        })
        extra = {"insider_trades": insider_trades}
        s = TenBaggerStrategy(top_n=10)
        result = s.screen(prices, fund, AS_OF, extra_data=extra)
        if not result.is_empty() and "INS_A" in result["ticker"].to_list() and "INS_B" in result["ticker"].to_list():
            score_a = result.filter(pl.col("ticker") == "INS_A")["score"][0]
            score_b = result.filter(pl.col("ticker") == "INS_B")["score"][0]
            assert score_a > score_b

    def test_no_extra_data_runs_without_error(self):
        """Passing no extra_data should work fine (no insider bonus)."""
        fund = two_period_fund("NOEXT")
        prices = make_prices("NOEXT")
        s = TenBaggerStrategy(top_n=10)
        result = s.screen(prices, fund, AS_OF, extra_data=None)
        assert not result.is_empty()
        assert "NOEXT" in result["ticker"].to_list()

    def test_single_period_fund_excluded(self):
        """Ticker with only one period cannot compute revenue growth → excluded."""
        fund = pl.DataFrame([make_fund("SINGLE")])  # only one period
        prices = make_prices("SINGLE")
        s = TenBaggerStrategy(top_n=10)
        result = s.screen(prices, fund, AS_OF)
        assert "SINGLE" not in result["ticker"].to_list()
