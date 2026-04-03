"""Tests for market data models."""

from datetime import date

from ck_trading.models.market_data import OHLCV, FinancialStatement, Market, StockInfo


def test_market_enum_values():
    assert Market.US == "US"
    assert Market.HK == "HK"


def test_ohlcv_required_fields():
    bar = OHLCV(
        ticker="AAPL",
        date=date(2024, 1, 2),
        open=185.0,
        high=186.0,
        low=184.0,
        close=185.5,
        volume=50_000_000,
    )
    assert bar.ticker == "AAPL"
    assert bar.close == 185.5
    assert bar.volume == 50_000_000


def test_ohlcv_defaults():
    bar = OHLCV(
        ticker="AAPL",
        date=date(2024, 1, 2),
        open=185.0,
        high=186.0,
        low=184.0,
        close=185.5,
        volume=50_000_000,
    )
    assert bar.adj_close is None
    assert bar.market == Market.US


def test_ohlcv_hk_market():
    bar = OHLCV(
        ticker="0700.HK",
        date=date(2024, 1, 2),
        open=300.0,
        high=305.0,
        low=298.0,
        close=302.0,
        volume=10_000_000,
        market=Market.HK,
    )
    assert bar.market == Market.HK


def test_financial_statement_optional_fields():
    stmt = FinancialStatement(
        ticker="AAPL",
        market=Market.US,
        period_end=date(2024, 9, 30),
        report_date=date(2024, 10, 31),
    )
    assert stmt.revenue is None
    assert stmt.pe_ratio is None
    assert stmt.free_cash_flow is None


def test_financial_statement_with_data():
    stmt = FinancialStatement(
        ticker="AAPL",
        market=Market.US,
        period_end=date(2024, 9, 30),
        report_date=date(2024, 10, 31),
        revenue=394e9,
        net_income=97e9,
        pe_ratio=30.5,
        roe=1.56,
    )
    assert stmt.revenue == 394e9
    assert stmt.pe_ratio == 30.5


def test_stock_info_defaults():
    info = StockInfo(ticker="AAPL", name="Apple Inc.", market=Market.US)
    assert info.sector == ""
    assert info.industry == ""
    assert info.market_cap is None
    assert info.currency == "USD"


def test_stock_info_full():
    info = StockInfo(
        ticker="AAPL",
        name="Apple Inc.",
        market=Market.US,
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3e12,
        currency="USD",
    )
    assert info.sector == "Technology"
    assert info.market_cap == 3e12
