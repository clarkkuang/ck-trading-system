"""Market data models."""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel


class Market(StrEnum):
    US = "US"
    HK = "HK"


class OHLCV(BaseModel):
    """Single OHLCV bar."""

    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float | None = None
    market: Market = Market.US


class FinancialStatement(BaseModel):
    """Standardized financial statement data."""

    ticker: str
    market: Market
    period_end: date
    report_date: date  # Point-in-time: when data became available

    # Income statement
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    ebit: float | None = None
    ebitda: float | None = None
    eps: float | None = None

    # Balance sheet
    total_assets: float | None = None
    total_liabilities: float | None = None
    total_equity: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    cash_and_equivalents: float | None = None
    total_debt: float | None = None
    book_value_per_share: float | None = None

    # Cash flow
    operating_cash_flow: float | None = None
    capex: float | None = None
    free_cash_flow: float | None = None
    dividends_paid: float | None = None

    # Ratios (derived or from data source)
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    roe: float | None = None
    roa: float | None = None
    current_ratio: float | None = None
    debt_to_equity: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    asset_turnover: float | None = None
    dividend_yield: float | None = None
    enterprise_value: float | None = None


class StockInfo(BaseModel):
    """Basic stock information."""

    ticker: str
    name: str
    market: Market
    sector: str = ""
    industry: str = ""
    market_cap: float | None = None
    currency: str = "USD"
