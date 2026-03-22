"""Shared test fixtures."""

from datetime import date
from pathlib import Path
import random

import polars as pl
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    for subdir in ["prices/us/daily", "prices/hk/daily", "fundamentals/us", "fundamentals/hk", "macro"]:
        (tmp_path / subdir).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_prices() -> pl.DataFrame:
    """Sample price data for testing."""
    dates = pl.date_range(date(2020, 1, 2), date(2020, 12, 31), eager=True)
    random.seed(42)

    records = []
    for ticker in ["AAPL", "MSFT", "SPY"]:
        base = {"AAPL": 300, "MSFT": 160, "SPY": 320}[ticker]
        for i, d in enumerate(dates):
            price = base + random.gauss(0, 2) + i * 0.1
            records.append({
                "ticker": ticker,
                "date": d,
                "open": price - 1,
                "high": price + 2,
                "low": price - 2,
                "close": price,
                "volume": random.randint(1000000, 10000000),
                "adj_close": price,
            })

    return pl.DataFrame(records)


@pytest.fixture
def sample_fundamentals() -> pl.DataFrame:
    """Sample fundamental data for testing."""
    return pl.DataFrame([
        {
            "ticker": "AAPL",
            "market": "US",
            "period_end": date(2020, 9, 26),
            "report_date": date(2020, 10, 30),
            "pe_ratio": 12.5,
            "pb_ratio": 1.2,
            "current_ratio": 2.5,
            "market_cap": 2e12,
            "net_income": 57e9,
            "revenue": 274e9,
            "ebit": 66e9,
            "enterprise_value": 2.1e12,
            "total_assets": 323e9,
            "current_liabilities": 105e9,
            "cash_and_equivalents": 38e9,
            "roe": 0.73,
            "roa": 0.17,
            "operating_margin": 0.24,
            "gross_margin": 0.38,
            "debt_to_equity": 1.73,
            "total_equity": 65e9,
            "book_value_per_share": 3.85,
            "operating_cash_flow": 80e9,
            "free_cash_flow": 73e9,
        },
        {
            "ticker": "MSFT",
            "market": "US",
            "period_end": date(2020, 6, 30),
            "report_date": date(2020, 7, 22),
            "pe_ratio": 10.0,
            "pb_ratio": 1.0,
            "current_ratio": 2.8,
            "market_cap": 1.6e12,
            "net_income": 44e9,
            "revenue": 143e9,
            "ebit": 53e9,
            "enterprise_value": 1.65e12,
            "total_assets": 301e9,
            "current_liabilities": 72e9,
            "cash_and_equivalents": 14e9,
            "roe": 0.40,
            "roa": 0.15,
            "operating_margin": 0.37,
            "gross_margin": 0.68,
            "debt_to_equity": 0.62,
            "total_equity": 118e9,
            "book_value_per_share": 15.45,
            "operating_cash_flow": 60e9,
            "free_cash_flow": 45e9,
        },
    ])
