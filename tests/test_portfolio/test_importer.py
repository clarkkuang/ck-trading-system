"""Tests for portfolio CSV importers."""

from datetime import date

import polars as pl
import pytest

from ck_trading.models.market_data import Market
from ck_trading.portfolio.importer import (
    _find_column,
    import_chase_csv,
    import_generic_csv,
)


def test_find_column_exact_match():
    df = pl.DataFrame({"ticker": ["AAPL"], "shares": [100]})
    assert _find_column(df, ["ticker"]) == "ticker"


def test_find_column_substring_match():
    df = pl.DataFrame({"my_ticker_col": ["AAPL"]})
    assert _find_column(df, ["ticker"]) == "my_ticker_col"


def test_find_column_exact_only():
    df = pl.DataFrame({"my_ticker_col": ["AAPL"]})
    assert _find_column(df, ["ticker"], exact_only=True) is None


def test_find_column_not_found():
    df = pl.DataFrame({"foo": [1]})
    assert _find_column(df, ["bar"]) is None


def test_import_generic_csv(tmp_path):
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text("ticker,shares,avg_cost,date\nAAPL,100,150.00,2024-01-15\nMSFT,50,370.00,2024-02-01\n")
    positions = import_generic_csv(csv_path)
    assert len(positions) == 2
    assert positions[0].ticker == "AAPL"
    assert positions[0].shares == 100
    assert positions[0].avg_cost == 150.0
    assert positions[0].date_acquired == date(2024, 1, 15)
    assert positions[0].market == Market.US


def test_import_generic_csv_hk_ticker(tmp_path):
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text("ticker,shares,avg_cost\n0700.HK,200,350.00\n")
    positions = import_generic_csv(csv_path)
    assert len(positions) == 1
    assert positions[0].market == Market.HK


def test_import_generic_csv_skips_zero_shares(tmp_path):
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text("ticker,shares,avg_cost\nAAPL,0,150.00\nMSFT,50,370.00\n")
    positions = import_generic_csv(csv_path)
    assert len(positions) == 1
    assert positions[0].ticker == "MSFT"


def test_import_generic_csv_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="Cannot find required columns"):
        import_generic_csv(csv_path)


def test_import_generic_csv_total_cost_fallback(tmp_path):
    csv_path = tmp_path / "portfolio.csv"
    csv_path.write_text("ticker,shares,cost_basis\nAAPL,100,15000.00\n")
    positions = import_generic_csv(csv_path)
    assert len(positions) == 1
    assert positions[0].avg_cost == pytest.approx(150.0)


def test_import_chase_csv(tmp_path):
    csv_path = tmp_path / "chase.csv"
    csv_path.write_text(
        "Account Number,Symbol,Description,Quantity,Cost Basis Per Share\n"
        "123,AAPL,Apple Inc,100,150.00\n"
        "123,MSFT,Microsoft Corp,50,370.00\n"
    )
    positions = import_chase_csv(csv_path)
    assert len(positions) == 2
    assert positions[0].ticker == "AAPL"
    assert positions[0].shares == 100


def test_import_chase_csv_skips_cash_rows(tmp_path):
    csv_path = tmp_path / "chase.csv"
    csv_path.write_text(
        "Symbol,Quantity,Cost Basis Per Share\n"
        "AAPL,100,150.00\n"
        "CASH,0,0\n"
        "TOTAL,0,0\n"
    )
    positions = import_chase_csv(csv_path)
    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"


def test_import_chase_csv_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="Cannot find required columns"):
        import_chase_csv(csv_path)
