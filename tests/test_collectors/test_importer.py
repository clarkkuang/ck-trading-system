"""Tests for portfolio CSV importer."""

from datetime import date
from pathlib import Path

import pytest

from ck_trading.models.market_data import Market
from ck_trading.portfolio.importer import import_chase_csv, import_generic_csv


@pytest.fixture
def chase_csv(tmp_path) -> Path:
    """Sample Chase CSV export."""
    csv_content = """Account Number,Account Name,Symbol,Description,Quantity,Price,Market Value,Cost Basis,Gain/Loss
XXX-1234,INVESTMENT,"AAPL","APPLE INC",100,$150.00,"$15,000.00","$12,000.00","$3,000.00"
XXX-1234,INVESTMENT,"MSFT","MICROSOFT CORP",50,$300.00,"$15,000.00","$10,000.00","$5,000.00"
XXX-1234,INVESTMENT,"CASH","CASH & MONEY MARKET",0,$1.00,"$5,000.00","$5,000.00","$0.00"
"""
    path = tmp_path / "chase_export.csv"
    path.write_text(csv_content)
    return path


@pytest.fixture
def generic_csv(tmp_path) -> Path:
    """Simple generic CSV."""
    csv_content = """ticker,shares,avg_cost,date
AAPL,100,120.00,2023-06-15
MSFT,50,200.00,2023-07-20
GOOG,25,100.00,2023-08-01
"""
    path = tmp_path / "portfolio.csv"
    path.write_text(csv_content)
    return path


@pytest.fixture
def generic_csv_alt_columns(tmp_path) -> Path:
    """Generic CSV with alternative column names."""
    csv_content = """Symbol,Quantity,Price
NVDA,30,$450.00
TSLA,20,$250.00
"""
    path = tmp_path / "alt_portfolio.csv"
    path.write_text(csv_content)
    return path


class TestImportChaseCSV:
    def test_basic_import(self, chase_csv):
        positions = import_chase_csv(chase_csv)

        assert len(positions) == 2  # CASH should be excluded
        tickers = {p.ticker for p in positions}
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "CASH" not in tickers

    def test_shares_parsed(self, chase_csv):
        positions = import_chase_csv(chase_csv)
        aapl = next(p for p in positions if p.ticker == "AAPL")
        assert aapl.shares == 100

    def test_cost_basis_computed(self, chase_csv):
        positions = import_chase_csv(chase_csv)
        aapl = next(p for p in positions if p.ticker == "AAPL")
        # Total cost basis $12,000 / 100 shares = $120 avg cost
        assert aapl.avg_cost == pytest.approx(120.0)

    def test_market_detected(self, chase_csv):
        positions = import_chase_csv(chase_csv)
        for p in positions:
            assert p.market == Market.US

    def test_missing_columns_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("name,value\nfoo,bar\n")
        with pytest.raises(ValueError, match="Cannot find required columns"):
            import_chase_csv(csv)


class TestImportGenericCSV:
    def test_basic_import(self, generic_csv):
        positions = import_generic_csv(generic_csv)

        assert len(positions) == 3
        tickers = {p.ticker for p in positions}
        assert tickers == {"AAPL", "MSFT", "GOOG"}

    def test_avg_cost_parsed(self, generic_csv):
        positions = import_generic_csv(generic_csv)
        aapl = next(p for p in positions if p.ticker == "AAPL")
        assert aapl.avg_cost == pytest.approx(120.0)

    def test_date_parsed(self, generic_csv):
        positions = import_generic_csv(generic_csv)
        aapl = next(p for p in positions if p.ticker == "AAPL")
        assert aapl.date_acquired == date(2023, 6, 15)

    def test_alt_column_names(self, generic_csv_alt_columns):
        positions = import_generic_csv(generic_csv_alt_columns)

        assert len(positions) == 2
        nvda = next(p for p in positions if p.ticker == "NVDA")
        assert nvda.shares == 30

    def test_hk_market_detection(self, tmp_path):
        csv = tmp_path / "hk.csv"
        csv.write_text("ticker,shares,avg_cost\n0005.HK,1000,50.00\n")
        positions = import_generic_csv(csv)

        assert len(positions) == 1
        assert positions[0].market == Market.HK
        assert positions[0].ticker == "0005.HK"

    def test_zero_shares_skipped(self, tmp_path):
        csv = tmp_path / "zero.csv"
        csv.write_text("ticker,shares,avg_cost\nAAPL,0,100\nMSFT,10,200\n")
        positions = import_generic_csv(csv)
        assert len(positions) == 1
        assert positions[0].ticker == "MSFT"

    def test_negative_shares_skipped(self, tmp_path):
        csv = tmp_path / "neg.csv"
        csv.write_text("ticker,shares,avg_cost\nAAPL,-5,100\n")
        positions = import_generic_csv(csv)
        assert len(positions) == 0

    def test_missing_columns_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("name,value\nfoo,bar\n")
        with pytest.raises(ValueError, match="Cannot find required columns"):
            import_generic_csv(csv)

    def test_empty_csv(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("ticker,shares,avg_cost\n")
        positions = import_generic_csv(csv)
        assert positions == []

    def test_cost_basis_column_not_treated_as_avg_cost(self, tmp_path):
        """Regression: 'cost_basis' substring-matched 'cost' candidate,
        causing total cost to be stored as per-share cost."""
        csv = tmp_path / "total_cost.csv"
        csv.write_text(
            "ticker,shares,cost_basis\n"
            "AAPL,100,15000\n"  # $15,000 total → should be $150/share
            "MSFT,50,10000\n"   # $10,000 total → should be $200/share
        )
        positions = import_generic_csv(csv)

        assert len(positions) == 2
        aapl = next(p for p in positions if p.ticker == "AAPL")
        msft = next(p for p in positions if p.ticker == "MSFT")

        # Must divide by shares, NOT use raw total as avg_cost
        assert aapl.avg_cost == pytest.approx(150.0), (
            f"Expected $150/share, got ${aapl.avg_cost} — "
            "cost_basis was likely treated as per-share cost"
        )
        assert msft.avg_cost == pytest.approx(200.0)

    def test_total_cost_column_divided_by_shares(self, tmp_path):
        """CSV with 'total_cost' column should derive avg_cost correctly."""
        csv = tmp_path / "total.csv"
        csv.write_text("ticker,shares,total_cost\nNVDA,20,18000\n")
        positions = import_generic_csv(csv)

        assert len(positions) == 1
        assert positions[0].avg_cost == pytest.approx(900.0)

    def test_avg_cost_column_preferred_over_total(self, tmp_path):
        """When both avg_cost and cost_basis exist, use avg_cost."""
        csv = tmp_path / "both.csv"
        csv.write_text(
            "ticker,shares,avg_cost,cost_basis\n"
            "AAPL,100,150.00,15000\n"
        )
        positions = import_generic_csv(csv)

        assert len(positions) == 1
        assert positions[0].avg_cost == pytest.approx(150.0)
