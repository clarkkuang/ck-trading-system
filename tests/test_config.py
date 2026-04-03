"""Tests for application configuration."""

from pathlib import Path

from ck_trading.config import Settings


def test_defaults():
    s = Settings(data_dir=Path("/tmp/test_data"), _env_file=None)
    assert s.data_dir == Path("/tmp/test_data")
    assert s.fmp_api_key == ""
    assert s.fred_api_key == ""
    assert s.notification_urls == ""
    assert s.default_us_benchmark == "SPY"
    assert s.default_hk_benchmark == "^HSI"
    assert s.default_initial_capital == 1_000_000
    assert s.default_transaction_cost_bps == 10.0
    assert s.default_max_positions == 20


def test_derived_paths():
    s = Settings(data_dir=Path("/tmp/test_data"), _env_file=None)
    assert s.prices_dir == Path("/tmp/test_data/prices")
    assert s.fundamentals_dir == Path("/tmp/test_data/fundamentals")
    assert s.macro_dir == Path("/tmp/test_data/macro")
    assert s.duckdb_path == Path("/tmp/test_data/analytics.duckdb")
    assert s.metadata_db_path == Path("/tmp/test_data/metadata.db")


def test_custom_values():
    s = Settings(
        data_dir=Path("/data"),
        fmp_api_key="test_key",
        default_initial_capital=500_000,
        default_max_positions=10,
        _env_file=None,
    )
    assert s.fmp_api_key == "test_key"
    assert s.default_initial_capital == 500_000
    assert s.default_max_positions == 10
