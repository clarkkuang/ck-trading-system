"""Shared helper to build extra_data dict for strategies."""

import polars as pl

from ck_trading.storage.parquet_store import ParquetStore


def build_extra_data(
    store: ParquetStore, tickers: list[str] | None = None
) -> dict[str, pl.DataFrame]:
    """Build extra_data dict from alternative and macro data sources.

    Returns a dict suitable for passing as ``extra_data`` to strategy.screen().
    """
    extra: dict[str, pl.DataFrame] = {}

    # Earnings
    df = store.load_alternative("earnings", "combined", tickers=tickers)
    if not df.is_empty():
        extra["earnings"] = df

    # Insider trades
    df = store.load_alternative("sec", "form4", tickers=tickers)
    if not df.is_empty():
        extra["insider_trades"] = df

    # Short interest
    df = store.load_alternative("finra", "daily", tickers=tickers)
    if not df.is_empty():
        extra["short_interest"] = df

    # Macro (no ticker filter)
    try:
        macro = store.load_macro()
        if not macro.is_empty():
            extra["macro"] = macro
    except Exception:
        pass

    return extra
