"""Cached data loading for the Streamlit dashboard.

Uses @st.cache_data to avoid re-reading thousands of Parquet files on every
page load / widget interaction.  Cache is keyed by market and invalidated
after ``ttl`` seconds (default 10 minutes).
"""

from __future__ import annotations

import streamlit as st
import polars as pl

from ck_trading.storage.parquet_store import ParquetStore

_TTL = 600  # seconds


@st.cache_data(ttl=_TTL, show_spinner=False)
def load_prices(market: str) -> pl.DataFrame:
    return ParquetStore().load_prices(market)


@st.cache_data(ttl=_TTL, show_spinner=False)
def load_prices_for_tickers(market: str, tickers: tuple[str, ...]) -> pl.DataFrame:
    """Load prices for specific tickers only — much faster than loading all."""
    return ParquetStore().load_prices(market, tickers=list(tickers))


@st.cache_data(ttl=_TTL, show_spinner=False)
def load_fundamentals(market: str) -> pl.DataFrame:
    return ParquetStore().load_fundamentals(market)


@st.cache_data(ttl=_TTL, show_spinner=False)
def load_macro() -> pl.DataFrame:
    return ParquetStore().load_macro()


@st.cache_data(ttl=_TTL, show_spinner=False)
def load_monitoring(name: str) -> pl.DataFrame:
    from ck_trading.monitoring.store import MonitoringStore

    return MonitoringStore().load(name)


@st.cache_data(ttl=_TTL, show_spinner=False)
def load_monitoring_alerts() -> dict:
    from ck_trading.monitoring.store import MonitoringStore

    return MonitoringStore().load_alerts()


@st.cache_data(ttl=_TTL, show_spinner=False)
def price_count(market: str) -> int:
    """Return row count without materialising the full DataFrame."""
    from pathlib import Path
    from ck_trading.config import settings

    data_dir = settings.data_dir / "prices" / market.lower() / "daily"
    if not data_dir.exists():
        return 0
    total = 0
    for f in data_dir.glob("*.parquet"):
        total += pl.scan_parquet(f).select(pl.len()).collect().item()
    return total


@st.cache_data(ttl=_TTL, show_spinner=False)
def fundamentals_count(market: str) -> int:
    from pathlib import Path
    from ck_trading.config import settings

    data_dir = settings.data_dir / "fundamentals" / market.lower()
    if not data_dir.exists():
        return 0
    total = 0
    for f in data_dir.glob("*_financials.parquet"):
        total += pl.scan_parquet(f).select(pl.len()).collect().item()
    return total
