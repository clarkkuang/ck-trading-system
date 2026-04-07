"""Stock screener page."""

from datetime import date

import polars as pl
import streamlit as st

st.set_page_config(page_title="Screener", page_icon="🔍", layout="wide")
st.title("Value Stock Screener")

from ck_trading.dashboard.widgets.stock_pool_selector import stock_pool_selector
from ck_trading.storage.metadata_store import MetadataStore
from ck_trading.strategies.registry import get_all_strategies

strategies = get_all_strategies()

strategy_name = st.selectbox(
    "Strategy",
    list(strategies.keys()),
)

strategy_instance = strategies[strategy_name]()
st.caption(f"\u2139\ufe0f {strategy_instance.description}")

market = st.selectbox("Market", ["us", "hk"])

# --- Stock pool selector ---
meta = MetadataStore()
selected_tickers = stock_pool_selector(meta, key_prefix="screener_")
meta.close()

if st.button("Run Screen", type="primary"):
    # Re-read from session state to survive the rerun
    tickers = st.session_state.get("screener_selected_tickers", selected_tickers)

    with st.spinner("Running screen..."):
        try:
            from ck_trading.storage.parquet_store import ParquetStore
            from ck_trading.dashboard.data_cache import load_prices, load_fundamentals

            store = ParquetStore()
            prices = load_prices(market)
            fundamentals = load_fundamentals(market)

            # Filter to selected stock pool
            if tickers:
                prices = prices.filter(pl.col("ticker").is_in(tickers))
                fundamentals = fundamentals.filter(pl.col("ticker").is_in(tickers))

            # Debug: show what actually goes into the strategy
            with st.expander("Debug: Stock Pool", expanded=False):
                actual_price_tickers = (
                    prices["ticker"].unique().sort().to_list()
                    if not prices.is_empty() else []
                )
                actual_fund_tickers = (
                    fundamentals["ticker"].unique().sort().to_list()
                    if not fundamentals.is_empty() else []
                )
                st.write(f"Selected tickers: {len(tickers)}")
                st.write(f"Tickers in prices: {len(actual_price_tickers)} — {actual_price_tickers}")
                st.write(f"Tickers in fundamentals: {len(actual_fund_tickers)} — {actual_fund_tickers}")

            if fundamentals.is_empty():
                st.warning("No fundamental data available. Run backfill_data.py first.")
            else:
                from ck_trading.dashboard.widgets.extra_data import build_extra_data

                extra_data = build_extra_data(store, tickers)
                strategy = strategies[strategy_name]()
                result = strategy.screen(prices, fundamentals, date.today(), extra_data=extra_data)

                # Post-filter: safety net to ensure only selected tickers in results
                if tickers and not result.is_empty():
                    result = result.filter(pl.col("ticker").is_in(tickers))

                if result.is_empty():
                    st.info("No stocks passed the screen criteria.")
                else:
                    st.success(f"Found {result.height} stocks")
                    st.dataframe(result.to_pandas(), use_container_width=True)

        except Exception as e:
            st.error(f"Error running screen: {e}")
