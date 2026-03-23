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

market = st.selectbox("Market", ["us", "hk"])

# --- Stock pool selector ---
meta = MetadataStore()
selected_tickers = stock_pool_selector(meta, key_prefix="screener_")
meta.close()

if st.button("Run Screen", type="primary"):
    with st.spinner("Running screen..."):
        try:
            from ck_trading.storage.parquet_store import ParquetStore

            store = ParquetStore()
            prices = store.load_prices(market)
            fundamentals = store.load_fundamentals(market)

            # Filter to selected stock pool
            if selected_tickers:
                prices = prices.filter(pl.col("ticker").is_in(selected_tickers))
                fundamentals = fundamentals.filter(pl.col("ticker").is_in(selected_tickers))

            if fundamentals.is_empty():
                st.warning("No fundamental data available. Run backfill_data.py first.")
            else:
                strategy = strategies[strategy_name]()
                result = strategy.screen(prices, fundamentals, date.today())

                if result.is_empty():
                    st.info("No stocks passed the screen criteria.")
                else:
                    st.success(f"Found {result.height} stocks")
                    st.dataframe(result.to_pandas(), use_container_width=True)

        except Exception as e:
            st.error(f"Error running screen: {e}")
