"""Portfolio page - view holdings, P&L, and allocation."""

import streamlit as st

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
st.title("Portfolio")

try:
    from ck_trading.portfolio.tracker import PortfolioTracker
    from ck_trading.storage.metadata_store import MetadataStore

    meta = MetadataStore()
    tracker = PortfolioTracker(meta)
    positions = tracker.get_positions()

    if not positions:
        st.info("No open positions. Add trades to see your portfolio here.")
    else:
        import polars as pl

        df = pl.DataFrame(positions)
        st.dataframe(df.to_pandas(), use_container_width=True)

        # Summary metrics
        total_cost = sum(p["shares"] * p["avg_cost"] for p in positions)
        st.metric("Total Cost Basis", f"${total_cost:,.2f}")

    meta.close()
except Exception as e:
    st.error(f"Error loading portfolio: {e}")
