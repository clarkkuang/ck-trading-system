"""Main Streamlit dashboard application."""

import streamlit as st

st.set_page_config(
    page_title="CK Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("CK Trading System")
st.markdown("Quantitative Value Investing Dashboard")

st.markdown("""
### Quick Links
- **Portfolio**: View current holdings and P&L
- **Screener**: Run value investing screens
- **Signals**: View and manage trading signals
- **Backtest**: Test strategies on historical data
- **Watchlist**: Monitor stocks of interest

Use the sidebar to navigate between pages.
""")

# System status
st.subheader("System Status")

try:
    from ck_trading.storage.metadata_store import MetadataStore
    from ck_trading.dashboard.data_cache import price_count, fundamentals_count

    meta = MetadataStore()

    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            st.metric("US Price Records", f"{price_count('us'):,}")
        except Exception:
            st.metric("US Price Records", "N/A")

    with col2:
        try:
            st.metric("US Fundamental Records", f"{fundamentals_count('us'):,}")
        except Exception:
            st.metric("US Fundamental Records", "N/A")

    with col3:
        try:
            universe = meta.get_universe()
            st.metric("Universe Size", len(universe))
        except Exception:
            st.metric("Universe Size", "N/A")

    meta.close()
except Exception:
    st.warning("Could not load system status.")
