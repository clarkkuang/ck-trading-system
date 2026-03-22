"""Watchlist page."""

import streamlit as st

st.set_page_config(page_title="Watchlist", page_icon="👀", layout="wide")
st.title("Watchlist")

try:
    from ck_trading.storage.metadata_store import MetadataStore

    meta = MetadataStore()

    # Add to watchlist
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_ticker = st.text_input("Add Ticker", placeholder="AAPL")
    with col2:
        new_market = st.selectbox("Market", ["US", "HK"])
    with col3:
        st.write("")  # Spacer
        st.write("")
        if st.button("Add", type="primary"):
            if new_ticker:
                meta.add_to_watchlist(new_ticker.upper(), new_market)
                st.success(f"Added {new_ticker.upper()} to watchlist")
                st.rerun()

    # Display watchlist
    watchlist = meta.get_watchlist()
    if not watchlist:
        st.info("Watchlist is empty. Add tickers above.")
    else:
        import polars as pl

        df = pl.DataFrame(watchlist)
        st.dataframe(df.to_pandas(), use_container_width=True)

    meta.close()
except Exception as e:
    st.error(f"Error: {e}")
