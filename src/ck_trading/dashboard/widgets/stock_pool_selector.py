"""Reusable stock pool selector widget for Streamlit pages."""

import streamlit as st

from ck_trading.storage.metadata_store import MetadataStore


POOL_OPTIONS = ["Full Universe", "Watchlist", "Portfolio Holdings", "Custom"]


def stock_pool_selector(meta: MetadataStore, key_prefix: str = "") -> list[str]:
    """Render a stock pool selector widget and return selected ticker strings.

    Includes a quick-add form so users can add tickers to the universe/watchlist
    without leaving the current page.

    Args:
        meta: MetadataStore instance for querying universe/watchlist/positions.
        key_prefix: Unique prefix for Streamlit widget keys (use when
            placing multiple selectors on the same page).

    Returns:
        List of ticker strings based on user selection.
    """
    pool = st.selectbox(
        "Stock Pool",
        POOL_OPTIONS,
        key=f"{key_prefix}stock_pool",
    )

    if pool == "Full Universe":
        universe = meta.get_universe()
        tickers = [u["ticker"] for u in universe]

    elif pool == "Watchlist":
        watchlist = meta.get_watchlist()
        tickers = [w["ticker"] for w in watchlist]
        if not tickers:
            st.warning("Watchlist is empty. Add tickers below or on the Watchlist page.")

    elif pool == "Portfolio Holdings":
        positions = meta.get_open_positions()
        tickers = list({p["ticker"] for p in positions})
        if not tickers:
            st.warning("No open positions. Import or add positions on the Portfolio page.")

    else:  # Custom
        universe = meta.get_universe()
        all_tickers = sorted([u["ticker"] for u in universe])
        tickers = st.multiselect(
            "Select tickers",
            all_tickers,
            key=f"{key_prefix}custom_tickers",
        )

    # Cache in session state for reliable access during button-click reruns
    st.session_state[f"{key_prefix}selected_tickers"] = tickers

    st.info(f"Selected **{len(tickers)}** tickers from **{pool}**")
    with st.expander(f"View {len(tickers)} tickers", expanded=False):
        st.text(", ".join(sorted(tickers)) if tickers else "(none)")

    # --- Quick add ticker ---
    with st.expander("Add ticker to pool", expanded=False):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            new_ticker = st.text_input(
                "Ticker", placeholder="AAPL", key=f"{key_prefix}add_ticker",
                label_visibility="collapsed",
            )
        with c2:
            new_market = st.selectbox(
                "Market", ["us", "hk"], key=f"{key_prefix}add_market",
                label_visibility="collapsed",
            )
        with c3:
            if st.button("Add", key=f"{key_prefix}add_btn", type="primary"):
                if new_ticker:
                    ticker_upper = new_ticker.strip().upper()
                    # Add to universe + watchlist so it appears in all pools
                    meta.add_to_universe(ticker_upper, new_market)
                    meta.add_to_watchlist(ticker_upper, new_market, source="manual")
                    st.toast(f"Added {ticker_upper} ({new_market.upper()})")
                    st.rerun()
                else:
                    st.warning("Enter a ticker first.")

    return tickers
