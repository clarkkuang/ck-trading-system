"""Reusable stock pool selector widget for Streamlit pages."""

import streamlit as st

from ck_trading.storage.metadata_store import MetadataStore


POOL_OPTIONS = ["Full Universe", "Watchlist", "Portfolio Holdings", "Custom"]


def stock_pool_selector(meta: MetadataStore, key_prefix: str = "") -> list[str]:
    """Render a stock pool selector widget and return selected ticker strings.

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
            st.warning("Watchlist is empty. Add tickers on the Watchlist page.")

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

    st.info(f"Selected **{len(tickers)}** tickers from **{pool}**")
    return tickers
