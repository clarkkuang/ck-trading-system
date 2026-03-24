"""Reusable stock pool selector widget for Streamlit pages."""

import streamlit as st

from ck_trading.storage.metadata_store import MetadataStore


POOL_OPTIONS = ["Full Universe", "Watchlist", "Portfolio Holdings", "Custom"]


def stock_pool_selector(meta: MetadataStore, key_prefix: str = "") -> list[str]:
    """Render a stock pool selector widget and return selected ticker strings.

    The pool selection sets the *default* tickers. A multiselect then lets the
    user add or remove individual tickers before running the action.

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

    # Get default tickers from the selected pool
    if pool == "Full Universe":
        universe = meta.get_universe()
        default_tickers = sorted([u["ticker"] for u in universe])

    elif pool == "Watchlist":
        watchlist = meta.get_watchlist()
        default_tickers = sorted([w["ticker"] for w in watchlist])
        if not default_tickers:
            st.warning("Watchlist is empty. Add tickers on the Watchlist page.")

    elif pool == "Portfolio Holdings":
        positions = meta.get_open_positions()
        default_tickers = sorted({p["ticker"] for p in positions})
        if not default_tickers:
            st.warning("No open positions. Import or add on the Portfolio page.")

    else:  # Custom — start with nothing, user picks manually
        default_tickers = []

    # All known tickers as options (union of universe so user can add any)
    universe = meta.get_universe()
    all_known = sorted({u["ticker"] for u in universe})

    # Ensure defaults are in options (portfolio tickers might not be in universe)
    all_options = sorted(set(all_known) | set(default_tickers))

    # Editable multiselect — pre-filled from pool, user can add/remove
    tickers = st.multiselect(
        "Tickers (add or remove as needed)",
        options=all_options,
        default=default_tickers,
        key=f"{key_prefix}editable_tickers",
    )

    # Cache in session state for reliable access during button-click reruns
    st.session_state[f"{key_prefix}selected_tickers"] = tickers

    st.info(f"**{len(tickers)}** tickers selected (from {pool}, editable)")
    return tickers
