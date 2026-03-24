"""Reusable stock pool selector widget for Streamlit pages."""

import streamlit as st

from ck_trading.storage.metadata_store import MetadataStore


POOL_OPTIONS = ["Full Universe", "Watchlist", "Portfolio Holdings", "Custom"]


def _get_pool_tickers(meta: MetadataStore, pool: str) -> list[str]:
    """Return sorted default tickers for a given pool."""
    if pool == "Full Universe":
        return sorted([u["ticker"] for u in meta.get_universe()])
    elif pool == "Watchlist":
        return sorted([w["ticker"] for w in meta.get_watchlist()])
    elif pool == "Portfolio Holdings":
        return sorted({p["ticker"] for p in meta.get_open_positions()})
    return []  # Custom


def stock_pool_selector(meta: MetadataStore, key_prefix: str = "") -> list[str]:
    """Render a stock pool selector and editable ticker multiselect."""
    pool_key = f"{key_prefix}stock_pool"
    prev_pool_key = f"{key_prefix}prev_pool"
    edit_key = f"{key_prefix}editable_tickers"

    pool = st.selectbox("Stock Pool", POOL_OPTIONS, key=pool_key)

    default_tickers = _get_pool_tickers(meta, pool)

    # When pool changes, overwrite the multiselect value directly
    prev_pool = st.session_state.get(prev_pool_key)
    if prev_pool is not None and prev_pool != pool:
        st.session_state[edit_key] = default_tickers
    st.session_state[prev_pool_key] = pool

    # Warnings for empty pools
    if pool == "Watchlist" and not default_tickers:
        st.warning("Watchlist is empty. Add tickers on the Watchlist page.")
    elif pool == "Portfolio Holdings" and not default_tickers:
        st.warning("No open positions. Import or add on the Portfolio page.")

    # All known tickers as options
    all_known = sorted({u["ticker"] for u in meta.get_universe()})
    all_options = sorted(set(all_known) | set(default_tickers))

    # Editable multiselect
    tickers = st.multiselect(
        "Tickers (add or remove as needed)",
        options=all_options,
        default=default_tickers,
        key=edit_key,
    )

    # Cache for button-click reruns
    st.session_state[f"{key_prefix}selected_tickers"] = tickers

    st.info(f"**{len(tickers)}** tickers selected (from {pool}, editable)")
    return tickers
