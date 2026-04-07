"""Signals page - view recent trading signals."""

import streamlit as st

st.set_page_config(page_title="Signals", page_icon="🚦", layout="wide")
st.title("Trading Signals")

try:
    from ck_trading.storage.metadata_store import MetadataStore

    meta = MetadataStore()
    signals = meta.get_recent_signals(100)

    if not signals:
        st.info("No signals yet. Run the signal generator to create signals.")
    else:
        import polars as pl

        df = pl.DataFrame(signals)
        st.dataframe(df.to_pandas(), use_container_width=True)

    # Manual signal generation
    st.subheader("Generate New Signals")

    market = st.selectbox("Market", ["us", "hk"], key="signals_market")

    # --- Stock pool selector ---
    from ck_trading.dashboard.widgets.stock_pool_selector import stock_pool_selector

    selected_tickers = stock_pool_selector(meta, key_prefix="signals_")

    if st.button("Run All Strategies", type="primary"):
        # Re-read from session state to survive the rerun
        tickers = st.session_state.get("signals_selected_tickers", selected_tickers)

        with st.spinner("Generating signals..."):
            try:
                import polars as pl

                from ck_trading.signals.generator import SignalGenerator
                from ck_trading.signals.manager import SignalManager
                from ck_trading.storage.parquet_store import ParquetStore
                from ck_trading.strategies.registry import get_all_strategies
                from ck_trading.dashboard.data_cache import load_prices, load_fundamentals

                store = ParquetStore()
                prices = load_prices(market)
                fundamentals = load_fundamentals(market)

                # Filter to selected stock pool
                if tickers:
                    prices = prices.filter(pl.col("ticker").is_in(tickers))
                    fundamentals = fundamentals.filter(pl.col("ticker").is_in(tickers))

                # Debug
                with st.expander("Debug: Stock Pool", expanded=False):
                    actual_tickers = (
                        prices["ticker"].unique().sort().to_list()
                        if not prices.is_empty() else []
                    )
                    st.write(f"Selected tickers: {len(tickers)}")
                    st.write(f"Tickers in prices: {len(actual_tickers)} — {actual_tickers}")

                from ck_trading.dashboard.widgets.extra_data import build_extra_data

                extra_data = build_extra_data(store, tickers)

                all_strategies = get_all_strategies()
                generator = SignalGenerator([
                    cls() for cls in all_strategies.values()
                ])

                raw_signals = generator.generate(prices, fundamentals, extra_data=extra_data)

                # Post-filter: only keep signals for selected tickers
                if tickers:
                    raw_signals = [s for s in raw_signals if s.ticker in set(tickers)]

                manager = SignalManager(meta)
                processed = manager.process_signals(raw_signals)

                st.success(f"Generated {len(processed)} signals")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating signals: {e}")

    meta.close()
except Exception as e:
    st.error(f"Error loading signals: {e}")
