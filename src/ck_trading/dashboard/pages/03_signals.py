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
    if st.button("Run All Strategies", type="primary"):
        with st.spinner("Generating signals..."):
            try:
                from ck_trading.signals.generator import SignalGenerator
                from ck_trading.signals.manager import SignalManager
                from ck_trading.storage.parquet_store import ParquetStore
                from ck_trading.strategies.registry import get_all_strategies

                store = ParquetStore()
                prices = store.load_prices("us")
                fundamentals = store.load_fundamentals("us")

                all_strategies = get_all_strategies()
                generator = SignalGenerator([
                    cls() for cls in all_strategies.values()
                ])

                raw_signals = generator.generate(prices, fundamentals)
                manager = SignalManager(meta)
                processed = manager.process_signals(raw_signals)

                st.success(f"Generated {len(processed)} signals")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating signals: {e}")

    meta.close()
except Exception as e:
    st.error(f"Error loading signals: {e}")
