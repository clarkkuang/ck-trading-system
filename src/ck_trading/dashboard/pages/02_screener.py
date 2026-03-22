"""Stock screener page."""

from datetime import date

import streamlit as st

st.set_page_config(page_title="Screener", page_icon="🔍", layout="wide")
st.title("Value Stock Screener")

strategy_name = st.selectbox(
    "Strategy",
    ["Graham Defensive", "Piotroski F-Score", "Magic Formula", "Composite Value"],
)

market = st.selectbox("Market", ["us", "hk"])

if st.button("Run Screen", type="primary"):
    with st.spinner("Running screen..."):
        try:
            from ck_trading.storage.parquet_store import ParquetStore
            from ck_trading.strategies.composite_value import CompositeValueStrategy
            from ck_trading.strategies.graham_defensive import GrahamDefensiveStrategy
            from ck_trading.strategies.magic_formula import MagicFormulaStrategy
            from ck_trading.strategies.piotroski_f_score import PiotroskiFScoreStrategy

            strategies = {
                "Graham Defensive": GrahamDefensiveStrategy,
                "Piotroski F-Score": PiotroskiFScoreStrategy,
                "Magic Formula": MagicFormulaStrategy,
                "Composite Value": CompositeValueStrategy,
            }

            store = ParquetStore()
            prices = store.load_prices(market)
            fundamentals = store.load_fundamentals(market)

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
