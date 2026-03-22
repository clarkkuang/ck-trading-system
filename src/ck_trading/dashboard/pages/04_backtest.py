"""Backtesting page."""

from datetime import date

import streamlit as st

st.set_page_config(page_title="Backtest", page_icon="📊", layout="wide")
st.title("Strategy Backtesting")

col1, col2 = st.columns(2)

with col1:
    strategy_name = st.selectbox(
        "Strategy",
        ["Graham Defensive", "Piotroski F-Score", "Magic Formula", "Composite Value"],
    )
    start_date = st.date_input("Start Date", date(2015, 1, 1))

with col2:
    max_positions = st.number_input("Max Positions", 5, 50, 20)
    end_date = st.date_input("End Date", date.today())

rebalance_freq = st.selectbox("Rebalance Frequency", ["quarterly", "monthly", "annual"])
initial_capital = st.number_input("Initial Capital ($)", 10000, 10000000, 1000000, step=100000)

if st.button("Run Backtest", type="primary"):
    with st.spinner("Running backtest... This may take a minute."):
        try:
            from ck_trading.backtesting.engine import BacktestEngine
            from ck_trading.models.backtest import BacktestConfig
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
            prices = store.load_prices("us")
            fundamentals = store.load_fundamentals("us")

            if prices.is_empty():
                st.error("No price data. Run backfill_data.py first.")
            else:
                tickers = prices["ticker"].unique().to_list()
                tickers = [t for t in tickers if t not in {"SPY", "QQQ", "IWM", "VTV"}]

                strategy = strategies[strategy_name]()
                config = BacktestConfig(
                    strategy_name=strategy.name,
                    universe=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=float(initial_capital),
                    max_positions=max_positions,
                    rebalance_freq=rebalance_freq,
                )

                engine = BacktestEngine(strategy, config, prices, fundamentals)
                result = engine.run()

                st.subheader("Results")
                st.text(result.summary())

                if result.metrics:
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("CAGR", f"{result.metrics.get('cagr', 0):.2%}")
                    mcol2.metric("Sharpe", f"{result.metrics.get('sharpe_ratio', 0):.2f}")
                    mcol3.metric("Max Drawdown", f"{result.metrics.get('max_drawdown', 0):.2%}")
                    mcol4.metric("Final Value", f"${result.metrics.get('final_value', 0):,.0f}")

                if not result.trades.is_empty():
                    st.subheader("Trade Log")
                    st.dataframe(result.trades.to_pandas(), use_container_width=True)

        except Exception as e:
            st.error(f"Error running backtest: {e}")
