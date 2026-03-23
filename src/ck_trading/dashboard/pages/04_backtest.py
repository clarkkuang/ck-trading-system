"""Backtesting page."""

from datetime import date

import streamlit as st

st.set_page_config(page_title="Backtest", page_icon="📊", layout="wide")
st.title("Strategy Backtesting")

from ck_trading.strategies.registry import get_all_strategies

strategies = get_all_strategies()

col1, col2 = st.columns(2)

with col1:
    strategy_name = st.selectbox(
        "Strategy",
        list(strategies.keys()),
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

                # Debug info
                with st.expander("Debug Info", expanded=False):
                    st.write(f"Config: start={config.start_date}, end={config.end_date}, "
                             f"max_pos={config.max_positions}, freq={config.rebalance_freq}")
                    st.write(f"Prices: {prices.height} rows, date range: "
                             f"{prices['date'].min()} - {prices['date'].max()}")
                    st.write(f"Fundamentals: {fundamentals.height} rows")
                    st.write(f"Trades: {result.trades.height if not result.trades.is_empty() else 0}")
                    st.write(f"Returns: {result.returns.height if not result.returns.is_empty() else 0}")
                    st.write(f"Metrics keys: {list(result.metrics.keys())}")
                    st.write(f"Metrics values: { {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in result.metrics.items()} }")

                st.subheader("Results")

                # Show active period notice if it differs from config
                active_start = result.metrics.get("active_start")
                if active_start and str(config.start_date) != active_start:
                    st.info(
                        f"📅 No signals before **{active_start}** (data gap). "
                        f"Metrics are calculated from {active_start} to "
                        f"{result.metrics.get('active_end', config.end_date)}."
                    )

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
