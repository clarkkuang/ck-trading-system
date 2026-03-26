"""Backtesting page."""

from datetime import date

import polars as pl
import streamlit as st

st.set_page_config(page_title="Backtest", page_icon="📊", layout="wide")
st.title("Strategy Backtesting")

from ck_trading.dashboard.widgets.stock_pool_selector import stock_pool_selector
from ck_trading.storage.metadata_store import MetadataStore
from ck_trading.strategies.registry import get_all_strategies

strategies = get_all_strategies()

col1, col2 = st.columns(2)

with col1:
    strategy_name = st.selectbox(
        "Strategy",
        list(strategies.keys()),
    )
    strategy_instance = strategies[strategy_name]()
    st.caption(f"\u2139\ufe0f {strategy_instance.description}")
    start_date = st.date_input("Start Date", date(2015, 1, 1))

with col2:
    max_positions = st.number_input("Max Positions", 5, 50, 20)
    end_date = st.date_input("End Date", date.today())

rebalance_freq = st.selectbox("Rebalance Frequency", ["quarterly", "monthly", "annual"])
initial_capital = st.number_input("Initial Capital ($)", 10000, 10000000, 1000000, step=100000)

benchmark = st.text_input("Benchmark Ticker", value="SPY")

market = st.selectbox("Market", ["us", "hk"], key="backtest_market")

# --- Stock pool selector ---
meta = MetadataStore()
selected_tickers = stock_pool_selector(meta, key_prefix="backtest_")
meta.close()

if st.button("Run Backtest", type="primary"):
    # Re-read from session state to survive the rerun
    tickers = st.session_state.get("backtest_selected_tickers", selected_tickers)

    with st.spinner("Running backtest... This may take a minute."):
        try:
            from ck_trading.backtesting.engine import BacktestEngine
            from ck_trading.models.backtest import BacktestConfig
            from ck_trading.storage.parquet_store import ParquetStore

            store = ParquetStore()
            prices = store.load_prices(market)
            fundamentals = store.load_fundamentals(market)

            # Filter to selected stock pool
            if tickers:
                # Keep benchmark ticker in prices even if not in selected pool
                pool_with_benchmark = set(tickers) | {benchmark}
                prices = prices.filter(pl.col("ticker").is_in(pool_with_benchmark))
                fundamentals = fundamentals.filter(pl.col("ticker").is_in(tickers))

            if prices.is_empty():
                st.error("No price data. Run backfill_data.py first.")
            else:
                # Only exclude the benchmark from tradeable universe, NOT all ETFs
                universe_tickers = prices["ticker"].unique().to_list()
                universe_tickers = [t for t in universe_tickers if t != benchmark]

                strategy = strategies[strategy_name]()
                config = BacktestConfig(
                    strategy_name=strategy.name,
                    universe=universe_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=float(initial_capital),
                    max_positions=max_positions,
                    rebalance_freq=rebalance_freq,
                    benchmark=benchmark,
                )

                from ck_trading.dashboard.widgets.extra_data import build_extra_data

                extra_data = build_extra_data(store, tickers)
                engine = BacktestEngine(strategy, config, prices, fundamentals, extra_data=extra_data)
                result = engine.run()

                # Debug info
                with st.expander("Debug: Stock Pool & Engine", expanded=False):
                    st.write(f"Selected tickers: {len(tickers)}")
                    st.write(f"Universe (tradeable): {sorted(universe_tickers)}")
                    st.write(f"Benchmark: {benchmark}")
                    st.write(f"Config: start={config.start_date}, end={config.end_date}, "
                             f"max_pos={config.max_positions}, freq={config.rebalance_freq}")
                    st.write(f"Prices: {prices.height} rows, date range: "
                             f"{prices['date'].min()} - {prices['date'].max()}")
                    st.write(f"Fundamentals: {fundamentals.height} rows")
                    st.write(f"Trades: {result.trades.height if not result.trades.is_empty() else 0}")
                    st.write(f"Returns: {result.returns.height if not result.returns.is_empty() else 0}")

                st.subheader("Results")

                # Show active period notice if it differs from config
                active_start = result.metrics.get("active_start")
                if active_start and str(config.start_date) != active_start:
                    st.info(
                        f"No signals before **{active_start}** (data gap). "
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
