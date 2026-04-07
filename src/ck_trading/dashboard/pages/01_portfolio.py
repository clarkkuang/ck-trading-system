"""Portfolio page - holdings, performance, rebalancing, dividends, risk, import, manual entry."""

from datetime import date

import streamlit as st

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
st.title("Portfolio")

# Session state init
_NOT_PARSED = "NOT_PARSED"
if "parsed_positions" not in st.session_state:
    st.session_state.parsed_positions = _NOT_PARSED
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Holdings"
if "import_success" not in st.session_state:
    st.session_state.import_success = 0
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

_NAV_OPTIONS = [
    "Holdings", "Performance", "Rebalancing", "Dividends", "Risk",
    "Import from CSV", "Manual Entry",
]

try:
    import polars as pl
    import plotly.graph_objects as go

    from ck_trading.dashboard.data_cache import load_prices, load_prices_for_tickers
    from ck_trading.dashboard.portfolio_helpers import (
        build_current_holdings,
        build_risk_positions,
        positions_to_trades,
    )
    from ck_trading.portfolio.importer import import_chase_csv, import_generic_csv
    from ck_trading.portfolio.tracker import PortfolioTracker
    from ck_trading.storage.metadata_store import MetadataStore

    meta = MetadataStore()
    tracker = PortfolioTracker(meta)

    # Navigation
    if st.session_state.active_tab not in _NAV_OPTIONS:
        st.session_state.active_tab = "Holdings"
    page = st.sidebar.radio(
        "Portfolio",
        _NAV_OPTIONS,
        index=_NAV_OPTIONS.index(st.session_state.active_tab),
        key="portfolio_nav",
    )
    st.session_state.active_tab = page

    # Shared: load positions once
    positions = tracker.get_positions()
    _pos_tickers = tuple(sorted(set(p["ticker"] for p in positions))) if positions else ()

    # ===================== Holdings =====================
    if page == "Holdings":
        if not positions:
            st.info("No positions yet. Use **Import from CSV** or **Manual Entry** in the sidebar.")
        else:
            df = pl.DataFrame(positions).select(["ticker", "market", "shares", "avg_cost", "date_acquired"])
            df = df.with_columns((pl.col("shares") * pl.col("avg_cost")).round(2).alias("cost_basis"))
            st.dataframe(df.to_pandas(), use_container_width=True, column_config={
                "avg_cost": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
                "cost_basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
            })
            total_cost = df["cost_basis"].sum()
            num_positions = df.height
            c1, c2, c3 = st.columns(3)
            c1.metric("Positions", num_positions)
            c2.metric("Total Cost Basis", f"${total_cost:,.2f}")
            c3.metric("Avg Position Size", f"${total_cost / num_positions:,.2f}")

    # ===================== Performance =====================
    elif page == "Performance":
        st.subheader("Portfolio Performance")
        if not positions:
            st.info("Add positions first to see performance.")
        else:
            from ck_trading.dashboard.portfolio_helpers import build_portfolio_value_series
            from ck_trading.portfolio.performance import (
                calculate_drawdowns,
                calculate_returns,
                period_returns,
            )

            prices = load_prices_for_tickers("us", _pos_tickers) if _pos_tickers else pl.DataFrame()
            if prices.is_empty():
                st.warning("No price data available.")
            else:
                start_date = st.date_input("Performance Start Date", date(2024, 1, 1), key="perf_start")
                filtered_prices = prices.filter(pl.col("date") >= start_date)
                values = build_portfolio_value_series(positions, filtered_prices)
                if values.is_empty():
                    st.warning("No price data for your holdings in this date range.")
                else:
                    # KPI row
                    pr = period_returns(values)
                    if pr:
                        cols = st.columns(5)
                        for col, (key, label) in zip(cols, [
                            ("mtd", "MTD"), ("qtd", "QTD"), ("ytd", "YTD"),
                            ("1y", "1Y"), ("since_inception", "Since Inception"),
                        ]):
                            val = pr.get(key, 0)
                            col.metric(label, f"{val * 100:+.2f}%")

                    st.divider()

                    # Portfolio value chart
                    fig_val = go.Figure()
                    fig_val.add_trace(go.Scatter(
                        x=values["date"].to_list(),
                        y=values["portfolio_value"].to_list(),
                        mode="lines", name="Portfolio Value",
                        line=dict(color="#1f77b4"),
                    ))
                    fig_val.update_layout(title="Portfolio Value", height=400,
                                         xaxis_title="Date", yaxis_title="Value ($)",
                                         yaxis_tickformat="$,.0f")
                    st.plotly_chart(fig_val, use_container_width=True)

                    # Drawdown chart
                    dd = calculate_drawdowns(values)
                    if not dd.is_empty():
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(
                            x=dd["date"].to_list(),
                            y=dd["drawdown"].to_list(),
                            mode="lines", fill="tozeroy", name="Drawdown",
                            line=dict(color="#d62728"),
                        ))
                        fig_dd.update_layout(title="Drawdown", height=300,
                                             xaxis_title="Date", yaxis_title="Drawdown",
                                             yaxis_tickformat=".1%")
                        st.plotly_chart(fig_dd, use_container_width=True)

    # ===================== Rebalancing =====================
    elif page == "Rebalancing":
        st.subheader("Portfolio Rebalancing")
        if not positions:
            st.info("Add positions first.")
        else:
            from ck_trading.portfolio.rebalancer import (
                calculate_drift,
                recommend_rebalance_trades,
            )

            prices = load_prices_for_tickers("us", _pos_tickers) if _pos_tickers else pl.DataFrame()
            # Get latest close per ticker
            latest_prices: dict[str, float] = {}
            if not prices.is_empty():
                for row in prices.sort("date", descending=True).group_by("ticker").first().iter_rows(named=True):
                    latest_prices[row["ticker"]] = row["close"]

            current = build_current_holdings(positions, latest_prices)
            total_value = sum(h["market_value"] for h in current.values())
            tickers = list(current.keys())

            if total_value <= 0:
                st.warning("No market value data available.")
            else:
                # Current weights
                st.markdown("**Current Weights**")
                weight_data = [
                    {"Ticker": t, "Shares": current[t]["shares"],
                     "Market Value": f"${current[t]['market_value']:,.2f}",
                     "Weight": f"{current[t]['market_value'] / total_value * 100:.1f}%"}
                    for t in tickers
                ]
                st.dataframe(weight_data, use_container_width=True)

                st.divider()

                # Target weights
                st.markdown("**Target Weights** (edit below, should sum to 1.0)")
                equal_w = round(1.0 / len(tickers), 4) if tickers else 0
                target_weights: dict[str, float] = {}
                cols = st.columns(min(len(tickers), 4))
                for i, t in enumerate(tickers):
                    with cols[i % len(cols)]:
                        target_weights[t] = st.number_input(
                            t, min_value=0.0, max_value=1.0,
                            value=equal_w, step=0.01, format="%.4f",
                            key=f"tw_{t}",
                        )

                total_target = sum(target_weights.values())
                if abs(total_target - 1.0) > 0.01:
                    st.warning(f"Target weights sum to {total_target:.4f} (should be ~1.0)")

                st.divider()

                # Drift analysis
                drift = calculate_drift(current, target_weights)
                drift_data = [
                    {"Ticker": t, "Drift": f"{d * 100:+.2f}%",
                     "Status": "Overweight" if d > 0.01 else "Underweight" if d < -0.01 else "On target"}
                    for t, d in sorted(drift.items(), key=lambda x: abs(x[1]), reverse=True)
                ]
                st.markdown("**Weight Drift**")
                st.dataframe(drift_data, use_container_width=True)

                # Trade recommendations
                trades = recommend_rebalance_trades(
                    current, target_weights, latest_prices, total_value
                )
                if trades:
                    st.markdown("**Recommended Trades**")
                    st.dataframe(
                        [{
                            "Ticker": t["ticker"], "Action": t["action"],
                            "Shares": t["shares"],
                            "Est. Value": f"${t['estimated_value']:,.2f}",
                        } for t in trades],
                        use_container_width=True,
                    )
                else:
                    st.success("Portfolio is balanced. No trades needed.")

    # ===================== Dividends =====================
    elif page == "Dividends":
        st.subheader("Dividend Tracker")

        from ck_trading.portfolio.dividends import DividendTracker

        if "dividend_tracker" not in st.session_state:
            st.session_state.dividend_tracker = DividendTracker()
        dt: DividendTracker = st.session_state.dividend_tracker

        # Entry form
        with st.expander("Add Dividend", expanded=True):
            tickers = [p["ticker"] for p in positions] if positions else []
            c1, c2 = st.columns(2)
            with c1:
                div_ticker = st.selectbox("Ticker", tickers if tickers else ["N/A"], key="div_ticker")
                div_ex = st.date_input("Ex-Date", key="div_ex")
                div_pay = st.date_input("Pay Date", key="div_pay")
            with c2:
                div_amt = st.number_input("Amount per Share ($)", min_value=0.0, step=0.01, key="div_amt")
                # Auto-fill shares from positions
                pos_shares = 0.0
                for p in (positions or []):
                    if p["ticker"] == div_ticker:
                        pos_shares = p["shares"]
                        break
                div_shares = st.number_input("Shares Held", min_value=0.0, value=pos_shares, step=1.0, key="div_shares")

            if st.button("Record Dividend", type="primary", key="add_div"):
                if div_ticker and div_ticker != "N/A" and div_amt > 0 and div_shares > 0:
                    dt.add_dividend(div_ticker, div_ex, div_pay, div_amt, div_shares)
                    st.toast(f"Recorded ${div_amt:.2f}/share for {div_ticker}")
                    st.rerun()
                else:
                    st.warning("Fill in all fields.")

        # Summary
        income = dt.total_income()
        by_ticker = dt.income_by_ticker()

        if income > 0:
            c1, c2 = st.columns(2)
            c1.metric("Total Dividend Income", f"${income:,.2f}")

            # Projected annual for first ticker with dividends
            projected = 0.0
            for t in by_ticker:
                for p in (positions or []):
                    if p["ticker"] == t:
                        projected += dt.projected_annual_income(t, p["shares"])
                        break
            c2.metric("Projected Annual Income", f"${projected:,.2f}")

            # Bar chart
            if by_ticker:
                fig = go.Figure(go.Bar(
                    x=list(by_ticker.keys()),
                    y=list(by_ticker.values()),
                    marker_color="#2ca02c",
                ))
                fig.update_layout(title="Dividend Income by Ticker", height=350,
                                  xaxis_title="Ticker", yaxis_title="Income ($)",
                                  yaxis_tickformat="$,.2f")
                st.plotly_chart(fig, use_container_width=True)

            # History table
            all_records = dt._records
            if all_records:
                st.markdown("**Dividend History**")
                st.dataframe([{
                    "Ticker": r.ticker,
                    "Ex-Date": r.ex_date,
                    "Pay Date": r.pay_date,
                    "$/Share": f"${r.amount_per_share:.4f}",
                    "Shares": r.shares_held,
                    "Total": f"${r.total_amount:.2f}",
                } for r in all_records], use_container_width=True)
        else:
            st.info("No dividends recorded yet. Use the form above to add one.")

    # ===================== Risk =====================
    elif page == "Risk":
        st.subheader("Risk Analytics")
        if not positions:
            st.info("Add positions first.")
        else:
            from ck_trading.portfolio.risk import (
                concentration_analysis,
                correlation_matrix,
                find_correlated_tickers,
                portfolio_var,
                stress_test,
            )

            prices = load_prices_for_tickers("us", _pos_tickers) if _pos_tickers else pl.DataFrame()
            latest_prices: dict[str, float] = {}
            if not prices.is_empty():
                for row in prices.sort("date", descending=True).group_by("ticker").first().iter_rows(named=True):
                    latest_prices[row["ticker"]] = row["close"]

            universe = meta.get_universe()
            risk_pos = build_risk_positions(positions, latest_prices, universe)
            tickers = [p["ticker"] for p in risk_pos]
            holdings = {p["ticker"]: p["market_value"] for p in risk_pos}
            total_value = sum(holdings.values())

            # --- Concentration ---
            st.markdown("### Concentration")
            conc = concentration_analysis(risk_pos)
            c1, c2, c3 = st.columns(3)
            c1.metric("Positions", conc.get("num_positions", 0))
            c2.metric("HHI", f"{conc.get('hhi', 0):.4f}")
            c3.metric("Top 5 Weight", f"{conc.get('top5_weight', 0) * 100:.1f}%")

            by_sector = conc.get("by_sector", {})
            if by_sector:
                fig_pie = go.Figure(go.Pie(
                    labels=list(by_sector.keys()),
                    values=list(by_sector.values()),
                    hole=0.3,
                ))
                fig_pie.update_layout(title="Sector Allocation", height=350)
                st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            # --- VaR ---
            st.markdown("### Value at Risk (95%)")
            if not prices.is_empty() and holdings:
                var_val = portfolio_var(prices, holdings, 0.95)
                st.metric("Daily VaR (95%)", f"${var_val:,.2f}")
                st.caption("Expected maximum daily loss at 95% confidence (historical method).")
            else:
                st.warning("Insufficient price data for VaR calculation.")

            st.divider()

            # --- Stress Test ---
            st.markdown("### Stress Test")
            scenarios = {
                "Market Crash (-20%)": {"all": -0.20},
                "Tech Sector (-15%)": {"Technology": -0.15},
                "Rate Shock (-10%)": {"all": -0.10},
            }
            results = stress_test(risk_pos, scenarios)
            stress_data = []
            for name, r in results.items():
                stress_data.append({
                    "Scenario": name,
                    "Dollar Impact": f"${r['dollar_impact']:,.2f}",
                    "Portfolio Impact": f"{r['portfolio_impact'] * 100:.1f}%",
                })
            st.dataframe(stress_data, use_container_width=True)

            st.divider()

            # --- Correlation Matrix ---
            st.markdown("### Correlation Matrix")
            if not prices.is_empty() and len(tickers) >= 2:
                corr = correlation_matrix(prices, tickers)
                if not corr.is_empty():
                    import plotly.express as px
                    available = [t for t in tickers if t in corr.columns]
                    fig_corr = px.imshow(
                        corr.select(available).to_pandas().values,
                        x=available, y=available,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        aspect="auto",
                    )
                    fig_corr.update_layout(title="Return Correlations", height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Not enough overlapping price data to compute correlations.")
            else:
                st.info("Need at least 2 positions with price data for correlation analysis.")

            st.divider()

            # --- Correlation Finder ---
            st.markdown("### 🔍 Correlation Finder")
            st.caption("Find stocks negatively correlated with a given ticker — useful for hedging.")

            cf_col1, cf_col2, cf_col3 = st.columns(3)
            with cf_col1:
                ref_ticker = st.text_input("Reference Ticker", value="NVDA", key="corr_ref_ticker").upper().strip()
            with cf_col2:
                corr_range = st.slider("Correlation Range", -1.0, 1.0, (-1.0, -0.1), step=0.05, key="corr_range")
            with cf_col3:
                lookback = st.number_input("Lookback Days", min_value=60, max_value=504, value=252, step=21, key="corr_lookback")

            if st.button("Find Correlated Stocks", type="primary", key="run_corr_finder"):
                if not ref_ticker:
                    st.warning("Please enter a reference ticker.")
                else:
                    with st.spinner(f"Computing correlations for {ref_ticker}..."):
                        # Load ALL prices for broad search (not just portfolio tickers)
                        all_prices = load_prices("us")
                        corr_result = find_correlated_tickers(
                            all_prices, ref_ticker, lookback_days=lookback,
                        )
                        if corr_result.is_empty():
                            st.info(f"No price data found for {ref_ticker}.")
                        else:
                            # Filter to user-selected correlation range
                            min_c, max_c = corr_range
                            filtered = corr_result.filter(
                                (pl.col("correlation") >= min_c) & (pl.col("correlation") <= max_c)
                            )
                            if filtered.is_empty():
                                st.info(f"No tickers found in correlation range [{min_c:.2f}, {max_c:.2f}] with {ref_ticker}.")
                            else:
                                st.success(f"Found {filtered.height} tickers correlated with {ref_ticker} in [{min_c:.2f}, {max_c:.2f}]")
                                st.dataframe(
                                    filtered.to_pandas(),
                                    use_container_width=True,
                                    column_config={
                                        "other_ticker": st.column_config.TextColumn("Ticker"),
                                        "correlation": st.column_config.NumberColumn("Correlation", format="%.4f"),
                                    },
                                )

    # ===================== Import from CSV =====================
    elif page == "Import from CSV":
        st.subheader("Import Positions from Broker CSV")

        if st.session_state.import_success:
            st.success(
                f"Successfully imported {st.session_state.import_success} "
                f"positions! You can view them in **Holdings**."
            )
            st.session_state.import_success = 0

        st.markdown(
            "**Chase**: log in -> Investments -> Positions -> "
            "Export (top right) -> save as CSV"
        )

        csv_format = st.radio(
            "CSV Format", ["Chase (JP Morgan)", "Generic CSV"], horizontal=True,
        )

        uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="csv_upload")

        is_new_file = (
            uploaded is not None
            and uploaded.name != st.session_state.last_uploaded_name
        )
        if is_new_file:
            import tempfile
            from pathlib import Path

            st.session_state.last_uploaded_name = uploaded.name

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
                f.write(uploaded.getvalue())
                tmp_path = f.name

            try:
                if csv_format == "Chase (JP Morgan)":
                    st.session_state.parsed_positions = import_chase_csv(tmp_path)
                else:
                    st.session_state.parsed_positions = import_generic_csv(tmp_path)
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
                st.session_state.parsed_positions = _NOT_PARSED
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        parsed = st.session_state.parsed_positions
        if parsed is _NOT_PARSED:
            pass
        elif isinstance(parsed, list) and len(parsed) == 0:
            st.warning("No positions found in the CSV. Check the file format and try again.")
            if st.button("Clear", key="clear_empty"):
                st.session_state.parsed_positions = _NOT_PARSED
                st.session_state.last_uploaded_name = None
                st.rerun()
        elif isinstance(parsed, list) and len(parsed) > 0:
            st.success(f"Found {len(parsed)} positions:")
            preview_data = [{
                "Ticker": p.ticker, "Shares": p.shares,
                "Avg Cost": f"${p.avg_cost:,.2f}",
                "Cost Basis": f"${p.cost_basis:,.2f}",
                "Market": p.market.value,
            } for p in parsed]
            st.dataframe(preview_data, use_container_width=True)

            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button(f"Confirm: import {len(parsed)} positions", type="primary", key="confirm_import"):
                    for p in parsed:
                        meta.save_position(p)
                        meta.add_to_watchlist(p.ticker, p.market.value, source="portfolio")
                    count = len(parsed)
                    st.session_state.parsed_positions = _NOT_PARSED
                    st.session_state.last_uploaded_name = None
                    st.session_state.import_success = count
                    st.rerun()
            with col_no:
                if st.button("Cancel", key="cancel_import"):
                    st.session_state.parsed_positions = _NOT_PARSED
                    st.session_state.last_uploaded_name = None
                    st.rerun()

    # ===================== Manual Entry =====================
    elif page == "Manual Entry":
        st.subheader("Add Position Manually")

        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker", placeholder="NVDA", key="manual_ticker")
            shares = st.number_input("Shares", min_value=0.0, step=1.0, key="manual_shares")
            market = st.selectbox("Market", ["US", "HK"], key="manual_market")
        with col2:
            avg_cost = st.number_input("Average Cost ($)", min_value=0.0, step=0.01, key="manual_cost")
            acq_date = st.date_input("Date Acquired", value=date.today(), key="manual_date")

        if st.button("Add Position", type="primary", key="add_manual"):
            if ticker and shares > 0:
                from ck_trading.models.market_data import Market
                from ck_trading.models.portfolio import Position

                pos = Position(
                    ticker=ticker.upper(),
                    market=Market.HK if market == "HK" else Market.US,
                    shares=shares,
                    avg_cost=avg_cost,
                    date_acquired=acq_date,
                )
                meta.save_position(pos)
                meta.add_to_watchlist(ticker.upper(), market.lower(), source="portfolio")
                st.toast(f"Added {shares:.0f} shares of {ticker.upper()} @ ${avg_cost:.2f}")
                st.rerun()
            else:
                st.warning("Please enter ticker and shares.")

    meta.close()
except Exception as e:
    st.error(f"Error: {e}")
