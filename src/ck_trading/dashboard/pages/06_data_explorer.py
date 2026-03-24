"""Data Explorer page - comprehensive overview of all collected data."""

from datetime import date, datetime

import streamlit as st

st.set_page_config(page_title="Data Explorer", page_icon="\U0001f5c4\ufe0f", layout="wide")
st.title("Data Explorer")

try:
    import polars as pl

    from ck_trading.storage.metadata_store import MetadataStore
    from ck_trading.storage.parquet_store import ParquetStore

    store = ParquetStore()
    meta = MetadataStore()

    # =====================================================================
    # 1. Data Health Overview
    # =====================================================================
    st.header("Data Health Overview")

    # Load all data once for reuse
    us_prices = store.load_prices("us")
    hk_prices = store.load_prices("hk")
    us_fundamentals = store.load_fundamentals("us")
    hk_fundamentals = store.load_fundamentals("hk")
    macro = store.load_macro()
    universe = meta.get_universe()
    positions = meta.get_open_positions()
    signals = meta.get_recent_signals(limit=10_000)

    def _date_range_str(df: pl.DataFrame, col: str = "date") -> str:
        """Return 'YYYY-MM-DD to YYYY-MM-DD' or 'N/A'."""
        if df.is_empty() or col not in df.columns:
            return "N/A"
        try:
            mn = df[col].min()
            mx = df[col].max()
            return f"{mn} to {mx}"
        except Exception:
            return "N/A"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("US Price Records", f"{us_prices.height:,}")
        st.caption(_date_range_str(us_prices))
    with col2:
        st.metric("HK Price Records", f"{hk_prices.height:,}")
        st.caption(_date_range_str(hk_prices))
    with col3:
        us_fund_tickers = us_fundamentals["ticker"].n_unique() if not us_fundamentals.is_empty() and "ticker" in us_fundamentals.columns else 0
        st.metric("US Fundamentals", f"{us_fundamentals.height:,} rows / {us_fund_tickers} tickers")
    with col4:
        hk_fund_tickers = hk_fundamentals["ticker"].n_unique() if not hk_fundamentals.is_empty() and "ticker" in hk_fundamentals.columns else 0
        st.metric("HK Fundamentals", f"{hk_fundamentals.height:,} rows / {hk_fund_tickers} tickers")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        macro_series = macro["series_id"].n_unique() if not macro.is_empty() and "series_id" in macro.columns else 0
        st.metric("Macro Series", f"{macro_series} series / {macro.height:,} records")
    with col6:
        st.metric("Universe Size", f"{len(universe):,}")
    with col7:
        st.metric("Open Positions", f"{len(positions):,}")
    with col8:
        st.metric("Total Signals", f"{len(signals):,}")

    st.divider()

    # =====================================================================
    # 2. Price Data Details
    # =====================================================================
    st.header("Price Data Details")

    price_market = st.selectbox("Select Market", ["US", "HK"], key="price_market")
    price_df = us_prices if price_market == "US" else hk_prices

    if price_df.is_empty():
        st.info(f"No price data found for {price_market} market.")
    else:
        # Build per-ticker summary
        ticker_summary = (
            price_df.group_by("ticker")
            .agg(
                pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
                pl.len().alias("row_count"),
            )
            .sort("ticker")
        )

        # Add days since last update
        today = date.today()
        ticker_summary = ticker_summary.with_columns(
            pl.col("last_date")
            .cast(pl.Date)
            .map_elements(lambda d: (today - d).days, return_dtype=pl.Int64)
            .alias("days_since_update")
        )

        with st.expander("Ticker Coverage Table", expanded=True):
            st.dataframe(
                ticker_summary.to_pandas(),
                use_container_width=True,
                column_config={
                    "ticker": "Ticker",
                    "first_date": "First Date",
                    "last_date": "Last Date",
                    "row_count": st.column_config.NumberColumn("Rows", format="%d"),
                    "days_since_update": st.column_config.NumberColumn(
                        "Days Since Update", format="%d"
                    ),
                },
            )

        # Ticker price chart
        available_tickers = sorted(ticker_summary["ticker"].to_list())
        selected_ticker = st.selectbox(
            "Select ticker to chart", available_tickers, key="chart_ticker"
        )

        if selected_ticker:
            import plotly.graph_objects as go

            tdf = price_df.filter(pl.col("ticker") == selected_ticker).sort("date")

            # Try common price column names
            price_col = None
            for candidate in ["close", "adj_close", "price", "Close", "Adj Close"]:
                if candidate in tdf.columns:
                    price_col = candidate
                    break

            if price_col:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=tdf["date"].to_list(),
                        y=tdf[price_col].to_list(),
                        mode="lines",
                        name=selected_ticker,
                    )
                )
                fig.update_layout(
                    title=f"{selected_ticker} - {price_col.replace('_', ' ').title()}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    f"No recognized price column found. "
                    f"Available columns: {tdf.columns}"
                )

        # Data freshness summary
        stale = ticker_summary.filter(pl.col("days_since_update") > 7)
        if not stale.is_empty():
            st.warning(
                f"{stale.height} ticker(s) have not been updated in over 7 days."
            )

    st.divider()

    # =====================================================================
    # 3. Fundamentals Coverage
    # =====================================================================
    st.header("Fundamentals Coverage")

    fund_tabs = st.tabs(["US Fundamentals", "HK Fundamentals"])

    KEY_FUNDAMENTAL_COLS = [
        "pe_ratio", "pb_ratio", "roe", "roa", "debt_to_equity",
        "current_ratio", "revenue", "net_income", "eps",
        "dividend_yield", "free_cash_flow",
    ]

    for idx, (label, fdf) in enumerate(
        [("US", us_fundamentals), ("HK", hk_fundamentals)]
    ):
        with fund_tabs[idx]:
            if fdf.is_empty():
                st.info(f"No fundamentals data for {label} market.")
                continue

            tickers_in_fund = sorted(fdf["ticker"].unique().to_list()) if "ticker" in fdf.columns else []

            # Build coverage table
            coverage_rows = []
            for tkr in tickers_in_fund:
                tkr_df = fdf.filter(pl.col("ticker") == tkr)
                row = {
                    "ticker": tkr,
                    "periods": tkr_df.height,
                }
                if "period_end" in tkr_df.columns:
                    row["latest_period_end"] = str(tkr_df["period_end"].max())
                else:
                    row["latest_period_end"] = "N/A"

                # Check key columns
                missing_cols = []
                for kcol in KEY_FUNDAMENTAL_COLS:
                    if kcol in tkr_df.columns:
                        null_pct = tkr_df[kcol].null_count() / max(tkr_df.height, 1)
                        row[kcol] = f"{(1 - null_pct) * 100:.0f}%"
                        if null_pct == 1.0:
                            missing_cols.append(kcol)
                    else:
                        row[kcol] = "N/A"
                        missing_cols.append(kcol)
                row["missing_critical"] = ", ".join(missing_cols) if missing_cols else ""
                coverage_rows.append(row)

            if coverage_rows:
                cov_df = pl.DataFrame(coverage_rows)
                st.dataframe(
                    cov_df.to_pandas(),
                    use_container_width=True,
                    height=min(400, 35 * len(coverage_rows) + 40),
                )

                # Highlight tickers missing critical data
                tickers_with_gaps = [
                    r["ticker"] for r in coverage_rows if r["missing_critical"]
                ]
                if tickers_with_gaps:
                    st.warning(
                        f"{len(tickers_with_gaps)} ticker(s) have missing critical "
                        f"fundamental data: {', '.join(tickers_with_gaps[:20])}"
                        + (" ..." if len(tickers_with_gaps) > 20 else "")
                    )
            else:
                st.info("No fundamental records to display.")

    st.divider()

    # =====================================================================
    # 4. Universe & Stock Pool
    # =====================================================================
    st.header("Universe & Stock Pool")

    if not universe:
        st.info("Universe is empty. Run `seed_universe.py` to populate it.")
    else:
        uni_df = pl.DataFrame(universe)

        # Filters
        filter_cols = st.columns(2)
        with filter_cols[0]:
            markets_available = sorted(uni_df["market"].unique().to_list()) if "market" in uni_df.columns else []
            market_filter = st.multiselect(
                "Filter by Market", markets_available, default=markets_available, key="uni_market"
            )
        with filter_cols[1]:
            sectors_available = sorted(
                [s for s in uni_df["sector"].unique().to_list() if s]
            ) if "sector" in uni_df.columns else []
            sector_filter = st.multiselect(
                "Filter by Sector", sectors_available, default=sectors_available, key="uni_sector"
            )

        filtered_uni = uni_df
        if market_filter and "market" in uni_df.columns:
            filtered_uni = filtered_uni.filter(pl.col("market").is_in(market_filter))
        if sector_filter and "sector" in uni_df.columns:
            filtered_uni = filtered_uni.filter(
                pl.col("sector").is_in(sector_filter) | pl.col("sector").is_null()
            )

        st.caption(f"Showing {filtered_uni.height} of {uni_df.height} tickers")
        display_cols = [c for c in ["ticker", "name", "sector", "industry", "market"] if c in filtered_uni.columns]
        st.dataframe(
            filtered_uni.select(display_cols).sort("ticker").to_pandas(),
            use_container_width=True,
            height=min(600, 35 * filtered_uni.height + 40),
        )

    st.divider()

    # =====================================================================
    # 5. Alternative Data Status
    # =====================================================================
    st.header("Alternative Data Status")

    ALT_DATA_SOURCES = [
        ("earnings", "combined"),
        ("short_interest", "finra"),
        ("insider_trades", "edgar"),
    ]

    alt_cols = st.columns(len(ALT_DATA_SOURCES) + 1)  # +1 for crypto

    for i, (source, name) in enumerate(ALT_DATA_SOURCES):
        with alt_cols[i]:
            st.subheader(source.replace("_", " ").title())
            try:
                adf = store.load_alternative(source, name)
                if adf.is_empty():
                    st.info("Not collected yet - run backfill")
                else:
                    st.metric("Records", f"{adf.height:,}")
                    date_col = None
                    for cand in ["date", "trade_date", "reported_date", "timestamp", "created_at"]:
                        if cand in adf.columns:
                            date_col = cand
                            break
                    if date_col:
                        st.caption(_date_range_str(adf, date_col))
            except Exception:
                st.info("Not collected yet - run backfill")

    # Crypto lives in prices/crypto
    with alt_cols[-1]:
        st.subheader("Crypto")
        try:
            crypto = store.load_prices("crypto")
            if crypto.is_empty():
                st.info("Not collected yet - run backfill")
            else:
                tickers = crypto["ticker"].unique().to_list()
                st.metric("Records", f"{crypto.height:,}")
                st.caption(f"Tickers: {', '.join(sorted(tickers))}")
                st.caption(_date_range_str(crypto, "date"))
        except Exception:
            st.info("Not collected yet - run backfill")

    meta.close()
except Exception as e:
    st.error(f"Error: {e}")
