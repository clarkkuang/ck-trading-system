"""Portfolio page - view holdings, P&L, import from CSV, manual entry."""

from datetime import date

import streamlit as st

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
st.title("Portfolio")

# Session state init
# _SENTINEL distinguishes "parsed but empty" from "not yet parsed"
_NOT_PARSED = "NOT_PARSED"
if "parsed_positions" not in st.session_state:
    st.session_state.parsed_positions = _NOT_PARSED
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Holdings"
if "import_success" not in st.session_state:
    st.session_state.import_success = 0
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

try:
    from ck_trading.portfolio.importer import import_chase_csv, import_generic_csv
    from ck_trading.portfolio.tracker import PortfolioTracker
    from ck_trading.storage.metadata_store import MetadataStore

    meta = MetadataStore()
    tracker = PortfolioTracker(meta)

    # Navigation via sidebar radio
    page = st.sidebar.radio(
        "Portfolio",
        ["Holdings", "Import from CSV", "Manual Entry"],
        index=["Holdings", "Import from CSV", "Manual Entry"].index(
            st.session_state.active_tab
        ),
        key="portfolio_nav",
    )
    # Sync back
    st.session_state.active_tab = page

    # ===================== Holdings =====================
    if page == "Holdings":
        positions = tracker.get_positions()

        if not positions:
            st.info(
                "No positions yet. Use **Import from CSV** or "
                "**Manual Entry** in the sidebar."
            )
        else:
            import polars as pl

            df = pl.DataFrame(positions).select([
                "ticker", "market", "shares", "avg_cost", "date_acquired",
            ])
            df = df.with_columns(
                (pl.col("shares") * pl.col("avg_cost"))
                .round(2)
                .alias("cost_basis")
            )

            st.dataframe(
                df.to_pandas(),
                use_container_width=True,
                column_config={
                    "avg_cost": st.column_config.NumberColumn(
                        "Avg Cost", format="$%.2f"
                    ),
                    "cost_basis": st.column_config.NumberColumn(
                        "Cost Basis", format="$%.2f"
                    ),
                },
            )

            total_cost = df["cost_basis"].sum()
            num_positions = df.height
            col1, col2, col3 = st.columns(3)
            col1.metric("Positions", num_positions)
            col2.metric("Total Cost Basis", f"${total_cost:,.2f}")
            col3.metric(
                "Avg Position Size", f"${total_cost / num_positions:,.2f}"
            )

    # ===================== Import from CSV =====================
    elif page == "Import from CSV":
        st.subheader("Import Positions from Broker CSV")

        # Show success message from previous import, then clear it
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
            "CSV Format",
            ["Chase (JP Morgan)", "Generic CSV"],
            horizontal=True,
        )

        uploaded = st.file_uploader(
            "Upload CSV file", type=["csv"], key="csv_upload"
        )

        # Parse uploaded file — only when a NEW file appears
        is_new_file = (
            uploaded is not None
            and uploaded.name != st.session_state.last_uploaded_name
        )
        if is_new_file:
            import tempfile
            from pathlib import Path

            st.session_state.last_uploaded_name = uploaded.name

            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="wb"
            ) as f:
                f.write(uploaded.getvalue())
                tmp_path = f.name

            try:
                if csv_format == "Chase (JP Morgan)":
                    st.session_state.parsed_positions = import_chase_csv(
                        tmp_path
                    )
                else:
                    st.session_state.parsed_positions = import_generic_csv(
                        tmp_path
                    )
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
                st.session_state.parsed_positions = _NOT_PARSED
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # Show preview and confirm/cancel
        parsed = st.session_state.parsed_positions
        if parsed is _NOT_PARSED:
            pass  # Nothing parsed yet — show uploader only
        elif isinstance(parsed, list) and len(parsed) == 0:
            # Parsed OK but no positions found
            st.warning(
                "No positions found in the CSV. Check the file format "
                "and try again."
            )
            if st.button("Clear", key="clear_empty"):
                st.session_state.parsed_positions = _NOT_PARSED
                st.session_state.last_uploaded_name = None
                st.rerun()
        elif isinstance(parsed, list) and len(parsed) > 0:
            st.success(f"Found {len(parsed)} positions:")

            preview_data = []
            for p in parsed:
                preview_data.append({
                    "Ticker": p.ticker,
                    "Shares": p.shares,
                    "Avg Cost": f"${p.avg_cost:,.2f}",
                    "Cost Basis": f"${p.cost_basis:,.2f}",
                    "Market": p.market.value,
                })
            st.dataframe(preview_data, use_container_width=True)

            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button(
                    f"Confirm: import {len(parsed)} positions",
                    type="primary",
                    key="confirm_import",
                ):
                    for p in parsed:
                        meta.save_position(p)
                        meta.add_to_watchlist(
                            p.ticker, p.market.value, source="portfolio"
                        )
                    count = len(parsed)
                    # Reset state so the same file can't be re-imported
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
            ticker = st.text_input(
                "Ticker", placeholder="NVDA", key="manual_ticker"
            )
            shares = st.number_input(
                "Shares", min_value=0.0, step=1.0, key="manual_shares"
            )
            market = st.selectbox(
                "Market", ["US", "HK"], key="manual_market"
            )
        with col2:
            avg_cost = st.number_input(
                "Average Cost ($)",
                min_value=0.0,
                step=0.01,
                key="manual_cost",
            )
            acq_date = st.date_input(
                "Date Acquired", value=date.today(), key="manual_date"
            )

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
                meta.add_to_watchlist(
                    ticker.upper(), market, source="portfolio"
                )
                st.toast(
                    f"Added {shares:.0f} shares of {ticker.upper()} "
                    f"@ ${avg_cost:.2f}"
                )
                st.rerun()
            else:
                st.warning("Please enter ticker and shares.")

    meta.close()
except Exception as e:
    st.error(f"Error: {e}")
