"""Import portfolio positions from broker CSV exports.

Supports:
- Chase (JP Morgan) investment account exports
- Generic CSV format (ticker, shares, avg_cost, date)
"""

from datetime import date
from pathlib import Path

import polars as pl

from ck_trading.models.market_data import Market
from ck_trading.models.portfolio import Position


def import_chase_csv(file_path: str | Path) -> list[Position]:
    """Import positions from Chase investment account CSV export.

    Chase CSV typically has columns like:
        Account Number, Account Name, Symbol, Description, Quantity,
        Price, Market Value, Cost Basis, Gain/Loss, ...

    Column names may vary; we match flexibly.
    """
    df = pl.read_csv(str(file_path), infer_schema_length=0)

    # Normalize column names
    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(col_map)

    # Find the right columns (Chase uses various names)
    ticker_col = _find_column(df, ["symbol", "ticker", "sym"])
    qty_col = _find_column(df, ["quantity", "qty", "shares"])
    cost_col = _find_column(
        df, ["cost_basis_per_share", "avg_cost", "average_cost", "unit_cost"]
    )
    # If no per-share cost, try total cost basis / quantity
    total_cost_col = _find_column(df, ["cost_basis", "total_cost", "cost"])

    if ticker_col is None or qty_col is None:
        raise ValueError(
            f"Cannot find required columns. Found: {df.columns}. "
            "Need at least a symbol/ticker column and a quantity/shares column."
        )

    positions = []
    for row in df.iter_rows(named=True):
        ticker = str(row[ticker_col]).strip().upper()
        if not ticker or ticker in ("", "CASH", "TOTAL", "--"):
            continue

        try:
            shares = float(str(row[qty_col]).replace(",", "").replace("$", ""))
        except (ValueError, TypeError):
            continue

        if shares <= 0:
            continue

        # Determine avg cost
        avg_cost = 0.0
        if cost_col and row.get(cost_col):
            try:
                avg_cost = float(
                    str(row[cost_col]).replace(",", "").replace("$", "")
                )
            except (ValueError, TypeError):
                pass

        if avg_cost == 0 and total_cost_col and row.get(total_cost_col):
            try:
                total = float(
                    str(row[total_cost_col]).replace(",", "").replace("$", "")
                )
                avg_cost = abs(total) / shares
            except (ValueError, TypeError):
                pass

        # Detect market from ticker
        market = Market.HK if ticker.endswith(".HK") else Market.US

        positions.append(Position(
            ticker=ticker,
            market=market,
            shares=shares,
            avg_cost=avg_cost,
            date_acquired=date.today(),
        ))

    return positions


def import_generic_csv(file_path: str | Path) -> list[Position]:
    """Import from a simple CSV with columns: ticker, shares, avg_cost, date.

    Flexible matching: any CSV with at least ticker and shares columns.
    """
    df = pl.read_csv(str(file_path), infer_schema_length=0)
    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(col_map)

    ticker_col = _find_column(df, ["ticker", "symbol", "sym", "stock"])
    qty_col = _find_column(df, ["shares", "quantity", "qty"])
    # Per-share cost candidates — order matters: exact match first, no "cost"
    # alone (ambiguous with cost_basis / total_cost)
    cost_col = _find_column(
        df, ["avg_cost", "average_cost", "cost_per_share", "unit_cost", "avg_price"],
        exact_only=True,
    )
    # If no per-share cost column, try total cost and divide by shares later
    total_cost_col = _find_column(
        df, ["cost_basis", "total_cost"],
        exact_only=True,
    )
    # Fallback: "price" or "cost" exact match (ambiguous but common in simple CSVs)
    if cost_col is None and total_cost_col is None:
        cost_col = _find_column(df, ["price", "cost"], exact_only=True)
    date_col = _find_column(df, ["date", "date_acquired", "purchase_date"])

    if ticker_col is None or qty_col is None:
        raise ValueError(
            f"Cannot find required columns. Found: {df.columns}. "
            "Need at least: ticker/symbol and shares/quantity."
        )

    positions = []
    for row in df.iter_rows(named=True):
        ticker = str(row[ticker_col]).strip().upper()
        if not ticker:
            continue

        try:
            shares = float(str(row[qty_col]).replace(",", ""))
        except (ValueError, TypeError):
            continue

        if shares <= 0:
            continue

        avg_cost = 0.0
        if cost_col and row.get(cost_col):
            try:
                avg_cost = float(
                    str(row[cost_col]).replace(",", "").replace("$", "")
                )
            except (ValueError, TypeError):
                pass

        # If no per-share cost found, derive from total cost / shares
        if avg_cost == 0 and total_cost_col and row.get(total_cost_col):
            try:
                total = float(
                    str(row[total_cost_col]).replace(",", "").replace("$", "")
                )
                avg_cost = abs(total) / shares
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        acq_date = date.today()
        if date_col and row.get(date_col):
            try:
                acq_date = date.fromisoformat(str(row[date_col]).strip())
            except ValueError:
                pass

        market = Market.HK if ticker.endswith(".HK") else Market.US

        positions.append(Position(
            ticker=ticker,
            market=market,
            shares=shares,
            avg_cost=avg_cost,
            date_acquired=acq_date,
        ))

    return positions


def _find_column(
    df: pl.DataFrame,
    candidates: list[str],
    exact_only: bool = False,
) -> str | None:
    """Find the first matching column name from candidates.

    Args:
        df: DataFrame to search columns in.
        candidates: Column name candidates, tried in order.
        exact_only: If True, only exact (case-insensitive) matches are used.
            If False (default), falls back to substring matching.
    """
    col_lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        # Exact match (case-insensitive)
        if candidate in col_lower_map:
            return col_lower_map[candidate]
    if not exact_only:
        # Partial / substring match
        for candidate in candidates:
            for col in df.columns:
                if candidate in col.lower():
                    return col
    return None
