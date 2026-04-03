"""Portfolio performance calculation."""

from datetime import date

import polars as pl


def build_portfolio_values(
    trades: list[dict],
    prices: pl.DataFrame,
    initial_capital: float = 0.0,
) -> pl.DataFrame:
    """Build daily portfolio value series from trade history and prices.

    Args:
        trades: list of dicts with keys: ticker, action, shares, price, date
        prices: DataFrame with columns [ticker, date, close]
        initial_capital: starting cash balance

    Returns:
        DataFrame with columns [date, portfolio_value]
    """
    if not trades or prices.is_empty():
        return pl.DataFrame()

    trades_sorted = sorted(trades, key=lambda t: t["date"])
    all_dates = prices.select("date").unique().sort("date")["date"].to_list()
    if not all_dates:
        return pl.DataFrame()

    # State: cash + holdings
    cash = initial_capital
    holdings: dict[str, float] = {}  # ticker -> shares

    # Index trades by date
    trades_by_date: dict[date, list[dict]] = {}
    for t in trades_sorted:
        d = t["date"]
        trades_by_date.setdefault(d, []).append(t)

    # Build latest price lookup per ticker per date
    price_map: dict[tuple[str, date], float] = {}
    for row in prices.iter_rows(named=True):
        price_map[(row["ticker"], row["date"])] = row["close"]

    results = []
    last_prices: dict[str, float] = {}

    for d in all_dates:
        # Execute trades for this date
        for t in trades_by_date.get(d, []):
            if t["action"] == "BUY":
                cost = t["shares"] * t["price"]
                cash -= cost
                holdings[t["ticker"]] = holdings.get(t["ticker"], 0) + t["shares"]
            elif t["action"] == "SELL":
                proceeds = t["shares"] * t["price"]
                cash += proceeds
                holdings[t["ticker"]] = holdings.get(t["ticker"], 0) - t["shares"]
                if holdings[t["ticker"]] <= 0:
                    del holdings[t["ticker"]]

        # Update prices for today
        for ticker in holdings:
            p = price_map.get((ticker, d))
            if p is not None:
                last_prices[ticker] = p

        # Calculate portfolio value
        holdings_value = sum(
            shares * last_prices.get(ticker, 0)
            for ticker, shares in holdings.items()
        )
        results.append({"date": d, "portfolio_value": cash + holdings_value})

    return pl.DataFrame(results)


def calculate_returns(values: pl.DataFrame) -> pl.DataFrame:
    """Calculate daily returns from portfolio value series.

    Args:
        values: DataFrame with columns [date, portfolio_value]

    Returns:
        DataFrame with columns [date, portfolio_value, daily_return]
    """
    if values.is_empty() or "portfolio_value" not in values.columns:
        return pl.DataFrame()

    sorted_vals = values.sort("date")
    return sorted_vals.with_columns(
        (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1)
        .fill_null(0.0)
        .alias("daily_return")
    )


def calculate_drawdowns(values: pl.DataFrame) -> pl.DataFrame:
    """Calculate drawdown series from portfolio values.

    Args:
        values: DataFrame with columns [date, portfolio_value]

    Returns:
        DataFrame with columns [date, portfolio_value, drawdown]
        drawdown is 0 at peaks, negative fraction during drawdowns.
    """
    if values.is_empty() or "portfolio_value" not in values.columns:
        return pl.DataFrame()

    sorted_vals = values.sort("date")
    return sorted_vals.with_columns(
        (
            pl.col("portfolio_value")
            / pl.col("portfolio_value").cum_max()
            - 1
        ).alias("drawdown")
    )


def period_returns(
    values: pl.DataFrame, as_of: date | None = None
) -> dict[str, float]:
    """Calculate period returns: MTD, QTD, YTD, 1Y, since inception.

    Args:
        values: DataFrame with columns [date, portfolio_value]
        as_of: reference date (defaults to last date in values)

    Returns:
        dict with keys: mtd, qtd, ytd, 1y, since_inception
    """
    if values.is_empty() or "portfolio_value" not in values.columns:
        return {}

    sorted_vals = values.sort("date")
    dates = sorted_vals["date"].to_list()
    vals = sorted_vals["portfolio_value"].to_list()

    if not dates:
        return {}

    if as_of is None:
        as_of = dates[-1]

    latest_val = vals[-1]
    first_val = vals[0]

    def _value_at_or_before(target: date) -> float | None:
        """Find portfolio value on or just before target date."""
        best = None
        for d, v in zip(dates, vals, strict=True):
            if d <= target:
                best = v
            else:
                break
        return best

    def _return_since(ref_date: date) -> float:
        ref_val = _value_at_or_before(ref_date)
        if ref_val is None or ref_val == 0:
            return 0.0
        return (latest_val - ref_val) / ref_val

    # Period start dates
    mtd_start = date(as_of.year, as_of.month, 1)
    q_month = ((as_of.month - 1) // 3) * 3 + 1
    qtd_start = date(as_of.year, q_month, 1)
    ytd_start = date(as_of.year, 1, 1)
    one_year_ago = date(as_of.year - 1, as_of.month, as_of.day)

    return {
        "mtd": _return_since(mtd_start),
        "qtd": _return_since(qtd_start),
        "ytd": _return_since(ytd_start),
        "1y": _return_since(one_year_ago),
        "since_inception": (latest_val - first_val) / first_val if first_val else 0.0,
    }
