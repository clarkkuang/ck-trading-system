"""Portfolio simulation utilities for backtesting."""

from datetime import date


def equal_weight_allocation(
    capital: float,
    tickers: list[str],
    prices: dict[str, float],
    max_positions: int = 20,
) -> dict[str, int]:
    """Calculate equal-weight share allocation.

    Returns: dict of ticker -> number of shares to buy
    """
    tickers = tickers[:max_positions]
    if not tickers:
        return {}

    value_per_position = capital / len(tickers)
    allocation = {}
    for ticker in tickers:
        price = prices.get(ticker, 0)
        if price > 0:
            shares = int(value_per_position / price)
            if shares > 0:
                allocation[ticker] = shares

    return allocation


def score_weighted_allocation(
    capital: float,
    ticker_scores: dict[str, float],
    prices: dict[str, float],
    max_positions: int = 20,
) -> dict[str, int]:
    """Calculate score-weighted share allocation.

    Higher score = larger position weight.
    """
    sorted_tickers = sorted(ticker_scores.keys(), key=lambda t: ticker_scores[t], reverse=True)
    sorted_tickers = sorted_tickers[:max_positions]

    if not sorted_tickers:
        return {}

    total_score = sum(ticker_scores[t] for t in sorted_tickers)
    if total_score <= 0:
        return equal_weight_allocation(capital, sorted_tickers, prices, max_positions)

    allocation = {}
    for ticker in sorted_tickers:
        weight = ticker_scores[ticker] / total_score
        value = capital * weight
        price = prices.get(ticker, 0)
        if price > 0:
            shares = int(value / price)
            if shares > 0:
                allocation[ticker] = shares

    return allocation


def generate_rebalance_schedule(
    start_date: date,
    end_date: date,
    trading_dates: list[date],
    frequency: str = "quarterly",
) -> list[date]:
    """Generate rebalance dates aligned to actual trading dates.

    Returns the closest trading date on or after each theoretical rebalance date.
    """
    from ck_trading.strategies.base import _generate_rebalance_dates

    theoretical = _generate_rebalance_dates(start_date, end_date, frequency)
    trading_set = sorted(trading_dates)

    actual = []
    for target in theoretical:
        # Find closest trading date on or after target
        for td in trading_set:
            if td >= target:
                actual.append(td)
                break

    return actual
