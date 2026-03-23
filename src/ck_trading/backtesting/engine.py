"""Vectorized backtesting engine for value investing strategies."""

from datetime import date

import polars as pl

from ck_trading.backtesting.metrics import compute_metrics
from ck_trading.models.backtest import BacktestConfig, BacktestResult
from ck_trading.strategies.base import Strategy, _generate_rebalance_dates


class BacktestEngine:
    """Run vectorized backtests for value investing strategies."""

    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig,
        prices: pl.DataFrame,
        fundamentals: pl.DataFrame,
        extra_data: dict[str, pl.DataFrame] | None = None,
    ):
        self.strategy = strategy
        self.config = config
        self.prices = prices
        self.fundamentals = fundamentals
        self.extra_data = extra_data

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        rebal_dates = _generate_rebalance_dates(
            self.config.start_date,
            self.config.end_date,
            self.config.rebalance_freq,
        )

        # Generate signals at each rebalance date
        signals = self.strategy.generate_signals(
            self.prices,
            self.fundamentals,
            self.config.start_date,
            self.config.end_date,
            rebal_dates,
            extra_data=self.extra_data,
        )

        if signals.is_empty():
            return self._empty_result()

        # Build portfolio at each rebalance
        all_trades = []
        all_positions = []

        cash = self.config.initial_capital
        holdings: dict[str, dict] = {}  # ticker -> {shares, cost_basis}

        for _i, rebal_date in enumerate(rebal_dates):
            # Get signals for this rebalance
            date_signals = signals.filter(pl.col("date") == rebal_date)
            if date_signals.is_empty():
                continue

            buy_signals = (
                date_signals
                .filter(pl.col("signal_type") == "BUY")
                .sort("score", descending=True)
                .head(self.config.max_positions)
            )

            # Get prices on rebalance date (or closest prior date)
            rebal_prices = self._get_prices_on_date(rebal_date)
            if rebal_prices.is_empty():
                continue

            target_tickers = buy_signals["ticker"].to_list()

            # Sell positions not in new target
            for ticker in list(holdings.keys()):
                if ticker not in target_tickers:
                    price_row = rebal_prices.filter(pl.col("ticker") == ticker)
                    if not price_row.is_empty():
                        sell_price = price_row["close"][0]
                        shares = holdings[ticker]["shares"]
                        proceeds = shares * sell_price
                        cost = proceeds * self.config.transaction_cost_bps / 10000
                        cash += proceeds - cost
                        all_trades.append({
                            "date": rebal_date,
                            "ticker": ticker,
                            "action": "SELL",
                            "shares": shares,
                            "price": sell_price,
                            "cost": cost,
                        })
                    del holdings[ticker]

            # Calculate position size for new buys
            # Total portfolio value
            portfolio_value = cash
            for ticker, pos in holdings.items():
                price_row = rebal_prices.filter(pl.col("ticker") == ticker)
                if not price_row.is_empty():
                    portfolio_value += pos["shares"] * price_row["close"][0]

            new_tickers = [t for t in target_tickers if t not in holdings]
            if new_tickers:
                # Equal weight across all target positions
                target_value_per_position = portfolio_value / len(target_tickers)

                for ticker in new_tickers:
                    price_row = rebal_prices.filter(pl.col("ticker") == ticker)
                    if price_row.is_empty():
                        continue
                    buy_price = price_row["close"][0]
                    if buy_price <= 0:
                        continue

                    shares = int(target_value_per_position / buy_price)
                    if shares <= 0:
                        continue

                    total_cost = shares * buy_price
                    tx_cost = total_cost * self.config.transaction_cost_bps / 10000

                    if total_cost + tx_cost > cash:
                        shares = int((cash - tx_cost) / buy_price)
                        if shares <= 0:
                            continue
                        total_cost = shares * buy_price
                        tx_cost = total_cost * self.config.transaction_cost_bps / 10000

                    cash -= total_cost + tx_cost
                    holdings[ticker] = {"shares": shares, "cost_basis": buy_price}
                    all_trades.append({
                        "date": rebal_date,
                        "ticker": ticker,
                        "action": "BUY",
                        "shares": shares,
                        "price": buy_price,
                        "cost": tx_cost,
                    })

            # Record positions
            for ticker, pos in holdings.items():
                price_row = rebal_prices.filter(pl.col("ticker") == ticker)
                price = price_row["close"][0] if not price_row.is_empty() else pos["cost_basis"]
                all_positions.append({
                    "date": rebal_date,
                    "ticker": ticker,
                    "shares": pos["shares"],
                    "value": pos["shares"] * price,
                    "weight": (pos["shares"] * price) / max(portfolio_value, 1),
                })

        # Calculate daily returns by replaying trades over trading days
        returns_df = self._calculate_daily_returns(all_trades, rebal_dates)

        trades_df = pl.DataFrame(all_trades) if all_trades else pl.DataFrame(
            schema={"date": pl.Date, "ticker": pl.Utf8, "action": pl.Utf8,
                    "shares": pl.Float64, "price": pl.Float64, "cost": pl.Float64}
        )
        positions_df = pl.DataFrame(all_positions) if all_positions else pl.DataFrame(
            schema={"date": pl.Date, "ticker": pl.Utf8, "shares": pl.Float64,
                    "value": pl.Float64, "weight": pl.Float64}
        )

        # Compute metrics — only from the first trade date onward
        # so CAGR isn't diluted by idle years with no data/signals
        metrics = {}
        if not returns_df.is_empty() and "portfolio_return" in returns_df.columns:
            active_returns = returns_df
            if all_trades:
                first_trade_date = min(t["date"] for t in all_trades)
                active_returns = returns_df.filter(pl.col("date") >= first_trade_date)

            if not active_returns.is_empty():
                metrics = compute_metrics(
                    active_returns,
                    initial_capital=self.config.initial_capital,
                    num_trades=len(all_trades),
                )
                # Record the active period for display
                metrics["active_start"] = str(active_returns["date"].min())
                metrics["active_end"] = str(active_returns["date"].max())

        return BacktestResult(
            config=self.config,
            returns=returns_df,
            trades=trades_df,
            positions=positions_df,
            metrics=metrics,
        )

    def _get_prices_on_date(self, target_date: date) -> pl.DataFrame:
        """Get the most recent price for each ticker on or before target_date."""
        return (
            self.prices
            .filter(pl.col("date") <= target_date)
            .sort(["ticker", "date"], descending=[False, True])
            .group_by("ticker")
            .first()
        )

    def _calculate_daily_returns(
        self,
        all_trades: list[dict],
        rebal_dates: list[date],
    ) -> pl.DataFrame:
        """Calculate daily portfolio returns by replaying trades over trading days."""
        if not rebal_dates:
            return pl.DataFrame()

        # Get all trading days in the backtest period
        daily_prices = (
            self.prices
            .filter(
                (pl.col("date") >= self.config.start_date)
                & (pl.col("date") <= self.config.end_date)
            )
            .sort("date")
        )

        if daily_prices.is_empty():
            return pl.DataFrame()

        dates = daily_prices["date"].unique().sort().to_list()

        # Build a lookup: date -> {ticker -> close}
        price_lookup: dict[date, dict[str, float]] = {}
        for row in daily_prices.iter_rows(named=True):
            d = row["date"]
            if d not in price_lookup:
                price_lookup[d] = {}
            price_lookup[d][row["ticker"]] = row["close"]

        # Get benchmark returns
        benchmark_prices = (
            self.prices
            .filter(
                (pl.col("ticker") == self.config.benchmark)
                & (pl.col("date") >= self.config.start_date)
                & (pl.col("date") <= self.config.end_date)
            )
            .sort("date")
        )

        if benchmark_prices.is_empty():
            benchmark_returns = {d: 0.0 for d in dates}
        else:
            bm_close = benchmark_prices.select(["date", "close"])
            bm_close = bm_close.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("return")
            )
            benchmark_returns = dict(zip(
                bm_close["date"].to_list(),
                bm_close["return"].fill_null(0).to_list(), strict=False,
            ))

        # Index trades by date for quick lookup
        # Snap non-trading-day trades to the next available trading day
        date_set = set(dates)
        trades_by_date: dict[date, list[dict]] = {}
        for trade in all_trades:
            td = trade["date"]
            # If trade date is not a trading day, find the next one
            if td not in date_set:
                snapped = None
                for candidate in dates:
                    if candidate >= td:
                        snapped = candidate
                        break
                if snapped is None:
                    # Trade is after all trading days — use last day
                    snapped = dates[-1] if dates else td
                td = snapped
            if td not in trades_by_date:
                trades_by_date[td] = []
            trades_by_date[td].append(trade)

        # Replay day by day
        cash = self.config.initial_capital
        holdings: dict[str, float] = {}  # ticker -> shares
        # Track last known price per ticker to handle missing-price days
        # (halts, cross-market calendar gaps, data holes)
        last_known_price: dict[str, float] = {}
        prev_portfolio_value = self.config.initial_capital

        records = []
        for d in dates:
            # Apply any trades that happened on or before this date but haven't been applied
            if d in trades_by_date:
                for trade in trades_by_date[d]:
                    ticker = trade["ticker"]
                    shares = trade["shares"]
                    price = trade["price"]
                    cost = trade["cost"]
                    if trade["action"] == "BUY":
                        holdings[ticker] = holdings.get(ticker, 0) + shares
                        cash -= shares * price + cost
                        last_known_price[ticker] = price
                    elif trade["action"] == "SELL":
                        holdings[ticker] = holdings.get(ticker, 0) - shares
                        cash += shares * price - cost
                        last_known_price[ticker] = price
                        if holdings.get(ticker, 0) <= 0:
                            holdings.pop(ticker, None)

            # Compute portfolio value at close
            # Use last known price when today's price is missing (halt, gap)
            day_prices = price_lookup.get(d, {})
            portfolio_value = cash
            for ticker, shares in holdings.items():
                price = day_prices.get(ticker)
                if price is not None:
                    last_known_price[ticker] = price
                else:
                    price = last_known_price.get(ticker, 0)
                portfolio_value += shares * price

            # Daily return
            if prev_portfolio_value > 0:
                daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            else:
                daily_return = 0.0

            records.append({
                "date": d,
                "portfolio_return": daily_return,
                "benchmark_return": benchmark_returns.get(d, 0.0),
            })
            prev_portfolio_value = portfolio_value

        return pl.DataFrame(records)

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            config=self.config,
            returns=pl.DataFrame(),
            trades=pl.DataFrame(),
            positions=pl.DataFrame(),
            metrics={},
        )
