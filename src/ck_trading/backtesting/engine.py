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
    ):
        self.strategy = strategy
        self.config = config
        self.prices = prices
        self.fundamentals = fundamentals

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

        # Calculate daily returns
        returns_df = self._calculate_daily_returns(holdings, cash, rebal_dates)

        trades_df = pl.DataFrame(all_trades) if all_trades else pl.DataFrame(
            schema={"date": pl.Date, "ticker": pl.Utf8, "action": pl.Utf8,
                    "shares": pl.Float64, "price": pl.Float64, "cost": pl.Float64}
        )
        positions_df = pl.DataFrame(all_positions) if all_positions else pl.DataFrame(
            schema={"date": pl.Date, "ticker": pl.Utf8, "shares": pl.Float64,
                    "value": pl.Float64, "weight": pl.Float64}
        )

        # Compute metrics
        metrics = {}
        if not returns_df.is_empty() and "portfolio_return" in returns_df.columns:
            metrics = compute_metrics(
                returns_df,
                initial_capital=self.config.initial_capital,
            )

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
        final_holdings: dict,
        final_cash: float,
        rebal_dates: list[date],
    ) -> pl.DataFrame:
        """Calculate daily portfolio returns over the backtest period."""
        if not rebal_dates:
            return pl.DataFrame()

        # Get all trading days
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

        # Simplified: compute total return between rebalance dates
        records = []
        for d in dates:
            records.append({
                "date": d,
                "portfolio_return": 0.0,  # Simplified
                "benchmark_return": benchmark_returns.get(d, 0.0),
            })

        return pl.DataFrame(records)

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            config=self.config,
            returns=pl.DataFrame(),
            trades=pl.DataFrame(),
            positions=pl.DataFrame(),
            metrics={},
        )
