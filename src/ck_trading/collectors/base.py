"""Base collector interface."""

from abc import ABC, abstractmethod
from datetime import date

import polars as pl

from ck_trading.models.market_data import Market


class BaseCollector(ABC):
    """Abstract base for all data collectors."""

    @abstractmethod
    def collect_prices(
        self, tickers: list[str], start_date: date, end_date: date
    ) -> pl.DataFrame:
        """Collect OHLCV price data.

        Returns DataFrame with columns:
            ticker, date, open, high, low, close, volume, adj_close
        """
        ...

    @abstractmethod
    def get_market(self) -> Market:
        """Return which market this collector handles."""
        ...
