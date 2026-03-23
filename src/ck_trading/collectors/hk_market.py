"""Hong Kong stock market data collector using yfinance."""

from datetime import date

import polars as pl
import yfinance as yf

from ck_trading.collectors.base import BaseCollector
from ck_trading.collectors.us_market import _empty_price_df, _pandas_to_polars_ohlcv
from ck_trading.models.market_data import Market


class HKMarketCollector(BaseCollector):
    def get_market(self) -> Market:
        return Market.HK

    def collect_prices(
        self, tickers: list[str], start_date: date, end_date: date
    ) -> pl.DataFrame:
        """Download HK stock prices via yfinance.

        Tickers can be provided as plain numbers (e.g., "0005", "700")
        and will be normalized to yfinance format (e.g., "0005.HK", "0700.HK").
        """
        yf_tickers = [self._normalize_ticker(t) for t in tickers]

        if not yf_tickers:
            return _empty_price_df()

        raw = yf.download(
            yf_tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )

        if raw.empty:
            return _empty_price_df()

        frames = []
        if len(yf_tickers) == 1:
            df = _pandas_to_polars_ohlcv(raw, yf_tickers[0])
            frames.append(df)
        else:
            for ticker in yf_tickers:
                try:
                    ticker_data = raw[ticker].dropna(how="all")
                    if not ticker_data.empty:
                        df = _pandas_to_polars_ohlcv(ticker_data, ticker)
                        frames.append(df)
                except KeyError:
                    continue

        if not frames:
            return _empty_price_df()

        return pl.concat(frames, how="diagonal")

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """Normalize HK ticker to yfinance format.

        Examples:
            "0005" -> "0005.HK"
            "700" -> "0700.HK"
            "0005.HK" -> "0005.HK"
        """
        ticker = ticker.strip().upper()
        if ticker.endswith(".HK"):
            return ticker
        # Remove any non-digit characters
        digits = "".join(c for c in ticker if c.isdigit())
        # Zero-pad to 4 digits
        return f"{int(digits):04d}.HK"
