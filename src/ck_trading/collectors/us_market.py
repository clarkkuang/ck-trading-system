"""US stock market data collector using yfinance."""

from datetime import date

import pandas as pd
import polars as pl
import yfinance as yf

from ck_trading.collectors.base import BaseCollector
from ck_trading.models.market_data import Market


class USMarketCollector(BaseCollector):
    def get_market(self) -> Market:
        return Market.US

    def collect_prices(
        self, tickers: list[str], start_date: date, end_date: date
    ) -> pl.DataFrame:
        """Download US stock prices via yfinance."""
        if not tickers:
            return _empty_price_df()

        raw = yf.download(
            tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )

        if raw.empty:
            return _empty_price_df()

        frames = []
        if len(tickers) == 1:
            # Newer yfinance returns MultiIndex columns (field, ticker) even
            # for a single ticker with group_by="ticker" — flatten first.
            single = raw
            if isinstance(raw.columns, pd.MultiIndex):
                try:
                    single = raw[tickers[0]]
                except KeyError:
                    single = raw.droplevel(
                        axis=1,
                        level=1 if tickers[0] in raw.columns.get_level_values(1)
                        else 0,
                    )
            df = _pandas_to_polars_ohlcv(single.dropna(how="all"), tickers[0])
            frames.append(df)
        else:
            for ticker in tickers:
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

    def collect_info(self, tickers: list[str]) -> pl.DataFrame:
        """Collect basic stock info (sector, industry, market cap)."""
        records = []
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                records.append({
                    "ticker": ticker,
                    "name": info.get("longName", info.get("shortName", ticker)),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "market_cap": info.get("marketCap"),
                    "currency": info.get("currency", "USD"),
                })
            except Exception:
                records.append({
                    "ticker": ticker,
                    "name": ticker,
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "currency": "USD",
                })
        return pl.DataFrame(records)


def _pandas_to_polars_ohlcv(pdf, ticker: str) -> pl.DataFrame:
    """Convert a pandas OHLCV DataFrame to a standardized Polars DataFrame."""
    pdf = pdf.reset_index()
    df = pl.from_pandas(pdf)

    # Normalize column names to lowercase
    col_map = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(col_map)

    # Ensure date column
    date_col = "date" if "date" in df.columns else df.columns[0]
    if date_col != "date":
        df = df.rename({date_col: "date"})

    df = df.with_columns(
        pl.col("date").cast(pl.Date),
        pl.lit(ticker).alias("ticker"),
    )

    # Normalize volume to Int64 (yfinance sometimes returns Float64)
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Int64))

    # Select and order columns
    available = df.columns
    columns = ["ticker", "date", "open", "high", "low", "close", "volume"]
    if "adj_close" in available or "adj close" in available:
        adj_col = "adj_close" if "adj_close" in available else "adj close"
        df = df.rename({adj_col: "adj_close"}) if adj_col != "adj_close" else df
        columns.append("adj_close")

    return df.select([c for c in columns if c in df.columns])


def _empty_price_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "adj_close": pl.Float64,
        }
    )
