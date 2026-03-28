"""Parquet file storage for time-series data."""

from pathlib import Path

import polars as pl

from ck_trading.config import settings


class ParquetStore:
    """Read/write Parquet files organized by data type and market."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or settings.data_dir

    # --- Prices ---

    def save_prices(self, df: pl.DataFrame, market: str, freq: str = "daily") -> None:
        """Save price data, one Parquet file per ticker."""
        if df.is_empty():
            return

        out_dir = self.base_dir / "prices" / market.lower() / freq
        out_dir.mkdir(parents=True, exist_ok=True)

        for ticker in df["ticker"].unique().to_list():
            ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
            safe_name = ticker.replace(".", "_").replace("/", "_")
            path = out_dir / f"{safe_name}.parquet"

            # Merge with existing data
            if path.exists():
                existing = pl.read_parquet(path)
                ticker_df = pl.concat([existing, ticker_df]).unique(
                    subset=["ticker", "date"], keep="last"
                ).sort("date")

            ticker_df.write_parquet(path)

    def load_prices(
        self,
        market: str,
        tickers: list[str] | None = None,
        freq: str = "daily",
    ) -> pl.DataFrame:
        """Load price data from Parquet files."""
        data_dir = self.base_dir / "prices" / market.lower() / freq
        if not data_dir.exists():
            return pl.DataFrame()

        if tickers:
            frames = []
            for ticker in tickers:
                safe_name = ticker.replace(".", "_").replace("/", "_")
                path = data_dir / f"{safe_name}.parquet"
                if path.exists():
                    frames.append(pl.read_parquet(path))
            return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

        # Load all
        files = list(data_dir.glob("*.parquet"))
        if not files:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(f) for f in files], how="diagonal")

    # --- Fundamentals ---

    def save_fundamentals(self, df: pl.DataFrame, market: str) -> None:
        """Save fundamental data, one Parquet file per ticker."""
        if df.is_empty():
            return

        out_dir = self.base_dir / "fundamentals" / market.lower()
        out_dir.mkdir(parents=True, exist_ok=True)

        for ticker in df["ticker"].unique().to_list():
            ticker_df = df.filter(pl.col("ticker") == ticker)
            safe_name = ticker.replace(".", "_").replace("/", "_")
            path = out_dir / f"{safe_name}_financials.parquet"

            if path.exists():
                existing = pl.read_parquet(path)
                ticker_df = pl.concat([existing, ticker_df]).unique(
                    subset=["ticker", "period_end"], keep="last"
                )

            ticker_df.write_parquet(path)

    # Numeric columns that yfinance sometimes returns as strings ("N/A", "-", etc.)
    _FUND_FLOAT_COLS = {
        "revenue", "gross_profit", "operating_income", "net_income", "ebit", "ebitda",
        "total_assets", "total_liabilities", "total_equity", "current_assets",
        "current_liabilities", "cash_and_equivalents", "total_debt",
        "operating_cash_flow", "capex", "free_cash_flow",
        "pe_ratio", "pb_ratio", "ps_ratio", "roe", "roa", "dividend_yield",
        "enterprise_value", "eps", "book_value_per_share", "market_cap",
        "current_ratio", "debt_to_equity", "gross_margin", "operating_margin",
        "net_margin", "asset_turnover",
    }

    def _normalize_fundamentals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast any string-typed numeric columns to Float64 (turns non-numeric to null)."""
        casts = []
        for col in df.columns:
            if col in self._FUND_FLOAT_COLS and df[col].dtype == pl.Utf8:
                casts.append(pl.col(col).cast(pl.Float64, strict=False))
        return df.with_columns(casts) if casts else df

    def load_fundamentals(
        self, market: str, tickers: list[str] | None = None
    ) -> pl.DataFrame:
        """Load fundamental data from Parquet files."""
        data_dir = self.base_dir / "fundamentals" / market.lower()
        if not data_dir.exists():
            return pl.DataFrame()

        if tickers:
            frames = []
            for ticker in tickers:
                safe_name = ticker.replace(".", "_").replace("/", "_")
                path = data_dir / f"{safe_name}_financials.parquet"
                if path.exists():
                    frames.append(self._normalize_fundamentals(pl.read_parquet(path)))
            return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

        files = list(data_dir.glob("*_financials.parquet"))
        if not files:
            return pl.DataFrame()
        return pl.concat(
            [self._normalize_fundamentals(pl.read_parquet(f)) for f in files],
            how="diagonal",
        )

    # --- Macro ---

    def save_macro(self, df: pl.DataFrame, name: str = "macro") -> None:
        """Save macro data."""
        if df.is_empty():
            return
        out_dir = self.base_dir / "macro"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.parquet"

        if path.exists():
            existing = pl.read_parquet(path)
            df = pl.concat([existing, df]).unique(
                subset=["series_id", "date"], keep="last"
            ).sort("date")

        df.write_parquet(path)

    def load_macro(self, name: str = "macro") -> pl.DataFrame:
        """Load macro data."""
        path = self.base_dir / "macro" / f"{name}.parquet"
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    # --- Alternative Data ---

    def save_alternative(self, df: pl.DataFrame, source: str, name: str) -> None:
        """Save alternative data to ``alternative/{source}/{name}.parquet``.

        Merges with existing data and deduplicates.
        """
        if df.is_empty():
            return

        out_dir = self.base_dir / "alternative" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.parquet"

        if path.exists():
            existing = pl.read_parquet(path)
            # Deduplicate using all columns
            df = pl.concat([existing, df], how="diagonal").unique()

        df.write_parquet(path)

    def load_alternative(
        self,
        source: str,
        name: str,
        tickers: list[str] | None = None,
    ) -> pl.DataFrame:
        """Load alternative data from ``alternative/{source}/{name}.parquet``.

        Returns an empty DataFrame if the file does not exist.
        """
        path = self.base_dir / "alternative" / source / f"{name}.parquet"
        if not path.exists():
            return pl.DataFrame()

        df = pl.read_parquet(path)
        if tickers and "ticker" in df.columns:
            df = df.filter(pl.col("ticker").is_in(tickers))
        return df
