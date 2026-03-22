"""DuckDB analytical query layer over Parquet files."""

from pathlib import Path

import duckdb
import polars as pl

from ck_trading.config import settings


class DuckDBStore:
    """Query engine using DuckDB over Parquet files."""

    def __init__(self, db_path: Path | None = None, data_dir: Path | None = None):
        self.db_path = str(db_path or settings.duckdb_path)
        self.data_dir = data_dir or settings.data_dir
        self.conn = duckdb.connect(self.db_path)
        self._register_views()

    def _register_views(self) -> None:
        """Create SQL views pointing to Parquet files."""
        views = {
            "us_prices": "prices/us/daily/*.parquet",
            "hk_prices": "prices/hk/daily/*.parquet",
            "us_fundamentals": "fundamentals/us/*_financials.parquet",
            "hk_fundamentals": "fundamentals/hk/*_financials.parquet",
        }
        for view_name, glob_pattern in views.items():
            full_path = self.data_dir / glob_pattern
            # Only create view if data exists
            parent = full_path.parent
            if parent.exists() and list(parent.glob(full_path.name)):
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {view_name} AS
                    SELECT * FROM read_parquet('{full_path}')
                """)

    def refresh_views(self) -> None:
        """Re-register views after new data arrives."""
        self._register_views()

    def query(self, sql: str) -> pl.DataFrame:
        """Execute SQL and return results as Polars DataFrame."""
        return self.conn.execute(sql).pl()

    def get_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        market: str = "us",
    ) -> pl.DataFrame:
        """Get price data for given tickers and date range."""
        view = f"{market.lower()}_prices"
        placeholders = ", ".join(f"'{t}'" for t in tickers)
        return self.query(f"""
            SELECT * FROM {view}
            WHERE ticker IN ({placeholders})
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY ticker, date
        """)

    def get_fundamentals(
        self,
        tickers: list[str],
        market: str = "us",
    ) -> pl.DataFrame:
        """Get fundamental data for given tickers."""
        view = f"{market.lower()}_fundamentals"
        placeholders = ", ".join(f"'{t}'" for t in tickers)
        return self.query(f"""
            SELECT * FROM {view}
            WHERE ticker IN ({placeholders})
            ORDER BY ticker, period_end
        """)

    def screen(self, sql: str) -> pl.DataFrame:
        """Run a custom screening query.

        Example:
            db.screen('''
                SELECT ticker, pe_ratio, pb_ratio, roe
                FROM us_fundamentals
                WHERE pe_ratio < 15 AND pb_ratio < 1.5
                ORDER BY pe_ratio
            ''')
        """
        return self.query(sql)

    def get_latest_prices(self, market: str = "us") -> pl.DataFrame:
        """Get the most recent price for each ticker."""
        view = f"{market.lower()}_prices"
        return self.query(f"""
            SELECT * FROM {view}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
            ORDER BY ticker
        """)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
