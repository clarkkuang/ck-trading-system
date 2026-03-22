"""Macro economic data collector (Treasury yields, inflation, etc.)."""

from datetime import date

import polars as pl

from ck_trading.config import settings


class MacroCollector:
    """Collects macro indicators from FRED."""

    # Common FRED series IDs
    SERIES = {
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "treasury_3m": "DTB3",
        "fed_funds_rate": "DFF",
        "cpi_yoy": "CPIAUCSL",
        "unemployment": "UNRATE",
        "gdp_growth": "A191RL1Q225SBEA",
        "vix": "VIXCLS",
    }

    def collect(
        self,
        series_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Collect macro data from FRED."""
        if not settings.fred_api_key:
            return pl.DataFrame()

        from fredapi import Fred

        fred = Fred(api_key=settings.fred_api_key)
        ids = series_ids or list(self.SERIES.values())

        frames = []
        for series_id in ids:
            try:
                data = fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                )
                if data is not None and not data.empty:
                    df = pl.DataFrame({
                        "date": data.index.tolist(),
                        "value": data.values.tolist(),
                        "series_id": series_id,
                    }).with_columns(pl.col("date").cast(pl.Date))
                    frames.append(df)
            except Exception:
                continue

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames)
