"""Earnings data collector using yfinance and optionally FMP."""

import logging
from datetime import date

import httpx
import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)


class EarningsCollector:
    """Collect earnings data from yfinance and optionally FMP."""

    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, fmp_api_key: str | None = None, timeout: float = 30.0):
        self.fmp_api_key = fmp_api_key
        self.timeout = timeout

    def collect_earnings_history(self, tickers: list[str]) -> pl.DataFrame:
        """Fetch historical earnings dates and surprises from yfinance.

        Returns DataFrame with columns:
        [ticker, date, eps_estimate, eps_actual, surprise_pct, revenue_estimate, revenue_actual]
        """
        if not tickers:
            return self._empty_earnings_df()

        all_rows: list[dict] = []
        for ticker in tickers:
            try:
                rows = self._collect_earnings_single(ticker)
                all_rows.extend(rows)
            except Exception as exc:
                logger.warning("Failed to collect earnings for %s: %s", ticker, exc)
                continue

        if not all_rows:
            return self._empty_earnings_df()

        df = pl.from_dicts(all_rows)
        if "date" in df.columns:
            try:
                df = df.with_columns(pl.col("date").cast(pl.Date))
            except Exception:
                pass
        return df

    def _collect_earnings_single(self, ticker: str) -> list[dict]:
        """Collect earnings history for a single ticker via yfinance."""
        yf_ticker = yf.Ticker(ticker)

        try:
            earnings_dates = yf_ticker.get_earnings_dates(limit=20)
        except Exception:
            earnings_dates = None

        if earnings_dates is None or earnings_dates.empty:
            return []

        rows: list[dict] = []
        for idx, row in earnings_dates.iterrows():
            try:
                earnings_date = idx.date() if hasattr(idx, "date") else idx
            except (AttributeError, TypeError):
                continue

            eps_estimate = row.get("EPS Estimate")
            eps_actual = row.get("Reported EPS")
            surprise_pct = row.get("Surprise(%)")

            # Clean NaN values
            eps_estimate = None if _is_nan(eps_estimate) else eps_estimate
            eps_actual = None if _is_nan(eps_actual) else eps_actual
            surprise_pct = None if _is_nan(surprise_pct) else surprise_pct

            rows.append({
                "ticker": ticker.upper(),
                "date": earnings_date,
                "eps_estimate": _to_float(eps_estimate),
                "eps_actual": _to_float(eps_actual),
                "surprise_pct": _to_float(surprise_pct),
                "revenue_estimate": None,
                "revenue_actual": None,
            })

        return rows

    def collect_analyst_estimates(self, tickers: list[str]) -> pl.DataFrame:
        """Fetch forward analyst estimates from FMP (if API key available).

        Returns DataFrame with columns:
        [ticker, date, estimated_eps, estimated_revenue, number_of_analysts]
        """
        if not self.fmp_api_key:
            logger.info("No FMP API key set, skipping analyst estimates")
            return self._empty_estimates_df()

        if not tickers:
            return self._empty_estimates_df()

        all_rows: list[dict] = []
        with httpx.Client(timeout=self.timeout) as client:
            for ticker in tickers:
                try:
                    rows = self._fetch_fmp_estimates(client, ticker)
                    all_rows.extend(rows)
                except Exception as exc:
                    logger.warning(
                        "Failed to collect analyst estimates for %s: %s", ticker, exc
                    )
                    continue

        if not all_rows:
            return self._empty_estimates_df()

        df = pl.from_dicts(all_rows)
        if "date" in df.columns:
            try:
                df = df.with_columns(pl.col("date").cast(pl.Date))
            except Exception:
                pass
        return df

    def _fetch_fmp_estimates(
        self, client: httpx.Client, ticker: str
    ) -> list[dict]:
        """Fetch analyst estimates for a single ticker from FMP."""
        url = f"{self.FMP_BASE_URL}/analyst-estimates/{ticker}"
        params = {"apikey": self.fmp_api_key}

        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            return []

        rows: list[dict] = []
        for entry in data:
            try:
                est_date = entry.get("date", "")
                rows.append({
                    "ticker": ticker.upper(),
                    "date": date.fromisoformat(est_date) if est_date else None,
                    "estimated_eps": _to_float(entry.get("estimatedEpsAvg")),
                    "estimated_revenue": _to_float(entry.get("estimatedRevenueAvg")),
                    "number_of_analysts": entry.get("numberAnalystEstimatedEps"),
                })
            except (ValueError, TypeError):
                continue

        return rows

    @staticmethod
    def _empty_earnings_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "eps_estimate": pl.Float64,
                "eps_actual": pl.Float64,
                "surprise_pct": pl.Float64,
                "revenue_estimate": pl.Float64,
                "revenue_actual": pl.Float64,
            }
        )

    @staticmethod
    def _empty_estimates_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "estimated_eps": pl.Float64,
                "estimated_revenue": pl.Float64,
                "number_of_analysts": pl.Int64,
            }
        )


def _is_nan(val) -> bool:
    """Check if a value is NaN."""
    if val is None:
        return True
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


def _to_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        import math
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None
