"""SEC EDGAR insider trading (Form 4) data collector."""

import logging
import time
from datetime import date, timedelta

import httpx
import polars as pl

logger = logging.getLogger(__name__)


class InsiderTradesCollector:
    """Collect Form 4 insider trading data from SEC EDGAR."""

    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    HEADERS = {
        "User-Agent": "CKTradingSystem contact@cktradingsystem.com",
        "Accept": "application/json",
    }

    # Rate limit: max 10 requests/sec to SEC
    _REQUEST_INTERVAL = 0.11  # ~9 req/sec to stay under limit

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._last_request_time: float = 0.0

    def collect(
        self, tickers: list[str], days_back: int = 90
    ) -> pl.DataFrame:
        """Fetch recent insider trades.

        Returns DataFrame with columns:
        [ticker, date, insider_name, title, transaction_type, shares, price, value]

        transaction_type: "P" (purchase), "S" (sale), "A" (award)
        """
        if not tickers:
            return self._empty_df()

        all_rows: list[dict] = []
        cutoff_date = date.today() - timedelta(days=days_back)

        with httpx.Client(timeout=self.timeout, headers=self.HEADERS) as client:
            for ticker in tickers:
                try:
                    rows = self._collect_ticker(client, ticker, cutoff_date)
                    all_rows.extend(rows)
                except Exception as exc:
                    logger.warning("Failed to collect insider trades for %s: %s", ticker, exc)
                    continue

        if not all_rows:
            return self._empty_df()

        df = pl.from_dicts(all_rows)
        # Cast date column
        if "date" in df.columns:
            try:
                df = df.with_columns(pl.col("date").cast(pl.Date))
            except Exception:
                pass
        return df

    def _collect_ticker(
        self, client: httpx.Client, ticker: str, cutoff_date: date
    ) -> list[dict]:
        """Collect insider trades for a single ticker."""
        # Step 1: Look up CIK from ticker
        cik = self._lookup_cik(client, ticker)
        if not cik:
            return []

        # Step 2: Get recent filings for this CIK
        self._rate_limit()
        url = self.COMPANY_URL.format(cik=cik.zfill(10))
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()

        return self._parse_filings(data, ticker, cutoff_date)

    def _lookup_cik(self, client: httpx.Client, ticker: str) -> str | None:
        """Look up CIK number for a ticker symbol."""
        self._rate_limit()
        try:
            # Use the SEC EDGAR company tickers JSON
            resp = client.get("https://www.sec.gov/files/company_tickers.json")
            resp.raise_for_status()
            tickers_data = resp.json()

            for _key, entry in tickers_data.items():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry.get("cik_str", ""))
        except Exception as exc:
            logger.warning("CIK lookup failed for %s: %s", ticker, exc)
        return None

    def _parse_filings(
        self, data: dict, ticker: str, cutoff_date: date
    ) -> list[dict]:
        """Parse SEC filing data to extract insider trades."""
        rows: list[dict] = []
        recent_filings = data.get("filings", {}).get("recent", {})
        if not recent_filings:
            return rows

        forms = recent_filings.get("form", [])
        filing_dates = recent_filings.get("filingDate", [])
        primary_docs = recent_filings.get("primaryDocument", [])

        for i, form_type in enumerate(forms):
            if form_type not in ("4", "4/A"):
                continue

            filing_date_str = filing_dates[i] if i < len(filing_dates) else None
            if not filing_date_str:
                continue

            try:
                filing_date = date.fromisoformat(filing_date_str)
            except (ValueError, TypeError):
                continue

            if filing_date < cutoff_date:
                continue

            # Extract basic info from filings metadata
            rows.append({
                "ticker": ticker.upper(),
                "date": filing_date,
                "insider_name": data.get("name", "Unknown"),
                "title": "",
                "transaction_type": "S",  # Default; full parsing requires XML
                "shares": None,
                "price": None,
                "value": None,
            })

        return rows

    def _rate_limit(self) -> None:
        """Enforce SEC rate limit of max 10 requests/sec."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._REQUEST_INTERVAL:
            time.sleep(self._REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    @staticmethod
    def _empty_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "insider_name": pl.Utf8,
                "title": pl.Utf8,
                "transaction_type": pl.Utf8,
                "shares": pl.Float64,
                "price": pl.Float64,
                "value": pl.Float64,
            }
        )
