"""FINRA short interest data collector.

Uses two FINRA API endpoints (no authentication required):
1. regShoDaily  – Reg SHO Daily Short Sale Volume (all exchange-listed + OTC)
2. EquityShortInterest – Bimonthly short interest positions (OTC only)

The primary source is regShoDaily which provides daily aggregated short sale
volume across all FINRA Trade Reporting Facilities.  Data is aggregated across
reporting facilities to produce one row per ticker per date.
"""

import logging
from datetime import date, timedelta
from collections import defaultdict

import httpx
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FINRA API constants
# ---------------------------------------------------------------------------
_FINRA_BASE = "https://api.finra.org/data/group/otcMarket/name"
_REG_SHO_URL = f"{_FINRA_BASE}/regShoDaily"
_EQUITY_SI_URL = f"{_FINRA_BASE}/EquityShortInterest"

_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Maximum records the FINRA API returns per request
_MAX_LIMIT = 5000


class ShortInterestCollector:
    """Collect short-sale volume data from FINRA Reg SHO daily reports.

    The ``collect`` method returns a DataFrame with daily short-sale volume
    aggregated across all reporting facilities for each ticker.

    For OTC securities, ``collect_otc_positions`` can retrieve the bimonthly
    short interest position snapshots from the EquityShortInterest dataset.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API – daily short-sale volume (primary)
    # ------------------------------------------------------------------

    def collect(
        self,
        tickers: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Fetch daily short-sale volume from FINRA Reg SHO reports.

        Returns DataFrame with columns:
            ticker, date, short_volume, total_volume, short_exempt_volume,
            short_ratio
        """
        if not tickers:
            return self._empty_df()

        all_records: list[dict] = []
        for ticker in tickers:
            records = self._fetch_reg_sho(ticker, start_date, end_date)
            all_records.extend(records)

        if not all_records:
            logger.info("No Reg SHO data returned for %s", tickers)
            return self._empty_df()

        return self._aggregate_reg_sho(all_records, tickers)

    # ------------------------------------------------------------------
    # Public API – bimonthly OTC short-interest positions (secondary)
    # ------------------------------------------------------------------

    def collect_otc_positions(
        self,
        tickers: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Fetch bimonthly OTC short interest positions.

        NOTE: This only covers OTC-listed securities. Exchange-listed stocks
        (e.g. AAPL, TSLA) will return no data from this endpoint.

        Returns DataFrame with columns:
            ticker, date, short_interest, prev_short_interest,
            avg_daily_volume, days_to_cover, change_pct
        """
        if not tickers:
            return self._empty_otc_df()

        all_records: list[dict] = []
        for ticker in tickers:
            records = self._fetch_equity_si(ticker, start_date, end_date)
            all_records.extend(records)

        if not all_records:
            return self._empty_otc_df()

        return self._parse_equity_si(all_records, tickers)

    # ------------------------------------------------------------------
    # Internal – Reg SHO daily short sale volume
    # ------------------------------------------------------------------

    def _fetch_reg_sho(
        self,
        ticker: str,
        start_date: date | None,
        end_date: date | None,
    ) -> list[dict]:
        """Fetch raw Reg SHO records for a single ticker."""
        payload: dict = {
            "compareFilters": [
                {
                    "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                    "fieldValue": ticker.upper(),
                    "compareType": "EQUAL",
                },
            ],
            "limit": _MAX_LIMIT,
            "offset": 0,
        }

        # Add date range filter if provided
        if start_date or end_date:
            dr: dict = {"fieldName": "tradeReportDate"}
            if start_date:
                dr["startDate"] = start_date.isoformat()
            if end_date:
                dr["endDate"] = end_date.isoformat()
            payload["dateRangeFilters"] = [dr]

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(_REG_SHO_URL, json=payload, headers=_HEADERS)
                resp.raise_for_status()
                if resp.status_code == 204:
                    return []
                return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "FINRA regShoDaily HTTP %s for %s", exc.response.status_code, ticker
            )
        except (httpx.RequestError, Exception) as exc:
            logger.warning("FINRA regShoDaily request failed for %s: %s", ticker, exc)
        return []

    def _aggregate_reg_sho(
        self, records: list[dict], tickers: list[str]
    ) -> pl.DataFrame:
        """Aggregate raw records across reporting facilities per ticker/date."""
        ticker_set = {t.upper() for t in tickers}

        # Accumulate per (ticker, date)
        agg: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"short": 0, "total": 0, "exempt": 0}
        )
        for rec in records:
            sym = rec.get("securitiesInformationProcessorSymbolIdentifier", "")
            if sym.upper() not in ticker_set:
                continue
            key = (sym.upper(), rec.get("tradeReportDate", ""))
            agg[key]["short"] += rec.get("shortParQuantity", 0) or 0
            agg[key]["total"] += rec.get("totalParQuantity", 0) or 0
            agg[key]["exempt"] += rec.get("shortExemptParQuantity", 0) or 0

        rows = []
        for (sym, dt), vals in agg.items():
            total = vals["total"]
            short = vals["short"]
            rows.append(
                {
                    "ticker": sym,
                    "date": dt,
                    "short_volume": int(short),
                    "total_volume": int(total),
                    "short_exempt_volume": int(vals["exempt"]),
                    "short_ratio": round(short / total, 4) if total > 0 else None,
                }
            )

        if not rows:
            return self._empty_df()

        df = pl.from_dicts(rows)
        try:
            df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        except Exception:
            pass
        return df.sort(["ticker", "date"], descending=[False, True])

    # ------------------------------------------------------------------
    # Internal – OTC Equity Short Interest positions
    # ------------------------------------------------------------------

    def _fetch_equity_si(
        self,
        ticker: str,
        start_date: date | None,
        end_date: date | None,
    ) -> list[dict]:
        """Fetch raw EquityShortInterest records for a single OTC ticker."""
        payload: dict = {
            "compareFilters": [
                {
                    "fieldName": "issueSymbolIdentifier",
                    "fieldValue": ticker.upper(),
                    "compareType": "EQUAL",
                },
            ],
            "limit": _MAX_LIMIT,
            "offset": 0,
        }

        if start_date or end_date:
            dr: dict = {"fieldName": "settlementDate"}
            if start_date:
                dr["startDate"] = start_date.isoformat()
            if end_date:
                dr["endDate"] = end_date.isoformat()
            payload["dateRangeFilters"] = [dr]

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(_EQUITY_SI_URL, json=payload, headers=_HEADERS)
                resp.raise_for_status()
                if resp.status_code == 204:
                    return []
                return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "FINRA EquityShortInterest HTTP %s for %s",
                exc.response.status_code,
                ticker,
            )
        except (httpx.RequestError, Exception) as exc:
            logger.warning(
                "FINRA EquityShortInterest request failed for %s: %s", ticker, exc
            )
        return []

    def _parse_equity_si(
        self, records: list[dict], tickers: list[str]
    ) -> pl.DataFrame:
        """Parse EquityShortInterest response into a DataFrame."""
        ticker_set = {t.upper() for t in tickers}
        rows = []
        for rec in records:
            sym = rec.get("issueSymbolIdentifier", "")
            if sym.upper() not in ticker_set:
                continue
            rows.append(
                {
                    "ticker": sym.upper(),
                    "date": rec.get("settlementDate", ""),
                    "short_interest": rec.get("currentShortShareNumber"),
                    "prev_short_interest": rec.get("previousShortShareNumber"),
                    "avg_daily_volume": rec.get("averageShortShareNumber"),
                    "days_to_cover": rec.get("daysToCoverNumber"),
                    "change_pct": rec.get("changePercent"),
                }
            )

        if not rows:
            return self._empty_otc_df()

        df = pl.from_dicts(rows)
        try:
            df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        except Exception:
            pass
        return df.sort(["ticker", "date"], descending=[False, True])

    # ------------------------------------------------------------------
    # Empty DataFrames
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "short_volume": pl.Int64,
                "total_volume": pl.Int64,
                "short_exempt_volume": pl.Int64,
                "short_ratio": pl.Float64,
            }
        )

    @staticmethod
    def _empty_otc_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "short_interest": pl.Int64,
                "prev_short_interest": pl.Int64,
                "avg_daily_volume": pl.Int64,
                "days_to_cover": pl.Float64,
                "change_pct": pl.Float64,
            }
        )
