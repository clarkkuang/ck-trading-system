"""npm + PyPI weekly download collector for the AI model share monitor.

npm:  GET https://api.npmjs.org/downloads/point/{start}:{end}/{package}
      Exact ISO-week windows (Mon..Sun). Falls back to /point/last-week on
      failure, recording the actual window the API returned. Scoped package
      names (@scope/name) keep their slash.

PyPI: GET https://pypistats.org/api/packages/{package}/recent
      Rolling last-7-days count (the API offers no custom window). Labeled
      with the previous complete ISO week relative to `as_of`; the rolling
      window is a documented approximation. pypistats rate limits hard —
      mandatory User-Agent, 6s sleep between packages, one retry on 429.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from datetime import date, timedelta

import httpx
import polars as pl

logger = logging.getLogger(__name__)

_NPM_POINT_URL = "https://api.npmjs.org/downloads/point/{window}/{package}"
_PYPI_RECENT_URL = "https://pypistats.org/api/packages/{package}/recent"
_USER_AGENT = "ck-trading-system/0.1 (weekly AI model share monitor)"


def previous_complete_iso_week(as_of: date) -> tuple[date, date]:
    """(Monday, Sunday) of the last fully-elapsed ISO week before as_of."""
    this_monday = as_of - timedelta(days=as_of.weekday())
    last_monday = this_monday - timedelta(days=7)
    return last_monday, last_monday + timedelta(days=6)


def iso_week_label(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class PackageDownloadsCollector:
    def __init__(self, timeout: float = 30.0, pypi_sleep_s: float = 12.0):
        self.timeout = timeout
        self.pypi_sleep_s = pypi_sleep_s

    # ------------------------------------------------------------------
    def collect_npm(self, packages: list[str], as_of: date | None = None) -> pl.DataFrame:
        """Weekly npm downloads for the previous complete ISO week."""
        as_of = as_of or date.today()
        monday, sunday = previous_complete_iso_week(as_of)
        window = f"{monday.isoformat()}:{sunday.isoformat()}"
        now = _utcnow()

        rows: list[dict] = []
        with httpx.Client(
            timeout=self.timeout, headers={"User-Agent": _USER_AGENT}
        ) as client:
            for pkg in packages:
                row = self._npm_one(client, pkg, window, monday, sunday)
                if row is not None:
                    row["collected_at"] = now
                    rows.append(row)

        if not rows:
            return self._empty_df()
        return pl.DataFrame(rows, schema=self._schema())

    def _npm_one(
        self, client: httpx.Client, pkg: str, window: str,
        monday: date, sunday: date,
    ) -> dict | None:
        for attempt_window in (window, "last-week"):
            url = _NPM_POINT_URL.format(window=attempt_window, package=pkg)
            try:
                resp = client.get(url)
                if resp.status_code != 200:
                    logger.warning(
                        "npm %s [%s]: HTTP %s", pkg, attempt_window, resp.status_code
                    )
                    continue
                data = resp.json()
                downloads = data.get("downloads")
                if downloads is None:
                    continue
                start = _safe_date(data.get("start"), monday)
                end = _safe_date(data.get("end"), sunday)
                return {
                    "registry": "npm",
                    "package": pkg,
                    "period_start": start,
                    "period_end": end,
                    "downloads": int(downloads),
                    "iso_week": iso_week_label(end),
                }
            except Exception as e:  # noqa: BLE001
                logger.warning("npm %s [%s] failed: %r", pkg, attempt_window, e)
        return None

    # ------------------------------------------------------------------
    def collect_pypi(self, packages: list[str], as_of: date | None = None) -> pl.DataFrame:
        """Rolling last-week PyPI downloads, labeled with the previous ISO week."""
        as_of = as_of or date.today()
        monday, sunday = previous_complete_iso_week(as_of)
        now = _utcnow()

        rows: list[dict] = []
        with httpx.Client(
            timeout=self.timeout, headers={"User-Agent": _USER_AGENT}
        ) as client:
            for i, pkg in enumerate(packages):
                if i > 0:
                    time.sleep(self.pypi_sleep_s)
                downloads = self._pypi_one(client, pkg)
                if downloads is None:
                    continue
                rows.append({
                    "registry": "pypi",
                    "package": pkg,
                    "period_start": as_of - timedelta(days=7),
                    "period_end": as_of,
                    "downloads": downloads,
                    "iso_week": iso_week_label(sunday),
                    "collected_at": now,
                })

        if not rows:
            return self._empty_df()
        return pl.DataFrame(rows, schema=self._schema())

    def _pypi_one(self, client: httpx.Client, pkg: str) -> int | None:
        url = _PYPI_RECENT_URL.format(package=pkg)
        for attempt in range(2):
            try:
                resp = client.get(url)
                if resp.status_code == 429:
                    if attempt == 0:
                        logger.info("pypistats 429 for %s — waiting 60s", pkg)
                        time.sleep(60)
                        continue
                    logger.warning("pypistats %s: still rate-limited", pkg)
                    return None
                if resp.status_code != 200:
                    logger.warning("pypistats %s: HTTP %s", pkg, resp.status_code)
                    return None
                data = resp.json().get("data") or {}
                lw = data.get("last_week")
                return int(lw) if lw is not None else None
            except Exception as e:  # noqa: BLE001
                logger.warning("pypistats %s failed: %r", pkg, e)
                return None
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _schema() -> dict:
        return {
            "registry": pl.Utf8,
            "package": pl.Utf8,
            "period_start": pl.Date,
            "period_end": pl.Date,
            "downloads": pl.Int64,
            "iso_week": pl.Utf8,
            "collected_at": pl.Datetime("us"),
        }

    @classmethod
    def _empty_df(cls) -> pl.DataFrame:
        return pl.DataFrame(schema=cls._schema())


def _safe_date(raw, fallback: date) -> date:
    try:
        return date.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return fallback
