"""CoinGecko crypto price data collector."""

import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import polars as pl

logger = logging.getLogger(__name__)


class CryptoCollector:
    """Collect BTC/ETH prices from CoinGecko free API."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    COIN_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
    }

    # Rate limit: ~10-30 calls/min on free tier
    _REQUEST_INTERVAL = 3.0  # seconds between requests

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._last_request_time: float = 0.0

    def collect_prices(
        self,
        coins: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Fetch crypto OHLCV prices.

        Args:
            coins: List of coin tickers like ["BTC", "ETH"]. Default: both.
            start_date: Start date (default: 30 days ago).
            end_date: End date (default: today).

        Returns DataFrame with columns:
        [ticker, date, open, high, low, close, volume, adj_close]
        (Same schema as stock prices for compatibility)
        """
        if coins is None:
            coins = list(self.COIN_MAP.keys())

        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = date(end_date.year, end_date.month, 1)

        frames: list[pl.DataFrame] = []

        # CoinGecko free tier limits to ~365 days per request
        # Chunk the date range into 360-day segments
        max_days = 360
        date_chunks: list[tuple[date, date]] = []
        chunk_start = start_date
        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=max_days), end_date)
            date_chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)

        with httpx.Client(timeout=self.timeout) as client:
            for coin_ticker in coins:
                coin_id = self.COIN_MAP.get(coin_ticker.upper())
                if not coin_id:
                    logger.warning("Unknown coin ticker: %s", coin_ticker)
                    continue

                for chunk_s, chunk_e in date_chunks:
                    from_ts = int(datetime.combine(chunk_s, datetime.min.time()).timestamp())
                    to_ts = int(datetime.combine(chunk_e, datetime.max.time()).timestamp())
                    try:
                        df = self._fetch_coin(client, coin_ticker.upper(), coin_id, from_ts, to_ts)
                        if df is not None and len(df) > 0:
                            frames.append(df)
                    except Exception as exc:
                        logger.warning("Failed to collect prices for %s chunk %s-%s: %s",
                                       coin_ticker, chunk_s, chunk_e, exc)
                        continue

        if not frames:
            return self._empty_df()

        result = pl.concat(frames, how="diagonal")

        # Filter to requested date range
        result = result.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        return result

    def _fetch_coin(
        self,
        client: httpx.Client,
        ticker: str,
        coin_id: str,
        from_ts: int,
        to_ts: int,
    ) -> pl.DataFrame | None:
        """Fetch price data for a single coin."""
        self._rate_limit()

        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": str(from_ts),
            "to": str(to_ts),
        }

        try:
            resp = client.get(url, params=params)
            if resp.status_code == 429:
                logger.warning("CoinGecko rate limit hit for %s, skipping", ticker)
                return None
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("CoinGecko HTTP error for %s: %s", ticker, exc.response.status_code)
            return None
        except httpx.RequestError as exc:
            logger.warning("CoinGecko request error for %s: %s", ticker, exc)
            return None

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            return None

        # Build volume lookup by date
        volume_map: dict[date, float] = {}
        for ts_ms, vol in volumes:
            d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
            volume_map[d] = vol

        rows: list[dict] = []
        for ts_ms, price in prices:
            d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
            vol = volume_map.get(d, 0.0)
            rows.append({
                "ticker": ticker,
                "date": d,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": int(vol),
                "adj_close": price,
            })

        if not rows:
            return None

        df = pl.from_dicts(rows)
        # Deduplicate by date (CoinGecko may return multiple points per day)
        df = df.unique(subset=["ticker", "date"], keep="last")
        df = df.with_columns(pl.col("date").cast(pl.Date))
        return df

    def _rate_limit(self) -> None:
        """Enforce rate limit between requests."""
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
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
                "adj_close": pl.Float64,
            }
        )
