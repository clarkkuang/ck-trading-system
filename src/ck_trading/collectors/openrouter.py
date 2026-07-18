"""OpenRouter data collector for the AI model share monitor.

Two data sets:

* Pricing snapshots — GET /api/v1/models (public, no auth). Per-token USD
  prices converted to per-Mtok floats.
* Daily token rankings — GET /api/v1/datasets/rankings-daily (needs any
  valid OpenRouter API key; rate limit 30 req/min, 500 req/day). Top-50
  models per day by prompt+completion tokens, plus one aggregated "other"
  row per day.

Category filtering ("programming") support on the datasets endpoint is
unconfirmed, so every rankings row records a `scope`:

    "programming"             response confirmed the category filter
    "programming:unverified"  we sent category=programming, got 200, but the
                              response did not echo the filter
    "all"                     no category filter applied

Citation requirement when republishing this data:
"Source: OpenRouter (openrouter.ai/rankings), as of {as_of}."

Known feed regression (2026-07): around 2026-07-07 the endpoint silently
switched from per-day rows to a trailing-~30-day window of WEEKLY buckets —
Monday-dated rows whose tokens are whole-week sums (~4x a normal day),
returned regardless of the requested ``date``. ``_fetch_day`` therefore
enforces the daily contract: any response row dated differently from the
requested day raises RankingsSchemaError instead of ingesting. The pipeline
additionally quarantines days whose token volume is anomalous vs stored
history (metrics.split_anomalous_days).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from datetime import date

import httpx
import polars as pl

from ck_trading.monitoring.blocs import OTHER_MODEL_ID, classify_model_id, org_of

logger = logging.getLogger(__name__)

_MODELS_URL = "https://openrouter.ai/api/v1/models"
_RANKINGS_URL = "https://openrouter.ai/api/v1/datasets/rankings-daily"

# Candidate field names in rankings rows — the datasets API is young and the
# docs don't pin an example payload; accept the plausible spellings and
# validate hard afterwards.
_SLUG_FIELDS = ("model_permaslug", "model_slug", "model", "permaslug", "slug", "id")
_TOKEN_FIELDS = ("total_tokens", "tokens", "total", "count", "token_count")
_DATE_FIELDS = ("date", "day", "d")


class RankingsSchemaError(RuntimeError):
    """The rankings response didn't match any known shape."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class OpenRouterCollector:
    def __init__(
        self,
        api_key: str = "",
        timeout: float = 30.0,
        rate_limit_sleep_s: float = 2.5,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit_sleep_s = rate_limit_sleep_s

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    def collect_pricing(self, snapshot_date: date | None = None) -> pl.DataFrame:
        """Snapshot every model's list price. Public endpoint, no auth."""
        snapshot_date = snapshot_date or date.today()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(_MODELS_URL, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("OpenRouter pricing fetch failed: %r", e)
            return self._empty_pricing_df()

        models = payload.get("data")
        if not isinstance(models, list):
            logger.warning("OpenRouter /models: unexpected payload shape")
            return self._empty_pricing_df()

        warned: set[str] = set()
        now = _utcnow()
        rows: list[dict] = []
        for m in models:
            model_id = m.get("id")
            if not model_id:
                continue
            pricing = m.get("pricing") or {}
            prompt = _parse_price(pricing.get("prompt"))
            completion = _parse_price(pricing.get("completion"))
            created = m.get("created")
            created_at = (
                dt.datetime.fromtimestamp(created, dt.timezone.utc)
                .replace(tzinfo=None)
                if isinstance(created, (int, float)) and created > 0
                else None
            )
            rows.append({
                "snapshot_date": snapshot_date,
                "model_id": model_id,
                "org": org_of(model_id),
                "bloc": classify_model_id(model_id, warned_orgs=warned),
                "prompt_usd_per_mtok": prompt,
                "completion_usd_per_mtok": completion,
                "context_length": m.get("context_length"),
                "created_at": created_at,
                "collected_at": now,
            })

        if not rows:
            return self._empty_pricing_df()
        return pl.DataFrame(rows, schema=self._pricing_schema())

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------
    def collect_rankings(
        self,
        start_date: date,
        end_date: date,
        category: str | None = "programming",
    ) -> pl.DataFrame:
        """Daily top-50 token rankings for [start_date, end_date].

        Raises RankingsSchemaError if the response shape is unrecognized
        (section-level failure the pipeline records without killing the run).
        Returns an empty frame if no API key is configured.
        """
        if not self.api_key:
            logger.info("No OPENROUTER_API_KEY — skipping rankings collection")
            return self._empty_rankings_df()

        headers = {"Authorization": f"Bearer {self.api_key}"}
        warned: set[str] = set()
        frames: list[pl.DataFrame] = []

        with httpx.Client(timeout=self.timeout) as client:
            day = start_date
            first = True
            while day <= end_date:
                if not first:
                    time.sleep(self.rate_limit_sleep_s)
                first = False
                df = self._fetch_day(client, headers, day, category, warned)
                if not df.is_empty():
                    frames.append(df)
                day += dt.timedelta(days=1)

        if not frames:
            return self._empty_rankings_df()
        return pl.concat(frames)

    def _fetch_day(
        self,
        client: httpx.Client,
        headers: dict,
        day: date,
        category: str | None,
        warned: set[str],
    ) -> pl.DataFrame:
        """One day's rankings, with category-param fallback."""
        base_params = {"date": day.isoformat()}
        attempts: list[tuple[dict, str]] = []
        if category:
            attempts.append((
                {**base_params, "category": category},
                category,  # scope if confirmed
            ))
        attempts.append((base_params, "all"))

        for params, want_scope in attempts:
            try:
                resp = client.get(_RANKINGS_URL, headers=headers, params=params)
            except Exception as e:  # noqa: BLE001
                logger.warning("Rankings fetch %s failed: %r", day, e)
                return self._empty_rankings_df()

            if resp.status_code in (400, 404, 422) and "category" in params:
                # category param rejected -> retry without it
                continue
            if resp.status_code != 200:
                raise RankingsSchemaError(
                    f"rankings-daily HTTP {resp.status_code}: {resp.text[:200]}"
                )

            payload = resp.json()
            rows = _parse_rankings(payload)
            if rows is None:
                raise RankingsSchemaError(
                    f"rankings-daily: unrecognized payload shape "
                    f"(keys={list(payload)[:5] if isinstance(payload, dict) else type(payload)})"
                )

            # Daily contract: every dated row must match the requested day.
            # The 2026-07 feed regression returns Monday-dated weekly buckets
            # for any requested date; merging those corrupts the daily table.
            foreign = sorted({str(r["date"]) for r in rows
                              if "date" in r and r["date"] != day})
            if foreign:
                raise RankingsSchemaError(
                    f"rankings-daily: requested {day} but response rows are "
                    f"dated {foreign[:5]} — looks like weekly/aggregate "
                    f"buckets, refusing to ingest"
                )

            # Determine scope: confirmed only if the response echoes the category
            if want_scope == "all":
                scope = "all"
            else:
                echoed = (
                    isinstance(payload, dict)
                    and str(payload.get("category", "")).lower() == want_scope
                )
                scope = want_scope if echoed else f"{want_scope}:unverified"

            return self._rows_to_df(rows, day, scope, warned)

        return self._empty_rankings_df()

    def _rows_to_df(
        self, rows: list[dict], day: date, scope: str, warned: set[str]
    ) -> pl.DataFrame:
        now = _utcnow()
        out: list[dict] = []
        rank = 0
        for r in rows:
            slug = r["slug"]
            tokens = r["tokens"]
            row_date = r.get("date") or day
            is_other = slug.lower() in ("other", OTHER_MODEL_ID)
            model_id = OTHER_MODEL_ID if is_other else slug
            if not is_other:
                rank += 1
            out.append({
                "date": row_date,
                "model_id": model_id,
                "org": OTHER_MODEL_ID if is_other else org_of(model_id),
                "bloc": classify_model_id(model_id, warned_orgs=warned),
                "tokens_total": int(tokens),
                "rank": None if is_other else rank,
                "scope": scope,
                "source": "api",
                "collected_at": now,
            })
        if not out:
            return self._empty_rankings_df()
        return pl.DataFrame(out, schema=self._rankings_schema())

    # ------------------------------------------------------------------
    # Schemas
    # ------------------------------------------------------------------
    @staticmethod
    def _pricing_schema() -> dict:
        return {
            "snapshot_date": pl.Date,
            "model_id": pl.Utf8,
            "org": pl.Utf8,
            "bloc": pl.Utf8,
            "prompt_usd_per_mtok": pl.Float64,
            "completion_usd_per_mtok": pl.Float64,
            "context_length": pl.Int64,
            "created_at": pl.Datetime("us"),
            "collected_at": pl.Datetime("us"),
        }

    @classmethod
    def _empty_pricing_df(cls) -> pl.DataFrame:
        return pl.DataFrame(schema=cls._pricing_schema())

    @staticmethod
    def _rankings_schema() -> dict:
        return {
            "date": pl.Date,
            "model_id": pl.Utf8,
            "org": pl.Utf8,
            "bloc": pl.Utf8,
            "tokens_total": pl.Int64,
            "rank": pl.Int32,
            "scope": pl.Utf8,
            "source": pl.Utf8,
            "collected_at": pl.Datetime("us"),
        }

    @classmethod
    def _empty_rankings_df(cls) -> pl.DataFrame:
        return pl.DataFrame(schema=cls._rankings_schema())


def _parse_price(raw) -> float | None:
    """OpenRouter price string (USD per token) -> USD per Mtok, or None."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return v * 1_000_000


def _parse_rankings(payload) -> list[dict] | None:
    """Normalize a rankings payload into [{slug, tokens, date?}, ...].

    Returns None if the shape is unrecognized. Parsing is deliberately
    concentrated here so an API change is a one-function fix.
    """
    if isinstance(payload, dict):
        for key in ("data", "rows", "results", "rankings"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            return None
    if not isinstance(payload, list):
        return None

    out: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        slug = _first_field(item, _SLUG_FIELDS)
        tokens = _first_field(item, _TOKEN_FIELDS)
        if slug is None or tokens is None:
            return None
        try:
            tokens_val = int(float(tokens))
        except (TypeError, ValueError):
            return None
        row: dict = {"slug": str(slug), "tokens": tokens_val}
        raw_date = _first_field(item, _DATE_FIELDS)
        if raw_date:
            try:
                row["date"] = date.fromisoformat(str(raw_date)[:10])
            except ValueError:
                pass
        out.append(row)
    return out


def _first_field(d: dict, names: tuple[str, ...]):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None
