"""Metric computation for the AI model share monitor.

Pure polars functions, no I/O. The central metric is the weekly
**dollar-weighted bloc share** on OpenRouter:

    dollar_volume(model, day) = tokens_total / 1e6 * blended_usd_per_mtok
    blended = 0.70 * prompt_price + 0.30 * completion_price

Why 70/30: rankings report ONE combined (prompt+completion) token count per
model per day while pricing is split. On real API traffic prompt tokens
dominate (chat history accumulation, RAG context, coding agents shipping
whole repos); public OpenRouter aggregates have historically shown ~70-85%
prompt share. 70/30 is a round, defensible midpoint. Absolute dollars don't
matter — the metric is a SHARE — but the convention must stay frozen so the
series is comparable over time. Changing it requires a documented full
rebuild of weekly_bloc_share (which is fully derivable from the raw tables).
"""

from __future__ import annotations

import datetime as dt
from typing import Final

import polars as pl

from ck_trading.monitoring.blocs import (
    OTHER_MODEL_ID,
    canonical_family_key,
    org_of,
    strip_variant,
)

# ---- frozen conventions (change => full series rebuild) --------------------
PROMPT_TOKEN_FRACTION: Final[float] = 0.70
COMPLETION_TOKEN_FRACTION: Final[float] = 0.30
MIN_DAYS_PER_WEEK: Final[int] = 5  # a week needs >=5 observed days to count


def blended_usd_per_mtok(prompt: pl.Expr, completion: pl.Expr) -> pl.Expr:
    """Blended list price per Mtok under the frozen 70/30 convention."""
    return PROMPT_TOKEN_FRACTION * prompt + COMPLETION_TOKEN_FRACTION * completion


def _with_iso_week(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col(date_col).dt.iso_year().cast(pl.Utf8)
            + pl.lit("-W")
            + pl.col(date_col).dt.week().cast(pl.Utf8).str.zfill(2)
        ).alias("iso_week"),
        # Monday of the ISO week
        (pl.col(date_col) - pl.duration(days=pl.col(date_col).dt.weekday() - 1))
        .alias("week_start"),
    )


def _attach_blended_price(
    daily_tokens: pl.DataFrame, pricing_snapshots: pl.DataFrame
) -> pl.DataFrame:
    """Assign each daily-token row a blended list price via a 3-tier cascade.

    OpenRouter's rankings permaslug (``claude-4.7-opus-20260416``) and its
    /models id (``claude-opus-4.7``) don't match directly, and many ranked
    SKUs have no exact listed price. To keep the dollar-share metric from
    zeroing out whole blocs we cascade:

        1. exact base model_id (``:variant`` stripped)
        2. canonical family key (order-insensitive, date-stamp stripped) —
           median blended price across matching priced models
        3. org median blended price

    Adds columns ``blended`` (USD/Mtok, null only if the org has no priced
    model at all) and ``price_match`` in {exact, family, org, none}. Uses the
    latest price per model — the dollar SHARE is insensitive to slow price
    drift, and price-CUT detection uses flagship_price_series() separately.
    """
    # Derive matching keys from model_id directly (don't trust upstream cols).
    tokens = daily_tokens.with_columns(
        pl.col("model_id").map_elements(strip_variant, return_dtype=pl.Utf8)
        .alias("base_model_id"),
        pl.col("model_id").map_elements(canonical_family_key, return_dtype=pl.Utf8)
        .alias("fam_key"),
        pl.col("model_id").map_elements(org_of, return_dtype=pl.Utf8)
        .alias("_org_key"),
    )

    if pricing_snapshots.is_empty():
        return tokens.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("blended"),
            pl.lit("none").alias("price_match"),
        )

    prices = pricing_snapshots.with_columns(
        pl.col("model_id").map_elements(strip_variant, return_dtype=pl.Utf8)
        .alias("base_model_id"),
        pl.col("model_id").map_elements(canonical_family_key, return_dtype=pl.Utf8)
        .alias("fam_key"),
        pl.col("model_id").map_elements(org_of, return_dtype=pl.Utf8).alias("org2"),
        blended_usd_per_mtok(
            pl.col("prompt_usd_per_mtok"), pl.col("completion_usd_per_mtok")
        ).alias("blended"),
    ).drop_nulls("blended")

    # Three price lookups as small frames, joined (order-safe) onto tokens.
    exact_df = (
        prices.sort("snapshot_date")
        .unique(subset=["base_model_id"], keep="last")
        .select("base_model_id", pl.col("blended").alias("_p_exact"))
    )
    fam_df = (
        prices.group_by("fam_key")
        .agg(pl.col("blended").median().alias("_p_fam"))
    )
    org_df = (
        prices.group_by("org2")
        .agg(pl.col("blended").median().alias("_p_org"))
        .rename({"org2": "_org_key"})
    )

    joined = (
        tokens.join(exact_df, on="base_model_id", how="left")
        .join(fam_df, on="fam_key", how="left")
        .join(org_df, on="_org_key", how="left")
    )
    return joined.with_columns(
        pl.coalesce("_p_exact", "_p_fam", "_p_org").alias("blended"),
        pl.when(pl.col("_p_exact").is_not_null()).then(pl.lit("exact"))
        .when(pl.col("_p_fam").is_not_null()).then(pl.lit("family"))
        .when(pl.col("_p_org").is_not_null()).then(pl.lit("org"))
        .otherwise(pl.lit("none")).alias("price_match"),
    ).drop("_p_exact", "_p_fam", "_p_org")


def compute_weekly_bloc_share(
    daily_tokens: pl.DataFrame,
    pricing_snapshots: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate daily token rankings into weekly bloc dollar/token shares.

    Args:
        daily_tokens: openrouter_daily_tokens schema
            [date, model_id, org, bloc, tokens_total, rank, scope, source,
             collected_at]
        pricing_snapshots: openrouter_pricing_snapshots schema
            [snapshot_date, model_id, org, bloc, prompt_usd_per_mtok,
             completion_usd_per_mtok, context_length, created_at, collected_at]

    Returns:
        weekly_bloc_share schema:
            [iso_week, week_start, scope, bloc, tokens_total,
             dollar_volume_usd, token_share, dollar_share, days_observed,
             is_complete, computed_at]
    """
    empty = pl.DataFrame(
        schema={
            "iso_week": pl.Utf8,
            "week_start": pl.Date,
            "scope": pl.Utf8,
            "bloc": pl.Utf8,
            "tokens_total": pl.Int64,
            "dollar_volume_usd": pl.Float64,
            "token_share": pl.Float64,
            "dollar_share": pl.Float64,
            "days_observed": pl.Int8,
            "is_complete": pl.Boolean,
            "computed_at": pl.Datetime("us"),
        }
    )
    if daily_tokens.is_empty():
        return empty

    priced = _attach_blended_price(daily_tokens, pricing_snapshots)
    priced = priced.with_columns(
        (pl.col("tokens_total") / 1e6 * pl.col("blended")).alias("dollar_volume_usd"),
        (pl.col("model_id") == OTHER_MODEL_ID).alias("is_other"),
    )

    weekly = (
        _with_iso_week(priced)
        .group_by(["iso_week", "week_start", "scope", "bloc"])
        .agg(
            pl.col("tokens_total").sum(),
            pl.col("dollar_volume_usd").sum().alias("dollar_volume_usd"),
            pl.col("date").n_unique().cast(pl.Int8).alias("days_observed"),
        )
    )

    # days_observed should be per (week, scope), not per bloc — a bloc missing
    # on some days still belongs to a complete week. Recompute from the source.
    week_days = (
        _with_iso_week(priced)
        .group_by(["iso_week", "scope"])
        .agg(pl.col("date").n_unique().cast(pl.Int8).alias("days_observed"))
    )
    weekly = weekly.drop("days_observed").join(
        week_days, on=["iso_week", "scope"], how="left"
    )

    # ---- shares -------------------------------------------------------------
    # token denominator: everything (incl. other + unpriced)
    # dollar denominator: priced, non-other rows only
    weekly = weekly.with_columns(
        pl.col("tokens_total").sum().over(["iso_week", "scope"]).alias("_tok_den"),
        pl.when(pl.col("bloc") != "other")
        .then(pl.col("dollar_volume_usd"))
        .otherwise(None)
        .sum()
        .over(["iso_week", "scope"])
        .alias("_usd_den"),
    ).with_columns(
        (pl.col("tokens_total") / pl.col("_tok_den")).alias("token_share"),
        pl.when(pl.col("bloc") != "other")
        .then(pl.col("dollar_volume_usd") / pl.col("_usd_den"))
        .otherwise(None)
        .alias("dollar_share"),
        (pl.col("days_observed") >= MIN_DAYS_PER_WEEK).alias("is_complete"),
    ).drop("_tok_den", "_usd_den")

    computed_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    weekly = weekly.with_columns(
        pl.lit(computed_at).cast(pl.Datetime("us")).alias("computed_at")
    )

    return weekly.select(
        "iso_week", "week_start", "scope", "bloc", "tokens_total",
        "dollar_volume_usd", "token_share", "dollar_share",
        "days_observed", "is_complete", "computed_at",
    ).sort(["week_start", "scope", "bloc"])


def flagship_price_series(
    pricing_snapshots: pl.DataFrame, family: str
) -> pl.DataFrame:
    """Per snapshot_date, the completion price of the CURRENT flagship of `family`.

    Family match: base model_id starts with the family prefix. Flagship at
    snapshot t = matching model with the max created_at present in snapshot t.
    When a cheaper successor replaces the flagship, the series switches — a
    cheaper successor IS an effective price cut, by design.

    Returns [period_key, period_start, value, model_id] sorted ascending.
    """
    empty = pl.DataFrame(
        schema={
            "period_key": pl.Utf8,
            "period_start": pl.Date,
            "value": pl.Float64,
            "model_id": pl.Utf8,
        }
    )
    if pricing_snapshots.is_empty():
        return empty

    fam = pricing_snapshots.with_columns(
        pl.col("model_id").map_elements(strip_variant, return_dtype=pl.Utf8)
        .alias("base_id")
    ).filter(
        pl.col("base_id").str.starts_with(family)
        & pl.col("completion_usd_per_mtok").is_not_null()
    )
    if fam.is_empty():
        return empty

    flagship = (
        fam.sort("created_at", descending=True)
        .group_by("snapshot_date")
        .first()
        .sort("snapshot_date")
    )
    return flagship.select(
        pl.col("snapshot_date").cast(pl.Utf8).alias("period_key"),
        pl.col("snapshot_date").alias("period_start"),
        pl.col("completion_usd_per_mtok").alias("value"),
        pl.col("model_id"),
    )


def pkg_weekly_series(
    pkg_downloads: pl.DataFrame, registry: str, package: str
) -> pl.DataFrame:
    """Weekly download series for one package.

    Returns [period_key, period_start, value] sorted ascending.
    """
    empty = pl.DataFrame(
        schema={
            "period_key": pl.Utf8,
            "period_start": pl.Date,
            "value": pl.Float64,
        }
    )
    if pkg_downloads.is_empty():
        return empty
    sub = pkg_downloads.filter(
        (pl.col("registry") == registry) & (pl.col("package") == package)
    )
    if sub.is_empty():
        return empty
    return (
        sub.select(
            pl.col("iso_week").alias("period_key"),
            pl.col("period_start"),
            pl.col("downloads").cast(pl.Float64).alias("value"),
        )
        .unique(subset=["period_key"], keep="last")
        .sort("period_start")
    )


def bloc_share_series(
    weekly_bloc_share: pl.DataFrame,
    bloc: str,
    scope: str,
    metric: str = "dollar_share",
) -> pl.DataFrame:
    """Weekly share series for one (bloc, scope), complete weeks only.

    Returns [period_key, period_start, value] sorted ascending.
    """
    empty = pl.DataFrame(
        schema={
            "period_key": pl.Utf8,
            "period_start": pl.Date,
            "value": pl.Float64,
        }
    )
    if weekly_bloc_share.is_empty():
        return empty
    sub = weekly_bloc_share.filter(
        (pl.col("bloc") == bloc)
        & (pl.col("scope") == scope)
        & pl.col("is_complete")
        & pl.col(metric).is_not_null()
    )
    if sub.is_empty():
        return empty
    return sub.select(
        pl.col("iso_week").alias("period_key"),
        pl.col("week_start").alias("period_start"),
        pl.col(metric).alias("value"),
    ).sort("period_start")
