"""Tests for dollar-weighted share computation and derived series."""

import datetime as dt
from datetime import date

import polars as pl
import pytest

from ck_trading.monitoring.metrics import (
    COMPLETION_TOKEN_FRACTION,
    MIN_DAYS_PER_WEEK,
    PROMPT_TOKEN_FRACTION,
    bloc_share_series,
    compute_weekly_bloc_share,
    flagship_price_series,
    pkg_weekly_series,
)

NOW = dt.datetime(2026, 7, 1, 0, 0, 0)


def _tokens_df(rows: list[dict]) -> pl.DataFrame:
    base = {
        "org": "x", "rank": 1, "scope": "all",
        "source": "api", "collected_at": NOW,
    }
    return pl.DataFrame(
        [{**base, **r} for r in rows],
        schema={
            "date": pl.Date, "model_id": pl.Utf8, "org": pl.Utf8,
            "bloc": pl.Utf8, "tokens_total": pl.Int64, "rank": pl.Int32,
            "scope": pl.Utf8, "source": pl.Utf8, "collected_at": pl.Datetime("us"),
        },
    )


def _pricing_df(rows: list[dict]) -> pl.DataFrame:
    base = {
        "org": "x", "bloc": "chinese", "context_length": 128000,
        "created_at": NOW, "collected_at": NOW,
    }
    return pl.DataFrame(
        [{**base, **r} for r in rows],
        schema={
            "snapshot_date": pl.Date, "model_id": pl.Utf8, "org": pl.Utf8,
            "bloc": pl.Utf8, "prompt_usd_per_mtok": pl.Float64,
            "completion_usd_per_mtok": pl.Float64, "context_length": pl.Int64,
            "created_at": pl.Datetime("us"), "collected_at": pl.Datetime("us"),
        },
    )


# 2026-06-01 is a Monday
WEEK1 = [date(2026, 6, 1) + dt.timedelta(days=i) for i in range(5)]


def _full_week_tokens(model_id: str, bloc: str, daily_tokens: int) -> list[dict]:
    return [
        {"date": d, "model_id": model_id, "bloc": bloc, "tokens_total": daily_tokens}
        for d in WEEK1
    ]


class TestDollarShare:
    def test_hand_computed_two_model_share(self):
        """CN model: 100M tok/day @ blended $1; US model: 50M tok/day @ blended $10."""
        tokens = _tokens_df(
            _full_week_tokens("deepseek/d1", "chinese", 100_000_000)
            + _full_week_tokens("anthropic/claude-x", "anthropic", 50_000_000)
        )
        pricing = _pricing_df([
            # blended = 0.7*prompt + 0.3*completion
            {"snapshot_date": date(2026, 6, 1), "model_id": "deepseek/d1",
             "prompt_usd_per_mtok": 1.0, "completion_usd_per_mtok": 1.0},
            {"snapshot_date": date(2026, 6, 1), "model_id": "anthropic/claude-x",
             "prompt_usd_per_mtok": 10.0, "completion_usd_per_mtok": 10.0},
        ])
        out = compute_weekly_bloc_share(tokens, pricing)
        assert out.height == 2

        cn = out.filter(pl.col("bloc") == "chinese").row(0, named=True)
        an = out.filter(pl.col("bloc") == "anthropic").row(0, named=True)

        # dollars/week: CN = 5 days * 100 Mtok * $1 = $500; AN = 5*50*$10 = $2500
        assert cn["dollar_volume_usd"] == pytest.approx(500.0)
        assert an["dollar_volume_usd"] == pytest.approx(2500.0)
        # dollar shares
        assert cn["dollar_share"] == pytest.approx(500 / 3000)
        assert an["dollar_share"] == pytest.approx(2500 / 3000)
        # token shares: CN 2/3, AN 1/3
        assert cn["token_share"] == pytest.approx(2 / 3)
        assert an["token_share"] == pytest.approx(1 / 3)
        assert cn["is_complete"] and an["is_complete"]

    def test_blend_ratio_applied(self):
        """Asymmetric prompt/completion price exercises the 70/30 blend."""
        tokens = _tokens_df(_full_week_tokens("openai/gpt-x", "western_closed", 1_000_000))
        pricing = _pricing_df([
            {"snapshot_date": date(2026, 6, 1), "model_id": "openai/gpt-x",
             "prompt_usd_per_mtok": 2.0, "completion_usd_per_mtok": 10.0},
        ])
        out = compute_weekly_bloc_share(tokens, pricing)
        row = out.row(0, named=True)
        blended = PROMPT_TOKEN_FRACTION * 2.0 + COMPLETION_TOKEN_FRACTION * 10.0  # 4.4
        assert row["dollar_volume_usd"] == pytest.approx(5 * 1.0 * blended)

    def test_other_row_excluded_from_dollar_share(self):
        tokens = _tokens_df(
            _full_week_tokens("deepseek/d1", "chinese", 100)
            + _full_week_tokens("__other__", "other", 900)
        )
        pricing = _pricing_df([
            {"snapshot_date": date(2026, 6, 1), "model_id": "deepseek/d1",
             "prompt_usd_per_mtok": 1.0, "completion_usd_per_mtok": 1.0},
        ])
        out = compute_weekly_bloc_share(tokens, pricing)
        cn = out.filter(pl.col("bloc") == "chinese").row(0, named=True)
        other = out.filter(pl.col("bloc") == "other").row(0, named=True)
        # dollar share computed only over priced non-other rows -> CN = 100%
        assert cn["dollar_share"] == pytest.approx(1.0)
        assert other["dollar_share"] is None
        # token share includes other: CN 10%, other 90%
        assert cn["token_share"] == pytest.approx(0.1)
        assert other["token_share"] == pytest.approx(0.9)

    def test_unpriced_model_excluded_from_dollar(self):
        tokens = _tokens_df(
            _full_week_tokens("deepseek/d1", "chinese", 100)
            + _full_week_tokens("ghost/unknown", "unclassified", 100)
        )
        pricing = _pricing_df([
            {"snapshot_date": date(2026, 6, 1), "model_id": "deepseek/d1",
             "prompt_usd_per_mtok": 1.0, "completion_usd_per_mtok": 1.0},
        ])
        out = compute_weekly_bloc_share(tokens, pricing)
        cn = out.filter(pl.col("bloc") == "chinese").row(0, named=True)
        # ghost has no price ever -> its dollar_volume is null; CN takes 100%
        assert cn["dollar_share"] == pytest.approx(1.0)

    def test_price_asof_join_uses_earlier_snapshot(self):
        """Mid-week price change: days before the change use the old price."""
        tokens = _tokens_df(_full_week_tokens("openai/gpt-x", "western_closed", 1_000_000))
        pricing = _pricing_df([
            {"snapshot_date": date(2026, 6, 1), "model_id": "openai/gpt-x",
             "prompt_usd_per_mtok": 10.0, "completion_usd_per_mtok": 10.0},
            {"snapshot_date": date(2026, 6, 4), "model_id": "openai/gpt-x",
             "prompt_usd_per_mtok": 5.0, "completion_usd_per_mtok": 5.0},
        ])
        out = compute_weekly_bloc_share(tokens, pricing)
        row = out.row(0, named=True)
        # Jun 1-3 @ $10 (3 days) + Jun 4-5 @ $5 (2 days), 1 Mtok/day
        assert row["dollar_volume_usd"] == pytest.approx(3 * 10 + 2 * 5)

    def test_incomplete_week_flagged(self):
        # only 4 observed days < MIN_DAYS_PER_WEEK
        tokens = _tokens_df([
            {"date": d, "model_id": "deepseek/d1", "bloc": "chinese",
             "tokens_total": 100}
            for d in WEEK1[:4]
        ])
        pricing = _pricing_df([
            {"snapshot_date": date(2026, 6, 1), "model_id": "deepseek/d1",
             "prompt_usd_per_mtok": 1.0, "completion_usd_per_mtok": 1.0},
        ])
        out = compute_weekly_bloc_share(tokens, pricing)
        assert MIN_DAYS_PER_WEEK == 5
        assert not out.row(0, named=True)["is_complete"]

    def test_empty_inputs(self):
        out = compute_weekly_bloc_share(
            pl.DataFrame(schema={
                "date": pl.Date, "model_id": pl.Utf8, "org": pl.Utf8,
                "bloc": pl.Utf8, "tokens_total": pl.Int64, "rank": pl.Int32,
                "scope": pl.Utf8, "source": pl.Utf8,
                "collected_at": pl.Datetime("us"),
            }),
            _pricing_df([]),
        )
        assert out.is_empty()


class TestFlagshipSeries:
    def test_latest_created_wins(self):
        pricing = _pricing_df([
            {"snapshot_date": date(2026, 6, 1), "model_id": "anthropic/claude-opus-4",
             "prompt_usd_per_mtok": 15.0, "completion_usd_per_mtok": 75.0,
             "created_at": dt.datetime(2025, 1, 1)},
            {"snapshot_date": date(2026, 6, 1), "model_id": "anthropic/claude-opus-5",
             "prompt_usd_per_mtok": 10.0, "completion_usd_per_mtok": 48.0,
             "created_at": dt.datetime(2026, 5, 1)},
        ])
        s = flagship_price_series(pricing, "anthropic/claude-opus")
        assert s.height == 1
        row = s.row(0, named=True)
        assert row["model_id"] == "anthropic/claude-opus-5"
        assert row["value"] == pytest.approx(48.0)

    def test_series_switches_on_churn(self):
        """New cheaper flagship appears in later snapshot -> series drops."""
        pricing = _pricing_df([
            {"snapshot_date": date(2026, 5, 1), "model_id": "anthropic/claude-opus-4",
             "prompt_usd_per_mtok": 15.0, "completion_usd_per_mtok": 75.0,
             "created_at": dt.datetime(2025, 1, 1)},
            {"snapshot_date": date(2026, 6, 1), "model_id": "anthropic/claude-opus-4",
             "prompt_usd_per_mtok": 15.0, "completion_usd_per_mtok": 75.0,
             "created_at": dt.datetime(2025, 1, 1)},
            {"snapshot_date": date(2026, 6, 1), "model_id": "anthropic/claude-opus-5",
             "prompt_usd_per_mtok": 10.0, "completion_usd_per_mtok": 48.0,
             "created_at": dt.datetime(2026, 5, 20)},
        ])
        s = flagship_price_series(pricing, "anthropic/claude-opus")
        vals = s["value"].to_list()
        assert vals == [75.0, 48.0]

    def test_family_prefix_filters(self):
        pricing = _pricing_df([
            {"snapshot_date": date(2026, 6, 1), "model_id": "anthropic/claude-sonnet-5",
             "prompt_usd_per_mtok": 3.0, "completion_usd_per_mtok": 15.0},
        ])
        assert flagship_price_series(pricing, "anthropic/claude-opus").is_empty()

    def test_empty(self):
        assert flagship_price_series(_pricing_df([]), "anthropic/claude-opus").is_empty()


class TestSeriesExtractors:
    def test_pkg_weekly_series(self):
        df = pl.DataFrame({
            "registry": ["npm", "npm", "pypi"],
            "package": ["a", "a", "a"],
            "period_start": [date(2026, 6, 1), date(2026, 6, 8), date(2026, 6, 1)],
            "period_end": [date(2026, 6, 7), date(2026, 6, 14), date(2026, 6, 7)],
            "downloads": [100, 200, 999],
            "iso_week": ["2026-W23", "2026-W24", "2026-W23"],
            "collected_at": [NOW] * 3,
        })
        s = pkg_weekly_series(df, "npm", "a")
        assert s["value"].to_list() == [100.0, 200.0]

    def test_bloc_share_series_filters_incomplete(self):
        df = pl.DataFrame({
            "iso_week": ["2026-W23", "2026-W24"],
            "week_start": [date(2026, 6, 1), date(2026, 6, 8)],
            "scope": ["all", "all"],
            "bloc": ["chinese", "chinese"],
            "tokens_total": [100, 100],
            "dollar_volume_usd": [1.0, 1.0],
            "token_share": [0.5, 0.5],
            "dollar_share": [0.4, 0.45],
            "days_observed": [5, 3],
            "is_complete": [True, False],
            "computed_at": [NOW, NOW],
        })
        s = bloc_share_series(df, "chinese", "all")
        assert s.height == 1
        assert s["value"][0] == pytest.approx(0.4)
