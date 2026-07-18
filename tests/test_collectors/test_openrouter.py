"""Tests for the OpenRouter collector (mocked httpx)."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from ck_trading.collectors.openrouter import (
    OpenRouterCollector,
    RankingsSchemaError,
    _parse_price,
    _parse_rankings,
)
from ck_trading.monitoring.blocs import OTHER_MODEL_ID


def _make_mock_client(get_side_effect):
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.side_effect = get_side_effect
    return mock_client


def _resp(status_code=200, json_data=None, text=""):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data
    r.text = text
    r.raise_for_status = MagicMock(
        side_effect=None if status_code == 200 else Exception(f"HTTP {status_code}")
    )
    return r


MODELS_PAYLOAD = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-5",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            "context_length": 200000,
            "created": 1750000000,
        },
        {
            "id": "deepseek/deepseek-v4:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 128000,
            "created": 1760000000,
        },
    ]
}


class TestCollectPricing:
    def test_parses_prices_to_per_mtok(self):
        client = _make_mock_client([_resp(json_data=MODELS_PAYLOAD)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = OpenRouterCollector().collect_pricing(date(2026, 7, 1))
        assert df.height == 2
        row = df.filter(df["model_id"] == "anthropic/claude-sonnet-5").row(
            0, named=True
        )
        assert row["prompt_usd_per_mtok"] == pytest.approx(3.0)
        assert row["completion_usd_per_mtok"] == pytest.approx(15.0)
        assert row["bloc"] == "anthropic"

    def test_no_key_no_auth_header(self):
        client = _make_mock_client([_resp(json_data=MODELS_PAYLOAD)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            OpenRouterCollector(api_key="").collect_pricing()
        _, kwargs = client.get.call_args
        assert "Authorization" not in kwargs.get("headers", {})

    def test_key_sends_auth_header(self):
        client = _make_mock_client([_resp(json_data=MODELS_PAYLOAD)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            OpenRouterCollector(api_key="sk-or-xxx").collect_pricing()
        _, kwargs = client.get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer sk-or-xxx"

    def test_http_error_returns_empty(self):
        client = _make_mock_client([_resp(status_code=500)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = OpenRouterCollector().collect_pricing()
        assert df.is_empty()
        assert "model_id" in df.columns

    def test_malformed_payload_returns_empty(self):
        client = _make_mock_client([_resp(json_data={"nope": 1})])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = OpenRouterCollector().collect_pricing()
        assert df.is_empty()


RANKINGS_DAY = {
    "data": [
        {"date": "2026-06-29", "model_permaslug": "anthropic/claude-sonnet-5",
         "total_tokens": 900_000_000},
        {"date": "2026-06-29", "model_permaslug": "deepseek/deepseek-v4",
         "total_tokens": 800_000_000},
        {"date": "2026-06-29", "model_permaslug": "other",
         "total_tokens": 5_000_000_000},
    ]
}


class TestCollectRankings:
    def _collector(self):
        return OpenRouterCollector(api_key="sk-or-xxx", rate_limit_sleep_s=0.0)

    def test_no_key_returns_empty(self):
        df = OpenRouterCollector(api_key="").collect_rankings(
            date(2026, 6, 29), date(2026, 6, 29)
        )
        assert df.is_empty()

    def test_category_echoed_scope_programming(self):
        payload = {**RANKINGS_DAY, "category": "programming"}
        client = _make_mock_client([_resp(json_data=payload)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = self._collector().collect_rankings(
                date(2026, 6, 29), date(2026, 6, 29)
            )
        assert set(df["scope"].to_list()) == {"programming"}

    def test_category_not_echoed_scope_unverified(self):
        client = _make_mock_client([_resp(json_data=RANKINGS_DAY)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = self._collector().collect_rankings(
                date(2026, 6, 29), date(2026, 6, 29)
            )
        assert set(df["scope"].to_list()) == {"programming:unverified"}

    def test_category_rejected_falls_back_to_all(self):
        client = _make_mock_client([
            _resp(status_code=400, json_data={"error": "bad param"}),
            _resp(json_data=RANKINGS_DAY),
        ])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = self._collector().collect_rankings(
                date(2026, 6, 29), date(2026, 6, 29)
            )
        assert set(df["scope"].to_list()) == {"all"}
        assert client.get.call_count == 2

    def test_other_row_mapped_to_sentinel(self):
        client = _make_mock_client([_resp(json_data=RANKINGS_DAY)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = self._collector().collect_rankings(
                date(2026, 6, 29), date(2026, 6, 29)
            )
        other = df.filter(df["model_id"] == OTHER_MODEL_ID)
        assert other.height == 1
        assert other["bloc"][0] == "other"
        assert other["rank"][0] is None
        # ranked models get 1..N
        ranked = df.filter(df["model_id"] != OTHER_MODEL_ID)
        assert ranked["rank"].to_list() == [1, 2]

    def test_unrecognized_shape_raises_schema_error(self):
        client = _make_mock_client([
            _resp(json_data={"data": [{"weird": 1}]}),
        ])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            with pytest.raises(RankingsSchemaError):
                self._collector().collect_rankings(
                    date(2026, 6, 29), date(2026, 6, 29)
                )

    def test_http_500_raises_schema_error(self):
        client = _make_mock_client([
            _resp(status_code=500, json_data={}, text="server err"),
        ])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            with pytest.raises(RankingsSchemaError):
                self._collector().collect_rankings(
                    date(2026, 6, 29), date(2026, 6, 29)
                )

    def test_multi_day_fetch(self):
        day2 = {
            "data": [
                {"date": "2026-06-30", "model_permaslug": "qwen/qwen3-coder",
                 "total_tokens": 700_000_000},
            ]
        }
        client = _make_mock_client([
            _resp(json_data=RANKINGS_DAY),
            _resp(json_data=day2),
        ])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = self._collector().collect_rankings(
                date(2026, 6, 29), date(2026, 6, 30)
            )
        assert df["date"].n_unique() == 2

    def test_weekly_bucket_payload_rejected(self):
        # 2026-07 feed regression: any requested date returns Monday-dated
        # weekly buckets for a trailing ~30-day window
        weekly = {
            "data": [
                {"date": monday, "model_permaslug": "deepseek/deepseek-v4",
                 "total_tokens": 20_000_000_000_000}
                for monday in ("2026-06-29", "2026-07-06", "2026-07-13")
            ],
            "meta": {"start_date": "2026-06-18", "end_date": "2026-07-17"},
        }
        client = _make_mock_client([_resp(json_data=weekly)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            with pytest.raises(RankingsSchemaError, match="weekly/aggregate"):
                self._collector().collect_rankings(
                    date(2026, 7, 15), date(2026, 7, 15)
                )

    def test_single_foreign_dated_row_rejected(self):
        # even one row dated off the requested day breaks the daily contract
        payload = {
            "data": RANKINGS_DAY["data"] + [
                {"date": "2026-06-22", "model_permaslug": "qwen/qwen3-coder",
                 "total_tokens": 21_000_000_000_000},
            ]
        }
        client = _make_mock_client([_resp(json_data=payload)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            with pytest.raises(RankingsSchemaError):
                self._collector().collect_rankings(
                    date(2026, 6, 29), date(2026, 6, 29)
                )

    def test_dateless_rows_inherit_requested_day(self):
        payload = {
            "data": [
                {"model_permaslug": "anthropic/claude-sonnet-5",
                 "total_tokens": 900_000_000},
            ]
        }
        client = _make_mock_client([_resp(json_data=payload)])
        with patch("ck_trading.collectors.openrouter.httpx.Client",
                   return_value=client):
            df = self._collector().collect_rankings(
                date(2026, 6, 29), date(2026, 6, 29)
            )
        assert df["date"].to_list() == [date(2026, 6, 29)]


class TestParsers:
    def test_parse_price(self):
        assert _parse_price("0.000003") == pytest.approx(3.0)
        assert _parse_price("0") == 0.0
        assert _parse_price(None) is None
        assert _parse_price("abc") is None
        assert _parse_price("-1") is None

    def test_parse_rankings_alt_field_names(self):
        rows = _parse_rankings({"rows": [{"slug": "a/b", "tokens": "123"}]})
        assert rows == [{"slug": "a/b", "tokens": 123}]

    def test_parse_rankings_bad_shapes(self):
        assert _parse_rankings("nope") is None
        assert _parse_rankings({"data": "nope"}) is None
        assert _parse_rankings({"data": [{"no_slug": 1}]}) is None
