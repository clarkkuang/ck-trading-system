"""Tests for the npm/PyPI download collector (mocked httpx)."""

from datetime import date
from unittest.mock import MagicMock, patch

from ck_trading.collectors.pkg_downloads import (
    PackageDownloadsCollector,
    iso_week_label,
    previous_complete_iso_week,
)


def _make_mock_client(get_side_effect):
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.side_effect = get_side_effect
    return mock_client


def _resp(status_code=200, json_data=None):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data
    return r


# 2026-07-01 is a Wednesday; previous complete ISO week = Jun 22 (Mon) .. Jun 28 (Sun)
AS_OF = date(2026, 7, 1)


class TestWeekHelpers:
    def test_previous_complete_iso_week(self):
        mon, sun = previous_complete_iso_week(AS_OF)
        assert mon == date(2026, 6, 22)
        assert sun == date(2026, 6, 28)
        assert mon.weekday() == 0 and sun.weekday() == 6

    def test_iso_week_label(self):
        assert iso_week_label(date(2026, 6, 28)) == "2026-W26"


class TestNpm:
    def test_exact_week_window(self):
        client = _make_mock_client([
            _resp(json_data={
                "downloads": 10_041_901,
                "start": "2026-06-22", "end": "2026-06-28",
                "package": "@anthropic-ai/claude-code",
            }),
        ])
        with patch("ck_trading.collectors.pkg_downloads.httpx.Client",
                   return_value=client):
            df = PackageDownloadsCollector().collect_npm(
                ["@anthropic-ai/claude-code"], AS_OF
            )
        assert df.height == 1
        row = df.row(0, named=True)
        assert row["downloads"] == 10_041_901
        assert row["iso_week"] == "2026-W26"
        assert row["period_start"] == date(2026, 6, 22)
        # scoped package name kept intact in the URL
        url = client.get.call_args[0][0]
        assert "@anthropic-ai/claude-code" in url
        assert "2026-06-22:2026-06-28" in url

    def test_fallback_to_last_week(self):
        client = _make_mock_client([
            _resp(status_code=404),
            _resp(json_data={
                "downloads": 500, "start": "2026-06-24", "end": "2026-06-30",
            }),
        ])
        with patch("ck_trading.collectors.pkg_downloads.httpx.Client",
                   return_value=client):
            df = PackageDownloadsCollector().collect_npm(["newpkg"], AS_OF)
        assert df.height == 1
        assert df["downloads"][0] == 500
        # actual window recorded, not the requested one
        assert df["period_end"][0] == date(2026, 6, 30)

    def test_one_package_failing_skips_only_it(self):
        client = _make_mock_client([
            _resp(status_code=500),  # pkg-a exact window
            _resp(status_code=500),  # pkg-a last-week fallback
            _resp(json_data={"downloads": 42, "start": "2026-06-22",
                             "end": "2026-06-28"}),  # pkg-b
        ])
        with patch("ck_trading.collectors.pkg_downloads.httpx.Client",
                   return_value=client):
            df = PackageDownloadsCollector().collect_npm(["pkg-a", "pkg-b"], AS_OF)
        assert df.height == 1
        assert df["package"][0] == "pkg-b"

    def test_all_fail_returns_empty_with_schema(self):
        client = _make_mock_client([_resp(status_code=500)] * 4)
        with patch("ck_trading.collectors.pkg_downloads.httpx.Client",
                   return_value=client):
            df = PackageDownloadsCollector().collect_npm(["a", "b"], AS_OF)
        assert df.is_empty()
        assert "downloads" in df.columns


class TestPypi:
    def _collector(self):
        return PackageDownloadsCollector(pypi_sleep_s=0.0)

    def test_last_week_extracted(self):
        client = _make_mock_client([
            _resp(json_data={"data": {"last_day": 1, "last_month": 100,
                                      "last_week": 2_000_000}}),
        ])
        with patch("ck_trading.collectors.pkg_downloads.httpx.Client",
                   return_value=client):
            df = self._collector().collect_pypi(["anthropic"], AS_OF)
        assert df.height == 1
        row = df.row(0, named=True)
        assert row["downloads"] == 2_000_000
        assert row["iso_week"] == "2026-W26"
        assert row["registry"] == "pypi"

    def test_429_retries_once(self):
        client = _make_mock_client([
            _resp(status_code=429),
            _resp(json_data={"data": {"last_week": 777}}),
        ])
        with patch("ck_trading.collectors.pkg_downloads.httpx.Client",
                   return_value=client), \
             patch("ck_trading.collectors.pkg_downloads.time.sleep"):
            df = self._collector().collect_pypi(["anthropic"], AS_OF)
        assert df.height == 1
        assert df["downloads"][0] == 777

    def test_persistent_429_skips_package(self):
        client = _make_mock_client([
            _resp(status_code=429), _resp(status_code=429),
        ])
        with patch("ck_trading.collectors.pkg_downloads.httpx.Client",
                   return_value=client), \
             patch("ck_trading.collectors.pkg_downloads.time.sleep"):
            df = self._collector().collect_pypi(["anthropic"], AS_OF)
        assert df.is_empty()

    def test_user_agent_set(self):
        client = _make_mock_client([
            _resp(json_data={"data": {"last_week": 1}}),
        ])
        with patch("ck_trading.collectors.pkg_downloads.httpx.Client",
                   return_value=client) as client_cls:
            self._collector().collect_pypi(["anthropic"], AS_OF)
        _, kwargs = client_cls.call_args
        assert "ck-trading-system" in kwargs["headers"]["User-Agent"]
