"""Tests for the NVDA framework pipeline."""

import datetime as dt
from datetime import date, timedelta
from unittest.mock import patch

import polars as pl
import pytest

from ck_trading.collectors.us_market import USMarketCollector
from ck_trading.monitoring.nvda_config import (
    DEFAULT_CHECKLIST,
    DEFAULT_FUNDAMENTALS,
    NVDA_DEDUPE_KEYS,
)
from ck_trading.monitoring.nvda_pipeline import (
    build_nvda_series_provider,
    run_weekly_update,
)
from ck_trading.monitoring.store import MonitoringStore

AS_OF = date(2026, 7, 3)


def _nvda_store(tmp_path) -> MonitoringStore:
    return MonitoringStore(
        tmp_path, dedupe_keys=NVDA_DEDUPE_KEYS, checklist_seed=DEFAULT_CHECKLIST
    )


def _weekday_series(ticker: str, closes: list[float],
                    end: date = AS_OF) -> pl.DataFrame:
    """Consecutive weekdays ENDING at `end` (or the prior weekday)."""
    rows = []
    d = end
    for c in reversed(closes):
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        rows.append({"ticker": ticker, "date": d, "close": c})
        d -= timedelta(days=1)
    rows.reverse()
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "date": pl.Date, "close": pl.Float64}
    )


def _uptrend(n=260, start=100.0, step=0.5) -> list[float]:
    return [start + i * step for i in range(n)]


def _all_tickers_daily(nvda_closes: list[float]) -> pl.DataFrame:
    frames = [_weekday_series("NVDA", nvda_closes)]
    for t in ("AMD", "AVGO", "TSM", "SMH"):
        frames.append(_weekday_series(t, _uptrend(len(nvda_closes), 50.0, 0.1)))
    return pl.concat(frames)


def _seed_fundamentals(store: MonitoringStore, quarters=None):
    store.save_json("fundamentals", {
        "version": 1,
        "quarters": quarters if quarters is not None
        else [dict(q) for q in DEFAULT_FUNDAMENTALS],
    })


def _outcome(result, name):
    return next(o for o in result.outcomes if o.name == name)


def _by_id(result):
    return {r.rule_id: r for r in result.rule_results}


@pytest.fixture
def steady_prices():
    """Uptrend, close near highs (no dip), well above SMA200."""
    with patch.object(
        USMarketCollector, "collect_prices",
        return_value=_all_tickers_daily(_uptrend()),
    ) as m:
        yield m


class TestNvdaPipeline:
    def test_happy_path(self, tmp_path, steady_prices):
        store = _nvda_store(tmp_path)
        _seed_fundamentals(store)
        result = run_weekly_update(as_of=AS_OF, store=store)

        assert _outcome(result, "prices").status == "ok"
        assert _outcome(result, "fundamentals").status == "ok"
        assert _outcome(result, "alerts").status == "ok"
        assert not result.total_failure
        assert "NVDA framework monitor" in result.summary()

        prices = store.load("prices")
        assert set(prices["ticker"].unique().to_list()) == {
            "NVDA", "AMD", "AVGO", "TSM", "SMH"
        }
        assert store.alerts_path.exists()

    def test_first_run_no_fundamentals(self, tmp_path, steady_prices):
        store = _nvda_store(tmp_path)
        result = run_weekly_update(as_of=AS_OF, store=store)
        by_id = _by_id(result)

        assert _outcome(result, "fundamentals").status == "empty"
        # price rules live
        assert by_id["nvda_buy_dip_gated"].status == "ok"
        assert by_id["nvda_sell_below_200dma_4w"].status == "ok"
        # P/E + thesis rules lack data
        for rid in ("nvda_buy_pe_below_25", "nvda_sell_pe_above_45",
                    "nvda_sell_dc_decel_2q", "nvda_sell_guide_miss",
                    "nvda_sell_asic_share_35", "nvda_sell_gm_below_70"):
            assert by_id[rid].status == "insufficient_data", rid

    def test_prefill_state_matrix(self, tmp_path, steady_prices):
        """With prefilled fundamentals: guide ok, gm ok, asic ok, decel insufficient."""
        store = _nvda_store(tmp_path)
        _seed_fundamentals(store)
        result = run_weekly_update(as_of=AS_OF, store=store)
        by_id = _by_id(result)

        assert by_id["nvda_sell_guide_miss"].status == "ok"       # +4.6
        assert by_id["nvda_sell_gm_below_70"].status == "ok"      # 75.0
        assert by_id["nvda_sell_asic_share_35"].status == "ok"    # 27.8
        # decel needs 3 points; prefill has 2
        assert by_id["nvda_sell_dc_decel_2q"].status == "insufficient_data"


class TestTechnicalRules:
    def test_dip_buy_fires_in_uptrend(self, tmp_path):
        # long uptrend then 13% drop from the 60d high, still above SMA200
        closes = _uptrend(260, 100.0, 1.0)   # ends ~359
        peak = closes[-2]
        closes[-1] = peak * 0.86             # 14% below the rolling high
        with patch.object(USMarketCollector, "collect_prices",
                          return_value=_all_tickers_daily(closes)):
            store = _nvda_store(tmp_path)
            result = run_weekly_update(as_of=AS_OF, store=store)
        r = _by_id(result)["nvda_buy_dip_gated"]
        assert r.status == "triggered"
        assert r.metric_value >= 0.127

    def test_dip_not_fired_when_gate_closed(self, tmp_path):
        # downtrend: deep dip but below SMA200 -> gated to 0
        closes = [400 - i * 1.0 for i in range(260)]
        with patch.object(USMarketCollector, "collect_prices",
                          return_value=_all_tickers_daily(closes)):
            store = _nvda_store(tmp_path)
            result = run_weekly_update(as_of=AS_OF, store=store)
        r = _by_id(result)["nvda_buy_dip_gated"]
        assert r.status == "ok"
        assert r.metric_value == 0.0

    def test_below_200dma_4w_fires(self, tmp_path):
        # flat 240d @ 300, then 20 weekdays (4 weeks) @ 250 (below SMA)
        closes = [300.0] * 240 + [250.0] * 20
        with patch.object(USMarketCollector, "collect_prices",
                          return_value=_all_tickers_daily(closes)):
            store = _nvda_store(tmp_path)
            result = run_weekly_update(as_of=AS_OF, store=store)
        r = _by_id(result)["nvda_sell_below_200dma_4w"]
        assert r.status == "triggered"
        assert r.streak >= 4

    def test_below_200dma_3w_not_enough(self, tmp_path):
        closes = [300.0] * 245 + [250.0] * 15  # 3 weeks below
        with patch.object(USMarketCollector, "collect_prices",
                          return_value=_all_tickers_daily(closes)):
            store = _nvda_store(tmp_path)
            result = run_weekly_update(as_of=AS_OF, store=store)
        r = _by_id(result)["nvda_sell_below_200dma_4w"]
        assert r.status == "ok"
        assert r.streak == 3


class TestValuationRules:
    def _run_with_price(self, tmp_path, last_close: float):
        closes = _uptrend(260, 100.0, 0.5)
        closes[-1] = last_close
        with patch.object(USMarketCollector, "collect_prices",
                          return_value=_all_tickers_daily(closes)):
            store = _nvda_store(tmp_path)
            _seed_fundamentals(store)  # eps 7.0 from 2026Q2
            return run_weekly_update(as_of=AS_OF, store=store)

    def test_pe_buy_fires(self, tmp_path):
        result = self._run_with_price(tmp_path, 168.0)  # 24.0x
        by_id = _by_id(result)
        assert by_id["nvda_buy_pe_below_25"].status == "triggered"
        assert by_id["nvda_sell_pe_above_45"].status == "ok"

    def test_pe_sell_fires(self, tmp_path):
        result = self._run_with_price(tmp_path, 322.0)  # 46.0x
        by_id = _by_id(result)
        assert by_id["nvda_sell_pe_above_45"].status == "triggered"
        assert by_id["nvda_buy_pe_below_25"].status == "ok"

    def test_pe_neutral(self, tmp_path):
        result = self._run_with_price(tmp_path, 208.0)  # ~29.7x
        by_id = _by_id(result)
        assert by_id["nvda_buy_pe_below_25"].status == "ok"
        assert by_id["nvda_sell_pe_above_45"].status == "ok"


class TestThesisRules:
    def _run_with_quarters(self, tmp_path, quarters, steady=True):
        closes = _uptrend()
        with patch.object(USMarketCollector, "collect_prices",
                          return_value=_all_tickers_daily(closes)):
            store = _nvda_store(tmp_path)
            _seed_fundamentals(store, quarters)
            return run_weekly_update(as_of=AS_OF, store=store)

    def test_dc_decel_2q_fires(self, tmp_path):
        quarters = [
            {"quarter": "2026Q1", "dc_qoq_growth_pct": 20.0},
            {"quarter": "2026Q2", "dc_qoq_growth_pct": 18.0},
            {"quarter": "2026Q3", "dc_qoq_growth_pct": 17.0},
        ]
        r = _by_id(self._run_with_quarters(tmp_path, quarters))
        assert r["nvda_sell_dc_decel_2q"].status == "triggered"

    def test_dc_accel_ok(self, tmp_path):
        quarters = [
            {"quarter": "2026Q1", "dc_qoq_growth_pct": 16.0},
            {"quarter": "2026Q2", "dc_qoq_growth_pct": 21.0},
            {"quarter": "2026Q3", "dc_qoq_growth_pct": 22.0},
        ]
        r = _by_id(self._run_with_quarters(tmp_path, quarters))
        assert r["nvda_sell_dc_decel_2q"].status == "ok"

    def test_guide_miss_fires(self, tmp_path):
        quarters = [{"quarter": "2026Q3", "guide_vs_consensus_pct": -1.0}]
        r = _by_id(self._run_with_quarters(tmp_path, quarters))
        assert r["nvda_sell_guide_miss"].status == "triggered"

    def test_asic_35_inclusive(self, tmp_path):
        quarters = [{"quarter": "2026Q3", "asic_server_share_pct": 35.0}]
        r = _by_id(self._run_with_quarters(tmp_path, quarters))
        assert r["nvda_sell_asic_share_35"].status == "triggered"

    def test_gm_below_70_fires(self, tmp_path):
        quarters = [{"quarter": "2026Q3", "gross_margin_pct": 69.9}]
        r = _by_id(self._run_with_quarters(tmp_path, quarters))
        assert r["nvda_sell_gm_below_70"].status == "triggered"


class TestInfrastructure:
    def test_dry_run_writes_nothing(self, tmp_path, steady_prices):
        store = _nvda_store(tmp_path)
        run_weekly_update(as_of=AS_OF, store=store, dry_run=True)
        assert store.load("prices").is_empty()
        assert not store.alerts_path.exists()

    def test_rerun_idempotent(self, tmp_path, steady_prices):
        store = _nvda_store(tmp_path)
        run_weekly_update(as_of=AS_OF, store=store)
        n1 = store.load("prices").height
        run_weekly_update(as_of=AS_OF, store=store)
        assert store.load("prices").height == n1

    def test_prices_failure_isolated(self, tmp_path):
        with patch.object(USMarketCollector, "collect_prices",
                          side_effect=RuntimeError("yf down")):
            store = _nvda_store(tmp_path)
            # pre-seed history so alerts still evaluate
            seeded = _all_tickers_daily(_uptrend())
            from ck_trading.monitoring.nvda_metrics import normalize_daily
            store.save("prices", normalize_daily(seeded))
            result = run_weekly_update(as_of=AS_OF, store=store)
        assert _outcome(result, "prices").status == "failed"
        assert result.total_failure
        assert _outcome(result, "alerts").status == "ok"
        assert _by_id(result)["nvda_buy_dip_gated"].status == "ok"

    def test_missing_ticker_tolerated(self, tmp_path):
        partial = _all_tickers_daily(_uptrend()).filter(
            pl.col("ticker") != "SMH"
        )
        with patch.object(USMarketCollector, "collect_prices",
                          return_value=partial):
            store = _nvda_store(tmp_path)
            result = run_weekly_update(as_of=AS_OF, store=store)
        o = _outcome(result, "prices")
        assert o.status == "ok"
        assert o.extra["missing"] == ["SMH"]


class TestProviderRouting:
    def test_metric_keys(self, tmp_path):
        store = _nvda_store(tmp_path)
        from ck_trading.monitoring.nvda_metrics import normalize_daily
        store.save("prices", normalize_daily(_all_tickers_daily(_uptrend())))
        _seed_fundamentals(store)
        provider = build_nvda_series_provider(store)

        assert not provider("nvda.gated_dip", {"ticker": "NVDA"}).is_empty()
        assert not provider("nvda.pct_vs_200dma", {"ticker": "NVDA"}).is_empty()
        assert not provider("nvda.forward_pe", {"ticker": "NVDA"}).is_empty()
        assert not provider(
            "nvda.fundamental", {"field": "dc_qoq_growth_pct"}
        ).is_empty()
        assert provider("nope", {}).is_empty()
