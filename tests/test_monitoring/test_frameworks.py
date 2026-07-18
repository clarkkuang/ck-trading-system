"""Tests for cross-system signal reconciliation."""

from ck_trading.monitoring.frameworks import (
    FRAMEWORK_TICKERS,
    buy_sell_summary,
    reconcile_verdict,
)


def _alerts(*rules) -> dict:
    return {"rules": list(rules), "updated_at": "2026-07-18T00:00:00Z"}


def _rule(rule_id: str, status: str) -> dict:
    return {"rule_id": rule_id, "status": status, "action_label": "x"}


class TestBuySellSummary:
    def test_split_by_convention(self):
        s = buy_sell_summary(_alerts(
            _rule("nflx_buy_ladder_t1", "triggered"),
            _rule("nflx_buy_ladder_t2", "ok"),
            _rule("nflx_sell_margin_below_28", "triggered"),
            _rule("nflx_sell_ads_off_track", "insufficient_data"),
        ))
        assert [r["rule_id"] for r in s["buy_triggered"]] == ["nflx_buy_ladder_t1"]
        assert [r["rule_id"] for r in s["sell_triggered"]] == [
            "nflx_sell_margin_below_28"
        ]
        assert s["n_rules"] == 4

    def test_empty_and_malformed(self):
        assert buy_sell_summary({})["buy_triggered"] == []
        assert buy_sell_summary({"rules": ["junk", {"no_id": 1}]})["sell_triggered"] == []


class TestReconcileVerdict:
    def test_sell_overrides_everything(self):
        head, _ = reconcile_verdict(False, True, "BUY")
        assert "🔴" in head

    def test_both_fired_is_conflict(self):
        head, _ = reconcile_verdict(True, True, "HOLD")
        assert "🟠" in head

    def test_double_confirmation(self):
        head, _ = reconcile_verdict(True, False, "BUY")
        assert head == "🟢 双确认"

    def test_framework_buy_with_knife_timing(self):
        head, why = reconcile_verdict(True, False, "WATCH (still falling)")
        assert head == "🟢 按预案执行"
        assert "创新低" in why

    def test_framework_buy_with_trend_block(self):
        head, why = reconcile_verdict(True, False, "BLOCKED (downtrend)")
        assert head == "🟢 按预案执行"
        assert "趋势" in why

    def test_dip_only(self):
        head, _ = reconcile_verdict(False, False, "BUY")
        assert "dip" in head

    def test_nothing(self):
        head, _ = reconcile_verdict(False, False, "HOLD")
        assert "⚪" in head


def test_framework_tickers_registered():
    assert FRAMEWORK_TICKERS == {
        "NVDA": "10_nvda_framework",
        "NFLX": "11_nflx_framework",
    }
