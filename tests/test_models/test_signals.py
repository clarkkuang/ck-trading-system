"""Tests for signal models."""

from datetime import datetime

from ck_trading.models.signals import Signal, SignalType


def test_signal_type_enum():
    assert SignalType.BUY == "BUY"
    assert SignalType.SELL == "SELL"
    assert SignalType.HOLD == "HOLD"


def test_signal_required_fields():
    sig = Signal(
        ticker="AAPL",
        signal_type=SignalType.BUY,
        strategy_name="Graham",
        score=0.85,
        generated_at=datetime(2024, 1, 15, 10, 30),
    )
    assert sig.ticker == "AAPL"
    assert sig.signal_type == SignalType.BUY
    assert sig.score == 0.85


def test_signal_defaults():
    sig = Signal(
        ticker="AAPL",
        signal_type=SignalType.BUY,
        strategy_name="Graham",
        score=0.85,
        generated_at=datetime(2024, 1, 15),
    )
    assert sig.rationale == ""
    assert sig.price_at_signal is None
    assert sig.target_price is None
    assert sig.stop_loss is None


def test_signal_with_price_targets():
    sig = Signal(
        ticker="MSFT",
        signal_type=SignalType.BUY,
        strategy_name="DCF",
        score=0.92,
        generated_at=datetime(2024, 1, 15),
        price_at_signal=380.0,
        target_price=450.0,
        stop_loss=350.0,
    )
    assert sig.price_at_signal == 380.0
    assert sig.target_price == 450.0
    assert sig.stop_loss == 350.0
