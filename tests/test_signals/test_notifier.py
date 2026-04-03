"""Tests for Notifier."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from ck_trading.models.signals import Signal, SignalType
from ck_trading.signals.notifier import Notifier


def _make_signal(ticker="AAPL", price=150.0):
    return Signal(
        ticker=ticker,
        signal_type=SignalType.BUY,
        strategy_name="Graham",
        score=0.85,
        generated_at=datetime(2024, 1, 15),
        price_at_signal=price,
    )


def test_no_urls_returns_false():
    notifier = Notifier(urls="")
    result = notifier.notify([_make_signal()])
    assert result is False


def test_no_signals_returns_false():
    notifier = Notifier(urls="tgram://token/chat_id")
    result = notifier.notify([])
    assert result is False


def test_format_signals():
    notifier = Notifier(urls="")
    signals = [_make_signal("AAPL", 150.0), _make_signal("MSFT", 370.0)]
    text = notifier._format_signals(signals)
    assert "AAPL" in text
    assert "MSFT" in text
    assert "BUY" in text
    assert "Graham" in text
    assert "$150.00" in text


def test_format_signal_no_price():
    notifier = Notifier(urls="")
    sig = Signal(
        ticker="AAPL",
        signal_type=SignalType.BUY,
        strategy_name="Test",
        score=0.5,
        generated_at=datetime(2024, 1, 15),
    )
    text = notifier._format_signals([sig])
    assert "AAPL" in text
    assert "$" not in text  # No price string


@patch("ck_trading.signals.notifier.apprise", create=True)
def test_notify_calls_apprise(mock_apprise_module):
    """Test that notify creates Apprise object and calls notify."""
    mock_apobj = MagicMock()
    mock_apobj.notify.return_value = True

    with patch("apprise.Apprise", return_value=mock_apobj):
        notifier = Notifier(urls="tgram://token/chat_id")
        result = notifier.notify([_make_signal()])

    assert result is True
    mock_apobj.add.assert_called_once()
    mock_apobj.notify.assert_called_once()
