"""Tests for the signal manager."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ck_trading.models.signals import Signal, SignalType
from ck_trading.signals.manager import SignalManager


def _make_signal(ticker: str, score: float, strategy: str = "Test") -> Signal:
    return Signal(
        ticker=ticker,
        signal_type=SignalType.BUY,
        strategy_name=strategy,
        score=score,
        rationale=f"Test signal for {ticker}",
        generated_at=datetime.now(),
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.save_signal = MagicMock()
    store.get_recent_signals = MagicMock(return_value=[])
    return store


class TestSignalManager:
    def test_deduplication_keeps_highest_score(self, mock_store):
        manager = SignalManager(mock_store)
        signals = [
            _make_signal("AAPL", 0.8, "Strategy A"),
            _make_signal("AAPL", 0.9, "Strategy B"),
            _make_signal("AAPL", 0.7, "Strategy C"),
        ]
        result = manager.process_signals(signals)

        assert len(result) == 1
        assert result[0].score == 0.9
        assert result[0].strategy_name == "Strategy B"

    def test_filter_by_min_score(self, mock_store):
        manager = SignalManager(mock_store)
        signals = [
            _make_signal("AAPL", 0.8),
            _make_signal("MSFT", 0.3),
            _make_signal("GOOG", 0.1),
        ]
        result = manager.process_signals(signals, min_score=0.5)

        assert len(result) == 1
        assert result[0].ticker == "AAPL"

    def test_max_signals_limit(self, mock_store):
        manager = SignalManager(mock_store)
        signals = [_make_signal(f"TICK{i}", float(i) / 10) for i in range(50)]
        result = manager.process_signals(signals, max_signals=5)

        assert len(result) == 5

    def test_sorted_by_score_descending(self, mock_store):
        manager = SignalManager(mock_store)
        signals = [
            _make_signal("AAPL", 0.5),
            _make_signal("MSFT", 0.9),
            _make_signal("GOOG", 0.7),
        ]
        result = manager.process_signals(signals)

        scores = [s.score for s in result]
        assert scores == sorted(scores, reverse=True)

    def test_signals_saved_to_store(self, mock_store):
        manager = SignalManager(mock_store)
        signals = [_make_signal("AAPL", 0.8), _make_signal("MSFT", 0.7)]
        result = manager.process_signals(signals)

        assert mock_store.save_signal.call_count == 2

    def test_empty_input(self, mock_store):
        manager = SignalManager(mock_store)
        result = manager.process_signals([])
        assert result == []
        assert mock_store.save_signal.call_count == 0

    def test_multiple_tickers_preserved(self, mock_store):
        manager = SignalManager(mock_store)
        signals = [
            _make_signal("AAPL", 0.8),
            _make_signal("MSFT", 0.7),
            _make_signal("GOOG", 0.6),
        ]
        result = manager.process_signals(signals)

        tickers = {s.ticker for s in result}
        assert tickers == {"AAPL", "MSFT", "GOOG"}
