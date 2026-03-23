"""Tests for the signal generator."""

from datetime import date, datetime

import polars as pl
import pytest

from ck_trading.models.signals import Signal, SignalType
from ck_trading.signals.generator import SignalGenerator
from ck_trading.strategies.graham_defensive import GrahamDefensiveStrategy
from ck_trading.strategies.magic_formula import MagicFormulaStrategy


class TestSignalGenerator:
    def test_generate_with_single_strategy(self, sample_prices, sample_fundamentals):
        gen = SignalGenerator([GrahamDefensiveStrategy()])
        signals = gen.generate(sample_prices, sample_fundamentals)

        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, Signal)
            assert s.strategy_name == "Graham Defensive"
            assert s.signal_type == SignalType.BUY

    def test_generate_with_multiple_strategies(self, sample_prices, sample_fundamentals):
        gen = SignalGenerator([
            GrahamDefensiveStrategy(),
            MagicFormulaStrategy(),
        ])
        signals = gen.generate(sample_prices, sample_fundamentals)

        assert len(signals) > 0

        strategies = {s.strategy_name for s in signals}
        # Both strategies must contribute signals with this fixture data
        assert "Graham Defensive" in strategies, (
            "Graham Defensive should produce signals with the sample data"
        )
        assert "Magic Formula" in strategies, (
            "Magic Formula should produce signals with the sample data"
        )

    def test_signals_sorted_by_score(self, sample_prices, sample_fundamentals):
        gen = SignalGenerator([GrahamDefensiveStrategy()])
        signals = gen.generate(sample_prices, sample_fundamentals)

        if len(signals) > 1:
            scores = [s.score for s in signals]
            assert scores == sorted(scores, reverse=True)

    def test_empty_fundamentals(self, sample_prices):
        gen = SignalGenerator([GrahamDefensiveStrategy()])
        signals = gen.generate(sample_prices, pl.DataFrame())
        assert signals == []

    def test_empty_prices(self, sample_fundamentals):
        gen = SignalGenerator([GrahamDefensiveStrategy()])
        signals = gen.generate(pl.DataFrame(), sample_fundamentals)
        # Should not crash, signals may have price_at_signal=None
        assert isinstance(signals, list)

    def test_rationale_never_none(self, sample_prices, sample_fundamentals):
        """Regression: rationale=None caused Pydantic validation error."""
        gen = SignalGenerator([GrahamDefensiveStrategy()])
        signals = gen.generate(sample_prices, sample_fundamentals)

        for s in signals:
            assert s.rationale is not None
            assert isinstance(s.rationale, str)

    def test_price_at_signal_populated(self, sample_prices, sample_fundamentals):
        gen = SignalGenerator([GrahamDefensiveStrategy()])
        signals = gen.generate(sample_prices, sample_fundamentals)

        for s in signals:
            if s.ticker in ["AAPL", "MSFT"]:
                assert s.price_at_signal is not None
                assert s.price_at_signal > 0

    def test_no_strategies(self, sample_prices, sample_fundamentals):
        gen = SignalGenerator([])
        signals = gen.generate(sample_prices, sample_fundamentals)
        assert signals == []

    def test_add_strategy(self):
        gen = SignalGenerator()
        assert len(gen.strategies) == 0
        gen.add_strategy(GrahamDefensiveStrategy())
        assert len(gen.strategies) == 1
