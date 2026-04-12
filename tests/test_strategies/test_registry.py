"""Tests for the strategy registry."""

import pytest

from ck_trading.strategies.registry import get_all_strategies, get_strategy


EXPECTED_NAMES = [
    "Graham Defensive",
    "Piotroski F-Score",
    "Magic Formula",
    "Composite Value",
    "DCF Intrinsic Value",
    "Dual Momentum",
    "Earnings Momentum",
    "GARP",
    "Mean Reversion",
    "Momentum",
    "Quality Factor",
    "Risk Parity",
    "Trend Following",
    "Low Volatility",
    "Insider Following",
    "Short Squeeze",
    "Sector Rotation",
    "Earnings Surprise",
    "Ten Bagger",
    "RL PPO",
]


def test_all_strategies_registered():
    """All 20 built-in strategies should be present in the registry."""
    strategies = get_all_strategies()
    for name in EXPECTED_NAMES:
        assert name in strategies, f"{name!r} not found in registry"


def test_get_all_strategies_returns_correct_count():
    """get_all_strategies() returns a dict with at least the 20 built-in entries."""
    strategies = get_all_strategies()
    assert len(strategies) >= 20


def test_get_all_strategies_values_are_classes():
    """Registry values should be classes (types), not instances."""
    for name, cls in get_all_strategies().items():
        assert isinstance(cls, type), f"{name} mapped to {type(cls)}, expected a class"


def test_get_strategy_instantiates():
    """get_strategy() should return a working strategy instance."""
    for name in EXPECTED_NAMES:
        instance = get_strategy(name)
        assert instance.name == name
        assert instance.description


def test_get_strategy_unknown_raises():
    """get_strategy() raises KeyError for an unknown name."""
    with pytest.raises(KeyError, match="not found"):
        get_strategy("NonexistentStrategy")
