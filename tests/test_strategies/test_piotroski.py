"""Tests for Piotroski F-Score strategy."""

from datetime import date

import polars as pl
import pytest

from ck_trading.strategies.piotroski_f_score import PiotroskiFScoreStrategy


def test_piotroski_properties():
    strategy = PiotroskiFScoreStrategy()
    assert strategy.name == "Piotroski F-Score"
    assert strategy.rebalance_frequency == "quarterly"


def test_piotroski_empty_data(sample_prices):
    strategy = PiotroskiFScoreStrategy()
    result = strategy.screen(sample_prices, pl.DataFrame(), date(2020, 12, 31))
    assert result.is_empty()
