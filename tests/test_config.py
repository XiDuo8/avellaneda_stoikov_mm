"""Tests for SimulationConfig."""

import pytest

from src.config import SimulationConfig


def test_dollar_sigma_is_relative_volatility_times_initial_price(config):
    """A 0.2% volatility on a price of 100 is 0.2 in price units."""
    assert config.dollar_sigma == pytest.approx(0.2)


def test_dollar_sigma_scales_with_initial_price():
    """The same relative volatility is a larger dollar move on a higher price."""
    config = SimulationConfig(
        s0=500.0, mu=0.0, sigma=0.002, T=1.0, n_steps=100, A=140.0, k=1.5
    )

    assert config.dollar_sigma == pytest.approx(1.0)
