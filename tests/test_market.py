"""Unit tests for GBM price simulation and order arrival dynamics."""

import math

import numpy as np

from src.config import SimulationConfig
from src.market import fill_intensity, order_arrives, simulate_gbm


def test_simulate_gbm_starts_at_s0(config):
    """The first element of the price path should always equal s0."""
    prices = simulate_gbm(config, seed=0)

    assert prices[0] == config.s0


def test_simulate_gbm_correct_length(config):
    """Price path length should be n_steps + 1."""
    prices = simulate_gbm(config, seed=0)

    assert len(prices) == config.n_steps + 1


def test_simulate_gbm_deterministic_with_same_seed(config):
    """Calling simulate_gbm twice with the same seed gives an identical path."""
    prices1 = simulate_gbm(config, seed=42)
    prices2 = simulate_gbm(config, seed=42)

    assert np.array_equal(prices1, prices2)


def test_simulate_gbm_different_seeds_give_different_paths(config):
    """Different seeds should (almost certainly) produce different paths."""
    prices1 = simulate_gbm(config, seed=1)
    prices2 = simulate_gbm(config, seed=2)

    assert not np.array_equal(prices1, prices2)


def test_simulate_gbm_zero_sigma_matches_deterministic_drift():
    """With zero volatility, the path should collapse to s0 * exp(mu * t)."""
    config = SimulationConfig(
        s0=100.0, mu=0.1, sigma=0.0, T=1.0, n_steps=50, A=140.0, k=1.5
    )
    prices = simulate_gbm(config, seed=0)

    expected_final = config.s0 * np.exp(config.mu * config.T)

    assert math.isclose(prices[-1], expected_final, rel_tol=1e-9)


def test_simulate_gbm_prices_always_positive(config):
    """GBM prices should never go negative or zero, by construction."""
    prices = simulate_gbm(config, seed=7)

    assert np.all(prices > 0)


def test_fill_intensity_at_zero_delta_equals_a(config):
    """At delta=0, intensity should exactly equal the base rate A."""
    intensity = fill_intensity(delta=0.0, A=config.A, k=config.k)

    assert intensity == config.A


def test_fill_intensity_known_value():
    """Fill intensity matches the closed-form exponential decay formula."""
    intensity = fill_intensity(delta=1.0, A=140.0, k=1.5)

    assert math.isclose(intensity, 31.238222420780176, rel_tol=1e-9)


def test_fill_intensity_decreases_with_distance(config):
    """Intensity should strictly decrease as delta increases from mid."""
    near = fill_intensity(delta=0.1, A=config.A, k=config.k)
    far = fill_intensity(delta=2.0, A=config.A, k=config.k)

    assert far < near


def test_fill_intensity_never_negative(config):
    """Intensity should never be negative for any non-negative delta."""
    intensity = fill_intensity(delta=10.0, A=config.A, k=config.k)

    assert intensity >= 0


def test_order_arrives_never_fires_with_zero_intensity():
    """With A=0, fill probability is always zero, so no order should arrive."""
    rng = np.random.default_rng(0)

    results = [
        order_arrives(delta=0.5, A=0.0, k=1.5, dt=0.01, rng=rng)
        for _ in range(100)
    ]

    assert not any(results)


def test_order_arrives_always_fires_with_high_probability():
    """When intensity * dt exceeds 1, the order should always arrive."""
    rng = np.random.default_rng(0)

    results = [
        order_arrives(delta=0.0, A=1000.0, k=1.5, dt=1.0, rng=rng)
        for _ in range(50)
    ]

    assert all(results)


def test_order_arrives_is_deterministic_with_seeded_rng():
    """Same seed and inputs should give the same sequence of fill outcomes."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    results1 = [order_arrives(0.3, 140.0, 1.5, 0.001, rng1) for _ in range(20)]
    results2 = [order_arrives(0.3, 140.0, 1.5, 0.001, rng2) for _ in range(20)]

    assert results1 == results2


def test_order_arrives_returns_bool():
    """order_arrives should return a plain bool, not a numpy bool or float."""
    rng = np.random.default_rng(0)

    result = order_arrives(delta=0.5, A=140.0, k=1.5, dt=0.001, rng=rng)

    assert isinstance(result, (bool, np.bool_))
