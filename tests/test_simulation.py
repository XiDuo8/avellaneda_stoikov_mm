"""Unit tests for the full single-path market-making simulation loop."""

import numpy as np

from src.config import SimulationConfig
from src.naive_quoting import make_naive_strategy
from src.quoting import make_as_strategy
from src.simulation import run_simulation


def test_run_simulation_returns_correct_lengths(config):
    """prices, q, and pnl should each have length n_steps + 1."""
    strategy = make_as_strategy(gamma=1.0, config=config)

    prices, q, pnl = run_simulation(strategy, config, seed=0)

    assert len(prices) == config.n_steps + 1
    assert len(q) == config.n_steps + 1
    assert len(pnl) == config.n_steps + 1


def test_run_simulation_starts_at_zero_inventory_and_pnl(config):
    """The simulation should start with no position and no PnL."""
    strategy = make_as_strategy(gamma=1.0, config=config)

    prices, q, pnl = run_simulation(strategy, config, seed=0)

    assert prices[0] == config.s0
    assert q[0] == 0
    assert pnl[0] == 0


def test_run_simulation_deterministic_with_same_seed(config):
    """Same strategy, config, and seed should reproduce identical output."""
    strategy = make_as_strategy(gamma=1.0, config=config)

    prices1, q1, pnl1 = run_simulation(strategy, config, seed=42)
    prices2, q2, pnl2 = run_simulation(strategy, config, seed=42)

    assert np.array_equal(prices1, prices2)
    assert np.array_equal(q1, q2)
    assert np.array_equal(pnl1, pnl2)


def test_run_simulation_price_path_independent_of_strategy(config):
    """The price path depends only on config/seed, not on which strategy quotes.

    Both the AS and naive strategies share the same underlying GBM path and
    fill-arrival RNG stream for a given seed; only inventory/PnL should differ.
    """
    as_strategy = make_as_strategy(gamma=1.0, config=config)
    naive_strategy = make_naive_strategy(delta=0.1)

    prices_as, q_as, _ = run_simulation(as_strategy, config, seed=5)
    prices_naive, q_naive, _ = run_simulation(naive_strategy, config, seed=5)

    assert np.array_equal(prices_as, prices_naive)
    assert not np.array_equal(q_as, q_naive)


def test_run_simulation_zero_fill_intensity_gives_no_fills():
    """With A=0, no orders should ever arrive, leaving inventory and PnL at zero."""
    config = SimulationConfig(
        s0=100.0, mu=0.1, sigma=0.2, T=1.0, n_steps=50, A=0.0, k=1.5
    )
    strategy = make_as_strategy(gamma=1.0, config=config)

    _, q, pnl = run_simulation(strategy, config, seed=0)

    assert np.all(q == 0)
    assert np.all(pnl == 0)


def test_run_simulation_inventory_changes_by_at_most_one_per_step(config):
    """Inventory can only move by -1, 0, or +1 each step (one fill per side)."""
    strategy = make_as_strategy(gamma=1.0, config=config)

    _, q, _ = run_simulation(strategy, config, seed=1)

    step_changes = np.diff(q)

    assert np.all(np.abs(step_changes) <= 1)


def test_run_simulation_outputs_are_finite(config):
    """No NaNs or infinities should appear in the price, inventory, or PnL paths."""
    strategy = make_as_strategy(gamma=1.0, config=config)

    prices, q, pnl = run_simulation(strategy, config, seed=3)

    assert np.all(np.isfinite(prices))
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(pnl))


def test_run_simulation_works_with_naive_strategy(config):
    """The simulation loop should run equally well with the naive strategy."""
    strategy = make_naive_strategy(delta=0.1)

    prices, q, pnl = run_simulation(strategy, config, seed=0)

    assert len(prices) == config.n_steps + 1
    assert np.all(np.isfinite(pnl))
