"""Unit tests for Monte Carlo aggregation across many simulated paths."""

import numpy as np

from src.naive_quoting import make_naive_strategy
from src.quoting import make_as_strategy
from src.simulation import run_simulation
from src.monte_carlo import (
    MonteCarloResults,
    run_monte_carlo,
    single_monte_carlo,
)


def test_single_monte_carlo_returns_monte_carlo_results(config):
    """single_monte_carlo should return a MonteCarloResults instance."""
    strategy = make_as_strategy(gamma=1.0, config=config)

    results = single_monte_carlo(strategy, config, n_runs=5, base_seed=0)

    assert isinstance(results, MonteCarloResults)


def test_single_monte_carlo_output_shapes(config):
    """terminal_pnl, inventory_paths, and pnl_paths have the right shapes."""
    strategy = make_as_strategy(gamma=1.0, config=config)
    n_runs = 10

    results = single_monte_carlo(strategy, config, n_runs=n_runs, base_seed=0)

    assert results.terminal_pnl.shape == (n_runs,)
    assert results.inventory_paths.shape == (n_runs, config.n_steps + 1)
    assert results.pnl_paths.shape == (n_runs, config.n_steps + 1)


def test_single_monte_carlo_terminal_pnl_matches_last_column(config):
    """terminal_pnl[i] should always equal pnl_paths[i, -1] by construction."""
    strategy = make_as_strategy(gamma=1.0, config=config)

    results = single_monte_carlo(strategy, config, n_runs=8, base_seed=0)

    assert np.array_equal(results.terminal_pnl, results.pnl_paths[:, -1])


def test_single_monte_carlo_run_i_matches_direct_run_simulation(config):
    """Run i should match run_simulation called with seed=base_seed+i."""
    strategy = make_as_strategy(gamma=1.0, config=config)
    base_seed = 0
    run_index = 3

    results = single_monte_carlo(strategy, config, n_runs=5, base_seed=0)
    _, expected_q, expected_pnl = run_simulation(
        strategy, config, seed=base_seed + run_index
    )

    assert np.array_equal(results.inventory_paths[run_index], expected_q)
    assert np.array_equal(results.pnl_paths[run_index], expected_pnl)


def test_single_monte_carlo_base_seed_offset_is_consistent(config):
    """Run i under base_seed=b should match run 0 under base_seed=b+i.

    This directly verifies the documented seed_i = base_seed + i behavior.
    """
    strategy = make_as_strategy(gamma=1.0, config=config)

    results_base_5 = single_monte_carlo(strategy, config, n_runs=1, base_seed=5)
    results_base_0 = single_monte_carlo(strategy, config, n_runs=6, base_seed=0)

    run5, run0 = results_base_5.pnl_paths[0], results_base_0.pnl_paths[5]

    assert np.array_equal(run5, run0)


def test_single_monte_carlo_deterministic_with_same_base_seed(config):
    """Same base_seed yield identical single_monte_carlo results."""
    strategy = make_as_strategy(gamma=1.0, config=config)

    results1 = single_monte_carlo(strategy, config, n_runs=5, base_seed=0)
    results2 = single_monte_carlo(strategy, config, n_runs=5, base_seed=0)

    assert np.array_equal(results1.terminal_pnl, results2.terminal_pnl)
    assert np.array_equal(results1.inventory_paths, results2.inventory_paths)


def test_run_monte_carlo_returns_two_results(config):
    """run_monte_carlo should return a (as_results, naive_results) tuple."""
    as_fn = make_as_strategy(gamma=1.0, config=config)
    naive_fn = make_naive_strategy(delta=0.1)

    as_results, naive_results = run_monte_carlo(
        as_fn, naive_fn, config, n_runs=5
    )

    assert isinstance(as_results, MonteCarloResults)
    assert isinstance(naive_results, MonteCarloResults)


def test_run_monte_carlo_matches_calling_single_monte_carlo_separately(config):
    """run_monte_carlo's outputs should match calling single_monte_carlo
    directly for each strategy with the same config, n_runs, and base_seed."""
    as_fn = make_as_strategy(gamma=1.0, config=config)
    naive_fn = make_naive_strategy(delta=0.1)

    as_results, naive_results = run_monte_carlo(
        as_fn, naive_fn, config, n_runs=5, base_seed=1
    )
    expected_as = single_monte_carlo(as_fn, config, n_runs=5, base_seed=1)
    expected_naive = single_monte_carlo(naive_fn, config, n_runs=5, base_seed=1)

    for result, expected in [
        (as_results, expected_as),
        (naive_results, expected_naive),
    ]:
        assert np.array_equal(result.terminal_pnl, expected.terminal_pnl)
