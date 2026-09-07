"""Unit tests for the gamma sensitivity sweep."""

import math

import numpy as np

from src.config import SimulationConfig
from src.monte_carlo import MonteCarloResults
from src.sweep import summarise_run, sweep_gamma


def test_summarise_run_known_values():
    """summarise_run matches hand-verified stats for a small known dataset."""
    results = MonteCarloResults(
        terminal_pnl=np.array([10.0, 20.0, 30.0]),
        inventory_paths=np.array(
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
        ),
        pnl_paths=np.zeros((3, 3)),
    )

    summary = summarise_run(results)

    assert math.isclose(summary["pnl_mean"], 20.0, rel_tol=1e-9)
    assert math.isclose(summary["pnl_variance"], 100.0, rel_tol=1e-9)
    assert math.isclose(
        summary["inventory_variance"], 0.2222222222222222, rel_tol=1e-9
    )


def test_summarise_run_returns_expected_keys():
    """summarise_run should return exactly the three documented keys."""
    results = MonteCarloResults(
        terminal_pnl=np.array([1.0, 2.0, 3.0]),
        inventory_paths=np.zeros((3, 3)),
        pnl_paths=np.zeros((3, 3)),
    )

    summary = summarise_run(results)

    assert set(summary.keys()) == {
        "inventory_variance", "pnl_mean", "pnl_variance"
    }


def test_summarise_run_zero_inventory_gives_zero_variance():
    """If inventory never moves, inventory_variance should be zero."""
    results = MonteCarloResults(
        terminal_pnl=np.array([1.0, 2.0, 3.0]),
        inventory_paths=np.zeros((3, 5)),
        pnl_paths=np.zeros((3, 5)),
    )

    summary = summarise_run(results)

    assert summary["inventory_variance"] == 0.0


def test_sweep_gamma_returns_one_result_per_gamma(config):
    """The sweep should return exactly one dict per input gamma value."""
    gammas = [0.5, 1.0, 2.0]

    sweep_results = sweep_gamma(gammas, config, n_runs=5, base_seed=0)

    assert len(sweep_results) == len(gammas)


def test_sweep_gamma_preserves_gamma_order_and_values(config):
    """Each result dict's 'gamma' key should match the input, in order."""
    gammas = [0.1, 1.0, 5.0]

    sweep_results = sweep_gamma(gammas, config, n_runs=5, base_seed=0)

    assert [r["gamma"] for r in sweep_results] == gammas


def test_sweep_gamma_results_have_summarise_run_keys(config):
    """Each result should contain 'gamma' plus summarise_run's three keys."""
    sweep_results = sweep_gamma([1.0], config, n_runs=5, base_seed=0)

    assert set(sweep_results[0].keys()) == {
        "gamma", "inventory_variance", "pnl_mean", "pnl_variance",
    }


def test_sweep_gamma_zero_fill_intensity_gives_zero_pnl_and_inventory():
    """With A=0, no fills ever occur, so every gamma should show zero PnL
    and zero inventory variance regardless of risk aversion."""
    config = SimulationConfig(
        s0=100.0, mu=0.1, sigma=0.2, T=1.0, n_steps=30, A=0.0, k=1.5
    )

    sweep_results = sweep_gamma([0.5, 5.0], config, n_runs=3, base_seed=0)

    for result in sweep_results:
        assert result["pnl_mean"] == 0.0
        assert result["inventory_variance"] == 0.0


def test_sweep_gamma_deterministic_with_same_base_seed(config):
    """Same gammas with the same base_seed should give identical results."""
    gammas = [0.5, 1.0]

    results1 = sweep_gamma(gammas, config, n_runs=5, base_seed=0)
    results2 = sweep_gamma(gammas, config, n_runs=5, base_seed=0)

    for r1, r2 in zip(results1, results2):
        assert r1 == r2
