"""Unit tests for Monte Carlo results summary statistics and diagnostics."""

import math

import numpy as np

from src.analysis import (
    find_representative_run,
    find_shared_representative_run,
    sharpe_ratio,
)
from src.monte_carlo import MonteCarloResults


def test_sharpe_ratio_known_value():
    """Sharpe ratio matches mean / sample std for a simpe known array."""
    terminal_pnl = np.array([10.0, 20.0, 30.0])

    result = sharpe_ratio(terminal_pnl)

    assert math.isclose(result, 2.0, rel_tol=1e-9)


def test_sharpe_ratio_uses_sample_std_ddof_1():
    """Sharpe ratio should use ddof=1 (sample std), not population std."""
    terminal_pnl = np.array([10.0, 20.0, 30.0])

    expected = np.mean(terminal_pnl) / np.std(terminal_pnl, ddof=1)
    result = sharpe_ratio(terminal_pnl)

    assert math.isclose(result, expected, rel_tol=1e-9)


def test_sharpe_ratio_negative_mean_gives_negative_sharpe():
    """Sharpe ratio should be negative when mean PnL is negative."""
    terminal_pnl = np.array([-10.0, -20.0, -5.0])

    result = sharpe_ratio(terminal_pnl)

    assert result < 0


def test_sharpe_ratio_constant_pnl_gives_infinite_sharpe():
    """Zero variance in terminal PnL divides by a zero std, givine +/- inf."""

    terminal_pnl = np.array([5.0, 5.0, 5.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        result = sharpe_ratio(terminal_pnl)

    assert np.isinf(result)


def test_find_representative_run_known_value():
    """Should return the index median terminal Pnl run."""
    results = MonteCarloResults(
        terminal_pnl=np.array([10.0, 50.0, 30.0, 20.0, 40.0]),
        inventory_paths=np.zeros((5, 3)),
        pnl_paths=np.zeros((5, 3)),
    )

    idx = find_representative_run(results)

    assert idx == 2
    assert results.terminal_pnl[idx] == 30.0


def test_find_representative_run_picks_closest_when_no_exact_median():
    """With an even number of runs, the median falls between two values;
    the function should pick whichever run is numerically closest to it."""
    results = MonteCarloResults(
        terminal_pnl=np.array([10.0, 20.0, 30.0, 100.0]),
        inventory_paths=np.zeros((4, 3)),
        pnl_paths=np.zeros((4, 3)),
    )

    idx = find_representative_run(results)

    assert idx == 1


def test_find_representative_run_returns_valid_index():
    """The returned index should always be a valid index into terminal_pnl."""
    n_runs = 20
    rng = np.random.default_rng(0)
    results = MonteCarloResults(
        terminal_pnl=rng.normal(size=n_runs),
        inventory_paths=np.zeros((n_runs, 3)),
        pnl_paths=np.zeros((n_runs, 3)),
    )

    idx = find_representative_run(results)

    assert 0 <= idx < n_runs


def _results(terminal_pnl: list[float]) -> MonteCarloResults:
    """Builds MonteCarloResults with the given terminal PnL and empty paths."""
    n_runs = len(terminal_pnl)

    return MonteCarloResults(
        terminal_pnl=np.array(terminal_pnl),
        inventory_paths=np.zeros((n_runs, 3)),
        pnl_paths=np.zeros((n_runs, 3)),
    )


def test_find_shared_representative_run_known_value():
    """Picks the run with the smallest combined standardised distance."""
    as_results = _results([10.0, 20.0, 30.0, 40.0, 50.0])
    naive_results = _results([0.0, 100.0, 900.0, 110.0, 200.0])

    idx = find_shared_representative_run(as_results, naive_results)

    assert idx == 3


def test_find_shared_representative_run_differs_from_single_strategy():
    """The shared run need not be the AS strategy's own median run."""
    as_results = _results([10.0, 20.0, 30.0, 40.0, 50.0])
    naive_results = _results([0.0, 100.0, 900.0, 110.0, 200.0])

    shared_idx = find_shared_representative_run(as_results, naive_results)
    as_idx = find_representative_run(as_results)

    assert as_idx == 2
    assert shared_idx != as_idx


def test_find_shared_representative_run_is_scale_invariant():
    """Rescaling one strategy's PnL should not change the chosen run."""
    as_results = _results([10.0, 20.0, 30.0, 40.0, 50.0])
    naive_results = _results([0.0, 100.0, 900.0, 110.0, 200.0])
    scaled_naive = _results([0.0, 1e5, 9e5, 1.1e5, 2e5])

    idx = find_shared_representative_run(as_results, naive_results)
    scaled_idx = find_shared_representative_run(as_results, scaled_naive)

    assert scaled_idx == idx


def test_find_shared_representative_run_same_median_run():
    """If one run is the median for both strategies, it should be chosen."""
    as_results = _results([0.0, 50.0, 55.0, 100.0, 200.0])
    naive_results = _results([-900.0, 500.0, 10.0, 0.0, 900.0])

    idx = find_shared_representative_run(as_results, naive_results)

    assert idx == 2
