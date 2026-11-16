"""Summary statistics and diagnostics for Monte Carlo results."""

import numpy as np

from src.monte_carlo import MonteCarloResults


def sharpe_ratio(
    terminal_pnl: np.ndarray,
) -> float:
    """Computes the cross-run Sharpe ratio from terminal PnL outcomes.

    Args:
        terminal_pnl (np.ndarray): Final PnL for each Monte Carlo run.

    Returns:
        float: Mean terminal PnL divided by its SD across runs.
    """
    mean = np.mean(terminal_pnl)
    std = np.std(terminal_pnl, ddof=1)

    return mean / std


def find_representative_run(
    results: MonteCarloResults,
) -> int:
    """Find the run whose terminal PnL is closest to the median.

    Args:
        results (MonteCarloResults): Aggregate results for one strategy.

    Returns:
        int: Index of the run closest to the median terminal PnL.
    """
    median = np.median(results.terminal_pnl)
    distances = np.abs(results.terminal_pnl - median)
    representative_idx = np.argmin(distances)

    return representative_idx


def find_shared_representative_run(
    as_results: MonteCarloResults,
    naive_results: MonteCarloResults,
) -> int:
    """Find one run that is close to the median outcome for both strategies.

    Each strategy's distance from its own median terminal PnL is measured in
    units of its own standard deviation, so that the naive strategy's much
    wider distribution does not dominate the choice. The run with the smallest
    combined distance is returned, giving a single price path on which both
    strategies behave typically.

    Args:
        as_results (MonteCarloResults): Aggregate results for the AS strategy.
        naive_results (MonteCarloResults): Aggregate results for the naive
            strategy, run over the same seeds as as_results.

    Returns:
        int: Index of the run with the smallest combined distance.
    """
    combined = np.zeros(len(as_results.terminal_pnl))

    for results in (as_results, naive_results):
        pnl = results.terminal_pnl
        combined += np.abs(pnl - np.median(pnl)) / np.std(pnl, ddof=1)

    return int(np.argmin(combined))
