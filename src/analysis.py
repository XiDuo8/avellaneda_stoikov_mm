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