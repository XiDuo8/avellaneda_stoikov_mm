import numpy as np


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
