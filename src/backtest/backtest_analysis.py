"""Summary statistics and diagnostics for LOB backtests"""

import numpy as np

from src.analysis import sharpe_ratio
from src.monte_carlo import MonteCarloResults


def summarise_backtest(
    prices: np.ndarray,
    q: np.ndarray,
    pnl: np.ndarray,
) -> dict[str, float]:
    pnl_increments = np.diff(pnl)

    return {
        "final_pnl": pnl[-1],
        "pnl_sharpe": sharpe_ratio(pnl_increments),
        "inventory_variance": np.var(q),
        "max_abs_inventory": np.abs(q).max(),
        "n_fills": int(np.sum(np.diff(q) != 0)),
    }


def compare_strategies(
    as_result: tuple[np.ndarray, np.ndarray, np.ndarray],
    naive_result: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Summarises AS and naive results from backtest."""
    return {
        "as": summarise_backtest(*as_result),
        "naive": summarise_backtest(*naive_result),
    }


def pnl_percentile_vs_monte_carlo(
    backtest_final_pnl: float,
    mc_results: MonteCarloResults,
) -> float:
    return 100 * np.mean(mc_results.terminal_pnl <= backtest_final_pnl)
