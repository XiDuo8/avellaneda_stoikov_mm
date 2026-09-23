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
