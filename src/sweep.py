import numpy as np

from monte_carlo import MonteCarloResults


def summarise_run(
    results: MonteCarloResults,
) -> dict[str, float]:
    """Computes inventory-risk and PnL summary statistics for one gamma value.

    Args:
        results: Monte Carlo results for a single strategy configuration.

    Returns:
        Dictionary with keys "inventory_variance", "pnl_mean", and
        "pnl_variance", summarising risk and return across runs.
    """
    per_run_variance = np.var(results.inventory_paths, axis=1)
    inventory_variance = per_run_variance.mean()

    pnl_mean = results.terminal_pnl.mean()
    pnl_variance = results.terminal_pnl.var(ddof=1)

    return {
        "inventory variance": inventory_variance,
        "pnl_mean": pnl_mean,
        "pnl_variance": pnl_variance,
    }