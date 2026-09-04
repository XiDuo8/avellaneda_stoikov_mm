import numpy as np

from src.monte_carlo import MonteCarloResults, single_monte_carlo
from src.quoting import make_as_strategy


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
        "inventory_variance": inventory_variance,
        "pnl_mean": pnl_mean,
        "pnl_variance": pnl_variance,
    }


def sweep_gamma(
    gammas: list[float],
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    A: float,
    k: float,
    n_runs: int,
    base_seed: int = 0,
) -> list[dict[str, float]]:
    """Runs the AS strategy Monte Carlo across a sweep of gamma values.

    Uses the same base_seed for every gamma value so that the price paths and
    fill randomness are identical across the sweep, isolating the effect of
    gamma from random path variation.

    Args:
        gammas: Risk-aversion values to sweep over.
        s0: Initial price.
        mu: Drift.
        sigma: Volatility.
        T: Time horizon.
        n_steps: Number of simulation steps.
        A: Fill intensity base rate.
        k: Fill intensity decay rate.
        n_runs: Number of Monte Carlo paths per gamma value.
        base_seed: Seed offset, forwarded unchanged for every gamma value.

    Returns:
        List of dicts, one per gamma value, each containing "gamma" plus the
        keys returned by summarise_run.
    """
    sweep_results = []

    for gamma in gammas:
        as_quote_fn = make_as_strategy(gamma, sigma, k)

        results = single_monte_carlo(
            as_quote_fn,
            s0,
            mu,
            sigma,
            T,
            n_steps,
            A,
            k,
            n_runs,
            base_seed,
        )

        summary = summarise_run(results)
        sweep_results.append({**summary, "gamma": gamma})

    return sweep_results