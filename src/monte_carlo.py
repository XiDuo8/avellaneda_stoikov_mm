"""Monte Carlo aggregation of simulation runs across many price paths."""

from dataclasses import dataclass

import numpy as np

from src.config import SimulationConfig
from src.simulation import run_simulation


@dataclass
class MonteCarloResults:
    """Aggregated results from running a strategy over many simulated paths.

    Attributes:
        terminal_pnl (np.ndarray): Final PnL for each run.
        inventory_paths (np.ndarray): Inventory over time for each run.
        pnl_paths (np.ndarray): PnL over time for each run.
    """
    terminal_pnl: np.ndarray
    inventory_paths: np.ndarray
    pnl_paths: np.ndarray


def run_monte_carlo(
    as_quote_fn,
    naive_quote_fn,
    config: SimulationConfig,
    n_runs: int,
    base_seed: int = 0,
) -> tuple[MonteCarloResults, MonteCarloResults]:
    """Runs both strategies over many simulated price paths.

    Each run index uses a shared seed across both strategies, so within a run
    they see identical price and fill randomness; only the quoting logic
    differs.

    Args:
        as_quote_fn: Quote function for the AS strategy, as returned
            by make_as_strategy.
        naive_quote_fn: Quote function for the naive strategy, as
            returned by make_naive_strategy.
        config: Simulation parameters (s0, mu, sigma, T, n_steps).
        n_runs: Number of Monte Carlo paths to simulate.
        base_seed: Seed offset for run 0: run i uses base_seed + i.

    Returns:
        tuple[MonteCarloResults, MonteCarloResults]: Results for AS and naive
        strategies, respectively.
    """
    as_results = single_monte_carlo(
        as_quote_fn,
        config,
        n_runs,
        base_seed,
    )
    naive_results = single_monte_carlo(
        naive_quote_fn,
        config,
        n_runs,
        base_seed,
    )

    return as_results, naive_results


def single_monte_carlo(
    quote_fn,
    config: SimulationConfig,
    n_runs: int,
    base_seed: int = 0,
) -> MonteCarloResults:
    """Runs one strategy over many simulated price paths.

    Args:
        quote_fn: Quote function matching the quote_fn(price, q, t, T)
            signature.
        config: Simulation parameters (s0, mu, sigma, T, n_steps).
        n_runs: Number of Monte Carlo paths to simulate.
        base_seed: Seed offset for run 0: run i uses base_seed + i.

    Returns:
        MonteCarloResults: Aggregated results across all runs.
    """
    terminal_pnl = np.empty(n_runs)
    inventory_paths = np.empty((n_runs, config.n_steps + 1))
    pnl_paths = np.empty((n_runs, config.n_steps + 1))

    for i in range(n_runs):
        seed_i = base_seed + i

        _, q, pnl = run_simulation(
            quote_fn, config, seed=seed_i
        )

        terminal_pnl[i] = pnl[-1]
        inventory_paths[i] = q
        pnl_paths[i] = pnl

    results = MonteCarloResults(
        terminal_pnl=terminal_pnl,
        inventory_paths=inventory_paths,
        pnl_paths=pnl_paths,
    )

    return results
