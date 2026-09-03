from dataclasses import dataclass

import numpy as np

from src.simulation import run_simulation


@dataclass
class MonteCarloResults:
    """Aggregated results from running a strategy over many simulated paths.

    Args:
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
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    A: float,
    k: float,
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
        s0: Initial price.
        mu: Drift.
        sigma: volatility.
        T: Time horizon.
        n_steps: Number of simulation steps.
        A: Fill intensity base rate.
        k: Fill intensity decay rate.
        n_runs: Number of Monte Carlo paths to simulate.
        base_seed: Seed offset for run 0: run i uses base_seed + i.

    Returns:
        tuple[MonteCarloResults, MonteCarloResults]: Results for AS and naive
        strategies, respectively.
    """
    as_terminal_pnl = np.empty(n_runs)
    as_inventory_paths = np.empty((n_runs, n_steps + 1))
    as_pnl_paths = np.empty((n_runs, n_steps + 1))

    naive_terminal_pnl = np.empty(n_runs)
    naive_inventory_paths = np.empty((n_runs, n_steps + 1))
    naive_pnl_paths = np.empty((n_runs, n_steps + 1))

    for i in range(n_runs):
        seed_i = base_seed + i

        as_prices, as_q, as_pnl = run_simulation(
            as_quote_fn, s0, mu, sigma, T, n_steps, A, k, seed=seed_i
        )
        naive_prices, naive_q, naive_pnl = run_simulation(
            naive_quote_fn, s0, mu, sigma, T, n_steps, A, k, seed=seed_i
        )

        as_terminal_pnl[i] = as_pnl[-1]
        as_inventory_paths[i] = as_q
        as_pnl_paths[i] = as_pnl

        naive_terminal_pnl[i] = naive_pnl[-1]
        naive_inventory_paths[i] = naive_q
        naive_pnl_paths[i] = naive_pnl

        as_results = MonteCarloResults(
            terminal_pnl=as_terminal_pnl,
            inventory_paths=as_inventory_paths,
            pnl_paths=as_pnl_paths,
        )
        naive_results = MonteCarloResults(
            terminal_pnl=naive_terminal_pnl,
            inventory_paths=naive_inventory_paths,
            pnl_paths=naive_pnl_paths,
        )

    return as_results, naive_results
