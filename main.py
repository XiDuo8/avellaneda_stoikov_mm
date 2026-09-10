"""Runs an AS vs naive Monte Carlo comparison with gamma sensitivity sweep."""

import numpy as np

from src.analysis import sharpe_ratio
from src.config import SimulationConfig
from src.monte_carlo import run_monte_carlo
from src.naive_quoting import make_naive_strategy
from src.plotting import (
    plot_representative_run,
    plot_pnl_distribution,
    plot_gamma_sweep,
)
from src.quoting import make_as_strategy
from src.sweep import summarise_run, sweep_gamma


def main():
    """Runs the AS vs naive Monte Carlo comparison and gamma sweep.

    Prints the Sharpe ratio for each strategy, then displays the terminal
    PnL distribution and the gamma sweep risk/return tradeof plots.
    """
    config = SimulationConfig(
        s0=100.0,
        mu=0.15,
        sigma=0.5,
        T=1.0,
        n_steps=1000,
        A=150.0,
        k=1.5,
    )
    gamma = 2
    n_runs = 1000
    variance = gamma * config.sigma**2 * config.T
    liquidity = (2 / gamma) * np.log(1 + gamma / config.k)
    delta = 0.5 * (variance + liquidity)

    as_fn = make_as_strategy(gamma, config)
    naive_fn = make_naive_strategy(delta)

    as_results, naive_results = run_monte_carlo(as_fn, naive_fn, config, n_runs)

    as_summary = summarise_run(as_results)
    naive_summary = summarise_run(naive_results)

    as_terminal_inv_std = np.std(
        as_results.inventory_paths[:, -1], ddof=1
    )
    naive_terminal_inv_std = np.std(
        naive_results.inventory_paths[:, -1], ddof=1
    )

    as_pnl_std = np.std(as_results.terminal_pnl, ddof=1)
    naive_pnl_std = np.std(naive_results.terminal_pnl, ddof=1)


    print(f"AS median terminal PnL: {np.median(as_results.terminal_pnl)}")
    print(f"Naive median terminal PnL {np.median(naive_results.terminal_pnl)}")
    print(f"AS terminal PnL std. dev. {as_pnl_std}")
    print(f"Naive terminal PnL std.dev. {naive_pnl_std}")
    print(f"AS Sharpe: {sharpe_ratio(as_results.terminal_pnl):.3f}")
    print(f"Naive Sharpe: {sharpe_ratio(naive_results.terminal_pnl):.3f}")
    print(f"AS mean inventory variance: "
          f"{as_summary['inventory_variance']}")
    print(f"Naive mean inventory variance: "
          f"{naive_summary['inventory_variance']}"
    )
    print(f"AS terminal inventory std. dev.: {as_terminal_inv_std}")
    print(f"Naive terminal inventory std. dev.: {naive_terminal_inv_std}")

    sweep_gammas = [0.01, 0.5, 1, 2, 5]

    sweep_results = sweep_gamma(sweep_gammas, config, n_runs)

    plot_pnl_distribution(as_results, naive_results)
    plot_representative_run(as_results, naive_results, config.T)
    plot_gamma_sweep(sweep_results)


if __name__ == "__main__":
    main()
