import numpy as np

from src.analysis import sharpe_ratio
from src.quoting import make_as_strategy
from src.naive_quoting import make_naive_strategy
from src.monte_carlo import run_monte_carlo
from src.plotting import plot_pnl_distribution, plot_gamma_sweep
from src.sweep import sweep_gamma


def main():
    s0 = 100.0
    mu = 0.15
    sigma = 0.46
    T = 1.0
    n_steps = 10000
    gamma = 2
    k = 1.5
    A = 140.0
    n_runs = 1000
    delta = (gamma * sigma ** 2 * T + (2 / gamma) * np.log(1 + gamma / k)) / 2

    as_fn = make_as_strategy(gamma, sigma, k)
    naive_fn = make_naive_strategy(delta)

    as_results, naive_results = run_monte_carlo(
        as_fn,
        naive_fn,
        s0,
        mu,
        sigma,
        T,
        n_steps,
        A,
        k,
        n_runs,
    )

    print(f"AS Sharpe: {sharpe_ratio(as_results.terminal_pnl):.3f}")
    print(f"Naive Sharpe: {sharpe_ratio(naive_results.terminal_pnl):.3f}")

    sweep_gammas = [0.01, 0.5, 1, 2, 5]

    sweep_results = sweep_gamma(
        sweep_gammas,
        s0,
        mu,
        sigma,
        T,
        n_steps,
        A,
        k,
        n_runs,
    )

    plot_pnl_distribution(as_results, naive_results)
    plot_gamma_sweep(sweep_results)

if __name__ == "__main__":
    main()