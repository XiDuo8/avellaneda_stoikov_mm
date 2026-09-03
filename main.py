import numpy as np

from src.quoting import make_as_strategy
from src.naive_quoting import make_naive_strategy
from src.simulation import run_simulation
from src.plotting import plot_comparison


def main():
    s0 = 100.0
    mu = 0.15
    sigma = 0.46
    T = 1.0
    n_steps = 10000
    gamma = 2
    k = 1.5
    A = 140.0
    seed = 11
    delta = (gamma * sigma ** 2 * T + (2 / gamma) * np.log(1 + gamma / k)) / 2

    as_strategy = make_as_strategy(gamma, sigma, k)
    naive_strategy = make_naive_strategy(delta)

    as_prices, as_q, as_pnl = run_simulation(as_strategy, s0, mu, sigma, T, n_steps, A, k, seed)
    naive_prices, naive_q, naive_pnl = run_simulation(naive_strategy, s0, mu, sigma, T, n_steps, A, k, seed)

    plot_comparison((as_prices, as_q, as_pnl), (naive_prices, naive_q, naive_pnl), T)

if __name__ == "__main__":
    main()