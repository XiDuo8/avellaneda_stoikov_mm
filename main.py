import numpy as np

from src.market import simulate_gbm, order_arrives
from src.quoting import compute_quotes


def main():
    s0 = 100.0
    mu = 0.0
    sigma = 0.2
    T = 1.0
    n_steps = 200
    gamma = 0.1
    k = 1.5
    A = 140.0
    seed = 67

    dt = T / n_steps

    rng = np.random.default_rng(seed)

    prices = simulate_gbm(s0, mu, sigma, T, n_steps, seed)

    q = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)
    pnl = np.zeros(n_steps + 1)


if __name__ == "__main__":
    main()