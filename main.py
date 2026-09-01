import numpy as np

from src.market import simulate_gbm, order_arrives
from src.quoting import compute_quotes
from src.plotting import plot_results


def main():
    s0 = 100.0
    mu = 0
    sigma = 2
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

    for i in range(n_steps):
        t = i * dt
        bid, ask = compute_quotes(prices[i], q[i], t, T, gamma, sigma, k)

        delta_bid = prices[i] - bid
        delta_ask = ask - prices[i]

        bid_filled = order_arrives(delta_bid, A, k, dt, rng)
        ask_filled = order_arrives(delta_ask, A, k, dt, rng)

        q[i + 1] = q[i]
        cash[i + 1] = cash[i]

        if bid_filled:
            q[i + 1] = q[i] + 1
            cash[i + 1] = cash[i] - bid

        if ask_filled:
            q[i + 1] -=  1
            cash[i + 1] += ask

        pnl[i + 1] = cash[i + 1] + q[i + 1] * prices[i + 1]

    plot_results(prices, q, pnl, T)


if __name__ == "__main__":
    main()