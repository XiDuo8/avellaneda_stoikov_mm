import numpy as np

from src.market import simulate_gbm, order_arrives


def run_simulation(
    quote_fn,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    A: float,
    k: float,
    seed: int,
):
    """Runs a full mm simulation for a given quoting strategy.

    Simulates on GBM price path, then steps through time computing quotes via
    quote_fn, checking for bid/ask fills, and updating inventory, cash, and
    mark-to-market PnL.

    Args:
        quote_fn: A callable (price, q, t, T) -> (bid, ask), e.g. one produced
            by make_as_strategy or make_naive_strategy.
        s0: Initial asset price.
        mu: Drift of the underlying GBM price process.
        sigma: Volatility of the underlying GBM price process.
        T: Simulation horizon (time to maturity).
        n_steps: Number of discrete time steps.
        A: Base fill-intensity parameter for order arrivals.
        k: Fill-intensity decay parameter for order arrivals.
        seed: Random seed controlling both the price path and fills.

    Returns:
        A tuple (prices, q, PnL) of arrays, each of length n_steps + 1, giving
        the simulated price path, inventory path and mark to market PnL path.
    """
    dt = T / n_steps

    rng = np.random.default_rng(seed)

    prices = simulate_gbm(s0, mu, sigma, T, n_steps, seed)

    q = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)
    pnl = np.zeros(n_steps + 1)

    for i in range(n_steps):
        t = i * dt
        bid, ask = quote_fn(prices[i], q[i], t, T)

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

    return prices, q, pnl
