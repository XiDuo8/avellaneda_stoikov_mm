"""Single-path market making simulation loop"""

import numpy as np

from src.config import SimulationConfig
from src.market import simulate_gbm, order_arrives


def run_simulation( # pylint: disable=too-many-locals
    quote_fn,
    config: SimulationConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs a full mm simulation for a given quoting strategy.

    Simulates on GBM price path, then steps through time computing quotes via
    quote_fn, checking for bid/ask fills, and updating inventory, cash, and
    mark-to-market PnL.

    Args:
        quote_fn: A callable (price, q, t, T) -> (bid, ask), e.g. one produced
            by make_as_strategy or make_naive_strategy.
        config: Simulation parameters (s0, mu, sigma, T, n_steps).
        seed: Random seed controlling both the price path and fills.

    Returns:
        A tuple (prices, q, PnL) of arrays, each of length n_steps + 1, giving
        the simulated price path, inventory path and mark to market PnL path.
    """
    dt = config.T / config.n_steps

    rng = np.random.default_rng(seed)

    prices = simulate_gbm(config, seed)

    q = np.zeros(config.n_steps + 1)
    cash = np.zeros(config.n_steps + 1)
    pnl = np.zeros(config.n_steps + 1)

    for i in range(config.n_steps):
        t = i * dt
        bid, ask = quote_fn(prices[i], q[i], t, config.T)

        delta_bid = prices[i] - bid
        delta_ask = ask - prices[i]

        bid_filled = order_arrives(delta_bid, config.A, config.k, dt, rng)
        ask_filled = order_arrives(delta_ask, config.A, config.k, dt, rng)

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
