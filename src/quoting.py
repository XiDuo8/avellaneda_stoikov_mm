"""Avellaneda-Stoikov reservation price, optimal spread, and quoting."""

import numpy as np

from src.config import SimulationConfig

def reservation_price(
    s: float,
    q: float,
    t: float,
    gamma: float,
    config: SimulationConfig
) -> float:
    """Computes the Avellaneda-Stoikov reservation price.

    Implements r(s, q, t) = s - q * gamma * sigma^2 * (T - t), the inventory
    adjusted fair price from Avellaneda-Stoikov (2008): it shifts below the mid
    price when the market maker is long inventory and above it when short,
    proportional to the variance remaining over the trading horizon.

    Args:
        s: Current mid-price of the asset.
        q: Current inventory held by the market maker (positive = long,
            negative = short).
        t: Current time.
        gamma: Risk aversion parameter. Must be positive.
        config: Simulation parameters (s0, mu, sigma, T, n_steps).

    Returns:
        The reservation price r(s, q, t).
    """
    r = s - q * gamma * config.sigma ** 2 * (config.T - t)

    return r

def optimal_spread(
    t: float,
    gamma: float,
    config: SimulationConfig
) -> float:
    """Computes the Avellaneda-Stoikov optimal total spread.

    Implements delta_a + delta_b = gamma * sigma^2 * (T-t) + (2 / gamma) *
    ln(1 + gamma / k), the total quoted spread width from Avellaneda-Stoikov
    (2008). The spread widens with more time remaining or higher volatility
    (greater inventory risk to be compensated for), and is shaped by risk
    aversion and the fill-decay rate k from the order arrival model.

    Args:
        t: Current time.
        gamma: Risk aversion parameter. Must be positive.
        config: Simulation parameters (s0, mu, sigma, T, n_steps).

    Returns:
        The total optimal spread width (delta_a + delta_b).
    """
    risk = gamma * config.sigma**2 * (config.T - t)
    liquidity = (2 / gamma) * np.log(1 + gamma / config.k)
    width = risk + liquidity

    return width

def compute_quotes(
    s: float,
    q: float,
    t: float,
    gamma: float,
    config: SimulationConfig
) -> tuple[float, float]:
    """Computes the bid and ask quotes from the AS reservation price and spread.

    Calls reservation_price to get the inventory-adjusted fair price, then calls
    optimal_spread to get the total spread width, and splits that width
    symmetrically around the reservation price to produce a bid and an ask.

    Args:
        s: Current mid-price of the asset.
        q: Current inventory held by the market maker (positive = long,
            negative = short).
        t: Current time.
        gamma: Risk aversion parameter. Must be positive.
        config: Simulation parameters (s0, mu, sigma, T, n_steps).

    Returns:
        A tuple (bid, ask) giving the market maker's quoted bid and ask prices
        for this timestep.
    """
    r = reservation_price(s, q, t, gamma, config)

    total_width = optimal_spread(t, gamma, config)
    width = total_width / 2

    return r - width, r + width

def make_as_strategy(
    gamma: float,
    config: SimulationConfig
):
    """Builds a quote_fn closure implementing the AS strategy.

    Args:
        gamma: Risk aversion parameter. Must be positive.
        config: Simulation parameters (s0, mu, sigma, T, n_steps).

    Returns:
        A callable strategy(price, q, t, T) -> (bid, ask), suitable for passing
            to run_simulation.

    """
    def strategy(
        price: float,
        q: float,
        t: float,
        T: float    # pylint: disable=unused-argument
    ) -> tuple[float, float]:
        return compute_quotes(price, q, t, gamma, config)

    return strategy
