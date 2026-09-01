import numpy as np

def reservation_price(
    s: float,
    q: int,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
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
        T: Terminal time of the trading session. Must satisfy T>=t.
        gamma: Risk aversion parameter. Must be positive.
        sigma: Volatility coefficient, on the same time-unit convention as t
            and T. Must be non-negative.

    Returns:
        The reservation price r(s, q, t).
    """
    r = s - q * gamma * sigma ** 2 * (T - t)

    return r

def optimatal_spread(
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> float:
    """Computes the Avellaneda-Stoikov optimal total spread.

    Implements delta_a + delta_b = gamma * sigma^2 * (T-t) + (2 / gamma) *
    ln(1 + gamma / k), the total quoted spread width from Avellaneda-Stoikov
    (2008). The spread widens with more time remaining or higher volatility
    (greater inventory risk to be compensated for), and is shaped by risk
    aversion and the fill-decay rate k from the order arrival model.

    Args:
        t: Current time.
        T: Terminal time of the trading session. Must satisfy T >= t.
        gamma: Risk aversion parameter. Must be positive.
        sigma: Volatility coefficient, on the same time-unit convention as t and
            T. Must be non-negative.
        k: Decay rate controlling how quickly fill intensity falls off with
            distance from mid (same k as in market.fill_intensity).

    Returns:
        The total optimal spread width (delta_a + delta_b).
    """
    width = gamma * sigma ** 2 * (T - t) + (2 / gamma) * np.log(1 + gamma / k)

    return width