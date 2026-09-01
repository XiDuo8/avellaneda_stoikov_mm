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
        q: Current inventory held by the market maker (positive = long),
            negative = short).
        t: Current time.
        T: Terminal time of the trading session. Must satisfy T>=t.
        gamma: Risk aversion parameter. Must be positive.
        sigma: Volatility coefficient, on the same time-unit convention as t
            and T. Must be non-negative.

    Returns:
        The reservation price r(s, q, t).
    """