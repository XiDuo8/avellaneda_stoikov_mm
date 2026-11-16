"""Loading and price handling for LOBSTER message and order book files."""

import numpy as np
import pandas as pd

TICK_SIZE = 0.01


def load_messages(path: str) -> pd.DataFrame:
    """Loads a LOBSTER message file and rescales price to dollars."""
    columns = ["time", "event_type", "order_id", "size", "price", "direction"]
    messages = pd.read_csv(path, names=columns)
    messages["price"] = messages["price"] / 10000
    return messages


def load_orderbook(path: str, n_levels: int = 5) -> pd.DataFrame:
    """Loads a LOBSTER orderbook and rescales all prices to dollars."""
    columns = []
    for level in range(1, n_levels + 1):
        columns += [
            f"ask_price_{level}",
            f"ask_size_{level}",
            f"bid_price_{level}",
            f"bid_size_{level}",
        ]

    orderbook = pd.read_csv(path, names=columns)

    price_columns = [c for c in columns if "price" in c]
    orderbook[price_columns] = orderbook[price_columns] / 10000

    return orderbook


def to_ticks(prices, tick_size: float = TICK_SIZE) -> np.ndarray:
    """Converts dollar prices to tick units, e.g. 585.56 -> 58556.0.

    Dollar prices are floats, so two prices that should be equal can differ in
    the last binary digit (585.56 / 0.01 is 58555.99999999999). The fill logic
    depends on exact price comparisons, so prices are converted to ticks once
    and rounded to remove that error. Two decimals are kept because hidden
    orders can execute at sub-penny prices such as the midpoint.

    Args:
        prices: Dollar price or array of dollar prices.
        tick_size: Minimum price increment in dollars.

    Returns:
        Array of prices in ticks. Whole-cent prices come out as whole numbers.
    """
    return np.round(np.asarray(prices, dtype=float) / tick_size, 2)
