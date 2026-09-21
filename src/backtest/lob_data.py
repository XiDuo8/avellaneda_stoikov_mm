import pandas as pd


def load_messages(path: str) -> pd.DataFrame:
    """Loads a LOBSTER message file and rescales price to dollars."""
    columns = ["time", "event_type", "order_id", "size", "price", "direction"]
    messages = pd.read_csv(path, names=columns)
    messages["price"] = messages["price"] / 10000
    return messages

def load_orderbook(path:str, n_levels: int = 5) -> pd.DataFrame:
    """Loads a LOBSTER orderbook and resclales all price to dollars."""
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
