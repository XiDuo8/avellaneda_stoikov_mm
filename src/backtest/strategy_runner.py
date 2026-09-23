import numpy as np
import pandas as pd

from src.backtest.lob_replay import LOBReplayEngine


def run_backtest(
    quote_fn,
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    quote_size: float,
    tick_size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs a full backtest of quote_fn against LOB data."""
    assert len(messages) == len(orderbook), (
        "messages/orderbook row count mismatch -- are these row-aligned files "
        "files from the same session?"
    )

    n = len(messages)
    engine = LOBReplayEngine()

    session_start = messages["time"].iloc[0]
    session_end = messages["time"].iloc[-1]
    total_T = session_end - session_start

    prices = np.zeros(n)
    q = np.zeros(n)
    pnl = np.zeros(n)

    q_current = 0.0
    cash_current = 0.0

    for i, (msg_row, ob_row) in enumerate(zip(messages.itertuples(), orderbook.itertuples())):
        fill_amount = engine.process_message_event(msg_row)

        if fill_amount is not None:
            side = "bid" if msg_row.direction == 1 else "ask"
            if side == "bid":
                q_current += fill_amount
                cash_current -= fill_amount * msg_row.price
            else:
                q_current -= fill_amount
                cash_current += fill_amount * msg_row.price

        mid = (ob_row.ask_price_1 + ob_row.bid_price_1) / 2
        prices[i] = mid
        q[i] = q_current
        pnl[i] = cash_current + q_current * mid

        engine.cancel_synthetic_order("bid")
        engine.cancel_synthetic_order("ask")

        t = msg_row.time - session_start
        new_bid, new_ask = quote_fn(mid, q_current, t, total_T)

        new_bid = round(new_bid / tick_size) * tick_size
        new_ask = round(new_ask / tick_size) * tick_size

        engine.place_synthetic_order("bid", new_bid, quote_size, msg_row.time)
        engine.place_synthetic_order("ask", new_ask, quote_size, msg_row.time)

    return prices, q, pnl
