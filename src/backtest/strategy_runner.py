"""Event-by-event backtest of a quoting strategy against LOBSTER data."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.lob_data import TICK_SIZE, to_ticks
from src.backtest.lob_replay import LOBReplayEngine


@dataclass
class BacktestResult:
    """Paths and fills from one backtest run.

    Attributes:
        time: Timestamp of each message, in seconds after midnight.
        prices: Mid-price after each message, in dollars.
        q: Inventory after each message, in shares.
        pnl: Mark-to-market PnL after each message, in dollars.
        fills: One row per fill, with columns time, side, price, size and mid
            (the mid-price when the filled quote was last updated).
    """
    time: np.ndarray
    prices: np.ndarray
    q: np.ndarray
    pnl: np.ndarray
    fills: pd.DataFrame


def quotes_to_ticks(
    bid: float,
    ask: float,
    best_bid: int,
    best_ask: int,
    tick_size: float = TICK_SIZE,
) -> tuple[int, int]:
    """Converts model quotes in dollars to passive quotes in ticks.

    The bid is rounded down and the ask up, so the quotes are never more
    aggressive than the model asked for. A quote that would cross the market
    is moved back to one tick inside the opposite best price, as an exchange
    does with a post-only order: the strategy only ever provides liquidity.

    Args:
        bid: Model bid in dollars.
        ask: Model ask in dollars.
        best_bid: Best bid in the real book, in ticks.
        best_ask: Best ask in the real book, in ticks.
        tick_size: Minimum price increment in dollars.

    Returns:
        A tuple (bid, ask) in ticks.
    """
    bid_ticks = math.floor(round(bid / tick_size, 6))
    ask_ticks = math.ceil(round(ask / tick_size, 6))

    bid_ticks = min(bid_ticks, best_ask - 1)
    ask_ticks = max(ask_ticks, best_bid + 1)

    return bid_ticks, ask_ticks


def requote(
    engine: LOBReplayEngine, side: str, price: int, size: float
) -> None:
    """Moves our quote on one side to price, keeping it if already there.

    An order that is already resting at the wanted price is left alone, so it
    keeps its place in the queue. Otherwise it is cancelled and a new order of
    the full size joins the back of the queue at the new price.
    """
    existing = engine.get_synthetic_order(side)
    if existing is not None and existing.price == price:
        return

    engine.cancel_synthetic_order(side)
    engine.place_synthetic_order(side, price, size)


def run_backtest( # pylint: disable=too-many-locals
    quote_fn,
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    quote_size: float,
    tick_size: float = TICK_SIZE,
) -> BacktestResult:
    """Runs a full backtest of quote_fn against LOB data.

    The strategy keeps one bid and one ask of quote_size shares in the book.
    It re-quotes from the new mid-price whenever the clock advances, with no
    latency. Messages that share a timestamp belong to one incoming order
    sweeping the book, so quotes are not moved between them: the strategy
    cannot step out of the way of an order that has already arrived.

    Fills come from LOBReplayEngine. Inventory is passed to quote_fn in units
    of quote_size, because the AS model is written for fills of one unit.

    Args:
        quote_fn: A callable (price, q, t, T) -> (bid, ask).
        messages: LOBSTER messages from load_messages.
        orderbook: LOBSTER order book from load_orderbook, row-aligned with
            messages.
        quote_size: Shares quoted on each side.
        tick_size: Minimum price increment in dollars.

    Returns:
        BacktestResult with the mid-price, inventory and PnL paths and the
        list of fills. Inventory left at the end is marked at the last mid.
    """
    assert len(messages) == len(orderbook), (
        "messages/orderbook row count mismatch -- are these row-aligned "
        "files from the same session?"
    )

    n = len(messages)
    engine = LOBReplayEngine()

    time = messages["time"].to_numpy()
    session_start = time[0]
    session_length = time[-1] - session_start

    events = messages.assign(price_ticks=to_ticks(messages["price"], tick_size))
    best_bid = to_ticks(orderbook["bid_price_1"], tick_size).astype(int)
    best_ask = to_ticks(orderbook["ask_price_1"], tick_size).astype(int)
    prices = (best_bid + best_ask) / 2 * tick_size

    q = np.zeros(n)
    pnl = np.zeros(n)
    fills = []

    q_current = 0.0
    cash_current = 0.0
    quoted_mid = prices[0]

    for i, event in enumerate(events.itertuples(index=False)):
        fill = engine.process_message_event(event)

        if fill is not None:
            signed_size = fill.size if fill.side == "bid" else -fill.size
            fill_price = fill.price * tick_size
            q_current += signed_size
            cash_current -= signed_size * fill_price
            fills.append(
                (time[i], fill.side, fill_price, fill.size, quoted_mid)
            )

        q[i] = q_current
        pnl[i] = cash_current + q_current * prices[i]

        if i + 1 < n and time[i + 1] == time[i]:
            continue

        quoted_mid = prices[i]
        t = time[i] - session_start
        bid, ask = quote_fn(quoted_mid, q_current / quote_size, t, session_length)
        bid_ticks, ask_ticks = quotes_to_ticks(
            bid, ask, best_bid[i], best_ask[i], tick_size
        )

        requote(engine, "bid", bid_ticks, quote_size)
        requote(engine, "ask", ask_ticks, quote_size)

    fills = pd.DataFrame(
        fills, columns=["time", "side", "price", "size", "mid"]
    )

    return BacktestResult(time=time, prices=prices, q=q, pnl=pnl, fills=fills)
