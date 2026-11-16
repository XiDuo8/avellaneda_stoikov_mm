"""Tests for quote conversion, re-quoting and the backtest loop."""

import pandas as pd
import pytest

from src.backtest.lob_replay import LOBReplayEngine
from src.backtest.strategy_runner import (
    quotes_to_ticks,
    requote,
    run_backtest,
)

BUY, SELL = 1, -1


def make_data(rows):
    """Builds row-aligned message and order book frames.

    Each row is (time, event_type, order_id, size, price, direction, best_bid,
    best_ask), with prices in dollars. The book columns give the top of book
    after that message.
    """
    messages = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["time", "event_type", "order_id", "size", "price", "direction"],
    )
    orderbook = pd.DataFrame(
        [row[6:] for row in rows], columns=["bid_price_1", "ask_price_1"]
    )
    return messages, orderbook


def fixed_quotes(bid, ask):
    """A quote function that always returns the same prices."""
    return lambda price, q, t, T: (bid, ask)


def pegged_quotes(delta):
    """A quote function that quotes delta either side of the mid."""
    return lambda price, q, t, T: (price - delta, price + delta)


def test_quotes_round_away_from_the_mid():
    """The bid rounds down and the ask rounds up to whole ticks."""
    assert quotes_to_ticks(99.957, 100.043, 9990, 10010) == (9995, 10005)


def test_quotes_already_on_a_tick_are_not_moved():
    """Float error must not push an on-tick price to the next tick."""
    assert 585.56 / 0.01 < 58556
    assert quotes_to_ticks(585.56, 585.71, 58550, 58580) == (58556, 58571)


def test_crossing_bid_is_moved_inside_the_best_ask():
    """A bid at or above the best ask is capped one tick below it."""
    bid, _ = quotes_to_ticks(100.50, 100.90, 9990, 10010)

    assert bid == 10009


def test_crossing_ask_is_moved_inside_the_best_bid():
    """An ask at or below the best bid is floored one tick above it."""
    _, ask = quotes_to_ticks(99.10, 99.50, 9990, 10010)

    assert ask == 9991


def test_requote_keeps_an_order_already_at_the_price():
    """Re-quoting the same price must not lose the queue position."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10000, 100)
    original = engine.get_synthetic_order("bid")

    requote(engine, "bid", 10000, 100)

    assert engine.get_synthetic_order("bid") is original


def test_requote_replaces_an_order_at_a_different_price():
    """A new price cancels the old order and places a full-size new one."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10000, 100)
    engine.get_synthetic_order("bid").size = 40

    requote(engine, "bid", 9999, 100)

    order = engine.get_synthetic_order("bid")
    assert (order.price, order.size) == (9999, 100)


def test_backtest_tracks_inventory_cash_and_pnl():
    """Fills at our quotes must flow through to inventory and PnL."""
    messages, orderbook = make_data([
        (1.0, 1, 11, 50, 99.90, BUY, 99.90, 100.10),
        (2.0, 4, 11, 30, 99.90, BUY, 99.90, 100.10),
        (3.0, 4, 12, 200, 100.10, SELL, 99.90, 100.10),
    ])

    result = run_backtest(
        fixed_quotes(99.95, 100.05), messages, orderbook, quote_size=100
    )

    assert result.q.tolist() == [0, 30, -70]
    assert result.pnl == pytest.approx([0, 1.5, 6.5])
    assert result.prices == pytest.approx([100, 100, 100])
    assert result.fills["side"].tolist() == ["bid", "ask"]
    assert result.fills["price"].tolist() == pytest.approx([99.95, 100.05])
    assert result.fills["size"].tolist() == [30, 100]
    assert result.fills["mid"].tolist() == pytest.approx([100, 100])


def test_backtest_marks_inventory_at_the_latest_mid():
    """Open inventory is revalued when the mid-price moves."""
    messages, orderbook = make_data([
        (1.0, 1, 11, 50, 99.90, BUY, 99.90, 100.10),
        (2.0, 4, 11, 100, 99.90, BUY, 99.90, 100.10),
        (3.0, 1, 12, 50, 100.40, BUY, 100.40, 100.60),
    ])

    result = run_backtest(
        fixed_quotes(99.95, 101.00), messages, orderbook, quote_size=100
    )

    assert result.q.tolist() == [0, 100, 100]
    assert result.pnl == pytest.approx([0, 5.0, 55.0])


def test_quotes_are_not_moved_inside_a_sweep():
    """Messages sharing a timestamp are one order: we cannot dodge it."""
    messages, orderbook = make_data([
        (1.0, 1, 11, 10, 99.90, BUY, 99.90, 100.10),
        (2.0, 4, 11, 10, 99.90, BUY, 99.70, 100.10),
        (2.0, 4, 99, 10, 99.70, BUY, 99.70, 100.10),
    ])

    result = run_backtest(pegged_quotes(0.25), messages, orderbook, 100)

    assert result.fills["price"].tolist() == pytest.approx([99.75])
    assert result.fills["size"].tolist() == [10]


def test_quotes_move_once_the_clock_advances():
    """The same second trade misses us if we had time to re-quote."""
    messages, orderbook = make_data([
        (1.0, 1, 11, 10, 99.90, BUY, 99.90, 100.10),
        (2.0, 4, 11, 10, 99.90, BUY, 99.70, 100.10),
        (3.0, 4, 99, 10, 99.70, BUY, 99.70, 100.10),
    ])

    result = run_backtest(pegged_quotes(0.25), messages, orderbook, 100)

    assert len(result.fills) == 0


def test_inventory_is_passed_to_the_strategy_in_lots():
    """The strategy sees inventory in units of quote_size."""
    seen = []

    def recording_quotes(price, q, t, T): # pylint: disable=unused-argument
        seen.append(q)
        return 99.95, 100.05

    messages, orderbook = make_data([
        (1.0, 1, 11, 50, 99.90, BUY, 99.90, 100.10),
        (2.0, 4, 11, 30, 99.90, BUY, 99.90, 100.10),
    ])

    run_backtest(recording_quotes, messages, orderbook, quote_size=100)

    assert seen == pytest.approx([0.0, 0.3])


def test_strategy_sees_time_since_the_session_start():
    """t starts at zero and T is the length of the session."""
    seen = []

    def recording_quotes(price, q, t, T): # pylint: disable=unused-argument
        seen.append((t, T))
        return 99.95, 100.05

    messages, orderbook = make_data([
        (34200.0, 1, 11, 50, 99.90, BUY, 99.90, 100.10),
        (34260.0, 1, 12, 50, 99.90, BUY, 99.90, 100.10),
    ])

    run_backtest(recording_quotes, messages, orderbook, quote_size=100)

    assert seen == [(0.0, 60.0), (60.0, 60.0)]


def test_mismatched_files_are_rejected():
    """Messages and order book must have one row each per event."""
    messages, orderbook = make_data([
        (1.0, 1, 11, 50, 99.90, BUY, 99.90, 100.10),
        (2.0, 1, 12, 50, 99.90, BUY, 99.90, 100.10),
    ])

    with pytest.raises(AssertionError):
        run_backtest(
            fixed_quotes(99.95, 100.05), messages, orderbook.iloc[:1], 100
        )
