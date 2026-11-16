"""Tests for the price-time priority fill logic in LOBReplayEngine."""

from collections import namedtuple

import pytest

from src.backtest.lob_replay import Fill, LOBReplayEngine

Event = namedtuple("Event", "event_type order_id size price_ticks direction")

BUY, SELL = 1, -1


def new_order(order_id, price, direction, size=100):
    """A real limit order joining the book."""
    return Event(1, order_id, size, price, direction)


def execution(order_id, price, direction, size=100):
    """A visible real order being executed."""
    return Event(4, order_id, size, price, direction)


def hidden_execution(price, direction, size=100):
    """A hidden real order being executed. LOBSTER gives these order id 0."""
    return Event(5, 0, size, price, direction)


def test_bid_fills_when_trade_happens_at_a_worse_price():
    """A sell order that reached a lower bid would have hit ours first."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10005, 100)

    fill = engine.process_message_event(execution(7, 10000, BUY, size=40))

    assert fill == Fill(side="bid", price=10005, size=40)


def test_bid_does_not_fill_when_trade_happens_at_a_better_price():
    """A bid above ours has priority, so its execution does not reach us."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10000, 100)

    assert engine.process_message_event(execution(7, 10005, BUY)) is None


def test_ask_fills_when_trade_happens_at_a_worse_price():
    """A buy order that reached a higher ask would have lifted ours first."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("ask", 10000, 100)

    fill = engine.process_message_event(execution(7, 10005, SELL, size=40))

    assert fill == Fill(side="ask", price=10000, size=40)


def test_ask_does_not_fill_when_trade_happens_at_a_better_price():
    """An ask below ours has priority, so its execution does not reach us."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("ask", 10005, 100)

    assert engine.process_message_event(execution(7, 10000, SELL)) is None


def test_order_that_arrived_before_us_is_ahead_in_the_queue():
    """At our price, an order that was there first executes before us."""
    engine = LOBReplayEngine()
    engine.process_message_event(new_order(7, 10000, BUY))
    engine.place_synthetic_order("bid", 10000, 100)

    assert engine.process_message_event(execution(7, 10000, BUY)) is None


def test_order_that_arrived_after_us_is_behind_in_the_queue():
    """At our price, an execution of a later order means we were hit."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10000, 100)
    engine.process_message_event(new_order(7, 10000, BUY))

    fill = engine.process_message_event(execution(7, 10000, BUY, size=60))

    assert fill == Fill(side="bid", price=10000, size=60)


def test_order_never_seen_arriving_is_treated_as_ahead():
    """Orders resting before the session started keep their priority."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10000, 100)

    assert engine.process_message_event(execution(99, 10000, BUY)) is None


def test_displayed_order_has_priority_over_hidden_at_same_price():
    """A hidden execution at our price means the trade reached us first."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("ask", 10000, 100)

    fill = engine.process_message_event(hidden_execution(10000, SELL, 25))

    assert fill == Fill(side="ask", price=10000, size=25)


def test_hidden_order_at_better_price_has_priority():
    """A hidden order inside our quote trades before we do."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("ask", 10000, 100)

    assert engine.process_message_event(hidden_execution(9999.5, SELL)) is None


def test_execution_on_the_other_side_does_not_fill():
    """A trade against resting asks cannot fill our bid."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10005, 100)

    assert engine.process_message_event(execution(7, 10000, SELL)) is None


def test_partial_fill_leaves_the_remainder_resting():
    """A small trade fills part of the order and the rest stays in place."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10005, 100)

    engine.process_message_event(execution(7, 10000, BUY, size=30))

    remaining = engine.get_synthetic_order("bid")
    assert remaining.size == 70
    assert remaining.price == 10005


def test_fill_is_capped_at_our_remaining_size():
    """We cannot be filled for more than we are quoting."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10005, 100)

    first = engine.process_message_event(execution(7, 10000, BUY, size=80))
    second = engine.process_message_event(execution(8, 10000, BUY, size=80))

    assert (first.size, second.size) == (80, 20)
    assert engine.get_synthetic_order("bid") is None


def test_no_fill_without_a_resting_order():
    """Executions with nothing of ours in the book return None."""
    engine = LOBReplayEngine()

    assert engine.process_message_event(execution(7, 10000, BUY)) is None


@pytest.mark.parametrize("event_type", [1, 2, 3, 6, 7])
def test_only_executions_can_fill(event_type):
    """New orders, cancellations, crosses and halts never fill us."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10005, 100)

    event = Event(event_type, 7, 100, 10000, BUY)

    assert engine.process_message_event(event) is None
    assert engine.get_synthetic_order("bid").size == 100


def test_cancel_returns_and_removes_the_order():
    """Cancelling hands back the order and frees the side."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("ask", 10010, 100)

    cancelled = engine.cancel_synthetic_order("ask")

    assert cancelled.price == 10010
    assert engine.get_synthetic_order("ask") is None
    assert engine.cancel_synthetic_order("ask") is None


def test_placing_a_second_order_on_one_side_is_an_error():
    """Only one synthetic order may rest on each side."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10000, 100)

    with pytest.raises(AssertionError):
        engine.place_synthetic_order("bid", 9999, 100)


def test_replacing_an_order_sends_it_to_the_back_of_the_queue():
    """An order placed again after a real order arrives is behind it."""
    engine = LOBReplayEngine()
    engine.place_synthetic_order("bid", 10000, 100)
    engine.process_message_event(new_order(7, 10000, BUY))

    engine.cancel_synthetic_order("bid")
    engine.place_synthetic_order("bid", 10000, 100)

    assert engine.process_message_event(execution(7, 10000, BUY)) is None
