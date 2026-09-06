"""Unit tests for the naive symmetric quoting strategy."""

import math

from src.naive_quoting import make_naive_strategy, naive_quotes


def test_naive_quotes_known_values():
    """Bid.ask are offset symmetrically around mid by delta."""
    bid, ask = naive_quotes(mid=100.0, delta=0.5)

    assert bid == 99.5
    assert ask == 100.5


def test_naive_quotes_bid_below_ask():
    """The bid should always be strictly betlow the ask for positive delta."""
    bid, ask = naive_quotes(mid=100.0, delta=0.5)

    assert bid < ask


def test_naive_quotes_symmetric_around_mid():
    """Mid should sit exactly halfway between bid and ask."""
    mid, delta = 100.0, 1.25

    bid, ask = naive_quotes(mid, delta)

    assert math.isclose((bid + ask) / 2, mid, rel_tol=1e-9)


def test_naive_quotes_zero_delta_collapses_to_mid():
    """Zero delta should produce identical bid and ask, both equal to mid."""
    bid, ask = naive_quotes(mid=100.0, delta=0.0)

    assert bid == ask == 100.0


def test_make_naive_strategy_matches_naive_quotes():
    """The strategy closure should delegate directly to naive_quotes."""
    delta = 0.75
    strategy = make_naive_strategy(delta)

    bid, ask = strategy(price=100.0, q=0.0, t=0.5, T=1.0)
    expected_bid, expected_ask = naive_quotes(100.0, delta)

    assert bid == expected_bid
    assert ask == expected_ask


def test_make_naive_strategy_ignores_inventory_and_time():
    """The naive strategy should give identical quotes regardless of q, t, T."""
    strategy = make_naive_strategy(delta=0.5)

    bid1, ask1 = strategy(price=100.0, q=0.0, t=0.0, T=1.0)
    bid2, ask2 = strategy(price=100.0, q=50.0, t=0.9, T=1.0)

    assert bid1 == bid2
    assert ask1 == ask2
