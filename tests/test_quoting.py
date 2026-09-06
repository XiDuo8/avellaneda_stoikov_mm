"""Unit tests for the AS math in src.quoting."""

import math

from src.quoting import compute_quotes, optimal_spread, reservation_price


def test_reservation_price_with_zero_inventory_equals_mid(config):
    """With no inventory, the reservation price should equal the mid price."""
    r = reservation_price(s=100.0, q=0.0, t=0.5, gamma=1.0, config=config)

    assert r == 100.0


def test_reservation_price_known_value(config):
    """Reservation price matches the closed-form AS formula for known inputs."""
    r = reservation_price(s=100.0, q=2.0, t=0.5, gamma=1.0, config=config)

    assert math.isclose(r, 99.96, rel_tol=1e-9)


def test_reservation_price_shifts_down_when_long(config):
    """Positive inventory (long) should shift the reservation price below mid."""
    r = reservation_price(s=100.0, q=5.0, t=0.5, gamma=1.0, config=config)

    assert r < 100.0


def test_reservation_price_shifts_up_when_short(config):
    """Negative inventory (short) should shift the reservation price above mid."""
    r = reservation_price(s=100.0, q=-5.0, t=0.5, gamma=1.0, config=config)

    assert r > 100.0


def test_optimal_spread_known_value(config):
    """Optimal spread matches the closed-form AS formula for known inputs."""
    width = optimal_spread(t=0.5, gamma=1.0, config=config)

    assert math.isclose(width, 1.0416512475319812, rel_tol=1e-9)


def test_optimal_spread_shrinks_as_time_runs_out(config):
    """Spread should narrow as t approaches T (less inventory risk remaining)."""
    early_width = optimal_spread(t=0.1, gamma=1.0, config=config)
    late_width = optimal_spread(t=0.9, gamma=1.0, config=config)

    assert late_width < early_width


def test_optimal_spread_is_positive(config):
    """Spread width should always be positive for valid inputs."""
    width = optimal_spread(t=0.5, gamma=1.0, config=config)

    assert width > 0


def test_compute_quotes_bid_below_ask(config):
    """The bid should always be strictly below the ask."""
    bid, ask = compute_quotes(s=100.0, q=0.0, t=0.5, gamma=1.0, config=config)

    assert bid < ask


def test_compute_quotes_known_values(config):
    """Bid/ask match hand-verified values derived from the AS formulas."""
    bid, ask = compute_quotes(s=100.0, q=2.0, t=0.5, gamma=1.0, config=config)

    assert math.isclose(bid, 99.439174376234, rel_tol=1e-9)
    assert math.isclose(ask, 100.48082562376598, rel_tol=1e-9)


def test_compute_quotes_midpoint_equals_reservation_price(config):
    """The bid/ask midpoint should equal the reservation price, by construction."""
    s, q, t, gamma = 100.0, 3.0, 0.4, 1.5

    bid, ask = compute_quotes(s, q, t, gamma, config)
    r = reservation_price(s, q, t, gamma, config)

    assert math.isclose((bid + ask) / 2, r, rel_tol=1e-9)


def test_compute_quotes_width_equals_optimal_spread(config):
    """The bid/ask width should equal optimal_spread's total width, by construction."""
    s, q, t, gamma = 100.0, 3.0, 0.4, 1.5

    bid, ask = compute_quotes(s, q, t, gamma, config)
    width = optimal_spread(t, gamma, config)

    assert math.isclose(ask - bid, width, rel_tol=1e-9)
