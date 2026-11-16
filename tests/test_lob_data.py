"""Tests for LOBSTER loading and tick conversion."""

import numpy as np

from src.backtest.lob_data import load_messages, load_orderbook, to_ticks


def test_to_ticks_gives_exact_whole_ticks():
    """A whole-cent price must convert to exactly its tick count."""
    assert 585.56 / 0.01 != 58556
    assert to_ticks(585.56) == 58556


def test_to_ticks_makes_equal_prices_compare_equal():
    """Every whole-cent price in a range must land on a whole number."""
    cents = np.arange(57700, 58900)
    ticks = to_ticks(cents / 100)

    assert np.array_equal(ticks, cents)


def test_to_ticks_keeps_sub_penny_prices():
    """A midpoint execution price must stay between its two ticks."""
    assert to_ticks(585.335) == 58533.5


def test_to_ticks_uses_tick_size():
    """A different tick size must rescale the result."""
    assert to_ticks(10.25, tick_size=0.05) == 205


def test_load_messages_rescales_price_to_dollars(tmp_path):
    """Message prices are stored as dollars times 10,000."""
    path = tmp_path / "messages.csv"
    path.write_text("34200.5,1,11,18,5853300,1\n34201.0,4,11,18,5853300,1\n")

    messages = load_messages(str(path))

    assert list(messages.columns) == [
        "time", "event_type", "order_id", "size", "price", "direction",
    ]
    assert messages["price"].tolist() == [585.33, 585.33]
    assert messages["event_type"].tolist() == [1, 4]


def test_load_orderbook_rescales_prices_but_not_sizes(tmp_path):
    """Only the price columns of the order book are rescaled."""
    path = tmp_path / "orderbook.csv"
    path.write_text("5859400,200,5853300,18\n")

    orderbook = load_orderbook(str(path), n_levels=1)

    assert orderbook["ask_price_1"].iloc[0] == 585.94
    assert orderbook["bid_price_1"].iloc[0] == 585.33
    assert orderbook["ask_size_1"].iloc[0] == 200
    assert orderbook["bid_size_1"].iloc[0] == 18
