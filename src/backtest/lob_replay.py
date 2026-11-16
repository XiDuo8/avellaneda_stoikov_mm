"""Replays a LOBSTER message stream and fills our own resting quotes."""

from dataclasses import dataclass

LIMIT_ORDER = 1
VISIBLE_EXECUTION = 4
HIDDEN_EXECUTION = 5

# Arrival index given to real orders whose submission is not in the message
# file: orders already resting at the open, or submitted outside the visible
# levels. They are treated as ahead of us in the queue, which can only delay
# our fills.
BEFORE_SESSION = -1


@dataclass
class SyntheticOrder:
    """Our own resting order, tracked separately from the real book.

    Attributes:
        side: "bid" or "ask".
        price: Limit price in ticks.
        size: Shares still unfilled.
        placed_seq: Number of messages processed before the order was placed.
            Real orders that arrive in a later message are behind it.
    """
    side: str
    price: int
    size: float
    placed_seq: int


@dataclass
class Fill:
    """A fill of one of our synthetic orders.

    Attributes:
        side: "bid" if we bought, "ask" if we sold.
        price: Our limit price in ticks, which is the price we trade at.
        size: Shares filled.
    """
    side: str
    price: int
    size: float


class LOBReplayEngine:
    """Replays LOBSTER messages and fills synthetic orders by price-time
    priority.

    The real order flow is replayed unchanged: our orders are assumed not to
    affect it. A synthetic order fills when a real resting order with lower
    priority than ours is executed, because the incoming order would have
    reached ours first.
    """

    def __init__(self):
        self._arrival_seq: dict[int, int] = {}
        self._n_processed = 0
        self._synthetic_orders: dict[str, SyntheticOrder | None] = {
            "bid": None,
            "ask": None,
        }

    def place_synthetic_order(
        self, side: str, price: int, size: float
    ) -> None:
        """Places a synthetic resting order at the back of the queue.

        Args:
            side: "bid" or "ask".
            price: Limit price in ticks.
            size: Number of shares to quote.
        """
        assert self._synthetic_orders[side] is None, (
            f"Synthetic order already resting on {side}; cancel it first."
        )

        self._synthetic_orders[side] = SyntheticOrder(
            side=side,
            price=price,
            size=size,
            placed_seq=self._n_processed,
        )

    def cancel_synthetic_order(self, side: str) -> SyntheticOrder | None:
        """Cancels the synthetic order resting on the given side, if any."""
        existing = self._synthetic_orders[side]
        self._synthetic_orders[side] = None
        return existing

    def get_synthetic_order(self, side: str) -> SyntheticOrder | None:
        """Returns the synthetic order resting on the given side, if any."""
        return self._synthetic_orders[side]

    def process_message_event(self, event) -> Fill | None:
        """Processes one LOBSTER message and checks our orders for a fill.

        Args:
            event: One message row with attributes event_type, order_id, size,
                direction and price_ticks. For executions, direction is the
                side of the resting order that was hit (1 = buy, -1 = sell).

        Returns:
            The Fill if this event filled one of our orders, else None.
        """
        seq = self._n_processed
        self._n_processed += 1

        if event.event_type == LIMIT_ORDER:
            self._arrival_seq[event.order_id] = seq
            return None

        if event.event_type not in (VISIBLE_EXECUTION, HIDDEN_EXECUTION):
            return None

        side = "bid" if event.direction == 1 else "ask"
        order = self._synthetic_orders[side]

        if order is None or not self._has_priority(order, event):
            return None

        fill_size = min(order.size, event.size)
        order.size -= fill_size
        if order.size <= 0:
            self._synthetic_orders[side] = None

        return Fill(side=side, price=order.price, size=fill_size)

    def _has_priority(self, order: SyntheticOrder, event) -> bool:
        """Checks whether our order ranks ahead of the executed real order.

        Priority is by price, then displayed before hidden, then by arrival.
        """
        sign = 1 if order.side == "bid" else -1
        price_edge = sign * (order.price - event.price_ticks)

        if price_edge != 0:
            return price_edge > 0

        if event.event_type == HIDDEN_EXECUTION:
            return True

        arrival = self._arrival_seq.get(event.order_id, BEFORE_SESSION)
        return arrival >= order.placed_seq
