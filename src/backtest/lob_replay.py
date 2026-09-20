from collections import deque
from dataclasses import dataclass

@dataclass
class RestingOrder:
    """One real resting order at a price level, as seen in LOBSTER message
    stream. """
    order_id: int
    size: float

@dataclass
class SyntheticOrder:
    """Our own resting order, tracked seperately from the real book."""
    side: str
    price: float
    size: float
    volume_ahead: float
    placed_at: float

class LOBReplayEngine:
    """Reconstructs a LOBSTER order book's evolution and tracks fills against
    synthetic resting orders under queue-position-aware logic.
    """

    def __init__(self):
        self._queues: dict[str, dict[float, deque]] = {"bid": {}, "ask": {}}
        self._order_price: dict[int, float] = {}
        self._synthetic_orders: dict[str, SyntheticOrder | None] = {
            "bid": None,
            "ask": None,
        }

    def process_message_event(self, event):
        """Processes one LOBSTER message row, updating real queues checking
        whether a resting synthetic order fills.

        Returns the fill amount if a synthetic order was filled by this event,
        else None.
        """
        side = "bid" if event.direction == 1 else "ask"

        if event.event_type == 1:
            resting = RestingOrder(order_id=event.order_id, size=event.size)
            self._queues[side].setdefault(event.price, deque()).append(resting)
            self._order_price[event.order_id] = event.price

        elif event.event_type in (2, 3):
            self._handle_cancellation(side, event)
            return None

        elif event.event_type in (4, 5):
            return self._handle_execution(side, event)

        return None

    def _handle_cancellation(self, side: str, event) -> None:
        """Handles a partial (type 2) or full (type 3) cancellation."""
        price = self._order_price.get(event.order_id)
        if price is None:
            return

        queue = self._queues[side].get(price)
        if queue is None:
            return

        ahead_of_cancelled = 0.0
        target = None
        for resting in queue:
            if resting.order_id == event.order_id:
                target = resting
                break
            ahead_of_cancelled += resting.size

        if target is None:
            return

        removed = target.size if event.event_type == 3 else min(event.size, target.size)

        synthetic = self._synthetic_orders[side]
        if synthetic is not None and synthetic.price == price:
            if ahead_of_cancelled < synthetic.volume_ahead:
                synthetic.volume_ahead = max(0.0, synthetic.volume_ahead - removed)

        target.size -= removed
        if target.size <= 0:
            queue.remove(target)
            del self._order_price[event.order_id]

    def _handle_execution(self, side: str, event) -> float | None:
        """Handles a visible (type 4) or hidden (type 5) execution."""
        queue = self._queues[side].get(event.price, deque())
        remaining = event.size
        fill_amount = 0.0

        synthetic = self._synthetic_orders[side]
        synthetic_here = synthetic is not None and synthetic.price == event.price

        if synthetic_here and synthetic.volume_ahead > 0:
            ahead_consumed = min(remaining, synthetic.volume_ahead)
            self._consume_from_front(queue, ahead_consumed)
            synthetic.volume_ahead -= ahead_consumed
            remaining -= ahead_consumed

        if synthetic_here and synthetic.volume_ahead <= 0 and remaining > 0:
            fill_amount = min(remaining, synthetic.size)
            synthetic.size -= fill_amount
            remaining -= fill_amount
            if synthetic.size <= 0:
                self._synthetic_orders[side] = None

        if remaining > 0:
            self._consume_from_front(queue, remaining)

        return fill_amount if fill_amount > 0 else None

    @staticmethod
    def _consume_from_front(queue: deque, amount: float) -> None:
        while amount > 0 and queue:
            front = queue[0]
            take = min(amount, front.size)
            front.size -= take
            amount -= take
            if front.size <= 0:
                queue.popleft()
