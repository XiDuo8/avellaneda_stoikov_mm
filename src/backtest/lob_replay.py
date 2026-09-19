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
