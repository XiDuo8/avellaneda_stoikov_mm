"""Naive symmetric quoting strategy, used as a baseline against AS."""

def naive_quotes(
    mid: float,
    delta: float,
) -> tuple[float, float]:
    """Computes symmetric fixed-offset quotes around the mid price.

    Args:
        mid: Current mid-market price of the asset.
        delta: Half-spread offset applied symmetrically to bid and ask.

    Returns:
        A tuple (bid, ask) of quoted prices, offset by delta on either side of
        mid.
    """
    return mid - delta, mid + delta


def make_naive_strategy(
    delta: float
):
    """Builds a quote_fn closure implementing the symmetric strategy.

    Args:
        delta: Half-spread offset applied symmetrically to bid and ask.

    Returns:
        A callable strategy(price, q, t, T) -> (bid, ask), suitable for
        passing to run_simulation
    """
    def strategy(
        price: float,
        q: float,   # pylint: disable=unused-argument
        t: float,   # pylint: disable=unused-argument
        T: float    # pylint: disable=unused-argument
    ) -> tuple[float, float]:
        return naive_quotes(price, delta)

    return strategy
