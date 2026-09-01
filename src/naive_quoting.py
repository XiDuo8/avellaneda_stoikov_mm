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