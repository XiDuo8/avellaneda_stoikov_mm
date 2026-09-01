import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    prices: np.ndarray,
    q: np.ndarray,
    pnl: np.ndarray,
    T: float,
) -> None:
    """Plots the price path, inventory, and PnL over the trading session.

    Produces three stacked subplots sharing a common time axis: the simulated
    mid-price path, the mm's inventory over time, and the mark-to-market PnL
    over time.

    Args:
        prices: Simulated mid-price path, length n_steps + 1.
        q: Inventory held over time, length n_steps + 1.
        pnl: Mark-to-market PnL over time, length n_steps + 1.
        T: terminal time of the trading session.

    Returns:
        None. Displays the figure via plt.show().
    """
    time = np.linspace(0, T, len(prices))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(time, prices)
    axes[0].set_title("Mid-Price Path")
    axes[0].set_ylabel("Price")

    axes[1].plot(time, q)
    axes[1].set_title("Inventory Over Time")
    axes[1].set_ylabel("Inventory (q)")

    axes[2].plot(time, pnl)
    axes[2].set_title("Mark-to-Market PnL")
    axes[2].set_ylabel("PnL")
    axes[2].set_xlabel("Time")

    plt.tight_layout()
    plt.show()
