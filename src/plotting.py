import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.monte_carlo import MonteCarloResults
from src.analysis import find_representative_run

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

def plot_comparison(
    as_results: tuple[np.ndarray, np.ndarray, np.ndarray],
    naive_results: tuple[np.ndarray, np.ndarray, np.ndarray],
    T: float,
) -> None:
    """Plots AS vs naive strategy results side by side for comparison.

    Produces three stacked subplots sharing a common time axis: the shared
    mid-price path, and overlaid inventory and mark-to-market PnL paths for the
    AS and naive strategies.

    Args:
        as_results: Tuple (prices, q, pnl) from run_simulation for the AS
            strategy.
        naive_results: Tuple (prices, q, pnl) from run_simulation for the naive
            fixed-spread strategy.
        T: Termainl time of the trading session.

    Returns:
        None. Displays the figure via plt.show().
    """
    as_prices, as_q, as_pnl = as_results
    naive_prices, naive_q, naive_pnl = naive_results

    time = np.linspace(0, T, len(as_prices))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(time, as_prices)
    axes[0].set_title("Mid-Price Path")
    axes[0].set_ylabel("Price")

    axes[1].plot(time, as_q, label="AS")
    axes[1].plot(time, naive_q, label="Naive")
    axes[1].set_title("Inventory Over Time")
    axes[1].set_ylabel("Inventory (q)")
    axes[1].legend()

    axes[2].plot(time, as_pnl, label="AS")
    axes[2].plot(time, naive_pnl, label="Naive")
    axes[2].set_title("PnL Over Time")
    axes[2].set_ylabel("PnL")
    axes[2].set_xlabel("Time")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_pnl_distribution(
    as_results: MonteCarloResults,
    naive_results: MonteCarloResults,
) -> None:
    """Plots overlaid termanal PnL distributions for AS and naive strategies.

    Args:
        as_results (MonteCarloResults): MonteCarlo results for the AS strategy.
        naive_results (MonteCarloResults): Monte Carlo results for the naive
            strategy.
    """
    sns.histplot(as_results.terminal_pnl, kde=True, label="AS", alpha=0.5)
    sns.histplot(naive_results.terminal_pnl, kde=True, label="Naive", alpha=0.5)
    plt.xlabel("Terminal PnL")
    plt.ylabel("Frequency")
    plt.title("Terminal PnL distribution: AS vs Naive")
    plt.legend()
    plt.show()

def plot_representative_run(
    as_results: MonteCarloResults,
    naive_results: MonteCarloResults,
    T: float,
) -> None:
    """Plots inventory and PnL paths for each's median-representative run.

    Args:
        as_results: Monte Carlo results for the AS strategy.
        naive_results: Monte Carlo results for the naive strategy.
    """
    as_idx = find_representative_run(as_results)
    naive_idx = find_representative_run(naive_results)

    as_q = as_results.inventory_paths[as_idx]
    as_pnl = as_results.pnl_paths[as_idx]

    naive_q = naive_results.inventory_paths[naive_idx]
    naive_pnl = naive_results.pnl_paths[naive_idx]

    time = np.linspace(0, T, len(as_q))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(time, as_q, label="AS")
    axes[0].plot(time, naive_q, label="Naive")
    axes[0].set_title("Inventory Over Time")
    axes[0].set_ylabel("Inventory (q)")
    axes[0].legend()

    axes[1].plot(time, as_pnl, label="AS")
    axes[1].plot(time, naive_pnl, label="Naive")
    axes[1].set_title("PnL Over Time")
    axes[1].set_ylabel("PnL")
    axes[1].set_xlabel("Time")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_gamma_sweep(
    sweep_results: list[dict[str, float]],
) -> None:
    """Plots the inventory-risk vs PnL tradeoff against a gamma sweep.

    Produces a scatter plot with inventory variance on the x-axis and PnL mean
    on the y-axis, one point per gamma value. Point size encodes PnL variance,
    and each point is annotated with its gamma value, so the plot shows how
    increasing risk-aversion (gamma) trades inventory risk against both the
    level and variability of PnL.

    Args:
        sweep_results: List of dicts as returned by sweep_gamma, each
            containing "gamma", "inventory_variance", "pnl_mean", and
            "pnl_variance".

    Returns:
        None. Displays the figure via plt.show().
    """
    gammas = [r["gamma"] for r in sweep_results]
    inventory_variance = [r["inventory_variance"] for r in sweep_results]
    pnl_mean = [r["pnl_mean"] for r in sweep_results]
    pnl_variance = [r["pnl_variance"] for r in sweep_results]

    max_pnl_variance = max(pnl_variance)
    point_sizes = [20 + 180 * (v / max_pnl_variance) for v in pnl_variance]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(inventory_variance, pnl_mean, s=point_sizes)

    for gamma, x, y in zip(gammas, inventory_variance, pnl_mean):
        ax.annotate(f"γ={gamma}",
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
        )

    ax.set_xlabel("Inventory Variance")
    ax.set_ylabel("PnL Mean")
    ax.set_title("Gamma Sweep: Inventory Risk vs PnL Tradeoff")

    plt.tight_layout()
    plt.show()