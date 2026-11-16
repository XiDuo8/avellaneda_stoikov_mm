"""Visualisation helpers for single-run and Monte Carlo simulation results."""

import matplotlib.pyplot as plt
import numpy as np

from src.monte_carlo import MonteCarloResults


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

    _, axes = plt.subplots(3, 1, figsize=(10, 8))

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
        T: Terminal time of the trading session.

    Returns:
        None. Displays the figure via plt.show().
    """
    as_prices, as_q, as_pnl = as_results
    _, naive_q, naive_pnl = naive_results

    time = np.linspace(0, T, len(as_prices))

    _, axes = plt.subplots(3, 1, figsize=(10, 8))

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
    save_path: str | None = None,
) -> None:
    """Plots overlaid terminal PnL distributions for AS and naive strategies.

    Both histograms share one set of bins and are scaled to densities, so
    their heights are directly comparable. The x-axis is clipped to the 1st to
    99th percentile of the naive outcomes: the naive strategy's extreme tails
    would otherwise squash the AS distribution into a single spike. Densities
    are still computed over every run, including those outside the axis.

    Args:
        as_results: Monte Carlo results for the AS strategy.
        naive_results: Monte Carlo results for the naive strategy.
        save_path: If given, the figure is also saved to this path.

    Returns:
        None. Displays the figure via plt.show().
    """
    lower, upper = np.percentile(naive_results.terminal_pnl, [1, 99])
    bins = np.linspace(lower, upper, 61)
    bin_width = bins[1] - bins[0]

    _, ax = plt.subplots(figsize=(9, 5.5))

    for results, name in ((as_results, "AS"), (naive_results, "Naive")):
        pnl = results.terminal_pnl
        median = np.median(pnl)
        std = np.std(pnl, ddof=1)

        ax.hist(
            pnl,
            bins=bins,
            weights=np.full(len(pnl), 1 / (len(pnl) * bin_width)),
            histtype="stepfilled",
            alpha=0.45,
            label=f"{name} (median {median:.0f}, std. dev. {std:.0f})",
        )

    ax.set_xlim(lower, upper)
    ax.set_xlabel(
        "Terminal PnL (axis clipped to the central 98% of naive runs)"
    )
    ax.set_ylabel("Density")
    ax.set_title("Terminal PnL distribution: AS vs Naive")
    ax.legend()

    plt.tight_layout()
    _show(save_path)


def plot_representative_run( # pylint: disable=too-many-arguments,too-many-positional-arguments
    prices: np.ndarray,
    as_results: MonteCarloResults,
    naive_results: MonteCarloResults,
    run_idx: int,
    T: float,
    save_path: str | None = None,
) -> None:
    """Plots both strategies' inventory and PnL on one shared price path.

    Produces three stacked subplots sharing a common time axis: the mid-price
    path of the chosen run, then the overlaid inventory and mark-to-market PnL
    paths of the AS and naive strategies on that same path. Because both
    strategies see identical prices and fill randomness within a run, any
    difference between the lines is due to the quoting logic alone.

    Args:
        prices: Mid-price path of the chosen run, length n_steps + 1.
        as_results: Monte Carlo results for the AS strategy.
        naive_results: Monte Carlo results for the naive strategy.
        run_idx: Index of the run to plot, e.g. as returned by
            find_shared_representative_run.
        T: Terminal time of the trading session.
        save_path: If given, the figure is also saved to this path.

    Returns:
        None. Displays the figure via plt.show().
    """
    time = np.linspace(0, T, len(prices))

    _, axes = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, height_ratios=[1, 1.4, 1.4]
    )

    axes[0].plot(time, prices, color="0.3")
    axes[0].set_title("Mid-Price Path (shared by both strategies)")
    axes[0].set_ylabel("Price")

    panels = (
        (axes[1], as_results.inventory_paths, naive_results.inventory_paths),
        (axes[2], as_results.pnl_paths, naive_results.pnl_paths),
    )

    for ax, as_paths, naive_paths in panels:
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.plot(time, as_paths[run_idx], label="AS")
        ax.plot(time, naive_paths[run_idx], label="Naive")

    for line in axes[1].get_lines()[1:]:
        line.set_drawstyle("steps-post")

    axes[1].set_title("Inventory Over Time", loc="left")
    axes[1].set_ylabel("Inventory (q)")
    axes[1].yaxis.get_major_locator().set_params(integer=True)
    axes[1].legend(
        loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False
    )

    axes[2].set_title("Mark-to-Market PnL Over Time", loc="left")
    axes[2].set_ylabel("PnL")
    axes[2].set_xlabel("Time")

    plt.tight_layout()
    _show(save_path)


def plot_gamma_sweep(
    sweep_results: list[dict[str, float]],
    save_path: str | None = None,
) -> None:
    """Plots the risk/return tradeoff traced out by a gamma sweep.

    Produces two side-by-side panels. The left panel plots inventory variance
    against mean PnL, one labelled point per gamma value, joined in order of
    gamma to show the frontier the AS strategy moves along. The right panel
    plots the Sharpe ratio (mean PnL over its standard deviation across runs)
    for each gamma value, showing what that tradeoff does to risk-adjusted
    return.

    Args:
        sweep_results: List of dicts as returned by sweep_gamma, each
            containing "gamma", "inventory_variance", "pnl_mean", and
            "pnl_variance".
        save_path: If given, the figure is also saved to this path.

    Returns:
        None. Displays the figure via plt.show().
    """
    ordered = sorted(sweep_results, key=lambda r: r["gamma"])

    gammas = [r["gamma"] for r in ordered]
    inventory_variance = [r["inventory_variance"] for r in ordered]
    pnl_mean = [r["pnl_mean"] for r in ordered]
    sharpe = [r["pnl_mean"] / np.sqrt(r["pnl_variance"]) for r in ordered]

    _, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot(inventory_variance, pnl_mean, marker="o")

    for gamma, x, y in zip(gammas, inventory_variance, pnl_mean):
        axes[0].annotate(
            f"γ={gamma}",
            (x, y),
            textcoords="offset points",
            xytext=(8, -12),
        )

    axes[0].margins(x=0.2, y=0.12)
    axes[0].set_xlabel("Mean Inventory Variance")
    axes[0].set_ylabel("Mean Terminal PnL")
    axes[0].set_title("Inventory Risk vs PnL Tradeoff")

    positions = np.arange(len(gammas))
    bars = axes[1].bar(positions, sharpe, width=0.6)
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"γ={gamma}" for gamma in gammas])
    axes[1].margins(y=0.12)
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_title("Risk-Adjusted Return by Risk Aversion")

    plt.tight_layout()
    _show(save_path)


def _show(save_path: str | None) -> None:
    """Saves the current figure if a path is given, then displays it."""
    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()
