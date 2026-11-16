"""Visualisation helpers for single-run and Monte Carlo simulation results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator, StrMethodFormatter

from src.backtest.backtest_analysis import spread_capture_path
from src.backtest.strategy_runner import BacktestResult
from src.market import fill_intensity
from src.monte_carlo import MonteCarloResults


def plot_results(
    prices: np.ndarray,
    q: np.ndarray,
    pnl: np.ndarray,
    T: float,
    title: str | None = None,
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

    if title:
        plt.suptitle(title)

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
    their heights are directly comparable. The x-axis covers the central 99%
    of the two strategies' outcomes taken together, so a few extreme runs
    cannot squash both distributions. Densities are still computed over every
    run, including those outside the axis.

    Args:
        as_results: Monte Carlo results for the AS strategy.
        naive_results: Monte Carlo results for the naive strategy.
        save_path: If given, the figure is also saved to this path.

    Returns:
        None. Displays the figure via plt.show().
    """
    pooled = np.concatenate(
        [as_results.terminal_pnl, naive_results.terminal_pnl]
    )
    lower, upper = np.percentile(pooled, [0.5, 99.5])
    bins = np.linspace(lower, upper, 41)
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
            label=f"{name} (median {median:.1f}, std. dev. {std:.1f})",
        )

    ax.set_xlim(lower, upper)
    ax.set_xlabel(
        "Terminal PnL (axis covers the central 99% of all runs)"
    )
    ax.set_ylabel("Density")
    ax.set_title("Terminal PnL distribution: AS vs Naive")
    ax.legend()

    plt.tight_layout()
    _show(save_path)


def plot_inventory_distribution(
    as_results: MonteCarloResults,
    naive_results: MonteCarloResults,
    save_path: str | None = None,
) -> None:
    """Plots overlaid terminal inventory distributions for both strategies.

    Inventory is a whole number of units, so each bar covers one inventory
    level and its height is the share of runs that finished there. This is
    the quantity the AS strategy controls directly, so the difference between
    the strategies is clearer here than in the PnL distribution.

    Args:
        as_results: Monte Carlo results for the AS strategy.
        naive_results: Monte Carlo results for the naive strategy.
        save_path: If given, the figure is also saved to this path.

    Returns:
        None. Displays the figure via plt.show().
    """
    as_inventory = as_results.inventory_paths[:, -1]
    naive_inventory = naive_results.inventory_paths[:, -1]

    lowest = min(as_inventory.min(), naive_inventory.min())
    highest = max(as_inventory.max(), naive_inventory.max())
    bins = np.arange(lowest - 0.5, highest + 1.5)

    _, ax = plt.subplots(figsize=(9, 5.5))

    for inventory, name in ((as_inventory, "AS"), (naive_inventory, "Naive")):
        std = np.std(inventory, ddof=1)

        ax.hist(
            inventory,
            bins=bins,
            weights=np.full(len(inventory), 1 / len(inventory)),
            histtype="stepfilled",
            alpha=0.45,
            label=f"{name} (std. dev. {std:.1f})",
        )

    ax.set_xlabel("Terminal inventory (q)")
    ax.set_ylabel("Share of runs")
    ax.set_title("Terminal inventory distribution: AS vs Naive")
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


def plot_backtest_comparison(
    as_result: BacktestResult,
    naive_result: BacktestResult,
    save_path: str | None = None,
) -> None:
    """Plots both strategies' inventory and PnL over one real trading day.

    Produces three stacked subplots sharing a clock-time axis: the mid-price,
    then the overlaid inventory and mark-to-market PnL paths of the AS and
    naive strategies. Both strategies are replayed against the same order
    flow, so any difference between the lines is due to the quoting logic.

    Args:
        as_result: Output of run_backtest for the AS strategy.
        naive_result: Output of run_backtest for the naive strategy.
        save_path: If given, the figure is also saved to this path.

    Returns:
        None. Displays the figure via plt.show().
    """
    _, axes = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, height_ratios=[1, 1.4, 1.4]
    )

    axes[0].plot(as_result.time, as_result.prices, color="0.3")
    axes[0].set_title("Mid-Price (shared by both strategies)", loc="left")
    axes[0].set_ylabel("Price ($)")

    for ax in axes[1:]:
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    for result, name in ((as_result, "AS"), (naive_result, "Naive")):
        axes[1].plot(result.time, result.q, drawstyle="steps-post", label=name)
        axes[2].plot(result.time, result.pnl, label=name)

    axes[1].set_title("Inventory Over Time", loc="left")
    axes[1].set_ylabel("Inventory (shares)")
    axes[1].legend(
        loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False
    )

    axes[2].set_title("Mark-to-Market PnL Over Time", loc="left")
    axes[2].set_ylabel("PnL ($)")
    _format_clock_axis(axes[2])

    plt.tight_layout()
    _show(save_path)


def plot_pnl_decomposition(
    as_result: BacktestResult,
    naive_result: BacktestResult,
    save_path: str | None = None,
) -> None:
    """Plots where each strategy's backtest PnL came from.

    Produces one panel per strategy. Each splits the mark-to-market PnL into
    the spread captured by quoting away from the mid-price and the PnL from
    holding inventory while the mid-price moved. The panels have separate
    y-axes because the two strategies differ in scale.

    Args:
        as_result: Output of run_backtest for the AS strategy.
        naive_result: Output of run_backtest for the naive strategy.
        save_path: If given, the figure is also saved to this path.

    Returns:
        None. Displays the figure via plt.show().
    """
    _, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    panels = ((axes[0], as_result, "AS"), (axes[1], naive_result, "Naive"))

    for ax, result, name in panels:
        captured = spread_capture_path(result)

        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.plot(result.time, captured, color="C2", label="Spread capture")
        ax.plot(
            result.time, result.pnl - captured, color="C4",
            label="Inventory PnL",
        )
        ax.plot(
            result.time, result.pnl, color="0.15", linewidth=2,
            label="Total PnL",
        )

        ax.set_title(f"{name}: Sources of PnL")
        ax.set_ylabel("PnL ($)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        _format_clock_axis(ax)

    axes[0].legend(frameon=False)

    plt.tight_layout()
    _show(save_path)


def plot_fill_intensity(
    deltas: np.ndarray,
    intensities: np.ndarray,
    A: float,
    k: float,
    save_path: str | None = None,
) -> None:
    """Plots measured fill intensity against the fitted AS fill model.

    The y-axis is logarithmic, so the model lambda(delta) = A * exp(-k * delta)
    is a straight line and the fit can be judged by eye.

    Args:
        deltas: Distances from the mid-price that were tested, in dollars.
        intensities: Measured fill intensity at each delta, in lots per second
            per side.
        A: Fitted base intensity.
        k: Fitted decay rate.
        save_path: If given, the figure is also saved to this path.

    Returns:
        None. Displays the figure via plt.show().
    """
    grid = np.linspace(0, max(deltas) * 1.05, 100)

    _, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        grid, fill_intensity(grid, A, k), color="0.3",
        label=f"Fit: A = {A:.2f}, k = {k:.1f}",
    )
    ax.plot(deltas, intensities, "o", markersize=8, label="Measured")

    ax.set_yscale("log")
    ax.set_xlabel("Quote distance from mid-price ($)")
    ax.set_ylabel("Fills per second per side (lots, log scale)")
    ax.set_title("Fill Intensity vs Quote Distance")
    ax.legend(frameon=False)

    plt.tight_layout()
    _show(save_path)


def _format_clock_axis(ax) -> None:
    """Labels an x-axis of seconds after midnight as hours and minutes."""
    ax.xaxis.set_major_locator(MultipleLocator(3600))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda s, _: f"{int(s // 3600)}:{int(s % 3600 // 60):02d}")
    )
    ax.set_xlabel("Time of day")


def _show(save_path: str | None) -> None:
    """Saves the current figure if a path is given, then displays it."""
    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()
