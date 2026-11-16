"""Summary statistics, calibration and diagnostics for LOB backtests."""

import numpy as np
import pandas as pd

from src.analysis import sharpe_ratio
from src.backtest.strategy_runner import BacktestResult, run_backtest
from src.monte_carlo import MonteCarloResults
from src.naive_quoting import make_naive_strategy

SHARPE_INTERVAL = 60.0


def spread_capture_path(result: BacktestResult) -> np.ndarray:
    """Computes the cumulative spread captured after each message.

    Each fill earns the distance between the mid-price it was quoted from and
    the fill price, times the shares filled. Whatever PnL is left over comes
    from holding inventory while the mid-price moves.

    Args:
        result: Output of run_backtest.

    Returns:
        Array aligned with result.time giving the spread captured so far, in
        dollars.
    """
    fills = result.fills
    signed_size = np.where(fills["side"] == "bid", 1, -1) * fills["size"]
    captured = np.cumsum(signed_size * (fills["mid"] - fills["price"]))
    captured = np.concatenate(([0.0], captured))

    n_fills_so_far = np.searchsorted(fills["time"], result.time, side="right")

    return captured[n_fills_so_far]


def summarise_backtest(result: BacktestResult) -> dict[str, float]:
    """Summarises one backtest run.

    Args:
        result: Output of run_backtest.

    Returns:
        A dict with:
            final_pnl: Mark-to-market PnL at the end of the session.
            spread_capture: Part of final_pnl earned from quoting away from
                the mid-price.
            inventory_pnl: The rest of final_pnl, from holding inventory while
                the mid-price moved. Negative values are adverse selection.
            pnl_sharpe: Mean over standard deviation of one-minute PnL
                changes. Not annualised. NaN if the PnL never changes.
            inventory_variance: Time-weighted variance of inventory in shares.
            max_abs_inventory: Largest position held, in shares.
            n_fills: Number of fills.
            volume: Shares traded.
    """
    durations = np.diff(result.time)
    weights = durations / durations.sum()
    mean_inventory = np.sum(weights * result.q[:-1])
    inventory_variance = np.sum(weights * (result.q[:-1] - mean_inventory) ** 2)

    sample_times = np.arange(result.time[0], result.time[-1], SHARPE_INTERVAL)
    last_seen = np.searchsorted(result.time, sample_times, side="right") - 1
    pnl_increments = np.diff(result.pnl[last_seen])

    pnl_sharpe = float("nan")
    if np.ptp(pnl_increments) > 0:
        pnl_sharpe = sharpe_ratio(pnl_increments)

    final_pnl = result.pnl[-1]
    spread_capture = spread_capture_path(result)[-1]

    return {
        "final_pnl": final_pnl,
        "spread_capture": spread_capture,
        "inventory_pnl": final_pnl - spread_capture,
        "pnl_sharpe": pnl_sharpe,
        "inventory_variance": inventory_variance,
        "max_abs_inventory": np.abs(result.q).max(),
        "n_fills": len(result.fills),
        "volume": result.fills["size"].sum(),
    }


def compare_strategies(
    as_result: BacktestResult,
    naive_result: BacktestResult,
) -> dict[str, dict[str, float]]:
    """Summarises AS and naive results from backtest."""
    return {
        "as": summarise_backtest(as_result),
        "naive": summarise_backtest(naive_result),
    }


def pnl_percentile_vs_monte_carlo(
    backtest_final_pnl: float,
    mc_results: MonteCarloResults,
    quote_size: float = 1.0,
) -> float:
    """Finds where the backtest PnL sits in the simulated distribution.

    Args:
        backtest_final_pnl: Final PnL of the backtest, in dollars.
        mc_results: Monte Carlo results for the same strategy.
        quote_size: Shares per fill in the backtest. Simulated fills are one
            unit, so simulated PnL is multiplied by this to get dollars.

    Returns:
        Percentage of simulated runs with PnL at or below the backtest's.
    """
    simulated_pnl = mc_results.terminal_pnl * quote_size

    return 100 * np.mean(simulated_pnl <= backtest_final_pnl)


def estimate_sigma(time: pd.Series, mid: pd.Series) -> float:
    """Estimates absolute price volatility for data.

    Takes the last mid-price in each one-second bucket and returns the
    standard deviation of its one-second changes, in dollars per root second.
    Seconds with no messages carry the previous mid-price forward, so every
    change covers exactly one second.
    """
    resampled = pd.DataFrame({"time": time, "mid": mid})
    resampled["t_bucket"] = resampled["time"].astype(int)
    one_second_mid = resampled.groupby("t_bucket")["mid"].last()

    every_second = range(one_second_mid.index[0], one_second_mid.index[-1] + 1)
    one_second_mid = one_second_mid.reindex(every_second).ffill()
    increments = one_second_mid.diff().dropna()

    return increments.std()


def measure_fill_intensity(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    deltas: np.ndarray,
    quote_size: float,
) -> np.ndarray:
    """Measures how often a quote at each distance from the mid is filled.

    For each delta, replays the session quoting quote_size shares at delta
    either side of the mid-price and counts the fills.

    Args:
        messages: LOBSTER messages from load_messages.
        orderbook: LOBSTER order book from load_orderbook.
        deltas: Distances from the mid-price to test, in dollars.
        quote_size: Shares quoted on each side.

    Returns:
        Array with one fill intensity per delta, in lots of quote_size per
        second per side.
    """
    session_length = messages["time"].iloc[-1] - messages["time"].iloc[0]
    intensities = np.empty(len(deltas))

    for i, delta in enumerate(deltas):
        result = run_backtest(
            make_naive_strategy(delta), messages, orderbook, quote_size
        )
        lots_filled = result.fills["size"].sum() / quote_size
        intensities[i] = lots_filled / (2 * session_length)

    return intensities


def fit_fill_intensity(
    deltas: np.ndarray,
    intensities: np.ndarray,
) -> tuple[float, float]:
    """Fits the AS fill model lambda(delta) = A * exp(-k * delta).

    Taking logs gives a straight line, log(lambda) = log(A) - k * delta, which
    is fitted by least squares.

    Args:
        deltas: Distances from the mid-price, in dollars.
        intensities: Measured fill intensity at each delta.

    Returns:
        A tuple (A, k).
    """
    slope, intercept = np.polyfit(deltas, np.log(intensities), 1)

    return np.exp(intercept), -slope
