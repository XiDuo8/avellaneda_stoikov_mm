"""Tests for backtest summary statistics and calibration."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtest_analysis import (
    compare_strategies,
    estimate_sigma,
    fit_fill_intensity,
    measure_fill_intensity,
    pnl_percentile_vs_monte_carlo,
    spread_capture_path,
    summarise_backtest,
)
from src.backtest.strategy_runner import BacktestResult
from src.monte_carlo import MonteCarloResults


@pytest.fixture(name="result")
def fixture_result() -> BacktestResult:
    """A four-minute run: buy 100 at 99.95, then sell 100 at 100.25.

    The mid-price is 100.00 for the buy and 100.20 for the sell, so each fill
    captures 5 cents a share and the inventory gains 20 cents a share.
    """
    fills = pd.DataFrame({
        "time": [60.0, 180.0],
        "side": ["bid", "ask"],
        "price": [99.95, 100.25],
        "size": [100, 100],
        "mid": [100.00, 100.20],
    })
    return BacktestResult(
        time=np.array([0.0, 60.0, 120.0, 180.0, 240.0]),
        prices=np.array([100.0, 100.0, 100.2, 100.2, 100.2]),
        q=np.array([0.0, 100.0, 100.0, 0.0, 0.0]),
        pnl=np.array([0.0, 5.0, 25.0, 30.0, 30.0]),
        fills=fills,
    )


def test_spread_capture_path_steps_up_at_each_fill(result):
    """Captured spread accumulates fill by fill."""
    path = spread_capture_path(result)

    assert path == pytest.approx([0.0, 5.0, 5.0, 10.0, 10.0])


def test_spread_capture_path_is_zero_without_fills(result):
    """A run with no fills captures nothing."""
    result.fills = result.fills.iloc[:0]

    assert spread_capture_path(result).tolist() == [0.0] * 5


def test_summary_splits_pnl_into_spread_and_inventory(result):
    """Spread capture and inventory PnL must add up to the final PnL."""
    summary = summarise_backtest(result)

    assert summary["final_pnl"] == pytest.approx(30.0)
    assert summary["spread_capture"] == pytest.approx(10.0)
    assert summary["inventory_pnl"] == pytest.approx(20.0)


def test_summary_counts_fills_and_volume(result):
    """Fills and traded shares come from the fill list."""
    summary = summarise_backtest(result)

    assert summary["n_fills"] == 2
    assert summary["volume"] == 200
    assert summary["max_abs_inventory"] == 100


def test_summary_inventory_variance_is_time_weighted(result):
    """Inventory is 100 for half the session and 0 for the other half."""
    summary = summarise_backtest(result)

    assert summary["inventory_variance"] == pytest.approx(2500.0)


def test_summary_inventory_variance_ignores_message_count():
    """A burst of messages must not change the time-weighted variance."""
    fills = pd.DataFrame(columns=["time", "side", "price", "size", "mid"])
    sparse = BacktestResult(
        time=np.array([0.0, 100.0, 200.0]),
        prices=np.full(3, 100.0),
        q=np.array([0.0, 100.0, 100.0]),
        pnl=np.zeros(3),
        fills=fills,
    )
    busy = BacktestResult(
        time=np.array([0.0, 1.0, 2.0, 3.0, 100.0, 200.0]),
        prices=np.full(6, 100.0),
        q=np.array([0.0, 0.0, 0.0, 0.0, 100.0, 100.0]),
        pnl=np.zeros(6),
        fills=fills,
    )

    sparse_variance = summarise_backtest(sparse)["inventory_variance"]
    busy_variance = summarise_backtest(busy)["inventory_variance"]

    assert busy_variance == pytest.approx(sparse_variance)


def test_summary_sharpe_uses_one_minute_pnl_changes(result):
    """PnL sampled each minute is 0, 5, 25, 30: changes of 5, 20, 5."""
    summary = summarise_backtest(result)
    increments = np.array([5.0, 20.0, 5.0])

    expected = increments.mean() / increments.std(ddof=1)
    assert summary["pnl_sharpe"] == pytest.approx(expected)


def test_summary_sharpe_is_nan_when_pnl_never_changes(result):
    """A flat PnL has no Sharpe ratio."""
    result.pnl = np.zeros(5)

    assert np.isnan(summarise_backtest(result)["pnl_sharpe"])


def test_compare_strategies_summarises_both(result):
    """Both strategies are summarised under their own key."""
    comparison = compare_strategies(result, result)

    assert set(comparison) == {"as", "naive"}
    assert comparison["as"] == summarise_backtest(result)


def test_pnl_percentile_scales_simulated_pnl_to_dollars():
    """Simulated unit-size PnL is multiplied by the backtest quote size."""
    mc_results = MonteCarloResults(
        terminal_pnl=np.array([1.0, 2.0, 3.0, 4.0]),
        inventory_paths=np.zeros((4, 2)),
        pnl_paths=np.zeros((4, 2)),
    )

    assert pnl_percentile_vs_monte_carlo(2.5, mc_results) == 50.0
    assert pnl_percentile_vs_monte_carlo(250.0, mc_results, 100) == 50.0
    assert pnl_percentile_vs_monte_carlo(2.5, mc_results, 100) == 0.0


def test_estimate_sigma_matches_known_changes():
    """Mid-price changes of +1, -1, +1, -1 have a known spread."""
    time = pd.Series([0.2, 1.5, 2.1, 3.9, 4.4])
    mid = pd.Series([10.0, 11.0, 10.0, 11.0, 10.0])

    expected = np.std([1.0, -1.0, 1.0, -1.0], ddof=1)
    assert estimate_sigma(time, mid) == pytest.approx(expected)


def test_estimate_sigma_uses_last_mid_in_each_second():
    """Only the final mid-price within a second counts."""
    time = pd.Series([0.1, 0.9, 1.1, 1.9, 2.5])
    mid = pd.Series([50.0, 10.0, 99.0, 12.0, 11.0])

    expected = np.std([2.0, -1.0], ddof=1)
    assert estimate_sigma(time, mid) == pytest.approx(expected)


def test_estimate_sigma_fills_quiet_seconds():
    """A second with no messages counts as a zero change, not a skip."""
    time = pd.Series([0.5, 1.5, 4.5])
    mid = pd.Series([10.0, 11.0, 13.0])

    expected = np.std([1.0, 0.0, 0.0, 2.0], ddof=1)
    assert estimate_sigma(time, mid) == pytest.approx(expected)


def test_fit_fill_intensity_recovers_known_parameters():
    """Points on an exact exponential must give back A and k."""
    deltas = np.array([0.02, 0.05, 0.10, 0.20])
    intensities = 0.6 * np.exp(-30.0 * deltas)

    A, k = fit_fill_intensity(deltas, intensities)

    assert A == pytest.approx(0.6)
    assert k == pytest.approx(30.0)


def test_measure_fill_intensity_counts_lots_per_second_per_side():
    """One 50-share fill in a 10 second session at the nearer quote only."""
    messages = pd.DataFrame(
        [
            (0.0, 1, 11, 50, 99.90, 1),
            (5.0, 4, 11, 50, 99.90, 1),
            (10.0, 1, 12, 50, 99.90, 1),
        ],
        columns=["time", "event_type", "order_id", "size", "price", "direction"],
    )
    orderbook = pd.DataFrame({
        "bid_price_1": [99.90] * 3,
        "ask_price_1": [100.10] * 3,
    })

    intensities = measure_fill_intensity(
        messages, orderbook, np.array([0.05, 0.20]), quote_size=100
    )

    assert intensities == pytest.approx([0.5 / 20, 0.0])
