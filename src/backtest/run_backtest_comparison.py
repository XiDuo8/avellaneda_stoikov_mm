"""Runs the AS vs naive comparison on one day of LOBSTER order book data."""

import numpy as np

from src.backtest.backtest_analysis import (
    compare_strategies,
    estimate_sigma,
    fit_fill_intensity,
    measure_fill_intensity,
    pnl_percentile_vs_monte_carlo,
)
from src.backtest.lob_data import load_messages, load_orderbook
from src.backtest.strategy_runner import run_backtest
from src.config import SimulationConfig
from src.monte_carlo import single_monte_carlo
from src.naive_quoting import make_naive_strategy
from src.plotting import (
    plot_backtest_comparison,
    plot_fill_intensity,
    plot_pnl_decomposition,
)
from src.quoting import make_as_strategy, optimal_spread

MESSAGE_FILE = "assets/lobster/AAPL_2012-06-21_34200000_57600000_message_5.csv"
ORDERBOOK_FILE = "assets/lobster/AAPL_2012-06-21_34200000_57600000_orderbook_5.csv"
GRAPH_DIR = "assets/graphs"

GAMMA = 0.001
QUOTE_SIZE = 100
CALIBRATION_DELTAS = np.arange(0.02, 0.21, 0.02)
N_SIMULATED_RUNS = 200

ROWS = (
    ("Final PnL ($)", "final_pnl", ",.0f"),
    ("  Spread capture ($)", "spread_capture", ",.0f"),
    ("  Inventory PnL ($)", "inventory_pnl", ",.0f"),
    ("Sharpe (1-minute PnL)", "pnl_sharpe", ".3f"),
    ("Inventory std. dev. (shares)", "inventory_std", ",.0f"),
    ("Max abs inventory (shares)", "max_abs_inventory", ",.0f"),
    ("Fills", "n_fills", ",.0f"),
    ("Volume (shares)", "volume", ",.0f"),
)


def run_lob_backtest(): # pylint: disable=too-many-locals
    """Backtests AS and naive quoting on real data and prints a comparison.

    Volatility and the fill-intensity parameters A and k are estimated from
    the same day of data, so the AS strategy runs with no hand-picked market
    parameters. The naive strategy quotes at the AS strategy's opening
    half-spread. The AS backtest PnL is then compared with what the AS model
    itself predicts for those parameters.
    """
    messages = load_messages(MESSAGE_FILE)
    orderbook = load_orderbook(ORDERBOOK_FILE)

    session_length = messages["time"].iloc[-1] - messages["time"].iloc[0]

    mid = (orderbook["ask_price_1"] + orderbook["bid_price_1"]) / 2
    sigma = estimate_sigma(messages["time"], mid)

    intensities = measure_fill_intensity(
        messages, orderbook, CALIBRATION_DELTAS, QUOTE_SIZE
    )
    A, k = fit_fill_intensity(CALIBRATION_DELTAS, intensities)

    config = SimulationConfig(
        s0=mid.iloc[0],
        mu=0.0,
        sigma=sigma / mid.iloc[0],
        T=session_length,
        n_steps=int(session_length),
        A=A,
        k=k,
    )
    delta = optimal_spread(0, GAMMA, config) / 2

    as_fn = make_as_strategy(GAMMA, config)
    naive_fn = make_naive_strategy(delta)

    as_result = run_backtest(as_fn, messages, orderbook, QUOTE_SIZE)
    naive_result = run_backtest(naive_fn, messages, orderbook, QUOTE_SIZE)

    comparison = compare_strategies(as_result, naive_result)
    for stats in comparison.values():
        stats["inventory_std"] = np.sqrt(stats["inventory_variance"])

    print()
    print("LOBSTER backtest")
    print(f"Volatility: {sigma:.4f} dollars per root second")
    print(f"Fill intensity fit: A = {A:.3f} lots per second, k = {k:.1f}")
    print(f"Half-spread at the open: {delta:.4f} dollars (both strategies)")
    print()
    print(f"{'':<30}{'AS':>12}{'Naive':>12}")
    for label, key, fmt in ROWS:
        as_value = format(comparison["as"][key], fmt)
        naive_value = format(comparison["naive"][key], fmt)
        print(f"{label:<30}{as_value:>12}{naive_value:>12}")

    # The simulator fills one unit at a time, so one simulated unit stands
    # for one lot of QUOTE_SIZE shares.
    mc_results = single_monte_carlo(as_fn, config, n_runs=N_SIMULATED_RUNS)
    simulated_pnl = mc_results.terminal_pnl * QUOTE_SIZE
    low, median, high = np.percentile(simulated_pnl, [5, 50, 95])

    as_final_pnl = as_result.pnl[-1]
    percentile = pnl_percentile_vs_monte_carlo(
        as_final_pnl, mc_results, QUOTE_SIZE
    )

    print()
    print(
        f"AS model's own simulation: median PnL {median:,.0f} "
        f"(5th to 95th percentile {low:,.0f} to {high:,.0f})"
    )
    print(
        f"AS real-data final PnL ({as_final_pnl:,.0f}) sits at the "
        f"{percentile:.1f}th percentile of that simulated distribution."
    )

    plot_fill_intensity(
        CALIBRATION_DELTAS, intensities, A, k,
        f"{GRAPH_DIR}/fill_intensity.png",
    )
    plot_backtest_comparison(
        as_result, naive_result, f"{GRAPH_DIR}/backtest_comparison.png"
    )
    plot_pnl_decomposition(
        as_result, naive_result, f"{GRAPH_DIR}/backtest_pnl_decomposition.png"
    )
