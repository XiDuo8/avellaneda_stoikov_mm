from src.quoting import make_as_strategy
from src.naive_quoting import make_naive_strategy
from src.monte_carlo import run_monte_carlo
from src.analysis import sharpe_ratio, find_representative_run
from src.plotting import plot_pnl_distribution, plot_representative_run

# Adjust these to match your current recalibrated params from main.py.
GAMMA = 2
SIGMA = 0.46
K = 1.5
A = 140
S0 = 100.0
MU = 0.15
T = 1.0
N_STEPS = 10000
N_RUNS = 1000

as_fn = make_as_strategy(gamma=GAMMA, sigma=SIGMA, k=K)
naive_fn = make_naive_strategy(delta=0.01)  # swap in your real delta formula

as_results, naive_results = run_monte_carlo(
    as_fn,
    naive_fn,
    s0=S0,
    mu=MU,
    sigma=SIGMA,
    T=T,
    n_steps=N_STEPS,
    A=A,
    k=K,
    n_runs=N_RUNS,
)

print("AS terminal P&L:", as_results.terminal_pnl)
print("AS Sharpe:", sharpe_ratio(as_results.terminal_pnl))
print("Naive terminal P&L:", naive_results.terminal_pnl)
print("Naive Sharpe:", sharpe_ratio(naive_results.terminal_pnl))

as_idx = find_representative_run(as_results)
naive_idx = find_representative_run(naive_results)
print(as_results.terminal_pnl[as_idx])
print(naive_results.terminal_pnl[naive_idx])
print("AS representative run index:", as_idx)
print("Naive representative run index:", naive_idx)

plot_pnl_distribution(as_results, naive_results)
plot_representative_run(as_results, naive_results, T)