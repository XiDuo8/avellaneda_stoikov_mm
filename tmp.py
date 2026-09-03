from src.quoting import make_as_strategy
from src.naive_quoting import make_naive_strategy
from src.monte_carlo import run_monte_carlo
from src.analysis import sharpe_ratio

as_fn = make_as_strategy(gamma=1.9, sigma=0.26, k=1.5)
naive_fn = make_naive_strategy(delta=0.01)

as_results, naive_results = run_monte_carlo(
    as_fn, naive_fn,
    s0=100.0, mu=0.0, sigma=0.26, T=1.0, n_steps=252,
    A=140, k=1.5, n_runs=5,
)

print("AS terminal P&L:", as_results.terminal_pnl)
print("AS Sharpe:", sharpe_ratio(as_results.terminal_pnl))
print("Naive terminal P&L:", naive_results.terminal_pnl)
print("Naive Sharpe:", sharpe_ratio(naive_results.terminal_pnl))