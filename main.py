from src.market import simulate_gbm

path = simulate_gbm(s0=100, mu=0, sigma=0.3, T=1.0, n_steps=100000000, seed=67)
print(path)