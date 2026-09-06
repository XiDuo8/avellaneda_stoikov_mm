"""Configuration objects bundling simulation parameters."""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Market and simulation parameters shared across a run.

    Attributes:
        s0: Initial asset price.
        mu: Drift of the underlying GBM price process.
        sigma: Volatility of the underlying GBM price process.
        T: Simulation horizon (time to maturity).
        n_steps: Number of discrete time steps.
        A: Base fill-intensity parameter for order arrivals.
        k: Fill-intensity decay parameter for order arrivals.
    """
    s0: float
    mu: float
    sigma: float
    T: float
    n_steps: int
    A: float
    k: float
