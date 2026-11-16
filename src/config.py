"""Configuration objects bundling simulation parameters."""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Market and simulation parameters shared across a run.

    Attributes:
        s0: Initial asset price.
        mu: Drift of the underlying GBM price process.
        sigma: Volatility of the underlying GBM price process, as a fraction
            of the price per root unit of time.
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

    @property
    def dollar_sigma(self) -> float:
        """Volatility in price units per root unit of time.

        The AS formulas assume a price that moves by a fixed dollar amount,
        while GBM moves it by a fixed fraction of the price. The two are
        matched at the initial price, so a GBM volatility of 0.005 on a
        price of 100 is a dollar volatility of 0.5.
        """
        return self.sigma * self.s0
