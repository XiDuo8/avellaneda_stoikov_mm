"""Shared pytest fixtures for the test suite."""

import pytest

from src.config import SimulationConfig


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig(
        s0=100.0,
        mu=0.1,
        sigma=0.2,
        T=1.0,
        n_steps=100,
        A=140.0,
        k=1.5,
    )
