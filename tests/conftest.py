"""Shared pytest fixtures and configuration for matviz tests."""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend for testing
matplotlib.use('Agg')


@pytest.fixture(scope="session", autouse=True)
def setup_matplotlib():
    """Configure matplotlib for testing."""
    plt.ioff()
    plt.style.use('default')
    yield
    plt.close('all')


@pytest.fixture
def sample_data():
    """Generate sample datasets for testing."""
    np.random.seed(42)
    return {
        'normal': np.random.randn(1000),
        'uniform': np.random.uniform(0, 10, 500),
        'integers': np.random.randint(0, 20, 300),
    }


@pytest.fixture
def sample_2d_data():
    """Generate sample 2D data for ndhist testing."""
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = 0.5 * x + np.random.randn(n) * 0.5
    return x, y


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Automatically clean up matplotlib figures after each test."""
    yield
    plt.close('all')
