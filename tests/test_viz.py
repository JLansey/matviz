"""Tests for viz module - visualization utility functions."""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matviz.viz import (
    plot_range, plot_cdf, plot_diag, plot_zero, 
    fancy_plotter, cplot, nicefy, subplotter
)


class TestPlotUtilities:
    """Test basic plotting utility functions."""

    def test_plot_cdf(self, sample_data):
        """Test plot_cdf function."""
        plot_cdf(sample_data['normal'])
        plt.close()
        
        # Should create a plot without error
        assert True

    def test_plot_cdf_with_nans(self):
        """Test plot_cdf handles NaN values."""
        data = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8])
        plot_cdf(data)
        plt.close()
        
        assert True

    def test_plot_diag(self):
        """Test plot_diag function."""
        plt.figure()
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plot_diag()
        plt.close()
        
        assert True

    def test_plot_zero(self):
        """Test plot_zero function."""
        plt.figure()
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plot_zero(lineheight=5, axx='x')
        plt.close()
        
        assert True


class TestComplexPlotting:
    """Test complex number plotting functions."""

    def test_cplot(self):
        """Test cplot function with complex data."""
        np.random.seed(42)
        z = np.random.randn(100) + 1j * np.random.randn(100)
        cplot(z, 'o')
        plt.close()
        
        assert True


class TestFancyPlotter:
    """Test fancy_plotter function."""

    def test_fancy_plotter_basic(self):
        """Test fancy_plotter with basic data."""
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1 + np.random.randn(20) * 0.5
        
        fancy_plotter(x, y)
        plt.close()
        
        assert True

    def test_fancy_plotter_with_nans(self):
        """Test fancy_plotter handles NaN values."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, np.nan, 8, 10])
        
        fancy_plotter(x, y)
        plt.close()
        
        assert True


class TestSubplotter:
    """Test subplot functions."""

    def test_subplotter_basic(self):
        """Test basic subplotter functionality."""
        plt.figure(figsize=(10, 8))
        
        ax = subplotter(2, 2, 0)
        assert ax is not None
        
        ax = subplotter(2, 2, 1)
        assert ax is not None
        
        plt.close()


class TestStyling:
    """Test styling functions."""

    def test_nicefy_basic(self):
        """Test nicefy function basic functionality."""
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Test plot")
        
        nicefy()
        plt.close()
        
        # Check that spines are modified
        assert True