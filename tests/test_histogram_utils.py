"""Tests for histogram_utils module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from matviz.histogram_utils import nhist, ndhist, choose_bins, isdiscrete


class TestNhist:
    """Test the nhist function - smart histogram plotting."""

    def test_nhist_single_array(self, sample_data):
        """Test nhist with a single array."""
        ax, N, bins = nhist(sample_data['normal'])
        plt.close()
        
        assert ax is not None
        assert isinstance(N, list) and len(N) == 1
        assert len(bins[0]) > 0

    def test_nhist_multiple_arrays(self, sample_data):
        """Test nhist with multiple arrays."""
        data = [sample_data['normal'], sample_data['uniform']]
        ax, N, bins = nhist(data, labels=['Normal', 'Uniform'])
        plt.close()
        
        assert len(N) == 2
        assert len(bins) == 2

    def test_nhist_dictionary_input(self, sample_data):
        """Test nhist with dictionary input."""
        data_dict = {
            'Normal': sample_data['normal'][:500],
            'Uniform': sample_data['uniform'][:500]
        }
        ax, N, bins = nhist(data_dict)
        plt.close()
        
        assert len(N) == 2
        assert len(bins) == 2

    def test_nhist_integer_bins(self, sample_data):
        """Test integer bins flag."""
        data = sample_data['integers']
        ax, N, bins = nhist(data, int_bins_flag=True)
        plt.close()
        
        # Bins should be on integer boundaries
        assert len(bins[0]) > 0


class TestNdhist:
    """Test the ndhist function - 2D histogram plotting."""

    def test_ndhist_basic(self, sample_2d_data):
        """Test basic ndhist functionality."""
        x, y = sample_2d_data
        counts, bins_x, bins_y = ndhist(x, y)
        plt.close()
        
        assert counts.shape[0] > 0
        assert counts.shape[1] > 0
        assert len(bins_x) == counts.shape[0] + 1
        assert len(bins_y) == counts.shape[1] + 1

    def test_ndhist_timeseries(self):
        """Test ndhist with timeseries (no y provided)."""
        np.random.seed(42)
        y = np.cumsum(np.random.randn(100))
        
        counts, bins_x, bins_y = ndhist(y)
        plt.close()
        
        assert counts.shape[0] > 0
        assert counts.shape[1] > 0

    def test_ndhist_complex_numbers(self):
        """Test ndhist with complex numbers."""
        np.random.seed(42)
        z = np.random.randn(100) + 1j * np.random.randn(100)
        
        counts, bins_x, bins_y = ndhist(z)
        plt.close()
        
        assert counts.shape[0] > 0


class TestChooseBins:
    """Test the choose_bins function."""

    def test_choose_bins_basic(self, sample_data):
        """Test basic bin choosing functionality."""
        X = [sample_data['normal']]
        bins, bin_widths = choose_bins(X)
        
        assert len(bins) == 1
        assert len(bin_widths) == 1
        assert len(bins[0]) > 10
        assert bin_widths[0] > 0

    def test_choose_bins_integer_flag(self, sample_data):
        """Test choose_bins with integer flag."""
        X = [sample_data['integers']]
        bins, bin_widths = choose_bins(X, int_bins_flag=True)
        
        # Bin width should be integer
        assert bin_widths[0] >= 1
        assert bin_widths[0] == int(bin_widths[0])


class TestUtilities:
    """Test utility functions."""

    def test_isdiscrete(self):
        """Test isdiscrete function."""
        assert isdiscrete([1, 2, 3, 4, 5])
        assert isdiscrete([1.0, 2.0, 3.0, 4.0, 5.0])
        assert not isdiscrete([1.1, 2.3, 3.7, 4.2, 5.9])
        assert isdiscrete(5)