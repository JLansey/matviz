"""Tests for etl module - data processing and utility functions."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from matviz.etl import (
    nan_smooth, flatten, unflatten, handle_dates,
    clean_whitespace, robust_floater, isdigit,
    hex2rgb, rgb2hex
)


class TestDataProcessing:
    """Test data processing functions."""

    def test_nan_smooth_basic(self):
        """Test basic nan_smooth functionality."""
        np.random.seed(42)
        y = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.3 * np.random.randn(100)
        
        smoothed = nan_smooth(y, n=5)
        
        # Should return same length
        assert len(smoothed) == len(y)
        # Should be smoother
        assert np.var(smoothed) < np.var(y)

    def test_nan_smooth_with_nans(self):
        """Test nan_smooth with NaN values."""
        y = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9])
        
        smoothed = nan_smooth(y, n=3, ignore_nans=True)
        
        # Should return same length
        assert len(smoothed) == len(y)

    def test_flatten_unflatten(self):
        """Test flatten and unflatten functions."""
        # Create nested structure with same-shaped arrays
        original = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
        ]
        
        # Flatten then unflatten
        flat = flatten(original)
        restored = unflatten(flat, original)
        
        # Should restore original structure
        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            np.testing.assert_array_equal(orig, rest)


class TestDateHandling:
    """Test date/time functions."""

    def test_handle_dates_datetime(self):
        """Test handle_dates with datetime objects."""
        import datetime
        dates = [
            datetime.datetime(2023, 1, 1, 12, 0, 0),
            datetime.datetime(2023, 1, 2, 12, 0, 0)
        ]
        X = [dates]
        
        result_X, datetime_flag, date_formatting = handle_dates(X)
        
        assert datetime_flag is True
        # handle_dates returns numpy arrays, not lists
        assert isinstance(result_X[0], np.ndarray)
        assert len(result_X[0]) == 2


class TestUtilityFunctions:
    """Test general utility functions."""

    def test_clean_whitespace(self):
        """Test clean_whitespace function."""
        text = "  hello\n\tworld  \r  "
        result = clean_whitespace(text)
        expected = "hello world"
        
        assert result == expected

    def test_robust_floater(self):
        """Test robust_floater function."""
        assert robust_floater("3.14") == 3.14
        assert np.isnan(robust_floater("hello"))
        assert np.isnan(robust_floater(None))
        assert robust_floater("-2.5") == -2.5

    def test_isdigit(self):
        """Test isdigit function."""
        assert isdigit(42) is True
        assert isdigit(3.14) is True
        assert isdigit("123") is True
        assert isdigit("3.14") is True
        assert isdigit("-5.2") is True
        assert isdigit("hello") is False

    def test_hex2rgb_rgb2hex(self):
        """Test hex2rgb and rgb2hex functions."""
        # Test hex string conversion
        hex_color = "#FF0080"
        rgb = hex2rgb(hex_color)
        assert len(rgb) == 3
        assert all(0 <= c <= 1 for c in rgb)
        
        # Test round-trip
        hex_result = rgb2hex(255, 0, 128)
        assert hex_result.lower() == "#ff0080"