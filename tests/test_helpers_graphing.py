"""Tests for helpers_graphing module."""

import pytest
import sys


# Wildcard imports inside functions are not allowed by Python's syntax/linter,
# so this test as written cannot work. Instead, you can check for basic importability:
def test_helpers_graphing_import():
    """Test that matviz.helpers_graphing can be imported and has expected symbols."""
    import importlib
    helpers_graphing = importlib.import_module('matviz.helpers_graphing')
    
    # Check for key matplotlib functions
    required = ['plot', 'hist', 'figure', 'xlabel', 'ylabel']
    for name in required:
        assert hasattr(helpers_graphing, name), f"helpers_graphing missing matplotlib function {name}"
    
    # Check for numpy functions
    numpy_funcs = ['mean', 'sqrt', 'linspace']
    for name in numpy_funcs:
        assert hasattr(helpers_graphing, name), f"helpers_graphing missing numpy function {name}"
    
    # Check for custom functions
    custom_funcs = ['tic', 'toc', 'zoom_plot', 'fig_sizer']
    for name in custom_funcs:
        assert hasattr(helpers_graphing, name), f"helpers_graphing missing custom function {name}"


def test_helpers_graphing_specific_imports():
    """Test importing specific items from helpers_graphing."""
    from matviz.helpers_graphing import tic, toc, fig_sizer, zoom_plot
    
    # These should all be callable
    assert callable(tic)
    assert callable(toc)
    assert callable(fig_sizer)
    assert callable(zoom_plot)


def test_tic_toc():
    """Test that tic/toc functions work."""
    import time
    from matviz.helpers_graphing import tic, toc, silent_toc
    
    tic()
    time.sleep(0.01)  # Sleep for 10ms
    elapsed = toc()
    
    assert elapsed is not None
    assert elapsed >= 0.01  # Should be at least 10ms
    
    # Test silent_toc
    tic()
    time.sleep(0.01)
    elapsed = silent_toc()
    assert elapsed >= 0.01


def test_fig_sizer():
    """Test fig_sizer function."""
    import matplotlib.pyplot as plt
    from matviz.helpers_graphing import fig_sizer
    
    # Test with no arguments
    fig_sizer()
    size = plt.rcParams["figure.figsize"]
    assert size == [15, 10]
    
    # Test with one argument
    fig_sizer(8)
    size = plt.rcParams["figure.figsize"]
    assert size == [8, 8]
    
    # Test with two arguments
    fig_sizer(6, 10)
    size = plt.rcParams["figure.figsize"]
    assert size == [10, 6]


def test_helper_functions():
    """Test various helper functions."""
    from matviz.helpers_graphing import display_pnct, return_pnct, count_helper
    
    # Test return_pnct
    result = return_pnct(25, 100)
    assert "%25" in result
    assert "25" in result
    assert "100" in result
    
    # Test count_helper (just make sure it doesn't crash)
    count_helper(10, S=100, freq=10)
    count_helper(10, S=100, freq=10, pcnt=True)


def test_custom_functions_available():
    """Test that custom matviz functions are available after star import."""
    from matviz.helpers_graphing import nhist, ndhist
    
    assert callable(nhist)
    assert callable(ndhist)


def test_matplotlib_defaults():
    """Test that matplotlib defaults are set correctly when helpers_graphing is imported."""
    import matplotlib.pyplot as plt
    import matplotlib
    
    # Reset to matplotlib defaults first
    matplotlib.rcdefaults()
    
    # Now import helpers_graphing which should set the defaults
    import importlib
    if 'matviz.helpers_graphing' in sys.modules:
        importlib.reload(sys.modules['matviz.helpers_graphing'])
    else:
        importlib.import_module('matviz.helpers_graphing')
    
    # Check default figure size (should be set by helpers_graphing)
    assert plt.rcParams["figure.figsize"] == [12, 9]
    
    # Check default colormap
    assert plt.rcParams['image.cmap'] == 'viridis'

