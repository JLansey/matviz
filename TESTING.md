# Testing Guide

## Quick Start

```bash
# Install with dev dependencies
make install
# or: pip install -e ".[dev]"

# Run tests
make test

# Run linting
make lint
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_histogram_utils.py  # Histogram tests
├── test_viz.py             # Visualization tests
└── test_etl.py             # Data processing tests
```

## Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_histogram_utils.py

# With coverage
pytest tests/ --cov=matviz

# Test notebooks
pytest --nbval examples/
```

## CI/CD

Tests run automatically on GitHub Actions for:
- Python 3.9, 3.10, 3.11
- Push to main/dev
- Pull requests

See `.github/workflows/test.yml` for details.

## Adding Tests

1. Add test functions to appropriate test file
2. Use fixtures from `conftest.py` (sample_data, sample_2d_data, etc.)
3. Always close matplotlib figures: `plt.close()`
4. Run `make test` to verify

Example:
```python
def test_my_function(sample_data):
    """Test description."""
    result = my_function(sample_data['normal'])
    plt.close()
    assert result is not None
```
