# Getting Started

## Installation

Install matviz from PyPI:

```bash
pip install matviz
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install matviz
```

## Quick start

The easiest way to get started is to import everything from `helpers`:

```python
from matviz.helpers import *
```

This gives you access to all matviz functions plus common imports from NumPy,
Matplotlib, Pandas, and SciPy -- similar to a MATLAB-like environment.

### Histograms

```python
import numpy as np
from matviz.histogram_utils import nhist, ndhist

# Single distribution
nhist(np.random.randn(10000))

# Compare distributions
data = {'Control': np.random.randn(10000), 'Treatment': np.random.randn(5000) + 1}
nhist(data)

# 2D histogram
x = np.random.randn(10000)
y = x + np.random.randn(10000)
ndhist(x, y)
```

### Selective imports

If you prefer explicit imports:

```python
from matviz.histogram_utils import nhist, ndhist
from matviz.viz import plot_range, nicefy, logfit, streamgraph
from matviz.etl import nan_smooth, start_and_ends, load_json
from matviz import cbrt_scale  # registers CubeRootScale
```

## Return values

`nhist` and `ndhist` return the matplotlib figure object with embedded data:

```python
fig = nhist(data)
fig.nhist['N']      # bin counts
fig.nhist['bins']   # bin edges
fig.nhist['rawN']   # raw counts before normalization

fig = ndhist(x, y)
fig.ndhist['counts']  # 2D bin counts
fig.ndhist['bins_x']  # x bin edges
fig.ndhist['bins_y']  # y bin edges
```
