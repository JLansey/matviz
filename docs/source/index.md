# matviz

**Data visualization for scientists, made easy.**

```{image} images/front.png
:align: center
:alt: matviz example output
```

## Installation

```bash
pip install matviz
```

## Quick start

```python
from matviz.helpers import *

# Beautiful histograms with automatic binning
data = {'Control': np.random.randn(10000), 'Treatment': np.random.randn(5000) + 1}
nhist(data)
```

---

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Histograms
Smart 1D and 2D histograms with automatic binning, comparison support, and sensible defaults.

[Learn more](api/histogram_utils.md)
:::

:::{grid-item-card} Visualization
Plot ranges, streamgraphs, log-fits, polar grids, complex number plots, and more.

[Learn more](api/viz.md)
:::

:::{grid-item-card} ETL Utilities
Smoothing, cross-correlation, JSON with complex numbers, time utilities, and data wrangling.

[Learn more](api/etl.md)
:::
::::

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

getting_started
examples/index
api/index
changelog
```
