# matviz

[![PyPI](https://img.shields.io/pypi/v/matviz)](https://pypi.org/project/matviz/)
[![Python](https://img.shields.io/pypi/pyversions/matviz)](https://pypi.org/project/matviz/)
[![Tests](https://github.com/JLansey/matviz/actions/workflows/test.yml/badge.svg)](https://github.com/JLansey/matviz/actions/workflows/test.yml)
[![Docs](https://readthedocs.org/projects/matviz/badge/?version=latest)](https://matviz.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/matviz)](https://github.com/JLansey/matviz/blob/main/LICENSE)

**Data visualization for scientists, made easy.** Smart histograms, publication-ready plots, and data wrangling utilities.

![matviz example output](https://raw.githubusercontent.com/JLansey/matviz/main/docs/source/images/front.png)

## Installation

```bash
pip install matviz
```

## Quick start

```python
from matviz.helpers import *

# Compare distributions with automatic binning
data = {'Control': np.random.randn(10000), 'Treatment': np.random.randn(5000) + 1}
nhist(data)

# 2D histogram from complex numbers
z = (5 + np.random.randn(1000)) * np.exp(1j * np.random.randn(1000))
ndhist(z, smooth=1)
```

## Modules

| Module | Description |
|--------|-------------|
| `histogram_utils` | Smart 1D (`nhist`) and 2D (`ndhist`) histograms with automatic binning |
| `viz` | Plot ranges, streamgraphs, log-fitting, polar grids, complex plots, `nicefy` |
| `etl` | `nan_smooth`, cross-correlation, JSON with complex numbers, time utilities |
| `cbrt_scale` | Cube-root axis scale for matplotlib |
| `circle_utils` | Smallest enclosing circle and convex hull area |
| `datetime_converter` | Reversible timestamp codec (`DateCodec`) |
| `doctools` | PDF image extraction and compression |
| `helpers` | MATLAB-like environment with all imports |

## Documentation

Full documentation at **[matviz.readthedocs.io](https://matviz.readthedocs.io/en/latest/)**.

## License

BSD-3-Clause
