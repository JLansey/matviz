# histogram_utils

Smart 1D and 2D histograms with automatic binning using Scott's normal reference rule.

## Example output

### nhist

```{image} ../images/Figure_1.png
:align: center
:alt: nhist example — comparing distributions
```

### ndhist

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item}
```{image} ../images/Figure_12_ndhist.png
:alt: ndhist example — 2D density
```
:::

:::{grid-item}
```{image} ../images/Figure_10_ndhist_timeseries.png
:alt: ndhist example — timeseries
```
:::
::::

## API reference

```{eval-rst}
.. automodule:: matviz.histogram_utils
   :members: nhist, ndhist
   :show-inheritance:
```
