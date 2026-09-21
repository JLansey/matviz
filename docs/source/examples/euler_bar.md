# Euler Bar Chart

The **Euler bar chart** visualizes overlapping sets. Each bar represents the
union of two sets, drawn as two offset, semi-transparent rectangles whose
physical overlap zone exactly represents the measured intersection between the
sets. Bar height equals `set_a + set_b - overlap` (the inclusion-exclusion
union).

This is different from a standard stacked bar chart because the segments are
**not mutually exclusive** -- they intentionally overlap, and the blended zone
where the two rectangles intersect shows the overlap visually.

Like the pebble bar chart, Euler bar charts are rendered as self-contained
**HTML files** with inline CSS and JavaScript (no external dependencies).

```{image} ../images/euler-bar-example.png
:align: center
:alt: Euler bar chart example
```

## Quick start

```python
from matviz import euler_bar_chart

categories = [
    {"name": "nyc", "label": "NYC",
     "set_a": 4, "set_b": 1905, "overlap": 4},
    {"name": "bos", "label": "Boston",
     "set_a": 12, "set_b": 340, "overlap": 8},
    {"name": "cam", "label": "Cambridge",
     "set_a": 3, "set_b": 120, "overlap": 2},
]

euler_bar_chart(
    categories,
    "overlap.html",
    title="City Reports",
    subtitle="Filed vs Total reports by city",
    set_a_label="Filed",
    set_b_label="Total",
    overlap_label="Both",
)
```

Open `overlap.html` in a browser to see the chart.

## How it works

Each bar has two semi-transparent rectangles drawn with a horizontal offset:

- **Set A** (left-aligned): height proportional to `set_a`, starting at the
  baseline.
- **Set B** (offset right): height proportional to `set_b`, starting at
  `set_a - overlap` from the baseline.

Because set B starts partway up set A, the physical overlap of the two
rectangles equals the measured overlap. Both rectangles use
`mix-blend-mode: multiply` so the intersection zone blends visually.

The total bar height is the **union**: `set_a + set_b - overlap`.

## Category data

Each category dict should have:

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Internal identifier |
| `label` | str | Display label under the bar |
| `set_a` | int/float | Size of set A |
| `set_b` | int/float | Size of set B |
| `overlap` | int/float | Size of the intersection |

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `set_a_label` | `"Set A"` | Display name for set A |
| `set_b_label` | `"Set B"` | Display name for set B |
| `overlap_label` | `"Overlap"` | Display name for the overlap |
| `color_a` | `"#6A9BCC"` | Color for set A rectangles |
| `color_b` | `"#C46686"` | Color for set B rectangles |
| `opacity_a` | 0.72 | Opacity for set A |
| `opacity_b` | 0.62 | Opacity for set B |
| `bar_width` | 80 | Pixel width of each bar |
| `bar_gap` | 40 | Pixels between bars |
| `sort_desc` | True | Sort by union total, largest first |
| `show_values` | True | Show union total above each bar |
| `show_table` | True | Show data table below chart |
| `open_browser` | False | Open file in default browser |

## Data table

When `show_table=True` (the default), a table is rendered below the chart with
columns: Category, Set A, Set B, Overlap, Union.  The column headers use
`set_a_label` and `set_b_label`.

## API reference

```{eval-rst}
.. autofunction:: matviz.euler_bar.euler_bar_chart
```
