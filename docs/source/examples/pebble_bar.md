# Pebble Bar Chart

The **pebble bar chart** is a logarithmic-density stacked bar chart where every
data point is individually visible.  Items stack from large squares at the bottom
to tiny ones at the top, creating a natural visual hierarchy that shows both
exact counts and relative magnitudes — like a histogram where every data point
gets its own pixel.

Each bar gets a **watercolor wash** by default — an SVG turbulence texture that
gives the chart a hand-painted, organic look.  Items with images sit on top of
the wash; items without images are represented by the wash alone.

Unlike the matplotlib-based charts in matviz, pebble bar charts are rendered as
self-contained **HTML files** with inline JavaScript.  This makes them ideal for
interactive reports where items can link to detail pages, show tooltips, or
display thumbnail images.

## Quick start

```python
from matviz import pebble_bar_chart

categories = [
    {"name": "trucks", "label": "Trucks", "color": "#b87200",
     "items": [{"id": str(i)} for i in range(87)]},
    {"name": "vans", "label": "Vans", "color": "#186b4e",
     "items": [{"id": str(i)} for i in range(54)]},
    {"name": "sedans", "label": "Sedans", "color": "#4e82b8",
     "items": [{"id": str(i)} for i in range(18)]},
]

pebble_bar_chart(
    categories,
    "fleet.html",
    title="Vehicle Fleet",
    subtitle="Each square = one vehicle",
    bar_width=150,
    log_base=1.1,
    item_offset=16,
    h_squeeze=0.7,
    bar_gap=21,
)
```

Open `fleet.html` in a browser to see the chart.

## How it works

Each bar is a vertical stack of rows.  Row *r* (counting from 0 at the bottom)
has `floor(base^r)` columns.  Each cell is `barWidth / base^r` pixels wide and
tall (square).  The logarithmic stacking means:

- **Bottom rows**: 1–2 large squares per row (prominent, individually visible)
- **Top rows**: many tiny squares packed together (conveys volume)

The `log_base` parameter controls how fast columns multiply:

| `log_base` | Behavior |
|------------|----------|
| √2 ≈ 1.414 (default) | Columns double every 2 rows — balanced |
| 1.1 | Slow growth — tall, narrow pyramids |
| 2.0 | Fast growth — short, wide pyramids |

## Item data

Each item in a category can carry:

| Key | Type | Description |
|-----|------|-------------|
| `id` | str | Unique identifier |
| `label` | str | Tooltip text on hover |
| `link` | str | URL opened on click |
| `src` | str | Image URL displayed in the cell |

```python
items = [
    {"id": "v001", "label": "Truck #1", "link": "https://example.com/v001",
     "src": "images/v001.jpg"},
    {"id": "v002", "label": "Truck #2"},  # represented by the watercolor wash
]
```

Items with `src` images are rendered on top of the watercolor wash, with a white
outline (controlled by `outline_radius`, default 2px) that follows the image's
alpha mask.  Items without images are represented by the wash alone.

## Watercolor styling

Watercolor is **on by default** — the `color` you set on each category is used
for the wash.  You only need `watercolor_color` if you want the wash to be a
different shade from the base color:

```python
categories = [
    # Watercolor uses the color automatically
    {"name": "alpha", "label": "Alpha", "color": "#b87200",
     "items": [...]},

    # Override: base color is gray, but wash is amber
    {"name": "beta", "label": "Beta", "color": "#888",
     "watercolor_color": "#b87200",
     "items": [...]},

    # Opt out of watercolor entirely — flat solid squares
    {"name": "gamma", "label": "Gamma", "color": "#4e82b8",
     "watercolor_color": "",
     "items": [...]},
]
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bar_width` | 120 | Pixel width of each bar before squeeze |
| `log_base` | √2 | Log base controlling density growth |
| `item_offset` | 0 | Invisible padding items to skip sparse bottom rows |
| `h_squeeze` | 1.0 | Horizontal squeeze factor (0–1) |
| `bar_gap` | 8 | Pixels between adjacent bars |
| `outline_radius` | 2 | White outline around image items (0 to disable) |
| `sort_desc` | True | Sort categories largest first |
| `show_count` | True | Show item count next to labels |
| `open_browser` | False | Open the file in the default browser |

## Full example

See `examples/pebble_bar_example.py` in the repository for a complete working
example with multiple categories.

## API reference

```{eval-rst}
.. autofunction:: matviz.pebble_bar.pebble_bar_chart
```
