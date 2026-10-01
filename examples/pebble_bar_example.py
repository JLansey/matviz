"""
Pebble Bar Chart — example
===========================

Generates a pebble bar chart showing fictional fleet vehicle categories.
Each bar stacks items with logarithmic density: large visible squares at
the bottom, tiny ones at the top. Every bar gets a hand-painted watercolor
wash by default.

Run this script to write, in the current directory:

- pebble_bar_demo.html — the interactive chart (one self-contained file);
- pebble_bar_demo.png  — the same chart as a static matplotlib figure.
"""

from matviz import pebble_bar_chart, pebble_bar_figure

# ── Build some sample data ──────────────────────────────────────────
# Each category has a name, display label, colors, and a list of items.
# Items can optionally carry link (URL), label (tooltip), and src (a local
# image file drawn in the square; relative paths resolve against image_dir).

categories = [
    {
        "name": "heavy",
        "label": "Heavy",
        "color": "#b87200",
        "items": [{"id": f"h{i}"} for i in range(87)],
    },
    {
        "name": "medium",
        "label": "Medium",
        "color": "#b05070",
        "items": [{"id": f"m{i}"} for i in range(54)],
    },
    {
        "name": "delivery",
        "label": "Delivery",
        "color": "#7a4f20",
        "items": [{"id": f"d{i}"} for i in range(41)],
    },
    {
        "name": "utility",
        "label": "Utility",
        "color": "#607a45",
        "items": [{"id": f"u{i}"} for i in range(33)],
    },
    {
        "name": "transit",
        "label": "Transit",
        "color": "#de9535",
        "items": [{"id": f"t{i}"} for i in range(22)],
    },
    {
        "name": "service",
        "label": "Service",
        "color": "#cc5f3e",
        "items": [{"id": f"s{i}"} for i in range(15)],
    },
    {
        "name": "compact",
        "label": "Compact",
        "color": "#186b4e",
        "items": [{"id": f"c{i}"} for i in range(9)],
    },
    {
        "name": "specialty",
        "label": "Specialty",
        "color": "#7b74c8",
        "items": [{"id": f"x{i}"} for i in range(5)],
    },
]

# ── Clickable items example ─────────────────────────────────────────
# Add links and tooltips to show the interactive capabilities.
for cat in categories:
    for item in cat["items"]:
        item["label"] = f'{cat["label"]} #{item["id"]}'
        # In a real application, items would link to detail pages:
        # item["link"] = f"https://example.com/fleet/{item['id']}"

# ── Render ──────────────────────────────────────────────────────────
# The bars are drawn in Python. seed fixes the overlap order and watercolor
# texture, so re-running gives identical output; scale is image pixels per
# CSS pixel (2 = sharp on high-density screens).
options = dict(
    bar_width=150,
    log_base=1.1,
    item_offset=16,
    h_squeeze=0.7,
    bar_gap=21,
    seed=0,
    scale=2,
)

html = pebble_bar_chart(
    categories,
    "pebble_bar_demo.html",
    title="Fleet Composition",
    subtitle="Vehicle categories · Each square = one vehicle",
    **options,
)
print(f"Written: pebble_bar_demo.html ({len(html):,} bytes)")

# The same bars as a static matplotlib figure.
fig = pebble_bar_figure(categories, **options)
fig.savefig("pebble_bar_demo.png", dpi=144, bbox_inches="tight")
print("Written: pebble_bar_demo.png")
