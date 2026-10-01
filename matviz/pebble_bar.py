"""
Pebble Bar Chart — logarithmic-density stacked bar chart.

Each bar is a vertical stack of rows where row r (0 = bottom) has floor(base^r)
columns. Items near the bottom are large and visible; items near the top are
tiny. This creates a natural visual hierarchy that simultaneously shows exact
counts and relative magnitude — like a histogram, but every data point is
individually visible and optionally clickable.

Every bar is drawn ahead of time in Python (numpy + Pillow) to one image, see
:mod:`matviz.pebble_render`. The same drawing feeds three outputs:

- :func:`pebble_bar_figure` — a static matplotlib figure (save it as PNG/PDF,
  or put it in a notebook);
- :func:`pebble_bar_chart` — an interactive HTML page where every pebble
  shows its tooltip and opens its link; one self-contained file by default;
- :func:`write_pebble_assets` — one image per bar plus ``manifest.json``, for
  websites that load ``pebble-view.js`` themselves.

Basic usage::

    from matviz.pebble_bar import pebble_bar_chart

    categories = [
        {"name": "alpha", "label": "Alpha", "color": "#4a90d9",
         "items": [{"id": "1"}, {"id": "2"}, {"id": "3"}]},
        {"name": "beta", "label": "Beta", "color": "#d94a4a",
         "items": [{"id": "4"}, {"id": "5"}]},
    ]
    pebble_bar_chart(categories, "my_chart.html", title="My Chart")

Items can include ``link`` (URL opened on click), ``label`` (tooltip text),
and ``src`` (a local image file drawn in the cell) or ``image`` (a PIL image).
"""

import base64
import html as html_lib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .pebble_layout import bar_height, log_ticks
from .pebble_render import PebbleBar, name_seed, render_bar


_ASSETS_DIR = Path(__file__).parent / "_assets"

# Chart frame around the bars (CSS px), shared by the HTML view and the figure.
_TOP_PAD = 32
_LABEL_HEIGHT = 36
_AXIS_WIDTH = 48


def _load_js() -> str:
    """Load the pebble-view JavaScript (displays pre-rendered bars)."""
    return (_ASSETS_DIR / "pebble-view.js").read_text(encoding="utf-8")


def _slug(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "bar"


@dataclass
class PebbleChart:
    """Rendered bars plus everything needed to lay out a chart around them."""

    categories: List[Dict[str, Any]]
    bars: List[PebbleBar]
    max_height: float
    ticks: List[Tuple[int, float]]
    bar_gap: float
    show_count: bool

    def manifest(self, images: Sequence[str],
                 item_fields: Sequence[str] = ("id", "link", "label")) -> Dict[str, Any]:
        """The data block read by ``pebble-view.js``; ``images`` are the URLs
        (or data URIs) of the bar images, in bar order."""
        def r3(v):
            return round(v, 3)

        cats = []
        for cat, bar, image in zip(self.categories, self.bars, images):
            cats.append({
                "name": cat.get("name", ""),
                "label": cat.get("label", cat.get("name", "")),
                "count": len(cat.get("items", [])),
                "image": image,
                "box": [r3(v) for v in bar.box],
                "width": r3(bar.width),
                "height": r3(bar.height),
                # Top-down; item in column c sits at x0 + c*step, index first + c.
                "rows": [[r3(r.y), r3(r.size), r.cols, r.first, r3(r.x0), r3(r.step)]
                         for r in reversed(bar.rows)],
                "items": [{k: item[k] for k in item_fields if item.get(k) not in (None, "")}
                          for item in cat.get("items", [])],
            })
        return {
            "options": {"barGap": self.bar_gap, "maxHeight": r3(self.max_height),
                        "topPad": _TOP_PAD, "labelHeight": _LABEL_HEIGHT,
                        "axisWidth": _AXIS_WIDTH, "showCount": self.show_count},
            "ticks": [{"value": v, "height": r3(h)} for v, h in self.ticks],
            "categories": cats,
        }


def render_pebble_chart(
    categories: Sequence[Dict[str, Any]],
    *,
    bar_width: float = 120,
    log_base: float = math.sqrt(2),
    item_offset: int = 0,
    h_squeeze: float = 1.0,
    gutter: float = 0,
    bar_gap: float = 8,
    sort_desc: bool = True,
    show_count: bool = True,
    outline_radius: float = 1,
    background: str = "#faf8f2",
    scale: int = 2,
    supersample: int = 4,
    seed: int = 0,
    image_dir: Union[str, Path, None] = None,
    image_loader=None,
    min_tick: int = 3,
    skip_ticks: Sequence[int] = (1000,),
) -> PebbleChart:
    """Render every bar of a pebble chart (the shared step of all outputs).

    Parameters
    ----------
    categories : list of dict
        Each dict has ``name``, ``label``, ``color`` (flat fill and default
        watercolor), optional ``watercolor_color`` (falsy for no wash) and
        ``items``. Items are drawn bottom-first in list order.
    bar_width, log_base, item_offset, h_squeeze, gutter
        Layout; see :func:`matviz.pebble_layout.bar_rows`.
    bar_gap : float
        CSS px between bars.
    sort_desc : bool
        Sort categories by item count, largest first.
    show_count : bool
        Show each bar's item count beside its label.
    outline_radius : float
        White outline around pictures, in CSS px (0 = none).
    background : str
        Page colour the watercolor is blended onto.
    scale : int
        Image pixels per CSS px. 2 is sharp on most high-density screens; 3 is
        a little sharper on some phones at about twice the file size.
    supersample : int
        Drawing resolution multiplier before shrinking to ``scale``; higher
        keeps more of the tiny top-row pebbles.
    seed : int
        Seed for the overlap order and the watercolor texture. Each bar uses
        ``seed`` mixed with its name, so rebuilding gives the same images.
    image_dir : path, optional
        Base directory for relative item ``src`` paths.
    image_loader : callable, optional
        ``image_loader(item) -> PIL.Image`` to load pictures some other way.
    min_tick, skip_ticks
        Y-axis ticks: 1-2-5 values from ``min_tick`` up, minus ``skip_ticks``.
    """
    cats = [dict(c) for c in categories]
    if sort_desc:
        cats.sort(key=lambda c: len(c.get("items", [])), reverse=True)
    layout = dict(bar_width=bar_width, log_base=log_base,
                  item_offset=item_offset, gutter=gutter)
    bars = []
    for cat in cats:
        name = cat.get("name", "")
        wash = cat.get("watercolor_color", cat.get("watercolorColor", cat.get("color", "#888")))
        bar_seed = name_seed(name, seed)
        bars.append(render_bar(
            cat.get("items", []), h_squeeze=h_squeeze, color=cat.get("color", "#888"),
            watercolor_color=wash or None, background=background,
            outline_radius=outline_radius, scale=scale, supersample=supersample,
            seed=bar_seed, wash_seeds=(name_seed(name + ":grain", seed),
                                       name_seed(name + ":warp", seed)),
            image_dir=image_dir, image_loader=image_loader, **layout))
    counts = [len(c.get("items", [])) for c in cats]
    max_count = max(counts, default=0)
    ticks = [(t, bar_height(t, **layout)) for t in log_ticks(max_count, min_tick)
             if t not in skip_ticks]
    return PebbleChart(cats, bars, bar_height(max_count, **layout), ticks,
                       bar_gap, show_count)


def _encode(image, fmt="webp", quality=90):
    buf = io.BytesIO()
    if fmt == "webp":
        image.save(buf, "WEBP", quality=quality, method=6)
    else:
        image.save(buf, fmt.upper())
    return buf.getvalue()


def write_pebble_assets(
    categories: Sequence[Dict[str, Any]],
    out_dir: Union[str, Path],
    *,
    manifest_name: str = "manifest.json",
    image_format: str = "webp",
    quality: int = 90,
    item_fields: Sequence[str] = ("id", "link", "label"),
    **opts,
) -> Dict[str, Any]:
    """Write one image per bar and a manifest for ``pebble-view.js``.

    Writes ``<out_dir>/<name>.<image_format>`` for each category and
    ``<out_dir>/<manifest_name>``. Image paths in the manifest are relative to
    the manifest, so pass ``baseUrl`` (the manifest's folder URL) to
    ``renderPebbleView``. Other keyword arguments go to
    :func:`render_pebble_chart`. Returns the manifest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chart = render_pebble_chart(categories, **opts)
    names = []
    for cat, bar in zip(chart.categories, chart.bars):
        fname = f"{_slug(cat.get('name', ''))}.{image_format}"
        (out_dir / fname).write_bytes(_encode(bar.image, image_format, quality))
        names.append(fname)
    manifest = chart.manifest(names, item_fields)
    (out_dir / manifest_name).write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return manifest


def pebble_bar_figure(categories: Sequence[Dict[str, Any]], *, ax=None,
                      figsize=None, label_rotation=0, **opts):
    """Draw a static pebble bar chart with matplotlib.

    Takes the same options as :func:`render_pebble_chart`. Pass ``ax`` to draw
    into existing axes. One axes unit is one CSS pixel of the HTML chart.
    Returns the figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    chart = render_pebble_chart(categories, **opts)
    background = opts.get("background", "#faf8f2")
    widths = [b.width for b in chart.bars]
    step = (max(widths, default=0)) + chart.bar_gap
    total_w = max(len(chart.bars) * step - chart.bar_gap, 1)
    if ax is None:
        figsize = figsize or ((total_w + _AXIS_WIDTH + 40) / 72,
                              (chart.max_height + _TOP_PAD + _LABEL_HEIGHT + 30) / 72)
        fig, ax = plt.subplots(figsize=figsize, facecolor=background)
    fig = ax.figure
    ax.set_facecolor(background)
    centres = []
    for i, (cat, bar) in enumerate(zip(chart.categories, chart.bars)):
        left = i * step + (step - chart.bar_gap - bar.width) / 2
        bx, by, bw, bh = bar.box
        top = bar.height - by
        ax.imshow(np.asarray(bar.image), extent=(left + bx, left + bx + bw, top - bh, top),
                  interpolation="antialiased", zorder=2)
        centres.append(left + bar.width / 2)
    ax.set_xlim(-12, total_w + 12)
    ax.set_ylim(0, chart.max_height + _TOP_PAD / 2)
    ax.set_aspect("equal")
    labels = [cat.get("label", cat.get("name", "")) +
              (f"\n{len(cat.get('items', []))}" if chart.show_count else "")
              for cat in chart.categories]
    ax.set_xticks(centres, labels, rotation=label_rotation, fontweight="bold")
    ax.set_yticks([h for _, h in chart.ticks], [str(v) for v, _ in chart.ticks])
    ax.grid(axis="y", color="#87867F", alpha=0.28, linewidth=0.75, zorder=0)
    ax.tick_params(length=0, colors="#5a5852")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#8a8780")
    return fig


def pebble_bar_chart(
    categories: Sequence[Dict[str, Any]],
    output: Union[str, Path, None] = None,
    *,
    title: str = "",
    subtitle: str = "",
    embed_images: bool = True,
    font_family: str = "system-ui, -apple-system, sans-serif",
    extra_css: str = "",
    extra_js: str = "",
    open_browser: bool = False,
    **opts,
) -> str:
    """Generate an interactive pebble bar chart as an HTML page.

    The bars are drawn in Python; the page only shows the bar images and
    maps the pointer to the item under it (tooltip from ``label`` or ``id``,
    click opens ``link``).

    Parameters
    ----------
    categories : list of dict
        Each category dict has ``name``, ``label``, ``color`` (also the
        default watercolor wash), optional ``watercolor_color`` (``None`` or
        ``""`` for flat squares without a wash), and ``items``: dicts with
        ``id``, ``label`` (tooltip), ``link`` (URL) and ``src`` (local image
        path) or ``image`` (PIL image).
    output : str or Path or None
        Path to write the HTML file. If None, the HTML string is returned
        without writing to disk.
    title, subtitle : str
        Heading and description line.
    embed_images : bool
        True (default) inlines the bar images as data URIs so the chart is a
        single file. False writes them to ``<output stem>_files/`` beside
        the page (requires ``output``).
    font_family, extra_css, extra_js
        Page styling and extra script run after the chart is drawn
        (``chart`` holds the return value of ``renderPebbleView``).
    open_browser : bool
        Open the file in the default browser after writing.
    **opts
        Chart options for :func:`render_pebble_chart`: ``bar_width``
        (default 120), ``log_base`` (√2), ``item_offset`` (0), ``h_squeeze``
        (1.0), ``bar_gap`` (8), ``sort_desc``, ``show_count``,
        ``outline_radius`` (1), ``scale`` (2), ``supersample`` (4),
        ``seed`` (0), ``image_dir``, ...

    Returns
    -------
    str
        The HTML string. Also written to ``output`` if provided.

    Examples
    --------
    >>> categories = [
    ...     {"name": "trucks", "label": "Trucks", "color": "#b87200",
    ...      "items": [{"id": str(i)} for i in range(45)]},
    ...     {"name": "vans", "label": "Vans", "color": "#186b4e",
    ...      "items": [{"id": str(i)} for i in range(30)]},
    ... ]
    >>> html = pebble_bar_chart(categories, "fleet.html",
    ...                         title="Vehicle Fleet", log_base=1.1,
    ...                         item_offset=16, h_squeeze=0.7)
    """
    if not embed_images and output is None:
        raise ValueError("embed_images=False needs an output path to write images next to")
    chart = render_pebble_chart(categories, **opts)
    if embed_images:
        images = ["data:image/webp;base64," + base64.b64encode(_encode(b.image)).decode()
                  for b in chart.bars]
    else:
        output = Path(output)
        files = output.parent / f"{output.stem}_files"
        files.mkdir(parents=True, exist_ok=True)
        images = []
        for cat, bar in zip(chart.categories, chart.bars):
            fname = f"{_slug(cat.get('name', ''))}.webp"
            (files / fname).write_bytes(_encode(bar.image))
            images.append(f"{files.name}/{fname}")
    manifest = json.dumps(chart.manifest(images)).replace("</", "<\\/")
    background = opts.get("background", "#faf8f2")
    heading = (f"<h2 class=\"pebble-title\">{html_lib.escape(title)}</h2>" if title else "") + \
        (f"<p class=\"pebble-subtitle\">{html_lib.escape(subtitle)}</p>" if subtitle else "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title or 'Pebble Bar Chart')}</title>
<style>
  :root {{
    color-scheme: light;
    --pebble-bg: {background};
    --pebble-text: #1a1a1c;
    --pebble-text-secondary: #5a5852;
    --pebble-text-muted: #8a8780;
    --pebble-grid: #87867F;
  }}
  body {{
    font-family: {font_family};
    background: var(--pebble-bg);
    color: var(--pebble-text);
    padding: 24px 20px 60px;
    margin: 0 auto;
    max-width: 1200px;
  }}
  .pebble-title {{ font-size: 1.5rem; font-weight: 700; margin: 0 0 4px; }}
  .pebble-subtitle {{ color: var(--pebble-text-secondary); font-size: 0.875rem; margin: 0 0 28px; }}
  #pebble-chart {{
    overflow-x: auto;
    padding-bottom: 60px;
    padding-left: 6px;
  }}
  .pebble-bar-wrapper {{
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
  }}
  .pebble-bar-label {{
    color: var(--pebble-text-secondary);
    text-align: center;
    white-space: nowrap;
  }}
  {extra_css}
</style>
</head>
<body>
{heading}
<div id="pebble-chart"></div>
<script>
{_load_js()}
</script>
<script>
(function() {{
  var data = {manifest};
  var chart = renderPebbleView(document.getElementById('pebble-chart'), data,
                               {{ fontFamily: {json.dumps(font_family)} }});
  {extra_js}
}})();
</script>
</body>
</html>"""

    if output is not None:
        output = Path(output)
        output.write_text(html, encoding="utf-8")
        if open_browser:
            import webbrowser
            webbrowser.open(output.resolve().as_uri())
    return html
