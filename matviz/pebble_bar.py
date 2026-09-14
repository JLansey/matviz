"""
Pebble Bar Chart — logarithmic-density stacked bar chart.

Each bar is a vertical stack of rows where row r (0 = bottom) has floor(base^r)
columns. Items near the bottom are large and visible; items near the top are
tiny. This creates a natural visual hierarchy that simultaneously shows exact
counts and relative magnitude — like a histogram, but every data point is
individually visible and optionally clickable.

The chart is rendered as a self-contained HTML file with inline JavaScript,
making it ideal for interactive reports where items can link to detail pages.

Basic usage::

    from matviz.pebble_bar import pebble_bar_chart

    categories = [
        {"name": "alpha", "label": "Alpha", "color": "#4a90d9",
         "items": [{"id": "1"}, {"id": "2"}, {"id": "3"}]},
        {"name": "beta", "label": "Beta", "color": "#d94a4a",
         "items": [{"id": "4"}, {"id": "5"}]},
    ]
    pebble_bar_chart(categories, "my_chart.html", title="My Chart")

Items can include ``link`` (URL opened on click), ``src`` (image URL),
and ``label`` (tooltip text) fields.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


_ASSETS_DIR = Path(__file__).parent / "_assets"


def _load_js() -> str:
    """Load the pebble-bar JavaScript engine."""
    js_path = _ASSETS_DIR / "pebble-bar.js"
    return js_path.read_text(encoding="utf-8")


def _build_html(
    categories: Sequence[Dict[str, Any]],
    *,
    title: str = "",
    subtitle: str = "",
    bar_width: int = 120,
    log_base: float = 1.414,
    item_offset: int = 0,
    h_squeeze: float = 1.0,
    bar_gap: int = 8,
    sort_desc: bool = True,
    show_count: bool = True,
    outline_radius: float = 2,
    font_family: str = "system-ui, -apple-system, sans-serif",
    extra_css: str = "",
    extra_js: str = "",
) -> str:
    """Build the full HTML string for a pebble bar chart.

    Parameters
    ----------
    categories : list of dict
        Each dict has keys:

        - ``name`` (str): internal identifier
        - ``label`` (str): display label under the bar
        - ``color`` (str): hex color for the bar; also used as the watercolor
          wash by default
        - ``watercolor_color`` (str, optional): override the watercolor wash
          color (defaults to ``color``).  Set to ``None`` or ``""`` to opt out
          of watercolor and get flat solid squares instead.
        - ``items`` (list of dict): each item can have ``id``, ``label``,
          ``link`` (URL), and ``src`` (image URL)

    title : str
        Chart heading.
    subtitle : str
        Subtitle / description line.
    bar_width : int
        Pixel width of each bar column (before squeeze).
    log_base : float
        Logarithmic base controlling density growth.  Default √2 ≈ 1.414 means
        columns double every 2 rows.  Lower values (e.g. 1.1) produce taller,
        skinnier pyramids; higher values (e.g. 2.0) produce shorter, wider ones.
    item_offset : int
        Number of invisible padding items prepended to skip the sparse bottom rows.
    h_squeeze : float
        Horizontal squeeze factor (0–1). At 1.0 bars use their full width;
        at 0.7 they are 70% as wide, leaving breathing room.
    bar_gap : int
        Pixel gap between adjacent bars.
    sort_desc : bool
        Sort categories by item count, largest first.
    show_count : bool
        Show item count next to each label.
    outline_radius : float
        Pixel radius of the white outline drawn around image items (follows the
        alpha mask).  Set to 0 to disable.  Default 2.
    font_family : str
        CSS font-family for the chart.
    extra_css : str
        Additional CSS injected into the ``<style>`` block.
    extra_js : str
        Additional JavaScript injected after the chart renders.

    Returns
    -------
    str
        A complete HTML document string.
    """
    # Normalize Python keys to JS camelCase
    cats_js = []
    for cat in categories:
        c = {
            "name": cat.get("name", ""),
            "label": cat.get("label", cat.get("name", "")),
            "color": cat.get("color", "#888"),
            "items": cat.get("items", []),
        }
        # Watercolor is the default look — use the category's color if no
        # explicit watercolor_color is provided.  Set watercolor_color=None
        # or watercolor_color="" to opt out and get flat solid squares.
        wc = cat.get("watercolor_color", cat.get("watercolorColor", cat.get("color", "#888")))
        if wc:
            c["watercolorColor"] = wc
        cats_js.append(c)

    opts_js = {
        "barWidth": bar_width,
        "logBase": log_base,
        "itemOffset": item_offset,
        "hSqueeze": h_squeeze,
        "barGap": bar_gap,
        "sortDesc": sort_desc,
        "showCount": show_count,
        "outlineRadius": outline_radius,
        "fontFamily": font_family,
        "title": title,
        "subtitle": subtitle,
    }

    js_engine = _load_js()
    cats_json = json.dumps(cats_js, indent=2)
    opts_json = json.dumps(opts_js, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title or 'Pebble Bar Chart'}</title>
<style>
  :root {{
    --pebble-bg: #faf8f2;
    --pebble-surface: #ffffff;
    --pebble-border: #e2e0db;
    --pebble-text: #1a1a1c;
    --pebble-text-secondary: #5a5852;
    --pebble-text-muted: #8a8780;
    --pebble-grid: #87867F;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --pebble-bg: #141416;
      --pebble-surface: #1e1e22;
      --pebble-border: #2e2e34;
      --pebble-text: #e8e6e2;
      --pebble-text-secondary: #a8a6a0;
      --pebble-text-muted: #6a6862;
      --pebble-grid: #55554F;
    }}
  }}
  :root[data-theme="dark"] {{
    --pebble-bg: #141416;
    --pebble-surface: #1e1e22;
    --pebble-border: #2e2e34;
    --pebble-text: #e8e6e2;
    --pebble-text-secondary: #a8a6a0;
    --pebble-text-muted: #6a6862;
    --pebble-grid: #55554F;
  }}

  body {{
    font-family: {font_family};
    background: var(--pebble-bg);
    color: var(--pebble-text);
    padding: 24px 20px 60px;
    margin: 0 auto;
    max-width: 1200px;
  }}

  #pebble-chart {{
    overflow-x: auto;
    /* Note: when overflow-x is not 'visible', browsers force overflow-y
       to 'auto' per CSS spec, so overflow-y: visible has no effect here.
       We use generous padding-bottom instead to keep labels in view. */
    padding-bottom: 60px;
    padding-left: 6px;
  }}

  .pebble-bar-wrapper {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    flex-shrink: 0;
  }}

  .pebble-bar-label {{
    font-size: 0.75rem;
    color: var(--pebble-text-secondary);
    text-align: center;
    white-space: nowrap;
  }}
  {extra_css}
</style>
</head>
<body>
<div id="pebble-chart"></div>
<script>
{js_engine}
</script>
<script>
(function() {{
  var categories = {cats_json};
  var opts = {opts_json};
  var container = document.getElementById('pebble-chart');
  renderPebbleChart(container, categories, opts);
  {extra_js}
}})();
</script>
</body>
</html>"""
    return html


def pebble_bar_chart(
    categories: Sequence[Dict[str, Any]],
    output: Union[str, Path, None] = None,
    *,
    title: str = "",
    subtitle: str = "",
    bar_width: int = 120,
    log_base: float = 1.414,
    item_offset: int = 0,
    h_squeeze: float = 1.0,
    bar_gap: int = 8,
    sort_desc: bool = True,
    show_count: bool = True,
    outline_radius: float = 2,
    font_family: str = "system-ui, -apple-system, sans-serif",
    extra_css: str = "",
    extra_js: str = "",
    open_browser: bool = False,
) -> str:
    """Generate a pebble bar chart as a self-contained HTML file.

    A pebble bar chart is a logarithmic-density stacked bar chart where each
    data point is individually visible. Items stack from large squares at the
    bottom to tiny ones at the top, creating a natural visual hierarchy that
    shows both exact counts and relative magnitudes.

    Parameters
    ----------
    categories : list of dict
        Each category dict should have:

        - ``name`` (str): internal identifier
        - ``label`` (str): display label shown under the bar
        - ``color`` (str): hex color, e.g. ``"#4a90d9"``; also used as the
          watercolor wash by default
        - ``watercolor_color`` (str, optional): override the watercolor wash
          color.  Set to ``None`` or ``""`` to disable watercolor and get
          flat solid squares instead.
        - ``items`` (list of dict): the data points.  Each item dict can have:

          - ``id`` (str): unique identifier
          - ``label`` (str): tooltip text
          - ``link`` (str): URL opened on click
          - ``src`` (str): image URL to display in the cell

    output : str or Path or None
        Path to write the HTML file.  If None, the HTML string is returned
        without writing to disk.
    title : str
        Chart title displayed as a heading.
    subtitle : str
        Description line below the title.
    bar_width : int
        Pixel width of each bar (before horizontal squeeze).  Default 120.
    log_base : float
        Logarithmic base that controls density growth.  Default √2 ≈ 1.414
        doubles columns every 2 rows.  Lower values make taller, skinnier bars.
    item_offset : int
        Invisible padding items at the bottom, hiding the sparse low rows.
    h_squeeze : float
        Horizontal squeeze (0–1).  Default 1.0 (no squeeze).
    bar_gap : int
        Pixels between bars.  Default 8.
    sort_desc : bool
        Sort categories largest-first.  Default True.
    show_count : bool
        Show item count next to labels.  Default True.
    outline_radius : float
        Pixel radius of the white outline drawn around image items (follows the
        alpha mask).  Set to 0 to disable.  Default 2.
    font_family : str
        CSS font stack.
    extra_css : str
        Extra CSS injected into the page.
    extra_js : str
        Extra JavaScript run after chart render.
    open_browser : bool
        Open the file in the default browser after writing.

    Returns
    -------
    str
        The HTML string.  Also written to ``output`` if provided.

    Examples
    --------
    Simple chart with colored bars:

    >>> categories = [
    ...     {"name": "trucks", "label": "Trucks", "color": "#b87200",
    ...      "items": [{"id": str(i)} for i in range(45)]},
    ...     {"name": "vans", "label": "Vans", "color": "#186b4e",
    ...      "items": [{"id": str(i)} for i in range(30)]},
    ...     {"name": "sedans", "label": "Sedans", "color": "#4e82b8",
    ...      "items": [{"id": str(i)} for i in range(18)]},
    ... ]
    >>> html = pebble_bar_chart(categories, "fleet.html",
    ...                         title="Vehicle Fleet", log_base=1.1,
    ...                         item_offset=16, h_squeeze=0.7)

    Watercolor mode for a hand-painted look:

    >>> categories = [
    ...     {"name": "a", "label": "Category A", "color": "#888",
    ...      "watercolor_color": "#b87200",
    ...      "items": [{"id": str(i)} for i in range(60)]},
    ... ]
    >>> pebble_bar_chart(categories, "watercolor.html")
    """
    html = _build_html(
        categories,
        title=title,
        subtitle=subtitle,
        bar_width=bar_width,
        log_base=log_base,
        item_offset=item_offset,
        h_squeeze=h_squeeze,
        bar_gap=bar_gap,
        sort_desc=sort_desc,
        show_count=show_count,
        outline_radius=outline_radius,
        font_family=font_family,
        extra_css=extra_css,
        extra_js=extra_js,
    )

    if output is not None:
        output = Path(output)
        output.write_text(html, encoding="utf-8")

        if open_browser:
            import webbrowser
            webbrowser.open(output.resolve().as_uri())

    return html
