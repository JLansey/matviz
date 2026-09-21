"""
Euler Bar Chart -- overlapping-set bar chart.

Each bar represents the union of two overlapping sets. The two sets are drawn
as two offset, semi-transparent rectangles whose physical overlap zone exactly
represents the measured overlap between the sets. Bar height equals the union
size: set_a + set_b - overlap.

This is different from a standard stacked bar because the segments are NOT
mutually exclusive -- they intentionally overlap, and that overlap is visible
as the blended zone where the two rectangles intersect.

The chart is rendered as a self-contained HTML file with inline CSS and
JavaScript (no external dependencies), making it ideal for quick reports and
data exploration.

Basic usage::

    from matviz.euler_bar import euler_bar_chart

    categories = [
        {"name": "nyc", "label": "NYC", "set_a": 4, "set_b": 1905, "overlap": 4},
        {"name": "bos", "label": "Boston", "set_a": 12, "set_b": 340, "overlap": 8},
    ]
    euler_bar_chart(categories, "overlap.html", title="City Overlap")
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


def _build_html(
    categories: Sequence[Dict[str, Any]],
    *,
    title: str = "",
    subtitle: str = "",
    set_a_label: str = "Set A",
    set_b_label: str = "Set B",
    overlap_label: str = "Overlap",
    color_a: str = "#6A9BCC",
    color_b: str = "#C46686",
    opacity_a: float = 0.72,
    opacity_b: float = 0.62,
    bar_width: int = 80,
    bar_gap: int = 40,
    sort_desc: bool = True,
    show_values: bool = True,
    show_table: bool = True,
    font_family: str = "system-ui, -apple-system, sans-serif",
    extra_css: str = "",
) -> str:
    """Build the full HTML string for an Euler bar chart.

    Parameters
    ----------
    categories : list of dict
        Each dict has keys:

        - ``name`` (str): internal identifier
        - ``label`` (str): display label under the bar
        - ``set_a`` (int or float): size of set A
        - ``set_b`` (int or float): size of set B
        - ``overlap`` (int or float): size of the intersection

    title : str
        Chart heading.
    subtitle : str
        Subtitle / description line.
    set_a_label : str
        Display name for set A (used in legend and table).
    set_b_label : str
        Display name for set B (used in legend and table).
    overlap_label : str
        Display name for the overlap (used in legend and table).
    color_a : str
        CSS color for set A rectangles.
    color_b : str
        CSS color for set B rectangles.
    opacity_a : float
        Opacity for set A rectangles (0--1).
    opacity_b : float
        Opacity for set B rectangles (0--1).
    bar_width : int
        Pixel width of each bar.
    bar_gap : int
        Pixel gap between adjacent bars.
    sort_desc : bool
        Sort categories by union total, largest first.
    show_values : bool
        Show the union total above each bar.
    show_table : bool
        Render a data table below the chart.
    font_family : str
        CSS font-family for the chart.
    extra_css : str
        Additional CSS injected into the ``<style>`` block.

    Returns
    -------
    str
        A complete HTML document string.
    """
    cats_js = []
    for cat in categories:
        set_a = cat.get("set_a", 0)
        set_b = cat.get("set_b", 0)
        overlap = cat.get("overlap", 0)
        cats_js.append({
            "name": cat.get("name", ""),
            "label": cat.get("label", cat.get("name", "")),
            "setA": set_a,
            "setB": set_b,
            "overlap": overlap,
            "union": set_a + set_b - overlap,
        })

    if sort_desc:
        cats_js.sort(key=lambda c: c["union"], reverse=True)

    opts_js = {
        "title": title,
        "subtitle": subtitle,
        "setALabel": set_a_label,
        "setBLabel": set_b_label,
        "overlapLabel": overlap_label,
        "colorA": color_a,
        "colorB": color_b,
        "opacityA": opacity_a,
        "opacityB": opacity_b,
        "barWidth": bar_width,
        "barGap": bar_gap,
        "sortDesc": sort_desc,
        "showValues": show_values,
        "showTable": show_table,
        "fontFamily": font_family,
    }

    cats_json = json.dumps(cats_js, indent=2)
    opts_json = json.dumps(opts_js, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title or 'Euler Bar Chart'}</title>
<style>
  :root {{
    --euler-bg: #faf8f2;
    --euler-surface: #ffffff;
    --euler-border: #e2e0db;
    --euler-text: #1a1a1c;
    --euler-text-secondary: #5a5852;
    --euler-text-muted: #8a8780;
    --euler-grid: #d4d2cc;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --euler-bg: #141416;
      --euler-surface: #1e1e22;
      --euler-border: #2e2e34;
      --euler-text: #e8e6e2;
      --euler-text-secondary: #a8a6a0;
      --euler-text-muted: #6a6862;
      --euler-grid: #3a3a40;
    }}
  }}
  :root[data-theme="dark"] {{
    --euler-bg: #141416;
    --euler-surface: #1e1e22;
    --euler-border: #2e2e34;
    --euler-text: #e8e6e2;
    --euler-text-secondary: #a8a6a0;
    --euler-text-muted: #6a6862;
    --euler-grid: #3a3a40;
  }}

  body {{
    font-family: {font_family};
    background: var(--euler-bg);
    color: var(--euler-text);
    padding: 24px 20px 60px;
    margin: 0 auto;
    max-width: 1200px;
  }}

  .euler-header {{
    margin-bottom: 24px;
  }}
  .euler-header h1 {{
    margin: 0 0 4px;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--euler-text);
  }}
  .euler-header .subtitle {{
    margin: 0;
    font-size: 0.9rem;
    color: var(--euler-text-secondary);
  }}

  .euler-legend {{
    display: flex;
    gap: 20px;
    margin-bottom: 16px;
    font-size: 0.8rem;
    color: var(--euler-text-secondary);
  }}
  .euler-legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .euler-legend-swatch {{
    width: 14px;
    height: 14px;
    border-radius: 2px;
    flex-shrink: 0;
  }}

  #euler-chart {{
    overflow-x: auto;
    padding-bottom: 80px;
  }}

  .euler-chart-area {{
    position: relative;
    display: flex;
    align-items: flex-end;
  }}

  .euler-y-axis {{
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    position: relative;
    margin-right: 8px;
    flex-shrink: 0;
  }}
  .euler-y-tick {{
    position: absolute;
    right: 0;
    font-size: 0.7rem;
    color: var(--euler-text-muted);
    white-space: nowrap;
    transform: translateY(50%);
  }}

  .euler-bars {{
    display: flex;
    align-items: flex-end;
    position: relative;
  }}

  .euler-bar-wrapper {{
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
  }}

  .euler-bar-container {{
    position: relative;
    isolation: isolate;
  }}

  .euler-rect {{
    position: absolute;
    bottom: 0;
    border-radius: 2px;
  }}

  .euler-value-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--euler-text);
    text-align: center;
    margin-bottom: 4px;
    white-space: nowrap;
  }}

  .euler-cat-label {{
    font-size: 0.72rem;
    color: var(--euler-text-secondary);
    text-align: center;
    margin-top: 8px;
    white-space: nowrap;
    transform-origin: top center;
  }}

  .euler-gridline {{
    position: absolute;
    left: 0;
    right: 0;
    border-top: 1px solid var(--euler-grid);
    pointer-events: none;
  }}

  .euler-table-wrap {{
    margin-top: 32px;
    overflow-x: auto;
  }}
  .euler-table {{
    border-collapse: collapse;
    font-size: 0.82rem;
    min-width: 400px;
  }}
  .euler-table th,
  .euler-table td {{
    padding: 6px 14px;
    text-align: right;
    border-bottom: 1px solid var(--euler-border);
  }}
  .euler-table th {{
    font-weight: 600;
    color: var(--euler-text-secondary);
    border-bottom: 2px solid var(--euler-border);
  }}
  .euler-table td:first-child,
  .euler-table th:first-child {{
    text-align: left;
  }}
  .euler-table tr:last-child td {{
    border-bottom: none;
  }}
  {extra_css}
</style>
</head>
<body>
<div id="euler-chart"></div>
<script>
(function() {{
  var categories = {cats_json};
  var opts = {opts_json};
  var container = document.getElementById('euler-chart');

  // Header
  if (opts.title || opts.subtitle) {{
    var header = document.createElement('div');
    header.className = 'euler-header';
    if (opts.title) {{
      var h1 = document.createElement('h1');
      h1.textContent = opts.title;
      header.appendChild(h1);
    }}
    if (opts.subtitle) {{
      var sub = document.createElement('p');
      sub.className = 'subtitle';
      sub.textContent = opts.subtitle;
      header.appendChild(sub);
    }}
    container.appendChild(header);
  }}

  // Legend
  var legend = document.createElement('div');
  legend.className = 'euler-legend';
  var items = [
    {{ label: opts.setALabel, color: opts.colorA, opacity: opts.opacityA }},
    {{ label: opts.setBLabel, color: opts.colorB, opacity: opts.opacityB }},
  ];
  items.forEach(function(it) {{
    var el = document.createElement('div');
    el.className = 'euler-legend-item';
    var sw = document.createElement('div');
    sw.className = 'euler-legend-swatch';
    sw.style.background = it.color;
    sw.style.opacity = it.opacity;
    el.appendChild(sw);
    var lbl = document.createElement('span');
    lbl.textContent = it.label;
    el.appendChild(lbl);
    legend.appendChild(el);
  }});
  container.appendChild(legend);

  if (categories.length === 0) return;

  // Determine max union for scaling
  var maxUnion = Math.max.apply(null, categories.map(function(c) {{ return c.union; }}));
  if (maxUnion <= 0) maxUnion = 1;

  var chartHeight = 320;
  var rectWidth = opts.barWidth * 0.42;
  var offsetX = 10;  // horizontal offset for set B

  // Nice tick values
  function niceTicks(maxVal, approxCount) {{
    if (maxVal <= 0) return [0];
    var raw = maxVal / approxCount;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step;
    if (norm <= 1.5) step = mag;
    else if (norm <= 3) step = 2 * mag;
    else if (norm <= 7) step = 5 * mag;
    else step = 10 * mag;
    var ticks = [];
    for (var v = 0; v <= maxVal + step * 0.01; v += step) {{
      ticks.push(Math.round(v * 1e6) / 1e6);
    }}
    return ticks;
  }}

  var ticks = niceTicks(maxUnion, 5);
  var scaleMax = ticks[ticks.length - 1] || maxUnion;
  function yPos(val) {{
    return (val / scaleMax) * chartHeight;
  }}

  // Chart area
  var chartArea = document.createElement('div');
  chartArea.className = 'euler-chart-area';

  // Y-axis
  var yAxis = document.createElement('div');
  yAxis.className = 'euler-y-axis';
  yAxis.style.height = chartHeight + 'px';
  yAxis.style.width = '48px';
  ticks.forEach(function(t) {{
    var tick = document.createElement('div');
    tick.className = 'euler-y-tick';
    tick.textContent = t >= 1000 ? (t / 1000) + 'k' : t;
    tick.style.bottom = yPos(t) + 'px';
    yAxis.appendChild(tick);
  }});
  chartArea.appendChild(yAxis);

  // Bars container (with gridlines)
  var barsOuter = document.createElement('div');
  barsOuter.style.position = 'relative';
  barsOuter.style.flexGrow = '1';

  // Gridlines
  var totalBarsWidth = categories.length * (opts.barWidth + opts.barGap) - opts.barGap;
  ticks.forEach(function(t) {{
    var line = document.createElement('div');
    line.className = 'euler-gridline';
    line.style.bottom = yPos(t) + 'px';
    line.style.width = totalBarsWidth + 'px';
    barsOuter.appendChild(line);
  }});

  var barsRow = document.createElement('div');
  barsRow.className = 'euler-bars';
  barsRow.style.height = chartHeight + 'px';

  categories.forEach(function(cat, i) {{
    var wrapper = document.createElement('div');
    wrapper.className = 'euler-bar-wrapper';
    wrapper.style.width = opts.barWidth + 'px';
    if (i < categories.length - 1) {{
      wrapper.style.marginRight = opts.barGap + 'px';
    }}

    // Value label
    if (opts.showValues) {{
      var vlbl = document.createElement('div');
      vlbl.className = 'euler-value-label';
      vlbl.textContent = cat.union;
      wrapper.appendChild(vlbl);
    }}

    // Bar container
    var barCont = document.createElement('div');
    barCont.className = 'euler-bar-container';
    barCont.style.width = (rectWidth + offsetX) + 'px';
    barCont.style.height = yPos(cat.union) + 'px';

    // Set A rect: flush left, starts at baseline, height = setA
    var rectA = document.createElement('div');
    rectA.className = 'euler-rect';
    rectA.style.left = '0';
    rectA.style.width = rectWidth + 'px';
    rectA.style.height = yPos(cat.setA) + 'px';
    rectA.style.background = opts.colorA;
    rectA.style.opacity = opts.opacityA;
    rectA.style.mixBlendMode = 'multiply';
    rectA.style.bottom = '0';
    rectA.title = opts.setALabel + ': ' + cat.setA;
    barCont.appendChild(rectA);

    // Set B rect: offset right, starts at (setA - overlap) from baseline
    var rectB = document.createElement('div');
    rectB.className = 'euler-rect';
    rectB.style.left = offsetX + 'px';
    rectB.style.width = rectWidth + 'px';
    rectB.style.height = yPos(cat.setB) + 'px';
    rectB.style.background = opts.colorB;
    rectB.style.opacity = opts.opacityB;
    rectB.style.mixBlendMode = 'multiply';
    rectB.style.bottom = yPos(cat.setA - cat.overlap) + 'px';
    rectB.title = opts.setBLabel + ': ' + cat.setB;
    barCont.appendChild(rectB);

    wrapper.appendChild(barCont);

    // Category label
    var clbl = document.createElement('div');
    clbl.className = 'euler-cat-label';
    clbl.textContent = cat.label;
    if (cat.label.length > 8) {{
      clbl.style.transform = 'rotate(-52deg)';
      clbl.style.transformOrigin = 'top center';
    }}
    wrapper.appendChild(clbl);

    barsRow.appendChild(wrapper);
  }});

  barsOuter.appendChild(barsRow);
  chartArea.appendChild(barsOuter);
  container.appendChild(chartArea);

  // Data table
  if (opts.showTable) {{
    var tableWrap = document.createElement('div');
    tableWrap.className = 'euler-table-wrap';
    var table = document.createElement('table');
    table.className = 'euler-table';

    // Header
    var thead = document.createElement('thead');
    var hrow = document.createElement('tr');
    ['Category', opts.setALabel, opts.setBLabel, opts.overlapLabel, 'Union'].forEach(function(h) {{
      var th = document.createElement('th');
      th.textContent = h;
      hrow.appendChild(th);
    }});
    thead.appendChild(hrow);
    table.appendChild(thead);

    // Body
    var tbody = document.createElement('tbody');
    categories.forEach(function(cat) {{
      var tr = document.createElement('tr');
      [cat.label, cat.setA, cat.setB, cat.overlap, cat.union].forEach(function(val) {{
        var td = document.createElement('td');
        td.textContent = val;
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    container.appendChild(tableWrap);
  }}
}})();
</script>
</body>
</html>"""
    return html


def euler_bar_chart(
    categories: Sequence[Dict[str, Any]],
    output: Union[str, Path, None] = None,
    *,
    title: str = "",
    subtitle: str = "",
    set_a_label: str = "Set A",
    set_b_label: str = "Set B",
    overlap_label: str = "Overlap",
    color_a: str = "#6A9BCC",
    color_b: str = "#C46686",
    opacity_a: float = 0.72,
    opacity_b: float = 0.62,
    bar_width: int = 80,
    bar_gap: int = 40,
    sort_desc: bool = True,
    show_values: bool = True,
    show_table: bool = True,
    font_family: str = "system-ui, -apple-system, sans-serif",
    extra_css: str = "",
    open_browser: bool = False,
) -> str:
    """Generate an Euler bar chart as a self-contained HTML file.

    An Euler bar chart visualizes overlapping sets. Each bar represents the
    union of two sets, drawn as two offset, semi-transparent rectangles whose
    physical overlap zone matches the measured intersection. Bar height equals
    set_a + set_b - overlap (the inclusion-exclusion union).

    Parameters
    ----------
    categories : list of dict
        Each category dict should have:

        - ``name`` (str): internal identifier
        - ``label`` (str): display label shown under the bar
        - ``set_a`` (int or float): size of set A
        - ``set_b`` (int or float): size of set B
        - ``overlap`` (int or float): size of the intersection (must be
          <= min(set_a, set_b))

    output : str or Path or None
        Path to write the HTML file.  If None, the HTML string is returned
        without writing to disk.
    title : str
        Chart title displayed as a heading.
    subtitle : str
        Description line below the title.
    set_a_label : str
        Display name for set A in legend and table.  Default ``"Set A"``.
    set_b_label : str
        Display name for set B in legend and table.  Default ``"Set B"``.
    overlap_label : str
        Display name for the overlap in the table.  Default ``"Overlap"``.
    color_a : str
        CSS color for set A rectangles.  Default ``"#6A9BCC"`` (blue).
    color_b : str
        CSS color for set B rectangles.  Default ``"#C46686"`` (rose).
    opacity_a : float
        Opacity for set A rectangles (0--1).  Default 0.72.
    opacity_b : float
        Opacity for set B rectangles (0--1).  Default 0.62.
    bar_width : int
        Pixel width of each bar column.  Default 80.
    bar_gap : int
        Pixel gap between adjacent bars.  Default 40.
    sort_desc : bool
        Sort categories by union total, largest first.  Default True.
    show_values : bool
        Show the union total above each bar.  Default True.
    show_table : bool
        Render a data table below the chart.  Default True.
    font_family : str
        CSS font stack.
    extra_css : str
        Extra CSS injected into the page.
    open_browser : bool
        Open the file in the default browser after writing.

    Returns
    -------
    str
        The HTML string.  Also written to ``output`` if provided.

    Examples
    --------
    Basic overlapping-set chart:

    >>> categories = [
    ...     {"name": "nyc", "label": "NYC",
    ...      "set_a": 4, "set_b": 1905, "overlap": 4},
    ...     {"name": "bos", "label": "Boston",
    ...      "set_a": 12, "set_b": 340, "overlap": 8},
    ...     {"name": "cam", "label": "Cambridge",
    ...      "set_a": 3, "set_b": 120, "overlap": 2},
    ... ]
    >>> html = euler_bar_chart(
    ...     categories, "overlap.html",
    ...     title="City Reports",
    ...     set_a_label="Filed",
    ...     set_b_label="Total",
    ...     overlap_label="Both",
    ... )

    Without the data table:

    >>> euler_bar_chart(categories, show_table=False)
    """
    html = _build_html(
        categories,
        title=title,
        subtitle=subtitle,
        set_a_label=set_a_label,
        set_b_label=set_b_label,
        overlap_label=overlap_label,
        color_a=color_a,
        color_b=color_b,
        opacity_a=opacity_a,
        opacity_b=opacity_b,
        bar_width=bar_width,
        bar_gap=bar_gap,
        sort_desc=sort_desc,
        show_values=show_values,
        show_table=show_table,
        font_family=font_family,
        extra_css=extra_css,
    )

    if output is not None:
        output = Path(output)
        output.write_text(html, encoding="utf-8")

        if open_browser:
            import webbrowser
            webbrowser.open(output.resolve().as_uri())

    return html
