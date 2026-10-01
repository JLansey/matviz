/**
 * Pebble view — shows a pebble bar chart pre-rendered by matviz
 * (matviz.pebble_bar) and links every pebble. All drawing happens in Python;
 * this only places the bar images, draws the axis at the given heights and
 * maps the pointer to the item under it.
 *
 *   var chart = renderPebbleView(container, manifest, { baseUrl, fontFamily });
 *   chart.yAxis  — the axis element (move it out of a scroll container to pin it)
 *   chart.bars   — one .pebble-bar-wrapper per category, in manifest order
 *
 * Part of the matviz library — https://github.com/JLansey/matviz
 */
function renderPebbleView(container, data, opts) {
  opts = opts || {};
  var o = data.options, cats = data.categories;
  var base = opts.baseUrl || '';
  var font = opts.fontFamily || 'system-ui, -apple-system, sans-serif';
  var bottom = o.topPad + o.maxHeight, fullH = bottom + o.labelHeight;
  var widths = cats.map(function (c) { return c.width; });
  var chartW = widths.reduce(function (s, w) { return s + w + o.barGap; }, 0) - o.barGap;
  function el(tag, css, parent) {
    var e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (parent) parent.appendChild(e);
    return e;
  }
  function lines(width, labels) {
    var s = '';
    data.ticks.forEach(function (t) {
      var y = (bottom - t.height).toFixed(1);
      if (labels) s += '<text x="4" y="' + (y - 4).toFixed(1) + '" fill="var(--pebble-text-muted, #87867F)" ' +
        'font-size="12" font-weight="500" font-family="' + font.replace(/"/g, "'") + '">' + t.value + '</text>';
      s += '<line x1="0" y1="' + y + '" x2="' + width + '" y2="' + y + '" stroke="var(--pebble-grid, #87867F)" stroke-opacity="0.28" stroke-width="0.75"/>';
    });
    return s + '<line x1="0" y1="' + bottom + '" x2="' + width + '" y2="' + bottom + '" stroke="var(--pebble-text-muted, #8a8780)" stroke-opacity="0.5" stroke-width="1.5"/>';
  }
  var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="%w" height="' + fullH + '" style="display:block;overflow:visible">';

  var outer = el('div', 'position:relative;overflow:visible;min-height:' + fullH + 'px;font-family:' + font, container);
  var yAxis = el('div', 'position:absolute;top:0;left:0;width:' + o.axisWidth + 'px;height:' + fullH + 'px;z-index:5;' +
    'pointer-events:none;background:var(--pebble-bg);padding-left:60px;margin-left:-60px;box-sizing:content-box', outer);
  yAxis.className = 'pebble-chart-yaxis';
  yAxis.innerHTML = svg.replace('%w', o.axisWidth) + lines(o.axisWidth, true) + '</svg>';
  var area = el('div', 'position:relative;overflow:visible;margin-left:' + o.axisWidth + 'px', outer);
  area.innerHTML = svg.replace('%w', chartW + 26).replace('style="', 'style="position:absolute;top:0;left:0;pointer-events:none;') +
    lines(chartW + 10, false) + '</svg>';
  var row = el('div', 'display:flex;align-items:flex-end;gap:' + o.barGap + 'px;padding:' + o.topPad +
    'px 6px 0;position:relative;z-index:1;overflow:visible', area);

  var bars = cats.map(function (cat) {
    var wrap = el('div', 'position:relative;overflow:visible;width:' + cat.width + 'px', row);
    wrap.className = 'pebble-bar-wrapper';
    var plot = el('div', 'position:relative;width:' + cat.width + 'px;height:' + cat.height + 'px', wrap);
    var img = el('img', 'position:absolute;display:block;max-width:none;left:' + cat.box[0] + 'px;top:' +
      cat.box[1] + 'px;width:' + cat.box[2] + 'px;height:' + cat.box[3] + 'px', plot);
    img.src = /^(data:|https?:|\/)/.test(cat.image) ? cat.image : base + cat.image;
    img.alt = cat.label + ': ' + cat.count;
    img.decoding = 'async';
    var label = el('div', 'margin-top:8px;font-size:0.875rem;font-weight:700;line-height:1.2;overflow:hidden;' +
      'text-overflow:ellipsis;max-width:' + (cat.width + 4) + 'px', wrap);
    label.className = 'pebble-bar-label';
    label.textContent = cat.label;
    if (o.showCount) el('span', 'margin-left:4px;font-weight:500;color:var(--pebble-text-muted, #8a8780)', label).textContent = cat.count;

    // Item under (x, y) in plot coordinates: binary search the row, then the
    // nearest column centre (squeezed neighbours overlap).
    function hit(x, y) {
      var rows = cat.rows, lo = 0, hi = rows.length - 1;
      while (lo < hi) { var mid = (lo + hi + 1) >> 1; if (rows[mid][0] <= y) lo = mid; else hi = mid - 1; }
      var r = rows[lo];
      if (!r || y < r[0] || y >= r[0] + r[1]) return null;
      var size = r[1], cols = r[2], c0 = r[5] ? Math.round((x - r[4] - size / 2) / r[5]) : 0, best = null, bestD = Infinity;
      for (var c = c0 - 1; c <= c0 + 1; c++) {
        var i = r[3] + c, left = r[4] + c * r[5];
        if (c < 0 || c >= cols || i < 0 || i >= cat.items.length || x < left || x >= left + size) continue;
        var d = Math.abs(x - left - size / 2);
        if (d < bestD) { bestD = d; best = { item: cat.items[i], x: left, y: r[0], size: size }; }
      }
      return best;
    }
    var hot = null, hotItem = null;
    function at(event) {
      var b = plot.getBoundingClientRect();
      return hit(event.clientX - b.left, event.clientY - b.top);
    }
    function clear() { if (hot) hot.remove(); hot = hotItem = null; }
    // One real link over the hovered item, recreated per item so the browser
    // tooltip, URL preview and cmd/middle-click all apply to that item.
    plot.addEventListener('mousemove', function (event) {
      var h = at(event);
      if ((h && h.item) === hotItem) return;
      clear();
      if (!h) return;
      hotItem = h.item;
      hot = el(h.item.link ? 'a' : 'div', 'position:absolute;z-index:2;left:' + h.x + 'px;top:' + h.y + 'px;width:' +
        h.size + 'px;height:' + h.size + 'px;cursor:' + (h.item.link ? 'pointer' : 'default'), plot);
      if (h.item.link) { hot.href = h.item.link; hot.target = '_blank'; hot.rel = 'noopener'; }
      hot.title = h.item.label || h.item.id || '';
    });
    plot.addEventListener('mouseleave', clear);
    // Taps without a preceding mousemove still open the item.
    plot.addEventListener('click', function (event) {
      if (event.target.closest && event.target.closest('a')) return;
      var h = at(event);
      if (h && h.item.link) window.open(h.item.link, '_blank', 'noopener');
    });
    return wrap;
  });
  return { root: outer, yAxis: yAxis, bars: bars };
}
