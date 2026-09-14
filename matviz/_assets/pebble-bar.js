/**
 * Pebble Bar Chart — logarithmic-density stacked bar chart
 *
 * Each bar is a vertical stack of rows. Row r (0 = bottom) has floor(base^r)
 * columns. Each cell is barWidth / base^r wide and tall (square). When
 * floor(base^r) cells don't fill the full bar width, horizontal gaps appear
 * between items (space-between justify). The log base is configurable
 * (default √2 ≈ 1.414 — columns double every 2 rows).
 *
 * itemOffset: prepend N invisible items so the visible chart starts at a
 * higher row (e.g. skip past the 1- and 2-column rows). The clipped bottom
 * rows are hidden and the count label shows real items only.
 *
 * Part of the matviz library — https://github.com/JLansey/matviz
 *
 * Usage:
 *   renderPebbleBar(container, { barWidth, items, label, color, logBase, itemOffset })
 *   renderPebbleChart(container, categories, opts)
 */

/* ── helpers ────────────────────────────────────────────────────────── */

function computeBarHeight(itemCount, barWidth, logBase, itemOffset, gutter) {
  if (itemCount <= 0) return 0;
  var allCount = itemOffset + itemCount;
  var remaining = allCount, r = 0, rows = [];
  while (remaining > 0) {
    var idealCols = Math.pow(logBase, r);
    var cols = Math.min(Math.floor(idealCols), remaining);
    var imageSize = barWidth / idealCols;
    rows.push({ cols: cols, imageSize: imageSize });
    remaining -= cols;
    r++;
  }
  var padRemaining = itemOffset, cropHeight = 0;
  for (var i = 0; i < rows.length; i++) {
    if (padRemaining >= rows[i].cols) {
      cropHeight += rows[i].imageSize + gutter;
      padRemaining -= rows[i].cols;
    } else break;
  }
  var fullHeight = rows.reduce(function (s, row) { return s + row.imageSize + gutter; }, 0)
                   - (rows.length > 0 ? gutter : 0);
  return Math.max(0, fullHeight - cropHeight);
}

function logTicks(maxVal, minVal) {
  // 1-2-5 sequence — evenly spaced in log space
  minVal = minVal || 3;
  var ticks = [];
  var seq = [1, 2, 5];
  var mag = 1;
  outer:
  while (true) {
    for (var i = 0; i < seq.length; i++) {
      var v = seq[i] * mag;
      if (v > maxVal * 1.1) break outer;
      if (v >= minVal) ticks.push(v);
    }
    mag *= 10;
  }
  return ticks;
}

function lightenHex(hex, amount) {
  var r = parseInt(hex.slice(1, 3), 16);
  var g = parseInt(hex.slice(3, 5), 16);
  var b = parseInt(hex.slice(5, 7), 16);
  r = Math.min(255, Math.round(r + (255 - r) * amount));
  g = Math.min(255, Math.round(g + (255 - g) * amount));
  b = Math.min(255, Math.round(b + (255 - b) * amount));
  return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/* ── single bar ─────────────────────────────────────────────────────── */

function renderPebbleBar(container, opts) {
  var barWidth   = opts.barWidth   || 120;
  var hSqueeze   = opts.hSqueeze   != null ? opts.hSqueeze : 1;
  var gutter     = opts.gutter     != null ? opts.gutter : 0;
  var items      = opts.items      || [];
  var label      = opts.label      || '';
  var color      = opts.color      || '#888';
  var logBase    = opts.logBase    || Math.SQRT2;
  var itemOffset = opts.itemOffset || 0;
  var watercolorColor = opts.watercolorColor || null;
  var rotateLabel = opts.rotateLabel || false;
  var labelHeight = opts.labelHeight || 70;
  var showCount   = opts.showCount != null ? opts.showCount : true;
  var outlineRadius = opts.outlineRadius != null ? opts.outlineRadius : 2;

  if (items.length === 0) return;

  // Prepend invisible padding items
  var allItems = [];
  for (var i = 0; i < itemOffset; i++) allItems.push({ _pad: true });
  allItems.push.apply(allItems, items);

  // Build row specs: row 0 = bottom (biggest), going up
  var rows = [];
  var remaining = allItems.length;
  var r = 0;
  while (remaining > 0) {
    var idealCols = Math.pow(logBase, r);
    var cols = Math.min(Math.floor(idealCols), remaining);
    var imageSize = barWidth / idealCols;
    rows.push({ cols: cols, imageSize: imageSize, r: r });
    remaining -= cols;
    r++;
  }

  // Find the crop point: height of rows that contain ONLY padding items
  var padRemaining = itemOffset;
  var cropHeight = 0;
  for (var ri = 0; ri < rows.length; ri++) {
    if (padRemaining >= rows[ri].cols) {
      cropHeight += rows[ri].imageSize + gutter;
      padRemaining -= rows[ri].cols;
    } else break;
  }

  // Total height of all rows
  var fullHeight = rows.reduce(function (s, row) { return s + row.imageSize + gutter; }, 0) - gutter;
  var visibleHeight = fullHeight - cropHeight;

  // Squeezed visual width
  var squeezedWidth = barWidth * hSqueeze;
  var squeezeInset = (barWidth - squeezedWidth) / 2;

  // Watercolor bleed
  var bleedX = watercolorColor ? 5 : 0;
  var bleedTop = watercolorColor ? 4 : 0;
  var bleedBot = watercolorColor ? 3 : 0;

  var wrapper = document.createElement('div');
  wrapper.className = 'pebble-bar-wrapper';
  wrapper.style.width = squeezedWidth + 'px';
  wrapper.style.position = 'relative';
  wrapper.style.overflow = 'visible';

  // ── watercolor SVG ──
  if (watercolorColor) {
    var seed1 = Math.floor(Math.random() * 10000);
    var seed2 = Math.floor(Math.random() * 10000);
    var filterId = 'wc-' + seed1 + '-' + seed2;
    var svgW = squeezedWidth + bleedX * 2;
    var svgH = visibleHeight + bleedTop + bleedBot;
    var svgNS = 'http://www.w3.org/2000/svg';
    var svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('width', svgW);
    svg.setAttribute('height', svgH);
    svg.style.position = 'absolute';
    svg.style.top = -bleedTop + 'px';
    svg.style.left = -bleedX + 'px';
    svg.style.pointerEvents = 'none';
    svg.style.zIndex = '0';
    svg.style.overflow = 'visible';
    svg.innerHTML =
      '<defs>' +
        '<filter id="' + filterId + '" x="-8%" y="-8%" width="116%" height="116%" color-interpolation-filters="sRGB">' +
          '<feTurbulence type="fractalNoise" baseFrequency="0.55" numOctaves="2" seed="' + seed1 + '" result="grain"/>' +
          '<feColorMatrix in="grain" type="luminanceToAlpha" result="grainAlpha"/>' +
          '<feComposite in="SourceGraphic" in2="grainAlpha" operator="arithmetic" k1="0" k2="1" k3="-0.30" k4="0" result="inked"/>' +
          '<feTurbulence type="fractalNoise" baseFrequency="0.11" numOctaves="2" seed="' + seed2 + '" result="warp"/>' +
          '<feDisplacementMap in="inked" in2="warp" scale="2.6" xChannelSelector="R" yChannelSelector="G"/>' +
        '</filter>' +
      '</defs>' +
      '<rect x="' + bleedX + '" y="' + bleedTop + '" width="' + squeezedWidth + '" height="' + visibleHeight + '" fill="var(--pebble-bg, #faf8f2)"/>' +
      '<g style="isolation:isolate;mix-blend-mode:multiply">' +
        '<rect x="0" y="0" width="' + svgW + '" height="' + svgH + '" ' +
          'fill="' + watercolorColor + '" fill-opacity="0.68" filter="url(#' + filterId + ')"/>' +
      '</g>';
    wrapper.appendChild(svg);
  }

  // Clip container
  var clipEl = document.createElement('div');
  clipEl.style.width = squeezedWidth + 'px';
  clipEl.style.height = visibleHeight + 'px';
  clipEl.style.position = 'relative';
  clipEl.style.zIndex = '1';
  clipEl.style.clipPath = 'inset(0px -50px 0px -50px)';

  var barEl = document.createElement('div');
  barEl.style.position = 'absolute';
  barEl.style.width = squeezedWidth + 'px';
  barEl.style.height = fullHeight + 'px';
  barEl.style.top = '0px';

  // Place items bottom-up
  var itemIdx = 0;
  var yFromBottom = 0;
  var placedEls = [];

  for (var ri = 0; ri < rows.length; ri++) {
    var row = rows[ri];
    var y = fullHeight - yFromBottom - row.imageSize;

    for (var c = 0; c < row.cols && itemIdx < allItems.length; c++) {
      var item = allItems[itemIdx];
      itemIdx++;
      if (item._pad) continue;

      // Skip items that have no visual content and no link — they still
      // count toward bar height, but an invisible div would just occlude
      // items beneath it.
      var hasVisual = item.src || !watercolorColor;
      var hasLink   = item.link;
      if (!hasVisual && !hasLink) continue;

      // Space-between justify
      var x;
      if (row.cols === 1) {
        x = (barWidth - row.imageSize) / 2;
      } else {
        var step = (barWidth - row.imageSize) / (row.cols - 1);
        x = c * step;
      }
      // Apply horizontal squeeze
      var barCenter = barWidth / 2;
      var imgCenter = x + row.imageSize / 2;
      imgCenter = barCenter + (imgCenter - barCenter) * hSqueeze;
      x = imgCenter - row.imageSize / 2 - squeezeInset;

      var el = document.createElement('div');
      el.style.position = 'absolute';
      el.style.left = x + 'px';
      el.style.top = y + 'px';
      el.style.width = row.imageSize + 'px';
      el.style.height = row.imageSize + 'px';
      el.style.borderRadius = Math.max(1, Math.min(3, row.imageSize * 0.05)) + 'px';

      if (item.src) {
        // Image items need overflow:hidden to clip to the rounded rect
        el.style.overflow = 'hidden';
        el.style.backgroundColor = 'transparent';

        var img = document.createElement('img');
        img.src = item.src;
        img.alt = item.label || item.id || '';
        img.loading = 'lazy';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        img.style.display = 'block';
        el.appendChild(img);

        // Solid white outline that follows the alpha mask
        if (outlineRadius > 0) {
          (function(imgEl, size, radius) {
            imgEl.addEventListener('load', function() {
              var canvas = document.createElement('canvas');
              canvas.width = size;
              canvas.height = size;
              var ctx = canvas.getContext('2d');
              // Draw white silhouette offset in 8 directions to form outline
              ctx.globalCompositeOperation = 'source-over';
              var steps = 8;
              for (var a = 0; a < steps; a++) {
                var angle = (a / steps) * Math.PI * 2;
                var dx = Math.cos(angle) * radius;
                var dy = Math.sin(angle) * radius;
                ctx.drawImage(imgEl, dx, dy, size, size);
              }
              // Turn everything drawn so far into solid white
              ctx.globalCompositeOperation = 'source-in';
              ctx.fillStyle = 'white';
              ctx.fillRect(0, 0, size, size);
              // Draw original image on top
              ctx.globalCompositeOperation = 'source-over';
              ctx.drawImage(imgEl, 0, 0, size, size);
              // Replace the img with the canvas
              canvas.style.width = '100%';
              canvas.style.height = '100%';
              canvas.style.display = 'block';
              if (imgEl.parentNode) {
                imgEl.parentNode.replaceChild(canvas, imgEl);
              }
            });
          })(img, Math.ceil(row.imageSize * (window.devicePixelRatio || 1)), outlineRadius * (window.devicePixelRatio || 1));
        }
      } else {
        // Non-image item: solid color square (flat-fill mode only;
        // watercolor items without src+link were already skipped above)
        el.style.backgroundColor = color;
      }

      // Tooltip
      if (item.label || item.id) {
        el.title = item.label || item.id || '';
      }

      // Clickable link
      if (item.link) {
        el.style.cursor = 'pointer';
        el.addEventListener('click', (function (link) {
          return function () { window.open(link, '_blank'); };
        })(item.link));
      }

      placedEls.push(el);
    }
    yFromBottom += row.imageSize + gutter;
  }

  // Shuffle DOM order for natural overlap
  for (var si = placedEls.length - 1; si > 0; si--) {
    var sj = Math.floor(Math.random() * (si + 1));
    var tmp = placedEls[si];
    placedEls[si] = placedEls[sj];
    placedEls[sj] = tmp;
  }
  for (var si = 0; si < placedEls.length; si++) {
    barEl.appendChild(placedEls[si]);
  }

  clipEl.appendChild(barEl);
  wrapper.appendChild(clipEl);

  // ── label ──
  var labelEl = document.createElement('div');
  labelEl.className = 'pebble-bar-label';
  if (rotateLabel) {
    labelEl.style.height = labelHeight + 'px';
    labelEl.style.overflow = 'visible';
    labelEl.style.position = 'relative';
    var span = document.createElement('span');
    span.style.position = 'absolute';
    span.style.top = '6px';
    span.style.left = '50%';
    span.style.transformOrigin = '100% 0';
    span.style.transform = 'translateX(-100%) rotate(-40deg)';
    span.style.whiteSpace = 'nowrap';
    span.style.fontSize = '1.05rem';
    span.style.fontWeight = '700';
    span.style.letterSpacing = '0.01em';
    span.innerHTML = label;
    labelEl.appendChild(span);
  } else {
    var countStr = showCount ? ' <span style="opacity:0.6">' + items.length + '</span>' : '';
    labelEl.innerHTML = '<strong>' + label + '</strong>' + countStr;
  }
  wrapper.appendChild(labelEl);

  container.appendChild(wrapper);
  return wrapper;
}

/* ── full chart with y-axis and paper grid ──────────────────────────── */

function renderPebbleChart(container, categories, opts) {
  opts = opts || {};
  var barWidth     = opts.barWidth     || 120;
  var gutter       = opts.gutter       != null ? opts.gutter : 0;
  var logBase      = opts.logBase      || Math.SQRT2;
  var itemOffset   = opts.itemOffset   || 0;
  var sortDesc     = opts.sortDesc     != null ? opts.sortDesc : true;
  var barGap       = opts.barGap       || 8;
  var hSqueeze     = opts.hSqueeze    != null ? opts.hSqueeze : 1;
  var showCount    = opts.showCount   != null ? opts.showCount : true;
  var outlineRadius = opts.outlineRadius != null ? opts.outlineRadius : 2;
  var title        = opts.title        || '';
  var subtitle     = opts.subtitle     || '';
  var fontFamily   = opts.fontFamily   || 'system-ui, -apple-system, sans-serif';

  var cats = categories.slice();
  if (sortDesc) {
    cats = cats.sort(function (a, b) { return b.items.length - a.items.length; });
  }

  // ── dimensions ──
  var maxCount = Math.max.apply(null, cats.map(function (c) { return c.items.length; }));
  var maxHeight = computeBarHeight(maxCount, barWidth, logBase, itemOffset, gutter);
  var yAxisWidth = 48;
  var labelHeight = 80;
  var topPad = 24;
  var squeezedBarWidth = barWidth * hSqueeze;
  var chartWidth = cats.length * (squeezedBarWidth + barGap) - barGap;

  // ── outer wrapper ──
  var outer = document.createElement('div');
  outer.style.position = 'relative';
  outer.style.minHeight = (topPad + maxHeight + labelHeight) + 'px';
  outer.style.overflow = 'visible';
  outer.style.fontFamily = fontFamily;

  // ── title ──
  if (title) {
    var titleEl = document.createElement('h2');
    titleEl.style.fontSize = '1.5rem';
    titleEl.style.fontWeight = '700';
    titleEl.style.margin = '0 0 4px';
    titleEl.style.color = 'var(--pebble-text, #1a1a1c)';
    titleEl.textContent = title;
    outer.appendChild(titleEl);
  }
  if (subtitle) {
    var subEl = document.createElement('p');
    subEl.style.color = 'var(--pebble-text-secondary, #5a5852)';
    subEl.style.fontSize = '0.875rem';
    subEl.style.margin = '0 0 28px';
    subEl.textContent = subtitle;
    outer.appendChild(subEl);
  }

  // ── chart area ──
  var chartArea = document.createElement('div');
  chartArea.style.position = 'relative';
  chartArea.style.minHeight = (topPad + maxHeight + labelHeight) + 'px';
  chartArea.style.overflow = 'visible';

  // ── SVG background: paper grid + y-axis + horizontal guides ──
  var svgH = topPad + maxHeight + labelHeight;
  var svgW = yAxisWidth + 6 + chartWidth + 6 + 20;
  var chartBottom = topPad + maxHeight;
  var svgNS = 'http://www.w3.org/2000/svg';
  var bgSvg = document.createElementNS(svgNS, 'svg');
  bgSvg.setAttribute('width', svgW);
  bgSvg.setAttribute('height', svgH);
  bgSvg.style.position = 'absolute';
  bgSvg.style.top = '0';
  bgSvg.style.left = '0';
  bgSvg.style.pointerEvents = 'none';
  bgSvg.style.overflow = 'visible';

  var svg = '';

  // Paper-rule filter
  svg += '<defs><filter id="mk-rule" x="-8%" y="-8%" width="116%" height="116%" color-interpolation-filters="sRGB">';
  svg += '<feTurbulence type="fractalNoise" baseFrequency="0.72" numOctaves="2" seed="5" result="grain"/>';
  svg += '<feColorMatrix in="grain" type="luminanceToAlpha" result="ga"/>';
  svg += '<feComposite in="SourceGraphic" in2="ga" operator="arithmetic" k1="0" k2="1" k3="-0.12" k4="0" result="inked"/>';
  svg += '<feTurbulence type="fractalNoise" baseFrequency="0.07" numOctaves="2" seed="3" result="warp"/>';
  svg += '<feDisplacementMap in="inked" in2="warp" scale="0.6" xChannelSelector="R" yChannelSelector="G"/>';
  svg += '</filter></defs>';

  // Paper grid
  var gridSpacing = 52;
  var gridPaths = '';
  for (var gy = 0; gy <= svgH; gy += gridSpacing) {
    gridPaths += '<path d="M0 ' + gy + ' L' + svgW + ' ' + gy + '" fill="none" stroke="var(--pebble-grid, #87867F)" stroke-opacity="0.24" stroke-width="1"/>';
  }
  for (var gx = 0; gx <= svgW; gx += gridSpacing) {
    gridPaths += '<path d="M' + gx + ' 0 L' + gx + ' ' + svgH + '" fill="none" stroke="var(--pebble-grid, #87867F)" stroke-opacity="0.24" stroke-width="1"/>';
  }
  svg += '<g filter="url(#mk-rule)" aria-hidden="true">' + gridPaths + '</g>';

  // Baseline
  svg += '<line x1="' + yAxisWidth + '" y1="' + chartBottom +
         '" x2="' + (yAxisWidth + chartWidth + 10) + '" y2="' + chartBottom +
         '" stroke="var(--pebble-text-muted, #8a8780)" stroke-opacity="0.5" stroke-width="1.5"/>';

  // Y-axis ticks — log-spaced (1-2-5 sequence)
  var ticks = logTicks(maxCount);
  for (var ti = 0; ti < ticks.length; ti++) {
    var tick = ticks[ti];
    var h = computeBarHeight(tick, barWidth, logBase, itemOffset, gutter);
    var ty = chartBottom - h;
    svg += '<line x1="' + yAxisWidth + '" y1="' + ty.toFixed(1) +
           '" x2="' + (yAxisWidth + chartWidth + 10) + '" y2="' + ty.toFixed(1) +
           '" stroke="var(--pebble-grid, #87867F)" stroke-opacity="0.28" stroke-width="0.75"/>';
    svg += '<text x="' + (yAxisWidth - 6) + '" y="' + (ty + 5).toFixed(1) +
           '" text-anchor="end" fill="var(--pebble-text-muted, #87867F)" font-size="16" font-weight="500" ' +
           'font-family="' + fontFamily + '">' + tick + '</text>';
  }

  bgSvg.innerHTML = svg;
  chartArea.appendChild(bgSvg);

  // ── bar container ──
  var barsDiv = document.createElement('div');
  barsDiv.style.display = 'flex';
  barsDiv.style.alignItems = 'flex-end';
  barsDiv.style.gap = barGap + 'px';
  barsDiv.style.marginLeft = yAxisWidth + 'px';
  barsDiv.style.paddingTop = topPad + 'px';
  barsDiv.style.paddingLeft = '6px';
  barsDiv.style.paddingRight = '6px';
  barsDiv.style.position = 'relative';
  barsDiv.style.zIndex = '1';
  barsDiv.style.overflow = 'visible';

  cats.forEach(function (cat) {
    renderPebbleBar(barsDiv, {
      barWidth: barWidth,
      hSqueeze: hSqueeze,
      outlineRadius: outlineRadius,
      gutter: gutter,
      logBase: logBase,
      itemOffset: itemOffset,
      items: cat.items,
      label: cat.label || cat.name,
      color: cat.color || '#888',
      watercolorColor: cat.watercolorColor || null,
      showCount: showCount,
      rotateLabel: true,
      labelHeight: labelHeight,
    });
  });

  chartArea.appendChild(barsDiv);
  outer.appendChild(chartArea);
  container.appendChild(outer);
  return outer;
}
