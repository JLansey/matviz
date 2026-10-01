"""
Pebble bar layout — where every item of a pebble bar goes (no drawing).

A bar is a stack of rows. Row ``r`` (0 = bottom) holds ``floor(log_base**r)``
items, each a square ``bar_width / log_base**r`` wide, spread with
space-between justification. The first ``item_offset`` slots are invisible
padding; rows made only of padding are cropped from the bottom of the bar.

All coordinates are CSS pixels. ``x`` is measured from the left edge of the
squeezed bar (``bar_width * h_squeeze`` wide) and ``y`` from the top of the
visible bar, so they can be used directly by renderers and hit tests.
"""

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Row:
    """One row of a bar, top-left based, already horizontally squeezed.

    The item in column ``c`` sits at ``x = x0 + c * step``; its index in the
    item list is ``first + c`` (negative indices are padding).
    """

    y: float
    size: float
    cols: int
    first: int
    x0: float
    step: float


def _raw_rows(n_slots, bar_width, log_base):
    rows, remaining, r = [], n_slots, 0
    while remaining > 0:
        ideal = log_base ** r
        cols = min(math.floor(ideal), remaining)
        rows.append((cols, bar_width / ideal))
        remaining -= cols
        r += 1
    return rows


def bar_rows(n_items, *, bar_width=120, log_base=math.sqrt(2), item_offset=0,
             gutter=0, h_squeeze=1.0) -> List[Row]:
    """Rows holding at least one real item, from the bottom up."""
    if n_items <= 0:
        return []
    raw = _raw_rows(item_offset + n_items, bar_width, log_base)
    full_height = sum(size + gutter for _, size in raw) - gutter
    inset = bar_width * (1 - h_squeeze) / 2
    rows, slot, y_from_bottom = [], 0, 0.0
    for cols, size in raw:
        y = full_height - y_from_bottom - size
        if slot + cols > item_offset:
            if cols == 1:
                base_x0, base_step = (bar_width - size) / 2, 0.0
            else:
                base_x0, base_step = 0.0, (bar_width - size) / (cols - 1)
            centre = bar_width / 2 + (base_x0 + size / 2 - bar_width / 2) * h_squeeze
            rows.append(Row(y=y, size=size, cols=cols, first=slot - item_offset,
                            x0=centre - size / 2 - inset, step=base_step * h_squeeze))
        slot += cols
        y_from_bottom += size + gutter
    return rows


def bar_height(n_items, *, bar_width=120, log_base=math.sqrt(2), item_offset=0,
               gutter=0) -> float:
    """Visible height of a bar: full height minus the cropped padding rows."""
    if n_items <= 0:
        return 0.0
    raw = _raw_rows(item_offset + n_items, bar_width, log_base)
    full_height = sum(size + gutter for _, size in raw) - gutter
    crop, pad = 0.0, item_offset
    for cols, size in raw:
        if pad < cols:
            break
        crop += size + gutter
        pad -= cols
    return max(0.0, full_height - crop)


def item_rects(n_items, **layout):
    """``(x, y, size)`` of every real item, in item order."""
    rects = []
    for row in bar_rows(n_items, **layout):
        for c in range(row.cols):
            if 0 <= row.first + c < n_items:
                rects.append((row.x0 + c * row.step, row.y, row.size))
    return rects


def overhang(n_items, *, bar_width=120, h_squeeze=1.0, **layout) -> float:
    """Largest distance an edge item extends past either side of the bar."""
    rects = item_rects(n_items, bar_width=bar_width, h_squeeze=h_squeeze, **layout)
    if not rects:
        return 0.0
    width = bar_width * h_squeeze
    return max(0.0, -min(x for x, _, _ in rects),
               max(x + s for x, _, s in rects) - width)


def log_ticks(max_count, min_count=3):
    """1-2-5 tick values from ``min_count`` up to about ``max_count``."""
    ticks, mag = [], 1
    while True:
        for m in (1, 2, 5):
            v = m * mag
            if v > max_count * 1.1:
                return ticks
            if v >= min_count:
                ticks.append(v)
        mag *= 10
