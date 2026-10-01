"""
Pebble bar renderer — draws one bar of a pebble chart to a single image.

Uses only numpy and Pillow. Each bar is drawn on a supersampled canvas
(``scale * supersample`` pixels per CSS pixel) and shrunk with area averaging,
so pebbles smaller than a pixel at the top of a bar blend into dots instead of
vanishing. The watercolor wash is drawn into the same image with the SVG
``feTurbulence`` noise algorithm, so it matches the look of the original SVG
filter.

Positions come from :mod:`matviz.pebble_layout`; see :func:`render_bar`.
"""

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from .pebble_layout import Row, bar_height, bar_rows

# Watercolor geometry (CSS px): how far the wash bleeds past the bar, and the
# filter settings of the original SVG wash.
WASH_BLEED = {"x": 5.0, "top": 4.0, "bottom": 3.0}
WASH_OPACITY = 0.68
WASH_GRAIN = {"base_frequency": 0.55, "num_octaves": 2, "strength": 0.30}
WASH_WARP = {"base_frequency": 0.11, "num_octaves": 2, "scale": 2.6}


@dataclass
class PebbleBar:
    """A rendered bar.

    ``box`` is ``(x, y, width, height)`` of ``image`` in CSS px, relative to
    the top-left corner of the visible bar (``x`` and ``y`` are negative when
    the image extends past the bar for overhanging pebbles or the wash).
    """

    image: Image.Image
    box: Tuple[float, float, float, float]
    width: float
    height: float
    scale: int
    rows: List[Row] = field(default_factory=list)


# ── SVG feTurbulence (fractal Perlin noise, per the SVG 1.1 spec) ──────────

_B = 0x100
_RAND_M, _RAND_A, _RAND_Q, _RAND_R = 2147483647, 16807, 127773, 2836


def _turbulence_tables(seed):
    def rand(s):
        s = _RAND_A * (s % _RAND_Q) - _RAND_R * (s // _RAND_Q)
        return s + _RAND_M if s <= 0 else s

    s = int(round(seed))
    if s <= 0:
        s = -(s % (_RAND_M - 1)) + 1
    s = min(s, _RAND_M - 1)
    lattice = list(range(_B))
    grad = np.zeros((4, _B + _B + 2, 2))
    for k in range(4):
        for i in range(_B):
            for j in range(2):
                s = rand(s)
                grad[k, i, j] = ((s % (_B + _B)) - _B) / _B
            norm = math.hypot(*grad[k, i])
            if norm:
                grad[k, i] /= norm
    for i in range(_B - 1, 0, -1):
        s = rand(s)
        j = s % _B
        lattice[i], lattice[j] = lattice[j], lattice[i]
    lattice = np.array(lattice + lattice[:_B + 2])
    grad[:, _B:] = grad[:, :_B + 2]
    return lattice, grad


def _noise2(lattice, grad, channel, vx, vy):
    def split(v):
        t = v + 0x1000
        ti = np.floor(t).astype(np.int64)
        return ti & 0xFF, (ti + 1) & 0xFF, t - ti

    bx0, bx1, rx0 = split(vx)
    by0, by1, ry0 = split(vy)
    rx1, ry1 = rx0 - 1, ry0 - 1
    i, j = lattice[bx0], lattice[bx1]
    g = grad[channel]
    b00, b10 = g[lattice[i + by0]], g[lattice[j + by0]]
    b01, b11 = g[lattice[i + by1]], g[lattice[j + by1]]
    sx = rx0 * rx0 * (3 - 2 * rx0)
    sy = ry0 * ry0 * (3 - 2 * ry0)
    u, v = rx0 * b00[..., 0] + ry0 * b00[..., 1], rx1 * b10[..., 0] + ry0 * b10[..., 1]
    a = u + sx * (v - u)
    u, v = rx0 * b01[..., 0] + ry1 * b01[..., 1], rx1 * b11[..., 0] + ry1 * b11[..., 1]
    b = u + sx * (v - u)
    return a + sy * (b - a)


def turbulence(x, y, *, base_frequency, num_octaves, seed, channels=(0, 1, 2, 3)):
    """``feTurbulence type="fractalNoise"`` sampled at points ``x, y``.

    Returns an array of shape ``x.shape + (len(channels),)`` in 0–1.
    """
    lattice, grad = _turbulence_tables(seed)
    out = []
    for ch in channels:
        total = np.zeros(np.shape(x))
        vx, vy, ratio = x * base_frequency, y * base_frequency, 1.0
        for _ in range(num_octaves):
            total += _noise2(lattice, grad, ch, vx, vy) / ratio
            vx, vy, ratio = vx * 2, vy * 2, ratio * 2
        out.append(np.clip((total + 1) / 2, 0, 1))
    return np.stack(out, axis=-1)


def _hex_rgb(color):
    return np.array(ImageColor.getrgb(color)[:3], dtype=float) / 255


def name_seed(name, seed=0):
    """Stable per-name seed (same name and seed → same number)."""
    return (zlib.crc32(str(name).encode("utf-8")) ^ int(seed)) % 10000 + 1


def watercolor_layer(shape, origin, scale, width, height, color, *,
                     background="#faf8f2", seeds=(1, 2)):
    """Premultiplied RGBA float array (``shape`` = (h, w)) holding the wash.

    ``origin`` is the CSS position of pixel (0, 0) relative to the bar's
    top-left; ``width`` and ``height`` are the bar's CSS size. Inside the bar
    the wash sits on an opaque ``background`` rectangle; outside it the wash
    is semi-transparent and is exact over a page of the same background.
    """
    h, w = shape
    bx, bt, bb = WASH_BLEED["x"], WASH_BLEED["top"], WASH_BLEED["bottom"]
    # CSS coordinates of each pixel centre, in the wash's own (SVG) space.
    u = origin[0] + (np.arange(w) + 0.5) / scale + bx
    v = origin[1] + (np.arange(h) + 0.5) / scale + bt
    uu, vv = np.meshgrid(u, v)
    wash_w, wash_h = width + 2 * bx, height + bt + bb

    grain = turbulence(uu, vv, seed=seeds[0], channels=(0, 1, 2),
                       base_frequency=WASH_GRAIN["base_frequency"],
                       num_octaves=WASH_GRAIN["num_octaves"])
    lum = grain @ np.array([0.2125, 0.7154, 0.0721])
    alpha = np.clip(WASH_OPACITY - WASH_GRAIN["strength"] * lum, 0, 1)
    ink = np.minimum(WASH_OPACITY * _hex_rgb(color), alpha[..., None])
    ink = np.concatenate([ink, alpha[..., None]], axis=-1)
    ink[(uu < 0) | (uu >= wash_w) | (vv < 0) | (vv >= wash_h)] = 0

    warp = turbulence(uu, vv, seed=seeds[1], channels=(0, 1),
                      base_frequency=WASH_WARP["base_frequency"],
                      num_octaves=WASH_WARP["num_octaves"])
    shift = WASH_WARP["scale"] * (warp - 0.5) * scale
    sx = np.round(np.arange(w)[None, :] + shift[..., 0]).astype(int)
    sy = np.round(np.arange(h)[:, None] + shift[..., 1]).astype(int)
    inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    wash = np.zeros((h, w, 4))
    wash[inside] = ink[sy[inside], sx[inside]]

    bg = _hex_rgb(background)
    wash[..., :3] *= bg  # multiply blend onto the background
    in_bar = ((uu >= bx) & (uu < bx + width) & (vv >= bt) & (vv < bt + height))
    under = np.where(in_bar, 1.0, 0.0)[..., None] * np.append(bg, 1.0)
    return wash + under * (1 - wash[..., 3:4])


# ── pebbles ────────────────────────────────────────────────────────────────

def default_image_loader(image_dir=None):
    """Loader for items whose ``src`` is a local file (relative to ``image_dir``)
    or whose ``image`` is already a PIL image."""
    base = Path(image_dir) if image_dir else None

    def load(item):
        if isinstance(item.get("image"), Image.Image):
            return item["image"]
        path = Path(item["src"])
        if base is not None and not path.is_absolute():
            path = base / path
        if not path.exists():
            raise FileNotFoundError(
                f"pebble item {item.get('id', '?')!r}: image not found: {path}")
        with Image.open(path) as im:
            im.load()
            return im
    return load


def _corner_mask(px, radius_px):
    mask = Image.new("L", (px, px), 0)
    r = min(radius_px, px / 2)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, px - 1, px - 1), radius=r, fill=255)
    return np.asarray(mask, dtype=float) / 255


def _shifted(alpha, dx, dy):
    out = np.zeros_like(alpha)
    h, w = alpha.shape
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        alpha[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def _image_tile(img, px, outline_px):
    """Stretch ``img`` to ``px`` square, with a white outline following its alpha."""
    tile = img.convert("RGBA").resize((px, px), Image.LANCZOS, reducing_gap=3.0)
    arr = np.asarray(tile, dtype=float) / 255
    if outline_px > 0:
        alpha, clear = arr[..., 3], np.ones((px, px))
        for a in range(8):
            ang = a * math.pi / 4
            clear *= 1 - _shifted(alpha, round(math.cos(ang) * outline_px),
                                  round(math.sin(ang) * outline_px))
        halo = 1 - clear
        a_top = arr[..., 3:4]
        out_a = a_top + halo[..., None] * (1 - a_top)
        rgb = (arr[..., :3] * a_top + (1 - a_top) * halo[..., None]) / np.maximum(out_a, 1e-6)
        arr = np.concatenate([rgb, out_a], axis=-1)
    return arr


def _to_image(arr):
    return Image.fromarray(np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA")


def render_bar(items: Sequence[Dict[str, Any]], *, bar_width=120,
               log_base=math.sqrt(2), item_offset=0, gutter=0, h_squeeze=1.0,
               color="#888888", watercolor_color=None, background="#faf8f2",
               image_loader: Optional[Callable[[Dict[str, Any]], Image.Image]] = None,
               image_dir=None, outline_radius=1.0, corner_radius=None, scale=2,
               supersample=4, seed=0, wash_seeds=None) -> PebbleBar:
    """Draw one pebble bar to an RGBA image.

    Parameters
    ----------
    items : list of dict
        Bar items, bottom (largest) first. Items with ``src`` (a local file
        path, resolved against ``image_dir``) or ``image`` (a PIL image) are
        drawn as pictures. Without a picture, an item is a solid ``color``
        square when there is no watercolor and is left to the wash otherwise.
    bar_width, log_base, item_offset, gutter, h_squeeze
        Layout, as in :func:`matviz.pebble_layout.bar_rows`.
    color : str
        Fill for items without a picture (flat mode only).
    watercolor_color : str or None
        Colour of the watercolor wash behind the bar; falsy for no wash.
    background : str
        Page colour the wash is blended onto (the bar itself is opaque).
    image_loader : callable, optional
        ``image_loader(item) -> PIL.Image``. Defaults to
        :func:`default_image_loader` with ``image_dir``.
    outline_radius : float
        White outline around each picture's alpha, in CSS px (0 = none).
    corner_radius : float or callable, optional
        Corner radius in CSS px, or ``f(size)``. Default
        ``max(1, min(3, size * 0.05))``.
    scale : int
        Output pixels per CSS px (2 = sharp on most high-density screens).
    supersample : int
        Extra drawing resolution before shrinking to ``scale``.
    seed : int
        Seed for which pebble overlaps which; the same seed gives the same image.
    wash_seeds : (int, int), optional
        Noise seeds for the wash grain and edge warp. Default derived from ``seed``.

    Returns
    -------
    PebbleBar
    """
    n = len(items)
    layout = dict(bar_width=bar_width, log_base=log_base,
                  item_offset=item_offset, gutter=gutter)
    rows = bar_rows(n, h_squeeze=h_squeeze, **layout)
    height = bar_height(n, **layout)
    width = bar_width * h_squeeze

    rects = []
    for row in rows:
        for c in range(row.cols):
            i = row.first + c
            if 0 <= i < n:
                rects.append((i, row.x0 + c * row.step, row.y, row.size))
    over = max([0.0] + [max(-x, x + s - width) for _, x, _, s in rects])

    # Image extent around the bar, in whole output pixels.
    wash = bool(watercolor_color)
    warp_px = WASH_WARP["scale"] / 2 + 1
    pad_x = max(over, WASH_BLEED["x"] + warp_px if wash else 0)
    pad_top = WASH_BLEED["top"] + warp_px if wash else 0
    pad_bottom = WASH_BLEED["bottom"] + warp_px if wash else 0
    left_px = math.ceil(pad_x * scale)
    top_px = math.ceil(pad_top * scale)
    w_px = 2 * left_px + math.ceil(width * scale)
    h_px = top_px + math.ceil((height + pad_bottom) * scale)
    box = (-left_px / scale, -top_px / scale, w_px / scale, h_px / scale)

    k = scale * supersample
    canvas = Image.new("RGBA", (w_px * supersample, h_px * supersample), (0, 0, 0, 0))
    loader = image_loader or default_image_loader(image_dir)
    fill = np.append(_hex_rgb(color), 1.0)
    radius_of = corner_radius if callable(corner_radius) else (
        (lambda s: corner_radius) if corner_radius is not None
        else (lambda s: max(1.0, min(3.0, s * 0.05))))
    cache: Dict[Tuple[Any, int], Image.Image] = {}

    order = np.random.default_rng(seed).permutation(len(rects))
    for j in order:
        i, x, y, size = rects[j]
        item = items[i]
        px = max(1, round(size * k))
        if item.get("src") or isinstance(item.get("image"), Image.Image):
            key = (item.get("src") or id(item.get("image")), px)
            if key not in cache:
                arr = _image_tile(loader(item), px, round(outline_radius * k))
                arr[..., 3] *= _corner_mask(px, radius_of(size) * k)
                cache[key] = _to_image(arr)
            tile = cache[key]
        elif not wash:
            arr = np.broadcast_to(fill, (px, px, 4)).copy()
            arr[..., 3] *= _corner_mask(px, radius_of(size) * k)
            tile = _to_image(arr)
        else:
            continue
        dx = round((x - box[0]) * k)
        dy = round((y - box[1]) * k)
        if dx < 0 or dy < 0:
            tile = tile.crop((max(0, -dx), max(0, -dy), px, px))
            dx, dy = max(dx, 0), max(dy, 0)
        canvas.alpha_composite(tile, dest=(dx, dy))

    # Area-average down to the output scale (Pillow premultiplies RGBA here).
    pebbles = canvas.resize((w_px, h_px), Image.BOX)
    if not wash:
        return PebbleBar(pebbles, box, width, height, scale, rows)

    top = np.asarray(pebbles, dtype=float) / 255
    top[..., :3] *= top[..., 3:4]
    seeds = wash_seeds or (name_seed("grain", seed), name_seed("warp", seed))
    under = watercolor_layer((h_px, w_px), box[:2], scale, width, height,
                             watercolor_color, background=background, seeds=seeds)
    out = top + under * (1 - top[..., 3:4])
    a = out[..., 3:4]
    out[..., :3] = np.where(a > 0, out[..., :3] / np.maximum(a, 1e-6), 0)
    return PebbleBar(_to_image(out), box, width, height, scale, rows)
