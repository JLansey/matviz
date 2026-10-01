"""Tests for the pebble bar renderer."""

import math

import numpy as np
import pytest
from PIL import Image

from matviz.pebble_layout import item_rects
from matviz.pebble_render import render_bar, turbulence

LAYOUT = dict(bar_width=150, log_base=1.1, item_offset=16, h_squeeze=0.7)


def _items(n):
    colors = [(200, 40, 40), (40, 160, 60), (30, 60, 200)]
    out = []
    for i in range(n):
        im = Image.new("RGBA", (20, 20), colors[i % 3] + (255,))
        out.append({"id": str(i), "image": im})
    return out


def test_output_size_and_box():
    bar = render_bar(_items(200), scale=2, supersample=2, **LAYOUT)
    x, y, w, h = bar.box
    assert bar.image.size == (round(w * 2), round(h * 2))
    assert y == 0 and x < 0  # no wash: image starts at the bar top
    assert -x == pytest.approx(math.ceil(7.0 * 2) / 2, abs=1)  # overhang
    assert h == math.ceil(bar.height * 2) / 2
    assert w == pytest.approx(2 * -x + math.ceil(105 * 2) / 2)


def test_every_item_centre_is_drawn():
    n = 3269  # top rows are about 0.5 CSS px
    bar = render_bar(_items(n), scale=2, supersample=4, outline_radius=0, **LAYOUT)
    alpha = np.asarray(bar.image)[..., 3]
    x0, y0 = bar.box[:2]
    tiny = 0
    for x, y, s in item_rects(n, **LAYOUT):
        cx = int((x + s / 2 - x0) * 2)
        cy = min(int((y + s / 2 - y0) * 2), alpha.shape[0] - 1)
        assert alpha[cy, cx] > 0, (x, y, s)
        tiny += s < 1
    assert tiny > 500


def test_same_seed_same_bytes():
    a = render_bar(_items(80), seed=3, watercolor_color="#b87200", **LAYOUT)
    b = render_bar(_items(80), seed=3, watercolor_color="#b87200", **LAYOUT)
    c = render_bar(_items(80), seed=4, watercolor_color="#b87200", **LAYOUT)
    assert a.image.tobytes() == b.image.tobytes()
    assert a.image.tobytes() != c.image.tobytes()


def test_watercolor_bleeds_past_bar():
    bar = render_bar([{"id": "1"}] * 30, watercolor_color="#186b4e", **LAYOUT)
    x, y, w, h = bar.box
    assert y < 0 and h > bar.height
    alpha = np.asarray(bar.image)[..., 3]
    # opaque inside the bar, partly transparent in the bleed
    assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255
    left_bleed = alpha[alpha.shape[0] // 2, :int(-x * 2)]
    assert 0 < left_bleed.max() < 255


def test_flat_mode_fills_items_with_color():
    bar = render_bar([{"id": str(i)} for i in range(20)], color="#ff0000",
                     outline_radius=0, **LAYOUT)
    px = np.asarray(bar.image)
    x, y, s = item_rects(20, **LAYOUT)[0]
    cx, cy = int((x + s / 2 - bar.box[0]) * 2), int((y + s / 2) * 2)
    assert tuple(px[cy, cx]) == (255, 0, 0, 255)


def test_missing_image_names_item(tmp_path):
    with pytest.raises(FileNotFoundError, match="'abc'"):
        render_bar([{"id": "abc", "src": "nope.png"}], image_dir=tmp_path)


def test_turbulence_range_and_determinism():
    x, y = np.meshgrid(np.arange(30) + 0.5, np.arange(20) + 0.5)
    t = turbulence(x, y, base_frequency=0.1, num_octaves=2, seed=7)
    assert t.shape == (20, 30, 4)
    assert 0 <= t.min() and t.max() <= 1 and t.std() > 0.02
    assert np.array_equal(t, turbulence(x, y, base_frequency=0.1, num_octaves=2, seed=7))
