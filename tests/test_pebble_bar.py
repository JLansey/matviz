"""Tests for the pebble bar chart outputs (HTML, assets, matplotlib)."""

import json
import re

import matplotlib
import pytest
from PIL import Image

from matviz.pebble_bar import (pebble_bar_chart, pebble_bar_figure,
                               render_pebble_chart, write_pebble_assets)
from matviz.pebble_layout import bar_rows

matplotlib.use("Agg")


def _sample_categories():
    return [
        {"name": "a", "label": "Alpha", "color": "#4a90d9",
         "items": [{"id": str(i)} for i in range(20)]},
        {"name": "b", "label": "Beta", "color": "#d94a4a",
         "items": [{"id": str(i)} for i in range(10)]},
        {"name": "c", "label": "Gamma", "color": "#49b675",
         "items": [{"id": str(i)} for i in range(5)]},
    ]


def _manifest(html):
    m = re.search(r"var data = (\{.*?\});\n", html, re.S)
    assert m
    return json.loads(m.group(1))


def test_returns_html_string():
    html = pebble_bar_chart(_sample_categories(), title="Test")
    assert "<!DOCTYPE html>" in html
    assert "renderPebbleView" in html
    assert "<title>Test</title>" in html


def test_writes_file(tmp_path):
    out = tmp_path / "test.html"
    html = pebble_bar_chart(_sample_categories(), out, title="File Test")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == html


def test_data_block_one_image_and_rows_per_category():
    cats = _sample_categories()
    data = _manifest(pebble_bar_chart(cats, log_base=1.1, item_offset=16, h_squeeze=0.7))
    assert [c["label"] for c in data["categories"]] == ["Alpha", "Beta", "Gamma"]
    for cat, src in zip(data["categories"], cats):
        assert cat["image"].startswith("data:image/webp;base64,")
        assert cat["count"] == len(src["items"])
        rows = bar_rows(cat["count"], bar_width=120, log_base=1.1, item_offset=16, h_squeeze=0.7)
        assert len(cat["rows"]) == len(rows)
        # top-down for the browser's binary search
        assert [r[0] for r in cat["rows"]] == sorted(r[0] for r in cat["rows"])


def test_separate_image_files(tmp_path):
    out = tmp_path / "chart.html"
    data = _manifest(pebble_bar_chart(_sample_categories(), out, embed_images=False))
    assert [c["image"] for c in data["categories"]] == [
        "chart_files/a.webp", "chart_files/b.webp", "chart_files/c.webp"]
    assert Image.open(tmp_path / "chart_files" / "a.webp").mode == "RGBA"
    with pytest.raises(ValueError):
        pebble_bar_chart(_sample_categories(), embed_images=False)


def test_items_with_links_and_labels():
    cats = [
        {"name": "linked", "label": "Linked", "color": "#888",
         "items": [
             {"id": "1", "link": "https://example.com/1", "label": "Item 1"},
             {"id": "2", "link": "https://example.com/2", "month": "2026-05"},
         ]},
    ]
    items = _manifest(pebble_bar_chart(cats))["categories"][0]["items"]
    assert items == [
        {"id": "1", "link": "https://example.com/1", "label": "Item 1"},
        {"id": "2", "link": "https://example.com/2"},
    ]


def test_empty_categories():
    html = pebble_bar_chart([])
    assert "<!DOCTYPE html>" in html


def test_title_is_escaped():
    html = pebble_bar_chart(_sample_categories(), title="A <b>", subtitle="Some description")
    assert "A &lt;b&gt;" in html
    assert "Some description" in html


def test_write_assets(tmp_path):
    manifest = write_pebble_assets(_sample_categories(), tmp_path, bar_gap=21, seed=5)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert manifest["options"]["barGap"] == 21
    for cat in manifest["categories"]:
        img = Image.open(tmp_path / cat["image"])
        x, y, w, h = cat["box"]
        assert img.size == (round(w * 2), round(h * 2))
    # rebuilding gives identical images
    first = (tmp_path / "a.webp").read_bytes()
    write_pebble_assets(_sample_categories(), tmp_path, bar_gap=21, seed=5)
    assert (tmp_path / "a.webp").read_bytes() == first


def test_ticks_follow_bar_heights():
    chart = render_pebble_chart([{"name": "x", "items": [{}] * 600}],
                                log_base=1.1, item_offset=16)
    assert [v for v, _ in chart.ticks] == [5, 10, 20, 50, 100, 200, 500]
    heights = [h for _, h in chart.ticks]
    assert heights == sorted(heights) and heights[-1] < chart.max_height


def test_figure():
    fig = pebble_bar_figure(_sample_categories(), log_base=1.1, item_offset=16, h_squeeze=0.7)
    ax = fig.axes[0]
    assert len(ax.images) == 3
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Alpha\n20", "Beta\n10", "Gamma\n5"]
