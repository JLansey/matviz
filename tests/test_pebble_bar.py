"""Tests for the pebble bar chart module."""

import os
import tempfile
from pathlib import Path

from matviz.pebble_bar import pebble_bar_chart, _build_html


def _sample_categories():
    return [
        {"name": "a", "label": "Alpha", "color": "#4a90d9",
         "items": [{"id": str(i)} for i in range(20)]},
        {"name": "b", "label": "Beta", "color": "#d94a4a",
         "items": [{"id": str(i)} for i in range(10)]},
        {"name": "c", "label": "Gamma", "color": "#49b675",
         "items": [{"id": str(i)} for i in range(5)]},
    ]


def test_returns_html_string():
    html = pebble_bar_chart(_sample_categories(), title="Test")
    assert "<!DOCTYPE html>" in html
    assert "renderPebbleChart" in html
    assert "<title>Test</title>" in html


def test_writes_file(tmp_path):
    out = tmp_path / "test.html"
    html = pebble_bar_chart(_sample_categories(), out, title="File Test")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == html


def test_categories_in_output():
    cats = _sample_categories()
    html = pebble_bar_chart(cats)
    assert '"Alpha"' in html
    assert '"Beta"' in html
    assert '"Gamma"' in html


def test_watercolor_mode():
    cats = [
        {"name": "wc", "label": "Watercolor", "color": "#888",
         "watercolor_color": "#b87200",
         "items": [{"id": "1"}, {"id": "2"}]},
    ]
    html = pebble_bar_chart(cats)
    assert "watercolorColor" in html
    assert "#b87200" in html


def test_empty_categories():
    html = pebble_bar_chart([])
    assert "<!DOCTYPE html>" in html


def test_items_with_links():
    cats = [
        {"name": "linked", "label": "Linked", "color": "#888",
         "items": [
             {"id": "1", "link": "https://example.com/1", "label": "Item 1"},
             {"id": "2", "link": "https://example.com/2"},
         ]},
    ]
    html = pebble_bar_chart(cats)
    assert "https://example.com/1" in html
    assert "https://example.com/2" in html


def test_custom_options():
    html = pebble_bar_chart(
        _sample_categories(),
        bar_width=200,
        log_base=1.1,
        item_offset=10,
        h_squeeze=0.8,
        bar_gap=15,
        sort_desc=False,
        show_count=False,
    )
    assert '"barWidth": 200' in html
    assert '"logBase": 1.1' in html
    assert '"itemOffset": 10' in html


def test_subtitle():
    html = pebble_bar_chart(
        _sample_categories(),
        title="T",
        subtitle="Some description",
    )
    assert "Some description" in html
