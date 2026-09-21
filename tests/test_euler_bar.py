"""Tests for the Euler bar chart module."""

import tempfile
from pathlib import Path

from matviz.euler_bar import euler_bar_chart, _build_html


def _sample_categories():
    return [
        {"name": "a", "label": "Alpha", "set_a": 20, "set_b": 50, "overlap": 10},
        {"name": "b", "label": "Beta", "set_a": 15, "set_b": 30, "overlap": 5},
        {"name": "c", "label": "Gamma", "set_a": 8, "set_b": 12, "overlap": 3},
    ]


def test_returns_html_string():
    html = euler_bar_chart(_sample_categories(), title="Test")
    assert "<!DOCTYPE html>" in html
    assert "<title>Test</title>" in html
    assert "euler-chart" in html


def test_writes_file(tmp_path):
    out = tmp_path / "test.html"
    html = euler_bar_chart(_sample_categories(), out, title="File Test")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == html


def test_categories_in_output():
    cats = _sample_categories()
    html = euler_bar_chart(cats)
    assert '"Alpha"' in html
    assert '"Beta"' in html
    assert '"Gamma"' in html


def test_empty_categories():
    html = euler_bar_chart([])
    assert "<!DOCTYPE html>" in html


def test_single_category():
    cats = [{"name": "solo", "label": "Solo", "set_a": 10, "set_b": 20, "overlap": 5}]
    html = euler_bar_chart(cats)
    assert '"Solo"' in html
    # Union = 10 + 20 - 5 = 25
    assert "25" in html


def test_sort_order():
    cats = [
        {"name": "small", "label": "Small", "set_a": 2, "set_b": 3, "overlap": 1},
        {"name": "big", "label": "Big", "set_a": 50, "set_b": 100, "overlap": 10},
    ]
    # Default sort_desc=True: Big (union=140) should come before Small (union=4)
    html = euler_bar_chart(cats, sort_desc=True)
    pos_big = html.find('"Big"')
    pos_small = html.find('"Small"')
    assert pos_big < pos_small

    # sort_desc=False: original order preserved
    html_unsorted = euler_bar_chart(cats, sort_desc=False)
    pos_big_u = html_unsorted.find('"Big"')
    pos_small_u = html_unsorted.find('"Small"')
    assert pos_small_u < pos_big_u


def test_no_overlap():
    cats = [{"name": "x", "label": "No Overlap", "set_a": 10, "set_b": 20, "overlap": 0}]
    html = euler_bar_chart(cats)
    # Union = 10 + 20 - 0 = 30
    assert "30" in html
    assert '"No Overlap"' in html


def test_complete_overlap():
    cats = [{"name": "x", "label": "Full", "set_a": 5, "set_b": 20, "overlap": 5}]
    html = euler_bar_chart(cats)
    # Union = 5 + 20 - 5 = 20
    assert "20" in html
    assert '"Full"' in html


def test_show_table_false():
    html = euler_bar_chart(_sample_categories(), show_table=False)
    assert "euler-table" not in html or "showTable" in html
    # The JS checks opts.showTable, so when false the table DOM shouldn't render.
    # We verify the option is set to false in the JSON config.
    assert '"showTable": false' in html


def test_custom_colors():
    html = euler_bar_chart(
        _sample_categories(),
        color_a="#FF0000",
        color_b="#00FF00",
    )
    assert "#FF0000" in html
    assert "#00FF00" in html
