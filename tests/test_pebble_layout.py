"""Tests for the pebble bar layout (positions only, no drawing)."""

import json
import math
from pathlib import Path

import pytest

from matviz.pebble_layout import (bar_height, bar_rows, item_rects, log_ticks,
                                  overhang)

# Recorded from the original browser renderer (pebble-bar.js) with the
# options in the file: every item's left/top/size and each visible height.
REF = json.loads((Path(__file__).parent / "data" / "pebble_js_layout.json").read_text())
FLEET = dict(REF["options"])
LAYOUT = {k: v for k, v in FLEET.items() if k != "h_squeeze"}


@pytest.mark.parametrize("n", sorted(REF["bars"], key=int))
def test_height_matches_js(n):
    assert bar_height(int(n), **LAYOUT) == pytest.approx(REF["bars"][n]["height"], abs=0.01)


@pytest.mark.parametrize("n", sorted(REF["bars"], key=int))
def test_item_rects_match_js(n):
    rects = item_rects(int(n), **FLEET)
    expected = REF["bars"][n]["rects"]
    assert len(rects) == len(expected) == int(n)
    for got, want in zip(rects, expected):
        assert got == pytest.approx(want, abs=0.01)


def test_fleet_heights_rounded_up():
    counts = {"unclear": 3269, "box_truck": 459, "fedex": 60, "prime": 33,
              "usps": 31, "bus": 23, "police": 17, "ups": 16, "wb_mason": 13,
              "motorcycle": 13}
    want = {"unclear": 521, "box_truck": 496, "fedex": 359, "prime": 281,
            "usps": 281, "bus": 256, "police": 200, "ups": 200, "wb_mason": 167,
            "motorcycle": 167}
    got = {k: math.ceil(bar_height(n, **LAYOUT)) for k, n in counts.items()}
    assert got == want


def test_rows_cover_every_item_once():
    rows = bar_rows(459, **FLEET)
    idx = [r.first + c for r in rows for c in range(r.cols) if 0 <= r.first + c < 459]
    assert idx == list(range(459))
    # rows are bottom-up and touch (no gutter)
    for lower, upper in zip(rows, rows[1:]):
        assert upper.y + upper.size == pytest.approx(lower.y)
    assert rows[-1].y == pytest.approx(0)


def test_overhang_fleet():
    assert overhang(3269, **FLEET) == pytest.approx(7.17, abs=0.5)
    assert overhang(5, bar_width=100) == 0


def test_empty_bar():
    assert bar_rows(0) == []
    assert bar_height(0) == 0
    assert item_rects(0) == []


def test_log_ticks():
    assert log_ticks(3269) == [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    assert log_ticks(13) == [5, 10]
    assert log_ticks(1) == []
