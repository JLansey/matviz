# Changelog

## [Unreleased]

### Changed
- **Breaking:** pebble bar charts are now drawn in Python (numpy + Pillow)
  instead of in the browser. Each bar is rendered ahead of time to one image,
  including the watercolor wash, outlines and rounded corners, and the page
  only maps the pointer to the item under it. Thousands of items now load as
  one image per bar, and sub-pixel pebbles at the top of big bars show up.
- **Breaking:** item `src` must be a local image file (relative to the new
  `image_dir` option), or pass `image` (a PIL image) or `image_loader`.
- The browser renderer `_assets/pebble-bar.js` is removed. Its replacement,
  `_assets/pebble-view.js`, only displays pre-rendered bars.
- `outline_radius` now defaults to 1.

### Added
- `pebble_bar_figure()` draws the same chart as a static matplotlib figure.
- `write_pebble_assets()` writes one image per bar plus `manifest.json` for
  websites that use `pebble-view.js`.
- `pebble_bar_chart()` options `embed_images` (single file by default),
  `scale`, `supersample`, `seed` and `image_dir`.
- `matviz.pebble_layout` (row layout, heights, ticks) and
  `matviz.pebble_render` (`render_bar`) modules.

## [v0.2.7] - 2026-02-03

### Added
- `re` module now included in standard imports via `helpers`.

## [v0.2.6] - 2026-02-03

### Changed
- **Breaking:** `nhist` now returns the figure object instead of `(ax, N, bins)`. Data is accessible via `fig.nhist` dict containing `N`, `bins`, and `rawN`.
- **Breaking:** `ndhist` now returns the figure object instead of `(counts, bins_x, bins_y)`. Data is accessible via `fig.ndhist` dict.
- Renamed `helpers_graphing` module to `helpers` (backward compatible shim in place).

### Added
- `drop_mostly_na()` function for filtering sparse columns/rows from DataFrames.

## [v0.2.5] - 2026-02-03

### Changed
- `robust_floater()` now returns `np.nan` instead of the original string when given a non-numeric string. Previously, passing `"hello"` would return `"hello"`, which could cause mixed-type issues in pandas/numpy operations. Now it returns `nan` for consistent numeric output.
