# Changelog

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
