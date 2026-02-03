# Changelog

## [v0.2.5] - 2026-02-03

### Changed
- `robust_floater()` now returns `np.nan` instead of the original string when given a non-numeric string. Previously, passing `"hello"` would return `"hello"`, which could cause mixed-type issues in pandas/numpy operations. Now it returns `nan` for consistent numeric output.
