# Changelog

## v0.2.5

### Breaking Change
- `robust_floater()` now returns `np.nan` instead of the original string when given a non-numeric string. This ensures consistent numeric output and avoids mixed-type issues downstream.
