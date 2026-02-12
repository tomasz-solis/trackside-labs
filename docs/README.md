# Documentation Index

This folder explains how the current system behaves in code.

## Start Here

- `../README.md`: project-level quick start and runtime summary
- `../ARCHITECTURE.md`: component map and data flow
- `../CONFIGURATION.md`: active vs secondary config paths

## Detailed Guides

### `WEIGHT_SCHEDULE_GUIDE.md`

What it covers:

- baseline/testing/current signal blending,
- race-by-race weight progression,
- why the model shifts toward in-season data quickly.

### `FP_BLENDING_SYSTEM.md`

What it covers:

- where session blending is used (qualifying path),
- session priority rules for normal and sprint weekends,
- caveats and fallback behavior.

### `WEEKEND_PREDICTIONS.md`

What it covers:

- normal vs sprint cascade output in the dashboard,
- when ACTUAL grids are used vs PREDICTED grids,
- how predictions are chained between sessions.

### `DASHBOARD_AUTO_UPDATE.md`

What it covers:

- what updates automatically during dashboard use,
- what still requires explicit script execution,
- cache behavior.

### `PREDICTION_TRACKING.md`

What it covers:

- how session-based prediction files are stored,
- how to attach actual results,
- how accuracy metrics are computed.

## Validation Notebooks

- `../notebooks/validate_testing_predictions.ipynb`
- `../notebooks/test_weight_schedules.ipynb`

Use these as supporting analysis, not as a substitute for checking runtime code.

## Scope Note

If you see a mismatch between docs and code, code behavior is the source of truth. This documentation is intended to mirror the current runtime path and gets updated as logic changes.
