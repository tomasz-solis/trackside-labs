# Weekend Prediction Flow

This guide describes the current dashboard behavior in `src/dashboard/pages.py` (entrypoint: `app.py`).

## Overview

The app produces a cascade of predictions based on weekend format.

- Normal weekend: 2 outputs
- Sprint weekend: 4 outputs

The race model can use ACTUAL grids from completed competitive sessions when available.

## Normal Weekend

Flow:

1. Qualifying prediction
2. Race prediction

Process:

- Predict qualifying grid.
- If qualifying session is already complete and results are available, replace predicted grid with ACTUAL qualifying results.
- Predict race from that grid.

## Sprint Weekend

Flow:

1. Sprint Qualifying prediction
2. Sprint Race prediction
3. Main Qualifying prediction
4. Main Race prediction

Process:

- Predict Sprint Qualifying.
- Sprint Race uses Sprint Qualifying grid (ACTUAL if available, otherwise predicted).
- Predict Main Qualifying.
- Main Race uses Main Qualifying grid (ACTUAL if available, otherwise predicted).

## ACTUAL vs PREDICTED Grid Source

Grid source is resolved in `fetch_grid_if_available()` in `src/dashboard/prediction_flow.py`.

- `ACTUAL`: completed competitive session results were fetched.
- `PREDICTED`: no completed results available (or fetch failed), so model output is used.

Competitive sessions checked for grid replacement:

- `SQ` (Sprint Qualifying)
- `Q` (Main Qualifying)

## Practice Data Use In This Flow

Practice/session blending is used in qualifying prediction through `Baseline2026Predictor.predict_qualifying()`.

Important detail:

- The predictor uses the **best single available session** by priority.
- It does **not** average FP1+FP2+FP3 together in the active implementation.

Priority from `src/utils/fp_blending.py`:

- Normal weekend: `FP3 > FP2 > FP1`
- Sprint weekend: `Sprint > Sprint Qualifying > FP1`

## Sprint Race Adjustments

Sprint race prediction calls `predict_race(..., is_sprint=True)`.

Current sprint adjustments in baseline predictor:

- lower chaos level,
- extra grid weight influence.

## Prediction Tracking Integration

When tracking is enabled in the **⚙️ Settings** expander:

- one prediction file is saved per detected completed session,
- max one file per race/session key,
- sprint weekends save the main qualifying + main race outputs for scoring.

See `docs/PREDICTION_TRACKING.md` for file structure and update workflow.

## Known Limits

1. ACTUAL grid replacement depends on FastF1 data availability.
2. If session metadata is late or missing, the app falls back to PREDICTED grids.
3. Weekend format detection depends on FastF1 event schedule (with local fallback in utilities).
