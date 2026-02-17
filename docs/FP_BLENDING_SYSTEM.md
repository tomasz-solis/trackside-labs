# Practice Session Blending

This file describes the current blending behavior implemented in `src/utils/fp_blending.py` and consumed by `Baseline2026Predictor.predict_qualifying()`.

## What It Does

Qualifying prediction combines:

- model team strength (from weight schedule), and
- team pace extracted from one selected session.

Active formula:

```text
blended_strength = 0.7 * session_strength + 0.3 * model_strength
```

In the active baseline path, this 70/30 split is fixed in predictor logic.

## Session Selection Priority

### Normal weekend

`FP3 > FP2 > FP1`

### Sprint weekend

`Sprint Qualifying > Sprint > FP1`

The first session that returns usable data is used.

## Important Behavior Detail

The current implementation uses **one best session**.
It does **not** merge multiple sessions together (for example FP1+FP2+FP3).

## How Session Strength Is Built

For each selected session:

1. Load laps.
2. Take each driver's best valid lap.
3. Compute median best lap per team.
4. Scale teams to a 0-1 performance band (fastest = 1.0).

## Fallbacks

- If no session data is available, qualifying uses model-only strength.
- If a team is missing from session data, that team keeps model-only strength.

## Where It Is Used

- `src/predictors/baseline_2026.py` (`predict_qualifying`)
- Dashboard prediction flow in `src/dashboard/prediction_flow.py` (called from `src/dashboard/pages.py`)

## Where It Is Not Used Directly

- Race scoring does not apply the FP blending function directly.
- Race model runs from grid + race simulation features.

## Practical Notes

- Blend source is shown in UI (`data_source` and `blend_used`).
- Session naming follows FastF1 conventions.
- Behavior depends on data availability and cache state.
