# Weight Schedule Guide

The weight schedule blends three team-strength signals:

1. baseline performance,
2. testing directionality modifier,
3. current-season performance.

Implementation is in `src/systems/weight_schedule.py` and used by `Baseline2026Predictor`.

## Why This Exists

In regulation-change seasons, old standings can become stale quickly. The schedule shifts trust toward current-season data early.

## Active Recommended Schedule

The baseline predictor uses `get_recommended_schedule(is_regulation_change=True)`, which maps to `extreme`.

Main checkpoints:

- Race 1: `30% baseline`, `20% testing`, `50% current`
- Race 2: `15% baseline`, `10% testing`, `75% current`
- Race 3+: `5% baseline`, `0% testing`, `95% current`

## How Inputs Are Built In Predictor

For a given team:

- `baseline`: `overall_performance`
- `testing_modifier`: track suitability from team directionality vs track profile
- `current`:
  - mean of `current_season_performance` if available,
  - otherwise falls back to `baseline` before races exist

That fallback behavior is important: pre-season `current` is not zero in the active predictor path.

## Example

```python
from src.systems.weight_schedule import calculate_blended_performance

score = calculate_blended_performance(
    baseline_score=0.85,
    testing_modifier=0.02,
    current_score=0.85,   # pre-season fallback in active predictor
    race_number=1,
    schedule="extreme",
)
```

## Where Race Updates Feed In

`update_from_race` appends new values to `current_season_performance`, which changes the running mean used as `current` in future predictions.

`overall_performance` and testing directionality remain separate inputs.

## Related Files

- `src/systems/weight_schedule.py`
- `src/predictors/baseline_2026.py`
- `scripts/update_from_race.py`
