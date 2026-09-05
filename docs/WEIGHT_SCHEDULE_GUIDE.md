# Weight Schedule Guide

The weight schedule blends three team-strength signals:

1. baseline performance
2. testing directionality modifier
3. current-season performance

Implementation: `src/systems/weight_schedule.py`, used by `Baseline2026Predictor`.

## Why This Exists

In a regulation-change season, last year's championship standings are a weak
anchor, not a season template. A team that dominated on the previous rules might
have gotten the new concept wrong. A midfield team might have nailed it.

Pre-season testing gives you directional signals - who looks fast, who looks
fragile - but teams deliberately mask their true pace. Treating testing lap times
as ground truth is a mistake.

The weight schedule handles this by being explicit about which source of evidence
you trust at each point in the season. At Race 1, you blend all three signals
because you don't have much else. By Race 3, you're almost entirely running on
what teams have actually shown in competition. The trust shift is configurable
and auditable - not hidden inside a model.

For 2026 analysis, treat 2022 and 2014 as the closest regulation-reset analogs.
Use 2025 as a carryover sanity check only; it should not be the representative
season for promotion decisions.

## Active Schedule

The runtime config currently sets `baseline_predictor.team_strength_schedule` to
`rapid_adaptive`. That keeps early learning fast without making three completed
weekends act like the whole 2026 order is settled.

| Race | Baseline | Testing | Current |
|------|----------|---------|---------|
| 1    | 35%      | 20%     | 45%     |
| 2    | 20%      | 10%     | 70%     |
| 3    | 8%       | 5%      | 87%     |
| 4+   | 5%       | 0%      | 95%     |

## How Inputs Are Built

For a given team:

- `baseline`: `overall_performance` from car characteristics
- `testing_modifier`: **also `baseline`.** This slot used to carry
  `baseline + track_suitability`, but track suitability no longer feeds the blend -
  an out-of-sample check across 2022-2026 found the term negative in 9 of 10
  season x session cells. Because the slot now receives `baseline` too, its weight
  folds into the baseline weight. The column is kept in `SCHEDULES` rather than
  removed, so the shared table and its other schedules stay untouched.
- `current`: **recency-weighted and stabilized** mean of `current_season_performance`
  if race results exist, otherwise falls back to `baseline`. Race `i` is weighted
  `i ** recency_exponent` and the result is pulled toward `baseline` by
  `stabilization_strength`; it is not a plain mean.

The pre-season fallback matters: before any races, `current` is not zero. It inherits the baseline value so the blended output stays sensible even at the first race.

## Example

```python
from src.systems.weight_schedule import calculate_blended_performance

score = calculate_blended_performance(
    baseline_score=0.85,
    testing_modifier=0.85,  # the caller passes baseline here; see above
    current_score=0.85,   # pre-season: inherits baseline
    race_number=1,
    schedule="rapid_adaptive",
)
```

## Where Race Updates Feed In

`update_from_race` appends new values to `current_season_performance`, which shifts
the recency-weighted mean used as `current` in future predictions. The pre-season
baseline is not overwritten by in-season data.

## Related

- `src/systems/weight_schedule.py`
- `src/predictors/baseline_2026.py`
- `scripts/update_from_race.py`
