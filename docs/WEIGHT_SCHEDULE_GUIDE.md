# Weight schedule

Team strength blends three signals: the baseline, a testing modifier and current-season results. Code: `src/systems/weight_schedule.py`, used by `Baseline2026Predictor`.

## Why

After a rules change, last year's order is a weak anchor. Testing shows direction, but teams hide pace. The schedule states how much each source is trusted at each race, so the shift is visible and configurable. At race 1 all three count. By race 3 the model runs almost entirely on results.

For 2026, the closest analogs are the 2022 and 2014 resets. Use 2025 only as a sanity check.

## Active schedule

`baseline_predictor.team_strength_schedule: rapid_adaptive`:

| Race | Baseline | Testing | Current |
|------|----------|---------|---------|
| 1    | 35%      | 20%     | 45%     |
| 2    | 20%      | 10%     | 70%     |
| 3    | 8%       | 5%      | 87%     |
| 4+   | 5%       | 0%      | 95%     |

## Inputs

- `baseline`: `overall_performance` from car characteristics.
- `testing_modifier`: currently also the baseline. Track suitability used to be added here, but it lost out of sample in 9 of 10 season and session cells (2022 to 2026), so it was removed. Its weight now just adds to the baseline weight.
- `current`: a recency-weighted mean of `current_season_performance`, pulled toward the baseline by `stabilization_strength`. Race `i` gets weight `i ** recency_exponent`. Before any race it equals the baseline, never zero.

`update_from_race` appends to `current_season_performance`. It never overwrites the baseline.

## Example

```python
from src.systems.weight_schedule import calculate_blended_performance

score = calculate_blended_performance(
    baseline_score=0.85,
    testing_modifier=0.85,  # the caller passes the baseline here
    current_score=0.85,     # before any race: equals the baseline
    race_number=1,
    schedule="rapid_adaptive",
)
```
