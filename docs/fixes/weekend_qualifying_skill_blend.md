# Weekend qualifying skill blend

Accepted for live 2026 forecasts on 2026-05-22.

## Decision

After each finished weekend, live qualifying moves the driver's static `skill_score` toward the Bayesian form the updater already refreshes:

```text
bayesian_quali_skill_blend_per_race = 0.45
bayesian_quali_skill_blend_cap = 0.90
```

Race skill keeps its slower schedule. Driver priors and seconds-based fields are unchanged.

## Evidence

Sequential 2026 replay over Australia, China, Japan and Miami against the baseline:

| Candidate | All-target MSE | Race-target MSE |
|---|---:|---:|
| Baseline | 34.289 | 35.909 |
| Slow qualifying blend (0.20, cap 0.60) | 34.126 | 35.709 |
| Fast qualifying and race blend (0.45, cap 0.90) | 33.818 | 35.309 |
| **Fast qualifying only (0.45, cap 0.90)** | **33.770** | **35.227** |

Australia is unchanged (no earlier weekend). All-target MSE: China -1.45%, Japan -2.23%, Miami -2.97%. Speeding up race skill did not help, so it stayed.

Whole field: 34 checkpoint-target rows, 22 drivers each, 748 comparisons. MSE 34.289 to 33.770, MAE 4.476 to 4.444. Gains sum to about -21.0, losses to about +9.6. Not every driver improves on four weekends.

This was a single-seed replay on four weekends, measured before the seed floor existed.

## Monitoring

Rerun the replay after new weekends. Keep the blend while field MSE and MAE hold and driver-level losses stay small.
