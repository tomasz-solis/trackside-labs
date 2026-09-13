# Team-Strength Margin Policy: Fold Comparison

Built at: `2026-09-07T14:17:08.458552+00:00`
Policies compared: `same_session_construct` (shipped, rank-based) vs `same_session_margin` (new, `field_median_s - team_median_s` seconds).
Harnesses used: `evaluate_within_season_folds` (2026 leave-one-round-out, the adopt/reject evidence) and `evaluate_policy_folds` (2022-2025 leave-one-season-out, context only).

## 2026 within-season leave-one-round-out (adopt/reject evidence)

| Session kind | Policy | Folds | Rows | Prediction slope | R2 | RMSE (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qualifying` | `same_session_construct` | 11 | 181 | 0.923 | 0.687 | 0.566 |
| `qualifying` | `same_session_margin` | 11 | 181 | 0.917 | 0.868 | 0.370 |
| `race` | `same_session_construct` | 11 | 193 | 0.969 | 0.740 | 0.677 |
| `race` | `same_session_margin` | 11 | 193 | 0.974 | 0.896 | 0.418 |

Row-weighted across the 11 leave-one-round-out folds per session kind (one fold per 2026 race with a construct-aligned session; 181 qualifying rows, 193 race rows total).

### Per-fold detail

| Session kind | Holdout race | Rows | Construct R2 | Construct RMSE | Margin R2 | Margin RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qualifying` | Australian Grand Prix | 13 | 0.575 | 0.537 | 0.711 | 0.443 |
| `qualifying` | Austrian Grand Prix | 19 | 0.788 | 0.387 | 0.845 | 0.331 |
| `qualifying` | Barcelona Grand Prix | 14 | 0.772 | 0.523 | 0.911 | 0.326 |
| `qualifying` | Belgian Grand Prix | 13 | 0.750 | 0.603 | 0.926 | 0.327 |
| `qualifying` | British Grand Prix | 13 | 0.280 | 0.637 | 0.747 | 0.377 |
| `qualifying` | Canadian Grand Prix | 19 | 0.840 | 0.444 | 0.880 | 0.385 |
| `qualifying` | Chinese Grand Prix | 19 | 0.775 | 0.510 | 0.911 | 0.321 |
| `qualifying` | Hungarian Grand Prix | 17 | 0.721 | 0.706 | 0.914 | 0.392 |
| `qualifying` | Japanese Grand Prix | 19 | 0.783 | 0.553 | 0.909 | 0.357 |
| `qualifying` | Miami Grand Prix | 14 | 0.286 | 0.599 | 0.852 | 0.272 |
| `qualifying` | Monaco Grand Prix | 21 | 0.752 | 0.676 | 0.880 | 0.470 |
| `race` | Australian Grand Prix | 13 | 0.809 | 0.727 | 0.953 | 0.360 |
| `race` | Austrian Grand Prix | 17 | 0.751 | 0.568 | 0.895 | 0.369 |
| `race` | Barcelona Grand Prix | 19 | 0.745 | 0.642 | 0.811 | 0.552 |
| `race` | Belgian Grand Prix | 17 | 0.732 | 0.846 | 0.950 | 0.364 |
| `race` | British Grand Prix | 21 | 0.899 | 0.478 | 0.942 | 0.362 |
| `race` | Canadian Grand Prix | 18 | 0.749 | 0.750 | 0.940 | 0.366 |
| `race` | Chinese Grand Prix | 13 | 0.717 | 0.625 | 0.902 | 0.368 |
| `race` | Hungarian Grand Prix | 21 | 0.810 | 0.594 | 0.880 | 0.472 |
| `race` | Japanese Grand Prix | 21 | 0.853 | 0.604 | 0.937 | 0.394 |
| `race` | Miami Grand Prix | 14 | 0.833 | 0.648 | 0.960 | 0.318 |
| `race` | Monaco Grand Prix | 19 | 0.242 | 0.890 | 0.726 | 0.535 |

## 2022-2025 leave-one-season-out (context only, inadmissible for adopt/reject)

2026 changed the technical regulations, so a slope or R2 fitted and measured entirely inside 2022-2025 describes a car field this mapping is not calibrated against. Reported for provenance only.

| Session kind | Policy | Folds | Rows | Prediction slope | R2 | RMSE (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qualifying` | `same_session_construct` | 4 | 1286 | 0.935 | 0.406 | 0.916 |
| `qualifying` | `same_session_margin` | 4 | 1286 | 0.906 | 0.837 | 0.360 |
| `race` | `same_session_construct` | 4 | 1604 | 0.908 | 0.425 | 0.891 |
| `race` | `same_session_margin` | 4 | 1604 | 0.937 | 0.838 | 0.440 |

## What this shows

On the only admissible 2026 evidence -- 11 leave-one-round-out within-season folds per session kind (181 qualifying rows, 193 race rows) -- the margin policy beats the shipped rank policy out-of-sample in both constructs: R2 rises from 0.687 to 0.868 in qualifying (+0.181) and 0.740 to 0.896 in race (+0.157), while RMSE drops 0.196s in qualifying and 0.258s in race; prediction slope stays close to 1.0 for both policies, so neither is badly mis-scaled. This direction and rough magnitude matches the earlier leave-one-driver-out measurement that motivated this change (+0.157 r2 qualifying, +0.122 r2 race), even though the fold construction differs (leave-one-race-out here vs leave-one-driver-out there). The fold count is thin: 11 races is one 2026 season, each fold trains on the other ~10 races and tests on 13-21 rows, so a single unusual race can move an individual fold's R2 by a lot (see per-fold detail above) even though the row-weighted aggregate is consistent across all 11. This is measurement only; it does not change the shipped prediction path.
