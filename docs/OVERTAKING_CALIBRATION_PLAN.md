# Overtaking calibration (complete, model 3.0)

Finished 2026-08-28. This is the record of what was found and shipped. The numbers are in `docs/MODEL_LEDGER.md` (entries 2026-08-26 to 2026-08-29).

## What started it

A 20-place grid penalty for ANT at the 2026 Italian GP. The site predicted him P3 from P22 (band P2 to P10).

The penalty was applied correctly. The chain that undid it:

1. The simulator moved cars about twice as much as real races (displacement 1.95x, churn 2.01x).
2. To stay accurate, the finish blend anchored hard to the grid (`grid_anchor_weight` 0.35 to 0.58).
3. So a penalised driver was penalised twice: once by starting P22, once by the anchor.
4. `resolve_pace_anchor` was added to undo that, and restored his qualifying position in full. That erased the penalty and produced P3.

## Root cause

Position changes were not caused by passing. Position came from cumulative lap time, so a faster car moved ahead whether or not the pass model fired.

| Circuit | Passes | Position changes | Passes per change |
|---|---|---|---|
| Monaco | 46 | 2949 | 0.016 |
| Hungarian | 196 | 3151 | 0.062 |
| Belgian | 357 | 2715 | 0.131 |

Cutting pass probability by 90% moved churn by 1%. Tuning the pass model could never work. The pass model did rank circuits correctly; it just had no control over the result.

## What shipped

- **Queue rule.** A position change needs a completed pass. Pitted cars, retired cars and safety car or VSC laps are exempt.
- **Pass cap** by contending pairs.
- `resolve_pace_anchor` deleted. The grid anchor and the `max_gain` floor are skipped for penalised drivers only (flagged with `is_penalised`).
- **Team race pace** measured from green-flag laps (`scripts/extract_team_race_pace.py`) instead of classified results.
- `skill_improvement_max` 0.75 to 1.75, fitted to two independent measurements (teammate lap gap 0.352 s, stronger driver win rate 0.667).

Result:

```text
ANT P22 ->  Monza P7   Hungary P7   Monaco P8     (before: P3 / P4 / P5)
BOT P22 ->  Monza P18  Hungary P20  Monaco P20
race MAE 3.5758 -> 3.5152, suite 1762 passed, 0 failed
```

MAE could not judge any of this. Per-race standard deviation is 1.339, so every configuration tried (3.5152 to 3.6364) sits within 0.16 standard errors of the old model. Decisions were made on the physics measurements instead.

## Measurements worth keeping

**Churn vs displacement.** Churn (position changes per lap) has per-track signal: reliability 0.596, ceiling 0.772. Displacement (grid to finish, ranked within finishers) has none at one race per circuit (reliability -0.223), so it only supports a pooled claim. The two are anti-correlated (-0.479). Always rank within classified finishers; raw grid minus finish mostly measures retirements.

**Recovery from P15 or worse, 2022 to 2025** (401 driver-races, ranked within finishers, by the car's season median finish):

| Car | n | Median | p75 | p90 | Max |
|---|---|---|---|---|---|
| Top (median finish 6 or better) | 31 | +7 | +10 | +13 | +13 |
| Upper mid (7 to 11) | 99 | +3 | +5 | +8 | +12 |
| Backmarker (12 or worse) | 271 | +1 | +3 | +4 | +12 |

2026 top cars: HAD P21 to P6, VER P20 to P6. Pooling all starters hides this; 271 of 401 are backmarkers.

**2026 churn per circuit** (one race each): Spanish 3.615, Dutch 3.557, Hungarian 3.391, Chinese 3.200, Belgian 3.163, British 3.137, Japanese 2.827, Miami 2.804, Austrian 2.800, Australian 2.298, Canadian 2.212, Monaco 1.263. These are the target races' own results, so the model does not use them for 2026 forecasts (see the 2026-09-25 ledger entry).

**Other defects found.** The simulated field at the end of lap 1 is 2.3x too compressed (9.8 s vs 22.2 s) and flat across circuits. Dirty air is about 24x too small (0.037 s/lap vs 0.884 s/lap measured). `dirty_air_penalty_*` config is inert, and `_expand_overtake_cfg` rebuilds the overtake config from five inputs, dropping anything set upstream.

## Rejected

| Change | Result |
|---|---|
| Pass cap by contending pairs alone | MAE 3.5833, worse |
| Queue rule alone | MAE 3.6364, worse |
| Dirty air at its measured size | Churn 1.76 to 1.74 |
| Per-circuit lap 1 field spread | Churn 1.76 to 1.69 |
| Pass probability x0.10 | Churn 1.76 to 1.75 |
| Teammate setup offset moved into base pace | MAE 3.6439, worse (it is random noise) |

## Traps

Four indirections each silently voided an experiment:

1. `build_finish_order` is imported by name into `race_simulation`.
2. `simulate_race_lap_by_lap` is injected through `deps` and bound in `prediction_mixin`. Patch it there.
3. `calculate_dirty_air_penalty` is called without `max_penalty_s`, so its config cap does nothing.
4. `_expand_overtake_cfg` discards expanded keys set upstream.

Compute displacement inside each simulated race, then average. Using the median finish order understates it.

## Still open

- What the site should show for a penalised driver's recovery: the figure as is, a warning, or nothing. No decision yet.
- Churn correlation reached +0.453 against a 0.617 target. Displacement per track cannot be judged until 2027 adds a second race per circuit.
