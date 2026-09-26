# DNF calibration brief

Retirements add about 1.2 to 1.4 MAE positions on incident weekends (2026: Australia +1.39, China +1.21, Monaco +1.20). This brief covers how DNFs are modelled today, what is wrong with it, and the technical/collision split that is built but switched off.

## How DNFs work today

1. **Per-driver rate.** `_update_dnf_rate_ema` in `src/systems/updater.py` blends each race into the driver's rate: `(1-w)*old + w*retired`, clipped. It ignores the track: a Monaco retirement counts the same as a Monza one.
2. **Experience add-on.** `_EXPERIENCE_DNF_MODIFIERS` in `baseline/race/preparation_flow.py`: rookie +0.05, second year +0.03, developing +0.02, established 0.
3. **Bounds** (`src/utils/config_schema.py`): missing driver 0.10, historical cap 0.20, final cap 0.35, floor 0.02.
4. **Draw.** Per lap in `src/utils/lap_by_lap_simulator.py`: `rng.random() < dnf_probability / race_distance`. An earlier version of this brief scoped a one-draw-per-race function; that function had no callers and was deleted on 2026-09-02.
5. **Scoring.** `src/analysis/model_evaluation.py` already computes DNF Brier, a baseline Brier and a skill score, so calibration can be measured without running the race simulation.

## The problem

A DNF is two different processes:

| | Mechanical | Collision |
|---|---|---|
| Level | Team or power unit | Driver, times track |
| Teammates | Correlated | Independent |
| Track effect | Weak | Strong (street circuits much higher) |
| In 2026 | High early after the reset, falling as reliability improves | Roughly stable |

The current model has one rate per driver with a fixed blend, so it cannot see a team-wide reliability problem and keeps the high early-season rate after cars become reliable. It also has no track, grid or first-lap effect. Track files carry `safety_car_prob`, which is not used for DNFs.

## Proposed changes, ranked

1. **Split into mechanical and collision.** `p_dnf = 1 - (1 - p_technical)(1 - p_collision)`. Mechanical is per team with a decay toward a mature floor. Collision is per driver times a track multiplier.
2. **Per-track collision multiplier** from historical retirement rates, 1.0 at an average circuit, seeded from `safety_car_prob` where history is thin.
3. **First-lap term** that grows with grid position in the pack.
4. **Order retirements by lap reached**, so a late retirement classifies ahead of an early one.

## Validation rules

- Walk-forward only: rates use races strictly before the target. A circuit's own race this year never sets its own prior.
- Score calibration offline with the DNF Brier from cached actuals. Run the full simulation only once the offline score improves.
- Judge skill on finisher MAE. Random retirements are a floor no model removes.
- Everything lands behind config flags that reproduce current behaviour.

## Built, switched off

Item 1 is built behind `baseline_predictor.race.dnf_technical_collision_split_enabled` (default `False`). Item 2 exists as an optional per-circuit `collision_multiplier` (default 1.0). Items 3 and 4 are not built.

**Retirement reasons.** Stored actuals keep only a boolean `dnf`. The raw FastF1 `Status` text is available in `updater.py` at update time, and `_classify_dnf_status_reason` sorts it into mechanical, collision or other by keywords. "Other" (retired, disqualified, did not start, anything unclear) counts toward neither, so both rates undercount.

**Mechanical rate.** `_resolve_technical_dnf_probability` starts from a decay prior (`floor + amplitude * exp(-races / tau)`, same for every team) and shrinks toward the team's own mechanical rate: `(prior*k + observed*n) / (k + n)`, `k = dnf_technical_prior_strength` (3.0). A new team with no races stays at the prior. The team rate is written by `_update_team_technical_dnf_rate_ema` with its own faster blend (`dnf_team_technical_update_blend` 0.35 vs 0.10 for drivers) and stored identically on both teammates' records as `team_technical_dnf_risk`.

**Collision rate.** `_resolve_driver_collision_crash_rate` shrinks from an experience-tier prior (`dnf_collision_base_rate` 0.05 plus the experience add-on) toward the driver's own collision record with `k = dnf_collision_prior_strength` (5.0). Counts (`races_observed`, `collisions_observed`) are cumulative in `collision_dnf_track_record`, so a driver who stops crashing converges to his own low rate.

**Switched off means unchanged.** With the flag off, or with the new fields missing, the DNF path is identical to the old model. A 64-case test checks this against a reference copy of the old formula.

This brief predates the planned DNF revamp. The revamp starts from uniform production labels and must beat a flat season base rate on 2026.
