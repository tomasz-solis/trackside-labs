# Tyre compound analysis

Teams differ on SOFT, MEDIUM and HARD. The model measures that from session laps and uses it in the race simulation.

## What is measured

Sources: race sessions (`src/systems/updater.py`), practice and pre-season testing (`src/systems/testing_updater.py`). Extraction: `src/systems/compound_analyzer.py`.

Per team and compound:

| Metric | Meaning |
|---|---|
| Median lap time | Typical pace on the compound (median resists outliers) |
| Degradation slope | Seconds lost per lap, linear fit over the stint; only -0.3 to +1.0 accepted |
| Consistency | Lap time standard deviation |
| Laps sampled | Used to weight reliability |

A compound needs at least 8 laps per team (`MIN_LAPS_PER_COMPOUND`). The first laps are tyre warm-up.

## Normalisation

Only within a track, never across tracks: a SOFT at Melbourne is not a SOFT at Monaco. For each compound at a track, teams are scaled 0 to 1 (1 is best) on pace, degradation and consistency (`pace_performance`, `tire_deg_performance`, `consistency_performance`). Raw values are kept too.

New data at the same track is blended 50/50 with the old. Data from a different track replaces it.

## How it is used

**Compound choice.** `get_tire_stress_score` in `src/data/track_data_loader.py` averages traction, braking, lateral and abrasion stress from `data/2025_pirelli_info.json`. `_sample_compound_sequence` in `src/simulation/pit_strategy.py` picks harder compounds for high stress. Thresholds: `baseline_predictor.compound_selection` (high 3.5, low 2.5, fallback 3.0).

**Team strength.** `src/data/compound_performance.py` turns 70% pace plus 30% degradation into a modifier of at most ±0.05 around neutral. Example: base 0.75, good on SOFT +0.03, adjusted 0.78. It applies only with at least 2 compounds, 10 laps in total and 3 laps per compound; otherwise the base strength is used.

**Race simulation.** `baseline/race/preparation_mixin.py` and `prediction_mixin.py` pass per-compound strength and degradation into the lap-by-lap Monte Carlo.

## The race simulation

Code: `src/simulation/tire_degradation.py` (tyres and fuel), `src/simulation/pit_strategy.py` (strategies), `src/utils/lap_by_lap_simulator.py` (the race).

- Two-compound rule enforced in dry races.
- Pit timing varies by about ±3 laps for a one-stop.
- Wear is linear from the compound's slope and grows with fuel load.
- Fresh tyres give SOFT 0.5s, MEDIUM 0.3s, HARD 0.1s over the first 2 to 3 laps.
- Cars at the front get about 5% more tyre life, cars at the back about 5% less.
- Pit loss is per track (Monaco 19s, Singapore 24s).
- High-stress tracks see about 80% two-stop strategies.

Settings: `baseline_predictor.race.tire_strategy.*`, `tire_physics.*`, `strategy_constraints.*`. Stored in `data/processed/car_characteristics/2026_car_characteristics.json` under `compound_characteristics`.

Tests: `tests/test_compound_analyzer.py`, `tests/test_tire_degradation.py`, `tests/test_pit_strategy.py`.

## Effect

With reliable data, a team's strength moves by 0.02 to 0.05, about 0.5 to 1 race position.

## Not modelled

- Intermediate and wet compounds are collected but unused; rain falls back to base strength.
- Track temperature. `get_fresh_tire_advantage()` already takes `track_temp`.
- Track evolution, driver-specific tyre management, undercut and overcut.
