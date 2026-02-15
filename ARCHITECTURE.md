# F1 2026 Prediction System Architecture

## Runtime Overview

Current runtime path in the dashboard:

```text
Streamlit UI (app.py)
  -> src/dashboard/layout.py
  -> src/dashboard/pages.py
       -> src/dashboard/update_flow.py
       -> src/dashboard/prediction_flow.py
       -> Baseline2026Predictor
            -> weight_schedule.py
            -> fp_blending.py (qualifying only)
       -> src/dashboard/rendering.py
  -> qualifying + race outputs
```

`src/predictors/qualifying.py` and `src/predictors/race.py` are compatibility wrappers; they call the baseline predictor internally.

## Dashboard Modules

- `src/dashboard/cache.py`: FastF1 cache setup, file timestamp tracking, cached predictor loading.
- `src/dashboard/layout.py`: page config, CSS/theme injection, header, sidebar controls.
- `src/dashboard/pages.py`: per-page orchestration (live prediction, insights, accuracy, about).
- `src/dashboard/prediction_flow.py`: cached weekend prediction cascade + ACTUAL/PREDICTED grid switching.
- `src/dashboard/rendering.py`: qualifying/race result tables and race-specific visual sections.
- `src/dashboard/update_flow.py`: auto-update hooks for completed races and FP practice capture.

## Core Components

### 1. Baseline Predictor

File: `src/predictors/baseline_2026.py`

Responsibilities:

- Load team, driver, and track data.
- Build blended team strength (baseline/testing/current).
- Predict qualifying (Monte Carlo, median position output).
- Predict race (Monte Carlo with stochastic race factors).

### 2. Weight Schedule

File: `src/systems/weight_schedule.py`

Responsibilities:

- Blend three signals:
  - baseline capability,
  - testing directionality modifier,
  - current season performance.
- Shift trust toward current season quickly in regulation-change mode.

### 3. FP Blending

File: `src/utils/fp_blending.py`

Responsibilities:

- Pull best available session performance by weekend type.
- Convert lap times to relative team performance.
- Blend session pace with model strength.

Note: in the active baseline path, qualifying blend is fixed at 70/30.

### 4. Auto Update From Completed Races

Files:

- `src/utils/auto_updater.py`
- `src/systems/updater.py`
- `scripts/update_from_race.py`

Responsibilities:

- Detect completed races.
- Update `current_season_performance` and related metadata.
- Keep baseline and testing directionality separate from in-season updates.

### 5. Testing/Practice Directionality Updater

Files:

- `src/systems/testing_updater.py`
- `scripts/update_from_testing.py`

Responsibilities:

- Explicit/manual extraction of directional car metrics.
- Supports testing and practice sessions.
- Writes updated directionality fields to car characteristics.

This updater does not run automatically.

## Data Model

### Car characteristics

File: `data/processed/car_characteristics/2026_car_characteristics.json`

Per team:

- `overall_performance` (baseline)
- `directionality` (testing/practice-derived directional metrics)
- `current_season_performance` (list of in-season values)
- `uncertainty`

### Track characteristics

File: `data/processed/track_characteristics/2026_track_characteristics.json`

Contains track profile and overtaking difficulty used by qualifying/race modeling.

### Driver characteristics

File: `data/processed/driver_characteristics.json`

Contains racecraft, pace, experience, and DNF-related inputs.

## Qualifying Flow

```text
load lineups
  -> blended team strength (weight schedule)
  -> optional session performance blend (FP or sprint session priority)
  -> combine team + driver skill
  -> Monte Carlo simulations
  -> median grid + confidence
```

## Race Flow

```text
input grid (predicted or actual)
  -> prepare driver/team context
  -> compute stochastic race scores per simulation
  -> include DNF, lap1 chaos, strategy, safety car effects
  -> aggregate positions across simulations
  -> final finish order + confidence + podium probability
```

## Session/Weekend Handling

File: `src/utils/weekend.py`

- Uses FastF1 event format to determine sprint vs conventional weekend.
- Falls back to local track characteristics when schedule cannot be fetched.

## Caching

- Primary FastF1 cache: `data/raw/.fastf1_cache`
- Testing updater cache (default): `data/raw/.fastf1_cache_testing`
- Streamlit cache is invalidated when relevant data file timestamps change.

## Notes On Legacy Components

- Bayesian ranking and learning modules still exist and are testable.
- They are not the direct scoring path used by the dashboard’s baseline predictor flow.

## Modularization Candidates (Next)

To continue reducing file size without changing outputs:

- `src/predictors/baseline_2026.py` (1495 lines)
- `src/systems/testing_updater.py` (1260 lines)
