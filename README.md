# Formula 1 2026 Prediction Engine

This repository predicts F1 weekends for the 2026 season using a simulation-first approach.

## What Runs In Production (Current Path)

The Streamlit app and the main weekend flow use:

- `app.py`
- `src/predictors/baseline_2026.py` (`Baseline2026Predictor`)
- `src/systems/weight_schedule.py`
- `src/utils/fp_blending.py`

`src/predictors/qualifying.py` and `src/predictors/race.py` are compatibility wrappers that delegate to `Baseline2026Predictor`.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

If port `8501` is already in use:

```bash
streamlit run app.py --server.port 8502
```

## Prediction Logic (As Implemented)

### Qualifying

- Builds team strength from baseline + testing directionality + current season performance (weight schedule).
- Pulls the best available session data:
  - Normal weekend: `FP3 > FP2 > FP1`
  - Sprint weekend: `Sprint > Sprint Qualifying > FP1`
- Blends session pace with model strength using a fixed `70/30` split in the active predictor.
- Runs Monte Carlo and returns median position with confidence bands.

### Race

- Starts from predicted qualifying grid, or actual qualifying results if already available.
- Simulates race outcomes with:
  - grid influence,
  - team pace,
  - driver skill,
  - overtaking context,
  - lap-1 chaos,
  - strategy variance,
  - safety car luck,
  - DNF probability.

## Data Update Flows

### 1. Automatic in dashboard

When you click **Generate Prediction**, the app can:

- learn from newly completed races, and
- capture completed FP sessions (FP1/FP2/FP3) into car characteristics.

### 2. Manual race update

```bash
python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026
```

### 3. Manual testing/practice directionality update

```bash
python scripts/update_from_testing.py "Testing 1" --year 2026 --sessions "Day 1"
```

To combine all available testing days, omit `--sessions`:

```bash
python scripts/update_from_testing.py "Testing 1" --year 2026
```

Useful flags:

```bash
python scripts/update_from_testing.py "Testing 1" \
  --year 2026 \
  --backend auto \
  --cache-dir data/raw/.fastf1_cache_testing \
  --session-aggregation laps_weighted \
  --run-profile balanced \
  --force-renew-cache \
  --dry-run
```

Testing cache defaults to `data/raw/.fastf1_cache_testing`.

## Important Data Files

- `data/processed/car_characteristics/2026_car_characteristics.json`
- `data/processed/track_characteristics/2026_track_characteristics.json`
- `data/processed/driver_characteristics.json`

## What Exists But Is Not The Main Dashboard Path

- Bayesian ranking components (`src/models/bayesian.py`)
- Learning method history (`src/systems/learning.py`)
- Additional scripts and legacy-compatible interfaces

These remain useful for experiments and extensions, but the app runtime path is the baseline predictor stack listed above.

## Documentation

- `ARCHITECTURE.md`
- `CONFIGURATION.md`
- `docs/README.md`
- `docs/WEEKEND_PREDICTIONS.md`
- `docs/FP_BLENDING_SYSTEM.md`
- `docs/DASHBOARD_AUTO_UPDATE.md`
- `docs/PREDICTION_TRACKING.md`
- `docs/WEIGHT_SCHEDULE_GUIDE.md`

## Tests

```bash
pytest tests/
```

Targeted examples:

```bash
pytest tests/test_baseline_2026_integration.py
pytest tests/test_dashboard_smoke.py
pytest tests/test_testing_updater.py
```
