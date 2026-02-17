# Dashboard Update Behavior

This guide explains what the app updates automatically and what still requires an explicit script run.

## Automatic During `Generate Prediction`

When the user clicks **Generate Prediction** in `src/dashboard/pages.py` (called by `app.py`):

1. `auto_update_if_needed()` (in `src/dashboard/update_flow.py`) checks for completed races not yet marked as learned.
2. If new races are found, it runs `auto_update_from_races()`.
3. Characteristics are updated through `src/systems/updater.py`.
4. Streamlit caches are cleared so the new data is used immediately.

This block is race-result ingestion.

### Practice-characteristics auto capture

During weekend predictions, the app also checks completed FP sessions and can update
car characteristics from FP telemetry (FP1/FP2/FP3). It only runs when a new FP session
is detected for that race, then stores an update state in:

- `data/systems/practice_characteristics_state.json`

Behavior can be tuned in `config/default.yaml` under:

- `baseline_predictor.practice_capture.*`

## Automatic Session Data Retrieval

Qualifying prediction also auto-fetches the best available session data through `src/utils/fp_blending.py`.

- Normal: `FP3 > FP2 > FP1`
- Sprint: `Sprint Qualifying > Sprint > FP1`

## Manual / Explicit Workflows

### 1. Force race update manually

```bash
python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026
```

### 2. Update testing/practice directionality

```bash
python scripts/update_from_testing.py "Testing 1" --year 2026 --sessions "Day 1" --apply
```

This updater is manual.
By default it runs as dry-run; pass `--apply` to write updates.

Clarification: dashboard FP auto-capture uses the same underlying testing updater logic for
completed race-weekend FP sessions, but explicit testing-event runs (for example pre-season testing)
still require manual script execution.

## Cache Locations

- Main FastF1 cache: `data/raw/.fastf1_cache`
- Testing updater cache (default): `data/raw/.fastf1_cache_testing`

If cache corruption is suspected for testing pulls, run with:

```bash
python scripts/update_from_testing.py "Testing 1" \
  --year 2026 \
  --sessions "Day 1" \
  --force-renew-cache \
  --apply
```

## What Is Not Automatic

- Pre-season testing directionality extraction.
- Historical notebook validation runs.
- Manual backfill decisions for special analysis workflows.

## Operational Note

The dashboard auto-update depends on FastF1 schedule/session availability. If session data is delayed, updates may appear later even when race date has passed.
