# Dashboard updates

The dashboard only reads. Workers and scripts do every update.

## What Predict does

When a user clicks Predict (`src/dashboard/pages.py`):

1. Resolves the weekend format.
2. Checks which sessions have finished.
3. Loads the warmed forecast for the current checkpoint, or the latest warmed one if the worker has not caught up.

It never runs updates, clears FastF1 caches or computes a forecast. If FastF1 data is late, the app keeps serving the last warmed checkpoint.

## What runs in the background

| Worker | Does |
|---|---|
| `scripts/warmup_precompute.py --year 2026 --require-db` | Warms the next 3 races and rebuilds accuracy for finished races. See `WARMUP_PRECOMPUTE.md`. |
| `scripts/run_session_automation.py --year 2026 --interval-seconds 300` | Polls recent events, applies session updates, saves forecast snapshots and attaches actuals after a race |

Practice-derived car characteristics are captured by these workers. State lives in the `runtime_state` table (namespace `practice_characteristics`) or `data/systems/practice_characteristics_state.json`. Lock leases in `runtime_processing_locks` stop two workers applying the same session. Settings: `baseline_predictor.practice_capture.*`.

When a session's completion status is unknown, forecast generation stops for that session instead of silently using a predicted grid. FastF1 failures raise runtime alerts.

## Manual scripts

Force a race update:

```bash
uv run python scripts/update_from_race.py "Spanish Grand Prix" --year 2026
```

Update from testing or practice (dry run unless `--apply`):

```bash
uv run python scripts/update_from_testing.py "Testing 1" --year 2026 --sessions "Day 1" --apply
```

Pre-season testing always needs this manual run. Add `--force-renew-cache` if the testing cache looks corrupt. In database modes, `--apply` writes through `ArtifactStore`.

## Caches

- FastF1: `data/raw/.fastf1_cache`
- Testing updater: `data/raw/.fastf1_cache_testing`
