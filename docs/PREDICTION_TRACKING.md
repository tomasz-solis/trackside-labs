# Prediction tracking

How checkpoint forecasts, actual results and accuracy snapshots are stored.

## Code

| File | Job |
|---|---|
| `src/utils/prediction_logger.py` | Save forecasts, attach actuals |
| `src/utils/session_detector.py` | Which sessions have finished |
| `src/utils/accuracy_targets.py` | Target mapping |
| `src/utils/prediction_metrics.py` | Metrics |
| `src/utils/accuracy_snapshots.py` | Snapshot helpers |
| `scripts/update_prediction_actuals.py` | Attach actuals |
| `scripts/backfill_accuracy_snapshots.py` | Backfill snapshots |
| `scripts/sync_dashboard_datapoints_to_db.py` | Compare or sync local rows with Supabase |

## Saving

1. The app finds the latest finished session. Before any session it saves as `PRE`.
2. It stores every target that is still a real forecast at that checkpoint.
3. It writes through `ArtifactStore`, at most once per race and session.

A session counts as finished at its scheduled time plus a buffer. With files enabled, forecasts go to `data/predictions/<year>/<race_slug>/<race_slug>_<session>.json`. Slugs are lowercase ASCII; path tricks (traversal, drive prefixes, separators, control characters) are rejected.

## Targets and checkpoints

| Target | Normal weekend | Sprint weekend |
|---|---|---|
| `main_qualifying` | PRE, FP1, FP2, FP3 | PRE, FP1, SQ, Sprint |
| `grand_prix_race` | PRE, FP1, FP2, FP3, Q | PRE, FP1, SQ, Sprint, Q |
| `sprint_qualifying` | | PRE, FP1 |
| `sprint_race` | | PRE, FP1, SQ |

## Saved payload

- `metadata`: year, race, session, time, weather, blend info, `run_id`, `weekend_format`
- `targets.<target>`: `target_session`, `predicted_order`, `result_mode`, `grid_source`, `fp_blend_info`, `mean_confidence`, `eligible_at_save`
- `shadow_challengers.<target>`: an alternate order for audits, built only from champion rows and earlier actuals. Never shown as the forecast.
- `actuals.targets.<target>`
- `qualifying.predicted_grid`, `race.predicted_results`, `actuals.qualifying`, `actuals.race`: the old shape, kept for compatibility. New code reads `targets` first.

## Attaching actuals

Workers do this automatically: session automation after a race, and warmup when `dashboard.prediction_precompute.reconcile_accuracy_after_warmup` is on. The Refresh Actuals button is for repairs. By hand:

```bash
uv run python scripts/update_prediction_actuals.py "Spanish Grand Prix" FP1 --year 2026
```

Backfill missing snapshots (only targets that already have actuals):

```bash
uv run python scripts/backfill_accuracy_snapshots.py --year 2026 --dry-run
uv run python scripts/backfill_accuracy_snapshots.py --year 2026
```

Compare one race's local rows with Supabase, and push the differences with `--sync` (add `--include-auxiliary-targets` for sprint targets):

```bash
uv run python scripts/sync_dashboard_datapoints_to_db.py --env-file .env.local --year 2026 --race-name "Australian Grand Prix" --checkpoint FP1 --checkpoint FP2 --checkpoint FP3
```

Remove warmup cache rows from old artifact hashes (add `--apply` to delete; saved forecasts and snapshots are untouched):

```bash
uv run python scripts/prune_stale_precompute_state.py --env-file .env.local --year 2026 --require-db
```

## Learning

Attaching actuals (`PredictionLogger.update_actuals()`) also updates the learner in `src/systems/systematic_learning.py`: per-driver and teammate-gap errors for qualifying and race, saved to `data/learning_state.json`.

It skips retrospective runs, duplicate run IDs, missing actuals and payloads with too few matching drivers. A skipped run is not marked processed, so a later complete result can still train it.

## Accuracy page

Metrics per target. Main: `overall_mae`, `top_3_hits`, `top_3_pct`, `top_10_hits`, `top_10_pct`. Also kept: `exact_accuracy`, `within_1`, `within_3`, `correlation`, `field_size`.

The page shows KPI cards for main qualifying and the Grand Prix, weekend and season charts split by normal and sprint weekends, sprint drilldowns, and a list of saved forecasts.

## Limits

1. Session labels must match saved keys exactly.
2. Actuals depend on FastF1.
3. Early sprint weekends can have gaps where a target was never saved.
4. Driver codes must match between forecasts and results.
