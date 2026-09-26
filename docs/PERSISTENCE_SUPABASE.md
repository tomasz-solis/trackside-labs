# Persistence and Supabase

Where artifacts, runtime state and accuracy snapshots are stored.

## Code

Core: `src/persistence/artifact_store.py`, `config.py`, `db.py`, `runtime_state_store.py`, and `src/utils/operational_observability.py`.

Main users: `baseline/data_mixin.py` (loads artifacts), `src/systems/updater.py` (saves car characteristics), `prediction_logger.py` and `accuracy_snapshots.py` (forecasts and scores), `src/dashboard/cache.py` (artifact versions), `baseline/race/preparation_mixin.py` (driver debut lookup).

## Storage modes

`USE_DB_STORAGE`, default `file_only`:

| Mode | Reads | Writes |
|---|---|---|
| `file_only` | Files | Files |
| `db_only` | Supabase | Supabase |
| `fallback` | Supabase, then files | Supabase |
| `dual_write` | Supabase, then files | Both |

Any mode except `file_only` needs `SUPABASE_URL` (checked at startup, must be `https://`) and `SUPABASE_KEY` (`service_role`). Use `file_only` locally and `dual_write` while migrating.

## Tables

| Table | Holds |
|---|---|
| `artifacts` | All artifacts, including forecasts and accuracy snapshots |
| `runtime_state` | Session boundary snapshots, practice update progress, race learning dedupe, warmup cache |
| `runtime_processing_locks` | Lease locks so workers do not apply a session twice |
| `operational_events` | Counters and alerts |
| `app_events` | Dashboard telemetry. RLS forced, `service_role` only, no IPs, emails or raw user agents |

Baseline artifact keys: `2026::car_characteristics`, `2026::driver_characteristics`, `2026::track_characteristics`, `driver_debuts`.

## Accuracy artifacts

`prediction` is the source of truth: legacy `qualifying` and `race` fields, `targets`, `actuals.targets` and metadata with `weekend_format`.

`accuracy_snapshot` holds one scored target at one checkpoint, key `YYYY::Race Name::CHECKPOINT::TARGET_KEY`. Metadata: year, race, checkpoint, weekend format, target, target session, predicted and generated times, source run ID, eligibility. Metrics: `field_size`, `overall_mae`, `top_3_hits`, `top_3_pct`, `top_10_hits`, `top_10_pct`, `exact_accuracy`, `within_1`, `within_3`, `correlation`.

The dashboard reads snapshots first and falls back to raw forecasts.

## Setup

1. Run the migrations in the Supabase SQL editor, in order: `migrations/001` to `006`. On tables created before the security defaults, `003_harden_rls_policies.sql` enforces RLS and removes `anon` and `authenticated` access.
2. Test the connection: `uv run python scripts/test_supabase_connection.py`
3. Check artifact keys: `uv run python scripts/normalize_dashboard_artifacts_in_db.py --env-file .env.local` (add `--apply` to fix).
4. Migrate data: `uv run python scripts/backfill_to_db.py --dry-run`, then with `USE_DB_STORAGE=dual_write` run it without `--dry-run`. This includes `driver_debuts.csv`.
5. Smoke test: `uv run python scripts/test_predictor_with_db.py`
6. Backfill snapshots: `uv run python scripts/backfill_accuracy_snapshots.py --year 2026`
7. Check that a Predict click writes `runtime_state`, that parallel practice runs show lock contention, that alerts appear in `operational_events`, and that `auto_update_from_races()` does not relearn processed races after a restart.

## Maintenance scripts

**Sync one race's dashboard rows.** Safer than a full backfill when the local repo has unrelated changes. Compares `prediction` artifacts and the main-target snapshots for the given checkpoints; `--include-auxiliary-targets` adds sprint targets and `--sync` pushes the differences.

```bash
uv run python scripts/sync_dashboard_datapoints_to_db.py --env-file .env.local --year 2026 --race-name "Chinese Grand Prix" --checkpoint FP1 --checkpoint SQ --checkpoint SPRINT
```

**Prune stale warmup rows.** Warmup rows are keyed by artifact hash and are cache, not history. Old hashes can surface stale data. This removes them from `precomputed_predictions`, `precomputed_prediction_base_features` and `prediction_precompute_horizon_index` only. Dry run by default, `--apply` deletes.

```bash
uv run python scripts/prune_stale_precompute_state.py --env-file .env.local --year 2026 --require-db
```

## Caveats

- Snapshots are derived. Missing ones are recomputed, but backfilling makes charts faster.
- File-mode `list_artifacts()` only covers mapped artifact types.
- Runtime state writes and updater side effects are not one transaction.
