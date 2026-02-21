# Persistence and Supabase

This guide describes the current persistence behavior in runtime code.

## Current Runtime Integration

Artifact persistence is already used in active runtime paths:

- `src/predictors/baseline/data_mixin.py` (load core artifacts)
- `src/systems/updater.py` (save updated car characteristics)
- `src/utils/prediction_logger.py` (save/load prediction tracking payloads)
- `src/dashboard/cache.py` (artifact version checks for cache invalidation)
- `src/predictors/baseline/race/preparation_mixin.py` (missing-driver debut-year lookup)

Core layer:

- `src/persistence/artifact_store.py`
- `src/persistence/config.py`
- `src/persistence/db.py`

## Storage Modes

Storage mode comes from `USE_DB_STORAGE` (default: `file_only`):

- `file_only`: read/write local JSON only
- `db_only`: read/write Supabase only
- `fallback`: read DB first, then file fallback; writes DB only
- `dual_write`: write both DB and file; reads file path first in current implementation

If mode is not `file_only`, these env vars are required:

- `SUPABASE_URL`
- `SUPABASE_KEY`

## Supabase Assets In Repo

- SQL migration: `migrations/001_create_artifacts_table.sql`
- Connection test: `scripts/test_supabase_connection.py`
- Backfill utility: `scripts/backfill_to_db.py` (migrates `driver_debuts.csv` too)
- Predictor + storage smoke test: `scripts/test_predictor_with_db.py`

## Baseline Artifacts

These keys are relevant for the baseline predictor stack:

- `car_characteristics` -> `2026::car_characteristics`
- `driver_characteristics` -> `2026::driver_characteristics`
- `track_characteristics` -> `2026::track_characteristics`
- `driver_debuts` -> `driver_debuts`

## Recommended Rollout Path

1. Run the migration in Supabase SQL Editor using `migrations/001_create_artifacts_table.sql`.
2. Validate credentials and table access:
   - `uv run --active python scripts/test_supabase_connection.py`
3. Dry-run data migration:
   - `uv run --active python scripts/backfill_to_db.py --dry-run`
4. Run backfill with DB writes enabled:
   - set `USE_DB_STORAGE=dual_write` (or `db_only` for isolated testing)
   - `uv run --active python scripts/backfill_to_db.py`
5. Run predictor smoke test:
   - `uv run --active python scripts/test_predictor_with_db.py`

6. Verify debut artifact:
   - `driver_debuts::driver_debuts` should be present and readable via `ArtifactStore`.

## Current Caveats

- Prediction tracking UI currently loads historical predictions from local files (`PredictionLogger.get_all_predictions()` scans `data/predictions/`).
- In pure `db_only` or `fallback`, prediction writes can succeed while the dashboard accuracy history appears empty if no local files exist.
- For dashboard usage during migration, `dual_write` is the safest mode.
- File-based `list_artifacts()` fallback is intentionally minimal outside mapped artifact types.

Keep `file_only` as the default for local-only usage. Use `fallback` or `db_only` when you need DB-first reads, and `dual_write` when migrating while preserving local history files.
