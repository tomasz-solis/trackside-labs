# Persistence and Supabase Status

This guide documents the current persistence state in code, including what is already integrated and what is still being stabilized.

## Current Runtime Integration

Artifact persistence is already used in active runtime paths:

- `src/predictors/baseline/data_mixin.py` (load core artifacts)
- `src/systems/updater.py` (save updated car characteristics)
- `src/utils/prediction_logger.py` (save/load prediction tracking payloads)
- `src/dashboard/cache.py` (artifact version checks for cache invalidation)

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
- Backfill utility: `scripts/backfill_to_db.py`
- Predictor + storage smoke test: `scripts/test_predictor_with_db.py`

## Recommended Rollout Path

1. Run the migration in Supabase SQL Editor using `migrations/001_create_artifacts_table.sql`.
2. Validate credentials and table access:
   - `python scripts/test_supabase_connection.py`
3. Dry-run data migration:
   - `python scripts/backfill_to_db.py --dry-run`
4. Run backfill with DB writes enabled:
   - set `USE_DB_STORAGE=dual_write` (or `db_only` for isolated testing)
   - `python scripts/backfill_to_db.py`
5. Run predictor smoke test:
   - `python scripts/test_predictor_with_db.py`

## Current Caveats

- Prediction tracking UI currently loads historical predictions from local files (`PredictionLogger.get_all_predictions()` scans `data/predictions/`).
- In pure `db_only` or `fallback`, prediction writes can succeed while the dashboard accuracy history appears empty if no local files exist.
- For dashboard usage during migration, `dual_write` is the safest mode.
- File-based `list_artifacts()` fallback is intentionally minimal (`_list_files` is not fully implemented).

## Status Note

Supabase connection hardening and migration safety work are currently in progress. Keep `file_only` as the default for conservative runs, or use `dual_write` when validating DB integration without losing file-based dashboard behavior.
