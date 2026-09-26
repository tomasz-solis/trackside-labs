# Configuration

Two config files:

- `config/default.yaml`: model and runtime parameters. The predictor reads it through `src/utils/config_loader.py`.
- `config/production_config.json`: strategy metadata and expected MAE references for `ProductionConfig` in `src/utils/config.py`. The predictor does not score with it.

Current values live in `config/default.yaml`. This page lists where to look, not the numbers.

## Sections that drive predictions

- `model.version` (currently `3.0`) and `model.regulation_eras`
- `baseline_predictor.qualifying.*`
- `baseline_predictor.race.*`
- `baseline_predictor.compound_selection.*`
- `baseline_predictor.practice_capture.*` (dashboard practice auto-capture)
- `learning.*` (sample gates and caps for `src/systems/systematic_learning.py`)

`bayesian` and `qualifying` are used by other modules and scripts, not by the main simulation. Race knobs live only under `baseline_predictor.race.*`; the old top-level `race:` section was removed on 2026-09-02 because nothing read it.

## Common changes

| Change | Keys |
|---|---|
| Qualifying team vs driver weight | `baseline_predictor.qualifying.team_weight`, `skill_weight` (must sum to 1.0) |
| Tyre compound choice | `baseline_predictor.compound_selection.*_threshold`, `default_stress_fallback` |
| Race randomness | `baseline_predictor.race.base_chaos.*`, `lap1_chaos.*`, `teammate_variance_std`, `track_chaos_multiplier` |
| Overtaking | `baseline_predictor.race.overtake_model.*`, `overtaking_transition.*`, `grid_weight_*`, `overtaking_skill_multiplier`, `final_blend.*`, `track_pass_cap_enabled` (A/B switch, on by default) |
| Retirements | `baseline_predictor.race.dnf_*` |
| Learning safeguards | `learning.min_samples`, `driver_error_scale`, `teammate_gap_scale`, `max_adjustment`, `interval_*` |
| Tyres and fuel | `baseline_predictor.race.tire_physics.*`, `fuel.*` |
| Pit strategy | `baseline_predictor.race.tire_strategy.*`, `pit_stops.*`, `strategy_constraints.*` |
| Lap times | `baseline_predictor.race.lap_time.*` |
| Paths | `paths.*` |

Track overtaking difficulty comes from `data/processed/track_characteristics/2026_track_characteristics.json`. Harder circuits anchor the grid more and cap position changes.

## Environment variables

**Storage.** `USE_DB_STORAGE` picks the mode (`src/persistence/config.py`):

| Value | Behaviour |
|---|---|
| `file_only` (default) | Local files only |
| `fallback` | Read the database first, fall back to files; write to the database |
| `dual_write` | Write both (use this to migrate) |
| `db_only` | Supabase only |

Any mode other than `file_only` needs `SUPABASE_URL` (an `https://` URL) and `SUPABASE_KEY` (the `service_role` key). The same credentials cover the `artifacts`, `runtime_state`, `runtime_processing_locks`, `operational_events` and `app_events` tables. For a rollout, start with `dual_write`, then move to `fallback` or `db_only`. See `docs/PERSISTENCE_SUPABASE.md` and `migrations/`.

**Paths and overrides.** `F1_CONFIG` (another config file), `F1_DATA_DIR` (predictor data root), `F1_CACHE_DIR` (auto-updater FastF1 cache), `TEAM_STRENGTH_SECONDS_MAPPING_PATH` (another seconds mapping file; the historical replay uses it).

**Operator panel.** The dashboard is public, so the admin panel sits behind a token. Set `TL_ADMIN_TOKEN` on the web service and open `https://<host>/?admin=<TL_ADMIN_TOKEN>`. The Admin tab has the grid penalty and driver substitution editors, precompute status, recent Render events and three buttons.

| Variable | Used for |
|---|---|
| `TL_ADMIN_TOKEN` | Unlocks the panel |
| `RENDER_API_KEY` | Render API token |
| `RENDER_PRECOMPUTE_CRON_ID` | The `preheat` cron job, for "Trigger precompute run" |
| `RENDER_WEB_SERVICE_ID` | The web service, for "Restart web service" |

The `RENDER_*` variables are optional. Without them the two Render buttons are disabled and say which variable is missing. "Clear dashboard caches" always works. Precompute never runs in the web process: the button starts the cron service, and a new trigger cancels a run in progress.

## Validation

Startup checks the config in two layers and raises on any failure:

1. `src/utils/config_schema.py` (Pydantic): types, ranges, unknown keys and cross-field rules such as the qualifying weights summing to 1.0.
2. `src/utils/config_loader.py`: required sections and ordering rules the schema cannot express.

## Changing config safely

1. Edit `config/default.yaml`.
2. If storage uses the database, check the connection: `uv run python scripts/test_supabase_connection.py`.
3. Run `uv run pre-commit run --all-files`, `make test-focused` and `make evaluation-gate`.
4. If model behaviour changed, run `make candidate-audit` and `make shadow-challenger-audit`, and measure it with the replay protocol in `docs/MODEL_LEDGER.md`.

Live FastF1 tests are opt-in: set `FASTF1_LIVE_TESTS=1` and run `pytest tests/test_fastf1_live_refresh.py -m live_fastf1`.
