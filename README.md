# Trackside Labs

F1 race weekend predictions for the 2026 regulation reset. The model forecasts qualifying and the race before the weekend, then updates after every practice, qualifying, sprint and race session.

The hard part is deciding how much to trust each source. The 2025 baseline is stale after a rules change. Testing is current, but teams hide pace. Friday practice is fresh, but noisy. The model is built around that tradeoff.

Model version: `3.0`. Every model change and its measured result is in `docs/MODEL_LEDGER.md`.

## How it works

Team strength blends three signals:

```text
strength = w_baseline * baseline + w_testing * testing + w_current * current_season
```

The weights move fast toward current-season results (`rapid_adaptive` schedule):

| Race | Baseline | Testing | Current |
|------|----------|---------|---------|
| 1    | 35%      | 20%     | 45%     |
| 2    | 20%      | 10%     | 70%     |
| 3    | 8%       | 5%      | 87%     |
| 4+   | 5%       | 0%      | 95%     |

Before any 2026 race, the current-season term falls back to the baseline.

**Qualifying.** Team strength, session pace from the available practice or sprint sessions, and driver skill go into a Monte Carlo simulation. Normal weekends weight FP3 most, then FP2 and FP1. With no session data the model runs on priors only.

**Race.** Starts from the actual or predicted grid and simulates the race lap by lap: tyre wear, fuel, pit strategy, traffic, circuit overtaking difficulty, pit loss, safety cars, lap 1 incidents and retirements. Each forecast uses 300 simulations. Output is the finish order, strategy mix, podium probability and uncertainty bands.

**Learning.** After each race the model updates driver ratings and teammate gaps from the actual results. It skips retrospective runs, duplicate run IDs and missing or partial actuals, so it never learns from bad records.

## Runtime

The dashboard only reads. Background workers write:

- `scripts/warmup_precompute.py`: precomputes forecasts for each checkpoint (production)
- `scripts/run_session_automation.py`: updates after each session and attaches actuals
- `scripts/update_from_race.py`: manual race update
- `scripts/update_from_testing.py`: manual testing and practice update

Storage goes through `ArtifactStore`:

| Mode | Behaviour |
|------|-----------|
| `file_only` | Local JSON only |
| `fallback` | Read from the database, fall back to files |
| `dual_write` | Write to both (migration) |
| `db_only` | Supabase only |

## Evaluation

Every checkpoint forecast is saved and scored against the actual result once the session ends. Targets are scored separately: main qualifying, Grand Prix race, sprint qualifying and sprint race.

Model changes are measured on a walk-forward replay of the 2026 season against a measured seed-noise floor. The protocol is in `docs/MODEL_LEDGER.md`.

Evaluation reports: `docs/MODEL_CALIBRATION.md` and `docs/MODEL_ERROR_ANALYSIS.md`. Regenerate after new races:

```bash
make evaluation-gate
make candidate-audit
make shadow-challenger-audit
```

## Quick start

Python 3.11.

```bash
uv sync --extra dev
uv run streamlit run app.py
```

Precompute the next weekend, or update after a race:

```bash
uv run python scripts/warmup_precompute.py --year 2026
uv run python scripts/update_from_race.py "Spanish Grand Prix" --year 2026
```

Checks before a commit:

```bash
uv run pre-commit run --all-files
make lint
make typecheck
make test-focused
```

The full test suite is split into chunks for CI: `make test-github-chunk-a` (also `-b`, `-c`, `-d`, `-q`, `-r`, `-s`, `-e`).
