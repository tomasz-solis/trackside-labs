# Trackside Labs

F1 race weekend predictions for the 2026 regulation reset. The model forecasts qualifying and the race before the weekend, then updates after every practice, qualifying, sprint and race session.

The hard part is deciding how much to trust each source. The 2025 baseline is stale after a rules change. Testing is current, but teams hide pace. Friday practice is fresh, but noisy. The model is built around that tradeoff.

Model version: `3.6`. Every model change and its measured result is in `docs/MODEL_LEDGER.md`.

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

Before any 2026 race, the current-season term falls back to the baseline. The testing term equals the baseline today: track suitability from testing lost out of sample, so it was removed.

**Qualifying.** Team strength, session pace from the available practice or sprint sessions, and driver skill go into a Monte Carlo simulation. Since 3.2 each car is also adjusted for the track: top speed, corner speeds and tyre wear against the circuit's layout. Normal weekends weight FP3 most, then FP2 and FP1. With no session data the model runs on priors only.

**Race.** Starts from the actual or predicted grid and simulates the race lap by lap: tyre wear, fuel, pit strategy, traffic, circuit overtaking difficulty, pit loss, safety cars, lap 1 incidents and retirements. Each forecast uses 300 simulations. Output is the finish order, a likely range per driver, win and podium probability and the strategy mix.

**Learning.** After each race the model scores its forecasts and learns how wide its bands need to be. It also tracks each driver's and teammate pair's errors, but since 3.6 it does not shift the order with them: on the leak-free replay that shift made qualifying and the race slightly worse. It skips retrospective runs, duplicate run IDs and missing or partial actuals, so it never learns from bad records.

## The plan and how it turned out

The plan: start from last year's order, add testing, and let each session's data take over. Qualifying from one-lap practice pace, the race from a lap-by-lap simulation.

| Plan | Outcome |
|------|---------|
| Blend baseline, testing and current results | Kept. Testing adds nothing of its own (see above), so races 1 and 2 lean on last year's order. |
| Reuse the 2022 to 2025 calibration | Wrong after the reset. Refitting the strength to seconds mapping on 2026 only cut qualifying MAE from 2.66 to 2.57 and race MAE from 4.06 to 3.92. It now refits after every qualifying and race. |
| Race order from simulated pace | The grid carries most of it. Halving the grid anchor cost about 0.017 race correlation. Three race pace additions (car traits, tyre wear, practice) were too small to measure. |
| Car traits by track | Adopted for qualifying as model 3.2: correlation +0.005 and +0.004 on two seeds, against a seed-noise floor of 0.003. No measurable gain for the race. |
| Use the sprint qualifying result | Adopted as model 3.4: after sprint qualifying, main qualifying is pulled 70% toward its order. MAE at that checkpoint 2.51 to 1.87 and 2.50 to 1.90 on two seeds, better on 5 of 6 sprint weekends. Race effect below the noise floor. |
| Honest race bands | Adopted as model 3.5: the race and sprint p5 to p95 band widens with learned residuals; the order is unchanged. On the leak-free replay it covers 85% of Grand Prix and 87% of sprint results, short of the 90% target. The dashboard shows a narrower likely range, about 4 to 7 places, that holds 58 to 63% of results. |
| Learn each driver's errors | Turned off in 3.6. On the leak-free replay, switching off the learned driver and teammate shift cut qualifying MAE by 0.08 to 0.09 and race MAE by 0.05 to 0.06 on two seeds. The old replay said the opposite because of the leak. |
| Remove hand-tuned race layers | The cap on places gained from the grid and the racecraft shift after the simulation stay: switching either off leaned worse on both seeds of the leak-free replay. |
| Train an ML model on the same data | Ridge and gradient boosting, refitted before each race on earlier races only. On the leak-free replay a ridge stacked on the live forecast beats the live model for qualifying (MAE 1.98 vs 2.12) and ties for the race (3.63 vs 3.69). The live `qualifying_ridge` challenger beats it on both seeds too. Promotion is open. |

Rows 2 to 5 were measured on a replay that leaked each race's result into its own practice forecasts, fixed 2026-10-31. Both sides of each comparison leaked, so the direction is likely right but the sizes are not exact.

Where it stands on the leak-free replay of 3.6, rounds 5 to 17, last checkpoint before qualifying: rank correlation 0.887 and MAE 2.12 places for qualifying, 0.642 and 3.69 for the race. Practice barely moves it. From the pre-weekend forecast to FP3, qualifying MAE improves by about 0.13 places and the race not at all. The forecasts actually published, scored over all 16 weekends, are in `docs/MODEL_CALIBRATION.md`: MAE 2.90 for qualifying and 3.38 for the race. Those cover every round, each forecast by the model version live at the time. On that record qualifying is 0.18 places worse than repeating the last race's order, so the production gate currently fails. Today's model, replayed, beats that naive order by about 0.5 places in qualifying and 1.0 in the race. What it still does badly is in `LIMITATIONS.md`.

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

Evaluation reports: `docs/MODEL_CALIBRATION.md` and `docs/MODEL_ERROR_ANALYSIS.md`. They are generated and carry their own date. Regenerate after new races:

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
