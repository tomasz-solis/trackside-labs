# Raw-laps replay (shelved)

Shelved research, not production. The code lives on `shelved/challenger-research`. Why the work is blocked: `docs/MODEL_LEDGER.md`.

## Goal

Make the challenger replay (`src/analysis/challenger_research_backend.py`) feed each checkpoint the same raw per-lap practice data the live predictor gets, instead of stored profile aggregates (`practice_signal_mode="stored_profiles"`).

Without it, Q1 (practice to qualifying) and R0 (long-run pace) silently return champion-identical forecasts: their guards need raw laps and the replay passes `session_laps_by_type = {}`. Q1 reports `fallback_reason: "no_raw_practice_laps"`. So the two variants that match the core idea (one-lap pace for qualifying, long-run pace for the race) were never tested. Only the grid variants (R1, R2) could be compared.

## Rules

- Leave `config/production_config.json`, champion weights, artifacts and served forecasts untouched.
- A checkpoint may load raw laps only from sessions that existed before its `information_cutoff_at`. Add raw-lap leakage tests.
- Fail closed per checkpoint when telemetry is thin (`CheckpointInputUnavailable`). Barcelona FP1 (`teams=1 mapped=0 selected_laps=0`) is the test case.
- Load FastF1 sessions one checkpoint at a time and release them. Full-season lap data in memory has crashed runs before (`eac843c2`, `bea5e2c6`).
- Put `practice_signal_mode` in the prediction cache key so old and new results never mix.
- Outputs go under `data/historical_replay/` or `data/model_diagnostics/`.

Pointers: Q1 guard in `baseline/qualifying_mixin.py` (keep the `retrospective_diagnostic` flag), R0 evidence in `src/features/race_practice_evidence.py`.

## Done when

1. A replayed practice checkpoint gets raw laps through the live loader, not a copy of it.
2. Q1 and R0 differ from champion on at least one checkpoint, or refuse with a recorded reason. No undisclosed champion-identical rows.
3. Champion is replayed in both modes and the per-checkpoint differences are reported.
4. Thin events refuse per checkpoint only.
5. Runtime is measured on one event before the full run.
6. Champion, q0, r0, r1 (and Q1 where eligible) rerun walk-forward at practice checkpoints, 500 simulations, seeds 17, 42 and 91, run tag `raw_laps`, with a side-by-side report.
7. Tests, ruff, mypy, config hash and a clean git status.

Meanwhile, preregister `r1_joint_grid` shadows each weekend (see `QUALIFYING_RACE_CHALLENGER.md`). It was the only variant with a positive signal (finisher MAE 4.27 vs 4.34, winner 19.0% vs 14.3%).
