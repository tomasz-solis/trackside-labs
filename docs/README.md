# Docs

If a doc and the code disagree, trust the code.

## Start here

- `../README.md`: what the model does and how to run it
- `../ARCHITECTURE.md`: components and data flow
- `../CONFIGURATION.md`: config files and environment variables
- `../LIMITATIONS.md`: what the model does badly
- `MODEL_LEDGER.md`: every model change, how it was measured and the verdict. Read it before proposing a change; several plausible ideas have already lost.

## How the model works

- `WEIGHT_SCHEDULE_GUIDE.md`: blending baseline, testing and current-season strength
- `FP_BLENDING_SYSTEM.md`: how practice feeds the qualifying forecast
- `COMPOUND_ANALYSIS.md`: tyre compounds and the race simulation
- `WEEKEND_PREDICTIONS.md`: the forecast chain on normal and sprint weekends
- `OVERTAKING_CALIBRATION_PLAN.md`: why positions now change only on a completed pass (model 3.0)
- `DNF_CALIBRATION_BRIEF.md`: retirements today, and the split that is built but off

## Running it

- `WARMUP_PRECOMPUTE.md`: the worker that precomputes forecasts
- `DASHBOARD_AUTO_UPDATE.md`: what the dashboard does and what the workers do
- `PREDICTION_TRACKING.md`: saving forecasts, attaching results, accuracy
- `PERSISTENCE_SUPABASE.md`: storage modes, tables, setup

## Judging a change

- `MODEL_PROMOTION.md`: production gate, promotion gate, seed floor
- `MODEL_CALIBRATION.md` and `MODEL_ERROR_ANALYSIS.md`: generated evaluation reports
- `../reports/backtest_2025/REVIEW_PACKET.md`: 2025 backtest summary
- `../data/model_diagnostics/2026/`: generated challenger and candidate audits

## Records

- `fixes/`: design and fix records, mostly the May 2026 work to separate driver ratings from car pace. Start with `fixes/master_execution_plan.md`.
- `QUALIFYING_RACE_CHALLENGER.md` and `RAW_LAPS_REPLAY_HANDOFF.md`: shelved research; the code is on `shelved/challenger-research`.

Supporting notebooks: `../notebooks/model_development/validate_testing_predictions.ipynb` and `test_weight_schedules.ipynb`.
