# Model promotion

A model change is a challenger until it proves it helps. Reset-year signals are easy to count twice: a testing seed, a residual model and a calibration layer can each look fine alone and hurt together.

## Current model

Version `3.0`. It changed a mechanism, not a calibration. Positions in the race simulation used to change whenever a faster car's cumulative time passed another's, whether or not a pass happened (98.4% of simulated position changes at Monaco had no pass). Now a position change needs a completed pass. The same release measured team race pace from green-flag laps, fitted `skill_improvement_max` to the real teammate gap, and stopped re-anchoring a penalised driver to his penalty grid slot. Numbers are in `docs/MODEL_LEDGER.md`.

Targets are judged separately: `main_qualifying`, `grand_prix_race`, `sprint_qualifying`, `sprint_race`. A challenger can win one target without being promoted for all.

## Safe defaults

- Residual models are off by default, and skipped when the team seed is `testing_model` unless an ablation opts in.
- Conformal calibration is judged as interval calibration, not as a ranking fix.
- Shadow challengers are saved for audits and never replace the champion without a promotion decision.

## Production gate

`scripts/generate_evaluation_report.py` writes `production_gate` to `data/evaluation/2026_evaluation_report.json` and `docs/MODEL_CALIBRATION.md`. It needs:

- a report newer than the last finished race
- at least 5 scored race weekends
- qualifying and race MAE better than the previous-race baseline
- qualifying interval coverage near 90%
- no unresolved large-miss bias group

```bash
make evaluation-gate
make candidate-audit
make shadow-challenger-audit
```

## Promotion gate

`src/analysis/promotion_gate.py`. A challenger must:

- improve combined race and qualifying MAE by enough to matter
- not make race or qualifying MAE worse beyond tolerance
- not lower winner accuracy, or top 3 accuracy beyond tolerance
- not be worse on more weekends than it is better, for race and for qualifying
- beat a measured seed floor on at least one target

It returns pass or fail with the reasons.

## The seed floor

Changing only the simulator seed moves the scores. A gain smaller than that is not a gain. The current floor (2026-09-25, leak-free replay, 14 rounds, seed 42 vs 43) is qualifying MAE 0.054 and race MAE 0.058; correlation floors are in `docs/MODEL_LEDGER.md`.

Correlation is the better primary metric. MAE rounds to whole positions, so in one comparison 7 of 46 checkpoints had different forecasts and identical MAE. Correlation resolves about twice as finely and barely moves with the seed.

Measure the floor for the comparison you are gating:

```bash
uv run python scripts/replay_historical_checkpoints.py --year 2026 --through-round 14 --seed 43 --output-root data/historical_replay_seed43
uv run python scripts/compare_replay_arms.py --baseline <baseline> --seed-floor <baseline> data/historical_replay_seed43 --candidate <candidate>
```

`compare_replay_arms.py` calls a result below the floor `unresolvable`, which is different from `noise`. The promotion gate takes `seed_floor={"race_mae": ..., "qualifying_mae": ...}` and fails without it. A floor belongs to its replay and does not transfer to other seasons or sample sizes. `scripts/evaluate_testing_team_seed_model.py` runs on one seed with no floor, so its comparisons stay blocked until one is measured.

## Movement diagnostics

`src/analysis/component_diagnostics.py` compares champion, challenger and actual per race, session and driver: how many drivers moved closer, farther or not at all, MAE before and after, mean movement. A residual model that improves the mean but moves most drivers the wrong way is not promoted.

## Shadow challengers

`src/models/shadow_challenger.py`, `scripts/audit_shadow_challengers.py`, `scripts/audit_model_candidates.py`. Challengers use only earlier finished actuals and saved champion forecasts; same-race actuals are leakage. The audit reports champion vs challenger MAE per target, the number of comparable events, checkpoint MAE decay and the best candidate per target.

## Workflow

1. Run champion and challenger with separate data roots.
2. Run the ablations with `scripts/evaluate_testing_team_seed_model.py`.
3. Read the promotion gate and movement diagnostics together, then the audits.
4. Promote the smallest set of components that passes on holdouts and live races.
