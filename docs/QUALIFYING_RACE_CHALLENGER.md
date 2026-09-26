# Qualifying and race challengers (shelved)

Shelved research, not production. The framework was shelved on 2026-07-29. Its code (`src/analysis/challenger_*`, `src/models/qualifying_practice_*`, `scripts/run_challenger_*` and tests) lives on the branch `shelved/challenger-research` and does not run against current `master`. The step-by-step commands are in this file's git history on that branch. Measured verdicts are in `docs/MODEL_LEDGER.md` ("Challengers tested").

Kept for the method: how to test a challenger without leaking the future.

## Variants

| Variant | Idea |
|---|---|
| `q0_driver_state` | Drop legacy form reuse where seconds-based driver state exists |
| `q1_qualifying_practice` | Practice pace to qualifying potential |
| `r0_long_run` | Long-run pace and tyre wear to race pace |
| `r1_joint_grid` | Sample the race grid from the joint qualifying distribution |
| `r2_no_anchor` / `r2_source_anchor` | Remove, or recalibrate, the second grid anchor |

The registry (`src/models/challenger_variants.py` on the branch) holds all 48 valid combinations, including `champion`. The two R2 modes are mutually exclusive.

Result: nothing beat the champion. Q0 lost under every tuning. R1 was noise. Q1 and R0 never ran, because replay fed stored profiles instead of raw laps (see `RAW_LAPS_REPLAY_HANDOFF.md`).

## Method rules

**Preregistration.** A run counts as a preregistered shadow only if the manifest and a scrubbed champion and challenger forecast pair were frozen before qualifying. Anything frozen later is a `retrospective_diagnostic`: valid for research, never for promotion.

**Chronology.** Every input must predate the checkpoint's information cutoff. Artifacts require `max_input_timestamp < cutoff <= generated_at <= manifest.created_at`. Never backdate `generated_at`. Training and calibration events must be disjoint.

**Provenance.** The manifest hashes the source, the dirty diff, both config files, the candidate definition, snapshot IDs, the cutoff, simulation counts and fixed seeds 17, 42 and 91. Chain: candidate definition and fitted artifacts, then manifest, then bundle, then launch envelope. Any changed binding falls back to the champion.

**Fail closed.** Wet, mixed, missing, wrong-weekend or thin inputs fall back to the champion with a recorded reason, per checkpoint, never voiding a whole variant.

**Data minimums.** Main Q1 needs 30 earlier training events plus a calibration holdout; sprint Q1 needs 8.

**Research relaxations are marked.** `research_gate_relaxation` (floored at Q1 >= 4, R2 source anchor >= 3) and `retrospective_diagnostic` must appear in the manifest, and release always rejects a manifest carrying either.

## Race evaluation

Two views with matched seeds: `conditional_actual_grid` (the official grid, pit lane starts included) and `end_to_end_predicted_grid`.

- R0 or R1 (race input or grid): end-to-end finisher MAE must improve by at least 0.10, conditional-grid MAE may worsen by at most 0.05.
- R2 or race physics: both views must improve by at least 0.10.

Grid-anchor calibration needs `event_at` on every row, a training cutoff, and no overlap with evaluation events.

## Release

Research helpers only produce decisions. A release also needs: each candidate and the combination passing; all audits passing (evaluation, candidate, shadow, movement, promotion, leakage); a weekday; `champion` as the immediate rollback; and the old champion in shadow for at least three weekends. The movement audit flags any change above two positions or ten head-to-head points.
