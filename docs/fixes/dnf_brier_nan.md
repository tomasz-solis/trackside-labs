# DNF Brier NaN and output calibration

2026-07-06. Part 1 is still in place. Part 2 was reversed; see the end.

## 1. `brier_skill_score: NaN` in the evaluation report

A race with no retirements has base rate 0, so its baseline Brier is 0 and its skill score is undefined. `_build_dnf_calibration_section` in `scripts/generate_evaluation_report.py` averaged the per-race skill scores, so one such race turned the total into NaN, and `json.dump` wrote it as invalid JSON.

Fix:

- The total skill is now `1 - weighted_brier / weighted_baseline` from pooled values, and `null` when no race had a retirement.
- The per-race NaN stays, because the skill really is undefined there.
- The report converts NaN and infinity to `null` and writes with `allow_nan=False`, so a future bad value fails loudly.

Tests: `tests/test_eval_metrics_split.py` (`test_dnf_calibration_zero_dnf_event_scores_brier_but_not_skill`) and `tests/test_generate_evaluation_report.py`.

## 2. Shrinking the reported probability

With the NaN fixed, the reported probabilities scored worse than a flat base rate (Brier 0.046 vs about 0.038, 13 races, 286 driver results, 11 DNFs). `scripts/probe_dnf_calibration.py` tested `p' = λp + (1-λ)r`, with `r` the base rate from earlier races only:

| λ | Pooled Brier |
|---|---:|
| 0.00 (base rate) | 0.0384 |
| 0.25 | 0.0370 |
| 0.50 | 0.0377 |
| 1.00 (unchanged) | 0.0457 |

`calibrated_dnf_probability` in `baseline/race/result_processing.py` applied it to the reported number only, with `dnf_probability_shrinkage_lambda` set to 0.25 and `dnf_probability_base_rate` 0.04. The simulation inputs were untouched.

**Reversed.** The probe's actuals held only 11 of the real 42 retirements. On complete actuals the ranking flips and the model under-forecasts. Since 2026-09-20, lambda is 1.0 and the dashboard hides the column. See `LIMITATIONS.md` section 10 and the 2026-08-26 retirements entry in `docs/MODEL_LEDGER.md`.
