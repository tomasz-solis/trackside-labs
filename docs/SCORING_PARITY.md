# O02 scoring parity

The qualifying report scores saved predictor rows without changing their order, probabilities or intervals. The old 40/60 blend and interval widening are optional counterfactual research. They cannot supply headline scores or gate inputs.

## Scope and meaning

The input is the selected saved target and its attached outcome. Explicit targets require predicted mode, stored and computed eligibility, a matching session, and their own prediction and outcome pair. Missing or mismatched target pairs cannot fall back to another session's rows. Malformed ranks and rows are rejected; scoring does not coerce or drop them. Legacy top-level fallback remains for records without target maps, unless recorded state marks the target actual or ineligible. The checkpoint selection policy remains unchanged. This is development reanalysis, with publication eligibility unverified. Main and sprint qualifying still share headline metrics. Separate raw model output was not stored and remains unavailable.

The scorer hashes a forecast projection separately from the outcome content. A changed outcome can change the score without changing the forecast hash. Legacy forecast IDs, outcome revision IDs and missing cutoffs remain unknown. A run ID or content hash does not prove publication. The projection is a scoring boundary, not the future `ForecastCoreV1` storage contract.

Row validation follows checkpoint selection. A malformed latest forecast remains selected with an unscorable status and its recorded identity. It cannot trigger substitution of an older checkpoint. Unknown hashes on invalid records remain null. Numeric bands and supplied probabilities must be finite numbers; valid values retain their stored units.

The dashboard's numerical output and predictive model are unchanged. Report, calibration, segment, bias and error consumers use the saved output. Changes to selection, publication, outcome storage and release certification belong to O04, O05 and O07.

## Frozen comparison

The [frozen comparison](scoring_parity/2026-10-14/README.md) uses 19 artifacts. The unchanged selector yields five qualifying pairs: three sprint qualifying and two main qualifying. There are five race pairs. Each qualifying pair has 22 aligned drivers, giving 110 intervals. Outcome revision IDs and publication evidence are absent.

| Qualifying metric | Legacy counterfactual | Saved output |
|---|---:|---:|
| MAE, positions | 2.6000 | 2.9455 |
| Weighted MAE, positions | 2.2491 | 3.5924 |
| Interval coverage | 91.8182% | 80.9091% |
| Mean interval width, positions | 10.2091 | 8.2909 |

These numbers describe different evaluated values on the same selected inputs. They do not measure a predictive model improvement. The comparison pins the existing local fallback order, which is not chronological. The counterfactual has no point-in-time validity claim.

The [June report](legacy/2026-06-16/README.md) is labeled legacy counterfactual. Its archived bytes and original JSON remain unchanged. The [O01 manifest](forecast_lineage/2026-10-13/manifest.json) identifies the local input files; only the two named O01 specimens are fully preserved in the repository.

## Validation, 16 October 2026

| Check | Result |
|---|---|
| `pytest` on `test_frozen_forecast_scoring.py`, `test_generate_evaluation_report.py` and `test_prediction_serving_module.py` | 107 passed |
| Repository test chunks `a`, `b`, `c`, `q`, `r`, `s`, `e` | 1,973 passed, 16 skipped, 3 xfailed, 6 existing failures |
| `make lint`, `make typecheck`, Ruff format check, `git diff --check` | PASS; mypy checked 176 source files |
| Original evaluator rebuilt from baseline commit and compared with the new report | Old unadjusted scores equal new saved scores; old adjusted scores equal counterfactual scores; race accuracy unchanged |
| Input and archive hashes | All 19 original forecast files, archived June files, original evaluation JSON, config and lock file unchanged |

Chunk `c` returned a failure exit. Its six failures are the existing `test_prediction_regression.py` golden comparisons for qualifying, race, sprint qualifying, sprint race, China main qualifying and China main race. All other chunks passed. No golden fixtures were changed.

The existing warmup integration test wrote a synthetic two-driver Japanese qualifying file outside its temporary directory. That test output was quarantined outside the prediction store. The original 19 files still match the manifest. Isolating that test's storage is an O03 reliability follow-up.

Implementation used GPT-6 Sol. A separate GPT-6 Astra reviewer with fresh context returned `PASS WITH NONBLOCKING FINDINGS`. It independently ran all 107 focused tests, checked lint and formatting, verified the 19-file reproduction, and rebuilt the original evaluator from the baseline commit. Earlier blocking findings were corrected with regression tests for target identity, state conflicts, malformed rows, probability bounds and checkpoint substitution.

At the O02 handoff, the nonblocking finding was repeated report scoring and artifact resolution. The closeout below addresses it. Full publication identity, prospective validity and browser behavior remain unverified.

## Findings closeout, 16 October 2026

O01 now flags unknown target declarations without exposing their names or values. O02 prepares selected events once and reuses scored rank results across headline, released, segment, error and baseline consumers. Interval sample collection and summary each have one shared implementation in `model_evaluation`; pooled coverage weights intervals, and average miss distance weights misses. Internal interval components do not enter public scoring records.

Baseline comparison aggregation and deltas also share one summary helper for saved and counterfactual paths. The combined five-module check passed 151 tests. Repository lint, format, typing and diff checks passed. Both full frozen JSON and Markdown report modes match the pre-closeout snapshots after normalizing their two generation timestamps. Scoring records, cohort hashes and all 19 source files remain unchanged. A fresh GPT-6 Astra reviewer returned `PASS` with no remaining implementation findings. It independently ran the 151 tests, reproduced both complete reports, checked all 19 source hashes, and instrumented exactly five qualifying scores, five race scores, eight preceding-event baseline scores and zero late pair resolutions. An earlier review identified duplicated baseline summary formulas; those were corrected before this fresh review.

All seven full-suite chunks ran. With affected chunk `c` rerun after the final shared-helper edit, the totals are 1,984 passed, 16 skipped, 3 xfailed and the same six existing golden failures named above. No golden fixtures were changed. The warmup test's synthetic output was quarantined at `/private/tmp/trackside-findings-closeout/synthetic-japanese-grand-prix-q.json`; all original input hashes still match. O03 has not started.

## Rollback and next work

This change needs no data migration. Keep the saved artifacts and archives. Reverting the O02 source diff restores the old evaluator, whose adjusted scores must retain their counterfactual label. Nothing is deployed.

O03 is next: repair reliability gates and truthful status copy. O04 can then define shared schemas with both producer and consumer fixtures. The current heuristic readiness score remains a legacy diagnostic, not release certification.
