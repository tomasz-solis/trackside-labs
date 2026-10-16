# Forecast lineage audit

O01 baseline, 13 October 2026. Evaluation-only qualifying adjustments remain active. Local stored forecasts cannot establish prospective publication. No predictor, serving, storage or scoring behavior changed in this task.

## Frozen baseline

| Item | Observed value |
|---|---|
| Starting commit | `ad6f0b65f9d4cb15e9dd4babec2dd5aa01800b3c` |
| Starting working tree | Clean |
| Configured model | `3.3`, `config/default.yaml:40` |
| Config SHA256 | `759b26c866d84ad24419e4c88cab5e68433901498ed0d992f99d51a856d39ccb` |
| `uv.lock` SHA256 | `99e22b1430407b04e6fd9bbfcdb2c66232807785c981dcdbbee37ef262f22664` |
| Environment | macOS 26.6.2, arm64, Python 3.11.14 |
| Installed tools | Ruff 0.15.1, mypy 1.19.1, pytest 9.0.2 |
| Installed runtime | FastF1 3.8.1, Streamlit 1.54.0, NumPy 2.4.2, pandas 2.3.3 |
| Storage inspected | Local files only; no credentials loaded or database contacted |
| Accessible archive | 19 prediction JSON files across eight event directories at initial collection and handoff |

Installed versions were measured without dependency synchronization. The lock hash records the source dependency specification, not proof that this environment exactly matches it. Public sources were not downloaded. Local archive coverage is not production coverage.

## Verified findings

| Finding | Current evidence | Disposition |
|---|---|---|
| DS-01 scoring differs from serving | `scripts/generate_evaluation_report.py:1245` calls `_apply_qualifying_production_adjustments`; `_blend_qualifying_rows:337` blends model/prior-result ranks 40/60 and changes intervals. `src/dashboard/prediction_serving.py:108` loads persisted values without that transform. | Confirmed; O02 |
| DS-02 old readiness claim | Original report generated `2026-06-16T19:46:42.011345+00:00`, seven qualifying and seven race evaluations, PASS/95. `evaluate_production_gate:1139` accepts a null latest completed race; score starts at 95. | Confirmed; archived and labeled legacy; gate repair pending |
| DS-03 CI data dependency | `.github/workflows/model-quality-gate.yml` chooses `file_only` without both secrets and runs the data-dependent generator. `.gitignore` excludes `data/predictions/`. | Confirmed source risk; remote run status NOT EXECUTED |
| DS-04 cross-model fallback | `prediction_serving.py:212` relaxes model identity and caches the result under the request key; `precomputed_predictions.py:269` scans across artifact hashes at an exact boundary. | Confirmed; no freshness/compatibility proof established here |
| DS-05/06 probability postprocessing | `result_processing.py:856` smooths podium probabilities; `_compute_podium_probabilities:936` draws independently per driver when sample augmentation is needed. | Confirmed code paths; numerical impact not measured |
| DS-07 simulation budget | `config/default.yaml:142` and `:143` set 300 simulations. | Confirmed config; production runtime/precision NOT EXECUTED |
| DS-08 nonstandard legacy JSON | Original report contains `brier_skill_score: NaN`. | Confirmed; archive preserves exact bytes |
| DS-10/VIS-01 sort caption | Config uses median rank; movement constraints may re-rank; `rendering_race.py:855` claims expected finishing position. | Confirmed source mismatch; UI repair pending |
| DS-11 stale version copy | README and promotion overview say 3.0; config and ledger say 3.3. | Confirmed; repair pending |
| DS-12 mypy failure | The ignore remains in `render_ops.py:234`; current local `make typecheck` passes. | Historical failure not reproduced; CI typing environment remains unknown |

The diagnostic reports recorded metadata, not verified lineage. `displayable_state` permits historical inspection only for forecast rows with nonempty driver/team identities, marks explicit actual/ineligible rows `not_a_forecast`, and marks missing rows `unavailable`. `scoreability_status` is `not_scoreable` for missing or ineligible pairs and `provenance_unknown` for present forecast/actual pairs. Neither status authorizes public scoring. Modern target fields take precedence over legacy aliases. A cutoff field does not prove input availability; a run ID is not a forecast content ID; actuals presence does not prove final classification. No local specimen has independently established public issuance, selection policy or immutable forecast core.

## Sample identities

Full SHA256 values identify the original file bytes, including mutable actuals. They are forensic checksums, not new forecast IDs. All specimens have unknown publication and prospective scoreability.

The [full frozen manifest](forecast_lineage/2026-10-13/manifest.json) preserves path, checksum and recorded identity for every original file. Two normal/sprint input specimens are preserved byte for byte in `tests/fixtures/forecast_lineage/2026-10-13/`; their hashes and paths are in the manifest. These are intentional offline test fixtures. The other 17 inputs are identified by the manifest but not copied into the diff. See [snapshot limits](forecast_lineage/2026-10-13/README.md).

| Local file | Recorded run/model/checkpoint | Recorded cutoff | SHA256 |
|---|---|---|---|
| `data/predictions/2026/australian_grand_prix/australian_grand_prix_fp1.json` | `56058218-7f9b-40f6-8849-6ce0307b4ca8`, 2.0, FP1 | `2026-03-06T04:59:59+00:00`; generated 12 April, so generation cannot establish contemporaneous publication | `249db1e6991571138140afb8eeb79ce2fbab4796e22759ab5c8959610fb5144d` |
| `data/predictions/2026/miami_grand_prix/miami_grand_prix_sq.json` | `d5a682a4-f217-480b-83f3-c8a0a344c6f4`, 2.0, SQ | Missing | `ff3f22324dcac2e645e0a814c083f663c313dc96ce7630affeecc68851998db2` |
| `data/predictions/2026/austrian_grand_prix/austrian_grand_prix_pre.json` | `1d8f0d1c-a186-459b-b4cc-846655d68afc`, 2.1, PRE | Missing | `c8b538135adbbd8a75cff9b48b024afcaaa9253526162a456c78d9d6eed69ce3` |

## Storage and consumer inventory

| Boundary | Owning source | Meaning and consumer |
|---|---|---|
| Saved checkpoint | `src/utils/prediction_logger.py:63` | Metadata, `targets.<target>.predicted_order`, legacy qualifying/race aliases; accuracy and evaluation consume it |
| Actuals update | `prediction_logger.py:307` | Writes target and legacy actuals into the same record; learning and accuracy use them |
| Artifact persistence | `src/persistence/artifact_store.py:37` | File/DB modes; serialized outputs and versions; no publication guarantee |
| Warmup cache | `precomputed_predictions.py:335` | Mutable runtime entry keyed by year/race/weather/artifact hash/boundary; contains `prediction_results`, metadata and update time |
| Dashboard serving | `prediction_serving.py:108` | Exact persisted lookup, prior boundary fallback, then different-model fallback |
| Target rules | `src/utils/accuracy_targets.py`, `docs/PREDICTION_TRACKING.md` | Main/sprint target eligibility; field size is dynamic |
| Evaluator | `scripts/generate_evaluation_report.py:1222` | Selects latest per event/target, transforms qualifying, joins actuals and scores |

No new DS/UI contract is introduced. Race, qualifying, saved-forecast browsing and accuracy remain downstream consumers of existing payloads. Freshness, fallback compatibility, finality and publication eligibility need O04 to O07 contracts before a redesigned UI can claim them.

## Reproduce the evidence

The new diagnostic needs only Python and explicit local inputs. It writes JSON to stdout, uses original-byte hashes, withholds unknown provenance, and never imports persistence or reads environment credentials.

```bash
.venv/bin/python scripts/audit_forecast_lineage.py data/predictions/2026
.venv/bin/python scripts/audit_forecast_lineage.py tests/fixtures/forecast_lineage/2026-10-13
.venv/bin/python scripts/audit_forecast_lineage.py data/evaluation/2026_evaluation_report.json
.venv/bin/pytest tests/test_audit_forecast_lineage.py -q
make lint
.venv/bin/ruff format --check src tests scripts app.py predict_weekend.py
make typecheck
make test-github-smoke
make test-github-chunk-a
make test-github-chunk-b
make test-github-chunk-c
make test-github-chunk-q
make test-github-chunk-r
make test-github-chunk-s
make test-github-chunk-e
```

This historical read-only reproduction uses the helpers from baseline commit `ad6f0b6` and local saved rows. O02 removed `_apply_qualifying_production_adjustments`; this snippet requires that baseline source. Use the [current frozen comparison](scoring_parity/2026-10-14/README.md) for the current evaluator. It does not regenerate the authoritative report or load ArtifactStore. It selects five eligible qualifying events from the local archive and orders them by recorded cutoff/generation time for demonstration, not as a validated scoring cohort.

```python
import copy
import json
from pathlib import Path
from scripts.generate_evaluation_report import (
    _apply_qualifying_production_adjustments,
    _resolve_session_pair,
    _select_latest_predictions,
    evaluate_production_gate,
)

payloads = [json.loads(p.read_text()) for p in sorted(Path("data/predictions/2026").rglob("*.json"))]
selected = _select_latest_predictions(payloads, session_kind="qualifying")
selected.sort(key=lambda p: p["metadata"].get("information_cutoff_at", p["metadata"].get("predicted_at", "")))
before = copy.deepcopy(selected)
adjusted, _ = _apply_qualifying_production_adjustments(selected)
for original, changed in zip(selected, adjusted, strict=True):
    source_rows, _ = _resolve_session_pair(original, session_kind="qualifying")
    adjusted_rows, _ = _resolve_session_pair(changed, session_kind="qualifying")
    print(original["metadata"]["race_name"], source_rows != adjusted_rows)
assert selected == before
legacy = json.loads(Path("data/evaluation/2026_evaluation_report.json").read_text())
print(evaluate_production_gate(legacy)["checks"]["report_after_latest_completed_race"])
```

Observed: rows differ for Australia, China, Japan, Miami and Canada; 76 driver positions change across the latter four events. Source Python objects remain unchanged. Null latest-race freshness returns `True`, and the legacy gate returns PASS/95. These are demonstrations of scoring transformation and a permissive gate, not predictive improvement estimates or a recreation of the June cohort.

## Validation and limits

| Check | Status | Executed evidence |
|---|---|---|
| Diagnostic tests | PASS | `.venv/bin/pytest tests/test_audit_forecast_lineage.py -q`: 16 passed |
| Lint and format | PASS | `make lint`; Ruff format check passed |
| Source typing | PASS | `make typecheck`: no issues in 174 source files |
| Original inputs and archive | PASS | All 19 forecast hashes plus original report hash unchanged; archived JSON/Markdown hash parity and two frozen specimen hashes verified; full identity manifest saved |
| Local diagnostic | PASS | Two runs over 19 files produce identical output; all prospective eligibility states unknown; legacy report flagged `invalid_json_nonfinite` |
| Deterministic evaluation/backtest smoke | FAIL | `make test-github-smoke`: 6 failed, 60 passed in 63.78 seconds; all six failures in existing golden regression comparisons |
| Full existing suite | FAIL | All seven chunks executed: 1867 passed, 16 skipped, 3 xfailed, 6 failed; the same six golden comparisons fail in chunk c |
| Remote CI, deployed serving and Supabase | NOT EXECUTED | No remote requests or database connections made |
| Browser screenshots, accessibility and full-season replay | NOT EXECUTED | No UI change or predictive claim in O01; these remain required for their release gates |
| Independent review | PASS WITH NONBLOCKING FINDINGS | Separate GPT-6 Sol reviewer with fresh context verified all 19 source hashes, both specimens, config/lock/archive parity, 16 tests and full-suite logs |

The smoke failures are `test_qualifying_regression`, `test_race_regression`, `test_sprint_qualifying_regression`, `test_sprint_race_regression`, `test_china_main_qualifying_regression` and `test_china_main_race_regression` in `tests/test_prediction_regression.py`. They compare current fixed-seed outputs with existing golden JSON. Examples include newly present `likely_lo`/`likely_hi` fields and changed confidence, ranks and DNF values. Cause is unverified; this task changes no imported predictor code, config or fixtures. The same failures recur in full-suite chunk c. No golden files were rewritten.

The existing `tests/test_warmup_to_serve_integration.py` stubs the cache but lets checkpoint logging reach the local prediction directory. During chunk e it created `japanese_grand_prix_q.json` with two synthetic drivers. This file was absent from the initial hash manifest and was moved to `/private/tmp/trackside-o01-validation/synthetic-japanese-grand-prix-q.json` after validation. Original inputs remain unchanged. Test-storage isolation belongs in the reliability follow-up; this synthetic file must not enter a scoring cohort.

No production state is inferred from local files or source inspection. The script inventories malformed artifacts with explicit statuses; its successful exit means the inventory ran, not that every artifact passed validation. Missing or empty input exits 2.

Implementation used GPT-6 Luna for the diagnostic and tests. Three initial independent FAIL reviews led to explicit display/scoreability states, deep-JSON handling, frozen specimens/manifest, stricter actual/driver/team row validation and bounded model-version redaction. The final reviewer started with fresh context and received only the original briefs, objective/criteria, repository diff and test access. Final verdict: PASS WITH NONBLOCKING FINDINGS for O01 only.

At the O01 handoff, the nonblocking finding was omitted unknown target keys when recognized keys coexist. The 16 October closeout adds `target_coverage` with a fixed status and unsupported declaration count across forecasts, outcomes and top-level target metadata. Mixed records report partial coverage and cannot claim whole-artifact scoreability. Unknown names and values remain redacted. Known-target summaries stay unchanged. `complete` means recognized target names, not schema or publication certification.

The other 17 source files have preserved identity but no preserved contents; only the two specimens can be reproduced without the local archive. Closeout validation and independent review are recorded in [O02 scoring parity](SCORING_PARITY.md#findings-closeout-16-october-2026).

## Next task and rollback

At the O01 handoff, O02 was next. It is now implemented; see [scoring parity](SCORING_PARITY.md). The champion remains unchanged. O03 is next after findings closeout and must investigate the observed golden failures before treating CI as a release gate; do not bless new fixtures without explaining the drift.

The safe fallback is to retain the current app and clearly labeled legacy evidence while live access remains unavailable. The owner can later supply a sealed snapshot or authorize a read-only live check. Recommend the sealed snapshot first; verify hashes and publication metadata before any current-performance claim.

Rollback removes the diagnostic, its tests, this audit, release template, frozen specimens/manifest, and the calibration notice. The legacy archive is byte-identical evidence. No migration, production write, deployment or external cost was introduced. Changes are left uncommitted for owner review.
