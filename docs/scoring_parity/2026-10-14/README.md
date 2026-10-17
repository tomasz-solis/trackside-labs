# Frozen O02 comparison

[comparison.json](comparison.json) records the same 19 local input files before and after O02. The unmodified evaluator at `ad6f0b65f9d4cb15e9dd4babec2dd5aa01800b3c` produced the legacy adjusted scores. The new opt-in counterfactual reproduces those scores. The new saved-output scores reproduce the old unadjusted scores. Race scores remain unchanged.

The saved JSON includes outcomes. All 19 file hashes were checked before and after scoring. Unknown outcome revision IDs remain null; outcome content hashes bind the attached results. The local fallback order is pinned in the comparison. It is not chronological and does not validate leakage safety.

The input files are ignored local artifacts, not a portable full fixture corpus. Reproduction requires those exact files. The O01 manifest preserves their hashes, and two full specimens are checked in. No database, network, environment file or report writer is needed for this check.

From the repository root:

```bash
.venv/bin/python - <<'PY'
import copy, hashlib, json
from pathlib import Path
from unittest.mock import patch
from scripts import generate_evaluation_report as module

root = Path.cwd()
evidence = json.loads((root / "docs/scoring_parity/2026-10-14/comparison.json").read_text())
manifest_path = root / evidence["manifest_path"]
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == evidence["manifest_sha256"]
manifest = json.loads(manifest_path.read_text())
predictions = []
for artifact in manifest["artifacts"]:
    path = root / artifact["source_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"], str(path)
    predictions.append(json.loads(path.read_text()))
original = copy.deepcopy(predictions)
with patch.object(module, "_load_predictions", return_value=(predictions, "frozen_local_manifest")), patch.object(module, "_calendar_order_map", return_value=evidence["calendar_order"]):
    report = module.build_report(2026, root / "data/predictions", include_counterfactual=True)
    repeat = module.build_report(2026, root / "data/predictions", include_counterfactual=True)
assert predictions == original
assert report["scoring_records"] == repeat["scoring_records"]
assert report["evaluation_scope"]["cohort_hash"] == evidence["selected_qualifying_cohort_hash"]
assert report["released_model_metrics"] == evidence["saved_output"]
assert report["counterfactual_metrics"]["qualifying_calibration"] == evidence["legacy_counterfactual"]["qualifying_calibration"]
assert report["counterfactual_metrics"]["qualifying_accuracy"]["mae"] == evidence["legacy_counterfactual"]["qualifying_accuracy"]["mae"]
for artifact in manifest["artifacts"]:
    assert hashlib.sha256((root / artifact["source_path"]).read_bytes()).hexdigest() == artifact["sha256"]
print("PASS: saved-output parity, stable cohort and unchanged files")
PY
```
