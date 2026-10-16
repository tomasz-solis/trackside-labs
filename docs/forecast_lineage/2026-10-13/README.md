# Frozen local baseline

[manifest.json](manifest.json) maps all 19 original forecast files to their SHA256 and recorded identity. It freezes the local inventory before validation. Metadata is recorded evidence, not verified publication or input availability.

Two byte-for-byte specimens are preserved under `tests/fixtures/forecast_lineage/2026-10-13/`: Australian FP1 for a normal weekend and Miami SQ for a sprint weekend. Their paths and hashes are in the manifest. These personally owned local model outputs are deliberate test fixtures, not current forecasts. They include mutable actuals as they existed at collection; the hashes identify whole file bytes, not immutable forecast cores.

Do not overwrite this snapshot or count it as prospective performance. A later snapshot needs a separate dated directory. The manifest preserves identity for the other 17 files; their full contents remain local and are not recoverable from a checksum alone. The two preserved specimens support offline reproduction without those local files.

```bash
.venv/bin/python scripts/audit_forecast_lineage.py tests/fixtures/forecast_lineage/2026-10-13
.venv/bin/pytest tests/test_audit_forecast_lineage.py -q
```
