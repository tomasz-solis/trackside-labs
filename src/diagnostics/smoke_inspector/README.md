# Smoke-session inspector

Read-only FastF1 inspection used to lock the smoke sessions in `docs/fixes/matched_lap_extractor_smoke_sessions.md`. Not used by production prediction; if a production module ever imports it, that is a bug.

It loads sessions and counts laps, retirements, weather samples and track status events, finds how far each driver got in qualifying, and writes evidence files. It does not pair laps, classify weather, or decide what a comparable lap is; that is the extractor's job.

## Usage

```bash
uv run python scripts/inspect_smoke_sessions.py --cache-dir data/raw/.fastf1_cache --output-dir data/diagnostics/smoke_session_inspections
```

Each session writes `<year>_<category>.json` (full summary) and `<year>_<category>.txt` (short summary, also printed). Sessions are listed in `SMOKE_SESSIONS` in the script; edit that list or call `run_inspections` with your own. Track status counts in the text output are FastF1 status rows (`SC_rows`, `VSC_rows`), not incident counts.

## Layout

```text
src/diagnostics/smoke_inspector/
    inspector.py    # pure summaries over FastF1 DataFrames
    loader.py       # the only FastF1 import
scripts/inspect_smoke_sessions.py
tests/test_smoke_inspector.py
```

The smoke sessions are locked, so this package can be archived or deleted and restored from git history if needed.
