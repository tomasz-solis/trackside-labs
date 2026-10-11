# Warmup precompute worker

Precomputes forecasts outside the Streamlit request, so the dashboard never computes on a click. The race dropdown shows only warmed races.

## Command

```bash
uv run python scripts/warmup_precompute.py --year 2026
```

| Flag | Effect |
|---|---|
| `--dry-run` | Plan only, no writes |
| `--verbose` | Per-target checkpoint detail and reuse counts |
| `--require-db` | Fail if storage is not database-backed or a write check warns |
| `--no-verify-writes` | Skip the read-back after each database write |

Exit code 0 means success, nothing to do, or checkpoint not ready. Anything else is a failure.

## What a run does

1. Picks the next race as the anchor and builds a 3-race horizon.
2. Checks which checkpoints have real session data. Normal weekend: PRE, FP1, FP2, FP3, Q. Sprint weekend: PRE, FP1, SQ, Sprint, Q.
3. If the expected data is not ready, writes a small status and exits.
4. Otherwise, for each race: computes the base features once, then only the missing weather scenarios (`dry`, `mixed`, `rain`). Each race uses its own checkpoint key.
5. Takes a database lock so two workers do not overlap.
6. Updates the horizon index (`ready_races`) that filters the dropdown.
7. Rebuilds accuracy snapshots for recently finished races.

Before computing, each run learns from finished races and refreshes three measured inputs: team race pace, the strength to seconds mapping and car traits. A slow first run after a deploy is a full recompute, not a stale lock.

PRE forecasts are computed on the first run. A logged PRE forecast is replaced when the model version changes or another race finishes, until the weekend's first session starts.

A sprint or race that has finished but has no published classification yet shows as pending and does not block the checkpoint. The payload is recomputed each run until the result is there. Once qualifying is done, qualifying is no longer forecast, and the race forecast uses the actual grid.

## Scheduling

Production runs it hourly at minute 5, Friday to Sunday UTC (`5 * * * 5,6,0` on the Render cron `trackside-labs-warmup`; `render.yaml` is out of date). The example below is a 5-minute schedule for a self-hosted worker:

```bash
*/5 * * * * cd /path/to/trackside-labs && .venv/bin/python scripts/warmup_precompute.py --year 2026 --require-db >> /tmp/f1_warmup.log 2>&1
```

Writes are idempotent, so several workers can share the schedule. With more than one instance (Render web plus worker), use a database-backed `USE_DB_STORAGE` mode; `file_only` does not share state.
