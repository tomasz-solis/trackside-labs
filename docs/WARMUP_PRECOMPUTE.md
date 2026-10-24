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

PRE forecasts are computed on the first run. A logged PRE forecast is replaced when the model version changes, another race finishes or its inputs change (artifact hash), until the weekend's first session starts.

A sprint or race that has finished but has no published classification yet shows as pending and does not block the checkpoint. The payload is recomputed each run until the result is there. Once qualifying is done, qualifying is no longer forecast, and the race forecast uses the actual grid.

## Data sources

Session data comes from FastF1. Its archive needs an F1 TV login since 2026-10 (FastF1 #978). When FastF1 cannot load a session, session completion, competitive results and the race starting grid come from OpenF1 (`src/data/openf1.py`). Those rows carry `source: "openf1"`, and the counter `openf1_fallback_used_total` records each use. OpenF1's grid has no pit-lane marker. Set `TRACKSIDE_OPENF1_DISABLED=1` to turn the fallback off.

Practice laps, weather and car data for the FP blend and practice capture come from OpenF1 too (`src/data/openf1_session.py`): it fills the FastF1 session, so callers do not change. Measured against FastF1 on 2026 sessions:

| Output | Match |
|---|---|
| Lap times, compounds, stints | identical where both have a time |
| Practice capture metrics (pace, corners, braking, tyre deg) | Spearman 1.00 on most; lowest 0.78 (Baku FP1 consistency) |
| Top speed | Spearman 0.78 to 1.00; OpenF1 has no finish-line trap |
| FP blend short run | Spearman 0.66 to 1.00; OpenF1 tyre age is wrong on some refitted sets |

Car traits, track profiles, practice gaps, team race pace and the seconds mapping load qualifying and race sessions the same way (`openf1_session.get_session`). `TrackStatus` and `Deleted` come from race control messages; corners come from FastF1 circuit info placed on the fastest lap. Measured against cached FastF1:

| Output | Match |
|---|---|
| Green-lap mask (13 races, R and Q, 19,504 laps) | 99.8% |
| Track profile, quali gaps (Belgium, Hungary, Monaco, Italy) | identical |
| Team race pace | within 0.05 s |
| Corner-class and braking traits | correlation 0.93 to 1.00; Monaco 0.58 (fast) and 0.85 (braking) |
| Team tyre deg | correlation 0.91 to 1.00 |
| Deleted qualifying laps (4 sessions) | 10 of 11 |

Misses: about 16 single laps a season disagree at a flag change; Italy's red-flag restart marks 21 laps safety car that FastF1 calls green (they are dropped, not kept). Not filled: `Position`.

A session is loaded from OpenF1 only after its scheduled end and once every driver's laps reach their `session_result` lap count (true on 15 published 2026 sessions). Otherwise the stage logs a warning and retries next run, so a partial session is never saved.

During a live session OpenF1 answers 401 to every free request, past sessions included. The client then fails fast for 5 minutes. Expect the cron run during and just after each session to skip OpenF1 work. Cached responses expire after 10 minutes, so the web process picks up results published after its first look.

## Scheduling

Production runs it hourly at minute 5, Friday to Sunday UTC (`5 * * * 5,6,0` on the Render cron `trackside-labs-warmup`). The example below is a 5-minute schedule for a self-hosted worker:

```bash
*/5 * * * * cd /path/to/trackside-labs && .venv/bin/python scripts/warmup_precompute.py --year 2026 --require-db >> /tmp/f1_warmup.log 2>&1
```

Writes are idempotent, so several workers can share the schedule. With more than one instance (Render web plus worker), use a database-backed `USE_DB_STORAGE` mode; `file_only` does not share state.
