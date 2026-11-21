#!/usr/bin/env python3
"""Check one race's served forecasts and accuracy against the deploy rollback trigger. Read-only.

Reads the newest served payload per checkpoint and weather scenario from the
``precomputed_predictions`` runtime state, prints one line per payload, and exits 1
on any problem:

- the payload's model version is not ``--model-version``
- a race finish order or grid is not positions 1 to N, or a driver code is not three
  capital letters (a FastF1 NaN once served as ``"nan"``), or the sections disagree on
  the field
- a win or podium chance is outside 0 to 100, below the driver's win chance, or the
  chances do not sum to 100 and 300 within whole-percent rounding (0.5 per driver)
- a forecast qualifying grid was not served by the ridge with ``champion_position``

After the race it also reads the accuracy snapshots: a target whose correlation is
below 0 (worse than a random order) at every scored checkpoint fails.

Usage:
  uv run python scripts/check_served_forecasts.py --race "Qatar Grand Prix" --model-version 3.9 --env-file .env.local
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

_NAMESPACE = "precomputed_predictions"


def _order_problems(label: str, rows: list[dict[str, Any]]) -> list[str]:
    """Positions 1 to N and one valid code per driver."""
    problems = []
    if sorted(row.get("position") or 0 for row in rows) != list(range(1, len(rows) + 1)):
        problems.append(f"{label}: positions are not 1 to {len(rows)}")
    drivers = [row.get("driver") for row in rows]
    bad = [
        d
        for d in drivers
        if not (isinstance(d, str) and len(d) == 3 and d.isalpha() and d.isupper())
    ]
    if bad or len(set(drivers)) != len(drivers):
        problems.append(f"{label}: bad or repeated driver codes {sorted(map(str, bad))}")
    return problems


def payload_problems(state: dict[str, Any], model_version: str) -> list[str]:
    """Return what breaks the rollback trigger in one served payload."""
    problems = []
    version = (state.get("metadata") or {}).get("model_version")
    if version != model_version:
        problems.append(f"model version {version}, expected {model_version}")
    results = state.get("prediction_results") or {}
    race = results.get("race") or {}
    qualifying = results.get("qualifying") or {}
    finish = race.get("finish_order") or []
    if not finish:
        return [*problems, "no race finish order"]
    problems += _order_problems("race finish order", finish)
    field = {row.get("driver") for row in finish}
    for label, rows in (
        ("starting grid", race.get("starting_grid")),
        ("qualifying grid", qualifying.get("grid")),
    ):
        if rows:
            problems += _order_problems(label, rows)
            if {row.get("driver") for row in rows} != field:
                problems.append(f"{label}: drivers differ from the race finish order")

    win = [row.get("win_probability") for row in finish]
    podium = [row.get("podium_probability") for row in finish]
    if any(not isinstance(v, int | float) or not 0 <= v <= 100 for v in win + podium):
        problems.append("win or podium chance missing or outside 0 to 100")
    else:
        if any(p < w for w, p in zip(win, podium, strict=True)):
            problems.append("a podium chance is below the same driver's win chance")
        slack = 0.5 * len(finish)
        if abs(sum(win) - 100) > slack:
            problems.append(f"win chances sum to {sum(win):g}, not 100")
        if abs(sum(podium) - 300) > slack:
            problems.append(f"podium chances sum to {sum(podium):g}, not 300")

    # shortcut: the ridge needs 4 earlier events, so rounds 1 to 4 of a season fail here by design.
    grid = qualifying.get("grid") or []
    if grid and not qualifying.get("result_mode"):
        if qualifying.get("qualifying_order_source") != "qualifying_ridge":
            problems.append("qualifying grid not served by the ridge")
        elif any("champion_position" not in row for row in grid):
            problems.append("ridge grid rows lack champion_position")
    return problems


def accuracy_problems(snapshots: dict[str, dict[str, Any]]) -> list[str]:
    """Fail a target whose correlation is below 0 at every scored checkpoint.

    ``snapshots`` maps ``year::race::checkpoint::target`` to the snapshot data.
    """
    by_target: dict[str, list[float]] = defaultdict(list)
    for key, data in snapshots.items():
        correlation = (data.get("metrics") or {}).get("correlation")
        if correlation is not None:
            by_target[key.split("::")[3]].append(float(correlation))
    return [
        f"{target}: correlation below 0 at all {len(values)} checkpoints"
        for target, values in sorted(by_target.items())
        if values and max(values) < 0
    ]


def _served_payloads(client: Any, year: int, race: str) -> list[dict[str, Any]]:
    """The newest payload per checkpoint and weather scenario; older ones are not served."""
    index = (
        client.table("runtime_state")
        .select(
            "state_key,updated_at,boundary:state->metadata->>boundary_session_name,weather:state->>weather"
        )
        .eq("namespace", _NAMESPACE)
        .eq("state->>race_name", race)
        .eq("state->>year", str(year))
        .order("updated_at", desc=True)
        .limit(1000)
        .execute()
        .data
    )
    newest: dict[tuple[str, str], str] = {}
    for row in index:
        newest.setdefault((str(row["boundary"]), str(row["weather"])), row["state_key"])
    if not newest:
        return []
    rows = (
        client.table("runtime_state")
        .select("state_key,updated_at,state")
        .eq("namespace", _NAMESPACE)
        .in_("state_key", list(newest.values()))
        .execute()
        .data
    )
    return sorted(rows, key=lambda r: r["updated_at"])


def _accuracy_snapshots(client: Any, year: int, race: str) -> dict[str, dict[str, Any]]:
    """The newest accuracy snapshot per checkpoint and target."""
    rows = (
        client.table("artifacts")
        .select("artifact_key,version,data")
        .eq("artifact_type", "accuracy_snapshot")
        .like("artifact_key", f"{year}::{race}::%")
        .order("version", desc=False)
        .execute()
        .data
    )
    return {row["artifact_key"]: row["data"] for row in rows}


def main() -> int:
    """Print one line per served payload and accuracy result; return 1 on any problem."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--race", required=True, help='Event name, e.g. "Qatar Grand Prix"')
    parser.add_argument("--year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    if args.env_file is not None:
        from src.utils.env_file import load_env_file

        load_env_file(args.env_file)
    from src.persistence.db import get_supabase_client

    client = get_supabase_client()
    failed = False
    payloads = _served_payloads(client, args.year, args.race)
    if not payloads:
        print(f"FAIL {args.race}: no served payload")
        return 1
    for row in payloads:
        state = row["state"]
        label = f"{(state.get('metadata') or {}).get('boundary_session_name')} {state.get('weather')} {row['updated_at'][:16]}"
        problems = payload_problems(state, args.model_version)
        failed = failed or bool(problems)
        print(f"{'FAIL' if problems else 'ok'} {label}" + "".join(f"\n  {p}" for p in problems))
    snapshots = _accuracy_snapshots(client, args.year, args.race)
    problems = accuracy_problems(snapshots)
    failed = failed or bool(problems)
    print(
        f"{'FAIL' if problems else 'ok'} accuracy: {len(snapshots)} snapshots"
        + "".join(f"\n  {p}" for p in problems)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
