"""Measured per-team race pace from green-flag laps, kept fresh after every race.

Method, per race: keep green-flag laps with a lap time and no pit in or out; take
each driver's median (drivers with fewer than 10 such laps are dropped), then each
team's median across its drivers; subtract the fastest team's value, giving a gap in
seconds (0.0 for the fastest team).

The artifact holds per-race gaps under ``races`` so a forecast averages only the
races before its target. It lives in ``ArtifactStore`` (``team_race_pace``,
``<year>::team_race_pace``), which in file mode is the committed
``data/processed/team_race_pace/<year>_team_race_pace.json``.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fastf1

from src.persistence.artifact_store import ArtifactStore
from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)

ARTIFACT_TYPE = "team_race_pace"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MIN_LAPS_PER_DRIVER = 10
_ROUND_PRECISION = 4
# Loading a session can fail for reasons outside this module (missing cache entry,
# laps not published yet); skip that race and retry on the next refresh.
_LOAD_ERRORS = (
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def artifact_key(year: int) -> str:
    """Return the store key for one season's team race pace."""
    return f"{int(year)}::{ARTIFACT_TYPE}"


def committed_path(year: int) -> Path:
    """Return the committed file that seeds the store and backs file mode."""
    return _PROJECT_ROOT / "data" / "processed" / "team_race_pace" / f"{year}_team_race_pace.json"


def measure_race(year: int, race_name: str) -> dict[str, float] | None:
    """Return one race's per-team gap to the fastest team in seconds, or None."""
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load(laps=True, telemetry=False, weather=False)
    except _LOAD_ERRORS as exc:
        logger.info("Skipping %s: failed to load session (%s)", race_name, exc)
        return None

    laps = getattr(session, "laps", None)
    if laps is None or laps.empty:
        return None

    green_laps = laps[
        (laps["TrackStatus"] == "1")
        & laps["LapTime"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
    ]
    if green_laps.empty:
        return None

    team_driver_medians: dict[str, list[float]] = defaultdict(list)
    for _driver, driver_laps in green_laps.groupby("Driver"):
        if len(driver_laps) < _MIN_LAPS_PER_DRIVER:
            continue
        raw_team = str(driver_laps["Team"].iloc[0])
        team = map_team_to_characteristics(raw_team) or raw_team
        team_driver_medians[team].append(driver_laps["LapTime"].median().total_seconds())

    if not team_driver_medians:
        return None

    team_medians = {team: statistics.median(values) for team, values in team_driver_medians.items()}
    fastest = min(team_medians.values())
    return {team: round(value - fastest, _ROUND_PRECISION) for team, value in team_medians.items()}


def aggregate_by_team(
    measurements: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    """Average each team's per-race gaps across every race it appears in."""
    gaps_by_team: dict[str, list[float]] = defaultdict(list)
    for race_gaps in measurements.values():
        for team, gap_s in race_gaps.items():
            gaps_by_team[team].append(gap_s)
    return {
        team: {"gap_s": round(statistics.mean(gaps), _ROUND_PRECISION), "races": len(gaps)}
        for team, gaps in gaps_by_team.items()
    }


def build_payload(year: int, measurements: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Return the artifact payload for a season's per-race measurements."""
    return {
        "year": int(year),
        "last_updated": datetime.now(UTC).isoformat(),
        "races": measurements,
        "teams": aggregate_by_team(measurements),
    }


def load_team_race_pace(year: int, store: ArtifactStore | None = None) -> dict[str, Any] | None:
    """Return the season's artifact from the store, falling back to the committed file."""
    try:
        payload = (store or ArtifactStore(data_root="data")).load_artifact(
            ARTIFACT_TYPE, artifact_key(year)
        )
    except Exception as exc:  # store outage must not stop a forecast
        logger.warning("Could not load %s from the store: %s", ARTIFACT_TYPE, exc)
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("races"), dict):
        return payload
    try:
        with open(committed_path(year)) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def refresh_team_race_pace(
    year: int,
    completed_races: list[str],
    store: ArtifactStore | None = None,
) -> list[str]:
    """Measure every completed race missing from the artifact and save; return what was added.

    Idempotent: a race already in the artifact is never re-measured, and a race whose
    laps cannot load yet is left for the next refresh.
    """
    store = store or ArtifactStore(data_root="data")
    payload = load_team_race_pace(year, store=store) or {}
    measurements: dict[str, dict[str, float]] = dict(payload.get("races") or {})

    added: list[str] = []
    for race_name in completed_races:
        if race_name in measurements:
            continue
        gaps = measure_race(year, race_name)
        if gaps:
            measurements[race_name] = gaps
            added.append(race_name)

    if added:
        store.save_artifact(ARTIFACT_TYPE, artifact_key(year), build_payload(year, measurements))
        logger.info("Team race pace updated for %s: %s", year, added)
    return added
