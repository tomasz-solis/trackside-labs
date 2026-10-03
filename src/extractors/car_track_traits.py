"""Measure car traits per team and the matching composition of each track.

Car traits, per team, relative to the field median of the same session:

- ``top_speed``: max speed on each driver's fastest qualifying lap (km/h).
- ``slow``, ``medium``, ``fast``: apex speed at each circuit corner, by the field
  median apex speed of that corner (< 140, 140 to 210, > 210 km/h).
- ``braking``: time from brake onset to apex into heavy-braking corners (seconds;
  negative is better).
- ``deg``: race lap-time loss per lap of tyre age, after removing the field's
  per-lap trend (fuel burn, track evolution); seconds per lap, negative is better.

Track composition, from one reference lap: share of lap time at full throttle, in
each corner class, and braking; plus the field's median degradation.

All functions take loaded FastF1 sessions, so callers control caching and leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SLOW_MAX_KPH = 140.0
FAST_MIN_KPH = 210.0
_APEX_WINDOW_M = 75.0
_BRAKE_LOOKBACK_M = 400.0
_HEAVY_BRAKING_DROP_KPH = 60.0
_FULL_THROTTLE = 98.0
_MIN_STINT_LAPS = 6


def _corner_class(apex_kph: float) -> str:
    """Return the corner class for a field-median apex speed."""
    if apex_kph < SLOW_MAX_KPH:
        return "slow"
    if apex_kph > FAST_MIN_KPH:
        return "fast"
    return "medium"


def _fastest_laps(session: Any) -> dict[str, Any]:
    """Return each driver's fastest timed lap."""
    laps = session.laps
    out = {}
    for driver in laps["Driver"].dropna().unique():
        lap = laps.pick_drivers(driver).pick_fastest()
        if lap is not None and not pd.isna(lap.get("LapTime")):
            out[str(driver)] = lap
    return out


def corner_distances(session: Any) -> np.ndarray:
    """Return corner apex distances from FastF1 circuit info, or from the speed trace.

    New circuits have no circuit info; then a corner is a speed minimum at least
    20 km/h below the maximum within 150 m either side on the session's fastest lap.
    """
    try:
        corners = session.get_circuit_info().corners
        return corners.drop_duplicates("Number")["Distance"].to_numpy(dtype=float)
    except Exception:  # FastF1 has no circuit info for this event
        pass
    tel = session.laps.pick_fastest().get_car_data().add_distance()
    dist = tel["Distance"].to_numpy(dtype=float)
    speed = tel["Speed"].to_numpy(dtype=float)
    apexes: list[float] = []
    for i in range(len(speed)):
        near = np.abs(dist - dist[i]) <= 150.0
        if speed[i] == speed[near].min() and speed[near].max() - speed[i] >= 20.0:
            if not apexes or dist[i] - apexes[-1] > 150.0:
                apexes.append(float(dist[i]))
    return np.array(apexes)


def _lap_corner_metrics(lap: Any, corner_distances: np.ndarray) -> dict[str, Any] | None:
    """Return top speed, apex speed per corner and braking time per corner for one lap."""
    try:
        tel = lap.get_car_data().add_distance()
    except Exception:  # missing telemetry for one lap must not stop the session
        return None
    if tel.empty:
        return None
    dist = tel["Distance"].to_numpy()
    speed = tel["Speed"].to_numpy(dtype=float)
    brake = tel["Brake"].to_numpy(dtype=bool)
    secs = tel["Time"].dt.total_seconds().to_numpy()
    apex, braking = [], []
    for d in corner_distances:
        window = np.abs(dist - d) <= _APEX_WINDOW_M
        if not window.any():
            apex.append(np.nan)
            braking.append(np.nan)
            continue
        i_apex = np.flatnonzero(window)[np.argmin(speed[window])]
        apex.append(speed[i_apex])
        approach = np.flatnonzero((dist >= d - _BRAKE_LOOKBACK_M) & (dist <= dist[i_apex]))
        if approach.size and speed[approach].max() - speed[i_apex] >= _HEAVY_BRAKING_DROP_KPH:
            braking_idx = approach[brake[approach]]
            braking.append(secs[i_apex] - secs[braking_idx[0]] if braking_idx.size else np.nan)
        else:
            braking.append(np.nan)
    return {"top_speed": float(np.nanmax(speed)), "apex": apex, "braking": braking}


def team_traits_from_qualifying(session: Any, team_of: dict[str, str]) -> pd.DataFrame:
    """Return per-team top speed, corner-class apex speed and braking, relative to the field.

    ``team_of`` maps driver code to team name. Rows are teams; values are team means of
    driver differences from the field median, per corner then averaged per class.
    """
    corner_d = corner_distances(session)
    per_driver = {
        drv: m
        for drv, lap in _fastest_laps(session).items()
        if drv in team_of and (m := _lap_corner_metrics(lap, corner_d)) is not None
    }
    if len(per_driver) < 6:
        return pd.DataFrame()

    drivers = list(per_driver)
    apex = np.array([per_driver[d]["apex"] for d in drivers], dtype=float)
    braking = np.array([per_driver[d]["braking"] for d in drivers], dtype=float)
    top = np.array([per_driver[d]["top_speed"] for d in drivers], dtype=float)
    field_apex = np.nanmedian(apex, axis=0)
    classes = np.array([_corner_class(v) for v in field_apex])

    rows = {}
    for i, drv in enumerate(drivers):
        row = {"top_speed": top[i] - np.nanmedian(top)}
        for cls in ("slow", "medium", "fast"):
            cols = classes == cls
            row[cls] = float(np.nanmean(apex[i, cols] - field_apex[cols])) if cols.any() else np.nan
        row["braking"] = float(np.nanmean(braking[i] - np.nanmedian(braking, axis=0)))
        rows[drv] = row
    frame = pd.DataFrame(rows).T
    frame["team"] = [team_of[d] for d in frame.index]
    return frame.groupby("team").mean()


def team_deg_from_race(session: Any, team_of: dict[str, str]) -> pd.Series:
    """Return per-team tyre degradation (s/lap per lap of tyre age) relative to the field."""
    laps = session.laps
    green = laps[
        (laps["TrackStatus"] == "1")
        & laps["LapTime"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
    ].copy()
    if green.empty:
        return pd.Series(dtype=float)
    green["t"] = green["LapTime"].dt.total_seconds()
    # Remove what every car shares lap by lap: fuel burn and track evolution.
    green["t"] -= green.groupby("LapNumber")["t"].transform("median")
    slopes: dict[str, list[float]] = {}
    for (drv, _stint), stint in green.groupby(["Driver", "Stint"]):
        stint = stint[stint["TyreLife"] >= 2]
        if len(stint) < _MIN_STINT_LAPS or drv not in team_of:
            continue
        x = stint["TyreLife"].to_numpy(dtype=float)
        y = stint["t"].to_numpy(dtype=float)
        keep = np.abs(y - np.median(y)) < 3.0  # drop traffic and mistakes
        if keep.sum() >= _MIN_STINT_LAPS:
            slopes.setdefault(team_of[str(drv)], []).append(
                float(np.polyfit(x[keep], y[keep], 1)[0])
            )
    team = pd.Series({t: float(np.median(v)) for t, v in slopes.items()})
    return team - team.median()


def team_quali_gaps(session: Any, team_of: dict[str, str]) -> dict[str, float]:
    """Return each team's best qualifying lap minus the fastest team's, in seconds."""
    laps = session.laps.dropna(subset=["LapTime", "Driver"])
    best: dict[str, float] = {}
    for drv, lap_time in zip(laps["Driver"], laps["LapTime"].dt.total_seconds(), strict=False):
        team = team_of.get(str(drv))
        if team is not None:
            best[team] = min(best.get(team, float("inf")), float(lap_time))
    if not best:
        return {}
    fastest = min(best.values())
    return {team: round(value - fastest, 4) for team, value in best.items()}


def track_profile(session: Any) -> dict[str, float]:
    """Return a track's lap composition from the session's fastest lap."""
    lap = session.laps.pick_fastest()
    tel = lap.get_car_data()
    dt = tel["Time"].dt.total_seconds().diff().fillna(0.0).to_numpy()
    speed = tel["Speed"].to_numpy(dtype=float)
    full = tel["Throttle"].to_numpy(dtype=float) >= _FULL_THROTTLE
    brake = tel["Brake"].to_numpy(dtype=bool)
    total = dt.sum()
    return {
        "full_throttle": float(dt[full].sum() / total),
        "slow": float(dt[~full & (speed < SLOW_MAX_KPH)].sum() / total),
        "medium": float(
            dt[~full & (speed >= SLOW_MAX_KPH) & (speed <= FAST_MIN_KPH)].sum() / total
        ),
        "fast": float(dt[~full & (speed > FAST_MIN_KPH)].sum() / total),
        "braking": float(dt[brake].sum() / total),
        "lap_time_s": float(lap["LapTime"].total_seconds()),
    }


def race_deg_severity(session: Any) -> float:
    """Return the field's median tyre degradation for a race (s/lap per lap)."""
    laps = session.laps
    green = laps[
        (laps["TrackStatus"] == "1")
        & laps["LapTime"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
    ].copy()
    green["t"] = green["LapTime"].dt.total_seconds()
    green["t"] -= green.groupby("LapNumber")["t"].transform("median")
    slopes = []
    for _, stint in green.groupby(["Driver", "Stint"]):
        stint = stint[stint["TyreLife"] >= 2]
        if len(stint) >= _MIN_STINT_LAPS:
            raw = stint["LapTime"].dt.total_seconds().to_numpy()
            slopes.append(float(np.polyfit(stint["TyreLife"].to_numpy(dtype=float), raw, 1)[0]))
    return float(np.median(slopes)) if slopes else float("nan")


# ---- Season artifact: store first, committed file fallback ----

ARTIFACT_TYPE = "car_track_traits"
_WEEKEND_SESSIONS = ("Q", "SQ", "FP3", "FP2", "FP1")


def artifact_key(year: int) -> str:
    """Return the store key for one season's traits and track profiles."""
    return f"{int(year)}::{ARTIFACT_TYPE}"


def committed_path(year: int) -> Path:
    """Return the committed file that seeds the store and backs file mode and the replay."""
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "processed" / "car_track_traits" / f"{year}_car_track_traits.json"


def load_car_track_traits(year: int, store: Any = None) -> dict[str, Any]:
    """Return ``{"races": {race: {traits, profile}}, "profiles": {race: profile}}``."""
    import json

    from src.persistence.artifact_store import ArtifactStore

    try:
        payload = (store or ArtifactStore(data_root="data")).load_artifact(
            ARTIFACT_TYPE, artifact_key(year)
        )
    except Exception:  # store outage must not stop a forecast
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("races"), dict):
        try:
            with open(committed_path(year)) as f:
                payload = json.load(f)
        except (OSError, ValueError):
            payload = {}
    return {
        "races": dict(payload.get("races") or {}),
        "profiles": dict(payload.get("profiles") or {}),
    }


def _team_of(session: Any) -> dict[str, str]:
    """Map driver code to characteristics team name for one session."""
    from src.utils.team_mapping import map_team_to_characteristics

    laps = session.laps.dropna(subset=["Driver", "Team"])
    return {
        str(d): (map_team_to_characteristics(str(t)) or str(t))
        for d, t in zip(laps["Driver"], laps["Team"], strict=False)
    }


def measure_completed_race(year: int, race_name: str) -> dict[str, Any] | None:
    """Return one completed race's team traits and track profile, or None if not loadable."""
    import fastf1

    try:
        quali = fastf1.get_session(year, race_name, "Q")
        quali.load(laps=True, telemetry=True, weather=False, messages=False)
        race = fastf1.get_session(year, race_name, "R")
        race.load(laps=True, telemetry=False, weather=False, messages=False)
        traits = team_traits_from_qualifying(quali, _team_of(quali))
        if traits.empty:
            return None
        traits["deg"] = team_deg_from_race(race, _team_of(race))
        profile = {**track_profile(quali), "deg_severity": race_deg_severity(race)}
        quali_gaps = team_quali_gaps(quali, _team_of(quali))
    except Exception as exc:  # laps not published yet or a malformed session
        import logging

        logging.getLogger(__name__).info("Skipping traits for %s: %s", race_name, exc)
        return None
    return {
        "traits": traits.round(4).to_dict(orient="index"),
        "profile": profile,
        "quali_gaps": quali_gaps,
    }


def measure_weekend_profile(year: int, race_name: str) -> dict[str, float] | None:
    """Return the track profile from this weekend's latest completed session, or None.

    Geometry, not a result, so any completed session of the weekend may be used.
    Degradation severity is unknown before the race; the season median stands in.
    """
    import fastf1

    for code in _WEEKEND_SESSIONS:
        try:
            session = fastf1.get_session(year, race_name, code)
            session.load(laps=True, telemetry=True, weather=False, messages=False)
            if session.laps.empty:
                continue
            return track_profile(session)
        except Exception:  # session missing on this weekend format or not run yet
            continue
    return None


def refresh_car_track_traits(
    year: int,
    completed_races: list[str],
    upcoming_race: str | None = None,
    store: Any = None,
) -> dict[str, list[str]]:
    """Measure completed races and the upcoming track the artifact lacks; save if changed."""
    from datetime import UTC, datetime

    from src.persistence.artifact_store import ArtifactStore

    store = store or ArtifactStore(data_root="data")
    payload = load_car_track_traits(year, store=store)
    added: dict[str, list[str]] = {"races": [], "profiles": []}
    for race_name in completed_races:
        if race_name not in payload["races"]:
            measured = measure_completed_race(year, race_name)
            if measured:
                payload["races"][race_name] = measured
                added["races"].append(race_name)
    if (
        upcoming_race
        and upcoming_race not in payload["races"]
        and upcoming_race not in payload["profiles"]
    ):
        profile = measure_weekend_profile(year, upcoming_race)
        if profile:
            severities = [r["profile"].get("deg_severity") for r in payload["races"].values()]
            severities = [s for s in severities if s is not None and s == s]
            profile["deg_severity"] = float(np.median(severities)) if severities else 0.0
            payload["profiles"][upcoming_race] = profile
            added["profiles"].append(upcoming_race)
    if added["races"] or added["profiles"]:
        store.save_artifact(
            ARTIFACT_TYPE,
            artifact_key(year),
            {"year": int(year), "last_updated": datetime.now(UTC).isoformat(), **payload},
        )
    return added
