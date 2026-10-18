"""Challenger ``qualifying_laptime``: qualifying as a lap-time model in seconds.

Each driver's predicted qualifying gap to pole is

    team part:   car pace + track fit + this weekend's practice gap
    driver part: earlier gap to his teammate (shrunk) + this weekend's practice delta

with every weight fitted on earlier events only and shrunk toward a prior (car pace 1,
everything else 0). The forecast is the order of those gaps. Sizing that shaped it is
in the ledger, 2026-10-18: car pace explains 85% of team gaps, track fit and driver
form are real but small, and upgrades do not show as drift, so car pace is the season
mean.

Sources, all from before the forecast: team qualifying gaps from ``car_track_traits``
(store first); the track fit the champion uses (``track_trait_adjustments``); practice
snapshots for sessions up to the checkpoint; earlier events' best qualifying laps from
OpenF1 for the driver part (without OpenF1 the driver part is practice only).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models import ml_challenger as ml
from src.models.challenger_ridge import (
    _LAST_PRE_QUALIFYING,
    _VISIBLE_SESSIONS,
    _Snapshots,
    _teams,
    _training_events,
)

logger = logging.getLogger(__name__)

TARGET = "main_qualifying"
MIN_TRAINING_EVENTS = 4
# Ridge strength on the deviation from the prior coefficients, in units of training rows.
_TEAM_SHRINK = 20.0
_DRIVER_SHRINK = 20.0
# Pseudo-events of zero teammate gap added to a driver's history.
_DRIVER_HISTORY_SHRINK = 2.0
_best_lap_cache: dict[tuple[int, str], dict[str, float] | None] = {}


def _shrunk_fit(x: np.ndarray, y: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    """Least squares with a ridge penalty pulling the coefficients toward ``prior``."""
    gram = x.T @ x + strength * np.eye(x.shape[1])
    return np.asarray(np.linalg.solve(gram, x.T @ y + strength * prior))


def _centred(values: dict[str, float]) -> pd.Series:
    series = pd.Series(values, dtype=float)
    return series - series.mean()


def _best_laps(year: int, race: str) -> dict[str, float] | None:
    """Best qualifying lap per driver for a finished earlier event, cached per process."""
    key = (year, race)
    if key not in _best_lap_cache:
        from src.data import openf1

        try:
            _best_lap_cache[key] = openf1.fetch_qualifying_best_laps(year, race) or None
        except openf1.OpenF1Unavailable as exc:
            logger.debug("No OpenF1 qualifying laps for %s: %s", race, exc)
            _best_lap_cache[key] = None
    return _best_lap_cache[key]


def _teammate_deltas(laps: dict[str, float], teams: dict[str, str]) -> dict[str, float]:
    """Driver lap minus his team's mean, percent of pole; needs both drivers' laps."""
    pole = min(laps.values())
    by_team: dict[str, list[str]] = {}
    for driver, team in teams.items():
        if driver in laps:
            by_team.setdefault(team, []).append(driver)
    deltas = {}
    for drivers in by_team.values():
        if len(drivers) != 2:
            continue
        mean = sum(laps[d] for d in drivers) / 2
        for driver in drivers:
            delta = (laps[driver] - mean) / pole * 100.0
            if abs(delta) < 2.0:  # beyond this it is a crash or a deleted lap
                deltas[driver] = delta
    return deltas


def _track_fit(year: int, race: str) -> dict[str, float]:
    """Seconds lost per lap from the champion's traits-by-track fit (positive = slower)."""
    from src.models.track_traits import track_trait_adjustments

    try:
        gained = track_trait_adjustments(year, race, "qualifying")
    except Exception as exc:  # noqa: BLE001 - a missing fit is zero, never a failure
        logger.debug("No track fit for %s: %s", race, exc)
        return {}
    return {team: -value for team, value in gained.items()}


def _team_frame(
    teams: list[str],
    car_pace: pd.Series,
    track: dict[str, float],
    practice: pd.DataFrame,
) -> pd.DataFrame:
    gap = practice["fp_short_gap_pct"] if "fp_short_gap_pct" in practice else pd.Series(dtype=float)
    gap = gap - gap.mean() if not gap.empty else gap
    return pd.DataFrame(
        {
            "car_pace": [float(car_pace.get(team, 0.0)) for team in teams],
            "track": [float(track.get(team, 0.0)) for team in teams],
            "practice": [float(gap.get(team, 0.0)) if team in gap.index else 0.0 for team in teams],
        },
        index=teams,
    ).fillna(0.0)


def qualifying_laptime(ctx: Any) -> dict[str, list[dict[str, Any]]]:
    """Forecast main qualifying as the order of predicted lap-time gaps."""
    from src.extractors.car_track_traits import load_car_track_traits

    champion_rows = ctx.champion_targets.get(TARGET)
    if not champion_rows:
        return {}
    events = _training_events(ctx)
    if len(events) < MIN_TRAINING_EVENTS:
        return {}
    store = getattr(ctx.predictor, "artifact_store", None)
    races = load_car_track_traits(ctx.year, store=store)["races"]
    gaps = {race: data["quali_gaps"] for race, data in races.items() if data.get("quali_gaps")}
    snapshots = _Snapshots(ctx.predictor, ctx.year)

    team_x, team_y, driver_x, driver_y = [], [], [], []
    seen_gaps: list[dict[str, float]] = []
    driver_history: dict[str, list[float]] = {}
    for race, prediction, actual_rows in events:
        fmt = str(prediction.get("metadata", {}).get("weekend_format", "normal"))
        # Tried 2026-10-18: training on only the sessions the checkpoint can see was worse
        # (corr 0.885 vs 0.886); the practice weight is learned on full pre-Q practice.
        payloads = snapshots.load(race, _VISIBLE_SESSIONS[_LAST_PRE_QUALIFYING.get(fmt, "FP3")])
        practice = ml.practice_features_from_payloads(payloads)
        driver_practice = ml.driver_practice_deltas_from_payloads(payloads)
        if race in gaps and len(seen_gaps) >= 2:
            actual = _centred(gaps[race])
            car_pace = pd.DataFrame(seen_gaps).mean()
            car_pace = car_pace - car_pace.mean()
            frame = _team_frame(list(actual.index), car_pace, _track_fit(ctx.year, race), practice)
            team_x.append(frame.to_numpy())
            team_y.append(actual.to_numpy())
        teams = _teams(actual_rows)
        laps = _best_laps(ctx.year, race)
        if laps:
            for driver, delta in _teammate_deltas(laps, teams).items():
                past = driver_history.get(driver, [])
                fp = driver_practice.get("fp_driver_short_s", pd.Series(dtype=float))
                driver_x.append(
                    [
                        sum(past) / (len(past) + _DRIVER_HISTORY_SHRINK),
                        float(fp.get(driver, 0.0)) if driver in fp.index else 0.0,
                    ]
                )
                driver_y.append(delta)
                driver_history.setdefault(driver, []).append(delta)
        if race in gaps:
            seen_gaps.append(gaps[race])

    if not team_x or not seen_gaps:
        return {}
    team_coef = _shrunk_fit(
        np.vstack(team_x), np.concatenate(team_y), np.array([1.0, 0.0, 0.0]), _TEAM_SHRINK
    )
    driver_coef = (
        _shrunk_fit(np.array(driver_x), np.array(driver_y), np.array([1.0, 0.0]), _DRIVER_SHRINK)
        if driver_x
        else np.array([1.0, 0.0])
    )

    # Current event: earlier events and sessions up to the checkpoint only.
    current_teams = _teams(champion_rows)
    payloads = snapshots.load(ctx.race_name, _VISIBLE_SESSIONS.get(ctx.checkpoint_session, ()))
    car_pace = pd.DataFrame(seen_gaps).mean()
    car_pace = car_pace - car_pace.mean()
    team_names = sorted(set(current_teams.values()))
    frame = _team_frame(
        team_names,
        car_pace,
        _track_fit(ctx.year, ctx.race_name),
        ml.practice_features_from_payloads(payloads),
    )
    team_gap = dict(zip(team_names, frame.to_numpy() @ team_coef, strict=True))
    fp = ml.driver_practice_deltas_from_payloads(payloads).get(
        "fp_driver_short_s", pd.Series(dtype=float)
    )
    lap_seconds = float(
        np.median(
            [min(laps.values()) for race, _, _ in events if (laps := _best_laps(ctx.year, race))]
            or [90.0]
        )
    )
    champion_position = {str(row["driver"]): int(row.get("position") or 0) for row in champion_rows}
    scored = []
    for driver, team in current_teams.items():
        past = driver_history.get(driver, [])
        driver_pct = float(
            np.array(
                [
                    sum(past) / (len(past) + _DRIVER_HISTORY_SHRINK),
                    float(fp.get(driver, 0.0)) if driver in fp.index else 0.0,
                ]
            )
            @ driver_coef
        )
        gap = team_gap.get(team, 0.0) + driver_pct * lap_seconds / 100.0
        scored.append((gap, champion_position.get(driver, 99), driver, team))
    scored.sort()
    return {
        TARGET: [
            {
                "driver": driver,
                "team": team,
                "position": position,
                "champion_position": champion_position.get(driver),
                "predicted_gap_s": round(gap - scored[0][0], 3),
            }
            for position, (gap, _, driver, team) in enumerate(scored, start=1)
        ]
    }
