"""Challenger ``qualifying_laptime_stacked``: the lap-time model plus the champion's order.

``qualifying_laptime`` orders the field well before practice and loses after it: the
champion's practice signal is its practice-updated car state, which no single snapshot
feature carries (ledger 2026-10-18). This variant adds the champion's checkpoint order as
two inputs, team (mean predicted position, centred) and driver (gap to his teammate in
that order), each with its own weight per checkpoint. At PRE it is the lap-time model.

Training rows: every earlier event at every checkpoint it saved, with the champion's
forecast from that checkpoint, so each checkpoint's weight is learned on forecasts of
that checkpoint. Everything else is as in ``challenger_laptime``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models import ml_challenger as ml
from src.models.challenger_laptime import (
    _DRIVER_HISTORY_SHRINK,
    _DRIVER_SHRINK,
    _TEAM_SHRINK,
    MIN_TRAINING_EVENTS,
    TARGET,
    _best_laps,
    _centred,
    _shrunk_fit,
    _team_frame,
    _teammate_deltas,
    _track_fit,
)
from src.models.challenger_ridge import (
    _LAST_PRE_QUALIFYING,
    _VISIBLE_SESSIONS,
    _Snapshots,
    _teams,
    _training_events,
)

LEVELS = ("FP1", "FP2", "FP3", "SQ")
_CHECKPOINT_LEVEL = {"FP1": "FP1", "FP2": "FP2", "FP3": "FP3", "SQ": "SQ", "SPRINT": "SQ"}


def _champion_parts(rows: list[dict[str, Any]]) -> tuple[pd.Series, dict[str, float]]:
    """Team mean predicted position (centred) and each driver's gap to that mean."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.Series(dtype=float), {}
    frame["position"] = frame["position"].astype(float)
    team_mean = frame.groupby("team")["position"].mean()
    driver = {
        str(row["driver"]): float(row["position"]) - float(team_mean[row["team"]])
        for row in rows
        if row.get("team") in team_mean.index
    }
    return team_mean - team_mean.mean(), driver


def _level_values(value: float, level: str | None) -> list[float]:
    return [value if level == name else 0.0 for name in LEVELS]


def _champion_by_checkpoint(ctx: Any) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Earlier events' champion main qualifying forecasts, by race and checkpoint."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for prediction in ctx.prior_predictions():
        metadata = prediction.get("metadata", {})
        target = (prediction.get("targets") or {}).get(TARGET) or {}
        if target.get("result_mode", "PREDICTED") != "PREDICTED":
            continue
        rows = target.get("predicted_order") or []
        level = _CHECKPOINT_LEVEL.get(str(metadata.get("session_name", "")).upper())
        if rows and level:
            out.setdefault(str(metadata.get("race_name", "")), {})[level] = rows
    return out


def qualifying_laptime_stacked(ctx: Any) -> dict[str, list[dict[str, Any]]]:
    """Forecast main qualifying from lap-time parts plus the champion's checkpoint order."""
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
    champion_history = _champion_by_checkpoint(ctx)

    team_x: list[list[float]] = []
    team_y: list[float] = []
    driver_x: list[list[float]] = []
    driver_y: list[float] = []
    seen_gaps: list[dict[str, float]] = []
    spreads: list[float] = []
    driver_history: dict[str, list[float]] = {}
    for race, prediction, actual_rows in events:
        fmt = str(prediction.get("metadata", {}).get("weekend_format", "normal"))
        payloads = snapshots.load(race, _VISIBLE_SESSIONS[_LAST_PRE_QUALIFYING.get(fmt, "FP3")])
        practice = ml.practice_features_from_payloads(payloads)
        driver_practice = ml.driver_practice_deltas_from_payloads(payloads)
        fp = driver_practice.get("fp_driver_short_s", pd.Series(dtype=float))
        # One row per checkpoint the event saved; PRE carries no champion input.
        levels: dict[str | None, list[dict[str, Any]]] = {None: []}
        levels.update(champion_history.get(race, {}))
        if race in gaps and len(seen_gaps) >= 2:
            actual = _centred(gaps[race])
            spreads.append(float(actual.std()))
            # Each race in units of its own spread, so ordering close teams counts as much
            # as the gap from front to back.
            actual = actual / spreads[-1]
            car_pace = pd.DataFrame(seen_gaps).mean()
            car_pace = (car_pace - car_pace.mean()) / float(car_pace.std())
            base = _team_frame(list(actual.index), car_pace, _track_fit(ctx.year, race), practice)
            for level, rows in levels.items():
                champion_team, _ = _champion_parts(rows)
                for team in actual.index:
                    value = (
                        float(champion_team.get(team, 0.0)) if team in champion_team.index else 0.0
                    )
                    team_x.append(list(base.loc[team]) + _level_values(value, level))
                    team_y.append(float(actual[team]))
        laps = _best_laps(ctx.year, race)
        if laps:
            for driver, delta in _teammate_deltas(laps, _teams(actual_rows)).items():
                past = driver_history.get(driver, [])
                own = [
                    sum(past) / (len(past) + _DRIVER_HISTORY_SHRINK),
                    float(fp.get(driver, 0.0)) if driver in fp.index else 0.0,
                ]
                for level, rows in levels.items():
                    _, champion_driver = _champion_parts(rows)
                    driver_x.append(own + _level_values(champion_driver.get(driver, 0.0), level))
                    driver_y.append(delta)
                driver_history.setdefault(driver, []).append(delta)
        if race in gaps:
            seen_gaps.append(gaps[race])

    if not team_x or not seen_gaps:
        return {}
    team_prior = np.zeros(3 + len(LEVELS))
    team_prior[0] = 1.0
    team_coef = _shrunk_fit(np.array(team_x), np.array(team_y), team_prior, _TEAM_SHRINK)
    driver_prior = np.zeros(2 + len(LEVELS))
    driver_prior[0] = 1.0
    driver_coef = (
        _shrunk_fit(np.array(driver_x), np.array(driver_y), driver_prior, _DRIVER_SHRINK)
        if driver_x
        else driver_prior
    )

    level = _CHECKPOINT_LEVEL.get(ctx.checkpoint_session)
    payloads = snapshots.load(ctx.race_name, _VISIBLE_SESSIONS.get(ctx.checkpoint_session, ()))
    current_teams = _teams(champion_rows)
    team_names = sorted(set(current_teams.values()))
    car_pace = pd.DataFrame(seen_gaps).mean()
    car_pace = (car_pace - car_pace.mean()) / float(car_pace.std())
    base = _team_frame(
        team_names,
        car_pace,
        _track_fit(ctx.year, ctx.race_name),
        ml.practice_features_from_payloads(payloads),
    )
    champion_team, champion_driver = _champion_parts(champion_rows)
    spread = float(np.median(spreads)) if spreads else 1.0
    team_gap = {
        team: spread
        * float(
            np.array(
                list(base.loc[team])
                + _level_values(
                    float(champion_team.get(team, 0.0)) if team in champion_team.index else 0.0,
                    level,
                )
            )
            @ team_coef
        )
        for team in team_names
    }
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
        row = [
            sum(past) / (len(past) + _DRIVER_HISTORY_SHRINK),
            float(fp.get(driver, 0.0)) if driver in fp.index else 0.0,
        ] + _level_values(champion_driver.get(driver, 0.0), level)
        driver_pct = float(np.array(row) @ driver_coef)
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
