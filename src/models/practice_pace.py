"""Race pace adjustment from this weekend's practice.

Signal per team: its average gap over earlier races (race pace, or qualifying gap)
minus its best-lap gap in this weekend's practice sessions so far (positive = quicker
in practice than usual). A single slope, fitted on earlier races only, turns that into
seconds per lap. Qualifying walk-forward R2: 0.227 (FP1) to 0.293 (FP1 to FP3). Sizing (MODEL_LEDGER 2026-10-03): the signal predicts
the model's own race errors at FP checkpoints, corr +0.30 / +0.32.

Only the sessions a checkpoint may see are used: the replay passes its checkpoint
through the prediction context; live has only completed sessions stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")
MIN_PRIOR_RACES = 4
RIDGE_LAMBDA = 1.0


def _signal(
    fp_gaps: Mapping[str, Mapping[str, float]],
    sessions: Sequence[str],
    base: Mapping[str, float],
) -> dict[str, float]:
    """Return centred (prior race pace gap - mean practice gap) per team."""
    out = {}
    for team, base_gap in base.items():
        gaps = [fp_gaps[c][team] for c in sessions if c in fp_gaps and team in fp_gaps[c]]
        if gaps:
            out[team] = float(base_gap) - float(np.mean(gaps))
    if not out:
        return {}
    mean = float(np.mean(list(out.values())))
    return {team: value - mean for team, value in out.items()}


def _base(pace: Mapping[str, Mapping[str, float]], races: Sequence[str]) -> dict[str, float]:
    """Return each team's mean race pace gap over ``races``."""
    sums: dict[str, list[float]] = {}
    for race in races:
        for team, gap in pace.get(race, {}).items():
            sums.setdefault(team, []).append(float(gap))
    return {team: float(np.mean(values)) for team, values in sums.items()}


def fit_and_predict(
    fp_by_race: Mapping[str, Mapping[str, Mapping[str, float]]],
    pace: Mapping[str, Mapping[str, float]],
    prior_races: Sequence[str],
    target_fp: Mapping[str, Mapping[str, float]],
    sessions: Sequence[str],
) -> dict[str, float]:
    """Return race pace seconds per lap each team gains (positive = faster)."""
    races = [r for r in prior_races if r in pace]
    if len(races) < MIN_PRIOR_RACES or not sessions:
        return {}
    xs: list[float] = []
    ys: list[float] = []
    for k in range(1, len(races)):
        race = races[k]
        base = _base(pace, races[:k])
        signal = _signal(fp_by_race.get(race, {}), sessions, base)
        dev = {t: -(pace[race][t] - base[t]) for t in signal if t in pace[race]}
        if len(dev) < 3:
            continue
        mean_dev = float(np.mean(list(dev.values())))
        for team, d in dev.items():
            xs.append(signal[team])
            ys.append(d - mean_dev)
    if not xs:
        return {}
    x, y = np.array(xs), np.array(ys)
    slope = float(x @ y / (x @ x + RIDGE_LAMBDA))
    target_signal = _signal(target_fp, sessions, _base(pace, races))
    return {team: slope * value for team, value in target_signal.items()}


@lru_cache(maxsize=128)
def practice_adjustments(
    year: int, race_name: str | None, sessions: tuple[str, ...], kind: str = "race"
) -> dict[str, float]:
    """Return the practice-based pace adjustment for ``race_name`` using ``sessions``.

    ``kind`` "race" fits against measured race pace; "qualifying" against each race's
    team qualifying gaps.
    """
    from src.extractors.car_track_traits import load_car_track_traits
    from src.extractors.team_race_pace import load_team_race_pace
    from src.utils.weekend import get_schedule_rows

    target = str(race_name or "").strip()
    try:
        schedule = [
            str(name).strip()
            for name, event_format in get_schedule_rows(year)
            if "testing" not in f"{name} {event_format}".lower()
        ]
    except Exception:  # schedule unavailable: no adjustment rather than a guess
        return {}
    if target not in schedule:
        return {}
    data = load_car_track_traits(year)
    fp_by_race: dict[str, Any] = data["fp_gaps"]
    if kind == "qualifying":
        pace = {r: v["quali_gaps"] for r, v in data["races"].items() if v.get("quali_gaps")}
    else:
        pace = (load_team_race_pace(year) or {}).get("races") or {}
    return fit_and_predict(
        fp_by_race,
        pace,
        schedule[: schedule.index(target)],
        fp_by_race.get(target, {}),
        sessions,
    )


def sessions_for_active_checkpoint(year: int, race_name: str | None) -> tuple[str, ...]:
    """Return the practice sessions the active prediction may use for ``race_name``."""
    from src.extractors.car_track_traits import load_car_track_traits
    from src.utils.prediction_context import get_active_prediction_context

    context = get_active_prediction_context()
    stored = list(load_car_track_traits(year)["fp_gaps"].get(str(race_name or "").strip(), {}))
    return allowed_sessions(context.checkpoint_session if context else None, stored)


def allowed_sessions(checkpoint: str | None, stored: Sequence[str]) -> tuple[str, ...]:
    """Return the practice sessions a checkpoint may use; all stored ones when unknown."""
    if checkpoint is None:
        return tuple(c for c in PRACTICE_SESSIONS if c in stored)
    from src.utils.historical_replay import session_is_available_at_checkpoint

    return tuple(
        c
        for c in PRACTICE_SESSIONS
        if c in stored and session_is_available_at_checkpoint(checkpoint, c)
    )
