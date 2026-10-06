"""Per-team race pace adjustment from car traits x track composition.

Each team's measured traits (top speed, slow/medium/fast corner apex speed, braking,
tyre degradation; see ``src/extractors/car_track_traits.py``) are averaged over the
races before the target and standardised across teams. Each trait pairs with one
track share (full throttle, corner class, braking, degradation severity), taken
relative to the average of those earlier tracks. Ridge coefficients are fitted on
earlier races only: target = how much a team beat its own earlier-race average pace
at that track. The fitted model then predicts the target track's adjustment.

Walk-forward sizing on 2026 races 5 to 15 (MODEL_LEDGER 2026-10-03): R2 0.113 against
a shuffled-track null of 0.048; top speed carries most of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

TRAIT_TO_SHARE: dict[str, str] = {
    "top_speed": "full_throttle",
    "slow": "slow",
    "medium": "medium",
    "fast": "fast",
    "braking": "braking",
    "deg": "deg_severity",
}
# Below this many earlier races the coefficients are noise; return no adjustment.
MIN_PRIOR_RACES = 4
RIDGE_LAMBDA = 1.0


def _standardised_traits(traits_by_race: Mapping[str, Any], races: Sequence[str]) -> pd.DataFrame:
    """Return each team's mean trait over ``races``, standardised across teams."""
    frames = [pd.DataFrame(traits_by_race[r]["traits"]).T for r in races]
    traits = pd.concat(frames).groupby(level=0).mean().reindex(columns=list(TRAIT_TO_SHARE))
    return (traits - traits.mean()) / traits.std(ddof=1).replace(0.0, np.nan)


def _features(
    traits: pd.DataFrame, profile: Mapping[str, float], mean_profile: pd.Series
) -> pd.DataFrame:
    """Return trait x (track share - mean share) per team, one column per trait."""

    def deviation(share: str) -> float:
        # A share that could not be measured counts as average: no effect.
        value = profile.get(share)
        return 0.0 if value is None else float(value) - float(mean_profile[share])

    return pd.DataFrame(
        {trait: traits[trait] * deviation(share) for trait, share in TRAIT_TO_SHARE.items()}
    ).fillna(0.0)


def fit_and_predict(
    traits_by_race: Mapping[str, Any],
    race_pace_gaps: Mapping[str, Mapping[str, float]],
    prior_races: Sequence[str],
    target_profile: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return seconds per lap each team gains at the target track (positive = faster).

    ``prior_races`` are the races before the target, in calendar order, that have both
    traits and measured pace. Returns {} when there are too few or no target profile.
    """
    races = [r for r in prior_races if r in traits_by_race and r in race_pace_gaps]
    if target_profile is None or len(races) < MIN_PRIOR_RACES:
        return {}

    rows_x: list[pd.DataFrame] = []
    rows_y: list[pd.Series] = []
    for k in range(1, len(races)):
        earlier, race = races[:k], races[k]
        traits = _standardised_traits(traits_by_race, earlier)
        mean_profile = pd.DataFrame([traits_by_race[r]["profile"] for r in earlier]).mean()
        gaps = pd.Series(race_pace_gaps[race], dtype=float)
        base = pd.DataFrame([race_pace_gaps[r] for r in earlier]).mean()
        dev = -(gaps - base.reindex(gaps.index))
        x = _features(traits, traits_by_race[race]["profile"], mean_profile)
        common = x.index.intersection(dev.dropna().index)
        rows_x.append(x.loc[common])
        rows_y.append(dev.loc[common] - dev.loc[common].mean())

    x_train = pd.concat(rows_x).to_numpy(dtype=float)
    y_train = pd.concat(rows_y).to_numpy(dtype=float)
    coef = np.linalg.solve(
        x_train.T @ x_train + RIDGE_LAMBDA * np.eye(x_train.shape[1]), x_train.T @ y_train
    )

    traits = _standardised_traits(traits_by_race, races)
    mean_profile = pd.DataFrame([traits_by_race[r]["profile"] for r in races]).mean()
    adjustment = _features(traits, target_profile, mean_profile) @ coef
    adjustment -= adjustment.mean()
    return {str(team): float(value) for team, value in adjustment.items()}


@lru_cache(maxsize=64)
def track_trait_adjustments(
    year: int, race_name: str | None, session_kind: str
) -> dict[str, float]:
    """Return seconds per lap each team gains at ``race_name`` for ``race`` or ``qualifying``.

    Fits on races the schedule places before the target only. Race uses measured race
    pace; qualifying uses each race's team qualifying gaps. Empty when the target is
    unknown, has no track profile yet, or too few races came before it.
    """
    from src.extractors.car_track_traits import load_car_track_traits
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
    if session_kind == "qualifying":
        gaps = {r: v["quali_gaps"] for r, v in data["races"].items() if v.get("quali_gaps")}
    else:
        from src.extractors.team_race_pace import load_team_race_pace

        gaps = (load_team_race_pace(year) or {}).get("races") or {}
    profile = (data["races"].get(target) or {}).get("profile") or data["profiles"].get(target)
    return fit_and_predict(data["races"], gaps, schedule[: schedule.index(target)], profile)
