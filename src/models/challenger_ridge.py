"""Challenger ``qualifying_ridge``: the ML probe's stacked qualifying ridge, run live.

Ledger 2026-10-08: a ridge on seven features, one of them the champion's own forecast,
scored corr 0.935 vs 0.930 and MAE 1.64 vs 1.76 on rounds 5 to 14 (``unresolvable``).
Here it forecasts main qualifying at every checkpoint, refitted each time on earlier
events only:

- training rows: each earlier event's forecast saved at its last checkpoint before
  qualifying (FP3, or SQ on a sprint weekend), with that event's actual qualifying;
- features: the champion's position, recent qualifying form of driver and team, the
  driver's gap to his teammate, the team's measured qualifying gap in seconds, and this
  weekend's practice pace (team gap and driver delta) from the sessions run so far.

Needs at least ``MIN_TRAINING_EVENTS`` earlier events; before that it forecasts nothing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models import ml_challenger as ml
from src.utils.car_snapshot_history import snapshot_artifact_key

logger = logging.getLogger(__name__)

TARGET = "main_qualifying"
FEATURES = (
    "champion_q",
    "drv_q_mean3",
    "team_q_mean3",
    "team_quali_gap_s",
    "fp_short_gap_pct",
    "fp_driver_short_s",
    "drv_vs_mate_q",
)
MIN_TRAINING_EVENTS = 4
# Practice sessions a qualifying forecast may see, by checkpoint. Sprint and qualifying
# results never enter the practice features.
_VISIBLE_SESSIONS = {
    "PRE": (),
    "FP1": ("FP1",),
    "FP2": ("FP1", "FP2"),
    "FP3": ("FP1", "FP2", "FP3"),
    "SQ": ("FP1", "SQ"),
    "SPRINT": ("FP1", "SQ"),
}
_LAST_PRE_QUALIFYING = {"normal": "FP3", "sprint": "SQ"}


def _order(rows: Sequence[dict[str, Any]] | None) -> dict[str, int]:
    return {
        str(row["driver"]): int(row["position"])
        for row in rows or []
        if row.get("driver") and row.get("position") is not None
    }


def _teams(rows: Sequence[dict[str, Any]] | None) -> dict[str, str]:
    return {str(row["driver"]): str(row.get("team", "")) for row in rows or [] if row.get("driver")}


class _Snapshots:
    """Practice snapshot payloads from the checkpoint predictor's artifact store."""

    def __init__(self, predictor: Any, year: int) -> None:
        self.store = getattr(predictor, "artifact_store", None)
        self.year = year
        self._cache: dict[tuple[str, str], dict[str, Any] | None] = {}

    def load(self, race: str, sessions: Sequence[str]) -> list[dict[str, Any]]:
        payloads = []
        for session in sessions:
            key = (race, session)
            if key not in self._cache:
                payload = None
                if self.store is not None:
                    try:
                        payload = self.store.load_artifact(
                            "car_characteristics_snapshot",
                            snapshot_artifact_key(self.year, race, session),
                        )
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                        logger.debug("No snapshot for %s %s: %s", race, session, exc)
                self._cache[key] = payload if isinstance(payload, dict) else None
            cached = self._cache[key]
            if cached is not None:
                payloads.append(cached)
        return payloads


def _quali_gaps(predictor: Any, year: int) -> dict[str, dict[str, float]]:
    """Measured qualifying gap per team per race, from the car track traits file."""
    roots = [Path(getattr(predictor, "data_dir", "") or ""), Path("data/processed")]
    for root in roots:
        path = root / "car_track_traits" / f"{year}_car_track_traits.json"
        if path.is_file():
            races = json.loads(path.read_text(encoding="utf-8")).get("races", {})
            return {
                race: data["quali_gaps"] for race, data in races.items() if data.get("quali_gaps")
            }
    return {}


def _feature_rows(
    *,
    race: str,
    champion_rows: Sequence[dict[str, Any]],
    prior: Sequence[tuple[str, dict[str, int], dict[str, str]]],
    practice: list[dict[str, Any]],
    quali_gaps: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """One row per driver for one event, from earlier events and visible practice only."""
    actual = {name: order for name, order, _ in prior}
    team_of = {name: teams for name, _, teams in prior}
    names = [name for name, _, _ in prior]
    last3 = names[-3:]
    team_practice = ml.practice_features_from_payloads(practice)
    driver_practice = ml.driver_practice_deltas_from_payloads(practice)
    team_gap = ml._centred_mean(quali_gaps, [name for name in names if name in quali_gaps])
    champion = _order(champion_rows)
    current_team = _teams(champion_rows)
    rows = []
    for driver, team in current_team.items():
        mates = [d for d, t in current_team.items() if t == team and d != driver]
        team_drivers = [d for d, t in current_team.items() if t == team]
        mate_gap = [
            actual[r][driver] - actual[r][mate]
            for r in names
            for mate in mates
            if driver in actual[r] and mate in actual[r] and team_of[r].get(mate) == team
        ]
        rows.append(
            {
                "race": race,
                "driver": driver,
                "team": team,
                "champion_q": champion.get(driver, np.nan),
                "drv_q_mean3": ml._mean_position(actual, [driver], last3),
                "team_q_mean3": ml._mean_position(actual, team_drivers, last3),
                "drv_vs_mate_q": float(np.mean(mate_gap)) if mate_gap else np.nan,
                "team_quali_gap_s": team_gap.get(team, np.nan),
                "fp_short_gap_pct": (
                    team_practice.at[team, "fp_short_gap_pct"]
                    if team in team_practice.index and "fp_short_gap_pct" in team_practice
                    else np.nan
                ),
                "fp_driver_short_s": (
                    driver_practice.at[driver, "fp_driver_short_s"]
                    if driver in driver_practice.index and "fp_driver_short_s" in driver_practice
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _training_events(ctx: Any) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    """Earlier events as (race, last pre-qualifying forecast, actual qualifying), in order."""
    events = []
    seen: set[str] = set()
    for prediction in ctx.prior_predictions():
        metadata = prediction.get("metadata", {})
        race = str(metadata.get("race_name", ""))
        fmt = str(metadata.get("weekend_format", "normal"))
        if race in seen or str(
            metadata.get("session_name", "")
        ).upper() != _LAST_PRE_QUALIFYING.get(fmt, "FP3"):
            continue
        actual = ((prediction.get("actuals") or {}).get("targets") or {}).get(TARGET)
        if actual:
            seen.add(race)
            events.append((race, prediction, actual))
    return events


def qualifying_ridge(ctx: Any) -> dict[str, list[dict[str, Any]]]:
    """Forecast main qualifying with a ridge refitted on earlier events."""
    champion_rows = ctx.champion_targets.get(TARGET)
    if not champion_rows:
        return {}
    events = _training_events(ctx)
    if len(events) < MIN_TRAINING_EVENTS:
        return {}
    snapshots = _Snapshots(ctx.predictor, ctx.year)
    quali_gaps = _quali_gaps(ctx.predictor, ctx.year)

    frames: list[pd.DataFrame] = []
    prior: list[tuple[str, dict[str, int], dict[str, str]]] = []
    for race, prediction, actual_rows in events:
        rows = (prediction.get("targets") or {}).get(TARGET, {}).get("predicted_order") or []
        fmt = str(prediction.get("metadata", {}).get("weekend_format", "normal"))
        checkpoint = _LAST_PRE_QUALIFYING.get(fmt, "FP3")
        frame = _feature_rows(
            race=race,
            champion_rows=rows,
            prior=prior,
            practice=snapshots.load(race, _VISIBLE_SESSIONS[checkpoint]),
            quali_gaps=quali_gaps,
        )
        actual = _order(actual_rows)
        frame["y"] = frame["driver"].map(actual)
        frames.append(frame)
        prior.append((race, actual, _teams(actual_rows)))

    train = pd.concat(frames, ignore_index=True).dropna(subset=["y"])
    current = _feature_rows(
        race=ctx.race_name,
        champion_rows=champion_rows,
        prior=prior,
        practice=snapshots.load(ctx.race_name, _VISIBLE_SESSIONS.get(ctx.checkpoint_session, ())),
        quali_gaps=quali_gaps,
    )
    model = ml.ridge_model().fit(train[list(FEATURES)], train["y"])
    current["score"] = model.predict(current[list(FEATURES)])
    current = current.sort_values(["score", "champion_q", "driver"])
    by_driver = {str(row["driver"]): row for row in champion_rows}
    return {
        TARGET: [
            {
                "driver": driver,
                "team": by_driver[driver].get("team"),
                "position": position,
                "champion_position": int(by_driver[driver].get("position") or 0),
                "score": round(float(score), 4),
            }
            for position, (driver, score) in enumerate(
                zip(current["driver"], current["score"], strict=True), start=1
            )
        ]
    }
