"""ML challenger: learn qualifying and race order from pace data, walk-forward.

One row per driver per race, built only from what was known at the last checkpoint
before qualifying (normal weekend: after FP3; sprint weekend: after sprint qualifying):

- this weekend's practice: short and long run pace, corner times, top speed, tyre deg,
  teammate split (the car characteristics snapshots the team comparison tab shows);
- earlier races: quali and race positions, DNF rate, measured team race pace and
  quali gap, car traits x track;
- the track: overtaking difficulty, street circuit, sprint weekend, wet.

The champion's own forecast can be added as a feature (stacked variant). Models are
refitted before every race on earlier races only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LAST_PRE_Q_CHECKPOINT = {"normal": "fp3", "sprint": "sq"}
VISIBLE_SESSIONS = {"normal": ("fp1", "fp2", "fp3"), "sprint": ("fp1", "sq")}
_DNF_PRIOR_RATE = 0.15
_DNF_PRIOR_RACES = 2.0

PRACTICE_FEATURES = (
    "fp_short_gap_pct",
    "fp_long_gap_pct",
    "fp_slow_gap_pct",
    "fp_medium_gap_pct",
    "fp_fast_gap_pct",
    "fp_top_speed_kph",
    "fp_deg_slope",
    "fp_driver_short_s",
    "fp_driver_long_s",
)
HISTORY_FEATURES = (
    "drv_q_mean3",
    "drv_q_season",
    "team_q_mean3",
    "drv_vs_mate_q",
    "drv_r_mean3",
    "team_r_mean3",
    "drv_dnf_rate",
    "team_dnf_rate",
    "team_race_pace_s",
    "team_quali_gap_s",
    "traits_q_adj_s",
)
TRACK_FEATURES = ("overtaking_difficulty", "street", "sprint_weekend", "wet")
CHAMPION_FEATURES = ("champion_q", "champion_r")
PACE_FEATURES = PRACTICE_FEATURES + HISTORY_FEATURES + TRACK_FEATURES


def _load(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _slug(race: str) -> str:
    return race.lower().replace(" ", "_")


def _order(rows: Sequence[dict[str, Any]] | None) -> dict[str, int]:
    """Return driver -> position, from list order."""
    return {str(r["driver"]): i + 1 for i, r in enumerate(rows or [])}


def _gap_pct(values: dict[str, float]) -> dict[str, float]:
    best = min(values.values())
    return {team: (v / best - 1.0) * 100.0 for team, v in values.items()}


def practice_features(snapshot_dir: Path, sessions: Sequence[str]) -> pd.DataFrame:
    """Return team practice features averaged over the visible sessions (team x feature)."""
    frames = []
    for session in sessions:
        path = snapshot_dir / f"{session}.json"
        if not path.exists():
            continue
        teams = _load(path).get("teams") or {}

        def metric(profile: str, key: str, _teams: dict[str, Any] = teams) -> dict[str, float]:
            out = {}
            for team, data in _teams.items():
                value = ((data.get("profiles") or {}).get(profile) or {}).get(key)
                if isinstance(value, int | float) and np.isfinite(value) and value > 0:
                    out[team] = float(value)
            return out

        cols: dict[str, dict[str, float]] = {}
        for name, profile, key in (
            ("fp_short_gap_pct", "short_run", "overall_pace_seconds"),
            ("fp_long_gap_pct", "long_run", "overall_pace_seconds"),
            ("fp_slow_gap_pct", "short_run", "slow_corner_seconds"),
            ("fp_medium_gap_pct", "short_run", "medium_corner_seconds"),
            ("fp_fast_gap_pct", "short_run", "fast_corner_seconds"),
        ):
            values = metric(profile, key)
            if len(values) >= 6:
                cols[name] = _gap_pct(values)
        for name, profile, key in (
            ("fp_top_speed_kph", "short_run", "top_speed_kph"),
            ("fp_deg_slope", "long_run", "tire_deg_slope"),
        ):
            values = metric(profile, key)
            if len(values) >= 6:
                median = float(np.median(list(values.values())))
                cols[name] = {team: v - median for team, v in values.items()}
        frames.append(pd.DataFrame(cols))
    if not frames:
        return pd.DataFrame(columns=list(PRACTICE_FEATURES[:7]))
    return pd.concat(frames).groupby(level=0).mean()


def driver_practice_deltas(snapshot_dir: Path, sessions: Sequence[str]) -> pd.DataFrame:
    """Return each driver's mean teammate delta in seconds (negative = faster)."""
    rows: dict[str, dict[str, list[float]]] = {}
    for session in sessions:
        path = snapshot_dir / f"{session}.json"
        if not path.exists():
            continue
        for data in (_load(path).get("teams") or {}).values():
            deltas = data.get("driver_deltas_seconds") or {}
            for profile, name in (
                ("short_run", "fp_driver_short_s"),
                ("long_run", "fp_driver_long_s"),
            ):
                for driver, value in (deltas.get(profile) or {}).items():
                    if isinstance(value, int | float) and abs(value) < 2.0:
                        rows.setdefault(driver, {}).setdefault(name, []).append(float(value))
    return pd.DataFrame(
        {d: {k: float(np.mean(v)) for k, v in cols.items()} for d, cols in rows.items()}
    ).T


def _mean_position(
    table: dict[str, dict[str, int]],
    who: Sequence[str],
    races: Sequence[str],
    dnf: dict[str, set[str]] | None = None,
) -> float:
    """Return the mean position of ``who`` over ``races``, skipping DNFs when ``dnf`` is given."""
    values = [
        table[r][d]
        for r in races
        for d in who
        if d in table[r] and not (dnf is not None and d in dnf[r])
    ]
    return float(np.mean(values)) if values else np.nan


def _centred_mean(per_race: dict[str, dict[str, float]], races: Sequence[str]) -> dict[str, float]:
    frame = pd.DataFrame({r: per_race[r] for r in races if r in per_race}).T
    if frame.empty:
        return {}
    gaps = frame.mean()
    return (gaps - gaps.mean()).to_dict()


def build_dataset(
    replay_root: Path,
    *,
    year: int = 2026,
    team_race_pace: dict[str, dict[str, float]] | None = None,
    car_track_traits: dict[str, Any] | None = None,
    track_characteristics: dict[str, Any] | None = None,
    traits_fn: Callable[..., dict[str, float]] | None = None,
    schedule: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return one row per driver per race with features, champion forecasts and actuals.

    ``schedule`` gives the calendar order (race names); without it races are ordered by
    their FP1 start time.
    """
    pred_root = replay_root / "predictions" / str(year)
    snap_root = replay_root / "car_characteristics_snapshot" / str(year)
    events = []
    for race_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        files = {p.stem.rsplit("_", 1)[-1]: p for p in race_dir.glob("*.json")}
        fmt = "sprint" if "sq" in files else "normal"
        checkpoint = LAST_PRE_Q_CHECKPOINT[fmt]
        if checkpoint not in files:
            continue
        pred = _load(files[checkpoint])
        snap_dir = snap_root / race_dir.name
        race = pred["metadata"]["race_name"]
        if schedule is not None:
            start: Any = list(schedule).index(race)
        elif (snap_dir / "fp1.json").exists():
            start = _load(snap_dir / "fp1.json").get("session_started_at", "")
        else:
            start = ""
        events.append((start, race_dir.name, fmt, pred, snap_dir))
    events.sort()

    races = [e[3]["metadata"]["race_name"] for e in events]
    actual_q, actual_r, dnf, team_of = {}, {}, {}, {}
    for (*_unused, pred, _d), race in zip(events, races, strict=True):
        targets = pred["actuals"]["targets"]
        actual_q[race] = _order(targets.get("main_qualifying"))
        actual_r[race] = _order(targets.get("grand_prix_race"))
        dnf[race] = {str(r["driver"]) for r in targets.get("grand_prix_race") or [] if r.get("dnf")}
        team_of[race] = {
            str(r["driver"]): str(r["team"]) for r in targets.get("grand_prix_race") or []
        }

    tracks = (track_characteristics or {}).get("tracks", {})
    quali_gaps = {
        r: v["quali_gaps"]
        for r, v in (car_track_traits or {}).get("races", {}).items()
        if v.get("quali_gaps")
    }
    rows = []
    for k, (_start, _name, fmt, pred, snap_dir) in enumerate(events):
        race, prior = races[k], races[:k]
        sessions = VISIBLE_SESSIONS[fmt]
        practice = practice_features(snap_dir, sessions)
        drv_fp = driver_practice_deltas(snap_dir, sessions)
        champ_q = _order(pred["targets"]["main_qualifying"]["predicted_order"])
        champ_r = _order(pred["targets"]["grand_prix_race"]["predicted_order"])
        race_pace = _centred_mean(team_race_pace or {}, prior)
        q_gap = _centred_mean(quali_gaps, prior)
        traits = traits_fn(race, prior) if traits_fn and len(prior) >= 4 else {}
        track = tracks.get(race, {})
        for driver, team in team_of[race].items():
            mate = [d for d, t in team_of[race].items() if t == team and d != driver]

            team_drivers = [d for d, t in team_of[race].items() if t == team]
            mate_gap = [
                actual_q[r][driver] - actual_q[r][m]
                for r in prior
                for m in mate
                if driver in actual_q[r] and m in actual_q[r] and team_of[r].get(m) == team
            ]
            drv_races = [r for r in prior if driver in actual_r[r]]
            team_entries = [(r, d) for r in prior for d, t in team_of[r].items() if t == team]
            row = {
                "race": race,
                "round": k + 1,
                "driver": driver,
                "team": team,
                **{
                    c: practice.at[team, c] if team in practice.index and c in practice else np.nan
                    for c in PRACTICE_FEATURES[:7]
                },
                **{
                    c: drv_fp.at[driver, c] if driver in drv_fp.index and c in drv_fp else np.nan
                    for c in PRACTICE_FEATURES[7:]
                },
                "drv_q_mean3": _mean_position(actual_q, [driver], prior[-3:]),
                "drv_q_season": _mean_position(actual_q, [driver], prior),
                "team_q_mean3": _mean_position(actual_q, team_drivers, prior[-3:]),
                "drv_vs_mate_q": float(np.mean(mate_gap)) if mate_gap else np.nan,
                "drv_r_mean3": _mean_position(actual_r, [driver], prior[-3:], dnf),
                "team_r_mean3": _mean_position(actual_r, team_drivers, prior[-3:], dnf),
                "drv_dnf_rate": (
                    sum(driver in dnf[r] for r in drv_races) + _DNF_PRIOR_RATE * _DNF_PRIOR_RACES
                )
                / (len(drv_races) + _DNF_PRIOR_RACES),
                "team_dnf_rate": (
                    sum(d in dnf[r] for r, d in team_entries) + _DNF_PRIOR_RATE * _DNF_PRIOR_RACES
                )
                / (len(team_entries) + _DNF_PRIOR_RACES),
                "team_race_pace_s": race_pace.get(team, np.nan),
                "team_quali_gap_s": q_gap.get(team, np.nan),
                "traits_q_adj_s": traits.get(team, np.nan),
                "overtaking_difficulty": float(track.get("overtaking_difficulty", np.nan)),
                "street": float(track.get("type") == "street"),
                "sprint_weekend": float(fmt == "sprint"),
                "wet": float(str(pred["metadata"].get("weather", "dry")) != "dry"),
                "champion_q": champ_q.get(driver, np.nan),
                "champion_r": champ_r.get(driver, np.nan),
                "y_q": actual_q[race].get(driver, np.nan),
                "y_r": actual_r[race].get(driver, np.nan),
                "dnf": driver in dnf[race],
            }
            rows.append(row)
    return pd.DataFrame(rows)


def ridge_model() -> Any:
    """Return a standardised ridge regression with median imputation."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-1, 3, 13)),
    )


def boosting_model() -> Any:
    """Return a small gradient boosting regressor (handles missing values natively)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0
    )


def walk_forward(
    data: pd.DataFrame,
    target: str,
    features: Sequence[str],
    make_model: Callable[[], Any],
    *,
    min_train_races: int = 4,
) -> tuple[pd.Series, dict[int, Any]]:
    """Predict ``target`` for each race from a model fitted on earlier races only.

    Returns predicted scores (lower = better position) and the fitted model per round.
    """
    preds = pd.Series(np.nan, index=data.index)
    models: dict[int, Any] = {}
    cols = list(features)
    for rnd in sorted(data["round"].unique()):
        train = data[(data["round"] < rnd) & data[target].notna()]
        if train["round"].nunique() < min_train_races:
            continue
        model = make_model().fit(train[cols], train[target])
        test = data["round"] == rnd
        preds[test] = model.predict(data.loc[test, cols])
        models[int(rnd)] = model
    return preds, models


def score(data: pd.DataFrame, target: str, scores: pd.Series) -> dict[str, float]:
    """Return per-race mean rank correlation, MAE in positions and top-3 hit rate."""
    corr, mae, top3 = [], [], []
    for _race, idx in data[scores.notna() & data[target].notna()].groupby("race").groups.items():
        actual = data.loc[idx, target].rank()
        predicted = scores[idx].rank(method="first")
        corr.append(actual.corr(predicted))
        mae.append(float((actual - predicted).abs().mean()))
        top3.append(len(set(actual.nsmallest(3).index) & set(predicted.nsmallest(3).index)) / 3)
    return {
        "corr": float(np.mean(corr)),
        "mae": float(np.mean(mae)),
        "top3": float(np.mean(top3)),
        "races": len(corr),
    }
