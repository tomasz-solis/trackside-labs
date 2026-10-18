"""Size the parts of a car x track x driver qualifying model before building it.

Walk-forward over 2026 only, each race predicted from earlier races:

1. Car pace: the team's mean qualifying gap (seconds) over earlier races.
2. Car x track: how much of the team gap left after car pace the traits-by-track fit
   (``src.models.track_traits.fit_and_predict``) explains, against a null that gives
   each race another race's track profile.
3. Driver: how well a driver's earlier gap to his teammate (percent of pole lap)
   predicts the next one, against a null that flips the sign of each earlier race.

Reads the season's car track traits from the artifact store (``--env-file .env.local``
for production) and qualifying laps from the local FastF1 cache.

    uv run python scripts/probe_car_track_driver.py --env-file .env.local
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.car_track_traits import load_car_track_traits  # noqa: E402
from src.models import track_traits as tt  # noqa: E402
from src.utils.env_file import load_env_file  # noqa: E402
from src.utils.team_mapping import map_team_to_characteristics  # noqa: E402
from src.utils.weekend import get_schedule_rows  # noqa: E402

FIRST_SCORED = 5  # predict from the 5th race on, as the ML probe did
NULL_DRAWS = 100


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(1.0 - np.sum((actual - predicted) ** 2) / np.sum(actual**2))


def team_parts(races: list[str], traits: dict, rng: np.random.Generator) -> dict[str, float]:
    """Variance of team gaps explained by car pace, then by the traits-by-track fit."""
    gaps = {r: traits[r]["quali_gaps"] for r in races}
    features = tuple(f for f in tt.TRAIT_TO_SHARE if f != "braking")
    gap_all, resid_all, fit_all = [], [], []
    null_fits: list[list[np.ndarray]] = [[] for _ in range(NULL_DRAWS)]
    for k in range(FIRST_SCORED - 1, len(races)):
        race, prior = races[k], races[:k]
        current = pd.Series(gaps[race], dtype=float)
        current -= current.mean()
        pace = pd.DataFrame([gaps[r] for r in prior]).mean().reindex(current.index)
        pace -= pace.mean()
        resid = current - pace
        fit = pd.Series(
            tt.fit_and_predict(traits, gaps, prior, traits[race]["profile"], features)
        ).reindex(current.index)
        # fit is seconds gained (positive = faster); gaps are seconds behind.
        common = resid.dropna().index.intersection(fit.dropna().index)
        gap_all.append(current.loc[common].to_numpy())
        resid_all.append(resid.loc[common].to_numpy())
        fit_all.append(-fit.loc[common].to_numpy())
        others = [r for r in races if r != race]
        for draw in range(NULL_DRAWS):
            profile = traits[others[rng.integers(len(others))]]["profile"]
            null = pd.Series(tt.fit_and_predict(traits, gaps, prior, profile, features))
            null_fits[draw].append(-null.reindex(common).fillna(0.0).to_numpy())
    gap = np.concatenate(gap_all)
    resid = np.concatenate(resid_all)
    fit = np.concatenate(fit_all)
    null_r2 = [_r2(resid, np.concatenate(draw)) for draw in null_fits]
    return {
        "races": len(gap_all),
        "team_gap_sd_s": float(np.std(gap)),
        "residual_sd_s": float(np.std(resid)),
        "car_pace_r2": _r2(gap, gap - resid),
        "track_fit_r2": _r2(resid, fit),
        "track_fit_null_p95": float(np.percentile(null_r2, 95)),
        "track_fit_p": float(np.mean([value >= _r2(resid, fit) for value in null_r2])),
    }


def car_pace_variants(races: list[str], traits: dict) -> dict[str, float]:
    """Walk-forward R2 of the team gap for season mean, last 3 races and decaying weights."""
    gaps = {r: traits[r]["quali_gaps"] for r in races}
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for k in range(FIRST_SCORED - 1, len(races)):
        race, prior = races[k], races[:k]
        current = pd.Series(gaps[race], dtype=float)
        current -= current.mean()
        history = pd.DataFrame([gaps[r] for r in prior]).reindex(columns=current.index)
        variants = {
            "season mean": history.mean(),
            "last 3 races": history.tail(3).mean(),
        }
        for half_life in (2, 4, 8):
            weights = 0.5 ** (np.arange(len(prior))[::-1] / half_life)
            variants[f"half-life {half_life} races"] = (history.mul(weights, axis=0)).sum() / (
                history.notna().mul(weights, axis=0).sum()
            )
        for name, pace in variants.items():
            pace = pace - pace.mean()
            common = current.dropna().index.intersection(pace.dropna().index)
            out.setdefault(name, []).append(
                (current.loc[common].to_numpy(), pace.loc[common].to_numpy())
            )
    return {
        name: _r2(np.concatenate([a for a, _ in pairs]), np.concatenate([p for _, p in pairs]))
        for name, pairs in out.items()
    }


def driver_deltas(year: int, races: list[str]) -> pd.DataFrame:
    """Each driver's best qualifying lap minus his team's mean, in percent of pole."""
    import fastf1

    rows = []
    for race in races:
        session = fastf1.get_session(year, race, "Q")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results
        best = results[["Q1", "Q2", "Q3"]].apply(lambda col: col.dt.total_seconds()).min(axis=1)
        frame = pd.DataFrame(
            {
                "race": race,
                "driver": results["Abbreviation"].astype(str),
                "team": [
                    map_team_to_characteristics(str(t)) or str(t) for t in results["TeamName"]
                ],
                "lap": best.to_numpy(),
            }
        ).dropna()
        pole = frame["lap"].min()
        frame["delta_pct"] = (
            (frame["lap"] - frame.groupby("team")["lap"].transform("mean")) / pole * 100.0
        )
        # A lap far off the pace is a crash or no time, not driver ability.
        frame = frame[frame["delta_pct"].abs() < 2.0]
        rows.append(frame[frame.groupby("team")["driver"].transform("count") == 2])
    return pd.concat(rows, ignore_index=True)


def driver_part(
    deltas: pd.DataFrame, races: list[str], rng: np.random.Generator
) -> dict[str, float]:
    """How well earlier teammate deltas, shrunk by sample size, predict the next one."""
    shrink = 2.0  # pseudo-races of zero delta
    actual, predicted = [], []
    history: dict[str, list[float]] = {}
    null_history: list[dict[str, list[float]]] = [{} for _ in range(NULL_DRAWS)]
    null_pred: list[list[float]] = [[] for _ in range(NULL_DRAWS)]
    for k, race in enumerate(races):
        current = deltas[deltas["race"] == race]
        for _, row in current.iterrows():
            if k >= FIRST_SCORED - 1:
                past = history.get(row["driver"], [])
                actual.append(row["delta_pct"])
                predicted.append(sum(past) / (len(past) + shrink))
                for draw in range(NULL_DRAWS):
                    null_past = null_history[draw].get(row["driver"], [])
                    null_pred[draw].append(sum(null_past) / (len(null_past) + shrink))
        # One sign per team, race and draw: teammates' deltas stay opposite.
        signs = {
            (team, draw): rng.choice([-1.0, 1.0])
            for team in current["team"].unique()
            for draw in range(NULL_DRAWS)
        }
        for _, row in current.iterrows():
            history.setdefault(row["driver"], []).append(row["delta_pct"])
            for draw in range(NULL_DRAWS):
                sign = signs[(row["team"], draw)]
                null_history[draw].setdefault(row["driver"], []).append(sign * row["delta_pct"])
    a, p = np.array(actual), np.array(predicted)
    null_r2 = [_r2(a, np.array(draw)) for draw in null_pred]
    return {
        "rows": len(a),
        "delta_sd_pct": float(np.std(a)),
        "persistence_r2": _r2(a, p),
        "persistence_corr": float(np.corrcoef(a, p)[0, 1]),
        "null_p95": float(np.percentile(null_r2, 95)),
        "p": float(np.mean([value >= _r2(a, p) for value in null_r2])),
    }


def main() -> int:
    """Print the sizing table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    if args.env_file is not None:
        load_env_file(args.env_file)
    logging.disable(logging.WARNING)
    rng = np.random.default_rng(0)

    traits = load_car_track_traits(args.year)["races"]
    order = [name for name, _ in get_schedule_rows(args.year)]
    races = [
        r for r in order if r in traits and traits[r].get("quali_gaps") and traits[r].get("profile")
    ]
    team = team_parts(races, traits, rng)
    driver = driver_part(driver_deltas(args.year, races), races, rng)

    print(f"Races with traits and qualifying gaps: {len(races)} ({races[0]} to {races[-1]})")
    print(
        f"Scored from race {FIRST_SCORED}: team {team['races']} races, driver {driver['rows']} rows"
    )
    print()
    print("| Part | Size | Explained (walk-forward R2) | Null 95th pct | p |")
    print("|---|---|---|---|---|")
    print(
        f"| Car pace (team mean so far) | team gap sd {team['team_gap_sd_s']:.3f} s | {team['car_pace_r2']:.3f} | | |"
    )
    print(
        f"| Car x track (traits fit on what car pace leaves) | residual sd {team['residual_sd_s']:.3f} s "
        f"| {team['track_fit_r2']:.3f} | {team['track_fit_null_p95']:.3f} | {team['track_fit_p']:.3f} |"
    )
    print(
        f"| Driver (earlier teammate gap) | delta sd {driver['delta_sd_pct']:.3f} % of pole "
        f"| {driver['persistence_r2']:.3f} (corr {driver['persistence_corr']:.3f}) "
        f"| {driver['null_p95']:.3f} | {driver['p']:.3f} |"
    )
    print()
    print("| Car pace estimate | Walk-forward R2 of team gap |")
    print("|---|---|")
    for name, value in car_pace_variants(races, traits).items():
        print(f"| {name} | {value:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
