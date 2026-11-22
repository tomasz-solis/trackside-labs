"""Size whether earlier seasons carry into the target season before building on them (plan row CH).

Two parts, both from official qualifying times (FastF1, fetched once, then cached):

1. Car x track. A team's residual: its best lap behind the fastest team, in percent of
   that lap, centred on the race, minus its car pace (mean over the target season's
   earlier races, walk-forward from race 5; full-season mean in a history season). Each
   target row is paired with the same team's mean residual at the same circuit
   (schedule ``Location``) in the history seasons. Null: its residual at another
   circuit it raced. Every p is two-sided.
2. Driver. A driver's teammate gap: their best lap minus the team's mean, in percent of
   pole, for teams with both laps within 2%. Candidates from the history seasons: the
   mean teammate gap (shrunk by 2 sessions of zero) and qualifying sessions entered
   (log, centred on the team). Each is set against the earlier-target-season gap the
   champion's driver term already holds. Null: each team and session's sign flipped.

``--target 2024 --history 2023`` is the positive control within one rule set.

    uv run python scripts/probe_history_carryover.py --target 2026 --history 2023 2024 2025
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

from src.dashboard.cache import enable_fastf1_cache  # noqa: E402
from src.utils.team_mapping import map_team_to_characteristics  # noqa: E402

FIRST_SCORED = 5  # as scripts/probe_car_track_driver.py
NULL_DRAWS = 1000
MAX_GAP_PCT = 5.0  # a team further off has no representative lap (crash, no time)
MAX_TEAMMATE_PCT = 2.0  # as scripts/probe_car_track_driver.py
SHRINK = 2.0  # pseudo-sessions of zero teammate gap


def driver_laps(year: int) -> pd.DataFrame:
    """One row per driver and qualifying with a time: their best of Q1 to Q3, seconds."""
    import fastf1

    rows = []
    for event in fastf1.get_event_schedule(year, include_testing=False).itertuples():
        try:
            session = fastf1.get_session(year, event.EventName, "Q")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as exc:  # an event not run yet, or no timing published
            logging.getLogger(__name__).info("skip %s %s: %s", year, event.EventName, exc)
            continue
        results = session.results
        best = results[["Q1", "Q2", "Q3"]].apply(lambda column: column.dt.total_seconds())
        frame = pd.DataFrame(
            {
                "year": year,
                "round": int(event.RoundNumber),
                "location": str(event.Location),
                "driver": results["Abbreviation"].astype(str).to_numpy(),
                "team": [
                    map_team_to_characteristics(str(t)) or str(t) for t in results["TeamName"]
                ],
                "lap": best.min(axis=1).to_numpy(),
            }
        ).dropna()
        if frame["team"].nunique() >= 8:
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def team_gaps(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per team and qualifying: its best lap behind the fastest team, centred, %."""
    best = laps.groupby(["year", "round", "location", "team"], as_index=False)["lap"].min()
    fastest = best.groupby(["year", "round"])["lap"].transform("min")
    best["gap"] = (best["lap"] - fastest) / fastest * 100.0
    best = best[best["gap"] < MAX_GAP_PCT]
    best["gap"] -= best.groupby(["year", "round"])["gap"].transform("mean")
    return best


def history_residuals(gaps: pd.DataFrame) -> pd.Series:
    """Mean residual per (team, location) over the history seasons."""
    pace = gaps.groupby(["year", "team"])["gap"].transform("mean")
    return (gaps["gap"] - pace).groupby([gaps["team"], gaps["location"]]).mean()


def target_residuals(gaps: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward residuals of the target season from its 5th race."""
    rounds = sorted(gaps["round"].unique())
    out = []
    for k in range(FIRST_SCORED - 1, len(rounds)):
        current = gaps[gaps["round"] == rounds[k]]
        pace = gaps[gaps["round"].isin(rounds[:k])].groupby("team")["gap"].mean()
        current = current.assign(resid=current["gap"] - current["team"].map(pace))
        out.append(current.dropna(subset=["resid"]))
    return pd.concat(out, ignore_index=True)


def _walk_forward_r2(actual: np.ndarray, prior: np.ndarray, rounds: np.ndarray) -> float:
    """R2 of slope * prior, the slope fitted through the origin on earlier rounds only."""
    predicted = np.full_like(actual, np.nan)
    for k in np.unique(rounds):
        earlier = rounds < k
        if earlier.sum() >= 10 and np.dot(prior[earlier], prior[earlier]) > 0:
            slope = np.dot(prior[earlier], actual[earlier]) / np.dot(prior[earlier], prior[earlier])
            predicted[rounds == k] = slope * prior[rounds == k]
    scored = ~np.isnan(predicted)
    return float(
        1 - np.sum((actual[scored] - predicted[scored]) ** 2) / np.sum(actual[scored] ** 2)
    )


def car_track_part(
    target: pd.DataFrame, history: pd.Series, rng: np.random.Generator
) -> dict[str, float]:
    """Correlation of target residuals with the same circuit's history, and its null."""
    keys = list(zip(target["team"], target["location"], strict=True))
    paired = target[[key in history.index for key in keys]].reset_index(drop=True)
    prior = np.array(
        [history[(t, loc)] for t, loc in zip(paired["team"], paired["location"], strict=True)]
    )
    actual = paired["resid"].to_numpy()
    by_team = {team: history.xs(team) for team in paired["team"].unique()}
    null = []
    for _ in range(NULL_DRAWS):
        draw = []
        for team, location in zip(paired["team"], paired["location"], strict=True):
            others = by_team[team].drop(location)
            draw.append(others.iloc[rng.integers(len(others))])
        null.append(np.corrcoef(actual, draw)[0, 1])
    observed = float(np.corrcoef(actual, prior)[0, 1])
    return {
        "rows": len(paired),
        "sd": float(np.std(actual)),
        "corr": observed,
        "null_p95": float(np.percentile(np.abs(null), 95)),
        "p": float(np.mean(np.abs(null) >= abs(observed))),
        "r2": _walk_forward_r2(actual, prior, paired["round"].to_numpy()),
    }


def teammate_deltas(laps: pd.DataFrame) -> pd.DataFrame:
    """Each driver's best lap minus the team's mean, % of pole; both laps within 2%."""
    frame = laps.copy()
    pole = frame.groupby(["year", "round"])["lap"].transform("min")
    team_mean = frame.groupby(["year", "round", "team"])["lap"].transform("mean")
    frame["delta"] = (frame["lap"] - team_mean) / pole * 100.0
    pair = frame.groupby(["year", "round", "team"])["driver"].transform("count") == 2
    within = (
        frame.groupby(["year", "round", "team"])["delta"].transform(lambda d: d.abs().max())
        < MAX_TEAMMATE_PCT
    )
    return frame[pair & within].reset_index(drop=True)


def driver_part(
    target: pd.DataFrame, history: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, float]]:
    """Each candidate's correlation with the target teammate gap, and R2 by season stage."""
    hist_sum = history.groupby("driver")["delta"].sum()
    hist_n = history.groupby("driver")["delta"].count()
    target = target.sort_values(["round", "team", "driver"]).reset_index(drop=True)
    shrunk_history = (
        target["driver"].map(hist_sum).fillna(0.0)
        / (target["driver"].map(hist_n).fillna(0.0) + SHRINK)
    ).to_numpy()
    sessions = np.log1p(
        target["driver"].map(history.groupby("driver")["round"].count()).fillna(0.0)
    )
    sessions = (
        sessions - sessions.groupby([target["round"], target["team"]]).transform("mean")
    ).to_numpy()
    earlier = np.zeros(len(target))
    blended = np.zeros(len(target))
    seen: dict[str, list[float]] = {}
    for index, row in target.iterrows():
        past = seen.get(row["driver"], [])
        earlier[index] = sum(past) / (len(past) + SHRINK)
        blended[index] = (sum(past) + SHRINK * shrunk_history[index]) / (len(past) + SHRINK)
        if index + 1 == len(target) or target.loc[index + 1, "round"] != row["round"]:
            for _, done in target[target["round"] == row["round"]].iterrows():
                seen.setdefault(done["driver"], []).append(done["delta"])
    actual = target["delta"].to_numpy()
    groups = target.groupby(["round", "team"]).ngroup().to_numpy()
    rounds = target["round"].to_numpy()
    early = rounds <= sorted(set(rounds))[3]
    flips = rng.choice([-1.0, 1.0], size=(NULL_DRAWS, groups.max() + 1))[:, groups]

    def score(candidate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        a, c = actual[mask], candidate[mask]
        if np.std(c) == 0:
            return {"corr": float("nan"), "null_p95": float("nan"), "p": float("nan")}
        null = [np.corrcoef(a * f[mask], c)[0, 1] for f in flips]
        observed = float(np.corrcoef(a, c)[0, 1])
        return {
            "corr": observed,
            "null_p95": float(np.percentile(np.abs(null), 95)),
            "p": float(np.mean(np.abs(null) >= abs(observed))),
        }

    everything = np.ones(len(target), dtype=bool)
    out = {
        name: {"all": score(values, everything), "races 1 to 4": score(values, early)}
        for name, values in (
            ("earlier target-season gap", earlier),
            ("history teammate gap", shrunk_history),
            ("history sessions entered", sessions),
        )
    }
    skill = {"rows": float(len(target))}
    for scope, mask in (("all", everything), ("races 1 to 4", early), ("race 5 on", ~early)):
        total = float(np.sum(actual[mask] ** 2))
        for name, predicted in (("earlier gap", earlier), ("with history prior", blended)):
            skill[f"{scope}|{name}"] = (
                1 - float(np.sum((actual[mask] - predicted[mask]) ** 2)) / total
            )
    return out, skill


def main() -> int:
    """Print both parts' carry-over from the history seasons into the target season."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=int, default=2026)
    parser.add_argument("--history", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    enable_fastf1_cache()
    rng = np.random.default_rng(0)

    target_laps = driver_laps(args.target)
    history_laps = pd.concat([driver_laps(year) for year in args.history], ignore_index=True)
    car = car_track_part(
        target_residuals(team_gaps(target_laps)), history_residuals(team_gaps(history_laps)), rng
    )
    print(f"Target {args.target}, history {args.history}")
    print(
        f"Car x track: {car['rows']} team rows, residual sd {car['sd']:.3f}% of pole; corr "
        f"{car['corr']:+.3f} (null |corr| 95th {car['null_p95']:.3f}, p {car['p']:.3f}); "
        f"walk-forward R2 {car['r2']:+.3f}"
    )
    driver, skill = driver_part(teammate_deltas(target_laps), teammate_deltas(history_laps), rng)
    print(f"Driver: {int(skill['rows'])} driver rows")
    for name, parts in driver.items():
        cells = "; ".join(
            f"{scope} corr {s['corr']:+.3f} (null |corr| 95th {s['null_p95']:.3f}, p {s['p']:.3f})"
            for scope, s in parts.items()
        )
        print(f"  {name}: {cells}")
    for scope in ("all", "races 1 to 4", "race 5 on"):
        print(
            f"  R2 of the teammate gap, {scope}: earlier gap {skill[scope + '|earlier gap']:+.3f}, "
            f"with the history gap as prior {skill[scope + '|with history prior']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
