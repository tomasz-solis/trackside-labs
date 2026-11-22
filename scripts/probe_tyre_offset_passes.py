"""Size whether real passes follow tyre offset once pace is held fixed (open risk B2).

Close pair-laps from race laps: adjacent at the end of lap L-1 with a gap of 1.2 s or
less, laps L-1 and L green, no pit stop by either car in either lap, L from 3. Outcome:
the follower is ahead at the end of lap L. Pace is each driver's median green, non-pit
lap in that race. Tyre offset is the car ahead's tyre age minus the follower's, so
positive means the follower is on fresher tyres. Logistic fits with a race bootstrap and
a leave-one-race-out log loss. Reads the FastF1 cache only; a race not cached is skipped.

    uv run python scripts/probe_tyre_offset_passes.py --year 2026 --skip "Canadian Grand Prix" "São Paulo Grand Prix"
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

PASS_WINDOW_S = 1.2  # the simulator's pass_window_s
BOOTSTRAP = 1000
FORMS = {
    "pace and gap": ["pace_adv", "gap"],
    "plus offset, clipped to 10 laps": ["pace_adv", "tyre_clip", "gap"],
    "plus 3+ laps fresher / older": ["pace_adv", "fresher3", "older3", "gap"],
}


def pair_laps(year: int, skip: set[str]) -> pd.DataFrame:
    """One row per close follower and car ahead on a green lap, from cached races."""
    import fastf1

    fastf1.Cache.offline_mode(True)
    rows = []
    for event in fastf1.get_event_schedule(year, include_testing=False).itertuples():
        if event.EventName in skip:
            continue
        try:
            session = fastf1.get_session(year, event.EventName, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps.copy()
        except Exception as exc:  # not cached, or not run yet
            logging.getLogger(__name__).info("skip %s: %s", event.EventName, exc)
            continue
        laps["green"] = laps["TrackStatus"].astype(str).str.fullmatch(r"1+")
        laps["pit"] = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
        clean = laps[laps["green"] & ~laps["pit"] & laps["LapTime"].notna()]
        pace = clean[clean["LapNumber"] > 1].groupby("Driver")["LapTime"].median()
        pace = pace.dt.total_seconds()
        by_lap = laps.set_index(["LapNumber", "Driver"])
        for lap in sorted(laps["LapNumber"].dropna().unique()):
            if lap < 3:
                continue
            before = laps[(laps["LapNumber"] == lap - 1) & laps["Position"].notna()]
            order = before.sort_values("Position")["Driver"].tolist()
            for ahead, follower in zip(order, order[1:], strict=False):
                keys = [(lap - 1, ahead), (lap - 1, follower), (lap, ahead), (lap, follower)]
                if any(key not in by_lap.index for key in keys):
                    continue
                a0, f0, a1, f1 = (by_lap.loc[key] for key in keys)
                if not all(row["green"] and not row["pit"] for row in (a0, f0, a1, f1)):
                    continue
                gap = (f0["Time"] - a0["Time"]).total_seconds()
                if not 0 < gap <= PASS_WINDOW_S or pd.isna(a1["Position"]):
                    continue
                if ahead not in pace or follower not in pace or pd.isna(f1["Position"]):
                    continue
                rows.append(
                    {
                        "race": event.EventName,
                        "gap": gap,
                        "passed": float(f1["Position"] < a1["Position"]),
                        "pace_adv": float(pace[ahead] - pace[follower]),
                        "tyre_adv": float(a0["TyreLife"] - f0["TyreLife"]),
                    }
                )
    frame = pd.DataFrame(rows).dropna(subset=["tyre_adv"])
    # A driver with a handful of clean laps gets a meaningless median.
    frame = frame[frame["pace_adv"].abs() <= 3.0].reset_index(drop=True)
    frame["tyre_clip"] = frame["tyre_adv"].clip(-10, 10)
    frame["fresher3"] = (frame["tyre_adv"] >= 3).astype(float)
    frame["older3"] = (frame["tyre_adv"] <= -3).astype(float)
    return frame


def logit(features: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Unpenalised logistic regression by Newton steps; intercept first."""
    design = np.column_stack([np.ones(len(features)), features])
    beta = np.zeros(design.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-design @ beta))
        hessian = design.T @ (design * (p * (1 - p))[:, None])
        beta += np.linalg.solve(hessian + 1e-9 * np.eye(len(beta)), design.T @ (outcome - p))
    return beta


def held_out_log_loss(frame: pd.DataFrame, columns: list[str]) -> float:
    """Mean log loss with each race predicted from a fit on the others."""
    total = 0.0
    for race in frame["race"].unique():
        train, test = frame[frame["race"] != race], frame[frame["race"] == race]
        beta = logit(train[columns].to_numpy(), train["passed"].to_numpy())
        design = np.column_stack([np.ones(len(test)), test[columns].to_numpy()])
        p = np.clip(1.0 / (1.0 + np.exp(-design @ beta)), 1e-6, 1 - 1e-6)
        y = test["passed"].to_numpy()
        total -= float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return total / len(frame)


def main() -> int:
    """Print the pass model's terms, their price in seconds of pace, and the rate table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--skip", nargs="*", default=[], help="Wet races to leave out.")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    enable_fastf1_cache()
    frame = pair_laps(args.year, set(args.skip))
    races = frame["race"].unique()
    print(f"{len(frame)} pair-laps, {int(frame['passed'].sum())} passes, {len(races)} races")
    rng = np.random.default_rng(0)
    for name, columns in FORMS.items():
        beta = logit(frame[columns].to_numpy(), frame["passed"].to_numpy())
        draws = []
        for _ in range(BOOTSTRAP):
            sample = pd.concat([frame[frame["race"] == r] for r in rng.choice(races, len(races))])
            draws.append(logit(sample[columns].to_numpy(), sample["passed"].to_numpy()))
        boot = np.array(draws)
        print(f"{name}: held-out log loss {held_out_log_loss(frame, columns):.4f}")
        for index, column in enumerate(columns, start=1):
            lo, hi = np.percentile(boot[:, index], [2.5, 97.5])
            line = f"  {column}: {beta[index]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            if column != "pace_adv" and column != "gap":
                price = boot[:, index] / boot[:, 1]
                line += (
                    f", worth {beta[index] / beta[1]:.3f} s of pace "
                    f"[{np.percentile(price, 2.5):.3f}, {np.percentile(price, 97.5):.3f}]"
                )
            print(line)
    frame["offset"] = pd.cut(
        frame["tyre_adv"], [-99, -3, 2.5, 99], labels=["older 3+", "within 3", "fresher 3+"]
    )
    frame["pace"] = pd.cut(
        frame["pace_adv"],
        [-99, -0.3, 0.0, 0.3, 99],
        labels=["slower 0.3+", "slower 0-0.3", "faster 0-0.3", "faster 0.3+"],
    )
    table = frame.pivot_table(
        index="offset", columns="pace", values="passed", aggfunc=["mean", "count"], observed=False
    )
    print("Pass rate %, follower's pace against the car ahead:")
    print((table["mean"] * 100).round(1).to_string())
    print(table["count"].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
