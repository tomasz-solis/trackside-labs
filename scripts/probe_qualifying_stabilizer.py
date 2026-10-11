"""Probe: does blending the previous race's qualifying result into the forecast help?

The evaluation report scores a "stabilizer" (40% model rank, 60% previous-race actual
rank) on the published forecasts. This scores the same rule, and the naive
"repeat the last race" order, against the current model on replay roots.

Each race uses the forecast at the last checkpoint before qualifying and only the
previous race's actual result. Diagnostics only: changes no product data.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_ml_challenger import build  # noqa: E402

import src.models.ml_challenger as ml  # noqa: E402

MODEL_WEIGHT = 0.40  # same as _QUALIFYING_STABILIZER_MODEL_WEIGHT in the evaluation report
TARGET = "y_q"


def candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the live, naive and stabilized qualifying scores, one value per driver row."""
    actual = df.pivot_table(index="driver", columns="round", values=TARGET)
    previous = actual.shift(axis=1)
    prev = pd.Series(
        [previous.at[d, r] for d, r in zip(df["driver"], df["round"], strict=True)],
        index=df.index,
    )
    live = df["champion_q"]
    # A driver with no previous result keeps the model's place, as in the report.
    prev_or_live = prev.fillna(live)
    return {
        "Live model": live,
        "Naive (previous race order)": prev_or_live,
        "Stabilizer (40% model, 60% previous)": MODEL_WEIGHT * live
        + (1 - MODEL_WEIGHT) * prev_or_live,
    }


def per_race_mae(df: pd.DataFrame, scores: pd.Series) -> pd.Series:
    """Return MAE in positions for each race."""
    out = {}
    for race, idx in df[scores.notna() & df[TARGET].notna()].groupby("race").groups.items():
        out[race] = float(
            (df.loc[idx, TARGET].rank() - scores[idx].rank(method="first")).abs().mean()
        )
    return pd.Series(out)


def paired_delta(a: pd.Series, b: pd.Series, draws: int = 5000) -> tuple[float, float, float]:
    """Return mean(a - b) per race and a 95% bootstrap interval, resampling races."""
    diff = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(0)
    means = rng.choice(diff, size=(draws, len(diff))).mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    """Print the comparison for each replay root and round window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "data/historical_replay_arms/Z_base_32_s42",
            "data/historical_replay_arms/Z_base_32_s43",
        ],
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    from src.utils.weekend import get_schedule_rows

    schedule = [str(name).strip() for name, _fmt in get_schedule_rows(2026)]
    for root in args.roots:
        df = build(Path(root), schedule)
        df = df[df["champion_q"].notna() & df[TARGET].notna()]
        for first_round in (2, 5):
            window = df[df["round"] >= first_round]
            scores = candidates(df)
            print(f"\n{Path(root).name}, rounds {first_round} to {int(df['round'].max())}")
            live_mae = per_race_mae(window, scores["Live model"][window.index])
            for name, values in scores.items():
                result = ml.score(window, TARGET, values[window.index])
                line = f"  {name:38s} corr {result['corr']:.3f}  MAE {result['mae']:.2f}"
                if name != "Live model":
                    mae = per_race_mae(window, values[window.index])
                    mean, lo, hi = paired_delta(mae, live_mae)
                    better = int((mae < live_mae).sum())
                    worse = int((mae > live_mae).sum())
                    line += (
                        f"  vs live {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                        f"  better {better} / worse {worse}"
                    )
                print(line)


if __name__ == "__main__":
    main()
