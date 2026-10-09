"""Probe: can an ML model on all the pace data beat the live model? Which features matter when?

Builds the driver-race dataset from two replay roots (see ``src/models/ml_challenger.py``),
fits ridge and gradient boosting walk-forward (each race predicted from earlier races
only), scores them against the live model's forecast at the same checkpoint, and
measures each feature group's importance round by round.

Writes ``data/model_diagnostics/2026/ml_challenger_probe.{md,json,html}``.
Diagnostics only: changes no product data.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models.ml_challenger as ml  # noqa: E402
import src.models.track_traits as tt  # noqa: E402

FEATURE_GROUPS = {
    "Practice pace (short and long run)": ("fp_short_gap_pct", "fp_long_gap_pct"),
    "Practice car traits (corners, top speed, deg)": (
        "fp_slow_gap_pct",
        "fp_medium_gap_pct",
        "fp_fast_gap_pct",
        "fp_top_speed_kph",
        "fp_deg_slope",
    ),
    "Practice teammate split": ("fp_driver_short_s", "fp_driver_long_s"),
    "Recent results (driver and team form)": (
        "drv_q_mean3",
        "drv_q_season",
        "team_q_mean3",
        "drv_vs_mate_q",
        "drv_r_mean3",
        "team_r_mean3",
    ),
    "Measured pace history (race pace, quali gap)": ("team_race_pace_s", "team_quali_gap_s"),
    "Car traits x track": ("traits_q_adj_s",),
    "Reliability (DNF rate)": ("drv_dnf_rate", "team_dnf_rate"),
    "Track and conditions": ml.TRACK_FEATURES,
}
SIMPLE_FEATURES = {
    "y_q": (
        "champion_q",
        "drv_q_mean3",
        "team_q_mean3",
        "team_quali_gap_s",
        "fp_short_gap_pct",
        "fp_driver_short_s",
        "drv_vs_mate_q",
    ),
    "y_r": (
        "champion_q",
        "drv_r_mean3",
        "team_r_mean3",
        "drv_dnf_rate",
        "team_dnf_rate",
        "team_race_pace_s",
        "fp_long_gap_pct",
    ),
}
TARGETS = {"y_q": ("Qualifying", "champion_q"), "y_r": ("Race", "champion_r")}
FIRST_SCORED_ROUND = 5
OUT_DIR = Path("data/model_diagnostics/2026")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build(root: Path, schedule: list[str]) -> pd.DataFrame:
    """Return the driver-race dataset for one replay root."""
    pace = _load("data/processed/team_race_pace/2026_team_race_pace.json")["races"]
    traits = _load("data/processed/car_track_traits/2026_car_track_traits.json")
    tracks = _load("data/processed/track_characteristics/2026_track_characteristics.json")
    gaps = {r: v["quali_gaps"] for r, v in traits["races"].items() if v.get("quali_gaps")}
    feats = tuple(f for f in tt.TRAIT_TO_SHARE if f != "braking")

    def traits_fn(race: str, prior: list[str]) -> dict[str, float]:
        profile = (traits["races"].get(race) or {}).get("profile") or traits.get(
            "profiles", {}
        ).get(race)
        return tt.fit_and_predict(traits["races"], gaps, prior, profile, feats)

    return ml.build_dataset(
        root,
        team_race_pace=pace,
        car_track_traits=traits,
        track_characteristics=tracks,
        traits_fn=traits_fn,
        schedule=schedule,
    )


def _ranks(df: pd.DataFrame, scores: pd.Series) -> pd.Series:
    return scores.groupby(df["race"]).rank()


def compare(df: pd.DataFrame) -> list[dict]:
    """Return scores for the live model and each challenger, per target."""
    scored = df["round"] >= FIRST_SCORED_ROUND
    out = []
    for target, (label, champion) in TARGETS.items():
        full = ml.PACE_FEATURES + ml.CHAMPION_FEATURES
        ridge_full = ml.walk_forward(df, target, full, ml.ridge_model)[0]
        candidates = {
            "Live model": df[champion],
            "Ridge, pace features only": ml.walk_forward(
                df, target, ml.PACE_FEATURES, ml.ridge_model
            )[0],
            "Boosting, pace features only": ml.walk_forward(
                df, target, ml.PACE_FEATURES, ml.boosting_model
            )[0],
            "Ridge, pace + live forecast": ridge_full,
            "Boosting, pace + live forecast": ml.walk_forward(df, target, full, ml.boosting_model)[
                0
            ],
            "Ridge, 7 simple features": ml.walk_forward(
                df, target, SIMPLE_FEATURES[target], ml.ridge_model
            )[0],
            "50/50 blend live + ridge": 0.5 * _ranks(df, df[champion])
            + 0.5 * _ranks(df, ridge_full),
        }
        for name, scores in candidates.items():
            out.append(
                {"target": label, "model": name, **ml.score(df, target, scores.where(scored))}
            )
    return out


def paired_delta(df: pd.DataFrame, target: str, scores: pd.Series, seed: int = 0) -> dict:
    """Return the per-race MAE change vs the live model with a bootstrap 95% CI."""
    champion = TARGETS[target][1]
    deltas = []
    for _race, idx in df[df["round"] >= FIRST_SCORED_ROUND].groupby("race").groups.items():
        actual = df.loc[idx, target].rank()
        new = (actual - scores[idx].rank(method="first")).abs().mean()
        old = (actual - df.loc[idx, champion].rank(method="first")).abs().mean()
        deltas.append(float(new - old))
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(rng.choice(deltas, len(deltas)))) for _ in range(4000)]
    return {
        "mae_delta": float(np.mean(deltas)),
        "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "better": int(sum(d < 0 for d in deltas)),
        "worse": int(sum(d > 0 for d in deltas)),
    }


def importance_by_round(df: pd.DataFrame, target: str, repeats: int = 20) -> pd.DataFrame:
    """Return each feature group's share of importance per round (pace-only ridge).

    Importance = rise in training MAE when the group's columns are shuffled together,
    on the model fitted before that round. Shares sum to 1 per round.
    """
    rng = np.random.default_rng(0)
    cols = list(ml.PACE_FEATURES)
    _preds, models = ml.walk_forward(df, target, cols, ml.ridge_model)
    rows = {}
    for rnd, model in models.items():
        train = df[(df["round"] < rnd) & df[target].notna()]
        x, y = train[cols], train[target].to_numpy()
        base = np.mean(np.abs(model.predict(x) - y))
        rise = {}
        for group, members in FEATURE_GROUPS.items():
            losses = []
            for _ in range(repeats):
                shuffled = x.copy()
                order = rng.permutation(len(x))
                shuffled[list(members)] = x[list(members)].to_numpy()[order]
                losses.append(np.mean(np.abs(model.predict(shuffled) - y)))
            rise[group] = max(float(np.mean(losses)) - base, 0.0)
        total = sum(rise.values()) or 1.0
        rows[rnd] = {g: v / total for g, v in rise.items()}
    return pd.DataFrame(rows).T.sort_index()


def write_chart(importance: dict[str, pd.DataFrame], races: dict[int, str], path: Path) -> None:
    """Write a two-panel stacked area chart of feature group shares by round."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=list(importance), shared_yaxes=True)
    colors = [
        "#2563eb",
        "#60a5fa",
        "#93c5fd",
        "#f97316",
        "#16a34a",
        "#a855f7",
        "#dc2626",
        "#9ca3af",
    ]
    for col, frame in enumerate(importance.values(), start=1):
        labels = [races[r].replace(" Grand Prix", "") for r in frame.index]
        for i, group in enumerate(frame.columns):
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=(frame[group] * 100).round(1),
                    name=group,
                    stackgroup=f"s{col}",
                    line={"width": 0.5, "color": colors[i % len(colors)]},
                    legendgroup=group,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
    fig.update_yaxes(title_text="Share of importance (%)", range=[0, 100], row=1, col=1)
    fig.update_layout(
        title="What the ML model leans on, race by race (2026, walk-forward ridge)",
        template="plotly_white",
        height=560,
        legend={"orientation": "h", "y": -0.25},
    )
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    """Run the probe and write the report."""
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
    data = {root: build(Path(root), schedule) for root in args.roots}

    results = pd.DataFrame([r | {"root": root} for root, df in data.items() for r in compare(df)])
    table = results.groupby(["target", "model"], sort=False)[["corr", "mae", "top3"]].mean()
    deltas = {
        f"{TARGETS[t][0]} {Path(root).name}": paired_delta(
            df, t, ml.walk_forward(df, t, SIMPLE_FEATURES[t], ml.ridge_model)[0]
        )
        for root, df in data.items()
        for t in TARGETS
    }
    first = data[args.roots[0]]
    races = first.groupby("round")["race"].first().to_dict()
    importance = {TARGETS[t][0]: importance_by_round(first, t) for t in TARGETS}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_chart(importance, races, OUT_DIR / "ml_challenger_probe.html")
    payload = {
        "roots": args.roots,
        "rows": int(len(first)),
        "races": int(first["round"].nunique()),
        "scored_from_round": FIRST_SCORED_ROUND,
        "scores": table.round(4).reset_index().to_dict(orient="records"),
        "simple_ridge_vs_live_mae": deltas,
        "importance_by_round": {
            k: {races[int(r)]: v for r, v in frame.round(4).to_dict(orient="index").items()}
            for k, frame in importance.items()
        },
    }
    (OUT_DIR / "ml_challenger_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# ML challenger probe",
        "",
        f"{payload['races']} races, {payload['rows']} driver rows per replay root. Each race is predicted from earlier races only, at the last checkpoint before qualifying. Scored from round {FIRST_SCORED_ROUND}, mean over {len(args.roots)} replay seeds.",
        "",
        "| Target | Model | Rank corr | MAE | Top 3 |",
        "|---|---|---|---|---|",
        *[
            f"| {t} | {m} | {r['corr']:.3f} | {r['mae']:.2f} | {r['top3']:.0%} |"
            for (t, m), r in table.iterrows()
        ],
        "",
        "Simple ridge vs live model, MAE change per race (negative = better):",
        "",
        "| Target and seed | MAE change | 95% CI | Races better / worse |",
        "|---|---|---|---|",
        *[
            f"| {k} | {v['mae_delta']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {v['better']} / {v['worse']} |"
            for k, v in deltas.items()
        ],
        "",
        "Feature group share of importance by round (pace-only ridge) is in `ml_challenger_probe.json` and the chart `ml_challenger_probe.html`.",
        "",
    ]
    for label, frame in importance.items():
        last = frame.iloc[-1].sort_values(ascending=False)
        lines.append(
            f"{label}, last round ({races[int(frame.index[-1])]}): "
            + ", ".join(f"{g} {v:.0%}" for g, v in last.items() if v >= 0.01)
            + "."
        )
        lines.append("")
    (OUT_DIR / "ml_challenger_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
