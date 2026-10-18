"""Score saved challenger forecasts against the champion, per target.

Reads every saved forecast that carries ``challengers`` and actual results, scores each
challenger with the champion's own metrics and prints a paired table: forecasts
compared, mean MAE and correlation for both, and how many forecasts the challenger
won, lost or tied on MAE. For the replay, prefer ``scripts/compare_replay_arms.py``
on ``<root>/challengers/<name>``, which also applies the seed floor.

    uv run python scripts/score_challengers.py --year 2026 --env-file .env.local
    uv run python scripts/score_challengers.py --predictions-dir <replay root>/predictions
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.challengers import challenger_prediction_view  # noqa: E402
from src.utils.env_file import load_env_file  # noqa: E402
from src.utils.prediction_logger import PredictionLogger  # noqa: E402
from src.utils.prediction_metrics import PredictionMetrics  # noqa: E402

# (challenger, target) -> list of (champion metrics, challenger metrics)
Pairs = dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]]


def collect_pairs(predictions: list[dict[str, Any]]) -> Pairs:
    """Pair champion and challenger metrics for every scored target of every forecast."""
    pairs: Pairs = defaultdict(list)
    for prediction in predictions:
        names = sorted(prediction.get("challengers") or {})
        if not names:
            continue
        is_sprint = str(prediction.get("metadata", {}).get("weekend_format", "")) == "sprint"
        champion = PredictionMetrics.calculate_prediction_target_metrics(
            prediction, is_sprint=is_sprint
        )
        for name in names:
            view = challenger_prediction_view(prediction, name)
            if view is None:
                continue
            challenger = PredictionMetrics.calculate_prediction_target_metrics(
                view, is_sprint=is_sprint
            )
            for target_key, metrics in challenger.items():
                if target_key in champion:
                    pairs[(name, target_key)].append((champion[target_key], metrics))
    return pairs


def summarize(pairs: Pairs) -> list[dict[str, Any]]:
    """One row per challenger and target; negative MAE delta means the challenger is better."""
    rows = []
    for (name, target_key), items in sorted(pairs.items()):
        champion_mae = [float(c["overall_mae"]) for c, _ in items]
        challenger_mae = [float(x["overall_mae"]) for _, x in items]
        deltas = [x - c for c, x in zip(champion_mae, challenger_mae, strict=True)]
        rows.append(
            {
                "challenger": name,
                "target": target_key,
                "n": len(items),
                "champion_mae": sum(champion_mae) / len(items),
                "challenger_mae": sum(challenger_mae) / len(items),
                "champion_corr": sum(float(c.get("correlation") or 0.0) for c, _ in items)
                / len(items),
                "challenger_corr": sum(float(x.get("correlation") or 0.0) for _, x in items)
                / len(items),
                "better": sum(delta < 0 for delta in deltas),
                "worse": sum(delta > 0 for delta in deltas),
                "tied": sum(delta == 0 for delta in deltas),
            }
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table of the summary rows."""
    lines = [
        "| Challenger | Target | n | Champion MAE | Challenger MAE | Champion corr | "
        "Challenger corr | Better / worse / tied |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['challenger']} | {row['target']} | {row['n']} | {row['champion_mae']:.3f} "
            f"| {row['challenger_mae']:.3f} | {row['champion_corr']:.3f} "
            f"| {row['challenger_corr']:.3f} | {row['better']} / {row['worse']} / {row['tied']} |"
        )
    return "\n".join(lines)


def main() -> int:
    """Print the paired challenger table for one season."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--predictions-dir", default="data/predictions")
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    if args.env_file is not None:
        load_env_file(args.env_file)
    predictions = PredictionLogger(predictions_dir=args.predictions_dir).get_all_predictions(
        args.year
    )
    rows = summarize(collect_pairs(predictions))
    if not rows:
        print("No scored challenger forecasts yet.")
        return 0
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
