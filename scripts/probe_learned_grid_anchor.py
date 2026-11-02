"""Size a learned grid anchor weight against the hand-set one (ledger 2026-11-01).

The race order blends the simulated median with the predicted grid:
``(1 - w) * sim + w * grid - racecraft``, floored at ``grid - max_gain``. Race rows
saved since 2026-11-01 keep those inputs. For each race this fits one ``w`` by least
squares on every Grand Prix checkpoint of the earlier races, rebuilds the blend for the
hand-set and the learned weight, and scores both against the actual result (Spearman
correlation and MAE, as ``PredictionMetrics``). The null refits on actuals shuffled
within each training checkpoint.

The rebuild leaves out the hypothetical-points floor and the rank-sample reorder, so it
compares two rebuilt orders, not the saved one; the saved-order agreement is printed.

Usage:
    uv run python scripts/probe_learned_grid_anchor.py --replay-root <root_s42> <root_s43>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TARGET = "grand_prix_race"
FIELDS = ("sim_median_position", "grid_reference_position", "grid_anchor_weight")


@dataclass
class Checkpoint:
    """One saved Grand Prix forecast with its blend inputs and the actual result."""

    race: str
    session: str
    predicted_at: str
    drivers: list[str]
    sim: np.ndarray
    grid: np.ndarray
    weight: np.ndarray
    racecraft: np.ndarray
    max_gain: np.ndarray
    saved: np.ndarray
    actual: np.ndarray

    @property
    def penalised(self) -> np.ndarray:
        """Penalised drivers carry weight 0 and no gain floor (``build_finish_order``)."""
        return self.weight == 0.0


def load_checkpoints(root: Path, year: int = 2026) -> list[Checkpoint]:
    """Read every Grand Prix forecast under a replay root that has blend inputs and actuals."""
    checkpoints = []
    for path in sorted((root / "predictions" / str(year)).glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = ((payload.get("targets") or {}).get(TARGET) or {}).get("predicted_order") or []
        actual_rows = ((payload.get("actuals") or {}).get("targets") or {}).get(TARGET) or []
        actual = {str(row["driver"]): int(row["position"]) for row in actual_rows}
        rows = [row for row in rows if row["driver"] in actual]
        if len(rows) < 2 or not all(field in rows[0] for field in FIELDS):
            continue
        metadata = payload["metadata"]
        checkpoints.append(
            Checkpoint(
                race=str(metadata["race_name"]),
                session=str(metadata["session_name"]),
                predicted_at=str(metadata["predicted_at"]),
                drivers=[str(row["driver"]) for row in rows],
                sim=np.array([row["sim_median_position"] for row in rows], dtype=float),
                grid=np.array([row["grid_reference_position"] for row in rows], dtype=float),
                weight=np.array([row["grid_anchor_weight"] for row in rows], dtype=float),
                racecraft=np.array([row["racecraft_adjustment"] for row in rows], dtype=float),
                max_gain=np.array([row["max_gain"] for row in rows], dtype=float),
                saved=np.array([row["position"] for row in rows], dtype=float),
                actual=np.array([actual[str(row["driver"])] for row in rows], dtype=float),
            )
        )
    return checkpoints


def rebuilt_positions(cp: Checkpoint, weight: float | None) -> np.ndarray:
    """Rank the rebuilt blend score; ``None`` keeps each row's hand-set weight."""
    w = cp.weight if weight is None else np.where(cp.penalised, 0.0, weight)
    score = (1.0 - w) * cp.sim + w * cp.grid - cp.racecraft
    floor = np.where(cp.penalised, 1.0, np.maximum(1.0, cp.grid - cp.max_gain))
    score = np.maximum(score, floor)
    order = np.lexsort((cp.grid, score))
    positions = np.empty(len(score))
    positions[order] = np.arange(1, len(score) + 1)
    return positions


def fit_weight(training: list[Checkpoint], actuals: list[np.ndarray] | None = None) -> float:
    """Least-squares ``w`` for ``actual = (1 - w) * sim + w * grid - racecraft``, in [0, 1]."""
    numerator = denominator = 0.0
    for index, cp in enumerate(training):
        keep = ~cp.penalised
        actual = cp.actual if actuals is None else actuals[index]
        gap = cp.grid[keep] - cp.sim[keep]
        numerator += float(np.sum((actual[keep] - cp.sim[keep] + cp.racecraft[keep]) * gap))
        denominator += float(np.sum(gap * gap))
    return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator else 0.5


def score(positions: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Spearman correlation and MAE of positions against the actual result."""
    return float(spearmanr(positions, actual)[0]), float(np.mean(np.abs(positions - actual)))


def walk_forward(
    checkpoints: list[Checkpoint],
    first_scored_round: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[Checkpoint, float]], float, float]:
    """Learned weight per scored checkpoint, and mean (corr, MAE) delta against hand-set."""
    races = sorted(
        {cp.race for cp in checkpoints},
        key=lambda race: min(cp.predicted_at for cp in checkpoints if cp.race == race),
    )
    fitted, corr_deltas, mae_deltas = [], [], []
    for round_index, race in enumerate(races, start=1):
        if round_index < first_scored_round:
            continue
        training = [cp for cp in checkpoints if races.index(cp.race) < round_index - 1]
        actuals = None
        if rng is not None:
            actuals = [rng.permutation(cp.actual) for cp in training]
        weight = fit_weight(training, actuals)
        for cp in (cp for cp in checkpoints if cp.race == race):
            current = score(rebuilt_positions(cp, None), cp.actual)
            learned = score(rebuilt_positions(cp, weight), cp.actual)
            corr_deltas.append(learned[0] - current[0])
            mae_deltas.append(learned[1] - current[1])
            fitted.append((cp, weight))
    return fitted, float(np.mean(corr_deltas)), float(np.mean(mae_deltas))


def report(root: Path, first_scored_round: int, draws: int, seed: int) -> None:
    """Print the sizing result for one replay root."""
    checkpoints = load_checkpoints(root)
    if not checkpoints:
        print(f"{root}: no Grand Prix rows with blend inputs")
        return
    agreement = np.mean([spearmanr(rebuilt_positions(cp, None), cp.saved)[0] for cp in checkpoints])
    fitted, corr_delta, mae_delta = walk_forward(checkpoints, first_scored_round)
    rng = np.random.default_rng(seed)
    null = [walk_forward(checkpoints, first_scored_round, rng)[1] for _ in range(draws)]
    null_95 = float(np.percentile(null, 95))
    hand_set = np.concatenate([cp.weight[~cp.penalised] for cp, _ in fitted])
    print(f"{root.name}: {len(checkpoints)} checkpoints, {len(fitted)} scored")
    print(f"  rebuilt hand-set order vs saved order, mean Spearman: {agreement:.4f}")
    print(
        f"  hand-set weight on scored rows: mean {hand_set.mean():.3f}, "
        f"range {hand_set.min():.3f} to {hand_set.max():.3f}"
    )
    weights = {cp.race: weight for cp, weight in fitted}
    print("  learned weight by race: " + ", ".join(f"{w:.3f}" for w in weights.values()))
    print(f"  learned minus hand-set: corr {corr_delta:+.4f}, MAE {mae_delta:+.4f}")
    print(f"  null corr delta ({draws} shuffles): 95th percentile {null_95:+.4f}")
    print(f"  passes (positive and above null): {corr_delta > max(0.0, null_95)}")


def main() -> int:
    """Run the sizing on each replay root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay-root", type=Path, nargs="+", required=True)
    parser.add_argument("--first-scored-round", type=int, default=5)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for root in args.replay_root:
        report(root, args.first_scored_round, args.draws, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
