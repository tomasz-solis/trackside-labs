#!/usr/bin/env python3
"""Compare accuracy-snapshot metrics between historical replay output roots.

Reads only the saved ``accuracy_snapshot/<year>/<race>/<checkpoint>/<target>.json``
files under each replay root -- it never re-runs the predictor. Qualifying and
race MAE deltas smaller than roughly 0.05 positions are indistinguishable from
simulator seed noise (``Baseline2026Predictor`` hardcodes ``seed=42`` through
most of the replay history), so ``--seed-floor`` lets a candidate delta be
gated against a measured seed-noise floor computed from two identically-coded
runs at different seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TARGETS = ("main_qualifying", "grand_prix_race", "sprint_race")
METRIC_ORDER = (
    "correlation",
    "overall_mae",
    "top_3_pct",
    "top_10_pct",
    "exact_accuracy",
    "within_1",
    "within_3",
)
LOWER_IS_BETTER = {"overall_mae"}
CHECKPOINT_ORDER = ("PRE", "FP1", "FP2", "FP3", "SQ")
BOOTSTRAP_RNG_SEED = 0

CheckpointKey = tuple[str, str]  # (race_name, checkpoint_session)
MetricsByCheckpoint = dict[CheckpointKey, dict[str, float]]


@dataclass
class MetricRow:
    """One metric's paired comparison for one target between two roots."""

    target: str
    metric: str
    n_paired: int
    baseline_mean: float
    candidate_mean: float
    delta_mean: float
    ci_lo: float
    ci_hi: float
    better: int
    worse: int
    tied: int
    verdict: str = ""


def _load_target_metrics(root: Path, year: int, target: str) -> MetricsByCheckpoint:
    """Load one target's metrics dict for every checkpoint under one replay root."""
    metrics_by_checkpoint: MetricsByCheckpoint = {}
    snapshot_dir = root / "accuracy_snapshot" / str(year)
    if not snapshot_dir.exists():
        return metrics_by_checkpoint
    for path in snapshot_dir.glob(f"*/*/{target}.json"):
        payload = json.loads(path.read_text())
        metadata = payload.get("metadata", {})
        race_name = str(metadata.get("race_name", "")).strip()
        checkpoint_session = str(metadata.get("checkpoint_session", "")).strip()
        if not race_name or not checkpoint_session:
            continue
        metrics_by_checkpoint[(race_name, checkpoint_session)] = payload.get("metrics", {})
    return metrics_by_checkpoint


_SUMMARY_CHECKPOINT_LINE = re.compile(r"^- (?P<race>.+) (?P<session>[A-Z0-9]+):$")


def _schedule_order(root: Path, fallback_race_names: set[str]) -> list[str]:
    """Return race names in replay (chronological) order.

    Reads ``reports/summary.md`` from the baseline root. Falls back to sorted
    race names if that file is absent, per spec (never calls
    ``src.utils.weekend.get_schedule_rows``, which reaches FastF1 and hangs
    offline).
    """
    summary_path = root / "reports" / "summary.md"
    if not summary_path.exists():
        return sorted(fallback_race_names)
    order: list[str] = []
    seen: set[str] = set()
    for line in summary_path.read_text().splitlines():
        match = _SUMMARY_CHECKPOINT_LINE.match(line)
        if not match:
            continue
        race_name = match.group("race")
        if race_name not in seen:
            seen.add(race_name)
            order.append(race_name)
    return order or sorted(fallback_race_names)


def _round_sort_key(schedule_order: list[str], key: CheckpointKey) -> tuple[int, int, str]:
    """Sort key placing a checkpoint in schedule (race, then session) order."""
    race_name, checkpoint_session = key
    race_index = (
        schedule_order.index(race_name) if race_name in schedule_order else len(schedule_order)
    )
    session_index = (
        CHECKPOINT_ORDER.index(checkpoint_session)
        if checkpoint_session in CHECKPOINT_ORDER
        else len(CHECKPOINT_ORDER)
    )
    return (race_index, session_index, race_name)


def _bootstrap_ci(
    deltas: np.ndarray, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a 95% paired bootstrap CI for the mean of ``deltas``."""
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0)
    sample_indices = rng.integers(0, n, size=(iterations, n))
    resampled_means = deltas[sample_indices].mean(axis=1)
    lo, hi = np.percentile(resampled_means, [2.5, 97.5])
    return float(lo), float(hi)


def compare_metric(
    paired_keys: list[CheckpointKey],
    baseline_by_key: MetricsByCheckpoint,
    candidate_by_key: MetricsByCheckpoint,
    target: str,
    metric: str,
    iterations: int,
    rng: np.random.Generator,
) -> MetricRow | None:
    """Compare one metric across every paired checkpoint. None if unavailable."""
    baseline_vals: list[float] = []
    candidate_vals: list[float] = []
    for key in paired_keys:
        baseline_metrics = baseline_by_key[key]
        candidate_metrics = candidate_by_key[key]
        if metric not in baseline_metrics or metric not in candidate_metrics:
            continue
        baseline_vals.append(float(baseline_metrics[metric]))
        candidate_vals.append(float(candidate_metrics[metric]))
    if not baseline_vals:
        return None

    baseline_arr = np.array(baseline_vals)
    candidate_arr = np.array(candidate_vals)
    deltas = candidate_arr - baseline_arr
    lower_is_better = metric in LOWER_IS_BETTER

    better = worse = tied = 0
    for baseline_val, candidate_val in zip(baseline_vals, candidate_vals, strict=True):
        if math.isclose(baseline_val, candidate_val, rel_tol=1e-9, abs_tol=1e-9):
            tied += 1
        elif (candidate_val < baseline_val) == lower_is_better:
            better += 1
        else:
            worse += 1

    ci_lo, ci_hi = _bootstrap_ci(deltas, iterations, rng)
    return MetricRow(
        target=target,
        metric=metric,
        n_paired=len(baseline_vals),
        baseline_mean=float(baseline_arr.mean()),
        candidate_mean=float(candidate_arr.mean()),
        delta_mean=float(deltas.mean()),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        better=better,
        worse=worse,
        tied=tied,
    )


def analyze_arm(
    baseline_by_target: dict[str, MetricsByCheckpoint],
    candidate_by_target: dict[str, MetricsByCheckpoint],
    targets: list[str],
    iterations: int,
) -> dict[str, list[MetricRow]]:
    """Compare a candidate root against a baseline root, target by target."""
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    rows_by_target: dict[str, list[MetricRow]] = {}
    for target in targets:
        baseline_by_key = baseline_by_target.get(target, {})
        candidate_by_key = candidate_by_target.get(target, {})
        paired_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
        rows: list[MetricRow] = []
        for metric in METRIC_ORDER:
            row = compare_metric(
                paired_keys, baseline_by_key, candidate_by_key, target, metric, iterations, rng
            )
            if row is not None:
                rows.append(row)
        rows_by_target[target] = rows
    return rows_by_target


def assign_verdicts(
    rows_by_target: dict[str, list[MetricRow]],
    floor_rows_by_target: dict[str, list[MetricRow]] | None,
) -> None:
    """Set ``verdict`` on every row in place, using the seed floor when given.

    The floor threshold is the widest absolute bound of the seed pair's confidence
    interval, not its point estimate. One seed pair's observed shift is a single draw;
    the plausible size of a seed-induced shift is what the interval spans, and gating on
    the point estimate alone lets noise-sized effects through as real.
    """
    floor_thresholds: dict[tuple[str, str], float] = {}
    if floor_rows_by_target is not None:
        for target, rows in floor_rows_by_target.items():
            for row in rows:
                floor_thresholds[(target, row.metric)] = max(abs(row.ci_lo), abs(row.ci_hi))

    for target, rows in rows_by_target.items():
        for row in rows:
            # Every checkpoint tied means the two runs produced the same predictions.
            # That is a definitive "this changed nothing", not a measurement that failed
            # to resolve - the ledger calls it `never activated` and it must not be
            # buried under the seed floor.
            if row.n_paired > 0 and row.tied == row.n_paired:
                row.verdict = "identical (never activated)"
                continue
            floor_threshold = floor_thresholds.get((target, row.metric))
            if floor_threshold is not None and abs(row.delta_mean) <= floor_threshold:
                row.verdict = "unresolvable (below seed floor)"
                continue
            if row.ci_lo <= 0.0 <= row.ci_hi:
                row.verdict = "noise"
                continue
            lower_is_better = row.metric in LOWER_IS_BETTER
            is_better = row.delta_mean < 0 if lower_is_better else row.delta_mean > 0
            row.verdict = "better" if is_better else "worse"


def _format_row(row: MetricRow, label: str = "") -> str:
    """Render one metric row as a single output line."""
    prefix = f"[{label}] " if label else ""
    return (
        f"  {prefix}{row.metric:<14} n={row.n_paired:<3} "
        f"baseline={row.baseline_mean:.4f} candidate={row.candidate_mean:.4f} "
        f"delta={row.delta_mean:+.4f} CI=[{row.ci_lo:+.4f}, {row.ci_hi:+.4f}] "
        f"better/worse/tied={row.better}/{row.worse}/{row.tied} -> {row.verdict}"
    )


def _print_movers(
    baseline_by_key: MetricsByCheckpoint,
    candidate_by_key: MetricsByCheckpoint,
    metric: str,
    schedule_order: list[str],
) -> None:
    """Print the non-tied checkpoints for one metric, sorted by schedule order."""
    paired_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    movers = [
        key
        for key in paired_keys
        if metric in baseline_by_key[key]
        and metric in candidate_by_key[key]
        and not math.isclose(
            baseline_by_key[key][metric], candidate_by_key[key][metric], rel_tol=1e-9, abs_tol=1e-9
        )
    ]
    if not movers:
        return
    movers.sort(key=lambda key: _round_sort_key(schedule_order, key))
    print(f"    movers ({metric}):")
    for race_name, checkpoint_session in movers:
        baseline_val = baseline_by_key[(race_name, checkpoint_session)][metric]
        candidate_val = candidate_by_key[(race_name, checkpoint_session)][metric]
        delta = candidate_val - baseline_val
        print(
            f"      {race_name} {checkpoint_session}: "
            f"baseline={baseline_val:.4f} candidate={candidate_val:.4f} delta={delta:+.4f}"
        )


def _print_rows_by_target(rows_by_target: dict[str, list[MetricRow]]) -> None:
    """Print PRIMARY/SECONDARY/other metric rows for every target."""
    for target, rows in rows_by_target.items():
        if not rows:
            print(f"  {target}: no paired checkpoints, skipped")
            continue
        print(f"  {target} (paired checkpoints: {rows[0].n_paired}):")
        rows_by_metric = {row.metric: row for row in rows}
        if "correlation" in rows_by_metric:
            print(_format_row(rows_by_metric["correlation"], label="PRIMARY"))
        if "overall_mae" in rows_by_metric:
            print(_format_row(rows_by_metric["overall_mae"], label="SECONDARY"))
        other_rows = [row for row in rows if row.metric not in ("correlation", "overall_mae")]
        if other_rows:
            print("  -- other metrics --")
            for row in other_rows:
                print(_format_row(row))


def _load_metrics_by_target(
    root: Path, year: int, targets: list[str]
) -> dict[str, MetricsByCheckpoint]:
    """Load every requested target's metrics for one replay root."""
    return {target: _load_target_metrics(root, year, target) for target in targets}


def _has_any_pairs(rows_by_target: dict[str, list[MetricRow]]) -> bool:
    """Whether at least one target produced a paired comparison."""
    return any(rows for rows in rows_by_target.values())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the replay-arm comparison tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare accuracy-snapshot metrics between historical replay output "
            "roots, without re-running the predictor."
        )
    )
    parser.add_argument("--baseline", required=True, help="Baseline replay output root.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate replay output root. Repeatable.",
    )
    parser.add_argument(
        "--seed-floor",
        nargs=2,
        metavar=("ROOT_A", "ROOT_B"),
        help=(
            "Two replay roots of IDENTICAL code at different seeds, used to "
            "compute a seed-noise floor that candidate deltas are gated against."
        ),
    )
    parser.add_argument("--year", type=int, default=2026, help="Season year to compare.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        choices=list(DEFAULT_TARGETS),
        help="Prediction targets to compare.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=4000, help="Bootstrap resample count for the CI."
    )
    return parser


def main() -> int:
    """Run the replay-arm comparison and print a report to stdout."""
    args = build_parser().parse_args()
    targets: list[str] = list(args.targets)

    baseline_root = Path(args.baseline)
    candidate_roots = [Path(candidate) for candidate in args.candidate]
    floor_roots = [Path(root) for root in args.seed_floor] if args.seed_floor else None

    roots_to_check = [baseline_root, *candidate_roots, *(floor_roots or [])]
    missing_roots = [str(root) for root in roots_to_check if not root.exists()]
    if missing_roots:
        print(f"Missing replay root(s): {', '.join(missing_roots)}", file=sys.stderr)
        return 2

    baseline_by_target = _load_metrics_by_target(baseline_root, args.year, targets)
    fallback_race_names = {key[0] for metrics in baseline_by_target.values() for key in metrics}
    schedule_order = _schedule_order(baseline_root, fallback_race_names)

    had_error = False

    floor_rows_by_target: dict[str, list[MetricRow]] | None = None
    if floor_roots is not None:
        floor_a_by_target = _load_metrics_by_target(floor_roots[0], args.year, targets)
        floor_b_by_target = _load_metrics_by_target(floor_roots[1], args.year, targets)
        floor_rows_by_target = analyze_arm(
            floor_a_by_target, floor_b_by_target, targets, args.bootstrap
        )
        if not _has_any_pairs(floor_rows_by_target):
            print(
                f"No paired checkpoints between seed-floor roots {floor_roots[0]} and "
                f"{floor_roots[1]}",
                file=sys.stderr,
            )
            had_error = True
        assign_verdicts(floor_rows_by_target, None)
        print(f"=== SEED FLOOR: {floor_roots[0]} vs {floor_roots[1]} ===")
        _print_rows_by_target(floor_rows_by_target)
        print()
    else:
        print(
            "WARNING: no --seed-floor supplied; deltas cannot be separated from "
            "simulator seed noise (see historical_replay seed threading)."
        )
        print()

    for candidate_root in candidate_roots:
        candidate_by_target = _load_metrics_by_target(candidate_root, args.year, targets)
        rows_by_target = analyze_arm(
            baseline_by_target, candidate_by_target, targets, args.bootstrap
        )
        if not _has_any_pairs(rows_by_target):
            print(
                f"No paired checkpoints between {baseline_root} and {candidate_root}",
                file=sys.stderr,
            )
            had_error = True
            continue
        assign_verdicts(rows_by_target, floor_rows_by_target)
        print(f"=== CANDIDATE: {candidate_root} vs BASELINE: {baseline_root} ===")
        _print_rows_by_target(rows_by_target)
        for target in targets:
            if "correlation" in {row.metric for row in rows_by_target.get(target, [])}:
                _print_movers(
                    baseline_by_target.get(target, {}),
                    candidate_by_target.get(target, {}),
                    "correlation",
                    schedule_order,
                )
        print()

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
