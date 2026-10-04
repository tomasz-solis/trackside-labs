"""Fit calibrated 50% likely-range offsets (q25/q75) for race and qualifying predictions.

Reads embedded actuals from replay checkpoint prediction files and computes,
per target (`grand_prix_race`, `sprint_race`, `main_qualifying`) and per
predicted-position bucket (1-5, 6-10, 11-16, 17-22), the 25th/75th percentile
of `actual_position - predicted_position`. Race targets use actual finishers
only (dnf is False). The point is the row's shown `position`, so the band
always sits around the place the dashboard prints.

The offsets pair with a row's shown position at output time:
`likely_lo = position + q25`, `likely_hi = position + q75`.

Also reports leave-one-race-out coverage of the fitted band: for each race,
fit q25/q75 on the other races and check whether the held-out race's
residuals land inside that band, pooling the result across races. This
validates the band out of sample instead of on its own fitting data.

Usage:
    uv run python scripts/fit_race_band_quantiles.py
    uv run python scripts/fit_race_band_quantiles.py --replay-root data/historical_replay_m1s42_r14 --year 2026
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

BUCKET_EDGES = [(1, 5), (6, 10), (11, 16), (17, 22)]
TARGETS = ("grand_prix_race", "sprint_race", "main_qualifying")
MIN_SPRINT_BUCKET_N = 40


def bucket_of(rank: int) -> str:
    """Return the predicted-position bucket label for a 1-based rank."""
    for lo, hi in BUCKET_EDGES:
        if lo <= rank <= hi:
            return f"{lo}-{hi}"
    return "23+"


class Residual(NamedTuple):
    """One finisher's predicted-vs-actual residual, tagged with race and bucket."""

    race: str
    bucket: str
    residual: float


def load_residuals(replay_root: Path, year: int, target: str) -> list[Residual]:
    """Pair predicted rows with embedded actuals and return finisher residuals.

    Only actual finishers (``dnf`` False) are used. ``residual = actual_position
    - position``, bucketed by the same predicted ``position``.
    """
    residuals: list[Residual] = []
    pred_root = replay_root / "predictions" / str(year)
    if not pred_root.is_dir():
        return residuals
    for race_dir in sorted(pred_root.iterdir()):
        if not race_dir.is_dir():
            continue
        for checkpoint_file in sorted(race_dir.glob("*.json")):
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            target_data = data.get("targets", {}).get(target)
            actuals = data.get("actuals", {}).get("targets", {}).get(target)
            if not target_data or not actuals:
                continue
            actual_by_driver = {row["driver"]: row for row in actuals}
            for row in target_data.get("predicted_order", []):
                actual = actual_by_driver.get(row.get("driver"))
                if actual is None or bool(actual.get("dnf", False)):
                    continue
                pred_rank = row.get("position")
                actual_pos = actual.get("position")
                if pred_rank is None or actual_pos is None:
                    continue
                residuals.append(
                    Residual(
                        race=race_dir.name,
                        bucket=bucket_of(int(pred_rank)),
                        residual=float(actual_pos) - float(pred_rank),
                    )
                )
    return residuals


def fit_quantiles(residuals: list[Residual]) -> dict[str, dict[str, float]]:
    """Return ``{bucket: {n, q25, q75}}`` fitted on the given residuals."""
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for residual in residuals:
        by_bucket[residual.bucket].append(residual.residual)
    return {
        bucket: {
            "n": len(values),
            "q25": round(float(np.percentile(values, 25)), 2),
            "q75": round(float(np.percentile(values, 75)), 2),
        }
        for bucket, values in by_bucket.items()
    }


def displayed_offsets(q25: float, q75: float) -> tuple[int, int]:
    """Return the whole-place offsets the dashboard shows for a fitted q25/q75.

    Mirrors ``assign_likely_range``: floor/ceil to whole places and always
    include the shown position (offset 0).
    """
    return min(0, math.floor(q25)), max(0, math.ceil(q75))


def leave_one_race_out_coverage(residuals: list[Residual]) -> dict[str, float]:
    """Return pooled out-of-sample coverage (%) of the band as displayed, per bucket.

    For each race, fit q25/q75 on the other races and check whether the
    held-out race's residuals fall inside that band, pooling hits across
    every held-out race.
    """
    races = sorted({residual.race for residual in residuals})
    covered_by_bucket: dict[str, list[bool]] = defaultdict(list)
    for held_out in races:
        train = [r for r in residuals if r.race != held_out]
        test = [r for r in residuals if r.race == held_out]
        if not train or not test:
            continue
        fitted = fit_quantiles(train)
        for residual in test:
            bounds = fitted.get(residual.bucket)
            if bounds is None:
                continue
            lo_offset, hi_offset = displayed_offsets(bounds["q25"], bounds["q75"])
            covered_by_bucket[residual.bucket].append(lo_offset <= residual.residual <= hi_offset)
    return {
        bucket: round(100.0 * sum(flags) / len(flags), 1)
        for bucket, flags in covered_by_bucket.items()
        if flags
    }


def _bucket_sort_key(bucket: str) -> tuple[int, int]:
    """Sort bucket labels numerically (``"23+"`` last)."""
    if bucket == "23+":
        return (23, 999)
    lo, _, hi = bucket.partition("-")
    return (int(lo), int(hi))


def _format_yaml_snippet(
    race_table: dict[str, dict[str, float]],
    sprint_table: dict[str, dict[str, float]],
    qualifying_table: dict[str, dict[str, float]],
    *,
    replay_root: Path,
    year: int,
    fitted_date: str,
) -> str:
    """Format the fitted per-bucket q25/q75 tables as a config YAML snippet."""
    lines = [
        "baseline_predictor:",
        "  race:",
        "    likely_range:",
        f"      # Fitted {fitted_date} by scripts/fit_race_band_quantiles.py",
        f"      # against {replay_root.as_posix()} ({year})",
    ]
    tables = (("race", race_table), ("sprint", sprint_table), ("qualifying", qualifying_table))
    for table_name, table in tables:
        lines.append(f"      {table_name}:")
        buckets = sorted((b for b in table if b != "23+"), key=_bucket_sort_key)
        for bucket in buckets:
            stats = table[bucket]
            lines.append(
                f'        "{bucket}": {{q25: {stats["q25"]:.2f}, q75: {stats["q75"]:.2f}}}'
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the quantile fitter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=Path("data/historical_replay_m1s42_r14"),
        help="Replay root containing predictions/<year>/<race>/*.json.",
    )
    parser.add_argument("--year", type=int, default=2026, help="Replay season to fit on.")
    parser.add_argument(
        "--fitted-date",
        default="2026-09-27",
        help="Date stamp to embed in the printed YAML comment.",
    )
    return parser


def main() -> None:
    """Fit likely-range quantiles per target and print the report + YAML snippet."""
    args = build_parser().parse_args()
    tables: dict[str, dict[str, dict[str, float]]] = {}
    coverage: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        residuals = load_residuals(args.replay_root, args.year, target)
        tables[target] = fit_quantiles(residuals)
        coverage[target] = leave_one_race_out_coverage(residuals)
        print(f"\n=== {target} ===")
        for bucket in sorted(tables[target], key=_bucket_sort_key):
            stats = tables[target][bucket]
            cov = coverage[target].get(bucket)
            cov_str = f"{cov:.1f}%" if cov is not None else "n/a"
            print(
                f"  {bucket:>6}: n={stats['n']:>4}  q25={stats['q25']:+.2f}  "
                f"q75={stats['q75']:+.2f}  loro_coverage={cov_str}"
            )

    race_table = tables.get("grand_prix_race", {})
    sprint_table = tables.get("sprint_race", {})
    required_buckets = {f"{lo}-{hi}" for lo, hi in BUCKET_EDGES}
    sprint_ok = required_buckets <= set(sprint_table) and all(
        sprint_table[bucket]["n"] >= MIN_SPRINT_BUCKET_N for bucket in required_buckets
    )
    if not sprint_ok:
        print(
            f"\nSprint bucket n below {MIN_SPRINT_BUCKET_N} (or bucket missing); "
            "reusing the race table for sprint."
        )
        sprint_table = race_table

    print("\n# Paste into config/default.yaml under baseline_predictor.race.likely_range")
    print(
        _format_yaml_snippet(
            race_table,
            sprint_table,
            tables.get("main_qualifying", {}),
            replay_root=args.replay_root,
            year=args.year,
            fitted_date=args.fitted_date,
        )
    )


if __name__ == "__main__":
    main()
