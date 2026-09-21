"""Report per-round DNF label coverage across stored race-like actuals. Read-only.

The DNF calibration cannot be refitted until every completed round carries the same
retirement signal: a half-labelled season deflates any team that retired in an
unlabelled round while measuring its recent form correctly. This script answers
"which rounds carry the signal" without writing anything.

It reads through ``PredictionLogger``, so it sees whichever backend
``USE_DB_STORAGE`` selects. Load ``.env.local`` and confirm the logged
``ArtifactStore initialized with mode: db_only`` line before trusting a production
answer; in ``file_only`` mode this reports the local artifacts instead.

Usage:
  uv run python scripts/report_dnf_label_coverage.py --year 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dnf_label_coverage")

_RACE_LIKE_TARGETS = ("grand_prix_race", "sprint_race")


def _round_order(year: int) -> dict[str, int]:
    """Map race name to schedule round, so output is chronological rather than alphabetical."""
    from src.utils.weekend import get_schedule_rows

    return {str(row[0]): index + 1 for index, row in enumerate(get_schedule_rows(year))}


def _coverage_rows(predictions: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    """Summarise, per race, how many race-like actual rows carry an explicit DNF flag."""
    from src.utils.accuracy_targets import explicit_target_actuals

    by_race: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        race_name = str(prediction.get("metadata", {}).get("race_name", "")).strip()
        if not race_name:
            continue
        actuals = explicit_target_actuals(prediction)
        for target_key in _RACE_LIKE_TARGETS:
            rows = actuals.get(target_key)
            if not rows:
                continue
            # Several checkpoints of one weekend repeat the same classification; keep the
            # richest one seen rather than summing duplicates.
            flagged = sum(1 for row in rows if "dnf" in row)
            retirements = sum(1 for row in rows if row.get("dnf"))
            key = f"{race_name}::{target_key}"
            previous = by_race.get(key)
            if previous is None or flagged > previous["flagged"]:
                by_race[key] = {
                    "race": race_name,
                    "target": target_key,
                    "rows": len(rows),
                    "flagged": flagged,
                    "retirements": retirements,
                }

    order = _round_order(year)
    return sorted(by_race.values(), key=lambda row: (order.get(row["race"], 999), row["target"]))


def main() -> None:
    """CLI entrypoint for the read-only coverage report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    from src.utils.prediction_logger import PredictionLogger

    predictions = PredictionLogger().get_all_predictions(args.year)
    logger.info("Loaded %s prediction artifact(s) for %s", len(predictions), args.year)

    rows = _coverage_rows(predictions, args.year)
    if not rows:
        logger.warning("No race-like target actuals found for %s", args.year)
        return

    print(f"{'round':>5}  {'race':<34} {'target':<16} {'rows':>5} {'flagged':>8} {'retired':>8}")
    order = _round_order(args.year)
    complete = partial = unlabelled = 0
    for row in rows:
        if row["flagged"] == 0:
            unlabelled += 1
        elif row["flagged"] == row["rows"]:
            complete += 1
        else:
            partial += 1
        print(
            f"{order.get(row['race'], 0):>5}  {row['race'][:34]:<34} {row['target']:<16} "
            f"{row['rows']:>5} {row['flagged']:>8} {row['retirements']:>8}"
        )

    print(
        f"\nblocks: {len(rows)} | fully labelled: {complete} | partial: {partial} | "
        f"unlabelled: {unlabelled}"
    )
    print(f"total retirements recorded: {sum(row['retirements'] for row in rows)}")
    if partial or unlabelled:
        print("Coverage is NOT uniform: do not refit DNF calibration against this state.")


if __name__ == "__main__":
    main()
