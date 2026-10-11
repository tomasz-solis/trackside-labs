"""Rebuild the adaptive learning state from saved forecasts and their actual results.

The state (per-driver errors, teammate gaps, interval residuals) used to live in a
local file only, so a deployment without a shared disk never kept it. This replays
every saved forecast that has actuals, oldest first, through the same update the
live path uses, and prints what was learned.

Dry run by default. ``--apply`` saves the rebuilt state to the artifact store,
which changes the next forecasts. Do not apply during a race weekend.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.persistence.artifact_store import ArtifactStore  # noqa: E402
from src.systems.systematic_learning import (  # noqa: E402
    _ARTIFACT_KEY,
    _ARTIFACT_TYPE,
    SystematicLearningSystem,
)
from src.utils.env_file import load_env_file  # noqa: E402
from src.utils.prediction_logger import PredictionLogger  # noqa: E402


def rebuild(predictions: list[dict], state_file: Path) -> tuple[SystematicLearningSystem, dict]:
    """Feed forecasts oldest first into a fresh learning system and count the outcomes."""
    system = SystematicLearningSystem(state_file=state_file)
    counts = {"learned": 0, "no_actuals_or_too_few_drivers": 0}
    ordered = sorted(predictions, key=lambda p: str(p.get("metadata", {}).get("predicted_at", "")))
    for prediction in ordered:
        summary = system.update_from_prediction_record(prediction)
        if summary.get("skipped"):
            reason = str(summary.get("skip_reason"))
            counts[reason] = counts.get(reason, 0) + 1
        elif summary.get("sessions_updated", 0) > 0:
            counts["learned"] += 1
        else:
            counts["no_actuals_or_too_few_drivers"] += 1
    return system, counts


def main() -> int:
    """Rebuild the state, print a summary and optionally save it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Save to the artifact store")
    args = parser.parse_args()
    if args.env_file is not None:
        load_env_file(args.env_file)

    predictions = PredictionLogger().get_all_predictions(args.year)
    with tempfile.TemporaryDirectory() as tmp:
        system, counts = rebuild(predictions, Path(tmp) / "learning_state.json")

    adaptive = system.state["adaptive_calibration"]
    print(f"Forecasts read: {len(predictions)}. Outcomes: {counts}")
    for session in ("qualifying", "race", "sprint"):
        radius = system.get_interval_calibration_summary(session)
        print(
            f"{session}: {len(adaptive['driver_position_error'].get(session, {}))} drivers, "
            f"{int(radius['sample_count'])} interval samples, "
            f"learned radius {radius['learned_radius']:.1f}"
        )

    if not args.apply:
        print("Dry run. Nothing saved. Pass --apply to save.")
        return 0
    ArtifactStore().save_artifact(_ARTIFACT_TYPE, _ARTIFACT_KEY, system.state)
    print(f"Saved {_ARTIFACT_TYPE}::{_ARTIFACT_KEY}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
