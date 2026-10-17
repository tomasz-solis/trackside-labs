"""Deterministic scoring of a saved, released forecast projection.

This is a scoring boundary for legacy artifacts, not forecast persistence or a
publication identity. The supplied rows must be the saved rows used by serving.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from src.analysis.model_evaluation import (
    build_calibration_components,
    compute_prediction_accuracy,
    summarize_calibration_components,
)

EVALUATION_SPEC_VERSION = "frozen_released_v1"


def copy_canonical_rows(rows: object, *, label: str, ranked_order: bool) -> list[dict[str, Any]]:
    """Copy stored rows exactly, rejecting repairs that would change their meaning."""
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    copied = copy.deepcopy(rows)
    seen_drivers: set[str] = set()
    seen_positions: set[int] = set()
    for index, row in enumerate(copied, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        driver = row.get("driver")
        team = row.get("team")
        position = row.get("position")
        if not isinstance(driver, str) or not driver or driver != driver.strip():
            raise ValueError(f"{label}[{index}].driver must be a nonempty canonical string")
        if not isinstance(team, str) or not team or team != team.strip():
            raise ValueError(f"{label}[{index}].team must be a nonempty canonical string")
        if type(position) is not int or position < 1:
            raise ValueError(f"{label}[{index}].position must be a positive integer")
        if driver in seen_drivers or position in seen_positions:
            raise ValueError(f"{label} has duplicate driver or position")
        if ranked_order and position != index:
            raise ValueError(f"{label} must be ordered by contiguous rank")
        if ranked_order:
            lower = row.get("p5")
            upper = row.get("p95")
            if (lower is None) != (upper is None):
                raise ValueError(f"{label}[{index}] must have both p5 and p95")
            if lower is not None and upper is not None:
                for field, value in (("p5", lower), ("p95", upper)):
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError(f"{label}[{index}].{field} must be finite numeric")
                if lower <= 0 or upper < lower:
                    raise ValueError(f"{label}[{index}] has an invalid p5/p95 interval")
            for field, value in row.items():
                if field.endswith("_probability") or field == "dnf_risk":
                    if value is not None and (
                        type(value) not in (int, float) or not math.isfinite(value)
                    ):
                        raise ValueError(f"{label}[{index}].{field} must be finite numeric")
                    if value is not None and not 0 <= value <= 100:
                        raise ValueError(f"{label}[{index}].{field} must be between 0 and 100")
        seen_drivers.add(driver)
        seen_positions.add(position)
    return copied


def _content_hash(value: dict[str, Any]) -> str:
    """Hash the forecast projection without outcome or scoring state."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def score_frozen_forecast(
    forecast_core: dict[str, Any],
    outcome_revision: dict[str, Any],
    evaluation_spec_version: str,
) -> dict[str, Any]:
    """Score exact saved rows against one outcome revision without mutating inputs.

    Expected projection keys are event_id, target, predicted_rows and optional
    checkpoint, information_cutoff_at and forecast_id. Outcome keys are event_id,
    target, actual_rows and optional revision_id. Missing legacy IDs remain null.
    """
    if evaluation_spec_version != EVALUATION_SPEC_VERSION:
        raise ValueError(f"Unsupported evaluation spec version: {evaluation_spec_version}")
    event_id = str(forecast_core.get("event_id") or "").strip()
    target = str(forecast_core.get("target") or "").strip()
    if not event_id or not target:
        raise ValueError("Forecast event_id and target are required")
    if event_id != str(outcome_revision.get("event_id") or "").strip():
        raise ValueError("Forecast and outcome event_id differ")
    if target != str(outcome_revision.get("target") or "").strip():
        raise ValueError("Forecast and outcome target differ")
    frozen = copy.deepcopy(forecast_core)
    released_rows = copy_canonical_rows(
        forecast_core.get("predicted_rows"), label="predicted_rows", ranked_order=True
    )
    actual_rows = copy_canonical_rows(
        outcome_revision.get("actual_rows"), label="actual_rows", ranked_order=False
    )
    core_hash = _content_hash(frozen)
    identity = {
        "event_id": event_id,
        "target": target,
        "checkpoint": forecast_core.get("checkpoint"),
        "information_cutoff_at": forecast_core.get("information_cutoff_at"),
        "predicted_at": forecast_core.get("predicted_at"),
        "forecast_id": forecast_core.get("forecast_id"),
        "run_id": forecast_core.get("run_id"),
        "model_version": forecast_core.get("model_version"),
        "forecast_core_hash": core_hash,
        "outcome_revision_id": outcome_revision.get("revision_id"),
        "outcome_content_hash": _content_hash(
            {"event_id": event_id, "target": target, "actual_rows": actual_rows}
        ),
        "evaluation_spec_version": evaluation_spec_version,
    }
    if not released_rows or not actual_rows:
        return {
            **identity,
            "status": "unscorable",
            "reason": "empty_predicted_or_actual_rows",
            "released_rows": released_rows,
            "actual_rows": actual_rows,
            "rank_metrics": None,
            "calibration_metrics": None,
        }
    rank = compute_prediction_accuracy(released_rows, actual_rows)
    if rank["compared_drivers"] == 0:
        return {
            **identity,
            "status": "unscorable",
            "reason": "no_matching_drivers",
            "released_rows": released_rows,
            "actual_rows": actual_rows,
            "rank_metrics": None,
            "calibration_metrics": None,
        }
    actual_by_driver: dict[str, int] = {row["driver"]: row["position"] for row in actual_rows}
    paired_bands = [
        (float(row["p5"]), float(row["p95"]), actual_by_driver[row["driver"]])
        for row in released_rows
        if row.get("p5") is not None
        and row.get("p95") is not None
        and row.get("driver") in actual_by_driver
    ]
    components = build_calibration_components(
        [(lower, upper) for lower, upper, _ in paired_bands],
        [actual for _, _, actual in paired_bands],
    )
    calibration = summarize_calibration_components([components])
    return {
        **identity,
        "status": "scored",
        "released_rows": released_rows,
        "actual_rows": actual_rows,
        "rank_metrics": rank,
        "calibration_metrics": calibration,
        "_calibration_components": components,
    }
