"""Opt-in legacy qualifying blend for retrospective research only.

Inputs are already selected saved rows and outcomes. Their ordering is supplied
by the caller and does not establish point-in-time eligibility.
"""

from __future__ import annotations

import math
from typing import Any

_QUALIFYING_STABILIZER_MODEL_WEIGHT = 0.40
_QUALIFYING_STABILIZER_NOMINAL_COVERAGE = 0.90
_QUALIFYING_STABILIZER_MIN_HISTORY_EVENTS = 3
_QUALIFYING_STABILIZER_DEFAULT_MARGIN = 1.0


def _position_by_driver(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Build a driver -> position lookup from ranked rows."""
    positions: dict[str, int] = {}
    for default_position, row in enumerate(rows, start=1):
        driver = str(row.get("driver", "")).strip()
        if not driver:
            continue
        try:
            position = int(row.get("position", default_position))
        except (TypeError, ValueError):
            position = default_position
        positions[driver] = position
    return positions


def _coerce_position(value: object, default: int) -> int:
    """Return an integer rank position with a stable fallback."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: object) -> float | None:
    """Return a float when a value is numeric enough for interval arithmetic."""
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _conformal_margin(residual_history: list[float]) -> float:
    """Return the nominal-coverage conformal interval margin from past residuals only."""
    if not residual_history:
        return _QUALIFYING_STABILIZER_DEFAULT_MARGIN
    ordered = sorted(float(value) for value in residual_history if math.isfinite(float(value)))
    if not ordered:
        return _QUALIFYING_STABILIZER_DEFAULT_MARGIN
    quantile_index = math.ceil(_QUALIFYING_STABILIZER_NOMINAL_COVERAGE * (len(ordered) + 1)) - 1
    quantile_index = min(max(quantile_index, 0), len(ordered) - 1)
    return float(ordered[quantile_index])


def _blend_qualifying_rows(
    rows: list[dict[str, Any]],
    *,
    previous_actual_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Blend saved rank with the preceding selected event's actual rank."""
    if not previous_actual_rows:
        return [dict(row) for row in rows]

    previous_positions = _position_by_driver(previous_actual_rows)
    scored_rows: list[dict[str, Any]] = []
    model_weight = _QUALIFYING_STABILIZER_MODEL_WEIGHT
    previous_weight = 1.0 - model_weight
    for default_position, row in enumerate(rows, start=1):
        adjusted = dict(row)
        model_position = _coerce_position(adjusted.get("position"), default_position)
        previous_position = previous_positions.get(
            str(adjusted.get("driver", "")).strip(),
            model_position,
        )
        adjusted["_counterfactual_model_position"] = model_position
        adjusted["_counterfactual_previous_position"] = previous_position
        adjusted["_counterfactual_score"] = (
            model_weight * model_position + previous_weight * previous_position
        )
        scored_rows.append(adjusted)

    scored_rows.sort(
        key=lambda row: (
            float(row["_counterfactual_score"]),
            int(row["_counterfactual_model_position"]),
            str(row.get("driver", "")),
        )
    )

    for new_position, row in enumerate(scored_rows, start=1):
        old_position = int(row.pop("_counterfactual_model_position"))
        previous_position = int(row.pop("_counterfactual_previous_position"))
        score = float(row.pop("_counterfactual_score"))
        position_delta = new_position - old_position
        row["released_position"] = old_position
        row["previous_selected_event_position"] = previous_position
        row["counterfactual_rank_score"] = round(score, 6)
        row["position"] = new_position

        median = _coerce_float(row.get("median_position"))
        if median is not None:
            row["median_position"] = median + position_delta

        lower = _coerce_float(row.get("p5"))
        upper = _coerce_float(row.get("p95"))
        if lower is not None and upper is not None:
            row["p5"] = lower + position_delta
            row["p95"] = upper + position_delta

    return scored_rows


def _qualifying_interval_residuals(
    predicted_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
) -> list[float]:
    """Return outside-interval distances before conformal widening."""
    actual_positions = _position_by_driver(actual_rows)
    residuals: list[float] = []
    for row in predicted_rows:
        driver = str(row.get("driver", "")).strip()
        if driver not in actual_positions:
            continue
        lower = _coerce_float(row.get("p5"))
        upper = _coerce_float(row.get("p95"))
        if lower is None or upper is None:
            continue
        actual_position = float(actual_positions[driver])
        residuals.append(max(lower - actual_position, actual_position - upper, 0.0))
    return residuals


def _widen_qualifying_intervals(
    rows: list[dict[str, Any]],
    *,
    margin: float,
) -> list[dict[str, Any]]:
    """Apply a conformal interval margin while preserving rank/median coherence."""
    field_size = len(rows)
    widened: list[dict[str, Any]] = []
    for default_position, row in enumerate(rows, start=1):
        adjusted = dict(row)
        position = _coerce_position(adjusted.get("position"), default_position)
        median = _coerce_float(adjusted.get("median_position"))
        if median is None:
            median = float(position)
        lower = _coerce_float(adjusted.get("p5"))
        upper = _coerce_float(adjusted.get("p95"))
        if lower is not None and upper is not None:
            widened_lower = max(1.0, min(lower, median, float(position)) - float(margin))
            widened_upper = min(
                float(field_size),
                max(upper, median, float(position)) + float(margin),
            )
            adjusted["p5"] = widened_lower
            adjusted["p95"] = widened_upper
            adjusted["median_position"] = median
            adjusted["conformal_interval_margin"] = float(margin)
        widened.append(adjusted)
    return widened


def apply_qualifying_counterfactual(
    events: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Blend saved ranks with preceding selected outcome and widen bands for research."""
    adjusted_rows: list[list[dict[str, Any]]] = []
    residual_history: list[float] = []
    previous_actual_rows: list[dict[str, Any]] | None = None
    event_margins: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        predicted_rows = event["predicted_rows"]
        actual_rows = event["actual_rows"]
        blended_rows = _blend_qualifying_rows(
            predicted_rows, previous_actual_rows=previous_actual_rows
        )
        if event_index < _QUALIFYING_STABILIZER_MIN_HISTORY_EVENTS:
            margin = _QUALIFYING_STABILIZER_DEFAULT_MARGIN
        else:
            margin = _conformal_margin(residual_history)
        adjusted_rows.append(_widen_qualifying_intervals(blended_rows, margin=margin))
        event_margins.append(
            {
                "race_name": event["race_name"],
                "target": event["target"],
                "margin": float(margin),
                "history_events_available": event_index,
            }
        )
        residual_history.extend(_qualifying_interval_residuals(blended_rows, actual_rows))
        previous_actual_rows = actual_rows
    return adjusted_rows, {
        "run_kind": "counterfactual",
        "selection_order_limitation": "Prior in selected report order; as-of eligibility unverified.",
        "qualifying_rank_stabilizer": {
            "method": "previous_selected_event_rank_blend",
            "model_rank_weight": _QUALIFYING_STABILIZER_MODEL_WEIGHT,
            "previous_selected_event_rank_weight": 1.0 - _QUALIFYING_STABILIZER_MODEL_WEIGHT,
            "uses_current_event_actuals": False,
        },
        "qualifying_interval_stabilizer": {
            "method": "preceding_selected_event_residual_widening",
            "nominal_coverage": _QUALIFYING_STABILIZER_NOMINAL_COVERAGE,
            "default_margin": _QUALIFYING_STABILIZER_DEFAULT_MARGIN,
            "min_history_events": _QUALIFYING_STABILIZER_MIN_HISTORY_EVENTS,
            "uses_current_event_actuals_for_current_margin": False,
        },
        "event_margins": event_margins,
    }
