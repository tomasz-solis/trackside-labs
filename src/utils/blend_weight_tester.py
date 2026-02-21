"""Helpers for evaluating FP blend weights against actual qualifying results."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _set_nested_config_value(config_obj: Any, dotted_key: str, value: Any) -> bool:
    """Set a nested config value for Config-like objects.

    Returns True when the value was updated, False otherwise.
    """
    storage = None
    for attr in ("_config", "_data"):
        candidate = getattr(config_obj, attr, None)
        if isinstance(candidate, dict):
            storage = candidate
            break

    if storage is None:
        return False

    keys = dotted_key.split(".")
    cursor = storage
    for key in keys[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[keys[-1]] = value
    return True


def _calculate_grid_mae(predicted: list[dict], actual: list[dict]) -> float:
    """Calculate mean absolute position error for shared drivers."""
    actual_positions = {entry["driver"]: entry["position"] for entry in actual}
    errors = []
    for prediction in predicted:
        driver = prediction.get("driver")
        if driver not in actual_positions:
            continue
        errors.append(abs(prediction["position"] - actual_positions[driver]))

    return (sum(errors) / len(errors)) if errors else 0.0


def test_blend_weights(
    predictor: Any,
    year: int,
    race_name: str,
    actual_quali_grid: list[dict],
    blend_weights: list[float] | None = None,
) -> dict[float, float]:
    """Evaluate candidate FP blend weights and return MAE for each."""
    if not actual_quali_grid:
        raise ValueError("actual_quali_grid cannot be empty")

    if blend_weights is None:
        blend_weights = [0.5, 0.6, 0.7, 0.8, 0.9]

    config_key = "baseline_predictor.qualifying.fp_blend_weight"
    original_weight = predictor.config.get(config_key, 0.7)
    results: dict[float, float] = {}

    for weight in blend_weights:
        updated = _set_nested_config_value(predictor.config, config_key, weight)
        if not updated:
            logger.warning("Could not set predictor blend weight dynamically; stopping sweep")
            break

        prediction = predictor.predict_qualifying(
            year=year,
            race_name=race_name,
            n_simulations=50,
        )
        mae = _calculate_grid_mae(prediction["grid"], actual_quali_grid)
        results[weight] = mae
        logger.info(f"Blend weight {weight:.2f}: MAE {mae:.2f}")

    _set_nested_config_value(predictor.config, config_key, original_weight)
    return results
