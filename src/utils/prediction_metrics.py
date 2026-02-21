"""Calculates accuracy metrics by comparing predictions to actual results."""

import logging
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.utils.driver_name_mapper import DriverNameMapper

logger = logging.getLogger(__name__)


class PredictionMetrics:
    """Calculates various accuracy metrics for race predictions."""

    @staticmethod
    def position_accuracy(predicted: list[dict], actual: list[dict]) -> float:
        """Calculate exact position accuracy (% of drivers in correct position)."""
        if not predicted or not actual:
            return 0.0

        # Normalize driver names
        predicted_norm = DriverNameMapper.normalize_result_list(predicted)
        actual_norm = DriverNameMapper.normalize_result_list(actual)

        # Build lookup: driver -> actual position
        actual_positions = {r["driver"]: r["position"] for r in actual_norm}

        # Check for missing drivers (substitutions)
        missing_from_actual = [
            p["driver"] for p in predicted_norm if p["driver"] not in actual_positions
        ]
        if missing_from_actual:
            logger.warning(
                f"Drivers in prediction but not in actuals "
                f"(possible substitution): {missing_from_actual}"
            )

        # Count exact matches
        correct = sum(
            1
            for p in predicted_norm
            if p["driver"] in actual_positions and actual_positions[p["driver"]] == p["position"]
        )

        return (correct / len(predicted_norm)) * 100

    @staticmethod
    def mean_absolute_error(predicted: list[dict], actual: list[dict]) -> float:
        """Calculate Mean Absolute Error (MAE) of position predictions."""
        if not predicted or not actual:
            return float("inf")

        # Normalize driver names
        predicted_norm = DriverNameMapper.normalize_result_list(predicted)
        actual_norm = DriverNameMapper.normalize_result_list(actual)

        # Build lookup: driver -> actual position
        actual_positions = {r["driver"]: r["position"] for r in actual_norm}

        # Calculate absolute errors for drivers present in both lists
        errors = [
            abs(p["position"] - actual_positions[p["driver"]])
            for p in predicted_norm
            if p["driver"] in actual_positions
        ]

        if not errors:
            return float("in")

        return np.mean(errors)

    @staticmethod
    def within_n_positions(predicted: list[dict], actual: list[dict], n: int = 1) -> float:
        """Calculate % of predictions within N positions of actual."""
        if not predicted or not actual:
            return 0.0

        # Normalize driver names
        predicted_norm = DriverNameMapper.normalize_result_list(predicted)
        actual_norm = DriverNameMapper.normalize_result_list(actual)

        # Build lookup: driver -> actual position
        actual_positions = {r["driver"]: r["position"] for r in actual_norm}

        # Count predictions within N positions
        within_n = sum(
            1
            for p in predicted_norm
            if p["driver"] in actual_positions
            and abs(p["position"] - actual_positions[p["driver"]]) <= n
        )

        return (within_n / len(predicted_norm)) * 100

    @staticmethod
    def correlation_coefficient(predicted: list[dict], actual: list[dict]) -> float:
        """Calculate Spearman correlation between predicted and actual positions."""
        if not predicted or not actual:
            return 0.0

        # Normalize driver names
        predicted_norm = DriverNameMapper.normalize_result_list(predicted)
        actual_norm = DriverNameMapper.normalize_result_list(actual)

        # Build lookup: driver -> actual position
        actual_positions = {r["driver"]: r["position"] for r in actual_norm}

        # Extract positions for drivers present in both lists
        pred_pos = []
        actual_pos = []
        for p in predicted_norm:
            if p["driver"] in actual_positions:
                pred_pos.append(p["position"])
                actual_pos.append(actual_positions[p["driver"]])

        if len(pred_pos) < 2:
            return 0.0

        # Calculate Spearman correlation
        corr, _ = spearmanr(pred_pos, actual_pos)
        return corr if not np.isnan(corr) else 0.0

    @staticmethod
    def podium_accuracy(predicted: list[dict], actual: list[dict]) -> dict[str, Any]:
        """Calculate podium prediction accuracy."""
        if not predicted or not actual:
            return {"correct_drivers": 0, "correct_positions": 0, "accuracy": 0.0}

        # Normalize driver names
        predicted_norm = DriverNameMapper.normalize_result_list(predicted)
        actual_norm = DriverNameMapper.normalize_result_list(actual)

        # Get top 3 from each
        pred_podium = {p["driver"]: p["position"] for p in predicted_norm[:3]}
        actual_podium = {r["driver"]: r["position"] for r in actual_norm[:3]}

        # Count correct drivers on podium (any position)
        correct_drivers = sum(1 for driver in pred_podium if driver in actual_podium)

        # Count correct positions (driver in correct podium spot)
        correct_positions = sum(
            1
            for driver, pos in pred_podium.items()
            if driver in actual_podium and actual_podium[driver] == pos
        )

        return {
            "correct_drivers": correct_drivers,
            "correct_positions": correct_positions,
            "accuracy": (correct_drivers / 3) * 100,
        }

    @staticmethod
    def winner_accuracy(predicted: list[dict], actual: list[dict]) -> bool:
        """Check if race winner was predicted correctly."""
        if not predicted or not actual:
            return False

        # Normalize driver names
        predicted_norm = DriverNameMapper.normalize_result_list(predicted)
        actual_norm = DriverNameMapper.normalize_result_list(actual)

        return predicted_norm[0]["driver"] == actual_norm[0]["driver"]

    @staticmethod
    def calculate_all_metrics(
        prediction_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Calculate all metrics for a prediction with actuals."""
        # Check if actuals are available
        if prediction_data.get("actuals") is None:
            return None

        quali_actual = prediction_data["actuals"].get("qualifying")
        race_actual = prediction_data["actuals"].get("race")

        if quali_actual is None and race_actual is None:
            return None

        metrics = {"metadata": prediction_data["metadata"]}

        # Qualifying metrics
        if quali_actual is not None:
            quali_pred = prediction_data["qualifying"]["predicted_grid"]
            metrics["qualifying"] = {
                "exact_accuracy": PredictionMetrics.position_accuracy(quali_pred, quali_actual),
                "mae": PredictionMetrics.mean_absolute_error(quali_pred, quali_actual),
                "within_1": PredictionMetrics.within_n_positions(quali_pred, quali_actual, 1),
                "within_3": PredictionMetrics.within_n_positions(quali_pred, quali_actual, 3),
                "correlation": PredictionMetrics.correlation_coefficient(quali_pred, quali_actual),
            }

        # Race metrics
        if race_actual is not None:
            race_pred = prediction_data["race"]["predicted_results"]
            metrics["race"] = {
                "exact_accuracy": PredictionMetrics.position_accuracy(race_pred, race_actual),
                "mae": PredictionMetrics.mean_absolute_error(race_pred, race_actual),
                "within_1": PredictionMetrics.within_n_positions(race_pred, race_actual, 1),
                "within_3": PredictionMetrics.within_n_positions(race_pred, race_actual, 3),
                "correlation": PredictionMetrics.correlation_coefficient(race_pred, race_actual),
                "podium": PredictionMetrics.podium_accuracy(race_pred, race_actual),
                "winner_correct": PredictionMetrics.winner_accuracy(race_pred, race_actual),
            }

        return metrics

    @staticmethod
    def aggregate_metrics(all_predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate metrics across multiple predictions."""
        # Calculate metrics for each prediction
        all_metrics = []
        for pred in all_predictions:
            metrics = PredictionMetrics.calculate_all_metrics(pred)
            if metrics is not None:
                all_metrics.append(metrics)

        if not all_metrics:
            return {"error": "No predictions with actuals found"}

        # Aggregate qualifying metrics
        quali_metrics = [m["qualifying"] for m in all_metrics if "qualifying" in m]
        race_metrics = [m["race"] for m in all_metrics if "race" in m]

        aggregated: dict[str, Any] = {}

        if quali_metrics:
            aggregated["qualifying"] = {
                "exact_accuracy": {
                    "mean": np.mean([m["exact_accuracy"] for m in quali_metrics]),
                    "std": np.std([m["exact_accuracy"] for m in quali_metrics]),
                },
                "mae": {
                    "mean": np.mean([m["mae"] for m in quali_metrics]),
                    "std": np.std([m["mae"] for m in quali_metrics]),
                },
                "within_1": {
                    "mean": np.mean([m["within_1"] for m in quali_metrics]),
                    "std": np.std([m["within_1"] for m in quali_metrics]),
                },
                "within_3": {
                    "mean": np.mean([m["within_3"] for m in quali_metrics]),
                    "std": np.std([m["within_3"] for m in quali_metrics]),
                },
                "correlation": {
                    "mean": np.mean([m["correlation"] for m in quali_metrics]),
                    "std": np.std([m["correlation"] for m in quali_metrics]),
                },
            }

        if race_metrics:
            aggregated["race"] = {
                "exact_accuracy": {
                    "mean": np.mean([m["exact_accuracy"] for m in race_metrics]),
                    "std": np.std([m["exact_accuracy"] for m in race_metrics]),
                },
                "mae": {
                    "mean": np.mean([m["mae"] for m in race_metrics]),
                    "std": np.std([m["mae"] for m in race_metrics]),
                },
                "within_1": {
                    "mean": np.mean([m["within_1"] for m in race_metrics]),
                    "std": np.std([m["within_1"] for m in race_metrics]),
                },
                "within_3": {
                    "mean": np.mean([m["within_3"] for m in race_metrics]),
                    "std": np.std([m["within_3"] for m in race_metrics]),
                },
                "correlation": {
                    "mean": np.mean([m["correlation"] for m in race_metrics]),
                    "std": np.std([m["correlation"] for m in race_metrics]),
                },
                "podium_accuracy": {
                    "mean": np.mean([m["podium"]["accuracy"] for m in race_metrics]),
                    "std": np.std([m["podium"]["accuracy"] for m in race_metrics]),
                },
                "winner_accuracy": {
                    "percentage": (
                        sum(1 for m in race_metrics if m["winner_correct"]) / len(race_metrics)
                    )
                    * 100
                },
            }

        aggregated["n_predictions"] = len(all_metrics)
        return aggregated
