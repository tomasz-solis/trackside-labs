"""Saves predictions to JSON for accuracy tracking."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Handles saving and loading race predictions for accuracy tracking."""

    def __init__(self, predictions_dir: str = "data/predictions"):
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def save_prediction(
        self,
        year: int,
        race_name: str,
        session_name: str,
        qualifying_prediction: list[dict[str, Any]],
        race_prediction: list[dict[str, Any]],
        weather: str,
        fp_blend_info: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a prediction to disk with validation."""
        # Validate inputs
        if not qualifying_prediction or not race_prediction:
            raise ValueError("Predictions cannot be empty")

        # Validate required fields in predictions
        for i, pred in enumerate(qualifying_prediction):
            if "driver" not in pred or "team" not in pred:
                raise ValueError(f"Qualifying prediction {i} missing 'driver' or 'team' field")

        for i, pred in enumerate(race_prediction):
            if "driver" not in pred or "team" not in pred:
                raise ValueError(f"Race prediction {i} missing 'driver' or 'team' field")

        # Validate weather
        valid_weather = ["dry", "rain", "mixed", "wet"]
        if weather not in valid_weather:
            logger.warning(f"Unusual weather value: {weather}. Expected one of {valid_weather}")

        # Sanitize race name for filename
        safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")

        # Create year directory
        year_dir = self.predictions_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        # Create race directory
        race_dir = year_dir / safe_race_name
        race_dir.mkdir(parents=True, exist_ok=True)

        # Filename includes session to allow multiple predictions per race
        filename = f"{safe_race_name}_{session_name.lower()}.json"
        filepath = race_dir / filename

        # Build prediction data structure
        prediction_data = {
            "metadata": {
                "year": year,
                "race_name": race_name,
                "session_name": session_name,
                "predicted_at": datetime.now(UTC).isoformat(),
                "weather": weather,
                "fp_blend_info": fp_blend_info or {},
                **({} if metadata is None else metadata),
            },
            "qualifying": {
                "predicted_grid": [
                    {
                        "position": i + 1,
                        "driver": result["driver"],
                        "team": result["team"],
                        "expected_time": result.get("expected_time"),
                        "confidence": result.get("confidence"),
                    }
                    for i, result in enumerate(qualifying_prediction)
                ]
            },
            "race": {
                "predicted_results": [
                    {
                        "position": i + 1,
                        "driver": result["driver"],
                        "team": result["team"],
                        "confidence": result.get("confidence"),
                        "dnf_risk": result.get("dnf_risk"),
                    }
                    for i, result in enumerate(race_prediction)
                ]
            },
            # Placeholder for actual results (to be filled later)
            "actuals": {"qualifying": None, "race": None},
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(prediction_data, f, indent=2)

        logger.info(f"Saved prediction to {filepath}")
        return filepath

    def load_prediction(
        self, year: int, race_name: str, session_name: str
    ) -> dict[str, Any] | None:
        """Load a saved prediction from disk with schema validation."""
        safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")
        filepath = (
            self.predictions_dir
            / str(year)
            / safe_race_name
            / f"{safe_race_name}_{session_name.lower()}.json"
        )

        if not filepath.exists():
            logger.warning(f"Prediction not found: {filepath}")
            return None

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Validate schema
            required_keys = ["metadata", "qualifying", "race", "actuals"]
            if not all(key in data for key in required_keys):
                logger.error(f"Invalid prediction schema in {filepath}: missing required keys")
                return None

            # Validate metadata
            if not isinstance(data["metadata"], dict):
                logger.error(f"Invalid metadata in {filepath}")
                return None

            # Validate prediction lists
            if "predicted_grid" not in data["qualifying"]:
                logger.error(f"Missing predicted_grid in {filepath}")
                return None

            if "predicted_results" not in data["race"]:
                logger.error(f"Missing predicted_results in {filepath}")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted JSON file {filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load prediction from {filepath}: {e}")
            return None

    def update_actuals(
        self,
        year: int,
        race_name: str,
        session_name: str,
        qualifying_results: list[dict[str, Any]] | None = None,
        race_results: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Update a saved prediction with actual race results."""
        prediction = self.load_prediction(year, race_name, session_name)
        if prediction is None:
            return False

        # Update actuals
        if qualifying_results is not None:
            prediction["actuals"]["qualifying"] = [
                {"position": i + 1, "driver": r["driver"], "team": r["team"]}
                for i, r in enumerate(qualifying_results)
            ]

        if race_results is not None:
            prediction["actuals"]["race"] = [
                {"position": i + 1, "driver": r["driver"], "team": r["team"]}
                for i, r in enumerate(race_results)
            ]

        # Save updated prediction
        safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")
        filepath = (
            self.predictions_dir
            / str(year)
            / safe_race_name
            / f"{safe_race_name}_{session_name.lower()}.json"
        )

        with open(filepath, "w") as f:
            json.dump(prediction, f, indent=2)

        logger.info(f"Updated actuals in {filepath}")
        return True

    def get_all_predictions(self, year: int) -> list[dict[str, Any]]:
        """Load all predictions for a given year."""
        year_dir = self.predictions_dir / str(year)
        if not year_dir.exists():
            return []

        predictions = []
        for race_dir in year_dir.iterdir():
            if race_dir.is_dir():
                for pred_file in race_dir.glob("*.json"):
                    with open(pred_file) as f:
                        predictions.append(json.load(f))

        return predictions

    def has_prediction_for_session(self, year: int, race_name: str, session_name: str) -> bool:
        """Check if a prediction exists for a given session."""
        safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")
        filepath = (
            self.predictions_dir
            / str(year)
            / safe_race_name
            / f"{safe_race_name}_{session_name.lower()}.json"
        )
        return filepath.exists()
