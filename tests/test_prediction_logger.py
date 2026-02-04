"""
Tests for Prediction Logger
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.utils.prediction_logger import PredictionLogger


@pytest.fixture
def temp_predictions_dir():
    """Create temporary directory for predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_quali_prediction():
    """Sample qualifying prediction."""
    return [
        {
            "driver": "Verstappen",
            "team": "Red Bull",
            "expected_time": 78.5,
            "confidence": 0.8,
        },
        {
            "driver": "Norris",
            "team": "McLaren",
            "expected_time": 78.7,
            "confidence": 0.75,
        },
        {
            "driver": "Leclerc",
            "team": "Ferrari",
            "expected_time": 78.8,
            "confidence": 0.72,
        },
    ]


@pytest.fixture
def sample_race_prediction():
    """Sample race prediction."""
    return [
        {
            "driver": "Verstappen",
            "team": "Red Bull",
            "confidence": 0.8,
            "dnf_risk": 0.05,
        },
        {"driver": "Norris", "team": "McLaren", "confidence": 0.75, "dnf_risk": 0.07},
        {"driver": "Leclerc", "team": "Ferrari", "confidence": 0.72, "dnf_risk": 0.08},
    ]


def test_save_prediction(temp_predictions_dir, sample_quali_prediction, sample_race_prediction):
    """Test saving a prediction."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    filepath = logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    assert filepath.exists()
    assert filepath.name == "bahrain_grand_prix_fp1.json"

    # Load and verify structure
    with open(filepath) as f:
        data = json.load(f)

    assert data["metadata"]["year"] == 2026
    assert data["metadata"]["race_name"] == "Bahrain Grand Prix"
    assert data["metadata"]["session_name"] == "FP1"
    assert data["metadata"]["weather"] == "dry"
    assert len(data["qualifying"]["predicted_grid"]) == 3
    assert len(data["race"]["predicted_results"]) == 3
    assert data["actuals"]["qualifying"] is None
    assert data["actuals"]["race"] is None


def test_load_prediction(temp_predictions_dir, sample_quali_prediction, sample_race_prediction):
    """Test loading a saved prediction."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # Save first
    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP2",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # Load
    prediction = logger.load_prediction(2026, "Bahrain Grand Prix", "FP2")

    assert prediction is not None
    assert prediction["metadata"]["race_name"] == "Bahrain Grand Prix"
    assert prediction["metadata"]["session_name"] == "FP2"


def test_load_nonexistent_prediction(temp_predictions_dir):
    """Test loading a prediction that doesn't exist."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    prediction = logger.load_prediction(2026, "Monaco Grand Prix", "FP1")

    assert prediction is None


def test_update_actuals(temp_predictions_dir, sample_quali_prediction, sample_race_prediction):
    """Test updating a prediction with actual results."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # Save prediction
    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP3",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # Add actuals
    actual_quali = [
        {"driver": "Verstappen", "team": "Red Bull"},
        {"driver": "Leclerc", "team": "Ferrari"},
        {"driver": "Norris", "team": "McLaren"},
    ]

    actual_race = [
        {"driver": "Verstappen", "team": "Red Bull"},
        {"driver": "Norris", "team": "McLaren"},
        {"driver": "Leclerc", "team": "Ferrari"},
    ]

    success = logger.update_actuals(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP3",
        qualifying_results=actual_quali,
        race_results=actual_race,
    )

    assert success is True

    # Verify actuals were saved
    prediction = logger.load_prediction(2026, "Bahrain Grand Prix", "FP3")
    assert prediction["actuals"]["qualifying"] is not None
    assert prediction["actuals"]["race"] is not None
    assert len(prediction["actuals"]["qualifying"]) == 3
    assert len(prediction["actuals"]["race"]) == 3


def test_has_prediction_for_session(
    temp_predictions_dir, sample_quali_prediction, sample_race_prediction
):
    """Test checking if prediction exists for session."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # No prediction yet
    assert logger.has_prediction_for_session(2026, "Bahrain Grand Prix", "FP1") is False

    # Save prediction
    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # Now it exists
    assert logger.has_prediction_for_session(2026, "Bahrain Grand Prix", "FP1") is True
    # But not for other sessions
    assert logger.has_prediction_for_session(2026, "Bahrain Grand Prix", "FP2") is False


def test_get_all_predictions(temp_predictions_dir, sample_quali_prediction, sample_race_prediction):
    """Test getting all predictions for a year."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # No predictions
    predictions = logger.get_all_predictions(2026)
    assert len(predictions) == 0

    # Save multiple predictions
    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP2",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    logger.save_prediction(
        year=2026,
        race_name="Saudi Arabian Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # Get all
    predictions = logger.get_all_predictions(2026)
    assert len(predictions) == 3


def test_save_prediction_empty_validation(temp_predictions_dir):
    """Test that empty predictions raise ValueError."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    with pytest.raises(ValueError, match="cannot be empty"):
        logger.save_prediction(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="FP1",
            qualifying_prediction=[],
            race_prediction=[{"driver": "VER", "team": "Red Bull"}],
            weather="dry",
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        logger.save_prediction(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="FP1",
            qualifying_prediction=[{"driver": "VER", "team": "Red Bull"}],
            race_prediction=[],
            weather="dry",
        )


def test_save_prediction_missing_fields(temp_predictions_dir):
    """Test that predictions with missing fields raise ValueError."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    with pytest.raises(ValueError, match="missing 'driver' or 'team'"):
        logger.save_prediction(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="FP1",
            qualifying_prediction=[{"driver": "VER"}],  # Missing team
            race_prediction=[{"driver": "VER", "team": "Red Bull"}],
            weather="dry",
        )


def test_load_prediction_invalid_schema(temp_predictions_dir):
    """Test loading a prediction with invalid schema."""
    import json

    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # Create invalid prediction file
    year_dir = Path(temp_predictions_dir) / "2026"
    race_dir = year_dir / "bahrain_grand_prix"
    race_dir.mkdir(parents=True, exist_ok=True)

    invalid_data = {"metadata": {}, "wrong_key": []}  # Missing required keys

    with open(race_dir / "bahrain_grand_prix_fp1.json", "w") as f:
        json.dump(invalid_data, f)

    # Load should return None for invalid schema
    prediction = logger.load_prediction(2026, "Bahrain Grand Prix", "FP1")
    assert prediction is None
