"""Tests for configuration schema validation."""

import pytest
from pydantic import ValidationError

from src.utils.config_schema import (
    BayesianConfig,
    BlendConfig,
    DNFConfig,
    RaceWeightsConfig,
    validate_config,
)


def test_bayesian_config_validates_volatility_range():
    """Volatility must be 0.0-1.0."""
    valid = BayesianConfig(base_volatility=0.5)
    assert valid.base_volatility == 0.5

    with pytest.raises(ValidationError):
        BayesianConfig(base_volatility=1.5)  # Too high

    with pytest.raises(ValidationError):
        BayesianConfig(base_volatility=-0.1)  # Negative


def test_race_weights_sum_not_enforced_but_capped():
    """Weights are individually capped but sum not enforced (flexibility)."""
    # Individual weights must be 0.0-1.0
    valid = RaceWeightsConfig(
        pace_weight=0.4, grid_weight=0.3, overtaking_weight=0.15, tire_deg_weight=0.15
    )
    assert valid.pace_weight == 0.4

    # Weight > 1.0 should fail
    with pytest.raises(ValidationError):
        RaceWeightsConfig(pace_weight=1.5)


def test_dnf_config_validates_probabilities():
    """DNF probabilities must be valid."""
    valid = DNFConfig(base_risk=0.05, driver_error_factor=0.15)
    assert valid.base_risk == 0.05

    with pytest.raises(ValidationError):
        DNFConfig(base_risk=1.5)  # > 1.0


def test_blend_config_validates_weights():
    """Blend weights must be 0.0-1.0."""
    valid = BlendConfig(default=0.7, fp3_only=0.8, fp1_only=0.4)
    assert valid.default == 0.7

    with pytest.raises(ValidationError):
        BlendConfig(default=1.2)  # Too high


def test_full_config_validation_with_valid_config():
    """Valid configuration passes validation."""
    config = {
        "paths": {
            "data_dir": "data",
            "processed": "data/processed",
            "raw": "data/raw",
            "driver_chars": "data/processed/driver_characteristics.json",
            "track_chars": "data/processed/track_characteristics.json",
            "lineups": "data/current_lineups.json",
            "cache": ".fastf1_cache",
        },
        "bayesian": {
            "base_volatility": 0.1,
            "base_observation_noise": 2.0,
            "shock_threshold": 2.0,
            "shock_multiplier": 0.5,
        },
        "race": {
            "weights": {
                "pace_weight": 0.4,
                "grid_weight": 0.3,
                "overtaking_weight": 0.15,
                "tire_deg_weight": 0.15,
            },
            "base_uncertainty": 2.5,
            "uncertainty_multipliers": {"rain": 1.5, "easy_overtaking": 0.8},
            "dnf": {
                "base_risk": 0.05,
                "driver_error_factor": 0.15,
                "street_circuit_risk": 0.05,
                "rain_risk": 0.10,
            },
            "lap1": {"midfield_variance": 1.5, "front_row_variance": 0.0},
            "tire": {"degradation_multiplier": 4.0, "skill_reduction_factor": 0.2},
            "weather": {"rain_position_swing": 6.0, "mixed_intensity": 0.5},
            "safety_car": {"compression_factor": 0.1},
            "pace": {"pace_delta_multiplier": 3.0},
            "dnf_position_penalty": 22,
        },
        "qualifying": {
            "blend": {"default": 0.7, "fp3_only": 0.8, "fp1_only": 0.4},
            "session_confidence": {"fp1": 0.2, "fp2": 0.5, "fp3": 0.9, "sprint_quali": 0.85},
            "base_uncertainty": 1.5,
        },
        "learning": {"performance_window": 5, "min_races_for_blend": 3},
    }

    validated = validate_config(config)
    assert validated.bayesian.base_volatility == 0.1
    assert validated.race.weights.pace_weight == 0.4
    assert validated.qualifying.blend.default == 0.7


def test_full_config_validation_fails_with_invalid_values():
    """Invalid configuration raises ValidationError."""
    invalid_config = {
        "paths": {"data_dir": "data", "processed": "data/processed", "raw": "data/raw"},
        "bayesian": {
            "base_volatility": 2.5,  # Invalid: > 1.0
            "base_observation_noise": 2.0,
            "shock_threshold": 2.0,
            "shock_multiplier": 0.5,
        },
        "race": {
            "weights": {
                "pace_weight": 0.4,
                "grid_weight": 0.3,
                "overtaking_weight": 0.15,
                "tire_deg_weight": 0.15,
            },
            "dnf": {"base_risk": 0.05},
            "lap1": {"midfield_variance": 1.5},
            "tire": {"degradation_multiplier": 4.0},
            "weather": {"rain_position_swing": 6.0},
            "safety_car": {"compression_factor": 0.1},
            "pace": {"pace_delta_multiplier": 3.0},
        },
        "qualifying": {
            "blend": {"default": 0.7},
            "session_confidence": {"fp1": 0.2, "fp2": 0.5, "fp3": 0.9},
        },
        "learning": {"performance_window": 5},
    }

    with pytest.raises(ValidationError) as exc_info:
        validate_config(invalid_config)

    error_msg = str(exc_info.value)
    assert "base_volatility" in error_msg.lower()


def test_config_loader_integration():
    """Config loader uses Pydantic validation."""
    from src.utils.config_loader import Config

    # This should load and validate the actual config file
    config = Config()
    assert config._config is not None

    # Verify config can retrieve values
    base_volatility = config.get("bayesian.base_volatility")
    assert isinstance(base_volatility, int | float)
    assert 0.0 <= base_volatility <= 1.0
