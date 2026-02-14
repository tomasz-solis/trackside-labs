"""
Configuration loader for F1 prediction system.

Loads settings from YAML config files with environment variable overrides.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Central configuration manager."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load()

    def _load(self):
        """Load config from YAML file."""
        # Find config file
        config_file = os.getenv("F1_CONFIG", "config/default.yaml")
        config_path = Path(config_file)

        if not config_path.is_absolute():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        # Validate required sections exist
        self._validate_config()
        logger.info(f"Configuration loaded successfully from {config_path}")

    def _validate_config(self):
        """Validate that required config sections exist, with correct structure and value ranges."""
        # 1. Check required sections exist
        required_sections = [
            "paths",
            "bayesian",
            "race",
            "qualifying",
            "baseline_predictor",
        ]

        missing = []
        for section in required_sections:
            if section not in self._config:
                missing.append(section)

        if missing:
            raise ValueError(
                f"Config validation failed. Missing required sections: {missing}. "
                "Check your config file structure."
            )

        # 2. Validate baseline_predictor subsections
        if "qualifying" not in self._config["baseline_predictor"]:
            raise ValueError("Config missing baseline_predictor.qualifying section")
        if "race" not in self._config["baseline_predictor"]:
            raise ValueError("Config missing baseline_predictor.race section")

        # 3. Type and range validation for critical parameters
        validations = [
            # Bayesian model parameters
            ("bayesian.base_volatility", float, 0.0, 1.0),
            ("bayesian.base_observation_noise", float, 0.0, 100.0),
            ("bayesian.shock_threshold", float, 0.0, 10.0),
            # Qualifying parameters
            ("baseline_predictor.qualifying.noise_std_sprint", float, 0.0, 0.5),
            ("baseline_predictor.qualifying.noise_std_normal", float, 0.0, 0.5),
            ("baseline_predictor.qualifying.team_weight", float, 0.0, 1.0),
            ("baseline_predictor.qualifying.skill_weight", float, 0.0, 1.0),
            # Race parameters - base chaos
            ("baseline_predictor.race.base_chaos.dry", float, 0.0, 1.0),
            ("baseline_predictor.race.base_chaos.wet", float, 0.0, 1.0),
            ("baseline_predictor.race.track_chaos_multiplier", float, 0.0, 1.0),
            # Race parameters - safety car
            ("baseline_predictor.race.sc_base_probability.dry", float, 0.0, 1.0),
            ("baseline_predictor.race.sc_base_probability.wet", float, 0.0, 1.0),
            ("baseline_predictor.race.sc_track_modifier", float, 0.0, 1.0),
            # Race parameters - grid and pace
            ("baseline_predictor.race.grid_weight_min", float, 0.0, 1.0),
            ("baseline_predictor.race.grid_weight_multiplier", float, 0.0, 1.0),
            ("baseline_predictor.race.pace_weight_base", float, 0.0, 1.0),
            ("baseline_predictor.race.pace_weight_track_modifier", float, 0.0, 1.0),
            # Race parameters - DNF caps
            ("baseline_predictor.race.dnf_rate_historical_cap", float, 0.0, 1.0),
            ("baseline_predictor.race.dnf_rate_final_cap", float, 0.0, 1.0),
        ]

        errors = []
        for key, expected_type, min_val, max_val in validations:
            value = self.get(key)

            # Check if value exists
            if value is None:
                errors.append(f"Missing required config key: {key}")
                continue

            # Check type
            if not isinstance(value, expected_type):
                errors.append(
                    f"Invalid type for {key}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
                continue

            # Check range
            if not (min_val <= value <= max_val):
                errors.append(
                    f"Value out of range for {key}: {value} "
                    f"(must be between {min_val} and {max_val})"
                )

        if errors:
            error_msg = "Config validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)

        # 4. Validate weight sums (qualifying)
        team_weight = self.get("baseline_predictor.qualifying.team_weight", 0.7)
        skill_weight = self.get("baseline_predictor.qualifying.skill_weight", 0.3)
        weight_sum = team_weight + skill_weight

        if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"baseline_predictor.qualifying weights must sum to 1.0, got {weight_sum:.3f} "
                f"(team_weight={team_weight}, skill_weight={skill_weight})"
            )

        logger.debug("Config validation passed")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation, returning default if not found."""
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> dict:
        """Get entire config section."""
        return self._config.get(section, {})

    def reload(self):
        """Force reload config from file."""
        self._config = None
        self._load()


# Singleton instance
_config = Config()


def get(key: str, default: Any = None) -> Any:
    """Get config value."""
    return _config.get(key, default)


def get_section(section: str) -> dict:
    """Get config section."""
    return _config.get_section(section)


def reload():
    """Reload config."""
    _config.reload()
