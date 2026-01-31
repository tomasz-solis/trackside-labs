"""
Configuration loader for F1 prediction system.

Loads settings from YAML config files with environment variable overrides.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

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
        """Validate that required config sections exist."""
        required_sections = ["paths", "bayesian", "race", "qualifying", "baseline_predictor"]

        missing = []
        for section in required_sections:
            if section not in self._config:
                missing.append(section)

        if missing:
            raise ValueError(
                f"Config validation failed. Missing required sections: {missing}. "
                f"Check your config file structure."
            )

        # Validate baseline_predictor subsections
        if "qualifying" not in self._config["baseline_predictor"]:
            raise ValueError("Config missing baseline_predictor.qualifying section")
        if "race" not in self._config["baseline_predictor"]:
            raise ValueError("Config missing baseline_predictor.race section")

        logger.debug("Config validation passed")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.

        Args:
            key: Config key like "bayesian.base_volatility"
            default: Default value if key not found

        Returns:
            Config value

        Example:
            config = Config()
            volatility = config.get('bayesian.base_volatility', 0.1)
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict:
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


def get_section(section: str) -> Dict:
    """Get config section."""
    return _config.get_section(section)


def reload():
    """Reload config."""
    _config.reload()
