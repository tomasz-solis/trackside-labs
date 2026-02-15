"""
2026 Baseline Predictor

Primary runtime predictor for 2026 qualifying and race simulations.

Combines:
- weight-scheduled team strength (baseline/testing/current season),
- optional session blending for qualifying,
- dynamic tire compound selection and performance adjustments,
- Monte Carlo race scoring with uncertainty and DNF modeling.
"""

import logging
import os
from pathlib import Path

from src.predictors.baseline import (
    BaselineDataMixin,
    BaselineQualifyingMixin,
    BaselineRaceMixin,
)
from src.utils import config_loader
from src.utils.data_generator import ensure_baseline_exists

logger = logging.getLogger(__name__)


class Baseline2026Predictor(
    BaselineDataMixin,
    BaselineQualifyingMixin,
    BaselineRaceMixin,
):
    """
    Primary 2026 predictor used by the dashboard and compatibility wrappers.

    Uses:
    - Team strength from car characteristics (baseline + directionality + current season)
    - Driver skill and risk inputs from driver characteristics
    - Session blending for qualifying when data is available
    - Monte Carlo simulation for qualifying and race predictions
    """

    def __init__(self, data_dir: str = "data/processed", seed: int = 42):
        """Initialize baseline 2026 predictor with team/driver data from data_dir."""
        self.seed = seed

        # Resolve data directory using env var or relative to cwd
        data_dir_path = Path(data_dir)
        if not data_dir_path.is_absolute():
            # Try environment variable first
            env_data_dir = os.getenv("F1_DATA_DIR")
            if env_data_dir:
                self.data_dir = (
                    Path(env_data_dir) / data_dir
                    if data_dir != "data/processed"
                    else Path(env_data_dir)
                )
            else:
                # Fall back to current working directory
                self.data_dir = Path.cwd() / data_dir
        else:
            self.data_dir = data_dir_path

        # Ensure baseline data exists (auto-generate if missing/outdated)
        logger.info("Ensuring baseline data is ready...")
        ensure_baseline_exists(self.data_dir)

        # Load configuration
        self.config = config_loader.get_section("baseline_predictor")
        self.load_data()
