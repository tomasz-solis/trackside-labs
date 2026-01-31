"""
Tire Degradation Predictor.
Uses Pirelli compound data + Track Surface + Driver Style.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from src.utils.schema_validation import validate_driver_characteristics

logger = logging.getLogger(__name__)


class TirePredictor:
    """
    Predict tire degradation with data hierarchy.
    """

    def __init__(self, year: int, driver_chars_path: str, data_dir: str = "data"):
        """
        Initialize tire predictor with season-specific Pirelli data.

        Args:
            year: Current season year
            driver_chars_path: Path to driver characteristics
            data_dir: Root data directory

        Raises:
            ValueError: If driver characteristics JSON is malformed
            FileNotFoundError: If required data files not found
        """
        self.year = year
        self.data_dir = Path(data_dir)

        # Load and validate driver characteristics
        with open(driver_chars_path) as f:
            data = json.load(f)
            # Validate JSON schema before using data
            try:
                validate_driver_characteristics(data)
            except ValueError as e:
                logger.error(f"Failed to load driver characteristics: {e}")
                raise
            self.drivers = data["drivers"]

        # Load Pirelli Data (Season Specific or Fallback)
        self.pirelli_data = self._load_pirelli_data(year)

        # Tunable Parameters
        self.skill_reduction_factor = 0.2
        self.track_effect_range = 0.1

    def _load_pirelli_data(self, year: int) -> Dict:
        """
        Load Pirelli track info (Compounds, Stress levels).
        Falls back to previous year if current year data is missing.
        """
        # Try current year first
        current_file = self.data_dir / f"{year}_pirelli_info.json"

        if current_file.exists():
            print(f"   🛞  Loaded Pirelli Data: {year}")
            with open(current_file) as f:
                return json.load(f)

        # Fallback to previous year
        prev_file = self.data_dir / f"{year-1}_pirelli_info.json"
        if prev_file.exists():
            print(f"   ⚠️  No {year} Pirelli data. Using {year-1} fallback.")
            with open(prev_file) as f:
                return json.load(f)

        print("   ❌ No Pirelli data found. Using defaults.")
        return {}

    def get_tire_impact(
        self,
        driver: str,
        team: str,
        track_name: str,
        fp2_data: Optional[Dict] = None,
        race_progress: float = 0.7,
    ) -> Dict[str, float]:
        """
        Calculate tire degradation impact.
        Priority: FP2 Data (Real) > Pirelli Baseline (Static) > Heuristic
        """
        # 1. Base Degradation
        if fp2_data and team in fp2_data:
            base_deg = fp2_data[team].get("degradation", 0.0)
            source = "fp2"
        else:
            base_deg = self._get_pirelli_baseline(track_name)
            source = "pirelli_baseline"

        # 2. Driver Skill Modifier
        driver_skill = self._get_driver_tire_skill(driver)
        driver_adjustment = 1.0 - (driver_skill * self.skill_reduction_factor)

        # 3. Race Progress (Non-linear wear)
        progress_factor = race_progress**1.5

        final_deg = base_deg * driver_adjustment * progress_factor

        return {"degradation": float(np.clip(final_deg, 0, 1.0)), "source": source}

    def _get_pirelli_baseline(self, track_name: str) -> float:
        """Get baseline deg from loaded Pirelli JSON."""
        if not track_name or track_name not in self.pirelli_data:
            return 0.5  # Default medium deg

        track_info = self.pirelli_data[track_name]

        # Calculate from stress scale (1-5)
        # 5/5 abrasion = High Deg (0.8). 1/5 = Low Deg (0.2)
        abrasion = track_info.get("abrasion", 3)
        stress = track_info.get("stress", 3)

        # Normalize 1-5 to 0.0-1.0 range
        avg_score = (abrasion + stress) / 2
        return avg_score / 5.0

    def _get_driver_tire_skill(self, driver: str) -> float:
        if driver not in self.drivers:
            return 0.5
        return self.drivers[driver].get("tire_management", {}).get("skill_score", 0.5)
