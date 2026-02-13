"""
Represents a Team's Car performance profile.
Calculates aggregate performance scores based on testing/practice telemetry.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class CarCharacteristics:
    """Raw performance metrics (0.0 to 1.0) derived from Telemetry."""

    slow_corner: float = 0.5
    medium_corner: float = 0.5
    fast_corner: float = 0.5
    straight_line: float = 0.5
    reliability: float = 0.8
    tire_preservation: float = 0.5

    # Aerodynamic Efficiency (Drag vs Downforce)
    drag_coefficient: float = 0.5
    downforce_level: float = 0.5


class Car:
    """
    Model representing a car's performance capabilities.
    """

    def __init__(self, name: str, year: int):
        self.name = name
        self.year = year
        self.characteristics = CarCharacteristics()
        self.development_rate = 0.01  # Base improvement per race

    def update_from_testing(self, testing_data: Dict[str, float]):
        """Update car characteristics from extracted testing metrics."""
        if not testing_data:
            return

        # Map testing extractors to car characteristics
        self.characteristics.slow_corner = testing_data.get(
            "slow_corner_performance", 0.5
        )
        self.characteristics.medium_corner = testing_data.get(
            "medium_corner_performance", 0.5
        )
        self.characteristics.fast_corner = testing_data.get(
            "fast_corner_performance", 0.5
        )

        # Normalize Top Speed (assuming 300-350 km/h range for 0-1 scaling)
        top_speed = testing_data.get("top_speed", 320.0)
        self.characteristics.straight_line = np.clip((top_speed - 300) / 50.0, 0.0, 1.0)

        self.characteristics.reliability = testing_data.get("consistency", 0.8)
        self.characteristics.tire_preservation = testing_data.get("tire_deg_slope", 0.5)

    def get_performance_score(self, track_type: str = "balanced") -> float:
        """
        Calculate a composite performance score (0-20 scale) based on track demands.
        """
        c = self.characteristics

        # Base Score (Weighted average of core attributes)
        # Medium corners and Straights usually dominate lap time
        base_score = (
            c.slow_corner * 0.15
            + c.medium_corner * 0.30
            + c.fast_corner * 0.25
            + c.straight_line * 0.30
        )

        # Track-Specific Modifiers
        if track_type == "monaco":  # High Downforce, Low Speed
            score = (
                (c.slow_corner * 0.6) + (c.medium_corner * 0.3) + (c.reliability * 0.1)
            )
        elif track_type == "monza":  # Low Drag, High Speed
            score = (
                (c.straight_line * 0.7) + (c.slow_corner * 0.2) + (c.fast_corner * 0.1)
            )
        elif track_type == "silverstone":  # High Speed Cornering
            score = (
                (c.fast_corner * 0.5)
                + (c.medium_corner * 0.3)
                + (c.straight_line * 0.2)
            )
        else:
            score = base_score

        return self._calculate_base_score(score)

    def _calculate_base_score(self, raw_score: float) -> float:
        """Convert 0-1 raw score to 0-20 rating scale for Bayesian priors."""
        # 0.0 -> 5.0 (Backmarker)
        # 1.0 -> 18.0 (Dominant)
        return 5.0 + (raw_score * 13.0)
