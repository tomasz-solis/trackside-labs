"""Qualifying and sprint-race mixin for Baseline2026Predictor."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.utils import config_loader
from src.utils.fp_blending import blend_team_strength, get_best_fp_performance
from src.utils.lineups import get_lineups
from src.utils.validation_helpers import (
    validate_enum,
    validate_positive_int,
    validate_year,
)
from src.utils.weekend import is_sprint_weekend

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineQualifyingMixin:
    """Shared qualifying and sprint-race methods for Baseline2026Predictor."""

    def predict_qualifying(
        self,
        year: int,
        race_name: str,
        n_simulations: int = 50,
        qualifying_stage: str = "auto",
    ) -> dict[str, Any]:
        """
        Predict qualifying order based on team + driver baseline.

        Runs multiple Monte Carlo simulations and returns median positions.
        Returns low confidence (40-60%) acknowledging regulation uncertainty.

        Sprint weekends:
        - qualifying_stage="sprint": predicts Friday Sprint Qualifying (sets Sprint Race grid)
        - qualifying_stage="main": predicts Saturday Main Qualifying (sets Sunday Race grid)
        - qualifying_stage="auto": legacy behavior (best available sprint-context session)

        Normal weekends always predict standard Saturday Qualifying.

        For F1 Fantasy users - this provides the grid BEFORE team lock:
        - Sprint: Lock before Sprint Race (Saturday) -> predict from Friday Sprint Quali
        - Normal: Lock before Qualifying (Saturday) -> predict qualifying outcome
        """

        base_seed = self.seed
        rng = np.random.default_rng(base_seed)

        # Validate inputs
        validate_year(year, "year", min_year=2020, max_year=2030)
        validate_positive_int(n_simulations, "n_simulations", min_val=1)
        validate_enum(qualifying_stage, "qualifying_stage", ["auto", "sprint", "main"])

        # Check if sprint weekend
        try:
            is_sprint = is_sprint_weekend(year, race_name)
        except (ValueError, KeyError, FileNotFoundError) as e:
            logger.warning(
                f"Could not determine sprint weekend for {race_name}: {e}. "
                "Using conventional fallback."
            )
            is_sprint = False

        # Get current lineups
        lineups = get_lineups(year, race_name)

        # Get FP practice data for blending (70% FP + 30% model)
        # Also returns session laps for compound analysis
        session_name, fp_performance, session_laps = get_best_fp_performance(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            qualifying_stage=qualifying_stage,
        )

        # Extract and update compound characteristics if we have session data
        if session_laps is not None:
            self._update_compound_characteristics_from_session(
                session_laps, race_name, year, is_sprint
            )

        # Build driver list
        all_drivers = []
        model_strengths = {}  # For blending
        teams_with_short_profile = 0

        short_profile_scale = config_loader.get(
            "baseline_predictor.qualifying.testing_short_run_modifier_scale", 0.04
        )
        short_profile_weights = {
            "overall_pace": 0.55,
            "top_speed": 0.20,
            "medium_corner_performance": 0.15,
            "fast_corner_performance": 0.10,
        }

        for team, _drivers in lineups.items():
            # Use blended team strength (baseline + testing + current season)
            model_strength = self.get_blended_team_strength(team, race_name)
            short_modifier, has_short_profile = self._compute_testing_profile_modifier(
                team=team,
                profile="short_run",
                metric_weights=short_profile_weights,
                scale=short_profile_scale,
            )
            model_strength = float(np.clip(model_strength + short_modifier, 0.0, 1.0))
            if has_short_profile:
                teams_with_short_profile += 1
            model_strengths[team] = model_strength

        # Blend model predictions with FP data (70% FP, 30% model)
        blended_strengths = blend_team_strength(model_strengths, fp_performance, blend_weight=0.7)

        # Build driver list with blended strengths
        for team, drivers in lineups.items():
            team_strength = blended_strengths[team]

            for driver_code in drivers:
                driver_data = self.drivers.get(driver_code, {})
                skill = driver_data.get("racecraft", {}).get("skill_score", 0.5)
                quali_pace = driver_data.get("pace", {}).get("quali_pace", 0.5)

                all_drivers.append(
                    {
                        "driver": driver_code,
                        "team": team,
                        "team_strength": team_strength,
                        "skill": skill,
                        "quali_pace": quali_pace,
                    }
                )

        # Run multiple simulations
        position_records = {d["driver"]: [] for d in all_drivers}

        # Load qualifying noise parameters from config
        # Sprint weekends have slightly more variance (less practice)
        noise_std_sprint = config_loader.get(
            "baseline_predictor.qualifying.noise_std_sprint", 0.025
        )
        noise_std_normal = config_loader.get("baseline_predictor.qualifying.noise_std_normal", 0.02)
        noise_std = noise_std_sprint if is_sprint else noise_std_normal

        # Load score composition weights from config
        team_weight = config_loader.get("baseline_predictor.qualifying.team_weight", 0.7)
        skill_weight = config_loader.get("baseline_predictor.qualifying.skill_weight", 0.3)
        # Compress team spread so car still matters most, but doesn't lock order.
        team_strength_compression = config_loader.get(
            "baseline_predictor.qualifying.team_strength_compression", 0.50
        )
        # Build a quali-specific driver signal from pure one-lap pace + racecraft.
        driver_quali_pace_weight = config_loader.get(
            "baseline_predictor.qualifying.driver_quali_pace_weight", 0.70
        )
        driver_skill_weight = config_loader.get(
            "baseline_predictor.qualifying.driver_skill_weight", 0.30
        )
        driver_weight_sum = driver_quali_pace_weight + driver_skill_weight
        if driver_weight_sum <= 0:
            driver_quali_pace_weight, driver_skill_weight = 0.70, 0.30
            driver_weight_sum = 1.0
        driver_offset_cap = float(
            config_loader.get("baseline_predictor.qualifying.driver_offset_cap", 0.18)
        )

        teammate_setup_std = config_loader.get(
            "baseline_predictor.qualifying.teammate_setup_std", 0.015
        )

        for _ in range(n_simulations):
            # Calculate scores with random noise
            driver_scores = []
            for driver_info in all_drivers:
                compressed_team_strength = 0.5 + (
                    (driver_info["team_strength"] - 0.5) * team_strength_compression
                )
                compressed_team_strength = float(np.clip(compressed_team_strength, 0.0, 1.0))

                driver_signal = (
                    (driver_info["quali_pace"] * driver_quali_pace_weight)
                    + (driver_info["skill"] * driver_skill_weight)
                ) / driver_weight_sum
                bounded_driver_signal = 0.5 + float(
                    np.clip(driver_signal - 0.5, -driver_offset_cap, driver_offset_cap)
                )

                # Combined score: team_weight% team, skill_weight% driver
                score = (compressed_team_strength * team_weight) + (
                    bounded_driver_signal * skill_weight
                )
                # Teammates in same car still diverge on setup/execution.
                score += rng.normal(0, teammate_setup_std)
                # Add random noise (more for sprint weekends)
                score += rng.normal(0, noise_std)

                driver_scores.append(
                    {
                        "driver": driver_info["driver"],
                        "team": driver_info["team"],
                        "score": score,
                    }
                )

            # Sort and record positions
            driver_scores.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(driver_scores):
                position_records[item["driver"]].append(i + 1)

        # Calculate median positions and prediction intervals
        grid = []
        for driver_info in all_drivers:
            positions = position_records[driver_info["driver"]]
            median_pos = int(np.median(positions))
            p5 = int(np.percentile(positions, 5))  # 5th percentile (optimistic)
            p95 = int(np.percentile(positions, 95))  # 95th percentile (pessimistic)

            # Confidence based on consistency
            position_std = np.std(positions)
            # Lower std = higher confidence
            confidence = max(40, min(60, 60 - (position_std * 5)))

            grid.append(
                {
                    "driver": driver_info["driver"],
                    "team": driver_info["team"],
                    "median_position": median_pos,
                    "p5": p5,
                    "p95": p95,
                    "confidence": round(confidence, 1),
                }
            )

        # Sort by median position
        grid.sort(key=lambda x: x["median_position"])

        # Assign final positions
        for i, item in enumerate(grid):
            item["position"] = i + 1

        return {
            "grid": grid,
            "data_source": session_name or "Model-only (no practice data)",
            "blend_used": session_name is not None,
            "qualifying_stage": qualifying_stage,
            "characteristics_profile_used": "short_run",
            "teams_with_characteristics_profile": teams_with_short_profile,
        }

    def predict_sprint_race(
        self,
        sprint_quali_grid: list[dict],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
    ) -> dict[str, Any]:
        """
        Predict Sprint Race result from Sprint Qualifying grid.

        Sprint races use the same base race model with sprint-specific adjustments
        (reduced chaos and increased grid influence).
        """
        # Validate inputs
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)

        # Sprint races are modeled with lower chaos and slightly stronger
        # grid-position influence than full races.

        # Use predict_race but with sprint-specific adjustments
        result = self.predict_race(
            qualifying_grid=sprint_quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
            is_sprint=True,  # Flag for sprint-specific behavior
        )

        return result
