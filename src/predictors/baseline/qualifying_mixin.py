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

    def _build_driver_list_with_strengths(
        self,
        lineups: dict[str, list[str]],
        fp_performance: dict[str, float] | None,
        race_name: str,
        is_sprint: bool,
    ) -> tuple[list[dict], int]:
        """Build driver list with blended team/driver strengths and testing modifiers."""
        all_drivers = []
        model_strengths = {}
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

        for team in lineups:
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

        blended_strengths = blend_team_strength(model_strengths, fp_performance, blend_weight=0.7)

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

        return all_drivers, teams_with_short_profile

    def _run_qualifying_simulations(
        self,
        all_drivers: list[dict],
        n_simulations: int,
        is_sprint: bool,
        rng: np.random.Generator,
    ) -> dict[str, list[int]]:
        """Run Monte Carlo qualifying simulations and return position records."""
        position_records = {d["driver"]: [] for d in all_drivers}

        noise_std_sprint = config_loader.get(
            "baseline_predictor.qualifying.noise_std_sprint", 0.025
        )
        noise_std_normal = config_loader.get("baseline_predictor.qualifying.noise_std_normal", 0.02)
        noise_std = noise_std_sprint if is_sprint else noise_std_normal

        team_weight = config_loader.get("baseline_predictor.qualifying.team_weight", 0.7)
        skill_weight = config_loader.get("baseline_predictor.qualifying.skill_weight", 0.3)
        team_strength_compression = config_loader.get(
            "baseline_predictor.qualifying.team_strength_compression", 0.50
        )
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

                score = (compressed_team_strength * team_weight) + (
                    bounded_driver_signal * skill_weight
                )
                score += rng.normal(0, teammate_setup_std)
                score += rng.normal(0, noise_std)

                driver_scores.append(
                    {
                        "driver": driver_info["driver"],
                        "team": driver_info["team"],
                        "score": score,
                    }
                )

            driver_scores.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(driver_scores):
                position_records[item["driver"]].append(i + 1)

        return position_records

    def _aggregate_grid_results(
        self, position_records: dict[str, list[int]], all_drivers: list[dict]
    ) -> list[dict]:
        """Aggregate simulation results into final grid with confidence intervals."""
        grid = []
        confidence_std_multiplier = config_loader.get(
            "baseline_predictor.qualifying.confidence_std_multiplier", 5.0
        )

        for driver_info in all_drivers:
            positions = position_records[driver_info["driver"]]
            median_pos = int(np.median(positions))
            p5 = int(np.percentile(positions, 5))
            p95 = int(np.percentile(positions, 95))

            position_std = np.std(positions)
            confidence = max(40, min(60, 60 - (position_std * confidence_std_multiplier)))

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

        grid.sort(key=lambda x: x["median_position"])

        for i, item in enumerate(grid):
            item["position"] = i + 1

        return grid

    def predict_qualifying(
        self,
        year: int,
        race_name: str,
        n_simulations: int = 50,
        qualifying_stage: str = "auto",
    ) -> dict[str, Any]:
        """Predict qualifying with Monte Carlo simulation (sprint/normal weekends)."""
        rng = np.random.default_rng(self.seed)

        validate_year(year, "year", min_year=2020, max_year=2030)
        validate_positive_int(n_simulations, "n_simulations", min_val=1)
        validate_enum(qualifying_stage, "qualifying_stage", ["auto", "sprint", "main"])

        try:
            is_sprint = is_sprint_weekend(year, race_name)
        except (ValueError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Could not determine sprint weekend for {race_name}: {e}")
            is_sprint = False

        lineups = get_lineups(year, race_name)

        session_name, fp_performance, session_laps = get_best_fp_performance(
            year=year,
            race_name=race_name,
            is_sprint=is_sprint,
            qualifying_stage=qualifying_stage,
        )

        if session_laps is not None:
            self._update_compound_characteristics_from_session(
                session_laps, race_name, year, is_sprint
            )

        all_drivers, teams_with_short_profile = self._build_driver_list_with_strengths(
            lineups, fp_performance, race_name, is_sprint
        )

        position_records = self._run_qualifying_simulations(
            all_drivers, n_simulations, is_sprint, rng
        )

        grid = self._aggregate_grid_results(position_records, all_drivers)

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
        """Predict Sprint Race with reduced chaos and increased grid influence."""
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)

        result = self.predict_race(
            qualifying_grid=sprint_quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
            is_sprint=True,
        )

        return result
