"""Race preparation helpers for Baseline2026Predictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.utils import config_loader
from src.utils.schema_validation import validate_track_characteristics

logger = logging.getLogger("src.predictors.baseline_2026")


class BaselineRacePreparationMixin:
    """Race preparation methods for Baseline2026Predictor."""

    def _load_track_overtaking_difficulty(self, race_name: str | None) -> float:
        """Load track overtaking difficulty from characteristics file."""
        if not race_name:
            return 0.5

        try:
            track_file = Path(
                "data/processed/track_characteristics/2026_track_characteristics.json"
            )
            with open(track_file) as f:
                track_data = json.load(f)
                validate_track_characteristics(track_data)
                tracks = track_data["tracks"]
                return tracks.get(race_name, {}).get("overtaking_difficulty", 0.5)
        except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not load track characteristics: {e}. Using default 0.5.")
            return 0.5

    def _prepare_driver_info(
        self,
        qualifying_grid: list[dict],
        race_name: str | None,
        race_compound: str = "MEDIUM",
    ) -> tuple[dict, int]:
        """Build driver info map with team strength, profile modifiers, skills, and DNF probabilities."""
        driver_info_map = {}
        teams_with_long_profile = set()

        dnf_rate_historical_cap = config_loader.get(
            "baseline_predictor.race.dnf_rate_historical_cap", 0.20
        )
        dnf_rate_final_cap = config_loader.get("baseline_predictor.race.dnf_rate_final_cap", 0.35)
        long_profile_scale = config_loader.get(
            "baseline_predictor.race.testing_long_run_modifier_scale", 0.05
        )
        long_profile_weights = {
            "overall_pace": 0.50,
            "tire_deg_performance": 0.35,
            "consistency": 0.15,
        }

        for entry in qualifying_grid:
            driver_code = entry["driver"]
            team = entry["team"]
            grid_pos = entry["position"]

            # Get base team strength with compound adjustment
            if race_name:
                team_strength = self.get_compound_adjusted_team_strength(
                    team, race_name, race_compound
                )
            else:
                team_strength = self.teams.get(team, {}).get("overall_performance", 0.50)

            # Add long-run testing profile modifier
            long_modifier, has_long_profile = self._compute_testing_profile_modifier(
                team=team,
                profile="long_run",
                metric_weights=long_profile_weights,
                scale=long_profile_scale,
            )
            team_strength = float(np.clip(team_strength + long_modifier, 0.0, 1.0))
            if has_long_profile:
                teams_with_long_profile.add(team)

            driver_data = self.drivers.get(driver_code, {})

            pace_data = driver_data.get("pace", {})
            quali_pace = pace_data.get("quali_pace", 0.5)
            race_pace = pace_data.get("race_pace", 0.5)
            race_advantage = race_pace - quali_pace

            racecraft = driver_data.get("racecraft", {})
            skill = racecraft.get("skill_score", 0.5)
            overtaking_skill = racecraft.get("overtaking_skill", 0.5)
            defensive_skill = racecraft.get("defensive_skill")
            if defensive_skill is None:
                # Backward-compatible fallback: reuse core racecraft signal when
                # explicit defensive skill is not present in extracted data.
                defensive_skill = (0.65 * skill) + (0.35 * overtaking_skill)
            defensive_skill = float(np.clip(defensive_skill, 0.0, 1.0))

            dnf_rate = min(
                driver_data.get("dnf_risk", {}).get("dnf_rate", 0.10),
                dnf_rate_historical_cap,
            )

            experience_tier = driver_data.get("experience", {}).get("tier", "established")
            experience_modifiers = {
                "rookie": 0.05,
                "developing": 0.02,
                "established": 0.00,
                "veteran": -0.01,
            }
            experience_dnf_modifier = experience_modifiers.get(experience_tier, 0.0)

            team_uncertainty = self.teams.get(team, {}).get("uncertainty", 0.30)
            if team_uncertainty >= 0.40:
                adjusted_dnf = dnf_rate + experience_dnf_modifier + (team_uncertainty * 0.20)
            else:
                adjusted_dnf = dnf_rate + experience_dnf_modifier

            driver_info_map[driver_code] = {
                "driver": driver_code,
                "team": team,
                "grid_pos": grid_pos,
                "team_strength": team_strength,
                "skill": skill,
                "race_advantage": race_advantage,
                "overtaking_skill": overtaking_skill,
                "defensive_skill": defensive_skill,
                "dnf_probability": max(0.0, min(adjusted_dnf, dnf_rate_final_cap)),
            }

        return driver_info_map, len(teams_with_long_profile)

    def _prepare_driver_info_with_compounds(
        self,
        qualifying_grid: list[dict],
        race_name: str | None,
    ) -> tuple[dict, int]:
        """Build driver info map with per-compound team strengths for lap-by-lap simulation."""
        from src.utils.compound_performance import get_compound_performance_modifier

        driver_info_map = {}
        teams_with_long_profile = set()

        dnf_rate_historical_cap = config_loader.get(
            "baseline_predictor.race.dnf_rate_historical_cap", 0.20
        )
        dnf_rate_final_cap = config_loader.get("baseline_predictor.race.dnf_rate_final_cap", 0.35)
        long_profile_scale = config_loader.get(
            "baseline_predictor.race.testing_long_run_modifier_scale", 0.05
        )
        long_profile_weights = {
            "overall_pace": 0.50,
            "tire_deg_performance": 0.35,
            "consistency": 0.15,
        }
        default_tire_deg_slope = config_loader.get(
            "baseline_predictor.race.tire_physics.default_deg_slope", 0.15
        )

        for entry in qualifying_grid:
            driver_code = entry["driver"]
            team = entry["team"]
            grid_pos = entry["position"]

            # Use weekend-aware team strength so race simulation starts from the same
            # blended baseline logic as qualifying.
            if race_name:
                base_team_strength = self.get_blended_team_strength(team, race_name)
            else:
                base_team_strength = self.teams.get(team, {}).get("overall_performance", 0.50)

            # Add long-run testing profile modifier
            long_modifier, has_long_profile = self._compute_testing_profile_modifier(
                team=team,
                profile="long_run",
                metric_weights=long_profile_weights,
                scale=long_profile_scale,
            )
            base_team_strength = float(np.clip(base_team_strength + long_modifier, 0.0, 1.0))
            if has_long_profile:
                teams_with_long_profile.add(team)

            # Pre-compute per-compound team strengths
            team_compound_chars = self.teams.get(team, {}).get("compound_characteristics", {})

            team_strength_by_compound = {}
            tire_deg_by_compound = {}

            for compound in ["SOFT", "MEDIUM", "HARD"]:
                if compound in team_compound_chars:
                    # Compound-specific modifier
                    modifier = get_compound_performance_modifier(team_compound_chars, compound)
                    adjusted_strength = base_team_strength + modifier

                    # Tire degradation slope
                    tire_deg_slope = team_compound_chars[compound].get(
                        "tire_deg_slope", default_tire_deg_slope
                    )
                else:
                    # Fallback: no compound data available
                    adjusted_strength = base_team_strength
                    tire_deg_slope = default_tire_deg_slope

                team_strength_by_compound[compound] = float(np.clip(adjusted_strength, 0.0, 1.0))
                tire_deg_by_compound[compound] = float(tire_deg_slope)

            # Get driver characteristics
            driver_data = self.drivers.get(driver_code, {})

            pace_data = driver_data.get("pace", {})
            quali_pace = pace_data.get("quali_pace", 0.5)
            race_pace = pace_data.get("race_pace", 0.5)
            race_advantage = race_pace - quali_pace

            racecraft = driver_data.get("racecraft", {})
            skill = racecraft.get("skill_score", 0.5)
            overtaking_skill = racecraft.get("overtaking_skill", 0.5)
            defensive_skill = racecraft.get("defensive_skill")
            if defensive_skill is None:
                defensive_skill = (0.65 * skill) + (0.35 * overtaking_skill)
            defensive_skill = float(np.clip(defensive_skill, 0.0, 1.0))

            # DNF probability
            dnf_rate = min(
                driver_data.get("dnf_risk", {}).get("dnf_rate", 0.10),
                dnf_rate_historical_cap,
            )

            experience_tier = driver_data.get("experience", {}).get("tier", "established")
            experience_modifiers = {
                "rookie": 0.05,
                "developing": 0.02,
                "established": 0.00,
                "veteran": -0.01,
            }
            experience_dnf_modifier = experience_modifiers.get(experience_tier, 0.0)

            team_uncertainty = self.teams.get(team, {}).get("uncertainty", 0.30)
            if team_uncertainty >= 0.40:
                adjusted_dnf = dnf_rate + experience_dnf_modifier + (team_uncertainty * 0.20)
            else:
                adjusted_dnf = dnf_rate + experience_dnf_modifier

            driver_info_map[driver_code] = {
                "driver": driver_code,
                "team": team,
                "grid_pos": grid_pos,
                "team_strength": base_team_strength,  # Base strength
                "team_strength_by_compound": team_strength_by_compound,  # Per-compound
                "tire_deg_by_compound": tire_deg_by_compound,  # Per-compound deg slopes
                "skill": skill,
                "race_advantage": race_advantage,
                "overtaking_skill": overtaking_skill,
                "defensive_skill": defensive_skill,
                "dnf_probability": max(0.0, min(adjusted_dnf, dnf_rate_final_cap)),
            }

        return driver_info_map, len(teams_with_long_profile)
