"""Qualifying and sprint-race mixin for Baseline2026Predictor."""

from __future__ import annotations

import logging
from hashlib import sha256
from typing import Any

import numpy as np

from src.types.prediction_types import QualifyingGridEntry
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

    def _get_testing_profile_weights(
        self, profile: str, defaults: dict[str, float]
    ) -> dict[str, float]:
        """Get configured testing profile weights with safe fallback."""
        cfg = getattr(self, "config", config_loader)
        weights = cfg.get(f"baseline_predictor.race.testing_profile_weights.{profile}", defaults)
        return weights if isinstance(weights, dict) and weights else defaults

    def _build_driver_list_with_strengths(
        self,
        lineups: dict[str, list[str]],
        fp_performance: dict[str, float] | None,
        race_name: str,
        is_sprint: bool,
    ) -> tuple[list[dict], int]:
        """Build driver list with blended team/driver strengths and testing modifiers."""
        cfg = getattr(self, "config", config_loader)
        all_drivers = []
        model_strengths = {}
        teams_with_short_profile = 0

        short_profile_scale = cfg.get(
            "baseline_predictor.qualifying.testing_short_run_modifier_scale", 0.04
        )
        short_profile_weights = self._get_testing_profile_weights(
            "short_run",
            {
                "overall_pace": 0.55,
                "top_speed": 0.20,
                "medium_corner_performance": 0.15,
                "fast_corner_performance": 0.10,
            },
        )
        fp_blend_weight = cfg.get("baseline_predictor.qualifying.fp_blend_weight", 0.7)
        default_skill = cfg.get("baseline_predictor.qualifying.default_skill", 0.5)
        default_team_strength = cfg.get("baseline_predictor.qualifying.default_team_strength", 0.5)

        for team in lineups:
            model_strength = self.get_blended_team_strength(team, race_name)
            short_modifier, has_short_profile = self._compute_testing_profile_modifier(
                team=team,
                profile="short_run",
                metric_weights=short_profile_weights,
                scale=short_profile_scale,
            )
            model_strength = np.clip(model_strength + short_modifier, 0.0, 1.0)
            if has_short_profile:
                teams_with_short_profile += 1
            model_strengths[team] = model_strength

        blended_strengths = blend_team_strength(
            model_strengths,
            fp_performance,
            blend_weight=fp_blend_weight,
        )

        for team, drivers in lineups.items():
            team_strength = blended_strengths.get(team, default_team_strength)
            for driver_code in drivers:
                driver_data = self.drivers.get(driver_code)
                if not driver_data:
                    fallback_loader = getattr(self, "_get_driver_data_or_fallback", None)
                    if callable(fallback_loader):
                        try:
                            driver_data = fallback_loader(driver_code, team)
                        except ValueError:
                            driver_data = {}
                    else:
                        driver_data = {}
                skill = driver_data.get("racecraft", {}).get("skill_score", default_skill)
                quali_pace = driver_data.get("pace", {}).get("quali_pace", 0.5)

                all_drivers.append(
                    {
                        "driver": driver_code,
                        "team": team,
                        "team_strength": team_strength,
                        "skill": skill,
                        "quali_pace": quali_pace,
                        "experience_tier": driver_data.get("experience", {}).get("tier", "unknown"),
                    }
                )

        return all_drivers, teams_with_short_profile

    def _run_qualifying_simulations(
        self,
        all_drivers: list[dict],
        n_simulations: int,
        is_sprint: bool,
        has_practice_data: bool,
        rng: np.random.Generator,
    ) -> dict[str, list[int]]:
        """Run Monte Carlo qualifying simulations and return position records."""
        cfg = getattr(self, "config", config_loader)
        position_records = {d["driver"]: [] for d in all_drivers}

        noise_std_sprint = cfg.get("baseline_predictor.qualifying.noise_std_sprint", 0.025)
        noise_std_normal = cfg.get("baseline_predictor.qualifying.noise_std_normal", 0.02)
        noise_std = noise_std_sprint if is_sprint else noise_std_normal

        team_weight = cfg.get("baseline_predictor.qualifying.team_weight", 0.7)
        skill_weight = cfg.get("baseline_predictor.qualifying.skill_weight", 0.3)
        team_strength_compression = cfg.get(
            "baseline_predictor.qualifying.team_strength_compression", 0.50
        )
        driver_quali_pace_weight = cfg.get(
            "baseline_predictor.qualifying.driver_quali_pace_weight", 0.70
        )
        driver_skill_weight = cfg.get("baseline_predictor.qualifying.driver_skill_weight", 0.30)
        driver_weight_sum = driver_quali_pace_weight + driver_skill_weight
        if driver_weight_sum <= 0:
            driver_quali_pace_weight, driver_skill_weight = 0.70, 0.30
            driver_weight_sum = 1.0
        model_only_driver_signal_shrink = cfg.get(
            "baseline_predictor.qualifying.model_only_driver_signal_shrink", 0.35
        )
        model_only_driver_signal_shrink = float(np.clip(model_only_driver_signal_shrink, 0.0, 1.0))
        model_only_experience_shrink = cfg.get(
            "baseline_predictor.qualifying.model_only_experience_shrink",
            {
                "rookie": 0.45,
                "developing": 0.20,
                "unknown": 0.30,
            },
        )
        if not isinstance(model_only_experience_shrink, dict):
            model_only_experience_shrink = {}
        team_driver_signal_means: dict[str, float] = {}
        for driver_info in all_drivers:
            driver_signal = (
                (driver_info["quali_pace"] * driver_quali_pace_weight)
                + (driver_info["skill"] * driver_skill_weight)
            ) / driver_weight_sum
            team_driver_signal_means.setdefault(driver_info["team"], 0.0)
            team_driver_signal_means[driver_info["team"]] += driver_signal
        team_counts: dict[str, int] = {}
        for driver_info in all_drivers:
            team_counts[driver_info["team"]] = team_counts.get(driver_info["team"], 0) + 1
        for team_name, total_signal in team_driver_signal_means.items():
            count = team_counts.get(team_name, 1)
            team_driver_signal_means[team_name] = total_signal / count

        driver_offset_cap = cfg.get("baseline_predictor.qualifying.driver_offset_cap", 0.18)
        driver_signal_softness = cfg.get(
            "baseline_predictor.qualifying.driver_signal_softness", 0.20
        )
        if driver_signal_softness <= 0:
            driver_signal_softness = 0.20
        teammate_setup_std = cfg.get("baseline_predictor.qualifying.teammate_setup_std", 0.015)
        if not has_practice_data:
            team_weight *= cfg.get(
                "baseline_predictor.qualifying.model_only_team_weight_multiplier", 0.82
            )
            skill_weight *= cfg.get(
                "baseline_predictor.qualifying.model_only_skill_weight_multiplier", 1.35
            )
            total_weight = team_weight + skill_weight
            if total_weight <= 0:
                team_weight, skill_weight = 0.66, 0.34
            else:
                team_weight /= total_weight
                skill_weight /= total_weight

            team_strength_compression *= cfg.get(
                "baseline_predictor.qualifying.model_only_team_compression_multiplier", 0.87
            )
            team_strength_compression = float(np.clip(team_strength_compression, 0.20, 1.0))

            driver_offset_cap *= cfg.get(
                "baseline_predictor.qualifying.model_only_driver_offset_cap_multiplier", 1.33
            )
            driver_offset_cap = float(np.clip(driver_offset_cap, 0.05, 0.30))

            noise_std *= cfg.get("baseline_predictor.qualifying.model_only_noise_multiplier", 1.12)
            teammate_setup_std *= cfg.get(
                "baseline_predictor.qualifying.model_only_teammate_setup_multiplier", 1.10
            )

        weekend_form_std = cfg.get("baseline_predictor.qualifying.weekend_form_std", 0.0)
        if not has_practice_data:
            weekend_form_std *= cfg.get(
                "baseline_predictor.qualifying.model_only_weekend_form_multiplier", 1.0
            )
        weekend_form = {
            d["driver"]: rng.normal(0, weekend_form_std) if weekend_form_std > 0 else 0.0
            for d in all_drivers
        }

        for _ in range(n_simulations):
            driver_scores = []
            for driver_info in all_drivers:
                compressed_team_strength = 0.5 + (
                    (driver_info["team_strength"] - 0.5) * team_strength_compression
                )
                compressed_team_strength = np.clip(compressed_team_strength, 0.0, 1.0)

                driver_signal = (
                    (driver_info["quali_pace"] * driver_quali_pace_weight)
                    + (driver_info["skill"] * driver_skill_weight)
                ) / driver_weight_sum
                if not has_practice_data and model_only_driver_signal_shrink > 0:
                    team_mean = team_driver_signal_means.get(driver_info["team"], driver_signal)
                    experience_tier = str(driver_info.get("experience_tier", "unknown"))
                    extra_shrink = model_only_experience_shrink.get(
                        experience_tier, model_only_experience_shrink.get("unknown", 0.0)
                    )
                    total_shrink = float(
                        np.clip(model_only_driver_signal_shrink + float(extra_shrink), 0.0, 0.95)
                    )
                    driver_signal = team_mean + ((driver_signal - team_mean) * (1.0 - total_shrink))
                bounded_driver_signal = 0.5 + (
                    np.tanh((driver_signal - 0.5) / driver_signal_softness) * driver_offset_cap
                )

                score = (compressed_team_strength * team_weight) + (
                    bounded_driver_signal * skill_weight
                )
                score += weekend_form.get(driver_info["driver"], 0.0)
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
    ) -> list[QualifyingGridEntry]:
        """Aggregate simulation results into final grid with confidence intervals."""
        cfg = getattr(self, "config", config_loader)
        grid: list[QualifyingGridEntry] = []
        confidence_std_multiplier = cfg.get(
            "baseline_predictor.qualifying.confidence_std_multiplier", 5.0
        )
        confidence_cap = cfg.get("baseline_predictor.qualifying.confidence_cap", 60)
        confidence_min = cfg.get("baseline_predictor.qualifying.confidence_min", 40)

        for driver_info in all_drivers:
            positions = position_records[driver_info["driver"]]
            median_pos = int(np.median(positions))
            mean_pos = float(np.mean(positions))
            p5 = int(np.percentile(positions, 5))
            p95 = int(np.percentile(positions, 95))

            position_std = np.std(positions)
            confidence = max(
                confidence_min,
                min(confidence_cap, confidence_cap - (position_std * confidence_std_multiplier)),
            )

            grid.append(
                {
                    "driver": driver_info["driver"],
                    "team": driver_info["team"],
                    "position": median_pos,
                    "median_position": median_pos,
                    "_mean_position": mean_pos,
                    "p5": p5,
                    "p95": p95,
                    "confidence": float(round(confidence, 1)),
                }
            )

        # Resolve median ties with the underlying simulation mean so teammate order
        # does not collapse into insertion-order blocks when medians are equal.
        grid.sort(key=lambda x: (x["median_position"], x["_mean_position"], x["driver"]))

        for i, item in enumerate(grid):
            item["position"] = i + 1
            item.pop("_mean_position", None)

        return grid

    def predict_qualifying(
        self,
        year: int,
        race_name: str,
        n_simulations: int = 50,
        qualifying_stage: str = "auto",
    ) -> dict[str, Any]:
        """Predict qualifying with Monte Carlo simulation (sprint/normal weekends)."""
        cfg = getattr(self, "config", config_loader)

        validate_year(year, "year", min_year=2020, max_year=2030)
        validate_positive_int(n_simulations, "n_simulations", min_val=1)
        validate_enum(qualifying_stage, "qualifying_stage", ["auto", "sprint", "main"])

        try:
            is_sprint = is_sprint_weekend(year, race_name)
        except (ValueError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Could not determine sprint weekend for {race_name}: {e}")
            is_sprint = False

        seed_material = f"{self.seed}:{year}:{race_name}:{qualifying_stage}:{int(is_sprint)}"
        seed = int(sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)

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
        if cfg.get("baseline_predictor.qualifying.enable_driver_fp_adjustment", True):
            from src.utils.driver_fp_adjustment import calculate_driver_fp_modifiers

            fp_session_types = ["FP1"] if is_sprint else ["FP1", "FP2", "FP3"]
            modifier_scale = cfg.get(
                "baseline_predictor.qualifying.driver_fp_adjustment_scale", 0.10
            )
            smoothing_seconds = cfg.get(
                "baseline_predictor.qualifying.driver_fp_adjustment_smoothing", 0.50
            )
            driver_fp_modifiers = calculate_driver_fp_modifiers(
                year=year,
                race_name=race_name,
                session_types=fp_session_types,
                scale=modifier_scale,
                smoothing_seconds=smoothing_seconds,
            )
            for driver_info in all_drivers:
                fp_modifier = driver_fp_modifiers.get(driver_info["driver"], 0.0)
                if fp_modifier == 0.0:
                    continue
                driver_info["skill"] = np.clip(driver_info["skill"] + fp_modifier, 0.01, 0.99)

        position_records = self._run_qualifying_simulations(
            all_drivers, n_simulations, is_sprint, session_name is not None, rng
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
