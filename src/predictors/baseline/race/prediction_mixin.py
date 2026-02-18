"""Race prediction method for Baseline2026Predictor."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils import config_loader
from src.utils.lap_by_lap_simulator import (
    aggregate_simulation_results,
    simulate_race_lap_by_lap,
)
from src.utils.pit_strategy import generate_pit_strategy
from src.utils.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
    resolve_race_distance_laps,
)
from src.utils.validation_helpers import validate_enum, validate_positive_int


class BaselineRacePredictionMixin:
    """Race prediction method implementation for Baseline2026Predictor."""

    def predict_race(
        self,
        qualifying_grid: list[dict],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
        is_sprint: bool = False,
        race_compound: str = "MEDIUM",
    ) -> dict[str, Any]:
        """Predict race result using lap-by-lap Monte Carlo simulation with tire deg and pit stops."""
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)

        # Load track-specific parameters (pit loss, safety car prob, overtaking)
        track_params = load_track_specific_params(race_name)

        # Load base race parameters from config
        base_params = self._load_race_params()

        # Merge track-specific overrides into base params
        race_params = {**base_params, **track_params}

        # Load additional params for lap-by-lap simulation
        race_params["fuel"] = {
            "initial_load_kg": config_loader.get(
                "baseline_predictor.race.fuel.initial_load_kg", 110.0
            ),
            "effect_per_lap": config_loader.get(
                "baseline_predictor.race.fuel.effect_per_lap", 0.035
            ),
            "burn_rate_kg_per_lap": config_loader.get(
                "baseline_predictor.race.fuel.burn_rate_kg_per_lap", 1.5
            ),
        }

        race_params["lap_time"] = {
            "reference_base": config_loader.get(
                "baseline_predictor.race.lap_time.reference_base", 90.0
            ),
            "team_pace_penalty_range": config_loader.get(
                "baseline_predictor.race.lap_time.team_pace_penalty_range", 5.0
            ),
            "skill_improvement_max": config_loader.get(
                "baseline_predictor.race.lap_time.skill_improvement_max", 0.5
            ),
            "bounds": config_loader.get("baseline_predictor.race.lap_time.bounds", [70.0, 120.0]),
            "elite_skill_threshold": config_loader.get(
                "baseline_predictor.race.lap_time.elite_skill_threshold", 0.88
            ),
            "elite_skill_lap_bonus_max": config_loader.get(
                "baseline_predictor.race.lap_time.elite_skill_lap_bonus_max", 0.09
            ),
            "elite_skill_exponent": config_loader.get(
                "baseline_predictor.race.lap_time.elite_skill_exponent", 1.3
            ),
        }
        race_params["team_strength_compression"] = config_loader.get(
            "baseline_predictor.race.lap_time.team_strength_compression", 0.35
        )
        race_params["start_grid_gap_seconds"] = config_loader.get(
            "baseline_predictor.race.start_grid_gap_seconds", 0.32
        )
        race_params["race_advantage_lap_impact"] = config_loader.get(
            "baseline_predictor.race.race_advantage_lap_impact", 0.35
        )
        race_params["overtake_model"] = {
            "dirty_air_window_s": config_loader.get(
                "baseline_predictor.race.overtake_model.dirty_air_window_s", 1.8
            ),
            "dirty_air_penalty_base": config_loader.get(
                "baseline_predictor.race.overtake_model.dirty_air_penalty_base", 0.05
            ),
            "dirty_air_penalty_track_scale": config_loader.get(
                "baseline_predictor.race.overtake_model.dirty_air_penalty_track_scale",
                0.12,
            ),
            "pass_window_s": config_loader.get(
                "baseline_predictor.race.overtake_model.pass_window_s", 1.2
            ),
            "pass_threshold_base": config_loader.get(
                "baseline_predictor.race.overtake_model.pass_threshold_base", 0.06
            ),
            "pass_threshold_track_scale": config_loader.get(
                "baseline_predictor.race.overtake_model.pass_threshold_track_scale",
                0.16,
            ),
            "pass_probability_base": config_loader.get(
                "baseline_predictor.race.overtake_model.pass_probability_base", 0.30
            ),
            "pass_probability_scale": config_loader.get(
                "baseline_predictor.race.overtake_model.pass_probability_scale", 0.45
            ),
            "pass_time_bonus_range": config_loader.get(
                "baseline_predictor.race.overtake_model.pass_time_bonus_range",
                [0.08, 0.35],
            ),
            "pace_diff_scale": config_loader.get(
                "baseline_predictor.race.overtake_model.pace_diff_scale", 0.55
            ),
            "skill_scale": config_loader.get(
                "baseline_predictor.race.overtake_model.skill_scale", 0.25
            ),
            "defense_scale": config_loader.get(
                "baseline_predictor.race.overtake_model.defense_scale", 0.28
            ),
            "race_adv_scale": config_loader.get(
                "baseline_predictor.race.overtake_model.race_adv_scale", 0.20
            ),
            "track_ease_scale": config_loader.get(
                "baseline_predictor.race.overtake_model.track_ease_scale", 0.18
            ),
        }

        # Prepare driver info with per-compound strengths
        driver_info_map, teams_with_long_profile = self._prepare_driver_info_with_compounds(
            qualifying_grid, race_name
        )

        # Determine race distance from FastF1 metadata, with safe fallback defaults.
        race_distance = resolve_race_distance_laps(
            year=2026,
            race_name=race_name,
            is_sprint=is_sprint,
        )

        # Get tire stress and available compounds
        tire_stress_score = get_tire_stress_score(race_name)
        available_compounds = get_available_compounds(race_name, weather=weather)
        # If no full-wet race is modeled, enforce FIA dry-compound diversity.
        # This keeps mixed forecasts from degenerating into illegal SOFT→SOFT plans.
        enforce_two_compound_rule = weather in {"dry", "mixed"}

        # Restructure race_params for lap_by_lap_simulator (expects nested dicts)
        race_params["base_chaos"] = {
            "dry": race_params.get("base_chaos_dry", 0.35),
            "wet": race_params.get("base_chaos_wet", 0.45),
        }
        race_params["lap1_chaos"] = {
            "front_row": race_params.get("lap1_front_row_chaos", 0.15),
            "upper_midfield": race_params.get("lap1_upper_midfield_chaos", 0.32),
            "midfield": race_params.get("lap1_midfield_chaos", 0.38),
            "back_field": race_params.get("lap1_back_field_chaos", 0.28),
        }
        if "track_overtaking" not in race_params:
            race_params["track_overtaking"] = config_loader.get(
                "track_defaults.overtaking_difficulty", 0.5
            )

        sc_weather_key = "sc_base_prob_wet" if weather in ["rain", "mixed"] else "sc_base_prob_dry"
        default_sc_probability = race_params.get(sc_weather_key, 0.45) + (
            race_params["track_overtaking"] * race_params.get("sc_track_modifier", 0.25)
        )
        race_params["sc_probability"] = race_params.get(
            "sc_probability", float(np.clip(default_sc_probability, 0.0, 1.0))
        )

        # Ensure pit_stops key exists (may come from track_params or need default)
        if "pit_stops" not in race_params:
            race_params["pit_stops"] = {
                "loss_duration": 22.0,  # Default average
                "overtake_loss_range": [0, 3],
            }

        # Run lap-by-lap simulations
        simulation_results = []
        base_seed = int(getattr(self, "seed", 42))

        for sim_idx in range(n_simulations):
            rng = np.random.default_rng(base_seed + sim_idx)

            # Generate pit strategies for all drivers (Monte Carlo)
            strategies = {}
            sprint_compound = (
                "SOFT"
                if "SOFT" in available_compounds
                else (available_compounds[0] if available_compounds else "MEDIUM")
            )
            for driver in driver_info_map.keys():
                if is_sprint:
                    # Sprint races run without scheduled pit stops in this model.
                    strategies[driver] = {
                        "num_stops": 0,
                        "pit_laps": [],
                        "compound_sequence": [sprint_compound],
                        "stint_lengths": [race_distance],
                    }
                else:
                    strategies[driver] = generate_pit_strategy(
                        race_distance=race_distance,
                        tire_stress_score=tire_stress_score,
                        available_compounds=available_compounds,
                        rng=rng,
                        enforce_two_compound_rule=enforce_two_compound_rule,
                    )

            # Simulate race lap-by-lap
            sim_result = simulate_race_lap_by_lap(
                driver_info_map=driver_info_map,
                strategies=strategies,
                race_params=race_params,
                race_distance=race_distance,
                weather=weather,
                rng=rng,
            )

            simulation_results.append(sim_result)

        # Aggregate results across all simulations
        aggregated = aggregate_simulation_results(simulation_results)

        # Blend race simulation output with grid anchoring based on overtaking difficulty.
        # Hard-to-pass tracks preserve more of qualifying order, while easy tracks let
        # pace and racecraft dominate more.
        track_overtaking = float(race_params.get("track_overtaking", 0.5))
        grid_anchor_weight = float(
            np.clip(
                config_loader.get("baseline_predictor.race.grid_anchor.base", 0.30)
                + (
                    track_overtaking
                    * config_loader.get("baseline_predictor.race.grid_anchor.track_scale", 0.35)
                ),
                0.20,
                0.85,
            )
        )
        grid_anchor_min = config_loader.get("baseline_predictor.race.grid_anchor.min", 0.62)
        sprint_grid_anchor_min = config_loader.get(
            "baseline_predictor.race.grid_anchor.sprint_min", 0.78
        )
        grid_anchor_weight = max(
            grid_anchor_weight,
            sprint_grid_anchor_min if is_sprint else grid_anchor_min,
        )
        overtaking_skill_blend_scale = config_loader.get(
            "baseline_predictor.race.final_blend.overtaking_skill_scale", 1.6
        )
        race_advantage_blend_scale = config_loader.get(
            "baseline_predictor.race.final_blend.race_advantage_scale", 1.3
        )
        driver_skill_blend_scale = config_loader.get(
            "baseline_predictor.race.final_blend.driver_skill_scale", 1.1
        )
        elite_driver_skill_threshold = float(
            config_loader.get(
                "baseline_predictor.race.final_blend.elite_driver_skill_threshold", 0.88
            )
        )
        elite_driver_scale = float(
            config_loader.get("baseline_predictor.race.final_blend.elite_driver_scale", 0.80)
        )
        elite_driver_exponent = float(
            config_loader.get("baseline_predictor.race.final_blend.elite_driver_exponent", 1.35)
        )
        max_driver_adjustment_positions = float(
            config_loader.get(
                "baseline_predictor.race.final_blend.max_driver_adjustment_positions",
                0.9,
            )
        )
        confidence_floor = float(config_loader.get("baseline_predictor.race.confidence.min", 40.0))
        weather_confidence_penalty = float(
            config_loader.get("baseline_predictor.race.confidence.weather_penalty_wet", 4.0)
            if weather in ("rain", "mixed")
            else 0.0
        )

        # Build finish order from blended position scores
        finish_order = []
        for driver_code, median_pos in aggregated["median_positions"].items():
            info = driver_info_map[driver_code]
            positions = aggregated["position_distributions"][driver_code]

            # Confidence based on consistency; wet/mixed runs receive an uncertainty penalty.
            position_std = np.std(positions)
            confidence = max(
                confidence_floor,
                min(60.0, 60.0 - (position_std * 3.0) - weather_confidence_penalty),
            )

            overtake_ease = 1.0 - track_overtaking
            racecraft_adjustment = (
                ((info["overtaking_skill"] - 0.5) * overtake_ease * overtaking_skill_blend_scale)
                + (info["race_advantage"] * race_advantage_blend_scale)
                + ((info["skill"] - 0.5) * driver_skill_blend_scale)
            )

            elite_denominator = max(1e-6, 1.0 - elite_driver_skill_threshold)
            elite_driver_normalized = max(
                0.0, (info["skill"] - elite_driver_skill_threshold) / elite_denominator
            )
            elite_driver_adjustment = (
                (elite_driver_normalized**elite_driver_exponent)
                * elite_driver_scale
                * (0.6 + (0.4 * overtake_ease))
            )
            racecraft_adjustment += elite_driver_adjustment

            is_elite_driver = info["skill"] >= elite_driver_skill_threshold
            if info["grid_pos"] <= 3 and not is_elite_driver:
                adjustment_cap_negative = max_driver_adjustment_positions * 0.5
                adjustment_cap_positive = max_driver_adjustment_positions
                racecraft_adjustment = float(
                    np.clip(
                        racecraft_adjustment,
                        -adjustment_cap_negative,
                        adjustment_cap_positive,
                    )
                )
            else:
                racecraft_adjustment = float(
                    np.clip(
                        racecraft_adjustment,
                        -max_driver_adjustment_positions,
                        max_driver_adjustment_positions,
                    )
                )

            position_blend_score = (
                ((1.0 - grid_anchor_weight) * median_pos)
                + (grid_anchor_weight * info["grid_pos"])
                - racecraft_adjustment
            )

            blended_position_samples = [
                ((1.0 - grid_anchor_weight) * p)
                + (grid_anchor_weight * info["grid_pos"])
                - racecraft_adjustment
                for p in positions
            ]

            p5 = int(np.percentile(blended_position_samples, 5))
            p95 = int(np.percentile(blended_position_samples, 95))

            podium_prob = (
                sum(1 for p in blended_position_samples if p <= 3.0)
                / len(blended_position_samples)
                * 100.0
            )

            finish_order.append(
                {
                    "driver": driver_code,
                    "team": info["team"],
                    "median_position": median_pos,
                    "position_blend_score": round(position_blend_score, 4),
                    "p5": p5,
                    "p95": p95,
                    "confidence": round(confidence, 1),
                    "podium_probability": round(podium_prob, 1),
                    "dnf_probability": round(aggregated["dnf_rates"].get(driver_code, 0.0), 3),
                }
            )

        # Sort by blended position score
        finish_order.sort(key=lambda x: x["position_blend_score"])

        # Assign final positions
        for i, item in enumerate(finish_order):
            item["position"] = i + 1

        return {
            "finish_order": finish_order,
            "characteristics_profile_used": "long_run",
            "teams_with_characteristics_profile": teams_with_long_profile,
            "compound_strategies": aggregated["compound_strategy_distribution"],
            "pit_lap_distribution": aggregated["pit_lap_distribution"],
        }
