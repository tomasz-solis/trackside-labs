"""
2026 Baseline Predictor

Primary runtime predictor for 2026 qualifying and race simulations.

Combines:
- weight-scheduled team strength (baseline/testing/current season),
- optional session blending for qualifying,
- Monte Carlo race scoring with uncertainty and DNF modeling.
"""

import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils import config_loader
from src.utils.validation_helpers import (
    validate_positive_int,
    validate_enum,
    validate_year,
)
from src.utils.schema_validation import (
    validate_driver_characteristics,
    validate_team_characteristics,
    validate_track_characteristics,
)
from src.utils.lineups import get_lineups
from src.utils.weekend import is_sprint_weekend
from src.utils.data_generator import ensure_baseline_exists
from src.systems.weight_schedule import (
    calculate_blended_performance,
    get_recommended_schedule,
)
from src.utils.fp_blending import get_best_fp_performance, blend_team_strength

logger = logging.getLogger(__name__)


class Baseline2026Predictor:
    """
    Primary 2026 predictor used by the dashboard and compatibility wrappers.

    Uses:
    - Team strength from car characteristics (baseline + directionality + current season)
    - Driver skill and risk inputs from driver characteristics
    - Session blending for qualifying when data is available
    - Monte Carlo simulation for qualifying and race predictions
    """

    def __init__(self, data_dir: str = "data/processed"):
        """Initialize baseline 2026 predictor with team/driver data from data_dir."""
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

    def load_data(self) -> None:
        """Load 2026 team data and driver characteristics with schema validation."""
        # Load and validate 2026 car characteristics
        car_file = self.data_dir / "car_characteristics/2026_car_characteristics.json"
        with open(car_file) as f:
            data = json.load(f)
            # Validate team characteristics before using
            try:
                validate_team_characteristics(data)
            except ValueError as e:
                logger.error(f"Failed to load team characteristics: {e}")
                raise
            self.teams = data["teams"]

            # Check data freshness and warn if stale
            data_freshness = data.get("data_freshness", "UNKNOWN")
            races_completed = data.get("races_completed", 0)
            data.get("last_updated")

            if data_freshness == "BASELINE_PRESEASON":
                logger.warning(
                    "⚠️  Using PRE-SEASON BASELINE data - team performance highly uncertain until races complete!"
                )
            elif data_freshness == "LIVE_UPDATED":
                logger.info(
                    f"✓ Using LIVE data updated from {races_completed} race(s) - confidence increasing"
                )
            else:
                logger.warning(
                    f"⚠️  Data freshness unknown ({data_freshness}) - predictions may be outdated"
                )

        # Load and validate driver characteristics
        driver_file = self.data_dir / "driver_characteristics.json"
        with open(driver_file) as f:
            data = json.load(f)
            # Validate driver characteristics before using
            try:
                validate_driver_characteristics(data)
            except ValueError as e:
                logger.error(f"Failed to load driver characteristics: {e}")
                raise

            # ERROR DETECTION: Check for extraction bugs (does NOT correct)
            from src.utils.driver_validation import validate_driver_data

            errors = validate_driver_data(data["drivers"])
            if errors:
                logger.warning(
                    f"⚠️  Driver data has {len(errors)} validation errors. "
                    "Consider re-running extraction: python scripts/extract_driver_characteristics.py --years 2023,2024,2025"
                )

            self.drivers = data["drivers"]

        # Load track characteristics for weight schedule system
        track_file = self.data_dir / "track_characteristics/2026_track_characteristics.json"
        try:
            with open(track_file) as f:
                track_data = json.load(f)
                self.tracks = track_data.get("tracks", {})
                logger.info(f"✓ Loaded track characteristics for {len(self.tracks)} circuits")
        except FileNotFoundError:
            logger.warning(f"⚠️  Track characteristics not found at {track_file}")
            self.tracks = {}

        # Store races completed and year for weight schedule
        car_file = self.data_dir / "car_characteristics/2026_car_characteristics.json"
        with open(car_file) as f:
            data = json.load(f)
            self.races_completed = data.get("races_completed", 0)
            self.year = data.get("year", 2026)

    def calculate_track_suitability(self, team: str, race_name: str) -> float:
        """Calculate track-car suitability modifier (-0.1 to +0.1) based on car directionality vs track composition."""
        team_data = self.teams.get(team, {})
        directionality = team_data.get("directionality", {})

        # If no directionality data, return neutral
        if not directionality:
            return 0.0

        track_profile = self.tracks.get(race_name, {})

        # If track has no telemetry data, return neutral
        if "straights_pct" not in track_profile:
            return 0.0

        # Calculate weighted suitability based on track composition
        total_pct = (
            track_profile.get("straights_pct", 0)
            + track_profile.get("slow_corners_pct", 0)
            + track_profile.get("medium_corners_pct", 0)
            + track_profile.get("high_corners_pct", 0)
        )

        if total_pct == 0:
            return 0.0

        # Weighted combination of car strengths × track demands
        suitability = (
            directionality.get("max_speed", 0) * (track_profile.get("straights_pct", 0) / total_pct)
            + directionality.get("slow_corner_speed", 0)
            * (track_profile.get("slow_corners_pct", 0) / total_pct)
            + directionality.get("medium_corner_speed", 0)
            * (track_profile.get("medium_corners_pct", 0) / total_pct)
            + directionality.get("high_corner_speed", 0)
            * (track_profile.get("high_corners_pct", 0) / total_pct)
        )

        return suitability

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        """
        Calculate blended team strength using weight schedule system.

        Combines:
        1. Baseline (2025 standings) - decreases over season
        2. Testing directionality (track suitability) - decreases over season
        3. Current season (running average) - increases over season
        """
        team_data = self.teams.get(team, {})

        # 1. Baseline from 2025 standings
        baseline = team_data.get("overall_performance", 0.5)

        # 2. Testing modifier (track suitability)
        testing_modifier = self.calculate_track_suitability(team, race_name)

        # 3. Current season running average
        current_season_performance = team_data.get("current_season_performance", [])
        if current_season_performance:
            current = np.mean(current_season_performance)
        else:
            # Pre-season: use baseline as fallback for current
            current = baseline

        # 4. Apply weight schedule
        race_number = self.races_completed + 1  # Next race

        # Use "extreme" schedule for 2026 (regulation change)
        schedule = get_recommended_schedule(is_regulation_change=True)

        blended = calculate_blended_performance(
            baseline_score=baseline,
            testing_modifier=testing_modifier,
            current_score=current,
            race_number=race_number,
            schedule=schedule,
        )

        return blended

    def _get_testing_characteristics_for_profile(self, team: str, profile: str) -> dict[str, float]:
        """Get testing/practice characteristics for a profile with backward-compatible fallbacks."""
        team_data = self.teams.get(team, {})

        profile_store = team_data.get("testing_characteristics_profiles")
        if isinstance(profile_store, dict):
            profile_data = profile_store.get(profile)
            if isinstance(profile_data, dict):
                return profile_data

        fallback = team_data.get("testing_characteristics")
        if not isinstance(fallback, dict):
            return {}

        fallback_profile = fallback.get("run_profile")
        if fallback_profile == profile:
            return fallback

        # Older files may only store one profile in testing_characteristics.
        if profile == "balanced":
            return fallback

        return {}

    def _compute_testing_profile_modifier(
        self,
        team: str,
        profile: str,
        metric_weights: dict[str, float],
        scale: float,
    ) -> tuple[float, bool]:
        """
        Compute a small team-strength modifier from testing/practice characteristics.

        Returns (modifier, has_profile_data). Modifier is bounded to avoid overpowering
        the existing baseline + track-suitability + season-performance logic.
        """
        profile_metrics = self._get_testing_characteristics_for_profile(team, profile)
        if not profile_metrics:
            return 0.0, False

        weighted_sum = 0.0
        total_weight = 0.0
        for metric_name, weight in metric_weights.items():
            value = profile_metrics.get(metric_name)
            if value is None:
                continue
            centered = float(value) - 0.5
            weighted_sum += centered * float(weight)
            total_weight += float(weight)

        if total_weight <= 0:
            return 0.0, False

        normalized_centered = weighted_sum / total_weight
        modifier = float(np.clip(normalized_centered * float(scale), -0.04, 0.04))
        return modifier, True

    def predict_qualifying(
        self, year: int, race_name: str, n_simulations: int = 50
    ) -> Dict[str, Any]:
        """
        Predict qualifying order based on team + driver baseline.

        Runs multiple Monte Carlo simulations and returns median positions.
        Returns low confidence (40-60%) acknowledging regulation uncertainty.

        Sprint weekends: Predicts Friday Sprint Qualifying (sets Sprint Race grid)
        Normal weekends: Predicts standard Saturday Qualifying (sets Sunday Race grid)

        For F1 Fantasy users - this provides the grid BEFORE team lock:
        - Sprint: Lock before Sprint Race (Saturday) → predict from Friday Sprint Quali
        - Normal: Lock before Qualifying (Saturday) → predict qualifying outcome
        """
        # Validate inputs
        validate_year(year, "year", min_year=2020, max_year=2030)
        validate_positive_int(n_simulations, "n_simulations", min_val=1)

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
        session_name, fp_performance = get_best_fp_performance(year, race_name, is_sprint)

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

        for team, drivers in lineups.items():
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

                all_drivers.append(
                    {
                        "driver": driver_code,
                        "team": team,
                        "team_strength": team_strength,
                        "skill": skill,
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

        for _ in range(n_simulations):
            # Calculate scores with random noise
            driver_scores = []
            for driver_info in all_drivers:
                # Combined score: team_weight% team, skill_weight% driver
                score = (driver_info["team_strength"] * team_weight) + (
                    driver_info["skill"] * skill_weight
                )
                # Add random noise (more for sprint weekends)
                score += np.random.normal(0, noise_std)

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
            "characteristics_profile_used": "short_run",
            "teams_with_characteristics_profile": teams_with_short_profile,
        }

    def predict_sprint_race(
        self,
        sprint_quali_grid: List[Dict],
        weather: str = "dry",
        race_name: Optional[str] = None,
        n_simulations: int = 50,
    ) -> Dict[str, Any]:
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

    def _load_track_overtaking_difficulty(self, race_name: Optional[str]) -> float:
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
        qualifying_grid: List[Dict],
        race_name: Optional[str],
    ) -> tuple[Dict, int]:
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

            team_strength = (
                self.get_blended_team_strength(team, race_name)
                if race_name
                else self.teams.get(team, {}).get("overall_performance", 0.50)
            )
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
                "dnf_probability": max(0.0, min(adjusted_dnf, dnf_rate_final_cap)),
            }

        return driver_info_map, len(teams_with_long_profile)

    def _calculate_driver_race_score(
        self,
        info: Dict,
        track_overtaking: float,
        weather: str,
        safety_car: bool,
        params: Dict,
    ) -> tuple[float, bool]:
        """Calculate single driver's race score for one simulation."""
        # Grid advantage
        grid_weight = params["grid_weight_min"] + (
            track_overtaking * params["grid_weight_multiplier"]
        )
        grid_advantage = 1.0 - ((info["grid_pos"] - 1) / 21.0)

        # Position-dependent scaling
        if info["grid_pos"] <= 3:
            position_scaling = 0.1
        elif info["grid_pos"] <= 7:
            position_scaling = 0.3
        elif info["grid_pos"] <= 12:
            position_scaling = 0.6
        else:
            position_scaling = 1.0

        # Systematic factors
        race_pace_boost = (
            info["race_advantage"] * params["race_advantage_multiplier"] * position_scaling
        )

        if (
            info["grid_pos"] > params["overtaking_grid_threshold"]
            and track_overtaking < params["overtaking_track_threshold"]
        ):
            overtaking_boost = (
                (info["overtaking_skill"] - 0.5)
                * params["overtaking_skill_multiplier"]
                * position_scaling
            )
        else:
            overtaking_boost = 0

        # Random factors
        if info["grid_pos"] <= 3:
            lap1_chaos = np.random.normal(0, params["lap1_front_row_chaos"])
        elif info["grid_pos"] <= 10:
            lap1_chaos = np.random.normal(0, params["lap1_upper_midfield_chaos"])
        elif info["grid_pos"] <= 15:
            lap1_chaos = np.random.normal(0, params["lap1_midfield_chaos"])
        else:
            lap1_chaos = np.random.normal(0, params["lap1_back_field_chaos"])

        strategy_std = params["strategy_variance_base"] * (
            1.0 - track_overtaking * params["strategy_track_modifier"]
        )
        strategy_factor = np.random.uniform(-strategy_std, strategy_std)

        sc_luck = (
            np.random.uniform(-params["safety_car_luck_range"], params["safety_car_luck_range"])
            if safety_car
            else 0
        )

        # Base score
        pace_weight = params["pace_weight_base"] - (
            track_overtaking * params["pace_weight_track_modifier"]
        )
        driver_weight = 0.20
        total_weight = grid_weight + pace_weight + driver_weight
        normalized_grid = grid_weight / total_weight
        normalized_pace = pace_weight / total_weight
        normalized_skill = driver_weight / total_weight

        base_score = (
            (grid_advantage * normalized_grid)
            + (info["team_strength"] * normalized_pace)
            + (info["skill"] * normalized_skill)
        )
        teammate_variance = np.random.normal(0, params["teammate_variance_std"])

        # DNF check
        dnf_occurred = np.random.random() < info["dnf_probability"]

        if dnf_occurred:
            score = -10.0 + np.random.uniform(-1, 0)
        else:
            score = (
                base_score
                + race_pace_boost
                + overtaking_boost
                + np.random.normal(0, params["base_chaos"])
                + lap1_chaos
                + strategy_factor
                + sc_luck
                + teammate_variance
            )

        return score, dnf_occurred

    def _load_race_params(self) -> Dict:
        """Load all race parameters from config once."""
        return {
            "base_chaos_dry": config_loader.get("baseline_predictor.race.base_chaos.dry", 0.35),
            "base_chaos_wet": config_loader.get("baseline_predictor.race.base_chaos.wet", 0.45),
            "track_chaos_multiplier": config_loader.get(
                "baseline_predictor.race.track_chaos_multiplier", 0.4
            ),
            "sc_base_prob_dry": config_loader.get(
                "baseline_predictor.race.sc_base_probability.dry", 0.45
            ),
            "sc_base_prob_wet": config_loader.get(
                "baseline_predictor.race.sc_base_probability.wet", 0.70
            ),
            "sc_track_modifier": config_loader.get(
                "baseline_predictor.race.sc_track_modifier", 0.25
            ),
            "grid_weight_min": config_loader.get("baseline_predictor.race.grid_weight_min", 0.15),
            "grid_weight_multiplier": config_loader.get(
                "baseline_predictor.race.grid_weight_multiplier", 0.35
            ),
            "race_advantage_multiplier": config_loader.get(
                "baseline_predictor.race.race_advantage_multiplier", 0.5
            ),
            "overtaking_skill_multiplier": config_loader.get(
                "baseline_predictor.race.overtaking_skill_multiplier", 0.25
            ),
            "overtaking_grid_threshold": config_loader.get(
                "baseline_predictor.race.overtaking_grid_threshold", 5
            ),
            "overtaking_track_threshold": config_loader.get(
                "baseline_predictor.race.overtaking_track_threshold", 0.5
            ),
            "lap1_front_row_chaos": config_loader.get(
                "baseline_predictor.race.lap1_chaos.front_row", 0.15
            ),
            "lap1_upper_midfield_chaos": config_loader.get(
                "baseline_predictor.race.lap1_chaos.upper_midfield", 0.32
            ),
            "lap1_midfield_chaos": config_loader.get(
                "baseline_predictor.race.lap1_chaos.midfield", 0.38
            ),
            "lap1_back_field_chaos": config_loader.get(
                "baseline_predictor.race.lap1_chaos.back_field", 0.28
            ),
            "strategy_variance_base": config_loader.get(
                "baseline_predictor.race.strategy_variance_base", 0.30
            ),
            "strategy_track_modifier": config_loader.get(
                "baseline_predictor.race.strategy_track_modifier", 0.5
            ),
            "safety_car_luck_range": config_loader.get(
                "baseline_predictor.race.safety_car_luck_range", 0.25
            ),
            "pace_weight_base": config_loader.get("baseline_predictor.race.pace_weight_base", 0.40),
            "pace_weight_track_modifier": config_loader.get(
                "baseline_predictor.race.pace_weight_track_modifier", 0.10
            ),
            "teammate_variance_std": config_loader.get(
                "baseline_predictor.race.teammate_variance_std", 0.15
            ),
        }

    def predict_race(
        self,
        qualifying_grid: List[Dict],
        weather: str = "dry",
        race_name: Optional[str] = None,
        n_simulations: int = 50,
        is_sprint: bool = False,
    ) -> Dict[str, Any]:
        """Predict race result from qualifying grid using Monte Carlo simulation."""
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)

        # Load track and driver data
        track_overtaking = self._load_track_overtaking_difficulty(race_name)
        driver_info_map, teams_with_long_profile = self._prepare_driver_info(
            qualifying_grid, race_name
        )
        params = self._load_race_params()

        # Calculate base chaos
        track_chaos_modifier = 1.0 - (track_overtaking * params["track_chaos_multiplier"])
        base_chaos = (
            params["base_chaos_dry"] if weather == "dry" else params["base_chaos_wet"]
        ) * track_chaos_modifier

        # Sprint adjustments
        if is_sprint:
            base_chaos *= 0.7
            params["grid_weight_min"] += 0.10
            params["grid_weight_multiplier"] += 0.10

        params["base_chaos"] = base_chaos

        # Run simulations
        position_records = {d: [] for d in driver_info_map.keys()}

        for _ in range(n_simulations):
            sc_base_prob = (
                params["sc_base_prob_dry"] if weather == "dry" else params["sc_base_prob_wet"]
            )
            sc_prob = sc_base_prob + (track_overtaking * params["sc_track_modifier"])
            safety_car = np.random.random() < sc_prob

            race_scores = []
            for driver_code, info in driver_info_map.items():
                score, dnf_occurred = self._calculate_driver_race_score(
                    info, track_overtaking, weather, safety_car, params
                )
                race_scores.append({"driver": driver_code, "score": score, "dn": dnf_occurred})

            race_scores.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(race_scores):
                position_records[item["driver"]].append(i + 1)

        # Build finish order from mean positions (allows realistic variance)
        # Using mean instead of median preserves race movement
        finish_order = []
        for driver_code, info in driver_info_map.items():
            positions = position_records[driver_code]
            mean_pos = np.mean(positions)
            median_pos = int(np.median(positions))
            p5 = int(np.percentile(positions, 5))  # 5th percentile (optimistic)
            p95 = int(np.percentile(positions, 95))  # 95th percentile (pessimistic)

            # Confidence based on consistency
            position_std = np.std(positions)
            confidence = max(40, min(60, 60 - (position_std * 3)))

            # Podium probability = % of simulations in top 3
            podium_prob = sum(1 for p in positions if p <= 3) / len(positions) * 100

            finish_order.append(
                {
                    "driver": driver_code,
                    "team": info["team"],
                    "mean_position": mean_pos,  # Use mean for sorting
                    "median_position": median_pos,
                    "p5": p5,
                    "p95": p95,
                    "confidence": round(confidence, 1),
                    "podium_probability": round(podium_prob, 1),
                    "dnf_probability": round(info["dnf_probability"], 3),
                }
            )

        # Sort by mean position (preserves fractional differences)
        finish_order.sort(key=lambda x: x["mean_position"])

        # Assign final positions
        for i, item in enumerate(finish_order):
            item["position"] = i + 1

        return {
            "finish_order": finish_order,
            "characteristics_profile_used": "long_run",
            "teams_with_characteristics_profile": teams_with_long_profile,
        }
