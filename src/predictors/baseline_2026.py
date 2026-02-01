"""
2026 Baseline Predictor

Used when no 2026 race data exists yet. Provides low-confidence predictions
based on team strength + driver skill, acknowledging regulation reset uncertainty.
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

logger = logging.getLogger(__name__)


class Baseline2026Predictor:
    """
    Simple predictor for 2026 season before real data is available.

    Uses:
    - 2026 team strength from car_characteristics
    - Driver skill from driver_characteristics
    - Much lower confidence than Bayesian model (40-60%)
    - Actual DNF risk per driver/team
    """

    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize baseline 2026 predictor.

        Args:
            data_dir: Path to processed data directory

        Raises:
            FileNotFoundError: If required data files not found
        """
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
        """
        Load 2026 team data and driver characteristics with schema validation.

        Raises:
            FileNotFoundError: If required JSON files not found
            ValueError: If JSON files have invalid structure/schema
            KeyError: If required keys missing from JSON files
        """
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
            last_updated = data.get("last_updated")

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
                    "Consider re-running extraction: python scripts/extract_driver_characteristics_fixed.py --years 2023,2024,2025"
                )

            self.drivers = data["drivers"]

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

        Args:
            year: Season year (2020-2030)
            race_name: Full race name (e.g., 'Bahrain Grand Prix')
            n_simulations: Number of Monte Carlo simulations (>= 10)

        Returns:
            Dict with 'grid' key containing list of predicted qualifying results

        Raises:
            ValueError: If year or n_simulations are outside valid ranges
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

        # Build driver list
        all_drivers = []
        for team, drivers in lineups.items():
            team_strength = self.teams.get(team, {}).get("overall_performance", 0.50)

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
        noise_std_normal = config_loader.get(
            "baseline_predictor.qualifying.noise_std_normal", 0.02
        )
        noise_std = noise_std_sprint if is_sprint else noise_std_normal

        # Load score composition weights from config
        team_weight = config_loader.get(
            "baseline_predictor.qualifying.team_weight", 0.7
        )
        skill_weight = config_loader.get(
            "baseline_predictor.qualifying.skill_weight", 0.3
        )

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

        # Calculate median positions
        grid = []
        for driver_info in all_drivers:
            positions = position_records[driver_info["driver"]]
            median_pos = int(np.median(positions))

            # Confidence based on consistency
            position_std = np.std(positions)
            # Lower std = higher confidence
            confidence = max(40, min(60, 60 - (position_std * 5)))

            grid.append(
                {
                    "driver": driver_info["driver"],
                    "team": driver_info["team"],
                    "median_position": median_pos,
                    "confidence": round(confidence, 1),
                }
            )

        # Sort by median position
        grid.sort(key=lambda x: x["median_position"])

        # Assign final positions
        for i, item in enumerate(grid):
            item["position"] = i + 1

        return {"grid": grid}

    def predict_race(
        self,
        qualifying_grid: List[Dict],
        weather: str = "dry",
        race_name: Optional[str] = None,
        n_simulations: int = 50,
    ) -> Dict[str, Any]:
        """
        Predict race result from qualifying grid using Monte Carlo simulation.

        Systematic factors (data-driven):
        - Driver quali vs race pace differential (some drivers better in races)
        - Driver overtaking skill (some drivers aggressive, others defensive)
        - Track overtaking difficulty (Monaco 0.9 = impossible, Bahrain 0.4 = easy)
        - Car race pace vs quali pace (tire deg, setup philosophy)

        Random factors (chaos):
        - Lap 1 incidents, strategy variance, safety car timing

        Result: Monaco ~1 avg change, Monza ~4 avg change

        Args:
            qualifying_grid: List of dicts with driver, team, position info
            weather: Weather condition ('dry', 'rain', 'mixed'), default 'dry'
            race_name: Optional full race name for track characteristics lookup
            n_simulations: Number of Monte Carlo simulations (>= 10)

        Returns:
            Dict with 'finish_order' key containing predicted race results

        Raises:
            ValueError: If weather or n_simulations are outside valid ranges
        """
        # Validate inputs
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])
        validate_positive_int(n_simulations, "n_simulations", min_val=1)
        # Load track characteristics if available
        track_overtaking = 0.5  # Default: medium difficulty
        if race_name:
            try:
                track_file = Path(
                    "data/processed/track_characteristics/2026_track_characteristics.json"
                )
                with open(track_file) as f:
                    track_data = json.load(f)
                    # Validate track characteristics before using
                    validate_track_characteristics(track_data)
                    tracks = track_data["tracks"]
                    track_overtaking = tracks.get(race_name, {}).get(
                        "overtaking_difficulty", 0.5
                    )
            except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Could not load track characteristics for {race_name}: {e}. "
                    "Using default overtaking difficulty of 0.5."
                )
                # Use default if can't load

        # Prepare driver info with systematic race pace advantages
        driver_info_map = {}
        for entry in qualifying_grid:
            driver_code = entry["driver"]
            team = entry["team"]
            grid_pos = entry["position"]

            team_strength = self.teams.get(team, {}).get("overall_performance", 0.50)
            driver_data = self.drivers.get(driver_code, {})

            # Get driver pace differential: race_pace vs quali_pace
            pace_data = driver_data.get("pace", {})
            quali_pace = pace_data.get("quali_pace", 0.5)
            race_pace = pace_data.get("race_pace", 0.5)
            # Positive = better in races, negative = better in quali
            race_advantage = race_pace - quali_pace

            # Get racecraft
            racecraft = driver_data.get("racecraft", {})
            skill = racecraft.get("skill_score", 0.5)
            overtaking_skill = racecraft.get("overtaking_skill", 0.5)

            # Get DNF risk - cap historical rates
            dnf_rate_historical_cap = config_loader.get(
                "baseline_predictor.race.dnf_rate_historical_cap", 0.20
            )
            dnf_rate = driver_data.get("dnf_risk", {}).get("dnf_rate", 0.10)
            dnf_rate = min(
                dnf_rate, dnf_rate_historical_cap
            )  # Cap at historical maximum before team adjustment

            # Experience-based DNF risk modifier (rookies crash more often)
            experience_tier = driver_data.get("experience", {}).get(
                "tier", "established"
            )
            experience_modifiers = {
                "rookie": 0.05,  # +5% crash risk for rookies
                "developing": 0.02,  # +2% for young drivers
                "established": 0.00,  # Baseline
                "veteran": -0.01,  # -1% for experienced drivers
            }
            experience_dnf_modifier = experience_modifiers.get(experience_tier, 0.0)

            # Team uncertainty penalty (new teams = higher reliability risk)
            team_uncertainty = self.teams.get(team, {}).get("uncertainty", 0.30)
            if team_uncertainty >= 0.40:
                # New teams (e.g., Cadillac)
                adjusted_dnf = (
                    dnf_rate + experience_dnf_modifier + (team_uncertainty * 0.20)
                )
            else:
                # Established teams
                adjusted_dnf = dnf_rate + experience_dnf_modifier

            # Load final DNF cap from config
            dnf_rate_final_cap = config_loader.get(
                "baseline_predictor.race.dnf_rate_final_cap", 0.35
            )

            driver_info_map[driver_code] = {
                "driver": driver_code,
                "team": team,
                "grid_pos": grid_pos,
                "team_strength": team_strength,
                "skill": skill,
                "race_advantage": race_advantage,  # Systematic race pace boost
                "overtaking_skill": overtaking_skill,
                "dnf_probability": max(0.0, min(adjusted_dnf, dnf_rate_final_cap)),
            }

        # Run multiple simulations
        position_records = {d: [] for d in driver_info_map.keys()}

        # Load ALL race parameters from config ONCE (not in loops!)
        base_chaos_dry = config_loader.get(
            "baseline_predictor.race.base_chaos.dry", 0.35
        )
        base_chaos_wet = config_loader.get(
            "baseline_predictor.race.base_chaos.wet", 0.45
        )
        track_chaos_multiplier = config_loader.get(
            "baseline_predictor.race.track_chaos_multiplier", 0.4
        )
        sc_base_prob_dry = config_loader.get(
            "baseline_predictor.race.sc_base_probability.dry", 0.45
        )
        sc_base_prob_wet = config_loader.get(
            "baseline_predictor.race.sc_base_probability.wet", 0.70
        )
        sc_track_modifier = config_loader.get(
            "baseline_predictor.race.sc_track_modifier", 0.25
        )
        grid_weight_min = config_loader.get(
            "baseline_predictor.race.grid_weight_min", 0.15
        )
        grid_weight_multiplier = config_loader.get(
            "baseline_predictor.race.grid_weight_multiplier", 0.35
        )
        race_advantage_multiplier = config_loader.get(
            "baseline_predictor.race.race_advantage_multiplier", 0.5
        )
        overtaking_skill_multiplier = config_loader.get(
            "baseline_predictor.race.overtaking_skill_multiplier", 0.25
        )
        overtaking_grid_threshold = config_loader.get(
            "baseline_predictor.race.overtaking_grid_threshold", 5
        )
        overtaking_track_threshold = config_loader.get(
            "baseline_predictor.race.overtaking_track_threshold", 0.5
        )
        lap1_front_row_chaos = config_loader.get(
            "baseline_predictor.race.lap1_chaos.front_row", 0.15
        )
        lap1_upper_midfield_chaos = config_loader.get(
            "baseline_predictor.race.lap1_chaos.upper_midfield", 0.32
        )
        lap1_midfield_chaos = config_loader.get(
            "baseline_predictor.race.lap1_chaos.midfield", 0.38
        )
        lap1_back_field_chaos = config_loader.get(
            "baseline_predictor.race.lap1_chaos.back_field", 0.28
        )
        strategy_variance_base = config_loader.get(
            "baseline_predictor.race.strategy_variance_base", 0.30
        )
        strategy_track_modifier = config_loader.get(
            "baseline_predictor.race.strategy_track_modifier", 0.5
        )
        safety_car_luck_range = config_loader.get(
            "baseline_predictor.race.safety_car_luck_range", 0.25
        )
        pace_weight_base = config_loader.get(
            "baseline_predictor.race.pace_weight_base", 0.40
        )
        pace_weight_track_modifier = config_loader.get(
            "baseline_predictor.race.pace_weight_track_modifier", 0.10
        )
        teammate_variance_std = config_loader.get(
            "baseline_predictor.race.teammate_variance_std", 0.15
        )

        # Chaos varies by track - harder to overtake = less variance
        # Monaco (0.9 difficulty) = low chaos, Monza (0.2) = high chaos
        track_chaos_modifier = 1.0 - (track_overtaking * track_chaos_multiplier)
        base_chaos = (
            base_chaos_dry if weather == "dry" else base_chaos_wet
        ) * track_chaos_modifier

        for _ in range(n_simulations):
            race_scores = []

            # Use pre-loaded config values
            sc_base_prob = sc_base_prob_dry if weather == "dry" else sc_base_prob_wet

            # Safety car probability varies by track (higher on street circuits)
            sc_base_prob = sc_base_prob_dry if weather == "dry" else sc_base_prob_wet
            sc_prob = sc_base_prob + (
                track_overtaking * sc_track_modifier
            )  # Street circuits = more SC
            safety_car = np.random.random() < sc_prob

            for driver_code, info in driver_info_map.items():
                # Grid advantage - varies by track overtaking difficulty (using pre-loaded config)
                # Monaco: grid position is KING (0.9 diff → 50% weight)
                # Monza: grid matters less (0.2 diff → 20% weight)
                grid_weight = grid_weight_min + (
                    track_overtaking * grid_weight_multiplier
                )  # 0.15-0.50 range
                grid_advantage = 1.0 - ((info["grid_pos"] - 1) / 21.0)

                # Position-dependent scaling: harder to overtake as you move up grid
                # P1-P3: 0.1x (front row - almost impossible to gain)
                # P4-P7: 0.3x (upper midfield - small gains)
                # P8-P12: 0.6x (midfield - moderate gains)
                # P13+: 1.0x (back - easy to pass slower cars)
                if info["grid_pos"] <= 3:
                    position_scaling = 0.1  # Front row: minimal gains
                elif info["grid_pos"] <= 7:
                    position_scaling = 0.3  # Upper midfield: small gains
                elif info["grid_pos"] <= 12:
                    position_scaling = 0.6  # Midfield: moderate gains
                else:
                    position_scaling = 1.0  # Back of grid: full gains

                # SYSTEMATIC: Driver race pace advantage (using pre-loaded config)
                # Scaled by position: easier to gain from back than from front
                # P15 Verstappen can climb to P5, but P8 Norris unlikely to reach P1
                race_pace_boost = (
                    info["race_advantage"]
                    * race_advantage_multiplier
                    * position_scaling
                )

                # SYSTEMATIC: Overtaking on easy tracks (using pre-loaded config)
                # Also scaled by position: only matters when starting lower
                # P20 → P12 easier than P8 → P1
                if (
                    info["grid_pos"] > overtaking_grid_threshold
                    and track_overtaking < overtaking_track_threshold
                ):
                    overtaking_boost = (
                        (info["overtaking_skill"] - 0.5)
                        * overtaking_skill_multiplier
                        * position_scaling
                    )
                else:
                    overtaking_boost = 0

                # Lap 1 chaos: varies by grid position and track (using pre-loaded config)
                if info["grid_pos"] <= 3:
                    lap1_chaos = np.random.normal(
                        0, lap1_front_row_chaos
                    )  # Front: safer
                elif info["grid_pos"] <= 10:
                    lap1_chaos = np.random.normal(
                        0, lap1_upper_midfield_chaos
                    )  # Upper mid: battles
                elif info["grid_pos"] <= 15:
                    lap1_chaos = np.random.normal(
                        0, lap1_midfield_chaos
                    )  # Midfield: chaos
                else:
                    lap1_chaos = np.random.normal(
                        0, lap1_back_field_chaos
                    )  # Back: fewer battles

                # Strategy variance: less on Monaco (follow leader), more on Bahrain (using pre-loaded config)
                # Monaco = 0.17, Bahrain = 0.24
                strategy_std = strategy_variance_base * (
                    1.0 - track_overtaking * strategy_track_modifier
                )
                strategy_factor = np.random.uniform(-strategy_std, strategy_std)

                # Safety car luck (position swing when safety car deployed, using pre-loaded config)
                if safety_car:
                    sc_luck = np.random.uniform(
                        -safety_car_luck_range, safety_car_luck_range
                    )
                else:
                    sc_luck = 0

                # Base race score (using pre-loaded config)
                # Team pace is MOST important (50-60% of performance)
                # Grid position matters (15-30% depending on track)
                # Driver skill is important but LIMITED by car (15-25%)
                pace_weight = pace_weight_base - (
                    track_overtaking * pace_weight_track_modifier
                )  # Team pace: 0.30-0.40 range

                # Driver skill: FIXED at 0.20 (20%) - can't overcome bad car
                # Great driver in slow car (OCO/Haas) can't beat average driver in fast car (NOR/McLaren)
                driver_weight = 0.20

                # Normalize so weights sum to 1.0
                total_weight = grid_weight + pace_weight + driver_weight
                normalized_grid = grid_weight / total_weight
                normalized_pace = pace_weight / total_weight
                normalized_skill = driver_weight / total_weight

                base_score = (
                    (grid_advantage * normalized_grid)
                    + (info["team_strength"] * normalized_pace)
                    + (info["skill"] * normalized_skill)
                )

                # Intra-team variance (setup/tire choices vary, using pre-loaded config)
                teammate_variance = np.random.normal(0, teammate_variance_std)

                # Check for DNF (simulate driver not finishing)
                dnf_occurred = np.random.random() < info["dnf_probability"]

                if dnf_occurred:
                    # DNF = very low score (ensures driver finishes last among DNFs)
                    score = -10.0 + np.random.uniform(-1, 0)  # Random to vary DNF order
                else:
                    # Total score: base + systematic factors + random chaos
                    score = (
                        base_score
                        + race_pace_boost
                        + overtaking_boost
                        + np.random.normal(0, base_chaos)
                        + lap1_chaos
                        + strategy_factor
                        + sc_luck
                        + teammate_variance
                    )

                race_scores.append(
                    {"driver": driver_code, "score": score, "dnf": dnf_occurred}
                )

            # Sort and record positions
            race_scores.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(race_scores):
                position_records[item["driver"]].append(i + 1)

        # Build finish order from mean positions (allows realistic variance)
        # Using mean instead of median preserves race movement
        finish_order = []
        for driver_code, info in driver_info_map.items():
            positions = position_records[driver_code]
            mean_pos = np.mean(positions)

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

        return {"finish_order": finish_order}
