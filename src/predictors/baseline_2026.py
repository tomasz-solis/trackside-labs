"""
2026 Baseline Predictor

Primary runtime predictor for 2026 qualifying and race simulations.

Combines:
- weight-scheduled team strength (baseline/testing/current season),
- optional session blending for qualifying,
- dynamic tire compound selection and performance adjustments,
- Monte Carlo race scoring with uncertainty and DNF modeling.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.systems.weight_schedule import (
    calculate_blended_performance,
    get_recommended_schedule,
)
from src.utils import config_loader
from src.utils.compound_performance import (
    get_compound_performance_modifier,
    should_use_compound_adjustments,
)
from src.utils.data_generator import ensure_baseline_exists
from src.utils.fp_blending import blend_team_strength, get_best_fp_performance
from src.utils.lap_by_lap_simulator import (
    aggregate_simulation_results,
    simulate_race_lap_by_lap,
)
from src.utils.lineups import get_lineups
from src.utils.pit_strategy import (
    generate_pit_strategy,
)
from src.utils.schema_validation import (
    validate_driver_characteristics,
    validate_team_characteristics,
    validate_track_characteristics,
)
from src.utils.track_data_loader import (
    get_available_compounds,
    get_tire_stress_score,
    load_track_specific_params,
)
from src.utils.validation_helpers import (
    validate_enum,
    validate_positive_int,
    validate_year,
)
from src.utils.weekend import is_sprint_weekend

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

    def _select_race_compound(self, race_name: str) -> str:
        """Select primary race compound based on track tire stress characteristics."""
        try:
            # Try 2026 pirelli info first
            pirelli_file_2026 = Path("data/2026_pirelli_info.json")
            pirelli_file_2025 = Path("data/2025_pirelli_info.json")

            pirelli_file = pirelli_file_2026 if pirelli_file_2026.exists() else pirelli_file_2025

            if not pirelli_file.exists():
                return "MEDIUM"  # Default fallback

            with open(pirelli_file) as f:
                pirelli_data = json.load(f)

            # Normalize race name to match keys (lowercase, underscores)
            race_key = race_name.lower().replace(" ", "_").replace("-", "_")
            track_info = pirelli_data.get(race_key, {})

            if not track_info or "tyre_stress" not in track_info:
                return "MEDIUM"

            tyre_stress = track_info["tyre_stress"]

            # Load thresholds from config
            high_threshold = config_loader.get(
                "baseline_predictor.compound_selection.high_stress_threshold", 3.5
            )
            low_threshold = config_loader.get(
                "baseline_predictor.compound_selection.low_stress_threshold", 2.5
            )
            default_stress = config_loader.get(
                "baseline_predictor.compound_selection.default_stress_fallback", 3.0
            )

            # Calculate total tire stress score (higher = more demanding)
            stress_score = (
                tyre_stress.get("traction", default_stress)
                + tyre_stress.get("braking", default_stress)
                + tyre_stress.get("lateral", default_stress)
                + tyre_stress.get("asphalt_abrasion", default_stress)
            ) / 4.0

            # Apply thresholds from config
            if stress_score > high_threshold:
                return "HARD"
            elif stress_score < low_threshold:
                return "SOFT"
            else:
                return "MEDIUM"

        except Exception as e:
            logger.debug(f"Could not determine race compound for {race_name}: {e}")
            return "MEDIUM"

    def get_compound_adjusted_team_strength(
        self, team: str, race_name: str, compound: str = "MEDIUM"
    ) -> float:
        """Get team strength (0-1) adjusted for tire compound performance."""
        # Get base blended team strength
        base_strength = self.get_blended_team_strength(team, race_name)

        # Get compound characteristics
        team_data = self.teams.get(team, {})
        compound_chars = team_data.get("compound_characteristics", {})

        # Check if we have reliable compound data
        if not should_use_compound_adjustments(compound_chars, min_laps_threshold=10):
            return base_strength

        # Calculate compound modifier
        compound_modifier = get_compound_performance_modifier(compound_chars, compound)

        # Apply modifier and clip to valid range
        adjusted_strength = float(np.clip(base_strength + compound_modifier, 0.0, 1.0))

        logger.debug(
            f"  {team} on {compound}: base={base_strength:.3f} + "
            f"compound={compound_modifier:+.3f} = {adjusted_strength:.3f}"
        )

        return adjusted_strength

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

    def _update_compound_characteristics_from_session(
        self,
        session_laps: "pd.DataFrame",
        race_name: str,
        year: int,
        is_sprint: bool,
    ) -> None:
        """
        Extract compound characteristics from session laps and update in-memory team data.
        This runs on every "Generate Prediction" click to use fresh FastF1 data.
        """
        from src.systems.compound_analyzer import (
            aggregate_compound_samples,
            extract_compound_metrics,
            normalize_compound_metrics_across_teams,
        )
        from src.utils.team_mapping import map_team_to_characteristics

        logger.info(f"Extracting compound metrics from session for {race_name}...")

        # Extract compound metrics per team
        race_compound_metrics = {}
        known_teams = set(self.teams.keys())

        for raw_team in session_laps["Team"].unique():
            if pd.isna(raw_team):
                continue

            canonical_team = map_team_to_characteristics(str(raw_team), known_teams=known_teams)
            if not canonical_team:
                continue

            team_laps = session_laps[session_laps["Team"] == raw_team]
            compound_data = extract_compound_metrics(team_laps, canonical_team, race_name)

            if compound_data:
                race_compound_metrics[canonical_team] = compound_data

        # Normalize compound metrics across teams (track-specific)
        if race_compound_metrics:
            normalized_compound_metrics = normalize_compound_metrics_across_teams(
                race_compound_metrics, race_name
            )

            # Get blend weight from config based on session type
            # Practice sessions are exploratory (lower weight), sprint/race are competitive (higher weight)
            if is_sprint:
                blend_weight = config_loader.get(
                    "baseline_predictor.compound_blend_weights.sprint", 0.50
                )
            else:
                blend_weight = config_loader.get(
                    "baseline_predictor.compound_blend_weights.practice", 0.30
                )

            # Update in-memory team data with blended compound characteristics
            for team_name, new_compounds in normalized_compound_metrics.items():
                if team_name not in self.teams:
                    continue

                existing_compound_chars = self.teams[team_name].get("compound_characteristics", {})
                if not isinstance(existing_compound_chars, dict):
                    existing_compound_chars = {}

                # Blend with existing compound data
                blended_compounds = aggregate_compound_samples(
                    existing_compound_chars,
                    new_compounds,
                    blend_weight=blend_weight,
                    race_name=race_name,
                )

                # Update in-memory (not saved to JSON - that happens in updater)
                self.teams[team_name]["compound_characteristics"] = blended_compounds

            logger.info(
                f"✓ Updated compound characteristics for {len(normalized_compound_metrics)} teams "
                f"(blend_weight={blend_weight:.0%})"
            )
        else:
            logger.debug("No compound metrics extracted from session")

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
        - Sprint: Lock before Sprint Race (Saturday) → predict from Friday Sprint Quali
        - Normal: Lock before Qualifying (Saturday) → predict qualifying outcome
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

    def _calculate_driver_race_score(
        self,
        info: dict,
        track_overtaking: float,
        weather: str,
        safety_car: bool,
        params: dict,
        rng: np.random.Generator,
    ) -> tuple[float, bool]:
        """Calculate single driver's race score for one simulation."""
        grid_weight = params["grid_weight_min"] + (
            track_overtaking * params["grid_weight_multiplier"]
        )
        grid_advantage = 1.0 - ((info["grid_pos"] - 1) / 21.0)

        if info["grid_pos"] <= 3:
            position_scaling = 0.1
        elif info["grid_pos"] <= 7:
            position_scaling = 0.3
        elif info["grid_pos"] <= 12:
            position_scaling = 0.6
        else:
            position_scaling = 1.0

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
            overtaking_boost = 0.0

        if info["grid_pos"] <= 3:
            lap1_chaos = rng.normal(0, params["lap1_front_row_chaos"])
        elif info["grid_pos"] <= 10:
            lap1_chaos = rng.normal(0, params["lap1_upper_midfield_chaos"])
        elif info["grid_pos"] <= 15:
            lap1_chaos = rng.normal(0, params["lap1_midfield_chaos"])
        else:
            lap1_chaos = rng.normal(0, params["lap1_back_field_chaos"])

        strategy_std = params["strategy_variance_base"] * (
            1.0 - track_overtaking * params["strategy_track_modifier"]
        )
        strategy_factor = rng.uniform(-strategy_std, strategy_std)

        sc_luck = (
            rng.uniform(-params["safety_car_luck_range"], params["safety_car_luck_range"])
            if safety_car
            else 0.0
        )

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

        teammate_variance = rng.normal(0, params["teammate_variance_std"])

        base_chaos_std = (
            params["base_chaos"]["wet"]
            if weather in ("rain", "mixed")
            else params["base_chaos"]["dry"]
        )

        dnf_occurred = rng.random() < info["dnf_probability"]

        if dnf_occurred:
            score = -10.0 + rng.uniform(-1.0, 0.0)
        else:
            score = (
                base_score
                + race_pace_boost
                + overtaking_boost
                + rng.normal(0, base_chaos_std)
                + lap1_chaos
                + strategy_factor
                + sc_luck
                + teammate_variance
            )

        return float(score), bool(dnf_occurred)

    def _load_race_params(self) -> dict:
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

        # Determine race distance
        race_distance = 20 if is_sprint else 60  # Simplified; could be track-specific

        # Get tire stress and available compounds
        tire_stress_score = get_tire_stress_score(race_name)
        available_compounds = get_available_compounds(race_name)

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
        base_seed = 42  # For reproducibility

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

        # Build finish order from blended position scores
        finish_order = []
        for driver_code, median_pos in aggregated["median_positions"].items():
            info = driver_info_map[driver_code]
            positions = aggregated["position_distributions"][driver_code]

            # Confidence based on consistency (keep as-is)
            position_std = np.std(positions)
            confidence = max(40, min(60, 60 - (position_std * 3)))

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

            # >>> THIS is the key: use blended samples for p5/p95 too
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
