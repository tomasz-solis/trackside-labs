"""
Race Predictor with Physics-Based Simulation.

Simulates race outcomes using a multi-factor model that accounts for:
- Starting grid positions and lap 1 chaos (higher variance in midfield)
- Raw car pace from practice sessions
- Tire degradation as continuous pace penalty (not discrete pit stops)
- Track-specific overtaking difficulty
- Weather effects on driver consistency
- DNF probability from reliability and driver errors
- Safety car compression of field gaps

The model combines:
1. Physical simulation (lap times, tire wear, pit stops)
2. Statistical variance (randomness in race incidents)
3. Driver skill modifiers (racecraft, consistency, tire management)

Results are probabilistic - run multiple simulations for uncertainty estimates.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils.validation_helpers import (
    validate_range,
    validate_positive_int,
    validate_enum,
    validate_position,
)

logger = logging.getLogger(__name__)


class RacePredictor:
    """
    Predict race finish positions using physics-based simulation.

    Attributes:
        year: Season year for predictions
        driver_chars: Driver characteristics dict (racecraft, consistency, tire mgmt)
        tire_predictor: Sub-model for tire degradation calculation
        track_data: Track-specific parameters (pit loss, overtaking difficulty)
        weights: Component importance weights
        uncertainty: Base prediction uncertainty

    Example:
        >>> predictor = RacePredictor(year=2026, driver_chars=chars)
        >>> result = predictor.predict(
        ...     year=2026,
        ...     race_name='Bahrain GP',
        ...     qualifying_grid=grid,
        ...     weather_forecast='dry'
        ... )
        >>> print(result['finish_order'][0])  # Winner
    """

    def __init__(
        self,
        year: int,
        data_dir="data",
        driver_chars: dict | None = None,
        driver_chars_path: str | Path | None = None,
        performance_tracker=None,
    ):
        """
        Initialize race predictor.

        Args:
            year: Season year
            data_dir: Path to data directory
            driver_chars: Pre-loaded driver characteristics dict
            driver_chars_path: Path to driver_characteristics.json
            performance_tracker: Optional tracker for config overrides

        Raises:
            ValueError: If neither driver_chars nor driver_chars_path provided
        """
        self.year = year
        self.data_dir = Path(data_dir)

        # Resolve paths
        if not self.data_dir.is_absolute():
            self.data_dir = Path(__file__).parent.parent.parent / data_dir

        self.driver_chars_path = (
            Path(driver_chars_path).resolve() if driver_chars_path is not None else None
        )

        # Initialize Tracker
        if performance_tracker is None:
            from src.utils.performance_tracker import get_tracker

            self.tracker = get_tracker()
        else:
            self.tracker = performance_tracker

        # Load Configs
        self.weights = self._load_weights()
        self.uncertainty = self._load_uncertainty()

        # Load Knowledge Bases
        if driver_chars is not None:
            self.driver_chars = driver_chars
        else:
            if self.driver_chars_path is None:
                raise ValueError("RacePredictor requires driver_chars or driver_chars_path")
            with self.driver_chars_path.open() as f:
                data = json.load(f)
            self.driver_chars = data.get("drivers", {})

        # Initialize Sub-Models
        self.tire_predictor = self._init_tire_predictor()
        self.track_data = self._load_track_data()

    def _init_tire_predictor(self) -> Optional[object]:
        """Initialize tire predictor with correct year and paths."""
        try:
            from src.predictors.tire import TirePredictor

            if self.driver_chars_path is None:
                return None

            predictor = TirePredictor(
                year=self.year,
                driver_chars_path=str(self.driver_chars_path),
                data_dir=str(self.data_dir),
            )
            # Apply overrides from Learning System if available
            if self.tracker:
                conf = self.tracker.get_config("tire")
                if conf:
                    predictor.skill_reduction_factor = conf.get("skill_reduction_factor", 0.2)
            return predictor
        except (ImportError, ValueError, TypeError, AttributeError) as e:
            logger.error(
                f"Failed to load tire predictor: {e}. Race predictions will not use tire degradation model."
            )
            return None

    def _load_track_data(self) -> Dict:
        """Load track characteristics (Pit loss, SC probability)."""
        try:
            p = self.data_dir / "processed/track_characteristics.json"
            if p.exists():
                with open(p) as f:
                    return json.load(f).get("tracks", {})
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logger.warning(
                f"Could not load track characteristics from {p}: {e}. Predictions will use default track parameters."
            )
        return {}

    def predict(
        self,
        year: int,
        race_name: str,
        qualifying_grid: List[Dict],
        fp2_pace: Optional[Dict] = None,
        overtaking_factor: Optional[float] = None,
        weather_forecast: Optional[str] = "dry",
        verbose: bool = False,
    ) -> Dict:
        """
        Run race simulation and predict finishing order.

        Args:
            year: Season year
            race_name: Full race name (e.g., 'Bahrain Grand Prix')
            qualifying_grid: List of dicts with 'driver', 'team', 'position'
            fp2_pace: Optional dict of team pace deltas from FP2
            overtaking_factor: Track difficulty (0=easy, 1=impossible). Auto-loaded if None.
            weather_forecast: 'dry', 'rain', or 'mixed'
            verbose: Print debug info

        Returns:
            Dict with:
                - finish_order: List of drivers sorted by predicted position
                    Each driver dict contains:
                        * driver: 3-letter code
                        * team: Team name
                        * position: Predicted finish position (1-20)
                        * confidence: Prediction confidence (0-100)
                        * podium_probability: Chance of top-3 finish (0-100)
                        * dnf_probability: Chance of not finishing (0.0-1.0)
                - metadata: Weather and track info

        Example:
            >>> result = predictor.predict(2026, 'Monaco GP', grid, weather_forecast='rain')
            >>> winner = result['finish_order'][0]
            >>> print(f"{winner['driver']} wins (confidence: {winner['confidence']:.0f}%)")
        """
        if verbose:
            logger.debug(f"Simulating {race_name} [Weather: {weather_forecast}]")

        # 1. Setup Environment
        track_info = self.track_data.get(race_name, {})
        if overtaking_factor is None:
            overtaking_factor = track_info.get("overtaking_difficulty", 0.5)

        if fp2_pace is None:
            fp2_pace = {}

        race_positions = []

        # 2. Driver-by-Driver Simulation
        for driver_quali in qualifying_grid:
            driver = driver_quali["driver"]
            team = driver_quali["team"]
            quali_pos = driver_quali["position"]

            # Retrieve Stats
            skills = self._get_driver_skills(driver)

            # --- PHASE 1: THE START ---
            # Lap 1 is high variance. Good starters gain, poor starters lose.
            pos_after_lap1 = self._simulate_lap_1_chaos(
                quali_pos, skills["racecraft"], skills["consistency"]
            )

            # --- PHASE 2: RACE PACE ---
            # Calculate raw pace advantage/deficit relative to field
            pace_delta = self._calculate_pace_delta(team, fp2_pace)

            # --- PHASE 3: TIRE PHYSICS ---
            # Calculate Degradation Profile
            deg_profile = self._calculate_degradation_profile(driver, team, race_name, fp2_pace)

            # UPDATED: Calculate Pace Penalty (Cumulative Time Loss)
            # Replaces the old specific "Pit Stop Count" logic
            tire_penalty = self._calculate_tire_pace_penalty(deg_profile, track_info)

            # --- PHASE 4: OVERTAKING ---
            # Can they actually use their pace?
            effective_pace_gain = self._calculate_effective_pace_gain(
                pos_after_lap1, pace_delta, overtaking_factor, skills["racecraft"]
            )

            # --- PHASE 5: EXTERNALITIES ---
            # Weather
            weather_impact = self._calculate_weather_impact(weather_forecast, skills["wet_weather"])

            # Safety Car Bunching (Reduces gaps, helps recovery drives)
            sc_impact = self._apply_safety_car_variance(track_info, pos_after_lap1)

            # --- AGGREGATION ---
            # Calculate Expected Finishing Position
            expected_pos = (
                pos_after_lap1 * 1.0  # Anchor
                + effective_pace_gain  # Speed delta
                + tire_penalty  # Tire wear impact
                + weather_impact
                + sc_impact
            )

            # --- PHASE 6: RELIABILITY ---
            dnf_prob = self._calculate_dnf_probability(
                team, skills["consistency"], weather_forecast, track_info
            )

            # Apply DNF Penalty to Expected Value (simulating statistical risk)
            if np.random.random() < dnf_prob:
                expected_pos += 22  # Push to back

            # --- CONFIDENCE INTERVALS ---
            uncertainty = self._calculate_uncertainty(
                expected_pos, weather_forecast, overtaking_factor
            )

            race_positions.append(
                {
                    "driver": driver,
                    "team": team,
                    "expected_position": expected_pos,
                    "start_pos": quali_pos,
                    "dnf_probability": dnf_prob,
                    "confidence_interval": (
                        max(1, expected_pos - uncertainty),
                        expected_pos + uncertainty,
                    ),
                    "podium_probability": self._calculate_podium_probability(
                        expected_pos, uncertainty
                    ),
                }
            )

        # 3. Final Ranking
        race_positions.sort(key=lambda x: x["expected_position"])

        finish_order = []
        for i, pred in enumerate(race_positions, 1):
            pred["position"] = i
            # Cap confidence based on spread
            spread = pred["confidence_interval"][1] - pred["confidence_interval"][0]
            pred["confidence"] = max(30, 98 - spread * 4)
            finish_order.append(pred)

        return {
            "finish_order": finish_order,
            "metadata": {
                "weather": weather_forecast,
                "track_sc_prob": track_info.get("safety_car_prob", 0.0),
            },
        }

    # =========================================================================
    # DETAILED SIMULATION HELPERS
    # =========================================================================

    def _simulate_lap_1_chaos(self, start_pos: int, racecraft: float, consistency: float) -> float:
        """
        Simulate the first lap variance.

        Veterans (high racecraft/consistency) hold/gain positions.

        Args:
            start_pos: Starting grid position (1-20)
            racecraft: Driver racecraft skill (0.0-1.0)
            consistency: Driver consistency (0.0-1.0)

        Returns:
            Position after lap 1 as float (may be fractional for averaging)

        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate inputs
        validate_position(start_pos, "start_pos")
        validate_range(racecraft, "racecraft", 0.0, 1.0)
        validate_range(consistency, "consistency", 0.0, 1.0)

        if start_pos <= 2:
            return float(start_pos)  # Front row usually holds

        # Variance decreases as you go back? No, midfield is chaotic (P8-P14)
        variance = 0.5
        if 8 <= start_pos <= 15:
            variance = 1.5

        # Skill modifier: High racecraft reduces negative variance
        skill_mod = (racecraft - 0.5) * 2.0  # -1.0 to +1.0

        # Random fluctuation based on skill
        change = np.random.normal(-skill_mod, variance)
        return max(1.0, start_pos + change)

    def _calculate_effective_pace_gain(
        self, current_pos: int, pace_delta: float, difficulty: float, skill: float
    ) -> float:
        """
        Calculate positions gained/lost purely on pace, constrained by track difficulty.

        Args:
            current_pos: Current race position (1-20)
            pace_delta: Pace advantage/disadvantage relative to field (can be any value)
            difficulty: Track overtaking difficulty (0.0=easy, 1.0=impossible)
            skill: Driver racecraft skill (0.0-1.0)

        Returns:
            Position change (can be negative for gain, positive for loss)

        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate inputs
        validate_position(current_pos, "current_pos")
        validate_range(difficulty, "difficulty", 0.0, 1.0)
        validate_range(skill, "skill", 0.0, 1.0)

        # Pace Delta: Negative = Faster. -1.0 means ~0.8s faster per lap.
        theoretical_gain = pace_delta * 3.0

        # Constrain by Overtaking Difficulty
        overtaking_efficiency = 1.0 - difficulty

        # Driver Skill helps overcome difficulty
        overtaking_efficiency += skill * 0.3

        return theoretical_gain * min(1.0, overtaking_efficiency)

    def _calculate_degradation_profile(
        self, driver: str, team: str, race_name: str, fp2_pace: Optional[Dict] = None
    ) -> float:
        """
        Get the tire wear factor for this specific driver/car combo.

        Args:
            driver: 3-letter driver code
            team: Team name
            race_name: Full race name
            fp2_pace: Optional FP2 pace data

        Returns:
            Degradation factor (0.0=low to 1.0=high)
        """
        if not self.tire_predictor:
            return 0.5

        impact = self.tire_predictor.get_tire_impact(driver, team, race_name, fp2_data=fp2_pace)
        return impact["degradation"]  # 0.0 (Low) to 1.0 (High)

    def _calculate_tire_pace_penalty(self, deg_factor: float, track_info: Dict) -> float:
        """
        Calculate cumulative race time penalty due to tire degradation.

        High Deg = Slower stints OR extra pit stop time. We convert this time loss
        into 'Position Loss' (approx).

        Args:
            deg_factor: Degradation factor (0.0=low, 1.0=high)
            track_info: Track characteristics dict with pit_stop_loss

        Returns:
            Position penalty (positive = positions lost)
        """
        # Base penalty derived from track pit loss (time cost of stopping)
        # Higher pit loss tracks punish high deg more severely.
        pit_time_loss = track_info.get("pit_stop_loss", 22.0)

        # Determine the penalty scale
        # 0.5 is average/neutral deg.
        # deg_factor 1.0 (High) -> Penalty
        # deg_factor 0.0 (Low) -> Advantage

        deg_delta = deg_factor - 0.5

        # If deg is high, you lose ~20s over race distance vs average
        # 20s is roughly equivalent to 1 pit stop time.
        # We scale this by the track's specific pit penalty.
        time_penalty_factor = pit_time_loss / 22.0

        # Convert to position impact
        # 4.0 multiplier: High deg (1.0) costs ~2.0 positions vs Average (0.5)
        position_impact = deg_delta * 4.0 * time_penalty_factor

        return position_impact

    def _apply_safety_car_variance(self, track_info: Dict, current_pos: int) -> float:
        """
        Safety Cars compress the field.

        Args:
            track_info: Track characteristics dict with safety_car_prob
            current_pos: Current race position (1-20)

        Returns:
            Position variance from safety car effects
        """
        prob_sc = track_info.get("safety_car_prob", 0.3)

        # If SC is likely, gap advantages are erased.
        impact = 0.0
        if prob_sc > 0.6:
            # Compress positions towards the mean
            dist_from_mean = current_pos - 10
            impact = -dist_from_mean * 0.1  # 10% compression

        return impact

    def _calculate_weather_impact(self, forecast: str, wet_skill: float) -> float:
        """
        Rain acts as a skill multiplier.

        Args:
            forecast: Weather forecast ('dry', 'rain', 'mixed')
            wet_skill: Driver wet weather skill (0.0-1.0)

        Returns:
            Position variance from weather effects
        """
        if forecast == "dry":
            return 0.0

        # Rain Intensity
        intensity = 1.0 if forecast == "rain" else 0.5

        # Skill Delta (0.5 is avg). Range -0.5 to +0.5
        skill_delta = wet_skill - 0.5

        # Good drivers gain 3 positions, Bad drivers lose 3
        return -(skill_delta * 6.0 * intensity)

    def _calculate_dnf_probability(
        self, team: str, consistency: float, weather: str, track_info: Dict
    ) -> float:
        """
        Calculate DNF risk based on Car, Driver, Track, and Weather.

        Args:
            team: Team name
            consistency: Driver consistency (0.0-1.0)
            weather: Weather condition ('dry', 'rain', 'mixed')
            track_info: Track characteristics dict

        Returns:
            DNF probability (0.0-1.0)

        Raises:
            ValueError: If weather is not a valid value
        """
        validate_range(consistency, "consistency", 0.0, 1.0)
        validate_enum(weather, "weather", ["dry", "rain", "mixed"])

        base = 0.05  # 5% baseline reliability failure

        # Driver Error
        driver_risk = (1.0 - consistency) * 0.15

        # Track Factor (Street circuits = higher crash risk)
        track_risk = 0.0
        if track_info.get("type") == "street":
            track_risk = 0.05

        # Weather Factor
        weather_risk = 0.0
        if weather != "dry":
            weather_risk = 0.10

        return base + driver_risk + track_risk + weather_risk

    def _calculate_pace_delta(self, team: str, fp2_pace: Optional[Dict]) -> float:
        """
        Calculate pace delta from FP2 data.

        Args:
            team: Team name
            fp2_pace: Optional dict with FP2 pace data by team

        Returns:
            Pace delta relative to field
        """
        if not fp2_pace or team not in fp2_pace:
            return 0.0
        return -fp2_pace[team].get("relative_pace", 0.0) * 8.0

    def _calculate_uncertainty(self, pos: float, weather: str, overtaking: float) -> float:
        """
        Calculate prediction uncertainty based on conditions.

        Args:
            pos: Expected finishing position
            weather: Weather condition ('dry', 'rain', 'mixed')
            overtaking: Track overtaking difficulty (0.0=easy, 1.0=hard)

        Returns:
            Uncertainty in positions
        """
        # Base uncertainty
        u = self.uncertainty["base"]
        # Rain increases variance
        if weather != "dry":
            u *= 1.5
        # Easy overtaking reduces variance (faster cars sort themselves out)
        if overtaking < 0.3:
            u *= 0.8
        return u

    def _calculate_podium_probability(self, pos: float, uncertainty: float) -> float:
        """
        Calculate probability of podium (top-3) finish.

        Args:
            pos: Expected finishing position
            uncertainty: Uncertainty in positions

        Returns:
            Podium probability (0-100)
        """
        if pos - uncertainty <= 3:
            return max(0, min(100, (3 - (pos - uncertainty)) * 25))
        return 0.0

    def _load_weights(self) -> Dict[str, float]:
        """
        Load race prediction weights from tracker or use defaults.

        Returns:
            Dict with component weights
        """
        if self.tracker:
            return self.tracker.get_config("race_weights")
        return {
            "pace_weight": 0.4,
            "grid_weight": 0.3,
            "overtaking_weight": 0.15,
            "tire_deg_weight": 0.15,
        }

    def _load_uncertainty(self) -> Dict[str, float]:
        """
        Load uncertainty parameters from tracker or use defaults.

        Returns:
            Dict with uncertainty settings
        """
        if self.tracker:
            return self.tracker.get_config("uncertainty")
        return {"base": 2.5}

    def _get_driver_skills(self, driver: str) -> Dict[str, float]:
        """
        Get driver skill attributes from driver characteristics.

        Args:
            driver: 3-letter driver code

        Returns:
            Dict with racecraft, consistency, and wet_weather skills (0.0-1.0)
        """
        default = {"racecraft": 0.5, "consistency": 0.5, "wet_weather": 0.5}
        if driver not in self.driver_chars:
            return default
        d = self.driver_chars[driver]
        return {
            "racecraft": d.get("racecraft", {}).get("skill_score", 0.5),
            "consistency": d.get("consistency", {}).get("score", 0.5),
            "wet_weather": 1.0 - d.get("consistency", {}).get("error_rate_wet", 0.5),
        }
