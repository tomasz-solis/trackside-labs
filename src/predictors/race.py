"""
Race Predictor with Deep Physics & Strategy Simulation.
Handles: 
- Grid & Lap 1 Chaos
- Pure Race Pace (Fuel Corrected)
- Overtaking Probability (DRS/Track Diff)
- Pit Strategy Decision Tree (1 vs 2 stop)
- Reliability & Safety Car Variance
- Dynamic Weather Physics
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class RacePredictor:
    """
    Predict race finish positions using a comprehensive physics & strategy model.
    """
    
    def __init__(
        self,
        year: int,
        data_dir='data',
        driver_chars: dict | None = None,
        driver_chars_path: str | Path | None = None,
        performance_tracker=None
    ):
        self.year = year
        self.data_dir = Path(data_dir)
        
        # Resolve paths
        if not self.data_dir.is_absolute():
            self.data_dir = Path(__file__).parent.parent.parent / data_dir

        self.driver_chars_path = (
            Path(driver_chars_path).resolve()
            if driver_chars_path is not None
            else None
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

    def _init_tire_predictor(self):
        """Initialize tire predictor with correct year and paths."""
        try:
            from src.predictors.tire import TirePredictor
            if self.driver_chars_path is None: return None
            
            predictor = TirePredictor(
                year=self.year,
                driver_chars_path=str(self.driver_chars_path),
                data_dir=str(self.data_dir)
            )
            # Apply overrides from Learning System if available
            if self.tracker:
                conf = self.tracker.get_config('tire')
                if conf:
                    predictor.skill_reduction_factor = conf.get('skill_reduction_factor', 0.2)
            return predictor
        except Exception as e:
            print(f"⚠️ Could not load tire predictor: {e}")
            return None

    def _load_track_data(self) -> Dict:
        """Load track characteristics (Pit loss, SC probability)."""
        try:
            p = self.data_dir / 'processed/track_characteristics.json'
            if p.exists():
                with open(p) as f:
                    return json.load(f).get('tracks', {})
        except:
            pass
        return {}

    def predict(
        self,
        year: int, 
        race_name: str,
        qualifying_grid: List[Dict],
        fp2_pace: Optional[Dict] = None,
        overtaking_factor: Optional[float] = None,
        weather_forecast: Optional[str] = 'dry', 
        verbose: bool = False
    ) -> Dict:
        """
        Run the full simulation loop.
        """
        if verbose:
            print(f"🏎️  Simulating {race_name} [Weather: {weather_forecast}]")
        
        # 1. Setup Environment
        track_info = self.track_data.get(race_name, {})
        if overtaking_factor is None:
            overtaking_factor = track_info.get('overtaking_difficulty', 0.5)
            
        if fp2_pace is None: fp2_pace = {} 
        
        race_positions = []
        
        # 2. Driver-by-Driver Simulation
        for driver_quali in qualifying_grid:
            driver = driver_quali['driver']
            team = driver_quali['team']
            quali_pos = driver_quali['position']
            
            # Retrieve Stats
            skills = self._get_driver_skills(driver)
            
            # --- PHASE 1: THE START ---
            # Lap 1 is high variance. Good starters gain, poor starters lose.
            pos_after_lap1 = self._simulate_lap_1_chaos(
                quali_pos, skills['racecraft'], skills['consistency']
            )
            
            # --- PHASE 2: RACE PACE ---
            # Calculate raw pace advantage/deficit relative to field
            pace_delta = self._calculate_pace_delta(team, fp2_pace)
            
            # --- PHASE 3: TIRE & STRATEGY ---
            # Calculate Degradation Profile
            deg_profile = self._calculate_degradation_profile(
                driver, team, race_name, fp2_pace
            )
            
            # Determine Pit Strategy (1-stop vs 2-stop)
            # This returns the TOTAL TIME LOST in pits
            strategy_loss = self._determine_pit_strategy_loss(
                race_name, deg_profile, track_info
            )
            
            # --- PHASE 4: OVERTAKING ---
            # Can they actually use their pace?
            # If track is Monaco (overtaking_factor > 0.8), pace advantage is nullified.
            effective_pace_gain = self._calculate_effective_pace_gain(
                pos_after_lap1, pace_delta, overtaking_factor, skills['racecraft']
            )
            
            # --- PHASE 5: EXTERNALITIES ---
            # Weather
            weather_impact = self._calculate_weather_impact(
                weather_forecast, skills['wet_weather']
            )
            
            # Safety Car Bunching (Reduces gaps, helps recovery drives)
            sc_impact = self._apply_safety_car_variance(
                track_info, pos_after_lap1
            )

            # --- AGGREGATION ---
            # Calculate Expected Finishing Position
            # We start from Lap 1 position, then apply modifiers
            expected_pos = (
                pos_after_lap1 * 1.0 +         # Anchor
                effective_pace_gain +          # Speed delta
                (strategy_loss / 10.0) +       # ~10s strategy diff = 1 position?
                weather_impact +
                sc_impact
            )
            
            # --- PHASE 6: RELIABILITY ---
            dnf_prob = self._calculate_dnf_probability(
                team, skills['consistency'], weather_forecast, track_info
            )
            
            # Apply DNF Penalty to Expected Value (simulating statistical risk)
            if np.random.random() < dnf_prob:
                expected_pos += 22  # Push to back
                
            # --- CONFIDENCE INTERVALS ---
            uncertainty = self._calculate_uncertainty(
                expected_pos, weather_forecast, overtaking_factor
            )
            
            race_positions.append({
                'driver': driver,
                'team': team,
                'expected_position': expected_pos,
                'start_pos': quali_pos,
                'dnf_probability': dnf_prob,
                'confidence_interval': (max(1, expected_pos - uncertainty), expected_pos + uncertainty),
                'podium_probability': self._calculate_podium_probability(expected_pos, uncertainty)
            })
        
        # 3. Final Ranking
        race_positions.sort(key=lambda x: x['expected_position'])
        
        finish_order = []
        for i, pred in enumerate(race_positions, 1):
            pred['position'] = i
            # Cap confidence based on spread
            spread = pred['confidence_interval'][1] - pred['confidence_interval'][0]
            pred['confidence'] = max(30, 98 - spread * 4)
            finish_order.append(pred)
            
        return {
            'finish_order': finish_order,
            'metadata': {
                'weather': weather_forecast,
                'track_sc_prob': track_info.get('safety_car_prob', 0.0)
            }
        }

    # =========================================================================
    # DETAILED SIMULATION HELPERS
    # =========================================================================

    def _simulate_lap_1_chaos(self, start_pos, racecraft, consistency):
        """
        Simulate the first lap variance.
        Veterans (high racecraft/consistency) hold/gain positions.
        Rookies or aggressive drivers have higher variance (gain or crash).
        """
        if start_pos <= 2: return start_pos # Front row usually holds
        
        # Variance decreases as you go back? No, midfield is chaotic (P8-P14)
        variance = 0.5
        if 8 <= start_pos <= 15: variance = 1.5
        
        # Skill modifier: High racecraft reduces negative variance
        skill_mod = (racecraft - 0.5) * 2.0 # -1.0 to +1.0
        
        # Random fluctuation based on skill
        # Good driver: tends to gain (-1) or hold (0)
        change = np.random.normal(-skill_mod, variance)
        
        return max(1, start_pos + change)

    def _calculate_effective_pace_gain(self, current_pos, pace_delta, difficulty, skill):
        """
        Calculate positions gained/lost purely on pace, constrained by track difficulty.
        """
        # Pace Delta: Negative = Faster. -1.0 means ~0.8s faster per lap.
        
        # Theoretical positions gained over race distance
        # ~55 laps * 0.1s advantage ~= 5.5s ~= 1-2 positions?
        theoretical_gain = pace_delta * 3.0 
        
        # Constrain by Overtaking Difficulty
        # If difficulty is 1.0 (Monaco), gain is 10% of theoretical.
        # If difficulty is 0.0 (Spa), gain is 100%.
        overtaking_efficiency = (1.0 - difficulty)
        
        # Driver Skill helps overcome difficulty
        # Max (0.9 skill) can pass at Monaco better than Latifi.
        overtaking_efficiency += (skill * 0.3)
        
        return theoretical_gain * min(1.0, overtaking_efficiency)

    def _calculate_degradation_profile(self, driver, team, race_name, fp2_pace):
        """Get the tire wear factor for this specific driver/car combo."""
        if not self.tire_predictor: return 0.5
        
        impact = self.tire_predictor.get_tire_impact(
            driver, team, race_name, fp2_pace=fp2_pace
        )
        return impact['degradation'] # 0.0 (Low) to 1.0 (High)

    def _determine_pit_strategy_loss(self, race_name, deg_factor, track_info):
        """
        Decide between 1-stop and 2-stop and calculate total pit time loss.
        """
        # Get Time Loss per Pit Stop (default 22s)
        pit_time_loss = track_info.get('pit_stop_loss', 22.0)
        
        # Logic:
        # Low Deg (< 0.4) -> Easy 1-stop
        # Med Deg (0.4-0.7) -> Marginal 1-stop / Fast 2-stop
        # High Deg (> 0.7) -> Forced 2-stop or 3-stop
        
        if deg_factor < 0.4:
            # 1 Stop
            stops = 1
        elif deg_factor > 0.75:
            # 2 Stops + potential fall off
            stops = 2
        else:
            # Mixed strategy. 
            # If overtaking is hard, prioritize track position (1 stop)
            # If overtaking is easy, prioritize fresh tires (2 stop)
            if track_info.get('overtaking_difficulty', 0.5) > 0.6:
                stops = 1 # Hold position
            else:
                stops = 2 # Attack
        
        total_loss = stops * pit_time_loss
        
        # Convert Time Loss to Position Loss (approx)
        # In a spread out field, 20s might be 1 position. In a train, it's 5.
        # We normalize relative to the 'Standard' strategy (say 1.5 stops avg)
        avg_stops = 1.5
        stop_delta = stops - avg_stops
        
        return stop_delta * 2.0 # Each extra stop costs ~2 net positions if not recovered

    def _apply_safety_car_variance(self, track_info, current_pos):
        """
        Safety Cars compress the field.
        """
        prob_sc = track_info.get('safety_car_prob', 0.3)
        
        # If SC is likely, gap advantages are erased.
        # This helps cars behind catch up, hurts leaders.
        impact = 0.0
        if prob_sc > 0.6:
            # High SC probability (e.g. Jeddah, Singapore)
            # Compress positions towards the mean
            # Leaders (Pos 1) get penalty (+), Backmarkers (Pos 20) get boost (-)
            dist_from_mean = current_pos - 10
            impact = -dist_from_mean * 0.1 # 10% compression
            
        return impact

    def _calculate_weather_impact(self, forecast, wet_skill):
        """Rain acts as a skill multiplier."""
        if forecast == 'dry': return 0.0
        
        # Rain Intensity
        intensity = 1.0 if forecast == 'rain' else 0.5
        
        # Skill Delta (0.5 is avg). Range -0.5 to +0.5
        skill_delta = wet_skill - 0.5
        
        # Good drivers gain 3 positions, Bad drivers lose 3
        return -(skill_delta * 6.0 * intensity)

    def _calculate_dnf_probability(self, team, consistency, weather, track_info):
        """
        Calculate DNF risk based on Car, Driver, Track, and Weather.
        """
        base = 0.05 # 5% baseline reliability failure
        
        # Driver Error
        driver_risk = (1.0 - consistency) * 0.15
        
        # Track Factor (Street circuits = higher crash risk)
        track_risk = 0.0
        if track_info.get('type') == 'street':
            track_risk = 0.05
            
        # Weather Factor
        weather_risk = 0.0
        if weather != 'dry':
            weather_risk = 0.10
            
        return base + driver_risk + track_risk + weather_risk

    def _calculate_pace_delta(self, team, fp2_pace):
        if not fp2_pace or team not in fp2_pace: return 0.0
        return -fp2_pace[team].get('relative_pace', 0.0) * 8.0 

    def _calculate_uncertainty(self, pos, weather, overtaking):
        # Base uncertainty
        u = self.uncertainty['base']
        # Rain increases variance
        if weather != 'dry': u *= 1.5
        # Easy overtaking reduces variance (faster cars sort themselves out)
        if overtaking < 0.3: u *= 0.8
        return u

    def _calculate_podium_probability(self, pos, uncertainty):
        if pos - uncertainty <= 3:
            return max(0, min(100, (3 - (pos - uncertainty)) * 25))
        return 0.0

    def _load_weights(self) -> Dict:
        if self.tracker: return self.tracker.get_config('race_weights')
        return {'pace_weight': 0.4, 'grid_weight': 0.3, 'overtaking_weight': 0.15, 'tire_deg_weight': 0.15}

    def _load_uncertainty(self) -> Dict:
        if self.tracker: return self.tracker.get_config('uncertainty')
        return {'base': 2.5}

    def _get_driver_skills(self, driver: str) -> Dict:
        default = {'racecraft': 0.5, 'consistency': 0.5, 'wet_weather': 0.5}
        if driver not in self.driver_chars: return default
        d = self.driver_chars[driver]
        return {
            'racecraft': d.get('racecraft', {}).get('skill_score', 0.5),
            'consistency': d.get('consistency', {}).get('score', 0.5),
            'wet_weather': 1.0 - d.get('consistency', {}).get('error_rate_wet', 0.5)
        }