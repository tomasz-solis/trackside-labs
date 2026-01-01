"""
Race Predictor with Performance Tracking

Uses PerformanceTracker for:
- Dynamic MAE estimates (no hardcoded values)
- Learned model weights
- Tunable parameters

All configuration loaded from tracker, updates automatically with new data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class RacePredictor:
    """
    Predict race finish positions with data-driven performance tracking.
    
    Model factors:
    1. Starting grid (quali)
    2. Long run pace (FP2)
    3. Track overtaking difficulty
    4. Driver overtaking skill
    5. Tire degradation
    
    UPDATED: All weights and MAEs from PerformanceTracker
    """
    
    def __init__(
        self,
        data_dir='data',
        driver_chars: dict | None = None,
        driver_chars_path: str | Path | None = None,
        performance_tracker=None
    ):
        """
        Initialize race predictor.
        
        Args:
            data_dir: Path to data directory
            driver_chars: Driver characteristics dict
            driver_chars_path: Path to driver characteristics file
            performance_tracker: PerformanceTracker instance (optional)
        """
        # Handle paths
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_absolute():
            current = Path(__file__).parent
            while current != current.parent:
                if (current / 'src').exists():
                    self.data_dir = current / data_dir
                    break
                current = current.parent
            else:
                self.data_dir = Path(data_dir)
        
        self.driver_chars_path = (
            Path(driver_chars_path).resolve()
            if driver_chars_path is not None
            else None
        )
        
        # Initialize performance tracker
        if performance_tracker is None:
            try:
                from src.utils.performance_tracker import get_tracker
                self.tracker = get_tracker()
            except ImportError:
                self.tracker = None
        else:
            self.tracker = performance_tracker
        
        # Load weights from tracker config (or use defaults)
        self.weights = self._load_weights()
        self.uncertainty = self._load_uncertainty()
        
        # Load driver characteristics
        if driver_chars is not None:
            self.driver_chars = driver_chars
        else:
            if self.driver_chars_path is None:
                raise ValueError(
                    "RacePredictor requires driver_chars or driver_chars_path"
                )
            path = self.driver_chars_path
            if not path.exists():
                raise FileNotFoundError(f"Driver characteristics not found at {path}")
            with path.open() as f:
                data = json.load(f)
            self.driver_chars = data.get("drivers", {})
        
        # Initialize tire predictor
        self.tire_predictor = self._init_tire_predictor()
    
    def _load_weights(self) -> Dict:
        """Load model weights from tracker config."""
        if self.tracker is None:
            # Fallback defaults
            return {
                'pace_weight': 0.4,
                'grid_weight': 0.3,
                'overtaking_weight': 0.2,
                'tire_deg_weight': 0.1
            }
        
        config = self.tracker.get_config('race_weights')
        if config:
            return {
                'pace_weight': config.get('pace_weight', 0.4),
                'grid_weight': config.get('grid_weight', 0.3),
                'overtaking_weight': config.get('overtaking_weight', 0.2),
                'tire_deg_weight': config.get('tire_deg_weight', 0.1)
            }
        
        return {
            'pace_weight': 0.4,
            'grid_weight': 0.3,
            'overtaking_weight': 0.2,
            'tire_deg_weight': 0.1
        }
    
    def _load_uncertainty(self) -> Dict:
        """Load uncertainty parameters from tracker config."""
        if self.tracker is None:
            return {
                'base': 2.5,
                'reliability': 0.15,
                'new_regs': True,
                'new_regs_multiplier': 1.3
            }
        
        config = self.tracker.get_config('uncertainty')
        if config:
            return {
                'base': config.get('base', 2.5),
                'reliability': config.get('reliability', 0.15),
                'new_regs': True,  # Will be updated manually for regulation changes
                'new_regs_multiplier': config.get('new_regs_multiplier', 1.3)
            }
        
        return {
            'base': 2.5,
            'reliability': 0.15,
            'new_regs': True,
            'new_regs_multiplier': 1.3
        }
    
    def _init_tire_predictor(self):
        """Initialize tire predictor."""
        try:
            try:
                from src.predictors.tire_predictor import TirePredictor
            except ImportError:
                from tire_predictor import TirePredictor
            
            if self.driver_chars_path is None:
                return None
            
            # Load tire parameters from config
            tire_config = {}
            if self.tracker:
                tire_config = self.tracker.get_config('tire')
            
            char_path = self.driver_chars_path
            track_path = self.data_dir / 'processed/track_characteristics/2025_track_characteristics.json'
            
            predictor = TirePredictor(
                driver_chars_path=str(char_path),
                track_chars_path=str(track_path) if track_path.exists() else None
            )
            
            # Update predictor parameters if available
            if tire_config:
                predictor.skill_reduction_factor = tire_config.get('skill_reduction_factor', 0.2)
                predictor.tire_racecraft_penalty = tire_config.get('tire_racecraft_penalty', 0.4)
                predictor.track_effect_range = tire_config.get('track_effect_range', 0.1)
            
            return predictor
        
        except Exception as e:
            print(f"Could not load tire predictor: {e}")
            return None
    
    def predict(
        self,
        year: int,
        race_name: str,
        qualifying_grid: List[Dict],
        fp2_pace: Optional[Dict] = None,
        overtaking_factor: Optional[float] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Predict race finish positions.
        
        Args:
            year: Season year
            race_name: Race name
            qualifying_grid: Qualifying results
            fp2_pace: FP2 pace data (optional)
            overtaking_factor: Track difficulty (optional)
            verbose: Print progress
            
        Returns:
            {
                'method': str,
                'expected_mae': float,  # From tracker or conservative default
                'finish_order': [...]
            }
        """
        if verbose:
            print(f"🏎️  Predicting race finish: {race_name}")
        
        # Load overtaking baseline
        if overtaking_factor is None:
            overtaking_factor = self._get_overtaking_difficulty(race_name)
        
        # Load FP2 pace if needed
        if fp2_pace is None:
            try:
                from src.extractors.race_pace_extractor import extract_fp2_pace
                fp2_pace = extract_fp2_pace(year, race_name, verbose)
            except Exception as e:
                if verbose:
                    print(f"   No FP2 pace: {e}")
                fp2_pace = {}
        
        # Predict each driver
        race_positions = []
        
        for driver_quali in qualifying_grid:
            driver = driver_quali['driver']
            team = driver_quali['team']
            quali_pos = driver_quali['position']
            
            # Get driver racecraft
            driver_racecraft = self._get_driver_racecraft(driver)
            
            # Calculate components
            pace_delta = self._calculate_pace_delta(team, fp2_pace, quali_pos)
            overtaking_gain = self._calculate_overtaking_gain(
                quali_pos, pace_delta, overtaking_factor, driver_racecraft
            )
            deg_impact = self._calculate_degradation_impact(
                driver, team, race_name, fp2_pace
            )
            
            # Combine with learned weights
            expected_pos = (
                quali_pos * self.weights['grid_weight'] +
                (quali_pos + pace_delta) * self.weights['pace_weight'] +
                (quali_pos + overtaking_gain) * self.weights['overtaking_weight'] +
                (quali_pos + deg_impact) * self.weights['tire_deg_weight']
            )
            
            # Apply reliability risk
            if np.random.random() < self.uncertainty['reliability']:
                expected_pos += 5
            
            # Calculate uncertainty
            uncertainty = self._calculate_position_uncertainty(expected_pos)
            ci_low = max(1, expected_pos - uncertainty)
            ci_high = min(len(qualifying_grid), expected_pos + uncertainty)
            
            race_positions.append({
                'driver': driver,
                'team': team,
                'expected_position': expected_pos,
                'confidence_interval': (ci_low, ci_high),
                'driver_racecraft': driver_racecraft,
                'pace_delta': pace_delta,
                'overtaking_gain': overtaking_gain,
                'deg_impact': deg_impact
            })
        
        # Sort by expected position
        race_positions.sort(key=lambda x: x['expected_position'])
        
        # Assign final positions
        finish_order = []
        for i, pred in enumerate(race_positions, 1):
            confidence = self._calculate_confidence(
                pred['confidence_interval'], i
            )
            
            podium_prob = self._calculate_podium_probability(
                pred['expected_position'],
                pred['confidence_interval']
            )
            
            finish_order.append({
                'position': i,
                'driver': pred['driver'],
                'team': pred['team'],
                'confidence': confidence,
                'confidence_interval': pred['confidence_interval'],
                'podium_probability': podium_prob,
                'driver_racecraft': pred['driver_racecraft'],
                'tire_deg_impact': pred['deg_impact']
            })
        
        # Get expected MAE from tracker (data-driven)
        expected_mae = self._get_expected_mae()
        
        return {
            'method': 'racecraft_with_tire_hierarchy',
            'expected_mae': expected_mae,
            'finish_order': finish_order,
            'factors_used': {
                'pace': fp2_pace is not None,
                'track_difficulty': overtaking_factor,
                'driver_racecraft': True,
                'tire_degradation': self.tire_predictor is not None
            },
            'weights_used': self.weights
        }
    
    def _get_expected_mae(self) -> float:
        """Get expected MAE from tracker or conservative default."""
        if self.tracker is None:
            # Conservative default for new regulations
            return 5.5 if self.uncertainty['new_regs'] else 4.8
        
        # Get from tracker (uses last 10 predictions)
        mae = self.tracker.get_expected_mae(
            prediction_type='race',
            method='racecraft_with_tire_hierarchy',
            window='last_10'
        )
        
        return mae
    
    def _get_driver_racecraft(self, driver: str) -> float:
        """Get driver racecraft from characteristics."""
        if driver not in self.driver_chars:
            return 0.5
        
        racecraft = self.driver_chars[driver].get('racecraft', {})
        return racecraft.get('overtaking_skill', 0.5)
    
    def _calculate_pace_delta(self, team, fp2_pace, quali_pos):
        """Calculate pace advantage from FP2."""
        if not fp2_pace or team not in fp2_pace:
            return 0.0
        
        team_pace = fp2_pace[team].get('relative_pace', 0.0)
        pace_delta = -team_pace * 10
        
        return np.clip(pace_delta, -5, 5)
    
    def _calculate_overtaking_gain(
        self, 
        quali_pos, 
        pace_delta, 
        overtaking_factor,
        driver_racecraft
    ):
        """Calculate overtaking gain with driver racecraft."""
        if pace_delta >= 0:
            return 0.0
        
        potential = abs(pace_delta)
        track_adjusted = potential * overtaking_factor
        achievable = track_adjusted * driver_racecraft
        
        if quali_pos <= 5:
            achievable *= 0.5
        elif quali_pos <= 10:
            achievable *= 0.75
        
        return -achievable
    
    def _calculate_degradation_impact(
        self, 
        driver: str, 
        team: str, 
        race_name: str,
        fp2_pace: Optional[Dict]
    ):
        """Calculate tire degradation impact."""
        if not self.tire_predictor:
            # Fallback
            if not fp2_pace or team not in fp2_pace:
                return 0.0
            deg_rate = fp2_pace[team].get('degradation', 0.0)
            deg_impact = deg_rate * 4
            return np.clip(deg_impact, 0, 3)
        
        # Use tire predictor
        tire_impact = self.tire_predictor.get_tire_impact(
            driver=driver,
            team=team,
            track_name=race_name.lower().replace(' ', '_'),
            fp2_data=fp2_pace,
            race_progress=0.7
        )
        
        deg_impact = tire_impact['degradation'] * 4
        return np.clip(deg_impact, 0, 3)
    
    def _calculate_position_uncertainty(self, position):
        """Calculate uncertainty for position."""
        base = self.uncertainty['base']
        
        if 6 <= position <= 15:
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        uncertainty = base * multiplier
        
        if self.uncertainty['new_regs']:
            uncertainty *= self.uncertainty['new_regs_multiplier']
        
        return uncertainty
    
    def _calculate_confidence(self, ci, position):
        """Calculate confidence from interval."""
        ci_low, ci_high = ci
        spread = ci_high - ci_low
        confidence = max(40, 95 - spread * 5)
        return confidence
    
    def _calculate_podium_probability(self, expected_pos, ci):
        """Calculate podium probability."""
        ci_low, ci_high = ci
        
        if ci_low <= 3:
            overlap = min(3, ci_high) - max(1, ci_low) + 1
            total_range = ci_high - ci_low + 1
            prob = (overlap / total_range) * 100
            
            if expected_pos <= 3:
                prob = min(95, prob * 1.5)
            
            return prob
        
        return 0.0
    
    def _get_overtaking_difficulty(self, race_name):
        """Get track overtaking difficulty."""
        baseline_path = self.data_dir / 'historical/overtaking_difficulty.json'
        
        if not baseline_path.exists():
            return 0.5
        
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        race_slug = race_name.lower().replace(' ', '_')
        
        if race_slug in baseline.get('tracks', {}):
            return baseline['tracks'][race_slug].get('difficulty', 0.5)
        
        return 0.5
    
    def update_weights(self, new_weights):
        """
        Update model weights and save to tracker.
        
        Args:
            new_weights: Dict of weight updates
        """
        self.weights.update(new_weights)
        
        # Save to tracker config
        if self.tracker:
            self.tracker.update_config({
                'race_weights': self.weights
            })
    
    def set_uncertainty(self, new_regs=True):
        """Set uncertainty level for regulation changes."""
        self.uncertainty['new_regs'] = new_regs
        
        # Save to tracker config
        if self.tracker:
            self.tracker.update_config({
                'uncertainty': self.uncertainty
            })
