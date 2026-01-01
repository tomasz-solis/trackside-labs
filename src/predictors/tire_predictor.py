"""
Tire Predictor with Configurable Parameters

Priority:
1. FP2 long-run pace (PRIMARY)
2. Pirelli track stress (CONTEXT)
3. Driver skill (MODIFIER)

UPDATED: All multipliers are now configurable parameters
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class TirePredictor:
    """
    Predict tire degradation with data hierarchy.
    
    UPDATED: Parameters are configurable (loaded from config or set manually)
    """
    
    def __init__(
        self,
        driver_chars_path: str,
        track_chars_path: str = None,
        skill_reduction_factor: float = 0.2,
        tire_racecraft_penalty: float = 0.4,
        track_effect_range: float = 0.1
    ):
        """
        Initialize tire predictor.
        
        Args:
            driver_chars_path: Path to driver_characteristics.json
            track_chars_path: Path to track_characteristics.json
            skill_reduction_factor: Max degradation reduction from driver skill (0-1)
            tire_racecraft_penalty: Max racecraft loss from tire deg (0-1)
            track_effect_range: Track-specific effect range (±)
        """
        # Load driver characteristics
        with open(driver_chars_path) as f:
            data = json.load(f)
            self.drivers = data['drivers']
        
        # Load track characteristics
        self.tracks = {}
        if track_chars_path and Path(track_chars_path).exists():
            with open(track_chars_path) as f:
                track_data = json.load(f)
                self.tracks = track_data.get('tracks', {})
        
        # Configurable parameters (can be updated from PerformanceTracker)
        self.skill_reduction_factor = skill_reduction_factor
        self.tire_racecraft_penalty = tire_racecraft_penalty
        self.track_effect_range = track_effect_range
    
    def get_tire_impact(
        self,
        driver: str,
        team: str,
        track_name: str,
        fp2_data: Optional[Dict] = None,
        race_progress: float = 0.7
    ) -> Dict[str, float]:
        """
        Calculate tire degradation impact with data hierarchy.
        
        Args:
            driver: Driver abbreviation
            team: Team name
            track_name: Track identifier
            fp2_data: Optional FP2 data {team: {'degradation': float}}
            race_progress: Race completion (0-1)
            
        Returns:
            {
                'degradation': float,
                'source': str,
                'driver_adjustment': float,
                'track_effect': float,
                'base_deg': float,
                'progress_factor': float
            }
        """
        # Step 1: Base degradation (FP2 > Pirelli)
        if fp2_data and team in fp2_data:
            base_deg = fp2_data[team].get('degradation', 0.0)
            source = 'fp2'
        else:
            base_deg = self._get_pirelli_baseline(track_name)
            source = 'pirelli_baseline'
        
        # Step 2: Driver skill modifier
        driver_skill = self._get_driver_tire_skill(driver)
        # Uses configurable skill_reduction_factor
        driver_adjustment = 1.0 - (driver_skill * self.skill_reduction_factor)
        
        # Step 3: Track-specific effects
        track_effect = self._get_track_driver_effect(track_name, driver)
        
        # Step 4: Race progress (deg increases non-linearly)
        progress_factor = race_progress ** 1.5
        
        # Combine all factors
        final_deg = base_deg * driver_adjustment * (1.0 + track_effect) * progress_factor
        
        return {
            'degradation': float(np.clip(final_deg, 0, 1.0)),
            'source': source,
            'driver_adjustment': float(driver_adjustment),
            'track_effect': float(track_effect),
            'base_deg': float(base_deg),
            'progress_factor': float(progress_factor)
        }
    
    def adjust_racecraft_for_tires(
        self,
        base_skill: float,
        driver: str,
        team: str,
        track_name: str,
        fp2_data: Optional[Dict] = None,
        race_progress: float = 0.7
    ) -> float:
        """
        Adjust racecraft skill based on tire degradation.
        
        Uses configurable tire_racecraft_penalty parameter.
        
        Args:
            base_skill: Base racecraft skill (0-1)
            driver: Driver abbreviation
            team: Team name
            track_name: Track name
            fp2_data: Optional FP2 data
            race_progress: When in race
            
        Returns:
            Adjusted skill (0-1)
        """
        tire_impact = self.get_tire_impact(
            driver, team, track_name, fp2_data, race_progress
        )
        
        # Uses configurable tire_racecraft_penalty
        tire_factor = 1.0 - (tire_impact['degradation'] * self.tire_racecraft_penalty)
        adjusted_skill = base_skill * tire_factor
        
        return float(np.clip(adjusted_skill, 0.1, 1.0))
    
    def update_parameters(
        self,
        skill_reduction_factor: Optional[float] = None,
        tire_racecraft_penalty: Optional[float] = None,
        track_effect_range: Optional[float] = None
    ):
        """
        Update configurable parameters.
        
        Useful for tuning based on actual performance.
        
        Args:
            skill_reduction_factor: New skill reduction factor
            tire_racecraft_penalty: New racecraft penalty
            track_effect_range: New track effect range
        """
        if skill_reduction_factor is not None:
            self.skill_reduction_factor = skill_reduction_factor
        
        if tire_racecraft_penalty is not None:
            self.tire_racecraft_penalty = tire_racecraft_penalty
        
        if track_effect_range is not None:
            self.track_effect_range = track_effect_range
    
    def _get_driver_tire_skill(self, driver: str) -> float:
        """Get driver's tire management skill."""
        if driver not in self.drivers:
            return 0.65  # Neutral default
        
        tire_mgmt = self.drivers[driver].get('tire_management', {})
        return tire_mgmt.get('skill', 0.65)
    
    def _get_pirelli_baseline(self, track_name: str) -> float:
        """Get Pirelli baseline degradation for track."""
        if not track_name or track_name not in self.tracks:
            return 0.5
        
        track = self.tracks[track_name]
        pirelli = track.get('pirelli_data', {})
        
        # Direct baseline if available
        if 'baseline_deg_rate' in pirelli:
            return pirelli['baseline_deg_rate']
        
        # Calculate from tyre stress
        stress = pirelli.get('tyre_stress', {})
        if stress:
            avg_stress = np.mean([
                stress.get('traction', 3),
                stress.get('braking', 3),
                stress.get('lateral', 3),
                stress.get('asphalt_abrasion', 3)
            ])
            # Normalize 1-5 scale to 0-1
            return (avg_stress - 1) / 4
        
        return 0.5
    
    def _get_track_driver_effect(self, track_name: str, driver: str) -> float:
        """
        Get track-specific effects on this driver.
        
        Uses configurable track_effect_range parameter.
        
        Returns:
            Track effect multiplier (±track_effect_range)
        """
        if not track_name or track_name not in self.tracks:
            return 0.0
        
        track = self.tracks[track_name]
        pirelli = track.get('pirelli_data', {})
        stress = pirelli.get('tyre_stress', {})
        
        if not stress:
            return 0.0
        
        # High abrasion tracks punish aggressive drivers
        abrasion = stress.get('asphalt_abrasion', 3)
        low_grip = 5 - stress.get('asphalt_grip', 3)
        
        # Get driver's tire management
        driver_skill = self._get_driver_tire_skill(driver)
        
        # Poor tire managers suffer more on demanding tracks
        if driver_skill < 0.7:  # Below average
            severity = (abrasion + low_grip) / 10  # 0-1 scale
            # Uses configurable track_effect_range
            return severity * self.track_effect_range
        elif driver_skill > 0.85:  # Elite
            severity = (abrasion + low_grip) / 10
            # Elite drivers get half the benefit range
            return -severity * (self.track_effect_range / 2)
        
        return 0.0


def example_usage():
    """Demonstrate tire predictor with configurable parameters."""
    
    # Initialize with custom parameters
    predictor = TirePredictor(
        driver_chars_path='data/processed/driver_characteristics/driver_characteristics.json',
        track_chars_path='data/processed/track_characteristics/2025_track_characteristics.json',
        skill_reduction_factor=0.25,  # Custom: Up to 25% reduction
        tire_racecraft_penalty=0.35,  # Custom: Up to 35% racecraft loss
        track_effect_range=0.12       # Custom: ±12% track effect
    )
    
    print("TIRE PREDICTOR - CONFIGURABLE PARAMETERS")
    print("="*70)
    print(f"Skill reduction factor: {predictor.skill_reduction_factor}")
    print(f"Tire racecraft penalty: {predictor.tire_racecraft_penalty}")
    print(f"Track effect range: ±{predictor.track_effect_range}")
    
    # Test with FP2 data
    fp2_data = {
        'Ferrari': {'degradation': 0.65},
        'Mercedes': {'degradation': 0.45}
    }
    
    lec_impact = predictor.get_tire_impact(
        driver='LEC',
        team='Ferrari',
        track_name='singapore_grand_prix',
        fp2_data=fp2_data,
        race_progress=0.7
    )
    
    print(f"\nLEC @ Singapore (70% race):")
    print(f"  Source: {lec_impact['source']}")
    print(f"  Base deg (FP2): {lec_impact['base_deg']:.3f}")
    print(f"  Driver adjustment: {lec_impact['driver_adjustment']:.3f}")
    print(f"  Track effect: {lec_impact['track_effect']:+.3f}")
    print(f"  Final deg: {lec_impact['degradation']:.3f}")
    
    # Update parameters based on "learning"
    print("\n" + "="*70)
    print("UPDATING PARAMETERS (simulating learning)")
    print("="*70)
    
    predictor.update_parameters(
        skill_reduction_factor=0.18,  # Learned: Less impact than expected
        tire_racecraft_penalty=0.42   # Learned: More impact than expected
    )
    
    print(f"New skill reduction factor: {predictor.skill_reduction_factor}")
    print(f"New tire racecraft penalty: {predictor.tire_racecraft_penalty}")
    
    # Recalculate with new parameters
    lec_impact_updated = predictor.get_tire_impact(
        driver='LEC',
        team='Ferrari',
        track_name='singapore_grand_prix',
        fp2_data=fp2_data,
        race_progress=0.7
    )
    
    print(f"\nLEC @ Singapore (with updated parameters):")
    print(f"  Final deg: {lec_impact_updated['degradation']:.3f}")
    print(f"  Change: {lec_impact_updated['degradation'] - lec_impact['degradation']:+.3f}")


if __name__ == '__main__':
    example_usage()
