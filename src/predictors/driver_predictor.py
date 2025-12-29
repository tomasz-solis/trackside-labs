"""
Driver position ranking from team performance.

Converts team performance predictions (ranks 1-10) to individual driver
positions (1-20) using driver characteristics and teammate comparisons.

"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DriverPrediction:
    """Individual driver position prediction with metadata."""
    driver: str
    team: str
    position: float
    confidence_lower: float
    confidence_upper: float
    experience_tier: str
    confidence_flag: str
    pace_used: float  # The pace metric used for ranking


class DriverRanker:
    """
    Ranks individual drivers based on team performance and driver characteristics.
    
    Takes team-level predictions (ranks 1-10) and splits them into driver positions
    (1-20) using:
    - Driver pace metrics (quali_pace or race_pace)
    - Experience tiers (rookie/developing/established/veteran)
    - Confidence intervals based on data quality
    """
    
    def __init__(self, characteristics_path: str):
        """
        Initialize ranker with driver characteristics.
        
        Args:
            characteristics_path: Path to driver_characteristics_enriched.json
        """
        self.characteristics_path = Path(characteristics_path)
        self._load_characteristics()
        
    def _load_characteristics(self):
        """Load and validate driver characteristics."""
        if not self.characteristics_path.exists():
            raise FileNotFoundError(
                f"Driver characteristics not found at {self.characteristics_path}"
            )
        
        with open(self.characteristics_path) as f:
            data = json.load(f)
        
        self.drivers = data['drivers']
        self.current_year = data.get('current_year', 2025)
        
        print(f"Loaded characteristics for {len(self.drivers)} drivers")
        
    def predict_positions(
        self,
        team_predictions: Dict[str, int],
        team_lineups: Dict[str, List[str]],
        session_type: str = 'qualifying'
    ) -> Dict:
        """
        Convert team ranks to driver positions.
        
        Args:
            team_predictions: Dict mapping team name -> predicted rank (1-10)
                Example: {'Red Bull Racing': 1, 'McLaren': 2, ...}
            team_lineups: Dict mapping team name -> list of driver abbreviations
                Example: {'Red Bull Racing': ['VER', 'LAW'], ...}
            session_type: 'qualifying' or 'race' (determines which pace metric to use)
            
        Returns:
            Dict containing:
                - predictions: List[DriverPrediction]
                - session_type: str
                - total_drivers: int
                - warnings: List[str] (missing drivers, etc.)
        """
        if session_type not in ['qualifying', 'race']:
            raise ValueError(f"session_type must be 'qualifying' or 'race', got {session_type}")
        
        predictions = []
        warnings = []
        
        # Validate inputs
        if len(team_predictions) != 10:
            warnings.append(f"Expected 10 teams, got {len(team_predictions)}")
        
        # Process each team
        for team_name, team_rank in sorted(team_predictions.items(), key=lambda x: x[1]):
            
            if team_name not in team_lineups:
                warnings.append(f"No lineup found for {team_name}")
                continue
            
            drivers = team_lineups[team_name]
            
            if len(drivers) != 2:
                warnings.append(f"{team_name} has {len(drivers)} drivers (expected 2)")
                continue
            
            # Get driver predictions for this team
            driver_preds = self._rank_team_drivers(
                team_name=team_name,
                team_rank=team_rank,
                drivers=drivers,
                session_type=session_type,
                warnings=warnings
            )
            
            predictions.extend(driver_preds)
        
        # Sort by predicted position
        predictions.sort(key=lambda x: x.position)
        
        return {
            'predictions': predictions,
            'session_type': session_type,
            'total_drivers': len(predictions),
            'warnings': warnings
        }
    
    def _rank_team_drivers(
        self,
        team_name: str,
        team_rank: int,
        drivers: List[str],
        session_type: str,
        warnings: List[str]
    ) -> List[DriverPrediction]:
        """
        Rank the two drivers within a team.
        
        Args:
            team_name: Name of the team
            team_rank: Team's predicted rank (1-10)
            drivers: List of 2 driver abbreviations
            session_type: 'qualifying' or 'race'
            warnings: List to append warnings to
            
        Returns:
            List of 2 DriverPrediction objects
        """
        d1_abbr, d2_abbr = drivers
        
        # Get driver characteristics
        d1_char = self.drivers.get(d1_abbr)
        d2_char = self.drivers.get(d2_abbr)
        
        # Calculate baseline positions from team rank
        # Team rank 1 → positions ~1-2
        # Team rank 5 → positions ~9-10
        # Team rank 10 → positions ~19-20
        baseline_pos = (team_rank - 1) * 2 + 1.5  # Midpoint between the two positions
        
        # Determine pace metric to use
        pace_metric = 'quali_pace' if session_type == 'qualifying' else 'race_pace'
        
        # Handle missing driver data
        if d1_char is None or d2_char is None:
            warnings.append(
                f"Missing characteristics for {d1_abbr if d1_char is None else d2_abbr} "
                f"- using equal split"
            )
            # Fall back to equal split
            return self._equal_split(team_name, baseline_pos, d1_abbr, d2_abbr)
        
        # Get pace ratios
        d1_pace = d1_char.get(pace_metric)
        d2_pace = d2_char.get(pace_metric)
        
        if d1_pace is None or d2_pace is None:
            warnings.append(
                f"Missing {pace_metric} data for {team_name} - using equal split"
            )
            return self._equal_split(team_name, baseline_pos, d1_abbr, d2_abbr)
        
        # Calculate relative strength
        # Lower pace ratio = faster driver
        # If d1_pace < d2_pace, d1 is faster
        if d1_pace < d2_pace:
            faster_driver = d1_abbr
            slower_driver = d2_abbr
            faster_char = d1_char
            slower_char = d2_char
            pace_diff = d2_pace - d1_pace  # How much faster is d1
        else:
            faster_driver = d2_abbr
            slower_driver = d1_abbr
            faster_char = d2_char
            slower_char = d1_char
            pace_diff = d1_pace - d2_pace  # How much faster is d2
        
        # Convert pace difference to position difference
        # Empirical scaling: 0.01 pace difference ≈ 0.5 positions
        # This is conservative - teammate gaps can be larger
        position_adjustment = pace_diff * 50.0  # 0.01 → 0.5 positions
        position_adjustment = np.clip(position_adjustment, 0, 1.0)  # Max 1 position gap
        
        # Calculate final positions
        faster_pos = baseline_pos - (position_adjustment / 2)
        slower_pos = baseline_pos + (position_adjustment / 2)
        
        # Calculate confidence intervals based on experience tier
        faster_pred = DriverPrediction(
            driver=faster_driver,
            team=team_name,
            position=faster_pos,
            confidence_lower=faster_pos - self._get_uncertainty(faster_char),
            confidence_upper=faster_pos + self._get_uncertainty(faster_char),
            experience_tier=faster_char['experience']['tier'],
            confidence_flag=faster_char['confidence'],
            pace_used=faster_char[pace_metric]
        )
        
        slower_pred = DriverPrediction(
            driver=slower_driver,
            team=team_name,
            position=slower_pos,
            confidence_lower=slower_pos - self._get_uncertainty(slower_char),
            confidence_upper=slower_pos + self._get_uncertainty(slower_char),
            experience_tier=slower_char['experience']['tier'],
            confidence_flag=slower_char['confidence'],
            pace_used=slower_char[pace_metric]
        )
        
        return [faster_pred, slower_pred]
    
    def _equal_split(
        self,
        team_name: str,
        baseline_pos: float,
        d1_abbr: str,
        d2_abbr: str
    ) -> List[DriverPrediction]:
        """
        Create equal split prediction when driver data is missing.
        
        Args:
            team_name: Team name
            baseline_pos: Baseline position (midpoint)
            d1_abbr: First driver abbreviation
            d2_abbr: Second driver abbreviation
            
        Returns:
            List of 2 DriverPrediction with equal positions
        """
        # Use large uncertainty for unknown drivers
        uncertainty = 3.0
        
        pred1 = DriverPrediction(
            driver=d1_abbr,
            team=team_name,
            position=baseline_pos - 0.5,
            confidence_lower=baseline_pos - 0.5 - uncertainty,
            confidence_upper=baseline_pos - 0.5 + uncertainty,
            experience_tier='unknown',
            confidence_flag='low',
            pace_used=1.0  # Neutral
        )
        
        pred2 = DriverPrediction(
            driver=d2_abbr,
            team=team_name,
            position=baseline_pos + 0.5,
            confidence_lower=baseline_pos + 0.5 - uncertainty,
            confidence_upper=baseline_pos + 0.5 + uncertainty,
            experience_tier='unknown',
            confidence_flag='low',
            pace_used=1.0  # Neutral
        )
        
        return [pred1, pred2]
    
    def _get_uncertainty(self, driver_char: Dict) -> float:
        """
        Calculate position uncertainty based on driver experience and confidence.
        
        Args:
            driver_char: Driver characteristics dict
            
        Returns:
            Uncertainty in positions (±)
        """
        # Base uncertainty by experience tier
        tier = driver_char['experience']['tier']
        tier_uncertainty = {
            'rookie': 5.0,        # High uncertainty for rookies
            'developing': 4.5,    
            'established': 4.0,   # Low uncertainty
            'veteran': 3.5,       # Very low uncertainty
            'unknown': 5.5        # High uncertainty for unknowns
        }
        
        base = tier_uncertainty.get(tier, 2.0)
        
        # Adjust for confidence flag
        confidence_flag = driver_char.get('confidence', 'high')
        if confidence_flag == 'gathering_info':
            base *= 1.3  # 30% more uncertainty for unusual patterns
        elif confidence_flag == 'low':
            base *= 1.5  # 50% more uncertainty for insufficient data
        
        # Adjust for variance in performance
        quali_std = driver_char.get('quali_std', 0.01)
        if quali_std > 0.02:  # High variance
            base *= 1.2
        
        return base
    
    def format_predictions(self, results: Dict, top_n: int = 20) -> str:
        """
        Format predictions as readable text.
        
        Args:
            results: Output from predict_positions()
            top_n: Number of drivers to show
            
        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"=== DRIVER POSITION PREDICTIONS ({results['session_type'].upper()}) ===\n")
        lines.append(f"Total drivers: {results['total_drivers']}\n")
        
        if results['warnings']:
            lines.append("⚠️  WARNINGS:")
            for warning in results['warnings']:
                lines.append(f"  - {warning}")
            lines.append("")
        
        lines.append("Pos  Driver  Team                    Confidence         Tier         Pace")
        lines.append("-" * 80)
        
        for i, pred in enumerate(results['predictions'][:top_n], 1):
            ci_range = f"[{pred.confidence_lower:.1f}-{pred.confidence_upper:.1f}]"
            lines.append(
                f"{i:2d}.  {pred.driver:3s}     {pred.team:20s}  "
                f"{pred.position:4.1f} {ci_range:12s}  "
                f"{pred.experience_tier:12s}  {pred.pace_used:.4f}"
            )
        
        return "\n".join(lines)


def example_usage():
    """Demonstrate driver ranker usage."""
    
    # Example team predictions (from your existing model)
    team_predictions = {
        'Red Bull Racing': 1,
        'McLaren': 2,
        'Ferrari': 3,
        'Mercedes': 4,
        'Aston Martin': 5,
        'Alpine': 6,
        'Haas F1 Team': 7,
        'RB': 8,
        'Williams': 9,
        'Kick Sauber': 10
    }
    
    # Current team lineups (2025)
    team_lineups = {
        'Red Bull Racing': ['VER', 'LAW'],
        'McLaren': ['NOR', 'PIA'],
        'Ferrari': ['LEC', 'HAM'],
        'Mercedes': ['RUS', 'ANT'],
        'Aston Martin': ['ALO', 'STR'],
        'Alpine': ['GAS', 'DOO'],
        'Haas F1 Team': ['OCO', 'BEA'],
        'RB': ['TSU', 'HAD'],
        'Williams': ['SAI', 'COL'],
        'Kick Sauber': ['HUL', 'BOR']
    }
    
    # Initialize ranker
    chars_path = '../data/processed/driver_characteristics/driver_characteristics_enriched.json'
    ranker = DriverRanker(chars_path)
    
    # Predict qualifying positions
    quali_results = ranker.predict_positions(
        team_predictions=team_predictions,
        team_lineups=team_lineups,
        session_type='qualifying'
    )
    
    print(ranker.format_predictions(quali_results))
    
    # Predict race positions
    print("\n")
    race_results = ranker.predict_positions(
        team_predictions=team_predictions,
        team_lineups=team_lineups,
        session_type='race'
    )
    
    print(ranker.format_predictions(race_results))


if __name__ == '__main__':
    example_usage()
