"""
Driver position ranking from team performance - FIXED

Fixed pace access to work with nested structure:
  driver['pace']['quali_pace'] instead of driver['quali_pace']
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
    pace_used: float


class DriverRanker:
    """Ranks drivers based on team performance and driver characteristics."""
    
    def __init__(self, characteristics_path: str):
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
        self.current_year = data.get('year', 2025)
        
        print(f"Loaded characteristics for {len(self.drivers)} drivers")
        
    def predict_positions(
        self,
        team_predictions: Dict[str, int],
        team_lineups: Dict[str, List[str]],
        session_type: str = 'qualifying'
    ) -> Dict:
        """Convert team ranks to driver positions."""
        
        if session_type not in ['qualifying', 'race']:
            raise ValueError(f"session_type must be 'qualifying' or 'race', got {session_type}")
        
        predictions = []
        warnings = []
        
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
        """Rank the two drivers within a team."""
        
        d1_abbr, d2_abbr = drivers
        
        # Get driver characteristics
        d1_char = self.drivers.get(d1_abbr)
        d2_char = self.drivers.get(d2_abbr)
        
        # Calculate baseline positions from team rank
        baseline_pos = (team_rank - 1) * 2 + 1.5
        
        # Determine pace metric to use
        pace_metric = 'quali_pace' if session_type == 'qualifying' else 'race_pace'
        
        # Handle missing driver data
        if d1_char is None or d2_char is None:
            warnings.append(
                f"Missing characteristics for {d1_abbr if d1_char is None else d2_abbr} "
                f"- using equal split"
            )
            return self._equal_split(team_name, baseline_pos, d1_abbr, d2_abbr)
        
        # FIXED: Access pace from nested structure
        d1_pace = d1_char.get('pace', {}).get(pace_metric)
        d2_pace = d2_char.get('pace', {}).get(pace_metric)
        
        if d1_pace is None or d2_pace is None:
            warnings.append(
                f"Missing {pace_metric} data for {team_name} - using equal split"
            )
            return self._equal_split(team_name, baseline_pos, d1_abbr, d2_abbr)
        
        # Calculate relative strength
        if d1_pace < d2_pace:
            faster_driver = d1_abbr
            slower_driver = d2_abbr
            faster_char = d1_char
            slower_char = d2_char
            pace_diff = d2_pace - d1_pace
        else:
            faster_driver = d2_abbr
            slower_driver = d1_abbr
            faster_char = d2_char
            slower_char = d1_char
            pace_diff = d1_pace - d2_pace
        
        # Convert pace difference to position difference
        position_adjustment = pace_diff * 50.0
        position_adjustment = np.clip(position_adjustment, 0, 1.0)
        
        # Calculate final positions
        faster_pos = baseline_pos - (position_adjustment / 2)
        slower_pos = baseline_pos + (position_adjustment / 2)
        
        # Calculate confidence intervals
        faster_pred = DriverPrediction(
            driver=faster_driver,
            team=team_name,
            position=faster_pos,
            confidence_lower=faster_pos - self._get_uncertainty(faster_char),
            confidence_upper=faster_pos + self._get_uncertainty(faster_char),
            experience_tier=faster_char.get('experience', {}).get('tier', 'unknown'),
            confidence_flag=faster_char.get('pace', {}).get('confidence', 'medium'),
            pace_used=faster_char['pace'][pace_metric]
        )
        
        slower_pred = DriverPrediction(
            driver=slower_driver,
            team=team_name,
            position=slower_pos,
            confidence_lower=slower_pos - self._get_uncertainty(slower_char),
            confidence_upper=slower_pos + self._get_uncertainty(slower_char),
            experience_tier=slower_char.get('experience', {}).get('tier', 'unknown'),
            confidence_flag=slower_char.get('pace', {}).get('confidence', 'medium'),
            pace_used=slower_char['pace'][pace_metric]
        )
        
        return [faster_pred, slower_pred]
    
    def _equal_split(
        self,
        team_name: str,
        baseline_pos: float,
        d1_abbr: str,
        d2_abbr: str
    ) -> List[DriverPrediction]:
        """Create equal split prediction when driver data is missing."""
        
        uncertainty = 3.0
        
        pred1 = DriverPrediction(
            driver=d1_abbr,
            team=team_name,
            position=baseline_pos - 0.5,
            confidence_lower=baseline_pos - 0.5 - uncertainty,
            confidence_upper=baseline_pos - 0.5 + uncertainty,
            experience_tier='unknown',
            confidence_flag='low',
            pace_used=1.0
        )
        
        pred2 = DriverPrediction(
            driver=d2_abbr,
            team=team_name,
            position=baseline_pos + 0.5,
            confidence_lower=baseline_pos + 0.5 - uncertainty,
            confidence_upper=baseline_pos + 0.5 + uncertainty,
            experience_tier='unknown',
            confidence_flag='low',
            pace_used=1.0
        )
        
        return [pred1, pred2]
    
    def _get_uncertainty(self, driver_char: Dict) -> float:
        """Calculate position uncertainty based on driver experience."""
        
        tier = driver_char.get('experience', {}).get('tier', 'unknown')
        tier_uncertainty = {
            'rookie': 5.0,
            'developing': 4.5,    
            'established': 4.0,
            'veteran': 3.5,
            'unknown': 5.5
        }
        
        base = tier_uncertainty.get(tier, 4.0)
        
        # Adjust for confidence flag
        confidence_flag = driver_char.get('pace', {}).get('confidence', 'high')
        if confidence_flag == 'low':
            base *= 1.5
        elif confidence_flag == 'medium':
            base *= 1.2
        
        return base
    
    def format_predictions(self, results: Dict, top_n: int = 20) -> str:
        """Format predictions as readable text."""
        
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
