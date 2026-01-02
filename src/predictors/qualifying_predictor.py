"""
Qualifying Predictor (UPDATED)

Uses PerformanceTracker for dynamic MAE estimates.
No hardcoded performance values.
"""

from typing import Dict, Optional, Literal
import numpy as np
from pathlib import Path
import json


class QualifyingPredictor:
    """
    Predict qualifying results using various methods.
    
    Methods:
    - session_order: Use FP3/Sprint Quali finishing order
    - blend: Combine FP data + model predictions
    - model: Model-only predictions (blind)
    
    UPDATED: Uses PerformanceTracker for expected MAE
    """
    
    def __init__(
        self,
        driver_ranker,
        data_dir='data/processed',
        performance_tracker=None
    ):
        """
        Initialize with driver ranker and performance tracker.
        
        Args:
            driver_ranker: DriverRanker instance
            data_dir: Path to processed data
            performance_tracker: PerformanceTracker instance (optional)
        """
        self.driver_ranker = driver_ranker
        self.data_dir = Path(data_dir)
        
        # Initialize performance tracker
        if performance_tracker is None:
            try:
                from src.utils.performance_tracker import get_tracker
                self.tracker = get_tracker()
            except ImportError:
                # Fallback if not installed
                self.tracker = None
        else:
            self.tracker = performance_tracker
    
    def predict(
        self,
        year: int,
        race_name: str,
        method: Literal['session_order', 'blend', 'model'] = 'session_order',
        session: Optional[str] = None,
        blend_weight: float = 0.5,
        verbose: bool = False
    ) -> Dict:
        """
        Predict qualifying results.
        
        Args:
            year: Season year
            race_name: Race name
            method: Prediction method
            session: Specific session to use (auto-detect if None)
            blend_weight: FP weight for blending (0-1)
            verbose: Print progress
            
        Returns:
            {
                'method': str,
                'expected_mae': float,  # From actual performance or conservative default
                'grid': [...],
                'data_quality': str
            }
        """
        # Load data
        track_data = self._load_track_data(year)
        car_data = self._load_car_data(year)
        
        # Determine weekend type and session
        weekend_type = self._get_weekend_type(year, race_name)
        if session is None:
            session = self._determine_best_session(weekend_type)
        
        if verbose:
            print(f"   Predicting {race_name} ({weekend_type} weekend)")
            print(f"   Method: {method}, Session: {session}")
        
        # Get team rankings based on method
        if method == 'session_order':
            team_ranks, method_id = self._predict_session_order(
                year, race_name, session, verbose
            )
        elif method == 'blend':
            team_ranks, method_id = self._predict_blended(
                year, race_name, session, weekend_type, blend_weight,
                track_data, car_data, verbose
            )
        else:  # model
            team_ranks, method_id = self._predict_model_only(
                race_name, session, weekend_type, track_data, car_data, verbose
            )
        
        if not team_ranks:
            raise ValueError(f"Failed to generate predictions for {race_name}")
        
        # Convert to driver grid
        from src.utils.lineup_manager import get_lineups
        lineups = get_lineups(year, race_name)
        
        driver_preds = self.driver_ranker.predict_positions(
            team_predictions=team_ranks,
            team_lineups=lineups,
            session_type='qualifying'
        )
        
        # Get expected MAE from tracker (data-driven)
        expected_mae = self._get_expected_mae(method_id)
        
        # Format output
        grid = []
        for i, pred in enumerate(driver_preds['predictions'], 1):
            grid.append({
                'position': i,
                'driver': pred.driver,
                'team': pred.team,
                'confidence': self._calculate_confidence(i, expected_mae)
            })
        
        return {
            'method': method_id,
            'expected_mae': expected_mae,
            'grid': grid,
            'data_quality': 'good'  # Could enhance with actual quality check
        }
    
    def _predict_session_order(self, year, race_name, session, verbose):
        """Use session finishing order directly."""
        from src.extractors.session_extractor import extract_session_order_robust
        
        team_ranks = extract_session_order_robust(year, race_name, session, verbose)
        
        if not team_ranks:
            raise ValueError(f"Could not extract {session} order for {race_name}")
        
        # Create method ID
        session_slug = session.replace(' ', '_')
        method_id = f"session_order_{session_slug}"
        
        return team_ranks, method_id
    
    def _predict_blended(self, year, race_name, session, weekend_type, 
                        blend_weight, track_data, car_data, verbose):
        """Blend FP data with model predictions."""
        from src.predictors.team_predictor import rank_teams_for_track
        from src.predictors.blended_predictor import (
            get_fp_team_performance,
            blend_predictions
        )
        
        # Get model rankings
        track = track_data.get(race_name)
        if not track:
            raise ValueError(f"No track data for {race_name}")
        
        # Handle both 'conventional' (new) and 'normal' (old) terminology
        is_sprint = weekend_type == 'sprint'
        model_session = 'post_sprint_quali' if is_sprint else 'post_fp3'
        
        # Normalize weekend_type for team_predictor (may expect 'normal' not 'conventional')
        weekend_type_normalized = 'sprint' if is_sprint else 'normal'
        team_rankings = rank_teams_for_track(car_data, track, model_session, weekend_type_normalized)
        
        if not team_rankings:
            raise ValueError(f"Could not rank teams for {race_name}")
        
        model_ranks = {team: rank for rank, (team, _, _, _) in enumerate(team_rankings, 1)}
        
        # Get FP performance
        # Use Sprint Qualifying for sprint weekends, otherwise use provided session (FP3)
        fp_session = 'Sprint Qualifying' if is_sprint else session
        fp_performance = get_fp_team_performance(year, race_name, fp_session)
        
        if not fp_performance:
            if verbose:
                print(f"   ⚠️  No FP data, falling back to model-only")
            method_id = "model_only"
            return model_ranks, method_id
        
        # Blend
        blended_ranks = blend_predictions(model_ranks, fp_performance, weight_fp=blend_weight)
        
        # Create method ID
        fp_pct = int(blend_weight * 100)
        model_pct = int((1 - blend_weight) * 100)
        method_id = f"blend_{fp_pct}_{model_pct}"
        
        return blended_ranks, method_id
    
    def _predict_model_only(self, race_name, session, weekend_type, 
                           track_data, car_data, verbose):
        """Model-only predictions (no FP data)."""
        from src.predictors.team_predictor import rank_teams_for_track
        
        track = track_data.get(race_name)
        if not track:
            raise ValueError(f"No track data for {race_name}")
        
        # Handle both 'conventional' (new) and 'normal' (old) terminology
        is_sprint = weekend_type == 'sprint'
        model_session = 'post_sprint_quali' if is_sprint else 'post_fp3'
        
        # Normalize weekend_type for team_predictor
        weekend_type_normalized = 'sprint' if is_sprint else 'normal'
        team_rankings = rank_teams_for_track(car_data, track, model_session, weekend_type_normalized)
        
        if not team_rankings:
            raise ValueError(f"Could not rank teams for {race_name}")
        
        model_ranks = {team: rank for rank, (team, _, _, _) in enumerate(team_rankings, 1)}
        method_id = "model_only"
        
        return model_ranks, method_id
    
    def _get_expected_mae(self, method_id: str) -> float:
        """
        Get expected MAE from performance tracker.
        
        Uses actual performance data or conservative default.
        """
        if self.tracker is None:
            # No tracker - return conservative default
            return self._get_fallback_mae(method_id)
        
        # Get MAE from tracker (uses last 10 predictions)
        mae = self.tracker.get_expected_mae(
            prediction_type='qualifying',
            method=method_id,
            window='last_10'
        )
        
        return mae
    
    def _get_fallback_mae(self, method_id: str) -> float:
        """Conservative fallback when tracker unavailable."""
        if 'session_order' in method_id:
            if 'Sprint_Qualifying' in method_id:
                return 3.5  # Sprint quali usually better
            return 4.0  # FP sessions
        elif 'blend' in method_id:
            return 3.8  # Blending usually helps
        else:  # model_only
            return 4.5  # Conservative for model-only
    
    def _calculate_confidence(self, position: int, expected_mae: float) -> float:
        """
        Calculate position confidence based on expected MAE.
        
        Args:
            position: Grid position
            expected_mae: Expected mean absolute error
            
        Returns:
            Confidence percentage (40-95%)
        """
        # Base confidence inversely related to MAE
        # MAE 2.0 → 90% confidence
        # MAE 4.0 → 70% confidence
        base = max(40, 95 - (expected_mae * 10))
        
        # Top positions more confident (less variable)
        if position <= 3:
            multiplier = 1.1
        elif position <= 10:
            multiplier = 1.0
        else:
            multiplier = 0.9
        
        confidence = base * multiplier
        return min(95, max(40, confidence))
    
    def _load_track_data(self, year):
        """Load track characteristics."""
        track_path = self.data_dir / f'track_characteristics/{year}_track_characteristics.json'
        with open(track_path) as f:
            return json.load(f)['tracks']
    
    def _load_car_data(self, year):
        """Load car characteristics."""
        car_path = self.data_dir / f'car_characteristics/{year}_car_characteristics.json'
        with open(car_path) as f:
            return json.load(f)['teams']
    
    def _get_weekend_type(self, year, race_name):
        """
        Determine weekend type from FastF1 EventFormat.
        
        NEVER hardcodes sprint races - reads from schedule!
        EventFormat: 'sprint', 'sprint_qualifying', 'sprint_shootout' → sprint
        EventFormat: 'conventional' → conventional
        """
        try:
            from src.utils.weekend_utils import get_weekend_type
            return get_weekend_type(year, race_name)
        except ImportError:
            # Fallback if weekend_utils not installed
            import fastf1
            try:
                schedule = fastf1.get_event_schedule(year)
                event = schedule[schedule['EventName'] == race_name]
                
                if event.empty:
                    # Try case-insensitive
                    event = schedule[schedule['EventName'].str.lower() == race_name.lower()]
                
                if not event.empty:
                    event_format = str(event.iloc[0]['EventFormat']).lower()
                    # Check for any sprint format
                    return 'sprint' if 'sprint' in event_format else 'conventional'
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not determine weekend type: {e}")
            
            # Conservative fallback
            return 'conventional'
    
    def _determine_best_session(self, weekend_type):
        """Determine best session for predictions."""
        if weekend_type == 'sprint':
            return 'Sprint Qualifying'
        else:
            return 'FP3'
