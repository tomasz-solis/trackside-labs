"""
Blended Predictor - Combines Model Predictions with Actual FP Data

For F1 Fantasy decisions, we have access to practice session times:
- Normal weekends: FP1, FP2, FP3 available before lineup lock (before Quali)
- Sprint weekends: FP1, Sprint Quali available before lineup lock (before Sprint)

This blends model predictions (car-track fit) with actual session times (reality).

Weight: 70% actual FP times, 30% model prediction
"""

import fastf1 as ff1
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.getLogger("fastf1").setLevel(logging.CRITICAL)


def get_fp_team_performance(year, race_name, session_type='FP3'):
    """
    Extract team performance from practice session.
    
    Uses median of each team's best lap times (robust to outliers).
    
    Args:
        year: Season year
        race_name: Race name
        session_type: 'FP1', 'FP2', 'FP3', or 'Sprint Qualifying'
        
    Returns:
        Dict mapping team -> relative performance (0-1, higher = faster)
    """
    try:
        session = ff1.get_session(year, race_name, session_type)
        session.load(laps=True, telemetry=False, weather=False)
        
        if not hasattr(session, 'laps') or session.laps is None:
            return None
        
        laps = session.laps
        
        # Get best lap per driver (excluding outliers)
        best_times = []
        
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver]
            
            # Filter valid laps
            valid_laps = driver_laps[
                (driver_laps['LapTime'].notna()) &
                (driver_laps['Compound'].notna())  # On tire compound
            ]
            
            if len(valid_laps) == 0:
                continue
            
            # Get best lap time
            best_lap = valid_laps['LapTime'].min()
            team = driver_laps['Team'].iloc[0]
            
            best_times.append({
                'driver': driver,
                'team': team,
                'time': best_lap.total_seconds()
            })
        
        if not best_times:
            return None
        
        # Get median time per team (robust to one driver having issues)
        team_times = {}
        
        for entry in best_times:
            team = entry['team']
            if team not in team_times:
                team_times[team] = []
            team_times[team].append(entry['time'])
        
        team_medians = {
            team: np.median(times)
            for team, times in team_times.items()
        }
        
        # Convert to relative performance (0-1 scale)
        # Invert: Faster time = Higher score
        fastest = min(team_medians.values())
        slowest = max(team_medians.values())
        
        if fastest == slowest:
            # All teams same pace (unlikely)
            return {team: 0.5 for team in team_medians}
        
        team_performance = {
            team: 1.0 - (time - fastest) / (slowest - fastest)
            for team, time in team_medians.items()
        }
        
        return team_performance
        
    except Exception as e:
        print(f"  Error loading {session_type}: {e}")
        return None


def blend_predictions(model_ranks, fp_performance, weight_fp=0.7):
    """
    Blend model predictions with actual FP performance.
    
    Args:
        model_ranks: Dict of team -> predicted rank (1-10)
        fp_performance: Dict of team -> FP performance (0-1)
        weight_fp: Weight for FP data (0.7 = 70% FP, 30% model)
        
    Returns:
        Dict of team -> blended rank (1-10)
    """
    # Convert model ranks to scores (invert: rank 1 = score 1.0)
    num_teams = len(model_ranks)
    model_scores = {
        team: 1.0 - (rank - 1) / (num_teams - 1)
        for team, rank in model_ranks.items()
    }
    
    # Blend scores
    blended_scores = {}
    
    for team in model_ranks:
        model_score = model_scores.get(team, 0.5)
        fp_score = fp_performance.get(team, 0.5)  # Default if team not in FP
        
        # Weighted average
        blended = weight_fp * fp_score + (1 - weight_fp) * model_score
        blended_scores[team] = blended
    
    # Convert back to ranks
    sorted_teams = sorted(blended_scores.items(), key=lambda x: x[1], reverse=True)
    blended_ranks = {team: rank for rank, (team, _) in enumerate(sorted_teams, 1)}
    
    return blended_ranks


def get_best_available_session(year, race_name, weekend_type='normal'):
    """
    Get the best available practice session for blending.
    
    Normal weekend priority: FP3 > FP2 > FP1
    Sprint weekend priority: Sprint Quali > FP1
    
    Args:
        year: Season year
        race_name: Race name
        weekend_type: 'normal' or 'sprint'
        
    Returns:
        Tuple of (session_type, fp_performance) or (None, None)
    """
    if weekend_type == 'sprint':
        # Sprint weekend: Try Sprint Quali first, then FP1
        for session_type in ['Sprint Qualifying', 'Sprint Shootout', 'FP1']:
            fp_perf = get_fp_team_performance(year, race_name, session_type)
            if fp_perf:
                return session_type, fp_perf
    else:
        # Normal weekend: Try FP3, FP2, FP1
        for session_type in ['FP3', 'FP2', 'FP1']:
            fp_perf = get_fp_team_performance(year, race_name, session_type)
            if fp_perf:
                return session_type, fp_perf
    
    return None, None


def predict_with_blending(
    year,
    race_name,
    model_predictions,
    weekend_type='normal',
    weight_fp=0.7
):
    """
    Make blended prediction using model + actual FP data.
    
    Args:
        year: Season year
        race_name: Race name
        model_predictions: Dict of team -> predicted rank
        weekend_type: 'normal' or 'sprint'
        weight_fp: Weight for FP data (default 0.7 = 70% FP)
        
    Returns:
        Dict with blended predictions and metadata
    """
    # Get best available session
    session_used, fp_performance = get_best_available_session(
        year, race_name, weekend_type
    )
    
    if fp_performance is None:
        # No FP data available - use model only
        return {
            'blended_ranks': model_predictions,
            'blend_type': 'model_only',
            'session_used': None,
            'fp_weight': 0.0
        }
    
    # Blend predictions
    blended_ranks = blend_predictions(
        model_predictions,
        fp_performance,
        weight_fp=weight_fp
    )
    
    return {
        'blended_ranks': blended_ranks,
        'blend_type': 'fp_blended',
        'session_used': session_used,
        'fp_weight': weight_fp,
        'fp_performance': fp_performance,
        'model_predictions': model_predictions
    }


def format_comparison(model_ranks, blended_ranks, actual_ranks=None):
    """
    Format comparison of model vs blended vs actual ranks.
    
    Args:
        model_ranks: Model predictions
        blended_ranks: Blended predictions
        actual_ranks: Actual results (optional)
    """
    print("\nPREDICTION COMPARISON")
    print("="*70)
    
    if actual_ranks:
        print(f"{'Team':<25} {'Model':<8} {'Blended':<8} {'Actual':<8} {'Δ Model':<8} {'Δ Blend':<8}")
    else:
        print(f"{'Team':<25} {'Model':<8} {'Blended':<8} {'Change':<8}")
    
    print("-"*70)
    
    all_teams = set(model_ranks.keys()) | set(blended_ranks.keys())
    sorted_teams = sorted(all_teams, key=lambda t: blended_ranks.get(t, 99))
    
    for team in sorted_teams:
        model_rank = model_ranks.get(team, '-')
        blend_rank = blended_ranks.get(team, '-')
        
        if isinstance(model_rank, (int, float)) and isinstance(blend_rank, (int, float)):
            change = int(model_rank) - int(blend_rank)
            change_str = f"{change:+d}" if change != 0 else "="
        else:
            change_str = "-"
        
        if actual_ranks:
            actual_rank = actual_ranks.get(team, '-')
            
            if isinstance(model_rank, (int, float)) and isinstance(actual_rank, (int, float)):
                model_error = abs(int(model_rank) - int(actual_rank))
            else:
                model_error = '-'
            
            if isinstance(blend_rank, (int, float)) and isinstance(actual_rank, (int, float)):
                blend_error = abs(int(blend_rank) - int(actual_rank))
            else:
                blend_error = '-'
            
            print(f"{team:<25} {model_rank:<8} {blend_rank:<8} {actual_rank:<8} "
                  f"{model_error:<8} {blend_error:<8}")
        else:
            print(f"{team:<25} {model_rank:<8} {blend_rank:<8} {change_str:<8}")


if __name__ == '__main__':
    # Demo: Bahrain 2025
    print("BLENDED PREDICTION DEMO")
    print("="*70)
    
    # Simulate model predictions (replace with actual model)
    model_preds = {
        'Mercedes': 1,
        'Red Bull Racing': 2,
        'McLaren': 3,
        'Ferrari': 4,
        'Racing Bulls': 5,
        'Williams': 6,
        'Aston Martin': 7,
        'Alpine': 8,
        'Kick Sauber': 9,
        'Haas F1 Team': 10
    }
    
    # Get blended prediction
    result = predict_with_blending(
        2025,
        'Bahrain Grand Prix',
        model_preds,
        weekend_type='normal',
        weight_fp=0.7
    )
    
    print(f"\nSession used: {result['session_used']}")
    print(f"Blend weight: {result['fp_weight']*100:.0f}% FP data")
    print(f"Blend type: {result['blend_type']}")
    
    if result['blend_type'] == 'fp_blended':
        format_comparison(
            result['model_predictions'],
            result['blended_ranks']
        )
