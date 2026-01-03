"""
Team Ranking - Fixed for user's key names

User's track file has: slow_corner_z, medium_corner_z, etc.
NOT: slow_corner_pct_z, medium_corner_pct_z
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
sys.path.append('/home/claude')
from src.extractors.performance import extract_all_teams_performance
from src.helpers.session_selector import get_prediction_context


def calculate_suitability(team_perf: Dict, track_chars: Dict) -> float:
    """
    Calculate how well a team suits a track.
    
    FIXED: Uses actual key names from user's track file.
    
    Args:
        team_perf: ALREADY NORMALIZED performance (0-1 scores)
        track_chars: Track characteristics (z-scores)
    
    Returns:
        Suitability score (0-1)
    """
    if not team_perf:
        return 0.0
    
    score = 0.0
    weight_sum = 0.0
    
    # Corner matching (60% of score)
    # Note: User's file has 'slow_corner_z' not 'slow_corner_pct_z'
    if 'slow_corner_performance' in team_perf and 'slow_corner_z' in track_chars:
        demand = (track_chars['slow_corner_z'] + 2) / 4
        demand = np.clip(demand, 0, 1)
        score += 0.24 * demand * team_perf['slow_corner_performance']
        weight_sum += 0.24
    
    if 'medium_corner_performance' in team_perf and 'medium_corner_z' in track_chars:
        demand = (track_chars['medium_corner_z'] + 2) / 4
        demand = np.clip(demand, 0, 1)
        score += 0.18 * demand * team_perf['medium_corner_performance']
        weight_sum += 0.18
    
    if 'fast_corner_performance' in team_perf and 'fast_corner_z' in track_chars:
        demand = (track_chars['fast_corner_z'] + 2) / 4
        demand = np.clip(demand, 0, 1)
        score += 0.18 * demand * team_perf['fast_corner_performance']
        weight_sum += 0.18
    
    # Speed characteristics (30% of score)
    if 'top_speed' in team_perf and 'full_throttle_z' in track_chars:
        demand = (track_chars['full_throttle_z'] + 2) / 4
        demand = np.clip(demand, 0, 1)
        score += 0.18 * demand * team_perf['top_speed']
        weight_sum += 0.18
    
    if 'braking_performance' in team_perf and 'heavy_braking_z' in track_chars:
        demand = (track_chars['heavy_braking_z'] + 2) / 4
        demand = np.clip(demand, 0, 1)
        score += 0.12 * demand * team_perf['braking_performance']
        weight_sum += 0.12
    
    # Consistency (10% of score)
    if 'consistency' in team_perf and 'is_street_circuit_z' in track_chars:
        is_street = track_chars['is_street_circuit_z'] > 0
        if is_street:
            score += 0.10 * team_perf['consistency']
            weight_sum += 0.10
    
    # Normalize by actual weight used
    if weight_sum > 0:
        score = score / weight_sum
    
    return float(np.clip(score, 0, 1))


def rank_teams_for_track(
    all_team_data: Dict[str, Dict],
    track_chars: Dict,
    current_session: str = 'post_fp1',
    weekend_type: str = 'normal'
) -> List[Tuple[str, float, float, str]]:
    """
    Rank all teams for a track with smart fallback.
    
    Args:
        all_team_data: {team: {session_key: session_data}}
        track_chars: Track characteristics
        current_session: 'pre_fp1', 'post_fp1', 'post_fp2', 'post_fp3'
        weekend_type: 'normal' or 'sprint'
    
    Returns:
        List of (team, score, confidence, reasoning) sorted by score
    """
    # Get prediction context
    context = get_prediction_context(current_session, weekend_type)
    available = context['available']
    base_confidence = context['confidence_base']
    
    # Determine which session to use
    # Priority scores are only for selecting which session to use, not for confidence
    session_priority = {
        'fp3': 1.0, 
        'fp2': 0.7, 
        'fp1': 0.4, 
        'sprint_qualifying': 0.9,
        'sprint_quali': 0.9  # Same priority, just different naming
    }
    
    best_session = None
    best_score = 0
    for sess in available:
        score = session_priority.get(sess, 0)
        if score > best_score:
            best_score = score
            best_session = sess
    
    if not best_session:
        # No data available
        return [(team, 0.0, base_confidence * 0.5, "No session data") 
                for team in all_team_data.keys()]
    
    # Confidence is just the base confidence for this stage
    # (session_priority is only for selecting which session to use)
    confidence = base_confidence
    reasoning = f"Using {best_session}"
    
    # Try to extract performance
    team_performance = extract_all_teams_performance(all_team_data, best_session)
    
    # Fallback logic: try each session in priority order
    if not team_performance:
        fallback_order = ['fp1', 'fp2', 'fp3', 'sprint_qualifying', 'sprint_quali']
        for fallback_session in fallback_order:
            if fallback_session != best_session:
                team_performance = extract_all_teams_performance(all_team_data, fallback_session)
                if team_performance:
                    reasoning = f"Using {fallback_session} (fallback from {best_session})"
                    confidence = confidence * 0.6  # Clear penalty for fallback
                    break
    
    if not team_performance:
        return [(team, 0.0, confidence * 0.5, f"No data available (tried {best_session})") 
                for team in all_team_data.keys()]
    
    # Calculate suitability for each team
    rankings = []
    for team, perf in team_performance.items():
        suitability = calculate_suitability(perf, track_chars)
        rankings.append((team, suitability, confidence, reasoning))
    
    # Sort by suitability (highest first)
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    return rankings
