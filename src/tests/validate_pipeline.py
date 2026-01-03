"""
Complete Pipeline Validation

Tests the entire prediction pipeline and shows diagnostics.
Run this after extracting data to validate everything works.
"""

import sys
import json
from pathlib import Path

sys.path.append('/home/claude')

from src.archive.normalization import extract_all_teams_performance
from src.predictors.team import rank_teams_for_track
from src.helpers.session_selector import get_prediction_context


def validate_data_structure(car_data, track_data):
    """Validate that data has correct structure."""
    print("=" * 70)
    print("DATA STRUCTURE VALIDATION")
    print("=" * 70)
    
    issues = []
    
    # Check teams
    if not car_data:
        issues.append("No car data loaded!")
    else:
        print(f"🟢 Teams loaded: {len(car_data)}")
        
        # Check first team
        first_team = list(car_data.keys())[0]
        first_sessions = car_data[first_team]
        print(f"🟢 {first_team} has {len(first_sessions)} sessions")
        
        # Check session structure
        first_session_key = list(first_sessions.keys())[0]
        first_session = first_sessions[first_session_key]
        
        print(f"\n  Session example: {first_session_key}")
        print(f"    Has sector_times: {'🟢' if 'sector_times' in first_session else '🔴 MISSING!'}")
        print(f"    Has speed_profile: {'🟢' if 'speed_profile' in first_session else '🔴 MISSING!'}")
        
        if 'sector_times' not in first_session:
            issues.append(f"Session {first_session_key} missing sector_times")
        if 'speed_profile' not in first_session:
            issues.append(f"Session {first_session_key} missing speed_profile")
    
    # Check tracks
    if not track_data:
        issues.append("No track data loaded!")
    else:
        print(f"\n🟢 Tracks loaded: {len(track_data)}")
        
        first_track = list(track_data.keys())[0]
        first_track_data = track_data[first_track]
        
        required_keys = ['slow_corner_z', 'medium_corner_z', 'fast_corner_z', 
                        'full_throttle_z', 'heavy_braking_z']
        
        print(f"\n  Track example: {first_track}")
        for key in required_keys:
            if key in first_track_data:
                print(f"    {key}: 🟢 ({first_track_data[key]:.2f})")
            else:
                print(f"    {key}: 🔴 MISSING!")
                issues.append(f"Track {first_track} missing {key}")
    
    if issues:
        print(f"\n🔴 ISSUES FOUND: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n🟢 All validation checks passed!")
        return True


def validate_extraction(car_data):
    """Validate that extraction works for different sessions."""
    print("\n" + "=" * 70)
    print("EXTRACTION VALIDATION")
    print("=" * 70)
    
    sessions_to_test = ['fp1', 'fp2', 'fp3', 'sprint_qualifying']
    
    for session in sessions_to_test:
        print(f"\nTesting extraction: {session}")
        perf = extract_all_teams_performance(car_data, session)
        
        if perf:
            print(f"  🟢 Extracted {len(perf)} teams")
            
            # Check data quality
            first_team = list(perf.keys())[0]
            first_perf = perf[first_team]
            
            corner_perf = first_perf.get('slow_corner_performance', 0)
            speed_perf = first_perf.get('top_speed', 0)
            
            print(f"  Example ({first_team}):")
            print(f"    Corner: {corner_perf:.3f}")
            print(f"    Speed: {speed_perf:.3f}")
            
            if speed_perf == 0:
                print(f"  🔴 WARNING: Speed is 0!")
        else:
            print(f"  - No data (expected if session doesn't exist)")


def validate_ranking(car_data, track_data):
    """Validate ranking for a specific track."""
    print("\n" + "=" * 70)
    print("RANKING VALIDATION")
    print("=" * 70)
    
    # Test with Bahrain (should exist for everyone)
    test_track = 'Bahrain Grand Prix'
    
    if test_track not in track_data:
        print(f"🔴 {test_track} not in track data!")
        return
    
    track_chars = track_data[test_track]
    
    print(f"\nTesting ranking: {test_track}")
    
    # Test normal weekend progression
    stages = [
        ('post_fp1', 'normal', 0.5),
        ('post_fp2', 'normal', 0.7),
        ('post_fp3', 'normal', 0.85)
    ]
    
    for stage, weekend_type, expected_conf in stages:
        print(f"\n{stage}:")
        rankings = rank_teams_for_track(car_data, track_chars, stage, weekend_type)
        
        if rankings:
            top_team, top_score, actual_conf, reason = rankings[0]
            print(f"  Winner: {top_team} ({top_score:.3f})")
            print(f"  Confidence: {actual_conf:.2f} (expected: {expected_conf:.2f})")
            print(f"  Reason: {reason}")
            
            if abs(actual_conf - expected_conf) > 0.01:
                print(f"  🔴 WARNING: Confidence mismatch!")
        else:
            print(f"  🔴 No rankings!")


def validate_sprint_weekend(car_data, track_data):
    """Validate sprint weekend handling."""
    print("\n" + "=" * 70)
    print("SPRINT WEEKEND VALIDATION")
    print("=" * 70)
    
    # Try to find a sprint track
    sprint_tracks = ['Miami Grand Prix', 'Chinese Grand Prix', 
                     'Austrian Grand Prix', 'United States Grand Prix',
                     'São Paulo Grand Prix', 'Qatar Grand Prix']
    
    sprint_track = None
    for track in sprint_tracks:
        if track in track_data:
            sprint_track = track
            break
    
    if not sprint_track:
        print("🔴 No sprint weekend tracks found in data")
        return
    
    track_chars = track_data[sprint_track]
    
    print(f"\nTesting sprint: {sprint_track}")
    
    # Test sprint weekend progression
    stages = [
        ('post_fp1', 'sprint', 0.4),
        ('post_sprint_quali', 'sprint', 0.6)
    ]
    
    for stage, weekend_type, expected_conf in stages:
        print(f"\n{stage}:")
        rankings = rank_teams_for_track(car_data, track_chars, stage, weekend_type)
        
        if rankings:
            top_team, top_score, actual_conf, reason = rankings[0]
            print(f"  Winner: {top_team} ({top_score:.3f})")
            print(f"  Confidence: {actual_conf:.2f} (expected: {expected_conf:.2f})")
            print(f"  Reason: {reason}")
            
            if abs(actual_conf - expected_conf) > 0.05:
                print(f"  🔴 WARNING: Confidence mismatch!")
                
            if 'fallback' in reason.lower():
                print(f"  🔴 WARNING: Using fallback - sprint_qualifying extraction may have failed!")
        else:
            print(f"  🔴 No rankings!")


def validate_track_differentiation(car_data, track_data):
    """Validate that different tracks produce different rankings."""
    print("\n" + "=" * 70)
    print("TRACK DIFFERENTIATION VALIDATION")
    print("=" * 70)
    
    tracks_to_test = [
        ('Monaco Grand Prix', 'tight/twisty'),
        ('Italian Grand Prix', 'high-speed'),
        ('Bahrain Grand Prix', 'mixed')
    ]
    
    results = {}
    
    for track_name, track_type in tracks_to_test:
        if track_name not in track_data:
            print(f"🔴 {track_name} not found")
            continue
        
        rankings = rank_teams_for_track(
            car_data, 
            track_data[track_name], 
            'post_fp3', 
            'normal'
        )
        
        if rankings:
            results[track_name] = rankings[0][0]  # Winner
            print(f"\n{track_name} ({track_type}): {rankings[0][0]}")
    
    # Check if winners are different
    if len(set(results.values())) == 1:
        print(f"\n🔴 WARNING: Same team wins all tracks!")
        print(f"  This might be OK if one team dominates, but check if scores vary")
    else:
        print(f"\n🟢 Different tracks have different winners")


if __name__ == "__main__":
    print("\nPIPELINE VALIDATION SCRIPT")
    print("=" * 70)
    
    # Load data
    car_path = Path('/home/claude/data/processed/car_characteristics/2025_car_characteristics.json')
    track_path = Path('/home/claude/data/processed/track_characteristics/2025_track_characteristics.json')
    
    if not car_path.exists():
        print(f"🔴 Car data not found: {car_path}")
        sys.exit(1)
    
    if not track_path.exists():
        print(f"🔴 Track data not found: {track_path}")
        sys.exit(1)
    
    with open(car_path) as f:
        car_data = json.load(f)['teams']
    
    with open(track_path) as f:
        track_data = json.load(f)['tracks']
    
    # Run all validations
    if not validate_data_structure(car_data, track_data):
        print("\n🔴 Data structure validation failed!")
        sys.exit(1)
    
    validate_extraction(car_data)
    validate_ranking(car_data, track_data)
    validate_sprint_weekend(car_data, track_data)
    validate_track_differentiation(car_data, track_data)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nIf all checks passed, pipeline is working correctly!")
    print("If warnings appeared, review the specific issues above.")
