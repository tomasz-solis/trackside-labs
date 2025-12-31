"""
Team Lineup Management - Race-by-Race

Extracts actual lineups from session data (handles all corner cases automatically).
Falls back to current_lineups.json for future predictions.
"""

import fastf1 as ff1
import json
from pathlib import Path
import logging

logging.getLogger("fastf1").setLevel(logging.CRITICAL)


def get_lineups_from_session(year, race_name, session_type='Q'):
    """
    Extract actual lineups from a specific race session.
    
    Handles all corner cases automatically:
    - Reserve drivers (BEA at Ferrari/Haas)
    - Mid-season swaps (LAW/TSU)
    - Injuries/replacements
    
    Args:
        year: Season year
        race_name: Race name (e.g. 'Bahrain Grand Prix')
        session_type: 'Q' for qualifying (default), 'R' for race
        
    Returns:
        Dict mapping team -> [driver1, driver2]
    """
    try:
        session = ff1.get_session(year, race_name, session_type)
        session.load(laps=False, telemetry=False, weather=False)
        
        if not hasattr(session, 'results') or session.results is None:
            return None
        
        lineups = {}
        
        # Extract actual participants
        for team in session.results['TeamName'].unique():
            team_results = session.results[session.results['TeamName'] == team]
            drivers = team_results['Abbreviation'].tolist()
            
            # Take whoever actually participated
            if drivers:
                # Handle 1 or 2 drivers (edge case: DNF/DNS)
                lineups[team] = drivers[:2] if len(drivers) >= 2 else drivers
        
        return lineups
        
    except Exception as e:
        # Session data not available
        return None


def load_current_lineups(config_path='../data/current_lineups.json'):
    """
    Load current team lineups from config file.
    
    Used for future predictions (2026+) or as fallback.
    
    Args:
        config_path: Path to current_lineups.json
        
    Returns:
        Dict mapping team -> [driver1, driver2]
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        return None
    
    with open(config_file) as f:
        data = json.load(f)
    
    return data.get('current_lineups', {})


def get_lineups(year, race_name=None, config_path='../data/current_lineups.json'):
    """
    Get team lineups for a race.
    
    Strategy:
    - For 2024-2025 with race_name: Extract from actual session data
    - For 2026+ or no race_name: Use current_lineups.json
    
    This handles ALL corner cases automatically:
    - BEA racing for Haas and Ferrari in 2024
    - LAW/TSU swap in 2025
    - Any other mid-season changes
    
    Args:
        year: Season year
        race_name: Specific race name (optional for future predictions)
        config_path: Path to current_lineups.json
        
    Returns:
        Dict mapping team -> [driver1, driver2]
        
    Examples:
        # Historical race (extracts actual participants)
        lineups = get_lineups(2024, 'Austrian Grand Prix')
        # Returns whoever actually raced (handles BEA reserve correctly)
        
        # Future prediction (uses config)
        lineups = get_lineups(2026)
        # Returns current_lineups.json
        
        # Future race prediction
        lineups = get_lineups(2026, 'Monaco Grand Prix')
        # Returns current_lineups.json (same as above)
    """
    # For historical seasons with specific race, extract from data
    if year <= 2025 and race_name:
        session_lineups = get_lineups_from_session(year, race_name, 'Q')
        
        if session_lineups:
            return session_lineups
        
        # If session data failed, fall through to config
    
    # For future seasons or fallback, use current config
    current_lineups = load_current_lineups(config_path)
    
    if current_lineups:
        return current_lineups
    
    raise ValueError(
        f"No lineup data available for {year}" + 
        (f" - {race_name}" if race_name else "") +
        f"\nCreate config file at {config_path}"
    )


def save_current_lineups(lineups, config_path='../data/current_lineups.json'):
    """
    Save current lineups to config file.
    
    Use this to update lineups when drivers change.
    
    Args:
        lineups: Dict mapping team -> [driver1, driver2]
        config_path: Path to save config
        
    Example:
        lineups = {
            'Red Bull Racing': ['VER', 'LAW'],
            'McLaren': ['NOR', 'PIA'],
            ...
        }
        save_current_lineups(lineups)
    """
    from datetime import datetime
    
    output = {
        'last_updated': datetime.now().isoformat(),
        'current_lineups': lineups
    }
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"🟢 Saved lineups to {config_file}")


def extract_lineups_for_season(year, output_path=None):
    """
    Extract lineups for all races in a season.
    
    Useful for debugging or creating reference data.
    
    Args:
        year: Season year
        output_path: Optional path to save results
        
    Returns:
        Dict mapping race_name -> lineups
    """
    import fastf1 as ff1
    
    schedule = ff1.get_event_schedule(year)
    all_lineups = {}
    
    print(f"Extracting lineups for {year} season...")
    print("=" * 70)
    
    for _, event in schedule.iterrows():
        race_name = event['EventName']
        
        if 'Testing' in str(race_name):
            continue
        
        lineups = get_lineups_from_session(year, race_name, 'Q')
        
        if lineups:
            all_lineups[race_name] = lineups
            print(f"🟢 {race_name}: {len(lineups)} teams")
        else:
            print(f"🔴 {race_name}: No data")
    
    print(f"\n🟢 Extracted {len(all_lineups)} races")
    
    if output_path:
        output = {
            'season': year,
            'races': all_lineups
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"🟢 Saved to {output_file}")
    
    return all_lineups
