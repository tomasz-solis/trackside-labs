"""
Weekend Type Utilities - NO HARDCODING!

ALWAYS uses FastF1's EventFormat - never hardcoded sprint lists.

EventFormat values:
- 'sprint', 'sprint_qualifying', 'sprint_shootout' → sprint weekend
- 'conventional' → normal weekend

Usage:
    from src.utils.weekend_utils import get_weekend_type, is_sprint_weekend
    
    weekend_type = get_weekend_type(2025, 'Chinese Grand Prix')
    # Returns: 'sprint' or 'conventional'
    
    if is_sprint_weekend(2025, race_name):
        session = 'Sprint Qualifying'
"""

import fastf1
from typing import Literal


def get_weekend_type(year: int, race_name: str) -> Literal['sprint', 'conventional']:
    """
    Get weekend type from FastF1 EventFormat.
    
    NEVER uses hardcoded sprint race lists!
    
    Args:
        year: Season year
        race_name: Race name (e.g., 'Chinese Grand Prix')
        
    Returns:
        'sprint' if sprint weekend, 'conventional' otherwise
        
    Raises:
        ValueError: If race not found in schedule
    """
    schedule = fastf1.get_event_schedule(year)
    
    # Try exact match first
    event = schedule[schedule['EventName'] == race_name]
    
    if event.empty:
        # Try case-insensitive match
        event = schedule[schedule['EventName'].str.lower() == race_name.lower()]
    
    if event.empty:
        raise ValueError(
            f"Race '{race_name}' not found in {year} schedule. "
            f"Available races: {list(schedule['EventName'])}"
        )
    
    # Get EventFormat
    event_format = str(event.iloc[0]['EventFormat']).lower()
    
    # Check if sprint weekend
    # Possible sprint formats: 'sprint', 'sprint_qualifying', 'sprint_shootout'
    if any(keyword in event_format for keyword in ['sprint']):
        return 'sprint'
    else:
        return 'conventional'


def is_sprint_weekend(year: int, race_name: str) -> bool:
    """
    Check if weekend has sprint format.
    
    Args:
        year: Season year
        race_name: Race name
        
    Returns:
        True if sprint weekend, False otherwise
    """
    try:
        return get_weekend_type(year, race_name) == 'sprint'
    except ValueError:
        return False


def get_event_format(year: int, race_name: str) -> str:
    """
    Get exact EventFormat from FastF1.
    
    Args:
        year: Season year
        race_name: Race name
        
    Returns:
        EventFormat string (e.g., 'sprint', 'sprint_qualifying', 'conventional')
    """
    schedule = fastf1.get_event_schedule(year)
    event = schedule[schedule['EventName'] == race_name]
    
    if event.empty:
        event = schedule[schedule['EventName'].str.lower() == race_name.lower()]
    
    if event.empty:
        raise ValueError(f"Race '{race_name}' not found in {year} schedule")
    
    return str(event.iloc[0]['EventFormat'])


def get_all_sprint_races(year: int) -> list[str]:
    """
    Get all sprint race names for a season.
    
    Dynamically from schedule - NO HARDCODING!
    
    Args:
        year: Season year
        
    Returns:
        List of sprint race names
    """
    schedule = fastf1.get_event_schedule(year)
    
    sprint_races = []
    for _, event in schedule.iterrows():
        event_format = str(event['EventFormat']).lower()
        # Check for any sprint format
        if 'sprint' in event_format:
            sprint_races.append(event['EventName'])
    
    return sprint_races


def get_all_conventional_races(year: int) -> list[str]:
    """
    Get all conventional race names for a season.
    
    Args:
        year: Season year
        
    Returns:
        List of conventional race names
    """
    schedule = fastf1.get_event_schedule(year)
    
    conventional_races = []
    for _, event in schedule.iterrows():
        event_format = str(event['EventFormat']).lower()
        if 'sprint' not in event_format:
            conventional_races.append(event['EventName'])
    
    return conventional_races


def get_best_qualifying_session(year: int, race_name: str) -> str:
    """
    Get best session for qualifying prediction.
    
    Sprint weekends: 'Sprint Qualifying'
    Normal weekends: 'FP3'
    
    Args:
        year: Season year
        race_name: Race name
        
    Returns:
        Session name
    """
    weekend_type = get_weekend_type(year, race_name)
    
    if weekend_type == 'sprint':
        return 'Sprint Qualifying'
    else:
        return 'FP3'


# Testing
if __name__ == '__main__':
    print("Weekend Type Utilities - NO HARDCODING!")
    print("="*70)
    
    year = 2025
    
    print(f"\n{year} Sprint Races (from FastF1 EventFormat):")
    sprint_races = get_all_sprint_races(year)
    for race in sprint_races:
        event_format = get_event_format(year, race)
        print(f"  - {race} ({event_format})")
    
    print(f"\nTotal: {len(sprint_races)} sprint weekends")
    
    print(f"\n{year} Conventional Races:")
    conventional = get_all_conventional_races(year)
    print(f"  Total: {len(conventional)} conventional weekends")
    
    # Test specific races
    print(f"\nTesting specific races:")
    test_races = [
        'Bahrain Grand Prix',
        'Chinese Grand Prix',
        'Miami Grand Prix',
        'Monaco Grand Prix'
    ]
    
    for race in test_races:
        try:
            weekend_type = get_weekend_type(year, race)
            event_format = get_event_format(year, race)
            best_session = get_best_qualifying_session(year, race)
            print(f"  {race}:")
            print(f"    Type: {weekend_type}")
            print(f"    Format: {event_format}")
            print(f"    Best session: {best_session}")
        except ValueError as e:
            print(f"  {race}: ERROR - {e}")
    
    print("\n✅ All data from FastF1 - NO HARDCODED LISTS!")
