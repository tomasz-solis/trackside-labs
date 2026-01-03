"""
Shared Race Data Extraction

Unified extraction logic used by both racecraft and DNF risk scripts.
Ensures consistency across all metrics.
"""

import fastf1 as ff1
import pandas as pd
import warnings
import logging

# Suppress noise
logging.getLogger("fastf1").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def extract_race_data(year, race_name):
    """
    Extract race data with unified logic.
    
    Used by both racecraft.py and dnf_risk.py to ensure consistency.
    
    Returns:
        dict: {
            driver: {
                'quali': int or None,
                'race': int,
                'gain': int or None,
                'dnf': bool,
                'has_quali': bool,
                'team': str,
                'status': str
            }
        }
    """
    try:
        quali = ff1.get_session(year, race_name, 'Q')
        quali.load(laps=False)
        
        race = ff1.get_session(year, race_name, 'R')
        race.load(laps=False)
        
        results = {}
        
        for idx, row in race.results.iterrows():
            driver = row['Abbreviation']
            
            # Get DNF flag (FastF1's actual property)
            is_dnf = row.dnf if hasattr(row, 'dnf') else False
            
            # Get race position
            if pd.notna(row['Position']):
                race_pos = int(row['Position'])
            else:
                # No race position = skip this entry
                continue
            
            # Get quali position (mark if missing)
            quali_row = quali.results[quali.results['Abbreviation'] == driver]
            if len(quali_row) > 0 and pd.notna(quali_row.iloc[0]['Position']):
                quali_pos = int(quali_row.iloc[0]['Position'])
                has_quali = True
            else:
                quali_pos = None
                has_quali = False
            
            # Calculate gain (only if we have quali)
            if has_quali:
                gain = quali_pos - race_pos
            else:
                gain = None
            
            results[driver] = {
                'quali': quali_pos,
                'race': race_pos,
                'gain': gain,
                'dnf': is_dnf,
                'has_quali': has_quali,
                'team': row['TeamName'],
                'status': row.get('Status', 'Unknown')
            }
        
        return results
        
    except Exception as e:
        # Silent failure - caller handles missing data
        return None


def extract_season(year, verbose=True):
    """
    Extract all races from a season.
    
    Args:
        year: Season year
        verbose: Print progress
        
    Returns:
        dict: {driver: [race_data, race_data, ...]}
    """
    from collections import defaultdict
    
    schedule = ff1.get_event_schedule(year)
    races = [event['EventName'] for idx, event in schedule.iterrows() 
             if 'testing' not in event.get('EventFormat', '').lower()]
    
    if verbose:
        print(f"Extracting {len(races)} races from {year}...")
        print("="*70)
    
    driver_data = defaultdict(list)
    
    for race_name in races:
        results = extract_race_data(year, race_name)
        
        if results:
            if verbose:
                print(f"  ✓ {race_name}: {len(results)} drivers")
            
            for driver, data in results.items():
                driver_data[driver].append(data)
        else:
            if verbose:
                print(f"  ✗ {race_name}: Failed")
    
    if verbose:
        print(f"\n✅ Extracted {len(races)} races")
        print(f"✅ Data for {len(driver_data)} drivers")
    
    return driver_data


def count_total_dnfs(driver_races):
    """
    Count total DNFs for a driver across all races.
    
    Uses all races, even those without qualifying data.
    
    Args:
        driver_races: List of race data dicts
        
    Returns:
        int: Total DNF count
    """
    return sum(1 for race in driver_races if race.get('dnf', False))


def get_valid_races(driver_races):
    """
    Filter to races with valid qualifying data.
    
    Used for skill calculation (need both quali and race).
    
    Args:
        driver_races: List of race data dicts
        
    Returns:
        list: Races with has_quali=True
    """
    return [race for race in driver_races if race.get('has_quali', True)]


def get_dnf_races(driver_races):
    """
    Get all DNF races.
    
    Args:
        driver_races: List of race data dicts
        
    Returns:
        list: Races where dnf=True
    """
    return [race for race in driver_races if race.get('dnf', False)]


def get_clean_races(driver_races):
    """
    Get races that finished (not DNF).
    
    Args:
        driver_races: List of race data dicts
        
    Returns:
        list: Races where dnf=False
    """
    return [race for race in driver_races if not race.get('dnf', False)]


if __name__ == '__main__':
    """Test the extraction."""
    import sys
    
    ff1.Cache.enable_cache('../data/raw/.fastf1_cache')
    
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    
    # Test extraction
    driver_data = extract_season(year)
    
    # Show summary
    print(f"\n📊 SUMMARY:")
    print("="*70)
    
    for driver in sorted(driver_data.keys())[:5]:
        races = driver_data[driver]
        total_dnfs = count_total_dnfs(races)
        valid_races = get_valid_races(races)
        
        print(f"{driver}:")
        print(f"  Total races: {len(races)}")
        print(f"  Valid races (with quali): {len(valid_races)}")
        print(f"  Total DNFs: {total_dnfs}")
        print(f"  Missing quali: {len(races) - len(valid_races)}")
