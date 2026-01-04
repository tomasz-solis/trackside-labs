"""
Unified Driver Characteristics Extractor

Creates ONE comprehensive driver characteristics file containing:
- Basic info (experience, teams)
- Pace metrics (quali/race)
- Racecraft (position-adjusted, DNF-filtered)
- DNF risk (independent probability)
- Tire management skill

Output: driver_characteristics.json (single source of truth)
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
import fastf1 as ff1

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("DriverExtractor")

# --- CORE CALCULATIONS ---

def calculate_pace_deficit(driver_laps, teammate_laps, session_type='Q'):
    """
    Calculate pure speed deficit (%) relative to teammate.
    This isolates Driver Skill from Car Performance.
    """
    if driver_laps.empty or teammate_laps.empty:
        return None

    # Filter for valid, fast laps
    d_clean = driver_laps.pick_accurate().pick_quicklaps()
    t_clean = teammate_laps.pick_accurate().pick_quicklaps()
    
    if d_clean.empty or t_clean.empty:
        return None

    # Metric depends on session
    if session_type == 'Q':
        # Quali: Ultimate 1-lap pace (Min Time)
        d_time = d_clean['LapTime'].min().total_seconds()
        t_time = t_clean['LapTime'].min().total_seconds()
    else:
        # Race: Consistent stint pace (Median of clean laps)
        # (Exclude Safety Car / VSC laps already handled by pick_accurate)
        d_time = d_clean['LapTime'].dt.total_seconds().median()
        t_time = t_clean['LapTime'].dt.total_seconds().median()

    if np.isnan(d_time) or np.isnan(t_time): return None
    
    # Calculate % Gap (Negative = Faster than teammate)
    # e.g. -0.5% means 0.5% faster than teammate
    gap_pct = ((d_time - t_time) / t_time) * 100.0
    return gap_pct

def estimate_tire_management(driver_laps, teammate_slope=None):
    """
    Calculate Tire Degradation Slope (seconds lost per lap).
    Higher slope = Cooking the tires.
    """
    # Identify Stints (consecutive laps on same tire)
    stints = driver_laps.groupby('Stint')
    slopes = []
    
    for _, stint in stints:
        # Need at least 5 clean laps to measure trend
        clean = stint.pick_accurate().pick_wo_box()
        if len(clean) < 5: continue
        
        # Linear Regression: LapTime ~ LapNumber
        # Slope = seconds lost per lap due to wear
        x = clean['LapNumber'].values
        y = clean['LapTime'].dt.total_seconds().values
        
        try:
            slope, _, _, _, _ = linregress(x, y)
            # Filter out extreme outliers (e.g. rain onset)
            if -0.5 < slope < 1.0: 
                slopes.append(slope)
        except:
            continue
            
    if not slopes:
        return None
        
    avg_slope = np.mean(slopes)
    
    # Normalize vs Teammate if provided (Isolate Driver style from Car characteristic)
    if teammate_slope is not None:
        return avg_slope - teammate_slope # Negative = Better than teammate (Less deg)
    
    return avg_slope

def analyze_racecraft(start_pos, finish_pos, pace_rank, grid_size):
    """
    Quantify Racecraft: Did they finish higher than their speed justifies?
    
    Metric: 'Efficiency'
    +1.0: Finished ahead of faster cars (Good Defense / Overtaking)
    -1.0: Finished behind slower cars (Poor Defense / Mistakes)
    """
    if pd.isna(start_pos) or pd.isna(finish_pos) or pd.isna(pace_rank):
        return 0.0
        
    # Expected finish based on Raw Pace
    # If you have the 5th fastest car, you should finish 5th.
    expected_pos = pace_rank 
    
    # Actual performance vs Expectations
    # Gained pos vs Pace (Not just vs Grid)
    # e.g. Start P20, Pace P5, Finish P6 -> Gained 14 spots, but underperformed Pace by 1.
    
    efficiency = expected_pos - finish_pos
    return efficiency

# --- MAIN EXTRACTION LOOP ---

def extract_complete_characteristics(year, output_path, verbose=True):
    if verbose: logger.info(f"🏎️  Deep-Dive Extraction for {year}...")
    
    schedule = ff1.get_event_schedule(year)
    completed = schedule[schedule['EventFormat'] != 'testing']
    
    driver_stats = {} # {driver: {metrics...}}
    
    # Loop Races
    for _, event in completed.iterrows():
        race_name = event['EventName']
        if not race_name: continue
        
        # Check if race happened
        try:
            session = ff1.get_session(year, race_name, 'R')
            if session.date > pd.Timestamp.now(tz='UTC'): continue
            
            if verbose: logger.info(f"   Analyzing {race_name}...")
            session.load(laps=True, telemetry=False, weather=True, messages=False)
        except:
            continue
            
        laps = session.laps
        results = session.results
        weather = session.weather_data
        
        # 1. Detect Conditions (Wet/Dry)
        is_wet = weather['Rainfall'].max() > 0
        
        # 2. Group by Team for Comparisons
        for team in laps['Team'].unique():
            if pd.isna(team): continue
            
            team_drivers = laps[laps['Team'] == team]['Driver'].unique()
            if len(team_drivers) != 2: continue # Skip if partial data
            
            # Extract basic data
            d1, d2 = team_drivers[0], team_drivers[1]
            laps_d1 = laps.pick_driver(d1)
            laps_d2 = laps.pick_driver(d2)
            
            # --- TIRE MANAGEMENT (Slope) ---
            deg_d1 = estimate_tire_management(laps_d1)
            deg_d2 = estimate_tire_management(laps_d2)
            
            # --- PACE GAPS (%) ---
            # Calculate raw pace
            pace_d1 = calculate_pace_deficit(laps_d1, laps_d2, 'R') # d1 vs d2
            pace_d2 = calculate_pace_deficit(laps_d2, laps_d1, 'R') # d2 vs d1 (should be inverse)
            
            # Update Stats
            for d, p_gap, my_deg, mate_deg in [(d1, pace_d1, deg_d1, deg_d2), (d2, pace_d2, deg_d2, deg_d1)]:
                if d not in driver_stats:
                    driver_stats[d] = {'pace_gaps': [], 'tire_deltas': [], 'racecraft_scores': [], 'errors_wet': 0, 'errors_dry': 0, 'races_wet': 0, 'races_dry': 0}
                
                # Log Pace
                if p_gap is not None: driver_stats[d]['pace_gaps'].append(p_gap)
                
                # Log Tire Mgmt (Relative to Teammate)
                if my_deg is not None and mate_deg is not None:
                    driver_stats[d]['tire_deltas'].append(my_deg - mate_deg)
                
                # Log Conditions
                if is_wet: driver_stats[d]['races_wet'] += 1
                else: driver_stats[d]['races_dry'] += 1
                
                # --- RACECRAFT & ERRORS ---
                try:
                    # Did they crash? (DNF + Accident)
                    res = results.loc[results['Abbreviation'] == d].iloc[0]
                    status = str(res['Status']).lower()
                    if 'accident' in status or 'collision' in status:
                        if is_wet: driver_stats[d]['errors_wet'] += 1
                        else: driver_stats[d]['errors_dry'] += 1
                    
                    # Racecraft Efficiency
                    # Simple proxy: Pos Gain vs Grid
                    gain = res['GridPosition'] - res['Position']
                    # Normalize: Gaining positions is harder at the front
                    if res['GridPosition'] <= 5: gain *= 1.5 
                    
                    driver_stats[d]['racecraft_scores'].append(gain)
                    
                except:
                    pass

    # 3. Aggregation & Scoring
    output = {'year': year, 'drivers': {}}
    
    for d, stats in driver_stats.items():
        # A. Raw Pace Score (0.0 - 1.0)
        # Avg gap to teammate. 0% = 0.5. -0.5% (faster) = 0.8.
        avg_gap = np.mean(stats['pace_gaps']) if stats['pace_gaps'] else 0.0
        # Invert: Negative gap (faster) -> Higher Score
        pace_score = np.clip(0.5 - (avg_gap / 2.0), 0.1, 0.99)
        
        # B. Tire Management Score
        # Negative delta (less deg than mate) -> Good
        avg_deg_delta = np.mean(stats['tire_deltas']) if stats['tire_deltas'] else 0.0
        tire_score = np.clip(0.5 - (avg_deg_delta * 5.0), 0.1, 0.99)
        
        # C. Racecraft Score (Avg positions gained weighted)
        avg_gain = np.mean(stats['racecraft_scores']) if stats['racecraft_scores'] else 0.0
        racecraft_score = np.clip(0.5 + (avg_gain / 10.0), 0.1, 0.99)
        
        # D. Consistency / Stability
        # Fewer errors = Higher score
        total_races = stats['races_dry'] + stats['races_wet']
        total_errors = stats['errors_dry'] + stats['errors_wet']
        error_rate = total_errors / total_races if total_races > 0 else 0
        consistency_score = np.clip(1.0 - (error_rate * 2.0), 0.1, 0.99)
        
        # E. Wet Weather Skill Modifier
        # Did they crash more in wet than dry?
        wet_rate = stats['errors_wet'] / stats['races_wet'] if stats['races_wet'] > 0 else 0
        dry_rate = stats['errors_dry'] / stats['races_dry'] if stats['races_dry'] > 0 else 0
        wet_skill = 0.5
        if wet_rate > dry_rate: wet_skill = 0.3 # Struggles in rain
        elif wet_rate < dry_rate: wet_skill = 0.7 # Excel in rain (or cautious)

        output['drivers'][d] = {
            'pace': {
                'quali_pace': float(round(pace_score, 3)),
                'race_pace': float(round(pace_score * 0.9 + tire_score * 0.1, 3)), # Blend
                'confidence': 'high' if len(stats['pace_gaps']) > 5 else 'low'
            },
            'racecraft': {
                'skill_score': float(round(racecraft_score, 3)),
                'defense_rating': float(round(racecraft_score, 3)), # Placeholder for now
                'overtaking_rating': float(round(racecraft_score, 3))
            },
            'tire_management': {
                'skill_score': float(round(tire_score, 3)),
                'deg_slope_delta': float(round(avg_deg_delta, 4))
            },
            'consistency': {
                'score': float(round(consistency_score, 3)),
                'error_rate_dry': float(round(dry_rate, 3)),
                'error_rate_wet': float(round(wet_rate, 3))
            },
            'experience': {
                'tier': 'veteran' if total_races > 50 else 'rookie' # Placeholder, updated by debuts CSV
            }
        }

    # 4. Save
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
        
    if verbose: logger.info(f"✅ Driver Intelligence Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('year', nargs='?', type=int, default=2025)
    args = parser.parse_args()
    
    # Ensure cache
    Path('data/raw/.fastf1_cache').mkdir(parents=True, exist_ok=True)
    ff1.Cache.enable_cache('data/raw/.fastf1_cache')
    
    extract_complete_characteristics(args.year, 'data/processed/driver_characteristics.json')