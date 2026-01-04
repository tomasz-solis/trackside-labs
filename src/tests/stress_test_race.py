import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.predictors.race import RacePredictor

# 1. Mock Data (So we don't need real JSON files to test logic)
MOCK_DRIVERS = {
    'VER': {'racecraft': {'skill_score': 0.95}, 'consistency': {'score': 0.9}, 'wet_weather': 0.95}, # God tier
    'NOR': {'racecraft': {'skill_score': 0.85}, 'consistency': {'score': 0.8}, 'wet_weather': 0.80}, # Top tier
    'COL': {'racecraft': {'skill_score': 0.40}, 'consistency': {'score': 0.3}, 'wet_weather': 0.20}, # Rookie/Prone to error
}

MOCK_GRID = [
    {'driver': 'VER', 'team': 'Red Bull Racing', 'position': 1},
    {'driver': 'NOR', 'team': 'McLaren', 'position': 2},
    {'driver': 'COL', 'team': 'Alpine', 'position': 3}, # COL starts high (unrealistic, but good for testing "drop")
]

def run_stress_test():
    print("🔥 INITIALIZING RACE PREDICTOR STRESS TEST...")
    
    # Initialize Predictor with Mock Data
    predictor = RacePredictor(
        year=2026,
        driver_chars=MOCK_DRIVERS
    )
    
    # --- SCENARIO 1: BORING DRY RACE ---
    print("\n☀️  SCENARIO 1: Sunny Day at Monza (Easy Overtaking)")
    dry_result = predictor.predict(
        year=2026,
        race_name="Monza Grand Prix",
        qualifying_grid=MOCK_GRID,
        overtaking_factor=0.2, # Easy overtaking
        weather_forecast='dry',
        verbose=True
    )
    
    print(f"   VER Finish: P{dry_result['finish_order'][0]['position']} (Expected)")
    print(f"   COL Finish: P{next(d['position'] for d in dry_result['finish_order'] if d['driver'] == 'COL')}")

    # --- SCENARIO 2: CHAOS WET RACE ---
    print("\n⛈️  SCENARIO 2: Monsoons at Monaco (Hard Overtaking)")
    wet_result = predictor.predict(
        year=2026,
        race_name="Monaco Grand Prix",
        qualifying_grid=MOCK_GRID,
        overtaking_factor=0.9, # Impossible overtaking
        weather_forecast='rain', # Triggers wet_skill logic
        verbose=True
    )
    
    # Analyze the chaos
    COL_wet = next(d for d in wet_result['finish_order'] if d['driver'] == 'COL')
    
    print(f"   COL Finish: P{COL_wet['position']}")
    print(f"   COL DNF Prob: {COL_wet['dnf_probability']:.1%} (Should be high!)")
    
    # Verification
    if COL_wet['dnf_probability'] > 0.2:
        print("\n✅ TEST PASSED: Weather logic correctly punished low-skill driver.")
    else:
        print("\n❌ TEST FAILED: Rookie survived the rain too easily.")

if __name__ == "__main__":
    run_stress_test()