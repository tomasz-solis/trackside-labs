# Weight Schedule System Guide

## Overview

The Weight Schedule System implements validated findings from historical analysis of the 2021→2022 F1 regulation change to optimize 2026 predictions.

### Key Finding

**Trust current season results quickly, minimize reliance on previous year standings.**

Validation using 2021→2022 transition showed:
- **Extreme schedule: 0.809 correlation** 🏆
- Ultra-aggressive: 0.790 correlation
- Aggressive: 0.740 correlation
- Conservative: 0.512 correlation

The difference between aggressive and extreme schedules is only 0.069 - this tells us the philosophy (trust current data quickly) matters more than the exact schedule.

**Validation notebooks:**
- [validate_testing_predictions.ipynb](../notebooks/validate_testing_predictions.ipynb) - 2021→2022 regulation change analysis
- [test_weight_schedules.ipynb](../notebooks/test_weight_schedules.ipynb) - 7 schedules tested on 2022 data

## Three Components

The system blends three signals:

1. **Baseline** - 2025 constructor standings
   - Represents team resources and capability
   - Weight decreases rapidly (30% → 5% by Race 3)

2. **Testing Directionality** - Track-car suitability from pre-season testing
   - Relative strengths: straightline speed, corner speeds
   - Weight decreases rapidly (20% → 0% by Race 3)

3. **Current Season** - Running average of 2026 race results
   - Most predictive signal during regulation changes
   - Weight increases rapidly (50% at Race 1 → 95% by Race 3)

## Weight Schedule Progression

### Extreme Schedule (Recommended for 2026)

| Race | Baseline | Testing | Current | Notes |
|------|----------|---------|---------|-------|
| 1 | 30% | 20% | **50%** | Trust early results heavily |
| 2 | 15% | 10% | **75%** | Rapid adaptation |
| 3+ | 5% | 0% | **95%** | Almost entirely data-driven |

### Why This Works

Counterintuitive finding from validation:
- **Testing correlation during regulation change: 0.137** ❌
- **Testing correlation during stable years: 0.422** ✅

**Regulation changes shuffle the deck** - pre-season testing is less predictive than usual, so we trust it less and adapt to actual results faster.

## Usage

### In Predictions

```python
from src.systems.weight_schedule import calculate_blended_performance

# For McLaren at Race 1 (Bahrain)
baseline = 0.85  # 2025 P1
testing_modifier = 0.02  # Slightly good at Bahrain (from directionality)
current = 0.0  # No races yet

score = calculate_blended_performance(
    baseline_score=baseline,
    testing_modifier=testing_modifier,
    current_score=current,
    race_number=1,
    schedule='extreme'
)
# Result: ~0.29 (30% of 0.85 + 20% of 0.02 + 50% of 0.0)
```

### After Race Updates

```python
# After Race 1, McLaren finishes P3
# update_from_race.py adds this to current_season_performance list
# McLaren now has: current_season_performance = [0.85]  # P3 → 0.85 score

# For Race 2 prediction:
baseline = 0.85  # Still 2025 standings
testing_modifier = 0.01  # Different track
current = 0.85  # Running average of [0.85]

score = calculate_blended_performance(
    baseline, testing_modifier, current,
    race_number=2,
    schedule='extreme'
)
# Result: ~0.76 (15% of 0.85 + 10% of 0.01 + 75% of 0.85)
```

## File Structure

### Data Files

**[data/processed/car_characteristics/2026_car_characteristics.json](data/processed/car_characteristics/2026_car_characteristics.json)**

```json
{
  "year": 2026,
  "races_completed": 0,
  "teams": {
    "McLaren": {
      "overall_performance": 0.85,  // BASELINE (2025 P1) - never changes
      "directionality": {             // TESTING - never changes
        "max_speed": -0.005,
        "slow_corner_speed": -0.002,
        "medium_corner_speed": 0.004,
        "high_corner_speed": 0.006
      },
      "current_season_performance": []  // CURRENT - grows with each race
    }
  }
}
```

After Race 1:
```json
{
  "McLaren": {
    "overall_performance": 0.85,  // Unchanged
    "current_season_performance": [0.85],  // Added Race 1 result
    "races_completed": 1
  }
}
```

After Race 5:
```json
{
  "McLaren": {
    "overall_performance": 0.85,  // Still unchanged
    "current_season_performance": [0.85, 0.80, 0.78, 0.75, 0.73],
    "races_completed": 5
    // Running average: 0.782
  }
}
```

## Implementation Flow

### 1. Generate Predictions (Button Click)

```python
from src.systems.weight_schedule import calculate_blended_performance
import json

# Load characteristics
with open('data/processed/car_characteristics/2026_car_characteristics.json') as f:
    cars = json.load(f)

# Load track characteristics
with open('data/processed/track_characteristics/2026_track_characteristics.json') as f:
    tracks = json.load(f)

race_number = cars['races_completed'] + 1  # Next race
track_name = "Bahrain Grand Prix"

for team, team_data in cars['teams'].items():
    # 1. Baseline
    baseline = team_data['overall_performance']

    # 2. Testing modifier (calculate track suitability)
    track_profile = tracks['tracks'][track_name]
    directionality = team_data['directionality']

    testing_modifier = (
        directionality['max_speed'] * (track_profile['straights_pct'] / 100) +
        directionality['slow_corner_speed'] * (track_profile['slow_corners_pct'] / 100) +
        directionality['medium_corner_speed'] * (track_profile['medium_corners_pct'] / 100) +
        directionality['high_corner_speed'] * (track_profile['high_corners_pct'] / 100)
    )

    # 3. Current season running average
    if race_number == 1:
        current = 0.0  # No data yet
    else:
        current = np.mean(team_data['current_season_performance'])

    # 4. Blend using weight schedule
    final_score = calculate_blended_performance(
        baseline, testing_modifier, current,
        race_number, schedule='extreme'
    )

    print(f"{team}: {final_score:.3f}")
```

### 2. Update After Race

```bash
# After Race 1 completes
python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026
```

This script:
1. Fetches race results from FastF1
2. Calculates team performance (position → score)
3. Appends to `current_season_performance` list
4. Does NOT modify `overall_performance` (baseline)
5. Does NOT modify `directionality` (testing)

### 3. Next Prediction Uses Updated Data

Race 2 prediction automatically uses:
- Same baseline (2025 standings)
- Same testing directionality
- **Updated** current_season_performance with Race 1 results
- Different weights (15%/10%/75% instead of 30%/20%/50%)

## Why Not Just Use Current Season 100%?

You might ask: "If current season is so predictive, why keep baseline at all?"

**Answer**: Small baseline anchor (5-10%) provides:
1. **Stability** - Prevents wild swings from single race outliers
2. **Team capability signal** - Mercedes with bad car is still Mercedes (resources, experience)
3. **Cold start problem** - Race 1 needs SOME prior, can't be 100% current when current doesn't exist yet

Validation showed 100% current (insane schedule) performed almost identically to 95% current (0.807 vs 0.809), confirming the small anchor doesn't hurt.

## Comparison: Old vs New System

### Old System (Before Weight Schedules)

```python
# After each race, blend old and new
old_rating = 0.85
new_result = 0.75
updated = old_rating * 0.7 + new_result * 0.3
# Result: 0.82
```

Problems:
- ❌ Slow to adapt (30% learning rate)
- ❌ No distinction between baseline and current season
- ❌ No track-specific modifiers
- ❌ Not optimized for regulation changes

### New System (Weight Schedules)

```python
# Maintain separate signals
baseline = 0.85  # Never changes
testing = 0.02   # Never changes
current = [0.75]  # Grows as list

# Blend dynamically by race number
race_1_score = 0.30*baseline + 0.20*testing + 0.50*current  # Trust current heavily
race_3_score = 0.05*baseline + 0.00*testing + 0.95*current  # Almost all current
```

Benefits:
- ✅ Fast adaptation (50% current at Race 1)
- ✅ Separate baseline, testing, current signals
- ✅ Track-specific suitability from testing
- ✅ Validated 0.809 correlation on regulation changes

## Alternative Schedules

If you want different philosophies:

### Conservative (Stable Regulations)
- Race 1: 70% baseline, 25% testing, 5% current
- Race 10+: 15% baseline, 5% testing, 80% current
- Use for: Stable regulation years (2027-2029)

### Ultra-Aggressive (Experimental)
- Race 1: 35% baseline, 20% testing, 45% current
- Race 4+: 5% baseline, 0% testing, 95% current
- Slightly faster than extreme (0.790 vs 0.809)

Change schedule:
```python
score = calculate_blended_performance(..., schedule='ultra_aggressive')
```

## Summary

1. **Keep three separate signals**: baseline (2025), testing (directionality), current (running avg)
2. **Trust current season heavily from Race 1**: 50% weight
3. **By Race 3, almost ignore last year**: 5% baseline, 0% testing, 95% current
4. **This isn't guesswork**: Validated 0.809 correlation on 2021→2022 transition
5. **The philosophy matters more than exact weights**: All aggressive variants (0.740-0.809) cluster together

**For 2026: Use "extreme" schedule and trust the data.**
