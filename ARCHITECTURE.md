# F1 2026 Prediction System - Architecture

## System Overview

Physics-based F1 prediction engine optimized for regulation change years (2026).

```
User (Streamlit) → Baseline2026Predictor → Weight Schedule → Predictions
                          ↓
                    update_from_race.py → JSON files → Next prediction
```

## Core Components

### 1. Prediction Engine
**File**: [src/predictors/baseline_2026.py](src/predictors/baseline_2026.py)

Main predictor. Uses:
- **Weight schedule system** (baseline + testing + current season)
- **Track-car suitability** (directionality × track profile)
- **Monte Carlo simulation** (50 runs for uncertainty)

Methods:
- `predict_qualifying()` - Grid prediction
- `predict_race()` - Race outcome from grid
- `get_blended_team_strength()` - Applies weight schedule

### 2. Weight Schedule System
**File**: [src/systems/weight_schedule.py](src/systems/weight_schedule.py)

Validated on 2021→2022 regulation change (0.809 correlation).

**Extreme schedule** (recommended for 2026):
- Race 1: 30% baseline | 20% testing | 50% current
- Race 2: 15% baseline | 10% testing | 75% current
- Race 3+: 5% baseline | 0% testing | 95% current

### 3. Data Update Flow
**File**: [scripts/update_from_race.py](scripts/update_from_race.py)

After each race:
```bash
python scripts/update_from_race.py "Race Name" --year 2026
```

Updates:
- Appends result to `current_season_performance` list
- Maintains running average
- **Does NOT modify** baseline or directionality

### 4. User Interface
**File**: [app.py](app.py)

Streamlit dashboard:
1. User selects race + weather
2. Clicks "Generate Prediction"
3. Calls `Baseline2026Predictor`
4. Displays quali grid + race predictions

## Data Files

### Car Characteristics
**File**: [data/processed/car_characteristics/2026_car_characteristics.json](data/processed/car_characteristics/2026_car_characteristics.json)

Structure:
```json
{
  "teams": {
    "McLaren": {
      "overall_performance": 0.85,    // Baseline (2025 P1) - never changes
      "directionality": {              // Testing - added after pre-season
        "max_speed": 0.0,
        "slow_corner_speed": 0.0,
        "medium_corner_speed": 0.0,
        "high_corner_speed": 0.0
      },
      "current_season_performance": [] // Grows with each race
    }
  }
}
```

### Track Characteristics
**File**: [data/processed/track_characteristics/2026_track_characteristics.json](data/processed/track_characteristics/2026_track_characteristics.json)

Telemetry-based profiles (2020-2025 historical data):
```json
{
  "tracks": {
    "Bahrain Grand Prix": {
      "straights_pct": 33.0,
      "slow_corners_pct": 9.0,
      "medium_corners_pct": 32.0,
      "high_corners_pct": 26.1,
      "overtaking_difficulty": 0.4
    }
  }
}
```

## Prediction Flow

**Race 1 (Pre-season)**:
```python
baseline = 0.85  # McLaren 2025 P1
testing = 0.0    # No testing yet
current = 0.85   # No races yet, use baseline

# Apply Race 1 weights (30/20/50)
blended = 0.30*0.85 + 0.20*0.0 + 0.50*0.85 = 0.68
```

**After Race 1**:
```python
# McLaren finishes P3 → performance = 0.85
current_season_performance = [0.85]
```

**Race 2 Prediction**:
```python
baseline = 0.85  # Still 2025 standings
testing = 0.0    # No testing yet
current = 0.85   # Running avg of [0.85]

# Apply Race 2 weights (15/10/75)
blended = 0.15*0.85 + 0.10*0.0 + 0.75*0.85 = 0.76
```

**Race 3+ Prediction**:
```python
# Weights: 5% / 0% / 95%
# Almost entirely data-driven from 2026 results
```

## Key Design Decisions

### Why Weight Schedule?
**Problem**: Regulation changes shuffle the grid. 2025 standings are weak predictors.

**Solution**: Trust current season results heavily from Race 1 (50% weight), almost exclusively by Race 3 (95% weight).

**Validation**: 2021→2022 regulation change showed 0.809 correlation with extreme schedule vs 0.512 with conservative.

### Why Running Averages?
**Before**: `new = 0.7 * old + 0.3 * race` (slow adaptation)

**After**: Maintain list, calculate mean. Weight schedule blends it dynamically.

**Benefit**: Separate baseline (capability) from current (performance).

### Why Track-Car Suitability?
**Problem**: Cars have strengths/weaknesses (straightline speed, cornering).

**Solution**: Calculate `directionality × track_profile` to estimate track-specific performance.

**Example**:
- Car A: +5% straightline, -3% corners
- Track X: 60% straights, 40% corners
- Modifier: 0.05*0.6 + (-0.03)*0.4 = +0.018 (slight advantage)

## Testing

Run validation:
```bash
python tests/test_weight_schedule_integration.py
```

Expected output:
- Weight schedule: ✓ Working
- Track suitability: ✓ Working
- Blended strength: ✓ Working
- Predictions: ✓ Using weight schedule

## Configuration

**File**: [config/default.yaml](config/default.yaml)

Key settings:
- Qualification noise (sprint vs normal weekends)
- Team/skill weights (70% team, 30% driver)
- Uncertainty floors

## Validation Framework

**Notebooks**:
- [notebooks/validate_testing_predictions.ipynb](notebooks/validate_testing_predictions.ipynb) - Regulation change analysis
- [notebooks/test_weight_schedules.ipynb](notebooks/test_weight_schedules.ipynb) - Schedule optimization

**Key Finding**: Testing is LESS predictive during regulation changes (0.137 vs 0.422 correlation).

## Future Work

1. **After 2026 Pre-Season Testing**: Update directionality from actual testing data
2. **Throughout Season**: Track correlation to validate weight schedule performance
3. **Post-Season**: Analyze if "extreme" schedule was optimal for 2026

## Entry Points

- **Dashboard**: `streamlit run app.py`
- **CLI Prediction**: `python predict_weekend.py "Race Name"`
- **Update After Race**: `python scripts/update_from_race.py "Race Name" --year 2026`
- **Tests**: `pytest tests/`
