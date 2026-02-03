# Prediction Tracking & Accuracy System

Track model accuracy over the season by saving predictions and comparing them to actual race results.

## Overview

The prediction tracking system allows you to:
- Save predictions after each practice session (FP1, FP2, FP3, SQ)
- Compare predictions to actual results after races
- Calculate accuracy metrics (position error, correlation, podium accuracy, etc.)
- Track improvement over the season

## How It Works

### 1. Enable Prediction Logging

In the dashboard sidebar, check **"Save Predictions for Accuracy Tracking"**

### 2. Generate Predictions

When you click "Generate Prediction" in the dashboard:
- System detects which practice sessions have completed (FP1, FP2, FP3, or SQ)
- Saves max 1 prediction per session to `data/predictions/2026/`
- Each prediction includes:
  - Qualifying grid prediction
  - Race finishing order prediction
  - Weather conditions
  - FP data blending info
  - Session name (which session data was available)

### 3. Session-Based Saving Strategy

**Normal Weekend:**
- After FP1: Save prediction (based on FP1 data only)
- After FP2: Save prediction (based on FP1 + FP2)
- After FP3: Save prediction (based on FP1 + FP2 + FP3)

**Sprint Weekend:**
- After FP1: Save prediction (based on FP1 only)
- After SQ: Save prediction (based on FP1 + Sprint Qualifying)

**Why max 1 per session?**
This lets you analyze when predictions are most accurate:
- Are FP3 predictions better than FP1?
- Should you predict quali early or wait for more data?
- Does the model improve as more practice data becomes available?

### 4. Add Actual Results

After each race completes, update predictions with actual results:

```bash
python scripts/update_prediction_actuals.py "Bahrain Grand Prix" FP1 --year 2026
```

This script:
- Fetches actual qualifying and race results from FastF1
- Updates the saved prediction with actuals
- Enables accuracy metrics calculation

### 5. View Accuracy

Go to **"Prediction Accuracy"** tab in dashboard to see:
- Overall accuracy metrics (aggregated across all races)
- Per-race breakdown
- Qualifying vs Race accuracy
- Position error (MAE)
- Correlation coefficients
- Podium and winner prediction accuracy

## Metrics Explained

### Position Accuracy
- **Exact**: % of drivers predicted in exact correct position
- **Within ±1**: % within 1 position of actual
- **Within ±3**: % within 3 positions of actual

### Mean Absolute Error (MAE)
Average position error across all drivers. Lower is better.
- MAE = 1.5 means average error is 1.5 positions

### Correlation Coefficient
Spearman correlation between predicted and actual positions.
- +1.0 = perfect correlation (perfect order prediction)
- 0.0 = no correlation (random guessing)
- -1.0 = inverse correlation (backwards prediction)

### Podium Accuracy
- **Correct Drivers**: How many of top 3 drivers were on actual podium?
- **Correct Positions**: How many drivers in exact podium position?

### Winner Accuracy
- Did we correctly predict the race winner?

## File Structure

```
data/predictions/2026/
├── bahrain_grand_prix/
│   ├── bahrain_grand_prix_fp1.json
│   ├── bahrain_grand_prix_fp2.json
│   └── bahrain_grand_prix_fp3.json
├── saudi_arabian_grand_prix/
│   ├── saudi_arabian_grand_prix_fp1.json
│   └── saudi_arabian_grand_prix_sq.json
...
```

Each JSON contains:
```json
{
  "metadata": {
    "year": 2026,
    "race_name": "Bahrain Grand Prix",
    "session_name": "FP1",
    "predicted_at": "2026-03-01T12:00:00",
    "weather": "dry",
    "fp_blend_info": {...}
  },
  "qualifying": {
    "predicted_grid": [...]
  },
  "race": {
    "predicted_results": [...]
  },
  "actuals": {
    "qualifying": [...],  // Added after race
    "race": [...]         // Added after race
  }
}
```

## Example Workflow

### Pre-Season
1. Enable prediction tracking in sidebar
2. Generate predictions before FP1 (baseline only)

### Race Weekend
1. After FP1 completes → Generate prediction
   - System saves: `bahrain_grand_prix_fp1.json`
2. After FP2 completes → Generate prediction
   - System saves: `bahrain_grand_prix_fp2.json`
3. After FP3 completes → Generate prediction
   - System saves: `bahrain_grand_prix_fp3.json`

### Post-Race
1. Race completes on Sunday
2. Update predictions with actuals:
   ```bash
   python scripts/update_prediction_actuals.py "Bahrain Grand Prix" FP1 --year 2026
   python scripts/update_prediction_actuals.py "Bahrain Grand Prix" FP2 --year 2026
   python scripts/update_prediction_actuals.py "Bahrain Grand Prix" FP3 --year 2026
   ```
3. View accuracy in dashboard "Prediction Accuracy" tab

### Analysis
Compare FP1 vs FP2 vs FP3 predictions:
- Which session gave most accurate predictions?
- Did waiting for FP3 data improve accuracy?
- Should you predict quali earlier or later?

## Tips

1. **Don't predict too early**: Before FP1, predictions are based only on testing/historical data (low accuracy)

2. **Track by session**: Different sessions give different insights:
   - FP1: Initial read on pace
   - FP2: Race simulation data
   - FP3: Qualifying simulation data

3. **Sprint weekends are different**: Only FP1 and Sprint Qualifying available before main race

4. **F1 is unpredictable**: Even with perfect data, chaos happens:
   - Crashes, red flags, strategy errors, driver mistakes
   - Target: Beat informed guessing, not perfect prediction

5. **Focus on trends**: Look at improvement over season, not individual race accuracy

## Technical Details

### Modules
- `src/utils/prediction_logger.py`: Save/load predictions
- `src/utils/prediction_metrics.py`: Calculate accuracy metrics
- `src/utils/session_detector.py`: Detect completed sessions
- `scripts/update_prediction_actuals.py`: Add actuals to predictions

### Tests
- `tests/test_prediction_logger.py`: Prediction logging tests
- `tests/test_prediction_metrics.py`: Metrics calculation tests

### Integration
- Dashboard: [app.py](../app.py) lines 180-195 (toggle + saving logic)
- Accuracy tab: [app.py](../app.py) "Prediction Accuracy" page
