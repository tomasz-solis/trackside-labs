# FP Blending System

**Validated Performance: 0.809 correlation** (vs 0.666 model-only)

The FP (Free Practice) Blending System improves prediction accuracy by combining model predictions with actual practice session lap times.

---

## Overview

**Core Concept:** Teams sometimes mess up setup, so actual practice performance is a better indicator than pre-race modeling.

**Blending Formula:**
```
final_strength = (70% × FP_data) + (30% × model_prediction)
```

This 70/30 split was validated to produce the best results (0.809 correlation on 2022 regulation change).

---

## How It Works

### Normal Weekends (18/24 races)

**Session Priority:**
1. FP3 (best indicator - right before qualifying)
2. FP2 (good indicator)
3. FP1 (fallback)

**Prediction Flow:**
```
FP3 times → Blend with model → Qualifying prediction → Race prediction
```

### Sprint Weekends (6/24 races)

**Session Priority:**
1. Sprint Race times (best - actual race performance)
2. Sprint Qualifying times (good indicator)
3. FP1 (only practice session)

**Prediction Flow:**
```
Friday:  FP1 times → Blend → Sprint Qualifying prediction
Saturday: Sprint Quali → Sprint Race
          Sprint Race times → Blend → Sunday Qualifying prediction
Sunday:   Sunday Quali → Sunday Race
```

---

## Implementation

### Core Module: `src/utils/fp_blending.py`

**Key Functions:**

1. **`get_fp_team_performance(year, race_name, session_type)`**
   - Extracts median lap times per team from session
   - Converts to 0-1 performance scale (1.0 = fastest)
   - Robust to one driver having issues (uses median)

2. **`get_best_fp_performance(year, race_name, is_sprint)`**
   - Automatically finds best available session data
   - Normal: FP3 > FP2 > FP1
   - Sprint: Sprint Race > Sprint Qualifying > FP1
   - Returns None if no data available (pre-race)

3. **`blend_team_strength(model_strength, fp_performance, blend_weight=0.7)`**
   - Blends model predictions with FP data
   - Default: 70% FP, 30% model
   - Falls back to model-only if no FP data

### Integration: `src/predictors/baseline_2026.py`

Modified `predict_qualifying()` method:

```python
# Get FP data
session_name, fp_performance = get_best_fp_performance(year, race_name, is_sprint)

# Calculate model strengths (weight schedule system)
model_strengths = {team: get_blended_team_strength(team, race_name) for team in lineups}

# Blend with FP data (70/30)
blended_strengths = blend_team_strength(model_strengths, fp_performance, blend_weight=0.7)

# Use blended strengths for predictions
# ...
```

---

## Dashboard Display

The dashboard shows which data source was used:

**With FP Data:**
```
✅ Using FP3 times (70% practice data + 30% model)
```

**Without FP Data (pre-race):**
```
ℹ️ Model-only (no practice data)
```

---

## Validation Results

Tested on 2021→2022 regulation change (similar to 2026):

| Approach | Avg Correlation | Notes |
|----------|----------------|-------|
| **Extreme schedule + FP blend** | **0.809** | Best performer ✅ |
| Model-only (no FP) | 0.666 | Baseline |
| FP-only (no model) | 0.585 | Worse than model |
| Testing-only | 0.585 | Unreliable |

**Key Insight:** Blending is crucial. Neither FP-only nor model-only perform as well as the combination.

**Validation notebooks:**
- [test_weight_schedules.ipynb](../notebooks/test_weight_schedules.ipynb) - Shows 0.809 correlation with FP blending
- [sensitivity_analysis.ipynb](../notebooks/sensitivity_analysis.ipynb) - FP blend weight optimization (70/30 optimal)

---

## Why 70/30 Split?

The 70% weight on FP data reflects:
- **Setup matters more than car potential** - A great car with poor setup loses
- **Real data beats predictions** - Actual lap times reveal true performance
- **Model still valuable** - 30% model prevents over-reacting to one-off issues

Tested splits:
- 80/20: Slightly worse (too reactive)
- **70/30: Best** ✅
- 60/40: Slightly worse (not reactive enough)

---

## Sprint Weekend Handling

### Why Sprint Race times are best

For Sunday Qualifying prediction on sprint weekends:

1. **Sprint Race times** (if available) - Actual race performance
2. **Sprint Qualifying times** - Qualifying pace
3. **FP1 times** - Only practice session

Sprint Race is the most accurate because:
- Shows true race pace (not just one lap)
- Reveals reliability issues
- Demonstrates driver racecraft
- Teams often try different strategies (valuable data)

### Example: Austrian Grand Prix 2026 (Sprint Weekend)

**Friday:**
- User wants to predict Sprint Qualifying
- System uses: FP1 times (70%) + model (30%)
- Output: Sprint Quali grid

**Saturday Morning:**
- User wants to predict Sprint Race
- System uses: Sprint Quali results (actual grid)

**Saturday Afternoon:**
- User wants to predict Sunday Qualifying
- System uses: Sprint Race times (70%) + model (30%)
- Sprint Race performance = best predictor for Sunday pace

**Sunday:**
- User wants to predict Sunday Race
- System uses: Sunday Quali results (actual grid)

---

## Code Example

```python
from src.predictors.baseline_2026 import Baseline2026Predictor

predictor = Baseline2026Predictor()

# Predict qualifying (automatically uses FP data if available)
result = predictor.predict_qualifying(
    year=2026,
    race_name="Bahrain Grand Prix",
    n_simulations=50
)

# Check what data was used
print(f"Data source: {result['data_source']}")
# Output (with FP data): "FP3 times"
# Output (without FP data): "Model-only (no practice data)"

print(f"Blend used: {result['blend_used']}")
# Output: True or False

# Get predictions
for entry in result['grid'][:3]:
    print(f"P{entry['position']}: {entry['driver']} ({entry['team']})")
```

---

## Technical Details

### Lap Time Extraction

```python
# For each team:
1. Get best lap time per driver (excluding outliers)
2. Take median of team's drivers (robust to one driver having issues)
3. Convert to relative performance scale (0-1)
   - 1.0 = fastest team
   - 0.0 = slowest team
```

### Performance Scaling

```python
fastest_time = min(team_times)
slowest_time = max(team_times)

team_performance = 1.0 - (team_time - fastest_time) / (slowest_time - fastest_time)
```

This inverts times so faster = higher score.

### Blending Logic

```python
for team in teams:
    model_score = weight_schedule_system(team)  # 0-1
    fp_score = fp_performance.get(team, model_score)  # 0-1, fallback to model

    blended = 0.7 * fp_score + 0.3 * model_score
```

---

## Edge Cases

### 1. Team not in FP data
**Cause:** DNS, technical issues, no valid laps
**Solution:** Fall back to model prediction for that team

### 2. No FP data available
**Cause:** Pre-race, data unavailable
**Solution:** Use model-only (100% weight schedule system)

### 3. Sprint weekend with no Sprint Race yet
**Cause:** Predicting before Sprint Race runs
**Solution:** Fall back to Sprint Qualifying times or FP1

---

## Performance Impact

**Before (model-only):**
- 0.666 correlation
- ~30-40% confidence
- Struggles with setup-dependent tracks (Monaco, Singapore)

**After (with FP blending):**
- 0.809 correlation (+21% improvement)
- ~50-70% confidence (when FP data available)
- Better handles setup variations

**Why the improvement:**
- Catches teams that mess up setup (e.g., Mercedes 2022 porpoising)
- Identifies surprise performers (e.g., Haas 2022 Bahrain)
- Accounts for track-specific adaptations

---

## Future Enhancements

1. **Adaptive blend weights** - Increase FP weight as weekend progresses (FP1: 60%, FP2: 65%, FP3: 70%)
2. **Tire compound adjustment** - Weight laps on race-relevant compounds higher
3. **Long run analysis** - Blend in race simulation data from FP2

## 2026 Season Tracking

After races begin, use these notebooks to validate system performance:
- [validation_metrics.ipynb](../notebooks/validation_metrics.ipynb) - Track MAE, RMSE, Top-N accuracy
- [sensitivity_analysis.ipynb](../notebooks/sensitivity_analysis.ipynb) - Re-tune FP blend weights if needed

---

## Files Modified

**Created:**
- `src/utils/fp_blending.py` - Core blending logic

**Modified:**
- `src/predictors/baseline_2026.py` - Integrated FP blending into predict_qualifying
- `app.py` - Display data source used

**Validation:**
- `notebooks/test_weight_schedules.ipynb` - Shows 0.809 correlation

---

## Summary

The FP Blending System combines the best of both worlds:
- **Model predictions** (baseline + testing + current season performance)
- **Actual practice data** (real lap times showing true pace)

Result: **21% improvement in prediction accuracy** (0.809 vs 0.666)

Key to success: **70% FP data + 30% model** - neither alone performs as well as the blend.
