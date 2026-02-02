# Dashboard Auto-Update Behavior

## Overview

The dashboard **automatically fetches FP practice data** when generating predictions via [fp_blending.py](../src/utils/fp_blending.py). Team characteristics must be updated manually after each race.

---

## What Auto-Updates

### ✅ Automatic (During Prediction)
**FP Blending** ([src/utils/fp_blending.py](../src/utils/fp_blending.py))
- When user clicks "Generate Predictions", system automatically:
  1. Calls `get_best_fp_performance(year, race_name, is_sprint)`
  2. Fetches latest FP1/FP2/FP3 or Sprint data via FastF1
  3. Blends 70% practice data + 30% model
- **No user action required**

### ❌ Manual (After Race Completion)
**Team Characteristics Update** ([scripts/update_from_race.py](../scripts/update_from_race.py))
- After each race completes, user must run:
  ```bash
  python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026
  ```
- This updates:
  - Team performance from race telemetry (median lap times)
  - Bayesian driver ratings
  - Data version (increments)
  - Uncertainty reduction

---

## Why Split Design?

### FP Data: Auto-Fetch
- **Frequency**: Multiple times per weekend (FP1→FP2→FP3)
- **Speed**: Fast (~2 seconds via FastF1 cache)
- **Risk**: Low (read-only, no data corruption risk)
- **User experience**: Seamless

### Team Characteristics: Manual Update
- **Frequency**: Once per race (every 2 weeks)
- **Speed**: Slower (~10-30 seconds with telemetry processing)
- **Risk**: Medium (writes to team characteristics JSON)
- **User control**: Explicit confirmation before updating baseline data

---

## Typical Workflow

### Weekend Flow
**Friday:**
```
User clicks "Predict Bahrain Grand Prix"
 ↓
Dashboard auto-fetches FP1 data (2 sec)
 ↓
Blends 70% FP1 + 30% model
 ↓
Shows Sprint Qualifying prediction
```

**Saturday (Sprint Weekend):**
```
User clicks "Predict Sprint Race"
 ↓
Dashboard auto-fetches Sprint Qualifying results
 ↓
Shows Sprint Race prediction

Later: User clicks "Predict Sunday Qualifying"
 ↓
Dashboard auto-fetches Sprint Race lap times (best indicator!)
 ↓
Blends 70% Sprint Race + 30% model
 ↓
Shows Sunday Qualifying prediction
```

**Sunday:**
```
User clicks "Predict Sunday Race"
 ↓
Dashboard auto-fetches Sunday Qualifying results
 ↓
Shows Sunday Race prediction
```

### Post-Race (Monday)
```bash
# User manually updates team characteristics
python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026

# Output:
# ✓ Loaded results for 20 drivers
# ✓ Updated team characteristics (v15) in data/processed/car_characteristics/2026_car_characteristics.json
#   McLaren: Race 0.85 → Avg 0.83 (3 races, uncertainty 0.30→0.27)
#   Ferrari: Race 0.78 → Avg 0.80 (3 races, uncertainty 0.30→0.27)
#   ...
```

Next weekend's predictions will now use updated team data.

---

## Technical Implementation

### FP Auto-Fetch Flow
```python
# app.py
def run_prediction(race_name, weather, _timestamps):
    predictor = get_predictor(_timestamps)

    # Automatic FP fetch happens here
    quali_result = predictor.predict_qualifying(year=2026, race_name=race_name)

    # Inside predict_qualifying:
    # 1. get_best_fp_performance() calls FastF1
    # 2. Returns FP3 (or FP2/FP1 if FP3 unavailable)
    # 3. Blends with model
```

### Manual Team Update Flow
```python
# scripts/update_from_race.py → src/systems/updater.py
def update_from_race(year, race_name, data_dir):
    # 1. Load race session via FastF1
    results, session = load_race_session(year, race_name)

    # 2. Extract telemetry (median lap times)
    race_pace = extract_team_performance_from_telemetry(session, team_names)

    # 3. Update team characteristics
    update_team_characteristics(results, session, char_file)
    #    - Appends to current_season_performance[]
    #    - Recalculates running average
    #    - Reduces uncertainty
    #    - Increments version

    # 4. Atomic write with backup
    atomic_json_write(char_file, char_data, create_backup=True)
```

---

## Cache Behavior

### FastF1 Cache
- Location: `data/raw/.fastf1_cache/`
- Auto-managed by FastF1
- Prevents re-downloading same session multiple times
- Cleared automatically after ~7 days

### Streamlit Cache
- `@st.cache_resource` on predictor instance
- Invalidates when data file timestamps change
- Ensures predictions use latest data

---

## Future Enhancements (Not Implemented)

1. **Automatic Team Updates**
   - Could run `update_from_race.py` on schedule
   - Risk: Requires monitoring for failures
   - Benefit: Fully automated system

2. **Real-Time FP Updates**
   - Could poll FastF1 during practice sessions
   - Update predictions as FP progresses
   - Benefit: Live timing integration

3. **Dashboard Button for Manual Update**
   - "Update from last race" button in UI
   - Benefit: No command line required
   - Risk: Must handle errors gracefully

---

## Verification

Check if auto-update is working:

```python
# In dashboard, after clicking "Generate Predictions":
# Look for this in logs:
# INFO: Using FP3 times for blending
# ✅ Using FP3 times (70% practice data + 30% model)

# If you see:
# ℹ️ Model-only (no practice data)
# → FP data not available (pre-weekend or data fetch failed)
```

---

## Summary

| Component | Update Mechanism | Trigger | User Action |
|-----------|------------------|---------|-------------|
| **FP Data** | Automatic | User clicks "Generate Predictions" | None |
| **Team Characteristics** | Manual | After race completes | Run `python scripts/update_from_race.py ...` |
| **Driver Ratings** | Manual | After race completes | Run `python scripts/update_from_race.py ...` |
| **Track Characteristics** | Static | Never (pre-generated) | None |

**Key Insight:** Dashboard auto-fetches **session data** (FP/Quali/Race results) but requires manual update of **learned characteristics** (team performance, driver ratings).
