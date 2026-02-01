# 📊 Data Generation & Update Workflow

## Why This Matters

**2026 has new regulations** → Nobody knows team performance yet!
**This system uses adaptive learning** → Starts with neutral baseline, learns from actual races.

---

## 🎯 Three-Phase Approach

### Phase 1: Pre-Season Baseline (NOW - Before Testing)

**Generate baseline from historical data (2023-2025):**

```bash
python scripts/generate_2026_baseline.py
```

**What this does:**
- **Track characteristics**: 3-year averages (pit times, SC probability, overtaking difficulty)
- **Team performance**: ALL teams start at 0.5 ± 0.3 (neutral + high uncertainty)
- **Driver skills**: Carried over from 2025 (skills persist across reg changes)
- **Learning state**: Reset to 0 races completed

**Output files:**
- `data/processed/track_characteristics/2026_track_characteristics.json`
- `data/processed/car_characteristics/2026_car_characteristics.json`
- `data/processed/driver_characteristics.json`
- `data/learning_state.json`

**Data freshness**: `BASELINE_PRESEASON`
**Prediction confidence**: LOW (30-40%)

---

### Phase 2: After Testing (Feb 2026)

**Update from Barcelona/Bahrain testing:**

```bash
python scripts/update_from_testing.py --testing-session "Barcelona Day 1"
```

**What this does:**
- Analyzes testing lap times
- Updates team performance ratings (reduces uncertainty to ~0.25)
- First real glimpse of 2026 pecking order

**Data freshness**: `POST_TESTING`
**Prediction confidence**: MEDIUM (50-60%)

---

### Phase 3: Live Season Updates (After Each Race)

**After each 2026 race:**

```bash
python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026
```

**What this does:**
- Updates team performance from actual race pace
- Updates Bayesian driver skill ratings
- Reduces uncertainty progressively (floor at 0.10)
- Learns track-specific factors if different from historical baseline

**Data freshness**: `LIVE_UPDATED`
**Prediction confidence**: HIGH (70-90% by mid-season)

---

## 📈 Adaptive Learning Process

```
Pre-Season Baseline
  ↓ (All teams: 0.5 ± 0.3)
Testing Updates
  ↓ (Uncertainty: 0.3 → 0.25)
Race 1: Bahrain
  ↓ (Uncertainty: 0.25 → 0.22)
Race 2: Saudi Arabia
  ↓ (Uncertainty: 0.22 → 0.20)
...
Race 10: British GP
  ↓ (Uncertainty: ~0.10, confidence 85%+)
```

**Key principle**: System starts knowing nothing, learns from reality.

---

## 🔍 Data Freshness Indicators

When you run predictions, the system checks data freshness:

### ⚠️ `BASELINE_PRESEASON`
```
WARNING: Using PRE-SEASON BASELINE - high uncertainty!
Confidence: 30-40%
```

### ✓ `POST_TESTING`
```
INFO: Using POST-TESTING data - moderate confidence
Confidence: 50-60%
```

### ✓ `LIVE_UPDATED`
```
INFO: Updated from 5 race(s) - high confidence
Confidence: 70-90%
```

---

## 📁 File Structure

```
data/
├── processed/
│   ├── car_characteristics/
│   │   └── 2026_car_characteristics.json    ← Team performance (adaptive)
│   ├── track_characteristics/
│   │   └── 2026_track_characteristics.json  ← Track stats (historical average)
│   └── driver_characteristics.json          ← Driver skills (from 2025)
├── learning_state.json                      ← Learning system state
└── raw/                                     ← FastF1 cache (auto-generated)
```

---

## 🛠️ Manual Overrides

If you want to manually adjust characteristics:

### Set specific team rating
```python
from pathlib import Path
import json

char_file = Path("data/processed/car_characteristics/2026_car_characteristics.json")
with open(char_file) as f:
    data = json.load(f)

# Example: Boost McLaren after strong testing
data["teams"]["McLaren"]["overall_performance"] = 0.75
data["teams"]["McLaren"]["uncertainty"] = 0.20
data["teams"]["McLaren"]["note"] = "Adjusted based on testing performance"

with open(char_file, "w") as f:
    json.dump(data, f, indent=2)
```

---

## 🚨 Common Issues

### "FileNotFoundError: 2026_car_characteristics.json"
**Solution**: Run baseline generation first
```bash
python scripts/generate_2026_baseline.py
```

### "Predictions seem off after a race"
**Solution**: Update from race results
```bash
python scripts/update_from_race.py "Last Race Name"
```

### "High uncertainty even after 5 races"
**Check**: Make sure `update_from_race.py` is being run after each race. Verify `races_completed` counter in characteristics file.

---

## 🎓 Best Practices

1. **Start fresh each season**: Run `generate_2026_baseline.py` at season start
2. **Update after every race**: Automate with cron job or GitHub Actions
3. **Check data freshness**: System logs warnings if data is stale
4. **Validate results**: Compare predictions to actual results (MAE should decrease over season)
5. **Reset for 2027**: Repeat process with 2024-2026 historical data

---

## 📊 Expected Accuracy Over Season

| Phase | Races Completed | Team Uncertainty | Prediction MAE | Confidence |
|-------|-----------------|------------------|----------------|------------|
| Pre-season | 0 | 0.30 | ~5.0 positions | 30-40% |
| Post-testing | 0 | 0.25 | ~4.0 positions | 50-60% |
| Early season | 1-3 | 0.22 | ~3.5 positions | 60-70% |
| Mid-season | 4-10 | 0.15 | ~2.5 positions | 70-85% |
| Late season | 11-24 | 0.10 | ~2.0 positions | 80-90% |

**Goal**: Achieve <2.5 position MAE by mid-season through adaptive learning.

---

## 🔗 See Also

- `scripts/generate_2026_baseline.py` - Initial baseline generation
- `scripts/update_from_race.py` - Post-race learning
- `src/systems/learning.py` - Adaptive learning implementation
- `src/models/bayesian.py` - Bayesian driver ranking updates
