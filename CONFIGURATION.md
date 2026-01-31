# Configuration Guide

This project uses two configuration files for different purposes.

## config/default.yaml

**Purpose:** Model hyperparameters and prediction weights

**When to use:** Tuning the prediction algorithms

**Key sections:**

### Bayesian Model
```yaml
bayesian:
  base_volatility: 0.1              # Driver performance variance
  base_observation_noise: 2.0       # Position uncertainty
  shock_threshold: 2.0              # Detect regulation changes
  shock_multiplier: 0.5             # Concept drift adjustment
```

### Race Simulation
```yaml
race:
  weights:
    pace_weight: 0.4                # Raw speed importance
    grid_weight: 0.3                # Starting position importance
    overtaking_weight: 0.15         # Track overtaking difficulty
    tire_deg_weight: 0.15           # Tire management importance

  dnf:
    base_risk: 0.05                 # Base DNF probability
    driver_error_factor: 0.15       # Driver mistakes
    street_circuit_risk: 0.05       # Extra risk on street tracks
    rain_risk: 0.10                 # Extra risk in wet conditions

  lap1:
    midfield_variance: 1.5          # Chaos at positions 8-15
    front_row_variance: 0.0         # P1-P2 usually stable
```

### Qualifying Prediction
```yaml
qualifying:
  blend:
    default: 0.7                    # Default practice/model blend
    fp3_only: 0.8                   # When FP3 available
    fp1_only: 0.4                   # When only FP1 available

  session_confidence:
    fp1: 0.2                        # Least reliable (Friday)
    fp2: 0.5                        # Medium reliability
    fp3: 0.9                        # Most reliable (Saturday AM)
    sprint_quali: 0.85              # Sprint Qualifying confidence
```

---

## config/production_config.json

**Purpose:** FastF1 session selection strategy for live predictions

**When to use:** Determining which practice session to use when multiple are available

**Location:** config/ directory

**Structure:**
```json
{
  "qualifying_methods": {
    "sprint_weekends": {
      "method": "session_order",
      "session": "Sprint Qualifying"
    },
    "conventional_weekends": {
      "method": "session_order",
      "preferred_sessions": ["FP3", "FP2", "FP1"]
    }
  }
}
```

**Session Selection Logic:**

### Sprint Weekends
- **Always use:** Sprint Qualifying (Friday evening)
- **Why:** This is the last session before Sprint Race (when fantasy locks)
- **Format:**
  - Friday: FP1 + Sprint Qualifying
  - Saturday: Sprint Race + Main Qualifying
  - Sunday: Grand Prix

### Normal Weekends
- **Preference order:** FP3 → FP2 → FP1
- **Why:** FP3 is closest to qualifying (Saturday morning)
- **Fallback logic:**
  1. Try FP3 (most recent, best indicator)
  2. If FP3 unavailable/cancelled, use FP2
  3. If FP2 unavailable, use FP1 (Friday, least reliable)

**When Sessions Get Cancelled:**
- Rain washouts → FP3 cancelled → automatically falls back to FP2
- Schedule changes → system adapts using available data
- No manual intervention needed

**Method Types:**
- `"session_order"`: Use sessions in priority order (current implementation)
- `"best_lap"`: Use session with fastest lap times (future enhancement)
- `"most_recent"`: Always use most recent session (experimental)

---

## Which Config to Modify?

| What you want to change | Which file |
|-------------------------|------------|
| Model trusts practice data too much | `config/default.yaml` → adjust `qualifying.blend` |
| DNF rates seem wrong | `config/default.yaml` → adjust `race.dnf` rates |
| Session selection order | `config/production_config.json` → adjust methods |
| Grid position too influential | `config/default.yaml` → adjust `race.weights.grid_weight` |
| Tire deg underweighted | `config/default.yaml` → adjust `race.weights.tire_deg_weight` |

---

## For 2026 Baseline Predictor

The baseline predictor (used before 2026 data exists) **does not use these configs**. It has hardcoded weights:

- Team strength: 70%
- Driver skill: 30%
- Confidence: 40-60% (based on consistency)

These are in `src/predictors/baseline_2026.py` and should remain fixed until testing data is available.

---

## Loading Configs in Code

### YAML Config
```python
import yaml

with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

# Access nested config values
grid_weight = config['race']['weights']['grid_weight']  # Returns: 0.3
pace_weight = config['race']['weights']['pace_weight']  # Returns: 0.4
base_volatility = config['bayesian']['base_volatility']  # Returns: 0.1
```

### Production Config (Session Selection)
```python
from src.utils.config import ProductionConfig

config = ProductionConfig()

# Get qualifying strategy
sprint_strategy = config.get_qualifying_strategy('sprint')
# Returns: {'method': 'session_order', 'session': 'Sprint Qualifying', 'expected_mae': 3.22, ...}

conv_strategy = config.get_qualifying_strategy('conventional')
# Returns: {'method': 'blend', 'blend_weight': 0.9, 'expected_mae': 3.60, ...}

# Get expected MAE
quali_mae = config.get_expected_mae('qualifying', weekend_type='sprint')
# Returns: 3.22
```

### Alternative: Direct JSON Loading
```python
import json
from pathlib import Path

with open('config/production_config.json') as f:
    config = json.load(f)

session = config['qualifying_methods']['sprint_weekends']['session']
# Returns: "Sprint Qualifying"
```

---

## Validation

After changing hyperparameters, validate with backtesting:

```bash
jupyter notebook notebooks/validation_report.ipynb
```

This runs the model against 2025 season results and shows MAE (Mean Absolute Error) per race.

**Target MAE: < 2.5 positions**
