# API Reference

## Core Predictors

### `Baseline2026Predictor`

**Location:** `src/predictors/baseline_2026.py`

**Purpose:** Pre-season predictions when no 2026 race data exists

**Usage:**
```python
from src.predictors.baseline_2026 import Baseline2026Predictor

predictor = Baseline2026Predictor()

# Predict qualifying
quali = predictor.predict_qualifying(
    year=2026,
    race_name="Bahrain Grand Prix",
    n_simulations=50  # Monte Carlo runs
)

# Predict race
race = predictor.predict_race(
    qualifying_grid=quali['grid'],
    weather='dry',  # 'dry', 'rain', or 'mixed'
    n_simulations=50
)
```

**Returns:**
- `predict_qualifying()`: Dict with 'grid' containing 22 drivers with positions and confidence
- `predict_race()`: Dict with 'finish_order' containing race results, DNF risk, podium probability

---

## Utilities

### Weekend Type Detection

**Location:** `src/utils/weekend.py`

```python
from src.utils.weekend import is_sprint_weekend, get_weekend_type

# Check if sprint weekend
is_sprint = is_sprint_weekend(2026, "Chinese Grand Prix")  # Returns: True

# Get weekend type
weekend_type = get_weekend_type(2026, "Bahrain Grand Prix")  # Returns: 'conventional'
```

### Driver Lineups

**Location:** `src/utils/lineups.py`

```python
from src.utils.lineups import get_lineups

# Get current lineups for a race
lineups = get_lineups(2026, "Bahrain Grand Prix")
# Returns: {'McLaren': ['NOR', 'PIA'], 'Ferrari': ['LEC', 'HAM'], ...}
```

---

## Configuration

### Load Config

**Location:** `src/utils/config.py`

```python
from src.utils.config import ProductionConfig

# Load production config (session selection strategies)
config = ProductionConfig()

# Get qualifying strategy
sprint_strategy = config.get_qualifying_strategy('sprint')
# Returns: {'method': 'session_order', 'session': 'Sprint Qualifying', ...}

conv_strategy = config.get_qualifying_strategy('conventional')
# Returns: {'method': 'blend', 'blend_weight': 0.9, ...}

# Get expected MAE
quali_mae = config.get_expected_mae('qualifying', weekend_type='sprint')
# Returns: 3.22
```

**For YAML config (hyperparameters):**
```python
import yaml

with open('config/default.yaml') as f:
    config = yaml.safe_load(f)

grid_weight = config['race']['weights']['grid_weight']
# Returns: 0.3
```

See [CONFIGURATION.md](CONFIGURATION.md) for detailed hyperparameter documentation.

---

## Data Models

### Qualifying Result
```python
{
    'grid': [
        {
            'driver': 'NOR',
            'team': 'McLaren',
            'position': 1,
            'median_position': 1,
            'confidence': 58.5
        },
        # ... 21 more drivers
    ]
}
```

### Race Result
```python
{
    'finish_order': [
        {
            'driver': 'NOR',
            'team': 'McLaren',
            'position': 1,
            'confidence': 54.2,
            'podium_probability': 68.5,
            'dnf_probability': 0.087  # 8.7%
        },
        # ... 21 more drivers
    ]
}
```

---

## Constants

### Weather Types
- `'dry'`: Normal conditions (chaos_factor: 0.05)
- `'rain'`: Wet conditions (chaos_factor: 0.15)
- `'mixed'`: Variable conditions (chaos_factor: 0.10)

### Confidence Ranges
- **Baseline Predictor**: 40-60% (regulation uncertainty)
- **Bayesian Predictor**: 50-90% (after 2026 data available)

### DNF Risk Levels
- **Low**: < 10%
- **Medium**: 10-20%
- **High**: > 20%

---

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
pytest tests/test_baseline_2026_integration.py -v
pytest tests/test_dashboard_smoke.py -v
```

### Check Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## Common Patterns

### Full Prediction Workflow
```python
from src.predictors.baseline_2026 import Baseline2026Predictor

# Initialize
predictor = Baseline2026Predictor()

# 1. Predict qualifying
quali = predictor.predict_qualifying(2026, "Bahrain Grand Prix")

# 2. Predict race from qualifying grid
race = predictor.predict_race(quali['grid'], weather='dry')

# 3. Extract results
winner = race['finish_order'][0]
print(f"Winner: {winner['driver']} ({winner['confidence']:.1f}% confidence)")

# 4. Find high DNF risk drivers
high_risk = [d for d in race['finish_order'] if d['dnf_probability'] > 0.20]
print(f"High DNF risk: {[d['driver'] for d in high_risk]}")
```

### Sprint Weekend Handling
```python
from src.utils.weekend import is_sprint_weekend

race_name = "Chinese Grand Prix"

if is_sprint_weekend(2026, race_name):
    print("🏃 Sprint Weekend - predict Sprint Shootout for Sunday grid")
else:
    print("📅 Standard Weekend - predict qualifying directly")
```

---

## Error Handling

All predictors and utilities raise standard Python exceptions:

```python
try:
    result = predictor.predict_qualifying(2026, "Invalid Race")
except ValueError as e:
    print(f"Invalid race name: {e}")
except FileNotFoundError as e:
    print(f"Missing data file: {e}")
```

---

## Performance Notes

- **Monte Carlo simulations**: 50 runs recommended for stability (trade-off: accuracy vs speed)
- **Caching**: FastF1 caches data in `.fastf1_cache/` (can grow to 1GB+)
- **Memory**: Baseline predictor uses ~50MB RAM for full season simulation

---

## Version History

- **v1.0** (2026-01-28): Initial release with baseline predictor and dashboard
- Validated against 2025 season
- 80 pytest tests, 98.75% passing
- Monte Carlo simulation for prediction stability
