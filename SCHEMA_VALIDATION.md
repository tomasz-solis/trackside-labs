# JSON Schema Validation Module

High-level validation system to prevent crashes from malformed JSON data.

## Overview

The `src/utils/schema_validation.py` module provides JSON schema validation for the F1 prediction system. It ensures that all loaded data matches the expected structure before being used, preventing runtime errors from malformed or incomplete JSON files.

## Supported Schemas

### 1. Driver Characteristics Schema

Validates driver performance data with required fields:
- `racecraft`: Driver racing ability (skill_score, overtaking_skill) - **0.0 to 1.0**
- `pace`: Qualifying and race pace (quali_pace, race_pace) - **0.0 to 1.0**
- `dnf_risk`: DNF/reliability data (dnf_rate) - **0.0 to 1.0**

**File**: `data/processed/driver_characteristics.json`

**Example**:
```json
{
  "drivers": {
    "VER": {
      "racecraft": {
        "skill_score": 0.85,
        "overtaking_skill": 0.90
      },
      "pace": {
        "quali_pace": 0.92,
        "race_pace": 0.88
      },
      "dnf_risk": {
        "dnf_rate": 0.05
      }
    }
  }
}
```

### 2. Team Characteristics Schema

Validates team performance and uncertainty data:
- `overall_performance`: Team performance rating - **0.0 to 1.0**
- `uncertainty`: Uncertainty factor for new teams - **0.0 to 1.0**

**File**: `data/processed/car_characteristics/2026_car_characteristics.json`

**Example**:
```json
{
  "teams": {
    "McLaren": {
      "overall_performance": 0.85,
      "uncertainty": 0.30
    },
    "Ferrari": {
      "overall_performance": 0.75,
      "uncertainty": 0.30
    }
  }
}
```

### 3. Track Characteristics Schema

Validates track-specific data:
- `pit_stop_loss`: Time lost in pit stop (seconds)
- `safety_car_prob`: Probability of safety car - **0.0 to 1.0**
- `overtaking_difficulty`: How hard to overtake (0=easy, 1=impossible) - **0.0 to 1.0**
- `has_sprint`: Optional boolean flag for sprint weekends

**File**: `data/processed/track_characteristics/2026_track_characteristics.json`

**Example**:
```json
{
  "tracks": {
    "Monaco Grand Prix": {
      "pit_stop_loss": 20.0,
      "safety_car_prob": 0.8,
      "overtaking_difficulty": 0.9,
      "type": "street"
    },
    "Monza Grand Prix": {
      "pit_stop_loss": 24.0,
      "safety_car_prob": 0.3,
      "overtaking_difficulty": 0.2,
      "type": "permanent"
    }
  }
}
```

## API Reference

### Main Validation Functions

#### `validate_driver_characteristics(data: Dict[str, Any]) -> None`
Validate driver characteristics JSON.

**Raises**: `ValueError` if validation fails

**Example**:
```python
from src.utils.schema_validation import validate_driver_characteristics
import json

with open('data/processed/driver_characteristics.json') as f:
    data = json.load(f)
    validate_driver_characteristics(data)
```

#### `validate_team_characteristics(data: Dict[str, Any]) -> None`
Validate team characteristics JSON.

**Raises**: `ValueError` if validation fails

#### `validate_track_characteristics(data: Dict[str, Any]) -> None`
Validate track characteristics JSON.

**Raises**: `ValueError` if validation fails

#### `validate_json(data: Dict, schema: Dict, filename: str) -> None`
Generic validation function for any schema.

**Parameters**:
- `data`: Dictionary to validate
- `schema`: JSON schema dictionary
- `filename`: Name for logging purposes

**Raises**: `ValueError` if validation fails

**Example**:
```python
from src.utils.schema_validation import validate_json, DRIVER_CHARACTERISTICS_SCHEMA

validate_json(my_data, DRIVER_CHARACTERISTICS_SCHEMA, "driver_characteristics.json")
```

### Schema Objects

The following schema dictionaries are available for direct use:
- `DRIVER_CHARACTERISTICS_SCHEMA`
- `TEAM_CHARACTERISTICS_SCHEMA`
- `TRACK_CHARACTERISTICS_SCHEMA`

## Integration Points

### Baseline 2026 Predictor
The `Baseline2026Predictor` class now validates both driver and team characteristics when loading data:

```python
from src.predictors.baseline_2026 import Baseline2026Predictor

predictor = Baseline2026Predictor('data/processed')
# ✓ Driver characteristics automatically validated
# ✓ Team characteristics automatically validated
```

### Tire Predictor
The `TirePredictor` class validates driver characteristics when loading:

```python
from src.predictors.tire import TirePredictor

tire = TirePredictor(
    year=2025,
    driver_chars_path='data/processed/driver_characteristics.json'
)
# ✓ Driver characteristics automatically validated
```

## Error Handling

When validation fails, a `ValueError` is raised with a descriptive error message:

```python
try:
    validate_driver_characteristics(malformed_data)
except ValueError as e:
    print(f"Validation error: {e}")
    # Handle gracefully or re-raise
```

Example error messages:
```
Invalid driver_characteristics.json: 'racecraft' is a required property
Invalid driver_characteristics.json: 1.5 is greater than the maximum of 1
Invalid team_characteristics.json: 'teams' is a required property
```

## Graceful Degradation

If the `jsonschema` library is not available, validation is skipped with a warning:

```
jsonschema library not available. Skipping validation of driver_characteristics.json.
Install jsonschema to enable validation: pip install jsonschema
```

## Testing

Comprehensive test suite in `tests/test_schema_validation.py`:

```bash
python -m pytest tests/test_schema_validation.py -v
```

Tests cover:
- Valid data from actual files
- Minimal valid data structures
- Missing required fields
- Out-of-range values
- Boundary conditions (0.0, 1.0)
- Optional fields
- Edge cases

## Common Issues and Solutions

### Issue: "drivers is a required property"
**Cause**: Driver characteristics JSON missing 'drivers' key

**Solution**: Ensure JSON has structure:
```json
{
  "drivers": { /* ... */ }
}
```

### Issue: "required is not a required property" (traceback shows multiple errors)
**Cause**: Invalid schema structure in data

**Solution**: Check that required fields exist:
- Driver: `racecraft`, `pace`, `dnf_risk`
- Team: `overall_performance`
- Track: (no required sub-fields, but valid structure needed)

### Issue: "X is greater than the maximum of 1"
**Cause**: Value outside 0.0-1.0 range for normalized fields

**Solution**: Normalize values to 0.0-1.0 range

## Migration Guide

If you have existing code that loads JSON without validation:

### Before
```python
with open('data/processed/driver_characteristics.json') as f:
    data = json.load(f)
    drivers = data['drivers']  # May crash if malformed
```

### After
```python
from src.utils.schema_validation import validate_driver_characteristics

with open('data/processed/driver_characteristics.json') as f:
    data = json.load(f)
    validate_driver_characteristics(data)  # Validates before use
    drivers = data['drivers']
```

Or use the integrated predictors which already include validation:

```python
from src.predictors.baseline_2026 import Baseline2026Predictor

predictor = Baseline2026Predictor('data/processed')
# Validation already done in load_data()
```

## Performance

Validation is performant for typical datasets:
- ~21 drivers: <1ms
- ~11 teams: <1ms
- ~24 tracks: <1ms

Validation is run once during initialization, not on every prediction.

## Future Enhancements

Potential improvements:
1. Custom error messages with specific field locations
2. Schema versioning for backward compatibility
3. Automatic schema migration tools
4. JSON schema generation from Python dataclasses
5. Integration with FastAPI for request validation
