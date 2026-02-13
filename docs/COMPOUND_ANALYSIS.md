# Compound Performance Analysis System

This system collects, normalizes, and applies tire compound-specific performance data to improve race predictions.

## Overview

F1 teams have varying performance characteristics on different tire compounds (SOFT, MEDIUM, HARD). Some teams excel on softer compounds with high grip but struggle with degradation, while others perform consistently across all compounds. This system captures these differences from session data and applies them during race predictions.

## What Compound Data Is Collected

### Data Sources

Compound performance is extracted from:
- **Race sessions** ([src/systems/updater.py](../src/systems/updater.py))
- **Practice sessions (FP1/FP2/FP3)** ([src/systems/testing_updater.py](../src/systems/testing_updater.py))
- **Pre-season testing** (via testing_updater)

### Metrics Extracted

For each team and compound combination, we collect:

1. **Median lap time** (raw seconds)
   - Most representative lap time on that compound
   - Robust to outliers (uses median, not mean)

2. **Tire degradation slope** (seconds/lap)
   - How much lap time increases per lap
   - Linear regression across stint
   - Filtered: only slopes between -0.3 and +1.0 accepted

3. **Consistency** (standard deviation)
   - Lap time variance across stint
   - Lower = more consistent

4. **Laps sampled** (count)
   - How many laps contributed to these metrics
   - Used for reliability weighting

### MIN_LAPS_PER_COMPOUND Threshold

**Threshold: 8 laps minimum** ([src/systems/compound_analyzer.py:22](../src/systems/compound_analyzer.py#L22))

**Rationale:**
- First 2-3 laps = tire warm-up (not representative)
- Laps 4-8+ = stable performance window
- <8 laps = insufficient data, skipped

This prevents noise from short stints or out-laps.

## How Compound Data Is Normalized

### Track-Specific Normalization

**Critical:** Compound performance is normalized **within each track**, never across tracks.

Why? Melbourne SOFT ≠ Monaco SOFT
- Different track surfaces
- Different temperatures
- Different layout characteristics

Process ([src/systems/compound_analyzer.py:196-274](../src/systems/compound_analyzer.py#L196-L274)):
1. Collect all teams' data for same compound at same track
2. Find best/worst values for each metric
3. Normalize to 0-1 scale (1.0 = best, 0.0 = worst)
4. Store both raw and normalized values

### Normalized Metrics Stored

- `pace_performance` (0-1): Inverted median lap time
- `tire_deg_performance` (0-1): Inverted degradation slope (1.0 = low deg)
- `consistency_performance` (0-1): Inverted std deviation

## How Compound Data Is Applied

### 1. Dynamic Compound Selection

**New in this update:** Races now select compounds based on tire stress data.

Function: [src/predictors/baseline_2026.py:237-284](../src/predictors/baseline_2026.py#L237-L284)

Uses [data/2025_pirelli_info.json](../data/2025_pirelli_info.json) (fallback for 2026):
- Calculate average stress: (traction + braking + lateral + abrasion) / 4
- **High stress (>threshold):** HARD compound (Bahrain, Singapore, Hungary)
- **Low stress (<threshold):** SOFT compound (Monaco, Canada)
- **Medium stress (between thresholds):** MEDIUM compound (most tracks)

**Thresholds configured in [config/default.yaml](../config/default.yaml):**
```yaml
baseline_predictor:
  compound_selection:
    high_stress_threshold: 3.5    # Above this: HARD
    low_stress_threshold: 2.5     # Below this: SOFT
    default_stress_fallback: 3.0  # Default if metric missing
```

This allows easy tuning without code changes.

### 2. Team Strength Adjustment

Function: [src/utils/compound_performance.py:17-70](../src/utils/compound_performance.py#L17-L70)

Modifier calculation:
- Weighted combination: 70% pace + 30% tire degradation
- Centered around 0.5 (neutral)
- Scaled to ±0.05 modifier range
- Applied to base team strength

Example:
- Team base strength: 0.75
- Compound modifier: +0.03 (good on SOFT)
- Adjusted strength: 0.78

### 3. Race Prediction Integration

Applied in [src/predictors/baseline_2026.py:926-936](../src/predictors/baseline_2026.py#L926-L936):
1. Determine race compound (tire stress based)
2. Get compound-adjusted team strength for each driver
3. Use adjusted strength in Monte Carlo race simulation

## When Compound Adjustments Are Used

**Reliability Check:** [src/utils/compound_performance.py:106-133](../src/utils/compound_performance.py#L106-L133)

Compound data is only applied if:
- ≥2 compounds have data
- ≥10 total laps sampled across compounds
- Each compound has ≥3 laps

If reliability check fails → use base team strength (no compound adjustment)

## Storage Format

Location: [data/processed/car_characteristics/2026_car_characteristics.json](../data/processed/car_characteristics/2026_car_characteristics.json)

```json
{
  "teams": {
    "McLaren": {
      "overall_performance": 0.85,
      "compound_characteristics": {
        "SOFT": {
          "track_name": "Bahrain Grand Prix",
          "median_lap_time": 91.234,
          "tire_deg_slope": 0.045,
          "consistency": 0.187,
          "pace_performance": 0.92,
          "tire_deg_performance": 0.78,
          "consistency_performance": 0.85,
          "laps_sampled": 24,
          "sessions_used": 2
        },
        "MEDIUM": { ... },
        "HARD": { ... }
      }
    }
  }
}
```

## Track-Aware Blending

When new session data arrives ([src/systems/compound_analyzer.py:277-351](../src/systems/compound_analyzer.py#L277-L351)):
- **Same track:** Blend old + new (default 50/50 weight)
- **Different track:** Replace entirely (no cross-track contamination)

This ensures Monaco SOFT data doesn't pollute Monza SOFT data.

## Multi-Stint Race Strategy System (NEW)

### Lap-by-Lap Simulation

**Status: ✅ Implemented**

The race predictor now uses full lap-by-lap simulation with multi-compound pit stop strategies:

**Architecture:** Three core modules
- [src/utils/tire_degradation.py](../src/utils/tire_degradation.py) - Tire physics and fuel effects
- [src/utils/pit_strategy.py](../src/utils/pit_strategy.py) - Monte Carlo pit strategy generation
- [src/utils/lap_by_lap_simulator.py](../src/utils/lap_by_lap_simulator.py) - Race simulation engine

**Key Features:**
1. **FIA Rule Enforcement:** All strategies use ≥2 different compounds per dry race
2. **Monte Carlo Pit Timing:** Realistic variance in pit stop laps (±3 laps for 1-stop)
3. **Tire Degradation Modeling:**
   - Linear degradation based on tire_deg_slope from compound_characteristics
   - Fuel load effect (heavier car = faster degradation)
   - Fresh tire advantage (SOFT: 0.5s, MEDIUM: 0.3s, HARD: 0.1s for first 2-3 laps)
   - Traffic effect (P1-5: 5% better, P16+: 5% worse tire life)
4. **Track-Specific Pit Loss:** Uses actual track data (Monaco 19s, Singapore 24s)
5. **Strategy Generation:** Tire stress drives stop count (high stress → 80% 2-stop probability)

**Configuration:** All parameters in [config/default.yaml](../config/default.yaml) under `baseline_predictor.race`:
- `tire_strategy.windows` - Pit stop lap windows (1-stop, 2-stop)
- `tire_strategy.stop_probability` - Stress-based stop count probabilities
- `tire_physics.fresh_tire_advantage` - Compound-specific fresh tire gains
- `strategy_constraints` - FIA rules, safety margins, optimality ratio

**Data Sources:**
- Tire degradation slopes: [data/processed/car_characteristics/2026_car_characteristics.json](../data/processed/car_characteristics/2026_car_characteristics.json)
- Tire stress scores: [data/2025_pirelli_info.json](../data/2025_pirelli_info.json)
- Track-specific pit loss: [data/processed/track_characteristics/2026_track_characteristics.json](../data/processed/track_characteristics/2026_track_characteristics.json)

**Example Output:**
```
🏎️ Tire Compound Strategies
SOFT→MEDIUM: 62.5%
MEDIUM→HARD: 28.3%
SOFT→HARD: 9.2%

⏱️ Pit Stop Windows
L25-30: 35 stops
L30-35: 28 stops
L20-25: 12 stops
```

**Testing:** Full test coverage in:
- [tests/test_tire_degradation.py](../tests/test_tire_degradation.py) (18 tests)
- [tests/test_pit_strategy.py](../tests/test_pit_strategy.py) (22 tests)

## Current Limitations

1. **No wet compound modeling**
   - INTERMEDIATE and WET compounds collected but not used in predictions
   - Fallback: Base team strength in rain conditions
   - Future: Apply compound adjustments for wet races

2. **Temperature sensitivity not modeled**
   - Track temperature affects compound performance
   - Hot tracks favor HARD, cool tracks favor SOFT
   - Future: Integrate temperature forecast data into compound selection

## Testing

Test coverage: [tests/test_compound_analyzer.py](../tests/test_compound_analyzer.py)
- Extraction from lap data
- Normalization across teams
- Aggregation with track awareness
- Minimum lap threshold enforcement

## Performance Impact

With compound adjustments (realistic scenario):
- Teams with good compound data: ±0.02 to ±0.05 strength adjustment
- Average impact: ~0.5-1.0 position change in race predictions
- Benefit: More accurate predictions for compound-sensitive tracks (Monaco, Singapore, Bahrain)

## Future Enhancements

1. **Temperature sensitivity**
   - Track temperature affects compound performance
   - Hot tracks favor HARD, cool tracks favor SOFT
   - Integration point: `get_fresh_tire_advantage()` already has track_temp parameter

2. **Track evolution**
   - Rubber buildup improves grip over race
   - Affects compound degradation patterns
   - Could reduce tire_deg_slope dynamically as race progresses

3. **Compound-specific driver skill**
   - Some drivers excel at managing tire deg
   - Could add driver × compound interaction effects
   - Would require collecting driver-level stint data

4. **Undercut/overcut dynamics**
   - Model strategic pit stop timing advantages
   - Track position changes during pit stop phases
   - Requires modeling gap intervals between cars
