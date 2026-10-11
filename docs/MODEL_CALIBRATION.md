# Model calibration report, 2026 season

*Generated: 2026-10-10T21:17:28.465284+00:00*

Checks three things: are the uncertainty intervals honest, is the model biased
for or against any driver or team, and does it beat a naive baseline.

Built from saved forecasts by `scripts/generate_evaluation_report.py`. Re-run
after each race.

---

## Production Readiness Gate

- Status: **FAIL**
- Score estimate: **88/100**

Blocking reasons:
- qualifying MAE does not beat the previous-race naive baseline
- qualifying interval coverage 0.755 is outside [0.870, 0.930]

| Metric | Value |
|---|---|
| Scored race weekends | 16 |
| Qualifying MAE improvement vs naive | -0.18 |
| Race MAE improvement vs naive | 0.93 |
| Qualifying interval coverage | 75.5% |

---

## Coverage

- Prediction source: **artifact_store**
- Event order: **season_calendar**
- Prediction artifacts analyzed: **72**
- Latest qualifying checkpoints selected: **16**
- Latest race checkpoints selected: **18**
- Qualifying races with actuals: **16**
- Race results with actuals: **18**
- Intermediate qualifying checkpoints ignored in canonical evaluation: **56**
- Intermediate race checkpoints ignored in canonical evaluation: **54**

---

## 0. Accuracy Overview

Both rows score the forecasts as the site served them.

| Session | Events | MAE | Exact match | Within-3 | Spearman rho | Kendall tau |
|---|---|---|---|---|---|---|
| Qualifying | 16 | 2.90 | 18.22% | 67.57% | 0.80 | 0.63 |
| Race | 18 | 3.38 | 14.39% | 67.17% | 0.70 | 0.57 |

### Candidate: qualifying stabilizer (not live)

A what-if on the same forecasts. The site does not run it and the gate ignores it.
The weights were set on this season, so the gain is partly in-sample.
Replayed on the current model it loses (`docs/MODEL_LEDGER.md`, 2026-10-10).

- Rank method: previous-race rank blend using only completed prior-race actuals.
- Model rank weight: **0.40**
- Previous-race rank weight: **0.60**
- Interval method: conformal widening from past interval residuals only.
- Default margin before history is sufficient: **1.00 position(s)**
- Minimum history events before learned margins: **3**
- Served qualifying MAE: **2.90**
- With stabilizer: **2.64**
- Served MAE improvement vs naive: **-0.18**
- With stabilizer: **0.09**
- Served interval coverage: **75.5%**
- With stabilizer: **88.6%**

| Race | Target | Margin | Prior events available |
|---|---|---|---|
| Australian Grand Prix | main_qualifying | 1.00 | 0 |
| Chinese Grand Prix | sprint_qualifying | 1.00 | 1 |
| Japanese Grand Prix | main_qualifying | 1.00 | 2 |
| Miami Grand Prix | sprint_qualifying | 1.00 | 3 |
| Canadian Grand Prix | sprint_qualifying | 1.00 | 4 |
| Monaco Grand Prix | main_qualifying | 2.00 | 5 |
| Barcelona Grand Prix | main_qualifying | 2.00 | 6 |
| Austrian Grand Prix | main_qualifying | 2.00 | 7 |
| British Grand Prix | sprint_qualifying | 2.00 | 8 |
| Belgian Grand Prix | main_qualifying | 1.00 | 9 |
| Hungarian Grand Prix | main_qualifying | 1.00 | 10 |
| Dutch Grand Prix | sprint_qualifying | 1.00 | 11 |
| Italian Grand Prix | main_qualifying | 1.00 | 12 |
| Spanish Grand Prix | main_qualifying | 2.00 | 13 |
| Azerbaijan Grand Prix | main_qualifying | 2.00 | 14 |
| Bahrain Grand Prix | main_qualifying | 2.00 | 15 |

### Weekend Format Coverage

| Format | Prediction artifacts | Qualifying pairs | Race pairs |
|---|---|---|---|
| normal | 52 | 41 | 48 |
| sprint | 20 | 10 | 17 |

---

## Selection Policy

Canonical evaluation uses `latest_checkpoint_per_race_and_target` so each race/target contributes at most one scored forecast.

### Selected Checkpoints

| Session | Checkpoint | Count |
|---|---|---|
| qualifying | FP1 | 5 |
| qualifying | FP2 | 1 |
| qualifying | FP3 | 10 |
| race | FP2 | 1 |
| race | FP3 | 3 |
| race | Q | 9 |
| race | SQ | 5 |

### Selected Targets

| Session | Target | Count |
|---|---|---|
| qualifying | main_qualifying | 11 |
| qualifying | sprint_qualifying | 5 |
| race | grand_prix_race | 13 |
| race | sprint_race | 5 |

---

## 1. Segmented Performance

### Qualifying

#### Weekend Format

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| normal | 11 | 2.92 | 19.01% | 67.77% | 0.79 |
| sprint | 5 | 2.85 | 16.49% | 67.14% | 0.81 |

#### Weather

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| dry | 16 | 2.90 | 18.22% | 67.57% | 0.80 |

#### Track Type

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| permanent | 11 | 2.79 | 18.65% | 66.88% | 0.82 |
| street | 4 | 3.43 | 14.77% | 64.77% | 0.71 |
| unknown | 1 | 1.91 | 27.27% | 86.36% | 0.92 |

### Race

#### Weekend Format

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| normal | 11 | 3.79 | 10.33% | 63.22% | 0.64 |
| sprint | 7 | 2.73 | 20.78% | 73.38% | 0.79 |

#### Weather

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| dry | 18 | 3.38 | 14.39% | 67.17% | 0.70 |

#### Track Type

| Bucket | Events | MAE | Exact match | Within-3 | Spearman rho |
|---|---|---|---|---|---|
| permanent | 13 | 3.01 | 17.48% | 70.98% | 0.75 |
| street | 4 | 4.75 | 5.68% | 50.00% | 0.51 |
| unknown | 1 | 2.64 | 9.09% | 86.36% | 0.74 |

---

## 2. Confidence Interval Calibration (Qualifying)

Each driver gets a p5 to p95 position interval. About 90% of actual results
should land inside it.

| Metric | Value |
|---|---|
| Races with interval data | 16 |
| Driver-race predictions covered | 351 |
| Nominal coverage (target) | 90.0% |
| Empirical coverage (actual) | 75.5% |
| Calibration error | -0.15 (-14.5%) |
| Mean interval width | 6.80 positions |

Warning: Intervals are too **tight** - model is overconfident by 14.5pp.

Negative calibration error: intervals too narrow (overconfident).
Positive: intervals too wide.

---

## 3. Error Analysis

### Qualifying

Evaluated **16** event(s).

Worst weekends:
- Monaco Grand Prix (`street`, `normal`, `dry`) MAE=4.36, winner=ANT -> ANT
- Barcelona Grand Prix (`permanent`, `normal`, `dry`) MAE=3.64, winner=VER -> RUS
- Canadian Grand Prix (`permanent`, `sprint`, `dry`) MAE=3.55, winner=VER -> RUS

Drivers that show up repeatedly among the largest misses:
- SAI: appearances=8, avg_abs_error=6.88
- LAW: appearances=5, avg_abs_error=9.00
- ALB: appearances=4, avg_abs_error=7.75

### Race

Evaluated **18** event(s).

Worst weekends:
- Monaco Grand Prix (`street`, `normal`, `dry`) MAE=6.91, winner=VER -> ANT
- Barcelona Grand Prix (`permanent`, `normal`, `dry`) MAE=4.73, winner=VER -> HAM
- Azerbaijan Grand Prix (`street`, `normal`, `dry`) MAE=4.73, winner=LEC -> RUS

Drivers that show up repeatedly among the largest misses:
- VER: appearances=4, avg_abs_error=14.25
- PIA: appearances=4, avg_abs_error=11.75
- HUL: appearances=4, avg_abs_error=9.00

---

## 4. Systematic Bias

Signed error = predicted position - actual position.
Negative = model predicted *better* than reality (overestimated the driver).
Positive = model predicted *worse* than reality (underestimated the driver).

### Qualifying

Based on 16 races.

**Most overestimated teams:**
- Williams: mean signed error -2.12 (MAE 4.44, n=32)
- Red Bull Racing: mean signed error -1.58 (MAE 2.87, n=31)
- Haas F1 Team: mean signed error -1.34 (MAE 3.16, n=32)

**Most underestimated teams:**
- McLaren: mean signed error 2.12 (MAE 3.00, n=32)
- Audi: mean signed error 2.06 (MAE 3.44, n=32)
- Alpine: mean signed error 1.97 (MAE 3.28, n=32)

**Most overestimated drivers:**
- TSU: mean signed error -4.00 (MAE 4.00, n=2)
- VER: mean signed error -2.50 (MAE 3.38, n=16)
- SAI: mean signed error -2.50 (MAE 4.62, n=16)

**Most underestimated drivers:**
- COL: mean signed error 2.69 (MAE 3.69, n=16)
- PIA: mean signed error 2.44 (MAE 2.81, n=16)
- BOR: mean signed error 2.06 (MAE 3.31, n=16)

### Race

Based on 18 races.

**Most overestimated drivers:**
- VER: mean signed error -3.72 (MAE 4.72, n=18)
- LEC: mean signed error -2.56 (MAE 3.78, n=18)
- HAD: mean signed error -2.29 (MAE 3.29, n=14)

**Most underestimated drivers:**
- STR: mean signed error 2.67 (MAE 2.67, n=18)
- LAW: mean signed error 2.61 (MAE 4.17, n=18)
- OCO: mean signed error 2.56 (MAE 3.22, n=18)

---

## 5. Baseline Comparison

Naive baseline: predict race N using the actual results of race N-1
(previous-race classification). No model, just last week's result.

### Qualifying

Based on 15 races.

| Metric | Model | Naive baseline | Delta |
|---|---|---|---|
| MAE | 2.90 | 2.72 | -0.18 |
| Within-3 rate | 66.3% | 68.8% | - |
| Spearman rho | 0.80 | 0.83 | -0.02 |
| Kendall tau | 0.63 | 0.66 | -0.04 |

Failed Model does not beat naive baseline on MAE

### Race

Based on 17 races.

| Metric | Model | Naive baseline | Delta |
|---|---|---|---|
| MAE | 3.33 | 4.26 | 0.93 |
| Within-3 rate | 67.6% | 58.0% | - |
| Spearman rho | 0.70 | 0.54 | 0.16 |
| Kendall tau | 0.58 | 0.42 | 0.16 |

OK Model beats naive baseline on MAE

---

## Notes

- Calibration data populates only for predictions generated after
  p5/p95 interval persistence was added. Older artifacts carry no band data.
- Segment breakdowns slice the same event-level metrics by weekend format, weather,
  and track type using saved metadata and local track characteristics.
- Error analysis highlights the worst weekends and repeat offenders, not just mean scores.
- Systematic bias analysis requires >= 2 races with saved actuals.
- Baseline comparison requires >= 2 races (first race has no predecessor).
- For known model limitations see [LIMITATIONS.md](../LIMITATIONS.md).
