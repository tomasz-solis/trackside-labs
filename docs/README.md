# Technical Documentation Index

Deep-dive guides for implementing and understanding the F1 2026 prediction system.

## Core Guides

### [WEIGHT_SCHEDULE_GUIDE.md](WEIGHT_SCHEDULE_GUIDE.md)
**What:** Dynamic blending of baseline, testing, and current season data

**When to read:** Understanding how the system adapts during regulation changes

**Key concepts:**
- Three-component blending (baseline 30% → 5%, testing 20% → 0%, current 50% → 95%)
- Validated 0.809 correlation on 2021→2022 transition
- Why extreme schedule outperforms conservative approaches

**Code:** [src/systems/weight_schedule.py](../src/systems/weight_schedule.py)

---

### [FP_BLENDING_SYSTEM.md](FP_BLENDING_SYSTEM.md)
**What:** Combining practice session lap times with model predictions

**When to read:** Understanding how predictions improve during race weekends

**Key concepts:**
- 70% FP data + 30% model = 21% accuracy improvement
- Session priority: FP3 > FP2 > FP1 (normal), Sprint Race > Sprint Quali > FP1 (sprint)
- Auto-fetch behavior during predictions

**Code:** [src/utils/fp_blending.py](../src/utils/fp_blending.py)

---

### [DASHBOARD_AUTO_UPDATE.md](DASHBOARD_AUTO_UPDATE.md)
**What:** What updates automatically vs. manually in the system

**When to read:** Understanding data refresh behavior and post-race workflows

**Key concepts:**
- FP data: Auto-fetches when user clicks "Generate Predictions"
- Team characteristics: Manual update via `scripts/update_from_race.py`
- Cache behavior and invalidation

**Code:** [app.py](../app.py), [scripts/update_from_race.py](../scripts/update_from_race.py)

---

## Validation Notebooks

Results-driven analysis proving system effectiveness:

### [../notebooks/validate_testing_predictions.ipynb](../notebooks/validate_testing_predictions.ipynb)
**Analysis:** 2021→2022 regulation change correlation study

**Key finding:** Pre-season testing correlation drops from 0.422 (stable years) to 0.137 (regulation changes)

**Conclusion:** Trust testing less during major regulation changes

---

### [../notebooks/test_weight_schedules.ipynb](../notebooks/test_weight_schedules.ipynb)
**Analysis:** 7 weight schedules tested on 2022 season data

**Result:** Extreme schedule (50%→95% current) achieved 0.809 correlation vs 0.512 for conservative

**Conclusion:** Aggressive adaptation works best for regulation changes

---

### [../notebooks/sensitivity_analysis.ipynb](../notebooks/sensitivity_analysis.ipynb)
**Analysis:** Parameter tuning for FP blend weights and weight schedule aggressiveness

**Result:** 70% FP blend + extreme schedule = optimal configuration

**Usage:** Template ready - run after races 3-5 to validate 2026 parameters

---

### [../notebooks/validation_metrics.ipynb](../notebooks/validation_metrics.ipynb)
**Analysis:** Prediction accuracy tracking (MAE, RMSE, Top-N accuracy)

**Targets:** MAE < 2.5, RMSE < 3.5, Top-3 > 60%, Top-10 > 70%

**Usage:** Template ready - populate with actual 2026 predictions vs results

---

## Architecture Overview

For high-level system design, see:
- [../ARCHITECTURE.md](../ARCHITECTURE.md) - Data flow and component structure
- [../CONFIGURATION.md](../CONFIGURATION.md) - How to configure the system
- [../README.md](../README.md) - Project overview and quick start

---

## Reading Path by Role

### Data Scientist / Analyst
1. [WEIGHT_SCHEDULE_GUIDE.md](WEIGHT_SCHEDULE_GUIDE.md) - Understand the core algorithm
2. [../notebooks/test_weight_schedules.ipynb](../notebooks/test_weight_schedules.ipynb) - See validation
3. [FP_BLENDING_SYSTEM.md](FP_BLENDING_SYSTEM.md) - Learn the enhancement layer
4. [../notebooks/sensitivity_analysis.ipynb](../notebooks/sensitivity_analysis.ipynb) - Parameter tuning

### Software Engineer
1. [../ARCHITECTURE.md](../ARCHITECTURE.md) - System design
2. [DASHBOARD_AUTO_UPDATE.md](DASHBOARD_AUTO_UPDATE.md) - Runtime behavior
3. [../CONFIGURATION.md](../CONFIGURATION.md) - Configuration system
4. [FP_BLENDING_SYSTEM.md](FP_BLENDING_SYSTEM.md) - Feature implementation

### F1 Fantasy User
1. [../README.md](../README.md) - Quick start
2. [DASHBOARD_AUTO_UPDATE.md](DASHBOARD_AUTO_UPDATE.md) - How to use dashboard
3. [WEIGHT_SCHEDULE_GUIDE.md](WEIGHT_SCHEDULE_GUIDE.md) - Why predictions change over season

---

## Quick Reference

| Topic | File | Section |
|-------|------|---------|
| Weight schedules | WEIGHT_SCHEDULE_GUIDE.md | Weight Schedule Progression |
| FP session priority | FP_BLENDING_SYSTEM.md | How It Works |
| Auto vs manual updates | DASHBOARD_AUTO_UPDATE.md | What Auto-Updates |
| Post-race workflow | DASHBOARD_AUTO_UPDATE.md | Post-Race (Monday) |
| Sprint weekend flow | FP_BLENDING_SYSTEM.md | Sprint Weekend Handling |
| 2022 validation | validate_testing_predictions.ipynb | Historical Analysis |
| Parameter targets | validation_metrics.ipynb | Metrics & Targets |

---

## Contributing to Docs

When adding new technical guides:

1. **Use concrete examples** - Code snippets, not just prose
2. **Include validation** - Show numbers (correlations, MAE, etc.)
3. **Link to code** - Reference actual implementation files
4. **Link to notebooks** - Show analysis that proves it works
5. **No AI fingerprints** - Keep language concise and professional

See existing guides as templates.
