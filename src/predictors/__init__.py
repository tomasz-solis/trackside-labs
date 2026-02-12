"""
F1 2026 Predictors

ACTIVE PREDICTOR:
- baseline_2026.py - Primary predictor for 2026 season (used by dashboard)

LEGACY COMPATIBILITY WRAPPERS:
- qualifying.py - Wrapper that delegates to baseline_2026 predictor
- race.py - Wrapper that delegates to baseline_2026 predictor

ARCHIVED PREDICTORS (moved to archive/predictors/):
- team.py, driver.py, blended.py, qualifying.py, tire.py, race.py
- These were used by legacy scripts (simulator.py, predict_weekend.py)
- Archived variants are NOT used by main dashboard (app.py)
- Kept in archive for reference but not actively maintained
"""
