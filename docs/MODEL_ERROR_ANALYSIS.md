# Model error analysis, 2026

*Generated: 2026-10-10T21:17:28.465284+00:00*

The worst weekends and the drivers the model misses most.

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

## Context

- Pair this with [MODEL_CALIBRATION.md](./MODEL_CALIBRATION.md) for calibration and baseline metrics.
- Pair this with [LIMITATIONS.md](../LIMITATIONS.md) for known structural gaps.
