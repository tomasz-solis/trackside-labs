# Configuration Guide

This project uses two main config files for different purposes:

- `config/default.yaml`: model and runtime parameters.
- `config/production_config.json`: strategy metadata used by helper utilities and experiments.

## 1. `config/default.yaml`

### What is actively used by the dashboard runtime path

`Baseline2026Predictor` reads values through `src/utils/config_loader.py`.
The most relevant active section is:

- `baseline_predictor.qualifying.*`
- `baseline_predictor.race.*`

Examples:

- `baseline_predictor.qualifying.noise_std_normal`
- `baseline_predictor.qualifying.team_weight`
- `baseline_predictor.qualifying.testing_short_run_modifier_scale`
- `baseline_predictor.race.base_chaos.dry`
- `baseline_predictor.race.grid_weight_min`
- `baseline_predictor.race.pace_weight_base`
- `baseline_predictor.race.dnf_rate_final_cap`
- `baseline_predictor.race.testing_long_run_modifier_scale`
- `baseline_predictor.practice_capture.*` (dashboard FP auto-capture behavior)

### Other sections in `default.yaml`

Sections such as `bayesian`, `race`, `qualifying`, and `learning` are still useful for other modules/scripts, but they are not the primary scoring knobs for the baseline predictor race/qualifying simulation path.

## 2. `config/production_config.json`

Used by `src/utils/config.py` (`ProductionConfig`) for strategy metadata and expected MAE references.

It is not the main parameter source for `Baseline2026Predictor` scoring in the dashboard path.

## Common Changes

### Change qualifying team vs driver weighting

Edit:

- `config/default.yaml` -> `baseline_predictor.qualifying.team_weight`
- `config/default.yaml` -> `baseline_predictor.qualifying.skill_weight`

These should sum to `1.0`.

### Change race volatility / chaos

Edit:

- `baseline_predictor.race.base_chaos.dry`
- `baseline_predictor.race.base_chaos.wet`
- `baseline_predictor.race.lap1_chaos.*`
- `baseline_predictor.race.strategy_variance_base`

### Change DNF behavior

Edit:

- `baseline_predictor.race.dnf_rate_historical_cap`
- `baseline_predictor.race.dnf_rate_final_cap`

### Change FastF1/cache paths

Edit:

- `paths.*` in `config/default.yaml`
- env vars (for example `F1_CONFIG`, `F1_DATA_DIR`, `F1_CACHE_DIR`) where supported by the relevant modules

## Validation Rules

`src/utils/config_loader.py` validates:

- required sections exist,
- critical values are present and in range,
- qualifying weights sum to ~1.0.

If validation fails, startup will raise an explicit error.

## Example: Read Config in Code

```python
from src.utils import config_loader

team_weight = config_loader.get("baseline_predictor.qualifying.team_weight", 0.7)
pace_weight = config_loader.get("baseline_predictor.race.pace_weight_base", 0.40)
```

## Safe Workflow For Config Changes

1. Edit `config/default.yaml`.
2. Run targeted tests:
   - `pytest tests/test_baseline_2026_integration.py`
   - `pytest tests/test_dashboard_smoke.py`
3. Run a dry prediction in the app/CLI and confirm behavior.

## Notes

- The baseline predictor currently uses a fixed 70/30 practice blend inside predictor logic for qualifying.
- `src/predictors/qualifying.py` and `src/predictors/race.py` preserve legacy method signatures and delegate to baseline logic.
