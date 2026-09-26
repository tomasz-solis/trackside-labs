# Practice session blending

How the qualifying forecast uses practice data. Code: `src/utils/fp_blending.py`, called from `Baseline2026Predictor.predict_qualifying()`.

## The blend

```text
strength = w * session_strength + (1 - w) * model_strength
w = clip(base + (confidence - 0.5) * scale, min, max)
```

Better data raises `w`. Current values in `config/default.yaml` (`baseline_predictor.qualifying`): `fp_blend_weight` 0.62, `fp_blend_confidence_scale` 0.22, `fp_blend_weight_min` 0.40, `fp_blend_weight_max` 0.72.

## Sessions used

- Normal weekend: FP3, FP2, FP1 (FP3 weighted most)
- Sprint weekend: sprint qualifying, FP1, sprint

## Session strength

For each session:

1. Load laps.
2. Take each driver's representative short-run pace (push laps, tyre age aware).
3. Take the median per team.
4. Normalise across teams. The default (`fp_normalization: robust`) centres the field median at 0.5 and scales by an outlier-resistant spread, so one sandbagging team cannot set the scale. `minmax` is the old fastest 1.0, slowest 0.0.
5. Combine the sessions with fixed weights.

A driver needs enough clean laps to count. With no session data, qualifying uses the model alone. A team missing from the session data keeps its model strength.

## Scope

Used for qualifying only. The race forecast runs from the grid and the race simulation. The UI shows the source in `data_source` and `blend_used`.
