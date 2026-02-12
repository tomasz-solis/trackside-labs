# Prediction Tracking

This system stores predictions by session and lets you attach actual results later for accuracy analysis.

## Where It Lives

- Logger: `src/utils/prediction_logger.py`
- Session detection: `src/utils/session_detector.py`
- Metrics: `src/utils/prediction_metrics.py`
- Update script: `scripts/update_prediction_actuals.py`
- Dashboard usage: `app.py` (Prediction toggle + Accuracy page)

## How Saving Works

1. Enable **Save Predictions for Accuracy Tracking** in the sidebar.
2. Click **Generate Prediction**.
3. App detects the latest completed session for the weekend type.
4. If no prediction exists yet for that race/session key, it writes one JSON file.

File root:

- `data/predictions/<year>/<race_slug>/`

Example file names:

- `bahrain_grand_prix_fp1.json`
- `bahrain_grand_prix_fp2.json`
- `bahrain_grand_prix_fp3.json`
- `chinese_grand_prix_sq.json`
- `chinese_grand_prix_sprint.json`

## Session Detection Rules

Normal weekend tracked sessions:

- `FP1`, `FP2`, `FP3`

Sprint weekend tracked sessions:

- `FP1`, `SQ`, `Sprint`

The detector uses scheduled session time plus buffer duration to decide whether a session is completed.

## Saved Payload (Current Shape)

Each file contains:

- `metadata` (year, race, session, timestamp, weather, optional blend info)
- `qualifying.predicted_grid`
- `race.predicted_results`
- `actuals` placeholder (`qualifying`, `race`)

## Add Actual Results Later

```bash
python scripts/update_prediction_actuals.py "Bahrain Grand Prix" FP1 --year 2026
```

This script fetches `Q` and `R` results from FastF1 and writes them into the matching prediction file.

## Accuracy View

In the dashboard **Prediction Accuracy** page, metrics are computed only for predictions that already include actual results.

Typical metrics shown:

- exact accuracy,
- MAE,
- within-band accuracy,
- correlation,
- winner/podium checks.

## Known Limits

1. Session labels must match saved filenames (`FP1`, `FP2`, `FP3`, `SQ`, `Sprint`).
2. Actuals update depends on FastF1 availability for qualifying and race sessions.
3. If no session has completed yet, nothing is saved.
