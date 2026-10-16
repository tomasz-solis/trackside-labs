# Release evidence template

Copy this template for a proposed release. Missing evidence means `NOT EXECUTED` or `[Unknown]`, never PASS. This template grants no deployment approval.

| Field | Evidence |
|---|---|
| Objective and acceptance criteria | |
| Commit, diff and dependency lock hash | |
| Model version, artifact ID and content hash | |
| Config, source snapshot, calibration and seed | |
| Training cutoff, information cutoff and publication proof | |
| Target, checkpoint, scenario and outcome revision | |
| Cohort: prospective, replay or counterfactual | |
| Baseline, matched independent weekends and exclusions | |
| Scores, paired uncertainty and Monte Carlo precision | |
| DS producer and UI consumer contract tests | |
| Commands, environment, counts and exit codes | |
| Live CI, serving, browser and accessibility evidence | |
| Independent reviewer verdict and blocking findings | |
| Known limitations, runtime and operating cost | |
| Rollout flag, rollback command and approval | |

Keep forecast values immutable. Attach outcomes and evaluations separately. Do not treat a cache timestamp as publication proof or a replay as independent future evidence.
