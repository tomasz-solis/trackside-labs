# Known limitations

What the model does badly, and what would fix it.

## 1. The weight schedule rests on one regulation change

The schedule that moves trust from the baseline to current results was studied on the 2021 to 2022 reset. That is one data point. The two best schedules differed by 0.002 Spearman (0.809 vs 0.807), and bootstrap resampling of 2025 swaps their order from draw to draw. Treat the choice as a sensible prior, not a proven optimum.

Fix: a second regulation change gives real out-of-sample evidence. Schedules can be swapped in `config/default.yaml` and compared with `scripts/backtest_2025_season.py`.

## 2. Pre-season team priors are guesses

`src/data/data_generator.py` seeds 2026 from the 2025 standings (for example McLaren 0.85, Mercedes 0.75). Those cars no longer exist after the reset, and the uncertainty is the same for every team. The baseline weight drops to 8% by race 3 and 5% from race 4, so races 1 and 2 carry the risk.

Fix: a prior on how much each team gets disrupted, built from past rule changes. Research idea, not built.

## 3. Testing sandbagging is not detected

Teams hide pace in testing with fuel, engine modes and setup. The model cannot tell slow from sandbagging. Testing gets the smallest weight (20% at race 1, 10% at race 2, 0% from race 3), so a misleading test corrects itself quickly.

Fix: correct testing laps for fuel load using telemetry. Real engineering work.

## 4. Compound effects need data first

Per-team compound adjustments at a circuit need at least 8 laps per compound per team. On the first visit to each circuit they are neutral, and the forecast relies on team strength alone.

Fix: a 2025 compound prior that fades as 2026 data arrives. The 2026 tyres differ, so the prior is weak at best.

## 5. Rookies and substitutes start conservative

A driver with no 2026 history falls back to the team baseline with extra uncertainty, set by experience tier. That avoids wild forecasts but underrates an exceptional rookie for a few weekends. A mid-season substitute starts from the same fallback unless someone edits the driver file by hand.

Fix: a career prior built from all past seasons, pulled automatically by `scripts/extract_driver_characteristics.py`.

## 6. Some constants were fitted on the season they predict

The walk-forward replay removes the leaks it can (team pace, seconds mapping, overtaking rates; see the 2026-09-25 entry in `docs/MODEL_LEDGER.md`). Constants fitted on the 2026 season are still read at every round: the retirement rate and its multiplier, and possibly `skill_improvement_max`. Replay accuracy is slightly optimistic because of them.

Fix: refit those constants per round inside the replay.

## 7. Safety car and lap 1 odds are curated, not learned

Each circuit has its own safety car probability and overtaking difficulty in `2026_track_characteristics.json`. They are hand-set priors. Lap 1 chaos is mostly global config.

Fix: extract per-circuit safety car and lap 1 incident rates from past races and test them against the current values in a backtest.

## 8. Intervals are too narrow

Each driver gets a p5 to p95 interval, and the learning system can widen it once enough residuals exist. The last calibration report (2026-04-20, 3 events, 66 intervals) showed 80.3% qualifying coverage against a 90% target.

Race p5/p95 has the same problem: measured 2026-09-27 on `data/historical_replay_m1s42_r14` (895 race finishers), coverage is 86% overall (92/95/89/67% by predicted-position bucket 1-5/6-10/11-16/17-22, the back of the field worst). An honest 90% band there needs about 10 to 11 positions, too wide to show as "the range a driver is likely to finish in". The dashboard's race table now shows a separate, narrower "Likely range" (a fitted 50% band, 3 to 5 positions wide, see `docs/MODEL_LEDGER.md` 2026-09-27) instead of p5/p95 for that purpose. p5/p95 itself is unchanged and still used for evaluation.

Fix: more races, then replay the widening before tightening anything.

## 9. Components can help alone and hurt together

A testing-based team seed and a residual model can encode the same thing, so stacking them overcorrects. Residual models are skipped by default when the seed is `testing_model`. The promotion gate requires lower MAE, no loss in winner or top 3 accuracy and no broad weekend damage.

Fix: rerun the ablations after each component change. Check which way residuals move drivers before tuning their clips.

## 10. Retirement risk has no measured skill yet

On 2026 (42 retirements in 264 driver-races, 0.201 each), a flat base rate scores better (Brier 0.16045) than the per-driver rates (0.17622). An earlier probe claimed the model overforecast; it had only 11 of the 42 retirements and is retired.

The shrinkage knob (`dnf_probability_shrinkage_lambda`) is 1.0 since 2026-09-20, so the reported number matches the simulation. The dashboard hides the column (`SHOW_DNF_RISK` in `src/dashboard/rendering_html.py`). Forecasts still store `dnf_probability` and the report still scores it.

The aggregate rate is roughly right: measured 2026-09-27 from `wf42_r14` PRE checkpoints against public classifications (repo DNF rule), predicted 4.04 vs actual 4.50 DNFs per race over 14 races (ratio 0.90). The per-lap draw (`p / race_distance` in `src/utils/lap_by_lap_simulator.py`) under-delivers that target rate by about 9% (15% at the 0.35 cap), not fixed yet. This is a count check, not a skill check, and does not change the Brier result above.

Separately, the per-driver rate's learning path (`_update_dnf_rate_ema` in `src/systems/updater.py`) had its own bug: a session with a NaN or missing `Status` and no `ClassifiedPosition` used to count as a DNF for every driver in it, permanently biasing `dnf_rate` toward 1.0 even though the race still counted as learned. Fixed 2026-09-27 to share the scoring rule (`ClassifiedPosition` authoritative, else `Status`; missing means no signal, not a DNF). See `docs/DNF_CALIBRATION_BRIEF.md`.

Fix: the planned DNF revamp. It has to beat the flat base rate on 2026, measured through the replay with the seed floor.
