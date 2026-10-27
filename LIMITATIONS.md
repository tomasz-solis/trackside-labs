# Known limitations

What the model does badly, and what would fix it.

## 1. The weight schedule rests on one regulation change

The schedule that moves trust from the baseline to current results was studied on the 2021 to 2022 reset. That is one data point. The two best schedules differed by 0.002 Spearman (0.809 vs 0.807), and bootstrap resampling of 2025 swaps their order from draw to draw. Treat the choice as a sensible prior, not a proven optimum.

Fix: a second regulation change gives real out-of-sample evidence. Schedules can be swapped in `config/default.yaml` and compared with `scripts/backtest_2025_season.py`.

## 2. Pre-season team priors are guesses

`src/data/data_generator.py` seeds 2026 from the 2025 standings (for example McLaren 0.85, Mercedes 0.75). Those cars no longer exist after the reset, and the uncertainty is the same for every team. The baseline weight drops to 8% by race 3 and 5% from race 4, so races 1 and 2 carry the risk.

Fix: a prior on how much each team gets disrupted, built from past rule changes. Research idea, not built.

## 3. Testing sandbagging is not detected

Teams hide pace in testing with fuel, engine modes and setup. The model cannot tell slow from sandbagging. Track suitability from testing lost out of sample and was removed, so the testing weight (20% at race 1, 10% at race 2, 5% at race 3, 0% after) now carries the baseline. Testing still feeds the qualifying forecast when a weekend has no practice yet.

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

## 8. Honest bands are wide, so the dashboard shows a narrower one

Each driver gets a p5 to p95 band with a 90% target. Since model 3.5 the learning system widens race and sprint bands as it already did for qualifying. On `Z_base_34_r17_*`, the 17-round replay of 3.4, the widening takes Grand Prix coverage from 81% to 88% and sprint from 74% to 90% (48% to 89% once the sprint grid is known). An honest 90% race band is about 10 to 11 places, too wide to show as the range a driver is likely to finish in.

So the race and qualifying tables show a "Likely range" instead: the 25th to 75th percentile of past error around the shown position, as whole places that include it. It is about 4 to 7 places and held 59 to 63% of results leave-one-race-out (refitted 2026-10-27, `docs/MODEL_LEDGER.md`). The back of the Grand Prix field is the loose part: 7 places, holding 73%, because finishers there gain places when cars ahead retire. p5/p95 stays for evaluation.

Production has no sprint residuals yet: the sprint history began after the last 2026 sprint, so live sprint bands stay unwidened until 20 sprint residuals exist.

Fix: refit the likely range as races are added; give the back of the field its own treatment of retirements ahead.

## 9. Components can help alone and hurt together

A testing-based team seed and a residual model can encode the same thing, so stacking them overcorrects. Residual models are skipped by default when the seed is `testing_model`. The promotion gate requires lower MAE, no loss in winner or top 3 accuracy and no broad weekend damage.

Fix: rerun the ablations after each component change. Check which way residuals move drivers before tuning their clips.

## 10. Retirement risk has no measured skill yet

On 2026 (42 retirements in 264 driver-races, 0.201 each), a flat base rate scores better (Brier 0.16045) than the per-driver rates (0.17622). An earlier probe claimed the model overforecast; it had only 11 of the 42 retirements and is retired.

The shrinkage knob (`dnf_probability_shrinkage_lambda`) is 1.0 since 2026-09-20, so the reported number matches the simulation. The dashboard hides the column (`SHOW_DNF_RISK` in `src/dashboard/rendering_html.py`). Forecasts still store `dnf_probability` and the report still scores it.

The aggregate rate is roughly right: measured 2026-09-27 from `wf42_r14` PRE checkpoints against public classifications (repo DNF rule), predicted 4.04 vs actual 4.50 DNFs per race over 14 races (ratio 0.90). The per-lap draw (`p / race_distance` in `src/utils/lap_by_lap_simulator.py`) under-delivers that target rate by about 9% (15% at the 0.35 cap), not fixed yet. This is a count check, not a skill check, and does not change the Brier result above.

Separately, the per-driver rate's learning path (`_update_dnf_rate_ema` in `src/systems/updater.py`) had its own bug: a session with a NaN or missing `Status` and no `ClassifiedPosition` used to count as a DNF for every driver in it, permanently biasing `dnf_rate` toward 1.0 even though the race still counted as learned. Fixed 2026-09-27 to share the scoring rule (`ClassifiedPosition` authoritative, else `Status`; missing means no signal, not a DNF). See `docs/DNF_CALIBRATION_BRIEF.md`.

Fix: the planned DNF revamp. It has to beat the flat base rate on 2026, measured through the replay with the seed floor.

## 11. The race forecast is mostly the grid

The predicted grid predicts race order better than simulated pace. Halving the grid anchor cost 0.016 to 0.019 race correlation on both seeds. Three race pace additions with real lap-time signal (car traits by track, carried tyre wear, weekend practice) did not move finishing order past the seed floor. Qualifying still leaves practice on the table: its errors correlate +0.20 with the weekend practice signal.

Fix: fit a practice term on the model's own residuals. The version fitted on raw gaps double counted practice and lost (`docs/MODEL_LEDGER.md`, 2026-10-04).
