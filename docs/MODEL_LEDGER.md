# Model ledger

What was tried, how it was measured, and whether it helped. `MODEL_PROMOTION.md` defines the gates; this file records what went through them.

Add new entries at the bottom. Never change a past verdict: when a later result contradicts one, mark it superseded and point to the new entry.

## Verdicts

| Verdict | Meaning |
|---|---|
| `adopted` | Measured better, now in the champion |
| `worse` | Measured, lost |
| `noise` | Measured, the effect straddles zero |
| `unresolvable` | Smaller than the seed floor: nothing was learned |
| `never activated` | Ran, but a guard made it identical to the champion: untested, not neutral |
| `refused` | Could not produce a scored result |
| `open` | Not measured yet |

`never activated` is the one that misleads: identical numbers look harmless but mean nobody tested the change. `unresolvable` is not `noise`: noise straddles zero, unresolvable means the comparison could not detect the effect at all.

## Baselines

Every result only compares to the champion it was measured against. On 2026-07-28 one bug fix moved qualifying MAE by 0.70 positions, about ten times the largest challenger effect recorded here. So every entry names its baseline, and old verdicts go stale when the champion changes.

## Measurement protocol

Rebuild the season from pre-season, change one thing, score on 2026 only (2022 to 2025 seasons cannot judge a 2026 calibration), and before adopting, check whether scaling an existing constant reproduces the gain.

```bash
# baseline and candidate, one variable apart, same round window
uv run python scripts/replay_historical_checkpoints.py --year 2026 --through-round 14 --output-root data/historical_replay_baseline
uv run python scripts/replay_historical_checkpoints.py --year 2026 --through-round 14 --output-root data/historical_replay_candidate

# seed floor: same code, second seed
uv run python scripts/replay_historical_checkpoints.py --year 2026 --through-round 14 --seed 43 --output-root data/historical_replay_seed43

# compare, gated on the floor
uv run python scripts/compare_replay_arms.py --baseline data/historical_replay_baseline --seed-floor data/historical_replay_baseline data/historical_replay_seed43 --candidate data/historical_replay_candidate
```

A 14-round replay takes 30 minutes to 3 hours depending on machine load. Run it in your own terminal; Claude Code's low-memory reaper has killed it as a background job.

**Correlation is the primary metric, MAE secondary.** Resolving power over 46 checkpoints (95% CI half-width on a paired delta divided by the metric's spread, lower is finer):

| Metric | Detectable / spread |
|---|---|
| **correlation** | **0.030** |
| overall_mae | 0.063 |
| top_3_pct | 0.076 |
| within_3 | 0.093 |
| top_10_pct | 0.160 |
| within_1 | 0.172 |
| exact_accuracy | 0.266 |

Correlation resolves about twice as finely as MAE and barely moves with the seed. MAE rounds to whole positions: in one comparison 7 of 46 checkpoints had different forecasts and identical MAE.

**Always pass `--seed-floor`.** Without it the script refuses to call anything `unresolvable`. The threshold is the widest bound of the seed pair's confidence interval, not its point estimate.

`compare_replay_arms.py` verdicts: `better` / `worse` (CI excludes zero and clears the floor), `noise` (CI includes zero), `unresolvable (below seed floor)`, `identical (never activated)` (every checkpoint tied, so the change provably did nothing, which is a finding).

**Current floor: 2026-09-25**, leak-free replay, baseline `data/historical_replay_wf42_r14`, `--through-round 14`:

| Target | Correlation | MAE |
|---|---|---|
| Qualifying | 0.0035 | 0.054 |
| Race | 0.0110 | 0.058 |
| Sprint race | 0.0078 | 0.079 |

Older floors (2026-09-12, 13 rounds; 2026-09-21, 14 rounds) were measured on a replay that leaked future inputs. Keep them only for reading the entries gated on them.

**Before 2026-09-12** entries used `scripts/run_challenger_research_walk_forward.py`: 9 events from `data/historical_replay/2026/event_catalog.json` (7 scored, two wet rounds excluded), checkpoint PRE, 3 seeds, 20 simulations. Champion-only entries used `predict_qualifying(..., practice_signal_mode="stored_profiles")` over the 9 events, after moving the prediction cache aside because its key ignored code version. That harness no longer exists.

## Champion history

| Date | Change | Effect | Commit |
|---|---|---|---|
| 2026-07-28 | Centre `quali_rating_mu_s` within each team when building the qualifying field. It carried a team component on top of team strength, so car pace counted twice. | Qualifying MAE 3.525 -> 2.828; mean per-driver \|bias\| 2.889 -> 1.677. HUL +6.11 -> +1.67, ALB -5.78 -> +0.11, GAS +4.33 -> +0.33 | `93bfbeb0` |
| 2026-08-04 | Refit `team_strength_seconds_mapping` on 2026 only. The 2022 to 2025 fit had never seen a 2026 lap and compressed team gaps all season. | Qualifying MAE 2.6599 -> 2.5724, \|bias\| 1.5017 -> 1.2997; race MAE 4.0606 -> 3.9192, \|bias\| 2.4242 -> 2.3434 (60 simulations). Slopes: qualifying 1.77417 -> 2.76281, race 1.97077 -> 3.89727 | `fdf7be6f` |

Residual after that, same measurement: SAI -5.67, LAW +6.44, BOR +4.44, ALO -4.11, VER -4.11. Team-strength errors (Williams over-rated, RB under-rated), which the centring fix does not touch.

## Challengers tested (2026-07-19/20)

Measured against the pre-centring champion and never re-baselined (see Blocked).

| Variant | Idea | Verdict | Evidence |
|---|---|---|---|
| `q0_driver_state` | Richer driver state for qualifying | worse | Qualifying MAE +0.19, 1 better / 5 worse |
| `q0_driver_state__baseline500` | Same, 500-simulation baseline | worse | +0.24, 2 / 13 |
| `q0_driver_state__fp_hisim2` | Same, more practice simulations | worse | +0.23, 2 / 7 |
| `q0_driver_state__pullcap025` | Same, pull cap 0.25 | worse | +0.24, 2 / 13 |
| `q0_driver_state__pullcap035` | Same, pull cap 0.35 | worse | +0.24, 2 / 13 |
| `r1_joint_grid` | Sample the race grid from the joint qualifying distribution | noise | Race MAE -0.074, but 4 better / 3 worse over 7 events |
| `r1_joint_grid__fp_hisim2` | Same, more simulations | noise | 5 / 4, mean 0.000 |
| `q1_qualifying_practice` | Practice one-lap pace to qualifying | never activated | `refused`: needs 4 prior dry same-class events, had 3; the retro run was identical and disclosed `no_raw_practice_laps` |
| `r0_long_run` | Practice long-run pace to race pace | never activated | 42 identity flags, `missing_race_practice_evidence` |
| `r0_long_run__fp_hisim2` | Same | never activated | 60 identity flags, `insufficient_field_evidence_coverage` |
| `r2_no_anchor`, `r1_r2_no_anchor` | Grid anchor variants | refused | A position fell outside its own p5 to p95 |
| `r2_source_anchor`, `r1_r2_source_anchor` | Grid anchor variants | refused | No eligible scored events |

309 champion vs challenger metric pairs, 179 identical, 130 different. Nothing beat the champion. Q0 lost under all four tunings.

Q1 and R0 match the core idea (one-lap pace for qualifying, long-run pace for the race) and neither has ever run: the replay feeds `stored_profiles`, so `session_laps_by_type = {}` and both fall back to the champion. So the honest summary is: the grid variants were tested and lost; the practice variants were never tested. `docs/RAW_LAPS_REPLAY_HANDOFF.md` is the fix.

## Blocked

Challenger work was shelved on 2026-07-29 and lives on the branch `shelved/challenger-research` (since 2026-07-31): `scripts/run_challenger_research_walk_forward.py`, `src/analysis/challenger_*`, `src/models/qualifying_practice_*`, their tests and the reports under `data/model_diagnostics/2026/race_mae_investigation/`. Nothing on `master` imports them (checked across 409 tracked Python files). The method docs stayed on `master`: `docs/QUALIFYING_RACE_CHALLENGER.md` and `docs/RAW_LAPS_REPLAY_HANDOFF.md`. The 27,340 lines of generated comparison JSON were not kept.

The walk-forward artifacts in `data/historical_replay/2026/` (909 MB, gitignored) are not on the branch. They exist only on local disk and back every number in the challenger table. Keep them.

Re-running the challengers on the fixed champion was stopped. The modules target production code that no longer exists:

| Gap | Outcome |
|---|---|
| `predict_qualifying(include_grid_scenarios=)` | Built, then reverted with the shelving |
| `predict_qualifying(include_challenger_evidence=)`, `q1_retrospective_diagnostic=` | Q1 only, cannot work in this harness |
| `QualifyingGridEntry.start_type` dropped by `validate_qualifying_grid` | Real bug, fixed in `3810c1ad` |
| `predict_race(grid_scenarios=)` | Stopped here |

The last one would mean inventing how scenarios map to simulation draws, how the marginal path stays seed-comparable and how the anchor is chosen, with no surviving source. A wrong reconstruction produces numbers that look valid.

## Open, worth testing

Ranked by expected value.

1. **Raw-laps replay** (`docs/RAW_LAPS_REPLAY_HANDOFF.md`). The only way to test Q1 and R0.
2. **Team-strength residual.** The largest known champion error. SAI -5.67 and LAW +6.44 are team-level, and `overall_performance` ranks Williams and Audi correctly while the blended strength does not.
3. **Centre driver seconds at fit time.** `_update_pair_constraint` only applies difference constraints, so the per-team level is unidentified and will drift again at the next seeding. Centre inside `attach_driver_rating_mus` before `team_target_s`, then refit the mapping. (The 2026-08-03 attempt at this lost; read it first.)
4. **Combinations.** Every variant was tested alone. No two changes have been tested together.

## 2026-07-30: learning-path fixes, measured separately

Baseline `3810c1ad`, rebuilt from the 2026-04-25 pre-season driver artifact (`710fb551`) with seconds re-seeded and all 11 rounds replayed: **qualifying MAE 2.6195, mean per-driver \|bias\| 1.5522**, 9 events x 3 seeds (594 driver-events).

Rebuilding matters. The same code scored against the stored 6-round artifact gives 2.8788, because practice capture had been resetting the season history every Friday. Comparing against that stale artifact credits the fixes with 0.164 MAE they did not earn.

| Variant | MAE | \|bias\| | Verdict |
|---|---|---|---|
| Database-first read + recency-weighted season mean + margin-scored fallback | **2.5993** | **1.4747** | `adopted` |
| Plus skipping unpaired drivers in the Bayesian update | 2.8148 | 1.6094 | `worse` |
| Same, with learn-time recency neutralised | 2.8013 | 1.6128 | `worse` |

As one package it looked like a 0.064 gain over the stale artifact and was really a 0.195 loss against a fair baseline. Reverting only the Bayesian change recovered 0.215.

Why skipping lost: `update_teammate_relative` gives a driver whose teammate retired a raw 1 to 22 rating, mixing that scale into a model centred on the field mean (32 such cases, one at the maximum 22.00). Dropping them costs more than the contamination. Next attempt should rescale them. Do not retest discarding.

Same day: turning off the `rating_mu` blend (`bayesian_quali_skill_blend_cap: 0.0`) scored 3.0505 against 2.8788. `rating_mu` correlates only -0.068 with qualifying position, but what it falls back to is worse.

### Bayesian update confidence rebalance: `worse`

`rating_mu` is one rating updated by race and qualifying, and it feeds the qualifying skill blend. Race observations carry confidence 0.35 and qualifying 0.15, so qualifying skill is shaped more by race results. It fits the 2026 pattern: backmarkers finish far better than they qualify (ALO -4.09, STR -4.00, PER -4.16), front-runners worse (ANT +2.78, RUS +0.94).

| Variant | MAE | \|bias\| | Verdict |
|---|---|---|---|
| Champion, quali 0.15 / race 0.35 | **2.5993** | **1.4747** | `adopted` |
| 0.35 / 0.35 | 2.6970 | 1.5354 | `worse` |
| 0.35 / 0.15 | 2.7811 | 1.6667 | `worse` |

Both directions lost. HUL's bias stayed between +4.5 and +5.3 in every arm. Do not retest without a new mechanism.

### Margin-scored telemetry race pace: `worse`

`extract_team_performance_from_telemetry` computes each team's median race lap, then throws it away for `1 - rank/(teams-1)`. This variant kept the margin (gap to the field median as a fraction of lap time, mapped to 0 to 1 by a spread setting).

| Variant | MAE | \|bias\| | Verdict |
|---|---|---|---|
| Champion, rank | **2.5993** | **1.4747** | `adopted` |
| Margin, spread 0.06 | 2.6700 | 1.7138 | `worse` |
| Margin, spread 0.10 | 2.7744 | 1.8586 | `worse` |

It worked as designed (Aston Martin's flat `[0.1, 0.0, ...]` became a visible climb to 0.447) and still lost, monotonically: the closer to rank, the better. The input is the problem. A race median lap carries strategy, traffic, fuel and safety cars; the team spread was 4.6 to 4.8 s, more than twice what the seconds mapping was fitted for. Rank is robust to that; margin passes it through. Do not retry margin scoring on race medians. The version worth testing converts the matched-lap construct the mapping was fitted on.

### Open after this entry

- The telemetry path still discards margin for most races; the 2026-07-30 fix only reached races without telemetry.
- HUL degrades as the season is learned (+1.52 at 6 rounds, +5.26 at 11); RUS +0.15 -> +2.19. **Superseded 2026-07-31:** not a learning-path problem, and HUL and RUS are not one problem.

## 2026-07-31: the HUL/RUS drift is not a driver-rating problem

No prediction run; everything comes from the shipped artifacts (`car_characteristics` v23 at 11 rounds, `driver_characteristics` v26, the seconds mapping), the 9-event catalog and code at `6771a8a5`. Written to stop runs on a path arithmetic already closes.

**The driver rating cannot produce this error.** HUL's centred `quali_rating_mu_s` is -0.1225 s. Qualifying projects it as `0.5 + delta / 1.9708`, so his driver term is 0.062 score units, at about 0.045 units per grid position: **about 1.4 positions of authority against a +5.26 bias.** All four failed hypotheses pulled that lever. Show a path has enough authority before testing another driver-rating variant.

**Both teammates share the sign, so it is a team offset.** HUL +5.26 / BOR +4.44, RUS +2.19 / ANT +2.78. "The HUL/RUS bias" is an Audi and Mercedes team-strength bias.

**HUL and RUS differ.** Model teammate order (centred rating) against actual head to head over the 9 events:

| Team | Model rates faster | Actual | Agrees |
|---|---|---|---|
| Audi | BOR (+0.1225) | HUL 5-4 | no |
| Red Bull | HAD (+0.0079) | VER 6-3 | no |
| RB | LIN (+0.0955) | LAW 6-3 | no |
| Cadillac | BOT (+0.1440) | PER 6-3 | no |
| Mercedes | ANT (+0.1003) | ANT 5-4 | yes |
| McLaren | NOR (+0.0878) | NOR 5-4 | yes |
| Ferrari | LEC (+0.0099) | LEC 5-4 | yes |
| Aston Martin | ALO (+0.3537) | ALO 7-2 | yes |
| Alpine | GAS (+0.0221) | GAS 6-3 | yes |
| Haas | BEA (+0.1161) | BEA 7-2 | yes |
| Williams | SAI (+0.0442) | SAI 7-2 | yes |

Four of eleven pairs are backwards, HUL among them. RUS is not: Antonelli really out-qualified him, so RUS's error comes from Mercedes team strength alone. Sign convention checked: positive seconds means faster.

**Observation count does not shrink teammate gaps.** Audi at 11 observations has the largest gap among well-observed teams (0.245 s), Red Bull at 12 the smallest (0.016 s), Aston Martin's 0.708 s sits on 3. Do not test "more learning shrinks gaps".

**Separate bug: qualifying used the race slope.** `team_strength_seconds_score_scale: 1.9707717329051126` is the race slope, applied in qualifying where the fitted slope is 1.7741686893278807, compressing qualifying by about 11%. Measure alone. (Fixed 2026-08-05.)

### Team vs driver decomposition

`identify_systematic_errors` already computes `team_bias` next to `driver_bias`; everything before read only the driver column. On cached champion predictions (`3f07ca70`, written 2026-07-19 to 22, before the centring fix `93bfbeb0`, so levels are stale), PRE qualifying, 7 events x 3 seeds, 462 observations, positive = too pessimistic:

| Team | Team bias | Driver 1 | Driver 2 | Spread |
|---|---|---|---|---|
| Audi | +6.71 | HUL +6.86 | BOR +6.57 | 0.29 |
| Williams | -7.10 | SAI -7.86 | ALB -6.33 | 1.53 |
| RB | +3.33 | LAW +2.48 | LIN +4.19 | 1.71 |
| McLaren | +1.74 | NOR +0.90 | PIA +2.57 | 1.67 |
| Mercedes | +1.14 | RUS +0.14 | ANT +2.14 | 2.00 |
| Alpine | +1.12 | GAS +1.48 | COL +0.76 | 0.72 |
| Cadillac | +0.05 | PER +1.29 | BOT -1.19 | 2.48 |
| Haas | -0.81 | OCO +0.62 | BEA -2.24 | 2.86 |
| Ferrari | -0.90 | LEC -0.76 | HAM -1.05 | 0.29 |
| Red Bull | -2.43 | VER -5.52 | HAD +0.67 | 6.19 |
| Aston Martin | -2.86 | ALO -5.52 | STR -0.19 | 5.33 |

**Correction, same day.** Audi +6.71 and Williams -7.10 are the uncentred driver rating that `93bfbeb0` already fixed, not open problems. The raw team-mean rating (Williams +0.412 s, Audi -0.387 s) alone implies +4.60 and -4.32 positions, an 8.9-position spread against an observed 9.67, on team strengths that are nearly equal (0.366 vs 0.354) and correctly ranked.

**The spread column is separation error, not driver-term size.** Bias is predicted minus actual, so the teammate spread is predicted gap minus actual gap.

**The clip binds at the front.** Score is `clip(0.5 + 1.7742*(strength-0.5)/1.9708 + mu/1.9708, 0, 1)`. Before the centring fix, ANT (1.136), LEC (1.060), HAM (1.050) and RUS (1.035) all clipped to 1.0, so their order was noise, and STR clipped to 0. After it, only STR. Any pre-fix result on the internal order of Mercedes or Ferrari reads noise. The clip does not explain Audi.

**Current order, and the one that is wrong** (current strengths and ratings, against head to head): Audi BOR 13th, HUL 18th while HUL leads 5-4, inverted by five places. Red Bull HAD 7th, VER 8th while VER leads 6-3. Mercedes, Williams and Aston Martin are correct. HUL/BOR is the largest live driver-level error, a sign error. Score order is not simulated position, so treat the sign as the finding.

### Mechanism: the rookie prior sigma sets the update gain

No sign bug (checked end to end: `matched_gap_s` positive means reference faster, positive innovation raises the reference). The prior had the right order; 2026 flipped it:

| Pair | Prior mu (a / b) | Prior sigma | Variance ratio | Now | Flipped |
|---|---|---|---|---|---|
| VER / HAD | +0.447 / -0.145 | 0.152 / 0.530 | 12.2 | +0.299 / +0.315 | yes |
| HUL / BOR | -0.461 / -0.554 | 0.231 / 0.530 | 5.3 | -0.509 / -0.264 | yes |
| RUS / ANT | +0.453 / +0.232 | 0.277 / 0.530 | 3.7 | +0.226 / +0.426 | yes |
| ALO / STR | +0.143 / -0.111 | 0.152 / 0.155 | 1.1 | +0.364 / -0.343 | no |
| LEC / HAM | +0.498 / +0.395 | 0.273 / 0.275 | 1.0 | +0.456 / +0.437 | no |
| SAI / ALB | +0.421 / +0.405 | 0.272 / 0.260 | 0.9 | +0.456 / +0.368 | no |

Every pair with a variance ratio above about 3 reordered; every pair near 1 held. The update splits each innovation by variance, so the wide-prior driver absorbs it (BOR +0.289 vs HUL -0.048; HAD +0.460 vs VER -0.148). BOR, HAD and ANT share sigma 0.5304206197376727 exactly, a default for low-observation drivers. Cadillac (ratio 12.2) did not flip because the prior already had BOT ahead. Correct Bayesian behaviour given those priors, and not always wrong (ANT does lead RUS). Most qualifying sessions give exactly 3 matched pairs, and 11 of 21 BOR/HUL sessions in 2025 gave none.

### Measured on current code

A champion-only scorer (production `predict_qualifying(..., practice_signal_mode="stored_profiles")` plus `identify_systematic_errors`), 7 dry events x 3 seeds x 20 simulations. Not a walk-forward: each event is predicted with its own result already in the artifacts, so levels are optimistic. MAE 2.6494, \|bias\| 1.7186.

| Team | Team bias | Driver 1 | Driver 2 | Spread |
|---|---|---|---|---|
| Audi | +2.26 | HUL +4.62 | BOR -0.10 | 4.71 |
| RB | +1.45 | LAW +3.38 | LIN -0.48 | 3.86 |
| Mercedes | +1.29 | RUS +3.14 | ANT -0.57 | 3.71 |
| Cadillac | +0.98 | PER +1.67 | BOT +0.29 | 1.38 |
| Ferrari | +0.55 | LEC +0.29 | HAM +0.81 | 0.52 |
| Haas | +0.31 | OCO +2.38 | BEA -1.76 | 4.14 |
| Alpine | -0.43 | GAS +0.24 | COL -1.10 | 1.33 |
| Williams | -1.38 | SAI -2.19 | ALB -0.57 | 1.62 |
| Red Bull | -1.52 | VER -3.95 | HAD +0.90 | 4.86 |
| McLaren | -1.69 | NOR -4.05 | PIA +0.67 | 4.71 |
| Aston Martin | -1.81 | ALO -4.14 | STR +0.52 | 4.67 |

The largest team offset fell from 7.10 to 2.26 and teammate spreads grew to 3.7 to 4.9, as predicted: the shared offset is gone and the driver error is what remains. Where a wide-prior rookie exists (Audi, RB, Mercedes, Haas), the veteran carries the positive bias and the rookie sits near zero. **It is overshoot, not only inversion**: Mercedes is ordered correctly and RUS still carries +3.14. Red Bull, McLaren and Aston Martin are a different problem (stronger driver predicted too well, near-equal sigmas); likely team strength.

### The prior sigma is a clamp

`_driver_sigma` in `scripts/build_teammate_network_prior.py` has three branches: unanchored or fewer than 24 observations gets `1.75 * population_sd` (0.530421, 14 of 31 drivers); otherwise `max(bootstrap, 0.5 * population_sd, floor)` (the 0.151549 floor binds for 9). **23 of 31 drivers carry a clamp.** DOO at 3 observations and LAW at 23 get the same sigma; LAW and RIC (31) differ by 8 observations and 3.5x in sigma.

Both tuning levers (raise the default, raise `min_matched_pairs_quali`) were rejected without scoring as counter-tuning a threshold artefact. The structural fix is a sigma continuous in evidence. Two caveats: nothing shows it converts to MAE, and `population_sd` is itself a fit output. Scoring a lever needs a prior rebuild, re-seed and full replay per arm, not a config edit.

### Open after this entry

- HUL/BOR is inverted by five places: overshoot from the rookie prior sigma. The open item is the continuous sigma, gated on a cheap proxy showing it pays.
- Red Bull, McLaren and Aston Martin: large spreads with no gain asymmetry; likely team strength.
- The walk-forward re-measure on current code was `refused` (`include_grid_scenarios`). The runner exits 0 after the traceback, so a "completed" rerun produced nothing.

## 2026-08-03: the low-observation prior sigma, scored: `noise`

Supersedes the unscored rejection above for the sigma lever. `min_matched_pairs_quali` stays unscored.

Change: the fallback multiplier `1.75 * population_sd` in `_driver_sigma`, exposed as `--low-observation-sigma-multiplier` (`2aafbfc2`). Baseline `b1381e06`, arms on `71fa3615`. Protocol: `scripts/champion_quali_bias.py` (deleted 2026-09-24, source at `71fa3615`), 9 events x 3 seeds x 20 simulations, wet rounds included, leakage-inclusive, so deltas only. Each arm rebuilt the prior and rookie fallback, restored the `710fb551` pre-season artifact, re-seeded and replayed 11 rounds: about 6 minutes per arm. The 1.75 arm reproduced the shipped artifact exactly (MAE 2.6734, HUL +4.89).

| Multiplier | Rookie vs veteran gain | MAE | \|bias\| | HUL | Audi spread |
|---|---|---|---|---|---|
| 1.75 (baseline) | 5.28x | 2.6734 | 1.4747 | +4.89 | 6.11 |
| 1.00 | 1.72x | 2.6734 | 1.4579 | +4.85 | 6.00 |
| 0.50 (bound only) | 0.43x | 2.6599 | 1.4141 | +4.74 | 5.81 |

**Verdict: `noise`.** Cutting rookie gain from 5.28x to 1.72x moves HUL 0.04 positions and leaves MAE identical. Even the indefensible 0.5 bound fixes about 3% of HUL's error. The scorer is deterministic, so these are real but immaterial. Direction is right (spreads shrink), the mechanism is real, and it is not load-bearing. RB moved most (LAW at 23 observations, one short of the cliff); Mercedes moved the wrong way; Aston Martin not at all. Fifth mechanism tested against this bias, fifth loss: the cause is not on the driver-rating path.

## 2026-08-03: centring the prior mu at fit time: `worse`, built and reverted

Built, measured, adopted, reverted the same day. It improved the headline metric and was still wrong.

The idea: finish what `93bfbeb0` started by centring inside `attach_driver_rating_mus` before `team_target_s`, then refit. Slopes steepened about 20% (qualifying 1.77417 -> 2.12643, race 1.97077 -> 2.33329). Qualifying MAE 2.6599 -> 2.5724, \|bias\| 1.5017 -> 1.3771 (60 simulations), suite green.

What killed it:

1. **The gain is a scalar.** Multiplying the shipped slopes by 1.1985, nothing else, reproduces MAE 2.5724 and \|bias\| 1.3771 exactly.
2. **The derived value is not the best one.**

   | Slope x | 0.8 | 1.0 | 1.1985 | 1.4 | 1.7 | 2.1 |
   |---|---|---|---|---|---|---|
   | MAE | 2.7845 | 2.6734 | 2.5993 | **2.5825** | 2.6162 | 2.7071 |
   | \|bias\| | 1.6566 | 1.4747 | 1.3906 | 1.3502 | **1.2492** | 1.3367 |

   \|bias\| and MAE optimise at different points; \|bias\| is not a proxy for MAE.
3. ~~Worse on held-out folds~~ (2025 qualifying r squared 0.3039 -> 0.1192). **Withdrawn the same day:** the folds are 2022 to 2025, which cannot judge a 2026 mapping.
4. **The premise was wrong.** About 82% of the prior's mu variance is between teams (within-team sd 0.2733, total 0.3014), largely real driver quality: fast drivers sit in fast cars. Centring moved real driver signal into `team_target_s`.

**Verdict: `worse`.** Reverted to the 2026-05-19 fit. Points 1, 2 and 4 are measured on 2026 and stand.

### The real finding: the mapping was fitted on the wrong regulations

`training_years` was 2022 to 2025. Fitting the same construct on 2026 alone (3,356 matched pairs, 242 aggregate rows, 11 rounds):

| Session | 2022 to 2025 slope | 2026 slope | Ratio |
|---|---|---|---|
| Qualifying | 1.77417 | 2.76281 | 1.557 |
| Race | 1.97077 | 3.89727 | 1.978 |

The 2026 field is about 56% more spread in qualifying and nearly twice in the race. That is why the centring hack seemed to work: it moved the slope a third of the way. Scored on champion `52dd79c6`, 60 simulations:

| Mapping | Source | MAE | \|bias\| |
|---|---|---|---|
| Shipped | 2022 to 2025 | 2.6599 | 1.5017 |
| Slope x1.1985 | Tuned | 2.5724 | 1.3771 |
| All 2026 | In sample | 2.5724 | 1.2997 |
| Rounds 10 and 11 only | Out of sample | 2.6263 | 1.3131 |

The last row fits on races outside the scoring catalog and still improves both. Three mappings landing on exactly 2.5724 shows MAE saturating. Left `open` that day; adopted the next.

## 2026-08-04: refit the seconds mapping on 2026: `adopted`

Champion `52dd79c6`, local artifacts equal to production (`car_characteristics` v101, `driver_characteristics` v26). Protocol: `scripts/champion_quali_bias.py` (deleted, source at `71fa3615`), 9 events x 3 seeds x 60 simulations, leakage-inclusive, deltas only; the mapping file is the only difference.

| | Champion | 2026 refit |
|---|---|---|
| Qualifying MAE | 2.6599 | **2.5724** |
| \|bias\| | 1.5017 | **1.2997** |
| Qualifying slope | 1.77417 | 2.76281 |
| Race slope | 1.97077 | 3.89727 |

Why 2026 only: per-season slopes are qualifying 1.895, 2.634, 1.351, 1.216 and race 2.165, 1.757, 1.902, 2.059 for 2022 to 2025. The race is a clean break (2026 nearly doubles any prior season); the pooled qualifying 1.774 is an average that describes no real field. Leave-one-round-out in 2026 (11 folds): slope range 2.662 to 2.853 (sd 0.068) qualifying and 3.813 to 4.040 (sd 0.069) race; held-out prediction slope 0.899 and 0.973. Held-out calibration beats the old mapping on its own seasons (mean \|slope - 1\| 0.101 vs 0.294 qualifying, 0.027 vs 0.093 race).

Also fixed or exposed:

- `PriorFitConfig` historical years were recorded but never applied, so adding 2026 to the shared observations would have pulled it into the historical prior. Now enforced.
- One-season fits had no valid validation; `evaluate_within_season_folds` adds leave-one-round-out as `primary_folds`.
- The golden fixtures are tolerance-based and passed both this change and the reverted one. They catch gross regressions, not recalibration.

## 2026-08-04: validate the race half, and scope calibration to a regulation era: `adopted`

`scripts/champion_race_bias.py` (deleted 2026-09-24, source at `ad77ae04`) predicts each event from its actual starting grid, so qualifying error cannot leak in. 9 events x 3 seeds x 60 simulations:

| | 2022 to 2025 mapping | 2026 refit | Delta |
|---|---|---|---|
| Race MAE | 4.0606 | **3.9192** | -0.1414 |
| Race \|bias\| | 2.4242 | **2.3434** | -0.0808 |

**Era policy.** `model.regulation_eras` replaces the hardcoded training years. Calibration is fitted within one era and never across a boundary; a new era is one config entry. It reproduced identical slopes.

**No decay for field compaction.** Per-round slope sd is about 0.73 and thirds of the season are not monotone (qualifying 2.481, 3.013, 2.777; race 4.420, 3.292, 4.082). An apparent -0.10 per round trend was one standard error from overlapping windows and was retracted. A drift diagnostic was built and removed the same day (0.14 and 0.96 SE: nothing to detect).

**Staleness guard.** `tests/test_team_strength_mapping_freshness.py` fails when refitting on current rows no longer reproduces the frozen slope within three standard errors of per-round scatter. Checked that it fires: a 40% perturbation reports `4.9 standard errors apart. Re-run scripts/freeze_team_strength_seconds_mapping.py`.

Race A/B arms take about 15 minutes against 4 for qualifying.

## 2026-08-05: couple the qualifying score scale to the mapping: `adopted`

Qualifying divides the seconds delta by `team_strength_seconds_score_scale` to reach its 0 to 1 score. That was a frozen constant, 1.9707717329051126: the old race slope, used for qualifying since `0590c376` (2026-05-19). The 2026-08-04 refit moved the mapping and not the divisor.

Saturation on the live Dutch GP state, 22 drivers:

| Scale | Drivers at a bound |
|---|---|
| 1.9708 frozen (after the refit) | 7 of 22: LEC, HAM, RUS, ANT, STR, PER, BOT |
| 1.9708 with the old slope | 1 (STR) |
| 2.7628 derived | 0 |

Saturation erases the driver rating (both teammates pin to the same value; Ferrari and Mercedes both hit 1.000). Dividing by the live qualifying slope recovers centred team strength exactly and keeps the signal inside 0 to 1. This also supersedes the 2026-07-31 claim that the team term alone cannot reach a bound.

Champion `76d26fcf`, `scripts/champion_quali_bias.py --all-events` (deleted, source at `71fa3615`), 9 x 3 x 20:

| | 1.9708 | 2.7628 |
|---|---|---|
| Qualifying MAE | 2.5758 | **2.5556** |
| \|bias\| | 1.2862 | **1.2660** |

| Scale | 1.9708 | 2.2 | 2.4 | 2.606 | 2.8 | 3.069 | 3.3 | 3.7 | 4.2 |
|---|---|---|---|---|---|---|---|---|---|
| MAE | 2.5758 | 2.5791 | 2.5791 | 2.5791 | **2.5488** | 2.5623 | 2.5657 | 2.5892 | 2.6027 |
| \|bias\| | 1.2862 | 1.2795 | 1.3064 | 1.2761 | 1.2559 | 1.2559 | 1.2458 | **1.2290** | 1.2458 |

The sweep spans 0.054 with no bowl, so the 0.0202 gain is jitter and is not claimed. **Adopted on the structural argument:** the two values were never coupled, so every refit would silently re-saturate. The race side divides nothing and cannot saturate. Tests: the scale must equal the live qualifying slope, must not be the race slope, and no strength in 0.02 to 0.98 may hit a bound (checked to fail under the old constant). An explicit config value still overrides for A/B arms.

## 2026-08-24: qualifying classification is not the starting grid: `never activated` on the replay, `adopted` for the residual dataset

Nothing read FastF1's `GridPosition`; every start position was the qualifying classification, which ignores penalties. `fetch_actual_starting_grid()` in `src/data/actual_results_fetcher.py` now produces the real grid and `start_type`.

Baseline `19085bd5`, fresh 12-round replay. 41 of 264 driver-races (15.5%) differ from the real grid:

| Error (positions) | -11 | -10 | -4 | -3 | -1 | +1 | +2 |
|---|---|---|---|---|---|---|---|
| Rows | 1 | 1 | 2 | 3 | 4 | 18 | 12 |

Worst: Spa, NOR qualified P3 and started P13; HAD P10 to P21.

**Replay: `never activated`.** Identical race MAE at all 42 checkpoints (3.8468). Every replay checkpoint is before qualifying, so the replay always predicts the grid. This corrects a planning claim that the ledger's race MAE had been scored on grids that never happened.

**Residual dataset: `adopted`.** `build_race_residual_dataset` built its grid feature and label from the classification, so 15.5% of rows were mislabelled. It now uses the real grid. The residual model is disabled, so no live forecast changes.

**Also fixed:** 2026 Barcelona FP1 has laps but no team names in FastF1 (refetch confirmed), and it aborted the whole replay at round 7. Practice sessions now log and skip (`skipped_sessions`); testing and competitive sessions still fail closed.

Side measurement, not walk-forward (3 seeds x 100): the six affected rounds averaged -0.056 MAE with the real grid; the six unaffected were identical. Inside noise, not claimed.

## 2026-08-26: track difficulty caps pass probability: `adopted`

Track difficulty only shifted a pass threshold and pace swamped it. At Monaco a car 3.4 s/lap quicker passed 95% of the time. The pass probability is now capped by the track's observed rate, `overtaking_avg_changes_per_lap / (field_size - 1)`: for a 4 s/lap edge, Monaco 0.048, Monza 0.142, Spa 0.200. Tracks without data keep 0.95.

Baseline `c7623714`, rebuilt: race MAE 3.6136 over 12 rounds from the actual qualifying classification, 100 simulations, seed 42. **Result 3.6136 -> 3.5909**, modest. (Below the later race floor; single seed.)

Rejected: a queue rule scored 3.5152 but broke four behaviours that pass on the champion (`test_higher_skill_driver_wins_majority_of_intra_team_battles` 58.8% vs 60%, `test_high_sc_probability_produces_variance` zero upsets, two plumbing tests where the faster car finished second). An MAE gain bought with false physics. (Adopted later, 2026-08-28, with a safety car exemption.)

Known limit: a penalised Mercedes from P22 is predicted P5 at Monza, P8 Hungary, P9 Monaco; Monaco is 4 to 6 places optimistic because the simulator retires 3.1 cars there against 6 in reality. `overtaking_observed_races` was 0 for all 25 tracks.

## 2026-08-26: the 2026 overtaking rates, measured: `adopted`

**Superseded 2026-09-25:** each 2026 value is the forecast race's own result, so the gain below is leakage, not evidence.

The cap's input came from 2022 to 2024 races. `scripts/extract_overtaking_rates.py` measured 2026 with the same counting rule:

| Race | 2026 | Prior | Race | 2026 | Prior |
|---|---:|---:|---|---:|---:|
| Australian | 2.30 | 2.81 | British | 3.14 | 2.53 |
| Chinese | 3.20 | 4.53 | Belgian | 3.16 | 5.15 |
| Japanese | 2.83 | 3.68 | Hungarian | 3.39 | 3.27 |
| Miami | 2.80 | 3.14 | Dutch | 3.56 | 3.11 |
| Canadian | 2.21 | 2.28 | Monaco | 1.26 | 1.12 |
| Austrian | 2.80 | 3.38 | Barcelona | 3.62 | n/a |

Nine of eleven circuits are lower in 2026 and the spread compressed from 4.03 to 2.36: the regulation change. The loader blends toward the previous era (12 to 19% weight at one race). **Result 3.5909 -> 3.5606** against the rebuilt `c7623714` baseline 3.6136. The transition schedule (`races_to_full_weight: 8`) was built for drift within an era, not an era break.

## 2026-08-26: retirements, both layers: `adopted`, and a probe whose evidence failed

Raw FastF1 has **42 retirements in 264 driver-races (0.201 each, 4.42 per race)**:

| Layer | Before | After | Actual |
|---|---:|---:|---:|
| Simulator input, field sum per race | 3.13 | 4.43 | 4.42 |
| Reported probability, mean | 0.064 | 0.200 | 0.201 |

The probe that justified shrinking the reported number (`dnf_calibration_probe.md`) counted 11 retirements across 13 events; its actuals predate `scripts/backfill_dnf_data.py`. Pooled Brier on complete actuals:

| λ | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|
| Brier, true rate 0.201 | 0.16045 | 0.16086 | 0.16362 | 0.16875 | 0.17622 |
| Brier, probe's 0.038 | 0.18694 | 0.18409 | 0.18135 | 0.17873 | 0.17622 |

The ranking flips: the model under-forecasts. Changed: `dnf_probability_base_rate` 0.04 -> 0.20, and `dnf_season_calibration_multiplier` 1.415 scales the simulation input to the observed rate (relative order kept). λ stayed 0.25. **Cost 3.5606 -> 3.5758 race MAE**, the price of simulating retirements at the true rate. Per-track attrition was rejected: observed sd (1.68) is below Poisson noise (2.10) at one race per circuit. (λ moved to 1.0 on 2026-09-20; the constants are fitted on 2026, see 2026-09-25.)

## 2026-08-26: teammate setup offset into base pace: `worse`, reverted

Moving the persistent teammate spread into `base_lap_time` so the overtake model sees it. **3.5758 -> 3.6439**, worse than champion. The offset is a random draw, so feeding it to the overtake model turns noise into position changes. A comment at the call site now records this.

## 2026-08-28: position changes were not caused by passing: `adopted` (anchor removal), queue rule adopted after 2026-08-29

A penalised ANT was forecast P3 from P22. Position came from cumulative lap time, so cars crossed whether or not a pass fired:

| Circuit | Passes | Changes | Rate |
|---|---:|---:|---:|
| Monaco | 46 | 2949 | 1.6% |
| Hungarian | 196 | 3151 | 6.2% |
| Belgian | 357 | 2715 | 13.1% |

Baseline `566b5d3b`: race MAE 3.5758, 12 rounds, 100 simulations, seed 42. Four dead ends (each about 1% on churn): cap by contending pairs alone (MAE 3.5833, worse), dirty air at its measured size (churn ratio 1.76 -> 1.74), per-circuit lap 1 spread (1.69), pass probability x0.10 (1.75).

**Adopted: `resolve_pace_anchor` deleted.** It restored a penalised driver's qualifying slot as his anchor, erasing the penalty. ANT goes P3 -> P18, MAE unchanged (3.5758; no scored round had a penalty).

**Queue rule plus contending-pairs cap** (a position change needs a completed pass; pitted, retired and neutralised laps exempt): MAE 3.5758 -> 3.5227. Displacement ratio 2.07 -> 1.76, churn ratio 1.76 -> 1.25, churn correlation +0.273 -> +0.453 (target 0.617). Each part alone was worse (3.5833, 3.6364), so the pair may be two errors cancelling. Adopted once the skill calibration (2026-08-29) fixed the last failing test; suite green.

**MAE cannot judge any of this.** Per-race sd 1.339, standard error 0.387; every configuration (3.5152 to 3.6364) sits within 0.16 SE of champion. Decisions were made on the physics metrics. An earlier harness patched the wrong binding, inflating two figures (corrected: churn ratio 2.01 -> 1.76, correlation +0.389 -> +0.273). Churn reliability 0.595 (ceiling 0.771); displacement reliability -0.223, so per-track displacement cannot be judged until a second season.

Recovery from P15 or worse, 2022 to 2025, 401 driver-races ranked within finishers, by the car's season median finish:

| Car | n | Median | p75 | p90 | Max |
|---|---:|---:|---:|---:|---:|
| Top (6 or better) | 31 | +7 | +10 | +13 | +13 |
| Upper mid (7 to 11) | 99 | +3 | +5 | +8 | +12 |
| Backmarker (12+) | 271 | +1 | +3 | +4 | +12 |

2026 top cars: HAD P21 -> P6, VER P20 -> P6.

## 2026-08-29: the blend was discarding a correct simulation: `adopted`, model version 3.0

With the queue rule, the simulator recovers a penalised driver on its own (ANT at Monza: P14 by lap 4, P4 by lap 48), but the reported finish was P17. For drivers flagged `is_penalised` only: `resolve_pace_anchor` is gone, the grid anchor and the `max_gain` floor are skipped. A flag, not a substitute position, because a boolean cannot erase a penalty.

**Measured team race pace.** `team_strength` comes from classified results, which mix pace with reliability and luck. The simulator had Red Bull fastest and Mercedes third; measured 2026 race pace has Mercedes fastest and Red Bull fourth. `scripts/extract_team_race_pace.py` measures median green-flag laps per team and the simulator prefers it. (Its season average leaked into the replay; fixed 2026-09-25.)

**Driver skill.** `skill_improvement_max` 0.75 -> 1.75, fitted to two independent measurements:

| Target | Measured | Model at 0.75 | Model at 1.75 |
|---|---|---|---|
| Teammate lap gap (114 team-races) | 0.352 s/lap | 0.150 | 0.350 |
| Stronger driver win rate (34 team-seasons) | 0.667 | 0.546 | 0.692 |

Team to driver influence ratio 2.33 -> 1.00, inside the 2.40 cap.

```text
ANT P22 ->  Monza P7   Hungary P7   Monaco P8     (champion: P3 / P4 / P5)
BOT P22 ->  Monza P18  Hungary P20  Monaco P20
MAE 3.5758 -> 3.5152   (0.16 SE, not distinguishable)
suite 1762 passed, 5 skipped, 3 xfailed, 0 failed
```

Watch: BOT drifted P22 -> P19 -> P18 at Monza across the three changes, each step inside the envelope. Pooling all back-of-grid starts dragged the p90 to 4.8 and nearly sent this work into an unneeded refit; bucket by car quality.

## 2026-09-04: car x track fit: `rejected`; season-form construct: `candidate`

Prompted by a user report that the model "overweights recent performance instead of getting the characteristics of the car vs track".

**Car x track fit cannot be recovered.** Leave-one-race-out over the calibration observations, with a permutation null:

| Season | Races | Race | Qualifying |
|---|---|---|---|
| 2022 | 19 | -4.9% (p=0.72) | -6.2% (p=0.76) |
| 2023 | 21 | -5.1% (p=0.80) | -3.7% (p=0.61) |
| 2024 | 22 | -4.6% (p=0.73) | -3.5% (p=0.45) |
| 2025 | 22 | -1.4% (p=0.25) | -9.8% (p=0.96) |
| 2026 | 10 | +12.1% (p=0.034) | -14.7% (p=0.93) |

Negative in 9 of 10 cells. The one positive came from 48 looks against a null 95th percentile of +11.4% and does not survive selection. Archetype bins matched shuffled labels (p 0.44 to 0.67). `calculate_track_suitability` was removed from `get_blended_team_strength` (its weight was already 0 from race 4); `qualifying_residual_model` still uses it.

**Recency exponent 1.8 -> 0.3: `candidate`, directional.** Walk-forward over 2026, scored in seconds on the held-out race:

| Exponent | Race MAE | Qualifying MAE |
|---|---|---|
| 0.0 | 0.6583 s | 0.5670 s |
| 0.9 | 0.6676 s | 0.5747 s |
| 1.8 (shipped) | 0.6775 s | 0.5851 s |
| 2.5 | 0.6824 s | 0.5902 s |

Monotone in both. 0.3 keeps some sensitivity to upgrades. A team-clustered bootstrap on 1.8 -> 0.0 includes zero (race [-0.0212, +0.0553], qualifying [-0.0113, +0.0481]). Not scored on a rebuilt baseline.

**Team strength is a rank statistic.** `score_teams_from_actual_rows` gives the fastest team 1.0 and the slowest 0.0 regardless of gaps, and the mapping's calibration data ranks `team_median_s` away (374 rows, 27 distinct values). This is why margin scoring lost twice: the slope was fitted on rank.

**DNF exclusion from season form: `candidate`.** A car retiring on lap 3 scored as a slow car. `row_is_dnf` now gates the position mean. `scripts/backfill_dnf_data.py` wrote 62 DNF row labels across 11 files (3 races, about 18 distinct retirements) and re-derived no probabilities (archive: `~/Documents/trackside-labs-archive/predictions_pre_dnf_backfill_20260904_231119.tar.gz`). 14 of 28 target sets moved, all race or sprint; at Australia (6 retirements) McLaren 0.5 -> 0.8, Red Bull 0.4 -> 0.7, Haas 0.8 -> 0.6, Alpine 0.6 -> 0.3. Retirements were scoring fast cars as slow, and recency amplified it: this, more than the exponent, explains the user report. Ceilings: a team whose cars all retired keeps its unfiltered rows; one surviving car is not comparable to two.

## 2026-09-04: prediction-path stress pass: three unreported defects, one dominant

Adversarial probes on real 2026 data. Invariants hold: all schedules sum to 1.0, current weight rises with race number, team scoring ignores row order, replay state cannot leak into production.

1. **Rank costs 0.605 s RMS and dominates.** Real per-team gaps round-tripped through the rank score:

   | | Race | Qualifying |
   |---|---|---|
   | RMS error | 0.605 s | 0.522 s |
   | Mean absolute error | 0.464 s | 0.376 s |
   | Worst | 2.019 s | 1.731 s |
   | Real median adjacent gap | 0.302 s | 0.234 s |
   | Gap the model assigns (constant) | 0.476 s | 0.362 s |
   | Adjacent pairs where model > 4x reality | 20/90 | 19/84 |

   About 25x the value of the recency change (0.019 s). One position swap moves a team 0.390 s at the race slope whatever the real gap.
2. **Teams in one race were scored on different constructs**, chosen per team by list length. Live and saved sources disagree by 0.098 units (0.38 s) on average, up to 0.308 (1.20 s, Audi). McLaren was scored on rank, the other ten on pace.
3. **Bad rows dropped silently; one team left becomes 0.5.** An unmapped team name took a rank slot (Ferrari 0.667 -> 0.750, Williams 0.333 -> 0.250). Aliases resolved 28 of 32 variants.
4. **Latent:** recency weights by list position, not race number.
5. **Reported DNF probability confined to [0.150, 0.238]**; certain retirement and the cap look the same, and the simulation samples different rates.

Fix 1 and 2 before touching any weighting lever again.

## 2026-09-05: season-form construct defects fixed: `adopted` (mechanism), lead measured

1. **One construct per race.** The source is chosen once for the whole field by total coverage, memoised per `(year, race)`, ties `replayed > saved > live`. A team missing from it falls back to the pre-season baseline.
2. **Bad input.** Unknown teams are excluded and logged; if nothing resolves, ERROR and `{}`. The single-team fabricated `{team: 0.5}` now returns `{}`. Four aliases added (`Cadillac F1 Team`, `Mercedes-AMG`, `McLaren Formula 1 Team`, `General Motors`): 32 of 32 resolve.
3. **Recency by race number.** Saved and replayed observations carry a race ordinal. The first attempt made the newest race the least weighted when it was unmapped (ordinals `[5, 6, 3]`, weights 0.343 / 0.362 / 0.294); review caught it. The fallback is now relative to the last ordinal with a clamp. `live` keeps index weighting.

**Lead measured, not adopted: margin vs rank.** Using `team_target_s` (real gap to the field median), walk-forward, exponent 0.3:

| Season | Race | Qualifying |
|---|---|---|
| 2026 | +15.7% (CI [+0.005, +0.202]) | +23.0% (CI [+0.058, +0.209]) |
| 2025 | -8.8% (significant, margin worse) | +1.5% (ns) |
| 2024 | -0.6% (ns) | +2.0% (ns) |

Best blend weight (0 rank, 1 margin): race 0.0, 0.0, 0.25, 0.0, **1.0** and qualifying 1.0, 0.0, 0.75, 0.75, **1.0** for 2022 to 2026. Qualifying margin wins or ties in 4 of 5 seasons. Indicative only (standalone harness, 11 rounds); it needs a mapping refitted on a margin-native predictor first.

Not changed: the DNF output band (an output-layer decision) and two couplings (the source cache key ignores `self.teams`; the completed-race count ignores the chosen source).

## 2026-09-07: the qualifying matched-pair gate: defect confirmed, `worse` fix rejected

**Falsified: driver ratings do not carry a per-team offset into qualifying.** `qualifying_preparation.py` centres `quali_rating_mu_s` per team before its only consumer, so the per-team mean reaching the score is zero. This is the surviving half of `93bfbeb0`; the 2026-08-03 revert was the fit-time half only. All 11 teams have two finite values and all 29 drivers a full state, so nothing escapes. The -11.9 s slope was measuring team pace. `race_rating_mu_s` is not centred at all; no symptom yet.

**Confirmed: qualifying observations are gated in a way that tracks team pace.** `min_matched_pairs_quali = 3` drops team-sessions below 3 pairs, which Q1-out teams hit far more often. 2026, 132 team-sessions:

| Pairs | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Team-sessions | 4 | 2 | **23** | 55 | 42 | 4 | 2 |

Qualifying updates that moved mu: Aston Martin 3 of 18, Cadillac 6 of 17 to 18, Williams 7 of 17, Haas 11 of 18, McLaren 18 of 18.

**Fix that loses: gate 3 -> 2.** Baseline `757087f3` plus season-form work, rebuilt, 13 rounds:

| Checkpoint | Gate 3 | Gate 2 |
|---|---|---|
| All | **2.3363** | 2.3931 |
| PRE | 2.5728 | 2.6141 |
| FP1 | 2.3959 | 2.4439 |
| FP2 | 2.1775 | 2.2738 |
| FP3 | 1.8831 | 1.9324 |
| SQ | 2.4545 | 2.6147 |

Mean delta +0.0633, CI [+0.0062, +0.1204], 12 better / 23 worse / 11 tied. Race: noise (-0.0179). Coverage equalised (Aston Martin 3 -> 13) and it still lost: at n=2 the SE understates uncertainty by 1.5 to 2x (|err|/SE median 1.20 against 0.674 expected), mean \|mu\| movement per update rose 25%, sigma collapsed field-wide and two teammate orders flipped (Haas, RB). **Verdict: `worse`, reverted.** Confirmed on correlation on 2026-09-12 (-0.0051, CI [-0.0095, -0.0009]). The test suite did not catch it.

**Follow-up sweep: inflate the n=2 SE by lambda.** Same baseline, 46 checkpoints:

| Arm | Qualifying MAE | Delta | CI | Verdict |
|---|---|---|---|---|
| Gate 3 (baseline) | 2.3363 | | | |
| Gate 2, λ 1 | 2.3931 | +0.0633 | [+0.0062, +0.1204] | `worse` |
| Gate 2, λ 2 | 2.3507 | +0.0221 | [-0.0285, +0.0749] | ~~`noise`~~ `unresolvable` (2026-09-12) |
| Gate 2, λ 4 | 2.3333 | +0.0023 | [-0.0448, +0.0471] | ~~`noise`~~ `unresolvable` (2026-09-12) |

Monotone toward the baseline, no interior optimum: the best the recovered rows can do is change nothing. PRE was the only consistently better checkpoint (λ2 -0.0360, λ4 -0.0280) with CIs including zero at n=13. **P0 closed.** The gate stands; reopen only by scoring PRE alone over more rounds. The sweep hooks were reverted.

Untested: `_selected_qualifying_matches` stops at the gate, so a team with three Q3 pairs never reads its Q2 and Q1 pairs.

## 2026-09-07: P1 deployment construct: `never activated`

`score_teams_from_actual_rows` changed from rank to the fixed-scale margin construct. Same baseline, 13 rounds: **46 of 46 checkpoints tied exactly.**

The field source resolves to `live` in 12 of 13 rounds (coverage 21 to 130), and to `replayed` with zero coverage at the opener. Production has 141 live observations. So the rank path runs only for backtests of other seasons. "Team strength is a rank statistic end to end" is false for the live path; `live` values are already margin-preserving. Reverted, because backtests of other seasons do use it and were not measured. Arm B (with a refitted mapping) was cancelled.

**Units mismatch, measured and dismissed.** Live values fed to the mapping vs the rank rows it was fitted on: mean 0.5041 vs 0.5000, sd 0.3128 vs 0.3233 (n 141 vs 196). +0.011 s qualifying, +0.016 s race. Immaterial. Live values also span exactly 0 to 1, so they saturate like rank.

What remains of P1: a construct in real lap-time seconds on both the calibration and live side, with the mapping refitted. On identical information, lap-time margin beats lap-time rank by +0.157 R squared in 2026 qualifying and +0.122 in race, and position-based constructs sit 0.10 to 0.25 below lap-time ones. Needs a per-team green-flag seconds source anchored on the field median.

## 2026-09-12: the protocol was unrunnable, and its seed floor was never measured

Not a model change. The protocol named `scripts/run_challenger_research_walk_forward.py` with 3 seeds from `DEFAULT_REPLAY_SEEDS`; neither exists (zero Python files), and two data paths are gone. So work fell back to `replay_historical_checkpoints.py`, which always ran seed 42: every verdict from 2026-09-04 to 2026-09-07 rests on one draw.

**Seed floor** (same code, seed 42 vs 43, 13 rounds, 46 checkpoints):

| Target | Metric | Mean delta | 95% CI |
|---|---|---|---|
| Qualifying | overall_mae | -0.0122 | [-0.0439, +0.0187] |
| Qualifying | correlation | -0.0007 | [-0.0039, +0.0024] |
| Race | overall_mae | +0.0342 | [-0.0184, +0.0868] |
| Race | correlation | -0.0009 | [-0.0076, +0.0059] |
| Sprint race | overall_mae | -0.0251 | [-0.1163, +0.0609] |

Per-checkpoint sd 0.11 qualifying, 0.18 race; the seed alone changes the score on 30 of 46 qualifying and 41 of 46 race checkpoints. Metric resolving power is the table in the protocol section.

**Tooling added.** `--seed` on the replay (default 42) threaded to the predictor, and `scripts/compare_replay_arms.py` (correlation first, floor-gated, separate `unresolvable` and `identical` verdicts).

**Verdicts revised** (qualifying correlation, 46 checkpoints):

| Arm | Delta | CI | Verdict |
|---|---|---|---|
| Seed floor | -0.0007 | [-0.0039, +0.0024] | noise |
| Gate 2, λ 1 | -0.0051 | [-0.0095, -0.0009] | worse |
| Gate 2, λ 2 | -0.0021 | [-0.0056, +0.0013] | unresolvable |
| Gate 2, λ 4 | -0.0008 | [-0.0038, +0.0021] | unresolvable |
| P1 margin construct | +0.0000 | [0, 0] | identical |

Gate 2 is confirmed worse and better evidenced than before (also MAE +0.0633, exact accuracy -3.85, top 10 -1.96). λ 2 and 4 were `noise` and are `unresolvable`. The P0 closure still holds.

Not done then: the promotion gate scored MAE only (fixed 2026-09-19 to 20, `d152509c`). The evaluation report and dashboard stay on MAE deliberately.

## 2026-09-21: baseline rebuilt over 14 rounds, floor re-measured, pipeline determinism proven

**Superseded 2026-09-25:** this replay leaked three future inputs. Its floor and levels are kept for reading entries gated on them.

No model change. Code `54810683` (the same day's λ 0.25 -> 1.0 moves only the reported DNF number). Pre-season reset from the committed flat `driver_characteristics.json` (sha256 `62ea9300…`, 2023 to 2025, `sessions_observed=0` for all 29 drivers). 14 of 25 events, last the Spanish GP on 2026-09-13; 51 paired checkpoints for qualifying and race, 15 for sprint.

| Root | Seed | Purpose |
|---|---|---|
| `data/historical_replay_base42_r14` | 42 | First run, inputs differ |
| `data/historical_replay_base42_r14_check` | 42 | Baseline |
| `data/historical_replay_base43_r14` | 43 | Floor partner |

**Determinism.** The two seed-42 runs gave byte-identical metrics in all 117 snapshots and identical predictions apart from timestamps and run IDs. Run 1 replayed 68 sessions and runs 2 and 3 replayed 69 (the extra `Spanish Grand Prix::R`, after the last scored forecast), so pair with `base42_r14_check`.

| Target | Metric | Mean delta | 95% CI | Verdict |
|---|---|---|---|---|
| Qualifying | correlation | -0.0003 | [-0.0035, +0.0026] | noise |
| Qualifying | overall_mae | -0.0157 | [-0.0458, +0.0141] | noise |
| Race | correlation | -0.0005 | [-0.0069, +0.0060] | noise |
| Race | overall_mae | +0.0327 | [-0.0190, +0.0840] | noise |
| Sprint race | correlation | -0.0019 | [-0.0094, +0.0042] | noise |
| Sprint race | overall_mae | -0.0251 | [-0.1114, +0.0606] | noise |

Floor then: qualifying correlation 0.0035 and MAE 0.046, race 0.0069 and 0.084, sprint 0.0094 and 0.111. Stable against 2026-09-12. Champion levels (leaky): qualifying correlation 0.8692 and MAE 2.3257, race 0.6652 and 3.5666, sprint 0.8366 and 2.5706.

Sprint `exact_accuracy` came back `better` (+3.03, CI [+0.30, +5.76]) with no model change: at n=15 that metric invents significance. Never let a sprint accuracy number carry a decision. Scoring seed 43 as a candidate returns `unresolvable` everywhere, confirming the gate fires.

## 2026-09-25: the replay leaked three future inputs; baseline and floor rebuilt without them

No model change. Code `66873af5` plus uncommitted changes: the three fixes below, `--through-round` on the replay, and `tests/test_forecast_information_cutoff.py`.

The 2026-09-21 replay reset driver and car state and built each checkpoint before later sessions, but read three inputs from the repository at their end-of-season state:

| Input | What leaked | Fix |
|---|---|---|
| `team_race_pace/2026_team_race_pace.json` (`lap_by_lap_simulator`) | A 12-race average used for every race forecast from round 1 PRE, preferred over the results-derived delta | Per-race gaps stored under `races`; the loader averages only races before the target and returns nothing for round 1, an unknown race or an old-format file |
| `team_strength_seconds_mapping/latest.json` | Slopes fitted on 11 rounds of 2026, used at every round, qualifying included | The replay refits per race on earlier 2026 rows, falling back to the 2022 to 2025 fit for round 1 (what live had then) |
| `overtaking_avg_changes_per_lap` in `2026_track_characteristics.json` | Each circuit's 2026 value is that race's own result | A measurement in the target season's own file is ignored (one race per circuit); the previous season's value is used |

The 2026-08-26 "2026 overtaking rates" gain was measured entirely on the third leak.

Protocol: `--through-round 14`, seeds 42 and 43, roots `data/historical_replay_wf42_r14` and `data/historical_replay_wf43_r14`, 14 races (Australia to Spain), 51 checkpoints, Barcelona FP1 skipped. Run in a separate terminal: two attempts inside Claude Code were killed for low memory, and a probe showed the replay flat at about 1.15 GB over four rounds.

**Seed floor, leak-free:**

| Target | Metric | Mean delta | 95% CI | Verdict |
|---|---|---|---|---|
| Qualifying | correlation | +0.0008 | [-0.0020, +0.0035] | noise |
| Qualifying | overall_mae | -0.0264 | [-0.0543, +0.0024] | noise |
| Race | correlation | +0.0042 | [-0.0025, +0.0110] | noise |
| Race | overall_mae | +0.0056 | [-0.0457, +0.0576] | noise |
| Sprint race | correlation | -0.0008 | [-0.0078, +0.0056] | noise |
| Sprint race | overall_mae | -0.0124 | [-0.0791, +0.0612] | noise |

Gate on the widest bound: qualifying correlation 0.0035 and MAE 0.054; race 0.0110 and 0.058; sprint 0.0078 and 0.079.

**What the leaks were worth** (old `base42_r14_check` vs new `wf42_r14`, same seed, gated on this floor):

| Target | Metric | Leaky | Leak-free | Delta | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| Qualifying | correlation | 0.8692 | 0.8639 | -0.0053 | [-0.0096, -0.0018] | worse |
| Qualifying | overall_mae | 2.3257 | 2.3631 | +0.0374 | [+0.0125, +0.0677] | unresolvable |
| Race | correlation | 0.6652 | 0.6574 | -0.0079 | [-0.0141, -0.0023] | unresolvable |
| Race | overall_mae | 3.5666 | 3.6289 | +0.0623 | [+0.0250, +0.1054] | worse |
| Sprint race | correlation | 0.8366 | 0.8406 | +0.0040 | [-0.0039, +0.0143] | unresolvable |

The leaks flattered both main targets, by about 0.06 race positions and 0.005 qualifying correlation. Qualifying moved because the mapping slope sets the qualifying score scale.

**Champion levels:** qualifying correlation 0.8639 and MAE 2.3631; race 0.6574 and 3.6289; sprint 0.8406 and 2.5278.

**Still in sample:** constants fitted on 2026 and read at every round: `dnf_season_calibration_multiplier` 1.415 and `dnf_probability_base_rate` 0.20; `skill_improvement_max` 1.75 (its 114 team-races may include 2026, unchecked); `driver_rating_mu_s` in the calibration observations (source state unchecked). Removing them needs a per-round refit.

## 2026-09-26: old adopted changes re-measured on the leak-free replay

Six arms, each the leak-free baseline (`data/historical_replay_wf42_r14`, `--through-round 14`, seed 42) with one adopted change put back. Gated on the 2026-09-25 floor. Code `66873af5` plus uncommitted changes, including two A/B switches added for this: `--previous-era-mapping` on the replay and `baseline_predictor.race.track_pass_cap_enabled` (default on). Arm roots: `data/historical_replay_arms/`. An arm that scores worse means the adopted change helps.

| Arm (puts back) | Change on trial | Qualifying correlation | Race correlation | Race MAE | Verdict for the adopted change |
|---|---|---|---|---|---|
| E: 2022 to 2025 mapping at every round | 2026-08-04 mapping refit | -0.0041 [-0.0069, -0.0013], 14 / 28 / 9 | -0.0001 | +0.0168 | **qualifying `better`**, race `unresolvable` |
| B: qualifying scale frozen at 1.9708 | 2026-08-05 scale coupling | -0.0010 | -0.0023 | +0.0318 | `unresolvable` |
| C: recency exponent 1.8 | 2026-09-04 recency 0.3 | +0.0005 | -0.0019 | +0.0201 | `unresolvable` |
| F: track pass cap off | 2026-08-26 pass cap | identical | -0.0014 | +0.0072 | `unresolvable` (race only) |
| A: `skill_improvement_max` 0.75 | 2026-08-29 skill 1.75 | identical | **+0.0028** | **-0.0131** | `unresolvable` (race only) |
| D: DNF multiplier 1.0 | 2026-08-26 retirement calibration | identical | **+0.0024** | **-0.0223** | `unresolvable` (race only) |

Deltas are arm minus baseline, 51 checkpoints (15 for sprint). Better / worse / tied is from the arm's side. Qualifying is identical for A, D and F because those are race-only settings, which confirms the arms isolate what they claim to.

**What this settles.** The 2026 seconds mapping refit is the only old change with a measured, floor-clearing effect: fitting on earlier 2026 rounds beats the 2022 to 2025 mapping on qualifying correlation. Every other adopted change is below the floor on this replay, so their original gains (all single-seed, several on the deleted leaky scorer) were never evidence. The coupling (B) was adopted on structural grounds and stays; nothing here argues against it.

**Arms A and D leaned the other way at seed 42** (putting back the old value made the race slightly better: 29 / 17 and 28 / 16 checkpoints, inside the floor), so both were repeated at seed 43 against `data/historical_replay_wf43_r14`:

| Arm | Seed | Race correlation | Race MAE | Better / worse / tied |
|---|---|---|---|---|
| A: skill 0.75 | 42 | +0.0028 [-0.0013, +0.0069] | -0.0131 | 29 / 17 / 5 |
| A: skill 0.75 | 43 | +0.0007 [-0.0046, +0.0058] | -0.0089 | 19 / 24 / 8 |
| D: DNF multiplier 1.0 | 42 | +0.0024 [-0.0007, +0.0059] | -0.0223 | 28 / 16 / 7 |
| D: DNF multiplier 1.0 | 43 | -0.0046 [-0.0103, -0.0001] | +0.0119 | 18 / 26 / 7 |

D flips sign between seeds, so its seed-42 lean was noise. A keeps the sign but shrinks to almost nothing and loses the checkpoint majority. Both stay `unresolvable`; neither is a candidate to revert.

**Not measurable on this replay.** 2026 overtaking rates (inactive in-season since 2026-09-25), `resolve_pace_anchor` removal and the penalised-driver blend (every checkpoint is before qualifying, so no penalties apply), and `757087f3` (cannot be reverted alone on current code).

## Adding an entry

- What changed, in one line: the idea, not the code.
- The baseline, by commit and replay root.
- The protocol, if it differs from the one above.
- The numbers, including how many checkpoints went each way. A mean alone hides a 4-3 split.
- A verdict from the table at the top.
- For `never activated` or `refused`, the reason word for word.

A result without its baseline cannot be reused. That is how logs like this go stale.
