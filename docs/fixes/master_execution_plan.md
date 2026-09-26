# Master plan: separating driver ratings from car pace

Written 2026-05-09. This is the order of work for removing car pace from driver ratings and splitting race and qualifying state. Each phase's detail lives in its own doc; this file owns the order. The full original text is in git history.

## Rules

1. Change the live updater last, once the prior, mappings, schema and diagnostics exist.
2. A phase starts only when the one before meets its acceptance criteria. A doc with TODOs in required fields is not closed.
3. **Removal rule (K=3):** old fields and fallback readers are removed only after 3 consecutive finished race weekends on the new path with no regression. A regression inside that window restarts the count. Removal is its own change.

## Phases

| # | Phase | Goal | Status (last recorded) |
|---|---|---|---|
| 0 | Doc lock | Design assumptions written and reviewable before code | Done |
| 1 | Validation evidence | External teammate deltas to grade the prior | Done 2026-05-17; PACETEQ rows downgraded to context (see `teammate_network_prior_construct_audit.md`) |
| 2 | Smoke sessions | Lock the extractor test sessions | Done 2026-05-12 |
| 3 | Extractor | `extract_matched_teammate_laps()`, one row per matched lap pair | Done |
| 4 | Extractor check | Run on the smoke sessions | Done 2026-05-13 |
| 5 | Bulk extraction | Historical observation set | Done |
| 6 | Prior fit | Race and qualifying teammate-network priors | Done |
| 7 | Seconds mapping | Team strength to seconds, separate race and qualifying | Done 2026-05-19 |
| 8 | Replay and leakage diagnostics | Prove orthogonality before touching live state | First artifact 2026-05-20 |
| 9 | Schema migration | Add seconds fields without breaking reads | Done 2026-05-21 (see `driver_seconds_and_wet_trace_remediation.md`) |
| 10 | Race/qualifying state split | Separate Bayesian state per session kind | Done 2026-05-21 (same doc) |
| 11 | Wet skill | Move wet skill to the same lap-time observations | Not recorded |
| 12 | Duplicate EMAs | Stop double counting through `race_pace` and `quali_pace` | Not recorded |
| 13 | Test rewrite | Tests match the new contract | Not recorded |
| 14 | Rollout | Validate the full path, then apply the K=3 removal | Not recorded |

Phases 11 to 14 have no status in this doc. Check the code before assuming either way.

## Weather routing (phases 3, 8, 11)

- All weather samples in a lap dry: the lap can feed dry ratings.
- All wet: it can feed `wet_skill`.
- Mixed, missing or unmappable: neither, with `lap_level_weather_unreliable`.
- A fully wet session never updates dry ratings.

## Phase 7 results worth keeping

- Separate race and qualifying mappings over one stored `team_strength` scalar, policy `same_session_construct`. First fit (2022 to 2025): race slope 1.970772, qualifying 1.774169. Refitted on 2026 on 2026-08-04 (see the ledger).
- The stored scalar was not split into short-run and long-run states: the shared state beat the split on combined MSE (0.5049 vs 0.5077) and the split won only 2 of 4 folds. Reopen only on consistent 2026 gains.
- Live 2026 artifacts are rebuilt by replaying finished weekends from the seed, never hand edited. The replay restores the baseline first, replays practice into car profiles, keeps race and qualifying driver state separate, takes DNFs only from explicit FastF1 `Status`, and resolves experience tier from `debut_year`.
- The full-race grid anchor cap went from 0.62 to 0.57: the median order moved 0.41 positions from qualifying while real 2026 races moved 2.00 to 4.55. 0.56 widened the top-grid tail too far. (Model 3.0 later reworked the anchor; see `OVERTAKING_CALIBRATION_PLAN.md`.)
- Known data issue then: McLaren got a 0.0 race signal in China from incomplete telemetry.
