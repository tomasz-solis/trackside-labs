# 2026 Qualifying Teammate Matched-Lap Observation Coverage

Built from `scripts/build_matched_lap_observations.py --years 2026 --output-dir data/diagnostics/2026_qualifying_observation_coverage`, run offline against the local FastF1 cache. Source outputs: `raw_matched_laps.csv`, `aggregated_observations.csv`, `filter_diagnostics.csv` in that directory. Config: `min_matched_pairs_quali=3`, `matched_gap_se_floor_s=0.02`.

## Load status

12 of 23 schedule rounds have session data in the local FastF1 cache and were extracted (rounds 1-12, Australian GP through Dutch GP). Rounds 13-22 (Italian GP through Abu Dhabi GP) raised `DataNotLoadedError` for both Race and Qualifying - these are 2026 calendar rounds that had not yet been held as of 2026-09-06 (today is the Italian GP date), so the cache holds only the schedule stub, not session data. Round 23, Emilia Romagna Grand Prix, is in `get_schedule_rows`'s local fallback schedule but absent from the FastF1 event schedule the extraction script itself iterates, so it was never attempted (no data, no load error, no cache entry) - see the matrix note below. No round with a race date in the past failed to load, and no `--online` fallback was used.

## (a) Per-team, per-round matched-pair matrix

Cell = total accepted matched pairs for that team-round (summed across weather buckets when a round split dry/wet). `*` marks a round where one weather bucket was accepted and another was skipped (session had both dry and wet laps). Otherwise the cell shows the skip reason (`insuff` = `insufficient_matched_pairs`, `no_laptime` = `missing_lap_time_data`).

Round legend: R1=Australian Grand Prix R2=Chinese Grand Prix R3=Japanese Grand Prix R4=Miami Grand Prix R5=Canadian Grand Prix R6=Monaco Grand Prix R7=Barcelona Grand Prix R8=Austrian Grand Prix R9=British Grand Prix R10=Belgian Grand Prix R11=Hungarian Grand Prix R12=Dutch Grand Prix

No round data: R13 Italian Grand Prix, R14 Spanish Grand Prix, R15 Azerbaijan Grand Prix, R16 Singapore Grand Prix, R17 United States Grand Prix, R18 Mexico City Grand Prix, R19 São Paulo Grand Prix, R20 Las Vegas Grand Prix, R21 Qatar Grand Prix, R22 Abu Dhabi Grand Prix (2026 rounds not yet held as of 2026-09-06 - see Load status). Emilia Romagna Grand Prix (R23) is in `get_schedule_rows`'s local fallback schedule but is not in the FastF1 event schedule the extractor itself queries, so it was never attempted at all - it has neither data nor a load error.

| Team | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alpine | 5 | 4 | 4 | 3 | 4 | 4 | 4 | 3 | insuff | 4 | 3 | 3 |
| Aston Martin | no_laptime | 3 | 3 | insuff | 3 | 3 | insuff | insuff | insuff | insuff | insuff | 3 |
| Audi | 4 | 3 | 5 | insuff | 3 | 3 | 3 | 4 | 3 | 4 | 3 | insuff |
| Cadillac | insuff | insuff | insuff | insuff | insuff | 3 | insuff | 3 | insuff | 3 | 3 | 3 |
| Ferrari | 3 | 4 | 4 | 4 | 3 | 4 | insuff | 3 | 4 | 3 | 3 | 3 |
| Haas F1 Team | 6 | 4 | 3 | 4 | 4 | 4 | 3 | 4 | insuff | insuff | insuff | insuff |
| McLaren | 4 | 4 | 4 | 4 | 3 | 4 | 3 | 3 | 4 | 3 | 3 | 3* |
| Mercedes | 3 | 3 | 4 | 3 | 4 | 5 | 3 | 3 | 4 | 3 | 4 | 3 |
| Racing Bulls | 3 | 4 | 4 | insuff | 6 | 3 | insuff | 4 | 3 | 4 | 3 | 5 |
| Red Bull Racing | insuff | 4 | 4 | 4 | 4 | 4 | 3 | 3 | 4 | insuff | 3 | 3* |
| Williams | no_laptime | 3 | 3 | 4 | 3 | 3 | 3 | 3 | 3 | insuff | 3 | insuff |

## (b) Per-team summary (12 rounds with data)

| Team | Total matched-pair rows | Rounds accepted (>=3 pairs) | Rounds skipped | Skip reasons |
|---|---|---|---|---|
| Alpine | 41 | 11/12 | 1/12 | insufficient_matched_pairs=1 |
| Aston Martin | 15 | 5/12 | 7/12 | insufficient_matched_pairs=6, missing_lap_time_data=1 |
| Audi | 35 | 10/12 | 2/12 | insufficient_matched_pairs=2 |
| Cadillac | 15 | 5/12 | 7/12 | insufficient_matched_pairs=7 |
| Ferrari | 38 | 11/12 | 1/12 | insufficient_matched_pairs=1 |
| Haas F1 Team | 32 | 8/12 | 4/12 | insufficient_matched_pairs=4 |
| McLaren | 43 | 12/12 | 0/12 | - |
| Mercedes | 42 | 12/12 | 0/12 | - |
| Racing Bulls | 39 | 10/12 | 2/12 | insufficient_matched_pairs=2 |
| Red Bull Racing | 37 | 10/12 | 2/12 | insufficient_matched_pairs=2 |
| Williams | 28 | 9/12 | 3/12 | insufficient_matched_pairs=2, missing_lap_time_data=1 |

## (c) Skip-reason frequency, whole season (134 team-round-weather rows, 12 rounds with data)

| Outcome | Count | Share |
|---|---|---|
| (accepted) | 103 | 76.9% |
| insufficient_matched_pairs | 29 | 21.6% |
| missing_lap_time_data | 2 | 1.5% |

## (d) Distribution of matched-pair counts per team-session

Two views, both derived from this run's own output files (no new extractor):

**View 1 - output-level (`aggregated_observations.csv`, 134 team-round-weather-bucket rows).** This is what `raw_matched_laps.csv` / the aggregate table actually emit. It undercounts near-misses: when a whole qualifying session already has fewer than 3 candidate matches, `_qualifying_pair_rows` returns before emitting any raw rows, so the aggregate placeholder records `n_matched_pairs=0` regardless of whether 0, 1, or 2 candidates actually existed.

| Pairs | Team-sessions | Share |
|---|---|---|
| 0 | 29 | 21.6% |
| 1 | 2 | 1.5% |
| 2 | 0 | 0.0% |
| 3 | 57 | 42.5% |
| 4 | 40 | 29.9% |
| 5+ | 6 | 4.5% |

**View 2 - pre-gate candidate count (`filter_diagnostics.csv`, 132 team-round rows, one row per team per round, computed session-wide before the 3-pair gate and before any weather-bucket split).** This is the real near-miss distribution and the number that answers what a lower gate would recover.

| Candidate pairs | Team-sessions | Share |
|---|---|---|
| 0 | 4 | 3.0% |
| 1 | 2 | 1.5% |
| 2 | 23 | 17.4% |
| 3 | 55 | 41.7% |
| 4 | 42 | 31.8% |
| 5+ | 6 | 4.5% |

**Key number:** 23 of 132 qualifying team-sessions (17.4%) sit at exactly 2 candidate matched pairs - one lap pair short of the current gate of 3. Lowering the gate from 3 to 2 would recover those 23 team-sessions. It would not touch the 6 team-sessions stuck at 0-1 candidates, which fail for a different reason (missing lap-time data, no common quali segment, or genuinely too few valid push laps).

## (e) What this shows

- `insufficient_matched_pairs` accounts for 27 of 29 skipped team-sessions (93%), so it dominates the skip reasons over the 12 extracted rounds. The remainder is `missing_lap_time_data` (2).
- Skips are concentrated on the slower group: teams in {Aston Martin, Cadillac, Williams, Haas F1 Team, Audi, Racing Bulls} were skipped in 35% of their team-rounds (25/72), versus 7% (4/60) for teams in {McLaren, Ferrari, Mercedes, Red Bull Racing, Alpine}.
- Per-team skip rate, all 132 team-round rows:
  - Aston Martin: 7/12 rounds skipped (58%)
  - Cadillac: 7/12 rounds skipped (58%)
  - Haas F1 Team: 4/12 rounds skipped (33%)
  - Williams: 3/12 rounds skipped (25%)
  - Red Bull Racing: 2/12 rounds skipped (17%)
  - Racing Bulls: 2/12 rounds skipped (17%)
  - Audi: 2/12 rounds skipped (17%)
  - Alpine: 1/12 rounds skipped (8%)
  - Ferrari: 1/12 rounds skipped (8%)
  - McLaren: 0/12 rounds skipped (0%)
  - Mercedes: 0/12 rounds skipped (0%)
- McLaren and Mercedes were never skipped across the 12 extracted rounds. Aston Martin and Cadillac were skipped in just over half their rounds. Red Bull Racing (in the faster group by car pace) still shows a 17% skip rate, tied with Audi and Racing Bulls - a reminder this is a matched-lap-count effect (how many comparable push laps both teammates set), not a pure car-pace ranking.

## Step 3 - `_bootstrap_median_se` at n=2 and n=3

`matched_gap_se_floor_s` = 0.02. `_bootstrap_median_se` special-cases `len(gaps) < 2` to return the floor directly, but does not special-case n=2 or n=3; both go through the full bootstrap (`bootstrap_samples=1000`, `bootstrap_random_seed=2026`) then apply `max(se, floor)`.

No qualifying team-session in this run ever reaches raw output with exactly 2 matched pairs (the extractor drops sub-gate candidates before emitting rows - see part (d)). So the n=2 vectors below are the first 2 laps of a real, accepted 3-pair group; the n=3 vectors are the same group's full 3 gaps.

| Team / Round / Weather | n=2 gaps (s) | n=2 SE | n=3 gaps (s) | n=3 SE |
|---|---|---|---|---|
| Alpine / Austrian Grand Prix / dry | [-0.305, 0.245] | 0.1933 | [-0.305, 0.245, -0.609] | 0.3135 |
| Alpine / Dutch Grand Prix / dry | [0.015, -0.207] | 0.0780 | [0.015, -0.207, -0.106] | 0.0789 |
| Alpine / Hungarian Grand Prix / dry | [0.268, -0.030] | 0.1047 | [0.268, -0.030, -0.183] | 0.1653 |
| Alpine / Miami Grand Prix / dry | [0.186, -0.053] | 0.0840 | [0.186, -0.053, 0.226] | 0.1116 |
| Aston Martin / Canadian Grand Prix / dry | [0.947, 0.438] | 0.1789 | [0.947, 0.438, 0.718] | 0.1811 |

Across these 5 sampled groups: n=2 SE bound at the 0.02 floor in 0/5 cases (range 0.0780-0.1933); n=3 SE bound at the floor in 0/5 cases (range 0.0789-0.3135). For a 2-point sample, the bootstrap median always resolves to one of the two input values, so its spread is mechanically wide relative to 2 points and the floor binds only when the two gaps happen to be very close together; with real qualifying gaps (order 0.1-1s apart) the unfloored bootstrap SE for n=2 is usually well above the 0.02s floor. At n=3 the same pattern holds for widely spread gaps, but tighter 3-lap groups can fall to or below the floor, which is exactly the case it exists to catch.
