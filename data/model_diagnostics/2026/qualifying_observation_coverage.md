# 2026 qualifying matched-lap coverage

Built with `scripts/build_matched_lap_observations.py --years 2026 --output-dir data/diagnostics/2026_qualifying_observation_coverage`, offline, from the local FastF1 cache. Outputs in that folder: `raw_matched_laps.csv`, `aggregated_observations.csv`, `filter_diagnostics.csv`. Config: `min_matched_pairs_quali=3`, `matched_gap_se_floor_s=0.02`.

## Load status

Rounds 1 to 12 (Australia to the Dutch GP) had session data and were extracted. Rounds 13 to 22 (Italy to Abu Dhabi) had not been held on 2026-09-06, so they raised `DataNotLoadedError`. Round 23 (Emilia Romagna) is only in the local fallback schedule, not in the FastF1 schedule the script reads, so it was never attempted. No past round failed and `--online` was not used.

## (a) Per-team, per-round matched-pair matrix

Each cell is the accepted matched pairs for that team and round (summed over weather buckets). `*` marks a round where one weather bucket was accepted and another skipped. Otherwise the cell shows the skip reason (`insuff` = `insufficient_matched_pairs`, `no_laptime` = `missing_lap_time_data`).

Round legend: R1=Australian Grand Prix R2=Chinese Grand Prix R3=Japanese Grand Prix R4=Miami Grand Prix R5=Canadian Grand Prix R6=Monaco Grand Prix R7=Barcelona Grand Prix R8=Austrian Grand Prix R9=British Grand Prix R10=Belgian Grand Prix R11=Hungarian Grand Prix R12=Dutch Grand Prix

No data: R13 to R22 (not yet held on 2026-09-06). R23 Emilia Romagna was never attempted (see above).

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

Two views from this run's own output files:

**View 1: output rows** (`aggregated_observations.csv`, 134 team-round-weather rows). This undercounts near misses: a session with fewer than 3 candidate pairs emits no raw rows, so its placeholder shows `n_matched_pairs=0` whether it had 0, 1 or 2 candidates.

| Pairs | Team-sessions | Share |
|---|---|---|
| 0 | 29 | 21.6% |
| 1 | 2 | 1.5% |
| 2 | 0 | 0.0% |
| 3 | 57 | 42.5% |
| 4 | 40 | 29.9% |
| 5+ | 6 | 4.5% |

**View 2: candidates before the gate** (`filter_diagnostics.csv`, 132 team-round rows, counted before the 3-pair gate and before the weather split). This is the real near-miss distribution.

| Candidate pairs | Team-sessions | Share |
|---|---|---|
| 0 | 4 | 3.0% |
| 1 | 2 | 1.5% |
| 2 | 23 | 17.4% |
| 3 | 55 | 41.7% |
| 4 | 42 | 31.8% |
| 5+ | 6 | 4.5% |

**Key number:** 23 of 132 team-sessions (17.4%) have exactly 2 candidate pairs, one short of the gate. A gate of 2 would recover them. The 6 sessions at 0 or 1 fail for other reasons (missing lap times, no common segment, too few push laps).

## (e) What this shows

- `insufficient_matched_pairs` is 27 of 29 skips (93%); the other 2 are `missing_lap_time_data`.
- Skips hit the slower teams: Aston Martin, Cadillac, Williams, Haas, Audi and Racing Bulls were skipped in 35% of their team-rounds (25/72), against 7% (4/60) for McLaren, Ferrari, Mercedes, Red Bull and Alpine.
- Skip rate by team, 132 team-rounds:
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
- McLaren and Mercedes were never skipped; Aston Martin and Cadillac in over half their rounds. Red Bull, a fast car, still sits at 17% with Audi and Racing Bulls: the effect is about how many comparable push laps both teammates set, not pure car pace.

## `_bootstrap_median_se` at n=2 and n=3

`matched_gap_se_floor_s` is 0.02. Samples below 2 return the floor directly; n=2 and n=3 go through the full bootstrap (1000 samples, seed 2026), then `max(se, floor)`.

No session reaches the output with exactly 2 pairs (the extractor drops them first), so the n=2 columns below use the first 2 laps of a real 3-pair group, and the n=3 columns the full group.

| Team / Round / Weather | n=2 gaps (s) | n=2 SE | n=3 gaps (s) | n=3 SE |
|---|---|---|---|---|
| Alpine / Austrian Grand Prix / dry | [-0.305, 0.245] | 0.1933 | [-0.305, 0.245, -0.609] | 0.3135 |
| Alpine / Dutch Grand Prix / dry | [0.015, -0.207] | 0.0780 | [0.015, -0.207, -0.106] | 0.0789 |
| Alpine / Hungarian Grand Prix / dry | [0.268, -0.030] | 0.1047 | [0.268, -0.030, -0.183] | 0.1653 |
| Alpine / Miami Grand Prix / dry | [0.186, -0.053] | 0.0840 | [0.186, -0.053, 0.226] | 0.1116 |
| Aston Martin / Canadian Grand Prix / dry | [0.947, 0.438] | 0.1789 | [0.947, 0.438, 0.718] | 0.1811 |

Across these 5 groups the floor never binds: n=2 SE ranges 0.0780 to 0.1933, n=3 SE 0.0789 to 0.3135. A 2-point bootstrap median is always one of the two values, so its spread is wide, and with real qualifying gaps (0.1 to 1 s apart) it sits well above the 0.02 floor. The floor catches tight 3-lap groups.
