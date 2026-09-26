# Matched-lap extractor: smoke results

2026-05-13. Passed on local cache (`data/raw/.fastf1_cache`), the five sessions locked in `matched_lap_extractor_smoke_sessions.md`. Harness: `scripts/validate_matched_lap_smoke_sessions.py`. Output: `data/diagnostics/matched_lap_extractor_smoke/`. The extractor was not tuned to these results.

| Session | Matched | Skipped | Weather | Result |
|---|---:|---:|---|---|
| 2024 Bahrain race (clean dry) | 371 | 0 | all dry | Pass, inside 80 to 600 |
| 2024 British race (wet/mixed) | 257 | 1 | 155 dry, 102 wet | Pass with deviation |
| 2024 Australian race (early DNF) | 231 | 3 | all dry | Pass |
| 2024 Miami race (strategy split) | 256 | 0 | all dry | Pass, above the 50 floor |
| 2024 Bahrain qualifying | 28 | 2 | all dry | Pass with deviation, above the 20 floor |

Filter counts (non-green, SC/VSC, pit, stint outlier, unreliable weather): Bahrain race 42, 0, 86, 95, 0. British 0, 0, 92, 88, 24. Australian 58, 54, 70, 82, 0. Miami 130, 130, 56, 103, 0. Bahrain qualifying 0, 0, 182, 0, 0.

## Notes

- **British:** Alpine skipped with `missing_lap_time_data` because Gasly did not start. More accurate than the weather reasons the lock expected.
- **Australian:** Red Bull and Mercedes skipped with `insufficient_matched_pairs` (Verstappen and Hamilton retired early); Williams with `single_car_session` (Sargeant did not race). The Red Bull no-update case is covered.
- **Bahrain qualifying:** Alpine now gives 3 rows after segment detection was fixed to use lap start. Kick Sauber and Williams have only two valid paired push laps and are skipped, because `min_matched_pairs_quali = 3` stays. No `no_common_quali_segment` row appears.

## Follow-up

- Keep `min_matched_pairs_quali = 3`. The lock's expectation for Q1-only teams was wrong, not the threshold.
- Rerun the smoke validator after any extractor change, before a bulk run.
