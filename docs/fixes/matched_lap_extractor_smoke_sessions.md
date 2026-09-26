# Matched-lap extractor: smoke sessions

Locked 2026-05-12. Five sessions to catch extractor bugs before the 2022 to 2025 bulk run. They do not show the prior is calibrated; that lives in `teammate_network_prior_validation_evidence.md`. Results: `matched_lap_extractor_phase4_smoke_results.md`.

Expectations come from the read-only FastF1 inspector (`data/diagnostics/smoke_session_inspections/`) and cached lap reads, stated before any extractor code existed. Bands are wide on purpose: they catch zero rows, far too many rows, or the wrong split. Tighter bands would need the extractor itself, which is circular.

| Session | Tests | Expected |
|---|---|---|
| 2024 Bahrain race | Plain dry extraction | 80 to 600 matched rows, all dry, no skips |
| 2024 British race | Lap-level dry/wet routing | At least 20 dry rows, some wet rows, mixed or unreliable laps excluded and counted |
| 2024 Australian race | Early teammate DNF blocks an update | Red Bull skipped (`insufficient_matched_pairs`), at least 60 rows elsewhere |
| 2024 Miami race | SC/VSC and strategy splits filtered | At least 50 rows, fewer than Bahrain |
| 2024 Bahrain qualifying | Common-segment logic, `min_matched_pairs_quali = 3` | At least 20 rows, all dry |

## Why these sessions

- **Bahrain race:** 157 dry weather samples, no rain, no retirements, ten two-car teams.
- **British race:** 51 rain and 96 dry samples (about 35% rain), enough for lap-level routing. If routing proved unusable, `weather_routing_excludes_session` would apply.
- **Australian race:** VER retired on lap 4 while PER finished. Lap 1 is excluded by rule and lap 4 as his last lap, leaving at most 2 candidate laps, below `min_matched_pairs_race = 8`. This separates `insufficient_matched_pairs` from `teammate_dnf_no_matched_laps`.
- **Miami race:** VSC on laps 22 to 23, SC on laps 28 to 32, with teammates splitting strategies around them (VER stopped before, PER during; NOR under SC while PIA had already stopped). Routine SC/VSC and outlier filtering are counts, not skip reasons.
- **Bahrain qualifying:** every branch of the common-segment logic. Both drivers in Q3: Red Bull, Ferrari, McLaren, Mercedes. Both Q2 only: RB. Both out in Q1: Alpine, Kick Sauber. Split: Aston Martin and Haas (Q2 common), Williams (Q1 common). All teams share Q1, so `no_common_quali_segment` is covered by a synthetic unit test instead.

Skip reason names follow `teammate_network_prior.md` section 4.6.
