# Teammate-network prior: validation evidence

Locked 2026-05-09, amended 2026-05-17. The external checks that grade the prior, written before the fit so the result cannot be rationalised afterwards.

**Current state: 0 hard rows.** After the construct audit (`teammate_network_prior_construct_audit.md`), every PACETEQ row is `EXTERNAL_CONTEXT`: useful, but not proven to measure what the extractor measures. Hard validation is provisional for race and qualifying, and internal checks (reproducibility, replay stability, held-out seasons) carry the load.

## Evidence tiers

| Tier | Meaning | Counts toward pass/fail |
|---|---|---|
| HARD | Independent seconds delta, stated method, same construct | Yes |
| EXTERNAL_CONTEXT | Independent seconds delta, construct not proven to match | No, reported separately |
| SUPPLEMENTAL | Model-derived (F1Metrics-style) or too near zero | No, reported separately |
| SMOKE_ONLY | Direction only, lives in `tests/test_prior_signs.py` | No, not reported |
| CUT | Researched and rejected, reason recorded | No |

## Source rules

A hard source must publish a teammate gap in seconds (or convertible without an undocumented factor), state the construct (race, qualifying, single lap, long run, session best) and the sample (sessions, lap filters), be teammate-relative, and match what the prior estimates. Judge each article on its method, not the outlet's reputation.

Conditionally usable when the translation is mechanical and written down: per-race deltas aggregated to a season median, a season chart matched to the prior's lap window, a percentage converted with a stated reference lap.

Rejected: unstated method; ratings on their own scale (Elo, 0 to 100); a model estimating the same quantity with no independent observations; marketing (AWS "Fastest Driver", team PR); memory, "common knowledge" or social media; this project's own output; an AI summary instead of the source.

**F1Metrics-style projects** are supplemental only. They may corroborate a hard source but never are one. A row with only such a source is cut.

**PACETEQ** (Motorsport.com, Motorsport-Total): external context, not hard, unless an article proves a same-construct match. Several PACETEQ articles are one source family, not independent corroboration. Gaps near 0.01 s/lap are never load-bearing.

**Construct mismatches to watch:** race vs qualifying pace; teammate gap vs global ranking; one season vs several seasons pooled; model rating vs timing delta; broadcast numbers without a named lap window.

## Rows

Race (s/lap, pass rule `A_mu_s - B_mu_s >= threshold`, all accessed 2026-05-12):

| Row | Threshold | Source | Tier |
|---|---:|---|---|
| VER-PER 2022 | 0.234 | PACETEQ Perez trend | External context |
| VER-PER 2023 | 0.451 | PACETEQ 2023 review | External context |
| VER-PER 2024 | 0.56 | PACETEQ Red Bull duel | External context |
| ALO-STR 2023 | 0.486 | PACETEQ 2023 review | External context |
| ALO-STR 2024 | 0.25 | PACETEQ Aston Martin duel | External context |
| ALB-SAR 2023 | 0.293 | PACETEQ 2023 review | External context |
| ALB-SAR 2024 | 0.38 | PACETEQ Williams duel (Sargeant's 2024 starts) | External context |
| BOT-ZHO 2024 | 0.01 | PACETEQ Sauber duel | Supplemental (too near zero) |

Qualifying (seconds):

| Row | Threshold | Source | Other published values | Tier |
|---|---:|---|---|---|
| VER-PER 2022 | 0.290 | PACETEQ Perez trend | | External context |
| VER-PER 2023 | 0.621 | PACETEQ 2023 review | RacingNews365 0.495 | External context |
| VER-PER 2024 | 0.66 | PACETEQ Red Bull duel | Motor Sport Magazine 0.486 (part season) | External context |
| RUS-HAM 2024 | 0.23 | PACETEQ Mercedes duel | Motor Sport Magazine 0.098 (part season) | External context |
| ALB-SAR 2023 | 0.522 | PACETEQ 2023 review | | External context |
| ALB-SAR 2024 | 0.66 | PACETEQ Williams duel | | External context |

Cut:

| Row | Reason |
|---|---|
| RUS-LAT race 2022 | Impossible pairing: Russell drove for Mercedes in 2022 |
| BOT-ZHO race 2022 | No numeric race-pace source |
| BOT-ZHO race 2023 | Source has Zhou 0.013 s/lap faster, against the expected direction |
| TSU-DEV race 2023 | About 10 races, too thin (smoke test only) |
| LEC-SAI qualifying 2022 to 2024 | Direction contested; a hedged check is a smoke test (smoke test only) |
| Multi-season rows (VER-PER, ALO-STR, ALB-SAR) | Replaced by one row per season |

## Source URLs

- PACETEQ Perez trend: https://lat.motorsport.com/f1/news/checo-perez-diferencia-verstappen-f1-2024/10627633/
- PACETEQ 2023 review: https://lat.motorsport.com/f1/news/verstappen-checo-perez-diferencia-f1-2023/10561671/
- PACETEQ Red Bull duel: https://www.motorsport-total.com/formel-1/news/maximal-ueberlegen-wie-verstappen-perez-2024-in-grund-und-boden-fuhr-24122902
- PACETEQ Aston Martin duel: https://www.motorsport-total.com/formel-1/news/analyse-ist-lance-stroll-wirklich-zu-langsam-24122701
- PACETEQ Williams duel: https://www.motorsport-total.com/formel-1/news/nach-sargeant-rauswurf-so-viel-schneller-war-franco-colapinto-wirklich-24122307
- PACETEQ Sauber duel: https://www.motorsport-total.com/formel-1/news/sauber-duell-das-war-2024-die-ganz-grosse-schwaeche-von-valtteri-bottas-24122202
- PACETEQ Mercedes duel: https://www.motorsport-total.com/formel-1/news/mercedes-fahrer-analysiert-hat-lewis-hamilton-seine-qualifyingpace-verloren-24122802

## Adding a row

1. Read the source rules first.
2. Candidate links go in `phase_1_source_research.md`, which is notes only and never evidence.
3. For each row: accept with source, type, threshold, pass rule and access date; or record the translation; or cut it with a reason. Never leave a row open, and never delete one silently.
4. Do not chase a row count. Cut what cannot be sourced; widen sigma and tighten replay checks instead of inventing thresholds.
5. Never take a threshold from this project's own fit.

## Report format

The fit report lists hard race and qualifying checks passed and failed with sources, then external context and supplemental rows separately, the cut rows with reasons, and whether hard validation is provisional and what carries the load instead. Smoke tests are not counted.
