# Teammate-network prior: construct audit

2026-05-17. The first prior fit failed its external validation rows. The question: do those rows measure the same thing as the local extractor?

## Answer

Mostly no for qualifying, not proven for the race. The failure is a validation-contract failure, not proof the extractor is wrong.

**Decisions (2026-05-17):**

1. The PACETEQ race and qualifying rows are now `EXTERNAL_CONTEXT`, not hard gates. Their numbers are not relaxed to make the check pass.
2. `quali_rating_mu_s` means repeatable qualifying execution for v1. Peak qualifying may return later as a separate candidate model.
3. The pipeline continues on internal reproducibility, replay stability and held-out seasons.

## What differs

**Local qualifying:** keeps quick laps (within 1.07x the segment best), pairs push laps by segment, compound, weather and run order, starts at the highest common segment and adds lower ones until there are 3 pairs, then takes the median. A repeatable-execution measure, not a single-lap peak.

**Local race:** strict pairing (same compound, weather and stint-lap index, green flag, outliers removed), then the median. Narrower than a season "race pace" figure.

**Sources:** 2024 PACETEQ compares best qualifying times. 2023 PACETEQ does not say how laps were chosen or weighted. Motor Sport Magazine (mid-2024) uses only sessions where teammates reached the same phase, closer to the local idea: VER over PER 0.486s, RUS over HAM 0.098s, ALB over SAR 0.257s.

**Weighting:** the fit weights sessions by pair count and inverse SE; sources are season averages. For VER-PER 2023, four races hold 51.8% of the weight. For ALB-SAR 2024, three races hold 95.1%, from only 5 valid dry rows.

## Qualifying rows

| Row | Source | Local WLS | Local equal mean | Best in highest common segment | Best anywhere |
|---|---:|---:|---:|---:|---:|
| VER-PER 2022 | 0.290 | 0.361 | 0.157 | 0.189 | 0.210 |
| VER-PER 2023 | 0.621 | 0.363 | 0.543 | 0.672 | 0.839 |
| VER-PER 2024 | 0.660 | 0.462 | 0.507 | 0.467 | 0.610 |
| RUS-HAM 2024 | 0.230 | 0.113 | 0.083 | 0.345 | 0.353 |
| ALB-SAR 2023 | 0.522 | 0.412 | 0.418 | 0.554 | 1.236 |
| ALB-SAR 2024 | 0.660 | 0.222 | 0.402 | 0.380 | 0.636 |

Seconds. `scripts/probe_teammate_network_constructs.py` reproduces every stored value within 1 ms from cache. Changing only the statistic moves a season by several tenths.

- **VER-PER 2023:** two published values already disagree (0.621 and 0.495), and the local value moves 0.18s between WLS and equal weighting.
- **VER-PER 2024 and RUS-HAM 2024:** the local value sits near the same-phase source and well below the best-lap source. RUS-HAM is the clearest mismatch.
- **ALB-SAR 2023:** the best-lap probe (0.554) lands near the source (0.522), as expected if the source measures best laps.
- **ALB-SAR 2024:** too thin (5 rows) and too concentrated to grade anything.

## Peak qualifying across the whole network

`scripts/probe_teammate_network_peak_constructs.py`, every stored dry qualifying pair-season:

| Measure | Current | Best in highest common segment | Best anywhere |
|---|---:|---:|---:|
| Session rows | 627 | 880 | 881 |
| Pair-seasons covered | 49 | 52 | 52 |
| Median within pair-season SD | 0.317s | 0.460s | 0.854s |
| P75 within pair-season SD | 0.412s | 0.648s | 1.307s |

The peak view adds 253 rows and three pair-seasons, but it is much noisier: 60 rows above 1s, 18 above 2s, the worst GAS-OCO Britain 2024 at -5.247s. Of its rows, 357 come from Q1, 248 from Q2, 275 from Q3, so it is not a Q3 measure either. At pair-season level it barely moves the typical pair (median shift 0.001s, max 0.490s). A peak target would need a stricter single-lap rule and its own uncertainty model first.

## Race rows

| Row | Source (s/lap) | Local equal mean | Broad valid-lap median | Broad valid-lap mean |
|---|---:|---:|---:|---:|
| VER-PER 2022 | 0.234 | 0.250 | 0.322 | 0.330 |
| VER-PER 2023 | 0.451 | 0.259 | 0.298 | 0.462 |
| VER-PER 2024 | 0.560 | 0.667 | 0.488 | 0.636 |
| ALO-STR 2023 | 0.486 | 0.364 | 0.325 | 0.372 |
| ALO-STR 2024 | 0.250 | 0.249 | 0.359 | 0.462 |
| ALB-SAR 2023 | 0.293 | 0.214 | 0.749 | 0.729 |
| ALB-SAR 2024 | 0.380 | 0.264 | 0.482 | 0.472 |

`scripts/probe_teammate_network_race_constructs.py` drops the strict pairing but keeps lap quality filters. Several rows move toward the source, ALO-STR 2023 does not. The broad view also produces nonsense when retained laps are lopsided (ALB-SAR Zandvoort 2023 at 11.082 s/lap with six Sargeant laps), which is why the paired extractor exists. ALO-STR 2024 passes in the pooled prior but its direct value (0.223) is below the source: a pooled pass can hide a mismatch.

Conclusion: the race rows carry useful context but are not proven same-construct gates.

## Next

- Every new source row records: lap unit (best lap, representative, paired, unknown), segment policy, weather policy, season aggregation, weighting, sample count, and whether it matches the local construct.
- Revisit peak qualifying as a separate candidate from the 2026 summer break (2026-07-27). The question is whether it beats the v1 model on replay stability and held-out seasons, not how to make old thresholds pass.

Evidence files: `data/diagnostics/teammate_network_construct_probe/`.
