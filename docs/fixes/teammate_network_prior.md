# Teammate-network prior: design

Locked 2026-05-09. Defines the historical prior that puts drivers on one cross-team scale, and the matched-lap data contract shared by the prior builder and the live updater. Gates: `teammate_network_prior_validation_evidence.md` and `matched_lap_extractor_smoke_sessions.md`. Order of work: `master_execution_plan.md`.

## 1. Purpose

Four separate signals:

| Signal | Meaning |
|---|---|
| `team_strength` | Car pace shared by both drivers |
| `race_rating_mu_s` | Driver's dry race residual, seconds |
| `quali_rating_mu_s` | Driver's dry qualifying residual, seconds |
| `wet_skill` | Wet vs dry change in teammate-relative pace |

A teammate comparison says A beat B. It cannot place both on a grid-wide scale. That comes from the historical network: drivers who changed teams link the teams. The prior is built offline, validated, versioned and never refitted during a season.

**Update order within a session:**

1. Build clean per-driver lap observations.
2. Update `team_strength` from team vs field evidence (used for the next forecast).
3. Compute driver residuals against the observed same-session team median: `observed_team_median_s - observed_driver_median_s`. Never against predicted or posterior team strength, which would leak car and mapping error into driver ratings.
4. Update race, qualifying or wet ratings by session type and weather.

**In-season learning** runs mainly through `team_strength`. Driver ratings move slowly with shrinkage; one weekend should not jump a rating. A full network refit is an offline job (season break or summer break). 2026 artifacts are rebuilt only by replaying session data, never hand edited. If a result looks wrong, audit the input or the updater rule.

**Known limits.**

- If both teammates improve together, the live updater sees it as car improvement. Only the historical network can separate it. So "car changed, ratings stayed" tests work on synthetic data only; real checks are indirect (residual diagnostics, fold stability, sensitivity to the extractor settings).
- Validation sources lean toward big teammate gaps, where bias matters least. Widen sigma on close pairs only if internal diagnostics justify it, and record the decision either way.

## 2. Scope

Covers: historical session scope, the extractor, the race and qualifying models, connected components, weighting, the output artifact and validation gates. Owned elsewhere: schema migration, live updater, wet-skill implementation, prediction blending, replay harness, live Bayesian calibration, the seconds mapping code, the test rewrite.

## 3. Historical scope

F1 2022 to 2025. 2022 starts the previous rules era; before that is a different car generation. 2026 is a transfer check only, never a fit input.

Separate race and qualifying observation sets. Excluded from the dry prior: wet and mixed sessions (they feed `wet_skill`), sprint and sprint qualifying (maybe later, lower weight), practice (run programmes are not comparable). Practice stays valid live evidence for car features and team strength. Pre-season testing can hint at car features but must never set a manufacturer order.

## 4. Matched-lap extractor

The single observation pipeline for the prior and the live updater.

### 4.1 Contract

```python
def extract_matched_teammate_laps(
    session: fastf1.core.Session,
    *,
    session_kind: Literal["race", "qualifying"],
    weather_mode: Literal["dry", "wet", "mixed", "unknown"],
    config: MatchedLapConfig,
) -> pd.DataFrame:
    """Return canonical teammate matched-lap observations for one session."""
```

One row per matched lap pair (`row_type = "matched_pair"`), never one per driver. A pair skipped before any match gets one `skipped_pair` row with a non-empty `skip_reason`.

Columns: `row_type`, `year`, `race_name`, `session_name`, `session_kind`, `team`, `reference_driver_code`, `comparison_driver_code`, `reference_lap_number`, `comparison_lap_number`, `reference_lap_time_s`, `comparison_lap_time_s`, `matched_gap_s`, `compound`, `reference_stint`, `comparison_stint`, `stint_lap_index`, `weather_bucket`, `track_status_bucket`, `reference_position_start`, `reference_position_end`, `comparison_position_start`, `comparison_position_end`, `skip_reason`.

Sign: `matched_gap_s = comparison_lap_time_s - reference_lap_time_s`. Positive means the reference driver was faster. The reference driver is first alphabetically by code; the order has no meaning beyond avoiding duplicates.

### 4.2 Config

```yaml
matched_laps:
  min_matched_pairs_race: 8
  min_matched_pairs_quali: 3
  max_position_change_for_clean_lap: 2
  traffic_stint_sigma_threshold: 1.5
  tire_age_fallback_window_laps: 3
```

Qualifying needs 3 pairs because a median of two points has no meaningful standard error. Accepting 2 would need its own SE rule.

### 4.3 Race matching

Match lap pairs directly, not driver median minus teammate median, so uneven lap coverage cannot bias the gap. A valid pair has: same team, same compound, same stint-lap index, green flag, weather samples matching the target bucket, no pit in or out, not lap 1, not either driver's last classified lap, a valid non-outlier lap time, at most `max_position_change_for_clean_lap` places gained or lost in the lap, and no lap slower than the stint median plus `traffic_stint_sigma_threshold` sigma.

The tyre-age fallback is used only when strict matching finds too few pairs and the smoke checks show it adds no strategy noise. If one teammate has too few samples (for example an early DNF), neither driver gets a race-rating update.

### 4.4 Qualifying matching

Use dry push laps from segments both teammates reached, from Q3 down to Q1. Within each (segment, compound), pair laps by run order, not by lap time rank. Add segments until `min_matched_pairs_quali` is reached; otherwise skip the pair. Exclude deleted, inaccurate, missing, pit and non-green laps. Wet or mixed qualifying never updates dry ratings.

### 4.5 Weather routing

Map `session.weather_data` onto each lap. All samples dry: dry lap. All wet: wet lap. Mixed, missing or unmappable: feeds nothing in v1. Dry ratings use dry laps, `wet_skill` uses wet laps. In a mixed session, dry ratings update only from reliable dry laps; if lap routing fails for the session, skip dry updates.

### 4.6 Skip reasons

Fixed strings, asserted by tests and traces:

```text
single_car_session
team_driver_set_ambiguous
teammate_dnf_no_matched_laps
weather_routing_excludes_session
lap_level_weather_unreliable
insufficient_matched_pairs
no_compound_overlap
no_common_quali_segment
all_laps_filtered_out
missing_lap_time_data
track_status_excluded_all_laps
```

`lap_level_weather_unreliable` is per lap. `weather_routing_excludes_session` is a session-level decision.

## 5. Aggregation

The fitter takes one row per teammate pair per session, never two mirrored driver rows (they are the same evidence). `matched_gap_median_s` is the median over matched lap pairs, `matched_gap_se_s` comes from a bootstrap over those pairs. If the bootstrap is unstable, mark the row low confidence or skip it; never emit a tiny SE.

Aggregated columns: `reference_driver_code`, `comparison_driver_code`, `team`, `year`, `race_name`, `session_name`, `session_kind`, `matched_gap_median_s`, `matched_gap_se_s`, `n_matched_pairs`, `weather_bucket`, `skip_reason`.

The live updater may split a pair into `+gap/2` and `-gap/2` per driver. Those split rows never enter the prior fit.

## 6. Model

Separate race and qualifying models: `y_i = theta_reference - theta_comparison + epsilon_i`, with `y_i = matched_gap_median_s` and `theta` each driver's residual skill in seconds. Fit jointly with the sum of `theta` equal to zero inside each connected component. Weighted least squares with heteroskedasticity-aware errors and a cluster bootstrap by session and team pair. A hierarchical Bayesian model is a later upgrade.

Output fields: `race_rating_mu_s`, `race_rating_sigma_s`, `quali_rating_mu_s`, `quali_rating_sigma_s`. Positive means faster than the average driver.

## 7. Connected components

Build the teammate graph first (drivers as nodes, valid observations as edges). If one component holds at least 90% of observations and 80% of relevant drivers, it is the anchored main component. Small components are centred at zero with inflated sigma. If several large components exist, stop: never relax thresholds just to connect the graph, unless the smoke checks show strict rules drop valid laps. Recentring in-season ratings is housekeeping, not cross-team identification.

## 8. Weights and uncertainty

`weight_i = capped_effective_n_i / max(matched_gap_se_s_i^2, se_floor^2)`

```yaml
prior:
  race_sigma_floor_s: 0.05
  quali_sigma_floor_s: 0.10
  min_driver_observations: 24   # 3 x min_matched_pairs_race
```

With `sd` the spread of the main component's ratings:

- Main component, fewer than 24 observations: `max(1.75 * sd, floor)`
- Main component, 24 or more: `max(bootstrap_sigma, 0.5 * sd, floor)`
- Small component: `max(1.75 * sd, floor)`
- Unanchored: `max(2.00 * sd, floor)`

## 9. Output artifact

`data/processed/teammate_network_prior/{built_at}.json` and `latest.json`. Top level: `built_at`, `config` (historical scope 2022 to 2025, matched-lap configs, 1000 bootstrap replicates, sigma floors, `min_driver_observations`), `race_network` and `quali_network` (each with `drivers`, `components`, `fit_diagnostics`) and `validation` (`source_backed_checks`, `all_hard_checks_passed`).

Each driver: `mu_s`, `sigma_s`, `n_observations`, `n_teammate_partners`, `component_id`, `component_anchored`, `first_session`, `last_session`.

## 10. Seconds and the team mapping

Driver ratings are native seconds; there is no driver-to-seconds mapping and there should not be one unless replay shows a stable nonlinearity nothing else explains. Only team strength is mapped:

```text
observed_driver_to_field_s  = field_median_s - driver_median_s
predicted_driver_to_field_s = team_strength_to_seconds(session_kind, team_strength) + driver_rating_mu_s
```

Race and qualifying get separate mappings over one stored `team_strength` scalar (centred at 0.5). Splitting the scalar into short-run and long-run states lost on combined MSE (0.5049 shared vs 0.5077 split), won 2 of 4 folds and no qualifying fold, so v1 keeps one scalar. Mappings are fitted once per model version; in-season learning moves team strength, not the slope.

### 10.5 Wet skill

```text
wet_skill_observation_s = (theta_ref_wet - theta_comp_wet) - (race_rating_mu_s(ref) - race_rating_mu_s(comp))
```

Positive means the reference driver gained in the wet relative to dry expectation. At prediction: `+ wet_context_weight * wet_skill_mu_s`, with weight 0 dry, 1 fully wet and a recorded fraction for mixed. Fully wet sessions update only `wet_skill`.

## 11. Regulation-reset monitoring

The frozen seconds scale assumes history transfers to 2026. Monitor rolling predicted vs observed team deltas (correlation, slope, R squared vs the historical fit) and per-driver residual means, on a dashboard monitoring tab that reads the same persisted artifacts as the background jobs. Sustained drift means a one-time refit between versions, never continuous refitting. (2026 did drift; the mapping was refitted on 2026 on 2026-08-04.)

## 12. Validation gates

- **Magnitude checks** (`teammate_network_prior_validation_evidence.md`): a check is hard only with a named comparison, scope, threshold in seconds, citation, source type, pass rule and access date. Direction-only checks are unit tests in `tests/test_prior_signs.py`. After the 2026-05-17 audit: 13 PACETEQ rows as context, one supplemental near-zero row, no same-construct hard rows.
- **Smoke sessions** (`matched_lap_extractor_smoke_sessions.md`): locked 2026-05-12.

## 13 to 16. Checks after extraction

- **Bulk dump before fitting:** pair counts by season, session and team; gap and SE distributions (cap tiny SEs); skip-reason counts; sessions with no observations; pair coverage; components; weather and compound counts.
- **Folds:** train on three of 2022 to 2025, validate on the fourth. Not random race holdouts; races in a season share too much.
- **Orthogonality:** shared movement goes to `team_strength`, teammate movement to the ratings, wet advantage to `wet_skill`. Fully wet sessions give zero dry updates (hard rule). Mixed sessions: `abs(corr(wet_skill, delta_race_rating_mu_s)) <= 0.20`. Dry leakage `corr(delta_race_rating_mu_s, delta_team_strength_for_driver_team)` is measured but has no pass threshold yet, and must not be compared with the old `rating_mu`.
- **Replay:** per-driver mean residual. A sustained non-zero mean flags that driver's rating for review.

## 17. Migration order

1. Schema accepts old and new fields (`_DRIVER_BAYESIAN_SCHEMA` had `additionalProperties: False`).
2. Writers write new fields.
3. Readers prefer new fields and fall back to old.
4. Local and Supabase artifacts migrate with rollback snapshots.
5. Old fields go only after validation.

Every reader moves together: validators, warmup, dashboard, reports, checkpoint reconstruction, local and Supabase storage, with the same field names, units and version metadata.
