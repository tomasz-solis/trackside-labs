# Driver seconds and wet trace

Started 2026-05-21. Steps 1 to 4 are done. Step 5 waits for a real wet 2026 race.

Two gaps in Model Diagnostics:

1. Dry leakage was measured on the legacy `bayesian.rating_mu`, which is not in seconds.
2. There was no wet 2026 replay sample and no session-level trace to prove that a wet race leaves dry driver state alone.

## Done when

- **Dry:** `corr(delta_race_rating_mu_s, delta_team_strength_for_driver_team)` is computed from real seconds fields that the updater moves.
- **Wet, in code:** the updater writes session trace rows and diagnostics prove a fully wet session changes dry state by zero.
- **Wet, in 2026:** a real wet replay sample passes that check. Until then the dashboard keeps saying 2026 wet coverage is missing.

## Steps

**1. Wet trace.** Done except historical wet tests.

- Fully wet race, qualifying and sprint updates no longer move dry state.
- Trace rows (session, weather route, dry state and wet skill before and after, applied flags) are written by the updater, carried through replay and checked by the diagnostics. A synthetic failure test exists.
- Open: tests on historical wet evidence.

**2. Seconds schema.** Done.

- Driver artifacts accept `race_rating_*` and `quali_rating_*` alongside the old fields, through files and Supabase.
- A migration script seeds seconds fields from the teammate-network prior and fails on a missing active driver instead of converting `rating_mu`.
- LIN had no prior node. Drivers like that get a rookie fallback: the median debut-season seconds estimate, with wide uncertainty, recorded as a fallback. Live updates replace it after 24 construct-aligned observations (`min_driver_observations`).

**3. Live cutover.** Done.

- Race and qualifying seconds update separately and never overwrite each other.
- Readers prefer seconds fields and fall back to legacy fields when a driver has no complete seconds state.
- If FastF1 cannot supply laps or weather for the matched-lap construct, the update is skipped with a warning; seconds are never derived from positions.
- Sprints: `auto_update_from_races()` runs `update_from_sprint_race()` before the main race, and replay does the same. SQ moves qualifying seconds, the sprint moves race seconds, each at half weight.
- Artifacts, replay and warmup were rebuilt and synced to Supabase with read-back checks.

**4. Leakage diagnostic.** Done. Dry leakage now uses `delta_race_rating_mu_s` and is synced to Supabase.

**5. Real wet 2026.** Open. When a wet race happens: rebuild matched-lap observations, replay the traced update and regenerate diagnostics.

## Rules

- Never hide missing evidence by suppressing a diagnostics message.
- Never infer seconds fields from `rating_mu`.
- Keep the legacy fallback until the removal rule in `master_execution_plan.md` is met (three clean production weekends).
