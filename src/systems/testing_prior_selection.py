"""Per team and run profile, choose last season's testing profile or this season's own.

A seeded season (``src/systems/season_rollover.py``) starts every team's testing profiles
from the previous season and blends new sessions into them. A car that changed a lot over
the winter stays blurred by that prior for weeks. So each session is also kept as a plain
running mean of this season only ("reset"). A team drops the prior for one profile once
its sessions contradict it beyond session noise. Q reads the short-run and balanced
profiles, the race the long-run one.

Scoring "which option predicted the next session better" missed a faked Audi jump on the
short-run profile in the real 2026 sessions: one session swings about 0.3 on the 0 to 1
pace scale. The level test below caught it in every profile from the last testing day.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)

PROFILES = ("short_run", "balanced", "long_run")
SCORE_METRIC = "overall_pace"
# Reset after 3 sessions whose mean sits 3 noise units from the prior. Noise is the
# median-based spread of every team's sessions around its own mean, so one team that
# changed a lot cannot hide by inflating it. z of 2 and 3 scored the same on 2026 Q and R
# pace; 3 resets fewer teams.
MIN_SESSIONS = 3
MIN_ABS_Z = 3.0


def track_session(
    team_data: dict[str, Any],
    profile: str,
    observed: dict[str, Any],
    prior_before: dict[str, Any],
    session_count: int,
) -> None:
    """Record one session for this season's own profile.

    ``prior_before`` is the stored (prior-blended) profile before this session is
    blended in; on the first session it is still last season's value, kept as the prior.
    """
    reset = team_data.setdefault("testing_reset", {}).setdefault(
        profile, {"metrics": {}, "sessions": 0, "observed": []}
    )
    prior = prior_before.get(SCORE_METRIC)
    if reset["sessions"] == 0 and isinstance(prior, int | float):
        reset["prior"] = round(float(prior), 4)
    observed_pace = observed.get(SCORE_METRIC)
    if isinstance(observed_pace, int | float):
        reset["observed"].append(round(float(observed_pace), 4))

    old_count, new_count = reset["sessions"], max(int(session_count), 1)
    for metric, value in observed.items():
        if not isinstance(value, int | float):
            continue
        old_value = reset["metrics"].get(metric)
        if isinstance(old_value, int | float):
            value = (old_value * old_count + value * new_count) / (old_count + new_count)
        reset["metrics"][metric] = round(float(value), 4)
    reset["sessions"] = old_count + new_count


def choose_resets(teams: dict[str, Any]) -> dict[str, list[str]]:
    """Return ``{profile: [teams that drop the prior]}`` from the tracked sessions."""
    chosen: dict[str, list[str]] = {}
    for profile in PROFILES:
        tracked = {
            team: (payload.get("testing_reset") or {}).get(profile) or {}
            for team, payload in teams.items()
            if isinstance(payload, dict)
        }
        residuals = [
            value - sum(entry["observed"]) / len(entry["observed"])
            for entry in tracked.values()
            if len(entry.get("observed") or []) >= 2
            for value in entry["observed"]
        ]
        if not residuals:
            continue
        noise = 1.4826 * median(abs(residual) for residual in residuals)
        picked = []
        for team, entry in tracked.items():
            observed, prior = entry.get("observed") or [], entry.get("prior")
            if len(observed) < MIN_SESSIONS or not isinstance(prior, int | float):
                continue
            shift = sum(observed) / len(observed) - prior
            if shift and (noise == 0 or abs(shift) / (noise / len(observed) ** 0.5) >= MIN_ABS_Z):
                picked.append(team)
        if picked:
            chosen[profile] = sorted(picked)
    return chosen


def apply_testing_prior_choice(payload: dict[str, Any]) -> dict[str, Any]:
    """Swap in this season's own profile where the prior lost. No-op for unseeded seasons."""
    teams = payload.get("teams")
    if payload.get("seeded_from") is None or not isinstance(teams, dict):
        return payload
    chosen = choose_resets(teams)
    if not chosen:
        return payload

    result = deepcopy(payload)
    for profile, team_names in chosen.items():
        for team in team_names:
            team_data = result["teams"][team]
            metrics = team_data["testing_reset"][profile]["metrics"]
            profiles = team_data.setdefault("testing_characteristics_profiles", {})
            profiles[profile] = {**(profiles.get(profile) or {}), **metrics}
            if profile == "balanced" and isinstance(team_data.get("testing_characteristics"), dict):
                team_data["testing_characteristics"] = {
                    **team_data["testing_characteristics"],
                    **metrics,
                }
    logger.info("Testing profiles without the %s prior: %s", payload["seeded_from"], chosen)
    return result
