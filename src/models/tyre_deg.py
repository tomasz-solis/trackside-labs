"""Race tyre degradation slopes: field slope per compound plus a carried team estimate.

Observed race stints barely slow down: across 2026 the median raw stint slope (lap
time per lap of tyre age, fuel burn included) is about +0.02 s/lap for SOFT and 0 for
MEDIUM and HARD, because fuel burn cancels tyre wear. In the simulator every car
carries the same fuel on a given lap, so fuel never separates cars; tyre age at the
same moment does, and that is governed by true wear. So the slope is the observed raw
slope plus the real per-lap fuel gain (about 1.6 kg/lap x 0.03 s/kg), not the
simulator's own fuel model.

Per team, degradation is a small, persistent car trait (true sd about 0.009 s/lap per
lap against one-race noise of about 0.032), so it is estimated by a one-dimensional
Kalman filter over earlier races: start from 0 (the field), update with each race's
measured team delta weighted by its noise, and add a little drift per race so a car
that changes can show it. Practice slopes are not used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from statistics import median
from typing import Any

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
TEAM_TRUE_SD = 0.009
RACE_NOISE_SD = 0.032
DRIFT_SD = 0.002
# With no earlier race for a compound, use the season-typical raw slope.
FALLBACK_RAW_SLOPE = {"SOFT": 0.017, "MEDIUM": 0.0, "HARD": 0.0}


def field_raw_slopes(races_data: Mapping[str, Any], prior_races: Sequence[str]) -> dict[str, float]:
    """Return the median observed raw stint slope per compound over ``prior_races``."""
    out = {}
    for compound in COMPOUNDS:
        values = [
            float(races_data[r]["compound_deg"][compound])
            for r in prior_races
            if compound in (races_data.get(r) or {}).get("compound_deg", {})
        ]
        out[compound] = median(values) if values else FALLBACK_RAW_SLOPE[compound]
    return out


def team_deg_deltas(races_data: Mapping[str, Any], prior_races: Sequence[str]) -> dict[str, float]:
    """Return each team's carried degradation delta vs the field (s/lap per lap)."""
    mean: dict[str, float] = {}
    var: dict[str, float] = {}
    for race in prior_races:
        traits = (races_data.get(race) or {}).get("traits", {})
        for team, values in traits.items():
            obs = values.get("deg")
            m = mean.get(team, 0.0)
            p = var.get(team, TEAM_TRUE_SD**2) + DRIFT_SD**2
            if obs is None or obs != obs:  # missing or NaN: drift only
                mean[team], var[team] = m, p
                continue
            gain = p / (p + RACE_NOISE_SD**2)
            mean[team] = m + gain * (float(obs) - m)
            var[team] = (1.0 - gain) * p
    return mean


@lru_cache(maxsize=64)
def tyre_deg_slopes(
    year: int, race_name: str | None, fuel_gain_per_lap: float
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Return (field slope per compound for the simulator, team deltas) for a race.

    Uses only races the schedule places before ``race_name``. None when the race is
    not in the schedule, so callers keep their current slopes.
    """
    from src.extractors.car_track_traits import load_car_track_traits
    from src.utils.weekend import get_schedule_rows

    target = str(race_name or "").strip()
    try:
        schedule = [
            str(name).strip()
            for name, event_format in get_schedule_rows(year)
            if "testing" not in f"{name} {event_format}".lower()
        ]
    except Exception:  # schedule unavailable: keep current slopes rather than guess
        return None
    if target not in schedule:
        return None
    races_data = load_car_track_traits(year)["races"]
    prior = schedule[: schedule.index(target)]
    raw = field_raw_slopes(races_data, prior)
    field = {c: max(0.0, raw[c] + float(fuel_gain_per_lap)) for c in COMPOUNDS}
    return field, team_deg_deltas(races_data, prior)
