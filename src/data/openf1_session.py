"""Fill a FastF1 ``Session`` from OpenF1 when FastF1 cannot load its laps.

FastF1's archive needs an F1 TV login since 2026-10 (FastF1 #978). OpenF1 serves
the same timing feed free for finished sessions. This module writes OpenF1 laps,
stints, weather and (optionally) car data into the session's private FastF1
attributes, so ``session.laps``, ``session.weather_data`` and
``lap.get_car_data()`` work unchanged for every caller.

Not filled: ``TrackStatus``, ``Deleted``, ``Position``, ``SpeedFL``, results and
position data. Callers that filter on ``TrackStatus == "1"`` get no laps, as they
do today without FastF1.
"""

from __future__ import annotations

import logging
from typing import Any

import fastf1
import pandas as pd

from src.data import openf1
from src.utils.operational_observability import record_counter

logger = logging.getLogger(__name__)


def _timestamp(value: Any) -> pd.Timestamp:
    """Parse an OpenF1 ISO time to a naive UTC timestamp, as FastF1 stores dates."""
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _seconds(value: Any) -> pd.Timedelta:
    return pd.to_timedelta(value, unit="s") if isinstance(value, int | float) else pd.NaT


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else float("nan")


def _stint_for_lap(stints: list[dict[str, Any]], lap_number: int) -> dict[str, Any] | None:
    for stint in stints:
        start, end = stint.get("lap_start"), stint.get("lap_end")
        if isinstance(start, int) and isinstance(end, int) and start <= lap_number <= end:
            return stint
    return None


def _carry_tyre_age(stints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy stints, carrying tyre age over when OpenF1 reports a refit set as new.

    OpenF1 gives ``tyre_age_at_start`` 0 on many stints that FastF1 shows as the
    previous set refitted. On 2026 FP sessions, for a stint on the same compound as
    the one before with age 0, the set continued in 76 of 129 cases and was new in
    29. A non-zero OpenF1 age matched FastF1 in 340 of 344 stints and is kept.
    """
    carried: list[dict[str, Any]] = []
    for stint in sorted(stints, key=lambda row: row.get("stint_number") or 0):
        stint = dict(stint)
        prev = carried[-1] if carried else None
        if (
            prev is not None
            and stint.get("tyre_age_at_start") == 0
            and stint.get("compound") == prev.get("compound")
            and all(
                isinstance(prev.get(k), int) for k in ("tyre_age_at_start", "lap_start", "lap_end")
            )
        ):
            # shortcut: wrong when a new set of the same compound is fitted (29 of 129); live timing has the real age.
            stint["tyre_age_at_start"] = (
                prev["tyre_age_at_start"] + prev["lap_end"] - prev["lap_start"] + 1
            )
        carried.append(stint)
    return carried


def build_laps_frame(
    laps: list[dict[str, Any]],
    stints: list[dict[str, Any]],
    drivers: dict[int, tuple[str, str]],
    t0: pd.Timestamp,
    *,
    last_lap_is_in_lap: bool = False,
) -> pd.DataFrame:
    """Turn OpenF1 lap and stint rows into FastF1 ``Laps`` columns, one row per lap.

    A lap followed by a pit-out lap is the in-lap; in practice every driver's last
    lap is one too. ``IsAccurate`` is true for timed laps that are neither. Out-laps
    have no lap time, as in FastF1. Tyre age follows ``_carry_tyre_age``.
    """
    stints_by_driver: dict[int, list[dict[str, Any]]] = {}
    for stint in stints:
        if isinstance(stint.get("driver_number"), int):
            stints_by_driver.setdefault(stint["driver_number"], []).append(stint)
    stints_by_driver = {number: _carry_tyre_age(rows) for number, rows in stints_by_driver.items()}

    by_driver: dict[int, list[dict[str, Any]]] = {}
    for lap in laps:
        number = lap.get("driver_number")
        if isinstance(number, int) and number in drivers and isinstance(lap.get("lap_number"), int):
            by_driver.setdefault(number, []).append(lap)

    rows: list[dict[str, Any]] = []
    for number, driver_laps in sorted(by_driver.items()):
        code, team = drivers[number]
        driver_laps.sort(key=lambda lap: lap["lap_number"])
        best = pd.Timedelta.max
        for index, lap in enumerate(driver_laps):
            start_date = _timestamp(lap["date_start"]) if lap.get("date_start") else pd.NaT
            start_time = start_date - t0 if start_date is not pd.NaT else pd.NaT
            lap_time = _seconds(lap.get("lap_duration"))
            next_lap = driver_laps[index + 1] if index + 1 < len(driver_laps) else None
            if lap_time is not pd.NaT:
                end_time = start_time + lap_time
            elif next_lap is not None and next_lap.get("date_start"):
                end_time = _timestamp(next_lap["date_start"]) - t0
            else:
                end_time = pd.NaT
            pit_out = bool(lap.get("is_pit_out_lap"))
            if pit_out:
                # FastF1's timing feed has no lap time for out-laps; OpenF1 times them.
                lap_time = pd.NaT
            if next_lap is None:
                pit_in = last_lap_is_in_lap
            else:
                pit_in = bool(next_lap.get("is_pit_out_lap"))
            lap_stint = _stint_for_lap(stints_by_driver.get(number, []), lap["lap_number"])
            tyre_life = float("nan")
            if lap_stint is not None and isinstance(lap_stint.get("tyre_age_at_start"), int):
                tyre_life = float(
                    lap_stint["tyre_age_at_start"] + lap["lap_number"] - lap_stint["lap_start"] + 1
                )
            accurate = lap_time is not pd.NaT and not pit_out and not pit_in
            # shortcut: no track-limit deletions in OpenF1, so a deleted lap can count as a personal best.
            personal_best = accurate and lap_time < best
            if personal_best:
                best = lap_time
            rows.append(
                {
                    "Time": end_time,
                    "Driver": code,
                    "DriverNumber": str(number),
                    "Team": team,
                    "LapTime": lap_time,
                    "LapNumber": float(lap["lap_number"]),
                    "Stint": _number(lap_stint.get("stint_number")) if lap_stint else float("nan"),
                    "PitOutTime": start_time if pit_out else pd.NaT,
                    "PitInTime": end_time if pit_in else pd.NaT,
                    "Sector1Time": _seconds(lap.get("duration_sector_1")),
                    "Sector2Time": _seconds(lap.get("duration_sector_2")),
                    "Sector3Time": _seconds(lap.get("duration_sector_3")),
                    "SpeedI1": _number(lap.get("i1_speed")),
                    "SpeedI2": _number(lap.get("i2_speed")),
                    "SpeedST": _number(lap.get("st_speed")),
                    "Compound": (lap_stint.get("compound") or None) if lap_stint else None,
                    "TyreLife": tyre_life,
                    "FreshTyre": bool(lap_stint and lap_stint.get("tyre_age_at_start") == 0),
                    "LapStartTime": start_time,
                    "LapStartDate": start_date,
                    "IsPersonalBest": personal_best,
                    "IsAccurate": accurate,
                    "Source": openf1.SOURCE,
                }
            )
    return pd.DataFrame(rows)


def build_weather_frame(weather: list[dict[str, Any]], t0: pd.Timestamp) -> pd.DataFrame:
    """Turn OpenF1 weather rows into FastF1 ``weather_data`` columns."""
    rows = [
        {
            "Time": _timestamp(row["date"]) - t0,
            "AirTemp": _number(row.get("air_temperature")),
            "Humidity": _number(row.get("humidity")),
            "Pressure": _number(row.get("pressure")),
            "Rainfall": bool(row.get("rainfall")),
            "TrackTemp": _number(row.get("track_temperature")),
            "WindDirection": _number(row.get("wind_direction")),
            "WindSpeed": _number(row.get("wind_speed")),
        }
        for row in weather
        if row.get("date")
    ]
    return pd.DataFrame(rows)


def _car_data(session: fastf1.core.Session, session_key: Any, numbers: list[int], t0: pd.Timestamp):
    """Per-driver ``Telemetry`` with the channels the updater reads (see load_car_data_only)."""
    car_data: dict[str, fastf1.core.Telemetry] = {}
    for number in numbers:
        # Not cached: about 19k samples a driver, read once per run.
        samples = openf1._get(
            "car_data", cache=False, session_key=session_key, driver_number=number
        )
        if not samples:
            continue
        frame = pd.DataFrame(samples)
        dates = frame["date"].map(_timestamp)
        car_data[str(number)] = fastf1.core.Telemetry(
            {
                "SessionTime": dates - t0,
                "Brake": pd.to_numeric(frame["brake"], errors="coerce").fillna(0) > 0,
            },
            session=session,
            driver=str(number),
            _cast_default_cols=True,
        )
    return car_data


def load_session(
    session: fastf1.core.Session,
    year: int,
    race_name: str,
    session_name: str,
    *,
    car_data: bool = False,
) -> None:
    """Fill ``session`` with OpenF1 laps and weather, and car data when asked.

    Raises ``OpenF1Unavailable`` when OpenF1 is off, unreachable or has no laps.
    """
    row = openf1.find_session(year, race_name, session_name)
    if row is None or not row.get("date_start"):
        raise openf1.OpenF1Unavailable(f"no OpenF1 session for {year} {race_name} {session_name}")
    key = row["session_key"]
    # Raw team names, as FastF1 laps carry; callers map them.
    drivers = openf1._driver_table(key, map_team=False)
    t0 = _timestamp(row["date_start"])
    laps = build_laps_frame(
        openf1._get("laps", session_key=key),
        openf1._get("stints", session_key=key),
        drivers,
        t0,
        last_lap_is_in_lap=str(session_name).startswith("FP"),
    )
    if laps.empty:
        raise openf1.OpenF1Unavailable(f"OpenF1 has no laps for {year} {race_name} {session_name}")

    session._t0_date = t0
    session._laps = fastf1.core.Laps(laps, session=session)
    session._weather_data = build_weather_frame(openf1._get("weather", session_key=key), t0)
    session._pos_data = {}
    session._car_data = _car_data(session, key, sorted(drivers), t0) if car_data else {}
    record_counter(
        "openf1_fallback_used_total",
        labels={"year": year, "race_name": race_name, "session": session_name, "what": "laps"},
    )
    logger.warning(
        "FastF1 had no laps for %s %s %s; loaded %s laps from OpenF1",
        year,
        race_name,
        session_name,
        len(laps),
    )
