"""Fill a FastF1 ``Session`` from OpenF1 when FastF1 cannot load its laps.

FastF1's archive needs an F1 TV login since 2026-10 (FastF1 #978). OpenF1 serves
the same timing feed free for finished sessions. This module writes OpenF1 laps,
stints, weather and (optionally) car data into the session's private FastF1
attributes, so ``session.laps``, ``session.weather_data`` and
``lap.get_car_data()`` work unchanged for every caller.

``TrackStatus`` and ``Deleted`` come from race control messages. On 13 cached
2026 races (R and Q, 19,504 laps) the green-lap mask matched FastF1 on 99.8%.
Not filled: ``Position``, ``SpeedFL``, results and position data.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import fastf1
import numpy as np
import pandas as pd
from fastf1.exceptions import DataNotLoadedError

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
    deleted: frozenset[tuple[int, float]] = frozenset(),
) -> pd.DataFrame:
    """Turn OpenF1 lap and stint rows into FastF1 ``Laps`` columns, one row per lap.

    A lap followed by a pit-out lap is the in-lap; in practice every driver's last
    lap is one too. ``IsAccurate`` is true for timed laps that are neither. Out-laps
    have no lap time, as in FastF1. Tyre age follows ``_carry_tyre_age``.
    ``deleted`` holds ``(driver_number, lap_seconds)`` of laps race control deleted.
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
            duration = lap.get("lap_duration")
            is_deleted = (
                isinstance(duration, int | float) and (number, round(duration, 3)) in deleted
            )
            personal_best = accurate and not is_deleted and lap_time < best
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
                    "Deleted": is_deleted,
                    "Source": openf1.SOURCE,
                }
            )
    return pd.DataFrame(rows)


def _status_changes(race_control: list[dict[str, Any]]) -> list[tuple[pd.Timestamp, str]]:
    """Return ``(date, code)`` each time the FastF1 track status code changes.

    Codes as FastF1: 1 clear, 2 yellow in any sector, 4 safety car, 5 red,
    6 VSC deployed, 7 VSC ending. A track-wide green or clear ends all of them.
    A restart after a red flag runs behind the safety car when race control says
    "SAFETY CAR LIGHTS ON" before the session restarts, until the standing start.
    """
    yellow: set[Any] = set()
    sc = vsc = red = ""
    suspended = sc_lights = False
    changes: list[tuple[pd.Timestamp, str]] = []
    for row in sorted(race_control, key=lambda r: str(r.get("date") or "")):
        if not row.get("date"):
            continue
        flag = str(row.get("flag") or "").upper()
        message = str(row.get("message") or "").upper()
        scope = str(row.get("scope") or "")
        if row.get("category") == "SafetyCar":
            if "VIRTUAL" in message or "VSC" in message:
                vsc = "7" if "ENDING" in message else "6"
            elif "DEPLOYED" in message:
                sc = "4"
        elif flag == "RED" or message.startswith("RED FLAG"):
            red = "5"
            suspended = True
        elif suspended and message.startswith("SAFETY CAR LIGHTS ON"):
            sc_lights = True
        elif suspended and row.get("category") == "SessionStatus" and "STARTED" in message:
            sc = "4" if sc_lights else sc
            suspended = sc_lights = False
        elif message.startswith("STANDING START"):
            sc = ""
        elif scope == "Sector" and "YELLOW" in flag:
            yellow.add(row.get("sector"))
        elif scope == "Sector" and flag in {"CLEAR", "GREEN"}:
            yellow.discard(row.get("sector"))
        elif scope == "Track" and flag in {"CLEAR", "GREEN"}:
            yellow.clear()
            sc = vsc = red = ""
        elif scope == "Track" and "YELLOW" in flag:
            yellow.add("track")
        code = red or sc or vsc or ("2" if yellow else "1")
        if not changes or changes[-1][1] != code:
            changes.append((_timestamp(row["date"]), code))
    return changes


# "CAR 55 (SAI) TIME 1:33.460 DELETED - TRACK LIMITS AT TURN 3 LAP 3 16:02:18" and
# "CAR 55 (SAI) LAP 1 WILL BE REINSTATED". The LAP number there is often one more
# than the timing lap, so a deleted lap is matched on its time.
_DELETED = re.compile(r"^CAR (\d+) \(\w+\) TIME (\d+):(\d+\.\d+) DELETED\b.*?\bLAP (\d+)\b")
_REINSTATED = re.compile(r"^CAR (\d+) \(\w+\) .*?\bLAP (\d+)\b.*\bREINSTATED\b")
_RESTART = re.compile(r"STANDING START|FORMATION LAP")


def deleted_laps(race_control: list[dict[str, Any]]) -> frozenset[tuple[int, float]]:
    """Return ``(driver_number, lap_seconds)`` for timed laps deleted and not reinstated.

    Untimed deletions ("LAP DELETED ... (PIT)") are aborted laps with no time to remove.
    """
    out: dict[tuple[int, int], float] = {}
    for row in sorted(race_control, key=lambda r: str(r.get("date") or "")):
        message = str(row.get("message") or "")
        if match := _DELETED.search(message):
            car, minutes, seconds, lap = match.groups()
            out[(int(car), int(lap))] = round(int(minutes) * 60 + float(seconds), 3)
        elif match := _REINSTATED.search(message):
            out.pop((int(match.group(1)), int(match.group(2))), None)
    return frozenset((car, seconds) for (car, _), seconds in out.items())


def untimed_restart_laps(
    laps: pd.DataFrame, race_control: list[dict[str, Any]], t0: pd.Timestamp
) -> pd.Series:
    """Return True for laps that hold a formation lap or standing start after a red flag.

    FastF1 has no lap time for these; OpenF1 times them, minutes slow.
    """
    dates = [
        _timestamp(row["date"])
        for row in race_control
        if row.get("date") and _RESTART.search(str(row.get("message") or ""))
    ]
    start, end = laps["LapStartDate"], t0 + laps["Time"]
    mask = pd.Series(False, index=laps.index)
    for date in dates:
        mask |= (start <= date) & (date < end)
    return mask


def drivers_short_of_results(
    laps: list[dict[str, Any]], results: list[dict[str, Any]]
) -> list[Any]:
    """Return drivers whose laps stop before their classified lap count; all when no results.

    On 15 published 2026 sessions (FP, SQ, Sprint, Q, R) every driver's last lap
    number was at least ``number_of_laps``, so a shortfall means laps are still arriving.
    """
    if not results:
        return ["no session_result"]
    last: dict[Any, int] = {}
    for lap in laps:
        number, lap_number = lap.get("driver_number"), lap.get("lap_number")
        if isinstance(lap_number, int):
            last[number] = max(last.get(number, 0), lap_number)
    return sorted(
        row["driver_number"]
        for row in results
        if isinstance(row.get("number_of_laps"), int)
        and last.get(row.get("driver_number"), 0) < row["number_of_laps"]
    )


def lap_track_status(
    laps: pd.DataFrame, race_control: list[dict[str, Any]], t0: pd.Timestamp
) -> pd.Series:
    """Return FastF1-style ``TrackStatus`` per lap: each code active during the lap, in order."""
    changes = _status_changes(race_control)
    out = []
    for start, end in zip(laps["LapStartDate"], laps["Time"], strict=False):
        if pd.isna(start) or pd.isna(end):
            out.append(pd.NA)
            continue
        finish = t0 + end
        codes = "1"
        for date, code in changes:
            if date <= start:
                codes = code
            elif date < finish and code != codes[-1]:
                codes += code
        out.append(codes)
    return pd.Series(out, index=laps.index, dtype=object)


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
    """Per-driver ``Telemetry`` with FastF1 car data channels; DRS is null in 2026."""
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

        def channel(name: str, frame: pd.DataFrame = frame) -> pd.Series:
            # 104 (no data) is kept, as FastF1 keeps it: the saved artifacts were built that way.
            return pd.to_numeric(frame.get(name), errors="coerce")

        car_data[str(number)] = fastf1.core.Telemetry(
            {
                "Date": dates,
                "Time": dates - t0,
                "SessionTime": dates - t0,
                "Speed": channel("speed"),
                "RPM": channel("rpm"),
                "nGear": channel("n_gear"),
                "Throttle": channel("throttle"),
                "Brake": channel("brake").fillna(0) > 0,
                "Source": "car",
            },
            session=session,
            driver=str(number),
            _cast_default_cols=True,
        )
    return car_data


def _circuit_info(session: fastf1.core.Session, row: dict[str, Any], t0: pd.Timestamp) -> Any:
    """Return FastF1 circuit info with corner distances on the fastest lap, or None.

    FastF1 places its markers through merged telemetry that needs every car's
    position data. This does the same nearest-point match with one request: the
    fastest lap's OpenF1 position samples, given distance from its car data.
    """
    lap = session.laps.pick_fastest()
    if lap is None or pd.isna(lap["LapStartTime"]) or pd.isna(lap["Time"]):
        return None
    info = fastf1.mvapi.get_circuit_info(year=session.event.year, circuit_key=row["circuit_key"])
    if info is None:
        return None
    rows = openf1._get(
        "location",
        cache=False,
        session_key=row["session_key"],
        driver_number=int(lap["DriverNumber"]),
        **{
            # urlencode turns these keys into date>= and date<=.
            "date>": (t0 + lap["LapStartTime"]).isoformat(),
            "date<": (t0 + lap["Time"]).isoformat(),
        },
    )
    if not rows:
        return None
    pos = pd.DataFrame(rows)
    pos_s = (pos["date"].map(_timestamp) - t0).dt.total_seconds().to_numpy()
    car = lap.get_car_data().add_distance()
    distance = np.interp(pos_s, car["SessionTime"].dt.total_seconds(), car["Distance"])
    xy = pos[["x", "y"]].to_numpy(dtype=float)
    for markers in (info.corners, info.marshal_sectors, info.marshal_lights):
        marker_xy = markers[["X", "Y"]].to_numpy(dtype=float)
        error = ((xy[None, :, :] - marker_xy[:, None, :]) ** 2).sum(axis=2)
        markers["Distance"] = distance[error.argmin(axis=1)]
    return info


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
    end = row.get("date_end")
    # Callers save what they measure once; a session still running must not be measured.
    if not end or pd.Timestamp.now(tz="UTC").tz_localize(None) < _timestamp(end):
        raise openf1.OpenF1Unavailable(f"{year} {race_name} {session_name} has not finished")
    key = row["session_key"]
    # Raw team names, as FastF1 laps carry; callers map them.
    drivers = openf1._driver_table(key, map_team=False)
    t0 = _timestamp(row["date_start"])
    race_control = openf1._get("race_control", session_key=key)
    lap_rows = openf1._get("laps", session_key=key)
    short = drivers_short_of_results(lap_rows, openf1._get("session_result", session_key=key))
    if short:
        raise openf1.OpenF1Unavailable(
            f"OpenF1 laps for {year} {race_name} {session_name} not fully published: {short}"
        )
    laps = build_laps_frame(
        lap_rows,
        openf1._get("stints", session_key=key),
        drivers,
        t0,
        last_lap_is_in_lap=str(session_name).startswith("FP"),
        deleted=deleted_laps(race_control),
    )
    if laps.empty:
        raise openf1.OpenF1Unavailable(f"OpenF1 has no laps for {year} {race_name} {session_name}")
    laps["TrackStatus"] = lap_track_status(laps, race_control, t0)
    restart = untimed_restart_laps(laps, race_control, t0)
    laps.loc[restart, "LapTime"] = pd.NaT
    laps.loc[restart, ["IsAccurate", "IsPersonalBest"]] = False

    session._t0_date = t0
    session._laps = fastf1.core.Laps(laps, session=session)
    session._weather_data = build_weather_frame(openf1._get("weather", session_key=key), t0)
    session._pos_data = {}
    if not hasattr(session, "_results"):
        # Entry list only, no finishing order; FastF1 reads session.drivers from it.
        session._results = fastf1.core.SessionResults(
            {
                "DriverNumber": [str(n) for n in drivers],
                "Abbreviation": [code for code, _ in drivers.values()],
                "TeamName": [team for _, team in drivers.values()],
            }
        )
    if not hasattr(session, "_session_status"):
        session._session_status = pd.DataFrame()
    session._car_data = _car_data(session, key, sorted(drivers), t0) if car_data else {}
    if car_data and row.get("circuit_key") is not None:
        try:
            info = _circuit_info(session, row, t0)
        except Exception as exc:  # circuit info is optional; corner callers have a fallback
            logger.info("No circuit info for %s %s: %s", race_name, session_name, exc)
            info = None
        # FastF1's own method needs session info and all position data, which OpenF1 lacks.
        session.get_circuit_info = lambda: info
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


def get_session(
    year: int,
    race_name: str,
    session_name: str,
    *,
    telemetry: bool = False,
    weather: bool = False,
) -> fastf1.core.Session:
    """Return a loaded FastF1 session, filled from OpenF1 when FastF1 has no laps.

    Raises ``OpenF1Unavailable`` when neither source has laps.
    """
    session = fastf1.get_session(year, race_name, session_name)
    session.load(laps=True, telemetry=telemetry, weather=weather, messages=False)
    try:
        if not session.laps.empty:
            return session
    except DataNotLoadedError:  # FastF1's archive refused the laps (FastF1 #978)
        pass
    load_session(session, year, race_name, session_name, car_data=telemetry)
    return session
