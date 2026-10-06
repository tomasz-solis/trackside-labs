"""Car traits v2: corners from track geometry, traits as time gained per corner class.

v1 found corners as speed dips, so corners taken flat or nearly flat were invisible
(Sepang: zero fast corners), and it read one apex sample at about 4 Hz telemetry, so
slow-corner apex speed was noise. v2:

- finds corners from the reference lap's X/Y geometry (curvature, radius < 500 m,
  at least 30 m long) and classes each by its minimum speed (slow < 140 km/h,
  medium, fast > 210 km/h); flat corners count like any other;
- finds straights (non-corner, >= 150 m, mean throttle > 90%) and braking zones
  (from the peak speed before a corner to its minimum, drop >= 40 km/h);
- measures each driver's time through every segment on their fastest qualifying lap,
  with segment positions as lap fractions so a longer line does not shift them;
- team trait per class = seconds gained per lap in that class versus the field
  median (positive = faster).

Track profile v2: share of the reference lap's time spent in each class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CLASSES = ("straight", "slow", "medium", "fast", "braking")
SLOW_MAX_KPH = 140.0
FAST_MIN_KPH = 210.0
_STEP_M = 5.0
_SMOOTH_POINTS = 9
_CORNER_MAX_RADIUS_M = 500.0
_CORNER_MIN_LENGTH_M = 30.0
_STRAIGHT_MIN_LENGTH_M = 150.0
_STRAIGHT_MIN_THROTTLE = 90.0
_BRAKING_LOOKBACK_M = 300.0
_BRAKING_MIN_DROP_KPH = 40.0
_MIN_DRIVERS = 6


def _resampled(lap: Any) -> dict[str, np.ndarray] | None:
    """Return the lap's telemetry resampled every 5 m along distance."""
    try:
        tel = lap.get_telemetry().add_distance()
    except Exception:  # missing position or car data for this lap
        return None
    if tel.empty or tel["Distance"].max() < 1000:
        return None
    dist = tel["Distance"].to_numpy(dtype=float)
    grid = np.arange(0.0, dist.max(), _STEP_M)
    out = {"distance": grid, "length": float(dist.max())}
    for name, column in (("x", "X"), ("y", "Y"), ("speed", "Speed"), ("throttle", "Throttle")):
        out[name] = np.interp(grid, dist, tel[column].to_numpy(dtype=float))
    out["time"] = np.interp(grid, dist, tel["Time"].dt.total_seconds().to_numpy())
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index runs where ``mask`` is True."""
    runs, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def segments(reference: dict[str, np.ndarray]) -> list[tuple[str, float, float]]:
    """Return (class, start fraction, end fraction) segments from a reference lap."""
    kernel = np.ones(_SMOOTH_POINTS) / _SMOOTH_POINTS
    x = np.convolve(reference["x"] / 10.0, kernel, "same")  # FastF1 X/Y are 1/10 m
    y = np.convolve(reference["y"] / 10.0, kernel, "same")
    dx, dy = np.gradient(x, _STEP_M), np.gradient(y, _STEP_M)
    ddx, ddy = np.gradient(dx, _STEP_M), np.gradient(dy, _STEP_M)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)
    corner = curvature > 1.0 / _CORNER_MAX_RADIUS_M
    speed, throttle, n = reference["speed"], reference["throttle"], len(reference["distance"])
    to_frac = 1.0 / n
    out: list[tuple[str, float, float]] = []

    for start, end in _runs(corner):
        if (end - start) * _STEP_M < _CORNER_MIN_LENGTH_M:
            continue
        low = float(speed[start:end].min())
        cls = "slow" if low < SLOW_MAX_KPH else ("fast" if low > FAST_MIN_KPH else "medium")
        out.append((cls, start * to_frac, end * to_frac))
        apex = start + int(np.argmin(speed[start:end]))
        back = max(0, apex - int(_BRAKING_LOOKBACK_M / _STEP_M))
        peak = back + int(np.argmax(speed[back : apex + 1]))
        if speed[peak] - speed[apex] >= _BRAKING_MIN_DROP_KPH and apex > peak:
            out.append(("braking", peak * to_frac, apex * to_frac))

    for start, end in _runs(~corner):
        if (end - start) * _STEP_M >= _STRAIGHT_MIN_LENGTH_M and throttle[
            start:end
        ].mean() > _STRAIGHT_MIN_THROTTLE:
            out.append(("straight", start * to_frac, end * to_frac))
    return out


def _segment_time(lap: dict[str, np.ndarray], start: float, end: float) -> float:
    """Return the lap's time between two lap fractions."""
    frac = lap["distance"] / lap["distance"][-1]
    return float(np.interp(end, frac, lap["time"]) - np.interp(start, frac, lap["time"]))


def _fastest_laps(session: Any) -> dict[str, Any]:
    """Return each driver's fastest timed lap."""
    laps = session.laps
    out = {}
    for driver in laps["Driver"].dropna().unique():
        lap = laps.pick_drivers(driver).pick_fastest()
        if lap is not None and not pd.isna(lap.get("LapTime")):
            out[str(driver)] = lap
    return out


def measure(session: Any, team_of: dict[str, str]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return (team traits in seconds gained per class, track profile shares).

    Empty frame and profile when too few drivers have usable telemetry.
    """
    fastest = _fastest_laps(session)
    laps = {d: r for d, lap in fastest.items() if d in team_of and (r := _resampled(lap))}
    if len(laps) < _MIN_DRIVERS:
        return pd.DataFrame(), {}
    reference_driver = min(laps, key=lambda d: fastest[d]["LapTime"])
    reference = laps[reference_driver]
    segs = segments(reference)

    rows = {}
    for driver, lap in laps.items():
        rows[driver] = [_segment_time(lap, s, e) for _cls, s, e in segs]
    times = pd.DataFrame(rows, index=range(len(segs))).T  # drivers x segments
    gained = -(times - times.median(axis=0))  # positive = faster than the field
    classes = [cls for cls, _s, _e in segs]
    per_class = pd.DataFrame(
        {cls: gained.loc[:, [i for i, c in enumerate(classes) if c == cls]].sum(axis=1) for cls in CLASSES if cls in classes}
    )
    per_class["team"] = [team_of[d] for d in per_class.index]
    traits = per_class.groupby("team").mean().reindex(columns=list(CLASSES))

    lap_time = float(reference["time"][-1] - reference["time"][0])
    profile = {
        cls: float(sum(_segment_time(reference, s, e) for c, s, e in segs if c == cls) / lap_time)
        for cls in CLASSES
    }
    return traits, profile
