"""Build Sepang track priors from the 1999-2017 Malaysian Grand Prix races.

The 2026 "Bahrain Grand Prix" (round 16) runs at Sepang, a circuit with no FastF1 lap
data (the last Malaysian GP was 2017). This reads Ergast history through FastF1 and
writes a ``Malaysian Grand Prix`` entry into ``2026_track_characteristics.json``:

- ``overtaking_avg_changes_per_lap``: position changes per lap, counted with the same
  rule as the baseline generator (skip lap 1, drop pit-out laps, need 5+ cars).
- ``pit_stop_loss``: pit lane time (Ergast ``duration``), outlier-filtered the same way.
- ``overtaking_difficulty``: the generator's mapping from changes per lap.
- ``safety_car_prob``: share of 1999-2017 races with a safety car. Ergast has no flag
  data, so a safety car is inferred from lap times: a run of laps at least 1.25x the
  race's median pace during which the leader-to-P5 gap closes to under 0.8x its size.
  Rain slows the field too but does not close the gaps. A VSC cannot be told apart.

Churn and pit loss use 2011-2017, the window with pit stop data (needed to drop
pit laps), DRS and Pirelli tyres and no refuelling.
These are previous-era priors: ``overtaking_observed_races`` is 0, so the loader treats
them as priors, not 2026 measurements.

Usage:
    uv run python scripts/extract_sepang_history.py --dry-run
    uv run python scripts/extract_sepang_history.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_2026_baseline import (  # noqa: E402
    _changes_per_lap_to_overtaking_difficulty,
    _estimate_overtaking_changes_per_lap,
    _filter_outlier_pit_losses,
)

YEARS = tuple(range(1999, 2018))
PIT_DATA_FROM = 2011
SLOW_LAP_FACTOR = 1.25
GAP_CLOSE_RATIO = 0.8
DATA_KEY = "Malaysian Grand Prix"
REQUEST_PAUSE_S = 0.4
TRACK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "track_characteristics"
    / "2026_track_characteristics.json"
)


def _fetch_politely(fetch: Any, **page: Any) -> Any:
    """Call Ergast (Jolpica) under its rate limit, retrying with backoff on HTTP 429."""
    from fastf1.exceptions import ErgastInvalidRequestError

    for attempt in range(6):
        time.sleep(REQUEST_PAUSE_S)
        try:
            return fetch(**page)
        except ErgastInvalidRequestError as exc:
            if "Too Many Requests" not in str(exc):
                raise
            time.sleep(2.0 * 2**attempt)
    raise RuntimeError("Ergast kept rate-limiting after 6 attempts")


def _all_pages(fetch: Any) -> list[pd.DataFrame]:
    """Collect every page of an Ergast endpoint by offset (100 rows per request)."""
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        response = _fetch_politely(fetch, limit=100, offset=offset)
        frames.extend(response.content)
        offset += 100
        if offset >= int(response.total_results):
            return frames


def laps_frame(lap_pages: list[pd.DataFrame], pit_stops: pd.DataFrame) -> pd.DataFrame:
    """Turn Ergast lap pages into the columns the baseline counter expects.

    Ergast gives the in-lap of each stop; FastF1 marks the lap after it with
    ``PitOutTime``, so the out-lap here is ``stop lap + 1``.
    """
    laps = pd.concat(lap_pages, ignore_index=True).rename(
        columns={"number": "LapNumber", "driverId": "Driver", "position": "Position"}
    )
    out_laps = {(row.driverId, int(row.lap) + 1) for row in pit_stops.itertuples()}
    laps["PitOutTime"] = [
        pd.Timedelta(0) if (driver, lap) in out_laps else pd.NaT
        for driver, lap in zip(laps["Driver"], laps["LapNumber"], strict=True)
    ]
    return laps


def had_safety_car(lap_pages: list[pd.DataFrame]) -> bool:
    """Return True when a slow run of laps also closed the leader-to-P5 gap."""
    laps = pd.concat(lap_pages, ignore_index=True).sort_values(["driverId", "number"])
    laps["cum"] = laps.groupby("driverId")["time"].cumsum().dt.total_seconds()
    median_pace = laps.groupby("number")["time"].median().dt.total_seconds()
    slow = median_pace[
        (median_pace.index >= 2) & (median_pace > SLOW_LAP_FACTOR * median_pace.iloc[1:].median())
    ]

    def p5_gap(lap: int) -> float:
        cum = laps.loc[laps["number"] == lap, "cum"].sort_values().to_numpy()
        return float(cum[min(4, len(cum) - 1)] - cum[0])

    for lap in slow.index:
        if lap - 1 in slow.index:
            continue  # only test each run once, from its first lap
        end = lap
        while end + 1 in slow.index:
            end += 1
        if p5_gap(min(end + 1, int(median_pace.index.max()))) < GAP_CLOSE_RATIO * p5_gap(lap - 1):
            return True
    return False


def measure_year(ergast: Any, year: int) -> dict[str, Any] | None:
    """Measure one Malaysian GP: safety car, and from 2011 churn and pit lane times."""
    results = _fetch_politely(ergast.get_race_results, season=year, circuit="sepang")
    if results.description.empty:
        return None
    race_round = int(results.description.iloc[0]["round"])
    lap_pages = _all_pages(
        lambda **page: ergast.get_lap_times(season=year, round=race_round, **page)
    )
    if not lap_pages:
        return None
    measurement = {
        "year": year,
        "round": race_round,
        "had_safety_car": had_safety_car(lap_pages),
        "changes_per_lap": None,
        "pit_losses": [],
    }
    if year < PIT_DATA_FROM:
        return measurement
    pit_pages = _all_pages(
        lambda **page: ergast.get_pit_stops(season=year, round=race_round, **page)
    )
    pit_stops = pd.concat(pit_pages, ignore_index=True)
    measurement["changes_per_lap"] = _estimate_overtaking_changes_per_lap(
        laps_frame(lap_pages, pit_stops)
    )
    measurement["pit_losses"] = [
        value.total_seconds() for value in pit_stops["duration"] if pd.notna(value)
    ]
    return measurement


def build_entry(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Average the yearly measurements into one track characteristics entry."""
    changes = [m["changes_per_lap"] for m in measurements if m["changes_per_lap"] is not None]
    pit_losses = _filter_outlier_pit_losses([v for m in measurements for v in m["pit_losses"]])
    avg_changes = statistics.mean(changes)
    return {
        "type": "permanent",
        "pit_stop_loss": round(statistics.mean(pit_losses), 1),
        "safety_car_prob": round(
            sum(m["had_safety_car"] for m in measurements) / len(measurements), 2
        ),
        "overtaking_difficulty": round(_changes_per_lap_to_overtaking_difficulty(avg_changes), 2),
        "overtaking_avg_changes_per_lap": round(avg_changes, 3),
        "overtaking_years_analyzed": len(changes),
        "overtaking_observed_races": 0,
    }


def main() -> None:
    """Measure Sepang history and write it to the 2026 track file unless --dry-run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    import fastf1
    from fastf1.ergast import Ergast

    fastf1.Cache.enable_cache(str(PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"))
    ergast = Ergast()
    measurements = [m for year in YEARS if (m := measure_year(ergast, year)) is not None]
    for m in measurements:
        line = f"{m['year']} round {m['round']}: safety car {m['had_safety_car']}"
        if m["changes_per_lap"] is not None:
            losses = m["pit_losses"]
            line += (
                f", {m['changes_per_lap']:.3f} changes/lap, {len(losses)} stops, "
                f"median pit {statistics.median(losses):.1f}s"
            )
        print(line)
    entry = build_entry(measurements)
    print(json.dumps({DATA_KEY: entry}, indent=2))
    if args.dry_run:
        return

    payload = json.loads(TRACK_FILE.read_text(encoding="utf-8"))
    existing = payload["tracks"].get(DATA_KEY, {})
    if existing.get("overtaking_observed_races"):
        raise SystemExit(
            f"{DATA_KEY} already holds a measured 2026 race "
            f"(overtaking_observed_races={existing['overtaking_observed_races']}); "
            "not overwriting it with historical priors."
        )
    payload["tracks"][DATA_KEY] = entry
    # Match the file's existing format (escaped non-ASCII, no trailing newline).
    TRACK_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {DATA_KEY} to {TRACK_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
