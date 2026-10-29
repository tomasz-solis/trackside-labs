"""Measure race tyre degradation per track and write a relative severity into the track file.

``measure --season Y`` fits one regression per dry race over clean green laps:
lap time = driver intercept + compound offset + deg x tyre age + trend x lap number.
Tyre age resets at a stop while lap number keeps rising, so deg is separated from fuel
burn and track evolution (both land in the trend). Laps more than 2 s off the fit are
dropped and the fit is redone. Writes ``data/processed/tyre_deg/<Y>_race_tyre_deg.json``.

``apply --year Y`` divides each race's deg by its season's median, takes each track's
median over the three seasons before Y, pulls it toward 1.0 by ``SHRINK_SEASONS``, and
writes ``tyre_deg_severity`` (1.0 = a typical track) and ``tyre_deg_seasons`` into the
Y track characteristics file. Seasons before Y only: every track races once a season,
so a same-season value never exists before the race it would inform.

Checked 2026-10-29 against 16 dry 2026 races: rank correlation 0.61 with the measured
2026 severity, error 0.49 against 0.57 for a flat 1.0 (docs/MODEL_LEDGER.md).

Usage:
    uv run python scripts/extract_tyre_deg_severity.py measure --season 2025
    uv run python scripts/extract_tyre_deg_severity.py apply --year 2026 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.circuit_registry import CircuitResolutionError, resolve_track_data_key  # noqa: E402
from src.data.openf1_session import get_session  # noqa: E402
from src.utils.schema_validation import validate_track_characteristics  # noqa: E402
from src.utils.session_detector import SessionDetector  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.req", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)

SLICKS = ("SOFT", "MEDIUM", "HARD")
HISTORY_SEASONS = 3
SHRINK_SEASONS = 2  # the 1.0 prior counts as two seasons
MAX_WET_SHARE = 0.10  # races with more intermediate or wet laps are not dry races
MIN_LAPS = 300
MAX_AGE_LAP_CORR = 0.95
_TRIM_S = 2.0


def race_tyre_deg(laps: pd.DataFrame) -> dict[str, Any] | None:
    """Return the field's fuel-corrected tyre deg for one race (s per lap of tyre age)."""
    tyres = laps["Compound"].astype(str).str.upper()
    wet_share = float(tyres.isin(["INTERMEDIATE", "WET"]).mean())
    green = laps[
        (laps["TrackStatus"].astype(str) == "1")
        & laps["LapTime"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & (laps["TyreLife"] >= 2)
        & (laps["LapNumber"] >= 3)
        & tyres.isin(SLICKS)
    ].copy()
    green["t"] = green["LapTime"].dt.total_seconds()
    # Traffic, mistakes and damage: far off the driver's own median lap.
    green = green[(green["t"] - green.groupby("Driver")["t"].transform("median")).abs() < 3.0]
    if len(green) < MIN_LAPS:
        return None
    # One stop on the same lap for the whole field (an early safety car) makes tyre age
    # track lap number within every driver, and deg cannot be told from the trend.
    age = green["TyreLife"] - green.groupby("Driver")["TyreLife"].transform("mean")
    lap = green["LapNumber"] - green.groupby("Driver")["LapNumber"].transform("mean")
    if abs(float(np.corrcoef(age, lap)[0, 1])) > MAX_AGE_LAP_CORR:
        return None
    columns = [(green["Driver"] == d).to_numpy(float) for d in sorted(green["Driver"].unique())]
    compounds = sorted(green["Compound"].astype(str).str.upper().unique())
    columns += [
        (green["Compound"].astype(str).str.upper() == c).to_numpy(float) for c in compounds[1:]
    ]
    columns += [green["TyreLife"].to_numpy(float), green["LapNumber"].to_numpy(float)]
    x, y = np.column_stack(columns), green["t"].to_numpy(float)
    keep = np.ones(len(y), dtype=bool)
    for _ in range(2):
        beta = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
        keep = np.abs(y - x @ beta) < _TRIM_S
    return {
        "deg": round(float(beta[-2]), 4),
        "trend": round(float(beta[-1]), 4),
        "laps_used": int(keep.sum()),
        "wet_share": round(wet_share, 3),
    }


def measure_season(season: int) -> dict[str, Any]:
    """Return ``{track key: measurement}`` for every finished, resolvable race of ``season``."""
    detector = SessionDetector()
    races: dict[str, Any] = {}
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    for _, event in schedule.iterrows():
        name = str(event["EventName"])
        try:
            key = resolve_track_data_key(name, year=season, location=event.get("Location"))
        except CircuitResolutionError:
            key = None
        if key is None or not detector.is_session_completed(season, name, "R"):
            continue
        try:
            measured = race_tyre_deg(get_session(season, name, "R").laps)
        except Exception as exc:  # neither FastF1 nor OpenF1 has the laps
            logger.warning("%s %s skipped: %s", season, name, exc)
            continue
        if measured is not None:
            races[key] = {"race": name, **measured}
            print(f"{season} {name:<28} {measured}", flush=True)
    return races


def dry_relative(races: dict[str, Any]) -> dict[str, float]:
    """Each dry race's deg divided by the season median of dry races."""
    dry = {
        key: float(r["deg"])
        for key, r in races.items()
        if r["wet_share"] <= MAX_WET_SHARE and r["laps_used"] >= MIN_LAPS
    }
    median = statistics.median(dry.values())
    return {key: value / median for key, value in dry.items()}


def severities(seasons: list[dict[str, Any]]) -> dict[str, tuple[float, int]]:
    """Return ``{track key: (severity, seasons measured)}``, shrunk toward 1.0."""
    by_track: dict[str, list[float]] = {}
    for races in seasons:
        for key, value in dry_relative(races).items():
            by_track.setdefault(key, []).append(value)
    return {
        key: (
            round((len(v) * statistics.median(v) + SHRINK_SEASONS) / (len(v) + SHRINK_SEASONS), 3),
            len(v),
        )
        for key, v in sorted(by_track.items())
    }


def _season_path(season: int) -> Path:
    return PROJECT_ROOT / "data" / "processed" / "tyre_deg" / f"{season}_race_tyre_deg.json"


def _track_path(year: int) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "track_characteristics"
        / f"{year}_track_characteristics.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="Measure one season's races")
    measure.add_argument("--season", type=int, required=True)
    apply = sub.add_parser("apply", help="Write severities into a year's track file")
    apply.add_argument("--year", type=int, default=2026)
    apply.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    if args.command == "measure":
        fastf1.Cache.configure(cache_dir=str(PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"))
        fastf1.Cache.offline_mode(
            True
        )  # the live archive asks for an F1 TV login; OpenF1 fills gaps
        payload = {"season": args.season, "races": measure_season(args.season)}
        path = _season_path(args.season)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {len(payload['races'])} races to {path}")
        return

    seasons = []
    for season in range(args.year - HISTORY_SEASONS, args.year):
        path = _season_path(season)
        if not path.exists():
            parser.error(f"{path} is missing; run: measure --season {season}")
        seasons.append(json.loads(path.read_text(encoding="utf-8"))["races"])
    measured = severities(seasons)

    path = _track_path(args.year)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, entry in data["tracks"].items():
        severity, count = measured.get(key, (None, 0))
        print(f"  {key:<28} severity {severity} from {count} seasons")
        if severity is None:
            entry.pop("tyre_deg_severity", None)
            entry.pop("tyre_deg_seasons", None)
        else:
            entry["tyre_deg_severity"] = severity
            entry["tyre_deg_seasons"] = count
    validate_track_characteristics(data, expected_year=args.year)
    if not args.dry_run:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(data, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
