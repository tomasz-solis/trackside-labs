"""Rebuild a season's measured team race pace from cached FastF1 lap data.

Live forecasts refresh this artifact automatically after each race (warmup step in
``src/dashboard/warmup.py``). This script rebuilds every race from scratch and
writes the committed file ``data/processed/team_race_pace/<year>_team_race_pace.json``,
which seeds the store and backs file mode and the replay. Method:
``src/extractors/team_race_pace.py``.

Usage:
    uv run python scripts/extract_team_race_pace.py --year 2026
    uv run python scripts/extract_team_race_pace.py --year 2026 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import fastf1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.team_race_pace import (  # noqa: E402
    build_payload,
    committed_path,
    measure_race,
)
from src.utils.session_detector import SessionDetector  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.ergast", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def collect_measurements(year: int) -> dict[str, dict[str, float]]:
    """Measure every completed race of ``year`` with loadable lap data."""
    cache_dir = PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    detector = SessionDetector()

    measurements: dict[str, dict[str, float]] = {}
    for _, event in fastf1.get_event_schedule(year).iterrows():
        race_name = str(event["EventName"])
        if "testing" in race_name.lower():
            continue
        # A future race loads without error but has no laps, so gate on completion.
        if not detector.is_session_completed(year, race_name, "R"):
            continue
        gaps = measure_race(year, race_name)
        if gaps:
            measurements[race_name] = gaps
    return measurements


def main() -> None:
    """Measure a season's team race pace and write it, unless --dry-run is set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the measured table without writing"
    )
    args = parser.parse_args()

    payload = build_payload(args.year, collect_measurements(args.year))
    teams = payload["teams"]
    print(f"{'team':<20} {'gap_s':>8} {'races':>6}")
    for team in sorted(teams, key=lambda name: teams[name]["gap_s"]):
        print(f"{team:<20} {teams[team]['gap_s']:>8.3f} {teams[team]['races']:>6}")

    if not teams:
        logger.warning("No completed, resolvable races found for %s", args.year)
        return
    if args.dry_run:
        return

    path = committed_path(args.year)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Wrote %d measured teams to %s", len(teams), path)


if __name__ == "__main__":
    main()
