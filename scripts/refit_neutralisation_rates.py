"""Refit per-track safety car and VSC rates from race control messages.

A race counts once per kind when race control deployed at least one safety car (or VSC),
which is what the simulator's race-level ``safety_car_prob`` and ``vsc_prob`` mean.
2026 writes "VSC DEPLOYED" where earlier seasons wrote "VIRTUAL SAFETY CAR DEPLOYED", so
the kind comes from the message text, not an exact phrase. Messages come from the FastF1
cache, then OpenF1 for races FastF1 can no longer serve (FastF1 #978).

The season rate is this year's count pulled toward the three earlier seasons, which weigh
as one extra season. Each track keeps its prior shape: every calendar track moves by the
same log-odds shift, chosen so the calendar mean hits the season rate. One race per track
is noise, so no track is refitted from its own race.

Usage:
    uv run python scripts/refit_neutralisation_rates.py --year 2026 --dry-run
    uv run python scripts/refit_neutralisation_rates.py --year 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path

import fastf1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import openf1  # noqa: E402
from src.data.circuit_registry import CircuitResolutionError, resolve_track_data_key  # noqa: E402
from src.utils.session_detector import SessionDetector  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for _name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.req", "requests_cache"):
    logging.getLogger(_name).setLevel(logging.ERROR)

KINDS = (("sc", "safety_car_prob"), ("vsc", "vsc_prob"))
HISTORY_SEASONS = 3
_PROB_FLOOR = 0.01


def neutralisations_in(messages: list[tuple[str, str]]) -> dict[str, bool]:
    """Return whether race control deployed a safety car and a VSC, from (category, text) rows."""
    deployed = [
        text.upper()
        for category, text in messages
        if category == "SafetyCar" and "DEPLOYED" in text.upper()
    ]
    is_vsc = ["VSC" in text or "VIRTUAL" in text for text in deployed]
    return {"sc": any(not vsc for vsc in is_vsc), "vsc": any(is_vsc)}


def _race_control(year: int, race_name: str) -> list[tuple[str, str]]:
    """Race control (category, text) rows: FastF1 cache first, then OpenF1."""
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load(laps=False, telemetry=False, weather=False, messages=True)
        messages = session.race_control_messages
        if messages is not None and not messages.empty:
            return list(
                zip(
                    messages["Category"].astype(str),
                    messages["Message"].fillna("").astype(str),
                    strict=True,
                )
            )
    except Exception as exc:  # not cached; offline mode refuses the archive
        logger.info("FastF1 has no messages for %s %s: %s", year, race_name, exc)
    row = openf1.find_session(year, race_name, "R")
    if row is None:
        return []
    return [
        (str(r.get("category")), str(r.get("message") or ""))
        for r in openf1._get("race_control", session_key=row["session_key"])
    ]


def count_season(year: int, detector: SessionDetector) -> dict[str, dict[str, bool]]:
    """Return ``{event name: {"sc": bool, "vsc": bool}}`` for every finished race of ``year``."""
    races = {}
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for name in schedule["EventName"].astype(str):
        if not detector.is_session_completed(year, name, "R"):
            continue
        messages = _race_control(year, name)
        if messages:
            races[name] = neutralisations_in(messages)
        else:
            logger.warning("%s %s finished but has no race control messages", year, name)
    return races


def season_rate(history_hits: int, history_races: int, hits: int, races: int) -> float:
    """This season's rate, with the earlier seasons weighing as one extra season."""
    prior_weight = history_races / HISTORY_SEASONS
    return (prior_weight * history_hits / history_races + hits) / (prior_weight + races)


def _logit(p: float) -> float:
    p = min(max(p, _PROB_FLOOR), 1.0 - _PROB_FLOOR)
    return math.log(p / (1.0 - p))


def logit_shift(probs: dict[str, float], target_mean: float) -> dict[str, float]:
    """Shift every probability by one log-odds step so their mean equals ``target_mean``."""

    def shifted(step: float) -> dict[str, float]:
        return {key: 1.0 / (1.0 + math.exp(-(_logit(p) + step))) for key, p in probs.items()}

    lo, hi = -10.0, 10.0
    for _ in range(100):  # the mean rises with the step, so bisection converges
        mid = (lo + hi) / 2.0
        if sum(shifted(mid).values()) / len(probs) < target_mean:
            lo = mid
        else:
            hi = mid
    return {key: round(p, 3) for key, p in shifted((lo + hi) / 2.0).items()}


def refit(data: dict, calendar_keys: set[str], targets: dict[str, float]) -> dict:
    """Write shifted rates for calendar tracks; every other field and track stays as loaded."""
    tracks = data["tracks"]
    for kind, field in KINDS:
        current = {
            key: float(tracks[key][field])
            for key in sorted(calendar_keys)
            if key in tracks and tracks[key].get(field) is not None
        }
        for key, value in logit_shift(current, targets[kind]).items():
            tracks[key][field] = value
    return data


def calendar_track_keys(year: int) -> set[str]:
    """Track entry keys of every race on the ``year`` calendar that resolves to one."""
    keys = set()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        try:
            key = resolve_track_data_key(
                str(event["EventName"]), year=year, location=event.get("Location")
            )
        except CircuitResolutionError:
            key = None
        if key is None:
            logger.warning("%s has no track entry; its rates are not refitted", event["EventName"])
        else:
            keys.add(key)
    return keys


def _track_characteristics_path(year: int) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "track_characteristics"
        / f"{year}_track_characteristics.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    fastf1.Cache.configure(cache_dir=str(PROJECT_ROOT / "data" / "raw" / ".fastf1_cache"))
    fastf1.Cache.offline_mode(True)  # the live archive asks for an F1 TV login; OpenF1 fills gaps
    detector = SessionDetector()
    history = [
        race
        for year in range(args.year - HISTORY_SEASONS, args.year)
        for race in count_season(year, detector).values()
    ]
    current = list(count_season(args.year, detector).values())

    path = _track_characteristics_path(args.year)
    data = json.loads(path.read_text(encoding="utf-8"))
    calendar_keys = calendar_track_keys(args.year)
    targets = {}
    for kind, field in KINDS:
        history_hits, hits = sum(r[kind] for r in history), sum(r[kind] for r in current)
        targets[kind] = season_rate(history_hits, len(history), hits, len(current))
        old = [
            data["tracks"][key][field]
            for key in calendar_keys
            if data["tracks"].get(key, {}).get(field) is not None
        ]
        print(
            f"{field}: {args.year - HISTORY_SEASONS}-{args.year - 1} {history_hits}/{len(history)}, "
            f"{args.year} {hits}/{len(current)}, season rate {targets[kind]:.3f} "
            f"(calendar mean was {sum(old) / len(old):.3f})"
        )

    before = json.loads(json.dumps(data["tracks"]))
    data = refit(data, calendar_keys, targets)
    for key in sorted(calendar_keys & set(data["tracks"])):
        cells = [f"{before[key].get(f)} to {data['tracks'][key].get(f)}" for _, f in KINDS]
        print(f"  {key:<28} sc {cells[0]:<14} vsc {cells[1]}")
    if args.dry_run:
        return

    data["note"] = (
        f"safety_car_prob and vsc_prob refit {date.today().isoformat()} by "
        "scripts/refit_neutralisation_rates.py from race control messages"
    )
    data["last_updated"] = date.today().isoformat()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=2))
    logger.warning("Wrote refitted rates for %d tracks to %s", len(calendar_keys), path)


if __name__ == "__main__":
    main()
