"""
Team Lineup Management - Race-by-Race

Extracts actual lineups from session data (handles all corner cases automatically).
Falls back to current_lineups.json for future predictions. For the current season the
configured line-up takes the drivers entered in the latest session before the
forecast's checkpoint, from FP2 on; FP1 lists rookies who drive only that session.
"""

import json
import logging
import re
from pathlib import Path

import fastf1 as ff1

from src.utils.driver_substitutions import apply_substitutions, entry_list_swaps, swap_drivers
from src.utils.team_mapping import map_team_to_characteristics

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def get_lineups_from_session(
    year: int,
    race_name: str,
    session_type: str = "Q",
) -> dict[str, list[str]] | None:
    """Extract actual race-weekend lineups from a loaded FastF1 session."""
    try:
        session = ff1.get_session(year, race_name, session_type)
        session.load(laps=False, telemetry=False, weather=False)

        if not hasattr(session, "results") or session.results is None:
            return None

        lineups = {}

        # Extract actual participants
        for team in session.results["TeamName"].unique():
            team_results = session.results[session.results["TeamName"] == team]
            drivers = team_results["Abbreviation"].tolist()

            # Take whoever actually participated
            if drivers:
                # Handle 1 or 2 drivers (edge case: DNF/DNS)
                lineups[team] = drivers[:2] if len(drivers) >= 2 else drivers

        return lineups

    except (
        AttributeError,
        ConnectionError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as e:
        logger.warning(
            "Failed to extract lineups from %s (%s) %s session: %s. Lineups from this session will be unavailable.",
            race_name,
            year,
            session_type,
            e,
        )
        # Session data not available
        return None


# Sessions in weekend order. FP1 never sets the line-up: rookies replace race drivers in it.
_WEEKEND_SESSIONS = ("FP1", "FP2", "FP3", "SQ", "Sprint", "Q")
_ENTRY_LISTS: dict[tuple[int, str, str], list[tuple[str, str]]] = {}


def _session_entry_list(year: int, race_name: str, session_name: str) -> list[tuple[str, str]]:
    """Return (driver code, raw team name) for one session, FastF1 first, then OpenF1.

    shortcut: a found list is cached for the process and a missing one is asked again,
    so a session that never loads costs two lookups per call.
    """
    key = (int(year), str(race_name), session_name)
    if key in _ENTRY_LISTS:
        return _ENTRY_LISTS[key]
    entered: list[tuple[str, str]] = []
    try:
        session = ff1.get_session(year, race_name, session_name)
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = getattr(session, "results", None)
        if results is not None and len(results):
            entered = [
                (str(code), str(team))
                for code, team in zip(results["Abbreviation"], results["TeamName"], strict=True)
                if str(code).strip() and str(team).strip()
            ]
    except Exception as exc:  # noqa: BLE001 - a missing entry list keeps the configured line-up
        logger.info("No FastF1 entry list for %s %s %s: %s", year, race_name, session_name, exc)
        entered = []
    if not _valid_entry_list(entered, "FastF1", key):
        from src.data import openf1

        try:
            entered = openf1.fetch_entry_list(year, race_name, session_name)
        except openf1.OpenF1Unavailable:
            entered = []
        if not _valid_entry_list(entered, "OpenF1", key):
            entered = []
    if entered:
        _ENTRY_LISTS[key] = entered
    return entered


def _valid_entry_list(entered: list[tuple[str, str]], source: str, key: tuple) -> bool:
    """True when every driver has a three-letter code.

    One FastF1 load of São Paulo 2026 Q named Câmara "nan" (ledger 2026-11-20). A list
    with such a row would seat "NAN" in his place, so the whole list is refused.
    """
    # No upper-casing first: "nan" would pass as "NAN".
    bad = [
        code
        for code, _ in entered
        if not re.fullmatch(r"[A-Z]{3}", str(code).strip()) or str(code).strip() == "NAN"
    ]
    if bad:
        logger.warning("Refusing the %s entry list for %s: bad driver codes %s", source, key, bad)
    return bool(entered) and not bad


def seat_entered_drivers(
    lineups: dict[str, list[str]], *, year: int, race_name: str, checkpoint_session: str
) -> dict[str, list[str]]:
    """Seat the drivers entered in the latest session before the checkpoint, from FP2 on."""
    checkpoint = str(checkpoint_session).strip().upper()
    order = [session.upper() for session in _WEEKEND_SESSIONS]
    if checkpoint not in order:
        return lineups
    for session_name in reversed(_WEEKEND_SESSIONS[1 : order.index(checkpoint) + 1]):
        entered: dict[str, list[str]] = {}
        for code, team in _session_entry_list(year, race_name, session_name):
            mapped = map_team_to_characteristics(team, known_teams=set(lineups))
            if mapped:
                entered.setdefault(mapped, []).append(code.strip().upper())
        if entered:
            return swap_drivers(
                lineups,
                entry_list_swaps(lineups, entered),
                source=f"{session_name} entry list",
                race_name=race_name,
                year=year,
            )
    return lineups


def load_current_lineups(
    config_path: str = "data/current_lineups.json",
) -> dict[str, list[str]] | None:
    """Load the current lineup config used for future events and fallbacks."""
    config_file = Path(config_path)

    if not config_file.exists():
        return None

    try:
        with open(config_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Could not load lineup config %s: %s", config_file, exc)
        return None

    if not isinstance(data, dict):
        return None

    current_lineups = data.get("current_lineups", {})
    if not isinstance(current_lineups, dict):
        return None

    return {
        str(team_name): [str(driver_code) for driver_code in drivers]
        for team_name, drivers in current_lineups.items()
        if isinstance(drivers, list)
    }


def get_lineups(
    year: int,
    race_name: str | None = None,
    config_path: str = "data/current_lineups.json",
    checkpoint_session: str | None = None,
) -> dict[str, list[str]]:
    """Return the best available lineup mapping for one season or event.

    With a checkpoint, a current-season line-up seats the drivers entered in the latest
    session before it (``seat_entered_drivers``). A race-scoped driver substitution typed
    in the admin page is applied last, so it overrides the entry list.
    """
    # For historical seasons with specific race, extract from data
    if year <= 2025 and race_name:
        session_lineups = get_lineups_from_session(year, race_name, "Q")

        if session_lineups:
            return apply_substitutions(session_lineups, race_name=race_name, year=year)

        # If session data failed, fall through to config

    # For future seasons or fallback, use current config
    current_lineups = load_current_lineups(config_path)

    if current_lineups:
        if race_name and checkpoint_session:
            current_lineups = seat_entered_drivers(
                current_lineups,
                year=year,
                race_name=race_name,
                checkpoint_session=checkpoint_session,
            )
        return apply_substitutions(current_lineups, race_name=race_name, year=year)

    raise ValueError(
        f"No lineup data available for {year}"
        + (f" - {race_name}" if race_name else "")
        + f"\nCreate config file at {config_path}"
    )
