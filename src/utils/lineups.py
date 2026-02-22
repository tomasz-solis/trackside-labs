"""
Team Lineup Management - Race-by-Race

Extracts actual lineups from session data (handles all corner cases automatically).
Falls back to current_lineups.json for future predictions.
"""

import json
import logging
from pathlib import Path

import fastf1 as ff1

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def get_lineups_from_session(year, race_name, session_type="Q"):
    """
    Extract actual lineups from a specific race session.

    Handles reserve drivers, mid-season swaps, and injuries automatically.
    """
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

    except Exception as e:
        logger.warning(
            f"Failed to extract lineups from {race_name} ({year}) {session_type} session: {e}. Lineups from this session will be unavailable."
        )
        # Session data not available
        return None


def load_current_lineups(config_path="data/current_lineups.json"):
    """
    Load current team lineups from config file for future predictions or fallback.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        return None

    with open(config_file) as f:
        data = json.load(f)

    return data.get("current_lineups", {})


def get_lineups(year, race_name=None, config_path="data/current_lineups.json"):
    """
    Get team lineups for a race.

    Extracts from session data for 2024-2025, uses config for 2026+.
    Handles reserve drivers and mid-season changes automatically.
    """
    # For historical seasons with specific race, extract from data
    if year <= 2025 and race_name:
        session_lineups = get_lineups_from_session(year, race_name, "Q")

        if session_lineups:
            return session_lineups

        # If session data failed, fall through to config

    # For future seasons or fallback, use current config
    current_lineups = load_current_lineups(config_path)

    if current_lineups:
        return current_lineups

    raise ValueError(
        f"No lineup data available for {year}"
        + (f" - {race_name}" if race_name else "")
        + f"\nCreate config file at {config_path}"
    )


def save_current_lineups(lineups, config_path="../data/current_lineups.json"):
    """
    Save current lineups to config file when drivers change.
    """
    from datetime import datetime

    output = {"last_updated": datetime.now().isoformat(), "current_lineups": lineups}

    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved lineups to {config_file}")


def extract_lineups_for_season(year, output_path=None):
    """
    Extract lineups for all races in a season for reference or debugging.
    """
    import fastf1 as ff1

    schedule = ff1.get_event_schedule(year)
    all_lineups = {}

    logger.info(f"Extracting lineups for {year} season...")

    for _, event in schedule.iterrows():
        race_name = event["EventName"]

        if "Testing" in str(race_name):
            continue

        lineups = get_lineups_from_session(year, race_name, "Q")

        if lineups:
            all_lineups[race_name] = lineups
            logger.info(f"{race_name}: {len(lineups)} teams")
        else:
            logger.warning(f"{race_name}: No data")

    logger.info(f"Extracted {len(all_lineups)} races")

    if output_path:
        output = {"season": year, "races": all_lineups}

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved to {output_file}")

    return all_lineups
