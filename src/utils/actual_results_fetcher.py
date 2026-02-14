"""Fetches actual results from competitive F1 sessions."""

import logging
from typing import Any

import fastf1

from src.utils.team_mapping import map_team_to_characteristics

logger = logging.getLogger(__name__)


def fetch_actual_session_results(
    year: int, race_name: str, session_name: str
) -> list[dict[str, Any]] | None:
    """Fetch actual results from competitive session (SQ, Sprint, Q, R)."""
    try:
        # Load session
        session = fastf1.get_session(year, race_name, session_name)
        session.load()

        # Get results
        results = session.results

        if results is None or len(results) == 0:
            logger.warning(f"No results available for {race_name} {session_name}")
            return None

        # Extract relevant data
        grid = []
        for fallback_position, (_, row) in enumerate(results.iterrows(), start=1):
            try:
                driver = row.get("Abbreviation", row.get("DriverNumber", "UNK"))
                team_raw = row.get("TeamName", "Unknown")
                team = map_team_to_characteristics(team_raw) or str(team_raw)
                position = row.get("Position", fallback_position)

                # Handle DNFs/DSQs
                if position is None or str(position) == "nan":
                    position = fallback_position

                grid.append(
                    {
                        "position": int(position),
                        "driver": str(driver),
                        "team": str(team),
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Could not parse result for driver at fallback position {fallback_position}: {e}"
                )
                continue

        # Sort by position
        grid.sort(key=lambda x: x["position"])

        logger.info(f"Fetched {len(grid)} results from {race_name} {session_name}")
        return grid

    except Exception as e:
        logger.error(f"Failed to fetch {session_name} results for {race_name}: {e}")
        return None


def is_competitive_session_completed(year: int, race_name: str, session_name: str) -> bool:
    """
    Check if a competitive session has completed and results are available.

    Only checks COMPETITIVE sessions (SQ, Sprint, Quali, Race).
    """
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    return detector.is_session_completed(year, race_name, session_name)
