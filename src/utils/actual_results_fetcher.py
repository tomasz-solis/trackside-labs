"""Fetches actual results from competitive F1 sessions."""

import logging
from typing import List, Dict, Optional
import fastf1

logger = logging.getLogger(__name__)


def fetch_actual_session_results(
    year: int, race_name: str, session_name: str
) -> Optional[List[Dict[str, any]]]:
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
        for idx, row in results.iterrows():
            try:
                driver = row.get("Abbreviation", row.get("DriverNumber", "UNK"))
                team = row.get("TeamName", "Unknown")
                position = row.get("Position", idx + 1)

                # Handle DNFs/DSQs
                if position is None or str(position) == "nan":
                    position = idx + 1

                grid.append(
                    {
                        "position": int(position),
                        "driver": str(driver),
                        "team": str(team),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not parse result for driver at index {idx}: {e}")
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
