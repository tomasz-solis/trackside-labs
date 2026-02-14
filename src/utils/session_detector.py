"""Detects completed F1 sessions for prediction tracking."""

import logging
from datetime import UTC, datetime
from pathlib import Path

import fastf1

logger = logging.getLogger(__name__)


class SessionDetector:
    """Detects completed F1 sessions for a race weekend."""

    # Session names mapping
    NORMAL_WEEKEND_SESSIONS = ["FP1", "FP2", "FP3"]
    SPRINT_WEEKEND_SESSIONS = [
        "FP1",
        "SQ",
        "Sprint",
    ]  # SQ = Sprint Qualifying, Sprint = Sprint Race

    # Session durations in hours (including buffer time)
    SESSION_DURATIONS = {
        "FP1": 1.5,
        "FP2": 1.5,
        "FP3": 1.5,
        "SQ": 1.5,  # Sprint Qualifying
        "Sprint": 1.0,  # Sprint race is shorter
        "Q": 1.5,  # Qualifying
        "R": 2.5,  # Race
    }

    def __init__(self):
        cache_dir = Path("data/raw/.fastf1_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            fastf1.Cache.enable_cache(str(cache_dir))
        except Exception as exc:
            logger.debug(f"Could not enable FastF1 cache in SessionDetector: {exc}")

    def get_completed_sessions(self, year: int, race_name: str, is_sprint: bool) -> list[str]:
        """Get completed sessions for a race weekend."""
        try:
            # Get event schedule
            event = fastf1.get_event(year, race_name)

            # Determine which sessions to check based on weekend type
            sessions_to_check = (
                self.SPRINT_WEEKEND_SESSIONS if is_sprint else self.NORMAL_WEEKEND_SESSIONS
            )

            completed = []
            now = datetime.now(UTC)

            for session_name in sessions_to_check:
                try:
                    # Get session info
                    event.get_session_name(session_name)
                    session_date = event.get_session_date(session_name)

                    # Check if session has ended (use per-session duration)
                    if session_date is not None:
                        from datetime import timedelta

                        duration = self.SESSION_DURATIONS.get(session_name, 2.0)
                        session_end = session_date + timedelta(hours=duration)
                        if now >= session_end:
                            completed.append(session_name)
                    else:
                        logger.debug(f"Session {session_name} date not available for {race_name}")
                except Exception as e:
                    logger.debug(f"Could not check session {session_name} for {race_name}: {e}")
                    continue

            if not completed:
                logger.info(
                    f"No completed sessions found for {race_name} {year} "
                    f"(checked: {', '.join(sessions_to_check)})"
                )

            return completed

        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower():
                logger.error(f"Race not found: {race_name} {year}. Check race name spelling.")
            elif "connect" in error_msg.lower() or "network" in error_msg.lower():
                logger.error(f"FastF1 API connection failed: {error_msg}")
            else:
                logger.error(f"Failed to detect sessions for {race_name} {year}: {error_msg}")
            return []

    def get_latest_completed_session(
        self, year: int, race_name: str, is_sprint: bool
    ) -> str | None:
        """Get the most recent completed session."""
        completed = self.get_completed_sessions(year, race_name, is_sprint)
        return completed[-1] if completed else None

    def is_session_completed(self, year: int, race_name: str, session_name: str) -> bool:
        """Check if a specific session has completed."""
        try:
            event = fastf1.get_event(year, race_name)
            session_date = event.get_session_date(session_name)

            if session_date is None:
                return False

            now = datetime.now(UTC)
            from datetime import timedelta

            duration = self.SESSION_DURATIONS.get(session_name, 2.0)
            session_end = session_date + timedelta(hours=duration)
            return now >= session_end

        except Exception as e:
            logger.warning(f"Could not check if {session_name} completed: {e}")
            return False

    def get_sessions_for_weekend(self, is_sprint: bool) -> list[str]:
        """Get session names for a weekend type."""
        return self.SPRINT_WEEKEND_SESSIONS if is_sprint else self.NORMAL_WEEKEND_SESSIONS
