"""
Automatic Race Data Updater

Checks for completed 2026 races and automatically updates team/driver characteristics.
Called transparently before predictions - no manual intervention needed!
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)


def get_completed_races(year: int = 2026) -> List[str]:
    """Get list of completed races for the given year."""
    try:
        # Ensure cache directory exists
        import os

        cache_dir = Path(os.getenv("F1_CACHE_DIR", "data/raw/.fastf1_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        fastf1.Cache.enable_cache(str(cache_dir))
        schedule = fastf1.get_event_schedule(year)

        completed = []
        now = datetime.now()

        for _, event in schedule.iterrows():
            # Check if race has happened (date in the past)
            if "EventDate" in event and pd.notna(event["EventDate"]):
                event_date = pd.to_datetime(event["EventDate"])
                if event_date < now:
                    race_name = event["EventName"]
                    # Try to load race session to confirm data is available
                    try:
                        session = fastf1.get_session(year, race_name, "R")
                        if session is not None:
                            completed.append(race_name)
                    except (
                        ValueError,
                        KeyError,
                        AttributeError,
                        TypeError,
                        FileNotFoundError,
                    ) as e:
                        logger.debug(f"Race {race_name} not available yet: {e}")
                        continue  # Race not available yet

        return completed

    except Exception as e:
        logger.warning(f"Could not check for completed races: {e}")
        return []


def get_learned_races() -> List[str]:
    """Get list of races we've already learned from."""
    learning_file = Path("data/learning_state.json")

    if not learning_file.exists():
        return []

    try:
        with open(learning_file) as f:
            state = json.load(f)
            # Check for races in history
            history = state.get("history", [])
            return [record["race"] for record in history if "race" in record]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Could not load learning state: {e}")
        return []


def needs_update() -> Tuple[bool, List[str]]:
    """Check if there are new races to learn from."""
    completed = get_completed_races()
    learned = get_learned_races()

    new_races = [race for race in completed if race not in learned]

    return len(new_races) > 0, new_races


def auto_update_from_races(progress_callback=None) -> int:
    """Automatically update characteristics from any new completed races."""
    needs_update_flag, new_races = needs_update()

    if not needs_update_flag:
        logger.info("✓ All completed races already learned from - data is fresh!")
        return 0

    logger.info(f"Found {len(new_races)} new race(s) to learn from: {new_races}")

    # Import here to avoid circular dependency
    from src.systems.updater import update_from_race

    updated_count = 0

    for i, race_name in enumerate(new_races):
        try:
            if progress_callback:
                progress_callback(i + 1, len(new_races), f"Learning from {race_name}...")

            logger.info(f"Updating from {race_name} ({i + 1}/{len(new_races)})...")

            # Update from race (loads results, updates teams & drivers)
            update_from_race(2026, race_name)

            # Mark as learned
            mark_race_as_learned(race_name)

            updated_count += 1
            logger.info(f"  ✓ Successfully learned from {race_name}")

        except Exception as e:
            logger.warning(f"  ⚠️  Could not update from {race_name}: {e}")
            # Continue with other races even if one fails

    if updated_count > 0:
        logger.info(f"✓ Updated from {updated_count} race(s) - data is now fresher!")

    return updated_count


def mark_race_as_learned(race_name: str) -> None:
    """Mark a race as learned in the learning state."""
    learning_file = Path("data/learning_state.json")
    learning_file.parent.mkdir(parents=True, exist_ok=True)

    if learning_file.exists():
        with open(learning_file) as f:
            state = json.load(f)
    else:
        state = {
            "season": 2026,
            "races_completed": 0,
            "history": [],
            "method_performance": {},
        }

    # Add to history if not already there
    if "history" not in state:
        state["history"] = []

    # Check if already marked
    if not any(r.get("race") == race_name for r in state["history"]):
        state["history"].append(
            {
                "race": race_name,
                "date": datetime.now().isoformat(),
                "method": "auto_update",
            }
        )
        state["races_completed"] = state.get("races_completed", 0) + 1

    state["last_updated"] = datetime.now().isoformat()

    with open(learning_file, "w") as f:
        json.dump(state, f, indent=2)
