"""
FP Practice Session Blending

Extracts team session performance and blends it with model team strength.
Also extracts compound-specific performance during FP sessions.

Current active usage:
- called by `Baseline2026Predictor.predict_qualifying`
- uses the best single available session by priority
- applies a 70/30 session/model blend in baseline predictor logic
"""

import logging

import fastf1 as ff1
import numpy as np
import pandas as pd

from src.utils.team_mapping import map_team_to_characteristics

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def get_fp_team_performance(
    year: int, race_name: str, session_type: str
) -> tuple[dict[str, float] | None, pd.DataFrame | None]:
    """
    Extract team performance and session laps from practice/qualifying session.
    Returns (team_performance, session_laps) or (None, None) if unavailable.
    """
    try:
        session = ff1.get_session(year, race_name, session_type)
        session.load(laps=True, telemetry=False, weather=False)

        if not hasattr(session, "laps") or session.laps is None or session.laps.empty:
            return None, None

        laps = session.laps

        # Get best lap per driver (excluding outliers)
        best_times = []

        for driver in laps["Driver"].unique():
            driver_laps = laps[laps["Driver"] == driver]

            # Filter valid laps
            valid_laps = driver_laps[
                (driver_laps["LapTime"].notna())
                & (driver_laps["Compound"].notna())  # On tire compound
            ]

            if len(valid_laps) == 0:
                continue

            # Get best lap time
            best_lap = valid_laps["LapTime"].min()
            team_raw = driver_laps["Team"].iloc[0]
            if pd.isna(team_raw):
                continue
            team = map_team_to_characteristics(team_raw) or str(team_raw)

            best_times.append({"driver": driver, "team": team, "time": best_lap.total_seconds()})

        if not best_times:
            return None, None

        # Get median time per team (robust to one driver having issues)
        team_times = {}

        for entry in best_times:
            team = entry["team"]
            if team not in team_times:
                team_times[team] = []
            team_times[team].append(entry["time"])

        team_medians = {team: np.median(times) for team, times in team_times.items()}

        # Convert to relative performance (0-1 scale)
        # Invert: Faster time = Higher score
        fastest = min(team_medians.values())
        slowest = max(team_medians.values())

        if fastest == slowest:
            # All teams same pace (unlikely)
            return {team: 0.5 for team in team_medians}, laps

        team_performance = {
            team: 1.0 - (time - fastest) / (slowest - fastest)
            for team, time in team_medians.items()
        }

        return team_performance, laps

    except Exception as e:
        logger.debug(
            f"Could not extract team performance from {session_type} for {race_name} ({year}): {e}"
        )
        return None, None


def get_best_fp_performance(
    year: int,
    race_name: str,
    is_sprint: bool = False,
    qualifying_stage: str = "auto",
) -> tuple[str | None, dict[str, float] | None, pd.DataFrame | None]:
    """
    Get the best available practice session data for blending.
    Returns (session_label, team_performance, session_laps).

    Normal weekend priority: FP3 > FP2 > FP1

    Sprint weekend priorities depend on qualifying stage:
    - stage="sprint": FP1 only (pre-SQ context)
    - stage="main": Sprint Race > Sprint Qualifying > FP1
    - stage="auto": Sprint Race > Sprint Qualifying > FP1 (legacy behavior)
    """
    stage = (qualifying_stage or "auto").strip().lower()
    if stage not in {"auto", "sprint", "main"}:
        raise ValueError("qualifying_stage must be one of: 'auto', 'sprint', 'main'")

    if is_sprint:
        if stage == "sprint":
            # Sprint Qualifying prediction should be anchored to pre-SQ context.
            sessions = [("FP1", "FP1 times")]
        else:
            # Main qualifying on sprint weekends can use sprint/SQ evidence.
            sessions = [
                ("Sprint", "Sprint Race times"),
                ("Sprint Qualifying", "Sprint Qualifying times"),
                ("FP1", "FP1 times"),
            ]
    else:
        # Normal weekend: Try FP3 > FP2 > FP1
        sessions = [
            ("FP3", "FP3 times"),
            ("FP2", "FP2 times"),
            ("FP1", "FP1 times"),
        ]

    for session_code, session_label in sessions:
        fp_data, session_laps = get_fp_team_performance(year, race_name, session_code)
        if fp_data is not None:
            logger.info(f"Using {session_label} for blending")
            return session_label, fp_data, session_laps

    logger.info("No practice data available - using model-only predictions")
    return None, None, None


def blend_team_strength(
    model_strength: dict[str, float],
    fp_performance: dict[str, float] | None,
    blend_weight: float = 0.7,
) -> dict[str, float]:
    """Blend model predictions with FP data (default: 70% practice + 30% model). Validates team name matches."""
    if fp_performance is None:
        return model_strength

    # Validate team name matches
    model_teams = set(model_strength.keys())
    fp_teams = set(fp_performance.keys())

    missing_from_fp = model_teams - fp_teams
    extra_in_fp = fp_teams - model_teams

    if missing_from_fp:
        logger.warning(
            f"Teams in model but missing from FP data (using model-only): {', '.join(sorted(missing_from_fp))}"
        )

    if extra_in_fp:
        logger.debug(
            f"Teams in FP data but not in model (ignoring): {', '.join(sorted(extra_in_fp))}"
        )

    blended = {}

    for team, model_score in model_strength.items():
        fp_score = fp_performance.get(team, model_score)

        if team in missing_from_fp:
            logger.debug(f"  {team}: Model-only (no FP data) = {model_score:.3f}")
            blended[team] = model_score
        else:
            blended_score = blend_weight * fp_score + (1 - blend_weight) * model_score
            logger.debug(
                f"  {team}: FP={fp_score:.3f}, Model={model_score:.3f} "
                f"→ Blended={blended_score:.3f}"
            )
            blended[team] = blended_score

    return blended
