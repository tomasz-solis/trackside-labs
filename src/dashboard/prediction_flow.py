"""Dashboard prediction orchestration and result caching."""

import time

import streamlit as st

from .cache import get_predictor


def fetch_grid_if_available(
    year: int,
    race_name: str,
    session_name: str,
    predicted_grid: list,
) -> tuple:
    """
    Fetch actual grid if session completed, otherwise use predicted grid.

    Returns: (grid, grid_source) where grid_source is "ACTUAL" or "PREDICTED".
    """
    from src.utils.actual_results_fetcher import (
        fetch_actual_session_results,
        is_competitive_session_completed,
    )

    if is_competitive_session_completed(year, race_name, session_name):
        actual_grid = fetch_actual_session_results(year, race_name, session_name)
        if actual_grid:
            return actual_grid, "ACTUAL"

    return predicted_grid, "PREDICTED"


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def run_prediction(
    race_name: str,
    weather: str,
    _timestamps: dict[str, float],
    is_sprint: bool = False,
) -> dict:
    """
    Run full weekend cascade prediction with caching.

    Returns dict with:
    - Normal weekend: {"qualifying": {...}, "race": {...}}
    - Sprint weekend: {"sprint_quali": {...}, "sprint_race": {...}, "main_quali": {...}, "main_race": {...}}
    """
    valid_weather = ["dry", "rain", "mixed"]
    if weather not in valid_weather:
        raise ValueError(f"Weather must be one of {valid_weather}, got '{weather}'")

    timing: dict[str, float] = {}
    overall_start = time.time()

    predictor = get_predictor(_timestamps)
    results = {}

    if is_sprint:
        # SPRINT WEEKEND CASCADE: SQ -> Sprint -> Main Quali -> Main Race

        sq_start = time.time()
        sq_result = predictor.predict_qualifying(
            year=2026,
            race_name=race_name,
            qualifying_stage="sprint",
        )
        timing["sprint_quali"] = time.time() - sq_start
        results["sprint_quali"] = sq_result

        sprint_start = time.time()
        sq_grid, grid_source = fetch_grid_if_available(2026, race_name, "SQ", sq_result["grid"])
        results["sprint_quali"]["grid_source"] = grid_source

        sprint_result = predictor.predict_sprint_race(
            sprint_quali_grid=sq_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=50,
        )
        timing["sprint_race"] = time.time() - sprint_start
        results["sprint_race"] = sprint_result

        mq_start = time.time()
        mq_result = predictor.predict_qualifying(
            year=2026,
            race_name=race_name,
            qualifying_stage="main",
        )
        timing["main_quali"] = time.time() - mq_start
        results["main_quali"] = mq_result

        mr_start = time.time()
        quali_grid, grid_source = fetch_grid_if_available(2026, race_name, "Q", mq_result["grid"])
        results["main_quali"]["grid_source"] = grid_source

        main_race_result = predictor.predict_race(
            qualifying_grid=quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=50,
        )
        timing["main_race"] = time.time() - mr_start
        results["main_race"] = main_race_result

    else:
        # NORMAL WEEKEND CASCADE: Quali -> Race

        quali_start = time.time()
        quali_result = predictor.predict_qualifying(
            year=2026,
            race_name=race_name,
            qualifying_stage="main",
        )
        timing["qualifying"] = time.time() - quali_start
        results["qualifying"] = quali_result

        race_start = time.time()
        quali_grid, grid_source = fetch_grid_if_available(
            2026, race_name, "Q", quali_result["grid"]
        )
        results["qualifying"]["grid_source"] = grid_source

        race_result = predictor.predict_race(
            qualifying_grid=quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=50,
        )
        timing["race"] = time.time() - race_start
        results["race"] = race_result

    timing["total"] = time.time() - overall_start

    for key in results:
        results[key]["timing"] = timing

    return results
