"""
Streamlit Dashboard for F1 2026 Predictions

Live race predictions with historical accuracy tracking.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import fastf1
import pandas as pd
import streamlit as st

logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1.api").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
_PRACTICE_UPDATE_STATE_FILE = Path("data/systems/practice_characteristics_state.json")
_FASTF1_CACHE_DIR = Path("data/raw/.fastf1_cache")


def _enable_fastf1_cache() -> None:
    """Ensure FastF1 uses project-local cache (avoids default-cache warnings/noise)."""
    _FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(str(_FASTF1_CACHE_DIR))
    except Exception as exc:
        logger.warning(f"Could not enable FastF1 cache at {_FASTF1_CACHE_DIR}: {exc}")


_enable_fastf1_cache()


# Get file modification times for cache invalidation
def get_data_file_timestamps():
    """Get modification timestamps of all data files."""
    import os
    from pathlib import Path

    files = [
        "data/processed/car_characteristics/2026_car_characteristics.json",
        "data/processed/driver_characteristics.json",
        "data/processed/track_characteristics/2026_track_characteristics.json",
        "data/2025_pirelli_info.json",  # Tire characteristics (fallback for 2026)
        "data/2026_pirelli_info.json",  # Tire characteristics (if available)
        "config/default.yaml",  # Model weight tuning
        "src/predictors/baseline_2026.py",  # Predictor logic changes
    ]

    timestamps = {}
    for file in files:
        path = Path(file)
        if path.exists():
            timestamps[file] = os.path.getmtime(path)
        else:
            timestamps[file] = 0

    return timestamps


# Cache predictor instance but check for file changes
@st.cache_resource(show_spinner=False)
def get_predictor(_timestamps):
    """Load and cache the baseline predictor instance (invalidates when data files change)."""
    import logging

    from src.predictors.baseline_2026 import Baseline2026Predictor

    # Temporarily suppress INFO logs during initialization to avoid clutter
    original_level = logging.getLogger("src.utils.data_generator").level
    logging.getLogger("src.utils.data_generator").setLevel(logging.WARNING)

    predictor = Baseline2026Predictor()

    # Restore original level
    logging.getLogger("src.utils.data_generator").setLevel(original_level)

    return predictor


# Auto-update from completed races and check file freshness
def auto_update_if_needed():
    """
    Check for and apply updates from completed races.
    Also refreshes predictor if characteristic files were manually updated.
    """
    # Check if we need to learn from new races
    from src.utils.auto_updater import auto_update_from_races, needs_update

    needs_update_flag, new_races = needs_update()

    if needs_update_flag:
        st.info(f"🔄 Found {len(new_races)} new race(s) to learn from! Updating characteristics...")

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(message)

        # Update from races
        updated_count = auto_update_from_races(progress_callback)

        progress_bar.empty()
        status_text.empty()

        if updated_count > 0:
            st.success(f"✅ Learned from {updated_count} race(s)! Predictions now use fresh data.")
            # Clear caches since data changed
            st.cache_resource.clear()
            st.cache_data.clear()
        else:
            st.warning("⚠️ Could not update from new races - using existing data")


def _load_practice_update_state() -> dict:
    """Load persisted state for practice characteristic updates."""
    if not _PRACTICE_UPDATE_STATE_FILE.exists():
        return {"races": {}}

    try:
        with open(_PRACTICE_UPDATE_STATE_FILE) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"races": {}}

    if not isinstance(state, dict):
        return {"races": {}}

    races = state.get("races")
    if not isinstance(races, dict):
        return {"races": {}}

    return {"races": races}


def _save_practice_update_state(state: dict) -> None:
    """Persist state for practice characteristic updates."""
    _PRACTICE_UPDATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PRACTICE_UPDATE_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def auto_update_practice_characteristics_if_needed(
    year: int,
    race_name: str,
    is_sprint: bool,
) -> dict:
    """
    Update car characteristics from completed free-practice sessions (FP1/FP2/FP3).

    This is conservative and only runs when new FP sessions are completed for a race.
    """
    from src.systems.testing_updater import update_from_testing_sessions
    from src.utils import config_loader
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    completed = detector.get_completed_sessions(year, race_name, is_sprint)
    completed_fp_sessions = [session for session in completed if session.startswith("FP")]

    if not completed_fp_sessions:
        return {"updated": False, "completed_fp_sessions": []}

    session_order = {"FP1": 1, "FP2": 2, "FP3": 3}
    completed_fp_sessions = sorted(
        set(completed_fp_sessions), key=lambda s: session_order.get(s, 99)
    )

    race_key = f"{year}::{race_name}"
    state = _load_practice_update_state()
    processed_sessions = set(state["races"].get(race_key, {}).get("sessions", []))
    latest_processed = set(completed_fp_sessions).issubset(processed_sessions)
    if latest_processed:
        return {"updated": False, "completed_fp_sessions": completed_fp_sessions}

    practice_new_weight = config_loader.get("baseline_predictor.practice_capture.new_weight", 0.35)
    practice_directionality_scale = config_loader.get(
        "baseline_predictor.practice_capture.directionality_scale", 0.08
    )
    practice_session_aggregation = config_loader.get(
        "baseline_predictor.practice_capture.session_aggregation", "laps_weighted"
    )
    practice_run_profile = config_loader.get(
        "baseline_predictor.practice_capture.run_profile", "balanced"
    )

    summary = update_from_testing_sessions(
        year=year,
        characteristics_year=year,
        events=[race_name],
        sessions=completed_fp_sessions,
        testing_backend="auto",
        cache_dir="data/raw/.fastf1_cache_testing",
        force_renew_cache=False,
        # Lower weight than pre-season testing to avoid abrupt directionality swings.
        new_weight=practice_new_weight,
        directionality_scale=practice_directionality_scale,
        session_aggregation=practice_session_aggregation,
        run_profile=practice_run_profile,
        dry_run=False,
    )

    state["races"][race_key] = {
        "sessions": completed_fp_sessions,
        "updated_at": datetime.now().isoformat(),
        "teams_updated": len(summary.get("updated_teams", [])),
    }
    _save_practice_update_state(state)

    return {
        "updated": True,
        "completed_fp_sessions": completed_fp_sessions,
        "teams_updated": len(summary.get("updated_teams", [])),
    }


def display_prediction_result(result: dict, prediction_name: str, is_race: bool = False):
    """Display a single prediction result (qualifying or race)."""
    st.markdown("---")
    icon = "🏎️" if is_race else "🏁"
    st.header(f"{icon} {prediction_name}")

    # Show grid source if available
    grid_source = result.get("grid_source")
    if grid_source:
        if grid_source == "ACTUAL":
            st.success("✅ Using ACTUAL grid from completed session")
        else:
            st.info("ℹ️ Using PREDICTED grid")

    # Show data source for qualifying predictions
    if not is_race:
        data_source = result.get("data_source", "Unknown")
        blend_used = result.get("blend_used", False)

        if blend_used:
            st.success(f"✅ Using {data_source} (70% practice data + 30% model)")
        else:
            st.info(f"ℹ️ {data_source}")

    characteristics_profile = result.get("characteristics_profile_used")
    teams_with_profile = result.get("teams_with_characteristics_profile", 0)
    compound_strategies = result.get("compound_strategies", {})
    pit_lap_distribution = result.get("pit_lap_distribution", {})

    if characteristics_profile and teams_with_profile:
        st.info(
            "📈 Car characteristics profile in use: "
            f"`{characteristics_profile}` ({teams_with_profile} teams)"
        )

    # Display compound strategies (multi-stint race simulation)
    if compound_strategies and is_race:
        st.subheader("🏎️ Tire Compound Strategies")

        # Sort strategies by frequency (descending)
        sorted_strategies = sorted(compound_strategies.items(), key=lambda x: x[1], reverse=True)

        # Display top 3 most common strategies
        cols = st.columns(min(3, len(sorted_strategies)))
        for idx, (strategy, frequency) in enumerate(sorted_strategies[:3]):
            with cols[idx]:
                percentage = frequency * 100
                st.metric(
                    label=strategy,
                    value=f"{percentage:.1f}%",
                    help="Frequency of this compound sequence across simulations",
                )

        # Show all strategies in expander if more than 3
        if len(sorted_strategies) > 3:
            with st.expander("📊 View all strategies"):
                for strategy, frequency in sorted_strategies:
                    percentage = frequency * 100
                    st.write(f"**{strategy}**: {percentage:.1f}%")

    # Display pit lap distribution
    if pit_lap_distribution and is_race:
        st.subheader("⏱️ Pit Stop Windows")

        # Sort by lap range start
        sorted_pit_laps = sorted(
            pit_lap_distribution.items(),
            key=lambda x: int(x[0].split("_")[1].split("-")[0]),
        )

        total_stops = sum(count for _, count in sorted_pit_laps) or 1

        # Convert to (label, count, pct)
        windows = []
        for lap_bin, count in sorted_pit_laps:
            label = lap_bin.replace("lap_", "L")  # lap_25-30 -> L25-30
            pct = 100 * (count / total_stops)
            windows.append((label, count, pct))

        # Show TOP windows by share (much more readable)
        top_windows = sorted(windows, key=lambda x: x[2], reverse=True)[:5]

        st.caption(
            "Share of all simulated pit events (all cars × all simulations). "
            "Windows are 5-lap bins, e.g. L25–30."
        )

        most_likely = top_windows[0]
        st.info(f"Most likely pit window: **{most_likely[0]}** ({most_likely[2]:.1f}%)")

        cols = st.columns(len(top_windows))
        for col, (label, count, pct) in zip(cols, top_windows, strict=False):
            with col:
                st.metric(
                    label,
                    f"{pct:.1f}%",
                    help=f"{count:,} of {total_stops:,} simulated pit events",
                )
                st.progress(min(pct / 100, 1.0))

        # Optional: show full distribution in expander
        with st.expander("View full pit stop distribution"):
            dist_df = pd.DataFrame(windows, columns=["Window", "Stops", "Share %"])
            dist_df["Share %"] = dist_df["Share %"].round(2)
            st.dataframe(dist_df, width="stretch", hide_index=True)

    # Determine which key to use for results
    results_key = "finish_order" if is_race else "grid"
    df = pd.DataFrame(result[results_key])
    df["position"] = df["position"].astype(int)

    if is_race:
        # Race display with full details
        df["confidence"] = df["confidence"].round(1)
        df["podium_probability"] = df["podium_probability"].round(1)
        df["dnf_probability"] = (df["dnf_probability"] * 100).round(1)

        df["dnf_risk"] = df["dnf_probability"].apply(
            lambda x: "⚠️ High" if x > 20 else "⚡ Medium" if x >= 10 else "✓ Low"
        )

        df_display = df[
            [
                "position",
                "driver",
                "team",
                "confidence",
                "podium_probability",
                "dnf_probability",
                "dnf_risk",
            ]
        ].copy()
        df_display.columns = [
            "Pos",
            "Driver",
            "Team",
            "Confidence %",
            "Podium %",
            "DNF Risk %",
            "Status",
        ]

        # Style the dataframe
        def color_position(val):
            if val == 1:
                return (
                    "background-color: rgba(255,215,0,0.18);"
                    "border-left: 4px solid #FFD700;"
                    "font-weight: 800;"
                    "color: rgba(237,239,243,0.95);"
                )
            if val == 2:
                return (
                    "background-color: rgba(192,192,192,0.14);"
                    "border-left: 4px solid #C0C0C0;"
                    "font-weight: 800;"
                    "color: rgba(237,239,243,0.95);"
                )
            if val == 3:
                return (
                    "background-color: rgba(205,127,50,0.16);"
                    "border-left: 4px solid #CD7F32;"
                    "font-weight: 800;"
                    "color: rgba(237,239,243,0.95);"
                )

            if val <= 10:
                return (
                    "background-color: rgba(227,242,253,0.07);"
                    "border-left: 4px solid rgba(227,242,253,0.30);"
                    "font-weight: 800;"
                    "color: rgba(237,239,243,0.95);"
                )

            return "border-left: 4px solid transparent; color: rgba(237,239,243,0.88);"

        def color_dnf_risk(val):
            if val > 20:
                return "background-color: rgba(198,40,40,0.22); color: rgba(255,255,255,0.92); font-weight: 700;"
            if val >= 10:
                return "background-color: rgba(245,127,23,0.20); color: rgba(255,255,255,0.92); font-weight: 700;"
            return "background-color: rgba(46,125,50,0.18); color: rgba(237,239,243,0.92); font-weight: 700;"

        styled_df = (
            df_display.style
            # base dark styling for ALL cells
            .set_properties(
                **{
                    "background-color": "#10141c",
                    "color": "rgba(237,239,243,0.88)",
                    "border-color": "rgba(255,255,255,0.06)",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "td",
                        "props": [
                            ("border-color", "rgba(255,255,255,0.06)"),
                            ("font-variant-numeric", "tabular-nums"),
                        ],
                    },
                    {
                        "selector": "td:nth-child(1)",
                        "props": [  # Pos
                            ("font-size", "0.98rem"),
                            ("font-weight", "800"),
                            ("text-align", "center"),
                            ("width", "64px"),
                        ],
                    },
                    {
                        "selector": "td:nth-child(4), td:nth-child(5), td:nth-child(6)",
                        "props": [  # numeric cols
                            ("font-weight", "700"),
                        ],
                    },
                ]
            )
            .map(color_position, subset=["Pos"])
            .map(color_dnf_risk, subset=["DNF Risk %"])
            .format({"Confidence %": "{:.1f}", "Podium %": "{:.1f}", "DNF Risk %": "{:.1f}"})
        )

        # Hide the index in the HTML output and keep your styling
        try:
            styled_df = styled_df.hide(axis="index")
        except Exception:
            pass  # older pandas

        st.markdown(
            f'<div class="rc-table">{styled_df.to_html()}</div>',
            unsafe_allow_html=True,
        )

        # DNF warnings
        high_dnf = df[df["dnf_probability"] > 20]
        if not high_dnf.empty:
            st.warning(
                f"⚠️ High DNF risk ({len(high_dnf)} drivers): {', '.join(high_dnf['driver'].values)}"
            )

        # Podium visualization
        st.subheader("🏆 Predicted Podium")
        podium = df[df["position"] <= 3].copy()

        col1, col2, col3 = st.columns(3)
        for i, (_idx, row) in enumerate(podium.iterrows()):
            col = [col2, col1, col3][i]
            with col:
                medal = ["🥇", "🥈", "🥉"][row["position"] - 1]
                st.markdown(f"### {medal} P{row['position']}")
                st.markdown(f"## **{row['driver']}**")
                st.markdown(f"*{row['team']}*")
                st.metric("Confidence", f"{row['confidence']:.1f}%")
                st.progress(row["confidence"] / 100)
    else:
        # Qualifying/grid display (compact)
        df_display = df[["position", "driver", "team"]].copy()
        df_display.columns = ["Grid", "Driver", "Team"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**P1-10**")
            st.markdown(
                f'<div class="rc-table">{df_display.head(10).to_html(index=False)}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("**P11-15**")
            st.markdown(
                f'<div class="rc-table">{df_display.iloc[10:15].to_html(index=False)}</div>',
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown("**P16-22**")
            st.markdown(
                f'<div class="rc-table">{df_display.iloc[15:].to_html(index=False)}</div>',
                unsafe_allow_html=True,
            )


def fetch_grid_if_available(
    year: int, race_name: str, session_name: str, predicted_grid: list
) -> tuple:
    """
    Fetch actual grid if session completed, otherwise use predicted grid.

    Returns: (grid, grid_source) where grid_source is "ACTUAL" or "PREDICTED"
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


# Cache prediction results
@st.cache_data(
    ttl=3600, show_spinner=False
)  # cache for 1 hour to avoid redundant runs during active sessions
def run_prediction(race_name: str, weather: str, _timestamps, is_sprint: bool = False) -> dict:
    """
    Run full weekend cascade prediction with caching.

    Returns dict with:
    - Normal weekend: {"qualifying": {...}, "race": {...}}
    - Sprint weekend: {"sprint_quali": {...}, "sprint_race": {...}, "main_quali": {...}, "main_race": {...}}
    """
    # Validate weather input
    valid_weather = ["dry", "rain", "mixed"]
    if weather not in valid_weather:
        raise ValueError(f"Weather must be one of {valid_weather}, got '{weather}'")

    # Start timing
    timing = {}
    overall_start = time.time()

    predictor = get_predictor(_timestamps)
    results = {}

    if is_sprint:
        # SPRINT WEEKEND CASCADE: SQ → Sprint → Main Quali → Main Race

        # 1. Sprint Qualifying Prediction
        sq_start = time.time()
        sq_result = predictor.predict_qualifying(
            year=2026,
            race_name=race_name,
            qualifying_stage="sprint",
        )
        timing["sprint_quali"] = time.time() - sq_start
        results["sprint_quali"] = sq_result

        # 2. Sprint Race Prediction (use actual SQ grid if available)
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

        # 3. Main Qualifying Prediction
        mq_start = time.time()
        mq_result = predictor.predict_qualifying(
            year=2026,
            race_name=race_name,
            qualifying_stage="main",
        )
        timing["main_quali"] = time.time() - mq_start
        results["main_quali"] = mq_result

        # 4. Main Race Prediction (use actual main quali grid if available)
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
        # NORMAL WEEKEND CASCADE: Quali → Race

        # 1. Qualifying Prediction
        quali_start = time.time()
        quali_result = predictor.predict_qualifying(
            year=2026,
            race_name=race_name,
            qualifying_stage="main",
        )
        timing["qualifying"] = time.time() - quali_start
        results["qualifying"] = quali_result

        # 2. Race Prediction (use actual quali grid if available)
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

    # Add timing to all results
    for key in results:
        results[key]["timing"] = timing

    return results


# Page config
# Page config
st.set_page_config(
    page_title="Racecraft Labs",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
# --- Retro Tech Lab (Option A) theme ---
st.markdown(
    """
<style>
/* ---- App background ---- */
[data-testid="stAppViewContainer"] {
    background: #0F1115;
}
[data-testid="stHeader"] {
    background: rgba(15,17,21,0.0);
}
[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ---- Typography ---- */
.main-header {
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  text-align: center;
  margin: 0.2rem 0 0.1rem 0;
  color: #EDEFF3;
}
.sub-header {
  font-size: 0.95rem;
  text-align: center;
  color: rgba(237,239,243,0.72);
  margin: 0 0 1.1rem 0;
}
.micro-disclaimer {
  font-size: 0.78rem;
  text-align: center;
  color: rgba(237,239,243,0.58);
  margin: 0 0 1.25rem 0;
}

/* Stronger global text defaults (Streamlit DOM-safe) */
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"] *,
[data-testid="stCaptionContainer"] *,
[data-testid="stHeader"] *,
[data-testid="stToolbar"] * {
  color: rgba(237,239,243,0.88);
}

/* Headings */
h1, h2, h3, h4 {
  color: #EDEFF3 !important;
  letter-spacing: 0.2px;
}
h1, h2 {
  text-shadow: 0 0 14px rgba(255, 46, 99, 0.10);
}

/* ---- Sidebar controls contrast ---- */
[data-testid="stSidebar"] * {
  color: rgba(237,239,243,0.86) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* ---- Inputs (readable values + placeholders) ---- */
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
  background: #10141c !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
[data-baseweb="select"] * {
  color: rgba(237,239,243,0.92) !important;
}
[data-baseweb="select"] input::placeholder {
  color: rgba(237,239,243,0.45) !important;
}
label {
  color: rgba(237,239,243,0.78) !important;
}


/* ---- Tables ---- */
[data-testid="stDataFrame"] {
  background: rgba(21,25,34,0.72);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 0.35rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* The grid itself */
[data-testid="stDataFrame"] [role="grid"] {
  background: #10141c !important;
  color: rgba(237,239,243,0.88) !important;
  border-radius: 12px;
}

/* ---- HTML tables (our controlled dark tables) ---- */
.rc-table {
  background: rgba(21,25,34,0.72);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 0.6rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  overflow: hidden;
}

.rc-table table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.92rem;
  color: rgba(237,239,243,0.88);
}

.rc-table thead th {
  background: #151922;
  color: rgba(237,239,243,0.92);
  text-align: left;
  font-weight: 700;
  padding: 0.55rem 0.7rem;
  border-bottom: 1px solid rgba(255,255,255,0.10);
}

.rc-table tbody tr:nth-child(even) td {
  background: rgba(16,20,28,0.92);
}

.rc-table table th:first-child,
.rc-table table td:first-child {
  width: 64px;
  text-align: center;
  font-weight: 800;
  color: rgba(237,239,243,0.92);
}

.rc-table table td:nth-child(2) {
  font-weight: 700;
  letter-spacing: 0.2px;
}

.rc-table tbody tr:hover td {
  background: rgba(255,255,255,0.03);
}

.rc-table { overflow: auto; }

.rc-table table th:first-child,
.rc-table table td:first-child {
  position: sticky;
  left: 0;
  z-index: 3;
  background: #10141c;
}

.rc-table table th:nth-child(2),
.rc-table table td:nth-child(2) {
  position: sticky;
  left: 64px; /* same as Pos width */
  z-index: 2;
  background: #10141c;
}

/* Header row */
[data-testid="stDataFrame"] [role="columnheader"] {
  background: #151922 !important;
  color: rgba(237,239,243,0.92) !important;
  border-bottom: 1px solid rgba(255,255,255,0.10) !important;
}

/* Body cells */
[data-testid="stDataFrame"] [role="gridcell"] {
  background: #10141c !important;
  color: rgba(237,239,243,0.88) !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}

/* Hover */
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
  background: rgba(255,255,255,0.03) !important;
}

/* Hide/neutralize Streamlit spinner/status blocks that show as white bars */
[data-testid="stSpinner"] {
  background: transparent !important;
}
[data-testid="stSpinner"] > div {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.stCacheStatus, [data-testid="stStatusWidget"] {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

/* hide Streamlit footer/menu noise */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Top-right Deploy / toolbar */
[data-testid="stToolbar"] { visibility: hidden !important; height: 0 !important; }
[data-testid="stToolbarActions"] { display: none !important; }
header[data-testid="stHeader"] { height: 0 !important; }

/* Cache + status widgets */
.stCacheStatus, [data-testid="stStatusWidget"] { display: none !important; }

/* Footer variants */
footer, [data-testid="stFooter"] { display: none !important; }
#MainMenu { visibility: hidden !important; }

[data-testid="stSpinner"] { display: none !important; }

</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">Racecraft Labs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Race Weekend Prediction Engine</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="micro-disclaimer">Independent fan project • not affiliated with any racing series, teams, or governing bodies</div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=240)
    else:
        st.markdown("### Racecraft Labs")

    # st.caption("Retro Tech Lab Edition")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Live Prediction", "Model Insights", "Prediction Accuracy", "About"],
    )

    st.markdown("---")

    # Prediction logging toggle
    st.markdown("**⚙️ Settings**")
    enable_logging = st.checkbox(
        "Save Predictions for Accuracy Tracking",
        value=False,
        help=(
            "When enabled, predictions are saved after each session (FP1/FP2/FP3/SQ) "
            "for later accuracy analysis. Max 1 prediction per session."
        ),
    )

    st.markdown("---")
    st.markdown("**Model Version:** v1.0")
    st.markdown("**Last Updated:** 2026-02-01")


# Page: Live Prediction
if page == "Live Prediction":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Race Weekend Prediction")

    # Load 2026 calendar dynamically
    try:
        schedule = fastf1.get_event_schedule(2026)
        # Filter out testing sessions - only include actual races
        race_events = schedule[
            (schedule["EventFormat"].notna())
            & (~schedule["EventName"].str.contains("Testing", case=False, na=False))
        ].copy()
        race_names = race_events["EventName"].tolist()

        # Add sprint indicator to race names (text only, no emoji)
        race_options = []
        for _, event in race_events.iterrows():
            race_name = event["EventName"]
            event_format = str(event["EventFormat"]).lower()
            if "sprint" in event_format:
                race_options.append(f"{race_name} (Sprint)")
            else:
                race_options.append(race_name)
    except Exception as e:
        st.error(f"Failed to load 2026 calendar: {e}")
        race_options = [
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Australian Grand Prix",
            "Japanese Grand Prix",
            "Chinese Grand Prix",
            "Miami Grand Prix",
        ]

    col1, col2 = st.columns(2)

    with col1:
        race_selection = st.selectbox("Select Grand Prix", race_options)
        # Remove sprint indicator for processing
        race_name = race_selection.replace(" (Sprint)", "")

    with col2:
        weather = st.selectbox("Weather Forecast", ["dry", "rain", "mixed"])

    if st.button("Generate Prediction", type="primary"):
        # STEP 0: Auto-update from any new completed races
        auto_update_if_needed()

        with st.spinner("Running simulation..."):
            try:
                from src.utils.weekend import is_sprint_weekend

                # Check if sprint weekend
                try:
                    is_sprint = is_sprint_weekend(2026, race_name)
                except (ValueError, KeyError, FileNotFoundError) as e:
                    logger.warning(f"Could not determine sprint weekend status: {e}")
                    is_sprint = False

                # Show warnings based on data freshness
                st.warning("⚠️ 2026 regulation reset - predictions uncertain until races complete")

                if is_sprint:
                    st.info(
                        "🏃 **Sprint Weekend** - System predicts Sprint Qualifying (Friday) → "
                        "Sprint Race (Saturday) → Sunday Qualifying → Sunday Race. "
                        "Sprint predictions use adjusted chaos modeling "
                        "(30% less variance, grid position +10% importance)."
                    )

                # STEP 1: Learn from completed FP sessions for this weekend.
                try:
                    practice_update = auto_update_practice_characteristics_if_needed(
                        year=2026,
                        race_name=race_name,
                        is_sprint=is_sprint,
                    )
                    if practice_update.get("updated"):
                        st.success(
                            "✅ Updated car characteristics from completed practice sessions: "
                            f"{', '.join(practice_update['completed_fp_sessions'])} "
                            f"({practice_update['teams_updated']} teams)"
                        )
                        st.cache_resource.clear()
                        st.cache_data.clear()
                    elif practice_update.get("completed_fp_sessions"):
                        st.info(
                            "ℹ️ Practice characteristics already up to date for sessions: "
                            f"{', '.join(practice_update['completed_fp_sessions'])}"
                        )
                except Exception as practice_exc:
                    st.warning(
                        "⚠️ Could not update practice characteristics automatically; "
                        f"continuing with current data ({practice_exc})"
                    )

                # Get current file timestamps for cache invalidation
                timestamps = get_data_file_timestamps()

                # Use cached prediction (invalidates if files changed)
                st.info("Running simulation (cached results will load instantly)...")
                prediction_results = run_prediction(race_name, weather, timestamps, is_sprint)

                # Save prediction if logging is enabled
                if enable_logging:
                    from src.utils.prediction_logger import PredictionLogger
                    from src.utils.session_detector import SessionDetector

                    detector = SessionDetector()
                    logger_inst = PredictionLogger()

                    # Get latest completed session
                    latest_session = detector.get_latest_completed_session(
                        2026, race_name, is_sprint
                    )

                    if latest_session:
                        # Check if we already saved a prediction for this session
                        if not logger_inst.has_prediction_for_session(
                            2026, race_name, latest_session
                        ):
                            # Save the prediction (use appropriate keys from prediction_results)
                            try:
                                # Get qualifying and race predictions based on weekend type
                                if is_sprint:
                                    # For sprint, save main quali and main race
                                    quali_grid = prediction_results["main_quali"]["grid"]
                                    race_finish = prediction_results["main_race"]["finish_order"]
                                    fp_blend_info = prediction_results.get("main_quali", {}).get(
                                        "fp_blend_info", {}
                                    )
                                else:
                                    quali_grid = prediction_results["qualifying"]["grid"]
                                    race_finish = prediction_results["race"]["finish_order"]
                                    fp_blend_info = prediction_results.get("qualifying", {}).get(
                                        "fp_blend_info", {}
                                    )

                                logger_inst.save_prediction(
                                    year=2026,
                                    race_name=race_name,
                                    session_name=latest_session,
                                    qualifying_prediction=quali_grid,
                                    race_prediction=race_finish,
                                    weather=weather,
                                    fp_blend_info=fp_blend_info,
                                )
                                st.info(
                                    f"📊 Prediction saved for accuracy tracking (after {latest_session})"
                                )
                            except Exception as e:
                                st.warning(f"Could not save prediction: {e}")
                        else:
                            st.info(
                                f"ℹ️ Prediction for {latest_session} already saved (max 1 per session)"
                            )
                    else:
                        st.info(
                            "ℹ️ No completed sessions yet - prediction not saved (will save after FP1/FP2/FP3/SQ)"
                        )

                # Display results with performance timing
                first_result = list(prediction_results.values())[0]
                timing = first_result.get("timing", {})
                if timing:
                    st.success(f"✅ Predictions complete in {timing['total']:.2f}s")
                else:
                    st.success("✅ Predictions complete!")

                # ========== WEEKEND CASCADE DISPLAY ==========
                if is_sprint:
                    # SPRINT WEEKEND: Show all 4 predictions
                    st.markdown("---")
                    st.header("🏃 Sprint Weekend Cascade")
                    st.info(
                        "Full weekend flow: Sprint Qualifying → Sprint Race → Main Qualifying → Main Race"
                    )

                    # 1. Sprint Qualifying Prediction
                    display_prediction_result(
                        prediction_results["sprint_quali"],
                        "Sprint Qualifying Prediction",
                        is_race=False,
                    )

                    # 2. Sprint Race Prediction
                    display_prediction_result(
                        prediction_results["sprint_race"],
                        "Sprint Race Prediction",
                        is_race=True,
                    )

                    # 3. Main Qualifying Prediction
                    display_prediction_result(
                        prediction_results["main_quali"],
                        "Main Qualifying Prediction",
                        is_race=False,
                    )

                    # 4. Main Race Prediction
                    display_prediction_result(
                        prediction_results["main_race"],
                        "Main Race Prediction",
                        is_race=True,
                    )
                else:
                    # NORMAL WEEKEND: Show 2 predictions
                    st.markdown("---")
                    st.header("🏁 Normal Weekend Cascade")
                    st.info("Weekend flow: Qualifying → Race")

                    # 1. Qualifying Prediction
                    display_prediction_result(
                        prediction_results["qualifying"],
                        "Qualifying Prediction",
                        is_race=False,
                    )

                    # 2. Race Prediction
                    display_prediction_result(
                        prediction_results["race"], "Race Prediction", is_race=True
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Make sure data files are generated. Run: "
                    "`python scripts/extract_driver_characteristics.py --years 2023,2024,2025`"
                )

    st.markdown("</div>", unsafe_allow_html=True)

# Page: Model Insights
elif page == "Model Insights":
    st.header("How the Model Works")

    st.markdown("""
    ### Runtime path

    The dashboard currently runs `Baseline2026Predictor` for both qualifying and race.

    **1. Team strength**
    - Uses baseline (pre-season), testing directionality, and current-season performance
    - Applies a race-by-race weight schedule that quickly shifts toward current-season data

    **2. Qualifying**
    - Pulls the best available session data (normal weekend: FP3 > FP2 > FP1)
    - Blends session pace with model strength (fixed 70/30 in the active predictor)
    - Applies a small short-run characteristics modifier when profile data exists
    - Runs Monte Carlo simulations and reports the median grid with confidence ranges

    **3. Race**
    - Uses either predicted qualifying grid or actual qualifying results when available
    - Scores drivers with grid position, team pace, driver skill, overtaking context, and stochastic effects
    - Applies a small long-run characteristics modifier when profile data exists
    - Includes lap-one chaos, strategy variance, safety car luck, and DNF probability

    **4. What exists in the repo but is not a direct race score term here**
    - Bayesian ranking components
    - Testing updater outputs (including tire degradation slope estimates)

    **5. Learning behavior**
    - Auto-updater can ingest completed races into team characteristics
    - Learning history tracks method MAE
    - In this runtime path, qualifying blend weight is fixed inside the predictor
    """)

    st.subheader("Key Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Qualifying (active path):**
        - Team/driver score: 70% team + 30% driver
        - Practice blend: 70% session pace + 30% model strength
        - Output: median grid from Monte Carlo runs
        """)

    with col2:
        st.markdown("""
        **Race (active path):**
        - Base pace weight: 40% (track-adjusted)
        - Grid influence: dynamic by overtaking difficulty
        - Driver skill term: 20%
        - DNF probability + chaos + strategy + safety car modifiers
        """)


# Page: Prediction Accuracy
elif page == "Prediction Accuracy":
    st.header("📊 Prediction Accuracy Tracker")

    from src.utils.prediction_logger import PredictionLogger
    from src.utils.prediction_metrics import PredictionMetrics

    logger_inst = PredictionLogger()
    metrics_calc = PredictionMetrics()

    # Load all predictions
    all_predictions = logger_inst.get_all_predictions(2026)

    if not all_predictions:
        st.info(
            "No predictions saved yet. Enable 'Save Predictions for Accuracy Tracking' "
            "in the sidebar and generate predictions after practice sessions."
        )
    else:
        st.success(f"Found {len(all_predictions)} saved prediction(s)")

        # Calculate metrics for predictions with actuals
        predictions_with_actuals = [
            p
            for p in all_predictions
            if p.get("actuals") and (p["actuals"].get("qualifying") or p["actuals"].get("race"))
        ]

        if predictions_with_actuals:
            st.markdown("---")
            st.subheader("📈 Overall Accuracy")

            # Aggregate metrics
            agg_metrics = metrics_calc.aggregate_metrics(predictions_with_actuals)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Qualifying Metrics**")
                if "qualifying" in agg_metrics:
                    q_metrics = agg_metrics["qualifying"]
                    st.metric(
                        "Exact Position Accuracy",
                        f"{q_metrics['exact_accuracy']['mean']:.1f}%",
                        help="% of drivers predicted in exact correct position",
                    )
                    st.metric(
                        "Mean Position Error (MAE)",
                        f"{q_metrics['mae']['mean']:.2f} positions",
                        help="Average position error",
                    )
                    st.metric(
                        "Within ±3 Positions",
                        f"{q_metrics['within_3']['mean']:.1f}%",
                        help="% of predictions within 3 positions",
                    )
                    st.metric(
                        "Correlation",
                        f"{q_metrics['correlation']['mean']:.3f}",
                        help="Spearman correlation (-1 to 1, higher is better)",
                    )

            with col2:
                st.markdown("**Race Metrics**")
                if "race" in agg_metrics:
                    r_metrics = agg_metrics["race"]
                    st.metric(
                        "Exact Position Accuracy",
                        f"{r_metrics['exact_accuracy']['mean']:.1f}%",
                        help="% of drivers predicted in exact correct position",
                    )
                    st.metric(
                        "Mean Position Error (MAE)",
                        f"{r_metrics['mae']['mean']:.2f} positions",
                        help="Average position error",
                    )
                    st.metric(
                        "Within ±3 Positions",
                        f"{r_metrics['within_3']['mean']:.1f}%",
                        help="% of predictions within 3 positions",
                    )
                    st.metric(
                        "Winner Prediction Accuracy",
                        f"{r_metrics['winner_accuracy']['percentage']:.1f}%",
                        help="% of races where winner was correctly predicted",
                    )

            st.markdown("---")
            st.subheader("🎯 Per-Race Breakdown")

            # Show metrics for each prediction
            for pred in predictions_with_actuals:
                metrics = metrics_calc.calculate_all_metrics(pred)
                if metrics:
                    race_name = metrics["metadata"]["race_name"]
                    session_name = metrics["metadata"]["session_name"]

                    with st.expander(f"{race_name} (Predicted after {session_name})"):
                        col1, col2 = st.columns(2)

                        with col1:
                            if "qualifying" in metrics:
                                st.markdown("**Qualifying**")
                                q = metrics["qualifying"]
                                st.write(f"- Exact: {q['exact_accuracy']:.1f}%")
                                st.write(f"- MAE: {q['mae']:.2f} positions")
                                st.write(f"- Within ±1: {q['within_1']:.1f}%")
                                st.write(f"- Correlation: {q['correlation']:.3f}")

                        with col2:
                            if "race" in metrics:
                                st.markdown("**Race**")
                                r = metrics["race"]
                                st.write(f"- Exact: {r['exact_accuracy']:.1f}%")
                                st.write(f"- MAE: {r['mae']:.2f} positions")
                                st.write(f"- Within ±3: {r['within_3']:.1f}%")
                                st.write(
                                    f"- Winner: {'✅ Correct' if r['winner_correct'] else '❌ Wrong'}"
                                )
                                st.write(
                                    f"- Podium: {r['podium']['correct_drivers']}/3 drivers correct"
                                )
        else:
            st.info(
                "Predictions saved, but no actual results added yet. After each race, "
                "you can update predictions with actual results to calculate accuracy."
            )

        st.markdown("---")
        st.subheader("📋 All Saved Predictions")

        # Show list of all predictions
        for pred in all_predictions:
            metadata = pred["metadata"]
            race_name = metadata["race_name"]
            session_name = metadata["session_name"]
            predicted_at = metadata["predicted_at"]
            has_actuals = bool(
                pred.get("actuals")
                and (pred["actuals"].get("qualifying") or pred["actuals"].get("race"))
            )

            status_icon = "✅" if has_actuals else "⏳"
            status_text = "Results added" if has_actuals else "Awaiting results"

            st.write(f"{status_icon} **{race_name}** (after {session_name}) - {status_text}")


# Page: About
else:
    st.header("About This Project")

    st.markdown("""
    #### Racecraft Labs Prediction Engine

    This project focuses on weekend predictions for the 2026 season.

    The core runtime in the app is:
    - Weight-scheduled team strength blending (baseline/testing/current season)
    - Practice-session blending for qualifying when data is available
    - Monte Carlo race simulation using grid, pace, skill, and reliability signals

    **Technology stack**
    - Python
    - FastF1 for session data
    - NumPy and pandas for modeling/data handling
    - Streamlit for the UI

    **Validation and testing**
    - Pytest coverage for key modules
    - Notebooks for schedule/blending validation
    - Session-based prediction tracking in the dashboard

    ---

    **Author:** Tomasz Solis

    **[GitHub](https://github.com/tomasz-solis)**

    **[LinkedIn](https://linkedin.com/in/tomaszsolis)**

    Private repository
    """)

    st.info(
        "Independent project for analysis and experimentation. Not affiliated with any racing series, teams, or governing bodies."
    )


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color: rgba(237,239,243,0.55); padding: 0.75rem 0;">Built with ❤️ for racing fans and data nerds</div>',
    unsafe_allow_html=True,
)
