"""Dashboard pages and page-level orchestration."""

import logging

import fastf1
import streamlit as st

from .cache import get_data_file_timestamps
from .prediction_flow import run_prediction
from .rendering import display_prediction_result
from .update_flow import auto_update_if_needed, auto_update_practice_characteristics_if_needed

logger = logging.getLogger(__name__)


def _load_race_options() -> list[str]:
    """Load race options from FastF1 schedule with sprint labels."""
    try:
        schedule = fastf1.get_event_schedule(2026)
        race_events = schedule[
            (schedule["EventFormat"].notna())
            & (~schedule["EventName"].str.contains("Testing", case=False, na=False))
        ].copy()

        race_options = []
        for _, event in race_events.iterrows():
            race_name = event["EventName"]
            event_format = str(event["EventFormat"]).lower()
            if "sprint" in event_format:
                race_options.append(f"{race_name} (Sprint)")
            else:
                race_options.append(race_name)

        return race_options
    except Exception as e:
        st.error(f"Failed to load 2026 calendar: {e}")
        return [
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Australian Grand Prix",
            "Japanese Grand Prix",
            "Chinese Grand Prix",
            "Miami Grand Prix",
        ]


def _save_prediction_if_enabled(
    enable_logging: bool,
    prediction_results: dict,
    is_sprint: bool,
    race_name: str,
    weather: str,
) -> None:
    if not enable_logging:
        return

    from src.utils.prediction_logger import PredictionLogger
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    logger_inst = PredictionLogger()

    latest_session = detector.get_latest_completed_session(2026, race_name, is_sprint)

    if latest_session:
        if not logger_inst.has_prediction_for_session(2026, race_name, latest_session):
            try:
                if is_sprint:
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
                st.info(f"📊 Prediction saved for accuracy tracking (after {latest_session})")
            except Exception as e:
                st.warning(f"Could not save prediction: {e}")
        else:
            st.info(f"ℹ️ Prediction for {latest_session} already saved (max 1 per session)")
    else:
        st.info(
            "ℹ️ No completed sessions yet - prediction not saved (will save after FP1/FP2/FP3/SQ)"
        )


def _render_prediction_results(prediction_results: dict, is_sprint: bool) -> None:
    first_result = list(prediction_results.values())[0]
    timing = first_result.get("timing", {})
    if timing:
        st.success(f"✅ Predictions complete in {timing['total']:.2f}s")
    else:
        st.success("✅ Predictions complete!")

    if is_sprint:
        st.markdown("---")
        st.header("🏃 Sprint Weekend Cascade")
        st.info("Full weekend flow: Sprint Qualifying → Sprint Race → Main Qualifying → Main Race")

        display_prediction_result(
            prediction_results["sprint_quali"],
            "Sprint Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["sprint_race"],
            "Sprint Race Prediction",
            is_race=True,
        )
        display_prediction_result(
            prediction_results["main_quali"],
            "Main Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["main_race"],
            "Main Race Prediction",
            is_race=True,
        )
    else:
        st.markdown("---")
        st.header("🏁 Normal Weekend Cascade")
        st.info("Weekend flow: Qualifying → Race")

        display_prediction_result(
            prediction_results["qualifying"],
            "Qualifying Prediction",
            is_race=False,
        )
        display_prediction_result(
            prediction_results["race"],
            "Race Prediction",
            is_race=True,
        )


def render_live_prediction_page(enable_logging: bool) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Race Weekend Prediction")

    race_options = _load_race_options()

    col1, col2 = st.columns(2)

    with col1:
        race_selection = st.selectbox("Select Grand Prix", race_options)
        race_name = race_selection.replace(" (Sprint)", "")

    with col2:
        weather = st.selectbox("Weather Forecast", ["dry", "rain", "mixed"])

    if st.button("Generate Prediction", type="primary"):
        auto_update_if_needed()

        with st.spinner("Running simulation..."):
            try:
                from src.utils.weekend import is_sprint_weekend

                try:
                    is_sprint = is_sprint_weekend(2026, race_name)
                except (ValueError, KeyError, FileNotFoundError) as e:
                    logger.warning(f"Could not determine sprint weekend status: {e}")
                    is_sprint = False

                st.warning("⚠️ 2026 regulation reset - predictions uncertain until races complete")

                if is_sprint:
                    st.info(
                        "🏃 **Sprint Weekend** - System predicts Sprint Qualifying (Friday) → "
                        "Sprint Race (Saturday) → Sunday Qualifying → Sunday Race. "
                        "Sprint predictions use adjusted chaos modeling "
                        "(30% less variance, grid position +10% importance)."
                    )

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

                timestamps = get_data_file_timestamps()

                st.info("Running simulation (cached results will load instantly)...")
                prediction_results = run_prediction(race_name, weather, timestamps, is_sprint)

                _save_prediction_if_enabled(
                    enable_logging=enable_logging,
                    prediction_results=prediction_results,
                    is_sprint=is_sprint,
                    race_name=race_name,
                    weather=weather,
                )

                _render_prediction_results(prediction_results, is_sprint)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Make sure data files are generated. Run: "
                    "`python scripts/extract_driver_characteristics.py --years 2023,2024,2025`"
                )

    st.markdown("</div>", unsafe_allow_html=True)


def render_model_insights_page() -> None:
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


def render_prediction_accuracy_page() -> None:
    st.header("📊 Prediction Accuracy Tracker")

    from src.utils.prediction_logger import PredictionLogger
    from src.utils.prediction_metrics import PredictionMetrics

    logger_inst = PredictionLogger()
    metrics_calc = PredictionMetrics()

    all_predictions = logger_inst.get_all_predictions(2026)

    if not all_predictions:
        st.info(
            "No predictions saved yet. Enable 'Save Predictions for Accuracy Tracking' "
            "in the sidebar and generate predictions after practice sessions."
        )
        return

    st.success(f"Found {len(all_predictions)} saved prediction(s)")

    predictions_with_actuals = [
        p
        for p in all_predictions
        if p.get("actuals") and (p["actuals"].get("qualifying") or p["actuals"].get("race"))
    ]

    if predictions_with_actuals:
        st.markdown("---")
        st.subheader("📈 Overall Accuracy")

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

    for pred in all_predictions:
        metadata = pred["metadata"]
        race_name = metadata["race_name"]
        session_name = metadata["session_name"]
        has_actuals = bool(
            pred.get("actuals")
            and (pred["actuals"].get("qualifying") or pred["actuals"].get("race"))
        )

        status_icon = "✅" if has_actuals else "⏳"
        status_text = "Results added" if has_actuals else "Awaiting results"

        st.write(f"{status_icon} **{race_name}** (after {session_name}) - {status_text}")


def render_about_page() -> None:
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


def render_page(page: str, enable_logging: bool) -> None:
    if page == "Live Prediction":
        render_live_prediction_page(enable_logging)
    elif page == "Model Insights":
        render_model_insights_page()
    elif page == "Prediction Accuracy":
        render_prediction_accuracy_page()
    else:
        render_about_page()
