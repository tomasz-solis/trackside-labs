"""
Streamlit Dashboard for F1 2026 Predictions

Live race predictions with historical accuracy tracking.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import fastf1
import time


# Get file modification times for cache invalidation
def get_data_file_timestamps():
    """Get modification timestamps of all data files."""
    from pathlib import Path
    import os

    files = [
        "data/processed/car_characteristics/2026_car_characteristics.json",
        "data/processed/driver_characteristics.json",
        "data/processed/track_characteristics/2026_track_characteristics.json",
        "data/2025_pirelli_info.json",  # Tire characteristics (fallback for 2026)
        "data/2026_pirelli_info.json",  # Tire characteristics (if available)
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
    from src.predictors.baseline_2026 import Baseline2026Predictor
    import logging

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


# Cache prediction results
@st.cache_data(ttl=3600)  # Cache for 1 hour
def run_prediction(race_name: str, weather: str, _timestamps):
    """Run prediction with caching (1 hour TTL). Weather must be 'dry', 'rain', or 'mixed'."""
    # Validate weather input
    valid_weather = ["dry", "rain", "mixed"]
    if weather not in valid_weather:
        raise ValueError(f"Weather must be one of {valid_weather}, got '{weather}'")

    # Start timing
    timing = {}
    overall_start = time.time()

    predictor = get_predictor(_timestamps)

    # STEP 1: Predict Qualifying (includes FP auto-fetch)
    quali_start = time.time()
    quali_result = predictor.predict_qualifying(year=2026, race_name=race_name)
    timing["qualifying"] = time.time() - quali_start

    # STEP 2: Predict Race using quali results
    race_start = time.time()
    race_result = predictor.predict_race(
        qualifying_grid=quali_result["grid"], weather=weather, race_name=race_name, n_simulations=50
    )
    timing["race"] = time.time() - race_start

    timing["total"] = time.time() - overall_start

    # Add timing to results
    quali_result["timing"] = timing
    race_result["timing"] = timing

    return quali_result, race_result


# Page config
st.set_page_config(
    page_title="F1 2026 Predictions",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #e10600;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e10600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🏎️ Formula 1 2026 Predictions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Physics-Based Race Simulation Engine</div>', unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png",
        width=150,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation", ["Live Prediction", "Model Insights", "Prediction Accuracy", "About"]
    )

    st.markdown("---")

    # Prediction logging toggle
    st.markdown("**⚙️ Settings**")
    enable_logging = st.checkbox(
        "Save Predictions for Accuracy Tracking",
        value=False,
        help="When enabled, predictions are saved after each session (FP1/FP2/FP3/SQ) for later accuracy analysis. Max 1 prediction per session.",
    )

    st.markdown("---")
    st.markdown("**Model Version:** v1.0")
    st.markdown("**Last Updated:** 2026-02-01")


# Page: Live Prediction
if page == "Live Prediction":
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

        # Add sprint indicator to race names
        race_options = []
        for _, event in race_events.iterrows():
            race_name = event["EventName"]
            event_format = str(event["EventFormat"]).lower()
            if "sprint" in event_format:
                race_options.append(f"{race_name} 🏃 (Sprint)")
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
        race_name = race_selection.replace(" 🏃 (Sprint)", "")

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
                        "🏃 **Sprint Weekend** - System predicts Sprint Qualifying (Friday) → Sprint Race (Saturday) → Sunday Qualifying → Sunday Race. "
                        "Sprint predictions use adjusted chaos modeling (30% less variance, grid position +10% importance)."
                    )

                # Get current file timestamps for cache invalidation
                timestamps = get_data_file_timestamps()

                # Use cached prediction (invalidates if files changed)
                st.info("Running simulation (cached results will load instantly)...")
                quali_result, race_result = run_prediction(race_name, weather, timestamps)

                # Save prediction if logging is enabled
                if enable_logging:
                    from src.utils.session_detector import SessionDetector
                    from src.utils.prediction_logger import PredictionLogger

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
                            # Save the prediction
                            try:
                                fp_blend_info = quali_result.get("fp_blend_info", {})
                                logger_inst.save_prediction(
                                    year=2026,
                                    race_name=race_name,
                                    session_name=latest_session,
                                    qualifying_prediction=quali_result["grid"],
                                    race_prediction=race_result["finish_order"],
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
                timing = quali_result.get("timing", {})
                if timing:
                    st.success(
                        f"✅ Predictions complete in {timing['total']:.2f}s "
                        f"(Qualifying: {timing['qualifying']:.2f}s, Race: {timing['race']:.2f}s)"
                    )
                else:
                    st.success("✅ Predictions complete!")

                # ========== QUALIFYING PREDICTION ==========
                st.markdown("---")
                st.header("🏁 Qualifying Prediction")

                # Show data source used for prediction
                data_source = quali_result.get("data_source", "Unknown")
                blend_used = quali_result.get("blend_used", False)

                if blend_used:
                    st.success(f"✅ Using {data_source} (70% practice data + 30% model)")
                else:
                    st.info(f"ℹ️ {data_source}")

                df_quali = pd.DataFrame(quali_result["grid"])
                df_quali["position"] = df_quali["position"].astype(int)
                df_quali["confidence"] = (
                    df_quali.get("confidence", 70.0).round(1) if "confidence" in df_quali else 70.0
                )

                df_quali_display = df_quali[["position", "driver", "team"]].copy()
                df_quali_display.columns = ["Grid", "Driver", "Team"]

                # Show quali grid in 3 columns: 1-10, 11-15, 16-22
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**P1-10**")
                    st.dataframe(df_quali_display.head(10), width="stretch", hide_index=True)

                with col2:
                    st.markdown("**P11-15**")
                    st.dataframe(df_quali_display.iloc[10:15], width="stretch", hide_index=True)

                with col3:
                    st.markdown("**P16-22**")
                    st.dataframe(df_quali_display.iloc[15:], width="stretch", hide_index=True)

                # ========== RACE PREDICTION ==========
                st.markdown("---")
                st.header("🏎️ Race Prediction")

                df = pd.DataFrame(race_result["finish_order"])
                df["position"] = df["position"].astype(int)
                df["confidence"] = df["confidence"].round(1)
                df["podium_probability"] = df["podium_probability"].round(1)
                df["dnf_probability"] = (df["dnf_probability"] * 100).round(1)

                # Add DNF risk indicator
                df["dnf_risk"] = df["dnf_probability"].apply(
                    lambda x: "⚠️ High" if x > 20 else "⚡ Medium" if x >= 10 else "✓ Low"
                )

                # Format for display
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
                    if val <= 3:
                        colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
                        return f"background-color: {colors[val]}; font-weight: bold; color: black"
                    elif val <= 10:
                        return "background-color: #e3f2fd; font-weight: bold"
                    return ""

                def color_dnf_risk(val):
                    if val > 20:
                        return "background-color: #ffcdd2; color: #c62828"  # Red
                    elif val >= 10:
                        return "background-color: #fff9c4; color: #f57f17"  # Yellow
                    return "background-color: #c8e6c9; color: #2e7d32"  # Green

                styled_df = (
                    df_display.style.map(color_position, subset=["Pos"])
                    .map(color_dnf_risk, subset=["DNF Risk %"])
                    .format(
                        {"Confidence %": "{:.1f}", "Podium %": "{:.1f}", "DNF Risk %": "{:.1f}"}
                    )
                )

                st.dataframe(styled_df, width="stretch", height=700)

                # DNF warnings
                high_dnf = df[df["dnf_probability"] > 20]
                if not high_dnf.empty:
                    st.warning(
                        f"⚠️ High DNF risk ({len(high_dnf)} drivers): {', '.join(high_dnf['driver'].values)}"
                    )

                # Podium visualization - more prominent
                st.subheader("🏆 Predicted Podium")
                podium = df[df["position"] <= 3].copy()

                col1, col2, col3 = st.columns(3)

                for i, (idx, row) in enumerate(podium.iterrows()):
                    col = [col2, col1, col3][i]  # P1 center, P2 left, P3 right
                    with col:
                        medal = ["🥇", "🥈", "🥉"][row["position"] - 1]
                        st.markdown(f"### {medal} P{row['position']}")
                        st.markdown(f"## **{row['driver']}**")
                        st.markdown(f"*{row['team']}*")
                        st.metric("Confidence", f"{row['confidence']:.1f}%")
                        st.progress(row["confidence"] / 100)

                # Split into two columns for better layout
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("📊 Confidence Distribution")

                    # Normalize confidence to 0-100 for better visual spread
                    df_viz = df.copy()
                    conf_min = df_viz["confidence"].min()
                    conf_max = df_viz["confidence"].max()

                    fig_conf = go.Figure()

                    # Add bars with custom colors based on position
                    colors = [
                        "#2ecc71" if i < 3 else "#3498db" if i < 10 else "#95a5a6"
                        for i in range(len(df_viz))
                    ]

                    fig_conf.add_trace(
                        go.Bar(
                            x=df_viz["driver"],
                            y=df_viz["confidence"],
                            marker_color=colors,
                            text=df_viz["confidence"].round(1),
                            textposition="outside",
                            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>",
                        )
                    )

                    fig_conf.update_layout(
                        height=500,
                        yaxis_title="Confidence (%)",
                        xaxis_title="Driver",
                        yaxis_range=[max(0, conf_min - 5), min(100, conf_max + 5)],
                        showlegend=False,
                        xaxis={"tickangle": -45},
                    )

                    st.plotly_chart(fig_conf, width="stretch")
                    st.caption(
                        f"Range: {conf_min:.1f}% - {conf_max:.1f}% | Avg: {df_viz['confidence'].mean():.1f}%"
                    )

                with col_right:
                    st.subheader("🎯 Key Stats")

                    st.info(
                        "⚠️ **2026 Reg Reset**: High uncertainty. Model learns from testing (Feb 2026) and races."
                    )

                    # Winner prediction
                    winner = df.iloc[0]
                    st.metric(
                        "Predicted Winner",
                        winner["driver"],
                        f"{winner['confidence']:.1f}% confidence",
                    )

                    # Podium certainty
                    podium_avg = df[df["position"] <= 3]["confidence"].mean()
                    st.metric("Podium Certainty", f"{podium_avg:.1f}%", "Average top 3")

                    # Average confidence
                    avg_confidence = df["confidence"].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%", "Low until testing")

                    # DNF predictions
                    high_dnf_count = len(df[df["dnf_probability"] > 20])
                    st.metric("High DNF Risk", f"{high_dnf_count}", "⚠️ >20% chance")

                    # Highest uncertainty
                    most_uncertain = df.loc[df["confidence"].idxmin()]
                    st.metric(
                        "Most Uncertain",
                        most_uncertain["driver"],
                        f"P{most_uncertain['position']} - {most_uncertain['confidence']:.1f}%",
                    )

                    st.markdown("---")
                    st.markdown("**Color Legend:**")
                    st.markdown("🟢 Podium | 🔵 Points | ⚪ No points")
                    st.markdown("**DNF Risk:**")
                    st.markdown("🟢 Low (<10%) | 🟡 Medium (10-20%) | 🔴 High (>20%)")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Make sure data files are generated. Run: `python scripts/extract_driver_characteristics_fixed.py --years 2023,2024,2025`"
                )


# Page: Model Insights
elif page == "Model Insights":
    st.header("How the Model Works")

    st.markdown("""
    ### Physics-First Approach

    Unlike typical ML models that treat F1 as a black box, this system simulates the actual physics:

    **1. Tire Degradation Model**
    - Measures deg slope from practice laps (seconds lost per lap)
    - Converts to cumulative race penalty (not discrete pit stops)
    - Accounts for driver tire management skill

    **2. Bayesian Driver Rankings**
    - Tracks each driver's performance rating with uncertainty
    - Updates beliefs after every race (Conjugate Normal-Normal)
    - Detects concept drift (upgrades, regulation changes)

    **3. Race Simulation**
    - Lap 1 chaos (variance by grid position)
    - Pace advantage limited by overtaking difficulty
    - Weather acts as skill multiplier
    - DNF probability from reliability + driver errors

    **4. Adaptive Learning**
    - Tracks prediction accuracy per method
    - Adjusts blend weights (practice vs model)
    - Improves over the season
    """)

    st.subheader("Key Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Bayesian Model:**
        - Base Volatility: 0.1
        - Observation Noise: 2.0
        - Shock Threshold: 2σ
        """)

    with col2:
        st.markdown("""
        **Race Simulation:**
        - Pace Weight: 40%
        - Grid Weight: 30%
        - Tire Deg Weight: 15%
        - Overtaking Weight: 15%
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
            "No predictions saved yet. Enable 'Save Predictions for Accuracy Tracking' in the sidebar and generate predictions after practice sessions."
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
                "Predictions saved, but no actual results added yet. After each race, you can update predictions with actual results to calculate accuracy."
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
    ### Formula 1 2026 Predictive Engine

    A physics-based simulation system for predicting F1 race outcomes under the 2026 regulation changes.

    **Why 2026?**

    Major regulation reset:
    - 50/50 electric/ICE power units (vs current 80/20)
    - Active aerodynamics
    - 30kg lighter cars
    - New teams (Cadillac)

    When regulations change, historical performance matters less. This model focuses on:
    - Car performance from practice telemetry
    - Driver skill in racecraft, tire management, consistency
    - Track-specific factors (overtaking, pit loss)

    **Technology Stack:**
    - Python 3.9+
    - FastF1 for telemetry data
    - NumPy/SciPy for Bayesian inference
    - Streamlit for visualization

    **Testing:**
    - 41 unit tests with pytest
    - Validated against 2025 season
    - CI/CD with GitHub Actions

    ---

    **Author:** Tomasz Solis 

    **[GitHub](https://github.com/tomasz-solis)**

    **[LinkedIn](linkedin.com/in/tomaszsolis)**

    MIT License
    """)

    st.info(
        "This is a passion project showcasing data science and domain expertise. Not affiliated with FIA or Formula 1."
    )


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">Built with ❤️ for F1 fans and data nerds</div>',
    unsafe_allow_html=True,
)
