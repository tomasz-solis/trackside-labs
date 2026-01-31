"""
Streamlit Dashboard for F1 2026 Predictions

Live race predictions with historical accuracy tracking.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import fastf1

# Page config
st.set_page_config(
    page_title="F1 2026 Predictions",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏎️ Formula 1 2026 Predictions</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Physics-Based Race Simulation Engine</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png", width=150)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Live Prediction", "Model Insights", "About"]
    )

    st.markdown("---")
    st.markdown("**Model Version:** v1.0")
    st.markdown("**Last Updated:** 2026-01-28")


# Page: Live Prediction
if page == "Live Prediction":
    st.header("Race Weekend Prediction")

    # Load 2026 calendar dynamically
    try:
        schedule = fastf1.get_event_schedule(2026)
        # Filter out testing sessions - only include actual races
        race_events = schedule[
            (schedule['EventFormat'].notna()) &
            (~schedule['EventName'].str.contains('Testing', case=False, na=False))
        ].copy()
        race_names = race_events['EventName'].tolist()

        # Add sprint indicator to race names
        race_options = []
        for _, event in race_events.iterrows():
            race_name = event['EventName']
            event_format = str(event['EventFormat']).lower()
            if 'sprint' in event_format:
                race_options.append(f"{race_name} 🏃 (Sprint)")
            else:
                race_options.append(race_name)
    except Exception as e:
        st.error(f"Failed to load 2026 calendar: {e}")
        race_options = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
                       "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix"]

    col1, col2 = st.columns(2)

    with col1:
        race_selection = st.selectbox("Select Grand Prix", race_options)
        # Remove sprint indicator for processing
        race_name = race_selection.replace(" 🏃 (Sprint)", "")

    with col2:
        weather = st.selectbox("Weather Forecast", ["dry", "rain", "mixed"])

    if st.button("Generate Prediction", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                from src.predictors.baseline_2026 import Baseline2026Predictor
                from src.utils.weekend import is_sprint_weekend

                # Check if sprint weekend
                try:
                    is_sprint = is_sprint_weekend(2026, race_name)
                except:
                    is_sprint = False

                # For 2026 before real race data: use baseline predictor
                st.warning("⚠️ 2026 regulation reset - predictions highly uncertain until testing data available")

                if is_sprint:
                    st.info("🏃 **Sprint Weekend** - Currently predicting Qualifying → Sunday Race only. Sprint race predictions not yet implemented.")

                predictor = Baseline2026Predictor()

                # STEP 1: Predict Qualifying
                st.info("Step 1/2: Predicting qualifying...")
                quali_result = predictor.predict_qualifying(year=2026, race_name=race_name)

                # STEP 2: Predict Race using quali results
                st.info("Step 2/2: Simulating race...")
                race_result = predictor.predict_race(
                    qualifying_grid=quali_result['grid'],
                    weather=weather,
                    race_name=race_name  # Pass track name for overtaking difficulty
                )

                # Display results
                st.success("Predictions complete!")

                # ========== QUALIFYING PREDICTION ==========
                st.markdown("---")
                st.header("🏁 Qualifying Prediction")

                df_quali = pd.DataFrame(quali_result['grid'])
                df_quali['position'] = df_quali['position'].astype(int)
                df_quali['confidence'] = df_quali.get('confidence', 70.0).round(1) if 'confidence' in df_quali else 70.0

                df_quali_display = df_quali[['position', 'driver', 'team']].copy()
                df_quali_display.columns = ['Grid', 'Driver', 'Team']

                # Show quali grid in 3 columns: 1-10, 11-15, 16-22
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**P1-10**")
                    st.dataframe(df_quali_display.head(10), width='stretch', hide_index=True)

                with col2:
                    st.markdown("**P11-15**")
                    st.dataframe(df_quali_display.iloc[10:15], width='stretch', hide_index=True)

                with col3:
                    st.markdown("**P16-22**")
                    st.dataframe(df_quali_display.iloc[15:], width='stretch', hide_index=True)

                # ========== RACE PREDICTION ==========
                st.markdown("---")
                st.header("🏎️ Race Prediction")

                df = pd.DataFrame(race_result['finish_order'])
                df['position'] = df['position'].astype(int)
                df['confidence'] = df['confidence'].round(1)
                df['podium_probability'] = df['podium_probability'].round(1)
                df['dnf_probability'] = (df['dnf_probability'] * 100).round(1)

                # Add DNF risk indicator
                df['dnf_risk'] = df['dnf_probability'].apply(
                    lambda x: '⚠️ High' if x > 20 else '⚡ Medium' if x > 10 else '✓ Low'
                )

                # Format for display
                df_display = df[['position', 'driver', 'team', 'confidence', 'podium_probability', 'dnf_probability', 'dnf_risk']].copy()
                df_display.columns = ['Pos', 'Driver', 'Team', 'Confidence %', 'Podium %', 'DNF Risk %', 'Status']

                # Style the dataframe
                def color_position(val):
                    if val <= 3:
                        colors = {1: '#FFD700', 2: '#C0C0C0', 3: '#CD7F32'}
                        return f'background-color: {colors[val]}; font-weight: bold; color: black'
                    elif val <= 10:
                        return 'background-color: #e3f2fd; font-weight: bold'
                    return ''

                def color_dnf_risk(val):
                    if val > 20:
                        return 'background-color: #ffcdd2; color: #c62828'  # Red
                    elif val > 10:
                        return 'background-color: #fff9c4; color: #f57f17'  # Yellow
                    return 'background-color: #c8e6c9; color: #2e7d32'  # Green

                styled_df = df_display.style.map(
                    color_position, subset=['Pos']
                ).map(
                    color_dnf_risk, subset=['DNF Risk %']
                ).format({
                    'Confidence %': '{:.1f}',
                    'Podium %': '{:.1f}',
                    'DNF Risk %': '{:.1f}'
                })

                st.dataframe(styled_df, width='stretch', height=700)

                # DNF warnings
                high_dnf = df[df['dnf_probability'] > 20]
                if not high_dnf.empty:
                    st.warning(f"⚠️ High DNF risk ({len(high_dnf)} drivers): {', '.join(high_dnf['driver'].values)}")

                # Podium visualization - more prominent
                st.subheader("🏆 Predicted Podium")
                podium = df[df['position'] <= 3].copy()

                col1, col2, col3 = st.columns(3)

                for i, (idx, row) in enumerate(podium.iterrows()):
                    col = [col2, col1, col3][i]  # P1 center, P2 left, P3 right
                    with col:
                        medal = ['🥇', '🥈', '🥉'][row['position']-1]
                        st.markdown(f"### {medal} P{row['position']}")
                        st.markdown(f"## **{row['driver']}**")
                        st.markdown(f"*{row['team']}*")
                        st.metric("Confidence", f"{row['confidence']:.1f}%")
                        st.progress(row['confidence']/100)

                # Split into two columns for better layout
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("📊 Confidence Distribution")

                    # Normalize confidence to 0-100 for better visual spread
                    df_viz = df.copy()
                    conf_min = df_viz['confidence'].min()
                    conf_max = df_viz['confidence'].max()

                    fig_conf = go.Figure()

                    # Add bars with custom colors based on position
                    colors = ['#2ecc71' if i < 3 else '#3498db' if i < 10 else '#95a5a6' for i in range(len(df_viz))]

                    fig_conf.add_trace(go.Bar(
                        x=df_viz['driver'],
                        y=df_viz['confidence'],
                        marker_color=colors,
                        text=df_viz['confidence'].round(1),
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
                    ))

                    fig_conf.update_layout(
                        height=500,
                        yaxis_title="Confidence (%)",
                        xaxis_title="Driver",
                        yaxis_range=[max(0, conf_min-5), min(100, conf_max+5)],
                        showlegend=False,
                        xaxis={'tickangle': -45}
                    )

                    st.plotly_chart(fig_conf, width='stretch')
                    st.caption(f"Range: {conf_min:.1f}% - {conf_max:.1f}% | Avg: {df_viz['confidence'].mean():.1f}%")

                with col_right:
                    st.subheader("🎯 Key Stats")

                    st.info("⚠️ **2026 Reg Reset**: High uncertainty. Model learns from testing (Feb 2026) and races.")

                    # Winner prediction
                    winner = df.iloc[0]
                    st.metric("Predicted Winner", winner['driver'], f"{winner['confidence']:.1f}% confidence")

                    # Podium certainty
                    podium_avg = df[df['position'] <= 3]['confidence'].mean()
                    st.metric("Podium Certainty", f"{podium_avg:.1f}%", "Average top 3")

                    # Average confidence
                    avg_confidence = df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%", "Low until testing")

                    # DNF predictions
                    high_dnf_count = len(df[df['dnf_probability'] > 20])
                    st.metric("High DNF Risk", f"{high_dnf_count}", "⚠️ >20% chance")

                    # Highest uncertainty
                    most_uncertain = df.loc[df['confidence'].idxmin()]
                    st.metric("Most Uncertain", most_uncertain['driver'],
                             f"P{most_uncertain['position']} - {most_uncertain['confidence']:.1f}%")

                    st.markdown("---")
                    st.markdown("**Color Legend:**")
                    st.markdown("🟢 Podium | 🔵 Points | ⚪ No points")
                    st.markdown("**DNF Risk:**")
                    st.markdown("🟢 Low (<10%) | 🟡 Medium (10-20%) | 🔴 High (>20%)")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure data files are generated. Run: `python -m scripts.extract_driver_characteristics 2025`")


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

    st.info("This is a passion project showcasing data science and domain expertise. Not affiliated with FIA or Formula 1.")


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">Built with ❤️ for F1 fans and data nerds</div>',
    unsafe_allow_html=True
)
