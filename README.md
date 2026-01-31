# Formula 1 2026 Predictive Engine

## Why 2026?

Major regulation changes coming:
- New power units (50/50 electric/ICE vs current 80/20)
- Active aero (adjustable front/rear wings)
- 30kg lighter (720kg → 690kg)
- New team: Cadillac joins (11 teams, 22 drivers)
- New Pirelli tire compounds

**Timeline:**
- 2025 season: Complete (NOR wins WDC)
- February 2026: Pre-season testing begins
- March 2026: Season opener in Bahrain

When regulations reset, historical performance matters less. Practice data and testing become critical. This model adapts to both scenarios.

---

## Logic

**A physics-first simulation engine for F1 strategy and results.**

Most F1 models are just black-box machine learning that overreacts to recent results. This project is different. It is built on a specific belief: **Drivers are car-limited, not skill-limited.**

This system simulates the physics of a race weekend—tire degradation, pit loss, weather chaos, and lap 1 variance—layered with a Bayesian model that separates the car's ceiling from the driver's ability to reach it.

It also self-corrects. After every race, it looks back, calculates where it was wrong, and adjusts how much it trusts "Practice Data" vs. "Historical Baselines" for the next round.

---

## How it Works

The system is split into three logical parts:

1. **Factory** (Extraction)
   Pulls raw telemetry to understand the physics. It doesn't just look at finishing positions; it calculates tire deg slopes, cornering speeds, and % gap to teammates to isolate driver skill.

2. **Engine** (Simulation)
   A Python implementation of a race. It simulates:
   * **Lap 1 Chaos:** Rookies are more likely to lose positions or crash.
   * **Tire Physics:** Uses degradation slopes to calculate a cumulative pace penalty. The model simulates how high-wear characteristics destroy a driver's net race time, regardless of the specific pit strategy chosen.
   * **Weather:** Rain acts as a "skill multiplier," punishing low-consistency drivers.

3. **Brain** (Learning)
   Tracks its own performance. If the model consistently underestimates a team, it re-weights the priors automatically.

---

## Getting Started

### Quick Start

**For 2026 Pre-Season Predictions (No Race Data Yet):**

```bash
# 1. Clone and setup
git clone https://github.com/tomasz-solis/formula1-2026.git
cd formula1-2026
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Extract 2025 driver characteristics (baseline for 2026)
python -m scripts.extract_driver_characteristics 2025

# 3. Run the dashboard
streamlit run app.py
```

The dashboard loads all 24 2026 races dynamically and provides Monte Carlo predictions based on 2025 team standings and driver skill.

### CLI Tool (For Live Race Weekends)

Once 2026 races start, use the CLI for session-by-session predictions:

```bash
# Predict from latest available data (FP1, FP2, FP3, or Qualifying)
python predict_weekend.py "Bahrain Grand Prix"
```

### Post-Race Learning

After each race, update the model with actual results:

```bash
python -m scripts.post_race_analysis "Bahrain Grand Prix"
```

## Two Prediction Modes

This project uses different predictors depending on data availability:

**1. Baseline Predictor (2026 Pre-Season)**
- Used when no 2026 race data exists
- Based on 2025 team standings + driver skill ratings
- Lower confidence (40-60%) acknowledging regulation uncertainty
- Monte Carlo simulation (50 runs) for stability

**2. Bayesian Predictor (After First Races)**
- Activates after 2026 testing/races begin
- Uses practice session telemetry + historical rankings
- Higher confidence as more data accumulates
- Adaptive learning from prediction errors

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                          │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │  app.py          │         │  predict_weekend.py     │  │
│  │  (Streamlit)     │         │  (CLI)                  │  │
│  └────────┬─────────┘         └──────────┬──────────────┘  │
└───────────┼────────────────────────────────┼─────────────────┘
            │                                │
            ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION LAYER                           │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │ Baseline2026         │    │ Bayesian Predictors      │  │
│  │ (Pre-season)         │    │ (Post-testing)           │  │
│  │ • Team strength 70%  │    │ • Practice telemetry     │  │
│  │ • Driver skill 30%   │    │ • Historical rankings    │  │
│  │ • Monte Carlo (50x)  │    │ • Adaptive learning      │  │
│  └──────────┬───────────┘    └──────────┬───────────────┘  │
└─────────────┼────────────────────────────┼─────────────────┘
              │                            │
              ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    PHYSICS ENGINE                            │
│  ┌────────────┐  ┌───────────┐  ┌────────────────────────┐ │
│  │ Tire Model │  │ Weather   │  │ DNF Risk               │ │
│  │ • Deg      │  │ • Skill   │  │ • Team reliability     │ │
│  │   slopes   │  │   mult.   │  │ • Driver errors        │ │
│  └────────────┘  └───────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ FastF1      │  │ Static Data  │  │ Learning System  │  │
│  │ • Telemetry │  │ • Teams 2026 │  │ • Performance    │  │
│  │ • Sessions  │  │ • Drivers    │  │   tracking       │  │
│  │ • Results   │  │ • Tracks     │  │ • Weight tuning  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. **Input**: Race selection + weather + session data
2. **Prediction**: Baseline (pre-season) or Bayesian (post-testing) predictor
3. **Simulation**: Physics engine models tire deg, weather, DNF risk
4. **Output**: Qualifying grid + race results with confidence scores
5. **Learning**: Post-race analysis updates model weights

---

## Dashboard Features

- **24 Race Calendar**: Dynamically loaded from FastF1
- **Sprint Detection**: 6 sprint weekends marked with 🏃
- **Monte Carlo Simulation**: 50 simulations per prediction for stability
- **DNF Risk Analysis**: Team reliability + driver error rates
- **Track Characteristics**: Circuit-specific safety car probability and overtaking difficulty

## Project Structure

```
formula1-2026/
├── app.py                    # Streamlit dashboard (main entry point)
├── predict_weekend.py        # CLI tool for live predictions
├── src/
│   ├── extractors/          # FastF1 telemetry extraction
│   ├── models/              # Physics models (tires, weather, DNF)
│   ├── predictors/          # Prediction engines
│   │   ├── baseline_2026.py # Pre-season predictor
│   │   ├── qualifying.py    # Bayesian qualifying predictor
│   │   └── race.py          # Race simulation engine
│   ├── systems/             # Meta-learning and adaptation
│   └── utils/               # Helper functions
├── scripts/                 # Utility scripts
│   └── simulator.py         # Full season simulation
├── tests/                   # pytest test suite
├── notebooks/               # Jupyter analysis notebooks
│   └── validation_report.ipynb  # 2025 backtesting results
├── data/
│   ├── current_lineups.json # 2026 driver lineups
│   └── processed/
│       ├── car_characteristics/2026_car_characteristics.json
│       ├── track_characteristics/2026_track_characteristics.json
│       └── driver_characteristics.json
└── config/
    ├── default.yaml         # Model hyperparameters
    └── production_config.json  # Session selection strategy
```

## License

MIT

## Contact

Tomasz Solis
- [LinkedIn](https://linkedin.com/in/tomaszsolis)
- [GitHub](https://github.com/tomasz-solis)
