# Formula 1 2026 Predictive Engine

## Why 2026?

Major regulation changes:
- New power units (50/50 electric/ICE vs current 80/20)
- Active aero (adjustable front/rear wings)
- 30kg lighter (720kg → 690kg)
- New team (Cadillac)
- New tires

When regs reset, historical performance matters less. Testing matters more. Current validation proves the system handles both scenarios - just needs weight adjustment.

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

### 1. Setup

```bash
git clone [https://github.com/tomasz-solis/formula1-2026.git](https://github.com/tomasz-solis/formula1-2026.git)
cd formula1-2026
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize the Data
Before you can predict anything, you need to build the knowledge base. This extracts the driver skills and track characteristics from the raw data.

```bash
# Calculate overtaking difficulty for every track
python -m scripts.extract_overtaking_likelihood

# Build driver profiles (Pace, Consistency, Tire Mgmt)
python -m scripts.extract_driver_characteristics 2025
```

## 3. The Race Weekend Loop
This is the actual workflow I use during a race weekend.

1. Friday / Saturday (Live Prediction)
Run this command whenever a session finishes. It detects what data is available (FP1, FP2, or Quali) and runs the best possible simulation.

```bash
# Example: It's Saturday afternoon
python predict_weekend.py "Bahrain Grand Prix"
```

If Qualifying is done, it simulates the race using the real grid + tire data. If not, it predicts the grid first.

2. Monday (The Learning Loop)
Run this after the weekend. It fetches the official results, checks how the model performed, and updates the weights for the next race.

```bash
python -m scripts.post_race_analysis "Bahrain Grand Prix"
```

## Simulation & Testing
You can run a full season simulation to test how 2026 regulation changes might play out:

```bash
python simulator.py
```

Or run the Stress Tests to verify the physics engine is actually working (e.g., checking if low-skill drivers actually crash more in wet simulations):

```bash
python src/tests/stress_test_race.py
```

## Project Structure
```src/extractors/``` FastF1 logic for telemetry analysis.

```src/models/``` Physics definitions (Car performance, Tire curves).

```src/predictors/``` The core simulation engines (Race, Quali).

```src/systems/``` The meta-learning logic.

```scripts/``` Offline tools for data extraction and post-race analysis.

```predict_weekend.py``` The main CLI tool.

## License

MIT

## Contact

Tomasz Solis
- Email: tomasz.solis@gmail.com
- [LinkedIn](linkedin.com/in/tomaszsolis)
- [GitHub](github.com/tomasz-solis)
