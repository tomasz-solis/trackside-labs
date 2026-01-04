"""
Run this AFTER the race weekend is complete.
1. Fetches official results from FastF1.
2. Re-runs predictions with different blend weights (Backtesting).
3. Calculates which strategy was actually best.
4. Updates the LearningSystem so 'predict_weekend.py' is smarter next time.
"""
import sys
import argparse
import fastf1 as ff1
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from src.systems.learning import LearningSystem
    from src.utils.performance_tracker import PerformanceTracker
    from src.predictors.qualifying import QualifyingPredictor
    from src.models.bayesian import BayesianDriverRanking
    from src.models.priors_factory import PriorsFactory
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from src.systems.learning import LearningSystem
    from src.utils.performance_tracker import PerformanceTracker
    from src.predictors.qualifying import QualifyingPredictor
    from src.models.bayesian import BayesianDriverRanking
    from src.models.priors_factory import PriorsFactory

def calculate_mae(predicted_grid, actual_map):
    """
    Calculate Mean Absolute Error between predicted and actual positions.
    """
    errors = []
    for row in predicted_grid:
        driver = row['driver']
        pred_pos = row['position']
        
        # Check if driver finished quali
        if driver in actual_map:
            actual_pos = actual_map[driver]
            errors.append(abs(pred_pos - actual_pos))
            
    if not errors: return 99.0
    return np.mean(errors)

def close_weekend_loop(year, race_name):
    print(f"📊 ANALYZING PERFORMANCE: {race_name} ({year})")
    
    # 1. Initialize Engines
    learner = LearningSystem()
    factory = PriorsFactory()
    base_priors = factory.create_priors()
    
    # We need a predictor to re-run the scenarios
    ranker = BayesianDriverRanking(base_priors)
    predictor = QualifyingPredictor(driver_ranker=ranker)
    
    # 2. Fetch Official Results (Ground Truth)
    print("   📡 Fetching Official Qualifying Results from FastF1...")
    try:
        # Enable cache if not already
        cache_dir = Path('data/raw/.fastf1_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        ff1.Cache.enable_cache(str(cache_dir))
        
        session = ff1.get_session(year, race_name, 'Q')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        
        if not hasattr(session, 'results'):
            print("   ❌ No results found. Did the race happen yet?")
            return

        # Create map: {'VER': 1, 'LEC': 2, ...}
        # Filter out DNFs/NCs if needed, but usually we want to predict them
        valid_results = session.results[['Abbreviation', 'Position']].dropna()
        gt_map = dict(zip(valid_results['Abbreviation'], valid_results['Position']))
        
        print(f"   ✅ Loaded {len(gt_map)} driver results.")
        
    except Exception as e:
        print(f"   ❌ FastF1 Error: {e}")
        return

    # 3. Backtest Strategies
    # We compare 4 standard strategies to see which one WINS for this track conditions
    strategies = {
        'blend_100_0': 1.0,  # Trust FP3 completely
        'blend_70_30': 0.7,  # Standard Blend
        'blend_50_50': 0.5,  # Conservative Blend
        'blend_0_100': 0.0   # Trust Model (Priors) completely
    }
    
    results = {}
    print("\n   🧪 Re-running simulations to find optimal strategy...")
    
    for label, weight in strategies.items():
        try:
            # Run the prediction as if we were there
            pred = predictor.predict(
                year=year,
                race_name=race_name,
                method='blend' if weight > 0 else 'model',
                blend_weight=weight,
                verbose=False
            )
            
            # Calculate how wrong we were
            mae = calculate_mae(pred['grid'], gt_map)
            results[label] = mae
            print(f"      - {label} ({int(weight*100)}% FP): MAE {mae:.2f}")
            
        except Exception as e:
            print(f"      - {label}: Failed ({e})")
            results[label] = 99.0

    # 4. Update the Brain
    # Find the strategy with the LOWEST Error
    best_strategy = min(results, key=results.get)
    best_mae = results[best_strategy]
    
    print(f"\n🏆 WINNER: {best_strategy} (MAE {best_mae:.2f})")
    
    # If the Model beat the Data (blend_0_100 wins), it means FP3 was misleading (Sandbagging/Rain/Red Flags).
    # If Data beat Model (blend_100_0 wins), it means FP3 was super representative.
    
    learner.update_after_race(
        race=race_name,
        actual_results=gt_map, 
        prediction_comparison={'qualifying': {'mae': best_mae, 'method': best_strategy}}
    )
    
    print("✅ Learning System updated. 'predict_weekend.py' will now use this weight.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("race_name", help="e.g. 'Bahrain Grand Prix'")
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()
    
    close_weekend_loop(args.year, args.race_name)