"""
Learning System

Learns from race results and adjusts strategy after races 1, 3, 5, 8.
Tracks: method performance, overtaking factors, pace adjustments.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class LearningSystem:
    """
    Learn from actual results and adapt strategy.
    
    Learning checkpoints: After races 1, 3, 5, 8
    - Track method performance (blend vs session_order)
    - Track overtaking factors (2026 vs 2025 baseline)
    - Track pace model accuracy
    """
    
    LEARNING_CHECKPOINTS = [1, 3, 5, 8]
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.learning_path = self.data_dir / 'learning_state.json'
        
        # Load or initialize
        if self.learning_path.exists():
            with open(self.learning_path) as f:
                self.state = json.load(f)
        else:
            self.state = self._initialize_state()
    
    def _initialize_state(self) -> Dict:
        """Initialize learning state."""
        return {
            'season': 2026,
            'races_completed': 0,
            'last_checkpoint': 0,
            
            # Method performance tracking
            'method_performance': {
                'blend_50_50': {'maes': [], 'avg': None},
                'blend_70_30': {'maes': [], 'avg': None},
                'blend_90_10': {'maes': [], 'avg': None},
                'session_order': {'maes': [], 'avg': None}
            },
            
            # Recommended defaults (updated at checkpoints)
            'recommended_method': 'blend',
            'recommended_split': '50/50',
            
            # Track-specific overtaking factors (2026 vs 2025)
            'overtaking_factors': {},
            
            # Pace model adjustments
            'pace_model_weights': {
                'pace_weight': 0.4,
                'grid_weight': 0.3,
                'overtaking_weight': 0.2,
                'tire_deg_weight': 0.1
            },
            
            # Insights from checkpoints
            'insights': []
        }
    
    def get_recommended_method(self, weekend_type='normal') -> Dict:
        """Get recommended prediction method based on learning."""
        
        if weekend_type == 'sprint':
            # Sprint Quali order works best (2.99 MAE from testing)
            return {
                'method': 'session_order',
                'session': 'sprint_quali',
                'expected_mae': 2.99,
                'confidence': 'high'
            }
        else:
            # Use learned recommendation
            method = self.state['recommended_method']
            split = self.state['recommended_split']
            
            if method == 'blend':
                # Get expected MAE for this split
                method_key = f"blend_{split.replace('/', '_')}"
                perf = self.state['method_performance'].get(method_key, {})
                expected_mae = perf.get('avg', 3.57)  # Default to 50/50 baseline
            else:
                # Session order
                perf = self.state['method_performance'].get('session_order', {})
                expected_mae = perf.get('avg', 3.73)  # FP3 baseline
            
            return {
                'method': method,
                'split': split if method == 'blend' else None,
                'expected_mae': expected_mae,
                'confidence': 'medium' if self.state['races_completed'] < 3 else 'high'
            }
    
    def update_after_race(
        self,
        race: str,
        actual_results: Dict,
        prediction_comparison: Optional[Dict] = None
    ) -> Dict:
        """
        Update learning system after each race.
        
        Returns insights if at checkpoint (races 1, 3, 5, 8).
        """
        self.state['races_completed'] += 1
        race_num = self.state['races_completed']
        
        insights = {
            'observations': [],
            'recommendations': []
        }
        
        # Record method performance
        if prediction_comparison:
            method_used = prediction_comparison.get('method')
            quali_mae = prediction_comparison['qualifying']['mae']
            
            if method_used and quali_mae:
                method_key = method_used.replace('/', '_')
                if method_key in self.state['method_performance']:
                    self.state['method_performance'][method_key]['maes'].append(quali_mae)
                    
                    # Update average
                    maes = self.state['method_performance'][method_key]['maes']
                    self.state['method_performance'][method_key]['avg'] = np.mean(maes)
        
        # Calculate overtaking factor if we have race data
        if 'race' in actual_results:
            overtaking_factor = self._calculate_overtaking_factor(race, actual_results)
            if overtaking_factor:
                self.state['overtaking_factors'][race] = overtaking_factor
        
        # Check if this is a learning checkpoint
        if race_num in self.LEARNING_CHECKPOINTS:
            insights = self._analyze_at_checkpoint(race_num)
            self.state['last_checkpoint'] = race_num
        
        # Save state
        self._save()
        
        return insights
    
    def _analyze_at_checkpoint(self, race_num: int) -> Dict:
        """Analyze performance at learning checkpoint."""
        insights = {
            'observations': [],
            'recommendations': [],
            'method_performance': {}
        }
        
        # Compare method performance
        best_method = None
        best_mae = float('inf')
        
        for method, perf in self.state['method_performance'].items():
            if perf['avg'] is not None and len(perf['maes']) >= 2:
                insights['method_performance'][method] = {
                    'avg_mae': perf['avg'],
                    'races': len(perf['maes'])
                }
                
                if perf['avg'] < best_mae:
                    best_mae = perf['avg']
                    best_method = method
        
        # Observations
        if best_method:
            insights['observations'].append(
                f"Best performing method: {best_method} (MAE: {best_mae:.2f})"
            )
        
        # Overtaking analysis
        if len(self.state['overtaking_factors']) >= 2:
            avg_factor = np.mean(list(self.state['overtaking_factors'].values()))
            
            if avg_factor > 1.2:
                insights['observations'].append(
                    f"Overtaking {int((avg_factor-1)*100)}% higher than 2025 baseline"
                )
                insights['observations'].append(
                    "Teams still adapting to new regulations"
                )
            elif avg_factor < 0.8:
                insights['observations'].append(
                    f"Overtaking {int((1-avg_factor)*100)}% lower than 2025 baseline"
                )
                insights['observations'].append(
                    "Field may be converging faster than expected"
                )
        
        # Recommendations at key checkpoints
        if race_num == 3:
            # After 3 races: Suggest method changes
            if best_method and best_method != self.state['recommended_method']:
                improvement = self.state['method_performance'][self.state['recommended_method']]['avg'] - best_mae
                
                if improvement > 0.3:  # Significant improvement
                    insights['recommendations'].append(
                        f"Switch to {best_method} (improves MAE by {improvement:.2f})"
                    )
        
        elif race_num == 5:
            # After 5 races: Adjust pace model weights
            insights['recommendations'].append(
                "Review pace model weights based on observed patterns"
            )
        
        elif race_num == 8:
            # After 8 races: Finalize strategy
            insights['recommendations'].append(
                f"Finalize strategy: Use {best_method} for remaining races"
            )
            
            # Auto-switch if clearly better
            if best_method and best_mae < self.state['method_performance'].get(self.state['recommended_method'], {}).get('avg', float('inf')) - 0.3:
                self.state['recommended_method'] = 'blend' if 'blend' in best_method else 'session_order'
                if 'blend' in best_method:
                    self.state['recommended_split'] = best_method.split('_')[1] + '/' + best_method.split('_')[2]
                
                insights['recommendations'].append(
                    f"✅ Auto-switched to {best_method}"
                )
        
        # Save insights
        self.state['insights'].append({
            'checkpoint': race_num,
            'insights': insights
        })
        
        return insights
    
    def _calculate_overtaking_factor(self, race: str, actual_results: Dict) -> Optional[float]:
        """
        Calculate overtaking factor: 2026 overtakes / 2025 baseline.
        
        Returns factor > 1 if more overtaking, < 1 if less.
        """
        # Load 2025 baseline
        baseline_path = Path('data/historical/overtaking_difficulty.json')
        if not baseline_path.exists():
            return None
        
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        race_slug = race.lower().replace(' ', '_')
        
        if race_slug not in baseline:
            # Find similar track
            return None
        
        baseline_overtakes = baseline[race_slug].get('avg_overtakes', 30)
        
        # Count actual overtakes (simplified - compare grid vs finish)
        grid_order = {p['driver']: p['position'] for p in actual_results['qualifying']}
        finish_order = {p['driver']: p['position'] for p in actual_results['race']}
        
        position_changes = []
        for driver in grid_order:
            if driver in finish_order:
                change = abs(grid_order[driver] - finish_order[driver])
                position_changes.append(change)
        
        if not position_changes:
            return None
        
        # Rough estimate: sum of position changes / 2 (since each overtake involves 2 drivers)
        estimated_overtakes = sum(position_changes) / 2
        
        factor = estimated_overtakes / baseline_overtakes
        
        return factor
    
    def get_race_count(self) -> int:
        """Get number of completed races."""
        return self.state['races_completed']
    
    def get_overtaking_factor(self, track: str) -> float:
        """Get learned overtaking factor for track, or default to 1.0."""
        return self.state['overtaking_factors'].get(track, 1.0)
    
    def get_pace_model_weights(self) -> Dict:
        """Get current pace model weights."""
        return self.state['pace_model_weights'].copy()
    
    def _save(self):
        """Save learning state."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.learning_path, 'w') as f:
            json.dump(self.state, f, indent=2)
