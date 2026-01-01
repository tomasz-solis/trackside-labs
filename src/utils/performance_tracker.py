"""
Performance Tracker for F1 Predictions

Logs predictions vs actuals, calculates metrics, learns over time.
No hardcoded MAEs - everything is data-driven.

Usage:
    tracker = PerformanceTracker()
    
    # Log prediction
    tracker.log_qualifying_prediction(
        year=2025,
        race='Bahrain Grand Prix',
        method='blend_70_30',
        predicted_grid=[...],
        actual_grid=[...]
    )
    
    # Get current performance
    mae = tracker.get_expected_mae('qualifying', 'blend_70_30')
    
    # Compare methods
    best = tracker.get_best_method('qualifying')
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, asdict


@dataclass
class PredictionLog:
    """Single prediction record."""
    year: int
    race: str
    method: str
    timestamp: str
    mae: float
    podium_accuracy: float
    top10_accuracy: float
    metadata: Dict


class PerformanceTracker:
    """
    Track prediction performance over time.
    
    Stores actual performance metrics, no hardcoded values.
    Updates automatically as new data comes in.
    """
    
    def __init__(self, data_dir: str = 'data/performance_metrics'):
        """
        Initialize tracker.
        
        Args:
            data_dir: Where to store performance data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.quali_file = self.data_dir / 'qualifying_performance.json'
        self.race_file = self.data_dir / 'race_performance.json'
        self.config_file = self.data_dir / 'model_config.json'
        
        # Load existing data
        self.quali_data = self._load_json(self.quali_file, default={
            'methods': {},
            'logs': [],
            'last_updated': None
        })
        
        self.race_data = self._load_json(self.race_file, default={
            'methods': {},
            'logs': [],
            'last_updated': None
        })
        
        self.config = self._load_json(self.config_file, default=self._default_config())
    
    def log_qualifying_prediction(
        self,
        year: int,
        race: str,
        method: str,
        predicted_grid: List[Dict],
        actual_grid: List[Dict],
        metadata: Optional[Dict] = None
    ):
        """
        Log qualifying prediction and calculate metrics.
        
        Args:
            year: Season year
            race: Race name
            method: Prediction method (e.g., 'blend_70_30', 'session_order_FP3')
            predicted_grid: Predicted positions [{'driver': str, 'position': int}, ...]
            actual_grid: Actual positions [{'driver': str, 'position': int}, ...]
            metadata: Additional info (session used, blend weight, etc.)
        """
        # Calculate metrics
        metrics = self._calculate_metrics(predicted_grid, actual_grid, 'qualifying')
        
        # Create log entry
        log = PredictionLog(
            year=year,
            race=race,
            method=method,
            timestamp=datetime.now().isoformat(),
            mae=metrics['mae'],
            podium_accuracy=metrics['podium_accuracy'],
            top10_accuracy=metrics['top10_accuracy'],
            metadata=metadata or {}
        )
        
        # Store log
        self.quali_data['logs'].append(asdict(log))
        
        # Update method statistics
        if method not in self.quali_data['methods']:
            self.quali_data['methods'][method] = {
                'count': 0,
                'mae_history': [],
                'podium_accuracy_history': [],
                'top10_accuracy_history': [],
                'best_mae': float('inf'),
                'worst_mae': 0.0
            }
        
        method_stats = self.quali_data['methods'][method]
        method_stats['count'] += 1
        method_stats['mae_history'].append(metrics['mae'])
        method_stats['podium_accuracy_history'].append(metrics['podium_accuracy'])
        method_stats['top10_accuracy_history'].append(metrics['top10_accuracy'])
        method_stats['best_mae'] = min(method_stats['best_mae'], metrics['mae'])
        method_stats['worst_mae'] = max(method_stats['worst_mae'], metrics['mae'])
        
        # Update timestamp
        self.quali_data['last_updated'] = datetime.now().isoformat()
        
        # Save
        self._save_json(self.quali_file, self.quali_data)
        
        return metrics
    
    def log_race_prediction(
        self,
        year: int,
        race: str,
        method: str,
        predicted_finish: List[Dict],
        actual_finish: List[Dict],
        metadata: Optional[Dict] = None
    ):
        """
        Log race prediction and calculate metrics.
        
        Args:
            year: Season year
            race: Race name
            method: Prediction method (e.g., 'racecraft_with_tire_hierarchy')
            predicted_finish: Predicted finish order
            actual_finish: Actual finish order
            metadata: Additional info (quali used, FP2 data quality, etc.)
        """
        # Calculate metrics
        metrics = self._calculate_metrics(predicted_finish, actual_finish, 'race')
        
        # Create log entry
        log = PredictionLog(
            year=year,
            race=race,
            method=method,
            timestamp=datetime.now().isoformat(),
            mae=metrics['mae'],
            podium_accuracy=metrics['podium_accuracy'],
            top10_accuracy=metrics['top10_accuracy'],
            metadata=metadata or {}
        )
        
        # Store log
        self.race_data['logs'].append(asdict(log))
        
        # Update method statistics
        if method not in self.race_data['methods']:
            self.race_data['methods'][method] = {
                'count': 0,
                'mae_history': [],
                'podium_accuracy_history': [],
                'top10_accuracy_history': [],
                'best_mae': float('inf'),
                'worst_mae': 0.0
            }
        
        method_stats = self.race_data['methods'][method]
        method_stats['count'] += 1
        method_stats['mae_history'].append(metrics['mae'])
        method_stats['podium_accuracy_history'].append(metrics['podium_accuracy'])
        method_stats['top10_accuracy_history'].append(metrics['top10_accuracy'])
        method_stats['best_mae'] = min(method_stats['best_mae'], metrics['mae'])
        method_stats['worst_mae'] = max(method_stats['worst_mae'], metrics['mae'])
        
        # Update timestamp
        self.race_data['last_updated'] = datetime.now().isoformat()
        
        # Save
        self._save_json(self.race_file, self.race_data)
        
        return metrics
    
    def get_expected_mae(
        self,
        prediction_type: Literal['qualifying', 'race'],
        method: str,
        window: Literal['last_5', 'last_10', 'season', 'all_time'] = 'last_10'
    ) -> float:
        """
        Get expected MAE for a prediction method based on actual performance.
        
        Args:
            prediction_type: 'qualifying' or 'race'
            method: Method identifier
            window: Time window for calculation
            
        Returns:
            Expected MAE (or default if insufficient data)
        """
        data = self.quali_data if prediction_type == 'qualifying' else self.race_data
        
        if method not in data['methods']:
            # No data - return default
            return self._get_default_mae(prediction_type, method)
        
        method_stats = data['methods'][method]
        mae_history = method_stats['mae_history']
        
        if not mae_history:
            return self._get_default_mae(prediction_type, method)
        
        # Calculate based on window
        if window == 'last_5':
            values = mae_history[-5:]
        elif window == 'last_10':
            values = mae_history[-10:]
        elif window == 'season':
            # Get current season logs only
            current_year = datetime.now().year
            values = [
                log['mae'] for log in data['logs']
                if log['method'] == method and log['year'] == current_year
            ]
        else:  # all_time
            values = mae_history
        
        if not values:
            return self._get_default_mae(prediction_type, method)
        
        # Return mean (could also use median for robustness)
        return float(np.mean(values))
    
    def get_method_performance(
        self,
        prediction_type: Literal['qualifying', 'race'],
        method: str
    ) -> Dict:
        """
        Get complete performance summary for a method.
        
        Returns:
            {
                'count': int,
                'mae': {'mean': float, 'std': float, 'best': float, 'worst': float},
                'podium_accuracy': {'mean': float, 'std': float},
                'top10_accuracy': {'mean': float, 'std': float},
                'trend': str  # 'improving', 'stable', 'degrading'
            }
        """
        data = self.quali_data if prediction_type == 'qualifying' else self.race_data
        
        if method not in data['methods']:
            return None
        
        stats = data['methods'][method]
        
        return {
            'count': stats['count'],
            'mae': {
                'mean': float(np.mean(stats['mae_history'])),
                'std': float(np.std(stats['mae_history'])),
                'best': stats['best_mae'],
                'worst': stats['worst_mae']
            },
            'podium_accuracy': {
                'mean': float(np.mean(stats['podium_accuracy_history'])),
                'std': float(np.std(stats['podium_accuracy_history']))
            },
            'top10_accuracy': {
                'mean': float(np.mean(stats['top10_accuracy_history'])),
                'std': float(np.std(stats['top10_accuracy_history']))
            },
            'trend': self._calculate_trend(stats['mae_history'])
        }
    
    def get_best_method(
        self,
        prediction_type: Literal['qualifying', 'race'],
        min_samples: int = 3
    ) -> str:
        """
        Get best performing method based on MAE.
        
        Args:
            prediction_type: 'qualifying' or 'race'
            min_samples: Minimum predictions required
            
        Returns:
            Method name with lowest MAE
        """
        data = self.quali_data if prediction_type == 'qualifying' else self.race_data
        
        best_method = None
        best_mae = float('inf')
        
        for method, stats in data['methods'].items():
            if stats['count'] < min_samples:
                continue
            
            mean_mae = np.mean(stats['mae_history'])
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_method = method
        
        return best_method
    
    def compare_methods(
        self,
        prediction_type: Literal['qualifying', 'race'],
        methods: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare multiple methods side by side.
        
        Args:
            prediction_type: 'qualifying' or 'race'
            methods: List of methods to compare (or all if None)
            
        Returns:
            {method: performance_summary, ...}
        """
        data = self.quali_data if prediction_type == 'qualifying' else self.race_data
        
        if methods is None:
            methods = list(data['methods'].keys())
        
        comparison = {}
        for method in methods:
            comparison[method] = self.get_method_performance(prediction_type, method)
        
        return comparison
    
    def update_config(self, updates: Dict):
        """
        Update tunable parameters.
        
        Args:
            updates: Dict of parameter updates
        """
        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    recursive_update(d[k], v)
                else:
                    d[k] = v
        
        recursive_update(self.config, updates)
        self.config['last_updated'] = datetime.now().isoformat()
        self._save_json(self.config_file, self.config)
    
    def get_config(self, section: Optional[str] = None) -> Dict:
        """
        Get current configuration.
        
        Args:
            section: Specific section ('race_weights', 'uncertainty', etc.)
            
        Returns:
            Config dict or section
        """
        if section:
            return self.config.get(section, {})
        return self.config
    
    def reset_for_new_season(self, year: int):
        """
        Mark start of new season (for regulation changes).
        
        Keeps historical data but flags it as pre-regulation change.
        """
        # Add metadata to all logs
        for log in self.quali_data['logs']:
            if 'regulation_era' not in log['metadata']:
                log['metadata']['regulation_era'] = f"pre_{year}"
        
        for log in self.race_data['logs']:
            if 'regulation_era' not in log['metadata']:
                log['metadata']['regulation_era'] = f"pre_{year}"
        
        # Save
        self._save_json(self.quali_file, self.quali_data)
        self._save_json(self.race_file, self.race_data)
        
        print(f"✅ Marked existing data as pre-{year} regulation era")
    
    def _calculate_metrics(
        self,
        predicted: List[Dict],
        actual: List[Dict],
        prediction_type: str
    ) -> Dict:
        """Calculate prediction metrics."""
        # Build lookup dicts
        pred_dict = {p['driver']: p['position'] for p in predicted}
        actual_dict = {a['driver']: a['position'] for a in actual}
        
        # Calculate MAE
        errors = []
        for driver in pred_dict:
            if driver in actual_dict:
                error = abs(pred_dict[driver] - actual_dict[driver])
                errors.append(error)
        
        mae = float(np.mean(errors)) if errors else 0.0
        
        # Podium accuracy (top 3)
        pred_podium = set([p['driver'] for p in predicted if p['position'] <= 3])
        actual_podium = set([a['driver'] for a in actual if a['position'] <= 3])
        podium_matches = len(pred_podium & actual_podium)
        podium_accuracy = (podium_matches / 3.0) * 100.0
        
        # Top 10 accuracy (or top 8 for sprint quali)
        cutoff = 8 if prediction_type == 'qualifying' else 10
        pred_top = set([p['driver'] for p in predicted if p['position'] <= cutoff])
        actual_top = set([a['driver'] for a in actual if a['position'] <= cutoff])
        top_matches = len(pred_top & actual_top)
        top_accuracy = (top_matches / cutoff) * 100.0
        
        return {
            'mae': mae,
            'podium_accuracy': podium_accuracy,
            'top10_accuracy': top_accuracy,
            'sample_size': len(errors)
        }
    
    def _calculate_trend(self, values: List[float], window: int = 5) -> str:
        """Calculate if metric is improving, stable, or degrading."""
        if len(values) < window * 2:
            return 'insufficient_data'
        
        recent = values[-window:]
        older = values[-window*2:-window]
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        # Lower MAE = better
        improvement = older_mean - recent_mean
        
        if improvement > 0.3:
            return 'improving'
        elif improvement < -0.3:
            return 'degrading'
        else:
            return 'stable'
    
    def _get_default_mae(self, prediction_type: str, method: str) -> float:
        """
        Get sensible default MAE when no data available.
        
        These are conservative estimates, not hardcoded targets.
        """
        defaults = self.config.get('default_mae', {})
        
        # Try to get specific default
        if prediction_type in defaults and method in defaults[prediction_type]:
            return defaults[prediction_type][method]
        
        # Fallback to conservative estimate
        if prediction_type == 'qualifying':
            if 'session_order' in method:
                return 4.0  # Conservative for session order
            elif 'blend' in method:
                return 3.8  # Conservative for blending
            else:
                return 4.5  # Conservative for model-only
        else:  # race
            return 5.5  # Conservative for race (more chaotic)
    
    def _default_config(self) -> Dict:
        """Default configuration with tunable parameters."""
        return {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            
            # Race predictor weights (learned from data)
            'race_weights': {
                'pace_weight': 0.4,
                'grid_weight': 0.3,
                'overtaking_weight': 0.2,
                'tire_deg_weight': 0.1,
                'note': 'Update these based on actual performance'
            },
            
            # Uncertainty parameters
            'uncertainty': {
                'base': 2.5,
                'reliability': 0.15,
                'new_regs_multiplier': 1.3
            },
            
            # Tire predictor parameters
            'tire': {
                'skill_reduction_factor': 0.2,  # Max 20% deg reduction from skill
                'tire_racecraft_penalty': 0.4,  # Max 40% racecraft loss from tires
                'track_effect_range': 0.1      # ±10% from track characteristics
            },
            
            # Conservative defaults (only used when no data)
            'default_mae': {
                'qualifying': {
                    'session_order_FP3': 4.0,
                    'session_order_Sprint_Qualifying': 3.5,
                    'blend_70_30': 3.8,
                    'blend_50_50': 3.9,
                    'model_only': 4.5
                },
                'race': {
                    'racecraft_with_tire_hierarchy': 5.5
                }
            }
        }
    
    def _load_json(self, path: Path, default: Dict) -> Dict:
        """Load JSON file or return default."""
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default
    
    def _save_json(self, path: Path, data: Dict):
        """Save JSON file with formatting."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Convenience functions
def get_tracker(data_dir: str = 'data/performance_metrics') -> PerformanceTracker:
    """Get or create performance tracker instance."""
    return PerformanceTracker(data_dir)
