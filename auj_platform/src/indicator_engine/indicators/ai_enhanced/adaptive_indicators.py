"""
Adaptive Indicators - Dynamic Parameter Optimization Engine
=========================================================

Advanced adaptive indicator system with machine learning parameter optimization.
Automatically adjusts indicator parameters based on market conditions and performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import scipy.optimize as optimize
from collections import deque

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class AdaptiveIndicators(StandardIndicatorInterface):
    """
    AI-powered adaptive indicator system that optimizes parameters in real-time.
    
    Features:
    - Dynamic parameter optimization using genetic algorithms
    - Market regime detection for parameter adaptation
    - Performance-based indicator selection
    - Multi-objective optimization (returns vs risk)
    - Ensemble learning for robust parameter estimation
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'base_indicators': ['rsi', 'macd', 'bollinger', 'stochastic'],
            'adaptation_period': 100,  # Periods to re-optimize
            'lookback_window': 500,   # Historical data for optimization
            'optimization_method': 'genetic',  # 'genetic', 'grid', 'bayesian'
            'performance_threshold': 0.05,  # Minimum improvement to adapt
            'regime_detection_periods': [20, 50, 100],
            'ensemble_size': 5,  # Number of parameter sets in ensemble
            'volatility_adjustment': True,
            'trend_adjustment': True,
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("AdaptiveIndicators", default_params)
        
        # Parameter history and performance tracking
        self.parameter_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.current_regime = 'neutral'
        self.regime_detector = RandomForestRegressor(n_estimators=100, random_state=42)
        self.parameter_ensemble = []
        self.last_optimization = 0
        
        # Initialize base parameter ranges
        self._initialize_parameter_ranges()
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=self.parameters['lookback_window'],
            lookback_periods=1000
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate adaptive indicators with optimized parameters.
        """
        try:
            # Detect current market regime
            current_regime = self._detect_market_regime(data)
            
            # Check if parameter optimization is needed
            if self._should_optimize_parameters(data):
                optimized_params = self._optimize_parameters(data)
                self._update_parameter_ensemble(optimized_params)
            
            # Calculate indicators with current best parameters
            indicator_values = {}
            ensemble_predictions = []
            
            for param_set in self.parameter_ensemble:
                values = self._calculate_indicators_with_params(data, param_set)
                indicator_values.update(values)
                
                # Generate ensemble prediction
                prediction = self._generate_prediction(values, data)
                ensemble_predictions.append(prediction)
            
            # Combine ensemble predictions
            final_prediction = self._combine_ensemble_predictions(ensemble_predictions)
            
            # Calculate adaptive signal strength
            signal_strength = self._calculate_adaptive_signal(indicator_values, current_regime)
            
            # Performance metrics
            performance_metrics = self._calculate_performance_metrics(data)
            
            return {
                'regime': current_regime,
                'indicator_values': indicator_values,
                'ensemble_prediction': final_prediction,
                'signal_strength': signal_strength,
                'adaptation_confidence': self._calculate_adaptation_confidence(),
                'parameter_stability': self._calculate_parameter_stability(),
                'performance_metrics': performance_metrics,
                'optimal_parameters': self.parameter_ensemble[0] if self.parameter_ensemble else {},
                'regime_probability': self._get_regime_probabilities(data)
            }
            
        except Exception as e:
            raise Exception(f"AdaptiveIndicators calculation failed: {str(e)}")
    
    def _initialize_parameter_ranges(self):
        """Initialize parameter ranges for different indicators."""
        self.parameter_ranges = {
            'rsi': {'period': (8, 30), 'overbought': (65, 85), 'oversold': (15, 35)},
            'macd': {'fast': (8, 16), 'slow': (20, 30), 'signal': (6, 12)},
            'bollinger': {'period': (15, 25), 'std_dev': (1.5, 2.5)},
            'stochastic': {'k_period': (10, 20), 'd_period': (3, 7)}
        }
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime using advanced analysis."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate regime features
            features = self._extract_regime_features(data)
            
            if len(features) < 50:
                return 'neutral'
            
            # Train regime detector if not trained
            if not hasattr(self.regime_detector, 'feature_importances_'):
                self._train_regime_detector(features, close)
            
            # Predict current regime
            latest_features = features[-1:].reshape(1, -1)
            regime_score = self.regime_detector.predict(latest_features)[0]
            
            # Classify regime
            if regime_score > 0.1:
                return 'trending_up'
            elif regime_score < -0.1:
                return 'trending_down'
            else:
                volatility = np.std(np.diff(close[-20:])) / np.mean(close[-20:])
                if volatility > 0.02:
                    return 'volatile'
                else:
                    return 'sideways'
                    
        except Exception:
            return 'neutral'
    
    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection."""
        close = data['close'].values
        volume = data['volume'].values
        high = data['high'].values
        low = data['low'].values
        
        features = []
        window = 20
        
        for i in range(window, len(close)):
            window_close = close[i-window:i]
            window_volume = volume[i-window:i]
            window_high = high[i-window:i]
            window_low = low[i-window:i]
            
            # Price features
            returns = np.diff(window_close) / window_close[:-1]
            volatility = np.std(returns)
            trend = np.polyfit(range(len(window_close)), window_close, 1)[0]
            
            # Volume features
            volume_trend = np.polyfit(range(len(window_volume)), window_volume, 1)[0]
            volume_volatility = np.std(window_volume)
            
            # Range features
            true_range = np.maximum(
                window_high[1:] - window_low[1:],
                np.maximum(
                    abs(window_high[1:] - window_close[:-1]),
                    abs(window_low[1:] - window_close[:-1])
                )
            )
            atr = np.mean(true_range)
            
            feature_vector = [
                trend / (np.mean(window_close) + 1e-8),
                volatility,
                volume_trend / (np.mean(window_volume) + 1e-8),
                volume_volatility / (np.mean(window_volume) + 1e-8),
                atr / np.mean(window_close),
                np.mean(returns),
                np.skew(returns) if len(returns) > 2 else 0,
                np.kurtosis(returns) if len(returns) > 2 else 0
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _train_regime_detector(self, features: np.ndarray, close: np.ndarray):
        """Train the regime detection model."""
        if len(features) < 50:
            return
        
        # Create target variable (future returns)
        targets = []
        for i in range(len(features) - 10):
            future_return = (close[i + 30] - close[i + 20]) / close[i + 20] if i + 30 < len(close) else 0
            targets.append(future_return)
        
        if len(targets) > 0:
            features_train = features[:len(targets)]
            self.regime_detector.fit(features_train, targets)
    
    def _should_optimize_parameters(self, data: pd.DataFrame) -> bool:
        """Determine if parameter optimization is needed."""
        # Check if enough time has passed
        if len(data) - self.last_optimization < self.parameters['adaptation_period']:
            return False
        
        # Check if performance has degraded
        if len(self.performance_history) > 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            historical_performance = np.mean(list(self.performance_history)[:-10])
            
            if recent_performance < historical_performance - self.parameters['performance_threshold']:
                return True
        
        return len(self.parameter_ensemble) == 0
    
    def _optimize_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize indicator parameters using selected method."""
        method = self.parameters['optimization_method']
        
        if method == 'genetic':
            return self._genetic_optimization(data)
        elif method == 'grid':
            return self._grid_search_optimization(data)
        elif method == 'bayesian':
            return self._bayesian_optimization(data)
        else:
            return self._default_parameters()
    
    def _genetic_optimization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Genetic algorithm parameter optimization."""
        try:
            def objective_function(params_flat):
                params = self._unflatten_parameters(params_flat)
                performance = self._evaluate_parameters(params, data)
                return -performance  # Minimize negative performance
            
            # Define bounds for all parameters
            bounds = self._get_parameter_bounds()
            
            # Run optimization
            result = optimize.differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42
            )
            
            optimized_params = self._unflatten_parameters(result.x)
            self.last_optimization = len(data)
            
            return optimized_params
            
        except Exception:
            return self._default_parameters()
    
    def _get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for indicator in self.parameters['base_indicators']:
            for param, (min_val, max_val) in self.parameter_ranges[indicator].items():
                bounds.append((min_val, max_val))
        return bounds
    
    def _unflatten_parameters(self, params_flat: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Convert flattened parameters back to structured format."""
        params = {}
        idx = 0
        
        for indicator in self.parameters['base_indicators']:
            params[indicator] = {}
            for param_name in self.parameter_ranges[indicator]:
                params[indicator][param_name] = params_flat[idx]
                idx += 1
        
        return params
    
    def _evaluate_parameters(self, params: Dict[str, Dict[str, float]], data: pd.DataFrame) -> float:
        """Evaluate parameter set performance."""
        try:
            # Calculate indicators with given parameters
            indicator_values = self._calculate_indicators_with_params(data, params)
            
            # Generate signals
            signals = self._generate_signals_from_indicators(indicator_values, data)
            
            # Calculate performance metrics
            returns = self._calculate_strategy_returns(signals, data)
            
            # Multi-objective evaluation
            total_return = np.sum(returns)
            volatility = np.std(returns)
            sharpe_ratio = total_return / (volatility + 1e-8)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Combined performance score
            performance = sharpe_ratio - max_drawdown * 0.5
            
            return performance
            
        except Exception:
            return -1.0
    
    def _calculate_indicators_with_params(self, data: pd.DataFrame, 
                                        params: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate indicators with specific parameters."""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        indicators = {}
        
        try:
            # RSI
            if 'rsi' in params:
                rsi_period = int(params['rsi']['period'])
                rsi = self._calculate_rsi(close, rsi_period)
                indicators['rsi'] = rsi[-1] if len(rsi) > 0 else 50
                indicators['rsi_signal'] = self._rsi_signal(rsi[-1], params['rsi'])
            
            # MACD
            if 'macd' in params:
                macd_line, signal_line = self._calculate_macd(
                    close, 
                    int(params['macd']['fast']), 
                    int(params['macd']['slow']),
                    int(params['macd']['signal'])
                )
                indicators['macd'] = macd_line[-1] - signal_line[-1] if len(macd_line) > 0 else 0
                indicators['macd_signal'] = 1 if indicators['macd'] > 0 else -1
            
            # Bollinger Bands
            if 'bollinger' in params:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                    close,
                    int(params['bollinger']['period']),
                    params['bollinger']['std_dev']
                )
                indicators['bollinger_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                indicators['bollinger_signal'] = self._bollinger_signal(indicators['bollinger_position'])
            
            # Stochastic
            if 'stochastic' in params:
                stoch_k, stoch_d = self._calculate_stochastic(
                    high, low, close,
                    int(params['stochastic']['k_period']),
                    int(params['stochastic']['d_period'])
                )
                indicators['stochastic_k'] = stoch_k[-1] if len(stoch_k) > 0 else 50
                indicators['stochastic_d'] = stoch_d[-1] if len(stoch_d) > 0 else 50
                indicators['stochastic_signal'] = self._stochastic_signal(stoch_k[-1], stoch_d[-1])
            
        except Exception:
            pass
        
        return indicators
    
    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI."""
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean().fillna(0)
        avg_losses = pd.Series(losses).rolling(window=period).mean().fillna(0)
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values
    
    def _calculate_macd(self, close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD."""
        ema_fast = pd.Series(close).ewm(span=fast).mean()
        ema_slow = pd.Series(close).ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line.values, signal_line.values
    
    def _calculate_bollinger_bands(self, close: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        sma = pd.Series(close).rolling(window=period).mean()
        std = pd.Series(close).rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper.values, sma.values, lower.values
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                            k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic oscillator."""
        lowest_low = pd.Series(low).rolling(window=k_period).min()
        highest_high = pd.Series(high).rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        d_percent = pd.Series(k_percent).rolling(window=d_period).mean()
        
        return k_percent, d_percent.values
    
    def _default_parameters(self) -> Dict[str, Any]:
        """Return default parameter set."""
        return {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2.0},
            'stochastic': {'k_period': 14, 'd_period': 3}
        }
    
    def _update_parameter_ensemble(self, new_params: Dict[str, Any]):
        """Update the parameter ensemble with new optimized parameters."""
        self.parameter_ensemble.insert(0, new_params)
        
        # Keep only top performing parameter sets
        if len(self.parameter_ensemble) > self.parameters['ensemble_size']:
            self.parameter_ensemble = self.parameter_ensemble[:self.parameters['ensemble_size']]
    
    def _generate_prediction(self, indicator_values: Dict[str, Any], data: pd.DataFrame) -> float:
        """Generate prediction from indicator values."""
        signals = []
        
        for key, value in indicator_values.items():
            if '_signal' in key:
                signals.append(value)
        
        if signals:
            return np.mean(signals)
        
        return 0.0
    
    def _combine_ensemble_predictions(self, predictions: List[float]) -> float:
        """Combine ensemble predictions."""
        if not predictions:
            return 0.0
        
        # Weighted average based on recent performance
        weights = np.exp(np.linspace(0, 1, len(predictions)))  # More weight to recent
        weights = weights / np.sum(weights)
        
        return float(np.average(predictions, weights=weights))
    
    def _calculate_adaptive_signal(self, indicator_values: Dict[str, Any], regime: str) -> float:
        """Calculate adaptive signal based on current regime."""
        base_signal = 0.0
        signal_count = 0
        
        # Weight signals based on regime
        regime_weights = {
            'trending_up': {'macd_signal': 1.5, 'rsi_signal': 0.8, 'bollinger_signal': 1.0, 'stochastic_signal': 0.9},
            'trending_down': {'macd_signal': 1.5, 'rsi_signal': 0.8, 'bollinger_signal': 1.0, 'stochastic_signal': 0.9},
            'sideways': {'rsi_signal': 1.5, 'bollinger_signal': 1.5, 'stochastic_signal': 1.2, 'macd_signal': 0.7},
            'volatile': {'bollinger_signal': 1.8, 'stochastic_signal': 1.3, 'rsi_signal': 1.1, 'macd_signal': 0.8},
            'neutral': {'rsi_signal': 1.0, 'macd_signal': 1.0, 'bollinger_signal': 1.0, 'stochastic_signal': 1.0}
        }
        
        weights = regime_weights.get(regime, regime_weights['neutral'])
        
        for signal_key, weight in weights.items():
            if signal_key in indicator_values:
                base_signal += indicator_values[signal_key] * weight
                signal_count += weight
        
        if signal_count > 0:
            return base_signal / signal_count
        
        return 0.0
    
    def _calculate_adaptation_confidence(self) -> float:
        """Calculate confidence in current parameter adaptation."""
        if len(self.performance_history) < 10:
            return 0.5
        
        recent_performance = list(self.performance_history)[-10:]
        performance_stability = 1.0 - np.std(recent_performance)
        
        return float(np.clip(performance_stability, 0, 1))
    
    def _calculate_parameter_stability(self) -> float:
        """Calculate stability of parameter values."""
        if len(self.parameter_history) < 5:
            return 0.5
        
        # Analyze parameter change rate
        recent_params = list(self.parameter_history)[-5:]
        param_changes = []
        
        for i in range(1, len(recent_params)):
            change = self._calculate_parameter_distance(recent_params[i-1], recent_params[i])
            param_changes.append(change)
        
        stability = 1.0 - np.mean(param_changes) if param_changes else 0.5
        
        return float(np.clip(stability, 0, 1))
    
    def _calculate_parameter_distance(self, params1: Dict, params2: Dict) -> float:
        """Calculate distance between parameter sets."""
        distance = 0.0
        count = 0
        
        for indicator in params1:
            if indicator in params2:
                for param in params1[indicator]:
                    if param in params2[indicator]:
                        normalized_diff = abs(params1[indicator][param] - params2[indicator][param]) / max(params1[indicator][param], params2[indicator][param], 1)
                        distance += normalized_diff
                        count += 1
        
        return distance / max(count, 1)
    
    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(self.performance_history) == 0:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}
        
        returns = list(self.performance_history)
        
        return {
            'total_return': np.sum(returns),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'avg_return': np.mean(returns),
            'volatility': np.std(returns)
        }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        return float(np.min(drawdown))
    
    def _get_regime_probabilities(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get probabilities for different market regimes."""
        # This is a simplified implementation
        # In a real system, you would use more sophisticated regime detection
        return {
            'trending_up': 0.25,
            'trending_down': 0.25,
            'sideways': 0.25,
            'volatile': 0.25
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on adaptive analysis."""
        signal_strength = value['signal_strength']
        adaptation_confidence = value['adaptation_confidence']
        
        # Adjust confidence based on adaptation confidence
        confidence = min(abs(signal_strength) * adaptation_confidence, 1.0)
        
        if signal_strength > 0.6:
            return SignalType.STRONG_BUY, confidence
        elif signal_strength > 0.2:
            return SignalType.BUY, confidence
        elif signal_strength < -0.6:
            return SignalType.STRONG_SELL, confidence
        elif signal_strength < -0.2:
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, confidence
