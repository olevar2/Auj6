"""
MACD Indicator - Advanced Implementation
======================================

Comprehensive MACD implementation with sophisticated exponential smoothing, ML-based signal optimization,
divergence detection, multi-timeframe analysis, parameter optimization, and volume-weighted calculations.

Features integrated from both MACD and MACD Signal implementations:
- Adaptive parameter optimization using differential evolution
- Volume-weighted MACD calculations for institutional flow detection
- Multi-timeframe coherence analysis
- Advanced histogram patterns with momentum acceleration detection
- ML-enhanced signal prediction and crossover detection
- Sophisticated divergence analysis with statistical validation
- Zero-line analysis with momentum tracking

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, FastICA
from scipy import signal as scipy_signal, stats
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class MACDIndicator(StandardIndicatorInterface):
    """
    Comprehensive Advanced MACD Indicator Implementation
    
    Features:
    - Sophisticated exponential smoothing with adaptive parameters
    - ML-based signal optimization and pattern recognition  
    - Advanced divergence detection algorithms
    - Multi-timeframe convergence analysis
    - Histogram pattern analysis with momentum acceleration
    - Zero-line cross detection with confirmation
    - Parameter optimization using differential evolution
    - Volume-weighted MACD calculations
    - Advanced signal line analysis with smoothing
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'adaptive_smoothing': True,
            'adaptive_periods': True,
            'volume_weighted': True,
            'ml_enhancement': True,
            'histogram_analysis': True,
            'multi_timeframe': True,
            'zero_line_analysis': True,
            'divergence_detection': True,
            'acceleration_detection': True,
            'smoothing_enabled': True,
            'ml_lookback': 80,
            'divergence_periods': 20,
            'zero_cross_confirmation': 3,
            'min_fast_period': 8,
            'max_fast_period': 18,
            'min_slow_period': 20,
            'max_slow_period': 35,
            'min_signal_period': 6,
            'max_signal_period': 15,
            'optimization_lookback': 100
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="MACDIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.ml_model = RandomForestClassifier(n_estimators=150, random_state=42)
        self.signal_classifier = RandomForestClassifier(n_estimators=150, random_state=42)
        self.crossover_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.histogram_analyzer = ExtraTreesRegressor(n_estimators=100, random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=5, random_state=42)
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        self.pca = PCA(n_components=3)
        self.ica = FastICA(n_components=2, random_state=42)
        self.ml_trained = False
        self.models_trained = False
        self.pattern_history = []
        
        self.history = {
            'macd_values': [],
            'signal_values': [],
            'histogram_values': [],
            'crossovers': [],
            'zero_line_crosses': [],
            'divergences': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['slow_period'], 
                        self.parameters['max_slow_period'] if 'max_slow_period' in self.parameters else self.parameters['slow_period'],
                        self.parameters['optimization_lookback'] if 'optimization_lookback' in self.parameters else self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max_period * 2 + 100,
            lookback_periods=300
        )
    
    def _optimize_parameters(self, data: pd.DataFrame) -> Tuple[int, int, int]:
        """Optimize MACD parameters using differential evolution"""
        if not self.parameters.get('adaptive_periods', False) or len(data) < self.parameters.get('optimization_lookback', 100):
            return (self.parameters['fast_period'], 
                   self.parameters['slow_period'], 
                   self.parameters['signal_period'])
        
        optimization_data = data.tail(self.parameters.get('optimization_lookback', 100))
        
        def objective_function(params):
            fast_period, slow_period, signal_period = int(params[0]), int(params[1]), int(params[2])
            
            # Ensure parameter constraints
            if fast_period >= slow_period or signal_period < 3:
                return 1000  # Penalty for invalid parameters
            
            try:
                # Calculate MACD with these parameters
                if self.parameters.get('volume_weighted', False):
                    ema_fast = self._calculate_volume_weighted_ema(optimization_data, fast_period)
                    ema_slow = self._calculate_volume_weighted_ema(optimization_data, slow_period)
                else:
                    ema_fast = optimization_data['close'].ewm(span=fast_period, adjust=False).mean()
                    ema_slow = optimization_data['close'].ewm(span=slow_period, adjust=False).mean()
                
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
                histogram = macd_line - signal_line
                
                # Calculate performance metrics
                returns = optimization_data['close'].pct_change()
                
                # Signal generation for optimization
                signals = pd.Series(0, index=macd_line.index)
                signals[macd_line > signal_line] = 1
                signals[macd_line < signal_line] = -1
                
                # Strategy returns
                strategy_returns = signals.shift(1) * returns
                
                # Optimization criteria
                total_return = strategy_returns.sum()
                volatility = strategy_returns.std()
                max_drawdown = self._calculate_max_drawdown(strategy_returns.cumsum())
                
                # Sharpe-like ratio with drawdown penalty
                if volatility > 0 and max_drawdown < 0.5:
                    score = total_return / volatility - max_drawdown * 2
                else:
                    score = -1000  # Penalty for high drawdown
                
                return -score  # Minimize negative score
            except:
                return 1000  # Penalty for calculation errors
        
        # Parameter bounds
        bounds = [
            (self.parameters.get('min_fast_period', 8), self.parameters.get('max_fast_period', 18)),
            (self.parameters.get('min_slow_period', 20), self.parameters.get('max_slow_period', 35)),
            (self.parameters.get('min_signal_period', 6), self.parameters.get('max_signal_period', 15))
        ]
        
        try:
            # Run optimization
            result = differential_evolution(
                objective_function, 
                bounds, 
                maxiter=50,
                popsize=10,
                seed=42,
                atol=1e-3
            )
            
            optimized_params = result.x
            fast_period = max(self.parameters.get('min_fast_period', 8), 
                            min(int(optimized_params[0]), self.parameters.get('max_fast_period', 18)))
            slow_period = max(self.parameters.get('min_slow_period', 20), 
                            min(int(optimized_params[1]), self.parameters.get('max_slow_period', 35)))
            signal_period = max(self.parameters.get('min_signal_period', 6), 
                              min(int(optimized_params[2]), self.parameters.get('max_signal_period', 15)))
            
            # Ensure constraints
            if fast_period >= slow_period:
                fast_period = slow_period - 2
                fast_period = max(fast_period, self.parameters.get('min_fast_period', 8))
            
            return fast_period, slow_period, signal_period
            
        except:
            # Fall back to default parameters if optimization fails
            return (self.parameters['fast_period'], 
                   self.parameters['slow_period'], 
                   self.parameters['signal_period'])
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdown.min())
    
    def _calculate_volume_weighted_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate volume-weighted EMA for more accurate MACD"""
        if not self.parameters.get('volume_weighted', False) or 'volume' not in data.columns:
            return data['close'].ewm(span=period, adjust=False).mean()
        
        close = data['close']
        volume = data['volume']
        
        # Volume-weighted price
        vwap = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # EMA of volume-weighted price
        vw_ema = vwap.ewm(span=period, adjust=False).mean()
        
        # Blend with regular EMA based on volume significance
        regular_ema = close.ewm(span=period, adjust=False).mean()
        volume_ratio = volume / volume.rolling(window=period).mean()
        volume_weight = np.tanh((volume_ratio - 1) * 0.3)  # Sigmoid weighting
        
        # Higher volume = more weight on volume-weighted EMA
        blended_ema = regular_ema * (1 - volume_weight) + vw_ema * volume_weight
        
        return blended_ema.fillna(regular_ema)
    
    def _calculate_macd_components(self, data: pd.DataFrame, fast_period: int, 
                                  slow_period: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram with advanced techniques"""
        # Calculate EMAs
        if self.parameters.get('volume_weighted', False):
            ema_fast = self._calculate_volume_weighted_ema(data, fast_period)
            ema_slow = self._calculate_volume_weighted_ema(data, slow_period)
        else:
            close = data['close']
            
            if self.parameters.get('adaptive_smoothing', False):
                # Adapt periods based on volatility
                volatility = close.pct_change().rolling(window=20).std()
                vol_factor = volatility / volatility.mean()
                
                # Adjust periods based on volatility
                fast_period = max(8, int(fast_period * vol_factor.iloc[-1]))
                slow_period = max(fast_period + 5, int(slow_period * vol_factor.iloc[-1]))
                signal_period = max(5, int(signal_period * vol_factor.iloc[-1]))
            
            ema_fast = close.ewm(span=fast_period).mean()
            ema_slow = close.ewm(span=slow_period).mean()
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line with optional smoothing
        if self.parameters.get('smoothing_enabled', False):
            # Apply additional smoothing to reduce noise
            smoothing_alpha = 2.0 / (signal_period + 1)
            signal_line = macd_line.ewm(alpha=smoothing_alpha, adjust=False).mean()
            
            # Optional Savitzky-Golay smoothing for signal line
            if len(signal_line.dropna()) > signal_period * 2:
                try:
                    valid_signal = signal_line.dropna()
                    if len(valid_signal) >= 7:
                        window_length = min(7, len(valid_signal) // 2 * 2 - 1)
                        smoothed_values = savgol_filter(valid_signal.values, window_length, 3)
                        signal_line.loc[valid_signal.index] = smoothed_values
                except:
                    pass  # Fall back to regular EMA if smoothing fails
        else:
            signal_line = macd_line.ewm(span=signal_period).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _detect_divergences(self, macd_line: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect MACD-Price divergences"""
        if len(macd_line) < self.parameters['divergence_periods']:
            return {'bullish': False, 'bearish': False, 'strength': 0.0}
        
        period = self.parameters['divergence_periods']
        recent_macd = macd_line.tail(period)
        recent_prices = data['close'].tail(period)
        
        # Find peaks and troughs
        macd_peaks, _ = scipy_signal.find_peaks(recent_macd.values, distance=3)
        macd_troughs, _ = scipy_signal.find_peaks(-recent_macd.values, distance=3)
        
        price_peaks, _ = scipy_signal.find_peaks(recent_prices.values, distance=3)
        price_troughs, _ = scipy_signal.find_peaks(-recent_prices.values, distance=3)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price lower low, MACD higher low
        if len(macd_troughs) >= 2 and len(price_troughs) >= 2:
            last_macd_trough = recent_macd.iloc[macd_troughs[-1]]
            prev_macd_trough = recent_macd.iloc[macd_troughs[-2]]
            
            last_price_trough = recent_prices.iloc[price_troughs[-1]]
            prev_price_trough = recent_prices.iloc[price_troughs[-2]]
            
            if last_price_trough < prev_price_trough and last_macd_trough > prev_macd_trough:
                bullish_divergence = True
                divergence_strength = abs(last_macd_trough - prev_macd_trough) / abs(prev_macd_trough) if prev_macd_trough != 0 else 0
        
        # Bearish divergence: price higher high, MACD lower high
        if len(macd_peaks) >= 2 and len(price_peaks) >= 2:
            last_macd_peak = recent_macd.iloc[macd_peaks[-1]]
            prev_macd_peak = recent_macd.iloc[macd_peaks[-2]]
            
            last_price_peak = recent_prices.iloc[price_peaks[-1]]
            prev_price_peak = recent_prices.iloc[price_peaks[-2]]
            
            if last_price_peak > prev_price_peak and last_macd_peak < prev_macd_peak:
                bearish_divergence = True
                divergence_strength = abs(last_macd_peak - prev_macd_peak) / abs(prev_macd_peak) if prev_macd_peak != 0 else 0
        
        return {
            'bullish': bullish_divergence,
            'bearish': bearish_divergence,
            'strength': min(divergence_strength, 1.0)
        }
    
    def _analyze_histogram_patterns(self, histogram: pd.Series) -> Dict[str, Any]:
        """Advanced histogram analysis for momentum acceleration detection"""
        if not self.parameters.get('histogram_analysis', True) or len(histogram) < 20:
            return {'acceleration': 'unknown', 'momentum_strength': 0.0, 'pattern': 'undefined'}
        
        recent_histogram = histogram.tail(20).dropna()
        
        if len(recent_histogram) < 10:
            return {'acceleration': 'unknown', 'momentum_strength': 0.0, 'pattern': 'undefined'}
        
        # Calculate histogram derivatives for acceleration analysis
        hist_diff = recent_histogram.diff()
        hist_diff2 = hist_diff.diff()  # Second derivative (acceleration)
        
        # Momentum strength
        momentum_strength = abs(recent_histogram.iloc[-1]) / (recent_histogram.std() + 1e-8)
        
        # Acceleration analysis
        recent_acceleration = hist_diff2.tail(5).mean()
        if recent_acceleration > 0.01:
            acceleration = 'increasing'
        elif recent_acceleration < -0.01:
            acceleration = 'decreasing'
        else:
            acceleration = 'stable'
        
        # Pattern recognition
        histogram_values = recent_histogram.values
        
        # Detect convergence/divergence patterns
        if len(histogram_values) >= 10:
            # Fit polynomial to detect pattern
            x = np.arange(len(histogram_values))
            poly_coeffs = np.polyfit(x, histogram_values, 2)
            
            # Analyze curvature
            curvature = poly_coeffs[0]
            trend = poly_coeffs[1]
            
            if abs(curvature) > 0.01:
                if curvature > 0:
                    if trend > 0:
                        pattern = 'bullish_acceleration'
                    else:
                        pattern = 'bearish_deceleration'
                else:  # curvature < 0
                    if trend > 0:
                        pattern = 'bullish_deceleration'
                    else:
                        pattern = 'bearish_acceleration'
            else:
                if abs(trend) > 0.01:
                    pattern = 'linear_trend'
                else:
                    pattern = 'sideways'
        else:
            pattern = 'insufficient_data'
        
        # Zero-line relationship
        positive_bars = len(recent_histogram[recent_histogram > 0])
        negative_bars = len(recent_histogram[recent_histogram < 0])
        zero_line_position = 'above' if positive_bars > negative_bars else 'below' if negative_bars > positive_bars else 'neutral'
        
        return {
            'acceleration': acceleration,
            'momentum_strength': float(momentum_strength),
            'pattern': pattern,
            'zero_line_position': zero_line_position,
            'positive_bars': positive_bars,
            'negative_bars': negative_bars,
            'recent_acceleration_value': float(recent_acceleration),
            'trend_coefficient': float(trend) if 'trend' in locals() else 0.0,
            'curvature_coefficient': float(curvature) if 'curvature' in locals() else 0.0
        }
    
    def _detect_crossovers(self, macd: pd.Series, signal: pd.Series) -> Dict[str, Any]:
        """Detect and analyze MACD-Signal crossovers"""
        crossovers = []
        
        if len(macd) < 2 or len(signal) < 2:
            return {'recent_crossovers': [], 'crossover_strength': 0.0}
        
        # Look for crossovers in recent data
        lookback = min(30, len(macd) - 1)
        recent_macd = macd.tail(lookback)
        recent_signal = signal.tail(lookback)
        
        for i in range(1, len(recent_macd)):
            prev_macd = recent_macd.iloc[i-1]
            curr_macd = recent_macd.iloc[i]
            prev_signal = recent_signal.iloc[i-1]
            curr_signal = recent_signal.iloc[i]
            
            # Bullish crossover
            if prev_macd <= prev_signal and curr_macd > curr_signal:
                strength = abs(curr_macd - curr_signal) / (recent_macd.std() + 1e-8)
                crossovers.append({
                    'type': 'bullish',
                    'position': len(recent_macd) - i,  # Bars ago
                    'strength': float(strength),
                    'macd_value': float(curr_macd),
                    'signal_value': float(curr_signal)
                })
            
            # Bearish crossover
            elif prev_macd >= prev_signal and curr_macd < curr_signal:
                strength = abs(curr_macd - curr_signal) / (recent_macd.std() + 1e-8)
                crossovers.append({
                    'type': 'bearish',
                    'position': len(recent_macd) - i,
                    'strength': float(strength),
                    'macd_value': float(curr_macd),
                    'signal_value': float(curr_signal)
                })
        
        # Calculate overall crossover strength
        if crossovers:
            recent_crossover = min(crossovers, key=lambda x: x['position'])
            crossover_strength = recent_crossover['strength']
        else:
            crossover_strength = 0.0
        
        return {
            'recent_crossovers': crossovers,
            'crossover_strength': crossover_strength,
            'total_crossovers': len(crossovers)
        }
    
    def _analyze_zero_line_behavior(self, macd: pd.Series) -> Dict[str, Any]:
        """Analyze MACD behavior around zero line"""
        if not self.parameters.get('zero_line_analysis', True) or len(macd) < 20:
            return {'zero_line_crosses': [], 'current_position': 'unknown'}
        
        recent_macd = macd.tail(20)
        zero_line_crosses = []
        
        # Detect zero line crossovers
        for i in range(1, len(recent_macd)):
            prev_macd = recent_macd.iloc[i-1]
            curr_macd = recent_macd.iloc[i]
            
            # Bullish zero line cross
            if prev_macd <= 0 and curr_macd > 0:
                momentum = abs(curr_macd) / (recent_macd.std() + 1e-8)
                zero_line_crosses.append({
                    'type': 'bullish',
                    'position': len(recent_macd) - i,
                    'momentum': float(momentum),
                    'value': float(curr_macd)
                })
            
            # Bearish zero line cross
            elif prev_macd >= 0 and curr_macd < 0:
                momentum = abs(curr_macd) / (recent_macd.std() + 1e-8)
                zero_line_crosses.append({
                    'type': 'bearish',
                    'position': len(recent_macd) - i,
                    'momentum': float(momentum),
                    'value': float(curr_macd)
                })
        
        # Current position relative to zero line
        current_macd = recent_macd.iloc[-1]
        if current_macd > 0.01:
            current_position = 'above'
        elif current_macd < -0.01:
            current_position = 'below'
        else:
            current_position = 'near_zero'
        
        # Distance from zero line
        zero_distance = abs(current_macd) / (recent_macd.std() + 1e-8)
        
        return {
            'zero_line_crosses': zero_line_crosses,
            'current_position': current_position,
            'zero_distance': float(zero_distance),
            'current_value': float(current_macd)
        }
    
    def _analyze_multi_timeframe_coherence(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze MACD coherence across multiple timeframes"""
        if not self.parameters.get('multi_timeframe', True) or len(data) < 100:
            return {'coherence': 'unknown', 'timeframe_signals': []}
        
        timeframe_signals = []
        
        # Short-term MACD (8, 17, 6)
        try:
            ema_fast_short = data['close'].ewm(span=8, adjust=False).mean()
            ema_slow_short = data['close'].ewm(span=17, adjust=False).mean()
            macd_short = ema_fast_short - ema_slow_short
            signal_short = macd_short.ewm(span=6, adjust=False).mean()
            
            short_trend = 'bullish' if macd_short.iloc[-1] > signal_short.iloc[-1] else 'bearish'
            timeframe_signals.append({
                'timeframe': 'short',
                'signal': short_trend,
                'macd': float(macd_short.iloc[-1]),
                'signal_line': float(signal_short.iloc[-1])
            })
        except:
            pass
        
        # Medium-term MACD (12, 26, 9) - standard
        try:
            ema_fast_med = data['close'].ewm(span=12, adjust=False).mean()
            ema_slow_med = data['close'].ewm(span=26, adjust=False).mean()
            macd_med = ema_fast_med - ema_slow_med
            signal_med = macd_med.ewm(span=9, adjust=False).mean()
            
            med_trend = 'bullish' if macd_med.iloc[-1] > signal_med.iloc[-1] else 'bearish'
            timeframe_signals.append({
                'timeframe': 'medium',
                'signal': med_trend,
                'macd': float(macd_med.iloc[-1]),
                'signal_line': float(signal_med.iloc[-1])
            })
        except:
            pass
        
        # Long-term MACD (19, 39, 15)
        if len(data) >= 39:
            try:
                ema_fast_long = data['close'].ewm(span=19, adjust=False).mean()
                ema_slow_long = data['close'].ewm(span=39, adjust=False).mean()
                macd_long = ema_fast_long - ema_slow_long
                signal_long = macd_long.ewm(span=15, adjust=False).mean()
                
                long_trend = 'bullish' if macd_long.iloc[-1] > signal_long.iloc[-1] else 'bearish'
                timeframe_signals.append({
                    'timeframe': 'long',
                    'signal': long_trend,
                    'macd': float(macd_long.iloc[-1]),
                    'signal_line': float(signal_long.iloc[-1])
                })
            except:
                pass
        
        # Analyze coherence
        if timeframe_signals:
            signals = [signal['signal'] for signal in timeframe_signals]
            bullish_count = signals.count('bullish')
            bearish_count = signals.count('bearish')
            
            if bullish_count == len(signals):
                coherence = 'strong_bullish'
            elif bearish_count == len(signals):
                coherence = 'strong_bearish'
            elif bullish_count > bearish_count:
                coherence = 'mixed_bullish'
            elif bearish_count > bullish_count:
                coherence = 'mixed_bearish'
            else:
                coherence = 'neutral'
        else:
            coherence = 'unknown'
        
        return {
            'coherence': coherence,
            'timeframe_signals': timeframe_signals,
            'alignment_ratio': max(bullish_count, bearish_count) / len(signals) if signals else 0.0
        }
    
    def _find_nearest_extreme(self, extremes: np.ndarray, target_idx: int) -> Optional[int]:
        """Find the nearest extreme point to a target index"""
        if len(extremes) == 0:
            return None
        
        min_distance = float('inf')
        nearest_extreme = None
        
        for extreme_idx in extremes:
            distance = abs(extreme_idx - target_idx)
            if distance < min_distance and distance < 10:
                min_distance = distance
                nearest_extreme = extreme_idx
        
        return nearest_extreme
    
    def _train_ml_model(self, macd_line: pd.Series, signal_line: pd.Series, 
                       histogram: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML model for pattern recognition"""
        if len(macd_line) < self.parameters.get('ml_lookback', 80) * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(macd_line, signal_line, histogram, data)
            if len(features) > 20:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                self.ml_model.fit(scaled_features, targets)
                self.ml_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_ml_data(self, macd_line: pd.Series, signal_line: pd.Series,
                        histogram: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 15
        
        for i in range(lookback, len(macd_line) - 5):
            macd_window = macd_line.iloc[i-lookback:i].values
            signal_window = signal_line.iloc[i-lookback:i].values
            hist_window = histogram.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values if 'volume' in data.columns else np.ones(lookback)
            
            feature_vector = [
                np.mean(macd_window), np.std(macd_window),
                np.mean(signal_window), np.std(signal_window),
                np.mean(hist_window), np.std(hist_window),
                macd_window[-1], signal_window[-1], hist_window[-1],
                macd_window[-1] - macd_window[0],  # MACD trend
                len([x for x in macd_window if x > 0]) / len(macd_window),  # Above zero ratio
                np.corrcoef(macd_window, price_window)[0, 1] if len(set(macd_window)) > 1 else 0,
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1
            ]
            
            # Target: future price movement
            future_return = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
            target = 2 if future_return > 0.02 else (0 if future_return < -0.02 else 1)
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def _get_ml_signal(self, macd_line: pd.Series, signal_line: pd.Series,
                      histogram: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based signal"""
        try:
            lookback = 15
            macd_window = macd_line.tail(lookback).values
            signal_window = signal_line.tail(lookback).values
            hist_window = histogram.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            volume_window = data['volume'].tail(lookback).values if 'volume' in data.columns else np.ones(lookback)
            
            feature_vector = np.array([[
                np.mean(macd_window), np.std(macd_window),
                np.mean(signal_window), np.std(signal_window),
                np.mean(hist_window), np.std(hist_window),
                macd_window[-1], signal_window[-1], hist_window[-1],
                macd_window[-1] - macd_window[0],
                len([x for x in macd_window if x > 0]) / len(macd_window),
                np.corrcoef(macd_window, price_window)[0, 1] if len(set(macd_window)) > 1 else 0,
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            ml_proba = self.ml_model.predict_proba(scaled_features)[0]
            
            if len(ml_proba) >= 3:
                if ml_proba[2] > 0.7:  # Strong bullish
                    return SignalType.BUY, ml_proba[2]
                elif ml_proba[0] > 0.7:  # Strong bearish
                    return SignalType.SELL, ml_proba[0]
        except:
            pass
        
        return None, 0.0
    
    def _detect_zero_line_crosses(self, macd_line: pd.Series) -> Dict[str, Any]:
        """Detect zero line crosses with confirmation"""
        if len(macd_line) < self.parameters.get('zero_cross_confirmation', 3) + 1:
            return {'cross_type': 'none', 'confirmed': False, 'bars_since': 0}
        
        current = macd_line.iloc[-1]
        previous = macd_line.iloc[-2]
        confirmation_bars = self.parameters.get('zero_cross_confirmation', 3)
        
        # Bullish cross (from below to above zero)
        if previous <= 0 and current > 0:
            # Check for confirmation
            confirmed = all(macd_line.iloc[-i] > 0 for i in range(1, min(confirmation_bars + 1, len(macd_line))))
            return {'cross_type': 'bullish', 'confirmed': confirmed, 'bars_since': 1}
        
        # Bearish cross (from above to below zero)
        elif previous >= 0 and current < 0:
            # Check for confirmation
            confirmed = all(macd_line.iloc[-i] < 0 for i in range(1, min(confirmation_bars + 1, len(macd_line))))
            return {'cross_type': 'bearish', 'confirmed': confirmed, 'bars_since': 1}
        
        # Count bars since last cross
        bars_since = 0
        for i in range(1, len(macd_line)):
            if (macd_line.iloc[-i] > 0) != (macd_line.iloc[-i-1] > 0):
                bars_since = i
                break
        
        return {'cross_type': 'none', 'confirmed': False, 'bars_since': bars_since}
    
    def _calculate_trend_strength(self, macd_line: pd.Series, signal_line: pd.Series) -> str:
        """Calculate trend strength"""
        if len(macd_line) < 10:
            return "insufficient_data"
        
        macd_trend = macd_line.tail(10).diff().mean()
        signal_trend = signal_line.tail(10).diff().mean()
        
        if abs(macd_trend) > macd_line.std() * 0.1 and abs(signal_trend) > signal_line.std() * 0.1:
            return "strong"
        elif abs(macd_trend) > macd_line.std() * 0.05 or abs(signal_trend) > signal_line.std() * 0.05:
            return "moderate"
        else:
            return "weak"
    
    def _determine_momentum_phase(self, macd_line: pd.Series, histogram: pd.Series) -> str:
        """Determine current momentum phase"""
        current_macd = macd_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        
        if current_macd > 0 and current_hist > 0:
            return "bullish_acceleration"
        elif current_macd > 0 and current_hist < 0:
            return "bullish_deceleration"
        elif current_macd < 0 and current_hist < 0:
            return "bearish_acceleration"
        elif current_macd < 0 and current_hist > 0:
            return "bearish_deceleration"
        else:
            return "neutral"
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'macd_comprehensive',
            'ml_trained': self.ml_trained,
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters.get('adaptive_periods', False),
            'volume_weighted': self.parameters.get('volume_weighted', False),
            'ml_enhancement': self.parameters.get('ml_enhancement', False),
            'histogram_analysis': self.parameters.get('histogram_analysis', True),
            'multi_timeframe': self.parameters.get('multi_timeframe', True),
            'zero_line_analysis': self.parameters.get('zero_line_analysis', True),
            'divergence_detection': self.parameters.get('divergence_detection', True),
            'acceleration_detection': self.parameters.get('acceleration_detection', True),
            'smoothing_enabled': self.parameters.get('smoothing_enabled', True),
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD with comprehensive analysis"""
        try:
            # Optimize parameters if enabled
            if self.parameters.get('adaptive_periods', False):
                fast_period, slow_period, signal_period = self._optimize_parameters(data)
            else:
                fast_period = self.parameters['fast_period']
                slow_period = self.parameters['slow_period']
                signal_period = self.parameters['signal_period']
            
            # Calculate MACD components
            macd_line, signal_line, histogram = self._calculate_macd_components(
                data, fast_period, slow_period, signal_period
            )
            
            # Advanced analysis
            histogram_analysis = self._analyze_histogram_patterns(histogram)
            crossover_analysis = self._detect_crossovers(macd_line, signal_line)
            zero_line_analysis = self._analyze_zero_line_behavior(macd_line)
            divergences = self._detect_divergences(macd_line, data)
            coherence_analysis = self._analyze_multi_timeframe_coherence(data)
            zero_crosses = self._detect_zero_line_crosses(macd_line)
            
            # Train ML models
            if not self.ml_trained:
                self._train_ml_model(macd_line, signal_line, histogram, data)
            
            # Generate comprehensive signal
            signal, confidence = self._generate_comprehensive_macd_signal(
                macd_line, signal_line, histogram, divergences,
                histogram_analysis, zero_crosses, crossover_analysis,
                zero_line_analysis, coherence_analysis, data
            )
            
            # Update history
            if len(macd_line) > 0 and not pd.isna(macd_line.iloc[-1]):
                self.history['macd_values'].append(float(macd_line.iloc[-1]))
                self.history['signal_values'].append(float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0)
                self.history['histogram_values'].append(float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0)
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'macd_line': float(macd_line.iloc[-1]) if len(macd_line) > 0 and not pd.isna(macd_line.iloc[-1]) else 0.0,
                'signal_line': float(signal_line.iloc[-1]) if len(signal_line) > 0 and not pd.isna(signal_line.iloc[-1]) else 0.0,
                'histogram': float(histogram.iloc[-1]) if len(histogram) > 0 and not pd.isna(histogram.iloc[-1]) else 0.0,
                'signal': signal,
                'confidence': confidence,
                'optimized_periods': {
                    'fast': fast_period,
                    'slow': slow_period,
                    'signal': signal_period
                },
                'divergences': divergences,
                'histogram_analysis': histogram_analysis,
                'crossover_analysis': crossover_analysis,
                'zero_line_analysis': zero_line_analysis,
                'zero_crosses': zero_crosses,
                'coherence_analysis': coherence_analysis,
                'trend_strength': self._calculate_trend_strength(macd_line, signal_line),
                'momentum_phase': self._determine_momentum_phase(macd_line, histogram),
                'market_regime': self._classify_market_regime(macd_line, signal_line, histogram),
                'values_history': {
                    'macd_line': macd_line.tail(30).tolist(),
                    'signal_line': signal_line.tail(30).tolist(),
                    'histogram': histogram.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate MACD: {str(e)}",
                cause=e
            )
    
    def _generate_macd_signal(self, macd_line: pd.Series, signal_line: pd.Series,
                             histogram: pd.Series, divergences: Dict, histogram_analysis: Dict,
                             zero_crosses: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive MACD signal"""
        signal_components = []
        confidence_components = []
        
        # Basic MACD signal crossover
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            signal_components.append(1.0)
            confidence_components.append(0.7)
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            signal_components.append(-1.0)
            confidence_components.append(0.7)
        
        # Zero line cross signals
        if zero_crosses['cross_type'] == 'bullish' and zero_crosses['confirmed']:
            signal_components.append(1.0)
            confidence_components.append(0.8)
        elif zero_crosses['cross_type'] == 'bearish' and zero_crosses['confirmed']:
            signal_components.append(-1.0)
            confidence_components.append(0.8)
        
        # Divergence signals
        if divergences['bullish']:
            signal_components.append(divergences['strength'])
            confidence_components.append(divergences['strength'])
        elif divergences['bearish']:
            signal_components.append(-divergences['strength'])
            confidence_components.append(divergences['strength'])
        
        # Histogram pattern signals
        if histogram_analysis['pattern'] == 'bullish_momentum_building':
            signal_components.append(histogram_analysis['confidence'])
            confidence_components.append(histogram_analysis['confidence'])
        elif histogram_analysis['pattern'] == 'bearish_momentum_building':
            signal_components.append(-histogram_analysis['confidence'])
            confidence_components.append(histogram_analysis['confidence'])
        
        # ML enhancement
        if self.ml_trained and len(macd_line) >= 15:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(macd_line, signal_line, histogram, data)
                if ml_signal:
                    signal_components.append(1.0 if ml_signal == SignalType.BUY else -1.0)
                    confidence_components.append(ml_confidence)
            except:
                pass
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        if weighted_signal > 0.4:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.4:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _generate_comprehensive_macd_signal(self, macd_line: pd.Series, signal_line: pd.Series,
                                           histogram: pd.Series, divergences: Dict, histogram_analysis: Dict,
                                           zero_crosses: Dict, crossover_analysis: Dict, zero_line_analysis: Dict,
                                           coherence_analysis: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive MACD signal with all analysis components"""
        signal_components = []
        confidence_components = []
        
        current_macd = macd_line.iloc[-1] if len(macd_line) > 0 and not pd.isna(macd_line.iloc[-1]) else 0
        current_signal = signal_line.iloc[-1] if len(signal_line) > 0 and not pd.isna(signal_line.iloc[-1]) else 0
        current_histogram = histogram.iloc[-1] if len(histogram) > 0 and not pd.isna(histogram.iloc[-1]) else 0
        
        # Basic MACD signal crossover
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            if current_macd > current_signal and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                signal_components.append(0.8)
                confidence_components.append(0.7)
            elif current_macd < current_signal and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                signal_components.append(-0.8)
                confidence_components.append(0.7)
        
        # Recent crossover signals from crossover analysis
        if crossover_analysis['recent_crossovers']:
            recent_crossover = crossover_analysis['recent_crossovers'][0]
            if recent_crossover['position'] <= 3:  # Recent crossover
                strength = recent_crossover['strength']
                if recent_crossover['type'] == 'bullish':
                    signal_components.append(0.9 * strength)
                    confidence_components.append(0.9)
                else:
                    signal_components.append(-0.9 * strength)
                    confidence_components.append(0.9)
        
        # Zero line cross signals
        if zero_crosses['cross_type'] == 'bullish' and zero_crosses['confirmed']:
            signal_components.append(1.0)
            confidence_components.append(0.8)
        elif zero_crosses['cross_type'] == 'bearish' and zero_crosses['confirmed']:
            signal_components.append(-1.0)
            confidence_components.append(0.8)
        
        # Zero line position signals from analysis
        zero_position = zero_line_analysis['current_position']
        if zero_position == 'above':
            signal_components.append(0.5)
            confidence_components.append(0.6)
        elif zero_position == 'below':
            signal_components.append(-0.5)
            confidence_components.append(0.6)
        
        # Divergence signals
        if divergences['bullish']:
            signal_components.append(divergences['strength'])
            confidence_components.append(divergences['strength'])
        elif divergences['bearish']:
            signal_components.append(-divergences['strength'])
            confidence_components.append(divergences['strength'])
        
        # Histogram pattern signals
        hist_pattern = histogram_analysis['pattern']
        momentum_strength = histogram_analysis['momentum_strength']
        
        if hist_pattern == 'bullish_acceleration' and momentum_strength > 0.5:
            signal_components.append(0.7)
            confidence_components.append(0.7)
        elif hist_pattern == 'bearish_acceleration' and momentum_strength > 0.5:
            signal_components.append(-0.7)
            confidence_components.append(0.7)
        elif hist_pattern == 'bullish_momentum_building':
            signal_components.append(histogram_analysis.get('confidence', 0.6))
            confidence_components.append(histogram_analysis.get('confidence', 0.6))
        elif hist_pattern == 'bearish_momentum_building':
            signal_components.append(-histogram_analysis.get('confidence', 0.6))
            confidence_components.append(histogram_analysis.get('confidence', 0.6))
        
        # Multi-timeframe coherence
        coherence = coherence_analysis['coherence']
        alignment_ratio = coherence_analysis['alignment_ratio']
        
        if coherence == 'strong_bullish' and alignment_ratio > 0.8:
            signal_components.append(0.8)
            confidence_components.append(0.8)
        elif coherence == 'strong_bearish' and alignment_ratio > 0.8:
            signal_components.append(-0.8)
            confidence_components.append(0.8)
        elif coherence in ['mixed_bullish', 'mixed_bearish']:
            direction = 0.6 if 'bullish' in coherence else -0.6
            signal_components.append(direction * alignment_ratio)
            confidence_components.append(0.6)
        
        # ML enhancement
        if self.ml_trained and len(macd_line) >= 15:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(macd_line, signal_line, histogram, data)
                if ml_signal:
                    signal_components.append(1.0 if ml_signal == SignalType.BUY else -1.0)
                    confidence_components.append(ml_confidence)
            except:
                pass
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        if weighted_signal > 0.5:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.5:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _classify_market_regime(self, macd: pd.Series, signal: pd.Series, histogram: pd.Series) -> Dict[str, Any]:
        """Classify market regime based on MACD characteristics"""
        if len(macd) < 30:
            return {'regime': 'unknown', 'trend_state': 'undefined', 'momentum_phase': 'unclear'}
        
        recent_macd = macd.tail(30)
        recent_signal = signal.tail(30)
        recent_histogram = histogram.tail(30)
        
        # Trend classification
        macd_above_zero = len(recent_macd[recent_macd > 0]) / len(recent_macd)
        signal_above_zero = len(recent_signal[recent_signal > 0]) / len(recent_signal)
        
        if macd_above_zero > 0.7 and signal_above_zero > 0.7:
            trend_state = 'strong_uptrend'
        elif macd_above_zero < 0.3 and signal_above_zero < 0.3:
            trend_state = 'strong_downtrend'
        elif macd_above_zero > 0.6 or signal_above_zero > 0.6:
            trend_state = 'uptrend'
        elif macd_above_zero < 0.4 or signal_above_zero < 0.4:
            trend_state = 'downtrend'
        else:
            trend_state = 'sideways'
        
        # Momentum phase
        histogram_trend = np.polyfit(range(len(recent_histogram)), recent_histogram.values, 1)[0]
        current_histogram = recent_histogram.iloc[-1]
        
        if current_histogram > 0 and histogram_trend > 0:
            momentum_phase = 'accelerating_up'
        elif current_histogram > 0 and histogram_trend < 0:
            momentum_phase = 'decelerating_up'
        elif current_histogram < 0 and histogram_trend < 0:
            momentum_phase = 'accelerating_down'
        elif current_histogram < 0 and histogram_trend > 0:
            momentum_phase = 'decelerating_down'
        else:
            momentum_phase = 'neutral'
        
        # Overall regime
        if trend_state in ['strong_uptrend', 'uptrend'] and momentum_phase in ['accelerating_up']:
            regime = 'bullish_momentum'
        elif trend_state in ['strong_downtrend', 'downtrend'] and momentum_phase in ['accelerating_down']:
            regime = 'bearish_momentum'
        elif 'decelerating' in momentum_phase:
            regime = 'momentum_exhaustion'
        elif trend_state == 'sideways':
            regime = 'consolidation'
        else:
            regime = 'transitional'
        
        return {
            'regime': regime,
            'trend_state': trend_state,
            'momentum_phase': momentum_phase,
            'macd_above_zero_ratio': float(macd_above_zero),
            'signal_above_zero_ratio': float(signal_above_zero),
            'histogram_trend': float(histogram_trend)
        }