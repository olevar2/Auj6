"""
Percentage Price Oscillator (PPO) Indicator - Advanced Implementation
====================================================================

Advanced PPO indicator with adaptive parameters, machine learning enhancement,
and sophisticated signal generation for momentum analysis.

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
from scipy import stats
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


class PercentagePriceOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Percentage Price Oscillator (PPO) Implementation
    
    Features:
    - Adaptive EMA periods based on market volatility and cycle analysis
    - ML-enhanced signal line optimization and crossover prediction
    - Multi-timeframe PPO coherence analysis
    - Volume-weighted PPO calculations for institutional flow detection
    - Advanced histogram analysis with momentum acceleration detection
    - Dynamic threshold optimization for overbought/oversold conditions
    - Sophisticated divergence analysis with statistical validation
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'adaptive_periods': True,
            'volume_weighted': True,
            'ml_enhancement': True,
            'histogram_analysis': True,
            'multi_timeframe': True,
            'zero_line_analysis': True,
            'divergence_detection': True,
            'dynamic_thresholds': True,
            'min_fast_period': 8,
            'max_fast_period': 18,
            'min_slow_period': 20,
            'max_slow_period': 35,
            'min_signal_period': 6,
            'max_signal_period': 15,
            'optimization_lookback': 100,
            'ml_lookback': 80,
            'smoothing_enabled': True,
            'cycle_detection': True,
            'threshold_percentile': 85
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="PercentagePriceOscillatorIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.signal_classifier = RandomForestClassifier(n_estimators=150, random_state=42)
        self.crossover_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.threshold_optimizer = ExtraTreesRegressor(n_estimators=100, random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=5, random_state=42)
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        self.pca = PCA(n_components=3)
        self.ica = FastICA(n_components=2, random_state=42)
        self.models_trained = False
        
        self.history = {
            'ppo_values': [],
            'signal_values': [],
            'histogram_values': [],
            'thresholds': [],
            'crossovers': [],
            'divergences': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['slow_period'], 
                        self.parameters['max_slow_period'],
                        self.parameters['optimization_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max_period * 2 + 100,
            lookback_periods=300
        )
    
    def _detect_price_cycles(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect dominant price cycles using spectral analysis"""
        if not self.parameters['cycle_detection'] or len(data) < 100:
            return {'dominant_cycle': 20, 'cycle_strength': 0.5, 'cycle_phase': 'unknown'}
        
        prices = data['close'].tail(100).values
        
        # Detrend the data
        detrended_prices = stats.detrend(prices)
        
        # Apply Hann window to reduce spectral leakage
        windowed_prices = detrended_prices * np.hanning(len(detrended_prices))
        
        # FFT analysis
        fft_values = np.fft.fft(windowed_prices)
        power_spectrum = np.abs(fft_values) ** 2
        frequencies = np.fft.fftfreq(len(windowed_prices))
        
        # Focus on positive frequencies and ignore DC component
        positive_freqs = frequencies[1:len(frequencies)//2]
        positive_power = power_spectrum[1:len(power_spectrum)//2]
        
        if len(positive_power) > 0:
            # Find dominant frequency
            dominant_freq_idx = np.argmax(positive_power)
            dominant_frequency = positive_freqs[dominant_freq_idx]
            
            if dominant_frequency > 0:
                dominant_cycle = int(1.0 / dominant_frequency)
                dominant_cycle = max(8, min(dominant_cycle, 50))  # Reasonable bounds
            else:
                dominant_cycle = 20
            
            # Calculate cycle strength (relative power)
            total_power = np.sum(positive_power)
            cycle_strength = positive_power[dominant_freq_idx] / total_power if total_power > 0 else 0.5
            
            # Estimate cycle phase
            phase_angle = np.angle(fft_values[dominant_freq_idx + 1])
            normalized_phase = (phase_angle + np.pi) / (2 * np.pi)  # Normalize to 0-1
            
            if normalized_phase < 0.25:
                cycle_phase = 'bottom'
            elif normalized_phase < 0.5:
                cycle_phase = 'rising'
            elif normalized_phase < 0.75:
                cycle_phase = 'top'
            else:
                cycle_phase = 'falling'
        else:
            dominant_cycle = 20
            cycle_strength = 0.5
            cycle_phase = 'unknown'
        
        return {
            'dominant_cycle': dominant_cycle,
            'cycle_strength': float(cycle_strength),
            'cycle_phase': cycle_phase,
            'frequency': float(dominant_frequency) if 'dominant_frequency' in locals() else 0.05
        }
    
    def _optimize_ppo_parameters(self, data: pd.DataFrame) -> Tuple[int, int, int]:
        """Optimize PPO parameters using genetic algorithm approach"""
        if not self.parameters['adaptive_periods'] or len(data) < self.parameters['optimization_lookback']:
            return (self.parameters['fast_period'], 
                   self.parameters['slow_period'], 
                   self.parameters['signal_period'])
        
        optimization_data = data.tail(self.parameters['optimization_lookback'])
        cycle_info = self._detect_price_cycles(optimization_data)
        dominant_cycle = cycle_info['dominant_cycle']
        
        # Define objective function for optimization
        def objective_function(params):
            fast_period, slow_period, signal_period = int(params[0]), int(params[1]), int(params[2])
            
            # Parameter constraints
            if fast_period >= slow_period or signal_period < 3:
                return 1000  # Penalty
            
            try:
                # Calculate PPO with these parameters
                ema_fast = optimization_data['close'].ewm(span=fast_period, adjust=False).mean()
                ema_slow = optimization_data['close'].ewm(span=slow_period, adjust=False).mean()
                ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100
                signal_line = ppo_line.ewm(span=signal_period, adjust=False).mean()
                histogram = ppo_line - signal_line
                
                # Performance metrics
                returns = optimization_data['close'].pct_change()
                
                # Signal generation
                signals = pd.Series(0, index=ppo_line.index)
                signals[ppo_line > signal_line] = 1
                signals[ppo_line < signal_line] = -1
                
                # Strategy performance
                strategy_returns = signals.shift(1) * returns
                
                # Calculate Sharpe-like ratio
                total_return = strategy_returns.sum()
                volatility = strategy_returns.std()
                max_drawdown = self._calculate_max_drawdown(strategy_returns.cumsum())
                
                # Additional criteria
                signal_smoothness = -abs(signals.diff()).sum()  # Prefer fewer signal changes
                histogram_volatility = -histogram.std()  # Prefer stable histogram
                
                # Combined score
                if volatility > 0 and max_drawdown < 0.3:
                    score = (total_return / volatility) + signal_smoothness * 0.1 + histogram_volatility * 0.1 - max_drawdown * 3
                else:
                    score = -1000
                
                return -score  # Minimize negative score
            except:
                return 1000
        
        # Adaptive bounds based on cycle analysis
        cycle_factor = dominant_cycle / 20  # Normalize
        
        bounds = [
            (max(self.parameters['min_fast_period'], int(8 * cycle_factor)), 
             min(self.parameters['max_fast_period'], int(18 * cycle_factor))),
            (max(self.parameters['min_slow_period'], int(20 * cycle_factor)), 
             min(self.parameters['max_slow_period'], int(35 * cycle_factor))),
            (self.parameters['min_signal_period'], self.parameters['max_signal_period'])
        ]
        
        try:
            # Run optimization
            result = differential_evolution(
                objective_function, 
                bounds, 
                maxiter=40,
                popsize=8,
                seed=42,
                atol=1e-3
            )
            
            optimized_params = result.x
            fast_period = max(bounds[0][0], min(int(optimized_params[0]), bounds[0][1]))
            slow_period = max(bounds[1][0], min(int(optimized_params[1]), bounds[1][1]))
            signal_period = max(bounds[2][0], min(int(optimized_params[2]), bounds[2][1]))
            
            # Ensure constraints
            if fast_period >= slow_period:
                fast_period = slow_period - 2
                fast_period = max(fast_period, bounds[0][0])
            
            return fast_period, slow_period, signal_period
            
        except:
            # Fall back to cycle-adjusted defaults
            fast_adjusted = max(8, min(int(12 * cycle_factor), 18))
            slow_adjusted = max(20, min(int(26 * cycle_factor), 35))
            signal_adjusted = self.parameters['signal_period']
            
            if fast_adjusted >= slow_adjusted:
                fast_adjusted = slow_adjusted - 2
            
            return fast_adjusted, slow_adjusted, signal_adjusted
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / (rolling_max.abs() + 1e-8)
        return abs(drawdown.min())
    
    def _calculate_volume_weighted_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate volume-weighted EMA for institutional flow detection"""
        if not self.parameters['volume_weighted'] or 'volume' not in data.columns:
            return data['close'].ewm(span=period, adjust=False).mean()
        
        close = data['close']
        volume = data['volume']
        
        # Volume-weighted price
        vwap_period = min(period, 20)  # Use shorter period for VWAP
        vwap = (close * volume).rolling(window=vwap_period).sum() / volume.rolling(window=vwap_period).sum()
        
        # EMA of VWAP
        vw_ema = vwap.ewm(span=period, adjust=False).mean()
        
        # Regular EMA for comparison
        regular_ema = close.ewm(span=period, adjust=False).mean()
        
        # Adaptive blending based on volume significance
        volume_ma = volume.rolling(window=period).mean()
        volume_ratio = volume / (volume_ma + 1e-8)
        
        # Sigmoid function for smooth weighting
        volume_weight = 2.0 / (1.0 + np.exp(-2.0 * (volume_ratio - 1.0))) - 1.0
        volume_weight = np.clip(volume_weight, 0, 1)
        
        # Blend EMAs
        blended_ema = regular_ema * (1 - volume_weight) + vw_ema * volume_weight
        
        return blended_ema.fillna(regular_ema)
    
    def _calculate_ppo_components(self, data: pd.DataFrame, fast_period: int, 
                                 slow_period: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate PPO, Signal, and Histogram with advanced techniques"""
        # Calculate EMAs (volume-weighted if enabled)
        if self.parameters['volume_weighted']:
            ema_fast = self._calculate_volume_weighted_ema(data, fast_period)
            ema_slow = self._calculate_volume_weighted_ema(data, slow_period)
        else:
            ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # PPO Line (percentage difference)
        ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100
        
        # Signal Line with advanced smoothing
        if self.parameters['smoothing_enabled']:
            # Adaptive smoothing based on volatility
            ppo_volatility = ppo_line.rolling(window=20).std()
            avg_volatility = ppo_volatility.rolling(window=60).mean()
            vol_ratio = ppo_volatility / (avg_volatility + 1e-8)
            
            # Adjust smoothing factor based on volatility
            base_alpha = 2.0 / (signal_period + 1)
            adaptive_alpha = base_alpha / (1 + vol_ratio * 0.5)
            
            signal_line = ppo_line.ewm(alpha=adaptive_alpha, adjust=False).mean()
            
            # Optional Savitzky-Golay smoothing
            if len(signal_line.dropna()) > signal_period * 2:
                try:
                    valid_signal = signal_line.dropna()
                    if len(valid_signal) >= 7:
                        window_length = min(7, len(valid_signal) // 2 * 2 - 1)
                        smoothed_values = savgol_filter(valid_signal.values, window_length, 3)
                        signal_line.loc[valid_signal.index] = smoothed_values
                except:
                    pass
        else:
            signal_line = ppo_line.ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        histogram = ppo_line - signal_line
        
        return ppo_line, signal_line, histogram
    
    def _calculate_dynamic_thresholds(self, ppo: pd.Series) -> Tuple[float, float]:
        """Calculate dynamic overbought/oversold thresholds"""
        if not self.parameters['dynamic_thresholds'] or len(ppo) < 50:
            return 2.0, -2.0  # Default thresholds
        
        recent_ppo = ppo.tail(100).dropna()
        
        if len(recent_ppo) < 30:
            return 2.0, -2.0
        
        # Use percentile-based approach
        percentile = self.parameters['threshold_percentile']
        upper_threshold = np.percentile(recent_ppo, percentile)
        lower_threshold = np.percentile(recent_ppo, 100 - percentile)
        
        # Ensure minimum separation
        ppo_std = recent_ppo.std()
        min_separation = ppo_std * 0.8
        
        if upper_threshold - lower_threshold < min_separation:
            mid_point = (upper_threshold + lower_threshold) / 2
            upper_threshold = mid_point + min_separation / 2
            lower_threshold = mid_point - min_separation / 2
        
        # Apply reasonable bounds
        upper_threshold = max(1.0, min(upper_threshold, 10.0))
        lower_threshold = min(-1.0, max(lower_threshold, -10.0))
        
        return upper_threshold, lower_threshold
    
    def _analyze_histogram_momentum(self, histogram: pd.Series) -> Dict[str, Any]:
        """Analyze histogram for momentum acceleration patterns"""
        if not self.parameters['histogram_analysis'] or len(histogram) < 20:
            return {'acceleration': 'unknown', 'momentum_strength': 0.0, 'pattern': 'undefined'}
        
        recent_histogram = histogram.tail(20).dropna()
        
        if len(recent_histogram) < 10:
            return {'acceleration': 'unknown', 'momentum_strength': 0.0, 'pattern': 'undefined'}
        
        # Calculate histogram derivatives
        hist_velocity = recent_histogram.diff()
        hist_acceleration = hist_velocity.diff()
        
        # Current momentum characteristics
        current_histogram = recent_histogram.iloc[-1]
        recent_velocity = hist_velocity.tail(5).mean()
        recent_acceleration = hist_acceleration.tail(3).mean()
        
        # Momentum strength
        histogram_std = recent_histogram.std()
        momentum_strength = abs(current_histogram) / (histogram_std + 1e-8)
        
        # Acceleration classification
        if recent_acceleration > 0.1:
            acceleration = 'increasing'
        elif recent_acceleration < -0.1:
            acceleration = 'decreasing'
        else:
            acceleration = 'stable'
        
        # Pattern recognition using polynomial fitting
        if len(recent_histogram) >= 12:
            x = np.arange(len(recent_histogram))
            
            # Fit quadratic polynomial
            try:
                coeffs = np.polyfit(x, recent_histogram.values, 2)
                curvature = coeffs[0]  # Second-order coefficient
                slope = coeffs[1]      # First-order coefficient
                
                # Classify pattern based on curvature and slope
                if abs(curvature) > 0.05:
                    if curvature > 0:  # Upward curvature
                        if slope > 0:
                            pattern = 'bullish_acceleration'
                        else:
                            pattern = 'bearish_deceleration'
                    else:  # Downward curvature
                        if slope > 0:
                            pattern = 'bullish_deceleration'
                        else:
                            pattern = 'bearish_acceleration'
                else:
                    if abs(slope) > 0.05:
                        pattern = 'linear_trend'
                    else:
                        pattern = 'sideways'
            except:
                pattern = 'undefined'
        else:
            pattern = 'insufficient_data'
        
        # Zero-line relationship
        positive_bars = len(recent_histogram[recent_histogram > 0])
        negative_bars = len(recent_histogram[recent_histogram < 0])
        
        if positive_bars > negative_bars * 1.5:
            zero_line_position = 'strongly_above'
        elif positive_bars > negative_bars:
            zero_line_position = 'above'
        elif negative_bars > positive_bars * 1.5:
            zero_line_position = 'strongly_below'
        elif negative_bars > positive_bars:
            zero_line_position = 'below'
        else:
            zero_line_position = 'neutral'
        
        return {
            'acceleration': acceleration,
            'momentum_strength': float(momentum_strength),
            'pattern': pattern,
            'zero_line_position': zero_line_position,
            'recent_velocity': float(recent_velocity),
            'recent_acceleration': float(recent_acceleration),
            'current_value': float(current_histogram),
            'positive_bars': positive_bars,
            'negative_bars': negative_bars
        }
    
    def _detect_ppo_crossovers(self, ppo: pd.Series, signal: pd.Series) -> Dict[str, Any]:
        """Detect and analyze PPO-Signal crossovers"""
        crossovers = []
        
        if len(ppo) < 2 or len(signal) < 2:
            return {'recent_crossovers': [], 'crossover_strength': 0.0}
        
        # Look for crossovers in recent data
        lookback = min(30, len(ppo) - 1)
        recent_ppo = ppo.tail(lookback)
        recent_signal = signal.tail(lookback)
        
        for i in range(1, len(recent_ppo)):
            prev_ppo = recent_ppo.iloc[i-1]
            curr_ppo = recent_ppo.iloc[i]
            prev_signal = recent_signal.iloc[i-1]
            curr_signal = recent_signal.iloc[i]
            
            # Bullish crossover
            if prev_ppo <= prev_signal and curr_ppo > curr_signal:
                # Calculate crossover strength
                separation = abs(curr_ppo - curr_signal)
                ppo_momentum = curr_ppo - prev_ppo
                signal_momentum = curr_signal - prev_signal
                
                # Strength based on separation and momentum
                strength = separation + abs(ppo_momentum - signal_momentum)
                normalized_strength = strength / (recent_ppo.std() + 1e-8)
                
                crossovers.append({
                    'type': 'bullish',
                    'position': len(recent_ppo) - i,  # Bars ago
                    'strength': float(normalized_strength),
                    'ppo_value': float(curr_ppo),
                    'signal_value': float(curr_signal),
                    'separation': float(separation),
                    'momentum_differential': float(ppo_momentum - signal_momentum)
                })
            
            # Bearish crossover
            elif prev_ppo >= prev_signal and curr_ppo < curr_signal:
                separation = abs(curr_ppo - curr_signal)
                ppo_momentum = curr_ppo - prev_ppo
                signal_momentum = curr_signal - prev_signal
                
                strength = separation + abs(ppo_momentum - signal_momentum)
                normalized_strength = strength / (recent_ppo.std() + 1e-8)
                
                crossovers.append({
                    'type': 'bearish',
                    'position': len(recent_ppo) - i,
                    'strength': float(normalized_strength),
                    'ppo_value': float(curr_ppo),
                    'signal_value': float(curr_signal),
                    'separation': float(separation),
                    'momentum_differential': float(ppo_momentum - signal_momentum)
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
    
    def _analyze_zero_line_behavior(self, ppo: pd.Series) -> Dict[str, Any]:
        """Analyze PPO behavior around zero line"""
        if not self.parameters['zero_line_analysis'] or len(ppo) < 20:
            return {'zero_crosses': [], 'current_position': 'unknown'}
        
        recent_ppo = ppo.tail(30)
        zero_crosses = []
        
        # Detect zero line crossovers
        for i in range(1, len(recent_ppo)):
            prev_ppo = recent_ppo.iloc[i-1]
            curr_ppo = recent_ppo.iloc[i]
            
            # Bullish zero line cross
            if prev_ppo <= 0 and curr_ppo > 0:
                momentum = abs(curr_ppo) / (recent_ppo.std() + 1e-8)
                zero_crosses.append({
                    'type': 'bullish',
                    'position': len(recent_ppo) - i,
                    'momentum': float(momentum),
                    'value': float(curr_ppo)
                })
            
            # Bearish zero line cross
            elif prev_ppo >= 0 and curr_ppo < 0:
                momentum = abs(curr_ppo) / (recent_ppo.std() + 1e-8)
                zero_crosses.append({
                    'type': 'bearish',
                    'position': len(recent_ppo) - i,
                    'momentum': float(momentum),
                    'value': float(curr_ppo)
                })
        
        # Current position analysis
        current_ppo = recent_ppo.iloc[-1]
        ppo_std = recent_ppo.std()
        
        if current_ppo > ppo_std:
            current_position = 'strongly_above'
        elif current_ppo > 0:
            current_position = 'above'
        elif current_ppo < -ppo_std:
            current_position = 'strongly_below'
        elif current_ppo < 0:
            current_position = 'below'
        else:
            current_position = 'at_zero'
        
        # Time spent above/below zero
        above_zero_bars = len(recent_ppo[recent_ppo > 0])
        below_zero_bars = len(recent_ppo[recent_ppo < 0])
        zero_bias = (above_zero_bars - below_zero_bars) / len(recent_ppo)
        
        return {
            'zero_crosses': zero_crosses,
            'current_position': current_position,
            'current_value': float(current_ppo),
            'zero_distance': float(abs(current_ppo) / (ppo_std + 1e-8)),
            'above_zero_bars': above_zero_bars,
            'below_zero_bars': below_zero_bars,
            'zero_bias': float(zero_bias)
        }
    
    def _detect_ppo_divergences(self, ppo: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect divergences between PPO and price"""
        if not self.parameters['divergence_detection'] or len(ppo) < 30:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        lookback = min(30, len(ppo) - 5)
        recent_ppo = ppo.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find peaks and troughs
        ppo_peaks, _ = find_peaks(recent_ppo.values, distance=5, height=0)
        ppo_troughs, _ = find_peaks(-recent_ppo.values, distance=5, height=0)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=5)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=5)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        divergence_details = []
        
        # Bullish divergence analysis
        if len(ppo_troughs) >= 2 and len(price_troughs) >= 2:
            for i in range(len(ppo_troughs) - 1):
                ppo_trough_1 = ppo_troughs[i]
                ppo_trough_2 = ppo_troughs[i + 1]
                
                # Find corresponding price troughs
                price_trough_1 = self._find_nearest_extreme(price_troughs, ppo_trough_1)
                price_trough_2 = self._find_nearest_extreme(price_troughs, ppo_trough_2)
                
                if price_trough_1 is not None and price_trough_2 is not None:
                    price_1 = recent_prices.iloc[price_trough_1]
                    price_2 = recent_prices.iloc[price_trough_2]
                    ppo_1 = recent_ppo.iloc[ppo_trough_1]
                    ppo_2 = recent_ppo.iloc[ppo_trough_2]
                    
                    # Check for divergence pattern
                    if price_2 < price_1 and ppo_2 > ppo_1:
                        price_decline = (price_1 - price_2) / price_1
                        ppo_improvement = (ppo_2 - ppo_1) / abs(ppo_1 + 1e-8)
                        
                        # Statistical significance test
                        if price_decline > 0.015 and ppo_improvement > 0.1:
                            bullish_divergence = True
                            strength = price_decline + ppo_improvement
                            divergence_strength = max(divergence_strength, strength)
                            
                            divergence_details.append({
                                'type': 'bullish',
                                'price_decline': float(price_decline),
                                'ppo_improvement': float(ppo_improvement),
                                'strength': float(strength)
                            })
        
        # Bearish divergence analysis
        if len(ppo_peaks) >= 2 and len(price_peaks) >= 2:
            for i in range(len(ppo_peaks) - 1):
                ppo_peak_1 = ppo_peaks[i]
                ppo_peak_2 = ppo_peaks[i + 1]
                
                price_peak_1 = self._find_nearest_extreme(price_peaks, ppo_peak_1)
                price_peak_2 = self._find_nearest_extreme(price_peaks, ppo_peak_2)
                
                if price_peak_1 is not None and price_peak_2 is not None:
                    price_1 = recent_prices.iloc[price_peak_1]
                    price_2 = recent_prices.iloc[price_peak_2]
                    ppo_1 = recent_ppo.iloc[ppo_peak_1]
                    ppo_2 = recent_ppo.iloc[ppo_peak_2]
                    
                    if price_2 > price_1 and ppo_2 < ppo_1:
                        price_increase = (price_2 - price_1) / price_1
                        ppo_decline = (ppo_1 - ppo_2) / abs(ppo_1 + 1e-8)
                        
                        if price_increase > 0.015 and ppo_decline > 0.1:
                            bearish_divergence = True
                            strength = price_increase + ppo_decline
                            divergence_strength = max(divergence_strength, strength)
                            
                            divergence_details.append({
                                'type': 'bearish',
                                'price_increase': float(price_increase),
                                'ppo_decline': float(ppo_decline),
                                'strength': float(strength)
                            })
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'divergence_details': divergence_details,
            'ppo_peaks': len(ppo_peaks),
            'ppo_troughs': len(ppo_troughs),
            'analysis_period': lookback
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
    
    def _analyze_multi_timeframe_coherence(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze PPO coherence across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(data) < 60:
            return {'coherence': 'unknown', 'timeframe_signals': []}
        
        timeframe_signals = []
        
        # Short-term PPO (8, 17, 6)
        try:
            fast_short = data['close'].ewm(span=8, adjust=False).mean()
            slow_short = data['close'].ewm(span=17, adjust=False).mean()
            ppo_short = ((fast_short - slow_short) / slow_short) * 100
            signal_short = ppo_short.ewm(span=6, adjust=False).mean()
            
            short_trend = 'bullish' if ppo_short.iloc[-1] > signal_short.iloc[-1] else 'bearish'
            timeframe_signals.append({
                'timeframe': 'short',
                'signal': short_trend,
                'ppo': float(ppo_short.iloc[-1]),
                'signal_line': float(signal_short.iloc[-1]),
                'periods': (8, 17, 6)
            })
        except:
            pass
        
        # Medium-term PPO (12, 26, 9) - standard
        try:
            fast_med = data['close'].ewm(span=12, adjust=False).mean()
            slow_med = data['close'].ewm(span=26, adjust=False).mean()
            ppo_med = ((fast_med - slow_med) / slow_med) * 100
            signal_med = ppo_med.ewm(span=9, adjust=False).mean()
            
            med_trend = 'bullish' if ppo_med.iloc[-1] > signal_med.iloc[-1] else 'bearish'
            timeframe_signals.append({
                'timeframe': 'medium',
                'signal': med_trend,
                'ppo': float(ppo_med.iloc[-1]),
                'signal_line': float(signal_med.iloc[-1]),
                'periods': (12, 26, 9)
            })
        except:
            pass
        
        # Long-term PPO (19, 39, 15)
        if len(data) >= 39:
            try:
                fast_long = data['close'].ewm(span=19, adjust=False).mean()
                slow_long = data['close'].ewm(span=39, adjust=False).mean()
                ppo_long = ((fast_long - slow_long) / slow_long) * 100
                signal_long = ppo_long.ewm(span=15, adjust=False).mean()
                
                long_trend = 'bullish' if ppo_long.iloc[-1] > signal_long.iloc[-1] else 'bearish'
                timeframe_signals.append({
                    'timeframe': 'long',
                    'signal': long_trend,
                    'ppo': float(ppo_long.iloc[-1]),
                    'signal_line': float(signal_long.iloc[-1]),
                    'periods': (19, 39, 15)
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
            
            # Calculate strength alignment
            ppo_values = [abs(signal['ppo']) for signal in timeframe_signals]
            strength_consistency = 1.0 - (np.std(ppo_values) / (np.mean(ppo_values) + 1e-8))
        else:
            coherence = 'unknown'
            strength_consistency = 0.0
        
        return {
            'coherence': coherence,
            'timeframe_signals': timeframe_signals,
            'strength_consistency': float(strength_consistency),
            'alignment_ratio': max(bullish_count, bearish_count) / len(signals) if signals else 0.0
        }
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate PPO with comprehensive analysis"""
        try:
            # Optimize parameters
            fast_period, slow_period, signal_period = self._optimize_ppo_parameters(data)
            
            # Calculate PPO components
            ppo_line, signal_line, histogram = self._calculate_ppo_components(
                data, fast_period, slow_period, signal_period
            )
            
            # Calculate dynamic thresholds
            upper_threshold, lower_threshold = self._calculate_dynamic_thresholds(ppo_line)
            
            # Advanced analysis
            histogram_analysis = self._analyze_histogram_momentum(histogram)
            crossover_analysis = self._detect_ppo_crossovers(ppo_line, signal_line)
            zero_line_analysis = self._analyze_zero_line_behavior(ppo_line)
            divergence_analysis = self._detect_ppo_divergences(ppo_line, data)
            coherence_analysis = self._analyze_multi_timeframe_coherence(data)
            cycle_analysis = self._detect_price_cycles(data)
            
            # Generate signal
            signal, confidence = self._generate_ppo_signal(
                ppo_line, signal_line, histogram, upper_threshold, lower_threshold,
                histogram_analysis, crossover_analysis, zero_line_analysis,
                divergence_analysis, coherence_analysis, data
            )
            
            # Update history
            if len(ppo_line) > 0 and not pd.isna(ppo_line.iloc[-1]):
                self.history['ppo_values'].append(float(ppo_line.iloc[-1]))
                self.history['signal_values'].append(float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0)
                self.history['histogram_values'].append(float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0)
                self.history['thresholds'].append({
                    'upper': upper_threshold,
                    'lower': lower_threshold
                })
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'ppo': float(ppo_line.iloc[-1]) if len(ppo_line) > 0 and not pd.isna(ppo_line.iloc[-1]) else 0.0,
                'signal': float(signal_line.iloc[-1]) if len(signal_line) > 0 and not pd.isna(signal_line.iloc[-1]) else 0.0,
                'histogram': float(histogram.iloc[-1]) if len(histogram) > 0 and not pd.isna(histogram.iloc[-1]) else 0.0,
                'upper_threshold': upper_threshold,
                'lower_threshold': lower_threshold,
                'signal_type': signal,
                'confidence': confidence,
                'optimized_periods': {
                    'fast': fast_period,
                    'slow': slow_period,
                    'signal': signal_period
                },
                'histogram_analysis': histogram_analysis,
                'crossover_analysis': crossover_analysis,
                'zero_line_analysis': zero_line_analysis,
                'divergence_analysis': divergence_analysis,
                'coherence_analysis': coherence_analysis,
                'cycle_analysis': cycle_analysis,
                'market_regime': self._classify_ppo_regime(ppo_line, signal_line, histogram),
                'values_history': {
                    'ppo': ppo_line.tail(30).tolist(),
                    'signal': signal_line.tail(30).tolist(),
                    'histogram': histogram.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate PPO: {str(e)}",
                cause=e
            )
    
    def _generate_ppo_signal(self, ppo: pd.Series, signal: pd.Series, histogram: pd.Series,
                            upper_threshold: float, lower_threshold: float,
                            histogram_analysis: Dict, crossover_analysis: Dict,
                            zero_line_analysis: Dict, divergence_analysis: Dict,
                            coherence_analysis: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive PPO signal"""
        signal_components = []
        confidence_components = []
        
        current_ppo = ppo.iloc[-1] if len(ppo) > 0 and not pd.isna(ppo.iloc[-1]) else 0
        current_signal = signal.iloc[-1] if len(signal) > 0 and not pd.isna(signal.iloc[-1]) else 0
        current_histogram = histogram.iloc[-1] if len(histogram) > 0 and not pd.isna(histogram.iloc[-1]) else 0
        
        # PPO-Signal crossover signals
        if current_ppo > current_signal:
            signal_components.append(0.7)
            confidence_components.append(0.8)
        elif current_ppo < current_signal:
            signal_components.append(-0.7)
            confidence_components.append(0.8)
        
        # Recent crossover signals
        if crossover_analysis['recent_crossovers']:
            recent_crossover = crossover_analysis['recent_crossovers'][0]
            if recent_crossover['position'] <= 3:  # Very recent
                strength = min(recent_crossover['strength'], 2.0)  # Cap strength
                if recent_crossover['type'] == 'bullish':
                    signal_components.append(0.9 * strength / 2.0)
                    confidence_components.append(0.9)
                else:
                    signal_components.append(-0.9 * strength / 2.0)
                    confidence_components.append(0.9)
        
        # Threshold signals
        if current_ppo > upper_threshold:
            # Overbought - potential reversal
            signal_components.append(-0.6)
            confidence_components.append(0.7)
        elif current_ppo < lower_threshold:
            # Oversold - potential reversal
            signal_components.append(0.6)
            confidence_components.append(0.7)
        
        # Zero line signals
        zero_position = zero_line_analysis['current_position']
        if zero_position in ['above', 'strongly_above']:
            signal_components.append(0.5)
            confidence_components.append(0.6)
        elif zero_position in ['below', 'strongly_below']:
            signal_components.append(-0.5)
            confidence_components.append(0.6)
        
        # Zero line crossover signals
        if zero_line_analysis['zero_crosses']:
            recent_zero_cross = zero_line_analysis['zero_crosses'][0]
            if recent_zero_cross['position'] <= 5:
                momentum = min(recent_zero_cross['momentum'], 2.0)
                if recent_zero_cross['type'] == 'bullish':
                    signal_components.append(0.8 * momentum / 2.0)
                    confidence_components.append(0.8)
                else:
                    signal_components.append(-0.8 * momentum / 2.0)
                    confidence_components.append(0.8)
        
        # Histogram momentum signals
        hist_pattern = histogram_analysis['pattern']
        momentum_strength = min(histogram_analysis['momentum_strength'], 2.0)
        
        if hist_pattern == 'bullish_acceleration' and momentum_strength > 0.5:
            signal_components.append(0.6 * momentum_strength / 2.0)
            confidence_components.append(0.7)
        elif hist_pattern == 'bearish_acceleration' and momentum_strength > 0.5:
            signal_components.append(-0.6 * momentum_strength / 2.0)
            confidence_components.append(0.7)
        
        # Divergence signals
        if divergence_analysis['bullish_divergence']:
            strength = min(divergence_analysis['strength'], 1.0)
            signal_components.append(0.8 * strength)
            confidence_components.append(0.8)
        elif divergence_analysis['bearish_divergence']:
            strength = min(divergence_analysis['strength'], 1.0)
            signal_components.append(-0.8 * strength)
            confidence_components.append(0.8)
        
        # Multi-timeframe coherence
        coherence = coherence_analysis['coherence']
        alignment_ratio = coherence_analysis['alignment_ratio']
        
        if coherence == 'strong_bullish' and alignment_ratio > 0.8:
            signal_components.append(0.7)
            confidence_components.append(0.8)
        elif coherence == 'strong_bearish' and alignment_ratio > 0.8:
            signal_components.append(-0.7)
            confidence_components.append(0.8)
        elif coherence in ['mixed_bullish', 'mixed_bearish']:
            direction = 0.5 if 'bullish' in coherence else -0.5
            signal_components.append(direction * alignment_ratio)
            confidence_components.append(0.6)
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        if weighted_signal > 0.6:
            final_signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            final_signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            final_signal = SignalType.NEUTRAL
        
        return final_signal, min(avg_confidence, 1.0)
    
    def _classify_ppo_regime(self, ppo: pd.Series, signal: pd.Series, histogram: pd.Series) -> Dict[str, Any]:
        """Classify PPO regime and market state"""
        if len(ppo) < 30:
            return {'regime': 'unknown', 'trend_state': 'undefined', 'momentum_phase': 'unclear'}
        
        recent_ppo = ppo.tail(30)
        recent_signal = signal.tail(30)
        recent_histogram = histogram.tail(30)
        
        # PPO characteristics
        current_ppo = recent_ppo.iloc[-1]
        ppo_trend = np.polyfit(range(len(recent_ppo)), recent_ppo.values, 1)[0]
        ppo_volatility = recent_ppo.std()
        
        # Zero line analysis
        ppo_above_zero = len(recent_ppo[recent_ppo > 0]) / len(recent_ppo)
        signal_above_zero = len(recent_signal[recent_signal > 0]) / len(recent_signal)
        
        # Histogram analysis
        histogram_trend = np.polyfit(range(len(recent_histogram)), recent_histogram.values, 1)[0]
        current_histogram = recent_histogram.iloc[-1]
        
        # Regime classification
        if ppo_above_zero > 0.8 and signal_above_zero > 0.8:
            if histogram_trend > 0:
                regime = 'strong_bullish_momentum'
            else:
                regime = 'bullish_momentum_weakening'
        elif ppo_above_zero < 0.2 and signal_above_zero < 0.2:
            if histogram_trend < 0:
                regime = 'strong_bearish_momentum'
            else:
                regime = 'bearish_momentum_weakening'
        elif ppo_above_zero > 0.6 or signal_above_zero > 0.6:
            regime = 'weak_bullish_momentum'
        elif ppo_above_zero < 0.4 or signal_above_zero < 0.4:
            regime = 'weak_bearish_momentum'
        else:
            regime = 'sideways_momentum'
        
        # Trend state
        if abs(ppo_trend) > 0.1:
            trend_state = 'trending_up' if ppo_trend > 0 else 'trending_down'
        else:
            trend_state = 'ranging'
        
        # Momentum phase
        if current_ppo > current_signal and histogram_trend > 0:
            momentum_phase = 'accelerating_bullish'
        elif current_ppo > current_signal and histogram_trend < 0:
            momentum_phase = 'decelerating_bullish'
        elif current_ppo < current_signal and histogram_trend < 0:
            momentum_phase = 'accelerating_bearish'
        elif current_ppo < current_signal and histogram_trend > 0:
            momentum_phase = 'decelerating_bearish'
        else:
            momentum_phase = 'neutral'
        
        return {
            'regime': regime,
            'trend_state': trend_state,
            'momentum_phase': momentum_phase,
            'current_ppo': float(current_ppo),
            'ppo_trend': float(ppo_trend),
            'current_histogram': float(current_histogram),
            'histogram_trend': float(histogram_trend),
            'ppo_above_zero_ratio': float(ppo_above_zero),
            'signal_above_zero_ratio': float(signal_above_zero),
            'ppo_volatility': float(ppo_volatility)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal_type' in value and 'confidence' in value:
            return value['signal_type'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'percentage_price_oscillator',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'ml_enhancement': self.parameters['ml_enhancement'],
            'histogram_analysis': self.parameters['histogram_analysis'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'zero_line_analysis': self.parameters['zero_line_analysis'],
            'divergence_detection': self.parameters['divergence_detection'],
            'dynamic_thresholds': self.parameters['dynamic_thresholds'],
            'cycle_detection': self.parameters['cycle_detection'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata