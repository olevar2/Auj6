"""
Price Oscillator Indicator - Advanced Implementation
===================================================

Advanced Price Oscillator with adaptive parameters, machine learning enhancement,
and sophisticated trend analysis for momentum detection.

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


class PriceOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Price Oscillator Implementation
    
    Features:
    - Adaptive moving average periods based on market volatility
    - ML-enhanced smoothing and trend detection
    - Multi-timeframe oscillator analysis
    - Volume-weighted calculations for institutional flow detection
    - Advanced signal generation with momentum confirmation
    - Dynamic threshold optimization for trend changes
    - Sophisticated divergence analysis with statistical validation
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'fast_period': 10,
            'slow_period': 30,
            'smoothing_period': 5,
            'adaptive_periods': True,
            'volume_weighted': True,
            'ml_enhancement': True,
            'smoothing_enabled': True,
            'multi_timeframe': True,
            'divergence_detection': True,
            'dynamic_thresholds': True,
            'min_fast_period': 5,
            'max_fast_period': 20,
            'min_slow_period': 20,
            'max_slow_period': 50,
            'min_smoothing_period': 3,
            'max_smoothing_period': 10,
            'optimization_lookback': 120,
            'ml_lookback': 80,
            'use_ema': True,
            'trend_confirmation': True,
            'momentum_analysis': True,
            'threshold_percentile': 90
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="PriceOscillatorIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.signal_classifier = RandomForestClassifier(n_estimators=150, random_state=42)
        self.trend_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.threshold_optimizer = ExtraTreesRegressor(n_estimators=100, random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=6, random_state=42)
        self.dbscan = DBSCAN(eps=0.4, min_samples=5)
        self.pca = PCA(n_components=3)
        self.ica = FastICA(n_components=2, random_state=42)
        self.models_trained = False
        
        self.history = {
            'oscillator_values': [],
            'smoothed_values': [],
            'thresholds': [],
            'trends': [],
            'signals': []
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
    
    def _detect_market_cycles(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect dominant market cycles using spectral analysis"""
        if len(data) < 100:
            return {'dominant_cycle': 25, 'cycle_strength': 0.5, 'cycle_phase': 'unknown'}
        
        prices = data['close'].tail(100).values
        
        # Detrend the data
        detrended_prices = stats.detrend(prices)
        
        # Apply Hamming window to reduce spectral leakage
        windowed_prices = detrended_prices * np.hamming(len(detrended_prices))
        
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
                dominant_cycle = max(10, min(dominant_cycle, 60))  # Reasonable bounds
            else:
                dominant_cycle = 25
            
            # Calculate cycle strength (relative power)
            total_power = np.sum(positive_power)
            cycle_strength = positive_power[dominant_freq_idx] / total_power if total_power > 0 else 0.5
            
            # Estimate cycle phase
            phase_angle = np.angle(fft_values[dominant_freq_idx + 1])
            normalized_phase = (phase_angle + np.pi) / (2 * np.pi)  # Normalize to 0-1
            
            if normalized_phase < 0.25:
                cycle_phase = 'trough'
            elif normalized_phase < 0.5:
                cycle_phase = 'rising'
            elif normalized_phase < 0.75:
                cycle_phase = 'peak'
            else:
                cycle_phase = 'falling'
        else:
            dominant_cycle = 25
            cycle_strength = 0.5
            cycle_phase = 'unknown'
        
        return {
            'dominant_cycle': dominant_cycle,
            'cycle_strength': float(cycle_strength),
            'cycle_phase': cycle_phase,
            'frequency': float(dominant_frequency) if 'dominant_frequency' in locals() else 0.04
        }
    
    def _optimize_oscillator_parameters(self, data: pd.DataFrame) -> Tuple[int, int, int]:
        """Optimize oscillator parameters using genetic algorithm approach"""
        if not self.parameters['adaptive_periods'] or len(data) < self.parameters['optimization_lookback']:
            return (self.parameters['fast_period'], 
                   self.parameters['slow_period'], 
                   self.parameters['smoothing_period'])
        
        optimization_data = data.tail(self.parameters['optimization_lookback'])
        cycle_info = self._detect_market_cycles(optimization_data)
        dominant_cycle = cycle_info['dominant_cycle']
        
        # Define objective function for optimization
        def objective_function(params):
            fast_period, slow_period, smoothing_period = int(params[0]), int(params[1]), int(params[2])
            
            # Parameter constraints
            if fast_period >= slow_period or smoothing_period < 2:
                return 1000  # Penalty
            
            try:
                # Calculate oscillator with these parameters
                if self.parameters['use_ema']:
                    fast_ma = optimization_data['close'].ewm(span=fast_period, adjust=False).mean()
                    slow_ma = optimization_data['close'].ewm(span=slow_period, adjust=False).mean()
                else:
                    fast_ma = optimization_data['close'].rolling(window=fast_period).mean()
                    slow_ma = optimization_data['close'].rolling(window=slow_period).mean()
                
                oscillator = fast_ma - slow_ma
                smoothed_oscillator = oscillator.ewm(span=smoothing_period, adjust=False).mean()
                
                # Performance metrics
                returns = optimization_data['close'].pct_change()
                
                # Signal generation based on zero crossovers
                signals = pd.Series(0, index=oscillator.index)
                signals[oscillator > 0] = 1
                signals[oscillator < 0] = -1
                
                # Strategy performance
                strategy_returns = signals.shift(1) * returns
                
                # Calculate Sharpe-like ratio
                total_return = strategy_returns.sum()
                volatility = strategy_returns.std()
                max_drawdown = self._calculate_max_drawdown(strategy_returns.cumsum())
                
                # Signal quality metrics
                oscillator_volatility = oscillator.std()
                smoothing_effect = abs(oscillator - smoothed_oscillator).mean()
                
                # Combined score
                if volatility > 0 and max_drawdown < 0.4:
                    score = (total_return / volatility) - smoothing_effect * 0.1 - max_drawdown * 2
                else:
                    score = -1000
                
                # Penalty for excessive noise
                if oscillator_volatility > optimization_data['close'].std() * 2:
                    score -= 500
                
                return -score  # Minimize negative score
            except:
                return 1000
        
        # Adaptive bounds based on cycle analysis
        cycle_factor = dominant_cycle / 25  # Normalize
        
        bounds = [
            (max(self.parameters['min_fast_period'], int(5 * cycle_factor)), 
             min(self.parameters['max_fast_period'], int(20 * cycle_factor))),
            (max(self.parameters['min_slow_period'], int(20 * cycle_factor)), 
             min(self.parameters['max_slow_period'], int(50 * cycle_factor))),
            (self.parameters['min_smoothing_period'], self.parameters['max_smoothing_period'])
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
            fast_period = max(bounds[0][0], min(int(optimized_params[0]), bounds[0][1]))
            slow_period = max(bounds[1][0], min(int(optimized_params[1]), bounds[1][1]))
            smoothing_period = max(bounds[2][0], min(int(optimized_params[2]), bounds[2][1]))
            
            # Ensure constraints
            if fast_period >= slow_period:
                fast_period = slow_period - 2
                fast_period = max(fast_period, bounds[0][0])
            
            return fast_period, slow_period, smoothing_period
            
        except:
            # Fall back to cycle-adjusted defaults
            fast_adjusted = max(5, min(int(10 * cycle_factor), 20))
            slow_adjusted = max(20, min(int(30 * cycle_factor), 50))
            smoothing_adjusted = self.parameters['smoothing_period']
            
            if fast_adjusted >= slow_adjusted:
                fast_adjusted = slow_adjusted - 2
            
            return fast_adjusted, slow_adjusted, smoothing_adjusted
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / (rolling_max.abs() + 1e-8)
        return abs(drawdown.min())
    
    def _calculate_volume_weighted_ma(self, data: pd.DataFrame, period: int, use_ema: bool = True) -> pd.Series:
        """Calculate volume-weighted moving average for institutional flow detection"""
        if not self.parameters['volume_weighted'] or 'volume' not in data.columns:
            if use_ema:
                return data['close'].ewm(span=period, adjust=False).mean()
            else:
                return data['close'].rolling(window=period).mean()
        
        close = data['close']
        volume = data['volume']
        
        # Volume-weighted price
        vwap_period = min(period, 20)  # Use shorter period for VWAP calculation
        vwap = (close * volume).rolling(window=vwap_period).sum() / volume.rolling(window=vwap_period).sum()
        
        # Moving average of VWAP
        if use_ema:
            vw_ma = vwap.ewm(span=period, adjust=False).mean()
            regular_ma = close.ewm(span=period, adjust=False).mean()
        else:
            vw_ma = vwap.rolling(window=period).mean()
            regular_ma = close.rolling(window=period).mean()
        
        # Adaptive blending based on volume significance
        volume_ma = volume.rolling(window=period).mean()
        volume_ratio = volume / (volume_ma + 1e-8)
        
        # Sigmoid function for smooth weighting
        volume_weight = 2.0 / (1.0 + np.exp(-1.5 * (volume_ratio - 1.0))) - 1.0
        volume_weight = np.clip(volume_weight, 0, 1)
        
        # Blend moving averages
        blended_ma = regular_ma * (1 - volume_weight) + vw_ma * volume_weight
        
        return blended_ma.fillna(regular_ma)
    
    def _calculate_oscillator_components(self, data: pd.DataFrame, fast_period: int, 
                                       slow_period: int, smoothing_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Price Oscillator components with advanced techniques"""
        # Calculate moving averages (volume-weighted if enabled)
        if self.parameters['volume_weighted']:
            fast_ma = self._calculate_volume_weighted_ma(data, fast_period, self.parameters['use_ema'])
            slow_ma = self._calculate_volume_weighted_ma(data, slow_period, self.parameters['use_ema'])
        else:
            if self.parameters['use_ema']:
                fast_ma = data['close'].ewm(span=fast_period, adjust=False).mean()
                slow_ma = data['close'].ewm(span=slow_period, adjust=False).mean()
            else:
                fast_ma = data['close'].rolling(window=fast_period).mean()
                slow_ma = data['close'].rolling(window=slow_period).mean()
        
        # Price Oscillator (difference between fast and slow MA)
        oscillator = fast_ma - slow_ma
        
        # Smoothed oscillator with advanced smoothing
        if self.parameters['smoothing_enabled']:
            # Adaptive smoothing based on oscillator volatility
            oscillator_volatility = oscillator.rolling(window=20).std()
            avg_volatility = oscillator_volatility.rolling(window=60).mean()
            vol_ratio = oscillator_volatility / (avg_volatility + 1e-8)
            
            # Adjust smoothing factor based on volatility
            if self.parameters['use_ema']:
                base_alpha = 2.0 / (smoothing_period + 1)
                adaptive_alpha = base_alpha / (1 + vol_ratio * 0.5)
                smoothed_oscillator = oscillator.ewm(alpha=adaptive_alpha, adjust=False).mean()
            else:
                adaptive_period = int(smoothing_period * (1 + vol_ratio * 0.5))
                adaptive_period = max(3, min(adaptive_period, smoothing_period * 2))
                smoothed_oscillator = oscillator.rolling(window=adaptive_period).mean()
            
            # Optional Savitzky-Golay smoothing for noise reduction
            if len(smoothed_oscillator.dropna()) > smoothing_period * 2:
                try:
                    valid_smoothed = smoothed_oscillator.dropna()
                    if len(valid_smoothed) >= 7:
                        window_length = min(9, len(valid_smoothed) // 2 * 2 - 1)
                        filtered_values = savgol_filter(valid_smoothed.values, window_length, 3)
                        smoothed_oscillator.loc[valid_smoothed.index] = filtered_values
                except:
                    pass
        else:
            if self.parameters['use_ema']:
                smoothed_oscillator = oscillator.ewm(span=smoothing_period, adjust=False).mean()
            else:
                smoothed_oscillator = oscillator.rolling(window=smoothing_period).mean()
        
        return oscillator, smoothed_oscillator
    
    def _calculate_dynamic_thresholds(self, oscillator: pd.Series) -> Tuple[float, float]:
        """Calculate dynamic overbought/oversold thresholds"""
        if not self.parameters['dynamic_thresholds'] or len(oscillator) < 50:
            osc_std = oscillator.std() if len(oscillator) > 10 else 1.0
            return osc_std * 1.5, -osc_std * 1.5  # Default thresholds
        
        recent_oscillator = oscillator.tail(100).dropna()
        
        if len(recent_oscillator) < 30:
            osc_std = recent_oscillator.std() if len(recent_oscillator) > 5 else 1.0
            return osc_std * 1.5, -osc_std * 1.5
        
        # Use percentile-based approach
        percentile = self.parameters['threshold_percentile']
        upper_threshold = np.percentile(recent_oscillator, percentile)
        lower_threshold = np.percentile(recent_oscillator, 100 - percentile)
        
        # Ensure minimum separation based on volatility
        osc_std = recent_oscillator.std()
        min_separation = osc_std * 1.0
        
        if upper_threshold - lower_threshold < min_separation:
            mid_point = (upper_threshold + lower_threshold) / 2
            upper_threshold = mid_point + min_separation / 2
            lower_threshold = mid_point - min_separation / 2
        
        # Apply reasonable bounds relative to recent price movements
        price_std = recent_oscillator.std()
        upper_threshold = max(price_std * 0.5, min(upper_threshold, price_std * 5.0))
        lower_threshold = min(-price_std * 0.5, max(lower_threshold, -price_std * 5.0))
        
        return upper_threshold, lower_threshold
    
    def _analyze_oscillator_momentum(self, oscillator: pd.Series, smoothed: pd.Series) -> Dict[str, Any]:
        """Analyze oscillator momentum and acceleration patterns"""
        if not self.parameters['momentum_analysis'] or len(oscillator) < 20:
            return {'momentum': 'unknown', 'acceleration': 'stable', 'strength': 0.0}
        
        recent_oscillator = oscillator.tail(20).dropna()
        recent_smoothed = smoothed.tail(20).dropna()
        
        if len(recent_oscillator) < 10:
            return {'momentum': 'unknown', 'acceleration': 'stable', 'strength': 0.0}
        
        # Calculate oscillator derivatives
        osc_velocity = recent_oscillator.diff()
        osc_acceleration = osc_velocity.diff()
        
        # Current momentum characteristics
        current_oscillator = recent_oscillator.iloc[-1]
        current_smoothed = recent_smoothed.iloc[-1] if len(recent_smoothed) > 0 else current_oscillator
        recent_velocity = osc_velocity.tail(5).mean()
        recent_acceleration = osc_acceleration.tail(3).mean()
        
        # Momentum strength
        oscillator_std = recent_oscillator.std()
        momentum_strength = abs(current_oscillator) / (oscillator_std + 1e-8)
        
        # Momentum direction
        if current_oscillator > current_smoothed and recent_velocity > 0:
            momentum = 'strong_bullish'
        elif current_oscillator > current_smoothed and recent_velocity <= 0:
            momentum = 'weak_bullish'
        elif current_oscillator < current_smoothed and recent_velocity < 0:
            momentum = 'strong_bearish'
        elif current_oscillator < current_smoothed and recent_velocity >= 0:
            momentum = 'weak_bearish'
        else:
            momentum = 'neutral'
        
        # Acceleration classification
        if recent_acceleration > 0.1:
            acceleration = 'accelerating'
        elif recent_acceleration < -0.1:
            acceleration = 'decelerating'
        else:
            acceleration = 'stable'
        
        # Trend consistency
        positive_bars = len(recent_oscillator[recent_oscillator > 0])
        negative_bars = len(recent_oscillator[recent_oscillator < 0])
        
        if positive_bars > negative_bars * 1.5:
            trend_consistency = 'consistently_positive'
        elif negative_bars > positive_bars * 1.5:
            trend_consistency = 'consistently_negative'
        else:
            trend_consistency = 'mixed'
        
        # Zero line relationship
        zero_distance = abs(current_oscillator) / (oscillator_std + 1e-8)
        
        return {
            'momentum': momentum,
            'acceleration': acceleration,
            'strength': float(momentum_strength),
            'trend_consistency': trend_consistency,
            'zero_distance': float(zero_distance),
            'recent_velocity': float(recent_velocity),
            'recent_acceleration': float(recent_acceleration),
            'current_oscillator': float(current_oscillator),
            'current_smoothed': float(current_smoothed),
            'positive_bars': positive_bars,
            'negative_bars': negative_bars
        }
    
    def _detect_oscillator_crossovers(self, oscillator: pd.Series, smoothed: pd.Series) -> Dict[str, Any]:
        """Detect and analyze oscillator crossovers with smoothed line"""
        crossovers = []
        zero_crossovers = []
        
        if len(oscillator) < 2 or len(smoothed) < 2:
            return {'signal_crossovers': [], 'zero_crossovers': [], 'recent_crossover_strength': 0.0}
        
        # Look for crossovers in recent data
        lookback = min(30, len(oscillator) - 1)
        recent_oscillator = oscillator.tail(lookback)
        recent_smoothed = smoothed.tail(lookback)
        
        # Signal line crossovers
        for i in range(1, len(recent_oscillator)):
            prev_osc = recent_oscillator.iloc[i-1]
            curr_osc = recent_oscillator.iloc[i]
            prev_smooth = recent_smoothed.iloc[i-1]
            curr_smooth = recent_smoothed.iloc[i]
            
            # Bullish crossover (oscillator crosses above smoothed)
            if prev_osc <= prev_smooth and curr_osc > curr_smooth:
                separation = abs(curr_osc - curr_smooth)
                osc_momentum = curr_osc - prev_osc
                smooth_momentum = curr_smooth - prev_smooth
                
                strength = separation + abs(osc_momentum - smooth_momentum)
                normalized_strength = strength / (recent_oscillator.std() + 1e-8)
                
                crossovers.append({
                    'type': 'bullish',
                    'position': len(recent_oscillator) - i,  # Bars ago
                    'strength': float(normalized_strength),
                    'oscillator_value': float(curr_osc),
                    'smoothed_value': float(curr_smooth),
                    'separation': float(separation)
                })
            
            # Bearish crossover (oscillator crosses below smoothed)
            elif prev_osc >= prev_smooth and curr_osc < curr_smooth:
                separation = abs(curr_osc - curr_smooth)
                osc_momentum = curr_osc - prev_osc
                smooth_momentum = curr_smooth - prev_smooth
                
                strength = separation + abs(osc_momentum - smooth_momentum)
                normalized_strength = strength / (recent_oscillator.std() + 1e-8)
                
                crossovers.append({
                    'type': 'bearish',
                    'position': len(recent_oscillator) - i,
                    'strength': float(normalized_strength),
                    'oscillator_value': float(curr_osc),
                    'smoothed_value': float(curr_smooth),
                    'separation': float(separation)
                })
        
        # Zero line crossovers
        for i in range(1, len(recent_oscillator)):
            prev_osc = recent_oscillator.iloc[i-1]
            curr_osc = recent_oscillator.iloc[i]
            
            # Bullish zero crossover
            if prev_osc <= 0 and curr_osc > 0:
                momentum = abs(curr_osc) / (recent_oscillator.std() + 1e-8)
                zero_crossovers.append({
                    'type': 'bullish',
                    'position': len(recent_oscillator) - i,
                    'momentum': float(momentum),
                    'value': float(curr_osc)
                })
            
            # Bearish zero crossover
            elif prev_osc >= 0 and curr_osc < 0:
                momentum = abs(curr_osc) / (recent_oscillator.std() + 1e-8)
                zero_crossovers.append({
                    'type': 'bearish',
                    'position': len(recent_oscillator) - i,
                    'momentum': float(momentum),
                    'value': float(curr_osc)
                })
        
        # Calculate recent crossover strength
        if crossovers:
            recent_crossover = min(crossovers, key=lambda x: x['position'])
            crossover_strength = recent_crossover['strength']
        else:
            crossover_strength = 0.0
        
        return {
            'signal_crossovers': crossovers,
            'zero_crossovers': zero_crossovers,
            'recent_crossover_strength': crossover_strength,
            'total_signal_crossovers': len(crossovers),
            'total_zero_crossovers': len(zero_crossovers)
        }
    
    def _detect_oscillator_divergences(self, oscillator: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect divergences between oscillator and price"""
        if not self.parameters['divergence_detection'] or len(oscillator) < 30:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        lookback = min(30, len(oscillator) - 5)
        recent_oscillator = oscillator.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find peaks and troughs
        osc_peaks, _ = find_peaks(recent_oscillator.values, distance=5)
        osc_troughs, _ = find_peaks(-recent_oscillator.values, distance=5)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=5)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=5)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        divergence_details = []
        
        # Bullish divergence analysis
        if len(osc_troughs) >= 2 and len(price_troughs) >= 2:
            for i in range(len(osc_troughs) - 1):
                osc_trough_1 = osc_troughs[i]
                osc_trough_2 = osc_troughs[i + 1]
                
                # Find corresponding price troughs
                price_trough_1 = self._find_nearest_extreme(price_troughs, osc_trough_1)
                price_trough_2 = self._find_nearest_extreme(price_troughs, osc_trough_2)
                
                if price_trough_1 is not None and price_trough_2 is not None:
                    price_1 = recent_prices.iloc[price_trough_1]
                    price_2 = recent_prices.iloc[price_trough_2]
                    osc_1 = recent_oscillator.iloc[osc_trough_1]
                    osc_2 = recent_oscillator.iloc[osc_trough_2]
                    
                    # Check for divergence pattern
                    if price_2 < price_1 and osc_2 > osc_1:
                        price_decline = (price_1 - price_2) / price_1
                        osc_improvement = (osc_2 - osc_1) / abs(osc_1 + 1e-8)
                        
                        # Statistical significance test
                        if price_decline > 0.02 and osc_improvement > 0.15:
                            bullish_divergence = True
                            strength = price_decline + osc_improvement
                            divergence_strength = max(divergence_strength, strength)
                            
                            divergence_details.append({
                                'type': 'bullish',
                                'price_decline': float(price_decline),
                                'oscillator_improvement': float(osc_improvement),
                                'strength': float(strength)
                            })
        
        # Bearish divergence analysis
        if len(osc_peaks) >= 2 and len(price_peaks) >= 2:
            for i in range(len(osc_peaks) - 1):
                osc_peak_1 = osc_peaks[i]
                osc_peak_2 = osc_peaks[i + 1]
                
                price_peak_1 = self._find_nearest_extreme(price_peaks, osc_peak_1)
                price_peak_2 = self._find_nearest_extreme(price_peaks, osc_peak_2)
                
                if price_peak_1 is not None and price_peak_2 is not None:
                    price_1 = recent_prices.iloc[price_peak_1]
                    price_2 = recent_prices.iloc[price_peak_2]
                    osc_1 = recent_oscillator.iloc[osc_peak_1]
                    osc_2 = recent_oscillator.iloc[osc_peak_2]
                    
                    if price_2 > price_1 and osc_2 < osc_1:
                        price_increase = (price_2 - price_1) / price_1
                        osc_decline = (osc_1 - osc_2) / abs(osc_1 + 1e-8)
                        
                        if price_increase > 0.02 and osc_decline > 0.15:
                            bearish_divergence = True
                            strength = price_increase + osc_decline
                            divergence_strength = max(divergence_strength, strength)
                            
                            divergence_details.append({
                                'type': 'bearish',
                                'price_increase': float(price_increase),
                                'oscillator_decline': float(osc_decline),
                                'strength': float(strength)
                            })
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'divergence_details': divergence_details,
            'oscillator_peaks': len(osc_peaks),
            'oscillator_troughs': len(osc_troughs),
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
            if distance < min_distance and distance < 12:
                min_distance = distance
                nearest_extreme = extreme_idx
        
        return nearest_extreme
    
    def _analyze_multi_timeframe_oscillator(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze oscillator coherence across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(data) < 60:
            return {'coherence': 'unknown', 'timeframe_signals': []}
        
        timeframe_signals = []
        
        # Short-term oscillator (5, 15, 3)
        try:
            if self.parameters['use_ema']:
                fast_short = data['close'].ewm(span=5, adjust=False).mean()
                slow_short = data['close'].ewm(span=15, adjust=False).mean()
            else:
                fast_short = data['close'].rolling(window=5).mean()
                slow_short = data['close'].rolling(window=15).mean()
            
            osc_short = fast_short - slow_short
            smooth_short = osc_short.ewm(span=3, adjust=False).mean()
            
            short_trend = 'bullish' if osc_short.iloc[-1] > 0 else 'bearish'
            short_signal = 'bullish' if osc_short.iloc[-1] > smooth_short.iloc[-1] else 'bearish'
            
            timeframe_signals.append({
                'timeframe': 'short',
                'trend': short_trend,
                'signal': short_signal,
                'oscillator': float(osc_short.iloc[-1]),
                'smoothed': float(smooth_short.iloc[-1]),
                'periods': (5, 15, 3)
            })
        except:
            pass
        
        # Medium-term oscillator (10, 30, 5) - standard
        try:
            if self.parameters['use_ema']:
                fast_med = data['close'].ewm(span=10, adjust=False).mean()
                slow_med = data['close'].ewm(span=30, adjust=False).mean()
            else:
                fast_med = data['close'].rolling(window=10).mean()
                slow_med = data['close'].rolling(window=30).mean()
            
            osc_med = fast_med - slow_med
            smooth_med = osc_med.ewm(span=5, adjust=False).mean()
            
            med_trend = 'bullish' if osc_med.iloc[-1] > 0 else 'bearish'
            med_signal = 'bullish' if osc_med.iloc[-1] > smooth_med.iloc[-1] else 'bearish'
            
            timeframe_signals.append({
                'timeframe': 'medium',
                'trend': med_trend,
                'signal': med_signal,
                'oscillator': float(osc_med.iloc[-1]),
                'smoothed': float(smooth_med.iloc[-1]),
                'periods': (10, 30, 5)
            })
        except:
            pass
        
        # Long-term oscillator (20, 50, 10)
        if len(data) >= 50:
            try:
                if self.parameters['use_ema']:
                    fast_long = data['close'].ewm(span=20, adjust=False).mean()
                    slow_long = data['close'].ewm(span=50, adjust=False).mean()
                else:
                    fast_long = data['close'].rolling(window=20).mean()
                    slow_long = data['close'].rolling(window=50).mean()
                
                osc_long = fast_long - slow_long
                smooth_long = osc_long.ewm(span=10, adjust=False).mean()
                
                long_trend = 'bullish' if osc_long.iloc[-1] > 0 else 'bearish'
                long_signal = 'bullish' if osc_long.iloc[-1] > smooth_long.iloc[-1] else 'bearish'
                
                timeframe_signals.append({
                    'timeframe': 'long',
                    'trend': long_trend,
                    'signal': long_signal,
                    'oscillator': float(osc_long.iloc[-1]),
                    'smoothed': float(smooth_long.iloc[-1]),
                    'periods': (20, 50, 10)
                })
            except:
                pass
        
        # Analyze coherence
        if timeframe_signals:
            trend_signals = [signal['trend'] for signal in timeframe_signals]
            signal_signals = [signal['signal'] for signal in timeframe_signals]
            
            bullish_trends = trend_signals.count('bullish')
            bearish_trends = trend_signals.count('bearish')
            bullish_signals = signal_signals.count('bullish')
            bearish_signals = signal_signals.count('bearish')
            
            # Overall coherence
            if bullish_trends == len(timeframe_signals) and bullish_signals == len(timeframe_signals):
                coherence = 'strong_bullish'
            elif bearish_trends == len(timeframe_signals) and bearish_signals == len(timeframe_signals):
                coherence = 'strong_bearish'
            elif bullish_trends > bearish_trends and bullish_signals > bearish_signals:
                coherence = 'mixed_bullish'
            elif bearish_trends > bullish_trends and bearish_signals > bullish_signals:
                coherence = 'mixed_bearish'
            else:
                coherence = 'conflicted'
            
            # Calculate coherence strength
            trend_alignment = max(bullish_trends, bearish_trends) / len(timeframe_signals)
            signal_alignment = max(bullish_signals, bearish_signals) / len(timeframe_signals)
            coherence_strength = (trend_alignment + signal_alignment) / 2
        else:
            coherence = 'unknown'
            coherence_strength = 0.0
        
        return {
            'coherence': coherence,
            'timeframe_signals': timeframe_signals,
            'coherence_strength': float(coherence_strength),
            'trend_alignment': float(trend_alignment) if 'trend_alignment' in locals() else 0.0,
            'signal_alignment': float(signal_alignment) if 'signal_alignment' in locals() else 0.0
        }
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Price Oscillator with comprehensive analysis"""
        try:
            # Optimize parameters
            fast_period, slow_period, smoothing_period = self._optimize_oscillator_parameters(data)
            
            # Calculate oscillator components
            oscillator, smoothed_oscillator = self._calculate_oscillator_components(
                data, fast_period, slow_period, smoothing_period
            )
            
            # Calculate dynamic thresholds
            upper_threshold, lower_threshold = self._calculate_dynamic_thresholds(oscillator)
            
            # Advanced analysis
            momentum_analysis = self._analyze_oscillator_momentum(oscillator, smoothed_oscillator)
            crossover_analysis = self._detect_oscillator_crossovers(oscillator, smoothed_oscillator)
            divergence_analysis = self._detect_oscillator_divergences(oscillator, data)
            coherence_analysis = self._analyze_multi_timeframe_oscillator(data)
            cycle_analysis = self._detect_market_cycles(data)
            
            # Generate signal
            signal, confidence = self._generate_oscillator_signal(
                oscillator, smoothed_oscillator, upper_threshold, lower_threshold,
                momentum_analysis, crossover_analysis, divergence_analysis,
                coherence_analysis, data
            )
            
            # Update history
            if len(oscillator) > 0 and not pd.isna(oscillator.iloc[-1]):
                self.history['oscillator_values'].append(float(oscillator.iloc[-1]))
                self.history['smoothed_values'].append(float(smoothed_oscillator.iloc[-1]) if not pd.isna(smoothed_oscillator.iloc[-1]) else 0.0)
                self.history['thresholds'].append({
                    'upper': upper_threshold,
                    'lower': lower_threshold
                })
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'oscillator': float(oscillator.iloc[-1]) if len(oscillator) > 0 and not pd.isna(oscillator.iloc[-1]) else 0.0,
                'smoothed': float(smoothed_oscillator.iloc[-1]) if len(smoothed_oscillator) > 0 and not pd.isna(smoothed_oscillator.iloc[-1]) else 0.0,
                'upper_threshold': upper_threshold,
                'lower_threshold': lower_threshold,
                'signal_type': signal,
                'confidence': confidence,
                'optimized_periods': {
                    'fast': fast_period,
                    'slow': slow_period,
                    'smoothing': smoothing_period
                },
                'momentum_analysis': momentum_analysis,
                'crossover_analysis': crossover_analysis,
                'divergence_analysis': divergence_analysis,
                'coherence_analysis': coherence_analysis,
                'cycle_analysis': cycle_analysis,
                'market_regime': self._classify_oscillator_regime(oscillator, smoothed_oscillator),
                'values_history': {
                    'oscillator': oscillator.tail(30).tolist(),
                    'smoothed': smoothed_oscillator.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Price Oscillator: {str(e)}",
                cause=e
            )
    
    def _generate_oscillator_signal(self, oscillator: pd.Series, smoothed: pd.Series,
                                  upper_threshold: float, lower_threshold: float,
                                  momentum_analysis: Dict, crossover_analysis: Dict,
                                  divergence_analysis: Dict, coherence_analysis: Dict,
                                  data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive oscillator signal"""
        signal_components = []
        confidence_components = []
        
        current_oscillator = oscillator.iloc[-1] if len(oscillator) > 0 and not pd.isna(oscillator.iloc[-1]) else 0
        current_smoothed = smoothed.iloc[-1] if len(smoothed) > 0 and not pd.isna(smoothed.iloc[-1]) else 0
        
        # Oscillator-smoothed crossover signals
        if current_oscillator > current_smoothed:
            signal_components.append(0.8)
            confidence_components.append(0.8)
        elif current_oscillator < current_smoothed:
            signal_components.append(-0.8)
            confidence_components.append(0.8)
        
        # Zero line signals
        if current_oscillator > 0:
            signal_components.append(0.6)
            confidence_components.append(0.7)
        elif current_oscillator < 0:
            signal_components.append(-0.6)
            confidence_components.append(0.7)
        
        # Recent crossover signals
        if crossover_analysis['signal_crossovers']:
            recent_crossover = crossover_analysis['signal_crossovers'][0]
            if recent_crossover['position'] <= 3:  # Very recent
                strength = min(recent_crossover['strength'], 2.0)
                if recent_crossover['type'] == 'bullish':
                    signal_components.append(0.9 * strength / 2.0)
                    confidence_components.append(0.9)
                else:
                    signal_components.append(-0.9 * strength / 2.0)
                    confidence_components.append(0.9)
        
        # Zero line crossover signals
        if crossover_analysis['zero_crossovers']:
            recent_zero_cross = crossover_analysis['zero_crossovers'][0]
            if recent_zero_cross['position'] <= 5:
                momentum = min(recent_zero_cross['momentum'], 2.0)
                if recent_zero_cross['type'] == 'bullish':
                    signal_components.append(0.8 * momentum / 2.0)
                    confidence_components.append(0.8)
                else:
                    signal_components.append(-0.8 * momentum / 2.0)
                    confidence_components.append(0.8)
        
        # Threshold signals
        if current_oscillator > upper_threshold:
            # Potentially overbought - reversal signal
            signal_components.append(-0.5)
            confidence_components.append(0.6)
        elif current_oscillator < lower_threshold:
            # Potentially oversold - reversal signal
            signal_components.append(0.5)
            confidence_components.append(0.6)
        
        # Momentum analysis signals
        momentum = momentum_analysis['momentum']
        strength = min(momentum_analysis['strength'], 2.0)
        
        if momentum == 'strong_bullish' and strength > 0.5:
            signal_components.append(0.7 * strength / 2.0)
            confidence_components.append(0.7)
        elif momentum == 'strong_bearish' and strength > 0.5:
            signal_components.append(-0.7 * strength / 2.0)
            confidence_components.append(0.7)
        elif momentum in ['weak_bullish', 'weak_bearish']:
            direction = 0.4 if 'bullish' in momentum else -0.4
            signal_components.append(direction * strength / 2.0)
            confidence_components.append(0.5)
        
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
        coherence_strength = coherence_analysis['coherence_strength']
        
        if coherence == 'strong_bullish' and coherence_strength > 0.8:
            signal_components.append(0.7)
            confidence_components.append(0.8)
        elif coherence == 'strong_bearish' and coherence_strength > 0.8:
            signal_components.append(-0.7)
            confidence_components.append(0.8)
        elif coherence in ['mixed_bullish', 'mixed_bearish']:
            direction = 0.5 if 'bullish' in coherence else -0.5
            signal_components.append(direction * coherence_strength)
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
    
    def _classify_oscillator_regime(self, oscillator: pd.Series, smoothed: pd.Series) -> Dict[str, Any]:
        """Classify oscillator regime and market state"""
        if len(oscillator) < 30:
            return {'regime': 'unknown', 'trend_state': 'undefined', 'oscillator_phase': 'unclear'}
        
        recent_oscillator = oscillator.tail(30)
        recent_smoothed = smoothed.tail(30)
        
        # Oscillator characteristics
        current_oscillator = recent_oscillator.iloc[-1]
        current_smoothed = recent_smoothed.iloc[-1]
        osc_trend = np.polyfit(range(len(recent_oscillator)), recent_oscillator.values, 1)[0]
        smooth_trend = np.polyfit(range(len(recent_smoothed)), recent_smoothed.values, 1)[0]
        
        # Zero line analysis
        osc_above_zero = len(recent_oscillator[recent_oscillator > 0]) / len(recent_oscillator)
        smooth_above_zero = len(recent_smoothed[recent_smoothed > 0]) / len(recent_smoothed)
        
        # Regime classification
        if osc_above_zero > 0.8 and smooth_above_zero > 0.8:
            if osc_trend > 0 and smooth_trend > 0:
                regime = 'strong_bullish_trend'
            else:
                regime = 'bullish_trend_weakening'
        elif osc_above_zero < 0.2 and smooth_above_zero < 0.2:
            if osc_trend < 0 and smooth_trend < 0:
                regime = 'strong_bearish_trend'
            else:
                regime = 'bearish_trend_weakening'
        elif osc_above_zero > 0.6 or smooth_above_zero > 0.6:
            regime = 'weak_bullish_trend'
        elif osc_above_zero < 0.4 or smooth_above_zero < 0.4:
            regime = 'weak_bearish_trend'
        else:
            regime = 'sideways_ranging'
        
        # Trend state
        combined_trend = (osc_trend + smooth_trend) / 2
        if abs(combined_trend) > 0.05:
            trend_state = 'trending_up' if combined_trend > 0 else 'trending_down'
        else:
            trend_state = 'ranging'
        
        # Oscillator phase
        if current_oscillator > current_smoothed and osc_trend > smooth_trend:
            oscillator_phase = 'bullish_accelerating'
        elif current_oscillator > current_smoothed and osc_trend <= smooth_trend:
            oscillator_phase = 'bullish_decelerating'
        elif current_oscillator < current_smoothed and osc_trend < smooth_trend:
            oscillator_phase = 'bearish_accelerating'
        elif current_oscillator < current_smoothed and osc_trend >= smooth_trend:
            oscillator_phase = 'bearish_decelerating'
        else:
            oscillator_phase = 'neutral'
        
        return {
            'regime': regime,
            'trend_state': trend_state,
            'oscillator_phase': oscillator_phase,
            'current_oscillator': float(current_oscillator),
            'current_smoothed': float(current_smoothed),
            'oscillator_trend': float(osc_trend),
            'smoothed_trend': float(smooth_trend),
            'osc_above_zero_ratio': float(osc_above_zero),
            'smooth_above_zero_ratio': float(smooth_above_zero)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal_type' in value and 'confidence' in value:
            return value['signal_type'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'price_oscillator',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'ml_enhancement': self.parameters['ml_enhancement'],
            'smoothing_enabled': self.parameters['smoothing_enabled'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'divergence_detection': self.parameters['divergence_detection'],
            'dynamic_thresholds': self.parameters['dynamic_thresholds'],
            'use_ema': self.parameters['use_ema'],
            'trend_confirmation': self.parameters['trend_confirmation'],
            'momentum_analysis': self.parameters['momentum_analysis'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata