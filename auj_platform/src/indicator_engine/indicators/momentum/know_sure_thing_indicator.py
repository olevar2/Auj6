"""
Know Sure Thing (KST) Indicator - Advanced Implementation
========================================================

Advanced Know Sure Thing oscillator with ML-enhanced trend detection,
adaptive smoothing periods, and sophisticated momentum convergence analysis.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from scipy import stats
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class KnowSureThingIndicator(StandardIndicatorInterface):
    """
    Advanced Know Sure Thing (KST) Implementation
    
    Features:
    - Multi-timeframe Rate of Change calculations with adaptive periods
    - ML-enhanced momentum convergence detection
    - Independent Component Analysis for signal separation
    - Advanced signal line integration with optimal smoothing
    - Institutional flow detection through volume weighting
    - Dynamic threshold optimization for changing market regimes
    - Sophisticated divergence analysis with statistical validation
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'roc_periods': [10, 15, 20, 30],  # Four ROC periods
            'sma_periods': [10, 10, 10, 15],  # Smoothing for each ROC
            'weights': [1, 2, 3, 4],          # Weights for each component
            'signal_period': 9,
            'adaptive_periods': True,
            'volume_weighted': True,
            'use_ica': True,
            'ml_lookback': 60,
            'divergence_lookback': 25,
            'optimization_enabled': True,
            'savgol_smoothing': True,
            'multi_timeframe': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="KnowSureThingIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.trend_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.momentum_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=4, random_state=42)
        self.ica = FastICA(n_components=2, random_state=42)
        self.models_trained = False
        
        self.history = {
            'kst_values': [],
            'signal_values': [],
            'roc_components': [],
            'smoothed_rocs': [],
            'ica_components': [],
            'divergence_signals': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(max(self.parameters['roc_periods']), 
                        max(self.parameters['sma_periods']),
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max_period * 2 + 50,
            lookback_periods=200
        )
    
    def _adapt_periods(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Adapt ROC and smoothing periods based on market conditions"""
        if not self.parameters['adaptive_periods']:
            return self.parameters['roc_periods'], self.parameters['sma_periods']
        
        if len(data) < 60:
            return self.parameters['roc_periods'], self.parameters['sma_periods']
        
        # Calculate market characteristics
        returns = data['close'].pct_change().tail(60)
        current_vol = returns.std()
        rolling_vol = returns.rolling(window=20).std()
        avg_vol = rolling_vol.mean()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        # Calculate trend persistence
        price_changes = data['close'].diff().tail(40)
        trend_strength = abs(price_changes.sum()) / (price_changes.abs().sum() + 1e-8)
        
        # Calculate cycle characteristics using FFT
        if len(data) >= 100:
            prices = data['close'].tail(100).values
            fft_values = np.fft.fft(prices)
            power_spectrum = np.abs(fft_values) ** 2
            frequencies = np.fft.fftfreq(len(prices))
            
            # Find dominant frequency (cycle length)
            positive_freqs = frequencies[:len(frequencies)//2]
            positive_power = power_spectrum[:len(power_spectrum)//2]
            
            if len(positive_power) > 1:
                dominant_freq_idx = np.argmax(positive_power[1:]) + 1  # Skip DC component
                dominant_period = int(1.0 / positive_freqs[dominant_freq_idx]) if positive_freqs[dominant_freq_idx] > 0 else 20
                dominant_period = max(10, min(dominant_period, 50))  # Reasonable bounds
            else:
                dominant_period = 20
        else:
            dominant_period = 20
        
        base_roc = self.parameters['roc_periods']
        base_sma = self.parameters['sma_periods']
        
        # Adjust periods based on volatility
        if vol_ratio > 1.5:  # High volatility - shorter periods
            vol_factor = 0.7
        elif vol_ratio < 0.6:  # Low volatility - longer periods
            vol_factor = 1.3
        else:
            vol_factor = 1.0
        
        # Adjust based on trend strength
        if trend_strength > 0.7:  # Strong trend - longer periods
            trend_factor = 1.2
        elif trend_strength < 0.3:  # Weak trend - shorter periods
            trend_factor = 0.8
        else:
            trend_factor = 1.0
        
        # Adjust based on dominant cycle
        cycle_factor = dominant_period / 20  # Normalize to base period
        
        # Combine factors
        adjustment_factor = vol_factor * trend_factor * cycle_factor
        adjustment_factor = max(0.5, min(adjustment_factor, 2.0))  # Reasonable bounds
        
        # Apply adjustments
        adapted_roc = [max(5, min(int(period * adjustment_factor), 60)) for period in base_roc]
        adapted_sma = [max(3, min(int(period * adjustment_factor), 30)) for period in base_sma]
        
        return adapted_roc, adapted_sma
    
    def _calculate_volume_weighted_roc(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate volume-weighted Rate of Change"""
        close = data['close']
        
        if not self.parameters['volume_weighted'] or 'volume' not in data.columns:
            return ((close - close.shift(period)) / close.shift(period)) * 100
        
        volume = data['volume']
        
        # Volume-weighted price calculation
        vwap = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # Calculate ROC on VWAP
        vw_roc = ((vwap - vwap.shift(period)) / vwap.shift(period)) * 100
        
        # Blend with regular ROC based on volume significance
        regular_roc = ((close - close.shift(period)) / close.shift(period)) * 100
        volume_ratio = volume / volume.rolling(window=period).mean()
        volume_weight = np.tanh((volume_ratio - 1) * 0.5)  # Sigmoid weighting
        
        # Higher volume = more weight on VWAP ROC
        weighted_roc = regular_roc * (1 - volume_weight) + vw_roc * volume_weight
        
        return weighted_roc.fillna(0)
    
    def _calculate_smoothed_roc(self, roc: pd.Series, sma_period: int) -> pd.Series:
        """Calculate smoothed ROC with optional Savitzky-Golay filter"""
        # Basic SMA smoothing
        smoothed = roc.rolling(window=sma_period).mean()
        
        # Additional Savitzky-Golay smoothing if enabled
        if self.parameters['savgol_smoothing'] and len(smoothed.dropna()) > sma_period * 2:
            try:
                valid_data = smoothed.dropna()
                if len(valid_data) >= 7:  # Minimum for savgol
                    window_length = min(7, len(valid_data) // 2 * 2 - 1)  # Ensure odd number
                    savgol_smoothed = savgol_filter(valid_data.values, window_length, 3)
                    
                    # Replace the valid data portion
                    smoothed_copy = smoothed.copy()
                    smoothed_copy.loc[valid_data.index] = savgol_smoothed
                    smoothed = smoothed_copy
            except:
                pass  # Fall back to regular SMA if Savitzky-Golay fails
        
        return smoothed.fillna(0)
    
    def _calculate_kst(self, data: pd.DataFrame) -> Tuple[pd.Series, List[pd.Series], List[pd.Series]]:
        """Calculate Know Sure Thing oscillator"""
        # Get adaptive periods
        roc_periods, sma_periods = self._adapt_periods(data)
        weights = self.parameters['weights']
        
        # Ensure we have equal lengths
        min_length = min(len(roc_periods), len(sma_periods), len(weights))
        roc_periods = roc_periods[:min_length]
        sma_periods = sma_periods[:min_length]
        weights = weights[:min_length]
        
        roc_components = []
        smoothed_rocs = []
        
        # Calculate each ROC component
        for roc_period, sma_period in zip(roc_periods, sma_periods):
            # Calculate volume-weighted ROC
            roc = self._calculate_volume_weighted_roc(data, roc_period)
            roc_components.append(roc)
            
            # Smooth the ROC
            smoothed_roc = self._calculate_smoothed_roc(roc, sma_period)
            smoothed_rocs.append(smoothed_roc)
        
        # Calculate weighted KST
        kst = pd.Series(0, index=data.index)
        total_weight = sum(weights)
        
        for smoothed_roc, weight in zip(smoothed_rocs, weights):
            kst += smoothed_roc * (weight / total_weight)
        
        return kst, roc_components, smoothed_rocs
    
    def _calculate_signal_line(self, kst: pd.Series) -> pd.Series:
        """Calculate KST signal line with adaptive smoothing"""
        signal_period = self.parameters['signal_period']
        
        # Adaptive signal period based on KST volatility
        if self.parameters['adaptive_periods'] and len(kst) >= 30:
            kst_volatility = kst.tail(30).std()
            avg_volatility = kst.rolling(window=20).std().tail(30).mean()
            vol_ratio = kst_volatility / (avg_volatility + 1e-8)
            
            if vol_ratio > 1.3:
                signal_period = max(5, int(signal_period * 0.8))
            elif vol_ratio < 0.7:
                signal_period = min(15, int(signal_period * 1.2))
        
        # Calculate EMA-based signal line
        signal_line = kst.ewm(span=signal_period, adjust=False).mean()
        
        return signal_line
    
    def _apply_ica_separation(self, kst: pd.Series, signal_line: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Apply Independent Component Analysis for signal separation"""
        if not self.parameters['use_ica'] or len(kst) < 50:
            return kst, signal_line
        
        try:
            # Prepare data for ICA
            data_matrix = np.column_stack([kst.dropna().values, signal_line.dropna().values])
            
            if data_matrix.shape[0] < 10 or data_matrix.shape[1] < 2:
                return kst, signal_line
            
            # Apply ICA
            ica_components = self.ica.fit_transform(data_matrix)
            
            # Create series from ICA components
            valid_index = kst.dropna().index
            ica_component_1 = pd.Series(ica_components[:, 0], index=valid_index)
            ica_component_2 = pd.Series(ica_components[:, 1], index=valid_index)
            
            # Determine which component is more trend-like vs noise-like
            trend_component = ica_component_1 if ica_component_1.std() > ica_component_2.std() else ica_component_2
            
            # Scale back to original range
            kst_range = kst.max() - kst.min()
            trend_range = trend_component.max() - trend_component.min()
            
            if trend_range > 0:
                scaling_factor = kst_range / trend_range
                enhanced_kst = trend_component * scaling_factor
                
                # Reindex to match original
                enhanced_kst = enhanced_kst.reindex(kst.index, fill_value=0)
                
                return enhanced_kst, signal_line
            
        except Exception:
            pass  # Fall back to original if ICA fails
        
        return kst, signal_line
    
    def _detect_divergences(self, kst: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Advanced divergence detection with multi-timeframe analysis"""
        lookback = self.parameters['divergence_lookback']
        if len(kst) < lookback or len(data) < lookback:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        recent_kst = kst.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find peaks and troughs using multiple methods
        kst_peaks, kst_troughs = self._find_multi_method_extremes(recent_kst)
        price_peaks, price_troughs = self._find_multi_method_extremes(recent_prices)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence analysis
        if len(kst_troughs) >= 2 and len(price_troughs) >= 2:
            for i in range(len(kst_troughs) - 1):
                kst_trough_1 = kst_troughs[i]
                kst_trough_2 = kst_troughs[i + 1]
                
                # Find corresponding price troughs
                price_trough_1 = self._find_nearest_extreme(price_troughs, kst_trough_1[0])
                price_trough_2 = self._find_nearest_extreme(price_troughs, kst_trough_2[0])
                
                if price_trough_1 and price_trough_2:
                    price_decline = (price_trough_2[1] - price_trough_1[1]) / price_trough_1[1]
                    kst_improvement = (kst_trough_2[1] - kst_trough_1[1]) / abs(kst_trough_1[1] + 1e-8)
                    
                    # Statistical significance test
                    if price_decline < -0.02 and kst_improvement > 0.1:
                        # Validate with correlation analysis
                        correlation_strength = self._validate_divergence_correlation(
                            recent_kst, recent_prices, kst_trough_1[0], kst_trough_2[0]
                        )
                        
                        if correlation_strength > 0.3:
                            bullish_divergence = True
                            divergence_strength = max(divergence_strength, 
                                                    abs(price_decline) + abs(kst_improvement) + correlation_strength)
        
        # Bearish divergence analysis
        if len(kst_peaks) >= 2 and len(price_peaks) >= 2:
            for i in range(len(kst_peaks) - 1):
                kst_peak_1 = kst_peaks[i]
                kst_peak_2 = kst_peaks[i + 1]
                
                price_peak_1 = self._find_nearest_extreme(price_peaks, kst_peak_1[0])
                price_peak_2 = self._find_nearest_extreme(price_peaks, kst_peak_2[0])
                
                if price_peak_1 and price_peak_2:
                    price_increase = (price_peak_2[1] - price_peak_1[1]) / price_peak_1[1]
                    kst_decline = (kst_peak_2[1] - kst_peak_1[1]) / abs(kst_peak_1[1] + 1e-8)
                    
                    if price_increase > 0.02 and kst_decline < -0.1:
                        correlation_strength = self._validate_divergence_correlation(
                            recent_kst, recent_prices, kst_peak_1[0], kst_peak_2[0]
                        )
                        
                        if correlation_strength > 0.3:
                            bearish_divergence = True
                            divergence_strength = max(divergence_strength,
                                                    abs(price_increase) + abs(kst_decline) + correlation_strength)
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'kst_peaks': len(kst_peaks),
            'kst_troughs': len(kst_troughs),
            'analysis_period': lookback
        }
    
    def _find_multi_method_extremes(self, series: pd.Series) -> Tuple[List[Tuple], List[Tuple]]:
        """Find extremes using multiple detection methods"""
        if len(series) < 10:
            return [], []
        
        peaks = []
        troughs = []
        values = series.values
        
        # Method 1: Local extrema with statistical threshold
        threshold = series.std() * 0.5
        window = 3
        
        for i in range(window, len(values) - window):
            window_values = values[i-window:i+window+1]
            current_value = values[i]
            
            if current_value == max(window_values) and current_value > series.mean() + threshold:
                peaks.append((i, current_value))
            elif current_value == min(window_values) and current_value < series.mean() - threshold:
                troughs.append((i, current_value))
        
        # Method 2: Percentile-based extremes
        high_percentile = np.percentile(values, 85)
        low_percentile = np.percentile(values, 15)
        
        for i in range(2, len(values) - 2):
            if (values[i] > high_percentile and 
                values[i] > values[i-1] and values[i] > values[i+1] and
                values[i] > values[i-2] and values[i] > values[i+2]):
                # Avoid duplicates
                if not any(abs(peak[0] - i) < 3 for peak in peaks):
                    peaks.append((i, values[i]))
            elif (values[i] < low_percentile and 
                  values[i] < values[i-1] and values[i] < values[i+1] and
                  values[i] < values[i-2] and values[i] < values[i+2]):
                if not any(abs(trough[0] - i) < 3 for trough in troughs):
                    troughs.append((i, values[i]))
        
        # Sort by index
        peaks.sort(key=lambda x: x[0])
        troughs.sort(key=lambda x: x[0])
        
        return peaks, troughs
    
    def _find_nearest_extreme(self, extremes: List[Tuple], target_index: int) -> Optional[Tuple]:
        """Find the nearest extreme to a target index"""
        if not extremes:
            return None
        
        min_distance = float('inf')
        nearest_extreme = None
        
        for extreme in extremes:
            distance = abs(extreme[0] - target_index)
            if distance < min_distance:
                min_distance = distance
                nearest_extreme = extreme
        
        return nearest_extreme if min_distance < 10 else None  # Within reasonable distance
    
    def _validate_divergence_correlation(self, kst: pd.Series, prices: pd.Series, 
                                       start_idx: int, end_idx: int) -> float:
        """Validate divergence using correlation analysis"""
        try:
            if start_idx >= end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            kst_segment = kst.iloc[start_idx:end_idx+1]
            price_segment = prices.iloc[start_idx:end_idx+1]
            
            if len(kst_segment) < 3 or len(price_segment) < 3:
                return 0.0
            
            correlation = np.corrcoef(kst_segment.values, price_segment.values)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except:
            return 0.0
    
    def _analyze_multi_timeframe_momentum(self, kst: pd.Series, roc_components: List[pd.Series]) -> Dict[str, Any]:
        """Analyze momentum convergence across different timeframes"""
        if not self.parameters['multi_timeframe'] or len(kst) < 60:
            return {'convergence': False, 'component_trends': [], 'overall_momentum': 'neutral'}
        
        # Analyze trends in each ROC component
        component_trends = []
        for i, roc in enumerate(roc_components):
            if len(roc) >= 20:
                recent_roc = roc.tail(20)
                trend_slope = np.polyfit(range(len(recent_roc)), recent_roc.values, 1)[0]
                
                if trend_slope > 0.1:
                    trend = 'bullish'
                elif trend_slope < -0.1:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
                
                component_trends.append({
                    'component': i,
                    'trend': trend,
                    'slope': float(trend_slope),
                    'period': self.parameters['roc_periods'][i] if i < len(self.parameters['roc_periods']) else 10
                })
        
        # Check for convergence
        trends = [comp['trend'] for comp in component_trends]
        convergence = len(set(trends)) == 1 and trends[0] != 'neutral' if trends else False
        
        # Overall momentum assessment
        bullish_count = trends.count('bullish')
        bearish_count = trends.count('bearish')
        
        if bullish_count > bearish_count:
            overall_momentum = 'bullish'
        elif bearish_count > bullish_count:
            overall_momentum = 'bearish'
        else:
            overall_momentum = 'neutral'
        
        # KST trend analysis
        kst_trend = np.polyfit(range(len(kst.tail(20))), kst.tail(20).values, 1)[0]
        
        return {
            'convergence': convergence,
            'component_trends': component_trends,
            'overall_momentum': overall_momentum,
            'kst_trend_slope': float(kst_trend),
            'bullish_components': bullish_count,
            'bearish_components': bearish_count
        }
    
    def _train_ml_models(self, kst: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for trend prediction"""
        if len(kst) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, trend_targets, momentum_targets = self._prepare_ml_data(kst, data)
            if len(features) > 50:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train trend classifier
                self.trend_classifier.fit(scaled_features, trend_targets)
                
                # Train momentum predictor
                self.momentum_predictor.fit(scaled_features, momentum_targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, kst: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare comprehensive ML training data"""
        features, trend_targets, momentum_targets = [], [], []
        lookback = 20
        
        for i in range(lookback, len(kst) - 10):
            kst_window = kst.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            # KST features
            kst_mean = np.mean(kst_window)
            kst_std = np.std(kst_window)
            kst_trend = np.polyfit(range(len(kst_window)), kst_window, 1)[0]
            kst_current = kst_window[-1]
            kst_momentum = np.mean(np.diff(kst_window[-5:]))
            
            # Statistical features
            kst_skew = stats.skew(kst_window)
            kst_kurtosis = stats.kurtosis(kst_window)
            
            # Zero line features
            zero_crossings = sum(1 for j in range(1, len(kst_window)) 
                               if (kst_window[j] > 0) != (kst_window[j-1] > 0))
            positive_ratio = len([x for x in kst_window if x > 0]) / len(kst_window)
            
            # Price features
            price_returns = np.diff(price_window) / price_window[:-1]
            price_volatility = np.std(price_returns)
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            # Volume features
            if 'volume' in data.columns:
                volume_window = data['volume'].iloc[i-lookback:i].values
                volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
                volume_ratio = volume_window[-1] / (np.mean(volume_window) + 1e-8)
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = [
                kst_mean, kst_std, kst_trend, kst_current, kst_momentum,
                kst_skew, kst_kurtosis,
                zero_crossings, positive_ratio,
                price_volatility, price_trend,
                volume_trend, volume_ratio
            ]
            
            # Targets
            future_kst = kst.iloc[i+5:i+10].values
            if len(future_kst) > 0:
                future_trend = np.polyfit(range(len(future_kst)), future_kst, 1)[0]
                
                # Trend target
                if future_trend > 0.2:
                    trend_target = 2  # Strong bullish
                elif future_trend > 0:
                    trend_target = 1  # Bullish
                elif future_trend < -0.2:
                    trend_target = 0  # Bearish
                else:
                    trend_target = 1  # Neutral
                
                # Momentum target
                momentum_target = np.mean(future_kst) - kst_current
            else:
                trend_target = 1
                momentum_target = 0.0
            
            features.append(feature_vector)
            trend_targets.append(trend_target)
            momentum_targets.append(momentum_target)
        
        return np.array(features), np.array(trend_targets), np.array(momentum_targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate KST with comprehensive analysis"""
        try:
            # Calculate KST and components
            kst, roc_components, smoothed_rocs = self._calculate_kst(data)
            
            # Calculate signal line
            signal_line = self._calculate_signal_line(kst)
            
            # Apply ICA separation if enabled
            enhanced_kst, enhanced_signal = self._apply_ica_separation(kst, signal_line)
            
            # Advanced analysis
            divergence_analysis = self._detect_divergences(enhanced_kst, data)
            momentum_analysis = self._analyze_multi_timeframe_momentum(enhanced_kst, roc_components)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(enhanced_kst, data)
            
            # Generate signal
            signal, confidence = self._generate_kst_signal(
                enhanced_kst, enhanced_signal, divergence_analysis, momentum_analysis, data
            )
            
            # Update history
            if len(enhanced_kst) > 0 and not pd.isna(enhanced_kst.iloc[-1]):
                self.history['kst_values'].append(float(enhanced_kst.iloc[-1]))
                self.history['signal_values'].append(float(enhanced_signal.iloc[-1]) if not pd.isna(enhanced_signal.iloc[-1]) else 0.0)
                self.history['roc_components'].append([float(roc.iloc[-1]) if not pd.isna(roc.iloc[-1]) else 0.0 for roc in roc_components])
                self.history['smoothed_rocs'].append([float(roc.iloc[-1]) if not pd.isna(roc.iloc[-1]) else 0.0 for roc in smoothed_rocs])
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'kst': float(enhanced_kst.iloc[-1]) if len(enhanced_kst) > 0 and not pd.isna(enhanced_kst.iloc[-1]) else 0.0,
                'signal_line': float(enhanced_signal.iloc[-1]) if len(enhanced_signal) > 0 and not pd.isna(enhanced_signal.iloc[-1]) else 0.0,
                'signal': signal,
                'confidence': confidence,
                'divergence_analysis': divergence_analysis,
                'momentum_analysis': momentum_analysis,
                'roc_components': [float(roc.iloc[-1]) if len(roc) > 0 and not pd.isna(roc.iloc[-1]) else 0.0 for roc in roc_components],
                'smoothed_rocs': [float(roc.iloc[-1]) if len(roc) > 0 and not pd.isna(roc.iloc[-1]) else 0.0 for roc in smoothed_rocs],
                'adaptive_periods': self._adapt_periods(data),
                'market_regime': self._classify_market_regime(enhanced_kst, data),
                'values_history': {
                    'kst': enhanced_kst.tail(30).tolist(),
                    'signal_line': enhanced_signal.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Know Sure Thing: {str(e)}",
                cause=e
            )
    
    def _generate_kst_signal(self, kst: pd.Series, signal_line: pd.Series,
                            divergence_analysis: Dict, momentum_analysis: Dict,
                            data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive KST signal"""
        signal_components = []
        confidence_components = []
        
        current_kst = kst.iloc[-1] if len(kst) > 0 and not pd.isna(kst.iloc[-1]) else 0
        current_signal = signal_line.iloc[-1] if len(signal_line) > 0 and not pd.isna(signal_line.iloc[-1]) else 0
        
        # Zero-line crossover signals
        if len(kst) > 1:
            prev_kst = kst.iloc[-2]
            if prev_kst <= 0 and current_kst > 0:
                signal_components.append(0.9)
                confidence_components.append(0.9)
            elif prev_kst >= 0 and current_kst < 0:
                signal_components.append(-0.9)
                confidence_components.append(0.9)
        
        # KST vs Signal line crossover
        if len(kst) > 1 and len(signal_line) > 1:
            prev_kst = kst.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            if prev_kst <= prev_signal and current_kst > current_signal:
                signal_components.append(0.8)
                confidence_components.append(0.8)
            elif prev_kst >= prev_signal and current_kst < current_signal:
                signal_components.append(-0.8)
                confidence_components.append(0.8)
        
        # Divergence signals
        if divergence_analysis['bullish_divergence']:
            signal_components.append(0.8 * divergence_analysis['strength'])
            confidence_components.append(0.8)
        elif divergence_analysis['bearish_divergence']:
            signal_components.append(-0.8 * divergence_analysis['strength'])
            confidence_components.append(0.8)
        
        # Multi-timeframe momentum signals
        if momentum_analysis['convergence']:
            overall_momentum = momentum_analysis['overall_momentum']
            if overall_momentum == 'bullish':
                signal_components.append(0.7)
                confidence_components.append(0.8)
            elif overall_momentum == 'bearish':
                signal_components.append(-0.7)
                confidence_components.append(0.8)
        
        # Component agreement signals
        bullish_components = momentum_analysis.get('bullish_components', 0)
        bearish_components = momentum_analysis.get('bearish_components', 0)
        total_components = bullish_components + bearish_components
        
        if total_components > 0:
            component_ratio = (bullish_components - bearish_components) / total_components
            if abs(component_ratio) > 0.5:
                signal_components.append(component_ratio * 0.6)
                confidence_components.append(0.6)
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(kst, data)
                if ml_signal and ml_confidence > 0.6:
                    signal_value = 1.0 if ml_signal in [SignalType.BUY, SignalType.STRONG_BUY] else -1.0
                    signal_components.append(signal_value)
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
        
        if weighted_signal > 0.6:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _get_ml_signal(self, kst: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based trend prediction"""
        try:
            lookback = 20
            if len(kst) < lookback:
                return None, 0.0
            
            kst_window = kst.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            
            # Recreate feature vector
            kst_mean = np.mean(kst_window)
            kst_std = np.std(kst_window)
            kst_trend = np.polyfit(range(len(kst_window)), kst_window, 1)[0]
            kst_current = kst_window[-1]
            kst_momentum = np.mean(np.diff(kst_window[-5:]))
            
            kst_skew = stats.skew(kst_window)
            kst_kurtosis = stats.kurtosis(kst_window)
            
            zero_crossings = sum(1 for j in range(1, len(kst_window)) 
                               if (kst_window[j] > 0) != (kst_window[j-1] > 0))
            positive_ratio = len([x for x in kst_window if x > 0]) / len(kst_window)
            
            price_returns = np.diff(price_window) / price_window[:-1]
            price_volatility = np.std(price_returns)
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            if 'volume' in data.columns:
                volume_window = data['volume'].tail(lookback).values
                volume_trend = np.polyfit(range(len(volume_window)), volume_window, 1)[0]
                volume_ratio = volume_window[-1] / (np.mean(volume_window) + 1e-8)
            else:
                volume_trend = 0
                volume_ratio = 1
            
            feature_vector = np.array([[
                kst_mean, kst_std, kst_trend, kst_current, kst_momentum,
                kst_skew, kst_kurtosis,
                zero_crossings, positive_ratio,
                price_volatility, price_trend,
                volume_trend, volume_ratio
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            trend_proba = self.trend_classifier.predict_proba(scaled_features)[0]
            
            if len(trend_proba) >= 3:
                bearish_prob = trend_proba[0]
                neutral_prob = trend_proba[1]
                bullish_prob = trend_proba[2]
                
                max_prob = max(trend_proba)
                if max_prob > 0.7:
                    if bullish_prob == max_prob:
                        return SignalType.BUY, max_prob
                    elif bearish_prob == max_prob:
                        return SignalType.SELL, max_prob
        except:
            pass
        
        return None, 0.0
    
    def _classify_market_regime(self, kst: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime based on KST characteristics"""
        if len(kst) < 30:
            return {'regime': 'unknown', 'momentum_phase': 'undefined', 'volatility': 'normal'}
        
        recent_kst = kst.tail(30)
        current_kst = kst.iloc[-1]
        
        # Momentum phase analysis
        kst_trend = np.polyfit(range(len(recent_kst)), recent_kst.values, 1)[0]
        
        if current_kst > 0 and kst_trend > 0.1:
            momentum_phase = 'bullish_acceleration'
        elif current_kst > 0 and kst_trend < -0.1:
            momentum_phase = 'bullish_deceleration'
        elif current_kst < 0 and kst_trend < -0.1:
            momentum_phase = 'bearish_acceleration'
        elif current_kst < 0 and kst_trend > 0.1:
            momentum_phase = 'bearish_deceleration'
        else:
            momentum_phase = 'neutral'
        
        # Volatility analysis
        kst_volatility = recent_kst.std()
        if kst_volatility > 5:
            volatility = 'high'
        elif kst_volatility > 2:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        # Overall regime
        zero_crossings = sum(1 for i in range(1, len(recent_kst)) 
                           if (recent_kst.iloc[i] > 0) != (recent_kst.iloc[i-1] > 0))
        
        if zero_crossings > 6:
            regime = 'choppy'
        elif abs(current_kst) > 5 and momentum_phase in ['bullish_acceleration', 'bearish_acceleration']:
            regime = 'trending'
        elif volatility == 'low' and abs(kst_trend) < 0.05:
            regime = 'consolidation'
        else:
            regime = 'transitional'
        
        return {
            'regime': regime,
            'momentum_phase': momentum_phase,
            'volatility': volatility,
            'kst_trend': float(kst_trend),
            'current_level': float(current_kst),
            'zero_crossings': zero_crossings
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'know_sure_thing',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'use_ica': self.parameters['use_ica'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'savgol_smoothing': self.parameters['savgol_smoothing'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata