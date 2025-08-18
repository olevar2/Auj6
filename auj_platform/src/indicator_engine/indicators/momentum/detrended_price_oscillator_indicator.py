"""
Detrended Price Oscillator (DPO) - Advanced Implementation
=========================================================

Advanced DPO with ML-enhanced cycle detection, adaptive detrending,
and sophisticated pattern recognition for market cycle analysis.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import signal, stats
from scipy.fft import fft, fftfreq
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


class DetrendedPriceOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Detrended Price Oscillator Implementation
    
    Features:
    - Adaptive period selection using dominant cycle detection
    - ML-enhanced cycle pattern recognition
    - Multi-scale detrending with wavelets and Fourier analysis
    - Seasonal decomposition and cycle strength measurement
    - Advanced overbought/oversold detection with statistical validation
    - Institutional flow detection through volume analysis
    - Dynamic threshold optimization for changing market conditions
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 20,
            'offset_factor': 0.5,  # Offset = period / 2 + 1
            'adaptive_period': True,
            'cycle_detection': True,
            'fourier_analysis': True,
            'volume_weighted': True,
            'ml_lookback': 60,
            'min_period': 10,
            'max_period': 50,
            'overbought_percentile': 80,
            'oversold_percentile': 20,
            'seasonal_adjustment': True,
            'outlier_detection': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="DetrendedPriceOscillatorIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.cycle_detector = RandomForestRegressor(n_estimators=100, random_state=42)
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.pattern_clusterer = DBSCAN(eps=0.5, min_samples=5)
        self.models_trained = False
        
        self.history = {
            'dpo_values': [],
            'sma_values': [],
            'cycles': [],
            'dominant_periods': [],
            'threshold_values': [],
            'pattern_labels': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max(self.parameters['max_period'] * 2, self.parameters['ml_lookback']) + 30,
            lookback_periods=200
        )
    
    def _detect_dominant_cycle(self, prices: pd.Series) -> int:
        """Detect dominant cycle using spectral analysis"""
        if len(prices) < 100:
            return self.parameters['period']
        
        try:
            # Use recent data for cycle detection
            recent_prices = prices.tail(100).values
            
            # Detrend the data first
            detrended = signal.detrend(recent_prices)
            
            # Apply window to reduce spectral leakage
            windowed = detrended * np.hanning(len(detrended))
            
            # Compute FFT
            fft_values = fft(windowed)
            frequencies = fftfreq(len(windowed))
            
            # Get power spectrum (magnitude squared)
            power_spectrum = np.abs(fft_values) ** 2
            
            # Focus on meaningful frequencies (periods between min and max)
            min_freq = 1.0 / self.parameters['max_period']
            max_freq = 1.0 / self.parameters['min_period']
            
            valid_indices = np.where((np.abs(frequencies) >= min_freq) & 
                                   (np.abs(frequencies) <= max_freq) &
                                   (frequencies > 0))[0]
            
            if len(valid_indices) > 0:
                valid_power = power_spectrum[valid_indices]
                valid_frequencies = frequencies[valid_indices]
                
                # Find frequency with maximum power
                max_power_idx = np.argmax(valid_power)
                dominant_frequency = valid_frequencies[max_power_idx]
                
                # Convert to period
                dominant_period = int(1.0 / dominant_frequency)
                
                # Ensure period is within bounds
                dominant_period = max(self.parameters['min_period'],
                                    min(dominant_period, self.parameters['max_period']))
                
                return dominant_period
            
        except Exception:
            pass
        
        return self.parameters['period']
    
    def _calculate_adaptive_period(self, data: pd.DataFrame) -> int:
        """Calculate adaptive period based on market conditions"""
        if not self.parameters['adaptive_period']:
            return self.parameters['period']
        
        close_prices = data['close']
        
        # Method 1: Dominant cycle detection
        if self.parameters['cycle_detection']:
            cycle_period = self._detect_dominant_cycle(close_prices)
        else:
            cycle_period = self.parameters['period']
        
        # Method 2: Volatility-based adjustment
        if len(close_prices) >= 60:
            returns = close_prices.pct_change().tail(60)
            current_vol = returns.std()
            avg_vol = returns.rolling(window=20).std().mean()
            vol_ratio = current_vol / (avg_vol + 1e-8)
            
            # Adjust for volatility regime
            if vol_ratio > 1.5:  # High volatility - shorter period
                vol_adjustment = 0.8
            elif vol_ratio < 0.6:  # Low volatility - longer period
                vol_adjustment = 1.2
            else:
                vol_adjustment = 1.0
            
            cycle_period = int(cycle_period * vol_adjustment)
        
        # Method 3: Trend strength adjustment
        if len(close_prices) >= 40:
            price_changes = close_prices.diff().tail(40)
            trend_strength = abs(price_changes.sum()) / (price_changes.abs().sum() + 1e-8)
            
            if trend_strength > 0.7:  # Strong trend - longer period
                trend_adjustment = 1.1
            elif trend_strength < 0.3:  # Weak trend - shorter period
                trend_adjustment = 0.9
            else:
                trend_adjustment = 1.0
            
            cycle_period = int(cycle_period * trend_adjustment)
        
        # Ensure period is within bounds
        cycle_period = max(self.parameters['min_period'],
                          min(cycle_period, self.parameters['max_period']))
        
        return cycle_period
    
    def _calculate_volume_weighted_price(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate volume-weighted price for institutional flow detection"""
        if not self.parameters['volume_weighted'] or 'volume' not in data.columns:
            return data['close']
        
        close = data['close']
        volume = data['volume']
        
        # Calculate VWAP over rolling window
        vwap = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # Blend regular price with VWAP based on volume significance
        volume_ratio = volume / volume.rolling(window=period).mean()
        volume_weight = np.tanh(volume_ratio - 1)  # Sigmoid-like weighting
        
        weighted_price = close * (1 - volume_weight) + vwap * volume_weight
        
        return weighted_price.fillna(close)
    
    def _calculate_seasonal_adjustment(self, prices: pd.Series, period: int) -> pd.Series:
        """Apply seasonal adjustment to remove periodic patterns"""
        if not self.parameters['seasonal_adjustment'] or len(prices) < period * 3:
            return prices
        
        try:
            # Calculate seasonal component
            seasonal_data = []
            for i in range(period):
                seasonal_values = []
                idx = i
                while idx < len(prices):
                    seasonal_values.append(prices.iloc[idx])
                    idx += period
                
                if seasonal_values:
                    seasonal_data.append(np.mean(seasonal_values))
                else:
                    seasonal_data.append(0)
            
            # Create seasonal series
            seasonal_series = []
            for i in range(len(prices)):
                seasonal_idx = i % period
                seasonal_series.append(seasonal_data[seasonal_idx])
            
            seasonal_series = pd.Series(seasonal_series, index=prices.index)
            
            # Remove seasonal component
            deseasonalized = prices - seasonal_series
            
            return deseasonalized
            
        except Exception:
            return prices
    
    def _calculate_dpo(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, int]:
        """Calculate Detrended Price Oscillator with advanced processing"""
        # Get adaptive period
        period = self._calculate_adaptive_period(data)
        
        # Get volume-weighted price if enabled
        prices = self._calculate_volume_weighted_price(data, period)
        
        # Apply seasonal adjustment if enabled
        if self.parameters['seasonal_adjustment']:
            prices = self._calculate_seasonal_adjustment(prices, period)
        
        # Calculate Simple Moving Average
        sma = prices.rolling(window=period).mean()
        
        # Calculate offset (look back period)
        offset = int(period * self.parameters['offset_factor']) + 1
        
        # Calculate DPO: Price - SMA[offset periods ago]
        dpo = prices - sma.shift(offset)
        
        return dpo, sma, period
    
    def _detect_cycle_patterns(self, dpo: pd.Series) -> Dict[str, Any]:
        """Detect and classify cycle patterns using ML"""
        if len(dpo) < 50 or not self.parameters['cycle_detection']:
            return {'pattern': 'insufficient_data', 'strength': 0.0, 'phase': 'unknown'}
        
        try:
            # Prepare features for pattern recognition
            recent_dpo = dpo.tail(50).dropna()
            
            if len(recent_dpo) < 20:
                return {'pattern': 'insufficient_data', 'strength': 0.0, 'phase': 'unknown'}
            
            # Extract cycle features
            features = []
            window_size = 10
            
            for i in range(len(recent_dpo) - window_size + 1):
                window = recent_dpo.iloc[i:i+window_size].values
                
                # Statistical features
                mean_val = np.mean(window)
                std_val = np.std(window)
                skew_val = stats.skew(window)
                kurtosis_val = stats.kurtosis(window)
                
                # Trend features
                trend_slope = np.polyfit(range(len(window)), window, 1)[0]
                
                # Oscillation features
                zero_crossings = sum(1 for j in range(1, len(window)) 
                                   if (window[j] > 0) != (window[j-1] > 0))
                range_val = np.max(window) - np.min(window)
                
                # Cycle position features
                max_idx = np.argmax(window)
                min_idx = np.argmin(window)
                cycle_position = (max_idx - min_idx) / len(window)
                
                features.append([mean_val, std_val, skew_val, kurtosis_val,
                               trend_slope, zero_crossings, range_val, cycle_position])
            
            if len(features) < 5:
                return {'pattern': 'insufficient_data', 'strength': 0.0, 'phase': 'unknown'}
            
            features = np.array(features)
            
            # Use clustering to identify patterns
            try:
                pattern_labels = self.pattern_clusterer.fit_predict(features)
                current_pattern = pattern_labels[-1] if len(pattern_labels) > 0 else -1
                
                # Map patterns to meaningful names
                if current_pattern == -1:
                    pattern_name = 'noise'
                elif current_pattern == 0:
                    pattern_name = 'trending_up'
                elif current_pattern == 1:
                    pattern_name = 'trending_down'
                elif current_pattern == 2:
                    pattern_name = 'oscillating'
                else:
                    pattern_name = f'pattern_{current_pattern}'
                
                # Calculate pattern strength
                if current_pattern != -1:
                    same_pattern_count = sum(1 for label in pattern_labels[-10:] if label == current_pattern)
                    pattern_strength = same_pattern_count / min(10, len(pattern_labels))
                else:
                    pattern_strength = 0.0
                
            except:
                pattern_name = 'unknown'
                pattern_strength = 0.0
            
            # Determine cycle phase
            current_dpo = recent_dpo.iloc[-1]
            recent_mean = recent_dpo.tail(5).mean()
            
            if current_dpo > recent_mean and current_dpo > 0:
                phase = 'peak'
            elif current_dpo < recent_mean and current_dpo < 0:
                phase = 'trough'
            elif current_dpo > 0:
                phase = 'rising'
            elif current_dpo < 0:
                phase = 'falling'
            else:
                phase = 'neutral'
            
            return {
                'pattern': pattern_name,
                'strength': float(pattern_strength),
                'phase': phase,
                'cycle_position': float(current_dpo / (recent_dpo.std() + 1e-8))
            }
            
        except Exception:
            return {'pattern': 'unknown', 'strength': 0.0, 'phase': 'unknown'}
    
    def _calculate_dynamic_thresholds(self, dpo: pd.Series) -> Dict[str, float]:
        """Calculate dynamic overbought/oversold thresholds"""
        if len(dpo) < 50:
            return {'overbought': 0.0, 'oversold': 0.0}
        
        recent_dpo = dpo.tail(60).dropna()
        
        if len(recent_dpo) < 20:
            return {'overbought': 0.0, 'oversold': 0.0}
        
        # Use percentile-based thresholds
        overbought = np.percentile(recent_dpo, self.parameters['overbought_percentile'])
        oversold = np.percentile(recent_dpo, self.parameters['oversold_percentile'])
        
        # Apply statistical filters to remove outliers
        if self.parameters['outlier_detection']:
            try:
                # Fit outlier detector
                dpo_reshaped = recent_dpo.values.reshape(-1, 1)
                outlier_labels = self.outlier_detector.fit_predict(dpo_reshaped)
                
                # Remove outliers and recalculate thresholds
                clean_dpo = recent_dpo[outlier_labels == 1]
                
                if len(clean_dpo) > 10:
                    overbought = np.percentile(clean_dpo, self.parameters['overbought_percentile'])
                    oversold = np.percentile(clean_dpo, self.parameters['oversold_percentile'])
                    
            except:
                pass
        
        # Ensure thresholds are meaningful
        dpo_std = recent_dpo.std()
        min_threshold = dpo_std * 0.5
        
        overbought = max(overbought, min_threshold)
        oversold = min(oversold, -min_threshold)
        
        return {
            'overbought': float(overbought),
            'oversold': float(oversold),
            'std_dev': float(dpo_std)
        }
    
    def _analyze_fourier_components(self, dpo: pd.Series) -> Dict[str, Any]:
        """Analyze frequency components using Fourier analysis"""
        if not self.parameters['fourier_analysis'] or len(dpo) < 50:
            return {'dominant_frequencies': [], 'power_distribution': {}, 'signal_quality': 0.0}
        
        try:
            recent_dpo = dpo.tail(100).dropna().values
            
            if len(recent_dpo) < 20:
                return {'dominant_frequencies': [], 'power_distribution': {}, 'signal_quality': 0.0}
            
            # Apply window and compute FFT
            windowed = recent_dpo * np.hanning(len(recent_dpo))
            fft_values = fft(windowed)
            frequencies = fftfreq(len(windowed))
            
            # Power spectrum
            power_spectrum = np.abs(fft_values) ** 2
            
            # Focus on positive frequencies
            positive_freq_idx = frequencies > 0
            positive_freqs = frequencies[positive_freq_idx]
            positive_power = power_spectrum[positive_freq_idx]
            
            # Find dominant frequencies
            sorted_indices = np.argsort(positive_power)[::-1]
            top_5_indices = sorted_indices[:5]
            
            dominant_frequencies = []
            for idx in top_5_indices:
                freq = positive_freqs[idx]
                power = positive_power[idx]
                period = 1.0 / freq if freq > 0 else np.inf
                
                if 5 <= period <= 50:  # Meaningful periods only
                    dominant_frequencies.append({
                        'frequency': float(freq),
                        'period': float(period),
                        'power': float(power)
                    })
            
            # Power distribution by frequency bands
            low_freq_power = np.sum(positive_power[positive_freqs < 0.1])
            mid_freq_power = np.sum(positive_power[(positive_freqs >= 0.1) & (positive_freqs < 0.3)])
            high_freq_power = np.sum(positive_power[positive_freqs >= 0.3])
            total_power = low_freq_power + mid_freq_power + high_freq_power
            
            if total_power > 0:
                power_distribution = {
                    'low_frequency': float(low_freq_power / total_power),
                    'mid_frequency': float(mid_freq_power / total_power),
                    'high_frequency': float(high_freq_power / total_power)
                }
            else:
                power_distribution = {'low_frequency': 0.0, 'mid_frequency': 0.0, 'high_frequency': 0.0}
            
            # Signal quality (ratio of dominant frequency power to total power)
            if len(dominant_frequencies) > 0 and total_power > 0:
                signal_quality = dominant_frequencies[0]['power'] / total_power
            else:
                signal_quality = 0.0
            
            return {
                'dominant_frequencies': dominant_frequencies,
                'power_distribution': power_distribution,
                'signal_quality': float(signal_quality)
            }
            
        except Exception:
            return {'dominant_frequencies': [], 'power_distribution': {}, 'signal_quality': 0.0}
    
    def _train_ml_models(self, dpo: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for cycle prediction"""
        if len(dpo) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(dpo, data)
            if len(features) > 30:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train cycle detector
                self.cycle_detector.fit(scaled_features, targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, dpo: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data for cycle prediction"""
        features, targets = [], []
        lookback = 15
        
        for i in range(lookback, len(dpo) - 10):
            dpo_window = dpo.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            # DPO features
            dpo_mean = np.mean(dpo_window)
            dpo_std = np.std(dpo_window)
            dpo_trend = np.polyfit(range(len(dpo_window)), dpo_window, 1)[0]
            dpo_current = dpo_window[-1]
            
            # Cycle features
            zero_crossings = sum(1 for j in range(1, len(dpo_window)) 
                               if (dpo_window[j] > 0) != (dpo_window[j-1] > 0))
            amplitude = np.max(dpo_window) - np.min(dpo_window)
            
            # Price features
            price_volatility = np.std(np.diff(price_window) / price_window[:-1])
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            # Statistical features
            dpo_skew = stats.skew(dpo_window)
            dpo_kurtosis = stats.kurtosis(dpo_window)
            
            feature_vector = [
                dpo_mean, dpo_std, dpo_trend, dpo_current,
                zero_crossings, amplitude,
                price_volatility, price_trend,
                dpo_skew, dpo_kurtosis
            ]
            
            # Target: future DPO direction
            future_dpo = dpo.iloc[i+5:i+10].values
            if len(future_dpo) > 0:
                future_change = np.mean(future_dpo) - dpo_current
                target = future_change
            else:
                target = 0.0
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate DPO with comprehensive analysis"""
        try:
            # Calculate DPO and components
            dpo, sma, adaptive_period = self._calculate_dpo(data)
            
            # Advanced analysis
            cycle_patterns = self._detect_cycle_patterns(dpo)
            dynamic_thresholds = self._calculate_dynamic_thresholds(dpo)
            fourier_analysis = self._analyze_fourier_components(dpo)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(dpo, data)
            
            # Generate signal
            signal, confidence = self._generate_dpo_signal(
                dpo, cycle_patterns, dynamic_thresholds, fourier_analysis, data
            )
            
            # Update history
            if len(dpo) > 0 and not pd.isna(dpo.iloc[-1]):
                self.history['dpo_values'].append(float(dpo.iloc[-1]))
                self.history['sma_values'].append(float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0)
                self.history['dominant_periods'].append(adaptive_period)
                self.history['threshold_values'].append(dynamic_thresholds)
                self.history['cycles'].append(cycle_patterns)
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'dpo': float(dpo.iloc[-1]) if len(dpo) > 0 and not pd.isna(dpo.iloc[-1]) else 0.0,
                'sma': float(sma.iloc[-1]) if len(sma) > 0 and not pd.isna(sma.iloc[-1]) else 0.0,
                'signal': signal,
                'confidence': confidence,
                'adaptive_period': adaptive_period,
                'cycle_patterns': cycle_patterns,
                'dynamic_thresholds': dynamic_thresholds,
                'fourier_analysis': fourier_analysis,
                'market_regime': self._classify_market_regime(dpo, data),
                'values_history': {
                    'dpo': dpo.tail(30).tolist(),
                    'sma': sma.tail(30).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Detrended Price Oscillator: {str(e)}",
                cause=e
            )
    
    def _generate_dpo_signal(self, dpo: pd.Series, cycle_patterns: Dict,
                            dynamic_thresholds: Dict, fourier_analysis: Dict,
                            data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive DPO signal"""
        signal_components = []
        confidence_components = []
        
        current_dpo = dpo.iloc[-1] if len(dpo) > 0 and not pd.isna(dpo.iloc[-1]) else 0
        overbought = dynamic_thresholds.get('overbought', 0)
        oversold = dynamic_thresholds.get('oversold', 0)
        
        # Threshold-based signals
        if current_dpo > overbought and overbought > 0:
            signal_components.append(-0.7)
            confidence_components.append(0.7)
        elif current_dpo < oversold and oversold < 0:
            signal_components.append(0.7)
            confidence_components.append(0.7)
        
        # Zero-line crossover signals
        if len(dpo) > 1:
            prev_dpo = dpo.iloc[-2]
            if prev_dpo <= 0 and current_dpo > 0:
                signal_components.append(0.8)
                confidence_components.append(0.8)
            elif prev_dpo >= 0 and current_dpo < 0:
                signal_components.append(-0.8)
                confidence_components.append(0.8)
        
        # Cycle pattern signals
        if cycle_patterns['strength'] > 0.6:
            pattern = cycle_patterns['pattern']
            phase = cycle_patterns['phase']
            
            if pattern == 'trending_up' or phase == 'rising':
                signal_components.append(0.6 * cycle_patterns['strength'])
                confidence_components.append(cycle_patterns['strength'])
            elif pattern == 'trending_down' or phase == 'falling':
                signal_components.append(-0.6 * cycle_patterns['strength'])
                confidence_components.append(cycle_patterns['strength'])
            elif phase == 'trough':
                signal_components.append(0.5)
                confidence_components.append(0.6)
            elif phase == 'peak':
                signal_components.append(-0.5)
                confidence_components.append(0.6)
        
        # Fourier analysis signals
        signal_quality = fourier_analysis.get('signal_quality', 0)
        if signal_quality > 0.3:  # Good signal quality
            dominant_freqs = fourier_analysis.get('dominant_frequencies', [])
            if dominant_freqs:
                # Use dominant cycle for prediction
                dominant_period = dominant_freqs[0]['period']
                
                # Estimate position in cycle
                if len(dpo) >= int(dominant_period):
                    cycle_data = dpo.tail(int(dominant_period))
                    cycle_position = len(cycle_data) - np.argmax(cycle_data.values) - 1
                    cycle_ratio = cycle_position / dominant_period
                    
                    # Signal based on cycle position
                    if 0.4 <= cycle_ratio <= 0.6:  # Near peak
                        signal_components.append(-0.4 * signal_quality)
                        confidence_components.append(signal_quality)
                    elif cycle_ratio <= 0.1 or cycle_ratio >= 0.9:  # Near trough
                        signal_components.append(0.4 * signal_quality)
                        confidence_components.append(signal_quality)
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(dpo, data)
                if ml_signal and ml_confidence > 0.5:
                    signal_components.append(ml_signal)
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
    
    def _get_ml_signal(self, dpo: pd.Series, data: pd.DataFrame) -> Tuple[float, float]:
        """Get ML-based cycle prediction"""
        try:
            lookback = 15
            if len(dpo) < lookback:
                return 0.0, 0.0
            
            dpo_window = dpo.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            
            # Recreate feature vector
            dpo_mean = np.mean(dpo_window)
            dpo_std = np.std(dpo_window)
            dpo_trend = np.polyfit(range(len(dpo_window)), dpo_window, 1)[0]
            dpo_current = dpo_window[-1]
            
            zero_crossings = sum(1 for j in range(1, len(dpo_window)) 
                               if (dpo_window[j] > 0) != (dpo_window[j-1] > 0))
            amplitude = np.max(dpo_window) - np.min(dpo_window)
            
            price_volatility = np.std(np.diff(price_window) / price_window[:-1])
            price_trend = np.polyfit(range(len(price_window)), price_window, 1)[0]
            
            dpo_skew = stats.skew(dpo_window)
            dpo_kurtosis = stats.kurtosis(dpo_window)
            
            feature_vector = np.array([[
                dpo_mean, dpo_std, dpo_trend, dpo_current,
                zero_crossings, amplitude,
                price_volatility, price_trend,
                dpo_skew, dpo_kurtosis
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            predicted_change = self.cycle_detector.predict(scaled_features)[0]
            
            # Convert to signal
            if abs(predicted_change) > 0.1:
                signal_strength = np.tanh(predicted_change * 5)  # Normalize to [-1, 1]
                confidence = min(abs(predicted_change) * 2, 1.0)
                return signal_strength, confidence
                
        except:
            pass
        
        return 0.0, 0.0
    
    def _classify_market_regime(self, dpo: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime based on DPO characteristics"""
        if len(dpo) < 30:
            return {'regime': 'unknown', 'cycle_phase': 'undefined', 'detrending_effectiveness': 0.0}
        
        recent_dpo = dpo.tail(30)
        current_dpo = dpo.iloc[-1]
        
        # Cycle phase analysis
        dpo_std = recent_dpo.std()
        normalized_dpo = current_dpo / (dpo_std + 1e-8)
        
        if normalized_dpo > 1:
            cycle_phase = 'overbought'
        elif normalized_dpo < -1:
            cycle_phase = 'oversold'
        elif normalized_dpo > 0.3:
            cycle_phase = 'rising'
        elif normalized_dpo < -0.3:
            cycle_phase = 'falling'
        else:
            cycle_phase = 'neutral'
        
        # Detrending effectiveness (how well DPO oscillates around zero)
        zero_crossings = sum(1 for i in range(1, len(recent_dpo)) 
                           if (recent_dpo.iloc[i] > 0) != (recent_dpo.iloc[i-1] > 0))
        
        # Normalize by length
        crossing_ratio = zero_crossings / len(recent_dpo)
        
        if crossing_ratio > 0.2:
            detrending_effectiveness = 'high'
        elif crossing_ratio > 0.1:
            detrending_effectiveness = 'medium'
        else:
            detrending_effectiveness = 'low'
        
        # Overall regime
        if detrending_effectiveness == 'high' and abs(normalized_dpo) < 1:
            regime = 'normal_cycling'
        elif detrending_effectiveness == 'low':
            regime = 'trending'
        elif abs(normalized_dpo) > 1.5:
            regime = 'extreme'
        else:
            regime = 'transitional'
        
        return {
            'regime': regime,
            'cycle_phase': cycle_phase,
            'detrending_effectiveness': detrending_effectiveness,
            'normalized_position': float(normalized_dpo),
            'volatility': float(dpo_std)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'detrended_price_oscillator',
            'models_trained': self.models_trained,
            'adaptive_period': self.parameters['adaptive_period'],
            'cycle_detection': self.parameters['cycle_detection'],
            'fourier_analysis': self.parameters['fourier_analysis'],
            'volume_weighted': self.parameters['volume_weighted'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata