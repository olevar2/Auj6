"""
Commodity Channel Index - AI-Enhanced Multi-Dimensional CCI System
================================================================

Advanced CCI implementation with machine learning, adaptive parameters,
multi-timeframe analysis, and sophisticated overbought/oversold detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class AICommodityChannelIndex(StandardIndicatorInterface):
    """
    AI-Enhanced Commodity Channel Index with advanced features.
    
    Features:
    - Adaptive period optimization using volatility
    - Multi-timeframe CCI analysis and confluence
    - Machine learning-based threshold optimization
    - Divergence detection with pattern recognition
    - Volatility-adjusted overbought/oversold levels
    - Trend strength integration
    - Mean reversion probability calculation
    - Dynamic channel boundary detection
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'base_period': 14,             # Base CCI period
            'adaptive_periods': [7, 14, 21, 28],  # Multiple periods for analysis
            'overbought_level': 100,       # Standard overbought level
            'oversold_level': -100,        # Standard oversold level
            'volatility_window': 20,       # Window for volatility calculation
            'divergence_window': 10,       # Window for divergence detection
            'trend_confirmation_period': 5, # Period for trend confirmation
            'ml_optimization': True,       # Enable ML threshold optimization
            'multi_timeframe': True,       # Enable multi-timeframe analysis
            'adaptive_thresholds': True,   # Enable adaptive threshold calculation
            'pattern_recognition': True,   # Enable pattern recognition
            'mean_reversion_analysis': True, # Enable mean reversion analysis
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name="CommodityChannelIndex")
        
        # Analysis components
        self.scaler = StandardScaler()
        self.threshold_history = []
        self.signal_history = []
        
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(self.parameters['adaptive_periods']) * 2,
            lookback_periods=100
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive CCI analysis.
        """
        try:
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values
            
            if len(closes) < max(self.parameters['adaptive_periods']) * 2:
                return self._default_result()
            
            # Calculate typical price
            typical_prices = (highs + lows + closes) / 3.0
            
            # Multi-period CCI calculation
            cci_analysis = self._calculate_multi_period_cci(typical_prices, volumes)
            
            # Adaptive period optimization
            optimal_period = self._optimize_adaptive_period(typical_prices, volumes) if self.parameters['ml_optimization'] else self.parameters['base_period']
            
            # Primary CCI with optimal period
            primary_cci = self._calculate_cci(typical_prices, optimal_period)
            
            # Adaptive threshold calculation
            thresholds = self._calculate_adaptive_thresholds(primary_cci, volumes) if self.parameters['adaptive_thresholds'] else {
                'overbought': self.parameters['overbought_level'],
                'oversold': self.parameters['oversold_level']
            }
            
            # Multi-timeframe analysis
            mtf_analysis = self._multi_timeframe_analysis(cci_analysis) if self.parameters['multi_timeframe'] else {}
            
            # Divergence detection
            divergence_analysis = self._detect_divergences(primary_cci, closes) if self.parameters['pattern_recognition'] else {}
            
            # Pattern recognition
            pattern_analysis = self._analyze_cci_patterns(primary_cci) if self.parameters['pattern_recognition'] else {}
            
            # Trend analysis
            trend_analysis = self._analyze_trend_strength(primary_cci, closes)
            
            # Mean reversion analysis
            mean_reversion = self._analyze_mean_reversion(primary_cci, typical_prices) if self.parameters['mean_reversion_analysis'] else {}
            
            # Volatility integration
            volatility_analysis = self._analyze_volatility_context(primary_cci, typical_prices, volumes)
            
            # Signal generation
            signal_strength = self._calculate_signal_strength(
                primary_cci, thresholds, mtf_analysis, divergence_analysis,
                pattern_analysis, trend_analysis, mean_reversion, volatility_analysis
            )
            
            return {
                'primary_cci': primary_cci[-1] if len(primary_cci) > 0 else 0,
                'cci_history': primary_cci[-20:].tolist() if len(primary_cci) >= 20 else primary_cci.tolist(),
                'cci_analysis': cci_analysis,
                'optimal_period': optimal_period,
                'thresholds': thresholds,
                'mtf_analysis': mtf_analysis,
                'divergence_analysis': divergence_analysis,
                'pattern_analysis': pattern_analysis,
                'trend_analysis': trend_analysis,
                'mean_reversion': mean_reversion,
                'volatility_analysis': volatility_analysis,
                'signal_strength': signal_strength,
                'market_state': self._classify_market_state(primary_cci, thresholds)
            }
            
        except Exception as e:
            raise Exception(f"CommodityChannelIndex calculation failed: {str(e)}")
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result for insufficient data."""
        return {
            'primary_cci': 0.0,
            'cci_history': [],
            'cci_analysis': {},
            'optimal_period': self.parameters['base_period'],
            'thresholds': {
                'overbought': self.parameters['overbought_level'],
                'oversold': self.parameters['oversold_level']
            },
            'mtf_analysis': {},
            'divergence_analysis': {},
            'pattern_analysis': {},
            'trend_analysis': {},
            'mean_reversion': {},
            'volatility_analysis': {},
            'signal_strength': 0.0,
            'market_state': 'neutral'
        }
    
    def _calculate_cci(self, typical_prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Commodity Channel Index for given period."""
        if len(typical_prices) < period:
            return np.array([])
        
        cci_values = []
        
        for i in range(period - 1, len(typical_prices)):
            # Moving average of typical price
            window = typical_prices[i - period + 1:i + 1]
            sma = np.mean(window)
            
            # Mean deviation
            mean_deviation = np.mean(np.abs(window - sma))
            
            # CCI calculation
            if mean_deviation != 0:
                cci = (typical_prices[i] - sma) / (0.015 * mean_deviation)
            else:
                cci = 0
            
            cci_values.append(cci)
        
        return np.array(cci_values)
    
    def _calculate_multi_period_cci(self, typical_prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Calculate CCI for multiple periods."""
        cci_results = {}
        
        for period in self.parameters['adaptive_periods']:
            cci_values = self._calculate_cci(typical_prices, period)
            
            if len(cci_values) > 0:
                # Basic statistics
                current_cci = cci_values[-1]
                avg_cci = np.mean(cci_values[-10:]) if len(cci_values) >= 10 else current_cci
                cci_volatility = np.std(cci_values[-20:]) if len(cci_values) >= 20 else 0
                
                # Momentum
                momentum = 0
                if len(cci_values) >= 5:
                    momentum = cci_values[-1] - cci_values[-5]
                
                # Trend direction
                trend_direction = 0
                if len(cci_values) >= 3:
                    recent_values = cci_values[-3:]
                    if len(recent_values) > 1:
                        slope, _ = np.polyfit(range(len(recent_values)), recent_values, 1)
                        trend_direction = np.sign(slope)
                
                cci_results[f'period_{period}'] = {
                    'current_value': float(current_cci),
                    'average_value': float(avg_cci),
                    'volatility': float(cci_volatility),
                    'momentum': float(momentum),
                    'trend_direction': float(trend_direction),
                    'extreme_count': int(np.sum(np.abs(cci_values[-10:]) > 100)) if len(cci_values) >= 10 else 0
                }
        
        # Consensus analysis
        if cci_results:
            current_values = [result['current_value'] for result in cci_results.values()]
            trend_directions = [result['trend_direction'] for result in cci_results.values()]
            
            consensus = {
                'average_cci': float(np.mean(current_values)),
                'cci_divergence': float(np.std(current_values)),
                'trend_consensus': float(np.mean(trend_directions)),
                'agreement_strength': float(1.0 - (np.std(current_values) / (np.mean(np.abs(current_values)) + 1e-8)))
            }
            
            cci_results['consensus'] = consensus
        
        return cci_results
    
    def _optimize_adaptive_period(self, typical_prices: np.ndarray, volumes: np.ndarray) -> int:
        """Optimize CCI period using machine learning approach."""
        try:
            # Calculate volatility for adaptive period selection
            volatility_window = self.parameters['volatility_window']
            
            if len(typical_prices) < volatility_window * 2:
                return self.parameters['base_period']
            
            # Calculate rolling volatility
            returns = np.diff(np.log(typical_prices + 1e-8))
            rolling_vol = []
            
            for i in range(volatility_window - 1, len(returns)):
                window_vol = np.std(returns[i - volatility_window + 1:i + 1])
                rolling_vol.append(window_vol)
            
            current_volatility = rolling_vol[-1] if rolling_vol else 0
            avg_volatility = np.mean(rolling_vol) if rolling_vol else 0
            
            # Adaptive period based on volatility regime
            if current_volatility > avg_volatility * 1.5:
                # High volatility - use shorter period
                optimal_period = min(self.parameters['adaptive_periods'])
            elif current_volatility < avg_volatility * 0.7:
                # Low volatility - use longer period
                optimal_period = max(self.parameters['adaptive_periods'])
            else:
                # Normal volatility - use base period
                optimal_period = self.parameters['base_period']
            
            # Machine learning optimization (simplified)
            period_performance = {}
            
            for period in self.parameters['adaptive_periods']:
                cci_values = self._calculate_cci(typical_prices, period)
                
                if len(cci_values) >= 20:
                    # Calculate signal quality metrics
                    extremes = np.abs(cci_values) > 100
                    extreme_ratio = np.sum(extremes) / len(cci_values)
                    
                    # Signal persistence
                    signal_changes = np.sum(np.diff(np.sign(cci_values)) != 0)
                    persistence = 1.0 - (signal_changes / len(cci_values))
                    
                    # Trend alignment
                    price_trend = np.polyfit(range(len(typical_prices[-len(cci_values):])), 
                                           typical_prices[-len(cci_values):], 1)[0]
                    cci_trend = np.polyfit(range(len(cci_values)), cci_values, 1)[0]
                    alignment = abs(np.corrcoef([price_trend], [cci_trend])[0, 1]) if cci_trend != 0 else 0
                    
                    # Combined score
                    score = (extreme_ratio * 0.3 + persistence * 0.4 + alignment * 0.3)
                    period_performance[period] = score
            
            if period_performance:
                ml_optimal = max(period_performance, key=period_performance.get)
                # Combine volatility-based and ML-based optimization
                optimal_period = int((optimal_period + ml_optimal) / 2)
            
            return optimal_period
            
        except Exception:
            return self.parameters['base_period']
    
    def _calculate_adaptive_thresholds(self, cci_values: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate adaptive overbought/oversold thresholds."""
        try:
            if len(cci_values) < 20:
                return {
                    'overbought': self.parameters['overbought_level'],
                    'oversold': self.parameters['oversold_level']
                }
            
            # Historical threshold analysis
            recent_cci = cci_values[-50:] if len(cci_values) >= 50 else cci_values
            
            # Volatility-based adjustment
            cci_volatility = np.std(recent_cci)
            base_threshold = 100
            
            # Adjust thresholds based on volatility
            volatility_multiplier = max(0.5, min(2.0, cci_volatility / 50))
            
            # Volume-weighted adjustment
            if len(volumes) >= len(recent_cci):
                recent_volumes = volumes[-len(recent_cci):]
                volume_weights = recent_volumes / np.mean(recent_volumes)
                
                # Higher volume during extremes suggests stronger thresholds
                extreme_volume_ratio = np.mean(volume_weights[np.abs(recent_cci) > base_threshold]) if np.any(np.abs(recent_cci) > base_threshold) else 1.0
                volume_adjustment = min(1.5, max(0.7, extreme_volume_ratio))
            else:
                volume_adjustment = 1.0
            
            # Market regime adjustment
            extreme_frequency = np.sum(np.abs(recent_cci) > base_threshold) / len(recent_cci)
            
            if extreme_frequency > 0.3:
                # Too many extremes - increase thresholds
                regime_adjustment = 1.2
            elif extreme_frequency < 0.1:
                # Too few extremes - decrease thresholds
                regime_adjustment = 0.8
            else:
                regime_adjustment = 1.0
            
            # Combined adaptive thresholds
            adaptive_multiplier = volatility_multiplier * volume_adjustment * regime_adjustment
            
            overbought = base_threshold * adaptive_multiplier
            oversold = -base_threshold * adaptive_multiplier
            
            # Ensure reasonable bounds
            overbought = np.clip(overbought, 80, 200)
            oversold = np.clip(oversold, -200, -80)
            
            return {
                'overbought': float(overbought),
                'oversold': float(oversold),
                'volatility_multiplier': float(volatility_multiplier),
                'volume_adjustment': float(volume_adjustment),
                'regime_adjustment': float(regime_adjustment)
            }
            
        except Exception:
            return {
                'overbought': self.parameters['overbought_level'],
                'oversold': self.parameters['oversold_level']
            }
    
    def _multi_timeframe_analysis(self, cci_analysis: Dict) -> Dict[str, Any]:
        """Analyze CCI across multiple timeframes for confluence."""
        try:
            if not cci_analysis or 'consensus' not in cci_analysis:
                return {}
            
            # Extract signals from different periods
            period_signals = {}
            
            for key, data in cci_analysis.items():
                if key.startswith('period_'):
                    period = int(key.split('_')[1])
                    current_value = data['current_value']
                    trend_direction = data['trend_direction']
                    
                    # Classify signal
                    if current_value > 100:
                        signal = 'overbought'
                    elif current_value < -100:
                        signal = 'oversold'
                    elif current_value > 0:
                        signal = 'bullish'
                    else:
                        signal = 'bearish'
                    
                    period_signals[period] = {
                        'signal': signal,
                        'strength': abs(current_value) / 100,
                        'trend': trend_direction
                    }
            
            if not period_signals:
                return {}
            
            # Confluence analysis
            signal_counts = {}
            trend_agreement = []
            strength_levels = []
            
            for period_data in period_signals.values():
                signal = period_data['signal']
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
                trend_agreement.append(period_data['trend'])
                strength_levels.append(period_data['strength'])
            
            # Dominant signal
            dominant_signal = max(signal_counts, key=signal_counts.get) if signal_counts else 'neutral'
            signal_confluence = signal_counts.get(dominant_signal, 0) / len(period_signals)
            
            # Trend agreement
            trend_consensus = np.mean(trend_agreement) if trend_agreement else 0
            
            # Average strength
            avg_strength = np.mean(strength_levels) if strength_levels else 0
            
            # Timeframe scoring
            short_term_score = np.mean([data['strength'] * data['trend'] 
                                      for period, data in period_signals.items() 
                                      if period <= 14])
            
            long_term_score = np.mean([data['strength'] * data['trend'] 
                                     for period, data in period_signals.items() 
                                     if period > 14])
            
            return {
                'dominant_signal': dominant_signal,
                'signal_confluence': float(signal_confluence),
                'trend_consensus': float(trend_consensus),
                'average_strength': float(avg_strength),
                'short_term_score': float(short_term_score),
                'long_term_score': float(long_term_score),
                'timeframe_alignment': float(abs(short_term_score - long_term_score) < 0.5),
                'period_signals': period_signals
            }
            
        except Exception:
            return {}
    
    def _detect_divergences(self, cci_values: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Detect bullish and bearish divergences."""
        try:
            if len(cci_values) < self.parameters['divergence_window'] or len(prices) < len(cci_values):
                return {}
            
            # Align data
            aligned_prices = prices[-len(cci_values):]
            window = self.parameters['divergence_window']
            
            divergences = []
            
            # Look for divergences in recent periods
            for i in range(window, len(cci_values)):
                # Find local extremes
                cci_window = cci_values[i-window:i+1]
                price_window = aligned_prices[i-window:i+1]
                
                # Find peaks and troughs
                cci_peaks = self._find_local_extremes(cci_window, 'peaks')
                cci_troughs = self._find_local_extremes(cci_window, 'troughs')
                price_peaks = self._find_local_extremes(price_window, 'peaks')
                price_troughs = self._find_local_extremes(price_window, 'troughs')
                
                # Check for bullish divergence (price lower lows, CCI higher lows)
                if len(cci_troughs) >= 2 and len(price_troughs) >= 2:
                    last_cci_trough = cci_troughs[-1]
                    prev_cci_trough = cci_troughs[-2]
                    last_price_trough = price_troughs[-1]
                    prev_price_trough = price_troughs[-2]
                    
                    if (cci_window[last_cci_trough] > cci_window[prev_cci_trough] and
                        price_window[last_price_trough] < price_window[prev_price_trough]):
                        
                        divergences.append({
                            'type': 'bullish',
                            'strength': abs(cci_window[last_cci_trough] - cci_window[prev_cci_trough]),
                            'position': i,
                            'confidence': 0.8
                        })
                
                # Check for bearish divergence (price higher highs, CCI lower highs)
                if len(cci_peaks) >= 2 and len(price_peaks) >= 2:
                    last_cci_peak = cci_peaks[-1]
                    prev_cci_peak = cci_peaks[-2]
                    last_price_peak = price_peaks[-1]
                    prev_price_peak = price_peaks[-2]
                    
                    if (cci_window[last_cci_peak] < cci_window[prev_cci_peak] and
                        price_window[last_price_peak] > price_window[prev_price_peak]):
                        
                        divergences.append({
                            'type': 'bearish',
                            'strength': abs(cci_window[last_cci_peak] - cci_window[prev_cci_peak]),
                            'position': i,
                            'confidence': 0.8
                        })
            
            # Analyze recent divergences
            recent_divergences = [div for div in divergences if div['position'] >= len(cci_values) - 5]
            
            # Classify current divergence state
            if recent_divergences:
                latest_divergence = max(recent_divergences, key=lambda x: x['position'])
                divergence_type = latest_divergence['type']
                divergence_strength = latest_divergence['strength']
            else:
                divergence_type = 'none'
                divergence_strength = 0
            
            return {
                'current_divergence': divergence_type,
                'divergence_strength': float(divergence_strength),
                'recent_divergences': recent_divergences,
                'total_divergences': len(divergences),
                'bullish_divergences': len([d for d in divergences if d['type'] == 'bullish']),
                'bearish_divergences': len([d for d in divergences if d['type'] == 'bearish'])
            }
            
        except Exception:
            return {}
    
    def _find_local_extremes(self, data: np.ndarray, extreme_type: str) -> List[int]:
        """Find local peaks or troughs in data."""
        extremes = []
        
        if len(data) < 3:
            return extremes
        
        for i in range(1, len(data) - 1):
            if extreme_type == 'peaks':
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    extremes.append(i)
            elif extreme_type == 'troughs':
                if data[i] < data[i-1] and data[i] < data[i+1]:
                    extremes.append(i)
        
        return extremes
    
    def _analyze_cci_patterns(self, cci_values: np.ndarray) -> Dict[str, Any]:
        """Analyze CCI patterns and formations."""
        try:
            if len(cci_values) < 10:
                return {}
            
            recent_cci = cci_values[-10:]
            
            # Pattern detection
            patterns = {}
            
            # Double top/bottom patterns
            patterns['double_top'] = self._detect_double_pattern(recent_cci, 'top')
            patterns['double_bottom'] = self._detect_double_pattern(recent_cci, 'bottom')
            
            # Trend patterns
            patterns['uptrend'] = self._detect_trend_pattern(recent_cci, 'up')
            patterns['downtrend'] = self._detect_trend_pattern(recent_cci, 'down')
            
            # Channel patterns
            patterns['channel'] = self._detect_channel_pattern(recent_cci)
            
            # Momentum patterns
            patterns['momentum_buildup'] = self._detect_momentum_pattern(recent_cci)
            
            # Current pattern strength
            pattern_strengths = [p.get('strength', 0) for p in patterns.values() if isinstance(p, dict)]
            overall_pattern_strength = max(pattern_strengths) if pattern_strengths else 0
            
            # Dominant pattern
            dominant_pattern = 'none'
            if pattern_strengths:
                pattern_names = list(patterns.keys())
                max_idx = np.argmax(pattern_strengths)
                if pattern_strengths[max_idx] > 0.5:
                    dominant_pattern = pattern_names[max_idx]
            
            return {
                'patterns': patterns,
                'dominant_pattern': dominant_pattern,
                'pattern_strength': float(overall_pattern_strength),
                'pattern_confidence': float(min(overall_pattern_strength * 2, 1.0))
            }
            
        except Exception:
            return {}
    
    def _detect_double_pattern(self, data: np.ndarray, pattern_type: str) -> Dict[str, Any]:
        """Detect double top or double bottom patterns."""
        if len(data) < 6:
            return {'detected': False, 'strength': 0}
        
        try:
            if pattern_type == 'top':
                # Find two highest points
                peaks = self._find_local_extremes(data, 'peaks')
                if len(peaks) >= 2:
                    last_two_peaks = peaks[-2:]
                    peak_values = [data[p] for p in last_two_peaks]
                    
                    # Check if peaks are similar height
                    height_similarity = 1 - abs(peak_values[0] - peak_values[1]) / (max(peak_values) + 1e-8)
                    
                    # Check if both peaks are in overbought territory
                    overbought_condition = all(v > 80 for v in peak_values)
                    
                    strength = height_similarity * (1 if overbought_condition else 0.7)
                    
                    return {
                        'detected': strength > 0.6,
                        'strength': float(strength),
                        'peaks': last_two_peaks,
                        'values': peak_values
                    }
            
            elif pattern_type == 'bottom':
                # Find two lowest points
                troughs = self._find_local_extremes(data, 'troughs')
                if len(troughs) >= 2:
                    last_two_troughs = troughs[-2:]
                    trough_values = [data[t] for t in last_two_troughs]
                    
                    # Check if troughs are similar depth
                    depth_similarity = 1 - abs(trough_values[0] - trough_values[1]) / (abs(min(trough_values)) + 1e-8)
                    
                    # Check if both troughs are in oversold territory
                    oversold_condition = all(v < -80 for v in trough_values)
                    
                    strength = depth_similarity * (1 if oversold_condition else 0.7)
                    
                    return {
                        'detected': strength > 0.6,
                        'strength': float(strength),
                        'troughs': last_two_troughs,
                        'values': trough_values
                    }
            
            return {'detected': False, 'strength': 0}
            
        except Exception:
            return {'detected': False, 'strength': 0}
    
    def _detect_trend_pattern(self, data: np.ndarray, trend_type: str) -> Dict[str, Any]:
        """Detect trend patterns in CCI."""
        if len(data) < 5:
            return {'detected': False, 'strength': 0}
        
        try:
            # Linear regression for trend
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((data - y_pred) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            if trend_type == 'up':
                strength = max(0, slope / 10) * r_squared if slope > 0 else 0
            else:  # down
                strength = max(0, -slope / 10) * r_squared if slope < 0 else 0
            
            return {
                'detected': strength > 0.3,
                'strength': float(min(strength, 1.0)),
                'slope': float(slope),
                'r_squared': float(r_squared)
            }
            
        except Exception:
            return {'detected': False, 'strength': 0}
    
    def _detect_channel_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect channel (sideways) patterns."""
        if len(data) < 5:
            return {'detected': False, 'strength': 0}
        
        try:
            # Check for sideways movement
            data_range = np.max(data) - np.min(data)
            data_std = np.std(data)
            
            # Channel strength based on low volatility and range
            range_threshold = 50  # CCI range threshold for channel
            volatility_ratio = data_std / (abs(np.mean(data)) + 1e-8)
            
            if data_range < range_threshold and volatility_ratio < 0.3:
                strength = 1 - (data_range / range_threshold) * (1 - volatility_ratio)
            else:
                strength = 0
            
            return {
                'detected': strength > 0.5,
                'strength': float(min(strength, 1.0)),
                'range': float(data_range),
                'volatility_ratio': float(volatility_ratio)
            }
            
        except Exception:
            return {'detected': False, 'strength': 0}
    
    def _detect_momentum_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect momentum buildup patterns."""
        if len(data) < 5:
            return {'detected': False, 'strength': 0}
        
        try:
            # Calculate momentum acceleration
            momentum = np.diff(data)
            acceleration = np.diff(momentum) if len(momentum) > 1 else np.array([0])
            
            # Recent momentum strength
            recent_momentum = np.mean(momentum[-3:]) if len(momentum) >= 3 else 0
            recent_acceleration = np.mean(acceleration[-2:]) if len(acceleration) >= 2 else 0
            
            # Momentum buildup occurs when acceleration and momentum align
            momentum_strength = abs(recent_momentum) / 20  # Normalize
            acceleration_strength = abs(recent_acceleration) / 10  # Normalize
            
            # Alignment check
            alignment = 1 if np.sign(recent_momentum) == np.sign(recent_acceleration) else 0.5
            
            strength = momentum_strength * acceleration_strength * alignment
            
            return {
                'detected': strength > 0.4,
                'strength': float(min(strength, 1.0)),
                'momentum': float(recent_momentum),
                'acceleration': float(recent_acceleration),
                'direction': 'bullish' if recent_momentum > 0 else 'bearish'
            }
            
        except Exception:
            return {'detected': False, 'strength': 0}
    
    def _analyze_trend_strength(self, cci_values: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze trend strength using CCI."""
        try:
            if len(cci_values) < self.parameters['trend_confirmation_period']:
                return {}
            
            # Recent CCI trend
            recent_period = self.parameters['trend_confirmation_period']
            recent_cci = cci_values[-recent_period:]
            
            # CCI trend analysis
            cci_slope, _ = np.polyfit(range(len(recent_cci)), recent_cci, 1)
            cci_trend_strength = abs(cci_slope) / 10  # Normalize
            
            # Price trend analysis (aligned with CCI)
            if len(prices) >= len(cci_values):
                aligned_prices = prices[-len(cci_values):]
                recent_prices = aligned_prices[-recent_period:]
                price_slope, _ = np.polyfit(range(len(recent_prices)), recent_prices, 1)
                
                # Trend alignment
                cci_direction = np.sign(cci_slope)
                price_direction = np.sign(price_slope)
                trend_alignment = 1 if cci_direction == price_direction else -1
            else:
                trend_alignment = 0
                price_slope = 0
            
            # CCI momentum analysis
            cci_momentum = cci_values[-1] - cci_values[-recent_period] if len(cci_values) >= recent_period else 0
            momentum_strength = abs(cci_momentum) / 50  # Normalize
            
            # Overall trend strength
            trend_strength = min(cci_trend_strength * abs(trend_alignment) * (1 + momentum_strength), 1.0)
            
            # Trend classification
            if trend_strength > 0.6 and cci_slope > 0:
                trend_type = 'strong_bullish'
            elif trend_strength > 0.6 and cci_slope < 0:
                trend_type = 'strong_bearish'
            elif trend_strength > 0.3:
                trend_type = 'moderate_' + ('bullish' if cci_slope > 0 else 'bearish')
            else:
                trend_type = 'weak'
            
            return {
                'trend_strength': float(trend_strength),
                'trend_type': trend_type,
                'trend_alignment': float(trend_alignment),
                'cci_slope': float(cci_slope),
                'momentum_strength': float(momentum_strength),
                'momentum_value': float(cci_momentum)
            }
            
        except Exception:
            return {}
    
    def _analyze_mean_reversion(self, cci_values: np.ndarray, typical_prices: np.ndarray) -> Dict[str, Any]:
        """Analyze mean reversion probability."""
        try:
            if len(cci_values) < 20:
                return {}
            
            current_cci = cci_values[-1]
            recent_cci = cci_values[-20:]
            
            # Extreme level analysis
            extreme_threshold = 150  # More extreme than standard 100
            is_extreme = abs(current_cci) > extreme_threshold
            extreme_level = abs(current_cci) / extreme_threshold if is_extreme else 0
            
            # Historical mean reversion analysis
            extreme_periods = np.where(np.abs(recent_cci) > 100)[0]
            reversion_success = 0
            
            if len(extreme_periods) > 0:
                reversion_count = 0
                total_extremes = 0
                
                for period in extreme_periods[:-1]:  # Exclude current period
                    if period + 5 < len(recent_cci):  # Check 5 periods ahead
                        initial_extreme = recent_cci[period]
                        future_values = recent_cci[period+1:period+6]
                        
                        # Check if CCI moved toward zero
                        if initial_extreme > 0:
                            reversion = any(v < initial_extreme * 0.7 for v in future_values)
                        else:
                            reversion = any(v > initial_extreme * 0.7 for v in future_values)
                        
                        if reversion:
                            reversion_count += 1
                        total_extremes += 1
                
                reversion_success = reversion_count / total_extremes if total_extremes > 0 else 0.5
            
            # Price momentum vs CCI divergence
            if len(typical_prices) >= len(cci_values):
                recent_prices = typical_prices[-len(recent_cci):]
                price_momentum = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
                cci_momentum = (recent_cci[-1] - recent_cci[-5]) / 100 if len(recent_cci) >= 5 else 0
                
                # Divergence suggests mean reversion opportunity
                momentum_divergence = abs(price_momentum - cci_momentum)
            else:
                momentum_divergence = 0
            
            # Mean reversion probability calculation
            base_probability = reversion_success
            extreme_boost = extreme_level * 0.3 if is_extreme else 0
            divergence_boost = min(momentum_divergence * 2, 0.3)
            
            mean_reversion_probability = min(base_probability + extreme_boost + divergence_boost, 1.0)
            
            # Time to reversion estimate
            if is_extreme and reversion_success > 0.5:
                time_to_reversion = max(2, int(extreme_level * 5))
            else:
                time_to_reversion = 0
            
            return {
                'mean_reversion_probability': float(mean_reversion_probability),
                'is_extreme': is_extreme,
                'extreme_level': float(extreme_level),
                'historical_reversion_rate': float(reversion_success),
                'momentum_divergence': float(momentum_divergence),
                'time_to_reversion': int(time_to_reversion),
                'reversion_signal': mean_reversion_probability > 0.7
            }
            
        except Exception:
            return {}
    
    def _analyze_volatility_context(self, cci_values: np.ndarray, typical_prices: np.ndarray, 
                                  volumes: np.ndarray) -> Dict[str, Any]:
        """Analyze CCI in the context of price and volume volatility."""
        try:
            if len(cci_values) < self.parameters['volatility_window']:
                return {}
            
            vol_window = self.parameters['volatility_window']
            
            # CCI volatility
            recent_cci = cci_values[-vol_window:]
            cci_volatility = np.std(recent_cci)
            cci_avg_volatility = np.mean([np.std(cci_values[i:i+vol_window]) 
                                        for i in range(len(cci_values)-vol_window)])
            
            # Price volatility
            if len(typical_prices) >= vol_window:
                price_returns = np.diff(np.log(typical_prices[-vol_window:] + 1e-8))
                price_volatility = np.std(price_returns) * np.sqrt(252)  # Annualized
                
                # Historical price volatility
                price_vol_history = []
                for i in range(vol_window, len(typical_prices)):
                    window_returns = np.diff(np.log(typical_prices[i-vol_window:i] + 1e-8))
                    price_vol_history.append(np.std(window_returns))
                
                avg_price_volatility = np.mean(price_vol_history) * np.sqrt(252) if price_vol_history else price_volatility
            else:
                price_volatility = 0
                avg_price_volatility = 0
            
            # Volume volatility
            if len(volumes) >= vol_window:
                recent_volumes = volumes[-vol_window:]
                volume_volatility = np.std(recent_volumes) / np.mean(recent_volumes)
                
                volume_vol_history = []
                for i in range(vol_window, len(volumes)):
                    window_volumes = volumes[i-vol_window:i]
                    vol_cv = np.std(window_volumes) / np.mean(window_volumes)
                    volume_vol_history.append(vol_cv)
                
                avg_volume_volatility = np.mean(volume_vol_history) if volume_vol_history else volume_volatility
            else:
                volume_volatility = 0
                avg_volume_volatility = 0
            
            # Volatility regime classification
            cci_vol_ratio = cci_volatility / (cci_avg_volatility + 1e-8)
            price_vol_ratio = price_volatility / (avg_price_volatility + 1e-8)
            volume_vol_ratio = volume_volatility / (avg_volume_volatility + 1e-8)
            
            # Combined volatility regime
            avg_vol_ratio = np.mean([cci_vol_ratio, price_vol_ratio, volume_vol_ratio])
            
            if avg_vol_ratio > 1.5:
                volatility_regime = 'high'
            elif avg_vol_ratio < 0.7:
                volatility_regime = 'low'
            else:
                volatility_regime = 'normal'
            
            # Volatility-adjusted signal strength
            if volatility_regime == 'high':
                volatility_adjustment = 0.8  # Reduce signal strength in high volatility
            elif volatility_regime == 'low':
                volatility_adjustment = 1.2  # Increase signal strength in low volatility
            else:
                volatility_adjustment = 1.0
            
            return {
                'volatility_regime': volatility_regime,
                'cci_volatility': float(cci_volatility),
                'cci_vol_ratio': float(cci_vol_ratio),
                'price_volatility': float(price_volatility),
                'price_vol_ratio': float(price_vol_ratio),
                'volume_volatility': float(volume_volatility),
                'volume_vol_ratio': float(volume_vol_ratio),
                'volatility_adjustment': float(volatility_adjustment),
                'combined_vol_ratio': float(avg_vol_ratio)
            }
            
        except Exception:
            return {}
    
    def _classify_market_state(self, cci_values: np.ndarray, thresholds: Dict) -> str:
        """Classify current market state based on CCI."""
        if len(cci_values) == 0:
            return 'neutral'
        
        current_cci = cci_values[-1]
        overbought = thresholds.get('overbought', 100)
        oversold = thresholds.get('oversold', -100)
        
        if current_cci > overbought:
            return 'overbought'
        elif current_cci < oversold:
            return 'oversold'
        elif current_cci > 50:
            return 'bullish'
        elif current_cci < -50:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_signal_strength(self, cci_values: np.ndarray, thresholds: Dict,
                                 mtf_analysis: Dict, divergence_analysis: Dict,
                                 pattern_analysis: Dict, trend_analysis: Dict,
                                 mean_reversion: Dict, volatility_analysis: Dict) -> float:
        """Calculate overall signal strength."""
        if len(cci_values) == 0:
            return 0.0
        
        signal_components = []
        current_cci = cci_values[-1]
        
        # Basic CCI signal
        overbought = thresholds.get('overbought', 100)
        oversold = thresholds.get('oversold', -100)
        
        if current_cci > overbought:
            basic_signal = -min((current_cci - overbought) / overbought, 1.0)  # Negative for sell
        elif current_cci < oversold:
            basic_signal = min((oversold - current_cci) / abs(oversold), 1.0)  # Positive for buy
        else:
            basic_signal = current_cci / 100  # Moderate signal
        
        signal_components.append(basic_signal * 0.3)
        
        # Multi-timeframe confirmation
        if mtf_analysis:
            mtf_score = mtf_analysis.get('short_term_score', 0) * 0.6 + mtf_analysis.get('long_term_score', 0) * 0.4
            confluence = mtf_analysis.get('signal_confluence', 0)
            mtf_signal = mtf_score * confluence
            signal_components.append(mtf_signal * 0.25)
        
        # Divergence signal
        if divergence_analysis:
            divergence_type = divergence_analysis.get('current_divergence', 'none')
            divergence_strength = divergence_analysis.get('divergence_strength', 0)
            
            if divergence_type == 'bullish':
                divergence_signal = divergence_strength / 50
            elif divergence_type == 'bearish':
                divergence_signal = -divergence_strength / 50
            else:
                divergence_signal = 0
            
            signal_components.append(divergence_signal * 0.2)
        
        # Pattern signal
        if pattern_analysis:
            pattern = pattern_analysis.get('dominant_pattern', 'none')
            pattern_strength = pattern_analysis.get('pattern_strength', 0)
            
            pattern_signals = {
                'double_bottom': 0.4,
                'double_top': -0.4,
                'uptrend': 0.3,
                'downtrend': -0.3,
                'momentum_buildup': 0.2
            }
            
            pattern_signal = pattern_signals.get(pattern, 0) * pattern_strength
            signal_components.append(pattern_signal * 0.15)
        
        # Mean reversion signal
        if mean_reversion:
            if mean_reversion.get('reversion_signal', False):
                reversion_prob = mean_reversion.get('mean_reversion_probability', 0)
                # Contrarian signal for mean reversion
                reversion_signal = -np.sign(current_cci) * reversion_prob * 0.5
                signal_components.append(reversion_signal * 0.1)
        
        # Trend confirmation
        if trend_analysis:
            trend_strength = trend_analysis.get('trend_strength', 0)
            trend_alignment = trend_analysis.get('trend_alignment', 0)
            trend_signal = trend_strength * trend_alignment * np.sign(current_cci)
            signal_components.append(trend_signal * 0.1)
        
        # Volatility adjustment
        volatility_adjustment = 1.0
        if volatility_analysis:
            volatility_adjustment = volatility_analysis.get('volatility_adjustment', 1.0)
        
        # Combine signals
        total_signal = np.sum(signal_components) * volatility_adjustment
        
        return float(np.clip(total_signal, -1, 1))
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on CCI analysis."""
        signal_strength = value['signal_strength']
        market_state = value['market_state']
        
        # Base confidence from signal strength
        confidence = min(abs(signal_strength), 1.0)
        
        # Adjust confidence based on market state
        if market_state in ['overbought', 'oversold']:
            confidence *= 1.2  # Higher confidence in extreme conditions
        elif market_state == 'neutral':
            confidence *= 0.7  # Lower confidence in neutral conditions
        
        # Ensure confidence is within bounds
        confidence = min(confidence, 1.0)
        
        # Generate signal based on strength
        if signal_strength > 0.6:
            return SignalType.STRONG_BUY, confidence
        elif signal_strength > 0.3:
            return SignalType.BUY, confidence
        elif signal_strength < -0.6:
            return SignalType.STRONG_SELL, confidence
        elif signal_strength < -0.3:
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, confidence
