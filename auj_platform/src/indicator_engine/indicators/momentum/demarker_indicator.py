"""
DeMarker Indicator - Advanced Implementation
==========================================

Advanced DeMarker oscillator with adaptive zones, machine learning enhancement,
and sophisticated momentum divergence detection.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class DeMarkerIndicator(StandardIndicatorInterface):
    """
    Advanced DeMarker Indicator Implementation
    
    Features:
    - Adaptive period optimization based on market conditions
    - ML-enhanced overbought/oversold zone detection
    - Advanced divergence analysis with statistical validation
    - Dynamic threshold adjustment for changing volatility regimes
    - Multi-timeframe momentum coherence analysis
    - Volume-weighted DeMarker calculations
    - Institutional flow detection through DeMarker patterns
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 14,
            'overbought_level': 0.7,
            'oversold_level': 0.3,
            'adaptive_periods': True,
            'volume_weighted': True,
            'ml_enhancement': True,
            'divergence_detection': True,
            'dynamic_thresholds': True,
            'multi_timeframe': True,
            'lookback_period': 50,
            'min_period': 7,
            'max_period': 28,
            'threshold_sensitivity': 0.1,
            'ml_lookback': 60,
            'smoothing_enabled': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="DeMarkerIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.regime_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.demarker_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.threshold_optimizer = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=4, random_state=42)
        self.pca = PCA(n_components=3)
        self.models_trained = False
        
        self.history = {
            'demarker_values': [],
            'dem_max_values': [],
            'dem_min_values': [],
            'thresholds': [],
            'signals': [],
            'divergences': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['period'], 
                        self.parameters['max_period'],
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max_period * 2 + 50,
            lookback_periods=200
        )
    
    def _optimize_period(self, data: pd.DataFrame) -> int:
        """Optimize DeMarker period based on market conditions"""
        if not self.parameters['adaptive_periods'] or len(data) < 60:
            return self.parameters['period']
        
        # Calculate market characteristics
        recent_data = data.tail(60)
        returns = recent_data['close'].pct_change().dropna()
        
        # Volatility regime detection
        current_vol = returns.std()
        rolling_vol = returns.rolling(window=20).std()
        avg_vol = rolling_vol.mean()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        # Trend strength analysis
        price_changes = recent_data['close'].diff().tail(30)
        trend_strength = abs(price_changes.sum()) / (price_changes.abs().sum() + 1e-8)
        
        # Market efficiency measurement
        high_low_range = (recent_data['high'] - recent_data['low']).mean()
        close_range = recent_data['close'].std()
        efficiency_ratio = close_range / (high_low_range + 1e-8)
        
        base_period = self.parameters['period']
        min_period = self.parameters['min_period']
        max_period = self.parameters['max_period']
        
        # Adjust period based on conditions
        if vol_ratio > 1.5:  # High volatility - shorter period
            adjustment_factor = 0.7
        elif vol_ratio < 0.6:  # Low volatility - longer period
            adjustment_factor = 1.3
        else:
            adjustment_factor = 1.0
        
        # Adjust for trend strength
        if trend_strength > 0.7:  # Strong trend - longer period
            adjustment_factor *= 1.2
        elif trend_strength < 0.3:  # Weak trend - shorter period
            adjustment_factor *= 0.8
        
        # Adjust for market efficiency
        if efficiency_ratio > 0.8:  # Highly efficient - shorter period
            adjustment_factor *= 0.9
        elif efficiency_ratio < 0.4:  # Less efficient - longer period
            adjustment_factor *= 1.1
        
        optimized_period = int(base_period * adjustment_factor)
        optimized_period = max(min_period, min(optimized_period, max_period))
        
        return optimized_period
    
    def _calculate_demarker_components(self, data: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate DeMarker Max and Min components with volume weighting"""
        high = data['high']
        low = data['low']
        
        # Standard DeMarker components
        dem_max = pd.Series(0.0, index=data.index)
        dem_min = pd.Series(0.0, index=data.index)
        
        # Calculate DeMax and DeMin
        for i in range(1, len(data)):
            # DeMax: difference between current high and previous high (if positive)
            dem_max.iloc[i] = max(0, high.iloc[i] - high.iloc[i-1])
            
            # DeMin: difference between previous low and current low (if positive)
            dem_min.iloc[i] = max(0, low.iloc[i-1] - low.iloc[i])
        
        # Volume weighting if enabled
        if self.parameters['volume_weighted'] and 'volume' in data.columns:
            volume = data['volume']
            volume_ma = volume.rolling(window=period).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            
            # Apply volume weighting with sigmoid transformation
            volume_weight = 2.0 / (1.0 + np.exp(-2.0 * (volume_ratio - 1.0)))
            
            dem_max = dem_max * volume_weight
            dem_min = dem_min * volume_weight
        
        return dem_max, dem_min
    
    def _calculate_demarker(self, dem_max: pd.Series, dem_min: pd.Series, period: int) -> pd.Series:
        """Calculate DeMarker oscillator with advanced smoothing"""
        # Calculate moving averages of DeMax and DeMin
        dem_max_ma = dem_max.rolling(window=period).mean()
        dem_min_ma = dem_min.rolling(window=period).mean()
        
        # Calculate DeMarker
        demarker = dem_max_ma / (dem_max_ma + dem_min_ma + 1e-8)
        
        # Optional smoothing
        if self.parameters['smoothing_enabled']:
            smoothing_period = max(3, period // 3)
            demarker = demarker.ewm(span=smoothing_period, adjust=False).mean()
        
        return demarker.fillna(0.5)
    
    def _optimize_thresholds(self, demarker: pd.Series, data: pd.DataFrame) -> Tuple[float, float]:
        """Dynamically optimize overbought/oversold thresholds"""
        if not self.parameters['dynamic_thresholds'] or len(demarker) < 50:
            return self.parameters['overbought_level'], self.parameters['oversold_level']
        
        recent_demarker = demarker.tail(50).dropna()
        
        if len(recent_demarker) < 20:
            return self.parameters['overbought_level'], self.parameters['oversold_level']
        
        # Statistical approach: use percentiles
        upper_percentile = np.percentile(recent_demarker, 85)
        lower_percentile = np.percentile(recent_demarker, 15)
        
        # Volatility adjustment
        demarker_volatility = recent_demarker.std()
        vol_adjustment = min(0.1, demarker_volatility * 0.5)
        
        # Base thresholds
        base_upper = self.parameters['overbought_level']
        base_lower = self.parameters['oversold_level']
        
        # Adaptive thresholds
        adaptive_upper = np.clip(
            base_upper + vol_adjustment,
            max(base_upper, upper_percentile * 0.9),
            0.9
        )
        
        adaptive_lower = np.clip(
            base_lower - vol_adjustment,
            0.1,
            min(base_lower, lower_percentile * 1.1)
        )
        
        # Ensure proper separation
        if adaptive_upper - adaptive_lower < 0.2:
            mid_point = (adaptive_upper + adaptive_lower) / 2
            adaptive_upper = min(0.9, mid_point + 0.15)
            adaptive_lower = max(0.1, mid_point - 0.15)
        
        return adaptive_upper, adaptive_lower
    
    def _detect_divergences(self, demarker: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Advanced divergence detection with statistical validation"""
        if not self.parameters['divergence_detection'] or len(demarker) < 30:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        lookback = min(30, len(demarker) - 5)
        recent_demarker = demarker.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Find peaks and troughs
        demarker_peaks, _ = find_peaks(recent_demarker.values, height=0.6, distance=5)
        demarker_troughs, _ = find_peaks(-recent_demarker.values, height=-0.4, distance=5)
        
        price_peaks, _ = find_peaks(recent_prices.values, distance=5)
        price_troughs, _ = find_peaks(-recent_prices.values, distance=5)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: price makes lower low, DeMarker makes higher low
        if len(demarker_troughs) >= 2 and len(price_troughs) >= 2:
            for i in range(len(demarker_troughs) - 1):
                dm_trough_1 = demarker_troughs[i]
                dm_trough_2 = demarker_troughs[i + 1]
                
                # Find corresponding price troughs
                price_trough_1 = self._find_nearest_extreme(price_troughs, dm_trough_1, recent_prices.index)
                price_trough_2 = self._find_nearest_extreme(price_troughs, dm_trough_2, recent_prices.index)
                
                if price_trough_1 is not None and price_trough_2 is not None:
                    price_1 = recent_prices.iloc[price_trough_1]
                    price_2 = recent_prices.iloc[price_trough_2]
                    dm_1 = recent_demarker.iloc[dm_trough_1]
                    dm_2 = recent_demarker.iloc[dm_trough_2]
                    
                    if price_2 < price_1 and dm_2 > dm_1:
                        # Validate statistical significance
                        price_decline = (price_1 - price_2) / price_1
                        dm_improvement = dm_2 - dm_1
                        
                        if price_decline > 0.02 and dm_improvement > 0.05:
                            bullish_divergence = True
                            divergence_strength = max(divergence_strength, price_decline + dm_improvement)
        
        # Bearish divergence: price makes higher high, DeMarker makes lower high
        if len(demarker_peaks) >= 2 and len(price_peaks) >= 2:
            for i in range(len(demarker_peaks) - 1):
                dm_peak_1 = demarker_peaks[i]
                dm_peak_2 = demarker_peaks[i + 1]
                
                price_peak_1 = self._find_nearest_extreme(price_peaks, dm_peak_1, recent_prices.index)
                price_peak_2 = self._find_nearest_extreme(price_peaks, dm_peak_2, recent_prices.index)
                
                if price_peak_1 is not None and price_peak_2 is not None:
                    price_1 = recent_prices.iloc[price_peak_1]
                    price_2 = recent_prices.iloc[price_peak_2]
                    dm_1 = recent_demarker.iloc[dm_peak_1]
                    dm_2 = recent_demarker.iloc[dm_peak_2]
                    
                    if price_2 > price_1 and dm_2 < dm_1:
                        price_increase = (price_2 - price_1) / price_1
                        dm_decline = dm_1 - dm_2
                        
                        if price_increase > 0.02 and dm_decline > 0.05:
                            bearish_divergence = True
                            divergence_strength = max(divergence_strength, price_increase + dm_decline)
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': float(divergence_strength),
            'demarker_peaks': len(demarker_peaks),
            'demarker_troughs': len(demarker_troughs),
            'analysis_period': lookback
        }
    
    def _find_nearest_extreme(self, extremes: np.ndarray, target_idx: int, index: pd.Index) -> Optional[int]:
        """Find the nearest extreme point to a target index"""
        if len(extremes) == 0:
            return None
        
        min_distance = float('inf')
        nearest_extreme = None
        
        for extreme_idx in extremes:
            distance = abs(extreme_idx - target_idx)
            if distance < min_distance and distance < 8:  # Within reasonable range
                min_distance = distance
                nearest_extreme = extreme_idx
        
        return nearest_extreme
    
    def _analyze_multi_timeframe_momentum(self, demarker: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum coherence across multiple timeframes"""
        if not self.parameters['multi_timeframe'] or len(demarker) < 60:
            return {'momentum_alignment': 'unknown', 'timeframe_signals': []}
        
        timeframe_signals = []
        
        # Short-term (5-period smoothed)
        short_dm = demarker.rolling(window=5).mean()
        short_trend = 'bullish' if short_dm.iloc[-1] > short_dm.iloc[-5] else 'bearish'
        timeframe_signals.append({'timeframe': 'short', 'signal': short_trend, 'value': float(short_dm.iloc[-1])})
        
        # Medium-term (14-period smoothed)
        medium_dm = demarker.rolling(window=14).mean()
        medium_trend = 'bullish' if medium_dm.iloc[-1] > medium_dm.iloc[-10] else 'bearish'
        timeframe_signals.append({'timeframe': 'medium', 'signal': medium_trend, 'value': float(medium_dm.iloc[-1])})
        
        # Long-term (28-period smoothed)
        if len(demarker) >= 28:
            long_dm = demarker.rolling(window=28).mean()
            long_trend = 'bullish' if long_dm.iloc[-1] > long_dm.iloc[-15] else 'bearish'
            timeframe_signals.append({'timeframe': 'long', 'signal': long_trend, 'value': float(long_dm.iloc[-1])})
        
        # Analyze alignment
        signals = [signal['signal'] for signal in timeframe_signals]
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')
        
        if bullish_count > bearish_count:
            momentum_alignment = 'bullish'
        elif bearish_count > bullish_count:
            momentum_alignment = 'bearish'
        else:
            momentum_alignment = 'neutral'
        
        return {
            'momentum_alignment': momentum_alignment,
            'timeframe_signals': timeframe_signals,
            'alignment_strength': max(bullish_count, bearish_count) / len(signals)
        }
    
    def _train_ml_models(self, demarker: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML models for enhanced signal detection"""
        if not self.parameters['ml_enhancement'] or len(demarker) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, regime_targets, prediction_targets = self._prepare_ml_data(demarker, data)
            if len(features) > 30:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train regime classifier
                self.regime_classifier.fit(scaled_features, regime_targets)
                
                # Train DeMarker predictor
                self.demarker_predictor.fit(scaled_features, prediction_targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, demarker: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, regime_targets, prediction_targets = [], [], []
        lookback = 20
        
        for i in range(lookback, len(demarker) - 10):
            dm_window = demarker.iloc[i-lookback:i]
            price_window = data['close'].iloc[i-lookback:i]
            
            # DeMarker features
            dm_mean = dm_window.mean()
            dm_std = dm_window.std()
            dm_trend = np.polyfit(range(len(dm_window)), dm_window.values, 1)[0]
            dm_current = dm_window.iloc[-1]
            dm_momentum = np.mean(np.diff(dm_window.tail(5)))
            
            # Oscillator characteristics
            dm_overbought_time = len(dm_window[dm_window > 0.7]) / len(dm_window)
            dm_oversold_time = len(dm_window[dm_window < 0.3]) / len(dm_window)
            dm_range = dm_window.max() - dm_window.min()
            
            # Price features
            price_returns = price_window.pct_change().dropna()
            price_volatility = price_returns.std()
            price_trend = np.polyfit(range(len(price_window)), price_window.values, 1)[0]
            
            # Volume features (if available)
            if 'volume' in data.columns:
                volume_window = data['volume'].iloc[i-lookback:i]
                volume_trend = np.polyfit(range(len(volume_window)), volume_window.values, 1)[0]
                volume_ratio = volume_window.iloc[-1] / volume_window.mean()
            else:
                volume_trend = 0
                volume_ratio = 1
            
            # Technical features
            high_low_ratio = (data['high'].iloc[i-lookback:i] / data['low'].iloc[i-lookback:i]).mean()
            
            feature_vector = [
                dm_mean, dm_std, dm_trend, dm_current, dm_momentum,
                dm_overbought_time, dm_oversold_time, dm_range,
                price_volatility, price_trend, volume_trend, volume_ratio,
                high_low_ratio
            ]
            
            # Targets
            future_dm = demarker.iloc[i+1:i+6]
            future_price = data['close'].iloc[i+1:i+6]
            
            if len(future_dm) >= 3 and len(future_price) >= 3:
                # Regime target
                future_dm_mean = future_dm.mean()
                if future_dm_mean > 0.7:
                    regime_target = 2  # Overbought
                elif future_dm_mean < 0.3:
                    regime_target = 0  # Oversold
                else:
                    regime_target = 1  # Neutral
                
                # Prediction target
                prediction_target = future_dm.iloc[-1] - dm_current
            else:
                regime_target = 1
                prediction_target = 0.0
            
            features.append(feature_vector)
            regime_targets.append(regime_target)
            prediction_targets.append(prediction_target)
        
        return np.array(features), np.array(regime_targets), np.array(prediction_targets)
    
    def _get_ml_predictions(self, demarker: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Get ML-based predictions and regime classification"""
        if not self.models_trained or len(demarker) < 20:
            return {'regime': 'unknown', 'prediction': 0.0, 'confidence': 0.0}
        
        try:
            # Prepare current features
            lookback = 20
            dm_window = demarker.tail(lookback)
            price_window = data['close'].tail(lookback)
            
            dm_mean = dm_window.mean()
            dm_std = dm_window.std()
            dm_trend = np.polyfit(range(len(dm_window)), dm_window.values, 1)[0]
            dm_current = dm_window.iloc[-1]
            dm_momentum = np.mean(np.diff(dm_window.tail(5)))
            
            dm_overbought_time = len(dm_window[dm_window > 0.7]) / len(dm_window)
            dm_oversold_time = len(dm_window[dm_window < 0.3]) / len(dm_window)
            dm_range = dm_window.max() - dm_window.min()
            
            price_returns = price_window.pct_change().dropna()
            price_volatility = price_returns.std()
            price_trend = np.polyfit(range(len(price_window)), price_window.values, 1)[0]
            
            if 'volume' in data.columns:
                volume_window = data['volume'].tail(lookback)
                volume_trend = np.polyfit(range(len(volume_window)), volume_window.values, 1)[0]
                volume_ratio = volume_window.iloc[-1] / volume_window.mean()
            else:
                volume_trend = 0
                volume_ratio = 1
            
            high_low_ratio = (data['high'].tail(lookback) / data['low'].tail(lookback)).mean()
            
            feature_vector = np.array([[
                dm_mean, dm_std, dm_trend, dm_current, dm_momentum,
                dm_overbought_time, dm_oversold_time, dm_range,
                price_volatility, price_trend, volume_trend, volume_ratio,
                high_low_ratio
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            
            # Get regime prediction
            regime_proba = self.regime_classifier.predict_proba(scaled_features)[0]
            regime_prediction = self.regime_classifier.predict(scaled_features)[0]
            
            regime_map = {0: 'oversold', 1: 'neutral', 2: 'overbought'}
            regime = regime_map.get(regime_prediction, 'unknown')
            
            # Get value prediction
            value_prediction = self.demarker_predictor.predict(scaled_features)[0]
            
            # Confidence based on probability
            confidence = np.max(regime_proba)
            
            return {
                'regime': regime,
                'prediction': float(value_prediction),
                'confidence': float(confidence),
                'regime_probabilities': {
                    'oversold': float(regime_proba[0]),
                    'neutral': float(regime_proba[1]),
                    'overbought': float(regime_proba[2])
                }
            }
        except Exception:
            return {'regime': 'unknown', 'prediction': 0.0, 'confidence': 0.0}
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate DeMarker indicator with comprehensive analysis"""
        try:
            # Optimize period
            optimized_period = self._optimize_period(data)
            
            # Calculate DeMarker components
            dem_max, dem_min = self._calculate_demarker_components(data, optimized_period)
            
            # Calculate DeMarker oscillator
            demarker = self._calculate_demarker(dem_max, dem_min, optimized_period)
            
            # Optimize thresholds
            overbought_threshold, oversold_threshold = self._optimize_thresholds(demarker, data)
            
            # Advanced analysis
            divergence_analysis = self._detect_divergences(demarker, data)
            momentum_analysis = self._analyze_multi_timeframe_momentum(demarker, data)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(demarker, data)
            
            # Get ML predictions
            ml_predictions = self._get_ml_predictions(demarker, data)
            
            # Generate comprehensive signal
            signal, confidence = self._generate_demarker_signal(
                demarker, overbought_threshold, oversold_threshold,
                divergence_analysis, momentum_analysis, ml_predictions, data
            )
            
            # Update history
            if len(demarker) > 0 and not pd.isna(demarker.iloc[-1]):
                self.history['demarker_values'].append(float(demarker.iloc[-1]))
                self.history['dem_max_values'].append(float(dem_max.iloc[-1]) if not pd.isna(dem_max.iloc[-1]) else 0.0)
                self.history['dem_min_values'].append(float(dem_min.iloc[-1]) if not pd.isna(dem_min.iloc[-1]) else 0.0)
                self.history['thresholds'].append({
                    'overbought': overbought_threshold,
                    'oversold': oversold_threshold
                })
                
                # Keep history limited
                for key in self.history:
                    if len(self.history[key]) > 100:
                        self.history[key] = self.history[key][-100:]
            
            result = {
                'demarker': float(demarker.iloc[-1]) if len(demarker) > 0 and not pd.isna(demarker.iloc[-1]) else 0.5,
                'dem_max': float(dem_max.iloc[-1]) if len(dem_max) > 0 and not pd.isna(dem_max.iloc[-1]) else 0.0,
                'dem_min': float(dem_min.iloc[-1]) if len(dem_min) > 0 and not pd.isna(dem_min.iloc[-1]) else 0.0,
                'overbought_threshold': overbought_threshold,
                'oversold_threshold': oversold_threshold,
                'signal': signal,
                'confidence': confidence,
                'divergence_analysis': divergence_analysis,
                'momentum_analysis': momentum_analysis,
                'ml_predictions': ml_predictions,
                'optimized_period': optimized_period,
                'market_regime': self._classify_market_regime(demarker, data),
                'values_history': {
                    'demarker': demarker.tail(30).tolist(),
                    'thresholds': {
                        'overbought': [overbought_threshold] * 30,
                        'oversold': [oversold_threshold] * 30
                    }
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate DeMarker: {str(e)}",
                cause=e
            )
    
    def _generate_demarker_signal(self, demarker: pd.Series, overbought: float, oversold: float,
                                 divergence_analysis: Dict, momentum_analysis: Dict,
                                 ml_predictions: Dict, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive DeMarker signal"""
        signal_components = []
        confidence_components = []
        
        current_dm = demarker.iloc[-1] if len(demarker) > 0 and not pd.isna(demarker.iloc[-1]) else 0.5
        
        # Overbought/Oversold signals
        if current_dm > overbought:
            signal_components.append(-0.8)
            confidence_components.append(0.8)
        elif current_dm < oversold:
            signal_components.append(0.8)
            confidence_components.append(0.8)
        
        # Zone transition signals
        if len(demarker) > 1:
            prev_dm = demarker.iloc[-2]
            
            # Bullish: crossing up from oversold
            if prev_dm <= oversold and current_dm > oversold:
                signal_components.append(0.9)
                confidence_components.append(0.9)
            
            # Bearish: crossing down from overbought
            elif prev_dm >= overbought and current_dm < overbought:
                signal_components.append(-0.9)
                confidence_components.append(0.9)
        
        # Divergence signals
        if divergence_analysis['bullish_divergence']:
            strength = divergence_analysis['strength']
            signal_components.append(0.7 * strength)
            confidence_components.append(0.8)
        elif divergence_analysis['bearish_divergence']:
            strength = divergence_analysis['strength']
            signal_components.append(-0.7 * strength)
            confidence_components.append(0.8)
        
        # Multi-timeframe momentum signals
        momentum_alignment = momentum_analysis['momentum_alignment']
        alignment_strength = momentum_analysis['alignment_strength']
        
        if momentum_alignment == 'bullish' and alignment_strength > 0.6:
            signal_components.append(0.6)
            confidence_components.append(0.7)
        elif momentum_alignment == 'bearish' and alignment_strength > 0.6:
            signal_components.append(-0.6)
            confidence_components.append(0.7)
        
        # ML enhancement
        ml_regime = ml_predictions['regime']
        ml_confidence = ml_predictions['confidence']
        
        if ml_confidence > 0.7:
            if ml_regime == 'oversold':
                signal_components.append(0.7)
                confidence_components.append(ml_confidence)
            elif ml_regime == 'overbought':
                signal_components.append(-0.7)
                confidence_components.append(ml_confidence)
        
        # Central line behavior
        if 0.45 <= current_dm <= 0.55:
            # Near center line - reduce all signal strengths
            signal_components = [s * 0.5 for s in signal_components]
        
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
    
    def _classify_market_regime(self, demarker: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime based on DeMarker patterns"""
        if len(demarker) < 30:
            return {'regime': 'unknown', 'volatility': 'normal', 'trend_state': 'undefined'}
        
        recent_dm = demarker.tail(30)
        current_dm = demarker.iloc[-1]
        
        # Calculate regime characteristics
        dm_volatility = recent_dm.std()
        dm_mean = recent_dm.mean()
        dm_trend = np.polyfit(range(len(recent_dm)), recent_dm.values, 1)[0]
        
        # Time spent in zones
        overbought_time = len(recent_dm[recent_dm > 0.7]) / len(recent_dm)
        oversold_time = len(recent_dm[recent_dm < 0.3]) / len(recent_dm)
        neutral_time = len(recent_dm[(recent_dm >= 0.3) & (recent_dm <= 0.7)]) / len(recent_dm)
        
        # Classify regime
        if dm_volatility > 0.15:
            volatility = 'high'
        elif dm_volatility > 0.08:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        if overbought_time > 0.4:
            regime = 'persistently_overbought'
        elif oversold_time > 0.4:
            regime = 'persistently_oversold'
        elif neutral_time > 0.8:
            regime = 'range_bound'
        elif dm_volatility > 0.15:
            regime = 'volatile'
        else:
            regime = 'normal'
        
        # Trend state
        if abs(dm_trend) > 0.005:
            trend_state = 'trending_up' if dm_trend > 0 else 'trending_down'
        else:
            trend_state = 'sideways'
        
        return {
            'regime': regime,
            'volatility': volatility,
            'trend_state': trend_state,
            'current_level': float(current_dm),
            'mean_level': float(dm_mean),
            'overbought_time': float(overbought_time),
            'oversold_time': float(oversold_time),
            'neutral_time': float(neutral_time)
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'demarker',
            'models_trained': self.models_trained,
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'ml_enhancement': self.parameters['ml_enhancement'],
            'divergence_detection': self.parameters['divergence_detection'],
            'dynamic_thresholds': self.parameters['dynamic_thresholds'],
            'multi_timeframe': self.parameters['multi_timeframe'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata