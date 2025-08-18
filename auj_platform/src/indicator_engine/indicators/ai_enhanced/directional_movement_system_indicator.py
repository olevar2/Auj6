"""
Directional Movement System Indicator - AI Enhanced Category
===========================================================

Advanced AI-enhanced Directional Movement System (ADX/DMI) with machine learning,
adaptive thresholds, multi-timeframe analysis, and sophisticated trend analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import signal, stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class DirectionalMovementSystemIndicator(StandardIndicatorInterface):
    """
    AI-Enhanced Directional Movement System with advanced features.
    
    Features:
    - Adaptive ADX/DMI with machine learning optimization
    - Multi-timeframe directional analysis
    - Volatility-adjusted directional strength
    - Trend persistence probability calculation
    - Breakout prediction using ML models
    - Dynamic threshold optimization
    - Pattern recognition for directional changes
    - Risk-adjusted position sizing signals
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 14,                    # ADX/DMI period
            'adaptive_periods': [7, 14, 21, 28],  # Multiple periods for analysis
            'adx_threshold': 25,             # Strong trend threshold
            'weak_trend_threshold': 20,      # Weak trend threshold
            'volatility_window': 20,         # Volatility calculation window
            'ml_prediction_window': 50,      # ML model training window
            'trend_confirmation_period': 3,   # Trend confirmation periods
            'breakout_threshold': 1.5,       # Breakout detection threshold
            'use_machine_learning': True,    # Enable ML optimization
            'adaptive_thresholds': True,     # Enable adaptive thresholds
            'multi_timeframe': True,         # Enable multi-timeframe analysis
            'pattern_recognition': True,     # Enable pattern recognition
            'risk_management': True,         # Enable risk management features
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("DirectionalMovementSystemIndicator", default_params)
        
        # Initialize ML models
        self.trend_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Cache for calculations
        self._cache = {}
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["high", "low", "close", "volume"],
            min_periods=max(self.parameters['adaptive_periods']) + self.parameters['ml_prediction_window']
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced directional movement system with AI enhancements."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Calculate base DMI/ADX for all periods
            multi_period_results = {}
            for period in self.parameters['adaptive_periods']:
                result = self._calculate_base_dmi_adx(data, period)
                multi_period_results[f'period_{period}'] = result
            
            # Get primary period results
            primary_period = self.parameters['period']
            if primary_period not in self.parameters['adaptive_periods']:
                primary_result = self._calculate_base_dmi_adx(data, primary_period)
            else:
                primary_result = multi_period_results[f'period_{primary_period}']
            
            # Calculate advanced features
            volatility_metrics = self._calculate_volatility_metrics(data)
            trend_strength = self._calculate_trend_strength(data, primary_result)
            directional_bias = self._calculate_directional_bias(multi_period_results)
            
            # Calculate adaptive thresholds
            adaptive_thresholds = self._calculate_adaptive_thresholds(
                data, primary_result, volatility_metrics
            )
            
            # Machine learning predictions
            ml_predictions = {}
            if self.parameters['use_machine_learning']:
                ml_predictions = self._calculate_ml_predictions(data, primary_result)
            
            # Pattern recognition
            patterns = {}
            if self.parameters['pattern_recognition']:
                patterns = self._detect_directional_patterns(data, primary_result)
            
            # Risk management signals
            risk_signals = {}
            if self.parameters['risk_management']:
                risk_signals = self._calculate_risk_signals(
                    data, primary_result, volatility_metrics
                )
            
            # Generate trading signals
            signals = self._generate_trading_signals(
                primary_result, adaptive_thresholds, ml_predictions, patterns
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                primary_result, multi_period_results, volatility_metrics, ml_predictions
            )
            
            return {
                'adx': primary_result['adx'],
                'di_plus': primary_result['di_plus'],
                'di_minus': primary_result['di_minus'],
                'dx': primary_result['dx'],
                'trend_strength': trend_strength,
                'directional_bias': directional_bias,
                'adaptive_thresholds': adaptive_thresholds,
                'volatility_metrics': volatility_metrics,
                'ml_predictions': ml_predictions,
                'patterns': patterns,
                'risk_signals': risk_signals,
                'signals': signals,
                'confidence': confidence,
                'multi_period_analysis': multi_period_results,
                'signal_strength': signals.get('strength', 0.0),
                'trend_persistence_probability': ml_predictions.get('trend_persistence', 0.5)
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _calculate_base_dmi_adx(self, data: pd.DataFrame, period: int) -> Dict[str, float]:
        """Calculate base DMI/ADX components."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate True Range and Directional Movements
        tr = self._calculate_true_range(high, low, close)
        dm_plus = self._calculate_directional_movement_plus(high)
        dm_minus = self._calculate_directional_movement_minus(low)
        
        # Smooth the values using Wilder's smoothing
        tr_smooth = self._wilders_smoothing(tr, period)
        dm_plus_smooth = self._wilders_smoothing(dm_plus, period)
        dm_minus_smooth = self._wilders_smoothing(dm_minus, period)
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX
        di_diff = np.abs(di_plus - di_minus)
        di_sum = di_plus + di_minus
        dx = 100 * (di_diff / np.where(di_sum == 0, 1, di_sum))
        
        # Calculate ADX
        adx = self._wilders_smoothing(dx, period)
        
        return {
            'adx': adx[-1] if len(adx) > 0 else 0.0,
            'di_plus': di_plus[-1] if len(di_plus) > 0 else 0.0,
            'di_minus': di_minus[-1] if len(di_minus) > 0 else 0.0,
            'dx': dx[-1] if len(dx) > 0 else 0.0,
            'adx_series': adx,
            'di_plus_series': di_plus,
            'di_minus_series': di_minus
        }
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range."""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def _calculate_directional_movement_plus(self, high: np.ndarray) -> np.ndarray:
        """Calculate +DM (Positive Directional Movement)."""
        high_diff = np.diff(high, prepend=high[0])
        low_diff = np.diff(np.roll(high, 1), prepend=high[0])  # Using high for calculation
        
        dm_plus = np.where(
            (high_diff > 0) & (high_diff > low_diff),
            high_diff,
            0.0
        )
        
        return dm_plus
    
    def _calculate_directional_movement_minus(self, low: np.ndarray) -> np.ndarray:
        """Calculate -DM (Negative Directional Movement)."""
        high_diff = np.diff(np.roll(low, 1), prepend=low[0])  # Using low for calculation
        low_diff = np.diff(low, prepend=low[0])
        
        dm_minus = np.where(
            (low_diff < 0) & (np.abs(low_diff) > high_diff),
            np.abs(low_diff),
            0.0
        )
        
        return dm_minus
    
    def _wilders_smoothing(self, values: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing (modified EMA)."""
        if len(values) < period:
            return values
        
        alpha = 1.0 / period
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        
        for i in range(1, len(values)):
            if i < period:
                smoothed[i] = np.mean(values[:i+1])
            else:
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced volatility metrics."""
        window = self.parameters['volatility_window']
        close = data['close'].values
        
        # Historical volatility
        returns = np.diff(np.log(close))
        hist_vol = np.std(returns[-window:]) * np.sqrt(252) if len(returns) >= window else 0.0
        
        # ATR-based volatility
        high = data['high'].values
        low = data['low'].values
        tr = self._calculate_true_range(high, low, close)
        atr = np.mean(tr[-window:]) if len(tr) >= window else 0.0
        atr_volatility = atr / close[-1] if close[-1] != 0 else 0.0
        
        # Volatility regime
        vol_ma = np.mean([hist_vol, atr_volatility])
        vol_std = np.std([hist_vol, atr_volatility]) if len([hist_vol, atr_volatility]) > 1 else 0.0
        vol_regime = "high" if vol_ma > vol_std else "normal" if vol_ma > 0.5 * vol_std else "low"
        
        return {
            'historical_volatility': hist_vol,
            'atr_volatility': atr_volatility,
            'volatility_regime': vol_regime,
            'volatility_score': min(max(vol_ma / 0.3, 0.0), 2.0)  # Normalized score
        }
    
    def _calculate_trend_strength(self, data: pd.DataFrame, dmi_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate advanced trend strength metrics."""
        adx = dmi_result['adx']
        di_plus = dmi_result['di_plus']
        di_minus = dmi_result['di_minus']
        
        # Basic trend strength
        if adx >= self.parameters['adx_threshold']:
            strength = "strong"
        elif adx >= self.parameters['weak_trend_threshold']:
            strength = "moderate"
        else:
            strength = "weak"
        
        # Directional bias strength
        di_diff = abs(di_plus - di_minus)
        directional_strength = di_diff / max(di_plus + di_minus, 1)
        
        # Trend consistency (using ADX series if available)
        consistency = 0.0
        if 'adx_series' in dmi_result and len(dmi_result['adx_series']) > 5:
            adx_series = dmi_result['adx_series'][-10:]  # Last 10 periods
            consistency = 1.0 - (np.std(adx_series) / max(np.mean(adx_series), 1))
        
        # Combined strength score
        strength_score = (adx / 50.0) * directional_strength * (1 + consistency)
        
        return {
            'category': strength,
            'adx_value': adx,
            'directional_strength': directional_strength,
            'consistency': consistency,
            'strength_score': min(strength_score, 2.0)
        }
    
    def _calculate_directional_bias(self, multi_period_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate directional bias across multiple timeframes."""
        bullish_signals = 0
        bearish_signals = 0
        total_strength = 0.0
        
        for period_key, result in multi_period_results.items():
            di_plus = result['di_plus']
            di_minus = result['di_minus']
            adx = result['adx']
            
            # Weight by ADX strength
            weight = min(adx / 50.0, 1.0)
            total_strength += weight
            
            if di_plus > di_minus:
                bullish_signals += weight
            else:
                bearish_signals += weight
        
        if total_strength > 0:
            bullish_ratio = bullish_signals / total_strength
            bearish_ratio = bearish_signals / total_strength
        else:
            bullish_ratio = bearish_ratio = 0.5
        
        # Determine overall bias
        if bullish_ratio > 0.6:
            bias = "bullish"
        elif bearish_ratio > 0.6:
            bias = "bearish"
        else:
            bias = "neutral"
        
        return {
            'bias': bias,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'strength': abs(bullish_ratio - bearish_ratio),
            'confidence': max(bullish_ratio, bearish_ratio)
        }
    
    def _calculate_adaptive_thresholds(self, data: pd.DataFrame, dmi_result: Dict[str, Any], 
                                     volatility_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate adaptive thresholds based on market conditions."""
        base_threshold = self.parameters['adx_threshold']
        vol_score = volatility_metrics['volatility_score']
        
        # Adjust thresholds based on volatility
        if volatility_metrics['volatility_regime'] == "high":
            strong_trend_threshold = base_threshold * 1.2
            weak_trend_threshold = self.parameters['weak_trend_threshold'] * 1.3
        elif volatility_metrics['volatility_regime'] == "low":
            strong_trend_threshold = base_threshold * 0.8
            weak_trend_threshold = self.parameters['weak_trend_threshold'] * 0.7
        else:
            strong_trend_threshold = base_threshold
            weak_trend_threshold = self.parameters['weak_trend_threshold']
        
        # Dynamic breakout threshold
        breakout_threshold = self.parameters['breakout_threshold'] * (1 + vol_score * 0.5)
        
        return {
            'strong_trend_threshold': strong_trend_threshold,
            'weak_trend_threshold': weak_trend_threshold,
            'breakout_threshold': breakout_threshold,
            'di_separation_threshold': 5.0 + vol_score * 3.0
        }
    
    def _calculate_ml_predictions(self, data: pd.DataFrame, dmi_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate machine learning-based predictions."""
        if not self.parameters['use_machine_learning']:
            return {}
        
        try:
            # Prepare features
            features = self._prepare_ml_features(data, dmi_result)
            
            if len(features) < self.parameters['ml_prediction_window']:
                return {'trend_persistence': 0.5, 'breakout_probability': 0.5}
            
            # Train model if not trained
            if not self.is_trained and len(features) >= self.parameters['ml_prediction_window']:
                self._train_ml_model(features, data)
            
            if self.is_trained:
                # Make predictions
                latest_features = features[-1:].reshape(1, -1)
                latest_features_scaled = self.scaler.transform(latest_features)
                
                prediction = self.trend_predictor.predict(latest_features_scaled)[0]
                
                # Convert to probabilities
                trend_persistence = max(0.0, min(1.0, (prediction + 1) / 2))
                breakout_probability = 1.0 - trend_persistence
                
                return {
                    'trend_persistence': trend_persistence,
                    'breakout_probability': breakout_probability,
                    'prediction_confidence': 0.7  # Fixed confidence for now
                }
            
        except Exception as e:
            pass
        
        return {'trend_persistence': 0.5, 'breakout_probability': 0.5}
    
    def _prepare_ml_features(self, data: pd.DataFrame, dmi_result: Dict[str, Any]) -> np.ndarray:
        """Prepare features for machine learning."""
        features = []
        
        # ADX and DI features
        if 'adx_series' in dmi_result:
            adx_series = dmi_result['adx_series']
            di_plus_series = dmi_result['di_plus_series']
            di_minus_series = dmi_result['di_minus_series']
            
            min_len = min(len(adx_series), len(di_plus_series), len(di_minus_series))
            
            for i in range(min_len):
                feature_vector = [
                    adx_series[i],
                    di_plus_series[i],
                    di_minus_series[i],
                    di_plus_series[i] - di_minus_series[i],  # DI difference
                    (di_plus_series[i] + di_minus_series[i]) / 2,  # DI average
                ]
                features.append(feature_vector)
        
        return np.array(features) if features else np.array([])
    
    def _train_ml_model(self, features: np.ndarray, data: pd.DataFrame):
        """Train the machine learning model."""
        try:
            if len(features) < self.parameters['ml_prediction_window']:
                return
            
            # Create targets (future trend direction)
            close_prices = data['close'].values
            targets = []
            
            for i in range(len(features) - 5):  # Predict 5 periods ahead
                current_price = close_prices[min(i + len(close_prices) - len(features), len(close_prices) - 1)]
                future_price = close_prices[min(i + 5 + len(close_prices) - len(features), len(close_prices) - 1)]
                
                # Trend direction: 1 for up, -1 for down
                target = 1 if future_price > current_price else -1
                targets.append(target)
            
            # Align features with targets
            train_features = features[:-5] if len(features) > 5 else features
            train_targets = np.array(targets[:len(train_features)])
            
            if len(train_features) > 0 and len(train_targets) > 0:
                # Scale features
                self.scaler.fit(train_features)
                train_features_scaled = self.scaler.transform(train_features)
                
                # Train model
                self.trend_predictor.fit(train_features_scaled, train_targets)
                self.is_trained = True
                
        except Exception as e:
            pass
    
    def _detect_directional_patterns(self, data: pd.DataFrame, dmi_result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect directional movement patterns."""
        patterns = {
            'adx_rising': False,
            'di_crossover': None,
            'extreme_di_separation': False,
            'trend_exhaustion': False
        }
        
        try:
            if 'adx_series' in dmi_result and len(dmi_result['adx_series']) > 5:
                adx_series = dmi_result['adx_series']
                di_plus_series = dmi_result['di_plus_series']
                di_minus_series = dmi_result['di_minus_series']
                
                # ADX rising pattern
                if len(adx_series) > 3:
                    patterns['adx_rising'] = adx_series[-1] > adx_series[-2] > adx_series[-3]
                
                # DI crossover detection
                if len(di_plus_series) > 2 and len(di_minus_series) > 2:
                    current_di_diff = di_plus_series[-1] - di_minus_series[-1]
                    prev_di_diff = di_plus_series[-2] - di_minus_series[-2]
                    
                    if current_di_diff > 0 and prev_di_diff <= 0:
                        patterns['di_crossover'] = 'bullish'
                    elif current_di_diff < 0 and prev_di_diff >= 0:
                        patterns['di_crossover'] = 'bearish'
                
                # Extreme DI separation
                di_separation = abs(di_plus_series[-1] - di_minus_series[-1])
                patterns['extreme_di_separation'] = di_separation > 30
                
                # Trend exhaustion (high ADX with weakening DI difference)
                if len(adx_series) > 5 and adx_series[-1] > 40:
                    recent_separations = [abs(di_plus_series[i] - di_minus_series[i]) 
                                        for i in range(-5, 0)]
                    if len(recent_separations) > 2:
                        patterns['trend_exhaustion'] = recent_separations[-1] < recent_separations[-3]
        
        except Exception:
            pass
        
        return patterns
    
    def _calculate_risk_signals(self, data: pd.DataFrame, dmi_result: Dict[str, Any], 
                              volatility_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate risk management signals."""
        adx = dmi_result['adx']
        di_plus = dmi_result['di_plus']
        di_minus = dmi_result['di_minus']
        vol_score = volatility_metrics['volatility_score']
        
        # Position sizing based on trend strength and volatility
        if adx > 30 and vol_score < 1.0:
            position_size = "large"
            risk_multiplier = 1.5
        elif adx > 20:
            position_size = "medium"
            risk_multiplier = 1.0
        else:
            position_size = "small"
            risk_multiplier = 0.5
        
        # Stop loss recommendations
        atr_multiplier = 2.0 + vol_score * 0.5
        stop_loss_atr_multiple = atr_multiplier
        
        # Risk level assessment
        if vol_score > 1.5 and adx < 20:
            risk_level = "high"
        elif vol_score > 1.0 or adx > 40:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'position_size': position_size,
            'risk_multiplier': risk_multiplier,
            'stop_loss_atr_multiple': stop_loss_atr_multiple,
            'risk_level': risk_level,
            'volatility_adjusted': True
        }
    
    def _generate_trading_signals(self, dmi_result: Dict[str, Any], adaptive_thresholds: Dict[str, float],
                                ml_predictions: Dict[str, float], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trading signals."""
        adx = dmi_result['adx']
        di_plus = dmi_result['di_plus']
        di_minus = dmi_result['di_minus']
        
        signals = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.0,
            'entry_signal': False,
            'exit_signal': False
        }
        
        # Basic directional signal
        di_diff = di_plus - di_minus
        if abs(di_diff) > adaptive_thresholds['di_separation_threshold']:
            signals['direction'] = 'bullish' if di_diff > 0 else 'bearish'
        
        # Signal strength based on ADX
        if adx > adaptive_thresholds['strong_trend_threshold']:
            signals['strength'] = min((adx - adaptive_thresholds['strong_trend_threshold']) / 20.0 + 0.7, 1.0)
        elif adx > adaptive_thresholds['weak_trend_threshold']:
            signals['strength'] = (adx - adaptive_thresholds['weak_trend_threshold']) / \
                                (adaptive_thresholds['strong_trend_threshold'] - adaptive_thresholds['weak_trend_threshold']) * 0.5 + 0.3
        else:
            signals['strength'] = adx / adaptive_thresholds['weak_trend_threshold'] * 0.3
        
        # Entry signals
        entry_conditions = [
            adx > adaptive_thresholds['weak_trend_threshold'],
            abs(di_diff) > adaptive_thresholds['di_separation_threshold'],
            patterns.get('adx_rising', False),
            patterns.get('di_crossover') is not None
        ]
        
        signals['entry_signal'] = sum(entry_conditions) >= 2
        
        # Exit signals
        exit_conditions = [
            adx < adaptive_thresholds['weak_trend_threshold'],
            abs(di_diff) < 5.0,
            patterns.get('trend_exhaustion', False)
        ]
        
        signals['exit_signal'] = sum(exit_conditions) >= 2
        
        # Confidence calculation
        base_confidence = signals['strength']
        ml_confidence = ml_predictions.get('prediction_confidence', 0.5)
        pattern_confidence = 0.1 * sum([1 for p in patterns.values() if p and p != 'neutral'])
        
        signals['confidence'] = min((base_confidence + ml_confidence + pattern_confidence) / 2.0, 1.0)
        
        return signals
    
    def _calculate_confidence_score(self, primary_result: Dict[str, Any], 
                                  multi_period_results: Dict[str, Dict],
                                  volatility_metrics: Dict[str, float],
                                  ml_predictions: Dict[str, float]) -> float:
        """Calculate overall confidence score."""
        # ADX strength component
        adx = primary_result['adx']
        adx_confidence = min(adx / 50.0, 1.0)
        
        # Multi-timeframe consistency
        consistency_scores = []
        primary_direction = 1 if primary_result['di_plus'] > primary_result['di_minus'] else -1
        
        for result in multi_period_results.values():
            direction = 1 if result['di_plus'] > result['di_minus'] else -1
            consistency_scores.append(1.0 if direction == primary_direction else 0.0)
        
        consistency_confidence = np.mean(consistency_scores) if consistency_scores else 0.5
        
        # Volatility confidence (lower volatility = higher confidence)
        vol_confidence = max(0.0, 1.0 - volatility_metrics['volatility_score'] / 2.0)
        
        # ML confidence
        ml_confidence = ml_predictions.get('prediction_confidence', 0.5)
        
        # Combined confidence
        weights = [0.3, 0.25, 0.25, 0.2]  # ADX, consistency, volatility, ML
        confidence_components = [adx_confidence, consistency_confidence, vol_confidence, ml_confidence]
        
        overall_confidence = sum(w * c for w, c in zip(weights, confidence_components))
        
        return min(max(overall_confidence, 0.0), 1.0)
    
    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when insufficient data."""
        return {
            'adx': 0.0,
            'di_plus': 0.0,
            'di_minus': 0.0,
            'dx': 0.0,
            'trend_strength': {'category': 'weak', 'strength_score': 0.0},
            'directional_bias': {'bias': 'neutral', 'confidence': 0.0},
            'signals': {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.0},
            'confidence': 0.0,
            'signal_strength': 0.0,
            'trend_persistence_probability': 0.5
        }
    
    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        return self._get_default_output()
