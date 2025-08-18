"""
Acceleration Deceleration Indicator (AC) - Advanced Implementation
================================================================

Bill Williams' Acceleration/Deceleration oscillator with ML enhancement
and sophisticated momentum analysis for detecting acceleration and deceleration
in price movements.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType,
    IndicatorResult
)
from ....core.exceptions import IndicatorCalculationException


class AccelerationDecelerationIndicator(StandardIndicatorInterface):
    """
    Advanced Acceleration/Deceleration Indicator Implementation
    
    Features:
    - Bill Williams' AC oscillator with adaptive parameters
    - Machine learning enhancement for anomaly detection
    - Multi-timeframe momentum analysis
    - Advanced signal generation with confidence scoring
    - Sophisticated trend acceleration detection
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'ao_fast_period': 5,           # Fast MA period for AO
            'ao_slow_period': 34,          # Slow MA period for AO
            'ac_period': 5,                # AC smoothing period
            'signal_threshold': 0.0,       # Signal threshold
            'ml_lookback': 50,             # ML analysis lookback
            'anomaly_sensitivity': 0.1,    # Anomaly detection sensitivity
            'momentum_threshold': 0.001,   # Momentum change threshold
            'trend_confirmation': 3        # Bars for trend confirmation
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="AccelerationDecelerationIndicator", parameters=default_params)
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=self.parameters['anomaly_sensitivity'],
            random_state=42
        )
        self.ml_trained = False
    
    def get_data_requirements(self) -> DataRequirement:
        """Define data requirements for AC calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close'],
            min_periods=max(self.parameters['ao_slow_period'], self.parameters['ml_lookback']) + 10,
            lookback_periods=200
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        required_params = ['ao_fast_period', 'ao_slow_period', 'ac_period']
        
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")
            
            if not isinstance(self.parameters[param], (int, float)) or self.parameters[param] <= 0:
                raise ValueError(f"Parameter {param} must be a positive number")
        
        if self.parameters['ao_fast_period'] >= self.parameters['ao_slow_period']:
            raise ValueError("Fast period must be less than slow period")
        
        return True
    
    def _calculate_awesome_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Awesome Oscillator (AO) with enhanced precision
        AO = SMA(HL2, fast_period) - SMA(HL2, slow_period)
        """
        hl2 = (data['high'] + data['low']) / 2
        
        # Use exponentially weighted moving averages for better responsiveness
        fast_ma = hl2.ewm(span=self.parameters['ao_fast_period']).mean()
        slow_ma = hl2.ewm(span=self.parameters['ao_slow_period']).mean()
        
        ao = fast_ma - slow_ma
        return ao
    
    def _calculate_ac_core(self, ao: pd.Series) -> pd.Series:
        """
        Calculate Acceleration/Deceleration core values
        AC = AO - SMA(AO, ac_period)
        """
        ao_sma = ao.rolling(window=self.parameters['ac_period']).mean()
        ac = ao - ao_sma
        return ac
    
    def _detect_momentum_anomalies(self, ac_values: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Use ML to detect momentum anomalies and enhance signal quality
        """
        if len(ac_values) < self.parameters['ml_lookback']:
            return pd.Series(0, index=ac_values.index)
        
        try:
            # Prepare features for ML analysis
            features = self._prepare_ml_features(ac_values, data)
            
            if not self.ml_trained and len(features) >= self.parameters['ml_lookback']:
                # Train anomaly detector
                scaled_features = self.scaler.fit_transform(features)
                self.anomaly_detector.fit(scaled_features)
                self.ml_trained = True
            
            if self.ml_trained:
                # Detect anomalies
                scaled_features = self.scaler.transform(features)
                anomaly_scores = self.anomaly_detector.decision_function(scaled_features)
                return pd.Series(anomaly_scores, index=ac_values.index[-len(anomaly_scores):])
            
        except Exception as e:
            # Fallback to standard calculation if ML fails
            pass
        
        return pd.Series(0, index=ac_values.index)
    
    def _prepare_ml_features(self, ac_values: pd.Series, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for machine learning analysis
        """
        features = []
        
        # AC-based features
        ac_array = ac_values.values
        for i in range(self.parameters['ml_lookback'], len(ac_array)):
            window = ac_array[i-self.parameters['ml_lookback']:i]
            
            feature_vector = [
                np.mean(window),                    # Average AC
                np.std(window),                     # AC volatility
                np.max(window) - np.min(window),    # AC range
                window[-1] - window[0],             # AC change
                np.sum(window > 0) / len(window),   # Positive ratio
                self._calculate_momentum_persistence(window),  # Momentum persistence
                self._calculate_acceleration_rate(window),     # Acceleration rate
            ]
            
            # Price-based features
            price_window = data['close'].iloc[i-self.parameters['ml_lookback']:i].values
            feature_vector.extend([
                np.corrcoef(window, price_window)[0, 1] if len(set(window)) > 1 else 0,  # AC-Price correlation
                np.std(price_window) / np.mean(price_window) if np.mean(price_window) != 0 else 0,  # Price volatility ratio
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _calculate_momentum_persistence(self, values: np.ndarray) -> float:
        """Calculate momentum persistence score"""
        if len(values) < 2:
            return 0.0
        
        direction_changes = 0
        for i in range(1, len(values)):
            if (values[i] > values[i-1]) != (values[i-1] > values[i-2] if i > 1 else True):
                direction_changes += 1
        
        return 1.0 - (direction_changes / (len(values) - 1))
    
    def _calculate_acceleration_rate(self, values: np.ndarray) -> float:
        """Calculate acceleration rate (second derivative approximation)"""
        if len(values) < 3:
            return 0.0
        
        # Calculate discrete second derivative
        second_derivatives = []
        for i in range(2, len(values)):
            second_deriv = values[i] - 2*values[i-1] + values[i-2]
            second_derivatives.append(abs(second_deriv))
        
        return np.mean(second_derivatives)
    
    def _generate_advanced_signals(self, ac: pd.Series, ao: pd.Series, anomaly_scores: pd.Series) -> Tuple[SignalType, float]:
        """
        Generate sophisticated trading signals with confidence scoring
        """
        if len(ac) < self.parameters['trend_confirmation']:
            return SignalType.NEUTRAL, 0.0
        
        current_ac = ac.iloc[-1]
        recent_ac = ac.iloc[-self.parameters['trend_confirmation']:]
        current_ao = ao.iloc[-1]
        recent_ao = ao.iloc[-self.parameters['trend_confirmation']:]
        
        # Initialize signal components
        signal_strength = 0.0
        confidence_factors = []
        
        # 1. Basic AC signal
        if current_ac > self.parameters['signal_threshold']:
            signal_strength += 1.0
        elif current_ac < -self.parameters['signal_threshold']:
            signal_strength -= 1.0
        
        # 2. AC trend analysis
        ac_trend = 1 if recent_ac.is_monotonic_increasing else (-1 if recent_ac.is_monotonic_decreasing else 0)
        signal_strength += ac_trend * 0.5
        confidence_factors.append(abs(ac_trend))
        
        # 3. AO confirmation
        ao_direction = 1 if current_ao > 0 else -1
        if (signal_strength > 0 and ao_direction > 0) or (signal_strength < 0 and ao_direction < 0):
            signal_strength += ao_direction * 0.3
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        # 4. Momentum persistence
        momentum_persistence = self._calculate_momentum_persistence(recent_ac.values)
        signal_strength += (momentum_persistence - 0.5) * 0.4
        confidence_factors.append(momentum_persistence)
        
        # 5. ML anomaly detection
        if len(anomaly_scores) > 0:
            anomaly_score = anomaly_scores.iloc[-1]
            # Strong anomalies might indicate reversal points
            if abs(anomaly_score) > 1.0:
                signal_strength *= 0.7  # Reduce confidence during anomalies
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.8)
        
        # 6. Volatility adjustment
        ac_volatility = recent_ac.std()
        if ac_volatility > recent_ac.abs().mean():
            signal_strength *= 0.8  # Reduce signal strength in high volatility
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.9)
        
        # Determine final signal
        signal = SignalType.NEUTRAL
        if signal_strength > 0.5:
            signal = SignalType.STRONG_BUY if signal_strength > 1.5 else SignalType.BUY
        elif signal_strength < -0.5:
            signal = SignalType.STRONG_SELL if signal_strength < -1.5 else SignalType.SELL
        
        # Calculate confidence
        confidence = min(np.mean(confidence_factors) * abs(signal_strength), 1.0)
        
        return signal, confidence
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate the Acceleration/Deceleration indicator with ML enhancement
        """
        try:
            # Calculate Awesome Oscillator
            ao = self._calculate_awesome_oscillator(data)
            
            # Calculate Acceleration/Deceleration
            ac = self._calculate_ac_core(ao)
            
            # ML-based anomaly detection
            anomaly_scores = self._detect_momentum_anomalies(ac, data)
            
            # Advanced signal generation
            signal, confidence = self._generate_advanced_signals(ac, ao, anomaly_scores)
            
            # Calculate additional metrics
            ac_velocity = ac.diff().iloc[-1] if len(ac) > 1 else 0.0
            ac_acceleration = ac.diff().diff().iloc[-1] if len(ac) > 2 else 0.0
            
            # Momentum strength analysis
            momentum_strength = self._analyze_momentum_strength(ac, ao)
            
            result = {
                'ac_value': float(ac.iloc[-1]),
                'ao_value': float(ao.iloc[-1]),
                'ac_velocity': float(ac_velocity),
                'ac_acceleration': float(ac_acceleration),
                'momentum_strength': momentum_strength,
                'signal': signal,
                'confidence': confidence,
                'anomaly_score': float(anomaly_scores.iloc[-1]) if len(anomaly_scores) > 0 else 0.0,
                'trend_quality': self._assess_trend_quality(ac, ao),
                'values_history': {
                    'ac': ac.tail(10).tolist(),
                    'ao': ao.tail(10).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate AC indicator: {str(e)}",
                cause=e
            )
    
    def _analyze_momentum_strength(self, ac: pd.Series, ao: pd.Series) -> str:
        """Analyze overall momentum strength"""
        current_ac = ac.iloc[-1]
        current_ao = ao.iloc[-1]
        
        # Classify momentum strength
        if abs(current_ac) > ac.std() * 2:
            if current_ac > 0:
                return "very_strong_bullish"
            else:
                return "very_strong_bearish"
        elif abs(current_ac) > ac.std():
            if current_ac > 0:
                return "strong_bullish"
            else:
                return "strong_bearish"
        elif abs(current_ac) > ac.std() * 0.5:
            if current_ac > 0:
                return "moderate_bullish"
            else:
                return "moderate_bearish"
        else:
            return "weak"
    
    def _assess_trend_quality(self, ac: pd.Series, ao: pd.Series) -> float:
        """Assess the quality of the current trend"""
        if len(ac) < 10:
            return 0.5
        
        recent_ac = ac.tail(10)
        recent_ao = ao.tail(10)
        
        # Check for consistency
        ac_consistency = len([x for x in recent_ac if x * recent_ac.iloc[-1] > 0]) / len(recent_ac)
        ao_consistency = len([x for x in recent_ao if x * recent_ao.iloc[-1] > 0]) / len(recent_ao)
        
        # Check for momentum alignment
        alignment = 1.0 if (recent_ac.iloc[-1] > 0 and recent_ao.iloc[-1] > 0) or \
                          (recent_ac.iloc[-1] < 0 and recent_ao.iloc[-1] < 0) else 0.5
        
        quality = (ac_consistency * 0.4 + ao_consistency * 0.4 + alignment * 0.2)
        return float(quality)
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate signal from calculated values"""
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get calculation metadata"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'acceleration_deceleration',
            'ml_trained': self.ml_trained,
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata