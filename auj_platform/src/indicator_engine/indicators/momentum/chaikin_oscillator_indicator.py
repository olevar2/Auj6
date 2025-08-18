"""
Chaikin Oscillator Indicator - Advanced Implementation
===================================================

Advanced Chaikin Oscillator with volume-weighted momentum analysis,
ML optimization, and sophisticated accumulation/distribution flow detection
for identifying institutional money flow and market momentum shifts.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
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


class ChaikinOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Chaikin Oscillator Indicator Implementation
    
    Features:
    - Volume-weighted accumulation/distribution analysis
    - ML-optimized parameter adaptation
    - Institutional flow detection
    - Advanced divergence analysis
    - Smart money detection algorithms
    - Multi-timeframe convergence analysis
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'fast_period': 3,            # Fast EMA period for CHO
            'slow_period': 10,           # Slow EMA period for CHO
            'volume_factor': 1000,       # Volume scaling factor
            'signal_threshold': 0.05,    # Signal generation threshold
            'ml_lookback': 50,           # ML analysis period
            'institutional_threshold': 2.0,  # Institutional flow detection
            'divergence_periods': 20,    # Divergence analysis period
            'smart_money_detection': True,   # Enable smart money analysis
            'adaptive_periods': True     # Enable adaptive periods
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="ChaikinOscillatorIndicator", parameters=default_params)
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.ml_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        self.ml_trained = False
        
        # Analysis storage
        self.flow_history = []
    
    def get_data_requirements(self) -> DataRequirement:
        """Define data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(self.parameters['slow_period'], self.parameters['ml_lookback']) + 20,
            lookback_periods=200
        )
    
    def validate_parameters(self) -> bool:
        """Validate parameters"""
        if self.parameters['fast_period'] >= self.parameters['slow_period']:
            raise ValueError("Fast period must be less than slow period")
        return True
    
    def _calculate_accumulation_distribution_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.fillna(0)  # Handle division by zero
        
        money_flow_volume = clv * data['volume']
        ad_line = money_flow_volume.cumsum()
        
        return ad_line
    
    def _calculate_chaikin_oscillator(self, ad_line: pd.Series) -> pd.Series:
        """Calculate Chaikin Oscillator"""
        if self.parameters['adaptive_periods']:
            # Adapt periods based on market volatility
            volatility = ad_line.pct_change().rolling(window=20).std()
            volatility_factor = volatility / volatility.mean()
            
            fast_period = max(2, int(self.parameters['fast_period'] * volatility_factor.iloc[-1]))
            slow_period = max(fast_period + 1, int(self.parameters['slow_period'] * volatility_factor.iloc[-1]))
        else:
            fast_period = self.parameters['fast_period']
            slow_period = self.parameters['slow_period']
        
        fast_ema = ad_line.ewm(span=fast_period).mean()
        slow_ema = ad_line.ewm(span=slow_period).mean()
        
        chaikin_osc = fast_ema - slow_ema
        return chaikin_osc
    
    def _detect_institutional_flows(self, chaikin_osc: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect institutional money flows"""
        if len(chaikin_osc) < 20:
            return {'detected': False, 'direction': 'neutral', 'strength': 0.0}
        
        # Volume-weighted analysis
        volume_sma = data['volume'].rolling(window=20).mean()
        volume_ratio = data['volume'] / volume_sma
        
        # Institutional flow indicators
        weighted_cho = chaikin_osc * volume_ratio
        
        # Detect unusual institutional activity
        cho_threshold = weighted_cho.rolling(window=20).std() * self.parameters['institutional_threshold']
        current_weighted = weighted_cho.iloc[-1]
        
        if abs(current_weighted) > cho_threshold.iloc[-1]:
            detected = True
            direction = 'bullish' if current_weighted > 0 else 'bearish'
            strength = min(abs(current_weighted) / cho_threshold.iloc[-1], 2.0) / 2.0
        else:
            detected = False
            direction = 'neutral'
            strength = 0.0
        
        return {
            'detected': detected,
            'direction': direction,
            'strength': strength,
            'weighted_cho': float(current_weighted)
        }
    
    def _detect_smart_money(self, chaikin_osc: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect smart money activity"""
        if not self.parameters['smart_money_detection'] or len(chaikin_osc) < 15:
            return {'activity': False, 'pattern': 'none', 'confidence': 0.0}
        
        recent_cho = chaikin_osc.tail(15)
        recent_volume = data['volume'].tail(15)
        recent_prices = data['close'].tail(15)
        
        # Smart money patterns
        # 1. Accumulation on weakness (price down, CHO up)
        price_change = recent_prices.diff().sum()
        cho_change = recent_cho.diff().sum()
        
        if price_change < 0 and cho_change > 0:
            pattern = 'accumulation_on_weakness'
            confidence = min(abs(cho_change) / recent_cho.std(), 1.0)
        # 2. Distribution on strength (price up, CHO down)
        elif price_change > 0 and cho_change < 0:
            pattern = 'distribution_on_strength'
            confidence = min(abs(cho_change) / recent_cho.std(), 1.0)
        # 3. Volume divergence
        elif self._detect_volume_divergence(recent_cho, recent_volume, recent_prices):
            pattern = 'volume_divergence'
            confidence = 0.7
        else:
            pattern = 'none'
            confidence = 0.0
        
        return {
            'activity': pattern != 'none',
            'pattern': pattern,
            'confidence': confidence
        }
    
    def _detect_volume_divergence(self, cho: pd.Series, volume: pd.Series, prices: pd.Series) -> bool:
        """Detect volume divergence patterns"""
        try:
            # Check if volume is increasing while CHO is decreasing (or vice versa)
            volume_trend = volume.diff().sum()
            cho_trend = cho.diff().sum()
            
            return (volume_trend > 0 and cho_trend < 0) or (volume_trend < 0 and cho_trend > 0)
        except:
            return False
    
    def _train_ml_model(self, chaikin_osc: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML model for optimization"""
        if len(chaikin_osc) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(chaikin_osc, data)
            
            if len(features) > 20:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                self.ml_model.fit(scaled_features, targets)
                self.ml_trained = True
                return True
        except:
            pass
        
        return False
    
    def _prepare_ml_data(self, chaikin_osc: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features = []
        targets = []
        
        lookback = 20
        
        for i in range(lookback, len(chaikin_osc) - 5):
            cho_window = chaikin_osc.iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            feature_vector = [
                np.mean(cho_window),
                np.std(cho_window),
                np.max(cho_window),
                np.min(cho_window),
                cho_window[-1] - cho_window[0],
                np.mean(volume_window),
                np.std(volume_window),
                np.corrcoef(cho_window, volume_window)[0, 1] if len(set(cho_window)) > 1 else 0,
                np.corrcoef(cho_window, price_window)[0, 1] if len(set(cho_window)) > 1 else 0,
                (price_window[-1] - price_window[0]) / price_window[0] if price_window[0] != 0 else 0
            ]
            
            # Target: future CHO direction
            future_cho = chaikin_osc.iloc[i+5] if i+5 < len(chaikin_osc) else chaikin_osc.iloc[-1]
            target = 1 if future_cho > chaikin_osc.iloc[i] else 0
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def _generate_ml_signal(self, chaikin_osc: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate ML-based signal"""
        if not self.ml_trained or len(chaikin_osc) < 20:
            return None, 0.0
        
        try:
            lookback = 20
            cho_window = chaikin_osc.tail(lookback).values
            volume_window = data['volume'].tail(lookback).values
            price_window = data['close'].tail(lookback).values
            
            feature_vector = np.array([[
                np.mean(cho_window),
                np.std(cho_window),
                np.max(cho_window),
                np.min(cho_window),
                cho_window[-1] - cho_window[0],
                np.mean(volume_window),
                np.std(volume_window),
                np.corrcoef(cho_window, volume_window)[0, 1] if len(set(cho_window)) > 1 else 0,
                np.corrcoef(cho_window, price_window)[0, 1] if len(set(cho_window)) > 1 else 0,
                (price_window[-1] - price_window[0]) / price_window[0] if price_window[0] != 0 else 0
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            prediction = self.ml_model.predict(scaled_features)[0]
            
            if prediction > 0.6:
                return SignalType.BUY, prediction
            elif prediction < 0.4:
                return SignalType.SELL, 1 - prediction
        except:
            pass
        
        return None, 0.0
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Chaikin Oscillator with advanced analysis"""
        try:
            # Calculate A/D Line
            ad_line = self._calculate_accumulation_distribution_line(data)
            
            # Calculate Chaikin Oscillator
            chaikin_osc = self._calculate_chaikin_oscillator(ad_line)
            
            # Detect institutional flows
            institutional_flows = self._detect_institutional_flows(chaikin_osc, data)
            
            # Detect smart money activity
            smart_money = self._detect_smart_money(chaikin_osc, data)
            
            # Train ML model
            if not self.ml_trained:
                self._train_ml_model(chaikin_osc, data)
            
            # Generate ML signal
            ml_signal, ml_confidence = self._generate_ml_signal(chaikin_osc, data)
            
            # Generate composite signal
            signal, confidence = self._generate_composite_signal(
                chaikin_osc, institutional_flows, smart_money, ml_signal, ml_confidence
            )
            
            result = {
                'chaikin_oscillator': float(chaikin_osc.iloc[-1]),
                'ad_line': float(ad_line.iloc[-1]),
                'signal': signal,
                'confidence': confidence,
                'institutional_flows': institutional_flows,
                'smart_money': smart_money,
                'ml_signal': ml_signal,
                'ml_confidence': ml_confidence,
                'momentum_strength': self._classify_momentum(chaikin_osc),
                'flow_analysis': self._analyze_flow_characteristics(chaikin_osc, data),
                'values_history': {
                    'chaikin_osc': chaikin_osc.tail(20).tolist(),
                    'ad_line': ad_line.tail(20).tolist()
                }
            }
            
            # Update history
            self.flow_history.append({
                'timestamp': data.index[-1],
                'chaikin_osc': float(chaikin_osc.iloc[-1]),
                'institutional': institutional_flows,
                'smart_money': smart_money
            })
            
            if len(self.flow_history) > 100:
                self.flow_history = self.flow_history[-100:]
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Chaikin Oscillator: {str(e)}",
                cause=e
            )
    
    def _generate_composite_signal(self, chaikin_osc: pd.Series, institutional: Dict, 
                                  smart_money: Dict, ml_signal: Optional[SignalType], 
                                  ml_confidence: float) -> Tuple[SignalType, float]:
        """Generate composite signal"""
        signal_components = []
        confidence_components = []
        
        current_cho = chaikin_osc.iloc[-1]
        
        # Basic CHO signal
        if current_cho > self.parameters['signal_threshold']:
            signal_components.append(1.0)
            confidence_components.append(0.6)
        elif current_cho < -self.parameters['signal_threshold']:
            signal_components.append(-1.0)
            confidence_components.append(0.6)
        
        # Institutional flow signals
        if institutional['detected']:
            if institutional['direction'] == 'bullish':
                signal_components.append(institutional['strength'])
            else:
                signal_components.append(-institutional['strength'])
            confidence_components.append(institutional['strength'])
        
        # Smart money signals
        if smart_money['activity']:
            if smart_money['pattern'] == 'accumulation_on_weakness':
                signal_components.append(smart_money['confidence'])
            elif smart_money['pattern'] == 'distribution_on_strength':
                signal_components.append(-smart_money['confidence'])
            confidence_components.append(smart_money['confidence'])
        
        # ML signal
        if ml_signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            signal_components.append(ml_confidence)
            confidence_components.append(ml_confidence)
        elif ml_signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            signal_components.append(-ml_confidence)
            confidence_components.append(ml_confidence)
        
        # Calculate final signal
        if signal_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        if weighted_signal > 0.3:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.7 else SignalType.BUY
        elif weighted_signal < -0.3:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.7 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _classify_momentum(self, chaikin_osc: pd.Series) -> str:
        """Classify momentum strength"""
        if len(chaikin_osc) < 20:
            return "insufficient_data"
        
        current = chaikin_osc.iloc[-1]
        std_dev = chaikin_osc.tail(20).std()
        
        if abs(current) > std_dev * 2:
            return "very_strong"
        elif abs(current) > std_dev * 1.5:
            return "strong"
        elif abs(current) > std_dev:
            return "moderate"
        else:
            return "weak"
    
    def _analyze_flow_characteristics(self, chaikin_osc: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze flow characteristics"""
        if len(chaikin_osc) < 10:
            return {}
        
        recent_cho = chaikin_osc.tail(10)
        recent_volume = data['volume'].tail(10)
        
        return {
            'flow_consistency': len([x for x in recent_cho if x * recent_cho.iloc[-1] > 0]) / len(recent_cho),
            'volume_support': recent_volume.iloc[-1] / recent_volume.mean(),
            'flow_acceleration': recent_cho.diff().iloc[-1],
            'trend_strength': abs(recent_cho.mean()) / (recent_cho.std() if recent_cho.std() != 0 else 1)
        }
    
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
            'indicator_category': 'chaikin_oscillator',
            'ml_trained': self.ml_trained,
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata