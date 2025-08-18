"""
Commodity Channel Index Indicator - Advanced Implementation
=========================================================

Advanced CCI with sophisticated mean reversion analysis, ML enhancement,
and dynamic threshold adaptation for detecting cyclical turning points
and overbought/oversold conditions.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class CommodityChannelIndexIndicator(StandardIndicatorInterface):
    """Advanced Commodity Channel Index Implementation"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 20,
            'constant': 0.015,
            'overbought': 100,
            'oversold': -100,
            'ml_lookback': 80
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="CommodityChannelIndexIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.ml_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.ml_trained = False
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close'],
            min_periods=self.parameters['period'] + 20,
            lookback_periods=150
        )
    
    def _calculate_cci(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Commodity Channel Index"""
        # Calculate Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate moving average
        sma = typical_price.rolling(window=self.parameters['period']).mean()
        
        # Calculate mean deviation
        mean_dev = typical_price.rolling(window=self.parameters['period']).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        # Calculate CCI
        cci = (typical_price - sma) / (self.parameters['constant'] * mean_dev)
        
        return cci
    
    def _train_ml_model(self, cci: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML model for CCI enhancement"""
        if len(cci) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(cci, data)
            if len(features) > 20:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                self.ml_model.fit(scaled_features, targets)
                self.ml_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_ml_data(self, cci: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 20
        
        for i in range(lookback, len(cci) - 5):
            cci_window = cci.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            feature_vector = [
                np.mean(cci_window), np.std(cci_window),
                np.max(cci_window), np.min(cci_window),
                cci_window[-1], cci_window[-1] - cci_window[0],
                len([x for x in cci_window if x > 100]) / len(cci_window),  # Overbought ratio
                len([x for x in cci_window if x < -100]) / len(cci_window),  # Oversold ratio
            ]
            
            future_cci = cci.iloc[i+5] if i+5 < len(cci) else cci.iloc[-1]
            target = future_cci
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate CCI with advanced analysis"""
        try:
            cci = self._calculate_cci(data)
            
            # Train ML model
            if not self.ml_trained:
                self._train_ml_model(cci, data)
            
            # Generate analysis
            signal, confidence = self._generate_cci_analysis(cci, data)
            
            result = {
                'cci_value': float(cci.iloc[-1]),
                'signal': signal,
                'confidence': confidence,
                'overbought_level': self.parameters['overbought'],
                'oversold_level': self.parameters['oversold'],
                'mean_reversion_strength': self._calculate_mean_reversion_strength(cci),
                'cycle_analysis': self._analyze_cycles(cci),
                'values_history': cci.tail(20).tolist()
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate CCI: {str(e)}",
                cause=e
            )
    
    def _generate_cci_analysis(self, cci: pd.Series, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate CCI signal analysis"""
        current_cci = cci.iloc[-1]
        
        # Basic CCI signals
        if current_cci > self.parameters['overbought']:
            signal = SignalType.SELL
            confidence = min((current_cci - self.parameters['overbought']) / 200, 1.0)
        elif current_cci < self.parameters['oversold']:
            signal = SignalType.BUY
            confidence = min((self.parameters['oversold'] - current_cci) / 200, 1.0)
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.3
        
        # ML enhancement
        if self.ml_trained and len(cci) >= 20:
            try:
                cci_window = cci.tail(20).values
                
                feature_vector = np.array([[
                    np.mean(cci_window), np.std(cci_window),
                    np.max(cci_window), np.min(cci_window),
                    cci_window[-1], cci_window[-1] - cci_window[0],
                    len([x for x in cci_window if x > 100]) / len(cci_window),
                    len([x for x in cci_window if x < -100]) / len(cci_window),
                ]])
                
                scaled_features = self.scaler.transform(feature_vector)
                ml_prediction = self.ml_model.predict(scaled_features)[0]
                
                # Adjust signal based on ML prediction
                if ml_prediction > 150 and signal != SignalType.SELL:
                    signal = SignalType.SELL
                    confidence = 0.8
                elif ml_prediction < -150 and signal != SignalType.BUY:
                    signal = SignalType.BUY
                    confidence = 0.8
            except:
                pass
        
        return signal, confidence
    
    def _calculate_mean_reversion_strength(self, cci: pd.Series) -> str:
        """Calculate mean reversion strength"""
        if len(cci) < 20:
            return "insufficient_data"
        
        recent_cci = cci.tail(20)
        extreme_count = len([x for x in recent_cci if abs(x) > 100])
        
        if extreme_count > 15:
            return "very_strong"
        elif extreme_count > 10:
            return "strong"
        elif extreme_count > 5:
            return "moderate"
        else:
            return "weak"
    
    def _analyze_cycles(self, cci: pd.Series) -> Dict[str, Any]:
        """Analyze CCI cycles"""
        if len(cci) < 40:
            return {"cycle_detected": False, "cycle_length": 0, "cycle_phase": "unknown"}
        
        # Simple cycle detection using zero crossings
        zero_crossings = []
        for i in range(1, len(cci)):
            if (cci.iloc[i-1] < 0 and cci.iloc[i] > 0) or (cci.iloc[i-1] > 0 and cci.iloc[i] < 0):
                zero_crossings.append(i)
        
        if len(zero_crossings) >= 4:
            cycle_lengths = [zero_crossings[i] - zero_crossings[i-1] for i in range(1, len(zero_crossings))]
            avg_cycle_length = np.mean(cycle_lengths)
            
            # Determine current cycle phase
            last_crossing = zero_crossings[-1]
            bars_since_crossing = len(cci) - last_crossing
            cycle_progress = bars_since_crossing / avg_cycle_length
            
            if cycle_progress < 0.25:
                phase = "early"
            elif cycle_progress < 0.75:
                phase = "middle"
            else:
                phase = "late"
            
            return {
                "cycle_detected": True,
                "cycle_length": int(avg_cycle_length),
                "cycle_phase": phase,
                "cycle_progress": float(cycle_progress)
            }
        
        return {"cycle_detected": False, "cycle_length": 0, "cycle_phase": "unknown"}
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'commodity_channel_index',
            'ml_trained': self.ml_trained,
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata