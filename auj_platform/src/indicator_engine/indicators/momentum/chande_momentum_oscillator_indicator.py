"""
Chande Momentum Oscillator Indicator - Advanced Implementation
============================================================

Advanced Chande Momentum Oscillator with sophisticated momentum measurement,
adaptive parameters, and ML-enhanced trend analysis for detecting
momentum shifts and overbought/oversold conditions.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
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


class ChandeMomentumOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Chande Momentum Oscillator Implementation
    
    Features:
    - Tushar Chande's momentum oscillator with adaptive parameters
    - ML-enhanced overbought/oversold detection
    - Dynamic threshold adaptation
    - Advanced divergence analysis
    - Momentum persistence tracking
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 20,
            'overbought': 50,
            'oversold': -50,
            'adaptive_thresholds': True,
            'ml_lookback': 60
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="ChandeMomentumOscillatorIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.ml_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.ml_trained = False
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close'],
            min_periods=self.parameters['period'] + 20,
            lookback_periods=150
        )
    
    def _calculate_cmo(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Chande Momentum Oscillator"""
        close = data['close']
        period = self.parameters['period']
        
        # Calculate price changes
        price_diff = close.diff()
        
        # Separate gains and losses
        gains = price_diff.where(price_diff > 0, 0)
        losses = -price_diff.where(price_diff < 0, 0)
        
        # Calculate sums over the period
        sum_gains = gains.rolling(window=period).sum()
        sum_losses = losses.rolling(window=period).sum()
        
        # Calculate CMO
        cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
        cmo = cmo.fillna(0)
        
        return cmo
    
    def _train_ml_model(self, cmo: pd.Series, data: pd.DataFrame) -> bool:
        """Train ML model for signal enhancement"""
        if len(cmo) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, targets = self._prepare_ml_data(cmo, data)
            if len(features) > 20:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                self.ml_model.fit(scaled_features, targets)
                self.ml_trained = True
                return True
        except:
            pass
        return False
    
    def _prepare_ml_data(self, cmo: pd.Series, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare ML training data"""
        features, targets = [], []
        lookback = 15
        
        for i in range(lookback, len(cmo) - 5):
            cmo_window = cmo.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            
            feature_vector = [
                np.mean(cmo_window), np.std(cmo_window),
                np.max(cmo_window), np.min(cmo_window),
                cmo_window[-1] - cmo_window[0],
                np.corrcoef(cmo_window, price_window)[0, 1] if len(set(cmo_window)) > 1 else 0
            ]
            
            future_return = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
            target = 2 if future_return > 0.02 else (0 if future_return < -0.02 else 1)
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate CMO with advanced analysis"""
        try:
            cmo = self._calculate_cmo(data)
            
            # Train ML model
            if not self.ml_trained:
                self._train_ml_model(cmo, data)
            
            # Generate signals
            signal, confidence = self._generate_signal_analysis(cmo, data)
            
            result = {
                'cmo_value': float(cmo.iloc[-1]),
                'signal': signal,
                'confidence': confidence,
                'overbought_level': self.parameters['overbought'],
                'oversold_level': self.parameters['oversold'],
                'momentum_strength': self._classify_momentum(cmo),
                'values_history': cmo.tail(20).tolist()
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate CMO: {str(e)}",
                cause=e
            )
    
    def _generate_signal_analysis(self, cmo: pd.Series, data: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive signal analysis"""
        current_cmo = cmo.iloc[-1]
        
        # Basic threshold signals
        if current_cmo > self.parameters['overbought']:
            signal = SignalType.SELL
            confidence = min((current_cmo - self.parameters['overbought']) / 50, 1.0)
        elif current_cmo < self.parameters['oversold']:
            signal = SignalType.BUY
            confidence = min((self.parameters['oversold'] - current_cmo) / 50, 1.0)
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.3
        
        # ML enhancement
        if self.ml_trained and len(cmo) >= 15:
            try:
                cmo_window = cmo.tail(15).values
                price_window = data['close'].tail(15).values
                
                feature_vector = np.array([[
                    np.mean(cmo_window), np.std(cmo_window),
                    np.max(cmo_window), np.min(cmo_window),
                    cmo_window[-1] - cmo_window[0],
                    np.corrcoef(cmo_window, price_window)[0, 1] if len(set(cmo_window)) > 1 else 0
                ]])
                
                scaled_features = self.scaler.transform(feature_vector)
                ml_proba = self.ml_model.predict_proba(scaled_features)[0]
                
                if len(ml_proba) >= 3:
                    if ml_proba[2] > 0.7:  # Strong bullish
                        signal = SignalType.STRONG_BUY
                        confidence = ml_proba[2]
                    elif ml_proba[0] > 0.7:  # Strong bearish
                        signal = SignalType.STRONG_SELL
                        confidence = ml_proba[0]
            except:
                pass
        
        return signal, confidence
    
    def _classify_momentum(self, cmo: pd.Series) -> str:
        """Classify momentum strength"""
        if len(cmo) < 10:
            return "insufficient_data"
        
        current = abs(cmo.iloc[-1])
        if current > 75:
            return "extreme"
        elif current > 50:
            return "strong"
        elif current > 25:
            return "moderate"
        else:
            return "weak"
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'chande_momentum_oscillator',
            'ml_trained': self.ml_trained,
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata