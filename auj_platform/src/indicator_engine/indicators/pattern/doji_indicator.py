"""
Doji Pattern Indicator - Advanced Candlestick Pattern Recognition
================================================================

This indicator implements sophisticated doji pattern detection with machine learning 
classification, volatility context analysis, and trend-aware signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import talib

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    IndicatorResult, 
    SignalType, 
    DataType, 
    DataRequirement
)
from ...core.exceptions import IndicatorCalculationException


@dataclass
class DojiClassification:
    """Classification result for doji patterns"""
    is_doji: bool
    doji_type: str
    strength: float  # 0.0 to 1.0
    body_to_range_ratio: float
    upper_shadow_ratio: float
    lower_shadow_ratio: float
    market_context: str
    trend_context: str


class DojiIndicator(StandardIndicatorInterface):
    """
    Advanced Doji Pattern Indicator
    
    Features:
    - Multi-type doji classification (standard, dragonfly, gravestone, long-legged)
    - Machine learning anomaly detection for pattern validation
    - Volatility-adjusted thresholds
    - Trend context analysis
    - Volume confirmation
    - Market regime awareness
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'doji_threshold': 0.1,  # Max body-to-range ratio for doji
            'volume_confirmation': True,
            'trend_lookback': 20,
            'volatility_adjustment': True,
            'min_shadow_ratio': 0.3,  # Min shadow for dragonfly/gravestone
            'ml_anomaly_detection': True,
            'market_regime_sensitivity': 0.7
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="DojiIndicator", parameters=default_params)
        
        # Initialize ML components
        self.anomaly_detector = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Pattern classification thresholds
        self.thresholds = {
            'standard_doji': {'body_ratio': 0.1, 'shadow_balance': 0.3},
            'dragonfly_doji': {'body_ratio': 0.1, 'lower_shadow': 0.6},
            'gravestone_doji': {'body_ratio': 0.1, 'upper_shadow': 0.6},
            'long_legged_doji': {'body_ratio': 0.05, 'total_shadow': 0.8}
        }
        
        logging.info(f"DojiIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=30,
            lookback_periods=100
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate doji patterns with advanced analysis"""
        try:
            if len(data) < self.parameters['trend_lookback']:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < {self.parameters['trend_lookback']}"
                )
            
            # Calculate candlestick metrics
            candle_metrics = self._calculate_candle_metrics(data)
            
            # Determine market context
            market_context = self._analyze_market_context(data, candle_metrics)
            
            # Classify doji patterns
            doji_classifications = self._classify_doji_patterns(candle_metrics, market_context)
            
            # Apply ML anomaly detection if enabled
            if self.parameters['ml_anomaly_detection']:
                doji_classifications = self._apply_ml_validation(
                    doji_classifications, candle_metrics
                )
            
            # Generate pattern summary
            pattern_summary = self._generate_pattern_summary(doji_classifications)
            
            return {
                'current_doji': doji_classifications[-1] if doji_classifications else None,
                'doji_history': doji_classifications[-10:],  # Last 10 periods
                'pattern_summary': pattern_summary,
                'market_context': market_context,
                'candle_metrics': candle_metrics.iloc[-1].to_dict()
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Doji calculation failed: {str(e)}", e
            )
    
    def _calculate_candle_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive candlestick metrics"""
        df = data.copy()
        
        # Basic candle components
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Ratios and percentages
        df['body_to_range'] = np.where(df['total_range'] > 0, 
                                      df['body'] / df['total_range'], 0)
        df['upper_shadow_ratio'] = np.where(df['total_range'] > 0,
                                          df['upper_shadow'] / df['total_range'], 0)
        df['lower_shadow_ratio'] = np.where(df['total_range'] > 0,
                                          df['lower_shadow'] / df['total_range'], 0)
        
        # Advanced metrics
        df['shadow_balance'] = abs(df['upper_shadow_ratio'] - df['lower_shadow_ratio'])
        df['total_shadow_ratio'] = df['upper_shadow_ratio'] + df['lower_shadow_ratio']
        
        # Volatility context
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility_percentile'] = df['atr'].rolling(50).rank(pct=True)
        
        # Volume context
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def _analyze_market_context(self, data: pd.DataFrame, metrics: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market context and regime"""
        df = data.copy()
        
        # Trend analysis
        sma_short = talib.SMA(df['close'], timeperiod=10)
        sma_long = talib.SMA(df['close'], timeperiod=50)
        
        current_trend = "neutral"
        if sma_short.iloc[-1] > sma_long.iloc[-1] * 1.02:
            current_trend = "uptrend"
        elif sma_short.iloc[-1] < sma_long.iloc[-1] * 0.98:
            current_trend = "downtrend"
        
        # Volatility regime
        current_volatility = metrics['volatility_percentile'].iloc[-1]
        volatility_regime = "normal"
        if current_volatility > 0.8:
            volatility_regime = "high"
        elif current_volatility < 0.2:
            volatility_regime = "low"
        
        # Support/Resistance levels
        recent_highs = df['high'].rolling(20).max()
        recent_lows = df['low'].rolling(20).min()
        current_price = df['close'].iloc[-1]
        
        near_resistance = abs(current_price - recent_highs.iloc[-1]) / current_price < 0.02
        near_support = abs(current_price - recent_lows.iloc[-1]) / current_price < 0.02
        
        return {
            'trend': current_trend,
            'volatility_regime': volatility_regime,
            'volatility_percentile': current_volatility,
            'near_resistance': near_resistance,
            'near_support': near_support,
            'atr_current': metrics['atr'].iloc[-1],
            'volume_context': 'high' if metrics['volume_ratio'].iloc[-1] > 1.5 else 'normal'
        }
    
    def _classify_doji_patterns(self, metrics: pd.DataFrame, context: Dict[str, Any]) -> List[DojiClassification]:
        """Classify doji patterns with context awareness"""
        classifications = []
        
        for i in range(len(metrics)):
            row = metrics.iloc[i]
            
            # Adjust thresholds based on volatility
            volatility_adjustment = 1.0
            if self.parameters['volatility_adjustment']:
                vol_percentile = row['volatility_percentile']
                volatility_adjustment = 0.5 + vol_percentile  # 0.5 to 1.5 range
            
            adjusted_threshold = self.parameters['doji_threshold'] * volatility_adjustment
            
            # Base doji check
            is_base_doji = row['body_to_range'] <= adjusted_threshold
            
            if not is_base_doji:
                classifications.append(DojiClassification(
                    is_doji=False, doji_type="none", strength=0.0,
                    body_to_range_ratio=row['body_to_range'],
                    upper_shadow_ratio=row['upper_shadow_ratio'],
                    lower_shadow_ratio=row['lower_shadow_ratio'],
                    market_context=context['volatility_regime'],
                    trend_context=context['trend']
                ))
                continue
            
            # Determine doji type and strength
            doji_type, strength = self._determine_doji_type(row, adjusted_threshold)
            
            # Apply volume confirmation if enabled
            if self.parameters['volume_confirmation']:
                volume_factor = min(row['volume_ratio'], 2.0) / 2.0
                strength *= (0.7 + 0.3 * volume_factor)
            
            classifications.append(DojiClassification(
                is_doji=True,
                doji_type=doji_type,
                strength=strength,
                body_to_range_ratio=row['body_to_range'],
                upper_shadow_ratio=row['upper_shadow_ratio'],
                lower_shadow_ratio=row['lower_shadow_ratio'],
                market_context=context['volatility_regime'],
                trend_context=context['trend']
            ))
        
        return classifications
    
    def _determine_doji_type(self, row: pd.Series, threshold: float) -> Tuple[str, float]:
        """Determine specific doji type and calculate strength"""
        body_ratio = row['body_to_range']
        upper_ratio = row['upper_shadow_ratio']
        lower_ratio = row['lower_shadow_ratio']
        shadow_balance = row['shadow_balance']
        
        # Long-legged doji (very small body, long shadows)
        if (body_ratio <= threshold * 0.5 and 
            row['total_shadow_ratio'] >= 0.8):
            strength = 0.9 - body_ratio * 5  # Higher strength for smaller body
            return "long_legged_doji", min(strength, 1.0)
        
        # Dragonfly doji (small body, long lower shadow, small upper shadow)
        if (body_ratio <= threshold and 
            lower_ratio >= 0.6 and 
            upper_ratio <= 0.2):
            strength = 0.8 - body_ratio * 3 + lower_ratio * 0.3
            return "dragonfly_doji", min(strength, 1.0)
        
        # Gravestone doji (small body, long upper shadow, small lower shadow)
        if (body_ratio <= threshold and 
            upper_ratio >= 0.6 and 
            lower_ratio <= 0.2):
            strength = 0.8 - body_ratio * 3 + upper_ratio * 0.3
            return "gravestone_doji", min(strength, 1.0)
        
        # Standard doji
        if body_ratio <= threshold:
            strength = 0.6 - body_ratio * 2 - shadow_balance * 0.3
            return "standard_doji", max(min(strength, 1.0), 0.1)
        
        return "none", 0.0
    
    def _apply_ml_validation(self, classifications: List[DojiClassification], 
                           metrics: pd.DataFrame) -> List[DojiClassification]:
        """Apply machine learning validation to pattern classifications"""
        try:
            # Prepare features for ML validation
            features = []
            for i, classification in enumerate(classifications):
                if i >= len(metrics):
                    break
                    
                row = metrics.iloc[i]
                feature_vector = [
                    row['body_to_range'],
                    row['upper_shadow_ratio'],
                    row['lower_shadow_ratio'],
                    row['shadow_balance'],
                    row['total_shadow_ratio'],
                    row['volatility_percentile'],
                    row['volume_ratio']
                ]
                features.append(feature_vector)
            
            if len(features) < 10:  # Not enough data for ML
                return classifications
            
            features_array = np.array(features)
            
            # Fit or use existing model
            if not self.is_fitted and len(features) >= 30:
                # Scale features
                features_scaled = self.scaler.fit_transform(features_array)
                # Fit anomaly detector
                self.anomaly_detector.fit(features_scaled)
                self.is_fitted = True
            elif self.is_fitted:
                # Apply anomaly detection
                features_scaled = self.scaler.transform(features_array)
                anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
                
                # Adjust classification strengths based on anomaly scores
                for i, (classification, score) in enumerate(zip(classifications, anomaly_scores)):
                    if classification.is_doji:
                        # Normalize anomaly score to 0-1 range
                        normalized_score = (score + 0.5) / 1.0  # Approximate normalization
                        normalized_score = max(0, min(1, normalized_score))
                        
                        # Adjust strength
                        classification.strength *= (0.5 + 0.5 * normalized_score)
            
            return classifications
            
        except Exception as e:
            logging.warning(f"ML validation failed: {str(e)}")
            return classifications
    
    def _generate_pattern_summary(self, classifications: List[DojiClassification]) -> Dict[str, Any]:
        """Generate summary statistics for doji patterns"""
        if not classifications:
            return {}
        
        recent_classifications = classifications[-20:]  # Last 20 periods
        doji_count = sum(1 for c in recent_classifications if c.is_doji)
        
        # Count by type
        type_counts = {}
        total_strength = 0.0
        
        for classification in recent_classifications:
            if classification.is_doji:
                type_counts[classification.doji_type] = type_counts.get(classification.doji_type, 0) + 1
                total_strength += classification.strength
        
        avg_strength = total_strength / max(doji_count, 1)
        
        return {
            'doji_frequency': doji_count / len(recent_classifications),
            'type_distribution': type_counts,
            'average_strength': avg_strength,
            'recent_doji_count': doji_count,
            'strongest_pattern': max(recent_classifications, 
                                   key=lambda x: x.strength if x.is_doji else 0).doji_type if doji_count > 0 else None
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on doji analysis"""
        current_doji = value.get('current_doji')
        market_context = value.get('market_context', {})
        
        if not current_doji or not current_doji.is_doji:
            return None, 0.0
        
        # Signal generation based on doji type and context
        base_confidence = current_doji.strength
        
        # Dragonfly doji in downtrend - potential bullish reversal
        if (current_doji.doji_type == "dragonfly_doji" and 
            current_doji.trend_context == "downtrend"):
            signal_confidence = base_confidence * 0.8
            if signal_confidence > 0.6:
                return SignalType.BUY, signal_confidence
        
        # Gravestone doji in uptrend - potential bearish reversal
        elif (current_doji.doji_type == "gravestone_doji" and 
              current_doji.trend_context == "uptrend"):
            signal_confidence = base_confidence * 0.8
            if signal_confidence > 0.6:
                return SignalType.SELL, signal_confidence
        
        # Long-legged doji - indecision, potential reversal
        elif current_doji.doji_type == "long_legged_doji":
            if market_context.get('near_resistance') and current_doji.trend_context == "uptrend":
                return SignalType.SELL, base_confidence * 0.6
            elif market_context.get('near_support') and current_doji.trend_context == "downtrend":
                return SignalType.BUY, base_confidence * 0.6
        
        # Standard doji - neutral/hold signal
        return SignalType.NEUTRAL, base_confidence * 0.4
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_fitted,
            'pattern_types_detected': len(self.thresholds),
            'volatility_adjustment': self.parameters['volatility_adjustment'],
            'volume_confirmation': self.parameters['volume_confirmation']
        })
        return base_metadata