"""
Doji Star Pattern Indicator - Advanced Multi-Candle Pattern Recognition
======================================================================

This indicator implements sophisticated doji star pattern detection with gap analysis,
machine learning validation, and trend reversal prediction capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
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
class DojiStarPattern:
    """Represents a detected doji star pattern"""
    pattern_type: str  # 'morning_doji_star' or 'evening_doji_star'
    start_index: int
    end_index: int
    strength: float
    gap_size: float
    confirmation_strength: float
    volume_confirmation: bool
    trend_context: str
    support_resistance_level: float


class DojiStarIndicator(StandardIndicatorInterface):
    """
    Advanced Doji Star Pattern Indicator
    
    Features:
    - Morning and evening doji star pattern detection
    - Gap analysis and validation
    - Volume confirmation analysis
    - Machine learning pattern validation
    - Support/resistance level integration
    - Trend context analysis
    - Multi-timeframe pattern strength assessment
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'min_gap_percentage': 0.2,  # Minimum gap size as % of ATR
            'doji_threshold': 0.15,     # Max body-to-range ratio for doji
            'volume_confirmation': True,
            'min_volume_ratio': 1.2,    # Minimum volume increase for confirmation
            'trend_lookback': 20,
            'pattern_lookback': 100,
            'ml_validation': True,
            'confirmation_candles': 2,  # Candles to wait for pattern confirmation
            'min_pattern_strength': 0.6
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="DojiStarIndicator", parameters=default_params)
        
        # Initialize ML components
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_patterns = []
        
        logging.info(f"DojiStarIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=50,
            lookback_periods=150
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate doji star patterns with advanced analysis"""
        try:
            if len(data) < self.parameters['pattern_lookback']:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < {self.parameters['pattern_lookback']}"
                )
            
            # Calculate candlestick and market metrics
            enhanced_data = self._enhance_data_with_metrics(data)
            
            # Detect doji star patterns
            detected_patterns = self._detect_doji_star_patterns(enhanced_data)
            
            # Validate patterns with ML if enabled
            if self.parameters['ml_validation'] and detected_patterns:
                validated_patterns = self._validate_patterns_with_ml(
                    detected_patterns, enhanced_data
                )
            else:
                validated_patterns = detected_patterns
            
            # Analyze current market state
            current_analysis = self._analyze_current_state(enhanced_data, validated_patterns)
            
            # Generate pattern statistics
            pattern_stats = self._generate_pattern_statistics(validated_patterns)
            
            return {
                'current_patterns': [p for p in validated_patterns if p.end_index >= len(data) - 5],
                'recent_patterns': validated_patterns[-10:],
                'pattern_statistics': pattern_stats,
                'current_analysis': current_analysis,
                'market_state': self._get_market_state(enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Doji star calculation failed: {str(e)}", e
            )
    
    def _enhance_data_with_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with comprehensive technical metrics"""
        df = data.copy()
        
        # Basic candlestick metrics
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['body_to_range'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
        
        # Gap analysis
        df['gap_up'] = df['low'] - df['high'].shift(1)
        df['gap_down'] = df['low'].shift(1) - df['high']
        df['has_gap_up'] = df['gap_up'] > 0
        df['has_gap_down'] = df['gap_down'] > 0
        
        # ATR for normalization
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['gap_up_normalized'] = df['gap_up'] / df['atr']
        df['gap_down_normalized'] = df['gap_down'] / df['atr']
        
        # Trend indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # Trend direction
        df['trend_short'] = np.where(df['close'] > df['sma_20'], 1, -1)
        df['trend_long'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_surge'] = df['volume_ratio'] > self.parameters['min_volume_ratio']
        
        # Support/Resistance levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['near_resistance'] = abs(df['close'] - df['resistance']) / df['atr'] < 0.5
        df['near_support'] = abs(df['close'] - df['support']) / df['atr'] < 0.5
        
        # Volatility percentile
        df['volatility_percentile'] = df['atr'].rolling(50).rank(pct=True)
        
        return df
    
    def _detect_doji_star_patterns(self, data: pd.DataFrame) -> List[DojiStarPattern]:
        """Detect doji star patterns in the data"""
        patterns = []
        min_gap = self.parameters['min_gap_percentage']
        
        for i in range(2, len(data) - self.parameters['confirmation_candles']):
            # Check for potential doji star pattern (3-candle pattern)
            candle1 = data.iloc[i-1]  # First candle
            candle2 = data.iloc[i]    # Doji candle
            candle3 = data.iloc[i+1]  # Third candle
            
            # Check if middle candle is a doji
            if candle2['body_to_range'] > self.parameters['doji_threshold']:
                continue
            
            # Morning Doji Star Pattern Detection
            if self._is_morning_doji_star(candle1, candle2, candle3, data.iloc[i-1:i+2]):
                pattern = self._create_pattern_object(
                    'morning_doji_star', i-1, i+1, candle1, candle2, candle3, data, i
                )
                if pattern.strength >= self.parameters['min_pattern_strength']:
                    patterns.append(pattern)
            
            # Evening Doji Star Pattern Detection
            elif self._is_evening_doji_star(candle1, candle2, candle3, data.iloc[i-1:i+2]):
                pattern = self._create_pattern_object(
                    'evening_doji_star', i-1, i+1, candle1, candle2, candle3, data, i
                )
                if pattern.strength >= self.parameters['min_pattern_strength']:
                    patterns.append(pattern)
        
        return patterns
    
    def _is_morning_doji_star(self, c1: pd.Series, c2: pd.Series, c3: pd.Series, 
                            context_data: pd.DataFrame) -> bool:
        """Check if pattern qualifies as morning doji star"""
        # 1. First candle should be bearish with significant body
        if c1['close'] >= c1['open'] or c1['body_to_range'] < 0.6:
            return False
        
        # 2. Second candle (doji) should gap down
        if c2['high'] >= c1['low']:  # No gap down
            return False
        
        # 3. Gap should be significant
        gap_size = (c1['low'] - c2['high']) / c1['atr']
        if gap_size < self.parameters['min_gap_percentage']:
            return False
        
        # 4. Third candle should be bullish and close above midpoint of first candle
        if c3['close'] <= c3['open']:
            return False
        
        first_candle_midpoint = (c1['high'] + c1['low']) / 2
        if c3['close'] <= first_candle_midpoint:
            return False
        
        # 5. Should occur in downtrend context
        if c1['trend_long'] != -1:
            return False
        
        return True
    
    def _is_evening_doji_star(self, c1: pd.Series, c2: pd.Series, c3: pd.Series, 
                            context_data: pd.DataFrame) -> bool:
        """Check if pattern qualifies as evening doji star"""
        # 1. First candle should be bullish with significant body
        if c1['close'] <= c1['open'] or c1['body_to_range'] < 0.6:
            return False
        
        # 2. Second candle (doji) should gap up
        if c2['low'] <= c1['high']:  # No gap up
            return False
        
        # 3. Gap should be significant
        gap_size = (c2['low'] - c1['high']) / c1['atr']
        if gap_size < self.parameters['min_gap_percentage']:
            return False
        
        # 4. Third candle should be bearish and close below midpoint of first candle
        if c3['close'] >= c3['open']:
            return False
        
        first_candle_midpoint = (c1['high'] + c1['low']) / 2
        if c3['close'] >= first_candle_midpoint:
            return False
        
        # 5. Should occur in uptrend context
        if c1['trend_long'] != 1:
            return False
        
        return True
    
    def _create_pattern_object(self, pattern_type: str, start_idx: int, end_idx: int,
                              c1: pd.Series, c2: pd.Series, c3: pd.Series,
                              data: pd.DataFrame, center_idx: int) -> DojiStarPattern:
        """Create a DojiStarPattern object with calculated metrics"""
        
        # Calculate gap size
        if pattern_type == 'morning_doji_star':
            gap_size = (c1['low'] - c2['high']) / c1['atr']
        else:  # evening_doji_star
            gap_size = (c2['low'] - c1['high']) / c1['atr']
        
        # Calculate pattern strength based on multiple factors
        strength_factors = []
        
        # 1. Gap size factor (0.3 weight)
        gap_factor = min(gap_size / (self.parameters['min_gap_percentage'] * 3), 1.0)
        strength_factors.append(gap_factor * 0.3)
        
        # 2. Doji quality factor (0.2 weight)
        doji_factor = 1.0 - min(c2['body_to_range'] / self.parameters['doji_threshold'], 1.0)
        strength_factors.append(doji_factor * 0.2)
        
        # 3. Trend context factor (0.2 weight)
        trend_factor = 0.8 if abs(c1['trend_long']) == 1 else 0.4
        strength_factors.append(trend_factor * 0.2)
        
        # 4. Volume confirmation factor (0.15 weight)
        volume_factor = min(c2['volume_ratio'], 2.0) / 2.0
        strength_factors.append(volume_factor * 0.15)
        
        # 5. Confirmation candle factor (0.15 weight)
        if pattern_type == 'morning_doji_star':
            conf_factor = min(c3['body_to_range'], 1.0)
        else:
            conf_factor = min(c3['body_to_range'], 1.0)
        strength_factors.append(conf_factor * 0.15)
        
        total_strength = sum(strength_factors)
        
        # Calculate confirmation strength (look ahead if possible)
        confirmation_strength = self._calculate_confirmation_strength(
            data, end_idx, pattern_type
        )
        
        # Volume confirmation
        volume_confirmation = (
            c2['volume_surge'] and 
            c3['volume_ratio'] > 1.0
        )
        
        # Support/resistance context
        if pattern_type == 'morning_doji_star':
            sr_level = c2['support']
            trend_context = 'downtrend_reversal'
        else:
            sr_level = c2['resistance']
            trend_context = 'uptrend_reversal'
        
        return DojiStarPattern(
            pattern_type=pattern_type,
            start_index=start_idx,
            end_index=end_idx,
            strength=total_strength,
            gap_size=gap_size,
            confirmation_strength=confirmation_strength,
            volume_confirmation=volume_confirmation,
            trend_context=trend_context,
            support_resistance_level=sr_level
        )
    
    def _calculate_confirmation_strength(self, data: pd.DataFrame, end_idx: int, 
                                       pattern_type: str) -> float:
        """Calculate confirmation strength based on subsequent price action"""
        confirmation_candles = self.parameters['confirmation_candles']
        
        if end_idx + confirmation_candles >= len(data):
            return 0.5  # Default when no confirmation data available
        
        pattern_close = data.iloc[end_idx]['close']
        confirmation_data = data.iloc[end_idx+1:end_idx+1+confirmation_candles]
        
        if pattern_type == 'morning_doji_star':
            # Look for bullish confirmation
            bullish_candles = sum(1 for _, row in confirmation_data.iterrows() 
                                if row['close'] > row['open'])
            price_movement = (confirmation_data['close'].iloc[-1] - pattern_close) / pattern_close
            
            confirmation = (bullish_candles / len(confirmation_data)) * 0.6
            if price_movement > 0:
                confirmation += min(price_movement * 10, 0.4)
        
        else:  # evening_doji_star
            # Look for bearish confirmation
            bearish_candles = sum(1 for _, row in confirmation_data.iterrows() 
                                if row['close'] < row['open'])
            price_movement = (pattern_close - confirmation_data['close'].iloc[-1]) / pattern_close
            
            confirmation = (bearish_candles / len(confirmation_data)) * 0.6
            if price_movement > 0:
                confirmation += min(price_movement * 10, 0.4)
        
        return min(confirmation, 1.0)
    
    def _validate_patterns_with_ml(self, patterns: List[DojiStarPattern], 
                                 data: pd.DataFrame) -> List[DojiStarPattern]:
        """Validate patterns using machine learning"""
        try:
            if not patterns:
                return patterns
            
            # Extract features for each pattern
            features = []
            for pattern in patterns:
                feature_vector = self._extract_pattern_features(pattern, data)
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Train model if not fitted and enough training data
            if not self.is_fitted and len(self.training_patterns) >= 50:
                self._train_ml_model()
            
            # Apply ML validation if model is fitted
            if self.is_fitted:
                features_scaled = self.scaler.transform(features_array)
                pattern_probabilities = self.pattern_classifier.predict_proba(features_scaled)
                
                # Adjust pattern strengths based on ML probabilities
                for i, (pattern, prob) in enumerate(zip(patterns, pattern_probabilities)):
                    # prob[1] is probability of positive class (valid pattern)
                    ml_confidence = prob[1] if len(prob) > 1 else 0.5
                    pattern.strength = (pattern.strength * 0.7) + (ml_confidence * 0.3)
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML validation failed: {str(e)}")
            return patterns
    
    def _extract_pattern_features(self, pattern: DojiStarPattern, data: pd.DataFrame) -> List[float]:
        """Extract features for ML model"""
        try:
            start_idx = max(0, pattern.start_index - 10)
            end_idx = min(len(data), pattern.end_index + 10)
            context_data = data.iloc[start_idx:end_idx]
            
            features = [
                pattern.gap_size,
                pattern.strength,
                pattern.confirmation_strength,
                1.0 if pattern.volume_confirmation else 0.0,
                context_data['volatility_percentile'].mean(),
                context_data['volume_ratio'].mean(),
                context_data['atr'].iloc[-1] / context_data['close'].iloc[-1],  # Normalized ATR
                1.0 if pattern.pattern_type == 'morning_doji_star' else 0.0,
                context_data['trend_long'].iloc[pattern.start_index - start_idx],
                context_data['near_support'].iloc[pattern.start_index - start_idx] if pattern.pattern_type == 'morning_doji_star' else context_data['near_resistance'].iloc[pattern.start_index - start_idx]
            ]
            
            return features
            
        except Exception:
            # Return default features if extraction fails
            return [0.5] * 10
    
    def _train_ml_model(self):
        """Train the ML model using historical patterns"""
        # This would typically use historical data with known outcomes
        # For now, implement a simplified training approach
        try:
            # Create synthetic training data based on pattern characteristics
            X_train = []
            y_train = []
            
            for pattern_data in self.training_patterns:
                X_train.append(pattern_data['features'])
                y_train.append(pattern_data['outcome'])
            
            if len(X_train) >= 20:
                X_scaled = self.scaler.fit_transform(X_train)
                self.pattern_classifier.fit(X_scaled, y_train)
                self.is_fitted = True
                logging.info("ML model trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _analyze_current_state(self, data: pd.DataFrame, patterns: List[DojiStarPattern]) -> Dict[str, Any]:
        """Analyze current market state and pattern context"""
        current_data = data.iloc[-1]
        recent_patterns = [p for p in patterns if p.end_index >= len(data) - 20]
        
        return {
            'current_trend': 'uptrend' if current_data['trend_long'] == 1 else 'downtrend',
            'volatility_level': 'high' if current_data['volatility_percentile'] > 0.7 else 'normal',
            'recent_pattern_count': len(recent_patterns),
            'dominant_pattern_type': self._get_dominant_pattern_type(recent_patterns),
            'near_key_level': current_data['near_resistance'] or current_data['near_support'],
            'volume_context': 'high' if current_data['volume_ratio'] > 1.5 else 'normal'
        }
    
    def _get_dominant_pattern_type(self, patterns: List[DojiStarPattern]) -> str:
        """Get the dominant pattern type from recent patterns"""
        if not patterns:
            return 'none'
        
        type_counts = {}
        for pattern in patterns:
            type_counts[pattern.pattern_type] = type_counts.get(pattern.pattern_type, 0) + 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'none'
    
    def _generate_pattern_statistics(self, patterns: List[DojiStarPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern statistics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-50:]  # Last 50 patterns
        
        # Basic statistics
        morning_stars = [p for p in recent_patterns if p.pattern_type == 'morning_doji_star']
        evening_stars = [p for p in recent_patterns if p.pattern_type == 'evening_doji_star']
        
        # Strength analysis
        avg_strength = sum(p.strength for p in recent_patterns) / len(recent_patterns)
        high_strength_patterns = [p for p in recent_patterns if p.strength > 0.8]
        
        # Success rate analysis (simplified)
        confirmed_patterns = [p for p in recent_patterns if p.confirmation_strength > 0.6]
        
        return {
            'total_patterns': len(recent_patterns),
            'morning_star_count': len(morning_stars),
            'evening_star_count': len(evening_stars),
            'average_strength': avg_strength,
            'high_strength_count': len(high_strength_patterns),
            'confirmation_rate': len(confirmed_patterns) / len(recent_patterns),
            'average_gap_size': sum(p.gap_size for p in recent_patterns) / len(recent_patterns),
            'volume_confirmation_rate': sum(1 for p in recent_patterns if p.volume_confirmation) / len(recent_patterns)
        }
    
    def _get_market_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get current market state information"""
        current = data.iloc[-1]
        return {
            'price': current['close'],
            'atr': current['atr'],
            'trend': 'bullish' if current['trend_long'] == 1 else 'bearish',
            'volatility_percentile': current['volatility_percentile'],
            'volume_ratio': current['volume_ratio'],
            'near_support': current['near_support'],
            'near_resistance': current['near_resistance']
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on doji star analysis"""
        current_patterns = value.get('current_patterns', [])
        current_analysis = value.get('current_analysis', {})
        
        if not current_patterns:
            return None, 0.0
        
        # Get the strongest recent pattern
        strongest_pattern = max(current_patterns, key=lambda p: p.strength)
        
        # Base confidence from pattern strength and confirmation
        base_confidence = (strongest_pattern.strength + strongest_pattern.confirmation_strength) / 2
        
        # Adjust based on volume confirmation
        if strongest_pattern.volume_confirmation:
            base_confidence *= 1.2
        
        # Adjust based on market context
        if current_analysis.get('near_key_level'):
            base_confidence *= 1.1
        
        # Generate signal based on pattern type
        if strongest_pattern.pattern_type == 'morning_doji_star':
            if base_confidence > 0.7:
                return SignalType.BUY, min(base_confidence, 0.95)
            elif base_confidence > 0.5:
                return SignalType.HOLD, base_confidence * 0.8
        
        elif strongest_pattern.pattern_type == 'evening_doji_star':
            if base_confidence > 0.7:
                return SignalType.SELL, min(base_confidence, 0.95)
            elif base_confidence > 0.5:
                return SignalType.HOLD, base_confidence * 0.8
        
        return SignalType.NEUTRAL, base_confidence * 0.6
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_fitted,
            'pattern_types': ['morning_doji_star', 'evening_doji_star'],
            'gap_analysis_enabled': True,
            'volume_confirmation_enabled': self.parameters['volume_confirmation'],
            'confirmation_candles': self.parameters['confirmation_candles']
        })
        return base_metadata