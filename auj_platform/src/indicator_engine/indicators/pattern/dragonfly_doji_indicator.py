"""
Dragonfly Doji Indicator - Advanced Reversal Pattern Recognition
===============================================================

This indicator implements sophisticated dragonfly doji detection with ML-enhanced
pattern validation, trend reversal prediction, and support level analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
import talib
from scipy import stats

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    IndicatorResult, 
    SignalType, 
    DataType, 
    DataRequirement
)
from ...core.exceptions import IndicatorCalculationException


@dataclass
class DragonflyDojiPattern:
    """Represents a detected dragonfly doji pattern"""
    timestamp: pd.Timestamp
    strength: float
    lower_shadow_ratio: float
    body_ratio: float
    upper_shadow_ratio: float
    support_level: float
    volume_confirmation: bool
    trend_context: str
    reversal_probability: float


class DragonflyDojiIndicator(StandardIndicatorInterface):
    """
    Advanced Dragonfly Doji Pattern Indicator
    
    Features:
    - Precise dragonfly doji identification with mathematical validation
    - Support level analysis and confluence detection
    - Machine learning reversal probability assessment
    - Volume surge analysis and confirmation
    - Multi-timeframe trend context evaluation
    - Statistical significance testing
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'max_body_ratio': 0.1,        # Maximum body size relative to total range
            'min_lower_shadow': 0.6,      # Minimum lower shadow ratio
            'max_upper_shadow': 0.15,     # Maximum upper shadow ratio
            'volume_surge_threshold': 1.3, # Volume increase for confirmation
            'trend_lookback': 30,
            'support_proximity_threshold': 0.02,  # 2% proximity to support
            'min_reversal_probability': 0.65,
            'statistical_validation': True,
            'ml_enhancement': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="DragonflyDojiIndicator", parameters=default_params)
        
        # Initialize ML components
        self.reversal_predictor = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = RobustScaler()
        self.is_ml_fitted = False
        
        logging.info(f"DragonflyDojiIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=50,
            lookback_periods=120
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate dragonfly doji patterns with advanced analysis"""
        try:
            if len(data) < self.parameters['trend_lookback']:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < {self.parameters['trend_lookback']}"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect dragonfly doji patterns
            detected_patterns = self._detect_dragonfly_doji_patterns(enhanced_data)
            
            # Apply statistical validation
            if self.parameters['statistical_validation']:
                validated_patterns = self._apply_statistical_validation(
                    detected_patterns, enhanced_data
                )
            else:
                validated_patterns = detected_patterns
            
            # Enhance with ML predictions
            if self.parameters['ml_enhancement'] and validated_patterns:
                ml_enhanced_patterns = self._enhance_with_ml_predictions(
                    validated_patterns, enhanced_data
                )
            else:
                ml_enhanced_patterns = validated_patterns
            
            # Analyze current market conditions
            current_analysis = self._analyze_current_conditions(enhanced_data)
            
            # Generate comprehensive statistics
            pattern_analytics = self._generate_pattern_analytics(ml_enhanced_patterns)
            
            return {
                'current_pattern': ml_enhanced_patterns[-1] if ml_enhanced_patterns else None,
                'recent_patterns': ml_enhanced_patterns[-10:],
                'pattern_analytics': pattern_analytics,
                'current_market_analysis': current_analysis,
                'support_levels': self._identify_support_levels(enhanced_data),
                'reversal_signals': self._generate_reversal_signals(ml_enhanced_patterns, enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Dragonfly doji calculation failed: {str(e)}", e
            )
    
    def _enhance_data_with_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with comprehensive technical indicators"""
        df = data.copy()
        
        # Candlestick components
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Ratios for pattern identification
        df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
        df['upper_shadow_ratio'] = np.where(df['total_range'] > 0, df['upper_shadow'] / df['total_range'], 0)
        df['lower_shadow_ratio'] = np.where(df['total_range'] > 0, df['lower_shadow'] / df['total_range'], 0)
        
        # Trend indicators
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
        
        # Trend classification
        df['short_trend'] = np.where(df['close'] > df['sma_20'], 1, -1)
        df['medium_trend'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['trend_strength'] = abs(df['close'] - df['sma_20']) / df['sma_20']
        
        # Volatility measures
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['volatility_ratio'] = df['true_range'] / df['atr']
        
        # Support/Resistance levels
        df['support_20'] = df['low'].rolling(20).min()
        df['support_50'] = df['low'].rolling(50).min()
        df['resistance_20'] = df['high'].rolling(20).max()
        
        # Distance to support levels
        df['distance_to_support_20'] = (df['close'] - df['support_20']) / df['close']
        df['distance_to_support_50'] = (df['close'] - df['support_50']) / df['close']
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_surge'] = df['volume_ratio'] > self.parameters['volume_surge_threshold']
        
        # Price position relative to daily range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Momentum indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # Statistical measures
        df['price_zscore'] = stats.zscore(df['close'].rolling(50))
        
        return df
    
    def _detect_dragonfly_doji_patterns(self, data: pd.DataFrame) -> List[DragonflyDojiPattern]:
        """Detect dragonfly doji patterns with precise criteria"""
        patterns = []
        
        for i in range(20, len(data)):  # Need lookback for context
            row = data.iloc[i]
            
            # Core dragonfly doji criteria
            if not self._meets_dragonfly_criteria(row):
                continue
            
            # Trend context analysis
            trend_context = self._analyze_trend_context(data, i)
            
            # Support level analysis
            support_analysis = self._analyze_support_confluence(data, i)
            
            # Volume confirmation
            volume_confirmation = self._check_volume_confirmation(data, i)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(
                row, trend_context, support_analysis, volume_confirmation
            )
            
            # Calculate reversal probability
            reversal_prob = self._calculate_base_reversal_probability(
                row, trend_context, support_analysis
            )
            
            if pattern_strength >= 0.6:  # Minimum threshold
                pattern = DragonflyDojiPattern(
                    timestamp=row.name,
                    strength=pattern_strength,
                    lower_shadow_ratio=row['lower_shadow_ratio'],
                    body_ratio=row['body_ratio'],
                    upper_shadow_ratio=row['upper_shadow_ratio'],
                    support_level=support_analysis['nearest_support'],
                    volume_confirmation=volume_confirmation,
                    trend_context=trend_context['description'],
                    reversal_probability=reversal_prob
                )
                patterns.append(pattern)
        
        return patterns
    
    def _meets_dragonfly_criteria(self, row: pd.Series) -> bool:
        """Check if candle meets dragonfly doji criteria"""
        # 1. Small body
        if row['body_ratio'] > self.parameters['max_body_ratio']:
            return False
        
        # 2. Long lower shadow
        if row['lower_shadow_ratio'] < self.parameters['min_lower_shadow']:
            return False
        
        # 3. Minimal or no upper shadow
        if row['upper_shadow_ratio'] > self.parameters['max_upper_shadow']:
            return False
        
        # 4. Minimum total range (not a tiny candle)
        if row['total_range'] < row['atr'] * 0.5:
            return False
        
        return True
    
    def _analyze_trend_context(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Analyze trend context around the pattern"""
        lookback = self.parameters['trend_lookback']
        start_idx = max(0, index - lookback)
        context_data = data.iloc[start_idx:index+1]
        
        current_row = data.iloc[index]
        
        # Trend direction analysis
        recent_trend = context_data['short_trend'].iloc[-10:].mean()
        overall_trend = context_data['medium_trend'].iloc[-1]
        
        # Price decline analysis (important for dragonfly doji)
        price_decline = (context_data['close'].iloc[0] - current_row['close']) / context_data['close'].iloc[0]
        
        # Trend strength
        trend_strength = current_row['trend_strength']
        
        # Determine trend description
        if recent_trend < -0.6 and overall_trend == -1:
            description = "strong_downtrend"
        elif recent_trend < -0.2:
            description = "downtrend"
        elif recent_trend > 0.6 and overall_trend == 1:
            description = "strong_uptrend"
        elif recent_trend > 0.2:
            description = "uptrend"
        else:
            description = "sideways"
        
        return {
            'description': description,
            'recent_trend': recent_trend,
            'overall_trend': overall_trend,
            'price_decline': price_decline,
            'trend_strength': trend_strength,
            'is_downtrend': recent_trend < -0.2  # Favorable for dragonfly doji
        }
    
    def _analyze_support_confluence(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Analyze support level confluence"""
        current_row = data.iloc[index]
        
        # Multiple support levels
        support_levels = [
            current_row['support_20'],
            current_row['support_50'],
        ]
        
        # Find nearest support
        current_price = current_row['close']
        distances = [abs(current_price - level) / current_price for level in support_levels]
        nearest_support = support_levels[np.argmin(distances)]
        nearest_distance = min(distances)
        
        # Check if near significant support
        near_support = nearest_distance < self.parameters['support_proximity_threshold']
        
        # Support strength (how many times tested)
        support_strength = self._calculate_support_strength(data, index, nearest_support)
        
        return {
            'nearest_support': nearest_support,
            'distance_to_support': nearest_distance,
            'near_support': near_support,
            'support_strength': support_strength,
            'confluence_score': self._calculate_confluence_score(data, index, support_levels)
        }
    
    def _calculate_support_strength(self, data: pd.DataFrame, index: int, support_level: float) -> float:
        """Calculate support level strength based on historical tests"""
        lookback_data = data.iloc[max(0, index-50):index]
        tolerance = support_level * 0.02  # 2% tolerance
        
        # Count touches of support level
        touches = 0
        for _, row in lookback_data.iterrows():
            if abs(row['low'] - support_level) <= tolerance:
                touches += 1
        
        return min(touches / 5.0, 1.0)  # Normalize to 0-1
    
    def _calculate_confluence_score(self, data: pd.DataFrame, index: int, support_levels: List[float]) -> float:
        """Calculate confluence score based on multiple support levels"""
        current_price = data.iloc[index]['close']
        tolerance = current_price * 0.01  # 1% tolerance
        
        confluence_count = 0
        for level in support_levels:
            if abs(current_price - level) <= tolerance:
                confluence_count += 1
        
        return confluence_count / len(support_levels)
    
    def _check_volume_confirmation(self, data: pd.DataFrame, index: int) -> bool:
        """Check for volume confirmation"""
        current_row = data.iloc[index]
        return current_row['volume_surge']
    
    def _calculate_pattern_strength(self, row: pd.Series, trend_context: Dict[str, Any], 
                                  support_analysis: Dict[str, Any], volume_confirmation: bool) -> float:
        """Calculate overall pattern strength"""
        strength_components = []
        
        # 1. Dragonfly quality (40% weight)
        dragonfly_quality = (
            (1 - row['body_ratio'] / self.parameters['max_body_ratio']) * 0.4 +
            (row['lower_shadow_ratio'] / self.parameters['min_lower_shadow']) * 0.4 +
            (1 - row['upper_shadow_ratio'] / self.parameters['max_upper_shadow']) * 0.2
        )
        strength_components.append(dragonfly_quality * 0.4)
        
        # 2. Trend context (25% weight)
        trend_favorability = 0.9 if trend_context['is_downtrend'] else 0.3
        if trend_context['description'] == 'strong_downtrend':
            trend_favorability = 1.0
        strength_components.append(trend_favorability * 0.25)
        
        # 3. Support confluence (20% weight)
        support_factor = (
            support_analysis['confluence_score'] * 0.5 +
            support_analysis['support_strength'] * 0.3 +
            (1.0 if support_analysis['near_support'] else 0.5) * 0.2
        )
        strength_components.append(support_factor * 0.2)
        
        # 4. Volume confirmation (10% weight)
        volume_factor = 1.0 if volume_confirmation else 0.6
        strength_components.append(volume_factor * 0.1)
        
        # 5. Market conditions (5% weight)
        rsi_oversold = 1.0 if row['rsi'] < 30 else 0.5
        strength_components.append(rsi_oversold * 0.05)
        
        return min(sum(strength_components), 1.0)
    
    def _calculate_base_reversal_probability(self, row: pd.Series, trend_context: Dict[str, Any], 
                                           support_analysis: Dict[str, Any]) -> float:
        """Calculate base reversal probability"""
        # Base probability starts at 0.5
        probability = 0.5
        
        # Adjust for trend context
        if trend_context['is_downtrend']:
            probability += 0.2
        if trend_context['description'] == 'strong_downtrend':
            probability += 0.1
        
        # Adjust for support proximity
        if support_analysis['near_support']:
            probability += 0.15
        
        # Adjust for support strength
        probability += support_analysis['support_strength'] * 0.1
        
        # Adjust for RSI oversold condition
        if row['rsi'] < 30:
            probability += 0.1
        
        return min(probability, 0.95)
    
    def _apply_statistical_validation(self, patterns: List[DragonflyDojiPattern], 
                                   data: pd.DataFrame) -> List[DragonflyDojiPattern]:
        """Apply statistical validation to patterns"""
        if not patterns:
            return patterns
        
        validated_patterns = []
        
        for pattern in patterns:
            # Get pattern index in data
            pattern_idx = data.index.get_loc(pattern.timestamp)
            
            # Statistical significance test
            if self._is_statistically_significant(data, pattern_idx, pattern):
                validated_patterns.append(pattern)
        
        return validated_patterns
    
    def _is_statistically_significant(self, data: pd.DataFrame, index: int, 
                                    pattern: DragonflyDojiPattern) -> bool:
        """Test statistical significance of the pattern"""
        try:
            # Get surrounding data
            lookback = 20
            start_idx = max(0, index - lookback)
            end_idx = min(len(data), index + lookback + 1)
            context_data = data.iloc[start_idx:end_idx]
            
            # Test if lower shadow is significantly larger than usual
            lower_shadows = context_data['lower_shadow_ratio']
            pattern_lower_shadow = pattern.lower_shadow_ratio
            
            # Perform t-test
            if len(lower_shadows) >= 10:
                t_stat, p_value = stats.ttest_1samp(
                    lower_shadows.drop(lower_shadows.index[index - start_idx]), 
                    pattern_lower_shadow
                )
                return p_value < 0.05 and t_stat < 0  # Pattern shadow should be larger
            
            return True  # Default to true if insufficient data
            
        except Exception:
            return True  # Default to true on error
    
    def _enhance_with_ml_predictions(self, patterns: List[DragonflyDojiPattern], 
                                   data: pd.DataFrame) -> List[DragonflyDojiPattern]:
        """Enhance patterns with ML-based reversal predictions"""
        if not patterns or not self.parameters['ml_enhancement']:
            return patterns
        
        try:
            # Extract features for ML model
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_ml_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 5:  # Not enough data for meaningful ML
                return patterns
            
            # Train model if needed (simplified approach)
            if not self.is_ml_fitted:
                self._train_ml_model(patterns, features)
            
            # Apply ML predictions if model is fitted
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                reversal_probabilities = self.reversal_predictor.predict_proba(features_scaled)
                
                # Update reversal probabilities
                for i, pattern in enumerate(patterns):
                    if len(reversal_probabilities[i]) > 1:
                        ml_probability = reversal_probabilities[i][1]  # Probability of reversal
                        # Combine with base probability
                        pattern.reversal_probability = (
                            pattern.reversal_probability * 0.6 + ml_probability * 0.4
                        )
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML enhancement failed: {str(e)}")
            return patterns
    
    def _extract_ml_features(self, data: pd.DataFrame, index: int, 
                           pattern: DragonflyDojiPattern) -> List[float]:
        """Extract features for ML model"""
        try:
            row = data.iloc[index]
            
            features = [
                pattern.strength,
                pattern.lower_shadow_ratio,
                pattern.body_ratio,
                pattern.upper_shadow_ratio,
                row['rsi'],
                row['trend_strength'],
                row['distance_to_support_20'],
                row['volume_ratio'],
                row['volatility_ratio'],
                row['price_position'],
                1.0 if pattern.volume_confirmation else 0.0,
                row['short_trend'],
                row['medium_trend']
            ]
            
            return features
            
        except Exception:
            return [0.5] * 13  # Default features
    
    def _train_ml_model(self, patterns: List[DragonflyDojiPattern], features: List[List[float]]):
        """Train ML model (simplified implementation)"""
        try:
            # Create synthetic labels based on pattern characteristics
            labels = []
            for pattern in patterns:
                # Higher probability patterns get positive label
                label = 1 if pattern.reversal_probability > 0.7 else 0
                labels.append(label)
            
            if len(set(labels)) > 1 and len(features) >= 10:
                features_scaled = self.scaler.fit_transform(features)
                self.reversal_predictor.fit(features_scaled, labels)
                self.is_ml_fitted = True
                logging.info("ML reversal predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _analyze_current_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions"""
        current = data.iloc[-1]
        
        return {
            'current_trend': 'downtrend' if current['short_trend'] == -1 else 'uptrend',
            'trend_strength': current['trend_strength'],
            'rsi_level': current['rsi'],
            'is_oversold': current['rsi'] < 30,
            'distance_to_support': current['distance_to_support_20'],
            'volume_context': 'high' if current['volume_ratio'] > 1.5 else 'normal',
            'volatility_level': 'high' if current['volatility_ratio'] > 1.5 else 'normal',
            'favorable_for_dragonfly': (
                current['short_trend'] == -1 and 
                current['rsi'] < 40 and 
                current['distance_to_support_20'] < 0.05
            )
        }
    
    def _identify_support_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identify key support levels"""
        current = data.iloc[-1]
        
        return {
            'immediate_support': current['support_20'],
            'strong_support': current['support_50'],
            'current_price': current['close'],
            'support_20_distance': current['distance_to_support_20'],
            'support_50_distance': current['distance_to_support_50']
        }
    
    def _generate_reversal_signals(self, patterns: List[DragonflyDojiPattern], 
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Generate reversal signals based on patterns"""
        if not patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        # Get most recent high-probability patterns
        recent_patterns = [p for p in patterns[-5:] if p.reversal_probability > self.parameters['min_reversal_probability']]
        
        if not recent_patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        # Calculate aggregate signal strength
        avg_strength = sum(p.strength for p in recent_patterns) / len(recent_patterns)
        avg_reversal_prob = sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns)
        
        return {
            'signal_strength': avg_strength,
            'reversal_probability': avg_reversal_prob,
            'pattern_count': len(recent_patterns),
            'strongest_pattern_strength': max(p.strength for p in recent_patterns),
            'most_recent_pattern': recent_patterns[-1].timestamp
        }
    
    def _generate_pattern_analytics(self, patterns: List[DragonflyDojiPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-30:]  # Last 30 patterns
        
        return {
            'total_patterns': len(recent_patterns),
            'average_strength': sum(p.strength for p in recent_patterns) / len(recent_patterns),
            'average_reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'volume_confirmation_rate': sum(1 for p in recent_patterns if p.volume_confirmation) / len(recent_patterns),
            'high_probability_patterns': len([p for p in recent_patterns if p.reversal_probability > 0.8]),
            'patterns_near_support': len([p for p in recent_patterns if abs(p.support_level - recent_patterns[-1].support_level) / p.support_level < 0.02])
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on dragonfly doji analysis"""
        current_pattern = value.get('current_pattern')
        reversal_signals = value.get('reversal_signals', {})
        current_analysis = value.get('current_market_analysis', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Only generate signals for high-quality patterns
        if current_pattern.strength < 0.7:
            return None, 0.0
        
        # Check if conditions are favorable for bullish reversal
        reversal_probability = reversal_signals.get('reversal_probability', 0.0)
        
        if (reversal_probability > self.parameters['min_reversal_probability'] and
            current_analysis.get('favorable_for_dragonfly', False)):
            
            # Calculate signal confidence
            confidence = (
                current_pattern.strength * 0.4 +
                reversal_probability * 0.4 +
                (1.0 if current_pattern.volume_confirmation else 0.6) * 0.2
            )
            
            return SignalType.BUY, min(confidence, 0.95)
        
        # Weak bullish signal for moderate patterns
        elif current_pattern.strength > 0.6 and reversal_probability > 0.6:
            confidence = current_pattern.strength * 0.7
            return SignalType.HOLD, confidence
        
        return SignalType.NEUTRAL, current_pattern.strength * 0.5
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'dragonfly_doji',
            'statistical_validation': self.parameters['statistical_validation'],
            'support_analysis_enabled': True,
            'reversal_prediction_enabled': True
        })
        return base_metadata