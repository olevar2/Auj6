"""
Gravestone Doji Indicator - Advanced Bearish Reversal Pattern Recognition
========================================================================

This indicator implements sophisticated gravestone doji detection with resistance
level analysis, ML-enhanced pattern validation, and bearish reversal prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import logging
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
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
class GravestoneDojiPattern:
    """Represents a detected gravestone doji pattern"""
    timestamp: pd.Timestamp
    strength: float
    upper_shadow_ratio: float
    body_ratio: float
    lower_shadow_ratio: float
    resistance_level: float
    volume_confirmation: bool
    trend_context: str
    bearish_reversal_probability: float
    market_structure_score: float


class GravestoneDojiIndicator(StandardIndicatorInterface):
    """
    Advanced Gravestone Doji Pattern Indicator
    
    Features:
    - Precise gravestone doji identification with mathematical validation
    - Resistance level analysis and confluence detection
    - Machine learning bearish reversal probability assessment
    - Market structure analysis for optimal entry timing
    - Volume distribution analysis and confirmation
    - Multi-timeframe trend exhaustion detection
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'max_body_ratio': 0.1,         # Maximum body size relative to total range
            'min_upper_shadow': 0.6,       # Minimum upper shadow ratio
            'max_lower_shadow': 0.15,      # Maximum lower shadow ratio
            'volume_surge_threshold': 1.4,  # Volume increase for confirmation
            'trend_lookback': 25,
            'resistance_proximity_threshold': 0.015,  # 1.5% proximity to resistance
            'min_bearish_probability': 0.7,
            'market_structure_analysis': True,
            'ml_enhancement': True,
            'trend_exhaustion_detection': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="GravestoneDojiIndicator", parameters=default_params)
        
        # Initialize ML components
        self.bearish_predictor = AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.8,
            random_state=42
        )
        self.scaler = MinMaxScaler()
        self.is_ml_fitted = False
        
        # Pattern validation history
        self.pattern_history = []
        
        logging.info(f"GravestoneDojiIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=60,
            lookback_periods=150
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate gravestone doji patterns with advanced analysis"""
        try:
            if len(data) < self.parameters['trend_lookback']:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < {self.parameters['trend_lookback']}"
                )
            
            # Enhance data with comprehensive indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect gravestone doji patterns
            detected_patterns = self._detect_gravestone_patterns(enhanced_data)
            
            # Apply market structure analysis
            if self.parameters['market_structure_analysis']:
                structure_enhanced_patterns = self._analyze_market_structure(
                    detected_patterns, enhanced_data
                )
            else:
                structure_enhanced_patterns = detected_patterns
            
            # Apply ML enhancement
            if self.parameters['ml_enhancement'] and structure_enhanced_patterns:
                ml_enhanced_patterns = self._enhance_with_ml_predictions(
                    structure_enhanced_patterns, enhanced_data
                )
            else:
                ml_enhanced_patterns = structure_enhanced_patterns
            
            # Detect trend exhaustion if enabled
            if self.parameters['trend_exhaustion_detection']:
                exhaustion_analysis = self._detect_trend_exhaustion(enhanced_data)
            else:
                exhaustion_analysis = {}
            
            # Generate comprehensive analysis
            current_analysis = self._analyze_current_market_state(enhanced_data)
            pattern_analytics = self._generate_pattern_analytics(ml_enhanced_patterns)
            
            return {
                'current_pattern': ml_enhanced_patterns[-1] if ml_enhanced_patterns else None,
                'recent_patterns': ml_enhanced_patterns[-8:],
                'pattern_analytics': pattern_analytics,
                'current_market_analysis': current_analysis,
                'resistance_levels': self._identify_resistance_levels(enhanced_data),
                'trend_exhaustion': exhaustion_analysis,
                'bearish_signals': self._generate_bearish_signals(ml_enhanced_patterns, enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Gravestone doji calculation failed: {str(e)}", e
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
        
        # Trend and momentum indicators
        df['sma_12'] = talib.SMA(df['close'], timeperiod=12)
        df['sma_26'] = talib.SMA(df['close'], timeperiod=26)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        
        # MACD for trend momentum
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['macd_divergence'] = df['macd'] - df['macd_signal']
        
        # Trend classification
        df['short_trend'] = np.where(df['close'] > df['sma_26'], 1, -1)
        df['medium_trend'] = np.where(df['sma_26'] > df['sma_50'], 1, -1)
        df['trend_momentum'] = (df['close'] - df['sma_26']) / df['sma_26']
        
        # Resistance levels (multiple timeframes)
        df['resistance_10'] = df['high'].rolling(10).max()
        df['resistance_20'] = df['high'].rolling(20).max()
        df['resistance_50'] = df['high'].rolling(50).max()
        
        # Distance to resistance levels
        df['distance_to_resistance_10'] = (df['resistance_10'] - df['close']) / df['close']
        df['distance_to_resistance_20'] = (df['resistance_20'] - df['close']) / df['close']
        
        # Volatility and momentum measures
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['volatility_ratio'] = df['true_range'] / df['atr']
        
        # RSI and momentum
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_overbought'] = df['rsi'] > 70
        df['momentum'] = talib.MOM(df['close'], timeperiod=10)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_surge'] = df['volume_ratio'] > self.parameters['volume_surge_threshold']
        
        # Price position and distribution
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['high_close_ratio'] = (df['high'] - df['close']) / df['total_range']
        
        # Bollinger Bands for overbought detection
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic for momentum exhaustion
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        return df
    
    def _detect_gravestone_patterns(self, data: pd.DataFrame) -> List[GravestoneDojiPattern]:
        """Detect gravestone doji patterns with precise criteria"""
        patterns = []
        
        for i in range(30, len(data)):  # Need substantial lookback for context
            row = data.iloc[i]
            
            # Core gravestone doji criteria
            if not self._meets_gravestone_criteria(row):
                continue
            
            # Trend context analysis (should be in uptrend for bearish reversal)
            trend_context = self._analyze_trend_context(data, i)
            
            # Resistance level analysis
            resistance_analysis = self._analyze_resistance_confluence(data, i)
            
            # Volume confirmation
            volume_confirmation = self._check_volume_confirmation(data, i)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(
                row, trend_context, resistance_analysis, volume_confirmation
            )
            
            # Calculate initial bearish reversal probability
            bearish_prob = self._calculate_base_bearish_probability(
                row, trend_context, resistance_analysis
            )
            
            if pattern_strength >= 0.65:  # Higher threshold for gravestone
                pattern = GravestoneDojiPattern(
                    timestamp=row.name,
                    strength=pattern_strength,
                    upper_shadow_ratio=row['upper_shadow_ratio'],
                    body_ratio=row['body_ratio'],
                    lower_shadow_ratio=row['lower_shadow_ratio'],
                    resistance_level=resistance_analysis['nearest_resistance'],
                    volume_confirmation=volume_confirmation,
                    trend_context=trend_context['description'],
                    bearish_reversal_probability=bearish_prob,
                    market_structure_score=0.0  # Will be calculated later
                )
                patterns.append(pattern)
        
        return patterns
    
    def _meets_gravestone_criteria(self, row: pd.Series) -> bool:
        """Check if candle meets gravestone doji criteria"""
        # 1. Small body
        if row['body_ratio'] > self.parameters['max_body_ratio']:
            return False
        
        # 2. Long upper shadow
        if row['upper_shadow_ratio'] < self.parameters['min_upper_shadow']:
            return False
        
        # 3. Minimal or no lower shadow
        if row['lower_shadow_ratio'] > self.parameters['max_lower_shadow']:
            return False
        
        # 4. Minimum total range (significant candle)
        if row['total_range'] < row['atr'] * 0.6:
            return False
        
        # 5. Close should be near the low of the candle
        close_position = (row['close'] - row['low']) / row['total_range']
        if close_position > 0.25:  # Close should be in bottom 25%
            return False
        
        return True
    
    def _analyze_trend_context(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Analyze trend context for gravestone doji (should be in uptrend)"""
        lookback = self.parameters['trend_lookback']
        start_idx = max(0, index - lookback)
        context_data = data.iloc[start_idx:index+1]
        
        current_row = data.iloc[index]
        
        # Trend analysis
        recent_trend = context_data['short_trend'].iloc[-10:].mean()
        overall_trend = context_data['medium_trend'].iloc[-1]
        
        # Price appreciation analysis (important for gravestone context)
        price_appreciation = (current_row['close'] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
        
        # Momentum analysis
        trend_momentum = current_row['trend_momentum']
        macd_momentum = current_row['macd_divergence']
        
        # Trend exhaustion indicators
        rsi_overbought = current_row['rsi'] > 70
        stoch_overbought = current_row['stoch_k'] > 80
        bb_stretched = current_row['bb_position'] > 0.85
        
        # Determine trend description
        if recent_trend > 0.6 and overall_trend == 1:
            description = "strong_uptrend"
        elif recent_trend > 0.2:
            description = "uptrend"
        elif recent_trend < -0.6 and overall_trend == -1:
            description = "strong_downtrend"
        elif recent_trend < -0.2:
            description = "downtrend"
        else:
            description = "sideways"
        
        return {
            'description': description,
            'recent_trend': recent_trend,
            'overall_trend': overall_trend,
            'price_appreciation': price_appreciation,
            'trend_momentum': trend_momentum,
            'is_uptrend': recent_trend > 0.2,  # Favorable for gravestone doji
            'momentum_exhaustion': rsi_overbought or stoch_overbought or bb_stretched,
            'macd_momentum': macd_momentum
        }
    
    def _analyze_resistance_confluence(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Analyze resistance level confluence"""
        current_row = data.iloc[index]
        
        # Multiple resistance levels
        resistance_levels = [
            current_row['resistance_10'],
            current_row['resistance_20'],
            current_row['resistance_50']
        ]
        
        # Find nearest resistance
        current_high = current_row['high']
        distances = [abs(current_high - level) / current_high for level in resistance_levels]
        nearest_resistance = resistance_levels[np.argmin(distances)]
        nearest_distance = min(distances)
        
        # Check if near significant resistance
        near_resistance = nearest_distance < self.parameters['resistance_proximity_threshold']
        
        # Resistance strength calculation
        resistance_strength = self._calculate_resistance_strength(data, index, nearest_resistance)
        
        # Confluence score
        confluence_score = self._calculate_resistance_confluence_score(data, index, resistance_levels)
        
        return {
            'nearest_resistance': nearest_resistance,
            'distance_to_resistance': nearest_distance,
            'near_resistance': near_resistance,
            'resistance_strength': resistance_strength,
            'confluence_score': confluence_score,
            'rejection_quality': self._assess_rejection_quality(current_row, nearest_resistance)
        }
    
    def _calculate_resistance_strength(self, data: pd.DataFrame, index: int, resistance_level: float) -> float:
        """Calculate resistance level strength based on historical tests"""
        lookback_data = data.iloc[max(0, index-60):index]
        tolerance = resistance_level * 0.015  # 1.5% tolerance
        
        # Count rejections at resistance level
        rejections = 0
        for _, row in lookback_data.iterrows():
            if (abs(row['high'] - resistance_level) <= tolerance and 
                row['close'] < row['high'] - tolerance):
                rejections += 1
        
        return min(rejections / 4.0, 1.0)  # Normalize to 0-1
    
    def _calculate_resistance_confluence_score(self, data: pd.DataFrame, index: int, 
                                             resistance_levels: List[float]) -> float:
        """Calculate confluence score based on multiple resistance levels"""
        current_high = data.iloc[index]['high']
        tolerance = current_high * 0.008  # 0.8% tolerance
        
        confluence_count = 0
        for level in resistance_levels:
            if abs(current_high - level) <= tolerance:
                confluence_count += 1
        
        return confluence_count / len(resistance_levels)
    
    def _assess_rejection_quality(self, row: pd.Series, resistance_level: float) -> float:
        """Assess the quality of rejection at resistance level"""
        # How close did the high get to resistance
        proximity_score = 1.0 - abs(row['high'] - resistance_level) / resistance_level
        
        # How far did it fall from the high
        rejection_depth = (row['high'] - row['close']) / row['high']
        
        # Combine scores
        rejection_quality = (proximity_score * 0.6) + (rejection_depth * 0.4)
        
        return min(rejection_quality, 1.0)
    
    def _check_volume_confirmation(self, data: pd.DataFrame, index: int) -> bool:
        """Check for volume confirmation with distribution analysis"""
        current_row = data.iloc[index]
        
        # Basic volume surge check
        basic_surge = current_row['volume_surge']
        
        # Check if volume is higher on rejection (bearish confirmation)
        if index > 0:
            prev_row = data.iloc[index-1]
            volume_on_rejection = current_row['volume'] > prev_row['volume'] * 1.2
        else:
            volume_on_rejection = True
        
        return basic_surge and volume_on_rejection
    
    def _calculate_pattern_strength(self, row: pd.Series, trend_context: Dict[str, Any], 
                                  resistance_analysis: Dict[str, Any], volume_confirmation: bool) -> float:
        """Calculate overall pattern strength"""
        strength_components = []
        
        # 1. Gravestone quality (35% weight)
        gravestone_quality = (
            (1 - row['body_ratio'] / self.parameters['max_body_ratio']) * 0.4 +
            (row['upper_shadow_ratio'] / self.parameters['min_upper_shadow']) * 0.45 +
            (1 - row['lower_shadow_ratio'] / self.parameters['max_lower_shadow']) * 0.15
        )
        strength_components.append(gravestone_quality * 0.35)
        
        # 2. Trend context (30% weight) - uptrend favors gravestone
        trend_favorability = 0.9 if trend_context['is_uptrend'] else 0.2
        if trend_context['description'] == 'strong_uptrend':
            trend_favorability = 1.0
        
        # Bonus for momentum exhaustion
        if trend_context['momentum_exhaustion']:
            trend_favorability = min(trend_favorability * 1.2, 1.0)
        
        strength_components.append(trend_favorability * 0.3)
        
        # 3. Resistance confluence (20% weight)
        resistance_factor = (
            resistance_analysis['confluence_score'] * 0.4 +
            resistance_analysis['resistance_strength'] * 0.3 +
            resistance_analysis['rejection_quality'] * 0.3
        )
        if resistance_analysis['near_resistance']:
            resistance_factor *= 1.2
        strength_components.append(min(resistance_factor, 1.0) * 0.2)
        
        # 4. Volume confirmation (10% weight)
        volume_factor = 1.0 if volume_confirmation else 0.5
        strength_components.append(volume_factor * 0.1)
        
        # 5. Technical indicators (5% weight)
        technical_factor = 0.5
        if row['rsi'] > 70:
            technical_factor += 0.3
        if row['stoch_k'] > 80:
            technical_factor += 0.2
        strength_components.append(min(technical_factor, 1.0) * 0.05)
        
        return min(sum(strength_components), 1.0)
    
    def _calculate_base_bearish_probability(self, row: pd.Series, trend_context: Dict[str, Any], 
                                          resistance_analysis: Dict[str, Any]) -> float:
        """Calculate base bearish reversal probability"""
        probability = 0.5
        
        # Adjust for trend context
        if trend_context['is_uptrend']:
            probability += 0.25
        if trend_context['description'] == 'strong_uptrend':
            probability += 0.1
        
        # Adjust for momentum exhaustion
        if trend_context['momentum_exhaustion']:
            probability += 0.15
        
        # Adjust for resistance proximity and strength
        if resistance_analysis['near_resistance']:
            probability += 0.1
        probability += resistance_analysis['resistance_strength'] * 0.1
        
        # Adjust for overbought conditions
        if row['rsi'] > 75:
            probability += 0.1
        if row['stoch_k'] > 85:
            probability += 0.05
        
        return min(probability, 0.95)
    
    def _analyze_market_structure(self, patterns: List[GravestoneDojiPattern], 
                                data: pd.DataFrame) -> List[GravestoneDojiPattern]:
        """Analyze market structure for each pattern"""
        if not self.parameters['market_structure_analysis']:
            return patterns
        
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            structure_score = self._calculate_market_structure_score(data, pattern_idx)
            pattern.market_structure_score = structure_score
            
            # Adjust strength based on market structure
            pattern.strength = (pattern.strength * 0.8) + (structure_score * 0.2)
        
        return patterns
    
    def _calculate_market_structure_score(self, data: pd.DataFrame, index: int) -> float:
        """Calculate market structure score"""
        try:
            lookback_data = data.iloc[max(0, index-20):index+1]
            
            # Higher highs analysis
            recent_highs = lookback_data['high'].rolling(5).max()
            higher_highs = (recent_highs.diff() > 0).sum()
            
            # Volume pattern analysis
            volume_trend = lookback_data['volume_ratio'].rolling(5).mean().iloc[-1]
            
            # Price momentum deterioration
            momentum_deterioration = (
                lookback_data['momentum'].iloc[-5:].mean() < 
                lookback_data['momentum'].iloc[-10:-5].mean()
            )
            
            # Combine factors
            structure_score = 0.5
            structure_score += (higher_highs / 15) * 0.3  # Normalize higher highs
            structure_score += min(volume_trend, 2.0) / 2.0 * 0.2  # Volume factor
            structure_score += 0.3 if momentum_deterioration else 0.1
            
            return min(structure_score, 1.0)
            
        except Exception:
            return 0.5  # Default score
    
    def _enhance_with_ml_predictions(self, patterns: List[GravestoneDojiPattern], 
                                   data: pd.DataFrame) -> List[GravestoneDojiPattern]:
        """Enhance patterns with ML-based bearish reversal predictions"""
        if not patterns or not self.parameters['ml_enhancement']:
            return patterns
        
        try:
            # Extract features for ML model
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_ml_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 5:
                return patterns
            
            # Train model if needed
            if not self.is_ml_fitted:
                self._train_ml_model(patterns, features)
            
            # Apply ML predictions if model is fitted
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                bearish_probabilities = self.bearish_predictor.predict_proba(features_scaled)
                
                # Update bearish reversal probabilities
                for i, pattern in enumerate(patterns):
                    if len(bearish_probabilities[i]) > 1:
                        ml_probability = bearish_probabilities[i][1]  # Probability of bearish reversal
                        # Combine with base probability
                        pattern.bearish_reversal_probability = (
                            pattern.bearish_reversal_probability * 0.7 + ml_probability * 0.3
                        )
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML enhancement failed: {str(e)}")
            return patterns
    
    def _extract_ml_features(self, data: pd.DataFrame, index: int, 
                           pattern: GravestoneDojiPattern) -> List[float]:
        """Extract features for ML model"""
        try:
            row = data.iloc[index]
            prev_rows = data.iloc[max(0, index-5):index]
            
            features = [
                pattern.strength,
                pattern.upper_shadow_ratio,
                pattern.body_ratio,
                pattern.lower_shadow_ratio,
                pattern.market_structure_score,
                row['rsi'],
                row['stoch_k'],
                row['bb_position'],
                row['distance_to_resistance_20'],
                row['volume_ratio'],
                row['volatility_ratio'],
                row['trend_momentum'],
                row['macd_divergence'],
                1.0 if pattern.volume_confirmation else 0.0,
                row['short_trend'],
                prev_rows['close'].pct_change().mean() if len(prev_rows) > 1 else 0.0  # Recent volatility
            ]
            
            return features
            
        except Exception:
            return [0.5] * 16  # Default features
    
    def _train_ml_model(self, patterns: List[GravestoneDojiPattern], features: List[List[float]]):
        """Train ML model for bearish reversal prediction"""
        try:
            # Create labels based on pattern characteristics
            labels = []
            for pattern in patterns:
                # Higher probability patterns get positive label
                label = 1 if pattern.bearish_reversal_probability > 0.75 else 0
                labels.append(label)
            
            if len(set(labels)) > 1 and len(features) >= 15:
                features_scaled = self.scaler.fit_transform(features)
                self.bearish_predictor.fit(features_scaled, labels)
                self.is_ml_fitted = True
                logging.info("ML bearish reversal predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _detect_trend_exhaustion(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect trend exhaustion signals"""
        current = data.iloc[-1]
        recent_data = data.iloc[-10:]
        
        # RSI divergence
        rsi_divergence = (
            recent_data['high'].iloc[-1] > recent_data['high'].iloc[-5] and
            recent_data['rsi'].iloc[-1] < recent_data['rsi'].iloc[-5]
        )
        
        # MACD momentum weakening
        macd_weakening = recent_data['macd_hist'].iloc[-3:].mean() < recent_data['macd_hist'].iloc[-6:-3].mean()
        
        # Volume divergence
        volume_divergence = (
            recent_data['high'].iloc[-1] > recent_data['high'].iloc[-5] and
            recent_data['volume'].iloc[-1] < recent_data['volume'].iloc[-5]
        )
        
        exhaustion_score = sum([rsi_divergence, macd_weakening, volume_divergence]) / 3.0
        
        return {
            'exhaustion_score': exhaustion_score,
            'rsi_divergence': rsi_divergence,
            'macd_weakening': macd_weakening,
            'volume_divergence': volume_divergence,
            'trend_exhausted': exhaustion_score > 0.6
        }
    
    def _analyze_current_market_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market state for gravestone context"""
        current = data.iloc[-1]
        
        return {
            'current_trend': 'uptrend' if current['short_trend'] == 1 else 'downtrend',
            'trend_momentum': current['trend_momentum'],
            'rsi_level': current['rsi'],
            'is_overbought': current['rsi'] > 70,
            'stoch_overbought': current['stoch_k'] > 80,
            'distance_to_resistance': current['distance_to_resistance_20'],
            'near_resistance': current['distance_to_resistance_20'] < 0.02,
            'volume_context': 'high' if current['volume_ratio'] > 1.5 else 'normal',
            'bb_position': current['bb_position'],
            'favorable_for_gravestone': (
                current['short_trend'] == 1 and 
                current['rsi'] > 65 and 
                current['distance_to_resistance_20'] < 0.03
            )
        }
    
    def _identify_resistance_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identify key resistance levels"""
        current = data.iloc[-1]
        
        return {
            'immediate_resistance': current['resistance_10'],
            'strong_resistance': current['resistance_20'],
            'major_resistance': current['resistance_50'],
            'current_price': current['close'],
            'current_high': current['high'],
            'resistance_10_distance': current['distance_to_resistance_10'],
            'resistance_20_distance': current['distance_to_resistance_20']
        }
    
    def _generate_bearish_signals(self, patterns: List[GravestoneDojiPattern], 
                                data: pd.DataFrame) -> Dict[str, Any]:
        """Generate bearish reversal signals based on patterns"""
        if not patterns:
            return {'signal_strength': 0.0, 'bearish_probability': 0.0}
        
        # Get recent high-probability patterns
        recent_patterns = [p for p in patterns[-5:] if p.bearish_reversal_probability > self.parameters['min_bearish_probability']]
        
        if not recent_patterns:
            return {'signal_strength': 0.0, 'bearish_probability': 0.0}
        
        # Calculate aggregate signal strength
        avg_strength = sum(p.strength for p in recent_patterns) / len(recent_patterns)
        avg_bearish_prob = sum(p.bearish_reversal_probability for p in recent_patterns) / len(recent_patterns)
        avg_structure_score = sum(p.market_structure_score for p in recent_patterns) / len(recent_patterns)
        
        return {
            'signal_strength': avg_strength,
            'bearish_probability': avg_bearish_prob,
            'market_structure_score': avg_structure_score,
            'pattern_count': len(recent_patterns),
            'strongest_pattern_strength': max(p.strength for p in recent_patterns),
            'highest_bearish_probability': max(p.bearish_reversal_probability for p in recent_patterns),
            'most_recent_pattern': recent_patterns[-1].timestamp
        }
    
    def _generate_pattern_analytics(self, patterns: List[GravestoneDojiPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-25:]  # Last 25 patterns
        
        return {
            'total_patterns': len(recent_patterns),
            'average_strength': sum(p.strength for p in recent_patterns) / len(recent_patterns),
            'average_bearish_probability': sum(p.bearish_reversal_probability for p in recent_patterns) / len(recent_patterns),
            'average_structure_score': sum(p.market_structure_score for p in recent_patterns) / len(recent_patterns),
            'volume_confirmation_rate': sum(1 for p in recent_patterns if p.volume_confirmation) / len(recent_patterns),
            'high_probability_patterns': len([p for p in recent_patterns if p.bearish_reversal_probability > 0.8]),
            'patterns_at_resistance': len([p for p in recent_patterns if abs(p.resistance_level - recent_patterns[-1].resistance_level) / p.resistance_level < 0.02])
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on gravestone doji analysis"""
        current_pattern = value.get('current_pattern')
        bearish_signals = value.get('bearish_signals', {})
        current_analysis = value.get('current_market_analysis', {})
        trend_exhaustion = value.get('trend_exhaustion', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Only generate signals for high-quality patterns
        if current_pattern.strength < 0.7:
            return None, 0.0
        
        # Check if conditions are favorable for bearish reversal
        bearish_probability = bearish_signals.get('bearish_probability', 0.0)
        
        if (bearish_probability > self.parameters['min_bearish_probability'] and
            current_analysis.get('favorable_for_gravestone', False)):
            
            # Calculate signal confidence
            confidence = (
                current_pattern.strength * 0.35 +
                bearish_probability * 0.35 +
                current_pattern.market_structure_score * 0.15 +
                (1.0 if current_pattern.volume_confirmation else 0.6) * 0.15
            )
            
            # Bonus for trend exhaustion
            if trend_exhaustion.get('trend_exhausted', False):
                confidence = min(confidence * 1.15, 0.95)
            
            return SignalType.SELL, min(confidence, 0.95)
        
        # Moderate bearish signal for decent patterns
        elif current_pattern.strength > 0.65 and bearish_probability > 0.65:
            confidence = current_pattern.strength * 0.7
            return SignalType.HOLD, confidence
        
        return SignalType.NEUTRAL, current_pattern.strength * 0.5
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'gravestone_doji',
            'market_structure_analysis': self.parameters['market_structure_analysis'],
            'trend_exhaustion_detection': self.parameters['trend_exhaustion_detection'],
            'resistance_analysis_enabled': True,
            'bearish_reversal_prediction_enabled': True
        })
        return base_metadata