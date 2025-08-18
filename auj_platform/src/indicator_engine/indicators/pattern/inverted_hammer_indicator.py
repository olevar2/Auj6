"""
Inverted Hammer Indicator - Advanced Single-Candle Bullish Reversal Pattern Detection
===================================================================================

This indicator implements sophisticated inverted hammer pattern detection with advanced
upper shadow analysis, trend context validation, and ML-enhanced reversal prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import logging
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import talib
from scipy import stats
from scipy.stats import linregress, zscore

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    IndicatorResult, 
    SignalType, 
    DataType, 
    DataRequirement
)
from ...core.exceptions import IndicatorCalculationException


@dataclass
class InvertedHammerPattern:
    """Represents a detected inverted hammer pattern"""
    timestamp: pd.Timestamp
    pattern_strength: float
    body_to_shadow_ratio: float
    upper_shadow_length: float
    lower_shadow_ratio: float
    body_position: float
    trend_context_score: float
    volume_confirmation: float
    support_strength: float
    reversal_probability: float
    accumulation_signal: float
    pattern_purity: float


class InvertedHammerIndicator(StandardIndicatorInterface):
    """
    Advanced Inverted Hammer Pattern Indicator
    
    Features:
    - Precise inverted hammer identification with upper shadow analysis
    - Advanced trend context validation for downtrend requirement
    - Volume-based confirmation and accumulation signal detection
    - ML-enhanced reversal probability prediction
    - Support level validation and strength assessment
    - Pattern purity scoring and reliability metrics
    - Statistical significance testing for pattern quality
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'min_upper_shadow_ratio': 0.6,     # Minimum upper shadow as % of total range
            'max_body_ratio': 0.3,             # Maximum body as % of total range
            'max_lower_shadow_ratio': 0.1,     # Maximum lower shadow as % of total range
            'min_trend_strength': 0.6,         # Minimum downtrend strength required
            'volume_surge_threshold': 1.2,     # Volume surge multiplier
            'trend_lookback': 15,              # Periods for trend analysis
            'support_analysis': True,
            'volume_analysis': True,
            'ml_reversal_prediction': True,
            'accumulation_analysis': True,
            'pattern_purity_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="InvertedHammerIndicator", parameters=default_params)
        
        # Initialize ML components
        self.reversal_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=120, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=80, random_state=42)),
            ('ada', AdaBoostRegressor(n_estimators=100, random_state=42))
        ])
        self.scaler = RobustScaler()
        self.is_ml_fitted = False
        
        logging.info(f"InvertedHammerIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=40,
            lookback_periods=80
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate inverted hammer patterns with advanced analysis"""
        try:
            if len(data) < 40:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 40"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect inverted hammer patterns
            detected_patterns = self._detect_inverted_hammer_patterns(enhanced_data)
            
            # Apply comprehensive analysis pipeline
            if self.parameters['volume_analysis']:
                detected_patterns = self._analyze_volume_confirmation(detected_patterns, enhanced_data)
            
            if self.parameters['support_analysis']:
                detected_patterns = self._analyze_support_strength(detected_patterns, enhanced_data)
            
            if self.parameters['accumulation_analysis']:
                detected_patterns = self._analyze_accumulation_signals(detected_patterns, enhanced_data)
            
            if self.parameters['pattern_purity_analysis']:
                detected_patterns = self._analyze_pattern_purity(detected_patterns, enhanced_data)
            
            if self.parameters['ml_reversal_prediction'] and detected_patterns:
                detected_patterns = self._predict_reversal_probability(detected_patterns, enhanced_data)
            
            # Generate comprehensive analysis
            pattern_analytics = self._generate_pattern_analytics(detected_patterns)
            trend_analysis = self._analyze_current_trend_context(enhanced_data)
            reversal_signals = self._generate_reversal_signals(detected_patterns, enhanced_data)
            
            return {
                'current_pattern': detected_patterns[-1] if detected_patterns else None,
                'recent_patterns': detected_patterns[-8:],
                'pattern_analytics': pattern_analytics,
                'trend_analysis': trend_analysis,
                'reversal_signals': reversal_signals,
                'market_structure': self._analyze_market_structure(enhanced_data),
                'pattern_reliability': self._assess_pattern_reliability(detected_patterns),
                'support_levels': self._identify_key_support_levels(enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Inverted hammer calculation failed: {str(e)}", e
            )
    
    def _enhance_data_with_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with comprehensive technical indicators"""
        df = data.copy()
        
        # Candlestick components
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Ratios and positions
        df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
        df['upper_shadow_ratio'] = np.where(df['total_range'] > 0, df['upper_shadow'] / df['total_range'], 0)
        df['lower_shadow_ratio'] = np.where(df['total_range'] > 0, df['lower_shadow'] / df['total_range'], 0)
        df['body_position'] = np.where(df['total_range'] > 0, 
                                      (np.minimum(df['open'], df['close']) - df['low']) / df['total_range'], 0.5)
        
        # Trend indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # Technical indicators
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility_ratio'] = df['total_range'] / df['atr']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Advanced metrics
        df['trend_strength'] = self._calculate_trend_strength_series(df)
        df['trend_direction'] = np.where(df['close'] > df['sma_20'], 1, -1)
        df['pivot_low'] = df['low'].rolling(5, center=True).min() == df['low']
        df['price_momentum'] = df['close'].pct_change(5)
        df['momentum_zscore'] = df.rolling(20)['price_momentum'].apply(lambda x: zscore(x)[-1] if len(x) == 20 else 0)
        
        return df
    
    def _calculate_trend_strength_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength as a rolling series"""
        def trend_strength_window(window_data):
            if len(window_data) < 8:
                return 0.5
            
            x = np.arange(len(window_data))
            slope, _, r_value, _, _ = linregress(x, window_data['close'])
            
            price_range = window_data['close'].max() - window_data['close'].min()
            if price_range == 0:
                return 0.5
            
            normalized_slope = abs(slope) / price_range * len(window_data)
            trend_quality = r_value ** 2
            strength = normalized_slope * trend_quality
            return min(max(strength, 0), 1)
        
        return df.rolling(self.parameters['trend_lookback']).apply(trend_strength_window, raw=False)['close']
    
    def _detect_inverted_hammer_patterns(self, data: pd.DataFrame) -> List[InvertedHammerPattern]:
        """Detect inverted hammer patterns with sophisticated analysis"""
        patterns = []
        
        for i in range(self.parameters['trend_lookback'], len(data)):
            candle = data.iloc[i]
            
            # Check for inverted hammer pattern
            if not self._is_inverted_hammer_pattern(candle):
                continue
            
            # Assess trend context (must be downtrend)
            trend_context_score = self._assess_downtrend_context(data, i)
            
            if trend_context_score < self.parameters['min_trend_strength']:
                continue
            
            # Calculate pattern metrics
            body_to_shadow_ratio = self._calculate_body_to_shadow_ratio(candle)
            pattern_strength = self._calculate_pattern_strength(candle, trend_context_score)
            
            if pattern_strength >= 0.6:
                pattern = InvertedHammerPattern(
                    timestamp=candle.name,
                    pattern_strength=pattern_strength,
                    body_to_shadow_ratio=body_to_shadow_ratio,
                    upper_shadow_length=candle['upper_shadow_ratio'],
                    lower_shadow_ratio=candle['lower_shadow_ratio'],
                    body_position=candle['body_position'],
                    trend_context_score=trend_context_score,
                    volume_confirmation=0.0,
                    support_strength=0.0,
                    reversal_probability=0.0,
                    accumulation_signal=0.0,
                    pattern_purity=0.0
                )
                patterns.append(pattern)
        
        return patterns
    
    def _is_inverted_hammer_pattern(self, candle: pd.Series) -> bool:
        """Check if candle meets inverted hammer criteria"""
        # Inverted hammer has long upper shadow, small body, minimal lower shadow
        return (candle['upper_shadow_ratio'] >= self.parameters['min_upper_shadow_ratio'] and
                candle['body_ratio'] <= self.parameters['max_body_ratio'] and
                candle['lower_shadow_ratio'] <= self.parameters['max_lower_shadow_ratio'] and
                candle['body_position'] <= 0.3 and
                candle['volatility_ratio'] >= 0.8)
    
    def _assess_downtrend_context(self, data: pd.DataFrame, candle_index: int) -> float:
        """Assess downtrend context before inverted hammer"""
        context_data = data.iloc[max(0, candle_index - self.parameters['trend_lookback']):candle_index + 1]
        
        if len(context_data) < 8:
            return 0.0
        
        context_factors = []
        
        # Price trend should be downward
        price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
        if price_change < -0.05:
            trend_factor = min(abs(price_change) / 0.15, 1.0)
            context_factors.append(trend_factor * 0.3)
        else:
            context_factors.append(0.0)
        
        # Moving average alignment (bearish)
        latest = context_data.iloc[-1]
        ma_bearish_score = 0
        if latest['close'] < latest['sma_5']:
            ma_bearish_score += 0.25
        if latest['sma_5'] < latest['sma_10']:
            ma_bearish_score += 0.25
        if latest['sma_10'] < latest['sma_20']:
            ma_bearish_score += 0.25
        if latest['ema_8'] < latest['ema_21']:
            ma_bearish_score += 0.25
        context_factors.append(ma_bearish_score * 0.25)
        
        # RSI oversold condition
        rsi_factor = 0
        if latest['rsi'] < 20:
            rsi_factor = 1.0
        elif latest['rsi'] < 30:
            rsi_factor = 0.8
        elif latest['rsi'] < 40:
            rsi_factor = 0.5
        context_factors.append(rsi_factor * 0.2)
        
        # Trend strength
        trend_strength = latest['trend_strength'] if not pd.isna(latest['trend_strength']) else 0.5
        context_factors.append(trend_strength * 0.15)
        
        # Lower lows pattern
        recent_lows = context_data['low'].iloc[-5:]
        lower_lows_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] < recent_lows.iloc[i-1])
        lower_lows_factor = lower_lows_count / 4
        context_factors.append(lower_lows_factor * 0.1)
        
        return sum(context_factors)
    
    def _calculate_body_to_shadow_ratio(self, candle: pd.Series) -> float:
        """Calculate body to shadow ratio quality"""
        if candle['upper_shadow_ratio'] == 0:
            return 0.0
        
        ratio = candle['body_ratio'] / candle['upper_shadow_ratio']
        
        if ratio <= 0.1:
            return 1.0
        elif ratio <= 0.2:
            return 0.9
        elif ratio <= 0.3:
            return 0.7
        elif ratio <= 0.5:
            return 0.5
        else:
            return max(0, 1.0 - ratio)
    
    def _calculate_pattern_strength(self, candle: pd.Series, trend_context_score: float) -> float:
        """Calculate overall inverted hammer pattern strength"""
        strength_components = [
            min(candle['upper_shadow_ratio'] / 0.8, 1.0) * 0.3,  # Upper shadow quality
            (1.0 - candle['body_ratio'] / self.parameters['max_body_ratio']) * 0.25,  # Body size quality
            (1.0 - candle['lower_shadow_ratio'] / self.parameters['max_lower_shadow_ratio']) * 0.2,  # Lower shadow quality
            min(max((0.3 - candle['body_position']) / 0.3, 0), 1) * 0.15,  # Body position quality
            trend_context_score * 0.1  # Trend context
        ]
        
        return sum(strength_components)
    
    def _analyze_volume_confirmation(self, patterns: List[InvertedHammerPattern], 
                                   data: pd.DataFrame) -> List[InvertedHammerPattern]:
        """Analyze volume confirmation for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            volume_score = self._calculate_volume_confirmation_score(data, pattern_idx)
            pattern.volume_confirmation = volume_score
            pattern.pattern_strength = (pattern.pattern_strength * 0.85 + volume_score * 0.15)
        
        return patterns
    
    def _calculate_volume_confirmation_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate volume confirmation score"""
        try:
            candle = data.iloc[pattern_idx]
            context_data = data.iloc[max(0, pattern_idx - 15):pattern_idx + 1]
            
            avg_volume = context_data['volume'].iloc[:-1].mean()
            volume_surge = candle['volume'] / avg_volume if avg_volume > 0 else 1.0
            surge_score = min(volume_surge / self.parameters['volume_surge_threshold'], 1.0)
            
            # Volume trend analysis
            recent_volumes = context_data['volume'].iloc[-5:]
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            volume_trend_score = min(max(volume_trend / avg_volume * 5, 0), 1) if avg_volume > 0 else 0.5
            
            # Accumulation volume (high volume with good price action suggests accumulation)
            price_progress = abs(candle['close'] - candle['open']) / candle['total_range'] if candle['total_range'] > 0 else 0
            accumulation_factor = volume_surge * price_progress
            
            return (surge_score * 0.4 + volume_trend_score * 0.3 + 
                   min(accumulation_factor / 2, 1.0) * 0.3)
            
        except Exception:
            return 0.5
    
    def _analyze_support_strength(self, patterns: List[InvertedHammerPattern], 
                                data: pd.DataFrame) -> List[InvertedHammerPattern]:
        """Analyze support strength for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            support_score = self._calculate_support_strength_score(data, pattern_idx)
            pattern.support_strength = support_score
        
        return patterns
    
    def _calculate_support_strength_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate support strength score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 40):pattern_idx + 1]
            inverted_hammer_low = data.iloc[pattern_idx]['low']
            
            # Historical support touches
            support_touches = 0
            price_tolerance = inverted_hammer_low * 0.01
            
            for i in range(len(context_data) - 1):
                if abs(context_data.iloc[i]['low'] - inverted_hammer_low) <= price_tolerance:
                    support_touches += 1
            
            support_strength = min(support_touches / 3, 1.0)
            
            # Pivot low confluence
            pivot_lows = context_data[context_data['pivot_low']]['low']
            pivot_confluence = 0.15
            if len(pivot_lows) > 0:
                nearest_pivot_distance = min(abs(pivot_lows - inverted_hammer_low))
                pivot_confluence = 1.0 / (1.0 + nearest_pivot_distance / inverted_hammer_low * 100) * 0.3
            
            # Volume at support
            support_volumes = []
            for i in range(len(context_data) - 1):
                if abs(context_data.iloc[i]['low'] - inverted_hammer_low) <= price_tolerance:
                    support_volumes.append(context_data.iloc[i]['volume'])
            
            volume_support_factor = 0.1
            if support_volumes:
                avg_support_volume = np.mean(support_volumes)
                avg_volume = context_data['volume'].mean()
                volume_support_factor = min(avg_support_volume / avg_volume, 1.0) * 0.2 if avg_volume > 0 else 0.1
            
            return support_strength * 0.4 + pivot_confluence + volume_support_factor + 0.1
            
        except Exception:
            return 0.5
    
    def _analyze_accumulation_signals(self, patterns: List[InvertedHammerPattern], 
                                    data: pd.DataFrame) -> List[InvertedHammerPattern]:
        """Analyze accumulation signals for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            accumulation_score = self._calculate_accumulation_signal_score(data, pattern_idx)
            pattern.accumulation_signal = accumulation_score
        
        return patterns
    
    def _calculate_accumulation_signal_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate accumulation signal score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 8):pattern_idx + 1]
            
            # Order flow analysis (positive for accumulation)
            obv_change = context_data['obv'].diff().iloc[-3:].mean()
            ad_line_change = context_data['ad_line'].diff().iloc[-3:].mean()
            
            flow_score = 0
            if obv_change > 0:  # Positive OBV suggests accumulation
                flow_score += 0.5
            if ad_line_change > 0:  # Positive A/D suggests accumulation
                flow_score += 0.5
            
            # CMF analysis
            cmf_value = data.iloc[pattern_idx]['cmf']
            cmf_score = 0
            if cmf_value > 0:  # Positive CMF suggests accumulation
                cmf_score = min(cmf_value / 100000, 1.0)
            
            # Volume vs price action
            price_action = abs(data.iloc[pattern_idx]['close'] - data.iloc[pattern_idx]['open'])
            volume_action = data.iloc[pattern_idx]['volume']
            avg_volume = context_data['volume'].iloc[:-1].mean()
            
            # High volume with good price action suggests accumulation
            accumulation_ratio = (volume_action / avg_volume) * (price_action + 0.001) if avg_volume > 0 else 1.0
            accumulation_score = min(accumulation_ratio / 3, 1.0)
            
            return flow_score * 0.4 + cmf_score * 0.3 + accumulation_score * 0.3
            
        except Exception:
            return 0.5
    
    def _analyze_pattern_purity(self, patterns: List[InvertedHammerPattern], 
                              data: pd.DataFrame) -> List[InvertedHammerPattern]:
        """Analyze pattern purity for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            purity_score = self._calculate_pattern_purity_score(data, pattern_idx)
            pattern.pattern_purity = purity_score
        
        return patterns
    
    def _calculate_pattern_purity_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate pattern purity score"""
        try:
            inverted_hammer_candle = data.iloc[pattern_idx]
            
            # Isolation quality
            surrounding_shadows = []
            for i in range(max(0, pattern_idx - 2), min(len(data), pattern_idx + 3)):
                if i != pattern_idx:
                    surrounding_shadows.append(data.iloc[i]['upper_shadow_ratio'])
            
            isolation_quality = 0.3
            if surrounding_shadows:
                inverted_hammer_shadow = inverted_hammer_candle['upper_shadow_ratio']
                max_surrounding = max(surrounding_shadows)
                isolation_quality = max(0, min(1, (inverted_hammer_shadow - max_surrounding) / inverted_hammer_shadow)) * 0.3
            
            # Geometric perfection
            ideal_proportions = {'upper_shadow': 0.75, 'body': 0.15, 'lower_shadow': 0.1}
            deviations = [
                abs(inverted_hammer_candle['upper_shadow_ratio'] - ideal_proportions['upper_shadow']),
                abs(inverted_hammer_candle['body_ratio'] - ideal_proportions['body']),
                abs(inverted_hammer_candle['lower_shadow_ratio'] - ideal_proportions['lower_shadow'])
            ]
            
            geometric_score = max(0, 1.0 - sum(deviations) / 0.5) * 0.25
            
            # Color and context consistency
            is_bullish = inverted_hammer_candle['close'] > inverted_hammer_candle['open']
            prev_trend = data.iloc[max(0, pattern_idx - 5):pattern_idx]['close'].mean()
            current_close = inverted_hammer_candle['close']
            
            color_score = 0.3 if (current_close < prev_trend and is_bullish) else 0.2
            
            # Shadow symmetry
            shadow_ratio = inverted_hammer_candle['lower_shadow_ratio'] / inverted_hammer_candle['upper_shadow_ratio'] if inverted_hammer_candle['upper_shadow_ratio'] > 0 else 1
            symmetry_score = (1.0 / (1.0 + shadow_ratio * 5)) * 0.15
            
            return isolation_quality + geometric_score + color_score + symmetry_score
            
        except Exception:
            return 0.5
    
    def _predict_reversal_probability(self, patterns: List[InvertedHammerPattern], 
                                    data: pd.DataFrame) -> List[InvertedHammerPattern]:
        """Predict reversal probability using ML"""
        if not patterns:
            return patterns
        
        try:
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_reversal_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 8:
                for pattern in patterns:
                    pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
                return patterns
            
            if not self.is_ml_fitted:
                self._train_reversal_model(patterns, features)
            
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                reversal_predictions = self.reversal_predictor.predict(features_scaled)
                
                for i, pattern in enumerate(patterns):
                    ml_probability = max(0.1, min(0.95, reversal_predictions[i]))
                    heuristic_prob = self._heuristic_reversal_probability(pattern)
                    pattern.reversal_probability = (ml_probability * 0.7 + heuristic_prob * 0.3)
            else:
                for pattern in patterns:
                    pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML reversal prediction failed: {str(e)}")
            for pattern in patterns:
                pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
            return patterns
    
    def _extract_reversal_features(self, data: pd.DataFrame, pattern_idx: int, 
                                 pattern: InvertedHammerPattern) -> List[float]:
        """Extract features for reversal prediction ML model"""
        try:
            current = data.iloc[pattern_idx]
            
            features = [
                pattern.pattern_strength,
                pattern.body_to_shadow_ratio,
                pattern.upper_shadow_length,
                pattern.lower_shadow_ratio,
                pattern.body_position,
                pattern.trend_context_score,
                pattern.volume_confirmation,
                pattern.support_strength,
                pattern.accumulation_signal,
                pattern.pattern_purity,
                current['rsi'] / 100.0 if not pd.isna(current['rsi']) else 0.5,
                current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
                current['volatility_ratio'] / 2.0,
                current['volume_ratio'],
                current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
                current['macd_hist'] if not pd.isna(current['macd_hist']) else 0,
                current['momentum_zscore'] if not pd.isna(current['momentum_zscore']) else 0,
                data.iloc[max(0, pattern_idx - 3):pattern_idx + 1]['obv'].diff().mean() / 1000000,
                data.iloc[max(0, pattern_idx - 3):pattern_idx + 1]['ad_line'].diff().mean() / 1000000,
                current['stoch_k'] / 100.0 if not pd.isna(current['stoch_k']) else 0.5
            ]
            
            return features
            
        except Exception:
            return [0.5] * 20
    
    def _train_reversal_model(self, patterns: List[InvertedHammerPattern], features: List[List[float]]):
        """Train ML model for reversal prediction"""
        try:
            targets = []
            for pattern in patterns:
                target = (
                    pattern.pattern_strength * 0.3 +
                    pattern.volume_confirmation * 0.25 +
                    pattern.support_strength * 0.2 +
                    pattern.accumulation_signal * 0.15 +
                    pattern.pattern_purity * 0.1
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 12:
                features_scaled = self.scaler.fit_transform(features)
                self.reversal_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML reversal predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _heuristic_reversal_probability(self, pattern: InvertedHammerPattern) -> float:
        """Calculate heuristic reversal probability"""
        return (
            pattern.pattern_strength * 0.35 +
            pattern.volume_confirmation * 0.25 +
            pattern.support_strength * 0.2 +
            pattern.accumulation_signal * 0.2
        )
    
    def _generate_pattern_analytics(self, patterns: List[InvertedHammerPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-15:]
        
        return {
            'total_patterns': len(recent_patterns),
            'average_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'average_reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'average_volume_confirmation': sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns),
            'high_strength_patterns': len([p for p in recent_patterns if p.pattern_strength > 0.8]),
            'high_probability_patterns': len([p for p in recent_patterns if p.reversal_probability > 0.7]),
            'accumulation_signals': len([p for p in recent_patterns if p.accumulation_signal > 0.7]),
            'high_purity_patterns': len([p for p in recent_patterns if p.pattern_purity > 0.8]),
            'strong_support_patterns': len([p for p in recent_patterns if p.support_strength > 0.7])
        }
    
    def _analyze_current_trend_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current trend context"""
        current = data.iloc[-1]
        
        return {
            'trend_strength': current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
            'is_downtrend': current['trend_direction'] == -1,
            'rsi_oversold': current['rsi'] < 30 if not pd.isna(current['rsi']) else False,
            'stoch_oversold': current['stoch_k'] < 20 if not pd.isna(current['stoch_k']) else False,
            'bb_oversold': current['bb_position'] < 0.2 if not pd.isna(current['bb_position']) else False,
            'momentum_extreme': abs(current['momentum_zscore']) > 1.5 if not pd.isna(current['momentum_zscore']) else False,
            'volatility_context': 'high' if current['volatility_ratio'] > 1.5 else 'normal'
        }
    
    def _generate_reversal_signals(self, patterns: List[InvertedHammerPattern], 
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Generate reversal signals based on patterns"""
        if not patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        recent_patterns = [p for p in patterns[-5:] if p.pattern_strength > 0.7]
        
        if not recent_patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        return {
            'signal_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'volume_confirmation': sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns),
            'pattern_count': len(recent_patterns),
            'accumulation_signal': sum(p.accumulation_signal for p in recent_patterns) / len(recent_patterns),
            'support_strength': sum(p.support_strength for p in recent_patterns) / len(recent_patterns),
            'pattern_purity': sum(p.pattern_purity for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': recent_patterns[-1].timestamp
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        current = data.iloc[-1]
        recent_data = data.iloc[-20:]
        
        return {
            'support_structure': {
                'major_support_levels': len(recent_data[recent_data['pivot_low']]),
                'current_position': 'near_support' if current['bb_position'] < 0.3 else 'neutral'
            },
            'volume_structure': {
                'average_volume': recent_data['volume'].mean(),
                'volume_trend': 'increasing' if recent_data['volume'].iloc[-5:].mean() > recent_data['volume'].iloc[-15:-5].mean() else 'decreasing'
            },
            'momentum_structure': {
                'momentum_divergence': self._detect_momentum_divergence(recent_data),
                'trend_exhaustion': current['rsi'] < 25 if not pd.isna(current['rsi']) else False
            }
        }
    
    def _detect_momentum_divergence(self, data: pd.DataFrame) -> bool:
        """Detect momentum divergence"""
        try:
            if len(data) < 10:
                return False
            
            recent_price_trend = np.polyfit(range(len(data)), data['low'], 1)[0]
            recent_rsi_trend = np.polyfit(range(len(data)), data['rsi'].fillna(50), 1)[0]
            
            # Bullish divergence: price making lower lows, RSI making higher lows
            return recent_price_trend < 0 and recent_rsi_trend > 0
            
        except Exception:
            return False
    
    def _assess_pattern_reliability(self, patterns: List[InvertedHammerPattern]) -> Dict[str, Any]:
        """Assess pattern reliability metrics"""
        if not patterns:
            return {}
        
        return {
            'consistency_score': 1.0 - np.std([p.pattern_strength for p in patterns]),
            'purity_average': np.mean([p.pattern_purity for p in patterns]),
            'volume_reliability': np.mean([p.volume_confirmation for p in patterns]),
            'accumulation_consistency': np.mean([p.accumulation_signal for p in patterns]),
            'reversal_success_estimate': np.mean([p.reversal_probability for p in patterns])
        }
    
    def _identify_key_support_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify key support levels"""
        recent_data = data.iloc[-50:]
        pivot_lows = recent_data[recent_data['pivot_low']]['low'].tolist()
        
        support_levels = []
        if pivot_lows:
            sorted_lows = sorted(pivot_lows)
            current_level = sorted_lows[0]
            level_touches = 1
            
            for low in sorted_lows[1:]:
                if abs(low - current_level) / current_level < 0.01:
                    level_touches += 1
                else:
                    if level_touches >= 2:
                        support_levels.append({
                            'level': current_level,
                            'touches': level_touches,
                            'strength': min(level_touches / 3, 1.0)
                        })
                    current_level = low
                    level_touches = 1
            
            if level_touches >= 2:
                support_levels.append({
                    'level': current_level,
                    'touches': level_touches,
                    'strength': min(level_touches / 3, 1.0)
                })
        
        return {
            'key_levels': support_levels,
            'nearest_support': min(support_levels, key=lambda x: x['level'])['level'] if support_levels else None,
            'support_density': len(support_levels)
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on inverted hammer analysis"""
        current_pattern = value.get('current_pattern')
        reversal_signals = value.get('reversal_signals', {})
        trend_analysis = value.get('trend_analysis', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Inverted hammer is a bullish reversal pattern
        if (current_pattern.pattern_strength > 0.8 and 
            current_pattern.reversal_probability > 0.7 and
            trend_analysis.get('is_downtrend', False) and
            current_pattern.support_strength > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.35 +
                current_pattern.reversal_probability * 0.25 +
                current_pattern.volume_confirmation * 0.2 +
                current_pattern.support_strength * 0.2
            )
            
            return SignalType.BUY, confidence
        
        elif (current_pattern.pattern_strength > 0.7 and 
              current_pattern.reversal_probability > 0.6 and
              trend_analysis.get('is_downtrend', False)):
            
            confidence = current_pattern.pattern_strength * 0.65
            return SignalType.BUY, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'inverted_hammer',
            'min_upper_shadow_ratio': self.parameters['min_upper_shadow_ratio'],
            'max_body_ratio': self.parameters['max_body_ratio'],
            'volume_analysis_enabled': self.parameters['volume_analysis'],
            'ml_reversal_prediction_enabled': self.parameters['ml_reversal_prediction']
        })
        return base_metadata