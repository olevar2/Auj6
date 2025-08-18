"""
Shooting Star Indicator - Advanced Single-Candle Bearish Reversal Pattern Detection
==================================================================================

This indicator implements sophisticated shooting star pattern detection with advanced
upper shadow analysis, uptrend context validation, and ML-enhanced reversal prediction.
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
class ShootingStarPattern:
    """Represents a detected shooting star pattern"""
    timestamp: pd.Timestamp
    pattern_strength: float
    body_to_shadow_ratio: float
    upper_shadow_length: float
    lower_shadow_ratio: float
    body_position: float
    trend_context_score: float
    volume_confirmation: float
    resistance_strength: float
    reversal_probability: float
    distribution_signal: float
    pattern_purity: float


class ShootingStarIndicator(StandardIndicatorInterface):
    """
    Advanced Shooting Star Pattern Indicator
    
    Features:
    - Precise shooting star identification with upper shadow analysis
    - Advanced trend context validation for uptrend requirement
    - Volume-based confirmation and distribution signal detection
    - ML-enhanced reversal probability prediction
    - Resistance level validation and strength assessment
    - Pattern purity scoring and reliability metrics
    - Statistical significance testing for pattern quality
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'min_upper_shadow_ratio': 0.65,    # Minimum upper shadow as % of total range
            'max_body_ratio': 0.25,            # Maximum body as % of total range
            'max_lower_shadow_ratio': 0.1,     # Maximum lower shadow as % of total range
            'min_body_position': 0.7,          # Minimum body position (near bottom)
            'min_trend_strength': 0.6,         # Minimum uptrend strength required
            'volume_surge_threshold': 1.3,     # Volume surge multiplier
            'trend_lookback': 15,              # Periods for trend analysis
            'resistance_analysis': True,
            'volume_analysis': True,
            'ml_reversal_prediction': True,
            'distribution_analysis': True,
            'pattern_purity_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="ShootingStarIndicator", parameters=default_params)
        
        # Initialize ML components
        self.reversal_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=150, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=90, random_state=42)),
            ('ada', AdaBoostRegressor(n_estimators=120, random_state=42))
        ])
        self.scaler = RobustScaler()
        self.is_ml_fitted = False
        
        logging.info(f"ShootingStarIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=40,
            lookback_periods=80
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate shooting star patterns with advanced analysis"""
        try:
            if len(data) < 40:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 40"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect shooting star patterns
            detected_patterns = self._detect_shooting_star_patterns(enhanced_data)
            
            # Apply comprehensive analysis pipeline
            if self.parameters['volume_analysis']:
                detected_patterns = self._analyze_volume_confirmation(detected_patterns, enhanced_data)
            
            if self.parameters['resistance_analysis']:
                detected_patterns = self._analyze_resistance_strength(detected_patterns, enhanced_data)
            
            if self.parameters['distribution_analysis']:
                detected_patterns = self._analyze_distribution_signals(detected_patterns, enhanced_data)
            
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
                'resistance_levels': self._identify_key_resistance_levels(enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Shooting star calculation failed: {str(e)}", e
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
        df['pivot_high'] = df['high'].rolling(5, center=True).max() == df['high']
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
    
    def _detect_shooting_star_patterns(self, data: pd.DataFrame) -> List[ShootingStarPattern]:
        """Detect shooting star patterns with sophisticated analysis"""
        patterns = []
        
        for i in range(self.parameters['trend_lookback'], len(data)):
            candle = data.iloc[i]
            
            # Check for shooting star pattern
            if not self._is_shooting_star_pattern(candle):
                continue
            
            # Assess trend context (must be uptrend)
            trend_context_score = self._assess_uptrend_context(data, i)
            
            if trend_context_score < self.parameters['min_trend_strength']:
                continue
            
            # Calculate pattern metrics
            body_to_shadow_ratio = self._calculate_body_to_shadow_ratio(candle)
            pattern_strength = self._calculate_pattern_strength(candle, trend_context_score)
            
            if pattern_strength >= 0.6:
                pattern = ShootingStarPattern(
                    timestamp=candle.name,
                    pattern_strength=pattern_strength,
                    body_to_shadow_ratio=body_to_shadow_ratio,
                    upper_shadow_length=candle['upper_shadow_ratio'],
                    lower_shadow_ratio=candle['lower_shadow_ratio'],
                    body_position=candle['body_position'],
                    trend_context_score=trend_context_score,
                    volume_confirmation=0.0,
                    resistance_strength=0.0,
                    reversal_probability=0.0,
                    distribution_signal=0.0,
                    pattern_purity=0.0
                )
                patterns.append(pattern)
        
        return patterns
    
    def _is_shooting_star_pattern(self, candle: pd.Series) -> bool:
        """Check if candle meets shooting star criteria"""
        # Shooting star has long upper shadow, small body near the low, minimal lower shadow
        return (candle['upper_shadow_ratio'] >= self.parameters['min_upper_shadow_ratio'] and
                candle['body_ratio'] <= self.parameters['max_body_ratio'] and
                candle['lower_shadow_ratio'] <= self.parameters['max_lower_shadow_ratio'] and
                candle['body_position'] >= self.parameters['min_body_position'] and
                candle['volatility_ratio'] >= 0.8)
    
    def _assess_uptrend_context(self, data: pd.DataFrame, candle_index: int) -> float:
        """Assess uptrend context before shooting star"""
        context_data = data.iloc[max(0, candle_index - self.parameters['trend_lookback']):candle_index + 1]
        
        if len(context_data) < 8:
            return 0.0
        
        context_factors = []
        
        # Price trend should be upward
        price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
        if price_change > 0.05:
            trend_factor = min(price_change / 0.15, 1.0)
            context_factors.append(trend_factor * 0.35)
        else:
            context_factors.append(0.0)
        
        # Moving average alignment (bullish)
        latest = context_data.iloc[-1]
        ma_bullish_score = 0
        if latest['close'] > latest['sma_5']:
            ma_bullish_score += 0.25
        if latest['sma_5'] > latest['sma_10']:
            ma_bullish_score += 0.25
        if latest['sma_10'] > latest['sma_20']:
            ma_bullish_score += 0.25
        if latest['ema_8'] > latest['ema_21']:
            ma_bullish_score += 0.25
        context_factors.append(ma_bullish_score * 0.25)
        
        # RSI overbought condition (good for reversal)
        rsi_factor = 0
        if latest['rsi'] > 85:
            rsi_factor = 1.0
        elif latest['rsi'] > 75:
            rsi_factor = 0.9
        elif latest['rsi'] > 65:
            rsi_factor = 0.6
        elif latest['rsi'] > 55:
            rsi_factor = 0.3
        context_factors.append(rsi_factor * 0.2)
        
        # Trend strength
        trend_strength = latest['trend_strength'] if not pd.isna(latest['trend_strength']) else 0.5
        context_factors.append(trend_strength * 0.1)
        
        # Higher highs pattern
        recent_highs = context_data['high'].iloc[-5:]
        higher_highs_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] > recent_highs.iloc[i-1])
        higher_highs_factor = higher_highs_count / 4
        context_factors.append(higher_highs_factor * 0.1)
        
        return sum(context_factors)
    
    def _calculate_body_to_shadow_ratio(self, candle: pd.Series) -> float:
        """Calculate body to shadow ratio quality"""
        if candle['upper_shadow_ratio'] == 0:
            return 0.0
        
        ratio = candle['body_ratio'] / candle['upper_shadow_ratio']
        
        # Better ratios get higher scores
        if ratio <= 0.08:
            return 1.0
        elif ratio <= 0.15:
            return 0.95
        elif ratio <= 0.25:
            return 0.8
        elif ratio <= 0.4:
            return 0.6
        else:
            return max(0, 1.0 - ratio * 1.5)
    
    def _calculate_pattern_strength(self, candle: pd.Series, trend_context_score: float) -> float:
        """Calculate overall shooting star pattern strength"""
        strength_components = [
            min(candle['upper_shadow_ratio'] / 0.8, 1.0) * 0.35,  # Upper shadow quality
            (1.0 - candle['body_ratio'] / self.parameters['max_body_ratio']) * 0.25,  # Body size quality
            (1.0 - candle['lower_shadow_ratio'] / self.parameters['max_lower_shadow_ratio']) * 0.15,  # Lower shadow quality
            min(max((candle['body_position'] - self.parameters['min_body_position']) / 0.3, 0), 1) * 0.15,  # Body position quality
            trend_context_score * 0.1  # Trend context
        ]
        
        return sum(strength_components)
    
    def _analyze_volume_confirmation(self, patterns: List[ShootingStarPattern], 
                                   data: pd.DataFrame) -> List[ShootingStarPattern]:
        """Analyze volume confirmation for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            volume_score = self._calculate_volume_confirmation_score(data, pattern_idx)
            pattern.volume_confirmation = volume_score
            pattern.pattern_strength = (pattern.pattern_strength * 0.8 + volume_score * 0.2)
        
        return patterns
    
    def _calculate_volume_confirmation_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate volume confirmation score"""
        try:
            candle = data.iloc[pattern_idx]
            context_data = data.iloc[max(0, pattern_idx - 15):pattern_idx + 1]
            
            avg_volume = context_data['volume'].iloc[:-1].mean()
            volume_surge = candle['volume'] / avg_volume if avg_volume > 0 else 1.0
            surge_score = min(volume_surge / self.parameters['volume_surge_threshold'], 1.0)
            
            # Volume spike with rejection (upper shadow) suggests distribution
            rejection_strength = candle['upper_shadow_ratio']
            volume_rejection_score = surge_score * rejection_strength * 2
            
            # Volume trend analysis
            recent_volumes = context_data['volume'].iloc[-5:]
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            volume_trend_score = min(max(volume_trend / avg_volume * 5, 0), 1) if avg_volume > 0 else 0.5
            
            # Distribution volume (high volume with rejection suggests distribution)
            distribution_factor = volume_surge * (1.0 - candle['body_ratio'])
            
            return (surge_score * 0.3 + volume_rejection_score * 0.4 + 
                   volume_trend_score * 0.2 + min(distribution_factor / 2, 1.0) * 0.1)
            
        except Exception:
            return 0.5
    
    def _analyze_resistance_strength(self, patterns: List[ShootingStarPattern], 
                                   data: pd.DataFrame) -> List[ShootingStarPattern]:
        """Analyze resistance strength for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            resistance_score = self._calculate_resistance_strength_score(data, pattern_idx)
            pattern.resistance_strength = resistance_score
        
        return patterns
    
    def _calculate_resistance_strength_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate resistance strength score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 40):pattern_idx + 1]
            shooting_star_high = data.iloc[pattern_idx]['high']
            
            # Historical resistance touches
            resistance_touches = 0
            price_tolerance = shooting_star_high * 0.012
            
            for i in range(len(context_data) - 1):
                if abs(context_data.iloc[i]['high'] - shooting_star_high) <= price_tolerance:
                    resistance_touches += 1
            
            resistance_strength = min(resistance_touches / 2.5, 1.0)
            
            # Pivot high confluence
            pivot_highs = context_data[context_data['pivot_high']]['high']
            pivot_confluence = 0.1
            if len(pivot_highs) > 0:
                nearest_pivot_distance = min(abs(pivot_highs - shooting_star_high))
                pivot_confluence = 1.0 / (1.0 + nearest_pivot_distance / shooting_star_high * 100) * 0.35
            
            # Volume at resistance
            resistance_volumes = []
            for i in range(len(context_data) - 1):
                if abs(context_data.iloc[i]['high'] - shooting_star_high) <= price_tolerance:
                    resistance_volumes.append(context_data.iloc[i]['volume'])
            
            volume_resistance_factor = 0.15
            if resistance_volumes:
                avg_resistance_volume = np.mean(resistance_volumes)
                avg_volume = context_data['volume'].mean()
                volume_resistance_factor = min(avg_resistance_volume / avg_volume, 1.0) * 0.25 if avg_volume > 0 else 0.15
            
            # Previous rejection strength
            rejection_strength = data.iloc[pattern_idx]['upper_shadow_ratio']
            rejection_factor = rejection_strength * 0.2
            
            return resistance_strength * 0.35 + pivot_confluence + volume_resistance_factor + rejection_factor
            
        except Exception:
            return 0.5
    
    def _analyze_distribution_signals(self, patterns: List[ShootingStarPattern], 
                                    data: pd.DataFrame) -> List[ShootingStarPattern]:
        """Analyze distribution signals for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            distribution_score = self._calculate_distribution_signal_score(data, pattern_idx)
            pattern.distribution_signal = distribution_score
        
        return patterns
    
    def _calculate_distribution_signal_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate distribution signal score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 8):pattern_idx + 1]
            
            # Order flow analysis (negative for distribution)
            obv_change = context_data['obv'].diff().iloc[-3:].mean()
            ad_line_change = context_data['ad_line'].diff().iloc[-3:].mean()
            
            flow_score = 0
            if obv_change < 0:  # Negative OBV suggests distribution
                flow_score += 0.5
            if ad_line_change < 0:  # Negative A/D suggests distribution
                flow_score += 0.5
            
            # CMF analysis
            cmf_value = data.iloc[pattern_idx]['cmf']
            cmf_score = 0
            if cmf_value < 0:  # Negative CMF suggests distribution
                cmf_score = min(abs(cmf_value) / 100000, 1.0)
            
            # High-low rejection analysis
            candle = data.iloc[pattern_idx]
            rejection_ratio = candle['upper_shadow_ratio']
            volume_surge = candle['volume'] / context_data['volume'].iloc[:-1].mean() if context_data['volume'].iloc[:-1].mean() > 0 else 1.0
            
            # Strong rejection with high volume suggests distribution
            rejection_distribution_score = rejection_ratio * min(volume_surge / 1.5, 1.0)
            
            return flow_score * 0.35 + cmf_score * 0.25 + rejection_distribution_score * 0.4
            
        except Exception:
            return 0.5
    
    def _analyze_pattern_purity(self, patterns: List[ShootingStarPattern], 
                              data: pd.DataFrame) -> List[ShootingStarPattern]:
        """Analyze pattern purity for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            purity_score = self._calculate_pattern_purity_score(data, pattern_idx)
            pattern.pattern_purity = purity_score
        
        return patterns
    
    def _calculate_pattern_purity_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate pattern purity score"""
        try:
            shooting_star_candle = data.iloc[pattern_idx]
            
            # Isolation quality
            surrounding_shadows = []
            for i in range(max(0, pattern_idx - 2), min(len(data), pattern_idx + 3)):
                if i != pattern_idx:
                    surrounding_shadows.append(data.iloc[i]['upper_shadow_ratio'])
            
            isolation_quality = 0.25
            if surrounding_shadows:
                shooting_star_shadow = shooting_star_candle['upper_shadow_ratio']
                max_surrounding = max(surrounding_shadows)
                isolation_quality = max(0, min(1, (shooting_star_shadow - max_surrounding) / shooting_star_shadow)) * 0.35
            
            # Geometric perfection
            ideal_proportions = {'upper_shadow': 0.8, 'body': 0.12, 'lower_shadow': 0.08}
            deviations = [
                abs(shooting_star_candle['upper_shadow_ratio'] - ideal_proportions['upper_shadow']),
                abs(shooting_star_candle['body_ratio'] - ideal_proportions['body']),
                abs(shooting_star_candle['lower_shadow_ratio'] - ideal_proportions['lower_shadow'])
            ]
            
            geometric_score = max(0, 1.0 - sum(deviations) / 0.6) * 0.3
            
            # Color and context consistency (bearish after uptrend)
            is_bearish = shooting_star_candle['close'] < shooting_star_candle['open']
            prev_trend = data.iloc[max(0, pattern_idx - 5):pattern_idx]['close'].mean()
            current_close = shooting_star_candle['close']
            
            color_score = 0.25 if (current_close > prev_trend and is_bearish) else 0.15
            
            # Shadow symmetry (minimal lower shadow)
            lower_shadow_penalty = shooting_star_candle['lower_shadow_ratio'] * 3
            symmetry_score = max(0, 0.1 - lower_shadow_penalty)
            
            return isolation_quality + geometric_score + color_score + symmetry_score
            
        except Exception:
            return 0.5
    
    def _predict_reversal_probability(self, patterns: List[ShootingStarPattern], 
                                    data: pd.DataFrame) -> List[ShootingStarPattern]:
        """Predict reversal probability using ML"""
        if not patterns:
            return patterns
        
        try:
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_reversal_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 10:
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
                    pattern.reversal_probability = (ml_probability * 0.75 + heuristic_prob * 0.25)
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
                                 pattern: ShootingStarPattern) -> List[float]:
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
                pattern.resistance_strength,
                pattern.distribution_signal,
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
                current['stoch_k'] / 100.0 if not pd.isna(current['stoch_k']) else 0.5,
                pattern.upper_shadow_length * pattern.volume_confirmation,  # Combined rejection signal
                min(current['rsi'] / 70, 1.0) if not pd.isna(current['rsi']) else 0.5  # Overbought factor
            ]
            
            return features
            
        except Exception:
            return [0.5] * 22
    
    def _train_reversal_model(self, patterns: List[ShootingStarPattern], features: List[List[float]]):
        """Train ML model for reversal prediction"""
        try:
            targets = []
            for pattern in patterns:
                target = (
                    pattern.pattern_strength * 0.3 +
                    pattern.volume_confirmation * 0.25 +
                    pattern.resistance_strength * 0.2 +
                    pattern.distribution_signal * 0.15 +
                    pattern.pattern_purity * 0.1
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 15:
                features_scaled = self.scaler.fit_transform(features)
                self.reversal_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML reversal predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _heuristic_reversal_probability(self, pattern: ShootingStarPattern) -> float:
        """Calculate heuristic reversal probability"""
        return (
            pattern.pattern_strength * 0.35 +
            pattern.volume_confirmation * 0.25 +
            pattern.resistance_strength * 0.2 +
            pattern.distribution_signal * 0.2
        )
    
    def _generate_pattern_analytics(self, patterns: List[ShootingStarPattern]) -> Dict[str, Any]:
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
            'distribution_signals': len([p for p in recent_patterns if p.distribution_signal > 0.7]),
            'high_purity_patterns': len([p for p in recent_patterns if p.pattern_purity > 0.8]),
            'strong_resistance_patterns': len([p for p in recent_patterns if p.resistance_strength > 0.7])
        }
    
    def _analyze_current_trend_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current trend context"""
        current = data.iloc[-1]
        
        return {
            'trend_strength': current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
            'is_uptrend': current['trend_direction'] == 1,
            'rsi_overbought': current['rsi'] > 70 if not pd.isna(current['rsi']) else False,
            'stoch_overbought': current['stoch_k'] > 80 if not pd.isna(current['stoch_k']) else False,
            'bb_overbought': current['bb_position'] > 0.8 if not pd.isna(current['bb_position']) else False,
            'momentum_extreme': abs(current['momentum_zscore']) > 1.5 if not pd.isna(current['momentum_zscore']) else False,
            'volatility_context': 'high' if current['volatility_ratio'] > 1.5 else 'normal'
        }
    
    def _generate_reversal_signals(self, patterns: List[ShootingStarPattern], 
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
            'distribution_signal': sum(p.distribution_signal for p in recent_patterns) / len(recent_patterns),
            'resistance_strength': sum(p.resistance_strength for p in recent_patterns) / len(recent_patterns),
            'pattern_purity': sum(p.pattern_purity for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': recent_patterns[-1].timestamp
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        current = data.iloc[-1]
        recent_data = data.iloc[-20:]
        
        return {
            'resistance_structure': {
                'major_resistance_levels': len(recent_data[recent_data['pivot_high']]),
                'current_position': 'near_resistance' if current['bb_position'] > 0.75 else 'neutral'
            },
            'volume_structure': {
                'average_volume': recent_data['volume'].mean(),
                'volume_trend': 'increasing' if recent_data['volume'].iloc[-5:].mean() > recent_data['volume'].iloc[-15:-5].mean() else 'decreasing',
                'distribution_volume': len([i for i in range(-5, 0) if recent_data.iloc[i]['volume'] > recent_data['volume'].mean() * 1.3])
            },
            'momentum_structure': {
                'momentum_divergence': self._detect_momentum_divergence(recent_data),
                'trend_exhaustion': current['rsi'] > 80 if not pd.isna(current['rsi']) else False,
                'overbought_conditions': sum([
                    current['rsi'] > 75 if not pd.isna(current['rsi']) else False,
                    current['stoch_k'] > 85 if not pd.isna(current['stoch_k']) else False,
                    current['bb_position'] > 0.9 if not pd.isna(current['bb_position']) else False
                ])
            }
        }
    
    def _detect_momentum_divergence(self, data: pd.DataFrame) -> bool:
        """Detect momentum divergence"""
        try:
            if len(data) < 10:
                return False
            
            recent_price_trend = np.polyfit(range(len(data)), data['high'], 1)[0]
            recent_rsi_trend = np.polyfit(range(len(data)), data['rsi'].fillna(50), 1)[0]
            
            # Bearish divergence: price making higher highs, RSI making lower highs
            return recent_price_trend > 0 and recent_rsi_trend < -0.3
            
        except Exception:
            return False
    
    def _assess_pattern_reliability(self, patterns: List[ShootingStarPattern]) -> Dict[str, Any]:
        """Assess pattern reliability metrics"""
        if not patterns:
            return {}
        
        return {
            'consistency_score': 1.0 - np.std([p.pattern_strength for p in patterns]),
            'purity_average': np.mean([p.pattern_purity for p in patterns]),
            'volume_reliability': np.mean([p.volume_confirmation for p in patterns]),
            'distribution_consistency': np.mean([p.distribution_signal for p in patterns]),
            'reversal_success_estimate': np.mean([p.reversal_probability for p in patterns])
        }
    
    def _identify_key_resistance_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify key resistance levels"""
        recent_data = data.iloc[-50:]
        pivot_highs = recent_data[recent_data['pivot_high']]['high'].tolist()
        
        resistance_levels = []
        if pivot_highs:
            sorted_highs = sorted(pivot_highs, reverse=True)
            current_level = sorted_highs[0]
            level_touches = 1
            
            for high in sorted_highs[1:]:
                if abs(high - current_level) / current_level < 0.01:
                    level_touches += 1
                else:
                    if level_touches >= 2:
                        resistance_levels.append({
                            'level': current_level,
                            'touches': level_touches,
                            'strength': min(level_touches / 3, 1.0)
                        })
                    current_level = high
                    level_touches = 1
            
            if level_touches >= 2:
                resistance_levels.append({
                    'level': current_level,
                    'touches': level_touches,
                    'strength': min(level_touches / 3, 1.0)
                })
        
        return {
            'key_levels': resistance_levels,
            'nearest_resistance': max(resistance_levels, key=lambda x: x['level'])['level'] if resistance_levels else None,
            'resistance_density': len(resistance_levels)
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on shooting star analysis"""
        current_pattern = value.get('current_pattern')
        reversal_signals = value.get('reversal_signals', {})
        trend_analysis = value.get('trend_analysis', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Shooting star is a bearish reversal pattern
        if (current_pattern.pattern_strength > 0.85 and 
            current_pattern.reversal_probability > 0.75 and
            trend_analysis.get('is_uptrend', False) and
            current_pattern.resistance_strength > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.35 +
                current_pattern.reversal_probability * 0.25 +
                current_pattern.volume_confirmation * 0.2 +
                current_pattern.resistance_strength * 0.2
            )
            
            return SignalType.SELL, confidence
        
        elif (current_pattern.pattern_strength > 0.75 and 
              current_pattern.reversal_probability > 0.65 and
              trend_analysis.get('is_uptrend', False)):
            
            confidence = current_pattern.pattern_strength * 0.7
            return SignalType.SELL, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'shooting_star',
            'min_upper_shadow_ratio': self.parameters['min_upper_shadow_ratio'],
            'max_body_ratio': self.parameters['max_body_ratio'],
            'volume_analysis_enabled': self.parameters['volume_analysis'],
            'ml_reversal_prediction_enabled': self.parameters['ml_reversal_prediction']
        })
        return base_metadata