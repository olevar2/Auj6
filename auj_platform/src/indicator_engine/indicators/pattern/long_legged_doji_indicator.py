"""
Long-Legged Doji Indicator - Advanced Market Indecision Pattern Recognition
==========================================================================

This indicator implements sophisticated long-legged doji detection with uncertainty
quantification, market equilibrium analysis, and ML-enhanced volatility prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import logging
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PowerTransformer
import talib
from scipy import stats
from scipy.stats import entropy

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    IndicatorResult, 
    SignalType, 
    DataType, 
    DataRequirement
)
from ...core.exceptions import IndicatorCalculationException


@dataclass
class LongLeggedDojiPattern:
    """Represents a detected long-legged doji pattern"""
    timestamp: pd.Timestamp
    strength: float
    upper_shadow_ratio: float
    lower_shadow_ratio: float
    body_ratio: float
    shadow_balance: float
    market_indecision_score: float
    volatility_context: str
    breakout_probability: float
    directional_uncertainty: float
    equilibrium_quality: float


class LongLeggedDojiIndicator(StandardIndicatorInterface):
    """
    Advanced Long-Legged Doji Pattern Indicator
    
    Features:
    - Precise long-legged doji identification with shadow balance analysis
    - Market indecision and uncertainty quantification using information theory
    - Volatility regime detection and context analysis
    - Machine learning breakout direction and probability prediction
    - Market equilibrium quality assessment
    - Multi-timeframe uncertainty convergence analysis
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'max_body_ratio': 0.08,        # Maximum body size for long-legged doji
            'min_total_shadow': 0.75,      # Minimum combined shadow ratio
            'max_shadow_imbalance': 0.25,  # Maximum imbalance between shadows
            'min_volatility_ratio': 0.8,   # Minimum volatility for significance
            'uncertainty_threshold': 0.7,   # Minimum uncertainty for pattern validity
            'volume_confirmation': False,   # Volume not critical for indecision patterns
            'market_equilibrium_analysis': True,
            'ml_breakout_prediction': True,
            'information_theory_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="LongLeggedDojiIndicator", parameters=default_params)
        
        # Initialize ML components
        self.breakout_predictor = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.scaler = PowerTransformer(method='yeo-johnson')
        self.is_ml_fitted = False
        
        # Pattern analysis components
        self.uncertainty_analyzer = self._initialize_uncertainty_analyzer()
        
        logging.info(f"LongLeggedDojiIndicator initialized with parameters: {self.parameters}")
    
    def _initialize_uncertainty_analyzer(self) -> Dict[str, Any]:
        """Initialize uncertainty analysis components"""
        return {
            'entropy_calculator': self._calculate_market_entropy,
            'equilibrium_detector': self._detect_market_equilibrium,
            'uncertainty_quantifier': self._quantify_directional_uncertainty
        }
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=50,
            lookback_periods=120
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate long-legged doji patterns with advanced uncertainty analysis"""
        try:
            if len(data) < 50:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 50"
                )
            
            # Enhance data with uncertainty and volatility indicators
            enhanced_data = self._enhance_data_with_uncertainty_indicators(data)
            
            # Detect long-legged doji patterns
            detected_patterns = self._detect_long_legged_doji_patterns(enhanced_data)
            
            # Apply information theory analysis
            if self.parameters['information_theory_analysis']:
                info_enhanced_patterns = self._apply_information_theory_analysis(
                    detected_patterns, enhanced_data
                )
            else:
                info_enhanced_patterns = detected_patterns
            
            # Apply market equilibrium analysis
            if self.parameters['market_equilibrium_analysis']:
                equilibrium_enhanced_patterns = self._analyze_market_equilibrium(
                    info_enhanced_patterns, enhanced_data
                )
            else:
                equilibrium_enhanced_patterns = info_enhanced_patterns
            
            # Apply ML breakout prediction
            if self.parameters['ml_breakout_prediction'] and equilibrium_enhanced_patterns:
                ml_enhanced_patterns = self._predict_breakout_probabilities(
                    equilibrium_enhanced_patterns, enhanced_data
                )
            else:
                ml_enhanced_patterns = equilibrium_enhanced_patterns
            
            # Generate comprehensive analysis
            uncertainty_analysis = self._analyze_market_uncertainty(enhanced_data)
            volatility_analysis = self._analyze_volatility_regime(enhanced_data)
            pattern_analytics = self._generate_pattern_analytics(ml_enhanced_patterns)
            
            return {
                'current_pattern': ml_enhanced_patterns[-1] if ml_enhanced_patterns else None,
                'recent_patterns': ml_enhanced_patterns[-6:],
                'pattern_analytics': pattern_analytics,
                'uncertainty_analysis': uncertainty_analysis,
                'volatility_analysis': volatility_analysis,
                'equilibrium_state': self._assess_current_equilibrium(enhanced_data),
                'breakout_signals': self._generate_breakout_signals(ml_enhanced_patterns, enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Long-legged doji calculation failed: {str(e)}", e
            )
    
    def _enhance_data_with_uncertainty_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with uncertainty and volatility indicators"""
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
        df['total_shadow_ratio'] = df['upper_shadow_ratio'] + df['lower_shadow_ratio']
        df['shadow_balance'] = abs(df['upper_shadow_ratio'] - df['lower_shadow_ratio'])
        
        # Volatility measures
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['volatility_ratio'] = df['true_range'] / df['atr']
        df['volatility_percentile'] = df['atr'].rolling(50).rank(pct=True)
        
        # Price movement and momentum
        df['price_change'] = df['close'].pct_change()
        df['returns'] = df['price_change']
        df['abs_returns'] = abs(df['returns'])
        df['return_volatility'] = df['returns'].rolling(20).std()
        
        # Trend indicators for uncertainty analysis
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # MACD for momentum uncertainty
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['macd_uncertainty'] = abs(df['macd'] - df['macd_signal']) / (abs(df['macd']) + abs(df['macd_signal']) + 1e-10)
        
        # RSI for momentum context
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_neutrality'] = 1.0 - abs(df['rsi'] - 50) / 50  # Higher when RSI near 50
        
        # Bollinger Bands for volatility context
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_neutrality'] = 1.0 - abs(df['bb_position'] - 0.5) * 2  # Higher when price near middle
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Market structure uncertainty
        df['trend_consistency'] = self._calculate_trend_consistency(df)
        df['price_stability'] = self._calculate_price_stability(df)
        
        return df
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend consistency metric"""
        def trend_consistency_window(window_data):
            if len(window_data) < 10:
                return 0.5
            
            # Count directional changes
            price_changes = window_data['close'].diff().dropna()
            direction_changes = (price_changes * price_changes.shift(1) < 0).sum()
            
            # Normalize: fewer changes = higher consistency
            consistency = 1.0 - (direction_changes / len(price_changes))
            return max(0.0, min(1.0, consistency))
        
        return df.rolling(20).apply(trend_consistency_window, raw=False)['close']
    
    def _calculate_price_stability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price stability metric"""
        def stability_window(window_data):
            if len(window_data) < 10:
                return 0.5
            
            # Calculate coefficient of variation
            price_cv = window_data['close'].std() / window_data['close'].mean()
            
            # Convert to stability (inverse of variation)
            stability = 1.0 / (1.0 + price_cv * 10)  # Scale factor for sensitivity
            return max(0.0, min(1.0, stability))
        
        return df.rolling(20).apply(stability_window, raw=False)['close']
    
    def _detect_long_legged_doji_patterns(self, data: pd.DataFrame) -> List[LongLeggedDojiPattern]:
        """Detect long-legged doji patterns with uncertainty analysis"""
        patterns = []
        
        for i in range(25, len(data)):  # Need context for uncertainty analysis
            row = data.iloc[i]
            
            # Core long-legged doji criteria
            if not self._meets_long_legged_criteria(row):
                continue
            
            # Calculate market indecision score
            indecision_score = self._calculate_market_indecision_score(data, i)
            
            # Calculate directional uncertainty
            directional_uncertainty = self._calculate_directional_uncertainty(data, i)
            
            # Assess volatility context
            volatility_context = self._assess_volatility_context(row)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(
                row, indecision_score, directional_uncertainty, volatility_context
            )
            
            # Calculate initial breakout probability (will be enhanced by ML)
            breakout_prob = self._calculate_base_breakout_probability(
                row, indecision_score, volatility_context
            )
            
            if pattern_strength >= 0.6 and directional_uncertainty >= self.parameters['uncertainty_threshold']:
                pattern = LongLeggedDojiPattern(
                    timestamp=row.name,
                    strength=pattern_strength,
                    upper_shadow_ratio=row['upper_shadow_ratio'],
                    lower_shadow_ratio=row['lower_shadow_ratio'],
                    body_ratio=row['body_ratio'],
                    shadow_balance=row['shadow_balance'],
                    market_indecision_score=indecision_score,
                    volatility_context=volatility_context,
                    breakout_probability=breakout_prob,
                    directional_uncertainty=directional_uncertainty,
                    equilibrium_quality=0.0  # Will be calculated later
                )
                patterns.append(pattern)
        
        return patterns
    
    def _meets_long_legged_criteria(self, row: pd.Series) -> bool:
        """Check if candle meets long-legged doji criteria"""
        # 1. Very small body
        if row['body_ratio'] > self.parameters['max_body_ratio']:
            return False
        
        # 2. Long combined shadows
        if row['total_shadow_ratio'] < self.parameters['min_total_shadow']:
            return False
        
        # 3. Balanced shadows (not too imbalanced)
        if row['shadow_balance'] > self.parameters['max_shadow_imbalance']:
            return False
        
        # 4. Sufficient volatility
        if row['volatility_ratio'] < self.parameters['min_volatility_ratio']:
            return False
        
        # 5. Minimum total range
        if row['total_range'] < row['atr'] * 0.7:
            return False
        
        return True
    
    def _calculate_market_indecision_score(self, data: pd.DataFrame, index: int) -> float:
        """Calculate market indecision score using multiple factors"""
        row = data.iloc[index]
        context_data = data.iloc[max(0, index-10):index+1]
        
        indecision_factors = []
        
        # 1. RSI neutrality (weight: 0.25)
        rsi_neutrality = row['rsi_neutrality']
        indecision_factors.append(rsi_neutrality * 0.25)
        
        # 2. Bollinger Band position neutrality (weight: 0.2)
        bb_neutrality = row['bb_neutrality']
        indecision_factors.append(bb_neutrality * 0.2)
        
        # 3. MACD uncertainty (weight: 0.2)
        macd_uncertainty = row['macd_uncertainty']
        indecision_factors.append(macd_uncertainty * 0.2)
        
        # 4. Trend consistency (weight: 0.2)
        trend_consistency = row['trend_consistency']
        # Invert because low consistency indicates indecision
        trend_indecision = 1.0 - trend_consistency
        indecision_factors.append(trend_indecision * 0.2)
        
        # 5. Price stability (weight: 0.15)
        price_stability = row['price_stability']
        # High stability in a range indicates indecision
        indecision_factors.append(price_stability * 0.15)
        
        return min(sum(indecision_factors), 1.0)
    
    def _calculate_directional_uncertainty(self, data: pd.DataFrame, index: int) -> float:
        """Calculate directional uncertainty using information theory"""
        context_data = data.iloc[max(0, index-15):index+1]
        
        if len(context_data) < 10:
            return 0.5
        
        # Price movement directions
        price_changes = context_data['close'].diff().dropna()
        
        # Classify movements: up, down, neutral
        movement_classes = []
        for change in price_changes:
            if change > context_data['atr'].mean() * 0.1:
                movement_classes.append('up')
            elif change < -context_data['atr'].mean() * 0.1:
                movement_classes.append('down')
            else:
                movement_classes.append('neutral')
        
        # Calculate entropy of movement distribution
        if not movement_classes:
            return 0.5
        
        unique, counts = np.unique(movement_classes, return_counts=True)
        probabilities = counts / len(movement_classes)
        
        # Higher entropy = higher uncertainty
        movement_entropy = entropy(probabilities)
        max_entropy = np.log(3)  # Maximum entropy for 3 classes
        
        uncertainty = movement_entropy / max_entropy if max_entropy > 0 else 0.5
        
        return min(max(uncertainty, 0.0), 1.0)
    
    def _assess_volatility_context(self, row: pd.Series) -> str:
        """Assess volatility context for the pattern"""
        vol_percentile = row['volatility_percentile']
        
        if vol_percentile > 0.8:
            return "high_volatility"
        elif vol_percentile > 0.6:
            return "elevated_volatility"
        elif vol_percentile > 0.4:
            return "normal_volatility"
        elif vol_percentile > 0.2:
            return "low_volatility"
        else:
            return "very_low_volatility"
    
    def _calculate_pattern_strength(self, row: pd.Series, indecision_score: float, 
                                  uncertainty: float, volatility_context: str) -> float:
        """Calculate overall pattern strength"""
        strength_components = []
        
        # 1. Doji quality (30% weight)
        doji_quality = (
            (1 - row['body_ratio'] / self.parameters['max_body_ratio']) * 0.4 +
            (row['total_shadow_ratio'] / self.parameters['min_total_shadow']) * 0.4 +
            (1 - row['shadow_balance'] / self.parameters['max_shadow_imbalance']) * 0.2
        )
        strength_components.append(doji_quality * 0.3)
        
        # 2. Market indecision score (25% weight)
        strength_components.append(indecision_score * 0.25)
        
        # 3. Directional uncertainty (25% weight)
        strength_components.append(uncertainty * 0.25)
        
        # 4. Volatility context (15% weight)
        volatility_factor = {
            "very_low_volatility": 0.3,
            "low_volatility": 0.6,
            "normal_volatility": 1.0,
            "elevated_volatility": 0.9,
            "high_volatility": 0.7
        }.get(volatility_context, 0.5)
        strength_components.append(volatility_factor * 0.15)
        
        # 5. Shadow balance quality (5% weight)
        balance_quality = 1.0 - row['shadow_balance']  # Better balance = higher quality
        strength_components.append(balance_quality * 0.05)
        
        return min(sum(strength_components), 1.0)
    
    def _calculate_base_breakout_probability(self, row: pd.Series, indecision_score: float, 
                                           volatility_context: str) -> float:
        """Calculate base breakout probability"""
        # Base probability starts neutral
        probability = 0.5
        
        # Higher indecision increases breakout likelihood
        probability += indecision_score * 0.2
        
        # Volatility context affects breakout probability
        volatility_multiplier = {
            "very_low_volatility": 0.8,
            "low_volatility": 0.9,
            "normal_volatility": 1.0,
            "elevated_volatility": 1.1,
            "high_volatility": 1.2
        }.get(volatility_context, 1.0)
        
        probability *= volatility_multiplier
        
        # Long shadows suggest potential for breakout
        shadow_factor = min(row['total_shadow_ratio'], 1.0)
        probability += shadow_factor * 0.1
        
        return min(max(probability, 0.1), 0.9)
    
    def _apply_information_theory_analysis(self, patterns: List[LongLeggedDojiPattern], 
                                         data: pd.DataFrame) -> List[LongLeggedDojiPattern]:
        """Apply information theory analysis to patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            
            # Calculate market entropy around the pattern
            market_entropy = self._calculate_market_entropy(data, pattern_idx)
            
            # Adjust uncertainty based on entropy
            entropy_factor = market_entropy / np.log(10)  # Normalize
            pattern.directional_uncertainty = (
                pattern.directional_uncertainty * 0.7 + entropy_factor * 0.3
            )
            
            # Adjust strength based on information content
            pattern.strength = (pattern.strength * 0.8 + entropy_factor * 0.2)
        
        return patterns
    
    def _calculate_market_entropy(self, data: pd.DataFrame, index: int) -> float:
        """Calculate market entropy using price and volume distribution"""
        try:
            context_data = data.iloc[max(0, index-20):index+1]
            
            if len(context_data) < 10:
                return 1.0
            
            # Price return distribution
            returns = context_data['returns'].dropna()
            
            # Discretize returns into bins
            bins = 5
            hist, _ = np.histogram(returns, bins=bins)
            hist = hist + 1e-10  # Avoid log(0)
            probabilities = hist / hist.sum()
            
            # Calculate entropy
            return entropy(probabilities)
            
        except Exception:
            return 1.0  # Default entropy
    
    def _analyze_market_equilibrium(self, patterns: List[LongLeggedDojiPattern], 
                                  data: pd.DataFrame) -> List[LongLeggedDojiPattern]:
        """Analyze market equilibrium for each pattern"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            equilibrium_quality = self._calculate_equilibrium_quality(data, pattern_idx)
            pattern.equilibrium_quality = equilibrium_quality
            
            # Adjust pattern strength based on equilibrium quality
            pattern.strength = (pattern.strength * 0.9 + equilibrium_quality * 0.1)
        
        return patterns
    
    def _calculate_equilibrium_quality(self, data: pd.DataFrame, index: int) -> float:
        """Calculate market equilibrium quality"""
        try:
            context_data = data.iloc[max(0, index-15):index+1]
            
            if len(context_data) < 10:
                return 0.5
            
            quality_factors = []
            
            # 1. Price range stability
            price_ranges = context_data['total_range']
            range_stability = 1.0 - (price_ranges.std() / price_ranges.mean())
            quality_factors.append(max(0, min(1, range_stability)) * 0.3)
            
            # 2. Volume consistency
            volume_cv = context_data['volume'].std() / context_data['volume'].mean()
            volume_consistency = 1.0 / (1.0 + volume_cv)
            quality_factors.append(volume_consistency * 0.2)
            
            # 3. Trend neutrality
            trend_neutrality = context_data['rsi_neutrality'].mean()
            quality_factors.append(trend_neutrality * 0.3)
            
            # 4. Bollinger Band position stability
            bb_stability = 1.0 - context_data['bb_position'].std()
            quality_factors.append(max(0, min(1, bb_stability)) * 0.2)
            
            return sum(quality_factors)
            
        except Exception:
            return 0.5
    
    def _predict_breakout_probabilities(self, patterns: List[LongLeggedDojiPattern], 
                                      data: pd.DataFrame) -> List[LongLeggedDojiPattern]:
        """Predict breakout probabilities using ML"""
        if not patterns or not self.parameters['ml_breakout_prediction']:
            return patterns
        
        try:
            # Extract features for ML model
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_breakout_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 5:
                return patterns
            
            # Train model if needed
            if not self.is_ml_fitted:
                self._train_breakout_model(patterns, features)
            
            # Apply ML predictions if model is fitted
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                breakout_predictions = self.breakout_predictor.predict(features_scaled)
                
                # Update breakout probabilities
                for i, pattern in enumerate(patterns):
                    ml_probability = max(0.1, min(0.9, breakout_predictions[i]))
                    # Combine with base probability
                    pattern.breakout_probability = (
                        pattern.breakout_probability * 0.6 + ml_probability * 0.4
                    )
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML breakout prediction failed: {str(e)}")
            return patterns
    
    def _extract_breakout_features(self, data: pd.DataFrame, index: int, 
                                 pattern: LongLeggedDojiPattern) -> List[float]:
        """Extract features for breakout prediction ML model"""
        try:
            row = data.iloc[index]
            context_data = data.iloc[max(0, index-10):index+1]
            
            features = [
                pattern.strength,
                pattern.market_indecision_score,
                pattern.directional_uncertainty,
                pattern.equilibrium_quality,
                pattern.shadow_balance,
                pattern.total_shadow_ratio,
                row['volatility_ratio'],
                row['volatility_percentile'],
                row['rsi_neutrality'],
                row['bb_neutrality'],
                row['macd_uncertainty'],
                row['trend_consistency'],
                row['price_stability'],
                context_data['volume_ratio'].mean(),
                context_data['abs_returns'].mean(),
                row['bb_width']
            ]
            
            return features
            
        except Exception:
            return [0.5] * 16  # Default features
    
    def _train_breakout_model(self, patterns: List[LongLeggedDojiPattern], features: List[List[float]]):
        """Train ML model for breakout prediction"""
        try:
            # Create synthetic targets based on pattern characteristics
            targets = []
            for pattern in patterns:
                # Higher uncertainty and volatility suggest higher breakout probability
                target = (
                    pattern.directional_uncertainty * 0.4 +
                    pattern.market_indecision_score * 0.3 +
                    pattern.strength * 0.3
                )
                targets.append(target)
            
            if len(features) >= 10:
                features_scaled = self.scaler.fit_transform(features)
                self.breakout_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML breakout predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _analyze_market_uncertainty(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market uncertainty"""
        current = data.iloc[-1]
        recent_data = data.iloc[-20:]
        
        return {
            'current_uncertainty': current['rsi_neutrality'] * current['bb_neutrality'],
            'trend_consistency': current['trend_consistency'],
            'price_stability': current['price_stability'],
            'volatility_regime': self._assess_volatility_context(current),
            'market_entropy': self._calculate_market_entropy(data, len(data)-1),
            'uncertainty_trend': recent_data['rsi_neutrality'].mean(),
            'equilibrium_strength': self._calculate_equilibrium_quality(data, len(data)-1)
        }
    
    def _analyze_volatility_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current volatility regime"""
        current = data.iloc[-1]
        recent_data = data.iloc[-30:]
        
        return {
            'current_volatility_percentile': current['volatility_percentile'],
            'volatility_trend': recent_data['volatility_ratio'].mean(),
            'volatility_stability': 1.0 - recent_data['volatility_ratio'].std(),
            'regime': self._assess_volatility_context(current),
            'breakout_conducive': current['volatility_percentile'] > 0.6
        }
    
    def _assess_current_equilibrium(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current market equilibrium state"""
        current = data.iloc[-1]
        
        equilibrium_score = self._calculate_equilibrium_quality(data, len(data)-1)
        
        return {
            'equilibrium_score': equilibrium_score,
            'is_equilibrium': equilibrium_score > 0.7,
            'rsi_neutrality': current['rsi_neutrality'],
            'bb_neutrality': current['bb_neutrality'],
            'trend_consistency': current['trend_consistency'],
            'price_stability': current['price_stability']
        }
    
    def _generate_breakout_signals(self, patterns: List[LongLeggedDojiPattern], 
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Generate breakout signals based on patterns"""
        if not patterns:
            return {'breakout_probability': 0.0, 'signal_strength': 0.0}
        
        # Get recent high-quality patterns
        recent_patterns = [p for p in patterns[-3:] if p.strength > 0.7]
        
        if not recent_patterns:
            return {'breakout_probability': 0.0, 'signal_strength': 0.0}
        
        # Calculate aggregate metrics
        avg_breakout_prob = sum(p.breakout_probability for p in recent_patterns) / len(recent_patterns)
        avg_strength = sum(p.strength for p in recent_patterns) / len(recent_patterns)
        avg_uncertainty = sum(p.directional_uncertainty for p in recent_patterns) / len(recent_patterns)
        
        return {
            'breakout_probability': avg_breakout_prob,
            'signal_strength': avg_strength,
            'uncertainty_level': avg_uncertainty,
            'pattern_count': len(recent_patterns),
            'equilibrium_quality': sum(p.equilibrium_quality for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': recent_patterns[-1].timestamp
        }
    
    def _generate_pattern_analytics(self, patterns: List[LongLeggedDojiPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-20:]  # Last 20 patterns
        
        return {
            'total_patterns': len(recent_patterns),
            'average_strength': sum(p.strength for p in recent_patterns) / len(recent_patterns),
            'average_uncertainty': sum(p.directional_uncertainty for p in recent_patterns) / len(recent_patterns),
            'average_breakout_probability': sum(p.breakout_probability for p in recent_patterns) / len(recent_patterns),
            'average_equilibrium_quality': sum(p.equilibrium_quality for p in recent_patterns) / len(recent_patterns),
            'high_uncertainty_patterns': len([p for p in recent_patterns if p.directional_uncertainty > 0.8]),
            'high_breakout_probability_patterns': len([p for p in recent_patterns if p.breakout_probability > 0.7])
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on long-legged doji analysis"""
        current_pattern = value.get('current_pattern')
        breakout_signals = value.get('breakout_signals', {})
        uncertainty_analysis = value.get('uncertainty_analysis', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Long-legged doji primarily signals uncertainty and potential breakout
        # Don't generate strong directional signals unless there's additional confirmation
        
        breakout_probability = breakout_signals.get('breakout_probability', 0.0)
        uncertainty_level = current_pattern.directional_uncertainty
        
        # High uncertainty with high breakout probability suggests waiting for direction
        if uncertainty_level > 0.8 and breakout_probability > 0.7:
            confidence = current_pattern.strength * 0.6
            return SignalType.NEUTRAL, confidence
        
        # Moderate uncertainty suggests cautious positioning
        elif uncertainty_level > 0.7 and current_pattern.strength > 0.7:
            confidence = current_pattern.strength * 0.5
            return SignalType.HOLD, confidence
        
        # Lower uncertainty with decent pattern strength
        elif current_pattern.strength > 0.75:
            confidence = current_pattern.strength * 0.4
            return SignalType.NEUTRAL, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'long_legged_doji',
            'information_theory_analysis': self.parameters['information_theory_analysis'],
            'market_equilibrium_analysis': self.parameters['market_equilibrium_analysis'],
            'uncertainty_quantification_enabled': True,
            'breakout_prediction_enabled': self.parameters['ml_breakout_prediction']
        })
        return base_metadata