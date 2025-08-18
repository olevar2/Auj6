"""
Spinning Top Candle Pattern Indicator

Spinning Tops are candlesticks with small real bodies and long upper and lower shadows,
indicating market indecision and equilibrium between buyers and sellers.

Pattern Characteristics:
1. Small real body (usually < 1/3 of total range)
2. Long upper and lower shadows of similar length
3. Body can be bullish or bearish (color less important)
4. Indicates market indecision and balance
5. Often appears during consolidation or at reversal points

Trading Significance:
- Shows equilibrium between buyers and sellers
- Indicates market uncertainty and potential direction change
- More significant when appearing after strong trends
- Volume confirmation enhances reliability
- Often precedes trend reversals or continued consolidation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
from scipy.stats import entropy
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class SpinningTopSignal:
    """Signal data for Spinning Top pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_quality: float
    equilibrium_strength: float
    body_size_ratio: float
    shadow_balance: float
    market_indecision_level: float
    trend_context: str
    volume_profile: str
    reversal_probability: float
    continuation_probability: float
    confidence_score: float
    entry_price: float
    stop_loss: float
    take_profit: float


class SpinningTopIndicator(BasePatternIndicator):
    """
    Spinning Top Pattern Recognition Indicator
    
    Identifies spinning top candlestick patterns that indicate market indecision
    and potential trend changes with advanced statistical analysis.
    """
    
    def __init__(self, 
                 max_body_ratio: float = 0.33,
                 min_shadow_ratio: float = 0.4,
                 shadow_balance_tolerance: float = 0.3,
                 trend_strength_periods: int = 20,
                 volume_ma_periods: int = 20,
                 atr_periods: int = 14,
                 ml_training_periods: int = 100):
        """
        Initialize Spinning Top indicator
        
        Args:
            max_body_ratio: Maximum body size as ratio of total range
            min_shadow_ratio: Minimum combined shadow ratio
            shadow_balance_tolerance: Tolerance for shadow length balance
            trend_strength_periods: Periods for trend strength calculation
            volume_ma_periods: Periods for volume analysis
            atr_periods: Periods for ATR calculation
            ml_training_periods: Periods for ML model training
        """
        super().__init__()
        self.max_body_ratio = max_body_ratio
        self.min_shadow_ratio = min_shadow_ratio
        self.shadow_balance_tolerance = shadow_balance_tolerance
        self.trend_strength_periods = trend_strength_periods
        self.volume_ma_periods = volume_ma_periods
        self.atr_periods = atr_periods
        self.ml_training_periods = ml_training_periods
        self.signals: List[SpinningTopSignal] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize machine learning model for pattern significance prediction"""
        self.ml_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        )
    
    def _calculate_atr(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Average True Range"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr_list = []
        for i in range(1, len(data)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        tr_array = np.array(tr_list)
        atr = np.zeros(len(data))
        
        if len(tr_array) >= self.atr_periods:
            atr[self.atr_periods:] = pd.Series(tr_array).rolling(
                window=self.atr_periods
            ).mean().values[self.atr_periods-1:]
        
        return atr
    
    def _analyze_candle_structure(self, candle: pd.Series) -> Dict:
        """Analyze the structure of a candlestick"""
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        
        # Calculate candle components
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        # Avoid division by zero
        if total_range == 0:
            return {
                'body_ratio': 0,
                'upper_shadow_ratio': 0,
                'lower_shadow_ratio': 0,
                'shadow_balance': 0,
                'total_range': 0,
                'is_bullish': close_price >= open_price
            }
        
        # Calculate ratios
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        combined_shadow_ratio = (upper_shadow + lower_shadow) / total_range
        
        # Calculate shadow balance (how similar the shadows are)
        if upper_shadow + lower_shadow > 0:
            shadow_balance = 1 - abs(upper_shadow - lower_shadow) / (upper_shadow + lower_shadow)
        else:
            shadow_balance = 0
        
        return {
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'combined_shadow_ratio': combined_shadow_ratio,
            'shadow_balance': shadow_balance,
            'total_range': total_range,
            'is_bullish': close_price >= open_price,
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow
        }
    
    def _is_spinning_top(self, candle_structure: Dict) -> Tuple[bool, float]:
        """Determine if candle qualifies as spinning top pattern"""
        body_ratio = candle_structure['body_ratio']
        combined_shadow_ratio = candle_structure['combined_shadow_ratio']
        shadow_balance = candle_structure['shadow_balance']
        
        # Basic qualification criteria
        small_body = body_ratio <= self.max_body_ratio
        long_shadows = combined_shadow_ratio >= self.min_shadow_ratio
        balanced_shadows = shadow_balance >= (1 - self.shadow_balance_tolerance)
        
        is_pattern = small_body and long_shadows and balanced_shadows
        
        # Calculate pattern quality score
        quality = 0.0
        if is_pattern:
            # Reward smaller body
            body_quality = (self.max_body_ratio - body_ratio) / self.max_body_ratio
            
            # Reward longer shadows
            shadow_quality = min(combined_shadow_ratio / self.min_shadow_ratio, 2.0) / 2.0
            
            # Reward better shadow balance
            balance_quality = shadow_balance
            
            quality = (body_quality * 0.4 + shadow_quality * 0.3 + balance_quality * 0.3)
        
        return is_pattern, quality
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict:
        """Calculate trend strength and direction"""
        if len(data) < self.trend_strength_periods:
            return {'strength': 0, 'direction': 'sideways', 'momentum': 0}
        
        recent_data = data.tail(self.trend_strength_periods)
        close_prices = recent_data['close']
        
        # Linear regression for trend
        x = np.arange(len(close_prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_prices)
        
        # Calculate trend strength and direction
        r_squared = r_value ** 2
        strength = r_squared
        
        if r_squared < 0.1:
            direction = 'sideways'
        elif slope > 0:
            direction = 'uptrend'
        else:
            direction = 'downtrend'
        
        # Calculate momentum using rate of change
        momentum = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        
        return {
            'strength': strength,
            'direction': direction,
            'momentum': momentum,
            'slope': slope,
            'r_squared': r_squared
        }
    
    def _calculate_market_indecision_level(self, data: pd.DataFrame) -> float:
        """Calculate market indecision level using price action entropy"""
        if len(data) < 10:
            return 0.5
        
        recent_data = data.tail(10)
        
        # Calculate price changes
        price_changes = recent_data['close'].pct_change().dropna()
        
        # Discretize price changes into bins for entropy calculation
        if len(price_changes) == 0:
            return 0.5
        
        bins = np.linspace(price_changes.min(), price_changes.max(), 5)
        if len(np.unique(bins)) < 2:
            return 0.5
        
        digitized = np.digitize(price_changes, bins)
        
        # Calculate probability distribution
        unique_vals, counts = np.unique(digitized, return_counts=True)
        probabilities = counts / len(digitized)
        
        # Calculate entropy (higher entropy = more indecision)
        market_entropy = entropy(probabilities, base=2)
        
        # Normalize to 0-1 scale
        max_entropy = np.log2(len(unique_vals)) if len(unique_vals) > 1 else 1
        normalized_entropy = market_entropy / max_entropy if max_entropy > 0 else 0.5
        
        return normalized_entropy
    
    def _analyze_volume_profile(self, data: pd.DataFrame, index: int) -> str:
        """Analyze volume profile for pattern confirmation"""
        if 'volume' not in data.columns or len(data) < self.volume_ma_periods:
            return 'average'
        
        volumes = data['volume']
        vol_ma = volumes.rolling(window=self.volume_ma_periods).mean()
        
        if index < len(vol_ma) and vol_ma.iloc[index] > 0:
            current_volume = volumes.iloc[index]
            avg_volume = vol_ma.iloc[index]
            vol_ratio = current_volume / avg_volume
            
            if vol_ratio >= 1.5:
                return 'high'
            elif vol_ratio <= 0.7:
                return 'low'
            else:
                return 'average'
        
        return 'average'
    
    def _calculate_equilibrium_strength(self, candle_structure: Dict, 
                                      volume_profile: str) -> float:
        """Calculate the strength of buyer-seller equilibrium"""
        # Base equilibrium from candle structure
        shadow_balance = candle_structure['shadow_balance']
        body_ratio = candle_structure['body_ratio']
        
        # Smaller body indicates stronger equilibrium
        body_equilibrium = 1 - body_ratio
        
        # Base equilibrium strength
        equilibrium = (shadow_balance + body_equilibrium) / 2
        
        # Adjust based on volume
        volume_multiplier = {
            'high': 1.2,    # High volume increases equilibrium significance
            'average': 1.0,
            'low': 0.8      # Low volume decreases significance
        }
        
        equilibrium *= volume_multiplier.get(volume_profile, 1.0)
        
        return min(equilibrium, 1.0)
    
    def _calculate_reversal_probability(self, trend_strength: Dict, 
                                      pattern_quality: float,
                                      equilibrium_strength: float) -> float:
        """Calculate probability of trend reversal"""
        # Higher trend strength increases reversal potential
        trend_factor = trend_strength['strength']
        
        # Quality and equilibrium strength contribute to reversal probability
        pattern_factor = (pattern_quality + equilibrium_strength) / 2
        
        # Strong trends with high-quality spinning tops have higher reversal probability
        reversal_prob = trend_factor * pattern_factor
        
        # Adjust based on trend direction momentum
        momentum_adj = min(abs(trend_strength['momentum']), 0.1) / 0.1
        reversal_prob *= (1 + momentum_adj * 0.3)
        
        return min(reversal_prob, 1.0)
    
    def _calculate_continuation_probability(self, trend_strength: Dict,
                                         reversal_probability: float) -> float:
        """Calculate probability of trend continuation"""
        # If trend is weak (sideways), continuation is more likely than reversal
        if trend_strength['direction'] == 'sideways':
            return 0.7
        
        # Otherwise, continuation probability is inverse of reversal probability
        # but adjusted for trend strength
        base_continuation = 1 - reversal_probability
        
        # Weak trends are more likely to continue as consolidation
        if trend_strength['strength'] < 0.3:
            base_continuation *= 1.2
        
        return min(base_continuation, 1.0)
    
    def _prepare_ml_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Prepare features for ML model"""
        if index < 20:
            return np.array([])
        
        features = []
        
        # Current candle structure
        current_candle = data.iloc[index]
        candle_structure = self._analyze_candle_structure(current_candle)
        
        features.extend([
            candle_structure['body_ratio'],
            candle_structure['upper_shadow_ratio'],
            candle_structure['lower_shadow_ratio'],
            candle_structure['shadow_balance'],
            candle_structure['combined_shadow_ratio']
        ])
        
        # Trend and momentum features
        recent_data = data.iloc[index-self.trend_strength_periods+1:index+1]
        trend_strength = self._calculate_trend_strength(recent_data)
        
        features.extend([
            trend_strength['strength'],
            trend_strength['momentum'],
            abs(trend_strength['slope'])
        ])
        
        # Market indecision level
        indecision_level = self._calculate_market_indecision_level(
            data.iloc[index-9:index+1]
        )
        features.append(indecision_level)
        
        # Price volatility features
        recent_closes = data.iloc[index-9:index+1]['close']
        price_volatility = recent_closes.std() / recent_closes.mean()
        features.append(price_volatility)
        
        # Volume features
        if 'volume' in data.columns:
            recent_volumes = data.iloc[index-9:index+1]['volume']
            vol_ratio = recent_volumes.iloc[-1] / recent_volumes.mean()
            features.append(vol_ratio)
        else:
            features.append(1.0)
        
        return np.array(features)
    
    def _train_ml_model(self, data: pd.DataFrame):
        """Train ML model for pattern significance prediction"""
        if len(data) < self.ml_training_periods:
            return
        
        features_list = []
        labels_list = []
        
        for i in range(20, len(data) - 10):
            features = self._prepare_ml_features(data, i)
            if len(features) == 0:
                continue
            
            # Create label based on future price action
            future_data = data.iloc[i+1:i+11]  # Next 10 periods
            current_close = data.iloc[i]['close']
            
            # Check for significant price movement after spinning top
            max_move = max(
                abs(future_data['high'].max() - current_close),
                abs(current_close - future_data['low'].min())
            ) / current_close
            
            # Label as significant if price moves > 2% in next 10 periods
            label = 1 if max_move > 0.02 else 0
            
            features_list.append(features)
            labels_list.append(label)
        
        if len(features_list) < 30:
            return
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)
    
    def _get_ml_prediction(self, data: pd.DataFrame, index: int) -> float:
        """Get ML model prediction for pattern significance"""
        if self.ml_model is None:
            return 0.5
        
        features = self._prepare_ml_features(data, index)
        if len(features) == 0:
            return 0.5
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            return probabilities[1] if len(probabilities) > 1 else 0.5
        except:
            return 0.5
    
    def _determine_signal_type(self, trend_strength: Dict,
                             reversal_probability: float,
                             continuation_probability: float) -> SignalType:
        """Determine trading signal type"""
        if trend_strength['direction'] == 'sideways':
            return SignalType.HOLD
        
        # If reversal probability is high, signal opposite to trend
        if reversal_probability > 0.6:
            if trend_strength['direction'] == 'uptrend':
                return SignalType.SELL
            else:
                return SignalType.BUY
        
        # Otherwise, hold position
        return SignalType.HOLD
    
    def _calculate_trading_levels(self, data: pd.DataFrame,
                                current_price: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        atr_values = self._calculate_atr(data)
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0
        
        if current_atr == 0:
            # Fallback to percentage levels
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.02
        else:
            # ATR-based levels (conservative for indecision patterns)
            stop_loss = current_price - (current_atr * 1.0)
            take_profit = current_price + (current_atr * 1.5)
        
        return stop_loss, take_profit
    
    def update(self, data: pd.DataFrame) -> Optional[SpinningTopSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            SpinningTopSignal if pattern detected, None otherwise
        """
        if len(data) < max(self.atr_periods, self.trend_strength_periods, 20):
            return None
        
        # Train ML model if enough data
        if len(data) >= self.ml_training_periods and self.ml_model is not None:
            self._train_ml_model(data)
        
        # Analyze current candle
        current_index = len(data) - 1
        current_candle = data.iloc[current_index]
        candle_structure = self._analyze_candle_structure(current_candle)
        
        # Check if it's a spinning top
        is_pattern, pattern_quality = self._is_spinning_top(candle_structure)
        
        if not is_pattern:
            return None
        
        # Analyze market context
        trend_strength = self._calculate_trend_strength(data)
        volume_profile = self._analyze_volume_profile(data, current_index)
        indecision_level = self._calculate_market_indecision_level(data)
        
        # Calculate equilibrium and probabilities
        equilibrium_strength = self._calculate_equilibrium_strength(
            candle_structure, volume_profile
        )
        
        reversal_probability = self._calculate_reversal_probability(
            trend_strength, pattern_quality, equilibrium_strength
        )
        
        continuation_probability = self._calculate_continuation_probability(
            trend_strength, reversal_probability
        )
        
        # Get ML prediction
        ml_confidence = self._get_ml_prediction(data, current_index)
        
        # Calculate overall confidence
        confidence_score = (
            pattern_quality * 0.3 +
            equilibrium_strength * 0.25 +
            ml_confidence * 0.25 +
            (0.1 if volume_profile == 'high' else 0.05) +
            indecision_level * 0.1
        )
        
        # Determine signal
        signal_type = self._determine_signal_type(
            trend_strength, reversal_probability, continuation_probability
        )
        
        # Calculate trading levels
        current_price = current_candle['close']
        stop_loss, take_profit = self._calculate_trading_levels(data, current_price)
        
        signal = SpinningTopSignal(
            timestamp=data.index[-1],
            signal_type=signal_type,
            pattern_quality=pattern_quality,
            equilibrium_strength=equilibrium_strength,
            body_size_ratio=candle_structure['body_ratio'],
            shadow_balance=candle_structure['shadow_balance'],
            market_indecision_level=indecision_level,
            trend_context=trend_strength['direction'],
            volume_profile=volume_profile,
            reversal_probability=reversal_probability,
            continuation_probability=continuation_probability,
            confidence_score=confidence_score,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[SpinningTopSignal]:
        """Get signals within date range"""
        signals = self.signals
        
        if start_date:
            signals = [s for s in signals if s.timestamp >= start_date]
        
        if end_date:
            signals = [s for s in signals if s.timestamp <= end_date]
        
        return signals
    
    def get_pattern_statistics(self) -> Dict:
        """Get pattern detection statistics"""
        if not self.signals:
            return {}
        
        return {
            'total_signals': len(self.signals),
            'avg_pattern_quality': np.mean([s.pattern_quality for s in self.signals]),
            'avg_equilibrium_strength': np.mean([s.equilibrium_strength for s in self.signals]),
            'avg_reversal_probability': np.mean([s.reversal_probability for s in self.signals]),
            'avg_continuation_probability': np.mean([s.continuation_probability for s in self.signals]),
            'avg_confidence_score': np.mean([s.confidence_score for s in self.signals]),
            'trend_context_distribution': {
                context: len([s for s in self.signals if s.trend_context == context])
                for context in ['uptrend', 'downtrend', 'sideways']
            },
            'volume_profile_distribution': {
                profile: len([s for s in self.signals if s.volume_profile == profile])
                for profile in ['high', 'average', 'low']
            },
            'high_confidence_signals': len([s for s in self.signals if s.confidence_score > 0.7])
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()