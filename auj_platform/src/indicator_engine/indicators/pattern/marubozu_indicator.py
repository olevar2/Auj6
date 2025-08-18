"""
Marubozu Candle Pattern Indicator

Marubozu is a candlestick pattern with little to no shadows, indicating strong directional movement
and conviction from either buyers or sellers.

Pattern Types:
1. White/Bullish Marubozu: Long white body with no shadows (open = low, close = high)
2. Black/Bearish Marubozu: Long black body with no shadows (open = high, close = low)
3. Opening Marubozu: No shadow on opening side
4. Closing Marubozu: No shadow on closing side

Trading Significance:
- Strong directional conviction
- Continuation pattern in trending markets
- Reversal pattern when appearing after opposite trends
- High volume confirmation increases reliability
- Indicates potential breakout or strong momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class MarubozuSignal:
    """Signal data for Marubozu pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    marubozu_type: str  # 'bullish_full', 'bearish_full', 'opening_bull', 'opening_bear', 'closing_bull', 'closing_bear'
    pattern_strength: float
    body_length: float
    shadow_ratio: float
    momentum_strength: float
    volume_confirmation: bool
    trend_alignment: bool
    breakout_potential: float
    conviction_score: float
    confidence_level: float
    entry_price: float
    stop_loss: float
    take_profit: float


class MarubozuIndicator(BasePatternIndicator):
    """
    Marubozu Pattern Recognition Indicator
    
    Identifies various types of Marubozu candlestick patterns with sophisticated
    momentum analysis and conviction measurement.
    """
    
    def __init__(self, 
                 min_body_ratio: float = 0.7,
                 max_shadow_ratio: float = 0.05,
                 min_body_size_atr: float = 0.5,
                 volume_threshold: float = 1.2,
                 trend_periods: int = 20,
                 atr_periods: int = 14,
                 momentum_periods: int = 10,
                 ml_training_periods: int = 150):
        """
        Initialize Marubozu indicator
        
        Args:
            min_body_ratio: Minimum body size as ratio of total range
            max_shadow_ratio: Maximum allowed shadow ratio
            min_body_size_atr: Minimum body size as multiple of ATR
            volume_threshold: Volume confirmation threshold
            trend_periods: Periods for trend analysis
            atr_periods: Periods for ATR calculation
            momentum_periods: Periods for momentum calculation
            ml_training_periods: Periods for ML model training
        """
        super().__init__()
        self.min_body_ratio = min_body_ratio
        self.max_shadow_ratio = max_shadow_ratio
        self.min_body_size_atr = min_body_size_atr
        self.volume_threshold = volume_threshold
        self.trend_periods = trend_periods
        self.atr_periods = atr_periods
        self.momentum_periods = momentum_periods
        self.ml_training_periods = ml_training_periods
        self.signals: List[MarubozuSignal] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize machine learning model for pattern validation"""
        self.ml_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
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
        """Analyze detailed candle structure for Marubozu identification"""
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
                'total_shadow_ratio': 0,
                'body_size': 0,
                'total_range': 0,
                'is_bullish': close_price >= open_price,
                'opening_shadow_ratio': 0,
                'closing_shadow_ratio': 0
            }
        
        # Calculate ratios
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        total_shadow_ratio = (upper_shadow + lower_shadow) / total_range
        
        # Calculate opening and closing side shadows
        is_bullish = close_price >= open_price
        if is_bullish:
            opening_shadow_ratio = lower_shadow_ratio  # Lower shadow for bullish
            closing_shadow_ratio = upper_shadow_ratio  # Upper shadow for bullish
        else:
            opening_shadow_ratio = upper_shadow_ratio  # Upper shadow for bearish
            closing_shadow_ratio = lower_shadow_ratio  # Lower shadow for bearish
        
        return {
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'total_shadow_ratio': total_shadow_ratio,
            'opening_shadow_ratio': opening_shadow_ratio,
            'closing_shadow_ratio': closing_shadow_ratio,
            'body_size': body_size,
            'total_range': total_range,
            'is_bullish': is_bullish,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow
        }
    
    def _identify_marubozu_type(self, candle_structure: Dict, 
                               current_atr: float) -> Tuple[Optional[str], float]:
        """Identify specific type of Marubozu pattern"""
        body_ratio = candle_structure['body_ratio']
        total_shadow_ratio = candle_structure['total_shadow_ratio']
        upper_shadow_ratio = candle_structure['upper_shadow_ratio']
        lower_shadow_ratio = candle_structure['lower_shadow_ratio']
        opening_shadow_ratio = candle_structure['opening_shadow_ratio']
        closing_shadow_ratio = candle_structure['closing_shadow_ratio']
        body_size = candle_structure['body_size']
        is_bullish = candle_structure['is_bullish']
        
        # Check minimum body size requirement
        if current_atr > 0 and body_size < (current_atr * self.min_body_size_atr):
            return None, 0.0
        
        # Check minimum body ratio
        if body_ratio < self.min_body_ratio:
            return None, 0.0
        
        pattern_type = None
        strength = 0.0
        
        # Full Marubozu (no shadows on either side)
        if total_shadow_ratio <= self.max_shadow_ratio:
            if is_bullish:
                pattern_type = 'bullish_full'
            else:
                pattern_type = 'bearish_full'
            strength = 1.0 - total_shadow_ratio / self.max_shadow_ratio
        
        # Opening Marubozu (no shadow on opening side)
        elif opening_shadow_ratio <= self.max_shadow_ratio and closing_shadow_ratio > self.max_shadow_ratio:
            if is_bullish:
                pattern_type = 'opening_bull'
            else:
                pattern_type = 'opening_bear'
            strength = 0.8 * (1.0 - opening_shadow_ratio / self.max_shadow_ratio)
        
        # Closing Marubozu (no shadow on closing side)
        elif closing_shadow_ratio <= self.max_shadow_ratio and opening_shadow_ratio > self.max_shadow_ratio:
            if is_bullish:
                pattern_type = 'closing_bull'
            else:
                pattern_type = 'closing_bear'
            strength = 0.8 * (1.0 - closing_shadow_ratio / self.max_shadow_ratio)
        
        # Adjust strength based on body ratio
        if pattern_type:
            body_strength_bonus = (body_ratio - self.min_body_ratio) / (1.0 - self.min_body_ratio)
            strength = min(1.0, strength + body_strength_bonus * 0.2)
        
        return pattern_type, strength
    
    def _calculate_momentum_strength(self, data: pd.DataFrame) -> float:
        """Calculate momentum strength using multiple indicators"""
        if len(data) < self.momentum_periods:
            return 0.0
        
        recent_data = data.tail(self.momentum_periods)
        close_prices = recent_data['close']
        
        # Rate of Change (ROC)
        roc = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        
        # Price acceleration (second derivative)
        price_changes = close_prices.diff().dropna()
        if len(price_changes) > 1:
            acceleration = price_changes.diff().iloc[-1]
        else:
            acceleration = 0
        
        # Momentum oscillator
        if len(close_prices) >= 4:
            momentum = close_prices.iloc[-1] - close_prices.iloc[-4]
        else:
            momentum = 0
        
        # Combine momentum indicators
        momentum_score = np.tanh(abs(roc) * 10) * 0.5  # ROC component (0-0.5)
        momentum_score += np.tanh(abs(momentum / close_prices.iloc[-1]) * 20) * 0.3  # Momentum component (0-0.3)
        momentum_score += np.tanh(abs(acceleration / close_prices.std()) * 5) * 0.2  # Acceleration component (0-0.2)
        
        return momentum_score
    
    def _analyze_trend_alignment(self, data: pd.DataFrame, 
                               marubozu_type: str) -> bool:
        """Check if Marubozu aligns with current trend"""
        if len(data) < self.trend_periods:
            return True  # Assume alignment if insufficient data
        
        recent_data = data.tail(self.trend_periods)
        close_prices = recent_data['close']
        
        # Calculate trend using linear regression
        x = np.arange(len(close_prices))
        slope, _, r_value, _, _ = stats.linregress(x, close_prices)
        
        r_squared = r_value ** 2
        
        # Only consider strong trends
        if r_squared < 0.3:
            return True  # No clear trend, assume alignment
        
        is_uptrend = slope > 0
        is_bullish_marubozu = 'bull' in marubozu_type
        
        # Alignment check
        return (is_uptrend and is_bullish_marubozu) or (not is_uptrend and not is_bullish_marubozu)
    
    def _analyze_volume_confirmation(self, data: pd.DataFrame, 
                                   current_index: int) -> bool:
        """Analyze volume for Marubozu confirmation"""
        if 'volume' not in data.columns or len(data) < 20:
            return False
        
        volumes = data['volume']
        
        # Calculate volume moving average
        vol_ma = volumes.rolling(window=20).mean()
        
        if current_index < len(vol_ma) and vol_ma.iloc[current_index] > 0:
            current_volume = volumes.iloc[current_index]
            avg_volume = vol_ma.iloc[current_index]
            
            # Strong volume confirmation for Marubozu
            return current_volume >= (avg_volume * self.volume_threshold)
        
        return False
    
    def _calculate_breakout_potential(self, data: pd.DataFrame,
                                    marubozu_type: str) -> float:
        """Calculate breakout potential based on price action"""
        if len(data) < 20:
            return 0.5
        
        recent_data = data.tail(20)
        current_price = data.iloc[-1]['close']
        
        # Calculate support and resistance levels
        recent_highs = recent_data['high']
        recent_lows = recent_data['low']
        
        resistance_level = recent_highs.rolling(window=10).max().iloc[-1]
        support_level = recent_lows.rolling(window=10).min().iloc[-1]
        
        # Calculate distance to key levels
        if 'bull' in marubozu_type:
            # For bullish Marubozu, check distance to resistance
            distance_to_resistance = (resistance_level - current_price) / current_price
            breakout_potential = max(0, 1 - distance_to_resistance * 20)  # Closer = higher potential
        else:
            # For bearish Marubozu, check distance to support
            distance_to_support = (current_price - support_level) / current_price
            breakout_potential = max(0, 1 - distance_to_support * 20)  # Closer = higher potential
        
        return min(1.0, breakout_potential)
    
    def _calculate_conviction_score(self, pattern_strength: float,
                                  momentum_strength: float,
                                  volume_confirmation: bool,
                                  trend_alignment: bool,
                                  breakout_potential: float) -> float:
        """Calculate overall conviction score"""
        score = pattern_strength * 0.3  # Pattern quality
        score += momentum_strength * 0.25  # Momentum strength
        score += (0.2 if volume_confirmation else 0)  # Volume confirmation
        score += (0.15 if trend_alignment else 0)  # Trend alignment
        score += breakout_potential * 0.1  # Breakout potential
        
        return score
    
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
            candle_structure['total_shadow_ratio'],
            candle_structure['upper_shadow_ratio'],
            candle_structure['lower_shadow_ratio'],
            1 if candle_structure['is_bullish'] else 0
        ])
        
        # Momentum features
        momentum_strength = self._calculate_momentum_strength(
            data.iloc[index-self.momentum_periods+1:index+1]
        )
        features.append(momentum_strength)
        
        # Trend features
        recent_data = data.iloc[index-19:index+1]
        close_prices = recent_data['close']
        
        if len(close_prices) >= 20:
            x = np.arange(len(close_prices))
            slope, _, r_value, _, _ = stats.linregress(x, close_prices)
            features.extend([slope, r_value ** 2])
        else:
            features.extend([0, 0])
        
        # Volatility features
        atr_values = self._calculate_atr(recent_data)
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0
        features.append(current_atr / current_candle['close'] if current_candle['close'] > 0 else 0)
        
        # Volume features
        if 'volume' in data.columns:
            recent_volumes = recent_data['volume']
            vol_ratio = recent_volumes.iloc[-1] / recent_volumes.mean()
            features.append(vol_ratio)
        else:
            features.append(1.0)
        
        return np.array(features)
    
    def _train_ml_model(self, data: pd.DataFrame):
        """Train ML model for pattern validation"""
        if len(data) < self.ml_training_periods:
            return
        
        features_list = []
        labels_list = []
        
        for i in range(20, len(data) - 10):
            features = self._prepare_ml_features(data, i)
            if len(features) == 0:
                continue
            
            # Create label based on future price movement
            current_close = data.iloc[i]['close']
            candle_structure = self._analyze_candle_structure(data.iloc[i])
            
            if candle_structure['is_bullish']:
                # For bullish Marubozu, check for continued upward movement
                future_high = data.iloc[i+1:i+6]['high'].max()
                price_move = (future_high - current_close) / current_close
            else:
                # For bearish Marubozu, check for continued downward movement
                future_low = data.iloc[i+1:i+6]['low'].min()
                price_move = (current_close - future_low) / current_close
            
            # Label as successful if price moves > 1.5% in expected direction
            label = 1 if price_move > 0.015 else 0
            
            features_list.append(features)
            labels_list.append(label)
        
        if len(features_list) < 30:
            return
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)
    
    def _get_ml_confidence(self, data: pd.DataFrame, index: int) -> float:
        """Get ML model confidence for pattern success"""
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
    
    def _determine_signal_type(self, marubozu_type: str) -> SignalType:
        """Determine trading signal based on Marubozu type"""
        if 'bull' in marubozu_type:
            return SignalType.BUY
        elif 'bear' in marubozu_type:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_trading_levels(self, data: pd.DataFrame,
                                current_price: float,
                                marubozu_type: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        atr_values = self._calculate_atr(data)
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0
        
        if current_atr == 0:
            # Fallback to percentage levels
            if 'bull' in marubozu_type:
                stop_loss = current_price * 0.97
                take_profit = current_price * 1.04
            else:
                stop_loss = current_price * 1.03
                take_profit = current_price * 0.96
        else:
            # ATR-based levels
            if 'bull' in marubozu_type:
                stop_loss = current_price - (current_atr * 1.5)
                take_profit = current_price + (current_atr * 2.5)
            else:
                stop_loss = current_price + (current_atr * 1.5)
                take_profit = current_price - (current_atr * 2.5)
        
        return stop_loss, take_profit
    
    def update(self, data: pd.DataFrame) -> Optional[MarubozuSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            MarubozuSignal if pattern detected, None otherwise
        """
        if len(data) < max(self.atr_periods, self.trend_periods, 20):
            return None
        
        # Train ML model if enough data
        if len(data) >= self.ml_training_periods and self.ml_model is not None:
            self._train_ml_model(data)
        
        # Analyze current candle
        current_index = len(data) - 1
        current_candle = data.iloc[current_index]
        candle_structure = self._analyze_candle_structure(current_candle)
        
        # Calculate ATR for body size validation
        atr_values = self._calculate_atr(data)
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0
        
        # Identify Marubozu type
        marubozu_type, pattern_strength = self._identify_marubozu_type(
            candle_structure, current_atr
        )
        
        if not marubozu_type:
            return None
        
        # Calculate momentum and market context
        momentum_strength = self._calculate_momentum_strength(data)
        volume_confirmation = self._analyze_volume_confirmation(data, current_index)
        trend_alignment = self._analyze_trend_alignment(data, marubozu_type)
        breakout_potential = self._calculate_breakout_potential(data, marubozu_type)
        
        # Calculate conviction score
        conviction_score = self._calculate_conviction_score(
            pattern_strength, momentum_strength, volume_confirmation,
            trend_alignment, breakout_potential
        )
        
        # Get ML confidence
        ml_confidence = self._get_ml_confidence(data, current_index)
        
        # Calculate overall confidence
        confidence_level = (conviction_score * 0.7 + ml_confidence * 0.3)
        
        # Determine signal
        signal_type = self._determine_signal_type(marubozu_type)
        
        # Calculate trading levels
        current_price = current_candle['close']
        stop_loss, take_profit = self._calculate_trading_levels(
            data, current_price, marubozu_type
        )
        
        signal = MarubozuSignal(
            timestamp=data.index[-1],
            signal_type=signal_type,
            marubozu_type=marubozu_type,
            pattern_strength=pattern_strength,
            body_length=candle_structure['body_size'],
            shadow_ratio=candle_structure['total_shadow_ratio'],
            momentum_strength=momentum_strength,
            volume_confirmation=volume_confirmation,
            trend_alignment=trend_alignment,
            breakout_potential=breakout_potential,
            conviction_score=conviction_score,
            confidence_level=confidence_level,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[MarubozuSignal]:
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
        
        marubozu_types = [s.marubozu_type for s in self.signals]
        
        return {
            'total_signals': len(self.signals),
            'avg_pattern_strength': np.mean([s.pattern_strength for s in self.signals]),
            'avg_momentum_strength': np.mean([s.momentum_strength for s in self.signals]),
            'avg_conviction_score': np.mean([s.conviction_score for s in self.signals]),
            'avg_confidence_level': np.mean([s.confidence_level for s in self.signals]),
            'marubozu_type_distribution': {
                mtype: marubozu_types.count(mtype) / len(marubozu_types)
                for mtype in set(marubozu_types)
            },
            'volume_confirmation_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'trend_alignment_rate': len([s for s in self.signals if s.trend_alignment]) / len(self.signals),
            'high_conviction_signals': len([s for s in self.signals if s.conviction_score > 0.7]),
            'high_confidence_signals': len([s for s in self.signals if s.confidence_level > 0.7])
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()