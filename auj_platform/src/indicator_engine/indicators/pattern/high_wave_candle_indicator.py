"""
High Wave Candle Pattern Indicator

High Wave Candles are characterized by long upper and lower shadows with relatively small bodies,
indicating high volatility and market indecision. They suggest uncertainty and potential trend changes.

Pattern Characteristics:
1. Very long upper and lower shadows (wicks)
2. Small real body relative to the total range
3. Body can be bullish or bearish
4. High intraday volatility
5. Market indecision and uncertainty

Trading Significance:
- Indicates high volatility and market uncertainty
- Often appears at market turning points
- Can signal potential trend reversal or continuation
- Volume analysis helps confirm the pattern's significance
- Best used in conjunction with support/resistance levels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class HighWaveCandleSignal:
    """Signal data for High Wave Candle pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_strength: float
    volatility_level: str  # 'normal', 'high', 'extreme'
    body_ratio: float
    upper_shadow_ratio: float
    lower_shadow_ratio: float
    total_range: float
    volume_confirmation: bool
    trend_context: str  # 'uptrend', 'downtrend', 'sideways'
    market_state: str  # 'uncertainty', 'indecision', 'reversal_potential'
    confidence_score: float
    entry_price: float
    stop_loss: float
    take_profit: float


class HighWaveCandleIndicator(BasePatternIndicator):
    """
    High Wave Candle Pattern Recognition Indicator
    
    Identifies high wave candles that indicate market volatility and indecision,
    with advanced volatility analysis and machine learning confirmation.
    """
    
    def __init__(self, 
                 min_shadow_ratio: float = 0.6,
                 max_body_ratio: float = 0.3,
                 volatility_periods: int = 20,
                 atr_periods: int = 14,
                 volume_ma_periods: int = 20,
                 trend_periods: int = 50,
                 ml_lookback_periods: int = 100):
        """
        Initialize High Wave Candle indicator
        
        Args:
            min_shadow_ratio: Minimum ratio of shadows to total range
            max_body_ratio: Maximum ratio of body to total range
            volatility_periods: Periods for volatility calculation
            atr_periods: Periods for ATR calculation
            volume_ma_periods: Periods for volume moving average
            trend_periods: Periods for trend analysis
            ml_lookback_periods: Periods for ML model training
        """
        super().__init__()
        self.min_shadow_ratio = min_shadow_ratio
        self.max_body_ratio = max_body_ratio
        self.volatility_periods = volatility_periods
        self.atr_periods = atr_periods
        self.volume_ma_periods = volume_ma_periods
        self.trend_periods = trend_periods
        self.ml_lookback_periods = ml_lookback_periods
        self.signals: List[HighWaveCandleSignal] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize machine learning model for pattern confirmation"""
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
    
    def _calculate_atr(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Average True Range for volatility measurement"""
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
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive volatility metrics"""
        returns = data['close'].pct_change().dropna()
        
        # Historical volatility
        hist_vol = returns.rolling(window=self.volatility_periods).std() * np.sqrt(252)
        
        # Realized volatility (using high-low estimator)
        hl_vol = np.log(data['high'] / data['low'])
        realized_vol = hl_vol.rolling(window=self.volatility_periods).mean()
        
        # ATR-based volatility
        atr_values = self._calculate_atr(data)
        atr_vol = pd.Series(atr_values).rolling(window=self.volatility_periods).mean()
        
        # Garman-Klass volatility estimator
        gk_vol = self._calculate_garman_klass_volatility(data)
        
        return {
            'historical_volatility': hist_vol.iloc[-1] if len(hist_vol) > 0 else 0,
            'realized_volatility': realized_vol.iloc[-1] if len(realized_vol) > 0 else 0,
            'atr_volatility': atr_vol.iloc[-1] if len(atr_vol) > 0 else 0,
            'garman_klass_volatility': gk_vol.iloc[-1] if len(gk_vol) > 0 else 0,
            'current_atr': atr_values[-1] if len(atr_values) > 0 else 0
        }
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        
        gk = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
        return gk.rolling(window=self.volatility_periods).mean()
    
    def _analyze_candle_structure(self, candle: pd.Series) -> Dict:
        """Analyze the structure of a single candle"""
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
                'total_range': 0,
                'shadow_ratio': 0,
                'is_bullish': close_price >= open_price
            }
        
        # Calculate ratios
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        shadow_ratio = (upper_shadow + lower_shadow) / total_range
        
        return {
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'total_range': total_range,
            'shadow_ratio': shadow_ratio,
            'is_bullish': close_price >= open_price,
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow
        }
    
    def _is_high_wave_candle(self, candle_structure: Dict) -> Tuple[bool, float]:
        """Determine if candle qualifies as high wave pattern"""
        body_ratio = candle_structure['body_ratio']
        shadow_ratio = candle_structure['shadow_ratio']
        upper_shadow_ratio = candle_structure['upper_shadow_ratio']
        lower_shadow_ratio = candle_structure['lower_shadow_ratio']
        
        # Basic qualification criteria
        body_small_enough = body_ratio <= self.max_body_ratio
        shadows_long_enough = shadow_ratio >= self.min_shadow_ratio
        
        # Both shadows should be significant
        both_shadows_present = (upper_shadow_ratio >= 0.2 and 
                               lower_shadow_ratio >= 0.2)
        
        is_pattern = body_small_enough and shadows_long_enough and both_shadows_present
        
        # Calculate pattern strength
        strength = 0.0
        if is_pattern:
            # Reward smaller body ratio
            body_strength = (self.max_body_ratio - body_ratio) / self.max_body_ratio
            
            # Reward longer shadows
            shadow_strength = min(shadow_ratio / self.min_shadow_ratio, 2.0) / 2.0
            
            # Reward balance between upper and lower shadows
            shadow_balance = 1 - abs(upper_shadow_ratio - lower_shadow_ratio)
            
            strength = (body_strength * 0.4 + shadow_strength * 0.4 + 
                       shadow_balance * 0.2)
        
        return is_pattern, strength
    
    def _analyze_trend_context(self, data: pd.DataFrame) -> str:
        """Analyze the current trend context"""
        if len(data) < self.trend_periods:
            return 'sideways'
        
        recent_data = data.tail(self.trend_periods)
        close_prices = recent_data['close']
        
        # Calculate trend using linear regression
        x = np.arange(len(close_prices))
        slope, _, r_value, _, _ = stats.linregress(x, close_prices)
        
        # Determine trend based on slope and R-squared
        r_squared = r_value ** 2
        
        if r_squared < 0.1:  # Low correlation, sideways market
            return 'sideways'
        elif slope > 0:
            return 'uptrend'
        else:
            return 'downtrend'
    
    def _analyze_volume_confirmation(self, data: pd.DataFrame, 
                                   current_index: int) -> bool:
        """Analyze volume for pattern confirmation"""
        if 'volume' not in data.columns or len(data) < self.volume_ma_periods:
            return False
        
        volumes = data['volume']
        
        # Calculate volume moving average
        vol_ma = volumes.rolling(window=self.volume_ma_periods).mean()
        
        if current_index < len(vol_ma) and vol_ma.iloc[current_index] > 0:
            current_volume = volumes.iloc[current_index]
            avg_volume = vol_ma.iloc[current_index]
            
            # Volume should be above average for confirmation
            return current_volume >= avg_volume * 1.2
        
        return False
    
    def _prepare_ml_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Prepare features for ML model"""
        if index < 20:  # Need enough historical data
            return np.array([])
        
        features = []
        
        # Current candle structure
        current_candle = data.iloc[index]
        candle_structure = self._analyze_candle_structure(current_candle)
        
        features.extend([
            candle_structure['body_ratio'],
            candle_structure['upper_shadow_ratio'],
            candle_structure['lower_shadow_ratio'],
            candle_structure['shadow_ratio']
        ])
        
        # Volatility features
        recent_data = data.iloc[index-19:index+1]
        volatility_metrics = self._calculate_volatility_metrics(recent_data)
        
        features.extend([
            volatility_metrics['historical_volatility'],
            volatility_metrics['realized_volatility'],
            volatility_metrics['current_atr']
        ])
        
        # Price action features
        recent_closes = recent_data['close']
        price_change = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
        price_volatility = recent_closes.std() / recent_closes.mean()
        
        features.extend([price_change, price_volatility])
        
        # Volume features if available
        if 'volume' in data.columns:
            recent_volumes = recent_data['volume']
            vol_ratio = recent_volumes.iloc[-1] / recent_volumes.mean()
            features.append(vol_ratio)
        else:
            features.append(1.0)  # Default volume ratio
        
        return np.array(features)
    
    def _train_ml_model(self, data: pd.DataFrame):
        """Train ML model for pattern recognition"""
        if len(data) < self.ml_lookback_periods:
            return
        
        features_list = []
        labels_list = []
        
        for i in range(20, len(data) - 5):  # Leave some data for future validation
            features = self._prepare_ml_features(data, i)
            if len(features) == 0:
                continue
            
            # Create label based on future price movement and volatility
            future_data = data.iloc[i+1:i+6]  # Next 5 periods
            future_volatility = future_data['high'].max() - future_data['low'].min()
            current_atr = self._calculate_atr(data.iloc[i-13:i+1])[-1]
            
            # Label as significant if future volatility is high
            label = 1 if future_volatility > current_atr * 1.5 else 0
            
            features_list.append(features)
            labels_list.append(label)
        
        if len(features_list) < 20:  # Need minimum samples
            return
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)
    
    def _get_ml_prediction(self, data: pd.DataFrame, index: int) -> float:
        """Get ML model prediction for pattern significance"""
        if self.ml_model is None:
            return 0.5  # Default confidence
        
        features = self._prepare_ml_features(data, index)
        if len(features) == 0:
            return 0.5
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            return probabilities[1] if len(probabilities) > 1 else 0.5
        except:
            return 0.5
    
    def _classify_volatility_level(self, volatility_metrics: Dict) -> str:
        """Classify current volatility level"""
        current_vol = volatility_metrics['historical_volatility']
        
        if current_vol > 0.4:  # > 40% annualized
            return 'extreme'
        elif current_vol > 0.25:  # > 25% annualized
            return 'high'
        else:
            return 'normal'
    
    def _determine_market_state(self, pattern_strength: float, 
                              volatility_level: str, 
                              trend_context: str) -> str:
        """Determine market state based on pattern characteristics"""
        if pattern_strength > 0.8 and volatility_level == 'extreme':
            return 'reversal_potential'
        elif pattern_strength > 0.6:
            return 'indecision'
        else:
            return 'uncertainty'
    
    def _calculate_trading_levels(self, data: pd.DataFrame, 
                                current_price: float,
                                volatility_metrics: Dict) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        current_atr = volatility_metrics['current_atr']
        
        if current_atr == 0:
            # Fallback to percentage-based levels
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.02
        else:
            # ATR-based levels
            stop_loss = current_price - (current_atr * 1.5)
            take_profit = current_price + (current_atr * 2.0)
        
        return stop_loss, take_profit
    
    def update(self, data: pd.DataFrame) -> Optional[HighWaveCandleSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            HighWaveCandleSignal if pattern detected, None otherwise
        """
        if len(data) < max(self.atr_periods, self.volatility_periods, 20):
            return None
        
        # Train ML model if we have enough data
        if len(data) >= self.ml_lookback_periods and self.ml_model is not None:
            self._train_ml_model(data)
        
        # Analyze current candle
        current_index = len(data) - 1
        current_candle = data.iloc[current_index]
        candle_structure = self._analyze_candle_structure(current_candle)
        
        # Check if it's a high wave candle
        is_pattern, pattern_strength = self._is_high_wave_candle(candle_structure)
        
        if not is_pattern:
            return None
        
        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(data)
        
        # Analyze context
        trend_context = self._analyze_trend_context(data)
        volume_confirmation = self._analyze_volume_confirmation(data, current_index)
        
        # Get ML prediction
        ml_confidence = self._get_ml_prediction(data, current_index)
        
        # Classify volatility and market state
        volatility_level = self._classify_volatility_level(volatility_metrics)
        market_state = self._determine_market_state(
            pattern_strength, volatility_level, trend_context
        )
        
        # Calculate confidence score
        confidence_score = (pattern_strength * 0.4 + 
                          ml_confidence * 0.3 + 
                          (0.2 if volume_confirmation else 0) +
                          (0.1 if volatility_level in ['high', 'extreme'] else 0))
        
        # Determine signal type based on context
        if market_state == 'reversal_potential':
            if trend_context == 'uptrend':
                signal_type = SignalType.SELL
            elif trend_context == 'downtrend':
                signal_type = SignalType.BUY
            else:
                signal_type = SignalType.HOLD
        else:
            signal_type = SignalType.HOLD
        
        # Calculate trading levels
        current_price = current_candle['close']
        stop_loss, take_profit = self._calculate_trading_levels(
            data, current_price, volatility_metrics
        )
        
        signal = HighWaveCandleSignal(
            timestamp=data.index[-1],
            signal_type=signal_type,
            pattern_strength=pattern_strength,
            volatility_level=volatility_level,
            body_ratio=candle_structure['body_ratio'],
            upper_shadow_ratio=candle_structure['upper_shadow_ratio'],
            lower_shadow_ratio=candle_structure['lower_shadow_ratio'],
            total_range=candle_structure['total_range'],
            volume_confirmation=volume_confirmation,
            trend_context=trend_context,
            market_state=market_state,
            confidence_score=confidence_score,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[HighWaveCandleSignal]:
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
        
        volatility_levels = [s.volatility_level for s in self.signals]
        market_states = [s.market_state for s in self.signals]
        
        return {
            'total_signals': len(self.signals),
            'avg_pattern_strength': np.mean([s.pattern_strength for s in self.signals]),
            'avg_confidence_score': np.mean([s.confidence_score for s in self.signals]),
            'volatility_distribution': {
                level: volatility_levels.count(level) / len(volatility_levels)
                for level in ['normal', 'high', 'extreme']
            },
            'market_state_distribution': {
                state: market_states.count(state) / len(market_states)
                for state in ['uncertainty', 'indecision', 'reversal_potential']
            },
            'volume_confirmation_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'reversal_signals': len([s for s in self.signals if s.market_state == 'reversal_potential'])
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()