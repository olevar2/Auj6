"""
Dark Cloud Cover Pattern Indicator

The Dark Cloud Cover is a two-candle bearish reversal pattern that appears at the top of uptrends.

Pattern Recognition:
1. First candle: Long bullish candle in an uptrend
2. Second candle: Bearish candle that:
   - Opens above the first candle's high (gap up)
   - Closes below the midpoint of the first candle's body
   - Penetrates at least 50% into the first candle's body

Trading Significance:
- Indicates potential bearish reversal at resistance levels
- The gap up shows initial bullish momentum
- The close below midpoint shows bears took control
- More reliable when appearing after extended uptrends
- Volume confirmation enhances reliability
- Deeper penetration indicates stronger bearish sentiment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class DarkCloudCoverSignal:
    """Signal data for Dark Cloud Cover pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'dark_cloud_cover'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    bullish_candle: Dict
    bearish_candle: Dict
    penetration_ratio: float
    gap_size: float
    volume_confirmation: bool
    trend_context: str
    reliability_score: float


class DarkCloudCoverIndicator(BasePatternIndicator):
    """
    Dark Cloud Cover Pattern Recognition Indicator
    
    Identifies two-candle bearish reversal patterns with gap up and deep penetration.
    """
    
    def __init__(self, 
                 min_penetration_ratio: float = 0.5,
                 min_gap_percentage: float = 0.0005,
                 min_body_ratio: float = 0.002,
                 volume_threshold: float = 1.3,
                 atr_periods: int = 14,
                 trend_periods: int = 15):
        """
        Initialize Dark Cloud Cover indicator
        
        Args:
            min_penetration_ratio: Minimum penetration into first candle (0.5 = 50%)
            min_gap_percentage: Minimum gap size as percentage of price
            min_body_ratio: Minimum candle body size as percentage of price
            volume_threshold: Volume multiplier for confirmation
            atr_periods: Periods for ATR calculation
            trend_periods: Periods to determine trend context
        """
        super().__init__()
        self.min_penetration_ratio = min_penetration_ratio
        self.min_gap_percentage = min_gap_percentage
        self.min_body_ratio = min_body_ratio
        self.volume_threshold = volume_threshold
        self.atr_periods = atr_periods
        self.trend_periods = trend_periods
        self.signals: List[DarkCloudCoverSignal] = []
        
    def _calculate_atr(self, data: pd.DataFrame, periods: int) -> np.ndarray:
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
        atr[periods:] = pd.Series(tr_array).rolling(window=periods-1).mean().values[periods-1:]
        
        return atr
    
    def _calculate_ema(self, data: pd.Series, periods: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=periods).mean()
    
    def _is_bullish_candle(self, candle: Dict) -> bool:
        """Check if candle is bullish"""
        return candle['close'] > candle['open']
    
    def _is_bearish_candle(self, candle: Dict) -> bool:
        """Check if candle is bearish"""
        return candle['close'] < candle['open']
    
    def _get_candle_body_size(self, candle: Dict) -> float:
        """Calculate candle body size as percentage of price"""
        return abs(candle['close'] - candle['open']) / candle['open']
    
    def _calculate_penetration_ratio(self, bullish_candle: Dict, bearish_candle: Dict) -> float:
        """
        Calculate how much the bearish candle penetrates into the bullish candle's body
        
        Returns:
            Penetration ratio (0.0 = no penetration, 1.0 = full penetration)
        """
        bullish_body_top = bullish_candle['close']
        bullish_body_bottom = bullish_candle['open']
        bearish_close = bearish_candle['close']
        
        bullish_body_size = bullish_body_top - bullish_body_bottom
        
        if bullish_body_size <= 0:
            return 0.0
        
        # How much the bearish candle penetrates from the top
        penetration = bullish_body_top - bearish_close
        
        if penetration <= 0:
            return 0.0
        
        return penetration / bullish_body_size
    
    def _calculate_gap_size(self, first_candle: Dict, second_candle: Dict) -> float:
        """Calculate gap up size as percentage of price"""
        gap = second_candle['open'] - first_candle['high']
        return gap / first_candle['close'] if first_candle['close'] > 0 else 0
    
    def _determine_trend_context(self, data: pd.DataFrame) -> str:
        """Determine if we're in uptrend, downtrend, or sideways"""
        if len(data) < self.trend_periods:
            return 'unknown'
        
        closes = data['close'].tail(self.trend_periods)
        ema_short = self._calculate_ema(closes, 6).iloc[-1]
        ema_long = self._calculate_ema(closes, self.trend_periods).iloc[-1]
        
        # Price trend analysis
        price_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        
        if ema_short > ema_long and price_change > 0.015:
            return 'uptrend'
        elif ema_short < ema_long and price_change < -0.015:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _check_dark_cloud_cover_pattern(self, candles: List[Dict], 
                                       trend_context: str) -> Tuple[bool, float, Dict]:
        """
        Check for Dark Cloud Cover pattern
        
        Returns:
            Tuple of (is_pattern, strength, pattern_data)
        """
        if len(candles) != 2:
            return False, 0.0, {}
        
        bullish_candle = candles[0]
        bearish_candle = candles[1]
        
        # Must be in uptrend context
        if trend_context != 'uptrend':
            return False, 0.0, {}
        
        # First candle must be bullish with sufficient body
        if not self._is_bullish_candle(bullish_candle):
            return False, 0.0, {}
        
        bullish_body_ratio = self._get_candle_body_size(bullish_candle)
        if bullish_body_ratio < self.min_body_ratio:
            return False, 0.0, {}
        
        # Second candle must be bearish
        if not self._is_bearish_candle(bearish_candle):
            return False, 0.0, {}
        
        # Check for gap up (bearish candle opens above bullish candle's high)
        gap_size = self._calculate_gap_size(bullish_candle, bearish_candle)
        if gap_size < self.min_gap_percentage:
            return False, 0.0, {}
        
        # Calculate penetration ratio
        penetration_ratio = self._calculate_penetration_ratio(bullish_candle, bearish_candle)
        if penetration_ratio < self.min_penetration_ratio:
            return False, 0.0, {}
        
        # Calculate pattern strength
        strength = bullish_body_ratio * 0.3  # First candle strength
        strength += penetration_ratio * 0.4  # Penetration depth
        strength += min(gap_size * 100, 0.2)  # Gap significance (capped)
        
        # Bonus for deeper penetration
        if penetration_ratio > 0.7:
            strength *= 1.2
        
        # Bonus for large bearish candle
        bearish_body_ratio = self._get_candle_body_size(bearish_candle)
        strength += bearish_body_ratio * 0.2
        
        # Check for minimal upper shadow on bearish candle (strong close near low)
        bearish_upper_shadow = bearish_candle['high'] - bearish_candle['open']
        bearish_total_range = bearish_candle['high'] - bearish_candle['low']
        
        if bearish_total_range > 0:
            shadow_ratio = bearish_upper_shadow / bearish_total_range
            strength *= (1.0 - shadow_ratio * 0.3)  # Penalty for large upper shadow
        
        pattern_data = {
            'penetration_ratio': penetration_ratio,
            'gap_size': gap_size,
            'bullish_body_ratio': bullish_body_ratio,
            'bearish_body_ratio': bearish_body_ratio
        }
        
        return True, max(0.1, min(2.0, strength)), pattern_data
    
    def _check_volume_confirmation(self, candles: List[Dict], avg_volume: float) -> bool:
        """Check volume confirmation for the pattern"""
        if len(candles) != 2:
            return False
        
        # Volume on bearish candle should be elevated
        bearish_volume = candles[1].get('volume', 0)
        return bearish_volume >= (avg_volume * self.volume_threshold)
    
    def _calculate_stop_loss(self, candles: List[Dict], atr: float) -> float:
        """Calculate stop loss level"""
        # Stop above the high of the bearish candle
        bearish_high = candles[1]['high']
        return bearish_high + (atr * 1.2)
    
    def _calculate_take_profit(self, entry_price: float, stop_loss: float,
                             penetration_ratio: float, risk_reward: float = 2.5) -> float:
        """Calculate take profit level"""
        risk = abs(entry_price - stop_loss)
        
        # Adjust risk-reward based on penetration depth
        adjusted_rr = risk_reward * (1 + penetration_ratio * 0.5)
        
        return entry_price - (risk * adjusted_rr)
    
    def _calculate_reliability_score(self, strength: float, penetration_ratio: float,
                                   gap_size: float, volume_confirmation: bool,
                                   trend_context: str) -> float:
        """Calculate pattern reliability score"""
        score = strength * 0.4
        
        # Penetration depth adds reliability
        score += penetration_ratio * 0.3
        
        # Gap size adds reliability (but capped)
        score += min(gap_size * 50, 0.1)
        
        # Volume confirmation adds reliability
        if volume_confirmation:
            score += 0.2
        
        # Strong uptrend context adds reliability
        if trend_context == 'uptrend':
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def update(self, data: pd.DataFrame) -> Optional[DarkCloudCoverSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DarkCloudCoverSignal if pattern detected, None otherwise
        """
        if len(data) < 2 + max(self.atr_periods, self.trend_periods):
            return None
        
        # Calculate ATR for stop loss calculation
        atr_values = self._calculate_atr(data, self.atr_periods)
        current_atr = atr_values[-1]
        
        # Determine trend context
        trend_context = self._determine_trend_context(data)
        
        # Get recent candles for pattern analysis
        recent_data = data.tail(2)
        candles = []
        
        for _, row in recent_data.iterrows():
            candle = {
                'timestamp': row.name,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row.get('volume', 0)
            }
            candles.append(candle)
        
        # Check for Dark Cloud Cover pattern
        is_pattern, strength, pattern_data = self._check_dark_cloud_cover_pattern(candles, trend_context)
        
        if not is_pattern:
            return None
        
        # Calculate average volume for confirmation
        volume_data = data['volume'].tail(20) if 'volume' in data.columns else pd.Series([0] * 20)
        avg_volume = volume_data.mean()
        
        volume_confirmation = self._check_volume_confirmation(candles, avg_volume)
        
        # Calculate trade parameters
        entry_price = candles[1]['close']
        stop_loss = self._calculate_stop_loss(candles, current_atr)
        take_profit = self._calculate_take_profit(entry_price, stop_loss, pattern_data['penetration_ratio'])
        
        # Calculate reliability score
        reliability = self._calculate_reliability_score(
            strength,
            pattern_data['penetration_ratio'],
            pattern_data['gap_size'],
            volume_confirmation,
            trend_context
        )
        
        signal = DarkCloudCoverSignal(
            timestamp=candles[1]['timestamp'],
            signal_type=SignalType.SELL,
            pattern_type='dark_cloud_cover',
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            bullish_candle=candles[0],
            bearish_candle=candles[1],
            penetration_ratio=pattern_data['penetration_ratio'],
            gap_size=pattern_data['gap_size'],
            volume_confirmation=volume_confirmation,
            trend_context=trend_context,
            reliability_score=reliability
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[DarkCloudCoverSignal]:
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
            'total_patterns': len(self.signals),
            'avg_strength': np.mean([s.strength for s in self.signals]),
            'avg_penetration_ratio': np.mean([s.penetration_ratio for s in self.signals]),
            'avg_gap_size': np.mean([s.gap_size for s in self.signals]),
            'avg_reliability': np.mean([s.reliability_score for s in self.signals]),
            'volume_confirmed_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'deep_penetration_rate': len([s for s in self.signals if s.penetration_ratio > 0.7]) / len(self.signals)
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()