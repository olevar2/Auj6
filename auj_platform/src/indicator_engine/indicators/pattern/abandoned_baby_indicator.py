"""
Abandoned Baby Pattern Indicator

The Abandoned Baby is a rare three-candle reversal pattern that signals strong trend reversals.

Pattern Recognition:
1. Bullish Abandoned Baby (appears at bottoms):
   - First candle: Long bearish candle in downtrend
   - Second candle: Doji that gaps down (below first candle's low)
   - Third candle: Bullish candle that gaps up (above doji's high)
   - The doji is "abandoned" with gaps on both sides

2. Bearish Abandoned Baby (appears at tops):
   - First candle: Long bullish candle in uptrend
   - Second candle: Doji that gaps up (above first candle's high)
   - Third candle: Bearish candle that gaps down (below doji's low)
   - The doji is "abandoned" with gaps on both sides

Trading Significance:
- Extremely rare and powerful reversal signal
- The gaps indicate strong momentum shifts
- Doji shows indecision at critical levels
- High reliability due to rarity and clear structure
- Often marks significant trend reversals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class AbandonedBabySignal:
    """Signal data for Abandoned Baby pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'bullish_abandoned_baby' or 'bearish_abandoned_baby'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    setup_candle: Dict
    doji_candle: Dict
    confirmation_candle: Dict
    gap_analysis: Dict
    volume_confirmation: bool
    trend_context: str
    reliability_score: float


class AbandonedBabyIndicator(BasePatternIndicator):
    """
    Abandoned Baby Pattern Recognition Indicator
    
    Identifies three-candle reversal patterns with gaps around a doji.
    """
    
    def __init__(self, 
                 min_gap_percentage: float = 0.001,
                 doji_body_threshold: float = 0.0005,
                 min_setup_body_ratio: float = 0.002,
                 volume_threshold: float = 1.5,
                 atr_periods: int = 14,
                 trend_periods: int = 20):
        """
        Initialize Abandoned Baby indicator
        
        Args:
            min_gap_percentage: Minimum gap size as percentage of price
            doji_body_threshold: Maximum doji body size as percentage of price
            min_setup_body_ratio: Minimum setup candle body ratio
            volume_threshold: Volume multiplier for confirmation
            atr_periods: Periods for ATR calculation
            trend_periods: Periods to determine trend context
        """
        super().__init__()
        self.min_gap_percentage = min_gap_percentage
        self.doji_body_threshold = doji_body_threshold
        self.min_setup_body_ratio = min_setup_body_ratio
        self.volume_threshold = volume_threshold
        self.atr_periods = atr_periods
        self.trend_periods = trend_periods
        self.signals: List[AbandonedBabySignal] = []
        
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
    
    def _is_doji(self, candle: Dict) -> bool:
        """Check if candle is a doji"""
        body_size = abs(candle['close'] - candle['open']) / candle['open']
        return body_size <= self.doji_body_threshold
    
    def _is_bullish_candle(self, candle: Dict) -> bool:
        """Check if candle is bullish"""
        return candle['close'] > candle['open']
    
    def _is_bearish_candle(self, candle: Dict) -> bool:
        """Check if candle is bearish"""
        return candle['close'] < candle['open']
    
    def _get_candle_body_size(self, candle: Dict) -> float:
        """Calculate candle body size as percentage of price"""
        return abs(candle['close'] - candle['open']) / candle['open']
    
    def _calculate_gap_size(self, candle1: Dict, candle2: Dict, gap_type: str) -> float:
        """
        Calculate gap size between two candles
        
        Args:
            candle1: First candle
            candle2: Second candle  
            gap_type: 'up' or 'down'
        """
        if gap_type == 'up':
            # Gap up: candle2's low should be above candle1's high
            gap = candle2['low'] - candle1['high']
        else:
            # Gap down: candle2's high should be below candle1's low
            gap = candle1['low'] - candle2['high']
        
        # Return as percentage of first candle's price
        return gap / candle1['close'] if candle1['close'] > 0 else 0
    
    def _has_gap_up(self, candle1: Dict, candle2: Dict) -> bool:
        """Check if candle2 gaps up from candle1"""
        gap_size = self._calculate_gap_size(candle1, candle2, 'up')
        return gap_size >= self.min_gap_percentage
    
    def _has_gap_down(self, candle1: Dict, candle2: Dict) -> bool:
        """Check if candle2 gaps down from candle1"""
        gap_size = self._calculate_gap_size(candle1, candle2, 'down')
        return gap_size >= self.min_gap_percentage
    
    def _determine_trend_context(self, data: pd.DataFrame) -> str:
        """Determine if we're in uptrend, downtrend, or sideways"""
        if len(data) < self.trend_periods:
            return 'unknown'
        
        closes = data['close'].tail(self.trend_periods)
        ema_short = self._calculate_ema(closes, 8).iloc[-1]
        ema_long = self._calculate_ema(closes, self.trend_periods).iloc[-1]
        
        # Price trend analysis
        price_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        
        if ema_short > ema_long and price_change > 0.02:
            return 'uptrend'
        elif ema_short < ema_long and price_change < -0.02:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _check_bullish_abandoned_baby(self, candles: List[Dict], 
                                    trend_context: str) -> Tuple[bool, float, Dict]:
        """
        Check for bullish abandoned baby pattern
        
        Returns:
            Tuple of (is_pattern, strength, gap_analysis)
        """
        if len(candles) != 3:
            return False, 0.0, {}
        
        setup_candle = candles[0]
        doji_candle = candles[1]
        confirmation_candle = candles[2]
        
        # Must be in downtrend or at least not in strong uptrend
        if trend_context == 'uptrend':
            return False, 0.0, {}
        
        # First candle should be bearish with good body size
        if not self._is_bearish_candle(setup_candle):
            return False, 0.0, {}
        
        setup_body_ratio = self._get_candle_body_size(setup_candle)
        if setup_body_ratio < self.min_setup_body_ratio:
            return False, 0.0, {}
        
        # Second candle should be doji
        if not self._is_doji(doji_candle):
            return False, 0.0, {}
        
        # Third candle should be bullish
        if not self._is_bullish_candle(confirmation_candle):
            return False, 0.0, {}
        
        # Check for gap down to doji
        if not self._has_gap_down(setup_candle, doji_candle):
            return False, 0.0, {}
        
        # Check for gap up from doji
        if not self._has_gap_up(doji_candle, confirmation_candle):
            return False, 0.0, {}
        
        # Calculate gap analysis
        gap_down_size = self._calculate_gap_size(setup_candle, doji_candle, 'down')
        gap_up_size = self._calculate_gap_size(doji_candle, confirmation_candle, 'up')
        
        gap_analysis = {
            'gap_down_size': gap_down_size,
            'gap_up_size': gap_up_size,
            'total_gap_distance': gap_down_size + gap_up_size,
            'gap_symmetry': min(gap_down_size, gap_up_size) / max(gap_down_size, gap_up_size)
        }
        
        # Calculate pattern strength
        strength = setup_body_ratio * 0.3  # Setup candle strength
        strength += gap_analysis['total_gap_distance'] * 50  # Gap significance
        strength += gap_analysis['gap_symmetry'] * 0.2  # Gap symmetry bonus
        
        # Bonus for trend context
        if trend_context == 'downtrend':
            strength *= 1.3
        
        # Bonus for confirmation candle strength
        confirmation_body_ratio = self._get_candle_body_size(confirmation_candle)
        strength += confirmation_body_ratio * 0.2
        
        return True, max(0.1, min(2.0, strength)), gap_analysis
    
    def _check_bearish_abandoned_baby(self, candles: List[Dict], 
                                    trend_context: str) -> Tuple[bool, float, Dict]:
        """
        Check for bearish abandoned baby pattern
        
        Returns:
            Tuple of (is_pattern, strength, gap_analysis)
        """
        if len(candles) != 3:
            return False, 0.0, {}
        
        setup_candle = candles[0]
        doji_candle = candles[1]
        confirmation_candle = candles[2]
        
        # Must be in uptrend or at least not in strong downtrend
        if trend_context == 'downtrend':
            return False, 0.0, {}
        
        # First candle should be bullish with good body size
        if not self._is_bullish_candle(setup_candle):
            return False, 0.0, {}
        
        setup_body_ratio = self._get_candle_body_size(setup_candle)
        if setup_body_ratio < self.min_setup_body_ratio:
            return False, 0.0, {}
        
        # Second candle should be doji
        if not self._is_doji(doji_candle):
            return False, 0.0, {}
        
        # Third candle should be bearish
        if not self._is_bearish_candle(confirmation_candle):
            return False, 0.0, {}
        
        # Check for gap up to doji
        if not self._has_gap_up(setup_candle, doji_candle):
            return False, 0.0, {}
        
        # Check for gap down from doji
        if not self._has_gap_down(doji_candle, confirmation_candle):
            return False, 0.0, {}
        
        # Calculate gap analysis
        gap_up_size = self._calculate_gap_size(setup_candle, doji_candle, 'up')
        gap_down_size = self._calculate_gap_size(doji_candle, confirmation_candle, 'down')
        
        gap_analysis = {
            'gap_up_size': gap_up_size,
            'gap_down_size': gap_down_size,
            'total_gap_distance': gap_up_size + gap_down_size,
            'gap_symmetry': min(gap_up_size, gap_down_size) / max(gap_up_size, gap_down_size)
        }
        
        # Calculate pattern strength
        strength = setup_body_ratio * 0.3  # Setup candle strength
        strength += gap_analysis['total_gap_distance'] * 50  # Gap significance
        strength += gap_analysis['gap_symmetry'] * 0.2  # Gap symmetry bonus
        
        # Bonus for trend context
        if trend_context == 'uptrend':
            strength *= 1.3
        
        # Bonus for confirmation candle strength
        confirmation_body_ratio = self._get_candle_body_size(confirmation_candle)
        strength += confirmation_body_ratio * 0.2
        
        return True, max(0.1, min(2.0, strength)), gap_analysis
    
    def _check_volume_confirmation(self, candles: List[Dict], avg_volume: float) -> bool:
        """Check volume confirmation for the pattern"""
        if len(candles) != 3:
            return False
        
        # Volume on confirmation candle should be elevated
        confirmation_volume = candles[2].get('volume', 0)
        volume_confirmed = confirmation_volume >= (avg_volume * self.volume_threshold)
        
        # Doji should have lower volume (indecision)
        doji_volume = candles[1].get('volume', 0)
        doji_volume_ok = doji_volume <= avg_volume
        
        return volume_confirmed and doji_volume_ok
    
    def _calculate_stop_loss(self, pattern_type: str, candles: List[Dict], atr: float) -> float:
        """Calculate stop loss level"""
        if pattern_type == 'bullish_abandoned_baby':
            # Stop below the doji's low
            return candles[1]['low'] - (atr * 1.5)
        else:  # bearish_abandoned_baby
            # Stop above the doji's high
            return candles[1]['high'] + (atr * 1.5)
    
    def _calculate_take_profit(self, pattern_type: str, entry_price: float, 
                             stop_loss: float, risk_reward: float = 3.0) -> float:
        """Calculate take profit level"""
        risk = abs(entry_price - stop_loss)
        
        if pattern_type == 'bullish_abandoned_baby':
            return entry_price + (risk * risk_reward)
        else:  # bearish_abandoned_baby
            return entry_price - (risk * risk_reward)
    
    def _calculate_reliability_score(self, pattern_type: str, strength: float,
                                   gap_analysis: Dict, volume_confirmation: bool,
                                   trend_context: str) -> float:
        """Calculate pattern reliability score"""
        score = strength * 0.4
        
        # Gap quality adds reliability
        gap_symmetry = gap_analysis.get('gap_symmetry', 0)
        total_gap = gap_analysis.get('total_gap_distance', 0)
        
        score += gap_symmetry * 0.2  # Symmetric gaps are better
        score += min(total_gap * 20, 0.2)  # Larger gaps are better (capped)
        
        # Volume confirmation adds reliability
        if volume_confirmation:
            score += 0.2
        
        # Trend context alignment
        if ((pattern_type == 'bullish_abandoned_baby' and trend_context == 'downtrend') or
            (pattern_type == 'bearish_abandoned_baby' and trend_context == 'uptrend')):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def update(self, data: pd.DataFrame) -> Optional[AbandonedBabySignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            AbandonedBabySignal if pattern detected, None otherwise
        """
        if len(data) < 3 + max(self.atr_periods, self.trend_periods):
            return None
        
        # Calculate ATR for stop loss calculation
        atr_values = self._calculate_atr(data, self.atr_periods)
        current_atr = atr_values[-1]
        
        # Determine trend context
        trend_context = self._determine_trend_context(data)
        
        # Get recent candles for pattern analysis
        recent_data = data.tail(3)
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
        
        # Calculate average volume for confirmation
        volume_data = data['volume'].tail(20) if 'volume' in data.columns else pd.Series([0] * 20)
        avg_volume = volume_data.mean()
        
        # Check for bullish abandoned baby
        is_bullish, bullish_strength, bullish_gaps = self._check_bullish_abandoned_baby(candles, trend_context)
        
        if is_bullish:
            volume_confirmation = self._check_volume_confirmation(candles, avg_volume)
            entry_price = candles[2]['close']
            stop_loss = self._calculate_stop_loss('bullish_abandoned_baby', candles, current_atr)
            take_profit = self._calculate_take_profit('bullish_abandoned_baby', entry_price, stop_loss)
            reliability = self._calculate_reliability_score('bullish_abandoned_baby', bullish_strength,
                                                          bullish_gaps, volume_confirmation, trend_context)
            
            signal = AbandonedBabySignal(
                timestamp=candles[2]['timestamp'],
                signal_type=SignalType.BUY,
                pattern_type='bullish_abandoned_baby',
                strength=bullish_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                setup_candle=candles[0],
                doji_candle=candles[1],
                confirmation_candle=candles[2],
                gap_analysis=bullish_gaps,
                volume_confirmation=volume_confirmation,
                trend_context=trend_context,
                reliability_score=reliability
            )
            
            self.signals.append(signal)
            return signal
        
        # Check for bearish abandoned baby
        is_bearish, bearish_strength, bearish_gaps = self._check_bearish_abandoned_baby(candles, trend_context)
        
        if is_bearish:
            volume_confirmation = self._check_volume_confirmation(candles, avg_volume)
            entry_price = candles[2]['close']
            stop_loss = self._calculate_stop_loss('bearish_abandoned_baby', candles, current_atr)
            take_profit = self._calculate_take_profit('bearish_abandoned_baby', entry_price, stop_loss)
            reliability = self._calculate_reliability_score('bearish_abandoned_baby', bearish_strength,
                                                          bearish_gaps, volume_confirmation, trend_context)
            
            signal = AbandonedBabySignal(
                timestamp=candles[2]['timestamp'],
                signal_type=SignalType.SELL,
                pattern_type='bearish_abandoned_baby',
                strength=bearish_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                setup_candle=candles[0],
                doji_candle=candles[1],
                confirmation_candle=candles[2],
                gap_analysis=bearish_gaps,
                volume_confirmation=volume_confirmation,
                trend_context=trend_context,
                reliability_score=reliability
            )
            
            self.signals.append(signal)
            return signal
        
        return None
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[AbandonedBabySignal]:
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
        
        bullish_signals = [s for s in self.signals if s.pattern_type == 'bullish_abandoned_baby']
        bearish_signals = [s for s in self.signals if s.pattern_type == 'bearish_abandoned_baby']
        
        return {
            'total_patterns': len(self.signals),
            'bullish_abandoned_babies': len(bullish_signals),
            'bearish_abandoned_babies': len(bearish_signals),
            'avg_bullish_strength': np.mean([s.strength for s in bullish_signals]) if bullish_signals else 0,
            'avg_bearish_strength': np.mean([s.strength for s in bearish_signals]) if bearish_signals else 0,
            'avg_reliability': np.mean([s.reliability_score for s in self.signals]),
            'volume_confirmed_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'avg_gap_distance': np.mean([s.gap_analysis['total_gap_distance'] for s in self.signals])
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()
