"""
Triangle Pattern Indicator

Triangle patterns are geometric chart formations that signal continuation or reversal depending on the type and breakout direction.

Pattern Recognition:
1. Ascending Triangle:
   - Horizontal resistance line (highs at same level)
   - Rising support line (higher lows)
   - Usually bullish continuation pattern
   - Breakout above resistance confirms pattern

2. Descending Triangle:
   - Horizontal support line (lows at same level)
   - Falling resistance line (lower highs)
   - Usually bearish continuation pattern
   - Breakdown below support confirms pattern

3. Symmetrical Triangle:
   - Converging trendlines
   - Higher lows and lower highs
   - Neutral pattern - direction depends on breakout
   - Volume typically decreases during formation

Trading Significance:
- Represents consolidation before major move
- Volume should increase on breakout
- Price target = base of triangle projected from breakout point
- False breakouts are common - need confirmation
- Pattern becomes invalid if price moves sideways too long
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class TrianglePatternSignal:
    """Signal data for Triangle pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'ascending_triangle', 'descending_triangle', 'symmetrical_triangle'
    formation_stage: str  # 'forming', 'breakout', 'confirmed'
    breakout_direction: str  # 'bullish', 'bearish', 'pending'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    support_line: Dict
    resistance_line: Dict
    convergence_point: Dict
    volume_confirmation: bool
    pattern_reliability: float
    time_to_breakout: Optional[int]


class TrianglePatternIndicator(BasePatternIndicator):
    """
    Triangle Pattern Recognition Indicator
    
    Identifies ascending, descending, and symmetrical triangle patterns
    with geometric analysis and breakout prediction.
    """
    
    def __init__(self, 
                 min_touches: int = 3,
                 min_pattern_periods: int = 10,
                 max_pattern_periods: int = 50,
                 line_tolerance: float = 0.002,
                 volume_confirmation_threshold: float = 1.3,
                 breakout_threshold: float = 0.001,
                 atr_periods: int = 14):
        """
        Initialize Triangle Pattern indicator
        
        Args:
            min_touches: Minimum touches required for trendline validation
            min_pattern_periods: Minimum periods for pattern formation
            max_pattern_periods: Maximum periods before pattern becomes invalid
            line_tolerance: Tolerance for trendline validation as % of price
            volume_confirmation_threshold: Volume multiplier for breakout confirmation
            breakout_threshold: Minimum breakout size as % of price
            atr_periods: Periods for ATR calculation
        """
        super().__init__()
        self.min_touches = min_touches
        self.min_pattern_periods = min_pattern_periods
        self.max_pattern_periods = max_pattern_periods
        self.line_tolerance = line_tolerance
        self.volume_confirmation_threshold = volume_confirmation_threshold
        self.breakout_threshold = breakout_threshold
        self.atr_periods = atr_periods
        self.signals: List[TrianglePatternSignal] = []
        
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
    
    def _find_swing_points(self, data: pd.DataFrame, lookback: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """Find swing highs and lows"""
        highs = data['high'].values
        lows = data['low'].values
        timestamps = data.index
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(data) - lookback):
            # Check for swing high
            is_swing_high = all(highs[i] >= highs[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'timestamp': timestamps[i],
                    'price': highs[i]
                })
            
            # Check for swing low
            is_swing_low = all(lows[i] <= lows[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'timestamp': timestamps[i],
                    'price': lows[i]
                })
        
        return swing_highs, swing_lows
    
    def _fit_trendline(self, points: List[Dict]) -> Dict:
        """Fit trendline to swing points using linear regression"""
        if len(points) < 2:
            return None
        
        x = np.array([p['index'] for p in points]).reshape(-1, 1)
        y = np.array([p['price'] for p in points])
        
        model = LinearRegression()
        model.fit(x, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(x, y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'points': points,
            'start_index': points[0]['index'],
            'end_index': points[-1]['index']
        }
    
    def _validate_trendline(self, trendline: Dict, data: pd.DataFrame) -> bool:
        """Validate trendline by checking how many points are close to it"""
        if not trendline or len(trendline['points']) < self.min_touches:
            return False
        
        touches = 0
        tolerance = np.mean(data['close']) * self.line_tolerance
        
        for point in trendline['points']:
            predicted_price = trendline['slope'] * point['index'] + trendline['intercept']
            if abs(point['price'] - predicted_price) <= tolerance:
                touches += 1
        
        return touches >= self.min_touches and trendline['r_squared'] > 0.5
    
    def _identify_triangle_type(self, support_line: Dict, resistance_line: Dict) -> str:
        """Identify triangle pattern type based on trendlines"""
        if not support_line or not resistance_line:
            return None
        
        support_slope = support_line['slope']
        resistance_slope = resistance_line['slope']
        
        # Ascending Triangle: flat resistance, rising support
        if abs(resistance_slope) < 0.0001 and support_slope > 0.0001:
            return 'ascending_triangle'
        
        # Descending Triangle: flat support, falling resistance
        if abs(support_slope) < 0.0001 and resistance_slope < -0.0001:
            return 'descending_triangle'
        
        # Symmetrical Triangle: converging lines
        if support_slope > 0 and resistance_slope < 0:
            return 'symmetrical_triangle'
        
        return None
    
    def _calculate_convergence_point(self, support_line: Dict, resistance_line: Dict) -> Dict:
        """Calculate where support and resistance lines converge"""
        if not support_line or not resistance_line:
            return None
        
        # Solve: support_slope * x + support_intercept = resistance_slope * x + resistance_intercept
        slope_diff = support_line['slope'] - resistance_line['slope']
        
        if abs(slope_diff) < 1e-10:  # Lines are parallel
            return None
        
        convergence_x = (resistance_line['intercept'] - support_line['intercept']) / slope_diff
        convergence_y = support_line['slope'] * convergence_x + support_line['intercept']
        
        return {
            'index': convergence_x,
            'price': convergence_y
        }
    
    def _check_breakout(self, data: pd.DataFrame, support_line: Dict, 
                       resistance_line: Dict, pattern_type: str) -> Tuple[bool, str]:
        """Check if price has broken out of triangle pattern"""
        latest_data = data.iloc[-1]
        latest_index = len(data) - 1
        
        if not support_line or not resistance_line:
            return False, None
        
        # Calculate current support and resistance levels
        current_support = support_line['slope'] * latest_index + support_line['intercept']
        current_resistance = resistance_line['slope'] * latest_index + resistance_line['intercept']
        
        breakout_threshold_price = latest_data['close'] * self.breakout_threshold
        
        # Check for bullish breakout
        if latest_data['high'] > current_resistance + breakout_threshold_price:
            return True, 'bullish'
        
        # Check for bearish breakout
        if latest_data['low'] < current_support - breakout_threshold_price:
            return True, 'bearish'
        
        return False, 'pending'
    
    def _check_volume_confirmation(self, data: pd.DataFrame, avg_volume: float) -> bool:
        """Check if breakout is confirmed by volume"""
        if 'volume' not in data.columns:
            return False
        
        recent_volume = data['volume'].iloc[-1]
        return recent_volume >= (avg_volume * self.volume_confirmation_threshold)
    
    def _calculate_price_targets(self, pattern_type: str, support_line: Dict,
                               resistance_line: Dict, breakout_direction: str,
                               current_price: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        if not support_line or not resistance_line:
            return current_price * 0.98, current_price * 1.02
        
        # Calculate triangle height at the widest point
        start_index = max(support_line['start_index'], resistance_line['start_index'])
        
        support_price_at_start = support_line['slope'] * start_index + support_line['intercept']
        resistance_price_at_start = resistance_line['slope'] * start_index + resistance_line['intercept']
        
        triangle_height = abs(resistance_price_at_start - support_price_at_start)
        
        if breakout_direction == 'bullish':
            stop_loss = current_price - triangle_height * 0.3
            take_profit = current_price + triangle_height
        else:
            stop_loss = current_price + triangle_height * 0.3
            take_profit = current_price - triangle_height
        
        return stop_loss, take_profit
    
    def _calculate_pattern_reliability(self, support_line: Dict, resistance_line: Dict,
                                     volume_confirmation: bool, pattern_periods: int) -> float:
        """Calculate overall pattern reliability score"""
        score = 0.0
        
        # Trendline quality (R-squared scores)
        if support_line:
            score += support_line['r_squared'] * 0.3
        if resistance_line:
            score += resistance_line['r_squared'] * 0.3
        
        # Volume confirmation
        if volume_confirmation:
            score += 0.2
        
        # Pattern duration (optimal range)
        duration_score = 1.0 - abs(pattern_periods - 25) / 25.0
        score += max(0, duration_score) * 0.2
        
        return max(0.0, min(1.0, score))
    
    def update(self, data: pd.DataFrame) -> Optional[TrianglePatternSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            TrianglePatternSignal if pattern detected, None otherwise
        """
        if len(data) < self.min_pattern_periods + self.atr_periods:
            return None
        
        # Calculate ATR for stop loss calculation
        atr_values = self._calculate_atr(data, self.atr_periods)
        current_atr = atr_values[-1]
        
        # Find swing points
        swing_highs, swing_lows = self._find_swing_points(data.tail(self.max_pattern_periods))
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        # Fit trendlines
        resistance_line = self._fit_trendline(swing_highs[-4:])  # Use recent highs
        support_line = self._fit_trendline(swing_lows[-4:])     # Use recent lows
        
        # Validate trendlines
        if not self._validate_trendline(resistance_line, data) or \
           not self._validate_trendline(support_line, data):
            return None
        
        # Identify triangle type
        pattern_type = self._identify_triangle_type(support_line, resistance_line)
        if not pattern_type:
            return None
        
        # Calculate convergence point
        convergence_point = self._calculate_convergence_point(support_line, resistance_line)
        
        # Check for breakout
        has_breakout, breakout_direction = self._check_breakout(
            data, support_line, resistance_line, pattern_type
        )
        
        # Determine formation stage
        if has_breakout:
            formation_stage = 'breakout'
        elif convergence_point and convergence_point['index'] - len(data) < 5:
            formation_stage = 'forming'  # Near convergence
        else:
            formation_stage = 'forming'
        
        # Volume confirmation
        volume_data = data['volume'].tail(20) if 'volume' in data.columns else pd.Series([1] * 20)
        avg_volume = volume_data.mean()
        volume_confirmation = self._check_volume_confirmation(data, avg_volume)
        
        # Calculate pattern parameters
        current_price = data['close'].iloc[-1]
        pattern_periods = len(data.tail(self.max_pattern_periods))
        
        stop_loss, take_profit = self._calculate_price_targets(
            pattern_type, support_line, resistance_line, breakout_direction, current_price
        )
        
        reliability = self._calculate_pattern_reliability(
            support_line, resistance_line, volume_confirmation, pattern_periods
        )
        
        # Determine signal type
        if has_breakout:
            signal_type = SignalType.BUY if breakout_direction == 'bullish' else SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Calculate time to potential breakout
        time_to_breakout = None
        if convergence_point and convergence_point['index'] > len(data):
            time_to_breakout = int(convergence_point['index'] - len(data))
        
        signal = TrianglePatternSignal(
            timestamp=data.index[-1],
            signal_type=signal_type,
            pattern_type=pattern_type,
            formation_stage=formation_stage,
            breakout_direction=breakout_direction,
            strength=reliability,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            support_line=support_line,
            resistance_line=resistance_line,
            convergence_point=convergence_point,
            volume_confirmation=volume_confirmation,
            pattern_reliability=reliability,
            time_to_breakout=time_to_breakout
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[TrianglePatternSignal]:
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
        
        pattern_counts = {}
        breakout_success = {}
        
        for signal in self.signals:
            pattern_type = signal.pattern_type
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            if signal.formation_stage == 'breakout':
                if pattern_type not in breakout_success:
                    breakout_success[pattern_type] = 0
                breakout_success[pattern_type] += 1
        
        return {
            'total_patterns': len(self.signals),
            'pattern_distribution': pattern_counts,
            'breakout_patterns': breakout_success,
            'avg_reliability': np.mean([s.pattern_reliability for s in self.signals]),
            'volume_confirmed_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'forming_patterns': len([s for s in self.signals if s.formation_stage == 'forming']),
            'breakout_patterns_total': len([s for s in self.signals if s.formation_stage == 'breakout'])
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()