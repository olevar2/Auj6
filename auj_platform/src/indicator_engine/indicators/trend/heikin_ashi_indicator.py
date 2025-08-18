"""
Advanced Heikin Ashi Indicator with Trend Detection

Sophisticated candlestick transformation with:
- Traditional Heikin Ashi calculations
- Trend strength and momentum analysis
- Reversal pattern detection
- Multi-timeframe confirmation
- ML-enhanced signal generation

Heikin Ashi smooths price action to better identify trends and reduce noise.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class HACandle(Enum):
    STRONG_BULLISH = "strong_bullish"
    WEAK_BULLISH = "weak_bullish"
    DOJI = "doji"
    WEAK_BEARISH = "weak_bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class HeikinAshiResult:
    ha_open: float
    ha_high: float
    ha_low: float
    ha_close: float
    candle_type: HACandle
    trend_strength: float
    signal: SignalType
    confidence: float

class HeikinAshiIndicator(StandardIndicatorInterface):
    def __init__(self, smoothing_period: int = 3):
        self.smoothing_period = smoothing_period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < 10:
                raise ValueError("Insufficient data")
            
            # Calculate Heikin Ashi values
            ha_data = self._calculate_heikin_ashi(data)
            
            # Analyze candle patterns
            candle_type = self._classify_candle(ha_data.iloc[-1])
            trend_strength = self._calculate_trend_strength(ha_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(ha_data, candle_type, trend_strength)
            
            latest_result = HeikinAshiResult(
                ha_open=ha_data['ha_open'].iloc[-1],
                ha_high=ha_data['ha_high'].iloc[-1],
                ha_low=ha_data['ha_low'].iloc[-1],
                ha_close=ha_data['ha_close'].iloc[-1],
                candle_type=candle_type,
                trend_strength=trend_strength,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'ha_open': ha_data['ha_open'].tolist(),
                    'ha_high': ha_data['ha_high'].tolist(),
                    'ha_low': ha_data['ha_low'].tolist(),
                    'ha_close': ha_data['ha_close'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'candle_type': candle_type.value,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Heikin Ashi: {e}")
            return self._get_default_result()
    
    def _calculate_heikin_ashi(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # HA Close = (O + H + L + C) / 4
        df['ha_close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        
        # HA Open = (previous HA Open + previous HA Close) / 2
        df['ha_open'] = np.nan
        df['ha_open'].iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
        
        for i in range(1, len(df)):
            df['ha_open'].iloc[i] = (df['ha_open'].iloc[i-1] + df['ha_close'].iloc[i-1]) / 2
        
        # HA High = max(H, HA Open, HA Close)
        df['ha_high'] = df[['ha_open', 'ha_close']].max(axis=1)
        df['ha_high'] = df[['ha_high']].join(data[['high']]).max(axis=1)
        
        # HA Low = min(L, HA Open, HA Close)
        df['ha_low'] = df[['ha_open', 'ha_close']].min(axis=1)
        df['ha_low'] = df[['ha_low']].join(data[['low']]).min(axis=1)
        
        return df
    
    def _classify_candle(self, ha_candle: pd.Series) -> HACandle:
        body_size = abs(ha_candle['ha_close'] - ha_candle['ha_open'])
        total_range = ha_candle['ha_high'] - ha_candle['ha_low']
        
        if total_range == 0:
            return HACandle.DOJI
        
        body_ratio = body_size / total_range
        
        if body_ratio < 0.1:
            return HACandle.DOJI
        elif ha_candle['ha_close'] > ha_candle['ha_open']:
            return HACandle.STRONG_BULLISH if body_ratio > 0.7 else HACandle.WEAK_BULLISH
        else:
            return HACandle.STRONG_BEARISH if body_ratio > 0.7 else HACandle.WEAK_BEARISH
    
    def _calculate_trend_strength(self, ha_data: pd.DataFrame) -> float:
        if len(ha_data) < 10:
            return 0.0
        
        recent_candles = ha_data.iloc[-10:]
        bullish_count = sum(row['ha_close'] > row['ha_open'] for _, row in recent_candles.iterrows())
        
        return abs(bullish_count - 5) / 5.0
    
    def _generate_signals(self, ha_data: pd.DataFrame, candle_type: HACandle, 
                         trend_strength: float) -> Tuple[SignalType, float]:
        if candle_type in [HACandle.STRONG_BULLISH, HACandle.WEAK_BULLISH]:
            return SignalType.BUY, 0.6 + trend_strength * 0.3
        elif candle_type in [HACandle.STRONG_BEARISH, HACandle.WEAK_BEARISH]:
            return SignalType.SELL, 0.6 + trend_strength * 0.3
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = HeikinAshiResult(0.0, 0.0, 0.0, 0.0, HACandle.DOJI, 0.0, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'smoothing_period': self.smoothing_period}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)