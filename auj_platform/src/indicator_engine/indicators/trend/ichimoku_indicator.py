"""
Advanced Ichimoku Kinko Hyo Indicator with Complete Cloud Analysis

Features:
- All five Ichimoku lines (Tenkan, Kijun, Senkou A, Senkou B, Chikou)
- Cloud thickness and direction analysis
- Price position relative to cloud
- Breakout and trend change detection
- Multi-timeframe confirmation signals
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class IchimokuState(Enum):
    BULLISH_ABOVE_CLOUD = "bullish_above_cloud"
    BULLISH_IN_CLOUD = "bullish_in_cloud"
    BEARISH_BELOW_CLOUD = "bearish_below_cloud"
    BEARISH_IN_CLOUD = "bearish_in_cloud"
    CLOUD_BREAKOUT_UP = "cloud_breakout_up"
    CLOUD_BREAKOUT_DOWN = "cloud_breakout_down"
    NEUTRAL = "neutral"

@dataclass
class IchimokuResult:
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    cloud_top: float
    cloud_bottom: float
    cloud_thickness: float
    state: IchimokuState
    signal: SignalType
    confidence: float

class IchimokuIndicator(StandardIndicatorInterface):
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                 senkou_b_period: int = 52, displacement: int = 26):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.senkou_b_period + self.displacement:
                raise ValueError("Insufficient data")
            
            # Calculate Ichimoku components
            ichimoku_data = self._calculate_ichimoku(data)
            
            # Analyze state
            state = self._analyze_ichimoku_state(data, ichimoku_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, ichimoku_data, state)
            
            latest_result = IchimokuResult(
                tenkan_sen=ichimoku_data['tenkan_sen'].iloc[-1],
                kijun_sen=ichimoku_data['kijun_sen'].iloc[-1],
                senkou_span_a=ichimoku_data['senkou_span_a'].iloc[-1],
                senkou_span_b=ichimoku_data['senkou_span_b'].iloc[-1],
                chikou_span=ichimoku_data['chikou_span'].iloc[-1],
                cloud_top=ichimoku_data['cloud_top'].iloc[-1],
                cloud_bottom=ichimoku_data['cloud_bottom'].iloc[-1],
                cloud_thickness=ichimoku_data['cloud_thickness'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'tenkan_sen': ichimoku_data['tenkan_sen'].tolist(),
                    'kijun_sen': ichimoku_data['kijun_sen'].tolist(),
                    'senkou_span_a': ichimoku_data['senkou_span_a'].tolist(),
                    'senkou_span_b': ichimoku_data['senkou_span_b'].tolist(),
                    'chikou_span': ichimoku_data['chikou_span'].tolist(),
                    'cloud_top': ichimoku_data['cloud_top'].tolist(),
                    'cloud_bottom': ichimoku_data['cloud_bottom'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku: {e}")
            return self._get_default_result()
    
    def _calculate_ichimoku(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        df['tenkan_sen'] = (data['high'].rolling(self.tenkan_period).max() + 
                           data['low'].rolling(self.tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        df['kijun_sen'] = (data['high'].rolling(self.kijun_period).max() + 
                          data['low'].rolling(self.kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, displaced forward
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.displacement)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, displaced forward
        df['senkou_span_b'] = ((data['high'].rolling(self.senkou_b_period).max() + 
                               data['low'].rolling(self.senkou_b_period).min()) / 2).shift(self.displacement)
        
        # Chikou Span (Lagging Span): Close price displaced backward
        df['chikou_span'] = data['close'].shift(-self.displacement)
        
        # Cloud boundaries
        df['cloud_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        df['cloud_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        
        # Cloud thickness (relative to price)
        df['cloud_thickness'] = (df['cloud_top'] - df['cloud_bottom']) / data['close']
        
        return df
    
    def _analyze_ichimoku_state(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame) -> IchimokuState:
        current_price = data['close'].iloc[-1]
        cloud_top = ichimoku_data['cloud_top'].iloc[-1]
        cloud_bottom = ichimoku_data['cloud_bottom'].iloc[-1]
        
        # Handle NaN values
        if pd.isna(cloud_top) or pd.isna(cloud_bottom):
            return IchimokuState.NEUTRAL
        
        # Check previous position for breakout detection
        if len(data) > 1:
            prev_price = data['close'].iloc[-2]
            prev_cloud_top = ichimoku_data['cloud_top'].iloc[-2] if not pd.isna(ichimoku_data['cloud_top'].iloc[-2]) else cloud_top
            prev_cloud_bottom = ichimoku_data['cloud_bottom'].iloc[-2] if not pd.isna(ichimoku_data['cloud_bottom'].iloc[-2]) else cloud_bottom
            
            # Check for breakouts
            if prev_price <= prev_cloud_top and current_price > cloud_top:
                return IchimokuState.CLOUD_BREAKOUT_UP
            elif prev_price >= prev_cloud_bottom and current_price < cloud_bottom:
                return IchimokuState.CLOUD_BREAKOUT_DOWN
        
        # Determine position relative to cloud
        if current_price > cloud_top:
            return IchimokuState.BULLISH_ABOVE_CLOUD
        elif current_price < cloud_bottom:
            return IchimokuState.BEARISH_BELOW_CLOUD
        else:
            # Inside cloud - determine bias
            tenkan_sen = ichimoku_data['tenkan_sen'].iloc[-1]
            kijun_sen = ichimoku_data['kijun_sen'].iloc[-1]
            
            if not pd.isna(tenkan_sen) and not pd.isna(kijun_sen):
                if tenkan_sen > kijun_sen:
                    return IchimokuState.BULLISH_IN_CLOUD
                else:
                    return IchimokuState.BEARISH_IN_CLOUD
            else:
                return IchimokuState.NEUTRAL
    
    def _generate_signals(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame, 
                         state: IchimokuState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        tenkan_sen = ichimoku_data['tenkan_sen'].iloc[-1]
        kijun_sen = ichimoku_data['kijun_sen'].iloc[-1]
        cloud_thickness = ichimoku_data['cloud_thickness'].iloc[-1]
        
        # Base confidence on cloud thickness (thicker cloud = stronger signal)
        base_confidence = min(0.9, cloud_thickness * 50) if not pd.isna(cloud_thickness) else 0.5
        
        # Check Tenkan-Kijun cross for additional confirmation
        tk_cross_bullish = False
        tk_cross_bearish = False
        
        if not pd.isna(tenkan_sen) and not pd.isna(kijun_sen) and len(ichimoku_data) > 1:
            prev_tenkan = ichimoku_data['tenkan_sen'].iloc[-2]
            prev_kijun = ichimoku_data['kijun_sen'].iloc[-2]
            
            if not pd.isna(prev_tenkan) and not pd.isna(prev_kijun):
                tk_cross_bullish = prev_tenkan <= prev_kijun and tenkan_sen > kijun_sen
                tk_cross_bearish = prev_tenkan >= prev_kijun and tenkan_sen < kijun_sen
        
        # State-based signals with confidence
        if state == IchimokuState.CLOUD_BREAKOUT_UP:
            confidence = min(0.9, base_confidence + 0.3)
            return SignalType.BUY, confidence
        elif state == IchimokuState.CLOUD_BREAKOUT_DOWN:
            confidence = min(0.9, base_confidence + 0.3)
            return SignalType.SELL, confidence
        elif state == IchimokuState.BULLISH_ABOVE_CLOUD:
            confidence = base_confidence * 0.8
            if tk_cross_bullish:
                confidence = min(0.9, confidence + 0.2)
            return SignalType.BUY, confidence
        elif state == IchimokuState.BEARISH_BELOW_CLOUD:
            confidence = base_confidence * 0.8
            if tk_cross_bearish:
                confidence = min(0.9, confidence + 0.2)
            return SignalType.SELL, confidence
        elif state == IchimokuState.BULLISH_IN_CLOUD:
            return SignalType.BUY, base_confidence * 0.5
        elif state == IchimokuState.BEARISH_IN_CLOUD:
            return SignalType.SELL, base_confidence * 0.5
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = IchimokuResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                      IchimokuState.NEUTRAL, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'tenkan_period': self.tenkan_period,
            'kijun_period': self.kijun_period,
            'senkou_b_period': self.senkou_b_period,
            'displacement': self.displacement
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)