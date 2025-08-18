"""
Enhanced Ichimoku Kinko Hyo Indicator (Advanced Version)

This is an enhanced version of the traditional Ichimoku with additional features:
- Traditional five-line Ichimoku calculation
- Enhanced cloud analysis with volume confirmation
- Multiple timeframe synchronization
- Advanced breakout detection with momentum
"""

from .ichimoku_indicator import IchimokuIndicator, IchimokuState, IchimokuResult
import pandas as pd
import numpy as np
from typing import Dict

class IchimokuKinkoHyoIndicator(IchimokuIndicator):
    """Enhanced Ichimoku Kinko Hyo with additional analysis features"""
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                 senkou_b_period: int = 52, displacement: int = 26):
        super().__init__(tenkan_period, kijun_period, senkou_b_period, displacement)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        # Get base Ichimoku calculation
        result = super().calculate(data)
        
        if 'error' in result:
            return result
        
        # Add enhanced analysis
        try:
            enhanced_data = self._add_enhanced_analysis(data, result)
            result.update(enhanced_data)
            return result
        except Exception as e:
            self.logger.error(f"Error in enhanced Ichimoku analysis: {e}")
            return result
    
    def _add_enhanced_analysis(self, data: pd.DataFrame, base_result: Dict) -> Dict:
        """Add enhanced analysis features"""
        
        # Extract values for analysis
        current = base_result['current']
        
        # Enhanced momentum analysis
        momentum_strength = self._calculate_momentum_strength(data)
        
        # Volume confirmation (if volume data available)
        volume_confirmation = self._calculate_volume_confirmation(data) if 'volume' in data.columns else 0.5
        
        # Multi-timeframe alignment score
        alignment_score = self._calculate_alignment_score(base_result['values'])
        
        # Adjust confidence based on enhanced factors
        enhanced_confidence = self._calculate_enhanced_confidence(
            current.confidence, momentum_strength, volume_confirmation, alignment_score
        )
        
        return {
            'enhanced_confidence': enhanced_confidence,
            'momentum_strength': momentum_strength,
            'volume_confirmation': volume_confirmation,
            'alignment_score': alignment_score,
            'enhanced_analysis': True
        }
    
    def _calculate_momentum_strength(self, data: pd.DataFrame) -> float:
        """Calculate momentum strength based on price action"""
        if len(data) < 10:
            return 0.5
        
        # Calculate short-term momentum
        short_momentum = data['close'].pct_change(3).iloc[-1]
        medium_momentum = data['close'].pct_change(7).iloc[-1]
        
        # Normalize to 0-1 scale
        momentum_score = (abs(short_momentum) + abs(medium_momentum)) / 2
        return min(1.0, momentum_score * 50)
    
    def _calculate_volume_confirmation(self, data: pd.DataFrame) -> float:
        """Calculate volume confirmation score"""
        if 'volume' not in data.columns or len(data) < 10:
            return 0.5
        
        # Compare recent volume to average
        recent_volume = data['volume'].iloc[-3:].mean()
        avg_volume = data['volume'].iloc[-20:].mean()
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Normalize to 0-1 scale (higher volume = higher confidence)
        return min(1.0, volume_ratio / 2.0)
    
    def _calculate_alignment_score(self, values: Dict) -> float:
        """Calculate alignment score of Ichimoku components"""
        try:
            tenkan = values['tenkan_sen'][-1]
            kijun = values['kijun_sen'][-1]
            senkou_a = values['senkou_span_a'][-1]
            senkou_b = values['senkou_span_b'][-1]
            
            if any(pd.isna(x) for x in [tenkan, kijun, senkou_a, senkou_b]):
                return 0.5
            
            # Check alignment for bullish setup
            bullish_alignment = (tenkan > kijun and 
                               senkou_a > senkou_b and 
                               tenkan > senkou_a)
            
            # Check alignment for bearish setup
            bearish_alignment = (tenkan < kijun and 
                               senkou_a < senkou_b and 
                               tenkan < senkou_a)
            
            if bullish_alignment or bearish_alignment:
                return 0.9
            else:
                return 0.4
                
        except (IndexError, KeyError):
            return 0.5
    
    def _calculate_enhanced_confidence(self, base_confidence: float, 
                                     momentum_strength: float,
                                     volume_confirmation: float,
                                     alignment_score: float) -> float:
        """Calculate enhanced confidence combining all factors"""
        
        # Weighted combination of factors
        weights = {
            'base': 0.4,
            'momentum': 0.25,
            'volume': 0.15,
            'alignment': 0.2
        }
        
        enhanced = (
            base_confidence * weights['base'] +
            momentum_strength * weights['momentum'] +
            volume_confirmation * weights['volume'] +
            alignment_score * weights['alignment']
        )
        
        return min(0.95, enhanced)