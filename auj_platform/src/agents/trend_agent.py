"""
Trend Agent for the AUJ Platform - ENHANCED with Ensemble Voting.

This agent specializes in trend identification and strength analysis.
ENHANCED VERSION: Uses all 29 assigned indicators with weighted ensemble voting
for maximum accuracy and robustness.

Improvements over original:
- Utilizes ALL 29 trend indicators (vs. 2 in original)
- Implements weighted ensemble voting system
- Better confidence scoring based on indicator consensus
- More robust trend strength calculation
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from .base_agent import BaseAgent, AnalysisResult
from ..core.data_contracts import MarketConditions
from ..core.exceptions import AgentError
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)

class TrendAgent(BaseAgent):
    """
    Trend Agent - Specializes in trend identification with ensemble voting.
    
    Uses all 29 trend indicators for maximum accuracy.
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        """Initialize the Enhanced Trend Agent."""
        assigned_indicators = [
            "adx_indicator",
            "alligator_indicator",
            "aroon_indicator",
            "aroon_oscillator_indicator",
            "chop_zone_indicator",
            "cloud_position_indicator",
            "directional_movement_index_indicator",
            "exponential_moving_average_indicator",
            "heikin_ashi_indicator",
            "hull_moving_average_indicator",
            "ichimoku_indicator",
            "ichimoku_kinko_hyo_indicator",
            "kaufman_adaptive_moving_average_indicator",
            "moving_average_indicator",
            "parabolic_sar_indicator",
            "renko_trend_indicator",
            "simple_moving_average_indicator",
            "sma_ema_indicator",
            "super_guppy_indicator",
            "super_trend_indicator",
            "trend_direction_indicator",
            "trend_following_system_indicator",
            "trend_strength_indicator",
            "triple_ema_indicator",
            "vortex_indicator",
            "wma_indicator",
            "zero_lag_ema_indicator",
            "zone_indicator"
        ]

        super().__init__(
            name="TrendAgent",
            specialization="Trend identification and strength analysis with ensemble voting",
            assigned_indicators=assigned_indicators,
            config_manager=config_manager
        )
        
        # Indicator weights (higher weight = more reliable)
        self.indicator_weights = {
            "adx_indicator": 1.0,
            "super_trend_indicator": 1.0,
            "ichimoku_indicator": 0.9,
            "trend_strength_indicator": 0.9,
            "parabolic_sar_indicator": 0.8,
            "alligator_indicator": 0.8,
            "aroon_indicator": 0.8,
            "vortex_indicator": 0.8,
            "directional_movement_index_indicator": 0.7,
            # Moving averages - medium weight
            "exponential_moving_average_indicator": 0.6,
            "simple_moving_average_indicator": 0.6,
            "hull_moving_average_indicator": 0.6,
            "wma_indicator": 0.5,
            # Others - standard weight
            "default": 0.5
        }

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform ensemble trend analysis using all 29 indicators."""
        try:
            # Analyze all trend indicators with ensemble voting
            trend_analysis = self._analyze_trend_ensemble(indicator_results)
            
            # Determine decision based on ensemble results
            decision = "HOLD"
            confidence = 0.5
            
            overall_trend = trend_analysis["overall_trend"]
            trend_strength = trend_analysis["trend_strength"]
            consensus_score = trend_analysis["consensus_score"]
            
            # Decision logic based on ensemble consensus
            if trend_strength > 0.4 and consensus_score > 0.5:  # Lowered threshold for better signal generation
                if overall_trend == "BULLISH":
                    decision = "BUY"
                    # Confidence is product of strength and consensus
                    confidence = min(trend_strength * consensus_score, 1.0)
                elif overall_trend == "BEARISH":
                    decision = "SELL"
                    confidence = min(trend_strength * consensus_score, 1.0)
            
            # Build comprehensive reasoning
            reasoning = self._build_reasoning(trend_analysis, decision, confidence)
            
            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "trend_analysis": trend_analysis,
                    "indicators_analyzed": trend_analysis["indicators_count"],
                    "ensemble_method": "weighted_voting"
                },
                risk_assessment={
                    "trend_strength": trend_strength,
                    "consensus_quality": consensus_score
                }
            )

        except Exception as e:
            logger.error(f"Enhanced trend analysis failed: {e}")
            raise AgentError(f"Enhanced trend analysis failed: {e}")

    def _analyze_trend_ensemble(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trend using weighted ensemble of all available indicators.
        
        Returns comprehensive trend analysis with:
        - overall_trend: BULLISH/BEARISH/NEUTRAL
        - trend_strength: 0.0 to 1.0
        - consensus_score: How well indicators agree (0.0 to 1.0)
        - individual_signals: Details of each indicator
        """
        
        bullish_votes = 0.0
        bearish_votes = 0.0
        neutral_votes = 0.0
        total_weight = 0.0
        
        individual_signals = {}
        indicators_analyzed = 0
        
        # Analyze each assigned indicator
        for indicator_name in self.assigned_indicators:
            if indicator_name not in indicator_results:
                continue
                
            indicators_analyzed += 1
            indicator_data = indicator_results[indicator_name]
            weight = self.indicator_weights.get(indicator_name, self.indicator_weights["default"])
            
            # Get signal from this indicator
            signal = self._interpret_indicator(indicator_name, indicator_data)
            individual_signals[indicator_name] = signal
            
            # Add weighted vote
            if signal == "BULLISH":
                bullish_votes += weight
            elif signal == "BEARISH":
                bearish_votes += weight
            else:  # NEUTRAL or unclear
                neutral_votes += weight
            
            total_weight += weight
        
        # Calculate ensemble results
        if total_weight > 0:
            bullish_ratio = bullish_votes / total_weight
            bearish_ratio = bearish_votes / total_weight
            neutral_ratio = neutral_votes / total_weight
        else:
            bullish_ratio = bearish_ratio = neutral_ratio = 0.33
        
        # Determine overall trend
        if bullish_ratio > bearish_ratio and bullish_ratio > neutral_ratio:
            overall_trend = "BULLISH"
            trend_strength = bullish_ratio
        elif bearish_ratio > bullish_ratio and bearish_ratio > neutral_ratio:
            overall_trend = "BEARISH"
            trend_strength = bearish_ratio
        else:
            overall_trend = "NEUTRAL"
            trend_strength = max(bullish_ratio, bearish_ratio, neutral_ratio)
        
        # Calculate consensus score (how strongly indicators agree)
        # High consensus = most indicators point same direction
        max_ratio = max(bullish_ratio, bearish_ratio, neutral_ratio)
        consensus_score = max_ratio  # 1.0 = perfect agreement, 0.33 = complete disagreement
        
        return {
            "overall_trend": overall_trend,
            "trend_strength": trend_strength,
            "consensus_score": consensus_score,
            "bullish_ratio": bullish_ratio,
            "bearish_ratio": bearish_ratio,
            "neutral_ratio": neutral_ratio,
            "individual_signals": individual_signals,
            "indicators_count": indicators_analyzed,
            "total_weight": total_weight
        }
    
    def _interpret_indicator(self, indicator_name: str, indicator_data: Any) -> str:
        """
        Interpret a single indicator's data to determine its trend signal.
        
        Returns: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        
        if not indicator_data or not isinstance(indicator_data, dict):
            return "NEUTRAL"
        
        # ADX - trend strength and direction
        if indicator_name == "adx_indicator":
            adx_value = indicator_data.get("adx", 0)
            plus_di = indicator_data.get("plus_di", 0)
            minus_di = indicator_data.get("minus_di", 0)
            if adx_value > 20:  # Strong trend
                return "BULLISH" if plus_di > minus_di else "BEARISH"
            return "NEUTRAL"
        
        # SuperTrend
        if indicator_name == "super_trend_indicator":
            trend = indicator_data.get("trend", "").upper()
            if "UP" in trend or "BULL" in trend:
                return "BULLISH"
            elif "DOWN" in trend or "BEAR" in trend:
                return "BEARISH"
            return "NEUTRAL"
        
        # Parabolic SAR
        if indicator_name == "parabolic_sar_indicator":
            signal = indicator_data.get("signal", "").upper()
            if "BUY" in signal or "BULL" in signal:
                return "BULLISH"
            elif "SELL" in signal or "BEAR" in signal:
                return "BEARISH"
            return "NEUTRAL"
        
        # Moving Averages (price vs MA)
        if "moving_average" in indicator_name or "ema" in indicator_name or "sma" in indicator_name:
            price_above_ma = indicator_data.get("price_above_ma", None)
            if price_above_ma is True:
                return "BULLISH"
            elif price_above_ma is False:
                return "BEARISH"
            # Try alternative naming
            signal = indicator_data.get("signal", "").upper()
            if "BULL" in signal or "UP" in signal:
                return "BULLISH"
            elif "BEAR" in signal or "DOWN" in signal:
                return "BEARISH"
            return "NEUTRAL"
        
        # Ichimoku
        if "ichimoku" in indicator_name:
            cloud_signal = indicator_data.get("cloud_signal", "").upper()
            if "BULL" in cloud_signal or "ABOVE" in cloud_signal:
                return "BULLISH"
            elif "BEAR" in cloud_signal or "BELOW" in cloud_signal:
                return "BEARISH"
            return "NEUTRAL"
        
        # Aroon
        if "aroon" in indicator_name:
            aroon_up = indicator_data.get("aroon_up", 50)
            aroon_down = indicator_data.get("aroon_down", 50)
            if aroon_up > 70 and aroon_up > aroon_down:
                return "BULLISH"
            elif aroon_down > 70 and aroon_down > aroon_up:
                return "BEARISH"
            return "NEUTRAL"
        
        # Vortex
        if indicator_name == "vortex_indicator":
            vi_plus = indicator_data.get("vi_plus", 1.0)
            vi_minus = indicator_data.get("vi_minus", 1.0)
            if vi_plus > vi_minus and vi_plus > 1.0:
                return "BULLISH"
            elif vi_minus > vi_plus and vi_minus > 1.0:
                return "BEARISH"
            return "NEUTRAL"
        
        # Generic interpretation - look for common keys
        signal = indicator_data.get("signal", "").upper()
        trend = indicator_data.get("trend", "").upper()
        direction = indicator_data.get("direction", "").upper()
        
        combined = signal + trend + direction
        
        if "BULL" in combined or "UP" in combined or "BUY" in combined:
            return "BULLISH"
        elif "BEAR" in combined or "DOWN" in combined or "SELL" in combined:
            return "BEARISH"
        
        return "NEUTRAL"
    
    def _build_reasoning(self, trend_analysis: Dict[str, Any], decision: str, confidence: float) -> str:
        """Build comprehensive reasoning string."""
        overall_trend = trend_analysis["overall_trend"]
        strength = trend_analysis["trend_strength"]
        consensus = trend_analysis["consensus_score"]
        indicators_count = trend_analysis["indicators_count"]
        
        bullish_pct = int(trend_analysis["bullish_ratio"] * 100)
        bearish_pct = int(trend_analysis["bearish_ratio"] * 100)
        
        reasoning = (
            f"Trend is {overall_trend} with strength {strength:.2f}. "
            f"Ensemble analysis of {indicators_count} trend indicators: "
            f"{bullish_pct}% Bullish, {bearish_pct}% Bearish. "
            f"Consensus score: {consensus:.2f}. "
            f"Decision: {decision} with {confidence:.2f} confidence."
        )
        
        return reasoning
    
    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually present in results."""
        return [k for k in self.assigned_indicators if k in indicator_results]

    def get_required_data_types(self) -> List[str]:
        return ["OHLCV"]

    def get_minimum_data_points(self) -> int:
        return 50
