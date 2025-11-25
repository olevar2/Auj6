"""
Trend Agent for the AUJ Platform.

This agent specializes in trend identification and strength analysis.
Extracted from the legacy StrategyExpert.
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
    Trend Agent - Specializes in trend identification.
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        """Initialize the Trend Agent."""
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
            specialization="Trend identification and strength analysis",
            assigned_indicators=assigned_indicators,
            config_manager=config_manager
        )

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform trend analysis."""
        try:
            trend_signals = self._analyze_trend(indicator_results)
            
            decision = "HOLD"
            confidence = 0.5
            
            strength = trend_signals.get("overall_strength", 0.0)
            trend = trend_signals.get("overall_trend", "NEUTRAL")
            
            if strength > 0.5:
                if trend == "BULLISH":
                    decision = "BUY"
                    confidence = strength
                elif trend == "BEARISH":
                    decision = "SELL"
                    confidence = strength

            reasoning = f"Trend is {trend} with strength {strength:.2f}. Signals: {trend_signals}"

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={"trend_signals": trend_signals},
                risk_assessment={}
            )

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise AgentError(f"Trend analysis failed: {e}")

    def _analyze_trend(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend indicators."""
        trend_signals = {
            "adx_trend": "NEUTRAL",
            "supertrend_signal": "NEUTRAL",
            "overall_trend": "NEUTRAL",
            "overall_strength": 0.0
        }

        # ADX
        if "adx_indicator" in indicator_results:
            adx = indicator_results["adx_indicator"]
            if adx.get("adx", 0) > 25:
                if adx.get("plus_di", 0) > adx.get("minus_di", 0):
                    trend_signals["adx_trend"] = "UPTREND"
                else:
                    trend_signals["adx_trend"] = "DOWNTREND"
            trend_signals["overall_strength"] = min(adx.get("adx", 0) / 50.0, 1.0)

        # SuperTrend
        if "super_trend_indicator" in indicator_results:
            st = indicator_results["super_trend_indicator"]
            trend_signals["supertrend_signal"] = st.get("trend", "NEUTRAL")

        # Overall
        votes = []
        if "UP" in trend_signals["adx_trend"]: votes.append("BULLISH")
        if "DOWN" in trend_signals["adx_trend"]: votes.append("BEARISH")
        if "UP" in trend_signals["supertrend_signal"] or "BULLISH" in trend_signals["supertrend_signal"]: votes.append("BULLISH")
        if "DOWN" in trend_signals["supertrend_signal"] or "BEARISH" in trend_signals["supertrend_signal"]: votes.append("BEARISH")

        if votes.count("BULLISH") > votes.count("BEARISH"):
            trend_signals["overall_trend"] = "BULLISH"
        elif votes.count("BEARISH") > votes.count("BULLISH"):
            trend_signals["overall_trend"] = "BEARISH"

        return trend_signals

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        return [k for k in self.assigned_indicators if k in indicator_results]

    def get_required_data_types(self) -> List[str]:
        return ["OHLCV"]

    def get_minimum_data_points(self) -> int:
        return 50
