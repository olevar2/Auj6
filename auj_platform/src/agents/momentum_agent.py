"""
Momentum Agent for the AUJ Platform.

This agent specializes in momentum analysis using various oscillators and indicators.
Extracted from the legacy StrategyExpert.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from .base_agent import BaseAgent, AnalysisResult, AgentState
from ..core.data_contracts import MarketConditions
from ..core.exceptions import AgentError
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)

class MomentumAgent(BaseAgent):
    """
    Momentum Agent - Specializes in momentum analysis.
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        """Initialize the Momentum Agent."""
        assigned_indicators = [
            "acceleration_deceleration_indicator",
            "awesome_oscillator_indicator",
            "bears_and_bulls_power_indicator",
            "chaikin_oscillator_indicator",
            "chande_momentum_oscillator_indicator",
            "commodity_channel_index_indicator",
            "coppock_curve_indicator",
            "demarker_indicator",
            "detrended_price_oscillator_indicator",
            "fisher_transform_indicator",
            "know_sure_thing_indicator",
            "macd_indicator",
            "market_cipher_b_indicator",
            "momentum_indicator",
            "money_flow_index_indicator",
            "percentage_price_oscillator_indicator",
            "qstick_indicator",
            "quantum_momentum_oracle_indicator",
            "quantum_phase_momentum_indicator",
            "rate_of_change_indicator",
            "relative_vigor_index_indicator",
            "rsi_indicator",
            "squeeze_momentum_indicator",
            "stochastic_rsi_indicator",
            "trix_indicator",
            "tsi_oscillator_indicator",
            "velocity_indicator",
            "wr_commodity_channel_index_indicator",
            "wr_degradation_indicator",
            "wr_signal_indicator"
        ]

        super().__init__(
            name="MomentumAgent",
            specialization="Momentum analysis and oscillator signals",
            assigned_indicators=assigned_indicators,
            config_manager=config_manager
        )

        # Load specific config
        self.rsi_oversold = self.config_manager.get_int('agents.momentum_agent.rsi_oversold', 30)
        self.rsi_overbought = self.config_manager.get_int('agents.momentum_agent.rsi_overbought', 70)

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform momentum analysis."""
        try:
            momentum_signals = self._analyze_momentum(indicator_results)
            
            # Determine decision based on momentum strength
            decision = "HOLD"
            confidence = 0.5
            
            strength = momentum_signals.get("momentum_strength", 0.0)
            direction = momentum_signals.get("momentum_direction", "NEUTRAL")
            
            if strength > 0.6:
                if direction == "BULLISH":
                    decision = "BUY"
                    confidence = strength
                elif direction == "BEARISH":
                    decision = "SELL"
                    confidence = strength

            reasoning = f"Momentum is {direction} with strength {strength:.2f}. Signals: {momentum_signals}"

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={"momentum_signals": momentum_signals},
                risk_assessment={}
            )

        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            raise AgentError(f"Momentum analysis failed: {e}")

    def _analyze_momentum(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        momentum_signals = {
            "rsi_signal": "NEUTRAL",
            "macd_signal": "NEUTRAL",
            "stochastic_signal": "NEUTRAL",
            "momentum_strength": 0.0,
            "momentum_direction": "NEUTRAL"
        }

        # RSI
        if "rsi_indicator" in indicator_results:
            rsi = indicator_results["rsi_indicator"].get("value", 50)
            if rsi < self.rsi_oversold:
                momentum_signals["rsi_signal"] = "OVERSOLD_BUY"
            elif rsi > self.rsi_overbought:
                momentum_signals["rsi_signal"] = "OVERBOUGHT_SELL"

        # MACD
        if "macd_indicator" in indicator_results:
            macd = indicator_results["macd_indicator"]
            if macd.get("histogram", 0) > 0:
                momentum_signals["macd_signal"] = "BULLISH"
            else:
                momentum_signals["macd_signal"] = "BEARISH"

        # Calculate strength
        bullish = 0
        bearish = 0
        total = 0
        
        for sig in momentum_signals.values():
            if isinstance(sig, str):
                if "BUY" in sig or "BULLISH" in sig:
                    bullish += 1
                    total += 1
                elif "SELL" in sig or "BEARISH" in sig:
                    bearish += 1
                    total += 1
        
        if total > 0:
            if bullish > bearish:
                momentum_signals["momentum_direction"] = "BULLISH"
                momentum_signals["momentum_strength"] = bullish / total
            elif bearish > bullish:
                momentum_signals["momentum_direction"] = "BEARISH"
                momentum_signals["momentum_strength"] = bearish / total

        return momentum_signals

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        return [k for k in self.assigned_indicators if k in indicator_results]

    def get_required_data_types(self) -> List[str]:
        return ["OHLCV"]

    def get_minimum_data_points(self) -> int:
        return 50
