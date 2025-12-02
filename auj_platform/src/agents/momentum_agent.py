"""
Momentum Agent for the AUJ Platform - ENHANCED with Ensemble Voting.

This agent specializes in momentum analysis using various oscillators and indicators.
ENHANCED VERSION: Uses all 30 assigned momentum indicators with weighted ensemble voting
for maximum accuracy in detecting momentum shifts and overbought/oversold conditions.

Improvements over original:
- Utilizes ALL 30 momentum indicators (vs. 2 in original)
- Implements weighted ensemble voting system
- Better momentum strength calculation
- Improved overbought/oversold detection
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
    Momentum Agent - Specializes in momentum analysis with ensemble voting.
    
    Uses all 30 momentum indicators for comprehensive momentum assessment.
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        """Initialize the Enhanced Momentum Agent."""
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
            specialization="Momentum analysis and oscillator signals with ensemble voting",
            assigned_indicators=assigned_indicators,
            config_manager=config_manager
        )

        # Configuration thresholds
        self.rsi_oversold = self.config_manager.get_int('agents.momentum_agent.rsi_oversold', 30)
        self.rsi_overbought = self.config_manager.get_int('agents.momentum_agent.rsi_overbought', 70)
        
        # Indicator weights (higher = more reliable)
        self.indicator_weights = {
            "rsi_indicator": 1.0,
            "macd_indicator": 1.0,
            "stochastic_rsi_indicator": 0.9,
            "commodity_channel_index_indicator": 0.9,
            "money_flow_index_indicator": 0.8,
            "awesome_oscillator_indicator": 0.8,
            "fisher_transform_indicator": 0.8,
            "rate_of_change_indicator": 0.7,
            "momentum_indicator": 0.7,
            "relative_vigor_index_indicator": 0.7,
            "default": 0.5
        }

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform ensemble momentum analysis using all 30 indicators."""
        try:
            # Analyze all momentum indicators with ensemble voting
            momentum_analysis = self._analyze_momentum_ensemble(indicator_results)
            
            # Determine decision based on ensemble results
            decision = "HOLD"
            confidence = 0.5
            
            momentum_direction = momentum_analysis["momentum_direction"]
            momentum_strength = momentum_analysis["momentum_strength"]
            consensus_score = momentum_analysis["consensus_score"]
            extreme_condition = momentum_analysis["extreme_condition"]
            
            # Decision logic
            if momentum_strength > 0.4 and consensus_score > 0.5:
                if momentum_direction == "BULLISH":
                    decision = "BUY"
                    confidence = min(momentum_strength * consensus_score, 1.0)
                    
                    # Reduce confidence if overbought
                    if extreme_condition == "OVERBOUGHT":
                        confidence *= 0.7  # 30% reduction for overbought
                        
                elif momentum_direction == "BEARISH":
                    decision = "SELL"
                    confidence = min(momentum_strength * consensus_score, 1.0)
                    
                    # Reduce confidence if oversold
                    if extreme_condition == "OVERSOLD":
                        confidence *= 0.7  # 30% reduction for oversold
            
            # Build comprehensive reasoning
            reasoning = self._build_reasoning(momentum_analysis, decision, confidence)
            
            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "momentum_analysis": momentum_analysis,
                    "indicators_analyzed": momentum_analysis["indicators_count"],
                    "ensemble_method": "weighted_voting"
                },
                risk_assessment={
                    "momentum_strength": momentum_strength,
                    "consensus_quality": consensus_score,
                    "extreme_condition": extreme_condition
                }
            )

        except Exception as e:
            logger.error(f"Enhanced momentum analysis failed: {e}")
            raise AgentError(f"Enhanced momentum analysis failed: {e}")

    def _analyze_momentum_ensemble(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze momentum using weighted ensemble of all available indicators.
        
        Returns:
        - momentum_direction: BULLISH/BEARISH/NEUTRAL
        - momentum_strength: 0.0 to 1.0
        - consensus_score: How well indicators agree
        - extreme_condition: OVERBOUGHT/OVERSOLD/NORMAL
        """
        
        bullish_votes = 0.0
        bearish_votes = 0.0
        neutral_votes = 0.0
        total_weight = 0.0
        
        oversold_count = 0
        overbought_count = 0
        normal_count = 0
        
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
            signal, extreme = self._interpret_momentum_indicator(indicator_name, indicator_data)
            individual_signals[indicator_name] = {"signal": signal, "extreme": extreme}
            
            # Add weighted vote
            if signal == "BULLISH":
                bullish_votes += weight
            elif signal == "BEARISH":
                bearish_votes += weight
            else:
                neutral_votes += weight
            
            # Track extreme conditions
            if extreme == "OVERSOLD":
                oversold_count += 1
            elif extreme == "OVERBOUGHT":
                overbought_count += 1
            else:
                normal_count += 1
            
            total_weight += weight
        
        # Calculate ensemble results
        if total_weight > 0:
            bullish_ratio = bullish_votes / total_weight
            bearish_ratio = bearish_votes / total_weight
            neutral_ratio = neutral_votes / total_weight
        else:
            bullish_ratio = bearish_ratio = neutral_ratio = 0.33
        
        # Determine overall momentum
        if bullish_ratio > bearish_ratio and bullish_ratio > neutral_ratio:
            momentum_direction = "BULLISH"
            momentum_strength = bullish_ratio
        elif bearish_ratio > bullish_ratio and bearish_ratio > neutral_ratio:
            momentum_direction = "BEARISH"
            momentum_strength = bearish_ratio
        else:
            momentum_direction = "NEUTRAL"
            momentum_strength = max(bullish_ratio, bearish_ratio, neutral_ratio)
        
        # Determine extreme condition
        total_extreme = oversold_count + overbought_count + normal_count
        if total_extreme > 0:
            if oversold_count / total_extreme > 0.5:
                extreme_condition = "OVERSOLD"
            elif overbought_count / total_extreme > 0.5:
                extreme_condition = "OVERBOUGHT"
            else:
                extreme_condition = "NORMAL"
        else:
            extreme_condition = "NORMAL"
        
        # Calculate consensus
        max_ratio = max(bullish_ratio, bearish_ratio, neutral_ratio)
        consensus_score = max_ratio
        
        return {
            "momentum_direction": momentum_direction,
            "momentum_strength": momentum_strength,
            "consensus_score": consensus_score,
            "extreme_condition": extreme_condition,
            "bullish_ratio": bullish_ratio,
            "bearish_ratio": bearish_ratio,
            "neutral_ratio": neutral_ratio,
            "oversold_count": oversold_count,
            "overbought_count": overbought_count,
            "individual_signals": individual_signals,
            "indicators_count": indicators_analyzed,
            "total_weight": total_weight
        }
    
    def _interpret_momentum_indicator(self, indicator_name: str, indicator_data: Any) -> tuple:
        """
        Interpret a single momentum indicator.
        
        Returns: (signal, extreme)
        - signal: "BULLISH", "BEARISH", or "NEUTRAL"
        - extreme: "OVERBOUGHT", "OVERSOLD", or "NORMAL"
        """
        
        if not indicator_data or not isinstance(indicator_data, dict):
            return ("NEUTRAL", "NORMAL")
        
        signal = "NEUTRAL"
        extreme = "NORMAL"
        
        # RSI
        if indicator_name == "rsi_indicator":
            rsi_value = indicator_data.get("value", 50)
            if rsi_value < self.rsi_oversold:
                signal = "BULLISH"  # Oversold = potential buy
                extreme = "OVERSOLD"
            elif rsi_value > self.rsi_overbought:
                signal = "BEARISH"  # Overbought = potential sell
                extreme = "OVERBOUGHT"
            elif rsi_value > 50:
                signal = "BULLISH"
            else:
                signal = "BEARISH"
        
        # MACD
        elif indicator_name == "macd_indicator":
            histogram = indicator_data.get("histogram", 0)
            macd_line = indicator_data.get("macd", 0)
            signal_line = indicator_data.get("signal", 0)
            
            if histogram > 0 and macd_line > signal_line:
                signal = "BULLISH"
            elif histogram < 0 and macd_line < signal_line:
                signal = "BEARISH"
        
        # Stochastic RSI
        elif indicator_name == "stochastic_rsi_indicator":
            k_value = indicator_data.get("k", 50)
            if k_value < 20:
                signal = "BULLISH"
                extreme = "OVERSOLD"
            elif k_value > 80:
                signal = "BEARISH"
                extreme = "OVERBOUGHT"
            elif k_value > 50:
                signal = "BULLISH"
            else:
                signal = "BEARISH"
        
        # CCI
        elif indicator_name == "commodity_channel_index_indicator":
            cci_value = indicator_data.get("value", 0)
            if cci_value < -100:
                signal = "BULLISH"
                extreme = "OVERSOLD"
            elif cci_value > 100:
                signal = "BEARISH"
                extreme = "OVERBOUGHT"
            elif cci_value > 0:
                signal = "BULLISH"
            else:
                signal = "BEARISH"
        
        # Money Flow Index
        elif indicator_name == "money_flow_index_indicator":
            mfi_value = indicator_data.get("value", 50)
            if mfi_value < 20:
                signal = "BULLISH"
                extreme = "OVERSOLD"
            elif mfi_value > 80:
                signal = "BEARISH"
                extreme = "OVERBOUGHT"
            elif mfi_value > 50:
                signal = "BULLISH"
            else:
                signal = "BEARISH"
        
        # Rate of Change
        elif indicator_name == "rate_of_change_indicator":
            roc_value = indicator_data.get("value", 0)
            if roc_value > 0:
                signal = "BULLISH"
            elif roc_value < 0:
                signal = "BEARISH"
        
        # Generic interpretation for other oscillators
        else:
            # Try common keys
            value = indicator_data.get("value", None)
            momentum_signal = indicator_data.get("signal", "").upper()
            direction = indicator_data.get("direction", "").upper()
            
            # Check for text signals
            combined = momentum_signal + direction
            if "BULL" in combined or "UP" in combined or "BUY" in combined or "POSITIVE" in combined:
                signal = "BULLISH"
            elif "BEAR" in combined or "DOWN" in combined or "SELL" in combined or "NEGATIVE" in combined:
                signal = "BEARISH"
            
            # Check numeric value if available
            elif value is not None:
                if value > 0:
                    signal = "BULLISH"
                elif value < 0:
                    signal = "BEARISH"
        
        return (signal, extreme)
    
    def _build_reasoning(self, momentum_analysis: Dict[str, Any], decision: str, confidence: float) -> str:
        """Build comprehensive reasoning string."""
        direction = momentum_analysis["momentum_direction"]
        strength = momentum_analysis["momentum_strength"]
        consensus = momentum_analysis["consensus_score"]
        extreme = momentum_analysis["extreme_condition"]
        indicators_count = momentum_analysis["indicators_count"]
        
        bullish_pct = int(momentum_analysis["bullish_ratio"] * 100)
        bearish_pct = int(momentum_analysis["bearish_ratio"] * 100)
        
        reasoning = (
            f"Momentum is {direction} with strength {strength:.2f}. "
            f"Ensemble analysis of {indicators_count} momentum indicators: "
            f"{bullish_pct}% Bullish, {bearish_pct}% Bearish. "
            f"Consensus score: {consensus:.2f}. "
            f"Market condition: {extreme}. "
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
