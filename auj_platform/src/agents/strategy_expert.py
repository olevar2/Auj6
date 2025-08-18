"""
Strategy Expert Agent for the AUJ Platform.

This agent specializes in core strategy analysis, trend following, and momentum shifts.
It focuses on 23 key indicators for strategic decision making and trend identification.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal
import yaml
import os

from .base_agent import BaseAgent, AnalysisResult, AgentState
from ..core.data_contracts import MarketConditions, TradeDirection, ConfidenceLevel
from ..core.exceptions import AgentError, ValidationError
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class StrategyExpert(BaseAgent):
    """
    Strategy Expert Agent - Core strategy, trend following, and momentum shifts.

    Specializes in:
    - Momentum analysis (MACD, RSI, Stochastic)
    - Trend identification (ADX, Super Trend, Parabolic SAR)
    - Strategic signal generation
    - Multi-timeframe analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager: Optional[UnifiedConfigManager] = None):
        """Initialize the Strategy Expert Agent."""
        # Define assigned indicators for this agent (from registry)
        assigned_indicators = [
            # ALL Momentum Indicators (30)
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
            "wr_signal_indicator",

            # ALL Trend Indicators (29)
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
            name="StrategyExpert",
            specialization="Core strategy, trend following, and momentum analysis",
            assigned_indicators=assigned_indicators,
            config=config,
            config_manager=config_manager
        )

        # Load configuration from YAML file
        self._load_agent_config()

        # Timeframe preferences from unified config
        self.primary_timeframe = self.config_manager.get_str('agents.strategy_expert.primary_timeframe', '1H')
        self.confirmation_timeframes = self.config_manager.get_list('agents.strategy_expert.confirmation_timeframes', ['4H', '1D'])

        logger.info(f"StrategyExpert initialized with {len(assigned_indicators)} indicators")

    def _load_agent_config(self):
        """Load agent-specific configuration from YAML file."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'strategy_expert.yaml'
            )

            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    self.agent_config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file not found: {config_path}, using defaults")
                self.agent_config = {}

        except Exception as e:
            logger.error(f"Error loading agent configuration: {e}, using defaults")
            self.agent_config = {}

        # Set parameters from unified config with fallback defaults
        self.momentum_threshold = self.config_manager.get_float('agents.strategy_expert.momentum_threshold', 0.6)
        self.trend_strength_threshold = self.config_manager.get_float('agents.strategy_expert.trend_strength_threshold', 0.7)
        self.confluence_required = self.config_manager.get_int('agents.strategy_expert.confluence_required', 3)
        self.risk_reward_ratio = self.config_manager.get_float('agents.strategy_expert.risk_reward_ratio', 2.0)

        # RSI parameters from unified config
        self.rsi_oversold_threshold = self.config_manager.get_int('agents.strategy_expert.rsi_oversold_threshold', 30)
        self.rsi_neutral_low = self.config_manager.get_int('agents.strategy_expert.rsi_neutral_low', 45)
        self.rsi_neutral_high = self.config_manager.get_int('agents.strategy_expert.rsi_neutral_high', 55)
        self.rsi_overbought_threshold = self.config_manager.get_int('agents.strategy_expert.rsi_overbought_threshold', 70)

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """
        Perform comprehensive data-aware strategy analysis.

        Args:
            symbol: Trading symbol
            market_data: Available market data
            market_conditions: Current market conditions
            indicator_results: Calculated indicator values

        Returns:
            AnalysisResult with strategic recommendation
        """
        try:
            # Data-aware assessment
            data_assessment = self._assess_data_availability(market_data)

            # 1. Momentum Analysis (adapted to data quality)
            momentum_signals = self._analyze_momentum_data_aware(indicator_results, data_assessment)

            # 2. Trend Analysis (enhanced with multi-timeframe if available)
            trend_signals = self._analyze_trend_data_aware(indicator_results, market_conditions, data_assessment)

            # 3. Strategic Confluence (weighted by data quality)
            confluence_analysis = self._analyze_confluence_enhanced(momentum_signals, trend_signals, data_assessment)

            # 4. Risk Assessment (adjusted for data limitations)
            risk_assessment = self._assess_strategy_risk_enhanced(
                symbol, market_data, indicator_results, market_conditions, data_assessment
            )

            # 5. Generate Decision (considering data quality)
            decision = self._generate_strategic_decision_enhanced(
                confluence_analysis, risk_assessment, market_conditions, data_assessment
            )

            # 6. Calculate Confidence (penalized by data quality)
            confidence = self._calculate_confidence_enhanced(
                momentum_signals, trend_signals, confluence_analysis, risk_assessment, data_assessment
            )

            # 7. Generate Reasoning
            reasoning = self._generate_reasoning_enhanced(
                decision, momentum_signals, trend_signals, confluence_analysis, data_assessment
            )

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "momentum_signals": momentum_signals,
                    "trend_signals": trend_signals,
                    "confluence_analysis": confluence_analysis,
                    "primary_timeframe": self.primary_timeframe,
                    "data_assessment": data_assessment
                },
                risk_assessment=risk_assessment,
                supporting_data={
                    "market_regime": market_conditions.regime.value,
                    "volatility_level": market_conditions.volatility,
                    "trend_strength": trend_signals.get("overall_strength", 0.0),
                    "data_quality_score": data_assessment.get("quality_score", 0.0)
                }
            )

        except Exception as e:
            logger.error(f"StrategyExpert analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Strategic analysis failed: {str(e)}")

    def _analyze_momentum(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators for signal generation."""
        momentum_signals = {
            "rsi_signal": "NEUTRAL",
            "macd_signal": "NEUTRAL",
            "stochastic_signal": "NEUTRAL",
            "momentum_strength": 0.0,
            "momentum_direction": "NEUTRAL",
            "divergence_detected": False
        }

        # RSI Analysis
        if "rsi_indicator" in indicator_results:
            rsi_value = indicator_results["rsi_indicator"].get("value", 50)
            if rsi_value < self.rsi_oversold_threshold:
                momentum_signals["rsi_signal"] = "OVERSOLD_BUY"
            elif rsi_value > self.rsi_overbought_threshold:
                momentum_signals["rsi_signal"] = "OVERBOUGHT_SELL"
            elif self.rsi_oversold_threshold <= rsi_value <= self.rsi_neutral_low:
                momentum_signals["rsi_signal"] = "WEAK_BUY"
            elif self.rsi_neutral_high <= rsi_value <= self.rsi_overbought_threshold:
                momentum_signals["rsi_signal"] = "WEAK_SELL"

        # MACD Analysis
        if "macd_indicator" in indicator_results:
            macd_data = indicator_results["macd_indicator"]
            macd_line = macd_data.get("macd", 0)
            signal_line = macd_data.get("signal", 0)
            histogram = macd_data.get("histogram", 0)

            if macd_line > signal_line and histogram > 0:
                momentum_signals["macd_signal"] = "BULLISH"
            elif macd_line < signal_line and histogram < 0:
                momentum_signals["macd_signal"] = "BEARISH"
            else:
                momentum_signals["macd_signal"] = "NEUTRAL"

        # Stochastic Analysis
        if "stochastic_oscillator_indicator" in indicator_results:
            stoch_data = indicator_results["stochastic_oscillator_indicator"]
            k_percent = stoch_data.get("k_percent", 50)
            d_percent = stoch_data.get("d_percent", 50)

            if k_percent < 20 and d_percent < 20:
                momentum_signals["stochastic_signal"] = "OVERSOLD_BUY"
            elif k_percent > 80 and d_percent > 80:
                momentum_signals["stochastic_signal"] = "OVERBOUGHT_SELL"
            elif k_percent > d_percent:
                momentum_signals["stochastic_signal"] = "BULLISH"
            else:
                momentum_signals["stochastic_signal"] = "BEARISH"

        # Calculate overall momentum strength
        bullish_count = sum(1 for signal in momentum_signals.values()
                           if isinstance(signal, str) and "BUY" in signal or signal == "BULLISH")
        bearish_count = sum(1 for signal in momentum_signals.values()
                           if isinstance(signal, str) and "SELL" in signal or signal == "BEARISH")

        total_signals = bullish_count + bearish_count
        if total_signals > 0:
            momentum_signals["momentum_strength"] = max(bullish_count, bearish_count) / total_signals
            if bullish_count > bearish_count:
                momentum_signals["momentum_direction"] = "BULLISH"
            elif bearish_count > bullish_count:
                momentum_signals["momentum_direction"] = "BEARISH"

        return momentum_signals

    def _analyze_trend(self, indicator_results: Dict[str, Any], market_conditions: MarketConditions) -> Dict[str, Any]:
        """Analyze trend indicators for direction and strength."""
        trend_signals = {
            "adx_trend": "NEUTRAL",
            "supertrend_signal": "NEUTRAL",
            "parabolic_sar_signal": "NEUTRAL",
            "overall_trend": "NEUTRAL",
            "overall_strength": 0.0,
            "trend_confirmed": False
        }

        # ADX Analysis
        if "adx_indicator" in indicator_results:
            adx_data = indicator_results["adx_indicator"]
            adx_value = adx_data.get("adx", 0)
            plus_di = adx_data.get("plus_di", 0)
            minus_di = adx_data.get("minus_di", 0)

            if adx_value > 25:  # Strong trend
                if plus_di > minus_di:
                    trend_signals["adx_trend"] = "STRONG_UPTREND"
                else:
                    trend_signals["adx_trend"] = "STRONG_DOWNTREND"
            elif adx_value > 20:  # Moderate trend
                if plus_di > minus_di:
                    trend_signals["adx_trend"] = "UPTREND"
                else:
                    trend_signals["adx_trend"] = "DOWNTREND"

            trend_signals["overall_strength"] = min(adx_value / 50.0, 1.0)

        # SuperTrend Analysis
        if "super_trend_indicator" in indicator_results:
            supertrend_data = indicator_results["super_trend_indicator"]
            trend_direction = supertrend_data.get("trend", "NEUTRAL")
            trend_signals["supertrend_signal"] = trend_direction

        # Parabolic SAR Analysis
        if "parabolic_sar_indicator" in indicator_results:
            sar_data = indicator_results["parabolic_sar_indicator"]
            sar_signal = sar_data.get("signal", "NEUTRAL")
            trend_signals["parabolic_sar_signal"] = sar_signal

        # Overall Trend Determination
        trend_votes = []
        for signal_key in ["adx_trend", "supertrend_signal", "parabolic_sar_signal"]:
            signal = trend_signals[signal_key]
            if "UP" in signal or signal == "BULLISH":
                trend_votes.append("BULLISH")
            elif "DOWN" in signal or signal == "BEARISH":
                trend_votes.append("BEARISH")

        if len(trend_votes) >= self.config_manager.get_int('agents.strategy_expert.min_trend_votes', 2):
            bullish_votes = trend_votes.count("BULLISH")
            bearish_votes = trend_votes.count("BEARISH")

            if bullish_votes > bearish_votes:
                trend_signals["overall_trend"] = "BULLISH"
                trend_signals["trend_confirmed"] = bullish_votes >= self.config_manager.get_int('agents.strategy_expert.min_bullish_votes', 2)
            elif bearish_votes > bullish_votes:
                trend_signals["overall_trend"] = "BEARISH"
                trend_signals["trend_confirmed"] = bearish_votes >= self.config_manager.get_int('agents.strategy_expert.min_bearish_votes', 2)

        return trend_signals

    def _analyze_confluence(self, momentum_signals: Dict[str, Any], trend_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confluence between momentum and trend signals."""
        confluence_analysis = {
            "signal_alignment": "NEUTRAL",
            "confluence_strength": 0.0,
            "conflicting_signals": 0,
            "supporting_signals": 0,
            "overall_confluence": "WEAK"
        }

        # Count bullish and bearish signals
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # Check momentum signals
        for key, signal in momentum_signals.items():
            if isinstance(signal, str):
                if "BUY" in signal or signal == "BULLISH":
                    bullish_signals += 1
                    total_signals += 1
                elif "SELL" in signal or signal == "BEARISH":
                    bearish_signals += 1
                    total_signals += 1

        # Check trend signals
        for key, signal in trend_signals.items():
            if isinstance(signal, str):
                if "UP" in signal or signal == "BULLISH":
                    bullish_signals += 1
                    total_signals += 1
                elif "DOWN" in signal or signal == "BEARISH":
                    bearish_signals += 1
                    total_signals += 1

        # Calculate confluence
        if total_signals > 0:
            confluence_analysis["supporting_signals"] = max(bullish_signals, bearish_signals)
            confluence_analysis["conflicting_signals"] = min(bullish_signals, bearish_signals)
            confluence_analysis["confluence_strength"] = confluence_analysis["supporting_signals"] / total_signals

            # Determine signal alignment
            if bullish_signals > bearish_signals and confluence_analysis["supporting_signals"] >= self.confluence_required:
                confluence_analysis["signal_alignment"] = "BULLISH"
            elif bearish_signals > bullish_signals and confluence_analysis["supporting_signals"] >= self.confluence_required:
                confluence_analysis["signal_alignment"] = "BEARISH"

            # Determine confluence quality
            if confluence_analysis["confluence_strength"] >= 0.8:
                confluence_analysis["overall_confluence"] = "VERY_STRONG"
            elif confluence_analysis["confluence_strength"] >= 0.6:
                confluence_analysis["overall_confluence"] = "STRONG"
            elif confluence_analysis["confluence_strength"] >= 0.4:
                confluence_analysis["overall_confluence"] = "MODERATE"

        return confluence_analysis

    def _assess_strategy_risk(self,
                            symbol: str,
                            market_data: Dict[str, pd.DataFrame],
                            indicator_results: Dict[str, Any],
                            market_conditions: MarketConditions) -> Dict[str, Any]:
        """Assess strategic risk factors."""
        risk_assessment = {
            "volatility_risk": "MEDIUM",
            "trend_risk": "MEDIUM",
            "momentum_risk": "MEDIUM",
            "overall_risk": "MEDIUM",
            "risk_score": 0.5,
            "position_sizing_factor": 1.0
        }

        # Volatility Risk Assessment
        volatility = market_conditions.volatility
        if volatility > 0.03:  # 3% daily volatility
            risk_assessment["volatility_risk"] = "HIGH"
        elif volatility < 0.01:  # 1% daily volatility
            risk_assessment["volatility_risk"] = "LOW"

        # Trend Risk Assessment
        if market_conditions.regime.value in ["SIDEWAYS", "HIGH_VOLATILITY"]:
            risk_assessment["trend_risk"] = "HIGH"
        elif market_conditions.regime.value in ["TRENDING_UP", "TRENDING_DOWN"]:
            risk_assessment["trend_risk"] = "LOW"

        # Calculate overall risk score
        risk_factors = []
        if risk_assessment["volatility_risk"] == "HIGH":
            risk_factors.append(0.8)
        elif risk_assessment["volatility_risk"] == "LOW":
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.5)

        if risk_assessment["trend_risk"] == "HIGH":
            risk_factors.append(0.8)
        elif risk_assessment["trend_risk"] == "LOW":
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.5)

        risk_assessment["risk_score"] = sum(risk_factors) / len(risk_factors)

        # Determine overall risk
        if risk_assessment["risk_score"] > 0.7:
            risk_assessment["overall_risk"] = "HIGH"
            risk_assessment["position_sizing_factor"] = 0.5
        elif risk_assessment["risk_score"] < 0.3:
            risk_assessment["overall_risk"] = "LOW"
            risk_assessment["position_sizing_factor"] = 1.5
        else:
            risk_assessment["position_sizing_factor"] = 1.0

        return risk_assessment

    def _generate_strategic_decision(self,
                                   confluence_analysis: Dict[str, Any],
                                   risk_assessment: Dict[str, Any],
                                   market_conditions: MarketConditions) -> str:
        """Generate strategic trading decision."""
        # Check confluence strength
        if confluence_analysis["overall_confluence"] in ["WEAK"]:
            return "NO_SIGNAL"

        # Check risk levels
        if risk_assessment["overall_risk"] == "HIGH" and confluence_analysis["confluence_strength"] < 0.8:
            return "NO_SIGNAL"

        # Generate signal based on alignment
        signal_alignment = confluence_analysis["signal_alignment"]

        if signal_alignment == "BULLISH":
            return "BUY"
        elif signal_alignment == "BEARISH":
            return "SELL"
        else:
            return "HOLD"

    def _calculate_confidence(self,
                            momentum_signals: Dict[str, Any],
                            trend_signals: Dict[str, Any],
                            confluence_analysis: Dict[str, Any],
                            risk_assessment: Dict[str, Any]) -> float:
        """Calculate confidence level for the strategic decision."""
        confidence_factors = []

        # Confluence strength factor (40% weight)
        confluence_strength = confluence_analysis.get("confluence_strength", 0.0)
        confidence_factors.append(confluence_strength * 0.4)

        # Trend confirmation factor (30% weight)
        trend_confirmed = trend_signals.get("trend_confirmed", False)
        trend_strength = trend_signals.get("overall_strength", 0.0)
        if trend_confirmed:
            confidence_factors.append(trend_strength * 0.3)
        else:
            confidence_factors.append(trend_strength * 0.15)

        # Momentum strength factor (20% weight)
        momentum_strength = momentum_signals.get("momentum_strength", 0.0)
        confidence_factors.append(momentum_strength * 0.2)

        # Risk adjustment factor (10% weight)
        risk_score = risk_assessment.get("risk_score", 0.5)
        risk_adjustment = 1.0 - risk_score  # Lower risk = higher confidence
        confidence_factors.append(risk_adjustment * 0.1)

        return min(sum(confidence_factors), 1.0)

    def _generate_reasoning(self,
                          decision: str,
                          momentum_signals: Dict[str, Any],
                          trend_signals: Dict[str, Any],
                          confluence_analysis: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the decision."""
        reasoning_parts = []

        # Decision explanation
        if decision == "BUY":
            reasoning_parts.append("Strategic BUY signal generated based on bullish confluence.")
        elif decision == "SELL":
            reasoning_parts.append("Strategic SELL signal generated based on bearish confluence.")
        elif decision == "HOLD":
            reasoning_parts.append("HOLD position recommended due to mixed signals.")
        else:
            reasoning_parts.append("NO clear strategic signal identified.")

        # Confluence details
        confluence_strength = confluence_analysis.get("confluence_strength", 0.0)
        supporting_signals = confluence_analysis.get("supporting_signals", 0)
        reasoning_parts.append(f"Confluence strength: {confluence_strength:.2f} with {supporting_signals} supporting indicators.")

        # Trend analysis
        overall_trend = trend_signals.get("overall_trend", "NEUTRAL")
        trend_strength = trend_signals.get("overall_strength", 0.0)
        reasoning_parts.append(f"Trend analysis shows {overall_trend} direction with {trend_strength:.2f} strength.")

        # Momentum analysis
        momentum_direction = momentum_signals.get("momentum_direction", "NEUTRAL")
        momentum_strength = momentum_signals.get("momentum_strength", 0.0)
        reasoning_parts.append(f"Momentum indicators suggest {momentum_direction} bias with {momentum_strength:.2f} strength.")

        return " ".join(reasoning_parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        used_indicators = []
        for indicator in self.assigned_indicators:
            if indicator in indicator_results:
                used_indicators.append(indicator)
        return used_indicators

    def get_required_data_types(self) -> List[str]:
        """Define required data types for strategy analysis."""
        return ["OHLCV"]  # Primary focus on price action

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed for analysis."""
        return 50  # Need sufficient history for trend and momentum analysis

    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return {
            "momentum_threshold": self.momentum_threshold,
            "trend_strength_threshold": self.trend_strength_threshold,
            "confluence_required": self.confluence_required,
            "risk_reward_ratio": self.risk_reward_ratio,
            "primary_timeframe": self.primary_timeframe,
            "confirmation_timeframes": self.confirmation_timeframes
        }

    def update_strategy_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters based on performance feedback."""
        if "momentum_threshold" in new_parameters:
            self.momentum_threshold = new_parameters["momentum_threshold"]
        if "trend_strength_threshold" in new_parameters:
            self.trend_strength_threshold = new_parameters["trend_strength_threshold"]
        if "confluence_required" in new_parameters:
            self.confluence_required = new_parameters["confluence_required"]

        logger.info(f"StrategyExpert parameters updated: {new_parameters}")
    def _assess_data_availability(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess available data sources and quality for strategy analysis."""
        assessment = {
            "primary_data_available": False,
            "multi_timeframe_available": False,
            "data_completeness": 0.0,
            "quality_score": 0.0,
            "fallback_mode": False,
            "available_timeframes": [],
            "data_source": "UNKNOWN"
        }

        # Check OHLCV data availability (primary requirement)
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            assessment["primary_data_available"] = True
            assessment["data_source"] = "MT5" if len(market_data['OHLCV']) > 1000 else "YAHOO"

            # Check data completeness
            ohlcv_data = market_data['OHLCV']
            null_percentage = ohlcv_data.isnull().sum().sum() / (len(ohlcv_data) * len(ohlcv_data.columns))
            assessment["data_completeness"] = 1.0 - null_percentage

        # Check for multi-timeframe data
        timeframe_indicators = ['1H', '4H', '1D', 'H1', 'H4', 'D1']
        for tf in timeframe_indicators:
            if tf in market_data and not market_data[tf].empty:
                assessment["available_timeframes"].append(tf)

        assessment["multi_timeframe_available"] = len(assessment["available_timeframes"]) > 1

        # Calculate quality score
        quality_factors = []
        quality_factors.append(1.0 if assessment["primary_data_available"] else 0.0)
        quality_factors.append(0.5 if assessment["multi_timeframe_available"] else 0.0)
        quality_factors.append(assessment["data_completeness"])
        assessment["quality_score"] = sum(quality_factors) / len(quality_factors)

        # Determine if fallback mode needed
        assessment["fallback_mode"] = assessment["quality_score"] < 0.6

        return assessment

    def _analyze_momentum_data_aware(self, indicator_results: Dict[str, Any], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced momentum analysis that adapts to data quality."""
        # Base momentum analysis
        momentum_signals = self._analyze_momentum(indicator_results)

        # Enhance with data-aware features
        momentum_signals["data_quality_adjustment"] = data_assessment["quality_score"]

        # Adjust momentum strength based on data quality
        if data_assessment["fallback_mode"]:
            # Reduce confidence in fallback mode
            momentum_signals["momentum_strength"] *= 0.7
            momentum_signals["reliability"] = "REDUCED_QUALITY"
        else:
            momentum_signals["reliability"] = "HIGH_QUALITY"

        # Multi-timeframe momentum if available
        if data_assessment["multi_timeframe_available"]:
            momentum_signals["multi_timeframe_confirmation"] = self._check_momentum_confluence(indicator_results)

            # Boost strength if multiple timeframes agree
            if momentum_signals["multi_timeframe_confirmation"]:
                momentum_signals["momentum_strength"] *= 1.2
                momentum_signals["momentum_strength"] = min(momentum_signals["momentum_strength"], 1.0)

        return momentum_signals

    def _analyze_trend_data_aware(self, indicator_results: Dict[str, Any], market_conditions: MarketConditions, data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced trend analysis that leverages available data sources."""
        # Base trend analysis
        trend_signals = self._analyze_trend(indicator_results, market_conditions)

        # Data quality adjustments
        trend_signals["data_source"] = data_assessment["data_source"]
        trend_signals["quality_adjustment"] = data_assessment["quality_score"]

        # Enhanced analysis for high-quality data
        if not data_assessment["fallback_mode"]:
            # More sophisticated trend analysis with better data
            trend_signals.update(self._enhanced_trend_analysis(indicator_results))
        else:
            # Conservative approach with limited data
            trend_signals["overall_strength"] *= 0.8
            trend_signals["confidence_penalty"] = 0.2

        # Multi-timeframe trend confirmation
        if data_assessment["multi_timeframe_available"]:
            mtf_trend = self._multi_timeframe_trend_analysis(indicator_results, data_assessment["available_timeframes"])
            trend_signals.update(mtf_trend)

            # Increase confidence if trends align across timeframes
            if mtf_trend.get("timeframe_alignment", False):
                trend_signals["overall_strength"] *= 1.3
                trend_signals["overall_strength"] = min(trend_signals["overall_strength"], 1.0)

        return trend_signals
    def _check_momentum_confluence(self, indicator_results: Dict[str, Any]) -> bool:
        """Check if momentum indicators agree across multiple timeframes."""
        momentum_indicators = ["rsi_indicator", "macd_indicator", "stochastic_oscillator_indicator"]

        for indicator in momentum_indicators:
            if indicator in indicator_results:
                # Check if indicator shows same direction across timeframes
                indicator_data = indicator_results[indicator]
                if isinstance(indicator_data, dict) and "timeframe_signals" in indicator_data:
                    signals = indicator_data["timeframe_signals"]
                    if len(set(signals)) == 1:  # All timeframes agree
                        return True

        return False

    def _enhanced_trend_analysis(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform enhanced trend analysis with high-quality data."""
        enhanced_trend = {
            "trend_quality": "HIGH",
            "trend_persistence": 0.0,
            "reversal_probability": 0.0
        }

        # Analyze trend persistence using multiple indicators
        if "adx_indicator" in indicator_results and "super_trend_indicator" in indicator_results:
            adx_value = indicator_results["adx_indicator"].get("adx", 0)
            supertrend_stability = indicator_results["super_trend_indicator"].get("stability", 0)

            enhanced_trend["trend_persistence"] = min((adx_value + supertrend_stability) / 100, 1.0)

        # Calculate reversal probability
        reversal_indicators = 0
        total_indicators = 0

        if "parabolic_sar_indicator" in indicator_results:
            sar_data = indicator_results["parabolic_sar_indicator"]
            if sar_data.get("reversal_signal", False):
                reversal_indicators += 1
            total_indicators += 1

        if total_indicators > 0:
            enhanced_trend["reversal_probability"] = reversal_indicators / total_indicators

        return enhanced_trend

    def _multi_timeframe_trend_analysis(self, indicator_results: Dict[str, Any], available_timeframes: List[str]) -> Dict[str, Any]:
        """Analyze trend consistency across multiple timeframes."""
        mtf_analysis = {
            "timeframe_alignment": False,
            "primary_timeframe_trend": "NEUTRAL",
            "higher_timeframe_trend": "NEUTRAL",
            "trend_harmony": 0.0
        }

        # Primary timeframe (1H)
        if self.primary_timeframe in available_timeframes:
            mtf_analysis["primary_timeframe_trend"] = self._extract_timeframe_trend(indicator_results, self.primary_timeframe)

        # Higher timeframe analysis (4H, 1D)
        higher_tf_trends = []
        for tf in self.confirmation_timeframes:
            if tf in available_timeframes:
                trend = self._extract_timeframe_trend(indicator_results, tf)
                higher_tf_trends.append(trend)

        if higher_tf_trends:
            # Check if higher timeframes agree
            if len(set(higher_tf_trends)) == 1:
                mtf_analysis["higher_timeframe_trend"] = higher_tf_trends[0]

                # Check alignment between primary and higher timeframes
                if mtf_analysis["primary_timeframe_trend"] == mtf_analysis["higher_timeframe_trend"]:
                    mtf_analysis["timeframe_alignment"] = True
                    mtf_analysis["trend_harmony"] = 1.0
                else:
                    mtf_analysis["trend_harmony"] = 0.5

        return mtf_analysis

    def _extract_timeframe_trend(self, indicator_results: Dict[str, Any], timeframe: str) -> str:
        """Extract trend direction for a specific timeframe."""
        # Look for timeframe-specific trend indicators
        trend_indicators = ["super_trend_indicator", "adx_indicator"]

        for indicator in trend_indicators:
            if indicator in indicator_results:
                indicator_data = indicator_results[indicator]
                if isinstance(indicator_data, dict) and f"{timeframe}_trend" in indicator_data:
                    return indicator_data[f"{timeframe}_trend"]

        return "NEUTRAL"
    def _analyze_confluence_enhanced(self, momentum_signals: Dict[str, Any], trend_signals: Dict[str, Any], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced confluence analysis weighted by data quality."""
        # Base confluence analysis
        confluence_analysis = self._analyze_confluence(momentum_signals, trend_signals)

        # Data quality weighting
        quality_score = data_assessment["quality_score"]
        confluence_analysis["data_quality_weight"] = quality_score

        # Adjust confluence strength based on data quality
        confluence_analysis["raw_confluence_strength"] = confluence_analysis["confluence_strength"]
        confluence_analysis["confluence_strength"] *= quality_score

        # Multi-timeframe confluence bonus
        if data_assessment["multi_timeframe_available"]:
            mtf_bonus = 0.0

            # Check for momentum multi-timeframe confirmation
            if momentum_signals.get("multi_timeframe_confirmation", False):
                mtf_bonus += 0.1

            # Check for trend timeframe alignment
            if trend_signals.get("timeframe_alignment", False):
                mtf_bonus += 0.15

            confluence_analysis["confluence_strength"] = min(confluence_analysis["confluence_strength"] + mtf_bonus, 1.0)
            confluence_analysis["multi_timeframe_bonus"] = mtf_bonus

        # Enhanced confluence categories based on data quality
        if quality_score >= 0.8:
            if confluence_analysis["confluence_strength"] >= 0.7:
                confluence_analysis["overall_confluence"] = "VERY_STRONG"
            elif confluence_analysis["confluence_strength"] >= 0.5:
                confluence_analysis["overall_confluence"] = "STRONG"
        else:
            # More conservative thresholds for lower quality data
            if confluence_analysis["confluence_strength"] >= 0.8:
                confluence_analysis["overall_confluence"] = "STRONG"
            elif confluence_analysis["confluence_strength"] >= 0.6:
                confluence_analysis["overall_confluence"] = "MODERATE"

        return confluence_analysis

    def _assess_strategy_risk_enhanced(self, symbol: str, market_data: Dict[str, pd.DataFrame],
                                     indicator_results: Dict[str, Any], market_conditions: MarketConditions,
                                     data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk assessment considering data quality and availability."""
        # Base risk assessment
        risk_assessment = self._assess_strategy_risk(symbol, market_data, indicator_results, market_conditions)

        # Data quality risk adjustment
        quality_score = data_assessment["quality_score"]
        risk_assessment["data_quality_risk"] = "LOW" if quality_score > 0.8 else "MEDIUM" if quality_score > 0.5 else "HIGH"

        # Adjust position sizing based on data quality
        data_quality_factor = quality_score
        risk_assessment["position_sizing_factor"] *= data_quality_factor

        # Additional risk from fallback mode
        if data_assessment["fallback_mode"]:
            risk_assessment["fallback_mode_penalty"] = 0.3
            risk_assessment["position_sizing_factor"] *= 0.7

            # Increase overall risk assessment
            if risk_assessment["overall_risk"] == "LOW":
                risk_assessment["overall_risk"] = "MEDIUM"
            elif risk_assessment["overall_risk"] == "MEDIUM":
                risk_assessment["overall_risk"] = "HIGH"

        # Multi-timeframe risk reduction
        if data_assessment["multi_timeframe_available"]:
            risk_assessment["multi_timeframe_risk_reduction"] = 0.1
            risk_assessment["position_sizing_factor"] *= 1.1

        # Recalculate overall risk score
        original_score = risk_assessment["risk_score"]
        data_quality_penalty = (1.0 - quality_score) * 0.3
        risk_assessment["risk_score"] = min(original_score + data_quality_penalty, 1.0)

        return risk_assessment
    def _generate_strategic_decision_enhanced(self, confluence_analysis: Dict[str, Any], risk_assessment: Dict[str, Any],
                                            market_conditions: MarketConditions, data_assessment: Dict[str, Any]) -> str:
        """Enhanced strategic decision generation considering data quality."""
        # Check data quality threshold
        if data_assessment["quality_score"] < 0.4:
            return "NO_SIGNAL"  # Insufficient data quality

        # Enhanced confluence requirements based on data quality
        quality_score = data_assessment["quality_score"]

        # Adjust confluence requirements based on data quality
        if quality_score >= 0.8:
            min_confluence = 0.5  # Lower threshold for high-quality data
        elif quality_score >= 0.6:
            min_confluence = 0.6  # Standard threshold
        else:
            min_confluence = 0.7  # Higher threshold for lower-quality data

        confluence_strength = confluence_analysis["confluence_strength"]

        # Check confluence strength against adaptive threshold
        if confluence_strength < min_confluence:
            return "NO_SIGNAL"

        # Enhanced risk check with data quality consideration
        overall_risk = risk_assessment["overall_risk"]
        data_quality_risk = risk_assessment.get("data_quality_risk", "MEDIUM")

        # Conservative approach if data quality risk is high
        if data_quality_risk == "HIGH" and confluence_strength < 0.8:
            return "NO_SIGNAL"

        # Conservative approach for high market risk unless confluence is very strong
        if overall_risk == "HIGH" and confluence_strength < 0.85:
            return "NO_SIGNAL"

        # Generate signal based on enhanced alignment
        signal_alignment = confluence_analysis["signal_alignment"]

        # Multi-timeframe confirmation boost
        if data_assessment["multi_timeframe_available"] and confluence_analysis.get("multi_timeframe_bonus", 0) > 0.1:
            if signal_alignment == "BULLISH":
                return "STRONG_BUY"
            elif signal_alignment == "BEARISH":
                return "STRONG_SELL"

        # Standard signals
        if signal_alignment == "BULLISH":
            return "BUY"
        elif signal_alignment == "BEARISH":
            return "SELL"
        else:
            return "HOLD"

    def _calculate_confidence_enhanced(self, momentum_signals: Dict[str, Any], trend_signals: Dict[str, Any],
                                     confluence_analysis: Dict[str, Any], risk_assessment: Dict[str, Any],
                                     data_assessment: Dict[str, Any]) -> float:
        """Enhanced confidence calculation incorporating data quality."""
        # Base confidence calculation
        base_confidence = self._calculate_confidence(momentum_signals, trend_signals, confluence_analysis, risk_assessment)

        # Data quality adjustment (20% weight)
        quality_score = data_assessment["quality_score"]
        quality_adjustment = quality_score * 0.2

        # Multi-timeframe confidence boost (10% weight)
        mtf_boost = 0.0
        if data_assessment["multi_timeframe_available"]:
            if momentum_signals.get("multi_timeframe_confirmation", False):
                mtf_boost += 0.05
            if trend_signals.get("timeframe_alignment", False):
                mtf_boost += 0.05

        # Fallback mode penalty
        fallback_penalty = 0.0
        if data_assessment["fallback_mode"]:
            fallback_penalty = 0.15

        # Calculate enhanced confidence
        enhanced_confidence = base_confidence + quality_adjustment + mtf_boost - fallback_penalty

        return max(0.0, min(enhanced_confidence, 1.0))

    def _generate_reasoning_enhanced(self, decision: str, momentum_signals: Dict[str, Any],
                                   trend_signals: Dict[str, Any], confluence_analysis: Dict[str, Any],
                                   data_assessment: Dict[str, Any]) -> str:
        """Enhanced reasoning generation including data quality context."""
        reasoning_parts = []

        # Decision explanation with data context
        if decision in ["STRONG_BUY", "STRONG_SELL"]:
            reasoning_parts.append(f"Enhanced {decision} signal based on multi-timeframe confluence.")
        elif decision == "BUY":
            reasoning_parts.append("Strategic BUY signal generated based on bullish confluence.")
        elif decision == "SELL":
            reasoning_parts.append("Strategic SELL signal generated based on bearish confluence.")
        elif decision == "HOLD":
            reasoning_parts.append("HOLD position recommended due to mixed signals.")
        else:
            reasoning_parts.append("NO clear strategic signal identified.")

        # Data quality context
        quality_score = data_assessment["quality_score"]
        data_source = data_assessment["data_source"]
        reasoning_parts.append(f"Data quality: {quality_score:.2f} from {data_source} source.")

        # Confluence details with data adjustments
        confluence_strength = confluence_analysis.get("confluence_strength", 0.0)
        supporting_signals = confluence_analysis.get("supporting_signals", 0)
        reasoning_parts.append(f"Data-adjusted confluence: {confluence_strength:.2f} with {supporting_signals} supporting indicators.")

        # Multi-timeframe context if available
        if data_assessment["multi_timeframe_available"]:
            available_tfs = len(data_assessment["available_timeframes"])
            reasoning_parts.append(f"Multi-timeframe analysis across {available_tfs} timeframes.")

            if trend_signals.get("timeframe_alignment", False):
                reasoning_parts.append("Strong timeframe alignment detected.")

        # Fallback mode notification
        if data_assessment["fallback_mode"]:
            reasoning_parts.append("Analysis operating in fallback mode due to limited data quality.")

        return " ".join(reasoning_parts)
