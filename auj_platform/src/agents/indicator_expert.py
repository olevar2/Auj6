"""
Indicator Expert Agent for the AUJ Platform.

This agent specializes in technical indicator analysis and signal processing.
It focuses on 23 core technical indicators for comprehensive analysis.
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


class IndicatorExpert(BaseAgent):
    """
    Indicator Expert Agent - Technical indicator analysis and signal processing.

    Specializes in:
    - Oscillator analysis and divergences
    - Moving average systems
    - Momentum and rate of change
    - Multi-timeframe indicator confluence
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager: Optional[UnifiedConfigManager] = None):
        """Initialize the Indicator Expert Agent."""

        assigned_indicators = [
            # Synthesis & Confluence Indicators
            "composite_signal_indicator",
            "confluence_area_indicator",
            "market_breadth_indicator",
            "sector_momentum_indicator",
            "sentiment_integration_indicator",

            # Signal Processing & Filtering
            "adaptive_indicators",
            "pattern_signal_indicator",
            "sd_channel_signal",

            # Technical Analysis Synthesis
            "fibonacci_arc_indicator",
            "fibonacci_expansion_indicator",
            "fibonacci_grid_indicator",
            "hidden_divergence_detector_indicator",
            "momentum_divergence_scanner_indicator",
            "price_volume_divergence_indicator",

            # Volume Analysis (remaining)
            "volume_mass_index_indicator"
        ]

        super().__init__(
            name="IndicatorExpert",
            specialization="Technical indicator analysis and signal processing",
            assigned_indicators=assigned_indicators,
            config=config,
            config_manager=config_manager
        )

        # Load configuration
        self.config = self._load_agent_config()

        # Technical indicator thresholds from unified configuration
        self.rsi_overbought = self.config_manager.get_float('agents.indicator_expert.rsi_thresholds.overbought', 70.0)
        self.rsi_oversold = self.config_manager.get_float('agents.indicator_expert.rsi_thresholds.oversold', 30.0)
        self.stoch_overbought = self.config_manager.get_float('agents.indicator_expert.stochastic_thresholds.overbought', 80.0)
        self.stoch_oversold = self.config_manager.get_float('agents.indicator_expert.stochastic_thresholds.oversold', 20.0)
        self.williams_r_overbought = self.config_manager.get_float('agents.indicator_expert.williams_r_thresholds.overbought', -20.0)
        self.williams_r_oversold = self.config_manager.get_float('agents.indicator_expert.williams_r_thresholds.oversold', -80.0)

        # Signal strength thresholds from configuration
        self.strong_signal_threshold = self.config.get('signal_strength', {}).get('strong_threshold', 0.7)
        self.weak_signal_threshold = self.config.get('signal_strength', {}).get('weak_threshold', 0.3)

        logger.info(f"IndicatorExpert initialized with {len(assigned_indicators)} indicators")

    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML file with fallback defaults."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'indicator_expert.yaml'
            )

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded IndicatorExpert configuration from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading IndicatorExpert configuration: {e}")

        # Fallback defaults
        return {
            'rsi_thresholds': {
                'overbought': 70,
                'oversold': 30,
                'divergence_threshold': 5
            },
            'macd_settings': {
                'signal_threshold': 0.0,
                'histogram_divergence_threshold': 3,
                'crossover_confirmation_bars': 2
            },
            'stochastic_thresholds': {
                'overbought': 80,
                'oversold': 20,
                'k_period': 14,
                'd_period': 3
            },
            'williams_r_thresholds': {
                'overbought': -20,
                'oversold': -80,
                'period': 14
            },
            'signal_strength': {
                'strong_threshold': 0.7,
                'weak_threshold': 0.3,
                'confluence_multiplier': 1.2
            },
            'data_requirements': {
                'min_periods': 50,
                'lookback_periods': 20
            }
        }

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform comprehensive technical indicator analysis."""
        try:
            # Oscillator Analysis
            oscillator_analysis = self._analyze_oscillators(indicator_results)

            # Moving Average Analysis
            ma_analysis = self._analyze_moving_averages(indicator_results)

            # Momentum Analysis
            momentum_analysis = self._analyze_momentum_indicators(indicator_results)

            # Generate Decision
            decision = self._generate_indicator_decision(oscillator_analysis, ma_analysis, momentum_analysis)

            # Calculate Confidence
            confidence = self._calculate_indicator_confidence(oscillator_analysis, ma_analysis, momentum_analysis)

            # Generate Reasoning
            reasoning = self._generate_indicator_reasoning(decision, oscillator_analysis, ma_analysis, momentum_analysis)

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "oscillator_analysis": oscillator_analysis,
                    "ma_analysis": ma_analysis,
                    "momentum_analysis": momentum_analysis
                },
                risk_assessment=self._assess_indicator_risk(oscillator_analysis, ma_analysis),
                supporting_data={
                    "oscillator_signal": oscillator_analysis.get("overall_signal", "NEUTRAL"),
                    "ma_signal": ma_analysis.get("overall_signal", "NEUTRAL"),
                    "momentum_signal": momentum_analysis.get("overall_signal", "NEUTRAL")
                }
            )

        except Exception as e:
            logger.error(f"IndicatorExpert analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Indicator analysis failed: {str(e)}")

    def _analyze_oscillators(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze oscillator indicators."""
        oscillator_analysis = {
            "rsi_signal": "NEUTRAL",
            "stoch_signal": "NEUTRAL",
            "williams_r_signal": "NEUTRAL",
            "macd_signal": "NEUTRAL",
            "awesome_signal": "NEUTRAL",
            "overall_signal": "NEUTRAL",
            "divergence_detected": False,
            "overbought_count": 0,
            "oversold_count": 0
        }

        # RSI Analysis
        if "rsi_indicator" in indicator_results:
            rsi_data = indicator_results["rsi_indicator"]
            rsi_value = rsi_data.get("rsi", 50)

            if rsi_value >= self.rsi_overbought:
                oscillator_analysis["rsi_signal"] = "OVERBOUGHT"
                oscillator_analysis["overbought_count"] += 1
            elif rsi_value <= self.rsi_oversold:
                oscillator_analysis["rsi_signal"] = "OVERSOLD"
                oscillator_analysis["oversold_count"] += 1
            elif rsi_value > 55:
                oscillator_analysis["rsi_signal"] = "BULLISH"
            elif rsi_value < 45:
                oscillator_analysis["rsi_signal"] = "BEARISH"

        # Stochastic Analysis
        if "stochastic_oscillator_indicator" in indicator_results:
            stoch_data = indicator_results["stochastic_oscillator_indicator"]
            k_value = stoch_data.get("k", 50)
            d_value = stoch_data.get("d", 50)

            if k_value >= self.stoch_overbought and d_value >= self.stoch_overbought:
                oscillator_analysis["stoch_signal"] = "OVERBOUGHT"
                oscillator_analysis["overbought_count"] += 1
            elif k_value <= self.stoch_oversold and d_value <= self.stoch_oversold:
                oscillator_analysis["stoch_signal"] = "OVERSOLD"
                oscillator_analysis["oversold_count"] += 1
            elif k_value > d_value and k_value > 50:
                oscillator_analysis["stoch_signal"] = "BULLISH"
            elif k_value < d_value and k_value < 50:
                oscillator_analysis["stoch_signal"] = "BEARISH"

        # Williams %R Analysis
        if "williams_r_indicator" in indicator_results:
            wr_data = indicator_results["williams_r_indicator"]
            wr_value = wr_data.get("williams_r", -50)

            if wr_value >= self.williams_r_overbought:
                oscillator_analysis["williams_r_signal"] = "OVERBOUGHT"
                oscillator_analysis["overbought_count"] += 1
            elif wr_value <= self.williams_r_oversold:
                oscillator_analysis["williams_r_signal"] = "OVERSOLD"
                oscillator_analysis["oversold_count"] += 1

        # MACD Analysis
        if "macd_indicator" in indicator_results:
            macd_data = indicator_results["macd_indicator"]
            macd_line = macd_data.get("macd", 0)
            signal_line = macd_data.get("signal", 0)
            histogram = macd_data.get("histogram", 0)

            if macd_line > signal_line and histogram > 0:
                oscillator_analysis["macd_signal"] = "BULLISH"
            elif macd_line < signal_line and histogram < 0:
                oscillator_analysis["macd_signal"] = "BEARISH"

        # Overall oscillator signal
        oscillator_analysis["overall_signal"] = self._determine_overall_oscillator_signal(oscillator_analysis)

        return oscillator_analysis

    def _analyze_moving_averages(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze moving average indicators."""
        ma_analysis = {
            "dema_signal": "NEUTRAL",
            "tema_signal": "NEUTRAL",
            "hull_signal": "NEUTRAL",
            "kama_signal": "NEUTRAL",
            "overall_signal": "NEUTRAL",
            "trend_strength": "WEAK",
            "ma_alignment": "MIXED"
        }

        # DEMA Analysis
        if "double_exponential_moving_average_indicator" in indicator_results:
            dema_data = indicator_results["double_exponential_moving_average_indicator"]
            dema_trend = dema_data.get("trend", "NEUTRAL")
            ma_analysis["dema_signal"] = dema_trend

        # TEMA Analysis
        if "triple_exponential_moving_average_indicator" in indicator_results:
            tema_data = indicator_results["triple_exponential_moving_average_indicator"]
            tema_trend = tema_data.get("trend", "NEUTRAL")
            ma_analysis["tema_signal"] = tema_trend

        # Hull MA Analysis
        if "hull_moving_average_indicator" in indicator_results:
            hull_data = indicator_results["hull_moving_average_indicator"]
            hull_trend = hull_data.get("trend", "NEUTRAL")
            ma_analysis["hull_signal"] = hull_trend

        # KAMA Analysis
        if "kaufman_adaptive_moving_average_indicator" in indicator_results:
            kama_data = indicator_results["kaufman_adaptive_moving_average_indicator"]
            kama_trend = kama_data.get("trend", "NEUTRAL")
            ma_analysis["kama_signal"] = kama_trend

        # Determine overall MA signal
        ma_analysis["overall_signal"] = self._determine_overall_ma_signal(ma_analysis)

        return ma_analysis

    def _analyze_momentum_indicators(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        momentum_analysis = {
            "momentum_signal": "NEUTRAL",
            "roc_signal": "NEUTRAL",
            "proc_signal": "NEUTRAL",
            "ao_signal": "NEUTRAL",
            "overall_signal": "NEUTRAL",
            "momentum_strength": 0.0,
            "momentum_divergence": False
        }

        # Momentum Indicator
        if "momentum_indicator" in indicator_results:
            mom_data = indicator_results["momentum_indicator"]
            mom_value = mom_data.get("momentum", 100)

            if mom_value > 100:
                momentum_analysis["momentum_signal"] = "BULLISH"
            elif mom_value < 100:
                momentum_analysis["momentum_signal"] = "BEARISH"

        # Rate of Change
        if "rate_of_change_indicator" in indicator_results:
            roc_data = indicator_results["rate_of_change_indicator"]
            roc_value = roc_data.get("roc", 0)

            if roc_value > 2:
                momentum_analysis["roc_signal"] = "STRONG_BULLISH"
            elif roc_value > 0:
                momentum_analysis["roc_signal"] = "BULLISH"
            elif roc_value < -2:
                momentum_analysis["roc_signal"] = "STRONG_BEARISH"
            elif roc_value < 0:
                momentum_analysis["roc_signal"] = "BEARISH"

        # Awesome Oscillator
        if "awesome_oscillator_indicator" in indicator_results:
            ao_data = indicator_results["awesome_oscillator_indicator"]
            ao_value = ao_data.get("ao", 0)

            if ao_value > 0:
                momentum_analysis["ao_signal"] = "BULLISH"
            elif ao_value < 0:
                momentum_analysis["ao_signal"] = "BEARISH"

        # Calculate momentum strength
        momentum_analysis["momentum_strength"] = self._calculate_momentum_strength(momentum_analysis)

        # Determine overall momentum signal
        momentum_analysis["overall_signal"] = self._determine_overall_momentum_signal(momentum_analysis)

        return momentum_analysis

    def _determine_overall_oscillator_signal(self, oscillator_analysis: Dict[str, Any]) -> str:
        """Determine overall oscillator signal."""
        signals = [
            oscillator_analysis["rsi_signal"],
            oscillator_analysis["stoch_signal"],
            oscillator_analysis["williams_r_signal"],
            oscillator_analysis["macd_signal"]
        ]

        bullish_count = sum(1 for s in signals if "BULLISH" in s)
        bearish_count = sum(1 for s in signals if "BEARISH" in s)
        overbought_count = oscillator_analysis["overbought_count"]
        oversold_count = oscillator_analysis["oversold_count"]

        if overbought_count >= 2:
            return "OVERBOUGHT"
        elif oversold_count >= 2:
            return "OVERSOLD"
        elif bullish_count >= 3:
            return "BULLISH"
        elif bearish_count >= 3:
            return "BEARISH"

        return "NEUTRAL"

    def _determine_overall_ma_signal(self, ma_analysis: Dict[str, Any]) -> str:
        """Determine overall moving average signal."""
        signals = [
            ma_analysis["dema_signal"],
            ma_analysis["tema_signal"],
            ma_analysis["hull_signal"],
            ma_analysis["kama_signal"]
        ]

        bullish_count = sum(1 for s in signals if "BULLISH" in s)
        bearish_count = sum(1 for s in signals if "BEARISH" in s)

        if bullish_count >= 3:
            return "BULLISH"
        elif bearish_count >= 3:
            return "BEARISH"

        return "NEUTRAL"

    def _determine_overall_momentum_signal(self, momentum_analysis: Dict[str, Any]) -> str:
        """Determine overall momentum signal."""
        signals = [
            momentum_analysis["momentum_signal"],
            momentum_analysis["roc_signal"],
            momentum_analysis["ao_signal"]
        ]

        strong_bullish = sum(1 for s in signals if "STRONG_BULLISH" in s)
        bullish_count = sum(1 for s in signals if "BULLISH" in s)
        bearish_count = sum(1 for s in signals if "BEARISH" in s)

        if strong_bullish >= 1 and bullish_count >= 2:
            return "STRONG_BULLISH"
        elif bullish_count >= 2:
            return "BULLISH"
        elif bearish_count >= 2:
            return "BEARISH"

        return "NEUTRAL"

    def _calculate_momentum_strength(self, momentum_analysis: Dict[str, Any]) -> float:
        """Calculate overall momentum strength."""
        strength = 0.0
        signal_count = 0

        signals = [
            momentum_analysis["momentum_signal"],
            momentum_analysis["roc_signal"],
            momentum_analysis["ao_signal"]
        ]

        for signal in signals:
            if signal != "NEUTRAL":
                signal_count += 1
                if "STRONG" in signal:
                    strength += 0.8
                else:
                    strength += 0.5

        return strength / max(signal_count, 1)

    def _generate_indicator_decision(self, oscillator_analysis: Dict[str, Any], ma_analysis: Dict[str, Any], momentum_analysis: Dict[str, Any]) -> str:
        """Generate decision based on indicator analysis."""
        osc_signal = oscillator_analysis["overall_signal"]
        ma_signal = ma_analysis["overall_signal"]
        mom_signal = momentum_analysis["overall_signal"]

        # Strong confluence signals
        if osc_signal == "BULLISH" and ma_signal == "BULLISH" and mom_signal in ["BULLISH", "STRONG_BULLISH"]:
            return "BUY"
        elif osc_signal == "BEARISH" and ma_signal == "BEARISH" and mom_signal == "BEARISH":
            return "SELL"

        # Oversold/Overbought with momentum
        if osc_signal == "OVERSOLD" and mom_signal == "BULLISH":
            return "BUY"
        elif osc_signal == "OVERBOUGHT" and mom_signal == "BEARISH":
            return "SELL"

        # Strong momentum with MA support
        if mom_signal == "STRONG_BULLISH" and ma_signal == "BULLISH":
            return "BUY"

        return "NO_SIGNAL"

    def _calculate_indicator_confidence(self, oscillator_analysis: Dict[str, Any], ma_analysis: Dict[str, Any], momentum_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in indicator analysis."""
        factors = []

        # Oscillator confidence
        osc_signal = oscillator_analysis["overall_signal"]
        if osc_signal in ["OVERBOUGHT", "OVERSOLD"]:
            factors.append(0.3)
        elif osc_signal in ["BULLISH", "BEARISH"]:
            factors.append(0.2)

        # Moving average confidence
        ma_signal = ma_analysis["overall_signal"]
        if ma_signal in ["BULLISH", "BEARISH"]:
            factors.append(0.2)

        # Momentum confidence
        momentum_strength = momentum_analysis.get("momentum_strength", 0.0)
        factors.append(momentum_strength * 0.3)

        return min(sum(factors), 1.0)

    def _assess_indicator_risk(self, oscillator_analysis: Dict[str, Any], ma_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk based on indicator analysis."""
        risk_level = "MEDIUM"

        # High risk if overbought/oversold
        if oscillator_analysis["overall_signal"] in ["OVERBOUGHT", "OVERSOLD"]:
            risk_level = "HIGH"

        return {
            "indicator_risk": risk_level,
            "oscillator_risk": oscillator_analysis["overall_signal"],
            "trend_risk": ma_analysis["overall_signal"]
        }

    def _generate_indicator_reasoning(self, decision: str, oscillator_analysis: Dict[str, Any], ma_analysis: Dict[str, Any], momentum_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for indicator analysis."""
        parts = [f"Technical indicator analysis decision: {decision}."]

        osc_signal = oscillator_analysis["overall_signal"]
        ma_signal = ma_analysis["overall_signal"]
        mom_signal = momentum_analysis["overall_signal"]

        parts.append(f"Oscillators: {osc_signal}, Moving Averages: {ma_signal}, Momentum: {mom_signal}.")

        if oscillator_analysis["overbought_count"] >= 2:
            parts.append("Multiple overbought conditions detected.")
        elif oscillator_analysis["oversold_count"] >= 2:
            parts.append("Multiple oversold conditions detected.")

        return " ".join(parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        return [ind for ind in self.assigned_indicators if ind in indicator_results]

    def get_required_data_types(self) -> List[str]:
        """Define required data types."""
        return ["OHLCV"]

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed."""
        return 50  # Need sufficient data for technical indicators
    # ============== DATA-AWARE ENHANCEMENT METHODS ==============

    def _assess_data_availability(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess data availability and quality for enhanced indicator analysis."""
        data_availability = {
            "ohlcv_available": False,
            "volume_available": False,
            "data_depth": 0,
            "data_quality_score": 0.0,
            "indicator_reliability": "LOW",
            "timeframe_coverage": "SINGLE",
            "sufficient_for_divergence": False
        }

        total_score = 0.0
        max_score = 0.0

        # Check OHLCV data
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            data_availability["ohlcv_available"] = True
            ohlcv_data = market_data['OHLCV']
            data_availability["data_depth"] = len(ohlcv_data)

            # Quality score based on data depth for indicators
            if len(ohlcv_data) >= 200:  # Sufficient for most indicators
                total_score += 0.5
                data_availability["sufficient_for_divergence"] = True
            elif len(ohlcv_data) >= 50:  # Minimum for meaningful analysis
                total_score += 0.4
            elif len(ohlcv_data) >= 20:  # Basic analysis only
                total_score += 0.2

            # Volume data check
            if 'volume' in ohlcv_data.columns and not ohlcv_data['volume'].isna().all():
                data_availability["volume_available"] = True
                total_score += 0.3

        max_score += 0.8

        # Check for multiple timeframes (if available)
        timeframe_count = len([k for k in market_data.keys() if 'OHLCV' in k or k == 'OHLCV'])
        if timeframe_count > 1:
            data_availability["timeframe_coverage"] = "MULTIPLE"
            total_score += 0.2
        max_score += 0.2

        # Calculate quality score
        data_availability["data_quality_score"] = total_score / max_score if max_score > 0 else 0.0

        # Determine indicator reliability
        if data_availability["data_quality_score"] >= 0.8:
            data_availability["indicator_reliability"] = "HIGH"
        elif data_availability["data_quality_score"] >= 0.6:
            data_availability["indicator_reliability"] = "MEDIUM"
        else:
            data_availability["indicator_reliability"] = "LOW"

        return data_availability

    def _analyze_oscillators_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced oscillator analysis with data-driven optimizations."""
        # Base oscillator analysis
        oscillator_analysis = self._analyze_oscillators(indicator_results)

        # Enhanced analysis with data-driven features
        oscillator_analysis.update({
            "data_enhanced": True,
            "divergence_analysis": {},
            "multi_timeframe_confirmation": {},
            "adaptive_thresholds": {},
            "signal_strength_weighted": 0.0
        })

        # Divergence analysis (if sufficient data)
        if data_availability.get("sufficient_for_divergence", False):
            divergence_results = self._analyze_oscillator_divergences(indicator_results, market_data['OHLCV'])
            oscillator_analysis["divergence_analysis"] = divergence_results

        # Adaptive threshold optimization
        if data_availability.get("data_depth", 0) >= 100:
            adaptive_thresholds = self._calculate_adaptive_oscillator_thresholds(market_data['OHLCV'])
            oscillator_analysis["adaptive_thresholds"] = adaptive_thresholds

        # Signal strength weighting based on data quality
        data_quality = data_availability.get("data_quality_score", 0.0)
        base_signal_strength = oscillator_analysis.get("signal_strength", 0.5)
        oscillator_analysis["signal_strength_weighted"] = base_signal_strength * (0.5 + data_quality * 0.5)

        return oscillator_analysis

    def _analyze_moving_averages_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced moving average analysis with adaptive periods."""
        # Base MA analysis
        ma_analysis = self._analyze_moving_averages(indicator_results)

        # Enhanced features
        ma_analysis.update({
            "data_enhanced": True,
            "adaptive_periods": {},
            "trend_strength_analysis": {},
            "crossover_validation": {},
            "noise_filtered_signals": {}
        })

        # Adaptive period optimization
        if data_availability.get("data_depth", 0) >= 100:
            adaptive_periods = self._calculate_adaptive_ma_periods(market_data['OHLCV'])
            ma_analysis["adaptive_periods"] = adaptive_periods

        # Enhanced trend strength analysis
        if data_availability.get("ohlcv_available", False):
            trend_strength = self._analyze_trend_strength(market_data['OHLCV'])
            ma_analysis["trend_strength_analysis"] = trend_strength

        # Noise filtering based on volatility
        if data_availability.get("data_depth", 0) >= 50:
            noise_filtered = self._apply_noise_filtering(indicator_results, market_data['OHLCV'])
            ma_analysis["noise_filtered_signals"] = noise_filtered

        return ma_analysis

    def _analyze_momentum_indicators_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced momentum analysis with volume confirmation."""
        # Base momentum analysis
        momentum_analysis = self._analyze_momentum_indicators(indicator_results)

        # Enhanced features
        momentum_analysis.update({
            "data_enhanced": True,
            "volume_confirmation": {},
            "momentum_persistence": {},
            "acceleration_analysis": {},
            "regime_adjusted_momentum": {}
        })

        # Volume confirmation (if available)
        if data_availability.get("volume_available", False):
            volume_confirmation = self._analyze_volume_momentum_confirmation(indicator_results, market_data['OHLCV'])
            momentum_analysis["volume_confirmation"] = volume_confirmation

        # Momentum persistence analysis
        if data_availability.get("data_depth", 0) >= 60:
            persistence_analysis = self._analyze_momentum_persistence(market_data['OHLCV'])
            momentum_analysis["momentum_persistence"] = persistence_analysis

        # Market regime adjustment
        if data_availability.get("data_depth", 0) >= 100:
            regime_adjustment = self._calculate_regime_adjusted_momentum(market_data['OHLCV'])
            momentum_analysis["regime_adjusted_momentum"] = regime_adjustment

        return momentum_analysis
    def _analyze_oscillator_divergences(self, indicator_results: Dict[str, Any], ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze oscillator divergences with price action."""
        divergence_analysis = {
            "bullish_divergences": 0,
            "bearish_divergences": 0,
            "divergence_strength": 0.0,
            "divergences_detected": []
        }

        if len(ohlcv_data) < 50:
            return divergence_analysis

        # Simple divergence detection using RSI and price
        if "rsi_indicator" in indicator_results:
            rsi_values = indicator_results["rsi_indicator"].get("rsi_values", [])
            if len(rsi_values) >= 20:
                # Analyze recent trends
                price_trend = self._calculate_price_trend(ohlcv_data[-20:])
                rsi_trend = self._calculate_indicator_trend(rsi_values[-20:])

                # Detect divergences
                if price_trend > 0 and rsi_trend < -0.5:  # Price up, RSI down
                    divergence_analysis["bearish_divergences"] += 1
                    divergence_analysis["divergences_detected"].append("RSI_BEARISH_DIVERGENCE")
                elif price_trend < 0 and rsi_trend > 0.5:  # Price down, RSI up
                    divergence_analysis["bullish_divergences"] += 1
                    divergence_analysis["divergences_detected"].append("RSI_BULLISH_DIVERGENCE")

        # Calculate overall divergence strength
        total_divergences = divergence_analysis["bullish_divergences"] + divergence_analysis["bearish_divergences"]
        divergence_analysis["divergence_strength"] = min(1.0, total_divergences * 0.3)

        return divergence_analysis

    def _calculate_adaptive_oscillator_thresholds(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate adaptive thresholds for oscillators based on market conditions."""
        adaptive_thresholds = {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "adaptive_adjustment": 0.0,
            "volatility_factor": 1.0
        }

        if len(ohlcv_data) < 50:
            return adaptive_thresholds

        # Calculate market volatility
        returns = ohlcv_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Adjust thresholds based on volatility
        volatility_factor = min(2.0, max(0.5, volatility / 0.2))  # Normalize around 20% volatility
        adaptive_thresholds["volatility_factor"] = volatility_factor

        # Higher volatility = more extreme thresholds
        base_adjustment = (volatility_factor - 1.0) * 10
        adaptive_thresholds["adaptive_adjustment"] = base_adjustment
        adaptive_thresholds["rsi_overbought"] = min(90, 70 + base_adjustment)
        adaptive_thresholds["rsi_oversold"] = max(10, 30 - base_adjustment)

        return adaptive_thresholds

    def _calculate_adaptive_ma_periods(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate adaptive moving average periods based on market characteristics."""
        adaptive_periods = {
            "short_period": 10,
            "medium_period": 20,
            "long_period": 50,
            "optimal_periods": {},
            "market_cycle_length": 0
        }

        if len(ohlcv_data) < 100:
            return adaptive_periods

        # Estimate market cycle length
        returns = ohlcv_data['close'].pct_change().dropna()

        # Simple cycle detection using autocorrelation
        autocorrelations = []
        for lag in range(5, 50):
            if len(returns) > lag:
                autocorr = returns.autocorr(lag=lag)
                if not pd.isna(autocorr):
                    autocorrelations.append((lag, abs(autocorr)))

        if autocorrelations:
            # Find lag with highest autocorrelation
            best_lag = max(autocorrelations, key=lambda x: x[1])[0]
            adaptive_periods["market_cycle_length"] = best_lag

            # Adjust periods based on cycle length
            cycle_factor = best_lag / 20  # Normalize around 20-day cycle
            adaptive_periods["short_period"] = max(5, int(10 * cycle_factor))
            adaptive_periods["medium_period"] = max(10, int(20 * cycle_factor))
            adaptive_periods["long_period"] = max(20, int(50 * cycle_factor))

        return adaptive_periods

    def _analyze_trend_strength(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend strength using multiple metrics."""
        trend_analysis = {
            "trend_direction": "NEUTRAL",
            "trend_strength": 0.0,
            "trend_consistency": 0.0,
            "trend_acceleration": 0.0
        }

        if len(ohlcv_data) < 20:
            return trend_analysis

        # Simple trend analysis
        close_prices = ohlcv_data['close']

        # Linear regression slope for trend direction
        x = np.arange(len(close_prices))
        z = np.polyfit(x, close_prices, 1)
        slope = z[0]

        # Normalize slope by price level
        normalized_slope = slope / close_prices.mean() * len(close_prices)

        if normalized_slope > 0.02:
            trend_analysis["trend_direction"] = "BULLISH"
        elif normalized_slope < -0.02:
            trend_analysis["trend_direction"] = "BEARISH"

        # Trend strength based on R-squared
        y_pred = np.polyval(z, x)
        ss_res = np.sum((close_prices - y_pred) ** 2)
        ss_tot = np.sum((close_prices - np.mean(close_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        trend_analysis["trend_strength"] = max(0, r_squared)

        # Trend consistency (percentage of periods in trend direction)
        price_changes = close_prices.diff().dropna()
        if normalized_slope > 0:
            consistency = (price_changes > 0).mean()
        else:
            consistency = (price_changes < 0).mean()
        trend_analysis["trend_consistency"] = consistency

        return trend_analysis

    def _apply_noise_filtering(self, indicator_results: Dict[str, Any], ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply noise filtering to indicator signals."""
        noise_filtered = {
            "filtered_signals": {},
            "noise_level": 0.0,
            "signal_clarity": 0.0,
            "filtering_applied": False
        }

        if len(ohlcv_data) < 30:
            return noise_filtered

        # Calculate market noise level
        returns = ohlcv_data['close'].pct_change().dropna()
        noise_level = returns.std() / abs(returns.mean()) if returns.mean() != 0 else 1.0
        noise_filtered["noise_level"] = min(5.0, noise_level)

        # Apply filtering if noise is high
        if noise_level > 2.0:
            noise_filtered["filtering_applied"] = True

            # Simple filtering: require signal persistence
            for indicator_name in ["rsi_indicator", "macd_indicator"]:
                if indicator_name in indicator_results:
                    original_signal = indicator_results[indicator_name].get("signal", "NEUTRAL")
                    # In a real implementation, you would check signal persistence over multiple periods
                    filtered_signal = original_signal  # Placeholder for actual filtering logic
                    noise_filtered["filtered_signals"][indicator_name] = filtered_signal

        # Signal clarity assessment
        noise_filtered["signal_clarity"] = max(0.0, min(1.0, 1.0 - (noise_level - 1.0) * 0.2))

        return noise_filtered

    def _analyze_volume_momentum_confirmation(self, indicator_results: Dict[str, Any], ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume confirmation for momentum signals."""
        volume_confirmation = {
            "volume_trend": "NEUTRAL",
            "momentum_volume_alignment": False,
            "volume_momentum_score": 0.0,
            "confirmation_strength": 0.0
        }

        if 'volume' not in ohlcv_data.columns or len(ohlcv_data) < 20:
            return volume_confirmation

        # Volume trend analysis
        volume_data = ohlcv_data['volume']
        recent_volume = volume_data[-10:].mean()
        historical_volume = volume_data[:-10].mean() if len(volume_data) > 10 else recent_volume

        if recent_volume > historical_volume * 1.2:
            volume_confirmation["volume_trend"] = "INCREASING"
        elif recent_volume < historical_volume * 0.8:
            volume_confirmation["volume_trend"] = "DECREASING"

        # Check momentum-volume alignment
        momentum_signal = "NEUTRAL"
        if "momentum_indicator" in indicator_results:
            momentum_signal = indicator_results["momentum_indicator"].get("signal", "NEUTRAL")

        if momentum_signal in ["BUY", "BULLISH"] and volume_confirmation["volume_trend"] == "INCREASING":
            volume_confirmation["momentum_volume_alignment"] = True
            volume_confirmation["confirmation_strength"] = 0.8
        elif momentum_signal in ["SELL", "BEARISH"] and volume_confirmation["volume_trend"] == "INCREASING":
            volume_confirmation["momentum_volume_alignment"] = True
            volume_confirmation["confirmation_strength"] = 0.8

        return volume_confirmation

    def _analyze_momentum_persistence(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum persistence characteristics."""
        persistence_analysis = {
            "momentum_half_life": 0,
            "persistence_score": 0.0,
            "momentum_consistency": 0.0,
            "mean_reversion_tendency": 0.0
        }

        if len(ohlcv_data) < 30:
            return persistence_analysis

        # Calculate momentum using price changes
        returns = ohlcv_data['close'].pct_change().dropna()

        # Simple momentum persistence using autocorrelation
        momentum_autocorr = returns.autocorr(lag=1) if len(returns) > 1 else 0.0
        persistence_analysis["persistence_score"] = max(0, momentum_autocorr)

        # Momentum consistency (what percentage of time momentum persists)
        momentum_signals = returns > 0
        persistence_count = 0
        for i in range(1, len(momentum_signals)):
            if momentum_signals.iloc[i] == momentum_signals.iloc[i-1]:
                persistence_count += 1

        persistence_analysis["momentum_consistency"] = persistence_count / (len(momentum_signals) - 1) if len(momentum_signals) > 1 else 0.0

        # Mean reversion tendency
        persistence_analysis["mean_reversion_tendency"] = 1.0 - persistence_analysis["persistence_score"]

        return persistence_analysis

    def _calculate_regime_adjusted_momentum(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum adjusted for market regime."""
        regime_adjustment = {
            "current_regime": "NORMAL",
            "regime_momentum_factor": 1.0,
            "volatility_regime": "NORMAL",
            "adjusted_momentum_threshold": 0.5
        }

        if len(ohlcv_data) < 50:
            return regime_adjustment

        # Volatility regime classification
        returns = ohlcv_data['close'].pct_change().dropna()
        recent_volatility = returns[-20:].std() if len(returns) >= 20 else returns.std()
        historical_volatility = returns.std()

        if recent_volatility > historical_volatility * 1.5:
            regime_adjustment["volatility_regime"] = "HIGH_VOLATILITY"
            regime_adjustment["regime_momentum_factor"] = 0.7  # Reduce momentum sensitivity
        elif recent_volatility < historical_volatility * 0.5:
            regime_adjustment["volatility_regime"] = "LOW_VOLATILITY"
            regime_adjustment["regime_momentum_factor"] = 1.3  # Increase momentum sensitivity

        # Adjust momentum thresholds based on regime
        base_threshold = 0.5
        regime_adjustment["adjusted_momentum_threshold"] = base_threshold * regime_adjustment["regime_momentum_factor"]

        return regime_adjustment

    def _calculate_price_trend(self, price_data: pd.DataFrame) -> float:
        """Calculate simple price trend slope."""
        if len(price_data) < 2:
            return 0.0

        close_prices = price_data['close']
        x = np.arange(len(close_prices))
        z = np.polyfit(x, close_prices, 1)
        return z[0] / close_prices.mean()  # Normalized slope

    def _calculate_indicator_trend(self, indicator_values: List[float]) -> float:
        """Calculate trend in indicator values."""
        if len(indicator_values) < 2:
            return 0.0

        x = np.arange(len(indicator_values))
        z = np.polyfit(x, indicator_values, 1)
        return z[0]  # Raw slope for indicators
