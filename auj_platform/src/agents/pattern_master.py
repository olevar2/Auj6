"""
Pattern Master Agent for the AUJ Platform.

This agent specializes in all forms of pattern recognition including candlestick patterns,
chart patterns, and wave analysis. It focuses on 37 pattern-related indicators.
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

logger = get_logger(__name__)


class PatternMaster(BaseAgent):
    """
    Pattern Master Agent - All forms of pattern recognition.

    Specializes in:
    - Candlestick pattern recognition
    - Chart pattern identification
    - Elliott Wave analysis
    - Fractal pattern detection
    """

    def __init__(self, config_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize the Pattern Master Agent."""
        # Load configuration first
        self.config = config or self._load_agent_config()
        self.config_manager = config_manager

        # Define assigned indicators for this agent (from registry)
        assigned_indicators = [
            # ALL Candlestick Patterns (29)
            "abandoned_baby_indicator",
            "belt_hold_indicator",
            "dark_cloud_cover_indicator",
            "doji_indicator",
            "doji_star_indicator",
            "dragonfly_doji_indicator",
            "engulfing_pattern_indicator",
            "evening_star_indicator",
            "gravestone_doji_indicator",
            "hammer_indicator",
            "hanging_man_indicator",
            "harami_cross_indicator",
            "harami_indicator",
            "head_and_shoulders_indicator",
            "high_wave_candle_indicator",
            "inverted_hammer_indicator",
            "kicking_indicator",
            "long_legged_doji_indicator",
            "marubozu_indicator",
            "morning_star_indicator",
            "piercing_line_indicator",
            "rickshaw_man_indicator",
            "rising_three_methods_indicator",
            "shooting_star_indicator",
            "spinning_top_indicator",
            "tasuki_gap_indicator",
            "three_black_crows_indicator",
            "three_line_strike_indicator",
            "three_white_soldiers_indicator",

            # Chart Patterns (2)
            "triangle_pattern_indicator",
            "wedge_pattern_indicator",

            # ALL Elliott Wave Indicators (6)
            "elliott_wave_oscillator_indicator",
            "fractal_wave_counter_indicator",
            "impulsive_corrective_classifier_indicator",
            "photonic_wavelength_analyzer",
            "wave_point_indicator",
            "wave_structure_indicator",

            # ALL Fractal Indicators (10)
            "chaos_fractal_dimension_indicator",
            "fractal_adaptive_moving_average_indicator",
            "fractal_breakout_indicator",
            "fractal_channel_indicator",
            "fractal_chaos_oscillator_indicator",
            "fractal_correlation_dimension_indicator",
            "fractal_efficiency_ratio_indicator",
            "fractal_market_hypothesis_indicator",
            "mandelbrot_fractal_indicator",
            "multi_fractal_dfa_indicator",

            # ALL Fibonacci Indicators (11)
            "fibonacci_arcs_indicator",
            "fibonacci_channel_indicator",
            "fibonacci_clusters_indicator",
            "fibonacci_extension_indicator",
            "fibonacci_fan_indicator",
            "fibonacci_retracement_indicator",
            "fibonacci_spirals_indicator",
            "fibonacci_time_extension_indicator",
            "fibonacci_time_zone_indicator",
            "projection_arc_calculator_indicator",
            "time_zone_analysis_indicator"
        ]

        super().__init__(
            name="PatternMaster",
            specialization="All forms of pattern recognition and wave analysis",
            assigned_indicators=assigned_indicators,
            config_manager=config_manager,
            config=self.config
        )

        # Load configuration from YAML file
        self._load_agent_config()

        # Pattern categories
        self.bullish_reversal_patterns = [
            "hammer_indicator", "inverted_hammer_indicator", "dragonfly_doji_indicator",
            "morning_star_indicator", "piercing_line_indicator", "abandoned_baby_indicator"
        ]

        self.bearish_reversal_patterns = [
            "hanging_man_indicator", "shooting_star_indicator", "gravestone_doji_indicator",
            "evening_star_indicator", "dark_cloud_cover_indicator", "belt_hold_indicator"
        ]

        self.continuation_patterns = [
            "rising_three_methods_indicator", "three_line_strike_indicator",
            "tasuki_gap_indicator", "marubozu_indicator"
        ]

        # Pattern analysis weights (from registry configuration)
        self.candlestick_weight = 0.4
        self.chart_pattern_weight = 0.35
        self.wave_analysis_weight = 0.25
        self.min_pattern_confluence = 2
        self.pattern_confidence_threshold = 0.7

        logger.info(f"PatternMaster initialized with {len(assigned_indicators)} pattern indicators")

    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML file with fallback defaults."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'pattern_master.yaml'
            )

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded PatternMaster configuration from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading PatternMaster configuration: {e}")

        # Fallback defaults
        return {
            'pattern_confidence': {
                'high_confidence_threshold': 0.8,
                'medium_confidence_threshold': 0.6,
                'low_confidence_threshold': 0.4,
                'reversal_pattern_weight': 1.2,
                'continuation_pattern_weight': 1.0
            },
            'data_quality': {
                'min_candles_required': 50,
                'min_volume_threshold': 1000,
                'data_freshness_minutes': 30
            },
            'elliott_wave': {
                'min_wave_length': 5,
                'fibonacci_tolerance': 0.1,
                'wave_completion_threshold': 0.7
            },
            'fractal_analysis': {
                'fractal_period': 5,
                'support_resistance_strength': 3,
                'breakout_confirmation_pips': 10
            }
        }

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """
        Perform comprehensive data-aware pattern analysis.

        Args:
            symbol: Trading symbol
            market_data: Available market data
            market_conditions: Current market conditions
            indicator_results: Calculated indicator values

        Returns:
            AnalysisResult with pattern analysis and recommendations
        """
        try:
            # Data-aware pattern assessment
            data_assessment = self._assess_pattern_data_quality(market_data)

            # 1. Enhanced Candlestick Pattern Analysis
            candlestick_analysis = self._analyze_candlestick_patterns_enhanced(indicator_results, data_assessment)

            # 2. Data-Aware Chart Pattern Analysis
            chart_pattern_analysis = self._analyze_chart_patterns_enhanced(indicator_results, market_data, data_assessment)

            # 3. Enhanced Wave Analysis
            wave_analysis = self._analyze_wave_patterns_enhanced(indicator_results, data_assessment)

            # 4. Data-Quality Weighted Pattern Confluence
            confluence_analysis = self._analyze_pattern_confluence_enhanced(
                candlestick_analysis, chart_pattern_analysis, wave_analysis, data_assessment
            )

            # 5. Generate Enhanced Pattern Decision
            decision = self._generate_pattern_decision_enhanced(
                confluence_analysis, market_conditions, data_assessment
            )

            # 6. Calculate Data-Adjusted Pattern Confidence
            confidence = self._calculate_pattern_confidence_enhanced(
                candlestick_analysis, chart_pattern_analysis, wave_analysis, confluence_analysis, data_assessment
            )

            # 7. Generate Enhanced Pattern Reasoning
            reasoning = self._generate_pattern_reasoning_enhanced(
                decision, candlestick_analysis, chart_pattern_analysis, wave_analysis, confluence_analysis, data_assessment
            )

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "candlestick_analysis": candlestick_analysis,
                    "chart_pattern_analysis": chart_pattern_analysis,
                    "wave_analysis": wave_analysis,
                    "confluence_analysis": confluence_analysis,
                    "data_assessment": data_assessment
                },
                risk_assessment=self._assess_pattern_risk_enhanced(confluence_analysis, market_conditions, data_assessment),
                supporting_data={
                    "strongest_pattern": confluence_analysis.get("strongest_pattern", "NONE"),
                    "pattern_strength": confluence_analysis.get("overall_strength", 0.0),
                    "reversal_probability": confluence_analysis.get("reversal_probability", 0.0),
                    "data_quality_score": data_assessment.get("quality_score", 0.0)
                }
            )

        except Exception as e:
            logger.error(f"PatternMaster analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Pattern analysis failed: {str(e)}")
    def _analyze_candlestick_patterns(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze candlestick patterns for reversal and continuation signals."""
        candlestick_analysis = {
            "bullish_reversal_signals": [],
            "bearish_reversal_signals": [],
            "continuation_signals": [],
            "doji_patterns": [],
            "strongest_candlestick": "NONE",
            "candlestick_strength": 0.0,
            "reversal_indication": "NEUTRAL"
        }

        # Analyze bullish reversal patterns
        for pattern in self.bullish_reversal_patterns:
            if pattern in indicator_results:
                pattern_data = indicator_results[pattern]
                if pattern_data.get("signal", "NEUTRAL") in ["BUY", "BULLISH"]:
                    strength = pattern_data.get("strength", 0.5)
                    candlestick_analysis["bullish_reversal_signals"].append({
                        "pattern": pattern,
                        "strength": strength,
                        "reliability": pattern_data.get("reliability", 0.5)
                    })

        # Analyze bearish reversal patterns
        for pattern in self.bearish_reversal_patterns:
            if pattern in indicator_results:
                pattern_data = indicator_results[pattern]
                if pattern_data.get("signal", "NEUTRAL") in ["SELL", "BEARISH"]:
                    strength = pattern_data.get("strength", 0.5)
                    candlestick_analysis["bearish_reversal_signals"].append({
                        "pattern": pattern,
                        "strength": strength,
                        "reliability": pattern_data.get("reliability", 0.5)
                    })

        # Analyze continuation patterns
        for pattern in self.continuation_patterns:
            if pattern in indicator_results:
                pattern_data = indicator_results[pattern]
                if pattern_data.get("signal", "NEUTRAL") != "NEUTRAL":
                    strength = pattern_data.get("strength", 0.5)
                    candlestick_analysis["continuation_signals"].append({
                        "pattern": pattern,
                        "signal": pattern_data.get("signal", "NEUTRAL"),
                        "strength": strength
                    })

        # Analyze Doji patterns
        doji_patterns = ["doji_indicator", "doji_star_indicator", "dragonfly_doji_indicator",
                        "gravestone_doji_indicator", "long_legged_doji_indicator", "rickshaw_man_indicator"]

        for doji_pattern in doji_patterns:
            if doji_pattern in indicator_results:
                doji_data = indicator_results[doji_pattern]
                if doji_data.get("detected", False):
                    candlestick_analysis["doji_patterns"].append({
                        "type": doji_pattern,
                        "strength": doji_data.get("strength", 0.5),
                        "significance": doji_data.get("significance", "MEDIUM")
                    })

        # Determine strongest pattern
        all_patterns = (candlestick_analysis["bullish_reversal_signals"] +
                       candlestick_analysis["bearish_reversal_signals"] +
                       candlestick_analysis["continuation_signals"])

        if all_patterns:
            strongest = max(all_patterns, key=lambda x: x.get("strength", 0))
            candlestick_analysis["strongest_candlestick"] = strongest["pattern"]
            candlestick_analysis["candlestick_strength"] = strongest["strength"]

        # Determine overall reversal indication
        bullish_strength = sum(p["strength"] for p in candlestick_analysis["bullish_reversal_signals"])
        bearish_strength = sum(p["strength"] for p in candlestick_analysis["bearish_reversal_signals"])

        if bullish_strength > bearish_strength and bullish_strength > 0.6:
            candlestick_analysis["reversal_indication"] = "BULLISH_REVERSAL"
        elif bearish_strength > bullish_strength and bearish_strength > 0.6:
            candlestick_analysis["reversal_indication"] = "BEARISH_REVERSAL"
        elif len(candlestick_analysis["doji_patterns"]) > 0:
            candlestick_analysis["reversal_indication"] = "INDECISION"

        return candlestick_analysis

    def _analyze_chart_patterns(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze chart patterns for breakout and reversal signals."""
        chart_analysis = {
            "head_and_shoulders": "NONE",
            "triangle_patterns": [],
            "wedge_patterns": [],
            "three_pattern_series": [],
            "chart_pattern_strength": 0.0,
            "breakout_probability": 0.0,
            "target_projection": None
        }

        # Head and Shoulders Analysis
        if "head_and_shoulders_indicator" in indicator_results:
            hs_data = indicator_results["head_and_shoulders_indicator"]
            pattern_type = hs_data.get("pattern_type", "NONE")
            if pattern_type in ["HEAD_AND_SHOULDERS", "INVERSE_HEAD_AND_SHOULDERS"]:
                chart_analysis["head_and_shoulders"] = pattern_type
                chart_analysis["chart_pattern_strength"] = hs_data.get("strength", 0.0)
                chart_analysis["target_projection"] = hs_data.get("target_price", None)

        # Triangle Pattern Analysis
        if "triangle_pattern_indicator" in indicator_results:
            triangle_data = indicator_results["triangle_pattern_indicator"]
            triangle_type = triangle_data.get("triangle_type", "NONE")
            if triangle_type != "NONE":
                chart_analysis["triangle_patterns"].append({
                    "type": triangle_type,
                    "completion": triangle_data.get("completion_percentage", 0),
                    "breakout_direction": triangle_data.get("expected_breakout", "UNKNOWN"),
                    "strength": triangle_data.get("strength", 0.5)
                })

        # Wedge Pattern Analysis
        if "wedge_pattern_indicator" in indicator_results:
            wedge_data = indicator_results["wedge_pattern_indicator"]
            wedge_type = wedge_data.get("wedge_type", "NONE")
            if wedge_type != "NONE":
                chart_analysis["wedge_patterns"].append({
                    "type": wedge_type,
                    "completion": wedge_data.get("completion_percentage", 0),
                    "reversal_probability": wedge_data.get("reversal_probability", 0.5),
                    "strength": wedge_data.get("strength", 0.5)
                })

        # Three-Pattern Series Analysis
        three_patterns = ["three_black_crows_indicator", "three_white_soldiers_indicator", "three_line_strike_indicator"]
        for pattern in three_patterns:
            if pattern in indicator_results:
                pattern_data = indicator_results[pattern]
                if pattern_data.get("detected", False):
                    chart_analysis["three_pattern_series"].append({
                        "pattern": pattern,
                        "strength": pattern_data.get("strength", 0.5),
                        "signal": pattern_data.get("signal", "NEUTRAL"),
                        "reliability": pattern_data.get("reliability", 0.6)
                    })

        # Calculate overall chart pattern strength
        pattern_strengths = []

        if chart_analysis["head_and_shoulders"] != "NONE":
            pattern_strengths.append(chart_analysis["chart_pattern_strength"])

        for triangle in chart_analysis["triangle_patterns"]:
            pattern_strengths.append(triangle["strength"])

        for wedge in chart_analysis["wedge_patterns"]:
            pattern_strengths.append(wedge["strength"])

        for three_pattern in chart_analysis["three_pattern_series"]:
            pattern_strengths.append(three_pattern["strength"])

        if pattern_strengths:
            chart_analysis["chart_pattern_strength"] = max(pattern_strengths)
            chart_analysis["breakout_probability"] = sum(pattern_strengths) / len(pattern_strengths)

        return chart_analysis

    def _analyze_wave_patterns(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze wave patterns including Elliott Wave analysis."""
        wave_analysis = {
            "elliott_wave_position": "UNKNOWN",
            "wave_structure": "UNDEFINED",
            "impulse_correction": "NEUTRAL",
            "wave_targets": [],
            "fractal_waves": [],
            "wave_strength": 0.0,
            "next_wave_direction": "UNKNOWN"
        }

        # Elliott Wave Analysis
        if "elliott_wave_oscillator_indicator" in indicator_results:
            ew_data = indicator_results["elliott_wave_oscillator_indicator"]
            wave_analysis["elliott_wave_position"] = ew_data.get("current_wave", "UNKNOWN")
            wave_analysis["wave_strength"] = ew_data.get("wave_strength", 0.0)
            wave_analysis["next_wave_direction"] = ew_data.get("next_wave_direction", "UNKNOWN")

        # Wave Structure Analysis
        if "wave_structure_indicator" in indicator_results:
            ws_data = indicator_results["wave_structure_indicator"]
            wave_analysis["wave_structure"] = ws_data.get("structure_type", "UNDEFINED")
            wave_analysis["wave_targets"] = ws_data.get("price_targets", [])

        # Impulse/Corrective Classification
        if "impulsive_corrective_classifier_indicator" in indicator_results:
            ic_data = indicator_results["impulsive_corrective_classifier_indicator"]
            wave_analysis["impulse_correction"] = ic_data.get("classification", "NEUTRAL")

        # Fractal Wave Analysis
        if "fractal_wave_counter_indicator" in indicator_results:
            fw_data = indicator_results["fractal_wave_counter_indicator"]
            wave_analysis["fractal_waves"] = fw_data.get("detected_waves", [])

        # Wave Points
        if "wave_point_indicator" in indicator_results:
            wp_data = indicator_results["wave_point_indicator"]
            significant_points = wp_data.get("significant_points", [])
            if significant_points:
                wave_analysis["wave_strength"] = max(wave_analysis["wave_strength"],
                                                   wp_data.get("strength", 0.0))

        return wave_analysis
    def _analyze_pattern_confluence(self,
                                  candlestick_analysis: Dict[str, Any],
                                  chart_pattern_analysis: Dict[str, Any],
                                  wave_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confluence between different pattern types."""
        confluence_analysis = {
            "pattern_count": 0,
            "bullish_patterns": 0,
            "bearish_patterns": 0,
            "overall_direction": "NEUTRAL",
            "overall_strength": 0.0,
            "strongest_pattern": "NONE",
            "confluence_quality": "WEAK",
            "reversal_probability": 0.0,
            "continuation_probability": 0.0
        }

        pattern_signals = []

        # Candlestick patterns contribution
        candlestick_weight = self.candlestick_weight
        if candlestick_analysis["reversal_indication"] == "BULLISH_REVERSAL":
            pattern_signals.append(("BULLISH", candlestick_analysis["candlestick_strength"] * candlestick_weight))
            confluence_analysis["bullish_patterns"] += 1
        elif candlestick_analysis["reversal_indication"] == "BEARISH_REVERSAL":
            pattern_signals.append(("BEARISH", candlestick_analysis["candlestick_strength"] * candlestick_weight))
            confluence_analysis["bearish_patterns"] += 1
        elif candlestick_analysis["reversal_indication"] == "INDECISION":
            pattern_signals.append(("NEUTRAL", candlestick_analysis["candlestick_strength"] * candlestick_weight * 0.5))

        # Chart patterns contribution
        chart_weight = self.chart_pattern_weight
        chart_strength = chart_pattern_analysis["chart_pattern_strength"]

        # Head and shoulders
        if chart_pattern_analysis["head_and_shoulders"] == "HEAD_AND_SHOULDERS":
            pattern_signals.append(("BEARISH", chart_strength * chart_weight))
            confluence_analysis["bearish_patterns"] += 1
        elif chart_pattern_analysis["head_and_shoulders"] == "INVERSE_HEAD_AND_SHOULDERS":
            pattern_signals.append(("BULLISH", chart_strength * chart_weight))
            confluence_analysis["bullish_patterns"] += 1

        # Triangle patterns
        for triangle in chart_pattern_analysis["triangle_patterns"]:
            if triangle["breakout_direction"] == "BULLISH":
                pattern_signals.append(("BULLISH", triangle["strength"] * chart_weight))
                confluence_analysis["bullish_patterns"] += 1
            elif triangle["breakout_direction"] == "BEARISH":
                pattern_signals.append(("BEARISH", triangle["strength"] * chart_weight))
                confluence_analysis["bearish_patterns"] += 1

        # Wave analysis contribution
        wave_weight = self.wave_analysis_weight
        wave_direction = wave_analysis["next_wave_direction"]
        wave_strength = wave_analysis["wave_strength"]

        if wave_direction == "UP" or wave_direction == "BULLISH":
            pattern_signals.append(("BULLISH", wave_strength * wave_weight))
            confluence_analysis["bullish_patterns"] += 1
        elif wave_direction == "DOWN" or wave_direction == "BEARISH":
            pattern_signals.append(("BEARISH", wave_strength * wave_weight))
            confluence_analysis["bearish_patterns"] += 1

        # Calculate overall metrics
        confluence_analysis["pattern_count"] = len(pattern_signals)

        if pattern_signals:
            # Calculate weighted direction
            bullish_weight = sum(strength for direction, strength in pattern_signals if direction == "BULLISH")
            bearish_weight = sum(strength for direction, strength in pattern_signals if direction == "BEARISH")

            if bullish_weight > bearish_weight and confluence_analysis["bullish_patterns"] >= self.min_pattern_confluence:
                confluence_analysis["overall_direction"] = "BULLISH"
            elif bearish_weight > bullish_weight and confluence_analysis["bearish_patterns"] >= self.min_pattern_confluence:
                confluence_analysis["overall_direction"] = "BEARISH"

            # Overall strength
            confluence_analysis["overall_strength"] = max(bullish_weight, bearish_weight)

            # Determine strongest pattern category
            if candlestick_analysis["candlestick_strength"] >= chart_strength and candlestick_analysis["candlestick_strength"] >= wave_strength:
                confluence_analysis["strongest_pattern"] = "CANDLESTICK"
            elif chart_strength >= wave_strength:
                confluence_analysis["strongest_pattern"] = "CHART_PATTERN"
            else:
                confluence_analysis["strongest_pattern"] = "WAVE_PATTERN"

            # Confluence quality
            high_threshold = self.config['pattern_confidence']['high_confidence_threshold']
            medium_threshold = self.config['pattern_confidence']['medium_confidence_threshold']
            low_threshold = self.config['pattern_confidence']['low_confidence_threshold']

            if confluence_analysis["overall_strength"] >= high_threshold and confluence_analysis["pattern_count"] >= 3:
                confluence_analysis["confluence_quality"] = "VERY_STRONG"
            elif confluence_analysis["overall_strength"] >= medium_threshold and confluence_analysis["pattern_count"] >= 2:
                confluence_analysis["confluence_quality"] = "STRONG"
            elif confluence_analysis["overall_strength"] >= low_threshold:
                confluence_analysis["confluence_quality"] = "MODERATE"

            # Calculate probabilities
            total_patterns = confluence_analysis["bullish_patterns"] + confluence_analysis["bearish_patterns"]
            if total_patterns > 0:
                if confluence_analysis["overall_direction"] != "NEUTRAL":
                    confluence_analysis["reversal_probability"] = confluence_analysis["overall_strength"]
                else:
                    confluence_analysis["continuation_probability"] = confluence_analysis["overall_strength"] * medium_threshold

        return confluence_analysis

    def _generate_pattern_decision(self,
                                 confluence_analysis: Dict[str, Any],
                                 market_conditions: MarketConditions) -> str:
        """Generate trading decision based on pattern analysis."""
        overall_direction = confluence_analysis["overall_direction"]
        confluence_quality = confluence_analysis["confluence_quality"]
        pattern_count = confluence_analysis["pattern_count"]

        # Require minimum confluence for signals
        if confluence_quality == "WEAK" or pattern_count < self.min_pattern_confluence:
            return "NO_SIGNAL"

        # Strong confluence required for reversal signals
        if overall_direction == "BULLISH":
            if confluence_quality in ["STRONG", "VERY_STRONG"]:
                return "BUY"
            elif confluence_quality == "MODERATE" and confluence_analysis["overall_strength"] > 0.6:
                return "BUY"
        elif overall_direction == "BEARISH":
            if confluence_quality in ["STRONG", "VERY_STRONG"]:
                return "SELL"
            elif confluence_quality == "MODERATE" and confluence_analysis["overall_strength"] > 0.6:
                return "SELL"

        # Consider market regime
        regime = market_conditions.regime.value
        if regime in ["SIDEWAYS", "ACCUMULATION", "DISTRIBUTION"]:
            # In sideways markets, require higher confluence
            if confluence_quality == "VERY_STRONG":
                return "BUY" if overall_direction == "BULLISH" else "SELL" if overall_direction == "BEARISH" else "HOLD"

        return "HOLD"

    def _calculate_pattern_confidence(self,
                                    candlestick_analysis: Dict[str, Any],
                                    chart_pattern_analysis: Dict[str, Any],
                                    wave_analysis: Dict[str, Any],
                                    confluence_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in pattern analysis."""
        confidence_factors = []

        # Pattern count factor (25% weight)
        pattern_count = confluence_analysis["pattern_count"]
        count_confidence = min(pattern_count / 4.0, 1.0)  # Max confidence at 4+ patterns
        confidence_factors.append(count_confidence * 0.25)

        # Confluence strength factor (40% weight)
        confluence_strength = confluence_analysis["overall_strength"]
        confidence_factors.append(confluence_strength * 0.4)

        # Individual pattern strengths (35% weight)
        individual_strengths = []
        if candlestick_analysis["candlestick_strength"] > 0:
            individual_strengths.append(candlestick_analysis["candlestick_strength"])
        if chart_pattern_analysis["chart_pattern_strength"] > 0:
            individual_strengths.append(chart_pattern_analysis["chart_pattern_strength"])
        if wave_analysis["wave_strength"] > 0:
            individual_strengths.append(wave_analysis["wave_strength"])

        if individual_strengths:
            avg_individual_strength = sum(individual_strengths) / len(individual_strengths)
            confidence_factors.append(avg_individual_strength * 0.35)
        else:
            confidence_factors.append(0.0)

        return min(sum(confidence_factors), 1.0)

    def _assess_pattern_risk(self, confluence_analysis: Dict[str, Any], market_conditions: MarketConditions) -> Dict[str, Any]:
        """Assess risk associated with pattern-based trading."""
        risk_assessment = {
            "pattern_reliability": "MEDIUM",
            "false_breakout_risk": "MEDIUM",
            "pattern_failure_risk": 0.3,
            "recommended_stop_distance": 2.0,
            "risk_reward_ratio": 2.0
        }

        confluence_quality = confluence_analysis["confluence_quality"]
        pattern_count = confluence_analysis["pattern_count"]

        # Pattern reliability assessment
        if confluence_quality == "VERY_STRONG" and pattern_count >= 3:
            risk_assessment["pattern_reliability"] = "HIGH"
            risk_assessment["pattern_failure_risk"] = 0.15
            risk_assessment["recommended_stop_distance"] = 1.5
        elif confluence_quality == "WEAK" or pattern_count < 2:
            risk_assessment["pattern_reliability"] = "LOW"
            risk_assessment["pattern_failure_risk"] = 0.5
            risk_assessment["recommended_stop_distance"] = 3.0

        # False breakout risk based on market conditions
        regime = market_conditions.regime.value
        if regime in ["HIGH_VOLATILITY", "SIDEWAYS"]:
            risk_assessment["false_breakout_risk"] = "HIGH"
            risk_assessment["recommended_stop_distance"] *= 1.5
        elif regime in ["TRENDING_UP", "TRENDING_DOWN"]:
            risk_assessment["false_breakout_risk"] = "LOW"

        return risk_assessment

    def _generate_pattern_reasoning(self,
                                  decision: str,
                                  candlestick_analysis: Dict[str, Any],
                                  chart_pattern_analysis: Dict[str, Any],
                                  wave_analysis: Dict[str, Any],
                                  confluence_analysis: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for pattern analysis."""
        reasoning_parts = []

        # Decision explanation
        if decision in ["BUY", "SELL"]:
            direction = "bullish" if decision == "BUY" else "bearish"
            reasoning_parts.append(f"Pattern analysis indicates {direction} signal based on confluence of multiple patterns.")
        elif decision == "HOLD":
            reasoning_parts.append("Pattern analysis suggests holding position due to moderate confluence.")
        else:
            reasoning_parts.append("No clear pattern signal identified.")

        # Confluence details
        pattern_count = confluence_analysis["pattern_count"]
        confluence_quality = confluence_analysis["confluence_quality"]
        reasoning_parts.append(f"Confluence analysis: {pattern_count} patterns with {confluence_quality} agreement.")

        # Strongest pattern category
        strongest_pattern = confluence_analysis["strongest_pattern"]
        if strongest_pattern != "NONE":
            reasoning_parts.append(f"Dominant pattern type: {strongest_pattern}.")

        # Specific pattern details
        if candlestick_analysis["reversal_indication"] != "NEUTRAL":
            indication = candlestick_analysis["reversal_indication"].replace("_", " ").lower()
            reasoning_parts.append(f"Candlestick analysis shows {indication}.")

        if chart_pattern_analysis["head_and_shoulders"] != "NONE":
            hs_pattern = chart_pattern_analysis["head_and_shoulders"].replace("_", " ").lower()
            reasoning_parts.append(f"Chart pattern: {hs_pattern} detected.")

        if wave_analysis["elliott_wave_position"] != "UNKNOWN":
            wave_pos = wave_analysis["elliott_wave_position"]
            reasoning_parts.append(f"Elliott Wave position: {wave_pos}.")

        return " ".join(reasoning_parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        used_indicators = []
        for indicator in self.assigned_indicators:
            if indicator in indicator_results:
                used_indicators.append(indicator)
        return used_indicators

    def get_required_data_types(self) -> List[str]:
        """Define required data types for pattern analysis."""
        return ["OHLCV"]  # Primary focus on price action for pattern recognition

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed for pattern analysis."""
        return 100  # Need substantial history for pattern recognition

    def get_pattern_parameters(self) -> Dict[str, Any]:
        """Get current pattern analysis parameters."""
        return {
            "pattern_confidence_threshold": self.pattern_confidence_threshold,
            "candlestick_weight": self.candlestick_weight,
            "chart_pattern_weight": self.chart_pattern_weight,
            "wave_analysis_weight": self.wave_analysis_weight,
            "min_pattern_confluence": self.min_pattern_confluence
        }

    def update_pattern_parameters(self, new_parameters: Dict[str, Any]):
        """Update pattern analysis parameters."""
        if "pattern_confidence_threshold" in new_parameters:
            self.pattern_confidence_threshold = new_parameters["pattern_confidence_threshold"]
        if "min_pattern_confluence" in new_parameters:
            self.min_pattern_confluence = new_parameters["min_pattern_confluence"]

        logger.info(f"PatternMaster parameters updated: {new_parameters}")
    def _assess_pattern_data_quality(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess data quality for pattern recognition."""
        assessment = {
            "quality_score": 0.0,
            "ohlc_completeness": 0.0,
            "pattern_recognition_reliability": "LOW",
            "candlestick_analysis_confidence": 0.0,
            "chart_pattern_confidence": 0.0,
            "wave_analysis_confidence": 0.0,
            "historical_depth": 0,
            "data_source": "UNKNOWN"
        }

        # Check OHLCV data quality (essential for pattern recognition)
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            ohlcv = market_data['OHLCV']
            assessment["historical_depth"] = len(ohlcv)

            # Check OHLC completeness (critical for candlestick patterns)
            required_columns = ['open', 'high', 'low', 'close']
            missing_data = ohlcv[required_columns].isnull().sum().sum()
            total_data_points = len(ohlcv) * len(required_columns)
            assessment["ohlc_completeness"] = 1.0 - (missing_data / total_data_points)

            # Determine data source and reliability
            min_candles_required = self.config['data_quality']['min_candles_required']
            if len(ohlcv) > min_candles_required:  # MT5 typically provides more historical data
                assessment["data_source"] = "MT5"
                assessment["pattern_recognition_reliability"] = "HIGH"
            else:
                assessment["data_source"] = "YAHOO"
                assessment["pattern_recognition_reliability"] = "MEDIUM"

        # Candlestick analysis confidence (needs clean OHLC data)
        assessment["candlestick_analysis_confidence"] = assessment["ohlc_completeness"]

        # Chart pattern confidence (needs sufficient history)
        high_threshold = self.config['pattern_confidence']['high_confidence_threshold']
        medium_threshold = self.config['pattern_confidence']['medium_confidence_threshold']
        low_threshold = self.config['pattern_confidence']['low_confidence_threshold']

        if assessment["historical_depth"] >= 100:
            assessment["chart_pattern_confidence"] = high_threshold + 0.1  # 0.9
        elif assessment["historical_depth"] >= 50:
            assessment["chart_pattern_confidence"] = medium_threshold + 0.1  # 0.7
        else:
            assessment["chart_pattern_confidence"] = low_threshold

        # Wave analysis confidence (needs substantial history)
        if assessment["historical_depth"] >= 200:
            assessment["wave_analysis_confidence"] = high_threshold + 0.1  # 0.9
        elif assessment["historical_depth"] >= 100:
            assessment["wave_analysis_confidence"] = medium_threshold
        else:
            assessment["wave_analysis_confidence"] = low_threshold - 0.1  # 0.3

        # Overall quality score
        quality_factors = [
            assessment["ohlc_completeness"],
            min(assessment["historical_depth"] / 200, 1.0),  # Normalize historical depth
            0.2 if assessment["pattern_recognition_reliability"] == "HIGH" else 0.1
        ]
        assessment["quality_score"] = sum(quality_factors) / len(quality_factors)

        return assessment

    def _analyze_candlestick_patterns_enhanced(self, indicator_results: Dict[str, Any], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced candlestick pattern analysis with data quality consideration."""
        # Base candlestick analysis
        candlestick_analysis = self._analyze_candlestick_patterns(indicator_results)

        # Data quality adjustments
        candlestick_confidence = data_assessment["candlestick_analysis_confidence"]
        candlestick_analysis["data_quality"] = candlestick_confidence

        # Adjust pattern strengths based on data quality
        if candlestick_confidence < 0.7:
            # Reduce strength of all patterns with poor data quality
            for signal_list in ["bullish_reversal_signals", "bearish_reversal_signals", "continuation_signals"]:
                for signal in candlestick_analysis.get(signal_list, []):
                    signal["strength"] *= candlestick_confidence
                    signal["reliability"] *= candlestick_confidence

            candlestick_analysis["candlestick_strength"] *= candlestick_confidence
            candlestick_analysis["quality_warning"] = "Reduced pattern reliability due to data quality"

        # Enhanced pattern validation with high-quality data
        elif candlestick_confidence > 0.9:
            candlestick_analysis.update(self._enhanced_candlestick_validation(indicator_results))

        return candlestick_analysis

    def _enhanced_candlestick_validation(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced candlestick pattern validation with high-quality data."""
        enhanced = {
            "pattern_validation": "HIGH_QUALITY",
            "multi_candle_confirmation": False,
            "volume_confirmation": False
        }

        # Check for multi-candle pattern confirmation
        multi_candle_patterns = ["morning_star_indicator", "evening_star_indicator",
                                "three_white_soldiers_indicator", "three_black_crows_indicator"]

        confirmed_patterns = 0
        for pattern in multi_candle_patterns:
            if pattern in indicator_results:
                pattern_data = indicator_results[pattern]
                if pattern_data.get("confirmation", False):
                    confirmed_patterns += 1

        if confirmed_patterns > 0:
            enhanced["multi_candle_confirmation"] = True

        return enhanced
    def _analyze_chart_patterns_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced chart pattern analysis with data quality considerations."""
        # Base chart pattern analysis
        chart_pattern_analysis = self._analyze_chart_patterns(indicator_results, market_data)

        # Data quality impact
        chart_confidence = data_assessment["chart_pattern_confidence"]
        chart_pattern_analysis["data_quality"] = chart_confidence

        # Historical depth requirements for chart patterns
        historical_depth = data_assessment["historical_depth"]

        if historical_depth < 50:
            chart_pattern_analysis["reliability"] = "LOW"
            chart_pattern_analysis["depth_warning"] = "Insufficient historical data for reliable chart pattern recognition"
        elif historical_depth < 100:
            chart_pattern_analysis["reliability"] = "MEDIUM"
        else:
            chart_pattern_analysis["reliability"] = "HIGH"

            # Enhanced pattern detection with sufficient data
            if chart_confidence > 0.8:
                chart_pattern_analysis.update(self._enhanced_chart_pattern_detection(indicator_results, market_data))

        return chart_pattern_analysis

    def _enhanced_chart_pattern_detection(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Enhanced chart pattern detection with high-quality data."""
        enhanced = {
            "pattern_maturity": "DEVELOPING",
            "breakout_probability": 0.0,
            "target_projection": 0.0
        }

        # Advanced pattern analysis
        if "head_and_shoulders_indicator" in indicator_results:
            hs_data = indicator_results["head_and_shoulders_indicator"]
            if hs_data.get("pattern_complete", False):
                enhanced["pattern_maturity"] = "COMPLETE"
                enhanced["breakout_probability"] = hs_data.get("breakout_probability", 0.5)

        if "triangle_pattern_indicator" in indicator_results:
            triangle_data = indicator_results["triangle_pattern_indicator"]
            enhanced["target_projection"] = triangle_data.get("target_projection", 0.0)

        return enhanced

    def _analyze_wave_patterns_enhanced(self, indicator_results: Dict[str, Any], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced wave pattern analysis with data quality considerations."""
        # Base wave analysis
        wave_analysis = self._analyze_wave_patterns(indicator_results)

        # Data quality impact on wave analysis
        wave_confidence = data_assessment["wave_analysis_confidence"]
        wave_analysis["data_quality"] = wave_confidence

        # Wave analysis requires substantial historical data
        if wave_confidence < 0.5:
            wave_analysis["reliability"] = "LOW"
            wave_analysis["wave_count_confidence"] = "UNCERTAIN"
        elif wave_confidence < 0.8:
            wave_analysis["reliability"] = "MEDIUM"
            wave_analysis["wave_count_confidence"] = "MODERATE"
        else:
            wave_analysis["reliability"] = "HIGH"
            wave_analysis["wave_count_confidence"] = "CONFIDENT"

            # Enhanced wave analysis with high-quality data
            wave_analysis.update(self._enhanced_wave_analysis(indicator_results))

        return wave_analysis

    def _enhanced_wave_analysis(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced wave analysis with high-quality data."""
        enhanced = {
            "wave_structure_quality": "HIGH",
            "fibonacci_alignment": False,
            "wave_completion_percentage": 0.0
        }

        # Elliott Wave structure analysis
        if "elliott_wave_oscillator_indicator" in indicator_results:
            ew_data = indicator_results["elliott_wave_oscillator_indicator"]
            enhanced["wave_completion_percentage"] = ew_data.get("completion_percentage", 0.0)

        if "wave_structure_indicator" in indicator_results:
            ws_data = indicator_results["wave_structure_indicator"]
            enhanced["fibonacci_alignment"] = ws_data.get("fibonacci_confluence", False)

        return enhanced

    def _analyze_pattern_confluence_enhanced(self, candlestick_analysis: Dict[str, Any], chart_pattern_analysis: Dict[str, Any],
                                           wave_analysis: Dict[str, Any], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced pattern confluence analysis weighted by data quality."""
        # Base confluence analysis
        confluence_analysis = self._analyze_pattern_confluence(candlestick_analysis, chart_pattern_analysis, wave_analysis)

        # Data quality weighting
        quality_score = data_assessment["quality_score"]
        confluence_analysis["data_quality_weight"] = quality_score

        # Adjust confluence strength based on data quality
        original_strength = confluence_analysis.get("overall_strength", 0.0)
        confluence_analysis["raw_strength"] = original_strength
        confluence_analysis["overall_strength"] = original_strength * quality_score

        # Component reliability weighting
        candlestick_weight = data_assessment["candlestick_analysis_confidence"] * self.candlestick_weight
        chart_weight = data_assessment["chart_pattern_confidence"] * self.chart_pattern_weight
        wave_weight = data_assessment["wave_analysis_confidence"] * self.wave_analysis_weight

        total_weight = candlestick_weight + chart_weight + wave_weight

        if total_weight > 0:
            # Reweight based on data quality
            confluence_analysis["adjusted_candlestick_weight"] = candlestick_weight / total_weight
            confluence_analysis["adjusted_chart_weight"] = chart_weight / total_weight
            confluence_analysis["adjusted_wave_weight"] = wave_weight / total_weight

        # Quality-based confluence categories
        if quality_score >= 0.8:
            if confluence_analysis["overall_strength"] >= 0.6:
                confluence_analysis["confluence_level"] = "VERY_STRONG"
            elif confluence_analysis["overall_strength"] >= 0.4:
                confluence_analysis["confluence_level"] = "STRONG"
        else:
            # More conservative with lower quality data
            if confluence_analysis["overall_strength"] >= 0.7:
                confluence_analysis["confluence_level"] = "STRONG"
            elif confluence_analysis["overall_strength"] >= 0.5:
                confluence_analysis["confluence_level"] = "MODERATE"

        return confluence_analysis
    def _generate_pattern_decision_enhanced(self, confluence_analysis: Dict[str, Any], market_conditions: MarketConditions, data_assessment: Dict[str, Any]) -> str:
        """Enhanced pattern decision generation considering data quality."""
        # Data quality threshold check
        quality_score = data_assessment["quality_score"]
        if quality_score < 0.4:
            return "NO_SIGNAL"  # Insufficient data quality for reliable pattern recognition

        # Adjust decision thresholds based on data quality
        confluence_strength = confluence_analysis.get("overall_strength", 0.0)

        # More conservative thresholds for lower quality data
        if quality_score < 0.7:
            min_strength_threshold = 0.7
        else:
            min_strength_threshold = 0.5

        if confluence_strength < min_strength_threshold:
            return "NO_SIGNAL"

        # Enhanced decision logic with high-quality data
        if quality_score > 0.8:
            confluence_level = confluence_analysis.get("confluence_level", "WEAK")
            if confluence_level == "VERY_STRONG":
                strongest_pattern = confluence_analysis.get("strongest_pattern", "NONE")
                if "BULLISH" in strongest_pattern or "BUY" in strongest_pattern:
                    return "STRONG_BUY"
                elif "BEARISH" in strongest_pattern or "SELL" in strongest_pattern:
                    return "STRONG_SELL"

        # Standard pattern-based decisions
        reversal_indication = confluence_analysis.get("reversal_indication", "NEUTRAL")

        if reversal_indication == "BULLISH_REVERSAL":
            return "BUY"
        elif reversal_indication == "BEARISH_REVERSAL":
            return "SELL"
        elif reversal_indication == "CONTINUATION":
            return "HOLD"
        else:
            return "NO_SIGNAL"

    def _calculate_pattern_confidence_enhanced(self, candlestick_analysis: Dict[str, Any], chart_pattern_analysis: Dict[str, Any],
                                             wave_analysis: Dict[str, Any], confluence_analysis: Dict[str, Any],
                                             data_assessment: Dict[str, Any]) -> float:
        """Enhanced pattern confidence calculation incorporating data quality."""
        # Base confidence calculation
        base_confidence = self._calculate_pattern_confidence(candlestick_analysis, chart_pattern_analysis, wave_analysis, confluence_analysis)

        # Data quality factor (30% weight)
        quality_score = data_assessment["quality_score"]
        quality_factor = quality_score * 0.3

        # Historical depth factor (15% weight)
        historical_depth = data_assessment["historical_depth"]
        depth_factor = min(historical_depth / 200, 1.0) * 0.15

        # OHLC completeness factor (15% weight)
        ohlc_completeness = data_assessment["ohlc_completeness"]
        completeness_factor = ohlc_completeness * 0.15

        # Component confidence weighting (40% weight)
        candlestick_conf = data_assessment["candlestick_analysis_confidence"]
        chart_conf = data_assessment["chart_pattern_confidence"]
        wave_conf = data_assessment["wave_analysis_confidence"]

        weighted_component_conf = (
            candlestick_conf * self.candlestick_weight +
            chart_conf * self.chart_pattern_weight +
            wave_conf * self.wave_analysis_weight
        ) * 0.4

        # Calculate enhanced confidence
        enhanced_confidence = base_confidence + quality_factor + depth_factor + completeness_factor + weighted_component_conf

        # Penalty for insufficient data
        if quality_score < 0.6:
            enhanced_confidence *= 0.8

        return max(0.0, min(enhanced_confidence, 1.0))

    def _assess_pattern_risk_enhanced(self, confluence_analysis: Dict[str, Any], market_conditions: MarketConditions, data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced pattern risk assessment with data quality considerations."""
        # Base pattern risk assessment
        risk_assessment = self._assess_pattern_risk(confluence_analysis, market_conditions)

        # Data quality risk factor
        quality_score = data_assessment["quality_score"]
        risk_assessment["data_quality_risk"] = "LOW" if quality_score > 0.8 else "MEDIUM" if quality_score > 0.5 else "HIGH"

        # Pattern reliability risk
        if data_assessment["pattern_recognition_reliability"] == "LOW":
            risk_assessment["pattern_reliability_risk"] = "HIGH"
            risk_assessment["false_signal_probability"] = 0.4
        else:
            risk_assessment["pattern_reliability_risk"] = "LOW"
            risk_assessment["false_signal_probability"] = 0.2

        # Historical depth risk
        if data_assessment["historical_depth"] < 50:
            risk_assessment["historical_depth_risk"] = "HIGH"
        elif data_assessment["historical_depth"] < 100:
            risk_assessment["historical_depth_risk"] = "MEDIUM"
        else:
            risk_assessment["historical_depth_risk"] = "LOW"

        return risk_assessment

    def _generate_pattern_reasoning_enhanced(self, decision: str, candlestick_analysis: Dict[str, Any],
                                           chart_pattern_analysis: Dict[str, Any], wave_analysis: Dict[str, Any],
                                           confluence_analysis: Dict[str, Any], data_assessment: Dict[str, Any]) -> str:
        """Enhanced pattern reasoning including data quality context."""
        reasoning_parts = []

        # Base reasoning
        base_reasoning = self._generate_pattern_reasoning(decision, candlestick_analysis, chart_pattern_analysis, wave_analysis, confluence_analysis)
        reasoning_parts.append(base_reasoning)

        # Data quality context
        quality_score = data_assessment["quality_score"]
        data_source = data_assessment["data_source"]
        historical_depth = data_assessment["historical_depth"]

        reasoning_parts.append(f"Pattern analysis based on {data_source} data with {quality_score:.2f} quality score using {historical_depth} historical periods.")

        # Component reliability context
        candlestick_conf = data_assessment["candlestick_analysis_confidence"]
        chart_conf = data_assessment["chart_pattern_confidence"]
        wave_conf = data_assessment["wave_analysis_confidence"]

        reasoning_parts.append(f"Analysis confidence: Candlestick {candlestick_conf:.2f}, Chart patterns {chart_conf:.2f}, Wave analysis {wave_conf:.2f}.")

        # Enhanced insights with high-quality data
        if quality_score > 0.8:
            reasoning_parts.append("High data quality enables enhanced pattern validation and reliability.")

            if confluence_analysis.get("confluence_level") == "VERY_STRONG":
                reasoning_parts.append("Very strong pattern confluence detected with high-quality data confirmation.")

        # Data quality warnings
        if quality_score < 0.6:
            reasoning_parts.append("Conservative pattern approach due to limited data quality.")

        if data_assessment["historical_depth"] < 50:
            reasoning_parts.append("Limited historical data may affect pattern reliability.")

        # Pattern-specific data requirements
        strongest_pattern = confluence_analysis.get("strongest_pattern", "NONE")
        if "wave" in strongest_pattern.lower() and wave_conf < 0.6:
            reasoning_parts.append("Wave pattern analysis has reduced confidence due to limited historical data.")

        return " ".join(reasoning_parts)
