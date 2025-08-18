"""
Risk Genius Agent for the AUJ Platform.

This agent specializes in volatility analysis, statistical risk assessment, and market stability.
It focuses on 23 risk-related indicators to provide comprehensive risk evaluation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal
import statistics
import yaml
import os

from .base_agent import BaseAgent, AnalysisResult, AgentState
from ..core.data_contracts import MarketConditions, TradeDirection, ConfidenceLevel
from ..core.exceptions import AgentError, ValidationError
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class RiskGenius(BaseAgent):
    """
    Risk Genius Agent - Volatility, statistical risk, and market stability analysis.

    Specializes in:
    - Volatility measurement and prediction
    - Statistical risk assessment
    - Market stability evaluation
    - Risk-adjusted position sizing recommendations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager: Optional[UnifiedConfigManager] = None):
        """Initialize the Risk Genius Agent."""
        # Define assigned indicators for this agent (from registry)
        assigned_indicators = [
            # Volatility Indicators (8)
            "bollinger_bands_indicator",
            "central_pivot_range_indicator",
            "chaikin_volatility_indicator",
            "historical_volatility_indicator",
            "keltner_channel_indicator",
            "mass_index_indicator",
            "relative_volatility_index_indicator",
            "ulcer_index_indicator",

            # ALL Statistical Risk Indicators (9)
            "autocorrelation_indicator",
            "garch_volatility_model_indicator",
            "hurst_exponent_indicator",
            "kalman_filter_indicator",
            "market_regime_detection_indicator",
            "skewness_indicator",
            "standard_deviation_channels_indicator",
            "variance_ratio_indicator",
            "zscore_indicator"
        ]

        super().__init__(
            name="RiskGenius",
            specialization="Volatility analysis, statistical risk assessment, and market stability",
            assigned_indicators=assigned_indicators,
            config=config,
            config_manager=config_manager
        )

        # Load configuration from YAML file
        self._load_agent_config()

        logger.info(f"RiskGenius initialized with {len(assigned_indicators)} risk indicators")

    def _load_agent_config(self):
        """Load agent-specific configuration from YAML file."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'risk_genius.yaml'
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
        self.volatility_threshold_high = self.config_manager.get_float('agents.risk_genius.volatility_threshold_high', 0.025)
        self.volatility_threshold_low = self.config_manager.get_float('agents.risk_genius.volatility_threshold_low', 0.008)
        self.risk_tolerance = self.config_manager.get_float('agents.risk_genius.risk_tolerance', 0.02)
        self.var_confidence_level = self.config_manager.get_float('agents.risk_genius.var_confidence_level', 0.95)
        self.drawdown_threshold = self.config_manager.get_float('agents.risk_genius.drawdown_threshold', 0.15)

        # Statistical parameters from unified config
        self.lookback_period = self.config_manager.get_int('agents.risk_genius.lookback_period', 20)
        self.volatility_window = self.config_manager.get_int('agents.risk_genius.volatility_window', 14)
        self.correlation_threshold = self.config_manager.get_float('agents.risk_genius.correlation_threshold', 0.7)

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """
        Perform comprehensive data-aware risk analysis.

        Args:
            symbol: Trading symbol
            market_data: Available market data
            market_conditions: Current market conditions
            indicator_results: Calculated indicator values

        Returns:
            AnalysisResult with risk assessment and recommendations
        """
        try:
            # Data-aware risk assessment
            data_assessment = self._assess_risk_data_quality(market_data)

            # 1. Enhanced Volatility Analysis
            volatility_analysis = self._analyze_volatility_enhanced(indicator_results, market_data, data_assessment)

            # 2. Data-Aware Statistical Risk Assessment
            statistical_risk = self._assess_statistical_risk_enhanced(indicator_results, market_data, data_assessment)

            # 3. Enhanced Market Stability Analysis
            stability_analysis = self._analyze_market_stability_enhanced(indicator_results, market_conditions, data_assessment)

            # 4. Data-Quality Adjusted Risk Recommendations
            risk_recommendations = self._generate_risk_recommendations_enhanced(
                volatility_analysis, statistical_risk, stability_analysis, data_assessment
            )

            # 5. Overall Risk Decision with Data Considerations
            decision = self._generate_risk_decision_enhanced(
                volatility_analysis, statistical_risk, stability_analysis, market_conditions, data_assessment
            )

            # 6. Data-Adjusted Risk Confidence
            confidence = self._calculate_risk_confidence_enhanced(
                volatility_analysis, statistical_risk, stability_analysis, data_assessment
            )

            # 7. Enhanced Risk Reasoning
            reasoning = self._generate_risk_reasoning_enhanced(
                decision, volatility_analysis, statistical_risk, stability_analysis, data_assessment
            )

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "volatility_analysis": volatility_analysis,
                    "statistical_risk": statistical_risk,
                    "stability_analysis": stability_analysis,
                    "risk_score": self._calculate_overall_risk_score_enhanced(volatility_analysis, statistical_risk, stability_analysis, data_assessment),
                    "data_assessment": data_assessment
                },
                risk_assessment=risk_recommendations,
                supporting_data={
                    "current_volatility": volatility_analysis.get("current_volatility", 0.0),
                    "volatility_regime": volatility_analysis.get("volatility_regime", "NORMAL"),
                    "risk_level": risk_recommendations.get("overall_risk_level", "MEDIUM"),
                    "data_quality_score": data_assessment.get("quality_score", 0.0)
                }
            )

        except Exception as e:
            logger.error(f"RiskGenius analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Risk analysis failed: {str(e)}")

    def _analyze_volatility(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze volatility indicators and patterns."""
        volatility_analysis = {
            "current_volatility": 0.0,
            "volatility_regime": "NORMAL",
            "volatility_trend": "STABLE",
            "atr_signal": "NEUTRAL",
            "bollinger_squeeze": False,
            "volatility_forecast": "STABLE",
            "volatility_percentile": 0.5
        }

        # ATR Analysis
        if "average_true_range_indicator" in indicator_results:
            atr_data = indicator_results["average_true_range_indicator"]
            current_atr = atr_data.get("atr", 0.0)
            atr_percentage = atr_data.get("atr_percentage", 0.0)

            volatility_analysis["current_volatility"] = atr_percentage / 100.0

            if atr_percentage > 2.5:  # 2.5% ATR
                volatility_analysis["volatility_regime"] = "HIGH"
                volatility_analysis["atr_signal"] = "HIGH_VOLATILITY"
            elif atr_percentage < 0.8:  # 0.8% ATR
                volatility_analysis["volatility_regime"] = "LOW"
                volatility_analysis["atr_signal"] = "LOW_VOLATILITY"

        # Bollinger Bands Squeeze Analysis
        if "bollinger_bands_indicator" in indicator_results:
            bb_data = indicator_results["bollinger_bands_indicator"]
            upper_band = bb_data.get("upper_band", 0)
            lower_band = bb_data.get("lower_band", 0)
            middle_band = bb_data.get("middle_band", 0)

            if middle_band > 0:
                band_width = (upper_band - lower_band) / middle_band
                if band_width < 0.1:  # Narrow bands indicate squeeze
                    volatility_analysis["bollinger_squeeze"] = True
                    volatility_analysis["volatility_forecast"] = "BREAKOUT_EXPECTED"

        # Historical Volatility Analysis
        if "historical_volatility_indicator" in indicator_results:
            hv_data = indicator_results["historical_volatility_indicator"]
            current_hv = hv_data.get("volatility", 0.0)
            hv_percentile = hv_data.get("percentile", 50)

            volatility_analysis["volatility_percentile"] = hv_percentile / 100.0

            if hv_percentile > 80:
                volatility_analysis["volatility_trend"] = "INCREASING"
            elif hv_percentile < 20:
                volatility_analysis["volatility_trend"] = "DECREASING"

        return volatility_analysis

    def _assess_statistical_risk(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess statistical risk measures."""
        statistical_risk = {
            "z_score": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "var_95": 0.0,
            "expected_shortfall": 0.0,
            "beta_to_market": 1.0,
            "correlation_risk": "LOW",
            "statistical_regime": "NORMAL"
        }

        # Z-Score Analysis
        if "zscore_indicator" in indicator_results:
            zscore_data = indicator_results["zscore_indicator"]
            z_score = zscore_data.get("zscore", 0.0)
            statistical_risk["z_score"] = z_score

            if abs(z_score) > 2.0:
                statistical_risk["statistical_regime"] = "EXTREME"
            elif abs(z_score) > 1.5:
                statistical_risk["statistical_regime"] = "UNUSUAL"

        # Skewness Analysis
        if "skewness_indicator" in indicator_results:
            skew_data = indicator_results["skewness_indicator"]
            skewness = skew_data.get("skewness", 0.0)
            statistical_risk["skewness"] = skewness

            # Extreme skewness indicates asymmetric risk
            if abs(skewness) > 1.0:
                statistical_risk["statistical_regime"] = "ASYMMETRIC"

        # Beta Coefficient Analysis
        if "beta_coefficient_indicator" in indicator_results:
            beta_data = indicator_results["beta_coefficient_indicator"]
            beta = beta_data.get("beta", 1.0)
            statistical_risk["beta_to_market"] = beta

        # Calculate VaR estimate
        if "OHLCV" in market_data:
            df = market_data["OHLCV"]
            if len(df) >= self.lookback_period:
                returns = df['close'].pct_change().dropna()
                if len(returns) >= 10:
                    var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
                    statistical_risk["var_95"] = abs(var_95)

                    # Expected Shortfall (average of losses beyond VaR)
                    tail_losses = returns[returns <= var_95]
                    if len(tail_losses) > 0:
                        statistical_risk["expected_shortfall"] = abs(tail_losses.mean())

        return statistical_risk

    def _analyze_market_stability(self, indicator_results: Dict[str, Any], market_conditions: MarketConditions) -> Dict[str, Any]:
        """Analyze market stability indicators."""
        stability_analysis = {
            "mass_index_signal": "STABLE",
            "ulcer_index": 0.0,
            "market_stress": "LOW",
            "regime_stability": "STABLE",
            "breakout_probability": 0.0,
            "mean_reversion_strength": 0.0
        }

        # Mass Index Analysis
        if "mass_index_indicator" in indicator_results:
            mi_data = indicator_results["mass_index_indicator"]
            mass_index = mi_data.get("mass_index", 25.0)

            if mass_index > 27.0:
                stability_analysis["mass_index_signal"] = "REVERSAL_WARNING"
                stability_analysis["market_stress"] = "HIGH"
            elif mass_index > 26.5:
                stability_analysis["mass_index_signal"] = "UNSTABLE"
                stability_analysis["market_stress"] = "MEDIUM"

        # Ulcer Index Analysis
        if "ulcer_index_indicator" in indicator_results:
            ui_data = indicator_results["ulcer_index_indicator"]
            ulcer_index = ui_data.get("ulcer_index", 0.0)
            stability_analysis["ulcer_index"] = ulcer_index

            if ulcer_index > 10.0:
                stability_analysis["market_stress"] = "HIGH"
            elif ulcer_index > 5.0:
                stability_analysis["market_stress"] = "MEDIUM"

        # Market Regime Stability
        regime = market_conditions.regime.value
        if regime in ["HIGH_VOLATILITY", "BREAKOUT", "REVERSAL"]:
            stability_analysis["regime_stability"] = "UNSTABLE"
        elif regime in ["SIDEWAYS", "ACCUMULATION", "DISTRIBUTION"]:
            stability_analysis["regime_stability"] = "TRANSITIONAL"

        # Hurst Exponent Analysis (if available)
        if "hurst_exponent_indicator" in indicator_results:
            hurst_data = indicator_results["hurst_exponent_indicator"]
            hurst_exp = hurst_data.get("hurst_exponent", 0.5)

            if hurst_exp > 0.6:
                stability_analysis["mean_reversion_strength"] = 1.0 - hurst_exp
            else:
                stability_analysis["mean_reversion_strength"] = hurst_exp

        return stability_analysis

    def _generate_risk_recommendations(self,
                                     volatility_analysis: Dict[str, Any],
                                     statistical_risk: Dict[str, Any],
                                     stability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk management recommendations."""
        recommendations = {
            "overall_risk_level": "MEDIUM",
            "position_size_multiplier": 1.0,
            "stop_loss_multiplier": 1.0,
            "max_exposure_percentage": 2.0,
            "diversification_needed": False,
            "hedge_recommendation": "NONE",
            "risk_warnings": []
        }

        # Assess overall risk level
        risk_factors = []

        # Volatility risk
        vol_regime = volatility_analysis.get("volatility_regime", "NORMAL")
        if vol_regime == "HIGH":
            risk_factors.append(0.8)
            recommendations["risk_warnings"].append("High volatility detected")
        elif vol_regime == "LOW":
            risk_factors.append(0.2)
        else:
            risk_factors.append(0.5)

        # Statistical risk
        stat_regime = statistical_risk.get("statistical_regime", "NORMAL")
        if stat_regime in ["EXTREME", "ASYMMETRIC"]:
            risk_factors.append(0.9)
            recommendations["risk_warnings"].append("Extreme statistical conditions")
        elif stat_regime == "UNUSUAL":
            risk_factors.append(0.7)
        else:
            risk_factors.append(0.4)

        # Stability risk
        stability = stability_analysis.get("regime_stability", "STABLE")
        if stability == "UNSTABLE":
            risk_factors.append(0.8)
            recommendations["risk_warnings"].append("Market instability detected")
        elif stability == "TRANSITIONAL":
            risk_factors.append(0.6)
        else:
            risk_factors.append(0.3)

        # Calculate overall risk score
        overall_risk = sum(risk_factors) / len(risk_factors)

        if overall_risk > 0.7:
            recommendations["overall_risk_level"] = "HIGH"
            recommendations["position_size_multiplier"] = 0.5
            recommendations["stop_loss_multiplier"] = 0.5  # Tighter stops
            recommendations["max_exposure_percentage"] = 1.0
            recommendations["diversification_needed"] = True
        elif overall_risk < 0.3:
            recommendations["overall_risk_level"] = "LOW"
            recommendations["position_size_multiplier"] = 1.5
            recommendations["stop_loss_multiplier"] = 1.5  # Wider stops allowed
            recommendations["max_exposure_percentage"] = 3.0
        else:
            recommendations["position_size_multiplier"] = 1.0
            recommendations["stop_loss_multiplier"] = 1.0
            recommendations["max_exposure_percentage"] = 2.0

        # VaR-based recommendations
        var_95 = statistical_risk.get("var_95", 0.02)
        if var_95 > 0.05:  # 5% daily VaR
            recommendations["hedge_recommendation"] = "CONSIDER_HEDGING"
            recommendations["risk_warnings"].append("High Value-at-Risk detected")

        return recommendations

    def _generate_risk_decision(self,
                              volatility_analysis: Dict[str, Any],
                              statistical_risk: Dict[str, Any],
                              stability_analysis: Dict[str, Any],
                              market_conditions: MarketConditions) -> str:
        """Generate risk-based trading decision."""
        # High risk conditions - recommend caution
        if (volatility_analysis.get("volatility_regime") == "HIGH" and
            statistical_risk.get("statistical_regime") in ["EXTREME", "ASYMMETRIC"]):
            return "NO_SIGNAL"  # Too risky to trade

        # Extreme instability
        if stability_analysis.get("market_stress") == "HIGH":
            return "NO_SIGNAL"

        # Bollinger Band squeeze - potential breakout
        if volatility_analysis.get("bollinger_squeeze", False):
            return "HOLD"  # Wait for breakout direction

        # Low volatility, stable conditions - favorable for trading
        if (volatility_analysis.get("volatility_regime") == "LOW" and
            stability_analysis.get("regime_stability") == "STABLE"):
            return "HOLD"  # Neutral, let other agents decide direction

        # Normal conditions
        return "HOLD"  # Risk agent provides risk assessment, not direction

    def _calculate_risk_confidence(self,
                                 volatility_analysis: Dict[str, Any],
                                 statistical_risk: Dict[str, Any],
                                 stability_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in risk assessment."""
        confidence_factors = []

        # Volatility confidence (40% weight)
        vol_data_quality = 1.0  # Assume good data
        if "current_volatility" in volatility_analysis:
            vol_data_quality = min(volatility_analysis["current_volatility"] * 10, 1.0)
        confidence_factors.append(vol_data_quality * 0.4)

        # Statistical confidence (35% weight)
        stat_confidence = 0.7  # Base statistical confidence
        if abs(statistical_risk.get("z_score", 0.0)) > 0.5:
            stat_confidence = 0.9  # High confidence in extreme values
        confidence_factors.append(stat_confidence * 0.35)

        # Stability confidence (25% weight)
        stability_confidence = 0.8
        if stability_analysis.get("market_stress") != "LOW":
            stability_confidence = 0.9  # High confidence in stress detection
        confidence_factors.append(stability_confidence * 0.25)

        return min(sum(confidence_factors), 1.0)

    def _calculate_overall_risk_score(self,
                                    volatility_analysis: Dict[str, Any],
                                    statistical_risk: Dict[str, Any],
                                    stability_analysis: Dict[str, Any]) -> float:
        """Calculate overall risk score (0 = low risk, 1 = high risk)."""
        risk_components = []

        # Volatility component
        vol_regime = volatility_analysis.get("volatility_regime", "NORMAL")
        if vol_regime == "HIGH":
            risk_components.append(0.8)
        elif vol_regime == "LOW":
            risk_components.append(0.2)
        else:
            risk_components.append(0.5)

        # Statistical component
        var_95 = statistical_risk.get("var_95", 0.02)
        stat_risk = min(var_95 * 20, 1.0)  # Scale 5% VaR to max risk
        risk_components.append(stat_risk)

        # Stability component
        market_stress = stability_analysis.get("market_stress", "LOW")
        if market_stress == "HIGH":
            risk_components.append(0.9)
        elif market_stress == "MEDIUM":
            risk_components.append(0.6)
        else:
            risk_components.append(0.3)

        return sum(risk_components) / len(risk_components)

    def _generate_risk_reasoning(self,
                               decision: str,
                               volatility_analysis: Dict[str, Any],
                               statistical_risk: Dict[str, Any],
                               stability_analysis: Dict[str, Any]) -> str:
        """Generate human-readable risk reasoning."""
        reasoning_parts = []

        # Decision explanation
        if decision == "NO_SIGNAL":
            reasoning_parts.append("Risk assessment indicates unfavorable trading conditions.")
        elif decision == "HOLD":
            reasoning_parts.append("Risk conditions are manageable, allowing other factors to guide decisions.")

        # Volatility assessment
        vol_regime = volatility_analysis.get("volatility_regime", "NORMAL")
        current_vol = volatility_analysis.get("current_volatility", 0.0)
        reasoning_parts.append(f"Current volatility regime: {vol_regime} ({current_vol:.3f}).")

        # Statistical risk
        var_95 = statistical_risk.get("var_95", 0.0)
        stat_regime = statistical_risk.get("statistical_regime", "NORMAL")
        reasoning_parts.append(f"Statistical assessment: {stat_regime} conditions with {var_95:.3f} daily VaR.")

        # Market stability
        stability = stability_analysis.get("regime_stability", "STABLE")
        market_stress = stability_analysis.get("market_stress", "LOW")
        reasoning_parts.append(f"Market stability: {stability} with {market_stress} stress level.")

        # Risk warnings
        risk_warnings = []
        if vol_regime == "HIGH":
            risk_warnings.append("elevated volatility")
        if stat_regime in ["EXTREME", "ASYMMETRIC"]:
            risk_warnings.append("statistical extremes")
        if market_stress == "HIGH":
            risk_warnings.append("market stress")

        if risk_warnings:
            reasoning_parts.append(f"Risk factors: {', '.join(risk_warnings)}.")

        return " ".join(reasoning_parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        used_indicators = []
        for indicator in self.assigned_indicators:
            if indicator in indicator_results:
                used_indicators.append(indicator)
        return used_indicators

    def get_required_data_types(self) -> List[str]:
        """Define required data types for risk analysis."""
        return ["OHLCV"]  # Primary focus on price data for volatility calculations

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed for risk analysis."""
        return 30  # Need sufficient history for volatility and statistical measures

    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get current risk management parameters."""
        return {
            "volatility_threshold_high": self.volatility_threshold_high,
            "volatility_threshold_low": self.volatility_threshold_low,
            "risk_tolerance": self.risk_tolerance,
            "var_confidence_level": self.var_confidence_level,
            "drawdown_threshold": self.drawdown_threshold,
            "lookback_period": self.lookback_period,
            "volatility_window": self.volatility_window
        }

    def update_risk_parameters(self, new_parameters: Dict[str, Any]):
        """Update risk parameters based on market conditions or performance."""
        if "volatility_threshold_high" in new_parameters:
            self.volatility_threshold_high = new_parameters["volatility_threshold_high"]
        if "volatility_threshold_low" in new_parameters:
            self.volatility_threshold_low = new_parameters["volatility_threshold_low"]
        if "risk_tolerance" in new_parameters:
            self.risk_tolerance = new_parameters["risk_tolerance"]

        logger.info(f"RiskGenius parameters updated: {new_parameters}")
    def _assess_risk_data_quality(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess data quality for risk analysis purposes."""
        assessment = {
            "quality_score": 0.0,
            "data_completeness": 0.0,
            "historical_depth": 0,
            "intraday_data_available": False,
            "tick_data_available": False,
            "data_source": "UNKNOWN",
            "risk_calculation_reliability": "LOW",
            "volatility_calc_confidence": 0.0
        }

        # Check OHLCV data quality
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            ohlcv = market_data['OHLCV']
            assessment["historical_depth"] = len(ohlcv)

            # Data completeness check
            null_percentage = ohlcv.isnull().sum().sum() / (len(ohlcv) * len(ohlcv.columns))
            assessment["data_completeness"] = 1.0 - null_percentage

            # Determine data source and reliability
            if len(ohlcv) > 1000:  # MT5 typically has more data
                assessment["data_source"] = "MT5"
                assessment["risk_calculation_reliability"] = "HIGH"
            else:
                assessment["data_source"] = "YAHOO"
                assessment["risk_calculation_reliability"] = "MEDIUM"

        # Check for higher frequency data
        if 'TICK' in market_data and not market_data['TICK'].empty:
            assessment["tick_data_available"] = True
            assessment["volatility_calc_confidence"] = 1.0
        elif any(tf in market_data for tf in ['1H', '5M', '15M']):
            assessment["intraday_data_available"] = True
            assessment["volatility_calc_confidence"] = 0.8
        else:
            assessment["volatility_calc_confidence"] = 0.6

        # Overall quality score
        quality_factors = []
        quality_factors.append(assessment["data_completeness"])
        quality_factors.append(min(assessment["historical_depth"] / 100, 1.0))  # Normalize historical depth
        quality_factors.append(0.3 if assessment["tick_data_available"] else 0.1 if assessment["intraday_data_available"] else 0.0)

        assessment["quality_score"] = sum(quality_factors) / len(quality_factors)

        return assessment

    def _analyze_volatility_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced volatility analysis accounting for data quality."""
        # Base volatility analysis
        volatility_analysis = self._analyze_volatility(indicator_results, market_data)

        # Data quality adjustments
        volatility_analysis["data_quality"] = data_assessment["quality_score"]
        volatility_analysis["calculation_confidence"] = data_assessment["volatility_calc_confidence"]

        # Enhanced volatility calculations with better data
        if data_assessment["tick_data_available"]:
            volatility_analysis.update(self._calculate_tick_based_volatility(market_data.get('TICK')))
        elif data_assessment["intraday_data_available"]:
            volatility_analysis.update(self._calculate_intraday_volatility(market_data))

        # Adjust volatility regime assessment based on data quality
        if data_assessment["quality_score"] < 0.5:
            volatility_analysis["regime_confidence"] = "LOW"
            # Be more conservative with regime classification
            if volatility_analysis["volatility_regime"] == "HIGH":
                volatility_analysis["volatility_regime"] = "ELEVATED"
            elif volatility_analysis["volatility_regime"] == "LOW":
                volatility_analysis["volatility_regime"] = "REDUCED"
        else:
            volatility_analysis["regime_confidence"] = "HIGH"

        # Historical depth adjustment
        if data_assessment["historical_depth"] < 30:
            volatility_analysis["historical_reliability"] = "INSUFFICIENT"
            volatility_analysis["volatility_forecast"] = "UNRELIABLE"
        elif data_assessment["historical_depth"] < 60:
            volatility_analysis["historical_reliability"] = "LIMITED"
        else:
            volatility_analysis["historical_reliability"] = "GOOD"

        return volatility_analysis

    def _calculate_tick_based_volatility(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced volatility metrics from tick data."""
        if tick_data is None or tick_data.empty:
            return {}

        tick_vol = {}

        # Calculate realized volatility from tick data
        tick_returns = tick_data['price'].pct_change().dropna()
        if len(tick_returns) > 10:
            # Realized volatility (annualized)
            tick_vol["realized_volatility"] = tick_returns.std() * np.sqrt(252 * 24 * 60)  # Annualized

            # Intraday volatility patterns
            tick_data_copy = tick_data.copy()
            tick_data_copy['hour'] = pd.to_datetime(tick_data_copy['timestamp']).dt.hour
            hourly_vol = tick_data_copy.groupby('hour')['price'].std()
            tick_vol["peak_volatility_hour"] = hourly_vol.idxmax()
            tick_vol["min_volatility_hour"] = hourly_vol.idxmin()

            # Microstructure volatility
            tick_vol["microstructure_noise"] = tick_returns.std() / tick_data['price'].mean()

        return tick_vol
    def _calculate_intraday_volatility(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate volatility metrics from intraday data."""
        intraday_vol = {}

        # Look for intraday timeframes
        intraday_timeframes = ['1H', '5M', '15M', '30M']

        for tf in intraday_timeframes:
            if tf in market_data and not market_data[tf].empty:
                data = market_data[tf]
                returns = data['close'].pct_change().dropna()

                if len(returns) > 10:
                    # Intraday volatility
                    intraday_vol[f"{tf}_volatility"] = returns.std()

                    # Time-of-day volatility pattern
                    if 'timestamp' in data.columns:
                        data_copy = data.copy()
                        data_copy['hour'] = pd.to_datetime(data_copy['timestamp']).dt.hour
                        hourly_patterns = data_copy.groupby('hour')['close'].std()
                        intraday_vol[f"{tf}_peak_hour"] = hourly_patterns.idxmax()
                break

        return intraday_vol

    def _assess_statistical_risk_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced statistical risk assessment with data quality consideration."""
        # Base statistical risk assessment
        statistical_risk = self._assess_statistical_risk(indicator_results, market_data)

        # Data quality impact on statistical measures
        quality_score = data_assessment["quality_score"]
        statistical_risk["statistical_reliability"] = quality_score

        # Enhanced VaR calculation with better data
        if data_assessment["historical_depth"] >= 60 and quality_score > 0.7:
            statistical_risk.update(self._calculate_enhanced_var(market_data, data_assessment))
        else:
            # Conservative VaR estimates with limited data
            statistical_risk["var_confidence"] = "LOW"
            statistical_risk["var_95"] *= 1.5  # Increase VaR estimate for safety

        # Tail risk assessment
        if data_assessment["tick_data_available"]:
            statistical_risk.update(self._assess_tail_risk_from_ticks(market_data.get('TICK')))

        # Adjust statistical regime based on data reliability
        if quality_score < 0.6:
            if statistical_risk["statistical_regime"] == "NORMAL":
                statistical_risk["statistical_regime"] = "UNCERTAIN"
            statistical_risk["regime_confidence"] = "LOW"
        else:
            statistical_risk["regime_confidence"] = "HIGH"

        return statistical_risk

    def _calculate_enhanced_var(self, market_data: Dict[str, pd.DataFrame], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced VaR with high-quality data."""
        enhanced_var = {}

        if 'OHLCV' in market_data:
            df = market_data['OHLCV']
            returns = df['close'].pct_change().dropna()

            if len(returns) >= 60:
                # Multiple VaR confidence levels
                enhanced_var["var_90"] = abs(np.percentile(returns, 10))
                enhanced_var["var_95"] = abs(np.percentile(returns, 5))
                enhanced_var["var_99"] = abs(np.percentile(returns, 1))

                # Expected Shortfall at multiple levels
                var_95 = enhanced_var["var_95"]
                tail_losses = returns[returns <= -var_95]
                if len(tail_losses) > 0:
                    enhanced_var["expected_shortfall_95"] = abs(tail_losses.mean())

                # Historical simulation confidence
                enhanced_var["var_confidence"] = "HIGH"
                enhanced_var["sample_size"] = len(returns)

        return enhanced_var

    def _assess_tail_risk_from_ticks(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess tail risk using tick-level data."""
        tail_risk = {}

        if tick_data is not None and not tick_data.empty and len(tick_data) > 100:
            tick_returns = tick_data['price'].pct_change().dropna()

            # Extreme return detection
            extreme_threshold = tick_returns.std() * 3
            extreme_returns = tick_returns[abs(tick_returns) > extreme_threshold]

            tail_risk["extreme_return_frequency"] = len(extreme_returns) / len(tick_returns)
            tail_risk["max_tick_loss"] = abs(tick_returns.min())
            tail_risk["max_tick_gain"] = tick_returns.max()

            # Jump detection
            jump_threshold = tick_returns.std() * 2
            jumps = tick_returns[abs(tick_returns) > jump_threshold]
            tail_risk["jump_frequency"] = len(jumps) / len(tick_returns)

            if tail_risk["extreme_return_frequency"] > 0.01:  # More than 1% extreme moves
                tail_risk["tail_risk_level"] = "HIGH"
            elif tail_risk["extreme_return_frequency"] > 0.005:
                tail_risk["tail_risk_level"] = "MEDIUM"
            else:
                tail_risk["tail_risk_level"] = "LOW"

        return tail_risk
    def _analyze_market_stability_enhanced(self, indicator_results: Dict[str, Any], market_conditions: MarketConditions, data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced market stability analysis with data quality considerations."""
        # Base stability analysis
        stability_analysis = self._analyze_market_stability(indicator_results, market_conditions)

        # Data quality impact on stability assessment
        quality_score = data_assessment["quality_score"]
        stability_analysis["assessment_reliability"] = quality_score

        # Enhanced stability metrics with better data
        if data_assessment["historical_depth"] >= 50:
            stability_analysis.update(self._calculate_stability_metrics(indicator_results))
        else:
            stability_analysis["stability_confidence"] = "LOW"
            stability_analysis["limited_history_warning"] = True

        # Adjust regime stability confidence based on data
        if quality_score < 0.6:
            if stability_analysis["regime_stability"] == "STABLE":
                stability_analysis["regime_stability"] = "UNCERTAIN"
            stability_analysis["regime_confidence"] = "LOW"
        else:
            stability_analysis["regime_confidence"] = "HIGH"

        return stability_analysis

    def _calculate_stability_metrics(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced stability metrics."""
        stability_metrics = {}

        # Regime persistence from indicators
        if "market_regime_detection_indicator" in indicator_results:
            regime_data = indicator_results["market_regime_detection_indicator"]
            stability_metrics["regime_persistence"] = regime_data.get("persistence", 0.5)
            stability_metrics["regime_transition_probability"] = regime_data.get("transition_prob", 0.1)

        # Volatility clustering
        if "garch_volatility_model_indicator" in indicator_results:
            garch_data = indicator_results["garch_volatility_model_indicator"]
            stability_metrics["volatility_clustering"] = garch_data.get("clustering_strength", 0.0)

        return stability_metrics

    def _generate_risk_recommendations_enhanced(self, volatility_analysis: Dict[str, Any], statistical_risk: Dict[str, Any],
                                              stability_analysis: Dict[str, Any], data_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk recommendations considering data quality."""
        # Base recommendations
        recommendations = self._generate_risk_recommendations(volatility_analysis, statistical_risk, stability_analysis)

        # Data quality adjustments
        quality_score = data_assessment["quality_score"]
        recommendations["data_quality_adjustment"] = quality_score

        # Conservative adjustments for lower quality data
        if quality_score < 0.6:
            recommendations["position_size_multiplier"] *= 0.8  # More conservative sizing
            recommendations["stop_loss_multiplier"] *= 0.9     # Tighter stops
            recommendations["max_exposure_percentage"] *= 0.8  # Lower exposure
            recommendations["data_quality_warning"] = "Limited data quality - conservative approach recommended"

        # Enhanced recommendations with better data
        elif quality_score > 0.8:
            # Fine-tune recommendations with tick data insights
            if data_assessment["tick_data_available"]:
                tick_vol = volatility_analysis.get("realized_volatility", 0)
                if tick_vol and tick_vol < volatility_analysis.get("current_volatility", 0.02):
                    recommendations["position_size_multiplier"] *= 1.1  # Slight increase

            recommendations["data_quality_bonus"] = "High data quality allows refined risk management"

        # Multi-timeframe risk adjustment
        if data_assessment["intraday_data_available"]:
            recommendations["intraday_risk_monitoring"] = True
            recommendations["dynamic_stop_adjustment"] = True

        return recommendations

    def _generate_risk_decision_enhanced(self, volatility_analysis: Dict[str, Any], statistical_risk: Dict[str, Any],
                                       stability_analysis: Dict[str, Any], market_conditions: MarketConditions,
                                       data_assessment: Dict[str, Any]) -> str:
        """Enhanced risk decision generation with data quality considerations."""
        # Data quality threshold check
        if data_assessment["quality_score"] < 0.3:
            return "NO_SIGNAL"  # Insufficient data quality for reliable risk assessment

        # Base risk decision
        base_decision = self._generate_risk_decision(volatility_analysis, statistical_risk, stability_analysis, market_conditions)

        # Data quality modifications
        quality_score = data_assessment["quality_score"]

        # Be more conservative with limited data
        if quality_score < 0.6:
            if base_decision == "HOLD":
                # Check if risk conditions warrant more caution
                if (volatility_analysis.get("volatility_regime") in ["HIGH", "ELEVATED"] or
                    statistical_risk.get("statistical_regime") in ["EXTREME", "UNUSUAL"]):
                    return "NO_SIGNAL"

        # Enhanced decision making with high-quality data
        elif quality_score > 0.8:
            # Can be more nuanced with high-quality data
            if data_assessment["tick_data_available"]:
                tail_risk_level = statistical_risk.get("tail_risk_level", "MEDIUM")
                if tail_risk_level == "HIGH":
                    return "NO_SIGNAL"  # Tick data reveals high tail risk

        return base_decision
    def _calculate_risk_confidence_enhanced(self, volatility_analysis: Dict[str, Any], statistical_risk: Dict[str, Any],
                                          stability_analysis: Dict[str, Any], data_assessment: Dict[str, Any]) -> float:
        """Enhanced risk confidence calculation incorporating data quality."""
        # Base confidence calculation
        base_confidence = self._calculate_risk_confidence(volatility_analysis, statistical_risk, stability_analysis)

        # Data quality factor (25% weight)
        quality_score = data_assessment["quality_score"]
        quality_factor = quality_score * 0.25

        # Historical depth factor (15% weight)
        historical_depth = data_assessment["historical_depth"]
        depth_factor = min(historical_depth / 100, 1.0) * 0.15

        # Data source reliability factor (10% weight)
        reliability_factor = 0.0
        if data_assessment["risk_calculation_reliability"] == "HIGH":
            reliability_factor = 0.10
        elif data_assessment["risk_calculation_reliability"] == "MEDIUM":
            reliability_factor = 0.05

        # Enhanced confidence with high-quality data
        enhanced_confidence = base_confidence + quality_factor + depth_factor + reliability_factor

        # Penalty for insufficient data
        if data_assessment["quality_score"] < 0.5:
            enhanced_confidence *= 0.8

        return max(0.0, min(enhanced_confidence, 1.0))

    def _calculate_overall_risk_score_enhanced(self, volatility_analysis: Dict[str, Any], statistical_risk: Dict[str, Any],
                                             stability_analysis: Dict[str, Any], data_assessment: Dict[str, Any]) -> float:
        """Enhanced overall risk score calculation with data quality adjustments."""
        # Base risk score
        base_risk_score = self._calculate_overall_risk_score(volatility_analysis, statistical_risk, stability_analysis)

        # Data quality adjustments
        quality_score = data_assessment["quality_score"]

        # Increase risk score for low-quality data (more conservative)
        if quality_score < 0.6:
            quality_penalty = (0.6 - quality_score) * 0.5  # Up to 0.3 penalty
            adjusted_risk_score = min(base_risk_score + quality_penalty, 1.0)
        else:
            # Fine-tune risk score with high-quality data
            if data_assessment["tick_data_available"]:
                tail_risk_level = statistical_risk.get("tail_risk_level", "MEDIUM")
                if tail_risk_level == "HIGH":
                    adjusted_risk_score = min(base_risk_score + 0.1, 1.0)
                elif tail_risk_level == "LOW":
                    adjusted_risk_score = max(base_risk_score - 0.05, 0.0)
                else:
                    adjusted_risk_score = base_risk_score
            else:
                adjusted_risk_score = base_risk_score

        return adjusted_risk_score

    def _generate_risk_reasoning_enhanced(self, decision: str, volatility_analysis: Dict[str, Any],
                                        statistical_risk: Dict[str, Any], stability_analysis: Dict[str, Any],
                                        data_assessment: Dict[str, Any]) -> str:
        """Enhanced risk reasoning including data quality context."""
        reasoning_parts = []

        # Base reasoning
        base_reasoning = self._generate_risk_reasoning(decision, volatility_analysis, statistical_risk, stability_analysis)
        reasoning_parts.append(base_reasoning)

        # Data quality context
        quality_score = data_assessment["quality_score"]
        data_source = data_assessment["data_source"]
        reasoning_parts.append(f"Risk analysis based on {data_source} data with {quality_score:.2f} quality score.")

        # Historical depth context
        historical_depth = data_assessment["historical_depth"]
        reasoning_parts.append(f"Analysis uses {historical_depth} historical data points.")

        # Enhanced insights with high-quality data
        if data_assessment["tick_data_available"]:
            reasoning_parts.append("Tick-level data provides enhanced volatility and tail risk assessment.")

            tail_risk_level = statistical_risk.get("tail_risk_level", "MEDIUM")
            if tail_risk_level != "MEDIUM":
                reasoning_parts.append(f"Tick analysis reveals {tail_risk_level} tail risk conditions.")

        elif data_assessment["intraday_data_available"]:
            reasoning_parts.append("Intraday data enables improved volatility pattern analysis.")

        # Data quality warnings or enhancements
        if quality_score < 0.5:
            reasoning_parts.append("Conservative risk approach due to limited data quality.")
        elif quality_score > 0.8:
            reasoning_parts.append("High data quality enables refined risk assessment.")

        # Risk calculation confidence
        calc_confidence = volatility_analysis.get("calculation_confidence", 0.0)
        if calc_confidence < 0.7:
            reasoning_parts.append("Volatility calculations have reduced confidence due to data limitations.")

        return " ".join(reasoning_parts)
