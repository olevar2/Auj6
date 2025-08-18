"""
Microstructure Agent for the AUJ Platform.

This agent specializes in market microstructure analysis using tick data.
It focuses on 14 microstructure and order flow indicators.
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


class MicrostructureAgent(BaseAgent):
    """
    Microstructure Agent - Market microstructure and tick data analysis.

    Specializes in:
    - Order flow analysis
    - Tick-by-tick price movements
    - Market maker behavior detection
    - Liquidity provision patterns
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager: Optional[UnifiedConfigManager] = None, messaging_service: Optional[Any] = None):
        """Initialize the Microstructure Agent."""
        assigned_indicators = [
            # Order Flow & Microstructure Indicators
            "bid_ask_spread_analyzer_indicator",
            "institutional_flow_detector",
            "liquidity_flow_indicator",
            "market_microstructure_indicator",
            "market_profile_value_area_indicator",
            "order_flow_block_trade_detector",
            "order_flow_imbalance_indicator",
            "order_flow_sequence_analyzer",
            "smart_money_indicators",
            "tick_volume_analyzer",
            "tick_volume_indicators",
            "volume_weighted_market_depth_indicator",

            # Advanced Microstructure
            "block_trade_signal",
            "institutional_flow_signal",
            "liquidity_flow_signal",
            "order_flow_sequence_signal"
        ]

        super().__init__(
            name="MicrostructureAgent",
            specialization="Market microstructure and tick data analysis",
            assigned_indicators=assigned_indicators,
            config=config,
            config_manager=config_manager
        )

        # Store messaging service
        self.messaging_service = messaging_service

        # Load configuration from YAML file
        self._load_agent_config()

        logger.info(f"MicrostructureAgent initialized with {len(assigned_indicators)} indicators")

    def _load_agent_config(self):
        """Load agent-specific configuration from YAML file."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'microstructure_agent.yaml'
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
        self.order_flow_threshold = self.config_manager.get_float('agents.microstructure_agent.order_flow_threshold', 0.6)
        self.tick_direction_sensitivity = self.config_manager.get_float('agents.microstructure_agent.tick_direction_sensitivity', 0.1)
        self.market_maker_threshold = self.config_manager.get_float('agents.microstructure_agent.market_maker_threshold', 0.7)
        self.liquidity_provision_threshold = self.config_manager.get_float('agents.microstructure_agent.liquidity_provision_threshold', 0.5)

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform microstructure analysis with graceful fallback for missing tick data."""
        try:
            # Check data availability and determine analysis mode
            tick_data_available = self._check_data_availability(market_data, 'TICK')
            order_book_available = self._check_data_availability(market_data, 'ORDER_BOOK')
            ohlcv_available = self._check_data_availability(market_data, 'OHLCV')

            # Validate minimum data requirements
            if not ohlcv_available:
                raise AgentError(f"No OHLCV data available for {symbol} - cannot perform any analysis")

            # Log data availability and switch to fallback mode if needed
            missing_data = []
            if not tick_data_available:
                missing_data.append('TICK')
            if not order_book_available:
                missing_data.append('ORDER_BOOK')

            if missing_data:
                logger.warning(f"MicrostructureAgent for {symbol}: Missing {', '.join(missing_data)} data. "
                             f"Operating in fallback mode using OHLCV data only. "
                             f"Analysis accuracy will be significantly reduced.")

            # Adapt analysis based on available data
            if tick_data_available:
                # Full microstructure analysis with tick data
                flow_analysis = self._analyze_order_flow(indicator_results, market_data.get('TICK'))
                tick_analysis = self._analyze_tick_dynamics(indicator_results, market_data.get('TICK'))
                maker_analysis = self._analyze_market_makers(indicator_results)
            else:
                # Fallback analysis using OHLCV data (limited capabilities)
                flow_analysis = self._analyze_order_flow_fallback(indicator_results, market_data.get('OHLCV'))
                tick_analysis = self._analyze_tick_dynamics_fallback(indicator_results, market_data.get('OHLCV'))
                maker_analysis = self._analyze_market_makers_fallback(indicator_results)

            # Generate Decision
            decision = self._generate_microstructure_decision(flow_analysis, tick_analysis, maker_analysis)

            # Calculate Confidence (significantly reduced for fallback mode)
            confidence = self._calculate_microstructure_confidence(flow_analysis, tick_analysis, maker_analysis)
            if missing_data:
                confidence *= 0.5  # Reduce confidence by 50% when using fallback

            # Generate Reasoning
            reasoning = self._generate_microstructure_reasoning(decision, flow_analysis, tick_analysis, maker_analysis)
            if missing_data:
                reasoning += f" Note: Analysis performed in fallback mode due to missing {', '.join(missing_data)} data, accuracy significantly reduced."

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "order_flow_analysis": flow_analysis,
                    "tick_analysis": tick_analysis,
                    "market_maker_analysis": maker_analysis
                },
                risk_assessment=self._assess_microstructure_risk(flow_analysis, maker_analysis),
                supporting_data={
                    "order_flow_imbalance": flow_analysis.get("imbalance_score", 0.0),
                    "tick_direction": tick_analysis.get("direction_score", 0.0),
                    "market_maker_activity": maker_analysis.get("activity_level", 0.0),
                    "liquidity_quality": maker_analysis.get("liquidity_quality", "MEDIUM"),
                    "fallback_mode": bool(missing_data)
                }
            )

        except Exception as e:
            logger.error(f"MicrostructureAgent analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Microstructure analysis failed: {str(e)}")

    def _check_data_availability(self, market_data: Dict[str, pd.DataFrame], data_type: str) -> bool:
        """Check if specific data type is available and valid."""
        data = market_data.get(data_type)
        return data is not None and not data.empty

    def _check_tick_data_availability(self, market_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if tick data is available for microstructure analysis (deprecated - use _check_data_availability)."""
        return self._check_data_availability(market_data, 'TICK')

    def _analyze_order_flow(self, indicator_results: Dict[str, Any], tick_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze order flow patterns using tick data."""
        flow_analysis = {
            "imbalance_score": 0.0,
            "flow_direction": "NEUTRAL",
            "buy_pressure": 0.0,
            "sell_pressure": 0.0,
            "flow_persistence": "WEAK",
            "aggressive_buying": False,
            "aggressive_selling": False,
            "data_quality": "HIGH" if tick_data is not None else "LOW"
        }

        # Enhanced tick data analysis
        if tick_data is not None:
            flow_analysis.update(self._calculate_tick_order_flow(tick_data))

        # Order flow imbalance from indicators
        if "order_flow_imbalance_indicator" in indicator_results:
            imb_data = indicator_results["order_flow_imbalance_indicator"]
            flow_analysis["imbalance_score"] = imb_data.get("imbalance", 0.0)
            flow_analysis["buy_pressure"] = imb_data.get("buy_pressure", 0.0)
            flow_analysis["sell_pressure"] = imb_data.get("sell_pressure", 0.0)

            # Determine flow direction
            if flow_analysis["imbalance_score"] > self.order_flow_threshold:
                flow_analysis["flow_direction"] = "BULLISH"
                flow_analysis["aggressive_buying"] = True
            elif flow_analysis["imbalance_score"] < -self.order_flow_threshold:
                flow_analysis["flow_direction"] = "BEARISH"
                flow_analysis["aggressive_selling"] = True

        # Trade size distribution analysis
        if "trade_size_distribution_indicator" in indicator_results:
            size_data = indicator_results["trade_size_distribution_indicator"]
            large_trade_ratio = size_data.get("large_trade_ratio", 0.0)

            if large_trade_ratio > 0.3:  # High proportion of large trades
                if flow_analysis["flow_direction"] == "BULLISH":
                    flow_analysis["flow_persistence"] = "STRONG"
                elif flow_analysis["flow_direction"] == "BEARISH":
                    flow_analysis["flow_persistence"] = "STRONG"
                else:
                    flow_analysis["flow_persistence"] = "MODERATE"

        return flow_analysis

    def _calculate_tick_order_flow(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate order flow metrics from tick data."""
        if tick_data.empty:
            return {}

        # Calculate tick direction (uptick vs downtick)
        tick_data = tick_data.copy()
        tick_data['price_change'] = tick_data['price'].diff()
        tick_data['tick_direction'] = np.where(tick_data['price_change'] > 0, 1,
                                              np.where(tick_data['price_change'] < 0, -1, 0))

        # Calculate buy/sell pressure
        upticks = tick_data[tick_data['tick_direction'] == 1]['volume'].sum()
        downticks = tick_data[tick_data['tick_direction'] == -1]['volume'].sum()
        total_volume = tick_data['volume'].sum()

        if total_volume > 0:
            buy_pressure = upticks / total_volume
            sell_pressure = downticks / total_volume
            imbalance = (upticks - downticks) / total_volume
        else:
            buy_pressure = sell_pressure = imbalance = 0.0

        return {
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "imbalance_score": imbalance,
            "total_ticks": len(tick_data),
            "avg_trade_size": tick_data['volume'].mean()
        }

    def _analyze_tick_dynamics(self, indicator_results: Dict[str, Any], tick_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze tick-by-tick price movements with enhanced data-aware capabilities."""
        tick_analysis = {
            "direction_score": 0.0,
            "tick_momentum": "NEUTRAL",
            "price_impact": 0.0,
            "arrival_rate": 0.0,
            "quote_updates": 0.0,
            "bid_ask_bounce": False,
            "tick_volatility": "MEDIUM",
            "data_quality": "HIGH" if tick_data is not None else "LOW"
        }

        # Enhanced tick data analysis
        if tick_data is not None:
            tick_analysis.update(self._calculate_tick_metrics(tick_data))

        # Tick direction analysis from indicators
        if "tick_direction_indicator" in indicator_results:
            tick_data_result = indicator_results["tick_direction_indicator"]
            tick_analysis["direction_score"] = tick_data_result.get("direction_score", 0.0)

            if abs(tick_analysis["direction_score"]) > 0.5:
                if tick_analysis["direction_score"] > 0:
                    tick_analysis["tick_momentum"] = "BULLISH"
                else:
                    tick_analysis["tick_momentum"] = "BEARISH"

        # Price impact per trade
        if "price_impact_per_trade_indicator" in indicator_results:
            impact_data = indicator_results["price_impact_per_trade_indicator"]
            tick_analysis["price_impact"] = impact_data.get("average_impact", 0.0)

        # Trade arrival rate
        if "trade_arrival_rate_indicator" in indicator_results:
            arrival_data = indicator_results["trade_arrival_rate_indicator"]
            tick_analysis["arrival_rate"] = arrival_data.get("arrival_rate", 0.0)

        # Quote update frequency
        if "quote_update_frequency_indicator" in indicator_results:
            quote_data = indicator_results["quote_update_frequency_indicator"]
            tick_analysis["quote_updates"] = quote_data.get("update_frequency", 0.0)

        # Bid-ask bounce detection
        if "bid_ask_bounce_indicator" in indicator_results:
            bounce_data = indicator_results["bid_ask_bounce_indicator"]
            tick_analysis["bid_ask_bounce"] = bounce_data.get("bounce_detected", False)

    def _calculate_tick_metrics(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed tick metrics from raw tick data."""
        if tick_data.empty:
            return {}

        # Price movement analysis
        tick_data = tick_data.copy()
        tick_data['price_change'] = tick_data['price'].diff()
        tick_data['price_change_abs'] = tick_data['price_change'].abs()

        # Direction score calculation
        positive_moves = (tick_data['price_change'] > 0).sum()
        negative_moves = (tick_data['price_change'] < 0).sum()
        total_moves = positive_moves + negative_moves

        if total_moves > 0:
            direction_score = (positive_moves - negative_moves) / total_moves
        else:
            direction_score = 0.0

        # Arrival rate (ticks per minute)
        if len(tick_data) > 1:
            time_span = (tick_data['timestamp'].iloc[-1] - tick_data['timestamp'].iloc[0]).total_seconds() / 60
            arrival_rate = len(tick_data) / max(time_span, 1)
        else:
            arrival_rate = 0.0

        # Volatility estimation
        price_volatility = tick_data['price_change'].std()
        vol_level = "HIGH" if price_volatility > tick_data['price'].mean() * 0.001 else "MEDIUM"
        vol_level = "LOW" if price_volatility < tick_data['price'].mean() * 0.0005 else vol_level

        return {
            "direction_score": direction_score,
            "arrival_rate": arrival_rate,
            "tick_volatility": vol_level,
            "avg_price_impact": tick_data['price_change_abs'].mean(),
            "tick_count": len(tick_data)
        }

    def _analyze_order_flow_fallback(self, indicator_results: Dict[str, Any], ohlcv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fallback order flow analysis using OHLCV data."""
        flow_analysis = {
            "imbalance_score": 0.0,
            "flow_direction": "NEUTRAL",
            "buy_pressure": 0.0,
            "sell_pressure": 0.0,
            "flow_persistence": "WEAK",
            "aggressive_buying": False,
            "aggressive_selling": False,
            "data_quality": "LOW"
        }

        # Use volume and price action as proxies
        if ohlcv_data is not None and not ohlcv_data.empty:
            # Estimate buy/sell pressure from volume and price movement
            latest = ohlcv_data.iloc[-1]
            price_move = (latest['close'] - latest['open']) / latest['open']
            volume_ratio = latest['volume'] / ohlcv_data['volume'].mean()

            # Simple heuristic for order flow
            if price_move > 0.001 and volume_ratio > 1.2:
                flow_analysis.update({
                    "flow_direction": "BULLISH",
                    "buy_pressure": min(0.6 + price_move * 10, 1.0),
                    "sell_pressure": max(0.4 - price_move * 10, 0.0),
                    "imbalance_score": price_move * volume_ratio
                })
            elif price_move < -0.001 and volume_ratio > 1.2:
                flow_analysis.update({
                    "flow_direction": "BEARISH",
                    "buy_pressure": max(0.4 + price_move * 10, 0.0),
                    "sell_pressure": min(0.6 - price_move * 10, 1.0),
                    "imbalance_score": price_move * volume_ratio
                })

        return flow_analysis

    def _analyze_market_makers(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market maker behavior and liquidity provision."""
        maker_analysis = {
            "activity_level": 0.0,
            "liquidity_quality": "MEDIUM",
            "provision_score": 0.0,
            "inventory_risk": "MEDIUM",
            "adverse_selection": 0.0,
            "effective_spread": 0.0,
            "realized_spread": 0.0,
            "maker_presence": False
        }

        # Market maker indicator
        if "market_maker_indicator" in indicator_results:
            mm_data = indicator_results["market_maker_indicator"]
            maker_analysis["activity_level"] = mm_data.get("activity_level", 0.0)
            maker_analysis["maker_presence"] = mm_data.get("presence_detected", False)

        # Liquidity provision
        if "liquidity_provision_indicator" in indicator_results:
            liq_data = indicator_results["liquidity_provision_indicator"]
            maker_analysis["provision_score"] = liq_data.get("provision_score", 0.0)

            if maker_analysis["provision_score"] > 0.8:
                maker_analysis["liquidity_quality"] = "HIGH"
            elif maker_analysis["provision_score"] < 0.3:
                maker_analysis["liquidity_quality"] = "LOW"

        # Inventory risk
        if "inventory_risk_indicator" in indicator_results:
            inv_data = indicator_results["inventory_risk_indicator"]
            risk_level = inv_data.get("risk_level", "MEDIUM")
            maker_analysis["inventory_risk"] = risk_level

        # Adverse selection
        if "adverse_selection_indicator" in indicator_results:
            adv_data = indicator_results["adverse_selection_indicator"]
            maker_analysis["adverse_selection"] = adv_data.get("adverse_selection", 0.0)

        # Spread analysis
        if "effective_spread_indicator" in indicator_results:
            eff_data = indicator_results["effective_spread_indicator"]
            maker_analysis["effective_spread"] = eff_data.get("effective_spread", 0.0)

        if "realized_spread_indicator" in indicator_results:
            real_data = indicator_results["realized_spread_indicator"]
            maker_analysis["realized_spread"] = real_data.get("realized_spread", 0.0)

        return maker_analysis

    def _generate_microstructure_decision(self, flow_analysis: Dict[str, Any], tick_analysis: Dict[str, Any], maker_analysis: Dict[str, Any]) -> str:
        """Generate decision based on microstructure analysis."""
        flow_direction = flow_analysis.get("flow_direction", "NEUTRAL")
        tick_momentum = tick_analysis.get("tick_momentum", "NEUTRAL")
        liquidity_quality = maker_analysis.get("liquidity_quality", "MEDIUM")
        aggressive_buying = flow_analysis.get("aggressive_buying", False)
        aggressive_selling = flow_analysis.get("aggressive_selling", False)

        # Strong microstructure signals
        if flow_direction == "BULLISH" and tick_momentum == "BULLISH" and liquidity_quality in ["MEDIUM", "HIGH"]:
            if aggressive_buying:
                return "BUY"

        if flow_direction == "BEARISH" and tick_momentum == "BEARISH" and liquidity_quality in ["MEDIUM", "HIGH"]:
            if aggressive_selling:
                return "SELL"

        # Moderate signals with good liquidity
        if liquidity_quality == "HIGH":
            if flow_direction == "BULLISH" or tick_momentum == "BULLISH":
                return "BUY"
            elif flow_direction == "BEARISH" or tick_momentum == "BEARISH":
                return "SELL"

        # Poor liquidity conditions
        if liquidity_quality == "LOW":
            return "HOLD"  # Avoid trading in poor liquidity

        return "NO_SIGNAL"

    def _calculate_microstructure_confidence(self, flow_analysis: Dict[str, Any], tick_analysis: Dict[str, Any], maker_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in microstructure analysis."""
        factors = []

        # Order flow confidence
        imbalance_score = abs(flow_analysis.get("imbalance_score", 0.0))
        factors.append(min(imbalance_score, 1.0) * 0.4)

        # Tick momentum confidence
        direction_score = abs(tick_analysis.get("direction_score", 0.0))
        factors.append(direction_score * 0.3)

        # Liquidity quality confidence
        liquidity_quality = maker_analysis.get("liquidity_quality", "MEDIUM")
        if liquidity_quality == "HIGH":
            factors.append(0.2)
        elif liquidity_quality == "MEDIUM":
            factors.append(0.1)

        # Market maker presence confidence
        if maker_analysis.get("maker_presence", False):
            factors.append(0.1)

        return min(sum(factors), 1.0)

    def _assess_microstructure_risk(self, flow_analysis: Dict[str, Any], maker_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess microstructure-related risks."""
        liquidity_quality = maker_analysis.get("liquidity_quality", "MEDIUM")
        inventory_risk = maker_analysis.get("inventory_risk", "MEDIUM")
        adverse_selection = maker_analysis.get("adverse_selection", 0.0)

        # Overall microstructure risk
        if liquidity_quality == "LOW" or inventory_risk == "HIGH":
            micro_risk = "HIGH"
        elif adverse_selection > 0.5:
            micro_risk = "HIGH"
        elif liquidity_quality == "HIGH" and inventory_risk == "LOW":
            micro_risk = "LOW"
        else:
            micro_risk = "MEDIUM"

        return {
            "microstructure_risk": micro_risk,
            "liquidity_risk": liquidity_quality,
            "inventory_risk": inventory_risk,
            "adverse_selection_risk": "HIGH" if adverse_selection > 0.5 else "MEDIUM"
        }

    def _generate_microstructure_reasoning(self, decision: str, flow_analysis: Dict[str, Any], tick_analysis: Dict[str, Any], maker_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for microstructure analysis."""
        parts = [f"Microstructure analysis decision: {decision}."]

        flow_direction = flow_analysis.get("flow_direction", "NEUTRAL")
        parts.append(f"Order flow direction: {flow_direction}.")

        imbalance_score = flow_analysis.get("imbalance_score", 0.0)
        parts.append(f"Flow imbalance: {imbalance_score:.3f}.")

        tick_momentum = tick_analysis.get("tick_momentum", "NEUTRAL")
        parts.append(f"Tick momentum: {tick_momentum}.")

        liquidity_quality = maker_analysis.get("liquidity_quality", "MEDIUM")
        parts.append(f"Liquidity quality: {liquidity_quality}.")

        if flow_analysis.get("aggressive_buying", False):
            parts.append("Aggressive buying detected.")
        elif flow_analysis.get("aggressive_selling", False):
            parts.append("Aggressive selling detected.")

        return " ".join(parts)

    def _analyze_tick_dynamics_fallback(self, indicator_results: Dict[str, Any], ohlcv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fallback tick dynamics analysis using OHLCV data."""
        tick_analysis = {
            "direction_score": 0.0,
            "tick_momentum": "NEUTRAL",
            "price_impact": 0.0,
            "arrival_rate": 0.0,
            "quote_updates": 0.0,
            "bid_ask_bounce": False,
            "tick_volatility": "MEDIUM",
            "data_quality": "LOW"
        }

        # Estimate metrics from OHLCV
        if ohlcv_data is not None and not ohlcv_data.empty:
            # Use price action as proxy for tick direction
            latest = ohlcv_data.iloc[-1]
            prev = ohlcv_data.iloc[-2] if len(ohlcv_data) > 1 else latest

            price_change = (latest['close'] - prev['close']) / prev['close']
            tick_analysis["direction_score"] = max(-1, min(1, price_change * 100))

            if abs(tick_analysis["direction_score"]) > 0.5:
                tick_analysis["tick_momentum"] = "BULLISH" if tick_analysis["direction_score"] > 0 else "BEARISH"

            # Estimate volatility from high-low range
            hl_range = (latest['high'] - latest['low']) / latest['close']
            if hl_range > 0.02:
                tick_analysis["tick_volatility"] = "HIGH"
            elif hl_range < 0.005:
                tick_analysis["tick_volatility"] = "LOW"

        return tick_analysis

    def _analyze_market_makers_fallback(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback market maker analysis without tick data."""
        maker_analysis = {
            "activity_level": 0.0,
            "liquidity_quality": "MEDIUM",
            "provision_score": 0.5,  # Default assumption
            "inventory_risk": "MEDIUM",
            "adverse_selection": 0.0,
            "effective_spread": 0.0,
            "realized_spread": 0.0,
            "maker_presence": False,
            "data_quality": "LOW"
        }

        # Use available indicator data only
        if "market_maker_indicator" in indicator_results:
            mm_data = indicator_results["market_maker_indicator"]
            maker_analysis["activity_level"] = mm_data.get("activity_level", 0.0)
            maker_analysis["maker_presence"] = mm_data.get("presence_detected", False)

        if "liquidity_provision_indicator" in indicator_results:
            liq_data = indicator_results["liquidity_provision_indicator"]
            maker_analysis["provision_score"] = liq_data.get("provision_score", 0.5)

            if maker_analysis["provision_score"] > 0.8:
                maker_analysis["liquidity_quality"] = "HIGH"
            elif maker_analysis["provision_score"] < 0.3:
                maker_analysis["liquidity_quality"] = "LOW"

        return maker_analysis

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        return [ind for ind in self.assigned_indicators if ind in indicator_results]

    def get_required_data_types(self) -> List[str]:
        """Define required data types."""
        return ["TICK", "ORDER_BOOK"]  # Requires tick data and order book

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed."""
        return 100  # Need substantial tick data for microstructure analysis
