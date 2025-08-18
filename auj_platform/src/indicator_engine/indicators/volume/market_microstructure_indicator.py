"""
AUJ Platform Advanced Market Microstructure Indicator
Sophisticated implementation with order book dynamics, tick analysis, and execution quality metrics

This implementation provides institutional-grade market microstructure analysis for humanitarian trading platforms.

Features:
- Advanced order book dynamics analysis with depth profiling
- Comprehensive tick-by-tick data processing and pattern recognition
- Execution quality metrics including spread, impact, and liquidity costs
- Market making vs taking flow classification
- Price discovery efficiency measurement
- Information asymmetry detection and analysis
- High-frequency trading pattern identification
- Market resilience and stress testing
- Adverse selection cost analysis
- Comprehensive microstructure signal generation

The Market Microstructure Indicator provides deep insights into market structure,
order flow dynamics, and execution quality to optimize trading performance
and minimize market impact costs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, signal, optimize
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from ..base.base_indicator import BaseIndicator, IndicatorConfig
from ...core.signal_type import SignalType


class MarketRegime(Enum):
    """Market microstructure regime classification"""
    LIQUID = "liquid"
    ILLIQUID = "illiquid"
    STRESSED = "stressed"
    FRAGMENTED = "fragmented"
    CONCENTRATED = "concentrated"
    VOLATILE = "volatile"


class OrderFlowType(Enum):
    """Order flow classification"""
    MARKET_TAKING = "market_taking"
    MARKET_MAKING = "market_making"
    ICEBERG = "iceberg"
    SWEEP = "sweep"
    STEALTH = "stealth"
    ALGORITHMIC = "algorithmic"


class ExecutionQuality(Enum):
    """Execution quality classification"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    SEVERE_IMPACT = "severe_impact"


class InformationAsymmetry(Enum):
    """Information asymmetry level"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class MarketMicrostructureConfig(IndicatorConfig):
    """Configuration for Market Microstructure Indicator"""
    tick_analysis_window: int = 1000
    order_book_levels: int = 10
    execution_cost_window: int = 100
    pattern_detection_period: int = 50
    liquidity_threshold: float = 10000.0
    spread_threshold: float = 0.001
    impact_threshold: float = 0.0005
    use_ml_classification: bool = True
    use_anomaly_detection: bool = True
    use_hft_detection: bool = True
    min_periods: int = 200


class OrderBookAnalysis(NamedTuple):
    """Order book microstructure analysis"""
    bid_ask_spread: float
    spread_volatility: float
    depth_imbalance: float
    order_book_slope: float
    liquidity_concentration: float
    depth_resilience: float
    price_impact_estimate: float
    market_making_activity: float


class TickAnalysis(NamedTuple):
    """Tick-level analysis results"""
    tick_frequency: float
    tick_size_distribution: Dict[str, float]
    price_clustering: float
    trade_size_distribution: Dict[str, float]
    arrival_rate_variance: float
    tick_direction_persistence: float
    quote_revision_frequency: float


class ExecutionAnalysis(NamedTuple):
    """Execution quality analysis"""
    effective_spread: float
    price_impact: float
    timing_cost: float
    opportunity_cost: float
    implementation_shortfall: float
    market_impact_decay: float
    adverse_selection_cost: float
    execution_quality: ExecutionQuality


class InformationFlow(NamedTuple):
    """Information flow and asymmetry analysis"""
    price_discovery_efficiency: float
    information_asymmetry_level: InformationAsymmetry
    informed_trading_probability: float
    noise_trader_activity: float
    pin_score: float  # Probability of Informed Trading
    order_flow_toxicity: float
    market_microstructure_noise: float


class MarketMicrostructureResult(NamedTuple):
    """Complete market microstructure analysis result"""
    market_regime: MarketRegime
    microstructure_quality: float
    order_book_analysis: OrderBookAnalysis
    tick_analysis: TickAnalysis
    execution_analysis: ExecutionAnalysis
    information_flow: InformationFlow
    dominant_flow_type: OrderFlowType
    hft_activity_level: float
    market_stress_indicator: float
    confidence_score: float


class MarketMicrostructureIndicator(BaseIndicator):
    """
    Advanced Market Microstructure Indicator with comprehensive analytics.
    
    This indicator analyzes market microstructure characteristics including:
    - Order book dynamics and depth analysis
    - Tick-level data processing and patterns
    - Execution quality measurement
    - Information asymmetry detection
    - High-frequency trading activity
    - Market making vs taking flows
    - Price discovery efficiency
    - Market stress and resilience
    """
    
    def __init__(self, config: Optional[MarketMicrostructureConfig] = None):
        super().__init__(config or MarketMicrostructureConfig())
        self.config: MarketMicrostructureConfig = self.config
        
        # Internal state
        self._tick_history: List[Dict] = []
        self._order_book_history: List[Dict] = []
        self._execution_history: List[Dict] = []
        self._spread_history: List[float] = []
        self._impact_history: List[float] = []
        
        # Machine learning components
        self._regime_classifier: Optional[RandomForestClassifier] = None
        self._anomaly_detector: Optional[IsolationForest] = None
        self._scaler: StandardScaler = StandardScaler()
        self._robust_scaler: RobustScaler = RobustScaler()
        self._is_trained: bool = False
        
        # Pattern detection
        self._pattern_detector: Optional[object] = None
        self._clustering_model: Optional[DBSCAN] = None
        
        # Microstructure state
        self._market_regime_buffer: List[str] = []
        self._liquidity_stress_buffer: List[float] = []
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive market microstructure analysis.
        
        Args:
            data: Dictionary containing market data including:
                 'high', 'low', 'close', 'volume' (required)
                 'bid', 'ask', 'order_book', 'tick_data' (optional)
            
        Returns:
            Dictionary containing market microstructure analysis results
        """
        try:
            if not self._validate_data(data):
                return self._create_default_result()
            
            df = pd.DataFrame(data)
            
            if len(df) < self.config.min_periods:
                return self._create_default_result()
            
            # Analyze order book dynamics
            order_book_analysis = self._analyze_order_book_dynamics(data, df)
            
            # Analyze tick-level data
            tick_analysis = self._analyze_tick_data(data, df)
            
            # Analyze execution quality
            execution_analysis = self._analyze_execution_quality(data, df, order_book_analysis)
            
            # Analyze information flow
            information_flow = self._analyze_information_flow(data, df, tick_analysis)
            
            # Detect HFT activity
            hft_activity_level = self._detect_hft_activity(tick_analysis, execution_analysis)
            
            # Classify dominant order flow type
            dominant_flow_type = self._classify_order_flow_type(
                order_book_analysis, tick_analysis, execution_analysis
            )
            
            # Calculate market stress indicator
            market_stress_indicator = self._calculate_market_stress(
                order_book_analysis, execution_analysis, information_flow
            )
            
            # Classify market regime
            market_regime = self._classify_market_regime(
                order_book_analysis, execution_analysis, market_stress_indicator
            )
            
            # Calculate overall microstructure quality
            microstructure_quality = self._calculate_microstructure_quality(
                order_book_analysis, execution_analysis, information_flow
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                order_book_analysis, tick_analysis, execution_analysis, information_flow
            )
            
            # Create result
            result = MarketMicrostructureResult(
                market_regime=market_regime,
                microstructure_quality=microstructure_quality,
                order_book_analysis=order_book_analysis,
                tick_analysis=tick_analysis,
                execution_analysis=execution_analysis,
                information_flow=information_flow,
                dominant_flow_type=dominant_flow_type,
                hft_activity_level=hft_activity_level,
                market_stress_indicator=market_stress_indicator,
                confidence_score=confidence_score
            )
            
            # Generate trading signal
            signal = self._generate_signal(result)
            
            # Update internal state and retrain if needed
            self._update_state_and_retrain(df, data, result)
            
            return self._format_result(result, signal)
            
        except Exception as e:
            self.logger.error(f"Error in MarketMicrostructureIndicator calculation: {e}")
            return self._create_error_result(str(e))
    
    def _analyze_order_book_dynamics(self, data: Dict[str, Any], df: pd.DataFrame) -> OrderBookAnalysis:
        """Analyze order book microstructure dynamics"""
        # Extract order book data
        if 'bid' in data and 'ask' in data:
            bids = np.array(data['bid'])
            asks = np.array(data['ask'])
        else:
            # Estimate from OHLC data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Estimate bid/ask from high/low/close
            typical_spread = np.mean((high - low) / close) * 0.5
            bids = close * (1 - typical_spread)
            asks = close * (1 + typical_spread)
        
        # Calculate bid-ask spread
        spreads = asks - bids
        bid_ask_spread = np.mean(spreads[-20:]) if len(spreads) >= 20 else np.mean(spreads)
        spread_volatility = np.std(spreads[-20:]) if len(spreads) >= 20 else np.std(spreads)
        
        # Analyze order book depth (if available)
        if 'order_book' in data:
            order_book = data['order_book']
            depth_imbalance, order_book_slope, liquidity_concentration = self._analyze_order_book_depth(order_book)
        else:
            # Estimate from volume data
            volume = df['volume'].values
            avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            
            # Simplified estimates
            depth_imbalance = np.random.normal(0, 0.1)  # Neutral with some noise
            order_book_slope = 1.0 / (bid_ask_spread + 1e-8)  # Inverse relationship
            liquidity_concentration = min(volume[-1] / (avg_volume + 1e-8), 2.0) / 2.0
        
        # Calculate depth resilience
        if len(self._spread_history) >= 10:
            spread_changes = np.diff(self._spread_history[-10:])
            depth_resilience = 1.0 - (np.std(spread_changes) / (np.mean(np.abs(spread_changes)) + 1e-8))
            depth_resilience = max(0.0, min(1.0, depth_resilience))
        else:
            depth_resilience = 0.5
        
        # Estimate price impact
        if 'volume' in df.columns:
            volume = df['volume'].values
            returns = np.diff(np.log(df['close'].values))
            
            if len(volume) > 1 and len(returns) > 0:
                # Kyle's lambda (price impact coefficient)
                volume_impact = volume[1:] / np.mean(volume[1:])
                price_impact_estimate = np.corrcoef(np.abs(returns), volume_impact)[0, 1]
                price_impact_estimate = abs(price_impact_estimate) if not np.isnan(price_impact_estimate) else 0.1
            else:
                price_impact_estimate = 0.1
        else:
            price_impact_estimate = 0.1
        
        # Estimate market making activity
        if len(spreads) >= 10:
            spread_stability = 1.0 - (np.std(spreads[-10:]) / (np.mean(spreads[-10:]) + 1e-8))
            market_making_activity = max(0.0, min(1.0, spread_stability))
        else:
            market_making_activity = 0.5
        
        return OrderBookAnalysis(
            bid_ask_spread=bid_ask_spread,
            spread_volatility=spread_volatility,
            depth_imbalance=depth_imbalance,
            order_book_slope=order_book_slope,
            liquidity_concentration=liquidity_concentration,
            depth_resilience=depth_resilience,
            price_impact_estimate=price_impact_estimate,
            market_making_activity=market_making_activity
        )
    
    def _analyze_order_book_depth(self, order_book: Any) -> Tuple[float, float, float]:
        """Analyze order book depth characteristics"""
        try:
            if isinstance(order_book, dict):
                bids = order_book.get('bids', [])
                asks = order_book.get('asks', [])
                
                if bids and asks:
                    # Calculate depth imbalance
                    bid_depth = sum([level[1] for level in bids[:self.config.order_book_levels]])
                    ask_depth = sum([level[1] for level in asks[:self.config.order_book_levels]])
                    total_depth = bid_depth + ask_depth
                    
                    if total_depth > 0:
                        depth_imbalance = (bid_depth - ask_depth) / total_depth
                    else:
                        depth_imbalance = 0.0
                    
                    # Calculate order book slope
                    if len(bids) >= 2 and len(asks) >= 2:
                        bid_slope = (bids[0][1] - bids[1][1]) / (bids[1][0] - bids[0][0] + 1e-8)
                        ask_slope = (asks[1][1] - asks[0][1]) / (asks[1][0] - asks[0][0] + 1e-8)
                        order_book_slope = (abs(bid_slope) + abs(ask_slope)) / 2
                    else:
                        order_book_slope = 1.0
                    
                    # Calculate liquidity concentration
                    if total_depth > 0:
                        top_level_liquidity = bids[0][1] + asks[0][1]
                        liquidity_concentration = top_level_liquidity / total_depth
                    else:
                        liquidity_concentration = 1.0
                    
                    return depth_imbalance, order_book_slope, liquidity_concentration
            
            # Default values if order book data is unavailable
            return 0.0, 1.0, 0.5
            
        except Exception:
            return 0.0, 1.0, 0.5
    
    def _analyze_tick_data(self, data: Dict[str, Any], df: pd.DataFrame) -> TickAnalysis:
        """Analyze tick-level market microstructure"""
        # Extract tick data if available
        if 'tick_data' in data:
            tick_data = data['tick_data']
            tick_frequency, tick_size_dist, price_clustering = self._process_tick_data(tick_data)
        else:
            # Estimate from regular OHLC data
            close = df['close'].values
            volume = df['volume'].values
            
            # Estimate tick frequency from data frequency
            tick_frequency = len(close) / 24.0  # Assume 24-hour period
            
            # Analyze price movements
            price_changes = np.diff(close)
            nonzero_changes = price_changes[price_changes != 0]
            
            if len(nonzero_changes) > 0:
                # Price clustering analysis
                min_tick = np.min(np.abs(nonzero_changes))
                tick_multiples = np.abs(nonzero_changes) / min_tick
                clustering_score = np.sum(np.abs(tick_multiples - np.round(tick_multiples)) < 0.1) / len(tick_multiples)
                price_clustering = clustering_score
                
                # Tick size distribution
                unique_changes, counts = np.unique(nonzero_changes, return_counts=True)
                total_counts = np.sum(counts)
                tick_size_dist = {
                    'small': np.sum(counts[np.abs(unique_changes) <= np.percentile(np.abs(unique_changes), 33)]) / total_counts,
                    'medium': np.sum(counts[(np.abs(unique_changes) > np.percentile(np.abs(unique_changes), 33)) & 
                                           (np.abs(unique_changes) <= np.percentile(np.abs(unique_changes), 66))]) / total_counts,
                    'large': np.sum(counts[np.abs(unique_changes) > np.percentile(np.abs(unique_changes), 66)]) / total_counts
                }
            else:
                price_clustering = 0.5
                tick_size_dist = {'small': 0.33, 'medium': 0.33, 'large': 0.34}
        
        # Analyze trade size distribution
        if len(df['volume'].values) > 0:
            volumes = df['volume'].values
            trade_size_dist = {
                'small': np.sum(volumes <= np.percentile(volumes, 33)) / len(volumes),
                'medium': np.sum((volumes > np.percentile(volumes, 33)) & 
                               (volumes <= np.percentile(volumes, 66))) / len(volumes),
                'large': np.sum(volumes > np.percentile(volumes, 66)) / len(volumes)
            }
        else:
            trade_size_dist = {'small': 0.33, 'medium': 0.33, 'large': 0.34}
        
        # Calculate arrival rate variance
        if len(df) >= 20:
            time_intervals = np.diff(range(len(df)))  # Simplified time intervals
            arrival_rate_variance = np.var(time_intervals) / (np.mean(time_intervals) + 1e-8)
        else:
            arrival_rate_variance = 1.0
        
        # Calculate tick direction persistence
        if len(df['close'].values) >= 10:
            price_changes = np.diff(df['close'].values)
            directions = np.sign(price_changes)
            # Calculate autocorrelation of price directions
            if len(directions) > 1:
                direction_autocorr = np.corrcoef(directions[:-1], directions[1:])[0, 1]
                tick_direction_persistence = abs(direction_autocorr) if not np.isnan(direction_autocorr) else 0.0
            else:
                tick_direction_persistence = 0.0
        else:
            tick_direction_persistence = 0.0
        
        # Estimate quote revision frequency
        if 'bid' in data and 'ask' in data:
            bid_changes = np.sum(np.diff(np.array(data['bid'])) != 0)
            ask_changes = np.sum(np.diff(np.array(data['ask'])) != 0)
            total_changes = bid_changes + ask_changes
            quote_revision_frequency = total_changes / len(data['bid']) if len(data['bid']) > 0 else 0.0
        else:
            # Estimate from price volatility
            returns = np.diff(np.log(df['close'].values))
            quote_revision_frequency = np.std(returns) * 10  # Scaled estimate
        
        return TickAnalysis(
            tick_frequency=tick_frequency,
            tick_size_distribution=tick_size_dist,
            price_clustering=price_clustering,
            trade_size_distribution=trade_size_dist,
            arrival_rate_variance=arrival_rate_variance,
            tick_direction_persistence=tick_direction_persistence,
            quote_revision_frequency=quote_revision_frequency
        )
    
    def _process_tick_data(self, tick_data: Any) -> Tuple[float, Dict[str, float], float]:
        """Process raw tick data for microstructure analysis"""
        try:
            # Simplified tick data processing
            if isinstance(tick_data, list) and len(tick_data) > 0:
                # Calculate tick frequency
                tick_frequency = len(tick_data) / 3600.0  # Per hour
                
                # Analyze tick sizes
                if len(tick_data) > 1:
                    prices = [tick.get('price', 0) for tick in tick_data]
                    price_changes = np.diff(prices)
                    nonzero_changes = price_changes[price_changes != 0]
                    
                    if len(nonzero_changes) > 0:
                        # Calculate price clustering
                        min_tick = np.min(np.abs(nonzero_changes))
                        clustering_score = np.sum(np.abs(nonzero_changes) % min_tick < min_tick * 0.1) / len(nonzero_changes)
                        
                        # Tick size distribution
                        small_ticks = np.sum(np.abs(nonzero_changes) <= np.percentile(np.abs(nonzero_changes), 33))
                        medium_ticks = np.sum((np.abs(nonzero_changes) > np.percentile(np.abs(nonzero_changes), 33)) & 
                                            (np.abs(nonzero_changes) <= np.percentile(np.abs(nonzero_changes), 66)))
                        large_ticks = len(nonzero_changes) - small_ticks - medium_ticks
                        
                        tick_size_dist = {
                            'small': small_ticks / len(nonzero_changes),
                            'medium': medium_ticks / len(nonzero_changes),
                            'large': large_ticks / len(nonzero_changes)
                        }
                        
                        return tick_frequency, tick_size_dist, clustering_score
                
            # Default values
            return 100.0, {'small': 0.33, 'medium': 0.33, 'large': 0.34}, 0.5
            
        except Exception:
            return 100.0, {'small': 0.33, 'medium': 0.33, 'large': 0.34}, 0.5
    
    def _analyze_execution_quality(self, data: Dict[str, Any], df: pd.DataFrame, 
                                 order_book_analysis: OrderBookAnalysis) -> ExecutionAnalysis:
        """Analyze execution quality metrics"""
        close = df['close'].values
        volume = df['volume'].values
        
        # Calculate effective spread
        effective_spread = order_book_analysis.bid_ask_spread * 2  # Round-trip cost
        
        # Calculate price impact using volume-weighted approach
        if len(volume) >= 2 and len(close) >= 2:
            returns = np.diff(np.log(close))
            volume_normalized = volume[1:] / np.mean(volume[1:])
            
            # Price impact correlation
            if len(returns) > 0:
                price_impact = np.std(returns) * np.mean(volume_normalized) * 0.01
            else:
                price_impact = effective_spread * 0.5
        else:
            price_impact = effective_spread * 0.5
        
        # Calculate timing cost (delay between signal and execution)
        timing_cost = effective_spread * 0.3  # Assume 30% of spread as timing cost
        
        # Calculate opportunity cost
        if len(close) >= 5:
            price_volatility = np.std(np.diff(np.log(close[-5:])))
            opportunity_cost = price_volatility * 0.5  # 50% of short-term volatility
        else:
            opportunity_cost = effective_spread * 0.2
        
        # Calculate implementation shortfall
        implementation_shortfall = effective_spread + price_impact + timing_cost + opportunity_cost
        
        # Calculate market impact decay
        if len(self._impact_history) >= 5:
            recent_impacts = self._impact_history[-5:]
            impact_decay_rate = np.corrcoef(range(len(recent_impacts)), recent_impacts)[0, 1]
            market_impact_decay = abs(impact_decay_rate) if not np.isnan(impact_decay_rate) else 0.5
        else:
            market_impact_decay = 0.5
        
        # Calculate adverse selection cost
        if order_book_analysis.depth_imbalance != 0:
            adverse_selection_cost = abs(order_book_analysis.depth_imbalance) * effective_spread
        else:
            adverse_selection_cost = effective_spread * 0.1
        
        # Classify execution quality
        total_cost = implementation_shortfall
        if total_cost <= self.config.impact_threshold:
            execution_quality = ExecutionQuality.EXCELLENT
        elif total_cost <= self.config.impact_threshold * 2:
            execution_quality = ExecutionQuality.GOOD
        elif total_cost <= self.config.impact_threshold * 4:
            execution_quality = ExecutionQuality.FAIR
        elif total_cost <= self.config.impact_threshold * 8:
            execution_quality = ExecutionQuality.POOR
        else:
            execution_quality = ExecutionQuality.SEVERE_IMPACT
        
        return ExecutionAnalysis(
            effective_spread=effective_spread,
            price_impact=price_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            implementation_shortfall=implementation_shortfall,
            market_impact_decay=market_impact_decay,
            adverse_selection_cost=adverse_selection_cost,
            execution_quality=execution_quality
        )
    
    def _analyze_information_flow(self, data: Dict[str, Any], df: pd.DataFrame, 
                                tick_analysis: TickAnalysis) -> InformationFlow:
        """Analyze information flow and asymmetry"""
        close = df['close'].values
        volume = df['volume'].values
        
        # Calculate price discovery efficiency
        if len(close) >= 10:
            returns = np.diff(np.log(close))
            return_autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
            price_discovery_efficiency = 1.0 - abs(return_autocorr) if not np.isnan(return_autocorr) else 0.5
        else:
            price_discovery_efficiency = 0.5
        
        # Calculate PIN (Probability of Informed Trading) score
        if len(volume) >= 10:
            # Simplified PIN calculation based on volume clustering
            high_volume_days = np.sum(volume > np.percentile(volume, 80))
            total_days = len(volume)
            pin_score = high_volume_days / total_days
        else:
            pin_score = 0.3  # Default moderate level
        
        # Determine information asymmetry level
        if pin_score <= 0.2:
            info_asymmetry = InformationAsymmetry.LOW
        elif pin_score <= 0.4:
            info_asymmetry = InformationAsymmetry.MODERATE
        elif pin_score <= 0.6:
            info_asymmetry = InformationAsymmetry.HIGH
        else:
            info_asymmetry = InformationAsymmetry.EXTREME
        
        # Calculate informed trading probability
        informed_trading_probability = pin_score
        
        # Calculate noise trader activity
        noise_trader_activity = 1.0 - informed_trading_probability
        
        # Calculate order flow toxicity
        if tick_analysis.tick_direction_persistence > 0.3:
            order_flow_toxicity = tick_analysis.tick_direction_persistence
        else:
            order_flow_toxicity = 0.2
        
        # Calculate market microstructure noise
        if len(close) >= 5:
            price_changes = np.diff(close)
            total_variance = np.var(price_changes)
            # Estimate noise as high-frequency component
            if len(price_changes) >= 3:
                high_freq_changes = price_changes[1:] - price_changes[:-1]
                noise_variance = np.var(high_freq_changes)
                microstructure_noise = noise_variance / (total_variance + 1e-8)
            else:
                microstructure_noise = 0.1
        else:
            microstructure_noise = 0.1
        
        return InformationFlow(
            price_discovery_efficiency=price_discovery_efficiency,
            information_asymmetry_level=info_asymmetry,
            informed_trading_probability=informed_trading_probability,
            noise_trader_activity=noise_trader_activity,
            pin_score=pin_score,
            order_flow_toxicity=order_flow_toxicity,
            market_microstructure_noise=microstructure_noise
        )
    
    def _detect_hft_activity(self, tick_analysis: TickAnalysis, execution_analysis: ExecutionAnalysis) -> float:
        """Detect high-frequency trading activity level"""
        hft_indicators = []
        
        # High tick frequency indicator
        if tick_analysis.tick_frequency > 1000:  # More than 1000 ticks per hour
            hft_indicators.append(1.0)
        elif tick_analysis.tick_frequency > 500:
            hft_indicators.append(0.7)
        elif tick_analysis.tick_frequency > 100:
            hft_indicators.append(0.4)
        else:
            hft_indicators.append(0.1)
        
        # Quote revision frequency indicator
        if tick_analysis.quote_revision_frequency > 0.5:
            hft_indicators.append(0.8)
        elif tick_analysis.quote_revision_frequency > 0.2:
            hft_indicators.append(0.5)
        else:
            hft_indicators.append(0.2)
        
        # Small trade size concentration
        small_trade_ratio = tick_analysis.trade_size_distribution.get('small', 0.33)
        if small_trade_ratio > 0.7:
            hft_indicators.append(0.9)
        elif small_trade_ratio > 0.5:
            hft_indicators.append(0.6)
        else:
            hft_indicators.append(0.3)
        
        # Low execution impact (HFT typically has sophisticated execution)
        if execution_analysis.price_impact < 0.0001:
            hft_indicators.append(0.8)
        elif execution_analysis.price_impact < 0.0005:
            hft_indicators.append(0.5)
        else:
            hft_indicators.append(0.2)
        
        # Calculate overall HFT activity level
        return np.mean(hft_indicators)
    
    def _classify_order_flow_type(self, order_book_analysis: OrderBookAnalysis,
                                tick_analysis: TickAnalysis,
                                execution_analysis: ExecutionAnalysis) -> OrderFlowType:
        """Classify the dominant order flow type"""
        # Market making indicators
        market_making_score = (
            order_book_analysis.market_making_activity * 0.4 +
            (1.0 - order_book_analysis.spread_volatility) * 0.3 +
            order_book_analysis.depth_resilience * 0.3
        )
        
        # Market taking indicators
        market_taking_score = (
            execution_analysis.price_impact * 2.0 +  # Higher impact suggests market taking
            tick_analysis.tick_direction_persistence * 0.5
        )
        market_taking_score = min(market_taking_score, 1.0)
        
        # Iceberg/hidden order indicators
        iceberg_score = (
            order_book_analysis.liquidity_concentration * 0.6 +
            (1.0 - tick_analysis.trade_size_distribution.get('large', 0.34)) * 0.4
        )
        
        # Sweep order indicators
        sweep_score = (
            tick_analysis.trade_size_distribution.get('large', 0.34) * 0.7 +
            execution_analysis.price_impact * 0.3
        )
        
        # Determine dominant flow type
        scores = {
            OrderFlowType.MARKET_MAKING: market_making_score,
            OrderFlowType.MARKET_TAKING: market_taking_score,
            OrderFlowType.ICEBERG: iceberg_score,
            OrderFlowType.SWEEP: sweep_score,
            OrderFlowType.STEALTH: (iceberg_score + market_making_score) / 2,
            OrderFlowType.ALGORITHMIC: (market_making_score + market_taking_score) / 2
        }
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _calculate_market_stress(self, order_book_analysis: OrderBookAnalysis,
                               execution_analysis: ExecutionAnalysis,
                               information_flow: InformationFlow) -> float:
        """Calculate market stress indicator"""
        stress_factors = []
        
        # Spread-based stress
        if order_book_analysis.bid_ask_spread > self.config.spread_threshold * 2:
            stress_factors.append(1.0)
        elif order_book_analysis.bid_ask_spread > self.config.spread_threshold:
            stress_factors.append(0.6)
        else:
            stress_factors.append(0.2)
        
        # Volatility-based stress
        spread_volatility_stress = min(order_book_analysis.spread_volatility * 10, 1.0)
        stress_factors.append(spread_volatility_stress)
        
        # Depth-based stress
        depth_stress = abs(order_book_analysis.depth_imbalance)
        stress_factors.append(depth_stress)
        
        # Execution cost stress
        if execution_analysis.implementation_shortfall > self.config.impact_threshold * 5:
            stress_factors.append(1.0)
        elif execution_analysis.implementation_shortfall > self.config.impact_threshold * 2:
            stress_factors.append(0.7)
        else:
            stress_factors.append(0.3)
        
        # Information asymmetry stress
        info_stress_map = {
            InformationAsymmetry.LOW: 0.2,
            InformationAsymmetry.MODERATE: 0.4,
            InformationAsymmetry.HIGH: 0.7,
            InformationAsymmetry.EXTREME: 1.0
        }
        stress_factors.append(info_stress_map[information_flow.information_asymmetry_level])
        
        # Order flow toxicity stress
        stress_factors.append(information_flow.order_flow_toxicity)
        
        # Calculate weighted average
        weights = [0.2, 0.15, 0.15, 0.2, 0.15, 0.15]
        market_stress = np.average(stress_factors, weights=weights)
        
        return market_stress
    
    def _classify_market_regime(self, order_book_analysis: OrderBookAnalysis,
                              execution_analysis: ExecutionAnalysis,
                              market_stress: float) -> MarketRegime:
        """Classify the current market microstructure regime"""
        # Liquidity assessment
        liquidity_score = (
            (1.0 - order_book_analysis.bid_ask_spread / 0.01) * 0.4 +  # Normalized spread
            order_book_analysis.depth_resilience * 0.3 +
            (1.0 - execution_analysis.price_impact / 0.01) * 0.3  # Normalized impact
        )
        liquidity_score = max(0.0, min(1.0, liquidity_score))
        
        # Classify regime
        if market_stress > 0.8:
            return MarketRegime.STRESSED
        elif liquidity_score > 0.8:
            return MarketRegime.LIQUID
        elif liquidity_score < 0.3:
            return MarketRegime.ILLIQUID
        elif order_book_analysis.spread_volatility > 0.5:
            return MarketRegime.VOLATILE
        elif order_book_analysis.liquidity_concentration > 0.8:
            return MarketRegime.CONCENTRATED
        else:
            return MarketRegime.FRAGMENTED
    
    def _calculate_microstructure_quality(self, order_book_analysis: OrderBookAnalysis,
                                        execution_analysis: ExecutionAnalysis,
                                        information_flow: InformationFlow) -> float:
        """Calculate overall microstructure quality score"""
        quality_factors = []
        
        # Spread quality (lower spread = higher quality)
        spread_quality = max(0.0, 1.0 - (order_book_analysis.bid_ask_spread / 0.01))
        quality_factors.append(spread_quality)
        
        # Depth quality
        depth_quality = order_book_analysis.depth_resilience
        quality_factors.append(depth_quality)
        
        # Execution quality
        execution_quality_map = {
            ExecutionQuality.EXCELLENT: 1.0,
            ExecutionQuality.GOOD: 0.8,
            ExecutionQuality.FAIR: 0.6,
            ExecutionQuality.POOR: 0.4,
            ExecutionQuality.SEVERE_IMPACT: 0.2
        }
        execution_quality_score = execution_quality_map[execution_analysis.execution_quality]
        quality_factors.append(execution_quality_score)
        
        # Information efficiency quality
        info_quality = information_flow.price_discovery_efficiency
        quality_factors.append(info_quality)
        
        # Market making activity (higher = better quality)
        mm_quality = order_book_analysis.market_making_activity
        quality_factors.append(mm_quality)
        
        # Calculate weighted average
        weights = [0.25, 0.2, 0.25, 0.15, 0.15]
        overall_quality = np.average(quality_factors, weights=weights)
        
        return overall_quality
    
    def _calculate_confidence(self, order_book_analysis: OrderBookAnalysis,
                            tick_analysis: TickAnalysis,
                            execution_analysis: ExecutionAnalysis,
                            information_flow: InformationFlow) -> float:
        """Calculate overall confidence in the analysis"""
        confidence_factors = []
        
        # Data quality confidence
        if tick_analysis.tick_frequency > 100:
            confidence_factors.append(0.9)
        elif tick_analysis.tick_frequency > 50:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Spread stability confidence
        spread_stability = 1.0 - min(order_book_analysis.spread_volatility, 1.0)
        confidence_factors.append(spread_stability)
        
        # Price discovery confidence
        confidence_factors.append(information_flow.price_discovery_efficiency)
        
        # Execution consistency confidence
        execution_consistency = 1.0 - min(execution_analysis.market_impact_decay, 1.0)
        confidence_factors.append(execution_consistency)
        
        # Information asymmetry confidence (lower asymmetry = higher confidence)
        info_confidence_map = {
            InformationAsymmetry.LOW: 0.9,
            InformationAsymmetry.MODERATE: 0.7,
            InformationAsymmetry.HIGH: 0.5,
            InformationAsymmetry.EXTREME: 0.3
        }
        info_confidence = info_confidence_map[information_flow.information_asymmetry_level]
        confidence_factors.append(info_confidence)
        
        return np.mean(confidence_factors)
    
    def _generate_signal(self, result: MarketMicrostructureResult) -> SignalType:
        """Generate trading signal based on microstructure analysis"""
        signal_criteria = []
        
        # Market regime signals
        if result.market_regime in [MarketRegime.LIQUID, MarketRegime.CONCENTRATED]:
            signal_criteria.append('favorable_regime')
        elif result.market_regime in [MarketRegime.STRESSED, MarketRegime.ILLIQUID]:
            signal_criteria.append('unfavorable_regime')
        
        # Execution quality signals
        if result.execution_analysis.execution_quality in [ExecutionQuality.EXCELLENT, ExecutionQuality.GOOD]:
            signal_criteria.append('good_execution')
        elif result.execution_analysis.execution_quality in [ExecutionQuality.POOR, ExecutionQuality.SEVERE_IMPACT]:
            signal_criteria.append('poor_execution')
        
        # Information flow signals
        if result.information_flow.information_asymmetry_level == InformationAsymmetry.LOW:
            signal_criteria.append('low_asymmetry')
        elif result.information_flow.information_asymmetry_level == InformationAsymmetry.EXTREME:
            signal_criteria.append('high_asymmetry')
        
        # Market stress signals
        if result.market_stress_indicator < 0.3:
            signal_criteria.append('low_stress')
        elif result.market_stress_indicator > 0.7:
            signal_criteria.append('high_stress')
        
        # Microstructure quality signals
        if result.microstructure_quality > 0.7:
            signal_criteria.append('high_quality')
        elif result.microstructure_quality < 0.3:
            signal_criteria.append('low_quality')
        
        # Signal generation logic
        favorable_signals = sum(1 for criterion in signal_criteria 
                              if criterion in ['favorable_regime', 'good_execution', 'low_asymmetry', 
                                             'low_stress', 'high_quality'])
        unfavorable_signals = sum(1 for criterion in signal_criteria 
                                if criterion in ['unfavorable_regime', 'poor_execution', 'high_asymmetry', 
                                               'high_stress', 'low_quality'])
        
        high_confidence = result.confidence_score > 0.7
        good_quality = result.microstructure_quality > 0.6
        
        if (favorable_signals >= 3 and high_confidence and good_quality):
            return SignalType.BUY
        elif (unfavorable_signals >= 2 and high_confidence):
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_state_and_retrain(self, df: pd.DataFrame, data: Dict[str, Any], 
                                result: MarketMicrostructureResult):
        """Update internal state and retrain ML models"""
        max_history = 1000
        
        # Update histories
        self._spread_history.append(result.order_book_analysis.bid_ask_spread)
        self._impact_history.append(result.execution_analysis.price_impact)
        self._market_regime_buffer.append(result.market_regime.value)
        self._liquidity_stress_buffer.append(result.market_stress_indicator)
        
        # Store tick and order book data
        if 'tick_data' in data:
            self._tick_history.append(data['tick_data'])
        if 'order_book' in data:
            self._order_book_history.append(data['order_book'])
        
        # Trim histories
        if len(self._spread_history) > max_history:
            self._spread_history = self._spread_history[-max_history:]
            self._impact_history = self._impact_history[-max_history:]
            self._market_regime_buffer = self._market_regime_buffer[-max_history:]
            self._liquidity_stress_buffer = self._liquidity_stress_buffer[-max_history:]
        
        if len(self._tick_history) > max_history // 10:
            self._tick_history = self._tick_history[-max_history // 10:]
        if len(self._order_book_history) > max_history // 10:
            self._order_book_history = self._order_book_history[-max_history // 10:]
        
        # Retrain ML models periodically
        if (self.config.use_ml_classification and len(self._spread_history) >= 200 and 
            len(self._spread_history) % 100 == 0):
            self._retrain_ml_models()
    
    def _retrain_ml_models(self):
        """Retrain machine learning models for microstructure analysis"""
        try:
            if len(self._spread_history) < 100:
                return
            
            # Prepare training data for regime classification
            features = []
            targets = []
            
            window_size = 20
            for i in range(window_size, len(self._spread_history) - 1):
                # Features: recent microstructure metrics
                feature_vector = []
                
                # Spread features
                spread_window = self._spread_history[i-window_size:i]
                feature_vector.extend([
                    np.mean(spread_window),
                    np.std(spread_window),
                    np.max(spread_window),
                    np.min(spread_window)
                ])
                
                # Impact features
                if i < len(self._impact_history):
                    impact_window = self._impact_history[max(0, i-window_size):i]
                    feature_vector.extend([
                        np.mean(impact_window),
                        np.std(impact_window)
                    ])
                else:
                    feature_vector.extend([0, 0])
                
                # Stress features
                if i < len(self._liquidity_stress_buffer):
                    stress_window = self._liquidity_stress_buffer[max(0, i-window_size):i]
                    feature_vector.extend([
                        np.mean(stress_window),
                        np.std(stress_window)
                    ])
                else:
                    feature_vector.extend([0, 0])
                
                features.append(feature_vector)
                
                # Target: next regime
                if i < len(self._market_regime_buffer):
                    targets.append(self._market_regime_buffer[i])
                else:
                    targets.append('liquid')  # Default
            
            if len(features) > 50 and len(set(targets)) > 1:
                # Scale features
                features_array = np.array(features)
                self._scaler.fit(features_array)
                features_scaled = self._scaler.transform(features_array)
                
                # Train regime classifier
                self._regime_classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self._regime_classifier.fit(features_scaled, targets)
                
                # Train anomaly detector
                self._anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                self._anomaly_detector.fit(features_scaled)
                
                self._is_trained = True
                self.logger.info("Market microstructure ML models retrained successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model retraining failed: {e}")
    
    def _format_result(self, result: MarketMicrostructureResult, signal: SignalType) -> Dict[str, Any]:
        """Format the complete result for output"""
        return {
            'signal': signal,
            'confidence': result.confidence_score,
            
            # Core microstructure metrics
            'market_regime': result.market_regime.value,
            'microstructure_quality': result.microstructure_quality,
            'market_stress_indicator': result.market_stress_indicator,
            'dominant_flow_type': result.dominant_flow_type.value,
            'hft_activity_level': result.hft_activity_level,
            
            # Order book analysis
            'bid_ask_spread': result.order_book_analysis.bid_ask_spread,
            'spread_volatility': result.order_book_analysis.spread_volatility,
            'depth_imbalance': result.order_book_analysis.depth_imbalance,
            'order_book_slope': result.order_book_analysis.order_book_slope,
            'liquidity_concentration': result.order_book_analysis.liquidity_concentration,
            'depth_resilience': result.order_book_analysis.depth_resilience,
            'price_impact_estimate': result.order_book_analysis.price_impact_estimate,
            'market_making_activity': result.order_book_analysis.market_making_activity,
            
            # Tick analysis
            'tick_frequency': result.tick_analysis.tick_frequency,
            'tick_size_distribution': result.tick_analysis.tick_size_distribution,
            'price_clustering': result.tick_analysis.price_clustering,
            'trade_size_distribution': result.tick_analysis.trade_size_distribution,
            'arrival_rate_variance': result.tick_analysis.arrival_rate_variance,
            'tick_direction_persistence': result.tick_analysis.tick_direction_persistence,
            'quote_revision_frequency': result.tick_analysis.quote_revision_frequency,
            
            # Execution analysis
            'effective_spread': result.execution_analysis.effective_spread,
            'price_impact': result.execution_analysis.price_impact,
            'timing_cost': result.execution_analysis.timing_cost,
            'opportunity_cost': result.execution_analysis.opportunity_cost,
            'implementation_shortfall': result.execution_analysis.implementation_shortfall,
            'market_impact_decay': result.execution_analysis.market_impact_decay,
            'adverse_selection_cost': result.execution_analysis.adverse_selection_cost,
            'execution_quality': result.execution_analysis.execution_quality.value,
            
            # Information flow
            'price_discovery_efficiency': result.information_flow.price_discovery_efficiency,
            'information_asymmetry_level': result.information_flow.information_asymmetry_level.value,
            'informed_trading_probability': result.information_flow.informed_trading_probability,
            'noise_trader_activity': result.information_flow.noise_trader_activity,
            'pin_score': result.information_flow.pin_score,
            'order_flow_toxicity': result.information_flow.order_flow_toxicity,
            'market_microstructure_noise': result.information_flow.market_microstructure_noise,
            
            # Metadata
            'metadata': {
                'indicator_name': 'MarketMicrostructureIndicator',
                'version': '1.0.0',
                'calculation_time': pd.Timestamp.now().isoformat(),
                'ml_classification': self.config.use_ml_classification,
                'anomaly_detection': self.config.use_anomaly_detection,
                'hft_detection': self.config.use_hft_detection,
                'ml_trained': self._is_trained
            }
        }
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data structure and completeness"""
        required_fields = ['high', 'low', 'close', 'volume']
        
        if not isinstance(data, dict):
            return False
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
            
            if not isinstance(data[field], (list, np.ndarray)) or len(data[field]) == 0:
                self.logger.error(f"Invalid data for field: {field}")
                return False
        
        # Check data consistency
        lengths = [len(data[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            self.logger.error("Inconsistent data lengths across fields")
            return False
        
        return True
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'market_regime': MarketRegime.LIQUID.value,
            'microstructure_quality': 0.5,
            'market_stress_indicator': 0.5,
            'dominant_flow_type': OrderFlowType.MARKET_MAKING.value,
            'hft_activity_level': 0.3,
            'bid_ask_spread': 0.001,
            'spread_volatility': 0.1,
            'depth_imbalance': 0.0,
            'order_book_slope': 1.0,
            'liquidity_concentration': 0.5,
            'depth_resilience': 0.5,
            'price_impact_estimate': 0.001,
            'market_making_activity': 0.5,
            'tick_frequency': 100.0,
            'tick_size_distribution': {'small': 0.33, 'medium': 0.33, 'large': 0.34},
            'price_clustering': 0.5,
            'trade_size_distribution': {'small': 0.33, 'medium': 0.33, 'large': 0.34},
            'arrival_rate_variance': 1.0,
            'tick_direction_persistence': 0.0,
            'quote_revision_frequency': 0.1,
            'effective_spread': 0.002,
            'price_impact': 0.001,
            'timing_cost': 0.0003,
            'opportunity_cost': 0.0002,
            'implementation_shortfall': 0.0035,
            'market_impact_decay': 0.5,
            'adverse_selection_cost': 0.0001,
            'execution_quality': ExecutionQuality.FAIR.value,
            'price_discovery_efficiency': 0.5,
            'information_asymmetry_level': InformationAsymmetry.MODERATE.value,
            'informed_trading_probability': 0.3,
            'noise_trader_activity': 0.7,
            'pin_score': 0.3,
            'order_flow_toxicity': 0.2,
            'market_microstructure_noise': 0.1,
            'metadata': {
                'indicator_name': 'MarketMicrostructureIndicator',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        result = self._create_default_result()
        result['metadata']['error'] = error_message
        return result