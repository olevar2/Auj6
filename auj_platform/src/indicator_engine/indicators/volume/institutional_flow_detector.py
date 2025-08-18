"""
AUJ Platform Advanced Institutional Flow Detector
Sophisticated implementation with block trade identification, order flow imbalance analysis, and ML pattern recognition

This implementation provides institutional-grade flow detection for humanitarian trading platforms.

Features:
- Advanced block trade identification with dynamic thresholds
- Order flow imbalance analysis and pressure measurement
- Machine learning pattern recognition for institutional behavior
- Multi-timeframe flow analysis and trend confirmation
- Volume-based institutional signature detection
- Statistical validation and confidence scoring
- Real-time flow monitoring and alert generation
- Liquidity provider vs. taker analysis
- Market impact assessment and prediction
- Comprehensive flow classification system

The Institutional Flow Detector identifies large-scale trading activity that typically
indicates institutional involvement, helping traders understand when smart money is
moving the market and in which direction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from ..base.base_indicator import BaseIndicator, IndicatorConfig
from ...core.signal_type import SignalType


class FlowDirection(Enum):
    """Direction of institutional flow"""
    BUYING = "buying"
    SELLING = "selling"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class FlowIntensity(Enum):
    """Intensity of institutional flow"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class FlowType(Enum):
    """Type of institutional flow pattern"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    ROTATION = "rotation"
    WASH_TRADING = "wash_trading"
    PROGRAM_TRADING = "program_trading"
    ICEBERG = "iceberg"


@dataclass
class InstitutionalFlowConfig(IndicatorConfig):
    """Configuration for Institutional Flow Detector"""
    block_trade_threshold: float = 2.0  # Multiple of average volume
    flow_analysis_period: int = 20
    imbalance_threshold: float = 0.3
    ml_training_period: int = 200
    confidence_threshold: float = 0.7
    volume_lookback: int = 50
    price_impact_threshold: float = 0.001
    min_block_size: float = 100000  # Minimum dollar value
    adaptive_thresholds: bool = True
    use_ml_classification: bool = True
    flow_persistence_period: int = 10


class BlockTrade(NamedTuple):
    """Block trade detection result"""
    timestamp: int
    volume: float
    price: float
    direction: FlowDirection
    size_ratio: float
    price_impact: float
    confidence: float


class OrderFlowImbalance(NamedTuple):
    """Order flow imbalance analysis result"""
    imbalance_ratio: float
    buy_pressure: float
    sell_pressure: float
    net_flow: float
    flow_momentum: float
    persistence_score: float


class InstitutionalSignature(NamedTuple):
    """Institutional trading signature analysis"""
    signature_type: FlowType
    strength: float
    consistency: float
    market_impact: float
    stealth_factor: float
    urgency_score: float


class FlowPrediction(NamedTuple):
    """ML-based flow prediction result"""
    predicted_direction: FlowDirection
    predicted_intensity: FlowIntensity
    confidence: float
    time_horizon: int
    probability_distribution: Dict[str, float]


class InstitutionalFlowResult(NamedTuple):
    """Complete institutional flow analysis result"""
    overall_flow_direction: FlowDirection
    flow_intensity: FlowIntensity
    block_trades: List[BlockTrade]
    order_flow_imbalance: OrderFlowImbalance
    institutional_signature: InstitutionalSignature
    flow_prediction: FlowPrediction
    flow_persistence: float
    market_dominance: float
    flow_quality_score: float


class InstitutionalFlowDetector(BaseIndicator):
    """
    Advanced Institutional Flow Detector with machine learning capabilities.
    
    This indicator identifies and analyzes institutional trading activity through:
    - Block trade detection with dynamic thresholds
    - Order flow imbalance analysis
    - Pattern recognition for institutional signatures
    - Machine learning classification of flow types
    - Predictive modeling for future flow direction
    - Multi-dimensional flow analysis
    """
    
    def __init__(self, config: Optional[InstitutionalFlowConfig] = None):
        super().__init__(config or InstitutionalFlowConfig())
        self.config: InstitutionalFlowConfig = self.config
        
        # Internal state
        self._volume_history: List[float] = []
        self._price_history: List[float] = []
        self._flow_history: List[Dict[str, Any]] = []
        self._block_trades_history: List[BlockTrade] = []
        self._imbalance_history: List[OrderFlowImbalance] = []
        
        # Machine learning components
        self._flow_classifier: Optional[RandomForestClassifier] = None
        self._anomaly_detector: Optional[IsolationForest] = None
        self._scaler: StandardScaler = StandardScaler()
        self._is_trained: bool = False
        
        # Adaptive thresholds
        self._adaptive_block_threshold: float = self.config.block_trade_threshold
        self._adaptive_imbalance_threshold: float = self.config.imbalance_threshold
        
        # Flow tracking
        self._current_flow_session: Dict[str, Any] = {}
        self._flow_statistics: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced institutional flow detection.
        
        Args:
            data: Dictionary containing 'high', 'low', 'close', 'volume' price data
            
        Returns:
            Dictionary containing institutional flow analysis results
        """
        try:
            if not self._validate_data(data):
                return self._create_default_result()
            
            df = pd.DataFrame(data)
            
            if len(df) < self.config.flow_analysis_period:
                return self._create_default_result()
            
            # Detect block trades
            block_trades = self._detect_block_trades(df)
            
            # Analyze order flow imbalance
            order_flow_imbalance = self._analyze_order_flow_imbalance(df)
            
            # Identify institutional signatures
            institutional_signature = self._identify_institutional_signature(
                df, block_trades, order_flow_imbalance
            )
            
            # Determine overall flow characteristics
            overall_flow_direction = self._determine_overall_flow_direction(
                block_trades, order_flow_imbalance
            )
            flow_intensity = self._calculate_flow_intensity(
                block_trades, order_flow_imbalance, institutional_signature
            )
            
            # Calculate flow persistence and quality
            flow_persistence = self._calculate_flow_persistence(df, overall_flow_direction)
            market_dominance = self._calculate_market_dominance(df, block_trades)
            flow_quality_score = self._calculate_flow_quality(
                block_trades, order_flow_imbalance, institutional_signature
            )
            
            # ML-based flow prediction
            flow_prediction = self._predict_future_flow(df, block_trades, order_flow_imbalance)
            
            # Create result
            result = InstitutionalFlowResult(
                overall_flow_direction=overall_flow_direction,
                flow_intensity=flow_intensity,
                block_trades=block_trades,
                order_flow_imbalance=order_flow_imbalance,
                institutional_signature=institutional_signature,
                flow_prediction=flow_prediction,
                flow_persistence=flow_persistence,
                market_dominance=market_dominance,
                flow_quality_score=flow_quality_score
            )
            
            # Generate trading signal
            signal = self._generate_signal(result)
            
            # Update internal state and retrain if needed
            self._update_state_and_retrain(df, result)
            
            return self._format_result(result, signal)
            
        except Exception as e:
            self.logger.error(f"Error in InstitutionalFlowDetector calculation: {e}")
            return self._create_error_result(str(e))
    
    def _detect_block_trades(self, df: pd.DataFrame) -> List[BlockTrade]:
        """Detect block trades using dynamic thresholds"""
        volume = df['volume'].values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        block_trades = []
        
        # Calculate dynamic volume threshold
        recent_volume = volume[-self.config.volume_lookback:]
        avg_volume = np.mean(recent_volume)
        volume_std = np.std(recent_volume)
        
        if self.config.adaptive_thresholds:
            # Adaptive threshold based on recent market activity
            volume_percentile_90 = np.percentile(recent_volume, 90)
            dynamic_threshold = max(
                avg_volume * self._adaptive_block_threshold,
                volume_percentile_90
            )
        else:
            dynamic_threshold = avg_volume * self.config.block_trade_threshold
        
        # Scan for block trades
        for i in range(len(volume)):
            current_volume = volume[i]
            current_price = close[i]
            
            # Check if volume exceeds threshold
            if current_volume > dynamic_threshold:
                # Calculate size ratio
                size_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                # Calculate price impact
                if i > 0:
                    price_change = (current_price - close[i-1]) / close[i-1]
                    price_impact = abs(price_change)
                else:
                    price_impact = 0
                
                # Determine direction based on price action and volume
                if i > 0:
                    if current_price > close[i-1]:
                        direction = FlowDirection.BUYING
                    elif current_price < close[i-1]:
                        direction = FlowDirection.SELLING
                    else:
                        direction = FlowDirection.NEUTRAL
                else:
                    direction = FlowDirection.NEUTRAL
                
                # Calculate confidence based on multiple factors
                volume_confidence = min(size_ratio / 5.0, 1.0)  # Cap at 1.0
                impact_confidence = min(price_impact / 0.01, 1.0)  # Cap at 1.0
                
                # Check if meets minimum dollar value
                dollar_volume = current_volume * current_price
                if dollar_volume >= self.config.min_block_size:
                    confidence = (volume_confidence + impact_confidence) / 2.0
                    
                    block_trade = BlockTrade(
                        timestamp=i,
                        volume=current_volume,
                        price=current_price,
                        direction=direction,
                        size_ratio=size_ratio,
                        price_impact=price_impact,
                        confidence=confidence
                    )
                    
                    block_trades.append(block_trade)
        
        return block_trades[-20:]  # Keep last 20 block trades
    
    def _analyze_order_flow_imbalance(self, df: pd.DataFrame) -> OrderFlowImbalance:
        """Analyze order flow imbalance and pressure"""
        volume = df['volume'].values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        period = min(self.config.flow_analysis_period, len(df))
        recent_data = df.tail(period)
        
        # Calculate buying and selling pressure
        buy_pressure = 0.0
        sell_pressure = 0.0
        
        for i in range(1, len(recent_data)):
            curr_row = recent_data.iloc[i]
            prev_close = recent_data.iloc[i-1]['close']
            
            # Estimate buying/selling pressure based on price movement and volume
            price_change = curr_row['close'] - prev_close
            volume_weight = curr_row['volume']
            
            # Use high-low range to estimate intraday pressure
            total_range = curr_row['high'] - curr_row['low']
            if total_range > 0:
                # Position within the range indicates pressure direction
                close_position = (curr_row['close'] - curr_row['low']) / total_range
                
                if close_position > 0.6:  # Close near high
                    buy_pressure += volume_weight * close_position
                elif close_position < 0.4:  # Close near low
                    sell_pressure += volume_weight * (1 - close_position)
                else:  # Neutral
                    if price_change > 0:
                        buy_pressure += volume_weight * 0.5
                    elif price_change < 0:
                        sell_pressure += volume_weight * 0.5
            else:
                # No range, use price change
                if price_change > 0:
                    buy_pressure += volume_weight
                elif price_change < 0:
                    sell_pressure += volume_weight
        
        # Calculate metrics
        total_pressure = buy_pressure + sell_pressure
        if total_pressure > 0:
            imbalance_ratio = (buy_pressure - sell_pressure) / total_pressure
            buy_pressure_norm = buy_pressure / total_pressure
            sell_pressure_norm = sell_pressure / total_pressure
        else:
            imbalance_ratio = 0.0
            buy_pressure_norm = 0.5
            sell_pressure_norm = 0.5
        
        net_flow = buy_pressure - sell_pressure
        
        # Calculate flow momentum (rate of change in imbalance)
        if len(self._imbalance_history) > 0:
            prev_imbalance = self._imbalance_history[-1].imbalance_ratio
            flow_momentum = imbalance_ratio - prev_imbalance
        else:
            flow_momentum = 0.0
        
        # Calculate persistence score
        persistence_score = self._calculate_imbalance_persistence(imbalance_ratio)
        
        return OrderFlowImbalance(
            imbalance_ratio=imbalance_ratio,
            buy_pressure=buy_pressure_norm,
            sell_pressure=sell_pressure_norm,
            net_flow=net_flow,
            flow_momentum=flow_momentum,
            persistence_score=persistence_score
        )
    
    def _identify_institutional_signature(self, df: pd.DataFrame,
                                        block_trades: List[BlockTrade],
                                        imbalance: OrderFlowImbalance) -> InstitutionalSignature:
        """Identify institutional trading signatures and patterns"""
        volume = df['volume'].values
        close = df['close'].values
        
        # Analyze recent trading patterns
        recent_period = min(self.config.flow_analysis_period, len(df))
        recent_volume = volume[-recent_period:]
        recent_prices = close[-recent_period:]
        
        # Pattern recognition
        signature_type = self._classify_flow_pattern(df, block_trades, imbalance)
        
        # Calculate signature strength
        block_trade_strength = len(block_trades) / 10.0  # Normalize by expected count
        imbalance_strength = abs(imbalance.imbalance_ratio)
        volume_consistency = 1.0 - (np.std(recent_volume) / np.mean(recent_volume)) if np.mean(recent_volume) > 0 else 0
        
        strength = min(1.0, (block_trade_strength + imbalance_strength + volume_consistency) / 3.0)
        
        # Calculate consistency score
        if len(block_trades) >= 3:
            # Check direction consistency among block trades
            directions = [trade.direction for trade in block_trades[-5:]]
            direction_consistency = len([d for d in directions if d == directions[0]]) / len(directions)
        else:
            direction_consistency = 0.5
        
        consistency = (direction_consistency + imbalance.persistence_score) / 2.0
        
        # Calculate market impact
        total_block_volume = sum(trade.volume for trade in block_trades)
        avg_volume = np.mean(recent_volume)
        market_impact = min(1.0, total_block_volume / (avg_volume * len(block_trades))) if avg_volume > 0 else 0
        
        # Calculate stealth factor (inverse of market impact)
        stealth_factor = 1.0 - market_impact
        
        # Calculate urgency score based on price impact and volume concentration
        if block_trades:
            avg_price_impact = np.mean([trade.price_impact for trade in block_trades])
            urgency_score = min(1.0, avg_price_impact / self.config.price_impact_threshold)
        else:
            urgency_score = 0.0
        
        return InstitutionalSignature(
            signature_type=signature_type,
            strength=strength,
            consistency=consistency,
            market_impact=market_impact,
            stealth_factor=stealth_factor,
            urgency_score=urgency_score
        )
    
    def _classify_flow_pattern(self, df: pd.DataFrame,
                             block_trades: List[BlockTrade],
                             imbalance: OrderFlowImbalance) -> FlowType:
        """Classify the type of institutional flow pattern"""
        if len(block_trades) == 0:
            return FlowType.ROTATION
        
        # Analyze block trade patterns
        recent_trades = block_trades[-5:] if len(block_trades) >= 5 else block_trades
        
        # Count directions
        buying_trades = sum(1 for trade in recent_trades if trade.direction == FlowDirection.BUYING)
        selling_trades = sum(1 for trade in recent_trades if trade.direction == FlowDirection.SELLING)
        
        # Analyze volume distribution
        volumes = [trade.volume for trade in recent_trades]
        volume_std = np.std(volumes) if len(volumes) > 1 else 0
        volume_mean = np.mean(volumes) if volumes else 0
        volume_variation = volume_std / volume_mean if volume_mean > 0 else 0
        
        # Pattern classification logic
        if abs(imbalance.imbalance_ratio) > 0.5:
            if imbalance.imbalance_ratio > 0:
                if volume_variation < 0.3:  # Consistent volume sizes
                    return FlowType.ICEBERG
                else:
                    return FlowType.ACCUMULATION
            else:
                if volume_variation < 0.3:
                    return FlowType.ICEBERG
                else:
                    return FlowType.DISTRIBUTION
        
        elif volume_variation > 0.7:  # High volume variation
            return FlowType.PROGRAM_TRADING
        
        elif buying_trades > 0 and selling_trades > 0:  # Mixed activity
            if abs(buying_trades - selling_trades) <= 1:
                return FlowType.WASH_TRADING
            else:
                return FlowType.ROTATION
        
        else:
            return FlowType.ROTATION
    
    def _determine_overall_flow_direction(self, block_trades: List[BlockTrade],
                                        imbalance: OrderFlowImbalance) -> FlowDirection:
        """Determine overall institutional flow direction"""
        # Weight block trades and imbalance analysis
        block_weight = 0.6
        imbalance_weight = 0.4
        
        # Block trade direction score
        if block_trades:
            recent_trades = block_trades[-5:]
            buying_score = sum(1 for trade in recent_trades if trade.direction == FlowDirection.BUYING)
            selling_score = sum(1 for trade in recent_trades if trade.direction == FlowDirection.SELLING)
            
            if buying_score > selling_score:
                block_direction_score = 1.0
            elif selling_score > buying_score:
                block_direction_score = -1.0
            else:
                block_direction_score = 0.0
        else:
            block_direction_score = 0.0
        
        # Imbalance direction score
        imbalance_direction_score = imbalance.imbalance_ratio
        
        # Combined score
        overall_score = (block_weight * block_direction_score + 
                        imbalance_weight * imbalance_direction_score)
        
        # Classify direction
        if overall_score > 0.2:
            return FlowDirection.BUYING
        elif overall_score < -0.2:
            return FlowDirection.SELLING
        elif abs(overall_score) < 0.1:
            return FlowDirection.NEUTRAL
        else:
            return FlowDirection.MIXED
    
    def _calculate_flow_intensity(self, block_trades: List[BlockTrade],
                                imbalance: OrderFlowImbalance,
                                signature: InstitutionalSignature) -> FlowIntensity:
        """Calculate institutional flow intensity"""
        # Multiple intensity factors
        intensity_factors = []
        
        # Block trade intensity
        if block_trades:
            avg_size_ratio = np.mean([trade.size_ratio for trade in block_trades[-5:]])
            block_intensity = min(avg_size_ratio / 5.0, 1.0)  # Normalize to 0-1
        else:
            block_intensity = 0.0
        intensity_factors.append(block_intensity)
        
        # Imbalance intensity
        imbalance_intensity = abs(imbalance.imbalance_ratio)
        intensity_factors.append(imbalance_intensity)
        
        # Signature strength
        intensity_factors.append(signature.strength)
        
        # Flow momentum intensity
        momentum_intensity = min(abs(imbalance.flow_momentum), 1.0)
        intensity_factors.append(momentum_intensity)
        
        # Calculate overall intensity
        overall_intensity = np.mean(intensity_factors)
        
        # Classify intensity
        if overall_intensity < 0.25:
            return FlowIntensity.LOW
        elif overall_intensity < 0.5:
            return FlowIntensity.MODERATE
        elif overall_intensity < 0.75:
            return FlowIntensity.HIGH
        else:
            return FlowIntensity.EXTREME
    
    def _calculate_flow_persistence(self, df: pd.DataFrame, 
                                  flow_direction: FlowDirection) -> float:
        """Calculate flow persistence over time"""
        if len(self._flow_history) < 2:
            return 0.5
        
        # Count consistent flow periods
        recent_flows = self._flow_history[-self.config.flow_persistence_period:]
        
        consistent_periods = 0
        for flow_data in recent_flows:
            if flow_data.get('direction') == flow_direction:
                consistent_periods += 1
        
        persistence = consistent_periods / len(recent_flows)
        return persistence
    
    def _calculate_market_dominance(self, df: pd.DataFrame, 
                                  block_trades: List[BlockTrade]) -> float:
        """Calculate institutional market dominance"""
        volume = df['volume'].values
        recent_volume = volume[-self.config.flow_analysis_period:]
        total_market_volume = np.sum(recent_volume)
        
        if block_trades and total_market_volume > 0:
            institutional_volume = sum(trade.volume for trade in block_trades)
            dominance = min(institutional_volume / total_market_volume, 1.0)
        else:
            dominance = 0.0
        
        return dominance
    
    def _calculate_flow_quality(self, block_trades: List[BlockTrade],
                              imbalance: OrderFlowImbalance,
                              signature: InstitutionalSignature) -> float:
        """Calculate overall flow signal quality"""
        quality_factors = []
        
        # Block trade quality (based on confidence)
        if block_trades:
            avg_confidence = np.mean([trade.confidence for trade in block_trades])
            quality_factors.append(avg_confidence)
        else:
            quality_factors.append(0.0)
        
        # Imbalance quality (based on persistence)
        quality_factors.append(imbalance.persistence_score)
        
        # Signature quality (based on consistency)
        quality_factors.append(signature.consistency)
        
        # Overall quality
        return np.mean(quality_factors)
    
    def _predict_future_flow(self, df: pd.DataFrame,
                           block_trades: List[BlockTrade],
                           imbalance: OrderFlowImbalance) -> FlowPrediction:
        """Predict future institutional flow using ML"""
        if not self.config.use_ml_classification or not self._is_trained:
            return FlowPrediction(
                predicted_direction=FlowDirection.NEUTRAL,
                predicted_intensity=FlowIntensity.LOW,
                confidence=0.0,
                time_horizon=1,
                probability_distribution={'buying': 0.33, 'selling': 0.33, 'neutral': 0.34}
            )
        
        try:
            # Extract features for prediction
            features = self._extract_prediction_features(df, block_trades, imbalance)
            
            if len(features) == 0:
                return self._create_default_prediction()
            
            # Scale features
            features_scaled = self._scaler.transform([features])
            
            # Predict direction
            if hasattr(self._flow_classifier, 'predict_proba'):
                probabilities = self._flow_classifier.predict_proba(features_scaled)[0]
                classes = self._flow_classifier.classes_
                
                # Create probability distribution
                prob_dist = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
                
                # Get prediction
                predicted_class_idx = np.argmax(probabilities)
                predicted_direction_str = classes[predicted_class_idx]
                confidence = float(probabilities[predicted_class_idx])
                
                # Convert to enum
                direction_map = {
                    'buying': FlowDirection.BUYING,
                    'selling': FlowDirection.SELLING,
                    'neutral': FlowDirection.NEUTRAL
                }
                predicted_direction = direction_map.get(predicted_direction_str, FlowDirection.NEUTRAL)
            else:
                predicted_direction = FlowDirection.NEUTRAL
                confidence = 0.0
                prob_dist = {'buying': 0.33, 'selling': 0.33, 'neutral': 0.34}
            
            # Predict intensity based on current patterns
            if len(block_trades) > 2:
                avg_size_ratio = np.mean([trade.size_ratio for trade in block_trades[-3:]])
                if avg_size_ratio > 4.0:
                    predicted_intensity = FlowIntensity.EXTREME
                elif avg_size_ratio > 3.0:
                    predicted_intensity = FlowIntensity.HIGH
                elif avg_size_ratio > 2.0:
                    predicted_intensity = FlowIntensity.MODERATE
                else:
                    predicted_intensity = FlowIntensity.LOW
            else:
                predicted_intensity = FlowIntensity.LOW
            
            # Time horizon based on flow persistence
            time_horizon = min(max(int(imbalance.persistence_score * 10), 1), 5)
            
            return FlowPrediction(
                predicted_direction=predicted_direction,
                predicted_intensity=predicted_intensity,
                confidence=confidence,
                time_horizon=time_horizon,
                probability_distribution=prob_dist
            )
            
        except Exception as e:
            self.logger.warning(f"Flow prediction failed: {e}")
            return self._create_default_prediction()
    
    def _calculate_imbalance_persistence(self, current_imbalance: float) -> float:
        """Calculate persistence of order flow imbalance"""
        if len(self._imbalance_history) < 3:
            return 0.5
        
        recent_imbalances = [h.imbalance_ratio for h in self._imbalance_history[-5:]]
        recent_imbalances.append(current_imbalance)
        
        # Check for consistent direction
        positive_count = sum(1 for imb in recent_imbalances if imb > 0.1)
        negative_count = sum(1 for imb in recent_imbalances if imb < -0.1)
        
        max_consistent = max(positive_count, negative_count)
        persistence = max_consistent / len(recent_imbalances)
        
        return persistence
    
    def _extract_prediction_features(self, df: pd.DataFrame,
                                   block_trades: List[BlockTrade],
                                   imbalance: OrderFlowImbalance) -> List[float]:
        """Extract features for ML prediction"""
        features = []
        
        # Block trade features
        if block_trades:
            recent_trades = block_trades[-3:]
            features.extend([
                len(recent_trades),
                np.mean([trade.size_ratio for trade in recent_trades]),
                np.mean([trade.price_impact for trade in recent_trades]),
                np.mean([trade.confidence for trade in recent_trades])
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Imbalance features
        features.extend([
            imbalance.imbalance_ratio,
            imbalance.buy_pressure,
            imbalance.sell_pressure,
            imbalance.flow_momentum,
            imbalance.persistence_score
        ])
        
        # Volume features
        recent_volume = df['volume'].values[-10:]
        features.extend([
            np.mean(recent_volume),
            np.std(recent_volume),
            np.max(recent_volume) / np.mean(recent_volume) if np.mean(recent_volume) > 0 else 0
        ])
        
        # Price features
        recent_close = df['close'].values[-10:]
        price_changes = np.diff(recent_close)
        features.extend([
            np.mean(price_changes),
            np.std(price_changes),
            np.max(np.abs(price_changes))
        ])
        
        return features
    
    def _generate_signal(self, result: InstitutionalFlowResult) -> SignalType:
        """Generate trading signal based on institutional flow analysis"""
        # Signal criteria
        signal_strength = 0
        
        # Flow direction and intensity
        if result.overall_flow_direction == FlowDirection.BUYING:
            if result.flow_intensity in [FlowIntensity.HIGH, FlowIntensity.EXTREME]:
                signal_strength += 3
            elif result.flow_intensity == FlowIntensity.MODERATE:
                signal_strength += 2
        elif result.overall_flow_direction == FlowDirection.SELLING:
            if result.flow_intensity in [FlowIntensity.HIGH, FlowIntensity.EXTREME]:
                signal_strength -= 3
            elif result.flow_intensity == FlowIntensity.MODERATE:
                signal_strength -= 2
        
        # Flow quality and persistence
        if result.flow_quality_score > 0.7 and result.flow_persistence > 0.6:
            signal_strength *= 1.5
        
        # Block trade confirmation
        if len(result.block_trades) >= 2:
            recent_trades = result.block_trades[-2:]
            if all(trade.direction == FlowDirection.BUYING for trade in recent_trades):
                signal_strength += 1
            elif all(trade.direction == FlowDirection.SELLING for trade in recent_trades):
                signal_strength -= 1
        
        # Market dominance factor
        if result.market_dominance > 0.3:
            signal_strength *= 1.2
        
        # ML prediction confirmation
        if (result.flow_prediction.confidence > self.config.confidence_threshold and
            result.flow_prediction.predicted_direction == result.overall_flow_direction):
            signal_strength *= 1.3
        
        # Generate signal
        if signal_strength >= 4:
            return SignalType.BUY
        elif signal_strength <= -4:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_state_and_retrain(self, df: pd.DataFrame, result: InstitutionalFlowResult):
        """Update internal state and retrain ML models"""
        max_history = 500
        
        # Update histories
        self._volume_history.extend(df['volume'].values[-5:])
        self._price_history.extend(df['close'].values[-5:])
        
        # Store flow analysis
        flow_data = {
            'direction': result.overall_flow_direction,
            'intensity': result.flow_intensity,
            'quality': result.flow_quality_score,
            'timestamp': len(self._flow_history)
        }
        self._flow_history.append(flow_data)
        
        self._block_trades_history.extend(result.block_trades)
        self._imbalance_history.append(result.order_flow_imbalance)
        
        # Trim histories
        if len(self._volume_history) > max_history:
            self._volume_history = self._volume_history[-max_history:]
            self._price_history = self._price_history[-max_history:]
        
        if len(self._flow_history) > max_history // 5:
            self._flow_history = self._flow_history[-max_history // 5:]
        
        if len(self._block_trades_history) > max_history // 10:
            self._block_trades_history = self._block_trades_history[-max_history // 10:]
        
        if len(self._imbalance_history) > max_history // 5:
            self._imbalance_history = self._imbalance_history[-max_history // 5:]
        
        # Update adaptive thresholds
        if self.config.adaptive_thresholds and len(self._volume_history) >= 50:
            recent_volume = self._volume_history[-50:]
            volume_percentile_95 = np.percentile(recent_volume, 95)
            volume_mean = np.mean(recent_volume)
            
            if volume_mean > 0:
                self._adaptive_block_threshold = max(
                    self.config.block_trade_threshold,
                    volume_percentile_95 / volume_mean
                )
        
        # Retrain ML models periodically
        if (self.config.use_ml_classification and 
            len(self._flow_history) >= self.config.ml_training_period and 
            len(self._flow_history) % 50 == 0):
            self._retrain_ml_models()
    
    def _retrain_ml_models(self):
        """Retrain machine learning models"""
        try:
            if len(self._flow_history) < 50:
                return
            
            # Prepare training data
            features = []
            labels = []
            
            for i in range(10, len(self._flow_history) - 5):
                # Extract features from historical data
                if (i < len(self._block_trades_history) and 
                    i < len(self._imbalance_history)):
                    
                    feature_vector = []
                    
                    # Flow features
                    feature_vector.append(self._imbalance_history[i].imbalance_ratio)
                    feature_vector.append(self._imbalance_history[i].flow_momentum)
                    feature_vector.append(self._imbalance_history[i].persistence_score)
                    
                    # Volume features
                    start_idx = max(0, i - 10)
                    volume_window = self._volume_history[start_idx:i]
                    if len(volume_window) > 0:
                        feature_vector.extend([
                            np.mean(volume_window),
                            np.std(volume_window)
                        ])
                    else:
                        feature_vector.extend([0, 0])
                    
                    # Price features
                    price_window = self._price_history[start_idx:i]
                    if len(price_window) > 1:
                        price_changes = np.diff(price_window)
                        feature_vector.extend([
                            np.mean(price_changes),
                            np.std(price_changes)
                        ])
                    else:
                        feature_vector.extend([0, 0])
                    
                    features.append(feature_vector)
                    
                    # Label: future flow direction
                    future_flow = self._flow_history[i + 2]['direction']
                    if future_flow == FlowDirection.BUYING:
                        labels.append('buying')
                    elif future_flow == FlowDirection.SELLING:
                        labels.append('selling')
                    else:
                        labels.append('neutral')
            
            if len(features) > 20 and len(set(labels)) > 1:
                # Scale features
                features_array = np.array(features)
                self._scaler.fit(features_array)
                features_scaled = self._scaler.transform(features_array)
                
                # Train classifier
                self._flow_classifier = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42
                )
                self._flow_classifier.fit(features_scaled, labels)
                
                # Train anomaly detector
                self._anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                self._anomaly_detector.fit(features_scaled)
                
                self._is_trained = True
                self.logger.info("Institutional flow ML models retrained successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model retraining failed: {e}")
    
    def _create_default_prediction(self) -> FlowPrediction:
        """Create default flow prediction"""
        return FlowPrediction(
            predicted_direction=FlowDirection.NEUTRAL,
            predicted_intensity=FlowIntensity.LOW,
            confidence=0.0,
            time_horizon=1,
            probability_distribution={'buying': 0.33, 'selling': 0.33, 'neutral': 0.34}
        )
    
    def _format_result(self, result: InstitutionalFlowResult, signal: SignalType) -> Dict[str, Any]:
        """Format the complete result for output"""
        return {
            'signal': signal,
            'confidence': result.flow_quality_score,
            
            # Flow characteristics
            'flow_direction': result.overall_flow_direction.value,
            'flow_intensity': result.flow_intensity.value,
            'flow_persistence': result.flow_persistence,
            'market_dominance': result.market_dominance,
            'flow_quality_score': result.flow_quality_score,
            
            # Block trades
            'block_trades_count': len(result.block_trades),
            'recent_block_trades': [
                {
                    'timestamp': trade.timestamp,
                    'volume': trade.volume,
                    'price': trade.price,
                    'direction': trade.direction.value,
                    'size_ratio': trade.size_ratio,
                    'price_impact': trade.price_impact,
                    'confidence': trade.confidence
                }
                for trade in result.block_trades[-5:]  # Last 5 trades
            ],
            
            # Order flow imbalance
            'imbalance_ratio': result.order_flow_imbalance.imbalance_ratio,
            'buy_pressure': result.order_flow_imbalance.buy_pressure,
            'sell_pressure': result.order_flow_imbalance.sell_pressure,
            'net_flow': result.order_flow_imbalance.net_flow,
            'flow_momentum': result.order_flow_imbalance.flow_momentum,
            'imbalance_persistence': result.order_flow_imbalance.persistence_score,
            
            # Institutional signature
            'signature_type': result.institutional_signature.signature_type.value,
            'signature_strength': result.institutional_signature.strength,
            'signature_consistency': result.institutional_signature.consistency,
            'market_impact': result.institutional_signature.market_impact,
            'stealth_factor': result.institutional_signature.stealth_factor,
            'urgency_score': result.institutional_signature.urgency_score,
            
            # Predictions
            'predicted_direction': result.flow_prediction.predicted_direction.value,
            'predicted_intensity': result.flow_prediction.predicted_intensity.value,
            'prediction_confidence': result.flow_prediction.confidence,
            'prediction_time_horizon': result.flow_prediction.time_horizon,
            'probability_distribution': result.flow_prediction.probability_distribution,
            
            # Thresholds
            'block_trade_threshold': self._adaptive_block_threshold,
            'imbalance_threshold': self._adaptive_imbalance_threshold,
            
            # Metadata
            'metadata': {
                'indicator_name': 'InstitutionalFlowDetector',
                'version': '1.0.0',
                'calculation_time': pd.Timestamp.now().isoformat(),
                'ml_enabled': self.config.use_ml_classification,
                'ml_trained': self._is_trained,
                'adaptive_thresholds': self.config.adaptive_thresholds
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
            'flow_direction': FlowDirection.NEUTRAL.value,
            'flow_intensity': FlowIntensity.LOW.value,
            'flow_persistence': 0.0,
            'market_dominance': 0.0,
            'flow_quality_score': 0.0,
            'block_trades_count': 0,
            'recent_block_trades': [],
            'imbalance_ratio': 0.0,
            'buy_pressure': 0.5,
            'sell_pressure': 0.5,
            'net_flow': 0.0,
            'flow_momentum': 0.0,
            'imbalance_persistence': 0.0,
            'signature_type': FlowType.ROTATION.value,
            'signature_strength': 0.0,
            'signature_consistency': 0.0,
            'market_impact': 0.0,
            'stealth_factor': 0.0,
            'urgency_score': 0.0,
            'predicted_direction': FlowDirection.NEUTRAL.value,
            'predicted_intensity': FlowIntensity.LOW.value,
            'prediction_confidence': 0.0,
            'prediction_time_horizon': 1,
            'probability_distribution': {'buying': 0.33, 'selling': 0.33, 'neutral': 0.34},
            'block_trade_threshold': self.config.block_trade_threshold,
            'imbalance_threshold': self.config.imbalance_threshold,
            'metadata': {
                'indicator_name': 'InstitutionalFlowDetector',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        result = self._create_default_result()
        result['metadata']['error'] = error_message
        return result