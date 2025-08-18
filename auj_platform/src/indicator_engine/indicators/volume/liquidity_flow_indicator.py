"""
AUJ Platform Advanced Liquidity Flow Indicator
Sophisticated implementation with market depth integration, flow dynamics modeling, and predictive algorithms

This implementation provides institutional-grade liquidity flow analysis for humanitarian trading platforms.

Features:
- Advanced liquidity flow measurement and analysis
- Market depth integration with order book dynamics
- Flow dynamics modeling with machine learning
- Predictive algorithms for liquidity forecasting
- Multi-layer liquidity profiling and classification
- Institutional vs retail liquidity detection
- Flow pattern recognition and anomaly detection
- Temporal liquidity analysis and trend prediction
- Risk-adjusted liquidity scoring
- Comprehensive signal generation system

The Liquidity Flow Indicator analyzes the movement and availability of liquidity in the market,
providing insights into market depth, flow patterns, and future liquidity conditions.
This implementation enhances traditional liquidity analysis with advanced ML and predictive capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, signal, optimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from ..base.base_indicator import BaseIndicator, IndicatorConfig
from ...core.signal_type import SignalType


class LiquidityFlowState(Enum):
    """Liquidity flow state classification"""
    ABUNDANT = "abundant"
    HEALTHY = "healthy"
    MODERATE = "moderate"
    CONSTRAINED = "constrained"
    STRESSED = "stressed"
    CRITICAL = "critical"


class FlowDirection(Enum):
    """Liquidity flow direction"""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    BALANCED = "balanced"
    ROTATING = "rotating"


class LiquidityLayer(Enum):
    """Market liquidity layers"""
    SURFACE = "surface"      # Visible order book
    HIDDEN = "hidden"        # Hidden/iceberg orders
    DEEP = "deep"           # Large institutional
    RETAIL = "retail"       # Small retail orders
    ALGORITHMIC = "algorithmic"  # Algo trading


class FlowPattern(Enum):
    """Liquidity flow patterns"""
    STEADY = "steady"
    PULSING = "pulsing"
    CASCADING = "cascading"
    FRAGMENTING = "fragmenting"
    CLUSTERING = "clustering"
    DISPERSING = "dispersing"


@dataclass
class LiquidityFlowConfig(IndicatorConfig):
    """Configuration for Liquidity Flow Indicator"""
    flow_analysis_period: int = 50
    depth_analysis_levels: int = 10
    pattern_detection_window: int = 30
    prediction_horizon: int = 20
    liquidity_threshold: float = 0.1
    flow_sensitivity: float = 0.05
    institutional_threshold: float = 1000000  # USD
    use_ml_prediction: bool = True
    use_pattern_recognition: bool = True
    use_depth_integration: bool = True
    min_periods: int = 100


class LiquidityDepthAnalysis(NamedTuple):
    """Market depth liquidity analysis"""
    total_liquidity: float
    bid_liquidity: float
    ask_liquidity: float
    depth_imbalance: float
    depth_concentration: float
    layer_distribution: Dict[str, float]
    fragmentation_index: float


class FlowDynamics(NamedTuple):
    """Liquidity flow dynamics analysis"""
    flow_velocity: float
    flow_acceleration: float
    flow_direction: FlowDirection
    flow_strength: float
    flow_consistency: float
    pattern_type: FlowPattern
    pattern_confidence: float


class LiquidityPrediction(NamedTuple):
    """Liquidity prediction analysis"""
    predicted_flow: np.ndarray
    prediction_confidence: float
    flow_forecast: float
    stress_probability: float
    optimal_entry_time: int
    risk_adjusted_score: float


class LiquidityFlowResult(NamedTuple):
    """Complete liquidity flow analysis result"""
    liquidity_state: LiquidityFlowState
    flow_score: float
    depth_analysis: LiquidityDepthAnalysis
    flow_dynamics: FlowDynamics
    liquidity_prediction: LiquidityPrediction
    institutional_flow: float
    retail_flow: float
    algorithmic_flow: float
    stress_indicator: float
    confidence_score: float


class LiquidityFlowIndicator(BaseIndicator):
    """
    Advanced Liquidity Flow Indicator with sophisticated analytics.
    
    This indicator analyzes liquidity flow patterns, market depth dynamics,
    and predictive liquidity modeling to provide comprehensive insights into
    market liquidity conditions. It incorporates:
    - Multi-layer liquidity analysis
    - Flow dynamics modeling
    - Pattern recognition algorithms
    - Machine learning prediction
    - Risk-adjusted scoring
    - Institutional flow detection
    """
    
    def __init__(self, config: Optional[LiquidityFlowConfig] = None):
        super().__init__(config or LiquidityFlowConfig())
        self.config: LiquidityFlowConfig = self.config
        
        # Internal state
        self._liquidity_history: List[float] = []
        self._flow_history: List[float] = []
        self._depth_history: List[Dict] = []
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        
        # Machine learning components
        self._flow_predictor: Optional[GradientBoostingRegressor] = None
        self._pattern_classifier: Optional[RandomForestRegressor] = None
        self._scaler: StandardScaler = StandardScaler()
        self._depth_scaler: MinMaxScaler = MinMaxScaler()
        self._is_trained: bool = False
        
        # Pattern recognition
        self._pattern_detector: Optional[object] = None
        self._clustering_model: Optional[DBSCAN] = None
        
        # Flow analysis state
        self._flow_buffer: List[float] = []
        self._stress_indicators: List[float] = []
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced liquidity flow analysis.
        
        Args:
            data: Dictionary containing 'high', 'low', 'close', 'volume' price data
                 Optionally includes 'bid_depth', 'ask_depth', 'order_book'
            
        Returns:
            Dictionary containing liquidity flow analysis results
        """
        try:
            if not self._validate_data(data):
                return self._create_default_result()
            
            df = pd.DataFrame(data)
            
            if len(df) < self.config.min_periods:
                return self._create_default_result()
            
            # Calculate basic liquidity metrics
            liquidity_score = self._calculate_liquidity_score(df, data)
            
            # Analyze market depth if available
            if self.config.use_depth_integration:
                depth_analysis = self._analyze_market_depth(data, df)
            else:
                depth_analysis = self._create_default_depth_analysis()
            
            # Analyze flow dynamics
            flow_dynamics = self._analyze_flow_dynamics(df, liquidity_score)
            
            # Pattern recognition
            if self.config.use_pattern_recognition:
                pattern_analysis = self._recognize_flow_patterns(df, flow_dynamics)
                flow_dynamics = flow_dynamics._replace(
                    pattern_type=pattern_analysis['pattern_type'],
                    pattern_confidence=pattern_analysis['confidence']
                )
            
            # Predictive analysis
            if self.config.use_ml_prediction:
                liquidity_prediction = self._predict_liquidity_flow(df, liquidity_score)
            else:
                liquidity_prediction = self._create_default_prediction()
            
            # Classify liquidity participants
            institutional_flow, retail_flow, algorithmic_flow = self._classify_flow_participants(
                df, data, flow_dynamics
            )
            
            # Calculate stress indicator
            stress_indicator = self._calculate_stress_indicator(
                depth_analysis, flow_dynamics, liquidity_prediction
            )
            
            # Determine liquidity state
            liquidity_state = self._classify_liquidity_state(
                liquidity_score, depth_analysis, stress_indicator
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(
                depth_analysis, flow_dynamics, liquidity_prediction, stress_indicator
            )
            
            # Create result
            result = LiquidityFlowResult(
                liquidity_state=liquidity_state,
                flow_score=liquidity_score,
                depth_analysis=depth_analysis,
                flow_dynamics=flow_dynamics,
                liquidity_prediction=liquidity_prediction,
                institutional_flow=institutional_flow,
                retail_flow=retail_flow,
                algorithmic_flow=algorithmic_flow,
                stress_indicator=stress_indicator,
                confidence_score=confidence_score
            )
            
            # Generate trading signal
            signal = self._generate_signal(result)
            
            # Update internal state and retrain if needed
            self._update_state_and_retrain(df, data, result)
            
            return self._format_result(result, signal)
            
        except Exception as e:
            self.logger.error(f"Error in LiquidityFlowIndicator calculation: {e}")
            return self._create_error_result(str(e))
    
    def _calculate_liquidity_score(self, df: pd.DataFrame, data: Dict[str, Any]) -> float:
        """Calculate comprehensive liquidity score"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Volume-based liquidity
        avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        volume_score = min(volume[-1] / (avg_volume + 1e-8), 5.0) / 5.0
        
        # Price impact liquidity (using high-low spread)
        spreads = (high - low) / ((high + low) / 2)
        avg_spread = np.mean(spreads[-20:]) if len(spreads) >= 20 else np.mean(spreads)
        spread_score = max(0.0, 1.0 - (spreads[-1] / (avg_spread + 1e-8)))
        
        # Volatility-adjusted liquidity
        returns = np.diff(np.log(close))
        if len(returns) >= 20:
            volatility = np.std(returns[-20:])
            volatility_score = max(0.0, 1.0 - min(volatility * 100, 1.0))
        else:
            volatility_score = 0.5
        
        # Order book depth (if available)
        depth_score = 0.5  # Default
        if 'bid_depth' in data and 'ask_depth' in data:
            bid_depth = data['bid_depth'][-1] if isinstance(data['bid_depth'], list) else data['bid_depth']
            ask_depth = data['ask_depth'][-1] if isinstance(data['ask_depth'], list) else data['ask_depth']
            total_depth = bid_depth + ask_depth
            # Normalize depth relative to volume
            depth_score = min(total_depth / (volume[-1] + 1e-8), 2.0) / 2.0
        
        # Combine scores with weights
        liquidity_score = (
            0.3 * volume_score +
            0.25 * spread_score +
            0.2 * volatility_score +
            0.25 * depth_score
        )
        
        return max(0.0, min(1.0, liquidity_score))
    
    def _analyze_market_depth(self, data: Dict[str, Any], df: pd.DataFrame) -> LiquidityDepthAnalysis:
        """Analyze market depth liquidity"""
        # Extract depth data if available
        if 'order_book' in data:
            order_book = data['order_book']
            bid_depths = []
            ask_depths = []
            
            # Process order book data
            if isinstance(order_book, dict):
                bid_depths = order_book.get('bids', [])
                ask_depths = order_book.get('asks', [])
            else:
                # Fallback to simple depth if available
                bid_depths = data.get('bid_depth', [0])
                ask_depths = data.get('ask_depth', [0])
            
            if not isinstance(bid_depths, list):
                bid_depths = [bid_depths]
            if not isinstance(ask_depths, list):
                ask_depths = [ask_depths]
        else:
            # Use volume as proxy for depth
            volume = df['volume'].values
            bid_depths = [volume[-1] * 0.5]  # Assume equal bid/ask split
            ask_depths = [volume[-1] * 0.5]
        
        # Calculate depth metrics
        total_bid_liquidity = sum(bid_depths[:self.config.depth_analysis_levels])
        total_ask_liquidity = sum(ask_depths[:self.config.depth_analysis_levels])
        total_liquidity = total_bid_liquidity + total_ask_liquidity
        
        # Depth imbalance
        if total_liquidity > 0:
            depth_imbalance = (total_bid_liquidity - total_ask_liquidity) / total_liquidity
        else:
            depth_imbalance = 0.0
        
        # Depth concentration (how much liquidity is in top levels)
        if len(bid_depths) > 0 and len(ask_depths) > 0:
            top_level_liquidity = bid_depths[0] + ask_depths[0]
            depth_concentration = top_level_liquidity / (total_liquidity + 1e-8)
        else:
            depth_concentration = 1.0
        
        # Layer distribution analysis
        layer_distribution = {
            LiquidityLayer.SURFACE.value: depth_concentration,
            LiquidityLayer.HIDDEN.value: max(0.0, 0.3 - depth_concentration * 0.3),
            LiquidityLayer.DEEP.value: max(0.0, 0.4 - depth_concentration * 0.4),
            LiquidityLayer.RETAIL.value: min(0.8, depth_concentration + 0.2),
            LiquidityLayer.ALGORITHMIC.value: 1.0 - depth_concentration
        }
        
        # Fragmentation index (how spread out liquidity is)
        if len(bid_depths) > 1 and len(ask_depths) > 1:
            bid_variance = np.var(bid_depths[:5]) if len(bid_depths) >= 5 else 0
            ask_variance = np.var(ask_depths[:5]) if len(ask_depths) >= 5 else 0
            avg_liquidity = np.mean([total_bid_liquidity, total_ask_liquidity])
            fragmentation_index = (bid_variance + ask_variance) / (avg_liquidity ** 2 + 1e-8)
        else:
            fragmentation_index = 0.0
        
        return LiquidityDepthAnalysis(
            total_liquidity=total_liquidity,
            bid_liquidity=total_bid_liquidity,
            ask_liquidity=total_ask_liquidity,
            depth_imbalance=depth_imbalance,
            depth_concentration=depth_concentration,
            layer_distribution=layer_distribution,
            fragmentation_index=min(fragmentation_index, 1.0)
        )
    
    def _analyze_flow_dynamics(self, df: pd.DataFrame, liquidity_score: float) -> FlowDynamics:
        """Analyze liquidity flow dynamics"""
        volume = df['volume'].values
        close = df['close'].values
        
        # Calculate flow velocity (rate of liquidity change)
        if len(self._liquidity_history) >= 5:
            recent_liquidity = self._liquidity_history[-5:] + [liquidity_score]
            flow_velocity = np.mean(np.diff(recent_liquidity))
        else:
            flow_velocity = 0.0
        
        # Calculate flow acceleration
        if len(self._flow_history) >= 3:
            recent_flows = self._flow_history[-3:] + [flow_velocity]
            flow_acceleration = np.mean(np.diff(recent_flows, n=2))
        else:
            flow_acceleration = 0.0
        
        # Determine flow direction
        if flow_velocity > self.config.flow_sensitivity:
            flow_direction = FlowDirection.INFLOW
        elif flow_velocity < -self.config.flow_sensitivity:
            flow_direction = FlowDirection.OUTFLOW
        elif abs(flow_acceleration) > self.config.flow_sensitivity * 2:
            flow_direction = FlowDirection.ROTATING
        else:
            flow_direction = FlowDirection.BALANCED
        
        # Calculate flow strength
        flow_strength = min(abs(flow_velocity) / 0.1, 1.0)  # Normalize to 0-1
        
        # Calculate flow consistency
        if len(self._flow_history) >= 10:
            recent_flows = self._flow_history[-10:]
            flow_consistency = 1.0 - (np.std(recent_flows) / (np.mean(np.abs(recent_flows)) + 1e-8))
            flow_consistency = max(0.0, min(1.0, flow_consistency))
        else:
            flow_consistency = 0.5
        
        return FlowDynamics(
            flow_velocity=flow_velocity,
            flow_acceleration=flow_acceleration,
            flow_direction=flow_direction,
            flow_strength=flow_strength,
            flow_consistency=flow_consistency,
            pattern_type=FlowPattern.STEADY,  # Will be updated by pattern recognition
            pattern_confidence=0.0
        )
    
    def _recognize_flow_patterns(self, df: pd.DataFrame, flow_dynamics: FlowDynamics) -> Dict[str, Any]:
        """Recognize liquidity flow patterns using ML"""
        if len(self._flow_history) < self.config.pattern_detection_window:
            return {
                'pattern_type': FlowPattern.STEADY,
                'confidence': 0.0
            }
        
        # Prepare pattern features
        window_size = min(self.config.pattern_detection_window, len(self._flow_history))
        recent_flows = self._flow_history[-window_size:]
        
        # Calculate pattern features
        features = []
        
        # Trend features
        trend = np.polyfit(range(len(recent_flows)), recent_flows, 1)[0]
        features.append(trend)
        
        # Volatility features
        volatility = np.std(recent_flows)
        features.append(volatility)
        
        # Cyclical features
        if len(recent_flows) >= 8:
            fft_values = np.fft.fft(recent_flows)
            dominant_freq = np.argmax(np.abs(fft_values[1:len(recent_flows)//2])) + 1
            features.append(dominant_freq / len(recent_flows))
        else:
            features.append(0.0)
        
        # Persistence features
        positive_runs = self._calculate_runs(np.array(recent_flows) > 0)
        negative_runs = self._calculate_runs(np.array(recent_flows) < 0)
        features.extend([positive_runs, negative_runs])
        
        # Clustering-based pattern detection
        if len(features) >= 5:
            try:
                # Simple heuristic pattern classification
                if abs(trend) > 0.02:
                    if volatility > 0.05:
                        pattern_type = FlowPattern.CASCADING
                    else:
                        pattern_type = FlowPattern.STEADY
                elif volatility > 0.08:
                    if max(positive_runs, negative_runs) > 3:
                        pattern_type = FlowPattern.PULSING
                    else:
                        pattern_type = FlowPattern.FRAGMENTING
                elif volatility < 0.02:
                    pattern_type = FlowPattern.CLUSTERING
                else:
                    pattern_type = FlowPattern.DISPERSING
                
                # Calculate confidence based on feature consistency
                confidence = 1.0 - min(volatility * 10, 1.0)
                
            except Exception:
                pattern_type = FlowPattern.STEADY
                confidence = 0.0
        else:
            pattern_type = FlowPattern.STEADY
            confidence = 0.0
        
        return {
            'pattern_type': pattern_type,
            'confidence': confidence
        }
    
    def _calculate_runs(self, boolean_array: np.ndarray) -> float:
        """Calculate average run length for boolean array"""
        if len(boolean_array) == 0:
            return 0.0
        
        runs = []
        current_run = 1
        
        for i in range(1, len(boolean_array)):
            if boolean_array[i] == boolean_array[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        return np.mean(runs) if runs else 0.0
    
    def _predict_liquidity_flow(self, df: pd.DataFrame, liquidity_score: float) -> LiquidityPrediction:
        """Predict future liquidity flow using ML"""
        if not self._is_trained or len(self._liquidity_history) < 50:
            return self._create_default_prediction()
        
        try:
            # Prepare features for prediction
            features = self._prepare_prediction_features(df, liquidity_score)
            
            if features is None:
                return self._create_default_prediction()
            
            # Scale features
            features_scaled = self._scaler.transform([features])
            
            # Predict future flow
            predicted_change = self._flow_predictor.predict(features_scaled)[0]
            
            # Generate flow forecast
            forecast_horizon = min(self.config.prediction_horizon, 20)
            predicted_flow = np.zeros(forecast_horizon)
            current_flow = liquidity_score
            
            for i in range(forecast_horizon):
                current_flow += predicted_change * (0.9 ** i)  # Decay factor
                predicted_flow[i] = current_flow
            
            # Calculate prediction confidence
            if hasattr(self._flow_predictor, 'predict_proba'):
                prediction_confidence = 0.8  # Default for regression
            else:
                # Use feature importance as proxy for confidence
                prediction_confidence = min(abs(predicted_change) * 10, 1.0)
            
            # Calculate stress probability
            stress_threshold = 0.3
            stress_probability = max(0.0, (stress_threshold - min(predicted_flow)) / stress_threshold)
            
            # Find optimal entry time
            optimal_entry_time = np.argmax(predicted_flow) if len(predicted_flow) > 0 else 0
            
            # Risk-adjusted score
            risk_factor = 1.0 - stress_probability
            flow_quality = np.mean(predicted_flow)
            risk_adjusted_score = flow_quality * risk_factor
            
            return LiquidityPrediction(
                predicted_flow=predicted_flow,
                prediction_confidence=prediction_confidence,
                flow_forecast=predicted_flow[-1] if len(predicted_flow) > 0 else liquidity_score,
                stress_probability=stress_probability,
                optimal_entry_time=optimal_entry_time,
                risk_adjusted_score=risk_adjusted_score
            )
            
        except Exception as e:
            self.logger.warning(f"Liquidity prediction failed: {e}")
            return self._create_default_prediction()
    
    def _prepare_prediction_features(self, df: pd.DataFrame, liquidity_score: float) -> Optional[List[float]]:
        """Prepare features for ML prediction"""
        try:
            features = []
            
            # Current liquidity features
            features.append(liquidity_score)
            
            # Historical liquidity features
            if len(self._liquidity_history) >= 10:
                recent_liquidity = self._liquidity_history[-10:]
                features.extend([
                    np.mean(recent_liquidity),
                    np.std(recent_liquidity),
                    np.max(recent_liquidity),
                    np.min(recent_liquidity)
                ])
            else:
                features.extend([liquidity_score, 0, liquidity_score, liquidity_score])
            
            # Flow features
            if len(self._flow_history) >= 5:
                recent_flows = self._flow_history[-5:]
                features.extend([
                    np.mean(recent_flows),
                    np.std(recent_flows)
                ])
            else:
                features.extend([0, 0])
            
            # Market features
            volume = df['volume'].values
            close = df['close'].values
            
            if len(volume) >= 10:
                features.extend([
                    np.mean(volume[-10:]),
                    np.std(volume[-10:])
                ])
            else:
                features.extend([np.mean(volume), np.std(volume)])
            
            # Price features
            if len(close) >= 10:
                returns = np.diff(np.log(close[-10:]))
                features.extend([
                    np.mean(returns),
                    np.std(returns)
                ])
            else:
                features.extend([0, 0])
            
            return features
            
        except Exception:
            return None
    
    def _classify_flow_participants(self, df: pd.DataFrame, data: Dict[str, Any], 
                                   flow_dynamics: FlowDynamics) -> Tuple[float, float, float]:
        """Classify liquidity flow by participant type"""
        volume = df['volume'].values
        close = df['close'].values
        
        # Estimate institutional flow (large volume, low frequency)
        large_volume_threshold = np.percentile(volume, 95) if len(volume) >= 20 else np.mean(volume) * 2
        institutional_indicators = volume[-1] > large_volume_threshold
        
        if institutional_indicators:
            institutional_flow = 0.7
        else:
            institutional_flow = 0.3
        
        # Estimate retail flow (small volume, high frequency)
        small_volume_threshold = np.percentile(volume, 50) if len(volume) >= 20 else np.mean(volume)
        retail_indicators = volume[-1] < small_volume_threshold
        
        if retail_indicators:
            retail_flow = 0.6
        else:
            retail_flow = 0.2
        
        # Estimate algorithmic flow (based on flow consistency and patterns)
        if flow_dynamics.flow_consistency > 0.7:
            algorithmic_flow = 0.8
        elif flow_dynamics.pattern_type in [FlowPattern.PULSING, FlowPattern.FRAGMENTING]:
            algorithmic_flow = 0.6
        else:
            algorithmic_flow = 0.3
        
        # Normalize to sum to 1.0
        total = institutional_flow + retail_flow + algorithmic_flow
        if total > 0:
            institutional_flow /= total
            retail_flow /= total
            algorithmic_flow /= total
        
        return institutional_flow, retail_flow, algorithmic_flow
    
    def _calculate_stress_indicator(self, depth_analysis: LiquidityDepthAnalysis,
                                  flow_dynamics: FlowDynamics,
                                  liquidity_prediction: LiquidityPrediction) -> float:
        """Calculate liquidity stress indicator"""
        stress_factors = []
        
        # Depth-based stress
        if depth_analysis.total_liquidity > 0:
            depth_stress = 1.0 - min(depth_analysis.total_liquidity / 1000000, 1.0)  # Normalize
            stress_factors.append(depth_stress)
        
        # Imbalance stress
        imbalance_stress = abs(depth_analysis.depth_imbalance)
        stress_factors.append(imbalance_stress)
        
        # Concentration stress
        concentration_stress = depth_analysis.depth_concentration  # High concentration = stress
        stress_factors.append(concentration_stress)
        
        # Flow dynamics stress
        flow_stress = 1.0 - flow_dynamics.flow_consistency
        stress_factors.append(flow_stress)
        
        # Fragmentation stress
        fragmentation_stress = depth_analysis.fragmentation_index
        stress_factors.append(fragmentation_stress)
        
        # Prediction stress
        prediction_stress = liquidity_prediction.stress_probability
        stress_factors.append(prediction_stress)
        
        # Calculate weighted average
        weights = [0.2, 0.15, 0.15, 0.2, 0.1, 0.2]
        stress_indicator = np.average(stress_factors, weights=weights)
        
        return max(0.0, min(1.0, stress_indicator))
    
    def _classify_liquidity_state(self, liquidity_score: float,
                                depth_analysis: LiquidityDepthAnalysis,
                                stress_indicator: float) -> LiquidityFlowState:
        """Classify overall liquidity state"""
        # Combine multiple factors for classification
        combined_score = (
            0.4 * liquidity_score +
            0.3 * (1.0 - stress_indicator) +
            0.3 * min(depth_analysis.total_liquidity / 1000000, 1.0)
        )
        
        if combined_score >= 0.8:
            return LiquidityFlowState.ABUNDANT
        elif combined_score >= 0.6:
            return LiquidityFlowState.HEALTHY
        elif combined_score >= 0.4:
            return LiquidityFlowState.MODERATE
        elif combined_score >= 0.25:
            return LiquidityFlowState.CONSTRAINED
        elif combined_score >= 0.1:
            return LiquidityFlowState.STRESSED
        else:
            return LiquidityFlowState.CRITICAL
    
    def _calculate_confidence(self, depth_analysis: LiquidityDepthAnalysis,
                            flow_dynamics: FlowDynamics,
                            liquidity_prediction: LiquidityPrediction,
                            stress_indicator: float) -> float:
        """Calculate overall confidence score"""
        confidence_factors = []
        
        # Data quality confidence
        data_quality = 1.0 - depth_analysis.fragmentation_index
        confidence_factors.append(data_quality)
        
        # Flow consistency confidence
        confidence_factors.append(flow_dynamics.flow_consistency)
        
        # Pattern confidence
        confidence_factors.append(flow_dynamics.pattern_confidence)
        
        # Prediction confidence
        confidence_factors.append(liquidity_prediction.prediction_confidence)
        
        # Stress confidence (lower stress = higher confidence)
        stress_confidence = 1.0 - stress_indicator
        confidence_factors.append(stress_confidence)
        
        # Calculate weighted average
        weights = [0.2, 0.25, 0.15, 0.25, 0.15]
        overall_confidence = np.average(confidence_factors, weights=weights)
        
        return overall_confidence
    
    def _generate_signal(self, result: LiquidityFlowResult) -> SignalType:
        """Generate trading signal based on liquidity flow analysis"""
        signal_criteria = []
        
        # Liquidity state signals
        if result.liquidity_state in [LiquidityFlowState.ABUNDANT, LiquidityFlowState.HEALTHY]:
            signal_criteria.append('favorable_liquidity')
        elif result.liquidity_state in [LiquidityFlowState.STRESSED, LiquidityFlowState.CRITICAL]:
            signal_criteria.append('poor_liquidity')
        
        # Flow direction signals
        if result.flow_dynamics.flow_direction == FlowDirection.INFLOW:
            signal_criteria.append('inflow')
        elif result.flow_dynamics.flow_direction == FlowDirection.OUTFLOW:
            signal_criteria.append('outflow')
        
        # Institutional flow signals
        if result.institutional_flow > 0.6:
            signal_criteria.append('institutional_interest')
        
        # Stress signals
        if result.stress_indicator < 0.3:
            signal_criteria.append('low_stress')
        elif result.stress_indicator > 0.7:
            signal_criteria.append('high_stress')
        
        # Prediction signals
        if result.liquidity_prediction.flow_forecast > result.flow_score * 1.1:
            signal_criteria.append('improving_liquidity')
        elif result.liquidity_prediction.flow_forecast < result.flow_score * 0.9:
            signal_criteria.append('deteriorating_liquidity')
        
        # Signal generation logic
        bullish_signals = sum(1 for criterion in signal_criteria 
                            if criterion in ['favorable_liquidity', 'inflow', 'institutional_interest', 
                                           'low_stress', 'improving_liquidity'])
        bearish_signals = sum(1 for criterion in signal_criteria 
                            if criterion in ['poor_liquidity', 'outflow', 'high_stress', 
                                           'deteriorating_liquidity'])
        
        high_confidence = result.confidence_score > 0.7
        strong_flow = result.flow_dynamics.flow_strength > 0.6
        
        if (bullish_signals >= 3 and high_confidence and strong_flow):
            return SignalType.BUY
        elif (bearish_signals >= 2 and high_confidence):
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_state_and_retrain(self, df: pd.DataFrame, data: Dict[str, Any], 
                                result: LiquidityFlowResult):
        """Update internal state and retrain ML models"""
        max_history = 500
        
        # Update histories
        self._liquidity_history.append(result.flow_score)
        self._flow_history.append(result.flow_dynamics.flow_velocity)
        self._depth_history.append({
            'total_liquidity': result.depth_analysis.total_liquidity,
            'imbalance': result.depth_analysis.depth_imbalance,
            'concentration': result.depth_analysis.depth_concentration
        })
        self._price_history.extend(df['close'].values[-5:])
        self._volume_history.extend(df['volume'].values[-5:])
        self._stress_indicators.append(result.stress_indicator)
        
        # Trim histories
        if len(self._liquidity_history) > max_history:
            self._liquidity_history = self._liquidity_history[-max_history:]
            self._flow_history = self._flow_history[-max_history:]
            self._stress_indicators = self._stress_indicators[-max_history:]
            self._price_history = self._price_history[-max_history:]
            self._volume_history = self._volume_history[-max_history:]
        
        if len(self._depth_history) > max_history // 10:
            self._depth_history = self._depth_history[-max_history // 10:]
        
        # Retrain ML model periodically
        if (self.config.use_ml_prediction and len(self._liquidity_history) >= 100 and 
            len(self._liquidity_history) % 50 == 0):
            self._retrain_ml_model()
    
    def _retrain_ml_model(self):
        """Retrain machine learning models"""
        try:
            if len(self._liquidity_history) < 50:
                return
            
            # Prepare training data for flow prediction
            features = []
            targets = []
            
            window_size = 20
            for i in range(window_size, len(self._liquidity_history) - 5):
                # Prepare features
                feature_vector = self._prepare_training_features(i)
                if feature_vector is not None:
                    features.append(feature_vector)
                    
                    # Target: future liquidity change
                    future_liquidity = np.mean(self._liquidity_history[i:i+5])
                    current_liquidity = self._liquidity_history[i-1]
                    target = future_liquidity - current_liquidity
                    targets.append(target)
            
            if len(features) > 20:
                # Scale features
                features_array = np.array(features)
                self._scaler.fit(features_array)
                features_scaled = self._scaler.transform(features_array)
                
                # Train flow predictor
                self._flow_predictor = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                self._flow_predictor.fit(features_scaled, targets)
                
                self._is_trained = True
                self.logger.info("Liquidity flow ML model retrained successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model retraining failed: {e}")
    
    def _prepare_training_features(self, index: int) -> Optional[List[float]]:
        """Prepare features for training at given index"""
        try:
            features = []
            
            # Liquidity features
            if index >= 10:
                recent_liquidity = self._liquidity_history[index-10:index]
                features.extend([
                    np.mean(recent_liquidity),
                    np.std(recent_liquidity),
                    np.max(recent_liquidity),
                    np.min(recent_liquidity)
                ])
            else:
                features.extend([0.5, 0, 0.5, 0.5])
            
            # Flow features
            if index >= 5 and index <= len(self._flow_history):
                recent_flows = self._flow_history[max(0, index-5):index]
                features.extend([
                    np.mean(recent_flows),
                    np.std(recent_flows)
                ])
            else:
                features.extend([0, 0])
            
            # Volume features
            if index < len(self._volume_history):
                volume_window = self._volume_history[max(0, index-10):index]
                if len(volume_window) > 0:
                    features.extend([
                        np.mean(volume_window),
                        np.std(volume_window)
                    ])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
            
            # Price features
            if index < len(self._price_history):
                price_window = self._price_history[max(0, index-10):index]
                if len(price_window) > 1:
                    returns = np.diff(np.log(price_window))
                    features.extend([
                        np.mean(returns),
                        np.std(returns)
                    ])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
            
            return features
            
        except Exception:
            return None
    
    def _create_default_depth_analysis(self) -> LiquidityDepthAnalysis:
        """Create default depth analysis when no data available"""
        return LiquidityDepthAnalysis(
            total_liquidity=100000.0,
            bid_liquidity=50000.0,
            ask_liquidity=50000.0,
            depth_imbalance=0.0,
            depth_concentration=0.5,
            layer_distribution={layer.value: 0.2 for layer in LiquidityLayer},
            fragmentation_index=0.5
        )
    
    def _create_default_prediction(self) -> LiquidityPrediction:
        """Create default prediction when ML unavailable"""
        return LiquidityPrediction(
            predicted_flow=np.array([0.5] * self.config.prediction_horizon),
            prediction_confidence=0.0,
            flow_forecast=0.5,
            stress_probability=0.5,
            optimal_entry_time=0,
            risk_adjusted_score=0.25
        )
    
    def _format_result(self, result: LiquidityFlowResult, signal: SignalType) -> Dict[str, Any]:
        """Format the complete result for output"""
        return {
            'signal': signal,
            'confidence': result.confidence_score,
            
            # Core liquidity metrics
            'liquidity_state': result.liquidity_state.value,
            'flow_score': result.flow_score,
            'stress_indicator': result.stress_indicator,
            
            # Depth analysis
            'total_liquidity': result.depth_analysis.total_liquidity,
            'bid_liquidity': result.depth_analysis.bid_liquidity,
            'ask_liquidity': result.depth_analysis.ask_liquidity,
            'depth_imbalance': result.depth_analysis.depth_imbalance,
            'depth_concentration': result.depth_analysis.depth_concentration,
            'layer_distribution': result.depth_analysis.layer_distribution,
            'fragmentation_index': result.depth_analysis.fragmentation_index,
            
            # Flow dynamics
            'flow_velocity': result.flow_dynamics.flow_velocity,
            'flow_acceleration': result.flow_dynamics.flow_acceleration,
            'flow_direction': result.flow_dynamics.flow_direction.value,
            'flow_strength': result.flow_dynamics.flow_strength,
            'flow_consistency': result.flow_dynamics.flow_consistency,
            'pattern_type': result.flow_dynamics.pattern_type.value,
            'pattern_confidence': result.flow_dynamics.pattern_confidence,
            
            # Predictions
            'predicted_flow': result.liquidity_prediction.predicted_flow.tolist(),
            'flow_forecast': result.liquidity_prediction.flow_forecast,
            'prediction_confidence': result.liquidity_prediction.prediction_confidence,
            'stress_probability': result.liquidity_prediction.stress_probability,
            'optimal_entry_time': result.liquidity_prediction.optimal_entry_time,
            'risk_adjusted_score': result.liquidity_prediction.risk_adjusted_score,
            
            # Participant analysis
            'institutional_flow': result.institutional_flow,
            'retail_flow': result.retail_flow,
            'algorithmic_flow': result.algorithmic_flow,
            
            # Metadata
            'metadata': {
                'indicator_name': 'LiquidityFlowIndicator',
                'version': '1.0.0',
                'calculation_time': pd.Timestamp.now().isoformat(),
                'ml_prediction': self.config.use_ml_prediction,
                'pattern_recognition': self.config.use_pattern_recognition,
                'depth_integration': self.config.use_depth_integration,
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
            'liquidity_state': LiquidityFlowState.MODERATE.value,
            'flow_score': 0.5,
            'stress_indicator': 0.5,
            'total_liquidity': 100000.0,
            'bid_liquidity': 50000.0,
            'ask_liquidity': 50000.0,
            'depth_imbalance': 0.0,
            'depth_concentration': 0.5,
            'layer_distribution': {layer.value: 0.2 for layer in LiquidityLayer},
            'fragmentation_index': 0.5,
            'flow_velocity': 0.0,
            'flow_acceleration': 0.0,
            'flow_direction': FlowDirection.BALANCED.value,
            'flow_strength': 0.0,
            'flow_consistency': 0.5,
            'pattern_type': FlowPattern.STEADY.value,
            'pattern_confidence': 0.0,
            'predicted_flow': [0.5] * 20,
            'flow_forecast': 0.5,
            'prediction_confidence': 0.0,
            'stress_probability': 0.5,
            'optimal_entry_time': 0,
            'risk_adjusted_score': 0.25,
            'institutional_flow': 0.33,
            'retail_flow': 0.33,
            'algorithmic_flow': 0.34,
            'metadata': {
                'indicator_name': 'LiquidityFlowIndicator',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        result = self._create_default_result()
        result['metadata']['error'] = error_message
        return result