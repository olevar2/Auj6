"""
Order Flow Block Trade Detector - Advanced Institutional Activity Detection
==========================================================================

The Order Flow Block Trade Detector is a sophisticated system for identifying large institutional
trades (block trades) and their market impact. This implementation uses advanced algorithms to
detect unusual trading activity, analyze order flow patterns, and predict institutional behavior.

Key Features:
- Real-time block trade detection with adaptive thresholds
- Order flow imbalance analysis and institutional footprint tracking
- Machine learning-based pattern recognition for trade classification
- Market impact analysis and liquidity assessment
- Multi-timeframe analysis with volume clustering algorithms
- Statistical anomaly detection and false positive filtering

Mathematical Foundation:
Block trades are typically defined as trades exceeding a certain volume threshold
(e.g., 10,000 shares or $1M notional value). The detector uses statistical analysis
to identify outliers in volume distribution and correlates them with price movements
to confirm institutional activity.

The system employs several detection methodologies:
1. Volume Threshold Analysis: Identifies trades above statistical thresholds
2. Order Flow Imbalance: Measures buy/sell pressure imbalances
3. Time-Volume Analysis: Analyzes clustering of large orders in time
4. Price Impact Assessment: Measures immediate and delayed price effects

Author: AUJ Platform Development Team
Created: 2025-06-21
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import find_peaks
import talib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockTradeType(Enum):
    """Block trade classification types."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    ICEBERG = "iceberg"
    STEALTH = "stealth"
    MOMENTUM = "momentum"
    UNKNOWN = "unknown"

class InstitutionalBehavior(Enum):
    """Institutional behavior patterns."""
    AGGRESSIVE_BUYING = "aggressive_buying"
    AGGRESSIVE_SELLING = "aggressive_selling"
    PASSIVE_ACCUMULATION = "passive_accumulation"
    PASSIVE_DISTRIBUTION = "passive_distribution"
    NEUTRAL = "neutral"

class BlockTradeImpact(Enum):
    """Market impact classification."""
    HIGH_IMPACT = "high_impact"
    MEDIUM_IMPACT = "medium_impact"
    LOW_IMPACT = "low_impact"
    NO_IMPACT = "no_impact"

@dataclass
class BlockTradeSignal:
    """
    Comprehensive block trade detection signal.
    
    Attributes:
        timestamp: Signal timestamp
        trade_size: Estimated trade size
        trade_type: Classification of block trade type
        institutional_behavior: Detected institutional behavior pattern
        market_impact: Assessed market impact level
        confidence: Confidence level (0-100)
        flow_imbalance: Order flow imbalance ratio
        price_impact_immediate: Immediate price impact (%)
        price_impact_delayed: Delayed price impact (%)
        volume_percentile: Volume percentile ranking
        detection_score: Overall detection score (0-100)
        stealth_factor: Stealth trading indicator (0-100)
        clustering_score: Volume clustering score
        liquidity_impact: Impact on market liquidity
        follow_through_probability: Probability of continued activity
        risk_level: Risk assessment (0-100)
    """
    timestamp: datetime
    trade_size: float
    trade_type: BlockTradeType
    institutional_behavior: InstitutionalBehavior
    market_impact: BlockTradeImpact
    confidence: float
    flow_imbalance: float
    price_impact_immediate: float
    price_impact_delayed: float
    volume_percentile: float
    detection_score: float
    stealth_factor: float
    clustering_score: float
    liquidity_impact: float
    follow_through_probability: float
    risk_level: float

class OrderFlowBlockTradeDetector:
    """
    Advanced Order Flow Block Trade Detector with institutional activity identification.
    
    This detector uses sophisticated algorithms to identify large institutional trades
    and analyze their market impact using advanced mathematical models and machine
    learning techniques.
    """
    
    def __init__(self,
                 volume_threshold_multiplier: float = 3.0,
                 price_impact_threshold: float = 0.001,
                 detection_window: int = 20,
                 clustering_eps: float = 0.5,
                 min_confidence: float = 70.0,
                 stealth_sensitivity: float = 0.1):
        """
        Initialize the Order Flow Block Trade Detector.
        
        Args:
            volume_threshold_multiplier: Multiplier for volume threshold calculation
            price_impact_threshold: Minimum price impact for block trade confirmation
            detection_window: Window size for block trade detection
            clustering_eps: DBSCAN clustering epsilon parameter
            min_confidence: Minimum confidence for signal generation
            stealth_sensitivity: Sensitivity for stealth trading detection
        """
        self.volume_threshold_multiplier = volume_threshold_multiplier
        self.price_impact_threshold = price_impact_threshold
        self.detection_window = detection_window
        self.clustering_eps = clustering_eps
        self.min_confidence = min_confidence
        self.stealth_sensitivity = stealth_sensitivity
        
        # Initialize machine learning models
        self._block_trade_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self._anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self._volume_clusterer = DBSCAN(eps=clustering_eps, min_samples=3)
        self._scaler = StandardScaler()
        self._robust_scaler = RobustScaler()
        
        # Historical data buffers
        self._volume_history = []
        self._price_history = []
        self._block_trade_history = []
        self._flow_imbalance_history = []
        
        logger.info(f"OrderFlowBlockTradeDetector initialized with threshold_multiplier={volume_threshold_multiplier}")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate block trade detection signals with comprehensive analysis.
        
        Args:
            data: DataFrame with OHLCV data and optional order flow data
            
        Returns:
            Dictionary containing block trade analysis and signals
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(data) < self.detection_window * 2:
                logger.warning(f"Insufficient data: {len(data)} < {self.detection_window * 2}")
                return self._generate_empty_result()
            
            # Perform block trade detection
            block_detection_results = self._detect_block_trades(data)
            
            # Analyze order flow patterns
            order_flow_analysis = self._analyze_order_flow_patterns(data)
            
            # Detect institutional behavior
            institutional_analysis = self._detect_institutional_behavior(data, block_detection_results)
            
            # Assess market impact
            impact_analysis = self._assess_market_impact(data, block_detection_results)
            
            # Perform volume clustering analysis
            clustering_analysis = self._perform_volume_clustering(data)
            
            # Detect stealth trading
            stealth_analysis = self._detect_stealth_trading(data)
            
            # Machine learning analysis
            ml_analysis = self._perform_ml_analysis(data, block_detection_results)
            
            # Generate comprehensive signals
            signals = self._generate_comprehensive_signals(
                data, block_detection_results, order_flow_analysis,
                institutional_analysis, impact_analysis, clustering_analysis,
                stealth_analysis, ml_analysis
            )
            
            return {
                'block_detection_results': block_detection_results,
                'order_flow_analysis': order_flow_analysis,
                'institutional_analysis': institutional_analysis,
                'impact_analysis': impact_analysis,
                'clustering_analysis': clustering_analysis,
                'stealth_analysis': stealth_analysis,
                'ml_analysis': ml_analysis,
                'signals': signals,
                'metadata': self._generate_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in OrderFlowBlockTradeDetector calculation: {str(e)}")
            return self._generate_error_result(str(e))
    
    def _detect_block_trades(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect block trades using statistical and threshold analysis."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate volume statistics
            volume_ma = talib.SMA(volume, timeperiod=20)
            volume_std = talib.STDDEV(volume, timeperiod=20)
            volume_zscore = np.zeros_like(volume)
            
            for i in range(20, len(volume)):
                if volume_std[i] > 0:
                    volume_zscore[i] = (volume[i] - volume_ma[i]) / volume_std[i]
            
            # Dynamic threshold calculation
            volume_threshold = volume_ma + (volume_std * self.volume_threshold_multiplier)
            
            # Block trade identification
            block_trades = np.zeros_like(volume)
            trade_sizes = np.zeros_like(volume)
            
            for i in range(len(volume)):
                if volume[i] > volume_threshold[i] and volume_zscore[i] > 2.0:
                    block_trades[i] = 1
                    trade_sizes[i] = volume[i]
            
            # Price impact calculation
            price_returns = np.zeros_like(close)
            price_returns[1:] = (close[1:] - close[:-1]) / close[:-1]
            
            # Immediate impact (next period)
            immediate_impact = np.zeros_like(close)
            immediate_impact[:-1] = price_returns[1:]
            
            # Delayed impact (5 periods ahead)
            delayed_impact = np.zeros_like(close)
            for i in range(len(close) - 5):
                delayed_impact[i] = (close[i + 5] - close[i]) / close[i]
            
            # Volume percentile ranking
            volume_percentile = np.zeros_like(volume)
            for i in range(20, len(volume)):
                window_volume = volume[max(0, i-20):i]
                volume_percentile[i] = stats.percentileofscore(window_volume, volume[i])
            
            # Block trade confidence scoring
            confidence_scores = self._calculate_block_trade_confidence(
                volume, volume_threshold, volume_zscore, immediate_impact
            )
            
            return {
                'block_trades': block_trades,
                'trade_sizes': trade_sizes,
                'volume_threshold': volume_threshold,
                'volume_zscore': volume_zscore,
                'immediate_impact': immediate_impact,
                'delayed_impact': delayed_impact,
                'volume_percentile': volume_percentile,
                'confidence_scores': confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error detecting block trades: {str(e)}")
            return {}
    
    def _analyze_order_flow_patterns(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Analyze order flow patterns and imbalances."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            
            # Estimate buy/sell volume using tick rule
            buy_volume = np.zeros_like(volume)
            sell_volume = np.zeros_like(volume)
            
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    buy_volume[i] = volume[i]
                elif close[i] < close[i-1]:
                    sell_volume[i] = volume[i]
                else:
                    # Split volume equally for unchanged prices
                    buy_volume[i] = volume[i] * 0.5
                    sell_volume[i] = volume[i] * 0.5
            
            # Order flow imbalance
            flow_imbalance = np.zeros_like(volume)
            for i in range(len(volume)):
                total_volume = buy_volume[i] + sell_volume[i]
                if total_volume > 0:
                    flow_imbalance[i] = (buy_volume[i] - sell_volume[i]) / total_volume
            
            # Rolling flow imbalance
            window_size = self.detection_window
            rolling_flow_imbalance = np.zeros_like(flow_imbalance)
            
            for i in range(window_size, len(flow_imbalance)):
                window_buy = np.sum(buy_volume[i-window_size:i])
                window_sell = np.sum(sell_volume[i-window_size:i])
                total_window_volume = window_buy + window_sell
                
                if total_window_volume > 0:
                    rolling_flow_imbalance[i] = (window_buy - window_sell) / total_window_volume
            
            # Flow momentum
            flow_momentum = np.gradient(rolling_flow_imbalance)
            
            # Volume-weighted average price for the period
            vwap = np.zeros_like(close)
            cumulative_volume = np.zeros_like(volume)
            cumulative_pv = np.zeros_like(volume)
            
            for i in range(len(close)):
                typical_price = (high[i] + low[i] + close[i]) / 3
                cumulative_volume[i] = cumulative_volume[i-1] + volume[i] if i > 0 else volume[i]
                cumulative_pv[i] = cumulative_pv[i-1] + (typical_price * volume[i]) if i > 0 else typical_price * volume[i]
                
                if cumulative_volume[i] > 0:
                    vwap[i] = cumulative_pv[i] / cumulative_volume[i]
            
            # VWAP deviation
            vwap_deviation = (close - vwap) / vwap
            
            return {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'flow_imbalance': flow_imbalance,
                'rolling_flow_imbalance': rolling_flow_imbalance,
                'flow_momentum': flow_momentum,
                'vwap': vwap,
                'vwap_deviation': vwap_deviation
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order flow patterns: {str(e)}")
            return {}
    
    def _detect_institutional_behavior(self, data: pd.DataFrame, block_results: Dict) -> Dict[str, Any]:
        """Detect institutional behavior patterns."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            block_trades = block_results.get('block_trades', np.zeros_like(volume))
            
            # Institutional behavior indicators
            institutional_patterns = []
            behavior_scores = np.zeros_like(volume)
            activity_intensity = np.zeros_like(volume)
            
            for i in range(self.detection_window, len(volume)):
                window_start = max(0, i - self.detection_window)
                window_volume = volume[window_start:i+1]
                window_close = close[window_start:i+1]
                window_blocks = block_trades[window_start:i+1]
                
                # Calculate behavior metrics
                block_frequency = np.sum(window_blocks) / len(window_blocks)
                price_trend = (window_close[-1] - window_close[0]) / window_close[0]
                volume_trend = np.polyfit(range(len(window_volume)), window_volume, 1)[0]
                
                # Behavior classification
                if block_frequency > 0.1:  # Significant block activity
                    if price_trend > 0.01 and volume_trend > 0:
                        behavior = InstitutionalBehavior.AGGRESSIVE_BUYING
                        score = 80
                    elif price_trend < -0.01 and volume_trend > 0:
                        behavior = InstitutionalBehavior.AGGRESSIVE_SELLING
                        score = 80
                    elif price_trend > 0 and volume_trend < 0:
                        behavior = InstitutionalBehavior.PASSIVE_ACCUMULATION
                        score = 60
                    elif price_trend < 0 and volume_trend < 0:
                        behavior = InstitutionalBehavior.PASSIVE_DISTRIBUTION
                        score = 60
                    else:
                        behavior = InstitutionalBehavior.NEUTRAL
                        score = 30
                else:
                    behavior = InstitutionalBehavior.NEUTRAL
                    score = 20
                
                institutional_patterns.append(behavior)
                behavior_scores[i] = score
                activity_intensity[i] = block_frequency * 100
            
            # Pad the beginning of the arrays
            while len(institutional_patterns) < len(volume):
                institutional_patterns.insert(0, InstitutionalBehavior.NEUTRAL)
            
            # Activity clustering
            activity_clusters = self._detect_activity_clusters(volume, block_trades)
            
            return {
                'institutional_patterns': institutional_patterns,
                'behavior_scores': behavior_scores,
                'activity_intensity': activity_intensity,
                'activity_clusters': activity_clusters
            }
            
        except Exception as e:
            logger.error(f"Error detecting institutional behavior: {str(e)}")
            return {}
    
    def _assess_market_impact(self, data: pd.DataFrame, block_results: Dict) -> Dict[str, np.ndarray]:
        """Assess market impact of block trades."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            block_trades = block_results.get('block_trades', np.zeros_like(volume))
            
            # Market impact metrics
            impact_scores = np.zeros_like(close)
            liquidity_impact = np.zeros_like(close)
            volatility_impact = np.zeros_like(close)
            
            for i in range(5, len(close) - 5):
                if block_trades[i] > 0:
                    # Price impact calculation
                    pre_price = close[i-1]
                    post_price_1 = close[i+1] if i+1 < len(close) else close[i]
                    post_price_5 = close[i+5] if i+5 < len(close) else close[i]
                    
                    immediate_impact = abs(post_price_1 - pre_price) / pre_price
                    delayed_impact = abs(post_price_5 - pre_price) / pre_price
                    
                    # Volatility impact
                    pre_volatility = np.std(close[max(0, i-10):i])
                    post_volatility = np.std(close[i:min(len(close), i+10)])
                    volatility_change = (post_volatility - pre_volatility) / pre_volatility if pre_volatility > 0 else 0
                    
                    # Liquidity impact (spread approximation)
                    spread_estimate = (high[i] - low[i]) / close[i]
                    avg_spread = np.mean([(high[j] - low[j]) / close[j] for j in range(max(0, i-5), i)])
                    liquidity_change = (spread_estimate - avg_spread) / avg_spread if avg_spread > 0 else 0
                    
                    # Composite impact score
                    impact_score = (immediate_impact * 0.4 + delayed_impact * 0.3 + 
                                  abs(volatility_change) * 0.2 + abs(liquidity_change) * 0.1)
                    
                    impact_scores[i] = impact_score * 100
                    liquidity_impact[i] = liquidity_change * 100
                    volatility_impact[i] = volatility_change * 100
            
            # Impact classification
            impact_levels = np.zeros_like(impact_scores)
            for i in range(len(impact_scores)):
                if impact_scores[i] > 5.0:
                    impact_levels[i] = 3  # High impact
                elif impact_scores[i] > 2.0:
                    impact_levels[i] = 2  # Medium impact
                elif impact_scores[i] > 0.5:
                    impact_levels[i] = 1  # Low impact
                else:
                    impact_levels[i] = 0  # No impact
            
            return {
                'impact_scores': impact_scores,
                'impact_levels': impact_levels,
                'liquidity_impact': liquidity_impact,
                'volatility_impact': volatility_impact
            }
            
        except Exception as e:
            logger.error(f"Error assessing market impact: {str(e)}")
            return {}
    
    def _perform_volume_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform volume clustering analysis."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            
            # Prepare features for clustering
            features = []
            for i in range(10, len(volume)):
                volume_features = [
                    volume[i],
                    np.mean(volume[i-5:i]),
                    np.std(volume[i-5:i]),
                    volume[i] / np.mean(volume[i-10:i]) if np.mean(volume[i-10:i]) > 0 else 1
                ]
                features.append(volume_features)
            
            if len(features) < 10:
                return {}
            
            features_array = np.array(features)
            features_scaled = self._scaler.fit_transform(features_array)
            
            # Perform clustering
            clusters = self._volume_clusterer.fit_predict(features_scaled)
            
            # Volume cluster analysis
            cluster_labels = np.full_like(volume, -1)
            cluster_labels[10:] = clusters
            
            clustering_scores = np.zeros_like(volume)
            for i in range(len(volume)):
                if cluster_labels[i] != -1:
                    # Calculate clustering score based on cluster membership
                    cluster_size = np.sum(cluster_labels == cluster_labels[i])
                    clustering_scores[i] = min(cluster_size / len(volume) * 100, 100)
            
            return {
                'cluster_labels': cluster_labels,
                'clustering_scores': clustering_scores,
                'num_clusters': len(np.unique(clusters[clusters != -1]))
            }
            
        except Exception as e:
            logger.error(f"Error in volume clustering: {str(e)}")
            return {}
    
    def _detect_stealth_trading(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect stealth trading patterns."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            
            # Stealth indicators
            stealth_scores = np.zeros_like(volume)
            volume_consistency = np.zeros_like(volume)
            price_efficiency = np.zeros_like(volume)
            
            for i in range(20, len(volume)):
                window_volume = volume[i-20:i]
                window_close = close[i-20:i]
                
                # Volume consistency (low variance indicates stealth)
                volume_cv = np.std(window_volume) / np.mean(window_volume) if np.mean(window_volume) > 0 else 1
                volume_consistency[i] = max(0, 100 - (volume_cv * 100))
                
                # Price efficiency (smooth price movement despite volume)
                price_returns = np.diff(window_close) / window_close[:-1]
                price_volatility = np.std(price_returns)
                avg_volume = np.mean(window_volume)
                
                # Stealth score: high volume with low price volatility
                if avg_volume > 0 and price_volatility > 0:
                    efficiency_ratio = (avg_volume / np.mean(volume[:i])) / (price_volatility * 100)
                    price_efficiency[i] = min(efficiency_ratio * 20, 100)
                
                # Combined stealth score
                stealth_scores[i] = (volume_consistency[i] * 0.6 + price_efficiency[i] * 0.4)
            
            return {
                'stealth_scores': stealth_scores,
                'volume_consistency': volume_consistency,
                'price_efficiency': price_efficiency
            }
            
        except Exception as e:
            logger.error(f"Error detecting stealth trading: {str(e)}")
            return {}
    
    def _perform_ml_analysis(self, data: pd.DataFrame, block_results: Dict) -> Dict[str, Any]:
        """Perform machine learning analysis on block trade patterns."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            block_trades = block_results.get('block_trades', np.zeros_like(volume))
            
            # Prepare features
            features = self._prepare_ml_features(data, block_results)
            
            if len(features) < 50:  # Need sufficient data for ML
                return {}
            
            # Anomaly detection
            anomaly_scores = self._detect_volume_anomalies(features)
            
            # Block trade classification
            if np.sum(block_trades) > 10:  # Need sufficient positive examples
                trade_predictions = self._classify_block_trades(features, block_trades)
            else:
                trade_predictions = np.zeros_like(volume)
            
            # Pattern recognition
            pattern_scores = self._recognize_trading_patterns(features)
            
            return {
                'anomaly_scores': anomaly_scores,
                'trade_predictions': trade_predictions,
                'pattern_scores': pattern_scores,
                'feature_importance': self._calculate_feature_importance(features)
            }
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {str(e)}")
            return {}
    
    def _generate_comprehensive_signals(self, data: pd.DataFrame, block_results: Dict,
                                      order_flow_analysis: Dict, institutional_analysis: Dict,
                                      impact_analysis: Dict, clustering_analysis: Dict,
                                      stealth_analysis: Dict, ml_analysis: Dict) -> List[BlockTradeSignal]:
        """Generate comprehensive block trade signals."""
        try:
            signals = []
            timestamps = pd.to_datetime(data.index) if hasattr(data.index, 'to_pydatetime') else data.index
            
            volume = data['volume'].values
            block_trades = block_results.get('block_trades', np.zeros_like(volume))
            trade_sizes = block_results.get('trade_sizes', np.zeros_like(volume))
            confidence_scores = block_results.get('confidence_scores', np.zeros_like(volume))
            
            flow_imbalance = order_flow_analysis.get('flow_imbalance', np.zeros_like(volume))
            institutional_patterns = institutional_analysis.get('institutional_patterns', [InstitutionalBehavior.NEUTRAL] * len(volume))
            impact_scores = impact_analysis.get('impact_scores', np.zeros_like(volume))
            impact_levels = impact_analysis.get('impact_levels', np.zeros_like(volume))
            clustering_scores = clustering_analysis.get('clustering_scores', np.zeros_like(volume))
            stealth_scores = stealth_analysis.get('stealth_scores', np.zeros_like(volume))
            
            for i in range(len(data)):
                if block_trades[i] > 0 and confidence_scores[i] >= self.min_confidence:
                    # Determine trade type
                    trade_type = self._classify_trade_type(i, data, block_results, order_flow_analysis)
                    
                    # Determine market impact
                    impact_level = self._classify_market_impact(impact_levels[i] if i < len(impact_levels) else 0)
                    
                    # Calculate additional metrics
                    immediate_impact = block_results.get('immediate_impact', np.zeros_like(volume))[i] if i < len(block_results.get('immediate_impact', [])) else 0
                    delayed_impact = block_results.get('delayed_impact', np.zeros_like(volume))[i] if i < len(block_results.get('delayed_impact', [])) else 0
                    volume_percentile = block_results.get('volume_percentile', np.zeros_like(volume))[i] if i < len(block_results.get('volume_percentile', [])) else 50
                    
                    # Generate signal
                    signal = BlockTradeSignal(
                        timestamp=timestamps[i] if i < len(timestamps) else datetime.now(),
                        trade_size=trade_sizes[i],
                        trade_type=trade_type,
                        institutional_behavior=institutional_patterns[i] if i < len(institutional_patterns) else InstitutionalBehavior.NEUTRAL,
                        market_impact=impact_level,
                        confidence=confidence_scores[i],
                        flow_imbalance=flow_imbalance[i] if i < len(flow_imbalance) else 0.0,
                        price_impact_immediate=immediate_impact * 100,
                        price_impact_delayed=delayed_impact * 100,
                        volume_percentile=volume_percentile,
                        detection_score=self._calculate_detection_score(i, block_results, institutional_analysis),
                        stealth_factor=stealth_scores[i] if i < len(stealth_scores) else 0.0,
                        clustering_score=clustering_scores[i] if i < len(clustering_scores) else 0.0,
                        liquidity_impact=impact_analysis.get('liquidity_impact', np.zeros_like(volume))[i] if i < len(impact_analysis.get('liquidity_impact', [])) else 0.0,
                        follow_through_probability=self._calculate_follow_through_probability(i, institutional_analysis, impact_analysis),
                        risk_level=self._calculate_risk_level(i, impact_analysis, stealth_analysis)
                    )
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {str(e)}")
            return []
    
    # Helper methods for calculations
    def _calculate_block_trade_confidence(self, volume: np.ndarray, volume_threshold: np.ndarray,
                                        volume_zscore: np.ndarray, price_impact: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for block trade detection."""
        try:
            confidence = np.zeros_like(volume)
            
            for i in range(len(volume)):
                score = 0.0
                
                # Volume threshold component
                if volume[i] > volume_threshold[i]:
                    threshold_ratio = volume[i] / volume_threshold[i]
                    score += min(threshold_ratio * 25, 40)
                
                # Z-score component
                if volume_zscore[i] > 2:
                    score += min(volume_zscore[i] * 10, 30)
                
                # Price impact component
                if abs(price_impact[i]) > self.price_impact_threshold:
                    score += min(abs(price_impact[i]) * 1000, 30)
                
                confidence[i] = min(score, 100)
            
            return confidence
        except:
            return np.zeros_like(volume)
    
    def _detect_activity_clusters(self, volume: np.ndarray, block_trades: np.ndarray) -> np.ndarray:
        """Detect clusters of block trading activity."""
        try:
            clusters = np.zeros_like(volume)
            cluster_id = 1
            
            for i in range(1, len(block_trades)):
                if block_trades[i] > 0:
                    # Check if this is part of an existing cluster
                    window_start = max(0, i - 5)
                    if np.any(block_trades[window_start:i]):
                        # Find the most recent cluster ID in the window
                        recent_clusters = clusters[window_start:i]
                        recent_clusters = recent_clusters[recent_clusters > 0]
                        if len(recent_clusters) > 0:
                            clusters[i] = recent_clusters[-1]
                        else:
                            clusters[i] = cluster_id
                            cluster_id += 1
                    else:
                        clusters[i] = cluster_id
                        cluster_id += 1
            
            return clusters
        except:
            return np.zeros_like(volume)
    
    def _prepare_ml_features(self, data: pd.DataFrame, block_results: Dict) -> np.ndarray:
        """Prepare features for machine learning analysis."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            
            features = []
            
            for i in range(20, len(volume)):
                # Volume features
                vol_features = [
                    volume[i],
                    np.mean(volume[i-5:i]),
                    np.std(volume[i-5:i]),
                    volume[i] / np.mean(volume[i-20:i]) if np.mean(volume[i-20:i]) > 0 else 1
                ]
                
                # Price features
                price_returns = [(close[i-j] - close[i-j-1]) / close[i-j-1] for j in range(5)]
                
                # Combined features
                feature_vector = vol_features + price_returns
                features.append(feature_vector)
            
            return np.array(features)
        except:
            return np.array([])
    
    def _detect_volume_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalies in volume patterns."""
        try:
            if len(features) < 10:
                return np.array([])
            
            features_scaled = self._scaler.fit_transform(features)
            anomaly_scores = self._anomaly_detector.fit_predict(features_scaled)
            
            return anomaly_scores
        except:
            return np.array([])
    
    def _classify_block_trades(self, features: np.ndarray, block_trades: np.ndarray) -> np.ndarray:
        """Classify block trades using machine learning."""
        try:
            if len(features) < 50:
                return np.zeros(len(features))
            
            # Prepare labels (block trade indicators)
            labels = block_trades[20:20+len(features)]  # Align with features
            
            # Train classifier
            self._block_trade_classifier.fit(features, labels)
            predictions = self._block_trade_classifier.predict_proba(features)[:, 1]
            
            return predictions
        except:
            return np.zeros(len(features))
    
    def _recognize_trading_patterns(self, features: np.ndarray) -> np.ndarray:
        """Recognize trading patterns using clustering."""
        try:
            if len(features) < 10:
                return np.array([])
            
            features_scaled = self._scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=5, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Calculate pattern scores based on cluster centers
            pattern_scores = np.zeros(len(features))
            centers = kmeans.cluster_centers_
            
            for i, label in enumerate(cluster_labels):
                # Distance from cluster center (inverted for score)
                distance = np.linalg.norm(features_scaled[i] - centers[label])
                pattern_scores[i] = max(0, 100 - distance * 20)
            
            return pattern_scores
        except:
            return np.array([])
    
    def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from analysis."""
        try:
            feature_names = [
                'current_volume', 'volume_mean_5', 'volume_std_5', 'volume_ratio',
                'return_0', 'return_1', 'return_2', 'return_3', 'return_4'
            ]
            
            if len(features) < 50 or features.shape[1] != len(feature_names):
                return {}
            
            importances = np.var(features, axis=0)
            return dict(zip(feature_names, importances.tolist()))
        except:
            return {}
    
    def _classify_trade_type(self, index: int, data: pd.DataFrame, 
                           block_results: Dict, order_flow_analysis: Dict) -> BlockTradeType:
        """Classify the type of block trade."""
        try:
            flow_imbalance = order_flow_analysis.get('flow_imbalance', np.zeros(len(data)))
            
            if index >= len(flow_imbalance):
                return BlockTradeType.UNKNOWN
            
            imbalance = flow_imbalance[index]
            
            if imbalance > 0.3:
                return BlockTradeType.ACCUMULATION
            elif imbalance < -0.3:
                return BlockTradeType.DISTRIBUTION
            elif abs(imbalance) < 0.1:
                return BlockTradeType.STEALTH
            else:
                return BlockTradeType.UNKNOWN
        except:
            return BlockTradeType.UNKNOWN
    
    def _classify_market_impact(self, impact_level: float) -> BlockTradeImpact:
        """Classify market impact level."""
        if impact_level >= 3:
            return BlockTradeImpact.HIGH_IMPACT
        elif impact_level >= 2:
            return BlockTradeImpact.MEDIUM_IMPACT
        elif impact_level >= 1:
            return BlockTradeImpact.LOW_IMPACT
        else:
            return BlockTradeImpact.NO_IMPACT
    
    def _calculate_detection_score(self, index: int, block_results: Dict, institutional_analysis: Dict) -> float:
        """Calculate overall detection score."""
        try:
            confidence = block_results.get('confidence_scores', np.array([]))[index] if index < len(block_results.get('confidence_scores', [])) else 0
            behavior_score = institutional_analysis.get('behavior_scores', np.array([]))[index] if index < len(institutional_analysis.get('behavior_scores', [])) else 0
            
            return (confidence * 0.7 + behavior_score * 0.3)
        except:
            return 0.0
    
    def _calculate_follow_through_probability(self, index: int, institutional_analysis: Dict, impact_analysis: Dict) -> float:
        """Calculate probability of follow-through activity."""
        try:
            activity_intensity = institutional_analysis.get('activity_intensity', np.array([]))[index] if index < len(institutional_analysis.get('activity_intensity', [])) else 0
            impact_score = impact_analysis.get('impact_scores', np.array([]))[index] if index < len(impact_analysis.get('impact_scores', [])) else 0
            
            # Higher activity intensity and impact suggest higher follow-through probability
            probability = min((activity_intensity + impact_score) / 2, 100)
            return probability
        except:
            return 0.0
    
    def _calculate_risk_level(self, index: int, impact_analysis: Dict, stealth_analysis: Dict) -> float:
        """Calculate risk level associated with the block trade."""
        try:
            volatility_impact = impact_analysis.get('volatility_impact', np.array([]))[index] if index < len(impact_analysis.get('volatility_impact', [])) else 0
            liquidity_impact = impact_analysis.get('liquidity_impact', np.array([]))[index] if index < len(impact_analysis.get('liquidity_impact', [])) else 0
            stealth_score = stealth_analysis.get('stealth_scores', np.array([]))[index] if index < len(stealth_analysis.get('stealth_scores', [])) else 0
            
            # Higher impact means higher risk, higher stealth means lower risk
            risk = abs(volatility_impact) * 0.4 + abs(liquidity_impact) * 0.4 - stealth_score * 0.2
            return max(0, min(risk, 100))
        except:
            return 50.0
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the calculation results."""
        return {
            'indicator_name': 'Order Flow Block Trade Detector',
            'version': '1.0.0',
            'parameters': {
                'volume_threshold_multiplier': self.volume_threshold_multiplier,
                'price_impact_threshold': self.price_impact_threshold,
                'detection_window': self.detection_window,
                'clustering_eps': self.clustering_eps,
                'min_confidence': self.min_confidence,
                'stealth_sensitivity': self.stealth_sensitivity
            },
            'features': [
                'Real-time block trade detection',
                'Order flow imbalance analysis',
                'Institutional behavior classification',
                'Market impact assessment',
                'Volume clustering analysis',
                'Stealth trading detection',
                'Machine learning pattern recognition'
            ],
            'calculation_timestamp': datetime.now(),
            'data_requirements': ['open', 'high', 'low', 'close', 'volume']
        }
    
    def _generate_empty_result(self) -> Dict[str, Any]:
        """Generate empty result structure."""
        return {
            'block_detection_results': {},
            'order_flow_analysis': {},
            'institutional_analysis': {},
            'impact_analysis': {},
            'clustering_analysis': {},
            'stealth_analysis': {},
            'ml_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': 'Insufficient data'
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            'block_detection_results': {},
            'order_flow_analysis': {},
            'institutional_analysis': {},
            'impact_analysis': {},
            'clustering_analysis': {},
            'stealth_analysis': {},
            'ml_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': error_message
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
    
    # Generate realistic OHLCV data with block trade patterns
    base_price = 100
    returns = np.random.normal(0, 0.01, 500)
    
    # Add block trade patterns every 50 periods
    block_volume = np.random.lognormal(8, 0.5, 500)
    for i in range(50, 450, 50):
        # Simulate block trades with higher volume and price impact
        block_volume[i:i+3] *= 5.0  # 5x normal volume
        returns[i:i+3] += np.random.choice([-0.02, 0.02], 3)  # Significant price moves
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 500))),
        'close': prices,
        'volume': block_volume
    }, index=dates)
    
    # Test the detector
    detector = OrderFlowBlockTradeDetector(
        volume_threshold_multiplier=2.5,
        detection_window=20,
        min_confidence=70.0
    )
    
    try:
        result = detector.calculate(sample_data)
        
        print("Order Flow Block Trade Detector Results:")
        print(f"- Calculation successful: {not result.get('error', False)}")
        print(f"- Block trades detected: {len(result.get('signals', []))}")
        print(f"- Detection window: {detector.detection_window}")
        print(f"- Volume threshold multiplier: {detector.volume_threshold_multiplier}")
        
        # Display some sample signals
        signals = result.get('signals', [])
        if signals:
            print(f"\nSample block trade signals (showing first 3):")
            for i, signal in enumerate(signals[:3]):
                print(f"Signal {i+1}:")
                print(f"  Timestamp: {signal.timestamp}")
                print(f"  Trade Size: {signal.trade_size:.0f}")
                print(f"  Trade Type: {signal.trade_type}")
                print(f"  Institutional Behavior: {signal.institutional_behavior}")
                print(f"  Market Impact: {signal.market_impact}")
                print(f"  Confidence: {signal.confidence:.2f}")
                print(f"  Detection Score: {signal.detection_score:.2f}")
                print(f"  Follow-through Probability: {signal.follow_through_probability:.2f}")
                print(f"  Risk Level: {signal.risk_level:.2f}")
        
        print(f"\nMetadata: {result.get('metadata', {}).get('indicator_name', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing Order Flow Block Trade Detector: {str(e)}")
        import traceback
        traceback.print_exc()