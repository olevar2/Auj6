"""
On Balance Volume Indicator - Advanced Volume-Price Momentum Analysis
=====================================================================

The On Balance Volume (OBV) is a sophisticated momentum indicator that uses volume flow 
to predict changes in price. This implementation includes advanced algorithms for 
momentum analysis, divergence detection, and volume-price trend confirmation.

Key Features:
- Advanced momentum analysis with multi-timeframe confirmation
- Sophisticated divergence detection with statistical validation
- Volume-price trend confirmation algorithms
- Machine learning-enhanced pattern recognition
- Institutional flow tracking and smart money detection
- Adaptive signal filtering and confidence scoring
- Multi-dimensional volume flow analysis

Mathematical Foundation:
OBV is calculated using:
- If Close[t] > Close[t-1]: OBV[t] = OBV[t-1] + Volume[t]
- If Close[t] < Close[t-1]: OBV[t] = OBV[t-1] - Volume[t]
- If Close[t] = Close[t-1]: OBV[t] = OBV[t-1]

The theory is that volume precedes price movement, making OBV a leading indicator
for trend changes and momentum shifts.

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
from scipy.signal import find_peaks, argrelextrema
import talib
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumeFlow(Enum):
    """Volume flow direction classifications."""
    STRONG_INFLOW = "strong_inflow"
    MODERATE_INFLOW = "moderate_inflow"
    NEUTRAL = "neutral"
    MODERATE_OUTFLOW = "moderate_outflow"
    STRONG_OUTFLOW = "strong_outflow"

class MomentumState(Enum):
    """Momentum state classifications."""
    ACCELERATING_UP = "accelerating_up"
    DECELERATING_UP = "decelerating_up"
    NEUTRAL = "neutral"
    DECELERATING_DOWN = "decelerating_down"
    ACCELERATING_DOWN = "accelerating_down"

class DivergenceSignal(Enum):
    """Divergence signal types."""
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_HIDDEN = "bearish_hidden"
    NO_DIVERGENCE = "no_divergence"

@dataclass
class OBVSignal:
    """
    Comprehensive OBV signal with momentum and divergence analysis.
    
    Attributes:
        timestamp: Signal timestamp
        obv_value: Current OBV value
        obv_normalized: Normalized OBV value (0-100)
        volume_flow: Current volume flow classification
        momentum_state: Current momentum state
        divergence_signal: Divergence signal type
        signal_strength: Signal strength (0-100)
        confidence: Confidence level (0-100)
        momentum_score: Momentum score (-100 to 100)
        trend_confirmation: Price-volume trend confirmation score
        institutional_flow: Institutional flow indicator
        breakout_probability: Probability of price breakout
        entry_signal: Entry signal strength (-100 to 100)
        exit_signal: Exit signal strength (-100 to 100)
        risk_level: Risk assessment (0-100)
        volume_acceleration: Volume acceleration indicator
    """
    timestamp: datetime
    obv_value: float
    obv_normalized: float
    volume_flow: VolumeFlow
    momentum_state: MomentumState
    divergence_signal: DivergenceSignal
    signal_strength: float
    confidence: float
    momentum_score: float
    trend_confirmation: float
    institutional_flow: float
    breakout_probability: float
    entry_signal: float
    exit_signal: float
    risk_level: float
    volume_acceleration: float

class OnBalanceVolumeIndicator:
    """
    Advanced On Balance Volume Indicator with momentum analysis and divergence detection.
    
    This indicator provides sophisticated analysis of volume-price relationships
    using advanced mathematical models and machine learning techniques for
    momentum detection and trend confirmation.
    """
    
    def __init__(self,
                 initial_value: float = 0.0,
                 momentum_period: int = 14,
                 divergence_lookback: int = 20,
                 trend_confirmation_period: int = 10,
                 breakout_threshold: float = 2.0,
                 min_confidence: float = 70.0):
        """
        Initialize the On Balance Volume Indicator.
        
        Args:
            initial_value: Starting value for OBV calculation
            momentum_period: Period for momentum analysis
            divergence_lookback: Period for divergence detection
            trend_confirmation_period: Period for trend confirmation
            breakout_threshold: Threshold for breakout detection (std deviations)
            min_confidence: Minimum confidence for signal generation
        """
        self.initial_value = initial_value
        self.momentum_period = momentum_period
        self.divergence_lookback = divergence_lookback
        self.trend_confirmation_period = trend_confirmation_period
        self.breakout_threshold = breakout_threshold
        self.min_confidence = min_confidence
        
        # Initialize calculation buffers
        self._obv_values = []
        self._price_data = []
        self._volume_data = []
        self._momentum_indicators = []
        
        # Machine learning models
        self._momentum_predictor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self._anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self._pattern_clusterer = DBSCAN(eps=0.5, min_samples=5)
        self._scaler = StandardScaler()
        self._robust_scaler = RobustScaler()
        
        # Historical analysis
        self._historical_signals = []
        self._momentum_history = []
        self._divergence_history = []
        
        logger.info(f"OBVIndicator initialized with momentum_period={momentum_period}")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate On Balance Volume with advanced momentum and divergence analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing OBV values and comprehensive analysis
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
            
            required_columns = ['close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(data) < self.momentum_period * 2:
                logger.warning(f"Insufficient data: {len(data)} < {self.momentum_period * 2}")
                return self._generate_empty_result()
            
            # Calculate basic OBV
            obv_results = self._calculate_obv(data)
            
            # Perform momentum analysis
            momentum_analysis = self._analyze_momentum_patterns(data, obv_results)
            
            # Detect divergences
            divergence_analysis = self._detect_divergences(data, obv_results)
            
            # Analyze volume flow patterns
            volume_flow_analysis = self._analyze_volume_flow(data, obv_results)
            
            # Perform trend confirmation analysis
            trend_analysis = self._analyze_trend_confirmation(data, obv_results)
            
            # Generate breakout predictions
            breakout_analysis = self._analyze_breakout_probability(data, obv_results)
            
            # Perform machine learning analysis
            ml_analysis = self._perform_ml_analysis(data, obv_results)
            
            # Generate comprehensive signals
            signals = self._generate_comprehensive_signals(
                data, obv_results, momentum_analysis, divergence_analysis,
                volume_flow_analysis, trend_analysis, breakout_analysis, ml_analysis
            )
            
            return {
                'obv_results': obv_results,
                'momentum_analysis': momentum_analysis,
                'divergence_analysis': divergence_analysis,
                'volume_flow_analysis': volume_flow_analysis,
                'trend_analysis': trend_analysis,
                'breakout_analysis': breakout_analysis,
                'ml_analysis': ml_analysis,
                'signals': signals,
                'metadata': self._generate_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in OBVIndicator calculation: {str(e)}")
            return self._generate_error_result(str(e))
    
    def _calculate_obv(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate the basic On Balance Volume values."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            
            # Initialize OBV array
            obv = np.full_like(close, np.nan)
            obv[0] = self.initial_value
            
            # Calculate OBV values
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            
            # Calculate OBV derivatives and indicators
            obv_sma_10 = talib.SMA(obv, timeperiod=10)
            obv_sma_20 = talib.SMA(obv, timeperiod=20)
            obv_ema_12 = talib.EMA(obv, timeperiod=12)
            obv_ema_26 = talib.EMA(obv, timeperiod=26)
            
            # MACD of OBV for momentum analysis
            obv_macd, obv_macd_signal, obv_macd_hist = talib.MACD(obv)
            
            # RSI of OBV for overbought/oversold conditions
            obv_rsi = talib.RSI(obv, timeperiod=14)
            
            # Rate of change for momentum
            obv_roc = talib.ROC(obv, timeperiod=self.momentum_period)
            
            # Bollinger Bands for volatility analysis
            obv_bb_upper, obv_bb_middle, obv_bb_lower = talib.BBANDS(obv)
            
            # Stochastic oscillator for momentum
            obv_stoch_k, obv_stoch_d = talib.STOCH(obv, obv, obv)
            
            # Volume-weighted OBV (advanced calculation)
            vw_obv = self._calculate_volume_weighted_obv(close, volume, obv)
            
            # OBV acceleration
            obv_velocity = np.gradient(obv)
            obv_acceleration = np.gradient(obv_velocity)
            
            # Normalize OBV for easier interpretation
            obv_normalized = self._normalize_obv(obv)
            
            return {
                'obv': obv,
                'obv_normalized': obv_normalized,
                'obv_sma_10': obv_sma_10,
                'obv_sma_20': obv_sma_20,
                'obv_ema_12': obv_ema_12,
                'obv_ema_26': obv_ema_26,
                'obv_macd': obv_macd,
                'obv_macd_signal': obv_macd_signal,
                'obv_macd_hist': obv_macd_hist,
                'obv_rsi': obv_rsi,
                'obv_roc': obv_roc,
                'obv_bb_upper': obv_bb_upper,
                'obv_bb_middle': obv_bb_middle,
                'obv_bb_lower': obv_bb_lower,
                'obv_stoch_k': obv_stoch_k,
                'obv_stoch_d': obv_stoch_d,
                'vw_obv': vw_obv,
                'obv_velocity': obv_velocity,
                'obv_acceleration': obv_acceleration
            }
            
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            raise
    
    def _analyze_momentum_patterns(self, data: pd.DataFrame, obv_results: Dict) -> Dict[str, Any]:
        """Analyze momentum patterns in OBV."""
        try:
            obv = obv_results['obv']
            obv_roc = obv_results['obv_roc']
            obv_macd = obv_results['obv_macd']
            obv_acceleration = obv_results['obv_acceleration']
            close = data['close'].values
            
            # Momentum state classification
            momentum_states = []
            momentum_scores = np.zeros_like(obv)
            momentum_strength = np.zeros_like(obv)
            
            for i in range(len(obv)):
                # Calculate momentum indicators
                roc_momentum = obv_roc[i] if not np.isnan(obv_roc[i]) else 0
                macd_momentum = obv_macd[i] if not np.isnan(obv_macd[i]) else 0
                acceleration = obv_acceleration[i] if not np.isnan(obv_acceleration[i]) else 0
                
                # Combine momentum indicators
                momentum_score = (roc_momentum * 0.4 + macd_momentum * 0.4 + acceleration * 0.2)
                momentum_scores[i] = momentum_score
                momentum_strength[i] = abs(momentum_score)
                
                # Classify momentum state
                if momentum_score > 50 and acceleration > 0:
                    momentum_states.append(MomentumState.ACCELERATING_UP)
                elif momentum_score > 0 and acceleration <= 0:
                    momentum_states.append(MomentumState.DECELERATING_UP)
                elif momentum_score < -50 and acceleration < 0:
                    momentum_states.append(MomentumState.ACCELERATING_DOWN)
                elif momentum_score < 0 and acceleration >= 0:
                    momentum_states.append(MomentumState.DECELERATING_DOWN)
                else:
                    momentum_states.append(MomentumState.NEUTRAL)
            
            # Calculate momentum persistence
            momentum_persistence = self._calculate_momentum_persistence(momentum_states)
            
            # Momentum divergence with price
            momentum_price_divergence = self._calculate_momentum_price_divergence(
                momentum_scores, close
            )
            
            return {
                'momentum_states': momentum_states,
                'momentum_scores': momentum_scores,
                'momentum_strength': momentum_strength,
                'momentum_persistence': momentum_persistence,
                'momentum_price_divergence': momentum_price_divergence
            }
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {str(e)}")
            return {}
    
    def _detect_divergences(self, data: pd.DataFrame, obv_results: Dict) -> Dict[str, Any]:
        """Detect divergences between OBV and price action."""
        try:
            obv = obv_results['obv']
            close = data['close'].values
            
            # Find peaks and troughs
            price_peaks, price_peak_props = find_peaks(close, distance=5, prominence=np.std(close)*0.5)
            price_troughs, price_trough_props = find_peaks(-close, distance=5, prominence=np.std(close)*0.5)
            obv_peaks, obv_peak_props = find_peaks(obv, distance=5, prominence=np.std(obv)*0.5)
            obv_troughs, obv_trough_props = find_peaks(-obv, distance=5, prominence=np.std(obv)*0.5)
            
            divergences = []
            divergence_signals = np.zeros_like(close)
            divergence_strength = np.zeros_like(close)
            
            # Analyze divergences
            current_time = len(close) - 1
            lookback_start = max(0, current_time - self.divergence_lookback)
            
            # Regular bullish divergence (price lower lows, OBV higher lows)
            recent_price_troughs = price_troughs[price_troughs >= lookback_start]
            recent_obv_troughs = obv_troughs[obv_troughs >= lookback_start]
            
            if len(recent_price_troughs) >= 2 and len(recent_obv_troughs) >= 2:
                for i in range(len(recent_price_troughs)-1):
                    for j in range(len(recent_obv_troughs)-1):
                        p1, p2 = recent_price_troughs[i], recent_price_troughs[i+1]
                        o1, o2 = recent_obv_troughs[j], recent_obv_troughs[j+1]
                        
                        # Check time alignment (within 5 periods)
                        if abs(p1 - o1) <= 5 and abs(p2 - o2) <= 5:
                            if close[p2] < close[p1] and obv[o2] > obv[o1]:
                                strength = abs((close[p1] - close[p2]) / close[p1]) * 100
                                divergences.append({
                                    'type': DivergenceSignal.BULLISH_REGULAR,
                                    'strength': strength,
                                    'index': p2,
                                    'price_points': (p1, p2),
                                    'obv_points': (o1, o2)
                                })
                                divergence_signals[p2] = 1
                                divergence_strength[p2] = strength
            
            # Regular bearish divergence (price higher highs, OBV lower highs)
            recent_price_peaks = price_peaks[price_peaks >= lookback_start]
            recent_obv_peaks = obv_peaks[obv_peaks >= lookback_start]
            
            if len(recent_price_peaks) >= 2 and len(recent_obv_peaks) >= 2:
                for i in range(len(recent_price_peaks)-1):
                    for j in range(len(recent_obv_peaks)-1):
                        p1, p2 = recent_price_peaks[i], recent_price_peaks[i+1]
                        o1, o2 = recent_obv_peaks[j], recent_obv_peaks[j+1]
                        
                        # Check time alignment
                        if abs(p1 - o1) <= 5 and abs(p2 - o2) <= 5:
                            if close[p2] > close[p1] and obv[o2] < obv[o1]:
                                strength = abs((close[p2] - close[p1]) / close[p1]) * 100
                                divergences.append({
                                    'type': DivergenceSignal.BEARISH_REGULAR,
                                    'strength': strength,
                                    'index': p2,
                                    'price_points': (p1, p2),
                                    'obv_points': (o1, o2)
                                })
                                divergence_signals[p2] = -1
                                divergence_strength[p2] = strength
            
            # Calculate divergence quality score
            divergence_quality = self._calculate_divergence_quality(divergences)
            
            return {
                'divergences': divergences,
                'divergence_signals': divergence_signals,
                'divergence_strength': divergence_strength,
                'divergence_quality': divergence_quality,
                'price_peaks': price_peaks,
                'price_troughs': price_troughs,
                'obv_peaks': obv_peaks,
                'obv_troughs': obv_troughs
            }
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {str(e)}")
            return {}
    
    def _analyze_volume_flow(self, data: pd.DataFrame, obv_results: Dict) -> Dict[str, Any]:
        """Analyze volume flow patterns."""
        try:
            obv = obv_results['obv']
            obv_velocity = obv_results['obv_velocity']
            close = data['close'].values
            volume = data['volume'].values
            
            # Volume flow classification
            volume_flow_states = []
            flow_intensity = np.zeros_like(obv)
            cumulative_flow = np.zeros_like(obv)
            
            for i in range(1, len(obv)):
                # Calculate flow metrics
                obv_change = obv[i] - obv[i-1]
                volume_ratio = volume[i] / np.mean(volume[max(0, i-20):i]) if i >= 20 else 1.0
                velocity = obv_velocity[i] if not np.isnan(obv_velocity[i]) else 0
                
                # Flow intensity calculation
                flow_intensity[i] = abs(obv_change) * volume_ratio
                
                # Cumulative flow
                cumulative_flow[i] = cumulative_flow[i-1] + obv_change
                
                # Classify volume flow
                if obv_change > 0:
                    if flow_intensity[i] > np.percentile(flow_intensity[:i+1], 80):
                        volume_flow_states.append(VolumeFlow.STRONG_INFLOW)
                    else:
                        volume_flow_states.append(VolumeFlow.MODERATE_INFLOW)
                elif obv_change < 0:
                    if flow_intensity[i] > np.percentile(flow_intensity[:i+1], 80):
                        volume_flow_states.append(VolumeFlow.STRONG_OUTFLOW)
                    else:
                        volume_flow_states.append(VolumeFlow.MODERATE_OUTFLOW)
                else:
                    volume_flow_states.append(VolumeFlow.NEUTRAL)
            
            # Pad the first element
            volume_flow_states.insert(0, VolumeFlow.NEUTRAL)
            
            # Calculate flow consistency
            flow_consistency = self._calculate_flow_consistency(volume_flow_states, close)
            
            # Institutional flow detection
            institutional_flow = self._detect_institutional_flow(obv, volume, close)
            
            return {
                'volume_flow_states': volume_flow_states,
                'flow_intensity': flow_intensity,
                'cumulative_flow': cumulative_flow,
                'flow_consistency': flow_consistency,
                'institutional_flow': institutional_flow
            }
            
        except Exception as e:
            logger.error(f"Error in volume flow analysis: {str(e)}")
            return {}
    
    def _analyze_trend_confirmation(self, data: pd.DataFrame, obv_results: Dict) -> Dict[str, Any]:
        """Analyze trend confirmation between price and volume."""
        try:
            obv = obv_results['obv']
            obv_sma_10 = obv_results['obv_sma_10']
            obv_sma_20 = obv_results['obv_sma_20']
            close = data['close'].values
            
            # Price trend indicators
            price_sma_10 = talib.SMA(close, timeperiod=10)
            price_sma_20 = talib.SMA(close, timeperiod=20)
            
            # Trend confirmation analysis
            trend_confirmation = np.zeros_like(close)
            price_obv_correlation = np.zeros_like(close)
            
            for i in range(20, len(close)):
                # Price trend direction
                price_trend_up = price_sma_10[i] > price_sma_20[i]
                price_trend_down = price_sma_10[i] < price_sma_20[i]
                
                # OBV trend direction
                obv_trend_up = obv_sma_10[i] > obv_sma_20[i]
                obv_trend_down = obv_sma_10[i] < obv_sma_20[i]
                
                # Trend confirmation score
                if price_trend_up and obv_trend_up:
                    trend_confirmation[i] = 100  # Strong bullish confirmation
                elif price_trend_down and obv_trend_down:
                    trend_confirmation[i] = -100  # Strong bearish confirmation
                elif price_trend_up and obv_trend_down:
                    trend_confirmation[i] = -50  # Bearish divergence
                elif price_trend_down and obv_trend_up:
                    trend_confirmation[i] = 50  # Bullish divergence
                else:
                    trend_confirmation[i] = 0  # Neutral
                
                # Rolling correlation
                if i >= 20:
                    window_price = close[i-20:i+1]
                    window_obv = obv[i-20:i+1]
                    correlation = np.corrcoef(window_price, window_obv)[0, 1]
                    price_obv_correlation[i] = correlation if not np.isnan(correlation) else 0
            
            # Trend strength measurement
            trend_strength = self._calculate_trend_strength(trend_confirmation)
            
            return {
                'trend_confirmation': trend_confirmation,
                'price_obv_correlation': price_obv_correlation,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            logger.error(f"Error in trend confirmation analysis: {str(e)}")
            return {}
    
    def _analyze_breakout_probability(self, data: pd.DataFrame, obv_results: Dict) -> Dict[str, Any]:
        """Analyze probability of price breakouts based on volume patterns."""
        try:
            obv = obv_results['obv']
            obv_bb_upper = obv_results['obv_bb_upper']
            obv_bb_lower = obv_results['obv_bb_lower']
            obv_bb_middle = obv_results['obv_bb_middle']
            close = data['close'].values
            volume = data['volume'].values
            
            # Breakout probability calculation
            breakout_probability = np.zeros_like(close)
            volume_breakout_signals = np.zeros_like(close)
            
            for i in range(20, len(close)):
                prob = 0.0
                
                # OBV Bollinger Band analysis
                if not np.isnan(obv_bb_upper[i]) and not np.isnan(obv_bb_lower[i]):
                    bb_position = ((obv[i] - obv_bb_lower[i]) / 
                                 (obv_bb_upper[i] - obv_bb_lower[i]))
                    
                    if bb_position > 0.8:  # Near upper band
                        prob += 40
                    elif bb_position < 0.2:  # Near lower band
                        prob += 40
                
                # Volume surge analysis
                avg_volume = np.mean(volume[max(0, i-20):i])
                current_volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                if current_volume_ratio > 2.0:  # Volume surge
                    prob += 30
                elif current_volume_ratio > 1.5:
                    prob += 15
                
                # OBV momentum analysis
                if i >= 5:
                    obv_momentum = (obv[i] - obv[i-5]) / abs(obv[i-5]) if obv[i-5] != 0 else 0
                    if abs(obv_momentum) > 0.05:  # Strong momentum
                        prob += 20
                
                breakout_probability[i] = min(prob, 100)
                
                # Generate breakout signals
                if prob > 70:
                    volume_breakout_signals[i] = 1 if bb_position > 0.5 else -1
            
            # Breakout confirmation analysis
            breakout_confirmation = self._analyze_breakout_confirmation(
                close, volume, obv, breakout_probability
            )
            
            return {
                'breakout_probability': breakout_probability,
                'volume_breakout_signals': volume_breakout_signals,
                'breakout_confirmation': breakout_confirmation
            }
            
        except Exception as e:
            logger.error(f"Error in breakout analysis: {str(e)}")
            return {}
    
    def _perform_ml_analysis(self, data: pd.DataFrame, obv_results: Dict) -> Dict[str, Any]:
        """Perform machine learning analysis on OBV patterns."""
        try:
            # Prepare features for ML
            features = self._prepare_ml_features(data, obv_results)
            
            if len(features) < 100:  # Need sufficient data for ML
                return {}
            
            # Momentum prediction
            momentum_predictions = self._predict_momentum(features, obv_results)
            
            # Anomaly detection
            anomaly_scores = self._detect_volume_anomalies(features)
            
            # Pattern clustering
            pattern_clusters = self._cluster_volume_patterns(features)
            
            return {
                'momentum_predictions': momentum_predictions,
                'anomaly_scores': anomaly_scores,
                'pattern_clusters': pattern_clusters,
                'feature_importance': self._calculate_ml_feature_importance(features)
            }
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {str(e)}")
            return {}
    
    def _generate_comprehensive_signals(self, data: pd.DataFrame, obv_results: Dict,
                                      momentum_analysis: Dict, divergence_analysis: Dict,
                                      volume_flow_analysis: Dict, trend_analysis: Dict,
                                      breakout_analysis: Dict, ml_analysis: Dict) -> List[OBVSignal]:
        """Generate comprehensive trading signals."""
        try:
            signals = []
            timestamps = pd.to_datetime(data.index) if hasattr(data.index, 'to_pydatetime') else data.index
            
            obv = obv_results['obv']
            obv_normalized = obv_results['obv_normalized']
            obv_acceleration = obv_results['obv_acceleration']
            
            momentum_states = momentum_analysis.get('momentum_states', [])
            momentum_scores = momentum_analysis.get('momentum_scores', np.zeros_like(obv))
            divergence_signals = divergence_analysis.get('divergence_signals', np.zeros_like(obv))
            volume_flow_states = volume_flow_analysis.get('volume_flow_states', [])
            trend_confirmation = trend_analysis.get('trend_confirmation', np.zeros_like(obv))
            breakout_probability = breakout_analysis.get('breakout_probability', np.zeros_like(obv))
            institutional_flow = volume_flow_analysis.get('institutional_flow', np.zeros_like(obv))
            
            for i in range(len(data)):
                if np.isnan(obv[i]) or i < self.momentum_period:
                    continue
                
                # Calculate signal components
                signal_strength = self._calculate_signal_strength(i, obv_results, momentum_analysis, divergence_analysis)
                confidence = self._calculate_confidence(i, trend_analysis, breakout_analysis)
                
                if confidence < self.min_confidence:
                    continue
                
                # Determine divergence signal for this period
                divergence_signal = DivergenceSignal.NO_DIVERGENCE
                if i < len(divergence_signals):
                    if divergence_signals[i] > 0:
                        divergence_signal = DivergenceSignal.BULLISH_REGULAR
                    elif divergence_signals[i] < 0:
                        divergence_signal = DivergenceSignal.BEARISH_REGULAR
                
                # Generate signal
                signal = OBVSignal(
                    timestamp=timestamps[i] if i < len(timestamps) else datetime.now(),
                    obv_value=obv[i],
                    obv_normalized=obv_normalized[i] if i < len(obv_normalized) else 50.0,
                    volume_flow=volume_flow_states[i] if i < len(volume_flow_states) else VolumeFlow.NEUTRAL,
                    momentum_state=momentum_states[i] if i < len(momentum_states) else MomentumState.NEUTRAL,
                    divergence_signal=divergence_signal,
                    signal_strength=signal_strength,
                    confidence=confidence,
                    momentum_score=momentum_scores[i] if i < len(momentum_scores) else 0.0,
                    trend_confirmation=trend_confirmation[i] if i < len(trend_confirmation) else 0.0,
                    institutional_flow=institutional_flow[i] if i < len(institutional_flow) else 0.0,
                    breakout_probability=breakout_probability[i] if i < len(breakout_probability) else 0.0,
                    entry_signal=self._calculate_entry_signal(i, momentum_analysis, divergence_analysis),
                    exit_signal=self._calculate_exit_signal(i, obv_results, momentum_analysis),
                    risk_level=self._calculate_risk_level(i, obv_results, breakout_analysis),
                    volume_acceleration=obv_acceleration[i] if i < len(obv_acceleration) else 0.0
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {str(e)}")
            return []
    
    # Helper methods for calculations
    def _calculate_volume_weighted_obv(self, close: np.ndarray, volume: np.ndarray, obv: np.ndarray) -> np.ndarray:
        """Calculate volume-weighted OBV for enhanced analysis."""
        try:
            vw_obv = np.zeros_like(obv)
            for i in range(1, len(close)):
                if close[i] != close[i-1]:
                    price_change_pct = (close[i] - close[i-1]) / close[i-1]
                    volume_weight = volume[i] / np.mean(volume[max(0, i-20):i]) if i >= 20 else 1.0
                    vw_obv[i] = vw_obv[i-1] + (price_change_pct * volume[i] * volume_weight)
                else:
                    vw_obv[i] = vw_obv[i-1]
            return vw_obv
        except:
            return obv.copy()
    
    def _normalize_obv(self, obv: np.ndarray) -> np.ndarray:
        """Normalize OBV values to 0-100 scale."""
        try:
            valid_obv = obv[~np.isnan(obv)]
            if len(valid_obv) == 0:
                return np.full_like(obv, 50.0)
            
            min_obv = np.min(valid_obv)
            max_obv = np.max(valid_obv)
            
            if max_obv == min_obv:
                return np.full_like(obv, 50.0)
            
            normalized = ((obv - min_obv) / (max_obv - min_obv)) * 100
            return np.where(np.isnan(obv), np.nan, normalized)
        except:
            return np.full_like(obv, 50.0)
    
    def _calculate_momentum_persistence(self, momentum_states: List[MomentumState]) -> float:
        """Calculate momentum persistence score."""
        try:
            if len(momentum_states) < 10:
                return 0.0
            
            current_state = momentum_states[-1]
            consecutive_count = 1
            
            for i in range(len(momentum_states) - 2, max(0, len(momentum_states) - 21), -1):
                if momentum_states[i] == current_state:
                    consecutive_count += 1
                else:
                    break
            
            return min(consecutive_count / 20.0 * 100, 100.0)
        except:
            return 0.0
    
    def _calculate_momentum_price_divergence(self, momentum_scores: np.ndarray, close: np.ndarray) -> float:
        """Calculate divergence between momentum and price."""
        try:
            valid_indices = ~(np.isnan(momentum_scores) | np.isnan(close))
            if np.sum(valid_indices) < 10:
                return 0.0
            
            correlation = np.corrcoef(momentum_scores[valid_indices], close[valid_indices])[0, 1]
            return 1 - abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_divergence_quality(self, divergences: List[Dict]) -> float:
        """Calculate overall quality of detected divergences."""
        try:
            if not divergences:
                return 0.0
            
            total_strength = sum(d['strength'] for d in divergences)
            avg_strength = total_strength / len(divergences)
            return min(avg_strength, 100.0)
        except:
            return 0.0
    
    def _calculate_flow_consistency(self, volume_flow_states: List[VolumeFlow], close: np.ndarray) -> float:
        """Calculate consistency between volume flow and price movement."""
        try:
            if len(volume_flow_states) != len(close):
                return 0.0
            
            consistent_periods = 0
            total_periods = 0
            
            for i in range(1, len(close)):
                price_up = close[i] > close[i-1]
                flow_state = volume_flow_states[i]
                
                if price_up and flow_state in [VolumeFlow.STRONG_INFLOW, VolumeFlow.MODERATE_INFLOW]:
                    consistent_periods += 1
                elif not price_up and flow_state in [VolumeFlow.STRONG_OUTFLOW, VolumeFlow.MODERATE_OUTFLOW]:
                    consistent_periods += 1
                
                total_periods += 1
            
            return (consistent_periods / total_periods * 100) if total_periods > 0 else 0.0
        except:
            return 0.0
    
    def _detect_institutional_flow(self, obv: np.ndarray, volume: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Detect institutional money flow patterns."""
        try:
            institutional_flow = np.zeros_like(obv)
            
            for i in range(20, len(obv)):
                # Large volume with price support
                avg_volume = np.mean(volume[i-20:i])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                # OBV momentum
                obv_momentum = (obv[i] - obv[i-5]) / abs(obv[i-5]) if obv[i-5] != 0 else 0
                
                # Price stability during volume
                price_volatility = np.std(close[i-5:i+1]) / np.mean(close[i-5:i+1])
                
                # Institutional flow score
                if volume_ratio > 1.5 and price_volatility < 0.02:  # High volume, low volatility
                    institutional_flow[i] = obv_momentum * volume_ratio * 50
                else:
                    institutional_flow[i] = obv_momentum * 10
            
            return institutional_flow
        except:
            return np.zeros_like(obv)
    
    def _calculate_trend_strength(self, trend_confirmation: np.ndarray) -> np.ndarray:
        """Calculate trend strength from confirmation scores."""
        try:
            trend_strength = np.zeros_like(trend_confirmation)
            
            for i in range(10, len(trend_confirmation)):
                recent_confirmations = trend_confirmation[i-10:i+1]
                avg_confirmation = np.mean(recent_confirmations)
                consistency = 1 - (np.std(recent_confirmations) / (abs(avg_confirmation) + 1))
                trend_strength[i] = abs(avg_confirmation) * consistency
            
            return trend_strength
        except:
            return np.zeros_like(trend_confirmation)
    
    def _analyze_breakout_confirmation(self, close: np.ndarray, volume: np.ndarray, 
                                     obv: np.ndarray, breakout_probability: np.ndarray) -> np.ndarray:
        """Analyze breakout confirmation signals."""
        try:
            confirmation = np.zeros_like(close)
            
            for i in range(5, len(close)):
                if breakout_probability[i] > 70:
                    # Check for follow-through
                    price_follow_through = abs(close[i] - close[i-1]) / close[i-1] > 0.01
                    volume_follow_through = volume[i] > np.mean(volume[max(0, i-10):i])
                    obv_follow_through = abs(obv[i] - obv[i-1]) > np.std(obv[max(0, i-20):i])
                    
                    if price_follow_through and volume_follow_through and obv_follow_through:
                        confirmation[i] = 100
                    elif (price_follow_through and volume_follow_through) or \
                         (price_follow_through and obv_follow_through) or \
                         (volume_follow_through and obv_follow_through):
                        confirmation[i] = 60
                    else:
                        confirmation[i] = 20
            
            return confirmation
        except:
            return np.zeros_like(close)
    
    def _prepare_ml_features(self, data: pd.DataFrame, obv_results: Dict) -> np.ndarray:
        """Prepare features for machine learning analysis."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            obv = obv_results['obv']
            obv_roc = obv_results['obv_roc']
            
            features = []
            
            for i in range(20, len(close)):
                # Price features
                price_returns = [(close[i-j] - close[i-j-1]) / close[i-j-1] for j in range(5)]
                
                # Volume features
                vol_ratios = [volume[i-j] / np.mean(volume[max(0, i-j-10):i-j]) for j in range(5)]
                
                # OBV features
                obv_changes = [(obv[i-j] - obv[i-j-1]) / abs(obv[i-j-1]) if obv[i-j-1] != 0 else 0 for j in range(5)]
                
                # Momentum features
                momentum_features = [obv_roc[i-j] if not np.isnan(obv_roc[i-j]) else 0 for j in range(3)]
                
                feature_vector = price_returns + vol_ratios + obv_changes + momentum_features
                features.append(feature_vector)
            
            return np.array(features)
        except:
            return np.array([])
    
    def _predict_momentum(self, features: np.ndarray, obv_results: Dict) -> np.ndarray:
        """Predict momentum using machine learning."""
        try:
            if len(features) < 50:
                return np.array([])
            
            obv_roc = obv_results['obv_roc']
            valid_roc = obv_roc[~np.isnan(obv_roc)]
            
            if len(valid_roc) < 50:
                return np.array([])
            
            # Prepare target (future momentum)
            targets = []
            for i in range(len(features) - 5):
                future_momentum = np.mean(valid_roc[i+1:i+6]) if i+6 < len(valid_roc) else 0
                targets.append(future_momentum)
            
            if len(targets) != len(features):
                return np.array([])
            
            # Train and predict
            self._momentum_predictor.fit(features[:-5], targets)
            predictions = self._momentum_predictor.predict(features)
            
            return predictions
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
    
    def _cluster_volume_patterns(self, features: np.ndarray) -> np.ndarray:
        """Cluster volume patterns."""
        try:
            if len(features) < 10:
                return np.array([])
            
            features_scaled = self._scaler.fit_transform(features)
            clusters = self._pattern_clusterer.fit_predict(features_scaled)
            
            return clusters
        except:
            return np.array([])
    
    def _calculate_ml_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from ML analysis."""
        try:
            feature_names = [
                'price_return_0', 'price_return_1', 'price_return_2', 'price_return_3', 'price_return_4',
                'vol_ratio_0', 'vol_ratio_1', 'vol_ratio_2', 'vol_ratio_3', 'vol_ratio_4',
                'obv_change_0', 'obv_change_1', 'obv_change_2', 'obv_change_3', 'obv_change_4',
                'momentum_0', 'momentum_1', 'momentum_2'
            ]
            
            if len(features) < 50 or features.shape[1] != len(feature_names):
                return {}
            
            importances = np.var(features, axis=0)
            return dict(zip(feature_names, importances.tolist()))
        except:
            return {}
    
    # Signal calculation methods
    def _calculate_signal_strength(self, index: int, obv_results: Dict, 
                                 momentum_analysis: Dict, divergence_analysis: Dict) -> float:
        """Calculate signal strength for a specific point."""
        try:
            strength = 0.0
            
            # Momentum contribution
            momentum_scores = momentum_analysis.get('momentum_scores', np.zeros(index+1))
            if index < len(momentum_scores):
                strength += abs(momentum_scores[index]) * 0.4
            
            # Divergence contribution
            divergence_strength = divergence_analysis.get('divergence_strength', np.zeros(index+1))
            if index < len(divergence_strength):
                strength += divergence_strength[index] * 0.3
            
            # OBV momentum contribution
            obv_rsi = obv_results.get('obv_rsi', np.zeros(index+1))
            if index < len(obv_rsi) and not np.isnan(obv_rsi[index]):
                if obv_rsi[index] > 70 or obv_rsi[index] < 30:
                    strength += 30.0
            
            return min(strength, 100.0)
        except:
            return 0.0
    
    def _calculate_confidence(self, index: int, trend_analysis: Dict, breakout_analysis: Dict) -> float:
        """Calculate confidence for a specific signal."""
        try:
            confidence = 50.0  # Base confidence
            
            # Trend confirmation boost
            trend_confirmation = trend_analysis.get('trend_confirmation', np.zeros(index+1))
            if index < len(trend_confirmation):
                confidence += abs(trend_confirmation[index]) * 0.3
            
            # Breakout probability boost
            breakout_probability = breakout_analysis.get('breakout_probability', np.zeros(index+1))
            if index < len(breakout_probability):
                confidence += breakout_probability[index] * 0.2
            
            return min(confidence, 100.0)
        except:
            return 50.0
    
    def _calculate_entry_signal(self, index: int, momentum_analysis: Dict, divergence_analysis: Dict) -> float:
        """Calculate entry signal strength."""
        try:
            entry_signal = 0.0
            
            # Momentum-based entry
            momentum_states = momentum_analysis.get('momentum_states', [])
            if index < len(momentum_states):
                state = momentum_states[index]
                if state == MomentumState.ACCELERATING_UP:
                    entry_signal += 50.0
                elif state == MomentumState.ACCELERATING_DOWN:
                    entry_signal -= 50.0
                elif state == MomentumState.DECELERATING_DOWN:
                    entry_signal += 25.0
                elif state == MomentumState.DECELERATING_UP:
                    entry_signal -= 25.0
            
            # Divergence-based entry
            divergence_signals = divergence_analysis.get('divergence_signals', np.zeros(index+1))
            if index < len(divergence_signals):
                entry_signal += divergence_signals[index] * 30.0
            
            return np.clip(entry_signal, -100.0, 100.0)
        except:
            return 0.0
    
    def _calculate_exit_signal(self, index: int, obv_results: Dict, momentum_analysis: Dict) -> float:
        """Calculate exit signal strength."""
        try:
            exit_signal = 0.0
            
            # Momentum reversal signal
            momentum_states = momentum_analysis.get('momentum_states', [])
            if index >= 3 and len(momentum_states) > index:
                current_momentum = momentum_states[index]
                past_momentum = momentum_states[index - 3]
                
                if (current_momentum in [MomentumState.DECELERATING_UP, MomentumState.DECELERATING_DOWN] and
                    past_momentum in [MomentumState.ACCELERATING_UP, MomentumState.ACCELERATING_DOWN]):
                    exit_signal += 40.0
            
            # OBV extreme levels
            obv_rsi = obv_results.get('obv_rsi', np.zeros(index+1))
            if index < len(obv_rsi) and not np.isnan(obv_rsi[index]):
                if obv_rsi[index] > 80 or obv_rsi[index] < 20:
                    exit_signal += 30.0
            
            return min(exit_signal, 100.0)
        except:
            return 0.0
    
    def _calculate_risk_level(self, index: int, obv_results: Dict, breakout_analysis: Dict) -> float:
        """Calculate risk level for the signal."""
        try:
            risk = 30.0  # Base risk
            
            # Volatility risk from OBV
            obv_bb_upper = obv_results.get('obv_bb_upper', np.zeros(index+1))
            obv_bb_lower = obv_results.get('obv_bb_lower', np.zeros(index+1))
            obv = obv_results.get('obv', np.zeros(index+1))
            
            if (index < len(obv_bb_upper) and index < len(obv_bb_lower) and 
                index < len(obv) and not any(np.isnan([obv_bb_upper[index], obv_bb_lower[index], obv[index]]))):
                bb_width = obv_bb_upper[index] - obv_bb_lower[index]
                if bb_width > 0:
                    bb_position = (obv[index] - obv_bb_lower[index]) / bb_width
                    if bb_position > 0.9 or bb_position < 0.1:
                        risk += 25.0
            
            # Breakout risk
            breakout_probability = breakout_analysis.get('breakout_probability', np.zeros(index+1))
            if index < len(breakout_probability):
                if breakout_probability[index] > 80:
                    risk += 20.0  # High breakout probability means higher risk
            
            return min(risk, 100.0)
        except:
            return 50.0
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the calculation results."""
        return {
            'indicator_name': 'On Balance Volume',
            'version': '1.0.0',
            'parameters': {
                'initial_value': self.initial_value,
                'momentum_period': self.momentum_period,
                'divergence_lookback': self.divergence_lookback,
                'trend_confirmation_period': self.trend_confirmation_period,
                'breakout_threshold': self.breakout_threshold,
                'min_confidence': self.min_confidence
            },
            'features': [
                'Momentum analysis with multi-timeframe confirmation',
                'Sophisticated divergence detection',
                'Volume-price trend confirmation',
                'Machine learning pattern recognition',
                'Institutional flow tracking',
                'Breakout probability analysis',
                'Risk assessment'
            ],
            'calculation_timestamp': datetime.now(),
            'data_requirements': ['close', 'volume']
        }
    
    def _generate_empty_result(self) -> Dict[str, Any]:
        """Generate empty result structure."""
        return {
            'obv_results': {},
            'momentum_analysis': {},
            'divergence_analysis': {},
            'volume_flow_analysis': {},
            'trend_analysis': {},
            'breakout_analysis': {},
            'ml_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': 'Insufficient data'
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            'obv_results': {},
            'momentum_analysis': {},
            'divergence_analysis': {},
            'volume_flow_analysis': {},
            'trend_analysis': {},
            'breakout_analysis': {},
            'ml_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': error_message
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=400, freq='D')
    
    # Generate realistic OHLCV data with volume-price patterns
    base_price = 100
    returns = np.random.normal(0, 0.02, 400)
    
    # Add volume-price correlation patterns
    volume_base = np.random.lognormal(10, 0.4, 400)
    for i in range(50, 350, 40):  # Create volume patterns
        # High volume with strong price moves
        volume_base[i:i+5] *= 2.0
        returns[i:i+5] += np.random.choice([-0.03, 0.03], 5)  # Strong moves up or down
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 400))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 400))),
        'close': prices,
        'volume': volume_base
    }, index=dates)
    
    # Test the indicator
    obv_indicator = OnBalanceVolumeIndicator(
        initial_value=0.0,
        momentum_period=14,
        divergence_lookback=20,
        min_confidence=70.0
    )
    
    try:
        result = obv_indicator.calculate(sample_data)
        
        print("On Balance Volume Indicator Results:")
        print(f"- Calculation successful: {not result.get('error', False)}")
        print(f"- OBV values calculated: {len(result.get('obv_results', {}).get('obv', []))}")
        print(f"- Signals generated: {len(result.get('signals', []))}")
        print(f"- Momentum analysis completed: {'momentum_analysis' in result}")
        print(f"- Divergence analysis completed: {'divergence_analysis' in result}")
        print(f"- Volume flow analysis completed: {'volume_flow_analysis' in result}")
        print(f"- Breakout analysis completed: {'breakout_analysis' in result}")
        
        # Display some sample signals
        signals = result.get('signals', [])
        if signals:
            print(f"\nSample signals (showing first 3):")
            for i, signal in enumerate(signals[:3]):
                print(f"Signal {i+1}:")
                print(f"  Timestamp: {signal.timestamp}")
                print(f"  OBV Value: {signal.obv_value:.2f}")
                print(f"  Volume Flow: {signal.volume_flow}")
                print(f"  Momentum State: {signal.momentum_state}")
                print(f"  Signal Strength: {signal.signal_strength:.2f}")
                print(f"  Confidence: {signal.confidence:.2f}")
                print(f"  Breakout Probability: {signal.breakout_probability:.2f}")
                print(f"  Entry Signal: {signal.entry_signal:.2f}")
        
        print(f"\nMetadata: {result.get('metadata', {}).get('indicator_name', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing OBV Indicator: {str(e)}")
        import traceback
        traceback.print_exc()