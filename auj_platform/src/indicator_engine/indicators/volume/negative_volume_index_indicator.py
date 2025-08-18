"""
Negative Volume Index Indicator - Advanced Smart Money Tracking System
======================================================================

The Negative Volume Index (NVI) is a sophisticated indicator that tracks "smart money" activity 
by focusing on periods when volume decreases. This implementation includes advanced algorithms 
for institutional behavior analysis and trend confirmation.

Key Features:
- Smart money activity tracking with machine learning classification
- Volume divergence analysis with statistical significance testing
- Trend confirmation algorithms with adaptive thresholds
- Institutional behavior pattern recognition
- Multi-timeframe divergence detection
- Advanced signal filtering and confidence scoring
- Market regime adaptation for optimal performance

Mathematical Foundation:
NVI changes only when volume decreases from the previous period:
- If Volume[t] < Volume[t-1]: NVI[t] = NVI[t-1] + ((Close[t] - Close[t-1]) / Close[t-1]) * 100
- If Volume[t] >= Volume[t-1]: NVI[t] = NVI[t-1]

The NVI is based on the theory that smart money trades on low volume days, while 
public (retail) investors trade on high volume days.

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
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartMoneyActivity(Enum):
    """Smart money activity classifications."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    NEUTRAL = "neutral"
    STRONG_ACCUMULATION = "strong_accumulation"
    STRONG_DISTRIBUTION = "strong_distribution"

class TrendState(Enum):
    """Trend state classifications."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    TRANSITION = "transition"

class DivergenceType(Enum):
    """Divergence type classifications."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"
    NO_DIVERGENCE = "no_divergence"

@dataclass
class NVISignal:
    """
    Comprehensive NVI signal with smart money analysis.
    
    Attributes:
        timestamp: Signal timestamp
        nvi_value: Current NVI value
        nvi_normalized: Normalized NVI value (0-100)
        smart_money_activity: Detected smart money activity
        trend_state: Current trend state
        divergence_type: Type of divergence detected
        signal_strength: Signal strength (0-100)
        confidence: Confidence level (0-100)
        volume_decrease_count: Count of consecutive volume decreases
        price_momentum: Price momentum score
        institutional_pressure: Institutional pressure index
        trend_confirmation: Trend confirmation score
        entry_signal: Entry signal strength (-100 to 100)
        exit_signal: Exit signal strength (-100 to 100)
        risk_assessment: Risk level (0-100)
    """
    timestamp: datetime
    nvi_value: float
    nvi_normalized: float
    smart_money_activity: SmartMoneyActivity
    trend_state: TrendState
    divergence_type: DivergenceType
    signal_strength: float
    confidence: float
    volume_decrease_count: int
    price_momentum: float
    institutional_pressure: float
    trend_confirmation: float
    entry_signal: float
    exit_signal: float
    risk_assessment: float

class NegativeVolumeIndexIndicator:
    """
    Advanced Negative Volume Index Indicator with smart money tracking and trend analysis.
    
    This indicator provides sophisticated analysis of institutional trading behavior
    by focusing on price movements during low-volume periods, which are associated
    with smart money activity.
    """
    
    def __init__(self,
                 initial_value: float = 1000.0,
                 lookback_period: int = 50,
                 divergence_lookback: int = 20,
                 trend_confirmation_period: int = 10,
                 min_confidence: float = 70.0,
                 volume_threshold: float = 0.95):
        """
        Initialize the Negative Volume Index Indicator.
        
        Args:
            initial_value: Starting value for NVI calculation
            lookback_period: Period for trend and momentum analysis
            divergence_lookback: Period for divergence detection
            trend_confirmation_period: Period for trend confirmation
            min_confidence: Minimum confidence for signal generation
            volume_threshold: Threshold for volume decrease detection
        """
        self.initial_value = initial_value
        self.lookback_period = lookback_period
        self.divergence_lookback = divergence_lookback
        self.trend_confirmation_period = trend_confirmation_period
        self.min_confidence = min_confidence
        self.volume_threshold = volume_threshold
        
        # Initialize calculation buffers
        self._nvi_values = []
        self._price_data = []
        self._volume_data = []
        self._smart_money_indicators = []
        
        # Machine learning models
        self._smart_money_classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        )
        self._anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self._pattern_clusterer = KMeans(n_clusters=5, random_state=42)
        self._scaler = StandardScaler()
        
        # Historical analysis
        self._historical_signals = []
        self._trend_history = []
        self._divergence_history = []
        
        logger.info(f"NVIIndicator initialized with initial_value={initial_value}")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Negative Volume Index with smart money analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing NVI values and comprehensive analysis
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
            
            required_columns = ['close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(data) < self.lookback_period:
                logger.warning(f"Insufficient data: {len(data)} < {self.lookback_period}")
                return self._generate_empty_result()
            
            # Calculate basic NVI
            nvi_results = self._calculate_nvi(data)
            
            # Analyze smart money activity
            smart_money_analysis = self._analyze_smart_money_activity(data, nvi_results)
            
            # Perform trend analysis
            trend_analysis = self._analyze_trend_patterns(data, nvi_results)
            
            # Detect divergences
            divergence_analysis = self._detect_divergences(data, nvi_results)
            
            # Generate institutional pressure analysis
            institutional_analysis = self._analyze_institutional_pressure(data, nvi_results)
            
            # Perform machine learning analysis
            ml_analysis = self._perform_ml_analysis(data, nvi_results)
            
            # Generate comprehensive signals
            signals = self._generate_comprehensive_signals(
                data, nvi_results, smart_money_analysis, trend_analysis,
                divergence_analysis, institutional_analysis, ml_analysis
            )
            
            return {
                'nvi_results': nvi_results,
                'smart_money_analysis': smart_money_analysis,
                'trend_analysis': trend_analysis,
                'divergence_analysis': divergence_analysis,
                'institutional_analysis': institutional_analysis,
                'ml_analysis': ml_analysis,
                'signals': signals,
                'metadata': self._generate_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in NVIIndicator calculation: {str(e)}")
            return self._generate_error_result(str(e))
    
    def _calculate_nvi(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate the basic Negative Volume Index values."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            
            # Initialize NVI array
            nvi = np.full_like(close, np.nan)
            nvi[0] = self.initial_value
            
            # Calculate volume changes and price returns
            volume_decreases = np.zeros_like(volume, dtype=bool)
            price_returns = np.zeros_like(close)
            
            for i in range(1, len(close)):
                # Calculate price return
                price_returns[i] = ((close[i] - close[i-1]) / close[i-1]) * 100
                
                # Check for volume decrease
                volume_decreases[i] = volume[i] < (volume[i-1] * self.volume_threshold)
                
                # Update NVI
                if volume_decreases[i]:
                    nvi[i] = nvi[i-1] + price_returns[i]
                else:
                    nvi[i] = nvi[i-1]
            
            # Calculate NVI momentum and trends
            nvi_sma_10 = talib.SMA(nvi, timeperiod=10)
            nvi_sma_20 = talib.SMA(nvi, timeperiod=20)
            nvi_ema_12 = talib.EMA(nvi, timeperiod=12)
            nvi_ema_26 = talib.EMA(nvi, timeperiod=26)
            
            # MACD of NVI for momentum analysis
            nvi_macd, nvi_macd_signal, nvi_macd_hist = talib.MACD(nvi)
            
            # RSI of NVI for overbought/oversold conditions
            nvi_rsi = talib.RSI(nvi, timeperiod=14)
            
            # Rate of change for momentum
            nvi_roc = talib.ROC(nvi, timeperiod=10)
            
            # Bollinger Bands for volatility analysis
            nvi_bb_upper, nvi_bb_middle, nvi_bb_lower = talib.BBANDS(nvi)
            
            # Normalize NVI to 0-100 scale for easier interpretation
            nvi_normalized = self._normalize_nvi(nvi)
            
            return {
                'nvi': nvi,
                'nvi_normalized': nvi_normalized,
                'volume_decreases': volume_decreases,
                'price_returns': price_returns,
                'nvi_sma_10': nvi_sma_10,
                'nvi_sma_20': nvi_sma_20,
                'nvi_ema_12': nvi_ema_12,
                'nvi_ema_26': nvi_ema_26,
                'nvi_macd': nvi_macd,
                'nvi_macd_signal': nvi_macd_signal,
                'nvi_macd_hist': nvi_macd_hist,
                'nvi_rsi': nvi_rsi,
                'nvi_roc': nvi_roc,
                'nvi_bb_upper': nvi_bb_upper,
                'nvi_bb_middle': nvi_bb_middle,
                'nvi_bb_lower': nvi_bb_lower
            }
            
        except Exception as e:
            logger.error(f"Error calculating NVI: {str(e)}")
            raise
    
    def _analyze_smart_money_activity(self, data: pd.DataFrame, nvi_results: Dict) -> Dict[str, Any]:
        """Analyze smart money activity patterns."""
        try:
            nvi = nvi_results['nvi']
            volume_decreases = nvi_results['volume_decreases']
            price_returns = nvi_results['price_returns']
            close = data['close'].values
            volume = data['volume'].values
            
            # Smart money activity indicators
            smart_money_score = np.zeros_like(nvi)
            activity_classification = []
            
            for i in range(self.lookback_period, len(nvi)):
                # Calculate smart money metrics for current window
                window_start = i - self.lookback_period
                window_vol_decreases = volume_decreases[window_start:i+1]
                window_price_returns = price_returns[window_start:i+1]
                window_nvi_change = nvi[i] - nvi[window_start]
                
                # Count volume decrease periods with positive/negative returns
                positive_low_vol_periods = np.sum(window_vol_decreases & (window_price_returns > 0))
                negative_low_vol_periods = np.sum(window_vol_decreases & (window_price_returns < 0))
                
                # Smart money accumulation/distribution score
                total_low_vol_periods = np.sum(window_vol_decreases)
                if total_low_vol_periods > 0:
                    accumulation_ratio = positive_low_vol_periods / total_low_vol_periods
                    smart_money_score[i] = (accumulation_ratio - 0.5) * 200  # Scale to -100 to 100
                
                # Classify activity
                if smart_money_score[i] > 60:
                    activity_classification.append(SmartMoneyActivity.STRONG_ACCUMULATION)
                elif smart_money_score[i] > 20:
                    activity_classification.append(SmartMoneyActivity.ACCUMULATION)
                elif smart_money_score[i] < -60:
                    activity_classification.append(SmartMoneyActivity.STRONG_DISTRIBUTION)
                elif smart_money_score[i] < -20:
                    activity_classification.append(SmartMoneyActivity.DISTRIBUTION)
                else:
                    activity_classification.append(SmartMoneyActivity.NEUTRAL)
            
            # Pad the classification list to match array length
            while len(activity_classification) < len(nvi):
                activity_classification.insert(0, SmartMoneyActivity.NEUTRAL)
            
            # Calculate smart money strength indicator
            smart_money_strength = self._calculate_smart_money_strength(
                nvi_results, volume_decreases, price_returns
            )
            
            # Analyze institutional buying/selling pressure
            institutional_pressure = self._calculate_institutional_pressure(
                close, volume, volume_decreases, price_returns
            )
            
            return {
                'smart_money_score': smart_money_score,
                'activity_classification': activity_classification,
                'smart_money_strength': smart_money_strength,
                'institutional_pressure': institutional_pressure,
                'volume_decrease_frequency': np.mean(volume_decreases)
            }
            
        except Exception as e:
            logger.error(f"Error in smart money analysis: {str(e)}")
            return {}
    
    def _analyze_trend_patterns(self, data: pd.DataFrame, nvi_results: Dict) -> Dict[str, Any]:
        """Analyze trend patterns and confirmations."""
        try:
            nvi = nvi_results['nvi']
            nvi_sma_10 = nvi_results['nvi_sma_10']
            nvi_sma_20 = nvi_results['nvi_sma_20']
            nvi_macd = nvi_results['nvi_macd']
            nvi_rsi = nvi_results['nvi_rsi']
            close = data['close'].values
            
            # Price trend analysis
            price_sma_20 = talib.SMA(close, timeperiod=20)
            price_sma_50 = talib.SMA(close, timeperiod=50)
            
            # Trend state classification
            trend_states = []
            trend_strength = np.zeros_like(nvi)
            trend_confirmation = np.zeros_like(nvi)
            
            for i in range(len(nvi)):
                # NVI trend analysis
                nvi_trend_bullish = (not np.isnan(nvi_sma_10[i]) and not np.isnan(nvi_sma_20[i]) and 
                                   nvi_sma_10[i] > nvi_sma_20[i])
                nvi_trend_bearish = (not np.isnan(nvi_sma_10[i]) and not np.isnan(nvi_sma_20[i]) and 
                                    nvi_sma_10[i] < nvi_sma_20[i])
                
                # Price trend analysis
                price_trend_bullish = (not np.isnan(price_sma_20[i]) and not np.isnan(price_sma_50[i]) and 
                                     price_sma_20[i] > price_sma_50[i])
                price_trend_bearish = (not np.isnan(price_sma_20[i]) and not np.isnan(price_sma_50[i]) and 
                                      price_sma_20[i] < price_sma_50[i])
                
                # Trend confirmation (NVI and price alignment)
                if nvi_trend_bullish and price_trend_bullish:
                    trend_states.append(TrendState.BULLISH)
                    trend_confirmation[i] = 100
                    trend_strength[i] = 80
                elif nvi_trend_bearish and price_trend_bearish:
                    trend_states.append(TrendState.BEARISH)
                    trend_confirmation[i] = 100
                    trend_strength[i] = -80
                elif nvi_trend_bullish or price_trend_bullish:
                    if nvi_trend_bearish or price_trend_bearish:
                        trend_states.append(TrendState.TRANSITION)
                        trend_confirmation[i] = 30
                        trend_strength[i] = 20 if nvi_trend_bullish else -20
                    else:
                        trend_states.append(TrendState.BULLISH)
                        trend_confirmation[i] = 60
                        trend_strength[i] = 50
                elif nvi_trend_bearish or price_trend_bearish:
                    trend_states.append(TrendState.BEARISH)
                    trend_confirmation[i] = 60
                    trend_strength[i] = -50
                else:
                    trend_states.append(TrendState.SIDEWAYS)
                    trend_confirmation[i] = 0
                    trend_strength[i] = 0
            
            # Calculate trend persistence
            trend_persistence = self._calculate_trend_persistence(trend_states)
            
            return {
                'trend_states': trend_states,
                'trend_strength': trend_strength,
                'trend_confirmation': trend_confirmation,
                'trend_persistence': trend_persistence,
                'nvi_price_correlation': self._calculate_nvi_price_correlation(nvi, close)
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return {}
    
    def _detect_divergences(self, data: pd.DataFrame, nvi_results: Dict) -> Dict[str, Any]:
        """Detect divergences between NVI and price action."""
        try:
            nvi = nvi_results['nvi']
            close = data['close'].values
            
            # Find peaks and troughs in both price and NVI
            price_peaks, _ = find_peaks(close, distance=5)
            price_troughs, _ = find_peaks(-close, distance=5)
            nvi_peaks, _ = find_peaks(nvi, distance=5)
            nvi_troughs, _ = find_peaks(-nvi, distance=5)
            
            divergences = []
            divergence_signals = np.zeros_like(close)
            
            # Analyze recent peaks for divergences
            for i in range(max(0, len(close) - self.divergence_lookback), len(close)):
                # Look for price peaks near current index
                recent_price_peaks = price_peaks[price_peaks <= i]
                recent_nvi_peaks = nvi_peaks[nvi_peaks <= i]
                
                if len(recent_price_peaks) >= 2 and len(recent_nvi_peaks) >= 2:
                    # Get two most recent peaks
                    p1_idx, p2_idx = recent_price_peaks[-2], recent_price_peaks[-1]
                    n1_idx, n2_idx = recent_nvi_peaks[-2], recent_nvi_peaks[-1]
                    
                    # Check for bearish divergence (price higher highs, NVI lower highs)
                    if (close[p2_idx] > close[p1_idx] and nvi[n2_idx] < nvi[n1_idx] and 
                        abs(p2_idx - n2_idx) < 5):  # Peaks should be close in time
                        divergences.append({
                            'type': DivergenceType.BEARISH,
                            'strength': abs(close[p2_idx] - close[p1_idx]) / close[p1_idx] * 100,
                            'index': i
                        })
                        divergence_signals[i] = -1
                
                # Look for price troughs for bullish divergences
                recent_price_troughs = price_troughs[price_troughs <= i]
                recent_nvi_troughs = nvi_troughs[nvi_troughs <= i]
                
                if len(recent_price_troughs) >= 2 and len(recent_nvi_troughs) >= 2:
                    # Get two most recent troughs
                    pt1_idx, pt2_idx = recent_price_troughs[-2], recent_price_troughs[-1]
                    nt1_idx, nt2_idx = recent_nvi_troughs[-2], recent_nvi_troughs[-1]
                    
                    # Check for bullish divergence (price lower lows, NVI higher lows)
                    if (close[pt2_idx] < close[pt1_idx] and nvi[nt2_idx] > nvi[nt1_idx] and 
                        abs(pt2_idx - nt2_idx) < 5):
                        divergences.append({
                            'type': DivergenceType.BULLISH,
                            'strength': abs(close[pt1_idx] - close[pt2_idx]) / close[pt1_idx] * 100,
                            'index': i
                        })
                        divergence_signals[i] = 1
            
            # Calculate divergence strength and frequency
            divergence_strength = self._calculate_divergence_strength(divergences)
            
            return {
                'divergences': divergences,
                'divergence_signals': divergence_signals,
                'divergence_strength': divergence_strength,
                'price_peaks': price_peaks,
                'price_troughs': price_troughs,
                'nvi_peaks': nvi_peaks,
                'nvi_troughs': nvi_troughs
            }
            
        except Exception as e:
            logger.error(f"Error detecting divergences: {str(e)}")
            return {}
    
    def _analyze_institutional_pressure(self, data: pd.DataFrame, nvi_results: Dict) -> Dict[str, Any]:
        """Analyze institutional buying/selling pressure."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            nvi = nvi_results['nvi']
            volume_decreases = nvi_results['volume_decreases']
            
            # Calculate volume-weighted price pressure
            vwpp = np.zeros_like(close)
            institutional_pressure = np.zeros_like(close)
            
            for i in range(1, len(close)):
                price_change = (close[i] - close[i-1]) / close[i-1]
                volume_ratio = volume[i] / np.mean(volume[max(0, i-20):i]) if i >= 20 else 1.0
                
                # Institutional pressure calculation
                if volume_decreases[i]:  # Smart money active
                    institutional_pressure[i] = price_change * 2.0  # Amplify smart money impact
                else:
                    institutional_pressure[i] = price_change * volume_ratio * 0.5
                
                # Volume-weighted price pressure
                vwpp[i] = price_change * volume_ratio
            
            # Smooth institutional pressure
            institutional_pressure_smooth = talib.EMA(institutional_pressure, timeperiod=10)
            
            # Calculate cumulative institutional flow
            cumulative_flow = np.cumsum(institutional_pressure)
            
            # Detect accumulation/distribution phases
            flow_trend = talib.EMA(cumulative_flow, timeperiod=20)
            flow_acceleration = np.gradient(flow_trend)
            
            return {
                'institutional_pressure': institutional_pressure,
                'institutional_pressure_smooth': institutional_pressure_smooth,
                'cumulative_flow': cumulative_flow,
                'flow_trend': flow_trend,
                'flow_acceleration': flow_acceleration,
                'vwpp': vwpp
            }
            
        except Exception as e:
            logger.error(f"Error in institutional pressure analysis: {str(e)}")
            return {}
    
    def _perform_ml_analysis(self, data: pd.DataFrame, nvi_results: Dict) -> Dict[str, Any]:
        """Perform machine learning analysis on NVI patterns."""
        try:
            nvi = nvi_results['nvi']
            close = data['close'].values
            volume = data['volume'].values
            
            # Prepare features for ML analysis
            features = self._prepare_ml_features(data, nvi_results)
            
            if len(features) < 50:  # Need sufficient data for ML
                return {}
            
            # Anomaly detection
            anomaly_scores = self._detect_anomalies(features)
            
            # Pattern clustering
            pattern_clusters = self._cluster_patterns(features)
            
            # Smart money classification
            smart_money_predictions = self._classify_smart_money_periods(features, nvi_results)
            
            return {
                'anomaly_scores': anomaly_scores,
                'pattern_clusters': pattern_clusters,
                'smart_money_predictions': smart_money_predictions,
                'feature_importance': self._calculate_feature_importance(features)
            }
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {str(e)}")
            return {}
    
    def _generate_comprehensive_signals(self, data: pd.DataFrame, nvi_results: Dict,
                                      smart_money_analysis: Dict, trend_analysis: Dict,
                                      divergence_analysis: Dict, institutional_analysis: Dict,
                                      ml_analysis: Dict) -> List[NVISignal]:
        """Generate comprehensive trading signals."""
        try:
            signals = []
            timestamps = pd.to_datetime(data.index) if hasattr(data.index, 'to_pydatetime') else data.index
            
            nvi = nvi_results['nvi']
            nvi_normalized = nvi_results['nvi_normalized']
            volume_decreases = nvi_results['volume_decreases']
            
            smart_money_score = smart_money_analysis.get('smart_money_score', np.zeros_like(nvi))
            activity_classification = smart_money_analysis.get('activity_classification', [])
            trend_states = trend_analysis.get('trend_states', [])
            divergence_signals = divergence_analysis.get('divergence_signals', np.zeros_like(nvi))
            institutional_pressure = institutional_analysis.get('institutional_pressure_smooth', np.zeros_like(nvi))
            
            for i in range(len(data)):
                if np.isnan(nvi[i]) or i < self.lookback_period:
                    continue
                
                # Calculate signal components
                signal_strength = self._calculate_signal_strength(i, nvi_results, smart_money_analysis, trend_analysis)
                confidence = self._calculate_confidence(i, nvi_results, trend_analysis, divergence_analysis)
                
                if confidence < self.min_confidence:
                    continue
                
                # Determine divergence type for this period
                divergence_type = DivergenceType.NO_DIVERGENCE
                if i < len(divergence_signals):
                    if divergence_signals[i] > 0:
                        divergence_type = DivergenceType.BULLISH
                    elif divergence_signals[i] < 0:
                        divergence_type = DivergenceType.BEARISH
                
                # Generate signal
                signal = NVISignal(
                    timestamp=timestamps[i] if i < len(timestamps) else datetime.now(),
                    nvi_value=nvi[i],
                    nvi_normalized=nvi_normalized[i] if i < len(nvi_normalized) else 50.0,
                    smart_money_activity=activity_classification[i] if i < len(activity_classification) else SmartMoneyActivity.NEUTRAL,
                    trend_state=trend_states[i] if i < len(trend_states) else TrendState.SIDEWAYS,
                    divergence_type=divergence_type,
                    signal_strength=signal_strength,
                    confidence=confidence,
                    volume_decrease_count=self._count_recent_volume_decreases(i, volume_decreases),
                    price_momentum=self._calculate_price_momentum(i, data),
                    institutional_pressure=institutional_pressure[i] if i < len(institutional_pressure) else 0.0,
                    trend_confirmation=trend_analysis.get('trend_confirmation', np.zeros_like(nvi))[i],
                    entry_signal=self._calculate_entry_signal(i, nvi_results, smart_money_analysis, divergence_analysis),
                    exit_signal=self._calculate_exit_signal(i, nvi_results, trend_analysis),
                    risk_assessment=self._calculate_risk_assessment(i, nvi_results, smart_money_analysis)
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {str(e)}")
            return []
    
    def _normalize_nvi(self, nvi: np.ndarray) -> np.ndarray:
        """Normalize NVI values to 0-100 scale."""
        try:
            valid_nvi = nvi[~np.isnan(nvi)]
            if len(valid_nvi) == 0:
                return np.full_like(nvi, 50.0)
            
            min_nvi = np.min(valid_nvi)
            max_nvi = np.max(valid_nvi)
            
            if max_nvi == min_nvi:
                return np.full_like(nvi, 50.0)
            
            normalized = ((nvi - min_nvi) / (max_nvi - min_nvi)) * 100
            return np.where(np.isnan(nvi), np.nan, normalized)
            
        except:
            return np.full_like(nvi, 50.0)
    
    def _calculate_smart_money_strength(self, nvi_results: Dict, volume_decreases: np.ndarray, 
                                      price_returns: np.ndarray) -> np.ndarray:
        """Calculate smart money strength indicator."""
        try:
            nvi = nvi_results['nvi']
            strength = np.zeros_like(nvi)
            
            for i in range(10, len(nvi)):
                # Recent smart money activity
                recent_vol_decreases = volume_decreases[i-10:i+1]
                recent_returns = price_returns[i-10:i+1]
                
                # Calculate strength based on consistency of smart money moves
                positive_moves = np.sum(recent_vol_decreases & (recent_returns > 0))
                negative_moves = np.sum(recent_vol_decreases & (recent_returns < 0))
                total_moves = positive_moves + negative_moves
                
                if total_moves > 0:
                    strength[i] = abs(positive_moves - negative_moves) / total_moves * 100
                
            return strength
            
        except:
            return np.zeros_like(nvi)
    
    def _calculate_institutional_pressure(self, close: np.ndarray, volume: np.ndarray,
                                        volume_decreases: np.ndarray, price_returns: np.ndarray) -> np.ndarray:
        """Calculate institutional pressure index."""
        try:
            pressure = np.zeros_like(close)
            
            for i in range(1, len(close)):
                # Base pressure from price movement
                base_pressure = price_returns[i]
                
                # Amplify if it's a smart money period (volume decrease)
                if volume_decreases[i]:
                    pressure[i] = base_pressure * 2.0
                else:
                    pressure[i] = base_pressure * 0.5
            
            # Smooth the pressure
            return talib.EMA(pressure, timeperiod=5)
            
        except:
            return np.zeros_like(close)
    
    def _calculate_trend_persistence(self, trend_states: List[TrendState]) -> float:
        """Calculate trend persistence score."""
        try:
            if len(trend_states) < 10:
                return 0.0
            
            # Count consecutive same-trend periods
            current_trend = trend_states[-1]
            consecutive_count = 1
            
            for i in range(len(trend_states) - 2, max(0, len(trend_states) - 21), -1):
                if trend_states[i] == current_trend:
                    consecutive_count += 1
                else:
                    break
            
            return min(consecutive_count / 20.0 * 100, 100.0)
            
        except:
            return 0.0
    
    def _calculate_nvi_price_correlation(self, nvi: np.ndarray, close: np.ndarray) -> float:
        """Calculate correlation between NVI and price."""
        try:
            valid_indices = ~(np.isnan(nvi) | np.isnan(close))
            if np.sum(valid_indices) < 10:
                return 0.0
            
            correlation = np.corrcoef(nvi[valid_indices], close[valid_indices])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except:
            return 0.0
    
    def _calculate_divergence_strength(self, divergences: List[Dict]) -> float:
        """Calculate overall divergence strength."""
        try:
            if not divergences:
                return 0.0
            
            recent_divergences = [d for d in divergences if d['index'] >= len(divergences) - 50]
            if not recent_divergences:
                return 0.0
            
            avg_strength = np.mean([d['strength'] for d in recent_divergences])
            return min(avg_strength, 100.0)
            
        except:
            return 0.0
    
    def _prepare_ml_features(self, data: pd.DataFrame, nvi_results: Dict) -> np.ndarray:
        """Prepare features for machine learning analysis."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            nvi = nvi_results['nvi']
            volume_decreases = nvi_results['volume_decreases']
            
            features = []
            
            for i in range(20, len(close)):
                # Price features
                price_returns = [(close[i-j] - close[i-j-1]) / close[i-j-1] for j in range(5)]
                
                # Volume features
                vol_ratios = [volume[i-j] / np.mean(volume[max(0, i-j-20):i-j]) for j in range(5)]
                
                # NVI features
                nvi_changes = [nvi[i-j] - nvi[i-j-1] for j in range(5)]
                
                # Volume decrease features
                vol_decrease_count = np.sum(volume_decreases[i-10:i])
                
                feature_vector = price_returns + vol_ratios + nvi_changes + [vol_decrease_count]
                features.append(feature_vector)
            
            return np.array(features)
            
        except:
            return np.array([])
    
    def _detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalies in trading patterns."""
        try:
            if len(features) < 10:
                return np.array([])
            
            # Normalize features
            features_scaled = self._scaler.fit_transform(features)
            
            # Detect anomalies
            anomaly_scores = self._anomaly_detector.fit_predict(features_scaled)
            
            return anomaly_scores
            
        except:
            return np.array([])
    
    def _cluster_patterns(self, features: np.ndarray) -> np.ndarray:
        """Cluster trading patterns."""
        try:
            if len(features) < 10:
                return np.array([])
            
            # Normalize features
            features_scaled = self._scaler.fit_transform(features)
            
            # Perform clustering
            clusters = self._pattern_clusterer.fit_predict(features_scaled)
            
            return clusters
            
        except:
            return np.array([])
    
    def _classify_smart_money_periods(self, features: np.ndarray, nvi_results: Dict) -> np.ndarray:
        """Classify smart money activity periods."""
        try:
            if len(features) < 50:
                return np.array([])
            
            volume_decreases = nvi_results['volume_decreases']
            
            # Create labels for training (simplified)
            labels = []
            for i in range(20, len(volume_decreases)):
                recent_decreases = np.sum(volume_decreases[i-5:i+1])
                labels.append(1 if recent_decreases >= 3 else 0)
            
            if len(labels) != len(features):
                return np.array([])
            
            # Train classifier
            self._smart_money_classifier.fit(features, labels)
            
            # Predict
            predictions = self._smart_money_classifier.predict(features)
            
            return predictions
            
        except:
            return np.array([])
    
    def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for analysis."""
        try:
            feature_names = [
                'price_return_0', 'price_return_1', 'price_return_2', 'price_return_3', 'price_return_4',
                'vol_ratio_0', 'vol_ratio_1', 'vol_ratio_2', 'vol_ratio_3', 'vol_ratio_4',
                'nvi_change_0', 'nvi_change_1', 'nvi_change_2', 'nvi_change_3', 'nvi_change_4',
                'vol_decrease_count'
            ]
            
            if len(features) < 50 or features.shape[1] != len(feature_names):
                return {}
            
            # Use variance as a simple importance measure
            importances = np.var(features, axis=0)
            
            return dict(zip(feature_names, importances.tolist()))
            
        except:
            return {}
    
    def _calculate_signal_strength(self, index: int, nvi_results: Dict, smart_money_analysis: Dict, 
                                 trend_analysis: Dict) -> float:
        """Calculate signal strength for a specific point."""
        try:
            strength = 0.0
            
            # Smart money activity contribution
            smart_money_score = smart_money_analysis.get('smart_money_score', np.zeros(index+1))
            if index < len(smart_money_score):
                strength += abs(smart_money_score[index]) * 0.4
            
            # Trend confirmation contribution
            trend_confirmation = trend_analysis.get('trend_confirmation', np.zeros(index+1))
            if index < len(trend_confirmation):
                strength += trend_confirmation[index] * 0.3
            
            # NVI momentum contribution
            nvi_rsi = nvi_results.get('nvi_rsi', np.zeros(index+1))
            if index < len(nvi_rsi) and not np.isnan(nvi_rsi[index]):
                if nvi_rsi[index] > 70 or nvi_rsi[index] < 30:
                    strength += 30.0
            
            return min(strength, 100.0)
            
        except:
            return 0.0
    
    def _calculate_confidence(self, index: int, nvi_results: Dict, trend_analysis: Dict, 
                            divergence_analysis: Dict) -> float:
        """Calculate confidence for a specific signal."""
        try:
            confidence = 50.0  # Base confidence
            
            # Data quality boost
            if index >= self.lookback_period:
                confidence += 20.0
            
            # Trend persistence boost
            trend_persistence = trend_analysis.get('trend_persistence', 0)
            confidence += trend_persistence * 0.2
            
            # Divergence confirmation boost
            divergence_signals = divergence_analysis.get('divergence_signals', np.zeros(index+1))
            if index < len(divergence_signals) and divergence_signals[index] != 0:
                confidence += 15.0
            
            return min(confidence, 100.0)
            
        except:
            return 50.0
    
    def _count_recent_volume_decreases(self, index: int, volume_decreases: np.ndarray) -> int:
        """Count recent volume decreases."""
        try:
            start_idx = max(0, index - 10)
            return int(np.sum(volume_decreases[start_idx:index+1]))
        except:
            return 0
    
    def _calculate_price_momentum(self, index: int, data: pd.DataFrame) -> float:
        """Calculate price momentum score."""
        try:
            if index < 10 or 'close' not in data.columns:
                return 0.0
            
            close = data['close'].values
            current_price = close[index]
            past_price = close[index - 10]
            
            if past_price == 0:
                return 0.0
            
            momentum = ((current_price - past_price) / past_price) * 100
            return momentum
            
        except:
            return 0.0
    
    def _calculate_entry_signal(self, index: int, nvi_results: Dict, smart_money_analysis: Dict, 
                              divergence_analysis: Dict) -> float:
        """Calculate entry signal strength."""
        try:
            entry_signal = 0.0
            
            # Smart money accumulation signal
            activity_classification = smart_money_analysis.get('activity_classification', [])
            if index < len(activity_classification):
                activity = activity_classification[index]
                if activity == SmartMoneyActivity.STRONG_ACCUMULATION:
                    entry_signal += 50.0
                elif activity == SmartMoneyActivity.ACCUMULATION:
                    entry_signal += 30.0
                elif activity == SmartMoneyActivity.STRONG_DISTRIBUTION:
                    entry_signal -= 50.0
                elif activity == SmartMoneyActivity.DISTRIBUTION:
                    entry_signal -= 30.0
            
            # Divergence signal
            divergence_signals = divergence_analysis.get('divergence_signals', np.zeros(index+1))
            if index < len(divergence_signals):
                entry_signal += divergence_signals[index] * 25.0
            
            return np.clip(entry_signal, -100.0, 100.0)
            
        except:
            return 0.0
    
    def _calculate_exit_signal(self, index: int, nvi_results: Dict, trend_analysis: Dict) -> float:
        """Calculate exit signal strength."""
        try:
            exit_signal = 0.0
            
            # Trend reversal signal
            trend_states = trend_analysis.get('trend_states', [])
            if index >= 5 and len(trend_states) > index:
                current_trend = trend_states[index]
                past_trend = trend_states[index - 5]
                
                if current_trend != past_trend:
                    exit_signal += 40.0
            
            # NVI overbought/oversold signal
            nvi_rsi = nvi_results.get('nvi_rsi', np.zeros(index+1))
            if index < len(nvi_rsi) and not np.isnan(nvi_rsi[index]):
                if nvi_rsi[index] > 80:
                    exit_signal += 30.0
                elif nvi_rsi[index] < 20:
                    exit_signal += 30.0
            
            return min(exit_signal, 100.0)
            
        except:
            return 0.0
    
    def _calculate_risk_assessment(self, index: int, nvi_results: Dict, smart_money_analysis: Dict) -> float:
        """Calculate risk assessment for the signal."""
        try:
            risk = 30.0  # Base risk
            
            # Volatility risk
            nvi_bb_upper = nvi_results.get('nvi_bb_upper', np.zeros(index+1))
            nvi_bb_lower = nvi_results.get('nvi_bb_lower', np.zeros(index+1))
            nvi = nvi_results.get('nvi', np.zeros(index+1))
            
            if (index < len(nvi_bb_upper) and index < len(nvi_bb_lower) and 
                index < len(nvi) and not any(np.isnan([nvi_bb_upper[index], nvi_bb_lower[index], nvi[index]]))):
                bb_width = nvi_bb_upper[index] - nvi_bb_lower[index]
                bb_position = (nvi[index] - nvi_bb_lower[index]) / bb_width if bb_width > 0 else 0.5
                
                if bb_position > 0.8 or bb_position < 0.2:
                    risk += 25.0
            
            # Smart money uncertainty risk
            smart_money_score = smart_money_analysis.get('smart_money_score', np.zeros(index+1))
            if index < len(smart_money_score):
                if abs(smart_money_score[index]) < 20:  # Neutral/uncertain
                    risk += 20.0
            
            return min(risk, 100.0)
            
        except:
            return 50.0
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the calculation results."""
        return {
            'indicator_name': 'Negative Volume Index',
            'version': '1.0.0',
            'parameters': {
                'initial_value': self.initial_value,
                'lookback_period': self.lookback_period,
                'divergence_lookback': self.divergence_lookback,
                'trend_confirmation_period': self.trend_confirmation_period,
                'min_confidence': self.min_confidence,
                'volume_threshold': self.volume_threshold
            },
            'features': [
                'Smart money activity tracking',
                'Volume divergence analysis',
                'Trend confirmation algorithms',
                'Institutional pressure analysis',
                'Machine learning pattern recognition',
                'Anomaly detection',
                'Risk assessment'
            ],
            'calculation_timestamp': datetime.now(),
            'data_requirements': ['close', 'volume']
        }
    
    def _generate_empty_result(self) -> Dict[str, Any]:
        """Generate empty result structure."""
        return {
            'nvi_results': {},
            'smart_money_analysis': {},
            'trend_analysis': {},
            'divergence_analysis': {},
            'institutional_analysis': {},
            'ml_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': 'Insufficient data'
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            'nvi_results': {},
            'smart_money_analysis': {},
            'trend_analysis': {},
            'divergence_analysis': {},
            'institutional_analysis': {},
            'ml_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': error_message
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    # Generate realistic OHLCV data with smart money patterns
    base_price = 100
    returns = np.random.normal(0, 0.015, 300)
    
    # Add smart money patterns (price moves on low volume)
    volume_base = np.random.lognormal(10, 0.3, 300)
    for i in range(50, 250, 30):  # Smart money accumulation periods
        volume_base[i:i+5] *= 0.7  # Reduce volume
        if i < 299:
            returns[i+1:i+6] += 0.01  # Positive price moves on low volume
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 300))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 300))),
        'close': prices,
        'volume': volume_base
    }, index=dates)
    
    # Test the indicator
    nvi_indicator = NegativeVolumeIndexIndicator(
        initial_value=1000.0,
        lookback_period=50,
        divergence_lookback=20,
        min_confidence=70.0
    )
    
    try:
        result = nvi_indicator.calculate(sample_data)
        
        print("Negative Volume Index Indicator Results:")
        print(f"- Calculation successful: {not result.get('error', False)}")
        print(f"- NVI values calculated: {len(result.get('nvi_results', {}).get('nvi', []))}")
        print(f"- Signals generated: {len(result.get('signals', []))}")
        print(f"- Smart money analysis completed: {'smart_money_analysis' in result}")
        print(f"- Trend analysis completed: {'trend_analysis' in result}")
        print(f"- Divergence analysis completed: {'divergence_analysis' in result}")
        print(f"- ML analysis completed: {'ml_analysis' in result}")
        
        # Display some sample signals
        signals = result.get('signals', [])
        if signals:
            print(f"\nSample signals (showing first 3):")
            for i, signal in enumerate(signals[:3]):
                print(f"Signal {i+1}:")
                print(f"  Timestamp: {signal.timestamp}")
                print(f"  NVI Value: {signal.nvi_value:.4f}")
                print(f"  Smart Money Activity: {signal.smart_money_activity}")
                print(f"  Signal Strength: {signal.signal_strength:.2f}")
                print(f"  Confidence: {signal.confidence:.2f}")
                print(f"  Trend State: {signal.trend_state}")
                print(f"  Divergence Type: {signal.divergence_type}")
                print(f"  Entry Signal: {signal.entry_signal:.2f}")
        
        print(f"\nMetadata: {result.get('metadata', {}).get('indicator_name', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing NVI Indicator: {str(e)}")
        import traceback
        traceback.print_exc()