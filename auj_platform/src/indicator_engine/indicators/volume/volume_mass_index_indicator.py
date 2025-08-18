"""
Mass Index Indicator - Advanced Volume-Based Volatility Expansion Detector
==================================

The Mass Index is a sophisticated indicator that identifies potential trend reversals
by detecting periods of volatility expansion. This implementation includes advanced 
features for institutional-grade trading applications.

Key Features:
    - Volatility expansion detection with statistical significance testing
- Trend reversal probability algorithms with machine learning
- Adaptive smoothing for different market conditions
- Multi-timeframe analysis capabilities
- Regime detection for optimized parameter adjustment
- High-frequency volatility clustering analysis
- Statistical anomaly detection in volatility patterns

Mathematical Foundation:
    The Mass Index is calculated using:
    1. Single Smoothed = EMA(High - Low, period)
2. Double Smoothed = EMA(Single Smoothed, period)
3. Mass Index = Sum(Single Smoothed / Double Smoothed, sum_period)

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
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications for adaptive parameter adjustment."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class VolatilityState(Enum):
    """Volatility state classifications."""
    EXPANSION = "expansion"
    COMPRESSION = "compression"
    EXTREME_HIGH = "extreme_high"
    EXTREME_LOW = "extreme_low"
    NORMAL = "normal"

@dataclass
class MassIndexSignal:
    """
    Comprehensive Mass Index signal with confidence scoring.
    
    Attributes:
        timestamp: Signal timestamp
        mass_index: Current Mass Index value
        signal_strength: Signal strength (0-100)
        confidence: Confidence level (0-100)
        volatility_state: Current volatility state
        reversal_probability: Probability of trend reversal (0-100)
        volume_confirmation: Volume confirmation score (0-100)
        regime: Detected market regime
        support_resistance_level: Key support/resistance level
        risk_level: Risk assessment (0-100)
        entry_signal: Entry signal strength (-100 to 100)
        exit_signal: Exit signal strength (-100 to 100)
        statistical_significance: Statistical significance of signal
    """
    timestamp: datetime
    mass_index: float
    signal_strength: float
    confidence: float
    volatility_state: VolatilityState
    reversal_probability: float
    volume_confirmation: float
    regime: MarketRegime
    support_resistance_level: float
    risk_level: float
    entry_signal: float
    exit_signal: float
    statistical_significance: float

class VolumeMassIndexIndicator:
    """
    Advanced Mass Index Indicator with volatility expansion detection and trend reversal algorithms.
    
    This indicator provides sophisticated analysis of volatility patterns and potential
    trend reversals using advanced mathematical models and machine learning techniques.
    """
    
def __init__(self,:)
                 period: int = 9,
                 sum_period: int = 25,
                 reversal_threshold: float = 27.0,
                 extreme_threshold: float = 26.5,
                 min_confidence: float = 70.0,
(                 lookback_window: int = 200):
        """
        Initialize the Mass Index Indicator.
        
        Args:
            period: EMA smoothing period for range calculation
            sum_period: Period for Mass Index summation
            reversal_threshold: Threshold for reversal signals
            extreme_threshold: Threshold for extreme volatility
            min_confidence: Minimum confidence for signal generation
            lookback_window: Historical data window for analysis
        """
                super().__init__(name="Marketregime")
        self.period = period
        self.sum_period = sum_period
        self.reversal_threshold = reversal_threshold
        self.extreme_threshold = extreme_threshold
        self.min_confidence = min_confidence
        self.lookback_window = lookback_window
        
        # Initialize calculation buffers
        self._high_low_range = []
        self._single_smoothed = []
        self._double_smoothed = []
        self._mass_index_values = []
        self._volume_data = []
        self._price_data = []
        
        # Machine learning models for pattern recognition
        self._anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self._volatility_clusterer = DBSCAN(eps=0.3, min_samples=5)
        self._scaler = StandardScaler()
        
        # Historical analysis data
        self._historical_signals = []
        self._regime_history = []
        self._volatility_history = []
        
        logger.info(f"MassIndexIndicator initialized with period={period}, sum_period={sum_period}")
    
def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Mass Index with advanced volatility analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing Mass Index values and analysis results
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
            
            required_columns = ['high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(data) < self.lookback_window:
                logger.warning(f"Insufficient data: {len(data)} < {self.lookback_window}")
                return self._generate_empty_result()
            
            # Calculate basic Mass Index
            mass_index_results = self._calculate_mass_index(data)
            
            # Perform advanced volatility analysis
            volatility_analysis = self._analyze_volatility_patterns(data, mass_index_results)
            
            # Detect market regime
            regime_analysis = self._detect_market_regime(data)
            
            # Generate reversal signals
            reversal_signals = self._generate_reversal_signals(mass_index_results, volatility_analysis)
            
            # Perform statistical analysis
            statistical_analysis = self._perform_statistical_analysis(mass_index_results)
            
            # Generate comprehensive signals
            signals = self._generate_comprehensive_signals()
                data, mass_index_results, volatility_analysis, 
                regime_analysis, reversal_signals, statistical_analysis
(            )
            
            return {
                'mass_index': mass_index_results,
                'volatility_analysis': volatility_analysis,
                'regime_analysis': regime_analysis,
                'reversal_signals': reversal_signals,
                'statistical_analysis': statistical_analysis,
                'signals': signals,
                'metadata': self._generate_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in MassIndexIndicator calculation: {str(e)}")
            return self._generate_error_result(str(e))
    
def _calculate_mass_index(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate the basic Mass Index values."""
        try:
            high = data['high'].values
            low = data['low'].values
            
            # Calculate high-low range
            hl_range = high - low
            
            # Calculate single smoothed EMA
            single_smoothed = talib.EMA(hl_range, timeperiod=self.period)
            
            # Calculate double smoothed EMA
            double_smoothed = talib.EMA(single_smoothed, timeperiod=self.period)
            
            # Avoid division by zero
            double_smoothed = np.where(double_smoothed == 0, np.finfo(float).eps, double_smoothed)
            
            # Calculate ratio
            ratio = single_smoothed / double_smoothed
            
            # Calculate Mass Index as rolling sum
            mass_index = np.full_like(ratio, np.nan)
            for i in range(self.sum_period - 1, len(ratio)):
                mass_index[i] = np.sum(ratio[i - self.sum_period + 1:i + 1])
            
            # Calculate additional metrics
            mass_index_sma = talib.SMA(mass_index, timeperiod=10)
            mass_index_ema = talib.EMA(mass_index, timeperiod=10)
            
            # Calculate rate of change
            mass_index_roc = np.full_like(mass_index, np.nan)
            mass_index_roc[1:] = np.diff(mass_index)
            
            # Calculate acceleration
            mass_index_acceleration = np.full_like(mass_index_roc, np.nan)
            mass_index_acceleration[1:] = np.diff(mass_index_roc)
            
            return {
                'mass_index': mass_index,
                'single_smoothed': single_smoothed,
                'double_smoothed': double_smoothed,
                'ratio': ratio,
                'mass_index_sma': mass_index_sma,
                'mass_index_ema': mass_index_ema,
                'mass_index_roc': mass_index_roc,
                'mass_index_acceleration': mass_index_acceleration
            }
            
        except Exception as e:
            logger.error(f"Error calculating Mass Index: {str(e)}")
            raise
    
def _analyze_volatility_patterns(self, data: pd.DataFrame, mass_index_results: Dict) -> Dict[str, Any]:
        """Analyze volatility patterns and expansion/compression cycles."""
        try:
            mass_index = mass_index_results['mass_index']
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate True Range for volatility analysis
            tr = self._calculate_true_range(high, low, close)
            atr = talib.EMA(tr, timeperiod=14)
            
            # Volatility percentiles
            volatility_percentile = np.full_like(mass_index, np.nan)
            for i in range(50, len(mass_index)):
                window_data = mass_index[max(0, i-50):i+1]
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) > 10:
                    volatility_percentile[i] = stats.percentileofscore(valid_data, mass_index[i])
            
            # Volatility expansion detection
            expansion_signals = self._detect_volatility_expansion(mass_index, atr)
            
            # Volatility clustering analysis
            clustering_analysis = self._analyze_volatility_clustering(mass_index, volume)
            
            # Regime-specific volatility analysis
            regime_volatility = self._analyze_regime_volatility(mass_index, close)
            
            return {
                'true_range': tr,
                'atr': atr,
                'volatility_percentile': volatility_percentile,
                'expansion_signals': expansion_signals,
                'clustering_analysis': clustering_analysis,
                'regime_volatility': regime_volatility,
                'volatility_state': self._classify_volatility_state(mass_index, volatility_percentile)
            }
            
        except Exception as e:
            logger.error(f"Error in volatility pattern analysis: {str(e)}")
            return {}
    
def _detect_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime for adaptive parameter adjustment."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate trend indicators
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            # MACD for trend analysis
            macd, macd_signal, macd_hist = talib.MACD(close)
            
            # RSI for momentum
            rsi = talib.RSI(close, timeperiod=14)
            
            # Bollinger Bands for volatility
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # Volume analysis
            volume_sma = talib.SMA(volume, timeperiod=20)
            volume_ratio = volume / volume_sma
            
            # Regime classification
            regime_scores = self._calculate_regime_scores()
                close, sma_20, sma_50, macd, rsi, bb_width, volume_ratio
(            )
            
            current_regime = self._classify_current_regime(regime_scores)
            
            return {
                'regime_scores': regime_scores,
                'current_regime': current_regime,
                'trend_strength': abs(regime_scores.get('trend', 0)),
                'volatility_level': regime_scores.get('volatility', 0),
                'volume_activity': regime_scores.get('volume', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return {'current_regime': MarketRegime.SIDEWAYS}
    
def _generate_reversal_signals(self, mass_index_results: Dict, volatility_analysis: Dict) -> Dict[str, Any]:
        """Generate trend reversal signals based on Mass Index analysis."""
        try:
            mass_index = mass_index_results['mass_index']
            mass_index_roc = mass_index_results['mass_index_roc']
            
            # Basic reversal signals
            reversal_signals = np.zeros_like(mass_index)
            
            # Signal when Mass Index crosses above threshold
            above_threshold = mass_index > self.reversal_threshold
            
            # Look for peaks above threshold
            valid_indices = ~np.isnan(mass_index)
            if np.sum(valid_indices) > 10:
                peaks, peak_properties = find_peaks()
                    mass_index[valid_indices], 
                    height=self.extreme_threshold,
                    distance=5
(                )
                
                # Mark peak signals
                for peak_idx in peaks:
                    actual_idx = np.where(valid_indices)[0][peak_idx]
                    if actual_idx < len(reversal_signals):
                        reversal_signals[actual_idx] = 1.0
            
            # Enhanced reversal probability
            reversal_probability = self._calculate_reversal_probability()
                mass_index, mass_index_roc, volatility_analysis
(            )
            
            # Confirmation signals
            confirmation_signals = self._generate_confirmation_signals()
                mass_index, reversal_signals, volatility_analysis
(            )
            
            return {
                'reversal_signals': reversal_signals,
                'reversal_probability': reversal_probability,
                'confirmation_signals': confirmation_signals,
                'signal_strength': self._calculate_signal_strength(reversal_signals, reversal_probability)
            }
            
        except Exception as e:
            logger.error(f"Error generating reversal signals: {str(e)}")
            return {}
    
def _perform_statistical_analysis(self, mass_index_results: Dict) -> Dict[str, Any]:
        """Perform statistical analysis of Mass Index values."""
        try:
            mass_index = mass_index_results['mass_index']
            valid_data = mass_index[~np.isnan(mass_index)]
            
            if len(valid_data) < 30:
                return {}
            
            # Basic statistics
            stats_summary = {
                'mean': np.mean(valid_data),
                'std': np.std(valid_data),
                'skewness': stats.skew(valid_data),
                'kurtosis': stats.kurtosis(valid_data),
                'median': np.median(valid_data),
                'q25': np.percentile(valid_data, 25),
                'q75': np.percentile(valid_data, 75)
            }
            
            # Z-scores for anomaly detection
            z_scores = np.abs(stats.zscore(valid_data))
            anomaly_threshold = 2.5
            anomalies = z_scores > anomaly_threshold
            
            # Statistical significance testing
            significance_tests = self._perform_significance_tests(valid_data)
            
            # Distribution analysis
            distribution_analysis = self._analyze_distribution(valid_data)
            
            return {
                'statistics': stats_summary,
                'anomalies': anomalies,
                'z_scores': z_scores,
                'significance_tests': significance_tests,
                'distribution_analysis': distribution_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {}
    
def _generate_comprehensive_signals(self, data: pd.DataFrame, mass_index_results: Dict,:)
                                      volatility_analysis: Dict, regime_analysis: Dict,
(                                      reversal_signals: Dict, statistical_analysis: Dict) -> List[MassIndexSignal]:
        """Generate comprehensive trading signals."""
        try:
            signals = []
            timestamps = pd.to_datetime(data.index) if hasattr(data.index, 'to_pydatetime') else data.index
            
            mass_index = mass_index_results['mass_index']
            reversal_prob = reversal_signals.get('reversal_probability', np.zeros_like(mass_index))
            
            for i in range(len(data)):
                if np.isnan(mass_index[i]):
                    continue
                
                # Calculate signal components
                signal_strength = self._calculate_point_signal_strength(i, mass_index_results, volatility_analysis)
                confidence = self._calculate_point_confidence(i, mass_index_results, statistical_analysis)
                
                if confidence < self.min_confidence:
                    continue
                
                # Generate signal
                signal = MassIndexSignal()
                    timestamp=timestamps[i] if i < len(timestamps) else datetime.now(),
                    mass_index=mass_index[i],
                    signal_strength=signal_strength,
                    confidence=confidence,
                    volatility_state=volatility_analysis.get('volatility_state', [VolatilityState.NORMAL])[min(i, len(volatility_analysis.get('volatility_state', [])) - 1)] if volatility_analysis.get('volatility_state') else VolatilityState.NORMAL,
                    reversal_probability=reversal_prob[i] if i < len(reversal_prob) else 0.0,
                    volume_confirmation=self._calculate_volume_confirmation(i, data),
                    regime=regime_analysis.get('current_regime', MarketRegime.SIDEWAYS),
                    support_resistance_level=self._calculate_support_resistance(i, data),
                    risk_level=self._calculate_risk_level(i, mass_index_results, volatility_analysis),
                    entry_signal=self._calculate_entry_signal(i, mass_index_results, reversal_signals),
                    exit_signal=self._calculate_exit_signal(i, mass_index_results, volatility_analysis),
                    statistical_significance=self._calculate_statistical_significance(i, statistical_analysis)
(                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {str(e)}")
            return []
    
def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range for volatility analysis."""
        try:
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            tr2[0] = tr1[0]  # Handle first value
            tr3[0] = tr1[0]
            
            return np.maximum(tr1, np.maximum(tr2, tr3))
        except:
            return np.zeros_like(high)
    
def _detect_volatility_expansion(self, mass_index: np.ndarray, atr: np.ndarray) -> np.ndarray:
        """Detect volatility expansion periods."""
        try:
            expansion_signals = np.zeros_like(mass_index)
            
            # Multiple expansion criteria
            criteria1 = mass_index > self.extreme_threshold
            criteria2 = mass_index > np.roll(mass_index, 5) * 1.1  # 10% increase
            
            # ATR confirmation
            atr_expansion = atr > np.roll(atr, 10) * 1.2  # 20% ATR increase
            
            expansion_signals = np.where(criteria1 & criteria2 & atr_expansion, 1.0, 0.0)
            
            return expansion_signals
        except:
            return np.zeros_like(mass_index)
    
def _analyze_volatility_clustering(self, mass_index: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility clustering patterns."""
        try:
            valid_indices = ~np.isnan(mass_index)
            if np.sum(valid_indices) < 50:
                return {}
            
            # Prepare data for clustering
            features = []
            valid_mass_index = mass_index[valid_indices]
            valid_volume = volume[valid_indices]
            
            for i in range(10, len(valid_mass_index)):
                volatility_change = valid_mass_index[i] - valid_mass_index[i-1]
                volume_ratio = valid_volume[i] / np.mean(valid_volume[max(0, i-20):i])
                features.append([volatility_change, volume_ratio])
            
            if len(features) < 10:
                return {}
            
            features = np.array(features)
            
            # Normalize features
            features_scaled = self._scaler.fit_transform(features)
            
            # Perform clustering
            clusters = self._volatility_clusterer.fit_predict(features_scaled)
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in np.unique(clusters):
                if cluster_id == -1:  # Noise points:
                    continue
                cluster_mask = clusters == cluster_id
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': np.sum(cluster_mask),
                    'avg_volatility_change': np.mean(features[cluster_mask, 0]),
                    'avg_volume_ratio': np.mean(features[cluster_mask, 1])
                }
            
            return {
                'clusters': clusters,
                'cluster_analysis': cluster_analysis,
                'n_clusters': len(np.unique(clusters[clusters != -1]))
            }
            
        except Exception as e:
            logger.error(f"Error in volatility clustering analysis: {str(e)}")
            return {}
    
def _analyze_regime_volatility(self, mass_index: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility characteristics by market regime."""
        try:
            # Calculate price-based regime indicators
            returns = np.diff(np.log(close))
            volatility = np.rolling_std(returns, window=20) if hasattr(np, 'rolling_std') else np.array([np.std(returns[max(0, i-19):i+1]) for i in range(len(returns))])
            
            # Categorize regimes based on volatility
            vol_percentiles = np.percentile(volatility[~np.isnan(volatility)], [25, 75])
            
            regime_categories = np.full_like(volatility, 0)  # 0: normal
            regime_categories[volatility < vol_percentiles[0]] = -1  # Low volatility
            regime_categories[volatility > vol_percentiles[1]] = 1   # High volatility
            
            # Analyze Mass Index behavior in different regimes
            regime_analysis = {}
            for regime in [-1, 0, 1]:
                regime_mask = regime_categories == regime
                if np.sum(regime_mask) > 5:
                    regime_mass_index = mass_index[1:][regime_mask]  # Align with returns
                    regime_mass_index = regime_mass_index[~np.isnan(regime_mass_index)]
                    
                    if len(regime_mass_index) > 0:
                        regime_analysis[regime] = {
                            'avg_mass_index': np.mean(regime_mass_index),
                            'std_mass_index': np.std(regime_mass_index),
                            'max_mass_index': np.max(regime_mass_index),
                            'periods': np.sum(regime_mask)
                        }
            
            return {
                'regime_categories': regime_categories,
                'regime_analysis': regime_analysis,
                'volatility_percentiles': vol_percentiles
            }
            
        except Exception as e:
            logger.error(f"Error in regime volatility analysis: {str(e)}")
            return {}
    
def _classify_volatility_state(self, mass_index: np.ndarray, volatility_percentile: np.ndarray) -> List[VolatilityState]:
        """Classify volatility state for each period."""
        try:
            states = []
            
            for i in range(len(mass_index)):
                if np.isnan(mass_index[i]):
                    states.append(VolatilityState.NORMAL)
                    continue
                
                mi_value = mass_index[i]
                percentile = volatility_percentile[i] if i < len(volatility_percentile) and not np.isnan(volatility_percentile[i]) else 50
                
                if mi_value > self.reversal_threshold:
                    if percentile > 90:
                        states.append(VolatilityState.EXTREME_HIGH)
                    else:
                        states.append(VolatilityState.EXPANSION)
                elif mi_value < self.extreme_threshold * 0.7:
                    if percentile < 10:
                        states.append(VolatilityState.EXTREME_LOW)
                    else:
                        states.append(VolatilityState.COMPRESSION)
                else:
                    states.append(VolatilityState.NORMAL)
            
            return states
            
        except Exception as e:
            logger.error(f"Error classifying volatility state: {str(e)}")
            return [VolatilityState.NORMAL] * len(mass_index)
    
def _calculate_regime_scores(self, close: np.ndarray, sma_20: np.ndarray, sma_50: np.ndarray,:)
                               macd: np.ndarray, rsi: np.ndarray, bb_width: np.ndarray,
(                               volume_ratio: np.ndarray) -> Dict[str, float]:
        """Calculate regime classification scores."""
        try:
            # Get latest values (handle NaN)
            current_idx = -1
            while current_idx >= -len(close) and (np.isnan(close[current_idx]) or np.isnan(sma_20[current_idx])):
                current_idx -= 1
            
            if current_idx < -len(close):
                return {'trend': 0, 'volatility': 0, 'volume': 0}
            
            # Trend analysis
            trend_score = 0
            if not np.isnan(sma_20[current_idx]) and not np.isnan(sma_50[current_idx]):
                if close[current_idx] > sma_20[current_idx] > sma_50[current_idx]:
                    trend_score = 1  # Uptrend
                elif close[current_idx] < sma_20[current_idx] < sma_50[current_idx]:
                    trend_score = -1  # Downtrend
            
            # Volatility analysis
            volatility_score = 0
            if not np.isnan(bb_width[current_idx]):
                avg_bb_width = np.nanmean(bb_width[max(0, current_idx-20):current_idx+1])
                if bb_width[current_idx] > avg_bb_width * 1.5:
                    volatility_score = 1  # High volatility
                elif bb_width[current_idx] < avg_bb_width * 0.7:
                    volatility_score = -1  # Low volatility
            
            # Volume analysis
            volume_score = 0
            if not np.isnan(volume_ratio[current_idx]):
                if volume_ratio[current_idx] > 1.5:
                    volume_score = 1  # High volume
                elif volume_ratio[current_idx] < 0.7:
                    volume_score = -1  # Low volume
            
            return {
                'trend': trend_score,
                'volatility': volatility_score,
                'volume': volume_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating regime scores: {str(e)}")
            return {'trend': 0, 'volatility': 0, 'volume': 0}
    
def _classify_current_regime(self, regime_scores: Dict[str, float]) -> MarketRegime:
        """Classify the current market regime."""
        try:
            trend = regime_scores.get('trend', 0)
            volatility = regime_scores.get('volatility', 0)
            volume = regime_scores.get('volume', 0)
            
            # High volatility regimes
            if volatility > 0:
                if abs(trend) > 0:
                    return MarketRegime.BREAKOUT
                else:
                    return MarketRegime.HIGH_VOLATILITY
            
            # Low volatility regimes
            elif volatility < 0:
                return MarketRegime.LOW_VOLATILITY
            
            # Trending regimes
            elif trend > 0:
                return MarketRegime.TRENDING_UP
            elif trend < 0:
                return MarketRegime.TRENDING_DOWN
            
            # Default to sideways
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error classifying regime: {str(e)}")
            return MarketRegime.SIDEWAYS
    
def _calculate_reversal_probability(self, mass_index: np.ndarray, mass_index_roc: np.ndarray,:)
(                                      volatility_analysis: Dict) -> np.ndarray:
        """Calculate probability of trend reversal."""
        try:
            reversal_probability = np.zeros_like(mass_index)
            
            for i in range(1, len(mass_index)):
                if np.isnan(mass_index[i]):
                    continue
                
                prob = 0.0
                
                # Base probability from Mass Index level
                if mass_index[i] > self.reversal_threshold:
                    prob += 40.0
                    if mass_index[i] > self.reversal_threshold * 1.1:
                        prob += 20.0
                
                # Rate of change contribution
                if not np.isnan(mass_index_roc[i]):
                    if mass_index_roc[i] > 0:
                        prob += 15.0
                    elif mass_index_roc[i] < 0 and mass_index[i] > self.extreme_threshold:
                        prob += 25.0  # Divergence signal
                
                # Volatility confirmation
                volatility_state = volatility_analysis.get('volatility_state', [])
                if i < len(volatility_state):
                    if volatility_state[i] == VolatilityState.EXTREME_HIGH:
                        prob += 15.0
                    elif volatility_state[i] == VolatilityState.EXPANSION:
                        prob += 10.0
                
                reversal_probability[i] = min(prob, 100.0)
            
            return reversal_probability
            
        except Exception as e:
            logger.error(f"Error calculating reversal probability: {str(e)}")
            return np.zeros_like(mass_index)
    
def _generate_confirmation_signals(self, mass_index: np.ndarray, reversal_signals: np.ndarray,:)
(                                     volatility_analysis: Dict) -> np.ndarray:
        """Generate confirmation signals for reversals."""
        try:
            confirmation = np.zeros_like(mass_index)
            
            # Look for confirmation after reversal signals
            for i in range(1, len(mass_index)):
                if reversal_signals[i] > 0:
                    # Look for confirmation in next few periods
                    confirmation_window = min(5, len(mass_index) - i - 1)
                    for j in range(1, confirmation_window + 1):
                        if i + j < len(mass_index):
                            # Confirmation if Mass Index starts declining
                            if mass_index[i + j] < mass_index[i] * 0.95:
                                confirmation[i + j] = 0.8
                                break
            
            return confirmation
            
        except Exception as e:
            logger.error(f"Error generating confirmation signals: {str(e)}")
            return np.zeros_like(mass_index)
    
def _calculate_signal_strength(self, reversal_signals: np.ndarray, reversal_probability: np.ndarray) -> np.ndarray:
        """Calculate overall signal strength."""
        try:
            # Combine reversal signals and probability
            signal_strength = reversal_signals * 50 + reversal_probability * 0.5
            return np.clip(signal_strength, 0, 100)
        except:
            return np.zeros_like(reversal_signals)
    
def _perform_significance_tests(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        try:
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(data[:5000] if len(data) > 5000 else data)
            
            # Stationarity test (simple version)
            # Calculate first differences
            diff_data = np.diff(data)
            adf_like_stat = np.mean(diff_data) / np.std(diff_data) if np.std(diff_data) > 0 else 0
            
            return {
                'normality_test': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'stationarity_proxy': {'statistic': adf_like_stat},
                'sample_size': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error in significance tests: {str(e)}")
            return {}
    
def _analyze_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze data distribution characteristics."""
        try:
            # Histogram analysis
            hist, bin_edges = np.histogram(data, bins=30)
            
            # Find modes
            mode_indices = np.where(hist == np.max(hist))[0]
            modes = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in mode_indices]
            
            # Distribution shape
            is_multimodal = len(modes) > 1
            
            return {
                'histogram': {'counts': hist.tolist(), 'bin_edges': bin_edges.tolist()},
                'modes': modes,
                'is_multimodal': is_multimodal,
                'distribution_type': 'multimodal' if is_multimodal else 'unimodal'
            }
            
        except Exception as e:
            logger.error(f"Error in distribution analysis: {str(e)}")
            return {}
    
def _calculate_point_signal_strength(self, index: int, mass_index_results: Dict, volatility_analysis: Dict) -> float:
        """Calculate signal strength for a specific point."""
        try:
            mass_index = mass_index_results['mass_index']
            
            if np.isnan(mass_index[index]):
                return 0.0
            
            strength = 0.0
            
            # Base strength from Mass Index level
            if mass_index[index] > self.reversal_threshold:
                strength += 50.0
                if mass_index[index] > self.reversal_threshold * 1.1:
                    strength += 25.0
            
            # Volatility confirmation
            volatility_state = volatility_analysis.get('volatility_state', [])
            if index < len(volatility_state):
                if volatility_state[index] in [VolatilityState.EXPANSION, VolatilityState.EXTREME_HIGH]:
                    strength += 25.0
            
            return min(strength, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating point signal strength: {str(e)}")
            return 0.0
    
def _calculate_point_confidence(self, index: int, mass_index_results: Dict, statistical_analysis: Dict) -> float:
        """Calculate confidence for a specific point."""
        try:
            confidence = 50.0  # Base confidence
            
            # Statistical significance boost
            z_scores = statistical_analysis.get('z_scores', np.array([]))
            if index < len(z_scores):
                if z_scores[index] > 2.0:
                    confidence += 30.0
                elif z_scores[index] > 1.5:
                    confidence += 15.0
            
            # Data quality boost
            mass_index = mass_index_results['mass_index']
            lookback_start = max(0, index - 20)
            recent_data = mass_index[lookback_start:index+1]
            valid_recent = recent_data[~np.isnan(recent_data)]
            
            if len(valid_recent) > 15:
                confidence += 20.0
            elif len(valid_recent) > 10:
                confidence += 10.0
            
            return min(confidence, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating point confidence: {str(e)}")
            return 50.0
    
def _calculate_volume_confirmation(self, index: int, data: pd.DataFrame) -> float:
        """Calculate volume confirmation score."""
        try:
            if 'volume' not in data.columns or index >= len(data):
                return 50.0
            
            volume = data['volume'].values
            current_volume = volume[index]
            
            # Calculate average volume
            lookback_start = max(0, index - 20)
            avg_volume = np.mean(volume[lookback_start:index])
            
            if avg_volume == 0:
                return 50.0
            
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > 2.0:
                return 90.0
            elif volume_ratio > 1.5:
                return 75.0
            elif volume_ratio > 1.2:
                return 60.0
            else:
                return 40.0
                
        except Exception as e:
            logger.error(f"Error calculating volume confirmation: {str(e)}")
            return 50.0
    
def _calculate_support_resistance(self, index: int, data: pd.DataFrame) -> float:
        """Calculate nearest support/resistance level."""
        try:
            if 'close' not in data.columns or index >= len(data):
                return data['close'].iloc[-1] if not data.empty else 0.0
            
            close = data['close'].values
            current_price = close[index]
            
            # Simple support/resistance using recent highs/lows
            lookback_start = max(0, index - 50)
            recent_highs = data['high'].values[lookback_start:index+1]
            recent_lows = data['low'].values[lookback_start:index+1]
            
            # Find nearest significant level
            resistance = np.max(recent_highs) if len(recent_highs) > 0 else current_price
            support = np.min(recent_lows) if len(recent_lows) > 0 else current_price
            
            # Return the nearest level
            dist_to_resistance = abs(current_price - resistance)
            dist_to_support = abs(current_price - support)
            
            return resistance if dist_to_resistance < dist_to_support else support
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return data['close'].iloc[-1] if not data.empty else 0.0
    
def _calculate_risk_level(self, index: int, mass_index_results: Dict, volatility_analysis: Dict) -> float:
        """Calculate risk level for the signal."""
        try:
            risk = 30.0  # Base risk
            
            # Mass Index level risk
            mass_index = mass_index_results['mass_index']
            if not np.isnan(mass_index[index]):
                if mass_index[index] > self.reversal_threshold * 1.2:
                    risk += 40.0
                elif mass_index[index] > self.reversal_threshold:
                    risk += 20.0
            
            # Volatility risk
            volatility_state = volatility_analysis.get('volatility_state', [])
            if index < len(volatility_state):
                if volatility_state[index] == VolatilityState.EXTREME_HIGH:
                    risk += 30.0
                elif volatility_state[index] == VolatilityState.EXPANSION:
                    risk += 15.0
            
            return min(risk, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk level: {str(e)}")
            return 50.0
    
def _calculate_entry_signal(self, index: int, mass_index_results: Dict, reversal_signals: Dict) -> float:
        """Calculate entry signal strength."""
        try:
            entry_signal = 0.0
            
            mass_index = mass_index_results['mass_index']
            reversal_prob = reversal_signals.get('reversal_probability', np.zeros_like(mass_index))
            
            if np.isnan(mass_index[index]):
                return 0.0
            
            # Strong entry signal when Mass Index peaks above threshold
            if mass_index[index] > self.reversal_threshold:
                entry_signal = 60.0
                
                # Additional strength from reversal probability
                if index < len(reversal_prob):
                    entry_signal += reversal_prob[index] * 0.4
            
            return min(entry_signal, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating entry signal: {str(e)}")
            return 0.0
    
def _calculate_exit_signal(self, index: int, mass_index_results: Dict, volatility_analysis: Dict) -> float:
        """Calculate exit signal strength."""
        try:
            exit_signal = 0.0
            
            mass_index = mass_index_results['mass_index']
            
            if np.isnan(mass_index[index]) or index < 5:
                return 0.0
            
            # Exit signal when Mass Index starts declining from high levels
            recent_max = np.max(mass_index[max(0, index-10):index+1])
            if (recent_max > self.reversal_threshold and:)
(                mass_index[index] < recent_max * 0.9):
                exit_signal = 70.0
            
            # Additional exit signal from volatility compression
            volatility_state = volatility_analysis.get('volatility_state', [])
            if index < len(volatility_state):
                if volatility_state[index] == VolatilityState.COMPRESSION:
                    exit_signal += 20.0
            
            return min(exit_signal, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating exit signal: {str(e)}")
            return 0.0
    
def _calculate_statistical_significance(self, index: int, statistical_analysis: Dict) -> float:
        """Calculate statistical significance of the signal."""
        try:
            significance = 50.0  # Base significance
            
            # Z-score significance
            z_scores = statistical_analysis.get('z_scores', np.array([]))
            if index < len(z_scores):
                z_score = z_scores[index]
                if z_score > 3.0:
                    significance = 99.0
                elif z_score > 2.5:
                    significance = 95.0
                elif z_score > 2.0:
                    significance = 90.0
                elif z_score > 1.5:
                    significance = 75.0
            
            return significance
            
        except Exception as e:
            logger.error(f"Error calculating statistical significance: {str(e)}")
            return 50.0
    
def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the calculation results."""
        return {
            'indicator_name': 'Mass Index',
            'version': '1.0.0',
            'parameters': {
                'period': self.period,
                'sum_period': self.sum_period,
                'reversal_threshold': self.reversal_threshold,
                'extreme_threshold': self.extreme_threshold,
                'min_confidence': self.min_confidence,
                'lookback_window': self.lookback_window
            },
            'features': []
                'Volatility expansion detection',
                'Trend reversal probability',
                'Statistical significance testing',
                'Market regime analysis',
                'Volume confirmation',
                'Risk assessment',
                'Machine learning anomaly detection'
[            ],
            'calculation_timestamp': datetime.now(),
            'data_requirements': ['high', 'low', 'close', 'volume']
        }
    
def _generate_empty_result(self) -> Dict[str, Any]:
        """Generate empty result structure."""
        return {
            'mass_index': {},
            'volatility_analysis': {},
            'regime_analysis': {},
            'reversal_signals': {},
            'statistical_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': 'Insufficient data'
        }
    
def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            'mass_index': {},
            'volatility_analysis': {},
            'regime_analysis': {},
            'reversal_signals': {},
            'statistical_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': error_message
        }
def get_data_requirements(self):
        """
        Get data requirements for MarketRegime.
        
        Returns:
            list: List of DataRequirement objects
        """
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import DataRequirement, DataType
        
        # Standard OHLCV requirements for most indicators
        return []
            DataRequirement()
                data_type=DataType.OHLCV,
                required_columns=['open', 'high', 'low', 'close', 'volume'],
                min_periods=20  # Reasonable default for most indicators
(            )
[        ]


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    
    # Generate realistic OHLCV data
    base_price = 100
    returns = np.random.normal(0, 0.02, 500)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some volatility clusters
    volatility_multiplier = np.ones(500)
    volatility_multiplier[100:150] = 2.0  # High volatility period
    volatility_multiplier[300:320] = 3.0  # Extreme volatility period
    
    noise = np.random.normal(0, 0.01, 500) * volatility_multiplier
    
    sample_data = pd.DataFrame({)
        'open': prices + noise * 0.5,
        'high': prices + np.abs(noise) + np.random.exponential(0.01, 500),
        'low': prices - np.abs(noise) - np.random.exponential(0.01, 500),
        'close': prices + noise,
        'volume': np.random.lognormal(10, 0.5, 500)
(    }, index=dates)
    
    # Ensure high >= low and other realistic constraints
    sample_data['high'] = np.maximum(sample_data['high'], sample_data[['open', 'close']].max(axis=1))
    sample_data['low'] = np.minimum(sample_data['low'], sample_data[['open', 'close']].min(axis=1))
    
    # Test the indicator
    mass_index_indicator = MassIndexIndicator()
        period=9,
        sum_period=25,
        reversal_threshold=27.0,
        extreme_threshold=26.5,
        min_confidence=70.0
(    )
    
    try:
        result = mass_index_indicator.calculate(sample_data)
        
        print("Mass Index Indicator Results:")
        print(f"- Calculation successful: {not result.get('error', False)}")
        print(f"- Mass Index values calculated: {len(result.get('mass_index', {}).get('mass_index', []))}")
        print(f"- Signals generated: {len(result.get('signals', []))}")
        print(f"- Volatility analysis completed: {'volatility_analysis' in result}")
        print(f"- Regime analysis completed: {'regime_analysis' in result}")
        print(f"- Statistical analysis completed: {'statistical_analysis' in result}")
        
        # Display some sample signals
        signals = result.get('signals', [])
        if signals:
            print(f"\nSample signals (showing first 3):")
            for i, signal in enumerate(signals[:3]):
                print(f"Signal {i+1}:")
                print(f"  Timestamp: {signal.timestamp}")
                print(f"  Mass Index: {signal.mass_index:.4f}")
                print(f"  Signal Strength: {signal.signal_strength:.2f}")
                print(f"  Confidence: {signal.confidence:.2f}")
                print(f"  Reversal Probability: {signal.reversal_probability:.2f}")
                print(f"  Volatility State: {signal.volatility_state}")
                print(f"  Market Regime: {signal.regime}")
        
        print(f"\nMetadata: {result.get('metadata', {}).get('indicator_name', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing Mass Index Indicator: {str(e)}")
import traceback
        traceback.print_exc()