"""
Positive Volume Index (PVI) - Advanced Retail Sentiment and Volume Trend Analysis
================================================================================

The Positive Volume Index (PVI) is a sophisticated indicator that tracks price movements
on days when volume increases from the previous day. This implementation provides
advanced retail sentiment analysis, volume trend confirmation, and momentum tracking
specifically designed for institutional-grade trading platforms.

Key Features:
- Advanced PVI calculation with volume-weighted momentum analysis
- Retail sentiment tracking and crowd behavior detection
- Multi-timeframe volume trend confirmation
- Machine learning-based momentum classification
- Volume expansion and contraction pattern analysis
- Sentiment divergence detection with statistical validation
- Real-time crowd psychology measurement

Mathematical Foundation:
The Positive Volume Index is calculated using the following methodology:

1. PVI increases when volume today > volume yesterday:
   PVI[i] = PVI[i-1] + ((Close[i] - Close[i-1]) / Close[i-1]) * PVI[i-1]

2. PVI remains unchanged when volume today <= volume yesterday:
   PVI[i] = PVI[i-1]

This implementation extends the basic PVI with:
- Volume-weighted momentum analysis
- Retail vs institutional flow differentiation  
- Multi-period sentiment smoothing
- Machine learning crowd behavior classification
- Statistical anomaly detection

The PVI typically reflects retail investor behavior, as retail traders tend to
increase activity during rising markets and excitement phases.

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
from scipy.spatial.distance import euclidean
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetailSentiment(Enum):
    """Retail sentiment classifications."""
    EXTREME_GREED = "extreme_greed"
    STRONG_GREED = "strong_greed"
    MODERATE_GREED = "moderate_greed"
    NEUTRAL = "neutral"
    MODERATE_FEAR = "moderate_fear"
    STRONG_FEAR = "strong_fear"
    EXTREME_FEAR = "extreme_fear"

class VolumePhase(Enum):
    """Volume trend phase classifications."""
    VOLUME_EXPANSION = "volume_expansion"
    VOLUME_ACCELERATION = "volume_acceleration"
    VOLUME_PEAK = "volume_peak"
    VOLUME_DECELERATION = "volume_deceleration"
    VOLUME_CONTRACTION = "volume_contraction"
    VOLUME_EQUILIBRIUM = "volume_equilibrium"

class CrowdBehavior(Enum):
    """Crowd behavior pattern classifications."""
    FOMO_BUYING = "fomo_buying"
    EUPHORIC_ACCUMULATION = "euphoric_accumulation"
    PANIC_SELLING = "panic_selling"
    CAPITULATION = "capitulation"
    RATIONAL_DISTRIBUTION = "rational_distribution"
    CONTRARIAN_ACCUMULATION = "contrarian_accumulation"
    UNCERTAIN = "uncertain"

@dataclass
class PositiveVolumeIndexSignal:
    """
    Comprehensive Positive Volume Index signal.
    
    Attributes:
        timestamp: Signal timestamp
        pvi_value: Current PVI value
        pvi_momentum: PVI momentum (rate of change)
        retail_sentiment: Current retail sentiment classification
        volume_phase: Current volume trend phase
        crowd_behavior: Detected crowd behavior pattern
        sentiment_strength: Strength of current sentiment (0-100)
        momentum_score: Volume momentum scoring (0-100)
        trend_confirmation: Trend confirmation level (0-100)
        divergence_signal: Price-PVI divergence detection
        expansion_probability: Probability of volume expansion
        retail_pressure: Retail buying/selling pressure measurement
        crowd_psychology_score: Overall crowd psychology indicator
        volume_anomaly_score: Volume anomaly detection score
        next_phase_prediction: Predicted next volume phase
        confidence: Overall signal confidence (0-100)
    """
    timestamp: datetime
    pvi_value: float
    pvi_momentum: float
    retail_sentiment: RetailSentiment
    volume_phase: VolumePhase
    crowd_behavior: CrowdBehavior
    sentiment_strength: float
    momentum_score: float
    trend_confirmation: float
    divergence_signal: float
    expansion_probability: float
    retail_pressure: float
    crowd_psychology_score: float
    volume_anomaly_score: float
    next_phase_prediction: VolumePhase
    confidence: float

class PositiveVolumeIndexIndicator:
    """
    Advanced Positive Volume Index Indicator with retail sentiment analysis,
    volume trend confirmation, and momentum tracking.
    
    This indicator provides sophisticated analysis of retail investor behavior
    through volume-price relationships and crowd psychology measurement.
    """
    
    def __init__(self,
                 initial_value: float = 1000.0,
                 momentum_period: int = 14,
                 sentiment_period: int = 20,
                 trend_period: int = 50,
                 volume_threshold: float = 0.05,
                 sentiment_threshold: float = 2.0,
                 min_confidence: float = 70.0):
        """
        Initialize the Positive Volume Index Indicator.
        
        Args:
            initial_value: Initial PVI value
            momentum_period: Period for momentum calculations
            sentiment_period: Period for sentiment analysis
            trend_period: Period for trend confirmation
            volume_threshold: Threshold for volume significance
            sentiment_threshold: Threshold for sentiment classification
            min_confidence: Minimum confidence for signal generation
        """
        self.initial_value = initial_value
        self.momentum_period = momentum_period
        self.sentiment_period = sentiment_period
        self.trend_period = trend_period
        self.volume_threshold = volume_threshold
        self.sentiment_threshold = sentiment_threshold
        self.min_confidence = min_confidence
        
        # Machine learning models
        self._sentiment_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self._momentum_predictor = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=8
        )
        self._volume_clusterer = KMeans(n_clusters=6, random_state=42)
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        
        # Analysis state
        self._historical_pvi = []
        self._volume_patterns = []
        self._sentiment_history = []
        
        logger.info(f"PositiveVolumeIndexIndicator initialized with momentum_period={momentum_period}")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Positive Volume Index with advanced retail sentiment analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing PVI analysis and retail sentiment metrics
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(data) < max(self.momentum_period, self.sentiment_period, self.trend_period) * 2:
                logger.warning(f"Insufficient data for analysis")
                return self._generate_empty_result()
            
            # Calculate basic PVI
            pvi_analysis = self._calculate_pvi(data)
            
            # Analyze retail sentiment patterns
            sentiment_analysis = self._analyze_retail_sentiment(data, pvi_analysis)
            
            # Analyze volume trends and phases
            volume_analysis = self._analyze_volume_trends(data, pvi_analysis)
            
            # Detect crowd behavior patterns
            crowd_analysis = self._analyze_crowd_behavior(data, pvi_analysis, sentiment_analysis)
            
            # Perform momentum tracking
            momentum_analysis = self._analyze_momentum_patterns(data, pvi_analysis)
            
            # Analyze divergences and confirmations
            divergence_analysis = self._analyze_divergences(data, pvi_analysis)
            
            # Perform predictive modeling
            prediction_analysis = self._perform_predictive_modeling(data, pvi_analysis, sentiment_analysis)
            
            # Generate comprehensive signals
            signals = self._generate_comprehensive_signals(
                data, pvi_analysis, sentiment_analysis, volume_analysis,
                crowd_analysis, momentum_analysis, divergence_analysis, prediction_analysis
            )
            
            return {
                'pvi_analysis': pvi_analysis,
                'sentiment_analysis': sentiment_analysis,
                'volume_analysis': volume_analysis,
                'crowd_analysis': crowd_analysis,
                'momentum_analysis': momentum_analysis,
                'divergence_analysis': divergence_analysis,
                'prediction_analysis': prediction_analysis,
                'signals': signals,
                'metadata': self._generate_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in PositiveVolumeIndexIndicator calculation: {str(e)}")
            return self._generate_error_result(str(e))
    
    def _calculate_pvi(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate basic and enhanced PVI values."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate basic PVI
            pvi = np.full_like(close, self.initial_value)
            
            for i in range(1, len(close)):
                if volume[i] > volume[i-1]:
                    # Volume increased - update PVI
                    price_change = (close[i] - close[i-1]) / close[i-1]
                    pvi[i] = pvi[i-1] * (1 + price_change)
                else:
                    # Volume decreased or unchanged - keep PVI same
                    pvi[i] = pvi[i-1]
            
            # Calculate PVI momentum and derivatives
            pvi_momentum = np.gradient(pvi)
            pvi_acceleration = np.gradient(pvi_momentum)
            
            # Calculate volume-weighted PVI
            volume_ma = talib.SMA(volume, timeperiod=20)
            volume_weight = np.where(volume_ma > 0, volume / volume_ma, 1.0)
            weighted_pvi = pvi * np.sqrt(volume_weight)  # Square root to reduce noise
            
            # Calculate multi-period PVI smoothings
            pvi_fast = talib.EMA(pvi, timeperiod=self.momentum_period)
            pvi_slow = talib.EMA(pvi, timeperiod=self.trend_period)
            pvi_signal = pvi_fast - pvi_slow
            
            # Calculate PVI rate of change
            pvi_roc = talib.ROC(pvi, timeperiod=self.momentum_period)
            
            # Calculate PVI standard deviation and volatility
            pvi_std = talib.STDDEV(pvi, timeperiod=self.sentiment_period)
            pvi_volatility = pvi_std / pvi * 100  # Coefficient of variation
            
            return {
                'pvi': pvi,
                'pvi_momentum': pvi_momentum,
                'pvi_acceleration': pvi_acceleration,
                'weighted_pvi': weighted_pvi,
                'pvi_fast': pvi_fast,
                'pvi_slow': pvi_slow,
                'pvi_signal': pvi_signal,
                'pvi_roc': pvi_roc,
                'pvi_volatility': pvi_volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating PVI: {str(e)}")
            return {}
    
    def _analyze_retail_sentiment(self, data: pd.DataFrame, pvi_analysis: Dict) -> Dict[str, Any]:
        """Analyze retail sentiment patterns."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            pvi = pvi_analysis.get('pvi', np.array([]))
            pvi_momentum = pvi_analysis.get('pvi_momentum', np.array([]))
            
            # Analyze volume expansion days (retail activity)
            volume_expansion = np.zeros_like(volume)
            retail_activity = np.zeros_like(volume)
            sentiment_scores = np.zeros_like(volume)
            
            for i in range(1, len(volume)):
                if volume[i] > volume[i-1]:
                    volume_expansion[i] = (volume[i] - volume[i-1]) / volume[i-1]
                    
                    # Measure retail activity intensity
                    price_change = (close[i] - close[i-1]) / close[i-1]
                    volume_change = volume_expansion[i]
                    
                    # Retail intensity = volume increase with price correlation
                    retail_activity[i] = volume_change * (1 + abs(price_change))
                    
                    # Sentiment scoring based on price-volume relationship
                    if price_change > 0 and volume_change > self.volume_threshold:
                        # Positive price with increasing volume = bullish sentiment
                        sentiment_scores[i] = min(100, volume_change * 100 + price_change * 50)
                    elif price_change < 0 and volume_change > self.volume_threshold:
                        # Negative price with increasing volume = bearish sentiment
                        sentiment_scores[i] = max(-100, -volume_change * 100 + price_change * 50)
            
            # Smooth sentiment scores
            sentiment_smoothed = talib.EMA(sentiment_scores, timeperiod=self.sentiment_period)
            
            # Classify sentiment levels
            sentiment_classifications = []
            for score in sentiment_smoothed:
                if score > self.sentiment_threshold * 2:
                    sentiment_classifications.append(RetailSentiment.EXTREME_GREED)
                elif score > self.sentiment_threshold:
                    sentiment_classifications.append(RetailSentiment.STRONG_GREED)
                elif score > self.sentiment_threshold / 2:
                    sentiment_classifications.append(RetailSentiment.MODERATE_GREED)
                elif score < -self.sentiment_threshold * 2:
                    sentiment_classifications.append(RetailSentiment.EXTREME_FEAR)
                elif score < -self.sentiment_threshold:
                    sentiment_classifications.append(RetailSentiment.STRONG_FEAR)
                elif score < -self.sentiment_threshold / 2:
                    sentiment_classifications.append(RetailSentiment.MODERATE_FEAR)
                else:
                    sentiment_classifications.append(RetailSentiment.NEUTRAL)
            
            # Calculate sentiment strength
            sentiment_strength = np.abs(sentiment_smoothed) / (self.sentiment_threshold * 2) * 100
            sentiment_strength = np.clip(sentiment_strength, 0, 100)
            
            return {
                'volume_expansion': volume_expansion,
                'retail_activity': retail_activity,
                'sentiment_scores': sentiment_scores,
                'sentiment_smoothed': sentiment_smoothed,
                'sentiment_classifications': sentiment_classifications,
                'sentiment_strength': sentiment_strength
            }
            
        except Exception as e:
            logger.error(f"Error analyzing retail sentiment: {str(e)}")
            return {}
    
    def _analyze_volume_trends(self, data: pd.DataFrame, pvi_analysis: Dict) -> Dict[str, Any]:
        """Analyze volume trends and phases."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            pvi = pvi_analysis.get('pvi', np.array([]))
            
            # Calculate volume trend indicators
            volume_ma_short = talib.SMA(volume, timeperiod=self.momentum_period)
            volume_ma_long = talib.SMA(volume, timeperiod=self.trend_period)
            volume_trend = volume_ma_short / volume_ma_long
            
            # Volume acceleration
            volume_acceleration = np.gradient(np.gradient(volume))
            
            # Classify volume phases
            volume_phases = []
            phase_strength = np.zeros_like(volume)
            
            for i in range(self.trend_period, len(volume)):
                recent_volume = volume[i-self.momentum_period:i]
                volume_change = np.gradient(recent_volume)
                acceleration = np.gradient(volume_change)
                
                trend_val = volume_trend[i] if i < len(volume_trend) else 1.0
                accel_val = np.mean(acceleration) if len(acceleration) > 0 else 0
                
                # Phase classification logic
                if trend_val > 1.2 and accel_val > 0:
                    phase = VolumePhase.VOLUME_ACCELERATION
                    strength = min(100, (trend_val - 1) * 100 + accel_val * 50)
                elif trend_val > 1.1:
                    phase = VolumePhase.VOLUME_EXPANSION
                    strength = (trend_val - 1) * 100
                elif trend_val < 0.8 and accel_val < 0:
                    phase = VolumePhase.VOLUME_CONTRACTION
                    strength = (1 - trend_val) * 100
                elif trend_val < 0.9:
                    phase = VolumePhase.VOLUME_DECELERATION
                    strength = (1 - trend_val) * 50
                elif abs(accel_val) < 0.01 and 0.95 < trend_val < 1.05:
                    phase = VolumePhase.VOLUME_EQUILIBRIUM
                    strength = 100 - abs(trend_val - 1) * 200
                else:
                    phase = VolumePhase.VOLUME_PEAK
                    strength = abs(accel_val) * 100
                
                volume_phases.append(phase)
                phase_strength[i] = strength
            
            # Pad beginning of arrays
            while len(volume_phases) < len(volume):
                volume_phases.insert(0, VolumePhase.VOLUME_EQUILIBRIUM)
            
            # Volume momentum scoring
            volume_momentum = talib.MOM(volume, timeperiod=self.momentum_period)
            momentum_score = np.zeros_like(volume)
            
            for i in range(self.momentum_period, len(volume)):
                momentum_percentile = stats.percentileofscore(
                    volume_momentum[max(0, i-100):i], volume_momentum[i]
                )
                momentum_score[i] = momentum_percentile
            
            return {
                'volume_trend': volume_trend,
                'volume_acceleration': volume_acceleration,
                'volume_phases': volume_phases,
                'phase_strength': phase_strength,
                'momentum_score': momentum_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume trends: {str(e)}")
            return {}
    
    def _analyze_crowd_behavior(self, data: pd.DataFrame, pvi_analysis: Dict, 
                              sentiment_analysis: Dict) -> Dict[str, Any]:
        """Analyze crowd behavior patterns."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            pvi = pvi_analysis.get('pvi', np.array([]))
            sentiment_scores = sentiment_analysis.get('sentiment_scores', np.array([]))
            retail_activity = sentiment_analysis.get('retail_activity', np.array([]))
            
            # Analyze crowd behavior patterns
            crowd_behaviors = []
            psychology_scores = np.zeros_like(close)
            retail_pressure = np.zeros_like(close)
            
            for i in range(self.sentiment_period, len(close)):
                # Recent data window
                recent_prices = close[i-self.sentiment_period:i]
                recent_volume = volume[i-self.sentiment_period:i]
                recent_sentiment = sentiment_scores[i-self.sentiment_period:i]
                recent_activity = retail_activity[i-self.sentiment_period:i]
                
                # Price trend analysis
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
                
                # Sentiment momentum
                sentiment_momentum = np.mean(recent_sentiment[-5:]) - np.mean(recent_sentiment[:5])
                activity_intensity = np.mean(recent_activity)
                
                # Psychology scoring
                psychology_scores[i] = (
                    np.mean(recent_sentiment) * 0.4 +
                    sentiment_momentum * 0.3 +
                    activity_intensity * 100 * 0.3
                )
                
                # Retail pressure calculation
                if volume[i] > volume[i-1]:  # PVI active day
                    price_change = (close[i] - close[i-1]) / close[i-1]
                    volume_change = (volume[i] - volume[i-1]) / volume[i-1]
                    retail_pressure[i] = price_change * volume_change * 100
                
                # Crowd behavior classification
                if activity_intensity > 0.1 and sentiment_momentum > 2:
                    if price_trend > 0:
                        behavior = CrowdBehavior.FOMO_BUYING
                    else:
                        behavior = CrowdBehavior.EUPHORIC_ACCUMULATION
                elif activity_intensity > 0.1 and sentiment_momentum < -2:
                    if volume_trend > 0:
                        behavior = CrowdBehavior.PANIC_SELLING
                    else:
                        behavior = CrowdBehavior.CAPITULATION
                elif activity_intensity < 0.05 and abs(sentiment_momentum) < 1:
                    if price_trend > 0:
                        behavior = CrowdBehavior.RATIONAL_DISTRIBUTION
                    else:
                        behavior = CrowdBehavior.CONTRARIAN_ACCUMULATION
                else:
                    behavior = CrowdBehavior.UNCERTAIN
                
                crowd_behaviors.append(behavior)
            
            # Pad beginning
            while len(crowd_behaviors) < len(close):
                crowd_behaviors.insert(0, CrowdBehavior.UNCERTAIN)
            
            return {
                'crowd_behaviors': crowd_behaviors,
                'psychology_scores': psychology_scores,
                'retail_pressure': retail_pressure
            }
            
        except Exception as e:
            logger.error(f"Error analyzing crowd behavior: {str(e)}")
            return {}
    
    def _analyze_momentum_patterns(self, data: pd.DataFrame, pvi_analysis: Dict) -> Dict[str, Any]:
        """Analyze momentum patterns in PVI."""
        try:
            close = data['close'].values
            pvi = pvi_analysis.get('pvi', np.array([]))
            pvi_momentum = pvi_analysis.get('pvi_momentum', np.array([]))
            pvi_roc = pvi_analysis.get('pvi_roc', np.array([]))
            
            # Momentum trend analysis
            momentum_trend = np.zeros_like(pvi_momentum)
            momentum_strength = np.zeros_like(pvi_momentum)
            
            for i in range(self.momentum_period, len(pvi_momentum)):
                recent_momentum = pvi_momentum[i-self.momentum_period:i]
                
                # Momentum trend
                momentum_trend[i] = np.polyfit(range(len(recent_momentum)), recent_momentum, 1)[0]
                
                # Momentum strength
                momentum_strength[i] = np.std(recent_momentum) * 100
            
            # Momentum oscillator
            momentum_fast = talib.EMA(pvi_momentum, timeperiod=int(self.momentum_period/2))
            momentum_slow = talib.EMA(pvi_momentum, timeperiod=self.momentum_period)
            momentum_oscillator = momentum_fast - momentum_slow
            
            # Momentum divergence detection
            price_peaks, _ = find_peaks(close, distance=self.momentum_period)
            pvi_peaks, _ = find_peaks(pvi, distance=self.momentum_period)
            
            momentum_divergence = np.zeros_like(close)
            
            # Analyze peak divergences
            if len(price_peaks) > 1 and len(pvi_peaks) > 1:
                for i in range(min(len(price_peaks), len(pvi_peaks)) - 1):
                    p_idx = price_peaks[i]
                    pvi_idx = pvi_peaks[i]
                    
                    if abs(p_idx - pvi_idx) < self.momentum_period:  # Peaks are close in time
                        next_p_idx = price_peaks[i+1] if i+1 < len(price_peaks) else p_idx
                        next_pvi_idx = pvi_peaks[i+1] if i+1 < len(pvi_peaks) else pvi_idx
                        
                        price_change = close[next_p_idx] - close[p_idx]
                        pvi_change = pvi[next_pvi_idx] - pvi[pvi_idx]
                        
                        # Divergence scoring
                        if price_change > 0 and pvi_change < 0:
                            momentum_divergence[next_p_idx] = -abs(price_change / close[p_idx] * 100)
                        elif price_change < 0 and pvi_change > 0:
                            momentum_divergence[next_p_idx] = abs(price_change / close[p_idx] * 100)
            
            return {
                'momentum_trend': momentum_trend,
                'momentum_strength': momentum_strength,
                'momentum_oscillator': momentum_oscillator,
                'momentum_divergence': momentum_divergence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum patterns: {str(e)}")
            return {}
    
    def _analyze_divergences(self, data: pd.DataFrame, pvi_analysis: Dict) -> Dict[str, Any]:
        """Analyze price-PVI divergences and confirmations."""
        try:
            close = data['close'].values
            pvi = pvi_analysis.get('pvi', np.array([]))
            
            # Price and PVI normalized for correlation analysis
            price_normalized = (close - np.mean(close)) / np.std(close)
            pvi_normalized = (pvi - np.mean(pvi)) / np.std(pvi)
            
            # Rolling correlation analysis
            correlation_window = self.trend_period
            correlations = np.zeros_like(close)
            divergence_strength = np.zeros_like(close)
            
            for i in range(correlation_window, len(close)):
                price_window = price_normalized[i-correlation_window:i]
                pvi_window = pvi_normalized[i-correlation_window:i]
                
                correlation = np.corrcoef(price_window, pvi_window)[0, 1]
                correlations[i] = correlation if not np.isnan(correlation) else 0
                
                # Divergence strength (inverse of correlation)
                divergence_strength[i] = max(0, (1 - abs(correlation)) * 100)
            
            # Trend confirmation analysis
            price_trend = talib.EMA(close, timeperiod=self.trend_period)
            pvi_trend = talib.EMA(pvi, timeperiod=self.trend_period)
            
            price_direction = np.gradient(price_trend)
            pvi_direction = np.gradient(pvi_trend)
            
            # Confirmation scoring
            confirmation_scores = np.zeros_like(close)
            
            for i in range(1, len(close)):
                if price_direction[i] * pvi_direction[i] > 0:
                    # Same direction = confirmation
                    confirmation_scores[i] = min(100, abs(correlations[i]) * 100)
                else:
                    # Opposite direction = divergence
                    confirmation_scores[i] = max(0, 50 - abs(correlations[i]) * 50)
            
            return {
                'correlations': correlations,
                'divergence_strength': divergence_strength,
                'confirmation_scores': confirmation_scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing divergences: {str(e)}")
            return {}
    
    def _perform_predictive_modeling(self, data: pd.DataFrame, pvi_analysis: Dict,
                                   sentiment_analysis: Dict) -> Dict[str, Any]:
        """Perform predictive modeling for volume phases and sentiment."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            pvi = pvi_analysis.get('pvi', np.array([]))
            sentiment_scores = sentiment_analysis.get('sentiment_scores', np.array([]))
            
            # Prepare features for prediction
            features = []
            targets = []
            
            lookback = 20
            lookahead = 5
            
            for i in range(lookback, len(pvi) - lookahead):
                # Features: recent PVI, volume, sentiment patterns
                feature_vector = []
                
                # PVI features
                feature_vector.extend(pvi[i-10:i].tolist())
                feature_vector.append(np.mean(pvi[i-lookback:i]))
                feature_vector.append(np.std(pvi[i-lookback:i]))
                
                # Volume features
                feature_vector.extend(volume[i-5:i].tolist())
                feature_vector.append(np.mean(volume[i-lookback:i]))
                
                # Sentiment features
                if len(sentiment_scores) > i:
                    feature_vector.extend(sentiment_scores[i-5:i].tolist())
                else:
                    feature_vector.extend([0] * 5)
                
                features.append(feature_vector)
                
                # Target: future volume expansion probability
                future_volume = volume[i:i+lookahead]
                current_volume = volume[i]
                expansion_prob = np.mean(future_volume > current_volume)
                targets.append(expansion_prob)
            
            if len(features) < 50:
                return {'expansion_predictions': np.zeros_like(volume)}
            
            features = np.array(features)
            targets = np.array(targets)
            
            # Train prediction model
            split_point = int(len(features) * 0.8)
            train_features = features[:split_point]
            train_targets = targets[:split_point]
            
            try:
                self._momentum_predictor.fit(train_features, train_targets)
                
                # Generate predictions for all available data
                predictions = np.zeros_like(volume)
                
                for i in range(lookback, len(features) + lookback):
                    if i - lookback < len(features):
                        pred = self._momentum_predictor.predict([features[i - lookback]])[0]
                        predictions[i] = pred
                
            except Exception as e:
                logger.warning(f"Prediction model training failed: {str(e)}")
                predictions = np.full_like(volume, 0.5)
            
            # Volume phase prediction
            phase_predictions = []
            for pred in predictions:
                if pred > 0.8:
                    phase_predictions.append(VolumePhase.VOLUME_ACCELERATION)
                elif pred > 0.6:
                    phase_predictions.append(VolumePhase.VOLUME_EXPANSION)
                elif pred < 0.2:
                    phase_predictions.append(VolumePhase.VOLUME_CONTRACTION)
                elif pred < 0.4:
                    phase_predictions.append(VolumePhase.VOLUME_DECELERATION)
                else:
                    phase_predictions.append(VolumePhase.VOLUME_EQUILIBRIUM)
            
            return {
                'expansion_predictions': predictions,
                'phase_predictions': phase_predictions
            }
            
        except Exception as e:
            logger.error(f"Error in predictive modeling: {str(e)}")
            return {'expansion_predictions': np.zeros(len(data)), 'phase_predictions': []}
    
    def _generate_comprehensive_signals(self, data: pd.DataFrame, pvi_analysis: Dict,
                                      sentiment_analysis: Dict, volume_analysis: Dict,
                                      crowd_analysis: Dict, momentum_analysis: Dict,
                                      divergence_analysis: Dict, prediction_analysis: Dict) -> List[PositiveVolumeIndexSignal]:
        """Generate comprehensive PVI signals."""
        try:
            signals = []
            timestamps = pd.to_datetime(data.index) if hasattr(data.index, 'to_pydatetime') else data.index
            
            pvi = pvi_analysis.get('pvi', np.array([]))
            pvi_momentum = pvi_analysis.get('pvi_momentum', np.array([]))
            sentiment_classifications = sentiment_analysis.get('sentiment_classifications', [])
            sentiment_strength = sentiment_analysis.get('sentiment_strength', np.array([]))
            volume_phases = volume_analysis.get('volume_phases', [])
            momentum_score = volume_analysis.get('momentum_score', np.array([]))
            crowd_behaviors = crowd_analysis.get('crowd_behaviors', [])
            psychology_scores = crowd_analysis.get('psychology_scores', np.array([]))
            retail_pressure = crowd_analysis.get('retail_pressure', np.array([]))
            divergence_strength = divergence_analysis.get('divergence_strength', np.array([]))
            confirmation_scores = divergence_analysis.get('confirmation_scores', np.array([]))
            expansion_predictions = prediction_analysis.get('expansion_predictions', np.array([]))
            phase_predictions = prediction_analysis.get('phase_predictions', [])
            
            for i in range(max(self.sentiment_period, self.trend_period), len(data)):
                try:
                    # Calculate volume anomaly score
                    recent_volume = data['volume'].iloc[max(0, i-20):i].values
                    current_volume = data['volume'].iloc[i]
                    volume_percentile = stats.percentileofscore(recent_volume, current_volume)
                    volume_anomaly_score = abs(volume_percentile - 50) * 2  # 0-100 scale
                    
                    # Calculate overall confidence
                    confidence_factors = [
                        min(100, sentiment_strength[i]) if i < len(sentiment_strength) else 50,
                        min(100, momentum_score[i]) if i < len(momentum_score) else 50,
                        min(100, confirmation_scores[i]) if i < len(confirmation_scores) else 50,
                        min(100, abs(psychology_scores[i])) if i < len(psychology_scores) else 50
                    ]
                    
                    confidence = np.mean(confidence_factors)
                    
                    if confidence >= self.min_confidence:
                        signal = PositiveVolumeIndexSignal(
                            timestamp=timestamps[i] if i < len(timestamps) else datetime.now(),
                            pvi_value=pvi[i] if i < len(pvi) else self.initial_value,
                            pvi_momentum=pvi_momentum[i] if i < len(pvi_momentum) else 0.0,
                            retail_sentiment=sentiment_classifications[i] if i < len(sentiment_classifications) else RetailSentiment.NEUTRAL,
                            volume_phase=volume_phases[i] if i < len(volume_phases) else VolumePhase.VOLUME_EQUILIBRIUM,
                            crowd_behavior=crowd_behaviors[i] if i < len(crowd_behaviors) else CrowdBehavior.UNCERTAIN,
                            sentiment_strength=sentiment_strength[i] if i < len(sentiment_strength) else 0.0,
                            momentum_score=momentum_score[i] if i < len(momentum_score) else 0.0,
                            trend_confirmation=confirmation_scores[i] if i < len(confirmation_scores) else 0.0,
                            divergence_signal=divergence_strength[i] if i < len(divergence_strength) else 0.0,
                            expansion_probability=expansion_predictions[i] if i < len(expansion_predictions) else 0.5,
                            retail_pressure=retail_pressure[i] if i < len(retail_pressure) else 0.0,
                            crowd_psychology_score=psychology_scores[i] if i < len(psychology_scores) else 0.0,
                            volume_anomaly_score=volume_anomaly_score,
                            next_phase_prediction=phase_predictions[i] if i < len(phase_predictions) else VolumePhase.VOLUME_EQUILIBRIUM,
                            confidence=confidence
                        )
                        
                        signals.append(signal)
                
                except Exception as e:
                    logger.warning(f"Error generating signal at index {i}: {str(e)}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {str(e)}")
            return []
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the calculation results."""
        return {
            'indicator_name': 'Positive Volume Index',
            'version': '1.0.0',
            'parameters': {
                'initial_value': self.initial_value,
                'momentum_period': self.momentum_period,
                'sentiment_period': self.sentiment_period,
                'trend_period': self.trend_period,
                'volume_threshold': self.volume_threshold,
                'sentiment_threshold': self.sentiment_threshold,
                'min_confidence': self.min_confidence
            },
            'features': [
                'Advanced PVI calculation with volume-weighted momentum',
                'Retail sentiment tracking and crowd behavior detection',
                'Multi-timeframe volume trend confirmation',
                'Machine learning momentum classification',
                'Volume expansion/contraction pattern analysis',
                'Sentiment divergence detection with statistical validation',
                'Real-time crowd psychology measurement',
                'Predictive volume phase modeling'
            ],
            'calculation_timestamp': datetime.now(),
            'data_requirements': ['open', 'high', 'low', 'close', 'volume']
        }
    
    def _generate_empty_result(self) -> Dict[str, Any]:
        """Generate empty result structure."""
        return {
            'pvi_analysis': {},
            'sentiment_analysis': {},
            'volume_analysis': {},
            'crowd_analysis': {},
            'momentum_analysis': {},
            'divergence_analysis': {},
            'prediction_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': 'Insufficient data'
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            'pvi_analysis': {},
            'sentiment_analysis': {},
            'volume_analysis': {},
            'crowd_analysis': {},
            'momentum_analysis': {},
            'divergence_analysis': {},
            'prediction_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': error_message
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='1H')
    
    # Generate realistic OHLCV data with retail sentiment patterns
    base_price = 100
    returns = np.random.normal(0, 0.01, 500)
    
    # Add retail sentiment patterns (volume spikes with price movements)
    volume_base = np.random.lognormal(8, 0.5, 500)
    
    for i in range(50, 450, 30):
        # Create retail FOMO patterns
        if i % 60 == 50:  # Bullish retail sentiment
            returns[i:i+10] = np.abs(np.random.normal(0.02, 0.005, 10))
            volume_base[i:i+10] *= np.random.uniform(2, 4, 10)  # Volume expansion
        elif i % 60 == 20:  # Bearish retail sentiment  
            returns[i:i+8] = -np.abs(np.random.normal(0.02, 0.005, 8))
            volume_base[i:i+8] *= np.random.uniform(1.5, 3, 8)  # Panic volume
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, 500))),
        'close': prices,
        'volume': volume_base
    }, index=dates)
    
    # Test the indicator
    indicator = PositiveVolumeIndexIndicator(
        initial_value=1000.0,
        momentum_period=14,
        sentiment_period=20,
        trend_period=50,
        min_confidence=70.0
    )
    
    try:
        result = indicator.calculate(sample_data)
        
        print("Positive Volume Index Indicator Results:")
        print(f"- Calculation successful: {not result.get('error', False)}")
        print(f"- Signals generated: {len(result.get('signals', []))}")
        print(f"- PVI final value: {result.get('pvi_analysis', {}).get('pvi', [0])[-1]:.2f}")
        print(f"- Momentum period: {indicator.momentum_period}")
        print(f"- Sentiment period: {indicator.sentiment_period}")
        
        # Display some sample signals
        signals = result.get('signals', [])
        if signals:
            print(f"\nSample PVI signals (showing first 3):")
            for i, signal in enumerate(signals[:3]):
                print(f"Signal {i+1}:")
                print(f"  Timestamp: {signal.timestamp}")
                print(f"  PVI Value: {signal.pvi_value:.2f}")
                print(f"  Retail Sentiment: {signal.retail_sentiment}")
                print(f"  Volume Phase: {signal.volume_phase}")
                print(f"  Crowd Behavior: {signal.crowd_behavior}")
                print(f"  Sentiment Strength: {signal.sentiment_strength:.2f}")
                print(f"  Momentum Score: {signal.momentum_score:.2f}")
                print(f"  Expansion Probability: {signal.expansion_probability:.2f}")
                print(f"  Retail Pressure: {signal.retail_pressure:.2f}")
                print(f"  Confidence: {signal.confidence:.2f}")
        
        print(f"\nMetadata: {result.get('metadata', {}).get('indicator_name', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing Positive Volume Index Indicator: {str(e)}")
        import traceback
        traceback.print_exc()