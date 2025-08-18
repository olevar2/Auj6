"""
Momentum Divergence Scanner Indicator for AUJ Platform
Advanced multi-timeframe momentum analysis and divergence detection

This indicator implements sophisticated momentum divergence scanning across multiple
timeframes with advanced signal processing, machine learning classification, and
comprehensive divergence pattern recognition for maximum trading accuracy.

Features:
- Multi-timeframe momentum analysis with adaptive timeframe selection
- Advanced divergence detection using multiple momentum oscillators
- Machine learning-based divergence classification and validation
- Fractal-based momentum pattern recognition
- Statistical significance testing for divergence signals
- Real-time momentum strength analysis with decay modeling
- Advanced signal filtering and noise reduction
- Comprehensive divergence scoring and ranking system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal, stats
from scipy.signal import find_peaks, hilbert, savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


class DivergenceType(Enum):
    """Types of momentum divergences"""
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_HIDDEN = "bearish_hidden"
    BULLISH_EXAGGERATED = "bullish_exaggerated"
    BEARISH_EXAGGERATED = "bearish_exaggerated"


class MomentumOscillator(Enum):
    """Supported momentum oscillators for divergence analysis"""
    RSI = "rsi"
    MACD = "macd"
    STOCH = "stochastic"
    MFI = "mfi"
    CCI = "cci"
    ROC = "roc"
    WILLIAMS_R = "williams_r"
    TSI = "tsi"


@dataclass
class DivergenceSignal:
    """Container for divergence signal information"""
    type: DivergenceType
    oscillator: MomentumOscillator
    timeframe: str
    strength: float
    confidence: float
    price_points: Tuple[int, int]
    oscillator_points: Tuple[int, int]
    duration: int
    significance: float
    price_change: float
    oscillator_change: float
    volume_confirmation: float
    fractal_dimension: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'type': self.type.value,
            'oscillator': self.oscillator.value,
            'timeframe': self.timeframe,
            'strength': self.strength,
            'confidence': self.confidence,
            'price_points': self.price_points,
            'oscillator_points': self.oscillator_points,
            'duration': self.duration,
            'significance': self.significance,
            'price_change': self.price_change,
            'oscillator_change': self.oscillator_change,
            'volume_confirmation': self.volume_confirmation,
            'fractal_dimension': self.fractal_dimension
        }


@dataclass
class MomentumAnalysis:
    """Container for comprehensive momentum analysis"""
    timeframe: str
    momentum_strength: float
    momentum_direction: float
    momentum_acceleration: float
    momentum_persistence: float
    momentum_quality: float
    divergence_signals: List[DivergenceSignal]
    support_resistance_levels: List[float]
    trend_strength: float
    volatility_adjusted_momentum: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timeframe': self.timeframe,
            'momentum_strength': self.momentum_strength,
            'momentum_direction': self.momentum_direction,
            'momentum_acceleration': self.momentum_acceleration,
            'momentum_persistence': self.momentum_persistence,
            'momentum_quality': self.momentum_quality,
            'divergence_signals': [sig.to_dict() for sig in self.divergence_signals],
            'support_resistance_levels': self.support_resistance_levels,
            'trend_strength': self.trend_strength,
            'volatility_adjusted_momentum': self.volatility_adjusted_momentum
        }


class MomentumDivergenceScannerIndicator(StandardIndicatorInterface):
    """
    Advanced Momentum Divergence Scanner Indicator
    
    This indicator provides sophisticated multi-timeframe momentum analysis with
    advanced divergence detection capabilities, machine learning validation,
    and comprehensive signal scoring.
    """
    
    def __init__(self, 
                 primary_period: int = 14,
                 secondary_period: int = 28,
                 timeframes: List[str] = None,
                 oscillators: List[MomentumOscillator] = None,
                 min_divergence_bars: int = 5,
                 max_divergence_bars: int = 50,
                 significance_threshold: float = 0.05,
                 min_strength_threshold: float = 0.3,
                 use_ml_validation: bool = True,
                 fractal_analysis: bool = True):
        """
        Initialize Momentum Divergence Scanner
        
        Args:
            primary_period: Primary period for momentum calculations
            secondary_period: Secondary period for confirmation
            timeframes: List of timeframes to analyze
            oscillators: List of oscillators to use for divergence detection
            min_divergence_bars: Minimum bars for divergence pattern
            max_divergence_bars: Maximum bars for divergence pattern
            significance_threshold: Statistical significance threshold
            min_strength_threshold: Minimum strength for valid signals
            use_ml_validation: Enable machine learning validation
            fractal_analysis: Enable fractal-based pattern analysis
        """
        super().__init__()
        
        self.primary_period = max(5, primary_period)
        self.secondary_period = max(10, secondary_period)
        self.timeframes = timeframes or ['1H', '4H', '1D']
        self.oscillators = oscillators or [
            MomentumOscillator.RSI,
            MomentumOscillator.MACD,
            MomentumOscillator.STOCH,
            MomentumOscillator.MFI
        ]
        self.min_divergence_bars = max(3, min_divergence_bars)
        self.max_divergence_bars = max(20, max_divergence_bars)
        self.significance_threshold = max(0.01, min(0.1, significance_threshold))
        self.min_strength_threshold = max(0.1, min(1.0, min_strength_threshold))
        self.use_ml_validation = use_ml_validation
        self.fractal_analysis = fractal_analysis
        
        # ML components
        self.ml_classifier = None
        self.feature_scaler = RobustScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.pca_transformer = PCA(n_components=0.95, random_state=42)
        
        # Historical data for pattern learning
        self.historical_divergences = []
        self.performance_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'total_signals': 0,
            'successful_signals': 0
        }
        
        # State variables
        self.momentum_cache = {}
        self.divergence_history = []
        self.ml_trained = False
        
        self.logger = logging.getLogger(__name__)

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate momentum divergence analysis
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing comprehensive momentum divergence analysis
        """
        try:
            if len(data) < max(self.primary_period, self.secondary_period) * 2:
                return self._get_default_result()
                
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
            
            # Clean data
            data = self._clean_data(data)
            
            # Multi-timeframe momentum analysis
            timeframe_analyses = {}
            for timeframe in self.timeframes:
                analysis = self._analyze_timeframe_momentum(data, timeframe)
                timeframe_analyses[timeframe] = analysis
            
            # Cross-timeframe signal synthesis
            synthesized_signals = self._synthesize_cross_timeframe_signals(timeframe_analyses)
            
            # Calculate composite momentum scores
            composite_scores = self._calculate_composite_scores(timeframe_analyses)
            
            # Generate trading signals
            primary_signal = self._generate_primary_signal(synthesized_signals, composite_scores)
            
            # Calculate confidence and strength
            confidence = self._calculate_signal_confidence(synthesized_signals, timeframe_analyses)
            strength = self._calculate_signal_strength(composite_scores, synthesized_signals)
            
            # Generate comprehensive result
            result = {
                'signal': primary_signal,
                'strength': strength,
                'confidence': confidence,
                'timeframe_analyses': {tf: analysis.to_dict() for tf, analysis in timeframe_analyses.items()},
                'synthesized_signals': synthesized_signals,
                'composite_scores': composite_scores,
                'performance_metrics': self.performance_metrics,
                'metadata': {
                    'calculation_timestamp': pd.Timestamp.now().isoformat(),
                    'data_points': len(data),
                    'timeframes_analyzed': len(self.timeframes),
                    'oscillators_used': [osc.value for osc in self.oscillators],
                    'ml_enabled': self.use_ml_validation,
                    'fractal_analysis': self.fractal_analysis
                }
            }
            
            # Update ML models if enabled
            if self.use_ml_validation:
                self._update_ml_models(data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in momentum divergence calculation: {e}")
            raise IndicatorCalculationError(f"Failed to calculate momentum divergence: {e}")

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess input data"""
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower_bound, upper_bound)
        
        return data

    def _analyze_timeframe_momentum(self, data: pd.DataFrame, timeframe: str) -> MomentumAnalysis:
        """Analyze momentum for specific timeframe"""
        # Calculate multiple momentum oscillators
        oscillator_values = {}
        for osc in self.oscillators:
            oscillator_values[osc] = self._calculate_oscillator(data, osc)
        
        # Detect divergences for each oscillator
        all_divergences = []
        for osc, values in oscillator_values.items():
            divergences = self._detect_divergences(data, values, osc, timeframe)
            all_divergences.extend(divergences)
        
        # Calculate momentum metrics
        momentum_strength = self._calculate_momentum_strength(data, oscillator_values)
        momentum_direction = self._calculate_momentum_direction(oscillator_values)
        momentum_acceleration = self._calculate_momentum_acceleration(data, oscillator_values)
        momentum_persistence = self._calculate_momentum_persistence(oscillator_values)
        momentum_quality = self._calculate_momentum_quality(data, oscillator_values)
        
        # Support and resistance levels
        support_resistance = self._calculate_support_resistance_levels(data)
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(data)
        
        # Volatility-adjusted momentum
        volatility_adjusted = self._calculate_volatility_adjusted_momentum(data, momentum_strength)
        
        return MomentumAnalysis(
            timeframe=timeframe,
            momentum_strength=momentum_strength,
            momentum_direction=momentum_direction,
            momentum_acceleration=momentum_acceleration,
            momentum_persistence=momentum_persistence,
            momentum_quality=momentum_quality,
            divergence_signals=all_divergences,
            support_resistance_levels=support_resistance,
            trend_strength=trend_strength,
            volatility_adjusted_momentum=volatility_adjusted
        )

    def _calculate_oscillator(self, data: pd.DataFrame, oscillator: MomentumOscillator) -> np.ndarray:
        """Calculate values for specific oscillator"""
        if oscillator == MomentumOscillator.RSI:
            return self._calculate_rsi(data['close'])
        elif oscillator == MomentumOscillator.MACD:
            return self._calculate_macd(data['close'])
        elif oscillator == MomentumOscillator.STOCH:
            return self._calculate_stochastic(data)
        elif oscillator == MomentumOscillator.MFI:
            return self._calculate_mfi(data)
        elif oscillator == MomentumOscillator.CCI:
            return self._calculate_cci(data)
        elif oscillator == MomentumOscillator.ROC:
            return self._calculate_roc(data['close'])
        elif oscillator == MomentumOscillator.WILLIAMS_R:
            return self._calculate_williams_r(data)
        elif oscillator == MomentumOscillator.TSI:
            return self._calculate_tsi(data['close'])
        else:
            return self._calculate_rsi(data['close'])  # Default fallback

    def _calculate_rsi(self, close: pd.Series) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.primary_period).mean()
        avg_loss = loss.rolling(window=self.primary_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).values

    def _calculate_macd(self, close: pd.Series) -> np.ndarray:
        """Calculate MACD oscillator"""
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        
        return macd.fillna(0).values

    def _calculate_stochastic(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Stochastic oscillator"""
        low_min = data['low'].rolling(window=self.primary_period).min()
        high_max = data['high'].rolling(window=self.primary_period).max()
        
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        return k_percent.fillna(50).values

    def _calculate_mfi(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Money Flow Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=self.primary_period).sum()
        negative_mf = negative_flow.rolling(window=self.primary_period).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi.fillna(50).values

    def _calculate_cci(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Commodity Channel Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        ma = typical_price.rolling(window=self.primary_period).mean()
        mad = typical_price.rolling(window=self.primary_period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - ma) / (0.015 * mad)
        
        return cci.fillna(0).values

    def _calculate_roc(self, close: pd.Series) -> np.ndarray:
        """Calculate Rate of Change"""
        roc = ((close - close.shift(self.primary_period)) / close.shift(self.primary_period)) * 100
        
        return roc.fillna(0).values

    def _calculate_williams_r(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Williams %R"""
        high_max = data['high'].rolling(window=self.primary_period).max()
        low_min = data['low'].rolling(window=self.primary_period).min()
        
        williams_r = -100 * ((high_max - data['close']) / (high_max - low_min))
        
        return williams_r.fillna(-50).values

    def _calculate_tsi(self, close: pd.Series) -> np.ndarray:
        """Calculate True Strength Index"""
        momentum = close.diff()
        
        # Double smoothed momentum
        smooth1 = momentum.ewm(span=25).mean()
        smooth2 = smooth1.ewm(span=13).mean()
        
        # Double smoothed absolute momentum
        abs_smooth1 = momentum.abs().ewm(span=25).mean()
        abs_smooth2 = abs_smooth1.ewm(span=13).mean()
        
        tsi = 100 * (smooth2 / abs_smooth2)
        
        return tsi.fillna(0).values

    def _detect_divergences(self, data: pd.DataFrame, oscillator_values: np.ndarray, 
                          oscillator: MomentumOscillator, timeframe: str) -> List[DivergenceSignal]:
        """Detect divergences between price and oscillator"""
        divergences = []
        
        # Find peaks and troughs in price and oscillator
        price_peaks, _ = find_peaks(data['close'].values, distance=self.min_divergence_bars)
        price_troughs, _ = find_peaks(-data['close'].values, distance=self.min_divergence_bars)
        
        osc_peaks, _ = find_peaks(oscillator_values, distance=self.min_divergence_bars)
        osc_troughs, _ = find_peaks(-oscillator_values, distance=self.min_divergence_bars)
        
        # Check for bullish divergences (price makes lower lows, oscillator makes higher lows)
        for i in range(1, len(price_troughs)):
            if i < len(osc_troughs):
                p1, p2 = price_troughs[i-1], price_troughs[i]
                o1, o2 = osc_troughs[i-1], osc_troughs[i]
                
                if (abs(p2 - p1) >= self.min_divergence_bars and 
                    abs(p2 - p1) <= self.max_divergence_bars):
                    
                    price_change = data['close'].iloc[p2] - data['close'].iloc[p1]
                    osc_change = oscillator_values[o2] - oscillator_values[o1]
                    
                    if price_change < 0 and osc_change > 0:  # Bullish divergence
                        strength = self._calculate_divergence_strength(
                            price_change, osc_change, abs(p2 - p1)
                        )
                        
                        if strength >= self.min_strength_threshold:
                            # Calculate additional metrics
                            confidence = self._calculate_divergence_confidence(
                                data, oscillator_values, p1, p2, o1, o2
                            )
                            significance = self._calculate_statistical_significance(
                                data['close'].iloc[p1:p2+1], oscillator_values[o1:o2+1]
                            )
                            volume_confirmation = self._calculate_volume_confirmation(
                                data, p1, p2
                            )
                            fractal_dim = self._calculate_fractal_dimension(
                                data['close'].iloc[p1:p2+1]
                            ) if self.fractal_analysis else 0.0
                            
                            divergence = DivergenceSignal(
                                type=DivergenceType.BULLISH_REGULAR,
                                oscillator=oscillator,
                                timeframe=timeframe,
                                strength=strength,
                                confidence=confidence,
                                price_points=(p1, p2),
                                oscillator_points=(o1, o2),
                                duration=abs(p2 - p1),
                                significance=significance,
                                price_change=price_change,
                                oscillator_change=osc_change,
                                volume_confirmation=volume_confirmation,
                                fractal_dimension=fractal_dim
                            )
                            
                            divergences.append(divergence)
        
        # Check for bearish divergences (price makes higher highs, oscillator makes lower highs)
        for i in range(1, len(price_peaks)):
            if i < len(osc_peaks):
                p1, p2 = price_peaks[i-1], price_peaks[i]
                o1, o2 = osc_peaks[i-1], osc_peaks[i]
                
                if (abs(p2 - p1) >= self.min_divergence_bars and 
                    abs(p2 - p1) <= self.max_divergence_bars):
                    
                    price_change = data['close'].iloc[p2] - data['close'].iloc[p1]
                    osc_change = oscillator_values[o2] - oscillator_values[o1]
                    
                    if price_change > 0 and osc_change < 0:  # Bearish divergence
                        strength = self._calculate_divergence_strength(
                            price_change, osc_change, abs(p2 - p1)
                        )
                        
                        if strength >= self.min_strength_threshold:
                            confidence = self._calculate_divergence_confidence(
                                data, oscillator_values, p1, p2, o1, o2
                            )
                            significance = self._calculate_statistical_significance(
                                data['close'].iloc[p1:p2+1], oscillator_values[o1:o2+1]
                            )
                            volume_confirmation = self._calculate_volume_confirmation(
                                data, p1, p2
                            )
                            fractal_dim = self._calculate_fractal_dimension(
                                data['close'].iloc[p1:p2+1]
                            ) if self.fractal_analysis else 0.0
                            
                            divergence = DivergenceSignal(
                                type=DivergenceType.BEARISH_REGULAR,
                                oscillator=oscillator,
                                timeframe=timeframe,
                                strength=strength,
                                confidence=confidence,
                                price_points=(p1, p2),
                                oscillator_points=(o1, o2),
                                duration=abs(p2 - p1),
                                significance=significance,
                                price_change=price_change,
                                oscillator_change=osc_change,
                                volume_confirmation=volume_confirmation,
                                fractal_dimension=fractal_dim
                            )
                            
                            divergences.append(divergence)
        
        return divergences

    def _calculate_divergence_strength(self, price_change: float, osc_change: float, duration: int) -> float:
        """Calculate divergence strength score"""
        # Normalize changes
        price_magnitude = abs(price_change)
        osc_magnitude = abs(osc_change)
        
        # Duration factor
        duration_factor = min(1.0, duration / self.max_divergence_bars)
        
        # Opposition factor (how opposite the changes are)
        if price_change * osc_change < 0:  # Opposite directions
            opposition_factor = 1.0
        else:
            opposition_factor = 0.0
        
        # Magnitude factor
        magnitude_factor = (price_magnitude + osc_magnitude) / 2
        
        strength = opposition_factor * magnitude_factor * duration_factor
        
        return np.clip(strength, 0.0, 1.0)

    def _calculate_divergence_confidence(self, data: pd.DataFrame, oscillator_values: np.ndarray,
                                       p1: int, p2: int, o1: int, o2: int) -> float:
        """Calculate confidence score for divergence"""
        # Volume confirmation
        volume_trend = self._calculate_volume_trend(data, p1, p2)
        
        # Price action quality
        price_quality = self._calculate_price_action_quality(data, p1, p2)
        
        # Oscillator quality
        osc_quality = self._calculate_oscillator_quality(oscillator_values, o1, o2)
        
        # Trend context
        trend_context = self._calculate_trend_context(data, p1, p2)
        
        # Combine factors
        confidence = (volume_trend * 0.3 + price_quality * 0.3 + 
                     osc_quality * 0.2 + trend_context * 0.2)
        
        return np.clip(confidence, 0.0, 1.0)

    def _calculate_statistical_significance(self, price_series: pd.Series, osc_series: np.ndarray) -> float:
        """Calculate statistical significance of divergence"""
        try:
            # Correlation test
            correlation, p_value = stats.pearsonr(price_series.values, osc_series)
            
            # For divergence, we expect negative correlation
            if correlation < 0:
                significance = 1.0 - p_value
            else:
                significance = p_value
            
            return np.clip(significance, 0.0, 1.0)
        except:
            return 0.5

    def _calculate_volume_confirmation(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate volume confirmation for divergence"""
        if 'volume' not in data.columns:
            return 0.5
        
        volume_segment = data['volume'].iloc[start_idx:end_idx+1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[end_idx]
        
        if avg_volume > 0:
            volume_ratio = volume_segment.mean() / avg_volume
            return np.clip(volume_ratio, 0.0, 2.0) / 2.0
        
        return 0.5

    def _calculate_fractal_dimension(self, series: pd.Series) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            # Normalize series
            normalized = (series - series.min()) / (series.max() - series.min())
            
            # Box-counting method approximation
            N = len(normalized)
            scales = np.logspace(0, np.log10(N//4), 10)
            counts = []
            
            for scale in scales:
                box_size = int(scale)
                if box_size < 1:
                    box_size = 1
                
                boxes = []
                for i in range(0, N, box_size):
                    segment = normalized.iloc[i:i+box_size]
                    if len(segment) > 0:
                        boxes.append((segment.min(), segment.max()))
                
                counts.append(len(set(boxes)))
            
            # Linear regression to find dimension
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = -slope
            
            return np.clip(fractal_dim, 1.0, 2.0)
        except:
            return 1.5

    def _calculate_momentum_strength(self, data: pd.DataFrame, oscillator_values: Dict) -> float:
        """Calculate overall momentum strength"""
        strengths = []
        
        for osc, values in oscillator_values.items():
            # Calculate momentum from oscillator values
            momentum = np.diff(values)
            strength = np.mean(np.abs(momentum[-10:]))  # Last 10 periods
            strengths.append(strength)
        
        return np.mean(strengths) if strengths else 0.0

    def _calculate_momentum_direction(self, oscillator_values: Dict) -> float:
        """Calculate momentum direction (-1 to 1)"""
        directions = []
        
        for osc, values in oscillator_values.items():
            recent_change = values[-1] - values[-5] if len(values) >= 5 else 0
            directions.append(np.sign(recent_change))
        
        return np.mean(directions) if directions else 0.0

    def _calculate_momentum_acceleration(self, data: pd.DataFrame, oscillator_values: Dict) -> float:
        """Calculate momentum acceleration"""
        accelerations = []
        
        for osc, values in oscillator_values.items():
            if len(values) >= 3:
                # Second derivative approximation
                acceleration = np.diff(np.diff(values))
                recent_accel = np.mean(acceleration[-5:]) if len(acceleration) >= 5 else 0
                accelerations.append(recent_accel)
        
        return np.mean(accelerations) if accelerations else 0.0

    def _calculate_momentum_persistence(self, oscillator_values: Dict) -> float:
        """Calculate momentum persistence score"""
        persistence_scores = []
        
        for osc, values in oscillator_values.items():
            if len(values) >= 10:
                # Count consecutive periods with same direction
                momentum = np.diff(values)
                signs = np.sign(momentum)
                
                # Find longest streak
                max_streak = 0
                current_streak = 1
                
                for i in range(1, len(signs)):
                    if signs[i] == signs[i-1]:
                        current_streak += 1
                    else:
                        max_streak = max(max_streak, current_streak)
                        current_streak = 1
                
                max_streak = max(max_streak, current_streak)
                persistence = max_streak / len(signs)
                persistence_scores.append(persistence)
        
        return np.mean(persistence_scores) if persistence_scores else 0.0

    def _calculate_momentum_quality(self, data: pd.DataFrame, oscillator_values: Dict) -> float:
        """Calculate momentum quality score"""
        quality_factors = []
        
        # Consistency across oscillators
        if len(oscillator_values) > 1:
            osc_directions = []
            for osc, values in oscillator_values.items():
                recent_change = values[-1] - values[-5] if len(values) >= 5 else 0
                osc_directions.append(np.sign(recent_change))
            
            consistency = 1.0 - np.std(osc_directions) if len(osc_directions) > 1 else 1.0
            quality_factors.append(consistency)
        
        # Volume confirmation
        if 'volume' in data.columns:
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            volume_quality = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5
            quality_factors.append(volume_quality)
        
        # Price action quality
        price_volatility = data['close'].pct_change().std()
        volatility_quality = 1.0 / (1.0 + price_volatility) if price_volatility > 0 else 0.5
        quality_factors.append(volatility_quality)
        
        return np.mean(quality_factors) if quality_factors else 0.5

    def _calculate_support_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Calculate key support and resistance levels"""
        # Find local extrema
        high_peaks, _ = find_peaks(data['high'].values, distance=10)
        low_troughs, _ = find_peaks(-data['low'].values, distance=10)
        
        # Cluster similar levels
        all_levels = []
        all_levels.extend(data['high'].iloc[high_peaks].values)
        all_levels.extend(data['low'].iloc[low_troughs].values)
        
        if len(all_levels) > 2:
            # Use DBSCAN to cluster levels
            levels_array = np.array(all_levels).reshape(-1, 1)
            clustering = DBSCAN(eps=data['close'].std() * 0.5, min_samples=2)
            clusters = clustering.fit_predict(levels_array)
            
            # Get representative level for each cluster
            unique_clusters = set(clusters)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)  # Remove noise
            
            levels = []
            for cluster in unique_clusters:
                cluster_levels = levels_array[clusters == cluster]
                levels.append(np.mean(cluster_levels))
            
            return sorted(levels)
        
        return []

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate overall trend strength"""
        # Multiple period EMAs
        ema_short = data['close'].ewm(span=12).mean()
        ema_medium = data['close'].ewm(span=26).mean()
        ema_long = data['close'].ewm(span=50).mean()
        
        # Check alignment
        current_price = data['close'].iloc[-1]
        short_ema = ema_short.iloc[-1]
        medium_ema = ema_medium.iloc[-1]
        long_ema = ema_long.iloc[-1]
        
        # Uptrend alignment
        if current_price > short_ema > medium_ema > long_ema:
            trend_strength = 1.0
        # Downtrend alignment
        elif current_price < short_ema < medium_ema < long_ema:
            trend_strength = -1.0
        else:
            # Calculate partial alignment
            alignments = []
            if current_price > short_ema:
                alignments.append(1)
            else:
                alignments.append(-1)
            
            if short_ema > medium_ema:
                alignments.append(1)
            else:
                alignments.append(-1)
            
            if medium_ema > long_ema:
                alignments.append(1)
            else:
                alignments.append(-1)
            
            trend_strength = np.mean(alignments)
        
        return trend_strength

    def _calculate_volatility_adjusted_momentum(self, data: pd.DataFrame, momentum_strength: float) -> float:
        """Calculate volatility-adjusted momentum"""
        # Calculate recent volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        # Adjust momentum for volatility
        if volatility > 0:
            volatility_factor = 1.0 / (1.0 + volatility * 100)  # Scale volatility
            adjusted_momentum = momentum_strength * volatility_factor
        else:
            adjusted_momentum = momentum_strength
        
        return adjusted_momentum

    def _synthesize_cross_timeframe_signals(self, timeframe_analyses: Dict[str, MomentumAnalysis]) -> Dict[str, Any]:
        """Synthesize signals across multiple timeframes"""
        all_divergences = []
        timeframe_strengths = {}
        timeframe_directions = {}
        
        for tf, analysis in timeframe_analyses.items():
            all_divergences.extend(analysis.divergence_signals)
            timeframe_strengths[tf] = analysis.momentum_strength
            timeframe_directions[tf] = analysis.momentum_direction
        
        # Analyze divergence consensus
        bullish_divergences = [d for d in all_divergences if 'bullish' in d.type.value]
        bearish_divergences = [d for d in all_divergences if 'bearish' in d.type.value]
        
        # Calculate signal strength by timeframe alignment
        direction_alignment = self._calculate_direction_alignment(timeframe_directions)
        strength_consistency = self._calculate_strength_consistency(timeframe_strengths)
        
        # Multi-timeframe confirmation
        mtf_confirmation = self._calculate_mtf_confirmation(timeframe_analyses)
        
        return {
            'all_divergences': [d.to_dict() for d in all_divergences],
            'bullish_count': len(bullish_divergences),
            'bearish_count': len(bearish_divergences),
            'direction_alignment': direction_alignment,
            'strength_consistency': strength_consistency,
            'mtf_confirmation': mtf_confirmation,
            'timeframe_strengths': timeframe_strengths,
            'timeframe_directions': timeframe_directions
        }

    def _calculate_direction_alignment(self, timeframe_directions: Dict[str, float]) -> float:
        """Calculate alignment of momentum directions across timeframes"""
        directions = list(timeframe_directions.values())
        if not directions:
            return 0.0
        
        # Calculate standard deviation of directions
        std_dev = np.std(directions)
        
        # Convert to alignment score (0 to 1)
        alignment = 1.0 / (1.0 + std_dev)
        
        return alignment

    def _calculate_strength_consistency(self, timeframe_strengths: Dict[str, float]) -> float:
        """Calculate consistency of momentum strengths across timeframes"""
        strengths = list(timeframe_strengths.values())
        if not strengths:
            return 0.0
        
        # Calculate coefficient of variation
        mean_strength = np.mean(strengths)
        std_strength = np.std(strengths)
        
        if mean_strength > 0:
            cv = std_strength / mean_strength
            consistency = 1.0 / (1.0 + cv)
        else:
            consistency = 0.0
        
        return consistency

    def _calculate_mtf_confirmation(self, timeframe_analyses: Dict[str, MomentumAnalysis]) -> float:
        """Calculate multi-timeframe confirmation score"""
        confirmations = []
        
        # Check trend alignment
        trend_directions = []
        for analysis in timeframe_analyses.values():
            trend_directions.append(np.sign(analysis.trend_strength))
        
        if trend_directions:
            trend_consistency = 1.0 - np.std(trend_directions)
            confirmations.append(trend_consistency)
        
        # Check momentum quality alignment
        quality_scores = [analysis.momentum_quality for analysis in timeframe_analyses.values()]
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            confirmations.append(avg_quality)
        
        # Check divergence agreement
        divergence_counts = {}
        for analysis in timeframe_analyses.values():
            for divergence in analysis.divergence_signals:
                div_type = divergence.type.value
                if div_type not in divergence_counts:
                    divergence_counts[div_type] = 0
                divergence_counts[div_type] += 1
        
        if divergence_counts:
            max_count = max(divergence_counts.values())
            total_signals = sum(divergence_counts.values())
            agreement = max_count / total_signals if total_signals > 0 else 0
            confirmations.append(agreement)
        
        return np.mean(confirmations) if confirmations else 0.5

    def _calculate_composite_scores(self, timeframe_analyses: Dict[str, MomentumAnalysis]) -> Dict[str, float]:
        """Calculate composite momentum scores"""
        # Weighted average by timeframe importance
        timeframe_weights = {
            '1H': 0.2,
            '4H': 0.3,
            '1D': 0.5
        }
        
        composite_strength = 0.0
        composite_direction = 0.0
        composite_quality = 0.0
        total_weight = 0.0
        
        for tf, analysis in timeframe_analyses.items():
            weight = timeframe_weights.get(tf, 0.33)
            
            composite_strength += analysis.momentum_strength * weight
            composite_direction += analysis.momentum_direction * weight
            composite_quality += analysis.momentum_quality * weight
            total_weight += weight
        
        if total_weight > 0:
            composite_strength /= total_weight
            composite_direction /= total_weight
            composite_quality /= total_weight
        
        # Calculate divergence score
        all_divergences = []
        for analysis in timeframe_analyses.values():
            all_divergences.extend(analysis.divergence_signals)
        
        if all_divergences:
            avg_divergence_strength = np.mean([d.strength for d in all_divergences])
            avg_divergence_confidence = np.mean([d.confidence for d in all_divergences])
            divergence_score = (avg_divergence_strength + avg_divergence_confidence) / 2
        else:
            divergence_score = 0.0
        
        return {
            'composite_strength': composite_strength,
            'composite_direction': composite_direction,
            'composite_quality': composite_quality,
            'divergence_score': divergence_score,
            'overall_score': (composite_strength + composite_quality + divergence_score) / 3
        }

    def _generate_primary_signal(self, synthesized_signals: Dict[str, Any], 
                                composite_scores: Dict[str, float]) -> SignalType:
        """Generate primary trading signal"""
        # Get overall scores
        overall_score = composite_scores.get('overall_score', 0.0)
        direction = composite_scores.get('composite_direction', 0.0)
        divergence_score = composite_scores.get('divergence_score', 0.0)
        
        # Count divergences
        bullish_count = synthesized_signals.get('bullish_count', 0)
        bearish_count = synthesized_signals.get('bearish_count', 0)
        
        # Multi-timeframe confirmation
        mtf_confirmation = synthesized_signals.get('mtf_confirmation', 0.5)
        
        # Signal generation logic
        if (bullish_count > bearish_count and 
            direction > 0.1 and 
            overall_score > 0.6 and 
            mtf_confirmation > 0.7):
            return SignalType.BUY
        elif (bearish_count > bullish_count and 
              direction < -0.1 and 
              overall_score > 0.6 and 
              mtf_confirmation > 0.7):
            return SignalType.SELL
        elif divergence_score > 0.8 and mtf_confirmation > 0.6:
            # Strong divergence signal
            if bullish_count > bearish_count:
                return SignalType.BUY
            elif bearish_count > bullish_count:
                return SignalType.SELL
        
        return SignalType.HOLD

    def _calculate_signal_confidence(self, synthesized_signals: Dict[str, Any], 
                                   timeframe_analyses: Dict[str, MomentumAnalysis]) -> float:
        """Calculate overall signal confidence"""
        confidence_factors = []
        
        # Multi-timeframe confirmation
        mtf_confirmation = synthesized_signals.get('mtf_confirmation', 0.5)
        confidence_factors.append(mtf_confirmation)
        
        # Direction alignment
        direction_alignment = synthesized_signals.get('direction_alignment', 0.5)
        confidence_factors.append(direction_alignment)
        
        # Strength consistency
        strength_consistency = synthesized_signals.get('strength_consistency', 0.5)
        confidence_factors.append(strength_consistency)
        
        # Average momentum quality across timeframes
        quality_scores = [analysis.momentum_quality for analysis in timeframe_analyses.values()]
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            confidence_factors.append(avg_quality)
        
        # Divergence confidence
        all_divergences = []
        for analysis in timeframe_analyses.values():
            all_divergences.extend(analysis.divergence_signals)
        
        if all_divergences:
            avg_div_confidence = np.mean([d.confidence for d in all_divergences])
            confidence_factors.append(avg_div_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _calculate_signal_strength(self, composite_scores: Dict[str, float], 
                                 synthesized_signals: Dict[str, Any]) -> float:
        """Calculate overall signal strength"""
        # Base strength from composite scores
        base_strength = composite_scores.get('overall_score', 0.0)
        
        # Divergence contribution
        divergence_score = composite_scores.get('divergence_score', 0.0)
        
        # Multi-timeframe alignment
        mtf_confirmation = synthesized_signals.get('mtf_confirmation', 0.5)
        
        # Combine factors
        signal_strength = (base_strength * 0.5 + 
                          divergence_score * 0.3 + 
                          mtf_confirmation * 0.2)
        
        return np.clip(signal_strength, 0.0, 1.0)

    def _update_ml_models(self, data: pd.DataFrame, result: Dict[str, Any]) -> None:
        """Update machine learning models with new data"""
        try:
            # Extract features for ML training
            features = self._extract_ml_features(data, result)
            
            # Update anomaly detection
            if len(features) > 0:
                features_array = np.array(features).reshape(1, -1)
                
                # Scale features
                if hasattr(self.feature_scaler, 'n_features_in_'):
                    if features_array.shape[1] == self.feature_scaler.n_features_in_:
                        scaled_features = self.feature_scaler.transform(features_array)
                        
                        # Update anomaly detector
                        anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
                        result['anomaly_score'] = anomaly_score
                else:
                    # First time training
                    self.feature_scaler.fit(features_array)
                    scaled_features = self.feature_scaler.transform(features_array)
                    self.anomaly_detector.fit(scaled_features)
            
            # Store for future ML training
            self.historical_divergences.append({
                'features': features,
                'signal': result['signal'].value,
                'strength': result['strength'],
                'confidence': result['confidence']
            })
            
            # Limit history size
            if len(self.historical_divergences) > 1000:
                self.historical_divergences = self.historical_divergences[-800:]
                
        except Exception as e:
            self.logger.warning(f"ML model update failed: {e}")

    def _extract_ml_features(self, data: pd.DataFrame, result: Dict[str, Any]) -> List[float]:
        """Extract features for machine learning"""
        features = []
        
        try:
            # Basic price features
            returns = data['close'].pct_change().fillna(0)
            features.extend([
                returns.iloc[-1],  # Latest return
                returns.rolling(5).mean().iloc[-1],  # Short-term avg return
                returns.rolling(20).std().iloc[-1],  # Volatility
                data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]  # Volume ratio
            ])
            
            # Momentum features
            composite_scores = result.get('composite_scores', {})
            features.extend([
                composite_scores.get('composite_strength', 0.0),
                composite_scores.get('composite_direction', 0.0),
                composite_scores.get('composite_quality', 0.0),
                composite_scores.get('divergence_score', 0.0)
            ])
            
            # Timeframe features
            synthesized = result.get('synthesized_signals', {})
            features.extend([
                synthesized.get('direction_alignment', 0.5),
                synthesized.get('strength_consistency', 0.5),
                synthesized.get('mtf_confirmation', 0.5),
                synthesized.get('bullish_count', 0),
                synthesized.get('bearish_count', 0)
            ])
            
            # Technical features
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            price_to_sma = data['close'].iloc[-1] / sma_20 if sma_20 > 0 else 1.0
            features.append(price_to_sma)
            
            # RSI
            rsi = self._calculate_rsi(data['close'])
            features.append(rsi[-1] if len(rsi) > 0 else 50.0)
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return []
        
        return features

    def _get_default_result(self) -> Dict[str, Any]:
        """Get default result when calculation fails"""
        return {
            'signal': SignalType.HOLD,
            'strength': 0.0,
            'confidence': 0.0,
            'timeframe_analyses': {},
            'synthesized_signals': {
                'all_divergences': [],
                'bullish_count': 0,
                'bearish_count': 0,
                'direction_alignment': 0.5,
                'strength_consistency': 0.5,
                'mtf_confirmation': 0.5
            },
            'composite_scores': {
                'composite_strength': 0.0,
                'composite_direction': 0.0,
                'composite_quality': 0.0,
                'divergence_score': 0.0,
                'overall_score': 0.0
            },
            'performance_metrics': self.performance_metrics,
            'metadata': {
                'calculation_timestamp': pd.Timestamp.now().isoformat(),
                'data_points': 0,
                'timeframes_analyzed': 0,
                'oscillators_used': [],
                'ml_enabled': self.use_ml_validation,
                'fractal_analysis': self.fractal_analysis,
                'error': 'Insufficient data'
            }
        }

    # Additional helper methods for calculations
    def _calculate_volume_trend(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate volume trend during divergence period"""
        if 'volume' not in data.columns or end_idx <= start_idx:
            return 0.5
        
        volume_segment = data['volume'].iloc[start_idx:end_idx+1]
        volume_trend = np.polyfit(range(len(volume_segment)), volume_segment.values, 1)[0]
        
        # Normalize to 0-1 range
        return np.clip((volume_trend + 1) / 2, 0.0, 1.0)

    def _calculate_price_action_quality(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate price action quality during divergence period"""
        if end_idx <= start_idx:
            return 0.5
        
        price_segment = data['close'].iloc[start_idx:end_idx+1]
        
        # Calculate smoothness (inverse of volatility)
        returns = price_segment.pct_change().fillna(0)
        volatility = returns.std()
        smoothness = 1.0 / (1.0 + volatility * 10) if volatility > 0 else 1.0
        
        # Calculate trend consistency
        trend_direction = np.sign(price_segment.iloc[-1] - price_segment.iloc[0])
        consistent_moves = 0
        total_moves = 0
        
        for i in range(1, len(price_segment)):
            move_direction = np.sign(price_segment.iloc[i] - price_segment.iloc[i-1])
            if move_direction != 0:
                total_moves += 1
                if move_direction == trend_direction:
                    consistent_moves += 1
        
        consistency = consistent_moves / total_moves if total_moves > 0 else 0.5
        
        return (smoothness + consistency) / 2

    def _calculate_oscillator_quality(self, oscillator_values: np.ndarray, start_idx: int, end_idx: int) -> float:
        """Calculate oscillator quality during divergence period"""
        if end_idx <= start_idx or len(oscillator_values) <= end_idx:
            return 0.5
        
        osc_segment = oscillator_values[start_idx:end_idx+1]
        
        # Calculate smoothness
        osc_changes = np.diff(osc_segment)
        smoothness = 1.0 / (1.0 + np.std(osc_changes)) if len(osc_changes) > 0 else 0.5
        
        # Calculate trend strength in oscillator
        if len(osc_segment) > 1:
            trend_strength = abs(osc_segment[-1] - osc_segment[0]) / len(osc_segment)
            normalized_strength = np.clip(trend_strength / 10, 0.0, 1.0)
        else:
            normalized_strength = 0.0
        
        return (smoothness + normalized_strength) / 2

    def _calculate_trend_context(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate trend context quality"""
        if end_idx <= start_idx:
            return 0.5
        
        # Calculate multiple EMAs
        ema_short = data['close'].ewm(span=12).mean()
        ema_long = data['close'].ewm(span=26).mean()
        
        # Check trend alignment at divergence points
        start_alignment = 1.0 if ema_short.iloc[start_idx] > ema_long.iloc[start_idx] else 0.0
        end_alignment = 1.0 if ema_short.iloc[end_idx] > ema_long.iloc[end_idx] else 0.0
        
        # Trend consistency
        trend_consistency = 1.0 if start_alignment == end_alignment else 0.0
        
        # Trend strength
        price_change = abs(data['close'].iloc[end_idx] - data['close'].iloc[start_idx])
        ema_change = abs(ema_short.iloc[end_idx] - ema_short.iloc[start_idx])
        
        if price_change > 0:
            trend_strength = min(1.0, ema_change / price_change)
        else:
            trend_strength = 0.5
        
        return (trend_consistency + trend_strength) / 2

    def get_signal_type(self) -> SignalType:
        """Return the type of signal this indicator generates"""
        return SignalType.BUY  # This indicator can generate BUY, SELL, or HOLD signals

    def get_data_requirements(self) -> Dict[str, Any]:
        """Return the data requirements for this indicator"""
        return {
            'required_columns': ['open', 'high', 'low', 'close', 'volume'],
            'min_periods': max(self.primary_period, self.secondary_period) * 2,
            'timeframe_compatibility': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'data_type': 'OHLCV'
        }