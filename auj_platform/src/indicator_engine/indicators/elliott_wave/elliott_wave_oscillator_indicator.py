"""
Elliott Wave Oscillator Indicator

Advanced implementation with machine learning-enhanced Elliott Wave theory to detect
wave momentum and oscillations for the humanitarian trading platform.

This indicator combines traditional Elliott Wave analysis with modern machine learning
techniques to identify wave structures and momentum patterns with high accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.signal import find_peaks, hilbert
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class WaveOscillatorConfig:
    """Configuration for Elliott Wave Oscillator."""
    short_period: int = 5
    long_period: int = 35
    signal_period: int = 5
    ml_lookback: int = 100
    confidence_threshold: float = 0.7
    momentum_weight: float = 0.4
    volume_weight: float = 0.3
    price_weight: float = 0.3


@dataclass
class WaveFeatures:
    """Features extracted for wave analysis."""
    momentum: float
    acceleration: float
    volume_momentum: float
    price_momentum: float
    volatility: float
    trend_strength: float
    wave_count: int
    fibonacci_level: float


class ElliottWaveOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Elliott Wave Oscillator with ML-enhanced wave detection.
    
    This indicator provides sophisticated analysis of Elliott Wave patterns
    using machine learning to enhance traditional oscillator calculations.
    """
    
    def __init__(self, config: Optional[WaveOscillatorConfig] = None):
        super().__init__()
        self.config = config or WaveOscillatorConfig()
        self.logger = logging.getLogger(__name__)
        
        # ML components
        self.wave_classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        # Historical data storage
        self.historical_features: List[WaveFeatures] = []
        self.historical_signals: List[float] = []
        
        # Wave analysis components
        self.fibonacci_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618]
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Elliott Wave Oscillator with advanced ML integration.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing oscillator values and signals
        """
        try:
            if len(data) < max(self.config.long_period, self.config.ml_lookback):
                raise IndicatorCalculationError(
                    f"Insufficient data: need at least {max(self.config.long_period, self.config.ml_lookback)} periods"
                )
            
            # Calculate base oscillator
            oscillator = self._calculate_base_oscillator(data)
            
            # Extract advanced features
            features = self._extract_wave_features(data)
            
            # Apply ML enhancement if available
            ml_signal = self._apply_ml_enhancement(features, data)
            
            # Generate final signals
            signals = self._generate_signals(oscillator, ml_signal, data)
            
            # Calculate wave structure analysis
            wave_analysis = self._analyze_wave_structure(data, oscillator)
            
            # Update ML training data
            self._update_training_data(features, oscillator)
            
            result = {
                'oscillator': oscillator,
                'signal_line': signals['signal_line'],
                'histogram': signals['histogram'],
                'wave_direction': signals['wave_direction'],
                'wave_strength': signals['wave_strength'],
                'ml_confidence': ml_signal['confidence'],
                'ml_prediction': ml_signal['prediction'],
                'wave_count': wave_analysis['wave_count'],
                'current_wave': wave_analysis['current_wave'],
                'wave_completion': wave_analysis['completion_ratio'],
                'fibonacci_projection': wave_analysis['fibonacci_projection'],
                'momentum_divergence': self._detect_momentum_divergence(data, oscillator),
                'volume_confirmation': self._analyze_volume_confirmation(data, oscillator),
                'volatility_adjustment': self._calculate_volatility_adjustment(data),
                'signal_type': self._determine_signal_type(signals),
                'raw_data': {
                    'short_ema': self._calculate_ema(data['close'], self.config.short_period),
                    'long_ema': self._calculate_ema(data['close'], self.config.long_period),
                    'features': features.__dict__ if features else None
                }
            }
            
            self.logger.info(f"Elliott Wave Oscillator calculated successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Elliott Wave Oscillator: {str(e)}")
            raise IndicatorCalculationError(f"Elliott Wave Oscillator calculation failed: {str(e)}")
    
    def _calculate_base_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the base Elliott Wave Oscillator."""
        # Calculate EMAs
        short_ema = self._calculate_ema(data['close'], self.config.short_period)
        long_ema = self._calculate_ema(data['close'], self.config.long_period)
        
        # Base oscillator (difference between EMAs)
        oscillator = short_ema - long_ema
        
        # Apply volume weighting
        volume_factor = self._calculate_volume_factor(data)
        oscillator = oscillator * volume_factor
        
        # Normalize the oscillator
        oscillator = self._normalize_oscillator(oscillator)
        
        return oscillator
    
    def _extract_wave_features(self, data: pd.DataFrame) -> Optional[WaveFeatures]:
        """Extract sophisticated features for ML analysis."""
        try:
            if len(data) < self.config.ml_lookback:
                return None
                
            recent_data = data.tail(self.config.ml_lookback)
            
            # Calculate momentum features
            momentum = self._calculate_momentum_features(recent_data)
            acceleration = self._calculate_acceleration(recent_data['close'])
            
            # Volume analysis
            volume_momentum = self._calculate_volume_momentum(recent_data)
            
            # Price momentum
            price_momentum = self._calculate_price_momentum(recent_data)
            
            # Volatility analysis
            volatility = self._calculate_volatility(recent_data)
            
            # Trend strength
            trend_strength = self._calculate_trend_strength(recent_data)
            
            # Wave counting
            wave_count = self._count_waves(recent_data)
            
            # Fibonacci analysis
            fibonacci_level = self._analyze_fibonacci_levels(recent_data)
            
            return WaveFeatures(
                momentum=momentum,
                acceleration=acceleration,
                volume_momentum=volume_momentum,
                price_momentum=price_momentum,
                volatility=volatility,
                trend_strength=trend_strength,
                wave_count=wave_count,
                fibonacci_level=fibonacci_level
            )
            
        except Exception as e:
            self.logger.warning(f"Error extracting wave features: {str(e)}")
            return None
    
    def _apply_ml_enhancement(self, features: Optional[WaveFeatures], data: pd.DataFrame) -> Dict[str, float]:
        """Apply machine learning enhancement to the oscillator."""
        if not features or not self.is_trained:
            return {'prediction': 0.0, 'confidence': 0.0}
        
        try:
            # Prepare feature vector
            feature_vector = np.array([
                features.momentum,
                features.acceleration,
                features.volume_momentum,
                features.price_momentum,
                features.volatility,
                features.trend_strength,
                features.wave_count,
                features.fibonacci_level
            ]).reshape(1, -1)
            
            # Scale features
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Make prediction
            if self.wave_classifier:
                prediction_proba = self.wave_classifier.predict_proba(feature_vector)[0]
                prediction = self.wave_classifier.predict(feature_vector)[0]
                confidence = max(prediction_proba)
                
                return {
                    'prediction': float(prediction),
                    'confidence': float(confidence)
                }
            
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {str(e)}")
        
        return {'prediction': 0.0, 'confidence': 0.0}
    
    def _generate_signals(self, oscillator: pd.Series, ml_signal: Dict[str, float], 
                         data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate trading signals with ML enhancement."""
        # Signal line (EMA of oscillator)
        signal_line = self._calculate_ema(oscillator, self.config.signal_period)
        
        # Histogram (difference between oscillator and signal line)
        histogram = oscillator - signal_line
        
        # Wave direction analysis
        wave_direction = self._analyze_wave_direction(oscillator, ml_signal)
        
        # Wave strength calculation
        wave_strength = self._calculate_wave_strength(oscillator, data)
        
        return {
            'signal_line': signal_line,
            'histogram': histogram,
            'wave_direction': wave_direction,
            'wave_strength': wave_strength
        }
    
    def _analyze_wave_structure(self, data: pd.DataFrame, oscillator: pd.Series) -> Dict[str, Any]:
        """Analyze Elliott Wave structure."""
        try:
            # Find peaks and troughs
            peaks, _ = find_peaks(oscillator.values, distance=5)
            troughs, _ = find_peaks(-oscillator.values, distance=5)
            
            # Combine and sort turning points
            turning_points = np.concatenate([peaks, troughs])
            turning_points = np.sort(turning_points)
            
            # Count waves
            wave_count = len(turning_points)
            
            # Determine current wave
            current_wave = self._determine_current_wave(turning_points, len(oscillator))
            
            # Calculate completion ratio
            completion_ratio = self._calculate_completion_ratio(turning_points, len(oscillator))
            
            # Fibonacci projection
            fibonacci_projection = self._calculate_fibonacci_projection(data, turning_points)
            
            return {
                'wave_count': wave_count,
                'current_wave': current_wave,
                'completion_ratio': completion_ratio,
                'fibonacci_projection': fibonacci_projection,
                'turning_points': turning_points.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Wave structure analysis failed: {str(e)}")
            return {'wave_count': 0, 'current_wave': 0, 'completion_ratio': 0.0, 'fibonacci_projection': 0.0}
    
    def _calculate_momentum_features(self, data: pd.DataFrame) -> float:
        """Calculate momentum features for ML."""
        close_prices = data['close'].values
        momentum = np.mean(np.diff(close_prices))
        return float(momentum)
    
    def _calculate_acceleration(self, prices: pd.Series) -> float:
        """Calculate price acceleration."""
        velocity = np.diff(prices.values)
        acceleration = np.mean(np.diff(velocity)) if len(velocity) > 1 else 0.0
        return float(acceleration)
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> float:
        """Calculate volume-weighted momentum."""
        volume_weighted_price = (data['close'] * data['volume']).sum() / data['volume'].sum()
        avg_price = data['close'].mean()
        return float(volume_weighted_price - avg_price)
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum using ROC."""
        roc_period = min(14, len(data) - 1)
        if roc_period <= 0:
            return 0.0
        roc = (data['close'].iloc[-1] - data['close'].iloc[-roc_period-1]) / data['close'].iloc[-roc_period-1]
        return float(roc)
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility measure."""
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 1 else 0.0
        return float(volatility)
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength."""
        close_prices = data['close'].values
        if len(close_prices) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(close_prices))
        slope = np.polyfit(x, close_prices, 1)[0]
        
        # Normalize by price level
        trend_strength = slope / close_prices[-1] if close_prices[-1] != 0 else 0.0
        return float(trend_strength)
    
    def _count_waves(self, data: pd.DataFrame) -> int:
        """Count Elliott waves in the data."""
        prices = data['close'].values
        peaks, _ = find_peaks(prices, distance=3)
        troughs, _ = find_peaks(-prices, distance=3)
        return len(peaks) + len(troughs)
    
    def _analyze_fibonacci_levels(self, data: pd.DataFrame) -> float:
        """Analyze current position relative to Fibonacci levels."""
        high = data['high'].max()
        low = data['low'].min()
        current = data['close'].iloc[-1]
        
        if high == low:
            return 0.5
        
        # Calculate retracement level
        retracement = (current - low) / (high - low)
        
        # Find closest Fibonacci level
        closest_fib = min(self.fibonacci_levels, key=lambda x: abs(x - retracement))
        return float(closest_fib)
    
    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    def _calculate_volume_factor(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume adjustment factor."""
        volume_ma = data['volume'].rolling(window=20).mean()
        volume_ratio = data['volume'] / volume_ma
        # Normalize between 0.5 and 2.0
        return np.clip(volume_ratio, 0.5, 2.0)
    
    def _normalize_oscillator(self, oscillator: pd.Series) -> pd.Series:
        """Normalize oscillator values."""
        rolling_std = oscillator.rolling(window=50).std()
        rolling_mean = oscillator.rolling(window=50).mean()
        
        # Z-score normalization with rolling statistics
        normalized = (oscillator - rolling_mean) / (rolling_std + 1e-8)
        return normalized.fillna(0)
    
    def _analyze_wave_direction(self, oscillator: pd.Series, ml_signal: Dict[str, float]) -> pd.Series:
        """Analyze wave direction with ML enhancement."""
        # Basic direction from oscillator slope
        direction = np.sign(oscillator.diff())
        
        # Apply ML enhancement if confident
        if ml_signal['confidence'] > self.config.confidence_threshold:
            ml_direction = np.sign(ml_signal['prediction'])
            # Blend traditional and ML signals
            direction = 0.7 * direction + 0.3 * ml_direction
        
        return pd.Series(direction, index=oscillator.index)
    
    def _calculate_wave_strength(self, oscillator: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate wave strength indicator."""
        # Combine oscillator magnitude with volume
        abs_oscillator = np.abs(oscillator)
        volume_strength = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Weighted strength calculation
        strength = (
            self.config.momentum_weight * abs_oscillator +
            self.config.volume_weight * volume_strength.fillna(1) +
            self.config.price_weight * np.abs(data['close'].pct_change()).fillna(0)
        )
        
        return strength
    
    def _determine_current_wave(self, turning_points: np.ndarray, total_length: int) -> int:
        """Determine current Elliott wave number."""
        if len(turning_points) == 0:
            return 1
        
        # Simple wave counting (1-5 impulse, A-C correction)
        waves_since_major = len(turning_points) % 8  # 5 impulse + 3 correction
        return max(1, waves_since_major)
    
    def _calculate_completion_ratio(self, turning_points: np.ndarray, total_length: int) -> float:
        """Calculate wave completion ratio."""
        if len(turning_points) == 0:
            return 0.0
        
        last_turning_point = turning_points[-1]
        distance_from_last = total_length - last_turning_point - 1
        
        if len(turning_points) >= 2:
            avg_wave_length = np.mean(np.diff(turning_points))
            completion = distance_from_last / avg_wave_length if avg_wave_length > 0 else 0.0
        else:
            completion = distance_from_last / total_length
        
        return min(1.0, max(0.0, completion))
    
    def _calculate_fibonacci_projection(self, data: pd.DataFrame, turning_points: np.ndarray) -> float:
        """Calculate Fibonacci projection for next target."""
        if len(turning_points) < 3:
            return 0.0
        
        # Use last three turning points for projection
        recent_points = turning_points[-3:]
        prices = data['close'].iloc[recent_points].values
        
        if len(prices) >= 3:
            # Calculate potential Fibonacci projection
            wave_a = abs(prices[1] - prices[0])
            wave_b = abs(prices[2] - prices[1])
            
            # Common Fibonacci ratios for projections
            projection_ratios = [0.618, 1.000, 1.618]
            avg_projection = np.mean([wave_a * ratio for ratio in projection_ratios])
            
            return float(avg_projection)
        
        return 0.0
    
    def _detect_momentum_divergence(self, data: pd.DataFrame, oscillator: pd.Series) -> bool:
        """Detect momentum divergence between price and oscillator."""
        if len(data) < 20:
            return False
        
        # Get recent peaks in price and oscillator
        price_peaks, _ = find_peaks(data['close'].values[-20:], distance=5)
        osc_peaks, _ = find_peaks(oscillator.values[-20:], distance=5)
        
        if len(price_peaks) >= 2 and len(osc_peaks) >= 2:
            # Check for divergence
            price_trend = data['close'].iloc[-20:].iloc[price_peaks[-1]] > data['close'].iloc[-20:].iloc[price_peaks[-2]]
            osc_trend = oscillator.iloc[-20:].iloc[osc_peaks[-1]] > oscillator.iloc[-20:].iloc[osc_peaks[-2]]
            
            return price_trend != osc_trend
        
        return False
    
    def _analyze_volume_confirmation(self, data: pd.DataFrame, oscillator: pd.Series) -> float:
        """Analyze volume confirmation of oscillator signals."""
        if len(data) < 5:
            return 0.5
        
        # Calculate volume trend
        volume_trend = data['volume'].rolling(window=5).mean().pct_change().fillna(0)
        osc_trend = oscillator.pct_change().fillna(0)
        
        # Calculate correlation
        correlation = volume_trend.corr(osc_trend)
        return float(correlation) if not np.isnan(correlation) else 0.5
    
    def _calculate_volatility_adjustment(self, data: pd.DataFrame) -> float:
        """Calculate volatility adjustment factor."""
        if len(data) < 20:
            return 1.0
        
        current_vol = data['close'].pct_change().rolling(window=20).std().iloc[-1]
        historical_vol = data['close'].pct_change().std()
        
        if historical_vol == 0:
            return 1.0
        
        vol_ratio = current_vol / historical_vol
        # Adjust signals based on volatility regime
        return float(np.clip(vol_ratio, 0.5, 2.0))
    
    def _determine_signal_type(self, signals: Dict[str, pd.Series]) -> SignalType:
        """Determine the overall signal type."""
        latest_histogram = signals['histogram'].iloc[-1]
        latest_direction = signals['wave_direction'].iloc[-1]
        latest_strength = signals['wave_strength'].iloc[-1]
        
        # Strong bullish signal
        if latest_histogram > 0 and latest_direction > 0 and latest_strength > 1.5:
            return SignalType.STRONG_BUY
        # Bullish signal
        elif latest_histogram > 0 and latest_direction > 0:
            return SignalType.BUY
        # Strong bearish signal
        elif latest_histogram < 0 and latest_direction < 0 and latest_strength > 1.5:
            return SignalType.STRONG_SELL
        # Bearish signal
        elif latest_histogram < 0 and latest_direction < 0:
            return SignalType.SELL
        # Neutral
        else:
            return SignalType.NEUTRAL
    
    def _update_training_data(self, features: Optional[WaveFeatures], oscillator: pd.Series):
        """Update training data for ML model."""
        if features and len(oscillator) > 0:
            self.historical_features.append(features)
            self.historical_signals.append(float(oscillator.iloc[-1]))
            
            # Keep only recent data
            max_history = 1000
            if len(self.historical_features) > max_history:
                self.historical_features = self.historical_features[-max_history:]
                self.historical_signals = self.historical_signals[-max_history:]
            
            # Retrain periodically
            if len(self.historical_features) % 100 == 0 and len(self.historical_features) >= 100:
                self._train_ml_model()
    
    def _train_ml_model(self):
        """Train the machine learning model."""
        try:
            if len(self.historical_features) < 50:
                return
            
            # Prepare training data
            X = np.array([[
                f.momentum, f.acceleration, f.volume_momentum, f.price_momentum,
                f.volatility, f.trend_strength, f.wave_count, f.fibonacci_level
            ] for f in self.historical_features])
            
            # Create labels (1 for positive oscillator, 0 for negative)
            y = np.array([1 if s > 0 else 0 for s in self.historical_signals])
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train classifier
            self.wave_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.wave_classifier.fit(X_scaled, y)
            self.is_trained = True
            
            self.logger.info("ML model trained successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model training failed: {str(e)}")
    
    def get_signal_type(self) -> SignalType:
        """Get the current signal type."""
        return getattr(self, '_last_signal_type', SignalType.NEUTRAL)
    
    def get_signal_strength(self) -> float:
        """Get the current signal strength."""
        return getattr(self, '_last_signal_strength', 0.0)