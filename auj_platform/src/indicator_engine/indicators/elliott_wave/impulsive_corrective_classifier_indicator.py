"""
Impulsive Corrective Classifier Indicator

Advanced ML-based implementation to distinguish between impulsive and corrective
wave patterns using sophisticated machine learning algorithms for the humanitarian
trading platform.

This indicator employs multiple ML models including neural networks, ensemble methods,
and deep learning to classify Elliott Wave patterns with high accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy import signal
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import skew, kurtosis, entropy
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class ClassifierConfig:
    """Configuration for Impulsive Corrective Classifier."""
    feature_window: int = 100
    prediction_window: int = 20
    confidence_threshold: float = 0.8
    retrain_frequency: int = 50
    ensemble_size: int = 5
    neural_hidden_layers: Tuple[int, ...] = (100, 50, 25)
    min_training_samples: int = 200


@dataclass
class WaveFeatures:
    """Comprehensive features for wave classification."""
    # Price-based features
    price_momentum: float
    price_acceleration: float
    price_volatility: float
    price_trend_strength: float
    
    # Volume features
    volume_momentum: float
    volume_trend: float
    volume_acceleration: float
    
    # Technical features
    rsi: float
    macd_signal: float
    bollinger_position: float
    
    # Wave-specific features
    wave_length: float
    wave_amplitude: float
    wave_slope: float
    wave_curvature: float
    
    # Fractal features
    fractal_dimension: float
    hurst_exponent: float
    
    # Frequency domain features
    dominant_frequency: float
    spectral_entropy: float
    power_spectrum_peak: float
    
    # Statistical features
    skewness: float
    kurtosis: float
    autocorrelation: float


@dataclass
class ClassificationResult:
    """Result of wave classification."""
    wave_type: str  # 'impulse' or 'corrective'
    confidence: float
    probability_impulse: float
    probability_corrective: float
    feature_importance: Dict[str, float]
    model_consensus: Dict[str, str]


class ImpulsiveCorrectiveClassifierIndicator(StandardIndicatorInterface):
    """
    Advanced ML-based classifier for Elliott Wave impulse vs corrective patterns.
    
    This indicator uses ensemble machine learning methods to classify wave patterns
    with high accuracy and provides detailed confidence metrics.
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        super().__init__()
        self.config = config or ClassifierConfig()
        self.logger = logging.getLogger(__name__)
        
        # ML components
        self.ensemble_model: Optional[VotingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_trained = False
        
        # Training data storage
        self.training_features: List[WaveFeatures] = []
        self.training_labels: List[str] = []
        self.feature_names: List[str] = []
        
        # Model performance tracking
        self.model_accuracy: float = 0.0
        self.confusion_matrix: Optional[np.ndarray] = None
        self.feature_importance: Dict[str, float] = {}
        
        # Pattern analysis
        self.historical_classifications: List[ClassificationResult] = []
        
        # Initialize feature names
        self._initialize_feature_names()
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify wave patterns using advanced ML techniques.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing classification results and analysis
        """
        try:
            if len(data) < self.config.feature_window:
                raise IndicatorCalculationError(
                    f"Insufficient data: need at least {self.config.feature_window} periods"
                )
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(data)
            
            # Perform classification if model is trained
            if self.is_trained and features:
                classification_result = self._classify_wave_pattern(features)
            else:
                classification_result = self._get_default_classification()
            
            # Analyze pattern trends
            pattern_trends = self._analyze_pattern_trends()
            
            # Calculate wave strength indicators
            wave_strength = self._calculate_wave_strength_indicators(data, features)
            
            # Generate market regime analysis
            market_regime = self._analyze_market_regime(data, classification_result)
            
            # Update training data with synthetic labels if needed
            self._update_training_data(features, data)
            
            result = {
                'wave_type': classification_result.wave_type,
                'confidence': classification_result.confidence,
                'probability_impulse': classification_result.probability_impulse,
                'probability_corrective': classification_result.probability_corrective,
                'model_consensus': classification_result.model_consensus,
                'feature_importance': classification_result.feature_importance,
                'pattern_trends': pattern_trends,
                'wave_strength_momentum': wave_strength['momentum'],
                'wave_strength_volume': wave_strength['volume'],
                'wave_strength_technical': wave_strength['technical'],
                'market_regime': market_regime['regime'],
                'regime_confidence': market_regime['confidence'],
                'trend_persistence': market_regime['trend_persistence'],
                'volatility_regime': market_regime['volatility_regime'],
                'model_accuracy': self.model_accuracy,
                'training_samples': len(self.training_features),
                'signal_type': self._determine_signal_type(classification_result, wave_strength),
                'raw_data': {
                    'features': features.__dict__ if features else None,
                    'feature_names': self.feature_names,
                    'model_performance': {
                        'accuracy': self.model_accuracy,
                        'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None
                    }
                }
            }
            
            # Store classification result
            self.historical_classifications.append(classification_result)
            
            self.logger.info(f"Wave classified as {classification_result.wave_type} with {classification_result.confidence:.3f} confidence")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Impulsive Corrective Classifier: {str(e)}")
            raise IndicatorCalculationError(f"Classification failed: {str(e)}")
    
    def _extract_comprehensive_features(self, data: pd.DataFrame) -> Optional[WaveFeatures]:
        """Extract comprehensive features for ML classification."""
        try:
            if len(data) < self.config.feature_window:
                return None
            
            recent_data = data.tail(self.config.feature_window)
            
            # Price-based features
            price_features = self._extract_price_features(recent_data)
            
            # Volume features
            volume_features = self._extract_volume_features(recent_data)
            
            # Technical indicator features
            technical_features = self._extract_technical_features(recent_data)
            
            # Wave-specific features
            wave_features = self._extract_wave_features(recent_data)
            
            # Fractal features
            fractal_features = self._extract_fractal_features(recent_data)
            
            # Frequency domain features
            frequency_features = self._extract_frequency_features(recent_data)
            
            # Statistical features
            statistical_features = self._extract_statistical_features(recent_data)
            
            return WaveFeatures(
                # Price features
                price_momentum=price_features['momentum'],
                price_acceleration=price_features['acceleration'],
                price_volatility=price_features['volatility'],
                price_trend_strength=price_features['trend_strength'],
                
                # Volume features
                volume_momentum=volume_features['momentum'],
                volume_trend=volume_features['trend'],
                volume_acceleration=volume_features['acceleration'],
                
                # Technical features
                rsi=technical_features['rsi'],
                macd_signal=technical_features['macd_signal'],
                bollinger_position=technical_features['bollinger_position'],
                
                # Wave features
                wave_length=wave_features['length'],
                wave_amplitude=wave_features['amplitude'],
                wave_slope=wave_features['slope'],
                wave_curvature=wave_features['curvature'],
                
                # Fractal features
                fractal_dimension=fractal_features['dimension'],
                hurst_exponent=fractal_features['hurst'],
                
                # Frequency features
                dominant_frequency=frequency_features['dominant_freq'],
                spectral_entropy=frequency_features['spectral_entropy'],
                power_spectrum_peak=frequency_features['power_peak'],
                
                # Statistical features
                skewness=statistical_features['skewness'],
                kurtosis=statistical_features['kurtosis'],
                autocorrelation=statistical_features['autocorrelation']
            )
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {str(e)}")
            return None
    
    def _extract_price_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract price-based features."""
        close_prices = data['close'].values
        returns = np.diff(np.log(close_prices))
        
        # Momentum (rate of change)
        momentum = (close_prices[-1] - close_prices[0]) / close_prices[0]
        
        # Acceleration (second derivative)
        if len(returns) > 1:
            acceleration = np.mean(np.diff(returns))
        else:
            acceleration = 0.0
        
        # Volatility
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        
        # Trend strength (linear regression slope)
        x = np.arange(len(close_prices))
        trend_strength = np.polyfit(x, close_prices, 1)[0] / close_prices[0] if len(close_prices) > 1 else 0.0
        
        return {
            'momentum': float(momentum),
            'acceleration': float(acceleration),
            'volatility': float(volatility),
            'trend_strength': float(trend_strength)
        }
    
    def _extract_volume_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract volume-based features."""
        volume = data['volume'].values
        
        # Volume momentum
        volume_ma = np.mean(volume)
        volume_momentum = (volume[-1] - volume_ma) / volume_ma if volume_ma > 0 else 0.0
        
        # Volume trend
        x = np.arange(len(volume))
        volume_trend = np.polyfit(x, volume, 1)[0] / np.mean(volume) if len(volume) > 1 and np.mean(volume) > 0 else 0.0
        
        # Volume acceleration
        volume_diff = np.diff(volume)
        volume_acceleration = np.mean(np.diff(volume_diff)) if len(volume_diff) > 1 else 0.0
        
        return {
            'momentum': float(volume_momentum),
            'trend': float(volume_trend),
            'acceleration': float(volume_acceleration)
        }
    
    def _extract_technical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract technical indicator features."""
        close_prices = data['close']
        
        # RSI
        rsi = self._calculate_rsi(close_prices, period=14)
        
        # MACD
        macd_signal = self._calculate_macd_signal(close_prices)
        
        # Bollinger Bands position
        bollinger_position = self._calculate_bollinger_position(close_prices)
        
        return {
            'rsi': float(rsi),
            'macd_signal': float(macd_signal),
            'bollinger_position': float(bollinger_position)
        }
    
    def _extract_wave_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract wave-specific features."""
        close_prices = data['close'].values
        
        # Wave length (time dimension)
        wave_length = len(close_prices)
        
        # Wave amplitude (price dimension)
        wave_amplitude = (np.max(close_prices) - np.min(close_prices)) / np.mean(close_prices)
        
        # Wave slope (overall direction)
        x = np.arange(len(close_prices))
        wave_slope = np.polyfit(x, close_prices, 1)[0] if len(close_prices) > 1 else 0.0
        
        # Wave curvature (second derivative)
        if len(close_prices) >= 3:
            second_derivative = np.gradient(np.gradient(close_prices))
            wave_curvature = np.mean(np.abs(second_derivative))
        else:
            wave_curvature = 0.0
        
        return {
            'length': float(wave_length),
            'amplitude': float(wave_amplitude),
            'slope': float(wave_slope),
            'curvature': float(wave_curvature)
        }
    
    def _extract_fractal_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract fractal-based features."""
        close_prices = data['close'].values
        
        # Simplified fractal dimension
        fractal_dimension = self._calculate_simple_fractal_dimension(close_prices)
        
        # Simplified Hurst exponent
        hurst_exponent = self._calculate_simple_hurst_exponent(close_prices)
        
        return {
            'dimension': float(fractal_dimension),
            'hurst': float(hurst_exponent)
        }
    
    def _extract_frequency_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract frequency domain features."""
        close_prices = data['close'].values
        
        # Power spectral density
        frequencies, power = welch(close_prices, nperseg=min(len(close_prices)//4, 32))
        
        # Dominant frequency
        dominant_freq = frequencies[np.argmax(power)] if len(power) > 0 else 0.0
        
        # Spectral entropy
        power_normalized = power / np.sum(power) if np.sum(power) > 0 else power
        spectral_entropy_val = entropy(power_normalized + 1e-10)
        
        # Power spectrum peak
        power_peak = np.max(power) if len(power) > 0 else 0.0
        
        return {
            'dominant_freq': float(dominant_freq),
            'spectral_entropy': float(spectral_entropy_val),
            'power_peak': float(power_peak)
        }
    
    def _extract_statistical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical features."""
        returns = data['close'].pct_change().dropna().values
        
        # Skewness
        skewness_val = skew(returns) if len(returns) > 3 else 0.0
        
        # Kurtosis
        kurtosis_val = kurtosis(returns) if len(returns) > 3 else 0.0
        
        # Autocorrelation (lag 1)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            autocorr = autocorr if not np.isnan(autocorr) else 0.0
        else:
            autocorr = 0.0
        
        return {
            'skewness': float(skewness_val),
            'kurtosis': float(kurtosis_val),
            'autocorrelation': float(autocorr)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """Calculate MACD signal."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        
        macd_signal = macd.iloc[-1] - signal_line.iloc[-1]
        return float(macd_signal) if not np.isnan(macd_signal) else 0.0
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate position within Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        current_price = prices.iloc[-1]
        upper_val = upper_band.iloc[-1]
        lower_val = lower_band.iloc[-1]
        
        if upper_val == lower_val:
            return 0.5
        
        position = (current_price - lower_val) / (upper_val - lower_val)
        return float(np.clip(position, 0, 1))
    
    def _calculate_simple_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate simplified fractal dimension."""
        try:
            # Simplified box-counting approach
            max_boxes = 32
            box_counts = []
            box_sizes = np.logspace(0, np.log10(len(prices)/4), max_boxes, dtype=int)
            box_sizes = np.unique(box_sizes)
            
            normalized_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
            
            for box_size in box_sizes:
                if box_size >= len(prices):
                    continue
                
                # Count boxes
                grid_boxes = int(1.0 / (1.0 / box_size)) + 1
                covered_boxes = set()
                
                for i in range(len(normalized_prices)):
                    price_box = int(normalized_prices[i] * grid_boxes)
                    time_box = int(i / box_size)
                    covered_boxes.add((time_box, price_box))
                
                box_counts.append(len(covered_boxes))
            
            if len(box_counts) >= 3:
                # Linear regression on log-log plot
                log_box_sizes = np.log(1.0 / np.array(box_sizes[:len(box_counts)]))
                log_box_counts = np.log(np.array(box_counts))
                
                slope = np.polyfit(log_box_sizes, log_box_counts, 1)[0]
                return max(1.0, min(2.0, abs(slope)))
            
            return 1.5
            
        except Exception:
            return 1.5
    
    def _calculate_simple_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate simplified Hurst exponent."""
        try:
            returns = np.diff(np.log(prices))
            
            # R/S analysis with limited window sizes
            window_sizes = [5, 10, 20, 40]
            rs_values = []
            
            for window_size in window_sizes:
                if window_size >= len(returns):
                    continue
                
                # Calculate R/S for this window size
                mean_return = np.mean(returns[:window_size])
                deviations = np.cumsum(returns[:window_size] - mean_return)
                
                R = np.max(deviations) - np.min(deviations)
                S = np.std(returns[:window_size])
                
                if S > 0:
                    rs_values.append(R / S)
            
            if len(rs_values) >= 2:
                # Simple linear regression
                log_window_sizes = np.log(window_sizes[:len(rs_values)])
                log_rs_values = np.log(rs_values)
                
                hurst = np.polyfit(log_window_sizes, log_rs_values, 1)[0]
                return max(0.0, min(1.0, hurst))
            
            return 0.5
            
        except Exception:
            return 0.5    
    def _classify_wave_pattern(self, features: WaveFeatures) -> ClassificationResult:
        """Classify wave pattern using trained ML models."""
        try:
            if not self.is_trained or not self.ensemble_model:
                return self._get_default_classification()
            
            # Prepare feature vector
            feature_vector = self._features_to_vector(features)
            feature_vector = feature_vector.reshape(1, -1)
            
            # Scale features
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Get predictions from ensemble
            prediction = self.ensemble_model.predict(feature_vector)[0]
            prediction_proba = self.ensemble_model.predict_proba(feature_vector)[0]
            
            # Get individual model predictions for consensus
            model_consensus = self._get_model_consensus(feature_vector)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance()
            
            # Determine confidence
            confidence = max(prediction_proba)
            
            # Map prediction to wave type
            wave_type = self.label_encoder.inverse_transform([prediction])[0] if self.label_encoder else 'impulse'
            
            return ClassificationResult(
                wave_type=wave_type,
                confidence=float(confidence),
                probability_impulse=float(prediction_proba[1] if len(prediction_proba) > 1 else 0.5),
                probability_corrective=float(prediction_proba[0] if len(prediction_proba) > 0 else 0.5),
                feature_importance=feature_importance,
                model_consensus=model_consensus
            )
            
        except Exception as e:
            self.logger.warning(f"Classification failed: {str(e)}")
            return self._get_default_classification()
    
    def _get_default_classification(self) -> ClassificationResult:
        """Get default classification when model is not available."""
        return ClassificationResult(
            wave_type='impulse',
            confidence=0.5,
            probability_impulse=0.5,
            probability_corrective=0.5,
            feature_importance={},
            model_consensus={}
        )
    
    def _features_to_vector(self, features: WaveFeatures) -> np.ndarray:
        """Convert features dataclass to numpy vector."""
        return np.array([
            features.price_momentum,
            features.price_acceleration,
            features.price_volatility,
            features.price_trend_strength,
            features.volume_momentum,
            features.volume_trend,
            features.volume_acceleration,
            features.rsi,
            features.macd_signal,
            features.bollinger_position,
            features.wave_length,
            features.wave_amplitude,
            features.wave_slope,
            features.wave_curvature,
            features.fractal_dimension,
            features.hurst_exponent,
            features.dominant_frequency,
            features.spectral_entropy,
            features.power_spectrum_peak,
            features.skewness,
            features.kurtosis,
            features.autocorrelation
        ])
    
    def _get_model_consensus(self, feature_vector: np.ndarray) -> Dict[str, str]:
        """Get consensus from individual models."""
        consensus = {}
        
        try:
            if hasattr(self.ensemble_model, 'estimators_'):
                for i, estimator in enumerate(self.ensemble_model.estimators_):
                    prediction = estimator.predict(feature_vector)[0]
                    wave_type = self.label_encoder.inverse_transform([prediction])[0] if self.label_encoder else 'impulse'
                    consensus[f'model_{i}'] = wave_type
        except Exception as e:
            self.logger.warning(f"Model consensus failed: {str(e)}")
        
        return consensus
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance from trained models."""
        importance_dict = {}
        
        try:
            if hasattr(self.ensemble_model, 'estimators_'):
                # Average importance across ensemble models
                total_importance = np.zeros(len(self.feature_names))
                model_count = 0
                
                for estimator in self.ensemble_model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        total_importance += estimator.feature_importances_
                        model_count += 1
                
                if model_count > 0:
                    avg_importance = total_importance / model_count
                    importance_dict = {
                        name: float(importance) 
                        for name, importance in zip(self.feature_names, avg_importance)
                    }
        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {str(e)}")
        
        return importance_dict
    
    def _analyze_pattern_trends(self) -> Dict[str, Any]:
        """Analyze trends in historical classifications."""
        if len(self.historical_classifications) < 5:
            return {
                'recent_trend': 'insufficient_data',
                'impulse_ratio': 0.5,
                'confidence_trend': 'stable',
                'pattern_stability': 0.5
            }
        
        recent_classifications = self.historical_classifications[-10:]
        
        # Calculate impulse ratio
        impulse_count = sum(1 for c in recent_classifications if c.wave_type == 'impulse')
        impulse_ratio = impulse_count / len(recent_classifications)
        
        # Analyze confidence trend
        confidences = [c.confidence for c in recent_classifications]
        confidence_trend = 'increasing' if confidences[-1] > confidences[0] else 'decreasing'
        
        # Pattern stability (consistency of classifications)
        wave_types = [c.wave_type for c in recent_classifications]
        unique_types = set(wave_types)
        stability = 1.0 - (len(unique_types) - 1) / len(recent_classifications)
        
        # Recent trend
        recent_trend = 'bullish_impulse' if impulse_ratio > 0.6 else 'corrective' if impulse_ratio < 0.4 else 'mixed'
        
        return {
            'recent_trend': recent_trend,
            'impulse_ratio': float(impulse_ratio),
            'confidence_trend': confidence_trend,
            'pattern_stability': float(stability)
        }
    
    def _calculate_wave_strength_indicators(self, data: pd.DataFrame, features: Optional[WaveFeatures]) -> Dict[str, float]:
        """Calculate various wave strength indicators."""
        if not features:
            return {'momentum': 0.5, 'volume': 0.5, 'technical': 0.5}
        
        # Momentum strength (combination of price features)
        momentum_strength = (
            abs(features.price_momentum) * 0.4 +
            abs(features.price_acceleration) * 0.3 +
            features.price_trend_strength * 0.3
        )
        
        # Volume strength
        volume_strength = (
            abs(features.volume_momentum) * 0.5 +
            abs(features.volume_trend) * 0.3 +
            abs(features.volume_acceleration) * 0.2
        )
        
        # Technical strength (RSI, MACD, Bollinger position)
        rsi_strength = abs(features.rsi - 50) / 50  # Distance from neutral
        macd_strength = abs(features.macd_signal)
        bollinger_strength = abs(features.bollinger_position - 0.5) * 2  # Distance from center
        
        technical_strength = (rsi_strength + macd_strength + bollinger_strength) / 3
        
        return {
            'momentum': float(min(1.0, momentum_strength)),
            'volume': float(min(1.0, volume_strength)),
            'technical': float(min(1.0, technical_strength))
        }
    
    def _analyze_market_regime(self, data: pd.DataFrame, classification: ClassificationResult) -> Dict[str, Any]:
        """Analyze current market regime."""
        try:
            recent_data = data.tail(50) if len(data) >= 50 else data
            
            # Trend analysis
            returns = recent_data['close'].pct_change().dropna()
            trend_strength = abs(returns.mean()) / (returns.std() + 1e-8)
            
            # Volatility regime
            volatility = returns.std()
            historical_vol = data['close'].pct_change().std() if len(data) > 100 else volatility
            vol_ratio = volatility / (historical_vol + 1e-8)
            
            # Trend persistence (autocorrelation)
            if len(returns) > 1:
                trend_persistence = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                trend_persistence = trend_persistence if not np.isnan(trend_persistence) else 0.0
            else:
                trend_persistence = 0.0
            
            # Determine regime
            if trend_strength > 1.5 and classification.wave_type == 'impulse':
                regime = 'strong_trending'
            elif trend_strength < 0.5:
                regime = 'sideways'
            elif vol_ratio > 1.5:
                regime = 'high_volatility'
            else:
                regime = 'normal'
            
            # Volatility regime classification
            if vol_ratio > 1.5:
                vol_regime = 'high'
            elif vol_ratio < 0.7:
                vol_regime = 'low'
            else:
                vol_regime = 'normal'
            
            return {
                'regime': regime,
                'confidence': float(classification.confidence),
                'trend_persistence': float(abs(trend_persistence)),
                'volatility_regime': vol_regime
            }
            
        except Exception as e:
            self.logger.warning(f"Market regime analysis failed: {str(e)}")
            return {
                'regime': 'unknown',
                'confidence': 0.5,
                'trend_persistence': 0.5,
                'volatility_regime': 'normal'
            }
    
    def _update_training_data(self, features: Optional[WaveFeatures], data: pd.DataFrame):
        """Update training data with new observations."""
        if not features:
            return
        
        # Generate synthetic label based on price action patterns
        synthetic_label = self._generate_synthetic_label(data)
        
        if synthetic_label:
            self.training_features.append(features)
            self.training_labels.append(synthetic_label)
            
            # Limit training data size
            max_samples = 1000
            if len(self.training_features) > max_samples:
                self.training_features = self.training_features[-max_samples:]
                self.training_labels = self.training_labels[-max_samples:]
            
            # Retrain model periodically
            if len(self.training_features) % self.config.retrain_frequency == 0:
                self._train_ensemble_model()
    
    def _generate_synthetic_label(self, data: pd.DataFrame) -> Optional[str]:
        """Generate synthetic labels based on Elliott Wave heuristics."""
        try:
            if len(data) < 20:
                return None
            
            recent_data = data.tail(20)
            
            # Calculate price characteristics
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            volatility = recent_data['close'].pct_change().std()
            
            # Volume characteristics
            volume_trend = np.polyfit(range(len(recent_data)), recent_data['volume'].values, 1)[0]
            
            # Heuristic rules for labeling
            # Impulse waves: strong directional movement with volume
            if abs(price_change) > 0.02 and volatility < 0.01 and volume_trend > 0:
                return 'impulse'
            
            # Corrective waves: choppy movement, lower volume
            elif abs(price_change) < 0.01 and volatility > 0.01:
                return 'corrective'
            
            # Default to None for ambiguous cases
            return None
            
        except Exception:
            return None
    
    def _train_ensemble_model(self):
        """Train the ensemble model with current training data."""
        try:
            if len(self.training_features) < self.config.min_training_samples:
                return
            
            # Prepare training data
            X = np.array([self._features_to_vector(f) for f in self.training_features])
            y = np.array(self.training_labels)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Create ensemble model
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            mlp_model = MLPClassifier(
                hidden_layer_sizes=self.config.neural_hidden_layers,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            svm_model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            
            # Create voting ensemble
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('gb', gb_model),
                    ('mlp', mlp_model),
                    ('svm', svm_model)
                ],
                voting='soft'
            )
            
            # Train ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.ensemble_model.score(X_train, y_train)
            test_score = self.ensemble_model.score(X_test, y_test)
            
            self.model_accuracy = float(test_score)
            
            # Generate confusion matrix
            y_pred = self.ensemble_model.predict(X_test)
            self.confusion_matrix = confusion_matrix(y_test, y_pred)
            
            # Calculate feature importance
            self.feature_importance = self._calculate_feature_importance()
            
            self.is_trained = True
            
            self.logger.info(f"Ensemble model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Model training failed: {str(e)}")
    
    def _determine_signal_type(self, classification: ClassificationResult, wave_strength: Dict[str, float]) -> SignalType:
        """Determine signal type based on classification and strength."""
        avg_strength = np.mean(list(wave_strength.values()))
        
        if classification.wave_type == 'impulse' and classification.confidence > 0.8:
            if avg_strength > 0.8:
                return SignalType.STRONG_BUY if classification.probability_impulse > 0.6 else SignalType.STRONG_SELL
            else:
                return SignalType.BUY if classification.probability_impulse > 0.6 else SignalType.SELL
        elif classification.wave_type == 'corrective':
            return SignalType.NEUTRAL
        else:
            return SignalType.NEUTRAL
    
    def _initialize_feature_names(self):
        """Initialize feature names for interpretability."""
        self.feature_names = [
            'price_momentum', 'price_acceleration', 'price_volatility', 'price_trend_strength',
            'volume_momentum', 'volume_trend', 'volume_acceleration',
            'rsi', 'macd_signal', 'bollinger_position',
            'wave_length', 'wave_amplitude', 'wave_slope', 'wave_curvature',
            'fractal_dimension', 'hurst_exponent',
            'dominant_frequency', 'spectral_entropy', 'power_spectrum_peak',
            'skewness', 'kurtosis', 'autocorrelation'
        ]
    
    def get_signal_type(self) -> SignalType:
        """Get the current signal type."""
        return getattr(self, '_last_signal_type', SignalType.NEUTRAL)
    
    def get_signal_strength(self) -> float:
        """Get the current signal strength."""
        if self.historical_classifications:
            return self.historical_classifications[-1].confidence
        return 0.0
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get detailed model performance metrics."""
        return {
            'accuracy': self.model_accuracy,
            'training_samples': len(self.training_features),
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None
        }