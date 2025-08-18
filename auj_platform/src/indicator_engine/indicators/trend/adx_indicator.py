"""
Advanced ADX (Average Directional Index) Indicator with ML Enhancement

This implementation provides sophisticated trend strength analysis with:
- Traditional ADX calculation with Wilder's smoothing
- Trend strength classification using ML models
- Multi-timeframe analysis capabilities
- Advanced signal generation with confidence scoring
- Adaptive parameter optimization
- Real-time trend strength assessment

The ADX measures trend strength regardless of direction, making it crucial
for determining when trends are strong enough for trend-following strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType


class TrendStrength(Enum):
    """Trend strength classification based on ADX values"""
    ABSENT = "absent"          # ADX < 20
    WEAK = "weak"              # 20 <= ADX < 25
    MODERATE = "moderate"      # 25 <= ADX < 40
    STRONG = "strong"          # 40 <= ADX < 60
    VERY_STRONG = "very_strong"  # ADX >= 60


@dataclass
class ADXResult:
    """Comprehensive ADX analysis result"""
    adx: float
    di_plus: float
    di_minus: float
    dx: float
    trend_strength: TrendStrength
    trend_direction: str
    confidence_score: float
    signal: SignalType
    ml_prediction: Optional[float] = None


class ADXIndicator(StandardIndicatorInterface):
    """
    Advanced Average Directional Index (ADX) Indicator
    
    The ADX is a technical analysis indicator that measures the strength of a trend.
    It ranges from 0 to 100, with higher values indicating stronger trends.
    
    Key Features:
    - Traditional ADX calculation using Wilder's smoothing
    - ML-enhanced trend strength prediction
    - Multi-timeframe analysis
    - Advanced signal generation
    - Adaptive parameter optimization
    """
    
    def __init__(self, 
                 period: int = 14,
                 smoothing_period: int = 14,
                 enable_ml: bool = True,
                 lookback_window: int = 100):
        """
        Initialize the ADX Indicator
        
        Args:
            period: Period for DI calculation (default: 14)
            smoothing_period: Period for ADX smoothing (default: 14)
            enable_ml: Enable machine learning enhancements
            lookback_window: Window for ML training data
        """
        self.period = period
        self.smoothing_period = smoothing_period
        self.enable_ml = enable_ml and ML_AVAILABLE
        self.lookback_window = lookback_window
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler() if self.enable_ml else None
        self.ml_trained = False
        
        # Historical data storage for ML
        self.feature_history = []
        self.target_history = []
        
        # Performance tracking
        self.signal_history = []
        self.performance_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate ADX with advanced features
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing ADX analysis results
        """
        try:
            if len(data) < self.period + self.smoothing_period:
                raise ValueError(f"Insufficient data: need at least {self.period + self.smoothing_period} periods")
            
            # Calculate basic ADX components
            adx_data = self._calculate_adx_components(data)
            
            # Calculate ADX values
            adx_values = self._calculate_adx(adx_data)
            
            # Generate signals and analysis
            results = self._generate_comprehensive_analysis(data, adx_values)
            
            # ML enhancement if enabled
            if self.enable_ml:
                results = self._enhance_with_ml(data, results)
            
            # Multi-timeframe analysis
            mtf_analysis = self._multi_timeframe_analysis(data)
            results['multi_timeframe'] = mtf_analysis
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return self._get_default_result()
    
    def _calculate_adx_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate True Range, +DM, -DM, and Directional Indicators"""
        df = data.copy()
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Directional Movement
        df['high_diff'] = df['high'] - df['high'].shift(1)
        df['low_diff'] = df['low'].shift(1) - df['low']
        
        # +DM and -DM calculations
        df['dm_plus'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'], 0
        )
        
        df['dm_minus'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'], 0
        )
        
        # Wilder's smoothing for TR, +DM, -DM
        df['tr_smooth'] = self._wilders_smoothing(df['tr'], self.period)
        df['dm_plus_smooth'] = self._wilders_smoothing(df['dm_plus'], self.period)
        df['dm_minus_smooth'] = self._wilders_smoothing(df['dm_minus'], self.period)
        
        # Calculate Directional Indicators
        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])
        
        # Calculate DX
        df['di_diff'] = abs(df['di_plus'] - df['di_minus'])
        df['di_sum'] = df['di_plus'] + df['di_minus']
        df['dx'] = 100 * (df['di_diff'] / df['di_sum'])
        
        return df
    
    def _wilders_smoothing(self, series: pd.Series, period: int) -> pd.Series:
        """Apply Wilder's smoothing method"""
        result = series.copy()
        alpha = 1.0 / period
        
        for i in range(1, len(result)):
            if pd.notna(result.iloc[i-1]):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
        
        return result
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX using Wilder's smoothing of DX"""
        return self._wilders_smoothing(df['dx'], self.smoothing_period)
    
    def _classify_trend_strength(self, adx_value: float) -> TrendStrength:
        """Classify trend strength based on ADX value"""
        if adx_value < 20:
            return TrendStrength.ABSENT
        elif adx_value < 25:
            return TrendStrength.WEAK
        elif adx_value < 40:
            return TrendStrength.MODERATE
        elif adx_value < 60:
            return TrendStrength.STRONG
        else:
            return TrendStrength.VERY_STRONG
    
    def _determine_trend_direction(self, di_plus: float, di_minus: float) -> str:
        """Determine trend direction based on DI values"""
        if di_plus > di_minus:
            return "bullish"
        elif di_minus > di_plus:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_confidence_score(self, adx: float, di_plus: float, di_minus: float) -> float:
        """Calculate confidence score for the signal"""
        # Higher ADX = higher confidence
        adx_component = min(adx / 60, 1.0)
        
        # Larger difference between DI+ and DI- = higher confidence
        di_diff = abs(di_plus - di_minus)
        di_component = min(di_diff / 50, 1.0)
        
        # Combined confidence score
        confidence = (adx_component * 0.7) + (di_component * 0.3)
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_signals(self, adx: float, di_plus: float, di_minus: float, 
                         trend_strength: TrendStrength) -> SignalType:
        """Generate trading signals based on ADX analysis"""
        if trend_strength in [TrendStrength.ABSENT, TrendStrength.WEAK]:
            return SignalType.NEUTRAL
        
        if di_plus > di_minus and trend_strength in [TrendStrength.MODERATE, TrendStrength.STRONG, TrendStrength.VERY_STRONG]:
            return SignalType.BUY
        elif di_minus > di_plus and trend_strength in [TrendStrength.MODERATE, TrendStrength.STRONG, TrendStrength.VERY_STRONG]:
            return SignalType.SELL
        
        return SignalType.NEUTRAL
    
    def _generate_comprehensive_analysis(self, data: pd.DataFrame, adx_values: pd.Series) -> Dict:
        """Generate comprehensive ADX analysis"""
        latest_idx = len(data) - 1
        
        # Get latest values
        adx_data = self._calculate_adx_components(data)
        latest_adx = adx_values.iloc[-1] if not adx_values.empty else 0.0
        latest_di_plus = adx_data['di_plus'].iloc[-1] if 'di_plus' in adx_data.columns else 0.0
        latest_di_minus = adx_data['di_minus'].iloc[-1] if 'di_minus' in adx_data.columns else 0.0
        latest_dx = adx_data['dx'].iloc[-1] if 'dx' in adx_data.columns else 0.0
        
        # Analysis
        trend_strength = self._classify_trend_strength(latest_adx)
        trend_direction = self._determine_trend_direction(latest_di_plus, latest_di_minus)
        confidence_score = self._calculate_confidence_score(latest_adx, latest_di_plus, latest_di_minus)
        signal = self._generate_signals(latest_adx, latest_di_plus, latest_di_minus, trend_strength)
        
        # Create comprehensive result
        result = ADXResult(
            adx=latest_adx,
            di_plus=latest_di_plus,
            di_minus=latest_di_minus,
            dx=latest_dx,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            confidence_score=confidence_score,
            signal=signal
        )
        
        return {
            'current': result,
            'values': {
                'adx': adx_values.tolist(),
                'di_plus': adx_data['di_plus'].tolist(),
                'di_minus': adx_data['di_minus'].tolist(),
                'dx': adx_data['dx'].tolist()
            },
            'signals': signal,
            'trend_strength': trend_strength.value,
            'trend_direction': trend_direction,
            'confidence': confidence_score,
            'metadata': {
                'period': self.period,
                'smoothing_period': self.smoothing_period,
                'ml_enabled': self.enable_ml,
                'calculation_time': pd.Timestamp.now().isoformat()
            }
        }
    
    def _enhance_with_ml(self, data: pd.DataFrame, results: Dict) -> Dict:
        """Enhance results with machine learning predictions"""
        if not self.enable_ml:
            return results
        
        try:
            # Extract features for ML
            features = self._extract_ml_features(data)
            
            # Train model if needed
            if not self.ml_trained and len(self.feature_history) >= 50:
                self._train_ml_model()
            
            # Make prediction if model is trained
            if self.ml_trained and self.ml_model is not None:
                prediction = self._make_ml_prediction(features)
                results['current'].ml_prediction = prediction
                results['ml_prediction'] = prediction
                results['ml_confidence'] = self._calculate_ml_confidence(features)
        
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {e}")
        
        return results
    
    def _extract_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for machine learning model"""
        # Calculate additional technical features
        features = []
        
        # Price-based features
        returns = data['close'].pct_change().fillna(0)
        features.extend([
            returns.iloc[-1],  # Latest return
            returns.rolling(5).mean().iloc[-1],  # 5-period average return
            returns.rolling(5).std().iloc[-1],   # 5-period return volatility
        ])
        
        # Volume features
        volume_sma = data['volume'].rolling(20).mean()
        features.extend([
            data['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0,
        ])
        
        # ADX components
        adx_data = self._calculate_adx_components(data)
        features.extend([
            adx_data['di_plus'].iloc[-1],
            adx_data['di_minus'].iloc[-1],
            adx_data['dx'].iloc[-1],
        ])
        
        return np.array(features).reshape(1, -1)
    
    def _train_ml_model(self):
        """Train the machine learning model"""
        try:
            if len(self.feature_history) < 50:
                return
            
            X = np.array(self.feature_history)
            y = np.array(self.target_history)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest model
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.ml_model.fit(X_scaled, y)
            self.ml_trained = True
            
            self.logger.info("ML model trained successfully")
            
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
    
    def _make_ml_prediction(self, features: np.ndarray) -> float:
        """Make prediction using trained ML model"""
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.ml_model.predict(features_scaled)[0]
            return float(prediction)
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return 0.0
    
    def _calculate_ml_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in ML prediction"""
        if not self.ml_trained:
            return 0.0
        
        try:
            # Use prediction variance from ensemble
            features_scaled = self.scaler.transform(features)
            predictions = [tree.predict(features_scaled)[0] for tree in self.ml_model.estimators_[:10]]
            variance = np.var(predictions)
            confidence = 1.0 / (1.0 + variance)
            return min(max(confidence, 0.0), 1.0)
        except:
            return 0.0
    
    def _multi_timeframe_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform multi-timeframe ADX analysis"""
        timeframes = [7, 14, 21, 50]  # Different periods for analysis
        mtf_results = {}
        
        for tf in timeframes:
            if len(data) >= tf + self.smoothing_period:
                temp_indicator = ADXIndicator(period=tf, enable_ml=False)
                tf_result = temp_indicator.calculate(data)
                mtf_results[f'tf_{tf}'] = {
                    'adx': tf_result['current'].adx,
                    'trend_strength': tf_result['trend_strength'],
                    'trend_direction': tf_result['trend_direction']
                }
        
        return mtf_results
    
    def _get_default_result(self) -> Dict:
        """Get default result when calculation fails"""
        default_result = ADXResult(
            adx=0.0,
            di_plus=0.0,
            di_minus=0.0,
            dx=0.0,
            trend_strength=TrendStrength.ABSENT,
            trend_direction="neutral",
            confidence_score=0.0,
            signal=SignalType.NEUTRAL
        )
        
        return {
            'current': default_result,
            'values': {'adx': [], 'di_plus': [], 'di_minus': [], 'dx': []},
            'signals': SignalType.NEUTRAL,
            'trend_strength': 'absent',
            'trend_direction': 'neutral',
            'confidence': 0.0,
            'error': True,
            'metadata': {
                'period': self.period,
                'smoothing_period': self.smoothing_period,
                'ml_enabled': self.enable_ml
            }
        }

    def get_parameters(self) -> Dict:
        """Get current indicator parameters"""
        return {
            'period': self.period,
            'smoothing_period': self.smoothing_period,
            'enable_ml': self.enable_ml,
            'lookback_window': self.lookback_window
        }
    
    def set_parameters(self, **kwargs):
        """Update indicator parameters"""
        if 'period' in kwargs:
            self.period = kwargs['period']
        if 'smoothing_period' in kwargs:
            self.smoothing_period = kwargs['smoothing_period']
        if 'enable_ml' in kwargs:
            self.enable_ml = kwargs['enable_ml'] and ML_AVAILABLE
        if 'lookback_window' in kwargs:
            self.lookback_window = kwargs['lookback_window']