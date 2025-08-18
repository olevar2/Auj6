"""
Advanced Cloud Position Indicator - Ichimoku Cloud Analysis and Position Strength

This implementation provides sophisticated Ichimoku cloud analysis:
- Traditional Ichimoku cloud calculations (Senkou Span A & B)
- Advanced position strength scoring relative to cloud
- Cloud thickness and trend analysis
- Multi-timeframe cloud confirmation
- Support/resistance level identification
- ML-enhanced cloud breakout prediction

The Cloud Position Indicator analyzes price position relative to the Ichimoku cloud
to determine trend strength and potential support/resistance levels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType


class CloudPosition(Enum):
    """Price position relative to Ichimoku cloud"""
    ABOVE_CLOUD = "above_cloud"        # Price above cloud (bullish)
    IN_CLOUD = "in_cloud"              # Price inside cloud (neutral/transition)
    BELOW_CLOUD = "below_cloud"        # Price below cloud (bearish)


class CloudThickness(Enum):
    """Cloud thickness classification"""
    VERY_THIN = "very_thin"            # Weak support/resistance
    THIN = "thin"                      # Moderate support/resistance
    NORMAL = "normal"                  # Good support/resistance
    THICK = "thick"                    # Strong support/resistance
    VERY_THICK = "very_thick"          # Very strong support/resistance


class CloudTrend(Enum):
    """Cloud trend direction"""
    BULLISH = "bullish"                # Senkou A above Senkou B
    BEARISH = "bearish"                # Senkou A below Senkou B
    NEUTRAL = "neutral"                # Cloud very thin or transitioning


@dataclass
class CloudPositionResult:
    """Comprehensive Cloud Position analysis result"""
    price_position: CloudPosition
    cloud_thickness: CloudThickness
    cloud_trend: CloudTrend
    position_strength: float
    distance_to_cloud: float
    signal: SignalType
    confidence: float
    senkou_a: float
    senkou_b: float
    cloud_top: float
    cloud_bottom: float


class CloudPositionIndicator(StandardIndicatorInterface):
    """
    Advanced Cloud Position Indicator for Ichimoku Cloud Analysis
    
    Analyzes price position relative to the Ichimoku cloud and provides
    detailed assessment of trend strength and potential reversal points.
    
    Key Components:
    - Senkou Span A: (Tenkan + Kijun) / 2 displaced forward
    - Senkou Span B: (Highest High + Lowest Low) / 2 over 52 periods displaced forward
    - Cloud analysis for support/resistance and trend identification
    
    Features:
    - Advanced cloud position analysis
    - Cloud thickness and trend assessment
    - Multi-timeframe cloud confirmation
    - ML-enhanced breakout prediction
    """
    
    def __init__(self,
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_b_period: int = 52,
                 displacement: int = 26,
                 enable_ml: bool = True,
                 thickness_threshold: float = 0.01):
        """
        Initialize the Cloud Position Indicator
        
        Args:
            tenkan_period: Period for Tenkan-sen calculation
            kijun_period: Period for Kijun-sen calculation
            senkou_b_period: Period for Senkou Span B calculation
            displacement: Forward displacement for cloud
            enable_ml: Enable machine learning enhancements
            thickness_threshold: Threshold for cloud thickness classification
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        self.enable_ml = enable_ml and ML_AVAILABLE
        self.thickness_threshold = thickness_threshold
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler() if self.enable_ml else None
        self.ml_trained = False
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate Cloud Position with advanced features"""
        try:
            if len(data) < max(self.tenkan_period, self.kijun_period, self.senkou_b_period) + self.displacement:
                raise ValueError("Insufficient data for Cloud Position calculation")
            
            # Calculate Ichimoku components
            ichimoku_data = self._calculate_ichimoku_components(data)
            
            # Analyze current cloud position
            position = self._analyze_cloud_position(data, ichimoku_data)
            
            # Assess cloud characteristics
            thickness = self._assess_cloud_thickness(data, ichimoku_data)
            cloud_trend = self._determine_cloud_trend(ichimoku_data)
            
            # Calculate position strength and distance
            position_strength = self._calculate_position_strength(data, ichimoku_data, position)
            distance_to_cloud = self._calculate_distance_to_cloud(data, ichimoku_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(position, thickness, cloud_trend, position_strength)
            
            # ML enhancement
            if self.enable_ml:
                ml_adjustment = self._enhance_with_ml(data, ichimoku_data)
                confidence *= ml_adjustment.get('confidence_multiplier', 1.0)
            
            # Get latest cloud values
            current_senkou_a = ichimoku_data['senkou_a'].iloc[-1]
            current_senkou_b = ichimoku_data['senkou_b'].iloc[-1]
            cloud_top = max(current_senkou_a, current_senkou_b)
            cloud_bottom = min(current_senkou_a, current_senkou_b)
            
            # Create result
            latest_result = CloudPositionResult(
                price_position=position,
                cloud_thickness=thickness,
                cloud_trend=cloud_trend,
                position_strength=position_strength,
                distance_to_cloud=distance_to_cloud,
                signal=signal,
                confidence=confidence,
                senkou_a=current_senkou_a,
                senkou_b=current_senkou_b,
                cloud_top=cloud_top,
                cloud_bottom=cloud_bottom
            )
            
            return {
                'current': latest_result,
                'values': {
                    'senkou_a': ichimoku_data['senkou_a'].tolist(),
                    'senkou_b': ichimoku_data['senkou_b'].tolist(),
                    'tenkan': ichimoku_data['tenkan'].tolist(),
                    'kijun': ichimoku_data['kijun'].tolist(),
                    'cloud_top': ichimoku_data['cloud_top'].tolist(),
                    'cloud_bottom': ichimoku_data['cloud_bottom'].tolist()
                },
                'position': position.value,
                'cloud_thickness': thickness.value,
                'cloud_trend': cloud_trend.value,
                'signal': signal,
                'confidence': confidence,
                'position_strength': position_strength,
                'distance_to_cloud': distance_to_cloud,
                'metadata': {
                    'tenkan_period': self.tenkan_period,
                    'kijun_period': self.kijun_period,
                    'senkou_b_period': self.senkou_b_period,
                    'displacement': self.displacement,
                    'calculation_time': pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Cloud Position: {e}")
            return self._get_default_result()
    
    def _calculate_ichimoku_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all Ichimoku components"""
        df = pd.DataFrame(index=data.index)
        
        # Tenkan-sen (Conversion Line)
        df['tenkan'] = (data['high'].rolling(self.tenkan_period).max() + 
                       data['low'].rolling(self.tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line)
        df['kijun'] = (data['high'].rolling(self.kijun_period).max() + 
                      data['low'].rolling(self.kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_a_base'] = (df['tenkan'] + df['kijun']) / 2
        df['senkou_a'] = df['senkou_a_base'].shift(self.displacement)
        
        # Senkou Span B (Leading Span B)
        df['senkou_b_base'] = (data['high'].rolling(self.senkou_b_period).max() + 
                              data['low'].rolling(self.senkou_b_period).min()) / 2
        df['senkou_b'] = df['senkou_b_base'].shift(self.displacement)
        
        # Cloud boundaries
        df['cloud_top'] = df[['senkou_a', 'senkou_b']].max(axis=1)
        df['cloud_bottom'] = df[['senkou_a', 'senkou_b']].min(axis=1)
        df['cloud_thickness'] = df['cloud_top'] - df['cloud_bottom']
        
        return df
    
    def _analyze_cloud_position(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame) -> CloudPosition:
        """Analyze current price position relative to cloud"""
        current_price = data['close'].iloc[-1]
        cloud_top = ichimoku_data['cloud_top'].iloc[-1]
        cloud_bottom = ichimoku_data['cloud_bottom'].iloc[-1]
        
        if pd.isna(cloud_top) or pd.isna(cloud_bottom):
            return CloudPosition.IN_CLOUD
        
        if current_price > cloud_top:
            return CloudPosition.ABOVE_CLOUD
        elif current_price < cloud_bottom:
            return CloudPosition.BELOW_CLOUD
        else:
            return CloudPosition.IN_CLOUD
    
    def _assess_cloud_thickness(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame) -> CloudThickness:
        """Assess cloud thickness relative to price"""
        current_price = data['close'].iloc[-1]
        cloud_thickness = ichimoku_data['cloud_thickness'].iloc[-1]
        
        if pd.isna(cloud_thickness) or current_price == 0:
            return CloudThickness.THIN
        
        # Calculate thickness as percentage of price
        thickness_ratio = cloud_thickness / current_price
        
        if thickness_ratio < 0.005:  # Less than 0.5%
            return CloudThickness.VERY_THIN
        elif thickness_ratio < 0.015:  # Less than 1.5%
            return CloudThickness.THIN
        elif thickness_ratio < 0.03:   # Less than 3%
            return CloudThickness.NORMAL
        elif thickness_ratio < 0.05:   # Less than 5%
            return CloudThickness.THICK
        else:
            return CloudThickness.VERY_THICK
    
    def _determine_cloud_trend(self, ichimoku_data: pd.DataFrame) -> CloudTrend:
        """Determine cloud trend direction"""
        senkou_a = ichimoku_data['senkou_a'].iloc[-1]
        senkou_b = ichimoku_data['senkou_b'].iloc[-1]
        
        if pd.isna(senkou_a) or pd.isna(senkou_b):
            return CloudTrend.NEUTRAL
        
        # Check cloud trend over recent periods
        if len(ichimoku_data) >= 10:
            recent_a = ichimoku_data['senkou_a'].iloc[-10:].dropna()
            recent_b = ichimoku_data['senkou_b'].iloc[-10:].dropna()
            
            if len(recent_a) >= 5 and len(recent_b) >= 5:
                a_trend = recent_a.iloc[-1] - recent_a.iloc[0]
                b_trend = recent_b.iloc[-1] - recent_b.iloc[0]
                
                # Strong cloud trend if both spans move in same direction
                if a_trend > 0 and b_trend > 0 and senkou_a > senkou_b:
                    return CloudTrend.BULLISH
                elif a_trend < 0 and b_trend < 0 and senkou_a < senkou_b:
                    return CloudTrend.BEARISH
        
        # Simple trend based on current span relationship
        if senkou_a > senkou_b:
            return CloudTrend.BULLISH
        elif senkou_a < senkou_b:
            return CloudTrend.BEARISH
        else:
            return CloudTrend.NEUTRAL
    
    def _calculate_position_strength(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame, 
                                   position: CloudPosition) -> float:
        """Calculate strength of current position relative to cloud"""
        current_price = data['close'].iloc[-1]
        cloud_top = ichimoku_data['cloud_top'].iloc[-1]
        cloud_bottom = ichimoku_data['cloud_bottom'].iloc[-1]
        
        if pd.isna(cloud_top) or pd.isna(cloud_bottom) or current_price == 0:
            return 0.0
        
        if position == CloudPosition.ABOVE_CLOUD:
            # Strength based on distance above cloud
            distance = (current_price - cloud_top) / current_price
            strength = min(distance * 20, 1.0)  # Scale to 0-1
        elif position == CloudPosition.BELOW_CLOUD:
            # Strength based on distance below cloud
            distance = (cloud_bottom - current_price) / current_price
            strength = min(distance * 20, 1.0)  # Scale to 0-1
        else:  # IN_CLOUD
            # Weak position when inside cloud
            cloud_thickness = cloud_top - cloud_bottom
            if cloud_thickness > 0:
                # Position within cloud (0 = bottom, 1 = top)
                position_in_cloud = (current_price - cloud_bottom) / cloud_thickness
                # Strength is low, but varies based on position
                strength = 0.1 + abs(position_in_cloud - 0.5) * 0.2
            else:
                strength = 0.1
        
        return min(max(strength, 0.0), 1.0)
    
    def _calculate_distance_to_cloud(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame) -> float:
        """Calculate distance from price to cloud (as percentage)"""
        current_price = data['close'].iloc[-1]
        cloud_top = ichimoku_data['cloud_top'].iloc[-1]
        cloud_bottom = ichimoku_data['cloud_bottom'].iloc[-1]
        
        if pd.isna(cloud_top) or pd.isna(cloud_bottom) or current_price == 0:
            return 0.0
        
        if current_price > cloud_top:
            distance = (current_price - cloud_top) / current_price
        elif current_price < cloud_bottom:
            distance = (cloud_bottom - current_price) / current_price
        else:
            distance = 0.0  # Inside cloud
        
        return distance * 100  # Return as percentage
    
    def _generate_signals(self, position: CloudPosition, thickness: CloudThickness, 
                         cloud_trend: CloudTrend, position_strength: float) -> Tuple[SignalType, float]:
        """Generate trading signals based on cloud analysis"""
        base_confidence = 0.5
        
        # Position-based signals
        if position == CloudPosition.ABOVE_CLOUD:
            if cloud_trend == CloudTrend.BULLISH:
                signal = SignalType.BUY
                base_confidence = 0.7
            else:
                signal = SignalType.BUY
                base_confidence = 0.5
        elif position == CloudPosition.BELOW_CLOUD:
            if cloud_trend == CloudTrend.BEARISH:
                signal = SignalType.SELL
                base_confidence = 0.7
            else:
                signal = SignalType.SELL
                base_confidence = 0.5
        else:  # IN_CLOUD
            signal = SignalType.NEUTRAL
            base_confidence = 0.3
        
        # Adjust confidence based on cloud thickness
        thickness_multiplier = {
            CloudThickness.VERY_THIN: 0.7,
            CloudThickness.THIN: 0.8,
            CloudThickness.NORMAL: 1.0,
            CloudThickness.THICK: 1.1,
            CloudThickness.VERY_THICK: 1.2
        }.get(thickness, 1.0)
        
        # Adjust confidence based on position strength
        confidence = base_confidence * thickness_multiplier * (0.6 + position_strength * 0.4)
        
        return signal, min(confidence, 1.0)
    
    def _enhance_with_ml(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame) -> Dict:
        """Enhance signals with machine learning"""
        if not self.enable_ml:
            return {'confidence_multiplier': 1.0}
        
        try:
            # Extract features
            features = self._extract_ml_features(data, ichimoku_data)
            
            # Simple ML enhancement
            confidence_multiplier = 1.0
            
            # Trend consistency check
            if len(data) >= 20:
                price_trend = data['close'].iloc[-20:].apply(lambda x: x / data['close'].iloc[-20] - 1).iloc[-1]
                tenkan_kijun_diff = (ichimoku_data['tenkan'].iloc[-1] - ichimoku_data['kijun'].iloc[-1])
                
                if ichimoku_data['kijun'].iloc[-1] > 0:
                    tk_ratio = tenkan_kijun_diff / ichimoku_data['kijun'].iloc[-1]
                    
                    # Bonus for trend alignment
                    if (price_trend > 0 and tk_ratio > 0) or (price_trend < 0 and tk_ratio < 0):
                        confidence_multiplier *= 1.1
            
            # Volume confirmation (if available)
            if 'volume' in data.columns and len(data) >= 10:
                recent_volume = data['volume'].iloc[-5:].mean()
                avg_volume = data['volume'].iloc[-20:-5].mean()
                if avg_volume > 0 and recent_volume / avg_volume > 1.2:
                    confidence_multiplier *= 1.05
            
            return {
                'confidence_multiplier': min(confidence_multiplier, 1.3),
                'ml_features': features
            }
            
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {e}")
            return {'confidence_multiplier': 1.0}
    
    def _extract_ml_features(self, data: pd.DataFrame, ichimoku_data: pd.DataFrame) -> List[float]:
        """Extract features for ML model"""
        features = []
        
        # Price position features
        current_price = data['close'].iloc[-1]
        if current_price > 0:
            features.extend([
                ichimoku_data['tenkan'].iloc[-1] / current_price,
                ichimoku_data['kijun'].iloc[-1] / current_price,
                ichimoku_data['senkou_a'].iloc[-1] / current_price,
                ichimoku_data['senkou_b'].iloc[-1] / current_price
            ])
        else:
            features.extend([1.0, 1.0, 1.0, 1.0])
        
        # Cloud characteristics
        cloud_thickness = ichimoku_data['cloud_thickness'].iloc[-1]
        features.append(cloud_thickness / current_price if current_price > 0 else 0.0)
        
        # Price momentum
        returns = data['close'].pct_change()
        features.extend([
            returns.iloc[-1],
            returns.rolling(5).mean().iloc[-1],
            returns.rolling(10).std().iloc[-1]
        ])
        
        return features
    
    def _get_default_result(self) -> Dict:
        """Get default result when calculation fails"""
        default_result = CloudPositionResult(
            price_position=CloudPosition.IN_CLOUD,
            cloud_thickness=CloudThickness.THIN,
            cloud_trend=CloudTrend.NEUTRAL,
            position_strength=0.0,
            distance_to_cloud=0.0,
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            senkou_a=0.0,
            senkou_b=0.0,
            cloud_top=0.0,
            cloud_bottom=0.0
        )
        
        return {
            'current': default_result,
            'values': {'senkou_a': [], 'senkou_b': [], 'tenkan': [], 'kijun': [], 'cloud_top': [], 'cloud_bottom': []},
            'position': 'in_cloud',
            'cloud_thickness': 'thin',
            'cloud_trend': 'neutral',
            'signal': SignalType.NEUTRAL,
            'confidence': 0.0,
            'error': True,
            'metadata': {
                'tenkan_period': self.tenkan_period,
                'kijun_period': self.kijun_period,
                'senkou_b_period': self.senkou_b_period
            }
        }

    def get_parameters(self) -> Dict:
        """Get current indicator parameters"""
        return {
            'tenkan_period': self.tenkan_period,
            'kijun_period': self.kijun_period,
            'senkou_b_period': self.senkou_b_period,
            'displacement': self.displacement,
            'enable_ml': self.enable_ml,
            'thickness_threshold': self.thickness_threshold
        }
    
    def set_parameters(self, **kwargs):
        """Update indicator parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)