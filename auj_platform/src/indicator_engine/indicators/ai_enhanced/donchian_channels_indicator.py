"""
Donchian Channels Indicator - AI Enhanced Category
==================================================

Advanced AI-enhanced Donchian Channels with adaptive periods, machine learning
breakout prediction, volatility-adjusted bands, and sophisticated trend analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import signal, stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class DonchianChannelsIndicator(StandardIndicatorInterface):
    """
    AI-Enhanced Donchian Channels with advanced features.
    
    Features:
    - Adaptive period optimization based on market volatility
    - Machine learning breakout prediction and validation
    - Multi-timeframe channel analysis and confluence
    - Volatility-adjusted channel expansion/contraction
    - Channel slope and trend strength analysis
    - Support/resistance strength quantification
    - False breakout detection and filtering
    - Dynamic stop-loss and take-profit calculations
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'base_period': 20,               # Base Donchian period
            'adaptive_periods': [10, 20, 30, 50],  # Multiple periods for analysis
            'volatility_window': 20,         # Volatility calculation window
            'breakout_confirmation': 2,      # Periods for breakout confirmation
            'min_channel_width': 0.02,       # Minimum channel width (2%)
            'max_channel_width': 0.15,       # Maximum channel width (15%)
            'slope_window': 10,              # Window for slope calculation
            'ml_prediction_window': 100,     # ML model training window
            'false_breakout_threshold': 0.5, # Threshold for false breakout detection
            'use_machine_learning': True,    # Enable ML breakout prediction
            'adaptive_periods_enabled': True, # Enable adaptive period optimization
            'volatility_adjustment': True,   # Enable volatility-based adjustments
            'multi_timeframe': True,         # Enable multi-timeframe analysis
            'trend_analysis': True,          # Enable trend strength analysis
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("DonchianChannelsIndicator", default_params)
        
        # Initialize ML models
        self.breakout_predictor = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.trend_predictor = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Cache for calculations
        self._cache = {}
        self._breakout_history = []
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["high", "low", "close", "volume"],
            min_periods=max(self.parameters['adaptive_periods']) + self.parameters['ml_prediction_window']
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced Donchian Channels with AI enhancements."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Calculate optimal period if adaptive is enabled
            optimal_period = self._calculate_optimal_period(data) if self.parameters['adaptive_periods_enabled'] else self.parameters['base_period']
            
            # Calculate base Donchian Channels for all periods
            multi_period_channels = {}
            for period in self.parameters['adaptive_periods']:
                channels = self._calculate_base_donchian(data, period)
                multi_period_channels[f'period_{period}'] = channels
            
            # Get primary period channels
            primary_channels = self._calculate_base_donchian(data, optimal_period)
            
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(data)
            
            # Apply volatility adjustments
            adjusted_channels = self._apply_volatility_adjustments(
                primary_channels, volatility_metrics, data
            )
            
            # Calculate channel properties
            channel_properties = self._calculate_channel_properties(
                adjusted_channels, data, optimal_period
            )
            
            # Trend analysis
            trend_analysis = {}
            if self.parameters['trend_analysis']:
                trend_analysis = self._analyze_trend_strength(
                    adjusted_channels, data, channel_properties
                )
            
            # Machine learning predictions
            ml_predictions = {}
            if self.parameters['use_machine_learning']:
                ml_predictions = self._calculate_ml_predictions(
                    data, adjusted_channels, channel_properties
                )
            
            # Multi-timeframe analysis
            mtf_analysis = {}
            if self.parameters['multi_timeframe']:
                mtf_analysis = self._analyze_multi_timeframe_confluence(
                    multi_period_channels, data
                )
            
            # Breakout detection and validation
            breakout_analysis = self._detect_and_validate_breakouts(
                data, adjusted_channels, ml_predictions, volatility_metrics
            )
            
            # Support/resistance analysis
            sr_analysis = self._analyze_support_resistance_strength(
                adjusted_channels, data, breakout_analysis
            )
            
            # Generate trading signals
            signals = self._generate_trading_signals(
                adjusted_channels, breakout_analysis, ml_predictions, 
                trend_analysis, volatility_metrics
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                adjusted_channels, channel_properties, ml_predictions, 
                trend_analysis, breakout_analysis
            )
            
            return {
                'upper_channel': adjusted_channels['upper'],
                'lower_channel': adjusted_channels['lower'],
                'middle_channel': adjusted_channels['middle'],
                'channel_width': channel_properties['width'],
                'optimal_period': optimal_period,
                'channel_properties': channel_properties,
                'volatility_metrics': volatility_metrics,
                'trend_analysis': trend_analysis,
                'ml_predictions': ml_predictions,
                'multi_timeframe_analysis': mtf_analysis,
                'breakout_analysis': breakout_analysis,
                'support_resistance_analysis': sr_analysis,
                'signals': signals,
                'confidence': confidence,
                'multi_period_channels': multi_period_channels,
                'channel_slope': channel_properties.get('slope', 0.0),
                'breakout_probability': ml_predictions.get('breakout_probability', 0.5),
                'trend_strength': trend_analysis.get('strength_score', 0.0)
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _calculate_optimal_period(self, data: pd.DataFrame) -> int:
        """Calculate optimal Donchian period based on market conditions."""
        try:
            # Calculate volatility regime
            returns = data['close'].pct_change().dropna()
            current_vol = returns.tail(20).std() * np.sqrt(252)
            
            # Calculate adaptive period based on volatility
            base_period = self.parameters['base_period']
            
            if current_vol > 0.4:  # High volatility
                optimal_period = int(base_period * 0.7)  # Shorter period
            elif current_vol < 0.15:  # Low volatility
                optimal_period = int(base_period * 1.3)  # Longer period
            else:
                optimal_period = base_period
            
            # Constrain to available periods
            available_periods = self.parameters['adaptive_periods']
            optimal_period = min(available_periods, key=lambda x: abs(x - optimal_period))
            
            return optimal_period
            
        except Exception:
            return self.parameters['base_period']
    
    def _calculate_base_donchian(self, data: pd.DataFrame, period: int) -> Dict[str, float]:
        """Calculate base Donchian Channels for given period."""
        high = data['high'].values
        low = data['low'].values
        
        if len(high) < period:
            return {'upper': 0.0, 'lower': 0.0, 'middle': 0.0}
        
        # Rolling max and min
        upper_channel = pd.Series(high).rolling(window=period).max().iloc[-1]
        lower_channel = pd.Series(low).rolling(window=period).min().iloc[-1]
        middle_channel = (upper_channel + lower_channel) / 2.0
        
        return {
            'upper': upper_channel,
            'lower': lower_channel,
            'middle': middle_channel
        }
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive volatility metrics."""
        window = self.parameters['volatility_window']
        
        try:
            # Price-based volatility
            returns = data['close'].pct_change().dropna()
            price_vol = returns.tail(window).std() * np.sqrt(252)
            
            # ATR-based volatility
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            atr = pd.Series(tr).rolling(window=window).mean().iloc[-1]
            atr_vol = atr / close[-1] if close[-1] != 0 else 0.0
            
            # Volume volatility
            volume_changes = data['volume'].pct_change().dropna()
            volume_vol = volume_changes.tail(window).std()
            
            # Volatility regime classification
            vol_percentile = stats.percentileofscore(returns.tail(100), returns.iloc[-1])
            
            if vol_percentile > 80:
                regime = "very_high"
            elif vol_percentile > 60:
                regime = "high"
            elif vol_percentile > 40:
                regime = "normal"
            elif vol_percentile > 20:
                regime = "low"
            else:
                regime = "very_low"
            
            # Combined volatility score
            vol_score = (price_vol + atr_vol * 252) / 2.0  # Normalize ATR to annual
            
            return {
                'price_volatility': price_vol,
                'atr_volatility': atr_vol,
                'volume_volatility': volume_vol,
                'volatility_regime': regime,
                'volatility_score': vol_score,
                'volatility_percentile': vol_percentile
            }
            
        except Exception:
            return {
                'price_volatility': 0.2,
                'atr_volatility': 0.02,
                'volume_volatility': 0.5,
                'volatility_regime': 'normal',
                'volatility_score': 0.2,
                'volatility_percentile': 50.0
            }
    
    def _apply_volatility_adjustments(self, base_channels: Dict[str, float], 
                                    volatility_metrics: Dict[str, float],
                                    data: pd.DataFrame) -> Dict[str, float]:
        """Apply volatility-based adjustments to channel bands."""
        if not self.parameters['volatility_adjustment']:
            return base_channels
        
        try:
            vol_score = volatility_metrics['volatility_score']
            current_price = data['close'].iloc[-1]
            
            # Calculate base width
            base_width = (base_channels['upper'] - base_channels['lower']) / current_price
            
            # Volatility adjustment factor
            if volatility_metrics['volatility_regime'] in ['very_high', 'high']:
                adjustment_factor = 1.0 + (vol_score - 0.2) * 0.5  # Expand channels
            elif volatility_metrics['volatility_regime'] in ['very_low', 'low']:
                adjustment_factor = max(0.7, 1.0 - (0.2 - vol_score) * 0.3)  # Contract channels
            else:
                adjustment_factor = 1.0
            
            # Apply constraints
            adjusted_width = base_width * adjustment_factor
            adjusted_width = max(self.parameters['min_channel_width'], 
                               min(adjusted_width, self.parameters['max_channel_width']))
            
            # Calculate adjusted channels
            middle = base_channels['middle']
            half_width = (adjusted_width * current_price) / 2.0
            
            return {
                'upper': middle + half_width,
                'lower': middle - half_width,
                'middle': middle,
                'adjustment_factor': adjustment_factor,
                'adjusted_width': adjusted_width
            }
            
        except Exception:
            return base_channels
    
    def _calculate_channel_properties(self, channels: Dict[str, float], 
                                    data: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate comprehensive channel properties."""
        properties = {
            'width': 0.0,
            'width_percentile': 50.0,
            'slope': 0.0,
            'position': 0.5,
            'squeeze_factor': 0.0,
            'expansion_rate': 0.0
        }
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Channel width
            width = (channels['upper'] - channels['lower']) / current_price
            properties['width'] = width
            
            # Width percentile (compared to recent history)
            if len(data) >= period * 2:
                historical_widths = []
                for i in range(period, len(data)):
                    historical_data = data.iloc[i-period:i]
                    historical_channels = self._calculate_base_donchian(historical_data, period)
                    historical_width = (historical_channels['upper'] - historical_channels['lower']) / historical_data['close'].iloc[-1]
                    historical_widths.append(historical_width)
                
                if historical_widths:
                    properties['width_percentile'] = stats.percentileofscore(historical_widths, width)
            
            # Channel slope
            slope_window = min(self.parameters['slope_window'], len(data) - 1)
            if slope_window > 2:
                recent_data = data.tail(slope_window)
                recent_middles = []
                
                for i in range(len(recent_data)):
                    if i >= period:
                        window_data = recent_data.iloc[i-period:i]
                        window_channels = self._calculate_base_donchian(window_data, period)
                        recent_middles.append(window_channels['middle'])
                
                if len(recent_middles) > 2:
                    x = np.arange(len(recent_middles))
                    slope, _, _, _, _ = stats.linregress(x, recent_middles)
                    properties['slope'] = slope / current_price  # Normalized slope
            
            # Position within channel
            if channels['upper'] != channels['lower']:
                properties['position'] = (current_price - channels['lower']) / (channels['upper'] - channels['lower'])
            
            # Squeeze factor (how tight the channel is)
            median_width = 0.05  # Assumed median width (5%)
            properties['squeeze_factor'] = max(0.0, (median_width - width) / median_width)
            
            # Expansion rate
            if len(data) >= 10:
                prev_channels = self._calculate_base_donchian(data.iloc[:-5], period)
                prev_width = (prev_channels['upper'] - prev_channels['lower']) / data['close'].iloc[-6]
                properties['expansion_rate'] = (width - prev_width) / prev_width if prev_width != 0 else 0.0
            
        except Exception:
            pass
        
        return properties
    
    def _analyze_trend_strength(self, channels: Dict[str, float], data: pd.DataFrame,
                              channel_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend strength using channel characteristics."""
        trend_analysis = {
            'direction': 'neutral',
            'strength_score': 0.0,
            'persistence': 0.0,
            'momentum': 0.0,
            'trend_quality': 'weak'
        }
        
        try:
            current_price = data['close'].iloc[-1]
            channel_slope = channel_properties.get('slope', 0.0)
            position = channel_properties.get('position', 0.5)
            
            # Trend direction
            if channel_slope > 0.001:  # 0.1% slope threshold
                direction = 'uptrend'
            elif channel_slope < -0.001:
                direction = 'downtrend'
            else:
                direction = 'sideways'
            
            # Strength based on slope magnitude and position
            slope_strength = min(abs(channel_slope) * 1000, 1.0)  # Scale slope
            
            if direction == 'uptrend':
                position_strength = position  # Higher in channel = stronger uptrend
            elif direction == 'downtrend':
                position_strength = 1.0 - position  # Lower in channel = stronger downtrend
            else:
                position_strength = 1.0 - abs(position - 0.5) * 2  # Middle = stronger sideways
            
            strength_score = (slope_strength + position_strength) / 2.0
            
            # Persistence analysis
            window = min(20, len(data) - 1)
            if window > 5:
                recent_positions = []
                for i in range(window):
                    idx = len(data) - 1 - i
                    if idx >= 0:
                        price = data['close'].iloc[idx]
                        pos = (price - channels['lower']) / (channels['upper'] - channels['lower'])
                        recent_positions.append(pos)
                
                if recent_positions:
                    # Persistence = consistency of position
                    persistence = 1.0 - np.std(recent_positions)
                    persistence = max(0.0, min(persistence, 1.0))
                else:
                    persistence = 0.0
            else:
                persistence = 0.0
            
            # Momentum analysis
            if len(data) >= 10:
                recent_closes = data['close'].tail(10).values
                momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
                momentum_score = min(abs(momentum) * 10, 1.0)  # Scale momentum
            else:
                momentum_score = 0.0
            
            # Overall trend quality
            combined_score = (strength_score + persistence + momentum_score) / 3.0
            
            if combined_score > 0.8:
                trend_quality = 'very_strong'
            elif combined_score > 0.6:
                trend_quality = 'strong'
            elif combined_score > 0.4:
                trend_quality = 'moderate'
            elif combined_score > 0.2:
                trend_quality = 'weak'
            else:
                trend_quality = 'very_weak'
            
            trend_analysis.update({
                'direction': direction,
                'strength_score': strength_score,
                'persistence': persistence,
                'momentum': momentum_score,
                'trend_quality': trend_quality
            })
            
        except Exception:
            pass
        
        return trend_analysis
    
    def _calculate_ml_predictions(self, data: pd.DataFrame, channels: Dict[str, float],
                                channel_properties: Dict[str, Any]) -> Dict[str, float]:
        """Calculate machine learning-based predictions."""
        if not self.parameters['use_machine_learning']:
            return {}
        
        try:
            # Prepare features
            features = self._prepare_ml_features(data, channels, channel_properties)
            
            if len(features) < self.parameters['ml_prediction_window']:
                return {
                    'breakout_probability': 0.5,
                    'trend_continuation_probability': 0.5,
                    'false_breakout_probability': 0.5
                }
            
            # Train models if not trained
            if not self.is_trained and len(features) >= self.parameters['ml_prediction_window']:
                self._train_ml_models(features, data)
            
            if self.is_trained:
                # Make predictions
                latest_features = features[-1:].reshape(1, -1)
                latest_features_scaled = self.scaler.transform(latest_features)
                
                # Breakout prediction
                breakout_prob = self.breakout_predictor.predict_proba(latest_features_scaled)[0][1]
                
                # Trend continuation prediction
                trend_prediction = self.trend_predictor.predict(latest_features_scaled)[0]
                trend_continuation_prob = max(0.0, min(1.0, (trend_prediction + 1) / 2))
                
                # False breakout probability (inverse of breakout confidence)
                false_breakout_prob = 1.0 - breakout_prob
                
                return {
                    'breakout_probability': breakout_prob,
                    'trend_continuation_probability': trend_continuation_prob,
                    'false_breakout_probability': false_breakout_prob,
                    'prediction_confidence': 0.7
                }
            
        except Exception:
            pass
        
        return {
            'breakout_probability': 0.5,
            'trend_continuation_probability': 0.5,
            'false_breakout_probability': 0.5
        }
    
    def _prepare_ml_features(self, data: pd.DataFrame, channels: Dict[str, float],
                           channel_properties: Dict[str, Any]) -> np.ndarray:
        """Prepare features for machine learning models."""
        features = []
        
        try:
            # Calculate features for each period
            min_period = min(self.parameters['adaptive_periods'])
            
            for i in range(min_period, len(data)):
                window_data = data.iloc[i-min_period:i]
                window_channels = self._calculate_base_donchian(window_data, min_period)
                
                # Price features
                current_price = window_data['close'].iloc[-1]
                price_position = (current_price - window_channels['lower']) / \
                               max(window_channels['upper'] - window_channels['lower'], 1e-10)
                
                # Volume features
                avg_volume = window_data['volume'].mean()
                current_volume = window_data['volume'].iloc[-1]
                volume_ratio = current_volume / max(avg_volume, 1)
                
                # Volatility features
                returns = window_data['close'].pct_change().dropna()
                volatility = returns.std()
                
                # Channel width
                width = (window_channels['upper'] - window_channels['lower']) / current_price
                
                # Momentum features
                momentum = (current_price - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
                
                feature_vector = [
                    price_position,
                    volume_ratio,
                    volatility,
                    width,
                    momentum,
                    current_price,
                    window_channels['upper'],
                    window_channels['lower']
                ]
                
                features.append(feature_vector)
            
        except Exception:
            pass
        
        return np.array(features) if features else np.array([])
    
    def _train_ml_models(self, features: np.ndarray, data: pd.DataFrame):
        """Train machine learning models."""
        try:
            if len(features) < self.parameters['ml_prediction_window']:
                return
            
            # Create targets for breakout prediction
            breakout_targets = []
            trend_targets = []
            
            for i in range(len(features) - 5):  # Predict 5 periods ahead
                current_idx = i + len(data) - len(features)
                future_idx = min(current_idx + 5, len(data) - 1)
                
                current_price = data['close'].iloc[current_idx]
                future_price = data['close'].iloc[future_idx]
                
                # Calculate channels for current period
                period = min(self.parameters['adaptive_periods'])
                start_idx = max(0, current_idx - period)
                window_data = data.iloc[start_idx:current_idx]
                
                if len(window_data) >= period:
                    channels = self._calculate_base_donchian(window_data, period)
                    
                    # Breakout target: 1 if price breaks channel, 0 otherwise
                    breakout = 1 if (future_price > channels['upper'] or future_price < channels['lower']) else 0
                    breakout_targets.append(breakout)
                    
                    # Trend target: 1 for up, -1 for down
                    trend = 1 if future_price > current_price else -1
                    trend_targets.append(trend)
            
            # Align features with targets
            train_features = features[:-5] if len(features) > 5 else features
            breakout_targets = np.array(breakout_targets[:len(train_features)])
            trend_targets = np.array(trend_targets[:len(train_features)])
            
            if len(train_features) > 0 and len(breakout_targets) > 0:
                # Scale features
                self.scaler.fit(train_features)
                train_features_scaled = self.scaler.transform(train_features)
                
                # Train breakout predictor
                self.breakout_predictor.fit(train_features_scaled, breakout_targets)
                
                # Train trend predictor
                self.trend_predictor.fit(train_features_scaled, trend_targets)
                
                self.is_trained = True
                
        except Exception:
            pass
    
    def _analyze_multi_timeframe_confluence(self, multi_period_channels: Dict[str, Dict],
                                          data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confluence across multiple timeframes."""
        confluence = {
            'resistance_confluence': 0.0,
            'support_confluence': 0.0,
            'channel_alignment': 0.0,
            'breakout_consensus': 'neutral'
        }
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Collect all resistance and support levels
            resistance_levels = []
            support_levels = []
            
            for period_key, channels in multi_period_channels.items():
                resistance_levels.append(channels['upper'])
                support_levels.append(channels['lower'])
            
            # Calculate confluence (how close levels are to each other)
            resistance_std = np.std(resistance_levels) / current_price if resistance_levels else 0
            support_std = np.std(support_levels) / current_price if support_levels else 0
            
            # Higher confluence = lower standard deviation
            confluence['resistance_confluence'] = max(0.0, 1.0 - resistance_std * 10)
            confluence['support_confluence'] = max(0.0, 1.0 - support_std * 10)
            
            # Channel alignment (all channels trending in same direction)
            channel_slopes = []
            for period_key, channels in multi_period_channels.items():
                period = int(period_key.split('_')[1])
                slope = self._calculate_channel_slope(data, period)
                channel_slopes.append(slope)
            
            if channel_slopes:
                # Alignment = consistency of slope direction
                positive_slopes = sum(1 for slope in channel_slopes if slope > 0)
                negative_slopes = sum(1 for slope in channel_slopes if slope < 0)
                
                alignment = max(positive_slopes, negative_slopes) / len(channel_slopes)
                confluence['channel_alignment'] = alignment
            
            # Breakout consensus
            upper_distances = [(level - current_price) / current_price for level in resistance_levels]
            lower_distances = [(current_price - level) / current_price for level in support_levels]
            
            avg_upper_distance = np.mean(upper_distances) if upper_distances else 0.1
            avg_lower_distance = np.mean(lower_distances) if lower_distances else 0.1
            
            if avg_upper_distance < 0.02:  # Within 2% of resistance
                confluence['breakout_consensus'] = 'bullish'
            elif avg_lower_distance < 0.02:  # Within 2% of support
                confluence['breakout_consensus'] = 'bearish'
            else:
                confluence['breakout_consensus'] = 'neutral'
            
        except Exception:
            pass
        
        return confluence
    
    def _calculate_channel_slope(self, data: pd.DataFrame, period: int) -> float:
        """Calculate channel slope for given period."""
        try:
            slope_window = min(self.parameters['slope_window'], len(data) - period)
            if slope_window < 3:
                return 0.0
            
            recent_data = data.tail(slope_window + period)
            middles = []
            
            for i in range(slope_window):
                window_data = recent_data.iloc[i:i+period]
                channels = self._calculate_base_donchian(window_data, period)
                middles.append(channels['middle'])
            
            if len(middles) > 2:
                x = np.arange(len(middles))
                slope, _, _, _, _ = stats.linregress(x, middles)
                return slope / data['close'].iloc[-1]  # Normalized slope
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_and_validate_breakouts(self, data: pd.DataFrame, channels: Dict[str, float],
                                     ml_predictions: Dict[str, float],
                                     volatility_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect and validate potential breakouts."""
        breakout_analysis = {
            'breakout_detected': False,
            'breakout_direction': 'none',
            'breakout_strength': 0.0,
            'false_breakout_risk': 0.5,
            'confirmation_signals': 0,
            'breakout_target': 0.0
        }
        
        try:
            current_price = data['close'].iloc[-1]
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].tail(20).mean()
            
            # Detect breakout
            upper_breakout = current_price > channels['upper']
            lower_breakout = current_price < channels['lower']
            
            if upper_breakout:
                breakout_analysis['breakout_detected'] = True
                breakout_analysis['breakout_direction'] = 'upward'
                breakout_distance = (current_price - channels['upper']) / channels['upper']
            elif lower_breakout:
                breakout_analysis['breakout_detected'] = True
                breakout_analysis['breakout_direction'] = 'downward'
                breakout_distance = (channels['lower'] - current_price) / channels['lower']
            else:
                breakout_distance = 0.0
            
            if breakout_analysis['breakout_detected']:
                # Breakout strength
                breakout_analysis['breakout_strength'] = min(breakout_distance * 50, 1.0)
                
                # Confirmation signals
                confirmation_count = 0
                
                # Volume confirmation
                if current_volume > avg_volume * 1.5:
                    confirmation_count += 1
                
                # ML confirmation
                ml_breakout_prob = ml_predictions.get('breakout_probability', 0.5)
                if ml_breakout_prob > 0.7:
                    confirmation_count += 1
                
                # Volatility confirmation
                if volatility_metrics['volatility_regime'] in ['high', 'very_high']:
                    confirmation_count += 1
                
                # Persistence confirmation (multiple periods above/below)
                confirmation_periods = self.parameters['breakout_confirmation']
                if len(data) >= confirmation_periods:
                    recent_prices = data['close'].tail(confirmation_periods).values
                    if breakout_analysis['breakout_direction'] == 'upward':
                        if all(price > channels['upper'] for price in recent_prices):
                            confirmation_count += 1
                    else:
                        if all(price < channels['lower'] for price in recent_prices):
                            confirmation_count += 1
                
                breakout_analysis['confirmation_signals'] = confirmation_count
                
                # False breakout risk
                false_breakout_prob = ml_predictions.get('false_breakout_probability', 0.5)
                risk_factors = [
                    1.0 - (confirmation_count / 4.0),  # Lack of confirmations
                    false_breakout_prob,
                    1.0 - breakout_analysis['breakout_strength']  # Weak breakout
                ]
                
                breakout_analysis['false_breakout_risk'] = np.mean(risk_factors)
                
                # Breakout target (projected move)
                channel_width = channels['upper'] - channels['lower']
                if breakout_analysis['breakout_direction'] == 'upward':
                    breakout_analysis['breakout_target'] = channels['upper'] + channel_width
                else:
                    breakout_analysis['breakout_target'] = channels['lower'] - channel_width
            
            # Store breakout in history
            self._breakout_history.append({
                'timestamp': len(data),
                'detected': breakout_analysis['breakout_detected'],
                'direction': breakout_analysis['breakout_direction'],
                'strength': breakout_analysis['breakout_strength']
            })
            
            # Keep only recent history
            if len(self._breakout_history) > 50:
                self._breakout_history = self._breakout_history[-50:]
            
        except Exception:
            pass
        
        return breakout_analysis
    
    def _analyze_support_resistance_strength(self, channels: Dict[str, float], 
                                           data: pd.DataFrame,
                                           breakout_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strength of support and resistance levels."""
        sr_analysis = {
            'resistance_strength': 0.0,
            'support_strength': 0.0,
            'resistance_tests': 0,
            'support_tests': 0,
            'breakout_potential': 0.0
        }
        
        try:
            # Look at recent price action near channels
            lookback = min(50, len(data))
            recent_data = data.tail(lookback)
            
            resistance_level = channels['upper']
            support_level = channels['lower']
            
            # Count tests of levels
            resistance_tests = 0
            support_tests = 0
            resistance_touches = []
            support_touches = []
            
            for i, row in recent_data.iterrows():
                high = row['high']
                low = row['low']
                close = row['close']
                
                # Resistance tests (price approaches but doesn't break)
                if abs(high - resistance_level) / resistance_level < 0.01:  # Within 1%
                    resistance_tests += 1
                    resistance_touches.append(close)
                
                # Support tests
                if abs(low - support_level) / support_level < 0.01:  # Within 1%
                    support_tests += 1
                    support_touches.append(close)
            
            # Strength based on number of tests and price behavior
            max_tests = 10  # Normalize by maximum expected tests
            
            sr_analysis['resistance_tests'] = resistance_tests
            sr_analysis['support_tests'] = support_tests
            
            # Calculate strength (more tests = stronger level, but diminishing returns)
            sr_analysis['resistance_strength'] = min(resistance_tests / max_tests, 1.0) * 0.7
            sr_analysis['support_strength'] = min(support_tests / max_tests, 1.0) * 0.7
            
            # Add strength based on volume at tests
            if resistance_touches:
                # Higher volume during resistance tests = stronger resistance
                resistance_volume_factor = 0.3  # Additional strength component
                sr_analysis['resistance_strength'] += resistance_volume_factor
            
            if support_touches:
                support_volume_factor = 0.3
                sr_analysis['support_strength'] += support_volume_factor
            
            # Breakout potential (weaker levels = higher breakout potential)
            avg_strength = (sr_analysis['resistance_strength'] + sr_analysis['support_strength']) / 2.0
            sr_analysis['breakout_potential'] = max(0.0, 1.0 - avg_strength)
            
            # Adjust based on breakout analysis
            if breakout_analysis['breakout_detected']:
                sr_analysis['breakout_potential'] = min(sr_analysis['breakout_potential'] + 0.3, 1.0)
            
        except Exception:
            pass
        
        return sr_analysis
    
    def _generate_trading_signals(self, channels: Dict[str, float], 
                                breakout_analysis: Dict[str, Any],
                                ml_predictions: Dict[str, float],
                                trend_analysis: Dict[str, Any],
                                volatility_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive trading signals."""
        signals = {
            'signal_type': 'neutral',
            'signal_strength': 0.0,
            'confidence': 0.0,
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': [],
            'position_size': 'normal'
        }
        
        try:
            # Breakout signals
            if breakout_analysis['breakout_detected']:
                if (breakout_analysis['confirmation_signals'] >= 2 and 
                    breakout_analysis['false_breakout_risk'] < 0.4):
                    
                    signals['signal_type'] = f"breakout_{breakout_analysis['breakout_direction']}"
                    signals['signal_strength'] = breakout_analysis['breakout_strength']
                    
                    # Entry and exit levels for breakout
                    if breakout_analysis['breakout_direction'] == 'upward':
                        signals['entry_price'] = channels['upper'] * 1.001  # Slight above resistance
                        signals['stop_loss'] = channels['middle']
                        signals['take_profit'] = [
                            breakout_analysis['breakout_target'],
                            breakout_analysis['breakout_target'] * 1.05
                        ]
                    else:
                        signals['entry_price'] = channels['lower'] * 0.999  # Slight below support
                        signals['stop_loss'] = channels['middle']
                        signals['take_profit'] = [
                            breakout_analysis['breakout_target'],
                            breakout_analysis['breakout_target'] * 0.95
                        ]
            
            # Range trading signals
            elif not breakout_analysis['breakout_detected']:
                trend_direction = trend_analysis.get('direction', 'neutral')
                trend_strength = trend_analysis.get('strength_score', 0.0)
                
                if trend_direction in ['uptrend', 'downtrend'] and trend_strength > 0.6:
                    signals['signal_type'] = f"trend_{trend_direction}"
                    signals['signal_strength'] = trend_strength
                    
                    if trend_direction == 'uptrend':
                        signals['entry_price'] = channels['lower'] * 1.01  # Near support
                        signals['stop_loss'] = channels['lower'] * 0.98
                        signals['take_profit'] = [channels['middle'], channels['upper'] * 0.99]
                    else:
                        signals['entry_price'] = channels['upper'] * 0.99  # Near resistance
                        signals['stop_loss'] = channels['upper'] * 1.02
                        signals['take_profit'] = [channels['middle'], channels['lower'] * 1.01]
                
                elif trend_direction == 'sideways':
                    signals['signal_type'] = 'range_trading'
                    signals['signal_strength'] = 1.0 - trend_strength  # Stronger sideways = better range
                    
                    # Range trading: buy support, sell resistance
                    signals['entry_price'] = channels['lower'] * 1.005  # Near support
                    signals['stop_loss'] = channels['lower'] * 0.995
                    signals['take_profit'] = [channels['upper'] * 0.995]
            
            # Position sizing based on volatility and confidence
            vol_regime = volatility_metrics.get('volatility_regime', 'normal')
            
            if vol_regime in ['very_high']:
                signals['position_size'] = 'small'
            elif vol_regime in ['high']:
                signals['position_size'] = 'normal'
            elif vol_regime in ['low', 'very_low']:
                signals['position_size'] = 'large'
            else:
                signals['position_size'] = 'normal'
            
            # Overall confidence
            ml_confidence = ml_predictions.get('prediction_confidence', 0.5)
            trend_confidence = trend_analysis.get('strength_score', 0.0)
            breakout_confidence = 1.0 - breakout_analysis.get('false_breakout_risk', 0.5)
            
            signals['confidence'] = (signals['signal_strength'] + ml_confidence + 
                                   trend_confidence + breakout_confidence) / 4.0
            
        except Exception:
            pass
        
        return signals
    
    def _calculate_confidence_score(self, channels: Dict[str, float],
                                  channel_properties: Dict[str, Any],
                                  ml_predictions: Dict[str, float],
                                  trend_analysis: Dict[str, Any],
                                  breakout_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        try:
            # Component confidences
            
            # Channel quality confidence
            width_percentile = channel_properties.get('width_percentile', 50.0)
            channel_confidence = min(width_percentile / 100.0, 1.0)
            
            # ML confidence
            ml_confidence = ml_predictions.get('prediction_confidence', 0.5)
            
            # Trend confidence
            trend_confidence = trend_analysis.get('strength_score', 0.0)
            
            # Breakout confidence
            if breakout_analysis.get('breakout_detected', False):
                breakout_confidence = 1.0 - breakout_analysis.get('false_breakout_risk', 0.5)
            else:
                breakout_confidence = 0.7  # Neutral confidence for no breakout
            
            # Combined confidence
            weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
            confidence_components = [channel_confidence, ml_confidence, trend_confidence, breakout_confidence]
            
            overall_confidence = sum(w * c for w, c in zip(weights, confidence_components))
            
            return min(max(overall_confidence, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when insufficient data."""
        return {
            'upper_channel': 0.0,
            'lower_channel': 0.0,
            'middle_channel': 0.0,
            'channel_width': 0.0,
            'optimal_period': self.parameters['base_period'],
            'signals': {'signal_type': 'neutral', 'signal_strength': 0.0, 'confidence': 0.0},
            'confidence': 0.0,
            'channel_slope': 0.0,
            'breakout_probability': 0.5,
            'trend_strength': 0.0
        }
    
    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        return self._get_default_output()
