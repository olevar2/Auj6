"""
Bears and Bulls Power Indicator - Advanced Implementation
======================================================

Dr. Alexander Elder's Bears and Bulls Power indicators with AI enhancement,
sophisticated market power analysis, and institutional flow detection for
identifying market dominance and potential reversal points.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType,
    IndicatorResult
)
from ....core.exceptions import IndicatorCalculationException


class BearsAndBullsPowerIndicator(StandardIndicatorInterface):
    """
    Advanced Bears and Bulls Power Indicator Implementation
    
    Features:
    - Elder's Bulls/Bears Power with AI enhancement
    - Institutional flow detection and analysis
    - Dynamic power threshold adaptation
    - Multi-timeframe power convergence
    - Advanced divergence detection
    - Market regime classification
    - Sophisticated signal generation with confidence scoring
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'ema_period': 13,               # EMA period for power calculation
            'power_threshold': 0.001,       # Power signal threshold
            'divergence_periods': 15,       # Divergence detection period
            'ml_lookback': 80,              # ML analysis lookback
            'regime_periods': 50,           # Market regime analysis period
            'institutional_threshold': 2.0,  # Institutional flow threshold
            'signal_confirmation': 3,       # Signal confirmation bars
            'adaptive_thresholds': True,    # Enable adaptive thresholds
            'volume_analysis': True,        # Include volume analysis
            'market_context': True          # Market context analysis
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="BearsAndBullsPowerIndicator", parameters=default_params)
        
        # Initialize AI components
        self.scaler = StandardScaler()
        self.regime_classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.power_predictor = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.ml_trained = False
        
        # Power analysis storage
        self.power_history = []
        self.regime_history = []
        self.institutional_flows = []
    
    def get_data_requirements(self) -> DataRequirement:
        """Define data requirements for Bears/Bulls Power calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(self.parameters['ema_period'], self.parameters['ml_lookback']) + 20,
            lookback_periods=200
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        required_params = ['ema_period', 'power_threshold']
        
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")
            
            if not isinstance(self.parameters[param], (int, float)) or self.parameters[param] <= 0:
                raise ValueError(f"Parameter {param} must be a positive number")
        
        return True
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average with enhanced precision"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_bulls_power(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Bulls Power: High - EMA(Close, period)
        Measures the ability of bulls to drive prices above the average
        """
        ema_close = self._calculate_ema(data['close'], self.parameters['ema_period'])
        bulls_power = data['high'] - ema_close
        return bulls_power
    
    def _calculate_bears_power(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Bears Power: Low - EMA(Close, period)
        Measures the ability of bears to drive prices below the average
        """
        ema_close = self._calculate_ema(data['close'], self.parameters['ema_period'])
        bears_power = data['low'] - ema_close
        return bears_power
    
    def _detect_institutional_flows(self, bulls_power: pd.Series, bears_power: pd.Series, 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect institutional money flow using power analysis and volume
        """
        if len(bulls_power) < 20:
            return {'detected': False, 'strength': 0.0, 'direction': 'neutral'}
        
        # Calculate volume-weighted power
        volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Institutional flow indicators
        bulls_institutional = bulls_power * volume_ratio
        bears_institutional = abs(bears_power) * volume_ratio
        
        # Detect unusual institutional activity
        bulls_threshold = bulls_institutional.rolling(window=20).mean() + \
                         bulls_institutional.rolling(window=20).std() * self.parameters['institutional_threshold']
        
        bears_threshold = bears_institutional.rolling(window=20).mean() + \
                         bears_institutional.rolling(window=20).std() * self.parameters['institutional_threshold']
        
        current_bulls_inst = bulls_institutional.iloc[-1]
        current_bears_inst = bears_institutional.iloc[-1]
        
        institutional_detected = False
        direction = 'neutral'
        strength = 0.0
        
        if current_bulls_inst > bulls_threshold.iloc[-1]:
            institutional_detected = True
            direction = 'bullish'
            strength = min((current_bulls_inst - bulls_threshold.iloc[-1]) / bulls_threshold.iloc[-1], 1.0)
        elif current_bears_inst > bears_threshold.iloc[-1]:
            institutional_detected = True
            direction = 'bearish'
            strength = min((current_bears_inst - bears_threshold.iloc[-1]) / bears_threshold.iloc[-1], 1.0)
        
        return {
            'detected': institutional_detected,
            'direction': direction,
            'strength': float(strength),
            'bulls_institutional': float(current_bulls_inst),
            'bears_institutional': float(current_bears_inst)
        }
    
    def _analyze_market_regime(self, bulls_power: pd.Series, bears_power: pd.Series, 
                              data: pd.DataFrame) -> str:
        """
        Classify current market regime using power analysis
        """
        if len(bulls_power) < self.parameters['regime_periods']:
            return 'undefined'
        
        recent_bulls = bulls_power.tail(self.parameters['regime_periods'])
        recent_bears = bears_power.tail(self.parameters['regime_periods'])
        recent_prices = data['close'].tail(self.parameters['regime_periods'])
        
        # Calculate regime indicators
        bulls_dominance = (recent_bulls > 0).sum() / len(recent_bulls)
        bears_dominance = (recent_bears < 0).sum() / len(recent_bears)
        
        bulls_strength = recent_bulls.mean()
        bears_strength = abs(recent_bears.mean())
        
        price_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        volatility = recent_prices.pct_change().std()
        
        # Regime classification
        if bulls_dominance > 0.7 and bulls_strength > bears_strength and price_trend > 0.02:
            return 'strong_bull_market'
        elif bulls_dominance > 0.6 and price_trend > 0.01:
            return 'bull_market'
        elif bears_dominance > 0.7 and bears_strength > bulls_strength and price_trend < -0.02:
            return 'strong_bear_market'
        elif bears_dominance > 0.6 and price_trend < -0.01:
            return 'bear_market'
        elif volatility > recent_prices.pct_change().rolling(window=50).std().mean():
            return 'high_volatility'
        else:
            return 'sideways_market'
    
    def _detect_power_divergences(self, bulls_power: pd.Series, bears_power: pd.Series, 
                                 prices: pd.Series) -> Dict[str, Any]:
        """
        Detect divergences between power indicators and price action
        """
        if len(bulls_power) < self.parameters['divergence_periods']:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0.0}
        
        period = self.parameters['divergence_periods']
        recent_bulls = bulls_power.tail(period)
        recent_bears = bears_power.tail(period)
        recent_prices = prices.tail(period)
        
        # Find local extremes
        price_highs = recent_prices.rolling(window=3).max() == recent_prices
        price_lows = recent_prices.rolling(window=3).min() == recent_prices
        
        bulls_highs = recent_bulls.rolling(window=3).max() == recent_bulls
        bears_lows = recent_bears.rolling(window=3).min() == recent_bears
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        
        # Bullish divergence: Price makes lower low, Bears Power makes higher low
        price_low_indices = recent_prices[price_lows].index
        bears_low_indices = recent_bears[bears_lows].index
        
        if len(price_low_indices) >= 2 and len(bears_low_indices) >= 2:
            last_price_low = recent_prices[price_low_indices[-1]]
            prev_price_low = recent_prices[price_low_indices[-2]]
            
            last_bears_low = recent_bears[bears_low_indices[-1]]
            prev_bears_low = recent_bears[bears_low_indices[-2]]
            
            if last_price_low < prev_price_low and last_bears_low > prev_bears_low:
                bullish_divergence = True
                divergence_strength = abs(last_bears_low - prev_bears_low) / abs(prev_bears_low) if prev_bears_low != 0 else 0
        
        # Bearish divergence: Price makes higher high, Bulls Power makes lower high
        price_high_indices = recent_prices[price_highs].index
        bulls_high_indices = recent_bulls[bulls_highs].index
        
        if len(price_high_indices) >= 2 and len(bulls_high_indices) >= 2:
            last_price_high = recent_prices[price_high_indices[-1]]
            prev_price_high = recent_prices[price_high_indices[-2]]
            
            last_bulls_high = recent_bulls[bulls_high_indices[-1]]
            prev_bulls_high = recent_bulls[bulls_high_indices[-2]]
            
            if last_price_high > prev_price_high and last_bulls_high < prev_bulls_high:
                bearish_divergence = True
                divergence_strength = abs(last_bulls_high - prev_bulls_high) / abs(prev_bulls_high) if prev_bulls_high != 0 else 0
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': min(divergence_strength, 1.0)
        }
    
    def _calculate_adaptive_thresholds(self, bulls_power: pd.Series, bears_power: pd.Series) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on historical power distribution
        """
        if not self.parameters['adaptive_thresholds'] or len(bulls_power) < 50:
            return {
                'bulls_threshold': self.parameters['power_threshold'],
                'bears_threshold': -self.parameters['power_threshold']
            }
        
        # Calculate dynamic thresholds based on historical distribution
        bulls_std = bulls_power.rolling(window=50).std().iloc[-1]
        bears_std = bears_power.rolling(window=50).std().iloc[-1]
        
        bulls_mean = bulls_power.rolling(window=50).mean().iloc[-1]
        bears_mean = bears_power.rolling(window=50).mean().iloc[-1]
        
        # Adaptive thresholds using statistical approach
        bulls_threshold = max(bulls_mean + 0.5 * bulls_std, self.parameters['power_threshold'])
        bears_threshold = min(bears_mean - 0.5 * bears_std, -self.parameters['power_threshold'])
        
        return {
            'bulls_threshold': float(bulls_threshold),
            'bears_threshold': float(bears_threshold)
        }
    
    def _train_ai_models(self, bulls_power: pd.Series, bears_power: pd.Series, data: pd.DataFrame) -> bool:
        """
        Train AI models for market regime classification and power prediction
        """
        if len(bulls_power) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            # Prepare features for training
            features, regime_targets, power_targets = self._prepare_training_data(bulls_power, bears_power, data)
            
            if len(features) > 50:  # Minimum samples for training
                # Train regime classifier
                X_train, X_test, y_regime_train, y_regime_test = train_test_split(
                    features, regime_targets, test_size=0.2, random_state=42
                )
                
                self.scaler.fit(X_train)
                X_train_scaled = self.scaler.transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                self.regime_classifier.fit(X_train_scaled, y_regime_train)
                
                # Train power predictor
                self.power_predictor.fit(X_train_scaled, power_targets[:-int(len(power_targets)*0.2)])
                
                self.ml_trained = True
                return True
                
        except Exception as e:
            pass
        
        return False    
    def _prepare_training_data(self, bulls_power: pd.Series, bears_power: pd.Series, 
                              data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for AI models
        """
        features = []
        regime_targets = []
        power_targets = []
        
        lookback = 30
        
        for i in range(lookback, len(bulls_power) - 5):
            # Feature vector
            bulls_window = bulls_power.iloc[i-lookback:i].values
            bears_window = bears_power.iloc[i-lookback:i].values
            price_window = data['close'].iloc[i-lookback:i].values
            volume_window = data['volume'].iloc[i-lookback:i].values
            
            feature_vector = [
                # Power statistics
                np.mean(bulls_window),
                np.std(bulls_window),
                np.max(bulls_window),
                np.min(bulls_window),
                np.mean(abs(bears_window)),
                np.std(abs(bears_window)),
                np.max(abs(bears_window)),
                np.min(abs(bears_window)),
                
                # Power dynamics
                bulls_window[-1] - bulls_window[0],
                bears_window[-1] - bears_window[0],
                np.corrcoef(bulls_window, bears_window)[0, 1] if len(set(bulls_window)) > 1 else 0,
                
                # Price-power relationships
                np.corrcoef(bulls_window, price_window)[0, 1] if len(set(bulls_window)) > 1 else 0,
                np.corrcoef(bears_window, price_window)[0, 1] if len(set(bears_window)) > 1 else 0,
                
                # Volume analysis
                np.mean(volume_window),
                np.std(volume_window),
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1,
                
                # Market dynamics
                (price_window[-1] - price_window[0]) / price_window[0] if price_window[0] != 0 else 0,
                np.std(price_window) / np.mean(price_window) if np.mean(price_window) != 0 else 0,
            ]
            
            features.append(feature_vector)
            
            # Regime target (simplified)
            future_price_change = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
            if future_price_change > 0.01:
                regime_target = 2  # Bull
            elif future_price_change < -0.01:
                regime_target = 0  # Bear
            else:
                regime_target = 1  # Neutral
            
            regime_targets.append(regime_target)
            
            # Power prediction target
            future_bulls_power = bulls_power.iloc[i+5] if i+5 < len(bulls_power) else bulls_power.iloc[-1]
            power_targets.append(future_bulls_power)
        
        return np.array(features), np.array(regime_targets), np.array(power_targets)
    
    def _generate_ai_enhanced_signals(self, bulls_power: pd.Series, bears_power: pd.Series, 
                                     data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate AI-enhanced trading signals
        """
        if not self.ml_trained or len(bulls_power) < 30:
            return None, 0.0
        
        try:
            # Prepare current features
            lookback = 30
            bulls_window = bulls_power.tail(lookback).values
            bears_window = bears_power.tail(lookback).values
            price_window = data['close'].tail(lookback).values
            volume_window = data['volume'].tail(lookback).values
            
            feature_vector = np.array([[
                np.mean(bulls_window),
                np.std(bulls_window),
                np.max(bulls_window),
                np.min(bulls_window),
                np.mean(abs(bears_window)),
                np.std(abs(bears_window)),
                np.max(abs(bears_window)),
                np.min(abs(bears_window)),
                bulls_window[-1] - bulls_window[0],
                bears_window[-1] - bears_window[0],
                np.corrcoef(bulls_window, bears_window)[0, 1] if len(set(bulls_window)) > 1 else 0,
                np.corrcoef(bulls_window, price_window)[0, 1] if len(set(bulls_window)) > 1 else 0,
                np.corrcoef(bears_window, price_window)[0, 1] if len(set(bears_window)) > 1 else 0,
                np.mean(volume_window),
                np.std(volume_window),
                volume_window[-1] / np.mean(volume_window) if np.mean(volume_window) != 0 else 1,
                (price_window[-1] - price_window[0]) / price_window[0] if price_window[0] != 0 else 0,
                np.std(price_window) / np.mean(price_window) if np.mean(price_window) != 0 else 0,
            ]])
            
            scaled_features = self.scaler.transform(feature_vector)
            
            # Get regime prediction
            regime_proba = self.regime_classifier.predict_proba(scaled_features)[0]
            
            # Get power prediction
            power_prediction = self.power_predictor.predict(scaled_features)[0]
            
            # Generate signal based on predictions
            if len(regime_proba) >= 3:
                bull_prob = regime_proba[2]  # Bull market probability
                bear_prob = regime_proba[0]  # Bear market probability
                
                if bull_prob > 0.7 and power_prediction > 0:
                    return SignalType.STRONG_BUY, bull_prob
                elif bull_prob > 0.6:
                    return SignalType.BUY, bull_prob * 0.8
                elif bear_prob > 0.7 and power_prediction < 0:
                    return SignalType.STRONG_SELL, bear_prob
                elif bear_prob > 0.6:
                    return SignalType.SELL, bear_prob * 0.8
                    
        except Exception as e:
            pass
        
        return None, 0.0
    
    def _calculate_power_momentum(self, bulls_power: pd.Series, bears_power: pd.Series) -> Dict[str, float]:
        """
        Calculate momentum indicators for power analysis
        """
        if len(bulls_power) < 5:
            return {'bulls_momentum': 0.0, 'bears_momentum': 0.0, 'power_divergence': 0.0}
        
        # Calculate momentum (rate of change)
        bulls_momentum = bulls_power.diff(periods=5).iloc[-1] if len(bulls_power) > 5 else 0.0
        bears_momentum = bears_power.diff(periods=5).iloc[-1] if len(bears_power) > 5 else 0.0
        
        # Power divergence (difference between bulls and bears momentum)
        power_divergence = bulls_momentum - abs(bears_momentum)
        
        return {
            'bulls_momentum': float(bulls_momentum),
            'bears_momentum': float(bears_momentum),
            'power_divergence': float(power_divergence)
        }
    
    def _generate_composite_signal(self, bulls_power: pd.Series, bears_power: pd.Series,
                                  thresholds: Dict[str, float], divergences: Dict[str, Any],
                                  institutional: Dict[str, Any], regime: str,
                                  ai_signal: Optional[SignalType], ai_confidence: float) -> Tuple[SignalType, float]:
        """
        Generate composite signal from all analysis components
        """
        signal_components = []
        confidence_components = []
        
        current_bulls = bulls_power.iloc[-1]
        current_bears = bears_power.iloc[-1]
        
        # Basic power signals
        if current_bulls > thresholds['bulls_threshold']:
            signal_components.append(1.0)
            confidence_components.append(0.6)
        elif current_bears < thresholds['bears_threshold']:
            signal_components.append(-1.0)
            confidence_components.append(0.6)
        else:
            signal_components.append(0.0)
            confidence_components.append(0.3)
        
        # Divergence signals
        if divergences['bullish_divergence']:
            signal_components.append(divergences['strength'])
            confidence_components.append(divergences['strength'])
        elif divergences['bearish_divergence']:
            signal_components.append(-divergences['strength'])
            confidence_components.append(divergences['strength'])
        
        # Institutional flow signals
        if institutional['detected']:
            if institutional['direction'] == 'bullish':
                signal_components.append(institutional['strength'])
                confidence_components.append(institutional['strength'] * 0.8)
            elif institutional['direction'] == 'bearish':
                signal_components.append(-institutional['strength'])
                confidence_components.append(institutional['strength'] * 0.8)
        
        # Market regime adjustment
        regime_multiplier = 1.0
        if regime in ['strong_bull_market', 'bull_market']:
            regime_multiplier = 1.2
        elif regime in ['strong_bear_market', 'bear_market']:
            regime_multiplier = 1.2
        elif regime == 'high_volatility':
            regime_multiplier = 0.8
        
        # AI signal
        if ai_signal == SignalType.STRONG_BUY:
            signal_components.append(ai_confidence)
            confidence_components.append(ai_confidence)
        elif ai_signal == SignalType.BUY:
            signal_components.append(ai_confidence * 0.8)
            confidence_components.append(ai_confidence * 0.8)
        elif ai_signal == SignalType.STRONG_SELL:
            signal_components.append(-ai_confidence)
            confidence_components.append(ai_confidence)
        elif ai_signal == SignalType.SELL:
            signal_components.append(-ai_confidence * 0.8)
            confidence_components.append(ai_confidence * 0.8)
        
        # Calculate weighted signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components) * regime_multiplier
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        # Determine final signal
        if weighted_signal > 0.4:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.4:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        # Calculate final confidence
        final_confidence = min(avg_confidence * abs(weighted_signal), 1.0)
        
        return signal, final_confidence
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Bears and Bulls Power with AI enhancement
        """
        try:
            # Calculate basic power indicators
            bulls_power = self._calculate_bulls_power(data)
            bears_power = self._calculate_bears_power(data)
            
            # Calculate adaptive thresholds
            thresholds = self._calculate_adaptive_thresholds(bulls_power, bears_power)
            
            # Detect institutional flows
            institutional_flows = self._detect_institutional_flows(bulls_power, bears_power, data)
            
            # Analyze market regime
            market_regime = self._analyze_market_regime(bulls_power, bears_power, data)
            
            # Detect divergences
            divergences = self._detect_power_divergences(bulls_power, bears_power, data['close'])
            
            # Train AI models if not trained
            if not self.ml_trained:
                self._train_ai_models(bulls_power, bears_power, data)
            
            # Generate AI-enhanced signals
            ai_signal, ai_confidence = self._generate_ai_enhanced_signals(bulls_power, bears_power, data)
            
            # Calculate power momentum
            momentum_analysis = self._calculate_power_momentum(bulls_power, bears_power)
            
            # Generate composite signal
            composite_signal, composite_confidence = self._generate_composite_signal(
                bulls_power, bears_power, thresholds, divergences,
                institutional_flows, market_regime, ai_signal, ai_confidence
            )
            
            # Market context analysis
            market_context = self._analyze_market_context(bulls_power, bears_power, data)
            
            result = {
                'bulls_power': float(bulls_power.iloc[-1]),
                'bears_power': float(bears_power.iloc[-1]),
                'bulls_normalized': float(bulls_power.iloc[-1] / bulls_power.std()) if bulls_power.std() != 0 else 0.0,
                'bears_normalized': float(bears_power.iloc[-1] / bears_power.std()) if bears_power.std() != 0 else 0.0,
                'signal': composite_signal,
                'confidence': composite_confidence,
                'thresholds': thresholds,
                'divergences': divergences,
                'institutional_flows': institutional_flows,
                'market_regime': market_regime,
                'ai_signal': ai_signal,
                'ai_confidence': ai_confidence,
                'momentum_analysis': momentum_analysis,
                'market_context': market_context,
                'power_balance': self._calculate_power_balance(bulls_power, bears_power),
                'trend_alignment': self._assess_trend_alignment(bulls_power, bears_power, data),
                'values_history': {
                    'bulls_power': bulls_power.tail(20).tolist(),
                    'bears_power': bears_power.tail(20).tolist(),
                    'regime_history': self.regime_history[-10:],
                    'institutional_history': self.institutional_flows[-10:]
                }
            }
            
            # Update history
            self.power_history.append({
                'timestamp': data.index[-1],
                'bulls_power': float(bulls_power.iloc[-1]),
                'bears_power': float(bears_power.iloc[-1])
            })
            
            self.regime_history.append({
                'timestamp': data.index[-1],
                'regime': market_regime
            })
            
            self.institutional_flows.append({
                'timestamp': data.index[-1],
                'flows': institutional_flows
            })
            
            # Keep history manageable
            if len(self.power_history) > 100:
                self.power_history = self.power_history[-100:]
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            if len(self.institutional_flows) > 100:
                self.institutional_flows = self.institutional_flows[-100:]
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Bears/Bulls Power: {str(e)}",
                cause=e
            )
    
    def _analyze_market_context(self, bulls_power: pd.Series, bears_power: pd.Series, 
                               data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze broader market context"""
        if not self.parameters['market_context'] or len(bulls_power) < 20:
            return {}
        
        # Calculate market stress indicators
        power_ratio = abs(bulls_power.iloc[-1]) / (abs(bears_power.iloc[-1]) + 0.001)
        volatility = data['close'].pct_change().rolling(window=20).std().iloc[-1]
        volume_surge = data['volume'].iloc[-1] / data['volume'].rolling(window=20).mean().iloc[-1]
        
        return {
            'power_ratio': float(power_ratio),
            'market_stress': 'high' if volatility > data['close'].pct_change().std() * 1.5 else 'normal',
            'volume_activity': 'surge' if volume_surge > 2.0 else 'normal',
            'market_efficiency': self._calculate_market_efficiency(bulls_power, bears_power, data)
        }
    
    def _calculate_market_efficiency(self, bulls_power: pd.Series, bears_power: pd.Series,
                                   data: pd.DataFrame) -> float:
        """Calculate market efficiency score"""
        try:
            # Measure how well power indicators predict price movements
            if len(bulls_power) < 10:
                return 0.5
            
            recent_bulls = bulls_power.tail(10)
            recent_bears = bears_power.tail(10)
            recent_returns = data['close'].pct_change().tail(10)
            
            # Calculate correlation between power and future returns
            bulls_correlation = abs(np.corrcoef(recent_bulls[:-1], recent_returns[1:])[0, 1])
            bears_correlation = abs(np.corrcoef(recent_bears[:-1], recent_returns[1:])[0, 1])
            
            efficiency = (bulls_correlation + bears_correlation) / 2
            return float(np.clip(efficiency, 0.0, 1.0))
        except:
            return 0.5
    
    def _calculate_power_balance(self, bulls_power: pd.Series, bears_power: pd.Series) -> str:
        """Calculate current power balance"""
        current_bulls = bulls_power.iloc[-1]
        current_bears = abs(bears_power.iloc[-1])
        
        if current_bulls > current_bears * 1.5:
            return "strong_bulls_dominance"
        elif current_bulls > current_bears * 1.2:
            return "bulls_dominance"
        elif current_bears > current_bulls * 1.5:
            return "strong_bears_dominance"
        elif current_bears > current_bulls * 1.2:
            return "bears_dominance"
        else:
            return "balanced"
    
    def _assess_trend_alignment(self, bulls_power: pd.Series, bears_power: pd.Series,
                               data: pd.DataFrame) -> Dict[str, Any]:
        """Assess alignment between power indicators and price trend"""
        if len(bulls_power) < 10:
            return {"alignment": "unknown", "strength": 0.0}
        
        # Calculate trend directions
        price_trend = data['close'].diff(periods=5).iloc[-1]
        bulls_trend = bulls_power.diff(periods=5).iloc[-1]
        bears_trend = bears_power.diff(periods=5).iloc[-1]
        
        # Assess alignment
        if price_trend > 0 and bulls_trend > 0:
            alignment = "bullish_aligned"
            strength = min(abs(bulls_trend) / (bulls_power.std() if bulls_power.std() != 0 else 1), 1.0)
        elif price_trend < 0 and bears_trend < 0:
            alignment = "bearish_aligned"
            strength = min(abs(bears_trend) / (bears_power.std() if bears_power.std() != 0 else 1), 1.0)
        else:
            alignment = "misaligned"
            strength = 0.3
        
        return {"alignment": alignment, "strength": float(strength)}
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate signal from calculated values"""
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get calculation metadata"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'bears_bulls_power',
            'ml_trained': self.ml_trained,
            'adaptive_thresholds': self.parameters['adaptive_thresholds'],
            'volume_analysis': self.parameters['volume_analysis'],
            'market_context': self.parameters['market_context'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata