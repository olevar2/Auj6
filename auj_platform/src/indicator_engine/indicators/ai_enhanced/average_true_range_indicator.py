"""
Average True Range - AI-Enhanced Volatility Analysis
==================================================

Advanced ATR implementation with volatility regime detection, adaptive periods,
and machine learning-based volatility prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class AverageTrueRange(StandardIndicatorInterface):
    """
    AI-Enhanced Average True Range with adaptive period optimization and volatility prediction.
    
    Features:
    - Adaptive ATR period based on market conditions
    - Volatility regime detection and classification
    - Machine learning volatility forecasting
    - True range breakout detection
    - Multi-timeframe volatility analysis
    - Volatility clustering identification
    - Risk-adjusted position sizing recommendations
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'base_period': 14,
            'adaptive_periods': True,
            'min_period': 7,
            'max_period': 30,
            'volatility_regimes': True,
            'ml_prediction': True,
            'breakout_detection': True,
            'clustering_analysis': True,
            'multi_timeframe': True,
            'timeframes': [7, 14, 21, 30],
            'prediction_horizon': 5,
            'regime_lookback': 50,
            'breakout_threshold': 2.0,  # ATR multiples for breakout
            'risk_adjustment': True,
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("AverageTrueRange", default_params)
        
        # ML components
        self.volatility_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.volatility_history = []
        self.regime_history = []
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(self.parameters['timeframes']) + self.parameters['regime_lookback'],
            lookback_periods=200
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate AI-enhanced Average True Range analysis.
        """
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate true range
            true_range = self._calculate_true_range(high, low, close)
            
            # Determine optimal ATR period
            optimal_period = self._calculate_optimal_period(true_range, close) if self.parameters['adaptive_periods'] else self.parameters['base_period']
            
            # Calculate ATR with optimal period
            atr = self._calculate_atr(true_range, optimal_period)
            
            # Multi-timeframe ATR analysis
            mtf_analysis = self._multi_timeframe_analysis(high, low, close) if self.parameters['multi_timeframe'] else {}
            
            # Volatility regime detection
            regime_analysis = self._detect_volatility_regime(atr, close) if self.parameters['volatility_regimes'] else {}
            
            # Machine learning volatility prediction
            ml_prediction = self._predict_volatility(true_range, close, volume) if self.parameters['ml_prediction'] else {}
            
            # Breakout detection
            breakout_analysis = self._detect_breakouts(atr, high, low, close) if self.parameters['breakout_detection'] else {}
            
            # Volatility clustering analysis
            clustering_analysis = self._analyze_volatility_clustering(true_range) if self.parameters['clustering_analysis'] else {}
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(atr, close, true_range)
            
            # Normalized ATR for comparison
            normalized_atr = self._normalize_atr(atr, close)
            
            # Signal generation
            signal_strength = self._calculate_signal_strength(
                atr, regime_analysis, breakout_analysis, ml_prediction, normalized_atr
            )
            
            return {
                'atr': atr[-1] if len(atr) > 0 else 0,
                'true_range': true_range[-1] if len(true_range) > 0 else 0,
                'normalized_atr': normalized_atr[-1] if len(normalized_atr) > 0 else 0,
                'optimal_period': optimal_period,
                'mtf_analysis': mtf_analysis,
                'regime_analysis': regime_analysis,
                'ml_prediction': ml_prediction,
                'breakout_analysis': breakout_analysis,
                'clustering_analysis': clustering_analysis,
                'risk_metrics': risk_metrics,
                'signal_strength': signal_strength,
                'volatility_percentile': self._calculate_volatility_percentile(atr),
                'trend_adjusted_atr': self._calculate_trend_adjusted_atr(atr, close)
            }
            
        except Exception as e:
            raise Exception(f"AverageTrueRange calculation failed: {str(e)}")
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range values."""
        true_range = np.zeros(len(high))
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]  # High - Low
            tr2 = abs(high[i] - close[i-1])  # High - Previous Close
            tr3 = abs(low[i] - close[i-1])   # Low - Previous Close
            
            true_range[i] = max(tr1, tr2, tr3)
        
        # First value is just high - low
        true_range[0] = high[0] - low[0]
        
        return true_range
    
    def _calculate_atr(self, true_range: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range using Wilder's smoothing."""
        if len(true_range) < period:
            return np.array([])
        
        atr = np.zeros(len(true_range))
        
        # Initial ATR is simple average
        atr[period-1] = np.mean(true_range[:period])
        
        # Subsequent ATR values use Wilder's smoothing
        for i in range(period, len(true_range)):
            atr[i] = (atr[i-1] * (period - 1) + true_range[i]) / period
        
        return atr
    
    def _calculate_optimal_period(self, true_range: np.ndarray, close: np.ndarray) -> int:
        """Calculate optimal ATR period based on market conditions."""
        if len(true_range) < 50:
            return self.parameters['base_period']
        
        # Test different periods and select based on predictive power
        periods = range(self.parameters['min_period'], self.parameters['max_period'] + 1)
        best_period = self.parameters['base_period']
        best_score = 0
        
        for period in periods:
            if len(true_range) >= period + 20:
                atr = self._calculate_atr(true_range, period)
                
                # Calculate predictive score based on correlation with future volatility
                if len(atr) >= period + 10:
                    current_atr = atr[period:period+10]
                    future_volatility = np.std(np.diff(close[period+5:period+15]) / close[period+4:period+14]) if len(close) >= period + 15 else 0
                    
                    if len(current_atr) > 0 and future_volatility > 0:
                        # Score based on correlation with future volatility
                        score = abs(np.corrcoef(current_atr, [future_volatility] * len(current_atr))[0, 1])
                        
                        if score > best_score:
                            best_score = score
                            best_period = period
        
        return best_period
    
    def _multi_timeframe_analysis(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Perform multi-timeframe ATR analysis."""
        mtf_results = {}
        
        for tf in self.parameters['timeframes']:
            if len(high) >= tf + 20:
                tr = self._calculate_true_range(high, low, close)
                atr = self._calculate_atr(tr, tf)
                
                if len(atr) > 0:
                    current_atr = atr[-1]
                    normalized_atr = (current_atr / close[-1]) * 100 if close[-1] > 0 else 0
                    
                    # ATR trend
                    atr_trend = 0
                    if len(atr) >= 5:
                        atr_trend = np.polyfit(range(5), atr[-5:], 1)[0]
                    
                    mtf_results[f'tf_{tf}'] = {
                        'atr': current_atr,
                        'normalized_atr': normalized_atr,
                        'atr_trend': atr_trend,
                        'volatility_rank': self._calculate_volatility_rank(atr, 20)
                    }
        
        # Calculate consensus
        if mtf_results:
            all_normalized = [result['normalized_atr'] for result in mtf_results.values()]
            all_trends = [result['atr_trend'] for result in mtf_results.values()]
            
            mtf_results['consensus'] = {
                'avg_normalized_atr': np.mean(all_normalized),
                'avg_trend': np.mean(all_trends),
                'volatility_consistency': 1 - (np.std(all_normalized) / (np.mean(all_normalized) + 1e-8)),
                'trend_agreement': len([t for t in all_trends if t * all_trends[0] > 0]) / len(all_trends) if all_trends else 0
            }
        
        return mtf_results
    
    def _detect_volatility_regime(self, atr: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Detect current volatility regime."""
        if len(atr) < self.parameters['regime_lookback']:
            return {}
        
        # Calculate volatility metrics for regime detection
        recent_atr = atr[-20:] if len(atr) >= 20 else atr
        historical_atr = atr[-self.parameters['regime_lookback']:]
        
        # Current volatility percentile
        current_percentile = stats.percentileofscore(historical_atr, atr[-1]) / 100
        
        # Volatility clustering detection
        volatility_changes = np.diff(recent_atr)
        clustering_factor = np.corrcoef(volatility_changes[:-1], volatility_changes[1:])[0, 1] if len(volatility_changes) > 1 else 0
        
        # Regime classification
        if current_percentile > 0.8:
            regime = 'high_volatility'
        elif current_percentile > 0.6:
            regime = 'elevated_volatility'
        elif current_percentile < 0.2:
            regime = 'low_volatility'
        elif current_percentile < 0.4:
            regime = 'below_average_volatility'
        else:
            regime = 'normal_volatility'
        
        # Regime persistence
        regime_stability = self._calculate_regime_stability(historical_atr)
        
        return {
            'current_regime': regime,
            'volatility_percentile': current_percentile,
            'clustering_factor': clustering_factor,
            'regime_stability': regime_stability,
            'regime_change_probability': self._calculate_regime_change_probability(historical_atr)
        }
    
    def _calculate_regime_stability(self, atr_history: np.ndarray) -> float:
        """Calculate how stable the current volatility regime is."""
        if len(atr_history) < 10:
            return 0.5
        
        # Calculate rolling volatility percentiles
        window = 10
        percentiles = []
        
        for i in range(window, len(atr_history)):
            historical = atr_history[:i]
            current = atr_history[i]
            percentile = stats.percentileofscore(historical, current) / 100
            percentiles.append(percentile)
        
        if len(percentiles) < 5:
            return 0.5
        
        # Stability as inverse of percentile variance
        percentile_variance = np.var(percentiles[-10:])
        stability = 1.0 / (1.0 + percentile_variance * 10)
        
        return float(np.clip(stability, 0, 1))
    
    def _calculate_regime_change_probability(self, atr_history: np.ndarray) -> float:
        """Calculate probability of volatility regime change."""
        if len(atr_history) < 20:
            return 0.5
        
        # Look for patterns that precede regime changes
        recent_trend = np.polyfit(range(10), atr_history[-10:], 1)[0]
        recent_acceleration = np.polyfit(range(5), atr_history[-5:], 2)[0]
        
        # Historical volatility of volatility
        atr_volatility = np.std(np.diff(atr_history[-20:]))
        
        # Combine indicators
        trend_factor = abs(recent_trend) / (np.mean(atr_history[-10:]) + 1e-8)
        acceleration_factor = abs(recent_acceleration) / (np.mean(atr_history[-5:]) + 1e-8)
        volatility_factor = atr_volatility / (np.mean(atr_history[-20:]) + 1e-8)
        
        change_probability = (trend_factor + acceleration_factor + volatility_factor) / 3
        
        return float(np.clip(change_probability, 0, 1))
    
    def _predict_volatility(self, true_range: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Use machine learning to predict future volatility."""
        if len(true_range) < 50:
            return {}
        
        try:
            # Prepare features and targets for ML
            features, targets = self._prepare_ml_data(true_range, close, volume)
            
            if len(features) < 20:
                return {}
            
            # Train model if not trained
            if not self.is_trained and len(features) >= 30:
                train_size = int(len(features) * 0.8)
                X_train = features[:train_size]
                y_train = targets[:train_size]
                
                X_scaled = self.scaler.fit_transform(X_train)
                self.volatility_predictor.fit(X_scaled, y_train)
                self.is_trained = True
            
            if self.is_trained:
                # Make prediction
                latest_features = features[-1:].reshape(1, -1)
                X_pred_scaled = self.scaler.transform(latest_features)
                prediction = self.volatility_predictor.predict(X_pred_scaled)[0]
                
                # Calculate prediction confidence
                confidence = self._calculate_prediction_confidence(features, targets)
                
                # Feature importance
                feature_importance = self.volatility_predictor.feature_importances_ if hasattr(self.volatility_predictor, 'feature_importances_') else []
                
                return {
                    'predicted_volatility': float(prediction),
                    'confidence': confidence,
                    'prediction_horizon': self.parameters['prediction_horizon'],
                    'feature_importance': feature_importance.tolist() if len(feature_importance) > 0 else [],
                    'model_trained': True
                }
        
        except Exception:
            pass
        
        return {'model_trained': False}
    
    def _prepare_ml_data(self, true_range: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML model."""
        features = []
        targets = []
        
        lookback = 20
        horizon = self.parameters['prediction_horizon']
        
        for i in range(lookback, len(true_range) - horizon):
            # Features: historical volatility metrics
            window_tr = true_range[i-lookback:i]
            window_close = close[i-lookback:i]
            window_volume = volume[i-lookback:i] if len(volume) > i else np.ones(lookback)
            
            feature_vector = [
                np.mean(window_tr),                    # Average TR
                np.std(window_tr),                     # TR volatility
                np.max(window_tr),                     # Max TR
                np.min(window_tr),                     # Min TR
                np.mean(window_tr[-5:]) / np.mean(window_tr[-10:-5]) - 1,  # Recent TR ratio
                np.std(np.diff(window_close) / window_close[:-1]),  # Price volatility
                np.mean(window_volume[-5:]) / np.mean(window_volume[-10:-5]) - 1,  # Volume change
                window_tr[-1] / np.mean(window_tr) - 1,  # Current TR vs average
            ]
            
            # Target: future volatility (standard deviation of future returns)
            future_returns = np.diff(close[i:i+horizon]) / close[i:i+horizon-1]
            target = np.std(future_returns) if len(future_returns) > 0 else 0
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def _calculate_prediction_confidence(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Calculate confidence in volatility prediction."""
        if len(features) < 10:
            return 0.5
        
        try:
            # Use recent data for validation
            test_size = min(10, len(features) // 4)
            X_test = features[-test_size:]
            y_test = targets[-test_size:]
            
            X_test_scaled = self.scaler.transform(X_test)
            predictions = self.volatility_predictor.predict(X_test_scaled)
            
            # Calculate R² score
            r2 = 1 - np.sum((y_test - predictions) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-8)
            confidence = max(0, r2)
            
            return float(np.clip(confidence, 0, 1))
        
        except Exception:
            return 0.5
    
    def _detect_breakouts(self, atr: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Detect volatility breakouts using ATR."""
        if len(atr) < 20:
            return {}
        
        # Current true range vs ATR
        current_tr = max(
            high[-1] - low[-1],
            abs(high[-1] - close[-2]) if len(close) > 1 else high[-1] - low[-1],
            abs(low[-1] - close[-2]) if len(close) > 1 else high[-1] - low[-1]
        )
        
        current_atr = atr[-1]
        tr_ratio = current_tr / (current_atr + 1e-8)
        
        # Breakout detection
        breakout_threshold = self.parameters['breakout_threshold']
        is_breakout = tr_ratio > breakout_threshold
        
        # Recent volatility expansion
        recent_atr_avg = np.mean(atr[-5:])
        historical_atr_avg = np.mean(atr[-20:-5])
        expansion_ratio = recent_atr_avg / (historical_atr_avg + 1e-8)
        
        # Breakout direction (if applicable)
        breakout_direction = 'neutral'
        if len(close) >= 2:
            price_change = (close[-1] - close[-2]) / close[-2]
            if is_breakout and abs(price_change) > 0.001:  # 0.1% threshold
                breakout_direction = 'bullish' if price_change > 0 else 'bearish'
        
        return {
            'is_breakout': is_breakout,
            'tr_ratio': tr_ratio,
            'expansion_ratio': expansion_ratio,
            'breakout_direction': breakout_direction,
            'breakout_strength': min(tr_ratio / breakout_threshold, 3.0) if breakout_threshold > 0 else 1.0
        }
    
    def _analyze_volatility_clustering(self, true_range: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility clustering patterns."""
        if len(true_range) < 30:
            return {}
        
        # Calculate squared returns as proxy for volatility
        volatility_proxy = true_range ** 2
        
        # Autocorrelation of volatility
        autocorr_lags = [1, 2, 3, 5, 10]
        autocorrelations = []
        
        for lag in autocorr_lags:
            if len(volatility_proxy) > lag:
                corr = np.corrcoef(volatility_proxy[:-lag], volatility_proxy[lag:])[0, 1]
                autocorrelations.append(corr if not np.isnan(corr) else 0)
            else:
                autocorrelations.append(0)
        
        # Clustering strength (average of significant autocorrelations)
        clustering_strength = np.mean([max(0, corr) for corr in autocorrelations])
        
        # GARCH-like effect detection
        recent_volatility = np.var(true_range[-10:])
        historical_volatility = np.var(true_range[-30:-10])
        garch_effect = recent_volatility / (historical_volatility + 1e-8)
        
        return {
            'clustering_strength': float(clustering_strength),
            'autocorrelations': autocorrelations,
            'garch_effect': float(garch_effect),
            'current_cluster_phase': 'high' if garch_effect > 1.2 else 'low' if garch_effect < 0.8 else 'normal'
        }
    
    def _calculate_risk_metrics(self, atr: np.ndarray, close: np.ndarray, true_range: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics based on ATR."""
        if len(atr) == 0 or len(close) == 0:
            return {}
        
        current_atr = atr[-1]
        current_price = close[-1]
        
        # Position sizing based on ATR
        risk_per_share = current_atr
        position_size_multiplier = 1.0 / (risk_per_share / current_price + 1e-8)
        
        # Stop loss levels
        stop_loss_long = current_price - (current_atr * 2)
        stop_loss_short = current_price + (current_atr * 2)
        
        # Risk-reward ratios
        risk_percentage = (current_atr / current_price) * 100
        
        # Volatility-adjusted returns
        if len(close) >= 20:
            returns = np.diff(close[-20:]) / close[-20:-1]
            volatility_adjusted_return = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            volatility_adjusted_return = 0
        
        return {
            'risk_per_share': float(risk_per_share),
            'position_size_multiplier': float(position_size_multiplier),
            'stop_loss_long': float(stop_loss_long),
            'stop_loss_short': float(stop_loss_short),
            'risk_percentage': float(risk_percentage),
            'volatility_adjusted_return': float(volatility_adjusted_return)
        }
    
    def _normalize_atr(self, atr: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Normalize ATR by price for comparison across different price levels."""
        if len(atr) == 0 or len(close) == 0:
            return np.array([])
        
        # Normalize ATR as percentage of price
        normalized = (atr / close) * 100
        
        return normalized
    
    def _calculate_volatility_percentile(self, atr: np.ndarray) -> float:
        """Calculate current ATR percentile vs historical values."""
        if len(atr) < 20:
            return 50.0
        
        current_atr = atr[-1]
        historical_atr = atr[-100:] if len(atr) >= 100 else atr[:-1]
        
        percentile = stats.percentileofscore(historical_atr, current_atr)
        
        return float(percentile)
    
    def _calculate_volatility_rank(self, atr: np.ndarray, lookback: int) -> float:
        """Calculate volatility rank over lookback period."""
        if len(atr) < lookback:
            return 0.5
        
        recent_atr = atr[-lookback:]
        current_atr = atr[-1]
        
        rank = (current_atr - np.min(recent_atr)) / (np.max(recent_atr) - np.min(recent_atr) + 1e-8)
        
        return float(rank)
    
    def _calculate_trend_adjusted_atr(self, atr: np.ndarray, close: np.ndarray) -> float:
        """Calculate trend-adjusted ATR."""
        if len(atr) == 0 or len(close) < 10:
            return 0.0
        
        # Calculate price trend
        trend = np.polyfit(range(len(close[-10:])), close[-10:], 1)[0]
        trend_strength = abs(trend) / (np.mean(close[-10:]) + 1e-8)
        
        # Adjust ATR based on trend strength
        current_atr = atr[-1]
        trend_adjustment = 1.0 + (trend_strength * 0.5)  # Increase ATR during strong trends
        
        return float(current_atr * trend_adjustment)
    
    def _calculate_signal_strength(self, atr: np.ndarray, regime_analysis: Dict,
                                 breakout_analysis: Dict, ml_prediction: Dict,
                                 normalized_atr: np.ndarray) -> float:
        """Calculate trading signal strength based on ATR analysis."""
        signal_components = []
        
        # Volatility regime signal
        if regime_analysis:
            regime = regime_analysis.get('current_regime', 'normal_volatility')
            if regime == 'low_volatility':
                signal_components.append(0.3)  # Anticipate breakout
            elif regime == 'high_volatility':
                signal_components.append(-0.2)  # Anticipate mean reversion
        
        # Breakout signal
        if breakout_analysis:
            if breakout_analysis.get('is_breakout', False):
                direction = breakout_analysis.get('breakout_direction', 'neutral')
                strength = breakout_analysis.get('breakout_strength', 1.0)
                
                if direction == 'bullish':
                    signal_components.append(0.4 * min(strength / 2, 1))
                elif direction == 'bearish':
                    signal_components.append(-0.4 * min(strength / 2, 1))
        
        # ML prediction signal
        if ml_prediction and ml_prediction.get('model_trained', False):
            predicted_vol = ml_prediction.get('predicted_volatility', 0)
            confidence = ml_prediction.get('confidence', 0.5)
            
            if len(atr) > 0:
                current_vol = atr[-1] / atr[-10:].mean() if len(atr) >= 10 else 1
                vol_signal = np.tanh((predicted_vol - current_vol) * 2) * confidence * 0.3
                signal_components.append(vol_signal)
        
        # Volatility trend signal
        if len(normalized_atr) >= 5:
            vol_trend = np.polyfit(range(5), normalized_atr[-5:], 1)[0]
            trend_signal = np.tanh(vol_trend * 10) * 0.2
            signal_components.append(trend_signal)
        
        # Combine signals
        if signal_components:
            total_signal = np.sum(signal_components)
            return float(np.clip(total_signal, -1, 1))
        
        return 0.0
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on ATR analysis."""
        signal_strength = value['signal_strength']
        volatility_percentile = value['volatility_percentile']
        
        # Adjust confidence based on volatility regime
        if volatility_percentile > 80:  # High volatility
            confidence_factor = 0.7
        elif volatility_percentile < 20:  # Low volatility
            confidence_factor = 0.9
        else:
            confidence_factor = 0.8
        
        confidence = min(abs(signal_strength) * confidence_factor, 1.0)
        
        # Generate signals
        if signal_strength > 0.5:
            return SignalType.STRONG_BUY, confidence
        elif signal_strength > 0.2:
            return SignalType.BUY, confidence
        elif signal_strength < -0.5:
            return SignalType.STRONG_SELL, confidence
        elif signal_strength < -0.2:
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, confidence
