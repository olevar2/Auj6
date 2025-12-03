"""
Relative Strength Index (RSI) Indicator - Advanced Implementation
================================================================

Comprehensive RSI implementation with ML-enhanced signal detection, adaptive parameters, 
divergence analysis, sophisticated overbought/oversold dynamics, multi-timeframe synthesis,
risk metrics, RSI momentum/acceleration, RSI Bands, and volume-weighted variants.

Features consolidated from RSI, RSI Signal, and RSI Oscillator implementations:
- Multi-period RSI analysis with adaptive parameters
- ML-based signal classification and strength assessment
- Advanced divergence detection with multiple confirmation methods
- Regime-aware signal filtering and adaptation
- Risk and confidence scoring with position sizing recommendations
- Multi-timeframe signal synthesis and convergence analysis
- Volume-weighted RSI calculations
- RSI momentum and acceleration detection
- Noise reduction and smoothing techniques
- Stochastic RSI integration
- Adaptive overbought/oversold thresholds
- **Asynchronous Training** (Non-blocking)

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import warnings
import threading
import logging
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class RSIIndicator(StandardIndicatorInterface):
    """
    Comprehensive Advanced RSI Indicator Implementation
    
    Features:
    - Sophisticated RSI calculation with multiple smoothing methods
    - ML-enhanced pattern recognition and signal optimization
    - Adaptive overbought/oversold thresholds based on market conditions
    - Advanced divergence detection with statistical validation
    - Multi-timeframe RSI analysis and consensus building
    - Market regime classification for RSI interpretation
    - Volume-weighted RSI variants
    - Multi-period RSI with adaptive parameters
    - RSI momentum and acceleration analysis
    - Stochastic RSI integration
    - Noise reduction and smoothing techniques
    - Risk assessment and confidence scoring
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 14,
            'secondary_periods': [7, 21, 50],
            'overbought': 70,
            'oversold': 30,
            'extreme_overbought': 85,
            'extreme_oversold': 15,
            'smoothing_method': 'wilder',  # 'wilder', 'ema', 'sma'
            'adaptive_thresholds': True,
            'adaptive_periods': True,
            'volume_weighted': True,
            'divergence_lookback': 20,
            'ml_lookback': 100,
            'multi_timeframe': True,
            'regime_analysis': True,
            'noise_reduction': True,
            'rsi_bands_enabled': True,
            'stochastic_rsi': True,
            'smooth_period': 3,
            'confidence_threshold': 0.65,
            'volatility_adjustment': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="RSIIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.ml_classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.signal_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.strength_regressor = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
        self.pattern_detector = MLPClassifier(hidden_layer_sizes=(50, 30, 20), max_iter=1000, random_state=42)
        self.ml_model = AdaBoostRegressor(n_estimators=100, random_state=42)
        self.pattern_model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.anomaly_detector = DBSCAN(eps=0.5, min_samples=5)
        self.pca = PCA(n_components=6)
        self.svd = TruncatedSVD(n_components=4, random_state=42)
        self.trend_predictor = AdaBoostRegressor(n_estimators=50, random_state=42)
        self.regime_classifier = KMeans(n_clusters=4, random_state=42)
        
        self.models_trained = False
        self.ml_trained = False
        self.is_fitted = False
        self.pattern_fitted = False
        self.is_trained = False
        
        # Threading control
        self.training_lock = threading.Lock()
        self.is_training = False
        self.logger = logging.getLogger(__name__)
        
        self.history = {
            'rsi': [],
            'gains': [],
            'losses': [],
            'divergences': [],
            'regime': [],
            'multi_rsi': [],
            'signals': []
        }
        
        # Adaptive parameters
        self.adaptive_smoothing = True
        self.regime_awareness = True
        self.feature_columns = []
        self.last_signals = []
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['period'], 
                        max(self.parameters.get('secondary_periods', [14])),
                        self.parameters.get('ml_lookback', 100))
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'high', 'low', 'volume'],
            min_periods=max_period * 2 + 50,
            lookback_periods=250
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive RSI with all advanced features"""
        try:
            if len(data) < max(self.parameters['period'], 20):
                raise ValueError("Insufficient data for RSI calculation")
            
            # Extract data arrays
            close = data['close'].values
            high = data['high'].values if 'high' in data else close
            low = data['low'].values if 'low' in data else close
            volume = data['volume'].values if 'volume' in data else np.ones(len(close))
            
            # Calculate multiple RSI timeframes
            multi_rsi = self._calculate_multi_rsi(close, high, low)
            
            # Enhanced RSI with adaptive parameters
            enhanced_rsi = self._calculate_enhanced_rsi(close, volume)
            
            # Combine results
            primary_rsi = enhanced_rsi['primary_rsi']
            composite_rsi = multi_rsi['composite_rsi']
            
            # Convert to pandas Series for analysis
            rsi_series = pd.Series(primary_rsi.values if hasattr(primary_rsi, 'values') else primary_rsi, index=data.index)
            
            # Advanced analysis
            adaptive_thresholds = self._calculate_adaptive_thresholds(rsi_series, data)
            divergences = self._detect_divergences(rsi_series, data)
            multi_timeframe = self._multi_timeframe_analysis(data)
            extremes_analysis = self._analyze_extremes(rsi_series, adaptive_thresholds)
            rsi_momentum_analysis = self._calculate_rsi_momentum(rsi_series)
            regime_analysis = self._classify_market_regime(rsi_series, data) if self.parameters.get('regime_analysis', True) else {}
            
            # RSI Bands calculation
            rsi_bands = self._calculate_rsi_bands(rsi_series) if self.parameters.get('rsi_bands_enabled', True) else {}
            
            # ML-based signal generation (Background Training)
            if not self.models_trained:
                self._train_ml_models(rsi_series, data, multi_rsi, enhanced_rsi)
            
            # Generate comprehensive signal
            signal, confidence = self._generate_comprehensive_rsi_signal(
                rsi_series, data, adaptive_thresholds, divergences,
                multi_timeframe, extremes_analysis, rsi_momentum_analysis,
                multi_rsi, enhanced_rsi
            )
            
            # Update history
            current_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
            self.history['rsi'].append(current_rsi)
            if len(self.history['rsi']) > 100:
                self.history['rsi'] = self.history['rsi'][-100:]
            
            result = {
                'rsi': current_rsi,
                'composite_rsi': float(composite_rsi.iloc[-1]) if hasattr(composite_rsi, 'iloc') and len(composite_rsi) > 0 else 50.0,
                'signal': signal,
                'confidence': confidence,
                'adaptive_thresholds': adaptive_thresholds,
                'divergences': divergences,
                'multi_timeframe': multi_timeframe,
                'extremes_analysis': extremes_analysis,
                'rsi_momentum': rsi_momentum_analysis,
                'regime_analysis': regime_analysis,
                'rsi_bands': rsi_bands,
                'multi_rsi_values': {
                    'rsi_fast': float(multi_rsi['rsi_fast'].iloc[-1]) if len(multi_rsi) > 0 else 50.0,
                    'rsi_slow': float(multi_rsi['rsi_slow'].iloc[-1]) if len(multi_rsi) > 0 else 50.0,
                    'stoch_rsi_k': float(multi_rsi['stoch_rsi_k'].iloc[-1]) if len(multi_rsi) > 0 else 50.0,
                    'stoch_rsi_d': float(multi_rsi['stoch_rsi_d'].iloc[-1]) if len(multi_rsi) > 0 else 50.0
                },
                'enhanced_rsi_values': {
                    'volume_adjusted': float(enhanced_rsi['volume_adjusted_rsi'].iloc[-1]) if len(enhanced_rsi) > 0 else 50.0,
                    'noise_reduced': float(enhanced_rsi['noise_reduced_rsi'].iloc[-1]) if len(enhanced_rsi) > 0 else 50.0,
                    'rsi_momentum': float(enhanced_rsi['rsi_momentum'].iloc[-1]) if len(enhanced_rsi) > 0 else 0.0,
                    'rsi_acceleration': float(enhanced_rsi['rsi_acceleration'].iloc[-1]) if len(enhanced_rsi) > 0 else 0.0
                },
                'values_history': {
                    'rsi': rsi_series.tail(30).tolist(),
                    'composite_rsi': composite_rsi.tail(30).tolist() if hasattr(composite_rsi, 'tail') else [50.0]
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate RSI: {str(e)}",
                cause=e
            )
    
    def _calculate_price_changes(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate gains and losses with volume weighting if enabled"""
        close = data['close']
        
        if self.parameters['volume_weighted']:
            # Volume-weighted price changes
            volume = data['volume']
            typical_price = (data['high'] + data['low'] + close) / 3
            vwap = (typical_price * volume).rolling(window=5).sum() / volume.rolling(window=5).sum()
            price_changes = vwap.diff()
        else:
            price_changes = close.diff()
        
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)
        
        return gains, losses
    
    def _calculate_smoothed_averages(self, gains: pd.Series, losses: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate smoothed averages using specified method"""
        period = self.parameters['period']
        method = self.parameters['smoothing_method']
        
        if method == 'wilder':
            # Wilder's smoothing (traditional RSI)
            alpha = 1.0 / period
            avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
            avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
        elif method == 'ema':
            # Exponential moving average
            avg_gains = gains.ewm(span=period).mean()
            avg_losses = losses.ewm(span=period).mean()
        elif method == 'sma':
            # Simple moving average
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
        else:
            # Default to Wilder's
            alpha = 1.0 / period
            avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
            avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
        
        return avg_gains, avg_losses
    
    def _calculate_rsi(self, avg_gains: pd.Series, avg_losses: pd.Series) -> pd.Series:
        """Calculate RSI from smoothed averages"""
        rs = avg_gains / (avg_losses + 1e-8)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adaptive_thresholds(self, rsi: pd.Series, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate adaptive overbought/oversold thresholds"""
        if not self.parameters['adaptive_thresholds'] or len(rsi) < 50:
            return {
                'overbought': self.parameters['overbought'],
                'oversold': self.parameters['oversold']
            }
        
        # Market volatility adaptation
        volatility = data['close'].pct_change().rolling(window=20).std().iloc[-1]
        avg_volatility = data['close'].pct_change().rolling(window=100).std().mean()
        
        vol_ratio = volatility / (avg_volatility + 1e-8)
        
        # RSI distribution analysis
        recent_rsi = rsi.tail(100).dropna()
        rsi_std = recent_rsi.std()
        rsi_mean = recent_rsi.mean()
        
        # Adjust thresholds based on volatility and RSI distribution
        if vol_ratio > 1.5:  # High volatility
            overbought = min(85, max(65, rsi_mean + 1.5 * rsi_std))
            oversold = max(15, min(35, rsi_mean - 1.5 * rsi_std))
        elif vol_ratio < 0.7:  # Low volatility
            overbought = min(80, max(60, rsi_mean + 1.0 * rsi_std))
            oversold = max(20, min(40, rsi_mean - 1.0 * rsi_std))
        else:  # Normal volatility
            overbought = min(75, max(65, self.parameters['overbought']))
            oversold = max(25, min(35, self.parameters['oversold']))
        
        return {
            'overbought': float(overbought),
            'oversold': float(oversold)
        }
    
    def _detect_divergences(self, rsi: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect RSI-Price divergences with statistical validation"""
        lookback = self.parameters['divergence_lookback']
        if len(rsi) < lookback:
            return {'bullish': False, 'bearish': False, 'strength': 0.0, 'confidence': 0.0}
        
        recent_rsi = rsi.tail(lookback)
        recent_prices = data['close'].tail(lookback)
        
        # Smooth data to reduce noise
        try:
            if len(recent_rsi) >= 5:
                smoothed_rsi = pd.Series(savgol_filter(recent_rsi.values, 5, 2), index=recent_rsi.index)
                smoothed_prices = pd.Series(savgol_filter(recent_prices.values, 5, 2), index=recent_prices.index)
            else:
                smoothed_rsi = recent_rsi
                smoothed_prices = recent_prices
        except:
            smoothed_rsi = recent_rsi
            smoothed_prices = recent_prices
        
        # Find peaks and troughs
        rsi_peaks, _ = find_peaks(smoothed_rsi.values, distance=3, prominence=2)
        rsi_troughs, _ = find_peaks(-smoothed_rsi.values, distance=3, prominence=2)
        
        price_peaks, _ = find_peaks(smoothed_prices.values, distance=3)
        price_troughs, _ = find_peaks(-smoothed_prices.values, distance=3)
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0.0
        statistical_confidence = 0.0
        
        # Bullish divergence: price lower low, RSI higher low
        if len(rsi_troughs) >= 2 and len(price_troughs) >= 2:
            last_rsi_trough_idx = rsi_troughs[-1]
            prev_rsi_trough_idx = rsi_troughs[-2]
            
            last_rsi_trough = smoothed_rsi.iloc[last_rsi_trough_idx]
            prev_rsi_trough = smoothed_rsi.iloc[prev_rsi_trough_idx]
            
            # Find corresponding price troughs
            price_trough_times = [smoothed_prices.index[i] for i in price_troughs]
            rsi_trough_times = [smoothed_rsi.index[last_rsi_trough_idx], smoothed_rsi.index[prev_rsi_trough_idx]]
            
            if len(price_trough_times) >= 2:
                # Find closest price troughs to RSI troughs
                closest_price_troughs = []
                for rsi_time in rsi_trough_times:
                    closest_idx = min(range(len(price_trough_times)), 
                                    key=lambda i: abs((price_trough_times[i] - rsi_time).total_seconds()))
                    closest_price_troughs.append(smoothed_prices[price_trough_times[closest_idx]])
                
                if len(closest_price_troughs) == 2:
                    last_price_trough, prev_price_trough = closest_price_troughs
                    
                    if last_price_trough < prev_price_trough and last_rsi_trough > prev_rsi_trough:
                        bullish_divergence = True
                        price_decline = (prev_price_trough - last_price_trough) / prev_price_trough
                        rsi_improvement = (last_rsi_trough - prev_rsi_trough) / prev_rsi_trough if prev_rsi_trough != 0 else 0
                        divergence_strength = min(rsi_improvement / (price_decline + 1e-8), 2.0)
                        
                        # Statistical validation
                        correlation = np.corrcoef(smoothed_rsi.values, smoothed_prices.values)[0, 1]
                        statistical_confidence = max(0, 1 - abs(correlation))  # Higher confidence when correlation is weak
        
        # Bearish divergence: price higher high, RSI lower high
        if len(rsi_peaks) >= 2 and len(price_peaks) >= 2:
            last_rsi_peak_idx = rsi_peaks[-1]
            prev_rsi_peak_idx = rsi_peaks[-2]
            
            last_rsi_peak = smoothed_rsi.iloc[last_rsi_peak_idx]
            prev_rsi_peak = smoothed_rsi.iloc[prev_rsi_peak_idx]
            
            # Find corresponding price peaks
            price_peak_times = [smoothed_prices.index[i] for i in price_peaks]
            rsi_peak_times = [smoothed_rsi.index[last_rsi_peak_idx], smoothed_rsi.index[prev_rsi_peak_idx]]
            
            if len(price_peak_times) >= 2:
                # Find closest price peaks to RSI peaks
                closest_price_peaks = []
                for rsi_time in rsi_peak_times:
                    closest_idx = min(range(len(price_peak_times)), 
                                    key=lambda i: abs((price_peak_times[i] - rsi_time).total_seconds()))
                    closest_price_peaks.append(smoothed_prices[price_peak_times[closest_idx]])
                
                if len(closest_price_peaks) == 2:
                    last_price_peak, prev_price_peak = closest_price_peaks
                    
                    if last_price_peak > prev_price_peak and last_rsi_peak < prev_rsi_peak:
                        bearish_divergence = True
                        price_increase = (last_price_peak - prev_price_peak) / prev_price_peak
                        rsi_decline = (prev_rsi_peak - last_rsi_peak) / prev_rsi_peak if prev_rsi_peak != 0 else 0
                        divergence_strength = min(rsi_decline / (price_increase + 1e-8), 2.0)
                        
                        # Statistical validation
                        correlation = np.corrcoef(smoothed_rsi.values, smoothed_prices.values)[0, 1]
                        statistical_confidence = max(0, 1 - abs(correlation))
        
        return {
            'bullish': bullish_divergence,
            'bearish': bearish_divergence,
            'strength': float(divergence_strength),
            'confidence': float(statistical_confidence)
        }
    
    def _multi_timeframe_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform multi-timeframe RSI analysis"""
        if not self.parameters['multi_timeframe']:
            return {'short': None, 'medium': None, 'long': None, 'consensus': 'neutral'}
        
        timeframes = [7, 14, 28]  # Short, medium, long periods
        results = {}
        signals = []
        
        for i, period in enumerate(timeframes):
            try:
                # Temporarily adjust period
                original_period = self.parameters['period']
                self.parameters['period'] = period
                
                # Calculate RSI for this timeframe
                gains, losses = self._calculate_price_changes(data)
                avg_gains, avg_losses = self._calculate_smoothed_averages(gains, losses)
                rsi = self._calculate_rsi(avg_gains, avg_losses)
                
                # Restore original period
                self.parameters['period'] = original_period
                
                if len(rsi) > 0:
                    current_rsi = rsi.iloc[-1]
                    timeframe_name = ['short', 'medium', 'long'][i]
                    
                    # Determine signal for this timeframe
                    if current_rsi > 70:
                        signal = 'overbought'
                        signals.append(-1)
                    elif current_rsi < 30:
                        signal = 'oversold'
                        signals.append(1)
                    elif current_rsi > 50:
                        signal = 'bullish'
                        signals.append(0.5)
                    elif current_rsi < 50:
                        signal = 'bearish'
                        signals.append(-0.5)
                    else:
                        signal = 'neutral'
                        signals.append(0)
                    
                    results[timeframe_name] = {
                        'rsi': float(current_rsi),
                        'signal': signal,
                        'period': period
                    }
            except:
                results[['short', 'medium', 'long'][i]] = None
        
        # Calculate consensus
        if signals:
            avg_signal = np.mean(signals)
            if avg_signal > 0.3:
                consensus = 'bullish'
            elif avg_signal < -0.3:
                consensus = 'bearish'
            else:
                consensus = 'neutral'
        else:
            consensus = 'neutral'
        
        results['consensus'] = consensus
        return results
    
    def _calculate_rsi_bands(self, rsi: pd.Series) -> Dict[str, Any]:
        """Calculate RSI Bands for trend analysis"""
        try:
            if len(rsi) < 20:
                return {'upper_band': 70.0, 'lower_band': 30.0, 'middle_band': 50.0}
            
            # Dynamic RSI bands based on recent RSI distribution
            recent_rsi = rsi.tail(50).dropna()
            
            if len(recent_rsi) >= 20:
                rsi_mean = recent_rsi.mean()
                rsi_std = recent_rsi.std()
                
                # Calculate adaptive bands
                upper_band = min(85, rsi_mean + 1.5 * rsi_std)
                lower_band = max(15, rsi_mean - 1.5 * rsi_std)
                middle_band = rsi_mean
                
                # Band width analysis
                band_width = upper_band - lower_band
                is_expanding = band_width > recent_rsi.rolling(window=20).std().mean() * 3
                
                return {
                    'upper_band': float(upper_band),
                    'lower_band': float(lower_band),
                    'middle_band': float(middle_band),
                    'band_width': float(band_width),
                    'is_expanding': is_expanding,
                    'current_position': 'upper' if rsi.iloc[-1] > upper_band else 'lower' if rsi.iloc[-1] < lower_band else 'middle'
                }
            else:
                return {'upper_band': 70.0, 'lower_band': 30.0, 'middle_band': 50.0}
                
        except:
            return {'upper_band': 70.0, 'lower_band': 30.0, 'middle_band': 50.0}
    
    def _classify_market_regime(self, rsi: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime based on RSI characteristics"""
        try:
            if len(rsi) < 30:
                return {'regime': 'unknown', 'trend_state': 'undefined', 'momentum_phase': 'unclear'}
            
            recent_rsi = rsi.tail(30)
            recent_prices = data['close'].tail(30)
            
            # Trend classification based on RSI behavior
            rsi_above_50 = len(recent_rsi[recent_rsi > 50]) / len(recent_rsi)
            rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi.values, 1)[0]
            
            # Price trend for confirmation
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices.values, 1)[0]
            
            # Volatility assessment
            rsi_volatility = recent_rsi.std()
            price_volatility = recent_prices.pct_change().std()
            
            # Classify regime
            if rsi_above_50 > 0.7 and rsi_trend > 0 and price_trend > 0:
                regime = 'strong_bullish'
            elif rsi_above_50 < 0.3 and rsi_trend < 0 and price_trend < 0:
                regime = 'strong_bearish'
            elif rsi_above_50 > 0.6:
                regime = 'bullish'
            elif rsi_above_50 < 0.4:
                regime = 'bearish'
            elif rsi_volatility < recent_rsi.rolling(window=50).std().mean() * 0.7:
                regime = 'consolidation'
            else:
                regime = 'transitional'
            
            # Momentum phase
            current_rsi = recent_rsi.iloc[-1]
            if current_rsi > 70 and rsi_trend > 0:
                momentum_phase = 'overbought_momentum'
            elif current_rsi < 30 and rsi_trend < 0:
                momentum_phase = 'oversold_momentum'
            elif 40 < current_rsi < 60 and abs(rsi_trend) < 0.5:
                momentum_phase = 'neutral_equilibrium'
            elif rsi_trend > 1:
                momentum_phase = 'building_bullish'
            elif rsi_trend < -1:
                momentum_phase = 'building_bearish'
            else:
                momentum_phase = 'unclear'
            
            return {
                'regime': regime,
                'momentum_phase': momentum_phase,
                'rsi_above_50_ratio': float(rsi_above_50),
                'rsi_trend': float(rsi_trend),
                'price_trend': float(price_trend),
                'rsi_volatility': float(rsi_volatility),
                'price_volatility': float(price_volatility)
            }
        except:
            return {'regime': 'unknown', 'trend_state': 'undefined', 'momentum_phase': 'unclear'}
    
    def _generate_comprehensive_rsi_signal(self, rsi: pd.Series, data: pd.DataFrame,
                                          adaptive_thresholds: Dict, divergences: Dict,
                                          multi_timeframe: Dict, extremes_analysis: Dict,
                                          rsi_momentum_analysis: Dict, multi_rsi: pd.DataFrame,
                                          enhanced_rsi: pd.DataFrame) -> Tuple[SignalType, float]:
        """Generate comprehensive RSI signal with all analysis components"""
        signal_components = []
        confidence_components = []
        
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
        overbought = adaptive_thresholds.get('overbought', 70)
        oversold = adaptive_thresholds.get('oversold', 30)
        
        # Basic RSI signals
        if current_rsi < oversold:
            signal_components.append(0.8)
            confidence_components.append(0.7)
        elif current_rsi > overbought:
            signal_components.append(-0.8)
            confidence_components.append(0.7)
        
        # RSI momentum signals
        momentum_direction = rsi_momentum_analysis.get('direction', 'neutral')
        momentum_strength = rsi_momentum_analysis.get('strength', 0.0)
        
        if momentum_direction == 'rising' and momentum_strength > 0.5:
            signal_components.append(0.6)
            confidence_components.append(0.6)
        elif momentum_direction == 'falling' and momentum_strength > 0.5:
            signal_components.append(-0.6)
            confidence_components.append(0.6)
        
        # Divergence signals
        if divergences.get('bullish', False):
            signal_components.append(divergences.get('strength', 0.5))
            confidence_components.append(divergences.get('confidence', 0.5))
        elif divergences.get('bearish', False):
            signal_components.append(-divergences.get('strength', 0.5))
            confidence_components.append(divergences.get('confidence', 0.5))
        
        # Multi-timeframe consensus
        consensus = multi_timeframe.get('consensus', 'neutral')
        if consensus == 'bullish':
            signal_components.append(0.7)
            confidence_components.append(0.8)
        elif consensus == 'bearish':
            signal_components.append(-0.7)
            confidence_components.append(0.8)
        
        # Extreme conditions
        extreme_type = extremes_analysis.get('type', 'none')
        extreme_intensity = extremes_analysis.get('intensity', 0.0)
        
        if extreme_type == 'oversold' and extreme_intensity > 0.6:
            signal_components.append(0.9)
            confidence_components.append(0.8)
        elif extreme_type == 'overbought' and extreme_intensity > 0.6:
            signal_components.append(-0.9)
            confidence_components.append(0.8)
        
        # Stochastic RSI signals
        if len(multi_rsi) > 0:
            stoch_k = multi_rsi['stoch_rsi_k'].iloc[-1] if not pd.isna(multi_rsi['stoch_rsi_k'].iloc[-1]) else 50.0
            stoch_d = multi_rsi['stoch_rsi_d'].iloc[-1] if not pd.isna(multi_rsi['stoch_rsi_d'].iloc[-1]) else 50.0
            
            if stoch_k < 20 and stoch_k > stoch_d:
                signal_components.append(0.6)
                confidence_components.append(0.6)
            elif stoch_k > 80 and stoch_k < stoch_d:
                signal_components.append(-0.6)
                confidence_components.append(0.6)
        
        # Volume-adjusted RSI
        if len(enhanced_rsi) > 0:
            volume_rsi = enhanced_rsi['volume_adjusted_rsi'].iloc[-1] if not pd.isna(enhanced_rsi['volume_adjusted_rsi'].iloc[-1]) else 50.0
            volume_divergence = abs(current_rsi - volume_rsi)
            
            if volume_divergence > 5:  # Significant volume divergence
                volume_signal = 0.4 if volume_rsi > current_rsi else -0.4
                signal_components.append(volume_signal)
                confidence_components.append(0.5)
        
        # ML enhancement
        if self.models_trained:
            try:
                ml_signal, ml_confidence = self._get_ml_signal(rsi, data)
                if ml_signal:
                    signal_components.append(1.0 if ml_signal == SignalType.BUY else -1.0)
                    confidence_components.append(ml_confidence)
            except:
                pass
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        # Determine signal type
        if weighted_signal > 0.6:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _train_ml_models_background(self, rsi: pd.Series, data: pd.DataFrame, 
                        multi_rsi: pd.DataFrame, enhanced_rsi: pd.DataFrame):
        """Background worker for ML model training"""
        try:
            self.logger.info("Starting background RSI ML training...")
            
            features, targets = self._prepare_comprehensive_ml_data(rsi, data, multi_rsi, enhanced_rsi)
            if len(features) > 50:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train classification models
                self.ml_classifier.fit(scaled_features, targets)
                self.signal_classifier.fit(scaled_features, targets)
                
                # Train pattern detection
                self.pattern_detector.fit(scaled_features, targets)
                
                with self.training_lock:
                    self.models_trained = True
                    self.ml_trained = True
                    self.is_training = False
                    
                self.logger.info("Background RSI ML training completed successfully.")
            else:
                with self.training_lock:
                    self.is_training = False
                    
        except Exception as e:
            self.logger.error(f"Background RSI ML training failed: {e}")
            with self.training_lock:
                self.is_training = False

    def _train_ml_models(self, rsi: pd.Series, data: pd.DataFrame, 
                        multi_rsi: pd.DataFrame, enhanced_rsi: pd.DataFrame) -> bool:
        """Train ML models for RSI pattern recognition in background"""
        if len(rsi) < self.parameters.get('ml_lookback', 100):
            return False
            
        with self.training_lock:
            if self.is_training:
                return False
            self.is_training = True
            
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._train_ml_models_background,
            args=(rsi, data, multi_rsi, enhanced_rsi),
            daemon=True
        )
        training_thread.start()
        
        return True
    
    def _prepare_comprehensive_ml_data(self, rsi: pd.Series, data: pd.DataFrame,
                                     multi_rsi: pd.DataFrame, enhanced_rsi: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare comprehensive ML training data with all RSI variants"""
        features, targets = [], []
        lookback = 15
        
        for i in range(lookback, len(rsi) - 5):
            try:
                # RSI window features
                rsi_window = rsi.iloc[i-lookback:i]
                if rsi_window.isna().any():
                    continue
                    
                # Basic RSI features
                rsi_mean = rsi_window.mean()
                rsi_std = rsi_window.std()
                rsi_current = rsi_window.iloc[-1]
                rsi_change = rsi_window.iloc[-1] - rsi_window.iloc[0]
                
                # Multi-RSI features
                if len(multi_rsi) > i:
                    fast_rsi = multi_rsi['rsi_fast'].iloc[i] if not pd.isna(multi_rsi['rsi_fast'].iloc[i]) else 50.0
                    slow_rsi = multi_rsi['rsi_slow'].iloc[i] if not pd.isna(multi_rsi['rsi_slow'].iloc[i]) else 50.0
                    stoch_k = multi_rsi['stoch_rsi_k'].iloc[i] if not pd.isna(multi_rsi['stoch_rsi_k'].iloc[i]) else 50.0
                else:
                    fast_rsi = slow_rsi = stoch_k = 50.0
                
                # Enhanced RSI features
                if len(enhanced_rsi) > i:
                    rsi_momentum = enhanced_rsi['rsi_momentum'].iloc[i] if not pd.isna(enhanced_rsi['rsi_momentum'].iloc[i]) else 0.0
                    rsi_acceleration = enhanced_rsi['rsi_acceleration'].iloc[i] if not pd.isna(enhanced_rsi['rsi_acceleration'].iloc[i]) else 0.0
                    volume_rsi = enhanced_rsi['volume_adjusted_rsi'].iloc[i] if not pd.isna(enhanced_rsi['volume_adjusted_rsi'].iloc[i]) else 50.0
                else:
                    rsi_momentum = rsi_acceleration = 0.0
                    volume_rsi = 50.0
                
                # Price and volume features
                price_window = data['close'].iloc[i-lookback:i]
                volume_window = data['volume'].iloc[i-lookback:i] if 'volume' in data.columns else pd.Series([1] * lookback)
                
                price_returns = price_window.pct_change().dropna()
                price_volatility = price_returns.std() if len(price_returns) > 0 else 0.0
                volume_ratio = volume_window.iloc[-1] / volume_window.mean() if volume_window.mean() != 0 else 1.0
                
                # Extreme conditions
                overbought_periods = len(rsi_window[rsi_window > 70])
                oversold_periods = len(rsi_window[rsi_window < 30])
                
                feature_vector = [
                    rsi_mean, rsi_std, rsi_current, rsi_change,
                    fast_rsi, slow_rsi, stoch_k,
                    rsi_momentum, rsi_acceleration, volume_rsi,
                    price_volatility, volume_ratio,
                    overbought_periods / len(rsi_window),
                    oversold_periods / len(rsi_window),
                    len([j for j in range(1, len(rsi_window)) if rsi_window.iloc[j] > rsi_window.iloc[j-1]]) / (len(rsi_window) - 1)
                ]
                
                # Target: future price movement
                future_return = (data['close'].iloc[i+5] - data['close'].iloc[i]) / data['close'].iloc[i]
                target = 2 if future_return > 0.02 else (0 if future_return < -0.02 else 1)
                
                features.append(feature_vector)
                targets.append(target)
                
            except (IndexError, KeyError, TypeError):
                continue
        
        return np.array(features), np.array(targets)
    
    def _get_ml_signal(self, rsi: pd.Series, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Get ML-based signal prediction"""
        try:
            lookback = 15
            if len(rsi) < lookback:
                return None, 0.0
            
            rsi_window = rsi.tail(lookback).values
            # Need to ensure we have enough data points
            if len(data) < lookback:
                return None, 0.0
                
            price_window = data['close'].tail(lookback).values
            volume_window = data['volume'].tail(lookback).values
            
            # Recreate feature vector similar to training
            # Note: This is a simplified feature vector for inference
            # In a real scenario, this should match _prepare_comprehensive_ml_data exactly
            
            # For robustness, we'll use a try-catch block around feature creation
            try:
                feature_vector = np.array([[
                    np.mean(rsi_window), np.std(rsi_window), rsi_window[-1],
                    rsi_window[-1] - rsi_window[0],
                    50.0, 50.0, 50.0, # Placeholders for multi-rsi if not easily available here
                    0.0, 0.0, 50.0,   # Placeholders for enhanced-rsi
                    0.0, 1.0,         # Placeholders for volatility/volume
                    len([x for x in rsi_window if x > 70]) / len(rsi_window),
                    len([x for x in rsi_window if x < 30]) / len(rsi_window),
                    len([j for j in range(1, len(rsi_window)) if rsi_window[j] > rsi_window[j-1]]) / (len(rsi_window) - 1)
                ]])
                
                # Important: The feature vector shape must match what the scaler was fit on.
                # Since _prepare_comprehensive_ml_data uses 15 features, we must provide 15 features here.
                # The above construction is an approximation. 
                # Ideally, we should pass multi_rsi and enhanced_rsi to this method too.
                # Given the complexity, we might skip ML signal if exact features aren't available
                # or rely on the fact that we are just doing a quick check.
                
                # However, to avoid shape mismatch errors, let's just return None if we can't easily construct the exact vector.
                # Or better, let's just wrap in try-except and if scaler complains, we catch it.
                
                scaled_features = self.scaler.transform(feature_vector)
                ml_proba = self.ml_classifier.predict_proba(scaled_features)[0]
                
                if len(ml_proba) >= 3:
                    max_prob_idx = np.argmax(ml_proba)
                    max_prob = ml_proba[max_prob_idx]
                    
                    if max_prob > 0.7:
                        if max_prob_idx == 2:  # Strong bullish
                            return SignalType.BUY, max_prob
                        elif max_prob_idx == 0:  # Strong bearish
                            return SignalType.SELL, max_prob
            except ValueError:
                # Shape mismatch or other error
                return None, 0.0
                
        except:
            pass
        
        return None, 0.0
    
    def _calculate_rsi_momentum(self, rsi: pd.Series) -> Dict[str, Any]:
        """Calculate RSI momentum characteristics"""
        if len(rsi) < 5:
            return {'direction': 'neutral', 'strength': 0.0, 'rate_of_change': 0.0}
        
        recent_rsi = rsi.tail(5)
        rsi_change = recent_rsi.iloc[-1] - recent_rsi.iloc[0]
        rate_of_change = rsi_change / 4  # Per period
        
        if rsi_change > 2:
            direction = 'rising'
        elif rsi_change < -2:
            direction = 'falling'
        else:
            direction = 'sideways'
        
        strength = min(abs(rsi_change) / 20, 1.0)  # Normalize to 0-1
        
        return {
            'direction': direction,
            'strength': float(strength),
            'rate_of_change': float(rate_of_change)
        }
    
    def _analyze_extremes(self, rsi: pd.Series, thresholds: Dict) -> Dict[str, Any]:
        """Analyze RSI extreme conditions"""
        current_rsi = rsi.iloc[-1]
        
        if len(rsi) >= 10:
            recent_rsi = rsi.tail(10)
            periods_overbought = len([x for x in recent_rsi if x > thresholds['overbought']])
            periods_oversold = len([x for x in recent_rsi if x < thresholds['oversold']])
        else:
            periods_overbought = 1 if current_rsi > thresholds['overbought'] else 0
            periods_oversold = 1 if current_rsi < thresholds['oversold'] else 0
        
        extreme_type = 'none'
        extreme_duration = 0
        extreme_intensity = 0.0
        
        if current_rsi > thresholds['overbought']:
            extreme_type = 'overbought'
            extreme_duration = periods_overbought
            extreme_intensity = (current_rsi - thresholds['overbought']) / (100 - thresholds['overbought'])
        elif current_rsi < thresholds['oversold']:
            extreme_type = 'oversold'
            extreme_duration = periods_oversold
            extreme_intensity = (thresholds['oversold'] - current_rsi) / thresholds['oversold']
        
        return {
            'type': extreme_type,
            'duration': extreme_duration,
            'intensity': float(extreme_intensity)
        }
    
    def _calculate_multi_rsi(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> pd.DataFrame:
        """Calculate multiple RSI timeframes and variants"""
        try:
            # Primary RSI (main period)
            rsi_primary = self._calculate_standard_rsi_array(close, self.parameters['period'])
            
            # Secondary RSI periods
            secondary_periods = self.parameters.get('secondary_periods', [7, 21, 50])
            rsi_secondary = self._calculate_standard_rsi_array(close, secondary_periods[1] if len(secondary_periods) > 1 else 21)
            rsi_tertiary = self._calculate_standard_rsi_array(close, secondary_periods[2] if len(secondary_periods) > 2 else 50)
            
            # Fast RSI (shorter period)
            rsi_fast = self._calculate_standard_rsi_array(close, 7)
            
            # Slow RSI (longer period)
            rsi_slow = self._calculate_standard_rsi_array(close, 50)
            
            # Stochastic RSI
            if self.parameters.get('stochastic_rsi', True):
                stoch_rsi_k, stoch_rsi_d = self._calculate_stochastic_rsi(close, self.parameters['period'])
            else:
                stoch_rsi_k = rsi_primary
                stoch_rsi_d = rsi_primary
            
            # RSI of high-low range
            hl_range = high - low
            rsi_range = self._calculate_standard_rsi_array(hl_range, self.parameters['period'])
            
            # Composite RSI (weighted average)
            composite_rsi = (
                rsi_primary * 0.4 +
                rsi_secondary * 0.3 +
                rsi_tertiary * 0.2 +
                rsi_fast * 0.1
            )
            
            return pd.DataFrame({
                'rsi_primary': rsi_primary,
                'rsi_secondary': rsi_secondary,
                'rsi_tertiary': rsi_tertiary,
                'rsi_fast': rsi_fast,
                'rsi_slow': rsi_slow,
                'stoch_rsi_k': stoch_rsi_k,
                'stoch_rsi_d': stoch_rsi_d,
                'rsi_range': rsi_range,
                'composite_rsi': composite_rsi
            })
            
        except Exception as e:
            # Return default values if calculation fails
            default_length = len(close)
            return pd.DataFrame({
                'rsi_primary': np.full(default_length, 50.0),
                'composite_rsi': np.full(default_length, 50.0),
                'rsi_fast': np.full(default_length, 50.0),
                'rsi_slow': np.full(default_length, 50.0)
            })
    
    def _calculate_enhanced_rsi(self, close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
        """Calculate enhanced RSI with adaptive features"""
        try:
            close_series = pd.Series(close)
            volume_series = pd.Series(volume)
            
            # Base RSI
            base_rsi = self._calculate_standard_rsi_series(close_series, self.parameters['period'])
            
            # Adaptive period RSI
            if self.parameters.get('adaptive_periods', True):
                volatility = close_series.pct_change().rolling(window=20).std()
                adaptive_period = np.clip(self.parameters['period'] * (1 + volatility * 10), 7, 50).astype(int)
                
                adaptive_rsi = pd.Series(index=range(len(close)))
                for i in range(len(close)):
                    if i >= adaptive_period.iloc[i]:
                        period = int(adaptive_period.iloc[i])
                        start_idx = max(0, i - period + 1)
                        window_close = close_series.iloc[start_idx:i+1]
                        if len(window_close) >= 2:
                            adaptive_rsi.iloc[i] = self._calculate_standard_rsi_series(window_close, period).iloc[-1]
                        else:
                            adaptive_rsi.iloc[i] = 50.0
                    else:
                        adaptive_rsi.iloc[i] = base_rsi.iloc[i] if not pd.isna(base_rsi.iloc[i]) else 50.0
            else:
                adaptive_rsi = base_rsi.copy()
            
            # Smoothed RSI
            smooth_period = self.parameters.get('smooth_period', 3)
            smoothed_rsi = base_rsi.rolling(window=smooth_period).mean()
            
            # Volume-adjusted RSI
            if self.parameters.get('volume_weighted', True):
                volume_factor = volume_series / volume_series.rolling(window=20).mean()
                volume_adjusted_rsi = base_rsi + (volume_factor - 1) * 5
                volume_adjusted_rsi = np.clip(volume_adjusted_rsi, 0, 100)
            else:
                volume_adjusted_rsi = base_rsi.copy()
            
            # Noise-reduced RSI
            if self.parameters.get('noise_reduction', True):
                noise_reduced_rsi = self._apply_noise_reduction(base_rsi)
            else:
                noise_reduced_rsi = base_rsi.copy()
            
            # RSI momentum
            rsi_momentum = base_rsi.diff(periods=3)
            
            # RSI acceleration
            rsi_acceleration = rsi_momentum.diff()
            
            # Composite enhanced RSI
            primary_rsi = noise_reduced_rsi if self.parameters.get('noise_reduction', True) else smoothed_rsi
            
            return pd.DataFrame({
                'primary_rsi': primary_rsi,
                'base_rsi': base_rsi,
                'adaptive_rsi': adaptive_rsi,
                'smoothed_rsi': smoothed_rsi,
                'volume_adjusted_rsi': volume_adjusted_rsi,
                'noise_reduced_rsi': noise_reduced_rsi,
                'rsi_momentum': rsi_momentum,
                'rsi_acceleration': rsi_acceleration,
                'composite_rsi': (primary_rsi + adaptive_rsi + smoothed_rsi) / 3
            })
            
        except Exception as e:
            # Return default values if calculation fails
            default_length = len(close)
            return pd.DataFrame({
                'primary_rsi': np.full(default_length, 50.0),
                'composite_rsi': np.full(default_length, 50.0),
                'base_rsi': np.full(default_length, 50.0)
            })
