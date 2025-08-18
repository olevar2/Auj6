"""
Advanced TSI Oscillator Indicator with Machine Learning Integration

This module implements a sophisticated True Strength Index (TSI) oscillator with:
- Advanced TSI calculation with adaptive parameters
- Machine learning signal enhancement and pattern recognition
- Multi-timeframe analysis and regime detection
- Advanced divergence detection and confirmation
- Risk assessment and confidence scoring

Part of the ASD trading platform's humanitarian mission.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface


class TSIOscillator(StandardIndicatorInterface):
    """
    Advanced True Strength Index (TSI) Oscillator with Machine Learning
    
    Features:
    - Multi-period TSI with adaptive smoothing parameters
    - ML-based signal classification and trend strength assessment
    - Advanced divergence detection with volume confirmation
    - Regime-aware oscillator interpretation and filtering
    - Risk and confidence scoring with position sizing recommendations
    - Multi-timeframe convergence and momentum analysis
    """
    
    def __init__(self, 
                 fast_period: int = 13,
                 slow_period: int = 25,
                 signal_period: int = 13,
                 smoothing_periods: List[int] = [7, 21],
                 price_change_period: int = 1,
                 ml_lookback: int = 252,
                 divergence_periods: List[int] = [10, 20, 40],
                 confidence_threshold: float = 0.6):
        """
        Initialize Advanced TSI Oscillator
        
        Args:
            fast_period: Fast smoothing period for TSI
            slow_period: Slow smoothing period for TSI
            signal_period: Signal line smoothing period
            smoothing_periods: Additional smoothing periods for multi-timeframe analysis
            price_change_period: Period for price change calculation
            ml_lookback: Lookback period for ML training
            divergence_periods: Periods for divergence detection
            confidence_threshold: Minimum confidence for signal generation
        """
        super().__init__()
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.smoothing_periods = smoothing_periods
        self.price_change_period = price_change_period
        self.ml_lookback = ml_lookback
        self.divergence_periods = divergence_periods
        self.confidence_threshold = confidence_threshold
        
        # ML components
        self.signal_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        self.trend_strength_regressor = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        self.pattern_detector = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.momentum_classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=1000,
            random_state=42
        )
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Clustering for regime detection
        self.regime_clusterer = KMeans(n_clusters=4, random_state=42)
        self.pca = PCA(n_components=0.95)
        
        # State tracking
        self.is_trained = False
        self.feature_columns = []
        self.last_regime = None
        
        # Adaptive parameters
        self.adaptive_smoothing = True
        self.volume_weighting = True
        self.regime_awareness = True
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced TSI oscillator with ML integration
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing TSI analysis, signals, and metadata
        """
        try:
            if len(data) < max(self.slow_period * 2, self.ml_lookback // 2):
                return self._create_error_result("Insufficient data for TSI calculation")
            
            # Calculate TSI components
            tsi_data = self._calculate_advanced_tsi(data)
            
            # Calculate adaptive parameters
            adaptive_params = self._calculate_adaptive_parameters(data, tsi_data)
            
            # Generate base oscillator signals
            base_signals = self._generate_base_signals(data, tsi_data, adaptive_params)
            
            # Detect divergences
            divergence_analysis = self._detect_tsi_divergences(data, tsi_data)
            
            # Perform regime and momentum analysis
            regime_analysis = self._analyze_market_regime(data, tsi_data)
            momentum_analysis = self._analyze_momentum_patterns(data, tsi_data)
            
            # Train or update ML models
            if not self.is_trained and len(data) >= self.ml_lookback:
                self._train_ml_models(data, tsi_data, base_signals)
            
            # Generate ML-enhanced signals
            ml_signals = self._generate_ml_signals(data, tsi_data, base_signals)
            
            # Calculate signal strength and confidence
            signal_analysis = self._analyze_signal_strength(
                data, tsi_data, base_signals, ml_signals, divergence_analysis
            )
            
            # Perform multi-timeframe synthesis
            synthesized_signals = self._synthesize_multi_timeframe_signals(
                data, tsi_data, signal_analysis, momentum_analysis
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(data, tsi_data, synthesized_signals)
            
            # Generate final recommendations
            recommendations = self._generate_trading_recommendations(
                synthesized_signals, risk_metrics, regime_analysis
            )
            
            return {
                'tsi_data': tsi_data,
                'base_signals': base_signals,
                'divergence_analysis': divergence_analysis,
                'ml_signals': ml_signals,
                'signal_analysis': signal_analysis,
                'synthesized_signals': synthesized_signals,
                'regime_analysis': regime_analysis,
                'momentum_analysis': momentum_analysis,
                'risk_metrics': risk_metrics,
                'recommendations': recommendations,
                'adaptive_parameters': adaptive_params,
                'confidence_score': float(signal_analysis.get('overall_confidence', 0.0)),
                'signal_strength': float(signal_analysis.get('signal_strength', 0.0)),
                'metadata': self._generate_metadata(data, tsi_data, synthesized_signals)
            }
            
        except Exception as e:
            return self._create_error_result(f"TSI oscillator calculation error: {str(e)}")
    
    def _calculate_advanced_tsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced TSI with multiple smoothing methods"""
        tsi_data = pd.DataFrame(index=data.index)
        
        # Price changes
        price_changes = data['close'].diff(self.price_change_period)
        abs_price_changes = abs(price_changes)
        
        # Standard TSI calculation
        tsi_data['tsi'] = self._calculate_standard_tsi(price_changes, abs_price_changes)
        
        # Signal line
        tsi_data['tsi_signal'] = tsi_data['tsi'].ewm(span=self.signal_period).mean()
        
        # TSI Histogram (difference between TSI and signal)
        tsi_data['tsi_histogram'] = tsi_data['tsi'] - tsi_data['tsi_signal']
        
        # Additional smoothing periods
        for period in self.smoothing_periods:
            tsi_data[f'tsi_smooth_{period}'] = tsi_data['tsi'].ewm(span=period).mean()
        
        # TSI momentum and acceleration
        tsi_data['tsi_momentum'] = tsi_data['tsi'].diff()
        tsi_data['tsi_acceleration'] = tsi_data['tsi_momentum'].diff()
        
        # TSI velocity (rate of change)
        tsi_data['tsi_velocity'] = tsi_data['tsi'].pct_change(5)
        
        # Adaptive TSI with volatility adjustment
        if self.adaptive_smoothing:
            tsi_data['adaptive_tsi'] = self._calculate_adaptive_tsi(data, price_changes, abs_price_changes)
        
        # Volume-weighted TSI
        if self.volume_weighting and 'volume' in data.columns:
            tsi_data['volume_weighted_tsi'] = self._calculate_volume_weighted_tsi(
                data, price_changes, abs_price_changes
            )
        
        # Multi-timeframe TSI convergence
        tsi_data['mtf_convergence'] = self._calculate_mtf_convergence(tsi_data)
        
        return tsi_data.fillna(0)
    
    def _calculate_standard_tsi(self, price_changes: pd.Series, abs_price_changes: pd.Series) -> pd.Series:
        """Calculate standard TSI"""
        # First smoothing
        pc_smooth1 = price_changes.ewm(span=self.slow_period).mean()
        apc_smooth1 = abs_price_changes.ewm(span=self.slow_period).mean()
        
        # Second smoothing
        pc_smooth2 = pc_smooth1.ewm(span=self.fast_period).mean()
        apc_smooth2 = apc_smooth1.ewm(span=self.fast_period).mean()
        
        # TSI calculation
        tsi = 100 * (pc_smooth2 / apc_smooth2)
        return tsi.fillna(0)
    
    def _calculate_adaptive_tsi(self, data: pd.DataFrame, price_changes: pd.Series, 
                               abs_price_changes: pd.Series) -> pd.Series:
        """Calculate TSI with adaptive parameters based on volatility"""
        # Market volatility for parameter adaptation
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        vol_percentile = volatility.rolling(window=60).rank(pct=True)
        
        # Adaptive parameter calculation
        adaptive_slow = self.slow_period * (1 + (vol_percentile - 0.5) * 0.4)
        adaptive_fast = self.fast_period * (1 + (vol_percentile - 0.5) * 0.3)
        
        # Calculate adaptive TSI
        tsi_values = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            if i >= self.slow_period:
                current_slow = max(5, int(adaptive_slow.iloc[i]) if not pd.isna(adaptive_slow.iloc[i]) else self.slow_period)
                current_fast = max(3, int(adaptive_fast.iloc[i]) if not pd.isna(adaptive_fast.iloc[i]) else self.fast_period)
                
                # Window data
                end_idx = i + 1
                start_idx = max(0, end_idx - current_slow * 2)
                
                window_pc = price_changes.iloc[start_idx:end_idx]
                window_apc = abs_price_changes.iloc[start_idx:end_idx]
                
                # Calculate TSI for this window
                if len(window_pc) >= current_slow:
                    pc_smooth1 = window_pc.ewm(span=current_slow).mean()
                    apc_smooth1 = window_apc.ewm(span=current_slow).mean()
                    
                    pc_smooth2 = pc_smooth1.ewm(span=current_fast).mean()
                    apc_smooth2 = apc_smooth1.ewm(span=current_fast).mean()
                    
                    if apc_smooth2.iloc[-1] != 0:
                        tsi_values.iloc[i] = 100 * (pc_smooth2.iloc[-1] / apc_smooth2.iloc[-1])
        
        return tsi_values.fillna(0)
    
    def _calculate_volume_weighted_tsi(self, data: pd.DataFrame, price_changes: pd.Series, 
                                     abs_price_changes: pd.Series) -> pd.Series:
        """Calculate volume-weighted TSI"""
        volume = data['volume']
        
        # Volume-weighted price changes
        vw_price_changes = price_changes * volume
        vw_abs_price_changes = abs_price_changes * volume
        
        # Volume-weighted smoothing
        vw_pc_smooth1 = (vw_price_changes.rolling(window=self.slow_period).sum() / 
                        volume.rolling(window=self.slow_period).sum())
        vw_apc_smooth1 = (vw_abs_price_changes.rolling(window=self.slow_period).sum() / 
                         volume.rolling(window=self.slow_period).sum())
        
        vw_pc_smooth2 = (vw_pc_smooth1.rolling(window=self.fast_period).mean())
        vw_apc_smooth2 = (vw_apc_smooth1.rolling(window=self.fast_period).mean())
        
        # Volume-weighted TSI
        vw_tsi = 100 * (vw_pc_smooth2 / vw_apc_smooth2)
        return vw_tsi.fillna(0)
    
    def _calculate_mtf_convergence(self, tsi_data: pd.DataFrame) -> pd.Series:
        """Calculate multi-timeframe TSI convergence"""
        convergence_score = pd.Series(0.0, index=tsi_data.index)
        
        # Base TSI direction
        base_direction = np.sign(tsi_data['tsi'])
        
        # Check convergence with smoothed versions
        for period in self.smoothing_periods:
            if f'tsi_smooth_{period}' in tsi_data.columns:
                smooth_direction = np.sign(tsi_data[f'tsi_smooth_{period}'])
                convergence_score += (base_direction == smooth_direction).astype(float)
        
        # Normalize convergence score
        convergence_score = convergence_score / len(self.smoothing_periods)
        
        return convergence_score
    
    def _calculate_adaptive_parameters(self, data: pd.DataFrame, tsi_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate adaptive parameters based on market conditions"""
        # Market volatility analysis
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        vol_percentile = volatility.rolling(window=60).rank(pct=True).iloc[-1] if len(volatility) > 60 else 0.5
        
        # TSI volatility and range
        tsi_volatility = tsi_data['tsi'].rolling(window=20).std().iloc[-1]
        tsi_range = (tsi_data['tsi'].rolling(window=20).max() - 
                    tsi_data['tsi'].rolling(window=20).min()).iloc[-1]
        
        # Adaptive thresholds
        base_threshold = 20.0
        vol_adjustment = (vol_percentile - 0.5) * base_threshold * 0.5
        
        overbought_threshold = base_threshold + vol_adjustment
        oversold_threshold = -(base_threshold + vol_adjustment)
        
        # Signal sensitivity
        sensitivity = 1.0 + (vol_percentile - 0.5) * 0.8
        
        return {
            'overbought_threshold': np.clip(overbought_threshold, 15, 35),
            'oversold_threshold': np.clip(oversold_threshold, -35, -15),
            'signal_sensitivity': np.clip(sensitivity, 0.3, 2.0),
            'volatility_percentile': vol_percentile,
            'tsi_volatility': tsi_volatility if not pd.isna(tsi_volatility) else 10.0,
            'tsi_range': tsi_range if not pd.isna(tsi_range) else 40.0
        }
    
    def _generate_base_signals(self, data: pd.DataFrame, tsi_data: pd.DataFrame, 
                              adaptive_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base TSI oscillator signals"""
        tsi = tsi_data['tsi']
        tsi_signal = tsi_data['tsi_signal']
        tsi_histogram = tsi_data['tsi_histogram']
        
        overbought = adaptive_params['overbought_threshold']
        oversold = adaptive_params['oversold_threshold']
        
        # Basic oscillator signals
        buy_signals = (tsi < oversold) & (tsi.shift(1) >= oversold)
        sell_signals = (tsi > overbought) & (tsi.shift(1) <= overbought)
        
        # Crossover signals
        bullish_crossover = (tsi > tsi_signal) & (tsi.shift(1) <= tsi_signal.shift(1))
        bearish_crossover = (tsi < tsi_signal) & (tsi.shift(1) >= tsi_signal.shift(1))
        
        # Zero line crossover
        zero_cross_up = (tsi > 0) & (tsi.shift(1) <= 0)
        zero_cross_down = (tsi < 0) & (tsi.shift(1) >= 0)
        
        # Histogram signals
        histogram_bullish = (tsi_histogram > 0) & (tsi_histogram.shift(1) <= 0)
        histogram_bearish = (tsi_histogram < 0) & (tsi_histogram.shift(1) >= 0)
        
        # Momentum signals
        momentum_acceleration = tsi_data['tsi_acceleration'] > 0
        momentum_deceleration = tsi_data['tsi_acceleration'] < 0
        
        # Multi-timeframe confirmation
        mtf_bullish = tsi_data['mtf_convergence'] > 0.6
        mtf_bearish = tsi_data['mtf_convergence'] < -0.6
        
        # Signal strength calculation
        signal_strength = np.abs(tsi / 100) * tsi_data['mtf_convergence']
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'bullish_crossover': bullish_crossover,
            'bearish_crossover': bearish_crossover,
            'zero_cross_up': zero_cross_up,
            'zero_cross_down': zero_cross_down,
            'histogram_bullish': histogram_bullish,
            'histogram_bearish': histogram_bearish,
            'momentum_acceleration': momentum_acceleration,
            'momentum_deceleration': momentum_deceleration,
            'mtf_bullish': mtf_bullish,
            'mtf_bearish': mtf_bearish,
            'signal_strength': signal_strength,
            'oscillator_bias': np.where(tsi > 10, 1, np.where(tsi < -10, -1, 0))
        }
    
    def _detect_tsi_divergences(self, data: pd.DataFrame, tsi_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect TSI divergences with multiple confirmation methods"""
        tsi = tsi_data['tsi']
        prices = data['close']
        
        divergences = {}
        
        for period in self.divergence_periods:
            if len(data) < period * 2:
                continue
            
            # Find peaks and troughs
            price_peaks = self._find_peaks(prices, period // 2)
            price_troughs = self._find_troughs(prices, period // 2)
            tsi_peaks = self._find_peaks(tsi, period // 2)
            tsi_troughs = self._find_troughs(tsi, period // 2)
            
            # Bullish divergence detection
            bullish_div = self._detect_bullish_divergence(
                prices, tsi, price_troughs, tsi_troughs, period
            )
            
            # Bearish divergence detection
            bearish_div = self._detect_bearish_divergence(
                prices, tsi, price_peaks, tsi_peaks, period
            )
            
            # Volume confirmation (if available)
            if 'volume' in data.columns:
                volume_confirmation = self._confirm_divergence_with_volume(
                    data['volume'], bullish_div, bearish_div, period
                )
                bullish_div = bullish_div & volume_confirmation['bullish']
                bearish_div = bearish_div & volume_confirmation['bearish']
            
            divergences[f'bullish_divergence_{period}'] = bullish_div
            divergences[f'bearish_divergence_{period}'] = bearish_div
        
        # Composite divergence signals
        bullish_composite = pd.Series(False, index=data.index)
        bearish_composite = pd.Series(False, index=data.index)
        
        for period in self.divergence_periods:
            if f'bullish_divergence_{period}' in divergences:
                bullish_composite |= divergences[f'bullish_divergence_{period}']
                bearish_composite |= divergences[f'bearish_divergence_{period}']
        
        # Divergence strength
        divergence_strength = self._calculate_divergence_strength(
            data, tsi_data, bullish_composite, bearish_composite
        )
        
        divergences.update({
            'bullish_composite': bullish_composite,
            'bearish_composite': bearish_composite,
            'divergence_strength': divergence_strength
        })
        
        return divergences
    
    def _detect_bullish_divergence(self, prices: pd.Series, tsi: pd.Series, 
                                  price_troughs: pd.Series, tsi_troughs: pd.Series, 
                                  period: int) -> pd.Series:
        """Detect bullish divergence patterns"""
        bullish_div = pd.Series(False, index=prices.index)
        
        for i in range(period, len(prices)):
            if price_troughs[i] and tsi_troughs[i]:
                # Look for previous trough
                for j in range(max(0, i - period), i):
                    if price_troughs[j] and tsi_troughs[j]:
                        # Check divergence condition
                        if (prices.iloc[i] < prices.iloc[j] and 
                            tsi.iloc[i] > tsi.iloc[j] and
                            abs(i - j) >= period // 2):
                            bullish_div.iloc[i] = True
                            break
        
        return bullish_div
    
    def _detect_bearish_divergence(self, prices: pd.Series, tsi: pd.Series, 
                                  price_peaks: pd.Series, tsi_peaks: pd.Series, 
                                  period: int) -> pd.Series:
        """Detect bearish divergence patterns"""
        bearish_div = pd.Series(False, index=prices.index)
        
        for i in range(period, len(prices)):
            if price_peaks[i] and tsi_peaks[i]:
                # Look for previous peak
                for j in range(max(0, i - period), i):
                    if price_peaks[j] and tsi_peaks[j]:
                        # Check divergence condition
                        if (prices.iloc[i] > prices.iloc[j] and 
                            tsi.iloc[i] < tsi.iloc[j] and
                            abs(i - j) >= period // 2):
                            bearish_div.iloc[i] = True
                            break
        
        return bearish_div
    
    def _find_peaks(self, series: pd.Series, window: int) -> pd.Series:
        """Find peaks in a series"""
        peaks = pd.Series(False, index=series.index)
        for i in range(window, len(series) - window):
            if (series.iloc[i] == series.iloc[i-window:i+window+1].max() and
                series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]):
                peaks.iloc[i] = True
        return peaks
    
    def _find_troughs(self, series: pd.Series, window: int) -> pd.Series:
        """Find troughs in a series"""
        troughs = pd.Series(False, index=series.index)
        for i in range(window, len(series) - window):
            if (series.iloc[i] == series.iloc[i-window:i+window+1].min() and
                series.iloc[i] < series.iloc[i-1] and series.iloc[i] < series.iloc[i+1]):
                troughs.iloc[i] = True
        return troughs
    
    def _confirm_divergence_with_volume(self, volume: pd.Series, bullish_div: pd.Series, 
                                       bearish_div: pd.Series, period: int) -> Dict[str, pd.Series]:
        """Confirm divergences with volume analysis"""
        avg_volume = volume.rolling(window=period).mean()
        
        # Volume confirmation for bullish divergence
        bullish_vol_confirm = pd.Series(False, index=volume.index)
        for i in range(len(bullish_div)):
            if bullish_div.iloc[i]:
                # Check if volume is above average during divergence
                recent_volume = volume.iloc[max(0, i-5):i+1].mean()
                if recent_volume > avg_volume.iloc[i] * 1.1:
                    bullish_vol_confirm.iloc[i] = True
        
        # Volume confirmation for bearish divergence
        bearish_vol_confirm = pd.Series(False, index=volume.index)
        for i in range(len(bearish_div)):
            if bearish_div.iloc[i]:
                recent_volume = volume.iloc[max(0, i-5):i+1].mean()
                if recent_volume > avg_volume.iloc[i] * 1.1:
                    bearish_vol_confirm.iloc[i] = True
        
        return {
            'bullish': bullish_vol_confirm,
            'bearish': bearish_vol_confirm
        }
    
    def _calculate_divergence_strength(self, data: pd.DataFrame, tsi_data: pd.DataFrame,
                                     bullish_div: pd.Series, bearish_div: pd.Series) -> pd.Series:
        """Calculate strength of divergence signals"""
        strength = pd.Series(0.0, index=data.index)
        
        # Price momentum
        price_momentum = data['close'].pct_change(5)
        
        # TSI momentum
        tsi_momentum = tsi_data['tsi_momentum']
        
        # Calculate strength based on momentum divergence
        for i in range(len(data)):
            if bullish_div.iloc[i]:
                # Strength based on price decline vs TSI improvement
                price_change = price_momentum.iloc[i] if not pd.isna(price_momentum.iloc[i]) else 0
                tsi_change = tsi_momentum.iloc[i] if not pd.isna(tsi_momentum.iloc[i]) else 0
                if price_change < 0 and tsi_change > 0:
                    strength.iloc[i] = min(1.0, abs(price_change) + abs(tsi_change))
            
            elif bearish_div.iloc[i]:
                # Strength based on price rise vs TSI decline
                price_change = price_momentum.iloc[i] if not pd.isna(price_momentum.iloc[i]) else 0
                tsi_change = tsi_momentum.iloc[i] if not pd.isna(tsi_momentum.iloc[i]) else 0
                if price_change > 0 and tsi_change < 0:
                    strength.iloc[i] = min(1.0, abs(price_change) + abs(tsi_change))
        
        return strength
    
    def _analyze_market_regime(self, data: pd.DataFrame, tsi_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime using TSI patterns"""
        if not self.regime_awareness:
            return {'regime': 'normal', 'confidence': 0.5}
        
        # TSI characteristics for regime detection
        tsi_mean = tsi_data['tsi'].rolling(window=20).mean().iloc[-1]
        tsi_std = tsi_data['tsi'].rolling(window=20).std().iloc[-1]
        tsi_range = (tsi_data['tsi'].rolling(window=20).max() - 
                    tsi_data['tsi'].rolling(window=20).min()).iloc[-1]
        
        # Price characteristics
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std().iloc[-1]
        vol_percentile = returns.rolling(window=20).std().rolling(window=60).rank(pct=True).iloc[-1]
        
        # Trend characteristics
        sma_10 = data['close'].rolling(window=10).mean()
        sma_50 = data['close'].rolling(window=50).mean()
        trend_strength = abs((sma_10.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1])
        
        # Regime classification
        if vol_percentile > 0.8 and tsi_std > 15:
            regime = 'high_volatility'
            confidence = vol_percentile
        elif vol_percentile < 0.2 and tsi_range < 20:
            regime = 'low_volatility'
            confidence = 1 - vol_percentile
        elif trend_strength > 0.05 and abs(tsi_mean) > 10:
            regime = 'trending'
            confidence = min(1.0, trend_strength * 10)
        elif tsi_std > 20 and tsi_range > 40:
            regime = 'choppy'
            confidence = min(1.0, tsi_std / 30)
        else:
            regime = 'normal'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': float(confidence) if not pd.isna(confidence) else 0.5,
            'tsi_mean': float(tsi_mean) if not pd.isna(tsi_mean) else 0.0,
            'tsi_std': float(tsi_std) if not pd.isna(tsi_std) else 10.0,
            'tsi_range': float(tsi_range) if not pd.isna(tsi_range) else 20.0,
            'volatility_percentile': float(vol_percentile) if not pd.isna(vol_percentile) else 0.5,
            'trend_strength': float(trend_strength) if not pd.isna(trend_strength) else 0.0
        }
    
    def _analyze_momentum_patterns(self, data: pd.DataFrame, tsi_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum patterns in TSI"""
        # Momentum cycles
        tsi_momentum = tsi_data['tsi_momentum']
        momentum_cycles = self._identify_momentum_cycles(tsi_momentum)
        
        # Acceleration patterns
        tsi_acceleration = tsi_data['tsi_acceleration']
        acceleration_patterns = self._identify_acceleration_patterns(tsi_acceleration)
        
        # Velocity analysis
        tsi_velocity = tsi_data['tsi_velocity']
        velocity_trends = self._analyze_velocity_trends(tsi_velocity)
        
        return {
            'momentum_cycles': momentum_cycles,
            'acceleration_patterns': acceleration_patterns,
            'velocity_trends': velocity_trends,
            'momentum_strength': float(abs(tsi_momentum.iloc[-1])) if len(tsi_momentum) > 0 else 0.0,
            'acceleration_strength': float(abs(tsi_acceleration.iloc[-1])) if len(tsi_acceleration) > 0 else 0.0
        }
    
    def _identify_momentum_cycles(self, momentum: pd.Series) -> Dict[str, Any]:
        """Identify momentum cycle patterns"""
        # Positive/negative momentum periods
        positive_momentum = momentum > 0
        negative_momentum = momentum < 0
        
        # Cycle lengths
        positive_cycles = []
        negative_cycles = []
        current_positive = 0
        current_negative = 0
        
        for value in positive_momentum:
            if value:
                current_positive += 1
                if current_negative > 0:
                    negative_cycles.append(current_negative)
                    current_negative = 0
            else:
                current_negative += 1
                if current_positive > 0:
                    positive_cycles.append(current_positive)
                    current_positive = 0
        
        return {
            'avg_positive_cycle': np.mean(positive_cycles) if positive_cycles else 0,
            'avg_negative_cycle': np.mean(negative_cycles) if negative_cycles else 0,
            'current_cycle_length': max(current_positive, current_negative),
            'cycle_consistency': np.std(positive_cycles + negative_cycles) if (positive_cycles + negative_cycles) else 0
        }
    
    def _identify_acceleration_patterns(self, acceleration: pd.Series) -> Dict[str, Any]:
        """Identify acceleration patterns"""
        # Acceleration phases
        accelerating = acceleration > 0
        decelerating = acceleration < 0
        
        # Pattern strength
        acceleration_strength = acceleration.rolling(window=5).mean()
        
        return {
            'current_phase': 'accelerating' if accelerating.iloc[-1] else 'decelerating',
            'acceleration_strength': float(acceleration_strength.iloc[-1]) if len(acceleration_strength) > 0 else 0.0,
            'acceleration_consistency': float(acceleration.rolling(window=10).std().iloc[-1]) if len(acceleration) > 10 else 0.0
        }
    
    def _analyze_velocity_trends(self, velocity: pd.Series) -> Dict[str, Any]:
        """Analyze velocity trend patterns"""
        # Velocity trend
        velocity_trend = velocity.rolling(window=5).mean()
        velocity_acceleration = velocity_trend.diff()
        
        return {
            'velocity_trend': float(velocity_trend.iloc[-1]) if len(velocity_trend) > 0 else 0.0,
            'velocity_acceleration': float(velocity_acceleration.iloc[-1]) if len(velocity_acceleration) > 0 else 0.0,
            'velocity_volatility': float(velocity.rolling(window=10).std().iloc[-1]) if len(velocity) > 10 else 0.0
        }
    
    def _train_ml_models(self, data: pd.DataFrame, tsi_data: pd.DataFrame, 
                        base_signals: Dict[str, Any]) -> None:
        """Train ML models for signal enhancement"""
        try:
            # Prepare features
            features = self._prepare_ml_features(data, tsi_data)
            
            if len(features) < self.ml_lookback // 2:
                return
            
            # Prepare labels
            future_returns = data['close'].pct_change(5).shift(-5)
            
            # Signal labels
            signal_labels = np.where(future_returns > 0.015, 1,
                                   np.where(future_returns < -0.015, -1, 0))
            
            # Trend strength labels
            trend_strength = abs(future_returns)
            
            # Remove NaN values
            valid_indices = ~(np.isnan(signal_labels) | np.any(np.isnan(features), axis=1) | 
                            np.isnan(trend_strength))
            
            features_clean = features[valid_indices]
            signal_labels_clean = signal_labels[valid_indices]
            trend_strength_clean = trend_strength[valid_indices]
            
            if len(features_clean) < 50:
                return
            
            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features_clean)
            features_robust = self.robust_scaler.fit_transform(features_clean)
            
            # Dimensionality reduction
            if features_scaled.shape[1] > 10:
                features_pca = self.pca.fit_transform(features_scaled)
            else:
                features_pca = features_scaled
            
            # Train models
            self.signal_classifier.fit(features_scaled, signal_labels_clean)
            self.trend_strength_regressor.fit(features_scaled, trend_strength_clean)
            self.pattern_detector.fit(features_robust, signal_labels_clean)
            self.momentum_classifier.fit(features_pca, signal_labels_clean)
            
            # Train regime clusterer
            regime_features = features_scaled[:, :min(8, features_scaled.shape[1])]
            self.regime_clusterer.fit(regime_features)
            
            self.is_trained = True
            self.feature_columns = [f'feature_{i}' for i in range(features_scaled.shape[1])]
            
        except Exception as e:
            print(f"ML training error: {e}")
    
    def _prepare_ml_features(self, data: pd.DataFrame, tsi_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models"""
        features = []
        
        # TSI features
        features.append(tsi_data['tsi'].values)
        features.append(tsi_data['tsi_signal'].values)
        features.append(tsi_data['tsi_histogram'].values)
        features.append(tsi_data['tsi_momentum'].values)
        features.append(tsi_data['tsi_acceleration'].values)
        features.append(tsi_data['tsi_velocity'].values)
        features.append(tsi_data['mtf_convergence'].values)
        
        # Multi-period TSI
        for period in self.smoothing_periods:
            if f'tsi_smooth_{period}' in tsi_data.columns:
                features.append(tsi_data[f'tsi_smooth_{period}'].values)
        
        # Volume-weighted TSI (if available)
        if 'volume_weighted_tsi' in tsi_data.columns:
            features.append(tsi_data['volume_weighted_tsi'].values)
        
        # Price features
        features.append(data['close'].pct_change().values)
        features.append(data['close'].pct_change(5).values)
        features.append(data['close'].rolling(window=10).mean().values)
        features.append(data['close'].rolling(window=20).std().values)
        
        # High-low range
        features.append(((data['high'] - data['low']) / data['close']).values)
        
        # Volume features (if available)
        if 'volume' in data.columns:
            features.append(data['volume'].rolling(window=10).mean().values)
            features.append(data['volume'].pct_change().values)
            features.append((data['volume'] / data['volume'].rolling(window=20).mean()).values)
        
        return np.column_stack(features)
    
    def _generate_ml_signals(self, data: pd.DataFrame, tsi_data: pd.DataFrame,
                            base_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML-enhanced signals"""
        if not self.is_trained:
            return {
                'ml_signal': pd.Series(0, index=data.index),
                'ml_strength': pd.Series(0.5, index=data.index),
                'ml_confidence': pd.Series(0.5, index=data.index),
                'pattern_signal': pd.Series(0, index=data.index),
                'momentum_signal': pd.Series(0, index=data.index)
            }
        
        try:
            # Prepare features
            features = self._prepare_ml_features(data, tsi_data)
            features_scaled = self.feature_scaler.transform(features)
            features_robust = self.robust_scaler.transform(features)
            
            # PCA features
            if features_scaled.shape[1] > 10:
                features_pca = self.pca.transform(features_scaled)
            else:
                features_pca = features_scaled
            
            # Generate predictions
            signal_pred = self.signal_classifier.predict(features_scaled)
            signal_proba = self.signal_classifier.predict_proba(features_scaled)
            
            trend_strength = self.trend_strength_regressor.predict(features_scaled)
            pattern_pred = self.pattern_detector.predict(features_robust)
            momentum_pred = self.momentum_classifier.predict(features_pca)
            
            # Calculate confidence
            confidence = np.max(signal_proba, axis=1)
            
            return {
                'ml_signal': pd.Series(signal_pred, index=data.index),
                'ml_strength': pd.Series(trend_strength, index=data.index),
                'ml_confidence': pd.Series(confidence, index=data.index),
                'pattern_signal': pd.Series(pattern_pred, index=data.index),
                'momentum_signal': pd.Series(momentum_pred, index=data.index)
            }
            
        except Exception as e:
            print(f"ML signal generation error: {e}")
            return {
                'ml_signal': pd.Series(0, index=data.index),
                'ml_strength': pd.Series(0.5, index=data.index),
                'ml_confidence': pd.Series(0.5, index=data.index),
                'pattern_signal': pd.Series(0, index=data.index),
                'momentum_signal': pd.Series(0, index=data.index)
            }
    
    def _analyze_signal_strength(self, data: pd.DataFrame, tsi_data: pd.DataFrame,
                                base_signals: Dict[str, Any], ml_signals: Dict[str, Any],
                                divergence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall signal strength and confidence"""
        # Combine different signal sources
        base_strength = base_signals['signal_strength']
        ml_strength = ml_signals['ml_strength']
        ml_confidence = ml_signals['ml_confidence']
        divergence_strength = divergence_analysis['divergence_strength']
        
        # Weighted combination
        combined_signal = (
            base_signals['oscillator_bias'] * 0.25 +
            ml_signals['ml_signal'] * 0.35 +
            ml_signals['pattern_signal'] * 0.2 +
            ml_signals['momentum_signal'] * 0.2
        )
        
        # Overall confidence calculation
        base_confidence = np.abs(base_strength)
        div_confidence = divergence_strength
        
        overall_confidence = (
            base_confidence * 0.3 +
            ml_confidence * 0.4 +
            div_confidence * 0.3
        )
        
        # Signal strength with confidence weighting
        signal_strength = np.abs(combined_signal) * overall_confidence
        
        return {
            'combined_signal': combined_signal,
            'overall_confidence': overall_confidence,
            'signal_strength': signal_strength,
            'base_contribution': base_confidence * 0.3,
            'ml_contribution': ml_confidence * 0.4,
            'divergence_contribution': div_confidence * 0.3
        }
    
    def _synthesize_multi_timeframe_signals(self, data: pd.DataFrame, tsi_data: pd.DataFrame,
                                          signal_analysis: Dict[str, Any],
                                          momentum_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize signals across multiple timeframes and momentum patterns"""
        combined_signal = signal_analysis['combined_signal']
        confidence = signal_analysis['overall_confidence']
        signal_strength = signal_analysis['signal_strength']
        
        # Momentum adjustment
        momentum_strength = momentum_analysis['momentum_strength']
        acceleration_strength = abs(momentum_analysis['acceleration_strength'])
        
        # Adjust signals based on momentum patterns
        momentum_adjusted_signal = combined_signal * (1 + momentum_strength * 0.2)
        momentum_adjusted_confidence = confidence * (1 + acceleration_strength * 0.1)
        
        # Generate final trading signals
        strong_buy = (momentum_adjusted_signal > 0.7) & (momentum_adjusted_confidence > self.confidence_threshold)
        buy = (momentum_adjusted_signal > 0.4) & (momentum_adjusted_confidence > self.confidence_threshold * 0.8)
        strong_sell = (momentum_adjusted_signal < -0.7) & (momentum_adjusted_confidence > self.confidence_threshold)
        sell = (momentum_adjusted_signal < -0.4) & (momentum_adjusted_confidence > self.confidence_threshold * 0.8)
        
        # Position sizing based on signal strength and confidence
        position_size = np.clip(
            signal_strength * momentum_adjusted_confidence * (1 + momentum_strength * 0.3), 
            0, 1
        )
        
        # Signal quality score
        signal_quality = momentum_adjusted_confidence * signal_strength * (1 + momentum_strength * 0.2)
        
        return {
            'final_signal': momentum_adjusted_signal,
            'strong_buy': strong_buy,
            'buy': buy,
            'strong_sell': strong_sell,
            'sell': sell,
            'position_size': position_size,
            'signal_quality': signal_quality,
            'momentum_adjustment': momentum_strength
        }
    
    def _calculate_risk_metrics(self, data: pd.DataFrame, tsi_data: pd.DataFrame,
                               synthesized_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        returns = data['close'].pct_change()
        
        # Volatility metrics
        volatility = returns.rolling(window=20).std()
        downside_vol = returns[returns < 0].rolling(window=20).std()
        
        # TSI-based risk assessment
        tsi_extreme_risk = np.where(
            abs(tsi_data['tsi']) > 50, 0.8,
            np.where(abs(tsi_data['tsi']) > 30, 0.5, 0.2)
        )
        
        # Signal consistency risk
        signal_volatility = synthesized_signals['final_signal'].rolling(window=10).std()
        signal_consistency_risk = np.clip(signal_volatility.fillna(0.5), 0, 1)
        
        # Momentum risk
        momentum_risk = np.abs(tsi_data['tsi_acceleration']).fillna(0) / 10
        momentum_risk = np.clip(momentum_risk, 0, 1)
        
        # Overall risk score
        overall_risk = (
            tsi_extreme_risk * 0.3 +
            signal_consistency_risk * 0.3 +
            momentum_risk * 0.2 +
            np.clip(volatility.fillna(0.02) / 0.05, 0, 1) * 0.2
        )
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=252, min_periods=1).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        return {
            'volatility': float(volatility.iloc[-1]) if len(volatility) > 0 else 0.02,
            'downside_volatility': float(downside_vol.iloc[-1]) if len(downside_vol) > 0 else 0.02,
            'max_drawdown': float(max_drawdown) if not pd.isna(max_drawdown) else 0.05,
            'current_drawdown': float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0,
            'tsi_risk': pd.Series(tsi_extreme_risk, index=data.index),
            'signal_consistency_risk': signal_consistency_risk,
            'momentum_risk': pd.Series(momentum_risk, index=data.index),
            'overall_risk': pd.Series(overall_risk, index=data.index)
        }
    
    def _generate_trading_recommendations(self, synthesized_signals: Dict[str, Any],
                                        risk_metrics: Dict[str, Any],
                                        regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final trading recommendations"""
        final_signal = synthesized_signals['final_signal']
        risk_score = risk_metrics['overall_risk']
        position_size = synthesized_signals['position_size']
        
        # Risk-adjusted position sizing
        risk_adjustment = 1 - risk_score
        adjusted_position = position_size * risk_adjustment
        
        # Regime-based adjustments
        regime = regime_analysis['regime']
        regime_multiplier = {
            'high_volatility': 0.6,
            'low_volatility': 1.1,
            'trending': 1.3,
            'choppy': 0.4,
            'normal': 1.0
        }.get(regime, 1.0)
        
        final_position = adjusted_position * regime_multiplier
        
        # Entry and exit conditions
        entry_signals = synthesized_signals['strong_buy'] | synthesized_signals['strong_sell']
        exit_signals = (abs(final_signal) < 0.3) | (risk_score > 0.7)
        
        # Action recommendations
        action = np.where(
            synthesized_signals['strong_buy'], 'STRONG_BUY',
            np.where(synthesized_signals['buy'], 'BUY',
                    np.where(synthesized_signals['strong_sell'], 'STRONG_SELL',
                            np.where(synthesized_signals['sell'], 'SELL', 'HOLD')))
        )
        
        return {
            'action': action,
            'position_size': final_position,
            'entry_signals': entry_signals,
            'exit_signals': exit_signals,
            'risk_level': np.where(risk_score > 0.7, 'HIGH',
                                 np.where(risk_score > 0.4, 'MEDIUM', 'LOW')),
            'regime_factor': regime_multiplier,
            'confidence_level': synthesized_signals['signal_quality'],
            'stop_loss_level': np.where(final_signal > 0, 0.98, 1.02),
            'take_profit_level': np.where(final_signal > 0, 1.03, 0.97)
        }
    
    def _generate_metadata(self, data: pd.DataFrame, tsi_data: pd.DataFrame,
                          synthesized_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        return {
            'indicator_name': 'TSIOscillator',
            'version': '1.0.0',
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period,
                'smoothing_periods': self.smoothing_periods,
                'ml_lookback': self.ml_lookback,
                'confidence_threshold': self.confidence_threshold
            },
            'data_points': len(data),
            'calculation_time': pd.Timestamp.now().isoformat(),
            'ml_model_trained': self.is_trained,
            'feature_count': len(self.feature_columns),
            'adaptive_features': {
                'adaptive_smoothing': self.adaptive_smoothing,
                'volume_weighting': self.volume_weighting,
                'regime_awareness': self.regime_awareness
            },
            'signal_distribution': {
                'strong_buy_count': int(synthesized_signals['strong_buy'].sum()),
                'buy_count': int(synthesized_signals['buy'].sum()),
                'strong_sell_count': int(synthesized_signals['strong_sell'].sum()),
                'sell_count': int(synthesized_signals['sell'].sum())
            },
            'performance_metrics': {
                'avg_signal_strength': float(synthesized_signals['signal_quality'].mean()),
                'max_signal_strength': float(synthesized_signals['signal_quality'].max()),
                'signal_consistency': float(synthesized_signals['signal_quality'].std()),
                'position_size_avg': float(synthesized_signals['position_size'].mean())
            },
            'tsi_statistics': {
                'current_tsi': float(tsi_data['tsi'].iloc[-1]) if len(tsi_data) > 0 else 0.0,
                'tsi_range': float(tsi_data['tsi'].max() - tsi_data['tsi'].min()) if len(tsi_data) > 0 else 0.0,
                'tsi_volatility': float(tsi_data['tsi'].std()) if len(tsi_data) > 0 else 0.0
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'error': error_message,
            'tsi_data': pd.DataFrame(),
            'base_signals': {},
            'ml_signals': {},
            'synthesized_signals': {},
            'recommendations': {},
            'confidence_score': 0.0,
            'signal_strength': 0.0,
            'metadata': {
                'indicator_name': 'TSIOscillator',
                'error': error_message,
                'calculation_time': pd.Timestamp.now().isoformat()
            }
        }
