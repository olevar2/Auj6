"""
Advanced Williams %R Commodity Channel Index Indicator with Machine Learning Integration

This module implements a sophisticated Williams %R CCI hybrid system with:
- Combined Williams %R and Commodity Channel Index calculations
- Machine learning pattern recognition and signal enhancement
- Advanced divergence detection and confirmation
- Multi-timeframe analysis and regime detection
- Risk assessment and confidence scoring

Part of the ASD trading platform's humanitarian mission.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore, pearsonr
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface


class WRCommodityChannelIndexIndicator(StandardIndicatorInterface):
    """
    Advanced Williams %R Commodity Channel Index Hybrid Indicator with Machine Learning
    
    Features:
    - Hybrid Williams %R and CCI calculations with adaptive parameters
    - ML-based signal classification and strength assessment
    - Advanced divergence detection with volume confirmation
    - Multi-timeframe convergence and regime analysis
    - Risk and confidence scoring with position sizing recommendations
    - Pattern recognition and anomaly detection
    """
    
    def __init__(self, 
                 wr_period: int = 14,
                 cci_period: int = 20,
                 wr_overbought: float = -20.0,
                 wr_oversold: float = -80.0,
                 cci_overbought: float = 100.0,
                 cci_oversold: float = -100.0,
                 multi_periods: List[int] = [7, 14, 21, 50],
                 ml_lookback: int = 252,
                 divergence_periods: List[int] = [10, 20, 40],
                 confidence_threshold: float = 0.65):
        """
        Initialize Advanced WR-CCI Hybrid Indicator
        
        Args:
            wr_period: Williams %R calculation period
            cci_period: CCI calculation period  
            wr_overbought: Williams %R overbought threshold
            wr_oversold: Williams %R oversold threshold
            cci_overbought: CCI overbought threshold
            cci_oversold: CCI oversold threshold
            multi_periods: Multiple periods for multi-timeframe analysis
            ml_lookback: Lookback period for ML training
            divergence_periods: Periods for divergence detection
            confidence_threshold: Minimum confidence for signal generation
        """
        super().__init__()
        
        self.wr_period = wr_period
        self.cci_period = cci_period
        self.wr_overbought = wr_overbought
        self.wr_oversold = wr_oversold
        self.cci_overbought = cci_overbought
        self.cci_oversold = cci_oversold
        self.multi_periods = multi_periods
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
        self.strength_regressor = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        self.pattern_detector = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        self.divergence_classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=1000,
            random_state=42
        )
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Clustering and dimensionality reduction
        self.regime_clusterer = KMeans(n_clusters=5, random_state=42)
        self.pca = PCA(n_components=0.95)
        
        # State tracking
        self.is_trained = False
        self.feature_columns = []
        self.last_regime = None
        
        # Advanced features
        self.adaptive_thresholds = True
        self.volume_confirmation = True
        self.regime_awareness = True
        self.anomaly_detection = True
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced WR-CCI hybrid indicator with ML integration
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing WR-CCI analysis, signals, and metadata
        """
        try:
            if len(data) < max(self.multi_periods + [self.ml_lookback // 2]):
                return self._create_error_result("Insufficient data for WR-CCI calculation")
            
            # Calculate Williams %R and CCI components
            oscillator_data = self._calculate_hybrid_oscillators(data)
            
            # Calculate adaptive thresholds
            adaptive_params = self._calculate_adaptive_thresholds(data, oscillator_data)
            
            # Generate base oscillator signals
            base_signals = self._generate_base_oscillator_signals(data, oscillator_data, adaptive_params)
            
            # Detect divergences with volume confirmation
            divergence_analysis = self._detect_hybrid_divergences(data, oscillator_data)
            
            # Perform regime and pattern analysis
            regime_analysis = self._analyze_oscillator_regime(data, oscillator_data)
            pattern_analysis = self._perform_pattern_analysis(data, oscillator_data)
            
            # Train or update ML models
            if not self.is_trained and len(data) >= self.ml_lookback:
                self._train_ml_models(data, oscillator_data, base_signals)
            
            # Generate ML-enhanced signals
            ml_signals = self._generate_ml_enhanced_signals(data, oscillator_data, base_signals)
            
            # Calculate signal strength and confidence
            signal_analysis = self._analyze_signal_strength(
                data, oscillator_data, base_signals, ml_signals, divergence_analysis
            )
            
            # Perform multi-timeframe synthesis
            synthesized_signals = self._synthesize_hybrid_signals(
                data, oscillator_data, signal_analysis, regime_analysis
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_hybrid_risk_metrics(data, oscillator_data, synthesized_signals)
            
            # Generate final recommendations
            recommendations = self._generate_hybrid_recommendations(
                synthesized_signals, risk_metrics, regime_analysis
            )
            
            return {
                'oscillator_data': oscillator_data,
                'base_signals': base_signals,
                'divergence_analysis': divergence_analysis,
                'ml_signals': ml_signals,
                'signal_analysis': signal_analysis,
                'synthesized_signals': synthesized_signals,
                'regime_analysis': regime_analysis,
                'pattern_analysis': pattern_analysis,
                'risk_metrics': risk_metrics,
                'recommendations': recommendations,
                'adaptive_parameters': adaptive_params,
                'confidence_score': float(signal_analysis.get('overall_confidence', 0.0)),
                'signal_strength': float(signal_analysis.get('signal_strength', 0.0)),
                'metadata': self._generate_metadata(data, oscillator_data, synthesized_signals)
            }
            
        except Exception as e:
            return self._create_error_result(f"WR-CCI hybrid calculation error: {str(e)}")
    
    def _calculate_hybrid_oscillators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R and CCI with multiple timeframes"""
        oscillator_data = pd.DataFrame(index=data.index)
        
        # Standard Williams %R
        oscillator_data['williams_r'] = self._calculate_williams_r(data, self.wr_period)
        
        # Standard CCI
        oscillator_data['cci'] = self._calculate_cci(data, self.cci_period)
        
        # Multi-timeframe Williams %R
        for period in self.multi_periods:
            oscillator_data[f'wr_{period}'] = self._calculate_williams_r(data, period)
            oscillator_data[f'cci_{period}'] = self._calculate_cci(data, period)
        
        # Hybrid composite oscillator
        oscillator_data['hybrid_oscillator'] = self._calculate_hybrid_composite(oscillator_data)
        
        # Smoothed versions
        oscillator_data['wr_smooth'] = oscillator_data['williams_r'].ewm(span=5).mean()
        oscillator_data['cci_smooth'] = oscillator_data['cci'].ewm(span=5).mean()
        oscillator_data['hybrid_smooth'] = oscillator_data['hybrid_oscillator'].ewm(span=5).mean()
        
        # Oscillator momentum
        oscillator_data['wr_momentum'] = oscillator_data['williams_r'].diff()
        oscillator_data['cci_momentum'] = oscillator_data['cci'].diff()
        oscillator_data['hybrid_momentum'] = oscillator_data['hybrid_oscillator'].diff()
        
        # Oscillator acceleration
        oscillator_data['wr_acceleration'] = oscillator_data['wr_momentum'].diff()
        oscillator_data['cci_acceleration'] = oscillator_data['cci_momentum'].diff()
        
        # Cross-oscillator relationships
        oscillator_data['wr_cci_ratio'] = oscillator_data['williams_r'] / (oscillator_data['cci'] / 100)
        oscillator_data['wr_cci_correlation'] = self._calculate_rolling_correlation(
            oscillator_data['williams_r'], oscillator_data['cci'], 20
        )
        
        # Volume-weighted oscillators (if volume available)
        if 'volume' in data.columns:
            oscillator_data['wr_volume_weighted'] = self._calculate_volume_weighted_wr(data)
            oscillator_data['cci_volume_weighted'] = self._calculate_volume_weighted_cci(data)
        
        return oscillator_data.fillna(0)
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
        return williams_r.fillna(0)
    
    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci.fillna(0)
    
    def _calculate_hybrid_composite(self, oscillator_data: pd.DataFrame) -> pd.Series:
        """Calculate hybrid composite oscillator combining WR and CCI"""
        # Normalize Williams %R to CCI scale
        wr_normalized = oscillator_data['williams_r'] * 2  # Scale from -100/0 to -200/0
        cci_values = oscillator_data['cci']
        
        # Weighted combination
        weights = self._calculate_dynamic_weights(oscillator_data)
        
        hybrid = (wr_normalized * weights['wr_weight'] + 
                 cci_values * weights['cci_weight'])
        
        return hybrid
    
    def _calculate_dynamic_weights(self, oscillator_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate dynamic weights for hybrid oscillator"""
        # Base weights
        base_wr_weight = 0.6
        base_cci_weight = 0.4
        
        # Adjust weights based on oscillator reliability
        wr_volatility = oscillator_data['williams_r'].rolling(window=20).std()
        cci_volatility = oscillator_data['cci'].rolling(window=20).std()
        
        # Lower volatility = higher weight
        total_vol = wr_volatility + cci_volatility
        wr_weight = base_wr_weight * (1 - wr_volatility / (total_vol + 1e-6))
        cci_weight = base_cci_weight * (1 - cci_volatility / (total_vol + 1e-6))
        
        # Normalize weights
        total_weight = wr_weight + cci_weight
        wr_weight = wr_weight / total_weight
        cci_weight = cci_weight / total_weight
        
        return {
            'wr_weight': wr_weight.fillna(base_wr_weight),
            'cci_weight': cci_weight.fillna(base_cci_weight)
        }
    
    def _calculate_volume_weighted_wr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume-weighted Williams %R"""
        volume = data['volume']
        
        # Volume-weighted high/low/close
        vw_high = (data['high'] * volume).rolling(window=self.wr_period).sum() / volume.rolling(window=self.wr_period).sum()
        vw_low = (data['low'] * volume).rolling(window=self.wr_period).sum() / volume.rolling(window=self.wr_period).sum()
        vw_close = (data['close'] * volume).rolling(window=self.wr_period).sum() / volume.rolling(window=self.wr_period).sum()
        
        # Volume-weighted Williams %R
        vw_williams_r = -100 * (vw_high - vw_close) / (vw_high - vw_low)
        return vw_williams_r.fillna(0)
    
    def _calculate_volume_weighted_cci(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume-weighted CCI"""
        volume = data['volume']
        
        # Volume-weighted typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vw_typical_price = (typical_price * volume).rolling(window=self.cci_period).sum() / volume.rolling(window=self.cci_period).sum()
        
        # Volume-weighted moving average
        vw_sma = vw_typical_price.rolling(window=self.cci_period).mean()
        
        # Volume-weighted mean deviation
        vw_mean_deviation = vw_typical_price.rolling(window=self.cci_period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        vw_cci = (vw_typical_price - vw_sma) / (0.015 * vw_mean_deviation)
        return vw_cci.fillna(0)
    
    def _calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """Calculate rolling correlation between two series"""
        correlation = pd.Series(index=series1.index, dtype=float)
        
        for i in range(window, len(series1)):
            subset1 = series1.iloc[i-window:i]
            subset2 = series2.iloc[i-window:i]
            
            if not subset1.isna().all() and not subset2.isna().all():
                corr_val, _ = pearsonr(subset1.fillna(0), subset2.fillna(0))
                correlation.iloc[i] = corr_val if not np.isnan(corr_val) else 0
        
        return correlation.fillna(0)
    
    def _calculate_adaptive_thresholds(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate adaptive thresholds based on market conditions"""
        if not self.adaptive_thresholds:
            return {
                'wr_overbought': self.wr_overbought,
                'wr_oversold': self.wr_oversold,
                'cci_overbought': self.cci_overbought,
                'cci_oversold': self.cci_oversold
            }
        
        # Market volatility for threshold adjustment
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        vol_percentile = volatility.rolling(window=60).rank(pct=True).iloc[-1] if len(volatility) > 60 else 0.5
        
        # Oscillator characteristics
        wr_volatility = oscillator_data['williams_r'].rolling(window=20).std().iloc[-1]
        cci_volatility = oscillator_data['cci'].rolling(window=20).std().iloc[-1]
        
        # Adaptive threshold adjustments
        vol_adjustment = (vol_percentile - 0.5) * 0.3
        
        # Williams %R adaptive thresholds
        wr_ob_adaptive = self.wr_overbought * (1 + vol_adjustment)
        wr_os_adaptive = self.wr_oversold * (1 - vol_adjustment)
        
        # CCI adaptive thresholds
        cci_ob_adaptive = self.cci_overbought * (1 + vol_adjustment)
        cci_os_adaptive = self.cci_oversold * (1 - vol_adjustment)
        
        return {
            'wr_overbought': np.clip(wr_ob_adaptive, -10, -30),
            'wr_oversold': np.clip(wr_os_adaptive, -70, -90),
            'cci_overbought': np.clip(cci_ob_adaptive, 80, 150),
            'cci_oversold': np.clip(cci_os_adaptive, -150, -80),
            'vol_percentile': vol_percentile,
            'wr_volatility': wr_volatility if not pd.isna(wr_volatility) else 10.0,
            'cci_volatility': cci_volatility if not pd.isna(cci_volatility) else 50.0
        }
    
    def _generate_base_oscillator_signals(self, data: pd.DataFrame, oscillator_data: pd.DataFrame,
                                        adaptive_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base oscillator signals"""
        wr = oscillator_data['williams_r']
        cci = oscillator_data['cci']
        hybrid = oscillator_data['hybrid_oscillator']
        
        # Adaptive thresholds
        wr_ob = adaptive_params['wr_overbought']
        wr_os = adaptive_params['wr_oversold']
        cci_ob = adaptive_params['cci_overbought']
        cci_os = adaptive_params['cci_oversold']
        
        # Williams %R signals
        wr_buy = (wr > wr_os) & (wr.shift(1) <= wr_os)
        wr_sell = (wr < wr_ob) & (wr.shift(1) >= wr_ob)
        
        # CCI signals
        cci_buy = (cci > cci_os) & (cci.shift(1) <= cci_os)
        cci_sell = (cci < cci_ob) & (cci.shift(1) >= cci_ob)
        
        # Hybrid signals
        hybrid_buy = (hybrid > -100) & (hybrid.shift(1) <= -100)
        hybrid_sell = (hybrid < 100) & (hybrid.shift(1) >= 100)
        
        # Cross-oscillator confirmation
        wr_cci_bullish = wr_buy & cci_buy
        wr_cci_bearish = wr_sell & cci_sell
        
        # Multi-timeframe confirmation
        mtf_bullish = self._calculate_mtf_confirmation(oscillator_data, 'buy', adaptive_params)
        mtf_bearish = self._calculate_mtf_confirmation(oscillator_data, 'sell', adaptive_params)
        
        # Momentum-based signals
        momentum_bullish = (oscillator_data['wr_momentum'] > 0) & (oscillator_data['cci_momentum'] > 0)
        momentum_bearish = (oscillator_data['wr_momentum'] < 0) & (oscillator_data['cci_momentum'] < 0)
        
        # Signal strength calculation
        wr_strength = np.abs(wr - (-50)) / 50  # Distance from midpoint
        cci_strength = np.abs(cci) / 100
        combined_strength = (wr_strength + cci_strength) / 2
        
        return {
            'wr_buy': wr_buy,
            'wr_sell': wr_sell,
            'cci_buy': cci_buy,
            'cci_sell': cci_sell,
            'hybrid_buy': hybrid_buy,
            'hybrid_sell': hybrid_sell,
            'wr_cci_bullish': wr_cci_bullish,
            'wr_cci_bearish': wr_cci_bearish,
            'mtf_bullish': mtf_bullish,
            'mtf_bearish': mtf_bearish,
            'momentum_bullish': momentum_bullish,
            'momentum_bearish': momentum_bearish,
            'signal_strength': combined_strength,
            'oscillator_bias': np.where(hybrid > 0, 1, np.where(hybrid < 0, -1, 0))
        }
    
    def _calculate_mtf_confirmation(self, oscillator_data: pd.DataFrame, signal_type: str,
                                  adaptive_params: Dict[str, Any]) -> pd.Series:
        """Calculate multi-timeframe confirmation"""
        confirmation = pd.Series(0.0, index=oscillator_data.index)
        
        for period in self.multi_periods:
            if f'wr_{period}' in oscillator_data.columns and f'cci_{period}' in oscillator_data.columns:
                wr_period = oscillator_data[f'wr_{period}']
                cci_period = oscillator_data[f'cci_{period}']
                
                if signal_type == 'buy':
                    period_signal = ((wr_period > adaptive_params['wr_oversold']) & 
                                   (cci_period > adaptive_params['cci_oversold']))
                else:  # sell
                    period_signal = ((wr_period < adaptive_params['wr_overbought']) & 
                                   (cci_period < adaptive_params['cci_overbought']))
                
                confirmation += period_signal.astype(float)
        
        # Normalize by number of timeframes
        confirmation = confirmation / len(self.multi_periods)
        return confirmation
    
    def _detect_hybrid_divergences(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect divergences between price and oscillators"""
        divergences = {}
        
        for period in self.divergence_periods:
            if len(data) < period * 2:
                continue
            
            # Price peaks and troughs
            price_peaks = self._find_peaks(data['close'], period // 2)
            price_troughs = self._find_troughs(data['close'], period // 2)
            
            # Oscillator peaks and troughs
            wr_peaks = self._find_peaks(oscillator_data['williams_r'], period // 2)
            wr_troughs = self._find_troughs(oscillator_data['williams_r'], period // 2)
            cci_peaks = self._find_peaks(oscillator_data['cci'], period // 2)
            cci_troughs = self._find_troughs(oscillator_data['cci'], period // 2)
            
            # WR divergences
            wr_bullish_div = self._detect_bullish_divergence(
                data['close'], oscillator_data['williams_r'], price_troughs, wr_troughs, period
            )
            wr_bearish_div = self._detect_bearish_divergence(
                data['close'], oscillator_data['williams_r'], price_peaks, wr_peaks, period
            )
            
            # CCI divergences
            cci_bullish_div = self._detect_bullish_divergence(
                data['close'], oscillator_data['cci'], price_troughs, cci_troughs, period
            )
            cci_bearish_div = self._detect_bearish_divergence(
                data['close'], oscillator_data['cci'], price_peaks, cci_peaks, period
            )
            
            # Volume confirmation (if available)
            if self.volume_confirmation and 'volume' in data.columns:
                volume_confirm = self._confirm_divergence_with_volume(
                    data['volume'], wr_bullish_div | cci_bullish_div, 
                    wr_bearish_div | cci_bearish_div, period
                )
                wr_bullish_div = wr_bullish_div & volume_confirm['bullish']
                wr_bearish_div = wr_bearish_div & volume_confirm['bearish']
                cci_bullish_div = cci_bullish_div & volume_confirm['bullish']
                cci_bearish_div = cci_bearish_div & volume_confirm['bearish']
            
            divergences[f'wr_bullish_div_{period}'] = wr_bullish_div
            divergences[f'wr_bearish_div_{period}'] = wr_bearish_div
            divergences[f'cci_bullish_div_{period}'] = cci_bullish_div
            divergences[f'cci_bearish_div_{period}'] = cci_bearish_div
        
        # Composite divergence signals
        bullish_composite = pd.Series(False, index=data.index)
        bearish_composite = pd.Series(False, index=data.index)
        
        for period in self.divergence_periods:
            bullish_composite |= (divergences.get(f'wr_bullish_div_{period}', pd.Series(False, index=data.index)) |
                                divergences.get(f'cci_bullish_div_{period}', pd.Series(False, index=data.index)))
            bearish_composite |= (divergences.get(f'wr_bearish_div_{period}', pd.Series(False, index=data.index)) |
                                divergences.get(f'cci_bearish_div_{period}', pd.Series(False, index=data.index)))
        
        # Divergence strength
        divergence_strength = self._calculate_divergence_strength(
            data, oscillator_data, bullish_composite, bearish_composite
        )
        
        divergences.update({
            'bullish_composite': bullish_composite,
            'bearish_composite': bearish_composite,
            'divergence_strength': divergence_strength
        })
        
        return divergences
    
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
    
    def _detect_bullish_divergence(self, prices: pd.Series, oscillator: pd.Series,
                                  price_troughs: pd.Series, osc_troughs: pd.Series, period: int) -> pd.Series:
        """Detect bullish divergence patterns"""
        bullish_div = pd.Series(False, index=prices.index)
        
        for i in range(period, len(prices)):
            if price_troughs[i] and osc_troughs[i]:
                # Look for previous trough
                for j in range(max(0, i - period), i):
                    if price_troughs[j] and osc_troughs[j]:
                        # Check divergence condition
                        if (prices.iloc[i] < prices.iloc[j] and 
                            oscillator.iloc[i] > oscillator.iloc[j] and
                            abs(i - j) >= period // 2):
                            bullish_div.iloc[i] = True
                            break
        
        return bullish_div
    
    def _detect_bearish_divergence(self, prices: pd.Series, oscillator: pd.Series,
                                  price_peaks: pd.Series, osc_peaks: pd.Series, period: int) -> pd.Series:
        """Detect bearish divergence patterns"""
        bearish_div = pd.Series(False, index=prices.index)
        
        for i in range(period, len(prices)):
            if price_peaks[i] and osc_peaks[i]:
                # Look for previous peak
                for j in range(max(0, i - period), i):
                    if price_peaks[j] and osc_peaks[j]:
                        # Check divergence condition
                        if (prices.iloc[i] > prices.iloc[j] and 
                            oscillator.iloc[i] < oscillator.iloc[j] and
                            abs(i - j) >= period // 2):
                            bearish_div.iloc[i] = True
                            break
        
        return bearish_div
    
    def _confirm_divergence_with_volume(self, volume: pd.Series, bullish_div: pd.Series,
                                       bearish_div: pd.Series, period: int) -> Dict[str, pd.Series]:
        """Confirm divergences with volume analysis"""
        avg_volume = volume.rolling(window=period).mean()
        
        # Volume confirmation for bullish divergence
        bullish_vol_confirm = pd.Series(False, index=volume.index)
        for i in range(len(bullish_div)):
            if bullish_div.iloc[i]:
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
    
    def _calculate_divergence_strength(self, data: pd.DataFrame, oscillator_data: pd.DataFrame,
                                     bullish_div: pd.Series, bearish_div: pd.Series) -> pd.Series:
        """Calculate strength of divergence signals"""
        strength = pd.Series(0.0, index=data.index)
        
        price_momentum = data['close'].pct_change(5)
        wr_momentum = oscillator_data['wr_momentum']
        cci_momentum = oscillator_data['cci_momentum']
        
        for i in range(len(data)):
            if bullish_div.iloc[i]:
                price_change = price_momentum.iloc[i] if not pd.isna(price_momentum.iloc[i]) else 0
                osc_change = (wr_momentum.iloc[i] + cci_momentum.iloc[i]) / 2
                if price_change < 0 and osc_change > 0:
                    strength.iloc[i] = min(1.0, abs(price_change) + abs(osc_change))
            
            elif bearish_div.iloc[i]:
                price_change = price_momentum.iloc[i] if not pd.isna(price_momentum.iloc[i]) else 0
                osc_change = (wr_momentum.iloc[i] + cci_momentum.iloc[i]) / 2
                if price_change > 0 and osc_change < 0:
                    strength.iloc[i] = min(1.0, abs(price_change) + abs(osc_change))
        
        return strength
    
    def _analyze_oscillator_regime(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current oscillator regime"""
        if not self.regime_awareness:
            return {'regime': 'normal', 'confidence': 0.5}
        
        # Oscillator characteristics
        wr_mean = oscillator_data['williams_r'].rolling(window=20).mean().iloc[-1]
        wr_std = oscillator_data['williams_r'].rolling(window=20).std().iloc[-1]
        cci_mean = oscillator_data['cci'].rolling(window=20).mean().iloc[-1]
        cci_std = oscillator_data['cci'].rolling(window=20).std().iloc[-1]
        
        # Correlation between oscillators
        correlation = oscillator_data['wr_cci_correlation'].iloc[-1]
        
        # Market volatility
        returns = data['close'].pct_change()
        vol_percentile = returns.rolling(window=20).std().rolling(window=60).rank(pct=True).iloc[-1]
        
        # Regime classification
        if vol_percentile > 0.8 and wr_std > 15 and cci_std > 60:
            regime = 'high_volatility'
            confidence = vol_percentile
        elif vol_percentile < 0.2 and wr_std < 5 and cci_std < 20:
            regime = 'low_volatility'
            confidence = 1 - vol_percentile
        elif abs(correlation) > 0.8:
            regime = 'trending'
            confidence = abs(correlation)
        elif abs(correlation) < 0.3:
            regime = 'choppy'
            confidence = 1 - abs(correlation)
        else:
            regime = 'normal'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': float(confidence) if not pd.isna(confidence) else 0.5,
            'wr_mean': float(wr_mean) if not pd.isna(wr_mean) else -50.0,
            'wr_std': float(wr_std) if not pd.isna(wr_std) else 10.0,
            'cci_mean': float(cci_mean) if not pd.isna(cci_mean) else 0.0,
            'cci_std': float(cci_std) if not pd.isna(cci_std) else 50.0,
            'correlation': float(correlation) if not pd.isna(correlation) else 0.0
        }
    
    def _perform_pattern_analysis(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform pattern analysis on oscillator data"""
        patterns = {}
        
        # Overbought/oversold extremes
        patterns['wr_extreme_oversold'] = oscillator_data['williams_r'] < -95
        patterns['wr_extreme_overbought'] = oscillator_data['williams_r'] > -5
        patterns['cci_extreme_oversold'] = oscillator_data['cci'] < -200
        patterns['cci_extreme_overbought'] = oscillator_data['cci'] > 200
        
        # Double bottom/top patterns
        patterns['wr_double_bottom'] = self._detect_double_bottom(oscillator_data['williams_r'])
        patterns['wr_double_top'] = self._detect_double_top(oscillator_data['williams_r'])
        patterns['cci_double_bottom'] = self._detect_double_bottom(oscillator_data['cci'])
        patterns['cci_double_top'] = self._detect_double_top(oscillator_data['cci'])
        
        # Oscillator consolidation patterns
        patterns['consolidation'] = self._detect_consolidation_pattern(oscillator_data)
        
        return patterns
    
    def _detect_double_bottom(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Detect double bottom patterns"""
        double_bottom = pd.Series(False, index=series.index)
        
        for i in range(window * 2, len(series)):
            recent_min = series.iloc[i-window:i].min()
            previous_min = series.iloc[i-window*2:i-window].min()
            
            # Check if current value is near recent minimum and similar to previous minimum
            if (abs(series.iloc[i] - recent_min) < abs(recent_min) * 0.05 and
                abs(recent_min - previous_min) < abs(recent_min) * 0.1):
                double_bottom.iloc[i] = True
        
        return double_bottom
    
    def _detect_double_top(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Detect double top patterns"""
        double_top = pd.Series(False, index=series.index)
        
        for i in range(window * 2, len(series)):
            recent_max = series.iloc[i-window:i].max()
            previous_max = series.iloc[i-window*2:i-window].max()
            
            # Check if current value is near recent maximum and similar to previous maximum
            if (abs(series.iloc[i] - recent_max) < abs(recent_max) * 0.05 and
                abs(recent_max - previous_max) < abs(recent_max) * 0.1):
                double_top.iloc[i] = True
        
        return double_top
    
    def _detect_consolidation_pattern(self, oscillator_data: pd.DataFrame, window: int = 15) -> pd.Series:
        """Detect consolidation patterns in oscillators"""
        wr_range = (oscillator_data['williams_r'].rolling(window=window).max() - 
                   oscillator_data['williams_r'].rolling(window=window).min())
        cci_range = (oscillator_data['cci'].rolling(window=window).max() - 
                    oscillator_data['cci'].rolling(window=window).min())
        
        # Consolidation when range is smaller than historical average
        wr_avg_range = wr_range.rolling(window=50).mean()
        cci_avg_range = cci_range.rolling(window=50).mean()
        
        consolidation = ((wr_range < wr_avg_range * 0.7) & 
                        (cci_range < cci_avg_range * 0.7))
        
        return consolidation
    
    def _train_ml_models(self, data: pd.DataFrame, oscillator_data: pd.DataFrame,
                        base_signals: Dict[str, Any]) -> None:
        """Train ML models for signal enhancement"""
        try:
            # Prepare features
            features = self._prepare_ml_features(data, oscillator_data)
            
            if len(features) < self.ml_lookback // 2:
                return
            
            # Prepare labels
            future_returns = data['close'].pct_change(5).shift(-5)
            
            # Signal labels
            signal_labels = np.where(future_returns > 0.015, 1,
                                   np.where(future_returns < -0.015, -1, 0))
            
            # Strength labels
            strength_labels = abs(future_returns)
            
            # Pattern labels
            pattern_labels = np.where(abs(future_returns) > 0.02, 1, 0)
            
            # Remove NaN values
            valid_indices = ~(np.isnan(signal_labels) | np.any(np.isnan(features), axis=1) |
                            np.isnan(strength_labels))
            
            features_clean = features[valid_indices]
            signal_labels_clean = signal_labels[valid_indices]
            strength_labels_clean = strength_labels[valid_indices]
            pattern_labels_clean = pattern_labels[valid_indices]
            
            if len(features_clean) < 50:
                return
            
            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features_clean)
            features_robust = self.robust_scaler.fit_transform(features_clean)
            
            # Train models
            self.signal_classifier.fit(features_scaled, signal_labels_clean)
            self.strength_regressor.fit(features_scaled, strength_labels_clean)
            self.pattern_detector.fit(features_robust, pattern_labels_clean)
            self.divergence_classifier.fit(features_scaled, signal_labels_clean)
            
            self.is_trained = True
            self.feature_columns = [f'feature_{i}' for i in range(features_scaled.shape[1])]
            
        except Exception as e:
            print(f"ML training error: {e}")
    
    def _prepare_ml_features(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models"""
        features = []
        
        # Oscillator features
        features.append(oscillator_data['williams_r'].values)
        features.append(oscillator_data['cci'].values)
        features.append(oscillator_data['hybrid_oscillator'].values)
        features.append(oscillator_data['wr_momentum'].values)
        features.append(oscillator_data['cci_momentum'].values)
        features.append(oscillator_data['wr_acceleration'].values)
        features.append(oscillator_data['cci_acceleration'].values)
        features.append(oscillator_data['wr_cci_ratio'].values)
        features.append(oscillator_data['wr_cci_correlation'].values)
        
        # Multi-period oscillators
        for period in self.multi_periods[:3]:  # Limit to prevent overfitting
            if f'wr_{period}' in oscillator_data.columns:
                features.append(oscillator_data[f'wr_{period}'].values)
            if f'cci_{period}' in oscillator_data.columns:
                features.append(oscillator_data[f'cci_{period}'].values)
        
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
            
            # Volume-weighted oscillators
            if 'wr_volume_weighted' in oscillator_data.columns:
                features.append(oscillator_data['wr_volume_weighted'].values)
            if 'cci_volume_weighted' in oscillator_data.columns:
                features.append(oscillator_data['cci_volume_weighted'].values)
        
        return np.column_stack(features)
    
    def _generate_ml_enhanced_signals(self, data: pd.DataFrame, oscillator_data: pd.DataFrame,
                                    base_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML-enhanced signals"""
        if not self.is_trained:
            return {
                'ml_signal': pd.Series(0, index=data.index),
                'ml_strength': pd.Series(0.5, index=data.index),
                'ml_confidence': pd.Series(0.5, index=data.index),
                'pattern_signal': pd.Series(0, index=data.index),
                'divergence_signal': pd.Series(0, index=data.index)
            }
        
        try:
            # Prepare features
            features = self._prepare_ml_features(data, oscillator_data)
            features_scaled = self.feature_scaler.transform(features)
            features_robust = self.robust_scaler.transform(features)
            
            # Generate predictions
            signal_pred = self.signal_classifier.predict(features_scaled)
            signal_proba = self.signal_classifier.predict_proba(features_scaled)
            
            strength_pred = self.strength_regressor.predict(features_scaled)
            pattern_pred = self.pattern_detector.predict(features_robust)
            divergence_pred = self.divergence_classifier.predict(features_scaled)
            
            # Calculate confidence
            confidence = np.max(signal_proba, axis=1)
            
            return {
                'ml_signal': pd.Series(signal_pred, index=data.index),
                'ml_strength': pd.Series(strength_pred, index=data.index),
                'ml_confidence': pd.Series(confidence, index=data.index),
                'pattern_signal': pd.Series(pattern_pred, index=data.index),
                'divergence_signal': pd.Series(divergence_pred, index=data.index)
            }
            
        except Exception as e:
            print(f"ML signal generation error: {e}")
            return {
                'ml_signal': pd.Series(0, index=data.index),
                'ml_strength': pd.Series(0.5, index=data.index),
                'ml_confidence': pd.Series(0.5, index=data.index),
                'pattern_signal': pd.Series(0, index=data.index),
                'divergence_signal': pd.Series(0, index=data.index)
            }
    
    def _analyze_signal_strength(self, data: pd.DataFrame, oscillator_data: pd.DataFrame,
                                base_signals: Dict[str, Any], ml_signals: Dict[str, Any],
                                divergence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall signal strength and confidence"""
        # Combine signal sources
        base_strength = base_signals['signal_strength']
        ml_confidence = ml_signals['ml_confidence']
        ml_strength = ml_signals['ml_strength']
        divergence_strength = divergence_analysis['divergence_strength']
        
        # Signal combination
        oscillator_signal = base_signals['oscillator_bias']
        ml_signal = ml_signals['ml_signal']
        pattern_signal = ml_signals['pattern_signal']
        
        combined_signal = (
            oscillator_signal * 0.3 +
            ml_signal * 0.4 +
            pattern_signal * 0.3
        )
        
        # Overall confidence calculation
        base_confidence = base_strength
        div_confidence = divergence_strength
        
        overall_confidence = (
            base_confidence * 0.25 +
            ml_confidence * 0.45 +
            div_confidence * 0.3
        )
        
        # Signal strength with confidence weighting
        signal_strength = np.abs(combined_signal) * overall_confidence * (1 + ml_strength)
        
        return {
            'combined_signal': pd.Series(combined_signal, index=data.index),
            'overall_confidence': overall_confidence,
            'signal_strength': signal_strength,
            'base_contribution': base_confidence * 0.25,
            'ml_contribution': ml_confidence * 0.45,
            'divergence_contribution': div_confidence * 0.3
        }
    
    def _synthesize_hybrid_signals(self, data: pd.DataFrame, oscillator_data: pd.DataFrame,
                                 signal_analysis: Dict[str, Any], regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize hybrid WR-CCI signals"""
        combined_signal = signal_analysis['combined_signal']
        confidence = signal_analysis['overall_confidence']
        signal_strength = signal_analysis['signal_strength']
        
        # Regime adjustment
        regime_confidence = regime_analysis.get('confidence', 0.5)
        regime_adjusted_signal = combined_signal * (0.5 + regime_confidence * 0.5)
        
        # Generate final trading signals
        strong_buy = (regime_adjusted_signal > 0.7) & (confidence > self.confidence_threshold)
        buy = (regime_adjusted_signal > 0.4) & (confidence > self.confidence_threshold * 0.8)
        strong_sell = (regime_adjusted_signal < -0.7) & (confidence > self.confidence_threshold)
        sell = (regime_adjusted_signal < -0.4) & (confidence > self.confidence_threshold * 0.8)
        
        # Position sizing
        position_size = np.clip(signal_strength * confidence * regime_confidence, 0, 1)
        
        # Signal quality
        signal_quality = confidence * signal_strength * regime_confidence
        
        return {
            'final_signal': regime_adjusted_signal,
            'strong_buy': strong_buy,
            'buy': buy,
            'strong_sell': strong_sell,
            'sell': sell,
            'position_size': position_size,
            'signal_quality': signal_quality,
            'regime_adjustment': regime_confidence
        }
    
    def _calculate_hybrid_risk_metrics(self, data: pd.DataFrame, oscillator_data: pd.DataFrame,
                                     synthesized_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate hybrid-specific risk metrics"""
        # Oscillator risk
        wr_extreme_risk = np.where(
            (oscillator_data['williams_r'] > -5) | (oscillator_data['williams_r'] < -95), 0.8, 0.3
        )
        cci_extreme_risk = np.where(
            abs(oscillator_data['cci']) > 200, 0.8, 0.3
        )
        
        # Signal consistency risk
        signal_volatility = synthesized_signals['final_signal'].rolling(window=10).std()
        signal_consistency_risk = np.clip(signal_volatility.fillna(0.5), 0, 1)
        
        # Correlation risk
        correlation_risk = 1 - abs(oscillator_data['wr_cci_correlation']).fillna(0.5)
        
        # Overall risk
        overall_risk = (
            wr_extreme_risk * 0.25 +
            cci_extreme_risk * 0.25 +
            signal_consistency_risk * 0.25 +
            correlation_risk * 0.25
        )
        
        return {
            'wr_extreme_risk': pd.Series(wr_extreme_risk, index=data.index),
            'cci_extreme_risk': pd.Series(cci_extreme_risk, index=data.index),
            'signal_consistency_risk': signal_consistency_risk,
            'correlation_risk': correlation_risk,
            'overall_risk': pd.Series(overall_risk, index=data.index)
        }
    
    def _generate_hybrid_recommendations(self, synthesized_signals: Dict[str, Any],
                                       risk_metrics: Dict[str, Any], regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final hybrid trading recommendations"""
        final_signal = synthesized_signals['final_signal']
        risk_score = risk_metrics['overall_risk']
        position_size = synthesized_signals['position_size']
        
        # Risk-adjusted position sizing
        risk_adjustment = 1 - risk_score
        adjusted_position = position_size * risk_adjustment
        
        # Regime-based adjustments
        regime = regime_analysis.get('regime', 'normal')
        regime_multipliers = {
            'high_volatility': 0.6,
            'low_volatility': 1.1,
            'trending': 1.3,
            'choppy': 0.5,
            'normal': 1.0
        }
        
        regime_multiplier = regime_multipliers.get(regime, 1.0)
        final_position = np.clip(adjusted_position * regime_multiplier, 0, 1)
        
        # Generate actions
        action = np.where(
            synthesized_signals['strong_buy'], 'STRONG_BUY',
            np.where(synthesized_signals['buy'], 'BUY',
                    np.where(synthesized_signals['strong_sell'], 'STRONG_SELL',
                            np.where(synthesized_signals['sell'], 'SELL', 'HOLD')))
        )
        
        # Entry and exit conditions
        entry_signals = synthesized_signals['strong_buy'] | synthesized_signals['strong_sell']
        exit_signals = (abs(final_signal) < 0.3) | (risk_score > 0.7)
        
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
            'take_profit_level': np.where(final_signal > 0, 1.04, 0.96)
        }
    
    def _generate_metadata(self, data: pd.DataFrame, oscillator_data: pd.DataFrame,
                          synthesized_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        return {
            'indicator_name': 'WRCommodityChannelIndexIndicator',
            'version': '1.0.0',
            'parameters': {
                'wr_period': self.wr_period,
                'cci_period': self.cci_period,
                'wr_thresholds': [self.wr_overbought, self.wr_oversold],
                'cci_thresholds': [self.cci_overbought, self.cci_oversold],
                'multi_periods': self.multi_periods,
                'ml_lookback': self.ml_lookback,
                'confidence_threshold': self.confidence_threshold
            },
            'data_points': len(data),
            'calculation_time': pd.Timestamp.now().isoformat(),
            'ml_model_trained': self.is_trained,
            'feature_count': len(self.feature_columns),
            'advanced_features': {
                'adaptive_thresholds': self.adaptive_thresholds,
                'volume_confirmation': self.volume_confirmation,
                'regime_awareness': self.regime_awareness,
                'anomaly_detection': self.anomaly_detection
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
            'oscillator_statistics': {
                'current_wr': float(oscillator_data['williams_r'].iloc[-1]) if len(oscillator_data) > 0 else -50.0,
                'current_cci': float(oscillator_data['cci'].iloc[-1]) if len(oscillator_data) > 0 else 0.0,
                'wr_range': float(oscillator_data['williams_r'].max() - oscillator_data['williams_r'].min()) if len(oscillator_data) > 0 else 100.0,
                'cci_range': float(oscillator_data['cci'].max() - oscillator_data['cci'].min()) if len(oscillator_data) > 0 else 400.0,
                'correlation': float(oscillator_data['wr_cci_correlation'].iloc[-1]) if len(oscillator_data) > 0 else 0.0
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'error': error_message,
            'oscillator_data': pd.DataFrame(),
            'base_signals': {},
            'ml_signals': {},
            'synthesized_signals': {},
            'recommendations': {},
            'confidence_score': 0.0,
            'signal_strength': 0.0,
            'metadata': {
                'indicator_name': 'WRCommodityChannelIndexIndicator',
                'error': error_message,
                'calculation_time': pd.Timestamp.now().isoformat()
            }
        }
