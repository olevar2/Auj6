"""
Institutional Flow Signal Indicator - AI Enhanced Category
==========================================================

Advanced AI-enhanced institutional flow detection system with smart money tracking,
order flow analysis, and institutional behavior pattern recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from scipy import signal, stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class InstitutionalFlowSignalIndicator(StandardIndicatorInterface):
    """
    AI-Enhanced Institutional Flow Signal with advanced features.
    
    Features:
    - Smart money flow detection using advanced algorithms
    - Large order identification and tracking
    - Institutional trading pattern recognition
    - Volume profile analysis for institutional activity
    - Time-based institutional behavior modeling
    - Block trade detection and impact analysis
    - Dark pool activity estimation
    - Market maker behavior identification
    - Liquidity provision pattern analysis
    - Cross-market institutional correlation
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'large_order_threshold': 2.0,    # Standard deviations above mean volume
            'institutional_volume_min': 1000000,  # Minimum volume for institutional consideration
            'block_trade_multiplier': 3.0,   # Multiplier for block trade detection
            'time_window': 20,               # Time window for analysis
            'smart_money_window': 50,        # Window for smart money analysis
            'pattern_detection_window': 100, # Window for pattern detection
            'volume_profile_bins': 20,       # Number of bins for volume profile
            'outlier_detection_threshold': 0.1,  # Outlier detection sensitivity
            'correlation_window': 30,        # Cross-market correlation window
            'activity_threshold': 0.7,       # Threshold for significant activity
            'confidence_threshold': 0.6,     # Minimum confidence for signals
            'dark_pool_estimation': True,    # Enable dark pool estimation
            'market_maker_detection': True,  # Enable market maker detection
            'cross_market_analysis': True,   # Enable cross-market analysis
            'adaptive_thresholds': True,     # Enable adaptive thresholds
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("InstitutionalFlowSignalIndicator", default_params)
        
        # Initialize ML models
        self.flow_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.outlier_detector = IsolationForest(
            contamination=self.parameters['outlier_detection_threshold'],
            random_state=42
        )
        self.pattern_clusterer = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.is_trained = False
        
        # State tracking
        self.institutional_trades = []
        self.flow_history = []
        self.pattern_cache = {}
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["high", "low", "close", "volume"],
            min_periods=self.parameters['pattern_detection_window']
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced institutional flow signals with AI enhancements."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Extract data arrays
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate basic flow metrics
            flow_metrics = self._calculate_flow_metrics(high, low, close, volume)
            
            # Detect large orders and block trades
            large_orders = self._detect_large_orders(high, low, close, volume)
            
            # Analyze volume profile for institutional activity
            volume_profile = self._analyze_volume_profile(high, low, close, volume)
            
            # Smart money flow analysis
            smart_money = self._analyze_smart_money_flow(close, volume, flow_metrics)
            
            # Institutional pattern recognition
            patterns = self._detect_institutional_patterns(high, low, close, volume)
            
            # Dark pool activity estimation
            dark_pool_activity = {}
            if self.parameters['dark_pool_estimation']:
                dark_pool_activity = self._estimate_dark_pool_activity(
                    high, low, close, volume
                )
            
            # Market maker behavior detection
            market_maker_activity = {}
            if self.parameters['market_maker_detection']:
                market_maker_activity = self._detect_market_maker_behavior(
                    high, low, close, volume
                )
            
            # Cross-market correlation analysis
            cross_market_signals = {}
            if self.parameters['cross_market_analysis']:
                cross_market_signals = self._analyze_cross_market_correlation(data)
            
            # Machine learning classification
            ml_signals = self._calculate_ml_signals(
                flow_metrics, large_orders, volume_profile, smart_money
            )
            
            # Adaptive threshold calculation
            adaptive_thresholds = {}
            if self.parameters['adaptive_thresholds']:
                adaptive_thresholds = self._calculate_adaptive_thresholds(
                    flow_metrics, volume
                )
            
            # Generate comprehensive signals
            signals = self._generate_institutional_signals(
                flow_metrics, large_orders, smart_money, patterns,
                ml_signals, adaptive_thresholds
            )
            
            # Calculate confidence and strength
            confidence_metrics = self._calculate_confidence_metrics(
                signals, flow_metrics, smart_money, patterns
            )
            
            # Risk assessment
            risk_assessment = self._assess_institutional_risk(
                signals, flow_metrics, volume_profile
            )
            
            return {
                'flow_metrics': flow_metrics,
                'large_orders': large_orders,
                'volume_profile': volume_profile,
                'smart_money': smart_money,
                'patterns': patterns,
                'dark_pool_activity': dark_pool_activity,
                'market_maker_activity': market_maker_activity,
                'cross_market_signals': cross_market_signals,
                'ml_signals': ml_signals,
                'adaptive_thresholds': adaptive_thresholds,
                'signals': signals,
                'confidence_metrics': confidence_metrics,
                'risk_assessment': risk_assessment,
                'institutional_sentiment': signals.get('sentiment', 'neutral'),
                'flow_strength': signals.get('strength', 0.0),
                'smart_money_direction': smart_money.get('direction', 'neutral')
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _calculate_flow_metrics(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Calculate basic institutional flow metrics."""
        # Price-Volume Trend (PVT)
        price_change = np.diff(close, prepend=close[0])
        price_change_pct = price_change / np.where(close[:-1] == 0, 1, close[:-1])
        pvt = np.cumsum(price_change_pct * volume)
        
        # On Balance Volume (OBV)
        obv_changes = np.where(price_change > 0, volume, 
                              np.where(price_change < 0, -volume, 0))
        obv = np.cumsum(obv_changes)
        
        # Chaikin Money Flow
        money_flow_multiplier = ((close - low) - (high - close)) / np.where((high - low) == 0, 1, (high - low))
        money_flow_volume = money_flow_multiplier * volume
        
        window = self.parameters['time_window']
        cmf = []
        for i in range(len(money_flow_volume)):
            start_idx = max(0, i - window + 1)
            period_mfv = np.sum(money_flow_volume[start_idx:i+1])
            period_vol = np.sum(volume[start_idx:i+1])
            cmf.append(period_mfv / period_vol if period_vol > 0 else 0)
        
        cmf = np.array(cmf)
        
        # Volume Rate of Change
        vol_roc = np.zeros_like(volume)
        for i in range(window, len(volume)):
            if volume[i-window] > 0:
                vol_roc[i] = (volume[i] - volume[i-window]) / volume[i-window] * 100
        
        # Accumulation/Distribution Line
        ad_line = np.cumsum(money_flow_volume)
        
        # Calculate flow direction and strength
        current_flow_direction = 'bullish' if cmf[-1] > 0 else 'bearish' if cmf[-1] < 0 else 'neutral'
        flow_strength = abs(cmf[-1])
        
        # Volume trend analysis
        recent_volume_avg = np.mean(volume[-window:]) if len(volume) >= window else np.mean(volume)
        volume_trend = 'increasing' if vol_roc[-1] > 5 else 'decreasing' if vol_roc[-1] < -5 else 'stable'
        
        return {
            'pvt': pvt[-1] if len(pvt) > 0 else 0.0,
            'obv': obv[-1] if len(obv) > 0 else 0.0,
            'cmf': cmf[-1] if len(cmf) > 0 else 0.0,
            'ad_line': ad_line[-1] if len(ad_line) > 0 else 0.0,
            'volume_roc': vol_roc[-1] if len(vol_roc) > 0 else 0.0,
            'flow_direction': current_flow_direction,
            'flow_strength': flow_strength,
            'volume_trend': volume_trend,
            'recent_volume_avg': recent_volume_avg,
            'pvt_series': pvt,
            'obv_series': obv,
            'cmf_series': cmf,
            'ad_series': ad_line
        }
    
    def _detect_large_orders(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Detect large orders and block trades."""
        # Calculate volume statistics
        volume_mean = np.mean(volume)
        volume_std = np.std(volume)
        
        # Adaptive thresholds
        large_order_threshold = volume_mean + (self.parameters['large_order_threshold'] * volume_std)
        block_trade_threshold = volume_mean + (self.parameters['block_trade_multiplier'] * volume_std)
        
        # Detect large orders
        large_order_indices = np.where(volume > large_order_threshold)[0]
        block_trade_indices = np.where(volume > block_trade_threshold)[0]
        
        # Analyze order characteristics
        large_orders = []
        for idx in large_order_indices[-20:]:  # Last 20 large orders
            if idx < len(close):
                order_info = {
                    'index': idx,
                    'volume': volume[idx],
                    'price': close[idx],
                    'price_impact': self._calculate_price_impact(high, low, close, idx),
                    'urgency': self._calculate_order_urgency(volume, idx),
                    'type': 'block_trade' if volume[idx] > block_trade_threshold else 'large_order'
                }
                large_orders.append(order_info)
        
        # Block trade analysis
        block_trades = []
        for idx in block_trade_indices[-10:]:  # Last 10 block trades
            if idx < len(close):
                block_info = {
                    'index': idx,
                    'volume': volume[idx],
                    'price': close[idx],
                    'market_impact': self._calculate_market_impact(high, low, close, volume, idx),
                    'execution_quality': self._assess_execution_quality(high, low, close, idx),
                    'institutional_probability': min(volume[idx] / self.parameters['institutional_volume_min'], 1.0)
                }
                block_trades.append(block_info)
        
        # Summary statistics
        recent_large_order_count = len([o for o in large_orders if o['index'] >= len(volume) - 20])
        recent_block_trade_count = len([b for b in block_trades if b['index'] >= len(volume) - 10])
        
        # Average characteristics
        avg_large_order_volume = np.mean([o['volume'] for o in large_orders]) if large_orders else 0
        avg_block_trade_impact = np.mean([b['market_impact'] for b in block_trades]) if block_trades else 0
        
        return {
            'large_orders': large_orders,
            'block_trades': block_trades,
            'large_order_threshold': large_order_threshold,
            'block_trade_threshold': block_trade_threshold,
            'recent_large_order_count': recent_large_order_count,
            'recent_block_trade_count': recent_block_trade_count,
            'avg_large_order_volume': avg_large_order_volume,
            'avg_block_trade_impact': avg_block_trade_impact,
            'institutional_activity_level': min((recent_large_order_count + recent_block_trade_count * 2) / 10.0, 1.0)
        }
    
    def _calculate_price_impact(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, idx: int) -> float:
        """Calculate price impact of a trade."""
        if idx == 0 or idx >= len(close):
            return 0.0
        
        # Simple price impact calculation
        pre_trade_price = close[idx-1] if idx > 0 else close[idx]
        post_trade_price = close[idx]
        
        if pre_trade_price != 0:
            impact = abs(post_trade_price - pre_trade_price) / pre_trade_price
        else:
            impact = 0.0
        
        return impact
    
    def _calculate_order_urgency(self, volume: np.ndarray, idx: int) -> float:
        """Calculate order urgency based on volume pattern."""
        if idx < 5 or idx >= len(volume):
            return 0.5
        
        # Compare current volume to recent average
        recent_avg = np.mean(volume[max(0, idx-5):idx])
        current_vol = volume[idx]
        
        urgency = min(current_vol / (recent_avg + 1e-10), 3.0) / 3.0
        return urgency
    
    def _calculate_market_impact(self, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, volume: np.ndarray, idx: int) -> float:
        """Calculate market impact of a block trade."""
        if idx < 3 or idx >= len(close) - 3:
            return 0.0
        
        # Pre-trade price range
        pre_range = np.mean(high[idx-3:idx]) - np.mean(low[idx-3:idx])
        
        # Post-trade price movement
        pre_price = close[idx-1] if idx > 0 else close[idx]
        post_prices = close[idx:min(idx+3, len(close))]
        
        if len(post_prices) > 0 and pre_price != 0:
            max_post_move = max(abs(p - pre_price) for p in post_prices)
            impact = max_post_move / pre_price if pre_range > 0 else 0.0
        else:
            impact = 0.0
        
        return impact
    
    def _assess_execution_quality(self, high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray, idx: int) -> float:
        """Assess execution quality of a trade."""
        if idx >= len(close):
            return 0.5
        
        # Compare execution price to VWAP-like measure
        window = 5
        start_idx = max(0, idx - window)
        end_idx = min(idx + window, len(close))
        
        avg_price = np.mean(close[start_idx:end_idx])
        execution_price = close[idx]
        
        if avg_price != 0:
            quality = 1.0 - abs(execution_price - avg_price) / avg_price
        else:
            quality = 0.5
        
        return max(0.0, min(1.0, quality))
    
    def _analyze_volume_profile(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze volume profile for institutional activity."""
        window = self.parameters['time_window']
        
        if len(close) < window:
            return {'poc': 0.0, 'value_area': [], 'volume_distribution': []}
        
        # Use recent data for volume profile
        recent_high = high[-window:]
        recent_low = low[-window:]
        recent_close = close[-window:]
        recent_volume = volume[-window:]
        
        # Create price levels
        price_min = np.min(recent_low)
        price_max = np.max(recent_high)
        
        if price_max <= price_min:
            return {'poc': recent_close[-1], 'value_area': [], 'volume_distribution': []}
        
        price_levels = np.linspace(price_min, price_max, self.parameters['volume_profile_bins'])
        volume_at_price = np.zeros(len(price_levels))
        
        # Distribute volume across price levels
        for i in range(len(recent_close)):
            typical_price = (recent_high[i] + recent_low[i] + recent_close[i]) / 3
            level_idx = np.argmin(np.abs(price_levels - typical_price))
            volume_at_price[level_idx] += recent_volume[i]
        
        # Find Point of Control (POC) - price with highest volume
        poc_idx = np.argmax(volume_at_price)
        poc = price_levels[poc_idx]
        
        # Calculate Value Area (70% of volume)
        total_volume = np.sum(volume_at_price)
        target_volume = total_volume * 0.7
        
        # Find value area around POC
        value_area_indices = [poc_idx]
        current_volume = volume_at_price[poc_idx]
        
        left_idx = poc_idx - 1
        right_idx = poc_idx + 1
        
        while current_volume < target_volume and (left_idx >= 0 or right_idx < len(volume_at_price)):
            left_vol = volume_at_price[left_idx] if left_idx >= 0 else 0
            right_vol = volume_at_price[right_idx] if right_idx < len(volume_at_price) else 0
            
            if left_vol >= right_vol and left_idx >= 0:
                value_area_indices.append(left_idx)
                current_volume += left_vol
                left_idx -= 1
            elif right_idx < len(volume_at_price):
                value_area_indices.append(right_idx)
                current_volume += right_vol
                right_idx += 1
            else:
                break
        
        value_area = [price_levels[i] for i in sorted(value_area_indices)]
        
        # Institutional activity indicators
        volume_concentration = np.max(volume_at_price) / np.mean(volume_at_price) if np.mean(volume_at_price) > 0 else 1.0
        institutional_interest = min(volume_concentration / 3.0, 1.0)
        
        return {
            'poc': poc,
            'value_area': value_area,
            'volume_distribution': list(zip(price_levels, volume_at_price)),
            'volume_concentration': volume_concentration,
            'institutional_interest': institutional_interest,
            'current_price_in_value_area': min(value_area) <= close[-1] <= max(value_area) if value_area else False
        }    
    def _analyze_smart_money_flow(self, close: np.ndarray, volume: np.ndarray, 
                                flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze smart money flow patterns."""
        window = self.parameters['smart_money_window']
        
        if len(close) < window:
            return {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.0}
        
        # Smart money indicators
        # 1. Price vs Volume divergence
        price_trend = self._calculate_price_trend(close[-window:])
        volume_trend = self._calculate_volume_trend(volume[-window:])
        
        # 2. Off-hours activity (simplified - using low volume periods as proxy)
        off_hours_activity = self._detect_off_hours_activity(volume[-window:])
        
        # 3. Stealth accumulation/distribution
        stealth_activity = self._detect_stealth_activity(close[-window:], volume[-window:])
        
        # 4. Large order timing analysis
        order_timing = self._analyze_order_timing(volume[-window:])
        
        # Combine indicators for smart money direction
        indicators = [
            ('price_volume_divergence', self._price_volume_divergence_signal(price_trend, volume_trend)),
            ('off_hours_activity', off_hours_activity),
            ('stealth_activity', stealth_activity),
            ('order_timing', order_timing)
        ]
        
        # Calculate weighted smart money direction
        total_weight = 0
        weighted_score = 0
        
        for name, indicator in indicators:
            weight = indicator.get('weight', 1.0)
            score = indicator.get('score', 0.0)
            total_weight += weight
            weighted_score += score * weight
        
        if total_weight > 0:
            smart_money_score = weighted_score / total_weight
        else:
            smart_money_score = 0.0
        
        # Determine direction and strength
        if smart_money_score > 0.2:
            direction = 'bullish'
            strength = min(smart_money_score, 1.0)
        elif smart_money_score < -0.2:
            direction = 'bearish'
            strength = min(abs(smart_money_score), 1.0)
        else:
            direction = 'neutral'
            strength = abs(smart_money_score)
        
        # Calculate confidence based on indicator agreement
        indicator_scores = [ind[1].get('score', 0.0) for ind in indicators]
        confidence = 1.0 - np.std(indicator_scores) if len(indicator_scores) > 1 else 0.5
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': min(max(confidence, 0.0), 1.0),
            'score': smart_money_score,
            'indicators': dict(indicators),
            'price_trend': price_trend,
            'volume_trend': volume_trend
        }
    
    def _calculate_price_trend(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate price trend characteristics."""
        if len(prices) < 2:
            return {'direction': 0.0, 'strength': 0.0, 'consistency': 0.0}
        
        # Linear regression for trend
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Normalize slope
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price != 0 else 0.0
        
        return {
            'direction': normalized_slope,
            'strength': abs(normalized_slope),
            'consistency': r_value ** 2,  # R-squared
            'slope': slope
        }
    
    def _calculate_volume_trend(self, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate volume trend characteristics."""
        if len(volumes) < 2:
            return {'direction': 0.0, 'strength': 0.0, 'consistency': 0.0}
        
        # Linear regression for volume trend
        x = np.arange(len(volumes))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, volumes)
        
        # Normalize slope
        avg_volume = np.mean(volumes)
        normalized_slope = slope / avg_volume if avg_volume != 0 else 0.0
        
        return {
            'direction': normalized_slope,
            'strength': abs(normalized_slope),
            'consistency': r_value ** 2,
            'slope': slope
        }
    
    def _price_volume_divergence_signal(self, price_trend: Dict[str, float], 
                                      volume_trend: Dict[str, float]) -> Dict[str, Any]:
        """Detect price-volume divergence signals."""
        price_dir = price_trend['direction']
        volume_dir = volume_trend['direction']
        
        # Classic divergence patterns
        if price_dir > 0.01 and volume_dir < -0.01:
            # Price up, volume down - bearish divergence
            score = -min(price_dir, abs(volume_dir))
            signal_type = 'bearish_divergence'
        elif price_dir < -0.01 and volume_dir > 0.01:
            # Price down, volume up - bullish divergence (accumulation)
            score = min(abs(price_dir), volume_dir)
            signal_type = 'bullish_divergence'
        elif price_dir > 0.01 and volume_dir > 0.01:
            # Price up, volume up - confirmation
            score = min(price_dir, volume_dir) * 0.5
            signal_type = 'bullish_confirmation'
        elif price_dir < -0.01 and volume_dir < -0.01:
            # Price down, volume down - weak selling
            score = max(price_dir, volume_dir) * 0.3
            signal_type = 'weak_selling'
        else:
            score = 0.0
            signal_type = 'neutral'
        
        return {
            'score': score,
            'weight': 1.5,  # High weight for divergence
            'type': signal_type,
            'strength': abs(score)
        }
    
    def _detect_off_hours_activity(self, volumes: np.ndarray) -> Dict[str, Any]:
        """Detect off-hours institutional activity."""
        if len(volumes) < 10:
            return {'score': 0.0, 'weight': 0.5}
        
        # Simplified: use low-volume periods as proxy for off-hours
        volume_threshold = np.percentile(volumes, 25)  # Bottom quartile
        low_volume_indices = np.where(volumes <= volume_threshold)[0]
        
        if len(low_volume_indices) == 0:
            return {'score': 0.0, 'weight': 0.5}
        
        # Check for significant price movements during low volume
        price_movements = []
        for i in low_volume_indices:
            if i > 0 and i < len(volumes) - 1:
                # This is simplified - in real implementation would need price data
                # Using volume change as proxy
                vol_change = (volumes[i+1] - volumes[i-1]) / volumes[i-1] if volumes[i-1] > 0 else 0
                price_movements.append(abs(vol_change))
        
        if price_movements:
            avg_movement = np.mean(price_movements)
            score = min(avg_movement * 2, 1.0) - 0.5  # Center around 0
        else:
            score = 0.0
        
        return {
            'score': score,
            'weight': 0.8,
            'activity_level': len(low_volume_indices) / len(volumes)
        }
    
    def _detect_stealth_activity(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Detect stealth accumulation/distribution patterns."""
        if len(prices) < 10 or len(volumes) < 10:
            return {'score': 0.0, 'weight': 1.0}
        
        # Look for consistent small volume with gradual price movement
        # Split into periods
        period_size = len(prices) // 3
        periods = [
            (prices[:period_size], volumes[:period_size]),
            (prices[period_size:2*period_size], volumes[period_size:2*period_size]),
            (prices[2*period_size:], volumes[2*period_size:])
        ]
        
        period_scores = []
        for p_prices, p_volumes in periods:
            if len(p_prices) < 3:
                continue
                
            # Calculate average volume (should be low for stealth)
            avg_volume = np.mean(p_volumes)
            volume_consistency = 1.0 - (np.std(p_volumes) / (avg_volume + 1e-10))
            
            # Calculate price trend (should be consistent)
            price_change = (p_prices[-1] - p_prices[0]) / p_prices[0] if p_prices[0] != 0 else 0
            
            # Stealth score: consistent low volume + consistent price direction
            stealth_score = volume_consistency * abs(price_change) * np.sign(price_change)
            period_scores.append(stealth_score)
        
        if period_scores:
            # Check for consistency across periods
            avg_score = np.mean(period_scores)
            consistency = 1.0 - (np.std(period_scores) / (abs(avg_score) + 1e-10))
            final_score = avg_score * consistency
        else:
            final_score = 0.0
        
        return {
            'score': final_score,
            'weight': 1.2,
            'consistency': consistency if period_scores else 0.0
        }
    
    def _analyze_order_timing(self, volumes: np.ndarray) -> Dict[str, Any]:
        """Analyze order timing patterns for institutional activity."""
        if len(volumes) < 20:
            return {'score': 0.0, 'weight': 0.8}
        
        # Look for clustered large orders (institutional coordination)
        volume_threshold = np.percentile(volumes, 75)  # Top quartile
        large_volume_indices = np.where(volumes >= volume_threshold)[0]
        
        if len(large_volume_indices) < 2:
            return {'score': 0.0, 'weight': 0.8}
        
        # Calculate clustering score
        intervals = np.diff(large_volume_indices)
        
        if len(intervals) > 0:
            # Look for regular intervals (institutional programs)
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            
            # Regular intervals suggest institutional program trading
            regularity = 1.0 / (1.0 + interval_std / (interval_mean + 1e-10))
            
            # Prefer shorter intervals (more coordinated activity)
            timing_score = regularity * (1.0 / (1.0 + interval_mean / 5.0))
        else:
            timing_score = 0.0
        
        return {
            'score': timing_score * 0.5,  # Scale down to reasonable range
            'weight': 0.8,
            'regularity': regularity if len(intervals) > 0 else 0.0,
            'large_order_frequency': len(large_volume_indices) / len(volumes)
        }
    
    def _detect_institutional_patterns(self, high: np.ndarray, low: np.ndarray, 
                                     close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Detect institutional trading patterns."""
        patterns = {
            'iceberg_orders': [],
            'time_weighted_average_price': {},
            'volume_weighted_average_price': {},
            'accumulation_distribution': {},
            'institutional_sequences': []
        }
        
        # Detect iceberg orders (repeated similar-sized large orders)
        patterns['iceberg_orders'] = self._detect_iceberg_orders(volume)
        
        # TWAP analysis
        patterns['time_weighted_average_price'] = self._analyze_twap_patterns(close, volume)
        
        # VWAP analysis
        patterns['volume_weighted_average_price'] = self._analyze_vwap_patterns(
            high, low, close, volume
        )
        
        # Accumulation/Distribution patterns
        patterns['accumulation_distribution'] = self._detect_accumulation_distribution(
            close, volume
        )
        
        # Institutional sequence detection
        patterns['institutional_sequences'] = self._detect_institutional_sequences(
            high, low, close, volume
        )
        
        return patterns
    
    def _detect_iceberg_orders(self, volume: np.ndarray) -> List[Dict[str, Any]]:
        """Detect iceberg order patterns."""
        if len(volume) < 10:
            return []
        
        iceberg_patterns = []
        
        # Look for sequences of similar large volumes
        volume_threshold = np.percentile(volume, 80)
        large_volumes = volume[volume >= volume_threshold]
        
        if len(large_volumes) < 3:
            return []
        
        # Group similar volumes
        tolerance = 0.15  # 15% tolerance
        
        for i in range(len(volume) - 5):
            if volume[i] >= volume_threshold:
                similar_volumes = []
                for j in range(i, min(i + 10, len(volume))):
                    if abs(volume[j] - volume[i]) / volume[i] <= tolerance:
                        similar_volumes.append((j, volume[j]))
                
                if len(similar_volumes) >= 3:
                    iceberg_patterns.append({
                        'start_index': i,
                        'similar_volumes': similar_volumes,
                        'pattern_length': len(similar_volumes),
                        'average_volume': np.mean([v[1] for v in similar_volumes]),
                        'consistency': 1.0 - np.std([v[1] for v in similar_volumes]) / np.mean([v[1] for v in similar_volumes])
                    })
        
        return iceberg_patterns[-5:]  # Return last 5 patterns
    
    def _analyze_twap_patterns(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze Time-Weighted Average Price patterns."""
        if len(close) < 20:
            return {'twap': 0.0, 'price_vs_twap': 0.0}
        
        # Calculate TWAP for different periods
        periods = [10, 20, 50]
        twap_analysis = {}
        
        for period in periods:
            if len(close) >= period:
                twap = np.mean(close[-period:])
                current_price = close[-1]
                
                price_vs_twap = (current_price - twap) / twap if twap != 0 else 0.0
                
                twap_analysis[f'twap_{period}'] = {
                    'twap': twap,
                    'price_vs_twap': price_vs_twap,
                    'adherence': 1.0 - abs(price_vs_twap)
                }
        
        return twap_analysis
    
    def _analyze_vwap_patterns(self, high: np.ndarray, low: np.ndarray, 
                             close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze Volume-Weighted Average Price patterns."""
        if len(close) < 20:
            return {'vwap': 0.0, 'price_vs_vwap': 0.0}
        
        periods = [10, 20, 50]
        vwap_analysis = {}
        
        for period in periods:
            if len(close) >= period:
                # Calculate typical price
                typical_price = (high[-period:] + low[-period:] + close[-period:]) / 3
                period_volume = volume[-period:]
                
                # Calculate VWAP
                if np.sum(period_volume) > 0:
                    vwap = np.sum(typical_price * period_volume) / np.sum(period_volume)
                    current_price = close[-1]
                    
                    price_vs_vwap = (current_price - vwap) / vwap if vwap != 0 else 0.0
                    
                    vwap_analysis[f'vwap_{period}'] = {
                        'vwap': vwap,
                        'price_vs_vwap': price_vs_vwap,
                        'adherence': 1.0 - abs(price_vs_vwap)
                    }
        
        return vwap_analysis
    
    def _detect_accumulation_distribution(self, close: np.ndarray, 
                                        volume: np.ndarray) -> Dict[str, Any]:
        """Detect accumulation/distribution patterns."""
        if len(close) < 30:
            return {'pattern': 'neutral', 'strength': 0.0}
        
        # Split into periods and analyze
        period_size = len(close) // 3
        periods = [
            (close[:period_size], volume[:period_size]),
            (close[period_size:2*period_size], volume[period_size:2*period_size]),
            (close[2*period_size:], volume[2*period_size:])
        ]
        
        period_patterns = []
        
        for p_close, p_volume in periods:
            if len(p_close) < 5:
                continue
            
            price_change = (p_close[-1] - p_close[0]) / p_close[0] if p_close[0] != 0 else 0
            volume_trend = np.polyfit(range(len(p_volume)), p_volume, 1)[0]
            
            # Accumulation: price stable/up + volume increasing
            # Distribution: price stable/down + volume increasing
            if abs(price_change) < 0.02:  # Price relatively stable
                if volume_trend > 0:
                    pattern = 'accumulation' if price_change >= 0 else 'distribution'
                else:
                    pattern = 'neutral'
            else:
                pattern = 'trending'
            
            period_patterns.append(pattern)
        
        # Determine overall pattern
        if period_patterns.count('accumulation') >= 2:
            overall_pattern = 'accumulation'
            strength = period_patterns.count('accumulation') / len(period_patterns)
        elif period_patterns.count('distribution') >= 2:
            overall_pattern = 'distribution'
            strength = period_patterns.count('distribution') / len(period_patterns)
        else:
            overall_pattern = 'neutral'
            strength = 0.0
        
        return {
            'pattern': overall_pattern,
            'strength': strength,
            'period_patterns': period_patterns
        }
    
    def _detect_institutional_sequences(self, high: np.ndarray, low: np.ndarray, 
                                      close: np.ndarray, volume: np.ndarray) -> List[Dict[str, Any]]:
        """Detect institutional trading sequences."""
        sequences = []
        
        if len(volume) < 20:
            return sequences
        
        # Look for sequences of coordinated activity
        volume_threshold = np.percentile(volume, 70)
        
        # Find sequences of high activity
        high_activity_indices = np.where(volume >= volume_threshold)[0]
        
        if len(high_activity_indices) < 3:
            return sequences
        
        # Group consecutive high activity periods
        current_sequence = [high_activity_indices[0]]
        
        for i in range(1, len(high_activity_indices)):
            if high_activity_indices[i] - high_activity_indices[i-1] <= 3:  # Within 3 periods
                current_sequence.append(high_activity_indices[i])
            else:
                if len(current_sequence) >= 3:
                    sequences.append(self._analyze_sequence(
                        current_sequence, high, low, close, volume
                    ))
                current_sequence = [high_activity_indices[i]]
        
        # Check last sequence
        if len(current_sequence) >= 3:
            sequences.append(self._analyze_sequence(
                current_sequence, high, low, close, volume
            ))
        
        return sequences[-3:]  # Return last 3 sequences
    
    def _analyze_sequence(self, indices: List[int], high: np.ndarray, 
                        low: np.ndarray, close: np.ndarray, 
                        volume: np.ndarray) -> Dict[str, Any]:
        """Analyze a sequence of institutional activity."""
        start_idx = indices[0]
        end_idx = indices[-1]
        
        # Price movement during sequence
        price_change = (close[end_idx] - close[start_idx]) / close[start_idx] if close[start_idx] != 0 else 0
        
        # Volume characteristics
        sequence_volume = volume[indices]
        avg_volume = np.mean(sequence_volume)
        volume_consistency = 1.0 - (np.std(sequence_volume) / (avg_volume + 1e-10))
        
        # Determine sequence type
        if price_change > 0.01:
            sequence_type = 'bullish_accumulation'
        elif price_change < -0.01:
            sequence_type = 'bearish_distribution'
        else:
            sequence_type = 'neutral_activity'
        
        return {
            'start_index': start_idx,
            'end_index': end_idx,
            'length': len(indices),
            'price_change': price_change,
            'avg_volume': avg_volume,
            'volume_consistency': volume_consistency,
            'type': sequence_type,
            'strength': abs(price_change) * volume_consistency
        }
    
    def _estimate_dark_pool_activity(self, high: np.ndarray, low: np.ndarray, 
                                   close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Estimate dark pool activity (simplified model)."""
        if len(volume) < 20:
            return {'estimated_dark_volume': 0.0, 'dark_pool_ratio': 0.0}
        
        # Look for price movements without corresponding volume
        dark_pool_indicators = []
        
        for i in range(1, len(close)):
            price_change = abs(close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0
            volume_ratio = volume[i] / np.mean(volume[max(0, i-10):i]) if np.mean(volume[max(0, i-10):i]) > 0 else 1
            
            # Significant price movement with low volume suggests dark pool activity
            if price_change > 0.005 and volume_ratio < 0.8:  # 0.5% price move, low volume
                dark_pool_indicators.append({
                    'index': i,
                    'price_change': price_change,
                    'volume_ratio': volume_ratio,
                    'dark_pool_probability': price_change / volume_ratio
                })
        
        if dark_pool_indicators:
            avg_dark_probability = np.mean([d['dark_pool_probability'] for d in dark_pool_indicators])
            estimated_dark_volume = np.sum(volume) * min(avg_dark_probability, 0.3)  # Cap at 30%
            dark_pool_ratio = min(avg_dark_probability, 0.3)
        else:
            estimated_dark_volume = 0.0
            dark_pool_ratio = 0.0
        
        return {
            'estimated_dark_volume': estimated_dark_volume,
            'dark_pool_ratio': dark_pool_ratio,
            'dark_pool_indicators': dark_pool_indicators[-10:],  # Last 10 indicators
            'activity_level': len(dark_pool_indicators) / len(close)
        }
    
    def _detect_market_maker_behavior(self, high: np.ndarray, low: np.ndarray, 
                                    close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Detect market maker behavior patterns."""
        if len(close) < 30:
            return {'activity_level': 0.0, 'spread_patterns': []}
        
        # Market maker indicators
        spread_patterns = []
        
        # Simplified spread analysis (using high-low as proxy)
        spreads = high - low
        avg_spread = np.mean(spreads)
        
        # Look for spread manipulation patterns
        for i in range(10, len(spreads)):
            recent_spread = np.mean(spreads[i-10:i])
            current_spread = spreads[i]
            
            # Sudden spread widening (market maker stepping back)
            if current_spread > recent_spread * 1.5:
                spread_patterns.append({
                    'index': i,
                    'type': 'spread_widening',
                    'spread_ratio': current_spread / recent_spread,
                    'volume': volume[i]
                })
            
            # Spread tightening (market maker providing liquidity)
            elif current_spread < recent_spread * 0.7:
                spread_patterns.append({
                    'index': i,
                    'type': 'spread_tightening',
                    'spread_ratio': current_spread / recent_spread,
                    'volume': volume[i]
                })
        
        # Market maker activity level
        activity_level = len(spread_patterns) / len(spreads) if len(spreads) > 0 else 0.0
        
        # Detect liquidity provision patterns
        liquidity_patterns = self._detect_liquidity_provision(close, volume)
        
        return {
            'activity_level': activity_level,
            'spread_patterns': spread_patterns[-10:],  # Last 10 patterns
            'liquidity_patterns': liquidity_patterns,
            'avg_spread': avg_spread,
            'current_spread': spreads[-1] if len(spreads) > 0 else 0.0
        }
    
    def _detect_liquidity_provision(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Detect liquidity provision patterns."""
        if len(close) < 20:
            return {'provision_events': [], 'provision_level': 0.0}
        
        provision_events = []
        
        # Look for volume spikes at price reversals (liquidity provision)
        for i in range(2, len(close) - 2):
            # Check for price reversal
            if ((close[i-1] < close[i-2] and close[i+1] > close[i]) or  # Bottom reversal
                (close[i-1] > close[i-2] and close[i+1] < close[i])):   # Top reversal
                
                # Check for volume spike
                avg_volume = np.mean(volume[max(0, i-5):i+5])
                if volume[i] > avg_volume * 1.5:
                    provision_events.append({
                        'index': i,
                        'reversal_type': 'bottom' if close[i+1] > close[i] else 'top',
                        'volume_ratio': volume[i] / avg_volume,
                        'price': close[i]
                    })
        
        provision_level = len(provision_events) / len(close) if len(close) > 0 else 0.0
        
        return {
            'provision_events': provision_events[-5:],  # Last 5 events
            'provision_level': provision_level
        }
    
    def _analyze_cross_market_correlation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cross-market correlation patterns (simplified)."""
        # This would normally require multiple market data
        # For now, return placeholder analysis
        return {
            'correlation_strength': 0.5,
            'lead_lag_relationship': 'neutral',
            'arbitrage_opportunities': [],
            'cross_market_flow': 'balanced'
        }
    
    def _calculate_ml_signals(self, flow_metrics: Dict[str, Any], 
                            large_orders: Dict[str, Any],
                            volume_profile: Dict[str, Any],
                            smart_money: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate machine learning-based signals."""
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(flow_metrics, large_orders, volume_profile, smart_money)
            
            if len(features) == 0:
                return {'prediction': 'neutral', 'confidence': 0.5}
            
            # For now, return rule-based signals (placeholder for ML)
            # In production, this would use trained models
            
            # Combine various signals
            flow_signal = 1 if flow_metrics.get('flow_direction') == 'bullish' else -1 if flow_metrics.get('flow_direction') == 'bearish' else 0
            smart_signal = 1 if smart_money.get('direction') == 'bullish' else -1 if smart_money.get('direction') == 'bearish' else 0
            institutional_signal = large_orders.get('institutional_activity_level', 0.5) - 0.5
            
            combined_signal = (flow_signal * 0.4 + smart_signal * 0.4 + institutional_signal * 0.2)
            
            if combined_signal > 0.3:
                prediction = 'bullish'
                confidence = min(combined_signal, 1.0)
            elif combined_signal < -0.3:
                prediction = 'bearish'
                confidence = min(abs(combined_signal), 1.0)
            else:
                prediction = 'neutral'
                confidence = 0.5
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'signal_strength': abs(combined_signal),
                'component_signals': {
                    'flow_signal': flow_signal,
                    'smart_signal': smart_signal,
                    'institutional_signal': institutional_signal
                }
            }
            
        except Exception:
            return {'prediction': 'neutral', 'confidence': 0.5}
    
    def _prepare_ml_features(self, flow_metrics: Dict[str, Any], 
                           large_orders: Dict[str, Any],
                           volume_profile: Dict[str, Any],
                           smart_money: Dict[str, Any]) -> List[float]:
        """Prepare features for machine learning models."""
        features = []
        
        # Flow metrics features
        features.extend([
            flow_metrics.get('cmf', 0.0),
            flow_metrics.get('flow_strength', 0.0),
            flow_metrics.get('volume_roc', 0.0)
        ])
        
        # Large orders features
        features.extend([
            large_orders.get('institutional_activity_level', 0.0),
            large_orders.get('recent_large_order_count', 0),
            large_orders.get('avg_block_trade_impact', 0.0)
        ])
        
        # Volume profile features
        features.extend([
            volume_profile.get('volume_concentration', 1.0),
            volume_profile.get('institutional_interest', 0.0),
            1.0 if volume_profile.get('current_price_in_value_area', False) else 0.0
        ])
        
        # Smart money features
        features.extend([
            smart_money.get('strength', 0.0),
            smart_money.get('confidence', 0.0),
            1.0 if smart_money.get('direction') == 'bullish' else -1.0 if smart_money.get('direction') == 'bearish' else 0.0
        ])
        
        return features
    
    def _calculate_adaptive_thresholds(self, flow_metrics: Dict[str, Any], 
                                     volume: np.ndarray) -> Dict[str, Any]:
        """Calculate adaptive thresholds based on market conditions."""
        if len(volume) < 20:
            return {'large_order_threshold': 1000, 'block_trade_threshold': 5000}
        
        # Calculate dynamic thresholds based on recent market activity
        recent_volume = volume[-20:]
        volume_volatility = np.std(recent_volume) / np.mean(recent_volume) if np.mean(recent_volume) > 0 else 1.0
        
        # Adjust thresholds based on volatility
        base_multiplier = self.parameters['large_order_threshold']
        volatility_adjustment = 1.0 + volume_volatility * 0.5
        
        adaptive_large_order = base_multiplier * volatility_adjustment
        adaptive_block_trade = self.parameters['block_trade_multiplier'] * volatility_adjustment
        
        return {
            'large_order_threshold': adaptive_large_order,
            'block_trade_threshold': adaptive_block_trade,
            'volume_volatility': volume_volatility,
            'volatility_adjustment': volatility_adjustment
        }
    
    def _generate_institutional_signals(self, flow_metrics: Dict[str, Any], 
                                      large_orders: Dict[str, Any],
                                      smart_money: Dict[str, Any],
                                      patterns: Dict[str, Any],
                                      ml_signals: Dict[str, Any],
                                      adaptive_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive institutional flow signals."""
        
        # Collect individual signal components
        signals = []
        
        # Flow metrics signal
        flow_direction = flow_metrics.get('flow_direction', 'neutral')
        flow_strength = flow_metrics.get('flow_strength', 0.0)
        if flow_direction == 'bullish':
            signals.append(('flow', flow_strength))
        elif flow_direction == 'bearish':
            signals.append(('flow', -flow_strength))
        else:
            signals.append(('flow', 0.0))
        
        # Smart money signal
        smart_direction = smart_money.get('direction', 'neutral')
        smart_strength = smart_money.get('strength', 0.0)
        if smart_direction == 'bullish':
            signals.append(('smart_money', smart_strength))
        elif smart_direction == 'bearish':
            signals.append(('smart_money', -smart_strength))
        else:
            signals.append(('smart_money', 0.0))
        
        # Institutional activity signal
        activity_level = large_orders.get('institutional_activity_level', 0.0)
        signals.append(('institutional_activity', (activity_level - 0.5) * 2))  # Center around 0
        
        # ML signal
        ml_prediction = ml_signals.get('prediction', 'neutral')
        ml_confidence = ml_signals.get('confidence', 0.5)
        if ml_prediction == 'bullish':
            signals.append(('ml', ml_confidence))
        elif ml_prediction == 'bearish':
            signals.append(('ml', -ml_confidence))
        else:
            signals.append(('ml', 0.0))
        
        # Weighted combination
        weights = {'flow': 0.25, 'smart_money': 0.35, 'institutional_activity': 0.2, 'ml': 0.2}
        
        weighted_signal = sum(weights[name] * signal for name, signal in signals)
        
        # Determine final signal
        if weighted_signal > self.parameters['confidence_threshold']:
            sentiment = 'bullish'
            strength = min(weighted_signal, 1.0)
            action = 'buy'
        elif weighted_signal < -self.parameters['confidence_threshold']:
            sentiment = 'bearish'
            strength = min(abs(weighted_signal), 1.0)
            action = 'sell'
        else:
            sentiment = 'neutral'
            strength = abs(weighted_signal)
            action = 'hold'
        
        # Calculate overall confidence
        individual_confidences = [
            flow_metrics.get('flow_strength', 0.5),
            smart_money.get('confidence', 0.5),
            min(activity_level * 2, 1.0),
            ml_signals.get('confidence', 0.5)
        ]
        
        overall_confidence = np.mean(individual_confidences)
        
        return {
            'sentiment': sentiment,
            'strength': strength,
            'confidence': overall_confidence,
            'action': action,
            'weighted_signal': weighted_signal,
            'component_signals': dict(signals),
            'signal_agreement': 1.0 - np.std([s[1] for s in signals]) if len(signals) > 1 else 1.0
        }
    
    def _calculate_confidence_metrics(self, signals: Dict[str, Any], 
                                    flow_metrics: Dict[str, Any],
                                    smart_money: Dict[str, Any],
                                    patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive confidence metrics."""
        
        # Base confidence from signals
        base_confidence = signals.get('confidence', 0.5)
        
        # Agreement between different signals
        agreement_score = signals.get('signal_agreement', 0.5)
        
        # Pattern confirmation
        pattern_confirmation = 0.5
        if patterns:
            accumulation_dist = patterns.get('accumulation_distribution', {})
            if accumulation_dist.get('pattern') in ['accumulation', 'distribution']:
                pattern_confirmation = accumulation_dist.get('strength', 0.5)
        
        # Volume confirmation
        volume_confirmation = flow_metrics.get('flow_strength', 0.5)
        
        # Smart money confirmation
        smart_confirmation = smart_money.get('confidence', 0.5)
        
        # Combined confidence
        confidence_components = [
            base_confidence * 0.3,
            agreement_score * 0.25,
            pattern_confirmation * 0.2,
            volume_confirmation * 0.15,
            smart_confirmation * 0.1
        ]
        
        combined_confidence = sum(confidence_components)
        
        return {
            'overall_confidence': min(max(combined_confidence, 0.0), 1.0),
            'base_confidence': base_confidence,
            'agreement_score': agreement_score,
            'pattern_confirmation': pattern_confirmation,
            'volume_confirmation': volume_confirmation,
            'smart_confirmation': smart_confirmation,
            'confidence_level': 'high' if combined_confidence > 0.7 else 'medium' if combined_confidence > 0.5 else 'low'
        }
    
    def _assess_institutional_risk(self, signals: Dict[str, Any], 
                                 flow_metrics: Dict[str, Any],
                                 volume_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk factors related to institutional flow."""
        
        risk_factors = []
        
        # Signal strength vs confidence mismatch
        strength = signals.get('strength', 0.0)
        confidence = signals.get('confidence', 0.5)
        
        if strength > 0.7 and confidence < 0.5:
            risk_factors.append('high_strength_low_confidence')
        
        # Volume concentration risk
        volume_concentration = volume_profile.get('volume_concentration', 1.0)
        if volume_concentration > 3.0:
            risk_factors.append('excessive_volume_concentration')
        
        # Flow direction vs volume trend mismatch
        flow_direction = flow_metrics.get('flow_direction', 'neutral')
        volume_trend = flow_metrics.get('volume_trend', 'stable')
        
        if flow_direction == 'bullish' and volume_trend == 'decreasing':
            risk_factors.append('bullish_flow_decreasing_volume')
        elif flow_direction == 'bearish' and volume_trend == 'decreasing':
            risk_factors.append('bearish_flow_decreasing_volume')
        
        # Overall risk assessment
        risk_level = len(risk_factors)
        if risk_level == 0:
            risk_assessment = 'low'
        elif risk_level <= 2:
            risk_assessment = 'medium'
        else:
            risk_assessment = 'high'
        
        return {
            'risk_assessment': risk_assessment,
            'risk_factors': risk_factors,
            'risk_score': min(risk_level / 5.0, 1.0),  # Normalize to 0-1
            'recommendations': self._generate_risk_recommendations(risk_factors)
        }
    
    def _generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if 'high_strength_low_confidence' in risk_factors:
            recommendations.append('Consider reducing position size due to signal uncertainty')
        
        if 'excessive_volume_concentration' in risk_factors:
            recommendations.append('Monitor for potential market manipulation or artificial volume')
        
        if 'bullish_flow_decreasing_volume' in risk_factors:
            recommendations.append('Bullish signal may lack conviction - wait for volume confirmation')
        
        if 'bearish_flow_decreasing_volume' in risk_factors:
            recommendations.append('Bearish signal may indicate exhaustion rather than strong selling')
        
        if not recommendations:
            recommendations.append('Risk factors are minimal - proceed with normal position sizing')
        
        return recommendations
    
    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when insufficient data."""
        return {
            'flow_metrics': {'flow_direction': 'neutral', 'flow_strength': 0.0},
            'large_orders': {'institutional_activity_level': 0.0},
            'volume_profile': {'institutional_interest': 0.0},
            'smart_money': {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.0},
            'patterns': {},
            'signals': {'sentiment': 'neutral', 'strength': 0.0, 'confidence': 0.0},
            'institutional_sentiment': 'neutral',
            'flow_strength': 0.0,
            'smart_money_direction': 'neutral'
        }
    
    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        return self._get_default_output()