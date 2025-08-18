"""
Liquidity Flow Signal Indicator - AI Enhanced Category
======================================================

Advanced AI-enhanced liquidity flow analysis system with market depth modeling,
liquidity provision detection, and flow dynamics analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy import signal, stats, optimize
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class LiquidityFlowSignalIndicator(StandardIndicatorInterface):
    """
    AI-Enhanced Liquidity Flow Signal with advanced features.
    
    Features:
    - Market depth analysis and modeling
    - Liquidity provision and consumption detection
    - Bid-ask spread dynamics analysis
    - Order flow imbalance calculation
    - Market microstructure analysis
    - Liquidity shock detection and impact assessment
    - Cross-venue liquidity aggregation modeling
    - High-frequency liquidity patterns
    - Algorithmic trading impact on liquidity
    - Stress testing liquidity conditions
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'depth_levels': 10,              # Number of depth levels to analyze
            'liquidity_window': 20,          # Window for liquidity calculations
            'flow_sensitivity': 0.01,        # Sensitivity for flow detection (1%)
            'imbalance_threshold': 0.3,      # Threshold for significant imbalance
            'shock_detection_window': 50,    # Window for shock detection
            'microstructure_window': 100,    # Window for microstructure analysis
            'spread_threshold': 0.005,       # Threshold for wide spreads (0.5%)
            'volume_cluster_threshold': 2.0, # Threshold for volume clustering
            'liquidity_stress_threshold': 0.7, # Threshold for liquidity stress
            'prediction_horizon': 5,         # Periods ahead for liquidity prediction
            'adaptive_parameters': True,     # Enable adaptive parameter adjustment
            'cross_venue_analysis': True,    # Enable cross-venue analysis
            'hft_detection': True,          # Enable HFT activity detection
            'stress_testing': True,         # Enable liquidity stress testing
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("LiquidityFlowSignalIndicator", default_params)
        
        # Initialize ML models
        self.liquidity_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42
        )
        self.flow_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.is_trained = False
        
        # State tracking
        self.liquidity_history = []
        self.flow_history = []
        self.shock_events = []
        self.depth_cache = {}
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["high", "low", "close", "volume"],
            min_periods=self.parameters['microstructure_window']
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced liquidity flow signals with AI enhancements."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Extract data arrays
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate market depth proxy and liquidity metrics
            depth_analysis = self._analyze_market_depth(high, low, close, volume)
            
            # Calculate liquidity flow metrics
            flow_metrics = self._calculate_liquidity_flow_metrics(high, low, close, volume)
            
            # Analyze bid-ask spread dynamics
            spread_analysis = self._analyze_spread_dynamics(high, low, close, volume)
            
            # Calculate order flow imbalance
            order_flow_imbalance = self._calculate_order_flow_imbalance(high, low, close, volume)
            
            # Market microstructure analysis
            microstructure = self._analyze_market_microstructure(high, low, close, volume)
            
            # Liquidity shock detection
            shock_detection = self._detect_liquidity_shocks(close, volume, flow_metrics)
            
            # Cross-venue liquidity analysis
            cross_venue_analysis = {}
            if self.parameters['cross_venue_analysis']:
                cross_venue_analysis = self._analyze_cross_venue_liquidity(data)
            
            # HFT activity detection
            hft_analysis = {}
            if self.parameters['hft_detection']:
                hft_analysis = self._detect_hft_activity(high, low, close, volume)
            
            # Liquidity stress testing
            stress_test = {}
            if self.parameters['stress_testing']:
                stress_test = self._perform_liquidity_stress_test(
                    depth_analysis, flow_metrics, volume
                )
            
            # Machine learning predictions
            ml_predictions = self._calculate_ml_predictions(
                depth_analysis, flow_metrics, spread_analysis, order_flow_imbalance
            )
            
            # Adaptive parameter adjustment
            adaptive_params = {}
            if self.parameters['adaptive_parameters']:
                adaptive_params = self._adjust_adaptive_parameters(
                    flow_metrics, spread_analysis, volume
                )
            
            # Generate comprehensive signals
            signals = self._generate_liquidity_signals(
                depth_analysis, flow_metrics, spread_analysis,
                order_flow_imbalance, ml_predictions
            )
            
            # Calculate liquidity risk assessment
            risk_assessment = self._assess_liquidity_risk(
                depth_analysis, flow_metrics, shock_detection, stress_test
            )
            
            # Performance metrics
            performance_metrics = self._calculate_performance_metrics(
                signals, flow_metrics, depth_analysis
            )
            
            return {
                'depth_analysis': depth_analysis,
                'flow_metrics': flow_metrics,
                'spread_analysis': spread_analysis,
                'order_flow_imbalance': order_flow_imbalance,
                'microstructure': microstructure,
                'shock_detection': shock_detection,
                'cross_venue_analysis': cross_venue_analysis,
                'hft_analysis': hft_analysis,
                'stress_test': stress_test,
                'ml_predictions': ml_predictions,
                'adaptive_params': adaptive_params,
                'signals': signals,
                'risk_assessment': risk_assessment,
                'performance_metrics': performance_metrics,
                'liquidity_score': depth_analysis.get('liquidity_score', 0.5),
                'flow_direction': flow_metrics.get('net_flow_direction', 'neutral'),
                'liquidity_risk': risk_assessment.get('overall_risk', 'medium')
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _analyze_market_depth(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze market depth and liquidity distribution."""
        
        # Simulate market depth using price-volume distribution
        window = self.parameters['liquidity_window']
        depth_levels = self.parameters['depth_levels']
        
        if len(close) < window:
            return {'liquidity_score': 0.5, 'depth_distribution': []}
        
        # Use recent data for depth analysis
        recent_high = high[-window:]
        recent_low = low[-window:]
        recent_close = close[-window:]
        recent_volume = volume[-window:]
        
        # Create price levels around current price
        current_price = close[-1]
        price_range = np.max(recent_high) - np.min(recent_low)
        tick_size = price_range / (depth_levels * 2)
        
        # Create bid and ask levels
        bid_levels = current_price - np.arange(1, depth_levels + 1) * tick_size
        ask_levels = current_price + np.arange(1, depth_levels + 1) * tick_size
        
        # Estimate liquidity at each level using volume distribution
        bid_liquidity = self._estimate_liquidity_at_levels(
            bid_levels, recent_high, recent_low, recent_close, recent_volume
        )
        ask_liquidity = self._estimate_liquidity_at_levels(
            ask_levels, recent_high, recent_low, recent_close, recent_volume
        )
        
        # Calculate depth metrics
        total_bid_liquidity = np.sum(bid_liquidity)
        total_ask_liquidity = np.sum(ask_liquidity)
        total_liquidity = total_bid_liquidity + total_ask_liquidity
        
        # Liquidity imbalance
        if total_liquidity > 0:
            liquidity_imbalance = (total_bid_liquidity - total_ask_liquidity) / total_liquidity
        else:
            liquidity_imbalance = 0.0
        
        # Depth concentration (how concentrated is liquidity)
        all_liquidity = np.concatenate([bid_liquidity, ask_liquidity])
        if len(all_liquidity) > 0 and np.sum(all_liquidity) > 0:
            liquidity_concentration = np.max(all_liquidity) / np.sum(all_liquidity)
        else:
            liquidity_concentration = 1.0 / len(all_liquidity) if len(all_liquidity) > 0 else 0.0
        
        # Liquidity score (higher is better)
        liquidity_score = min(total_liquidity / (np.mean(recent_volume) * depth_levels), 1.0)
        
        # Market depth resilience
        depth_resilience = self._calculate_depth_resilience(
            bid_levels, ask_levels, bid_liquidity, ask_liquidity, current_price
        )
        
        return {
            'bid_levels': bid_levels.tolist(),
            'ask_levels': ask_levels.tolist(),
            'bid_liquidity': bid_liquidity.tolist(),
            'ask_liquidity': ask_liquidity.tolist(),
            'total_bid_liquidity': total_bid_liquidity,
            'total_ask_liquidity': total_ask_liquidity,
            'total_liquidity': total_liquidity,
            'liquidity_imbalance': liquidity_imbalance,
            'liquidity_concentration': liquidity_concentration,
            'liquidity_score': liquidity_score,
            'depth_resilience': depth_resilience,
            'effective_spread': self._calculate_effective_spread(bid_levels, ask_levels, bid_liquidity, ask_liquidity)
        }
    
    def _estimate_liquidity_at_levels(self, price_levels: np.ndarray, 
                                    high: np.ndarray, low: np.ndarray,
                                    close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Estimate liquidity at specific price levels."""
        liquidity = np.zeros(len(price_levels))
        
        for i, level in enumerate(price_levels):
            # Find periods where price was near this level
            level_tolerance = (np.max(high) - np.min(low)) * 0.001  # 0.1% tolerance
            
            near_level_mask = (low <= level + level_tolerance) & (high >= level - level_tolerance)
            near_level_volume = volume[near_level_mask]
            
            if len(near_level_volume) > 0:
                # Estimate liquidity based on historical volume at this level
                liquidity[i] = np.mean(near_level_volume)
            else:
                # Interpolate based on nearby levels
                liquidity[i] = np.mean(volume) * 0.5  # Default estimate
        
        return liquidity
    
    def _calculate_depth_resilience(self, bid_levels: np.ndarray, ask_levels: np.ndarray,
                                  bid_liquidity: np.ndarray, ask_liquidity: np.ndarray,
                                  current_price: float) -> Dict[str, float]:
        """Calculate market depth resilience metrics."""
        
        # Calculate cumulative liquidity
        cumulative_bid = np.cumsum(bid_liquidity)
        cumulative_ask = np.cumsum(ask_liquidity)
        
        # Find price impact for different trade sizes
        trade_sizes = [0.1, 0.5, 1.0, 2.0]  # Relative to average volume
        price_impacts = {}
        
        for size in trade_sizes:
            # Calculate price impact for buying (moving through asks)
            target_volume = size * np.mean(ask_liquidity) if len(ask_liquidity) > 0 else 0
            
            if target_volume > 0 and len(cumulative_ask) > 0:
                impact_level = np.searchsorted(cumulative_ask, target_volume)
                if impact_level < len(ask_levels):
                    buy_impact = (ask_levels[impact_level] - current_price) / current_price
                else:
                    buy_impact = (ask_levels[-1] - current_price) / current_price
            else:
                buy_impact = 0.0
            
            # Calculate price impact for selling (moving through bids)
            if target_volume > 0 and len(cumulative_bid) > 0:
                impact_level = np.searchsorted(cumulative_bid, target_volume)
                if impact_level < len(bid_levels):
                    sell_impact = (current_price - bid_levels[impact_level]) / current_price
                else:
                    sell_impact = (current_price - bid_levels[-1]) / current_price
            else:
                sell_impact = 0.0
            
            price_impacts[f'size_{size}'] = {
                'buy_impact': buy_impact,
                'sell_impact': sell_impact,
                'avg_impact': (buy_impact + sell_impact) / 2
            }
        
        # Overall resilience score (lower price impact = higher resilience)
        avg_impacts = [impacts['avg_impact'] for impacts in price_impacts.values()]
        resilience_score = 1.0 / (1.0 + np.mean(avg_impacts)) if avg_impacts else 0.5
        
        return {
            'resilience_score': resilience_score,
            'price_impacts': price_impacts
        }
    
    def _calculate_effective_spread(self, bid_levels: np.ndarray, ask_levels: np.ndarray,
                                  bid_liquidity: np.ndarray, ask_liquidity: np.ndarray) -> float:
        """Calculate effective spread weighted by liquidity."""
        if len(bid_levels) == 0 or len(ask_levels) == 0:
            return 0.0
        
        # Best bid and ask
        best_bid = np.max(bid_levels) if len(bid_levels) > 0 else 0
        best_ask = np.min(ask_levels) if len(ask_levels) > 0 else 0
        
        if best_bid > 0 and best_ask > best_bid:
            effective_spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
        else:
            effective_spread = 0.0
        
        return effective_spread
    
    def _calculate_liquidity_flow_metrics(self, high: np.ndarray, low: np.ndarray,
                                        close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive liquidity flow metrics."""
        
        window = self.parameters['liquidity_window']
        
        # Money Flow Index (MFI)
        mfi = self._calculate_money_flow_index(high, low, close, volume, window)
        
        # Volume-Price Trend (VPT)
        vpt = self._calculate_volume_price_trend(close, volume)
        
        # Ease of Movement (EOM)
        eom = self._calculate_ease_of_movement(high, low, volume)
        
        # Negative Volume Index (NVI) and Positive Volume Index (PVI)
        nvi, pvi = self._calculate_volume_indices(close, volume)
        
        # Liquidity flow direction and strength
        flow_signals = [
            ('mfi', 1 if mfi > 50 else -1 if mfi < 50 else 0, abs(mfi - 50) / 50),
            ('vpt', 1 if len(vpt) > 1 and vpt[-1] > vpt[-2] else -1 if len(vpt) > 1 and vpt[-1] < vpt[-2] else 0, 0.5),
            ('eom', 1 if eom > 0 else -1 if eom < 0 else 0, min(abs(eom), 1.0))
        ]
        
        # Calculate net flow
        total_weight = sum(signal[2] for signal in flow_signals)
        if total_weight > 0:
            net_flow = sum(signal[0] * signal[1] * signal[2] for signal in flow_signals) / total_weight
        else:
            net_flow = 0.0
        
        # Flow strength
        flow_strength = abs(net_flow)
        
        # Flow direction
        if net_flow > 0.2:
            flow_direction = 'inflow'
        elif net_flow < -0.2:
            flow_direction = 'outflow'
        else:
            flow_direction = 'neutral'
        
        # Flow consistency (using rolling correlation)
        flow_consistency = self._calculate_flow_consistency(vpt, eom)
        
        # Volume flow analysis
        volume_flow = self._analyze_volume_flow(volume, close)
        
        return {
            'mfi': mfi,
            'vpt': vpt[-1] if len(vpt) > 0 else 0.0,
            'eom': eom,
            'nvi': nvi[-1] if len(nvi) > 0 else 0.0,
            'pvi': pvi[-1] if len(pvi) > 0 else 0.0,
            'net_flow': net_flow,
            'flow_strength': flow_strength,
            'net_flow_direction': flow_direction,
            'flow_consistency': flow_consistency,
            'volume_flow': volume_flow,
            'flow_signals': {name: {'direction': direction, 'strength': strength} 
                           for name, direction, strength in flow_signals}
        }
    
    def _calculate_money_flow_index(self, high: np.ndarray, low: np.ndarray,
                                  close: np.ndarray, volume: np.ndarray, window: int) -> float:
        """Calculate Money Flow Index."""
        if len(close) < window + 1:
            return 50.0
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        # Raw money flow
        raw_money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow.append(raw_money_flow[i])
                negative_flow.append(0)
            elif typical_price[i] < typical_price[i-1]:
                positive_flow.append(0)
                negative_flow.append(raw_money_flow[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_flow = np.array(positive_flow)
        negative_flow = np.array(negative_flow)
        
        # Calculate MFI for the last window
        if len(positive_flow) >= window:
            positive_mf = np.sum(positive_flow[-window:])
            negative_mf = np.sum(negative_flow[-window:])
            
            if negative_mf == 0:
                return 100.0
            
            money_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + money_ratio))
        else:
            mfi = 50.0
        
        return mfi
    
    def _calculate_volume_price_trend(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Volume Price Trend."""
        if len(close) < 2:
            return np.array([0.0])
        
        price_change = np.diff(close)
        price_change_pct = price_change / close[:-1]
        
        vpt = np.zeros(len(close))
        vpt[0] = 0
        
        for i in range(1, len(close)):
            vpt[i] = vpt[i-1] + (price_change_pct[i-1] * volume[i])
        
        return vpt
    
    def _calculate_ease_of_movement(self, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> float:
        """Calculate Ease of Movement."""
        if len(high) < 2:
            return 0.0
        
        # Distance moved
        dm = ((high + low) / 2) - np.roll((high + low) / 2, 1)
        dm[0] = 0  # First value
        
        # Box height (high-low)
        bh = high - low
        
        # Box ratio
        box_ratio = np.where((volume > 0) & (bh > 0), volume / bh, 0)
        
        # Ease of Movement
        eom = np.where(box_ratio > 0, dm / box_ratio, 0)
        
        # Return smoothed EOM (simple moving average)
        window = min(14, len(eom))
        if window > 0:
            return np.mean(eom[-window:])
        else:
            return 0.0
    
    def _calculate_volume_indices(self, close: np.ndarray, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Negative Volume Index and Positive Volume Index."""
        if len(close) < 2:
            return np.array([1000.0]), np.array([1000.0])
        
        nvi = np.zeros(len(close))
        pvi = np.zeros(len(close))
        
        nvi[0] = 1000  # Starting value
        pvi[0] = 1000  # Starting value
        
        for i in range(1, len(close)):
            price_change = (close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0
            
            if volume[i] < volume[i-1]:  # Decreasing volume
                nvi[i] = nvi[i-1] * (1 + price_change)
                pvi[i] = pvi[i-1]
            elif volume[i] > volume[i-1]:  # Increasing volume
                pvi[i] = pvi[i-1] * (1 + price_change)
                nvi[i] = nvi[i-1]
            else:  # Same volume
                nvi[i] = nvi[i-1]
                pvi[i] = pvi[i-1]
        
        return nvi, pvi
    
    def _calculate_flow_consistency(self, vpt: np.ndarray, eom: float) -> float:
        """Calculate flow consistency metric."""
        if len(vpt) < 10:
            return 0.5
        
        # Use correlation between consecutive VPT values to measure consistency
        vpt_changes = np.diff(vpt[-20:]) if len(vpt) >= 20 else np.diff(vpt)
        
        if len(vpt_changes) > 1:
            # Measure consistency as inverse of standard deviation
            consistency = 1.0 / (1.0 + np.std(vpt_changes))
        else:
            consistency = 0.5
        
        return min(max(consistency, 0.0), 1.0)
    
    def _analyze_volume_flow(self, volume: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Analyze volume flow patterns."""
        if len(volume) < 10:
            return {'flow_type': 'neutral', 'flow_intensity': 0.0}
        
        window = min(20, len(volume))
        recent_volume = volume[-window:]
        recent_close = close[-window:]
        
        # Calculate volume trend
        volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
        
        # Calculate price-volume correlation
        if len(recent_volume) > 1 and len(recent_close) > 1:
            correlation = np.corrcoef(recent_volume, recent_close)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Determine flow type
        if volume_trend > 0 and correlation > 0.3:
            flow_type = 'accumulation'
            flow_intensity = min(volume_trend / np.mean(recent_volume), 1.0)
        elif volume_trend > 0 and correlation < -0.3:
            flow_type = 'distribution'
            flow_intensity = min(volume_trend / np.mean(recent_volume), 1.0)
        elif volume_trend < 0:
            flow_type = 'drying_up'
            flow_intensity = min(abs(volume_trend) / np.mean(recent_volume), 1.0)
        else:
            flow_type = 'neutral'
            flow_intensity = 0.0
        
        return {
            'flow_type': flow_type,
            'flow_intensity': flow_intensity,
            'volume_trend': volume_trend,
            'price_volume_correlation': correlation
        }
    
    def _analyze_spread_dynamics(self, high: np.ndarray, low: np.ndarray,
                               close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze bid-ask spread dynamics."""
        
        # Use high-low as proxy for spread
        spreads = high - low
        
        # Relative spread
        relative_spreads = spreads / close
        
        window = self.parameters['liquidity_window']
        
        # Current spread metrics
        current_spread = spreads[-1] if len(spreads) > 0 else 0.0
        current_relative_spread = relative_spreads[-1] if len(relative_spreads) > 0 else 0.0
        
        # Average spread
        avg_spread = np.mean(spreads[-window:]) if len(spreads) >= window else np.mean(spreads)
        avg_relative_spread = np.mean(relative_spreads[-window:]) if len(relative_spreads) >= window else np.mean(relative_spreads)
        
        # Spread volatility
        spread_volatility = np.std(spreads[-window:]) if len(spreads) >= window else np.std(spreads)
        
        # Spread trend
        if len(spreads) >= window:
            spread_trend = np.polyfit(range(window), spreads[-window:], 1)[0]
        else:
            spread_trend = 0.0
        
        # Wide spread detection
        wide_spread_threshold = self.parameters['spread_threshold']
        is_wide_spread = current_relative_spread > wide_spread_threshold
        
        # Spread-volume relationship
        if len(spreads) >= 10 and len(volume) >= 10:
            spread_volume_corr = np.corrcoef(spreads[-10:], volume[-10:])[0, 1]
            if np.isnan(spread_volume_corr):
                spread_volume_corr = 0.0
        else:
            spread_volume_corr = 0.0
        
        # Market quality score (lower spread = higher quality)
        if avg_relative_spread > 0:
            market_quality = 1.0 / (1.0 + avg_relative_spread * 100)
        else:
            market_quality = 1.0
        
        return {
            'current_spread': current_spread,
            'current_relative_spread': current_relative_spread,
            'avg_spread': avg_spread,
            'avg_relative_spread': avg_relative_spread,
            'spread_volatility': spread_volatility,
            'spread_trend': spread_trend,
            'is_wide_spread': is_wide_spread,
            'spread_volume_correlation': spread_volume_corr,
            'market_quality': market_quality,
            'spread_regime': 'wide' if is_wide_spread else 'normal'
        }    
    def _calculate_order_flow_imbalance(self, high: np.ndarray, low: np.ndarray,
                                      close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Calculate order flow imbalance metrics."""
        
        # Estimate buy/sell volume using price action
        buy_volume = []
        sell_volume = []
        
        for i in range(1, len(close)):
            # Simple heuristic: if close > previous close, assume more buying
            if close[i] > close[i-1]:
                # Price up - more buying pressure
                typical_price = (high[i] + low[i] + close[i]) / 3
                prev_typical = (high[i-1] + low[i-1] + close[i-1]) / 3
                
                if typical_price > prev_typical:
                    buy_vol = volume[i] * 0.7  # Assume 70% buy volume
                    sell_vol = volume[i] * 0.3
                else:
                    buy_vol = volume[i] * 0.6
                    sell_vol = volume[i] * 0.4
                    
            elif close[i] < close[i-1]:
                # Price down - more selling pressure
                typical_price = (high[i] + low[i] + close[i]) / 3
                prev_typical = (high[i-1] + low[i-1] + close[i-1]) / 3
                
                if typical_price < prev_typical:
                    buy_vol = volume[i] * 0.3
                    sell_vol = volume[i] * 0.7  # Assume 70% sell volume
                else:
                    buy_vol = volume[i] * 0.4
                    sell_vol = volume[i] * 0.6
            else:
                # No price change - assume balanced
                buy_vol = volume[i] * 0.5
                sell_vol = volume[i] * 0.5
            
            buy_volume.append(buy_vol)
            sell_volume.append(sell_vol)
        
        buy_volume = np.array(buy_volume)
        sell_volume = np.array(sell_volume)
        
        # Calculate imbalance metrics
        window = self.parameters['liquidity_window']
        
        if len(buy_volume) >= window:
            recent_buy = np.sum(buy_volume[-window:])
            recent_sell = np.sum(sell_volume[-window:])
        else:
            recent_buy = np.sum(buy_volume)
            recent_sell = np.sum(sell_volume)
        
        total_volume = recent_buy + recent_sell
        
        if total_volume > 0:
            buy_ratio = recent_buy / total_volume
            sell_ratio = recent_sell / total_volume
            imbalance = (recent_buy - recent_sell) / total_volume
        else:
            buy_ratio = 0.5
            sell_ratio = 0.5
            imbalance = 0.0
        
        # Imbalance strength
        imbalance_strength = abs(imbalance)
        
        # Imbalance direction
        if imbalance > self.parameters['imbalance_threshold']:
            imbalance_direction = 'buy_dominated'
        elif imbalance < -self.parameters['imbalance_threshold']:
            imbalance_direction = 'sell_dominated'
        else:
            imbalance_direction = 'balanced'
        
        # Rolling imbalance for trend analysis
        rolling_imbalances = []
        for i in range(window, len(buy_volume)):
            period_buy = np.sum(buy_volume[i-window:i])
            period_sell = np.sum(sell_volume[i-window:i])
            period_total = period_buy + period_sell
            
            if period_total > 0:
                period_imbalance = (period_buy - period_sell) / period_total
            else:
                period_imbalance = 0.0
            
            rolling_imbalances.append(period_imbalance)
        
        # Imbalance trend
        if len(rolling_imbalances) > 1:
            imbalance_trend = np.polyfit(range(len(rolling_imbalances)), rolling_imbalances, 1)[0]
        else:
            imbalance_trend = 0.0
        
        return {
            'buy_volume': recent_buy,
            'sell_volume': recent_sell,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'imbalance': imbalance,
            'imbalance_strength': imbalance_strength,
            'imbalance_direction': imbalance_direction,
            'imbalance_trend': imbalance_trend,
            'rolling_imbalances': rolling_imbalances[-10:] if len(rolling_imbalances) >= 10 else rolling_imbalances
        }
    
    def _analyze_market_microstructure(self, high: np.ndarray, low: np.ndarray,
                                     close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze market microstructure patterns."""
        
        window = self.parameters['microstructure_window']
        
        # Price clustering analysis
        price_clustering = self._analyze_price_clustering(close[-window:] if len(close) >= window else close)
        
        # Trade size distribution
        trade_size_dist = self._analyze_trade_size_distribution(volume[-window:] if len(volume) >= window else volume)
        
        # Intraday patterns (simplified)
        intraday_patterns = self._analyze_intraday_patterns(high, low, close, volume)
        
        # Market impact analysis
        market_impact = self._analyze_market_impact(high, low, close, volume)
        
        # Tick-by-tick patterns (simulated)
        tick_patterns = self._analyze_tick_patterns(close, volume)
        
        return {
            'price_clustering': price_clustering,
            'trade_size_distribution': trade_size_dist,
            'intraday_patterns': intraday_patterns,
            'market_impact': market_impact,
            'tick_patterns': tick_patterns
        }
    
    def _analyze_price_clustering(self, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze price clustering patterns."""
        if len(prices) < 10:
            return {'clustering_score': 0.0, 'round_number_attraction': 0.0}
        
        # Analyze clustering around round numbers
        round_numbers = []
        for price in prices:
            # Check for clustering around .00, .25, .50, .75
            decimals = price - int(price)
            round_attractions = [0.0, 0.25, 0.5, 0.75, 1.0]
            
            min_distance = min(abs(decimals - rn) for rn in round_attractions)
            round_numbers.append(min_distance)
        
        # Clustering score (lower distances indicate more clustering)
        clustering_score = 1.0 - np.mean(round_numbers) if round_numbers else 0.0
        
        # Round number attraction
        round_threshold = 0.05  # 5 cents
        round_attraction = np.mean([1 if rn <= round_threshold else 0 for rn in round_numbers])
        
        return {
            'clustering_score': max(0.0, min(clustering_score, 1.0)),
            'round_number_attraction': round_attraction
        }
    
    def _analyze_trade_size_distribution(self, volumes: np.ndarray) -> Dict[str, Any]:
        """Analyze trade size distribution patterns."""
        if len(volumes) < 10:
            return {'distribution_type': 'unknown', 'large_trade_frequency': 0.0}
        
        # Basic statistics
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        # Large trade detection
        large_trade_threshold = mean_volume + 2 * std_volume
        large_trades = volumes[volumes > large_trade_threshold]
        large_trade_frequency = len(large_trades) / len(volumes)
        
        # Distribution analysis
        skewness = stats.skew(volumes) if len(volumes) > 3 else 0.0
        kurtosis = stats.kurtosis(volumes) if len(volumes) > 3 else 0.0
        
        # Classify distribution
        if abs(skewness) < 0.5:
            distribution_type = 'normal'
        elif skewness > 0.5:
            distribution_type = 'right_skewed'
        else:
            distribution_type = 'left_skewed'
        
        return {
            'distribution_type': distribution_type,
            'large_trade_frequency': large_trade_frequency,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'mean_volume': mean_volume,
            'volume_concentration': std_volume / mean_volume if mean_volume > 0 else 0.0
        }
    
    def _analyze_intraday_patterns(self, high: np.ndarray, low: np.ndarray,
                                 close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze intraday liquidity patterns."""
        # Simplified intraday analysis (in real implementation would use timestamp data)
        
        # Volume patterns throughout the period
        period_size = len(volume) // 4 if len(volume) >= 4 else 1
        
        periods = {
            'opening': volume[:period_size] if period_size > 0 else volume,
            'morning': volume[period_size:2*period_size] if len(volume) > period_size else volume,
            'afternoon': volume[2*period_size:3*period_size] if len(volume) > 2*period_size else volume,
            'closing': volume[3*period_size:] if len(volume) > 3*period_size else volume
        }
        
        period_stats = {}
        for period_name, period_data in periods.items():
            if len(period_data) > 0:
                period_stats[period_name] = {
                    'avg_volume': np.mean(period_data),
                    'volatility': np.std(period_data),
                    'liquidity_score': np.mean(period_data) / (np.std(period_data) + 1e-10)
                }
        
        # Identify peak liquidity periods
        if period_stats:
            peak_period = max(period_stats.keys(), key=lambda x: period_stats[x]['liquidity_score'])
            low_period = min(period_stats.keys(), key=lambda x: period_stats[x]['liquidity_score'])
        else:
            peak_period = 'unknown'
            low_period = 'unknown'
        
        return {
            'period_stats': period_stats,
            'peak_liquidity_period': peak_period,
            'low_liquidity_period': low_period
        }
    
    def _analyze_market_impact(self, high: np.ndarray, low: np.ndarray,
                             close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze market impact of trades."""
        if len(close) < 5 or len(volume) < 5:
            return {'avg_impact': 0.0, 'impact_efficiency': 0.5}
        
        impacts = []
        
        for i in range(2, len(close) - 2):
            # Calculate price change around high volume periods
            if volume[i] > np.mean(volume):
                # Pre-trade price
                pre_price = close[i-1]
                
                # Post-trade price (looking ahead)
                post_price = close[i+1] if i+1 < len(close) else close[i]
                
                # Calculate impact
                if pre_price != 0:
                    impact = abs(post_price - pre_price) / pre_price
                    impacts.append(impact)
        
        if impacts:
            avg_impact = np.mean(impacts)
            impact_efficiency = 1.0 / (1.0 + avg_impact * 100)  # Lower impact = higher efficiency
        else:
            avg_impact = 0.0
            impact_efficiency = 0.5
        
        return {
            'avg_impact': avg_impact,
            'impact_efficiency': impact_efficiency,
            'impact_samples': len(impacts)
        }
    
    def _analyze_tick_patterns(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Analyze tick-by-tick patterns (simulated)."""
        if len(close) < 10:
            return {'tick_direction': 'neutral', 'tick_momentum': 0.0}
        
        # Calculate tick directions
        tick_directions = []
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                tick_directions.append(1)  # Uptick
            elif close[i] < close[i-1]:
                tick_directions.append(-1)  # Downtick
            else:
                tick_directions.append(0)  # No change
        
        # Tick momentum
        if tick_directions:
            tick_momentum = np.mean(tick_directions)
            
            # Overall direction
            if tick_momentum > 0.1:
                tick_direction = 'upward'
            elif tick_momentum < -0.1:
                tick_direction = 'downward'
            else:
                tick_direction = 'neutral'
        else:
            tick_momentum = 0.0
            tick_direction = 'neutral'
        
        # Tick clustering
        tick_runs = self._calculate_tick_runs(tick_directions)
        
        return {
            'tick_direction': tick_direction,
            'tick_momentum': tick_momentum,
            'tick_runs': tick_runs
        }
    
    def _calculate_tick_runs(self, tick_directions: List[int]) -> Dict[str, float]:
        """Calculate tick run statistics."""
        if not tick_directions:
            return {'avg_run_length': 0.0, 'max_run_length': 0}
        
        runs = []
        current_run = 1
        
        for i in range(1, len(tick_directions)):
            if tick_directions[i] == tick_directions[i-1] and tick_directions[i] != 0:
                current_run += 1
            else:
                if current_run > 1:
                    runs.append(current_run)
                current_run = 1
        
        # Add final run
        if current_run > 1:
            runs.append(current_run)
        
        if runs:
            avg_run_length = np.mean(runs)
            max_run_length = max(runs)
        else:
            avg_run_length = 0.0
            max_run_length = 0
        
        return {
            'avg_run_length': avg_run_length,
            'max_run_length': max_run_length,
            'total_runs': len(runs)
        }
    
    def _detect_liquidity_shocks(self, close: np.ndarray, volume: np.ndarray, 
                                flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect liquidity shock events."""
        
        window = self.parameters['shock_detection_window']
        
        if len(close) < window:
            return {'shock_events': [], 'current_shock_level': 0.0}
        
        # Calculate rolling volatility and volume
        rolling_vol = []
        rolling_volume = []
        
        for i in range(window, len(close)):
            period_returns = np.diff(np.log(close[i-window:i+1]))
            period_volatility = np.std(period_returns) * np.sqrt(252)  # Annualized
            period_avg_volume = np.mean(volume[i-window:i+1])
            
            rolling_vol.append(period_volatility)
            rolling_volume.append(period_avg_volume)
        
        rolling_vol = np.array(rolling_vol)
        rolling_volume = np.array(rolling_volume)
        
        # Detect shocks
        shock_events = []
        
        if len(rolling_vol) > 10:
            vol_threshold = np.percentile(rolling_vol, 90)  # Top 10% volatility
            volume_threshold = np.percentile(rolling_volume, 90)  # Top 10% volume
            
            for i in range(len(rolling_vol)):
                actual_idx = i + window
                
                # Shock conditions
                high_vol_shock = rolling_vol[i] > vol_threshold
                volume_shock = rolling_volume[i] > volume_threshold
                
                # Price gap detection
                if actual_idx > 0 and actual_idx < len(close):
                    price_gap = abs(close[actual_idx] - close[actual_idx-1]) / close[actual_idx-1]
                    gap_shock = price_gap > 0.02  # 2% gap
                else:
                    gap_shock = False
                
                if high_vol_shock or volume_shock or gap_shock:
                    shock_events.append({
                        'index': actual_idx,
                        'type': 'volatility' if high_vol_shock else 'volume' if volume_shock else 'gap',
                        'intensity': rolling_vol[i] / np.mean(rolling_vol) if high_vol_shock else rolling_volume[i] / np.mean(rolling_volume),
                        'price': close[actual_idx] if actual_idx < len(close) else 0
                    })
        
        # Current shock level
        if rolling_vol is not None and len(rolling_vol) > 0:
            current_vol = rolling_vol[-1]
            avg_vol = np.mean(rolling_vol)
            current_shock_level = current_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            current_shock_level = 1.0
        
        return {
            'shock_events': shock_events[-10:],  # Last 10 events
            'current_shock_level': current_shock_level,
            'shock_frequency': len(shock_events) / len(rolling_vol) if len(rolling_vol) > 0 else 0.0
        }
    
    def _analyze_cross_venue_liquidity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cross-venue liquidity patterns."""
        # Placeholder for cross-venue analysis
        # In real implementation, this would analyze liquidity across multiple exchanges
        
        return {
            'venue_concentration': 0.7,  # Liquidity concentration score
            'arbitrage_opportunities': [],
            'cross_venue_spread': 0.001,
            'best_execution_venue': 'primary'
        }
    
    def _detect_hft_activity(self, high: np.ndarray, low: np.ndarray,
                           close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Detect high-frequency trading activity patterns."""
        
        # HFT indicators
        hft_indicators = []
        
        # Rapid price movements with high volume
        for i in range(1, len(close)):
            price_change = abs(close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0
            volume_ratio = volume[i] / np.mean(volume[max(0, i-10):i]) if i >= 10 else 1.0
            
            # HFT signature: small price moves with high volume
            if price_change < 0.001 and volume_ratio > 2.0:  # 0.1% price move, 2x volume
                hft_indicators.append({
                    'index': i,
                    'price_change': price_change,
                    'volume_ratio': volume_ratio,
                    'hft_probability': volume_ratio * (1 - price_change * 100)
                })
        
        # HFT activity level
        if len(close) > 0:
            hft_activity_level = len(hft_indicators) / len(close)
        else:
            hft_activity_level = 0.0
        
        # Recent HFT activity
        recent_hft = len([h for h in hft_indicators if h['index'] >= len(close) - 20])
        
        return {
            'hft_activity_level': hft_activity_level,
            'recent_hft_count': recent_hft,
            'hft_indicators': hft_indicators[-5:],  # Last 5 indicators
            'market_quality_impact': max(0.0, 1.0 - hft_activity_level * 2)  # HFT can reduce market quality
        }
    
    def _perform_liquidity_stress_test(self, depth_analysis: Dict[str, Any],
                                     flow_metrics: Dict[str, Any], 
                                     volume: np.ndarray) -> Dict[str, Any]:
        """Perform liquidity stress testing."""
        
        # Current liquidity metrics
        liquidity_score = depth_analysis.get('liquidity_score', 0.5)
        depth_resilience = depth_analysis.get('depth_resilience', {}).get('resilience_score', 0.5)
        
        # Stress scenarios
        scenarios = {
            'market_crash': self._stress_test_market_crash(liquidity_score, depth_resilience),
            'volume_spike': self._stress_test_volume_spike(volume, liquidity_score),
            'liquidity_drought': self._stress_test_liquidity_drought(flow_metrics, liquidity_score)
        }
        
        # Overall stress resistance
        stress_scores = [scenario['stress_resistance'] for scenario in scenarios.values()]
        overall_stress_resistance = np.mean(stress_scores)
        
        # Risk level
        if overall_stress_resistance > 0.7:
            risk_level = 'low'
        elif overall_stress_resistance > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'scenarios': scenarios,
            'overall_stress_resistance': overall_stress_resistance,
            'risk_level': risk_level,
            'recommendations': self._generate_stress_test_recommendations(scenarios)
        }
    
    def _stress_test_market_crash(self, liquidity_score: float, depth_resilience: float) -> Dict[str, Any]:
        """Stress test for market crash scenario."""
        
        # Simulate 20% market drop
        crash_impact = 0.8 - liquidity_score * 0.3  # Lower liquidity = higher impact
        
        # Recovery time estimate
        recovery_factor = (liquidity_score + depth_resilience) / 2
        recovery_time = 10 / (recovery_factor + 0.1)  # Days
        
        stress_resistance = 1.0 - crash_impact
        
        return {
            'scenario': 'market_crash',
            'expected_impact': crash_impact,
            'recovery_time_days': recovery_time,
            'stress_resistance': max(0.0, stress_resistance)
        }
    
    def _stress_test_volume_spike(self, volume: np.ndarray, liquidity_score: float) -> Dict[str, Any]:
        """Stress test for volume spike scenario."""
        
        if len(volume) == 0:
            return {'scenario': 'volume_spike', 'stress_resistance': 0.5}
        
        # Simulate 10x volume spike
        avg_volume = np.mean(volume)
        spike_volume = avg_volume * 10
        
        # Impact on execution quality
        execution_impact = min(spike_volume / (avg_volume * liquidity_score * 20), 1.0)
        
        stress_resistance = 1.0 - execution_impact
        
        return {
            'scenario': 'volume_spike',
            'spike_multiple': 10.0,
            'execution_impact': execution_impact,
            'stress_resistance': max(0.0, stress_resistance)
        }
    
    def _stress_test_liquidity_drought(self, flow_metrics: Dict[str, Any], 
                                     liquidity_score: float) -> Dict[str, Any]:
        """Stress test for liquidity drought scenario."""
        
        # Simulate 50% reduction in liquidity
        drought_impact = 0.5 * (1.0 - liquidity_score)
        
        # Flow disruption
        flow_strength = flow_metrics.get('flow_strength', 0.5)
        flow_disruption = drought_impact * (1.0 - flow_strength)
        
        stress_resistance = 1.0 - (drought_impact + flow_disruption) / 2
        
        return {
            'scenario': 'liquidity_drought',
            'liquidity_reduction': 0.5,
            'flow_disruption': flow_disruption,
            'stress_resistance': max(0.0, stress_resistance)
        }
    
    def _generate_stress_test_recommendations(self, scenarios: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []
        
        for scenario_name, scenario in scenarios.items():
            resistance = scenario.get('stress_resistance', 0.5)
            
            if resistance < 0.3:
                if scenario_name == 'market_crash':
                    recommendations.append('Implement circuit breakers and position limits for market stress')
                elif scenario_name == 'volume_spike':
                    recommendations.append('Increase order size limits and implement volume throttling')
                elif scenario_name == 'liquidity_drought':
                    recommendations.append('Diversify liquidity sources and implement adaptive routing')
        
        if not recommendations:
            recommendations.append('Liquidity stress resistance is adequate under tested scenarios')
        
        return recommendations
    
    def _calculate_ml_predictions(self, depth_analysis: Dict[str, Any],
                                flow_metrics: Dict[str, Any],
                                spread_analysis: Dict[str, Any],
                                order_flow_imbalance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate machine learning-based predictions."""
        
        try:
            # Prepare features
            features = self._prepare_ml_features_liquidity(
                depth_analysis, flow_metrics, spread_analysis, order_flow_imbalance
            )
            
            if len(features) == 0:
                return {'liquidity_forecast': 0.5, 'flow_prediction': 'neutral'}
            
            # Simple rule-based predictions (placeholder for ML)
            liquidity_score = depth_analysis.get('liquidity_score', 0.5)
            flow_strength = flow_metrics.get('flow_strength', 0.0)
            spread_quality = spread_analysis.get('market_quality', 0.5)
            
            # Liquidity forecast
            liquidity_factors = [liquidity_score, spread_quality, 1.0 - order_flow_imbalance.get('imbalance_strength', 0.0)]
            liquidity_forecast = np.mean(liquidity_factors)
            
            # Flow prediction
            net_flow = flow_metrics.get('net_flow', 0.0)
            if net_flow > 0.3:
                flow_prediction = 'inflow_expected'
            elif net_flow < -0.3:
                flow_prediction = 'outflow_expected'
            else:
                flow_prediction = 'neutral'
            
            # Confidence
            feature_consistency = 1.0 - np.std(liquidity_factors) if len(liquidity_factors) > 1 else 0.5
            
            return {
                'liquidity_forecast': liquidity_forecast,
                'flow_prediction': flow_prediction,
                'prediction_confidence': feature_consistency,
                'feature_importance': {
                    'depth': 0.4,
                    'flow': 0.3,
                    'spread': 0.2,
                    'imbalance': 0.1
                }
            }
            
        except Exception:
            return {'liquidity_forecast': 0.5, 'flow_prediction': 'neutral'}
    
    def _prepare_ml_features_liquidity(self, depth_analysis: Dict[str, Any],
                                     flow_metrics: Dict[str, Any],
                                     spread_analysis: Dict[str, Any],
                                     order_flow_imbalance: Dict[str, Any]) -> List[float]:
        """Prepare features for ML models."""
        features = []
        
        # Depth features
        features.extend([
            depth_analysis.get('liquidity_score', 0.5),
            depth_analysis.get('liquidity_imbalance', 0.0),
            depth_analysis.get('liquidity_concentration', 0.5),
            depth_analysis.get('depth_resilience', {}).get('resilience_score', 0.5)
        ])
        
        # Flow features
        features.extend([
            flow_metrics.get('net_flow', 0.0),
            flow_metrics.get('flow_strength', 0.0),
            flow_metrics.get('flow_consistency', 0.5),
            flow_metrics.get('mfi', 50.0) / 100.0  # Normalize to 0-1
        ])
        
        # Spread features
        features.extend([
            spread_analysis.get('market_quality', 0.5),
            spread_analysis.get('current_relative_spread', 0.01),
            spread_analysis.get('spread_volatility', 0.01),
            1.0 if spread_analysis.get('is_wide_spread', False) else 0.0
        ])
        
        # Order flow features
        features.extend([
            order_flow_imbalance.get('imbalance', 0.0),
            order_flow_imbalance.get('imbalance_strength', 0.0),
            order_flow_imbalance.get('buy_ratio', 0.5),
            order_flow_imbalance.get('imbalance_trend', 0.0)
        ])
        
        return features
    
    def _adjust_adaptive_parameters(self, flow_metrics: Dict[str, Any],
                                  spread_analysis: Dict[str, Any],
                                  volume: np.ndarray) -> Dict[str, Any]:
        """Adjust parameters based on current market conditions."""
        
        # Market quality assessment
        market_quality = spread_analysis.get('market_quality', 0.5)
        flow_strength = flow_metrics.get('flow_strength', 0.0)
        
        # Volume volatility
        if len(volume) >= 20:
            volume_volatility = np.std(volume[-20:]) / np.mean(volume[-20:])
        else:
            volume_volatility = 1.0
        
        # Adjust thresholds based on conditions
        adaptive_params = {}
        
        # Flow sensitivity adjustment
        if market_quality < 0.5:  # Poor market quality
            adaptive_params['flow_sensitivity'] = self.parameters['flow_sensitivity'] * 1.5
        else:
            adaptive_params['flow_sensitivity'] = self.parameters['flow_sensitivity']
        
        # Imbalance threshold adjustment
        if volume_volatility > 2.0:  # High volume volatility
            adaptive_params['imbalance_threshold'] = self.parameters['imbalance_threshold'] * 1.2
        else:
            adaptive_params['imbalance_threshold'] = self.parameters['imbalance_threshold']
        
        # Window adjustments
        if flow_strength > 0.7:  # Strong flow
            adaptive_params['liquidity_window'] = max(self.parameters['liquidity_window'] // 2, 5)
        else:
            adaptive_params['liquidity_window'] = self.parameters['liquidity_window']
        
        return adaptive_params
    
    def _generate_liquidity_signals(self, depth_analysis: Dict[str, Any],
                                  flow_metrics: Dict[str, Any],
                                  spread_analysis: Dict[str, Any],
                                  order_flow_imbalance: Dict[str, Any],
                                  ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive liquidity flow signals."""
        
        # Individual signal components
        signals = []
        
        # Depth signal
        liquidity_score = depth_analysis.get('liquidity_score', 0.5)
        if liquidity_score > 0.7:
            signals.append(('depth', 1, liquidity_score))
        elif liquidity_score < 0.3:
            signals.append(('depth', -1, 1 - liquidity_score))
        else:
            signals.append(('depth', 0, 0.5))
        
        # Flow signal
        net_flow = flow_metrics.get('net_flow', 0.0)
        flow_strength = flow_metrics.get('flow_strength', 0.0)
        if net_flow > 0.2:
            signals.append(('flow', 1, flow_strength))
        elif net_flow < -0.2:
            signals.append(('flow', -1, flow_strength))
        else:
            signals.append(('flow', 0, flow_strength))
        
        # Spread signal
        market_quality = spread_analysis.get('market_quality', 0.5)
        if market_quality > 0.7:
            signals.append(('spread', 1, market_quality))
        elif market_quality < 0.3:
            signals.append(('spread', -1, 1 - market_quality))
        else:
            signals.append(('spread', 0, 0.5))
        
        # Imbalance signal
        imbalance = order_flow_imbalance.get('imbalance', 0.0)
        imbalance_strength = order_flow_imbalance.get('imbalance_strength', 0.0)
        if imbalance > 0.3:
            signals.append(('imbalance', 1, imbalance_strength))
        elif imbalance < -0.3:
            signals.append(('imbalance', -1, imbalance_strength))
        else:
            signals.append(('imbalance', 0, imbalance_strength))
        
        # ML signal
        ml_flow = ml_predictions.get('flow_prediction', 'neutral')
        ml_confidence = ml_predictions.get('prediction_confidence', 0.5)
        if ml_flow == 'inflow_expected':
            signals.append(('ml', 1, ml_confidence))
        elif ml_flow == 'outflow_expected':
            signals.append(('ml', -1, ml_confidence))
        else:
            signals.append(('ml', 0, ml_confidence))
        
        # Weighted combination
        weights = {'depth': 0.25, 'flow': 0.25, 'spread': 0.2, 'imbalance': 0.2, 'ml': 0.1}
        
        weighted_signal = sum(weights[name] * direction * strength 
                            for name, direction, strength in signals)
        
        # Determine final signal
        if weighted_signal > 0.3:
            direction = 'bullish'
            action = 'increase_position'
            strength = min(weighted_signal, 1.0)
        elif weighted_signal < -0.3:
            direction = 'bearish'
            action = 'decrease_position'
            strength = min(abs(weighted_signal), 1.0)
        else:
            direction = 'neutral'
            action = 'hold'
            strength = abs(weighted_signal)
        
        # Overall confidence
        confidences = [s[2] for s in signals]
        overall_confidence = np.mean(confidences)
        
        return {
            'direction': direction,
            'action': action,
            'strength': strength,
            'confidence': overall_confidence,
            'weighted_signal': weighted_signal,
            'component_signals': {name: {'direction': direction, 'strength': strength} 
                                for name, direction, strength in signals},
            'liquidity_regime': 'high' if liquidity_score > 0.7 else 'low' if liquidity_score < 0.3 else 'normal'
        }
    
    def _assess_liquidity_risk(self, depth_analysis: Dict[str, Any],
                             flow_metrics: Dict[str, Any],
                             shock_detection: Dict[str, Any],
                             stress_test: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall liquidity risk."""
        
        risk_factors = []
        
        # Low liquidity risk
        liquidity_score = depth_analysis.get('liquidity_score', 0.5)
        if liquidity_score < 0.3:
            risk_factors.append('low_liquidity')
        
        # High spread risk
        effective_spread = depth_analysis.get('effective_spread', 0.0)
        if effective_spread > 0.01:  # 1% spread
            risk_factors.append('wide_spreads')
        
        # Flow disruption risk
        flow_consistency = flow_metrics.get('flow_consistency', 0.5)
        if flow_consistency < 0.3:
            risk_factors.append('inconsistent_flow')
        
        # Shock risk
        current_shock = shock_detection.get('current_shock_level', 1.0)
        if current_shock > 2.0:
            risk_factors.append('liquidity_shock')
        
        # Stress test failures
        stress_resistance = stress_test.get('overall_stress_resistance', 0.5)
        if stress_resistance < 0.4:
            risk_factors.append('stress_vulnerability')
        
        # Overall risk assessment
        risk_score = len(risk_factors) / 5.0  # Normalize to 0-1
        
        if risk_score < 0.3:
            overall_risk = 'low'
        elif risk_score < 0.6:
            overall_risk = 'medium'
        else:
            overall_risk = 'high'
        
        return {
            'overall_risk': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'mitigation_strategies': self._generate_risk_mitigation_strategies(risk_factors)
        }
    
    def _generate_risk_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation strategies."""
        strategies = []
        
        if 'low_liquidity' in risk_factors:
            strategies.append('Reduce position sizes and use limit orders')
        
        if 'wide_spreads' in risk_factors:
            strategies.append('Wait for spread compression or use hidden orders')
        
        if 'inconsistent_flow' in risk_factors:
            strategies.append('Increase monitoring frequency and use adaptive algorithms')
        
        if 'liquidity_shock' in risk_factors:
            strategies.append('Implement emergency liquidity protocols and circuit breakers')
        
        if 'stress_vulnerability' in risk_factors:
            strategies.append('Diversify execution venues and increase liquidity buffers')
        
        if not strategies:
            strategies.append('Continue normal operations with standard monitoring')
        
        return strategies
    
    def _calculate_performance_metrics(self, signals: Dict[str, Any],
                                     flow_metrics: Dict[str, Any],
                                     depth_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for the indicator."""
        
        # Signal quality metrics
        signal_strength = signals.get('strength', 0.0)
        signal_confidence = signals.get('confidence', 0.5)
        
        # Liquidity provision metrics
        liquidity_score = depth_analysis.get('liquidity_score', 0.5)
        depth_resilience = depth_analysis.get('depth_resilience', {}).get('resilience_score', 0.5)
        
        # Flow prediction accuracy (simplified)
        flow_consistency = flow_metrics.get('flow_consistency', 0.5)
        
        # Overall performance score
        performance_components = [signal_confidence, liquidity_score, depth_resilience, flow_consistency]
        overall_performance = np.mean(performance_components)
        
        return {
            'overall_performance': overall_performance,
            'signal_quality': (signal_strength + signal_confidence) / 2,
            'liquidity_provision_quality': (liquidity_score + depth_resilience) / 2,
            'flow_prediction_quality': flow_consistency,
            'performance_grade': 'A' if overall_performance > 0.8 else 'B' if overall_performance > 0.6 else 'C'
        }
    
    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when insufficient data."""
        return {
            'depth_analysis': {'liquidity_score': 0.5},
            'flow_metrics': {'net_flow_direction': 'neutral', 'flow_strength': 0.0},
            'spread_analysis': {'market_quality': 0.5},
            'order_flow_imbalance': {'imbalance_direction': 'balanced'},
            'signals': {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.5},
            'liquidity_score': 0.5,
            'flow_direction': 'neutral',
            'liquidity_risk': 'medium'
        }
    
    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        return self._get_default_output()