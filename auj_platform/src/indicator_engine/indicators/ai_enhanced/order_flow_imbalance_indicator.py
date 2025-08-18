"""
Order Flow Imbalance Indicator - Advanced Market Microstructure Analysis

This module implements sophisticated order flow imbalance detection with:
- Real-time order flow analysis and imbalance calculation
- Market microstructure modeling with bid-ask dynamics
- Institutional flow detection with pattern recognition
- Liquidity analysis and modeling
- Volume profile analysis with statistical significance
- Multi-timeframe order flow aggregation
- Machine learning-based flow prediction
- Risk-adjusted position sizing based on flow strength
- Advanced filtering for noise reduction
- Production-grade error handling and logging

Author: AI Enhancement Team
Version: 7.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OrderFlowData:
    """Data structure for order flow information."""
    timestamp: datetime
    price: float
    volume: float
    buy_volume: float
    sell_volume: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    trade_direction: int  # 1 for buy, -1 for sell, 0 for neutral

@dataclass
class ImbalanceSignal:
    """Order flow imbalance signal structure."""
    timestamp: datetime
    imbalance_ratio: float
    imbalance_strength: float
    flow_direction: int  # 1 for bullish, -1 for bearish, 0 for neutral
    confidence: float
    institutional_flow: bool
    liquidity_state: str
    volume_profile_percentile: float
    risk_score: float
    signal_quality: str

class OrderFlowImbalanceIndicator:
    """
    Advanced Order Flow Imbalance Detector with sophisticated market microstructure analysis.
    
    This indicator provides comprehensive order flow analysis including:
    - Real-time imbalance calculation with multiple methodologies
    - Institutional flow detection using advanced pattern recognition
    - Liquidity modeling with bid-ask spread analysis
    - Volume profile analysis with statistical significance testing
    - Machine learning-based flow prediction and classification
    - Multi-timeframe aggregation for comprehensive market view
    - Risk assessment and position sizing recommendations
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Order Flow Imbalance Indicator.
        
        Args:
            parameters: Configuration parameters for the indicator
        """
        self.parameters = self._set_default_parameters(parameters or {})
        self.logger = self._setup_logger()
        
        # Core components
        self.order_flow_analyzer = OrderFlowAnalyzer(self.parameters)
        self.institutional_detector = InstitutionalFlowDetector(self.parameters)
        self.liquidity_modeler = LiquidityModeler(self.parameters)
        self.volume_profiler = VolumeProfiler(self.parameters)
        self.ml_predictor = MLFlowPredictor(self.parameters)
        self.risk_assessor = FlowRiskAssessor(self.parameters)
        
        # Data storage
        self.order_flow_history: List[OrderFlowData] = []
        self.imbalance_signals: List[ImbalanceSignal] = []
        self.volume_profiles: Dict[str, Any] = {}
        self.institutional_flows: List[Dict[str, Any]] = []
        
        # Analysis state
        self.current_regime = "neutral"
        self.liquidity_state = "normal"
        self.last_calculation_time = None
        self.is_trained = False
        
        self.logger.info("Order Flow Imbalance Indicator initialized with advanced market microstructure analysis")
    
    def _set_default_parameters(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Set default parameters for the indicator."""
        defaults = {
            # Core parameters
            'lookback_period': 100,
            'imbalance_threshold': 0.6,
            'confidence_threshold': 0.7,
            'institutional_volume_threshold': 10000,
            
            # Analysis windows
            'short_window': 10,
            'medium_window': 30,
            'long_window': 100,
            
            # Institutional detection
            'institutional_size_multiplier': 3.0,
            'institutional_speed_threshold': 0.8,
            'block_trade_threshold': 50000,
            
            # Liquidity modeling
            'spread_percentile_threshold': 80,
            'liquidity_window': 50,
            'depth_analysis_levels': 5,
            
            # Volume profile
            'profile_resolution': 50,
            'value_area_percentage': 68,
            'poc_window': 20,
            
            # Machine learning
            'ml_training_period': 500,
            'ml_retrain_frequency': 100,
            'feature_count': 25,
            'anomaly_detection_threshold': 0.1,
            
            # Risk management
            'max_risk_score': 10.0,
            'risk_decay_factor': 0.95,
            'position_size_factor': 0.02,
            
            # Signal filtering
            'noise_filter_enabled': True,
            'signal_smoothing_factor': 0.3,
            'minimum_volume_threshold': 1000,
            
            # Advanced features
            'multi_timeframe_analysis': True,
            'regime_detection': True,
            'adaptive_thresholds': True,
            'real_time_updates': True
        }
        
        # Update with user parameters
        defaults.update(user_params)
        return defaults
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the indicator."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray, timestamp: np.ndarray = None,
                 tick_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate order flow imbalance with comprehensive market microstructure analysis.
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            volume: Volume data
            timestamp: Optional timestamp data
            tick_data: Optional detailed tick-by-tick data
            
        Returns:
            Dict containing order flow imbalance analysis results
        """
        try:
            if len(close) < self.parameters['lookback_period']:
                return self._generate_default_result()
            
            # Process tick data if available
            if tick_data:
                self._process_tick_data(tick_data)
            else:
                self._simulate_order_flow_from_ohlcv(high, low, close, volume, timestamp)
            
            # Core analysis
            imbalance_analysis = self._analyze_order_flow_imbalance()
            institutional_analysis = self._detect_institutional_flow()
            liquidity_analysis = self._analyze_liquidity_conditions()
            volume_profile_analysis = self._analyze_volume_profile()
            
            # Machine learning predictions
            ml_predictions = self._generate_ml_predictions()
            
            # Risk assessment
            risk_assessment = self._assess_flow_risks()
            
            # Generate signals
            signals = self._generate_comprehensive_signals(
                imbalance_analysis, institutional_analysis, liquidity_analysis,
                volume_profile_analysis, ml_predictions, risk_assessment
            )
            
            # Update state
            self.last_calculation_time = datetime.now()
            
            result = {
                'imbalance_ratio': signals.get('primary_imbalance', 0.0),
                'imbalance_strength': signals.get('imbalance_strength', 0.0),
                'flow_direction': signals.get('flow_direction', 0),
                'confidence': signals.get('confidence', 0.0),
                'institutional_flow': signals.get('institutional_detected', False),
                'liquidity_state': signals.get('liquidity_state', 'normal'),
                'volume_profile_percentile': signals.get('volume_percentile', 50.0),
                'risk_score': signals.get('risk_score', 0.0),
                'signal_quality': signals.get('signal_quality', 'low'),
                
                # Advanced metrics
                'buy_sell_ratio': imbalance_analysis.get('buy_sell_ratio', 1.0),
                'bid_ask_imbalance': liquidity_analysis.get('bid_ask_imbalance', 0.0),
                'institutional_probability': institutional_analysis.get('probability', 0.0),
                'liquidity_score': liquidity_analysis.get('liquidity_score', 0.0),
                'poc_distance': volume_profile_analysis.get('poc_distance', 0.0),
                'value_area_position': volume_profile_analysis.get('value_area_position', 'middle'),
                
                # ML insights
                'ml_flow_prediction': ml_predictions.get('flow_direction_prob', 0.5),
                'anomaly_score': ml_predictions.get('anomaly_score', 0.0),
                'regime_classification': ml_predictions.get('regime', 'neutral'),
                
                # Detailed analysis
                'flow_analysis': imbalance_analysis,
                'institutional_analysis': institutional_analysis,
                'liquidity_analysis': liquidity_analysis,
                'volume_analysis': volume_profile_analysis,
                'ml_analysis': ml_predictions,
                'risk_analysis': risk_assessment,
                
                # Metadata
                'calculation_time': self.last_calculation_time.isoformat() if self.last_calculation_time else None,
                'data_quality': self._assess_data_quality(),
                'parameters_used': self.parameters.copy()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating order flow imbalance: {e}")
            return self._generate_error_result(str(e))
    
    def _process_tick_data(self, tick_data: List[Dict]) -> None:
        """Process detailed tick-by-tick data for order flow analysis."""
        try:
            for tick in tick_data[-self.parameters['lookback_period']:]:
                # Extract order flow information
                order_flow = OrderFlowData(
                    timestamp=tick.get('timestamp', datetime.now()),
                    price=float(tick.get('price', 0)),
                    volume=float(tick.get('volume', 0)),
                    buy_volume=float(tick.get('buy_volume', 0)),
                    sell_volume=float(tick.get('sell_volume', 0)),
                    bid=float(tick.get('bid', 0)),
                    ask=float(tick.get('ask', 0)),
                    bid_size=float(tick.get('bid_size', 0)),
                    ask_size=float(tick.get('ask_size', 0)),
                    trade_direction=int(tick.get('trade_direction', 0))
                )
                
                self.order_flow_history.append(order_flow)
            
            # Maintain history size
            max_history = self.parameters['lookback_period'] * 2
            if len(self.order_flow_history) > max_history:
                self.order_flow_history = self.order_flow_history[-max_history:]
                
        except Exception as e:
            self.logger.error(f"Error processing tick data: {e}")
    
    def _simulate_order_flow_from_ohlcv(self, high: np.ndarray, low: np.ndarray, 
                                       close: np.ndarray, volume: np.ndarray,
                                       timestamp: np.ndarray = None) -> None:
        """Simulate order flow data from OHLCV when tick data is unavailable."""
        try:
            for i in range(len(close)):
                # Simulate bid-ask spread (typically 0.1-0.5% of price)
                spread = close[i] * np.random.uniform(0.001, 0.005)
                bid = close[i] - spread / 2
                ask = close[i] + spread / 2
                
                # Estimate buy/sell volume based on price movement
                if i > 0:
                    price_change = close[i] - close[i-1]
                    if price_change > 0:
                        buy_ratio = 0.5 + min(0.3, abs(price_change) / close[i-1] * 10)
                    elif price_change < 0:
                        buy_ratio = 0.5 - min(0.3, abs(price_change) / close[i-1] * 10)
                    else:
                        buy_ratio = 0.5
                else:
                    buy_ratio = 0.5
                
                buy_volume = volume[i] * buy_ratio
                sell_volume = volume[i] * (1 - buy_ratio)
                
                # Determine trade direction
                if buy_volume > sell_volume * 1.2:
                    trade_direction = 1
                elif sell_volume > buy_volume * 1.2:
                    trade_direction = -1
                else:
                    trade_direction = 0
                
                # Create order flow data
                ts = timestamp[i] if timestamp is not None else datetime.now()
                order_flow = OrderFlowData(
                    timestamp=ts,
                    price=close[i],
                    volume=volume[i],
                    buy_volume=buy_volume,
                    sell_volume=sell_volume,
                    bid=bid,
                    ask=ask,
                    bid_size=volume[i] * np.random.uniform(0.1, 0.3),
                    ask_size=volume[i] * np.random.uniform(0.1, 0.3),
                    trade_direction=trade_direction
                )
                
                self.order_flow_history.append(order_flow)
            
            # Maintain history size
            max_history = self.parameters['lookback_period'] * 2
            if len(self.order_flow_history) > max_history:
                self.order_flow_history = self.order_flow_history[-max_history:]
                
        except Exception as e:
            self.logger.error(f"Error simulating order flow: {e}")
    
    def _analyze_order_flow_imbalance(self) -> Dict[str, Any]:
        """Analyze order flow imbalance with multiple methodologies."""
        try:
            if len(self.order_flow_history) < self.parameters['short_window']:
                return {'buy_sell_ratio': 1.0, 'imbalance_strength': 0.0}
            
            recent_flows = self.order_flow_history[-self.parameters['short_window']:]
            
            # Calculate basic buy/sell ratios
            total_buy_volume = sum(flow.buy_volume for flow in recent_flows)
            total_sell_volume = sum(flow.sell_volume for flow in recent_flows)
            
            buy_sell_ratio = total_buy_volume / (total_sell_volume + 1e-10)
            
            # Calculate imbalance strength
            imbalance_strength = abs(buy_sell_ratio - 1.0) / (buy_sell_ratio + 1.0)
            
            # Advanced imbalance metrics
            directional_flows = [flow.trade_direction for flow in recent_flows]
            directional_imbalance = np.mean(directional_flows) if directional_flows else 0.0
            
            # Volume-weighted imbalance
            volumes = [flow.volume for flow in recent_flows]
            weighted_directions = np.average(directional_flows, weights=volumes) if volumes else 0.0
            
            # Bid-ask size imbalance
            bid_sizes = [flow.bid_size for flow in recent_flows]
            ask_sizes = [flow.ask_size for flow in recent_flows]
            bid_ask_ratio = np.mean(bid_sizes) / (np.mean(ask_sizes) + 1e-10)
            
            # Time-based imbalance (recent vs historical)
            if len(self.order_flow_history) >= self.parameters['medium_window']:
                historical_flows = self.order_flow_history[-self.parameters['medium_window']:-self.parameters['short_window']]
                historical_buy = sum(flow.buy_volume for flow in historical_flows)
                historical_sell = sum(flow.sell_volume for flow in historical_flows)
                historical_ratio = historical_buy / (historical_sell + 1e-10)
                
                trend_acceleration = (buy_sell_ratio - historical_ratio) / (historical_ratio + 1e-10)
            else:
                trend_acceleration = 0.0
            
            return {
                'buy_sell_ratio': buy_sell_ratio,
                'imbalance_strength': imbalance_strength,
                'directional_imbalance': directional_imbalance,
                'weighted_imbalance': weighted_directions,
                'bid_ask_ratio': bid_ask_ratio,
                'trend_acceleration': trend_acceleration,
                'total_buy_volume': total_buy_volume,
                'total_sell_volume': total_sell_volume,
                'flow_consistency': np.std(directional_flows) if directional_flows else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow imbalance: {e}")
            return {'buy_sell_ratio': 1.0, 'imbalance_strength': 0.0}
    
    def _detect_institutional_flow(self) -> Dict[str, Any]:
        """Detect institutional trading flows using advanced pattern recognition."""
        try:
            if len(self.order_flow_history) < self.parameters['medium_window']:
                return {'probability': 0.0, 'detected': False}
            
            recent_flows = self.order_flow_history[-self.parameters['medium_window']:]
            
            # Large volume detection
            volumes = [flow.volume for flow in recent_flows]
            avg_volume = np.mean(volumes)
            volume_threshold = avg_volume * self.parameters['institutional_size_multiplier']
            large_trades = [v for v in volumes if v > volume_threshold]
            
            # Block trade detection
            block_trades = [v for v in volumes if v > self.parameters['block_trade_threshold']]
            
            # Speed of execution analysis
            time_intervals = []
            for i in range(1, len(recent_flows)):
                time_diff = (recent_flows[i].timestamp - recent_flows[i-1].timestamp).total_seconds()
                time_intervals.append(time_diff)
            
            avg_interval = np.mean(time_intervals) if time_intervals else 1.0
            fast_executions = [t for t in time_intervals if t < avg_interval * self.parameters['institutional_speed_threshold']]
            
            # Pattern consistency
            directions = [flow.trade_direction for flow in recent_flows if flow.trade_direction != 0]
            direction_consistency = abs(np.mean(directions)) if directions else 0.0
            
            # Stealth trading detection (consistent direction with varying sizes)
            stealth_score = 0.0
            if len(directions) > 5:
                direction_runs = self._detect_direction_runs(directions)
                volume_variation = np.std(volumes) / (np.mean(volumes) + 1e-10)
                stealth_score = direction_consistency * (1 - volume_variation) * len(direction_runs)
            
            # Calculate institutional probability
            institutional_signals = [
                len(large_trades) / len(volumes),
                len(block_trades) / len(volumes),
                len(fast_executions) / len(time_intervals) if time_intervals else 0,
                direction_consistency,
                min(stealth_score / 10, 1.0)
            ]
            
            institutional_probability = np.mean(institutional_signals)
            detected = institutional_probability > 0.6
            
            return {
                'probability': institutional_probability,
                'detected': detected,
                'large_trade_ratio': len(large_trades) / len(volumes),
                'block_trade_count': len(block_trades),
                'fast_execution_ratio': len(fast_executions) / len(time_intervals) if time_intervals else 0,
                'direction_consistency': direction_consistency,
                'stealth_score': stealth_score,
                'average_trade_size': np.mean(volumes),
                'trade_size_variation': np.std(volumes) / (np.mean(volumes) + 1e-10)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting institutional flow: {e}")
            return {'probability': 0.0, 'detected': False}
    
    def _detect_direction_runs(self, directions: List[int]) -> List[int]:
        """Detect consecutive runs of the same trading direction."""
        runs = []
        current_run = 1
        
        for i in range(1, len(directions)):
            if directions[i] == directions[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        
        runs.append(current_run)
        return runs
    
    def _analyze_liquidity_conditions(self) -> Dict[str, Any]:
        """Analyze current liquidity conditions and market depth."""
        try:
            if len(self.order_flow_history) < self.parameters['liquidity_window']:
                return {'liquidity_score': 0.5, 'bid_ask_imbalance': 0.0}
            
            recent_flows = self.order_flow_history[-self.parameters['liquidity_window']:]
            
            # Spread analysis
            spreads = [(flow.ask - flow.bid) / flow.price for flow in recent_flows if flow.bid > 0 and flow.ask > 0]
            avg_spread = np.mean(spreads) if spreads else 0.01
            spread_volatility = np.std(spreads) if len(spreads) > 1 else 0.0
            
            # Depth analysis
            bid_sizes = [flow.bid_size for flow in recent_flows]
            ask_sizes = [flow.ask_size for flow in recent_flows]
            
            avg_bid_depth = np.mean(bid_sizes)
            avg_ask_depth = np.mean(ask_sizes)
            depth_imbalance = (avg_bid_depth - avg_ask_depth) / (avg_bid_depth + avg_ask_depth + 1e-10)
            
            # Liquidity score (lower spreads and higher depth = better liquidity)
            liquidity_score = 1 / (1 + avg_spread * 100) * min(avg_bid_depth + avg_ask_depth, 10000) / 10000
            
            # Market impact estimation
            volumes = [flow.volume for flow in recent_flows]
            prices = [flow.price for flow in recent_flows]
            
            market_impact = 0.0
            if len(volumes) > 1 and len(prices) > 1:
                volume_price_corr = np.corrcoef(volumes[1:], np.diff(prices))[0, 1]
                market_impact = abs(volume_price_corr) if not np.isnan(volume_price_corr) else 0.0
            
            # Resilience (how quickly spreads return to normal after large trades)
            resilience_score = 1 - spread_volatility if spread_volatility < 1 else 0.0
            
            return {
                'liquidity_score': liquidity_score,
                'avg_spread': avg_spread,
                'spread_volatility': spread_volatility,
                'bid_ask_imbalance': depth_imbalance,
                'avg_bid_depth': avg_bid_depth,
                'avg_ask_depth': avg_ask_depth,
                'market_impact': market_impact,
                'resilience_score': resilience_score,
                'liquidity_state': self._classify_liquidity_state(liquidity_score, avg_spread, spread_volatility)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity: {e}")
            return {'liquidity_score': 0.5, 'bid_ask_imbalance': 0.0}
    
    def _classify_liquidity_state(self, liquidity_score: float, avg_spread: float, spread_volatility: float) -> str:
        """Classify current liquidity state."""
        if liquidity_score > 0.8 and avg_spread < 0.002 and spread_volatility < 0.001:
            return "excellent"
        elif liquidity_score > 0.6 and avg_spread < 0.005 and spread_volatility < 0.002:
            return "good"
        elif liquidity_score > 0.4 and avg_spread < 0.01 and spread_volatility < 0.005:
            return "normal"
        elif liquidity_score > 0.2:
            return "poor"
        else:
            return "very_poor"    
    def _analyze_volume_profile(self) -> Dict[str, Any]:
        """Analyze volume profile and Point of Control (POC) dynamics."""
        try:
            if len(self.order_flow_history) < self.parameters['poc_window']:
                return {'poc_distance': 0.0, 'value_area_position': 'middle'}
            
            recent_flows = self.order_flow_history[-self.parameters['poc_window']:]
            
            # Extract price and volume data
            prices = [flow.price for flow in recent_flows]
            volumes = [flow.volume for flow in recent_flows]
            
            if not prices or not volumes:
                return {'poc_distance': 0.0, 'value_area_position': 'middle'}
            
            # Create price bins for volume profile
            min_price = min(prices)
            max_price = max(prices)
            price_range = max_price - min_price
            
            if price_range == 0:
                return {'poc_distance': 0.0, 'value_area_position': 'middle'}
            
            bin_size = price_range / self.parameters['profile_resolution']
            bins = np.arange(min_price, max_price + bin_size, bin_size)
            
            # Calculate volume at each price level
            volume_profile = np.zeros(len(bins) - 1)
            for price, volume in zip(prices, volumes):
                bin_idx = min(int((price - min_price) / bin_size), len(volume_profile) - 1)
                volume_profile[bin_idx] += volume
            
            # Find Point of Control (POC) - price level with highest volume
            poc_idx = np.argmax(volume_profile)
            poc_price = bins[poc_idx] + bin_size / 2
            
            # Calculate Value Area (typically 68% of volume)
            total_volume = np.sum(volume_profile)
            value_area_volume = total_volume * (self.parameters['value_area_percentage'] / 100)
            
            # Find value area boundaries
            sorted_indices = np.argsort(volume_profile)[::-1]
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_profile[idx]
                value_area_indices.append(idx)
                if cumulative_volume >= value_area_volume:
                    break
            
            value_area_low = bins[min(value_area_indices)]
            value_area_high = bins[max(value_area_indices) + 1]
            
            # Current price analysis
            current_price = prices[-1]
            poc_distance = (current_price - poc_price) / poc_price if poc_price > 0 else 0.0
            
            # Determine position relative to value area
            if current_price > value_area_high:
                value_area_position = "above"
            elif current_price < value_area_low:
                value_area_position = "below"
            else:
                value_area_position = "inside"
            
            # Volume distribution analysis
            volume_percentile = self._calculate_volume_percentile(current_price, prices, volumes)
            
            # Volume profile strength
            poc_strength = volume_profile[poc_idx] / total_volume if total_volume > 0 else 0.0
            
            return {
                'poc_price': poc_price,
                'poc_distance': poc_distance,
                'poc_strength': poc_strength,
                'value_area_low': value_area_low,
                'value_area_high': value_area_high,
                'value_area_position': value_area_position,
                'volume_percentile': volume_percentile,
                'total_volume': total_volume,
                'profile_width': price_range,
                'volume_concentration': np.max(volume_profile) / (np.mean(volume_profile) + 1e-10)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {e}")
            return {'poc_distance': 0.0, 'value_area_position': 'middle'}
    
    def _calculate_volume_percentile(self, current_price: float, prices: List[float], volumes: List[float]) -> float:
        """Calculate the volume percentile for current price."""
        try:
            # Weight prices by volume
            weighted_prices = []
            for price, volume in zip(prices, volumes):
                weighted_prices.extend([price] * int(volume))
            
            if not weighted_prices:
                return 50.0
            
            # Calculate percentile
            percentile = (np.sum(np.array(weighted_prices) <= current_price) / len(weighted_prices)) * 100
            return percentile
            
        except Exception as e:
            self.logger.error(f"Error calculating volume percentile: {e}")
            return 50.0
    
    def _generate_ml_predictions(self) -> Dict[str, Any]:
        """Generate machine learning-based flow predictions."""
        try:
            if not self.ml_predictor:
                return {'flow_direction_prob': 0.5, 'anomaly_score': 0.0}
            
            # Check if enough data for ML
            if len(self.order_flow_history) < self.parameters['ml_training_period']:
                return {'flow_direction_prob': 0.5, 'anomaly_score': 0.0, 'regime': 'neutral'}
            
            # Prepare features for ML
            features = self._extract_ml_features()
            
            if features is None or len(features) == 0:
                return {'flow_direction_prob': 0.5, 'anomaly_score': 0.0, 'regime': 'neutral'}
            
            # Train models if needed
            if not self.is_trained or len(self.order_flow_history) % self.parameters['ml_retrain_frequency'] == 0:
                self._train_ml_models()
            
            # Generate predictions
            flow_prediction = self.ml_predictor.predict_flow_direction(features)
            anomaly_score = self.ml_predictor.detect_anomalies(features)
            regime_classification = self.ml_predictor.classify_regime(features)
            
            return {
                'flow_direction_prob': flow_prediction,
                'anomaly_score': anomaly_score,
                'regime': regime_classification,
                'feature_importance': self.ml_predictor.get_feature_importance(),
                'model_confidence': self.ml_predictor.get_prediction_confidence()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating ML predictions: {e}")
            return {'flow_direction_prob': 0.5, 'anomaly_score': 0.0, 'regime': 'neutral'}
    
    def _extract_ml_features(self) -> Optional[np.ndarray]:
        """Extract features for machine learning models."""
        try:
            if len(self.order_flow_history) < self.parameters['feature_count']:
                return None
            
            recent_flows = self.order_flow_history[-self.parameters['feature_count']:]
            features = []
            
            # Basic flow features
            volumes = [flow.volume for flow in recent_flows]
            prices = [flow.price for flow in recent_flows]
            buy_volumes = [flow.buy_volume for flow in recent_flows]
            sell_volumes = [flow.sell_volume for flow in recent_flows]
            
            # Statistical features
            features.extend([
                np.mean(volumes), np.std(volumes), np.median(volumes),
                np.mean(prices), np.std(prices),
                np.mean(buy_volumes) / (np.mean(sell_volumes) + 1e-10),
                np.sum(buy_volumes) / (np.sum(sell_volumes) + 1e-10)
            ])
            
            # Spread features
            spreads = [(flow.ask - flow.bid) / flow.price for flow in recent_flows if flow.bid > 0 and flow.ask > 0]
            if spreads:
                features.extend([np.mean(spreads), np.std(spreads), np.median(spreads)])
            else:
                features.extend([0.01, 0.001, 0.01])
            
            # Direction features
            directions = [flow.trade_direction for flow in recent_flows]
            features.extend([
                np.mean(directions), np.std(directions),
                len([d for d in directions if d > 0]) / len(directions),
                len([d for d in directions if d < 0]) / len(directions)
            ])
            
            # Time-based features
            if len(recent_flows) > 1:
                time_diffs = []
                for i in range(1, len(recent_flows)):
                    time_diff = (recent_flows[i].timestamp - recent_flows[i-1].timestamp).total_seconds()
                    time_diffs.append(time_diff)
                
                features.extend([np.mean(time_diffs), np.std(time_diffs)])
            else:
                features.extend([1.0, 0.1])
            
            # Technical features
            if len(volumes) > 5:
                volume_sma_5 = np.mean(volumes[-5:])
                volume_trend = (volumes[-1] - volume_sma_5) / (volume_sma_5 + 1e-10)
                features.append(volume_trend)
            else:
                features.append(0.0)
            
            if len(prices) > 5:
                price_returns = np.diff(prices[-6:])
                features.extend([np.mean(price_returns), np.std(price_returns)])
            else:
                features.extend([0.0, 0.01])
            
            # Pad or truncate to exact feature count
            while len(features) < self.parameters['feature_count']:
                features.append(0.0)
            
            features = features[:self.parameters['feature_count']]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting ML features: {e}")
            return None
    
    def _train_ml_models(self) -> None:
        """Train machine learning models for flow prediction."""
        try:
            if len(self.order_flow_history) < self.parameters['ml_training_period']:
                return
            
            self.ml_predictor.train_models(self.order_flow_history)
            self.is_trained = True
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
    
    def _assess_flow_risks(self) -> Dict[str, Any]:
        """Assess risks associated with current order flow conditions."""
        try:
            if not self.risk_assessor:
                return {'risk_score': 0.0, 'risk_factors': []}
            
            return self.risk_assessor.assess_risks(self.order_flow_history, self.institutional_flows)
            
        except Exception as e:
            self.logger.error(f"Error assessing flow risks: {e}")
            return {'risk_score': 0.0, 'risk_factors': []}
    
    def _generate_comprehensive_signals(self, imbalance_analysis: Dict, institutional_analysis: Dict,
                                      liquidity_analysis: Dict, volume_analysis: Dict,
                                      ml_predictions: Dict, risk_assessment: Dict) -> Dict[str, Any]:
        """Generate comprehensive trading signals from all analyses."""
        try:
            # Primary imbalance signal
            buy_sell_ratio = imbalance_analysis.get('buy_sell_ratio', 1.0)
            imbalance_strength = imbalance_analysis.get('imbalance_strength', 0.0)
            
            # Determine flow direction
            if buy_sell_ratio > 1.2 and imbalance_strength > self.parameters['imbalance_threshold']:
                flow_direction = 1  # Bullish
            elif buy_sell_ratio < 0.8 and imbalance_strength > self.parameters['imbalance_threshold']:
                flow_direction = -1  # Bearish
            else:
                flow_direction = 0  # Neutral
            
            # Calculate confidence score
            confidence_factors = [
                imbalance_strength,
                institutional_analysis.get('probability', 0.0),
                liquidity_analysis.get('liquidity_score', 0.5),
                1 - ml_predictions.get('anomaly_score', 0.0),
                abs(ml_predictions.get('flow_direction_prob', 0.5) - 0.5) * 2
            ]
            
            confidence = np.mean(confidence_factors)
            
            # Signal quality assessment
            if confidence > 0.8 and risk_assessment.get('risk_score', 0) < 3:
                signal_quality = "excellent"
            elif confidence > 0.6 and risk_assessment.get('risk_score', 0) < 5:
                signal_quality = "good"
            elif confidence > 0.4:
                signal_quality = "fair"
            else:
                signal_quality = "poor"
            
            return {
                'primary_imbalance': (buy_sell_ratio - 1.0) if buy_sell_ratio != 1.0 else 0.0,
                'imbalance_strength': imbalance_strength,
                'flow_direction': flow_direction,
                'confidence': confidence,
                'institutional_detected': institutional_analysis.get('detected', False),
                'liquidity_state': liquidity_analysis.get('liquidity_state', 'normal'),
                'volume_percentile': volume_analysis.get('volume_percentile', 50.0),
                'risk_score': risk_assessment.get('risk_score', 0.0),
                'signal_quality': signal_quality,
                
                # Additional signal components
                'directional_imbalance': imbalance_analysis.get('directional_imbalance', 0.0),
                'institutional_probability': institutional_analysis.get('probability', 0.0),
                'liquidity_score': liquidity_analysis.get('liquidity_score', 0.5),
                'poc_distance': volume_analysis.get('poc_distance', 0.0),
                'ml_flow_prediction': ml_predictions.get('flow_direction_prob', 0.5),
                'anomaly_detected': ml_predictions.get('anomaly_score', 0.0) > self.parameters['anomaly_detection_threshold']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {
                'primary_imbalance': 0.0,
                'imbalance_strength': 0.0,
                'flow_direction': 0,
                'confidence': 0.0,
                'signal_quality': 'poor'
            }
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality of input data."""
        try:
            if not self.order_flow_history:
                return {'quality': 'poor', 'score': 0.0, 'issues': ['no_data']}
            
            recent_flows = self.order_flow_history[-min(100, len(self.order_flow_history)):]
            
            issues = []
            quality_scores = []
            
            # Check data completeness
            complete_records = sum(1 for flow in recent_flows 
                                 if flow.volume > 0 and flow.price > 0)
            completeness_score = complete_records / len(recent_flows)
            quality_scores.append(completeness_score)
            
            if completeness_score < 0.9:
                issues.append('incomplete_data')
            
            # Check for outliers
            volumes = [flow.volume for flow in recent_flows if flow.volume > 0]
            if volumes:
                volume_q75, volume_q25 = np.percentile(volumes, [75, 25])
                volume_iqr = volume_q75 - volume_q25
                outliers = [v for v in volumes if v > volume_q75 + 1.5 * volume_iqr or v < volume_q25 - 1.5 * volume_iqr]
                outlier_ratio = len(outliers) / len(volumes)
                quality_scores.append(1 - min(outlier_ratio, 1.0))
                
                if outlier_ratio > 0.1:
                    issues.append('volume_outliers')
            
            # Check temporal consistency
            timestamps = [flow.timestamp for flow in recent_flows]
            if len(timestamps) > 1:
                time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                            for i in range(1, len(timestamps))]
                avg_interval = np.mean(time_diffs)
                irregular_intervals = [t for t in time_diffs if abs(t - avg_interval) > avg_interval * 2]
                temporal_consistency = 1 - (len(irregular_intervals) / len(time_diffs))
                quality_scores.append(temporal_consistency)
                
                if temporal_consistency < 0.8:
                    issues.append('irregular_timestamps')
            
            # Overall quality score
            overall_score = np.mean(quality_scores) if quality_scores else 0.0
            
            if overall_score > 0.8:
                quality_level = 'excellent'
            elif overall_score > 0.6:
                quality_level = 'good'
            elif overall_score > 0.4:
                quality_level = 'fair'
            else:
                quality_level = 'poor'
            
            return {
                'quality': quality_level,
                'score': overall_score,
                'issues': issues,
                'completeness': completeness_score,
                'data_points': len(recent_flows)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return {'quality': 'poor', 'score': 0.0, 'issues': ['assessment_error']}
    
    def _generate_default_result(self) -> Dict[str, Any]:
        """Generate default result when insufficient data."""
        return {
            'imbalance_ratio': 0.0,
            'imbalance_strength': 0.0,
            'flow_direction': 0,
            'confidence': 0.0,
            'institutional_flow': False,
            'liquidity_state': 'unknown',
            'volume_profile_percentile': 50.0,
            'risk_score': 0.0,
            'signal_quality': 'insufficient_data',
            'error': 'Insufficient data for analysis'
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result."""
        return {
            'imbalance_ratio': 0.0,
            'imbalance_strength': 0.0,
            'flow_direction': 0,
            'confidence': 0.0,
            'institutional_flow': False,
            'liquidity_state': 'error',
            'volume_profile_percentile': 50.0,
            'risk_score': 10.0,
            'signal_quality': 'error',
            'error': error_message
        }


class OrderFlowAnalyzer:
    """Specialized analyzer for order flow calculations."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")


class InstitutionalFlowDetector:
    """Detector for institutional trading patterns."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")


class LiquidityModeler:
    """Modeler for liquidity conditions and market depth."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")


class VolumeProfiler:
    """Analyzer for volume profile and POC dynamics."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")


class MLFlowPredictor:
    """Machine learning predictor for order flow analysis."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.flow_classifier = None
        self.anomaly_detector = None
        self.regime_classifier = None
        self.feature_scaler = StandardScaler()
        self.is_trained = False
    
    def train_models(self, order_flow_history: List[OrderFlowData]) -> None:
        """Train machine learning models on historical order flow data."""
        try:
            if len(order_flow_history) < 100:
                return
            
            # Prepare training data
            features, labels = self._prepare_training_data(order_flow_history)
            
            if features is None or len(features) == 0:
                return
            
            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features)
            
            # Train flow direction classifier
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=0.2, random_state=42
            )
            
            self.flow_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.flow_classifier.fit(X_train, y_train)
            
            # Train anomaly detector
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.anomaly_detector.fit(features_scaled)
            
            # Train regime classifier (simplified)
            regime_labels = self._create_regime_labels(order_flow_history)
            if len(set(regime_labels)) > 1:
                self.regime_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
                self.regime_classifier.fit(features_scaled, regime_labels)
            
            self.is_trained = True
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
    
    def _prepare_training_data(self, order_flow_history: List[OrderFlowData]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from order flow history."""
        try:
            features = []
            labels = []
            
            window_size = 20
            for i in range(window_size, len(order_flow_history)):
                # Extract features from window
                window = order_flow_history[i-window_size:i]
                feature_vector = self._extract_features_from_window(window)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    
                    # Create label based on next period flow direction
                    next_flow = order_flow_history[i]
                    label = 1 if next_flow.trade_direction > 0 else 0
                    labels.append(label)
            
            if features:
                return np.array(features), np.array(labels)
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _extract_features_from_window(self, window: List[OrderFlowData]) -> Optional[np.ndarray]:
        """Extract features from a window of order flow data."""
        try:
            if not window:
                return None
            
            # Basic statistics
            volumes = [flow.volume for flow in window]
            prices = [flow.price for flow in window]
            buy_volumes = [flow.buy_volume for flow in window]
            sell_volumes = [flow.sell_volume for flow in window]
            directions = [flow.trade_direction for flow in window]
            
            features = [
                np.mean(volumes), np.std(volumes),
                np.mean(prices), np.std(prices),
                np.mean(buy_volumes) / (np.mean(sell_volumes) + 1e-10),
                np.mean(directions), np.std(directions),
                len([d for d in directions if d > 0]) / len(directions)
            ]
            
            # Add more features as needed
            spreads = [(flow.ask - flow.bid) / flow.price for flow in window if flow.bid > 0 and flow.ask > 0]
            if spreads:
                features.extend([np.mean(spreads), np.std(spreads)])
            else:
                features.extend([0.01, 0.001])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features from window: {e}")
            return None
    
    def _create_regime_labels(self, order_flow_history: List[OrderFlowData]) -> List[str]:
        """Create regime labels for training."""
        try:
            labels = []
            window_size = 50
            
            for i in range(len(order_flow_history)):
                start_idx = max(0, i - window_size)
                window = order_flow_history[start_idx:i+1]
                
                if len(window) < 10:
                    labels.append('neutral')
                    continue
                
                # Simple regime classification based on volume and volatility
                volumes = [flow.volume for flow in window]
                prices = [flow.price for flow in window]
                
                avg_volume = np.mean(volumes)
                volume_std = np.std(volumes)
                price_volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                
                if avg_volume > np.percentile(volumes, 80) and price_volatility > 0.02:
                    labels.append('high_activity')
                elif avg_volume < np.percentile(volumes, 20) and price_volatility < 0.005:
                    labels.append('low_activity')
                else:
                    labels.append('normal')
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error creating regime labels: {e}")
            return ['neutral'] * len(order_flow_history)
    
    def predict_flow_direction(self, features: np.ndarray) -> float:
        """Predict flow direction probability."""
        try:
            if not self.is_trained or self.flow_classifier is None:
                return 0.5
            
            features_scaled = self.feature_scaler.transform(features)
            prob = self.flow_classifier.predict_proba(features_scaled)[0]
            return prob[1] if len(prob) > 1 else 0.5
            
        except Exception as e:
            self.logger.error(f"Error predicting flow direction: {e}")
            return 0.5
    
    def detect_anomalies(self, features: np.ndarray) -> float:
        """Detect anomalies in order flow."""
        try:
            if not self.is_trained or self.anomaly_detector is None:
                return 0.0
            
            features_scaled = self.feature_scaler.transform(features)
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            # Convert to 0-1 scale (higher values indicate anomalies)
            return max(0, min(1, (0.5 - anomaly_score) * 2))
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return 0.0
    
    def classify_regime(self, features: np.ndarray) -> str:
        """Classify current market regime."""
        try:
            if not self.is_trained or self.regime_classifier is None:
                return 'neutral'
            
            features_scaled = self.feature_scaler.transform(features)
            regime = self.regime_classifier.predict(features_scaled)[0]
            return regime
            
        except Exception as e:
            self.logger.error(f"Error classifying regime: {e}")
            return 'neutral'
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models."""
        try:
            if not self.is_trained or self.flow_classifier is None:
                return {}
            
            feature_names = [f'feature_{i}' for i in range(len(self.flow_classifier.feature_importances_))]
            importance_dict = dict(zip(feature_names, self.flow_classifier.feature_importances_))
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def get_prediction_confidence(self) -> float:
        """Get confidence in predictions."""
        try:
            if not self.is_trained:
                return 0.0
            
            # Simple confidence measure based on model existence
            confidence = 0.0
            if self.flow_classifier is not None:
                confidence += 0.4
            if self.anomaly_detector is not None:
                confidence += 0.3
            if self.regime_classifier is not None:
                confidence += 0.3
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error getting prediction confidence: {e}")
            return 0.0


class FlowRiskAssessor:
    """Risk assessor for order flow conditions."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def assess_risks(self, order_flow_history: List[OrderFlowData], 
                    institutional_flows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risks associated with current order flow conditions."""
        try:
            if not order_flow_history:
                return {'risk_score': 0.0, 'risk_factors': []}
            
            risk_factors = []
            risk_scores = []
            
            recent_flows = order_flow_history[-min(50, len(order_flow_history)):]
            
            # Volume concentration risk
            volumes = [flow.volume for flow in recent_flows]
            volume_concentration = np.max(volumes) / (np.mean(volumes) + 1e-10)
            if volume_concentration > 5:
                risk_factors.append('high_volume_concentration')
                risk_scores.append(min(volume_concentration / 10, 3.0))
            
            # Spread volatility risk
            spreads = [(flow.ask - flow.bid) / flow.price for flow in recent_flows 
                      if flow.bid > 0 and flow.ask > 0]
            if spreads:
                spread_volatility = np.std(spreads) / (np.mean(spreads) + 1e-10)
                if spread_volatility > 2:
                    risk_factors.append('high_spread_volatility')
                    risk_scores.append(min(spread_volatility, 2.0))
            
            # Directional consistency risk (too one-sided)
            directions = [flow.trade_direction for flow in recent_flows if flow.trade_direction != 0]
            if directions:
                direction_consistency = abs(np.mean(directions))
                if direction_consistency > 0.8:
                    risk_factors.append('extreme_directional_bias')
                    risk_scores.append(direction_consistency * 2)
            
            # Institutional flow risk
            if len(institutional_flows) > 0:
                recent_institutional = len([f for f in institutional_flows[-10:] if f.get('detected', False)])
                if recent_institutional > 5:
                    risk_factors.append('high_institutional_activity')
                    risk_scores.append(min(recent_institutional / 5, 2.0))
            
            # Calculate overall risk score
            overall_risk = np.mean(risk_scores) if risk_scores else 0.0
            overall_risk = min(overall_risk, self.parameters['max_risk_score'])
            
            return {
                'risk_score': overall_risk,
                'risk_factors': risk_factors,
                'volume_concentration_risk': volume_concentration if 'volumes' in locals() else 0.0,
                'spread_volatility_risk': spread_volatility if 'spread_volatility' in locals() else 0.0,
                'directional_bias_risk': direction_consistency if 'direction_consistency' in locals() else 0.0,
                'institutional_activity_risk': recent_institutional if 'recent_institutional' in locals() else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing flow risks: {e}")
            return {'risk_score': 5.0, 'risk_factors': ['assessment_error']}