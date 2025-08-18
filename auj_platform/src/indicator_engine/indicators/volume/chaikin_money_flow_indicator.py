"""
Advanced Chaikin Money Flow (CMF) Indicator
=========================================

A sophisticated implementation of Chaikin Money Flow with:
- Multi-timeframe analysis
- Volume pressure algorithms
- Smart money detection
- Accumulation/Distribution pattern recognition
- Statistical validation

Mathematical Foundation:
1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
2. Money Flow Volume = Money Flow Multiplier × Volume
3. CMF = Sum(Money Flow Volume, n) / Sum(Volume, n)

Advanced Features:
1. Multi-timeframe CMF analysis for trend confirmation
2. Volume pressure algorithms for institutional flow detection
3. Smart money identification through volume clustering
4. Pattern recognition for accumulation/distribution phases
5. Statistical significance testing for signal validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Import base class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ..indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from ..core.signal_type import SignalType

@dataclass
class CMFConfig:
    """Configuration for Chaikin Money Flow calculation"""
    primary_period: int = 21
    secondary_periods: List[int] = None
    volume_threshold_percentile: float = 80.0
    smart_money_threshold: float = 2.0
    pattern_detection_window: int = 50
    statistical_confidence: float = 0.95
    smoothing_factor: float = 0.1
    accumulation_threshold: float = 0.1
    distribution_threshold: float = -0.1
    
    def __post_init__(self):
        if self.secondary_periods is None:
            self.secondary_periods = [10, 14, 34, 55]

@dataclass
class SmartMoneyFlow:
    """Smart money flow detection result"""
    flow_strength: float
    flow_direction: str  # 'accumulation', 'distribution', 'neutral'
    institutional_activity: float
    volume_signature: str  # 'block_buying', 'block_selling', 'normal'

@dataclass
class CMFResult:
    """Result structure for Chaikin Money Flow analysis"""
    cmf_primary: float
    cmf_multi_timeframe: Dict[str, float]
    volume_pressure: float
    smart_money_flow: SmartMoneyFlow
    accumulation_distribution_state: str
    signal_strength: float
    statistical_significance: float
    trend_confirmation: float

class ChaikinMoneyFlowIndicator(StandardIndicatorInterface):
    """
    Advanced Chaikin Money Flow Indicator
    
    This indicator provides sophisticated money flow analysis through:
    1. Multi-timeframe CMF calculation for trend confirmation
    2. Volume pressure algorithms for institutional detection
    3. Smart money flow identification using clustering techniques
    4. Accumulation/Distribution pattern recognition
    5. Statistical validation of signal strength
    """
    
    def __init__(self, config: Optional[CMFConfig] = None):
        """Initialize the Chaikin Money Flow Indicator"""
        super().__init__()
        self.config = config or CMFConfig()
        self.logger = logging.getLogger(__name__)
        
        # Analysis components
        self._scaler = StandardScaler()
        self._clustering_model = DBSCAN(eps=0.3, min_samples=3)
        
        # Historical data storage
        self._cmf_history: List[float] = []
        self._volume_pressure_history: List[float] = []
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        self._money_flow_history: List[float] = []
        
        # Pattern tracking
        self._accumulation_periods: List[int] = []
        self._distribution_periods: List[int] = []
        self._smart_money_events: List[Dict[str, Any]] = []
        
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced Chaikin Money Flow with comprehensive analysis
        
        Args:
            data: Dictionary containing OHLCV data
            
        Returns:
            Dictionary with CMF analysis results
        """
        try:
            # Validate and extract data
            if not self._validate_data(data):
                raise ValueError("Invalid or insufficient data provided")
            
            df = pd.DataFrame(data)
            
            # Ensure minimum data requirements
            if len(df) < max(self.config.secondary_periods + [self.config.primary_period]):
                return self._create_default_result()
            
            # Calculate core CMF components
            cmf_results = self._calculate_multi_timeframe_cmf(df)
            
            # Calculate volume pressure analysis
            pressure_results = self._calculate_volume_pressure(df)
            
            # Detect smart money flows
            smart_money_results = self._detect_smart_money_flow(df)
            
            # Analyze accumulation/distribution patterns
            pattern_results = self._analyze_accumulation_distribution(df, cmf_results)
            
            # Statistical validation
            stats_results = self._validate_statistical_significance(cmf_results)
            
            # Calculate trend confirmation
            trend_results = self._calculate_trend_confirmation(df, cmf_results)
            
            # Compile final result
            result = self._compile_cmf_result(
                cmf_results, pressure_results, smart_money_results,
                pattern_results, stats_results, trend_results
            )
            
            # Update historical data
            self._update_history(df, result)
            
            # Generate trading signal
            signal = self._generate_signal(result)
            
            return {
                'signal': signal,
                'confidence': result.signal_strength,
                'cmf_value': result.cmf_primary,
                'volume_pressure': result.volume_pressure,
                'smart_money_strength': result.smart_money_flow.flow_strength,
                'smart_money_direction': result.smart_money_flow.flow_direction,
                'accumulation_distribution_state': result.accumulation_distribution_state,
                'trend_confirmation': result.trend_confirmation,
                'statistical_significance': result.statistical_significance,
                'multi_timeframe_cmf': result.cmf_multi_timeframe,
                'institutional_activity': result.smart_money_flow.institutional_activity,
                'metadata': {
                    'indicator_name': 'ChaikinMoneyFlow',
                    'calculation_method': 'multi_timeframe_analysis',
                    'parameters': {
                        'primary_period': self.config.primary_period,
                        'secondary_periods': self.config.secondary_periods,
                        'volume_threshold': self.config.volume_threshold_percentile
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Chaikin Money Flow: {str(e)}")
            return self._create_error_result(str(e))
    
    def _calculate_multi_timeframe_cmf(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate CMF across multiple timeframes"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Calculate Money Flow Multiplier
        high_low_diff = high - low
        high_low_diff = np.where(high_low_diff == 0, 1e-8, high_low_diff)
        mf_multiplier = ((close - low) - (high - close)) / high_low_diff
        
        # Calculate Money Flow Volume
        money_flow_volume = mf_multiplier * volume
        
        cmf_results = {}
        
        # Calculate CMF for each timeframe
        periods = [self.config.primary_period] + self.config.secondary_periods
        
        for period in periods:
            if len(df) >= period:
                # Calculate CMF for this period
                mf_sum = np.sum(money_flow_volume[-period:])
                volume_sum = np.sum(volume[-period:])
                
                if volume_sum > 0:
                    cmf_value = mf_sum / volume_sum
                else:
                    cmf_value = 0.0
                
                # Apply smoothing
                if len(self._cmf_history) > 0:
                    prev_cmf = self._cmf_history[-1] if period == self.config.primary_period else cmf_value
                    cmf_value = (self.config.smoothing_factor * cmf_value + 
                               (1 - self.config.smoothing_factor) * prev_cmf)
                
                cmf_results[f'cmf_{period}'] = cmf_value
        
        return cmf_results
    
    def _calculate_volume_pressure(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume pressure for institutional flow detection"""
        volume = df['volume'].values
        close = df['close'].values
        
        if len(volume) < 20:
            return {'volume_pressure': 0.0, 'pressure_trend': 0.0}
        
        # Calculate volume percentiles
        volume_threshold = np.percentile(volume, self.config.volume_threshold_percentile)
        
        # Identify high-volume periods
        high_volume_mask = volume > volume_threshold
        
        if not np.any(high_volume_mask):
            return {'volume_pressure': 0.0, 'pressure_trend': 0.0}
        
        # Calculate price movements during high volume
        price_changes = np.diff(close)
        
        # Align arrays (price_changes is one element shorter)
        if len(price_changes) < len(high_volume_mask):
            high_volume_mask = high_volume_mask[1:]  # Skip first element
        elif len(price_changes) > len(high_volume_mask):
            price_changes = price_changes[-len(high_volume_mask):]
        
        # Calculate volume-weighted price pressure
        high_volume_price_changes = price_changes[high_volume_mask]
        high_volume_weights = volume[high_volume_mask][1:] if len(volume[high_volume_mask]) > len(high_volume_price_changes) else volume[high_volume_mask]
        
        if len(high_volume_price_changes) > 0 and len(high_volume_weights) == len(high_volume_price_changes):
            # Normalize weights
            total_weight = np.sum(high_volume_weights)
            if total_weight > 0:
                weights = high_volume_weights / total_weight
                volume_pressure = np.sum(high_volume_price_changes * weights)
            else:
                volume_pressure = 0.0
        else:
            volume_pressure = 0.0
        
        # Calculate pressure trend (last 10 periods)
        recent_periods = min(10, len(volume))
        if recent_periods > 1:
            recent_volume = volume[-recent_periods:]
            recent_prices = close[-recent_periods:]
            
            # Calculate correlation between volume and price changes
            if len(recent_prices) > 1:
                recent_price_changes = np.diff(recent_prices)
                recent_volume_aligned = recent_volume[1:]  # Align with price changes
                
                if len(recent_price_changes) == len(recent_volume_aligned) and len(recent_price_changes) > 1:
                    pressure_trend = np.corrcoef(recent_price_changes, recent_volume_aligned)[0, 1]
                    if np.isnan(pressure_trend):
                        pressure_trend = 0.0
                else:
                    pressure_trend = 0.0
            else:
                pressure_trend = 0.0
        else:
            pressure_trend = 0.0
        
        return {
            'volume_pressure': volume_pressure,
            'pressure_trend': pressure_trend
        }
    
    def _detect_smart_money_flow(self, df: pd.DataFrame) -> SmartMoneyFlow:
        """Detect smart money flow using advanced volume analysis"""
        volume = df['volume'].values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        if len(volume) < 30:
            return SmartMoneyFlow(
                flow_strength=0.0,
                flow_direction='neutral',
                institutional_activity=0.0,
                volume_signature='normal'
            )
        
        # Calculate average volume and price metrics
        avg_volume = np.mean(volume)
        volume_threshold = avg_volume * self.config.smart_money_threshold
        
        # Identify potential smart money transactions
        smart_money_mask = volume > volume_threshold
        smart_money_indices = np.where(smart_money_mask)[0]
        
        if len(smart_money_indices) == 0:
            return SmartMoneyFlow(
                flow_strength=0.0,
                flow_direction='neutral',
                institutional_activity=0.0,
                volume_signature='normal'
            )
        
        # Analyze price action during smart money events
        institutional_buying = 0
        institutional_selling = 0
        total_smart_volume = 0
        
        for idx in smart_money_indices:
            if idx < len(close):
                volume_weight = volume[idx] / avg_volume
                
                # Analyze price position within the bar
                price_range = high[idx] - low[idx]
                if price_range > 0:
                    close_position = (close[idx] - low[idx]) / price_range
                    
                    # Higher close position suggests buying pressure
                    if close_position > 0.6:
                        institutional_buying += volume_weight
                    elif close_position < 0.4:
                        institutional_selling += volume_weight
                
                total_smart_volume += volume_weight
        
        # Calculate flow metrics
        if total_smart_volume > 0:
            net_flow = institutional_buying - institutional_selling
            flow_strength = abs(net_flow) / total_smart_volume
            
            if net_flow > 0.2 * total_smart_volume:
                flow_direction = 'accumulation'
            elif net_flow < -0.2 * total_smart_volume:
                flow_direction = 'distribution'
            else:
                flow_direction = 'neutral'
        else:
            flow_strength = 0.0
            flow_direction = 'neutral'
        
        # Determine volume signature
        recent_volume = volume[-10:] if len(volume) >= 10 else volume
        avg_recent_volume = np.mean(recent_volume)
        
        if avg_recent_volume > avg_volume * 1.5:
            if institutional_buying > institutional_selling:
                volume_signature = 'block_buying'
            else:
                volume_signature = 'block_selling'
        else:
            volume_signature = 'normal'
        
        # Calculate institutional activity score
        institutional_activity = min(1.0, len(smart_money_indices) / len(volume))
        
        return SmartMoneyFlow(
            flow_strength=flow_strength,
            flow_direction=flow_direction,
            institutional_activity=institutional_activity,
            volume_signature=volume_signature
        )
    
    def _analyze_accumulation_distribution(self, df: pd.DataFrame, 
                                         cmf_results: Dict[str, float]) -> Dict[str, str]:
        """Analyze accumulation/distribution patterns"""
        primary_cmf = cmf_results.get(f'cmf_{self.config.primary_period}', 0.0)
        
        # Get recent CMF values for pattern analysis
        window = min(self.config.pattern_detection_window, len(df))
        
        # Determine current state based on CMF value and trend
        if primary_cmf > self.config.accumulation_threshold:
            if len(self._cmf_history) >= 5:
                recent_trend = np.mean(self._cmf_history[-5:]) - np.mean(self._cmf_history[-10:-5]) if len(self._cmf_history) >= 10 else 0
                if recent_trend > 0:
                    state = 'strong_accumulation'
                else:
                    state = 'accumulation'
            else:
                state = 'accumulation'
        elif primary_cmf < self.config.distribution_threshold:
            if len(self._cmf_history) >= 5:
                recent_trend = np.mean(self._cmf_history[-5:]) - np.mean(self._cmf_history[-10:-5]) if len(self._cmf_history) >= 10 else 0
                if recent_trend < 0:
                    state = 'strong_distribution'
                else:
                    state = 'distribution'
            else:
                state = 'distribution'
        else:
            state = 'neutral'
        
        return {'accumulation_distribution_state': state}
    
    def _validate_statistical_significance(self, cmf_results: Dict[str, float]) -> Dict[str, float]:
        """Validate statistical significance of CMF signals"""
        if len(self._cmf_history) < 30:
            return {'significance': 0.0}
        
        primary_cmf = cmf_results.get(f'cmf_{self.config.primary_period}', 0.0)
        
        # Test if current CMF is significantly different from historical mean
        historical_cmf = np.array(self._cmf_history[-30:])
        
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(historical_cmf, primary_cmf)
        
        # Calculate significance score
        significance = max(0.0, 1.0 - p_value) if p_value <= (1 - self.config.statistical_confidence) else 0.0
        
        return {'significance': significance}
    
    def _calculate_trend_confirmation(self, df: pd.DataFrame, 
                                    cmf_results: Dict[str, float]) -> Dict[str, float]:
        """Calculate trend confirmation using multi-timeframe analysis"""
        primary_cmf = cmf_results.get(f'cmf_{self.config.primary_period}', 0.0)
        
        # Calculate agreement across timeframes
        timeframe_signals = []
        
        for period in self.config.secondary_periods:
            cmf_key = f'cmf_{period}'
            if cmf_key in cmf_results:
                cmf_value = cmf_results[cmf_key]
                
                # Convert CMF to signal direction
                if cmf_value > 0.05:
                    timeframe_signals.append(1)  # Bullish
                elif cmf_value < -0.05:
                    timeframe_signals.append(-1)  # Bearish
                else:
                    timeframe_signals.append(0)  # Neutral
        
        if not timeframe_signals:
            return {'trend_confirmation': 0.0}
        
        # Calculate confirmation strength
        signal_sum = sum(timeframe_signals)
        max_possible = len(timeframe_signals)
        
        if max_possible > 0:
            confirmation = abs(signal_sum) / max_possible
        else:
            confirmation = 0.0
        
        return {'trend_confirmation': confirmation}
    
    def _compile_cmf_result(self, cmf_results: Dict[str, float],
                          pressure_results: Dict[str, float],
                          smart_money_results: SmartMoneyFlow,
                          pattern_results: Dict[str, str],
                          stats_results: Dict[str, float],
                          trend_results: Dict[str, float]) -> CMFResult:
        """Compile all CMF analysis results"""
        primary_cmf = cmf_results.get(f'cmf_{self.config.primary_period}', 0.0)
        
        # Calculate overall signal strength
        signal_components = [
            abs(primary_cmf) * 2,  # Primary CMF weight
            smart_money_results.flow_strength,
            smart_money_results.institutional_activity,
            stats_results['significance'],
            trend_results['trend_confirmation']
        ]
        
        signal_strength = min(1.0, np.mean(signal_components))
        
        return CMFResult(
            cmf_primary=primary_cmf,
            cmf_multi_timeframe=cmf_results,
            volume_pressure=pressure_results['volume_pressure'],
            smart_money_flow=smart_money_results,
            accumulation_distribution_state=pattern_results['accumulation_distribution_state'],
            signal_strength=signal_strength,
            statistical_significance=stats_results['significance'],
            trend_confirmation=trend_results['trend_confirmation']
        )    
    def _generate_signal(self, result: CMFResult) -> SignalType:
        """Generate trading signal based on comprehensive CMF analysis"""
        # Minimum confidence threshold
        if result.signal_strength < 0.3:
            return SignalType.HOLD
        
        # Signal scoring system
        bullish_score = 0
        bearish_score = 0
        
        # Primary CMF analysis
        if result.cmf_primary > 0.1:
            bullish_score += 2
        elif result.cmf_primary < -0.1:
            bearish_score += 2
        elif result.cmf_primary > 0:
            bullish_score += 1
        elif result.cmf_primary < 0:
            bearish_score += 1
        
        # Volume pressure analysis
        if result.volume_pressure > 0:
            bullish_score += 1
        elif result.volume_pressure < 0:
            bearish_score += 1
        
        # Smart money flow analysis
        if result.smart_money_flow.flow_direction == 'accumulation':
            bullish_score += 2
        elif result.smart_money_flow.flow_direction == 'distribution':
            bearish_score += 2
        
        # Institutional activity factor
        if result.smart_money_flow.institutional_activity > 0.5:
            if result.smart_money_flow.volume_signature == 'block_buying':
                bullish_score += 1
            elif result.smart_money_flow.volume_signature == 'block_selling':
                bearish_score += 1
        
        # Accumulation/Distribution state
        if result.accumulation_distribution_state in ['accumulation', 'strong_accumulation']:
            bullish_score += 2 if 'strong' in result.accumulation_distribution_state else 1
        elif result.accumulation_distribution_state in ['distribution', 'strong_distribution']:
            bearish_score += 2 if 'strong' in result.accumulation_distribution_state else 1
        
        # Trend confirmation factor
        if result.trend_confirmation > 0.6:
            if bullish_score > bearish_score:
                bullish_score += 1
            elif bearish_score > bullish_score:
                bearish_score += 1
        
        # Statistical significance factor
        if result.statistical_significance > 0.7:
            if bullish_score > bearish_score:
                bullish_score += 1
            elif bearish_score > bullish_score:
                bearish_score += 1
        
        # Generate final signal
        score_difference = abs(bullish_score - bearish_score)
        min_score_threshold = 3
        
        if (score_difference >= 2 and max(bullish_score, bearish_score) >= min_score_threshold 
            and result.signal_strength > 0.5):
            if bullish_score > bearish_score:
                return SignalType.BUY
            else:
                return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_history(self, df: pd.DataFrame, result: CMFResult):
        """Update historical data for future analysis"""
        max_history = 200
        
        # Update histories
        self._cmf_history.append(result.cmf_primary)
        self._volume_pressure_history.append(result.volume_pressure)
        self._price_history.append(df['close'].iloc[-1])
        self._volume_history.append(df['volume'].iloc[-1])
        
        # Calculate money flow for history
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        volume = df['volume'].iloc[-1]
        
        if high != low:
            mf_multiplier = ((close - low) - (high - close)) / (high - low)
            money_flow = mf_multiplier * volume
        else:
            money_flow = 0.0
        
        self._money_flow_history.append(money_flow)
        
        # Track accumulation/distribution periods
        current_period = len(self._cmf_history) - 1
        if result.accumulation_distribution_state in ['accumulation', 'strong_accumulation']:
            self._accumulation_periods.append(current_period)
        elif result.accumulation_distribution_state in ['distribution', 'strong_distribution']:
            self._distribution_periods.append(current_period)
        
        # Track smart money events
        if result.smart_money_flow.institutional_activity > 0.7:
            self._smart_money_events.append({
                'period': current_period,
                'flow_direction': result.smart_money_flow.flow_direction,
                'flow_strength': result.smart_money_flow.flow_strength,
                'volume_signature': result.smart_money_flow.volume_signature
            })
        
        # Trim histories to maximum length
        histories = [
            self._cmf_history, self._volume_pressure_history,
            self._price_history, self._volume_history, self._money_flow_history
        ]
        
        for history in histories:
            if len(history) > max_history:
                history[:] = history[-max_history:]
        
        # Trim event histories
        if len(self._accumulation_periods) > max_history // 2:
            self._accumulation_periods = self._accumulation_periods[-max_history // 2:]
        
        if len(self._distribution_periods) > max_history // 2:
            self._distribution_periods = self._distribution_periods[-max_history // 2:]
        
        if len(self._smart_money_events) > max_history // 4:
            self._smart_money_events = self._smart_money_events[-max_history // 4:]
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data structure and completeness"""
        required_fields = ['high', 'low', 'close', 'volume']
        
        if not isinstance(data, dict):
            return False
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
            
            if not isinstance(data[field], (list, np.ndarray)) or len(data[field]) == 0:
                self.logger.error(f"Invalid data for field: {field}")
                return False
        
        # Check data consistency
        lengths = [len(data[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            self.logger.error("Inconsistent data lengths across fields")
            return False
        
        return True
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'cmf_value': 0.0,
            'volume_pressure': 0.0,
            'smart_money_strength': 0.0,
            'smart_money_direction': 'neutral',
            'accumulation_distribution_state': 'neutral',
            'trend_confirmation': 0.0,
            'statistical_significance': 0.0,
            'multi_timeframe_cmf': {},
            'institutional_activity': 0.0,
            'metadata': {
                'indicator_name': 'ChaikinMoneyFlow',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'cmf_value': 0.0,
            'volume_pressure': 0.0,
            'smart_money_strength': 0.0,
            'smart_money_direction': 'neutral',
            'accumulation_distribution_state': 'neutral',
            'trend_confirmation': 0.0,
            'statistical_significance': 0.0,
            'multi_timeframe_cmf': {},
            'institutional_activity': 0.0,
            'metadata': {
                'indicator_name': 'ChaikinMoneyFlow',
                'error': error_message
            }
        }