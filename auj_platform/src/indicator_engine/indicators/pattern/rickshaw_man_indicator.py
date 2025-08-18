"""
Rickshaw Man Indicator - Advanced Market Equilibrium and Uncertainty Detection
=============================================================================

This indicator implements sophisticated rickshaw man doji detection with advanced
equilibrium analysis, market neutrality quantification, and ML-enhanced transition prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import logging
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
import talib
from scipy import stats
from scipy.stats import kurtosis, skew

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    IndicatorResult, 
    SignalType, 
    DataType, 
    DataRequirement
)
from ...core.exceptions import IndicatorCalculationException


@dataclass
class RickshawManPattern:
    """Represents a detected rickshaw man pattern"""
    timestamp: pd.Timestamp
    strength: float
    body_ratio: float
    shadow_symmetry: float
    market_neutrality_score: float
    equilibrium_strength: float
    transition_probability: float
    volatility_context: str
    institutional_presence: float
    market_efficiency_score: float


class RickshawManIndicator(StandardIndicatorInterface):
    """
    Advanced Rickshaw Man Pattern Indicator
    
    Features:
    - Precise rickshaw man identification with symmetry analysis
    - Market neutrality and equilibrium quantification
    - Advanced transition probability prediction using ensemble ML
    - Institutional presence detection through volume and order flow analysis
    - Market efficiency scoring and microstructure analysis
    - Multi-dimensional uncertainty and stability assessment
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'max_body_ratio': 0.05,        # Very small body for rickshaw man
            'min_shadow_symmetry': 0.8,    # High symmetry requirement
            'min_total_shadow': 0.85,      # Very long combined shadows
            'neutrality_threshold': 0.75,   # High neutrality requirement
            'volume_analysis': True,
            'institutional_detection': True,
            'market_efficiency_analysis': True,
            'ml_transition_prediction': True,
            'equilibrium_strength_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="RickshawManIndicator", parameters=default_params)
        
        # Initialize ensemble ML components
        self.transition_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        self.scaler = QuantileTransformer(n_quantiles=100, random_state=42)
        self.is_ml_fitted = False
        
        # Market analysis components
        self.equilibrium_analyzer = self._initialize_equilibrium_analyzer()
        self.efficiency_calculator = self._initialize_efficiency_calculator()
        
        logging.info(f"RickshawManIndicator initialized with parameters: {self.parameters}")
    
    def _initialize_equilibrium_analyzer(self) -> Dict[str, Any]:
        """Initialize equilibrium analysis components"""
        return {
            'symmetry_calculator': self._calculate_price_symmetry,
            'balance_detector': self._detect_market_balance,
            'stability_assessor': self._assess_equilibrium_stability
        }
    
    def _initialize_efficiency_calculator(self) -> Dict[str, Any]:
        """Initialize market efficiency calculation components"""
        return {
            'efficiency_scorer': self._calculate_market_efficiency,
            'microstructure_analyzer': self._analyze_microstructure,
            'randomness_detector': self._detect_price_randomness
        }
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=60,
            lookback_periods=150
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate rickshaw man patterns with advanced equilibrium analysis"""
        try:
            if len(data) < 60:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 60"
                )
            
            # Enhance data with advanced indicators
            enhanced_data = self._enhance_data_with_advanced_indicators(data)
            
            # Detect rickshaw man patterns
            detected_patterns = self._detect_rickshaw_man_patterns(enhanced_data)
            
            # Apply equilibrium strength analysis
            if self.parameters['equilibrium_strength_analysis']:
                equilibrium_enhanced_patterns = self._analyze_equilibrium_strength(
                    detected_patterns, enhanced_data
                )
            else:
                equilibrium_enhanced_patterns = detected_patterns
            
            # Apply institutional presence detection
            if self.parameters['institutional_detection']:
                institutional_enhanced_patterns = self._detect_institutional_presence(
                    equilibrium_enhanced_patterns, enhanced_data
                )
            else:
                institutional_enhanced_patterns = equilibrium_enhanced_patterns
            
            # Apply market efficiency analysis
            if self.parameters['market_efficiency_analysis']:
                efficiency_enhanced_patterns = self._analyze_market_efficiency(
                    institutional_enhanced_patterns, enhanced_data
                )
            else:
                efficiency_enhanced_patterns = institutional_enhanced_patterns
            
            # Apply ML transition prediction
            if self.parameters['ml_transition_prediction'] and efficiency_enhanced_patterns:
                ml_enhanced_patterns = self._predict_market_transitions(
                    efficiency_enhanced_patterns, enhanced_data
                )
            else:
                ml_enhanced_patterns = efficiency_enhanced_patterns
            
            # Generate comprehensive analysis
            neutrality_analysis = self._analyze_market_neutrality(enhanced_data)
            equilibrium_state = self._assess_current_equilibrium_state(enhanced_data)
            transition_signals = self._generate_transition_signals(ml_enhanced_patterns, enhanced_data)
            pattern_analytics = self._generate_pattern_analytics(ml_enhanced_patterns)
            
            return {
                'current_pattern': ml_enhanced_patterns[-1] if ml_enhanced_patterns else None,
                'recent_patterns': ml_enhanced_patterns[-5:],
                'pattern_analytics': pattern_analytics,
                'neutrality_analysis': neutrality_analysis,
                'equilibrium_state': equilibrium_state,
                'transition_signals': transition_signals,
                'market_efficiency': self._assess_current_market_efficiency(enhanced_data),
                'institutional_flow': self._assess_institutional_flow(enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Rickshaw man calculation failed: {str(e)}", e
            )
    
    def _enhance_data_with_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with advanced technical and microstructure indicators"""
        df = data.copy()
        
        # Candlestick components with precision
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Precise ratios for rickshaw man detection
        df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
        df['upper_shadow_ratio'] = np.where(df['total_range'] > 0, df['upper_shadow'] / df['total_range'], 0)
        df['lower_shadow_ratio'] = np.where(df['total_range'] > 0, df['lower_shadow'] / df['total_range'], 0)
        df['total_shadow_ratio'] = df['upper_shadow_ratio'] + df['lower_shadow_ratio']
        
        # Shadow symmetry calculation
        df['shadow_symmetry'] = 1.0 - abs(df['upper_shadow_ratio'] - df['lower_shadow_ratio'])
        
        # Advanced volatility measures
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['volatility_ratio'] = df['true_range'] / df['atr']
        df['volatility_percentile'] = df['atr'].rolling(50).rank(pct=True)
        
        # Price distribution and efficiency measures
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['return_kurtosis'] = df['returns'].rolling(20).apply(lambda x: kurtosis(x.dropna()))
        df['return_skewness'] = df['returns'].rolling(20).apply(lambda x: skew(x.dropna()))
        
        # Market neutrality indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_neutrality'] = 1.0 - abs(df['rsi'] - 50) / 50
        
        # MACD for momentum neutrality
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['macd_neutrality'] = 1.0 / (1.0 + abs(df['macd_hist']))
        
        # Bollinger Bands for price position
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_neutrality'] = 1.0 - abs(df['bb_position'] - 0.5) * 2
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic for momentum
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_neutrality'] = 1.0 - abs(df['stoch_k'] - 50) / 50
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_volatility'] = df['volume'].rolling(20).std() / df['volume_sma']
        
        # Advanced volume indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Price efficiency measures
        df['price_efficiency'] = self._calculate_price_efficiency_series(df)
        df['market_randomness'] = self._calculate_market_randomness_series(df)
        
        # Microstructure proxies
        df['bid_ask_proxy'] = (df['high'] - df['low']) / df['close']  # Proxy for spread
        df['price_impact_proxy'] = abs(df['returns']) / df['volume_ratio']  # Proxy for impact
        
        return df
    
    def _calculate_price_efficiency_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price efficiency as a rolling series"""
        def efficiency_window(window_data):
            if len(window_data) < 10:
                return 0.5
            
            # Measure of how much price movement is permanent vs temporary
            returns = window_data['returns'].dropna()
            if len(returns) < 5:
                return 0.5
            
            # Autocorrelation at lag 1 (lower = more efficient)
            autocorr = returns.autocorr(lag=1)
            if pd.isna(autocorr):
                return 0.5
            
            # Convert to efficiency score (0 = inefficient, 1 = efficient)
            efficiency = 1.0 - abs(autocorr)
            return max(0.0, min(1.0, efficiency))
        
        return df.rolling(20).apply(efficiency_window, raw=False)['returns']
    
    def _calculate_market_randomness_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market randomness as a rolling series"""
        def randomness_window(window_data):
            if len(window_data) < 15:
                return 0.5
            
            returns = window_data['returns'].dropna()
            if len(returns) < 10:
                return 0.5
            
            # Runs test for randomness
            median_return = returns.median()
            runs, n1, n2 = self._calculate_runs_test(returns, median_return)
            
            if n1 == 0 or n2 == 0:
                return 0.5
            
            # Expected number of runs
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            
            # Standardized runs statistic
            if expected_runs > 0:
                randomness = abs(runs - expected_runs) / expected_runs
                return 1.0 - min(randomness, 1.0)
            
            return 0.5
        
        return df.rolling(20).apply(randomness_window, raw=False)['returns']
    
    def _calculate_runs_test(self, returns: pd.Series, median: float) -> Tuple[int, int, int]:
        """Calculate runs test statistics"""
        above = (returns > median).astype(int)
        n1 = above.sum()
        n2 = len(above) - n1
        
        # Count runs
        runs = 1
        for i in range(1, len(above)):
            if above.iloc[i] != above.iloc[i-1]:
                runs += 1
        
        return runs, n1, n2
    
    def _detect_rickshaw_man_patterns(self, data: pd.DataFrame) -> List[RickshawManPattern]:
        """Detect rickshaw man patterns with precise criteria"""
        patterns = []
        
        for i in range(30, len(data)):  # Need substantial context
            row = data.iloc[i]
            
            # Core rickshaw man criteria (very strict)
            if not self._meets_rickshaw_man_criteria(row):
                continue
            
            # Calculate market neutrality score
            neutrality_score = self._calculate_market_neutrality_score(data, i)
            
            # Calculate equilibrium strength
            equilibrium_strength = self._calculate_base_equilibrium_strength(data, i)
            
            # Assess volatility context
            volatility_context = self._assess_volatility_context(row)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(
                row, neutrality_score, equilibrium_strength, volatility_context
            )
            
            # Calculate initial transition probability
            transition_prob = self._calculate_base_transition_probability(
                row, neutrality_score, equilibrium_strength
            )
            
            if (pattern_strength >= 0.75 and 
                neutrality_score >= self.parameters['neutrality_threshold']):
                
                pattern = RickshawManPattern(
                    timestamp=row.name,
                    strength=pattern_strength,
                    body_ratio=row['body_ratio'],
                    shadow_symmetry=row['shadow_symmetry'],
                    market_neutrality_score=neutrality_score,
                    equilibrium_strength=equilibrium_strength,
                    transition_probability=transition_prob,
                    volatility_context=volatility_context,
                    institutional_presence=0.0,  # Will be calculated later
                    market_efficiency_score=0.0  # Will be calculated later
                )
                patterns.append(pattern)
        
        return patterns
    
    def _meets_rickshaw_man_criteria(self, row: pd.Series) -> bool:
        """Check if candle meets rickshaw man criteria (most strict doji variant)"""
        # 1. Extremely small body
        if row['body_ratio'] > self.parameters['max_body_ratio']:
            return False
        
        # 2. Very long combined shadows
        if row['total_shadow_ratio'] < self.parameters['min_total_shadow']:
            return False
        
        # 3. High shadow symmetry (balanced)
        if row['shadow_symmetry'] < self.parameters['min_shadow_symmetry']:
            return False
        
        # 4. Significant volatility (meaningful pattern)
        if row['volatility_ratio'] < 0.8:
            return False
        
        # 5. Minimum total range relative to ATR
        if row['total_range'] < row['atr'] * 0.8:
            return False
        
        # 6. Body should be extremely centered
        body_center = abs((row['open'] + row['close']) / 2 - (row['high'] + row['low']) / 2)
        if body_center > row['total_range'] * 0.05:  # Body must be very centered
            return False
        
        return True
    
    def _calculate_market_neutrality_score(self, data: pd.DataFrame, index: int) -> float:
        """Calculate comprehensive market neutrality score"""
        row = data.iloc[index]
        context_data = data.iloc[max(0, index-15):index+1]
        
        neutrality_factors = []
        
        # 1. RSI neutrality (weight: 0.2)
        rsi_neutrality = row['rsi_neutrality']
        neutrality_factors.append(rsi_neutrality * 0.2)
        
        # 2. Bollinger Band position neutrality (weight: 0.2)
        bb_neutrality = row['bb_neutrality']
        neutrality_factors.append(bb_neutrality * 0.2)
        
        # 3. MACD neutrality (weight: 0.15)
        macd_neutrality = row['macd_neutrality']
        neutrality_factors.append(macd_neutrality * 0.15)
        
        # 4. Stochastic neutrality (weight: 0.15)
        stoch_neutrality = row['stoch_neutrality']
        neutrality_factors.append(stoch_neutrality * 0.15)
        
        # 5. Price efficiency (weight: 0.15)
        price_efficiency = row['price_efficiency']
        neutrality_factors.append(price_efficiency * 0.15)
        
        # 6. Market randomness (weight: 0.1)
        market_randomness = row['market_randomness']
        neutrality_factors.append(market_randomness * 0.1)
        
        # 7. Return distribution neutrality (weight: 0.05)
        return_kurtosis = abs(row['return_kurtosis']) if not pd.isna(row['return_kurtosis']) else 0
        kurtosis_neutrality = 1.0 / (1.0 + return_kurtosis)
        neutrality_factors.append(kurtosis_neutrality * 0.05)
        
        return min(sum(neutrality_factors), 1.0)
    
    def _calculate_base_equilibrium_strength(self, data: pd.DataFrame, index: int) -> float:
        """Calculate base equilibrium strength"""
        context_data = data.iloc[max(0, index-20):index+1]
        
        if len(context_data) < 15:
            return 0.5
        
        equilibrium_factors = []
        
        # 1. Price range stability
        ranges = context_data['total_range']
        range_stability = 1.0 - (ranges.std() / ranges.mean()) if ranges.mean() > 0 else 0.5
        equilibrium_factors.append(max(0, min(1, range_stability)) * 0.3)
        
        # 2. Volume consistency
        volume_cv = context_data['volume'].std() / context_data['volume'].mean()
        volume_consistency = 1.0 / (1.0 + volume_cv)
        equilibrium_factors.append(volume_consistency * 0.25)
        
        # 3. Price oscillation around mean
        price_mean = context_data['close'].mean()
        price_deviations = abs(context_data['close'] - price_mean) / price_mean
        oscillation_quality = 1.0 - price_deviations.mean()
        equilibrium_factors.append(max(0, min(1, oscillation_quality)) * 0.25)
        
        # 4. Neutrality persistence
        neutrality_persistence = context_data['rsi_neutrality'].mean()
        equilibrium_factors.append(neutrality_persistence * 0.2)
        
        return sum(equilibrium_factors)
    
    def _assess_volatility_context(self, row: pd.Series) -> str:
        """Assess volatility context for rickshaw man"""
        vol_percentile = row['volatility_percentile']
        
        if vol_percentile > 0.8:
            return "high_volatility_equilibrium"
        elif vol_percentile > 0.6:
            return "elevated_volatility_equilibrium"
        elif vol_percentile > 0.4:
            return "normal_volatility_equilibrium"
        elif vol_percentile > 0.2:
            return "low_volatility_equilibrium"
        else:
            return "very_low_volatility_equilibrium"
    
    def _calculate_pattern_strength(self, row: pd.Series, neutrality_score: float, 
                                  equilibrium_strength: float, volatility_context: str) -> float:
        """Calculate overall rickshaw man pattern strength"""
        strength_components = []
        
        # 1. Rickshaw man quality (35% weight)
        rickshaw_quality = (
            (1 - row['body_ratio'] / self.parameters['max_body_ratio']) * 0.35 +
            (row['total_shadow_ratio'] / self.parameters['min_total_shadow']) * 0.35 +
            (row['shadow_symmetry'] / self.parameters['min_shadow_symmetry']) * 0.3
        )
        strength_components.append(rickshaw_quality * 0.35)
        
        # 2. Market neutrality score (30% weight)
        strength_components.append(neutrality_score * 0.3)
        
        # 3. Equilibrium strength (20% weight)
        strength_components.append(equilibrium_strength * 0.2)
        
        # 4. Volatility context appropriateness (10% weight)
        volatility_factor = {
            "very_low_volatility_equilibrium": 0.6,
            "low_volatility_equilibrium": 0.8,
            "normal_volatility_equilibrium": 1.0,
            "elevated_volatility_equilibrium": 0.9,
            "high_volatility_equilibrium": 0.7
        }.get(volatility_context, 0.5)
        strength_components.append(volatility_factor * 0.1)
        
        # 5. Shadow symmetry bonus (5% weight)
        symmetry_bonus = row['shadow_symmetry']
        strength_components.append(symmetry_bonus * 0.05)
        
        return min(sum(strength_components), 1.0)
    
    def _calculate_base_transition_probability(self, row: pd.Series, neutrality_score: float, 
                                             equilibrium_strength: float) -> float:
        """Calculate base transition probability from equilibrium"""
        # Base probability of transition from equilibrium
        probability = 0.4  # Lower than other doji types (rickshaw man is more stable)
        
        # Higher neutrality might paradoxically increase transition probability
        # (coiled spring effect)
        if neutrality_score > 0.9:
            probability += 0.2
        elif neutrality_score > 0.8:
            probability += 0.1
        
        # Strong equilibrium can lead to sharper transitions
        probability += equilibrium_strength * 0.2
        
        # Volatility context
        vol_percentile = row['volatility_percentile']
        if vol_percentile > 0.7:
            probability += 0.15  # High volatility increases transition likelihood
        
        # Market efficiency (paradox: too efficient markets can snap)
        if row['price_efficiency'] > 0.9:
            probability += 0.1
        
        return min(max(probability, 0.1), 0.8)  # Rickshaw man is inherently stable
    
    def _analyze_equilibrium_strength(self, patterns: List[RickshawManPattern], 
                                    data: pd.DataFrame) -> List[RickshawManPattern]:
        """Analyze equilibrium strength for each pattern"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            enhanced_equilibrium = self._calculate_enhanced_equilibrium_strength(data, pattern_idx)
            pattern.equilibrium_strength = enhanced_equilibrium
            
            # Adjust pattern strength based on enhanced equilibrium analysis
            pattern.strength = (pattern.strength * 0.85 + enhanced_equilibrium * 0.15)
        
        return patterns
    
    def _calculate_enhanced_equilibrium_strength(self, data: pd.DataFrame, index: int) -> float:
        """Calculate enhanced equilibrium strength with advanced metrics"""
        try:
            context_data = data.iloc[max(0, index-25):index+1]
            
            if len(context_data) < 20:
                return 0.5
            
            enhancement_factors = []
            
            # 1. Statistical stationarity test
            returns = context_data['returns'].dropna()
            if len(returns) > 10:
                # Simple stationarity proxy: consistency of mean and variance
                half1 = returns[:len(returns)//2]
                half2 = returns[len(returns)//2:]
                
                mean_stability = 1.0 - abs(half1.mean() - half2.mean()) / (abs(half1.mean()) + abs(half2.mean()) + 1e-10)
                var_stability = 1.0 - abs(half1.var() - half2.var()) / (half1.var() + half2.var() + 1e-10)
                
                stationarity = (mean_stability + var_stability) / 2
                enhancement_factors.append(max(0, min(1, stationarity)) * 0.3)
            else:
                enhancement_factors.append(0.5 * 0.3)
            
            # 2. Volume-price relationship stability
            volume_price_corr = abs(context_data['volume_ratio'].corr(context_data['returns']))
            if pd.isna(volume_price_corr):
                volume_price_corr = 0.5
            correlation_stability = 1.0 - volume_price_corr  # Lower correlation = more equilibrium
            enhancement_factors.append(correlation_stability * 0.25)
            
            # 3. Bid-ask spread proxy stability
            spread_proxy = context_data['bid_ask_proxy']
            spread_stability = 1.0 - (spread_proxy.std() / spread_proxy.mean()) if spread_proxy.mean() > 0 else 0.5
            enhancement_factors.append(max(0, min(1, spread_stability)) * 0.2)
            
            # 4. Order flow balance (using OBV and AD line)
            obv_trend = context_data['obv'].diff().mean()
            ad_trend = context_data['ad_line'].diff().mean()
            
            # Normalize trends
            obv_balance = 1.0 / (1.0 + abs(obv_trend))
            ad_balance = 1.0 / (1.0 + abs(ad_trend))
            
            flow_balance = (obv_balance + ad_balance) / 2
            enhancement_factors.append(flow_balance * 0.15)
            
            # 5. Microstructure efficiency
            efficiency = context_data['price_efficiency'].mean()
            enhancement_factors.append(efficiency * 0.1)
            
            return sum(enhancement_factors)
            
        except Exception:
            return 0.5
    
    def _detect_institutional_presence(self, patterns: List[RickshawManPattern], 
                                     data: pd.DataFrame) -> List[RickshawManPattern]:
        """Detect institutional presence around patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            institutional_score = self._calculate_institutional_presence_score(data, pattern_idx)
            pattern.institutional_presence = institutional_score
        
        return patterns
    
    def _calculate_institutional_presence_score(self, data: pd.DataFrame, index: int) -> float:
        """Calculate institutional presence score"""
        try:
            context_data = data.iloc[max(0, index-10):index+1]
            
            presence_factors = []
            
            # 1. Volume profile analysis
            avg_volume = context_data['volume'].mean()
            volume_at_pattern = data.iloc[index]['volume']
            volume_significance = min(volume_at_pattern / avg_volume, 3.0) / 3.0
            presence_factors.append(volume_significance * 0.3)
            
            # 2. Price impact analysis
            price_impact = context_data['price_impact_proxy'].mean()
            # Lower price impact with high volume suggests institutional presence
            impact_efficiency = 1.0 / (1.0 + price_impact * 10)
            presence_factors.append(impact_efficiency * 0.25)
            
            # 3. Order flow analysis (using CMF)
            cmf_neutrality = 1.0 - abs(data.iloc[index]['cmf']) / 100000  # Normalize CMF
            presence_factors.append(max(0, min(1, cmf_neutrality)) * 0.2)
            
            # 4. Volume volatility (institutional flow tends to be more consistent)
            volume_consistency = 1.0 - context_data['volume_volatility'].iloc[-1]
            presence_factors.append(max(0, min(1, volume_consistency)) * 0.15)
            
            # 5. Block trade proxy (large relative volume with small price impact)
            block_trade_indicator = (
                volume_significance * impact_efficiency * 
                (1.0 if volume_at_pattern > avg_volume * 1.5 else 0.5)
            )
            presence_factors.append(block_trade_indicator * 0.1)
            
            return sum(presence_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_market_efficiency(self, patterns: List[RickshawManPattern], 
                                 data: pd.DataFrame) -> List[RickshawManPattern]:
        """Analyze market efficiency for each pattern"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            efficiency_score = self._calculate_comprehensive_market_efficiency(data, pattern_idx)
            pattern.market_efficiency_score = efficiency_score
        
        return patterns
    
    def _calculate_comprehensive_market_efficiency(self, data: pd.DataFrame, index: int) -> float:
        """Calculate comprehensive market efficiency score"""
        try:
            context_data = data.iloc[max(0, index-30):index+1]
            
            if len(context_data) < 20:
                return 0.5
            
            efficiency_factors = []
            
            # 1. Price efficiency (already calculated)
            price_eff = data.iloc[index]['price_efficiency']
            efficiency_factors.append(price_eff * 0.3)
            
            # 2. Market randomness
            randomness = data.iloc[index]['market_randomness']
            efficiency_factors.append(randomness * 0.25)
            
            # 3. Return distribution normality
            returns = context_data['returns'].dropna()
            if len(returns) > 15:
                kurtosis_val = abs(kurtosis(returns))
                skewness_val = abs(skew(returns))
                
                # Lower kurtosis and skewness indicate more efficient markets
                kurtosis_efficiency = 1.0 / (1.0 + kurtosis_val)
                skewness_efficiency = 1.0 / (1.0 + skewness_val * 5)
                
                distribution_efficiency = (kurtosis_efficiency + skewness_efficiency) / 2
                efficiency_factors.append(distribution_efficiency * 0.2)
            else:
                efficiency_factors.append(0.5 * 0.2)
            
            # 4. Arbitrage opportunity measure (using bid-ask proxy)
            bid_ask_efficiency = 1.0 - context_data['bid_ask_proxy'].mean()
            efficiency_factors.append(max(0, min(1, bid_ask_efficiency)) * 0.15)
            
            # 5. Information incorporation speed (price response to volume)
            volume_price_corr = abs(context_data['volume_ratio'].corr(abs(context_data['returns'])))
            if pd.isna(volume_price_corr):
                volume_price_corr = 0.5
            
            # Moderate correlation suggests efficient information incorporation
            incorporation_efficiency = 1.0 - abs(volume_price_corr - 0.3) / 0.7
            efficiency_factors.append(max(0, min(1, incorporation_efficiency)) * 0.1)
            
            return sum(efficiency_factors)
            
        except Exception:
            return 0.5
    
    def _predict_market_transitions(self, patterns: List[RickshawManPattern], 
                                  data: pd.DataFrame) -> List[RickshawManPattern]:
        """Predict market transitions using ensemble ML"""
        if not patterns or not self.parameters['ml_transition_prediction']:
            return patterns
        
        try:
            # Extract features for ML model
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_transition_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 8:
                return patterns
            
            # Train model if needed
            if not self.is_ml_fitted:
                self._train_transition_model(patterns, features)
            
            # Apply ML predictions if model is fitted
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                transition_predictions = self.transition_predictor.predict(features_scaled)
                
                # Update transition probabilities
                for i, pattern in enumerate(patterns):
                    ml_probability = max(0.1, min(0.9, transition_predictions[i]))
                    # Combine with base probability
                    pattern.transition_probability = (
                        pattern.transition_probability * 0.5 + ml_probability * 0.5
                    )
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML transition prediction failed: {str(e)}")
            return patterns
    
    def _extract_transition_features(self, data: pd.DataFrame, index: int, 
                                   pattern: RickshawManPattern) -> List[float]:
        """Extract features for transition prediction ML model"""
        try:
            row = data.iloc[index]
            context_data = data.iloc[max(0, index-15):index+1]
            
            features = [
                pattern.strength,
                pattern.market_neutrality_score,
                pattern.equilibrium_strength,
                pattern.institutional_presence,
                pattern.market_efficiency_score,
                pattern.shadow_symmetry,
                row['volatility_ratio'],
                row['volatility_percentile'],
                row['rsi_neutrality'],
                row['bb_neutrality'],
                row['bb_width'],
                row['volume_ratio'],
                row['price_efficiency'],
                row['market_randomness'],
                context_data['volume_volatility'].mean(),
                context_data['return_kurtosis'].iloc[-1] if not pd.isna(context_data['return_kurtosis'].iloc[-1]) else 0,
                abs(row['cmf']) / 100000,  # Normalized CMF
                row['bid_ask_proxy'],
                context_data['obv'].diff().mean(),
                context_data['ad_line'].diff().mean()
            ]
            
            return features
            
        except Exception:
            return [0.5] * 20  # Default features
    
    def _train_transition_model(self, patterns: List[RickshawManPattern], features: List[List[float]]):
        """Train ensemble ML model for transition prediction"""
        try:
            # Create targets based on pattern characteristics
            targets = []
            for pattern in patterns:
                # Transition probability based on multiple factors
                target = (
                    (1.0 - pattern.market_neutrality_score) * 0.3 +  # High neutrality = low transition
                    pattern.institutional_presence * 0.25 +  # Institutional presence increases transition
                    (1.0 - pattern.market_efficiency_score) * 0.25 +  # Lower efficiency = higher transition
                    pattern.equilibrium_strength * 0.2  # Strong equilibrium can lead to sharp transitions
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 12:
                features_scaled = self.scaler.fit_transform(features)
                self.transition_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML transition predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _analyze_market_neutrality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market neutrality state"""
        current = data.iloc[-1]
        recent_data = data.iloc[-20:]
        
        return {
            'current_neutrality': current['rsi_neutrality'] * current['bb_neutrality'] * current['stoch_neutrality'],
            'neutrality_persistence': recent_data['rsi_neutrality'].mean(),
            'price_efficiency': current['price_efficiency'],
            'market_randomness': current['market_randomness'],
            'equilibrium_indicators': {
                'macd_neutrality': current['macd_neutrality'],
                'bb_width': current['bb_width'],
                'volatility_regime': self._assess_volatility_context(current)
            }
        }
    
    def _assess_current_equilibrium_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current market equilibrium state comprehensively"""
        equilibrium_score = self._calculate_enhanced_equilibrium_strength(data, len(data)-1)
        current = data.iloc[-1]
        
        return {
            'equilibrium_score': equilibrium_score,
            'is_strong_equilibrium': equilibrium_score > 0.8,
            'neutrality_factors': {
                'rsi_neutrality': current['rsi_neutrality'],
                'bb_neutrality': current['bb_neutrality'],
                'stoch_neutrality': current['stoch_neutrality'],
                'macd_neutrality': current['macd_neutrality']
            },
            'efficiency_metrics': {
                'price_efficiency': current['price_efficiency'],
                'market_randomness': current['market_randomness'],
                'return_distribution': {
                    'kurtosis': current['return_kurtosis'] if not pd.isna(current['return_kurtosis']) else 0,
                    'skewness': current['return_skewness'] if not pd.isna(current['return_skewness']) else 0
                }
            }
        }
    
    def _generate_transition_signals(self, patterns: List[RickshawManPattern], 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """Generate transition signals based on rickshaw man patterns"""
        if not patterns:
            return {'transition_probability': 0.0, 'signal_strength': 0.0}
        
        # Get recent high-quality patterns
        recent_patterns = [p for p in patterns[-3:] if p.strength > 0.8]
        
        if not recent_patterns:
            return {'transition_probability': 0.0, 'signal_strength': 0.0}
        
        # Calculate aggregate metrics
        avg_transition_prob = sum(p.transition_probability for p in recent_patterns) / len(recent_patterns)
        avg_strength = sum(p.strength for p in recent_patterns) / len(recent_patterns)
        avg_neutrality = sum(p.market_neutrality_score for p in recent_patterns) / len(recent_patterns)
        avg_equilibrium = sum(p.equilibrium_strength for p in recent_patterns) / len(recent_patterns)
        
        return {
            'transition_probability': avg_transition_prob,
            'signal_strength': avg_strength,
            'market_neutrality': avg_neutrality,
            'equilibrium_strength': avg_equilibrium,
            'pattern_count': len(recent_patterns),
            'institutional_presence': sum(p.institutional_presence for p in recent_patterns) / len(recent_patterns),
            'market_efficiency': sum(p.market_efficiency_score for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': recent_patterns[-1].timestamp
        }
    
    def _assess_current_market_efficiency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current market efficiency comprehensively"""
        efficiency_score = self._calculate_comprehensive_market_efficiency(data, len(data)-1)
        current = data.iloc[-1]
        
        return {
            'efficiency_score': efficiency_score,
            'is_highly_efficient': efficiency_score > 0.8,
            'price_efficiency': current['price_efficiency'],
            'market_randomness': current['market_randomness'],
            'microstructure_quality': {
                'bid_ask_proxy': current['bid_ask_proxy'],
                'price_impact_proxy': current['price_impact_proxy'],
                'volume_consistency': 1.0 - current['volume_volatility']
            }
        }
    
    def _assess_institutional_flow(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess institutional flow characteristics"""
        institutional_score = self._calculate_institutional_presence_score(data, len(data)-1)
        current = data.iloc[-1]
        
        return {
            'institutional_presence_score': institutional_score,
            'volume_significance': current['volume_ratio'],
            'order_flow_balance': {
                'obv_trend': data['obv'].diff().iloc[-5:].mean(),
                'ad_line_trend': data['ad_line'].diff().iloc[-5:].mean(),
                'cmf_neutrality': 1.0 - abs(current['cmf']) / 100000
            },
            'block_trade_indicators': {
                'volume_surge': current['volume_ratio'] > 2.0,
                'low_price_impact': current['price_impact_proxy'] < 0.1
            }
        }
    
    def _generate_pattern_analytics(self, patterns: List[RickshawManPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-15:]  # Last 15 patterns
        
        return {
            'total_patterns': len(recent_patterns),
            'average_strength': sum(p.strength for p in recent_patterns) / len(recent_patterns),
            'average_neutrality': sum(p.market_neutrality_score for p in recent_patterns) / len(recent_patterns),
            'average_equilibrium_strength': sum(p.equilibrium_strength for p in recent_patterns) / len(recent_patterns),
            'average_transition_probability': sum(p.transition_probability for p in recent_patterns) / len(recent_patterns),
            'average_institutional_presence': sum(p.institutional_presence for p in recent_patterns) / len(recent_patterns),
            'average_market_efficiency': sum(p.market_efficiency_score for p in recent_patterns) / len(recent_patterns),
            'high_strength_patterns': len([p for p in recent_patterns if p.strength > 0.9]),
            'high_neutrality_patterns': len([p for p in recent_patterns if p.market_neutrality_score > 0.9]),
            'strong_equilibrium_patterns': len([p for p in recent_patterns if p.equilibrium_strength > 0.8])
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on rickshaw man analysis"""
        current_pattern = value.get('current_pattern')
        transition_signals = value.get('transition_signals', {})
        equilibrium_state = value.get('equilibrium_state', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Rickshaw man primarily signals extreme market equilibrium/indecision
        # Signals are generated based on potential for transition from this state
        
        transition_probability = transition_signals.get('transition_probability', 0.0)
        market_neutrality = current_pattern.market_neutrality_score
        
        # Very high neutrality with high transition probability suggests major move pending
        if (market_neutrality > 0.9 and 
            transition_probability > 0.7 and 
            current_pattern.strength > 0.85):
            
            # Don't predict direction, but signal that a significant move is likely
            confidence = current_pattern.strength * 0.7
            return SignalType.NEUTRAL, confidence  # Wait for direction confirmation
        
        # High institutional presence with strong equilibrium might precede institutional move
        elif (current_pattern.institutional_presence > 0.8 and 
              current_pattern.equilibrium_strength > 0.8):
            
            confidence = (current_pattern.strength + current_pattern.institutional_presence) / 2 * 0.6
            return SignalType.HOLD, confidence
        
        # Strong equilibrium with moderate transition probability
        elif current_pattern.strength > 0.8 and equilibrium_state.get('is_strong_equilibrium', False):
            confidence = current_pattern.strength * 0.5
            return SignalType.NEUTRAL, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'rickshaw_man_doji',
            'equilibrium_analysis_enabled': self.parameters['equilibrium_strength_analysis'],
            'institutional_detection_enabled': self.parameters['institutional_detection'],
            'market_efficiency_analysis_enabled': self.parameters['market_efficiency_analysis'],
            'transition_prediction_enabled': self.parameters['ml_transition_prediction'],
            'neutrality_threshold': self.parameters['neutrality_threshold']
        })
        return base_metadata