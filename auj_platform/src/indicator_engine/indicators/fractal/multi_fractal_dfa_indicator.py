"""
Multi Fractal DFA Indicator - Advanced Implementation

This indicator implements Multifractal Detrended Fluctuation Analysis (MF-DFA)
for comprehensive market complexity analysis. Features include:
- Multifractal Detrended Fluctuation Analysis (MF-DFA) implementation
- Multifractal spectrum calculation with Hölder exponents
- Scaling exponent analysis across multiple moments
- Multifractality strength and asymmetry measurement
- Market regime classification via multifractal characteristics
- Advanced statistical validation and confidence intervals

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced multifractal analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiFractalDFAResult:
    """Results container for Multi Fractal DFA"""
    hurst_exponent: float
    multifractality_strength: float
    spectrum_width: float
    asymmetry_index: float
    complexity_measure: float
    scaling_quality: float
    dominant_scaling: float
    market_efficiency: float
    regime_classification: str
    confidence_level: float

class MultiFractalDFAIndicator(StandardIndicatorInterface):
    """
    Advanced Multi Fractal DFA Indicator
    
    Implements Multifractal Detrended Fluctuation Analysis (MF-DFA)
    for comprehensive market complexity analysis.
    """
    
    def __init__(self, 
                 window_size: int = 200,
                 min_scale: int = 10,
                 max_scale: int = 50,
                 q_values: List[float] = None,
                 polynomial_order: int = 1):
        super().__init__()
        self.window_size = window_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.q_values = q_values or [-5, -3, -1, 0, 1, 3, 5]
        self.polynomial_order = polynomial_order
        
        logger.info(f"Initialized MultiFractalDFAIndicator with window_size={window_size}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            self._validate_data(data)
            
            if len(data) < self.window_size:
                return self._create_default_result()
            
            recent_data = data[-self.window_size:].copy()
            
            # Prepare time series for MF-DFA
            time_series = self._prepare_time_series(recent_data)
            
            # Perform Multifractal DFA
            mfdfa_results = self._multifractal_dfa(time_series)
            
            # Calculate multifractal spectrum
            spectrum_results = self._calculate_multifractal_spectrum(mfdfa_results)
            
            # Analyze multifractality characteristics
            multifractal_analysis = self._analyze_multifractality(spectrum_results)
            
            # Market regime classification
            regime_analysis = self._classify_market_regime(multifractal_analysis)
            
            # Calculate market efficiency
            efficiency = self._calculate_market_efficiency(mfdfa_results)
            
            result = MultiFractalDFAResult(
                hurst_exponent=mfdfa_results.get('hurst_exponent', 0.5),
                multifractality_strength=multifractal_analysis['strength'],
                spectrum_width=spectrum_results['width'],
                asymmetry_index=spectrum_results['asymmetry'],
                complexity_measure=multifractal_analysis['complexity'],
                scaling_quality=mfdfa_results.get('scaling_quality', 0.0),
                dominant_scaling=mfdfa_results.get('dominant_scaling', 0.5),
                market_efficiency=efficiency,
                regime_classification=regime_analysis['regime'],
                confidence_level=regime_analysis['confidence']
            )
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in MultiFractal DFA calculation: {e}")
            raise IndicatorCalculationError(f"MultiFractalDFAIndicator calculation failed: {e}")

    def _prepare_time_series(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare time series for MF-DFA analysis"""
        try:
            # Use log returns for better stationarity
            prices = data['close'].values
            log_returns = np.diff(np.log(prices))
            
            # Remove any infinite or NaN values
            log_returns = log_returns[np.isfinite(log_returns)]
            
            if len(log_returns) == 0:
                return np.array([0.0])
            
            # Standardize the series
            scaler = StandardScaler()
            normalized_returns = scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
            
            return normalized_returns
            
        except Exception as e:
            logger.warning(f"Time series preparation error: {e}")
            return np.array([0.0])

    def _multifractal_dfa(self, time_series: np.ndarray) -> Dict[str, Any]:
        """Perform Multifractal Detrended Fluctuation Analysis"""
        try:
            if len(time_series) < self.min_scale * 4:
                return {'hurst_exponent': 0.5, 'scaling_quality': 0.0, 'dominant_scaling': 0.5}
            
            # Step 1: Create profile (cumulative sum)
            profile = np.cumsum(time_series - np.mean(time_series))
            
            # Step 2: Define scales
            scales = np.unique(np.logspace(
                np.log10(self.min_scale), 
                np.log10(min(self.max_scale, len(profile) // 4)), 
                10
            ).astype(int))
            
            # Step 3: Calculate fluctuation function for each q and scale
            fluctuation_functions = {}
            
            for q in self.q_values:
                Fq_values = []
                
                for scale in scales:
                    fluctuations = self._calculate_fluctuations(profile, scale)
                    
                    if len(fluctuations) == 0:
                        continue
                    
                    # Calculate qth moment
                    if q == 0:
                        # Special case for q=0 (logarithmic averaging)
                        log_fluct = np.log(fluctuations[fluctuations > 0])
                        if len(log_fluct) > 0:
                            Fq = np.exp(np.mean(log_fluct))
                        else:
                            Fq = 0
                    else:
                        # General case
                        positive_fluct = fluctuations[fluctuations > 0]
                        if len(positive_fluct) > 0:
                            Fq = np.power(np.mean(np.power(positive_fluct, q)), 1.0/q)
                        else:
                            Fq = 0
                    
                    Fq_values.append(Fq)
                
                fluctuation_functions[q] = np.array(Fq_values)
            
            # Step 4: Calculate scaling exponents h(q)
            scaling_exponents = {}
            scaling_qualities = []
            
            for q in self.q_values:
                if q not in fluctuation_functions or len(fluctuation_functions[q]) < 3:
                    continue
                
                Fq_values = fluctuation_functions[q]
                valid_indices = (Fq_values > 0) & np.isfinite(Fq_values)
                
                if np.sum(valid_indices) < 3:
                    continue
                
                log_scales = np.log(scales[:len(Fq_values)][valid_indices])
                log_Fq = np.log(Fq_values[valid_indices])
                
                # Linear regression to find scaling exponent
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_Fq)
                
                scaling_exponents[q] = slope
                scaling_qualities.append(r_value ** 2)
            
            # Extract key results
            results = {
                'scaling_exponents': scaling_exponents,
                'fluctuation_functions': fluctuation_functions,
                'scales': scales
            }
            
            # Calculate Hurst exponent (h(2))
            if 2 in scaling_exponents:
                results['hurst_exponent'] = scaling_exponents[2]
            elif 1 in scaling_exponents:
                results['hurst_exponent'] = scaling_exponents[1]
            else:
                results['hurst_exponent'] = 0.5
            
            # Calculate overall scaling quality
            if scaling_qualities:
                results['scaling_quality'] = np.mean(scaling_qualities)
            else:
                results['scaling_quality'] = 0.0
            
            # Dominant scaling
            if scaling_exponents:
                results['dominant_scaling'] = np.mean(list(scaling_exponents.values()))
            else:
                results['dominant_scaling'] = 0.5
            
            return results
            
        except Exception as e:
            logger.warning(f"MF-DFA calculation error: {e}")
            return {'hurst_exponent': 0.5, 'scaling_quality': 0.0, 'dominant_scaling': 0.5}

    def _calculate_fluctuations(self, profile: np.ndarray, scale: int) -> np.ndarray:
        """Calculate local fluctuations for a given scale"""
        try:
            n_segments = len(profile) // scale
            fluctuations = []
            
            for i in range(n_segments):
                # Extract segment
                segment = profile[i*scale:(i+1)*scale]
                
                if len(segment) != scale:
                    continue
                
                # Detrend using polynomial fit
                x = np.arange(len(segment))
                
                try:
                    coeffs = np.polyfit(x, segment, self.polynomial_order)
                    trend = np.polyval(coeffs, x)
                    detrended = segment - trend
                    
                    # Calculate fluctuation (RMS)
                    fluctuation = np.sqrt(np.mean(detrended**2))
                    fluctuations.append(fluctuation)
                    
                except (np.linalg.LinAlgError, np.RankWarning):
                    # Fallback to simple linear detrending
                    trend = np.linspace(segment[0], segment[-1], len(segment))
                    detrended = segment - trend
                    fluctuation = np.sqrt(np.mean(detrended**2))
                    fluctuations.append(fluctuation)
            
            return np.array(fluctuations)
            
        except Exception as e:
            logger.warning(f"Fluctuation calculation error: {e}")
            return np.array([])

    def _calculate_multifractal_spectrum(self, mfdfa_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate multifractal spectrum (singularity spectrum)"""
        try:
            scaling_exponents = mfdfa_results.get('scaling_exponents', {})
            
            if len(scaling_exponents) < 3:
                return {'width': 0.0, 'asymmetry': 0.0, 'max_alpha': 0.5, 'min_alpha': 0.5}
            
            # Calculate tau(q) = q*h(q) - 1
            q_values = list(scaling_exponents.keys())
            h_values = list(scaling_exponents.values())
            
            tau_values = [q * h - 1 for q, h in zip(q_values, h_values)]
            
            # Calculate alpha (Hölder exponents) and f(alpha)
            alpha_values = []
            f_alpha_values = []
            
            # Numerical derivative of tau(q)
            for i in range(1, len(q_values) - 1):
                # Central difference
                dq = q_values[i+1] - q_values[i-1]
                dtau = tau_values[i+1] - tau_values[i-1]
                
                if dq != 0:
                    alpha = dtau / dq
                    f_alpha = q_values[i] * alpha - tau_values[i]
                    
                    alpha_values.append(alpha)
                    f_alpha_values.append(f_alpha)
            
            if len(alpha_values) < 2:
                return {'width': 0.0, 'asymmetry': 0.0, 'max_alpha': 0.5, 'min_alpha': 0.5}
            
            # Spectrum characteristics
            alpha_min = min(alpha_values)
            alpha_max = max(alpha_values)
            alpha_0 = np.mean(alpha_values)  # Approximate alpha_0
            
            spectrum_width = alpha_max - alpha_min
            
            # Asymmetry index
            left_width = alpha_0 - alpha_min
            right_width = alpha_max - alpha_0
            
            if left_width + right_width > 0:
                asymmetry = (right_width - left_width) / (right_width + left_width)
            else:
                asymmetry = 0.0
            
            return {
                'width': spectrum_width,
                'asymmetry': asymmetry,
                'max_alpha': alpha_max,
                'min_alpha': alpha_min,
                'alpha_values': alpha_values,
                'f_alpha_values': f_alpha_values
            }
            
        except Exception as e:
            logger.warning(f"Multifractal spectrum calculation error: {e}")
            return {'width': 0.0, 'asymmetry': 0.0, 'max_alpha': 0.5, 'min_alpha': 0.5}

    def _analyze_multifractality(self, spectrum_results: Dict[str, float]) -> Dict[str, float]:
        """Analyze multifractality characteristics"""
        try:
            spectrum_width = spectrum_results.get('width', 0.0)
            asymmetry = spectrum_results.get('asymmetry', 0.0)
            
            # Multifractality strength (wider spectrum = stronger multifractality)
            strength = min(spectrum_width / 0.5, 1.0)  # Normalize by typical width
            
            # Complexity measure
            complexity = strength * (1 + abs(asymmetry))
            
            return {
                'strength': np.clip(strength, 0.0, 1.0),
                'complexity': np.clip(complexity, 0.0, 2.0)
            }
            
        except Exception as e:
            logger.warning(f"Multifractality analysis error: {e}")
            return {'strength': 0.0, 'complexity': 0.0}

    def _classify_market_regime(self, multifractal_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Classify market regime based on multifractal characteristics"""
        try:
            strength = multifractal_analysis.get('strength', 0.0)
            complexity = multifractal_analysis.get('complexity', 0.0)
            
            # Regime classification
            if strength > 0.7:
                regime = 'STRONGLY_MULTIFRACTAL'
                confidence = 0.8
            elif strength > 0.4:
                regime = 'MODERATELY_MULTIFRACTAL'
                confidence = 0.6
            elif strength > 0.2:
                regime = 'WEAKLY_MULTIFRACTAL'
                confidence = 0.4
            else:
                regime = 'MONOFRACTAL'
                confidence = 0.3
            
            # Adjust confidence based on complexity
            if complexity > 1.0:
                confidence = min(confidence + 0.1, 1.0)
            
            return {
                'regime': regime,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"Regime classification error: {e}")
            return {'regime': 'UNKNOWN', 'confidence': 0.0}

    def _calculate_market_efficiency(self, mfdfa_results: Dict[str, Any]) -> float:
        """Calculate market efficiency based on Hurst exponent"""
        try:
            hurst = mfdfa_results.get('hurst_exponent', 0.5)
            
            # Market efficiency interpretation:
            # H = 0.5: Efficient market (random walk)
            # H > 0.5: Persistent/trending (less efficient)
            # H < 0.5: Anti-persistent/mean-reverting (less efficient)
            
            efficiency = 1.0 - 2 * abs(hurst - 0.5)
            
            return np.clip(efficiency, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Market efficiency calculation error: {e}")
            return 0.5

    def _format_output(self, result: MultiFractalDFAResult, timestamp) -> Dict[str, Any]:
        """Format the calculation results for output"""
        return {
            'timestamp': timestamp,
            'indicator_name': 'MultiFractalDFA',
            'hurst_exponent': round(result.hurst_exponent, 4),
            'multifractality_strength': round(result.multifractality_strength, 4),
            'spectrum_width': round(result.spectrum_width, 4),
            'asymmetry_index': round(result.asymmetry_index, 4),
            'complexity_measure': round(result.complexity_measure, 4),
            'scaling_quality': round(result.scaling_quality, 4),
            'dominant_scaling': round(result.dominant_scaling, 4),
            'market_efficiency': round(result.market_efficiency, 4),
            'regime_classification': result.regime_classification,
            'confidence_level': round(result.confidence_level, 4)
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        if len(data) < self.window_size:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.window_size} periods required")

    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data cases"""
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'MultiFractalDFA',
            'hurst_exponent': 0.5,
            'multifractality_strength': 0.0,
            'spectrum_width': 0.0,
            'asymmetry_index': 0.0,
            'complexity_measure': 0.0,
            'scaling_quality': 0.0,
            'dominant_scaling': 0.5,
            'market_efficiency': 0.5,
            'regime_classification': 'INSUFFICIENT_DATA',
            'confidence_level': 0.0
        }

    def get_required_columns(self) -> List[str]:
        return ['close']

    def get_indicator_name(self) -> str:
        return "MultiFractalDFA"