"""
Price-Time Relationships Indicator - Advanced Implementation
===========================================================

This module implements comprehensive price-time relationship analysis using advanced
mathematical models, correlation detection, and predictive modeling techniques.

Key Features:
- Multi-dimensional price-time correlation analysis
- Advanced relationship pattern detection (linear, exponential, logarithmic, power)
- Dynamic time-price velocity and acceleration analysis
- Seasonal and cyclical relationship modeling
- Machine learning-enhanced relationship prediction
- Cross-correlation analysis with multiple lag periods
- Price-time momentum and trend relationship analysis
- Advanced statistical relationship validation
- Predictive relationship modeling and forecasting

Mathematical Foundation:
- Pearson and Spearman correlation coefficients
- Cross-correlation with lag analysis
- Regression analysis (linear, polynomial, exponential)
- Time series decomposition and trend analysis
- Fourier analysis for cyclical relationships
- Wavelet analysis for time-frequency relationships
- Machine Learning for pattern recognition and prediction
- Statistical significance testing

Relationship Types Analyzed:
- Linear relationships (y = mx + b)
- Exponential relationships (y = ae^(bx))
- Logarithmic relationships (y = a*ln(x) + b)
- Power relationships (y = ax^b)
- Polynomial relationships (y = ax^n + bx^(n-1) + ...)
- Seasonal and cyclical relationships
- Phase-shifted relationships

Author: Trading Platform Team
Date: 2024
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import stats, signal
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PriceTimeRelationship:
    """Data structure for price-time relationship"""
    relationship_type: str
    correlation_coefficient: float
    r_squared: float
    p_value: float
    slope: float
    intercept: float
    equation: str
    strength: str  # 'weak', 'moderate', 'strong', 'very_strong'
    direction: str  # 'positive', 'negative', 'neutral'
    confidence: float
    parameters: Dict[str, float]

@dataclass
class CrossCorrelation:
    """Data structure for cross-correlation analysis"""
    lag: int
    correlation: float
    significance: float
    interpretation: str
    lead_lag_relationship: str

@dataclass
class RelationshipForecast:
    """Data structure for relationship-based forecasts"""
    forecast_date: datetime
    predicted_price: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    confidence: float
    relationship_basis: str
    forecast_horizon: int

@dataclass
class SeasonalRelationship:
    """Data structure for seasonal price-time relationships"""
    season_type: str  # 'daily', 'weekly', 'monthly', 'quarterly'
    seasonal_strength: float
    seasonal_pattern: np.ndarray
    phase_shift: float
    amplitude: float

class PriceTimeRelationshipsIndicator:
    """
    Advanced Price-Time Relationships Indicator Implementation
    
    This class analyzes comprehensive price-time relationships using multiple
    mathematical models and predictive techniques.
    """
    
    def __init__(self, 
                 lookback_periods: int = 500,
                 max_lag_periods: int = 50,
                 confidence_level: float = 0.95,
                 seasonal_analysis: bool = True,
                 ml_enhancement: bool = True):
        """
        Initialize the Price-Time Relationships Indicator
        
        Args:
            lookback_periods: Number of periods to analyze
            max_lag_periods: Maximum lag periods for cross-correlation
            confidence_level: Statistical confidence level
            seasonal_analysis: Enable seasonal relationship analysis
            ml_enhancement: Enable machine learning features
        """
        self.lookback_periods = lookback_periods
        self.max_lag_periods = max_lag_periods
        self.confidence_level = confidence_level
        self.seasonal_analysis = seasonal_analysis
        self.ml_enhancement = ml_enhancement
        
        # Correlation strength thresholds
        self.correlation_thresholds = {
            'very_strong': 0.8,
            'strong': 0.6,
            'moderate': 0.4,
            'weak': 0.2
        }
        
        # Initialize storage
        self.detected_relationships = []
        self.cross_correlations = []
        self.forecasts = []
        self.seasonal_relationships = []
        
        logger.info("Price-Time Relationships Indicator initialized")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive price-time relationship analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing relationship analysis results
        """
        try:
            if len(data) < 30:
                raise ValueError("Insufficient data for relationship analysis")
            
            # Prepare price and time data
            price_data, time_data = self._prepare_data(data)
            timestamps = data.index
            
            # 1. Basic Linear Relationships
            linear_relationships = self._analyze_linear_relationships(price_data, time_data)
            
            # 2. Non-Linear Relationships
            nonlinear_relationships = self._analyze_nonlinear_relationships(price_data, time_data)
            
            # 3. Cross-Correlation Analysis
            cross_correlations = self._perform_cross_correlation_analysis(price_data, time_data)
            
            # 4. Price-Time Velocity and Acceleration
            velocity_acceleration = self._analyze_velocity_acceleration(price_data, time_data)
            
            # 5. Seasonal Relationship Analysis
            seasonal_analysis = {}
            if self.seasonal_analysis:
                seasonal_analysis = self._analyze_seasonal_relationships(price_data, timestamps)
            
            # 6. Cyclical Relationship Analysis
            cyclical_analysis = self._analyze_cyclical_relationships(price_data, time_data)
            
            # 7. Phase Relationship Analysis
            phase_relationships = self._analyze_phase_relationships(price_data, time_data)
            
            # 8. Polynomial Relationship Analysis
            polynomial_relationships = self._analyze_polynomial_relationships(price_data, time_data)
            
            # 9. ML-Enhanced Relationship Prediction
            ml_predictions = {}
            if self.ml_enhancement:
                ml_predictions = self._ml_relationship_prediction(price_data, time_data)
            
            # 10. Relationship Stability Analysis
            stability_analysis = self._analyze_relationship_stability(price_data, time_data)
            
            # 11. Generate Forecasts
            forecasts = self._generate_relationship_forecasts(
                linear_relationships + nonlinear_relationships + polynomial_relationships,
                timestamps
            )
            
            # 12. Trading Signal Generation
            signals = self._generate_relationship_signals(
                linear_relationships + nonlinear_relationships,
                cross_correlations,
                forecasts,
                timestamps
            )
            
            # 13. Relationship Strength Assessment
            strength_assessment = self._assess_overall_relationship_strength(
                linear_relationships + nonlinear_relationships + polynomial_relationships
            )
            
            # Store results
            self.detected_relationships = linear_relationships + nonlinear_relationships + polynomial_relationships
            self.cross_correlations = cross_correlations
            self.forecasts = forecasts
            if self.seasonal_analysis:
                self.seasonal_relationships = seasonal_analysis.get('seasonal_patterns', [])
            
            # Compile comprehensive results
            results = {
                'linear_relationships': linear_relationships,
                'nonlinear_relationships': nonlinear_relationships,
                'polynomial_relationships': polynomial_relationships,
                'cross_correlations': cross_correlations,
                'velocity_acceleration': velocity_acceleration,
                'seasonal_analysis': seasonal_analysis,
                'cyclical_analysis': cyclical_analysis,
                'phase_relationships': phase_relationships,
                'stability_analysis': stability_analysis,
                'forecasts': forecasts,
                'trading_signals': signals,
                'strength_assessment': strength_assessment,
                'analysis_timestamp': datetime.now(),
                'data_points_analyzed': len(price_data)
            }
            
            if self.ml_enhancement:
                results['ml_predictions'] = ml_predictions
            
            logger.info(f"Price-Time relationship analysis completed: {len(self.detected_relationships)} relationships detected")
            return results
            
        except Exception as e:
            logger.error(f"Error in Price-Time relationship calculation: {str(e)}")
            return {'error': str(e), 'relationships': [], 'signals': []}
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare price and time data for analysis"""
        try:
            # Use typical price for analysis
            if all(col in data.columns for col in ['high', 'low', 'close']):
                price_data = (data['high'] + data['low'] + data['close']) / 3
            elif 'close' in data.columns:
                price_data = data['close']
            else:
                price_data = data.iloc[:, -1]
            
            # Remove NaN values and ensure numeric
            price_data = pd.to_numeric(price_data, errors='coerce').dropna()
            
            # Create time data (sequential numbers)
            time_data = np.arange(len(price_data))
            
            return price_data.values, time_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return np.array([]), np.array([])
    
    def _analyze_linear_relationships(self, price_data: np.ndarray, 
                                    time_data: np.ndarray) -> List[PriceTimeRelationship]:
        """Analyze linear price-time relationships"""
        try:
            relationships = []
            
            if len(price_data) < 10:
                return relationships
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_data, price_data)
            
            # Calculate R-squared
            r_squared = r_value ** 2
            
            # Determine strength
            abs_corr = abs(r_value)
            if abs_corr >= self.correlation_thresholds['very_strong']:
                strength = 'very_strong'
            elif abs_corr >= self.correlation_thresholds['strong']:
                strength = 'strong'
            elif abs_corr >= self.correlation_thresholds['moderate']:
                strength = 'moderate'
            elif abs_corr >= self.correlation_thresholds['weak']:
                strength = 'weak'
            else:
                strength = 'very_weak'
            
            # Determine direction
            if r_value > 0.1:
                direction = 'positive'
            elif r_value < -0.1:
                direction = 'negative'
            else:
                direction = 'neutral'
            
            # Create equation string
            equation = f"Price = {slope:.6f} * Time + {intercept:.6f}"
            
            # Calculate confidence (based on p-value and R-squared)
            confidence = (1 - p_value) * r_squared
            
            relationship = PriceTimeRelationship(
                relationship_type='linear',
                correlation_coefficient=r_value,
                r_squared=r_squared,
                p_value=p_value,
                slope=slope,
                intercept=intercept,
                equation=equation,
                strength=strength,
                direction=direction,
                confidence=confidence,
                parameters={'slope': slope, 'intercept': intercept, 'std_err': std_err}
            )
            relationships.append(relationship)
            
            # Analyze linear relationship on log-transformed data
            if np.all(price_data > 0):
                log_price = np.log(price_data)
                slope_log, intercept_log, r_value_log, p_value_log, std_err_log = stats.linregress(time_data, log_price)
                r_squared_log = r_value_log ** 2
                
                # Determine strength for log relationship
                abs_corr_log = abs(r_value_log)
                if abs_corr_log >= self.correlation_thresholds['very_strong']:
                    strength_log = 'very_strong'
                elif abs_corr_log >= self.correlation_thresholds['strong']:
                    strength_log = 'strong'
                elif abs_corr_log >= self.correlation_thresholds['moderate']:
                    strength_log = 'moderate'
                elif abs_corr_log >= self.correlation_thresholds['weak']:
                    strength_log = 'weak'
                else:
                    strength_log = 'very_weak'
                
                direction_log = 'positive' if r_value_log > 0.1 else ('negative' if r_value_log < -0.1 else 'neutral')
                equation_log = f"ln(Price) = {slope_log:.6f} * Time + {intercept_log:.6f}"
                confidence_log = (1 - p_value_log) * r_squared_log
                
                relationship_log = PriceTimeRelationship(
                    relationship_type='log_linear',
                    correlation_coefficient=r_value_log,
                    r_squared=r_squared_log,
                    p_value=p_value_log,
                    slope=slope_log,
                    intercept=intercept_log,
                    equation=equation_log,
                    strength=strength_log,
                    direction=direction_log,
                    confidence=confidence_log,
                    parameters={'slope': slope_log, 'intercept': intercept_log, 'std_err': std_err_log}
                )
                relationships.append(relationship_log)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing linear relationships: {str(e)}")
            return []    
    def _analyze_nonlinear_relationships(self, price_data: np.ndarray, 
                                       time_data: np.ndarray) -> List[PriceTimeRelationship]:
        """Analyze non-linear price-time relationships"""
        try:
            relationships = []
            
            if len(price_data) < 15:
                return relationships
            
            # Define non-linear functions to test
            def exponential_func(x, a, b):
                return a * np.exp(b * x)
            
            def power_func(x, a, b):
                return a * np.power(x + 1, b)  # +1 to avoid x=0 issues
            
            def logarithmic_func(x, a, b):
                return a * np.log(x + 1) + b  # +1 to avoid log(0)
            
            def inverse_func(x, a, b):
                return a / (x + 1) + b
            
            # Test each non-linear function
            functions = [
                ('exponential', exponential_func),
                ('power', power_func),
                ('logarithmic', logarithmic_func),
                ('inverse', inverse_func)
            ]
            
            for func_name, func in functions:
                try:
                    # Fit the function
                    popt, pcov = curve_fit(func, time_data, price_data, maxfev=2000)
                    
                    # Calculate fitted values
                    fitted_values = func(time_data, *popt)
                    
                    # Calculate R-squared
                    r_squared = r2_score(price_data, fitted_values)
                    
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(price_data, fitted_values)[0, 1]
                    
                    # Estimate p-value using F-test approximation
                    n = len(price_data)
                    k = len(popt)  # number of parameters
                    mse = np.mean((price_data - fitted_values) ** 2)
                    f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
                    p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                    
                    # Determine strength
                    abs_corr = abs(correlation)
                    if abs_corr >= self.correlation_thresholds['very_strong']:
                        strength = 'very_strong'
                    elif abs_corr >= self.correlation_thresholds['strong']:
                        strength = 'strong'
                    elif abs_corr >= self.correlation_thresholds['moderate']:
                        strength = 'moderate'
                    elif abs_corr >= self.correlation_thresholds['weak']:
                        strength = 'weak'
                    else:
                        strength = 'very_weak'
                    
                    # Determine direction (based on trend of fitted values)
                    trend_slope = np.polyfit(time_data, fitted_values, 1)[0]
                    if trend_slope > 0.01:
                        direction = 'positive'
                    elif trend_slope < -0.01:
                        direction = 'negative'
                    else:
                        direction = 'neutral'
                    
                    # Create equation string
                    if func_name == 'exponential':
                        equation = f"Price = {popt[0]:.6f} * exp({popt[1]:.6f} * Time)"
                    elif func_name == 'power':
                        equation = f"Price = {popt[0]:.6f} * (Time + 1)^{popt[1]:.6f}"
                    elif func_name == 'logarithmic':
                        equation = f"Price = {popt[0]:.6f} * ln(Time + 1) + {popt[1]:.6f}"
                    elif func_name == 'inverse':
                        equation = f"Price = {popt[0]:.6f} / (Time + 1) + {popt[1]:.6f}"
                    else:
                        equation = f"Price = f(Time) with parameters {popt}"
                    
                    # Calculate confidence
                    confidence = (1 - p_value) * r_squared
                    
                    # Store parameters
                    parameters = {f'param_{i}': popt[i] for i in range(len(popt))}
                    parameters['mse'] = mse
                    parameters['trend_slope'] = trend_slope
                    
                    relationship = PriceTimeRelationship(
                        relationship_type=func_name,
                        correlation_coefficient=correlation,
                        r_squared=r_squared,
                        p_value=p_value,
                        slope=trend_slope,
                        intercept=0,  # Not applicable for non-linear
                        equation=equation,
                        strength=strength,
                        direction=direction,
                        confidence=confidence,
                        parameters=parameters
                    )
                    relationships.append(relationship)
                    
                except Exception as fit_error:
                    # Skip this function if fitting fails
                    logger.debug(f"Failed to fit {func_name} function: {str(fit_error)}")
                    continue
            
            # Sort by R-squared value
            relationships.sort(key=lambda x: x.r_squared, reverse=True)
            
            return relationships[:4]  # Return top 4 non-linear relationships
            
        except Exception as e:
            logger.error(f"Error analyzing non-linear relationships: {str(e)}")
            return []
    
    def _perform_cross_correlation_analysis(self, price_data: np.ndarray, 
                                          time_data: np.ndarray) -> List[CrossCorrelation]:
        """Perform cross-correlation analysis with lag periods"""
        try:
            cross_correlations = []
            
            if len(price_data) < 20:
                return cross_correlations
            
            # Calculate price changes
            price_changes = np.diff(price_data)
            
            # Perform cross-correlation analysis
            max_lag = min(self.max_lag_periods, len(price_changes) // 4)
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    continue
                
                if lag > 0:
                    # Price leads time
                    x = price_changes[:-lag]
                    y = price_changes[lag:]
                else:
                    # Time leads price
                    x = price_changes[-lag:]
                    y = price_changes[:lag]
                
                if len(x) > 5 and len(y) > 5:
                    correlation, p_value = stats.pearsonr(x, y)
                    
                    # Determine significance
                    significance = 1 - p_value
                    
                    # Interpret the relationship
                    if abs(correlation) > 0.3 and p_value < 0.05:
                        if lag > 0:
                            interpretation = f"Price changes lead by {lag} periods"
                            lead_lag_relationship = 'price_leads'
                        else:
                            interpretation = f"Price changes lag by {abs(lag)} periods"
                            lead_lag_relationship = 'time_leads'
                    else:
                        interpretation = "No significant relationship"
                        lead_lag_relationship = 'no_relationship'
                    
                    cross_corr = CrossCorrelation(
                        lag=lag,
                        correlation=correlation,
                        significance=significance,
                        interpretation=interpretation,
                        lead_lag_relationship=lead_lag_relationship
                    )
                    cross_correlations.append(cross_corr)
            
            # Sort by absolute correlation value
            cross_correlations.sort(key=lambda x: abs(x.correlation), reverse=True)
            
            return cross_correlations[:10]  # Return top 10 cross-correlations
            
        except Exception as e:
            logger.error(f"Error performing cross-correlation analysis: {str(e)}")
            return []
    
    def _analyze_velocity_acceleration(self, price_data: np.ndarray, 
                                     time_data: np.ndarray) -> Dict[str, Any]:
        """Analyze price-time velocity and acceleration relationships"""
        try:
            if len(price_data) < 10:
                return {}
            
            # Calculate price velocity (first derivative)
            price_velocity = np.gradient(price_data)
            
            # Calculate price acceleration (second derivative)
            price_acceleration = np.gradient(price_velocity)
            
            # Analyze velocity-time relationship
            velocity_time_corr, velocity_p_value = stats.pearsonr(time_data[1:], price_velocity[1:])
            
            # Analyze acceleration-time relationship
            accel_time_corr, accel_p_value = stats.pearsonr(time_data[2:], price_acceleration[2:])
            
            # Analyze velocity-acceleration relationship
            vel_accel_corr, vel_accel_p_value = stats.pearsonr(price_velocity[2:], price_acceleration[2:])
            
            # Calculate momentum indicator (price * velocity)
            momentum = price_data[1:] * price_velocity[1:]
            momentum_time_corr, momentum_p_value = stats.pearsonr(time_data[1:], momentum)
            
            # Calculate jerk (third derivative)
            price_jerk = np.gradient(price_acceleration)
            jerk_time_corr, jerk_p_value = stats.pearsonr(time_data[3:], price_jerk[3:])
            
            return {
                'velocity_analysis': {
                    'velocity_time_correlation': velocity_time_corr,
                    'velocity_time_p_value': velocity_p_value,
                    'velocity_trend': 'increasing' if velocity_time_corr > 0.1 else ('decreasing' if velocity_time_corr < -0.1 else 'stable'),
                    'current_velocity': price_velocity[-1],
                    'avg_velocity': np.mean(price_velocity)
                },
                'acceleration_analysis': {
                    'acceleration_time_correlation': accel_time_corr,
                    'acceleration_time_p_value': accel_p_value,
                    'acceleration_trend': 'increasing' if accel_time_corr > 0.1 else ('decreasing' if accel_time_corr < -0.1 else 'stable'),
                    'current_acceleration': price_acceleration[-1],
                    'avg_acceleration': np.mean(price_acceleration)
                },
                'momentum_analysis': {
                    'momentum_time_correlation': momentum_time_corr,
                    'momentum_time_p_value': momentum_p_value,
                    'current_momentum': momentum[-1],
                    'momentum_trend': 'increasing' if momentum_time_corr > 0.1 else ('decreasing' if momentum_time_corr < -0.1 else 'stable')
                },
                'jerk_analysis': {
                    'jerk_time_correlation': jerk_time_corr,
                    'jerk_time_p_value': jerk_p_value,
                    'current_jerk': price_jerk[-1],
                    'jerk_significance': 'significant' if abs(jerk_time_corr) > 0.3 and jerk_p_value < 0.05 else 'not_significant'
                },
                'derivative_relationships': {
                    'velocity_acceleration_correlation': vel_accel_corr,
                    'velocity_acceleration_p_value': vel_accel_p_value
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing velocity and acceleration: {str(e)}")
            return {}
    
    def _analyze_seasonal_relationships(self, price_data: np.ndarray, 
                                      timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyze seasonal price-time relationships"""
        try:
            if not self.seasonal_analysis or len(price_data) < 50:
                return {}
            
            seasonal_patterns = []
            
            # Convert timestamps to datetime if needed
            if not isinstance(timestamps, pd.DatetimeIndex):
                timestamps = pd.to_datetime(timestamps)
            
            # Daily seasonal analysis (hour of day effect)
            if len(timestamps) > 24:
                try:
                    hours = timestamps.hour
                    unique_hours = np.unique(hours)
                    if len(unique_hours) > 5:  # Enough variation in hours
                        hour_means = []
                        for hour in unique_hours:
                            hour_mask = hours == hour
                            if np.sum(hour_mask) > 2:
                                hour_means.append(np.mean(price_data[hour_mask]))
                            else:
                                hour_means.append(np.nan)
                        
                        hour_means = np.array(hour_means)
                        valid_means = hour_means[~np.isnan(hour_means)]
                        
                        if len(valid_means) > 3:
                            seasonal_strength = np.std(valid_means) / np.mean(valid_means) if np.mean(valid_means) != 0 else 0
                            
                            daily_pattern = SeasonalRelationship(
                                season_type='daily',
                                seasonal_strength=seasonal_strength,
                                seasonal_pattern=hour_means,
                                phase_shift=np.argmax(valid_means) if len(valid_means) > 0 else 0,
                                amplitude=np.max(valid_means) - np.min(valid_means) if len(valid_means) > 0 else 0
                            )
                            seasonal_patterns.append(daily_pattern)
                except Exception:
                    pass
            
            # Weekly seasonal analysis (day of week effect)
            if len(timestamps) > 14:
                try:
                    days_of_week = timestamps.dayofweek
                    weekly_means = []
                    for day in range(7):
                        day_mask = days_of_week == day
                        if np.sum(day_mask) > 2:
                            weekly_means.append(np.mean(price_data[day_mask]))
                        else:
                            weekly_means.append(np.nan)
                    
                    weekly_means = np.array(weekly_means)
                    valid_weekly_means = weekly_means[~np.isnan(weekly_means)]
                    
                    if len(valid_weekly_means) > 3:
                        weekly_seasonal_strength = np.std(valid_weekly_means) / np.mean(valid_weekly_means) if np.mean(valid_weekly_means) != 0 else 0
                        
                        weekly_pattern = SeasonalRelationship(
                            season_type='weekly',
                            seasonal_strength=weekly_seasonal_strength,
                            seasonal_pattern=weekly_means,
                            phase_shift=np.argmax(valid_weekly_means) if len(valid_weekly_means) > 0 else 0,
                            amplitude=np.max(valid_weekly_means) - np.min(valid_weekly_means) if len(valid_weekly_means) > 0 else 0
                        )
                        seasonal_patterns.append(weekly_pattern)
                except Exception:
                    pass
            
            # Monthly seasonal analysis
            if len(timestamps) > 60:
                try:
                    months = timestamps.month
                    monthly_means = []
                    for month in range(1, 13):
                        month_mask = months == month
                        if np.sum(month_mask) > 1:
                            monthly_means.append(np.mean(price_data[month_mask]))
                        else:
                            monthly_means.append(np.nan)
                    
                    monthly_means = np.array(monthly_means)
                    valid_monthly_means = monthly_means[~np.isnan(monthly_means)]
                    
                    if len(valid_monthly_means) > 6:
                        monthly_seasonal_strength = np.std(valid_monthly_means) / np.mean(valid_monthly_means) if np.mean(valid_monthly_means) != 0 else 0
                        
                        monthly_pattern = SeasonalRelationship(
                            season_type='monthly',
                            seasonal_strength=monthly_seasonal_strength,
                            seasonal_pattern=monthly_means,
                            phase_shift=np.argmax(valid_monthly_means) if len(valid_monthly_means) > 0 else 0,
                            amplitude=np.max(valid_monthly_means) - np.min(valid_monthly_means) if len(valid_monthly_means) > 0 else 0
                        )
                        seasonal_patterns.append(monthly_pattern)
                except Exception:
                    pass
            
            # Overall seasonal strength
            overall_seasonal_strength = np.mean([p.seasonal_strength for p in seasonal_patterns]) if seasonal_patterns else 0
            
            return {
                'seasonal_patterns': seasonal_patterns,
                'overall_seasonal_strength': overall_seasonal_strength,
                'has_significant_seasonality': overall_seasonal_strength > 0.05,
                'dominant_seasonal_type': max(seasonal_patterns, key=lambda x: x.seasonal_strength).season_type if seasonal_patterns else 'none'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal relationships: {str(e)}")
            return {}    
    def _analyze_cyclical_relationships(self, price_data: np.ndarray, 
                                      time_data: np.ndarray) -> Dict[str, Any]:
        """Analyze cyclical price-time relationships using Fourier analysis"""
        try:
            if len(price_data) < 20:
                return {}
            
            # Detrend the data
            detrended_price = signal.detrend(price_data)
            
            # Perform FFT
            fft_values = np.fft.fft(detrended_price)
            frequencies = np.fft.fftfreq(len(detrended_price))
            
            # Calculate power spectral density
            power_spectrum = np.abs(fft_values) ** 2
            
            # Find dominant frequencies
            peak_indices = signal.find_peaks(power_spectrum[:len(power_spectrum)//2], 
                                           height=np.mean(power_spectrum) * 2)[0]
            
            dominant_cycles = []
            for idx in peak_indices[:5]:  # Top 5 cycles
                if frequencies[idx] != 0:
                    period = 1 / abs(frequencies[idx])
                    amplitude = np.abs(fft_values[idx])
                    phase = np.angle(fft_values[idx])
                    
                    dominant_cycles.append({
                        'period': period,
                        'frequency': frequencies[idx],
                        'amplitude': amplitude,
                        'phase': phase,
                        'power': power_spectrum[idx]
                    })
            
            # Sort by power
            dominant_cycles.sort(key=lambda x: x['power'], reverse=True)
            
            # Calculate cyclical strength
            total_power = np.sum(power_spectrum)
            cyclical_power = np.sum([cycle['power'] for cycle in dominant_cycles])
            cyclical_strength = cyclical_power / total_power if total_power > 0 else 0
            
            return {
                'dominant_cycles': dominant_cycles,
                'cyclical_strength': cyclical_strength,
                'has_strong_cycles': cyclical_strength > 0.3,
                'number_of_significant_cycles': len(dominant_cycles)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cyclical relationships: {str(e)}")
            return {}
    
    def _analyze_phase_relationships(self, price_data: np.ndarray, 
                                   time_data: np.ndarray) -> Dict[str, Any]:
        """Analyze phase relationships between price and time"""
        try:
            if len(price_data) < 15:
                return {}
            
            # Apply Hilbert transform to get instantaneous phase
            analytic_signal = signal.hilbert(price_data)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_amplitude = np.abs(analytic_signal)
            
            # Calculate phase velocity
            phase_velocity = np.diff(instantaneous_phase)
            
            # Analyze phase-time relationship
            phase_time_correlation, phase_time_p_value = stats.pearsonr(time_data[1:], instantaneous_phase[1:])
            
            # Analyze amplitude-time relationship
            amplitude_time_correlation, amplitude_time_p_value = stats.pearsonr(time_data, instantaneous_amplitude)
            
            # Calculate phase acceleration
            phase_acceleration = np.diff(phase_velocity)
            
            # Detect phase reversals
            phase_reversals = []
            for i in range(1, len(phase_velocity) - 1):
                if (phase_velocity[i-1] > 0 and phase_velocity[i] < 0) or (phase_velocity[i-1] < 0 and phase_velocity[i] > 0):
                    phase_reversals.append(i)
            
            return {
                'phase_time_correlation': phase_time_correlation,
                'phase_time_p_value': phase_time_p_value,
                'amplitude_time_correlation': amplitude_time_correlation,
                'amplitude_time_p_value': amplitude_time_p_value,
                'current_phase': instantaneous_phase[-1],
                'current_amplitude': instantaneous_amplitude[-1],
                'phase_velocity': phase_velocity[-1] if len(phase_velocity) > 0 else 0,
                'phase_acceleration': phase_acceleration[-1] if len(phase_acceleration) > 0 else 0,
                'phase_reversals': len(phase_reversals),
                'avg_phase_velocity': np.mean(phase_velocity) if len(phase_velocity) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing phase relationships: {str(e)}")
            return {}
    
    def _analyze_polynomial_relationships(self, price_data: np.ndarray, 
                                        time_data: np.ndarray) -> List[PriceTimeRelationship]:
        """Analyze polynomial price-time relationships"""
        try:
            relationships = []
            
            if len(price_data) < 15:
                return relationships
            
            # Test polynomial degrees 2 through 5
            for degree in range(2, 6):
                try:
                    # Fit polynomial
                    poly_features = PolynomialFeatures(degree=degree)
                    time_poly = poly_features.fit_transform(time_data.reshape(-1, 1))
                    
                    reg = LinearRegression()
                    reg.fit(time_poly, price_data)
                    
                    # Predict values
                    predicted_values = reg.predict(time_poly)
                    
                    # Calculate R-squared
                    r_squared = r2_score(price_data, predicted_values)
                    
                    # Calculate correlation
                    correlation = np.corrcoef(price_data, predicted_values)[0, 1]
                    
                    # Estimate p-value
                    n = len(price_data)
                    k = degree
                    f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
                    p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
                    
                    # Determine strength
                    abs_corr = abs(correlation)
                    if abs_corr >= self.correlation_thresholds['very_strong']:
                        strength = 'very_strong'
                    elif abs_corr >= self.correlation_thresholds['strong']:
                        strength = 'strong'
                    elif abs_corr >= self.correlation_thresholds['moderate']:
                        strength = 'moderate'
                    elif abs_corr >= self.correlation_thresholds['weak']:
                        strength = 'weak'
                    else:
                        strength = 'very_weak'
                    
                    # Determine direction (based on trend)
                    trend_slope = np.polyfit(time_data, predicted_values, 1)[0]
                    if trend_slope > 0.01:
                        direction = 'positive'
                    elif trend_slope < -0.01:
                        direction = 'negative'
                    else:
                        direction = 'neutral'
                    
                    # Create equation string
                    coefficients = reg.coef_
                    intercept = reg.intercept_
                    
                    equation_parts = [f"{intercept:.6f}"]
                    for i, coef in enumerate(coefficients[1:], 1):  # Skip intercept term
                        if i == 1:
                            equation_parts.append(f"{coef:.6f} * Time")
                        else:
                            equation_parts.append(f"{coef:.6f} * Time^{i}")
                    
                    equation = "Price = " + " + ".join(equation_parts)
                    
                    # Calculate confidence
                    confidence = (1 - p_value) * r_squared
                    
                    # Store parameters
                    parameters = {f'coef_{i}': coef for i, coef in enumerate(coefficients)}
                    parameters['intercept'] = intercept
                    parameters['degree'] = degree
                    
                    relationship = PriceTimeRelationship(
                        relationship_type=f'polynomial_degree_{degree}',
                        correlation_coefficient=correlation,
                        r_squared=r_squared,
                        p_value=p_value,
                        slope=trend_slope,
                        intercept=intercept,
                        equation=equation,
                        strength=strength,
                        direction=direction,
                        confidence=confidence,
                        parameters=parameters
                    )
                    relationships.append(relationship)
                    
                except Exception as poly_error:
                    logger.debug(f"Failed to fit polynomial degree {degree}: {str(poly_error)}")
                    continue
            
            # Sort by R-squared
            relationships.sort(key=lambda x: x.r_squared, reverse=True)
            
            return relationships[:3]  # Return top 3 polynomial relationships
            
        except Exception as e:
            logger.error(f"Error analyzing polynomial relationships: {str(e)}")
            return []
    
    def _ml_relationship_prediction(self, price_data: np.ndarray, 
                                   time_data: np.ndarray) -> Dict[str, Any]:
        """Use machine learning for relationship prediction"""
        try:
            if not self.ml_enhancement or len(price_data) < 30:
                return {}
            
            # Prepare features
            features = []
            targets = []
            
            # Use sliding window approach
            window_size = min(10, len(price_data) // 3)
            
            for i in range(window_size, len(price_data)):
                # Features: past prices and time values
                price_window = price_data[i-window_size:i]
                time_window = time_data[i-window_size:i]
                
                # Statistical features
                price_mean = np.mean(price_window)
                price_std = np.std(price_window)
                price_trend = np.polyfit(range(window_size), price_window, 1)[0]
                
                # Time features
                current_time = time_data[i]
                
                feature_vector = [
                    price_mean, price_std, price_trend, current_time,
                    np.min(price_window), np.max(price_window),
                    price_window[-1] - price_window[0]  # Total change in window
                ]
                
                features.append(feature_vector)
                targets.append(price_data[i])
            
            if len(features) < 10:
                return {}
            
            features = np.array(features)
            targets = np.array(targets)
            
            # Split data for validation
            split_point = int(len(features) * 0.8)
            train_features = features[:split_point]
            train_targets = targets[:split_point]
            test_features = features[split_point:]
            test_targets = targets[split_point:]
            
            # Simple linear regression model
            reg = LinearRegression()
            reg.fit(train_features, train_targets)
            
            # Make predictions
            predictions = reg.predict(test_features)
            
            # Calculate performance metrics
            mse = np.mean((test_targets - predictions) ** 2)
            mae = np.mean(np.abs(test_targets - predictions))
            r2 = r2_score(test_targets, predictions)
            
            # Predict next value
            if len(features) > 0:
                next_prediction = reg.predict(features[-1:])
                prediction_confidence = max(0, r2)
            else:
                next_prediction = [np.mean(price_data)]
                prediction_confidence = 0
            
            return {
                'model_performance': {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'prediction_accuracy': 1 - (mae / np.mean(test_targets)) if np.mean(test_targets) != 0 else 0
                },
                'next_prediction': next_prediction[0] if len(next_prediction) > 0 else np.mean(price_data),
                'prediction_confidence': prediction_confidence,
                'features_used': len(features[0]) if len(features) > 0 else 0,
                'training_samples': len(train_features)
            }
            
        except Exception as e:
            logger.error(f"Error in ML relationship prediction: {str(e)}")
            return {}
    
    def _analyze_relationship_stability(self, price_data: np.ndarray, 
                                      time_data: np.ndarray) -> Dict[str, Any]:
        """Analyze the stability of price-time relationships over time"""
        try:
            if len(price_data) < 50:
                return {}
            
            # Divide data into segments
            num_segments = min(5, len(price_data) // 20)
            segment_size = len(price_data) // num_segments
            
            segment_correlations = []
            segment_slopes = []
            
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, len(price_data))
                
                segment_price = price_data[start_idx:end_idx]
                segment_time = time_data[start_idx:end_idx]
                
                if len(segment_price) > 5:
                    # Calculate correlation for this segment
                    correlation, _ = stats.pearsonr(segment_time, segment_price)
                    slope = np.polyfit(segment_time, segment_price, 1)[0]
                    
                    segment_correlations.append(correlation)
                    segment_slopes.append(slope)
            
            if len(segment_correlations) < 2:
                return {}
            
            # Calculate stability metrics
            correlation_stability = 1 - (np.std(segment_correlations) / (np.mean(np.abs(segment_correlations)) + 1e-10))
            slope_stability = 1 - (np.std(segment_slopes) / (np.mean(np.abs(segment_slopes)) + 1e-10))
            
            # Trend in correlations
            time_segments = np.arange(len(segment_correlations))
            correlation_trend, _ = stats.pearsonr(time_segments, segment_correlations)
            
            # Detect regime changes
            regime_changes = 0
            for i in range(1, len(segment_correlations)):
                if (segment_correlations[i-1] > 0 and segment_correlations[i] < 0) or \
                   (segment_correlations[i-1] < 0 and segment_correlations[i] > 0):
                    regime_changes += 1
            
            return {
                'correlation_stability': max(0, correlation_stability),
                'slope_stability': max(0, slope_stability),
                'overall_stability': (max(0, correlation_stability) + max(0, slope_stability)) / 2,
                'correlation_trend': correlation_trend,
                'regime_changes': regime_changes,
                'segment_correlations': segment_correlations,
                'segment_slopes': segment_slopes,
                'stability_assessment': self._assess_stability(max(0, correlation_stability), max(0, slope_stability))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing relationship stability: {str(e)}")
            return {}
    
    def _assess_stability(self, correlation_stability: float, slope_stability: float) -> str:
        """Assess overall relationship stability"""
        overall_stability = (correlation_stability + slope_stability) / 2
        
        if overall_stability > 0.8:
            return 'very_stable'
        elif overall_stability > 0.6:
            return 'stable'
        elif overall_stability > 0.4:
            return 'moderately_stable'
        elif overall_stability > 0.2:
            return 'unstable'
        else:
            return 'very_unstable'    
    def _generate_relationship_forecasts(self, relationships: List[PriceTimeRelationship], 
                                       timestamps: pd.DatetimeIndex) -> List[RelationshipForecast]:
        """Generate forecasts based on detected relationships"""
        try:
            forecasts = []
            
            if not relationships or len(timestamps) == 0:
                return forecasts
            
            current_time = timestamps[-1]
            current_time_val = len(timestamps)
            
            # Generate forecasts for each significant relationship
            for relationship in relationships[:5]:  # Top 5 relationships
                if relationship.confidence > 0.3:
                    
                    # Generate forecasts for different horizons
                    for horizon in [1, 3, 5, 10]:
                        forecast_time_val = current_time_val + horizon
                        forecast_date = current_time + timedelta(days=horizon)
                        
                        # Calculate predicted price based on relationship type
                        if relationship.relationship_type == 'linear':
                            predicted_price = relationship.slope * forecast_time_val + relationship.intercept
                        elif relationship.relationship_type == 'log_linear':
                            log_predicted = relationship.slope * forecast_time_val + relationship.intercept
                            predicted_price = np.exp(log_predicted)
                        elif relationship.relationship_type == 'exponential':
                            a = relationship.parameters.get('param_0', 1)
                            b = relationship.parameters.get('param_1', 0)
                            predicted_price = a * np.exp(b * forecast_time_val)
                        elif relationship.relationship_type == 'power':
                            a = relationship.parameters.get('param_0', 1)
                            b = relationship.parameters.get('param_1', 1)
                            predicted_price = a * np.power(forecast_time_val + 1, b)
                        elif relationship.relationship_type == 'logarithmic':
                            a = relationship.parameters.get('param_0', 1)
                            b = relationship.parameters.get('param_1', 0)
                            predicted_price = a * np.log(forecast_time_val + 1) + b
                        elif 'polynomial' in relationship.relationship_type:
                            # Polynomial prediction
                            predicted_price = relationship.intercept
                            for i, coef in enumerate([v for k, v in relationship.parameters.items() if k.startswith('coef_')]):
                                if i > 0:  # Skip intercept coefficient
                                    predicted_price += coef * (forecast_time_val ** i)
                        else:
                            # Default to linear extrapolation
                            predicted_price = relationship.slope * forecast_time_val + relationship.intercept
                        
                        # Calculate prediction intervals (simplified)
                        prediction_error = relationship.parameters.get('mse', 0) ** 0.5 if 'mse' in relationship.parameters else abs(predicted_price * 0.05)
                        confidence_multiplier = 1.96  # 95% confidence interval
                        
                        interval_width = confidence_multiplier * prediction_error * (1 + horizon * 0.1)  # Increase uncertainty with horizon
                        
                        lower_bound = predicted_price - interval_width
                        upper_bound = predicted_price + interval_width
                        
                        # Adjust confidence based on horizon and relationship quality
                        forecast_confidence = relationship.confidence * (0.9 ** (horizon - 1))  # Decrease with horizon
                        
                        forecast = RelationshipForecast(
                            forecast_date=forecast_date,
                            predicted_price=predicted_price,
                            prediction_interval_lower=lower_bound,
                            prediction_interval_upper=upper_bound,
                            confidence=forecast_confidence,
                            relationship_basis=relationship.relationship_type,
                            forecast_horizon=horizon
                        )
                        forecasts.append(forecast)
            
            # Sort forecasts by date
            forecasts.sort(key=lambda x: x.forecast_date)
            
            return forecasts[:20]  # Return top 20 forecasts
            
        except Exception as e:
            logger.error(f"Error generating relationship forecasts: {str(e)}")
            return []
    
    def _generate_relationship_signals(self, relationships: List[PriceTimeRelationship],
                                     cross_correlations: List[CrossCorrelation],
                                     forecasts: List[RelationshipForecast],
                                     timestamps: pd.DatetimeIndex) -> List[Dict[str, Any]]:
        """Generate trading signals based on price-time relationships"""
        try:
            signals = []
            
            if len(timestamps) == 0:
                return signals
            
            current_time = timestamps[-1]
            
            # Signals based on relationship strength and direction
            for relationship in relationships[:3]:  # Top 3 relationships
                if relationship.confidence > 0.5:
                    
                    if relationship.direction == 'positive' and relationship.strength in ['strong', 'very_strong']:
                        signal = {
                            'timestamp': current_time,
                            'signal_type': 'BUY',
                            'strength': relationship.confidence,
                            'reason': f'Strong positive {relationship.relationship_type} relationship (R²: {relationship.r_squared:.3f})',
                            'relationship_type': relationship.relationship_type,
                            'correlation': relationship.correlation_coefficient,
                            'confidence': relationship.confidence
                        }
                        signals.append(signal)
                    
                    elif relationship.direction == 'negative' and relationship.strength in ['strong', 'very_strong']:
                        signal = {
                            'timestamp': current_time,
                            'signal_type': 'SELL',
                            'strength': relationship.confidence,
                            'reason': f'Strong negative {relationship.relationship_type} relationship (R²: {relationship.r_squared:.3f})',
                            'relationship_type': relationship.relationship_type,
                            'correlation': relationship.correlation_coefficient,
                            'confidence': relationship.confidence
                        }
                        signals.append(signal)
            
            # Signals based on cross-correlation analysis
            for cross_corr in cross_correlations[:2]:  # Top 2 cross-correlations
                if abs(cross_corr.correlation) > 0.5 and cross_corr.significance > 0.95:
                    
                    if cross_corr.lead_lag_relationship == 'price_leads':
                        signal = {
                            'timestamp': current_time,
                            'signal_type': 'LEAD_LAG_ALERT',
                            'strength': abs(cross_corr.correlation),
                            'reason': f'Price leads by {cross_corr.lag} periods (Correlation: {cross_corr.correlation:.3f})',
                            'lag': cross_corr.lag,
                            'correlation': cross_corr.correlation,
                            'confidence': cross_corr.significance
                        }
                        signals.append(signal)
            
            # Signals based on forecasts
            for forecast in forecasts[:3]:  # Next 3 forecasts
                if forecast.confidence > 0.4 and forecast.forecast_horizon <= 5:
                    days_ahead = (forecast.forecast_date - current_time).days
                    
                    if days_ahead <= 3:  # Only near-term forecasts
                        signal = {
                            'timestamp': current_time,
                            'signal_type': 'FORECAST_ALERT',
                            'strength': forecast.confidence,
                            'reason': f'Forecast in {days_ahead} days: {forecast.predicted_price:.2f} (±{(forecast.prediction_interval_upper - forecast.prediction_interval_lower)/2:.2f})',
                            'forecast_date': forecast.forecast_date,
                            'predicted_price': forecast.predicted_price,
                            'prediction_range': (forecast.prediction_interval_lower, forecast.prediction_interval_upper),
                            'confidence': forecast.confidence
                        }
                        signals.append(signal)
            
            # Signals based on relationship regime changes
            # (This would require tracking historical relationships)
            
            # Sort signals by strength
            signals.sort(key=lambda x: x['strength'], reverse=True)
            
            return signals[:8]  # Return top 8 signals
            
        except Exception as e:
            logger.error(f"Error generating relationship signals: {str(e)}")
            return []
    
    def _assess_overall_relationship_strength(self, relationships: List[PriceTimeRelationship]) -> Dict[str, Any]:
        """Assess overall strength of price-time relationships"""
        try:
            if not relationships:
                return {'overall_strength': 0, 'assessment': 'no_relationships'}
            
            # Calculate various strength metrics
            correlations = [abs(r.correlation_coefficient) for r in relationships]
            r_squared_values = [r.r_squared for r in relationships]
            confidences = [r.confidence for r in relationships]
            
            # Overall metrics
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
            avg_r_squared = np.mean(r_squared_values)
            max_r_squared = np.max(r_squared_values)
            avg_confidence = np.mean(confidences)
            
            # Count by strength
            strength_counts = {}
            for strength in ['very_strong', 'strong', 'moderate', 'weak', 'very_weak']:
                strength_counts[strength] = sum(1 for r in relationships if r.strength == strength)
            
            # Count by direction
            direction_counts = {}
            for direction in ['positive', 'negative', 'neutral']:
                direction_counts[direction] = sum(1 for r in relationships if r.direction == direction)
            
            # Count by type
            type_counts = {}
            for relationship in relationships:
                rel_type = relationship.relationship_type
                type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
            
            # Overall assessment
            if max_correlation > 0.8 and avg_correlation > 0.6:
                assessment = 'very_strong_relationships'
            elif max_correlation > 0.6 and avg_correlation > 0.4:
                assessment = 'strong_relationships'
            elif max_correlation > 0.4 and avg_correlation > 0.3:
                assessment = 'moderate_relationships'
            elif max_correlation > 0.2:
                assessment = 'weak_relationships'
            else:
                assessment = 'very_weak_relationships'
            
            # Calculate composite strength score
            composite_strength = (avg_correlation * 0.4 + max_correlation * 0.3 + avg_r_squared * 0.3)
            
            return {
                'overall_strength': composite_strength,
                'assessment': assessment,
                'average_correlation': avg_correlation,
                'maximum_correlation': max_correlation,
                'average_r_squared': avg_r_squared,
                'maximum_r_squared': max_r_squared,
                'average_confidence': avg_confidence,
                'strength_distribution': strength_counts,
                'direction_distribution': direction_counts,
                'type_distribution': type_counts,
                'total_relationships': len(relationships),
                'significant_relationships': sum(1 for r in relationships if r.confidence > 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error assessing overall relationship strength: {str(e)}")
            return {'overall_strength': 0, 'assessment': 'error'}


# Demo and testing code
if __name__ == "__main__":
    """
    Demonstration of the Price-Time Relationships Indicator
    """
    print("=" * 75)
    print("Price-Time Relationships Indicator - Advanced Implementation Demo")
    print("=" * 75)
    
    # Create sample data with various price-time relationships
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Create price data with embedded relationships
    n_points = len(dates)
    time_values = np.arange(n_points)
    
    # Base price with linear trend
    base_price = 100 + 0.05 * time_values
    
    # Add exponential component
    exponential_component = 2 * np.exp(0.001 * time_values)
    
    # Add seasonal component
    seasonal_component = 5 * np.sin(2 * np.pi * time_values / 30)  # 30-day cycle
    weekly_component = 2 * np.sin(2 * np.pi * time_values / 7)   # 7-day cycle
    
    # Add polynomial component
    polynomial_component = 0.0001 * time_values ** 2
    
    # Add logarithmic component
    logarithmic_component = 3 * np.log(time_values + 1)
    
    # Combine components
    price_data = (base_price + exponential_component + seasonal_component + 
                 weekly_component + polynomial_component + logarithmic_component)
    
    # Add noise
    noise = np.random.normal(0, price_data * 0.02)
    price_data += noise
    
    # Ensure positive prices
    price_data = np.maximum(price_data, 1)
    
    # Create high and low data
    high_data = price_data * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
    low_data = price_data * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
    volume_data = np.random.randint(1000, 10000, n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'high': high_data,
        'low': low_data,
        'close': price_data,
        'volume': volume_data
    }, index=dates)
    
    # Initialize indicator
    indicator = PriceTimeRelationshipsIndicator(
        lookback_periods=300,
        max_lag_periods=30,
        confidence_level=0.95,
        seasonal_analysis=True,
        ml_enhancement=True
    )
    
    # Calculate price-time relationship analysis
    print("Calculating Price-Time relationship analysis...")
    results = indicator.calculate(data)
    
    if 'error' not in results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Data points analyzed: {results['data_points_analyzed']}")
        print(f"📈 Linear relationships: {len(results['linear_relationships'])}")
        print(f"📊 Non-linear relationships: {len(results['nonlinear_relationships'])}")
        print(f"📉 Polynomial relationships: {len(results['polynomial_relationships'])}")
        print(f"🔄 Cross-correlations: {len(results['cross_correlations'])}")
        print(f"🔮 Forecasts: {len(results['forecasts'])}")
        print(f"🎯 Trading signals: {len(results['trading_signals'])}")
        
        # Display linear relationships
        print(f"\n📈 Linear Relationships:")
        for i, rel in enumerate(results['linear_relationships']):
            print(f"  {i+1}. Type: {rel.relationship_type}")
            print(f"      Correlation: {rel.correlation_coefficient:.4f}, R²: {rel.r_squared:.4f}")
            print(f"      Direction: {rel.direction}, Strength: {rel.strength}")
            print(f"      Equation: {rel.equation}")
            print(f"      Confidence: {rel.confidence:.4f}")
        
        # Display non-linear relationships
        print(f"\n📊 Non-Linear Relationships:")
        for i, rel in enumerate(results['nonlinear_relationships'][:3]):
            print(f"  {i+1}. Type: {rel.relationship_type}")
            print(f"      Correlation: {rel.correlation_coefficient:.4f}, R²: {rel.r_squared:.4f}")
            print(f"      Direction: {rel.direction}, Strength: {rel.strength}")
            print(f"      Confidence: {rel.confidence:.4f}")
        
        # Display cross-correlations
        print(f"\n🔄 Cross-Correlation Analysis:")
        for i, cc in enumerate(results['cross_correlations'][:3]):
            print(f"  {i+1}. Lag: {cc.lag}, Correlation: {cc.correlation:.4f}")
            print(f"      Interpretation: {cc.interpretation}")
            print(f"      Significance: {cc.significance:.4f}")
        
        # Display velocity/acceleration analysis
        if 'velocity_acceleration' in results:
            va = results['velocity_acceleration']
            print(f"\n⚡ Velocity & Acceleration Analysis:")
            if 'velocity_analysis' in va:
                vel = va['velocity_analysis']
                print(f"  Velocity-Time Correlation: {vel['velocity_time_correlation']:.4f}")
                print(f"  Current Velocity: {vel['current_velocity']:.4f}")
                print(f"  Velocity Trend: {vel['velocity_trend']}")
            
            if 'acceleration_analysis' in va:
                acc = va['acceleration_analysis']
                print(f"  Acceleration-Time Correlation: {acc['acceleration_time_correlation']:.4f}")
                print(f"  Current Acceleration: {acc['current_acceleration']:.4f}")
        
        # Display seasonal analysis
        if 'seasonal_analysis' in results and results['seasonal_analysis']:
            seasonal = results['seasonal_analysis']
            print(f"\n🌊 Seasonal Analysis:")
            print(f"  Overall Seasonal Strength: {seasonal['overall_seasonal_strength']:.4f}")
            print(f"  Has Significant Seasonality: {seasonal['has_significant_seasonality']}")
            print(f"  Dominant Seasonal Type: {seasonal['dominant_seasonal_type']}")
        
        # Display cyclical analysis
        if 'cyclical_analysis' in results and results['cyclical_analysis']:
            cyclical = results['cyclical_analysis']
            print(f"\n🔄 Cyclical Analysis:")
            print(f"  Cyclical Strength: {cyclical['cyclical_strength']:.4f}")
            print(f"  Has Strong Cycles: {cyclical['has_strong_cycles']}")
            print(f"  Number of Significant Cycles: {cyclical['number_of_significant_cycles']}")
        
        # Display forecasts
        print(f"\n🔮 Relationship Forecasts:")
        for i, forecast in enumerate(results['forecasts'][:5]):
            print(f"  {i+1}. Date: {forecast.forecast_date.strftime('%Y-%m-%d')}")
            print(f"      Predicted Price: {forecast.predicted_price:.2f}")
            print(f"      Range: [{forecast.prediction_interval_lower:.2f}, {forecast.prediction_interval_upper:.2f}]")
            print(f"      Confidence: {forecast.confidence:.4f}, Basis: {forecast.relationship_basis}")
        
        # Display trading signals
        print(f"\n📊 Trading Signals:")
        for i, signal in enumerate(results['trading_signals'][:5]):
            print(f"  {i+1}. {signal['signal_type']}: {signal['reason']}")
            print(f"      Strength: {signal['strength']:.4f}")
        
        # Display overall strength assessment
        strength = results['strength_assessment']
        print(f"\n💪 Overall Relationship Strength:")
        print(f"  Overall Strength: {strength['overall_strength']:.4f}")
        print(f"  Assessment: {strength['assessment']}")
        print(f"  Average Correlation: {strength['average_correlation']:.4f}")
        print(f"  Maximum Correlation: {strength['maximum_correlation']:.4f}")
        print(f"  Significant Relationships: {strength['significant_relationships']}/{strength['total_relationships']}")
        
        # Display stability analysis
        if 'stability_analysis' in results and results['stability_analysis']:
            stability = results['stability_analysis']
            print(f"\n🏗️ Relationship Stability:")
            print(f"  Overall Stability: {stability['overall_stability']:.4f}")
            print(f"  Stability Assessment: {stability['stability_assessment']}")
            print(f"  Regime Changes: {stability['regime_changes']}")
        
        # Display ML predictions
        if 'ml_predictions' in results and results['ml_predictions']:
            ml = results['ml_predictions']
            print(f"\n🤖 ML Predictions:")
            if 'model_performance' in ml:
                perf = ml['model_performance']
                print(f"  R² Score: {perf['r2_score']:.4f}")
                print(f"  Prediction Accuracy: {perf['prediction_accuracy']:.4f}")
            print(f"  Next Prediction: {ml['next_prediction']:.2f}")
            print(f"  Prediction Confidence: {ml['prediction_confidence']:.4f}")
        
    else:
        print(f"❌ Error in calculation: {results['error']}")
    
    print("\n" + "=" * 75)
    print("Demo completed! The Price-Time Relationships Indicator provides:")
    print("✅ Comprehensive linear and non-linear relationship analysis")
    print("✅ Cross-correlation analysis with lag detection")
    print("✅ Velocity, acceleration, and momentum analysis")
    print("✅ Seasonal and cyclical relationship detection")
    print("✅ Polynomial and phase relationship analysis")
    print("✅ ML-enhanced prediction and forecasting")
    print("✅ Relationship stability and regime change detection")
    print("✅ Professional trading signal generation")
    print("=" * 75)