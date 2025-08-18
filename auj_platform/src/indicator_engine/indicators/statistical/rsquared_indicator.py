"""
Advanced R-Squared Indicator with Goodness-of-Fit Analysis and Model Validation

This indicator implements comprehensive R-squared analysis including:
- Multiple R-squared variants (ordinary, adjusted, predicted)
- Cross-validation and out-of-sample testing
- Model comparison and selection criteria
- Regression diagnostics and residual analysis
- Time-varying R-squared with regime detection
- Goodness-of-fit hypothesis testing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class RSquaredResult:
    """Result container for R-squared analysis"""
    r_squared: float
    adjusted_r_squared: float
    predicted_r_squared: float
    time_varying_r_squared: np.ndarray
    cross_validation_r_squared: float
    f_statistic: float
    f_p_value: float
    residual_diagnostics: Dict[str, float]
    model_selection_criteria: Dict[str, float]
    goodness_of_fit_tests: Dict[str, float]
    regression_stability: float
    outlier_influence: np.ndarray
    confidence_interval: Tuple[float, float]


class RSquaredIndicator(StandardIndicatorInterface):
    """
    Advanced R-Squared Indicator
    
    Implements comprehensive goodness-of-fit analysis with multiple
    R-squared variants and extensive model validation.
    """
    
    def __init__(self, 
                 window: int = 60,
                 cv_folds: int = 5,
                 confidence_level: float = 0.95):
        """
        Initialize R-Squared Indicator
        
        Args:
            window: Rolling window for time-varying analysis
            cv_folds: Number of cross-validation folds
            confidence_level: Confidence level for intervals
        """
        super().__init__()
        self.window = window
        self.cv_folds = cv_folds
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate R-squared analysis
        
        Args:
            data: DataFrame with dependent variable OHLCV data
            reference_data: DataFrame with independent variable data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing R-squared analysis results
        """
        try:
            if data.empty or len(data) < self.window:
                raise IndicatorCalculationError("Insufficient data for R-squared analysis")
            
            # Create reference data if not provided
            if reference_data is None:
                reference_data = self._create_reference_data(data)
            
            # Prepare dependent and independent variables
            y = data['close'].values
            X = self._prepare_independent_variables(reference_data, y)
            
            if len(y) != len(X):
                min_length = min(len(y), len(X))
                y = y[-min_length:]
                X = X[-min_length:]
            
            if len(y) < 20:
                raise IndicatorCalculationError("Insufficient aligned data")
            
            # Perform R-squared analysis
            rsquared_result = self._calculate_rsquared_analysis(y, X)
            
            # Generate trading signal
            signal = self._generate_signal(rsquared_result, y, X)
            
            return {
                'signal': signal,
                'r_squared': rsquared_result.r_squared,
                'adjusted_r_squared': rsquared_result.adjusted_r_squared,
                'predicted_r_squared': rsquared_result.predicted_r_squared,
                'time_varying_r_squared': rsquared_result.time_varying_r_squared.tolist(),
                'cross_validation_r_squared': rsquared_result.cross_validation_r_squared,
                'f_statistic': rsquared_result.f_statistic,
                'f_p_value': rsquared_result.f_p_value,
                'residual_diagnostics': rsquared_result.residual_diagnostics,
                'model_selection_criteria': rsquared_result.model_selection_criteria,
                'goodness_of_fit_tests': rsquared_result.goodness_of_fit_tests,
                'regression_stability': rsquared_result.regression_stability,
                'outlier_influence': rsquared_result.outlier_influence.tolist(),
                'confidence_interval': rsquared_result.confidence_interval,
                'strength': self._calculate_signal_strength(rsquared_result),
                'confidence': self._calculate_confidence(rsquared_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating R-squared: {str(e)}")
            raise IndicatorCalculationError(f"R-squared calculation failed: {str(e)}")
    
    def _create_reference_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic reference data for regression"""
        np.random.seed(42)
        
        # Create market-like independent variables
        n = len(data)
        
        # Market index (correlated with price)
        market_factor = np.cumsum(np.random.normal(0, 0.01, n)) + data['close'].iloc[0]
        
        # Volume factor
        volume_factor = data['volume'].values if 'volume' in data.columns else np.ones(n)
        
        # Volatility factor
        returns = np.diff(np.log(data['close'].values))
        volatility = np.concatenate([[np.std(returns[:10])], 
                                   [np.std(returns[max(0, i-9):i+1]) for i in range(1, len(returns))]])
        
        return pd.DataFrame({
            'market_factor': market_factor,
            'volume_factor': volume_factor,
            'volatility_factor': volatility,
            'close': data['close'].values  # For compatibility
        }, index=data.index)
    
    def _prepare_independent_variables(self, reference_data: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Prepare independent variables matrix"""
        # Extract features
        features = []
        
        if 'market_factor' in reference_data.columns:
            features.append(reference_data['market_factor'].values)
        
        if 'volume_factor' in reference_data.columns:
            volume = reference_data['volume_factor'].values
            # Log transform volume to reduce skewness
            log_volume = np.log(volume + 1)
            features.append(log_volume)
        
        if 'volatility_factor' in reference_data.columns:
            features.append(reference_data['volatility_factor'].values)
        
        # Add lagged dependent variable
        if len(y) > 1:
            lagged_y = np.concatenate([[y[0]], y[:-1]])
            features.append(lagged_y)
        
        # Stack features and add constant term
        if features:
            X = np.column_stack(features)
            # Add constant term
            X = np.column_stack([np.ones(len(X)), X])
        else:
            # Fallback: just constant and time trend
            X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
        
        return X
    
    def _calculate_rsquared_analysis(self, y: np.ndarray, X: np.ndarray) -> RSquaredResult:
        """Perform comprehensive R-squared analysis"""
        n, k = X.shape
        
        # Ordinary R-squared
        r_squared = self._calculate_ordinary_r_squared(y, X)
        
        # Adjusted R-squared
        adjusted_r_squared = self._calculate_adjusted_r_squared(r_squared, n, k)
        
        # Predicted R-squared (cross-validation)
        predicted_r_squared = self._calculate_predicted_r_squared(y, X)
        
        # Time-varying R-squared
        time_varying_r2 = self._calculate_time_varying_r_squared(y, X)
        
        # Cross-validation R-squared
        cv_r_squared = self._calculate_cv_r_squared(y, X)
        
        # F-statistic and p-value
        f_stat, f_p_val = self._calculate_f_test(y, X, r_squared)
        
        # Residual diagnostics
        residual_diag = self._calculate_residual_diagnostics(y, X)
        
        # Model selection criteria
        model_criteria = self._calculate_model_selection_criteria(y, X, r_squared)
        
        # Goodness-of-fit tests
        gof_tests = self._calculate_goodness_of_fit_tests(y, X)
        
        # Regression stability
        stability = self._calculate_regression_stability(y, X)
        
        # Outlier influence
        influence = self._calculate_outlier_influence(y, X)
        
        # Confidence interval for R-squared
        conf_interval = self._calculate_r_squared_confidence_interval(r_squared, n, k)
        
        return RSquaredResult(
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            predicted_r_squared=predicted_r_squared,
            time_varying_r_squared=time_varying_r2,
            cross_validation_r_squared=cv_r_squared,
            f_statistic=f_stat,
            f_p_value=f_p_val,
            residual_diagnostics=residual_diag,
            model_selection_criteria=model_criteria,
            goodness_of_fit_tests=gof_tests,
            regression_stability=stability,
            outlier_influence=influence,
            confidence_interval=conf_interval
        )
    
    def _calculate_ordinary_r_squared(self, y: np.ndarray, X: np.ndarray) -> float:
        """Calculate ordinary R-squared"""
        try:
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            
            # R-squared calculation
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return max(0.0, min(r_squared, 1.0))
            
        except np.linalg.LinAlgError:
            return 0.0
    
    def _calculate_adjusted_r_squared(self, r_squared: float, n: int, k: int) -> float:
        """Calculate adjusted R-squared"""
        if n <= k or r_squared <= 0:
            return 0.0
        
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
        return max(0.0, min(adj_r_squared, 1.0))
    
    def _calculate_predicted_r_squared(self, y: np.ndarray, X: np.ndarray) -> float:
        """Calculate predicted R-squared using PRESS statistic"""
        n = len(y)
        press_residuals = []
        
        for i in range(n):
            # Leave-one-out cross-validation
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[i:i+1]
            y_test = y[i]
            
            try:
                beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                y_pred = X_test @ beta
                press_residuals.append((y_test - y_pred[0]) ** 2)
            except:
                press_residuals.append(0.0)
        
        press = np.sum(press_residuals)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        predicted_r_squared = 1 - (press / ss_tot) if ss_tot > 0 else 0.0
        return max(0.0, min(predicted_r_squared, 1.0))
    
    def _calculate_time_varying_r_squared(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Calculate time-varying R-squared using rolling windows"""
        n = len(y)
        rolling_r2 = np.zeros(n)
        
        for i in range(self.window, n):
            start_idx = i - self.window
            end_idx = i
            
            y_window = y[start_idx:end_idx]
            X_window = X[start_idx:end_idx]
            
            rolling_r2[i] = self._calculate_ordinary_r_squared(y_window, X_window)
        
        # Fill initial values
        if self.window < n:
            rolling_r2[:self.window] = rolling_r2[self.window]
        
        return rolling_r2
    
    def _calculate_cv_r_squared(self, y: np.ndarray, X: np.ndarray) -> float:
        """Calculate cross-validation R-squared"""
        if len(y) < self.cv_folds * 2:
            return self._calculate_ordinary_r_squared(y, X)
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                y_pred = X_test @ beta
                score = r2_score(y_test, y_pred)
                cv_scores.append(max(0.0, score))
            except:
                cv_scores.append(0.0)
        
        return np.mean(cv_scores) if cv_scores else 0.0
    
    def _calculate_f_test(self, y: np.ndarray, X: np.ndarray, r_squared: float) -> Tuple[float, float]:
        """Calculate F-statistic and p-value"""
        n, k = X.shape
        
        if n <= k or r_squared <= 0 or r_squared >= 1:
            return 0.0, 1.0
        
        # F-statistic for overall regression significance
        f_stat = (r_squared / (k - 1)) / ((1 - r_squared) / (n - k))
        
        # P-value
        p_value = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        
        return f_stat, p_value
    
    def _calculate_residual_diagnostics(self, y: np.ndarray, X: np.ndarray) -> Dict[str, float]:
        """Calculate residual diagnostics"""
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            
            # Durbin-Watson statistic
            dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2) if np.sum(residuals ** 2) > 0 else 2.0
            
            # Jarque-Bera test for normality
            jb_stat, jb_p = stats.jarque_bera(residuals)
            
            # Breusch-Pagan test for heteroscedasticity (simplified)
            bp_stat = self._breusch_pagan_test(residuals, X)
            
            return {
                'durbin_watson': dw_stat,
                'jarque_bera_statistic': jb_stat,
                'jarque_bera_p_value': jb_p,
                'breusch_pagan': bp_stat,
                'residual_skewness': stats.skew(residuals),
                'residual_kurtosis': stats.kurtosis(residuals)
            }
        except:
            return {
                'durbin_watson': 2.0,
                'jarque_bera_statistic': 0.0,
                'jarque_bera_p_value': 1.0,
                'breusch_pagan': 0.0,
                'residual_skewness': 0.0,
                'residual_kurtosis': 0.0
            }
    
    def _breusch_pagan_test(self, residuals: np.ndarray, X: np.ndarray) -> float:
        """Simplified Breusch-Pagan test for heteroscedasticity"""
        try:
            # Regress squared residuals on X
            squared_residuals = residuals ** 2
            beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
            pred_squared = X @ beta
            
            # LM statistic
            n = len(residuals)
            explained_var = np.var(pred_squared)
            total_var = np.var(squared_residuals)
            
            lm_stat = n * (explained_var / total_var) if total_var > 0 else 0
            return lm_stat
        except:
            return 0.0
    
    def _calculate_model_selection_criteria(self, y: np.ndarray, X: np.ndarray, r_squared: float) -> Dict[str, float]:
        """Calculate model selection criteria"""
        n, k = X.shape
        
        # Log-likelihood (assuming normal errors)
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            sigma_squared = np.mean(residuals ** 2)
            
            log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma_squared) + 1)
            
            # AIC and BIC
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
        except:
            aic = np.inf
            bic = np.inf
        
        return {
            'aic': aic,
            'bic': bic,
            'adjusted_r_squared': self._calculate_adjusted_r_squared(r_squared, n, k)
        }
    
    def _calculate_goodness_of_fit_tests(self, y: np.ndarray, X: np.ndarray) -> Dict[str, float]:
        """Calculate goodness-of-fit tests"""
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            
            # Anderson-Darling test for normality of residuals
            ad_stat, ad_crit, ad_sig = stats.anderson(residuals, dist='norm')
            
            # Shapiro-Wilk test (if sample size allows)
            if len(residuals) <= 5000:
                sw_stat, sw_p = stats.shapiro(residuals)
            else:
                sw_stat, sw_p = 1.0, 1.0
            
            return {
                'anderson_darling': ad_stat,
                'shapiro_wilk_statistic': sw_stat,
                'shapiro_wilk_p_value': sw_p
            }
        except:
            return {
                'anderson_darling': 0.0,
                'shapiro_wilk_statistic': 1.0,
                'shapiro_wilk_p_value': 1.0
            }
    
    def _calculate_regression_stability(self, y: np.ndarray, X: np.ndarray) -> float:
        """Calculate regression stability using recursive residuals"""
        n = len(y)
        if n < 20:
            return 0.5
        
        # Split sample and compare R-squared
        mid_point = n // 2
        
        r2_first = self._calculate_ordinary_r_squared(y[:mid_point], X[:mid_point])
        r2_second = self._calculate_ordinary_r_squared(y[mid_point:], X[mid_point:])
        
        # Stability as inverse of difference
        stability = 1 / (1 + abs(r2_first - r2_second))
        return stability
    
    def _calculate_outlier_influence(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Calculate outlier influence using leverage"""
        try:
            # Hat matrix diagonal (leverage values)
            H = X @ np.linalg.inv(X.T @ X) @ X.T
            leverage = np.diag(H)
            
            # Standardized leverage
            k = X.shape[1]
            n = len(y)
            threshold = 2 * k / n
            
            influence = leverage / threshold
            return np.clip(influence, 0, 5)  # Cap at 5 for numerical stability
            
        except:
            return np.ones(len(y))
    
    def _calculate_r_squared_confidence_interval(self, r_squared: float, n: int, k: int) -> Tuple[float, float]:
        """Calculate confidence interval for R-squared"""
        if n <= k or r_squared <= 0:
            return (0.0, 0.1)
        
        # Fisher transformation
        z = 0.5 * np.log((1 + np.sqrt(r_squared)) / (1 - np.sqrt(r_squared)))
        se = 1 / np.sqrt(n - k - 3)
        
        alpha = 1 - self.confidence_level
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        # Transform back
        r_lower = (np.tanh(z_lower)) ** 2
        r_upper = (np.tanh(z_upper)) ** 2
        
        return (max(0.0, r_lower), min(1.0, r_upper))
    
    def _generate_signal(self, result: RSquaredResult, y: np.ndarray, X: np.ndarray) -> SignalType:
        """Generate trading signal based on R-squared analysis"""
        current_r2 = result.r_squared
        stability = result.regression_stability
        f_p_value = result.f_p_value
        
        # High quality model with good fit
        if current_r2 > 0.7 and stability > 0.8 and f_p_value < 0.05:
            # Use model prediction for signal
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                current_pred = X[-1] @ beta
                current_actual = y[-1]
                
                prediction_error = (current_pred - current_actual) / current_actual
                
                if prediction_error > 0.02:  # Model predicts higher price
                    return SignalType.BUY
                elif prediction_error < -0.02:  # Model predicts lower price
                    return SignalType.SELL
            except:
                pass
        
        # Time-varying R-squared trend
        if len(result.time_varying_r_squared) >= 10:
            recent_r2_trend = np.mean(result.time_varying_r_squared[-5:]) - np.mean(result.time_varying_r_squared[-10:-5])
            
            # Improving model fit suggests trend continuation
            if recent_r2_trend > 0.1 and current_r2 > 0.5:
                # Check if recent price trend is positive
                recent_returns = np.diff(y[-10:])
                if np.mean(recent_returns) > 0:
                    return SignalType.BUY
                else:
                    return SignalType.SELL
        
        # Poor model fit suggests market inefficiency
        if current_r2 < 0.3 and f_p_value > 0.1:
            return SignalType.HOLD  # Avoid trading in unpredictable markets
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: RSquaredResult) -> float:
        """Calculate signal strength based on model quality"""
        # R-squared strength
        r2_strength = result.r_squared
        
        # Statistical significance strength
        significance_strength = max(0, 1 - result.f_p_value)
        
        # Stability strength
        stability_strength = result.regression_stability
        
        return (r2_strength + significance_strength + stability_strength) / 3
    
    def _calculate_confidence(self, result: RSquaredResult) -> float:
        """Calculate confidence based on model validation"""
        # Cross-validation confidence
        cv_conf = result.cross_validation_r_squared
        
        # Confidence interval width (narrower = higher confidence)
        interval_width = result.confidence_interval[1] - result.confidence_interval[0]
        interval_conf = max(0, 1 - interval_width)
        
        # Statistical significance confidence
        significance_conf = max(0, 1 - result.f_p_value)
        
        return (cv_conf + interval_conf + significance_conf) / 3