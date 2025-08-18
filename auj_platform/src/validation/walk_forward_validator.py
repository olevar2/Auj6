"""
Walk-Forward Validation Engine for AUJ Platform.

This module implements the cornerstone validation system that prevents overfitting
by rigorously testing strategies and indicators on out-of-sample data using
walk-forward analysis methodology.

The engine ensures that any strategy or indicator approved for live trading has
demonstrated consistent performance on unseen data, not just historical fitting.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..core.data_contracts import (
    ValidationResult, TradeSignal, GradedDeal, OHLCVData,
    IndicatorResult, MarketRegime
)
from ..core.exceptions import (
    ValidationError, InsufficientDataError, 
    OptimizationError, AUJPlatformError
)


class ValidationPeriodType(str, Enum):
    """Types of validation periods."""
    IN_SAMPLE = "IN_SAMPLE"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"


@dataclass
class ValidationWindow:
    """Represents a validation window with training and testing periods."""
    start_date: datetime
    training_end: datetime
    testing_end: datetime
    period_type: ValidationPeriodType
    

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for validation."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: Decimal
    avg_pnl_per_trade: Decimal
    max_drawdown: Decimal
    max_profit: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    recovery_factor: float
    risk_adjusted_return: float
    volatility: float
    stability_score: float


class WalkForwardValidator:
    """
    Advanced Walk-Forward Validation Engine.
    
    This is the cornerstone component that ensures all strategies and indicators
    are validated on out-of-sample data before being approved for live trading.
    
    Key Features:
    - Prevents overfitting through rigorous out-of-sample testing
    - Multiple validation windows with different time horizons
    - Comprehensive performance metrics including risk-adjusted returns
    - Robustness scoring based on consistency across time periods
    - Anti-fragility measures to identify strategies that improve under stress
    """
    
    def __init__(
        self,
        training_window_days: int = 90,
        testing_window_days: int = 30,
        step_days: int = 7,
        min_trades_per_window: int = 10,
        overfitting_threshold: float = 0.3,
        robustness_threshold: float = 0.7
    ):
        """
        Initialize the Walk-Forward Validator.
        
        Args:
            training_window_days: Size of training window in days
            testing_window_days: Size of testing window in days  
            step_days: Step size for rolling window in days
            min_trades_per_window: Minimum trades required per window
            overfitting_threshold: Threshold for detecting overfitting (0-1)
            robustness_threshold: Threshold for robustness score (0-1)
        """
        self.training_window_days = training_window_days
        self.testing_window_days = testing_window_days
        self.step_days = step_days
        self.min_trades_per_window = min_trades_per_window
        self.overfitting_threshold = overfitting_threshold
        self.robustness_threshold = robustness_threshold
        
        self.logger = logging.getLogger(__name__)
        
    def validate_strategy(
        self,
        strategy_name: str,
        historical_deals: List[GradedDeal],
        indicator_set: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ValidationResult:
        """
        Perform comprehensive walk-forward validation on a strategy.
        
        Args:
            strategy_name: Name of the strategy being validated
            historical_deals: List of historical deals/trades
            indicator_set: List of indicators used by the strategy
            start_date: Start date for validation (optional)
            end_date: End date for validation (optional)
            
        Returns:
            ValidationResult: Comprehensive validation results
            
        Raises:
            ValidationError: If validation cannot be performed
            InsufficientDataError: If insufficient data available
        """
        try:
            self.logger.info(f"Starting walk-forward validation for strategy: {strategy_name}")
            
            # Validate input data
            if not historical_deals:
                raise InsufficientDataError("No historical deals provided for validation")
                
            if len(historical_deals) < self.min_trades_per_window:
                raise InsufficientDataError(
                    f"Insufficient trades for validation. Need at least {self.min_trades_per_window}, got {len(historical_deals)}"
                )
            
            # Sort deals by entry time
            sorted_deals = sorted(historical_deals, key=lambda d: d.entry_time)
            
            # Determine validation period
            if not start_date:
                start_date = sorted_deals[0].entry_time
            if not end_date:
                end_date = sorted_deals[-1].entry_time
                
            # Generate validation windows
            windows = self._generate_validation_windows(start_date, end_date)
            
            if not windows:
                raise ValidationError("Could not generate sufficient validation windows")
            
            # Perform validation across all windows
            in_sample_results = []
            out_of_sample_results = []
            
            for window in windows:
                # Split deals into training and testing sets
                training_deals = [
                    d for d in sorted_deals 
                    if window.start_date <= d.entry_time <= window.training_end
                ]
                
                testing_deals = [
                    d for d in sorted_deals 
                    if window.training_end < d.entry_time <= window.testing_end
                ]
                
                # Skip windows with insufficient data
                if len(training_deals) < self.min_trades_per_window or len(testing_deals) < 5:
                    continue
                
                # Calculate performance metrics
                training_metrics = self._calculate_performance_metrics(training_deals)
                testing_metrics = self._calculate_performance_metrics(testing_deals)
                
                in_sample_results.append(training_metrics)
                out_of_sample_results.append(testing_metrics)
            
            if not in_sample_results or not out_of_sample_results:
                raise ValidationError("Insufficient validation windows generated")
            
            # Aggregate results
            in_sample_performance = self._aggregate_performance_metrics(in_sample_results)
            out_of_sample_performance = self._aggregate_performance_metrics(out_of_sample_results)
            
            # Calculate overfitting and robustness scores
            overfitting_score = self._calculate_overfitting_score(
                in_sample_performance, out_of_sample_performance
            )
            
            robustness_score = self._calculate_robustness_score(out_of_sample_results)
            
            # Determine if strategy is recommended for live trading
            recommended_for_live = (
                overfitting_score <= self.overfitting_threshold and
                robustness_score >= self.robustness_threshold and
                out_of_sample_performance['win_rate'] > 0.5 and
                out_of_sample_performance['sharpe_ratio'] > 1.0
            )
            
            # Create validation result
            validation_result = ValidationResult(
                strategy_name=strategy_name,
                indicator_set=indicator_set,
                in_sample_period={
                    'start': start_date,
                    'end': end_date - timedelta(days=self.testing_window_days)
                },
                out_of_sample_period={
                    'start': start_date + timedelta(days=self.training_window_days),
                    'end': end_date
                },
                in_sample_performance=in_sample_performance,
                out_of_sample_performance=out_of_sample_performance,
                overfitting_score=overfitting_score,
                robustness_score=robustness_score,
                recommended_for_live=recommended_for_live
            )
            
            self.logger.info(
                f"Validation completed for {strategy_name}. "
                f"Overfitting: {overfitting_score:.3f}, "
                f"Robustness: {robustness_score:.3f}, "
                f"Recommended: {recommended_for_live}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating strategy {strategy_name}: {str(e)}")
            raise ValidationError(f"Strategy validation failed: {str(e)}") from e
    
    def validate_indicator_effectiveness(
        self,
        indicator_name: str,
        indicator_results: List[IndicatorResult],
        corresponding_deals: List[GradedDeal]
    ) -> Dict[str, float]:
        """
        Validate the effectiveness of a single indicator using walk-forward analysis.
        
        Args:
            indicator_name: Name of the indicator
            indicator_results: Historical indicator calculation results
            corresponding_deals: Deals that used this indicator
            
        Returns:
            Dict containing effectiveness metrics
        """
        try:
            if not indicator_results or not corresponding_deals:
                raise InsufficientDataError("Insufficient data for indicator validation")
            
            # Group deals by validation periods
            windows = self._generate_validation_windows(
                indicator_results[0].timestamp,
                indicator_results[-1].timestamp
            )
            
            effectiveness_scores = []
            
            for window in windows:
                # Get indicator results for this window
                window_indicators = [
                    ir for ir in indicator_results
                    if window.start_date <= ir.timestamp <= window.testing_end
                ]
                
                # Get corresponding deals
                window_deals = [
                    d for d in corresponding_deals
                    if (window.start_date <= d.entry_time <= window.testing_end and
                        indicator_name in d.original_signal.indicators_used)
                ]
                
                if len(window_deals) >= 5:  # Minimum for statistical significance
                    effectiveness = self._calculate_indicator_effectiveness(
                        window_indicators, window_deals
                    )
                    effectiveness_scores.append(effectiveness)
            
            if not effectiveness_scores:
                return {'effectiveness_score': 0.0, 'consistency_score': 0.0}
            
            # Calculate overall metrics
            avg_effectiveness = np.mean(effectiveness_scores)
            consistency = 1.0 - np.std(effectiveness_scores) / (avg_effectiveness + 1e-8)
            
            return {
                'effectiveness_score': float(avg_effectiveness),
                'consistency_score': float(max(0.0, consistency)),
                'num_validation_windows': len(effectiveness_scores),
                'min_effectiveness': float(min(effectiveness_scores)),
                'max_effectiveness': float(max(effectiveness_scores))
            }
            
        except Exception as e:
            self.logger.error(f"Error validating indicator {indicator_name}: {str(e)}")
            return {'effectiveness_score': 0.0, 'consistency_score': 0.0}    
    def _generate_validation_windows(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[ValidationWindow]:
        """
        Generate overlapping validation windows for walk-forward analysis.
        
        Args:
            start_date: Start date for validation period
            end_date: End date for validation period
            
        Returns:
            List of ValidationWindow objects
        """
        windows = []
        current_start = start_date
        
        while True:
            training_end = current_start + timedelta(days=self.training_window_days)
            testing_end = training_end + timedelta(days=self.testing_window_days)
            
            # Check if we have enough data for this window
            if testing_end > end_date:
                break
                
            window = ValidationWindow(
                start_date=current_start,
                training_end=training_end,
                testing_end=testing_end,
                period_type=ValidationPeriodType.OUT_OF_SAMPLE
            )
            
            windows.append(window)
            current_start += timedelta(days=self.step_days)
        
        return windows
    
    def _calculate_performance_metrics(self, deals: List[GradedDeal]) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for a set of deals.
        
        Args:
            deals: List of graded deals
            
        Returns:
            PerformanceMetrics object with calculated metrics
        """
        if not deals:
            return PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl=Decimal('0'), avg_pnl_per_trade=Decimal('0'),
                max_drawdown=Decimal('0'), max_profit=Decimal('0'),
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                profit_factor=0.0, recovery_factor=0.0, risk_adjusted_return=0.0,
                volatility=0.0, stability_score=0.0
            )
        
        # Basic metrics
        total_trades = len(deals)
        winning_trades = sum(1 for d in deals if d.pnl and d.pnl > 0)
        losing_trades = sum(1 for d in deals if d.pnl and d.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # PnL metrics
        pnls = [d.pnl for d in deals if d.pnl is not None]
        total_pnl = sum(pnls) if pnls else Decimal('0')
        avg_pnl_per_trade = total_pnl / len(pnls) if pnls else Decimal('0')
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(deals)
        max_profit = max(pnls) if pnls else Decimal('0')
        
        # Convert to float for ratio calculations
        pnl_values = [float(pnl) for pnl in pnls]
        
        # Advanced ratios
        sharpe_ratio = self._calculate_sharpe_ratio(pnl_values)
        sortino_ratio = self._calculate_sortino_ratio(pnl_values)
        calmar_ratio = self._calculate_calmar_ratio(pnl_values, float(max_drawdown))
        profit_factor = self._calculate_profit_factor(pnl_values)
        recovery_factor = self._calculate_recovery_factor(pnl_values, float(max_drawdown))
        
        # Risk-adjusted return
        volatility = np.std(pnl_values) if len(pnl_values) > 1 else 0.0
        risk_adjusted_return = float(avg_pnl_per_trade) / (volatility + 1e-8)
        
        # Stability score (consistency of returns)
        stability_score = self._calculate_stability_score(pnl_values)
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl_per_trade,
            max_drawdown=max_drawdown,
            max_profit=max_profit,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            risk_adjusted_return=risk_adjusted_return,
            volatility=volatility,
            stability_score=stability_score
        )
    
    def _calculate_max_drawdown(self, deals: List[GradedDeal]) -> Decimal:
        """Calculate maximum drawdown from a series of deals."""
        if not deals:
            return Decimal('0')
        
        # Sort deals by time and calculate cumulative PnL
        sorted_deals = sorted(deals, key=lambda d: d.entry_time)
        cumulative_pnl = Decimal('0')
        peak = Decimal('0')
        max_drawdown = Decimal('0')
        
        for deal in sorted_deals:
            if deal.pnl is not None:
                cumulative_pnl += deal.pnl
                peak = max(peak, cumulative_pnl)
                drawdown = peak - cumulative_pnl
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        excess_return = avg_return - (risk_free_rate / 252)
        sharpe = (excess_return * np.sqrt(252)) / std_return
        
        return float(sharpe)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)."""
        if not returns or len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf') if avg_return > 0 else 0.0
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        excess_return = avg_return - (risk_free_rate / 252)
        sortino = (excess_return * np.sqrt(252)) / downside_std
        
        return float(sortino)
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if not returns or max_drawdown <= 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252  # Annualized
        return annual_return / max_drawdown
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not returns:
            return 0.0
        
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_recovery_factor(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate recovery factor (total return / max drawdown)."""
        if not returns or max_drawdown <= 0:
            return 0.0
        
        total_return = sum(returns)
        return total_return / max_drawdown
    
    def _calculate_stability_score(self, returns: List[float]) -> float:
        """
        Calculate stability score based on consistency of returns.
        Higher score means more consistent performance.
        """
        if not returns or len(returns) < 3:
            return 0.0
        
        # Use coefficient of variation (inverse relationship with stability)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0 or std_return == 0:
            return 0.5  # Neutral score
        
        cv = abs(std_return / mean_return)
        
        # Convert to stability score (0-1, higher is better)
        stability = 1.0 / (1.0 + cv)
        
        return float(stability)    
    def _aggregate_performance_metrics(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, float]:
        """
        Aggregate performance metrics across multiple validation windows.
        
        Args:
            metrics_list: List of PerformanceMetrics objects
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not metrics_list:
            return {}
        
        # Calculate weighted averages based on number of trades
        total_trades = sum(m.total_trades for m in metrics_list)
        
        if total_trades == 0:
            return {
                'win_rate': 0.0,
                'avg_pnl_per_trade': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'stability_score': 0.0,
                'total_trades': 0
            }
        
        # Weighted averages
        win_rate = sum(m.win_rate * m.total_trades for m in metrics_list) / total_trades
        avg_pnl = sum(float(m.avg_pnl_per_trade) * m.total_trades for m in metrics_list) / total_trades
        
        # For ratios, use simple averages (they're already normalized)
        sharpe_ratio = np.mean([m.sharpe_ratio for m in metrics_list])
        sortino_ratio = np.mean([m.sortino_ratio for m in metrics_list])
        profit_factor = np.mean([m.profit_factor for m in metrics_list])
        stability_score = np.mean([m.stability_score for m in metrics_list])
        
        # Maximum drawdown is the worst across all periods
        max_drawdown = max(float(m.max_drawdown) for m in metrics_list)
        
        return {
            'win_rate': win_rate,
            'avg_pnl_per_trade': avg_pnl,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'stability_score': stability_score,
            'total_trades': total_trades
        }
    
    def _calculate_overfitting_score(
        self, 
        in_sample_performance: Dict[str, float], 
        out_of_sample_performance: Dict[str, float]
    ) -> float:
        """
        Calculate overfitting score by comparing in-sample vs out-of-sample performance.
        
        Score ranges from 0 (no overfitting) to 1 (severe overfitting).
        
        Args:
            in_sample_performance: Performance metrics on training data
            out_of_sample_performance: Performance metrics on testing data
            
        Returns:
            Overfitting score (0-1, lower is better)
        """
        try:
            # Key metrics to compare
            key_metrics = ['win_rate', 'sharpe_ratio', 'profit_factor', 'stability_score']
            
            overfitting_signals = []
            
            for metric in key_metrics:
                if metric in in_sample_performance and metric in out_of_sample_performance:
                    in_sample_value = in_sample_performance[metric]
                    out_of_sample_value = out_of_sample_performance[metric]
                    
                    # Skip if values are zero or invalid
                    if in_sample_value <= 0 or out_of_sample_value <= 0:
                        continue
                    
                    # Calculate degradation ratio
                    degradation = (in_sample_value - out_of_sample_value) / in_sample_value
                    
                    # Clip to reasonable bounds
                    degradation = max(0.0, min(1.0, degradation))
                    overfitting_signals.append(degradation)
            
            if not overfitting_signals:
                return 0.5  # Neutral score if no valid comparisons
            
            # Average degradation across metrics
            avg_degradation = np.mean(overfitting_signals)
            
            # Apply penalty for high variance in degradation (inconsistent performance)
            variance_penalty = np.std(overfitting_signals) * 0.5
            
            overfitting_score = avg_degradation + variance_penalty
            
            return float(min(1.0, max(0.0, overfitting_score)))
            
        except Exception as e:
            self.logger.warning(f"Error calculating overfitting score: {str(e)}")
            return 0.5
    
    def _calculate_robustness_score(self, out_of_sample_results: List[PerformanceMetrics]) -> float:
        """
        Calculate robustness score based on consistency of out-of-sample performance.
        
        Score ranges from 0 (not robust) to 1 (highly robust).
        
        Args:
            out_of_sample_results: List of out-of-sample performance metrics
            
        Returns:
            Robustness score (0-1, higher is better)
        """
        try:
            if not out_of_sample_results or len(out_of_sample_results) < 2:
                return 0.0
            
            # Extract key performance metrics
            win_rates = [m.win_rate for m in out_of_sample_results]
            sharpe_ratios = [m.sharpe_ratio for m in out_of_sample_results]
            profit_factors = [m.profit_factor for m in out_of_sample_results]
            
            robustness_scores = []
            
            # Consistency in win rate
            if win_rates and any(wr > 0 for wr in win_rates):
                win_rate_consistency = 1.0 - (np.std(win_rates) / (np.mean(win_rates) + 1e-8))
                robustness_scores.append(max(0.0, win_rate_consistency))
            
            # Consistency in Sharpe ratio
            if sharpe_ratios and any(sr > 0 for sr in sharpe_ratios):
                sharpe_consistency = 1.0 - (np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-8))
                robustness_scores.append(max(0.0, sharpe_consistency))
            
            # Consistency in profit factor
            if profit_factors and any(pf > 0 for pf in profit_factors):
                pf_consistency = 1.0 - (np.std(profit_factors) / (np.mean(profit_factors) + 1e-8))
                robustness_scores.append(max(0.0, pf_consistency))
            
            # Percentage of profitable periods
            profitable_periods = sum(1 for m in out_of_sample_results if m.win_rate > 0.5)
            profitability_ratio = profitable_periods / len(out_of_sample_results)
            robustness_scores.append(profitability_ratio)
            
            if not robustness_scores:
                return 0.0
            
            # Weighted average (profitability gets higher weight)
            weights = [1.0, 1.0, 1.0, 2.0][:len(robustness_scores)]
            weighted_score = np.average(robustness_scores, weights=weights)
            
            return float(min(1.0, max(0.0, weighted_score)))
            
        except Exception as e:
            self.logger.warning(f"Error calculating robustness score: {str(e)}")
            return 0.0
    
    def _calculate_indicator_effectiveness(
        self,
        indicator_results: List[IndicatorResult],
        corresponding_deals: List[GradedDeal]
    ) -> float:
        """
        Calculate the effectiveness of an indicator based on its correlation with successful trades.
        
        Args:
            indicator_results: List of indicator calculation results
            corresponding_deals: Deals that used this indicator
            
        Returns:
            Effectiveness score (0-1, higher is better)
        """
        try:
            if not indicator_results or not corresponding_deals:
                return 0.0
            
            # Create time-aligned data
            indicator_signals = []
            deal_outcomes = []
            
            for deal in corresponding_deals:
                # Find indicator result closest to deal entry time
                closest_indicator = min(
                    indicator_results,
                    key=lambda ir: abs((ir.timestamp - deal.entry_time).total_seconds())
                )
                
                # Convert indicator signal to numeric value
                signal_value = self._convert_signal_to_numeric(closest_indicator)
                
                # Deal outcome (1 for profitable, 0 for loss)
                outcome = 1.0 if deal.pnl and deal.pnl > 0 else 0.0
                
                indicator_signals.append(signal_value)
                deal_outcomes.append(outcome)
            
            if len(indicator_signals) < 3:  # Need minimum data for correlation
                return 0.0
            
            # Calculate correlation between indicator signals and outcomes
            correlation = np.corrcoef(indicator_signals, deal_outcomes)[0, 1]
            
            # Handle NaN correlation
            if np.isnan(correlation):
                return 0.0
            
            # Convert correlation to effectiveness score (0-1)
            effectiveness = (abs(correlation) + 1) / 2
            
            return float(min(1.0, max(0.0, effectiveness)))
            
        except Exception as e:
            self.logger.warning(f"Error calculating indicator effectiveness: {str(e)}")
            return 0.0
    
    def _convert_signal_to_numeric(self, indicator_result: IndicatorResult) -> float:
        """
        Convert indicator signal to numeric value for correlation analysis.
        
        Args:
            indicator_result: Indicator result to convert
            
        Returns:
            Numeric representation of the signal
        """
        try:
            # If signal is already numeric
            if isinstance(indicator_result.value, (int, float)):
                return float(indicator_result.value)
            
            # If signal is a dictionary, use the primary value
            if isinstance(indicator_result.value, dict):
                # Common keys for primary values
                primary_keys = ['value', 'signal', 'main', 'close', 'price']
                for key in primary_keys:
                    if key in indicator_result.value:
                        return float(indicator_result.value[key])
                # If no primary key found, use first numeric value
                for value in indicator_result.value.values():
                    if isinstance(value, (int, float)):
                        return float(value)
            
            # If signal is a list, use the last value
            if isinstance(indicator_result.value, list) and indicator_result.value:
                return float(indicator_result.value[-1])
            
            # If there's a strength value, use it
            if indicator_result.strength is not None:
                return indicator_result.strength
            
            # Convert text signal to numeric
            if indicator_result.signal:
                signal_map = {
                    'BUY': 1.0,
                    'STRONG_BUY': 1.5,
                    'SELL': -1.0,
                    'STRONG_SELL': -1.5,
                    'HOLD': 0.0,
                    'NEUTRAL': 0.0
                }
                return signal_map.get(indicator_result.signal.upper(), 0.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def get_validation_summary(
        self, 
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Generate a summary of multiple validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with summary statistics
        """
        if not validation_results:
            return {}
        
        # Count strategies by recommendation status
        recommended_count = sum(1 for vr in validation_results if vr.recommended_for_live)
        
        # Average scores
        avg_overfitting = np.mean([vr.overfitting_score for vr in validation_results])
        avg_robustness = np.mean([vr.robustness_score for vr in validation_results])
        
        # Best performing strategy
        best_strategy = max(
            validation_results,
            key=lambda vr: vr.out_of_sample_performance.get('sharpe_ratio', 0)
        )
        
        return {
            'total_strategies_tested': len(validation_results),
            'strategies_recommended': recommended_count,
            'recommendation_rate': recommended_count / len(validation_results),
            'average_overfitting_score': float(avg_overfitting),
            'average_robustness_score': float(avg_robustness),
            'best_strategy_name': best_strategy.strategy_name,
            'best_strategy_sharpe': best_strategy.out_of_sample_performance.get('sharpe_ratio', 0),
            'validation_date': datetime.utcnow().isoformat()
        }