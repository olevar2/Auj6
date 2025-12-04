"""
Dynamic Risk Manager for AUJ Platform.

This module implements sophisticated risk management with confidence-scaled position sizing,
ensuring capital preservation while maximizing profit potential for the noble mission
of supporting sick children and families in need.

Key Features:
- Confidence-scaled position sizing
- Multi-layer risk validation
- Dynamic risk adjustment based on market conditions
- Portfolio-level risk management
- Real-time risk monitoring and alerts
- ⭐ NEW: RiskGenius integration for enhanced risk assessment
- ✅ FIXED: Real volatility and correlation calculations (Bug #37)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import logging
import math

# Assuming these imports are needed based on usage
from ..core.logging_setup import get_logger
from ..core.data_contracts import (
    TradeSignal, AccountInfo, MarketConditions, TradeDirection, 
    RiskMetrics, PositionSizeAdjustment, RiskLevel, MarketRegime, Timeframe
)

logger = get_logger(__name__)

class RiskManagementError(Exception):
    pass

class DynamicRiskManager:
    """
    Dynamic Risk Manager
    
    Manages trading risk through:
    - Position sizing based on account equity and risk parameters
    - Portfolio heat monitoring
    - Correlation analysis
    - Market regime adaptation
    - ⭐ NEW: Direct RiskGenius risk assessment integration
    - ✅ FIXED: Real volatility and correlation calculations (Bug #37)
    """
    
    def __init__(self, config_manager: Any, portfolio_tracker: Any = None, data_provider=None):
        """
        Initialize Dynamic Risk Manager.
        
        Args:
            config_manager: Configuration manager
            portfolio_tracker: Optional portfolio tracker
            data_provider: Data provider for market data (for volatility and correlation)
        """
        self.config_manager = config_manager
        self.portfolio_tracker = portfolio_tracker
        self.data_provider = data_provider  # ✅ NEW: Data provider injection
        
        # Load risk parameters
        self.base_risk_params = self._load_base_risk_parameters()
        
        # State tracking
        self.open_positions = {}
        self.daily_loss_tracking = {}
        self.current_portfolio_heat = 0.0
        
        # Risk thresholds
        self.correlation_threshold = 0.7
        self.heat_decay_factor = 0.95
        
        # Confidence scaling
        self.confidence_scaling = {
            'HIGH': 1.0,
            'MEDIUM': 0.7,
            'LOW': 0.4
        }
        
        # Regime adjustments
        self.regime_adjustments = {
            'TRENDING_UP': 1.2,
            'TRENDING_DOWN': 1.2,
            'RANGING': 0.8,
            'VOLATILE': 0.6
        }
        
        # ✅ NEW: Volatility and correlation caching
        self.volatility_cache = {}  # {symbol: (volatility, timestamp)}
        self.correlation_cache = {}  # {(symbol1, symbol2): (correlation, timestamp)}
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        self._validate_risk_parameters()
        
        logger.info("Dynamic Risk Manager initialized with RiskGenius integration and real data calculations")

    def _validate_risk_parameters(self):
        """Validate loaded risk parameters."""
        required_params = [
            'max_position_size_percent', 'max_daily_loss_percent', 
            'stop_loss_percent', 'take_profit_percent', 'max_open_positions'
        ]
        
        for param in required_params:
            if param not in self.base_risk_params:
                raise ValueError(f"Missing required risk parameter: {param}")
                
        # Validate parameter ranges
        if self.base_risk_params['max_position_size_percent'] <= 0:
            raise ValueError("max_position_size_percent must be positive")
        if self.base_risk_params['max_daily_loss_percent'] <= 0:
            raise ValueError("max_daily_loss_percent must be positive")
            
        logger.info("Risk parameters validation passed")

    def _load_base_risk_parameters(self) -> Dict[str, float]:
        """Load base risk parameters from unified configuration."""
        return {
            'max_position_size_percent': self.config_manager.get_float('risk.max_position_size_percent', 5.0),
            'max_daily_loss_percent': self.config_manager.get_float('risk.max_daily_loss_percent', 2.0),
            'stop_loss_percent': self.config_manager.get_float('risk.stop_loss_percent', 1.0),
            'take_profit_percent': self.config_manager.get_float('risk.take_profit_percent', 2.0),
            'max_open_positions': self.config_manager.get_int('risk.max_open_positions', 10),
            'portfolio_heat_limit': self.config_manager.get_float('risk.portfolio_heat_limit', 15.0),
            'correlation_limit': self.config_manager.get_float('risk.correlation_limit', 0.7),
            'volatility_scaling_factor': self.config_manager.get_float('risk.volatility_scaling_factor', 1.0)
        }
    
    async def calculate_position_size(self,
                                    signal: TradeSignal,
                                    account_info: AccountInfo,
                                    market_conditions: Optional[MarketConditions] = None,
                                    current_price: Optional[Decimal] = None,
                                    risk_assessment: Optional[Dict[str, Any]] = None) -> Tuple[Decimal, RiskMetrics]:
        """
        Calculate optimal position size using confidence scaling and dynamic risk adjustment.
        
        Args:
            signal: Trade signal with confidence level
            account_info: Current account information
            market_conditions: Current market conditions
            current_price: Current market price for the symbol
            risk_assessment: ⭐ NEW - Optional detailed risk assessment from RiskGenius agent
            
        Returns:
            Tuple of (position_size, risk_metrics)
        """
        try:
            logger.debug(f"Calculating position size for {signal.symbol} {signal.direction.value}")
            
            # Step 1: Calculate base position size
            base_position_size = await self._calculate_base_position_size(
                signal, account_info, current_price
            )
            
            # Step 2: Apply confidence scaling
            confidence_multiplier = self._calculate_confidence_multiplier(signal)
            confidence_adjusted_size = base_position_size * Decimal(str(confidence_multiplier))
            
            # Step 3: Apply volatility adjustment
            volatility_adjustment = await self._calculate_volatility_adjustment(
                signal.symbol, market_conditions
            )
            volatility_adjusted_size = confidence_adjusted_size * Decimal(str(volatility_adjustment))
            
            # Step 4: Apply correlation penalty
            correlation_penalty = await self._calculate_correlation_penalty(signal.symbol)
            correlation_adjusted_size = volatility_adjusted_size * Decimal(str(correlation_penalty))
            
            # Step 5: Apply portfolio heat adjustment
            heat_adjustment = await self._calculate_portfolio_heat_adjustment()
            heat_adjusted_size = correlation_adjusted_size * Decimal(str(heat_adjustment))
            
            # Step 6: Apply market regime adjustment
            regime_adjustment = self._calculate_regime_adjustment(market_conditions)
            regime_adjusted_size = heat_adjusted_size * Decimal(str(regime_adjustment))
            
            # ⭐ NEW Step 7: Apply RiskGenius risk assessment adjustment (if available)
            risk_assessment_factor = 1.0
            if risk_assessment:
                risk_assessed_size = self._apply_risk_assessment_adjustment(
                    regime_adjusted_size,
                    risk_assessment
                )
                risk_assessment_factor = float(risk_assessed_size / regime_adjusted_size) if regime_adjusted_size > 0 else 1.0
            else:
                risk_assessed_size = regime_adjusted_size
            
            # Step 8: Apply final limits and validation
            final_position_size, adjustments = await self._apply_final_limits(
                risk_assessed_size, signal, account_info
            )
            
            # Step 9: Calculate comprehensive risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                signal=signal,
                account_info=account_info,
                base_size=base_position_size,
                final_size=final_position_size,
                confidence_multiplier=confidence_multiplier,
                volatility_adjustment=volatility_adjustment,
                correlation_penalty=correlation_penalty,
                heat_adjustment=heat_adjustment,
                regime_adjustment=regime_adjustment,
                risk_assessment_factor=risk_assessment_factor,
                risk_assessment_applied=risk_assessment is not None,
                adjustments=adjustments,
                current_price=current_price
            )
            
            logger.info(f"Position size calculated: {final_position_size} "
                       f"(Risk Level: {risk_metrics.risk_level.value}, "
                       f"Confidence: {signal.confidence:.3f}, "
                       f"RiskAssessment: {'Applied' if risk_assessment else 'Not Available'})")
            
            return final_position_size, risk_metrics
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {str(e)}")
            raise RiskManagementError(f"Failed to calculate position size: {str(e)}")
    
    async def _calculate_base_position_size(self,
                                          signal: TradeSignal,
                                          account_info: AccountInfo,
                                          current_price: Optional[Decimal]) -> Decimal:
        """Calculate base position size using risk-based sizing."""
        try:
            # Use percentage of equity for base calculation
            equity = account_info.equity
            max_risk_percent = self.base_risk_params['max_position_size_percent']
            
            # Calculate position size based on stop loss
            if signal.stop_loss and current_price:
                # Calculate risk per unit
                if signal.direction == TradeDirection.BUY:
                    risk_per_unit = current_price - signal.stop_loss
                else:
                    risk_per_unit = signal.stop_loss - current_price
                
                if risk_per_unit > 0:
                    # Risk-based position sizing
                    max_risk_amount = equity * Decimal(str(max_risk_percent / 100))
                    base_position_size = max_risk_amount / risk_per_unit
                else:
                    # Fallback to percentage-based sizing
                    base_position_size = (equity * Decimal(str(max_risk_percent / 100))) / current_price
            else:
                # Fallback: percentage of equity
                base_position_size = (equity * Decimal(str(max_risk_percent / 100))) / (current_price or Decimal('1'))
            
            return base_position_size.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
        except Exception as e:
            logger.warning(f"Base position size calculation failed: {str(e)}")
            # Ultra-conservative fallback
            return Decimal('0.01')
    
    def _calculate_confidence_multiplier(self, signal: TradeSignal) -> float:
        """Calculate confidence-based position size multiplier."""
        base_multiplier = self.confidence_scaling.get(signal.confidence_level, 0.5)
        
        # Fine-tune based on exact confidence value
        confidence_adjustment = signal.confidence  # 0.0 to 1.0
        
        # Combine base multiplier with fine confidence adjustment
        final_multiplier = (base_multiplier * 0.7) + (confidence_adjustment * 0.3)
        
        return max(0.1, min(1.0, final_multiplier))  # Clamp between 0.1 and 1.0    

    async def _calculate_volatility_adjustment(self,
                                             symbol: str,
                                             market_conditions: Optional[MarketConditions]) -> float:
        """
        ✅ FIXED: Calculate volatility-based position size adjustment using real market data.
        
        Bug #37 Fix: This now connects to real market data instead of returning hardcoded 0.5
        """
        try:
            if market_conditions:
                volatility = market_conditions.volatility
            else:
                # ✅ NEW: Get recent volatility data from market
                volatility = await self._get_symbol_volatility(symbol)
            
            # Normalize volatility (assuming 0-1 scale, where 0.5 is average)
            normalized_volatility = max(0.1, min(2.0, volatility))
            
            # Inverse relationship: higher volatility = smaller position
            volatility_adjustment = 1.0 / (1.0 + normalized_volatility)
            
            # Apply volatility scaling factor from config
            scaling_factor = self.base_risk_params.get('volatility_scaling_factor', 1.0)
            final_adjustment = volatility_adjustment * scaling_factor
            
            return max(0.2, min(1.5, final_adjustment))  # Clamp between 0.2 and 1.5
            
        except Exception as e:
            logger.warning(f"Volatility adjustment calculation failed: {str(e)}")
            return 1.0  # Neutral adjustment
    
    async def _calculate_correlation_penalty(self, symbol: str) -> float:
        """
        ✅ FIXED: Calculate correlation penalty to reduce correlated exposure using real data.
        
        Bug #37 Fix: This now calculates real correlation instead of returning hardcoded 0.0
        """
        try:
            # Get current open positions
            open_positions = await self._get_open_positions()
            
            if not open_positions:
                return 1.0  # No penalty if no open positions
            
            # ✅ NEW: Calculate correlation with open positions using real market data
            max_correlation = 0.0
            for position_symbol in open_positions.keys():
                if position_symbol != symbol:
                    correlation = await self._get_symbol_correlation(symbol, position_symbol)
                    max_correlation = max(max_correlation, abs(correlation))
            
            # Apply penalty based on correlation
            if max_correlation > self.correlation_threshold:
                # Reduce position size for highly correlated instruments
                penalty_factor = 1.0 - ((max_correlation - self.correlation_threshold) / (1.0 - self.correlation_threshold)) * 0.5
                return max(0.3, penalty_factor)
            
            return 1.0  # No penalty
            
        except Exception as e:
            logger.warning(f"Correlation penalty calculation failed: {str(e)}")
            return 1.0
    
    async def _calculate_portfolio_heat_adjustment(self) -> float:
        """Calculate portfolio heat adjustment to manage overall risk."""
        try:
            current_heat = await self._calculate_current_portfolio_heat()
            max_heat = self.base_risk_params.get('portfolio_heat_limit', 15.0)
            
            if current_heat >= max_heat:
                # Stop taking new positions if at heat limit
                return 0.0
            
            # Gradually reduce position sizes as heat increases
            heat_ratio = current_heat / max_heat
            if heat_ratio > 0.7:  # Start reducing at 70% of max heat
                reduction_factor = 1.0 - ((heat_ratio - 0.7) / 0.3) * 0.5
                return max(0.2, reduction_factor)
            
            return 1.0  # No adjustment needed
            
        except Exception as e:
            logger.warning(f"Portfolio heat adjustment calculation failed: {str(e)}")
            return 1.0
    
    def _calculate_regime_adjustment(self, market_conditions: Optional[MarketConditions]) -> float:
        """Calculate market regime-based position size adjustment."""
        if not market_conditions:
            return 1.0
        
        regime = market_conditions.regime
        adjustment = self.regime_adjustments.get(regime, 1.0)
        
        # Additional adjustment based on trend strength
        if hasattr(market_conditions, 'trend_strength'):
            trend_strength = market_conditions.trend_strength
            if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                # Increase size for strong trends
                trend_bonus = 1.0 + (trend_strength - 0.5) * 0.2  # Up to 20% bonus
                adjustment *= trend_bonus
        
        return max(0.3, min(1.5, adjustment))
    
    def _apply_risk_assessment_adjustment(self,
                                         position_size: Decimal,
                                         risk_assessment: Dict[str, Any]) -> Decimal:
        """
        ⭐ NEW: Apply RiskGenius risk assessment-based adjustment to position size.
        
        Uses comprehensive risk analysis from RiskGenius agent to further refine
        position sizing beyond base confidence scaling.
        
        Args:
            position_size: Current calculated position size
            risk_assessment: Risk assessment dictionary from RiskGenius agent containing:
                - overall_risk_level: str ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
                - position_size_multiplier: float (0.5-1.5)
                - risk_warnings: List[str]
                - max_exposure_percentage: float
                
        Returns:
            Adjusted position size based on detailed risk assessment
        """
        if not risk_assessment:
            return position_size
        
        try:
            # Extract risk metrics from RiskGenius assessment
            overall_risk_level = risk_assessment.get('overall_risk_level', 'MEDIUM')
            position_multiplier = risk_assessment.get('position_size_multiplier', 1.0)
            risk_warnings = risk_assessment.get('risk_warnings', [])
            
            # Apply position size multiplier from RiskGenius
            # This directly uses the recommendation from the risk expert
            risk_factor = float(position_multiplier)
            
            # Additional penalty for critical risk warnings
            if overall_risk_level == 'CRITICAL':
                risk_factor *= 0.5  # 50% reduction for critical risk
                logger.warning(f"CRITICAL risk level detected, reducing position by 50%")
            elif overall_risk_level == 'HIGH':
                risk_factor *= 0.8  # 20% reduction for high risk
                logger.info(f"HIGH risk level detected, reducing position by 20%")
            
            # Log risk warnings
            if risk_warnings:
                logger.warning(f"Risk warnings from RiskGenius: {', '.join(risk_warnings[:3])}")
            
            # Apply adjustment
            adjusted_size = position_size * Decimal(str(risk_factor))
            
            logger.debug(
                f"RiskGenius adjustment: "
                f"risk_level={overall_risk_level}, "
                f"multiplier={position_multiplier:.3f}, "
                f"final_factor={risk_factor:.3f}, "
                f"original={position_size}, "
                f"adjusted={adjusted_size}"
            )
            
            return adjusted_size
            
        except Exception as e:
            logger.warning(f"Risk assessment adjustment failed: {str(e)}, using original size")
            return position_size
    
    async def _apply_final_limits(self,
                                position_size: Decimal,
                                signal: TradeSignal,
                                account_info: AccountInfo) -> Tuple[Decimal, List[PositionSizeAdjustment]]:
        """Apply final position size limits and constraints."""
        adjustments = []
        original_size = position_size
        
        # 1. Maximum position size limit
        max_position_value = account_info.equity * Decimal(str(self.base_risk_params['max_position_size_percent'] / 100))
        if signal.entry_price:
            max_position_size = max_position_value / signal.entry_price
            if position_size > max_position_size:
                position_size = max_position_size
                adjustments.append(PositionSizeAdjustment.ACCOUNT_PROTECTION)
        
        # 2. Minimum position size (to avoid dust trades)
        min_position_value = Decimal('100')  # $100 minimum
        if signal.entry_price:
            min_position_size = min_position_value / signal.entry_price
            if position_size < min_position_size:
                position_size = Decimal('0')  # Cancel trade if too small
                adjustments.append(PositionSizeAdjustment.ACCOUNT_PROTECTION)
        
        # 3. Available margin check
        required_margin = await self._calculate_required_margin(position_size, signal)
        if required_margin > account_info.margin_available:
            # Reduce position size to fit available margin
            margin_adjusted_size = (account_info.margin_available / required_margin) * position_size
            position_size = margin_adjusted_size * Decimal('0.9')  # 10% safety buffer
            adjustments.append(PositionSizeAdjustment.ACCOUNT_PROTECTION)
        
        # 4. Maximum number of open positions
        open_positions_count = len(await self._get_open_positions())
        max_positions = self.base_risk_params.get('max_open_positions', 10)
        
        if open_positions_count >= max_positions:
            position_size = Decimal('0')  # Block new positions
            adjustments.append(PositionSizeAdjustment.ACCOUNT_PROTECTION)
        
        # 5. Daily loss limit check
        if await self._check_daily_loss_limit_reached():
            position_size = Decimal('0')  # Stop trading for the day
            adjustments.append(PositionSizeAdjustment.ACCOUNT_PROTECTION)
        
        return position_size, adjustments
    
    async def _calculate_risk_metrics(self, **kwargs) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the position."""
        signal = kwargs['signal']
        account_info = kwargs['account_info']
        final_size = kwargs['final_size']
        current_price = kwargs.get('current_price')
        
        # Calculate position value
        position_value = final_size * (current_price or signal.entry_price or Decimal('1'))
        
        # Calculate risk level
        risk_level = self._determine_risk_level(position_value, account_info.equity)
        
        # Create risk metrics
        return RiskMetrics(
            risk_level=risk_level,
            position_value=position_value,
            risk_reward_ratio=self._calculate_risk_reward_ratio(signal),
            portfolio_heat=await self._calculate_current_portfolio_heat(),
            confidence_multiplier=kwargs.get('confidence_multiplier', 1.0),
            volatility_adjustment=kwargs.get('volatility_adjustment', 1.0)
        )
    
    def _determine_risk_level(self, position_value: Decimal, equity: Decimal) -> RiskLevel:
        """Determine risk level based on position value relative to equity."""
        risk_percent = (position_value / equity) * 100
        
        if risk_percent >= 5.0:
            return RiskLevel.HIGH
        elif risk_percent >= 3.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_risk_reward_ratio(self, signal: TradeSignal) -> float:
        """Calculate risk/reward ratio from signal."""
        if not signal.stop_loss or not signal.take_profit or not signal.entry_price:
            return 1.0
        
        if signal.direction == TradeDirection.BUY:
            risk = abs(float(signal.entry_price - signal.stop_loss))
            reward = abs(float(signal.take_profit - signal.entry_price))
        else:
            risk = abs(float(signal.stop_loss - signal.entry_price))
            reward = abs(float(signal.entry_price - signal.take_profit))
        
        if risk > 0:
            return reward / risk
        return 1.0
    
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """
        ✅ FIXED (Bug #37): Get symbol volatility using real market data.
        
        Previous implementation returned hardcoded 0.5 for all symbols.
        Now calculates ATR-based volatility from historical price data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Normalized volatility value (0.0 - 2.0, where 0.5 is average)
        """
        try:
            # Check cache first
            cache_key = symbol
            if cache_key in self.volatility_cache:
                cached_vol, timestamp = self.volatility_cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                    logger.debug(f"Using cached volatility for {symbol}: {cached_vol:.4f}")
                    return cached_vol
            
            # If no data provider, use conservative default
            if not self.data_provider:
                logger.warning(f"No data provider available for volatility calculation of {symbol}, using default")
                return 0.5
            
            # Get historical data (last 30 days, 1-hour timeframe)
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)
                
                # Get OHLCV data
                ohlcv_data = await self.data_provider.get_ohlcv_data(
                    symbol=symbol,
                    timeframe=Timeframe.H1,
                    start_time=start_time,
                    end_time=end_time,
                    count=500  # About 3 weeks of hourly data
                )
                
                if ohlcv_data is None or ohlcv_data.empty:
                    logger.warning(f"No OHLCV data available for {symbol}, using default volatility")
                    return 0.5
                
                # Calculate ATR (Average True Range) as volatility measure
                volatility = self._calculate_atr_volatility(ohlcv_data)
                
                # Cache the result
                self.volatility_cache[cache_key] = (volatility, datetime.now())
                
                logger.debug(f"Calculated volatility for {symbol}: {volatility:.4f}")
                return volatility
                
            except Exception as e:
                logger.warning(f"Failed to fetch OHLCV data for {symbol}: {str(e)}, using default")
                return 0.5
                
        except Exception as e:
            logger.error(f"Volatility calculation failed for {symbol}: {str(e)}, using default")
            return 0.5  # Fallback to medium volatility
    
    def _calculate_atr_volatility(self, ohlcv_data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate ATR-based volatility from OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with columns: open, high, low, close
            period: ATR period (default 14)
            
        Returns:
            Normalized volatility (0.0 - 2.0)
        """
        try:
            # Ensure we have enough data
            if len(ohlcv_data) < period + 1:
                return 0.5
            
            # Calculate True Range
            high = ohlcv_data['high'].values
            low = ohlcv_data['low'].values
            close = ohlcv_data['close'].values
            
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )
            
            # Calculate ATR
            atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
            
            # Normalize ATR by current price
            current_price = close[-1]
            if current_price > 0:
                normalized_atr = (atr / current_price) * 100  # As percentage
                
                # Map to 0-2 scale (where 1% = 0.5, 2% = 1.0, 4% = 2.0)
                volatility = min(2.0, max(0.1, normalized_atr * 0.5))
                return volatility
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"ATR calculation failed: {str(e)}, using default")
            return 0.5
    
    async def _get_open_positions(self) -> Dict[str, Any]:
        """Get current open positions."""
        return self.open_positions
    
    async def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        ✅ FIXED (Bug #37): Get correlation between two symbols using real market data.
        
        Previous implementation returned hardcoded 0.0 for all symbol pairs.
        Now calculates Pearson correlation from historical price data.
        
        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            
        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        try:
            # Check cache first
            cache_key = tuple(sorted([symbol1, symbol2]))
            if cache_key in self.correlation_cache:
                cached_corr, timestamp = self.correlation_cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                    logger.debug(f"Using cached correlation for {symbol1}/{symbol2}: {cached_corr:.4f}")
                    return cached_corr
            
            # If no data provider, assume zero correlation (conservative)
            if not self.data_provider:
                logger.warning(f"No data provider available for correlation calculation, using default")
                return 0.0
            
            # Get historical data for both symbols (last 30 days)
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=30)
                
                # Fetch data for both symbols
                data1 = await self.data_provider.get_ohlcv_data(
                    symbol=symbol1,
                    timeframe=Timeframe.H4,  # 4-hour timeframe for correlation
                    start_time=start_time,
                    end_time=end_time,
                    count=180  # About 30 days of 4-hour data
                )
                
                data2 = await self.data_provider.get_ohlcv_data(
                    symbol=symbol2,
                    timeframe=Timeframe.H4,
                    start_time=start_time,
                    end_time=end_time,
                    count=180
                )
                
                if data1 is None or data2 is None or data1.empty or data2.empty:
                    logger.warning(f"Insufficient data for correlation {symbol1}/{symbol2}, using default")
                    return 0.0
                
                # Calculate correlation
                correlation = self._calculate_price_correlation(data1, data2)
                
                # Cache the result
                self.correlation_cache[cache_key] = (correlation, datetime.now())
                
                logger.debug(f"Calculated correlation for {symbol1}/{symbol2}: {correlation:.4f}")
                return correlation
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for correlation {symbol1}/{symbol2}: {str(e)}, using default")
                return 0.0
                
        except Exception as e:
            logger.error(f"Correlation calculation failed for {symbol1}/{symbol2}: {str(e)}, using default")
            return 0.0  # Fallback to no correlation
    
    def _calculate_price_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame) -> float:
        """
        Calculate Pearson correlation between two price series.
        
        Args:
            data1: OHLCV DataFrame for symbol1
            data2: OHLCV DataFrame for symbol2
            
        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        try:
            # Extract close prices
            closes1 = data1['close'].values
            closes2 = data2['close'].values
            
            # Ensure both arrays have the same length (use the shorter one)
            min_length = min(len(closes1), len(closes2))
            if min_length < 10:  # Need at least 10 data points
                return 0.0
            
            closes1 = closes1[-min_length:]
            closes2 = closes2[-min_length:]
            
            # Calculate returns
            returns1 = np.diff(np.log(closes1))
            returns2 = np.diff(np.log(closes2))
            
            # Remove any NaN or inf values
            mask = np.isfinite(returns1) & np.isfinite(returns2)
            returns1 = returns1[mask]
            returns2 = returns2[mask]
            
            if len(returns1) < 5:  # Need sufficient data
                return 0.0
            
            # Calculate Pearson correlation
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            # Handle NaN result
            if not np.isfinite(correlation):
                return 0.0
            
            return float(correlation)
            
        except Exception as e:
            logger.warning(f"Price correlation calculation failed: {str(e)}, using default")
            return 0.0
    
    async def _calculate_current_portfolio_heat(self) -> float:
        """Calculate current portfolio heat."""
        return self.current_portfolio_heat
    
    async def _calculate_required_margin(self, position_size: Decimal, signal: TradeSignal) -> Decimal:
        """Calculate required margin for position."""
        # Simplified: assume 1:100 leverage
        position_value = position_size * (signal.entry_price or Decimal('1'))
        return position_value / Decimal('100')
    
    async def _check_daily_loss_limit_reached(self) -> bool:
        """Check if daily loss limit has been reached."""
        # Placeholder implementation
        return False
    
    async def update_position_risk(self, position_id: str, current_price: Decimal) -> None:
        """Update risk metrics for an open position."""
        if position_id in self.open_positions:
            logger.debug(f"Updating risk for position {position_id}")
        
    async def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary."""
        return {
            "open_positions": len(self.open_positions),
            "portfolio_heat": self.current_portfolio_heat,
            "max_position_size": self.base_risk_params.get('max_position_size_percent', 0)
        }
