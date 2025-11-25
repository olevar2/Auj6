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
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
import logging
import math
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
                                    current_price: Optional[Decimal] = None) -> Tuple[Decimal, RiskMetrics]:
        """
        Calculate optimal position size using confidence scaling and dynamic risk adjustment.
        
        Args:
            signal: Trade signal with confidence level
            account_info: Current account information
            market_conditions: Current market conditions
            current_price: Current market price for the symbol
            
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
            
            # Step 7: Apply final limits and validation
            final_position_size, adjustments = await self._apply_final_limits(
                regime_adjusted_size, signal, account_info
            )
            
            # Step 8: Calculate comprehensive risk metrics
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
                adjustments=adjustments,
                current_price=current_price
            )
            
            logger.info(f"Position size calculated: {final_position_size} "
                       f"(Risk Level: {risk_metrics.risk_level.value}, "
                       f"Confidence: {signal.confidence:.3f})")
            
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
        """Calculate volatility-based position size adjustment."""
        try:
            if market_conditions:
                volatility = market_conditions.volatility
            else:
                # Get recent volatility data
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
        """Calculate correlation penalty to reduce correlated exposure."""
        try:
            # Get current open positions
            open_positions = await self._get_open_positions()
            
            if not open_positions:
                return 1.0  # No penalty if no open positions
            
            # Calculate correlation with open positions
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
            position_size = Decimal('0')
            adjustments.append(PositionSizeAdjustment.PORTFOLIO_HEAT)
        
        # 5. Daily loss limit check
        if await self._check_daily_loss_limit():
            position_size = Decimal('0')
            adjustments.append(PositionSizeAdjustment.ACCOUNT_PROTECTION)
        
        return position_size.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP), adjustments
    
    async def _calculate_risk_metrics(self,
                                    signal: TradeSignal,
                                    account_info: AccountInfo,
                                    base_size: Decimal,
                                    final_size: Decimal,
                                    confidence_multiplier: float,
                                    volatility_adjustment: float,
                                    correlation_penalty: float,
                                    heat_adjustment: float,
                                    regime_adjustment: float,
                                    adjustments: List[PositionSizeAdjustment],
                                    current_price: Optional[Decimal]) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the trade."""
        try:
            # Calculate position risk as percentage of equity
            position_value = final_size * (current_price or signal.entry_price or Decimal('1'))
            position_risk_percent = float((position_value / account_info.equity) * 100)
            
            # Calculate maximum loss amount
            if signal.stop_loss and current_price:
                if signal.direction == TradeDirection.BUY:
                    max_loss_per_unit = current_price - signal.stop_loss
                else:
                    max_loss_per_unit = signal.stop_loss - current_price
                max_loss_amount = final_size * max_loss_per_unit
            else:
                # Estimate based on typical stop loss percentage
                stop_loss_percent = self.base_risk_params.get('stop_loss_percent', 1.0)
                max_loss_amount = position_value * Decimal(str(stop_loss_percent / 100))
            
            # Calculate risk-reward ratio
            if signal.take_profit and signal.stop_loss and current_price:
                if signal.direction == TradeDirection.BUY:
                    potential_profit = (signal.take_profit - current_price) * final_size
                    potential_loss = (current_price - signal.stop_loss) * final_size
                else:
                    potential_profit = (current_price - signal.take_profit) * final_size
                    potential_loss = (signal.stop_loss - current_price) * final_size
                
                risk_reward_ratio = float(potential_profit / max(potential_loss, Decimal('0.01')))
            else:
                # Default risk-reward based on config
                risk_reward_ratio = self.base_risk_params.get('take_profit_percent', 2.0) / self.base_risk_params.get('stop_loss_percent', 1.0)
            
            # Calculate current portfolio heat
            portfolio_heat = await self._calculate_current_portfolio_heat()
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                position_risk_percent, portfolio_heat, max_loss_amount, account_info.equity
            )
            
            # Generate warnings
            warnings = self._generate_risk_warnings(
                position_risk_percent, portfolio_heat, risk_level, adjustments
            )
            
            return RiskMetrics(
                position_risk_percent=position_risk_percent,
                portfolio_heat=portfolio_heat,
                volatility_adjustment=volatility_adjustment,
                correlation_penalty=correlation_penalty,
                confidence_multiplier=confidence_multiplier,
                final_position_size=final_size,
                max_loss_amount=max_loss_amount,
                risk_reward_ratio=risk_reward_ratio,
                risk_level=risk_level,
                adjustments_applied=adjustments,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {str(e)}")
            # Return safe default metrics
            return RiskMetrics(
                position_risk_percent=0.0,
                portfolio_heat=0.0,
                volatility_adjustment=1.0,
                correlation_penalty=1.0,
                confidence_multiplier=0.5,
                final_position_size=Decimal('0'),
                max_loss_amount=Decimal('0'),
                risk_reward_ratio=1.0,
                risk_level=RiskLevel.VERY_HIGH,
                adjustments_applied=[PositionSizeAdjustment.ACCOUNT_PROTECTION],
                warnings=["Risk calculation error - trade rejected"]
            )    
    def _determine_risk_level(self,
                            position_risk_percent: float,
                            portfolio_heat: float,
                            max_loss_amount: Decimal,
                            account_equity: Decimal) -> RiskLevel:
        """Determine overall risk level for the trade."""
        # Calculate loss as percentage of equity
        loss_percent = float((max_loss_amount / account_equity) * 100)
        
        # Risk level thresholds
        if (position_risk_percent > 8.0 or 
            portfolio_heat > 20.0 or 
            loss_percent > 3.0):
            return RiskLevel.CRITICAL
        elif (position_risk_percent > 6.0 or 
              portfolio_heat > 15.0 or 
              loss_percent > 2.0):
            return RiskLevel.VERY_HIGH
        elif (position_risk_percent > 4.0 or 
              portfolio_heat > 10.0 or 
              loss_percent > 1.5):
            return RiskLevel.HIGH
        elif (position_risk_percent > 2.0 or 
              portfolio_heat > 5.0 or 
              loss_percent > 1.0):
            return RiskLevel.MEDIUM
        elif (position_risk_percent > 1.0 or 
              portfolio_heat > 2.0 or 
              loss_percent > 0.5):
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _generate_risk_warnings(self,
                              position_risk_percent: float,
                              portfolio_heat: float,
                              risk_level: RiskLevel,
                              adjustments: List[PositionSizeAdjustment]) -> List[str]:
        """Generate risk warnings based on current conditions."""
        warnings = []
        
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.VERY_HIGH]:
            warnings.append(f"High risk level: {risk_level.value}")
        
        if position_risk_percent > 5.0:
            warnings.append(f"Large position size: {position_risk_percent:.1f}% of equity")
        
        if portfolio_heat > 12.0:
            warnings.append(f"High portfolio heat: {portfolio_heat:.1f}%")
        
        if PositionSizeAdjustment.CORRELATION_REDUCTION in adjustments:
            warnings.append("Position reduced due to correlation risk")
        
        if PositionSizeAdjustment.PORTFOLIO_HEAT in adjustments:
            warnings.append("Position limited due to portfolio heat")
        
        if PositionSizeAdjustment.ACCOUNT_PROTECTION in adjustments:
            warnings.append("Position adjusted for account protection")
        
        return warnings
    
    # Helper methods for data retrieval and calculations
    
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get recent volatility for a symbol."""
        try:
            # This would typically fetch from market data provider
            # For now, return a default volatility estimate
            return 0.5  # Default moderate volatility
        except Exception as e:
            logger.warning(f"Failed to get volatility for {symbol}: {str(e)}")
            return 0.5
    
    async def _get_open_positions(self) -> Dict[str, Any]:
        """Get currently open positions."""
        try:
            if self.portfolio_tracker:
                return await self.portfolio_tracker.get_open_positions()
            else:
                return self.open_positions
        except Exception as e:
            logger.warning(f"Failed to get open positions: {str(e)}")
            return {}
    
    async def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        try:
            # This would typically calculate from historical price data
            # For now, return estimated correlation based on symbol types
            
            # Simple correlation estimation based on currency pairs
            if symbol1[:3] == symbol2[:3] or symbol1[3:] == symbol2[3:]:
                return 0.8  # High correlation for shared currencies
            elif symbol1[:3] == symbol2[3:] or symbol1[3:] == symbol2[:3]:
                return -0.6  # Negative correlation for inverse pairs
            else:
                return 0.2  # Low correlation for unrelated pairs
                
        except Exception as e:
            logger.warning(f"Failed to get correlation between {symbol1} and {symbol2}: {str(e)}")
            return 0.0
    
    async def _calculate_current_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total risk exposure)."""
        try:
            open_positions = await self._get_open_positions()
            total_heat = 0.0
            
            for position_info in open_positions.values():
                # Calculate heat contribution of each position
                position_risk = position_info.get('risk_percent', 0.0)
                total_heat += position_risk
            
            return total_heat
            
        except Exception as e:
            logger.warning(f"Failed to calculate portfolio heat: {str(e)}")
            return 0.0
    
    async def _calculate_required_margin(self, position_size: Decimal, signal: TradeSignal) -> Decimal:
        """Calculate required margin for the position."""
        try:
            # Simplified margin calculation (would vary by broker and instrument)
            position_value = position_size * (signal.entry_price or Decimal('1'))
            
            # Assume 1:100 leverage for forex (1% margin requirement)
            margin_requirement = position_value * Decimal('0.01')
            
            return margin_requirement
            
            # Clean up old daily loss tracking
            keys_to_remove = [date for date in self.daily_loss_tracking.keys() if date < yesterday]
            for date in keys_to_remove:
                del self.daily_loss_tracking[date]
            
            # Apply heat decay
            self.current_portfolio_heat *= self.heat_decay_factor
            
            logger.info("Daily risk tracking reset")
            
        except Exception as e:
            logger.error(f"Failed to reset daily tracking: {str(e)}")
    
    def __str__(self) -> str:
        return f"DynamicRiskManager(heat={self.current_portfolio_heat:.1f}%, positions={len(self.open_positions)})"
    
    def __repr__(self) -> str:
        return (f"DynamicRiskManager(portfolio_heat={self.current_portfolio_heat:.1f}, "
                f"open_positions={len(self.open_positions)}, "
                f"max_position_size={self.base_risk_params.get('max_position_size_percent', 0)}%)")