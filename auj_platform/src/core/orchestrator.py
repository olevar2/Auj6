"""
Trading Orchestrator for AUJ Platform - CRITICAL BUG #35 FIX

This module implements the missing trading loop that executes analysis cycles
every hour. Without this orchestrator, the platform is a "Zombie" - it starts
and runs but NEVER places any trades!

BUG #35 FIX: NO TRADING LOOP
- ISSUE: execute_analysis_cycle() exists but is never called
- IMPACT: Platform never trades - 100% missed opportunities
- FIX: New orchestrator module that runs hourly trading cycles

Key Features:
- Hourly analysis cycle execution
- Configurable trading hours (market hours only)
- Automatic symbol rotation
- Graceful shutdown handling
- Integration with GeniusAgentCoordinator

Author: Antigravity AI - Bug Fix Team
Date: 2025-12-02
Version: 1.0.0 (BUG #35 FIX)
"""

import asyncio
from datetime import datetime, time
from typing import Optional, List
import logging
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class TradingOrchestrator:
    """
    ✅ BUG #35 FIX: Trading Orchestrator - The Missing Link!
    
    This orchestrator runs the actual trading loop that was missing from the platform.
    It executes analysis cycles every hour during market hours and generates trading signals.
    
    WITHOUT THIS: Platform = Zombie (starts but never trades)
    WITH THIS: Platform = Active Trader (hourly analysis + signals)
    """
    
    def __init__(self,
                 genius_coordinator,
                 config_manager: UnifiedConfigManager,
                 execution_handler=None):
        """
        Initialize the Trading Orchestrator.
        
        Args:
            genius_coordinator: GeniusAgentCoordinator instance
            config_manager: Configuration manager
            execution_handler: ExecutionHandler instance (optional)
        """
        self.coordinator = genius_coordinator
        self.config = config_manager
        self.execution_handler = execution_handler
        
        # Orchestrator state
        self.running = False
        self.current_task = None
        
        # Configuration
        self.cycle_interval_seconds = config_manager.get_int(
            'orchestrator.cycle_interval_seconds', 3600  # Default: 1 hour
        )
        self.trading_symbols = config_manager.get_list(
            'orchestrator.trading_symbols', 
            ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        )
        self.current_symbol_index = 0
        
        # Trading hours (UTC)
        self.trading_start_hour = config_manager.get_int('orchestrator.trading_start_hour', 0)
        self.trading_end_hour = config_manager.get_int('orchestrator.trading_end_hour', 23)
        
        # Enable/disable trading hours check
        self.respect_trading_hours = config_manager.get_bool(
            'orchestrator.respect_trading_hours', True
        )
        
        # Statistics
        self.total_cycles = 0
        self.successful_cycles = 0
        self.signals_generated = 0
        self.trades_executed = 0
        
        logger.info("✅ Trading Orchestrator initialized - BUG #35 FIXED!")
        logger.info(f"📊 Cycle interval: {self.cycle_interval_seconds}s ({self.cycle_interval_seconds/3600:.1f} hours)")
        logger.info(f"📈 Trading symbols: {', '.join(self.trading_symbols)}")
        logger.info(f"⏰ Trading hours: {self.trading_start_hour:02d}:00 - {self.trading_end_hour:02d}:00 UTC")
    
    async def start(self):
        """
        Start the trading orchestrator loop.
        
        ✅ BUG #35 FIX: This is the CRITICAL method that was missing!
        Without this, the platform never executes any trading cycles.
        """
        if self.running:
            logger.warning("Orchestrator is already running")
            return
        
        self.running = True
        logger.info("🚀 Starting Trading Orchestrator - HOURLY TRADING LOOP ENABLED!")
        logger.info("🎯 Platform will now execute analysis cycles and generate signals")
        logger.info("=" * 80)
        
        # Start the main trading loop
        self.current_task = asyncio.create_task(self._trading_loop())
        
        logger.info("✅ Trading loop task created and running")
    
    async def stop(self):
        """Stop the trading orchestrator gracefully."""
        if not self.running:
            return
        
        logger.info("🛑 Stopping Trading Orchestrator...")
        self.running = False
        
        # Cancel the current task
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                logger.info("✅ Trading loop task cancelled")
        
        logger.info("✅ Trading Orchestrator stopped")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Total Cycles: {self.total_cycles}")
        logger.info(f"   Successful: {self.successful_cycles}")
        logger.info(f"   Signals Generated: {self.signals_generated}")
        logger.info(f"   Trades Executed: {self.trades_executed}")
    
    async def _trading_loop(self):
        """
        Main trading loop - executes analysis cycles periodically.
        
        ✅ BUG #35 FIX: This is the HEART of the fix!
        This loop runs continuously and calls execute_analysis_cycle() every hour.
        """
        logger.info("🔄 Entering main trading loop...")
        
        while self.running:
            try:
                # Check if we're within trading hours
                if self._is_trading_time():
                    # Get next symbol to analyze
                    symbol = self._get_next_symbol()
                    
                    logger.info("=" * 80)
                    logger.info(f"🎯 Starting Analysis Cycle #{self.total_cycles + 1}")
                    logger.info(f"📈 Symbol: {symbol}")
                    logger.info(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    logger.info("=" * 80)
                    
                    # ✅ BUG #35 FIX: THIS IS THE CRITICAL CALL!
                    # This actually executes the trading analysis cycle
                    signal = await self._execute_cycle(symbol)
                    
                    # Update statistics
                    self.total_cycles += 1
                    if signal:
                        self.signals_generated += 1
                        logger.info(f"✅ Signal generated: {signal.direction.value} {signal.symbol}")
                        
                        # Execute the signal if execution_handler is available
                        if self.execution_handler:
                            try:
                                report = await self.execution_handler.execute_trade_signal(signal)
                                if report.success:
                                    self.trades_executed += 1
                                    logger.info(f"✅ Trade executed: {report.execution_id}")
                                else:
                                    logger.warning(f"⚠️ Trade execution failed: {report.errors}")
                            except Exception as e:
                                logger.error(f"❌ Failed to execute signal: {e}")
                    else:
                        logger.info("📊 No trading signal generated this cycle")
                    
                    self.successful_cycles += 1
                    
                else:
                    logger.info(f"⏸️ Outside trading hours - skipping cycle (current: {datetime.utcnow().hour:02d}:00 UTC)")
                
                # Wait for next cycle
                logger.info(f"⏳ Waiting {self.cycle_interval_seconds/60:.0f} minutes until next cycle...")
                await asyncio.sleep(self.cycle_interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                logger.error(f"Traceback: ", exc_info=True)
                
                # Wait before retrying
                logger.info("⏳ Waiting 5 minutes before retry...")
                await asyncio.sleep(300)  # 5 minutes
    
    async def _execute_cycle(self, symbol: str):
        """
        Execute a single analysis cycle.
        
        ✅ BUG #35 FIX: This wraps execute_analysis_cycle() with error handling
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Generated trade signal or None
        """
        try:
            # ✅ THIS IS THE CRITICAL LINE - Call to execute_analysis_cycle()
            signal = await self.coordinator.execute_analysis_cycle(symbol=symbol)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Analysis cycle failed for {symbol}: {e}")
            logger.error("Traceback: ", exc_info=True)
            return None
    
    def _is_trading_time(self) -> bool:
        """
        Check if current time is within configured trading hours.
        
        Returns:
            True if within trading hours, False otherwise
        """
        if not self.respect_trading_hours:
            return True
        
        current_hour = datetime.utcnow().hour
        
        # Handle wrapping (e.g., 22:00 - 02:00)
        if self.trading_start_hour <= self.trading_end_hour:
            return self.trading_start_hour <= current_hour <= self.trading_end_hour
        else:
            return current_hour >= self.trading_start_hour or current_hour <= self.trading_end_hour
    
    def _get_next_symbol(self) -> str:
        """
        Get the next symbol to analyze (round-robin).
        
        Returns:
            Trading symbol
        """
        symbol = self.trading_symbols[self.current_symbol_index]
        self.current_symbol_index = (self.current_symbol_index + 1) % len(self.trading_symbols)
        return symbol
    
    async def execute_immediate_cycle(self, symbol: Optional[str] = None):
        """
        Execute an immediate analysis cycle (for testing or manual triggers).
        
        Args:
            symbol: Optional symbol (uses next in rotation if not provided)
            
        Returns:
            Generated trade signal or None
        """
        if not symbol:
            symbol = self._get_next_symbol()
        
        logger.info(f"🎯 Executing immediate analysis cycle for {symbol}")
        
        signal = await self._execute_cycle(symbol)
        
        if signal:
            self.signals_generated += 1
            logger.info(f"✅ Immediate cycle generated signal: {signal.direction.value} {signal.symbol}")
        else:
            logger.info("📊 No signal generated from immediate cycle")
        
        return signal
    
    def get_statistics(self) -> dict:
        """
        Get orchestrator statistics.
        
        Returns:
            Dictionary with statistics
        """
        success_rate = (self.successful_cycles / self.total_cycles * 100) if self.total_cycles > 0 else 0
        signal_rate = (self.signals_generated / self.total_cycles * 100) if self.total_cycles > 0 else 0
        execution_rate = (self.trades_executed / self.signals_generated * 100) if self.signals_generated > 0 else 0
        
        return {
            'running': self.running,
            'total_cycles': self.total_cycles,
            'successful_cycles': self.successful_cycles,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'success_rate': success_rate,
            'signal_rate': signal_rate,
            'execution_rate': execution_rate,
            'current_symbol_index': self.current_symbol_index,
            'next_symbol': self.trading_symbols[self.current_symbol_index]
        }
