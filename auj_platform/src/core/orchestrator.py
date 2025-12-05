"""
Trading Orchestrator for AUJ Platform - ENHANCED WITH OPPORTUNITY RADAR

This module implements the trading loop that executes analysis cycles every hour.
ENHANCED: Now supports OpportunityRadar for intelligent pair selection instead
of blind rotation - scans ALL pairs and picks the BEST opportunity!

BUG #35 FIX: NO TRADING LOOP - FIXED
ENHANCEMENT: Intelligent Pair Selection with OpportunityRadar

Key Features:
- Hourly analysis cycle execution
- ✨ NEW: OpportunityRadar intelligent pair selection
- Configurable trading hours (market hours only)
- Extended pair support (12 pairs: majors + crosses + metals + crypto)
- Graceful shutdown handling
- Integration with GeniusAgentCoordinator

Author: Antigravity AI - Bug Fix Team + Crown Jewel Innovation
Date: 2025-12-05
Version: 2.0.0 (WITH OPPORTUNITY RADAR)
"""

import asyncio
from datetime import datetime, time
from typing import Optional, List, Any, Dict
import logging
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class TradingOrchestrator:
    """
    ✅ BUG #35 FIX + ✨ CROWN JEWEL ENHANCEMENT
    
    This orchestrator runs the trading loop with INTELLIGENT PAIR SELECTION.
    
    Two modes:
    1. INTELLIGENT MODE (use_intelligent_selection=True):
       - Uses OpportunityRadar to scan ALL pairs
       - Ranks by Opportunity Score
       - Deep-analyzes TOP 3
       - Trades the BEST opportunity
    
    2. ROTATION MODE (use_intelligent_selection=False):
       - Original round-robin symbol rotation
       - Backward compatible
    
    WITHOUT THIS: Platform = Zombie (starts but never trades)
    WITH THIS: Platform = Smart Active Trader (hourly analysis + BEST signals)
    """
    
    # Extended pair list for intelligent selection
    EXTENDED_PAIRS = [
        # Major Pairs
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        # Cross Pairs
        'EURJPY', 'GBPJPY', 'EURGBP',
        # Commodities
        'XAUUSD',  # Gold
        # Crypto
        'BTCUSD'
    ]
    
    def __init__(self,
                 genius_coordinator,
                 config_manager: UnifiedConfigManager,
                 execution_handler=None,
                 economic_monitor=None,
                 opportunity_radar=None):
        """
        Initialize the Trading Orchestrator.
        
        Args:
            genius_coordinator: GeniusAgentCoordinator instance
            config_manager: Configuration manager
            execution_handler: ExecutionHandler instance (optional)
            economic_monitor: EconomicMonitor instance (optional)
            opportunity_radar: OpportunityRadar for intelligent selection (optional)
        """
        self.coordinator = genius_coordinator
        self.config = config_manager
        self.execution_handler = execution_handler
        self.economic_monitor = economic_monitor
        self.opportunity_radar = opportunity_radar
        
        # Orchestrator state
        self.running = False
        self.current_task = None
        
        # Configuration
        self.cycle_interval_seconds = config_manager.get_int(
            'orchestrator.cycle_interval_seconds', 3600  # Default: 1 hour
        )
        
        # ✨ Intelligent Selection Mode
        self.use_intelligent_selection = config_manager.get_bool(
            'orchestrator.use_intelligent_selection', True
        )
        
        # Extended trading symbols (12 pairs)
        self.trading_symbols = config_manager.get_list(
            'orchestrator.trading_symbols', 
            self.EXTENDED_PAIRS
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
        self.intelligent_selections = 0  # Count of OpportunityRadar selections
        
        # Log initialization
        logger.info("=" * 80)
        logger.info("✅ Trading Orchestrator initialized - ENHANCED WITH OPPORTUNITY RADAR!")
        logger.info(f"📊 Cycle interval: {self.cycle_interval_seconds}s ({self.cycle_interval_seconds/3600:.1f} hours)")
        logger.info(f"📈 Trading pairs: {len(self.trading_symbols)} configured")
        
        if self.use_intelligent_selection and self.opportunity_radar:
            logger.info("🎯 Mode: INTELLIGENT SELECTION (OpportunityRadar)")
            logger.info(f"   → Scans ALL pairs, picks BEST opportunity")
        else:
            logger.info("🔄 Mode: ROTATION (round-robin)")
            logger.info(f"   → Pairs: {', '.join(self.trading_symbols[:4])}...")
        
        logger.info(f"⏰ Trading hours: {self.trading_start_hour:02d}:00 - {self.trading_end_hour:02d}:00 UTC")
        logger.info("=" * 80)
    
    async def start(self):
        """
        Start the trading orchestrator loop.
        
        ✅ BUG #35 FIX + ✨ CROWN JEWEL: This is the CRITICAL method!
        """
        if self.running:
            logger.warning("Orchestrator is already running")
            return
        
        self.running = True
        logger.info("🚀 Starting Trading Orchestrator - INTELLIGENT TRADING LOOP ENABLED!")
        logger.info("🎯 Platform will now execute analysis cycles and generate OPTIMAL signals")
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
        logger.info(f"   Intelligent Selections: {self.intelligent_selections}")
    
    async def _trading_loop(self):
        """
        Main trading loop - executes analysis cycles periodically.
        
        ✅ BUG #35 FIX + ✨ CROWN JEWEL: Now with intelligent pair selection!
        """
        logger.info("🔄 Entering main trading loop...")
        
        while self.running:
            try:
                # Check if we're within trading hours
                if self._is_trading_time():
                    logger.info("=" * 80)
                    logger.info(f"🎯 Starting Analysis Cycle #{self.total_cycles + 1}")
                    logger.info(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    
                    # ✨ CROWN JEWEL: Use intelligent selection if available
                    if self.use_intelligent_selection and self.opportunity_radar:
                        signal = await self._execute_intelligent_cycle()
                    else:
                        # Fallback to rotation mode
                        symbol = self._get_next_symbol()
                        logger.info(f"📈 Symbol (rotation): {symbol}")
                        signal = await self._execute_cycle(symbol)
                    
                    logger.info("=" * 80)
                    
                    # ✅ INTEGRATION FIX: Execute economic monitoring cycle
                    if self.economic_monitor:
                        try:
                            await self.economic_monitor.execute_monitoring_cycle()
                            logger.debug("✅ Economic monitoring cycle completed")
                        except Exception as e:
                            logger.error(f"❌ Economic monitoring cycle failed: {e}")
                    
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
    
    async def _execute_intelligent_cycle(self):
        """
        Execute analysis cycle using OpportunityRadar for intelligent pair selection.
        
        ✨ CROWN JEWEL: Scans ALL pairs, picks BEST opportunity!
        
        Returns:
            Generated trade signal or None
        """
        try:
            logger.info("🎯 INTELLIGENT SELECTION MODE: Scanning all pairs...")
            
            # Use OpportunityRadar to find best opportunity
            radar_result = await self.opportunity_radar.find_best_opportunity()
            
            self.intelligent_selections += 1
            
            if not radar_result.best_pair:
                logger.warning("⚠️ OpportunityRadar: No suitable opportunities found")
                return None
            
            logger.info(f"🏆 BEST OPPORTUNITY: {radar_result.best_pair}")
            logger.info(f"   Score: {radar_result.best_score:.2f}")
            logger.info(f"   Direction: {radar_result.best_direction.value}")
            logger.info(f"   Scan time: {radar_result.total_time_seconds:.2f}s")
            logger.info(f"   Pairs scanned: {radar_result.pairs_scanned}")
            logger.info(f"   Deep analyzed: {radar_result.pairs_deep_analyzed}")
            
            # Return the signal from deep analysis if available
            if radar_result.best_analysis:
                # The deep analysis already produced a signal via GeniusAgentCoordinator
                # We can get it from the coordinator's last signal or return a synthesized one
                return await self._execute_cycle(radar_result.best_pair)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Intelligent cycle failed: {e}")
            logger.error("Traceback: ", exc_info=True)
            
            # Fallback to rotation
            logger.info("⚠️ Falling back to rotation mode...")
            symbol = self._get_next_symbol()
            return await self._execute_cycle(symbol)
    
    async def _execute_cycle(self, symbol: str):
        """
        Execute a single analysis cycle.
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Generated trade signal or None
        """
        try:
            # Call execute_analysis_cycle()
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
    
    async def execute_immediate_cycle(self, symbol: Optional[str] = None, use_radar: bool = True):
        """
        Execute an immediate analysis cycle (for testing or manual triggers).
        
        Args:
            symbol: Optional symbol (if None, uses intelligent selection or rotation)
            use_radar: If True and symbol is None, use OpportunityRadar
            
        Returns:
            Generated trade signal or None
        """
        if symbol:
            logger.info(f"🎯 Executing immediate analysis cycle for {symbol}")
            signal = await self._execute_cycle(symbol)
        elif use_radar and self.opportunity_radar and self.use_intelligent_selection:
            logger.info("🎯 Executing immediate intelligent cycle (OpportunityRadar)")
            signal = await self._execute_intelligent_cycle()
        else:
            symbol = self._get_next_symbol()
            logger.info(f"🎯 Executing immediate rotation cycle for {symbol}")
            signal = await self._execute_cycle(symbol)
        
        if signal:
            self.signals_generated += 1
            logger.info(f"✅ Immediate cycle generated signal: {signal.direction.value} {signal.symbol}")
        else:
            logger.info("📊 No signal generated from immediate cycle")
        
        return signal
    
    def set_opportunity_radar(self, radar):
        """
        Set the OpportunityRadar instance.
        
        Args:
            radar: OpportunityRadar instance
        """
        self.opportunity_radar = radar
        logger.info("🎯 OpportunityRadar attached to TradingOrchestrator")
    
    def get_statistics(self) -> dict:
        """
        Get orchestrator statistics.
        
        Returns:
            Dictionary with statistics
        """
        success_rate = (self.successful_cycles / self.total_cycles * 100) if self.total_cycles > 0 else 0
        signal_rate = (self.signals_generated / self.total_cycles * 100) if self.total_cycles > 0 else 0
        execution_rate = (self.trades_executed / self.signals_generated * 100) if self.signals_generated > 0 else 0
        
        stats = {
            'running': self.running,
            'total_cycles': self.total_cycles,
            'successful_cycles': self.successful_cycles,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'success_rate': success_rate,
            'signal_rate': signal_rate,
            'execution_rate': execution_rate,
            'current_symbol_index': self.current_symbol_index,
            'next_symbol': self.trading_symbols[self.current_symbol_index],
            'intelligent_selections': self.intelligent_selections,
            'use_intelligent_selection': self.use_intelligent_selection,
            'opportunity_radar_enabled': self.opportunity_radar is not None
        }
        
        # Add OpportunityRadar stats if available
        if self.opportunity_radar:
            radar_stats = self.opportunity_radar.get_statistics()
            stats['opportunity_radar'] = radar_stats
        
        return stats
