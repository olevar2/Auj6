#!/usr/bin/env python3
"""
Trading Metrics Tracker for AUJ Platform
=========================================

Specialized metrics collector for trading-specific performance indicators.
Tracks P&L, positions, agent performance, and trading statistics to ensure
optimal performance for the mission of helping sick children.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json


@dataclass
class TradeMetric:
    """Individual trade metric data."""
    agent_name: str
    symbol: str
    trade_type: str  # 'buy', 'sell'
    entry_price: float
    exit_price: Optional[float] = None
    volume: float = 0.0
    profit_loss: Optional[float] = None
    duration: Optional[float] = None  # in seconds
    timestamp: datetime = field(default_factory=datetime.now)
    trade_id: Optional[str] = None


@dataclass
class AgentPerformanceMetric:
    """Agent performance metrics."""
    agent_name: str
    timeframe: str
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    total_loss: float
    average_win: float
    average_loss: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class TradingMetricsTracker:
    """
    Comprehensive trading metrics tracking system.
    
    Monitors and analyzes trading performance to ensure the platform
    generates sustainable profits for the humanitarian mission.
    """
    
    def __init__(self, config=None, database=None, metrics_collector=None):
        """Initialize trading metrics tracker."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.database = database
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self._trades_buffer = deque(maxlen=10000)
        self._agent_performance = {}
        self._daily_metrics = defaultdict(dict)
        
        # Performance tracking
        self._tracking_period_days = self.config_manager.get_dict('platform_settings', {}).get(
            'agent_performance_window_days', 30
        )
        
        # Mission tracking
        self._total_profit_for_mission = 0.0
        self._children_helped_estimate = 0
        
        # Market data monitoring
        self.messaging_service = None
        self.market_data_subscription_active = False
        self.position_monitoring_active = False
        self._current_positions = {}
        self._position_prices = {}
        
        self.logger.info("📊 TradingMetricsTracker initialized")
    
    async def initialize(self):
        """Initialize the trading metrics tracker."""
        try:
            self.logger.info("🔧 Initializing trading metrics tracking...")
            
            # Load historical data if available
            await self._load_historical_metrics()
            
            # Initialize mission metrics
            await self._initialize_mission_metrics()
            
            self.logger.info("✅ Trading metrics tracker initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trading metrics tracker: {e}")
            return False
    
    async def _load_historical_metrics(self):
        """Load historical trading metrics from database."""
        if not self.database:
            self.logger.info("📋 No database available for historical metrics")
            return
        
        try:
            # This would load from database in a real implementation
            self.logger.info("📈 Historical metrics loading simulated")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load historical metrics: {e}")
    
    async def _initialize_mission_metrics(self):
        """Initialize mission-specific tracking metrics."""
        # Estimate children helped based on profit (example: $100 helps 1 child)
        cost_per_child_helped = 100.0  # USD
        
        if self._total_profit_for_mission > 0:
            self._children_helped_estimate = int(self._total_profit_for_mission / cost_per_child_helped)
        
        self.logger.info(f"💝 Mission tracking: ${self._total_profit_for_mission:.2f} profit, "
                        f"~{self._children_helped_estimate} children helped estimate")
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade."""
        try:
            trade_metric = TradeMetric(
                agent_name=trade_data['agent_name'],
                symbol=trade_data['symbol'],
                trade_type=trade_data['trade_type'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data.get('exit_price'),
                volume=trade_data.get('volume', 0.0),
                profit_loss=trade_data.get('profit_loss'),
                duration=trade_data.get('duration'),
                trade_id=trade_data.get('trade_id')
            )
            
            self._trades_buffer.append(trade_metric)
            
            # Update agent performance
            self._update_agent_performance(trade_metric)
            
            # Update mission metrics
            if trade_metric.profit_loss:
                self._total_profit_for_mission += trade_metric.profit_loss
                self._update_mission_metrics()
            
            # Update Prometheus metrics
            if self.metrics_collector:
                self._update_prometheus_metrics(trade_metric)
            
            self.logger.debug(f"📈 Trade recorded: {trade_metric.agent_name} "
                            f"{trade_metric.symbol} P&L: {trade_metric.profit_loss}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to record trade: {e}")
    
    def _update_agent_performance(self, trade: TradeMetric):
        """Update agent performance metrics."""
        agent_name = trade.agent_name
        
        if agent_name not in self._agent_performance:
            self._agent_performance[agent_name] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'profits': [],
                'losses': [],
                'trade_durations': [],
                'last_updated': datetime.now()
            }
        
        agent_perf = self._agent_performance[agent_name]
        agent_perf['total_trades'] += 1
        agent_perf['last_updated'] = datetime.now()
        
        if trade.profit_loss is not None:
            if trade.profit_loss > 0:
                agent_perf['winning_trades'] += 1
                agent_perf['total_profit'] += trade.profit_loss
                agent_perf['profits'].append(trade.profit_loss)
            else:
                agent_perf['losing_trades'] += 1
                agent_perf['total_loss'] += abs(trade.profit_loss)
                agent_perf['losses'].append(abs(trade.profit_loss))
        
        if trade.duration:
            agent_perf['trade_durations'].append(trade.duration)
        
        # Limit history size
        if len(agent_perf['profits']) > 1000:
            agent_perf['profits'] = agent_perf['profits'][-1000:]
        if len(agent_perf['losses']) > 1000:
            agent_perf['losses'] = agent_perf['losses'][-1000:]
        if len(agent_perf['trade_durations']) > 1000:
            agent_perf['trade_durations'] = agent_perf['trade_durations'][-1000:]
    
    def _update_mission_metrics(self):
        """Update mission-specific metrics."""
        cost_per_child_helped = 100.0  # USD
        new_estimate = int(self._total_profit_for_mission / cost_per_child_helped)
        
        if new_estimate > self._children_helped_estimate:
            children_helped_today = new_estimate - self._children_helped_estimate
            self._children_helped_estimate = new_estimate
            
            if children_helped_today > 0:
                self.logger.info(f"💝 Mission update: ~{children_helped_today} more children "
                               f"can be helped! Total: ~{self._children_helped_estimate}")
    
    def _update_prometheus_metrics(self, trade: TradeMetric):
        """Update Prometheus metrics with trade data."""
        if not self.metrics_collector:
            return
        
        try:
            # Record trade execution
            result = "win" if trade.profit_loss and trade.profit_loss > 0 else "loss"
            self.metrics_collector.record_trade_execution(
                agent_name=trade.agent_name,
                result=result,
                execution_time=trade.duration or 0.0
            )
            
            # Update agent performance metrics
            agent_perf = self.get_agent_performance(trade.agent_name)
            if agent_perf:
                self.metrics_collector.record_agent_performance(
                    agent_name=trade.agent_name,
                    performance_data=agent_perf
                )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update Prometheus metrics: {e}")
    
    def get_agent_performance(self, agent_name: str, timeframe: str = "1H") -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific agent."""
        if agent_name not in self._agent_performance:
            return None
        
        agent_perf = self._agent_performance[agent_name]
        
        # Calculate derived metrics
        total_trades = agent_perf['total_trades']
        winning_trades = agent_perf['winning_trades']
        losing_trades = agent_perf['losing_trades']
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        profits = agent_perf['profits']
        losses = agent_perf['losses']
        
        average_win = sum(profits) / len(profits) if profits else 0.0
        average_loss = sum(losses) / len(losses) if losses else 0.0
        
        net_profit = agent_perf['total_profit'] - agent_perf['total_loss']
        
        # Calculate profit factor
        profit_factor = agent_perf['total_profit'] / agent_perf['total_loss'] if agent_perf['total_loss'] > 0 else 0.0
        
        # Calculate Sharpe ratio (simplified)
        if profits:
            profit_std = self._calculate_std(profits) if len(profits) > 1 else 0.0
            sharpe_ratio = average_win / profit_std if profit_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'agent_name': agent_name,
            'timeframe': timeframe,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_profit': agent_perf['total_profit'],
            'total_loss': agent_perf['total_loss'],
            'net_profit': net_profit,
            'average_win': average_win,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'last_updated': agent_perf['last_updated']
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_platform_performance_summary(self) -> Dict[str, Any]:
        """Get overall platform performance summary."""
        all_agents = list(self._agent_performance.keys())
        
        if not all_agents:
            return {
                'total_agents': 0,
                'total_trades': 0,
                'total_profit': 0.0,
                'platform_win_rate': 0.0,
                'mission_metrics': self._get_mission_metrics()
            }
        
        # Aggregate metrics across all agents
        total_trades = sum(perf['total_trades'] for perf in self._agent_performance.values())
        total_winning_trades = sum(perf['winning_trades'] for perf in self._agent_performance.values())
        total_profit = sum(perf['total_profit'] for perf in self._agent_performance.values())
        total_loss = sum(perf['total_loss'] for perf in self._agent_performance.values())
        
        platform_win_rate = total_winning_trades / total_trades if total_trades > 0 else 0.0
        net_profit = total_profit - total_loss
        
        # Get top performing agents
        agent_performances = []
        for agent_name in all_agents:
            agent_perf = self.get_agent_performance(agent_name)
            if agent_perf:
                agent_performances.append(agent_perf)
        
        # Sort by net profit
        top_agents = sorted(agent_performances, key=lambda x: x['net_profit'], reverse=True)[:5]
        
        return {
            'total_agents': len(all_agents),
            'total_trades': total_trades,
            'total_winning_trades': total_winning_trades,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'platform_win_rate': platform_win_rate,
            'top_agents': top_agents,
            'mission_metrics': self._get_mission_metrics()
        }
    
    def _get_mission_metrics(self) -> Dict[str, Any]:
        """Get mission-specific metrics."""
        return {
            'total_profit_for_mission': self._total_profit_for_mission,
            'estimated_children_helped': self._children_helped_estimate,
            'cost_per_child': 100.0,  # USD
            'mission_statement': 'Generate sustainable profits for sick children and families in need'
        }
    
    def get_daily_summary(self, date: datetime = None) -> Dict[str, Any]:
        """Get daily trading summary."""
        if date is None:
            date = datetime.now()
        
        date_key = date.strftime('%Y-%m-%d')
        
        # Filter trades for the specific date
        daily_trades = [
            trade for trade in self._trades_buffer
            if trade.timestamp.strftime('%Y-%m-%d') == date_key
        ]
        
        if not daily_trades:
            return {
                'date': date_key,
                'total_trades': 0,
                'total_profit': 0.0,
                'win_rate': 0.0,
                'agents_active': 0
            }
        
        # Calculate daily metrics
        total_trades = len(daily_trades)
        winning_trades = sum(1 for trade in daily_trades if trade.profit_loss and trade.profit_loss > 0)
        total_profit = sum(trade.profit_loss for trade in daily_trades if trade.profit_loss)
        agents_active = len(set(trade.agent_name for trade in daily_trades))
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            'date': date_key,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'agents_active': agents_active,
            'trades': [
                {
                    'agent': trade.agent_name,
                    'symbol': trade.symbol,
                    'profit_loss': trade.profit_loss,
                    'duration': trade.duration
                }
                for trade in daily_trades[-10:]  # Last 10 trades
            ]
        }
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            platform_summary = self.get_platform_performance_summary()
            daily_summary = self.get_daily_summary()
            
            # Agent rankings
            agent_rankings = []
            for agent_name in self._agent_performance.keys():
                agent_perf = self.get_agent_performance(agent_name)
                if agent_perf and agent_perf['total_trades'] >= 5:  # Minimum trades for ranking
                    agent_rankings.append(agent_perf)
            
            # Sort by win rate, then by net profit
            agent_rankings.sort(key=lambda x: (x['win_rate'], x['net_profit']), reverse=True)
            
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'platform_summary': platform_summary,
                'daily_summary': daily_summary,
                'agent_rankings': agent_rankings[:10],  # Top 10 agents
                'mission_impact': {
                    'profit_generated': self._total_profit_for_mission,
                    'children_helped_estimate': self._children_helped_estimate,
                    'humanitarian_impact': 'Sustainable trading profits supporting sick children and families'
                }
            }
            
            self.logger.info(f"📊 Performance report generated - Platform win rate: "
                           f"{platform_summary['platform_win_rate']:.1%}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate performance report: {e}")
            return {}
    
    async def start_market_data_monitoring(self, messaging_service=None):
        """
        Start monitoring market data for position updates.
        
        Args:
            messaging_service: Messaging service to subscribe to market data
        """
        try:
            if self.market_data_subscription_active:
                self.logger.warning("Market data monitoring is already active")
                return
            
            if messaging_service:
                self.messaging_service = messaging_service
            
            if not self.messaging_service:
                self.logger.error("No messaging service available for market data monitoring")
                return
            
            # Subscribe to market data messages
            await self.messaging_service.subscribe(
                message_type='market_data',
                handler=self._handle_market_data_update
            )
            
            self.market_data_subscription_active = True
            self.logger.info("📈 Market data monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start market data monitoring: {e}")
    
    async def start_position_monitoring(self, execution_handler=None):
        """
        Start monitoring open positions for real-time P&L updates.
        
        Args:
            execution_handler: Trading engine execution handler for position data
        """
        try:
            if self.position_monitoring_active:
                self.logger.warning("Position monitoring is already active")
                return
            
            # Get current positions from execution handler
            if execution_handler:
                positions = await execution_handler.get_open_positions()
                if positions:
                    for position in positions:
                        symbol = position.get('symbol')
                        if symbol:
                            self._current_positions[symbol] = position
                            self.logger.info(f"📊 Monitoring position: {symbol}")
            
            self.position_monitoring_active = True
            self.logger.info("📈 Position monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start position monitoring: {e}")
    
    async def _handle_market_data_update(self, message):
        """
        Handle incoming market data updates and update position monitoring.
        
        Args:
            message: Market data message containing price updates
        """
        try:
            # Extract message data
            symbol = message.get('symbol')
            data_type = message.get('data_type')
            price_data = message.get('data', {})
            
            if not symbol or not price_data:
                return
            
            # Update price tracking
            if data_type == 'price':
                current_price = price_data.get('bid', price_data.get('last', 0))
                if current_price > 0:
                    self._position_prices[symbol] = current_price
                    
                    # Update position P&L if we have an open position
                    if symbol in self._current_positions:
                        await self._update_position_pnl(symbol, current_price)
            
            elif data_type == 'ohlcv':
                # Use close price for position updates
                close_price = price_data.get('close', 0)
                if close_price > 0:
                    self._position_prices[symbol] = close_price
                    
                    if symbol in self._current_positions:
                        await self._update_position_pnl(symbol, close_price)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling market data update: {e}")
    
    async def _update_position_pnl(self, symbol: str, current_price: float):
        """
        Update position P&L based on current market price.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        try:
            position = self._current_positions.get(symbol)
            if not position:
                return
            
            entry_price = position.get('entry_price', 0)
            volume = position.get('volume', 0)
            position_type = position.get('type', 'buy')  # 'buy' or 'sell'
            
            if entry_price > 0 and volume > 0:
                # Calculate unrealized P&L
                if position_type.lower() == 'buy':
                    unrealized_pnl = (current_price - entry_price) * volume
                else:  # sell position
                    unrealized_pnl = (entry_price - current_price) * volume
                
                # Update position with current P&L
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
                position['last_update'] = datetime.now().isoformat()
                
                # Log significant P&L changes
                previous_pnl = position.get('previous_pnl', 0)
                pnl_change = abs(unrealized_pnl - previous_pnl)
                if pnl_change > 10:  # Log changes > $10
                    self.logger.info(f"📊 {symbol} P&L: ${unrealized_pnl:.2f} "
                                   f"(${current_price:.4f})")
                
                position['previous_pnl'] = unrealized_pnl
                
        except Exception as e:
            self.logger.error(f"❌ Error updating position P&L for {symbol}: {e}")
    
    def update_position(self, position_data: Dict[str, Any]):
        """
        Update tracked position data.
        
        Args:
            position_data: Position information from trading engine
        """
        try:
            symbol = position_data.get('symbol')
            if symbol:
                if position_data.get('status') == 'closed':
                    # Remove closed positions
                    self._current_positions.pop(symbol, None)
                    self.logger.info(f"📊 Position closed: {symbol}")
                else:
                    # Update or add open position
                    self._current_positions[symbol] = position_data
                    self.logger.debug(f"📊 Position updated: {symbol}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error updating position: {e}")
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """
        Get current position summary with real-time P&L.
        
        Returns:
            Dictionary with position summary and total unrealized P&L
        """
        try:
            total_unrealized_pnl = 0
            position_count = 0
            position_summary = []
            
            for symbol, position in self._current_positions.items():
                unrealized_pnl = position.get('unrealized_pnl', 0)
                total_unrealized_pnl += unrealized_pnl
                position_count += 1
                
                position_summary.append({
                    'symbol': symbol,
                    'type': position.get('type'),
                    'volume': position.get('volume', 0),
                    'entry_price': position.get('entry_price', 0),
                    'current_price': position.get('current_price', 0),
                    'unrealized_pnl': unrealized_pnl
                })
            
            return {
                'total_positions': position_count,
                'total_unrealized_pnl': total_unrealized_pnl,
                'positions': position_summary,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating position summary: {e}")
            return {}
    
    async def stop_monitoring(self):
        """Stop all monitoring activities."""
        try:
            self.market_data_subscription_active = False
            self.position_monitoring_active = False
            
            if self.messaging_service:
                await self.messaging_service.unsubscribe(
                    message_type='market_data',
                    handler=self._handle_market_data_update
                )
            
            self.logger.info("📊 Trading monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping monitoring: {e}")

    async def health_check(self) -> bool:
        """Health check for trading metrics tracker."""
        try:
            # Check if we have recent trade data
            if self._trades_buffer:
                latest_trade = self._trades_buffer[-1]
                time_since_last_trade = datetime.now() - latest_trade.timestamp
                
                # If no trades in last 24 hours, might be an issue
                if time_since_last_trade > timedelta(hours=24):
                    self.logger.warning("⚠️ No recent trades recorded")
                    return False
            
            # Check agent performance data
            if not self._agent_performance:
                self.logger.warning("⚠️ No agent performance data available")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trading metrics tracker health check failed: {e}")
            return False