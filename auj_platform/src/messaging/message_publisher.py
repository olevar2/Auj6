#!/usr/bin/env python3
"""
Message Publisher for AUJ Platform
==================================

High-level message publishing interface with reliability features,
routing intelligence, and integration with the platform's message types.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import json
import uuid

from .message_broker import MessageBroker
from .message_types import (
    BaseMessage, MessageType, MessagePriority,
    TradingSignalMessage, AgentCoordinationMessage,
    SystemStatusMessage, MarketDataMessage,
    RiskManagementMessage, PerformanceUpdateMessage,
    ErrorNotificationMessage
)


class MessagePublisher:
    """
    High-level message publisher for AUJ platform.
    
    Provides a simple interface for publishing different types of messages
    with automatic routing, priority handling, and reliability features.
    """
    
    def __init__(self, message_broker: MessageBroker, config: Dict[str, Any] = None):
        """Initialize message publisher."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.message_broker = message_broker
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Publisher configuration
        self.publisher_config = self.config_manager.get_dict('message_publisher', {})
        self.default_exchange = self.publisher_config.get('default_exchange', 'auj.platform')
        self.enable_confirmations = self.publisher_config.get('enable_confirmations', True)
        self.max_retry_attempts = self.publisher_config.get('max_retry_attempts', 3)
        self.retry_delay = self.publisher_config.get('retry_delay', 1.0)
        
        # Message tracking
        self._published_messages = {}
        self._failed_messages = []
        self._publish_stats = {
            'total_published': 0,
            'successful_publishes': 0,
            'failed_publishes': 0,
            'retry_attempts': 0
        }
        
        self.logger.info("📤 MessagePublisher initialized")
    
    async def publish_message(
        self,
        message: BaseMessage,
        exchange_name: Optional[str] = None,
        routing_key: Optional[str] = None,
        retry_on_failure: bool = True
    ) -> bool:
        """
        Publish a message with automatic routing and error handling.
        
        Args:
            message: The message to publish
            exchange_name: Override default exchange
            routing_key: Override message routing key
            retry_on_failure: Whether to retry on failure
            
        Returns:
            bool: True if message was published successfully
        """
        try:
            # Validate message
            if not message.validate():
                self.logger.error(f"❌ Invalid message: {message.message_id}")
                return False
            
            # Determine exchange and routing key
            exchange = exchange_name or self._get_exchange_for_message(message)
            routing = routing_key or message.routing_key
            
            # Serialize message
            message_body = message.to_json()
            
            # Prepare headers
            headers = {
                'message_id': message.message_id,
                'message_type': message.message_type.value,
                'source': message.source,
                'timestamp': message.timestamp.isoformat(),
                **message.headers
            }
            
            # Add correlation ID if present
            if message.correlation_id:
                headers['correlation_id'] = message.correlation_id
            
            # Publish with retry logic
            success = await self._publish_with_retry(
                message_body=message_body,
                exchange_name=exchange,
                routing_key=routing,
                priority=message.priority.value,
                expiration=message.expiration,
                headers=headers,
                retry_on_failure=retry_on_failure
            )
            
            # Update statistics
            self._publish_stats['total_published'] += 1
            if success:
                self._publish_stats['successful_publishes'] += 1
                self.logger.debug(f"📤 Message published: {message.message_id} "
                                f"to {exchange}/{routing}")
            else:
                self._publish_stats['failed_publishes'] += 1
                self.logger.error(f"❌ Failed to publish message: {message.message_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error publishing message {message.message_id}: {e}")
            self._publish_stats['failed_publishes'] += 1
            return False
    
    def _get_exchange_for_message(self, message: BaseMessage) -> str:
        """Determine the appropriate exchange for a message type."""
        exchange_mapping = {
            MessageType.TRADING_SIGNAL: 'auj.platform',
            MessageType.AGENT_COORDINATION: 'auj.coordination',
            MessageType.SYSTEM_STATUS: 'auj.broadcast',
            MessageType.MARKET_DATA: 'auj.platform',
            MessageType.RISK_MANAGEMENT: 'auj.platform',
            MessageType.PERFORMANCE_UPDATE: 'auj.platform',
            MessageType.ERROR_NOTIFICATION: 'auj.platform'
        }
        
        return exchange_mapping.get(message.message_type, self.default_exchange)
    
    async def _publish_with_retry(
        self,
        message_body: str,
        exchange_name: str,
        routing_key: str,
        priority: int,
        expiration: Optional[int],
        headers: Dict[str, Any],
        retry_on_failure: bool
    ) -> bool:
        """Publish message with retry logic."""
        attempt = 0
        
        while attempt <= self.max_retry_attempts:
            try:
                await self.message_broker.publish_message(
                    message_body=message_body,
                    exchange_name=exchange_name,
                    routing_key=routing_key,
                    priority=priority,
                    expiration=expiration,
                    headers=headers
                )
                
                return True
                
            except Exception as e:
                attempt += 1
                self._publish_stats['retry_attempts'] += 1
                
                if attempt > self.max_retry_attempts or not retry_on_failure:
                    self.logger.error(f"❌ Failed to publish after {attempt} attempts: {e}")
                    return False
                
                # Exponential backoff
                delay = self.retry_delay * (2 ** (attempt - 1))
                self.logger.warning(f"⚠️ Publish attempt {attempt} failed, retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        return False
    
    async def publish_trading_signal(
        self,
        agent_name: str,
        symbol: str,
        signal_type: str,
        confidence: float,
        price: float,
        volume: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        indicators: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Publish a trading signal message."""
        message = TradingSignalMessage.create_signal(
            agent_name=agent_name,
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicators=indicators
        )
        
        # High confidence signals get high priority routing
        if confidence > 0.8:
            message.routing_key = f"trading.signal.{symbol}.high"
        
        return await self.publish_message(message)
    
    async def publish_agent_coordination(
        self,
        sender_agent: str,
        coordination_type: str,
        target_agents: List[str],
        coordination_data: Dict[str, Any],
        requires_response: bool = False
    ) -> bool:
        """Publish an agent coordination message."""
        message = AgentCoordinationMessage.create_coordination(
            sender_agent=sender_agent,
            coordination_type=coordination_type,
            target_agents=target_agents,
            coordination_data=coordination_data,
            requires_response=requires_response
        )
        
        return await self.publish_message(message)
    
    async def publish_system_status(
        self,
        component: str,
        status: str,
        details: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Publish a system status message."""
        message = SystemStatusMessage.create_status(
            component=component,
            status=status,
            details=details,
            metrics=metrics
        )
        
        return await self.publish_message(message)
    
    async def publish_market_data(
        self,
        symbol: str,
        data_type: str,
        data: Dict[str, Any],
        provider: str
    ) -> bool:
        """Publish market data update."""
        message = MarketDataMessage.create_market_data(
            symbol=symbol,
            data_type=data_type,
            data=data,
            provider=provider
        )
        
        return await self.publish_message(message)
    
    async def publish_risk_alert(
        self,
        risk_type: str,
        severity: str,
        affected_symbols: List[str],
        risk_data: Dict[str, Any],
        recommended_actions: List[str]
    ) -> bool:
        """Publish a risk management alert."""
        message = RiskManagementMessage.create_risk_alert(
            risk_type=risk_type,
            severity=severity,
            affected_symbols=affected_symbols,
            risk_data=risk_data,
            recommended_actions=recommended_actions
        )
        
        return await self.publish_message(message)
    
    async def publish_performance_update(
        self,
        component_name: str,
        component_type: str,
        performance_metrics: Dict[str, float],
        time_period: str = "1h"
    ) -> bool:
        """Publish performance update."""
        message = PerformanceUpdateMessage.create_performance_update(
            component_name=component_name,
            component_type=component_type,
            performance_metrics=performance_metrics,
            time_period=time_period
        )
        
        return await self.publish_message(message)
    
    async def publish_error_notification(
        self,
        component: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        recovery_actions: Optional[List[str]] = None
    ) -> bool:
        """Publish an error notification."""
        message = ErrorNotificationMessage.create_error_notification(
            component=component,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            recovery_actions=recovery_actions
        )
        
        return await self.publish_message(message)
    
    async def publish_batch(
        self,
        messages: List[BaseMessage],
        fail_on_first_error: bool = False
    ) -> Dict[str, bool]:
        """
        Publish multiple messages in batch.
        
        Args:
            messages: List of messages to publish
            fail_on_first_error: Stop on first failure
            
        Returns:
            Dict mapping message IDs to success status
        """
        results = {}
        
        for message in messages:
            success = await self.publish_message(message)
            results[message.message_id] = success
            
            if not success and fail_on_first_error:
                # Mark remaining messages as failed
                for remaining_message in messages[messages.index(message) + 1:]:
                    results[remaining_message.message_id] = False
                break
        
        successful = sum(1 for success in results.values() if success)
        total = len(messages)
        
        self.logger.info(f"📦 Batch publish complete: {successful}/{total} successful")
        
        return results
    
    async def publish_delayed(
        self,
        message: BaseMessage,
        delay_seconds: int
    ) -> bool:
        """
        Publish a message after a delay.
        
        Args:
            message: The message to publish
            delay_seconds: Delay in seconds
            
        Returns:
            bool: True if message was scheduled successfully
        """
        try:
            await asyncio.sleep(delay_seconds)
            return await self.publish_message(message)
            
        except Exception as e:
            self.logger.error(f"❌ Error in delayed publish: {e}")
            return False
    
    def schedule_recurring_message(
        self,
        message_factory: callable,
        interval_seconds: int,
        max_iterations: Optional[int] = None
    ) -> str:
        """
        Schedule a recurring message publication.
        
        Args:
            message_factory: Function that returns a BaseMessage
            interval_seconds: Interval between publications
            max_iterations: Maximum number of iterations (None for infinite)
            
        Returns:
            str: Task ID for cancellation
        """
        task_id = str(uuid.uuid4())
        
        async def recurring_task():
            iteration = 0
            while max_iterations is None or iteration < max_iterations:
                try:
                    message = message_factory()
                    await self.publish_message(message)
                    iteration += 1
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in recurring message task: {e}")
                    break
        
        # Start the task
        task = asyncio.create_task(recurring_task())
        self._scheduled_tasks = getattr(self, '_scheduled_tasks', {})
        self._scheduled_tasks[task_id] = task
        
        self.logger.info(f"⏰ Scheduled recurring message: {task_id}")
        return task_id
    
    def cancel_scheduled_task(self, task_id: str) -> bool:
        """Cancel a scheduled recurring task."""
        scheduled_tasks = getattr(self, '_scheduled_tasks', {})
        
        if task_id in scheduled_tasks:
            scheduled_tasks[task_id].cancel()
            del scheduled_tasks[task_id]
            self.logger.info(f"❌ Cancelled scheduled task: {task_id}")
            return True
        
        return False
    
    def get_publish_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        return {
            **self._publish_stats,
            'success_rate': (
                self._publish_stats['successful_publishes'] / 
                max(self._publish_stats['total_published'], 1)
            ),
            'scheduled_tasks': len(getattr(self, '_scheduled_tasks', {}))
        }
    
    async def health_check(self) -> bool:
        """Perform health check of the publisher."""
        try:
            # Test by publishing a system status message
            test_message = SystemStatusMessage.create_status(
                component='message_publisher',
                status='health_check',
                details={'timestamp': datetime.now().isoformat()}
            )
            
            return await self.publish_message(test_message)
            
        except Exception as e:
            self.logger.error(f"❌ Publisher health check failed: {e}")
            return False